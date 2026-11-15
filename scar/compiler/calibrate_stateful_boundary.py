"""Independent canonical guards/FFI and tagged scalar resource cost models.

The real shared runtime performs the Python checks and ctypes packing. Only
the final native table is a no-op with metadata guards. No target function is
imported or executed. Canonical source/receiver metadata and scalar chains are
declared models: they are never observations of the complete target call.
"""
import argparse
from copy import deepcopy
import ctypes
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

from scar.compiler.stateful_model import stateful_from_dict
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.stateful_cpu_choices import _domain
from scar.optimization.stateful_cpu_model import stateful_cpu_demand
from scar.runtime.native_stateful import NativeStateful, _Scalar, _Table, _resolve_providers
from scar.runtime.stateful_contract import (ASSUMPTIONS, FP_POLICY, MEMORY_SCOPE,
    canonical_sha256, validate_contract)
from scar.runtime.stateful_guard import guarded_stateful_function
from scar.trace.code_snapshot import code_key

_CODEGEN = Path(__file__).resolve().parents[1] / 'codegen'
_COMPONENTS = ('GUARDED_METHOD_AND_NATIVE_ARTIFACT_BOUNDARY', 'TAGGED_SCALAR_SETUP_STAGE')
_OPS = dict(add=1, sub=2, mul=3, div=4, neg=5)


def _need(condition, message):
    if not condition: raise ValueError(message)


def _json(value): return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
def _now(): return datetime.now(timezone.utc).isoformat()
def _receipt(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return dict(path=str(path), sha256=sha256(raw).hexdigest(), bytes=len(raw))


def _source(fn, path):
    return dict(path=str(path), sha256=_receipt(path)['sha256'],
        code_key=code_key(fn.__code__), qualname=fn.__qualname__)


def _ordinary_node(counter):
    counter[0] += 1
    return type('CanonicalReceiver'+str(counter[0]), (), {})()


def _canonical_binding(region, plan, directory):
    """Retain actual path sharing/names, while replacing all executable bodies."""
    import numpy as np
    counter = [0]
    parameters = list(region.parameter_names)
    _need(all(p.isidentifier() for p in parameters), 'ordinary parameters required')
    namespace = dict(__name__='scar_canonical_boundary_probe')
    lines = ['def _scar_boundary('+', '.join(parameters)+'):',
        "    raise RuntimeError('canonical probe fallback must never execute')"]
    for index, callee in enumerate(region.callees):
        lines += [f'def _scar_callee_{index}():',
            "    raise RuntimeError('canonical callee body must never execute')"]
    path = directory/'canonical_boundary.py'; path.write_text('\n'.join(lines)+'\n')
    exec(compile(path.read_text(), str(path), 'exec'), namespace)
    original = namespace['_scar_boundary']
    bound = {name:_ordinary_node(counter) for name in parameters}
    inputs = []; array_values = []
    for row in region.inputs:
        array = row.kind == 'array'
        shape = list(plan['input_shapes'][row.id]) if array else None
        value = np.full(shape, 1.25, dtype=np.float64) if array else 0.75
        written = row.id in plan['shadows']
        if array:
            value.flags.writeable = written
            array_values.append(value)
        parts = list(row.path)
        if len(parts) == 1: bound[parts[0]] = value
        else:
            current = bound[parts[0]]
            for name in parts[1:-1]:
                if name not in vars(current): setattr(current, name, _ordinary_node(counter))
                current = vars(current)[name]
            setattr(current, parts[-1], value)
        inputs.append(dict(id=row.id, path=parts, kind=row.kind, shape=shape,
            nbytes=None if not array else value.nbytes, written=written))

    def bind_lookup(parts, value, static=False):
        parts = list(parts)
        mapping = bound if parts[0] in bound else namespace
        if len(parts) == 1:
            _need(parts[0] not in mapping, 'lookup path conflicts with canonical input')
            mapping[parts[0]] = value
            return
        if parts[0] not in mapping: mapping[parts[0]] = types.ModuleType('CanonicalGlobals')
        current = mapping[parts[0]]
        for name in parts[1:-1]:
            _need(type(current) is types.ModuleType,
                'canonical intermediate callee/provider path requires module lookup')
            if name not in vars(current): setattr(current, name, types.ModuleType('CanonicalGlobals'))
            current = vars(current)[name]
        if type(current) is types.ModuleType: setattr(current, parts[-1], value)
        else:
            _need(parts[-1] not in vars(current), 'lookup shadows canonical input')
            setattr(type(current), parts[-1], staticmethod(value))

    callees = []
    for index, row in enumerate(region.callees):
        fn = namespace[f'_scar_callee_{index}']; bind_lookup(row.path, fn, static=True)
        callees.append(dict(path=list(row.path), kind=row.kind, source=_source(fn, path)))
    providers = []
    for row in region.providers:
        _need(row.module == 'numpy' and row.symbol == 'power', 'original NumPy power provider required')
        bind_lookup(row.lookup_path, np.power)
        providers.append(dict(id=row.id, module=row.module, symbol=row.symbol,
            lookup_path=list(row.lookup_path), semantics=row.semantics))
    structure = dict(parameters=parameters, inputs=inputs,
        callee_paths=[list(row.path) for row in region.callees],
        provider_paths=[list(row.lookup_path) for row in region.providers],
        scalar_types=['float']*sum(row.kind == 'scalar' for row in region.inputs),
        array_base_policy='INDEPENDENT_CONTIGUOUS_OWNING_ALLOCATIONS',
        read_only_arrays=[row['id'] for row in inputs if row['kind']=='array' and not row['written']],
        canonical_source=_receipt(path), source_body='RAISE_IF_FALLBACK_NO_TARGET_DAG')
    return original, tuple(bound[name] for name in parameters), inputs, callees, providers, structure


def _build_probe(context, parameters, memory, core, directory):
    directory.mkdir(parents=True)
    cpus = context['cpu_ids']; compiler = context['compiler']
    version = subprocess.run([*compiler['argv'], '--version'], capture_output=True,
        text=True, check=True).stdout.splitlines()[0]
    _need(version == compiler['identity'], 'C++ compiler differs from recorded context')
    _need(('-fopenmp' in compiler['options']) == (len(cpus)>1), 'OpenMP compiler options differ')
    runtime = (_CODEGEN/'loop_v1.cpp.in').read_text()
    affinity_start = runtime.index('class ThreadAffinity {')
    floating_start = runtime.index('class FloatingScope {')
    affinity = runtime[affinity_start:runtime.index('\n#endif', affinity_start)]
    floating = runtime[floating_start:runtime.index('\n};', floating_start)+3]
    stateful = (_CODEGEN/'stateful_v1.cpp.in').read_text()
    helpers = stateful[stateful.index('bool finite('):stateful.index('struct Scalar {')]
    scalar_helpers = stateful[stateful.index('struct Scalar {'):stateful.index('\nint32_t run(')]
    capacities = parameters['workspace_slot_capacities']
    replacements = dict(CPU_SLOTS=str(max(cpus)+1), CPU_IDS=', '.join(map(str, cpus)),
        AFFINITY_CLASS=affinity, FLOATING_CLASS=floating, ARRAY_HELPERS=helpers,
        SCALAR_HELPERS=scalar_helpers, ARRAY_COUNT=str(parameters['arrays']),
        SCALAR_COUNT=str(parameters['scalars']), PROVIDER_COUNT=str(parameters['providers']),
        ARRAY_GUARDS='\n'.join(f'    if (!valid_array(arrays[{i}],{s[0]},{s[1]})) return 2;'
            for i,s in enumerate(parameters['array_shapes'])),
        SLOT_COUNT=str(len(capacities)), SLOT_CAPACITIES=', '.join(map(str, capacities)),
        PAYLOAD_BYTES=str(memory['workspace_payload_bytes']),
        EXTERNAL_BYTES=str(memory['external_array_bytes']), TOTAL_BYTES=str(memory['total_region_array_payload_bytes']),
        BUDGET=str(memory['memory_budget_bytes']), REGION_SHA256=core['region_sha256'],
        PLAN_SHA256=core['plan_sha256'], CONTRACT_SHA256=canonical_sha256(core))
    source = (_CODEGEN/'stateful_boundary_probe_v1.cpp.in').read_text()
    for key, value in replacements.items(): source = source.replace('@'+key+'@', value)
    source_path = directory/'probe.cpp'; source_path.write_text(source)
    headers = {}
    for name in ('stateful_abi_v1.h', 'scar_abi_v1.h'):
        (directory/name).write_bytes((_CODEGEN/name).read_bytes()); headers[name] = _receipt(directory/name)
    library = directory/'stateful_boundary_probe.so'
    command = [*compiler['argv'], *compiler['options'], str(source_path), '-o', str(library)]
    result = subprocess.run(command, capture_output=True, text=True)
    (directory/'build.log').write_text(result.stdout+result.stderr)
    _need(result.returncode == 0, 'canonical boundary compilation failed: '+result.stderr[-2000:])
    implementation_paths = [Path(__file__), _CODEGEN/'stateful_boundary_probe_v1.cpp.in',
        _CODEGEN/'stateful_v1.cpp.in', _CODEGEN/'loop_v1.cpp.in',
        Path(__file__).resolve().parents[1]/'runtime'/'native_stateful.py',
        Path(__file__).resolve().parents[1]/'runtime'/'stateful_guard.py']
    metadata = dict(source=_receipt(source_path), headers=headers, library=_receipt(library),
        compiler=compiler, compiler_command=command,
        implementation_files=[_receipt(path) for path in implementation_paths],
        scalar_helpers_sha256=sha256(scalar_helpers.encode()).hexdigest(),
        scalar_helpers_copied_verbatim=True, target_dag_compiled=False)
    handle = ctypes.CDLL(str(library)); getter = handle.scar_get_stateful_table_v1
    getter.argtypes=[]; getter.restype=ctypes.POINTER(_Table); table=getter()
    _need(table and table.contents.abi_version==1 and table.contents.struct_size==ctypes.sizeof(_Table)
        and table.contents.run, 'no-op stateful ABI differs')
    _need((table.contents.array_count, table.contents.scalar_count, table.contents.provider_count)==
        (parameters['arrays'],parameters['scalars'],parameters['providers']), 'no-op table count differs')
    for key in ('region_sha256', 'plan_sha256', 'contract_sha256'):
        expected = canonical_sha256(core) if key=='contract_sha256' else core[key]
        _need(getattr(table.contents,key)==expected.encode(), 'no-op table identity differs')
    return handle, table, metadata


def _sample_guard(original, args, contract, library, table, directory, repetitions, warmup):
    providers, owners, power = _resolve_providers(contract)
    report = dict(contract['native']['ids'])
    artifact = NativeStateful(directory, contract, report, library, table, providers, owners, power)
    wrapper = guarded_stateful_function(original, artifact, owner=type(args[0]),
        exclusive_inputs=True, selection=contract['selection'])
    samples = []
    for rep in range(repetitions+warmup):
        start = time.perf_counter_ns(); result = wrapper(*args); elapsed = time.perf_counter_ns()-start
        _need(result is None, 'canonical native boundary returned an unexpected value')
        if rep>=warmup: samples.append(elapsed)
    _need(wrapper.scar_stats['native_calls']==repetitions+warmup and wrapper.scar_stats['fallback_calls']==0,
        'canonical boundary did not traverse the real shared native path')
    return dict(samples_ns=samples, runtime_stats=deepcopy(wrapper.scar_stats),
        scope='SHARED_GUARDED_METHOD_TO_NATIVE_NOOP_RETURN',
        includes=['LIVE_FUNCTION_RECEIVER_CALLEE_PROVIDER_INPUT_GUARDS',
            'SHAPE_LAYOUT_ALIAS_BUDGET_SCALAR_GUARDS', 'ACTUAL_CTYPES_DESCRIPTOR_AND_SCALAR_PACKING',
            'ACTUAL_PROVIDER_CAPSULE_REFERENCES_NONBLOCKING_LOCK_FFI', 'NATIVE_METADATA_VALIDATION'],
        excludes=['ONE_TIME_INSTALL_AND_FIRST_RECEIVER_SOURCE_HASH', 'INPUT_FINITE_SCAN',
            'WORKSPACE_ALLOCATION_AND_RELEASE', 'CALLER_WORKER_AFFINITY_FENV_TEAM',
            'TARGET_ARRAY_AND_SCALAR_DAG', 'COMMIT'],
        callable_dispatch='POSITIONAL_PARAMETERS_NO_DEFAULTS_CANONICAL_SOURCE',
        clock_overhead_retained=True, target_program_executed=False, whole_method_boundary_measured=False)


def _sample_scalar(library, parameters, cpus, repetitions, warmup):
    operations = parameters['operations']
    _need(operations and all(op in _OPS for op in operations), 'supported nonempty tagged scalar operations required')
    _need(parameters['workers']==len(cpus), 'tagged scalar workers differ')
    codes = (ctypes.c_int32*len(operations))(*[_OPS[op] for op in operations])
    operands = (_Scalar*(len(operations)+1))(*[_Scalar(1.25 if i==0 else 0.75,0)
        for i in range(len(operations)+1)])
    values = (ctypes.c_double*repetitions)(); placement=(ctypes.c_int32*(2+4*len(cpus)))()
    checksum=ctypes.c_uint64(); run=library.scar_tagged_scalar_probe_v1
    run.argtypes=[ctypes.POINTER(ctypes.c_int32),ctypes.c_uint64,ctypes.POINTER(_Scalar),
        ctypes.c_uint64,ctypes.c_uint64,ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_uint64)]
    run.restype=ctypes.c_int32
    status=run(codes,len(operations),operands,repetitions,warmup,values,placement,ctypes.byref(checksum))
    _need(status==0, 'tagged scalar canonical stage declined: '+str(status))
    workers=[dict(planned_cpu=cpu,entry_cpu=placement[2+i*4],bound_cpu=placement[3+i*4],
        restored_cpu=placement[4+i*4],original_affinity_restored=bool(placement[5+i*4])) for i,cpu in enumerate(cpus)]
    _need(all(row['planned_cpu']==row['bound_cpu'] and row['original_affinity_restored'] for row in workers),
        'tagged scalar worker placement/restoration differs')
    return dict(samples_ns=list(values), scope='MAX_WORKER_LOCAL_SINGLE_STAGE_WITH_OWN_IMPLICIT_BARRIER',
        persistent_team=True, parallel_cpu_ids=cpus, worker_placement=workers,
        caller_cpu_before=placement[0],caller_cpu_after=placement[1],checksum=checksum.value,
        includes_stage_completion_barrier=True, setup_and_team_cost_included=False,
        separate_barrier_cost_added=False, instrumentation_rendezvous_outside_timing=True,
        canonical_operands=dict(initial=1.25,right=0.75,integer_tags=False),
        operation_topology='CANONICAL_LEFT_DEPENDENT_CHAIN_FROM_OPERATION_LIST',
        generic_operation_dispatch_overhead_retained=True,
        target_program_executed=False, whole_method_boundary_measured=False)


def calibrate_stateful_boundary(domain,candidate_id,out,*,repetitions=9,warmup=2):
    """Write two resource signatures for one mask, with retained model caveats."""
    _need(sys.gettrace() is None and sys.getprofile() is None, 'calibration requires no Python trace/profile hook')
    _need(type(repetitions) is int and 3<=repetitions<=10000 and type(warmup) is int and 1<=warmup<=10000,
        'at least three samples and one warmup required')
    domain=_domain(domain); out=Path(out).resolve()
    _need(not out.exists(), 'fresh boundary calibration output required')
    candidate=next((r for r in domain['candidates'] if r['id']==candidate_id),None)
    _need(candidate is not None and candidate['legal_under_guards'], 'legal CPU candidate required')
    actual=sorted(os.sched_getaffinity(0));cpus=candidate['binding']['cpu_ids']
    _need(actual==domain['live_constraints']['process_affinity_cpu_ids'] and set(cpus)<=set(actual),
        'caller affinity must match exact calibration context')
    thread_env={key:os.environ.get(key) for key in domain['live_constraints']['native_thread_environment']}
    _need(thread_env==domain['live_constraints']['native_thread_environment'], 'native thread environment differs')
    demand=stateful_cpu_demand(domain,candidate_id)
    terms=[];seen=set()
    for term in demand['terms']:
        if term['component'] in _COMPONENTS and term['key'] not in seen:
            seen.add(term['key']);terms.append(term)
    guard=next(t for t in terms if t['component']==_COMPONENTS[0])
    region=stateful_from_dict(read_artifact(domain['region']));plan=read_artifact(domain['base_stage_plan'])
    backend=domain['backend_capabilities']['serial' if len(cpus)==1 else 'parallel']
    context=dict(cpu_ids=cpus,caller_affinity_cpu_ids=actual,compiler=backend['compiler'],
        provider_runtime=backend['provider_runtime'],native_thread_environment=thread_env)
    memory={key:plan[key] for key in ('external_array_bytes','workspace_payload_bytes',
        'total_region_array_payload_bytes','memory_budget_bytes')}
    core=dict(region_sha256=canonical_sha256(guard['parameters']),
        plan_sha256=canonical_sha256(dict(parameters=guard['parameters'],cpu_ids=cpus,memory=memory)),
        provider_runtime=context['provider_runtime'])
    observed_from=_now(); started=time.perf_counter();load_before=list(os.getloadavg())
    out.mkdir(parents=True)
    original,args,inputs,callees,providers,structure=_canonical_binding(region,plan,out)
    library,table,probe=_build_probe(context,guard['parameters'],memory,core,out/'native')
    ids=dict(build_id=probe['library']['sha256'],region_sha256=core['region_sha256'],
        plan_sha256=core['plan_sha256'],contract_sha256=canonical_sha256(core))
    selection=dict(decision='TRANSFORM_UNDER_CONTRACT',selected=True,artifact=ids,
        input_shapes={row['id']:row['shape'] for row in inputs if row['kind']=='array'},
        external_assumptions=sorted(ASSUMPTIONS))
    contract=validate_contract(dict(schema='scar.stateful-runtime-contract',schema_version=1,
        profile='cpu-f64-rank2-stateful-v1',native=dict(library=Path(probe['library']['path']).name,
            library_sha256=probe['library']['sha256'],stateful_abi_version=1,ids=ids,contract=core),
        source=_source(original,out/'canonical_boundary.py'),parameters=list(region.parameter_names),
        inputs=inputs,callee_guards=callees,providers=providers,
        cpu=dict(ids=cpus,threads=len(cpus),architecture='x86_64',os='linux',pointer_bits=64,
            byteorder='little',required_isa='BASELINE_X86_64'),
        memory=dict(**memory,scope=MEMORY_SCOPE,policy='UNBOUND'),provider_runtime=context['provider_runtime'],
        guards=['CANONICAL_STRUCTURAL_GUARD_AND_PACKING_MODEL'],external_assumptions=sorted(ASSUMPTIONS),
        floating_environment_policy=FP_POLICY,selection=selection))
    (out/'canonical-contract.json').write_bytes(_json(contract))
    components=[]
    try:
        for term in terms:
            measurement=_sample_guard(original,args,contract,library,table,out,repetitions,warmup) if term['component']==_COMPONENTS[0] else \
                _sample_scalar(library,term['parameters'],cpus,repetitions,warmup)
            samples=measurement['samples_ns']
            _need(len(samples)==repetitions and all(math.isfinite(v) and v>=0 for v in samples), 'invalid component samples')
            components.append(dict(component=term['component'],parameters=term['parameters'],
                basis='DECLARED_COMPONENT_MODEL',median_ns=statistics.median(samples),min_ns=min(samples),
                max_ns=max(samples),stdev_ns=statistics.stdev(samples),samples_ns=samples,
                model_basis='MEASURED_CANONICAL_COMPONENT_USING_SHARED_IMPLEMENTATION',measurement=measurement,
                assumptions=['STRUCTURAL_SIGNATURE_OMITS_ACTUAL_SOURCE_CODE_AND_FUNCTION_DEFAULTS',
                    'CANONICAL_FLOAT_SCALARS_AND_OWNING_ARRAYS_ARE_NOT_TARGET_VALUE_DISTRIBUTION',
                    'PATH_NAMES_AND_SHARING_RETAINED_IN_RECEIPT_NOT_RESOURCE_KEY',
                    'SCALAR_OPERATION_LIST_DOES_NOT_SPECIFY_TARGET_TOPOLOGY_OR_INTEGER_TAGS',
                    'MODEL_COMPOSITION_IS_NOT_AN_ADDITIVITY_THEOREM',
                    'SOURCE_AND_REGION_IDENTITIES_DO_NOT_ENTER_PRIMITIVE_RESOURCE_KEY']))
    finally:
        observed_after=sorted(os.sched_getaffinity(0))
        if observed_after!=actual: os.sched_setaffinity(0,actual)
    _need(observed_after==actual, 'probe failed to restore exact caller affinity')
    _need({key:os.environ.get(key) for key in thread_env}==thread_env, 'probe changed thread environment')
    document=dict(schema='scar.stateful-resource-calibration',schema_version=1,
        resource_context=demand['resource_context'],probe_context=context,components=components,
        unsupported_components=[],extra_observations=[],probe=probe,canonical_structure=structure,
        canonical_runtime_contract=_receipt(out/'canonical-contract.json'),
        source_receipts=[_receipt(row.path) for row in (region.source,*(r.source for r in region.callees))],
        source_scope='SOURCE_RECEIPTS_ARE_DERIVATION_ONLY_NOT_PRIMITIVE_APPLICABILITY',
        repetitions=repetitions,warmup=warmup,observed_from=observed_from,observed_until=_now(),
        seconds=time.perf_counter()-started,host_load=dict(before=load_before,after=list(os.getloadavg()),isolated=False),
        observed_caller_affinity_before=actual,observed_caller_affinity_after=observed_after,
        observed_native_thread_environment=thread_env,caller_affinity_restored=True,
        caller_placement='ORIGINAL_CALLER_MASK_UNBOUND_SEPARATE_FROM_PINNED_SCALAR_TEAM',
        numa_page_residency_proven=False,no_scope_median_subtraction=True,
        implicit_stage_barrier_cost_already_included=True,whole_method_boundary_measured=False,
        target_workload_executed=False,target_variants_compiled=False,target_variants_ranked=False,
        physical_optimality_proven=False)
    path=out/'observations.json';path.write_bytes(_json(document))
    result=dict(schema='scar.stateful-resource-calibration-result',schema_version=1,
        observations=artifact_reference(path,document['schema']),candidate_id=candidate_id,
        measured_component_signatures=len(components),components=[dict(component=r['component'],
            median_ns=r['median_ns'],basis=r['basis']) for r in components],
        target_workload_executed=False,target_variants_ranked=False)
    (out/'result.json').write_bytes(_json(result))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain',required=True);parser.add_argument('--candidate',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--repetitions',type=int,default=9)
    parser.add_argument('--warmup',type=int,default=2);args=parser.parse_args(argv)
    print(json.dumps(calibrate_stateful_boundary(args.domain,args.candidate,args.out,
        repetitions=args.repetitions,warmup=args.warmup),sort_keys=True))
    return 0


if __name__=='__main__': raise SystemExit(main())
