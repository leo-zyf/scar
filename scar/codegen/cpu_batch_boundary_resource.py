"""Compile canonical auxiliary probes once per mode, sample each exact mask.

Only CPU selection and ABI identity slots become mutable, through setters
outside every clock. Original resource/scalar stages and shared Python guard
and FFI implementation are reused. A sample is never transferred to a mask.
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
import resource
import shutil
import statistics
import subprocess
import sys
import threading
import time
from types import SimpleNamespace as NS

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
SCHEMA = 'scar.cpu-batch-boundary-resource-pack'
_SAMPLED_PID = None
_LIMIT = 250*2**20
_BOUNDARY = ('GUARDED_METHOD_AND_NATIVE_ARTIFACT_BOUNDARY', 'TAGGED_SCALAR_SETUP_STAGE')


def _need(value, detail):
    if not value: raise ValueError(detail)


def _json(value): return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
def _semantic(value): return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def _receipt(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return dict(path=str(path), sha256=sha256(raw).hexdigest(), bytes=len(raw))
def _reference(path, schema):
    row = _receipt(path)
    return dict(uri=row['path'], sha256=row['sha256'], size_bytes=row['bytes'], schema=schema)
def _check(row):
    _need(_receipt(row['path']) == row, 'auxiliary producer/library receipt changed: '+row['path'])
def _save(path, value):
    with Path(path).open('xb') as stream: stream.write(_json(value))
    return _reference(path, value['schema'])
def _now(): return datetime.now(timezone.utc).isoformat()
def _rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024


def _implementation():
    names = ('codegen/cpu_batch_boundary_resource.py', 'codegen/cpu_resource_probe.py',
        'codegen/cpu_resource_probe_v1.cpp.in', 'codegen/stateful_boundary_probe_v1.cpp.in',
        'codegen/stateful_v1.cpp.in', 'codegen/loop_v1.cpp.in', 'codegen/stateful_abi_v1.h',
        'codegen/scar_abi_v1.h', 'codegen/stateful.py', 'compiler/calibrate_stateful_boundary.py',
        'runtime/native_stateful.py', 'runtime/stateful_guard.py', 'runtime/stateful_contract.py',
        'runtime/native_loop.py', 'trace/code_snapshot.py')
    return [_receipt(_ROOT/name) for name in names]


def _loaded_openmp(handle):
    class DlInfo(ctypes.Structure):
        _fields_ = [('name', ctypes.c_char_p), ('base', ctypes.c_void_p),
            ('symbol', ctypes.c_char_p), ('address', ctypes.c_void_p)]
    loader = ctypes.CDLL(None); function = loader.dladdr
    function.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]; function.restype = ctypes.c_int
    info = DlInfo(); symbol = ctypes.cast(handle.omp_get_num_threads, ctypes.c_void_p)
    _need(function(symbol, ctypes.byref(info)) != 0 and info.name, 'loaded OpenMP symbol cannot be identified')
    return _receipt(Path(info.name.decode()).resolve())


def _replace_once(source, old, new):
    _need(source.count(old) == 1, 'canonical probe changed at a declared substitution')
    return source.replace(old, new)


def _cpu_source(source, cpus, maximum_slots):
    source = _replace_once(source, 'constexpr size_t minimum_cpu_slots = @CPU_SLOTS@;',
        'size_t minimum_cpu_slots = @CPU_SLOTS@;')
    source = source.replace('@CPU_SLOTS@', str(maximum_slots))
    return _replace_once(source, 'constexpr int planned_cpu_ids[] = {@CPU_IDS@};',
        'std::vector<int> planned_cpu_ids = {'+', '.join(map(str, cpus))+'};')


def _setter(parallel, *, boundary):
    name = 'scar_batch_set_context_v1' if boundary else 'scar_batch_set_cpu_ids_v1'
    extra_args = ', const char *region, const char *plan, const char *contract' if boundary else ''
    extra = '''
    const char *keys[]={region,plan,contract};
    for(const char *key:keys) {
        if(!key || std::strlen(key)!=64)return 1;
        for(int i=0;i<64;++i)if(!((key[i]>='0'&&key[i]<='9')||(key[i]>='a'&&key[i]<='f')))return 1;
    }
    std::memcpy(batch_region,region,65);std::memcpy(batch_plan,plan,65);std::memcpy(batch_contract,contract,65);
''' if boundary else ''
    return f'''
extern "C" SCAR_EXPORT_V1 int32_t {name}(const int32_t *ids,uint64_t count{extra_args}) {{
    if(!ids || count<{'2' if parallel else '1'} || count>{maximum_count(parallel)})return 1;
    {'if(count!=1)return 1;' if not parallel else ''}
    ThreadAffinity caller;
    for(uint64_t i=0;i<count;++i) {{
        if(!caller.contains(ids[i]))return 4;
        for(uint64_t j=0;j<i;++j)if(ids[i]==ids[j])return 1;
    }}
    {extra}
    size_t mask_slots=1;
    for(uint64_t i=0;i<count;++i)mask_slots=std::max(mask_slots,size_t(ids[i])+1);
    minimum_cpu_slots=mask_slots;
    planned_cpu_ids.assign(ids,ids+count);
    return 0;
}}
'''


def maximum_count(parallel): return 1048576 if parallel else 1


def _sources(cpus, maximum_slots, parallel, parameters, memory):
    runtime = (_HERE/'loop_v1.cpp.in').read_text()
    start = runtime.index('class ThreadAffinity {')
    affinity = runtime[start:runtime.index('\n#endif', start)]
    start = runtime.index('class FloatingScope {')
    floating = runtime[start:runtime.index('\n};', start)+3]
    substitutions = dict(AFFINITY_CLASS=affinity, FLOATING_CLASS=floating)
    resource_source = _cpu_source((_HERE/'cpu_resource_probe_v1.cpp.in').read_text(), cpus, maximum_slots)
    for key, value in substitutions.items(): resource_source = resource_source.replace('@'+key+'@', value)
    resource_source = resource_source.replace('@MAX_PAYLOAD@', str(64*2**20))
    resource_source = _replace_once(resource_source, 'int(sizeof(planned_cpu_ids)/sizeof(int))', 'int(planned_cpu_ids.size())')
    resource_source += _setter(parallel, boundary=False)
    stateful = (_HERE/'stateful_v1.cpp.in').read_text()
    helpers = stateful[stateful.index('bool finite('):stateful.index('struct Scalar {')]
    scalar_helpers = stateful[stateful.index('struct Scalar {'):stateful.index('\nint32_t run(')]
    boundary = _cpu_source((_HERE/'stateful_boundary_probe_v1.cpp.in').read_text(), cpus, maximum_slots)
    boundary = boundary.replace('#include <cstdint>', '#include <cstdint>\n#include <cstring>')
    capacities = parameters['workspace_slot_capacities']
    replacements = dict(**substitutions, ARRAY_HELPERS=helpers, SCALAR_HELPERS=scalar_helpers,
        ARRAY_COUNT=str(parameters['arrays']), SCALAR_COUNT=str(parameters['scalars']), PROVIDER_COUNT=str(parameters['providers']),
        ARRAY_GUARDS='\n'.join(f'    if (!valid_array(arrays[{i}],{s[0]},{s[1]})) return 2;' for i,s in enumerate(parameters['array_shapes'])),
        SLOT_COUNT=str(len(capacities)), SLOT_CAPACITIES=', '.join(map(str, capacities)),
        PAYLOAD_BYTES=str(memory['workspace_payload_bytes']), EXTERNAL_BYTES=str(memory['external_array_bytes']),
        TOTAL_BYTES=str(memory['total_region_array_payload_bytes']), BUDGET=str(memory['memory_budget_bytes']),
        REGION_SHA256='0'*64, PLAN_SHA256='0'*64, CONTRACT_SHA256='0'*64)
    for key, value in replacements.items(): boundary = boundary.replace('@'+key+'@', value)
    boundary = _replace_once(boundary, 'const scar_stateful_table_v1 table={',
        'char batch_region[65]={},batch_plan[65]={},batch_contract[65]={};\nscar_stateful_table_v1 table={')
    boundary = _replace_once(boundary, '"'+'0'*64+'","'+'0'*64+'","'+'0'*64+'"};',
        'batch_region,batch_plan,batch_contract};')
    boundary = _replace_once(boundary, 'constexpr int workers=sizeof(planned_cpu_ids)/sizeof(int);',
        'const int workers=static_cast<int>(planned_cpu_ids.size());')
    boundary += _setter(parallel, boundary=True)
    _need('@' not in resource_source and '@' not in boundary, 'unresolved canonical probe substitution')
    return resource_source, boundary, sha256(scalar_helpers.encode()).hexdigest()


def _specialized_demand(demand, row):
    result = deepcopy(demand); workers = row['worker_count']
    context = result['context']; context['candidate_id'] = row['id']
    context['cpu_binding'] = {key:value for key,value in row['binding'].items() if key != 'evidence'}
    result['resource_context']['worker_cpu_ids'] = row['binding']['cpu_ids']
    result['resource_context']['executor_id'] = row['binding']['executor_id']
    for term in result['terms']:
        if 'workers' in term['parameters'] and term['component'] != 'ROW_COPY_SERIAL':
            term['parameters']['workers'] = workers
        term['key'] = _semantic(dict(component=term['component'], parameters=term['parameters']))
    return result


def build_aux_pack(domain, mode, out):
    """One full domain admission and two canonical compilations per mode."""
    from scar.optimization.stateful_cpu_model import _domain
    from scar.optimization.stateful_cpu_model import stateful_cpu_demand
    from scar.ir.execution_recipe import read_artifact
    from scar.compiler.stateful_model import stateful_from_dict
    _need(mode in ('serial', 'parallel'), 'serial or parallel mode required')
    original_path = Path(domain).resolve() if isinstance(domain, (str, Path)) else None
    document = _domain(domain); out = Path(out).resolve()
    _need(not out.exists(), 'fresh auxiliary pack required')
    rows = [row for row in document['candidates'] if row['legal_under_guards'] and (row['worker_count'] > 1) == (mode == 'parallel')]
    _need(rows, 'at least one admitted CPU context required for this mode')
    exemplar = rows[0]; demand = stateful_cpu_demand(document, exemplar['id'])
    region = stateful_from_dict(read_artifact(document['region'])); plan = read_artifact(document['base_stage_plan'])
    backend = document['backend_capabilities'][mode]; compiler = backend['compiler']; cpus = exemplar['binding']['cpu_ids']
    affinity = sorted(os.sched_getaffinity(0)); _need(affinity == document['live_constraints']['process_affinity_cpu_ids'], 'caller mask differs')
    _need(('-fopenmp' in compiler['options']) == (mode == 'parallel'), 'mode compiler flags differ')
    identity = subprocess.run([*compiler['argv'], '--version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    _need(identity == compiler['identity'], 'compiler identity differs')
    guard = next(term for term in demand['terms'] if term['component'] == _BOUNDARY[0])
    memory = {key:plan[key] for key in ('external_array_bytes','workspace_payload_bytes','total_region_array_payload_bytes','memory_budget_bytes')}
    _need(memory['total_region_array_payload_bytes'] <= 64*2**20, 'canonical boundary payload exceeds bounded pack profile')
    out.mkdir(parents=True)
    if original_path is None:
        original_path = out/'domain.json'; original_path.write_bytes(_json(document))
    domain_ref = _reference(original_path, document['schema'])
    source, boundary, scalar_sha = _sources(cpus, max(affinity)+1, mode == 'parallel', guard['parameters'], memory)
    libraries = {}; sources = {}; commands = []
    for name in ('stateful_abi_v1.h','scar_abi_v1.h'): (out/name).write_bytes((_HERE/name).read_bytes())
    for name, text in (('resource', source), ('boundary', boundary)):
        path = out/(name+'.cpp'); path.write_text(text); library = out/(name+'.so')
        command = [*compiler['argv'], *compiler['options'], str(path), '-o', str(library)]
        result = subprocess.run(command, capture_output=True, text=True); (out/(name+'-build.log')).write_text(result.stdout+result.stderr)
        _need(result.returncode == 0, 'canonical '+name+' pack compilation failed: '+result.stderr[-1800:])
        sources[name] = _receipt(path); libraries[name] = _receipt(library); commands.append(command)
    handles = [ctypes.CDLL(row['path']) for row in libraries.values()]
    core = dict(region_sha256=_semantic(guard['parameters']),
        plan_sha256=_semantic(dict(parameters=guard['parameters'],cpu_ids=cpus,memory=memory)),
        provider_runtime=backend['provider_runtime'])
    _set(handles[0], cpus); _set(handles[1], cpus, core)
    openmp = None
    if mode == 'parallel':
        path = subprocess.run([*compiler['argv'], '-print-file-name=libgomp.so.1'], capture_output=True, text=True, check=True).stdout.strip()
        selected = _receipt(Path(path).resolve()); actual = [_loaded_openmp(h) for h in handles]
        _need(actual[0] == actual[1], 'auxiliary libraries resolved different OpenMP runtimes')
        openmp = dict(compiler_selected=selected, actual_loader=actual[0],
            identity_rule='DLADDR_RESOLVED_OMP_SYMBOL_SEPARATE_FROM_COMPILER_PRINT_FILE_NAME')
    candidates = {}
    for row in rows:
        expected = _specialized_demand(demand, row)
        candidates[row['id']] = dict(binding=row['binding'], worker_count=row['worker_count'],
            resource_context_sha256=_semantic(expected['resource_context']), cost_context_sha256=_semantic(expected['context']),
            demand_sha256=_semantic(expected))
    canonical = dict(parameter_names=list(region.parameter_names),
        inputs=[dict(id=r.id,path=list(r.path),kind=r.kind) for r in region.inputs],
        callees=[dict(path=list(r.path),kind=r.kind) for r in region.callees],
        providers=[dict(id=r.id,module=r.module,symbol=r.symbol,lookup_path=list(r.lookup_path),semantics=r.semantics) for r in region.providers])
    metadata = dict(schema=SCHEMA, schema_version=1, mode=mode, domain=domain_ref, domain_id=document['domain_id'],
        source_region=document['region'], base_stage_plan=document['base_stage_plan'], candidates=candidates,
        compiler=compiler, compiler_executable=_receipt(Path(shutil.which(compiler['argv'][0])).resolve()),
        compiler_commands=commands, openmp=openmp, libraries=libraries, sources=sources,
        headers=[_receipt(out/name) for name in ('stateful_abi_v1.h','scar_abi_v1.h')],
        implementation_receipts=_implementation(), provider_runtime=backend['provider_runtime'],
        caller_affinity_cpu_ids=affinity, native_thread_environment=document['live_constraints']['native_thread_environment'],
        guard_parameters=guard['parameters'], canonical_region=canonical,
        canonical_plan={key:plan[key] for key in ('input_shapes','shadows')}, memory=memory,
        scalar_helpers_sha256=scalar_sha, scalar_helpers_copied_verbatim=True,
        context_setters_loaded_and_checked_without_sampling=True,
        setters_outside_all_timed_intervals=True, original_stage_bodies_reused=True,
        allowed_substitutions=['PLANNED_CPU_VECTOR','DYNAMIC_WORKER_COUNT','MUTABLE_ABI_IDENTITY_BYTES',
            'MINIMUM_CPU_SLOTS_SET_TO_MAX_SELECTED_CPU_PLUS_ONE'],
        affinity_helper_body_copied_verbatim=True,
        dynamic_mask_reads_are_retained_proxy_overhead=True,
        sampling_policy=dict(repetitions=3,warmup=1,one_candidate_per_fresh_process=True),
        max_resource_payload_bytes=64*2**20, max_worker_rss_bytes=_LIMIT,
        peak_builder_rss_bytes=_rss(),
        target_program_executed=False,target_dag_compiled=False,target_variants_ranked=False)
    _need(_rss() < _LIMIT, 'pack builder exceeded the explicit RSS limit')
    ref = _save(out/'pack.json', metadata)
    return dict(pack=ref, directory=str(out), mode=mode)


def _load_pack(pack):
    from scar.ir.execution_recipe import read_artifact
    if isinstance(pack, dict): value = read_artifact(pack)
    else:
        path = Path(pack).resolve(); path = path/'pack.json' if path.is_dir() else path
        value = read_artifact(_reference(path, SCHEMA))
    _need((value.get('schema'),value.get('schema_version')) == (SCHEMA,1), 'admitted auxiliary pack required')
    for row in (*value['implementation_receipts'], *value['sources'].values(), *value['libraries'].values(),
            *value['headers'], value['compiler_executable']): _check(row)
    return value


def _context(pack, context):
    from scar.ir.execution_recipe import read_artifact
    _need((context.get('schema'),context.get('schema_version')) == ('scar.cpu-batch-calibration-context',1), 'frozen batch context required')
    domain = read_artifact(context['domain']); _need(domain['domain_id'] == pack['domain_id'], 'context domain differs')
    _need(_semantic(domain) == _semantic(read_artifact(pack['domain'])), 'domain parent bytes/semantic scope differ')
    for key in ('source_region','base_stage_plan'):
        _need(context[key] == pack[key], 'original '+key+' parent differs'); read_artifact(context[key],json_document=False)
    expected = pack['candidates'].get(context['candidate_id']); _need(expected is not None, 'context mask is outside pack mode')
    _need(context['binding'] == expected['binding'], 'candidate CPU binding differs')
    for key in ('resource_context','cost_context','demand'):
        _need(_semantic(context[key]) == expected[key+'_sha256'], 'frozen candidate '+key+' differs')
    _need(context['demand']['context'] == context['cost_context']
        and context['demand']['resource_context'] == context['resource_context'], 'demand scope is inconsistent')
    return expected


def _set(handle, cpus, core=None):
    array = (ctypes.c_int32*len(cpus))(*cpus)
    function = handle.scar_batch_set_cpu_ids_v1 if core is None else handle.scar_batch_set_context_v1
    function.argtypes = [ctypes.POINTER(ctypes.c_int32),ctypes.c_uint64] + ([] if core is None else [ctypes.c_char_p]*3)
    function.restype = ctypes.c_int32
    keys = [] if core is None else [core['region_sha256'].encode(),core['plan_sha256'].encode(),_semantic(core).encode()]
    _need(function(array,len(cpus),*keys) == 0, 'native context setter declined CPU mask or table identities')


def _row(component, parameters, measurement, basis):
    samples = measurement['samples_ns']
    _need(len(samples) == 3 and all(type(x) in (int,float) and math.isfinite(x) and x >= 0 for x in samples), 'invalid primitive observations')
    return dict(component=component,parameters=deepcopy(parameters),basis=basis,samples_ns=samples,
        median_ns=statistics.median(samples),min_ns=min(samples),max_ns=max(samples),stdev_ns=statistics.stdev(samples),measurement=measurement)


def sample_aux_pack(pack, candidate_context, out, repetitions=3, warmup=1):
    """One process/context, real same-mask samples; no build or domain replay."""
    global _SAMPLED_PID
    from scar.ir.execution_recipe import read_artifact
    from scar.runtime.compiled_deployment import thread_environment
    from scar.codegen.cpu_resource_probe import ResourceProbe,_Config
    from scar.compiler.calibrate_stateful_boundary import _canonical_binding,_sample_guard,_sample_scalar,_source
    from scar.runtime.native_stateful import _Table
    from scar.runtime.stateful_contract import ASSUMPTIONS,FP_POLICY,MEMORY_SCOPE,validate_contract
    _need(_SAMPLED_PID != os.getpid(), 'one candidate per fresh sample process required')
    _SAMPLED_PID = os.getpid()
    _need(repetitions == 3 and warmup == 1, 'sampling declaration is fixed at three repetitions/one warmup')
    _need(sys.gettrace() is None and sys.getprofile() is None and threading.active_count() == 1,
        'ordinary single Python sampling thread without trace/profile required')
    document = _load_pack(pack)
    context = read_artifact(candidate_context) if isinstance(candidate_context,dict) and 'uri' in candidate_context else (
        json.loads(Path(candidate_context).read_bytes()) if isinstance(candidate_context,(str,Path)) else deepcopy(candidate_context))
    expected = _context(document, context); cpus = context['binding']['cpu_ids']; out = Path(out).resolve()
    _need(not out.exists(), 'fresh per-mask auxiliary output required')
    actual = sorted(os.sched_getaffinity(0)); environment = thread_environment()
    _need(actual == document['caller_affinity_cpu_ids'] and set(cpus) <= set(actual), 'active original caller affinity differs')
    _need(environment == document['native_thread_environment'], 'native thread environment differs')
    _need(_rss() < _LIMIT, 'sample process is already over the explicit RSS limit')
    observed_from = _now(); started = time.perf_counter(); load_before = list(os.getloadavg()); out.mkdir(parents=True)
    resource_library = ctypes.CDLL(document['libraries']['resource']['path'])
    boundary_library = ctypes.CDLL(document['libraries']['boundary']['path'])
    openmp = None
    if document['openmp'] is not None:
        for receipt in document['openmp'].values():
            if type(receipt) is dict: _check(receipt)
        observed = [_loaded_openmp(handle) for handle in (resource_library,boundary_library)]
        _need(observed[0] == observed[1] == document['openmp']['actual_loader'], 'actual process OpenMP loader identity changed')
        openmp = dict(compiler_selected=document['openmp']['compiler_selected'],actual_loader=observed[0])
    memory = document['memory']; guard = next(t for t in context['demand']['terms'] if t['component'] == _BOUNDARY[0])
    _need(guard['parameters'] == document['guard_parameters'], 'canonical boundary structural signature differs')
    core = dict(region_sha256=_semantic(guard['parameters']),plan_sha256=_semantic(dict(parameters=guard['parameters'],cpu_ids=cpus,memory=memory)),
        provider_runtime=document['provider_runtime'])
    _set(resource_library,cpus); _set(boundary_library,cpus,core)
    getter = boundary_library.scar_get_stateful_table_v1; getter.argtypes=[]; getter.restype=ctypes.POINTER(_Table); table=getter()
    _need(table and table.contents.abi_version == 1 and table.contents.struct_size == ctypes.sizeof(_Table), 'boundary table ABI differs')
    _need((table.contents.array_count,table.contents.scalar_count,table.contents.provider_count) ==
        (guard['parameters']['arrays'],guard['parameters']['scalars'],guard['parameters']['providers']), 'boundary table counts differ')
    for key in ('region_sha256','plan_sha256','contract_sha256'):
        _need(getattr(table.contents,key) == (_semantic(core) if key == 'contract_sha256' else core[key]).encode(), 'set table identity differs')
    run = resource_library.scar_resource_probe_v1
    run.argtypes=[ctypes.POINTER(_Config),ctypes.c_void_p,ctypes.c_uint64,ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_int32),ctypes.POINTER(ctypes.c_uint64)]
    run.restype=ctypes.c_int32
    probe_context=dict(cpu_ids=cpus,caller_affinity_cpu_ids=actual,compiler=document['compiler'],
        provider_runtime=document['provider_runtime'],native_thread_environment=environment)
    probe=ResourceProbe(Path(document['libraries']['resource']['path']).parent,dict(context=probe_context,
        shared_pack=pack if isinstance(pack,dict) else str(pack),dynamic_setter_outside_timing=True),resource_library,run)
    terms=[];seen=set()
    for term in context['demand']['terms']:
        if term['key'] not in seen: seen.add(term['key']);terms.append(term)
    resource_components=[];boundary_components=[];unsupported=[];extras=[]
    canonical=NS(**{key:([NS(**row) for row in value] if key in ('inputs','callees','providers') else value)
        for key,value in document['canonical_region'].items()})
    original,args,inputs,callees,providers,structure=_canonical_binding(canonical,document['canonical_plan'],out)
    ids=dict(build_id=document['libraries']['boundary']['sha256'],region_sha256=core['region_sha256'],
        plan_sha256=core['plan_sha256'],contract_sha256=_semantic(core))
    selection=dict(decision='TRANSFORM_UNDER_CONTRACT',selected=True,artifact=ids,
        input_shapes={row['id']:row['shape'] for row in inputs if row['kind']=='array'},external_assumptions=sorted(ASSUMPTIONS))
    contract=validate_contract(dict(schema='scar.stateful-runtime-contract',schema_version=1,profile='cpu-f64-rank2-stateful-v1',
        native=dict(library=Path(document['libraries']['boundary']['path']).name,library_sha256=document['libraries']['boundary']['sha256'],
            stateful_abi_version=1,ids=ids,contract=core),source=_source(original,out/'canonical_boundary.py'),
        parameters=document['canonical_region']['parameter_names'],inputs=inputs,callee_guards=callees,providers=providers,
        cpu=dict(ids=cpus,threads=len(cpus),architecture='x86_64',os='linux',pointer_bits=64,byteorder='little',required_isa='BASELINE_X86_64'),
        memory=dict(**memory,scope=MEMORY_SCOPE,policy='UNBOUND'),provider_runtime=document['provider_runtime'],
        guards=['CANONICAL_STRUCTURAL_GUARD_AND_PACKING_MODEL'],external_assumptions=sorted(ASSUMPTIONS),floating_environment_policy=FP_POLICY,selection=selection))
    _save(out/'canonical-contract.json',contract)
    try:
        for term in terms:
            if term['component'] in _BOUNDARY:
                measurement = _sample_guard(original,args,contract,boundary_library,table,out,3,1) if term['component'] == _BOUNDARY[0] else _sample_scalar(boundary_library,term['parameters'],cpus,3,1)
                row=_row(term['component'],term['parameters'],measurement,'DECLARED_COMPONENT_MODEL')
                row.update(model_basis='MEASURED_CANONICAL_COMPONENT_USING_SHARED_IMPLEMENTATION',
                    assumptions=['CANONICAL_SOURCE_AND_FINITE_VALUES_NOT_TARGET_SOURCE_OR_VALUES','MODEL_COMPOSITION_NOT_AN_ADDITIVITY_THEOREM',
                        'SHARED_RUNTIME_GUARD_FFI_AND_NATIVE_NOOP_BODY','DYNAMIC_CPU_AND_TABLE_ID_SETTERS_OUTSIDE_TIMING'])
                boundary_components.append(row)
            else:
                try:
                    measurement=probe.sample(term['component'],term['parameters'],repetitions=3,warmup=1)
                    resource_components.append(_row(term['component'],term['parameters'],measurement,'DIRECT_COMPONENT_TIMING'))
                except NotImplementedError as error:
                    unsupported.append(dict(component=term['component'],parameters=term['parameters'],reason=str(error)))
            _need(_rss() < _LIMIT, 'sample process exceeded the explicit RSS limit')
        extras.append(dict(component='OMP_BARRIER_ONLY',measurement=probe.sample('OMP_BARRIER_ONLY',dict(workers=len(cpus)),3,1),
            used_in_composition=False,reason='stage timings already include their own completion barrier'))
    finally:
        after=sorted(os.sched_getaffinity(0))
        if after != actual: os.sched_setaffinity(0,actual)
    _need(after == actual and thread_environment() == environment, 'sampling changed caller affinity/thread controls')
    for receipt in document['implementation_receipts']: _check(receipt)
    for key in ('source_region','base_stage_plan'): read_artifact(context[key],json_document=False)
    shared=dict(schema='scar.stateful-resource-calibration',schema_version=1,resource_context=context['resource_context'],probe_context=probe_context,
        repetitions=3,warmup=1,observed_from=observed_from,observed_until=_now(),
        host_load=dict(before=load_before,after=list(os.getloadavg()),isolated=False,contention_not_controlled=True),
        observed_caller_affinity_before=actual,observed_caller_affinity_after=after,observed_native_thread_environment=environment,
        caller_affinity_restored=True,numa_page_residency_proven=False,implicit_stage_barrier_cost_already_included=True,
        caller_placement='ORIGINAL_CALLER_MASK_UNBOUND_SEPARATE_FROM_PINNED_WORKERS',no_scope_median_subtraction=True,
        whole_method_boundary_measured=False,target_workload_executed=False,target_variants_compiled=False,target_variants_ranked=False,
        physical_optimality_proven=False,shared_pack=pack if isinstance(pack,dict) else str(pack),process_id=os.getpid(),
        one_candidate_context_per_process=True,native_recompiled=False,peak_rss_bytes=_rss())
    rref=_save(out/'resource-calibration.json',dict(**shared,components=resource_components,unsupported_components=unsupported,extra_observations=extras,
        timing_scope='DIRECT_INDEPENDENT_RESOURCE_PRIMITIVES_NOT_TARGET_MATH_OR_WHOLE_CALL'))
    bref=_save(out/'boundary-calibration.json',dict(**shared,components=boundary_components,unsupported_components=[],extra_observations=[],
        canonical_structure=structure,canonical_runtime_contract=_receipt(out/'canonical-contract.json'),
        timing_scope='CANONICAL_SHARED_BOUNDARY_AND_TAGGED_SCALAR_MODEL_NOT_TARGET_CALL'))
    witness=_save(out/'runtime-witness.json',dict(schema='scar.cpu-batch-aux-runtime-witness',schema_version=1,
        process_id=os.getpid(),candidate_id=context['candidate_id'],mode=document['mode'],cpu_ids=cpus,
        actual_caller_affinity_before=actual,actual_caller_affinity_after=after,openmp=openmp,
        libraries=document['libraries'],table_ids=ids,table_setter_verified_before_sampling=True,
        setters_outside_all_timed_intervals=True,native_recompiled=False,original_stage_bodies_reused=True,
        peak_rss_bytes=_rss(),max_rss_bytes=_LIMIT,target_workload_executed=False))
    result=dict(resource_calibration=rref,boundary_calibration=bref,runtime_witness=witness,sampling_seconds=time.perf_counter()-started)
    _save(out/'result.json',dict(schema='scar.cpu-batch-aux-sample-result',schema_version=1,**result))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack',required=True);parser.add_argument('--context',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    print(json.dumps(sample_aux_pack(args.pack,args.context,args.out),sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
