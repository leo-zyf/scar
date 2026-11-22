"""Exclusive same-clock cuts through canonical compiler stateful boundaries.

Runs the real shared guard, compiler adapter and StatefulArtifact.run against a
source-derived no-op table. It never imports a target module or executes its
DAG. These samples predict implementation fragments conditionally, not target
call latency, and never authorize a transformation.
"""
import argparse
from copy import deepcopy
import ctypes
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from scar.compiler.calibrate_stateful_boundary import (_canonical_binding,
    _json, _now, _receipt, _source)
from scar.compiler.stateful_model import stateful_from_dict
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.stateful_cpu_model import _boundary_parameters, _domain, _resource_context
from scar.runtime.stateful_contract import (ASSUMPTIONS, FP_POLICY, MEMORY_SCOPE,
    canonical_sha256, validate_contract)

SCHEMA = 'scar.stateful-boundary-cut-calibration'
_ROOT = Path(__file__).resolve().parents[1]
CUTS = ('outer_enter', 'native_enter', 'metadata_ready', 'capture_ready',
        'binding_ready', 'worker_checked', 'restore_ready', 'outer_return')
_KINDS = (('PYTHON_ENTRY_GUARD_PACKING',), ('NATIVE_METADATA_GUARD',),
          ('CALLER_CAPTURE_SCALAR_GUARD',), ('TEAM_ENTER',), ('WORKER_EXIT',),
          ('CALLER_RESTORE',), ('NATIVE_RETURN', 'PYTHON_STATUS_RETURN'))
PHASES = tuple(dict(id=name, start=CUTS[index], end=CUTS[index+1], phase_kinds=list(kinds))
    for index, (name, kinds) in enumerate(zip(('python_entry', 'metadata', 'capture',
        'serial_bind', 'worker_check', 'caller_restore', 'return_handoff'), _KINDS)))
CONDITIONS = (
    'ONE_STANDALONE_SUCCESSFUL_SERIAL_COMPILER_ARTIFACT_CALL_NOT_COMPACT_OR_REPEAT',
    'WARM_CANONICAL_ORDINARY_RECEIVERS_ALL_FLOAT_SCALARS_AND_INDEPENDENT_ARRAY_ROOTS',
    'EXACT_RETAINED_SOURCE_BACKEND_SHAPES_CPU_MASK_AND_RUNTIME_CONTEXT',
    'SKIPPED_POOL_SCANS_DAG_AND_COMMITS_CAN_CHANGE_CACHE_ALLOCATOR_AND_PLACEMENT_REGIME',
    'SAME_CLOCK_CUTS_PARTITION_ONE_CANONICAL_CALL_NOT_A_TARGET_LATENCY_OBSERVATION',
    'CLOCK_READ_AND_CUT_STORAGE_OVERHEAD_RETAINED_NOT_MEDIAN_SUBTRACTED',
    'CANONICAL_SAMPLE_RANGE_IS_NOT_A_GUARANTEED_TARGET_BOUND',
)
MISSING = ('TARGET_IMPLEMENTATION_AND_DATA_REGIME_DELTA_NOT_VALIDATED',
    'INSTRUMENTATION_PERTURBATION_NOT_BOUNDED',
    'POST_PROVIDER_AFFINITY_AND_FLOATING_STATE_NOT_REPRODUCED_BY_EMPTY_DAG')
SCOPE_SCHEMA = 'scar.stateful-scope-boundary-calibration'
SCOPE_CUTS = (*CUTS[:-1], 'helper_cleanup_ready', 'outer_return')
SCOPE_PHASES = (*deepcopy(PHASES[:-1]),
    dict(id='helper_cleanup', start='restore_ready', end='helper_cleanup_ready',
        phase_kinds=['NATIVE_HELPER_CLEANUP']),
    dict(id='return_handoff', start='helper_cleanup_ready', end='outer_return',
        phase_kinds=['NATIVE_RETURN', 'PYTHON_STATUS_RETURN']))
SCOPE_CONDITIONS = (*CONDITIONS,
    'ORIGINAL_HELPER_DESTRUCTION_PRECEDES_WORKSPACE_POOL_RELEASE',
    'EXPLICIT_HELPER_CLEANUP_CUT_SEPARATES_DESTRUCTION_FROM_ABI_PYTHON_HANDOFF')
SCOPE_MISSING = (*MISSING, 'HELPER_SCOPE_REASSEMBLY_AND_ADDITIONAL_CLOCK_CUT_DELTA',
    'ABSENT_WORKSPACE_DESTRUCTION_CAN_CHANGE_CANONICAL_HANDOFF_REGIME')


@dataclass(frozen=True, slots=True)
class _Profile:
    schema: str
    result_schema: str
    cuts: tuple[str, ...]
    phases: tuple[dict, ...]
    conditions: tuple[str, ...]
    missing: tuple[str, ...]
    sample_name: str


_MIXED = _Profile(SCHEMA, 'scar.stateful-boundary-cut-result', CUTS, PHASES, CONDITIONS,
                  MISSING, 'canonical-call')
_SCOPE = _Profile(SCOPE_SCHEMA, 'scar.stateful-scope-boundary-result', SCOPE_CUTS, SCOPE_PHASES,
                  SCOPE_CONDITIONS, SCOPE_MISSING, 'canonical-scope-call')
_CLOCK = dict(api='clock_gettime', clock_id='CLOCK_MONOTONIC', unit='ns')
_RUNTIME_PATH = (
    'scar.runtime.stateful_guard.guarded_stateful_function.call',
    'scar.compiler.stateful_runtime._ArtifactAdapter.run',
    'scar.codegen.stateful.StatefulArtifact.run',
)
_ABI_HEADERS = ('stateful_abi_v1.h', 'scar_abi_v1.h')
_FILES = ('compiler/calibrate_boundary_cuts.py', 'compiler/calibrate_stateful_boundary.py',
    'compiler/stateful_runtime.py', 'codegen/boundary_cuts.py',
    'codegen/stateful_boundary_cuts_v1.cpp.in', 'codegen/stateful.py',
    'codegen/stateful_v1.cpp.in', 'codegen/loop_v1.cpp.in', 'codegen/cpp.py',
    'codegen/stateful_abi_v1.h', 'codegen/scar_abi_v1.h',
    'runtime/stateful_guard.py', 'runtime/stateful_contract.py', 'trace/code_snapshot.py',
    'optimization/stateful_cpu_model.py')
_SCOPE_FILES = ('analysis/implementation_scope.py', 'codegen/scope_boundary_probe.py',
                'codegen/stateful_scope_boundary_v1.cpp.in')


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _pins(profile):
    files = (*_FILES, *_SCOPE_FILES) if profile is _SCOPE else _FILES
    return {name: sha256((_ROOT/name).read_bytes()).hexdigest() for name in files}


def boundary_cut_statistics(document):
    """Validate the exclusive partition before computing any numeric costs."""
    return _statistics(document, _MIXED)


def scope_boundary_statistics(document):
    """Validate the helper-destructor cut before computing costs."""
    return _statistics(document, _SCOPE)


def _statistics(document, profile):
    _need((document.get('schema'), document.get('schema_version')) == (profile.schema, 1),
          'versioned boundary cut calibration required')
    _need(document.get('cuts') == list(profile.cuts) and document.get('phases') == list(profile.phases),
          'exact source boundary cut order and contiguous exclusive phases required')
    _need(document.get('conditions') == list(profile.conditions)
          and document.get('missing_terms') == list(profile.missing),
          'canonical boundary applicability and implementation residuals required')
    _need(document.get('clock') == _CLOCK,
          'shared native/Python integer monotonic clock required')
    repetitions, warmup = document.get('repetitions'), document.get('warmup')
    _need(type(repetitions) is int and 3 <= repetitions <= 10000
          and type(warmup) is int and 1 <= warmup <= 10000, 'bounded warm repeated samples required')
    samples = document.get('samples')
    _need(type(samples) is list and len(samples) == repetitions, 'all original cut samples required')
    seen, columns = set(), [[] for _ in profile.phases]
    for sample in samples:
        identity, cuts = sample.get('id'), sample.get('timestamps_ns')
        _need(type(identity) is str and identity and identity not in seen, 'unique shared sample identities required')
        seen.add(identity)
        _need(type(cuts) is list and len(cuts) == len(profile.cuts)
              and all(type(value) is int and value > 0 for value in cuts)
              and all(left <= right for left, right in zip(cuts, cuts[1:])),
              'same-clock monotone integer cuts required; no negative or missing interval')
        durations = [right-left for left, right in zip(cuts, cuts[1:])]
        _need(sum(durations) == cuts[-1]-cuts[0], 'exclusive intervals must exactly partition each call')
        for column, duration in zip(columns, durations):
            column.append(duration)
    return [dict(phase=phase['id'], samples_ns=values, median_ns=statistics.median(values),
        min_ns=min(values), max_ns=max(values), stdev_ns=statistics.stdev(values))
        for phase, values in zip(profile.phases, columns)]


def _spec(domain, candidate_id, profile):
    candidate = next((row for row in domain['candidates'] if row['id'] == candidate_id), None)
    _need(candidate is not None and candidate['legal_under_guards'] and candidate['worker_count'] == 1,
          'boundary cut v1 requires one admitted serial CPU candidate')
    region = stateful_from_dict(read_artifact(domain['region']))
    plan = read_artifact(domain['base_stage_plan'])
    parameters = _boundary_parameters(region, plan)
    memory = {key: plan[key] for key in ('external_array_bytes', 'workspace_payload_bytes',
        'total_region_array_payload_bytes', 'memory_budget_bytes')}
    backend = domain['backend_capabilities']['serial']
    cpus = candidate['binding']['cpu_ids']
    core = dict(region_sha256=canonical_sha256(parameters),
        plan_sha256=canonical_sha256(dict(parameters=parameters, cpu_ids=cpus, memory=memory)),
        provider_runtime=backend['provider_runtime'])
    if profile is _SCOPE:
        from scar.codegen.scope_boundary_probe import render_scope_boundary_probe as render
    else:
        from scar.codegen.boundary_cuts import render_boundary_cut_probe as render
    source, provenance = render(parameters, memory, cpus, core)
    return dict(region=region, plan=plan, memory=memory, cpus=cpus, core=core,
        compiler=backend['compiler'], source=source, provenance=provenance,
        resource_context=_resource_context(domain, candidate_id, candidate))


def _build(spec, directory, profile):
    from scar.codegen.stateful import _Table
    directory.mkdir()
    compiler = spec['compiler']
    identity = subprocess.run([*compiler['argv'], '--version'], text=True, capture_output=True, check=True).stdout.splitlines()[0]
    _need(identity == compiler['identity'] and '-fopenmp' not in compiler['options'],
          'canonical serial compiler differs from source context')
    source = directory/'probe.cpp'; source.write_text(spec['source'])
    headers = {}
    for name in _ABI_HEADERS:
        (directory/name).write_bytes((_ROOT/'codegen'/name).read_bytes())
        headers[name] = _receipt(directory/name)
    library = directory/'boundary_cuts.so'
    command = [*compiler['argv'], *compiler['options'], str(source), '-o', str(library), '-lm']
    result = subprocess.run(command, text=True, capture_output=True)
    (directory/'build.log').write_text(result.stdout+result.stderr)
    _need(result.returncode == 0, 'boundary cuts compilation failed: '+result.stderr[-2000:])
    handle = ctypes.CDLL(str(library))
    getter = handle.scar_get_stateful_table_v1
    getter.argtypes = []; getter.restype = ctypes.POINTER(_Table)
    table = getter()
    _need(table and table.contents.abi_version == 1 and table.contents.struct_size == ctypes.sizeof(_Table)
          and table.contents.run, 'canonical compiler boundary ABI differs')
    _need((table.contents.array_count, table.contents.scalar_count, table.contents.provider_count) == (
        sum(row.kind == 'array' for row in spec['region'].inputs),
        sum(row.kind == 'scalar' for row in spec['region'].inputs), len(spec['region'].providers)),
        'canonical native input counts differ')
    for key in ('region_sha256', 'plan_sha256', 'contract_sha256'):
        expected = canonical_sha256(spec['core']) if key == 'contract_sha256' else spec['core'][key]
        _need(getattr(table.contents, key) == expected.encode(), 'canonical table identity differs')
    return handle, table, dict(source=_receipt(source), library=_receipt(library), headers=headers,
        compiler=compiler, compiler_command=command, source_fragments=spec['provenance'],
        implementation_files=_pins(profile), target_dag_compiled=False)


def _contract(original, inputs, callees, providers, region, spec, probe, out):
    ids = dict(build_id=probe['library']['sha256'], region_sha256=spec['core']['region_sha256'],
        plan_sha256=spec['core']['plan_sha256'], contract_sha256=canonical_sha256(spec['core']))
    selection = dict(decision='TRANSFORM_UNDER_CONTRACT', selected=True, artifact=ids,
        input_shapes={row['id']: row['shape'] for row in inputs if row['kind'] == 'array'},
        external_assumptions=sorted(ASSUMPTIONS))
    return validate_contract(dict(schema='scar.stateful-runtime-contract', schema_version=1,
        profile='cpu-f64-rank2-stateful-v1',
        native=dict(library=Path(probe['library']['path']).name, library_sha256=probe['library']['sha256'],
            stateful_abi_version=1, ids=ids, contract=spec['core']),
        source=_source(original, out/'canonical_boundary.py'), parameters=list(region.parameter_names),
        inputs=inputs, callee_guards=callees, providers=providers,
        cpu=dict(ids=spec['cpus'], threads=1, architecture='x86_64', os='linux', pointer_bits=64,
            byteorder='little', required_isa='BASELINE_X86_64'),
        memory=dict(**spec['memory'], scope=MEMORY_SCOPE, policy='UNBOUND'),
        provider_runtime=spec['core']['provider_runtime'], guards=['CANONICAL_COMPILER_BOUNDARY_CUTS'],
        external_assumptions=sorted(ASSUMPTIONS), floating_environment_policy=FP_POLICY, selection=selection))


def calibrate_boundary_cuts(domain, candidate_id, out, *, repetitions=7, warmup=2):
    return _calibrate(domain, candidate_id, out, _MIXED, repetitions, warmup)


def calibrate_scope_boundary(domain, candidate_id, out, *, repetitions=7, warmup=2):
    return _calibrate(domain, candidate_id, out, _SCOPE, repetitions, warmup)


def _calibrate(domain, candidate_id, out, profile, repetitions, warmup):
    from scar.codegen.stateful import StatefulArtifact, _providers
    from scar.compiler.stateful_runtime import _ArtifactAdapter
    from scar.runtime.stateful_guard import guarded_stateful_function
    from scar.runtime.compiled_deployment import thread_environment
    _need(sys.gettrace() is None and sys.getprofile() is None, 'boundary cuts require no trace/profile hook')
    _need(type(repetitions) is int and 3 <= repetitions <= 10000
          and type(warmup) is int and 1 <= warmup <= 10000, 'bounded warm repeated samples required')
    started = time.monotonic(); observed_from = _now()
    domain = _domain(domain); spec = _spec(domain, candidate_id, profile)
    actual = sorted(os.sched_getaffinity(0)); environment = thread_environment()
    _need(actual == domain['live_constraints']['process_affinity_cpu_ids']
          and set(spec['cpus']) <= set(actual) and environment == domain['live_constraints']['native_thread_environment'],
          'caller mask/thread environment must match original domain')
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    original, args, inputs, callees, providers, structure = _canonical_binding(spec['region'], spec['plan'], out)
    library, table, probe = _build(spec, out/'native', profile)
    contract = _contract(original, inputs, callees, providers, spec['region'], spec, probe, out)
    (out/'canonical-contract.json').write_bytes(_json(contract))
    runtime, native_providers, owners, power = _providers(spec['region'])
    _need(runtime == spec['core']['provider_runtime'], 'canonical provider binary differs from admitted backend')
    metadata = dict(contract['native']['ids'])
    artifact = StatefulArtifact(spec['region'], spec['plan'], out, metadata, library, table,
        native_providers, owners, power)
    # Installation/projection is deliberately canonical. Inside the timed call
    # these are the unchanged actual compiler adapter and artifact methods.
    adapter = object.__new__(_ArtifactAdapter)
    adapter.original, adapter.contract, adapter.report = artifact, contract, artifact.report
    wrapper = guarded_stateful_function(original, adapter, owner=type(args[0]),
        exclusive_inputs=True, selection=contract['selection'])
    read_cuts = library.scar_boundary_cut_read_v1
    read_cuts.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint64]
    read_cuts.restype = ctypes.c_int32
    native_count = len(profile.cuts) - 2
    native_cuts = (ctypes.c_uint64 * native_count)()
    samples = []; load_before = list(os.getloadavg())
    try:
        for rep in range(repetitions+warmup):
            enter = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            result = wrapper(*args)
            returned = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            _need(result is None and read_cuts(native_cuts, native_count) == 0,
                  'canonical call or cut collection failed')
            if rep >= warmup:
                samples.append(dict(id=f'{profile.sample_name}:{rep-warmup}',
                    timestamps_ns=[enter, *native_cuts, returned]))
        _need(wrapper.scar_stats['native_calls'] == repetitions+warmup
              and wrapper.scar_stats['fallback_calls'] == 0, 'actual compiler native path was not used')
    finally:
        after = sorted(os.sched_getaffinity(0))
        if after != actual:
            os.sched_setaffinity(0, actual)
    _need(after == actual and thread_environment() == environment, 'canonical caller context was not restored')
    document = dict(schema=profile.schema, schema_version=1, domain_id=domain['domain_id'], candidate_id=candidate_id,
        domain_source=dict(region=domain['region'], stage_plan=domain['base_stage_plan'],
            target=domain['target_snapshot']), resource_context=spec['resource_context'],
        cuts=list(profile.cuts), phases=deepcopy(list(profile.phases)), samples=samples,
        repetitions=repetitions, warmup=warmup, clock=dict(_CLOCK),
        conditions=list(profile.conditions), missing_terms=list(profile.missing),
        probe=probe, canonical_structure=structure,
        canonical_runtime_contract=_receipt(out/'canonical-contract.json'),
        runtime_path=list(_RUNTIME_PATH),
        runtime_stats=deepcopy(wrapper.scar_stats), observed_from=observed_from, observed_until=_now(),
        seconds=time.monotonic()-started, host_load=dict(before=load_before, after=list(os.getloadavg()), isolated=False),
        observed_caller_affinity_before=actual, observed_caller_affinity_after=after,
        caller_affinity_restored=True, observed_native_thread_environment=environment,
        exclusive_partition_verified=True, clock_overhead_subtracted=False,
        target_workload_executed=False, target_variants_ranked=False, target_variants_compiled=False,
        whole_target_call_measured=False, physical_optimality_proven=False, deployment_authorized=False)
    if profile is _SCOPE:
        document.update(helper_cleanup_separated=True, pool_release_measured=False)
    document['phase_statistics'] = _statistics(document, profile)
    path = out/'observations.json'; path.write_bytes(_json(document))
    result = dict(schema=profile.result_schema, schema_version=1,
        observations=artifact_reference(path, profile.schema), phases=document['phase_statistics'],
        target_workload_executed=False, deployment_authorized=False)
    if profile is _SCOPE:
        result['helper_cleanup_separated'] = True
    else:
        result['exclusive_partition_verified'] = True
    (out/'result.json').write_bytes(_json(result))
    return result


def read_boundary_cuts(reference, *, domain, candidate_id):
    """Replay source fragments and all cuts; never compile or run a probe."""
    return _read_boundary(reference, domain, candidate_id, _MIXED)


def read_scope_boundary_calibration(reference, *, domain, candidate_id):
    """Replay the explicit helper cleanup partition."""
    return _read_boundary(reference, domain, candidate_id, _SCOPE)


def _retained_bytes(receipt):
    path = Path(receipt['path']).resolve()
    raw = path.read_bytes()
    _need(dict(path=str(path), sha256=sha256(raw).hexdigest(), bytes=len(raw)) == receipt,
          'canonical boundary retained artifact changed')
    return raw


def _read_boundary(reference, domain, candidate_id, profile):
    if isinstance(reference, (str, Path)):
        reference = artifact_reference(Path(reference).resolve(), profile.schema)
    document = read_artifact(reference)
    domain = _domain(domain) if isinstance(domain, (str, Path)) else domain
    spec = _spec(domain, candidate_id, profile)
    _need(document.get('domain_id') == domain['domain_id'] and document.get('candidate_id') == candidate_id
          and document.get('domain_source') == dict(region=domain['region'], stage_plan=domain['base_stage_plan'],
              target=domain['target_snapshot']) and document.get('resource_context') == spec['resource_context'],
          'boundary cuts belong to a different source/resource/candidate context')
    _need(document.get('phase_statistics') == _statistics(document, profile),
          'phase statistics differ from retained exclusive cuts')
    probe = document.get('probe', {})
    _need(probe.get('implementation_files') == _pins(profile)
          and probe.get('source_fragments') == spec['provenance']
          and probe.get('compiler') == spec['compiler'] and probe.get('target_dag_compiled') is False,
          'canonical compiler boundary source provenance differs')
    structure = document['canonical_structure']
    source = _retained_bytes(probe['source'])
    _retained_bytes(probe['library'])
    raw_contract = _retained_bytes(document['canonical_runtime_contract'])
    _retained_bytes(structure['canonical_source'])
    _need(source == spec['source'].encode(), 'native boundary source differs from exact fragment replay')
    _need(set(probe['headers']) == set(_ABI_HEADERS)
          and all(_retained_bytes(receipt) == (_ROOT/'codegen'/name).read_bytes()
                  for name, receipt in probe['headers'].items()),
          'canonical ABI headers differ from current source')
    _need(probe.get('compiler_command') == [*spec['compiler']['argv'], *spec['compiler']['options'],
        probe['source']['path'], '-o', probe['library']['path'], '-lm'], 'canonical compile invocation differs')
    contract = validate_contract(json.loads(raw_contract))
    _need(contract['native']['contract'] == spec['core'] and contract['cpu']['ids'] == spec['cpus']
          and contract['native']['library_sha256'] == probe['library']['sha256'], 'canonical runtime contract differs')
    _need(contract['parameters'] == list(spec['region'].parameter_names) == structure['parameters']
          and contract['inputs'] == structure['inputs']
          and [row['path'] for row in contract['inputs']] == [list(row.path) for row in spec['region'].inputs]
          and structure['source_body'] == 'RAISE_IF_FALLBACK_NO_TARGET_DAG'
          and contract['source']['sha256'] == structure['canonical_source']['sha256'],
          'canonical receiver/source structure differs')
    _need(document.get('runtime_path') == list(_RUNTIME_PATH),
        'compiler boundary observations must use actual compiler adapter/artifact path')
    _need(document.get('runtime_stats', {}).get('native_calls') == document['repetitions']+document['warmup']
          and document['runtime_stats'].get('fallback_calls') == 0
          and document.get('observed_caller_affinity_before') == spec['resource_context']['caller_affinity_cpu_ids']
          and document.get('observed_caller_affinity_after') == document['observed_caller_affinity_before']
          and document.get('observed_native_thread_environment') == spec['resource_context']['native_thread_environment']
          and document.get('caller_affinity_restored') is True, 'canonical runtime/placement observations differ')
    for name in ('target_workload_executed', 'target_variants_ranked', 'target_variants_compiled',
                 'whole_target_call_measured', 'clock_overhead_subtracted', 'physical_optimality_proven', 'deployment_authorized'):
        _need(document.get(name) is False, 'canonical fragment measurement must not claim '+name)
    _need(document.get('exclusive_partition_verified') is True, 'exclusive boundary cut partition required')
    if profile is _SCOPE:
        _need(document.get('helper_cleanup_separated') is True and document.get('pool_release_measured') is False,
              'explicit helper cleanup separation without a pool release measurement required')
    read_artifact(reference, json_document=False)
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', required=True); parser.add_argument('--candidate', required=True)
    parser.add_argument('--profile', choices=('mixed', 'scope'), default='mixed')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--out'); source.add_argument('--check')
    parser.add_argument('--repetitions', type=int, default=7); parser.add_argument('--warmup', type=int, default=2)
    args = parser.parse_args(argv)
    profile = _SCOPE if args.profile == 'scope' else _MIXED
    if args.check:
        document = _read_boundary(args.check, args.domain, args.candidate, profile)
        result = dict(phases=document['phase_statistics'], target_workload_executed=False)
        if profile is _SCOPE:
            result.update(source_replayed=True, helper_cleanup_separated=True)
        else:
            result['exclusive_partition_verified'] = True
    else:
        result = _calibrate(args.domain, args.candidate, args.out, profile, args.repetitions, args.warmup)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
