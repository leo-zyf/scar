"""Source-faithful canonical component samples for original scalar/array DAGs.

The component topology is specialized and compiled. Only independent canonical
stage inputs are used: neither a target entry nor its whole pipeline is run.
All predictions retain implementation, data-regime and placement residuals.
"""
import argparse
from copy import deepcopy
import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

from scar.compiler.calibrate_stateful_boundary import _json, _now, _receipt
from scar.compiler.stateful_model import stateful_from_dict
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.stateful_cpu_choices import _domain
from scar.optimization.stateful_cpu_model import stateful_cpu_demand

SCHEMA = 'scar.stateful-stage-dag-calibration'
_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = (
    'ONE_SOURCE_DERIVED_SERIAL_COMPUTE_OR_SCALAR_COMPONENT_NOT_WHOLE_TARGET_EXECUTION',
    'EXACT_ORIGINAL_STAGE_BODY_ORDERED_DAG_ENTRY_PORTS_AND_PHYSICAL_SLOT_ALIASING',
    'CANONICAL_ARRAYS_1_25_RUNTIME_AND_READY_SCALARS_0_75_WITH_FLOAT_TAGS',
    'WARM_REPEATED_BUFFERS_WITH_FIRST_TOUCH_AND_NUMA_RESIDENCY_NOT_PROVEN',
    'POST_CLOCK_CHECKSUM_RESCAN_CAN_WARM_OUTPUT_BETWEEN_COMPONENT_SAMPLES',
    'EXACT_RETAINED_COMPILER_OPTIONS_CPU_MASK_RUNTIME_AND_HARDWARE_CONTEXT',
    'COMPONENT_TOPOLOGY_SPECIALIZED_COMPILATION_NOT_COUNTS_ONLY_GENERIC_PRIMITIVE',
    'CLOCK_FENCES_ENTRY_PORT_LOADS_AND_ISOLATED_FUNCTION_CONTEXT_RETAINED',
    'CANONICAL_SAMPLE_RANGE_IS_NOT_A_GUARANTEED_TARGET_BOUND',
)
MISSING = ('TARGET_DATA_SCALAR_TAG_AND_FLOATING_EXCEPTION_REGIME_NOT_VALIDATED',
    'WHOLE_FUNCTION_COMPILATION_REGISTER_VECTOR_AND_MEMORY_CONTEXT_DELTA',
    'POST_PROVIDER_AFFINITY_FLOATING_STATE_AND_INTERFERENCE_DELTA',
    'INSTRUMENTATION_PERTURBATION_NOT_BOUNDED')
_FILES = ('compiler/calibrate_stage_dag.py', 'compiler/calibrate_stateful_boundary.py',
    'codegen/stateful_stage_capture.py', 'codegen/stateful_stage_probe.py',
    'codegen/stateful_stage_probe_v1.cpp.in', 'codegen/stateful.py',
    'codegen/stateful_v1.cpp.in', 'codegen/loop_v1.cpp.in', 'codegen/cpp.py',
    'codegen/boundary_cuts.py', 'codegen/stateful_abi_v1.h', 'codegen/scar_abi_v1.h',
    'optimization/stateful_stage_signature.py')
_CLOCK = dict(api='clock_gettime', clock_id='CLOCK_MONOTONIC', unit='ns')


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _wire(value):
    return json.loads(_json(value))


def _pins():
    return {name: sha256((_ROOT/name).read_bytes()).hexdigest() for name in _FILES}


def _statistics(samples, repetitions):
    _need(type(samples) is list and len(samples) == repetitions
          and all(type(value) is int and value >= 0 for value in samples),
          'retained integer nanosecond samples required for each original component')
    return dict(median_ns=statistics.median(samples), min_ns=min(samples),
        max_ns=max(samples), stdev_ns=statistics.stdev(samples))


def _spec(domain, candidate_id, stage_indices=None, *, max_payload_bytes=256*2**20):
    from scar.codegen.stateful_stage_capture import capture_stateful_stage_sources
    from scar.codegen.stateful_stage_probe import render_stage_dag_probe
    from scar.optimization.stateful_stage_signature import project_stage_signature
    candidate = next((row for row in domain['candidates'] if row['id'] == candidate_id), None)
    _need(candidate is not None and candidate['legal_under_guards'] and candidate['worker_count'] == 1,
          'stage-DAG v1 requires one original admitted serial CPU candidate')
    region = stateful_from_dict(read_artifact(domain['region']))
    plan = read_artifact(domain['base_stage_plan'])
    _need(plan['cpu_threads'] == 1, 'canonical stage capture requires original serial base plan')
    demand = stateful_cpu_demand(domain, candidate_id)
    eligible = [int(term['source_ref']['pointer'].rsplit('/', 1)[1]) for term in demand['terms']
        if term['component'] in ('FUSED_SCALAR_DAG_ROW_STAGE', 'TAGGED_SCALAR_SETUP_STAGE')]
    indices = eligible if stage_indices is None else list(stage_indices)
    _need(indices and len(indices) == len(set(indices)) and len(indices) <= 1024
          and all(type(index) is int and index in eligible for index in indices),
          'unique original fused-DAG/tagged-scalar component stages required')
    cpus = candidate['binding']['cpu_ids']; backend = domain['backend_capabilities']['serial']
    contract = dict(region_sha256=domain['region_sha256'], plan_sha256=semantic_sha256(plan),
        provider_runtime=backend['provider_runtime'])
    capture = _wire(capture_stateful_stage_sources(region, plan, SimpleNamespace(cpu_ids=cpus), contract))
    signatures = [_wire(project_stage_signature(region, plan, capture, index)) for index in indices]
    source, provenance = render_stage_dag_probe(region, plan, capture, indices, cpus,
        max_payload_bytes=max_payload_bytes)
    return dict(region=region, plan=plan, cpus=cpus, compiler=backend['compiler'],
        stage_indices=indices, capture=capture, signatures=signatures, source=source,
        provenance=_wire(provenance), resource_context=demand['resource_context'],
        max_payload_bytes=max_payload_bytes)


def _build(spec, directory):
    directory.mkdir()
    compiler = spec['compiler']
    identity = subprocess.run([*compiler['argv'], '--version'], text=True, capture_output=True, check=True).stdout.splitlines()[0]
    _need(identity == compiler['identity'] and '-fopenmp' not in compiler['options'],
          'serial component compiler differs from retained backend')
    source = directory/'probe.cpp'; source.write_text(spec['source'])
    headers = {}
    for name in ('stateful_abi_v1.h', 'scar_abi_v1.h'):
        (directory/name).write_bytes((_ROOT/'codegen'/name).read_bytes())
        headers[name] = _receipt(directory/name)
    library = directory/'stage_dag_probe.so'
    command = [*compiler['argv'], *compiler['options'], str(source), '-o', str(library), '-lm']
    result = subprocess.run(command, text=True, capture_output=True)
    (directory/'build.log').write_text(result.stdout+result.stderr)
    _need(result.returncode == 0, 'source-faithful stage probe compilation failed: '+result.stderr[-2000:])
    handle = ctypes.CDLL(str(library)); run = handle.scar_stage_dag_probe_v1
    run.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint64)]
    run.restype = ctypes.c_int32
    return handle, run, dict(source=_receipt(source), library=_receipt(library), headers=headers,
        compiler=compiler, compiler_command=command, implementation_files=_pins(),
        source_provenance=spec['provenance'], max_payload_bytes=spec['max_payload_bytes'],
        component_topology_specialized=True, target_fragment_compiled=True,
        full_target_program_executed=False, whole_target_call_measured=False)


def calibrate_stage_dag(domain, candidate_id, out, *, repetitions=7, warmup=2,
                        stage_indices=None, max_payload_bytes=256*2**20):
    from scar.runtime.compiled_deployment import thread_environment
    _need(sys.gettrace() is None and sys.getprofile() is None, 'stage-DAG sampling requires no trace/profile hook')
    _need(type(repetitions) is int and 3 <= repetitions <= 10000
          and type(warmup) is int and 1 <= warmup <= 10000, 'bounded warm repeated samples required')
    started = time.monotonic(); observed_from = _now()
    domain = _domain(domain); spec = _spec(domain, candidate_id, stage_indices,
        max_payload_bytes=max_payload_bytes)
    actual = sorted(os.sched_getaffinity(0)); environment = thread_environment()
    _need(actual == domain['live_constraints']['process_affinity_cpu_ids']
          and set(spec['cpus']) <= set(actual)
          and environment == domain['live_constraints']['native_thread_environment'],
          'stage-DAG caller mask and native thread environment must match original domain')
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=False)
    capture_path = out/'source-capture.json'; capture_path.write_bytes(_json(spec['capture']))
    library, run, probe = _build(spec, out/'native')
    components = []; load_before = list(os.getloadavg())
    try:
        for component_index, (stage, signature) in enumerate(zip(spec['stage_indices'], spec['signatures'])):
            values = (ctypes.c_uint64*repetitions)(); placement = (ctypes.c_int32*3)()
            checksum = ctypes.c_uint64()
            status = run(component_index, repetitions, warmup, values, placement, ctypes.byref(checksum))
            _need(status == 0, 'canonical source stage declined (arguments/FP/CPU/allocation): '+str(status))
            _need(placement[1] == spec['cpus'][0] and sorted(os.sched_getaffinity(0)) == actual,
                  'canonical component selected CPU/restored caller mask differs')
            samples = list(values)
            components.append(dict(stage_index=stage,
                stage_ref=dict(artifact=domain['base_stage_plan'], locator='#/stages/'+str(stage)),
                signature=signature, samples_ns=samples, statistics=_statistics(samples, repetitions),
                placement=dict(caller_cpu_before=placement[0], selected_cpu=placement[1],
                    caller_cpu_after=placement[2], exact_caller_affinity_restored=True), checksum=checksum.value,
                basis='SOURCE_TOPOLOGY_SPECIALIZED_CANONICAL_COMPONENT', target_activity_observed=False))
    finally:
        after = sorted(os.sched_getaffinity(0))
        if after != actual:
            os.sched_setaffinity(0, actual)
    _need(after == actual and thread_environment() == environment, 'component probe failed to restore caller context')
    document = dict(schema=SCHEMA, schema_version=1, domain_id=domain['domain_id'], candidate_id=candidate_id,
        domain_source=dict(region=domain['region'], stage_plan=domain['base_stage_plan'], target=domain['target_snapshot']),
        resource_context=spec['resource_context'], components=components,
        source_capture=artifact_reference(capture_path, spec['capture']['schema']),
        stage_indices=spec['stage_indices'], conditions=list(CONDITIONS), missing_terms=list(MISSING),
        repetitions=repetitions, warmup=warmup, clock=_CLOCK, probe=probe,
        observed_from=observed_from, observed_until=_now(), seconds=time.monotonic()-started,
        host_load=dict(before=load_before, after=list(os.getloadavg()), isolated=False),
        observed_caller_affinity_before=actual, observed_caller_affinity_after=after,
        caller_affinity_restored=True, observed_native_thread_environment=environment,
        component_topology_specialized=True, target_fragment_compiled=True,
        full_target_program_executed=False, target_variants_ranked=False, whole_target_call_measured=False,
        independent_canonical_stage_inputs=True, clock_overhead_subtracted=False,
        physical_optimality_proven=False, deployment_authorized=False)
    path = out/'observations.json'; path.write_bytes(_json(document))
    result = dict(schema='scar.stateful-stage-dag-result', schema_version=1,
        observations=artifact_reference(path, SCHEMA), components=[dict(stage_index=row['stage_index'],
            component=row['signature']['component'], signature_key=row['signature']['key'],
            **row['statistics']) for row in components],
        component_topology_specialized=True, full_target_program_executed=False, deployment_authorized=False)
    (out/'result.json').write_bytes(_json(result))
    return result


def read_stage_dag_calibration(reference, *, domain, candidate_id):
    """Reproduce source capture, DAG signatures and native source without running."""
    if isinstance(reference, (str, Path)):
        reference = artifact_reference(Path(reference).resolve(), SCHEMA)
    document = read_artifact(reference)
    _need((document.get('schema'), document.get('schema_version')) == (SCHEMA, 1),
          'versioned stage-DAG component calibration required')
    domain = _domain(domain) if isinstance(domain, (str, Path)) else domain
    probe = document['probe']
    spec = _spec(domain, candidate_id, document['stage_indices'], max_payload_bytes=probe['max_payload_bytes'])
    _need(document.get('domain_id') == domain['domain_id'] and document.get('candidate_id') == candidate_id
          and document.get('resource_context') == spec['resource_context']
          and document.get('domain_source') == dict(region=domain['region'], stage_plan=domain['base_stage_plan'],
              target=domain['target_snapshot']), 'original stage-DAG source/resource context differs')
    _need(read_artifact(document['source_capture']) == spec['capture'], 'retained stage capture differs from original renderer replay')
    _need(probe.get('implementation_files') == _pins() and probe.get('source_provenance') == spec['provenance']
          and probe.get('compiler') == spec['compiler'], 'stage-DAG source and implementation provenance differs')
    _need(set(probe['headers']) == {'stateful_abi_v1.h', 'scar_abi_v1.h'}, 'complete original operator ABI headers required')
    for receipt in (probe['source'], probe['library'], *probe['headers'].values()):
        _need(_receipt(receipt['path']) == receipt, 'retained stage-DAG native artifact changed')
    _need(Path(probe['source']['path']).read_text() == spec['source']
          and all(Path(receipt['path']).read_bytes() == (_ROOT/'codegen'/name).read_bytes()
                  for name, receipt in probe['headers'].items()), 'source-faithful native stage source/headers differ')
    _need(probe.get('compiler_command') == [*spec['compiler']['argv'], *spec['compiler']['options'],
        probe['source']['path'], '-o', probe['library']['path'], '-lm'], 'stage-DAG compilation command differs')
    _need(document.get('clock') == _CLOCK and document.get('conditions') == list(CONDITIONS)
          and document.get('missing_terms') == list(MISSING), 'component clock/applicability/residual contract differs')
    repetitions, warmup = document['repetitions'], document['warmup']
    _need(type(repetitions) is int and 3 <= repetitions <= 10000
          and type(warmup) is int and 1 <= warmup <= 10000, 'original warm repeated component samples required')
    _need(len(document['components']) == len(spec['stage_indices']), 'all original component signatures required')
    for index, signature, component in zip(spec['stage_indices'], spec['signatures'], document['components']):
        _need(component.get('stage_index') == index and component.get('signature') == signature
              and component.get('stage_ref') == dict(artifact=domain['base_stage_plan'], locator='#/stages/'+str(index)),
              'component signature differs from full original DAG/entry/source-stage replay')
        _need(component.get('statistics') == _statistics(component.get('samples_ns'), repetitions)
              and component.get('basis') == 'SOURCE_TOPOLOGY_SPECIALIZED_CANONICAL_COMPONENT'
              and component.get('target_activity_observed') is False, 'retained canonical component statistics/scope differ')
        _need(component['placement'].get('selected_cpu') == spec['cpus'][0]
              and component['placement'].get('exact_caller_affinity_restored') is True,
              'component placement differs from source/candidate resource context')
    _need(document.get('observed_caller_affinity_before') == spec['resource_context']['caller_affinity_cpu_ids']
          and document.get('observed_caller_affinity_after') == document['observed_caller_affinity_before']
          and document.get('observed_native_thread_environment') == spec['resource_context']['native_thread_environment']
          and document.get('caller_affinity_restored') is True, 'component caller resource context differs')
    for name in ('component_topology_specialized', 'target_fragment_compiled'):
        _need(document.get(name) is True and probe.get(name) is True, 'component source specialization must be explicit')
    _need(document.get('independent_canonical_stage_inputs') is True, 'independent canonical stage regime required')
    for name in ('full_target_program_executed', 'target_variants_ranked', 'whole_target_call_measured',
                 'clock_overhead_subtracted', 'physical_optimality_proven', 'deployment_authorized'):
        _need(document.get(name) is False, 'component evidence must not claim '+name)
    _need(probe.get('full_target_program_executed') is False and probe.get('whole_target_call_measured') is False,
          'isolated stage probe cannot claim complete target scope')
    read_artifact(reference)
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', required=True); parser.add_argument('--candidate', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--out'); source.add_argument('--check')
    parser.add_argument('--stage', action='append', type=int)
    parser.add_argument('--repetitions', type=int, default=7); parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--max-payload-bytes', type=int, default=256*2**20)
    args = parser.parse_args(argv)
    if args.check:
        _need(args.stage is None, 'check replays only the original retained stage set')
        document = read_stage_dag_calibration(args.check, domain=args.domain, candidate_id=args.candidate)
        result = dict(verified_components=len(document['components']), source_replayed=True,
            full_target_program_executed=False, deployment_authorized=False)
    else:
        result = calibrate_stage_dag(args.domain, args.candidate, args.out, repetitions=args.repetitions,
            warmup=args.warmup, stage_indices=args.stage, max_payload_bytes=args.max_payload_bytes)
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
