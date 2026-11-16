"""Measure independent canonical workspace boundary fragments, not a target DAG."""
import argparse
import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.versioned_schedule import schedule_from_dict
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.workspace_boundary_costs import (SCHEMA, PHASE_DEFINITIONS,
    cut_layout, boundary_statistics)

_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = (
    'SOURCE_ADMITTED_SERIAL_WORKSPACE_NORMAL_PATH_CANONICAL_BOUNDARY_COMPONENTS_ONLY',
    'EXACT_RETAINED_SOURCE_SHAPES_COUNT_COMPILER_SELECTED_CPU_AND_CANONICAL_CALLER_MASK',
    'FINITE_UNIT_CANONICAL_ARRAYS_AND_FLOAT_SCALARS_NOT_TARGET_INPUTS',
    'LOOP_PHASE_MEDIANS_POOL_RETAINED_INDEX_SAMPLES_NOT_AN_OBSERVED_SEQUENCE_TOTAL',
    'CLOCK_AND_CUT_STORAGE_OVERHEAD_RETAINED_NOT_SUBTRACTED',
    'COMPONENT_SAMPLE_RANGES_ARE_NOT_GUARANTEED_TARGET_LATENCY_BOUNDS',
)
MISSING = (
    'SKIPPED_TARGET_DAG_COMMIT_AND_PROVIDER_STATE_DELTA_NOT_VALIDATED',
    'CANONICAL_CACHE_ALLOCATOR_DATA_AND_FLOATING_STATE_DELTA_NOT_VALIDATED',
    'OPAQUE_PROVIDER_AFFINITY_EFFECTS_NOT_REPRODUCED',
    'INDEX_DEPENDENCE_AND_INSTRUMENTATION_PERTURBATION_NOT_BOUNDED',
    'UNMAPPED_LOOP_CONTROL_GAPS_AND_PRECLOCK_SETUP_AND_FINAL_RETURN',
    'ORIGINAL_CALLER_AFFINITY_EQUAL_TO_CANONICAL_HEADER_REQUIRES_LIVE_CONTRACT',
    'PYTHON_PACKING_LOCK_AND_FFI_NOT_MEASURED',
)


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def _receipt(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return dict(path=str(path), sha256=sha256(raw).hexdigest(), bytes=len(raw))


def _pins():
    return {name:sha256((_ROOT/name).read_bytes()).hexdigest() for name in
        ('compiler/calibrate_workspace_boundary.py','codegen/workspace_boundary_probe.py',
         'optimization/workspace_boundary_costs.py','analysis/workspace_source.py',
         'codegen/stateful.py','codegen/stateful_v1.cpp.in','codegen/loop_v1.cpp.in',
         'codegen/stateful_workspace_v1.cpp.in','codegen/stateful_abi_v1.h','codegen/scar_abi_v1.h')}


def _spec(reference, *, qualified_time=None):
    from scar.compiler.project_workspace_demands import read_workspace_demands
    from scar.codegen.workspace_boundary_probe import render_workspace_boundary_probe
    if qualified_time is None:
        time = read_workspace_demands(reference['uri'])
    else:
        # Private bridge used only after the calling producer's complete
        # resource/time source replay. Never replace it by a supplied boolean.
        time = qualified_time
    _need(read_artifact(reference) == time
        and time.get('schema') == 'scar.workspace-demand-time-preparation',
        'exact source-refined workspace time reference required')
    schedule = schedule_from_dict(read_artifact(time['schedule']))
    proof = next(row for row in schedule.temporal.source_refs
        if row.get('schema') == 'scar.workspace-cpu-demand-source-proof')
    metadata = read_artifact(time['native_manifest']); plan = read_artifact(proof['stage_plan'])
    base = read_artifact(metadata['base_native_reference'])
    base_directory = Path(metadata['base_native_reference']['uri']).parent
    region = read_artifact(artifact_reference(base_directory/base['files']['region'],'scar.stateful-array-region'))
    cpp = proof['cpp_source']
    source, provenance = render_workspace_boundary_probe(metadata, plan,
        native_reference=time['native_manifest'], cpp_reference=cpp)
    count = metadata['trip_count']; layout = cut_layout(count)
    _need(count <= 10000 and provenance['cut_count'] == layout['timestamps_per_sample']
        and plan['total_region_array_payload_bytes'] <= plan['memory_budget_bytes'],
        'bounded canonical loop and original complete payload budget required')
    context = dict(target=time['target'], selected_cpu_ids=list(metadata['cpu_ids']),
        step_plan=proof['stage_plan'], step_plan_sha256=semantic_sha256(plan),
        input_shapes=plan['input_shapes'], base_implementation_sha256=proof['source_contract']['base_implementation_sha256'],
        compiler=metadata['compiler'])
    return dict(time=time, schedule=schedule, plan=plan, metadata=metadata, source=source,
        provenance=provenance, cpp=cpp, count=count, layout=layout, context=context,
        array_shapes=[plan['input_shapes'][row['id']] for row in region['inputs'] if row['kind'] == 'array'])


def _build(spec, directory):
    directory.mkdir(); source = directory/'probe.cpp'; source.write_text(spec['source'])
    headers = {}
    for name in ('stateful_abi_v1.h','scar_abi_v1.h'):
        dest = directory/name; dest.write_bytes((_ROOT/'codegen'/name).read_bytes()); headers[name] = _receipt(dest)
    compiler = spec['metadata']['compiler']
    identity = subprocess.run([*compiler['argv'],'--version'],capture_output=True,text=True,check=True).stdout.splitlines()[0]
    _need(identity == compiler['identity'] and '-fopenmp' not in compiler['options'],
        'original serial C++ compiler/options required')
    library = directory/'workspace_boundary.so'
    command = [*compiler['argv'],*compiler['options'],str(source),'-o',str(library),'-lm']
    process = subprocess.run(command,capture_output=True,text=True)
    (directory/'build.log').write_text(process.stdout+process.stderr)
    _need(process.returncode == 0, 'canonical workspace boundary compile failed: '+process.stderr[-2500:])
    return dict(source=_receipt(source),library=_receipt(library),headers=headers,
        compiler=compiler,compiler_command=command,source_provenance=spec['provenance'],
        implementation_files=_pins(),target_dag_compiled=False,target_program_executed=False,
        measurement_boundary='CANONICAL_NATIVE_WORKSPACE_BOUNDARY_FRAGMENTS')


def calibrate_workspace_boundary(time, out, *, repetitions=7, warmup=2):
    import numpy as np
    from scar.runtime.native_loop import _Array, _descriptor
    from scar.runtime.native_stateful import _Scalar, _Provider
    from scar.codegen.workspace_boundary_probe import SYMBOL
    _need(type(repetitions) is int and 3 <= repetitions <= 10000
        and type(warmup) is int and 1 <= warmup <= 10000, 'bounded warm repetitions required')
    output = Path(out).resolve(); _need(not output.exists(), 'fresh boundary calibration output required')
    reference = artifact_reference(Path(time).resolve(),'scar.workspace-demand-time-preparation')
    spec = _spec(reference); output.mkdir(parents=True)
    probe = _build(spec, output/'native')
    handle = ctypes.CDLL(probe['library']['path']); run = getattr(handle,SYMBOL)
    run.argtypes = [ctypes.POINTER(_Array),ctypes.c_uint64,ctypes.POINTER(_Scalar),ctypes.c_uint64,
        ctypes.POINTER(_Provider),ctypes.c_uint64,ctypes.c_uint64,ctypes.POINTER(ctypes.c_uint64),ctypes.c_uint64]
    run.restype = ctypes.c_int32
    arrays = [np.ones(shape,dtype=np.float64) for shape in spec['array_shapes']]
    packed = (_Array*len(arrays))(*(_descriptor(array,np) for array in arrays))
    counts = spec['provenance']['abi_counts']
    _need(counts['arrays'] == len(arrays), 'canonical descriptor count differs')
    scalars = (_Scalar*counts['scalars'])(*(_Scalar(1.0,0) for _ in range(counts['scalars'])))
    # Prefixes only test these handles for non-null. Their function is never
    # called: this observer contains no provider stage or target DAG.
    dummy = ctypes.c_uint64(1); address = ctypes.addressof(dummy)
    providers = (_Provider*counts['providers'])(*(_Provider(address,address,None) for _ in range(counts['providers'])))
    cuts = (ctypes.c_uint64*spec['layout']['timestamps_per_sample'])()
    libc = ctypes.CDLL(None); libc.fegetround.restype = ctypes.c_int
    affinity = sorted(os.sched_getaffinity(0)); rounding = libc.fegetround(); samples = []
    _need(set(spec['metadata']['cpu_ids']) <= set(affinity), 'selected source CPU unavailable to this probe')
    try:
        for index in range(warmup+repetitions):
            status = int(run(packed,len(arrays),scalars,len(scalars),providers,len(providers),spec['count'],cuts,len(cuts)))
            row = dict(index=index-warmup,timestamps_ns=list(cuts),status=status,
                caller_affinity_restored=sorted(os.sched_getaffinity(0)) == affinity,
                caller_rounding_restored=libc.fegetround() == rounding,
                input_arrays_unchanged=all(bool(np.all(array == 1.0)) for array in arrays))
            _need(status == 0 and all(row[key] is True for key in
                ('caller_affinity_restored','caller_rounding_restored','input_arrays_unchanged')),
                'canonical boundary input or caller-state validation failed')
            if index >= warmup:
                samples.append(row)
    finally:
        os.sched_setaffinity(0,affinity); libc.fesetround(rounding)
    context = dict(spec['context'],original_caller_cpu_ids=affinity)
    document = dict(schema=SCHEMA,schema_version=1,source_time=reference,
        workspace_plan=spec['time']['workspace_plan'],native_manifest=spec['time']['native_manifest'],
        cpp_source=spec['cpp'],trip_count=spec['count'],resource_context=context,
        repetitions=repetitions,warmup=warmup,cut_layout=spec['layout'],phase_definitions=list(PHASE_DEFINITIONS),
        clock=dict(api='clock_gettime',clock_id='CLOCK_MONOTONIC',unit='ns'),samples=samples,
        conditions=list(CONDITIONS),missing_terms=list(MISSING),probe=probe,
        target_workload_executed=False,target_variants_ranked=False,target_variants_compiled=False,
        whole_target_call_measured=False,clock_overhead_subtracted=False,
        physical_optimality_proven=False,deployment_authorized=False)
    document['statistics'] = boundary_statistics(document)
    path = output/'observations.json'; path.write_bytes(_json(document))
    result = dict(schema='scar.workspace-boundary-result',schema_version=1,
        observations=artifact_reference(path,SCHEMA),phases=document['statistics']['phases'],
        target_workload_executed=False,target_dag_compiled=False,deployment_authorized=False)
    (output/'result.json').write_bytes(_json(result)); return result


def _read_workspace_boundary(reference, *, qualified_time=None):
    document = read_artifact(reference)
    spec = _spec(document['source_time'],qualified_time=qualified_time)
    probe = document['probe']; context = dict(spec['context'],
        original_caller_cpu_ids=document['resource_context']['original_caller_cpu_ids'])
    _need(document['resource_context'] == context and document['workspace_plan'] == spec['time']['workspace_plan']
        and document['native_manifest'] == spec['time']['native_manifest'] and document['cpp_source'] == spec['cpp']
        and document['trip_count'] == spec['count'] and document['conditions'] == list(CONDITIONS)
        and document['missing_terms'] == list(MISSING), 'canonical workspace boundary source/context differs')
    _need(probe['implementation_files'] == _pins() and probe['source_provenance'] == spec['provenance']
        and probe['compiler'] == spec['metadata']['compiler'] and probe['target_dag_compiled'] is False
        and probe['target_program_executed'] is False, 'canonical workspace observer implementation differs')
    for receipt in (probe['source'],probe['library'],*probe['headers'].values()):
        _need(_receipt(receipt['path']) == receipt, 'retained workspace observer bytes changed')
    _need(Path(probe['source']['path']).read_text() == spec['source']
        and set(probe['headers']) == {'stateful_abi_v1.h','scar_abi_v1.h'}
        and all(Path(row['path']).read_bytes() == (_ROOT/'codegen'/name).read_bytes()
            for name,row in probe['headers'].items()), 'source/header reconstruction differs')
    compiler = probe['compiler']
    _need(probe['compiler_command'] == [*compiler['argv'],*compiler['options'],probe['source']['path'],
        '-o',probe['library']['path'],'-lm'], 'workspace observer compile command differs')
    _need(document['statistics'] == boundary_statistics(document)
        and all(document.get(key) is False for key in ('target_workload_executed','target_variants_ranked',
            'target_variants_compiled','whole_target_call_measured','clock_overhead_subtracted',
            'physical_optimality_proven','deployment_authorized')), 'canonical observation scope/statistics differ')
    read_artifact(reference); return document


def read_workspace_boundary(reference):
    if isinstance(reference,(str,Path)):
        reference = artifact_reference(Path(reference).resolve(),SCHEMA)
    return _read_workspace_boundary(reference)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--time-schedule'); source.add_argument('--check')
    parser.add_argument('--out'); parser.add_argument('--repetitions',type=int,default=7)
    parser.add_argument('--warmup',type=int,default=2); args = parser.parse_args(argv)
    if args.check:
        _need(args.out is None, 'check only reads a retained calibration')
        document = read_workspace_boundary(args.check)
        print(json.dumps(dict(phases=len(document['statistics']['phases']),source_replayed=True,
            target_workload_executed=False,deployment_authorized=False)))
    else:
        _need(args.out is not None, 'calibration output required')
        result = calibrate_workspace_boundary(args.time_schedule,args.out,repetitions=args.repetitions,warmup=args.warmup)
        print(json.dumps(dict(observations=result['observations'],target_workload_executed=False,
            target_dag_compiled=False,deployment_authorized=False),sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
