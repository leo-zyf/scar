"""Collect bounded actual regional host cuts; no whole-program timing claim."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
import time

from scar.ir.execution_recipe import artifact_reference

SCHEMA='scar.pipeline-host-sample-observations'


def collect(native,out,*,warmup=2,repetitions=7):
    from scar.codegen.pipeline_host_probe import build,SCHEMA as PROBE
    import numpy as np
    if type(warmup)is not int or type(repetitions)is not int or not 1<=warmup<=1000 or not 3<=repetitions<=1000:
        raise ValueError('bounded warm repeated observations required')
    output=Path(out).resolve()
    if output.exists():raise ValueError('fresh host observations directory required')
    output.mkdir(parents=True)
    started=time.monotonic();probe=build(native,output/'probe');artifact=probe.artifact
    memory=artifact.memory_plan;certificate=artifact.execution_schedule
    rng=np.random.default_rng(47)
    arrays=[rng.uniform(.5,1.,size=memory['input_shapes'][name])for name in artifact.pipeline.parameters]
    original_namespace=runpy.run_path(artifact.pipeline.source.path,run_name='scar_host_probe_reference')
    original=original_namespace
    for part in artifact.pipeline.source.qualname.split('.'):
        original=original[part]if type(original)is dict else getattr(original,part)
    expected=original(*arrays)
    expected_digest=sha256(expected.tobytes()).hexdigest()
    clean=artifact.run(*arrays)
    if not np.array_equal(expected.view(np.uint64),clean.view(np.uint64)):
        raise ValueError('source and original clean native differ before instrumentation')
    before=[sha256(a.tobytes()).hexdigest()for a in arrays]
    affinity=os.sched_getaffinity(0);samples=[]
    for i in range(warmup+repetitions):
        result=probe.run(*arrays)
        if result['status']!=0 or not result['trace']['normal_complete']:
            raise ValueError('instrumented regional call failed: '+str(result['status']))
        actual=result['output']
        if not np.array_equal(expected.view(np.uint64),actual.view(np.uint64)):
            raise ValueError('instrumented original operator output differs from source')
        if i>=warmup:samples.append(dict(id='sample:'+str(i-warmup),status=result['status'],
            output_sha256=sha256(actual.tobytes()).hexdigest(),trace=result['trace']))
    if before!=[sha256(a.tobytes()).hexdigest()for a in arrays]or os.sched_getaffinity(0)!=affinity:
        raise ValueError('host observations changed source inputs or caller CPU affinity')
    metadata_ref=artifact_reference(output/'probe/probe.json',PROBE)
    document=dict(schema=SCHEMA,schema_version=1,probe=metadata_ref,
        source_refs=[dict(artifact=probe.metadata['original_artifact'],role='ORIGINAL_CPP_DELIVERY'),
            dict(artifact=probe.metadata['source'],role='MECHANICALLY_INSTRUMENTED_HOST_SOURCE')],
        call_predecessors=certificate['call_predecessors'],lane_call_order=certificate['lane_call_order'],
        expected_cpu_ids=certificate['cpu_ids'],warmup=warmup,repetitions=repetitions,
        expected_output_sha256=expected_digest,samples=samples,
        input_metadata=[dict(shape=list(a.shape),dtype=str(a.dtype),layout='C_CONTIGUOUS')for a in arrays],
        original_source_path=artifact.pipeline.source.path,original_source_sha256=artifact.pipeline.source.sha256,
        input_values_retained=False,output_values_retained=False,original_inputs_unchanged=True,
        target_region_executed=True,full_target_program_executed=False,clean_native_matches_original=True,
        instrumented_outputs_match_original=True,affinity_restored=True,
        original_caller_affinity=sorted(affinity),elapsed_seconds=time.monotonic()-started,
        conditions=['ONE_ADMITTED_SELECTED_CPP_REGION_REPEATED_WITH_STABLE_READONLY_INPUTS',
            'NATIVE_CALLER_LANES_CALLS_AND_POOL_GEOMETRY_PINNED_BY_ORIGINAL_DELIVERY',
            'TIMESTAMP_STORAGE_PREALLOCATED_DIFFERENT_WORKERS_WRITE_DISJOINT_SLOTS',
            'HOST_PYTHON_GUARDS_FINAL_NUMPY_ALLOCATION_AND_CTYPES_OUTSIDE_NATIVE_SCOPE'],
        residuals=['CLOCK_AND_TRACE_STORES_NOT_SUBTRACTED','INSTRUMENTATION_COMPILER_AND_SCHEDULER_PERTURBATION',
            'SAMPLING_REGIME_IS_NOT_A_PHYSICAL_LATENCY_GUARANTEE','WAITING_IS_NOT_INTRINSIC_CPU_SERVICE'],
        whole_program_performance_measured=False,deployment_authorized=False)
    path=output/'observations.json';path.write_text(json.dumps(document,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return document


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--warmup',type=int,default=2);parser.add_argument('--repetitions',type=int,default=7)
    args=parser.parse_args(argv)
    result=collect(args.native,args.out,warmup=args.warmup,repetitions=args.repetitions)
    print(json.dumps(dict(observations=str(Path(args.out).resolve()/'observations.json'),
        samples=len(result['samples']),source_and_instrumented_outputs_equal=True,elapsed_seconds=result['elapsed_seconds'])));return 0


if __name__=='__main__':raise SystemExit(main())
