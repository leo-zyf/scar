"""Calibrate source-qualified ABI1 services, never complete target strategies.

Each leaf is a topology-specialized component probe. Input/output arrays are
preallocated, timing occurs in C++, and the worker starts on its chosen CPU.
Host startup, allocation, scheduling, joins and concurrent contention are NOT
measured by this experiment and remain explicit applicability conditions.
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import ctypes
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from scar.ir.execution_recipe import artifact_reference
from scar.optimization.implementation_costs import CostQuery, query_key, resolve_cost, SCHEMA
from scar.optimization.pipeline_implementation import SCOPE

_PROBE = r'''
#include "scar_abi_v1.h"
#include <time.h>
extern "C" SCAR_EXPORT_V1 int scar_time_service_v1(
    const scar_operator_table_v1 *table, const scar_array_v1 *inputs,
    uint64_t count, scar_array_v1 *output, uint64_t repeat, uint64_t *elapsed) {
    if (!table || !table->run || !elapsed || !repeat) return -1;
    timespec before{},after{};
    if (clock_gettime(CLOCK_MONOTONIC,&before)) return -2;
    for (uint64_t i=0;i<repeat;++i) {
        const auto status=table->run(inputs,count,output);
        if (status) return status;
    }
    if (clock_gettime(CLOCK_MONOTONIC,&after)) return -3;
    *elapsed=uint64_t((after.tv_sec-before.tv_sec)*1000000000LL+after.tv_nsec-before.tv_nsec);
    return 0;
}
'''


def _reference_value(region,arrays):
    """Evaluate only the already-admitted arithmetic profile for probe checking."""
    values={};inputs=dict(zip(region.parameters,arrays))
    for node in region.nodes:
        arguments=[values[k]for k in node.inputs];op=node.op
        if op=='input':value=inputs[node.attrs['name']]
        elif op=='const':value=node.attrs['value']
        elif op=='slice':value=arguments[0][tuple(slice(a,b)for a,b in node.attrs['bounds'])]
        elif op=='copy':value=arguments[0].copy(order='C')
        elif op=='neg':value=-arguments[0]
        elif op=='add':value=arguments[0]+arguments[1]
        elif op=='sub':value=arguments[0]-arguments[1]
        elif op=='mul':value=arguments[0]*arguments[1]
        elif op=='div':value=arguments[0]/arguments[1]
        else:raise ValueError('probe arithmetic not in original ABI1 profile')
        values[node.id]=value
    return values[region.output]


def calibrate(preparation,out,*,repetitions=7,warmup=2,batch=64):
    from scar.compiler.compile_pipeline_implementation import open_pipeline_implementation,SCHEMA as PREPARATION
    from scar.optimization.hardware import propose_cpu_placements,bind_cpu_placement
    from scar.optimization.schedule import build_region_schedule
    from scar.compiler.model import document as leaf_document
    from scar.optimization.resource_solver import semantic_sha256
    from scar.codegen import cpp
    import numpy as np
    if type(repetitions)is not int or repetitions<3 or repetitions%2!=1 or type(warmup)is not int or warmup<1 or type(batch)is not int or not 1<=batch<=10000:
        raise ValueError('odd repetitions >=3, positive warmup and bounded positive batch required')
    directory=Path(out).resolve()
    if directory.exists():raise ValueError('fresh service calibration directory required')
    reference=artifact_reference(Path(preparation).resolve(),PREPARATION)
    started=time.monotonic();directory.mkdir(parents=True)
    header=Path(cpp.__file__).parent
    source=directory/'probe.cpp';source.write_text(_PROBE)
    command=cpp._compiler_command() if hasattr(cpp,'_compiler_command')else None
    if command is None:
        import shlex
        command=shlex.split(os.environ.get('CXX','g++'))
    argv=[*command,*cpp.COMPILER_OPTIONS,'-I',str(header),str(source),'-o',str(directory/'probe.so')]
    subprocess.run(argv,check=True,capture_output=True,text=True)
    library=ctypes.CDLL(str(directory/'probe.so'));probe=library.scar_time_service_v1
    probe.argtypes=[ctypes.POINTER(cpp._Table),ctypes.POINTER(cpp._Array),ctypes.c_uint64,
                   ctypes.POINTER(cpp._Array),ctypes.c_uint64,ctypes.POINTER(ctypes.c_uint64)]
    probe.restype=ctypes.c_int
    entries={};notes=[];original_affinity=os.sched_getaffinity(0)
    with open_pipeline_implementation(reference)as(_,pipeline,memory,target,problem,_,_):
        calls={call.id:call for call in pipeline.calls}
        for operation in problem.operations:
            call=calls[operation.id]
            for mode in operation.options:
                context=mode.parameters['context'];cpu=mode.parameters['cpu_id']
                placements=propose_cpu_placements(call.plan,target,affinity=[cpu],
                    memory_budget_bytes=memory['memory_budget_bytes'],max_threads=1)
                binding=next(p for p in placements if p.cpu_ids==(cpu,))
                plan=bind_cpu_placement(call.plan,binding)
                if semantic_sha256(leaf_document(plan))!=mode.parameters['bound_leaf_plan_sha256']:
                    raise ValueError('canonical service source/CPU differs from exact cost query')
                operator=cpp.build(plan,directory/'components'/str(len(entries)),resource_schedule=build_region_schedule(plan,target))
                rng=np.random.default_rng(31)
                arrays=[rng.uniform(.5,1.,size=record['shape'])for record in context['inputs']]
                expected=np.asarray(_reference_value(call.plan.region,arrays),dtype=np.float64)
                output=np.empty(context['output']['shape'],dtype=np.float64)
                descriptors=(cpp._Array*len(arrays))(*(cpp._descriptor(a)for a in arrays))
                out_descriptor=cpp._descriptor(output);table=operator._library.scar_get_operator_table_v1()
                elapsed=ctypes.c_uint64();samples=[];raw=[]
                try:
                    os.sched_setaffinity(0,{cpu})
                    for index in range(warmup+repetitions):
                        status=probe(table,descriptors,len(arrays),ctypes.byref(out_descriptor),batch,ctypes.byref(elapsed))
                        if status:raise ValueError('canonical ABI1 service rejected inputs: '+str(status))
                        if not np.array_equal(expected.view(np.uint64),output.view(np.uint64)):
                            raise ValueError('canonical component differs from original ordered arithmetic')
                        if index>=warmup:raw.append(elapsed.value);samples.append(elapsed.value//batch)
                finally:os.sched_setaffinity(0,original_affinity)
                query=CostQuery(SCOPE,context)
                key=query_key(query)
                entries[key]=dict(query=dict(scope=SCOPE,context=context),basis='EMPIRICAL_MODEL',
                    estimate_ns=int(statistics.median(samples)),lower_ns=min(samples),upper_ns=max(samples),missing_terms=[],
                    observations=dict(samples_ns=samples,warmup=warmup,repetitions=repetitions,
                        batch=batch,raw_batch_ns=raw,quantization='INTEGER_FLOOR_BATCH_NS_DIVIDED_BY_BATCH_ERROR_LT_1NS'),
                    conditions=['ISOLATED_ABI1_SERVICE_WITH_PREALLOCATED_INPUT_OUTPUT',
                        'CALLING_THREAD_STARTS_ON_EXACT_SINGLETON_CPU',
                        'REPEATED_INPUT_WARM_DATA_REGIME_NOT_A_COLD_OR_CONCURRENT_GUARANTEE',
                        'HOST_ENVELOPE_AND_SHARED_MEMORY_BANDWIDTH_INTERFERENCE_NOT_INCLUDED'],
                    native=artifact_reference(operator.directory/'artifact.json',operator.report['schema']),
                    output_sha256=sha256(output.tobytes()).hexdigest())
                notes.append(dict(operation=operation.id,option=mode.id,median_ns=entries[key]['estimate_ns']))
    profile=dict(schema=SCHEMA,schema_version=1,entries=entries,preparation=reference,
        experiment=dict(kind='SOURCE_QUALIFIED_TOPOLOGY_SPECIALIZED_COMPONENT_PROBES',clock='CLOCK_MONOTONIC_INSIDE_CPP',
            target_program_executed=False,whole_pipeline_variants_executed=False,
            scalar_dag_fragments_compiled=True,python_dispatch_inside_timed_region=False,
            input_output_allocated_outside_timing=True,source_header_sha256=sha256((header/'scar_abi_v1.h').read_bytes()).hexdigest(),
            probe_source_sha256=sha256(source.read_bytes()).hexdigest(),probe_library_sha256=sha256((directory/'probe.so').read_bytes()).hexdigest(),
            command=argv,affinity_restored=os.sched_getaffinity(0)==original_affinity,
            elapsed_seconds=time.monotonic()-started,physical_optimality_proven=False))
    path=directory/'profile.json';path.write_text(json.dumps(profile,sort_keys=True,indent=2,allow_nan=False)+'\n')
    ref=artifact_reference(path,SCHEMA)
    for row in entries.values():resolve_cost(CostQuery(row['query']['scope'],row['query']['context']),ref)
    result=dict(schema='scar.pipeline-service-calibration',schema_version=1,profile=ref,components=notes,
                source_expression_outputs_bitwise_equal=True,**profile['experiment'])
    (directory/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--repetitions',type=int,default=7);parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--batch',type=int,default=64);args=parser.parse_args(argv)
    result=calibrate(args.preparation,args.out,repetitions=args.repetitions,warmup=args.warmup,batch=args.batch)
    print(json.dumps(dict(profile=result['profile'],components=len(result['components']),elapsed_seconds=result['elapsed_seconds'])));return 0


if __name__=='__main__':raise SystemExit(main())
