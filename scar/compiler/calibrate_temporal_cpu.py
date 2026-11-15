"""Independent fixed-family temporal CPU resource models, not target trials.

The probe accepts hardware, masks, shape/count and a frozen canonical registry.
Target proofs and DAGs are accepted only later by fact projection, which must
prove the ordered anonymous DAG matches a measured registry family exactly.
All canonical boundary/component estimates remain DECLARED_COMPONENT_MODEL.
"""
import argparse
import ctypes
from dataclasses import replace
from hashlib import sha256
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.hardware import _target_materials, propose_cpu_allocations, validate_cpu_placement
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.temporal_cpu_choices import (temporal_primitive_families,
    match_temporal_primitive_family, temporal_resource_context, temporal_cpu_cost_demand)

_CODEGEN=Path(__file__).resolve().parents[1]/'codegen'
_ASSUMPTIONS=[
    'DECLARED_CANONICAL_COMPONENT_MODEL_NOT_ACTUAL_TARGET_CALL_TIMING',
    'COMPONENT_SUM_SERIAL_PLUS_MAX_WORKER_IS_A_MODEL_NOT_AN_ADDITIVITY_THEOREM',
    'PROBE_RUNTIME_MASK_AND_FAMILY_DISPATCH_DIFFER_FROM_TARGET_STATIC_SPECIALIZATION',
    'CANONICAL_CALLABLE_METADATA_DIFFERS_FROM_TARGET_NAMES_DEFAULTS_AND_CODE_IDENTITIES',
    'CANONICAL_FINITE_CONSTANT_INPUTS_DIFFER_FROM_TARGET_VALUE_AND_ALIAS_DISTRIBUTION',
    'WARM_REPEATED_BUFFERS_DO_NOT_PROVE_TARGET_CACHE_OR_NUMA_PAGE_RESIDENCY',
    'HOST_CONTENTION_FREQUENCY_AND_OS_SCHEDULING_NOT_ISOLATED',
    'NATIVE_TEAM_TIMER_INCLUDES_EMPTY_TEAM_SETUP_FENV_AFFINITY_RESTORATION_AND_JOIN',
    'WORKER_CHUNK_TIMERS_EXCLUDE_TEAM_SETUP_AND_JOIN_NO_DOUBLE_CHARGE',
    'LOGICAL_SHAPE_OR_CORE_COUNT_IS_NOT_DRAM_TRAFFIC_OR_THROUGHPUT']


def _save(path,value):
    path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')


def _receipt(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return dict(path=str(path),sha256=sha256(raw).hexdigest(),bytes=len(raw))


def _summary(samples):
    return dict(samples_ns=samples,median_ns=statistics.median(samples),min_ns=min(samples),max_ns=max(samples),
        stdev_ns=statistics.stdev(samples),sample_count=len(samples))


def _samples(function,repetitions,warmup):
    for _ in range(warmup):result=function();del result
    values=[]
    for _ in range(repetitions):
        start=time.perf_counter_ns();result=function();elapsed=time.perf_counter_ns()-start
        values.append(elapsed);del result
    return _summary(values)


def _canonical_loop(carry,readonly,count):
    raise RuntimeError('canonical boundary callable metadata must never execute')


def _canonical_leaf(carry,readonly):
    raise RuntimeError('canonical leaf metadata must never execute')


def _native(backend,cpu_slots,out):
    from scar.codegen import cpp
    header=(_CODEGEN/'kernel_v1.cpp.in').read_text()
    affinity=header[header.index('class ThreadAffinity {'):header.index('\n#endif',header.index('class ThreadAffinity {'))]
    helpers=header[header.index('struct Shape {'):header.index('\nint32_t run(')]
    functions=[]
    for ordinal,name in enumerate(temporal_primitive_families()):
        recurrence=('carry=carry+invariant;' if ordinal==0 else
            'const double scaled=carry*0x1.0000000000000p-1; carry=scaled+invariant;')
        functions.append(f'''__attribute__((noinline)) void family_{ordinal}(const Args &a,int64_t begin,int64_t end,
    uint32_t &nonfinite,uint64_t &preheaders,uint64_t &iterations) {{
    for(int64_t row=begin;row<end;++row)for(int64_t col=0;col<a.output->shape[1];++col){{
        double carry=read_value(static_cast<const unsigned char *>(a.inputs[0].data)+row*a.inputs[0].strides[0]+col*a.inputs[0].strides[1],nonfinite);
        const double fixed=read_value(static_cast<const unsigned char *>(a.inputs[1].data)+row*a.inputs[1].strides[0]+col*a.inputs[1].strides[1],nonfinite);
        const double invariant=fixed*fixed;++preheaders;
        if(nonfinite)continue;
        #pragma GCC unroll 1
        for(uint64_t i=0;i<a.steps;++i){{ {recurrence} ++iterations;
            if(!finite_bits(carry)){{nonfinite|=1;break;}} }}
        static_cast<double *>(a.output->data)[row*a.output->shape[1]+col]=carry;
    }}
}}''')
    template=_CODEGEN/'temporal_resource_probe_v1.cpp.in';raw=template.read_text()
    for key,value in dict(CPU_SLOTS=str(cpu_slots),AFFINITY_CLASS=affinity,ARRAY_HELPERS=helpers,
            FAMILY_FUNCTIONS='\n'.join(functions),FAMILY_POINTERS=', '.join('&family_'+str(i) for i in range(len(functions)))).items():
        raw=raw.replace('@'+key+'@',value)
    if '@' in raw:raise ValueError('unresolved independent probe template')
    out.mkdir();(out/'probe.cpp').write_text(raw);(out/'scar_abi_v1.h').write_bytes((_CODEGEN/'scar_abi_v1.h').read_bytes())
    compiler=backend['compiler'];identity=subprocess.run([*compiler['argv'],'--version'],check=True,capture_output=True,text=True).stdout.splitlines()[0]
    if identity!=compiler['identity']:raise ValueError('canonical probe compiler differs from backend context')
    library=out/'temporal_resource_probe.so';remarks=out/'vectorization.txt'
    command=[*compiler['argv'],*compiler['options'],'-fopt-info-vec-optimized='+str(remarks),str(out/'probe.cpp'),'-o',str(library)]
    built=subprocess.run(command,capture_output=True,text=True);(out/'compile.log').write_text(built.stdout+built.stderr)
    if built.returncode:raise RuntimeError('independent temporal resource compilation failed:\n'+built.stderr)
    assembly=subprocess.run(['objdump','-d','-C',str(library)],check=True,capture_output=True,text=True).stdout
    (out/'assembly.txt').write_text(assembly)
    handle=ctypes.CDLL(str(library))
    from scar.runtime.native_loop import _Array
    boundary=handle.scar_canonical_boundary_v1
    boundary.argtypes=[ctypes.POINTER(_Array),ctypes.c_uint64,ctypes.c_uint64,ctypes.POINTER(_Array),ctypes.POINTER(ctypes.c_uint64)]
    boundary.restype=ctypes.c_int32
    team=handle.scar_canonical_team_samples_v1
    team.argtypes=[ctypes.POINTER(ctypes.c_int32),ctypes.c_int32,ctypes.c_uint64,ctypes.c_uint64,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_int32)]
    team.restype=ctypes.c_int32
    chunks=handle.scar_canonical_chunk_samples_v1
    chunks.argtypes=[ctypes.POINTER(ctypes.c_int32),ctypes.c_int32,ctypes.c_int32,ctypes.POINTER(_Array),ctypes.POINTER(_Array),ctypes.c_uint64,
        ctypes.c_uint64,ctypes.c_uint64,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_int32)]
    chunks.restype=ctypes.c_int32
    return handle,dict(boundary=boundary,team=team,chunks=chunks),dict(compiler=compiler,command=command,
        source=_receipt(out/'probe.cpp'),library=_receipt(library),header=_receipt(out/'scar_abi_v1.h'),
        assembly=_receipt(out/'assembly.txt'),vectorization_remarks=_receipt(remarks),
        diagnostic_compiler_option='-fopt-info-vec-optimized',target_source_or_dag_compiled=False)


def calibrate_temporal_cpu(target,cpu_masks,input_shape,trip_count,out,*,repetitions=7,warmup=2):
    """Compile one fixed registry probe and measure the explicitly supplied masks."""
    import numpy as np
    from scar.compiler.parallel_temporal_invariant import describe_parallel_temporal_capability
    from scar.runtime.native_loop import _Array,_descriptor,_cpu_available
    from scar.runtime.loop_guard import _state,_same_state,_range
    out=Path(out).resolve();shape=list(input_shape)
    if out.exists():raise ValueError('fresh independent calibration directory required')
    if (len(shape)!=2 or any(type(n) is not int or n<=0 for n in shape) or shape[0]*shape[1]>4*1024*1024
            or type(trip_count) is not int or not 0<trip_count<2**63
            or type(repetitions) is not int or repetitions<3 or type(warmup) is not int or warmup<1):
        raise ValueError('bounded positive shape/count, at least three samples and one warmup required')
    if sys.gettrace() is not None or sys.getprofile() is not None:raise ValueError('independent calibration cannot run under a Python trace/profile hook')
    graph,snapshot,resources,_=_target_materials(target);backend=describe_parallel_temporal_capability()
    caller=sorted(os.sched_getaffinity(0));environment={name:os.environ.get(name) for name in
        ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}
    masks=[tuple(sorted(mask)) for mask in cpu_masks]
    if not masks or len(set(masks))!=len(masks) or any(not mask or len(set(mask))!=len(mask)
            or any(type(cpu) is not int or cpu<0 for cpu in mask) for mask in masks):raise ValueError('distinct explicit nonempty CPU masks required')
    seeds=propose_cpu_allocations(graph,memory_budget_bytes=shape[0]*shape[1]*8,max_threads=None);contexts=[]
    for mask in masks:
        matches=[seed for seed in seeds if set(mask)<=set(seed.cpu_ids)]
        if not matches:raise ValueError('canonical mask needs one retained executor and host-memory access')
        seed=min(matches,key=lambda row:(row.executor_id,-len(row.cpu_ids)));binding=replace(seed,cpu_ids=mask)
        validate_cpu_placement(binding)
        contexts.append(temporal_resource_context(snapshot.sha256,binding.executor_id,mask,caller,environment,backend))
    registry=temporal_primitive_families();out.mkdir(parents=True);started=time.time()
    design=dict(schema='scar.temporal-cpu-primitive-design',schema_version=1,target_snapshot=encode_snapshot(snapshot),
        cpu_masks=masks,shape=shape,trip_count=trip_count,family_registry=registry,
        resource_contexts=contexts,repetitions=repetitions,warmup=warmup,
        implementation=_receipt(__file__),template=_receipt(_CODEGEN/'temporal_resource_probe_v1.cpp.in'),
        native_compilations=1,target_workload_executed=False,target_variants_ranked=False,
        target_body_or_domain_accepted_by_probe=False,assumptions=_ASSUMPTIONS)
    _save(out/'design.json',design);design_ref=artifact_reference(out/'design.json',design['schema'])
    handle,functions,native=_native(backend,max(caller)+1,out/'native');compile_seconds=time.time()-started
    observations=[];load_before=list(os.getloadavg())
    carry=np.full(shape,1.25,dtype=np.float64);readonly=np.full(shape,.125,dtype=np.float64)
    original_state=_state(_canonical_loop);leaf_state=_state(_canonical_leaf)
    signature=inspect.signature(_canonical_loop,follow_wrapped=False)
    for mask,context in zip(masks,contexts):
        cpus=(ctypes.c_int32*len(mask))(*mask);placements=(ctypes.c_int32*(4*len(mask)))()
        native_samples=(ctypes.c_double*repetitions)()
        status=functions['team'](cpus,len(mask),repetitions,warmup,native_samples,placements)
        if status:raise ValueError('canonical empty-team FP/affinity guard failed: '+str(status))
        team=_summary(list(native_samples));team_placements=_placements(mask,placements)
        output=np.empty(shape,dtype=np.float64);descriptors=(_Array*2)(_descriptor(carry,np),_descriptor(readonly,np))
        output_descriptor=_descriptor(output,np);counters=(ctypes.c_uint64*2)()
        def boundary():
            if threading.active_count()!=1 or sys.gettrace() is not None or sys.getprofile() is not None:
                raise ValueError('canonical synchronous boundary scope changed')
            monitoring=getattr(sys,'monitoring',None)
            if monitoring is not None and any(monitoring.get_tool(i) is not None for i in range(6)):raise ValueError('canonical monitoring observer active')
            if not _same_state(_canonical_loop,original_state) or not _same_state(_canonical_leaf,leaf_state) or _range(_canonical_loop) is not range:
                raise ValueError('canonical callable metadata changed')
            bound=signature.bind(carry,readonly,trip_count);bound.apply_defaults()
            if type(bound.arguments['count']) is not int or bound.arguments['count']!=trip_count:raise ValueError('canonical count changed')
            _cpu_available(list(mask));values=[bound.arguments['carry'],bound.arguments['readonly']]
            packed=[]
            for value in values:
                descriptor=_descriptor(value,np)
                if list(value.shape)!=shape or not value.flags.c_contiguous:raise ValueError('canonical layout changed')
                packed.append(descriptor)
            descriptor=_descriptor(output,np);counter=(ctypes.c_uint64*2)()
            result=functions['boundary']((_Array*2)(*packed),2,trip_count,ctypes.byref(descriptor),counter)
            if result:raise ValueError('canonical descriptor/FFI guard failed')
            return counter
        boundary_cost=_samples(boundary,repetitions,warmup)
        allocation=_samples(lambda:np.empty(shape,dtype=np.float64),repetitions,warmup)
        publication_report={};publication_stats=dict(native_calls=0)
        counters[0]=shape[0]*shape[1];counters[1]=counters[0]*trip_count
        def publication():
            publication_report['last_execution']=dict(native_calls=1,element_preheaders=counters[0],
                element_iterations=counters[1],logical_iterations=trip_count,status=0)
            publication_stats['native_calls']+=1
            return output
        publication_cost=_samples(publication,repetitions,warmup)
        for family_index,(family,body) in enumerate(registry.items()):
            samples=(ctypes.c_double*(repetitions*len(mask)))();checksum=ctypes.c_uint64()
            status=functions['chunks'](cpus,len(mask),family_index,descriptors,ctypes.byref(output_descriptor),trip_count,
                repetitions,warmup,samples,ctypes.byref(checksum),placements)
            if status:raise ValueError('canonical worker chunk FP/affinity guard failed: '+str(status))
            expected=1.25
            for _ in range(trip_count):expected=(expected if family_index==0 else expected*.5)+.125*.125
            if not np.all(output.view(np.uint64)==np.float64(expected).view(np.uint64)):
                raise ValueError('canonical primitive numerical self-check failed')
            chunks=[]
            for worker,cpu in enumerate(mask):
                begin=worker*(shape[0]//len(mask))+min(worker,shape[0]%len(mask));end=begin+shape[0]//len(mask)+(worker<shape[0]%len(mask))
                chunks.append(dict(worker=worker,cpu_id=cpu,row_begin=begin,row_end=end,
                    **_summary([samples[iteration*len(mask)+worker] for iteration in range(repetitions)])))
            observations.append(dict(family=family,family_match=match_temporal_primitive_family(body),
                resource_context=context,shape=shape,trip_count=trip_count,
                serial_components=dict(boundary=boundary_cost,allocation=allocation,team=team,publication=publication_cost),
                worker_chunks=chunks,team_placements=team_placements,chunk_placements=_placements(mask,placements),
                checksum=checksum.value,canonical_output_bitwise_checked=True,canonical_expected_scalar=expected))
            _save(out/'raw-observations.json',dict(schema='scar.temporal-canonical-resource-observations',schema_version=1,
                frozen_design=design_ref,native=native,observations=observations))
    if (sorted(os.sched_getaffinity(0))!=caller or any(os.environ.get(key)!=value for key,value in environment.items())
            or artifact_reference(out/'design.json',design['schema'])!=design_ref):raise ValueError('calibration caller or frozen design changed')
    report=dict(schema='scar.temporal-cpu-resource-calibration',schema_version=1,frozen_design=design_ref,
        raw_observations=artifact_reference(out/'raw-observations.json','scar.temporal-canonical-resource-observations'),
        target_snapshot=encode_snapshot(snapshot),backend=backend,resource_contexts=contexts,family_registry=registry,
        observations=observations,repetitions=repetitions,warmup=warmup,native=native,compile_seconds=compile_seconds,
        seconds=time.time()-started,host_load=dict(before=load_before,after=list(os.getloadavg()),isolated=False),
        caller_affinity_restored=True,numa_page_residency_proven=False,assumptions=_ASSUMPTIONS,
        basis='DECLARED_COMPONENT_MODEL',target_workload_executed=False,target_variants_ranked=False,
        target_body_or_domain_accepted_by_probe=False,source_code_ids_or_function_names_in_primitive_keys=False,
        complete_target_call_measured=False,physical_optimality_proven=False)
    _save(out/'result.json',report)
    return report


def encode_snapshot(snapshot):
    from scar.ir.record_codec import encode
    return encode(snapshot)


def _placements(mask,observations):
    rows=[dict(planned_cpu=cpu,actual_cpu=observations[i*4],bound=bool(observations[i*4+1]),
        fenv_restored=bool(observations[i*4+2]),affinity_restored=bool(observations[i*4+3])) for i,cpu in enumerate(mask)]
    if any(row['planned_cpu']!=row['actual_cpu'] or not row['bound'] or not row['fenv_restored'] or not row['affinity_restored'] for row in rows):
        raise ValueError('canonical worker placement/restoration observation differs')
    return rows


def _projection_materials(calibration):
    """Authenticate one immutable observation set once per projection batch."""
    reference=artifact_reference(calibration,'scar.temporal-cpu-resource-calibration');measured=read_artifact(reference)
    design=read_artifact(measured['frozen_design']);raw=read_artifact(measured['raw_observations'])
    if measured['family_registry']!=temporal_primitive_families() or design['family_registry']!=measured['family_registry']:
        raise ValueError('frozen independent family registry differs')
    if measured['target_workload_executed'] is not False or measured['target_body_or_domain_accepted_by_probe'] is not False:
        raise ValueError('source-independent primitive calibration required')
    if raw['observations']!=measured['observations'] or raw['frozen_design']!=measured['frozen_design']:
        raise ValueError('calibration summary differs from retained raw observation rows')
    index={}
    for ordinal,row in enumerate(measured['observations']):
        key=(row['family'],semantic_sha256(row['resource_context']),tuple(row['shape']),row['trip_count'])
        index.setdefault(key,[]).append((ordinal,row))
    return reference,measured,index


def _project_temporal_facts(domain,candidate_id,out,reference,measured,index):
    out=Path(out).resolve()
    if out.exists():raise ValueError('fresh projected temporal resource facts output required')
    demand=temporal_cpu_cost_demand(domain,candidate_id);components=[];unresolved=[];matching=[];family=None
    try:
        family=match_temporal_primitive_family(domain['body'])
        matching=index.get((family['family'],semantic_sha256(demand['resource_context']),
            tuple(domain['shape']),domain['trip_count']),[])
        if len(matching)!=1:raise ValueError('no unique independently measured family/mask/shape/count context')
        observation_index,row=matching[0]
        for term in demand['terms']:
            if term['id'].startswith('worker:'):
                worker=int(term['id'].split(':')[1]);measurement=row['worker_chunks'][worker]
                if any(measurement[key]!=term['parameters'][key] for key in ('cpu_id','row_begin','row_end')):
                    raise ValueError('measured canonical partition differs from actual anonymous demand')
                locator=f'#/observations/{observation_index}/worker_chunks/{worker}'
            else:
                key=dict(boundary='boundary',allocate='allocation',team='team',**{'return':'publication'})[term['id']]
                measurement=row['serial_components'][key];locator=f'#/observations/{observation_index}/serial_components/{key}'
            components.append(dict(component=term['component'],parameters=term['parameters'],basis='DECLARED_COMPONENT_MODEL',
                **measurement,measurement_source=dict(artifact=reference,locator=locator),family_match_proof=family,
                assumptions=measured['assumptions'],complete_target_component_measured=False))
    except ValueError as error:
        components=[];unresolved=[dict(component=term['component'],parameters=term['parameters'],reason=str(error)) for term in demand['terms']]
    result=dict(schema='scar.temporal-resource-calibration',schema_version=1,resource_context=demand['resource_context'],
        source_calibration=reference,components=components,unsupported_components=unresolved,family_match_proof=family,
        target_workload_executed=False,target_variants_ranked=False,worker_scaling_inferred=False,
        complete_target_call_measured=False,physical_optimality_proven=False)
    out.mkdir(parents=True);_save(out/'observations.json',result)
    return result


def project_temporal_resource_facts(domain,candidate_id,calibration,out):
    """Bind independently measured family/geometry to one actual demand later."""
    materials=_projection_materials(calibration)
    result=_project_temporal_facts(domain,candidate_id,out,*materials)
    for reference in (materials[0],materials[1]['frozen_design'],materials[1]['raw_observations']):
        read_artifact(reference,json_document=False)
    return result


def project_temporal_resource_fact_batch(domain,calibration,out):
    """Share one verified raw index; emit independent per-candidate shards.

    Parsing and indexing scale with observation count plus candidate count.
    Input receipts are checked again before publishing the batch manifest.
    """
    out=Path(out).resolve()
    if out.exists():raise ValueError('fresh temporal fact batch output required')
    materials=_projection_materials(calibration);out.mkdir(parents=True);rows=[]
    for index,candidate in enumerate(domain['candidates']):
        folder=out/('candidate-'+str(index))
        result=_project_temporal_facts(domain,candidate['id'],folder,*materials)
        rows.append(dict(candidate_id=candidate['id'],complete=bool(result['components']) and not result['unsupported_components'],
            observations=artifact_reference(folder/'observations.json',result['schema'])))
    for reference in (materials[0],materials[1]['frozen_design'],materials[1]['raw_observations']):
        read_artifact(reference,json_document=False)
    report=dict(schema='scar.temporal-resource-fact-batch',schema_version=1,
        source_calibration=materials[0],domain_id=domain['id'],candidates=rows,
        raw_observation_sets_parsed=1,source_receipts_rechecked=True,target_program_executed=False)
    _save(out/'result.json',report);return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',required=True);parser.add_argument('--cpu-masks',required=True,help='semicolon-separated comma CPU masks')
    parser.add_argument('--shape',required=True,nargs=2,type=int);parser.add_argument('--trip-count',required=True,type=int)
    parser.add_argument('--repetitions',type=int,default=7);parser.add_argument('--warmup',type=int,default=2);parser.add_argument('--out',required=True)
    args=parser.parse_args(argv);masks=[[int(cpu) for cpu in mask.split(',')] for mask in args.cpu_masks.split(';')]
    result=calibrate_temporal_cpu(args.target,masks,args.shape,args.trip_count,args.out,repetitions=args.repetitions,warmup=args.warmup)
    print(json.dumps(dict(result=str(Path(args.out).resolve()/'result.json'),observations=len(result['observations']),
        native_compilations=1,seconds=result['seconds'],basis=result['basis'],target_workload_executed=False),sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
