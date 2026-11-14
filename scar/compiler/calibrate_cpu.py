"""Calibrate a fixed CPU/backend context using canonical primitive programs.

No target workload or optimization alternative is executed. Nonnegative fitting
produces an effective model, with rank/residual diagnostics; it does not identify
instruction latencies, DRAM traffic, or a unique physical cost decomposition.
"""
from dataclasses import replace
from hashlib import sha256
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np

from scar.codegen import build
from scar.codegen.cpp import runtime_implementation_sha256
from scar.compiler.frontend import discover_regions
from scar.compiler.model import read_plan
from scar.optimization.demand import analyze_region_demand
from scar.optimization.hardware import bind_cpu_placement,validate_cpu_placement
from scar.optimization.optimizer import optimize_region
from scar.optimization.resource_solver import hardware_model_skeleton,solve_resource_schedule
from scar.optimization.schedule import build_region_schedule
from scar.ir.program.providers import ProviderArtifact,ProviderFactReference
from scar.ir.record_codec import encode

def _write(path,document):
    path.write_text(json.dumps(document,indent=2,allow_nan=False)+'\n')

def _timed(function,repetitions,warmup):
    for _ in range(warmup):function()
    samples=[]
    for _ in range(repetitions):
        start=time.perf_counter_ns();value=function();elapsed=time.perf_counter_ns()-start
        del value
        samples.append(elapsed)
    return dict(median_ns=statistics.median(samples),samples_ns=samples,
        min_ns=min(samples),max_ns=max(samples))

def _primitive_sources(rows,cols):
    result=[]
    def add(parameters,expression,*,views=0,holdout=False):
        name='primitive_'+str(len(result))
        text=f'def {name}('+','.join(parameters)+'):\n    return '+expression+'\n'
        result.append(dict(name=name,source=text,parameters=parameters,views=views,holdout=holdout))
    for op,symbol in (('add','+'),('sub','-'),('mul','*'),('div','/'),('neg','-')):
        for count in (1,3,6):
            expression='p0'
            for _ in range(count):expression=f'-({expression})' if op=='neg' else f'({expression}{symbol}p1)'
            add(['p0'] if op=='neg' else ['p0','p1'],expression)
    for count in (1,2,3,5,7):
        operands=[f'p0[{i}:{rows+i},{i}:{cols+i}]' for i in range(count)]
        expression=operands[0]+'+0.125' if count==1 else '+'.join(operands)
        add(['p0'],expression,views=count)
    for count in (1,3,5,7):
        parameters=['p'+str(i) for i in range(count)]
        add(parameters,'+'.join(parameters)+'+0.125')
    for count in (1,3,6):
        expression='p0'
        for i in range(count):expression=f'({expression}*{1.01+i*0.001!r})'
        add(['p0'],expression)
    add(['p0','p1'],'(p0+p1)*p0-p1',holdout=True)
    add(['p0'],'(p0[0:'+str(rows)+',0:'+str(cols)+']+p0[1:'+str(rows+1)+',1:'+str(cols+1)+'])*1.125',views=2,holdout=True)
    add(['p0','p1'],'-(p0-p1)/(p1+1.125)',holdout=True)
    return result

def _fit(names,rows,targets):
    from scipy.optimize import nnls
    matrix=np.asarray([[float(row.get(name,0)) for name in names] for row in rows])
    active=np.any(matrix!=0,axis=0);values=np.zeros(len(names))
    scales=np.max(np.abs(matrix[:,active]),axis=0)
    solved,_=nnls(matrix[:,active]/scales,np.asarray(targets,dtype=float),maxiter=10000)
    values[active]=solved/scales
    predicted=matrix@values;errors=predicted-np.asarray(targets)
    return dict(coefficients={name:float(value) for name,value in zip(names,values)},
        active_features=[name for name,use in zip(names,active) if use],
        rank=int(np.linalg.matrix_rank(matrix[:,active]/scales)),parameters=int(np.sum(active)),
        unique_parameters_identified=bool(np.linalg.matrix_rank(matrix[:,active]/scales)==sum(active)),
        rmse_ticks=float(np.sqrt(np.mean(errors**2))),
        max_relative_error=float(np.max(np.abs(errors)/np.maximum(np.abs(targets),1))),
        negative_adjusted_observations=sum(value<0 for value in targets))

def calibrate_cpu(plan_path,out,*,input_shapes=None,metadata_path=None,repetitions=9,warmup=3):
    from scar.optimization.cost_model import cost_feature_units,features_for_demand
    if sys.getprofile() is not None or sys.gettrace() is not None:
        raise ValueError('primitive calibration must not have Python trace/profile hooks')
    if repetitions<3 or warmup<1:raise ValueError('at least three repetitions and one warmup required')
    plan=read_plan(plan_path);binding=plan.resource_binding;shape_evidence=None
    if metadata_path is not None:
        if input_shapes is not None:raise ValueError('choose declared shapes or observed input metadata')
        from .input_metadata import shapes_from_metadata
        input_shapes,shape_evidence=shapes_from_metadata(metadata_path,plan.region.source,plan.region.parameters)
    if binding is None:raise ValueError('explicit recorded CPU binding required for calibration')
    validate_cpu_placement(binding)
    skeleton=hardware_model_skeleton(plan,build_region_schedule(plan,binding.target_path),input_shapes=input_shapes)
    shape=skeleton['model_instantiation']['output_shape']
    if shape is None or min(shape)<1:raise ValueError('nonempty rank-two output shape required')
    elements=math.prod(shape)
    if elements>4*1024*1024:raise ValueError('primitive calibration output is limited to four million elements')
    calibration_budget=plan.cost.get('memory_budget_bytes')
    if calibration_budget is None:calibration_budget=max(4096,elements*16)
    binding=replace(binding,memory_budget_bytes=calibration_budget)
    out=Path(out).absolute()
    if out.exists():raise ValueError('fresh calibration output required')
    out.mkdir(parents=True);cases=_primitive_sources(*shape)
    source=out/'primitives.py';source.write_text('\n'.join(case['source'] for case in cases))
    regions,rejections=discover_regions(source)
    if rejections or len(regions)!=len(cases):raise ValueError('canonical primitive generation did not match the closed profile')
    lookup={region.source.qualname:region for region in regions}
    namespace={};exec(compile(source.read_bytes(),str(source),'exec'),namespace)
    units=cost_feature_units();compute_names=['intercept',*units['COMPUTE']]
    rng=np.random.default_rng(11);observations=[];context=None
    allocation=[]
    # These probes include immediate release. This is retained as an explicit
    # effective boundary model, rather than mislabeled allocator-only latency.
    for count in sorted(set((max(1,elements//4),elements))):
        def allocate(count=count):
            for _ in range(32):np.empty(count,dtype=np.float64)
        times=_timed(allocate,repetitions,warmup)
        allocation.append(dict(output_bytes=count*8,median_ns=times['median_ns']/32,
            samples_ns=[value/32 for value in times['samples_ns']]))
    export_probe=_timed(lambda:None,repetitions*8,warmup)
    allocation_fit=_fit(['intercept','output_bytes'],
        [dict(intercept=1,output_bytes=row['output_bytes']) for row in allocation],
        [row['median_ns'] for row in allocation])
    export_fit=dict(coefficients={'intercept':float(export_probe['median_ns'])})
    for index,case in enumerate(cases):
        region=lookup[case['name']]
        primitive=optimize_region(region,memory_budget_bytes=calibration_budget,cpu_threads=plan.cpu_threads)
        primitive=bind_cpu_placement(primitive,binding)
        input_shape=(shape[0]+case['views'],shape[1]+case['views'])
        arrays=[rng.uniform(0.5,1.0,size=input_shape) for _ in case['parameters']]
        resource_schedule=build_region_schedule(primitive,binding.target_path)
        primitive_model=hardware_model_skeleton(primitive,resource_schedule,input_shapes=input_shape)
        # Declared unit ticks enable the identical timed guard path. They do
        # not drive execution or become calibration measurements.
        guard_plan=solve_resource_schedule(resource_schedule,
            task_costs={'allocate:output':1,'compute:region':1,'export:output':1},
            **{key:primitive_model[key] for key in ('task_placements','resource_capacities',
                'memory_capacities','materialization_sizes','memory_locations','model_instantiation')})
        artifact=build(primitive,out/f'primitive-{index}'/'native',resource_schedule=resource_schedule,timed_plan=guard_plan)
        context=artifact.report if context is None else context
        expected=namespace[case['name']](*arrays);actual=artifact.run(*arrays)
        if not np.array_equal(expected.view(np.uint64),actual.view(np.uint64)):
            raise ValueError('primitive output differs from its canonical Python expression')
        timing=_timed(lambda:artifact.run(*arrays),repetitions,warmup)
        demand=analyze_region_demand(primitive)
        quantities=features_for_demand(demand,elements)['COMPUTE']
        features={name:value[0] for name,value in quantities.items()}
        alloc_ticks=(allocation_fit['coefficients']['intercept']+
            allocation_fit['coefficients']['output_bytes']*elements*8)
        adjusted=timing['median_ns']-alloc_ticks-export_fit['coefficients']['intercept']
        observations.append(dict(name=case['name'],holdout=case['holdout'],features=features,
            output_elements=elements,input_shapes=[list(a.shape) for a in arrays],
            bitwise_validated=True,timing=timing,adjusted_compute_ticks=adjusted,
            native_plan_sha256=artifact.report['plan_sha256']))
    train=[row for row in observations if not row['holdout']]
    compute_fit=_fit(compute_names,[dict(intercept=1,**row['features']) for row in train],
        [row['adjusted_compute_ticks'] for row in train])
    holdout=[]
    for row in observations:
        if not row['holdout']:continue
        predicted=sum(compute_fit['coefficients'][name]*value for name,value in dict(intercept=1,**row['features']).items())
        actual=row['adjusted_compute_ticks']
        holdout.append(dict(name=row['name'],predicted_compute_ticks=predicted,observed_adjusted_ticks=actual,
            relative_error=abs(predicted-actual)/max(abs(actual),1)))
    fits={'ALLOCATION':allocation_fit,'COMPUTE':compute_fit,'EXPORT':export_fit}
    document=dict(schema='scar.cpu-primitive-calibration',schema_version=1,
        target=dict(target_sha256=binding.target_sha256,executor_id=binding.executor_id,cpu_ids=list(binding.cpu_ids)),
        compiler=dict(identity=context['compiler_version'],options=context['compiler_options'],argv=context['compiler']),
        runtime_implementation_sha256=runtime_implementation_sha256(),
        tick_seconds=1e-9,output_shape=list(shape),output_elements=elements,
        input_shape_evidence=shape_evidence,
        allocation_observations=allocation,export_observations=export_probe,observations=observations,
        export_probe_scope='EMPTY_PYTHON_CALL_RETURN_PROXY_NOT_ARRAY_EXPORT_OR_OWNERSHIP_TRANSFER',
        fits=fits,holdout=holdout,repetitions=repetitions,warmup=warmup,
        execution_boundary='TIMED_SYNCHRONOUS_CPU_PROFILE',calibration_output_budget_bytes=calibration_budget,
        timing_scope='CPP_ARTIFACT_RUN_EXCLUDES_COMPATIBILITY_WRAPPER_AND_CALLER',
        declared_guard_plan_ticks='1/1/1_ONLY_ENABLE_GUARDS_NOT_MEASURED_OR_SLEEP_TIMINGS',
        target_workload_executed=False,program_variants_ranked=False,
        regime='CACHE_WARM_REPEATED_HOST_ARRAYS_FIXED_OUTPUT_ELEMENTS',
        caveats=['Effective affine fit; coefficients may be non-identifiable and zero at an active NNLS constraint.',
            'Logical payload bytes are not measured DRAM transfers or cache misses.',
            'Allocation probe includes immediate release; compute observations subtract the fitted boundary terms.',
            'Export coefficient uses an empty Python call/return proxy, not an independent ndarray export measurement.',
            'Cache/NUMA residency and other workload contention are not controlled or proven.',
            'Calibration of canonical primitives does not prove accuracy for an arbitrary composed region.'])
    data_path=out/'observations.json';_write(data_path,document)
    data=data_path.read_bytes();reference=ProviderArtifact(str(data_path),sha256(data).hexdigest(),len(data),document['schema'])
    def evidence(pointer):return [encode(ProviderFactReference(reference,pointer))]
    def parameter(value,unit,pointer):
        result=dict(value=value,unit=unit,evidence=evidence(pointer))
        if value==0:result['zero_basis']='NONNEGATIVE_FIT_ACTIVE_CONSTRAINT'
        return result
    task_models={}
    for kind,feature_units in units.items():
        coefficients=fits[kind]['coefficients']
        task_models[kind]=dict(intercept=parameter(coefficients['intercept'],'ticks',f'#/fits/{kind}/coefficients/intercept'),
            coefficients={name:parameter(coefficients.get(name,0),unit,f'#/fits/{kind}/coefficients/{name}')
                for name,unit in feature_units.items() if name in coefficients
                and (kind!='COMPUTE' or name in compute_fit['active_features'])})
    profile=dict(schema='scar.region-cost-profile',schema_version=1,target=document['target'],
        backend=dict(name='cpp',execution_profile=plan.profile,operator_abi_version=plan.operator_abi_version,
            guard_mode='TIMED_SYNCHRONOUS_CPU_PROFILE',
            runtime_implementation_sha256=document['runtime_implementation_sha256'],
            context_evidence=evidence('#'),
            compiler={**document['compiler'],'evidence':evidence('#/compiler')}),
        applicability=dict(layout='C_CONTIGUOUS_FLOAT64_2D',working_set=dict(regime=document['regime'],
            output_elements_min=elements,output_elements_max=elements,
            logical_payload_bytes_min=min(row['features']['logical_payload_bytes'] for row in observations),
            logical_payload_bytes_max=max(row['features']['logical_payload_bytes'] for row in observations),
            evidence=evidence('#/observations'))),
        model=dict(kind='EFFECTIVE_AFFINE',composition='SUM_EFFECTIVE_TERMS',
            tick_seconds=parameter(1e-9,'seconds/tick','#/tick_seconds'),task_models=task_models),
        validation=dict(fit_rank=compute_fit['rank'],fit_parameters=compute_fit['parameters'],
            parameters_unique=compute_fit['unique_parameters_identified'],fit_rmse_ticks=compute_fit['rmse_ticks'],
            maximum_holdout_relative_error=max(row['relative_error'] for row in holdout),
            scope='CANONICAL_PRIMITIVES_ONLY_NOT_TARGET_WORKLOAD',physical_optimality_proven=False))
    _write(out/'profile.json',profile)
    return profile

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True)
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--shape',type=int,nargs=2,help='declared common input shape of the source region')
    inputs.add_argument('--inputs',help='versioned actual function-boundary input metadata')
    p.add_argument('--out',required=True);p.add_argument('--repetitions',type=int,default=9);p.add_argument('--warmup',type=int,default=3)
    a=p.parse_args();profile=calibrate_cpu(a.plan,a.out,input_shapes=a.shape,metadata_path=a.inputs,repetitions=a.repetitions,warmup=a.warmup)
    print(json.dumps(dict(profile=str(Path(a.out)/'profile.json'),validation=profile['validation'])))

if __name__=='__main__':main()
