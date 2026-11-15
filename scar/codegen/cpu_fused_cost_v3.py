"""Fixed-primitive, overlap-aware declared cost of shared-root CPU stages.

Separate canonical barrier, memory/fanin, independent arithmetic chains and
dependent chains determine effective proxy costs. The forward composition is
barrier + max(memory, arithmetic throughput, weighted dependency path).
Neither proxy coefficients nor the engineering holdout policy are hardware
instruction latencies, physical bounds or a claim of optimal placement.
"""
import argparse
from collections import Counter
from copy import deepcopy
import ctypes
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time

from . import cpu_fused_cost_v2 as V2
from .cpu_fused_cost import _HERE,_Layout,_hash,_json,_llc,_receipt,_require,_save

_POLICY=dict(maximum_holdout_relative_error=.20,
    authority='EXPLICIT_ENGINEERING_POLICY_NOT_A_THEORETICAL_ERROR_BOUND',
    coefficient_sources='SEPARATE_FIXED_CANONICAL_PRIMITIVES_ONLY',
    holdout_used_for_fitting=False,holdout_used_for_model_choice=False)
_ASSUMPTIONS=[
    'CANONICAL_MEMORY_FANIN_IS_A_PROXY_NOT_ISOLATED_MEMORY_OR_DRAM_BANDWIDTH',
    'ARITHMETIC_CHAIN_COSTS_INCLUDE_LOAD_STORE_LOOP_AND_FENCE_AMORTIZATION',
    'DEPENDENT_CHAIN_PROXY_IS_NOT_INTRINSIC_CPU_INSTRUCTION_LATENCY',
    'MAX_COMPOSITION_IS_A_DECLARED_OVERLAP_MODEL_NOT_A_HARDWARE_SCHEDULING_THEOREM',
    'GEOMETRIC_INTERPOLATION_IS_NOT_A_WORKER_SCALING_OR_AVAILABLE_CACHE_CLAIM',
    'WARM_REPEATED_BUFFERS_CACHE_NUMA_AND_CONTENTION_NOT_CONTROLLED',
    'CANONICAL_VECTORIZATION_PATH_IS_NOT_TARGET_COMPILER_PATH_IDENTITY',
    'FRESH_HOLDOUT_ERRORS_ARE_NOT_BOUNDS_ON_TARGET_ERROR',
    'WHOLE_CALL_ADDITIVITY_AND_DEPLOYMENT_PROFITABILITY_REQUIRE_SEPARATE_VALIDATION']


def _registry():
    result=[dict(id='primitive_barrier',role='BARRIER',array_loads=0,graph=[],output=None,scalar_count=0,holdout=False)]
    for loads in (1,2,4,6,8):
        c=V2._case('BALANCED_FANIN',loads,0);c.update(id='primitive_fanin_'+str(loads),role='MEMORY_PROXY')
        result.append(c)
    for family,chains in (('THROUGHPUT_PROXY',4),('DEPENDENCY_PROXY',1)):
        for op in ('add','sub','mul'):
            graph=[];values=[];scalar_count=0
            for chain in range(chains):
                value='x0'
                for step in range(16):
                    key='v'+str(len(graph));scalar='s'+str(scalar_count);scalar_count+=1
                    graph.append(dict(id=key,op=op,inputs=[value,scalar]));value=key
                values.append(value)
            while len(values)>1:
                key='v'+str(len(graph));graph.append(dict(id=key,op='add',inputs=values[:2]));values=[key,*values[2:]]
            result.append(dict(id='primitive_'+family.lower()+'_'+op,role=family,operation=op,
                array_loads=1,graph=graph,output=values[0],scalar_count=scalar_count,holdout=False))
    for i,(family,loads,extra) in enumerate((
        ('SERIAL_FANIN',5,7),('SERIAL_FANIN',7,11),
        ('BALANCED_FANIN',3,11),('BALANCED_FANIN',5,9),
        ('PARALLEL_BRANCHES',5,5),('PARALLEL_BRANCHES',7,11),
        ('SHARED_BRANCHES',3,9),('SHARED_BRANCHES',7,7))):
        case=V2._case(family,loads,extra,seed=61+i,holdout=True)
        case.update(role='FRESH_HOLDOUT');result.append(case)
    return result


def _configurations(context):
    rows=[];workers=len(context['worker_cpu_ids'])
    for index,case in enumerate(_registry()):
        shapes=((472,608),(601,735)) if case['holdout'] else ((256,320),(513,575),(770,832))
        groups=(1,2) if case['role'] in ('MEMORY_PROXY','FRESH_HOLDOUT') else (1,)
        if case['role']=='BARRIER':shapes=((1,1),)
        for group in groups:
            if group==2 and case['array_loads']==1:continue
            for r,c in shapes:
                layouts=V2._layouts(case['array_loads'],c,group)
                counts=Counter(n['op'] for n in case['graph'])
                rows.append(dict(id='v3-configuration-'+str(len(rows)),case_index=index,kernel_index=index*2+group-1,
                    role=case['role'],operation=case.get('operation'),holdout=case['holdout'],groups=group,
                    rows=r,cols=c,workers=workers,elements=((r+workers-1)//workers)*c,
                    array_loads=case['array_loads'],layouts=layouts,arithmetic={op:counts[op] for op in ('add','sub','mul')}))
    random.Random(208199).shuffle(rows);return rows


def _source(context):
    """Reuse the unchanged V2 native harness; replace only fixed registry text."""
    original=V2._source(context)
    head=original[:original.index('__attribute__((noinline)) void registry_')]
    head=head.replace('double scalars[16]','double scalars[128]')
    tail=original[original.index('extern "C" SCAR_EXPORT_V1 int32_t scar_fused_cost_probe_v2'):]
    tail=tail.replace('i<16;++i)a.scalars','i<128;++i)a.scalars')
    # Barrier-only output has no producer. Zero it before (outside) timing.
    tail=tail.replace('double[l.rows*l.cols]);','double[l.rows*l.cols]());')
    functions=[];names=[];loads=[]
    for case in _registry():
        for groups in (1,2):
            name=case['id']+'_g'+str(groups);body=[]
            for i,layout in enumerate(V2._layouts(case['array_loads'],100,groups)):
                padding=layout['stride']-100;r,c=divmod(layout['offset'],layout['stride'])
                body.append(f'const double x{i}=a.inputs[{layout["storage_group"]}][({r}+row)*(a.cols+{padding})+{c}+col];')
            for node in case['graph']:
                args=['a.scalars['+k[1:]+']' if k.startswith('s') else k for k in node['inputs']]
                body.append('const double '+node['id']+'=('+args[0]+dict(add='+',sub='-',mul='*')[node['op']]+args[1]+');')
            if case['role']=='BARRIER':text='#pragma omp barrier\n'
            else:
                body.append('a.output[row*a.cols+col]='+case['output']+';')
                text='#pragma omp for schedule(static)\nfor(int64_t row=0;row<a.rows;++row)\nfor(int64_t col=0;col<a.cols;++col){\n'+'\n'.join(body)+'\n}\n'
            functions.append('__attribute__((noinline)) void '+name+'(const Args &a){\n'+text+'}\n')
            names.append('&'+name);loads.append(str(case['array_loads']))
    return head+'\n'.join(functions)+'\nusing Kernel=void(*)(const Args &);\nconstexpr Kernel kernels[]={'+', '.join(names)+'};\nconstexpr int required_loads[]={'+', '.join(loads)+'};\n}\n'+tail


def _native(context,directory):
    compiler=context['backend']['compiler'];cpus=context['worker_cpu_ids']
    _require(len(cpus)>1 and set(cpus)<=os.sched_getaffinity(0)
        and sorted(os.sched_getaffinity(0))==context['caller_affinity_cpu_ids'], 'exact OpenMP CPU context required')
    identity=subprocess.run([*compiler['argv'],'--version'],check=True,capture_output=True,text=True).stdout.splitlines()[0]
    _require(identity==compiler['identity'] and 'g++' in identity and '-fopenmp' in compiler['options'],'exact GCC/OpenMP required')
    directory.mkdir();source=directory/'canonical.cpp';source.write_text(_source(context))
    (directory/'scar_abi_v1.h').write_bytes((_HERE/'scar_abi_v1.h').read_bytes())
    library=directory/'canonical.so';remarks=directory/'vectorization.txt'
    command=[*compiler['argv'],*compiler['options'],'-fopt-info-vec-optimized='+str(remarks),str(source),'-o',str(library)]
    p=subprocess.run(command,capture_output=True,text=True);(directory/'compile.log').write_text(p.stdout+p.stderr)
    _require(p.returncode==0,'canonical probe failed: '+p.stderr[-1200:])
    assembly=subprocess.run(['objdump','-d','-C','--no-show-raw-insn',str(library)],check=True,capture_output=True,text=True).stdout
    (directory/'assembly.txt').write_text(assembly)
    handle=ctypes.CDLL(str(library));function=handle.scar_fused_cost_probe_v2
    function.argtypes=[ctypes.c_int32,ctypes.POINTER(_Layout),ctypes.c_uint64,ctypes.c_uint64,ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_int32)]
    function.restype=ctypes.c_int32
    return handle,function,dict(compiler=compiler,command=command,source=_receipt(source),library=_receipt(library),
        assembly=_receipt(directory/'assembly.txt'),vectorization_remarks=_receipt(remarks),
        target_body_compiled=False,runtime_vector_branch_observed=False)


def _interpolate(points,x):
    points=sorted(points);_require(points[0][0]<=x<=points[-1][0],'outside primitive interpolation range')
    for a,b in zip(points,points[1:]):
        if a[0]<=x<=b[0]:return a[1]+(b[1]-a[1])*(x-a[0])/(b[0]-a[0])
    return points[0][1]


def _derive(observations):
    barrier=next(r['median_ns'] for r in observations if r['role']=='BARRIER')
    memory={};throughput={};dependency={}
    for group in (1,2):
        memory[str(group)]={}
        for loads in ((1,2,4,6,8) if group==1 else (2,4,6,8)):
            rows=[r for r in observations if r['role']=='MEMORY_PROXY' and r['groups']==group and r['array_loads']==loads]
            memory[str(group)][str(loads)]=[dict(elements=r['elements'],rows=r['rows'],cols=r['cols'],
                proxy_ns_per_element=max(0,r['median_ns']-barrier)/r['elements'],configuration=r['id']) for r in rows]
    for role,destination,count in (('THROUGHPUT_PROXY',throughput,64),('DEPENDENCY_PROXY',dependency,16)):
        medians={op:statistics.median(max(0,r['median_ns']-barrier)/r['elements'] for r in observations
            if r['role']==role and r['operation']==op) for op in ('add','sub','mul')}
        # Independent-chain outputs have three explicit add merges.
        merges=3 if role=='THROUGHPUT_PROXY' else 0
        destination['add']=medians['add']/(count+merges)
        for op in ('sub','mul'):destination[op]=max(0,medians[op]-merges*destination['add'])/count
    return dict(barrier_ns=barrier,memory=memory,arithmetic_ns_per_operation=throughput,
        dependency_ns_per_operation=dependency,coefficient_authority='SEPARATE_FIXED_CANONICAL_PROXY_MEASUREMENTS',
        physical_instruction_latency_or_bandwidth_measured=False)


def _predict(case,config,coefficients):
    group=str(config['groups']);elements=config['elements'];loads=config['array_loads']
    points=[]
    for count,rows in coefficients['memory'][group].items():
        value=_interpolate([(r['elements'],r['proxy_ns_per_element']) for r in rows],elements)
        points.append((int(count),value))
    memory=_interpolate(points,loads)*elements
    arithmetic=sum(config['arithmetic'][op]*coefficients['arithmetic_ns_per_operation'][op] for op in ('add','sub','mul'))*elements
    paths={};arrays={'x'+str(i) for i in range(loads)}
    for node in case['graph']:
        paths[node['id']]=max((paths.get(k,0) for k in node['inputs']),default=0)+coefficients['dependency_ns_per_operation'][node['op']]
    dependency=paths.get(case['output'],0)*elements
    return dict(duration_ns=coefficients['barrier_ns']+max(memory,arithmetic,dependency),
        barrier_ns=coefficients['barrier_ns'],memory_proxy_ns=memory,arithmetic_throughput_proxy_ns=arithmetic,
        weighted_dependency_proxy_ns=dependency,composition='BARRIER_PLUS_MAX',
        dominant_proxy=max((('MEMORY',memory),('ARITHMETIC',arithmetic),('DEPENDENCY',dependency)),key=lambda r:r[1])[0])


def _qualify(observations,coefficients):
    result={};registry=_registry()
    for group in (1,2):
        holdout=[]
        for row in observations:
            if row['role']!='FRESH_HOLDOUT' or row['groups']!=group:continue
            cost=_predict(registry[row['case_index']],row,coefficients)
            holdout.append(dict(configuration=row['id'],observed_ns=row['median_ns'],prediction=cost,
                relative_error=abs(cost['duration_ns']-row['median_ns'])/row['median_ns']))
        maximum=max(r['relative_error'] for r in holdout)
        result[str(group)]=dict(holdout=holdout,maximum_holdout_relative_error=maximum,
            admitted=maximum<=_POLICY['maximum_holdout_relative_error'],precision_policy=_POLICY)
    return result


def calibrate_fused_cpu_v3(resource_context,out,*,repetitions=5,warmup=2,batch=8):
    from scar.ir.execution_recipe import artifact_reference
    context=deepcopy(resource_context);out=Path(out).resolve()
    _require(not out.exists() and repetitions>=3 and warmup>=1 and batch>=1,'fresh output and repeated warm samples required')
    _require(all(os.environ.get(k)==v for k,v in context['native_thread_environment'].items()),'thread environment differs')
    capacity,cache=_llc(context);configurations=_configurations(context)
    implementation={name:_hash((_HERE/name).read_bytes()) for name in
        ('cpu_fused_cost_v3.py','cpu_fused_cost_v2.py','cpu_fused_cost_v2.cpp.in','cpu_fused_cost.py','loop_v1.cpp.in','scar_abi_v1.h')}
    out.mkdir(parents=True);start=time.time();before=list(os.getloadavg())
    design=dict(schema='scar.cpu-fused-v3-design',schema_version=1,registry=_registry(),configurations=configurations,
        resource_context=context,cache_geometry=cache,precision_policy=_POLICY,implementation_files=implementation,
        repetitions=repetitions,warmup=warmup,batch=batch,order_seed=208199,
        composition='BARRIER_PLUS_MAX_MEMORY_ARITHMETIC_DEPENDENCY',
        coefficient_derivation='MEDIAN_EFFECTIVE_CANONICAL_UNIT_COSTS_AND_PIECEWISE_GEOMETRY_INTERPOLATION',
        model_frozen_before_measurement=True,target_workload_executed=False,target_variants_ranked=False,
        inspected_v1_v2_observations_are_not_new_holdouts=True)
    _save(out/'design.json',design);design_ref=artifact_reference(out/'design.json',design['schema'])
    handle,function,native=_native(context,out/'native');observations=[]
    for config in configurations:
        layout=_Layout();layout.rows=config['rows'];layout.cols=config['cols'];layout.loads=config['array_loads']
        for i,row in enumerate(config['layouts']):
            layout.strides[i]=row['stride'];layout.offsets[i]=row['offset'];layout.groups[i]=row['storage_group']
        samples=(ctypes.c_double*repetitions)();sink=ctypes.c_uint64();cpus=(ctypes.c_int32*2)()
        status=function(config['kernel_index'],ctypes.byref(layout),repetitions,warmup,batch,samples,ctypes.byref(sink),cpus)
        _require(status==0,'canonical FP/affinity/allocation guard: '+str(status))
        values=list(samples);_require(all(math.isfinite(x) and x>0 for x in values),'positive finite samples required')
        observations.append(dict(**config,samples_ns=values,median_ns=statistics.median(values),min_ns=min(values),
            max_ns=max(values),stdev_ns=statistics.stdev(values),checksum=sink.value,
            caller_cpu_before=cpus[0],caller_cpu_after=cpus[1]))
    raw=dict(schema='scar.cpu-fused-v3-observations',schema_version=1,frozen_design=design_ref,native=native,observations=observations)
    _save(out/'raw-observations.json',raw);coefficients=_derive(observations)
    _require(artifact_reference(out/'design.json',design_ref['schema'])==design_ref
        and sorted(os.sched_getaffinity(0))==context['caller_affinity_cpu_ids']
        and all(os.environ.get(k)==v for k,v in context['native_thread_environment'].items()),'frozen design/context changed')
    model=dict(schema='scar.cpu-fused-bottleneck-model',schema_version=1,resource_context=context,
        model_kind='DECLARED_CANONICAL_PROXY_BARRIER_PLUS_MAX',frozen_design=design_ref,
        raw_observations=artifact_reference(out/'raw-observations.json',raw['schema']),coefficients=coefficients,
        local_models=_qualify(observations,coefficients),precision_policy=_POLICY,implementation_files=implementation,
        native=native,cache_geometry=cache,assumptions=_ASSUMPTIONS,calibration_seconds=time.time()-start,
        host_load=dict(before=before,after=list(os.getloadavg()),isolated=False),
        target_workload_executed=False,target_variants_ranked=False,physical_optimality_proven=False,deployment_authorized=False)
    _save(out/'profile.json',model);return model


def _verify_model(path):
    from scar.ir.execution_recipe import artifact_reference,read_artifact
    reference=artifact_reference(path,'scar.cpu-fused-bottleneck-model');model=read_artifact(reference)
    _require(model['schema_version']==1 and model['precision_policy']==_POLICY,'model/policy differs')
    design=read_artifact(model['frozen_design']);raw=read_artifact(model['raw_observations'])
    _require(design['implementation_files']==model['implementation_files']=={
        name:_hash((_HERE/name).read_bytes()) for name in model['implementation_files']},'current producer differs')
    _require(design['registry']==_registry() and design['configurations']==_configurations(model['resource_context'])
        and raw['frozen_design']==model['frozen_design'] and raw['native']==model['native'],'canonical design/native differs')
    _require(len(raw['observations'])==len(design['configurations']),'incomplete design')
    for row,config in zip(raw['observations'],design['configurations']):
        _require(all(row[k]==v for k,v in config.items()) and len(row['samples_ns'])==design['repetitions']
            and all(math.isfinite(x) and x>0 for x in row['samples_ns'])
            and statistics.median(row['samples_ns'])==row['median_ns'],'canonical samples differ')
    _require(_derive(raw['observations'])==model['coefficients']
        and _qualify(raw['observations'],model['coefficients'])==model['local_models'],'coefficient/holdout replay differs')
    _require(_source(model['resource_context'])==Path(model['native']['source']['path']).read_text(),'native source differs')
    for row in model['native'].values():
        if type(row) is dict and {'path','sha256','bytes'}<=set(row):
            _require(_receipt(Path(row['path']))==row,'native receipt differs')
    _require(_llc(model['resource_context'])[1]==model['cache_geometry'],'cache geometry differs')
    return reference,model


def predict_fused_components_v3(domain,candidate_id,profile,out):
    from scar.optimization.stateful_cpu_choices import _domain
    from scar.optimization.stateful_cpu_model import stateful_cpu_demand
    domain=_domain(domain);demand=stateful_cpu_demand(domain,candidate_id);reference,model=_verify_model(profile)
    _require(demand['resource_context']==model['resource_context'],'exact resource/compiler/runtime context differs')
    dags=V2._domain_dags(domain);capacity=model['cache_geometry']['reported_capacity_bytes'];components=[];unsupported=[];proposals=[]
    for term in demand['terms']:
        if term['component']!='FUSED_SCALAR_DAG_ROW_STAGE':continue
        proposal=None
        try:
            p=term['parameters'];dag=dags[term['source_ref']['pointer']]
            _,parameters,geometry=V2.dag_features(dag['graph'],dag['output'],dag['layouts'],p['rows'],p['cols'],p['workers'],capacity)
            _require(parameters==p,'anonymous DAG does not replay public demand')
            group=p['backing_storage_groups'];qualified=model['local_models'].get(str(group))
            _require(qualified is not None and 256<=p['rows']<=770 and 320<=p['cols']<=832,
                'uncalibrated roots/row/column geometry')
            expected=V2._layouts(p['array_loads'],p['cols'],group)
            key=lambda r:(r['storage_group'],r['stride'],r['offset'])
            _require(sorted(p['operand_layouts'],key=key)==sorted(expected,key=key),
                'layout is outside canonical shared-root axial-neighborhood geometry')
            _require(geometry['backing_bytes']+p['rows']*p['cols']*8<=capacity,'reported cache extent regime exceeded')
            config=dict(groups=group,elements=((p['rows']+p['workers']-1)//p['workers'])*p['cols'],
                array_loads=p['array_loads'],arithmetic={op:p['arithmetic'][op] for op in ('add','sub','mul')})
            cost=_predict(dag,config,model['coefficients'])
            proposal=dict(component=term['component'],parameters=p,prediction=cost,
                maximum_holdout_relative_error=qualified['maximum_holdout_relative_error'],precision_policy=_POLICY,
                measured_target_component=False,selectable=False)
            _require(qualified['admitted'],'fresh independent holdout maximum relative error '
                +str(qualified['maximum_holdout_relative_error'])+' exceeds frozen 0.20 policy')
            components.append(dict(component=term['component'],parameters=p,basis='DECLARED_COMPONENT_MODEL',
                median_ns=cost['duration_ns'],prediction=cost,
                source_model=dict(artifact=reference,locator='#/coefficients'),
                source_validation=dict(artifact=reference,locator='#/local_models/'+str(group)+'/holdout'),
                source_provenance=term['source_ref'],anonymous_graph_derived_after_calibration=True,
                source_graph_hash_is_not_primitive_key=True,measured_target_component=False,
                precision_policy=_POLICY,maximum_holdout_relative_error=qualified['maximum_holdout_relative_error'],
                physical_optimality_proven=False,assumptions=_ASSUMPTIONS))
        except ValueError as error:
            unsupported.append(dict(component=term['component'],parameters=term['parameters'],reason=str(error)))
            if proposal is not None:proposals.append(dict(**proposal,rejection_reason=str(error)))
    result=dict(schema='scar.stateful-resource-calibration',schema_version=1,resource_context=demand['resource_context'],
        components=components,unsupported_components=unsupported,proposed_estimates=proposals,source_model=reference,
        all_fused_components_covered=not unsupported,complete_method_cost_measured=False,
        target_workload_executed=False,target_variants_ranked=False,physical_optimality_proven=False)
    out=Path(out).resolve();_require(not out.exists(),'fresh projection required');out.mkdir(parents=True)
    _save(out/'observations.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    c=s.add_parser('calibrate');c.add_argument('--domain',required=True);c.add_argument('--candidate-id',required=True);c.add_argument('--out',required=True)
    c.add_argument('--repetitions',type=int,default=5);c.add_argument('--warmup',type=int,default=2);c.add_argument('--batch',type=int,default=8)
    c=s.add_parser('project');c.add_argument('--domain',required=True);c.add_argument('--candidate-id',required=True);c.add_argument('--profile',required=True);c.add_argument('--out',required=True)
    a=p.parse_args(argv)
    if a.command=='project':
        r=predict_fused_components_v3(a.domain,a.candidate_id,a.profile,a.out)
        print(json.dumps(dict(components=len(r['components']),unsupported=r['unsupported_components']),sort_keys=True));return 0
    from scar.optimization.stateful_cpu_choices import _domain
    from scar.optimization.stateful_cpu_model import stateful_resource_context
    r=calibrate_fused_cpu_v3(stateful_resource_context(_domain(a.domain),a.candidate_id),a.out,
        repetitions=a.repetitions,warmup=a.warmup,batch=a.batch)
    print(json.dumps(dict(profile=str(Path(a.out).resolve()/'profile.json'),local_models={k:dict(admitted=v['admitted'],
        maximum_holdout_relative_error=v['maximum_holdout_relative_error']) for k,v in r['local_models'].items()}),sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
