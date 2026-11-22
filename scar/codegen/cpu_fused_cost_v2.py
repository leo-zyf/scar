"""Independent shared-root/DAG-aware CPU stage calibration.

The fixed registry is compiled before projection can see a target graph.
Fresh holdouts qualify a declared effective component model, never a latency
bound or a proof that a target uses the canonical compiler execution path.
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

from .cpu_fused_cost import (_HERE, _Layout, _hash, _json, _llc, _receipt,
    _require, _save, fused_features)

_FEATURES = ('stage', 'elements', 'loads', 'addsub', 'mul', 'path_addsub',
    'path_mul', 'scalar_edges', 'array_edges', 'width', 'reused_edges',
    'row_padding', 'misaligned_loads')
_POLICY = dict(maximum_holdout_relative_error=0.20,
    authority='EXPLICIT_ENGINEERING_POLICY_NOT_A_THEORETICAL_ERROR_BOUND',
    training_loss='NONNEGATIVE_LEAST_SQUARES_OF_RELATIVE_TRAINING_RESIDUALS',
    holdout_used_for_fitting=False, holdout_used_for_model_choice=False)
_ASSUMPTIONS = [
    'FIXED_SOURCE_INDEPENDENT_DAGS_NOT_TARGET_DAGS_OR_TARGET_DATA',
    'SHARED_ROOT_CANONICAL_POINTERS_ARE_NOT_TARGET_COMPILER_PATH_IDENTITY',
    'FRESH_HOLDOUT_ERROR_IS_NOT_A_BOUND_ON_TARGET_ERROR',
    'REPEATED_WARM_CANONICAL_BUFFERS_NO_AVAILABLE_CACHE_OR_NUMA_RESIDENCY_PROOF',
    'LOGICAL_WORK_AND_BACKING_EXTENTS_ARE_NOT_MEASURED_DRAM_TRAFFIC',
    'RELATIVE_NNLS_COEFFICIENTS_ARE_EFFECTIVE_NOT_INSTRUCTION_COSTS',
    'COMPONENT_ADDITIVITY_AND_WHOLE_CALL_PROFITABILITY_REQUIRE_SEPARATE_VALIDATION']


def _case(family, loads, extra, *, seed=0, holdout=False):
    graph=[]; scalar_count=0
    sequence = ('add','sub','mul') if not holdout else ('mul','add','sub','mul','sub')
    def emit(op,left,right):
        key='v'+str(len(graph));graph.append(dict(id=key,op=op,inputs=[left,right]));return key
    def scalar(value, n):
        nonlocal scalar_count
        for i in range(n):
            token='s'+str(scalar_count);scalar_count+=1
            op=sequence[(i+seed)%len(sequence)]
            value=emit(op,value,token) if i%3 else emit(op,token,value)
        return value
    def combine(values, balanced=False):
        while len(values)>1:
            if balanced:
                values=[emit('add',values[i],values[i+1]) if i+1<len(values) else values[i]
                    for i in range(0,len(values),2)]
            else:values=[emit('add',values[0],values[1]),*values[2:]]
        return values[0]
    leaves=['x'+str(i) for i in range(loads)]
    if family=='PARALLEL_BRANCHES':
        a=scalar(combine(leaves[::2]),extra//2)
        b=scalar(combine(leaves[1::2]),extra-extra//2)
        output=emit('sub',a,b)
    elif family=='SHARED_BRANCHES':
        base=combine(leaves,True)
        a=scalar(base,extra//2);b=scalar(base,extra-extra//2)
        output=emit('add',a,b)
    else:output=scalar(combine(leaves,family=='BALANCED_FANIN'),extra)
    return dict(id='registry_'+family.lower()+'_'+str(loads)+'_'+str(extra)+'_'+str(seed),
        family=family,array_loads=loads,graph=graph,output=output,
        scalar_count=scalar_count,holdout=holdout)


def canonical_registry():
    """No argument can supply a source body, graph or workload identifier."""
    families=('SERIAL_FANIN','BALANCED_FANIN','PARALLEL_BRANCHES','SHARED_BRANCHES')
    result=[_case(f,l,n) for f in families for l in (2,4,6,8) for n in (0,4,8,12)]
    # This registry, sequences, shapes and policy are frozen before measurement.
    # These eight DAGs and all their measured rows are absent from the fit.
    result += [_case(f,l,n,seed=17+i,holdout=True) for i,(f,l,n) in enumerate((
        ('SERIAL_FANIN',3,5),('SERIAL_FANIN',7,7),
        ('BALANCED_FANIN',5,5),('BALANCED_FANIN',7,9),
        ('PARALLEL_BRANCHES',3,7),('PARALLEL_BRANCHES',7,9),
        ('SHARED_BRANCHES',5,7),('SHARED_BRANCHES',7,5)))]
    return result


def _layouts(loads,cols,groups):
    # Generic rank-2 axial and diagonal neighborhood; roots are compiler-visible.
    shifts=((1,1),(0,1),(1,2),(2,1),(1,0),(0,0),(2,2),(0,2))
    if groups==1:
        return [dict(stride=cols+2,offset=r*(cols+2)+c,storage_group=0)
            for r,c in shifts[:loads]]
    result=[dict(stride=cols,offset=0,storage_group=0)]
    return result+[dict(stride=cols+2,offset=r*(cols+2)+c,storage_group=1)
        for r,c in shifts[:loads-1]]


def dag_features(graph,output,layouts,rows,cols,workers,capacity):
    """Anonymous exact DAG metrics; identifiers only connect edges locally."""
    nodes={n['id']:n for n in graph};depth={};addpath={};mulpath={};arrays={};uses=Counter()
    operations=Counter();scalar_edges=0;array_edges=0
    for i in range(len(layouts)):
        key='x'+str(i);depth[key]=addpath[key]=mulpath[key]=0;arrays[key]=True
    for node in graph:
        _require(node['op'] in ('add','sub','mul') and len(node['inputs'])==2,
            'only strict ordered add/sub/mul DAGs are in the fixed registry')
        key=node['id'];left,right=node['inputs']
        for child in (left,right):
            if child.startswith('s') and child not in depth:
                depth[child]=addpath[child]=mulpath[child]=0;arrays[child]=False
            _require(child in depth,'acyclic topological DAG required');uses[child]+=1
        _require(arrays[left] or arrays[right],'scalar-only work belongs to its own boundary component')
        arrays[key]=True;operations[node['op']]+=1
        scalar_edges+=int(not arrays[left])+int(not arrays[right])
        array_edges+=int(arrays[left])+int(arrays[right])
        depth[key]=1+max(depth[left],depth[right])
        addpath[key]=int(node['op'] in ('add','sub'))+max(addpath[left],addpath[right])
        mulpath[key]=int(node['op']=='mul')+max(mulpath[left],mulpath[right])
    _require(output in depth,'declared DAG result absent')
    arithmetic={op:operations[op] for op in ('add','sub','mul','div','neg')}
    parameters=dict(rows=rows,cols=cols,workers=workers,arithmetic=arithmetic,array_loads=len(layouts),
        output_stores=1,dependent_operation_depth=depth[output],operand_layouts=layouts,
        backing_storage_groups=len({x['storage_group'] for x in layouts}))
    legacy,geometry=fused_features(parameters,capacity)
    # OpenMP static rows: largest exact worker chunk, not a throughput/W claim.
    elements=((rows+workers-1)//workers)*cols
    levels=Counter(depth[n['id']] for n in graph)
    values=dict(stage=1.,elements=float(elements),loads=float(elements*len(layouts)),
        addsub=float(elements*(operations['add']+operations['sub'])),mul=float(elements*operations['mul']),
        path_addsub=float(elements*addpath[output]),path_mul=float(elements*mulpath[output]),
        scalar_edges=float(elements*scalar_edges),array_edges=float(elements*array_edges),
        width=float(elements*max(levels.values(),default=0)),
        reused_edges=float(elements*sum(max(0,n-1) for n in uses.values())),
        row_padding=legacy['row_padding'],misaligned_loads=legacy['misaligned_loads'])
    return values,parameters,geometry


def _configurations(context,capacity):
    configurations=[];workers=len(context['worker_cpu_ids'])
    for case_index,case in enumerate(canonical_registry()):
        shapes=((449,577),(577,705)) if case['holdout'] else ((193,255),(385,511),(641,767))
        for groups in (1,2):
            for rows,cols in shapes:
                layouts=_layouts(case['array_loads'],cols,groups)
                features,parameters,geometry=dag_features(case['graph'],case['output'],layouts,rows,cols,workers,capacity)
                configurations.append(dict(id='configuration-'+str(len(configurations)),case_index=case_index,
                    kernel_index=case_index*2+groups-1,holdout=case['holdout'],family=case['family'],
                    groups=groups,features=features,parameters=parameters,geometry=geometry))
    random.Random(192721).shuffle(configurations)
    return configurations


def _source(context):
    original=(_HERE/'loop_v1.cpp.in').read_text();start=original.index('class ThreadAffinity {')
    affinity=original[start:original.index('\n#endif',start)]
    start=original.index('class FloatingScope {');floating=original[start:original.index('\n};',start)+3]
    functions=[];pointers=[];loads=[]
    for case in canonical_registry():
        for groups in (1,2):
            name=case['id']+'_g'+str(groups);body=[]
            for i,layout in enumerate(_layouts(case['array_loads'],100,groups)):
                padding=layout['stride']-100;r,c=divmod(layout['offset'],layout['stride'])
                body.append(f'const double x{i}=a.inputs[{layout["storage_group"]}][({r}+row)*(a.cols+{padding})+{c}+col];')
            for node in case['graph']:
                args=[('a.scalars['+key[1:]+']') if key.startswith('s') else key for key in node['inputs']]
                body.append('const double '+node['id']+'=('+args[0]+dict(add='+',sub='-',mul='*')[node['op']]+args[1]+');')
            body.append('a.output[row*a.cols+col]='+case['output']+';')
            functions.append('__attribute__((noinline)) void '+name+'(const Args &a){\n'
                '#pragma omp for schedule(static)\nfor(int64_t row=0;row<a.rows;++row)\n'
                'for(int64_t col=0;col<a.cols;++col){\n'+'\n'.join(body)+'\n}}')
            pointers.append('&'+name);loads.append(str(case['array_loads']))
    source=(_HERE/'cpu_fused_cost_v2.cpp.in').read_text()
    for key,value in dict(CPU_IDS=', '.join(map(str,context['worker_cpu_ids'])),
        CPU_SLOTS=str(max(context['worker_cpu_ids'])+1),AFFINITY_CLASS=affinity,FLOATING_CLASS=floating,
        CANONICAL_FUNCTIONS='\n'.join(functions),CANONICAL_POINTERS=', '.join(pointers),
        CANONICAL_LOADS=', '.join(loads)).items():source=source.replace('@'+key+'@',value)
    return source


def _native(context,directory):
    compiler=context['backend']['compiler'];cpus=context['worker_cpu_ids']
    _require(len(cpus)>1 and set(cpus)<=os.sched_getaffinity(0)
        and sorted(os.sched_getaffinity(0))==context['caller_affinity_cpu_ids'],
        'exact supported OpenMP worker and caller affinity required')
    identity=subprocess.run([*compiler['argv'],'--version'],check=True,capture_output=True,text=True).stdout.splitlines()[0]
    _require(identity==compiler['identity'] and 'g++' in identity and '-fopenmp' in compiler['options'],
        'exact recorded GCC/OpenMP toolchain required')
    directory.mkdir();source=directory/'canonical.cpp';source.write_text(_source(context))
    (directory/'scar_abi_v1.h').write_bytes((_HERE/'scar_abi_v1.h').read_bytes())
    library=directory/'canonical.so';remarks=directory/'vectorization.txt'
    command=[*compiler['argv'],*compiler['options'],'-fopt-info-vec-optimized='+str(remarks),str(source),'-o',str(library)]
    result=subprocess.run(command,capture_output=True,text=True);(directory/'compile.log').write_text(result.stdout+result.stderr)
    _require(result.returncode==0,'fixed canonical compilation failed: '+result.stderr[-1200:])
    assembly=subprocess.run(['objdump','-d','-C','--no-show-raw-insn',str(library)],check=True,capture_output=True,text=True).stdout
    (directory/'assembly.txt').write_text(assembly)
    handle=ctypes.CDLL(str(library));function=handle.scar_fused_cost_probe_v2
    function.argtypes=[ctypes.c_int32,ctypes.POINTER(_Layout),ctypes.c_uint64,ctypes.c_uint64,ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_int32)]
    function.restype=ctypes.c_int32
    return handle,function,dict(compiler=compiler,command=command,source=_receipt(source),library=_receipt(library),
        assembly=_receipt(directory/'assembly.txt'),vectorization_remarks=_receipt(remarks),
        shared_roots_compiler_visible=True,target_body_compiled=False,runtime_vector_branch_observed=False)


def _fit(observations,groups):
    import numpy as np
    from scipy.optimize import nnls
    train=[r for r in observations if r['groups']==groups and not r['holdout']]
    hold=[r for r in observations if r['groups']==groups and r['holdout']]
    matrix=np.asarray([[r['features'][f] for f in _FEATURES] for r in train])
    y=np.asarray([r['median_ns'] for r in train]);scales=np.max(abs(matrix),axis=0);active=scales>0
    normalized=matrix[:,active]/scales[active]
    weights=1/np.maximum(y,1)
    solved,_=nnls(normalized*weights[:,None],y*weights,maxiter=10000)
    coefficients=np.zeros(len(_FEATURES));coefficients[active]=solved/scales[active]
    predictions=[]
    for row in hold:
        value=sum(row['features'][f]*c for f,c in zip(_FEATURES,coefficients))
        predictions.append(dict(configuration=row['id'],observed_ns=row['median_ns'],predicted_ns=float(value),
            relative_error=float(abs(value-row['median_ns'])/max(row['median_ns'],1))))
    maximum=max(row['relative_error'] for row in predictions)
    return dict(coefficients={f:float(c) for f,c in zip(_FEATURES,coefficients)},
        training_count=len(train),holdout_count=len(hold),holdout=predictions,
        maximum_holdout_relative_error=maximum,admitted=bool(maximum<=_POLICY['maximum_holdout_relative_error']),
        training_maximum_relative_error=float(np.max(abs(matrix@coefficients-y)/np.maximum(y,1))),
        rank=int(np.linalg.matrix_rank(normalized)),active_features=int(sum(active)),
        feature_scales=scales.tolist(),training_feature_matrix=matrix.tolist(),precision_policy=_POLICY)


def calibrate_fused_cpu_v2(resource_context,out,*,repetitions=5,warmup=2,batch=8):
    """One native compile and fixed design; no target-program argument exists."""
    from scar.ir.execution_recipe import artifact_reference
    context=deepcopy(resource_context);out=Path(out).resolve()
    _require(not out.exists() and repetitions>=3 and warmup>=1 and batch>=1,'fresh output and repeated warm samples required')
    _require(all(os.environ.get(k)==v for k,v in context['native_thread_environment'].items()),'thread environment differs')
    capacity,cache=_llc(context);configurations=_configurations(context,capacity)
    implementation={name:_hash((_HERE/name).read_bytes()) for name in
        ('cpu_fused_cost_v2.py','cpu_fused_cost_v2.cpp.in','cpu_fused_cost.py','loop_v1.cpp.in','scar_abi_v1.h')}
    out.mkdir(parents=True);started=time.time();before=list(os.getloadavg())
    design=dict(schema='scar.cpu-fused-v2-design',schema_version=1,registry=canonical_registry(),
        configurations=configurations,resource_context=context,cache_geometry=cache,features=list(_FEATURES),
        precision_policy=_POLICY,implementation_files=implementation,repetitions=repetitions,warmup=warmup,batch=batch,
        order_seed=192721,source_independent=True,target_workload_executed=False,target_variants_ranked=False,
        prior_v1_and_topology_data_used_as_holdout=False,model_architecture_frozen_before_measurement=True)
    _save(out/'design.json',design);design_ref=artifact_reference(out/'design.json',design['schema'])
    handle,function,native=_native(context,out/'native');observations=[]
    for config in configurations:
        p=config['parameters'];layout=_Layout();layout.rows=p['rows'];layout.cols=p['cols'];layout.loads=p['array_loads']
        for i,row in enumerate(p['operand_layouts']):
            layout.strides[i]=row['stride'];layout.offsets[i]=row['offset'];layout.groups[i]=row['storage_group']
        samples=(ctypes.c_double*repetitions)();sink=ctypes.c_uint64();cpus=(ctypes.c_int32*2)()
        status=function(config['kernel_index'],ctypes.byref(layout),repetitions,warmup,batch,samples,ctypes.byref(sink),cpus)
        _require(status==0,'canonical FP/affinity/allocation guard: '+str(status))
        values=list(samples);_require(all(math.isfinite(x) and x>0 for x in values),'positive finite stage samples required')
        observations.append(dict(**config,samples_ns=values,median_ns=statistics.median(values),min_ns=min(values),
            max_ns=max(values),stdev_ns=statistics.stdev(values),checksum=sink.value,
            caller_cpu_before=cpus[0],caller_cpu_after=cpus[1]))
    raw=dict(schema='scar.cpu-fused-v2-observations',schema_version=1,frozen_design=design_ref,native=native,observations=observations)
    _save(out/'raw-observations.json',raw)
    _require(artifact_reference(out/'design.json',design_ref['schema'])==design_ref,'frozen design changed')
    _require(sorted(os.sched_getaffinity(0))==context['caller_affinity_cpu_ids']
        and all(os.environ.get(k)==v for k,v in context['native_thread_environment'].items()),'caller context changed')
    model=dict(schema='scar.cpu-fused-shared-root-model',schema_version=1,resource_context=context,
        frozen_design=design_ref,raw_observations=artifact_reference(out/'raw-observations.json',raw['schema']),
        local_models={str(g):_fit(observations,g) for g in (1,2)},features=list(_FEATURES),
        precision_policy=_POLICY,implementation_files=implementation,native=native,cache_geometry=cache,
        model_kind='DECLARED_SHARED_ROOT_RELATIVE_NNLS_DAG_MODEL',assumptions=_ASSUMPTIONS,
        timing_scope='AMORTIZED_MAX_WORKER_NATIVE_ROW_STAGE_WITH_OWN_IMPLICIT_BARRIER',
        timer_batch=batch,calibration_seconds=time.time()-started,host_load=dict(before=before,after=list(os.getloadavg()),isolated=False),
        target_workload_executed=False,target_variants_ranked=False,physical_optimality_proven=False,deployment_authorized=False)
    _save(out/'profile.json',model)
    return model


def _verify_model(path):
    from scar.ir.execution_recipe import artifact_reference,read_artifact
    reference=artifact_reference(path,'scar.cpu-fused-shared-root-model');model=read_artifact(reference)
    _require(model['schema_version']==1 and model['features']==list(_FEATURES) and model['precision_policy']==_POLICY,
        'model schema/features/frozen precision policy differs')
    design=read_artifact(model['frozen_design']);raw=read_artifact(model['raw_observations'])
    current={name:_hash((_HERE/name).read_bytes()) for name in model['implementation_files']}
    _require(current==model['implementation_files']==design['implementation_files'],'model implementation changed')
    _require(raw['frozen_design']==model['frozen_design'] and raw['native']==model['native'], 'native/design evidence differs')
    _require(design['registry']==canonical_registry() and design['precision_policy']==_POLICY
        and design['configurations']==_configurations(model['resource_context'],model['cache_geometry']['reported_capacity_bytes']),
        'fixed independent calibration design differs')
    _require(len(raw['observations'])==len(design['configurations']),'complete calibration required')
    for row,config in zip(raw['observations'],design['configurations']):
        _require(all(row[k]==v for k,v in config.items()) and len(row['samples_ns'])==design['repetitions']
            and all(math.isfinite(x) and x>0 for x in row['samples_ns'])
            and statistics.median(row['samples_ns'])==row['median_ns'],'sample/configuration replay differs')
    _require(model['local_models']=={str(g):_fit(raw['observations'],g) for g in (1,2)},'fitted coefficients/holdouts differ')
    _require(_source(model['resource_context'])==Path(model['native']['source']['path']).read_text(),'canonical emitted text differs')
    for receipt in model['native'].values():
        if type(receipt) is dict and {'path','sha256','bytes'}<=set(receipt):
            _require(_receipt(Path(receipt['path']))==receipt,'native diagnostic receipt differs')
    _require(_llc(model['resource_context'])[1]==model['cache_geometry'],'cache receipt differs')
    return reference,model,raw


def _domain_dags(domain):
    """Read anonymous ordered stage DAGs; never compile or run those DAGs."""
    from scar.ir.execution_recipe import read_artifact
    region=read_artifact(domain['region']);plan=read_artifact(domain['base_stage_plan'])
    nodes={n['id']:n for n in region['nodes']};shapes={};views={};origins={};available={};payloads={};result={}
    for node in region['nodes']:
        key=node['id'];op=node['op'];attrs=node['attrs']
        if op=='input':
            shapes[key]=plan['input_shapes'].get(attrs['name'])
            if attrs['kind']=='array':
                views[key]=dict(stride=shapes[key][1],offset=0);origins[key]=(attrs['name'],attrs['epoch'])
        elif op=='const':shapes[key]=None
        elif op=='slice':
            parent=shapes[node['inputs'][0]];spans=[slice(*b).indices(n) for b,n in zip(attrs['bounds'],parent)]
            shapes[key]=[max(0,b-a) for a,b,_ in spans];source=views[node['inputs'][0]]
            views[key]=dict(stride=source['stride'],offset=source['offset']+spans[0][0]*source['stride']+spans[1][0])
            origins[key]=origins[node['inputs'][0]]
        else:shapes[key]=next((shapes[i] for i in node['inputs'] if shapes[i] is not None),None)
    for stage in plan['stages']:
        root=stage.get('root_expr');shape=stage.get('shape')
        if stage['kind']=='COMPUTE':
            graph=[];loads={};groups={};scalars={};cache={}
            def visit(key):
                if shapes[key] is None:return scalars.setdefault(key,'s'+str(len(scalars)))
                descriptor=available.get(key,views.get(key))
                if descriptor is not None:
                    origin=str(origins.get(key,('computed',key)))
                    identity=(origin,descriptor['stride'],descriptor['offset'])
                    if identity not in loads:
                        loads[identity]=dict(token='x'+str(len(loads)),**descriptor,
                            storage_group=groups.setdefault(origin,len(groups)))
                    return loads[identity]['token']
                if key in cache:return cache[key]
                node=nodes[key];_require(node['op'] in ('add','sub','mul'),
                    'array operation is outside fixed add/sub/mul calibration')
                inputs=[visit(i) for i in node['inputs']];token='v'+str(len(graph))
                graph.append(dict(id=token,op=node['op'],inputs=inputs));cache[key]=token;return token
            output=visit(root);ordered=sorted(loads.values(),key=lambda x:(x['stride'],x['offset'],x['storage_group']))
            mapping={row['token']:'x'+str(i) for i,row in enumerate(ordered)};canonical_groups={}
            layouts=[dict(stride=row['stride'],offset=row['offset'],
                storage_group=canonical_groups.setdefault(row['storage_group'],len(canonical_groups))) for row in ordered]
            for node in graph:node['inputs']=[mapping.get(key,key) for key in node['inputs']]
            result['#/stages/'+str(stage['index'])]=dict(graph=graph,output=mapping.get(output,output),layouts=layouts)
        if stage.get('output_payload') is not None and root is not None:
            available[root]=dict(stride=shape[1],offset=0);payloads[root]=stage['output_payload']
        released=set(stage.get('release_values_after',()))
        available={key:value for key,value in available.items() if payloads[key] not in released}
        payloads={key:value for key,value in payloads.items() if key in available}
    return result


def _identified_value(fitted,features):
    import numpy as np
    from scipy.optimize import linprog
    matrix=np.asarray(fitted['training_feature_matrix']);scales=np.asarray(fitted['feature_scales']);active=scales>0
    vector=np.asarray([features[f] for f in _FEATURES]);coefficients=np.asarray([fitted['coefficients'][f] for f in _FEATURES])
    _require(np.all(vector[~active]==0),'uncalibrated DAG feature is nonzero')
    scaled=matrix[:,active]/scales[active];normal=vector[active]/scales[active]
    weights=np.linalg.lstsq(scaled.T,normal,rcond=None)[0];value=float(vector@coefficients)
    if np.linalg.norm(scaled.T@weights-normal)<=1e-8*max(np.linalg.norm(normal),1):
        return value,dict(identified=True,method='CALIBRATED_FEATURE_ROW_SPAN',scope='MODEL_NOT_PHYSICAL_ACCURACY')
    _,_,vectors=np.linalg.svd(scaled,full_matrices=False);rank=np.linalg.matrix_rank(scaled)
    basis=vectors[:rank];rhs=basis@(coefficients[active]*scales[active])
    low=linprog(normal,A_eq=basis,b_eq=rhs,bounds=(0,None),method='highs')
    high=linprog(-normal,A_eq=basis,b_eq=rhs,bounds=(0,None),method='highs')
    _require(low.success and high.success and -high.fun-low.fun<=max(1e-5,abs(value)*1e-7),
        'prediction differs over fitted nonnegative optimal face')
    return value,dict(identified=True,method='NNLS_OPTIMAL_FACE',minimum_ns=low.fun,maximum_ns=-high.fun,
        scope='MODEL_NOT_PHYSICAL_ACCURACY')


def predict_fused_components_v2(domain,candidate_id,profile,out):
    """Project fixed independent facts; target DAG is read only at this step."""
    from scar.optimization.stateful_cpu_model import _domain
    from scar.optimization.stateful_cpu_model import stateful_cpu_demand
    domain=_domain(domain);demand=stateful_cpu_demand(domain,candidate_id)
    reference,model,raw=_verify_model(profile)
    _require(model['resource_context']==demand['resource_context'],'exact hardware/compiler/runtime context differs')
    dags=_domain_dags(domain);capacity=model['cache_geometry']['reported_capacity_bytes']
    components=[];unsupported=[];proposals=[]
    for term in demand['terms']:
        if term['component']!='FUSED_SCALAR_DAG_ROW_STAGE':continue
        proposal=None
        try:
            p=term['parameters'];dag=dags[term['source_ref']['pointer']]
            features,parameters,geometry=dag_features(dag['graph'],dag['output'],dag['layouts'],
                p['rows'],p['cols'],p['workers'],capacity)
            _require(parameters==p,'anonymous graph does not replay exact public demand parameters')
            groups=p['backing_storage_groups'];fitted=model['local_models'].get(str(groups))
            _require(fitted is not None,'uncalibrated backing-root group count')
            _require(193<=p['rows']<=641 and 255<=p['cols']<=767
                and p['rows']%2==p['cols']%2==1
                and geometry['max_row_padding']<=2 and geometry['max_row_offset']<=2
                and geometry['max_col_offset']<=2
                and geometry['backing_bytes']+p['rows']*p['cols']*8<=capacity,
                'shape/parity/neighborhood/reported-cache scope is outside fixed calibration')
            train=[r for r in raw['observations'] if not r['holdout'] and r['groups']==groups]
            elements=features['elements']
            for f in _FEATURES:
                if f in ('stage','elements','row_padding','misaligned_loads'):continue
                ratios=[r['features'][f]/r['features']['elements'] for r in train]
                _require(min(ratios)-1e-10<=features[f]/elements<=max(ratios)+1e-10,
                    'DAG feature outside frozen training range: '+f)
            value,identified=_identified_value(fitted,features)
            proposal=dict(component=term['component'],parameters=p,features=features,geometry=geometry,
                point_estimate_ns=value,maximum_holdout_relative_error=fitted['maximum_holdout_relative_error'],
                precision_policy=_POLICY,identification=identified,selectable=False,measured_target_component=False)
            _require(fitted['admitted'],'fresh independent holdout maximum relative error '
                +str(fitted['maximum_holdout_relative_error'])+' exceeds frozen 0.20 policy')
            _require(math.isfinite(value) and value>=0,'invalid model duration')
            components.append(dict(component=term['component'],parameters=p,basis='DECLARED_COMPONENT_MODEL',median_ns=value,
                features=features,geometry=geometry,identification=identified,
                source_model=dict(artifact=reference,locator='#/local_models/'+str(groups)+'/coefficients'),
                source_validation=dict(artifact=reference,locator='#/local_models/'+str(groups)+'/holdout'),
                source_provenance=term['source_ref'],source_graph_hash_is_not_primitive_key=True,
                precision_policy=_POLICY,maximum_holdout_relative_error=fitted['maximum_holdout_relative_error'],
                anonymous_graph_derived_after_calibration=True,measured_target_component=False,
                physical_optimality_proven=False,assumptions=_ASSUMPTIONS))
        except ValueError as error:
            unsupported.append(dict(component=term['component'],parameters=term['parameters'],reason=str(error)))
            if proposal is not None:proposals.append(dict(**proposal,rejection_reason=str(error)))
    result=dict(schema='scar.stateful-resource-calibration',schema_version=1,resource_context=demand['resource_context'],
        components=components,unsupported_components=unsupported,proposed_estimates=proposals,
        source_model=reference,all_fused_components_covered=not unsupported,complete_method_cost_measured=False,
        target_workload_executed=False,target_variants_ranked=False,physical_optimality_proven=False)
    out=Path(out).resolve();_require(not out.exists(),'fresh projection output required');out.mkdir(parents=True)
    _save(out/'observations.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('calibrate');p.add_argument('--domain',required=True);p.add_argument('--candidate-id',required=True)
    p.add_argument('--out',required=True);p.add_argument('--repetitions',type=int,default=5)
    p.add_argument('--warmup',type=int,default=2);p.add_argument('--batch',type=int,default=8)
    p=sub.add_parser('project');p.add_argument('--domain',required=True);p.add_argument('--candidate-id',required=True)
    p.add_argument('--profile',required=True);p.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    from scar.optimization.stateful_cpu_model import _domain
    from scar.optimization.stateful_cpu_model import stateful_resource_context
    if args.command=='project':
        result=predict_fused_components_v2(args.domain,args.candidate_id,args.profile,args.out)
        print(json.dumps(dict(components=len(result['components']),unsupported=result['unsupported_components']),sort_keys=True));return 0
    model=calibrate_fused_cpu_v2(stateful_resource_context(_domain(args.domain),args.candidate_id),args.out,
        repetitions=args.repetitions,warmup=args.warmup,batch=args.batch)
    print(json.dumps(dict(profile=str(Path(args.out).resolve()/'profile.json'),
        local_models={k:dict(admitted=v['admitted'],maximum_holdout_relative_error=v['maximum_holdout_relative_error'])
            for k,v in model['local_models'].items()}),sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
