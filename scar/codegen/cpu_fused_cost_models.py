"""Versioned frozen DAG probe calibration, cold replay and cost projection."""

import argparse
from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

from scar.ir.execution_recipe import artifact_reference, read_artifact

from .cpu_fused_cost import (
    _HERE, _compile_fused_probe, _dag_layouts, _dag_probe_template, _hash,
    _interpolate, _llc, _measure_dag_probe, _receipt, _render_dag_probe, _require, _save,
    fused_features,
)


_V2_FEATURES = ('stage', 'elements', 'loads', 'addsub', 'mul', 'path_addsub',
    'path_mul', 'scalar_edges', 'array_edges', 'width', 'reused_edges',
    'row_padding', 'misaligned_loads')


_V2_POLICY = dict(maximum_holdout_relative_error=0.20,
    authority='EXPLICIT_ENGINEERING_POLICY_NOT_A_THEORETICAL_ERROR_BOUND',
    training_loss='NONNEGATIVE_LEAST_SQUARES_OF_RELATIVE_TRAINING_RESIDUALS',
    holdout_used_for_fitting=False, holdout_used_for_model_choice=False)


_V2_ASSUMPTIONS = [
    'FIXED_SOURCE_INDEPENDENT_DAGS_NOT_TARGET_DAGS_OR_TARGET_DATA',
    'SHARED_ROOT_CANONICAL_POINTERS_ARE_NOT_TARGET_COMPILER_PATH_IDENTITY',
    'FRESH_HOLDOUT_ERROR_IS_NOT_A_BOUND_ON_TARGET_ERROR',
    'REPEATED_WARM_CANONICAL_BUFFERS_NO_AVAILABLE_CACHE_OR_NUMA_RESIDENCY_PROOF',
    'LOGICAL_WORK_AND_BACKING_EXTENTS_ARE_NOT_MEASURED_DRAM_TRAFFIC',
    'RELATIVE_NNLS_COEFFICIENTS_ARE_EFFECTIVE_NOT_INSTRUCTION_COSTS',
    'COMPONENT_ADDITIVITY_AND_WHOLE_CALL_PROFITABILITY_REQUIRE_SEPARATE_VALIDATION']


_V3_POLICY=dict(maximum_holdout_relative_error=.20,
    authority='EXPLICIT_ENGINEERING_POLICY_NOT_A_THEORETICAL_ERROR_BOUND',
    coefficient_sources='SEPARATE_FIXED_CANONICAL_PRIMITIVES_ONLY',
    holdout_used_for_fitting=False,holdout_used_for_model_choice=False)


_V3_ASSUMPTIONS=[
    'CANONICAL_MEMORY_FANIN_IS_A_PROXY_NOT_ISOLATED_MEMORY_OR_DRAM_BANDWIDTH',
    'ARITHMETIC_CHAIN_COSTS_INCLUDE_LOAD_STORE_LOOP_AND_FENCE_AMORTIZATION',
    'DEPENDENT_CHAIN_PROXY_IS_NOT_INTRINSIC_CPU_INSTRUCTION_LATENCY',
    'MAX_COMPOSITION_IS_A_DECLARED_OVERLAP_MODEL_NOT_A_HARDWARE_SCHEDULING_THEOREM',
    'GEOMETRIC_INTERPOLATION_IS_NOT_A_WORKER_SCALING_OR_AVAILABLE_CACHE_CLAIM',
    'WARM_REPEATED_BUFFERS_CACHE_NUMA_AND_CONTENTION_NOT_CONTROLLED',
    'CANONICAL_VECTORIZATION_PATH_IS_NOT_TARGET_COMPILER_PATH_IDENTITY',
    'FRESH_HOLDOUT_ERRORS_ARE_NOT_BOUNDS_ON_TARGET_ERROR',
    'WHOLE_CALL_ADDITIVITY_AND_DEPLOYMENT_PROFITABILITY_REQUIRE_SEPARATE_VALIDATION']


def _dag_case(family, loads, extra, *, seed=0, holdout=False):
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


def registry_v2():
    """No argument can supply a source body, graph or workload identifier."""
    families=('SERIAL_FANIN','BALANCED_FANIN','PARALLEL_BRANCHES','SHARED_BRANCHES')
    result=[_dag_case(f,l,n) for f in families for l in (2,4,6,8) for n in (0,4,8,12)]
    # This registry, sequences, shapes and policy are frozen before measurement.
    # These eight DAGs and all their measured rows are absent from the fit.
    result += [_dag_case(f,l,n,seed=17+i,holdout=True) for i,(f,l,n) in enumerate((
        ('SERIAL_FANIN',3,5),('SERIAL_FANIN',7,7),
        ('BALANCED_FANIN',5,5),('BALANCED_FANIN',7,9),
        ('PARALLEL_BRANCHES',3,7),('PARALLEL_BRANCHES',7,9),
        ('SHARED_BRANCHES',5,7),('SHARED_BRANCHES',7,5)))]
    return result


def _configurations_v2(context, capacity, *, registry=None):
    configurations=[];workers=len(context['worker_cpu_ids'])
    for case_index,case in enumerate(registry if registry is not None else registry_v2()):
        shapes=((449,577),(577,705)) if case['holdout'] else ((193,255),(385,511),(641,767))
        for groups in (1,2):
            for rows,cols in shapes:
                layouts=_dag_layouts(case['array_loads'],cols,groups)
                features,parameters,geometry=dag_features(case['graph'],case['output'],layouts,rows,cols,workers,capacity)
                configurations.append(dict(id='configuration-'+str(len(configurations)),case_index=case_index,
                    kernel_index=case_index*2+groups-1,holdout=case['holdout'],family=case['family'],
                    groups=groups,features=features,parameters=parameters,geometry=geometry))
    random.Random(192721).shuffle(configurations)
    return configurations


def _source_v2(context, *, registry=None):
    registry = registry if registry is not None else registry_v2()
    return _render_dag_probe(_dag_probe_template(context), registry, compact=True)


def _fit_v2(observations,groups):
    import numpy as np
    from scipy.optimize import nnls
    train=[r for r in observations if r['groups']==groups and not r['holdout']]
    hold=[r for r in observations if r['groups']==groups and r['holdout']]
    matrix=np.asarray([[r['features'][f] for f in _V2_FEATURES] for r in train])
    y=np.asarray([r['median_ns'] for r in train]);scales=np.max(abs(matrix),axis=0);active=scales>0
    normalized=matrix[:,active]/scales[active]
    weights=1/np.maximum(y,1)
    solved,_=nnls(normalized*weights[:,None],y*weights,maxiter=10000)
    coefficients=np.zeros(len(_V2_FEATURES));coefficients[active]=solved/scales[active]
    predictions=[]
    for row in hold:
        value=sum(row['features'][f]*c for f,c in zip(_V2_FEATURES,coefficients))
        predictions.append(dict(configuration=row['id'],observed_ns=row['median_ns'],predicted_ns=float(value),
            relative_error=float(abs(value-row['median_ns'])/max(row['median_ns'],1))))
    maximum=max(row['relative_error'] for row in predictions)
    return dict(coefficients={f:float(c) for f,c in zip(_V2_FEATURES,coefficients)},
        training_count=len(train),holdout_count=len(hold),holdout=predictions,
        maximum_holdout_relative_error=maximum,admitted=bool(maximum<=_V2_POLICY['maximum_holdout_relative_error']),
        training_maximum_relative_error=float(np.max(abs(matrix@coefficients-y)/np.maximum(y,1))),
        rank=int(np.linalg.matrix_rank(normalized)),active_features=int(sum(active)),
        feature_scales=scales.tolist(),training_feature_matrix=matrix.tolist(),precision_policy=_V2_POLICY)


def _identified_value_v2(fitted,features):
    import numpy as np
    from scipy.optimize import linprog
    matrix=np.asarray(fitted['training_feature_matrix']);scales=np.asarray(fitted['feature_scales']);active=scales>0
    vector=np.asarray([features[f] for f in _V2_FEATURES]);coefficients=np.asarray([fitted['coefficients'][f] for f in _V2_FEATURES])
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


def registry_v3():
    result=[dict(id='primitive_barrier',role='BARRIER',array_loads=0,graph=[],output=None,scalar_count=0,holdout=False)]
    for loads in (1,2,4,6,8):
        c=_dag_case('BALANCED_FANIN',loads,0);c.update(id='primitive_fanin_'+str(loads),role='MEMORY_PROXY')
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
        case=_dag_case(family,loads,extra,seed=61+i,holdout=True)
        case.update(role='FRESH_HOLDOUT');result.append(case)
    return result


def _configurations_v3(context, _capacity=None, *, registry=None):
    rows=[];workers=len(context['worker_cpu_ids'])
    for index,case in enumerate(registry if registry is not None else registry_v3()):
        shapes=((472,608),(601,735)) if case['holdout'] else ((256,320),(513,575),(770,832))
        groups=(1,2) if case['role'] in ('MEMORY_PROXY','FRESH_HOLDOUT') else (1,)
        if case['role']=='BARRIER':shapes=((1,1),)
        for group in groups:
            if group==2 and case['array_loads']==1:continue
            for r,c in shapes:
                layouts=_dag_layouts(case['array_loads'],c,group)
                counts=Counter(n['op'] for n in case['graph'])
                rows.append(dict(id='v3-configuration-'+str(len(rows)),case_index=index,kernel_index=index*2+group-1,
                    role=case['role'],operation=case.get('operation'),holdout=case['holdout'],groups=group,
                    rows=r,cols=c,workers=workers,elements=((r+workers-1)//workers)*c,
                    array_loads=case['array_loads'],layouts=layouts,arithmetic={op:counts[op] for op in ('add','sub','mul')}))
    random.Random(208199).shuffle(rows);return rows


def _source_v3(context, *, registry=None):
    registry = registry if registry is not None else registry_v3()
    return _render_dag_probe(_dag_probe_template(context, expanded=True), registry,
                             boundary_bodies={'BARRIER': '#pragma omp barrier\n'})


def _derive_v3(observations):
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


def _predict_v3(case,config,coefficients):
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


def _qualify_v3(observations, coefficients, *, registry=None):
    registry = registry if registry is not None else registry_v3()
    result={}
    for group in (1,2):
        holdout=[]
        for row in observations:
            if row['role']!='FRESH_HOLDOUT' or row['groups']!=group:continue
            cost=_predict_v3(registry[row['case_index']],row,coefficients)
            holdout.append(dict(configuration=row['id'],observed_ns=row['median_ns'],prediction=cost,
                relative_error=abs(cost['duration_ns']-row['median_ns'])/row['median_ns']))
        maximum=max(r['relative_error'] for r in holdout)
        result[str(group)]=dict(holdout=holdout,maximum_holdout_relative_error=maximum,
            admitted=maximum<=_V3_POLICY['maximum_holdout_relative_error'],precision_policy=_V3_POLICY)
    return result


def predict_fused_components_v2(domain,candidate_id,profile,out):
    """Project fixed independent facts; target DAG is read only at this step."""
    from scar.optimization.stateful_cpu_model import _domain
    from scar.optimization.stateful_cpu_model import stateful_cpu_demand
    domain=_domain(domain);demand=stateful_cpu_demand(domain,candidate_id)
    reference,model,raw=verify_fused_model(profile, version=2)
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
            for f in _V2_FEATURES:
                if f in ('stage','elements','row_padding','misaligned_loads'):continue
                ratios=[r['features'][f]/r['features']['elements'] for r in train]
                _require(min(ratios)-1e-10<=features[f]/elements<=max(ratios)+1e-10,
                    'DAG feature outside frozen training range: '+f)
            value,identified=_identified_value_v2(fitted,features)
            proposal=dict(component=term['component'],parameters=p,features=features,geometry=geometry,
                point_estimate_ns=value,maximum_holdout_relative_error=fitted['maximum_holdout_relative_error'],
                precision_policy=_V2_POLICY,identification=identified,selectable=False,measured_target_component=False)
            _require(fitted['admitted'],'fresh independent holdout maximum relative error '
                +str(fitted['maximum_holdout_relative_error'])+' exceeds frozen 0.20 policy')
            _require(math.isfinite(value) and value>=0,'invalid model duration')
            components.append(dict(component=term['component'],parameters=p,basis='DECLARED_COMPONENT_MODEL',median_ns=value,
                features=features,geometry=geometry,identification=identified,
                source_model=dict(artifact=reference,locator='#/local_models/'+str(groups)+'/coefficients'),
                source_validation=dict(artifact=reference,locator='#/local_models/'+str(groups)+'/holdout'),
                source_provenance=term['source_ref'],source_graph_hash_is_not_primitive_key=True,
                precision_policy=_V2_POLICY,maximum_holdout_relative_error=fitted['maximum_holdout_relative_error'],
                anonymous_graph_derived_after_calibration=True,measured_target_component=False,
                physical_optimality_proven=False,assumptions=_V2_ASSUMPTIONS))
        except ValueError as error:
            unsupported.append(dict(component=term['component'],parameters=term['parameters'],reason=str(error)))
            if proposal is not None:proposals.append(dict(**proposal,rejection_reason=str(error)))
    result=dict(schema='scar.stateful-resource-calibration',schema_version=1,resource_context=demand['resource_context'],
        components=components,unsupported_components=unsupported,proposed_estimates=proposals,
        source_model=reference,all_fused_components_covered=not unsupported,complete_method_cost_measured=False,
        target_workload_executed=False,target_variants_ranked=False,physical_optimality_proven=False)
    out=Path(out).resolve();_require(not out.exists(),'fresh projection output required');out.mkdir(parents=True)
    _save(out/'observations.json',result);return result


@dataclass(frozen=True)
class _Profile:
    schema: str
    registry: Callable
    configurations: Callable
    render: Callable
    fit: Callable
    policy: dict
    assumptions: list
    design: dict
    model: dict


def _replay_v2_fit(observations, _registry):
    return dict(local_models={str(group): _fit_v2(observations, group) for group in (1, 2)})


def _replay_v3_fit(observations, registry):
    coefficients = _derive_v3(observations)
    return dict(coefficients=coefficients, local_models=_qualify_v3(observations, coefficients, registry=registry))


_PROFILES = {
    2: _Profile('scar.cpu-fused-shared-root-model', registry_v2, _configurations_v2, _source_v2,
                _replay_v2_fit, _V2_POLICY, _V2_ASSUMPTIONS,
                dict(features=list(_V2_FEATURES), order_seed=192721, source_independent=True,
                     prior_v1_and_topology_data_used_as_holdout=False,
                     model_architecture_frozen_before_measurement=True),
                dict(features=list(_V2_FEATURES), model_kind='DECLARED_SHARED_ROOT_RELATIVE_NNLS_DAG_MODEL',
                     timing_scope='AMORTIZED_MAX_WORKER_NATIVE_ROW_STAGE_WITH_OWN_IMPLICIT_BARRIER')),
    3: _Profile('scar.cpu-fused-bottleneck-model', registry_v3, _configurations_v3, _source_v3,
                _replay_v3_fit, _V3_POLICY, _V3_ASSUMPTIONS,
                dict(order_seed=208199, composition='BARRIER_PLUS_MAX_MEMORY_ARITHMETIC_DEPENDENCY',
                     coefficient_derivation='MEDIAN_EFFECTIVE_CANONICAL_UNIT_COSTS_AND_PIECEWISE_GEOMETRY_INTERPOLATION',
                     model_frozen_before_measurement=True, inspected_v1_v2_observations_are_not_new_holdouts=True),
                dict(model_kind='DECLARED_CANONICAL_PROXY_BARRIER_PLUS_MAX')),
}
_CALIBRATION_FILES = (Path(__file__).name, 'cpu_fused_cost.py', 'cpu_fused_cost_v2.cpp.in',
                      'loop_v1.cpp.in', 'scar_abi_v1.h')


def _implementation():
    return {name: _hash((_HERE / name).read_bytes()) for name in _CALIBRATION_FILES}


def _matching_environment(context):
    return all(os.environ.get(key) == value for key, value in context['native_thread_environment'].items())


def calibrate_fused_model(resource_context, out, *, version, repetitions=5, warmup=2, batch=8):
    profile = _PROFILES[version]
    context, out = deepcopy(resource_context), Path(out).resolve()
    _require(not out.exists() and repetitions >= 3 and warmup >= 1 and batch >= 1,
             'fresh output and repeated warm samples required')
    _require(_matching_environment(context), 'thread environment differs')
    capacity, cache = _llc(context)
    registry = profile.registry()
    configurations = profile.configurations(context, capacity, registry=registry)
    implementation = _implementation()
    out.mkdir(parents=True)
    started, before = time.time(), list(os.getloadavg())
    design = dict(schema=f'scar.cpu-fused-v{version}-design', schema_version=1, registry=registry,
                  configurations=configurations, resource_context=context, cache_geometry=cache,
                  precision_policy=profile.policy, implementation_files=implementation,
                  repetitions=repetitions, warmup=warmup, batch=batch, target_workload_executed=False,
                  target_variants_ranked=False, **profile.design)
    design_path = out / 'design.json'
    _save(design_path, design)
    design_ref = artifact_reference(design_path, design['schema'])
    handle, function, native, _ = _compile_fused_probe(
        context, out / 'native', lambda resource: profile.render(resource, registry=registry), version=2)
    native.update(target_body_compiled=False, runtime_vector_branch_observed=False)
    if version == 2:
        native['shared_roots_compiler_visible'] = True
    observations = _measure_dag_probe(function, configurations, repetitions, warmup, batch)
    raw = dict(schema=f'scar.cpu-fused-v{version}-observations', schema_version=1,
               frozen_design=design_ref, native=native, observations=observations)
    raw_path = out / 'raw-observations.json'
    _save(raw_path, raw)
    _require(artifact_reference(design_path, design['schema']) == design_ref, 'frozen design changed')
    _require(sorted(os.sched_getaffinity(0)) == context['caller_affinity_cpu_ids']
             and _matching_environment(context), 'caller context changed')
    model = dict(schema=profile.schema, schema_version=1, resource_context=context, frozen_design=design_ref,
                 raw_observations=artifact_reference(raw_path, raw['schema']), precision_policy=profile.policy,
                 implementation_files=implementation, native=native, cache_geometry=cache,
                 assumptions=profile.assumptions, **profile.fit(observations, registry), **profile.model,
                 calibration_seconds=time.time() - started,
                 host_load=dict(before=before, after=list(os.getloadavg()), isolated=False),
                 target_workload_executed=False, target_variants_ranked=False,
                 physical_optimality_proven=False, deployment_authorized=False)
    if version == 2:
        model['timer_batch'] = batch
    _save(out / 'profile.json', model)
    return model


def verify_fused_model(path, *, version):
    profile = _PROFILES[version]
    reference = artifact_reference(path, profile.schema)
    model = read_artifact(reference)
    _require(model['schema_version'] == 1 and model['precision_policy'] == profile.policy,
             'model schema or frozen precision policy differs')
    if version == 2:
        _require(model['features'] == list(_V2_FEATURES), 'model features differ')
    design = read_artifact(model['frozen_design'])
    raw = read_artifact(model['raw_observations'])
    _require(design['implementation_files'] == model['implementation_files'] == _implementation(),
             'current producer differs')
    context = model['resource_context']
    capacity, cache = _llc(context)
    _require(cache == model['cache_geometry'], 'cache geometry differs')
    registry = profile.registry()
    configurations = profile.configurations(context, capacity, registry=registry)
    _require(design['registry'] == registry and design['configurations'] == configurations
             and raw['frozen_design'] == model['frozen_design'] and raw['native'] == model['native'],
             'canonical design or native evidence differs')
    if version == 2:
        _require(design['precision_policy'] == profile.policy, 'frozen design precision differs')
    observations = raw['observations']
    _require(len(observations) == len(configurations), 'incomplete design')
    for row, config in zip(observations, configurations):
        samples = row['samples_ns']
        _require(all(row[key] == value for key, value in config.items()) and len(samples) == design['repetitions']
                 and all(math.isfinite(x) and x > 0 for x in samples)
                 and statistics.median(samples) == row['median_ns'], 'canonical samples differ')
    _require(all(model[key] == value for key, value in profile.fit(observations, registry).items()),
             'coefficient or holdout replay differs')
    _require(profile.render(context, registry=registry) == Path(model['native']['source']['path']).read_text(),
             'canonical emitted text differs')
    for receipt in model['native'].values():
        if type(receipt) is dict and {'path', 'sha256', 'bytes'} <= set(receipt):
            _require(_receipt(Path(receipt['path'])) == receipt, 'native diagnostic receipt differs')
    return reference, model, raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', type=int, choices=tuple(_PROFILES), required=True)
    subcommands = parser.add_subparsers(dest='command', required=True)
    calibrate = subcommands.add_parser('calibrate')
    project = subcommands.add_parser('project')
    for command in (calibrate, project):
        command.add_argument('--domain', required=True)
        command.add_argument('--candidate-id', required=True)
        command.add_argument('--out', required=True)
    calibrate.add_argument('--repetitions', type=int, default=5)
    calibrate.add_argument('--warmup', type=int, default=2)
    calibrate.add_argument('--batch', type=int, default=8)
    project.add_argument('--profile', required=True)
    args = parser.parse_args(argv)
    if args.command == 'project':
        predict = predict_fused_components_v2 if args.version == 2 else predict_fused_components_v3
        result = predict(args.domain, args.candidate_id, args.profile, args.out)
        print(json.dumps(dict(components=len(result['components']),
                              unsupported=result['unsupported_components']), sort_keys=True))
        return 0
    from scar.optimization.stateful_cpu_model import _domain, stateful_resource_context

    context = stateful_resource_context(_domain(args.domain), args.candidate_id)
    model = calibrate_fused_model(context, args.out, version=args.version, repetitions=args.repetitions,
                                 warmup=args.warmup, batch=args.batch)
    print(json.dumps(dict(profile=str(Path(args.out).resolve() / 'profile.json'), local_models={
        key: dict(admitted=value['admitted'], maximum_holdout_relative_error=value['maximum_holdout_relative_error'])
        for key, value in model['local_models'].items()}), sort_keys=True))
    return 0


def predict_fused_components_v3(domain,candidate_id,profile,out):
    from scar.optimization.stateful_cpu_model import _domain
    from scar.optimization.stateful_cpu_model import stateful_cpu_demand
    domain=_domain(domain);demand=stateful_cpu_demand(domain,candidate_id);reference,model,_=verify_fused_model(profile, version=3)
    _require(demand['resource_context']==model['resource_context'],'exact resource/compiler/runtime context differs')
    dags=_domain_dags(domain);capacity=model['cache_geometry']['reported_capacity_bytes'];components=[];unsupported=[];proposals=[]
    for term in demand['terms']:
        if term['component']!='FUSED_SCALAR_DAG_ROW_STAGE':continue
        proposal=None
        try:
            p=term['parameters'];dag=dags[term['source_ref']['pointer']]
            _,parameters,geometry=dag_features(dag['graph'],dag['output'],dag['layouts'],p['rows'],p['cols'],p['workers'],capacity)
            _require(parameters==p,'anonymous DAG does not replay public demand')
            group=p['backing_storage_groups'];qualified=model['local_models'].get(str(group))
            _require(qualified is not None and 256<=p['rows']<=770 and 320<=p['cols']<=832,
                'uncalibrated roots/row/column geometry')
            expected=_dag_layouts(p['array_loads'],p['cols'],group)
            key=lambda r:(r['storage_group'],r['stride'],r['offset'])
            _require(sorted(p['operand_layouts'],key=key)==sorted(expected,key=key),
                'layout is outside canonical shared-root axial-neighborhood geometry')
            _require(geometry['backing_bytes']+p['rows']*p['cols']*8<=capacity,'reported cache extent regime exceeded')
            config=dict(groups=group,elements=((p['rows']+p['workers']-1)//p['workers'])*p['cols'],
                array_loads=p['array_loads'],arithmetic={op:p['arithmetic'][op] for op in ('add','sub','mul')})
            cost=_predict_v3(dag,config,model['coefficients'])
            proposal=dict(component=term['component'],parameters=p,prediction=cost,
                maximum_holdout_relative_error=qualified['maximum_holdout_relative_error'],precision_policy=_V3_POLICY,
                measured_target_component=False,selectable=False)
            _require(qualified['admitted'],'fresh independent holdout maximum relative error '
                +str(qualified['maximum_holdout_relative_error'])+' exceeds frozen 0.20 policy')
            components.append(dict(component=term['component'],parameters=p,basis='DECLARED_COMPONENT_MODEL',
                median_ns=cost['duration_ns'],prediction=cost,
                source_model=dict(artifact=reference,locator='#/coefficients'),
                source_validation=dict(artifact=reference,locator='#/local_models/'+str(group)+'/holdout'),
                source_provenance=term['source_ref'],anonymous_graph_derived_after_calibration=True,
                source_graph_hash_is_not_primitive_key=True,measured_target_component=False,
                precision_policy=_V3_POLICY,maximum_holdout_relative_error=qualified['maximum_holdout_relative_error'],
                physical_optimality_proven=False,assumptions=_V3_ASSUMPTIONS))
        except ValueError as error:
            unsupported.append(dict(component=term['component'],parameters=term['parameters'],reason=str(error)))
            if proposal is not None:proposals.append(dict(**proposal,rejection_reason=str(error)))
    result=dict(schema='scar.stateful-resource-calibration',schema_version=1,resource_context=demand['resource_context'],
        components=components,unsupported_components=unsupported,proposed_estimates=proposals,source_model=reference,
        all_fused_components_covered=not unsupported,complete_method_cost_measured=False,
        target_workload_executed=False,target_variants_ranked=False,physical_optimality_proven=False)
    out=Path(out).resolve();_require(not out.exists(),'fresh projection required');out.mkdir(parents=True)
    _save(out/'observations.json',result);return result


if __name__ == '__main__':
    raise SystemExit(main())
