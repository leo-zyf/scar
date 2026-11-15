"""Forward component composition for a fixed stateful CPU implementation.

Canonical resource measurements are not target-program trials. Composition is
an explicit effective model, not an additivity theorem or a physical optimum.
Unknown components are retained and prevent exporting a complete cost profile.
"""
from copy import deepcopy
import math
from pathlib import Path
import sys

from scar.ir.execution_recipe import artifact_reference,read_artifact
from scar.compiler.stateful_model import stateful_from_dict
from .deployment_cost import _fact,_profile
from .resource_solver import semantic_sha256
from .stateful_cpu_choices import _domain,stateful_cpu_cost_context


def component_key(component,parameters):
    return semantic_sha256(dict(component=component,parameters=parameters))


def stateful_resource_context(domain,candidate_id):
    """Hardware/backend applicability, independent of program/region/domain IDs."""
    row=next((r for r in domain['candidates'] if r['id']==candidate_id),None)
    if row is None and candidate_id!='NO_OP':raise ValueError('unknown CPU candidate')
    backend=domain['backend_capabilities']['serial' if row is None or row['worker_count']==1 else 'parallel']
    return dict(target_sha256=domain['target_snapshot']['sha256'],
        executor_id=None if row is None else row['binding']['executor_id'],
        worker_cpu_ids=[] if row is None else row['binding']['cpu_ids'],
        caller_affinity_cpu_ids=domain['live_constraints']['process_affinity_cpu_ids'],
        native_thread_environment=domain['live_constraints']['native_thread_environment'],
        hardware_context=domain['hardware_context'],backend=backend,
        language_runtime=dict(version=sys.version,cache_tag=sys.implementation.cache_tag),
        dtype='float64',layout='RANK2_UNIT_STEP_ROW_STRIDES',memory_policy='UNBOUND',
        regime='CANONICAL_FINITE_REPEATED_BUFFERS_CACHE_AND_NUMA_RESIDENCY_NOT_PROVEN')


def stateful_cpu_demand(domain,candidate_id):
    """Count original stages and boundary work without deriving hardware time."""
    region=stateful_from_dict(read_artifact(domain['region']));plan=read_artifact(domain['base_stage_plan'])
    context=stateful_cpu_cost_context(domain,candidate_id)
    row=next((r for r in domain['candidates'] if r['id']==candidate_id),None)
    workers=1 if row is None else row['worker_count'];nodes={n.id:n for n in region.nodes}
    shapes={};views={};origins={};available={};available_payloads={};terms=[]
    def term(component,parameters,source_ref=None):
        terms.append(dict(id='term:'+str(len(terms)),component=component,parameters=parameters,
            key=component_key(component,parameters),count=1,source_ref=source_ref))
    def view(key):
        if key in available: return available[key]
        return views.get(key)
    for node in region.nodes:
        if node.op=='input':
            shapes[node.id]=plan['input_shapes'].get(node.attrs['name'])
            if node.attrs['kind']=='array':
                views[node.id]=dict(stride=shapes[node.id][1],offset=0)
                origins[node.id]=(node.attrs['name'],node.attrs['epoch'])
        elif node.op=='const': shapes[node.id]=None
        elif node.op=='slice':
            parent=shapes[node.inputs[0]];starts=[];shape=[]
            for bound,extent in zip(node.attrs['bounds'],parent):
                start,stop,_=slice(*bound).indices(extent);starts.append(start);shape.append(max(0,stop-start))
            source=views[node.inputs[0]];shapes[node.id]=shape
            views[node.id]=dict(stride=source['stride'],offset=source['offset']+starts[0]*source['stride']+starts[1])
            origins[node.id]=origins[node.inputs[0]]
        else: shapes[node.id]=next((shapes[i] for i in node.inputs if shapes[i] is not None),None)
    if row is None:
        term('ORIGINAL_METHOD_BINDINGS_SCALARS_AND_VIEWS',dict(inputs=len(region.inputs),
            descriptors=sum(n.op=='slice' for n in region.nodes)))
        for index,action in enumerate(region.actions):
            node=nodes[action.value]
            parameters=dict(op=node.op if action.kind=='evaluate' else action.op,shape=shapes[action.value],
                operands=[view(i) if shapes[i] is not None else dict(scalar=True) for i in node.inputs])
            if action.kind=='write':
                shape=plan['input_shapes'][action.target];bounds=action.bounds or ((None,None),(None,None))
                spans=[slice(*b).indices(n) for b,n in zip(bounds,shape)]
                parameters.update(shape=[max(0,b-a) for a,b,_ in spans],target_stride=shape[1],
                    target_offset=spans[0][0]*shape[1]+spans[1][0],rhs=view(action.value) if shapes[action.value] is not None else dict(scalar=True))
            component=('NUMPY_EAGER_OPERATION_WITH_OWN_ALLOCATION_RELEASE' if shapes[action.value] is not None
                       else 'PYTHON_TAGGED_SCALAR_OPERATION') if action.kind=='evaluate' else 'NUMPY_INPLACE_WRITE'
            term(component,parameters,
                dict(artifact_role='region',pointer=f'#/actions/{index}'))
            if action.kind=='evaluate' and shapes[action.value] is not None:
                available[action.value]=dict(stride=shapes[action.value][1],offset=0)
    else:
        term('GUARDED_METHOD_AND_NATIVE_ARTIFACT_BOUNDARY',dict(arrays=sum(i.kind=='array' for i in region.inputs),
            scalars=sum(i.kind=='scalar' for i in region.inputs),callees=len(region.callees),providers=len(region.providers),
            input_path_depths=[len(i.path)-1 for i in region.inputs],
            callee_path_depths=[len(i.path)-1 for i in region.callees],
            provider_lookup_depths=[len(i.lookup_path)-1 for i in region.providers],
            array_shapes=[plan['input_shapes'][i.id] for i in region.inputs if i.kind=='array'],
            written_arrays=[i.id in plan['shadows'] for i in region.inputs if i.kind=='array'],
            workspace_slot_capacities=[r['capacity_bytes'] for r in plan['workspace_slots']]))
        capacities=[r['capacity_bytes'] for r in plan['workspace_slots']]
        term('ALLOCATE_FIXED_POOL',dict(capacities=capacities))
        for shape in plan['input_shapes'].values(): term('INPUT_FINITE_SCAN_SERIAL',dict(rows=shape[0],cols=shape[1]))
        term('CALLER_WORKER_AFFINITY_FENV_TEAM',dict(workers=workers))
        for stage in plan['stages']:
            kind=stage['kind'];shape=stage['shape'];root=stage['root_expr']
            ref=dict(artifact_role='base_stage_plan',pointer=f"#/stages/{stage['index']}")
            if kind=='SNAPSHOT_ROOT':
                term('ROW_COPY_STAGE',dict(rows=shape[0],cols=shape[1],source_stride=shape[1],source_offset=0,
                    destination_stride=shape[1],destination_offset=0,workers=workers),ref)
            elif kind=='PROVIDER':
                operand=view(nodes[root].inputs[0])
                term('NUMPY_POWER2_NATIVE_ROW_STAGE',dict(rows=shape[0],cols=shape[1],
                    source_stride=operand['stride'],source_offset=operand['offset'],workers=workers),ref)
            elif kind=='WRITE_SHADOW':
                target=stage['target'];r,c=[axis[0] for axis in stage['normalized_bounds']]
                term('INPLACE_ROW_OPERATION_STAGE',dict(op=stage['op'],rows=shape[0],cols=shape[1],
                    source_stride=shape[1],source_offset=0,destination_stride=plan['input_shapes'][target][1],
                    destination_offset=r*plan['input_shapes'][target][1]+c,workers=workers),ref)
            elif kind=='COMMIT':
                target=stage['target'];r,c=[axis[0] for axis in stage['normalized_bounds']];stride=plan['input_shapes'][target][1]
                term('ROW_COPY_SERIAL',dict(rows=shape[0],cols=shape[1],source_stride=stride,
                    source_offset=r*stride+c,destination_stride=stride,destination_offset=r*stride+c,workers=1),ref)
            elif kind=='SCALAR':
                term('TAGGED_SCALAR_SETUP_STAGE',dict(operations=[nodes[i].op for i in stage['expr_ids']
                    if nodes[i].op not in ('input','const','slice')],workers=workers),ref)
            else:
                operations=[nodes[i] for i in stage['expr_ids'] if nodes[i].op not in ('input','const','slice')]
                if len(operations)==1 and all(view(i) is not None for i in nodes[root].inputs):
                    operands=[view(i) for i in nodes[root].inputs]
                    term('ROW_OPERATION_STAGE',dict(op=nodes[root].op,rows=shape[0],cols=shape[1],
                        operands=operands,workers=workers),ref)
                else:
                    loads={};depth={};groups={}
                    def visit(key):
                        if key in depth:return depth[key]
                        descriptor=view(key)
                        if descriptor is not None:
                            identity=(str(origins.get(key,('computed',key))),descriptor['stride'],descriptor['offset'])
                            storage=str(origins.get(key,('computed',key)))
                            group=groups.setdefault(storage,len(groups))
                            loads[identity]=dict(**descriptor,storage_group=group);depth[key]=0
                        elif shapes[key] is None:depth[key]=0
                        else:depth[key]=1+max((visit(i) for i in nodes[key].inputs),default=0)
                        return depth[key]
                    span=visit(root)
                    layouts=sorted(loads.values(),key=lambda d:(d['stride'],d['offset'],d['storage_group']))
                    canonical_groups={}
                    layouts=[{**layout,'storage_group':canonical_groups.setdefault(layout['storage_group'],len(canonical_groups))}
                             for layout in layouts]
                    term('FUSED_SCALAR_DAG_ROW_STAGE',dict(rows=shape[0],cols=shape[1],workers=workers,
                        arithmetic={op:sum(n.op==op and shapes[n.id] is not None for n in operations)
                                    for op in ('add','sub','mul','div','neg')},
                        array_loads=len(loads),output_stores=1,dependent_operation_depth=span,
                        backing_storage_groups=len(canonical_groups),operand_layouts=layouts),ref)
            if stage['output_payload'] is not None and root is not None:
                available[root]=dict(stride=shape[1],offset=0)
                available_payloads[root]=stage['output_payload']
            # Match codegen's stage-local residency. A released workspace
            # value is not an available load in a later fused expression.
            released=set(stage.get('release_values_after',()))
            available={key:value for key,value in available.items()
                if available_payloads[key] not in released}
            available_payloads={key:value for key,value in available_payloads.items() if key in available}
        term('SUCCESS_CHECK_AND_CALLER_RESTORATION',dict(workers=workers))
        term('RELEASE_FIXED_POOL',dict(capacities=capacities))
    return dict(schema='scar.stateful-cpu-demand',schema_version=1,context=context,
        resource_context=stateful_resource_context(domain,candidate_id),terms=terms,
        logical_bytes_are_dram_bytes=False,source_graph_unrolled=False,
        scope='ONE_ORIGINAL_METHOD_CALL_SITE',costs_inferred=False)


def _save(path,value):
    import json
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    return artifact_reference(path,value['schema'])


def predict_stateful_cpu_model(domain,candidate_id,calibrations,*,out):
    """Compose exact matching resource signatures; leave unsupported fusion unknown."""
    domain=_domain(domain)
    return _predict_stateful_cpu_model(domain,candidate_id,calibrations,out=out)


def _predict_stateful_cpu_model(domain,candidate_id,calibrations,*,out):
    """Internal composition within an already admitted domain session."""
    demand=stateful_cpu_demand(domain,candidate_id);context=demand['context'];sources=[];records={};issues=[];rejected={}
    for value in calibrations:
        document,source=_profile(value)
        if document.get('schema')!='scar.stateful-resource-calibration' or document.get('schema_version')!=1:
            raise ValueError('stateful independent resource calibration required')
        if semantic_sha256(document['resource_context'])!=semantic_sha256(demand['resource_context']): continue
        # The complete source document is retained through matching FactRefs.
        artifact=artifact_reference(source['path'],document['schema']) if source['kind']=='PATH' else None
        if artifact is None: raise ValueError('retained calibration JSON path required')
        sources.append(artifact)
        for index,row in enumerate(document.get('proposed_estimates',())):
            if row.get('selectable') is not False:
                raise ValueError('diagnostic estimates cannot authorize selection')
            key=component_key(row['component'],row['parameters'])
            rejected.setdefault(key,[]).append(dict(
                point_estimate_ns=row.get('point_estimate_ns'),reason=row.get('rejection_reason'),
                evidence=dict(artifact=artifact,locator=f'#/proposed_estimates/{index}')))
        for index,row in enumerate(document['components']):
            key=component_key(row['component'],row['parameters'])
            if key in records: raise ValueError('ambiguous resource component observations')
            record, _=_fact(dict(artifact=artifact,locator=f'#/components/{index}'))
            if (record['basis'] not in ('DIRECT_COMPONENT_TIMING','DECLARED_COMPONENT_MODEL')
                    or record.get('selectable') is False or record.get('precision_admitted') is False
                    or record.get('exportable_as_complete_cost') is False
                    or type(record['median_ns']) not in (int,float)
                    or not math.isfinite(record['median_ns']) or record['median_ns']<0):
                raise ValueError('sourced direct or explicitly declared nonnegative resource cost required')
            records[key]=dict(duration_ns=record['median_ns'],basis=record['basis'],
                evidence=dict(artifact=artifact,locator=f'#/components/{index}'))
    evaluated=[];total=0
    for term in demand['terms']:
        record=records.get(term['key'])
        proposals=rejected.get(term['key'],[])
        if record is None: issues.append(dict(term=term['id'],component=term['component'],parameters=term['parameters'],
            reason='independent component model rejected by its admission policy' if proposals else
                'no exact-context independent component model or direct measurement',rejected_estimates=proposals))
        else:total+=record['duration_ns']*term['count']
        evaluated.append(dict(**term,status=('MODEL_REJECTED' if proposals else 'UNKNOWN')
            if record is None else 'SOURCED_EFFECTIVE_COMPONENT',measurement=record,rejected_estimates=proposals))
    out=Path(out).absolute()
    if out.exists(): raise ValueError('fresh stateful CPU model output required')
    report=dict(schema='scar.stateful-cpu-model',schema_version=1,context=context,demand=demand,
        resource_context=demand['resource_context'],components=evaluated,unresolved=issues,known_component_sum_ns=total,
        complete=not issues,duration_ticks=math.ceil(total) if not issues else None,tick_seconds=1e-9,
        source_calibrations=sources,profile=None,
        assumptions=['EFFECTIVE_COMPONENT_SUM_IS_A_MODEL_NOT_AN_ADDITIVITY_THEOREM',
            'STAGE_MEASUREMENTS_INCLUDE_THEIR_OWN_IMPLICIT_BARRIER_NO_DOUBLE_CHARGE',
            'CACHE_NUMA_ALIGNMENT_CONTENTION_REGIMES_REMAIN_APPLICABILITY_ASSUMPTIONS',
            'CANONICAL_OPERAND_VALUES_ARE_NOT_THE_PROGRAM_DATA_DISTRIBUTION',
            'RESOURCE_CONTENTION_IS_RECORDED_NOT_ASSUMED_ISOLATED',
            'NO_LINEAR_DIVISION_BY_WORKER_COUNT_NO_DRAM_TRAFFIC_INFERENCE'],
        target_workload_executed=False,target_variants_ranked=False,physical_optimality_proven=False)
    if not issues:
        facts=dict(schema='scar.stateful-cpu-model-facts',schema_version=1,context=context,
            estimate=dict(duration_ticks=report['duration_ticks'],tick_seconds=1e-9),
            derivation=dict(origin='INDEPENDENT_PRIMITIVE_COMPOSITION' if all(
                r['measurement']['basis']=='DIRECT_COMPONENT_TIMING' for r in evaluated) else 'DECLARED_FORWARD_MODEL',target_variant_trials=False,
                assumptions=report['assumptions'],component_refs=[r['measurement']['evidence'] for r in evaluated]))
        artifact=_save(out/'facts.json',facts)
        fact=lambda pointer:[dict(artifact=artifact,locator=pointer)]
        profile=dict(schema='scar.stateful-cpu-cost-profile',schema_version=1,context=context,
            context_evidence=fact('#/context'),estimate={**facts['estimate'],'evidence':fact('#/estimate')},
            derivation={**facts['derivation'],'evidence':fact('#/derivation')})
        _save(out/'profile.json',profile);report['profile']=str(out/'profile.json')
    _save(out/'model.json',report)
    return report


def build_stateful_cpu_profiles(domain,calibrations,*,out):
    """Write the mapping consumed directly by choose-stateful-cpu, including gaps."""
    domain=_domain(domain);out=Path(out).absolute()
    if out.exists():raise ValueError('fresh CPU profile-set output required')
    mapping={};reports=[];baseline=None
    for identifier in [r['id'] for r in domain['candidates']]+['NO_OP']:
        report=_predict_stateful_cpu_model(domain,identifier,calibrations,out=out/identifier)
        reports.append(dict(id=identifier,model=str(out/identifier/'model.json'),complete=report['complete'],
            known_components=sum(r['status']=='SOURCED_EFFECTIVE_COMPONENT' for r in report['components']),missing_components=len(report['unresolved'])))
        if report['profile'] is not None:
            if identifier=='NO_OP':baseline=report['profile']
            else:mapping[identifier]=report['profile']
    # Recheck original hardware/source/ABI receipts before publishing the set.
    # This avoids rebuilding the identical 106-mask domain for each estimate,
    # without introducing a global cache or accepting caller-supplied flags.
    _domain(domain)
    result=dict(schema='scar.stateful-cpu-profile-set',schema_version=1,cost_profiles=mapping,
        no_op_profile=baseline,reports=reports,complete_domain=len(mapping)==len(domain['candidates']) and baseline is not None,
        target_workload_executed=False,physical_optimality_proven=False)
    import json
    out.mkdir(parents=True,exist_ok=True)
    with (out/'profile-map.json').open('x') as stream: json.dump(mapping,stream,sort_keys=True,indent=2);stream.write('\n')
    result['cost_profiles_path']=str(out/'profile-map.json')
    _save(out/'profiles.json',result);return result


def wrap_original_call_cost(domain,observation,*,out):
    """Replay an observed original call as a forward expectation at the same site.

    This is not a native variant trial or a primitive decomposition. No load,
    setup or per-stage cost is inferred from the original complete-call total.
    """
    domain=_domain(domain);observed,source=_profile(observation)
    expected=stateful_cpu_cost_context(domain,'NO_OP')
    if (observed.get('schema'),observed.get('schema_version'))!=('scar.original-stateful-call-observation',1):
        raise ValueError('original method-call observation required')
    if (semantic_sha256(observed['context'])!=semantic_sha256(expected)
            or observed['measurement_scope']!='ONE_ORIGINAL_METHOD_CALL_SITE'
            or observed.get('target_variants_ranked') is not False
            or observed['observed_caller_affinity_cpu_ids']!=domain['live_constraints']['process_affinity_cpu_ids']
            or observed['observed_native_thread_environment']!=domain['live_constraints']['native_thread_environment']):
        raise ValueError('original method boundary/runtime context differs')
    samples=observed['samples_ns']
    import statistics
    if (not isinstance(samples,list) or len(samples)<3 or any(type(v) not in (int,float) or
            not math.isfinite(v) or v<=0 for v in samples) or observed['median_ns']!=statistics.median(samples)):
        raise ValueError('retained original samples/median differ')
    if source['kind']=='PATH': ref=artifact_reference(source['path'],observed['schema'])
    elif source['kind']=='FACT_REFERENCE':ref=source['reference']['artifact']
    else:raise ValueError('original observation must have retained artifact provenance')
    facts=dict(schema='scar.original-stateful-forward-cost',schema_version=1,context=expected,
        estimate=dict(duration_ticks=math.ceil(observed['median_ns']),tick_seconds=1e-9),
        derivation=dict(origin='DECLARED_FORWARD_MODEL',basis='OBSERVED_ORIGINAL_SAME_SCOPE_MEDIAN_AS_FORWARD_EXPECTATION',
            target_variant_trials=False,source_observation=ref,
            assumptions=['ORIGINAL_CALL_SAME_SCOPE_RUNTIME_CONTEXT_AND_SAMPLE_DISTRIBUTION_REMAIN_APPLICABLE',
                'ORIGINAL_HAS_NO_NATIVE_GUARD_DISPATCH_SHADOW_OR_COMMIT_COMPONENTS',
                'COMPLETE_CALL_OBSERVATION_NOT_SPLIT_OR_SUBTRACTED_FROM_ANOTHER_SCOPE']))
    out=Path(out).absolute()
    if out.exists():raise ValueError('fresh original cost profile output required')
    artifact=_save(out/'facts.json',facts);fact=lambda pointer:[dict(artifact=artifact,locator=pointer)]
    profile=dict(schema='scar.stateful-cpu-cost-profile',schema_version=1,context=expected,context_evidence=fact('#/context'),
        estimate={**facts['estimate'],'evidence':fact('#/estimate')},derivation={**facts['derivation'],'evidence':fact('#/derivation')})
    _save(out/'profile.json',profile);return str(out/'profile.json')


__all__=['component_key','stateful_resource_context','stateful_cpu_demand','predict_stateful_cpu_model',
    'build_stateful_cpu_profiles','wrap_original_call_cost']
