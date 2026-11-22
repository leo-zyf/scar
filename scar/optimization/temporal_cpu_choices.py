"""Hardware-derived CPU-team choices with sourced, forward component costs.

Independent facts must match the exact backend/mask/context and component.
Neither logical work nor CPU/cache capacity is converted into time. The cost
composition is an explicit model: serial terms plus the slowest worker chunk,
not an assumption that work takes serial-time divided by the worker count.
"""
from copy import deepcopy
from dataclasses import asdict, replace
from fractions import Fraction
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from scar.compiler.invariant_loop_frontend import validate_invariant_loop_proof
from scar.compiler.pipeline_frontend import prepare_pipeline_leaf
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program import target_hardware as H
from scar.ir.record_codec import encode
from scar.optimization.deployment_cost import _fact, _profile
from scar.optimization.hardware import _target_materials, propose_cpu_allocations, validate_cpu_placement
from scar.optimization.optimizer import optimize_region
from scar.optimization.resource_solver import semantic_sha256
from scar.runtime.compiled_deployment import thread_environment

_PARTITION='CONTIGUOUS_ROWS_FIRST_REMAINDER_EXPLICIT_V1'
_OPS=('add','sub','mul','div','neg')
_SCOPE='ONE_GUARDED_ORIGINAL_LOOP_CALL_DISPATCH_THROUGH_RETURN_ARGUMENT_EXPRESSIONS_ALREADY_EVALUATED'


def temporal_primitive_families():
    """Fixed source-independent registry, not populated from target programs."""
    base=[dict(op='input',inputs=[],attrs=dict(role='CARRY',argument_index=0),placement='TEMPORAL_BODY'),
        dict(op='input',inputs=[],attrs=dict(role='READONLY',argument_index=1),placement='INVARIANT_PREHEADER'),
        dict(op='mul',inputs=[1,1],attrs={},placement='INVARIANT_PREHEADER')]
    def body(nodes,output):return dict(nodes=nodes,output=output,argument_count=2,carry_argument_index=0,source_ids_are_not_primitive_keys=True)
    return dict(FIXED_SQUARE_ADD_RECURRENCE=body([*deepcopy(base),dict(op='add',inputs=[0,2],attrs={},placement='TEMPORAL_BODY')],3),
        SCALED_FIXED_SQUARE_ADD_RECURRENCE=body([*deepcopy(base),
            dict(op='const',inputs=[],attrs=dict(value=.5),placement='INVARIANT_PREHEADER'),
            dict(op='mul',inputs=[0,3],attrs={},placement='TEMPORAL_BODY'),
            dict(op='add',inputs=[4,2],attrs={},placement='TEMPORAL_BODY')],5))


def temporal_family_signature(body):
    """Normalize by value roles and ordered DAG edges, never source names/IDs."""
    if body.get('argument_count')!=2:raise ValueError('canonical temporal families require exactly two array inputs')
    nodes=body['nodes'];visited=set();active=set();cache={}
    def visit(index):
        if index in active:raise ValueError('cyclic anonymous temporal body')
        if index in cache:return cache[index]
        active.add(index);visited.add(index);node=nodes[index];op=node['op']
        if op=='input':
            if node['attrs']['role'] not in ('CARRY','READONLY'):raise ValueError('unsupported primitive input role')
            result=(op,node['attrs']['role'])
        elif op=='const':
            value=node['attrs']['value']
            if type(value) not in (int,float):raise ValueError('exact primitive numeric constant required')
            result=(op,type(value).__name__,value.hex() if type(value) is float else str(value))
        else:result=(op,node['placement'],tuple(visit(key) for key in node['inputs']))
        active.remove(index);cache[index]=result;return result
    signature=visit(body['output'])
    if visited!=set(range(len(nodes))):raise ValueError('canonical family cannot hide unused input loads or computations')
    return signature


def match_temporal_primitive_family(body):
    signature=temporal_family_signature(body)
    matched=[name for name,known in temporal_primitive_families().items() if temporal_family_signature(known)==signature]
    if len(matched)!=1:raise ValueError('anonymous ordered temporal DAG has no supported canonical primitive family')
    return dict(family=matched[0],normalization='ORDERED_VALUE_ROLE_DAG_WITH_EXACT_TYPED_CONSTANTS',
        signature_sha256=semantic_sha256(signature),source_code_ids_or_function_names_used=False)


def temporal_resource_context(snapshot_sha256,executor_id,cpu_ids,caller_affinity,native_thread_environment,backend):
    return dict(target_snapshot_sha256=snapshot_sha256,executor_id=executor_id,
        worker_cpu_ids=list(cpu_ids),caller_affinity_cpu_ids=list(caller_affinity),
        native_thread_environment=native_thread_environment,backend=backend,
        python_runtime=dict(version=sys.version,cache_tag=sys.implementation.cache_tag),dtype='float64',
        layout='C_CONTIGUOUS_EQUAL_RANK2_READONLY',memory_policy='UNBOUND',
        scope=_SCOPE,composition='SERIAL_OVERHEAD_PLUS_MAX_EXACT_WORKER_CHUNKS')


def _anonymous_body(source_body,carry_parameter,budget):
    from scar.analysis.array_motion import derive_array_motion
    execution=prepare_pipeline_leaf(optimize_region(source_body,memory_budget_bytes=budget))
    body=execution.region; ordinals={}; result=[]
    motion=derive_array_motion(body,carry_parameter=carry_parameter,
        loop_interface=dict(carry_parameter=carry_parameter,
            readonly_parameters=[name for name in body.parameters if name!=carry_parameter]))
    for index,node in enumerate(body.nodes):
        ordinals[node.id]=index
        attrs=deepcopy(node.attrs)
        if node.op=='input':
            attrs=dict(role='CARRY' if node.attrs['name']==carry_parameter else 'READONLY',
                argument_index=body.parameters.index(node.attrs['name']))
        result.append(dict(op=node.op,inputs=[ordinals[key] for key in node.inputs],attrs=attrs,
            placement='INVARIANT_PREHEADER' if motion['placements'][node.id]['temporal_placement']==
                'PRE_TEMPORAL_LOOP' else 'TEMPORAL_BODY'))
    return dict(nodes=result,output=ordinals[body.output],argument_count=len(body.parameters),
        carry_argument_index=body.parameters.index(carry_parameter),source_ids_are_not_primitive_keys=True)


def build_temporal_cpu_domain(proof,target,*,input_shapes,trip_count,memory_budget_bytes,max_threads=None):
    """Enumerate representative physical-core/SMT masks; never compile a target."""
    proof=validate_invariant_loop_proof(proof)
    return _build_temporal_body_cpu_domain(proof.body,proof.carry_parameter,target,
        proof_id=proof.id,proof_sha256=semantic_sha256(asdict(proof)),input_shapes=input_shapes,
        trip_count=trip_count,memory_budget_bytes=memory_budget_bytes,max_threads=max_threads)


def _build_temporal_body_cpu_domain(source_body,carry_parameter,target,*,proof_id,proof_sha256,
        input_shapes,trip_count,memory_budget_bytes,max_threads=None):
    """Shared derived-body/resource algorithm; this is not a source proof cast."""
    from scar.compiler.parallel_temporal_invariant import describe_parallel_temporal_capability
    if type(trip_count) is not int or not -(2**63)<=trip_count<2**63:
        raise ValueError('exact signed int64 trip count required')
    if type(memory_budget_bytes) is not int or memory_budget_bytes<8:
        raise ValueError('explicit incremental output payload budget required')
    if max_threads is not None and (type(max_threads) is not int or max_threads<1):
        raise ValueError('positive exact worker cap or None required')
    if type(input_shapes) is not dict or set(input_shapes)!=set(source_body.parameters):
        raise ValueError('complete original leaf input shape map required')
    shapes={name:list(shape) for name,shape in input_shapes.items()}
    if any(len(shape)!=2 or any(type(value) is not int or value<=0 for value in shape) for shape in shapes.values()):
        raise ValueError('positive exact rank-two input shapes required')
    shape=shapes[carry_parameter]
    if any(value!=shape for value in shapes.values()):raise ValueError('equal elementwise input shapes required')
    from scar.compiler.invariant_loop import plan_invariant_loop
    cut=plan_invariant_loop(source_body,carry_parameter=carry_parameter,
        loop_interface=dict(carry_parameter=carry_parameter,
            readonly_parameters=[name for name in source_body.parameters if name!=carry_parameter]),
        memory_budget_bytes=memory_budget_bytes)
    graph,snapshot,resources,connections=_target_materials(target)
    # A worker-count cap must not truncate the executor CPU universe: SMT
    # representatives can contain high CPU IDs even for a two-worker team.
    seeds=propose_cpu_allocations(graph,memory_budget_bytes=memory_budget_bytes,max_threads=None)
    largest={}
    for seed in seeds:
        if seed.executor_id not in largest or len(seed.cpu_ids)>len(largest[seed.executor_id].cpu_ids):largest[seed.executor_id]=seed
    backend=describe_parallel_temporal_capability()
    if backend['row_partition']!=_PARTITION:raise ValueError('exact deterministic worker row partition required')
    processors=graph.inventory.get('cpu_topology',{}).get('processors') or graph.inventory['cpu']['processors']
    rows=[];frontier=[]
    for executor_id,seed in sorted(largest.items()):
        validate_cpu_placement(seed);executor=next(row for row in resources if str(row.id)==executor_id)
        groups={};missing=[]
        for cpu in seed.cpu_ids:
            processor=next((row for row in processors if row['cpu_id']==cpu),None)
            if processor is None or processor.get('package_id') is None or processor.get('core_id') is None:missing.append(cpu)
            else:groups.setdefault((processor['package_id'],processor.get('die_id'),processor['core_id']),[]).append(cpu)
        orders={'LOGICAL_ID_PREFIX':list(seed.cpu_ids)}
        if not missing:
            ordered=[sorted(group) for _,group in sorted(groups.items(),key=lambda pair:
                (pair[0][0],-1 if pair[0][1] is None else pair[0][1],pair[0][2]))]
            orders.update(CORE_FIRST_THEN_SMT=[group[level] for level in range(max(map(len,ordered))) for group in ordered if level<len(group)],
                CORE_GROUPED_SMT_FIRST=[cpu for group in ordered for cpu in group])
        else:frontier.append(dict(executor=executor_id,cpu_ids=missing,reason='physical-core/SMT identity unavailable'))
        masks={}
        for policy,order in orders.items():
            limit=len(order) if max_threads is None else min(len(order),max_threads)
            for count in range(1,limit+1):masks.setdefault(tuple(sorted(order[:count])),[]).append(policy)
        for mask,policies in sorted(masks.items(),key=lambda pair:(len(pair[0]),pair[0])):
            binding=replace(seed,cpu_ids=mask);workers=len(mask);issues=[]
            if not backend['worker_count_bounds'][0]<=workers<=backend['worker_count_bounds'][1]:issues.append('backend worker bounds exceeded')
            if shape[0]*shape[1]*8>binding.memory_budget_bytes:issues.append('private final output exceeds effective explicit budget')
            if trip_count<=0:issues.append('nonpositive count returns original carry alias without native execution')
            physical_used=sum(bool(set(group)&set(mask)) for group in groups.values())
            identifier='temporal-cpu:'+semantic_sha256(dict(executor=executor_id,cpu_ids=mask))[:24]
            rows.append(dict(id=identifier,resource_binding=asdict(binding),cpu_threads=workers,
                representative_policies=policies,physical_core_groups_used=None if missing else physical_used,
                smt_siblings_beyond_first=None if missing else workers-physical_used,
                physical_cores_reported=executor.physical_cores,
                legal_under_guards=not issues,frontier=issues,
                capability=dict(status='SUPPORTED_UNDER_RUNTIME_GUARDS' if not issues else 'UNRESOLVED',
                    backend_sha256=semantic_sha256(backend),workers=workers,cpu_ids=list(mask)),
                lowering_admitted=False,cost_status='UNKNOWN'))
    source=asdict(source_body.source);source['origins']=[]
    result=dict(schema='scar.temporal-cpu-domain',schema_version=1,proof_id=proof_id,
        proof_sha256=proof_sha256,source_loop_original_f4_replay_required=True,
        body_sha256=semantic_sha256(asdict(source_body)),motion_analysis=cut['motion_analysis'],
        source_interface=dict(body_source=source,
            parameters=list(source_body.parameters),carry_parameter=carry_parameter),
        target_path=str(graph.path),target_snapshot=encode(snapshot),
        all_resources=[encode(H.TargetResourceRef(snapshot,row.id)) for row in resources],
        hardware_connections_retained_count=len(connections),backend=backend,
        input_shapes=shapes,shape=shape,trip_count=trip_count,memory_budget_bytes=memory_budget_bytes,
        max_threads=max_threads,body=_anonymous_body(source_body,carry_parameter,memory_budget_bytes),
        live_constraints=dict(process_affinity_cpu_ids=sorted(os.sched_getaffinity(0)),
            native_thread_environment=thread_environment()),
        candidates=rows,topology_frontier=frontier,
        runtime_guards=[*cut['Q'],'EXACT_C_CONTIGUOUS_INPUTS_AT_RUNTIME',
            'ORIGINAL_F4_AND_CALLABLE_SOURCE_IDENTITIES_MATCH','EXACT_SHAPES_COUNT_AND_C_CONTIGUOUS_INPUTS',
            'NO_INPUT_WRITES_OR_INTERMEDIATE_OBSERVERS','NO_FOREIGN_CONCURRENT_INPUT_WRITERS',
            'EXACT_WORKER_TEAM_AFFINITY_AND_FENV_RESTORATION','PRIVATE_FINAL_PAYLOAD_WITHIN_EXPLICIT_BUDGET'],
        enumeration=dict(worker_counts='EVERY_INTEGER_THROUGH_EXECUTOR_LOGICAL_CAP',
            includes_physical_core_counts=True,includes_explicit_smt_representative_masks=True,
            all_cpu_subsets_enumerated=False,cross_socket_teams=False,processes_mpi_gpu_implementations=False),
        original_no_op_available=True,serial_cpu_ids=[row['id'] for row in rows if row['cpu_threads']==1],
        selection_scope='EXPLICIT_CPP_OPENMP_PER_NUMA_MASK_FAMILY_PLUS_ORIGINAL_NO_OP',
        memory_policy='UNBOUND',physical_numa_memory_placement_proven=False,
        logical_work_divided_by_worker_count_to_infer_time=False,capacity_used_as_bandwidth=False,
        target_variants_compiled=False,target_variants_executed=False,physical_optimality_proven=False)
    result['id']='temporal-cpu-domain:'+semantic_sha256(result)
    return result


def verify_temporal_cpu_domain(proof,target,domain):
    expected=build_temporal_cpu_domain(proof,target,input_shapes=domain['input_shapes'],trip_count=domain['trip_count'],
        memory_budget_bytes=domain['memory_budget_bytes'],max_threads=domain['max_threads'])
    if semantic_sha256(expected)!=semantic_sha256(domain):raise ValueError('temporal CPU domain differs from original proof/hardware/capability replay')
    return expected


def temporal_cpu_cost_context(domain,candidate_id):
    row=next((row for row in domain['candidates'] if row['id']==candidate_id),None)
    if row is None and candidate_id!='NO_OP':raise ValueError('unknown temporal CPU implementation')
    result=dict(proof_sha256=domain['proof_sha256'],target_snapshot_sha256=domain['target_snapshot']['sha256'],
        input_shapes=domain['input_shapes'],trip_count=domain['trip_count'],memory_budget_bytes=domain['memory_budget_bytes'],
        resource_binding=None if row is None else row['resource_binding'],
        backend=None if row is None else domain['backend'],live_constraints=domain['live_constraints'],scope=_SCOPE)
    if domain.get('proof_kind')=='COMPOSITE':result.update(proof_kind='COMPOSITE',guard_profile=domain['guard_profile'])
    return result


def temporal_cpu_cost_demand(domain,candidate_id):
    """Describe serial overhead and exact worker chunks without deriving time."""
    row=next((row for row in domain['candidates'] if row['id']==candidate_id),None)
    if row is None:raise ValueError('native CPU implementation required for component demand')
    binding=row['resource_binding'];workers=row['cpu_threads'];rows,cols=domain['shape'];steps=domain['trip_count']
    context=temporal_resource_context(domain['target_snapshot']['sha256'],binding['executor_id'],binding['cpu_ids'],
        domain['live_constraints']['process_affinity_cpu_ids'],domain['live_constraints']['native_thread_environment'],domain['backend'])
    terms=[]
    def term(identifier,component,parameters,composition='SERIAL'):
        terms.append(dict(id=identifier,component=component,parameters=parameters,composition=composition,
            key=semantic_sha256(dict(component=component,parameters=parameters))))
    if domain.get('proof_kind')=='COMPOSITE':
        term('boundary','COMPOSITE_GUARDED_TEMPORAL_CALL_BOUNDARY',dict(inputs=domain['body']['argument_count'],
            shape=[rows,cols],guard_profile=domain['guard_profile']))
    else:term('boundary','GUARDED_TEMPORAL_CALL_BOUNDARY',dict(inputs=domain['body']['argument_count'],shape=[rows,cols]))
    term('allocate','NUMPY_ALLOCATE_ONE_FRESH_OUTPUT',dict(shape=[rows,cols],nbytes=8*rows*cols))
    term('team','EXACT_CPU_TEAM_AFFINITY_FENV_SETUP_AND_COMPLETION',dict(workers=workers,cpu_ids=binding['cpu_ids']))
    term('return','FINAL_GUARD_RESULT_PUBLICATION',dict(shape=[rows,cols]))
    chunks=[]
    for worker,cpu in enumerate(binding['cpu_ids']):
        begin=worker*(rows//workers)+min(worker,rows%workers);end=begin+rows//workers+(worker<rows%workers)
        parameters=dict(cpu_id=cpu,row_begin=begin,row_end=end,rows=end-begin,cols=cols,trip_count=steps,
            arithmetic_topology=domain['body'],finite_guards='INITIAL_EACH_INPUT_AND_EVERY_UPDATED_CARRY',
            counter_observers='ELEMENT_PREHEADER_AND_ITERATION_COUNTERS',implicit_join_included=False)
        term('worker:'+str(worker),'TEMPORAL_SCALAR_DAG_WORKER_CHUNK',parameters,'PARALLEL_MAX')
        chunks.append(dict(worker=worker,cpu_id=cpu,row_begin=begin,row_end=end))
    return dict(schema='scar.temporal-cpu-resource-demand',schema_version=1,candidate_id=candidate_id,
        context=temporal_cpu_cost_context(domain,candidate_id),resource_context=context,terms=terms,worker_chunks=chunks,
        composition='SUM_SERIAL_COMPONENTS_PLUS_MAX_WORKER_COMPONENTS',
        assumptions=['COMPONENT_COMPOSITION_IS_AN_EFFECTIVE_MODEL_NOT_AN_ADDITIVITY_THEOREM',
            'NO_DOUBLE_CHARGE_TEAM_JOIN_IN_WORKER_TIMINGS','CACHE_NUMA_INPUT_ALIAS_AND_FP_VALUE_DISTRIBUTION_REMAIN_APPLICABILITY_ASSUMPTIONS'],
        original_source_ids_in_primitive_keys=False,logical_bytes_are_dram_bytes=False,worker_scaling_inferred=False)


def _prepare_facts(facts):
    contexts={};references=[]
    for fact in facts:
        document,source=_profile(fact)
        if (document.get('schema') not in ('scar.temporal-resource-calibration','scar.stateful-resource-calibration')
                or document.get('schema_version')!=1):raise ValueError('independent retained resource calibration required')
        if document.get('target_workload_executed') is not False or document.get('target_variants_ranked') is not False:
            raise ValueError('independent resource facts cannot come from target workload or variant-ranking trials')
        if source['kind']!='PATH':raise ValueError('retained resource calibration JSON path required')
        reference=artifact_reference(source['path'],document['schema']);read_artifact(reference);references.append(reference)
        contexts.setdefault(semantic_sha256(document['resource_context']),[]).append((document,reference))
    return contexts,references


def _component_cost(demand,facts):
    records={};sources=[]
    for document,reference in facts.get(semantic_sha256(demand['resource_context']),()):
        sources.append(reference)
        for index,component in enumerate(document['components']):
            key=semantic_sha256(dict(component=component['component'],parameters=component['parameters']))
            if key in records:raise ValueError('ambiguous exact-context resource component')
            value=component['median_ns']
            if (component['basis'] not in ('DIRECT_COMPONENT_TIMING','DECLARED_COMPONENT_MODEL')
                    or type(value) not in (int,float) or not math.isfinite(value) or value<0
                    or component.get('selectable') is False or component.get('precision_admitted') is False
                    or component.get('exportable_as_complete_cost') is False):
                raise ValueError('nonnegative sourced and admitted component estimate required')
            records[key]=dict(duration_ns=value,evidence=dict(artifact=reference,locator=f'#/components/{index}'),basis=component['basis'])
    terms=[dict(**term,measurement=records.get(term['key'])) for term in demand['terms']]
    unresolved=[dict(id=row['id'],component=row['component'],reason='NO_EXACT_CONTEXT_INDEPENDENT_COMPONENT_FACT') for row in terms if row['measurement'] is None]
    total=None
    if not unresolved:
        serial=sum((Fraction(str(row['measurement']['duration_ns'])) for row in terms if row['composition']=='SERIAL'),Fraction())
        parallel=max((Fraction(str(row['measurement']['duration_ns'])) for row in terms if row['composition']=='PARALLEL_MAX'),default=Fraction())
        total=math.ceil(serial+parallel)
    return dict(terms=terms,unresolved=unresolved,source_facts=sources,duration_ns=total,complete=not unresolved,
        derivation='INDEPENDENT_COMPONENT_FORWARD_MODEL_SUM_SERIAL_PLUS_MAX_WORKER',target_variants_ranked=False)


def _original_cost(profile,context):
    if profile is None:return dict(duration_ns=None,complete=False,unresolved=['NO_ORIGINAL_COMPLETE_CALL_COST'])
    document,source=_profile(profile)
    if (document.get('schema'),document.get('schema_version'))!=('scar.temporal-cpu-cost-profile',1):raise ValueError('scoped temporal original call profile required')
    if semantic_sha256(document['context'])!=semantic_sha256(context):raise ValueError('original call cost context differs')
    if source['kind']!='PATH':raise ValueError('retained original call cost profile JSON required')
    reference=artifact_reference(source['path'],document['schema']);read_artifact(reference)
    value=document['duration_ns']
    if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('nonnegative original duration required')
    witnesses=document.get('cost_evidence',())
    if not witnesses or not any(_fact(witness)[0]==value for witness in witnesses):raise ValueError('original duration needs matching retained FactReference')
    if not any(semantic_sha256(_fact(witness)[0])==semantic_sha256(context)
            for witness in document.get('context_evidence',())):
        raise ValueError('original call context needs matching retained FactReference')
    if document.get('target_variants_ranked') is not False:raise ValueError('target variant ranking is outside forward cost authority')
    return dict(duration_ns=math.ceil(value),complete=True,source=reference,cost_evidence=witnesses,unresolved=[])


def plan_temporal_cpu_choices(domain,*,resource_facts=(),no_op_profile=None):
    """Select only with a complete sourced domain; partial knowledge retains NO_OP."""
    rows=[];facts,references=_prepare_facts(resource_facts)
    for candidate in domain['candidates']:
        demand=temporal_cpu_cost_demand(domain,candidate['id']);model=_component_cost(demand,facts)
        rows.append(dict(id=candidate['id'],cpu_threads=candidate['cpu_threads'],resource_binding=candidate['resource_binding'],
            legal_under_guards=candidate['legal_under_guards'],model=model,cost_status='KNOWN_FORWARD_MODEL' if model['complete'] else 'UNKNOWN'))
    original=_original_cost(no_op_profile,temporal_cpu_cost_context(domain,'NO_OP'))
    legal=[row for row in rows if row['legal_under_guards']]
    complete=not domain['topology_frontier'] and all(row['model']['complete'] for row in legal) and original['complete']
    selected=None
    if complete:
        scored=[(original['duration_ns'],0,'NO_OP')]+[(row['model']['duration_ns'],1,row['id']) for row in legal]
        selected=min(scored)[2]
    for reference in references:read_artifact(reference)
    return dict(schema='scar.temporal-cpu-choice',schema_version=1,domain_id=domain['id'],variants=rows,
        original_no_op=original,complete_cost_domain=complete,selected_id=selected,
        decision='SELECT_MODEL_IMPLEMENTATION_FOR_LOWERING' if selected not in (None,'NO_OP') else 'NO_OP',
        status='MINIMUM_WITHIN_COMPLETE_DECLARED_FORWARD_MODEL' if complete else 'UNRESOLVED_COST_DOMAIN',
        physical_optimality_proven=False,target_variants_compiled=False,target_variants_executed=False,
        selected_for_use=False,scope=domain['selection_scope'])


def preview_temporal_cpu_choice(domain,*,cpu_threads):
    """Explicit requested-team preview, separate from cost-aware selection."""
    if type(cpu_threads) is not int or cpu_threads<1:raise ValueError('positive exact requested CPU thread count required')
    matches=[row for row in domain['candidates'] if row['cpu_threads']==cpu_threads and row['legal_under_guards']]
    if not matches:raise ValueError('requested worker count has no legal hardware-derived placement')
    row=min(matches,key=lambda value:(value['resource_binding']['cpu_ids'],value['resource_binding']['executor_id']))
    return _candidate_temporal_cpu_plan(domain,row)


def _candidate_temporal_cpu_plan(domain,row):
    """Lower the exact candidate row, without reselecting by worker count."""
    cpu_threads=row['cpu_threads']
    from scar.compiler.model import CpuPlacement,SourceRef
    from scar.optimization.invariant_loop_schedule import _schedule
    graph,snapshot,resources,connections=_target_materials(domain['target_path'])
    if encode(snapshot)!=domain['target_snapshot']:raise ValueError('temporal preview target snapshot changed')
    binding=dict(row['resource_binding']);binding['cpu_ids']=tuple(binding['cpu_ids']);binding['evidence']=tuple(binding['evidence'])
    binding=CpuPlacement(**binding);validate_cpu_placement(binding)
    source=dict(domain['source_interface']['body_source']);source['origins']=tuple(source['origins'])
    proof=SimpleNamespace(id=domain['proof_id'],carry_parameter=domain['source_interface']['carry_parameter'],
        body=SimpleNamespace(parameters=domain['source_interface']['parameters'],source=SourceRef(**source)))
    shape=domain['shape'];elements=shape[0]*shape[1];steps=domain['trip_count'];operations={op:0 for op in _OPS};loads=0
    for node in domain['body']['nodes']:
        if node['op'] in _OPS:operations[node['op']]+=elements*(steps if node['placement']=='TEMPORAL_BODY' else 1)
        if node['op']=='input':loads+=elements
    logical_work=dict(arithmetic_operations=operations,logical_array_load_elements=loads,
        logical_array_store_elements=elements,logical_bytes_are_dram_bytes=False,worker_scaling_inferred=False)
    demand=temporal_cpu_cost_demand(domain,row['id'])
    schedule=_schedule(proof,binding,resources,snapshot,shape,steps,binding.memory_budget_bytes,
        'REGISTER_TEMPORAL_FUSION',logical_work)
    schedule['proof'].update(worker_team_contract=domain['backend']['team_policy'],
        row_partition=domain['backend']['row_partition'],worker_cpu_ids=list(binding.cpu_ids))
    for task in schedule['tasks']:
        if task['kind']=='COMPUTE':task['attrs'].update(worker_chunks=demand['worker_chunks'],
            cpu_threads=cpu_threads,row_partition=domain['backend']['row_partition'])
    from scar.ir.versioned_schedule import schedule_from_dict
    schedule_from_dict(schedule)
    plan=dict(schema='scar.parallel-temporal-invariant-loop-resource-plan',schema_version=1,
        domain_id=domain['id'],domain_max_threads=domain['max_threads'],candidate_id=row['id'],
        proof_id=domain['proof_id'],proof_sha256=domain['proof_sha256'],body_sha256=domain['body_sha256'],
        selected_kind='REGISTER_TEMPORAL_FUSION',status='EXPLICIT_CPU_TEAM_PREVIEW_UNDER_GUARDS',
        resource_binding=asdict(binding),cpu_threads=cpu_threads,
        input_shapes=deepcopy(domain['input_shapes']),trip_count=steps,
        memory_budget_bytes=domain['memory_budget_bytes'],effective_incremental_budget_bytes=binding.memory_budget_bytes,
        resource_schedule=schedule,all_resources=deepcopy(domain['all_resources']),
        target_path=domain['target_path'],target_snapshot=deepcopy(domain['target_snapshot']),
        backend=deepcopy(domain['backend']),worker_chunks=demand['worker_chunks'],logical_work=logical_work,
        required_contract=deepcopy(domain['runtime_guards']),input_layout_obligation='EXACT_C_CONTIGUOUS_INPUTS_AT_RUNTIME',
        external_input_size_authority='LOGICAL_INPUT_PAYLOAD_NOT_STORAGE_SPAN_PROOF',
        memory_policy='UNBOUND',numa_page_placement_proven=False,
        selection_authority='EXPLICIT_REQUESTED_THREAD_COUNT_AND_DETERMINISTIC_LOWEST_CPU_MASK',
        cost_status='UNKNOWN',profitability_status='UNPROVEN_NO_SOURCED_TIMING_MODEL',duration_seconds=None,loop_unrolled=False,
        source_loop_replacement_authorized=False,runtime_guards_required=True,
        performance_qualified=False,physical_optimality_proven=False,selected_for_use=False)
    plan['id']='parallel-temporal-resource-plan:'+semantic_sha256(plan)
    return plan


def verify_temporal_cpu_preview(proof,target,plan):
    domain=build_temporal_cpu_domain(proof,target,input_shapes=plan['input_shapes'],trip_count=plan['trip_count'],
        memory_budget_bytes=plan['memory_budget_bytes'],max_threads=plan['domain_max_threads'])
    expected=preview_temporal_cpu_choice(domain,cpu_threads=plan['cpu_threads'])
    if semantic_sha256(expected)!=semantic_sha256(plan):raise ValueError('parallel temporal preview differs from original proof/hardware/backend replay')
    return expected


def plan_selected_temporal_cpu_choice(domain,*,resource_facts=(),no_op_profile=None):
    """Bridge a complete sourced choice to exactly its winning resource plan.

    The return is ``{choice: ..., schedule: None | plan}``. A partial domain
    or an original-execution winner never creates a native schedule. Full
    entry profitability still needs separate measured validation.
    """
    resource_facts=tuple(resource_facts)
    choice=plan_temporal_cpu_choices(domain,resource_facts=resource_facts,no_op_profile=no_op_profile)
    result=dict(choice=choice,schedule=None)
    if not choice['complete_cost_domain'] or choice['selected_id'] in (None,'NO_OP'):return result
    row=next(row for row in domain['candidates'] if row['id']==choice['selected_id'])
    if not row['legal_under_guards']:raise ValueError('model selected an inadmissible temporal CPU implementation')
    plan=_candidate_temporal_cpu_plan(domain,row)
    references=[]
    for value in resource_facts:
        document,source=_profile(value)
        if source['kind']!='PATH':raise ValueError('selected resource plan requires retained fact JSON files')
        reference=artifact_reference(source['path'],document['schema']);read_artifact(reference);references.append(reference)
    document,source=_profile(no_op_profile)
    if source['kind']!='PATH':raise ValueError('selected resource plan requires retained original cost profile')
    original=artifact_reference(source['path'],document['schema']);read_artifact(original)
    selected_cost=next(value['model']['duration_ns'] for value in choice['variants'] if value['id']==row['id'])
    plan.update(schema='scar.cost-selected-temporal-cpu-resource-plan',
        status='COMPLETE_SOURCED_MODEL_SELECTED_IMPLEMENTATION_UNDER_GUARDS',
        resource_facts=references,no_op_profile=original,choice=choice,
        selection_authority='COMPLETE_SOURCED_FORWARD_MODEL_OVER_EXPLICIT_CPU_MASK_DOMAIN',
        cost_status='KNOWN_FORWARD_MODEL',resource_cost_basis='DECLARED_COMPONENT_MODEL',
        predicted_complete_call_ns=selected_cost,original_complete_call_ns=choice['original_no_op']['duration_ns'],
        complete_call_cost_scope=_SCOPE,complete_call_cost_basis='DECLARED_COMPONENT_MODEL',
        full_entry_performance_qualified=False,fresh_process_scope_measured=False)
    plan.pop('id');plan['id']='cost-selected-temporal-resource-plan:'+semantic_sha256(plan)
    result['schedule']=plan
    return result


def verify_selected_temporal_cpu_choice(proof,target,plan):
    """Rebuild the original domain, re-read all facts, and recompute the winner."""
    if (plan.get('schema'),plan.get('schema_version'))!=('scar.cost-selected-temporal-cpu-resource-plan',1):
        raise ValueError('versioned cost-selected temporal resource plan required')
    references=plan['resource_facts'];original=plan['no_op_profile']
    for reference in (*references,original):read_artifact(reference)
    domain=build_temporal_cpu_domain(proof,target,input_shapes=plan['input_shapes'],trip_count=plan['trip_count'],
        memory_budget_bytes=plan['memory_budget_bytes'],max_threads=plan['domain_max_threads'])
    result=plan_selected_temporal_cpu_choice(domain,resource_facts=[reference['uri'] for reference in references],
        no_op_profile=original['uri'])
    expected=result['schedule']
    if expected is None or semantic_sha256(expected)!=semantic_sha256(plan):
        raise ValueError('cost-selected temporal schedule differs from original full-domain cost/hardware replay')
    return expected


def __getattr__(name):
    if name in ('build_composite_temporal_cpu_domain','verify_composite_temporal_cpu_domain',
            'preview_composite_temporal_cpu_choice','verify_composite_temporal_cpu_preview',
            'plan_selected_composite_temporal_cpu_choice','verify_selected_composite_temporal_cpu_choice'):
        from scar.optimization import composite_temporal_cpu_choices
        return getattr(composite_temporal_cpu_choices,name)
    raise AttributeError(name)
