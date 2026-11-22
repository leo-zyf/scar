"""Pipeline schedule."""

from copy import deepcopy
from dataclasses import asdict, replace
import os
from pathlib import Path
from scar.compiler.pipeline_frontend import document, pipeline_digest, validate_pipeline
from scar.ir.execution_recipe import read_artifact
from scar.ir.program import target_hardware as H
from scar.ir.record_codec import encode, loads
from scar.ir.versioned_schedule import (
    VersionedLogicalValue as Value,
    StorageSlot,
    StorageRegion,
    VersionedMaterialization as Material,
    VersionedTask as Task,
    VersionedTaskKind as K,
    VersionedResourceSchedule,
    schedule_document,
    validate_versioned_schedule,
)
from scar.optimization.hardware import (
    _target_materials,
    _indexed_resources,
    _record_evidence,
    _connection_evidence,
    _accesses,
)
from scar.optimization.pipeline_memory import (
    verify_pipeline_memory,
    plan_pipeline_memory,
    plan_pipeline_partial_order_memory,
)
from scar.optimization.resource_solver import semantic_sha256
from itertools import combinations
from scar.ir.program.target_hardware import CpuCore, CpuExecutor, HardwareThread
from scar.optimization.entry_mandatory_bound import CostQuery, query_key, resolve_cost


# Pipeline schedule.
# Resource-safe static waves for a closed readonly array-call DAG.
#
# Logical call outputs, reusable storage slots and execution waves are separate.
# Unknown durations do not prevent a legal dependency plan, and are never filled
# with zero or unit costs. The host must join each wave and retain its ordinary
# FP/guard rollback contract before publishing the fresh final result.
SCHEMA='scar.pipeline-resource-plan'
FAMILY='DEPENDENCY_WAVES_CPU_SERIAL_LEAVES_V1'


def _need(value,message):
    if not value:raise ValueError(message)


def _dependencies(pipeline,memory):
    values={row['id']:row for row in memory['values']};order={call.id:i for i,call in enumerate(pipeline.calls)}
    predecessors={call.id:set()for call in pipeline.calls};reasons=[]
    def edge(before,after,kind,**detail):
        _need(before!=after and order[before]<order[after],'pool reuse must not overwrite a call input or precede its consumers')
        predecessors[after].add(before);reasons.append(dict(before=before,after=after,kind=kind,**detail))
    for call in pipeline.calls:
        for value in dict.fromkeys(call.inputs):
            producer=values[value]['producer_call']
            if producer is not None:edge(producer,call.id,'RAW_VALUE',value=value)
    for slot in memory['workspace_slots']:
        versions=sorted((values[value]for value in slot['values']),key=lambda row:row['producer_index'])
        for index,new in enumerate(versions):
            for old in versions[:index]:
                edge(old['producer_call'],new['producer_call'],'WAW_SLOT_REUSE',slot=slot['id'],
                    previous_value=old['id'],next_value=new['id'])
                for consumer in old['consumer_indices']:
                    edge(pipeline.calls[consumer].id,new['producer_call'],'WAR_SLOT_REUSE',slot=slot['id'],
                        previous_value=old['id'],next_value=new['id'])
    return {key:sorted(value,key=order.get)for key,value in predecessors.items()},reasons


def _waves(pipeline,predecessors,cpu_ids):
    completed=set();pending={call.id for call in pipeline.calls};waves=[];bindings={}
    while pending:
        ready=[call.id for call in pipeline.calls if call.id in pending
            and set(predecessors[call.id])<=completed]
        _need(ready,'call data/storage dependency graph contains a cycle')
        wave=ready[:len(cpu_ids)];waves.append(wave)
        bindings.update(zip(wave,cpu_ids));completed.update(wave);pending.difference_update(wave)
    return waves,bindings


def _validate_waves(pipeline,memory,predecessors,waves,call_cpu_ids,cpu_ids):
    indices={call:i for i,wave in enumerate(waves)for call in wave}
    calls={call.id:call for call in pipeline.calls}
    _need(sum(map(len,waves))==len(indices)==len(calls) and set(indices)==set(calls),
        'waves must cover each original call exactly once')
    def slot(value):
        assignment=memory['assignment'][value]
        return ('EXTERNAL',assignment['input_index'])if assignment['kind']=='EXTERNAL_INPUT'else (
            'FINAL',0)if assignment['kind']=='FINAL_OUTPUT'else ('WORKSPACE',assignment['slot_id'])
    for wave in waves:
        _need(len(wave)<=len(cpu_ids) and [call_cpu_ids[c]for c in wave]==list(cpu_ids[:len(wave)]),
            'wave CPU bindings differ from the explicit ordered hardware budget')
        _need(len({call_cpu_ids[c]for c in wave})==len(wave),'concurrent calls overlap the same HardwareThread')
        for call in wave:
            _need(all(indices[p]<indices[call]for p in predecessors[call]),'wave starts before a data/storage predecessor joins')
            writes={slot(calls[call].output)}
            for other in wave:
                if other==call:continue
                reads={slot(value)for value in calls[other].inputs};other_writes={slot(calls[other].output)}
                _need(not writes&(reads|other_writes),'same-wave pool storage has a read/write or write/write conflict')
    return dict(complete_original_call_coverage=True,all_predecessors_in_earlier_waves=True,
        unique_cpu_per_concurrent_call=True,same_wave_storage_conflicts=False,
        external_input_aliasing_safe_only_because_all_inputs_are_readonly=True,
        implementation_barriers_are_not_source_or_storage_dependencies=True)


def _closed(pipeline):
    for call in pipeline.calls:
        effects=call.plan.proof.get('effects',{})
        _need(effects.get('input_writes')is False and effects.get('output_policy')=='fresh_each_call'
            and effects.get('output_cache')is False and effects.get('purity_basis')=='closed_operator_semantics',
            'concurrent calls require the original closed readonly/fresh-output leaf profile')
        for field in ('escapes','external_effect','rng_effect'):
            _need(field not in effects or effects[field]in(False,(),[]),
                'concurrent leaf has unknown or escaping effects: '+field)


def _target(target,cpu_ids):
    if isinstance(target,dict):
        _need(target.get('schema')=='scar.target-hardware-graph','original target ArtifactRef required')
        read_artifact(target);target=Path(target['uri']).parent
    else:
        target=Path(target).resolve()
        if target.is_file():target=target.parent
    graph,snapshot,resources,connections=_target_materials(target)
    _need(type(cpu_ids)in(list,tuple),'CPU ids require an explicit ordered list/tuple')
    selected=tuple(cpu_ids)
    _need(selected and len(set(selected))==len(selected) and all(type(cpu)is int and cpu>=0 for cpu in selected),
        'explicit ordered distinct HardwareThread CPU ids required')
    _need(hasattr(os,'sched_getaffinity'),'current CPU affinity is unavailable')
    affinity=sorted(os.sched_getaffinity(0))
    _need(set(selected)<=set(affinity),'selected CPU outside current caller affinity')
    indexed=_indexed_resources(resources);threads={r.cpu_id:r for r in resources if type(r)is H.HardwareThread}
    executors=[r for r in resources if type(r)is H.CpuExecutor]
    memories=[r for r in resources if type(r)is H.MemoryDomain and r.kind is H.MemoryKind.HOST]
    bindings=[]
    for cpu in selected:
        thread=threads.get(cpu);_need(thread is not None and thread.online is not False,'selected HardwareThread not retained/online')
        owners=[r for r in executors if cpu in r.cpu_ids];_need(len(owners)==1,'selected thread needs one original CPU executor')
        executor=owners[0];access=[]
        for memory in memories:
            links=[(i,row)for i,row in enumerate(connections)if _accesses(executor,memory,row)]
            if links:
                access.append(dict(memory=_record_evidence(graph,snapshot,memory,indexed[str(memory.id)][1]),
                    connections=[_connection_evidence(graph,snapshot,row,i)for i,row in links]))
        _need(access,'no retained executor-to-host-memory access relation')
        evidence=_record_evidence(graph,snapshot,thread,indexed[str(thread.id)][1]);evidence['kind']='hardware_thread'
        bindings.append(dict(cpu_id=cpu,thread=evidence,
            executor=_record_evidence(graph,snapshot,executor,indexed[str(executor.id)][1]),host_memory_access=access))
    return graph,encode(snapshot),selected,affinity,bindings


def _task_graph(pipeline,memory,predecessors,waves,call_cpu_ids,resource_refs):
    location='host-address-space:$CALL_PROCESS';values=[];materials=[];slots=[];tasks=[]
    digest=pipeline_digest(pipeline);metadata={row['id']:row for row in memory['values']}
    call_indices={call.id:i for i,call in enumerate(pipeline.calls)}
    task_ids={call.id:'compute:'+str(i)for i,call in enumerate(pipeline.calls)}
    budget_ref=dict(artifact_role='pipeline_memory',pointer='#/memory_budget_bytes')
    scope_ref=dict(artifact_role='pipeline_memory',pointer='#/budget_scope')
    material_ids={value:'mat:'+value for value in metadata};slot_ids={}
    for index,value in enumerate(pipeline.input_values):
        row=metadata[value];slot='formal-input:'+str(index);slot_ids[value]=slot
        slots.append(StorageSlot(slot,location,row['nbytes'],ownership='EXTERNAL',resource_refs=(dict(
            kind='FORMAL_READONLY_INPUT_DESCRIPTOR_SPAN_NOT_PHYSICAL_ALLOCATION_ID',input_index=index,
            input_aliasing='MAY_ALIAS_READ_ONLY_NO_DISTINCT_ALLOCATION_INFERRED',backing_allocation_extent='UNKNOWN'),)))
        values.append(Value(value,source_refs=tuple(deepcopy(row['source_refs'])),attrs=dict(
            input_parameter=pipeline.parameters[index],shape=row['shape'],readonly=True,
            value_identity='FORMAL_CALL_SCOPED_PARAMETER_NOT_RUNTIME_CONTENT_EQUALITY')))
        materials.append(Material(material_ids[value],value,StorageRegion(slot,0,row['nbytes']),0,external_ready=True))
    final_slot='final:output';allocations=[]
    for slot,size in [(final_slot,memory['final_bytes']),*((r['id'],r['capacity_bytes'])for r in memory['workspace_slots'])]:
        task='allocate:'+slot;allocations.append(task)
        slots.append(StorageSlot(slot,location,size,allocated_by=task,
            released_by=None if slot==final_slot else 'release:'+slot,resource_refs=resource_refs,
            budget_refs=(budget_ref,scope_ref)))
        tasks.append(Task(task,K.ALLOCATE,depends_on=tuple(allocations[-2:-1]),allocates=(slot,),
            capability='CPU_ALLOCATE',resource_refs=resource_refs,attrs=dict(capacity_bytes=size,
                policy='ENTRY_ONCE_FRESH_FINAL_AND_FIXED_POOL',establishes_value_ready=False)))
    versions={}
    for call in pipeline.calls:
        row=metadata[call.output];assignment=memory['assignment'][call.output]
        slot=final_slot if assignment['kind']=='FINAL_OUTPUT' else assignment['slot_id'];slot_ids[call.output]=slot
        version=versions.get(slot,-1)+1;versions[slot]=version
        values.append(Value(call.output,producer=task_ids[call.id],source_refs=tuple(deepcopy(row['source_refs'])),
            attrs=dict(shape=row['shape'],source_call=call.id,fresh_logical_call_result=True,
                object_or_runtime_logical_version_identity_inferred=False)))
        materials.append(Material(material_ids[call.output],call.output,StorageRegion(slot,0,row['nbytes']),
            version,ready_task=task_ids[call.id]))
        depends=list(dict.fromkeys([*allocations,*(task_ids[p]for p in predecessors[call.id])]))
        tasks.append(Task(task_ids[call.id],K.COMPUTE,
            reads=tuple(dict.fromkeys(material_ids[v]for v in call.inputs)),writes=(material_ids[call.output],),
            depends_on=tuple(depends),capability='CPU_SCALAR_DAG',resource_refs=resource_refs,
            source_refs=(dict(pipeline_sha256=digest,locator='#/calls/'+str(call_indices[call.id])),),
            attrs=dict(call_id=call.id,call_index=call_indices[call.id],cpu_id=call_cpu_ids[call.id],cpu_threads=1,
                leaf_plan_sha256=memory['calls'][call_indices[call.id]]['plan_sha256'],
                output_policy='FRESH_LOGICAL_VALUE_PRIVATE_SLOT_UNTIL_SUCCESS',input_writes=False)))
    guarded=tuple(t.id for t in tasks);fence='success:all-calls'
    tasks.append(Task(fence,K.SUCCESS_FENCE,depends_on=guarded,validates=guarded,
        capability='CPU_SUCCESS_FENCE',attrs=dict(all_workers_joined=True,
            failure='STOP_NEW_DISPATCH_JOIN_ALL_RUNNING_DISCARD_PRIVATE_OUTPUT_ORIGINAL_ONCE',
            checks=['ALL_ORIGINAL_CALL_STATUSES_SUCCESS','WORKER_FENV_AND_AFFINITY_RESTORED'])))
    releases=[]
    for row in memory['workspace_slots']:
        task='release:'+row['id'];releases.append(task)
        tasks.append(Task(task,K.RELEASE,depends_on=(fence,),releases=(row['id'],),capability='CPU_RELEASE'))
    tasks.append(Task('export:final',K.EXPORT,reads=(material_ids[pipeline.output],),
        depends_on=(fence,*releases),capability='CPU_EXPORT',attrs=dict(ownership='CALLER',fresh_output=True)))
    external=sum(metadata[value]['nbytes']for value in pipeline.input_values)
    graph=VersionedResourceSchedule(pipeline.id,tuple(values),tuple(materials),tuple(slots),tuple(tasks),
        resource_refs=resource_refs,budget_bytes=memory['memory_budget_bytes']+external,
        budget_scope='ENCODED_FORMAL_EXTERNAL_PAYLOAD_PLUS_ORIGINAL_INCREMENTAL_PIPELINE_PAYLOAD',
        budget_refs=(budget_ref,scope_ref),proof=dict(
            profile=FAMILY,source_call_order_only_relaxed_under_closed_normal_success_Q=True,
            allocation_never_establishes_readiness=True,physical_slots_are_not_logical_values=True,
            slot_versions='REUSE_ORDER_NOT_OBSERVED_RUNTIME_CONTENT_VERSION',
            external_input_aliasing='MAY_ALIAS_READ_ONLY_NO_DISTINCT_ALLOCATION_INFERRED',
            external_slots='FORMAL_READONLY_DESCRIPTOR_SPANS_NOT_OBSERVED_PHYSICAL_ALLOCATIONS',
            external_backing_allocation_extent='UNKNOWN',external_input_layout='C_CONTIGUOUS_REQUIRED',
            encoded_budget_is_algebraic_adapter_not_rss_or_physical_allocation_bound=True,
            original_incremental_payload_budget_bytes=memory['memory_budget_bytes'],
            original_pool_plus_fresh_final_bytes=memory['peak_allocation_bytes'],
            execution_waves_are_separate_dependency_preserving_refinement=True,
            final_publication_after_success_and_workspace_release=True),executable=True)
    validate_versioned_schedule(graph)
    return graph


def plan_pipeline_schedule(pipeline,memory_plan,target,*,cpu_ids,input_graph=None):
    pipeline=validate_pipeline(pipeline);_closed(pipeline)
    memory=verify_pipeline_memory(pipeline,memory_plan)
    complete_profiles={('scar.pipeline-memory-plan',1,'OPTIMAL_WITHIN_FIXED_POOL_MODEL'),
        ('scar.pipeline-partial-order-memory-plan',1,'OPTIMAL_WITHIN_RAW_PARTIAL_ORDER_POOL_MODEL')}
    _need((memory.get('schema'),memory.get('schema_version'),memory['status'])in complete_profiles
        and memory['search_budget_exhausted']is False and not memory['frontier']
        and memory['budget_satisfied']is True and memory['assignment']is not None,
        'complete original fixed-pool lifetime/shape/budget plan required')
    graph,snapshot,cpus,affinity,bindings=_target(target,cpu_ids)
    if input_graph is not None:read_artifact(input_graph,json_document=False)
    predecessors,edges=_dependencies(pipeline,memory);waves,calls=_waves(pipeline,predecessors,cpus)
    wave_validation=_validate_waves(pipeline,memory,predecessors,waves,calls,cpus)
    resources=(dict(kind='EXPLICIT_HARDWARE_THREAD_SET',target=snapshot,cpu_ids=list(cpus),
        thread_refs=[row['thread']['target']for row in bindings],
        original_fact_details_plan_pointer='#/hardware_bindings',global_exclusive_reservation_proven=False),)
    tasks=_task_graph(pipeline,memory,predecessors,waves,calls,resources)
    thread_ids={row['cpu_id']:row['thread']['record']['id']['wire']for row in bindings}
    return dict(schema=SCHEMA,schema_version=1,pipeline_sha256=pipeline_digest(pipeline),
        memory_plan_sha256=semantic_sha256(memory),target=snapshot,cpu_ids=list(cpus),
        call_predecessors=predecessors,waves=waves,call_cpu_ids=calls,task_graph=schedule_document(tasks),
        wave_validation=wave_validation,
        requires_serial_leaf_operators=True,execution_family=FAMILY,
        input_graph=input_graph,input_graph_role='PROVENANCE_ONLY_NOT_DEPENDENCY_AUTHORITY',
        dependency_authority='ORIGINAL_CALL_INPUT_VALUE_EDGES_AND_REPLAYED_POOL_STORAGE_REUSE',
        dependency_reasons=edges,wave_policy='SOURCE_INDEX_READY_PREFIX_UP_TO_CPU_COUNT_WITH_FULL_WAVE_JOIN',
        caller_affinity_cpu_ids=affinity,hardware_bindings=bindings,
        cpu_budget=dict(units='EXPLICIT_HARDWARE_THREADS',capacity=len(cpus),global_reservation=False),
        resource_capacities={thread_ids[cpu]:1 for cpu in cpus},
        call_resource_demands={call:{thread_ids[cpu]:1}for call,cpu in calls.items()},
        original_incremental_memory_budget_bytes=memory['memory_budget_bytes'],
        original_pool_plus_fresh_final_bytes=memory['peak_allocation_bytes'],
        encoded_graph_budget_bytes=tasks.budget_bytes,
        external_input_aliasing='MAY_ALIAS_READ_ONLY_NO_DISTINCT_ALLOCATION_INFERRED',
        external_input_layout='C_CONTIGUOUS_REQUIRED',external_backing_allocation_extent='UNKNOWN',
        memory_policy='UNBOUND',numa_page_residency_proven=False,
        durations={t.id:dict(status='UNKNOWN',duration_ticks=None)for t in tasks.tasks},
        normal_completion_only=True,original_internal_exception_or_allocator_sequence_preserved=False,
        source_order_independent='ONLY_CLOSED_READONLY_ARRAY_PROFILE_UNDER_ORIGINAL_GUARDS',
        failure_policy='STOP_DISPATCH_JOIN_ALL_RUNNING_DISCARD_PRIVATE_RESULTS_ORIGINAL_ONCE',
        executable_under_required_host_contract=True,model_optimum_proven=False,
        physical_optimality_proven=False,duration_predicted=False,deployment_authorized=False,
        target_program_executed=False,original_pure_call_graph_unrolled=False)


def verify_pipeline_schedule(pipeline,memory_plan,target,report):
    value=loads(Path(report).read_bytes())if isinstance(report,(str,Path))else deepcopy(report)
    _need((value.get('schema'),value.get('schema_version'))==(SCHEMA,1),'versioned pipeline resource plan required')
    expected=plan_pipeline_schedule(pipeline,memory_plan,target,cpu_ids=value['cpu_ids'],input_graph=value['input_graph'])
    _need(semantic_sha256(expected)==semantic_sha256(value),
        'pipeline resource plan differs from original calls/pool/topology/wave replay')
    return value


# Dependency-ready CPU dispatch over the original fixed-pool call graph.
#
# Each call is statically bound to one original HardwareThread. A lane chooses
# its lowest-index ready call, independently of other lanes; only the final
# success fence joins all work. This is a legal unranked trial implementation,
# not a timing model or a claim that round-robin placement is fastest.
READY_SCHEMA = 'scar.pipeline-ready-resource-plan'
READY_FAMILY = 'DEPENDENCY_READY_CPU_SERIAL_LEAVES_V1'
DISPATCH_POLICY = 'LOWEST_ORIGINAL_CALL_INDEX_READY_ON_FIXED_CPU_LANE'


def _call_bindings(pipeline, cpu_ids):
    """Deterministic placement, independent of durations and observations."""
    return {call.id: cpu_ids[index % len(cpu_ids)]
            for index, call in enumerate(pipeline.calls)}


def _ready_calls(call_order, predecessors, call_cpu_ids, completed, cpu_ids, *, busy_cpu_ids=()):
    """One dispatch decision per idle lane; unfinished queue heads do not block.

    This is the declarative policy mirrored by the host. It supplies no time,
    speculative completion, storage release, or permanent lane-order edge.
    """
    completed = set(completed)
    busy = set(busy_cpu_ids)
    result = {}
    for call in call_order:
        cpu = call_cpu_ids[call]
        if (call not in completed and cpu not in busy and cpu not in result
                and set(predecessors[call]) <= completed):
            result[cpu] = call
    _need(set(result) <= set(cpu_ids), 'dispatch references a CPU outside the explicit lanes')
    return result


def _dependency_graph(pipeline, memory, predecessors, call_cpu_ids, resource_refs):
    # The old producer already emits a wave-independent RAW/WAR/WAW graph.
    # Reuse its exact value/storage construction; specialize only the dispatch
    # proof. No source-order or implementation-wave edge is added or removed.
    graph = _task_graph(pipeline, memory, predecessors, (), call_cpu_ids, resource_refs)
    proof = dict(graph.proof)
    proof.pop('execution_waves_are_separate_dependency_preserving_refinement')
    proof.update(profile=READY_FAMILY,
        dependency_ready_dispatch_is_separate_resource_capacity_refinement=True,
        per_cpu_mutual_exclusion_is_resource_capacity_not_semantic_order=True,
        intermediate_global_wave_barriers=False,
        dispatch_policy=DISPATCH_POLICY)
    graph = replace(graph, proof=proof)
    validate_versioned_schedule(graph)
    return graph


def plan_pipeline_ready_schedule(pipeline, memory_plan, target, *, cpu_ids, input_graph=None):
    """Retain true data/storage predecessors and bind unranked CPU lanes."""
    pipeline = validate_pipeline(pipeline)
    _closed(pipeline)
    memory = verify_pipeline_memory(pipeline, memory_plan)
    complete_profiles={('scar.pipeline-memory-plan',1,'OPTIMAL_WITHIN_FIXED_POOL_MODEL'),
        ('scar.pipeline-partial-order-memory-plan',1,'OPTIMAL_WITHIN_RAW_PARTIAL_ORDER_POOL_MODEL')}
    _need((memory.get('schema'),memory.get('schema_version'),memory['status'])in complete_profiles
          and memory['search_budget_exhausted'] is False
          and not memory['frontier'] and memory['budget_satisfied'] and memory['assignment'] is not None,
          'complete original fixed-pool lifetime/shape/budget plan required')
    _, snapshot, cpus, affinity, bindings = _target(target, cpu_ids)
    if input_graph is not None:
        read_artifact(input_graph, json_document=False)
    predecessors, reasons = _dependencies(pipeline, memory)
    calls = _call_bindings(pipeline, cpus)
    resources = (dict(kind='EXPLICIT_HARDWARE_THREAD_SET', target=snapshot, cpu_ids=list(cpus),
        thread_refs=[row['thread']['target'] for row in bindings],
        original_fact_details_plan_pointer='#/hardware_bindings',
        global_exclusive_reservation_proven=False),)
    tasks = _dependency_graph(pipeline, memory, predecessors, calls, resources)
    thread_ids = {row['cpu_id']: row['thread']['record']['id']['wire'] for row in bindings}
    return dict(schema=READY_SCHEMA, schema_version=1,
        pipeline_sha256=pipeline_digest(pipeline), memory_plan_sha256=semantic_sha256(memory),
        target=snapshot, cpu_ids=list(cpus), call_predecessors=predecessors, call_cpu_ids=calls,
        call_order=[call.id for call in pipeline.calls], task_graph=schedule_document(tasks),
        requires_serial_leaf_operators=True, execution_family=READY_FAMILY,
        input_graph=input_graph, input_graph_role='PROVENANCE_ONLY_NOT_DEPENDENCY_AUTHORITY',
        dependency_authority='ORIGINAL_CALL_INPUT_VALUE_EDGES_AND_REPLAYED_POOL_STORAGE_REUSE',
        dependency_reasons=reasons,
        placement_policy='ORIGINAL_CALL_INDEX_MODULO_EXPLICIT_ORDERED_CPU_IDS_UNRANKED',
        dispatch_policy=DISPATCH_POLICY,
        dispatch_contract=dict(workers_per_cpu_lane=1,
            prerequisites='ALL_TRUE_CALL_PREDECESSORS_COMPLETED_AND_ASSIGNED_CPU_LANE_IDLE',
            pending_selection='SCAN_ALL_PENDING_ASSIGNED_CALLS_IN_ORIGINAL_INDEX_ORDER',
            unfinished_queue_head_blocks_other_ready_calls=False,
            completion_visibility='RELEASE_ACQUIRE_OR_MUTEX_SYNCHRONIZED_CALL_COMPLETION',
            wake_on='ANY_CALL_COMPLETION_OR_FAILURE',
            intermediate_wave_barriers=False,
            added_lane_order_edges=False,
            final_join_required_before_export_or_fallback=True),
        readiness_validation=dict(original_call_coverage=True,
            data_and_all_old_pool_consumers_preserved=True,
            pool_overwrite_happens_before_verified=True,
            dag_basis='EVERY_TRUE_PREDECESSOR_HAS_SMALLER_ORIGINAL_CALL_INDEX',
            progress_requires='EACH_LANE_EVENTUALLY_SCANS_ALL_PENDING_CALLS_AFTER_COMPLETION_WAKE',
            same_cpu_nonoverlap='ONE_SERIAL_WORKER_PER_EXPLICIT_HARDWARE_THREAD',
            implementation_dispatch_does_not_change_semantic_dependency_graph=True),
        caller_affinity_cpu_ids=affinity, hardware_bindings=bindings,
        cpu_budget=dict(units='EXPLICIT_HARDWARE_THREADS', capacity=len(cpus), global_reservation=False),
        resource_capacities={thread_ids[cpu]: 1 for cpu in cpus},
        call_resource_demands={call: {thread_ids[cpu]: 1} for call, cpu in calls.items()},
        original_incremental_memory_budget_bytes=memory['memory_budget_bytes'],
        original_pool_plus_fresh_final_bytes=memory['peak_allocation_bytes'],
        encoded_graph_budget_bytes=tasks.budget_bytes,
        external_input_aliasing='MAY_ALIAS_READ_ONLY_NO_DISTINCT_ALLOCATION_INFERRED',
        external_input_layout='C_CONTIGUOUS_REQUIRED', external_backing_allocation_extent='UNKNOWN',
        memory_policy='UNBOUND', numa_page_residency_proven=False,
        durations={task.id: dict(status='UNKNOWN', duration_ticks=None) for task in tasks.tasks},
        normal_completion_only=True, original_internal_exception_or_allocator_sequence_preserved=False,
        source_order_independent='ONLY_CLOSED_READONLY_ARRAY_PROFILE_UNDER_ORIGINAL_GUARDS',
        failure_policy='STOP_DISPATCH_JOIN_ALL_RUNNING_DISCARD_PRIVATE_RESULTS_ORIGINAL_ONCE',
        executable_under_required_host_contract=True, model_optimum_proven=False,
        physical_optimality_proven=False, duration_predicted=False, deployment_authorized=False,
        target_program_executed=False, original_pure_call_graph_unrolled=False)


def verify_pipeline_ready_schedule(pipeline, memory_plan, target, report):
    """Rebuild original data/storage hazards, pool versions, lanes and contract."""
    value = loads(Path(report).read_bytes()) if isinstance(report, (str, Path)) else deepcopy(report)
    _need(value.get('schema') == READY_SCHEMA and type(value.get('schema_version')) is int
          and value['schema_version'] == 1, 'versioned dependency-ready resource plan required')
    expected = plan_pipeline_ready_schedule(pipeline, memory_plan, target,
        cpu_ids=value['cpu_ids'], input_graph=value['input_graph'])
    _need(semantic_sha256(expected) == semantic_sha256(value),
          'ready resource plan differs from original calls/pool/topology/dispatch replay')
    return value


# Finite CPU resource scenarios over replayed caller values and arena plans.
def _eligible_cpu_sets(target, affinity):
    graph, snapshot, resources, _ = _target_materials(target)
    indexed = _indexed_resources(resources)
    online = {x.cpu_id: x for x in resources if type(x) is HardwareThread and x.online is not False}
    available = set(affinity) & online.keys()
    cores = {}
    for core in (x for x in resources if type(x) is CpuCore):
        for cpu in available.intersection(core.cpu_ids):
            cores.setdefault(cpu, []).append(core)
    executors = [x for x in resources if type(x) is CpuExecutor]
    per_cpu = {cpu: [executor for executor in executors if cpu in executor.cpu_ids]
               for cpu in sorted(available)}
    admitted = [cpu for cpu, owners in per_cpu.items() if len(owners) == 1]
    pair = next((pair for pair in combinations(admitted, 2)
        if all(len(cores.get(cpu, ())) == 1 for cpu in pair)
        and cores[pair[0]][0].id != cores[pair[1]][0].id
        and per_cpu[pair[0]][0].numa_node is not None
        and per_cpu[pair[0]][0].numa_node == per_cpu[pair[1]][0].numa_node), ())
    serial = pair[:1] or tuple(admitted[:1])
    def evidence(cpu):
        owner, = per_cpu[cpu]
        cpu_cores = cores.get(cpu, ())
        return dict(cpu_id=cpu, numa_node=owner.numa_node,
            thread=_record_evidence(graph, snapshot, online[cpu], indexed[str(online[cpu].id)][1]),
            executor=_record_evidence(graph, snapshot, owner, indexed[str(owner.id)][1]),
            core=_record_evidence(graph, snapshot, cpu_cores[0], indexed[str(cpu_cores[0].id)][1])
            if len(cpu_cores) == 1 else None)
    return serial, pair, {cpu: evidence(cpu) for cpu in set(serial) | set(pair)}


def _ordered_calls(ready):
    """The diagnostic host realizes fixed lane heads and true predecessors."""
    lanes = {str(cpu): [] for cpu in ready['cpu_ids']}
    ancestors, ranks, combined = {}, {}, {}
    previous = {}
    for call in ready['call_order']:
        cpu = str(ready['call_cpu_ids'][call])
        deps = set(ready['call_predecessors'][call])
        if cpu in previous:
            deps.add(previous[cpu])
        combined[call] = sorted(deps)
        ancestors[call] = deps.union(*(ancestors[dep] for dep in deps))
        ranks[call] = 1 + max((ranks[dep] for dep in deps), default=0)
        lanes[cpu].append(call)
        previous[cpu] = call
    incomparable = [[first, second] for first, second in combinations(ready['call_order'], 2)
        if ready['call_cpu_ids'][first] != ready['call_cpu_ids'][second]
        and first not in ancestors[second]]
    successors = {call: [] for call in ready['call_order']}
    for after, parents in ready['call_predecessors'].items():
        for before in parents:
            if ready['call_cpu_ids'][before] != ready['call_cpu_ids'][after]:
                successors[before].append(after)
    return dict(lane_call_order=lanes, schedule_predecessors=combined,
        longest_dependency_and_lane_path_call_count=max(ranks.values()),
        potentially_overlapping_cross_lane_calls=incomparable,
        cross_lane_predecessors={call: children for call, children in successors.items() if children})


def domain_cases(pipeline, source_memory, source_ready, target, source_values, *, search_budget):
    """Construct one retained original and the declared CPU × arena domain."""
    serial, pair, evidence = _eligible_cpu_sets(target, source_ready['caller_affinity_cpu_ids'])
    plans = {name: planner(pipeline, input_shapes=source_memory['input_shapes'],
        memory_budget_bytes=source_memory['memory_budget_bytes'], search_budget=search_budget)
        for name, planner in (('source', plan_pipeline_memory), ('dependencies', plan_pipeline_partial_order_memory))}
    yield dict(id='original', implementation='ORIGINAL_PYTHON', arena_policy=None,
        legal_status='ORIGINAL_EXECUTION', cpu_ids=None, hardware_bindings=[], legality_frontier=[],
        schedules=None, dispatch_policy='ORIGINAL_PYTHON_CALL_ORDER',
        call_order=source_ready['call_order'], peak_payload_bytes=None), None, None
    for implementation, cpus in (('cpu-serial', serial), ('cpu-ready', pair)):
        for arena, memory in plans.items():
            issues = []
            if not cpus:
                issues.append('NO_REPLAYED_SAME_NUMA_DISTINCT_PHYSICAL_CORE_PAIR' if implementation == 'cpu-ready'
                    else 'NO_REPLAYED_ONLINE_AFFINITY_CPU')
            if memory['status'].startswith('INFEASIBLE'):
                issues.append('ARENA_CAPACITY_LOWER_BOUND_EXCEEDS_DECLARED_PAYLOAD_BUDGET')
            elif memory['search_budget_exhausted'] or memory['frontier'] or not memory['budget_satisfied']:
                issues.append('ARENA_LIFETIMES_OR_FINITE_CAPACITY_SEARCH_UNRESOLVED')
            ready = None if issues else plan_pipeline_ready_schedule(pipeline, memory, target,
                cpu_ids=cpus, input_graph=source_values)
            yield dict(id=implementation+':'+arena, implementation=implementation, arena_policy=arena,
                legal_status=('ADMITTED_EXPLICIT_TRIAL' if ready is not None else 'UNRESOLVED'
                    if 'ARENA_LIFETIMES_OR_FINITE_CAPACITY_SEARCH_UNRESOLVED' in issues else 'INELIGIBLE'),
                cpu_ids=list(cpus), hardware_bindings=[evidence[cpu] for cpu in cpus],
                legality_frontier=issues, peak_payload_bytes=memory['peak_allocation_bytes'],
                dispatch_policy='FIXED_CPU_LANE_HEADS_WAIT_ON_CALL_AND_SLOT_PREDECESSORS',
                call_order=source_ready['call_order'],
                schedules=None if ready is None else _ordered_calls(ready)), memory, ready


_ORIGINAL_SCOPE = 'ORIGINAL_SINGLE_PROTECTED_CALL_WARM_COMPOSED_TOTAL'
_REGIME = 'FIXED_LANE_HEADS_WARM_GUARDED_CXX20_ABI1'


def _cost(role, query, profile, *, bases=('EMPIRICAL_MODEL', 'ANALYTICAL_MODEL')):
    estimate = resolve_cost(query, profile)
    missing = list(estimate.missing_terms)
    if estimate.basis not in bases and estimate.basis != 'UNKNOWN':
        missing.append('COST_BASIS_IS_NOT_QUALIFIED_FOR_PHYSICAL_COMPOSITION')
    if any(getattr(estimate, field) is None for field in ('estimate_ns', 'lower_ns', 'upper_ns')):
        if not missing:
            missing.append('FULL_CONDITIONAL_INTERVAL_REQUIRED')
    return dict(role=role, key=query_key(query), query=dict(scope=query.scope, context=query.context),
        basis=estimate.basis, estimate_ns=None if missing else estimate.estimate_ns,
        lower_ns=None if missing else estimate.lower_ns, upper_ns=None if missing else estimate.upper_ns,
        evidence=list(estimate.source_refs), missing_terms=missing)


def case_cost(pipeline, memory, ready, row, source_context, leaf_problem, profile):
    """Compose guarded scope terms and only count shared lifecycle work once."""
    if row['legal_status'] in ('UNRESOLVED', 'INELIGIBLE'):
        return dict(status='NO_ADMITTED_NATIVE_CASE', terms=[], predicted_ns=None)
    if row['implementation'] == 'ORIGINAL_PYTHON':
        context = dict(source_context, implementation='ORIGINAL_PYTHON',
            regime='UNCHANGED_WARM_CALLER_REGION', sample_unit='ONE_INVOCATION')
        term = _cost('original_region', CostQuery(_ORIGINAL_SCOPE, context), profile,
            bases=('ANALYTICAL_MODEL',))
        return _summarize_terms([term], None)

    machine = dict(cpu_ids=row['cpu_ids'], call_cpu_ids=ready['call_cpu_ids'],
        memory_policy=ready['memory_policy'], caller_affinity=ready['caller_affinity_cpu_ids'])
    context = dict(source_context, implementation=row['implementation'], regime=_REGIME,
        sample_unit='ONE_INVOCATION', cpu_resources=machine,
        memory_plan_sha256=semantic_sha256(memory), peak_payload_bytes=memory['peak_allocation_bytes'],
        workspace_slot_capacities=[x['capacity_bytes'] for x in memory['workspace_slots']],
        final_bytes=memory['final_bytes'])
    schedules = row['schedules']
    terms = []
    for role in ('entry_guards_and_ffi', 'fresh_final_allocation_and_publish'):
        terms.append(_cost(role, CostQuery(f'{_REGIME}/{role}',
            dict(context, scope_qualification='EXCLUSIVE_HOST_WORK_OUTSIDE_CALL_AND_LANE_CRITICAL_PATH')), profile))
    for role in ('workspace_arena_lifecycle', 'worker_start_and_join'):
        terms.append(_cost(role, CostQuery(f'{_REGIME}/{role}',
            dict(context, scope_qualification='EXCLUSIVE_CRITICAL_PATH_RESIDUAL_AFTER_ACCOUNTING_FOR_OVERLAP')),
            profile, bases=('ANALYTICAL_MODEL',)))
    if schedules['cross_lane_predecessors']:
        terms.append(_cost('dependency_handoffs', CostQuery(f'{_REGIME}/dependency_handoffs',
            dict(context, cross_lane_predecessors=schedules['cross_lane_predecessors'],
                scope_qualification='CRITICAL_PATH_RESIDUAL_BEYOND_ALL_ABI1_SERVICES_AND_WORKER_LIFECYCLE')),
            profile, bases=('ANALYTICAL_MODEL',)))
    possible_overlap = schedules['potentially_overlapping_cross_lane_calls']
    from .pipeline_implementation import SCOPE
    for operation in leaf_problem.operations:
        call = operation.id
        option = next(mode for mode in operation.options if mode.parameters['cpu_id'] == ready['call_cpu_ids'][call])
        leaf = option.parameters['context']
        partners = [pair[1-pair.index(call)] for pair in possible_overlap if call in pair]
        if partners:
            # A warmed isolated ABI1 probe cannot stand in for simultaneous
            # demand on the memory hierarchy. Its exact concurrent scope is separate.
            query = CostQuery(f'{_REGIME}/abi1_service_with_possible_overlap',
                dict(context, abi1_service=leaf, possibly_overlapping_calls=partners))
        else:
            query = CostQuery(SCOPE, leaf)
        terms.append(dict(_cost('abi1_service', query, profile), call=call))
        terms.append(dict(_cost('call_wrapper', CostQuery(f'{_REGIME}/call_wrapper',
            dict(context, call=call, scope_qualification='EXCLUDES_ABI1_SERVICE_AND_DEPENDENCY_WAIT')), profile),
            call=call))
    return _summarize_terms(terms, schedules['schedule_predecessors'])


def _summarize_terms(terms, predecessors):
    missing = [dict(role=term['role'], call=term.get('call'), key=term['key'],
        reasons=term['missing_terms']) for term in terms if term['missing_terms']]
    summary = None
    if not missing:
        summary = {}
        for field in ('estimate_ns', 'lower_ns', 'upper_ns'):
            shared = sum(term[field] for term in terms if 'call' not in term)
            core = {}
            weights = {}
            for term in (term for term in terms if 'call' in term):
                call = term['call']
                weights[call] = weights.get(call, 0) + term[field]
            for call, weight in weights.items():
                core[call] = weight + max((core[parent] for parent in predecessors[call]), default=0)
            summary[field] = shared + max(core.values(), default=0)
    return dict(status='CONDITIONAL_COMPONENT_MODEL' if summary else 'UNRESOLVED',
        terms=terms, missing=missing, predicted_ns=summary,
        composition='SHARED_EXCLUSIVE_HOST_TERMS_ONCE_PLUS_WEIGHTED_CALL_AND_LANE_CRITICAL_PATH',
        interval_is_physical_guarantee=False)


def cost_context(pipeline, source_memory, source_ready, source_inputs):
    source = pipeline.source
    return dict(pipeline_sha256=pipeline_digest(pipeline), target=source_ready['target'],
        input_shapes=source_memory['input_shapes'], input_layout='CPU_NUMPY_C_CONTIGUOUS_FLOAT64',
        memory_budget_bytes=source_memory['memory_budget_bytes'],
        caller_source=dict(path=source.path, sha256=source.sha256, qualname=source.qualname,
            line=source.line, column=source.column, code_key=source.code_key,
            origins_sha256=semantic_sha256(source.origins)), original_proof_sha256=source_inputs['proof']['sha256'])


def decide(cases, *, policy_version=2):
    """Select only interval-separated conditional improvements over original."""
    if type(policy_version) is not int or policy_version not in (1, 2):
        raise ValueError('unsupported resource-event decision policy')
    inclusive = policy_version == 1
    gain_denominator = 20 if inclusive else 100
    eligible = [row for row in cases if row['legal_status'] != 'INELIGIBLE']
    unresolved = [row['id'] for row in eligible if row['cost']['predicted_ns'] is None]
    original = cases[0]['cost']['predicted_ns']
    priced = [row for row in eligible if row['cost']['predicted_ns'] is not None]
    best = min(priced, key=lambda row: (row['cost']['predicted_ns']['estimate_ns'], row['id'])) if priced else None
    selected = None
    if unresolved:
        reason = 'REQUIRED_COST_OR_LEGALITY_UNRESOLVED'
    elif best['id'] == 'original':
        reason = 'ORIGINAL_HAS_LOWEST_CONDITIONAL_POINT_COST'
    else:
        upper = best['cost']['predicted_ns']['upper_ns']
        alternative_lower = min(row['cost']['predicted_ns']['lower_ns'] for row in priced if row is not best)
        baseline = original['lower_ns']
        savings = gain_denominator * (baseline - upper)
        gain_qualified = savings >= baseline if inclusive else savings > baseline
        if upper >= alternative_lower or not gain_qualified:
            reason = 'COST_INTERVALS_NOT_SEPARATED_OR_MINIMUM_GAIN_NOT_PROVEN_WITHIN_MODEL'
        else:
            selected = best['id']
            reason = 'CONDITIONALLY_LEAST_COST_WITH_INTERVAL_SEPARATION_AND_MINIMUM_GAIN'
    result = dict(deployment_default='NO_OP', selected_case_id=selected,
        selection_decision='COST_SELECTED_EXPLICIT_CASE_ONLY' if selected else 'NO_OP',
        reason=reason, unresolved_case_ids=unresolved, minimum_gain_fraction=dict(numerator=1, denominator=gain_denominator),
        full_program_variant_search_performed=False, deployment_authorized=False)
    if not inclusive:
        result['inclusive_minimum_gain'] = False
    return result


__all__ = [
    'plan_pipeline_schedule',
    'verify_pipeline_schedule',
    'READY_SCHEMA',
    'READY_FAMILY',
    'DISPATCH_POLICY',
    'plan_pipeline_ready_schedule',
    'verify_pipeline_ready_schedule',
    'SCHEMA',
    'FAMILY',
    'domain_cases',
    'case_cost',
    'cost_context',
    'decide',
]
