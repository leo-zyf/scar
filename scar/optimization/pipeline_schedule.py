"""Resource-safe static waves for a closed readonly array-call DAG.

Logical call outputs, reusable storage slots and execution waves are separate.
Unknown durations do not prevent a legal dependency plan, and are never filled
with zero or unit costs. The host must join each wave and retain its ordinary
FP/guard rollback contract before publishing the fresh final result.
"""
from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path

from scar.compiler.pipeline_model import document,pipeline_digest,validate_pipeline
from scar.ir.execution_recipe import read_artifact
from scar.ir.program import target_hardware as H
from scar.ir.record_codec import encode,loads
from scar.ir.versioned_schedule import (VersionedLogicalValue as Value,StorageSlot,StorageRegion,
    VersionedMaterialization as Material,VersionedTask as Task,VersionedTaskKind as K,
    VersionedResourceSchedule,schedule_document,validate_versioned_schedule)
from .hardware import _target_materials,_indexed_resources,_record_evidence,_connection_evidence,_accesses
from .pipeline_memory import verify_pipeline_memory
from .resource_solver import semantic_sha256


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


__all__=['plan_pipeline_schedule','verify_pipeline_schedule']
