"""Bind existing time occupancies to retained CPU/host topology candidates.

No live hardware query, resource reservation, page placement or cost inference
occurs here. Candidate domains and source-selected CPU requirements differ.
"""
from copy import deepcopy
from pathlib import Path

from scar.analysis.target_graph import TargetHardwareGraph
from scar.ir.execution_recipe import artifact_reference,read_artifact
from scar.ir.program import target_hardware as H
from scar.ir.program.providers import ProviderArtifact
from scar.ir.record_codec import encode
from scar.ir.temporal_resources import (ExecutionBinding,StorageBinding,OriginalBudget,
    ResourceProjection,validate_resource_projection)
from scar.ir.versioned_schedule import TemporalOccupancyKind as K,schedule_document,validate_versioned_schedule
from scar.optimization.resource_solver import semantic_sha256


def _cpu_ids(value,field):
    if value is None:return None
    if (type(value)not in(list,tuple)or any(type(x)is not int or x<0 for x in value)
            or len(set(value))!=len(value)):
        raise ValueError(field+' must be an original distinct CPU-id sequence or UNKNOWN')
    return tuple(value)


def derive_temporal_resources(schedule,target,*,source_context):
    """Return one typed binding per original occupancy, with guarded domains."""
    validate_versioned_schedule(schedule)
    if schedule.temporal is None:raise ValueError('original schema3 temporal schedule required')
    if type(source_context)is not dict:raise TypeError('explicit source resource context required')
    if type(target)is TargetHardwareGraph:
        graph=target
    else:
        path=Path(target).resolve();graph=TargetHardwareGraph(path.parent if path.is_file()else path)
    target_ref=artifact_reference(graph.path/'result.json','scar.target-hardware-graph')
    if encode(source_context.get('target_reference'))!=target_ref:
        raise ValueError('source context and original hardware snapshot differ')
    if read_artifact(target_ref)!=graph.report:raise ValueError('original target report changed')
    snapshot=ProviderArtifact(**target_ref);resources=graph.resources();links=graph.connections()
    by_id={r.id:r for r in resources};by_wire={str(r.id):r for r in resources}
    threads={r.cpu_id:r for r in resources if type(r)is H.HardwareThread}
    caller=_cpu_ids(source_context.get('caller_affinity_cpu_ids'),'caller_affinity_cpu_ids')
    recipe=_cpu_ids(source_context.get('recipe_binding_cpu_ids'),'recipe_binding_cpu_ids')
    refs=source_context.get('source_refs',())
    if type(refs)not in(list,tuple)or any(type(row)is not dict for row in refs):
        raise ValueError('source_refs must retain original reference dictionaries')
    refs=tuple(encode(refs));digest=semantic_sha256(schedule_document(schedule))
    reference=lambda row:H.TargetResourceRef(snapshot,row.id)
    unresolved=[]
    def frontier(field,reason,**facts):unresolved.append(dict(field=field,reason=reason,**facts))
    # Executor/thread membership and host access are original target edges.
    contains={};access={}
    for ordinal,edge in enumerate(links):
        left,right=by_id[edge.source],by_id[edge.target]
        if edge.kind is H.ConnectionKind.CONTAINS and type(right)is H.HardwareThread:
            contains.setdefault(right.id,[]).append((ordinal,left))
        if (edge.kind is H.ConnectionKind.MEMORY_ACCESS and type(left)is H.CpuExecutor
                and type(right)is H.MemoryDomain and right.kind is H.MemoryKind.HOST):
            access.setdefault(left.id,[]).append((ordinal,right))
    reachable_memories={memory.id:memory for rows in access.values()for _,memory in rows}
    memory_rows=tuple(sorted(reachable_memories.values(),key=lambda r:str(r.id)))
    memory_relations=[]
    for executor_id,rows in access.items():
        for ordinal,memory in rows:
            edge=links[ordinal]
            memory_relations.append(dict(connection_ordinal=ordinal,executor=str(executor_id),
                memory=str(memory.id),locality=edge.locality,conditions=list(edge.conditions),
                evidence=encode(edge.evidence)))
    for relation in memory_relations:
        if relation['conditions']:
            frontier('target.connections/'+str(relation['connection_ordinal']),
                'ORIGINAL_HOST_ACCESS_POLICY_CONDITIONS_NOT_DISCHARGED',conditions=relation['conditions'])
    caller_threads=tuple(threads[cpu]for cpu in(caller or())
        if cpu in threads and threads[cpu].online is not False)
    if caller is not None:
        missing=[cpu for cpu in caller if cpu not in threads or threads[cpu].online is False]
        if missing:frontier('source_context.caller_affinity_cpu_ids',
            'DECLARED_CPU_IDS_MISSING_OR_OFFLINE_IN_ORIGINAL_TARGET',cpu_ids=missing)
    executions=[];storages=[]
    slots={slot.id:slot for slot in schedule.storage_slots};tasks={task.id:task for task in schedule.tasks}
    for ordinal,occupancy in enumerate(schedule.temporal.occupancies):
        source=({'kind':'ORIGINAL_TEMPORAL_OCCUPANCY','source_schedule_sha256':digest,
            'locator':'#/temporal/occupancies/'+str(ordinal)},)
        if occupancy.kind is K.EXECUTION_DEMAND:
            index=len(executions);record=by_wire.get(occupancy.resource)
            exact=record if type(record)is H.HardwareThread else None
            candidates=(exact,)if exact is not None else caller_threads if record is None else()
            if exact is not None and exact.online is False:
                frontier('executions/'+str(index),'SOURCE_SELECTED_CPU_OFFLINE_IN_TARGET_SNAPSHOT')
            if record is not None and exact is None:
                frontier('executions/'+str(index),'EXECUTION_RESOURCE_IS_NOT_A_CPU_HARDWARE_THREAD',
                    original_resource=occupancy.resource)
            if exact is None and not candidates:
                frontier('executions/'+str(index)+'/candidate_threads','ORIGINAL_CALLER_CPU_DOMAIN_UNSPECIFIED')
            if occupancy.units is None:
                frontier('executions/'+str(index)+'/units','ORIGINAL_EXECUTION_DEMAND_UNITS_UNKNOWN')
            core_rows=tuple(r for r in resources if type(r)is H.CpuCore
                and any(t.cpu_id in r.cpu_ids for t in candidates))
            executors={row.id:row for thread in candidates for _,row in contains.get(thread.id,())
                if type(row)is H.CpuExecutor and thread.cpu_id in row.cpu_ids}
            connection_ordinals=tuple(i for thread in candidates for i,_ in contains.get(thread.id,()))
            access_ordinals=tuple(i for executor in executors for i,_ in access.get(executor,()))
            if candidates and not core_rows:
                frontier('executions/'+str(index)+'/core_refs','ORIGINAL_PHYSICAL_CORE_TOPOLOGY_UNAVAILABLE')
            topology=dict(kind='ORIGINAL_CPU_TOPOLOGY_CANDIDATES',
                contains_connection_ordinals=connection_ordinals,memory_access_connection_ordinals=access_ordinals,
                executor_candidates=tuple(str(key)for key in executors),
                numa_memory_candidates=tuple(str(row.id)for executor in executors for _,row in access.get(executor,())),
                smt_sibling_cpu_ids={str(t.id):t.thread_siblings for t in candidates},
                actual_thread_execution_proven=False,exclusive_physical_core_reservation=False)
            executions.append(ExecutionBinding(occupancy.id,occupancy.resource,
                None if exact is None else reference(exact),tuple(reference(r)for r in core_rows),
                tuple(reference(r)for r in candidates),occupancy.units,(*source,encode(topology))))
        elif occupancy.kind is K.PHYSICAL_SLOT_BYTES:
            slot=slots[occupancy.storage_slot];index=len(storages)
            allocation=tasks[slot.allocated_by]
            supported=slot.device=='cpu'and allocation.capability=='CPU_ALLOCATE'
            candidates=memory_rows if supported else()
            if not candidates:frontier('storages/'+str(index)+'/candidate_memories',
                'NO_ORIGINAL_HOST_ACCESS_DOMAIN_FOR_THIS_SLOT_LOCATION')
            frontier('storages/'+str(index)+'/exact_memory','PHYSICAL_PAGE_OR_NUMA_RESIDENCY_UNPROVEN')
            storages.append(StorageBinding(occupancy.id,slot.id,tuple(reference(r)for r in candidates),
                None,slot.capacity_bytes,(*source,encode(dict(kind='ORIGINAL_HOST_LOCATION_CANDIDATES',
                    slot_location=slot.location,memory_access_connection_ordinals=tuple(
                        relation['connection_ordinal']for relation in memory_relations),
                    reported_capacity_is_available_budget=False,allocation_or_page_binding_authorized=False)))))
        else:raise ValueError('unsupported original temporal occupancy kind')
    connections=graph.report['connections']
    context=dict(kind='ORIGINAL_SOURCE_RESOURCE_CONTEXT',caller_affinity_cpu_ids=caller,
        recipe_binding_cpu_ids=recipe,
        recipe_binding_authority='ONLY_SELECTED_COMPUTE_DOMAIN_NO_HOST_CONTROL_AUTHORITY',
        candidate_domains_are_runtime_binding=False,
        target_connection_table=dict(uri=str(graph.path/connections['path']),sha256=connections['sha256'],
            size_bytes=connections['bytes'],decoded_sha256=connections['decoded_sha256']),
        memory_access_relations=memory_relations,
        source_budget_interpretation='PRESERVE_ORIGINAL_SCOPE_NOT_RSS_OR_NEW_AVAILABLE_MEMORY_BUDGET')
    result=ResourceProjection(snapshot,digest,tuple(executions),tuple(storages),
        OriginalBudget(schedule.budget_bytes,schedule.budget_scope,deepcopy(schedule.budget_refs)),
        tuple(unresolved),(*refs,encode(context)))
    validate_resource_projection(result,schedule,resources=resources)
    read_artifact(target_ref)
    return result


__all__=['derive_temporal_resources']
