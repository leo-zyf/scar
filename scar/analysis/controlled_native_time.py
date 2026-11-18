"""Bind existing synchronous native tasks to explicitly controlled lanes.

Lane participation is neither total physical CPU demand nor an exclusive core
reservation.  The caller must cold-admit the three input publications.  This
projection preserves their original task/storage identities and derives no
durations, runtime dispatch, placement, RSS bound or deployment permission.
"""
from copy import deepcopy
from dataclasses import replace

from scar.analysis.schedule_time import attach_task_time
from scar.ir.versioned_schedule import (
    TemporalConstraint, TemporalConstraintKind as CK, TemporalOccupancyKind as OK,
    VersionedTaskKind as TK, schedule_from_dict, schedule_document, validate_versioned_schedule,
)
from scar.optimization.resource_solver import semantic_sha256


SCHEMA='scar.controlled-native-time'


def _need(value,message):
    if not value:raise ValueError(message)


def bind_controlled_native_time(stage_context,contract,correspondence):
    """Return ``(VersionedResourceSchedule, report)`` without running native code."""
    for value,schema in ((stage_context,'scar.native-stage-context'),
            (contract,'scar.controlled-region-resource-contract'),
            (correspondence,'scar.stateful-source-native-correspondence')):
        _need(type(value) is dict and value.get('schema')==schema
            and type(value.get('schema_version')) is int and value['schema_version']==1,
            'original typed-context publication required: '+schema)
    _need(contract['kind']=='STATEFUL_SPECIALIZATION','stateful synchronous native contract required')
    original=schedule_from_dict(stage_context['versioned_schedule'])
    identity=contract['identity'];implementation=contract['implementation']
    _need(original.region_id==identity['region_id']==correspondence['region_id']
        and stage_context['native_identity']['region_sha256']==identity['region_sha256']==correspondence['region_sha256']
        and stage_context['native_identity']['plan_sha256']==identity['stage_plan_sha256']==correspondence['stage_plan_sha256']
        and stage_context['native_identity']['build_id']==implementation['build_id']
        and stage_context['source_refs']['native_manifest']==implementation['native_manifest']
        and stage_context['source_refs']['plan']==identity['stage_plan']==correspondence['stage_plan'],
        'native task graph, source correspondence and contract identities differ')
    memory=contract['memory'];payload=stage_context['payload_budget']
    _need(original.budget_bytes==memory['budget_bytes']==payload['memory_budget_bytes']
        and original.budget_scope==memory['budget_scope']==payload['budget_scope']
        and payload['external_array_bytes']==memory['borrowed_payload_bytes']
        and payload['workspace_payload_bytes']==memory['private_workspace_bytes']
        and payload['total_region_array_payload_bytes']==memory['included_payload_bytes']
        and payload['budget_excludes']==memory['budget_excludes'],
        'original formal payload scope or byte budget differs')
    private={row.id:row.capacity_bytes for row in original.storage_slots if row.ownership=='SCHEDULE'}
    borrowed={row.id:row.capacity_bytes for row in original.storage_slots if row.ownership=='EXTERNAL'}
    _need(private=={row['id']:row['capacity_bytes'] for row in memory['private_slots']}
        and sum(private.values())==memory['private_workspace_bytes']
        and sum(borrowed.values())==memory['borrowed_payload_bytes'],
        'native storage slots differ from the admitted fixed payload allocation')
    execution=contract['execution'];caller=execution['caller_lane'];native=execution['native']
    _need(caller['kind']=='CONTROLLED_SYNCHRONOUS_CALL_CONCURRENCY'
        and caller['capacity']==1 and caller['units_per_invocation']==1
        and caller['physical_cpu_id'] is None and caller['exclusive_core'] is False
        and native['asynchronous'] is False,
        'one synchronous controlled caller lane without a physical core assignment required')
    contract_sha=semantic_sha256(contract);context_sha=semantic_sha256(stage_context)
    correspondence_sha=semantic_sha256(correspondence)
    def reference(schema,sha,locator):
        return dict(schema=schema,semantic_sha256=sha,locator=locator,
            parent_admission=deepcopy(contract['parent_admission']))
    caller_refs=[reference(contract['schema'],contract_sha,'#/execution/caller_lane')]
    caller_id='controlled:caller-lane:'+contract_sha
    caller_resource=dict(id=caller_id,capacity=1,kind=caller['kind'],source_refs=caller_refs,
        physical_cpu_id=None,exclusive_core=False,
        meaning='PARTICIPATION_IN_ONE_SYNCHRONOUS_ACTIVATION_NOT_TOTAL_CPU_DEMAND',
        cross_invocation_serialization_enforced=caller['cross_invocation_serialization_enforced'])
    affinity=execution['affinity_stage'];binding=stage_context['runtime_context']['original_native_binding']
    cpus=affinity['cpu_ids'];native_refs=deepcopy(affinity['target_resource_refs'])
    lane_supported=(affinity['policy']=='NATIVE_STAGE_REQUIRED_CPU_MASK'
        and affinity['phase']=='PRIVATE_SNAPSHOTS_AND_COMPUTATION_UNTIL_PRECOMMIT_ENVIRONMENT_RESTORATION'
        and affinity['caller_mask_restored_by_native'] is True
        and native['cpu_threads']==1 and native['creates_worker_threads'] is False
        and type(cpus) is list and len(cpus)==1 and type(cpus[0]) is int
        and binding['cpu_ids']==cpus and binding['executor_id']==affinity['executor_id']
        and cpus[0] in execution['allowed_cpu_ids'] and bool(native_refs))
    target_refs={semantic_sha256(row) for row in contract['target_resources']}
    _need(all(semantic_sha256(row) in target_refs and row['snapshot']==contract['target']
        for row in native_refs),'native lane binding references a different retained hardware target')
    provider=implementation.get('runtime_stamps',{}).get('provider_runtime',{})
    serial_provider=(native['creates_worker_threads'] is False and native['cpu_threads']==1
        and provider.get('requires_pyapi') is False
        and provider.get('execution')=='ORIGINAL_NUMPY_UFUNC_STRIDED_LOOP_SERIAL_PROVIDER_STAGE')
    native_id='controlled:native-stage-lane:'+contract_sha
    native_resource=None
    if lane_supported:
        native_resource=dict(id=native_id,capacity=1,kind='CONTROLLED_NATIVE_STAGE_PARTICIPATION',
            cpu_ids=deepcopy(cpus),executor_id=affinity['executor_id'],target_resource_refs=native_refs,
            source_refs=[reference(contract['schema'],contract_sha,'#/execution/affinity_stage')],
            exclusive_core=False,physical_cpu_reservation=False,
            meaning='ADMITTED_SINGLE_NATIVE_STAGE_LANE_WITH_REQUIRED_CPU_MASK_UNDER_RUNTIME_GUARDS')
    contexts={row['id']:row for row in stage_context['stages']}
    stages={row['stage']:row for row in correspondence['stages']}
    _need(len(contexts)==len(stage_context['stages']) and len(stages)==len(correspondence['stages'])
        and set(contexts)==set(stages),'one exact source-native stage correspondence required')
    tasks={row.id:row for row in original.tasks}
    _need(set(stages)<=set(tasks),'source-native stage is absent from the original task graph')
    demands={};rows=[];frontier=[]
    for task in original.tasks:
        stage=stages.get(task.id);resources={caller_id:1};issues=[]
        if stage is not None:
            context=contexts[task.id]
            _need(stage['index']==context['index'] and stage['kind']==context['kind']
                and context['source_stage_ref']==stage['source_ref']
                and stage['predecessors']==context['predecessor_stage_ids'],
                'stage binding differs from its original native source/plan coordinate')
            if task.kind is TK.COMMIT:
                _need(stage['kind']=='COMMIT'
                    and stage['hardware_execution']['policy']=='RESTORED_CALLER_AFFINITY_NO_CORE_ASSIGNMENT',
                    'COMMIT must retain restored caller affinity')
            else:
                _need(stage['hardware_execution']==affinity,
                    'non-COMMIT stage has a different affinity scope')
                supported=lane_supported and (stage['kind']!='PROVIDER' or serial_provider)
                if supported:resources[native_id]=1
                else:
                    issues.append(dict(field='native_stage_controlled_execution_lane',
                        reason='Exact single-stage CPU-mask contract or serial provider capability remains unproved.',
                        source_refs=[reference(contract['schema'],contract_sha,'#/execution/affinity_stage'),
                            deepcopy(stage['source_ref'])]))
        demands[task.id]=resources
        for issue in issues:frontier.append(dict(task=task.id,**deepcopy(issue)))
        rows.append(dict(task=task.id,stage=None if stage is None else stage['stage'],
            caller_lane=caller_id,native_lane=native_id if native_id in resources else None,
            execution_demands=resources,status='UNRESOLVED' if issues else 'CONDITIONAL',
            unresolved_resources=issues,total_physical_cpu_demand='UNKNOWN_NOT_THIS_PARTICIPATION_MODEL'))
    capacities={caller_id:1}
    if native_resource is not None:capacities[native_id]=1
    untimed=replace(original,temporal=None)
    bound=attach_task_time(untimed,execution_demands=demands,resource_capacities=capacities,
        source_refs=() if original.temporal is None else original.temporal.source_refs)
    # The generic adapter keeps task dependencies/readiness and allocation
    # lifetimes. Preserve the separate original order/overwrite witnesses too.
    temporal=bound.temporal
    if original.temporal is not None:
        def preserve(old,new):
            by_id={row.id:row for row in new}
            for row in old:
                _need(row.id not in by_id or by_id[row.id]==row,
                    'reattached time differs from an original semantic record')
                by_id[row.id]=row
            return tuple(by_id.values())
        temporal=replace(temporal,events=preserve(original.temporal.events,temporal.events),
            activities=preserve(original.temporal.activities,temporal.activities),
            constraints=preserve(original.temporal.constraints,temporal.constraints),
            scopes=preserve(original.temporal.scopes,temporal.scopes))
    activities={row.task:row for row in temporal.activities}
    constraints=list(temporal.constraints)
    original_sha=semantic_sha256(schedule_document(original))
    def relation(row):
        return (row.before,row.after,row.kind,row.min_delay_ticks,row.max_delay_ticks,
            row.iteration_distance,row.loop_scope,row.before_iteration,row.after_iteration)
    relations={}
    for row in constraints:relations.setdefault(relation(row),row.id)
    constraint_bindings=[]
    def bind_constraint(row,witness):
        key=relation(row);existing=relations.get(key)
        if existing is None:
            constraints.append(row);relations[key]=row.id
        constraint_bindings.append(dict(source_edge=witness,
            constraint_id=row.id if existing is None else existing,reused=existing is not None))
    for index,order in enumerate(original.order_constraints):
        witness=reference(stage_context['schema'],context_sha,
            '#/versioned_schedule/order_constraints/'+str(index))
        bind_constraint(TemporalConstraint('controlled-order:'+semantic_sha256(
            dict(schedule=original_sha,index=index)),
            activities[order.before].completion,activities[order.after].start,CK.RESOURCE_ORDER,
            source_refs=(witness,)),witness)
    kinds={'ORIGINAL_STAGE_DEPENDENCY':CK.CAUSAL,'MATERIAL_READ_AFTER_WRITE':CK.READINESS,
        'READY_SCALAR_DEPENDENCY':CK.READINESS,'PHYSICAL_SLOT_OVERWRITE_ORDER_NOT_VALUE_FLOW':CK.RESOURCE_ORDER,
        'READ_BEFORE_PHYSICAL_SLOT_OVERWRITE':CK.RESOURCE_ORDER}
    for index,edge in enumerate(stage_context['dependency_edges']):
        _need(edge['kind'] in kinds and edge['before'] in activities and edge['after'] in activities,
            'one existing native readiness/order edge required')
        witness=reference(stage_context['schema'],context_sha,'#/dependency_edges/'+str(index))
        bind_constraint(TemporalConstraint('controlled-native-dependency:'+semantic_sha256(
            dict(context=context_sha,index=index)),activities[edge['before']].completion,
            activities[edge['after']].start,kinds[edge['kind']],source_refs=(witness,)),witness)
    occupancies=[]
    for occupancy in temporal.occupancies:
        refs=[]
        if occupancy.resource==caller_id:refs=caller_refs
        elif occupancy.resource==native_id and native_resource is not None:refs=native_resource['source_refs']
        occupancies.append(replace(occupancy,source_refs=occupancy.source_refs+tuple(deepcopy(refs))))
    bound=replace(bound,temporal=replace(temporal,constraints=tuple(constraints),occupancies=tuple(occupancies)))
    validate_versioned_schedule(bound)
    _need(replace(bound,temporal=None)==untimed,'time binding changed original task/value/storage identities')
    report=dict(schema=SCHEMA,schema_version=1,status='UNRESOLVED' if frontier else 'CONDITIONAL',
        stage_context_sha256=context_sha,contract_sha256=contract_sha,correspondence_sha256=correspondence_sha,
        original_schedule_sha256=semantic_sha256(schedule_document(original)),
        bound_schedule_sha256=semantic_sha256(schedule_document(bound)),
        resources=dict(caller_lane=caller_resource,native_lane=native_resource),task_bindings=rows,frontier=frontier,
        constraint_bindings=constraint_bindings,
        coverage=dict(tasks=len(original.tasks),stages=len(stages),materializations=len(original.materializations),
            storage_slots=len(original.storage_slots),private_slots=len(private),borrowed_slots=len(borrowed),
            caller_lane_participations=len(rows),native_lane_participations=sum(native_id in value for value in demands.values()),
            original_order_constraints=len(original.order_constraints),native_dependency_edges=len(stage_context['dependency_edges']),
            original_temporal_constraints=0 if original.temporal is None else len(original.temporal.constraints),
            bound_temporal_constraints=len(bound.temporal.constraints),
            reused_constraint_bindings=sum(row['reused'] for row in constraint_bindings),
            new_constraint_relations=sum(not row['reused'] for row in constraint_bindings)),
        payload_scope=deepcopy(memory),durations_known=False,numeric_durations_derived=False,
        task_materialization_and_slot_identities_preserved=True,original_native_order_preserved=True,
        original_semantic_temporal_records_preserved=True,
        physical_inventory_is_not_exclusive_capacity=True,full_entry_obligations_discharged=False,
        physical_cpu_assignment_proven=False,physical_feasibility_proven=False,
        physical_total_cpu_demand='UNKNOWN',process_RSS='UNKNOWN',numa_residency='UNKNOWN',
        caller_participation_is_not_total_cpu_demand=True,exclusive_cpu_reservation=False,
        runtime_guards_discharged=False,timing_qualified=False,source_replacement_authorized=False,
        deployment_authorized=False,target_program_executed=False,native_loaded=False,compiled=False)
    report['id']='controlled-native-time:'+semantic_sha256(report)
    return bound,report


__all__=['bind_controlled_native_time']
