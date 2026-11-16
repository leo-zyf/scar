"""G1 temporal structure only: declared examples, no solver or measurements."""
from dataclasses import replace
import json

import pytest

from test_versioned_schedule import ordered_state_schedule
from scar.ir.versioned_schedule import (
    TemporalMode as Mode, TemporalEventKind as EK, TemporalScopeKind as SK,
    TemporalDurationBasis as Basis, TemporalOccupancyKind as OK,
    TemporalConstraintKind as CK, TemporalEvent as Event, TemporalScope as Scope,
    TemporalActivity as Activity, TemporalOccupancy as Occupancy,
    TemporalConstraint as Constraint, TemporalModel as Model,
    schedule_document, schedule_from_dict, validate_versioned_schedule,
    validate_temporal_model,
)
from scar.optimization.resource_solver import semantic_sha256

REFS = ({'fixture_authority':'DECLARED_STRUCTURAL_EXAMPLE_NOT_RUNTIME_OBSERVATION'},)
LEGACY_V2_SHA256 = 'd3988fd8be63259b92ba802ae8d5653a6e0382b9d45652bf990b0d45dcc9fe12'


def _model(schedule, *, scope_kind=None, trip_count=None, times=None):
    root = schedule.call_scope
    scope = root if scope_kind is None else 'scope:shared-body'
    boundary = (Event(root+':entry',EK.SCOPE_ENTRY,root), Event(root+':exit',EK.SCOPE_EXIT,root))
    scopes = (Scope(root,SK.ROOT,boundary[0].id,boundary[1].id),)
    if scope != root:
        boundary += (Event(scope+':entry',EK.SCOPE_ENTRY,scope), Event(scope+':exit',EK.SCOPE_EXIT,scope))
        scopes += (Scope(scope,scope_kind,boundary[-2].id,boundary[-1].id,parent=root,
            iteration_symbol='i' if scope_kind is SK.LOOP else None,trip_count=trip_count),)
    values = times or {}
    events = boundary + tuple(Event('compute:'+name,kind,scope,task='compute',
        materialization='rhs0' if kind is EK.READY else None,
        timestamp_ticks=values.get(name),source_refs=REFS if name in values else ())
        for name,kind in (('start',EK.START),('return',EK.SUBMISSION_RETURN),
                          ('complete',EK.COMPLETION),('ready',EK.READY)))
    duration = values['complete']-values['start'] if times else None
    activity = Activity('activity:compute','compute',scope,'compute:start','compute:complete',
        submission_return='compute:return',ready_events=('compute:ready',),
        duration_basis=Basis.DECLARED_MODEL if times else Basis.UNKNOWN,
        duration_lower_ticks=duration,duration_upper_ticks=duration,source_refs=REFS if times else ())
    return Model(Mode.PLANNED if times else Mode.SEMANTIC,events,(activity,),(),(),scopes,
        tick_numerator=1 if times else None,tick_denominator=10**9 if times else None,
        clock_domain='DECLARED_TEST_MODEL_NOT_MEASURED' if times else None)


def test_v2_wire_hash_stays_unchanged_and_submit_return_is_not_data_ready():
    schedule = ordered_state_schedule()
    old = schedule_document(schedule)
    assert old['schema_version']==2 and 'temporal' not in old
    assert semantic_sha256(old)==LEGACY_V2_SHA256
    assert schedule_from_dict(json.loads(json.dumps(old)))==schedule
    model = _model(schedule,times={'start':0,'return':1,'complete':5,'ready':5})
    validate_temporal_model(model,schedule)
    wire = schedule_document(replace(schedule,temporal=model))
    assert wire['schema_version']==3
    assert schedule_from_dict(json.loads(json.dumps(wire)))==replace(schedule,temporal=model)
    early = replace(model,events=tuple(replace(event,timestamp_ticks=1)
        if event.id=='compute:ready' else event for event in model.events))
    with pytest.raises(ValueError,match='same-iteration'):
        validate_temporal_model(early,schedule)
    # A short asynchronous operation can finish before the submission API returns.
    early_complete = _model(schedule,times={'start':0,'return':5,'complete':1,'ready':1})
    validate_temporal_model(early_complete,schedule)
    # Separate host/API and execution phases may reference the same original
    # task. Only the explicitly bound producing phase establishes data READY.
    root=schedule.call_scope
    host_events=(Event('compute:host-start',EK.START,root,task='compute',timestamp_ticks=0,source_refs=REFS),
                 Event('compute:host-complete',EK.COMPLETION,root,task='compute',timestamp_ticks=5,source_refs=REFS))
    host=Activity('activity:host-submit','compute',root,host_events[0].id,host_events[1].id,
        duration_basis=Basis.DECLARED_MODEL,duration_lower_ticks=5,duration_upper_ticks=5,source_refs=REFS)
    validate_temporal_model(replace(early_complete,events=(*early_complete.events,*host_events),
        activities=(*early_complete.activities,host)),schedule)
    allocation_ready = replace(model,events=tuple(replace(event,task='allocate')
        if event.id=='compute:ready' else event for event in model.events))
    with pytest.raises(ValueError,match='actual materialization producer'):
        validate_temporal_model(allocation_ready,schedule)


def test_composite_is_container_and_shared_slot_versions_count_one_physical_occupation():
    schedule = ordered_state_schedule()
    model = _model(schedule,scope_kind=SK.COMPOSITE)
    root = schedule.call_scope
    events = model.events + (Event('allocate:start',EK.START,root,task='allocate'),
        Event('release:complete',EK.COMPLETION,root,task='release'))
    demand = Occupancy('occupancy:compute',OK.EXECUTION_DEMAND,'declared:cpu',
        'compute:start','compute:complete',model.activities[0].scope,units=1,activity='activity:compute')
    physical = Occupancy('occupancy:temp',OK.PHYSICAL_SLOT_BYTES,'declared:payload',
        'allocate:start','release:complete',root,units=16,storage_slot='temp')
    model = replace(model,events=events,occupancies=(demand,physical),
        resource_capacities={'declared:cpu':1,'declared:payload':16})
    result = validate_versioned_schedule(replace(schedule,temporal=model))['temporal']
    assert len([m for m in schedule.materializations if m.region.slot=='temp'])==2
    assert result['scopes']==2 and result['activities']==1 and result['physical_slots_counted_once']==1
    assert result['physical_slot_coverage_complete'] is False
    assert result['activity_task_coverage_complete'] is False
    assert result['resource_capacity_feasibility_proven'] is False
    with pytest.raises(ValueError,match='double-count'):
        validate_temporal_model(replace(model,occupancies=(*model.occupancies,
            replace(demand,id='occupancy:duplicate-work'))),schedule)
    with pytest.raises(ValueError,match='exactly once'):
        validate_temporal_model(replace(model,occupancies=(*model.occupancies,
            replace(physical,id='occupancy:duplicate-slot'))),schedule)


def test_million_iteration_loop_is_symbolic_and_recurrence_is_not_a_same_iteration_cycle():
    schedule = ordered_state_schedule()
    model = _model(schedule,scope_kind=SK.LOOP,trip_count=10**6)
    recurrence = Constraint('edge:carry','compute:complete','compute:start',CK.LOOP_CARRIED,
        min_delay_ticks=1,iteration_distance=1,loop_scope='scope:shared-body')
    model = replace(model,constraints=(recurrence,))
    summary = validate_temporal_model(model,schedule)
    assert summary['loop_scopes']==1 and summary['symbolic_iteration_constraints']==1
    assert summary['iterations_unrolled']==0 and summary['events']==8 and summary['activities']==1
    assert summary['periodic_feasibility_proven'] is False
    small = replace(model,scopes=tuple(replace(scope,trip_count=2) if scope.kind is SK.LOOP else scope
        for scope in model.scopes))
    assert len(small.events)==len(model.events) and len(small.activities)==len(model.activities)
    wire = schedule_document(replace(schedule,temporal=model))
    assert schedule_from_dict(wire)==replace(schedule,temporal=model)
    same_iteration = replace(recurrence,kind=CK.CAUSAL,iteration_distance=0,loop_scope=None)
    with pytest.raises(ValueError,match='same-iteration'):
        validate_temporal_model(replace(model,constraints=(same_iteration,)),schedule)


def test_mechanical_cpu_adapter_covers_tasks_and_legacy_solver_cannot_ignore_time():
    from scar.analysis.schedule_time import attach_task_time
    from scar.optimization.schedule import schedule_from_dict as public_reader
    from scar.optimization.resource_solver import solve_resource_schedule
    original = ordered_state_schedule()
    schedule = attach_task_time(original, source_refs=REFS)
    assert public_reader(schedule_document(schedule)) == schedule
    summary = validate_versioned_schedule(schedule)['temporal']
    assert summary['activity_task_coverage_complete']
    assert summary['physical_slot_coverage_complete']
    assert summary['unknown_durations'] == len(original.tasks)
    assert summary['unknown_execution_demands'] == len(original.tasks)
    assert summary['unbound_materialization_ready_events'] == 0
    # Even a fully supplied old task model cannot silently discard event and
    # recurrence constraints. This must stop before exploring a search state.
    sizes = {slot.id:slot.capacity_bytes for slot in original.storage_slots if slot.ownership=='SCHEDULE'}
    result = solve_resource_schedule(schedule,
        task_costs={task.id:1 for task in original.tasks},
        task_placements={task.id:{} for task in original.tasks}, resource_capacities={},
        materialization_sizes=sizes, memory_locations={key:'pool' for key in sizes},
        memory_capacities={'pool':sum(sizes.values())})
    assert result['status']=='UNRESOLVED' and result['states_explored']==0
    assert [issue['field'] for issue in result['unresolved']] == ['temporal']
