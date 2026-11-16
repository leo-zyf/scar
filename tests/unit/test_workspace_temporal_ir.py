"""Selected loop boundaries stay indexed; legacy schedule bytes stay unchanged."""
from dataclasses import replace
import json

import pytest

from test_temporal_schedule import _model, LEGACY_V2_SHA256
from test_versioned_schedule import ordered_state_schedule
from scar.analysis.schedule_time import attach_task_time
from scar.ir.versioned_schedule import (
    TemporalEvent as Event, TemporalEventKind as EK, TemporalScope as Scope,
    TemporalScopeKind as SK, TemporalConstraint as Constraint, TemporalConstraintKind as CK,
    TemporalIterationSelection as Selection, TemporalIterationSelector as Selector,
    TemporalPathCondition as Path, TemporalOccupancyKind as OK, TemporalContentCarry as Carry,
    schedule_document, schedule_from_dict, validate_versioned_schedule,
)
from scar.optimization.resource_solver import semantic_sha256

REFS = ({'authority': 'STRUCTURAL_SOURCE_EXAMPLE_NOT_OBSERVATION'},)


def _workspace_example(count):
    schedule = replace(ordered_state_schedule(), executable=False)
    model = attach_task_time(schedule, source_refs=REFS).temporal
    root, body = schedule.call_scope, 'scope:shared-workspace-step'
    outer = {'allocate', 'release'}
    events = tuple(replace(e, scope=body) if e.task not in outer
        and e.kind not in (EK.SCOPE_ENTRY, EK.SCOPE_EXIT) else e for e in model.events)
    entry, exit = Event(body+':entry', EK.SCOPE_ENTRY, body), Event(body+':exit', EK.SCOPE_EXIT, body)
    events = (*events, entry, exit)
    by_event = {event.id: event for event in events}
    activities = tuple(replace(a, scope=root if a.task in outer else body) for a in model.activities)
    scopes = (*model.scopes, Scope(body, SK.LOOP, entry.id, exit.id, parent=root,
        iteration_symbol='i', trip_count=count, source_refs=REFS,
        iteration_local_values=tuple(v.id for v in schedule.values),
        iteration_local_materializations=tuple(m.id for m in schedule.materializations),
        content_carries=(Carry('external1', 'external0', source_refs=REFS),)))
    constraints = []
    for edge in model.constraints:
        left, right = by_event[edge.before], by_event[edge.after]
        if left.task == 'allocate' and right.task == 'snapshot':
            edge = replace(edge, after_iteration=Selection(body, Selector.FIRST))
        elif left.task == 'commit' and right.task == 'release':
            edge = replace(edge, before_iteration=Selection(body, Selector.LAST_NORMAL))
        elif right.materialization == 'external0':
            edge = replace(edge, before=entry.id)
        constraints.append(edge)
    constraints.append(Constraint('carry', exit.id, entry.id, CK.LOOP_CARRIED,
        iteration_distance=1, loop_scope=body, source_refs=REFS))
    constraints.append(Constraint('material-carry',
        next(a.completion for a in activities if a.task == 'commit'),
        next(e.id for e in events if e.materialization == 'external0'), CK.LOOP_CARRIED,
        iteration_distance=1, loop_scope=body, source_refs=REFS))
    occupancies = tuple(replace(o, scope=body) if o.kind is OK.EXECUTION_DEMAND
        and next(a.task for a in activities if a.id == o.activity) not in outer else o
        for o in model.occupancies)
    return replace(schedule, temporal=replace(model, events=events, activities=activities,
        scopes=scopes, constraints=tuple(constraints), occupancies=occupancies,
        path_condition=Path.NORMAL_COMPLETE))


def test_workspace_first_last_boundaries_and_legacy_wire_are_explicit():
    legacy = ordered_state_schedule()
    assert semantic_sha256(schedule_document(legacy)) == LEGACY_V2_SHA256
    legacy_time = replace(legacy, temporal=_model(legacy))
    legacy_wire = schedule_document(legacy_time)
    assert semantic_sha256(legacy_wire) == '686c02d69a3b2d39ec3bf9c817fb6da18dc7fe99e4e10aa2936a1ecbe8d51d71'
    assert schedule_from_dict(json.loads(json.dumps(legacy_wire))) == legacy_time
    assert 'path_condition' not in legacy_wire['temporal']
    for count in (1, 10**6):
        schedule = _workspace_example(count)
        wire = schedule_document(schedule)
        assert wire['schema_version'] == 4
        assert schedule_from_dict(json.loads(json.dumps(wire))) == schedule
        result = validate_versioned_schedule(schedule)['temporal']
        assert result['indexed_boundary_constraints'] == 2
        assert result['indexed_bounds_feasibility_proven'] is False
        assert result['physical_slots_counted_once'] == 2
        assert result['iterations_unrolled'] == 0 and result['activities'] == 8
        assert result['iteration_local_materializations'] == 6
        assert result['materialization_carry_count'] == 1
    model = schedule.temporal
    first = next(c for c in model.constraints if c.after_iteration is not None)
    last = next(c for c in model.constraints if c.before_iteration is not None)
    mutations = (
        replace(first, after_iteration=Selection(first.after_iteration.loop_scope, Selector.LAST_NORMAL)),
        replace(first, iteration_distance=1),
        replace(first, source_refs=()),
        replace(last, before_iteration=Selection('missing-loop', Selector.LAST_NORMAL)),
    )
    for changed in mutations:
        with pytest.raises(ValueError):
            validate_versioned_schedule(replace(schedule, temporal=replace(model,
                constraints=tuple(changed if c.id == changed.id else c for c in model.constraints))))
    with pytest.raises(ValueError, match='exact selected'):
        validate_versioned_schedule(replace(schedule, temporal=replace(model,
            constraints=tuple(c for c in model.constraints if c.id != first.id))))
    with pytest.raises(ValueError, match='positive fixed'):
        validate_versioned_schedule(_workspace_example(0))
    for carry in (Carry('rhs1', 'external0', source_refs=REFS),
                  Carry('external1', 'shadow0', source_refs=REFS),
                  Carry('external1', 'external0', iteration_distance=0, source_refs=REFS)):
        with pytest.raises(ValueError, match='content carry'):
            validate_versioned_schedule(replace(schedule, temporal=replace(model, scopes=tuple(
                replace(s, content_carries=(carry,)) if s.kind is SK.LOOP else s for s in model.scopes))))
    with pytest.raises(ValueError, match='matching COMMIT-completion'):
        validate_versioned_schedule(replace(schedule, temporal=replace(model,
            constraints=tuple(c for c in model.constraints if c.id != 'material-carry'))))
    with pytest.raises(ValueError, match='unique loop'):
        validate_versioned_schedule(replace(schedule, temporal=replace(model, scopes=tuple(
            replace(s, iteration_local_values=('initial',)) if s.kind is SK.LOOP else s for s in model.scopes))))
    downgraded = json.loads(json.dumps(wire)); downgraded['schema_version'] = 3
    with pytest.raises(ValueError, match='v3 resource schedule'):
        schedule_from_dict(downgraded)
    empty = replace(schedule, values=(), materializations=(), storage_slots=(), tasks=(), budget_bytes=0,
        temporal=replace(model, events=model.events[:2], scopes=model.scopes[:1],
            activities=(), occupancies=(), constraints=(), resource_capacities={}))
    assert schedule_document(empty)['schema_version'] == 4
    assert validate_versioned_schedule(empty)['temporal']['activities'] == 0
