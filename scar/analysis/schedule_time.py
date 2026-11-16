"""Mechanical semantic-time projection of existing synchronous CPU task graphs.

This attaches time to the existing schedule IR; it neither estimates durations
nor chooses a schedule. Physical topology aggregation belongs to the next gate.
"""
from dataclasses import replace
from hashlib import sha256

from scar.ir.versioned_schedule import (
    TemporalMode as Mode, TemporalEventKind as EK, TemporalScopeKind as SK,
    TemporalOccupancyKind as OK, TemporalConstraintKind as CK,
    TemporalEvent as Event, TemporalScope as Scope, TemporalActivity as Activity,
    TemporalOccupancy as Occupancy, TemporalConstraint as Constraint,
    TemporalModel as Model, VersionedTaskKind as TK, validate_versioned_schedule,
)


def attach_task_time(schedule, *, execution_demands=None, resource_capacities=None,
                     source_refs=(), loop_refs=(), trip_count_symbol=None):
    """Project task completion, readiness and allocation lifetimes without clocks.

    Optional loop_refs describe an already source-replayed sequential repeated
    call. One shared body and one distance-one edge are retained. Slot/value IDs
    inside that body are iteration-indexed templates, not inter-iteration aliases.
    Callers must establish source authority before passing those loop references.
    """
    validate_versioned_schedule(schedule)
    if schedule.temporal is not None:
        raise ValueError('attach semantic time only to an original untimed schedule')
    if any(slot.device != 'cpu' for slot in schedule.storage_slots) or any(
            task.kind is TK.COPY for task in schedule.tasks):
        raise ValueError('this adapter requires the existing synchronous CPU task profile')
    if bool(loop_refs) != bool(trip_count_symbol):
        raise ValueError('a source-replayed loop needs its original trip-count parameter')
    demands = execution_demands or {}
    if set(demands) - {task.id for task in schedule.tasks}:
        raise ValueError('resource demand references an unknown original task')
    capacities = dict(resource_capacities or {})
    root = schedule.call_scope

    def identity(*parts):
        key = repr((schedule.region_id, root, *parts)).encode()
        return 'time:' + sha256(key).hexdigest()[:32]

    events, activities, occupancies, constraints = [], [], [], []

    def event(key, kind, scope, **kwargs):
        name = identity(*key)
        events.append(Event(name, kind, scope, **kwargs))
        return name

    root_entry = event(('root', 'entry'), EK.SCOPE_ENTRY, root, source_refs=source_refs)
    root_exit = event(('root', 'exit'), EK.SCOPE_EXIT, root, source_refs=source_refs)
    scopes = [Scope(root, SK.ROOT, root_entry, root_exit, source_refs=source_refs)]
    body, entry, exit = root, root_entry, root_exit
    if loop_refs:
        body = identity('shared-loop-body')
        entry = event(('loop', 'entry'), EK.SCOPE_ENTRY, body, source_refs=loop_refs)
        exit = event(('loop', 'exit'), EK.SCOPE_EXIT, body, source_refs=loop_refs)
        scopes.append(Scope(body, SK.LOOP, entry, exit, parent=root,
            iteration_symbol='i', trip_count_symbol=trip_count_symbol, source_refs=loop_refs))
        constraints.append(Constraint(identity('iteration-carry'), exit, entry,
            CK.LOOP_CARRIED, iteration_distance=1, loop_scope=body, source_refs=loop_refs))

    starts, completes, activity_ids = {}, {}, {}
    ready_by_task = {}
    for task in schedule.tasks:
        refs = (*source_refs, *task.source_refs)
        starts[task.id] = event(('task', task.id, 'start'), EK.START, body,
            task=task.id, source_refs=refs)
        completes[task.id] = event(('task', task.id, 'completion'), EK.COMPLETION, body,
            task=task.id, source_refs=refs)
        activity_ids[task.id] = identity('activity', task.id)

    def edge(before, after, kind, key, *, refs=source_refs, equal=False):
        constraints.append(Constraint(identity('constraint', *key), before, after, kind,
            max_delay_ticks=0 if equal else None, source_refs=refs))

    for material in schedule.materializations:
        name = event(('material', material.id, 'ready'), EK.READY, body,
            task=material.ready_task, materialization=material.id, source_refs=source_refs)
        before = entry if material.external_ready else completes[material.ready_task]
        edge(before, name, CK.READINESS, ('ready', material.id), equal=True)
        if material.ready_task is not None:
            ready_by_task.setdefault(material.ready_task, []).append(name)
        for task in schedule.tasks:
            if material.id in task.reads:
                edge(name, starts[task.id], CK.READINESS, ('read', material.id, task.id))

    for task in schedule.tasks:
        refs = (*source_refs, *task.source_refs)
        activities.append(Activity(activity_ids[task.id], task.id, body,
            starts[task.id], completes[task.id],
            ready_events=tuple(ready_by_task.get(task.id, ())), source_refs=refs))
        for predecessor in task.depends_on:
            edge(completes[predecessor], starts[task.id], CK.CAUSAL,
                ('dependency', predecessor, task.id), refs=refs)
        # Unknown control/service demand is explicit, never treated as free.
        resources = demands.get(task.id)
        if not resources:
            resources = {identity('unknown-service', task.id): None}
        for resource, units in sorted(resources.items()):
            capacities.setdefault(resource, None)
            occupancies.append(Occupancy(identity('execution', task.id, resource),
                OK.EXECUTION_DEMAND, resource, starts[task.id], completes[task.id], body,
                units=units, activity=activity_ids[task.id], source_refs=refs))

    for slot in schedule.storage_slots:
        if slot.ownership != 'SCHEDULE':
            continue
        # This identifies one allocation, not a global memory-domain budget.
        resource = identity('physical-slot-resource', slot.id)
        capacities[resource] = slot.capacity_bytes
        retained = slot.released_by is None
        occupancies.append(Occupancy(identity('slot-occupancy', slot.id),
            OK.PHYSICAL_SLOT_BYTES, resource, starts[slot.allocated_by],
            exit if retained else completes[slot.released_by], body,
            units=slot.capacity_bytes, storage_slot=slot.id,
            continues_after_scope=retained,
            source_refs=(*source_refs, *slot.resource_refs, *slot.budget_refs)))
    model = Model(Mode.SEMANTIC, tuple(events), tuple(activities), tuple(occupancies),
        tuple(constraints), tuple(scopes), resource_capacities=capacities,
        source_refs=source_refs)
    result = replace(schedule, temporal=model)
    validate_versioned_schedule(result)
    return result
