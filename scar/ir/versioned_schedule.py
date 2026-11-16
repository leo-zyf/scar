"""Call-scoped immutable values and versioned physical storage task graphs.

Version numbers order uses of one allocation; they are not source epochs,
LogicalVersions or content-equality proofs. This bounded CPU lowering contract
does not replace the program IR or admit generic/device transfers.
"""
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from scar.ir.record_codec import decode, encode


class VersionedTaskKind(str, Enum):
    ALLOCATE = 'ALLOCATE'
    SNAPSHOT = 'SNAPSHOT'
    COMPUTE = 'COMPUTE'
    STATE_WRITE = 'STATE_WRITE'
    SUCCESS_FENCE = 'SUCCESS_FENCE'
    COMMIT = 'COMMIT'
    RELEASE = 'RELEASE'
    COPY = 'COPY'
    EXPORT = 'EXPORT'
    IMPLEMENTATION_PHASE = 'IMPLEMENTATION_PHASE'


@dataclass(frozen=True, slots=True)
class VersionedLogicalValue:
    id: str
    producer: str | None = None
    call_scope: str = '$CALL_INSTANCE'
    source_refs: tuple[dict[str, Any], ...] = ()
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StorageSlot:
    id: str
    location: str
    capacity_bytes: int
    ownership: str = 'SCHEDULE'
    allocated_by: str | None = None
    released_by: str | None = None
    device: str = 'cpu'
    resource_refs: tuple[dict[str, Any], ...] = ()
    budget_refs: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class StorageRegion:
    slot: str
    offset_bytes: int
    nbytes: int


@dataclass(frozen=True, slots=True)
class VersionedMaterialization:
    id: str
    value: str
    region: StorageRegion
    version: int
    ready_task: str | None = None
    external_ready: bool = False


@dataclass(frozen=True, slots=True)
class VersionedTask:
    id: str
    kind: VersionedTaskKind
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    allocates: tuple[str, ...] = ()
    releases: tuple[str, ...] = ()
    validates: tuple[str, ...] = ()
    capability: str | None = None
    resource_refs: tuple[dict[str, Any], ...] = ()
    source_refs: tuple[dict[str, Any], ...] = ()
    world_before: tuple[dict[str, Any], ...] = ()
    world_after: tuple[dict[str, Any], ...] = ()
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OrderConstraint:
    before: str
    after: str
    kind: str
    source_refs: tuple[dict[str, Any], ...] = ()


class TemporalMode(str, Enum):
    SEMANTIC = 'SEMANTIC'
    PLANNED = 'PLANNED'
    OBSERVED = 'OBSERVED'


class TemporalEventKind(str, Enum):
    START = 'START'
    SUBMISSION_RETURN = 'SUBMISSION_RETURN'
    COMPLETION = 'COMPLETION'
    READY = 'READY'
    SCOPE_ENTRY = 'SCOPE_ENTRY'
    SCOPE_EXIT = 'SCOPE_EXIT'


class TemporalScopeKind(str, Enum):
    ROOT = 'ROOT'
    COMPOSITE = 'COMPOSITE'
    LOOP = 'LOOP'


class TemporalDurationBasis(str, Enum):
    UNKNOWN = 'UNKNOWN'
    DECLARED_MODEL = 'DECLARED_MODEL'
    OBSERVED = 'OBSERVED'
    SOURCE_CONTROL_MILESTONE = 'SOURCE_CONTROL_MILESTONE'


class TemporalOccupancyKind(str, Enum):
    EXECUTION_DEMAND = 'EXECUTION_DEMAND'
    PHYSICAL_SLOT_BYTES = 'PHYSICAL_SLOT_BYTES'


class TemporalConstraintKind(str, Enum):
    CAUSAL = 'CAUSAL'
    READINESS = 'READINESS'
    RESOURCE_ORDER = 'RESOURCE_ORDER'
    STORAGE_LIFETIME = 'STORAGE_LIFETIME'
    LOOP_CARRIED = 'LOOP_CARRIED'


@dataclass(frozen=True, slots=True)
class TemporalEvent:
    id: str
    kind: TemporalEventKind
    scope: str
    task: str | None = None
    materialization: str | None = None
    timestamp_ticks: int | None = None
    source_refs: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class TemporalScope:
    id: str
    kind: TemporalScopeKind
    entry: str
    exit: str
    parent: str | None = None
    iteration_symbol: str | None = None
    trip_count: int | None = None
    trip_count_symbol: str | None = None
    source_refs: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class TemporalActivity:
    id: str
    task: str
    scope: str
    start: str
    completion: str
    submission_return: str | None = None
    ready_events: tuple[str, ...] = ()
    duration_basis: TemporalDurationBasis = TemporalDurationBasis.UNKNOWN
    duration_lower_ticks: int | None = None
    duration_upper_ticks: int | None = None
    source_refs: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class TemporalOccupancy:
    id: str
    kind: TemporalOccupancyKind
    resource: str
    start: str
    end: str
    scope: str
    units: int | None = None
    activity: str | None = None
    storage_slot: str | None = None
    continues_after_scope: bool = False
    source_refs: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class TemporalConstraint:
    """after[i + iteration_distance] - before[i] lies in the stated bounds."""
    id: str
    before: str
    after: str
    kind: TemporalConstraintKind
    min_delay_ticks: int = 0
    max_delay_ticks: int | None = None
    iteration_distance: int = 0
    loop_scope: str | None = None
    source_refs: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class TemporalModel:
    mode: TemporalMode
    events: tuple[TemporalEvent, ...]
    activities: tuple[TemporalActivity, ...]
    occupancies: tuple[TemporalOccupancy, ...]
    constraints: tuple[TemporalConstraint, ...]
    scopes: tuple[TemporalScope, ...]
    resource_capacities: dict[str, int | None] = field(default_factory=dict)
    tick_numerator: int | None = None
    tick_denominator: int | None = None
    clock_domain: str | None = None
    source_refs: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class VersionedResourceSchedule:
    region_id: str
    values: tuple[VersionedLogicalValue, ...]
    materializations: tuple[VersionedMaterialization, ...]
    storage_slots: tuple[StorageSlot, ...]
    tasks: tuple[VersionedTask, ...]
    call_scope: str = '$CALL_INSTANCE'
    order_constraints: tuple[OrderConstraint, ...] = ()
    resource_refs: tuple[dict[str, Any], ...] = ()
    budget_bytes: int | None = None
    budget_scope: str = ''
    budget_refs: tuple[dict[str, Any], ...] = ()
    proof: dict[str, Any] = field(default_factory=dict)
    unresolved: tuple[dict[str, Any], ...] = ()
    executable: bool = False
    temporal: TemporalModel | None = None


CPU_CAPABILITIES = {
    VersionedTaskKind.ALLOCATE: frozenset(('CPU_ALLOCATE',)),
    VersionedTaskKind.SNAPSHOT: frozenset(('CPU_HOST_SNAPSHOT',)),
    VersionedTaskKind.COMPUTE: frozenset(('CPU_SCALAR_DAG', 'CPU_NUMPY_UFUNC_PROVIDER', 'CPU_SCALAR_SETUP')),
    VersionedTaskKind.STATE_WRITE: frozenset(('CPU_STATE_WRITE',)),
    VersionedTaskKind.SUCCESS_FENCE: frozenset(('CPU_SUCCESS_FENCE',)),
    VersionedTaskKind.COMMIT: frozenset(('CPU_COMMIT',)),
    VersionedTaskKind.RELEASE: frozenset(('CPU_RELEASE',)),
    VersionedTaskKind.EXPORT: frozenset(('CPU_EXPORT',)),
}
_PRODUCING = frozenset((VersionedTaskKind.COMPUTE, VersionedTaskKind.STATE_WRITE, VersionedTaskKind.COMMIT))
_PRECOMMIT = frozenset((VersionedTaskKind.ALLOCATE, VersionedTaskKind.SNAPSHOT,
    VersionedTaskKind.COMPUTE, VersionedTaskKind.STATE_WRITE, VersionedTaskKind.COPY))


def _text(value):
    return type(value) is str and bool(value)


def _integer(value):
    return type(value) is int and value >= 0


def _refs(rows):
    if type(rows) is not tuple or any(type(row) is not dict for row in rows):
        raise ValueError('references must retain original record dictionaries in a tuple')
    encode(rows)


def _unique(rows, cls, label):
    if type(rows) is not tuple or any(type(row) is not cls for row in rows):
        raise ValueError('typed immutable ' + label + ' tuple required')
    result = {}
    for row in rows:
        if not _text(row.id) or row.id in result:
            raise ValueError('invalid or duplicate ' + label + ' ID')
        result[row.id] = row
    return result


def _identities(ids, domain, label):
    if type(ids) is not tuple or any(not _text(v) or v not in domain for v in ids) or len(set(ids)) != len(ids):
        raise ValueError('dangling or duplicate ' + label)


def _overlap(left, right):
    return (left.slot == right.slot and left.nbytes and right.nbytes
        and left.offset_bytes < right.offset_bytes + right.nbytes
        and right.offset_bytes < left.offset_bytes + left.nbytes)


def _validate_temporal_model(model, schedule):
    if type(model) is not TemporalModel or type(model.mode) is not TemporalMode:
        raise ValueError('one typed temporal model and provenance mode required')
    events = _unique(model.events, TemporalEvent, 'temporal event')
    activities = _unique(model.activities, TemporalActivity, 'temporal activity')
    occupancies = _unique(model.occupancies, TemporalOccupancy, 'temporal occupancy')
    constraints = _unique(model.constraints, TemporalConstraint, 'temporal constraint')
    scopes = _unique(model.scopes, TemporalScope, 'temporal scope')
    tasks = {task.id: task for task in schedule.tasks}
    slots = {slot.id: slot for slot in schedule.storage_slots}
    materials = {material.id: material for material in schedule.materializations}
    _refs(model.source_refs)
    if type(model.resource_capacities) is not dict or any(
            not _text(key) or value is not None and not _integer(value)
            for key, value in model.resource_capacities.items()):
        raise ValueError('temporal resource capacities must be nonnegative integers or UNKNOWN')
    tick = (model.tick_numerator, model.tick_denominator)
    if tick != (None, None) and any(type(value) is not int or value <= 0 for value in tick):
        raise ValueError('temporal tick_seconds requires a positive rational pair or UNKNOWN')
    if model.clock_domain is not None and not _text(model.clock_domain):
        raise ValueError('temporal clock domain must be a nonempty name or UNKNOWN')
    roots = [scope for scope in scopes.values() if scope.kind is TemporalScopeKind.ROOT]
    if len(roots) != 1 or roots[0].id != schedule.call_scope or roots[0].parent is not None:
        raise ValueError('one root temporal scope matching the original call scope required')
    for scope in scopes.values():
        _refs(scope.source_refs)
        if type(scope.kind) is not TemporalScopeKind:
            raise ValueError('typed temporal scope kind required')
        if scope is not roots[0] and scope.parent not in scopes:
            raise ValueError('non-root temporal scope needs its original containing scope')
        if (scope.entry not in events or scope.exit not in events or scope.entry == scope.exit
                or events[scope.entry].kind is not TemporalEventKind.SCOPE_ENTRY
                or events[scope.exit].kind is not TemporalEventKind.SCOPE_EXIT
                or events[scope.entry].scope != scope.id or events[scope.exit].scope != scope.id):
            raise ValueError('scope entry/exit must be distinct correctly owned boundary events')
        if scope.kind is TemporalScopeKind.LOOP:
            if (not _text(scope.iteration_symbol)
                    or scope.trip_count is not None and not _integer(scope.trip_count)
                    or scope.trip_count_symbol is not None and not _text(scope.trip_count_symbol)
                    or scope.trip_count is not None and scope.trip_count_symbol is not None):
                raise ValueError('loop scope needs a symbolic iteration variable and one optional trip count')
        elif any(value is not None for value in (
                scope.iteration_symbol, scope.trip_count, scope.trip_count_symbol)):
            raise ValueError('iteration metadata belongs only to a symbolic loop scope')
    scope_ancestors = {}
    for identity in scopes:
        chain = set()
        current = identity
        while current is not None:
            if current in chain:
                raise ValueError('temporal containment contains a cycle')
            chain.add(current)
            current = scopes[current].parent
        scope_ancestors[identity] = chain
    # Each edge means time(after) >= time(before) + lower. Upper bounds add
    # the inverse inequality. Positive-distance recurrences are not unrolled
    # or mistaken for same-iteration cycles.
    edges = []

    def bound(before, after, lower=0, upper=None):
        edges.append((before, after, lower))
        if upper is not None:
            edges.append((after, before, -upper))

    known_times = 0
    anchor = object()
    for event in events.values():
        _refs(event.source_refs)
        if type(event.kind) is not TemporalEventKind or event.scope not in scopes:
            raise ValueError('typed event kind and containing temporal scope required')
        if event.task is not None and event.task not in tasks:
            raise ValueError('temporal event references an unknown original task')
        if event.kind in (TemporalEventKind.SCOPE_ENTRY, TemporalEventKind.SCOPE_EXIT):
            if event.task is not None or event.materialization is not None:
                raise ValueError('scope boundaries are containers, not additional original work')
        elif event.kind is TemporalEventKind.READY:
            if event.materialization is None and event.task is None:
                raise ValueError('ready event needs its original task or materialization')
        elif event.task is None:
            raise ValueError('task lifecycle event requires its original task ID')
        if event.materialization is not None:
            material = materials.get(event.materialization)
            if (event.kind is not TemporalEventKind.READY or material is None
                    or event.task != material.ready_task):
                raise ValueError('data readiness must name its actual materialization producer')
        if event.timestamp_ticks is not None:
            if (not _integer(event.timestamp_ticks) or model.mode is TemporalMode.SEMANTIC
                    or model.clock_domain is None or not event.source_refs
                    or model.mode is TemporalMode.OBSERVED and tick == (None, None)):
                raise ValueError('known timestamps need one declared/observed clock and original evidence')
            bound(anchor, event.id, event.timestamp_ticks, event.timestamp_ticks)
            known_times += 1
        scope = scopes[event.scope]
        bound(scope.entry, event.id)
        bound(event.id, scope.exit)
    for scope in scopes.values():
        if scope.parent is not None:
            parent = scopes[scope.parent]
            bound(parent.entry, scope.entry)
            bound(scope.exit, parent.exit)
    task_activities = defaultdict(list)
    control_milestones = set()
    ready_owners = {}
    pairs = set()
    for activity in activities.values():
        _refs(activity.source_refs)
        if activity.task not in tasks or activity.scope not in scopes:
            raise ValueError('temporal activity must belong to an original task and containing scope')
        for name, kind in ((activity.start, TemporalEventKind.START),
                           (activity.completion, TemporalEventKind.COMPLETION)):
            event = events.get(name)
            if (event is None or event.kind is not kind or event.task != activity.task
                    or event.scope != activity.scope):
                raise ValueError('activity start/completion events must match its original task and scope')
        pair = (activity.start, activity.completion)
        if pair in pairs:
            raise ValueError('duplicate activity would count the same task lifecycle twice')
        pairs.add(pair)
        if type(activity.duration_basis) is not TemporalDurationBasis:
            raise ValueError('typed temporal duration basis required')
        lower, upper = activity.duration_lower_ticks, activity.duration_upper_ticks
        if any(value is not None and not _integer(value) for value in (lower, upper)) or (
                upper is not None and upper < (lower or 0)):
            raise ValueError('duration bounds must be ordered nonnegative integers or UNKNOWN')
        if activity.duration_basis is TemporalDurationBasis.UNKNOWN:
            if lower is not None or upper is not None:
                raise ValueError('UNKNOWN duration must not silently carry a numeric estimate')
        elif activity.duration_basis is TemporalDurationBasis.SOURCE_CONTROL_MILESTONE:
            scope_proof = schedule.proof.get('implementation_scope_cleanup', {})
            marker = scope_proof.get('milestone', {})
            if (model.mode not in (TemporalMode.SEMANTIC, TemporalMode.PLANNED)
                    or (lower, upper) != (0, 0) or not activity.source_refs
                    or tasks[activity.task].kind is not VersionedTaskKind.SUCCESS_FENCE
                    or scope_proof.get('schema') != 'scar.stateful-implementation-scope-proof'
                    or marker.get('task_id') != activity.task
                    or marker.get('basis') != 'SOURCE_PROVEN_ZERO_ADDITIONAL_SERVICE'
                    or marker.get('no_additional_emitted_fence_statement') is not True):
                raise ValueError('zero additional service needs the original source-proven control milestone')
            control_milestones.add(activity.id)
        elif (not activity.source_refs or lower is None and upper is None
                or activity.duration_basis is TemporalDurationBasis.DECLARED_MODEL
                    and model.mode is not TemporalMode.PLANNED
                or activity.duration_basis is TemporalDurationBasis.OBSERVED
                    and (model.mode is not TemporalMode.OBSERVED or tick == (None, None)
                         or model.clock_domain is None)):
            raise ValueError('duration evidence must match the temporal model provenance mode')
        bound(activity.start, activity.completion, lower or 0, upper)
        if activity.submission_return is not None:
            event = events.get(activity.submission_return)
            if (event is None or event.kind is not TemporalEventKind.SUBMISSION_RETURN
                    or event.task != activity.task or event.scope != activity.scope):
                raise ValueError('submission-return event must match its original activity')
            # Completion may precede API return for a short asynchronous task.
            # Submission return is not a data-readiness dependency.
            bound(activity.start, event.id)
        _identities(activity.ready_events, events, 'activity ready event')
        for identity in activity.ready_events:
            event = events[identity]
            if (event.kind is not TemporalEventKind.READY or event.task != activity.task
                    or event.scope != activity.scope):
                raise ValueError('activity ready event must match its actual producer')
            if identity in ready_owners:
                raise ValueError('ready event must identify one producing activity, not every phase of a task')
            ready_owners[identity] = activity.id
            bound(activity.completion, identity)
        task_activities[activity.task].append(activity)
    # Original task dependencies remain true completion dependencies. Temporal
    # metadata cannot erase the source data/storage safety checked above.
    for task in tasks.values():
        for dependency in task.depends_on:
            for before in task_activities[dependency]:
                for after in task_activities[task.id]:
                    bound(before.completion, after.start)
    exported = {materials[name].region.slot for task in tasks.values()
                if task.kind is VersionedTaskKind.EXPORT for name in task.reads}
    physical = set()
    demands = set()
    resource_kinds = {}
    for occupancy in occupancies.values():
        _refs(occupancy.source_refs)
        if (type(occupancy.kind) is not TemporalOccupancyKind or occupancy.scope not in scopes
                or occupancy.resource not in model.resource_capacities
                or occupancy.start not in events or occupancy.end not in events
                or type(occupancy.continues_after_scope) is not bool):
            raise ValueError('occupancy needs typed resource, scope and endpoint declarations')
        previous_kind = resource_kinds.setdefault(occupancy.resource, occupancy.kind)
        if previous_kind is not occupancy.kind:
            raise ValueError('one temporal resource cannot mix execution units and physical-slot bytes')
        if any(occupancy.scope not in scope_ancestors[events[identity].scope]
               for identity in (occupancy.start, occupancy.end)):
            raise ValueError('occupancy endpoints escape their declared containing scope')
        if occupancy.kind is TemporalOccupancyKind.EXECUTION_DEMAND:
            activity = activities.get(occupancy.activity)
            if (activity is None or occupancy.storage_slot is not None or occupancy.continues_after_scope
                    or occupancy.activity in control_milestones
                    or occupancy.units is not None and (not _integer(occupancy.units) or occupancy.units == 0)
                    or occupancy.scope != activity.scope
                    or any(events[name].task != activity.task for name in (occupancy.start, occupancy.end))):
                raise ValueError('execution demand must reference one original activity, never free/zero units')
            key = (activity.id, occupancy.resource)
            if key in demands:
                raise ValueError('duplicate activity/resource occupation would double-count work')
            demands.add(key)
        else:
            slot = slots.get(occupancy.storage_slot)
            start, end = events[occupancy.start], events[occupancy.end]
            if (slot is None or slot.ownership != 'SCHEDULE' or slot.id in physical
                    or occupancy.activity is not None or type(occupancy.units) is not int
                    or occupancy.units != slot.capacity_bytes
                    or start.task != slot.allocated_by or start.kind is not TemporalEventKind.START):
                raise ValueError('physical-slot bytes must count one actual owned allocation exactly once')
            if occupancy.continues_after_scope:
                if (slot.released_by is not None or slot.id not in exported
                        or end.kind is not TemporalEventKind.SCOPE_EXIT or end.scope != occupancy.scope):
                    raise ValueError('retained exported storage must explicitly continue after its scope boundary')
            elif (slot.released_by is None or end.task != slot.released_by
                    or end.kind is not TemporalEventKind.COMPLETION):
                raise ValueError('storage lifetime ends only at its actual release completion, not an invented free')
            physical.add(slot.id)
        bound(occupancy.start, occupancy.end)
    periodic = 0
    for constraint in constraints.values():
        _refs(constraint.source_refs)
        if (type(constraint.kind) is not TemporalConstraintKind or constraint.before not in events
                or constraint.after not in events or not _integer(constraint.min_delay_ticks)
                or constraint.max_delay_ticks is not None and (
                    not _integer(constraint.max_delay_ticks)
                    or constraint.max_delay_ticks < constraint.min_delay_ticks)
                or not _integer(constraint.iteration_distance)):
            raise ValueError('temporal constraint requires valid event IDs, bounds and iteration distance')
        if constraint.loop_scope is not None:
            scope = scopes.get(constraint.loop_scope)
            if (scope is None or scope.kind is not TemporalScopeKind.LOOP
                    or any(scope.id not in scope_ancestors[events[name].scope]
                           for name in (constraint.before, constraint.after))):
                raise ValueError('iteration constraint endpoints must belong to the same enclosing loop')
        if constraint.iteration_distance:
            if constraint.kind is not TemporalConstraintKind.LOOP_CARRIED or constraint.loop_scope is None:
                raise ValueError('positive iteration distance requires an explicit loop-carried constraint')
            periodic += 1
        else:
            if constraint.kind is TemporalConstraintKind.LOOP_CARRIED:
                raise ValueError('loop-carried dependency needs a positive iteration distance')
            bound(constraint.before, constraint.after, constraint.min_delay_ticks, constraint.max_delay_ticks)
    # Difference-bound feasibility only; no timestamps are chosen or optimized.
    # A zero-delay cycle is equality, not automatically an invalid DAG cycle.
    distance = {name: 0 for name in (*events, anchor)}
    for step in range(len(distance)):
        changed = False
        for before, after, lower in edges:
            candidate = distance[before] + lower
            if candidate > distance[after]:
                distance[after] = candidate
                changed = True
        if not changed:
            break
    else:
        raise ValueError('same-iteration temporal bounds are inconsistent')
    owned = {slot.id for slot in slots.values() if slot.ownership == 'SCHEDULE'}
    occupied_activities = {occupancy.activity for occupancy in occupancies.values()
                          if occupancy.kind is TemporalOccupancyKind.EXECUTION_DEMAND}
    summary = dict(mode=model.mode.value, events=len(events), activities=len(activities),
        scopes=len(scopes), loop_scopes=sum(scope.kind is TemporalScopeKind.LOOP for scope in scopes.values()),
        symbolic_iteration_constraints=periodic, iterations_unrolled=0,
        occupancies=len(occupancies), physical_slots_counted_once=len(physical),
        physical_slot_coverage_complete=physical == owned,
        activity_task_coverage_complete={activity.task for activity in activities.values()} == set(tasks),
        activities_without_execution_demand=len(set(activities) - occupied_activities),
        execution_demand_coverage_complete=set(activities) - control_milestones <= occupied_activities,
        unbound_materialization_ready_events=sum(event.kind is TemporalEventKind.READY
            and event.materialization is not None and event.task is not None
            and event.id not in ready_owners for event in events.values()),
        task_dependency_interpretation='ORIGINAL_FULL_TASK_COMPLETION_TO_DEPENDENT_TASK_PHASE_STARTS',
        known_timestamps=known_times,
        unknown_durations=sum(activity.duration_basis is TemporalDurationBasis.UNKNOWN
                              for activity in activities.values()),
        unknown_execution_demands=sum(occupancy.kind is TemporalOccupancyKind.EXECUTION_DEMAND
                                     and occupancy.units is None for occupancy in occupancies.values()),
        same_iteration_bounds_consistent=True, periodic_feasibility_proven=False,
        resource_capacity_feasibility_proven=False, execution_authorized_by_temporal_model=False)
    if control_milestones:
        summary['source_control_milestones'] = len(control_milestones)
        summary['executable_activities_without_execution_demand'] = len(
            set(activities) - control_milestones - occupied_activities)
    return summary


def validate_temporal_model(model, schedule):
    """Check typed time facts against original tasks without authorizing execution."""
    validate_versioned_schedule(replace(schedule, temporal=None))
    return _validate_temporal_model(model, schedule)


def validate_versioned_schedule(schedule):
    """Check producers, readiness, allocation and overwrite/release causality.

    Tuple order never establishes happens-before. STATE_WRITE/COMMIT alone
    have intrinsic old-state-read-before-new-state-write semantics; all other
    readers of overlapping old versions must strictly precede the new writer.
    """
    K = VersionedTaskKind
    if type(schedule) is not VersionedResourceSchedule or not _text(schedule.region_id) or not _text(schedule.call_scope):
        raise ValueError('versioned call-scoped resource schedule required')
    values = _unique(schedule.values, VersionedLogicalValue, 'logical value')
    materials = _unique(schedule.materializations, VersionedMaterialization, 'materialization')
    slots = _unique(schedule.storage_slots, StorageSlot, 'storage slot')
    tasks = _unique(schedule.tasks, VersionedTask, 'task')
    if type(schedule.executable) is not bool or schedule.executable and schedule.unresolved:
        raise ValueError('unresolved schedules cannot be executable')
    if type(schedule.proof) is not dict:
        raise ValueError('schedule proof must retain a record dictionary')
    for refs in (schedule.resource_refs, schedule.budget_refs, schedule.unresolved): _refs(refs)
    if schedule.budget_bytes is not None and not _integer(schedule.budget_bytes):
        raise ValueError('explicit nonnegative storage budget required')
    if type(schedule.budget_scope) is not str:
        raise ValueError('budget scope must remain explicit')

    users = {key: [] for key in tasks}; indegrees = {}
    for task in tasks.values():
        if type(task.kind) is not K:
            raise ValueError('typed versioned task kind required')
        if type(task.attrs) is not dict or task.capability is not None and not _text(task.capability):
            raise ValueError('task attributes and optional lowering capability must be typed')
        for refs in (task.resource_refs, task.source_refs, task.world_before, task.world_after): _refs(refs)
        for ids, domain, label in ((task.depends_on, tasks, 'task dependency'), (task.reads, materials, 'read'),
                (task.writes, materials, 'write'), (task.allocates, slots, 'allocation'),
                (task.releases, slots, 'release'), (task.validates, tasks, 'validated task')):
            _identities(ids, domain, label)
        indegrees[task.id] = len(task.depends_on)
        for dependency in task.depends_on: users[dependency].append(task.id)
        if task.kind is K.ALLOCATE:
            if not task.allocates or task.reads or task.writes or task.releases:
                raise ValueError('allocation prepares slots, never values/readiness')
        elif task.allocates:
            raise ValueError('only ALLOCATE may prepare storage')
        if task.kind is K.RELEASE:
            if not task.releases or task.reads or task.writes:
                raise ValueError('release references physical slots only')
        elif task.releases:
            raise ValueError('only RELEASE may release storage')
        if task.capability == 'CPU_HOST_COPY_REGION' and task.kind is not K.COPY:
            raise ValueError('host copy region must have the explicit COPY task kind')
        if task.kind is K.COPY and task.capability == 'CPU_HOST_COPY_REGION':
            pairs = task.attrs.get('copy_pairs')
            children = task.attrs.get('child_task_ids')
            shape = task.attrs.get('shape')
            if (len(task.reads) < 2 or len(task.reads) != len(task.writes)
                    or task.attrs.get('stage_kind') != 'FUSED_SNAPSHOT_ROOT'
                    or task.attrs.get('dtype') != 'float64'
                    or type(shape) not in (list, tuple) or len(shape) != 2
                    or any(type(n) is not int or n <= 0 for n in shape)
                    or type(children) not in (list, tuple) or len(children) != len(task.reads)
                    or any(not _text(child) or child == task.id for child in children)
                    or len(set(children)) != len(children)
                    or type(pairs) not in (list, tuple) or len(pairs) != len(children)
                    or any(type(pair) is not dict or set(pair) != {'read', 'write', 'source_task'}
                        or pair != dict(read=source, write=target, source_task=child)
                        for pair, source, target, child in zip(pairs, task.reads, task.writes, children))
                    or task.world_before or task.world_after or task.validates
                    or task.allocates or task.releases or not task.source_refs):
                raise ValueError('fused host copy requires exact independent raw-copy ports and shared domain')
        elif task.kind in (K.SNAPSHOT, K.COPY) and (len(task.reads) != 1 or len(task.writes) != 1):
            raise ValueError('snapshot/copy requires one real source and destination')
        if task.kind in (K.STATE_WRITE, K.COMMIT) and (not task.reads or not task.writes):
            raise ValueError('state transition requires prior data/state and a new materialized version')
        if task.kind is K.COMPUTE and not task.writes and task.capability != 'CPU_SCALAR_SETUP':
            raise ValueError('compute requires outputs, except explicit scalar setup without array payload')
        if task.kind is K.IMPLEMENTATION_PHASE:
            attrs = task.attrs
            if (schedule.executable or task.capability != 'CPU_IMPLEMENTATION_PHASE'
                    or task.writes or task.allocates or task.releases or task.validates
                    or not task.source_refs or not _text(attrs.get('phase_kind'))
                    or attrs.get('actor') not in ('CALLER_UNBOUND', 'SELECTED_TEAM', 'WORKER')
                    or attrs.get('failure_domain') not in ('PRECOMMIT', 'POSTCOMMIT')):
                raise ValueError('source-backed non-executable implementation phase cannot write or own payload work')
            if any(not (type(value) is bool or type(value) is str and value == 'UNKNOWN')
                   for value in (attrs.get('may_fail'), attrs.get('blocks'))):
                raise ValueError('phase may_fail and blocks must be explicit bool or UNKNOWN')
            effects = attrs.get('phase_effects')
            if type(effects) not in (tuple, list) or not effects or any(not _text(value) for value in effects):
                raise ValueError('implementation phase effects must be explicit nonempty strings')
            if attrs['failure_domain'] == 'POSTCOMMIT' and attrs['may_fail'] is not False:
                raise ValueError('normal-only postcommit implementation phase cannot fail or permit fallback')
        if task.kind in (K.SUCCESS_FENCE, K.EXPORT) and task.writes:
            raise ValueError('fence/export cannot modify values or external state')
        if task.kind is K.EXPORT and not task.reads:
            raise ValueError('export requires a ready value')
        if task.kind is not K.SUCCESS_FENCE and task.validates:
            raise ValueError('only SUCCESS_FENCE may validate precommit work')
        if schedule.executable and task.capability not in CPU_CAPABILITIES.get(task.kind, ()):
            raise ValueError('task lowering capability unsupported; generic COPY/device transfer is not CPU snapshot')

    queue = deque(key for key in tasks if not indegrees[key]); ancestors = {}; order = []
    while queue:
        key = queue.popleft(); ancestors[key] = set(tasks[key].depends_on)
        for dependency in tasks[key].depends_on: ancestors[key].update(ancestors[dependency])
        order.append(key)
        for user in users[key]:
            indegrees[user] -= 1
            if not indegrees[user]: queue.append(user)
    if len(order) != len(tasks): raise ValueError('task dependency graph contains a cycle')
    if type(schedule.order_constraints) is not tuple:
        raise ValueError('typed order constraints required')
    for constraint in schedule.order_constraints:
        if (type(constraint) is not OrderConstraint or constraint.kind not in ('WORLD', 'CONTROL', 'STATE', 'SOURCE_ORDER')
                or constraint.before not in tasks or constraint.after not in tasks):
            raise ValueError('invalid original state/control order constraint')
        _refs(constraint.source_refs)
        if constraint.before not in ancestors[constraint.after]:
            raise ValueError('original state/control order is not represented by happens-before')

    for slot in slots.values():
        _refs(slot.resource_refs); _refs(slot.budget_refs)
        if (not _text(slot.location) or not _text(slot.device) or not _integer(slot.capacity_bytes)
                or slot.ownership not in ('EXTERNAL', 'SCHEDULE')):
            raise ValueError('storage requires a physical location, capacity and ownership')
        if slot.ownership == 'EXTERNAL':
            if slot.allocated_by is not None or slot.released_by is not None:
                raise ValueError('external caller storage cannot be allocated/released by the schedule')
        else:
            allocation = tasks.get(slot.allocated_by)
            if allocation is None or allocation.kind is not K.ALLOCATE or slot.id not in allocation.allocates:
                raise ValueError('scheduled slot needs its actual allocation task')
            if slot.released_by is not None:
                release = tasks.get(slot.released_by)
                if release is None or release.kind is not K.RELEASE or slot.id not in release.releases:
                    raise ValueError('slot release declaration differs from its actual release task')
        if schedule.executable and slot.device != 'cpu':
            raise ValueError('CPU lowering capabilities do not admit device copies or non-CPU storage')

    writers = defaultdict(list); consumers = defaultdict(set); slot_uses = defaultdict(set)
    for task in tasks.values():
        for identity in task.writes: writers[identity].append(task.id)
        for identity in task.reads: consumers[identity].add(task.id)
        for identity in (*task.reads, *task.writes):
            material = materials[identity]
            if (type(material.region) is not StorageRegion or not _text(material.region.slot)
                    or material.region.slot not in slots):
                raise ValueError('materialization references missing physical storage region')
            slot = slots[material.region.slot]; slot_uses[slot.id].add(task.id)
            if slot.allocated_by is not None and slot.allocated_by not in ancestors[task.id]:
                raise ValueError('physical storage used before its allocation dependency')
        for identity in task.allocates:
            if slots[identity].ownership != 'SCHEDULE' or slots[identity].allocated_by != task.id:
                raise ValueError('allocation/slot ownership mismatch')
        for identity in task.releases:
            if slots[identity].ownership != 'SCHEDULE' or slots[identity].released_by != task.id:
                raise ValueError('release/slot ownership mismatch')
    for material in materials.values():
        if (not _text(material.value) or material.value not in values or type(material.region) is not StorageRegion
                or not _text(material.region.slot) or material.region.slot not in slots):
            raise ValueError('dangling logical value or storage region')
        region, slot = material.region, slots[material.region.slot]
        if (not _integer(region.offset_bytes) or not _integer(region.nbytes)
                or region.offset_bytes + region.nbytes > slot.capacity_bytes or not _integer(material.version)
                or type(material.external_ready) is not bool):
            raise ValueError('materialization region/version exceeds allocation or is malformed')
        value = values[material.value]
        if material.external_ready:
            if (material.ready_task is not None or writers[material.id] or slot.ownership != 'EXTERNAL'
                    or value.producer is not None):
                raise ValueError('external readiness cannot invent a scheduled logical/storage producer')
        else:
            writer = tasks.get(material.ready_task)
            if (writer is None or writer.kind not in (*_PRODUCING, K.SNAPSHOT, K.COPY)
                    or writers[material.id] != [writer.id]):
                raise ValueError('materialized version needs its unique actual producer, not allocation/fence')
            if slot.ownership == 'EXTERNAL' and writer.kind is not K.COMMIT:
                raise ValueError('only COMMIT may write external state; EXPORT is not a state write')
            if slot.ownership == 'SCHEDULE' and writer.kind is K.COMMIT:
                raise ValueError('COMMIT is reserved for explicit external state transitions')
            if value.producer is not None and value.producer != writer.id and value.producer not in ancestors[writer.id]:
                raise ValueError('materialization readiness precedes its logical producer')
        for consumer in consumers[material.id]:
            if not material.external_ready and material.ready_task not in ancestors[consumer]:
                raise ValueError('value read before its actual readiness dependency')
    for value in values.values():
        _refs(value.source_refs)
        if type(value.attrs) is not dict or value.producer is not None and not _text(value.producer):
            raise ValueError('logical value attributes and optional producer must be typed')
        if value.call_scope != schedule.call_scope:
            raise ValueError('different invocation scopes must not share logical values')
        if value.producer is not None:
            producer = tasks.get(value.producer)
            if (producer is None or producer.kind not in _PRODUCING
                    or not any(materials[key].value == value.id for key in producer.writes)):
                raise ValueError('logical producer must really produce this value')
        elif not any(m.value == value.id and m.external_ready for m in materials.values()):
            raise ValueError('unproduced logical value needs actual external readiness')

    by_slot = defaultdict(list)
    for material in materials.values(): by_slot[material.region.slot].append(material)
    for versions in by_slot.values():
        versions.sort(key=lambda m: (m.version, m.id))
        for index, old in enumerate(versions):
            for new in versions[index + 1:]:
                if not _overlap(old.region, new.region): continue
                if old.version == new.version or new.external_ready:
                    raise ValueError('overlapping materializations need distinct ordered physical versions')
                writer = new.ready_task
                if not old.external_ready and old.ready_task not in ancestors[writer]:
                    raise ValueError('overlapping slot versions have unordered producers')
                for consumer in consumers[old.id]:
                    if tasks[consumer].kind is K.EXPORT:
                        raise ValueError('exported immutable value storage cannot be overwritten')
                    if consumer == writer and tasks[writer].kind in (K.STATE_WRITE, K.COMMIT):
                        continue  # Typed task intrinsically reads old state before writing new state.
                    if consumer not in ancestors[writer]:
                        raise ValueError('overlapping old-version consumer must happen-before slot overwrite')
    for task in tasks.values():
        if task.kind in (K.SNAPSHOT, K.COPY):
            for read, write in zip(task.reads, task.writes):
                source, target = materials[read], materials[write]
                if source.value != target.value or source.region.nbytes != target.region.nbytes:
                    raise ValueError('snapshot/copy must preserve logical identity and represented extent')
                if slots[target.region.slot].ownership != 'SCHEDULE' or _overlap(source.region, target.region):
                    raise ValueError('snapshot/copy needs separate private destination storage')
            if task.capability == 'CPU_HOST_COPY_REGION':
                regions = [materials[key].region for key in (*task.reads, *task.writes)]
                extent = task.attrs['shape'][0] * task.attrs['shape'][1] * 8
                if any(region.nbytes != extent or slots[region.slot].device != 'cpu' for region in regions):
                    raise ValueError('fused host copies must retain concrete CPU float64 extents')
                sources = [materials[key].region for key in task.reads]
                destinations = [materials[key].region for key in task.writes]
                if (any(_overlap(target, source) for target in destinations for source in sources)
                        or any(_overlap(left, right) for i, left in enumerate(destinations)
                            for right in destinations[i+1:])):
                    raise ValueError('fused host copies require disjoint private writes and no cross-copy alias')
        elif task.kind in (K.STATE_WRITE, K.COMMIT):
            for identity in task.writes:
                destination = materials[identity]
                previous = [materials[source] for source in task.reads
                    if _overlap(materials[source].region, destination.region)
                    and materials[source].version < destination.version]
                if not previous:
                    raise ValueError('state write must consume the prior overlapping state version')
                logical_producer = values[destination.value].producer
                if destination.value in {old.value for old in previous}:
                    raise ValueError('state transition must produce a distinct new immutable logical value')
                if task.kind is K.STATE_WRITE and logical_producer != task.id:
                    raise ValueError('state write must produce a distinct new immutable logical value')
                if task.kind is K.COMMIT and logical_producer != task.id:
                    # A final-root commit may materialize the already-produced
                    # complete private result. It is not a second logical
                    # computation. Partial ROI overlays need a new value unless
                    # the owning profile replays and references full identity.
                    source_id = task.attrs.get('identity_source_material')
                    reference = task.attrs.get('identity_proof_ref')
                    source = materials.get(source_id) if _text(source_id) else None
                    pointer = reference.get('schedule_pointer') if type(reference) is dict else None
                    proof = {'proof': schedule.proof}
                    if type(pointer) is str and pointer.startswith('#/proof/'):
                        try:
                            for key in pointer[2:].split('/'):
                                proof = proof[key.replace('~1', '/').replace('~0', '~')]
                        except (KeyError, TypeError): proof = None
                    else: proof = None
                    if (task.attrs.get('logical_identity') != 'VALUE_PRESERVING_FINAL_ROOT_COMMIT'
                            or type(proof) is not dict or not proof or source is None
                            or source_id not in task.reads or source.external_ready
                            or slots[source.region.slot].ownership != 'SCHEDULE'
                            or source.value != destination.value
                            or logical_producer not in ancestors[task.id]
                            or source.region.slot == destination.region.slot
                            or source.region.offset_bytes != 0 or destination.region.offset_bytes != 0
                            or source.region.nbytes != slots[source.region.slot].capacity_bytes
                            or destination.region.nbytes != slots[destination.region.slot].capacity_bytes
                            or source.region.nbytes != destination.region.nbytes):
                        raise ValueError('value-preserving commit needs a ready private full-result materialization and profile identity proof')

    precommit = {t.id for t in tasks.values() if t.kind in _PRECOMMIT or
                 t.kind is K.IMPLEMENTATION_PHASE and t.attrs['failure_domain'] == 'PRECOMMIT'}
    fences = []
    for task in tasks.values():
        if task.kind is K.SUCCESS_FENCE:
            if set(task.validates) != precommit or not precommit.issubset(ancestors[task.id]):
                raise ValueError('success fence must cover every precommit allocation/computation/state task')
            fences.append(task.id)
    for task in tasks.values():
        if task.kind is K.COMMIT and not any(fence in ancestors[task.id] for fence in fences):
            raise ValueError('external commit requires a complete precommit success fence')
    exported_slots = {materials[key].region.slot for task in tasks.values() if task.kind is K.EXPORT for key in task.reads}
    for slot in slots.values():
        if slot.released_by is None:
            if schedule.executable and slot.ownership == 'SCHEDULE' and slot.id not in exported_slots:
                raise ValueError('scheduled storage must be released or explicitly retained by export')
            continue
        if slot.id in exported_slots: raise ValueError('escaping storage cannot be released')
        if slot.allocated_by not in ancestors[slot.released_by]:
            raise ValueError('release must follow allocation')
        if any(use not in ancestors[slot.released_by] for use in slot_uses[slot.id]):
            raise ValueError('storage release is unordered with or precedes a physical use')
    total = sum(slot.capacity_bytes for slot in slots.values())
    if schedule.budget_bytes is not None and total > schedule.budget_bytes:
        raise ValueError('fixed storage capacities exceed declared region budget')
    if schedule.executable and (schedule.budget_bytes is None or not schedule.budget_scope or not schedule.budget_refs):
        raise ValueError('executable fixed-storage plan needs an explicit budget scope and original budget reference')
    summary = dict(schema_version=2 if schedule.temporal is None else 3,
        tasks=len(tasks), logical_values=len(values), materializations=len(materials),
        storage_slots=len(slots), topological_order=order, executable=schedule.executable,
        fixed_capacity_bytes=total, overwrite_happens_before_verified=True, external_commit_fenced=True)
    if schedule.temporal is not None:
        summary['temporal'] = _validate_temporal_model(schedule.temporal, schedule)
    return summary


def schedule_document(schedule):
    validate_versioned_schedule(schedule)
    value = encode(schedule)
    if schedule.temporal is None:
        value.pop('temporal')
    return dict(schema='scar.resource-schedule', schema_version=2 if schedule.temporal is None else 3, **value)


def schedule_from_dict(document):
    value = dict(document)
    schema, version = value.pop('schema', None), value.pop('schema_version', None)
    if schema != 'scar.resource-schedule' or type(version) is not int or version not in (2, 3):
        raise ValueError('unsupported versioned resource schedule schema')
    if version == 2:
        if 'temporal' in value:
            raise ValueError('legacy v2 schedule must not carry temporal fields')
        value['temporal'] = None
    elif value.get('temporal') is None:
        raise ValueError('v3 resource schedule requires an explicit temporal model')
    result = decode(VersionedResourceSchedule, value)
    validate_versioned_schedule(result)
    return result


validate_schedule = validate_versioned_schedule
__all__ = ['VersionedLogicalValue', 'StorageSlot', 'StorageRegion', 'VersionedMaterialization',
    'VersionedTask', 'VersionedTaskKind', 'OrderConstraint', 'VersionedResourceSchedule',
    'TemporalMode', 'TemporalEventKind', 'TemporalScopeKind', 'TemporalDurationBasis',
    'TemporalOccupancyKind', 'TemporalConstraintKind', 'TemporalEvent', 'TemporalScope',
    'TemporalActivity', 'TemporalOccupancy', 'TemporalConstraint', 'TemporalModel', 'validate_temporal_model',
    'CPU_CAPABILITIES', 'schedule_document', 'schedule_from_dict', 'validate_schedule', 'validate_versioned_schedule']
