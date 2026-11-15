"""Call-scoped immutable values and versioned physical storage task graphs.

Version numbers order uses of one allocation; they are not source epochs,
LogicalVersions or content-equality proofs. This bounded CPU lowering contract
does not replace the program IR or admit generic/device transfers.
"""
from collections import defaultdict, deque
from dataclasses import dataclass, field
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
        if task.kind in (K.SNAPSHOT, K.COPY) and (len(task.reads) != 1 or len(task.writes) != 1):
            raise ValueError('snapshot/copy requires one real source and destination')
        if task.kind in (K.STATE_WRITE, K.COMMIT) and (not task.reads or not task.writes):
            raise ValueError('state transition requires prior data/state and a new materialized version')
        if task.kind is K.COMPUTE and not task.writes and task.capability != 'CPU_SCALAR_SETUP':
            raise ValueError('compute requires outputs, except explicit scalar setup without array payload')
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
            source, target = materials[task.reads[0]], materials[task.writes[0]]
            if source.value != target.value or source.region.nbytes != target.region.nbytes:
                raise ValueError('snapshot/copy must preserve logical identity and represented extent')
            if slots[target.region.slot].ownership != 'SCHEDULE' or _overlap(source.region, target.region):
                raise ValueError('snapshot/copy needs separate private destination storage')
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

    precommit = {t.id for t in tasks.values() if t.kind in _PRECOMMIT}
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
    return dict(schema_version=2, tasks=len(tasks), logical_values=len(values), materializations=len(materials),
        storage_slots=len(slots), topological_order=order, executable=schedule.executable,
        fixed_capacity_bytes=total, overwrite_happens_before_verified=True, external_commit_fenced=True)


def schedule_document(schedule):
    validate_versioned_schedule(schedule)
    return dict(schema='scar.resource-schedule', schema_version=2, **encode(schedule))


def schedule_from_dict(document):
    value = dict(document)
    if (value.pop('schema', None), value.pop('schema_version', None)) != ('scar.resource-schedule', 2):
        raise ValueError('unsupported versioned resource schedule schema')
    result = decode(VersionedResourceSchedule, value)
    validate_versioned_schedule(result)
    return result


validate_schedule = validate_versioned_schedule
__all__ = ['VersionedLogicalValue', 'StorageSlot', 'StorageRegion', 'VersionedMaterialization',
    'VersionedTask', 'VersionedTaskKind', 'OrderConstraint', 'VersionedResourceSchedule',
    'CPU_CAPABILITIES', 'schedule_document', 'schedule_from_dict', 'validate_schedule', 'validate_versioned_schedule']
