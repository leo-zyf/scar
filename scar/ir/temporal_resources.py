"""Typed hardware bindings for existing temporal occupancies, not a task graph.

Candidate locations are possibilities, CPU cores are contention groups, and
reported inventory capacity never becomes an available allocation budget.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re

from scar.ir.program.providers import ProviderArtifact
from scar.ir.program.target_hardware import TargetResourceRef, HardwareThread, CpuCore, MemoryDomain, MemoryKind
from scar.ir.record_codec import decode, encode
from scar.ir.versioned_schedule import TemporalOccupancyKind, schedule_document, validate_versioned_schedule

SCHEMA = 'scar.temporal-resource-projection'


@dataclass(frozen=True, slots=True)
class ExecutionBinding:
    occupancy: str
    resource: str
    exact_thread: TargetResourceRef | None
    core_refs: tuple[TargetResourceRef, ...]
    candidate_threads: tuple[TargetResourceRef, ...]
    units: int | None
    source_refs: tuple[dict, ...] = ()


@dataclass(frozen=True, slots=True)
class StorageBinding:
    occupancy: str
    slot: str
    candidate_memories: tuple[TargetResourceRef, ...]
    exact_memory: TargetResourceRef | None
    payload_bytes: int
    source_refs: tuple[dict, ...] = ()


@dataclass(frozen=True, slots=True)
class OriginalBudget:
    bytes: int | None
    scope: str
    source_refs: tuple[dict, ...]


@dataclass(frozen=True, slots=True)
class ResourceProjection:
    target: ProviderArtifact
    source_schedule_sha256: str
    executions: tuple[ExecutionBinding, ...]
    storages: tuple[StorageBinding, ...]
    budget: OriginalBudget
    unresolved: tuple[dict, ...] = ()
    source_refs: tuple[dict, ...] = ()


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _typed(value):
    _need(type(value) is ResourceProjection, 'typed temporal resource projection required')
    # This also validates nested TargetResourceRef/ProviderArtifact and exact
    # scalar/tuple types; codec admission alone establishes no source authority.
    detached = decode(ResourceProjection, encode(value))
    _need(detached.target.schema == 'scar.target-hardware-graph'
          and Path(detached.target.uri).is_absolute(), 'original absolute target snapshot required')
    _need(re.fullmatch('[0-9a-f]{64}', detached.source_schedule_sha256) is not None,
          'source temporal schedule digest required')
    _need(detached.budget.bytes is None or detached.budget.bytes >= 0, 'nonnegative original budget or UNKNOWN required')
    for row in (*detached.executions, *detached.storages):
        _need(bool(row.occupancy), 'binding must name an original temporal occupancy')
    return detached


def projection_document(projection):
    """Encode typed records; source/inventory admission is a separate check."""
    return dict(schema=SCHEMA, schema_version=1, **encode(_typed(projection)))


def projection_from_dict(document):
    """Decode exact fields without probing hardware or asserting placement."""
    value = dict(document)
    _need(value.pop('schema', None) == SCHEMA and type(value.get('schema_version')) is int
          and value.pop('schema_version') == 1, 'unsupported temporal resource projection schema')
    return _typed(decode(ResourceProjection, value))


def validate_resource_projection(projection, schedule, *, resources):
    """Check bindings against the original time graph and retained inventory.

    The caller separately replays the target snapshot and candidate reachability
    from its original hardware tables. This validator checks references and
    units, without inventing timing, reservations, NUMA residency or free RAM.
    """
    projection = _typed(projection)
    source = validate_versioned_schedule(schedule)
    _need(schedule.temporal is not None, 'explicit temporal schedule required')
    _need(projection.source_schedule_sha256 == _digest(schedule_document(schedule)),
          'resource projection refers to a different temporal source schedule')
    _need(projection.budget == OriginalBudget(schedule.budget_bytes, schedule.budget_scope, schedule.budget_refs),
          'original budget bytes, scope and evidence must be preserved without reinterpretation')
    inventory = {row.id: row for row in resources}
    _need(len(inventory) == len(resources), 'retained inventory resource identities must be unique')
    by_wire = {row.id.wire: row for row in resources}
    occupied = {row.id: row for row in schedule.temporal.occupancies}
    expected_execution = {key for key, row in occupied.items()
                          if row.kind is TemporalOccupancyKind.EXECUTION_DEMAND}
    expected_storage = set(occupied) - expected_execution
    _need(len(projection.executions) == len(expected_execution)
          and {row.occupancy for row in projection.executions} == expected_execution,
          'every original execution occupancy requires exactly one binding')
    _need(len(projection.storages) == len(expected_storage)
          and {row.occupancy for row in projection.storages} == expected_storage,
          'every original physical-slot occupancy requires exactly one binding')

    def original_refs(refs, kind):
        _need(len({ref.resource for ref in refs}) == len(refs), 'duplicate hardware binding references')
        result = []
        for ref in refs:
            _need(ref.snapshot == projection.target, 'hardware reference belongs to a different target snapshot')
            record = inventory.get(ref.resource)
            _need(type(record) is kind, 'hardware binding reference has the wrong original resource kind')
            result.append(record)
        return result

    all_threads, all_cores = set(), set()
    for row in projection.executions:
        occupancy = occupied[row.occupancy]
        _need(row.resource == occupancy.resource and row.units == occupancy.units,
              'execution resource or demand differs from the original temporal occupancy')
        _need(row.units is None or row.units > 0, 'UNKNOWN execution units cannot become zero/free work')
        threads = original_refs(row.candidate_threads, HardwareThread)
        cores = original_refs(row.core_refs, CpuCore)
        source_resource = by_wire.get(occupancy.resource)
        if type(source_resource) is HardwareThread:
            _need(row.exact_thread is not None and row.exact_thread.resource == source_resource.id,
                  'exact thread must preserve the original occupancy HardwareThread identity')
        else:
            _need(row.exact_thread is None, 'unknown service has no source authority for an exact thread binding')
        if row.exact_thread is not None:
            original_refs((row.exact_thread,), HardwareThread)
            _need(row.candidate_threads == (row.exact_thread,), 'exact thread requires its singleton candidate domain')
        expected_cores = {record.id for record in inventory.values() if type(record) is CpuCore
                          and any(thread.cpu_id in record.cpu_ids for thread in threads)}
        _need({record.id for record in cores} == expected_cores,
              'SMT/core contention groups must match original candidate-thread membership')
        all_threads.update(thread.id for thread in threads)
        all_cores.update(record.id for record in cores)
    slots = {row.id: row for row in schedule.storage_slots}
    bound_slots = set()
    for row in projection.storages:
        occupancy = occupied[row.occupancy]
        slot = slots.get(row.slot)
        _need(slot is not None and slot.ownership == 'SCHEDULE' and row.slot == occupancy.storage_slot
              and row.slot not in bound_slots and row.payload_bytes == occupancy.units == slot.capacity_bytes,
              'storage binding must preserve one owned physical slot and its exact payload bytes')
        bound_slots.add(row.slot)
        memories = original_refs(row.candidate_memories, MemoryDomain)
        _need(all(memory.kind is MemoryKind.HOST for memory in memories), 'CPU storage candidates must be original HOST memory domains')
        _need(row.exact_memory is None, 'UNBOUND storage candidates do not prove physical memory residency')
    return dict(schema=SCHEMA, occupancy_binding_coverage_complete=True,
        execution_binding_count=len(projection.executions), storage_binding_count=len(projection.storages),
        exact_thread_binding_count=sum(row.exact_thread is not None for row in projection.executions),
        unknown_units=sum(row.units is None for row in projection.executions),
        unknown_thread_domains=sum(not row.candidate_threads for row in projection.executions),
        referenced_candidate_thread_count=len(all_threads), referenced_core_group_count=len(all_cores),
        unresolved_slotplacement=sum(row.exact_memory is None for row in projection.storages),
        source_activity_demand_coverage_complete=source['temporal']['execution_demand_coverage_complete'],
        original_budget_scope=projection.budget.scope, original_budget_bytes=projection.budget.bytes,
        physical_slot_payload_bytes=sum(row.payload_bytes for row in projection.storages),
        smt_core_groups_imply_exclusive_execution=False, thread_count_implies_throughput=False,
        reported_memory_capacity_is_available_budget=False, exact_memory_residency_proven=False,
        resource_capacity_feasibility_proven=False, cost_model_available=False, execution_authorized=False)


__all__ = ['SCHEMA', 'ExecutionBinding', 'StorageBinding', 'OriginalBudget', 'ResourceProjection',
           'projection_document', 'projection_from_dict', 'validate_resource_projection']
