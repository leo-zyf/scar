"""Project source-admitted host occupancies onto retained target descriptions.

This reads retained target evidence, never probes hardware or runs a program.
Source admission of the host graph belongs to its independent publisher.
"""
from copy import deepcopy
from pathlib import Path

from scar.analysis.target_graph import TargetHardwareGraph
from scar.ir.execution_recipe import read_artifact
from scar.ir.program import target_hardware as H
from scar.ir.program.providers import ProviderArtifact
from scar.ir.record_codec import encode
from scar.ir.temporal_resources import (ExecutionBinding, StorageBinding, OriginalBudget,
    ResourceProjection, validate_resource_projection)
from scar.ir.versioned_schedule import (TemporalOccupancyKind as K, VersionedTaskKind as T,
    schedule_document, validate_versioned_schedule)
from scar.optimization.hardware import _target_materials, _accesses
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.pipeline-host-resource-source'


def _need(condition, message):
    if not condition: raise ValueError(message)


def derive_pipeline_host_resources(schedule, target, *, source_refs=()):
    """Keep every occupancy; bind only source-qualified original ABI threads.

    ``target`` is its retained directory, result.json, TargetHardwareGraph, or
    complete snapshot ArtifactRef. Candidate HOST domains prove neither page
    placement nor available memory; logical tokens prove no CPU reservation.
    """
    validate_versioned_schedule(schedule)
    proof = schedule.proof.get('pipeline_host_envelope', {})
    _need(schedule.temporal is not None and proof.get('schema') == 'scar.pipeline-host-envelope'
          and proof.get('schema_version') == 1, 'source-admitted pipeline host envelope required')
    _need(type(source_refs) in (tuple, list) and all(type(r) is dict for r in source_refs),
          'original source reference dictionaries required')
    expected = encode(target) if type(target) in (dict, ProviderArtifact) else None
    if expected is not None:
        read_artifact(expected); path = Path(expected['uri']).parent
    else:
        path = target.path if type(target) is TargetHardwareGraph else Path(target)
        if path.is_file(): path = path.parent
    graph, snapshot, inventory, connections = _target_materials(path)
    target_ref = encode(snapshot)
    _need(expected is None or expected == target_ref, 'retained target snapshot identity differs')
    by_wire = {r.id.wire:r for r in inventory}
    threads = {r.cpu_id:r for r in inventory if type(r) is H.HardwareThread}
    contexts = [r for task in schedule.tasks for r in task.resource_refs
                if r.get('kind') == 'EXPLICIT_HARDWARE_THREAD_SET']
    _need(bool(contexts), 'original explicit ABI thread/target evidence required')
    for context in contexts:
        cpus, refs = context['cpu_ids'], context['thread_refs']
        _need(context['target'] == target_ref and len(cpus) == len(refs)
              and len(set(cpus)) == len(cpus), 'original CPU set or target evidence differs')
        for cpu, ref in zip(cpus, refs):
            _need(cpu in threads and ref == encode(H.TargetResourceRef(snapshot, threads[cpu].id)),
                  'source CPU id and original HardwareThread reference differ')
    reference = lambda row:H.TargetResourceRef(snapshot, row.id)
    executors = [r for r in inventory if type(r) is H.CpuExecutor]
    hosts = [r for r in inventory if type(r) is H.MemoryDomain and r.kind is H.MemoryKind.HOST]
    access = [(i,e,m) for i,c in enumerate(connections) for e in executors for m in hosts if _accesses(e,m,c)]
    memories = tuple(m for m in hosts if any(m.id == row[2].id for row in access))
    relations = [dict(connection_ordinal=i, executor=e.id.wire, memory=m.id.wire,
        conditions=list(connections[i].conditions), locality=connections[i].locality,
        evidence=encode(connections[i].evidence)) for i,e,m in access]
    digest = semantic_sha256(schedule_document(schedule))
    activities = {a.id:a for a in schedule.temporal.activities}
    tasks = {t.id:t for t in schedule.tasks}; slots = {s.id:s for s in schedule.storage_slots}
    executions, storages, unresolved = [], [], []
    for ordinal, occupancy in enumerate(schedule.temporal.occupancies):
        origin = dict(source_schedule_sha256=digest, locator='#/temporal/occupancies/'+str(ordinal))
        if occupancy.kind is K.EXECUTION_DEMAND:
            record = by_wire.get(occupancy.resource)
            exact = record if type(record) is H.HardwareThread else None
            logical = any(r.get('logical_worker_not_physical_cpu') is True or
                r.get('logical_mutex_not_cpu_service') is True for r in occupancy.source_refs)
            if exact is not None:
                activity = activities[occupancy.activity]
                _need(not logical and tasks[activity.task].kind is T.COMPUTE
                    and any(r.get('phase_role') == 'ABI_SERVICE' for r in activity.source_refs),
                    'exact hardware-thread demand must belong to the original ABI service')
                refs = [r for r in occupancy.source_refs if r.get('kind') == 'EXPLICIT_HARDWARE_THREAD_SET']
                _need(refs and all(r in contexts for r in refs)
                    and all(exact.cpu_id in r['cpu_ids'] for r in refs), 'ABI occupancy thread lacks original evidence')
            candidates = () if exact is None else (reference(exact),)
            cores = tuple(reference(c) for c in inventory if type(c) is H.CpuCore
                          and exact is not None and exact.cpu_id in c.cpu_ids)
            role = 'LOGICAL_TOKEN_NOT_PHYSICAL_CPU' if logical else 'SOURCE_SELECTED_ABI_THREAD' if exact else 'UNKNOWN_HOST_CPU_DOMAIN'
            executions.append(ExecutionBinding(occupancy.id, occupancy.resource,
                None if exact is None else reference(exact), cores, candidates, occupancy.units,
                (dict(**origin, kind=role, logical_units_are_cpu_demands=False if logical else None,
                    actual_cpu_observation=False, exclusive_core_reservation=False),)))
            if exact is None:
                unresolved.append(dict(occupancy=occupancy.id, reason=role,
                    physical_cpu_feasibility_established=False))
            elif exact.online is False:
                unresolved.append(dict(occupancy=occupancy.id, reason='SOURCE_SELECTED_THREAD_OFFLINE_IN_SNAPSHOT'))
            if occupancy.units is None:
                unresolved.append(dict(occupancy=occupancy.id, reason='ORIGINAL_EXECUTION_UNITS_UNKNOWN'))
        elif occupancy.kind is K.PHYSICAL_SLOT_BYTES:
            slot = slots[occupancy.storage_slot]
            supported = slot.device == 'cpu' and tasks[slot.allocated_by].capability == 'CPU_ALLOCATE'
            storages.append(StorageBinding(occupancy.id, slot.id,
                tuple(reference(m) for m in memories) if supported else (), None, slot.capacity_bytes,
                (dict(**origin, kind='HOST_PROCESS_ADDRESS_SPACE_UNBOUND',
                    access_relations_locator='#/source_refs/'+str(len(source_refs))+'/memory_access_relations',
                    original_slot_location=slot.location, page_binding_authorized=False),)))
            unresolved.append(dict(occupancy=occupancy.id, reason='PHYSICAL_PAGE_AND_NUMA_RESIDENCY_UNPROVEN'))
        else: raise ValueError('unsupported original temporal occupancy kind')
    context = dict(schema=SCHEMA, schema_version=1, target=target_ref,
        memory_access_relations=relations, candidate_access_requires_original_policy_conditions=True,
        source_admission_required_upstream=True, hardware_probed=False,
        logical_tokens_are_physical_cpu_capacity=False, global_reservation=False,
        reported_capacity_is_available_budget=False, budget_reinterpreted_as_rss=False)
    result = ResourceProjection(snapshot, digest, tuple(executions), tuple(storages),
        OriginalBudget(schedule.budget_bytes, schedule.budget_scope, deepcopy(schedule.budget_refs)),
        tuple(unresolved), (*tuple(deepcopy(source_refs)), context))
    validate_resource_projection(result, schedule, resources=inventory)
    _need({r.slot for r in storages} == {s.id for s in schedule.storage_slots if s.ownership == 'SCHEDULE'},
          'every owned physical slot must be represented exactly once')
    read_artifact(target_ref)
    return result


__all__ = ['SCHEMA', 'derive_pipeline_host_resources']
