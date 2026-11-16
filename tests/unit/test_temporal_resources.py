"""G2 uses retained synthetic topology; no costs, solver, or workload runs."""
from dataclasses import replace
import json

import pytest

from test_versioned_schedule import ordered_state_schedule
from scar.analysis.schedule_time import attach_task_time
from scar.analysis.target_graph import TargetHardwareGraph, build_target_graph
from scar.analysis.temporal_resources import derive_temporal_resources
from scar.ir.execution_recipe import artifact_reference
from scar.ir.program import target_hardware as H
from scar.ir.temporal_resources import (
    projection_document, projection_from_dict, validate_resource_projection,
)


@pytest.fixture
def inputs(tmp_path):
    inventory=dict(schema='scar.target-inventory',schema_version=1,sources=[],diagnostics=[],
        cpu=dict(online_cpu_ids=[0,1,2],model_names=['synthetic SMT fixture'],
            processors=[dict(cpu_id=i,package_id=0,core_id=0 if i<2 else 1) for i in range(3)],
            numa_nodes=[dict(node_id=0,cpu_ids=[0,1],memory_total_bytes=2**20),
                        dict(node_id=1,cpu_ids=[2],memory_total_bytes=2**20)]),
        dram=dict(memory_total_bytes=2**21),gpus=[],storage=dict(disks=[],block_devices=[]),paths=[],
        cpu_topology=dict(processors=[dict(cpu_id=i,online=True,package_id=0,core_id=0 if i<2 else 1,
            die_id=0,thread_siblings=[0,1] if i<2 else [2]) for i in range(3)],isa_groups=[],caches=[]))
    path=tmp_path/'inventory.json';path.write_text(json.dumps(inventory))
    target=tmp_path/'target';build_target_graph(target,inventory=path)
    resources=TargetHardwareGraph(target).resources()
    threads={row.cpu_id:row for row in resources if type(row) is H.HardwareThread}
    demands={'compute':{threads[0].id.wire:1},'reuse':{threads[1].id.wire:1}}
    # The original CPU_ALLOCATE contract, not this opaque location's spelling,
    # establishes the HOST candidate family.
    schedule=attach_task_time(ordered_state_schedule(),execution_demands=demands,
        resource_capacities={threads[i].id.wire:1 for i in (0,1)},
        source_refs=({'fixture':'DECLARED_STRUCTURAL_INPUT_NOT_RUNTIME_OBSERVATION'},))
    context=dict(target_reference=artifact_reference(target/'result.json','scar.target-hardware-graph'),
        caller_affinity_cpu_ids=[0,1],recipe_binding_cpu_ids=[0],
        source_refs=({'fixture':'EXPLICIT_RETAINED_TEST_CONTEXT'},))
    return schedule,target,resources,threads,context


def test_smt_threads_share_a_core_without_speedup_and_unknown_control_is_not_recipe_binding(inputs):
    schedule,target,resources,threads,context=inputs
    projection=derive_temporal_resources(schedule,target,source_context=context)
    summary=validate_resource_projection(projection,schedule,resources=resources)
    known=[row for row in projection.executions if row.exact_thread is not None]
    assert {row.exact_thread.resource for row in known}=={threads[0].id,threads[1].id}
    assert len(known)==2 and all(row.units==1 for row in known)
    assert known[0].core_refs==known[1].core_refs and len(known[0].core_refs)==1
    assert summary['smt_core_groups_imply_exclusive_execution'] is False
    assert summary['thread_count_implies_throughput'] is False
    unknown=[row for row in projection.executions if row.exact_thread is None]
    assert unknown and all(row.units is None for row in unknown)
    assert all({ref.resource for ref in row.candidate_threads}=={threads[0].id,threads[1].id}
               for row in unknown)
    # Candidate/ref membership alone cannot change CPU0's original binding.
    first=next(row for row in known if row.exact_thread.resource==threads[0].id)
    other=next(row for row in known if row.exact_thread.resource==threads[1].id)
    swapped=replace(first,exact_thread=other.exact_thread,candidate_threads=other.candidate_threads,
                    core_refs=other.core_refs)
    with pytest.raises(ValueError,match='original occupancy HardwareThread'):
        validate_resource_projection(replace(projection,executions=tuple(
            swapped if row.occupancy==first.occupancy else row for row in projection.executions)),
            schedule,resources=resources)
    forged=replace(unknown[0],exact_thread=first.exact_thread,candidate_threads=first.candidate_threads,
                   core_refs=first.core_refs)
    with pytest.raises(ValueError,match='unknown service'):
        validate_resource_projection(replace(projection,executions=tuple(
            forged if row.occupancy==forged.occupancy else row for row in projection.executions)),
            schedule,resources=resources)


def test_slots_count_once_host_candidates_are_not_residency_and_total_is_not_budget(inputs):
    schedule,target,resources,_,context=inputs
    projection=derive_temporal_resources(schedule,target,source_context=context)
    summary=validate_resource_projection(projection,schedule,resources=resources)
    assert projection_from_dict(json.loads(json.dumps(projection_document(projection))))==projection
    assert {row.slot for row in projection.storages}=={'shadow','temp'}
    assert len(projection.storages)==2 and sum(row.payload_bytes for row in projection.storages)==32
    assert len([row for row in schedule.materializations if row.region.slot=='temp'])==2
    assert all(row.exact_memory is None and row.candidate_memories for row in projection.storages)
    assert summary['exact_memory_residency_proven'] is False
    assert summary['reported_memory_capacity_is_available_budget'] is False
    assert summary['resource_capacity_feasibility_proven'] is False
    assert projection.budget.bytes==schedule.budget_bytes==48
    assert projection.budget.scope==schedule.budget_scope
    assert all(row.budget_bytes is None and row.capacity_bytes==2**20
               for row in resources if type(row) is H.MemoryDomain)
    with pytest.raises(ValueError,match='residency'):
        validate_resource_projection(replace(projection,storages=(
            replace(projection.storages[0],exact_memory=projection.storages[0].candidate_memories[0]),
            *projection.storages[1:])),schedule,resources=resources)
    with pytest.raises(ValueError,match='budget'):
        validate_resource_projection(replace(projection,budget=replace(projection.budget,bytes=2**20)),
                                     schedule,resources=resources)
    # A selected recipe CPU is not evidence about the original caller's
    # allocator/control thread; an absent caller domain remains absent.
    no_caller=derive_temporal_resources(schedule,target,
        source_context={**context,'caller_affinity_cpu_ids':None})
    assert all(row.candidate_threads==() and row.units is None
               for row in no_caller.executions if row.exact_thread is None)
