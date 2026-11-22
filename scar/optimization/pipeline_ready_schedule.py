"""Dependency-ready CPU dispatch over the original fixed-pool call graph.

Each call is statically bound to one original HardwareThread. A lane chooses
its lowest-index ready call, independently of other lanes; only the final
success fence joins all work. This is a legal unranked trial implementation,
not a timing model or a claim that round-robin placement is fastest.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from scar.compiler.pipeline_frontend import pipeline_digest, validate_pipeline
from scar.ir.execution_recipe import read_artifact
from scar.ir.record_codec import loads
from scar.ir.versioned_schedule import schedule_document, validate_versioned_schedule
from scar.optimization.pipeline_memory import verify_pipeline_memory
from .pipeline_schedule import _closed, _dependencies, _need, _target, _task_graph
from .resource_solver import semantic_sha256


SCHEMA = 'scar.pipeline-ready-resource-plan'
FAMILY = 'DEPENDENCY_READY_CPU_SERIAL_LEAVES_V1'
DISPATCH_POLICY = 'LOWEST_ORIGINAL_CALL_INDEX_READY_ON_FIXED_CPU_LANE'


def _call_bindings(pipeline, cpu_ids):
    """Deterministic placement, independent of durations and observations."""
    return {call.id: cpu_ids[index % len(cpu_ids)]
            for index, call in enumerate(pipeline.calls)}


def _ready_calls(call_order, predecessors, call_cpu_ids, completed, cpu_ids, *, busy_cpu_ids=()):
    """One dispatch decision per idle lane; unfinished queue heads do not block.

    This is the declarative policy mirrored by the host. It supplies no time,
    speculative completion, storage release, or permanent lane-order edge.
    """
    completed = set(completed)
    busy = set(busy_cpu_ids)
    result = {}
    for call in call_order:
        cpu = call_cpu_ids[call]
        if (call not in completed and cpu not in busy and cpu not in result
                and set(predecessors[call]) <= completed):
            result[cpu] = call
    _need(set(result) <= set(cpu_ids), 'dispatch references a CPU outside the explicit lanes')
    return result


def _dependency_graph(pipeline, memory, predecessors, call_cpu_ids, resource_refs):
    # The old producer already emits a wave-independent RAW/WAR/WAW graph.
    # Reuse its exact value/storage construction; specialize only the dispatch
    # proof. No source-order or implementation-wave edge is added or removed.
    graph = _task_graph(pipeline, memory, predecessors, (), call_cpu_ids, resource_refs)
    proof = dict(graph.proof)
    proof.pop('execution_waves_are_separate_dependency_preserving_refinement')
    proof.update(profile=FAMILY,
        dependency_ready_dispatch_is_separate_resource_capacity_refinement=True,
        per_cpu_mutual_exclusion_is_resource_capacity_not_semantic_order=True,
        intermediate_global_wave_barriers=False,
        dispatch_policy=DISPATCH_POLICY)
    graph = replace(graph, proof=proof)
    validate_versioned_schedule(graph)
    return graph


def plan_pipeline_ready_schedule(pipeline, memory_plan, target, *, cpu_ids, input_graph=None):
    """Retain true data/storage predecessors and bind unranked CPU lanes."""
    pipeline = validate_pipeline(pipeline)
    _closed(pipeline)
    memory = verify_pipeline_memory(pipeline, memory_plan)
    complete_profiles={('scar.pipeline-memory-plan',1,'OPTIMAL_WITHIN_FIXED_POOL_MODEL'),
        ('scar.pipeline-partial-order-memory-plan',1,'OPTIMAL_WITHIN_RAW_PARTIAL_ORDER_POOL_MODEL')}
    _need((memory.get('schema'),memory.get('schema_version'),memory['status'])in complete_profiles
          and memory['search_budget_exhausted'] is False
          and not memory['frontier'] and memory['budget_satisfied'] and memory['assignment'] is not None,
          'complete original fixed-pool lifetime/shape/budget plan required')
    _, snapshot, cpus, affinity, bindings = _target(target, cpu_ids)
    if input_graph is not None:
        read_artifact(input_graph, json_document=False)
    predecessors, reasons = _dependencies(pipeline, memory)
    calls = _call_bindings(pipeline, cpus)
    resources = (dict(kind='EXPLICIT_HARDWARE_THREAD_SET', target=snapshot, cpu_ids=list(cpus),
        thread_refs=[row['thread']['target'] for row in bindings],
        original_fact_details_plan_pointer='#/hardware_bindings',
        global_exclusive_reservation_proven=False),)
    tasks = _dependency_graph(pipeline, memory, predecessors, calls, resources)
    thread_ids = {row['cpu_id']: row['thread']['record']['id']['wire'] for row in bindings}
    return dict(schema=SCHEMA, schema_version=1,
        pipeline_sha256=pipeline_digest(pipeline), memory_plan_sha256=semantic_sha256(memory),
        target=snapshot, cpu_ids=list(cpus), call_predecessors=predecessors, call_cpu_ids=calls,
        call_order=[call.id for call in pipeline.calls], task_graph=schedule_document(tasks),
        requires_serial_leaf_operators=True, execution_family=FAMILY,
        input_graph=input_graph, input_graph_role='PROVENANCE_ONLY_NOT_DEPENDENCY_AUTHORITY',
        dependency_authority='ORIGINAL_CALL_INPUT_VALUE_EDGES_AND_REPLAYED_POOL_STORAGE_REUSE',
        dependency_reasons=reasons,
        placement_policy='ORIGINAL_CALL_INDEX_MODULO_EXPLICIT_ORDERED_CPU_IDS_UNRANKED',
        dispatch_policy=DISPATCH_POLICY,
        dispatch_contract=dict(workers_per_cpu_lane=1,
            prerequisites='ALL_TRUE_CALL_PREDECESSORS_COMPLETED_AND_ASSIGNED_CPU_LANE_IDLE',
            pending_selection='SCAN_ALL_PENDING_ASSIGNED_CALLS_IN_ORIGINAL_INDEX_ORDER',
            unfinished_queue_head_blocks_other_ready_calls=False,
            completion_visibility='RELEASE_ACQUIRE_OR_MUTEX_SYNCHRONIZED_CALL_COMPLETION',
            wake_on='ANY_CALL_COMPLETION_OR_FAILURE',
            intermediate_wave_barriers=False,
            added_lane_order_edges=False,
            final_join_required_before_export_or_fallback=True),
        readiness_validation=dict(original_call_coverage=True,
            data_and_all_old_pool_consumers_preserved=True,
            pool_overwrite_happens_before_verified=True,
            dag_basis='EVERY_TRUE_PREDECESSOR_HAS_SMALLER_ORIGINAL_CALL_INDEX',
            progress_requires='EACH_LANE_EVENTUALLY_SCANS_ALL_PENDING_CALLS_AFTER_COMPLETION_WAKE',
            same_cpu_nonoverlap='ONE_SERIAL_WORKER_PER_EXPLICIT_HARDWARE_THREAD',
            implementation_dispatch_does_not_change_semantic_dependency_graph=True),
        caller_affinity_cpu_ids=affinity, hardware_bindings=bindings,
        cpu_budget=dict(units='EXPLICIT_HARDWARE_THREADS', capacity=len(cpus), global_reservation=False),
        resource_capacities={thread_ids[cpu]: 1 for cpu in cpus},
        call_resource_demands={call: {thread_ids[cpu]: 1} for call, cpu in calls.items()},
        original_incremental_memory_budget_bytes=memory['memory_budget_bytes'],
        original_pool_plus_fresh_final_bytes=memory['peak_allocation_bytes'],
        encoded_graph_budget_bytes=tasks.budget_bytes,
        external_input_aliasing='MAY_ALIAS_READ_ONLY_NO_DISTINCT_ALLOCATION_INFERRED',
        external_input_layout='C_CONTIGUOUS_REQUIRED', external_backing_allocation_extent='UNKNOWN',
        memory_policy='UNBOUND', numa_page_residency_proven=False,
        durations={task.id: dict(status='UNKNOWN', duration_ticks=None) for task in tasks.tasks},
        normal_completion_only=True, original_internal_exception_or_allocator_sequence_preserved=False,
        source_order_independent='ONLY_CLOSED_READONLY_ARRAY_PROFILE_UNDER_ORIGINAL_GUARDS',
        failure_policy='STOP_DISPATCH_JOIN_ALL_RUNNING_DISCARD_PRIVATE_RESULTS_ORIGINAL_ONCE',
        executable_under_required_host_contract=True, model_optimum_proven=False,
        physical_optimality_proven=False, duration_predicted=False, deployment_authorized=False,
        target_program_executed=False, original_pure_call_graph_unrolled=False)


def verify_pipeline_ready_schedule(pipeline, memory_plan, target, report):
    """Rebuild original data/storage hazards, pool versions, lanes and contract."""
    value = loads(Path(report).read_bytes()) if isinstance(report, (str, Path)) else deepcopy(report)
    _need(value.get('schema') == SCHEMA and type(value.get('schema_version')) is int
          and value['schema_version'] == 1, 'versioned dependency-ready resource plan required')
    expected = plan_pipeline_ready_schedule(pipeline, memory_plan, target,
        cpu_ids=value['cpu_ids'], input_graph=value['input_graph'])
    _need(semantic_sha256(expected) == semantic_sha256(value),
          'ready resource plan differs from original calls/pool/topology/dispatch replay')
    return value


__all__ = ['SCHEMA', 'FAMILY', 'DISPATCH_POLICY',
           'plan_pipeline_ready_schedule', 'verify_pipeline_ready_schedule']
