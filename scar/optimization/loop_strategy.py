"""Exact elementwise recurrence fusion, with one shared body and explicit count.

This minimizes incremental array payload only. It neither expands iterations
nor predicts runtime, cache behavior, DRAM traffic or physical optimality.
"""
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

from scar.compiler.loop_model import loop_digest, read_loop, validate_loop
from scar.compiler.model import document as plan_document
from scar.compiler.pipeline_model import pipeline_digest
from scar.ir.record_codec import loads
from .pipeline_memory import _shape, plan_pipeline_memory
from .pipeline_strategy import _fusion, _shape_replay
from .resource_solver import semantic_sha256


_CAPABILITY = 'EXACT_ELEMENTWISE_TEMPORAL_FUSION'


def _loop(value):
    return read_loop(value) if isinstance(value, (str, Path)) else validate_loop(value)


def _body_plan(loop, budget):
    # Raw certificate and optimized execution plan are distinct authorities.
    return _fusion(loop.body, budget)


def plan_loop_strategy(loop, *, input_shapes, trip_count, memory_budget_bytes,
                       search_budget=100000, backend_support=_CAPABILITY):
    """Plan a single carry's per-element temporal loop, never a partial fallback.

    input_shapes is keyed by the OUTER state parameter; a common rank-two shape
    is also accepted. trip_count is an observed/declared exact Python int, not a
    statically established loop count. Nonpositive counts retain original input
    alias semantics through NO_OP. The body must preserve the carry shape.
    """
    loop = _loop(loop)
    frontier = []
    supplied = dict(input_shapes) if isinstance(input_shapes, Mapping) else {loop.state_parameter: input_shapes}
    unsupported_shapes = [str(key) for key in supplied if key != loop.state_parameter]
    for key in supplied:
        if key != loop.state_parameter:
            frontier.append(dict(field='input_shapes.' + str(key), reason='only the outer array state has a shape; the trip count is scalar'))
    state_shape = _shape(supplied.get(loop.state_parameter))
    outer_shapes = {} if state_shape is None else {loop.state_parameter: state_shape}
    body_parameter = loop.body.parameters[0]
    body_shapes = {} if state_shape is None else {body_parameter: state_shape}
    if state_shape is None:
        frontier.append(dict(field='input_shapes.' + loop.state_parameter, reason='exact rank-two carry shape required'))
    exact_count = type(trip_count) is int
    modeled_count = trip_count if exact_count else None
    retained_count = trip_count if trip_count is None or type(trip_count) in (int, bool, str) else None
    if not exact_count:
        frontier.append(dict(field='trip_count', reason='EXACT_INT_TRIP_COUNT_REQUIRED'))
    elif trip_count <= 0:
        frontier.append(dict(field='trip_count', reason='NONPOSITIVE_BUILTIN_RANGE_RETURNS_ORIGINAL_INPUT_ALIAS'))
    elif trip_count > 2**63 - 1:
        frontier.append(dict(field='trip_count', reason='NATIVE_SIGNED_INT64_TRIP_COUNT_CONTRACT_UNSUPPORTED'))
    if backend_support != _CAPABILITY:
        frontier.append(dict(field='backend_support', reason='exact temporal-fusion backend capability unavailable'))
    memory = plan_pipeline_memory(loop.body, input_shapes=body_shapes,
                                  memory_budget_bytes=memory_budget_bytes, search_budget=search_budget)
    frontier.extend(dict(field='body.' + row['field'], reason=row['reason']) for row in memory['frontier'])
    final = next(value for value in memory['values'] if value['id'] == loop.body.output)
    output_shape, final_bytes = final['shape'], final['nbytes']
    if output_shape is None or state_shape is None or output_shape != state_shape:
        frontier.append(dict(field='body.output_shape', reason='BODY_MUST_PRESERVE_THE_EXACT_CARRY_SHAPE'))
    raw, certificate, plan = None, None, None
    shape_rows = []
    try:
        raw, certificate, plan = _body_plan(loop, memory_budget_bytes)
        if raw is not None:
            shape_rows, issues = _shape_replay(loop.body, raw, plan, certificate, memory)
            frontier.extend(dict(field='body.' + row['field'], reason=row['reason']) for row in issues)
    except (ValueError, TypeError, KeyError) as error:
        frontier.append(dict(field='body.execution_profile', reason=str(error)))
        if certificate is None:
            from .pipeline_fusion import fuse_pipeline_region
            raw, certificate = fuse_pipeline_region(loop.body)
    if certificate is None or not certificate.get('supported') or plan is None:
        frontier.append(dict(field='body.fusion', reason='CLOSED_ELEMENTWISE_BODY_FUSION_CONTRACT_REQUIRED'))
        if certificate is not None:
            frontier.extend(dict(field='body.' + row['field'], reason=row['reason'])
                            for row in certificate.get('frontiers', ()))
    if (memory_budget_bytes is None or final_bytes is None or final_bytes > memory_budget_bytes):
        frontier.append(dict(field='memory_budget_bytes', reason='KNOWN_BUDGET_MUST_FIT_ONE_FRESH_FINAL_PAYLOAD'))
    selected = not frontier
    elements = None if state_shape is None else state_shape[0] * state_shape[1]
    kind = 'TEMPORAL_FUSION' if selected else 'NO_OP'
    selected_memory = None
    if selected:
        selected_memory = dict(
            schema='scar.temporal-loop-memory-plan', schema_version=1, loop_sha256=loop_digest(loop),
            input_shapes=outer_shapes, body_input_shapes=body_shapes,
            state_shape=state_shape, output_shape=output_shape, trip_count=modeled_count,
            final_bytes=final_bytes, total_workspace_bytes=0, peak_allocation_bytes=final_bytes,
            memory_budget_bytes=memory_budget_bytes, budget_satisfied=True,
            status='OPTIMAL_WITHIN_TEMPORAL_PAYLOAD_MODEL', frontier=[], memory_policy='UNBOUND',
            execution_representation=_CAPABILITY, allocation_policy='ONE_FRESH_CALLER_OWNED_FINAL_NO_INTERMEDIATE_ARRAYS',
            budget_scope=memory['budget_scope'], budget_excludes=memory['budget_excludes'],
            physical_optimality_proven=False, duration_predicted=False)
    positive_count = exact_count and 0 < trip_count <= 2**63 - 1
    return dict(
        schema='scar.loop-strategy-plan', schema_version=1, loop_id=loop.id, loop_sha256=loop_digest(loop),
        body_pipeline_sha256=pipeline_digest(loop.body), capability=_CAPABILITY, backend_support=backend_support,
        parameters=list(loop.parameters), state_parameter=loop.state_parameter,
        trip_count_parameter=loop.trip_count_parameter, trip_count=retained_count,
        trip_count_contract=dict(authority='CALL_OBSERVATION_OR_DECLARATION_NOT_STATIC_PROOF',
                                 exact_python_type='int' if exact_count else 'NOT_EXACT_INT',
                                 guard='EXACT_PYTHON_INT_EQUAL_TO_MODELED_POSITIVE_INT64_COUNT'),
        array_parameter_map={loop.state_parameter: body_parameter}, input_shapes=outer_shapes,
        input_shape_frontier_parameters=unsupported_shapes,
        body_input_shapes=body_shapes, state_shape=state_shape, output_shape=output_shape,
        final_bytes=final_bytes, memory_budget_bytes=memory_budget_bytes, search_budget=search_budget,
        selected_strategy=kind, selected=dict(kind=kind), selected_memory_plan=selected_memory,
        status='SELECTED_UNDER_MODEL_CONTRACT' if selected else 'NO_NATIVE_LOOP_STRATEGY_ADMITTED',
        frontier=frontier, body_memory_plan=memory, body_fusion_certificate=certificate,
        body_raw_region_sha256=None if certificate is None else certificate.get('region_sha256'),
        loop_execution_plan_sha256=None if plan is None else semantic_sha256(plan_document(plan)),
        body_shape_replay=shape_rows, shape_invariant=state_shape is not None and output_shape == state_shape,
        control=dict(kind='REAL_RUNTIME_LOOP', trip_count_parameter=loop.trip_count_parameter,
                     modeled_trip_count=modeled_count, iteration_order='range(trip_count)', unrolled=False,
                     array_carry='INITIAL_LOAD_LOCAL_SCALAR_RECURRENCE_FINAL_STORE',
                     logical_value_scope='LOOP_ITERATION_INSTANCE_SHARED_TEMPLATE'),
        representation_size=dict(body_expressions=None if plan is None else len(plan.region.nodes),
                                 temporal_loop_nodes=1, iteration_expression_copies=0),
        logical_array_traffic=None if not selected else dict(initial_carry_loads=elements, final_stores=elements,
            initial_carry_load_bytes=8 * elements, final_store_bytes=8 * elements,
            scope='ALGORITHMIC_ARRAY_PAYLOAD_ONLY_NOT_DRAM_OR_CACHE_TRAFFIC'),
        original_materializations_if_all_iterations_complete=(trip_count * len(loop.body.calls) if positive_count else None),
        native_materializations=1 if selected else None,
        materialization_count_scope=dict(
            original='ORIGINAL_LEAF_RESULT_ARRAYS_ONLY_EXCLUDES_LEAF_INTERNAL_NUMPY_TEMPORARIES',
            native='FINAL_ARRAY_PAYLOAD_ONLY_EXCLUDES_THREAD_STACK_AND_ALLOCATOR_ALLOCATIONS'),
        payload_lower_bound_bytes=final_bytes if positive_count else 0 if exact_count else None,
        payload_gap_bytes=0 if selected else None,
        optimality='TEMPORAL_PAYLOAD_MODEL_OPTIMUM_ESTABLISHED' if selected else 'NOT_ESTABLISHED',
        optimality_scope='EXACT_ELEMENTWISE_RECURRENCE_FRESH_FINAL_INCREMENTAL_ARRAY_PAYLOAD_ONLY',
        execution_plan_role='SHARED_BODY_COMPONENT_NOT_A_STANDALONE_LOOP_KERNEL',
        guard_source_refs=['#/source', '#/body_callee', '#/body/source', '#/body/calls'],
        runtime_guards=['OUTER_FUNCTION_BODY_AND_ALL_LEAF_CALLEE_IDENTITIES_MATCH',
                        'BUILTIN_RANGE_IDENTITY_MATCHES', 'EXACT_POSITIVE_INT64_TRIP_COUNT_MATCHES',
                        'EXACT_FLOAT64_CPU_RANK2_STATE_SHAPE_AND_LAYOUT',
                        'INITIAL_AND_EVERY_ITERATION_CARRY_FINITE', 'EVERY_ITERATION_FLOATING_STATUS_GUARDED',
                        'FRESH_FINAL_AND_PAYLOAD_BUDGET', 'NO_PARTIAL_RESULT_EXPORTED'],
        failure_policy='WITHHOLD_OUTPUT_AND_CALL_ORIGINAL_LOOP_ONCE',
        external_assumptions=['SYNCHRONOUS_CPU_EXECUTION', 'NO_FOREIGN_CONCURRENT_INPUT_WRITERS',
                              'FP_STATUS_FLAGS_NOT_USED_AS_PROGRAM_OUTPUT'],
        floating_environment_policy='SAVE_RESTORE_CALLER_WORKER_ENV_NOT_FULL_NUMPY_STATUS_FLAG_EQUIVALENCE',
        original_python_no_op=dict(available=True, nonpositive_count_retains_original_alias=True),
        physical_optimality_proven=False, duration_predicted=False, dram_traffic_predicted=False,
        placement_optimized=False, target_variants_executed=False, deployment_authorized=False)


def verify_loop_strategy(loop, strategy):
    """Pure source/body/shape/control/budget replay, with no target execution."""
    value = loads(Path(strategy).read_bytes()) if isinstance(strategy, (str, Path)) else deepcopy(strategy)
    if (value.get('schema'), value.get('schema_version')) != ('scar.loop-strategy-plan', 1):
        raise ValueError('versioned loop strategy required')
    shapes = dict(value['input_shapes'])
    shapes.update({key: None for key in value['input_shape_frontier_parameters']})
    expected = plan_loop_strategy(loop, input_shapes=shapes, trip_count=value['trip_count'],
        memory_budget_bytes=value['memory_budget_bytes'], search_budget=value['search_budget'],
        backend_support=value['backend_support'])
    if semantic_sha256(expected) != semantic_sha256(value):
        raise ValueError('loop strategy differs from complete source/body/control replay')
    return value


def loop_execution_plan(loop, strategy):
    """Return exact shared body P; a temporal backend must supply the real loop."""
    loop = _loop(loop); report = verify_loop_strategy(loop, strategy)
    if report['selected_strategy'] != 'TEMPORAL_FUSION':
        raise ValueError('exact temporal fusion is not the selected strategy')
    _, _, plan = _body_plan(loop, report['memory_budget_bytes'])
    if plan is None or semantic_sha256(plan_document(plan)) != report['loop_execution_plan_sha256']:
        raise ValueError('shared loop body execution-plan receipt differs')
    return plan


__all__ = ['plan_loop_strategy', 'verify_loop_strategy', 'loop_execution_plan']
