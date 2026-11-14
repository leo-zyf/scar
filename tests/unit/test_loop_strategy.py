"""Temporal strategy keeps one body, real count guards and original aliases."""
from copy import deepcopy

import pytest

from test_pipeline_memory import call, chain
from scar.compiler.loop_model import ArrayLoopPlan
from scar.compiler.model import ArrayRegion, Expr, SourceRef, document
from scar.compiler.pipeline_model import CalleeGuard, compose_pipeline
from scar.optimization import optimize_region
from scar.optimization.loop_strategy import plan_loop_strategy, verify_loop_strategy, loop_execution_plan
from scar.optimization.resource_solver import semantic_sha256


def make_loop(body=None):
    body = chain() if body is None else body
    source = SourceRef('/tmp/loop-strategy.py', '2' * 64, 'evolve', 1, 0, 'evolve-key')
    return ArrayLoopPlan('loop:test', source, ('steps', 'state'), 'state', 'steps', body,
                         CalleeGuard(body.source, ('advance',)))


def test_temporal_fusion_has_constant_body_size_and_only_fresh_final_payload():
    loop = make_loop()
    report = plan_loop_strategy(loop, input_shapes={'state': [2, 3]}, trip_count=400, memory_budget_bytes=48)
    assert report['selected_strategy'] == report['selected']['kind'] == 'TEMPORAL_FUSION'
    assert report['body_memory_plan']['status'] == 'INFEASIBLE_WITHIN_FIXED_POOL_MODEL'
    assert report['array_parameter_map'] == {'state': 'x'}
    assert report['input_shapes'] == {'state': [2, 3]} and report['body_input_shapes'] == {'x': [2, 3]}
    memory = report['selected_memory_plan']
    assert memory['total_workspace_bytes'] == 0 and memory['peak_allocation_bytes'] == memory['final_bytes'] == 48
    assert report['payload_lower_bound_bytes'] == 48 and report['payload_gap_bytes'] == 0
    assert report['logical_array_traffic']['initial_carry_loads'] == report['logical_array_traffic']['final_stores'] == 6
    assert not report['control']['unrolled'] and report['representation_size']['iteration_expression_copies'] == 0
    assert 'FP_STATUS_FLAGS_NOT_USED_AS_PROGRAM_OUTPUT' in report['external_assumptions']
    assert report['original_materializations_if_all_iterations_complete'] == 400 * len(loop.body.calls)
    assert 'EXCLUDES_LEAF_INTERNAL_NUMPY_TEMPORARIES' in report['materialization_count_scope']['original']
    plan = loop_execution_plan(loop, report)
    assert plan.region.parameters == ('x',) and semantic_sha256(document(plan)) == report['loop_execution_plan_sha256']
    large = plan_loop_strategy(loop, input_shapes=(2, 3), trip_count=10**9, memory_budget_bytes=48)
    assert large['representation_size'] == report['representation_size']
    assert large['loop_execution_plan_sha256'] == report['loop_execution_plan_sha256']
    assert verify_loop_strategy(loop, report) == report
    tampered = deepcopy(report); tampered['trip_count'] = 401
    with pytest.raises(ValueError, match='source/body/control replay'):
        verify_loop_strategy(loop, tampered)


def test_alias_counts_shape_changes_budget_and_missing_backend_remain_noop():
    loop = make_loop()
    for count in (0, -3, True):
        report = plan_loop_strategy(loop, input_shapes=(2, 3), trip_count=count, memory_budget_bytes=48)
        assert report['selected_strategy'] == 'NO_OP' and report['selected_memory_plan'] is None
        assert report['original_python_no_op']['nonpositive_count_retains_original_alias']
        verify_loop_strategy(loop, report)
    low = plan_loop_strategy(loop, input_shapes=(2, 3), trip_count=2, memory_budget_bytes=47)
    assert low['selected_strategy'] == 'NO_OP' and any(row['field'] == 'memory_budget_bytes' for row in low['frontier'])
    unavailable = plan_loop_strategy(loop, input_shapes=(2, 3), trip_count=2, memory_budget_bytes=48, backend_support=False)
    assert unavailable['selected_strategy'] == 'NO_OP' and any(row['field'] == 'backend_support' for row in unavailable['frontier'])
    source = SourceRef('/tmp/loop-crop.py', '3' * 64, 'crop', 1, 0, 'crop-key')
    leaf = optimize_region(ArrayRegion('crop', source, ('a',), (
        Expr('a', 'input', attrs={'name': 'a'}),
        Expr('view', 'slice', ('a',), {'bounds': [[1, None], [None, None]]}),
        Expr('negative', 'neg', ('view',)), Expr('result', 'neg', ('negative',))), 'result'))
    body_source = SourceRef('/tmp/loop-body.py', '4' * 64, 'crop_body', 1, 0, 'crop-body-key')
    body = compose_pipeline(body_source, ('x',), (call('crop', leaf, ('arg:0',), 'cropped'),), 'cropped')
    changing = plan_loop_strategy(make_loop(body), input_shapes=(4, 4), trip_count=2, memory_budget_bytes=128)
    assert changing['selected_strategy'] == 'NO_OP' and not changing['shape_invariant']
    assert any(row['field'] == 'body.output_shape' for row in changing['frontier'])
