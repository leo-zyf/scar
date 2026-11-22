"""Payload strategy can admit fusion when the original fixed pool does not fit."""
from copy import deepcopy

import pytest

from test_pipeline_memory import call, chain
from scar.compiler.model import ArrayRegion, Expr, SourceRef, document
from scar.compiler.pipeline_frontend import compose_pipeline
from scar.optimization import optimize_region
from scar.optimization.pipeline_strategy import fused_execution_plan, plan_pipeline_strategy, verify_pipeline_strategy
from scar.optimization.resource_solver import semantic_sha256


def test_fusion_reaches_final_payload_bound_even_when_original_pool_is_infeasible():
    pipeline = chain()
    result = plan_pipeline_strategy(pipeline, input_shapes=(2, 3), memory_budget_bytes=48)
    assert result['preserve_memory_plan']['status'] == 'INFEASIBLE_WITHIN_FIXED_POOL_MODEL'
    assert result['selected_strategy'] == result['selected']['kind'] == 'FULL_FUSION'
    memory = result['selected_memory_plan']
    assert memory['total_workspace_bytes'] == 0 and memory['peak_allocation_bytes'] == 48
    assert result['payload_lower_bound_bytes'] == 48 and result['payload_gap_bytes'] == 0
    assert memory['budget_satisfied'] and not result['physical_optimality_proven']
    plan = fused_execution_plan(pipeline, result)
    assert semantic_sha256(document(plan)) == result['fusion_execution_plan_sha256']
    assert plan.proof['source_operation_map']
    assert 'fusion_execution_plan' not in result and 'fusion_region' not in result
    assert verify_pipeline_strategy(pipeline, result) == result
    tampered = deepcopy(result); tampered['selected_memory_plan']['peak_allocation_bytes'] += 8
    with pytest.raises(ValueError, match='representation/source/shape replay'):
        verify_pipeline_strategy(pipeline, tampered)


def test_slice_frontier_preserves_original_calls_or_admits_no_native_strategy():
    source = SourceRef('/tmp/strategy-slice.py', '0' * 64, 'sliced', 1, 0, 'sliced-key')
    region = ArrayRegion('sliced', source, ('a',), (
        Expr('a', 'input', attrs={'name': 'a'}),
        Expr('view', 'slice', ('a',), {'bounds': [[None, None], [None, None]]}),
        Expr('negative', 'neg', ('view',)), Expr('result', 'neg', ('negative',))), 'result')
    leaf = optimize_region(region)
    pipeline = compose_pipeline(source, ('x',),
                                (call('a', leaf, ('arg:0',), 'a'), call('b', leaf, ('a',), 'b')), 'b')
    preserved = plan_pipeline_strategy(pipeline, input_shapes=(2, 3), memory_budget_bytes=96)
    assert preserved['selected_strategy'] == 'PRESERVE_CALLS_HOST_POOL'
    assert preserved['fusion_frontier'] and not preserved['fusion_certificate']['supported']
    rejected = plan_pipeline_strategy(pipeline, input_shapes=(2, 3), memory_budget_bytes=48)
    assert rejected['selected_strategy'] == 'NO_OP' and rejected['selected_memory_plan'] is None
    assert rejected['original_python_no_op']['available']


def test_unused_external_parameter_keeps_its_full_shape_obligation_in_fused_form():
    original = chain()
    pipeline = compose_pipeline(original.source, ('x', 'unused'), original.calls, original.output,
                                input_values=('arg:0', 'unused-value'))
    report = plan_pipeline_strategy(pipeline, input_shapes={'x': [2, 3], 'unused': [7, 1]}, memory_budget_bytes=48)
    assert report['selected_strategy'] == 'FULL_FUSION'
    assert fused_execution_plan(pipeline, report).region.parameters == ('x', 'unused')
    unused = next(row for row in report['fusion_shape_replay'] if row['value_id'] == 'unused-value')
    assert unused['original_shape'] == unused['merged_shape'] == [7, 1]
    missing = plan_pipeline_strategy(pipeline, input_shapes={'x': [2, 3]}, memory_budget_bytes=48)
    assert missing['selected_strategy'] == 'NO_OP'
    assert any('unused' in row['field'] for row in missing['frontier'])
