"""Logical array versions can reuse storage only after strict last use."""
from copy import deepcopy

import pytest

from scar.compiler.model import ArrayRegion, Expr, SourceRef
from scar.compiler.pipeline_model import CalleeGuard, RegionCall, compose_pipeline
from scar.optimization import optimize_region
from scar.optimization.pipeline_memory import plan_pipeline_memory, verify_pipeline_memory


def leaf(arity=1):
    parameters = ('a',) if arity == 1 else ('a', 'b')
    source = SourceRef('/tmp/pipeline-memory-leaf.py', '0' * 64, 'leaf' + str(arity), 1, 0, 'leaf:' + str(arity))
    nodes = tuple(Expr(name, 'input', attrs={'name': name}) for name in parameters)
    operation = Expr('intermediate', 'neg' if arity == 1 else 'add', parameters)
    result = Expr('result', 'neg', ('intermediate',))
    return optimize_region(ArrayRegion('leaf:' + str(arity), source, parameters, nodes + (operation, result), 'result'))


def call(identifier, plan, inputs, output):
    return RegionCall(identifier, plan, tuple(inputs), output,
                      CalleeGuard(plan.region.source, (plan.region.source.qualname,)),
                      ({'kind': 'test_original_call', 'id': identifier},))


def chain():
    plan = leaf(); previous = 'arg:0'; calls = []
    for index in range(5):
        output = 'v' + str(index)
        calls.append(call('call:' + str(index), plan, (previous,), output))
        previous = output
    source = SourceRef('/tmp/pipeline-memory.py', '1' * 64, 'chain', 1, 0, 'chain-key')
    return compose_pipeline(source, ('x',), calls, previous)


def test_chain_reuses_slots_without_merging_values_or_overwriting_call_inputs():
    pipeline = chain()
    report = plan_pipeline_memory(pipeline, input_shapes=(2, 3), memory_budget_bytes=144)
    assert report['status'] == 'OPTIMAL_WITHIN_FIXED_POOL_MODEL' and report['budget_satisfied']
    assert report['input_shapes'] == {'x': [2, 3]}
    assert report['total_workspace_bytes'] == 96 and report['final_bytes'] == 48
    assert report['peak_allocation_bytes'] == report['lower_bound_peak_allocation_bytes'] == 144
    assignment = report['assignment']
    assert assignment['v0']['slot_id'] == assignment['v2']['slot_id']
    assert assignment['v1']['slot_id'] == assignment['v3']['slot_id']
    assert assignment['v4'] == {'kind': 'FINAL_OUTPUT'}
    for task in report['calls']:
        output = assignment[task['output_value']]
        assert all(output.get('slot_id') != assignment[value].get('slot_id')
                   for value in task['input_values'] if output['kind'] == 'WORKSPACE'
                   and assignment[value]['kind'] == 'WORKSPACE')
    assert verify_pipeline_memory(pipeline, report) == report
    altered = deepcopy(report); altered['assignment']['v2'] = assignment['v1']
    with pytest.raises(ValueError, match='source/lifetime replay'):
        verify_pipeline_memory(pipeline, altered)


def test_fanout_preserves_value_through_its_last_consumer_and_keeps_final_fresh():
    unary, binary = leaf(), leaf(2)
    calls = (call('a', unary, ('arg:0',), 'a'), call('b', unary, ('a',), 'b'),
             call('c', unary, ('a',), 'c'), call('d', binary, ('b', 'c'), 'd'))
    source = SourceRef('/tmp/pipeline-memory.py', '1' * 64, 'fanout', 1, 0, 'fanout-key')
    pipeline = compose_pipeline(source, ('x',), calls, 'd')
    report = plan_pipeline_memory(pipeline, input_shapes={'x': [2, 3]}, memory_budget_bytes=192)
    values = {value['id']: value for value in report['values']}
    assert values['a']['consumer_indices'] == [1, 2] and values['a']['release_after'] == 2
    assert report['assignment']['c']['slot_id'] != report['assignment']['a']['slot_id']
    assert len(report['workspace_slots']) == 3 and report['total_workspace_bytes'] == 144
    assert report['assignment']['d']['kind'] == 'FINAL_OUTPUT' and values['d']['release_after'] is None
    assert values['d']['escape'] and values['d']['ownership'] == 'CALLER'
    assert report['logical_release'] == 'REUSE_PERMISSION_ONLY_NOT_PHYSICAL_FREE'


def test_budget_rejection_and_unknown_fields_never_invent_free_memory_or_sizes():
    pipeline = chain()
    rejected = plan_pipeline_memory(pipeline, input_shapes=(2, 3), memory_budget_bytes=143)
    assert rejected['status'] == 'INFEASIBLE_WITHIN_FIXED_POOL_MODEL' and not rejected['budget_satisfied']
    assert rejected['lower_bound_peak_allocation_bytes'] == 144
    unknown_budget = plan_pipeline_memory(pipeline, input_shapes=(2, 3), memory_budget_bytes=None)
    assert unknown_budget['status'] == 'UNRESOLVED' and unknown_budget['assignment'] is None
    assert any(item['field'] == 'memory_budget_bytes' for item in unknown_budget['frontier'])
    unknown_shape = plan_pipeline_memory(pipeline, input_shapes=None, memory_budget_bytes=1000)
    assert unknown_shape['status'] == 'UNRESOLVED' and unknown_shape['final_bytes'] is None
    assert all(value['nbytes'] is None for value in unknown_shape['values'])
    assert 'THREAD_SCRATCH' in rejected['budget_excludes'] and not rejected['physical_optimality_proven']
