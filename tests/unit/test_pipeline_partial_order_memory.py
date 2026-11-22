from copy import deepcopy

import pytest

from test_pipeline_schedule import _pipeline
from scar.optimization.pipeline_memory import plan_pipeline_memory
from scar.optimization.pipeline_schedule import _dependencies
from scar.optimization.pipeline_memory import OPTIMAL, plan_pipeline_partial_order_memory, verify_pipeline_partial_order_memory


def test_independent_writer_keeps_its_slot_instead_of_serializing_behind_old_consumer():
    pipeline = _pipeline([('a', ('arg:0',)), ('b', ('a',)),
                          ('c', ('arg:0',)), ('d', ('b', 'c'))])
    old = plan_pipeline_memory(pipeline, input_shapes=[2, 3], memory_budget_bytes=4096)
    old_preds, old_reasons = _dependencies(pipeline, old)
    assert 'b' in old_preds['c']
    assert any(row['kind']=='WAR_SLOT_REUSE' and row['before']=='b' and row['after']=='c'
               for row in old_reasons)
    plan = plan_pipeline_partial_order_memory(pipeline, input_shapes=[2, 3], memory_budget_bytes=4096)
    assert plan['status']==OPTIMAL and plan['search_budget_exhausted'] is False
    assert verify_pipeline_partial_order_memory(pipeline, plan)==plan
    predecessors, _ = _dependencies(pipeline, plan)
    assert predecessors['c']==[]
    assert predecessors['b']==['a'] and predecessors['d']==['b', 'c']
    assert plan['total_workspace_bytes'] > old['total_workspace_bytes']
    assert plan['additional_storage_order_reachability'] is False


def test_same_call_input_cannot_be_overwritten_but_strict_raw_order_allows_later_reuse():
    pipeline = _pipeline([('a', ('arg:0',)), ('b', ('a',)),
                          ('c', ('b',)), ('d', ('c',))])
    plan = plan_pipeline_partial_order_memory(pipeline, input_shapes=[2, 3], memory_budget_bytes=4096)
    assignment = plan['assignment']
    assert assignment['a']['slot_id'] != assignment['b']['slot_id']
    assert assignment['a']['slot_id'] == assignment['c']['slot_id']
    proof = next(row for row in plan['reuse_proofs'] if row['previous_value']=='a' and row['next_value']=='c')
    assert proof['all_old_consumers']==['b'] and proof['new_producer_is_not_old_consumer']
    assert plan['raw_predecessors']=={'a':[], 'b':['a'], 'c':['b'], 'd':['c']}
    bad = deepcopy(plan)
    bad['assignment']['b'] = deepcopy(bad['assignment']['a'])
    with pytest.raises(ValueError, match='RAW/shape/interference/search/budget replay'):
        verify_pipeline_partial_order_memory(pipeline, bad)
