from test_pipeline_schedule import _pipeline

from scar.ir.versioned_schedule import validate_versioned_schedule
from scar.optimization.pipeline_memory import plan_pipeline_memory
from scar.optimization.pipeline_schedule import _dependencies
from scar.optimization.pipeline_ready_schedule import _call_bindings, _dependency_graph, _ready_calls


def test_true_ready_successor_needs_no_other_lane_join_and_pool_consumer_still_orders_reuse():
    pipeline = _pipeline([('a', ('arg:0',)), ('b', ('arg:0',)),
                          ('c', ('a',)), ('d', ('b', 'c'))])
    memory = plan_pipeline_memory(pipeline, input_shapes=[2, 3], memory_budget_bytes=4096)
    predecessors, _ = _dependencies(pipeline, memory)
    calls = _call_bindings(pipeline, (4, 9))
    assert predecessors['c'] == ['a']
    # b is still running on the other lane. c can begin immediately after a.
    assert _ready_calls([call.id for call in pipeline.calls], predecessors, calls,
                        {'a'}, (4, 9), busy_cpu_ids=(9,)) == {4: 'c'}
    graph = _dependency_graph(pipeline, memory, predecessors, calls,
                              ({'kind': 'STRUCTURAL_TEST_ONLY'},))
    assert validate_versioned_schedule(graph)['overwrite_happens_before_verified'] is True
    compute_c = next(task for task in graph.tasks if task.id == 'compute:2')
    assert 'compute:0' in compute_c.depends_on and 'compute:1' not in compute_c.depends_on
    assert graph.proof['intermediate_global_wave_barriers'] is False

    # Here independent c reuses a's old pool slot, so b (a's consumer) really
    # must finish first even though c's value inputs have no dependency on it.
    reused = _pipeline([('a', ('arg:0',)), ('b', ('a',)),
                        ('c', ('arg:0',)), ('d', ('b', 'c'))])
    pool = plan_pipeline_memory(reused, input_shapes=[2, 3], memory_budget_bytes=4096)
    required, reasons = _dependencies(reused, pool)
    assert any(row['kind'] == 'WAR_SLOT_REUSE' and row['before'] == 'b'
               and row['after'] == 'c' for row in reasons)
    bindings = _call_bindings(reused, (4, 9))
    assert _ready_calls([call.id for call in reused.calls], required, bindings,
                        {'a'}, (4, 9), busy_cpu_ids=(9,)) == {}
    assert _ready_calls([call.id for call in reused.calls], required, bindings,
                        {'a', 'b'}, (4, 9)) == {4: 'c'}
    validate_versioned_schedule(_dependency_graph(reused, pool, required, bindings,
                                                 ({'kind': 'STRUCTURAL_TEST_ONLY'},)))
