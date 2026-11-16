"""Projection mechanics only: these records do not authorize source lowering."""
import pytest

from test_pipeline_schedule import _pipeline
from scar.ir.versioned_schedule import schedule_document
from scar.optimization.pipeline_execution_refinement import _acyclic, _projection
from scar.optimization.pipeline_partial_order_memory import plan_pipeline_partial_order_memory
from scar.optimization.pipeline_ready_schedule import _dependencies, _dependency_graph


def test_same_tick_zero_call_precedes_positive_and_lane_cycles_are_rejected():
    pipeline = _pipeline([('a', ('arg:0',)), ('b', ('arg:0',)), ('c', ('a', 'b'))])
    memory = plan_pipeline_partial_order_memory(pipeline, input_shapes=[2, 3], memory_budget_bytes=4096)
    predecessors, _ = _dependencies(pipeline, memory)
    cpus = {call.id: 4 for call in pipeline.calls}
    graph = _dependency_graph(pipeline, memory, predecessors, cpus, ({'kind': 'STRUCTURAL_TEST_ONLY'},))
    times = {'compute:0': (0, 3), 'compute:1': (0, 0), 'compute:2': (3, 4)}
    rows = []
    for task in graph.tasks:
        start, end = times.get(task.id, (4, 4) if not task.id.startswith('allocate:') else (0, 0))
        rows.append(dict(task=task.id, start_ticks=start, end_ticks=end))
    temporal = dict(cost_model={'role': 'STRUCTURAL_PROJECTION_TEST_NOT_A_SOURCE_CERTIFICATE'},
        compute_call_map={'compute:0': 'a', 'compute:1': 'b', 'compute:2': 'c'},
        solver_report=dict(status='OPTIMAL_WITHIN_MODEL', optimality='MODEL_OPTIMUM_ESTABLISHED',
            search_budget_exhausted=False, unresolved=[], execution_unresolved=[],
            source_schedule_executable=True, solution={'tasks': rows}))
    ready = dict(task_graph=schedule_document(graph), cpu_ids=[4], call_cpu_ids=cpus,
                 call_predecessors=predecessors)
    lanes, projected, _, _ = _projection(pipeline, ready, temporal)
    assert lanes == {'4': ['b', 'a', 'c']}
    assert len(projected) == 3  # A zero-duration call is retained, not deleted.
    with pytest.raises(ValueError, match='cycle'):
        _acyclic(['a', 'b', 'c'], predecessors, {'4': ['c', 'a', 'b']})
