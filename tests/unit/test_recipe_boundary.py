from copy import deepcopy
from dataclasses import replace
import json

import pytest

from test_loop_strategy import make_loop
from scar.analysis.target_graph import build_target_graph
from scar.compiler.recipe_boundary import build_loop_boundary, verify_loop_boundary
from scar.optimization import bind_cpu_placement, propose_cpu_placements
from scar.optimization.loop_strategy import plan_loop_strategy, loop_execution_plan


def test_resource_independent_boundary_preserves_count_q_and_exact_p(tmp_path, monkeypatch):
    monkeypatch.setattr('os.sched_getaffinity', lambda _: {0, 1, 2, 3})
    inventory = dict(schema='scar.target-inventory', schema_version=1, sources=[], diagnostics=[],
        cpu=dict(online_cpu_ids=[0, 1, 2, 3], model_names=['fixture'],
            processors=[dict(cpu_id=c, package_id=0, core_id=c) for c in range(4)],
            numa_nodes=[dict(node_id=0, cpu_ids=[0, 1, 2, 3], memory_total_bytes=1024)]),
        dram=dict(memory_total_bytes=1024), gpus=[], storage=dict(disks=[], block_devices=[]), paths=[])
    path = tmp_path / 'inventory.json'; path.write_text(json.dumps(inventory))
    target = tmp_path / 'target'; build_target_graph(target, inventory=path)
    loop = make_loop()
    strategy = plan_loop_strategy(loop, input_shapes={'state': [2, 3]}, trip_count=5, memory_budget_bytes=48)
    plan = loop_execution_plan(loop, strategy)
    original = deepcopy((loop, strategy, plan))
    bindings = propose_cpu_placements(plan, target, max_threads=4, memory_budget_bytes=48)
    two, four = (bind_cpu_placement(plan, next(p for p in bindings if len(p.cpu_ids) == count))
                 for count in (2, 4))
    boundary = build_loop_boundary(loop, strategy, two)
    assert boundary == build_loop_boundary(loop, strategy, four) == build_loop_boundary(loop, strategy, plan)
    assert boundary['input']['shape'] == boundary['output']['shape'] == [2, 3]
    assert boundary['iteration']['count'] == 5 and boundary['iteration']['unrolled'] is False
    assert boundary['parallelism']['ordered_axes'] == ['time']
    assert boundary['dependencies']['device_difference_removes_dependencies'] is False
    assert boundary['state']['body_input']['id'] == 'body:input'
    assert 'NO_ITERATION_DELETION' in boundary['Q']['algorithm_invariants']
    assert boundary['state']['content_versions_inferred'] is False
    assert boundary['input']['shape_authority']['observed_vs_declared'] == 'NOT_DISTINGUISHED_BY_INPUT_STRATEGY'
    assert boundary['provenance']['source_operation_map'] == plan.proof['source_operation_map']
    assert len(boundary['provenance']['calls']) == len(loop.body.calls)
    assert 'cpu_threads' not in json.dumps(boundary) and 'resource_binding' not in json.dumps(boundary)
    assert verify_loop_boundary(loop, strategy, four, json.loads(json.dumps(boundary))) == boundary
    changed = plan_loop_strategy(loop, input_shapes={'state': [2, 3]}, trip_count=6, memory_budget_bytes=48)
    assert build_loop_boundary(loop, changed, four)['id'] != boundary['id']
    malformed = deepcopy(four.proof); malformed['source_operation_map'] = {}
    with pytest.raises(ValueError, match='exact replay'):
        build_loop_boundary(loop, strategy, replace(four, proof=malformed))
    malformed = deepcopy(four.cost); malformed['baseline_array_allocations'] += 1
    with pytest.raises(ValueError, match='exact replay'):
        build_loop_boundary(loop, strategy, replace(four, cost=malformed))
    tampered = deepcopy(boundary); tampered['Q']['reassociation'] = True
    with pytest.raises(ValueError, match='complete replay'):
        verify_loop_boundary(loop, strategy, four, tampered)
    assert (loop, strategy, plan) == original
