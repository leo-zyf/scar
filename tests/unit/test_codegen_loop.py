"""A small CPU-bound recurrence executes one shared DAG and retains fallbacks."""
import ctypes
from hashlib import sha256
import json
import os
import shutil

import numpy as np
import pytest

from scar.analysis.target_graph import build_target_graph
from scar.codegen import cpp
from scar.codegen.loop import GuardFailure, build, load
from scar.compiler.loop_model import ArrayLoopPlan
from scar.compiler.loop_runtime import guarded_loop_function
from scar.compiler.model import ArrayRegion, Expr, SourceRef
from scar.compiler.pipeline_model import CalleeGuard, RegionCall, compose_pipeline
from scar.optimization import bind_cpu_placement, optimize_region, propose_cpu_placements
from scar.optimization.loop_strategy import plan_loop_strategy, loop_execution_plan
from scar.trace.code_snapshot import code_key


def test_bound_temporal_loop_one_entry_fresh_bits_guards_and_environment(tmp_path):
    if shutil.which("g++") is None or not hasattr(os, "sched_getaffinity"):
        pytest.skip("Linux and g++ required for bound loop execution")
    body_path = tmp_path / "body.py"
    body_path.write_text("def leaf(x):\n    return (x + 1.0) * 2.0\n\n"
                         "def advance(u):\n    a = leaf(u)\n    return leaf(a)\n")
    body = {}; exec(compile(body_path.read_text(), str(body_path), "exec"), body)
    outer_path = tmp_path / "outer.py"
    outer_path.write_text("def recurrence(steps, x):\n    state = x\n    for _ in range(steps):\n"
                          "        state = advance(state)\n    return state\n")
    outer = {"advance": body["advance"]}
    exec(compile(outer_path.read_text(), str(outer_path), "exec"), outer)
    def source(namespace, path, name):
        function = namespace[name]
        return SourceRef(str(path), sha256(path.read_bytes()).hexdigest(), name,
                         function.__code__.co_firstlineno, 0, code_key(function.__code__))
    leaf_source = source(body, body_path, "leaf")
    leaf = optimize_region(ArrayRegion("leaf", leaf_source, ("x",), (
        Expr("x", "input", attrs={"name": "x"}), Expr("one", "const", attrs={"value": 1.0}),
        Expr("a", "add", ("x", "one")), Expr("two", "const", attrs={"value": 2.0}),
        Expr("out", "mul", ("a", "two"))), "out"))
    body_source = source(body, body_path, "advance")
    calls = (RegionCall("a", leaf, ("arg:0",), "a", CalleeGuard(leaf_source, ("leaf",))),
             RegionCall("b", leaf, ("a",), "b", CalleeGuard(leaf_source, ("leaf",))))
    pipeline = compose_pipeline(body_source, ("u",), calls, "b")
    loop = ArrayLoopPlan("recurrence", source(outer, outer_path, "recurrence"),
                        ("steps", "x"), "x", "steps", pipeline, CalleeGuard(body_source, ("advance",)))
    strategy = plan_loop_strategy(loop, input_shapes={"x": [3, 5]}, trip_count=7, memory_budget_bytes=120)
    assert strategy["selected"]["kind"] == "TEMPORAL_FUSION"
    plan = loop_execution_plan(loop, strategy)
    cpus = sorted(os.sched_getaffinity(0))[:2]
    inventory = {"schema": "scar.target-inventory", "schema_version": 1, "sources": [],
        "diagnostics": [], "dram": {"memory_total_bytes": 4096},
        "cpu": {"processors": [{"cpu_id": cpu, "package_id": 0, "core_id": i} for i, cpu in enumerate(cpus)],
                "online_cpu_ids": cpus, "model_names": ["fixture CPU"],
                "numa_nodes": [{"node_id": 0, "cpu_ids": cpus, "memory_total_bytes": 4096}]},
        "gpus": [], "paths": [], "storage": {"disks": [], "block_devices": []}}
    inventory_path = tmp_path / "inventory.json"; inventory_path.write_text(json.dumps(inventory))
    target = tmp_path / "target"; build_target_graph(target, inventory=inventory_path)
    placements = propose_cpu_placements(plan, target, affinity=cpus, memory_budget_bytes=120, max_threads=len(cpus))
    placement = next(value for value in placements if len(value.cpu_ids) == len(cpus))
    bound = bind_cpu_placement(plan, placement)
    artifact = build(loop, tmp_path / "native", strategy=strategy, execution_plan=bound)
    assert artifact.loop == loop and artifact.execution_plan == bound
    assert artifact.report["workspace_bytes"] == 0 and artifact.report["trip_count"] == 7
    assert artifact.report["body_source_unrolled"] is False
    text = (artifact.directory / "loop.cpp").read_text()
    assert text.count("for (uint64_t iteration") == 1
    invocations = []; native = artifact._run
    def count(*arguments):
        invocations.append(1)
        return native(*arguments)
    artifact._run = count
    data = np.arange(15, dtype=np.float64).reshape(3, 5); data[0, 0] = -0.0
    before = data.copy(); affinity = os.sched_getaffinity(0); errors = np.geterr()
    original = outer["recurrence"]
    actual = artifact.run(data, 7)
    assert invocations == [1]
    np.testing.assert_array_equal(actual.view(np.uint64), original(7, data).view(np.uint64))
    np.testing.assert_array_equal(data.view(np.uint64), before.view(np.uint64))
    assert actual.flags.owndata and not np.shares_memory(actual, data)
    assert os.sched_getaffinity(0) == affinity and np.geterr() == errors
    actual[:] = -123
    np.testing.assert_array_equal(load(artifact.directory).run(data, 7), original(7, data))
    for steps in (0, -1, 6, True, np.int64(7)):
        with pytest.raises(GuardFailure, match="trip count"):
            artifact.run(data, steps)
    with pytest.raises(GuardFailure, match="shape/layout"):
        artifact.run(np.ones((4, 5)), 7)
    with pytest.raises(GuardFailure, match="shape/layout"):
        artifact.run(np.arange(30.).reshape(6, 5)[::2], 7)
    with pytest.raises(GuardFailure, match="non-finite"):
        artifact.run(np.full((3, 5), np.nan), 7)
    descriptor = cpp._descriptor(data); output = np.full((3, 5), 123.0); out = cpp._descriptor(output)
    assert native(ctypes.byref(descriptor), 6, ctypes.byref(out)) == 5
    assert np.all(output == 123.0)
    wrapped = guarded_loop_function(original, artifact, exclusive_inputs=True, trial=True)
    np.testing.assert_array_equal(wrapped(7, data), original(7, data))
    # The true leaf lives in the body's globals, not the loop's globals.
    outer["leaf"] = lambda x: x - 1.0
    np.testing.assert_array_equal(wrapped(7, data), original(7, data))
    original_leaf = body["leaf"]; body["leaf"] = lambda x: x - 1.0
    np.testing.assert_array_equal(wrapped(7, data), original(7, data))
    assert wrapped.scar_stats["reasons"] == {"actual_nested_callee_changed": 1}
    body["leaf"] = original_leaf
    with np.errstate(over="raise"):
        with pytest.raises(FloatingPointError):
            wrapped(7, np.full((3, 5), 1e308))
    assert wrapped.scar_stats["fallback_calls"] == 2
    assert os.sched_getaffinity(0) == affinity and np.geterr() == errors
    if len(cpus) == 2:
        try:
            os.sched_setaffinity(0, {cpus[0]})
            with pytest.raises(GuardFailure, match="affinity"):
                artifact.run(data, 7)
        finally:
            os.sched_setaffinity(0, affinity)
    strategy_path = artifact.directory / "strategy.json"
    raw = strategy_path.read_bytes(); strategy_path.write_bytes(raw + b" ")
    try:
        with pytest.raises(ValueError, match="receipt differs"):
            load(artifact.directory)
    finally:
        strategy_path.write_bytes(raw)
