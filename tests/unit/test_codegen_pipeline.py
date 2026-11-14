"""A native four-call chain reuses storage and exports only a fresh final."""
from copy import deepcopy
import ctypes
from hashlib import sha256
import json
import os
import shutil

import numpy as np
import pytest

from scar.analysis.target_graph import build_target_graph
from scar.codegen import cpp
from scar.codegen.pipeline import GuardFailure, build, load
from scar.compiler.model import ArrayRegion, Expr, SourceRef
from scar.compiler.pipeline_model import CalleeGuard, RegionCall, compose_pipeline
from scar.optimization import bind_cpu_placement, optimize_region, propose_cpu_placements
from scar.optimization.pipeline_memory import plan_pipeline_memory
from scar.optimization.resource_solver import hardware_model_skeleton, solve_resource_schedule
from scar.optimization.schedule import build_region_schedule
from scar.trace.code_snapshot import code_key


def test_native_pipeline_reuses_slots_and_guards_one_ffi_entry(tmp_path, monkeypatch):
    if shutil.which("g++") is None or not hasattr(os, "sched_getaffinity"):
        pytest.skip("Linux and g++ required for native pipeline")
    source_path = tmp_path / "functions.py"
    source_path.write_text("def leaf(x):\n    return (x + 1.0) * 0.5\n\n"
                           "def composed(x):\n    a = leaf(x)\n    b = leaf(a)\n"
                           "    c = leaf(b)\n    return leaf(c)\n")
    namespace = {}; exec(compile(source_path.read_text(), str(source_path), "exec"), namespace)
    digest = sha256(source_path.read_bytes()).hexdigest()
    def ref(name):
        fn = namespace[name]
        return SourceRef(str(source_path), digest, name, fn.__code__.co_firstlineno, 0, code_key(fn.__code__))
    region = ArrayRegion("leaf", ref("leaf"), ("x",), (
        Expr("x", "input", attrs={"name": "x"}), Expr("one", "const", attrs={"value": 1.0}),
        Expr("a", "add", ("x", "one")), Expr("half", "const", attrs={"value": 0.5}),
        Expr("result", "mul", ("a", "half"))), "result")
    leaf = optimize_region(region)
    calls = tuple(RegionCall(f"call:{i}", leaf, ("arg:0" if i == 0 else f"value:{i-1}",),
                             f"value:{i}", CalleeGuard(ref("leaf"), ("leaf",))) for i in range(4))
    pipeline = compose_pipeline(ref("composed"), ("x",), calls, "value:3")
    memory = plan_pipeline_memory(pipeline, input_shapes=(3, 5), memory_budget_bytes=360)
    assert memory["total_workspace_bytes"] == 240 and memory["final_bytes"] == 120
    assert any(len(slot["values"]) > 1 for slot in memory["workspace_slots"])

    # A real timed leaf tests that C++ dispatch cannot bypass Python-layer
    # model shape/layout/budget obligations merely by calling its ABI table.
    cpu = min(os.sched_getaffinity(0))
    inventory = {"schema": "scar.target-inventory", "schema_version": 1, "sources": [],
        "diagnostics": [], "dram": {"memory_total_bytes": 4096},
        "cpu": {"processors": [{"cpu_id": cpu, "package_id": 0, "core_id": 0}],
                "online_cpu_ids": [cpu], "model_names": ["fixture CPU"],
                "numa_nodes": [{"node_id": 0, "cpu_ids": [cpu], "memory_total_bytes": 4096}]},
        "gpus": [], "paths": [], "storage": {"disks": [], "block_devices": []}}
    inventory_path = tmp_path / "inventory.json"; inventory_path.write_text(json.dumps(inventory))
    target = tmp_path / "target"; build_target_graph(target, inventory=inventory_path)
    placement = propose_cpu_placements(leaf, target, affinity=(cpu,), memory_budget_bytes=120, max_threads=1)[0]
    bound = bind_cpu_placement(leaf, placement)
    schedule = build_region_schedule(bound, target)
    model = hardware_model_skeleton(bound, schedule, input_shapes=(3, 5))
    solved = solve_resource_schedule(schedule,
        task_costs={"allocate:output": 1, "compute:region": 1, "export:output": 1},
        **{key: model[key] for key in ("task_placements", "resource_capacities", "memory_capacities",
             "materialization_sizes", "memory_locations", "model_instantiation")})
    operator = cpp.build(bound, tmp_path / "operator", resource_schedule=schedule, timed_plan=solved)
    artifact = build(pipeline, tmp_path / "host", memory_plan=memory,
                     operators={call.id: operator for call in calls})
    count = [0]; native = artifact._run
    def one_entry(*arguments):
        count[0] += 1
        return native(*arguments)
    artifact._run = one_entry
    monkeypatch.setattr(cpp.CppArtifact, "run", lambda *args: pytest.fail("Python leaf dispatch was used"))
    data = np.arange(15, dtype=np.float64).reshape(3, 5)
    snapshot = data.copy(); affinity = os.sched_getaffinity(0)
    actual = artifact.run(data)
    assert count == [1]
    np.testing.assert_array_equal(actual.view(np.uint64), namespace["composed"](data).view(np.uint64))
    np.testing.assert_array_equal(data, snapshot)
    assert actual.flags.owndata and not np.shares_memory(actual, data)
    assert os.sched_getaffinity(0) == affinity
    actual[:] = -999; data[:] += 0.125
    np.testing.assert_array_equal(artifact.run(data), namespace["composed"](data))
    np.testing.assert_array_equal(load(artifact.directory).run(data), namespace["composed"](data))
    with pytest.raises(GuardFailure, match="input shape"):
        artifact.run(np.ones((4, 5)))
    strided = np.arange(30, dtype=np.float64).reshape(6, 5)[::2]
    with pytest.raises(GuardFailure, match="shape/layout"):
        artifact.run(strided)
    with pytest.raises(GuardFailure, match="floating-point"):
        artifact.run(np.full((3, 5), np.nan))
    assert os.sched_getaffinity(0) == affinity

    # Wrong source table is rejected before any operator can execute.
    wrong = cpp._Table.from_buffer_copy(artifact._tables[0].contents)
    wrong.plan_sha256 = b"0" * 64
    table_type = type(artifact._tables)
    tables = table_type(ctypes.pointer(wrong), *list(artifact._tables)[1:])
    output = np.empty((3, 5)); output_descriptor = cpp._descriptor(output)
    inputs = (cpp._Array * 1)(cpp._descriptor(data)); failed = ctypes.c_uint64()
    assert native(inputs, 1, ctypes.byref(output_descriptor), tables, 4, ctypes.byref(failed)) == 14
    bad = deepcopy(memory)
    bad["assignment"]["value:1"] = deepcopy(bad["assignment"]["value:0"])
    with pytest.raises(ValueError, match="source/lifetime replay"):
        build(pipeline, tmp_path / "bad", memory_plan=bad, operators={call.id: operator for call in calls})
    assert not (tmp_path / "bad").exists()
