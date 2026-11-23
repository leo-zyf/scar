"""One exact model, its dependency lowering, and incompatible constraints."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

import pytest

from scar.analysis.target_graph import build_target_graph
from scar.codegen.timed import admit_timed_plan
from scar.codegen import build, cpp, load
from scar.codegen.cpp import runtime_implementation_sha256
from scar.compiler.model import ArrayRegion, Expr, SourceRef
from scar.compiler.runtime import guarded_function
from scar.trace.code_snapshot import code_key
from scar.optimization import bind_cpu_placement, optimize_region, propose_cpu_placements
from scar.optimization.resource_solver import (
    hardware_model_skeleton, semantic_sha256, solve_resource_schedule,
)
from scar.optimization.schedule import build_region_schedule, Task, TaskKind


def test_fixed_cpu_timed_admission_checks_receipts_resources_and_lifetimes(tmp_path, monkeypatch):
    cpu = min(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 0
    inventory = {
        "schema": "scar.target-inventory", "schema_version": 1,
        "sources": [], "diagnostics": [], "dram": {"memory_total_bytes": 1024},
        "cpu": {"processors": [{"cpu_id": cpu, "package_id": 0, "core_id": 0}],
                "online_cpu_ids": [cpu], "model_names": ["fixture CPU"],
                "numa_nodes": [{"node_id": 0, "cpu_ids": [cpu], "memory_total_bytes": 1024}]},
        "gpus": [], "paths": [], "storage": {"disks": [], "block_devices": []},
    }
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory))
    target = tmp_path / "target"
    build_target_graph(target, inventory=inventory_path)
    source_path = tmp_path / "f.py"
    source_path.write_text("def f(x):\n    v = x[1:-1, :-1]\n    return (v + v) * v\n")
    namespace = {}
    exec(compile(source_path.read_text(), str(source_path), "exec"), namespace)
    original = namespace["f"]
    source = SourceRef(str(source_path), sha256(source_path.read_bytes()).hexdigest(),
                       "f", 1, 0, code_key(original.__code__))
    region = ArrayRegion("timed-region", source, ("x",), (
        Expr("x", "input", attrs={"name": "x"}),
        Expr("view", "slice", ("x",), {"bounds": [[1, -1], [None, -1]]}),
        Expr("a", "add", ("view", "view")), Expr("b", "mul", ("a", "view")),
    ), "b")
    initial = optimize_region(region, memory_budget_bytes=128)
    binding = propose_cpu_placements(initial, target, affinity=(cpu,), max_threads=1)[0]
    plan = bind_cpu_placement(initial, binding)
    schedule = build_region_schedule(plan, target)
    skeleton = hardware_model_skeleton(plan, schedule, input_shapes={"x": [4, 4]})
    # Explicit caller-declared model inputs. No task kind supplies a duration.
    result = solve_resource_schedule(schedule,
        task_costs={"allocate:output": 3, "compute:region": 11, "export:output": 2},
        **{key: skeleton[key] for key in (
            "task_placements", "resource_capacities", "memory_capacities",
            "materialization_sizes", "memory_locations", "model_instantiation")})
    assert result["status"] == "OPTIMAL_WITHIN_MODEL"
    admitted = admit_timed_plan(plan, schedule, result)
    assert admitted["dependency_order"] == ["allocate:output", "compute:region", "export:output"]
    assert admitted["model_makespan_ticks"] == 16
    assert admitted["must_guard"]["input_shapes"] == {"x": [4, 4]}
    assert admitted["must_guard"]["input_layout"] == "C_CONTIGUOUS"
    assert admitted["must_guard"]["output_shape"] == [2, 3]
    assert admitted["must_guard"]["output_nbytes"] == 48
    assert admitted["lowering"][1]["resources"] == {binding.executor_id: 1}
    assert not admitted["execution_linked"] and not admitted["deployment_authorized"]
    assert not admitted["physical_optimality_proven"] and admitted["memory_policy"] == "UNBOUND"

    changed = deepcopy(result)
    changed["schedule_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="schedule receipt"):
        admit_timed_plan(plan, schedule, changed)
    changed = deepcopy(result)
    changed["physical_optimality_proven"] = True
    with pytest.raises(ValueError, match="physical"):
        admit_timed_plan(plan, schedule, changed)
    changed = deepcopy(result)
    changed["deadline_ticks"] = 5
    with pytest.raises(ValueError, match="additional solver constraints"):
        admit_timed_plan(plan, schedule, changed)
    changed = deepcopy(result)
    del changed["model"]["instantiation"]
    changed["model_sha256"] = semantic_sha256(changed["model"])
    with pytest.raises(ValueError, match="shape instantiation"):
        admit_timed_plan(plan, schedule, changed)
    changed = deepcopy(result)
    changed["model"]["instantiation"]["output_shape"] = [4, 4]
    changed["model_sha256"] = semantic_sha256(changed["model"])
    with pytest.raises(ValueError, match="shape/source receipt"):
        admit_timed_plan(plan, schedule, changed)
    changed = deepcopy(result)
    changed["model"]["resource_demands"]["compute:region"] = {"unimplemented-gpu": 1}
    changed["model"]["resource_capacities"]["unimplemented-gpu"] = 1
    changed["model_sha256"] = semantic_sha256(changed["model"])
    with pytest.raises(ValueError, match="resource_demands"):
        admit_timed_plan(plan, schedule, changed)
    changed = deepcopy(result)
    task = next(row for row in changed["solution"]["tasks"] if row["task"] == "compute:region")
    task.update(start_ticks=0, end_ticks=11)
    with pytest.raises(ValueError, match="dependency"):
        admit_timed_plan(plan, schedule, changed)
    changed = deepcopy(result)
    changed["solution"]["allocation_lifetimes"][0]["end_ticks"] = 0
    with pytest.raises(ValueError, match="allocation lifetime"):
        admit_timed_plan(plan, schedule, changed)
    waiting = Task("unsupported-wait", TaskKind.WAIT, depends_on=("compute:region",))
    with pytest.raises(ValueError, match="COPY/WAIT"):
        admit_timed_plan(plan, replace(schedule, tasks=(*schedule.tasks, waiting)), result)

    if sys.platform != "linux" or shutil.which("g++") is None:
        return
    artifact = build(plan, tmp_path / "native", resource_schedule=schedule, timed_plan=result)
    loaded = load(artifact.directory)
    assert loaded.timed_admission["execution_linked"] is True
    assert loaded.report["timed_plan_execution"]["execution_linked"] is True
    assert loaded.report["runtime_implementation_sha256"] == runtime_implementation_sha256()
    assert not loaded.report["deployment_authorized"]
    data = np.arange(16, dtype=np.float64).reshape(4, 4)
    np.testing.assert_array_equal(artifact.run(data), original(data))
    np.testing.assert_array_equal(loaded.run(data), original(data))
    assert loaded.run(data).nbytes == 48
    wrapped = guarded_function(original, loaded, exclusive_inputs=True, trial=True)
    wrong_shape = np.ones((5, 4))
    np.testing.assert_array_equal(wrapped(wrong_shape), original(wrong_shape))
    assert wrapped.scar_stats["native_calls"] == 0
    assert wrapped.scar_stats["fallback_calls"] == 1
    assert any("timed plan input_shapes" in reason for reason in wrapped.scar_stats["reasons"])
    noncontiguous = data.T
    assert noncontiguous.shape == data.shape and not noncontiguous.flags.c_contiguous
    np.testing.assert_array_equal(wrapped(noncontiguous), original(noncontiguous))
    assert wrapped.scar_stats["native_calls"] == 0
    assert wrapped.scar_stats["fallback_calls"] == 2
    assert any("timed plan input_layout" in reason for reason in wrapped.scar_stats["reasons"])

    # Raw solver integrity and linked guard semantics are independently checked.
    raw_path = loaded.timed_plan_path
    guard_path = loaded.timed_admission_path
    metadata_path = loaded.metadata_path
    raw_bytes, guard_bytes, metadata_bytes = raw_path.read_bytes(), guard_path.read_bytes(), metadata_path.read_bytes()
    try:
        raw_path.write_bytes(raw_bytes + b" ")
        with pytest.raises(ValueError, match="timed_plan digest"):
            load(artifact.directory)
        raw_path.write_bytes(raw_bytes)
        changed_admission = json.loads(guard_bytes)
        changed_admission["must_guard"]["output_nbytes"] = 72
        changed_bytes = (json.dumps(changed_admission, sort_keys=True, indent=2) + "\n").encode()
        guard_path.write_bytes(changed_bytes)
        changed_metadata = json.loads(metadata_bytes)
        changed_metadata["sha256"]["timed_admission"] = sha256(changed_bytes).hexdigest()
        changed_metadata["timed_admission_sha256"] = sha256(changed_bytes).hexdigest()
        metadata_path.write_text(json.dumps(changed_metadata))
        with pytest.raises(ValueError, match="timed admission differs"):
            load(artifact.directory)
    finally:
        raw_path.write_bytes(raw_bytes)
        guard_path.write_bytes(guard_bytes)
        metadata_path.write_bytes(metadata_bytes)
    np.testing.assert_array_equal(load(artifact.directory).run(data), original(data))

    write = Path.write_bytes
    for case, member, reason in (
        ("staged-plan", tmp_path / "staged-plan" / "plan.json", "artifact plan bytes differ"),
        ("staged-source", source_path, "metadata dependency byte identity differs"),
    ):
        output = tmp_path / case
        saved = []

        def change_after_manifest(path, data):
            written = write(path, data)
            if path == output / "artifact.json":
                saved.append(member.read_bytes())
                write(member, saved[0] + b" ")
            return written

        def unexpected_binding(*args):
            pytest.fail("changed artifact reached native library binding")

        try:
            with monkeypatch.context() as patched:
                patched.setattr(Path, "write_bytes", change_after_manifest)
                patched.setattr(cpp, "_bind_abi", unexpected_binding)
                with pytest.raises(ValueError, match=reason):
                    build(plan, output, resource_schedule=schedule, timed_plan=result)
        finally:
            if saved:
                write(member, saved[0])
        assert saved
