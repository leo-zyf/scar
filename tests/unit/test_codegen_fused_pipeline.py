"""Complete fusion retains the original boundary and invokes one native kernel."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from scar.codegen import cpp
from scar.codegen.fused_pipeline import GuardFailure, build, load
from scar.compiler.model import ArrayRegion, Expr, SourceRef
from scar.compiler.pipeline_frontend import CalleeGuard, RegionCall, compose_pipeline
from scar.compiler.pipeline_runtime import guarded_pipeline_function
from scar.optimization import optimize_region
from scar.optimization.pipeline_strategy import fused_execution_plan, plan_pipeline_strategy
from scar.trace.code_snapshot import code_key


def test_complete_fusion_keeps_original_guards_and_one_fresh_native_result(tmp_path, monkeypatch):
    if shutil.which("g++") is None:
        pytest.skip("g++ required for fused C++ execution")
    path = tmp_path / "functions.py"
    path.write_text("def leaf(x):\n    return (x + 1.0) * 0.5\n\n"
                    "def composed(x):\n    a = leaf(x)\n    b = leaf(a)\n"
                    "    c = leaf(b)\n    return leaf(c)\n")
    namespace = {}; exec(compile(path.read_text(), str(path), "exec"), namespace)
    def source(name):
        function = namespace[name]
        return SourceRef(str(path), sha256(path.read_bytes()).hexdigest(), name,
                         function.__code__.co_firstlineno, 0, code_key(function.__code__))
    leaf = optimize_region(ArrayRegion("leaf", source("leaf"), ("x",), (
        Expr("x", "input", attrs={"name": "x"}), Expr("one", "const", attrs={"value": 1.0}),
        Expr("a", "add", ("x", "one")), Expr("half", "const", attrs={"value": 0.5}),
        Expr("result", "mul", ("a", "half"))), "result"))
    calls = tuple(RegionCall(f"call:{i}", leaf, ("arg:0" if i == 0 else f"value:{i-1}",),
                             f"value:{i}", CalleeGuard(source("leaf"), ("leaf",))) for i in range(4))
    pipeline = compose_pipeline(source("composed"), ("x",), calls, "value:3")
    strategy = plan_pipeline_strategy(pipeline, input_shapes={"x": [3, 5]}, memory_budget_bytes=120)
    assert strategy["selected"]["kind"] == "FULL_FUSION"
    operator = cpp.build(fused_execution_plan(pipeline, strategy), tmp_path / "kernel")
    artifact = build(pipeline, tmp_path / "fused", strategy=strategy, operator=operator)
    assert artifact.pipeline == pipeline and artifact.pipeline.calls == calls
    assert artifact.memory_plan["total_workspace_bytes"] == artifact.report["workspace_bytes"] == 0
    assert artifact.report["kernel_count"] == artifact.report["native_ffi_calls_per_run"] == 1
    invocations = []; native = artifact._operator._run
    def count(*arguments):
        invocations.append(1)
        return native(*arguments)
    artifact._operator._run = count
    array = np.arange(15, dtype=np.float64).reshape(3, 5); array[0, 0] = -0.0
    before = array.copy(); original = namespace["composed"]
    actual = artifact.run(array)
    assert invocations == [1]
    np.testing.assert_array_equal(actual.view(np.uint64), original(array).view(np.uint64))
    np.testing.assert_array_equal(array.view(np.uint64), before.view(np.uint64))
    assert actual.flags.owndata and not np.shares_memory(actual, array)
    actual[:] = -123; array += 0.125
    np.testing.assert_array_equal(load(artifact.directory).run(array), original(array))
    with pytest.raises(GuardFailure, match="shape/type/layout"):
        artifact.run(np.ones((4, 5)))
    with pytest.raises(GuardFailure, match="shape/type/layout"):
        artifact.run(np.arange(30., dtype=np.float64).reshape(6, 5)[::2])
    wrapped = guarded_pipeline_function(original, artifact, exclusive_inputs=True, trial=True)
    np.testing.assert_array_equal(wrapped(array), original(array))
    assert wrapped.scar_stats["native_calls"] == 1
    namespace["leaf"] = lambda x: x - 1.0
    np.testing.assert_array_equal(wrapped(array), original(array))
    assert wrapped.scar_stats["fallback_calls"] == 1
    assert wrapped.scar_stats["reasons"] == {"actual_callee_changed": 1}
    bad = deepcopy(strategy); bad["selected_memory_plan"]["final_bytes"] -= 8
    with pytest.raises(ValueError):
        build(pipeline, tmp_path / "bad", strategy=bad, operator=operator)
    assert not (tmp_path / "bad").exists()
    strategy_path = artifact.directory / "strategy.json"
    raw = strategy_path.read_bytes(); strategy_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="sidecar receipt"):
        load(artifact.directory)
    strategy_path.write_bytes(raw)
    plan_path = operator.plan_path
    raw = plan_path.read_bytes(); plan_path.write_bytes(raw + b"tamper")
    try:
        with pytest.raises(ValueError, match="operator receipt"):
            load(artifact.directory)
    finally:
        plan_path.write_bytes(raw)

    write = Path.write_bytes
    for folder, member, reason in (
        ("changed-sidecar", tmp_path / "changed-sidecar" / "strategy.json", "sidecar receipt"),
        ("changed-operator", plan_path, "operator receipt"),
        ("changed-source", path, "original pipeline/callee source snapshot differs"),
    ):
        destination = tmp_path / folder
        saved = []

        def write_then_change(filename, data):
            written = write(filename, data)
            if filename == destination / "artifact.json":
                saved.append(member.read_bytes())
                write(member, saved[0] + b" ")
            return written

        try:
            with monkeypatch.context() as patched:
                patched.setattr(Path, "write_bytes", write_then_change)
                with pytest.raises(ValueError, match=reason):
                    build(pipeline, destination, strategy=strategy, operator=operator)
        finally:
            if saved:
                write(member, saved[0])
        assert saved
