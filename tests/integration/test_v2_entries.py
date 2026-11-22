"""V2 artifact producers keep source and observation semantics distinct."""
import json

from scar.ir.frontend_v2 import build_semantic
from scar.ir.v2 import IRBundle
from scar.ir.v2.codec import semantic_from_dict
from scar.trace.normalize_v2 import normalize_trace


def test_model_v2_does_not_execute_source(tmp_path):
    program = tmp_path / "entry.py"
    marker = tmp_path / "executed"
    program.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
        "def compute(x):\n    return x + 1\nresult = compute(2)\n")
    result = build_semantic(program)
    graph = semantic_from_dict(result.graph.to_dict())
    assert graph.assert_valid()["valid"]
    assert graph.definitions and graph.slots and graph.edges
    assert result.coverage["target_executed"] is False
    assert not marker.exists()


def test_normalize_preserves_aggregate_without_fabricating_invocations(tmp_path):
    trace = tmp_path / "trace"
    trace.mkdir()
    event = {"index": 1, "kind": "torch_op", "code_id": "aten::mm",
             "invocation_id": None, "ts_ns": 100, "duration_ns": 2000,
             "inputs": [], "outputs": [], "labels": ["VAL"],
             "metadata": {"aggregate": True, "count": 7, "process_id": 1,
                          "thread_id": 2}, "resource": {}, "effects": {}}
    (trace / "events.jsonl").write_text(json.dumps(event) + "\n")
    result = normalize_trace(trace)
    bundle = IRBundle.from_dict(result.bundle.to_dict())
    assert bundle.assert_valid()["valid"]
    assert len(bundle.evidence.instances) == 0
    assert bundle.evidence.measurements
    assert bundle.optimization is None
    assert result.coverage["raw_records"] == result.coverage["accounted_records"] == 1
