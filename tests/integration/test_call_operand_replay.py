"""Replay typed call operand routes from actual Python trace profiles."""
import json

import pytest

from scar.analysis.call_operand_routes import CallOperandReport, link_call_operands
from scar.analysis.program_inventory import inventory_program
from scar.ir.program.call_operand_routes import CoreReferenceKind, OperandQueryRouteKind
from tests.fixtures.capture_cli import run_trace


@pytest.fixture(scope="module", params=[("windows", "distinct"), ("windows", "shared"),
                                        ("windows", "snapshot"), ("instructions", "shared")])
def linked(tmp_path_factory, request):
    mode, alias = request.param
    project = tmp_path_factory.mktemp(f"operand-{mode}-{alias}")
    (project / "helper.py").write_text(
        "def choose(left, right):\n    return left\n"
        "def relay(a, b):\n    selected = choose(a, b)\n    return selected\n",
        encoding="utf-8",
    )
    inputs = {"distinct": "left = Token()\nright = Token()\n",
              "shared": "left = Token()\nright = left\n",
              "snapshot": "left = []\nright = left\n"}[alias]
    iterations = 100_000 if (mode, alias) == ("windows", "shared") else 3
    entry = project / "entry.py"
    entry.write_text(
        "from helper import relay\nclass Token: pass\n"
        "def preparation(n):\n    pairs = [(i, i + 1) for i in range(n)]\n    return len(pairs)\n"
        "def run(x, y):\n    first = relay(x, y)\n    second = relay(y, x)\n    return first\n"
        f"assert preparation({iterations}) == {iterations}\n"
        + inputs + "assert run(left, right) is left\nprint('completed')\n",
        encoding="utf-8",
    )
    trace = project / "trace"
    captured = run_trace(entry, trace, "--call-operands", "--operand-capture", mode,
                         environment={"SCAR_OPERAND_CAPTURE": "invalid-inherited-choice"})
    assert captured.returncode == 0, captured.stdout + captured.stderr
    assert "completed" in captured.stdout and "injector disabled" not in captured.stdout + captured.stderr
    metadata_path, = trace.glob("call-operand-meta.*.json")
    metadata = json.loads(metadata_path.read_text())
    inventory = inventory_program(entry, project_root=project, calls_trace=trace)
    assert inventory.to_dict()["schema_version"] == 6
    return mode, inventory, link_call_operands(trace, project, inventory=inventory), metadata


def test_operand_routes_find_the_upper_formal_without_aliasing_inputs(linked):
    mode, inventory, report, metadata = linked
    assert metadata["rule"] == {"windows": "CPYTHON_CALL_WINDOWS_V1",
                                "instructions": "CPYTHON_LOAD_OCCURRENCES_V1"}[mode]
    assert metadata["dropped_events"] == 0 and not metadata["errors"] and not metadata["io_failed"]
    if mode == "windows":
        assert metadata["events"] < 2_000 and metadata["bytes"] < 1_000_000
    document = report.to_dict()
    replay = CallOperandReport.from_dict(document, inventory=inventory)
    names = {definition["id"]["wire"]: definition["qualname"]
             for file in inventory.to_dict()["files"] for definition in file["definitions"]}
    returned, = [item for item in replay.routing.returns if names[item.definition.wire] == "run"]
    query = replay.trace_reference(returned.invocation)
    assert {(item.invocation, names[item.definition.wire], item.formal_name)
            for item in query.endpoints if item.kind is CoreReferenceKind.FORMAL} == {
                (returned.invocation, "run", "x")}
    completions = [step.completion for step in query.routes if step.kind is OperandQueryRouteKind.COMPLETION]
    assert len(completions) == 2
    assert {names[completion.callee_definition.wire] for completion in completions} == {"relay", "choose"}
    assert query.conditional_only and len(query.conditions) == 8 and query.obligations
    assert not any((query.contents_or_effects_established, query.function_environment_established,
                    query.world_equivalence_established, query.transformation_authorized))
    by_callee = {}
    for route in replay.routing.argument_routes:
        if names[route.callee_definition.wire] == "choose":
            by_callee.setdefault(route.callee, []).append(route)
    assert len(by_callee) == 2
    for routes in by_callee.values():
        assert {route.actual_index for route in routes} == {0, 1}
        assert len({route.actual_use for route in routes}) == 2
        assert len({(route.load_anchor, route.load_suboperand) for route in routes}) == 2
    assert "operand_events" not in document
    assert document["whole_program_complete"] is document["transformation_authorized"] is False


@pytest.mark.parametrize("tamper", ["actual_index", "load_suboperand", "attempt", "condition", "summary"])
def test_replay_rejects_modified_operand_claims(linked, tamper):
    _, inventory, report, _ = linked
    document = report.to_dict()
    route = document["routing"]["argument_routes"][0]
    if tamper in ("actual_index", "load_suboperand"):
        route[tamper] += 1
    elif tamper == "attempt":
        route["attempt"]["index"] += 1
    elif tamper == "condition":
        route["conditions"] = []
    else:
        document["whole_program_complete"] = True
    with pytest.raises(ValueError):
        CallOperandReport.from_dict(document, inventory=inventory)


@pytest.mark.parametrize("budget,field", [("frame_budget", "max_events"), ("operand_budget", "max_events"),
                                          ("route_budget", "max_routes")])
def test_independent_budgets_precede_trace_io(linked, tmp_path, budget, field):
    _, inventory, report, _ = linked
    document = report.to_dict()
    document[budget][field] += 1
    document["trace"] = str(tmp_path / "missing-trace")
    with pytest.raises(ValueError, match="independent verifier budget"):
        CallOperandReport.from_dict(document, inventory=inventory)


def test_default_operand_capture_ignores_inherited_mode(tmp_path):
    entry = tmp_path / "entry.py"
    entry.write_text("def identity(x):\n    return x\nx = []\ny = identity(x)\n", encoding="utf-8")
    trace = tmp_path / "trace"
    captured = run_trace(entry, trace, "--call-operands", environment={"SCAR_OPERAND_CAPTURE": "instructions"})
    assert captured.returncode == 0, captured.stdout + captured.stderr
    path, = trace.glob("call-operand-meta.*.json")
    assert json.loads(path.read_text())["rule"] == "CPYTHON_CALL_WINDOWS_V1"
