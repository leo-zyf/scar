"""Connect captured call sites and replay their evidence with independent budgets."""
from dataclasses import replace

import pytest

from scar.analysis.call_edges import CallEdgeReport, link_calls
from scar.analysis.program_inventory import inventory_program
from scar.ir.program.call_edges import CallEdgeBudget, CallEdgeIssueKind, CallLinkStatus, ReferenceRole
from tests.fixtures.capture_cli import run_trace


@pytest.fixture(scope="module")
def capture(tmp_path_factory):
    project = tmp_path_factory.mktemp("call-link-project")
    (project / "helper.py").write_text(
        "def identity(value):\n    return value\n"
        "def relay(value):\n    return identity(value)\n",
        encoding="utf-8",
    )
    entry = project / "entry.py"
    entry.write_text(
        "from helper import relay\n"
        "class Token: pass\n"
        "def run(left, right):\n    a = relay(left)\n    b = relay(right)\n    return a\n"
        "left = Token()\nassert run(left, Token()) is left\nprint('completed')\n",
        encoding="utf-8",
    )
    trace = project / "trace"
    result = run_trace(entry, trace, "--calls-only")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "completed" in result.stdout
    return entry, trace, inventory_program(entry, project_root=project)


def connect(capture, budget=CallEdgeBudget()):
    entry, trace, inventory = capture
    return link_calls(trace, entry.parent, inventory=inventory, budget=budget)


def trace_run_result(report):
    names = {capture.code.key: capture.code.qualname for capture in report.model.codes}
    run, = [edge for edge in report.model.edges if names[edge.entry.code] == "run"]
    return report.trace_reference(run.entry.invocation)


def test_captured_source_connections_and_identity_only_routes(capture):
    entry, trace, inventory = capture
    assert not (trace / "events.jsonl").exists()
    assert not (trace / "entities.jsonl").exists()
    budget = CallEdgeBudget(16_000_000, 100_000, 100_000, 2, 4096, 50_000,
                            32, 4_000_000, 250_000, 64)
    report = connect(capture, budget)
    document = report.to_dict()
    assert report.model.budget == budget
    assert document["summary"]["direct_core_call_observations"] >= 4
    assert document["summary"]["reference_routes"] > 0
    assert document["transformation_authorized"] is False
    replay = CallEdgeReport.from_dict(document, inventory=inventory, verification_budget=budget)
    query = trace_run_result(replay)
    endpoints = [point for route in query.routes for point in (route.destination, route.source)]
    assert all(point.owner != query.invocation or point.role is not ReferenceRole.PARAMETER
               for point in endpoints)
    assert len({point.owner for point in endpoints if point.owner}) >= 3
    assert CallEdgeIssueKind.ARGUMENT_ROUTING_UNVERIFIED in {issue.kind for issue in query.issues}
    assert query.identity_only and not query.content_or_effect_equivalence_established
    with pytest.raises(ValueError, match="independent verifier budget: max_references"):
        CallEdgeReport.from_dict(document, inventory=inventory,
                                 verification_budget=replace(budget, max_references=249_999))
    repeated = run_trace(entry, trace, "--calls-only")
    assert repeated.returncode != 0 and "fresh" in repeated.stderr


@pytest.mark.parametrize("tamper", ["position", "condition", "inventory", "summary"])
def test_replay_rejects_modified_claims(capture, tamper):
    document = connect(capture).to_dict()
    model = document["model"]
    if tamper == "position":
        next(edge for edge in model["edges"] if edge["path"])["path"][0]["site"]["offset"] += 2
    elif tamper == "condition":
        model["edges"][0]["conditions"] = []
    elif tamper == "inventory":
        model["inventory_sha256"] = "0" * 64
    else:
        document["summary"]["contents_or_effects_established"] = True
    with pytest.raises(ValueError):
        CallEdgeReport.from_dict(document, inventory=capture[2])


@pytest.mark.parametrize("field", ["max_raw_bytes", "max_lines", "max_line_bytes", "max_events",
                                  "max_codes", "max_references", "max_path"])
def test_low_trace_reference_and_path_caps_fail_independently(capture, field):
    with pytest.raises(ValueError, match="budget"):
        connect(capture, CallEdgeBudget(**{field: 1}))


@pytest.mark.parametrize("field,detail", [("max_source_files", "source file budget exceeded"),
                                         ("max_source_bytes", "source byte budget exceeded")])
def test_low_source_caps_leave_explicit_source_boundaries(capture, field, detail):
    report = connect(capture, CallEdgeBudget(**{field: 1}))
    bounded = [link for link in report.model.code_links
               if any(issue.kind is CallEdgeIssueKind.SOURCE_BOUNDARY and issue.detail == detail
                      for issue in link.issues)]
    assert bounded and all(link.status is CallLinkStatus.INCOMPLETE and link.definition is None
                           and link.source is None for link in bounded)
    sources = {link.snapshot.path for link in report.model.code_links if link.snapshot is not None}
    if field == "max_source_files":
        assert len(sources) <= 1
    else:
        assert not sources
        assert report.summary()["verified_source_call_sites"] == 0
        assert report.summary()["target_body_observations"] == 0
    document = report.to_dict()
    assert document["transformation_authorized"] is False
    assert document["function_environment_established"] is False
    assert document["contents_or_effects_established"] is False
    assert CallEdgeReport.from_dict(document, inventory=capture[2]).to_dict() == document


def test_process_cap_applies_across_real_capture_processes(capture, tmp_path):
    entry, first, inventory = capture
    second = tmp_path / "second-trace"
    captured = run_trace(entry, second, "--calls-only")
    assert captured.returncode == 0, captured.stdout + captured.stderr
    combined = tmp_path / "combined-trace"
    combined.mkdir()
    for directory in (first, second):
        for path in directory.glob("call-*"):
            destination = combined / path.name
            assert not destination.exists()
            destination.hardlink_to(path)
    assert len(list(combined.glob("call-meta.*.json"))) == 2
    with pytest.raises(ValueError, match="process budget exceeded"):
        link_calls(combined, entry.parent, inventory=inventory, budget=CallEdgeBudget(max_processes=1))


@pytest.mark.parametrize("ceiling", [0, -1])
def test_nonpositive_budget_is_invalid(ceiling):
    with pytest.raises(ValueError, match="exact integer"):
        CallEdgeBudget(max_processes=ceiling)
