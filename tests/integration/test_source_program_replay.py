"""Replay frame and structured source models against the original source."""
from copy import deepcopy

import pytest

from scar.analysis.program_lowering import SourceProgram, lower_source
from scar.ir.program.functions import BodyLoweringProfile


def source(tmp_path, body=None):
    path = tmp_path / "program.py"
    path.write_text(body or "def f(x):\n    if x:\n        y = 1\n    else:\n        y = 2\n    return y\n")
    return path


def test_automatic_source_lowering_replays_without_execution(tmp_path):
    marker = tmp_path / "executed"
    path = source(tmp_path, f"open({str(marker)!r}, 'w').write('executed')\ndef f(x):\n    return x + x\n")
    before = path.read_bytes()
    document = lower_source(path, "f").to_dict()
    summary = document["summary"]
    assert document["status"] == "LOWERED_WITH_GAPS"
    assert summary["counts"]["use_source_links"] == summary["counts"]["uses"]
    assert summary["whole_program_coverage_established"] is False
    assert summary["runtime_target_verified"] is False
    replay = SourceProgram.from_dict(document)
    assert replay.graph is not None
    assert replay.to_dict() == document
    assert document["transformation_authorized"] is False
    assert path.read_bytes() == before
    assert not marker.exists()


def test_unsupported_body_is_a_located_report_not_a_partial_complete_graph(tmp_path):
    path = source(tmp_path, "def f(x):\n    for y in x:\n        y()\n    return x\n")
    document = lower_source(path, "f").to_dict()
    assert document["status"] == "UNSUPPORTED"
    assert document["graph"] is None
    assert not document["metadata"]["operation_links"]
    assert not document["metadata"]["value_links"]
    assert not document["metadata"]["use_links"]
    assert any(gap["kind"] == "UNSUPPORTED_SYNTAX" for gap in document["metadata"]["gaps"])
    assert SourceProgram.from_dict(document).summary()["graph_present"] is False


def test_replay_rejects_source_drift_and_uses_independent_budget(tmp_path):
    path = source(tmp_path)
    document = lower_source(path, "f").to_dict()
    with pytest.raises(ValueError):
        SourceProgram.from_dict(document, max_operations=1)
    path.write_text(path.read_text().replace("y = 2", "y = 3"))
    with pytest.raises(ValueError):
        SourceProgram.from_dict(document)


@pytest.mark.parametrize("field,value", [
    ("transformation_authorized", True),
    ("runtime_target_verified", True),
    ("schema_version", True),
    ("status", "COMPLETE"),
    ("scope", "whole program"),
    ("extra", "ignored"),
])
def test_persistence_cannot_forge_authority_or_scope(tmp_path, field, value):
    document = lower_source(source(tmp_path), "f").to_dict()
    document[field] = value
    with pytest.raises(ValueError):
        SourceProgram.from_dict(document)


@pytest.mark.parametrize("section", ["metadata", "graph", "semantic", "semantics", "control", "summary"])
def test_all_graph_layers_and_correspondences_replay(tmp_path, section):
    document = lower_source(source(tmp_path), "f").to_dict()
    damaged = deepcopy(document)
    if section == "metadata":
        damaged[section]["operation_links"].pop()
    elif section == "graph":
        damaged[section]["operations"][0]["label"] = "forged"
    elif section == "summary":
        damaged[section]["counts"]["operations"] = 0
    else:
        damaged[section]["extra"] = "must not be discarded"
    with pytest.raises((ValueError, TypeError)):
        SourceProgram.from_dict(damaged)


def test_candidate_parameter_binding_tamper_is_rejected(tmp_path):
    path = source(tmp_path, "def helper(a, b):\n    return a + b\ndef f(x, y):\n    return helper(x, y)\n")
    document = lower_source(path, "f").to_dict()
    target = document["metadata"]["call_targets"][0]
    assert len(target["arguments"]) == 3
    first, second = target["arguments"][:2]
    first["parameter"], second["parameter"] = second["parameter"], first["parameter"]
    with pytest.raises(ValueError):
        SourceProgram.from_dict(document)


def test_structured_loop_model_replays_without_importing_target(tmp_path):
    marker = tmp_path / "must-not-execute"
    path = source(tmp_path, f"open({str(marker)!r}, 'w').write('executed')\n"
                  "def f(items, callback):\n"
                  "    result = None\n"
                  "    for item in items:\n"
                  "        result = callback(item, item)\n"
                  "    return result\n")
    original = path.read_bytes()
    document = lower_source(path, "f", profile=BodyLoweringProfile.STRUCTURED_BODY).to_dict()
    assert document["schema_version"] == 3
    assert document["profile"] == "STRUCTURED_BODY_V1"
    assert document["graph"]["schema_version"] == 3
    assert document["summary"]["counts"]["loops"] == 1
    assert document["summary"]["counts"]["binding_reads"] > 0
    assert document["transformation_authorized"] is False
    assert SourceProgram.from_dict(document).to_dict() == document
    assert not marker.exists() and path.read_bytes() == original


def test_structured_replay_rejects_missing_protocol_and_source_drift(tmp_path):
    path = source(tmp_path, "def f(flag):\n    while flag():\n        pass\n    return None\n")
    document = lower_source(path, "f", profile=BodyLoweringProfile.STRUCTURED_BODY).to_dict()
    altered = deepcopy(document)
    altered["structured"]["protocols"] = []
    with pytest.raises(ValueError):
        SourceProgram.from_dict(altered)
    path.write_text("def f(flag):\n    return 7\n")
    with pytest.raises(ValueError):
        SourceProgram.from_dict(document)


def test_unsupported_suspension_stays_explicit_in_structured_model(tmp_path):
    path = source(tmp_path, "def f(items):\n    for item in items:\n        yield item\n")
    document = lower_source(path, "f", profile=BodyLoweringProfile.STRUCTURED_BODY).to_dict()
    assert document["graph"] is None and document["structured"] is None
    assert any(gap["kind"] == "UNSUPPORTED_SIGNATURE" for gap in document["metadata"]["gaps"])
    assert document["summary"]["whole_program_coverage_established"] is False
    assert SourceProgram.from_dict(document).to_dict() == document
