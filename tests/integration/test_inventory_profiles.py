"""Structured and environment inventories replay both source and observed calls."""
import gzip
import json

import pytest

from scar.analysis.program_inventory import ProgramInventory, inventory_program
from scar.analysis.program_lowering import SourceProgram, lower_source
from scar.analysis.source_layers import reconstruct_source_layers
from scar.ir.program.functions import BodyLoweringProfile
from tests.fixtures.capture_cli import run_trace


def _captured_calls(entry):
    trace = entry.parent / "trace"
    captured = run_trace(entry, trace, "--calls-only")
    assert captured.returncode == 0, captured.stdout + captured.stderr
    return trace


def _reconstruct_archive(tmp_path, document):
    path = tmp_path / "inventory.json.gz"
    path.write_bytes(gzip.compress(json.dumps(document).encode(), mtime=0))
    archive = reconstruct_source_layers(path, tmp_path / "source-layers")
    assert archive["input"]["schema_version"] == document["schema_version"]


def test_structured_source_inventory_replays_without_executing(tmp_path):
    marker = tmp_path / "executed"
    entry = tmp_path / "entry.py"
    entry.write_text(
        f"open({str(marker)!r}, 'w').write('executed')\n"
        "def f(items):\n"
        "    result = None\n"
        "    for item in items:\n        result = item\n"
        "    return result\n",
        encoding="utf-8",
    )
    document = inventory_program(entry, project_root=tmp_path,
                                 body_profile=BodyLoweringProfile.STRUCTURED_BODY).to_dict()
    assert document["schema_version"] == 7
    assert document["summary"]["structured"]["loops"] == 1
    assert document["call_sources"] is document["call_source_scope"] is None
    assert ProgramInventory.from_dict(document).to_dict() == document
    _reconstruct_archive(tmp_path, document)
    assert not marker.exists()


def test_actual_call_trace_enters_structured_inventory_and_replays(tmp_path):
    entry = tmp_path / "entry.py"
    entry.write_text(
        "def leaf(left, right):\n    return left\n"
        "def walk(items, shared):\n"
        "    result = shared\n"
        "    for item in items:\n        result = leaf(shared, shared)\n"
        "    return result\n"
        "shared = object()\nassert walk([1, 2], shared) is shared\n",
        encoding="utf-8",
    )
    trace = _captured_calls(entry)
    inventory = inventory_program(entry, project_root=tmp_path, calls_trace=trace,
                                  body_profile=BodyLoweringProfile.STRUCTURED_BODY)
    document = inventory.to_dict()
    assert document["schema_version"] == 7 and document["runtime_scope"] is None
    admissions = document["call_source_scope"]["admissions"]
    assert len(admissions) == 2 and all(item["status"] == "CORE_LINKED" for item in admissions)
    assert len(document["structured"]) == 2
    assert inventory.source_analysis_measurements["source_prepare_max_per_path"] == 1
    assert ProgramInventory.from_dict(document).to_dict() == document
    original = inventory_program(entry, project_root=tmp_path, calls_trace=trace)
    assert original.to_dict()["schema_version"] == 6


def test_environment_inventory_and_source_body_replay_without_executing(tmp_path):
    marker = tmp_path / "executed"
    entry = tmp_path / "entry.py"
    entry.write_text(
        f"open({str(marker)!r}, 'w').write('executed')\n"
        "def factory(shared):\n"
        "    def inner(value):\n        return shared + value\n"
        "    return inner\n"
        "def lookup(value):\n    return abs(value)\n",
        encoding="utf-8",
    )
    profile = BodyLoweringProfile.STRUCTURED_ENV
    document = inventory_program(entry, project_root=tmp_path, body_profile=profile).to_dict()
    assert document["schema_version"] == 8 and document["profile"] == "STRUCTURED_ENV_V1"
    assert "frames" not in document
    assert len(document["structured"]) == 2
    assert ProgramInventory.from_dict(document).to_dict() == document
    model = lower_source(entry, "lookup", profile=profile).to_dict()
    assert model["schema_version"] == 4
    assert SourceProgram.from_dict(model).to_dict() == model
    _reconstruct_archive(tmp_path, document)
    assert not marker.exists()


def test_actual_shared_code_closures_are_recovered_without_claiming_runtime_environment(tmp_path):
    entry = tmp_path / "entry.py"
    entry.write_text(
        "def factory(shared):\n"
        "    def inner(value):\n        return shared + value\n"
        "    return inner\n"
        "left = factory(10)\nright = factory(20)\n"
        "assert left(1) == 11\nassert right(1) == 21\n",
        encoding="utf-8",
    )
    trace = _captured_calls(entry)
    document = inventory_program(entry, project_root=tmp_path, calls_trace=trace,
                                 body_profile=BodyLoweringProfile.STRUCTURED_ENV).to_dict()
    assert document["schema_version"] == 8
    assert len(document["structured"]) == 1
    assert document["summary"]["whole_program_graph_established"] is False
    assert document["transformation_authorized"] is False
    assert document["call_source_scope"]["admissions"]
    assert ProgramInventory.from_dict(document).to_dict() == document


@pytest.mark.parametrize("version,profile", [(7, "STRUCTURED_ENV_V1"), (8, "STRUCTURED_BODY_V1")])
def test_inventory_profile_version_cannot_reinterpret_historical_graphs(tmp_path, version, profile):
    entry = tmp_path / "entry.py"
    entry.write_text("def f(value):\n    return value\n")
    document = inventory_program(entry, project_root=tmp_path,
                                 body_profile=BodyLoweringProfile.STRUCTURED_ENV).to_dict()
    document["schema_version"], document["profile"] = version, profile
    with pytest.raises(ValueError, match="exact structured profile"):
        ProgramInventory.from_dict(document)
