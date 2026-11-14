"""Only a fully revalidated CPU domain may reach selected-only compilation."""

from copy import deepcopy
import importlib
import json
from pathlib import Path

import pytest

from test_cpu_choices import setup_choices
from test_cpu_hardware_placement import hardware
from scar.compiler.model import read_plan
from scar.optimization.cpu_choices import plan_cpu_choices
from scar.optimization.resource_solver import semantic_sha256


def test_lower_selected_only_and_reject_tampered_or_incomplete_domains(hardware, tmp_path, monkeypatch):
    module = importlib.import_module("scar.compiler.lower_cpu_choices")
    base, variants = setup_choices(hardware, tmp_path)
    choices = tmp_path / "choices"
    choice = plan_cpu_choices(base, variants, choices, input_shapes=(4, 5))
    assert choice["variants"][0]["prediction"]["locator"] == "#/prediction"
    calls = []

    def selected_lower(plan_path, resource_path, planning_path, out):
        selected = read_plan(plan_path)
        planning = json.loads(Path(planning_path).read_bytes())
        profile = json.loads(Path(planning["cost_profile"]["path"]).read_bytes())
        assert selected.cpu_threads == 2
        assert planning["solver"] == choice["selected_bound_solver_result"]
        assert planning["cost_prediction"]["profile_sha256"] == semantic_sha256(profile)
        assert planning["source_schedule"]["path"] == str(resource_path)
        calls.append(selected.resource_binding.cpu_ids)
        Path(out).mkdir()
        result = {"artifact": {"artifact_directory": str(out), "build_id": "fixture-selected-only"}}
        (Path(out) / "lowering.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(module, "lower_resources", selected_lower)
    lowered = module.lower_cpu_choices(choices / "choice.json", tmp_path / "lowered")
    assert calls == [(0, 1)]
    assert lowered["selected_variant"] == {"id": "cpu-2", "index": 1}
    assert lowered["compiled_variant_count"] == 1 and lowered["selected_total_ticks"] == 48
    assert len(lowered["model_sources"]) == 2 and lowered["all_declared_variants_revalidated"]
    assert lowered["semantic_contract_sha256"] == choice["semantic_contract_sha256"]
    assert not lowered["selected_for_use"] and not lowered["deployment_authorized"]
    assert not lowered["physical_optimality_proven"]
    assert Path(lowered["native_lowering_report"]).is_file()

    choice_path = choices / "choice.json"
    original_choice = choice_path.read_bytes()
    try:
        changed = deepcopy(choice)
        changed["selected_variant"] = {"id": "cpu-1", "index": 0}
        choice_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="sourced replay"):
            module.lower_cpu_choices(choice_path, tmp_path / "tampered")
        assert not (tmp_path / "tampered").exists() and len(calls) == 1
    finally:
        choice_path.write_bytes(original_choice)

    # Even an unselected variant's original source receipt remains mandatory.
    unselected_profile = Path(choice["variants"][0]["profile"]["path"])
    original_profile = unselected_profile.read_bytes()
    try:
        unselected_profile.write_bytes(original_profile + b" ")
        with pytest.raises(ValueError, match="source receipt"):
            module.lower_cpu_choices(choice_path, tmp_path / "stale-domain")
        assert not (tmp_path / "stale-domain").exists() and len(calls) == 1
    finally:
        unselected_profile.write_bytes(original_profile)

    incomplete = tmp_path / "incomplete"
    partial = plan_cpu_choices(base, [variants[0], {**variants[1], "profile": None,
                                                  "frontier_reason": "missing domain cost"}],
                               incomplete, input_shapes=(4, 5))
    assert partial["selected_variant"]["id"] == "cpu-1" and not partial["complete_domain"]
    with pytest.raises(ValueError, match="complete finite implementation domain"):
        module.lower_cpu_choices(incomplete / "choice.json", tmp_path / "partial-lowered")
    assert not (tmp_path / "partial-lowered").exists() and len(calls) == 1
