"""Recheck a complete finite CPU choice domain and lower its selected call.

All alternatives are mathematical models of the same region and host storage.
Only the chosen implementation reaches the existing synchronous timed backend;
neither model optimality nor compilation authorizes deployment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from scar.codegen.timed import admit_timed_plan
from scar.compiler.lower_resources import lower_resources
from scar.compiler.model import document, plan_from_dict
from scar.optimization.hardware import validate_cpu_placement
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.schedule import schedule_from_dict


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _read_receipt(receipt, label):
    _require(type(receipt) is dict and type(receipt.get("path")) is str,
             f"retained {label} receipt is required")
    path = Path(receipt["path"])
    _require(path.is_absolute(), f"retained {label} path must be absolute")
    raw = path.read_bytes()
    _require(sha256(raw).hexdigest() == receipt.get("sha256") and
             type(receipt.get("bytes")) is int and len(raw) == receipt["bytes"],
             f"retained {label} raw receipt differs")
    data = json.loads(raw)
    _require(semantic_sha256(data) == receipt.get("semantic_sha256"),
             f"retained {label} semantic receipt differs")
    return data, raw


def _write(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.write_bytes(raw)
    return raw


def lower_cpu_choices(choice_path, out) -> dict:
    """Revalidate every declared option, then compile exactly the selected one."""
    from scar.optimization.cpu_choices import read_cpu_choice_prediction, verify_cpu_choice_report

    choice_path = Path(choice_path).absolute()
    if choice_path.is_dir():
        choice_path /= "choice.json"
    out = Path(out).resolve()
    _require(not out.exists(), "fresh CPU choice lowering output required")
    raw = choice_path.read_bytes()
    original = json.loads(raw)
    _require((original.get("schema"), original.get("schema_version")) ==
             ("scar.cpu-implementation-choice", 1) and type(original["schema_version"]) is int,
             "versioned CPU implementation choice report required")
    # The shared pure verifier reads the independent domain and every original
    # plan/schedule/profile receipt, predicts all costs again, and replays the
    # finite solver. It performs no compilation or target execution.
    checked = verify_cpu_choice_report(choice_path)
    _require(semantic_sha256(checked) == semantic_sha256(original),
             "CPU choice report changed or differs from complete model replay")
    solver = checked["solver"]
    _require(checked.get("complete_domain") is True and
             checked.get("selection_scope") == "CPP_IMPLEMENTATION_FOR_LOWERING" and
             solver.get("status") == "OPTIMAL_WITHIN_FINITE_MODEL" and
             solver.get("optimality") == "FINITE_MODEL_OPTIMUM_ESTABLISHED" and
             solver.get("physical_optimality_proven") is False and
             solver.get("frontier") == [] and solver.get("unresolved") == [] and
             solver.get("execution_unresolved") == [] and
             solver.get("search_budget_exhausted") is False and
             type(solver.get("gap_ticks")) is int and solver["gap_ticks"] == 0,
             "lowering requires a complete finite implementation domain with a proved model optimum")
    base_document, _ = _read_receipt(checked.get("base_plan"), "base plan")
    base = plan_from_dict(base_document)
    domain, domain_raw = _read_receipt(checked.get("domain"), "finite choice domain")
    _require((domain.get("schema"), domain.get("schema_version")) == ("scar.cpu-choice-domain", 1) and
             type(domain["schema_version"]) is int,
             "versioned finite choice domain proof required")
    _require(domain.get("completeness") == "ALL_EXPLICIT_INPUT_VARIANTS_RETAINED",
             "the complete declared finite domain must be retained")
    contract = checked.get("semantic_contract")
    _require(type(contract) is dict and
             checked.get("semantic_contract_sha256") == semantic_sha256(contract) and
             semantic_sha256(domain.get("semantic_contract")) == semantic_sha256(contract),
             "choice and domain semantic contracts differ or are missing")
    variants = checked.get("variants")
    selected = checked.get("selected_variant")
    _require(type(variants) is list and bool(variants) and type(selected) is dict and
             type(selected.get("index")) is int and 0 <= selected["index"] < len(variants),
             "one selected implementation from the complete domain is required")
    identifiers = [variant.get("id") for variant in variants]
    _require(all(type(identifier) is str and identifier for identifier in identifiers) and
             len(set(identifiers)) == len(identifiers) and
             selected.get("id") == identifiers[selected["index"]],
             "selected implementation ID/index does not match the retained domain")

    retained = []
    for index, variant in enumerate(variants):
        _require(variant.get("index") == index and variant.get("family_issues") == [],
                 "unknown or mismatched CPU implementation families cannot be lowered")
        plan_doc, plan_raw = _read_receipt(variant.get("plan"), f"variant {index} plan")
        schedule_doc, schedule_raw = _read_receipt(variant.get("schedule"), f"variant {index} schedule")
        profile, profile_raw = _read_receipt(variant.get("profile"), f"variant {index} cost profile")
        plan = plan_from_dict(plan_doc)
        schedule = schedule_from_dict(schedule_doc)
        _require(semantic_sha256(document(plan)["region"]) == semantic_sha256(document(base)["region"]) and
                 plan.schedule == base.schedule and plan.profile == base.profile and
                 plan.operator_abi_version == base.operator_abi_version and
                 plan.device == base.device and plan.buffer_policy == base.buffer_policy and
                 plan.cost.get("memory_budget_bytes") == base.cost.get("memory_budget_bytes"),
                 "CPU choice changes region work, source, storage, budget or loop semantics")
        binding = validate_cpu_placement(plan.resource_binding)
        prediction = read_cpu_choice_prediction(variant)
        _require(type(prediction) is dict and prediction.get("prediction_available") is True and
                 prediction.get("physical_optimality_proven") is False and
                 semantic_sha256(profile) == prediction.get("profile_sha256"),
                 "every declared implementation needs an applicable sourced cost model")
        solved = variant.get("bound_solver_result")
        admission = admit_timed_plan(plan, schedule, solved)
        durations = {key: value["duration_ticks"] for key, value in prediction["task_costs"].items()}
        _require(durations == solved["model"]["task_durations_ticks"],
                 "implementation solver durations differ from its complete call cost prediction")
        total = sum(durations.values())
        _require(total == solved["solution"]["makespan_ticks"],
                 "CPU choice must account for allocation, compute and export costs")
        retained.append(dict(variant=variant, plan=plan, plan_raw=plan_raw,
                             schedule_raw=schedule_raw, profile_raw=profile_raw, profile=profile,
                             prediction=prediction, admission=admission, total=total, binding=binding))
    chosen = retained[selected["index"]]
    selected_solver = checked.get("selected_bound_solver_result")
    _require(semantic_sha256(selected_solver) == semantic_sha256(chosen["variant"]["bound_solver_result"]) and
             chosen["total"] == min(row["total"] for row in retained) and
             solver.get("global_lower_bound_ticks") == chosen["total"] and
             solver.get("solution", {}).get("makespan_ticks") == chosen["total"],
             "selected implementation does not realize the complete finite-domain optimum")

    # Retain selected inputs so the standard lowering report never references
    # a deleted temporary planning file. No nonselected variant is compiled.
    inputs = out / "selected-inputs"
    inputs.mkdir(parents=True)
    (out / "validated-choice.json").write_bytes(raw)
    (out / "validated-domain.json").write_bytes(domain_raw)
    paths = {key: inputs / filename for key, filename in (
        ("plan", "plan.json"), ("schedule", "resource-plan.json"),
        ("profile", "profile.json"), ("planning", "planning-report.json"))}
    paths["plan"].write_bytes(chosen["plan_raw"])
    paths["schedule"].write_bytes(chosen["schedule_raw"])
    paths["profile"].write_bytes(chosen["profile_raw"])
    planning = {
        "schema": "scar.resource-planning-report", "schema_version": 1,
        "solver": selected_solver,
        "source_schedule": {"path": str(paths["schedule"]),
                            "sha256": sha256(chosen["schedule_raw"]).hexdigest()},
        "cost_prediction": chosen["prediction"],
        "cost_profile": {"path": str(paths["profile"]),
                         "sha256": sha256(chosen["profile_raw"]).hexdigest()},
        "cost_model": None, "deployment_authorized": False,
        "upstream_choice_id": checked["choice_id"],
    }
    _write(paths["planning"], planning)
    lowered = lower_resources(paths["plan"], paths["schedule"], paths["planning"], out / "native")
    result = {
        "schema": "scar.lowered-cpu-implementation-choice", "schema_version": 1,
        "choice_id": checked["choice_id"],
        "upstream_choice": {"path": str(choice_path), "sha256": sha256(raw).hexdigest(),
                            "semantic_sha256": semantic_sha256(checked)},
        "base_plan": checked["base_plan"], "domain": checked["domain"],
        "semantic_contract": contract,
        "semantic_contract_sha256": semantic_sha256(contract),
        "selected_variant": selected, "selected_binding": asdict(chosen["plan"].resource_binding),
        "selected_binding_validation": chosen["binding"],
        "selected_total_ticks": chosen["total"],
        "selected_prediction_status": chosen["prediction"]["status"],
        "tick_seconds": chosen["profile"]["model"]["tick_seconds"],
        "choice_solver_model": {key: solver.get(key) for key in (
            "status", "optimality", "option_domain_sha256", "global_model_sha256",
            "global_lower_bound_ticks", "gap_ticks", "optimality_scope")},
        "model_sources": [{"id": row["variant"]["id"], "plan": row["variant"]["plan"],
                           "schedule": row["variant"]["schedule"], "profile": row["variant"]["profile"],
                           "total_ticks": row["total"],
                           "uncertainty": row["variant"].get("uncertainty")} for row in retained],
        "selected_uncertainty": chosen["variant"].get("uncertainty"),
        "uncertainty_scope": "PROFILE_CALIBRATION_DIAGNOSTICS_NOT_TARGET_MEASUREMENTS_OR_PHYSICAL_PROOFS",
        "selected_model_assumptions": chosen["prediction"]["model_assumptions"],
        "artifact": lowered["artifact"], "native_lowering_report": str(out / "native" / "lowering.json"),
        "selected_inputs": {key: str(path) for key, path in paths.items()},
        "compiled_variant_count": 1, "all_declared_variants_revalidated": True,
        "optimality_scope": "COMPLETE_DECLARED_FINITE_CPU_MODEL_ONLY",
        "physical_optimality_proven": False, "actual_target_duration_seconds": None,
        "target_program_executed": False, "selected_for_use": False,
        "deployment_authorized": False, "whole_program_compiled": False,
        "original_python_no_op": "AVAILABLE_NOT_COSTED_OR_COMPARED",
    }
    _write(out / "lowering.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--choices", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = lower_cpu_choices(args.choices, args.out)
    print(json.dumps({"choice_id": report["choice_id"], "selected_variant": report["selected_variant"],
                      "artifact": report["artifact"]["artifact_directory"], "deployment_authorized": False}))


if __name__ == "__main__":
    main()
