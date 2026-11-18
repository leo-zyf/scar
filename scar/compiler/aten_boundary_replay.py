"""Cold replay a measured standalone ATen boundary before C++ delivery."""

from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re

from scar.compiler.plan_aten_boundary import (
    OBSERVATIONS, _admit_observation, _build_boundary_problem, _published_selection, _require,
)
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import PLAN_SCHEMA, SCHEMA as PROBLEM_SCHEMA, problem_document
from scar.optimization.aten_boundary_trials import EVIDENCE, project_aten_boundary_trial
from scar.optimization.implementation_costs import SCHEMA as COST_PROFILE
from scar.optimization.joint_planner import verify_execution_plan

SELECTION_SCHEMA = 'scar.aten-boundary-trial-selection'


def _member(directory, filename, schema):
    path = directory / filename
    _require(path.resolve(strict=True) == path and path.is_file(),
             'expected original ATen plan member: ' + filename)
    return artifact_reference(path, schema)


def _verify_document(reference, document):
    expected = (json.dumps(document, sort_keys=True, allow_nan=False, indent=2) + '\n').encode()
    _require(len(expected) == reference['size_bytes'] and sha256(expected).hexdigest() == reference['sha256'],
             'ATen plan member differs from cold replay: ' + Path(reference['uri']).name)


def replay_aten_boundary(directory, *, expected_selection_sha256=None):
    """Rebuild the cost selection and plan from the exact original trial members."""
    directory = Path(directory).resolve(strict=True)
    receipts = {
        'selection': _member(directory, 'selection.json', SELECTION_SCHEMA),
        'evidence': _member(directory, 'evidence.json', EVIDENCE),
        'cost_profile': _member(directory, 'cost-profile.json', COST_PROFILE),
        'problem': _member(directory, 'problem.json', PROBLEM_SCHEMA),
        'plan': _member(directory, 'plan.json', PLAN_SCHEMA),
    }
    selection_ref = receipts['selection']
    if expected_selection_sha256 is not None:
        _require(selection_ref['sha256'] == expected_selection_sha256,
                 'ATen selection digest differs from the expected receipt')
    saved_selection = read_artifact(selection_ref)
    evidence_ref = receipts['evidence']
    saved_evidence = read_artifact(evidence_ref)
    profile_ref = receipts['cost_profile']
    source = saved_evidence['context']['original_observation']
    raw_ref = saved_selection['original_observation']
    _require(source['artifact'] == raw_ref and raw_ref.get('schema') == OBSERVATIONS
             and saved_selection['original_evidence'] == evidence_ref
             and saved_selection['cost_profile'] == profile_ref,
             'ATen measurement or local plan receipt differs')
    observations = read_artifact(raw_ref)
    case_locator = re.fullmatch(r'#/cases/(0|[1-9][0-9]*)', source['locator'])
    _require(case_locator is not None and int(case_locator[1]) < len(observations['cases']),
             'ATen evidence does not name an original measured case')
    case_id = observations['cases'][int(case_locator[1])]['id']
    hardware = Path(observations['context']['target_graph']['path']).parent
    target, cpu_resources, modes, evidence = _admit_observation(raw_ref, hardware, case_id)
    _verify_document(evidence_ref, evidence)
    selection, profile, queries = project_aten_boundary_trial(
        evidence_ref, expected_context=evidence['context'])
    _verify_document(profile_ref, profile)
    selection = _published_selection(selection, evidence_ref, profile_ref, raw_ref, directory)
    _verify_document(selection_ref, selection)
    problem, plan = _build_boundary_problem(target, cpu_resources, modes, evidence, profile_ref,
                                             selection, queries, directory)
    _verify_document(receipts['problem'], problem_document(problem))
    saved_plan = read_artifact(receipts['plan'])
    # Recomputing the plan changes the planner's own wall time.
    elapsed = saved_plan.get('elapsed_seconds')
    _require(type(elapsed) is float and isfinite(elapsed) and elapsed >= 0,
             'invalid elapsed time in retained ATen plan')
    plan['elapsed_seconds'] = elapsed
    _verify_document(receipts['plan'], plan)
    verify_execution_plan(problem, saved_plan)
    receipts['observations'] = raw_ref
    return dict(selection=selection, evidence=evidence, problem=problem, plan=saved_plan,
                case_id=case_id, receipts=receipts)
