"""Admit native CUDA service costs for one archived physical placement."""

from dataclasses import dataclass, replace
import json
from pathlib import Path, PurePosixPath

from scar.compiler.calibrate_native_device import replay_native_device_calibration
from scar.compiler.compile_exported_aten_cuda import ACTIVITY_BINDING, PLACEMENT_RECORDS
from scar.compiler.native_device_pair import CUDA_SCHEMA, _identity, _manifest, _member
from scar.ir.implementation_plan import digest, problem_document, problem_from_dict
from scar.ir.v2.schemas import canonical_json
from scar.optimization.joint_planner import verify_execution_plan


@dataclass(frozen=True, slots=True)
class NativeHybridPlacementCosts:
    record: dict
    original_problem: dict
    original_schedule: dict
    bindings: dict
    repriced_problem: dict
    repriced_schedule: dict


def _receipt_status(observation, summary):
    if observation['phase'] != 'COMPLETE':
        return observation['phase']
    if summary['unknown_activities'] or not summary['known_activity_count']:
        return 'INCOMPLETE_NATIVE_ACTIVITY_COSTS'
    if summary['schedule_status'] != 'FEASIBLE_WITHIN_MODEL':
        return 'NO_VERIFIED_NATIVE_SERVICE_SCHEDULE'
    return 'EXACT_OBSERVED_TREATMENT'


def load_native_hybrid_placement_costs(calibration, *, calibration_sha256, cpu_project, cuda_project):
    """Replay locked artifacts and retain their positive or negative observation."""
    calibration = Path(calibration).resolve(strict=True)
    report = replay_native_device_calibration(calibration, calibration_sha256=calibration_sha256,
                                              cpu_project=cpu_project, cuda_project=cuda_project)
    pair = report['pair']
    cuda = Path(cuda_project).resolve(strict=True)
    manifest = _manifest(cuda, pair['cuda']['manifest_sha256'], CUDA_SCHEMA)
    placement_record = PLACEMENT_RECORDS.get(manifest['status'])
    if placement_record is None:
        raise ValueError('empirical placement costs require a compiled typed placement archive')
    proposal = json.loads(_member(cuda, manifest, f'plan/{placement_record}.json'))
    original_problem = json.loads(_member(cuda, manifest, 'plan/problem.json'))
    original_schedule = json.loads(_member(cuda, manifest, 'plan/schedule.json'))
    bindings = json.loads(_member(cuda, manifest, 'plan/bindings.json'))
    observation = json.loads(_member(calibration.parent, report, 'observation.json'))
    repriced_problem = json.loads(_member(calibration.parent, report, 'problem.json'))
    repriced_schedule = json.loads(_member(calibration.parent, report, 'schedule.json'))

    reason = _receipt_status(observation, report['summary'])
    record = dict(calibration_sha256=calibration_sha256,
        calibration_files=report['files'], activity_binding=_identity(manifest['files'][ACTIVITY_BINDING]),
        cpu_manifest_sha256=pair['cpu']['manifest_sha256'], cuda_manifest_sha256=pair['cuda']['manifest_sha256'],
        cpu_profile=pair['cpu_profile'], symbolic_profile=pair['symbolic_profile'],
        source=pair['source'], program=pair['program'], profile_id=original_problem['target']['sha256'],
        placement=proposal['selected_placement'], device_ordinal=bindings['device'],
        inputs=[_identity(row) for row in observation['inputs']],
        references=[_identity(row) for row in observation['references']],
        context=observation['context'], status=observation['phase'], reason=reason,
        cost_available=reason == 'EXACT_OBSERVED_TREATMENT')
    return NativeHybridPlacementCosts(record, original_problem, original_schedule, bindings,
                                      repriced_problem, repriced_schedule)


def _relative(path):
    if type(path) is not str or not path or '\\' in path:
        return False
    parsed = PurePosixPath(path)
    return (not parsed.is_absolute() and parsed.as_posix() == path
            and all(part != '..' for part in parsed.parts) and bool(parsed.parts))


def _anchor(record):
    return record.get('sha256'), record.get('size_bytes')


def _relative_refs(value, paths):
    if isinstance(value, list):
        return [_relative_refs(item, paths) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _relative_refs(item, paths) for key, item in value.items()}
    original_paths = paths.get(_anchor(value))
    if original_paths is not None:
        original_path, expected_path = original_paths
        if 'path' in value and value['path'] == expected_path:
            result['path'] = original_path
        if value.get('uri') == 'declared:' + expected_path:
            result['uri'] = 'declared:' + original_path
    return result


def _reference_paths(original, expected):
    original_program, = original['source_refs']
    expected_program, = expected['source_refs']
    original_target, expected_target = original['target'], expected['target']
    old_uri, new_uri = original_target['uri'], expected_target['uri']
    if not old_uri.startswith('declared:') or not new_uri.startswith('declared:'):
        return None
    anchors = ((original_program, expected_program, original_program['path'], expected_program['path']),
               (original_target, expected_target, old_uri.removeprefix('declared:'), new_uri.removeprefix('declared:')))
    if any(_identity(old) != _identity(new) or not (_relative(old_path) and _relative(new_path))
           for old, new, old_path, new_path in anchors):
        return None
    return {_anchor(old): (old_path, new_path) for old, _, old_path, new_path in anchors}


def _same_schedule(original, expected, original_digest, expected_digest):
    return (original['problem_sha256'] == original_digest and expected['problem_sha256'] == expected_digest
            and canonical_json({key: value for key, value in original.items()
                                if key not in ('elapsed_seconds', 'problem_sha256')})
            == canonical_json({key: value for key, value in expected.items()
                               if key not in ('elapsed_seconds', 'problem_sha256')}))


def native_hybrid_placement_cost_matches(evidence, placement, built, *, profile_id, input_identities,
                                         reference_identities):
    """Match an exact treatment, including a negative no-device receipt."""
    record = evidence.record
    if (placement != record['placement'] or profile_id != record['profile_id']
            or [_identity(row) for row in input_identities] != record['inputs']
            or [_identity(row) for row in reference_identities] != record['references']):
        return False
    expected, expected_schedule, bindings = built
    if canonical_json(bindings) != canonical_json(evidence.bindings):
        return False
    original = evidence.original_problem
    if len(expected['source_refs']) != 1:
        return False
    paths = _reference_paths(original, expected)
    return (paths is not None and canonical_json(original) == canonical_json(_relative_refs(expected, paths))
            and _same_schedule(evidence.original_schedule, expected_schedule,
                               digest(problem_from_dict(original)), digest(problem_from_dict(expected))))


def match_native_hybrid_placement_costs(evidence, placement, built, *, profile_id, input_identities,
                                        reference_identities):
    """Return only measured costs and the corresponding locked lane/wait schedule."""
    if not evidence.record['cost_available'] or not native_hybrid_placement_cost_matches(
            evidence, placement, built, profile_id=profile_id, input_identities=input_identities,
            reference_identities=reference_identities):
        return None
    original = problem_from_dict(built[0])
    measured = problem_from_dict(evidence.repriced_problem)
    operations = tuple(replace(operation, options=(replace(operation.options[0], cost=priced.options[0].cost),))
                       for operation, priced in zip(original.operations, measured.operations, strict=True))
    rebased = replace(original, operations=operations, assumptions=measured.assumptions,
                      timing_constraints=measured.timing_constraints)
    # Only archive anchors changed. Preserve the replayed schedule's timeline.
    schedule = dict(evidence.repriced_schedule, problem_sha256=digest(rebased))
    schedule.pop('elapsed_seconds', None)
    verify_execution_plan(rebased, schedule)
    return problem_document(rebased), schedule, built[2]
