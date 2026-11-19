"""Observe instrumented hybrid activity costs without changing the native entry choice."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scar.codegen.native_provider_bundle import _content
from scar.compiler.assess_native_device import (
    _check_file_records, _locked_files, _locked_trial_context, _package_identities, _paths,
)
from scar.compiler.compile_exported_aten import _write_record
from scar.compiler.compile_exported_aten_cuda import ACTIVITY_BINDING
from scar.compiler.native_device_pair import admit_native_device_pair, replay_native_device_pair
from scar.ir.implementation_plan import problem_from_dict
from scar.runtime.native_device_activity_probe import (
    SCHEMA as OBSERVATION_SCHEMA, decode_activity_attempt, observe_device_activities,
)
from scar.runtime.native_device_probe import _locked_json


SCHEMA = 'scar.native-device-activity-calibration'
_FILES = ('observation.json', 'problem.json', 'schedule.json')


def _settings(warmup, count):
    if type(warmup) is not int or not 0 <= warmup <= 100 or type(count) is not int or not 3 <= count <= 100:
        raise ValueError('expected 0–100 activity warmup runs and 3–100 measurement runs')
    return dict(warmup_runs=warmup, measurement_runs=count)


def _archive_inputs(cuda):
    manifest = json.loads((cuda / 'artifact.json').read_bytes())
    if manifest.get('activity_calibration_abi') != 1:
        raise ValueError('rebuild the hybrid candidate with --with-activity-calibration')
    binding = _locked_json(cuda, manifest, ACTIVITY_BINDING)
    problem = problem_from_dict(_locked_json(cuda, manifest, 'plan/problem.json'))
    return binding, dict(path=ACTIVITY_BINDING, **manifest['files'][ACTIVITY_BINDING]), problem


def _derive(cpu, cuda, witness, observation, observation_ref, settings):
    from scar.optimization.native_device_activity_costs import reprice_native_device_activities

    binding, binding_ref, problem = _archive_inputs(cuda)
    if (observation['schema'] != OBSERVATION_SCHEMA or observation['schema_version'] != 1
            or any(observation[name] != value for name, value in settings.items())):
        raise ValueError('activity observation differs from the measurement protocol')
    references, reason, device = _locked_trial_context(witness, observation, cpu, cuda)
    if reason == 'CUDA_DEVICE_UNAVAILABLE':
        phase = 'DEVICE_UNAVAILABLE'
    elif reason is not None:
        phase = 'PROBE_FAILED'
    elif device is None:
        phase = 'DEVICE_SCOPE_UNVERIFIED'
    else:
        phase = 'COMPLETE' if observation['failure'] is None else 'CALIBRATION_FAILED'
    if observation['phase'] != phase:
        raise ValueError('activity observation phase differs from its recorded preflight and attempts')
    if phase == 'CALIBRATION_FAILED':
        failure = observation['failure']
        attempt = failure['attempt']
        _, observed_reason = decode_activity_attempt(attempt, references, binding['activities'])
        if (failure['ordinal'] != len(observation['runs']) or attempt['pair'] != failure['ordinal']
                or attempt['mode'] != 'CUDA' or observed_reason is None or observed_reason != failure['reason']):
            raise ValueError('activity failure does not identify its attempted execution')
    elif observation['failure'] is not None:
        raise ValueError('activity failure is outside the sampled execution phase')
    return reprice_native_device_activities(problem, binding, observation,
                                           binding_ref=binding_ref, observation_ref=observation_ref)


def calibrate_native_device(cpu_project, cuda_project, *, cpu_manifest_sha256, cuda_manifest_sha256,
                            out, named_inputs, references, cpu_profile=None, warmup_runs=1, measurement_runs=3):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    settings = _settings(warmup_runs, measurement_runs)
    cpu, cuda, witness = admit_native_device_pair(cpu_project, cuda_project, cpu_manifest_sha256=cpu_manifest_sha256,
        cuda_manifest_sha256=cuda_manifest_sha256, cpu_profile=cpu_profile)
    inputs = _locked_files(_paths(named_inputs, witness), [row for row in witness['inputs'] if row['kind'] == 'INPUT'])
    references = _locked_files(references, witness['outputs'])
    _check_file_records(references, witness['outputs'], reference=True)
    _, binding_ref, _ = _archive_inputs(cuda)
    observation = observe_device_activities(cpu, cuda, witness=witness, binding_ref=binding_ref, inputs=inputs,
        references=references, expected_identity=_package_identities(witness, cpu, cuda), **settings)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-device-calibration-', dir=out.parent) as temporary:
        stage = Path(temporary)
        _write_record(stage / 'observation.json', observation)
        observation_ref = dict(path='observation.json', **_content(stage / 'observation.json'))
        problem, schedule, summary = _derive(cpu, cuda, witness, observation, observation_ref, settings)
        _write_record(stage / 'problem.json', problem)
        _write_record(stage / 'schedule.json', schedule)
        report = dict(schema=SCHEMA, schema_version=1, pair=witness, settings=settings, summary=summary,
                      deployment_authorized=False, files={name: _content(stage / name) for name in _FILES})
        _write_record(stage / 'calibration.json', report)
        stage.rename(out)
    return report


def replay_native_device_calibration(path, *, calibration_sha256, cpu_project, cuda_project):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != calibration_sha256:
        raise ValueError('activity calibration differs from its pinned digest')
    report = json.loads(raw)
    if (report['schema'] != SCHEMA or report['schema_version'] != 1
            or report['deployment_authorized'] is not False or set(report['files']) != set(_FILES)):
        raise ValueError('unsupported native device calibration')
    files = report['files']
    documents = {}
    for name in _FILES:
        member = path.with_name(name)
        if member.resolve(strict=True) != member or _content(member) != files[name]:
            raise ValueError(f'activity calibration member differs from the accepted artifact: {name}')
        documents[name] = json.loads(member.read_bytes())
    cpu, cuda, witness = replay_native_device_pair(report['pair'], cpu_project, cuda_project)
    stored = report['settings']
    settings = _settings(stored['warmup_runs'], stored['measurement_runs'])
    problem, schedule, summary = _derive(cpu, cuda, witness, documents['observation.json'],
        dict(path='observation.json', **files['observation.json']), settings)
    schedule.pop('elapsed_seconds', None)
    documents['schedule.json'].pop('elapsed_seconds', None)
    if (settings != stored or summary != report['summary'] or problem != documents['problem.json']
            or schedule != documents['schedule.json']):
        raise ValueError('activity calibration cost or schedule differs from its replayed evidence')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar calibrate-native-device', description=__doc__)
    parser.add_argument('--cpu-project', required=True)
    parser.add_argument('--cuda-project', required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--out')
    action.add_argument('--check', help='replay calibration.json without running any calibration')
    parser.add_argument('--check-sha256')
    parser.add_argument('--cpu-sha256', dest='cpu_manifest_sha256')
    parser.add_argument('--cuda-sha256', dest='cuda_manifest_sha256')
    parser.add_argument('--cpu-profile')
    parser.add_argument('--input', action='append', default=[], dest='named_inputs', metavar='NAME=FILE')
    parser.add_argument('--reference', action='append', default=[], dest='references')
    parser.add_argument('--warmup-runs', type=int, default=1)
    parser.add_argument('--measurement-runs', type=int, default=3)
    args = parser.parse_args(argv)
    if args.check:
        if not args.check_sha256:
            parser.error('--check requires --check-sha256')
        result = replay_native_device_calibration(args.check, calibration_sha256=args.check_sha256,
            cpu_project=args.cpu_project, cuda_project=args.cuda_project)
    else:
        if not args.cpu_manifest_sha256 or not args.cuda_manifest_sha256:
            parser.error('--out requires --cpu-sha256 and --cuda-sha256')
        options = vars(args).copy()
        del options['check'], options['check_sha256']
        result = calibrate_native_device(**options)
    print(json.dumps(result['summary'], sort_keys=True))
    return 0
