"""Gate a CUDA native trial against the comparable CPU entry and replay its evidence."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from scar.codegen.native_provider_bundle import _content
from scar.compiler.native_device_pair import admit_native_device_pair, replay_native_device_pair
from scar.optimization.native_device_comparison import compare_native_device, native_device_probe_reason
from scar.optimization.paired_entry_selection import DEFAULT_MINIMUM_GAIN


SCHEMA = 'scar.native-device-assessment'
_SCOPE = 'FRESH_NATIVE_PROCESS_WALL_CLOCK_INCLUDING_FILE_IO'
_DIGEST = re.compile(r'[0-9a-f]{64}')


def _content_fields(row):
    return {key: row[key] for key in ('sha256', 'size_bytes')}


def _trial_settings(warmup, count, minimum, *, strict=True):
    if (type(warmup) is not int or not 0 <= warmup <= 100
            or type(count) is not int or not 3 <= count <= 500
            or type(minimum) not in (int, float) or not 0 <= minimum < 1 or type(strict) is not bool):
        raise ValueError('expected 0–100 warmup pairs, 3–500 measured pairs and a finite gain fraction')
    settings = dict(warmup_pairs=warmup, measurement_pairs=count, minimum_gain=minimum)
    if strict:
        settings['strict_minimum_gain'] = True
    return settings


def _paths(named, witness):
    paths = {}
    for item in named:
        name, separator, path = item.partition('=')
        if not separator or not name or not path or name in paths:
            raise ValueError('use --input NAME=FILE exactly once per user input')
        paths[name] = path
    ports = [row['id'] for row in witness['inputs'] if row['kind'] == 'INPUT']
    if set(paths) != set(ports):
        raise ValueError(f'paired inputs must match the user file ports {ports}')
    return [paths[name] for name in ports]


def _locked_files(paths, ports):
    if len(paths) != len(ports):
        raise ValueError('provide one raw file per admitted input or reference port')
    evidence = []
    for raw, port in zip(paths, ports, strict=True):
        path = Path(raw).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f'paired native file must be an ordinary file: {path}')
        lock = dict(path=str(path), **_content(path))
        if lock['size_bytes'] != port['size_bytes']:
            raise ValueError(f'native file byte extent differs from port {port["id"]}')
        evidence.append(lock)
    return evidence


def _check_file_records(records, ports, *, reference=False):
    if type(records) is not list or len(records) != len(ports):
        raise ValueError('paired evidence lacks one file receipt per admitted port')
    identities, repeated = [], {}
    for row, port in zip(records, ports, strict=True):
        if (type(row) is not dict or row.keys() != {'path', 'sha256', 'size_bytes'}
                or type(row['path']) is not str or not Path(row['path']).is_absolute()
                or type(row['sha256']) is not str or _DIGEST.fullmatch(row['sha256']) is None
                or type(row['size_bytes']) is not int or row['size_bytes'] != port['size_bytes']):
            raise ValueError('paired evidence file receipt differs from its admitted port')
        identity = _content_fields(row)
        identities.append(identity)
        if reference and repeated.setdefault(port['id'], identity) != identity:
            raise ValueError('references for repeated output values must carry identical bytes')
    return identities


def _phase(decision):
    if decision['performance_measured']:
        return 'COMPLETE'
    reason, = decision['reasons']
    if reason.endswith('_OUTPUT_MISMATCH'):
        return 'OUTPUT_MISMATCH'
    return {'CUDA_DEVICE_UNAVAILABLE': 'DEVICE_UNAVAILABLE', 'CUDA_PROBE_FAILED': 'PROBE_FAILED',
            'UNQUALIFIED_CUDA_CONTEXT': 'DEVICE_SCOPE_UNVERIFIED'}.get(
                reason, 'INCUMBENT_FAILED' if reason.startswith('CPU_') else 'CANDIDATE_FAILED')


def _package_identities(witness, cpu, cuda):
    return {mode: dict(manifest=dict(sha256=locked['manifest_sha256'],
                                    size_bytes=(root / 'artifact.json').stat().st_size),
                       executable=_content_fields(locked['executable']), launcher=_content_fields(locked['launcher']))
            for mode, root, locked in (('CPU', cpu, witness['cpu']), ('CUDA', cuda, witness['cuda']))}


def _locked_trial_context(witness, observation, cpu, cuda):
    from scar.compiler.compile_exported_aten_cuda import CUDA_ENVIRONMENT
    ports = [row for row in witness['inputs'] if row['kind'] == 'INPUT']
    _check_file_records(observation['inputs'], ports)
    references = _check_file_records(observation['references'], witness['outputs'], reference=True)
    if observation['package_identity'] != _package_identities(witness, cpu, cuda):
        raise ValueError('paired observation names a different manifest, binary or launcher')
    context = observation['context']
    if (context['device_ordinal'] != witness['cuda']['device']
            or sorted(context['benchmark_cpus']) != sorted(witness['cpu']['cpu_ids'])
            or not set(context['benchmark_cpus']).issubset(context['caller_affinity'])
            or not set(context['cuda_environment']).issubset(CUDA_ENVIRONMENT)
            or any(type(value) is not str for value in context['cuda_environment'].values())):
        raise ValueError('paired observation differs from the admitted CPU mask or device ordinal')
    device = context['cuda_identity']
    probe_reason = native_device_probe_reason(observation['probe'], device=context['device_ordinal'])
    if (probe_reason is not None and (device is not None or context['cuda_identity_error'] is not None)
            or probe_reason is None and device is None and not context['cuda_identity_error']):
        raise ValueError('paired device identity does not agree with the recorded probe outcome')
    if device is not None:
        dependencies = json.loads((cuda / 'plan/dependencies.json').read_bytes())
        provider = device['runtime_library']
        if (not any(Path(row['path']).name.startswith('libcudart.so.')
                    and Path(row['path']).resolve() == Path(provider['path']).resolve()
                    and _content_fields(row) == _content_fields(provider) for row in dependencies)
                or context['cuda_identity_error'] is not None
                or re.fullmatch(r'[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]',
                                device['pci_bus_id']) is None
                or any(type(device[key]) is not int or device[key] <= 0
                       for key in ('device_count', 'runtime_version', 'driver_version'))
                or device['device_count'] <= context['device_ordinal']):
            raise ValueError('paired CUDA runtime identity lacks its locked provider or device facts')
    return references, probe_reason, device


def _replay_observation(witness, observation, settings, cpu, cuda):
    from scar.runtime.native_device_probe import SCHEMA as OBSERVATION_SCHEMA

    if (observation['schema'] != OBSERVATION_SCHEMA or observation['schema_version'] != 1
            or any(observation[key] != settings[key] for key in ('warmup_pairs', 'measurement_pairs'))):
        raise ValueError('paired observation differs from the declared measurement protocol')
    references, _, device = _locked_trial_context(witness, observation, cpu, cuda)
    decision = compare_native_device(observation['probe'], observation['runs'], references=references,
        output_shapes=[item['shape'] for item in witness['outputs']], device=witness['cuda']['device'],
        byte_order=witness['abi']['byte_order'], runtime_qualified=device is not None,
        **dict({'strict_minimum_gain': False}, **settings))
    if _phase(decision) != observation['phase']:
        raise ValueError('paired observation phase differs from its replayed probe and attempts')
    return decision


def assess_native_device(cpu_project, cuda_project, *, cpu_manifest_sha256, cuda_manifest_sha256, out,
                         named_inputs, references, cpu_profile=None, warmup_pairs=1,
                         measurement_pairs=4, minimum_gain=DEFAULT_MINIMUM_GAIN):
    from scar.runtime.native_device_probe import probe_native_device_pair

    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    settings = _trial_settings(warmup_pairs, measurement_pairs, minimum_gain)
    cpu, cuda, witness = admit_native_device_pair(cpu_project, cuda_project,
        cpu_manifest_sha256=cpu_manifest_sha256, cuda_manifest_sha256=cuda_manifest_sha256,
        cpu_profile=cpu_profile)
    ports = [row for row in witness['inputs'] if row['kind'] == 'INPUT']
    inputs = _locked_files(_paths(named_inputs, witness), ports)
    references = _locked_files(references, witness['outputs'])
    _check_file_records(references, witness['outputs'], reference=True)
    observation = probe_native_device_pair(cpu, cuda,
        inputs=[row['path'] for row in inputs], references=[row['path'] for row in references],
        cpu_profile=witness['cpu_profile'], warmup_pairs=warmup_pairs, measurement_pairs=measurement_pairs,
        expected_identity=_package_identities(witness, cpu, cuda))
    if observation['inputs'] != inputs or observation['references'] != references:
        raise ValueError('paired inputs or references changed before measurement')
    report = dict(schema=SCHEMA, schema_version=1, timing_scope=_SCOPE, pair=witness, settings=settings, observation=observation,
                  decision=_replay_observation(witness, observation, settings, cpu, cuda))
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-device-assessment-', dir=out.parent) as temporary:
        stage = Path(temporary)
        (stage / 'assessment.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
        stage.rename(out)
    return report


def replay_native_device_assessment(path, *, assessment_sha256, cpu_project, cuda_project):
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != assessment_sha256:
        raise ValueError('native device assessment differs from its pinned digest')
    report = json.loads(raw)
    if (report['schema'] != SCHEMA or type(report['schema_version']) is not int or report['schema_version'] != 1
            or report['timing_scope'] != _SCOPE):
        raise ValueError('unsupported native device assessment')
    cpu, cuda, witness = replay_native_device_pair(report['pair'], cpu_project, cuda_project)
    stored = report['settings']
    settings = _trial_settings(stored['warmup_pairs'], stored['measurement_pairs'], stored['minimum_gain'],
                               strict=stored.get('strict_minimum_gain', False))
    if settings != stored or _replay_observation(witness, report['observation'], settings, cpu, cuda) != report['decision']:
        raise ValueError('native device assessment decision differs from offline replay')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar assess-native-device', description=__doc__)
    parser.add_argument('--cpu-project', required=True)
    parser.add_argument('--cuda-project', required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--out', help='publish a new assessment directory')
    action.add_argument('--check', help='replay an existing assessment.json without timing either program')
    parser.add_argument('--check-sha256')
    parser.add_argument('--cpu-sha256', dest='cpu_manifest_sha256')
    parser.add_argument('--cuda-sha256', dest='cuda_manifest_sha256')
    parser.add_argument('--cpu-profile')
    parser.add_argument('--input', action='append', default=[], dest='named_inputs', metavar='NAME=FILE')
    parser.add_argument('--reference', action='append', default=[], dest='references')
    parser.add_argument('--warmup-pairs', type=int, default=1)
    parser.add_argument('--measurement-pairs', type=int, default=4)
    parser.add_argument('--minimum-gain', type=float, default=DEFAULT_MINIMUM_GAIN)
    args = parser.parse_args(argv)
    if args.check:
        if not args.check_sha256:
            parser.error('--check requires --check-sha256')
        report = replay_native_device_assessment(args.check, assessment_sha256=args.check_sha256,
            cpu_project=args.cpu_project, cuda_project=args.cuda_project)
    else:
        if not args.cpu_manifest_sha256 or not args.cuda_manifest_sha256:
            parser.error('--out requires --cpu-sha256 and --cuda-sha256')
        options = vars(args).copy()
        del options['check'], options['check_sha256']
        report = assess_native_device(**options)
    print(json.dumps(report['decision'], sort_keys=True))
    return 0
