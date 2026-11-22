"""Compare qualified ATen candidates at one original Python entry boundary."""

import argparse
import json
import os
from pathlib import Path
import stat
from tempfile import TemporaryDirectory

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.record_codec import loads
from scar.optimization.aten_entry_candidates import entry_round_orders, select_aten_entry_candidate
from scar.runtime.aten_python_trial import _read_manifest, _trial_layout, _verify
from scar.runtime.validate_aten_python_entry import _file_receipt, _output_tree, _PLACEHOLDER, _run_entry


SCHEMA = 'scar.aten-entry-candidate-validation'
_TRIAL_SCHEMA = 'scar.aten-python-trial'
_SCOPE = 'FRESH_PROCESS_ORIGINAL_AND_GUARDED_ATEN_CANDIDATES'
_TIMING_ENVIRONMENT = dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
_RUN_ANNOTATIONS = {'dir', 'entry_record', 'process_ns', 'exit_status', 'stdout', 'stderr', 'output_tree'}


def _common_context(manifest):
    provider = manifest['provider']
    libraries = sorted((row['path'], row['size_bytes'], row['sha256']) for row in provider['libraries'])
    return dict(cpus=_trial_layout(manifest)[0], source=manifest['source_proof'],
                recipe=manifest['recipe'], contract=manifest['contract'], guard=manifest.get('input_guard'),
                parameter_bytes=[(row['size_bytes'], row['sha256']) for row in manifest['inputs']],
                provider=(provider['name'], provider['version'], provider['revision'], libraries))


def _trial_context(trials, candidates):
    trial_refs, context = {}, None
    for name in candidates:
        trial, digest = trials[name]
        path = Path(trial).resolve(strict=True)
        if path.is_dir():
            path = path / 'trial.json'
        if path.name != 'trial.json':
            raise ValueError('an ATen trial directory or its trial.json is required')
        manifest = _read_manifest(path.parent, digest)
        ref = artifact_reference(path, _TRIAL_SCHEMA)
        if ref['sha256'] != digest:
            raise ValueError('ATen trial manifest changed during candidate admission')
        current = _common_context(manifest)
        if context is None:
            context = current
        elif current != context:
            raise ValueError('ATen candidates require identical source, inputs, guards, provider and ordered CPUs')
        extension = manifest['extension']
        if Path(extension['name']).name != extension['name']:
            raise ValueError('ATen trial extension must be a package member')
        _verify(path.parent / extension['name'], extension)
        trial_refs[name] = ref
    source = context['source']['source']
    _verify(Path(source['path']), source)
    return list(context['cpus']), trial_refs


def _entry_request(function, args, ambient, expected_calls):
    if type(function) is not str or not function.isidentifier():
        raise ValueError('a named Python application entry is required')
    if (type(args) not in (list, tuple) or not all(type(value) is str for value in args)
            or not any(_PLACEHOLDER in value for value in args)):
        raise ValueError('an entry argument must contain {output_dir} to isolate candidate outputs')
    if type(ambient) is not bool:
        raise ValueError('ambient GradMode permission must be explicit')
    if expected_calls is not None and (type(expected_calls) is not int or expected_calls < 1):
        raise ValueError('the expected native count per measured process must be positive')


def _run_location(index, name):
    mode = 'ORIGINAL' if name == 'ORIGINAL' else 'GUARDED'
    return mode, Path(f'round-{index:03d}') / name.lower() / mode.lower()


def _matching_trial(name, candidates, trial_refs):
    return trial_refs[candidates[0] if name == 'ORIGINAL' else name]


def validate_aten_entry_candidates(trials, *, entry_function, entry_args, out,
                                   warmup_rounds=1, measured_rounds=4, allow_ambient_grad=False,
                                   expected_native_calls=None):
    """Run one original and every admitted candidate per counterbalanced round."""
    candidates = list(trials)
    orders = entry_round_orders(candidates, warmup_rounds=warmup_rounds, measured_rounds=measured_rounds)
    _entry_request(entry_function, entry_args, allow_ambient_grad, expected_native_calls)
    entry_args = list(entry_args)
    cpus, trial_refs = _trial_context(trials, candidates)
    if not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError('candidate CPUs are outside the current allowed affinity')
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **_TIMING_ENVIRONMENT)

    rounds = []
    with TemporaryDirectory(prefix='.scar-aten-entry-candidates-', dir=out.parent) as temporary:
        stage = Path(temporary)
        for index, order in enumerate(orders):
            runs = {}
            for name in order:
                mode, relative = _run_location(index, name)
                trial = _matching_trial(name, candidates, trial_refs)
                directory = stage / relative
                run = _run_entry(directory.parent, mode, Path(trial['uri']).parent, trial['sha256'],
                                 entry_function, entry_args, cpus, env, allow_ambient_grad)
                runs[name] = dict(run, dir=relative.as_posix(), entry_record=_file_receipt(directory / 'entry.json'))
            rounds.append(dict(round=index, warmup=index < warmup_rounds, order=order, runs=runs))

        selection = select_aten_entry_candidate(rounds, candidates, warmup_rounds=warmup_rounds,
            measured_rounds=measured_rounds, expected_native_calls=expected_native_calls)
        report = dict(schema=SCHEMA, schema_version=1, scope=_SCOPE,
            trials=trial_refs, candidate_order=candidates, cpu_ids=cpus, entry_function=entry_function,
            entry_args=entry_args, allow_ambient_grad=allow_ambient_grad, expected_native_calls=expected_native_calls,
            warmup_rounds=warmup_rounds, measured_rounds=measured_rounds, rounds=rounds,
            timing_environment=_TIMING_ENVIRONMENT, execution_root=str(stage), **selection)
        (stage / 'validation.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
        stage.rename(out)
    return report


def _verify_run(out, report, index, name, run):
    mode, relative = _run_location(index, name)
    trial = _matching_trial(name, report['candidate_order'], report['trials'])
    directory = out / relative
    original_output = Path(report['execution_root']) / relative / 'output'
    args = [value.replace(_PLACEHOLDER, str(original_output)) for value in report['entry_args']]
    if (run['dir'] != relative.as_posix() or run['mode'] != mode or
            run['trial_sha256'] != trial['sha256'] or run['entry_args'] != args or
            run['entry_function'] != report['entry_function'] or run['cpu_ids'] != sorted(report['cpu_ids'])):
        raise ValueError('a retained entry run ran outside its recorded invocation')
    raw = directory / 'entry.json'
    output = directory / 'output'
    if (_file_receipt(raw) != run['entry_record'] or
            loads(raw.read_bytes()) != {key: value for key, value in run.items() if key not in _RUN_ANNOTATIONS} or
            _file_receipt(directory / 'stdout') != run['stdout'] or
            _file_receipt(directory / 'stderr') != run['stderr'] or
            not stat.S_ISDIR(output.lstat().st_mode) or _output_tree(output) != run['output_tree']):
        raise ValueError('retained entry run data differs from its validation receipt')


def read_entry_candidate_validation(path, digest):
    """Replay pinned trial identity, retained run data and selection without running an entry."""
    path = Path(path).resolve(strict=True)
    ref = artifact_reference(path, SCHEMA)
    if ref['sha256'] != digest:
        raise ValueError('entry candidate validation differs from its declared digest')
    report = read_artifact(ref)
    if (type(report['schema_version']) is not int or report['schema_version'] != 1 or report['scope'] != _SCOPE
            or report['timing_environment'] != _TIMING_ENVIRONMENT or not Path(report['execution_root']).is_absolute()):
        raise ValueError('unsupported ATen entry candidate validation context')
    _entry_request(report['entry_function'], report['entry_args'], report['allow_ambient_grad'],
                   report['expected_native_calls'])
    candidates = report['candidate_order']
    replay = select_aten_entry_candidate(report['rounds'], candidates, warmup_rounds=report['warmup_rounds'],
        measured_rounds=report['measured_rounds'], expected_native_calls=report['expected_native_calls'],
        gain_policy=report.get('gain_policy'))
    if any(report.get(key) != value for key, value in replay.items()):
        raise ValueError('the recorded ATen entry choice does not match the retained round evidence')
    trials = {name: (row['uri'], row['sha256']) for name, row in report['trials'].items()}
    cpus, refs = _trial_context(trials, candidates)
    if report['cpu_ids'] != cpus or report['trials'] != refs:
        raise ValueError('retained trial identity or CPU order differs from the candidate validation')
    for index, round_data in enumerate(report['rounds']):
        for name, run in round_data['runs'].items():
            _verify_run(path.parent, report, index, name, run)
    return report


def _named_options(options, parser):
    fields = {}
    for option in options:
        name, delimiter, value = option.partition('=')
        if not delimiter or not name or not value or name in fields:
            parser.error('provide each named trial option once as NAME=VALUE')
        fields[name] = value
    return fields


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar validate-aten-entry-candidates', description=__doc__)
    run_mode = parser.add_mutually_exclusive_group(required=True)
    run_mode.add_argument('--trial', action='append', metavar='NAME=PATH', help='nominee in incumbent-first order')
    run_mode.add_argument('--check', metavar='REPORT', help='cold-replay retained ATen entry evidence')
    parser.add_argument('--trial-sha256', action='append', default=[], metavar='NAME=SHA256')
    parser.add_argument('--check-sha256')
    parser.add_argument('--entry-function', default='main')
    parser.add_argument('--allow-ambient-grad', action='store_true')
    parser.add_argument('--expected-native-calls', type=int)
    parser.add_argument('--warmup-rounds', type=int, default=1)
    parser.add_argument('--measured-rounds', type=int, default=4)
    parser.add_argument('--out')
    parser.add_argument('entry_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.check is not None:
        if args.check_sha256 is None or args.trial_sha256 or args.out or args.entry_args:
            parser.error('--check requires --check-sha256')
        report = read_entry_candidate_validation(args.check, args.check_sha256)
        location = Path(args.check).resolve()
    else:
        if args.out is None or args.check_sha256 is not None:
            parser.error('candidate execution requires --out and named trial digests')
        paths = _named_options(args.trial, parser)
        digests = _named_options(args.trial_sha256, parser)
        if paths.keys() != digests.keys():
            parser.error('trial paths and SHA256 options must name the same candidates')
        entry_args = args.entry_args[1:] if args.entry_args[:1] == ['--'] else args.entry_args
        report = validate_aten_entry_candidates({name: (path, digests[name]) for name, path in paths.items()},
            entry_function=args.entry_function, entry_args=entry_args, out=args.out,
            warmup_rounds=args.warmup_rounds, measured_rounds=args.measured_rounds,
            allow_ambient_grad=args.allow_ambient_grad, expected_native_calls=args.expected_native_calls)
        location = Path(args.out).resolve() / 'validation.json'
    print(json.dumps(dict(report=str(location), selected=report['selected'], recommendation=report['recommendation'],
                          speedups=report['speedups'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
