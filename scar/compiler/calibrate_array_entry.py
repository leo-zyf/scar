"""Compare sealed standalone C++ CPU entry profiles with fresh original processes."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
from statistics import pvariance
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter_ns

from scar.analysis import native_array_entry_recipe
from scar.codegen import native_array_entry
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.paired_entry_selection import compare_entry_pairs, replay_entry_pairs
from scar.trace.numpy_partition_entry_validation import normalize_report


SCHEMA = 'scar.native-array-entry-calibration'
_RUN = 'scar.native-array-entry-run'


def _need(ok, reason):
    if not ok:
        raise ValueError('array entry calibration: ' + reason)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _write(path, payload):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def _normalize(payload, recipe):
    document = json.loads(payload)
    return json.dumps(normalize_report(document, omit_paths=recipe['runtime_fields']),
                      sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()


def _replay(row, recipe, expected):
    raw = read_artifact(row['stdout'], json_document=False)
    state = json.loads(raw)
    published = json.loads(read_artifact(row['published'], json_document=False))
    stderr = read_artifact(row['stderr'], json_document=False)
    _need(row['returncode'] == 0 and not stderr and state == published
          and _normalize(raw, recipe) == expected,
          'original normal output or published report differs')
    return dict(pair=row['pair'], mode=row['mode'], returncode=row['returncode'],
                wall_ns=row['duration_ns'], stdout_sha256=sha256(expected).hexdigest(),
                stderr_sha256=sha256(stderr).hexdigest())


def _input_argv(profile, recipe):
    flag, = recipe['options'][recipe['output_option']]['flags']
    argv = profile['original_argv']
    _need(argv[-2] == flag and argv.count(flag) == 1, 'original argv has no single terminal output option')
    return [*argv[:-2], flag, 'published.json']


def _run(command, argv, env, folder, pair, mode):
    folder.mkdir()
    start = perf_counter_ns()
    result = subprocess.run([*command, *argv], cwd=folder, env=env, capture_output=True, timeout=120)
    duration = perf_counter_ns() - start
    stdout, stderr = folder / 'stdout.json', folder / 'stderr.log'
    stdout.write_bytes(result.stdout)
    stderr.write_bytes(result.stderr)
    _need(result.returncode == 0, 'process failed: ' + str(stderr))
    return dict(schema=_RUN, schema_version=1, mode=mode, pair=pair, duration_ns=duration,
        returncode=result.returncode, argv=argv, command=command,
        stdout=_ref(stdout, 'application/json'), stderr=_ref(stderr, 'application/octet-stream'),
        published=_ref(folder / 'published.json', 'application/json'))


def _load(path):
    ref = _ref(path, native_array_entry.SCHEMA)
    compiled = read_artifact(ref)
    recipe_ref = compiled['recipe']
    recipe = native_array_entry_recipe.check(recipe_ref['uri'])
    _need(recipe == read_artifact(recipe_ref) and compiled['cpu_ids'] == recipe['cpu_ids']
          and compiled['python_runtime_required'] is False and compiled['scope'] == 'EXPLICIT_QUALIFIED_PROFILE_SELECTION',
          'entry or resource profile changed')
    for artifact in compiled['files'].values():
        read_artifact(artifact, json_document=False)
    return ref, compiled, recipe


def _frontier(recipe, folder, command, argv, env):
    selectors = set(recipe['profiles'][0]['parameters'])
    field = next(name for name in sorted(selectors) if recipe['options'][name]['kind'] == 'int')
    flag, = recipe['options'][field]['flags']
    unseen = max(row['parameters'][field] for row in recipe['profiles']) + 1
    folder.mkdir()
    result = subprocess.run([*command, *argv, flag, str(unseen)], cwd=folder, env=env, capture_output=True, timeout=120)
    _need(result.returncode == 2 and not result.stdout and result.stderr == b'SCAR_PROFILE_UNAVAILABLE\n'
          and not (folder / 'published.json').exists(), 'unsupported profile published visible effects')
    return dict(field=field, unseen_value=unseen, declined_before_publication=True,
                returncode=result.returncode, response=result.stderr.decode())


def calibrate(package, out, *, repetitions=3, warmups=1):
    _need(type(repetitions) is int and repetitions >= 3 and type(warmups) is int and warmups >= 1,
          'at least three clean pairs and one warmup are required')
    entry, compiled, recipe = _load(package)
    out = Path(out).absolute()
    out.mkdir(parents=True)
    cpus = ','.join(map(str, recipe['cpu_ids']))
    launcher = ['/usr/bin/taskset', '-c', cpus]
    original = [*launcher, recipe['source_python']['uri'], '-B', recipe['entry']['uri']]
    native = [*launcher, compiled['files'][compiled['binary']]['uri']]
    env = recipe['environment']
    frontier = _frontier(recipe, out / 'frontier', native,
                         _input_argv(recipe['profiles'][0], recipe), env)
    cohorts = []
    for index, profile in enumerate(recipe['profiles']):
        directory = out / f'profile-{index}'
        directory.mkdir()
        argv = _input_argv(profile, recipe)
        expected = _normalize(json.dumps(profile['report']).encode(), recipe)
        runs = []
        for pair in range(warmups + repetitions):
            arms = (('original', original), ('native', native))
            for mode, command in arms if pair % 2 == 0 else reversed(arms):
                runs.append(_run(command, argv, env, directory / f'{pair}-{mode}', pair, mode))
        measured = compare_entry_pairs([_replay(row, recipe, expected) for row in runs], incumbent='original',
            challenger='native', warmup_pairs=warmups, measurement_pairs=repetitions)
        variance = {key: pvariance(row['samples_ns']) for key, row in measured['metrics'].items()}
        # Check private state in a different process so captures never enter timing samples.
        diagnostic = directory / 'relocation-diagnostic'
        with TemporaryDirectory(prefix='scar-cpu-array-entry-') as temporary:
            relocated = Path(temporary)
            for key, ref in compiled['files'].items():
                if key == compiled['binary'] or key.startswith('provider/objects/'):
                    member = relocated / key
                    member.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ref['uri'], member)
            diagnostic_env = dict(env, SCAR_ARRAY_ENTRY_STATES=str(diagnostic / 'state'))
            observed = _run([*launcher, str(relocated / compiled['binary'])], argv, diagnostic_env,
                            diagnostic, 0, 'native')
            _replay(observed, recipe, expected)
        oracle = read_artifact(profile['package'])['checkpoint_validation']
        rounds = [_ref(diagnostic / f'state.{i}.bin', 'scar.ndarray-native-checkpoint') for i in range(len(oracle))]
        _need(all(read_artifact(actual, json_document=False) == read_artifact(original, json_document=False)
                  for actual, original in zip(rounds, oracle, strict=True)), 'whole entry changed a validated round state')
        cohorts.append(dict(profile=index, parameters=profile['parameters'], runs=runs, relocation_diagnostic=observed,
            checkpoint_validation=rounds, pairing=measured, process_population_variance_ns2=variance,
            timing_scope='FRESH_WHOLE_ENTRY_INCLUDING_REPORT_FILE_AND_STDOUT'))
    report = dict(schema=SCHEMA, schema_version=1, entry=entry, cpu_ids=recipe['cpu_ids'],
        normalized_only=recipe['runtime_fields'], cohorts=cohorts, unsupported_profile=frontier,
        original_default_retained=True, dynamic_seed_generation_tested=False, deployment_authorized=False)
    _write(out / 'result.json', report)
    return report


def check(path):
    ref = _ref(path, SCHEMA)
    report = read_artifact(ref)
    _, compiled, recipe = _load(report['entry']['uri'])
    _need(report['entry'] == _ref(report['entry']['uri'], native_array_entry.SCHEMA)
          and report['cpu_ids'] == compiled['cpu_ids'], 'entry or hardware resource changed')
    for cohort, profile in zip(report['cohorts'], recipe['profiles'], strict=True):
        expected = _normalize(json.dumps(profile['report']).encode(), recipe)
        observed = cohort['pairing']
        paired = replay_entry_pairs([_replay(row, recipe, expected) for row in cohort['runs']], observed,
            incumbent='original', challenger='native', warmup_pairs=observed['warmup_pairs'],
            measurement_pairs=observed['measurement_pairs'])
        _need(paired == observed and cohort['parameters'] == profile['parameters'],
              'clean result changed under independent replay')
        _replay(cohort['relocation_diagnostic'], recipe, expected)
        for actual, expected in zip(cohort['checkpoint_validation'],
                                    read_artifact(profile['package'])['checkpoint_validation'], strict=True):
            _need(read_artifact(actual, json_document=False) == read_artifact(expected, json_document=False),
                  'retained full numerical state differs')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument('--package')
    command.add_argument('--check')
    parser.add_argument('--out')
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    args = parser.parse_args(argv)
    if args.check and not args.out:
        result = check(args.check)
    elif args.package and args.out:
        result = calibrate(args.package, args.out, repetitions=args.repetitions, warmups=args.warmups)
    else:
        parser.error('use --check FILE or --package FILE --out DIRECTORY')
    print(json.dumps([dict(parameters=row['parameters'], metrics=row['pairing']['metrics'],
                           verdict=row['pairing']['verdict']) for row in result['cohorts']], sort_keys=True))


if __name__ == '__main__':
    main()
