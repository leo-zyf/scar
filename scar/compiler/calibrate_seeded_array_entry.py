"""Validate runtime seeds and optional loop counts against independent original runs."""

import argparse
import json
from pathlib import Path
import shutil
from statistics import pvariance
import subprocess
from tempfile import TemporaryDirectory

from scar.analysis import native_array_entry_recipe
from scar.compiler.calibrate_array_entry import _normalize, _ref, _replay, _run, _write
from scar.ir.execution_recipe import read_artifact
from scar.optimization.paired_entry_selection import compare_entry_pairs


SCHEMA = 'scar.native-seeded-array-entry-calibration'
_NATIVE = 'scar.native-seeded-array-entry'
_LOOP = 'scar.native-seeded-array-loop-entry'
_HOLDOUT = 'scar.array-entry-full-state-holdout'
_HANDOFFS = 'scar.array-loop-handoff-counts'


def _need(condition, detail):
    if not condition:
        raise ValueError('seeded entry calibration: ' + detail)


def _load(path):
    schema = json.loads(Path(path).read_bytes())['schema']
    _need(schema in (_NATIVE, _LOOP), 'expected a compiled seeded entry')
    entry = _ref(path, schema)
    native = read_artifact(entry)
    base = native_array_entry_recipe.check(native['recipe']['uri'])
    _need(base == read_artifact(native['recipe']) and native['cpu_ids'] == base['cpu_ids']
          and native['python_runtime_required'] is False and native['independent_runtime_seeds'] is True,
          'native package or original recipe differs')
    for reference in native['files'].values():
        read_artifact(reference, json_document=False)
    return entry, native, base


def _heldout(reference, native, base):
    case = read_artifact(reference)
    seed_option = native['seed_option']
    seed = case['parameters'][seed_option]
    round_option = native.get('round_option')
    params = {key: value for key, value in case['parameters'].items() if key not in (seed_option, round_option)}
    _need(case['entry'] == base['entry'] and case['cpu_ids'] == base['cpu_ids']
          and case['source_contract'] == base['source_contract'] and case['interpreter'] == base['source_python']
          and case['independent_original_run'] is True and case['valid_for'] == 'VALIDATION_ONLY'
          and case['topology_generation_authorized'] is False and case['tuning_authorized'] is False
          and case['executed_worker_process_count'] == 1
          and all(seed != row['parameters'][seed_option] for row in base['profiles'])
          and params == native['source_parameters'], 'holdout crosses the original entry or seed domain')
    if round_option is not None:
        rounds = case['parameters'][round_option]
        _need(type(rounds) is int and rounds >= 1 and rounds == len(case['states']),
              'holdout must record every original round')
    _need(not read_artifact(case['stderr'], json_document=False), 'original holdout reported a frontier')
    expected = read_artifact(case['stdout'], json_document=False)
    decoded = json.loads(expected)
    _need(decoded == json.loads(read_artifact(case['files']['out'], json_document=False))
          and decoded['parameters'] == case['parameters'], 'original holdout file/stdout differ')
    flag, = base['options'][base['output_option']]['flags']
    argv = case['argv']
    _need(argv[-2] == flag and argv.count(flag) == 1, 'expected one original file option')
    return case, [*argv[:-2], flag, 'published.json'], _normalize(expected, base)


def _state_compare(oracle, checkpoints):
    for index, (original, candidate) in enumerate(zip(oracle, checkpoints, strict=True)):
        _need(original['round_index'] == index and read_artifact(original['data'], json_document=False)
              == read_artifact(candidate, json_document=False), 'complete held-out state differs in round ' + str(index))


def _prepare_relocation(native, directory):
    for name, reference in native['files'].items():
        if name == native['binary'] or name.startswith('provider/objects/'):
            out = directory / name
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(reference['uri'], out)
    return directory / native['binary']


def _loop_handoffs(reference, native, case):
    counts = json.loads(read_artifact(reference, json_document=False))
    _need(set(counts) == {'copies', 'same_span_transfers'}
          and all(type(count) is int and count >= 0 for count in counts.values())
          and sum(counts.values()) == case['parameters'][native['round_option']] - 1,
          'round transitions do not match the diagnostic')
    return dict(artifact=reference, counts=counts,
                carried_bytes=counts['copies'] * native['maximum_carry_copy_bytes_per_transition'])


def _require_decline(command, environment, directory):
    directory.mkdir()
    result = subprocess.run(command, cwd=directory, env=environment, capture_output=True, timeout=30)
    _need(result.returncode == 2 and result.stderr == b'SCAR_PROFILE_UNAVAILABLE\n'
          and not result.stdout and not (directory / 'published.json').exists(),
          'out-of-scope invocation published effects')


def calibrate(package, holdouts, out, *, warmups=1, repetitions=3):
    _need(len(holdouts) >= 2 and warmups >= 1 and repetitions >= 3, 'two seeds and clean paired runs required')
    entry, native, base = _load(package)
    out = Path(out).absolute()
    out.mkdir(parents=True)
    launcher = ['/usr/bin/taskset', '-c', ','.join(map(str, native['cpu_ids']))]
    original = [*launcher, base['source_python']['uri'], '-B', base['entry']['uri']]
    binary = [*launcher, native['files'][native['binary']]['uri']]
    env = base['environment']
    references = [_ref(path, _HOLDOUT) for path in holdouts]
    _need(len({row['sha256'] for row in references}) == len(references), 'duplicate original holdout')
    rows = []
    for index, reference in enumerate(references):
        case, argv, expected = _heldout(reference, native, base)
        folder = out / f'case-{index}'
        folder.mkdir()
        raw = []
        for pair in range(warmups + repetitions):
            arms = [('original', original), ('native', binary)]
            for mode, command in arms if pair % 2 == 0 else reversed(arms):
                raw.append(_run(command, argv, env, folder / f'{pair}-{mode}', pair, mode))
        pairing = compare_entry_pairs([_replay(run, base, expected) for run in raw],
            incumbent='original', challenger='native', warmup_pairs=warmups, measurement_pairs=repetitions)
        variance = {key: pvariance(row['samples_ns']) for key, row in pairing['metrics'].items()}

        diagnostic = folder / 'relocated'
        with TemporaryDirectory(prefix='scar-seeded-entry-') as temporary:
            moved = _prepare_relocation(native, Path(temporary))
            observed = _run([*launcher, str(moved)], argv,
                            dict(env, SCAR_ARRAY_ENTRY_STATES=str(diagnostic / 'state')), diagnostic, 0, 'native')
            _replay(observed, base, expected)
        checkpoints = [_ref(diagnostic / f'state.{round_}.bin', 'scar.ndarray-native-checkpoint')
                       for round_ in range(len(case['states']))]
        _state_compare(case['states'], checkpoints)
        row = dict(holdout=reference, parameters=case['parameters'], clean_pairs=raw,
                   relocation=observed, checkpoints=checkpoints, pairing=pairing, population_variance_ns2=variance)
        if native.get('round_option') is not None:
            row['handoffs'] = _loop_handoffs(_ref(diagnostic / 'state.handoffs.json', _HANDOFFS), native, case)
        rows.append(row)

    seed_flag, = base['options'][native['seed_option']]['flags']
    _require_decline([*binary, *argv, seed_flag, str(2**32)], env, out / 'outside-source-seed-range')
    report = dict(schema=SCHEMA, schema_version=1, entry=entry, cases=rows, cpu_ids=native['cpu_ids'],
        timing_scope='FRESH_PROCESS_WITH_SEED_CONSTRUCTION_NUMERIC_LOOP_TERMINAL_REPORT_AND_FILE_OUTPUT',
        excluded_original_timing_fields=base['runtime_fields'], seed_guard_verified=True,
        compiled_from_holdout=False, deployment_authorized=False)
    if native.get('round_option') is not None:
        round_flag, = base['options'][native['round_option']]['flags']
        _require_decline([*binary, *argv, round_flag, '0'], env, out / 'outside-positive-round-count')
        report['positive_round_count_guard_verified'] = True
    _write(out / 'result.json', report)
    return report


def check(path):
    stored = read_artifact(_ref(path, SCHEMA))
    ref, native, base = _load(stored['entry']['uri'])
    _need(ref == stored['entry'] and native['cpu_ids'] == stored['cpu_ids'], 'compiled scope changed')
    for row in stored['cases']:
        case, argv, expected = _heldout(row['holdout'], native, base)
        _need(row['parameters'] == case['parameters'] and all(run['argv'] == argv for run in row['clean_pairs']),
              'original held-out arguments changed')
        prior = row['pairing']
        rebuilt = compare_entry_pairs([_replay(run, base, expected) for run in row['clean_pairs']],
            incumbent='original', challenger='native', warmup_pairs=prior['warmup_pairs'],
            measurement_pairs=prior['measurement_pairs'])
        _need(rebuilt == prior, 'clean pairing differs')
        _replay(row['relocation'], base, expected)
        _state_compare(case['states'], row['checkpoints'])
        if native.get('round_option') is not None:
            _need(_loop_handoffs(row['handoffs']['artifact'], native, case) == row['handoffs'],
                  'recorded loop transitions differ')
    return stored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--package')
    source.add_argument('--check')
    parser.add_argument('--holdout', action='append')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check and not any((args.holdout, args.out)):
        report = check(args.check)
    elif args.package and args.holdout and args.out:
        report = calibrate(args.package, args.holdout, args.out)
    else:
        parser.error('use --check FILE or --package FILE --holdout FILE --holdout FILE --out DIRECTORY')
    print(json.dumps([dict(parameters=row['parameters'], pairing=row['pairing']['metrics'])
                      for row in report['cases']], sort_keys=True))


if __name__ == '__main__':
    main()
