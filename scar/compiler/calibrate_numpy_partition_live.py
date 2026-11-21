"""Measure guarded native publication at an unchanged CPU workload entry."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import pvariance
import subprocess
import sys
from time import perf_counter_ns

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.paired_entry_selection import compare_entry_pairs
from scar.runtime.entry_runner import run_original_script
from scar.runtime.native_numpy_partition_trial import run
from scar.trace.numpy_partition_entry_validation import compare_trajectories, normalize_report, observe_states


SCHEMA = 'scar.numpy-partition-live-calibration'
_CONFIG = 'scar.numpy-partition-live-experiment'
_NATIVE = 'scar.native-numpy-partition-live'
_RUN = 'scar.numpy-partition-live-run'
_MODES = ('original', 'trial')


def _write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def _open(path, schema):
    reference = artifact_reference(path, schema)
    return reference, read_artifact(reference)


def _worker(task_path):
    task = json.loads(Path(task_path).read_text())
    experiment = read_artifact(task['config'])
    profile = read_artifact(task['profile'])
    native = read_artifact(task['native'])
    if (native['profile'], native['extension']) != (task['profile'], task['extension']):
        raise ValueError('worker profile and binary differ from the verified package')
    read_artifact(profile['entry'], json_document=False)
    entry = profile['entry']['uri']
    case = experiment['cases'][task['case']]
    argv = case['argv']
    if sorted(os.sched_getaffinity(0)) != profile['cpu_ids']:
        raise ValueError('worker affinity differs from the qualified CPU scope')

    def execute():
        started = perf_counter_ns()
        if task['mode'] == 'trial':
            result = run(entry, argv, profile, task['extension'], exclusive_normal_completion=True)
        else:
            result = run_original_script(entry, argv)
        return result, perf_counter_ns() - started

    trajectory = None
    if task['diagnostic']:
        with observe_states(experiment['observer']) as observer:
            execution, entry_ns = execute()
        trajectory = observer.report()
    else:
        execution, entry_ns = execute()
    _write(Path(task['sidecar']), dict(schema=_RUN, schema_version=1, mode=task['mode'],
        case=task['case'], diagnostic=task['diagnostic'], affinity=profile['cpu_ids'], execution=execution,
        worker_entry_ns=entry_ns if trajectory is None else None, trajectory=trajectory))


def _projection(row, inputs, experiment):
    task = read_artifact(row['task'])
    run = read_artifact(row['sidecar'])
    # Earlier observations seal the extension through the outer package only.
    if ((task['config'], task['profile'], task['extension'], task.get('native', inputs[3])) != inputs
            or (task['mode'], task['case'], task['diagnostic']) !=
               (run['mode'], run['case'], run['diagnostic'])):
        raise ValueError('execution and report do not describe the same verified run')
    stdout = read_artifact(row['stdout'], json_document=False)
    stderr = read_artifact(row['stderr'], json_document=False)
    clean = normalize_report(json.loads(stdout), omit_paths=experiment['stdout_omit_paths'])
    digest = sha256(json.dumps(clean, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
    return run, dict(mode=task['mode'], wall_ns=row['wall_ns'], returncode=row['returncode'],
                     stdout_sha256=digest.hexdigest(), stderr_sha256=sha256(stderr).hexdigest())


def _reduce(rows, inputs, experiment):
    observed, timing, workers = {}, [], []
    profile = read_artifact(inputs[1])
    clean_trial_calls = []
    for row in rows:
        run, projected = _projection(row, inputs, experiment)
        case = experiment['cases'][run['case']]
        execution = run['execution']
        if (row['returncode'] != 0 or execution['exit_code'] != 0
                or execution['argv'] != case['argv'] or execution['entry_path'] != profile['entry']['uri']
                or run['affinity'] != profile['cpu_ids']):
            raise ValueError('source argv, CPU scope or normal completion differs')
        statistics = execution['statistics']
        if run['mode'] == 'trial' and execution['installed_bindings'] != 1:
            raise ValueError('scoped boundary did not attach: ' + str(execution))
        if run['diagnostic']:
            group = observed.setdefault(run['case'], {})
            if run['mode'] in group:
                raise ValueError('duplicate diagnostic mode')
            group[run['mode']] = (run, projected)
        else:
            if run['case'] != 0 or run['trajectory'] is not None:
                raise ValueError('timing cohort must be clean and use the primary entry')
            timing.append(dict(projected, pair=row['pair']))
            workers.append(dict(projected, pair=row['pair'], wall_ns=run['worker_entry_ns']))
            if run['mode'] == 'trial':
                clean_trial_calls.append(dict(pair=row['pair'], binding_ns=execution['binding_ns'], **statistics))

    validations = []
    if sorted(observed) != list(range(len(experiment['cases']))):
        raise ValueError('independent diagnostic coverage differs')
    for index, modes in sorted(observed.items()):
        original, source = modes['original']
        trial, candidate = modes['trial']
        left, right = original['trajectory'], trial['trajectory']
        expected = experiment['cases'][index]['expected_state_reads']
        trajectory = compare_trajectories(left, right, require_recurrence=True)
        if (trajectory['matched_calls'] != expected
                or any(source[key] != candidate[key] for key in ('stdout_sha256', 'stderr_sha256'))):
            raise ValueError('diagnostic trajectory, stdout, or stderr differs')
        validations.append(dict(case=index, reads=expected, trajectory=trajectory,
                                normalized_stdout_equal=True, stderr_equal=True,
                                native_execution_observed=trial['execution']['statistics']['native_calls'] > 0,
                                trial_statistics=trial['execution']['statistics']))

    def compare(runs):
        return compare_entry_pairs(runs, incumbent='original', challenger='trial',
                                   warmup_pairs=experiment['warmup_pairs'],
                                   measurement_pairs=experiment['measured_pairs'])

    whole, inside = compare(timing), compare(workers)
    variance = {name: pvariance(metrics['samples_ns']) for name, metrics in whole['metrics'].items()}
    admitted = (all(row['native_execution_observed'] for row in validations)
                and all(row['native_calls'] for row in clean_trial_calls
                        if row['pair'] >= experiment['warmup_pairs']))
    verdict = whole['verdict'] if admitted else 'KEEP_INCUMBENT_INCOMPLETE_LIVE_ADMISSION'
    return dict(validation=validations, whole_process=whole, worker_entry=inside, clean_trial_calls=clean_trial_calls,
                whole_process_population_variance_ns2=variance,
                live_admission_observed_on_all_required_runs=admitted, decision=verdict,
                default_entry_backend='NO_OP', deployment_authorized=False)


def calibrate(config_path, native_path, out):
    from scar.codegen.native_numpy_partition_live import check as check_native

    config_ref, experiment = _open(config_path, _CONFIG)
    native_ref, native = _open(native_path, _NATIVE)
    if check_native(native_path) != native:
        raise ValueError('native package changed after full verification')
    profile = read_artifact(native['profile'])
    if experiment['observer']['caller_source'] != profile['entry']:
        raise ValueError('observer was qualified against another entry')
    if (not experiment['cases'] or any(case['expected_state_reads'] < 1 for case in experiment['cases'])
            or len({tuple(case['argv']) for case in experiment['cases']}) != len(experiment['cases'])
            or experiment['warmup_pairs'] < 0 or experiment['measured_pairs'] < 3):
        raise ValueError('distinct diagnostic cases, valid warmup and at least three timing pairs required')
    python = profile['source_python']
    read_artifact(python, json_document=False)
    inputs = config_ref, native['profile'], native['extension'], native_ref
    out = Path(out).resolve()
    out.mkdir(parents=True)
    env = os.environ.copy()
    env.pop('PYTHONPROFILEIMPORTTIME', None)
    env['PYTHONHASHSEED'] = '0'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    # Use the same launcher and pinned process resources in both arms.
    launcher = ['taskset', '-c', ','.join(map(str, profile['cpu_ids'])),
                python['uri'], '-B', '-m', __name__ if __name__ != '__main__' else __spec__.name, '--worker']
    rows = []

    def sample(label, mode, case, diagnostic, pair=None):
        folder = out / (label + '-' + mode)
        folder.mkdir()
        task, sidecar = folder / 'task.json', folder / 'run.json'
        _write(task, dict(schema='scar.numpy-partition-live-task', schema_version=1, mode=mode,
            case=case, diagnostic=diagnostic, config=config_ref, profile=native['profile'],
            extension=native['extension'], native=native_ref, sidecar=str(sidecar)))
        started = perf_counter_ns()
        result = subprocess.run([*launcher, str(task)], env=env, capture_output=True, check=False)
        elapsed = perf_counter_ns() - started
        stdout, stderr = folder / 'stdout.json', folder / 'stderr.log'
        stdout.write_bytes(result.stdout)
        stderr.write_bytes(result.stderr)
        if result.returncode != 0:
            raise RuntimeError('worker failed, see ' + str(stderr))
        rows.append(dict(pair=pair, returncode=result.returncode, wall_ns=elapsed,
                         task=artifact_reference(task, 'scar.numpy-partition-live-task'),
                         sidecar=artifact_reference(sidecar, _RUN),
                         stdout=artifact_reference(stdout, 'scar.numpy-partition-source-stdout'),
                         stderr=artifact_reference(stderr, 'scar.numpy-partition-source-stderr')))

    for case in range(len(experiment['cases'])):
        for mode in _MODES:
            sample(f'diagnostic-{case}', mode, case, True)
    for pair in range(experiment['warmup_pairs'] + experiment['measured_pairs']):
        for mode in _MODES[::(-1 if pair % 2 else 1)]:
            sample(f'clean-{pair}', mode, 0, False, pair)
    result = dict(schema=SCHEMA, schema_version=1, experiment=config_ref, native=native_ref,
                  processes=rows, source_python=python, cpu_ids=profile['cpu_ids'],
                  timing='FRESH_PINNED_SUBPROCESSES_WITHOUT_TRAJECTORY_TRACING',
                  full_program_cpp=False, logical_copy_bytes_saved_claim=False,
                  **_reduce(rows, inputs, experiment))
    _write(out / 'result.json', result)
    return result


def check(path):
    _, result = _open(path, SCHEMA)
    experiment = read_artifact(result['experiment'])
    native = read_artifact(result['native'])
    inputs = result['experiment'], native['profile'], native['extension'], result['native']
    reduction = _reduce(result['processes'], inputs, experiment)
    if any(result[key] != value for key, value in reduction.items()):
        raise ValueError('stored decision differs from paired raw observations')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--config', type=Path)
    mode.add_argument('--check', type=Path)
    mode.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--native', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.worker:
        _worker(args.worker)
        return
    if args.config and args.native and args.out:
        report = calibrate(args.config, args.native, args.out)
    elif args.check and args.native is None and args.out is None:
        report = check(args.check)
    else:
        parser.error('supply --check RESULT or --config EXPERIMENT --native PACKAGE --out DIRECTORY')
    print(json.dumps({key: report[key] for key in ('validation', 'whole_process', 'worker_entry',
                                                   'decision', 'default_entry_backend')}, sort_keys=True))


if __name__ == '__main__':
    main()
