"""Compare original and guarded Python entry points in fresh processes."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from statistics import median, variance
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter_ns

from scar.ir.record_codec import loads
from scar.optimization.paired_entry_selection import _metrics
from scar.runtime.aten_python_trial import _read_manifest


_MODES = ('ORIGINAL', 'GUARDED')
_PLACEHOLDER = '{output_dir}'


def _file_receipt(path):
    with path.open('rb') as source:
        digest = sha256()
        size = 0
        while data := source.read(1024 * 1024):
            digest.update(data)
            size += len(data)
    return dict(size_bytes=size, sha256=digest.hexdigest())


def _output_tree(directory):
    files, directories = [], []
    for root, names, filenames in os.walk(directory, followlinks=False):
        root = Path(root)
        for name in sorted(names):
            child = root / name
            if not stat.S_ISDIR(child.lstat().st_mode):
                raise ValueError(f'entry output tree contains a non-directory at {child.relative_to(directory)}')
            directories.append(child.relative_to(directory).as_posix())
        for name in sorted(filenames):
            child = root / name
            relative = child.relative_to(directory).as_posix()
            if not stat.S_ISREG(child.lstat().st_mode):
                raise ValueError(f'entry output tree contains a non-regular file at {relative}')
            files.append(dict(path=relative, **_file_receipt(child)))
        names.sort()
    return dict(directories=sorted(directories), files=sorted(files, key=lambda file: file['path']))


def _run_entry(base, mode, trial, digest, entry_function, argv, cpus, env, allow_ambient_grad):
    directory = base / mode.lower()
    output = directory / 'output'
    output.mkdir(parents=True)
    report_path = directory / 'entry.json'
    command = [
        'taskset', '-c', ','.join(map(str, cpus)), sys.executable, '-m', 'scar.runtime.aten_python_entry',
        '--trial', str(trial), '--trial-sha256', digest, '--entry-function', entry_function,
        '--mode', mode, '--report', str(report_path),
    ]
    if allow_ambient_grad:
        command.append('--allow-ambient-grad')
    expanded_args = [arg.replace(_PLACEHOLDER, str(output)) for arg in argv]
    command.extend(['--', *expanded_args])

    stdout, stderr = directory / 'stdout', directory / 'stderr'
    with stdout.open('wb') as out_stream, stderr.open('wb') as err_stream:
        start = perf_counter_ns()
        result = subprocess.run(command, env=env, stdout=out_stream, stderr=err_stream, check=False)
        process_ns = perf_counter_ns() - start
    if result.returncode:
        with stderr.open('rb') as errors:
            errors.seek(max(0, stderr.stat().st_size - 2048))
            details = errors.read().decode(errors='replace').strip()
        raise ValueError(f'entry {mode} in {base.name} exited with status {result.returncode}\n{details}')
    report = loads(report_path.read_bytes())
    returned = report.get('entry_return') if type(report) is dict else None
    reasons = report.get('fallback_reasons') if type(report) is dict else None
    if (type(report) is not dict or report.get('schema') != 'scar.aten-python-entry-run'
            or report.get('schema_version') != 1 or report.get('mode') != mode
            or report.get('entry_function') != entry_function or report.get('cpu_ids') != sorted(cpus)
            or report.get('trial_sha256') != digest or report.get('entry_args') != expanded_args
            or type(report.get('process_id')) is not int or report['process_id'] <= 0
            or type(report.get('entry_ns')) is not int or not 0 < report['entry_ns'] <= process_ns
            or any(type(report.get(field)) is not int or report[field] < 0
                   for field in ('setup_ns', 'teardown_ns', 'native_calls', 'fallback_calls'))
            or type(reasons) is not dict or any(not key or type(count) is not int or count <= 0
                                               for key, count in reasons.items())
            or sum(reasons.values()) != report['fallback_calls']
            or type(report.get('grad_enabled_at_entry')) is not bool
            or type(report.get('grad_enabled_at_return')) is not bool
            or type(returned) is not dict or set(returned) != {'kind', 'value'}
            or type(returned['value']) not in (type(None), bool, int, str)
            or returned['kind'] != type(returned['value']).__name__
            or report.get('affinity_restored') is not True or report.get('original_binding_restored') is not True):
        raise ValueError(f'entry {mode} in {base.name} returned invalid run evidence')
    if mode == 'ORIGINAL' and (report['native_calls'] or report['fallback_calls']):
        raise ValueError(f'entry {mode} in {base.name} unexpectedly used the trial guard')
    if not stat.S_ISDIR(output.lstat().st_mode):
        raise ValueError(f'entry {mode} in {base.name} changed the output directory type')
    return dict(report, process_ns=process_ns, exit_status=result.returncode,
                dir=f'{base.name}/{mode.lower()}', stdout=_file_receipt(stdout), stderr=_file_receipt(stderr),
                output_tree=_output_tree(output))


def _native_coverage(timed, *, required, expected_native_calls):
    guarded = [pair['runs']['GUARDED'] for pair in timed]
    native = [run['native_calls'] for run in guarded]
    fallback = [run['fallback_calls'] for run in guarded]
    without_fallback = sum(count > 0 and misses == 0 for count, misses in zip(native, fallback))
    count_matches = None
    if expected_native_calls is not None:
        count_matches = sum(count == expected_native_calls for count in native)
    covered = without_fallback == len(guarded) and (count_matches is None or count_matches == len(guarded))
    return dict(scope='MEASURED_GUARDED_ENTRY_PROCESSES', measured_processes=len(guarded),
                native_calls=sum(native), fallback_calls=sum(fallback),
                minimum_native_calls_per_process=min(native), maximum_native_calls_per_process=max(native),
                processes_with_native_and_no_fallback=without_fallback, expected_native_calls=expected_native_calls,
                processes_matching_expected_native_calls=count_matches,
                full_coverage_required=required, coverage_requirement_met=not required or covered)


def _summary(pairs, *, require_full_coverage=False, expected_native_calls=None):
    timed = [pair for pair in pairs if not pair['warmup']]
    boundaries = ('process_ns', 'entry_ns')
    metrics = {}
    for mode in _MODES:
        mode_metrics = {}
        for boundary in boundaries:
            values = [pair['runs'][mode][boundary] for pair in timed]
            mode_metrics[boundary] = dict(_metrics(values), variance_ns2=variance(values))
        metrics[mode] = mode_metrics
    speedups = {boundary: metrics['ORIGINAL'][boundary]['median_ns'] / metrics['GUARDED'][boundary]['median_ns']
                for boundary in boundaries}
    deltas = {'AB': [], 'BA': []}
    for pair in timed:
        delta = pair['runs']['ORIGINAL']['process_ns'] - pair['runs']['GUARDED']['process_ns']
        deltas[pair['order']].append(delta)
    baseline = metrics['ORIGINAL']['process_ns']['median_ns']
    guarded = metrics['GUARDED']['process_ns']['median_ns']
    faster_pairs = sum(delta > 0 for group in deltas.values() for delta in group)
    group_medians = {order: median(values) for order, values in deltas.items() if values}
    checks = dict(median_gain_at_least_five_percent=20 * (baseline - guarded) >= baseline,
                  at_least_three_quarters_pairs_faster=4 * faster_pairs >= 3 * len(timed),
                  positive_median_gain_in_each_order=all(value > 0 for value in group_medians.values()))
    coverage = _native_coverage(timed, required=require_full_coverage, expected_native_calls=expected_native_calls)
    if require_full_coverage:
        checks['guarded_dispatch_coverage'] = coverage['coverage_requirement_met']
    decision = dict(boundary='FRESH_PROCESS_WALL_TIME', checks=checks, faster_pairs=faster_pairs,
                    median_delta_ns_by_order=group_medians)
    return dict(metrics=metrics, speedups=speedups, full_process_decision=decision, native_coverage=coverage,
                recommendation='CANDIDATE_ENTRY_GAIN_OBSERVED' if all(checks.values()) else 'KEEP_ORIGINAL',
                automatic_deployment_authorized=False)


def validate_aten_python_entry(trial, *, trial_sha256, entry_function, entry_args, out,
                               warmup_pairs=2, measured_pairs=7, allow_ambient_grad=False,
                               require_full_coverage=False, expected_native_calls=None):
    """Run an explicit paired trial while preserving the application's entry function."""
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    if not 0 <= warmup_pairs <= 100 or not 3 <= measured_pairs <= 501:
        raise ValueError('entry validation requires 0–100 warmup pairs and 3–501 measured pairs')
    if expected_native_calls is not None and (type(expected_native_calls) is not int or expected_native_calls < 1):
        raise ValueError('the expected native count per measured process must be a positive integer')
    require_full_coverage = require_full_coverage or expected_native_calls is not None
    if not any(_PLACEHOLDER in arg for arg in entry_args):
        raise ValueError('an entry argument must include {output_dir} to isolate paired outputs')
    trial = Path(trial).resolve(strict=True)
    manifest = _read_manifest(trial, trial_sha256)
    cpus = manifest['cpus']
    if (type(cpus) is not list or len(cpus) != 2 or any(type(cpu) is not int for cpu in cpus)
            or len(set(cpus)) != 2 or not set(cpus) <= os.sched_getaffinity(0)):
        raise ValueError('trial CPU pair is outside the current allowed affinity')
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs = []
    with TemporaryDirectory(prefix='.scar-aten-entry-validation-', dir=out.parent) as temporary:
        stage = Path(temporary)
        for index in range(warmup_pairs + measured_pairs):
            order = _MODES if index % 2 == 0 else tuple(reversed(_MODES))
            base = stage / f'pair-{index:03d}'
            runs = {mode: _run_entry(base, mode, trial, trial_sha256, entry_function, entry_args, cpus,
                                     env, allow_ambient_grad) for mode in order}
            original, guarded = (runs[mode] for mode in _MODES)
            if not require_full_coverage and guarded['native_calls'] == 0:
                raise ValueError(f'entry GUARDED in {base.name} has unexpected native dispatch count 0')
            for channel in ('stdout', 'stderr', 'output_tree'):
                if original[channel] != guarded[channel]:
                    raise ValueError(f'entry pair {index} differs in {channel}')
            if original['entry_return'] != guarded['entry_return']:
                raise ValueError(f'entry pair {index} returned a different value')
            if any(original[field] is not guarded[field] for field in ('grad_enabled_at_entry', 'grad_enabled_at_return')):
                raise ValueError(f'entry pair {index} changed application GradMode')
            pairs.append(dict(pair=index, warmup=index < warmup_pairs, order='AB' if order == _MODES else 'BA',
                              runs=runs))
        report = dict(schema='scar.aten-python-entry-validation', schema_version=1,
                      scope='FRESH_PROCESS_ORIGINAL_ENTRY_WITH_GUARDED_NATIVE_REGION',
                      trial=str(trial / 'trial.json'), trial_sha256=trial_sha256, entry_function=entry_function,
                      entry_args=entry_args, allow_ambient_grad=allow_ambient_grad, cpu_ids=cpus,
                      warmup_pairs=warmup_pairs, measured_pairs=measured_pairs, pairs=pairs,
                      matched_stdout=True, matched_stderr=True, matched_output_tree=True,
                      matched_entry_return=True, matched_grad_mode=True,
                      **_summary(pairs, require_full_coverage=require_full_coverage,
                                 expected_native_calls=expected_native_calls))
        (stage / 'validation.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
        stage.rename(out)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar validate-aten-python-entry', description=__doc__)
    parser.add_argument('--trial', required=True)
    parser.add_argument('--trial-sha256', required=True)
    parser.add_argument('--entry-function', default='main')
    parser.add_argument('--allow-ambient-grad', action='store_true')
    parser.add_argument('--require-full-coverage', action='store_true',
                        help='require native dispatch without fallback in every measured guarded process')
    parser.add_argument('--expected-native-calls', type=int,
                        help='require this exact native count and no fallback in every measured guarded process')
    parser.add_argument('--warmup-pairs', type=int, default=2)
    parser.add_argument('--measured-pairs', type=int, default=7)
    parser.add_argument('--out', required=True)
    parser.add_argument('entry_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.entry_args and args.entry_args[0] == '--':
        args.entry_args = args.entry_args[1:]
    report = validate_aten_python_entry(**vars(args))
    print(json.dumps(dict(recommendation=report['recommendation'], speedups=report['speedups'],
                          report=str(Path(args.out) / 'validation.json'))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
