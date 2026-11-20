#!/usr/bin/env python3
"""Compare clean original and native process boundaries with exact output checks."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import select
import shutil
import signal
import statistics
import subprocess
from time import perf_counter_ns

SCHEMA = 'scar.native-process-boundary-benchmark'
CONFIG = 'scar.native-process-boundary-cases'
_OUTPUT = '{output_dir}'
_BLOCK = 1024 * 1024


def _need(condition, detail):
    if not condition:
        raise ValueError(detail)


def _file(path):
    path = Path(path).absolute()
    _need(path.is_file() and not path.is_symlink(), f'expected exact regular artifact: {path}')
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        while block := stream.read(_BLOCK):
            digest.update(block)
            size += len(block)
    return dict(path=str(path), size_bytes=size, sha256=digest.hexdigest())


def _same(left, right):
    if ((left['size_bytes'], left['sha256']) != (right['size_bytes'], right['sha256'])):
        return False
    with Path(left['path']).open('rb') as first, Path(right['path']).open('rb') as second:
        while block := first.read(_BLOCK):
            if block != second.read(_BLOCK):
                return False
        return not second.read(1)


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def _command(raw):
    argv, names = raw['argv'], raw['outputs']
    _need(type(argv) is list and argv and all(type(arg) is str and '\0' not in arg for arg in argv)
          and Path(argv[0]).is_absolute() and Path(argv[0]).is_file()
          and sum(arg.count(_OUTPUT) for arg in argv) == 1,
          'one explicit output-dir placeholder in a known absolute executable command required')
    _need(type(names) is dict and names and len(set(names.values())) == len(names) and all(
        type(name) is str and name and type(value) is str and value not in ('', '.')
        and value == str(PurePosixPath(value)) and not PurePosixPath(value).is_absolute()
        and '..' not in PurePosixPath(value).parts for name, value in names.items()),
        'output map must name distinct original regular files beneath its publication directory')


def _preflight(configuration, baseline_only):
    _need(configuration['schema'] == CONFIG and configuration['schema_version'] == 1,
          'unexpected benchmark case schema')
    cpus = configuration['cpu_ids']
    _need(cpus and len(set(cpus)) == len(cpus) and all(type(cpu) is int and cpu >= 0 for cpu in cpus)
          and set(cpus) <= os.sched_getaffinity(0), 'requested logical CPUs unavailable')
    cwd = Path(configuration['working_directory'])
    _need(cwd.is_absolute() and cwd.is_dir(), 'absolute original launch directory required')
    env = configuration['environment']
    _need(type(env) is dict and all(type(name) is str and '=' not in name and type(value) is str
                                   for name, value in env.items()), 'exact shared process environment required')
    cases = configuration['cases']
    names = [case['id'] for case in cases]
    _need(bool(cases) and len(set(names)) == len(names) and all(re.fullmatch(r'[A-Za-z0-9_-]+', item) for item in names),
          'fixed distinct case IDs required')
    references = {}
    for case in cases:
        _command(case['original'])
        references[f'{case["id"]}:original_executable'] = _file(Path(case['original']['argv'][0]).resolve(strict=True))
        if not baseline_only:
            _need(case.get('candidate'), f'candidate command missing for case {case["id"]}')
            _command(case['candidate'])
            _need(case['original']['outputs'].keys() == case['candidate']['outputs'].keys(),
                  'paired commands must expose the same logical publications')
            references[f'{case["id"]}:candidate_executable'] = _file(Path(case['candidate']['argv'][0]).resolve(strict=True))
    for label, path in configuration.get('identity_files', {}).items():
        references[label] = _file(path)
    for name, path in configuration.get('path_resolution', {}).items():
        _need(shutil.which(name, path=env.get('PATH', '')) == path,
              f'inherited provider PATH selects a different executable: {name}')
        references[f'path:{name}'] = _file(path)
    return references


def _wait_exit(process, timeout_seconds):
    # Popen.wait(timeout=...) polls with sleeps in CPython. A pidfd reports
    # exit readiness to the waiting process without quantizing wall time.
    with_fd = os.pidfd_open(process.pid)
    try:
        ready = select.poll()
        ready.register(with_fd, select.POLLIN)
        if not ready.poll(timeout_seconds * 1000):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            return None
        return process.wait()
    finally:
        os.close(with_fd)


def _run(configuration, command, directory, label):
    run = directory / label
    run.mkdir()
    output = run / 'published'
    argv = [arg.replace(_OUTPUT, str(output)) for arg in command['argv']]
    launch = ['/usr/bin/taskset', '-c', ','.join(map(str, configuration['cpu_ids'])), *argv]
    stdout, stderr = run / 'stdout.bin', run / 'stderr.bin'
    with stdout.open('xb') as out, stderr.open('xb') as err:
        start = perf_counter_ns()
        with subprocess.Popen(launch, cwd=configuration['working_directory'], start_new_session=True,
                              env=configuration['environment'], stdout=out, stderr=err) as process:
            status = _wait_exit(process, configuration['timeout_seconds'])
            duration = perf_counter_ns() - start
    launch_record = dict(argv=argv, timed_argv=launch, elapsed_ns=duration, status=status,
                         timed_out=status is None,
                         stdout=_file(stdout), stderr=_file(stderr))
    _write(run / 'launch.json', launch_record)
    _need(status == 0, f'{label} failed; retained actual process output at {run}')
    _need(output.is_dir() and not output.is_symlink(), f'{label} did not create its fresh publication directory')
    outputs = {}
    published = set()
    for path in output.rglob('*'):
        _need(not path.is_symlink() and (path.is_file() or path.is_dir()), f'invalid publication type: {path}')
        if path.is_file():
            published.add(path.relative_to(output).as_posix())
    _need(published == set(command['outputs'].values()), f'{label} published unexpected or missing files: {published}')
    for key, name in command['outputs'].items():
        outputs[key] = _file(output / name)
    return dict(**launch_record, outputs=outputs, published_directory=str(output))


def _compare(original, candidate):
    matches = {key: _same(original['outputs'][key], candidate['outputs'][key]) for key in original['outputs']}
    stdout = _same(original['stdout'], candidate['stdout'])
    return dict(stdout_exact=stdout, publication_bytes_exact=matches,
                stderr_exact=_same(original['stderr'], candidate['stderr']),
                all_required_bytes_equal=stdout and all(matches.values()))


def _stats(runs):
    values = [row['elapsed_ns'] for row in runs]
    return dict(median_ns=statistics.median(values), min_ns=min(values), max_ns=max(values),
                population_stdev_ns=statistics.pstdev(values), repetitions=len(values))


def _case(configuration, case, directory, *, only, warmups, repetitions):
    case_dir = directory / case['id']
    case_dir.mkdir()
    competitors = ('original',) if only else ('original', 'candidate')
    warm = {mode: [] for mode in competitors}
    for round_ in range(warmups):
        for mode in competitors:
            warm[mode].append(_run(configuration, case[mode], case_dir, f'warmup-{round_}-{mode}'))
    warmup_pairs = [] if only else [_compare(original, candidate) for original, candidate in zip(
        warm['original'], warm['candidate'], strict=True)]
    samples = {mode: [] for mode in competitors}
    pairs = []
    for round_ in range(repetitions):
        order = competitors if round_ % 2 == 0 else competitors[::-1]
        paired = {}
        for mode in order:
            sample = _run(configuration, case[mode], case_dir, f'sample-{round_}-{mode}')
            samples[mode].append(sample)
            paired[mode] = sample
        if not only:
            verdict = _compare(paired['original'], paired['candidate'])
            _write(case_dir / f'comparison-{round_}.json', verdict)
            ratio = (paired['original']['elapsed_ns'] / paired['candidate']['elapsed_ns']
                     if verdict['all_required_bytes_equal'] else None)
            pairs.append(dict(order=list(order), comparison=verdict,
                              observed_ratio=ratio))
    original = samples['original']
    invariant = [_compare(original[0], row) for row in [*warm['original'], *original[1:]]]
    exact = all(row['all_required_bytes_equal'] for row in [*invariant, *warmup_pairs]) and (only or all(
        pair['comparison']['all_required_bytes_equal'] for pair in pairs))
    stats = {mode: _stats(rows) for mode, rows in samples.items()}
    return dict(id=case['id'], arguments=case.get('arguments'), warmup=warm, samples=samples,
                original_repeat_comparisons=invariant, warmup_pair_comparisons=warmup_pairs, pairs=pairs, stats=stats,
                outputs_equal_for_required_comparisons=exact,
                observed_median_speedup=(stats['original']['median_ns'] / stats['candidate']['median_ns']
                                        if exact and not only else None),
                status=('BASELINE_VERIFIED' if only else 'PAIRED_SCOPE_VERIFIED') if exact else 'OUTPUT_MISMATCH')


def measure(config, out, *, baseline_only, warmups, repetitions):
    _need(hasattr(os, 'pidfd_open'), 'run the harness with a Python build exposing Linux pidfd_open')
    config = Path(config).resolve(strict=True)
    identity = _file(config)
    configuration = json.loads(config.read_bytes())
    _need(type(configuration['timeout_seconds']) is int and configuration['timeout_seconds'] > 0,
          'explicit positive process timeout required')
    inputs = _preflight(configuration, baseline_only)
    directory = Path(out).absolute()
    directory.mkdir()
    cases = [_case(configuration, case, directory, only=baseline_only, warmups=warmups, repetitions=repetitions)
             for case in configuration['cases']]
    _need(identity == _file(config) and inputs == _preflight(configuration, baseline_only),
          'original invocation or provider bytes changed across runs')
    report = dict(schema=SCHEMA, schema_version=1, source_case_file=identity, pinned_inputs=inputs,
                  benchmark_scope=configuration['boundary_description'], baseline_only=baseline_only,
                  launched_process_includes_startup_and_descendant_wait=True, instrumented=False,
                  wall_clock_completion='LINUX_PIDFD_EXIT_READINESS_AND_WAITPID',
                  original_and_candidate_share_environment_and_affinity=not baseline_only,
                  cpu_ids=configuration['cpu_ids'],
                  environment=configuration['environment'], working_directory=configuration['working_directory'],
                  correctness_scope='EXACT_STDOUT_AND_EVERY_MAPPED_PUBLISHED_FILE_BYTE',
                  stderr_policy='CAPTURE_COMPLETE_BYTES_AND_REPORT_EQUALITY_SEPARATELY',
                  warmups=warmups, paired_repetitions=repetitions, evidence=cases,
                  qualified=all(row['outputs_equal_for_required_comparisons'] for row in cases),
                  whole_application_equivalence_proven=False, deployment_authorized=False)
    _write(directory / 'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args()
    if args.warmups < 0 or args.repetitions < 1:
        parser.error('warmups must be nonnegative and repetitions must be positive')
    result = measure(args.config, args.out, baseline_only=args.baseline_only,
                     warmups=args.warmups, repetitions=args.repetitions)
    print(json.dumps({case['id']: dict(status=case['status'], medians={
        side: stat['median_ns'] for side, stat in case['stats'].items()}, speedup=case['observed_median_speedup'])
        for case in result['evidence']}, sort_keys=True))
    if not result['qualified']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
