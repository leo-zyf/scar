"""Run a sealed partition trial inside normally spawned CPython workers."""

import argparse
import atexit
import json
import os
from pathlib import Path
import sys
from time import perf_counter_ns

from scar.ir.execution_recipe import artifact_reference, read_artifact


_ENV = 'SCAR_PARTITION_SPAWN_PLAN'
_PLAN = 'scar.partition-spawn-plan'
_WORKER = 'scar.partition-spawn-worker'
_RESULT = 'scar.partition-spawn-result'
_TRIAL = 'scar.partition-copy-trial'


def _write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _affinity():
    return sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None


def install_spawn_worker():
    """Join only a directly spawned worker in a pinned caller's trial."""
    # Resource trackers and unrelated Python subprocesses lack this CPython
    # spawn marker. Read the token before importing the worker's application.
    if len(sys.argv) < 2 or sys.argv[1] != '--multiprocessing-fork':
        return
    from importlib.machinery import PathFinder

    ref = json.loads(os.environ[_ENV])
    plan = read_artifact(ref)
    parent = os.getppid()
    if (plan.get('schema') != _PLAN or plan.get('schema_version') != 1 or parent != plan['parent_pid']):
        return
    output = Path(ref['uri']).parent
    pid = os.getpid()
    affinity, started = _affinity(), perf_counter_ns()
    scope = None
    frontier = []
    if plan['mode'] == 'TRIAL':
        from .entry_runner import _Finder
        from .partition_trial import PartitionScope

        try:
            scope = PartitionScope(plan['trial'], parent_image=plan['parent_image'])
            if sum(finder is PathFinder for finder in sys.meta_path) == 1:
                sys.meta_path[:] = [_Finder(scope) if finder is PathFinder else finder for finder in sys.meta_path]
            else:
                frontier.append('standard_pathfinder_position_unavailable')
        except (OSError, ValueError, ImportError) as error:
            frontier.append('worker_trial_rejected:' + str(error))

    def report():
        evidence = scope.worker_evidence() if scope else dict(bindings=[], frontier=[])
        evidence['frontier'] = frontier + evidence['frontier']
        _write(output / f'worker-{pid}.json', dict(schema=_WORKER, schema_version=1,
            plan_sha256=ref['sha256'], pid=pid, spawn_parent_pid=parent, mode=plan['mode'],
            allowed_cpu_ids_at_start=affinity, allowed_cpu_ids_at_exit=_affinity(),
            elapsed_boot_to_exit_ns=perf_counter_ns() - started, **evidence))

    atexit.register(report)


def run_spawn_trial(entry, argv, trial, out, *, exclusive_normal_completion=False):
    """Launch the unchanged entry and read worker evidence after it has returned."""
    from .entry_runner import _arguments, _execute

    entry, argv = _arguments(entry, argv)
    if trial['schema'] != _TRIAL:
        raise ValueError('partition trial artifact required')
    receipt = read_artifact(trial)
    directory = Path(out).resolve()
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    path = directory / 'plan.json'
    mode = 'TRIAL' if exclusive_normal_completion is True else 'ORIGINAL'
    parent = os.getpid()
    parent_image = None
    parent_image_verification_ns = None
    if mode == 'TRIAL':
        from scar.codegen.partition_multiply import verify_running_python

        started = perf_counter_ns()
        parent_image = verify_running_python(receipt['native'])
        parent_image_verification_ns = perf_counter_ns() - started
    _write(path, dict(schema=_PLAN, schema_version=1, parent_pid=parent, entry=str(entry),
        argv=argv, trial=trial, mode=mode, allowed_cpu_ids=_affinity(), parent_image=parent_image,
        parent_image_verification_ns=parent_image_verification_ns))
    ref = artifact_reference(path, _PLAN)
    prior = os.environ.get(_ENV)
    os.environ[_ENV] = json.dumps(ref, separators=(',', ':'))
    try:
        execution = _execute(entry, argv, None, 'launch_original_parent_for_partition_spawn_trial')
    finally:
        if prior is None:
            del os.environ[_ENV]
        else:
            os.environ[_ENV] = prior

    from scar.ir.execution.coordinates import ProcessInstance
    from scar.ir.execution.ids import ProcessInstanceID, RunID
    from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
    from scar.ir.record_codec import encode

    run = RunID(ref['sha256'])

    def process(pid, artifact, locator):
        return ProcessInstance(ProcessInstanceID(f'{run.value}:{pid}'), run, pid,
            ProviderFactReference(ProviderArtifact(**artifact), locator))

    owner = process(parent, ref, '#/parent_pid')
    workers = []
    statistics = dict(native_calls=0, fallback_calls=0, logical_copy_bytes_elided=0)
    for file in sorted(directory.glob('worker-*.json')):
        reference = artifact_reference(file, _WORKER)
        row = read_artifact(reference)
        pid = row['pid']
        if (type(pid) is not int or pid == parent or file.name != f'worker-{pid}.json'
                or row['plan_sha256'] != ref['sha256'] or row['spawn_parent_pid'] != parent or row['mode'] != mode):
            raise ValueError('worker report does not belong to the selected launch')
        worker = process(pid, reference, '#/pid')
        workers.append(dict(process=encode(worker), spawn_parent=owner.id.as_dict(), record=reference))
        for binding in row['bindings']:
            for key in statistics:
                statistics[key] += binding[key]
    result = dict(schema=_RESULT, schema_version=1, mode=mode, plan=ref, parent=encode(owner),
        workers=workers, statistics=statistics, parent_execution=execution,
        scope='direct spawn children that wrote a normal interpreter-exit report',
        worker_coverage_assumed=False, deployment_authorized=False)
    _write(directory / 'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', required=True)
    parser.add_argument('--trial-sha256', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--exclusive-normal-completion', action='store_true')
    parser.add_argument('program', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.program[1:] if args.program[:1] == ['--'] else args.program
    if not command:
        parser.error('supply one Python entry and its arguments after --')
    trial = artifact_reference(args.trial, _TRIAL)
    if trial['sha256'] != args.trial_sha256:
        parser.error('partition trial SHA-256 differs')
    run_spawn_trial(command[0], command[1:], trial, args.out,
        exclusive_normal_completion=args.exclusive_normal_completion)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
