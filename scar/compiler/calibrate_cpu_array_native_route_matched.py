"""Compare native dispatch services using the same complete task waves and seeds."""

import argparse
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import subprocess

from scar.analysis.replay_transaction import replay_transaction
from scar.codegen.cpu_native_route_package import (
    ARRAY_SCHEMA as PACKAGE, _rebind_probed_package, verify_cpu_array_native_route_package,
)
from scar.compiler.calibrate_array_entry import _normalize, _ref, _write
from scar.compiler.calibrate_array_task_entry import _report as _worker_report
from scar.compiler.cpu_array_native_route_phase_probe import _project
from scar.ir.execution_recipe import read_artifact
from scar.optimization import independent_array_task_schedule as process


SCHEMA = 'scar.cpu-array-native-route-matched'
_MODES = ('retained_process', 'grouped_openmp')
_PARTS = ('empty', 'loaded')
_ENV = 'SCAR_ARRAY_ROUTE_EMPTY_PROBE'


def _need(ok, reason):
    if not ok:
        raise ValueError('CPU matched dispatch calibration: ' + reason)


def _render(raw, route):
    edits = []

    def replace(original, replacement, role):
        _need(raw.count(original) == 1, 'generated source does not identify ' + role)
        begin = raw.index(original)
        edits.append((begin, begin + len(original), replacement, role))

    declaration = b'using namespace scar::array_entry_io;\n'
    replace(declaration, declaration + b'bool probe_empty = false;\n', 'declare_probe_control')
    execute = b'        const auto status = execute(result);\n'
    replace(execute, b'        probe_empty = std::getenv("' + _ENV.encode() + b'") != nullptr;\n' + execute,
            'select_probe_body_before_timed_dispatch')
    begin_task = b'    std::string checkpoint;\n'
    empty = b'    if (probe_empty) {\n        result.pid = getpid();\n        return 0;\n    }\n'
    replace(begin_task, empty + begin_task, 'same_abi_empty_worker')
    begin = b'        const auto& path = '
    end = b'        write_result(std::cout, result, false);'
    _need(raw.count(begin) == raw.count(end) == 1 and raw.index(begin) < raw.index(end),
          'one terminal publication is required')
    replace(raw[raw.index(begin):raw.index(end)], b'', 'omit_file_publication_after_timed_dispatch')
    result, sites, cursor = [], [], 0
    for start, stop, replacement, role in sorted(edits):
        _need(start >= cursor, 'calibration sites overlap')
        result.extend((raw[cursor:start], replacement))
        sites.append(dict(role=role, byte_start=start, byte_end=stop,
                          original_sha256=sha256(raw[start:stop]).hexdigest()))
        cursor = stop
    result.append(raw[cursor:])
    return b''.join(result), sites


def _project_arms(root, package, *, build):
    grouped = read_artifact(package['routes'][_MODES[1]]['package'])
    compiler = grouped['compiler']
    read_artifact(compiler, json_document=False)
    return {mode: _project(root/mode/'project', package['routes'][mode], mode, compiler,
                           build=build, renderer=_render, witness_members={}) for mode in _MODES}


def _launch(project, package, mode, argv, part):
    arm = package['routes'][mode]
    base = arm['launch']['environment']
    _need(_ENV not in base and arm['launch']['command'][4:] == package['argv'],
          'calibration needs the admitted command and an unset control variable')
    command = [*arm['launch']['command'][:3], project['diagnostic_executable']['uri'], *argv]
    return command, dict(base, **({_ENV: '1'} if part == 'empty' else {}))


def _seeds(proof, package, seeds):
    schedule = read_artifact(proof['evidence']['schedule'])
    original = read_artifact(schedule['process_plan'])
    serial = read_artifact(schedule['native'])
    prior = [*read_artifact(schedule['thread_evidence'])['seeds'], *read_artifact(original['evidence'])['seeds']]
    holdout = read_artifact(proof['workload']['holdout'])
    key = serial['seed_option']
    prior.append(holdout['parameters'][key])
    width = proof['scope']['task_count']
    limits = serial['seed_range']
    _need(type(seeds) is list and 4 <= len(seeds) <= 16
          and all(type(seed) is int and limits['minimum'] <= seed <= limits['maximum'] - (width - 1)
                  and all(abs(seed - previous) >= width for previous in prior) for seed in seeds)
          and all(right - left >= width for left, right in zip(sorted(seeds), sorted(seeds)[1:])),
          'calibration needs independent, disjoint seed ranges and one warmup')
    recipe = read_artifact(package['recipe'])
    flag, = recipe['options'][key]['flags']
    argv = package['argv']
    _need(argv.count(flag) == 1, 'calibration requires one explicit seed option')
    index = argv.index(flag) + 1
    _need(int(argv[index]) == holdout['parameters'][key], 'held-out seed differs from the sealed invocation')
    return key, {seed: [*argv[:index], str(seed), *argv[index+1:]] for seed in seeds}


def _capture(folder, command, environment):
    folder.mkdir(parents=True)
    completed = subprocess.run(command, cwd=folder, env=environment, stdin=subprocess.DEVNULL,
                               capture_output=True, timeout=120)
    (folder/'stdout.json').write_bytes(completed.stdout)
    _need(completed.returncode == 0 and not completed.stderr,
          'isolated dispatcher failed: ' + str(folder) + ' ' + repr(completed.stderr[:300]))


def _nanoseconds(value):
    _need(type(value) in (int, float) and isfinite(value) and value >= 0, 'invalid internal native timestamp')
    return round(value * 1_000_000_000)


def _sample(folder, command, runtime, recipe, holdout, mode, seed_key, seed, part):
    _need({path.name for path in folder.iterdir()} == {'stdout.json'}, 'isolated dispatch published external output')
    stdout = _ref(folder/'stdout.json', 'application/json')
    raw = read_artifact(stdout, json_document=False)
    params = dict(holdout['parameters'], **{seed_key: seed})
    fields = runtime['output_fields']
    worker_count = runtime['decision']['worker_count']
    parameter_field = read_artifact(recipe['source_contract'])['report']['parameters']['field']
    workers, active = _worker_report(raw, runtime, parameter_field, fields['worker_pid'], params,
                                     (worker_count if mode == _MODES[0] else 1,))
    if mode == _MODES[0]:
        lanes = [row[fields['worker_pid']] for row in workers[:worker_count]]
        _need(set(active) == set(lanes) and all(row[fields['worker_pid']] == lanes[index % worker_count]
                                               for index, row in enumerate(workers)),
              'process task moved to another ordered lane')
    elapsed = _nanoseconds(json.loads(raw)[fields['elapsed']])
    durations = [_nanoseconds(row[fields['worker_wall_seconds']]) for row in workers]
    _need(elapsed > 0 and all(0 < value <= elapsed if part == 'loaded' else value == 0 for value in durations),
          'task service does not fit its dispatcher span')
    return dict(command=command, seed=seed, part=part, stdout=stdout, elapsed_ns=elapsed, task_ns=durations,
                normalized_sha256=sha256(_normalize(raw, recipe)).hexdigest())


def _model(rows, worker_cpus, count, mode):
    loaded = [row for row in rows if row['part'] == 'loaded'][1:]
    empty = [row['elapsed_ns'] for row in rows if row['part'] == 'empty'][1:]
    width = len(worker_cpus)
    by_lane = {cpu: process._summary([row['task_ns'][lane] for row in loaded])
               for lane, cpu in enumerate(worker_cpus)}
    overhead = process._summary(empty)
    if mode == _MODES[0]:
        cost = process._cost(count, worker_cpus, by_lane, overhead)
    else:
        waves, remainder = divmod(count, width)
        numeric = {side: max(by_lane[cpu][side] * (waves + (lane < remainder))
                             for lane, cpu in enumerate(worker_cpus)) for side in process._SIDES}
        cost = dict(worker_service_intervals_ns={str(cpu): values for cpu, values in by_lane.items()},
                    ordered_numeric_completion_ns=numeric, additive_transport_interval_ns=overhead,
                    modeled_elapsed_ns={side: numeric[side] + overhead[side] for side in process._SIDES})
    return cost, process._summary([row['elapsed_ns'] for row in loaded])


def _receipt(reference, package, proof, root, seeds, *, build=False):
    key, arguments = _seeds(proof, package, seeds)
    _need(package['cpu_ids'] == proof['resources']['launch_cpu_ids'], 'calibration hardware differs from route proof')
    projects = _project_arms(root, package, build=build)
    workload = proof['workload']
    recipe, holdout = read_artifact(workload['recipe']), read_artifact(workload['holdout'])
    runtimes = {mode: read_artifact(proof[field]) for mode, field in zip(
        _MODES, ('retained_runtime', 'grouped_runtime'), strict=True)}
    rows = {mode: [] for mode in _MODES}
    for index, seed in enumerate(seeds):
        for mode in _MODES if index % 2 == 0 else _MODES[::-1]:
            for part in _PARTS:
                command, env = _launch(projects[mode], package, mode, arguments[seed], part)
                folder = root/mode/f'{index:02d}-{part}'
                if build:
                    _capture(folder, command, env)
                rows[mode].append(_sample(folder, command, runtimes[mode], recipe, holdout, mode, key, seed, part))
    arms = {}
    for mode in _MODES:
        cpus = runtimes[mode]['decision']['worker_cpu_ids']
        conditional, loaded = _model(rows[mode], cpus, proof['scope']['task_count'], mode)
        arms[mode] = dict(executable=package['routes'][mode]['executable'], worker_cpu_ids=cpus, samples=rows[mode],
                          project=projects[mode], conditional_model=conditional, loaded_dispatch_span_ns=loaded)
    _need(all(a['normalized_sha256'] == b['normalized_sha256']
              for a, b in zip(rows[_MODES[0]], rows[_MODES[1]], strict=True)),
          'matched routes disagree on the calibration inputs or calculated outputs')
    return dict(schema=SCHEMA, schema_version=1, package=reference, proof=read_artifact(package['plan'])['proof'],
                scope=proof['scope'], cpu_ids=package['cpu_ids'], seeds=seeds, warmups=1, route_arms=arms,
                boundary='NATIVE_EXECUTE_THROUGH_TEAM_JOIN_BEFORE_OWNER_REDUCTION_AND_PUBLICATION',
                same_protocol=True, physical_bounds_proven=False, whole_entry_executed=False,
                deployment_authorized=False)


def _verify(reference, proof_ref=None, source=None, *, checked_package=None):
    saved = read_artifact(reference)
    _need(reference['schema'] == SCHEMA and saved['schema_version'] == 1, 'unsupported matched dispatch receipt')
    if checked_package is None:
        package = verify_cpu_array_native_route_package(saved['package'])
    else:
        package = _rebind_probed_package(saved['package'], checked_package)
    original = read_artifact(package['plan'])['proof']
    if source is None:
        source = read_artifact(original)  # Validated by the package's route plan.
    else:
        _need(original == proof_ref, 'matched dispatch receipt belongs to another route proof')
    out = Path(reference['uri']).parent
    _need(saved == _receipt(saved['package'], package, source, out, saved['seeds']),
          'matched dispatch receipt differs on replay')
    return saved


def verify_cpu_array_native_route_matched(reference):
    with replay_transaction():
        return _verify(reference)


def measure(package_ref, out, *, seeds):
    package = verify_cpu_array_native_route_package(package_ref)
    proof = read_artifact(read_artifact(package['plan'])['proof'])
    _need(set(package['cpu_ids']) <= os.sched_getaffinity(0), 'recorded launch cohort is unavailable')
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = _receipt(package_ref, package, proof, out, seeds, build=True)
    _write(out/'result.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--package', type=Path)
    mode.add_argument('--check', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--seed', type=int, action='append', default=[])
    args = parser.parse_args(argv)
    if args.check is not None:
        if args.seed or args.out is not None:
            parser.error('--check reads an existing dispatch calibration')
        report = verify_cpu_array_native_route_matched(_ref(args.check, SCHEMA))
    else:
        if args.out is None or len(args.seed) < 4:
            parser.error('--package requires --out and at least four independent seeds')
        with replay_transaction():
            report = measure(_ref(args.package, PACKAGE), args.out, seeds=args.seed)
    print(json.dumps({name: arm['loaded_dispatch_span_ns'] for name, arm in report['route_arms'].items()}))


if __name__ == '__main__':
    raise SystemExit(main())
