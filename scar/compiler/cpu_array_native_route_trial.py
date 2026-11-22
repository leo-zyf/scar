"""Compare two admitted CPU array task routes on their shared held-out entry."""

import argparse
import json
from os import sched_getaffinity
from pathlib import Path
from statistics import pvariance

from scar.analysis.replay_transaction import replay_transaction
from scar.compiler import calibrate_array_task_entry as calibration
from scar.compiler.calibrate_array_entry import _ref, _run, _write
from scar.compiler.cpu_task_arena_study import _checkpoint_files
from scar.ir.execution_recipe import read_artifact
from scar.optimization.paired_entry_selection import compare_entry_pairs, replay_entry_pairs


SCHEMA = 'scar.cpu-array-native-route-observation'
_RETENTION = 'scar.retained-native-array-task-entry'
_GROUPED = 'scar.native-cpu-fp-guard-groups'
_MODES = ('retained_process', 'grouped_openmp')
_STATES = 'SCAR_ARRAY_ENTRY_STATES'
_LANES = 'SCAR_ARRAY_ENTRY_LANE_REPORT'


def _need(condition, message):
    if not condition:
        raise ValueError('CPU native array route: ' + message)


def _scope(retention_ref, grouped_ref):
    retention = read_artifact(retention_ref)
    grouped = read_artifact(grouped_ref)
    acceptance = read_artifact(retention['existing_acceptance'])
    holdout = read_artifact(acceptance['holdout'])
    process = read_artifact(retention['retained_release'])
    openmp = read_artifact(grouped['runtime'])
    recipe_ref = process['recipe']
    recipe = read_artifact(recipe_ref)
    _need(retention_ref['schema'] == _RETENTION and grouped_ref['schema'] == _GROUPED
          and grouped['runtime'] == acceptance['entry'] and acceptance['validation_controls']['processes']['entry']
          == retention['retained_release'] and process['decision']['mode'] == calibration._PROCESSES
          and openmp['decision']['mode'] == calibration._OPENMP, 'routes differ from the retained acceptance')
    _need(openmp['recipe'] == grouped['recipe'] == holdout['recipe'] == recipe_ref
          and process['cpu_ids'] == grouped['cpu_ids'] == openmp['cpu_ids'] == holdout['cpu_ids']
          and process['scope'] == grouped['scope'] == openmp['scope']
          and process['output_fields'] == openmp['output_fields'] and holdout['environment'] == recipe['environment'],
          'routes do not share a recipe, workload, report, CPU mask and original environment')
    flag, = recipe['options'][recipe['output_option']]['flags']
    _need(holdout['argv'][-2:] == [flag, 'published.json'] and holdout['argv'].count(flag) == 1,
          'holdout must publish once at the isolated run path')
    _need(grouped['binary'] == openmp['binary'], 'grouped release binary does not match the native member role')
    grouped_view = dict(openmp, files={**openmp['files'], openmp['binary']: grouped['executable']},
                        launch_environment=grouped['launch_environment'])
    packages = dict(zip(_MODES, (process, grouped_view), strict=True))
    for package in packages.values():
        executable = read_artifact(package['files'][package['binary']], json_document=False)
        env = {**recipe['environment'], **package['launch_environment']}
        _need(executable.startswith(b'\x7fELF') and _STATES not in env and _LANES not in env,
              'attached routes must be native executables with diagnostics off during timing')
    case = dict(argv=holdout['argv'], cpu_ids=holdout['cpu_ids'], recipe=recipe_ref, holdout=acceptance['holdout'])
    return case, holdout, recipe, packages


def _validated_runs(runs, mode, package, recipe, holdout):
    return calibration._validate_runs(
        runs, package, recipe, holdout, read_artifact(holdout['normalized'], json_document=False),
        read_artifact(recipe['source_contract'])['report']['parameters']['field'],
        package['output_fields']['worker_pid'], native_label=mode)


def _summarize(runs, pairs, holdout, recipe, packages, *, archived_pairing=None):
    sequence = [(pair, mode) for pair in range(1 + pairs)
                for mode in (_MODES if pair % 2 == 0 else _MODES[::-1])]
    _need([(row['pair'], row['mode']) for row in runs] == sequence, 'alternating clean run order changed')
    rows = []
    for mode, package in packages.items():
        checked, _ = _validated_runs([run for run in runs if run['mode'] == mode], mode, package, recipe, holdout)
        rows.extend(checked)
    context = dict(incumbent=_MODES[0], challenger=_MODES[1], warmup_pairs=1, measurement_pairs=pairs)
    selection = (compare_entry_pairs(rows, **context) if archived_pairing is None else
                 replay_entry_pairs(rows, archived_pairing, **context))
    variance = {mode: pvariance(row['samples_ns']) for mode, row in selection['metrics'].items()}
    return selection, variance


def _launch(output, pair, mode, package, recipe, holdout, *, diagnostic=False):
    override = package['launch_environment']
    env = {**recipe['environment'], **override}
    if diagnostic:
        env[_STATES] = str(output/'state')
        if mode == _MODES[1]:
            env[_LANES] = str(output/'placement.json')
    row = _run(calibration._commands(package, recipe)['native'], holdout['argv'], env, output, pair, mode)
    return dict(row, launch_environment=override, diagnostics_enabled=diagnostic)


def _state_diagnostics(runs, holdout, recipe, packages):
    _need([row['mode'] for row in runs] == list(_MODES), 'missing diagnostic for one of the routes')
    checkpoints = {}
    placement = None
    for run in runs:
        mode = run['mode']
        _need(run['diagnostics_enabled'] is True and run['pair'] == 0, 'invalid private-state diagnostic')
        package = packages[mode]
        _, observations = _validated_runs([run], mode, package, recipe, holdout)
        folder = Path(run['stdout']['uri']).parent
        checkpoints[mode] = _checkpoint_files(folder, package)
        if mode == _MODES[1]:
            reference = _ref(folder/'placement.json', calibration._LANES)
            placement = dict(reference=reference, **calibration._lane_assignments(reference, package, observations[0]))
    first, second = (checkpoints[mode] for mode in _MODES)
    _need(all(left['size_bytes'] == right['size_bytes'] and left['sha256'] == right['sha256']
              for left, right in zip(first, second, strict=True)), 'routes differ at a full round-state checkpoint')
    return dict(runs=runs, checkpoints=checkpoints, exact_checkpoints=len(first),
                exact_bytes=sum(row['size_bytes'] for row in first), grouped_placement=placement)


def observe(retention, grouped, out, *, pairs=7):
    _need(type(pairs) is int and 3 <= pairs <= 32, 'three to thirty-two measured pairs required')
    case, holdout, recipe, packages = _scope(retention, grouped)
    _need(set(case['cpu_ids']) <= sched_getaffinity(0), 'current process cannot run on the recorded CPU mask')
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    runs = [_launch(out/f'{pair:02d}-{mode}', pair, mode, packages[mode], recipe, holdout)
            for pair in range(1 + pairs) for mode in (_MODES if pair % 2 == 0 else _MODES[::-1])]
    selection, variance = _summarize(runs, pairs, holdout, recipe, packages)
    diagnostic_runs = [_launch(out/f'state-{mode}', 0, mode, packages[mode], recipe, holdout, diagnostic=True)
                       for mode in _MODES]
    report = dict(schema=SCHEMA, schema_version=1, retention=retention, grouped=grouped, case=case,
                  runs=runs, pairs=selection, population_variance_ns2=variance,
                  state_diagnostics=_state_diagnostics(diagnostic_runs, holdout, recipe, packages),
                  timing_scope=calibration._TIMING_SCOPE, instrumentation_in_timing=False,
                  target_executed=True, deployment_authorized=False)
    _write(out/'result.json', report)
    return report


def verify_cpu_array_native_route_observation(reference):
    report = read_artifact(reference)
    case, holdout, recipe, packages = _scope(report['retention'], report['grouped'])
    _need(reference['schema'] == SCHEMA and report['schema_version'] == 1 and case == report['case']
          and report['timing_scope'] == calibration._TIMING_SCOPE and report['target_executed'] is True
          and report['deployment_authorized'] is False and report['instrumentation_in_timing'] is False
          and all(row['diagnostics_enabled'] is False for row in report['runs']), 'observation scope changed')
    pairs = report['pairs']['measurement_pairs']
    _need(type(pairs) is int and 3 <= pairs <= 32, 'unsupported clean pair count')
    selection, variance = _summarize(report['runs'], pairs, holdout, recipe, packages,
                                     archived_pairing=report['pairs'])
    states = _state_diagnostics(report['state_diagnostics']['runs'], holdout, recipe, packages)
    _need(selection == report['pairs'] and variance == report['population_variance_ns2']
          and states == report['state_diagnostics'], 'observation projections differ on replay')
    return report


def study(out, *, retention=None, grouped=None, observation=None, pairs=7, replications=()):
    from scar.analysis.cpu_array_native_route import derive_cpu_array_native_route
    from scar.codegen.cpu_native_route_package import export_cpu_array_native_route_package
    from scar.optimization.cpu_array_native_route_plan import derive_cpu_array_native_route_plan

    if observation is not None:
        measured = verify_cpu_array_native_route_observation(observation)
        retention, grouped = measured['retention'], measured['grouped']
    proof = derive_cpu_array_native_route(retention, grouped)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    _write(out/'proof.json', proof)
    if observation is None:
        observe(retention, grouped, out/'observation', pairs=pairs)
        observation = _ref(out/'observation/result.json', SCHEMA)
    plan = derive_cpu_array_native_route_plan(_ref(out/'proof.json', proof['schema']), observation,
                                             replications=replications)
    _write(out/'plan.json', plan)
    return export_cpu_array_native_route_package(_ref(out/'plan.json', plan['schema']), out/'package')


def main(argv=None):
    from scar.codegen import cpu_native_route_package as delivery

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--retention', type=Path)
    mode.add_argument('--observation', type=Path)
    mode.add_argument('--check', type=Path)
    parser.add_argument('--grouped', type=Path)
    parser.add_argument('--replication', action='append', type=Path, default=[],
                        help='additional independent clean campaign on the same held-out entry')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--pairs', type=int, default=7)
    args = parser.parse_args(argv)
    with replay_transaction():
        if args.check is not None:
            if args.grouped is not None or args.out is not None or args.replication:
                parser.error('--check reads an existing route package')
            reference = _ref(args.check, delivery.ARRAY_SCHEMA)
        elif args.out is None:
            parser.error('a fresh --out path is required')
        elif args.observation is not None:
            if args.grouped is not None:
                parser.error('--observation already fixes both routes')
            reference = study(args.out, observation=_ref(args.observation, SCHEMA),
                              replications=tuple(_ref(path, SCHEMA) for path in args.replication))
        elif args.grouped is None:
            parser.error('--retention requires --grouped')
        else:
            reference = study(args.out, retention=_ref(args.retention, _RETENTION),
                              grouped=_ref(args.grouped, _GROUPED), pairs=args.pairs,
                              replications=tuple(_ref(path, SCHEMA) for path in args.replication))
        package = delivery.verify_cpu_array_native_route_package(reference)
    print(json.dumps(dict(package=reference['uri'], status=package['status'],
                          selected_route=package['selected_route'], original_action=package['original_python_action'])))


if __name__ == '__main__':
    raise SystemExit(main())
