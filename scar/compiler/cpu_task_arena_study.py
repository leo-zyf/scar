"""Build and verify a fixed-lane CPU arena candidate against its original entry."""

import argparse
import json
from pathlib import Path
from statistics import pvariance

from scar.analysis.replay_transaction import replay_transaction
from scar.compiler import calibrate_array_task_entry as calibration
from scar.compiler.calibrate_array_entry import _ref, _replay, _run, _write
from scar.compiler.cpu_entry_replay_adapters import _match_json_report
from scar.ir.execution_recipe import read_artifact
from scar.optimization.paired_entry_selection import compare_entry_pairs


SCHEMA = 'scar.cpu-task-arena-study'
_MODES = ('original', 'task_private', 'lane_reuse')
_RAW = 'application/octet-stream'
_CHECKPOINT = 'scar.ndarray-native-checkpoint'
_PLACEMENT = 'scar.native-array-openmp-lanes'


def _need(ok, reason):
    if not ok:
        raise ValueError('CPU task arena study: ' + reason)


def _source(acceptance):
    admitted = read_artifact(acceptance)
    parent, runtime, recipe, prior, parameters, pid = calibration._load(admitted['entry']['uri'])
    _need(acceptance['schema'] == calibration.SCHEMA and admitted['schema_version'] == 1
          and parent == admitted['entry'] and admitted['selected'] == runtime['decision']
          and admitted['schedule'] == runtime['schedule'] and admitted['cpu_ids'] == runtime['cpu_ids']
          and admitted['deployment_authorized'] is False
          and admitted['pairing']['verdict'] == 'ADVANCE_TO_RELEASE_REVIEW',
          'retained original-entry acceptance differs from the task provider')
    case, expected = calibration._holdout(admitted['holdout'], runtime, recipe, prior, parameters, pid)
    return parent, runtime, recipe, case, expected


def _commands(runtime, recipe, variant):
    raw = calibration._commands(runtime, recipe)
    binary = variant['executable']
    _need(read_artifact(binary, json_document=False).startswith(b'\x7fELF'), 'candidate needs a sealed native executable')
    return dict(original=raw['original'], task_private=raw['native'], lane_reuse=[*raw['native'][:-1], binary['uri']])


def _summarize(runs, modes, commands, argv, recipe, expected, pairs):
    order = [(index, mode) for index in range(1 + pairs)
             for mode in (modes if index % 2 == 0 else modes[::-1])]
    _need([(row['pair'], row['mode']) for row in runs] == order, 'A/B order differs from the frozen campaign')
    compared = []
    for row in runs:
        _need(row['command'] == commands[row['mode']] and row['argv'] == argv
              and _match_json_report(read_artifact(row['stdout'], json_document=False),
                  read_artifact(row['published'], json_document=False), recipe, expected),
              'whole-entry output or launch differs')
        compared.append(_replay(row, recipe, expected))
    result = compare_entry_pairs(compared, incumbent=modes[0], challenger=modes[1],
                               warmup_pairs=1, measurement_pairs=pairs)
    return dict(selection=result, population_variance_ns2={
        mode: pvariance(values['samples_ns']) for mode, values in result['metrics'].items()})


def _campaign(folder, modes, pairs, commands, argv, environment, native_environment, recipe, expected):
    folder.mkdir()
    runs = []
    for pair in range(1 + pairs):
        for mode in modes if pair % 2 == 0 else modes[::-1]:
            env = environment if mode == 'original' else native_environment
            runs.append(_run(commands[mode], argv, env, folder/f'{pair:02d}-{mode}', pair, mode))
    return dict(runs=runs, **_summarize(runs, modes, commands, argv, recipe, expected, pairs))


def _checkpoint_files(folder, runtime):
    scope = runtime['scope']
    return [_ref(folder/f'state.{task}.{round_}.bin', _CHECKPOINT)
            for task in range(scope['task_count']) for round_ in range(scope['round_count'])]


def _compare_checkpoints(rows, runtime, commands, argv, recipe, expected):
    _need([row['run']['mode'] for row in rows] == list(_MODES[1:]), 'expected both native state diagnostics')
    for row in rows:
        run = row['run']
        _need(run['command'] == commands[run['mode']] and run['argv'] == argv
              and _match_json_report(read_artifact(run['stdout'], json_document=False),
                  read_artifact(run['published'], json_document=False), recipe, expected),
              'native diagnostic changed the core report or invocation')
        _replay(run, recipe, expected)
        folder = Path(run['stdout']['uri']).parent
        _need(row['checkpoints'] == _checkpoint_files(folder, runtime)
              and row['placement'] == _ref(folder/'placement.json', _PLACEMENT),
              'native diagnostic material differs at replay')
        observation = dict(active_worker_pids=json.loads(read_artifact(run['stdout'], json_document=False))[
            runtime['output_fields']['pid_list']])
        _need(calibration._lane_assignments(row['placement'], runtime, observation) == row['placement_check'],
              'observed task affinity or lane changed')
    control, candidate = (row['checkpoints'] for row in rows)
    _need(all(left['size_bytes'] == right['size_bytes'] and left['sha256'] == right['sha256']
              for left, right in zip(control, candidate, strict=True)), 'arena reuse changed a round-state checkpoint')
    return dict(checkpoints=len(control), exact_bytes=sum(row['size_bytes'] for row in control))


def _diagnostics(folder, commands, argv, environment, runtime, recipe, expected):
    folder.mkdir()
    runs = []
    for mode in _MODES[1:]:
        output = folder/mode
        env = dict(environment, SCAR_ARRAY_ENTRY_STATES=str(output/'state'),
                   SCAR_ARRAY_ENTRY_LANE_REPORT=str(output/'placement.json'))
        run = _run(commands[mode], argv, env, output, 0, mode)
        placement = _ref(output/'placement.json', _PLACEMENT)
        observation = dict(active_worker_pids=json.loads(read_artifact(run['stdout'], json_document=False))[
            runtime['output_fields']['pid_list']])
        runs.append(dict(run=run, placement=placement, checkpoints=_checkpoint_files(output, runtime),
                         placement_check=calibration._lane_assignments(placement, runtime, observation)))
    return dict(runs=runs, **_compare_checkpoints(runs, runtime, commands, argv, recipe, expected))


def study(acceptance, out, *, native_pairs=5, original_pairs=3):
    from scar.analysis.cpu_task_arena_reuse import derive_cpu_task_arena_reuse
    from scar.codegen.native_cpu_arena_reuse import export_native_cpu_arena_reuse, verify_native_cpu_arena_reuse
    from scar.optimization.cpu_task_arena_plan import derive_cpu_task_arena_plan

    _need(all(type(item) is int and 3 <= item <= 32 for item in (native_pairs, original_pairs)),
          'three to thirty-two measured pairs required for each campaign')
    out = Path(out).absolute()
    out.mkdir(parents=True, exist_ok=False)
    with replay_transaction():
        parent, runtime, recipe, case, expected = _source(acceptance)
        proof = derive_cpu_task_arena_reuse(parent)
        _write(out/'proof.json', proof)
        proof_ref = _ref(out/'proof.json', proof['schema'])
        plan = derive_cpu_task_arena_plan(proof_ref)
        _write(out/'plan.json', plan)
        if plan['status'] != 'ELIGIBLE_FOR_EXPLICIT_VALIDATION':
            return dict(schema=SCHEMA, schema_version=1, acceptance=acceptance,
                proof=proof_ref, plan=_ref(out/'plan.json', plan['schema']),
                status='NO_APPLICABLE_ARENA_REWRITE', target_executed=False,
                default_action='KEEP_ORIGINAL', deployment_authorized=False)
        variant_ref = export_native_cpu_arena_reuse(_ref(out/'plan.json', plan['schema']), out/'native')
        variant = verify_native_cpu_arena_reuse(variant_ref)
        _need(variant['runtime'] == parent, 'candidate changed the original native task provider')
        commands = _commands(runtime, recipe, variant)
    environment = recipe['environment']
    native_env = dict(environment, **runtime['launch_environment'])
    _need(not any(name in native_env for name in ('SCAR_ARRAY_ENTRY_STATES', 'SCAR_ARRAY_ENTRY_LANE_REPORT')),
          'clean benchmark needs diagnostic capture disabled')
    argv = case['argv']
    states = _diagnostics(out/'state-check', commands, argv, native_env, runtime, recipe, expected)
    native = _campaign(out/'native-pairs', _MODES[1:], native_pairs, commands, argv, environment,
                       native_env, recipe, expected)
    original = _campaign(out/'original-pairs', (_MODES[0], _MODES[2]), original_pairs, commands, argv,
                         environment, native_env, recipe, expected)
    return dict(schema=SCHEMA, schema_version=1, acceptance=acceptance, proof=proof_ref,
        plan=_ref(out/'plan.json', plan['schema']), variant=variant_ref, cpu_ids=runtime['cpu_ids'],
        argv=argv, native_pairs=native, original_pairs=original, state_diagnostics=states,
        status='VALIDATED_SCOPED_ARENA_REWRITE', target_executed=True, default_action='KEEP_ORIGINAL',
        deployment_authorized=False)


def check(path):
    from scar.codegen.native_cpu_arena_reuse import verify_native_cpu_arena_reuse
    from scar.optimization.cpu_task_arena_plan import verify_cpu_task_arena_plan

    report = read_artifact(_ref(path, SCHEMA))
    _need(report['schema_version'] == 1 and report['default_action'] == 'KEEP_ORIGINAL'
          and report['deployment_authorized'] is False, 'unsupported study authority')
    with replay_transaction():
        plan = verify_cpu_task_arena_plan(report['plan'])
        _need(plan['proof'] == report['proof'], 'study references a different source proof')
        parent, runtime, recipe, case, expected = _source(report['acceptance'])
        _need(plan['runtime'] == parent, 'study references a different original entry')
        if report['status'] == 'NO_APPLICABLE_ARENA_REWRITE':
            _need(plan['status'] == 'REJECTED' and report['target_executed'] is False,
                  'rejected study must retain the original and execute no target')
            return report
        variant = verify_native_cpu_arena_reuse(report['variant'])
        _need(variant['runtime'] == parent and variant['plan'] == report['plan']
              and report['cpu_ids'] == runtime['cpu_ids'] and report['argv'] == case['argv']
              and report['status'] == 'VALIDATED_SCOPED_ARENA_REWRITE' and report['target_executed'] is True,
              'study references a different executable, option or invocation')
        commands = _commands(runtime, recipe, variant)
    for name, modes in (('native_pairs', _MODES[1:]), ('original_pairs', (_MODES[0], _MODES[2]))):
        data = report[name]
        compared = _summarize(data['runs'], modes, commands, case['argv'], recipe, expected,
                              data['selection']['measurement_pairs'])
        _need(all(data[key] == value for key, value in compared.items()), 'study timing receipts differ at replay')
    diagnostic = report['state_diagnostics']
    states = _compare_checkpoints(diagnostic['runs'], runtime, commands, case['argv'], recipe, expected)
    _need(all(diagnostic[key] == value for key, value in states.items()), 'study state comparison differs at replay')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--acceptance', type=Path)
    mode.add_argument('--check', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--native-pairs', type=int, default=5)
    parser.add_argument('--original-pairs', type=int, default=3)
    args = parser.parse_args(argv)
    if args.check is not None:
        if args.out is not None:
            parser.error('--check reads the archived study')
        print(json.dumps(dict(result=str(args.check), status=check(args.check)['status'], replayed=True)))
        return
    if args.out is None:
        parser.error('--acceptance requires --out')
    report = study(_ref(args.acceptance, calibration.SCHEMA), args.out,
                   native_pairs=args.native_pairs, original_pairs=args.original_pairs)
    _write(args.out/'result.json', report)
    print(json.dumps(dict(result=str(args.out/'result.json'), status=report['status'],
                          candidate_executed=report['target_executed'])))
