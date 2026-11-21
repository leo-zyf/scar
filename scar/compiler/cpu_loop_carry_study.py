"""Build and test a proved steady-loop carry redirection."""

import argparse
import json
from pathlib import Path

from scar.analysis.replay_transaction import replay_transaction
from scar.compiler import calibrate_array_task_entry as calibration
from scar.compiler import cpu_task_arena_study as validation
from scar.compiler.calibrate_array_entry import _ref, _write
from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.cpu-loop-carry-study'
_MODES = ('carry_copy', 'carry_redirect')
_COUNTS = ('copies', 'same_span_transfers', 'redirected')
_HANDOFF = 'application/json'


def _need(ok, reason):
    if not ok:
        raise ValueError('CPU loop carry study: ' + reason)


def _handoffs(diagnostics, scope):
    records = {}
    for row in diagnostics['runs']:
        run = row['run']
        folder = Path(run['stdout']['uri']).parent
        items = []
        for ordinal in range(scope['task_count']):
            fail = 'invalid handoff counts for task ' + str(ordinal)
            path = folder/f'state.{ordinal}.handoffs.json'
            reference = _ref(path, _HANDOFF)
            data = json.loads(read_artifact(reference, json_document=False))
            _need(set(data) in (set(_COUNTS[:2]), set(_COUNTS))
                  and all(type(value) is int and value >= 0 for value in data.values())
                  and sum(data.values()) == scope['round_count'] - 1, fail)
            counts = {key: data.get(key, 0) for key in _COUNTS}
            items.append(dict(ordinal=ordinal, artifact=reference, counts=counts))
        records[run['mode']] = items
    previous, replaced = (records[mode] for mode in _MODES)
    for left, right in zip(previous, replaced, strict=True):
        lhs, rhs = left['counts'], right['counts']
        _need(lhs['redirected'] == 0 and rhs['copies'] + rhs['redirected'] == lhs['copies']
              and lhs['same_span_transfers'] == rhs['same_span_transfers'],
              'candidate changed the number or class of observed steady transitions')
    return dict(per_task=records, totals={mode: {key: sum(row['counts'][key] for row in items)
                for key in _COUNTS} for mode, items in records.items()})


def study(acceptance, arena_proof, out, *, native_pairs=7, original_pairs=3):
    from scar.analysis.cpu_loop_carry_redirect import derive_cpu_loop_carry_redirect
    from scar.codegen.native_cpu_loop_carry import export_native_cpu_loop_carry, verify_native_cpu_loop_carry
    from scar.optimization.cpu_loop_carry_plan import derive_cpu_loop_carry_plan

    _need(all(type(value) is int and 3 <= value <= 32 for value in (native_pairs, original_pairs)),
          'three to thirty-two measured pairs required per campaign')
    out = Path(out).absolute()
    out.mkdir(parents=True, exist_ok=False)
    with replay_transaction():
        parent, runtime, recipe, case, expected = validation._source(acceptance)
        proof = derive_cpu_loop_carry_redirect(arena_proof)
        _need(proof['runtime'] == parent, 'carry proof belongs to a different native task provider')
        _write(out/'proof.json', proof)
        proof_ref = _ref(out/'proof.json', proof['schema'])
        plan = derive_cpu_loop_carry_plan(proof_ref)
        _write(out/'plan.json', plan)
        plan_ref = _ref(out/'plan.json', plan['schema'])
        review = dict(schema=SCHEMA, schema_version=1, acceptance=acceptance,
                      proof=proof_ref, plan=plan_ref, default_action='KEEP_ORIGINAL', deployment_authorized=False)
        if plan['status'] == 'REJECTED':
            return dict(review, status='NO_APPLICABLE_CARRY_REDIRECT', target_executed=False)
        variant_ref = export_native_cpu_loop_carry(plan_ref, out/'native')
        variant = verify_native_cpu_loop_carry(variant_ref)
        _need(variant['runtime'] == parent and variant['plan'] == plan_ref,
              'emitted carry variant differs from its reviewed program')
        commands = validation._commands(runtime, recipe, variant, _MODES)
    environment = recipe['environment']
    native_env = dict(environment, **runtime['launch_environment'])
    _need(not any(name in native_env for name in ('SCAR_ARRAY_ENTRY_STATES', 'SCAR_ARRAY_ENTRY_LANE_REPORT')),
          'clean benchmark needs diagnostic capture disabled')
    argv = case['argv']
    states = validation._diagnostics(out/'state-check', commands, argv, native_env, runtime, recipe, expected, _MODES)
    handoffs = _handoffs(states, runtime['scope'])
    native = validation._campaign(out/'native-pairs', _MODES, native_pairs, commands, argv, environment,
                                  native_env, recipe, expected)
    original = validation._campaign(out/'original-pairs', ('original', _MODES[1]), original_pairs, commands, argv,
                                    environment, native_env, recipe, expected)
    return dict(review, variant=variant_ref, cpu_ids=runtime['cpu_ids'], argv=argv,
                native_pairs=native, original_pairs=original, state_diagnostics=states, handoffs=handoffs,
                status='VALIDATED_SCOPED_CARRY_REDIRECT', target_executed=True)


def check(path):
    from scar.codegen.native_cpu_loop_carry import verify_native_cpu_loop_carry
    from scar.optimization.cpu_loop_carry_plan import verify_cpu_loop_carry_plan

    report = read_artifact(_ref(path, SCHEMA))
    _need(report['schema_version'] == 1 and report['default_action'] == 'KEEP_ORIGINAL'
          and report['deployment_authorized'] is False, 'unsupported study authority')
    with replay_transaction():
        plan = verify_cpu_loop_carry_plan(report['plan'])
        _need(plan['proof'] == report['proof'], 'study references a different carry proof')
        parent, runtime, recipe, case, expected = validation._source(report['acceptance'])
        _need(plan['runtime'] == parent, 'study references a different original entry')
        if report['status'] == 'NO_APPLICABLE_CARRY_REDIRECT':
            _need(plan['status'] == 'REJECTED' and report['target_executed'] is False,
                  'rejected study must execute no target')
            return report
        variant = verify_native_cpu_loop_carry(report['variant'])
        _need(variant['runtime'] == parent and variant['plan'] == report['plan']
              and report['cpu_ids'] == runtime['cpu_ids'] and report['argv'] == case['argv']
              and report['status'] == 'VALIDATED_SCOPED_CARRY_REDIRECT' and report['target_executed'] is True,
              'study references a different executable, option or invocation')
        commands = validation._commands(runtime, recipe, variant, _MODES)
    for name, modes in (('native_pairs', _MODES), ('original_pairs', ('original', _MODES[1]))):
        data = report[name]
        replay = validation._summarize(data['runs'], modes, commands, case['argv'], recipe, expected,
                                      data['selection']['measurement_pairs'])
        _need(all(data[key] == value for key, value in replay.items()), 'study timing receipts differ at replay')
    diagnostics = report['state_diagnostics']
    comparison = validation._compare_checkpoints(diagnostics['runs'], runtime, commands, case['argv'],
                                                  recipe, expected, _MODES)
    _need(all(diagnostics[key] == value for key, value in comparison.items())
          and report['handoffs'] == _handoffs(diagnostics, runtime['scope']), 'steady state receipts differ at replay')
    return report


def main(argv=None):
    from scar.analysis.cpu_task_arena_reuse import SCHEMA as ARENA_PROOF

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--acceptance', type=Path)
    mode.add_argument('--check', type=Path)
    parser.add_argument('--arena-proof', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--native-pairs', type=int, default=7)
    parser.add_argument('--original-pairs', type=int, default=3)
    args = parser.parse_args(argv)
    if args.check is not None:
        if args.out is not None or args.arena_proof is not None:
            parser.error('--check reads the archived study')
        print(json.dumps(dict(result=str(args.check), status=check(args.check)['status'], replayed=True)))
        return
    if args.out is None or args.arena_proof is None:
        parser.error('--acceptance requires --arena-proof and --out')
    report = study(_ref(args.acceptance, calibration.SCHEMA), _ref(args.arena_proof, ARENA_PROOF), args.out,
                   native_pairs=args.native_pairs, original_pairs=args.original_pairs)
    _write(args.out/'result.json', report)
    print(json.dumps(dict(result=str(args.out/'result.json'), status=report['status'],
                          candidate_executed=report['target_executed'])))
