"""Measure fixed-shape arithmetic dispatch on an admitted native CPU entry."""

import argparse
import json
from pathlib import Path

from scar.analysis.replay_transaction import replay_transaction
from scar.codegen import native_cpu_fp_guard_groups as grouped
from scar.compiler import calibrate_array_task_entry as calibration
from scar.compiler import cpu_task_arena_study as validation
from scar.compiler.calibrate_array_entry import _ref, _write
from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.cpu-arithmetic-dispatch-study'
_BASELINE = 'grouped_guards'
_ORIGINAL = 'per_step'
_CANDIDATE = 'fixed_shapes'
_CAMPAIGNS = {'dispatch_pairs': (_BASELINE, _CANDIDATE), 'combined_pairs': (_ORIGINAL, _CANDIDATE)}


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU arithmetic dispatch study: ' + detail)


def _commands(runtime, recipe, baseline, variant, candidate=_CANDIDATE):
    commands = validation._commands(runtime, recipe, variant, (_ORIGINAL, candidate))
    binary = baseline['executable']
    _need(read_artifact(binary, json_document=False).startswith(b'\x7fELF'), 'grouped baseline needs an exact ELF')
    commands[_BASELINE] = [*commands[_ORIGINAL][:-1], binary['uri']]
    return commands


def _scope(acceptance, baseline_ref):
    parent, runtime, recipe, case, expected = validation._source(acceptance)
    baseline = read_artifact(baseline_ref)
    _need(baseline_ref['schema'] == grouped.SCHEMA and baseline['runtime'] == parent
          and baseline['scope'] == runtime['scope'] and baseline['cpu_ids'] == runtime['cpu_ids'],
          'grouped baseline differs from the original native entry')
    return runtime, recipe, case, expected, baseline


def _environment(runtime, recipe):
    environment = dict(recipe['environment'], **runtime['launch_environment'])
    _need(not set(environment).intersection(('SCAR_ARRAY_ENTRY_STATES', 'SCAR_ARRAY_ENTRY_LANE_REPORT')),
          'clean timing requires the original diagnostics to be off')
    return environment


def _measure_comparisons(out, commands, runtime, recipe, case, expected, pairs, campaigns):
    environment, argv = _environment(runtime, recipe), case['argv']
    states = validation._diagnostics(out/'state-check', commands, argv, environment, runtime, recipe, expected,
                                     next(iter(campaigns.values())))
    measured = {
        name: validation._campaign(out/name.replace('_', '-'), modes, pairs, commands, argv,
                                   environment, environment, recipe, expected)
        for name, modes in campaigns.items()
    }
    return dict(state_diagnostics=states, **measured)


def _check_comparisons(report, commands, runtime, recipe, case, expected, campaigns):
    states = report['state_diagnostics']
    expected_states = validation._compare_checkpoints(states['runs'], runtime, commands, case['argv'], recipe,
                                                    expected, next(iter(campaigns.values())))
    _need(all(states[key] == value for key, value in expected_states.items()), 'state replay differs')
    for name, modes in campaigns.items():
        rows = report[name]
        pairs = rows['selection']['measurement_pairs']
        expected_rows = validation._summarize(rows['runs'], modes, commands, case['argv'], recipe, expected, pairs)
        _need(all(rows[key] == value for key, value in expected_rows.items()), 'native performance replay differs')


def study(acceptance, baseline_ref, out, *, pairs=7):
    from scar.analysis.cpu_arithmetic_dispatch import derive_cpu_arithmetic_dispatch
    from scar.codegen.native_cpu_arithmetic_dispatch import export_native_cpu_arithmetic_dispatch
    from scar.optimization.cpu_arithmetic_dispatch_plan import derive_cpu_arithmetic_dispatch_plan

    _need(type(pairs) is int and 3 <= pairs <= 32, 'three to thirty-two measured pairs required')
    out = Path(out).absolute()
    out.mkdir(parents=True, exist_ok=False)
    with replay_transaction():
        runtime, recipe, case, expected, baseline = _scope(acceptance, baseline_ref)
        proof = derive_cpu_arithmetic_dispatch(baseline_ref)
        _write(out/'proof.json', proof)
        proof_ref = _ref(out/'proof.json', proof['schema'])
        plan = derive_cpu_arithmetic_dispatch_plan(proof_ref)
        _write(out/'plan.json', plan)
        plan_ref = _ref(out/'plan.json', plan['schema'])
        variant_ref = export_native_cpu_arithmetic_dispatch(plan_ref, out/'native')
        variant = read_artifact(variant_ref)
        _need(variant['baseline'] == baseline_ref and variant['plan'] == plan_ref,
              'released members differ from the planned shapes')
        commands = _commands(runtime, recipe, baseline, variant)
    comparisons = _measure_comparisons(out, commands, runtime, recipe, case, expected, pairs, _CAMPAIGNS)
    return dict(schema=SCHEMA, schema_version=1, status='VALIDATED_SCOPED_ARITHMETIC_DISPATCH',
                acceptance=acceptance, baseline=baseline_ref, proof=proof_ref, plan=plan_ref, variant=variant_ref,
                argv=case['argv'], cpu_ids=runtime['cpu_ids'], **comparisons,
                target_executed=True, default_action='KEEP_ORIGINAL', deployment_authorized=False)


def check(path):
    from scar.codegen.native_cpu_arithmetic_dispatch import verify_native_cpu_arithmetic_dispatch

    report = read_artifact(_ref(path, SCHEMA))
    _need(report['schema_version'] == 1 and report['status'] == 'VALIDATED_SCOPED_ARITHMETIC_DISPATCH'
          and report['target_executed'] is True and report['default_action'] == 'KEEP_ORIGINAL'
          and report['deployment_authorized'] is False, 'unsupported study authority')
    with replay_transaction():
        runtime, recipe, case, expected, baseline = _scope(report['acceptance'], report['baseline'])
        variant = verify_native_cpu_arithmetic_dispatch(report['variant'])
        plan = read_artifact(variant['plan'])
        _need(variant['baseline'] == report['baseline'] and variant['plan'] == report['plan']
              and plan['proof'] == report['proof'] and report['argv'] == case['argv']
              and report['cpu_ids'] == runtime['cpu_ids'], 'confirmed program, plan or argv differs')
        commands = _commands(runtime, recipe, baseline, variant)
    _check_comparisons(report, commands, runtime, recipe, case, expected, _CAMPAIGNS)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--acceptance', type=Path)
    mode.add_argument('--check', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--pairs', type=int, default=7)
    args = parser.parse_args(argv)
    if args.check is not None:
        if args.baseline is not None or args.out is not None:
            parser.error('--check reads an existing study')
        report = check(args.check)
        print(json.dumps(dict(status=report['status'], result=str(args.check), replayed=True)))
        return
    if args.baseline is None or args.out is None:
        parser.error('--acceptance requires --baseline and --out')
    report = study(_ref(args.acceptance, calibration.SCHEMA), _ref(args.baseline, grouped.SCHEMA), args.out,
                   pairs=args.pairs)
    _write(args.out/'result.json', report)
    print(json.dumps(dict(status=report['status'], result=str(args.out/'result.json'), target_executed=True)))


if __name__ == '__main__':
    raise SystemExit(main())
