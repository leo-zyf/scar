"""Validate one-loop coefficient dispatch against two admitted native CPU paths."""

import argparse
import json
from pathlib import Path

from scar.analysis.replay_transaction import replay_transaction
from scar.codegen import native_cpu_fp_guard_groups as grouped
from scar.compiler import calibrate_array_task_entry as calibration
from scar.compiler import cpu_arithmetic_dispatch_study as comparison
from scar.compiler.calibrate_array_entry import _ref, _write
from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.cpu-coefficient-dispatch-study'
_CANDIDATE = 'single_loop'
_CAMPAIGNS = {'dispatch_pairs': ('grouped_guards', _CANDIDATE), 'combined_pairs': ('per_step', _CANDIDATE)}


def _need(ok, reason):
    if not ok:
        raise ValueError('CPU coefficient dispatch study: ' + reason)


def study(acceptance, baseline_ref, out, *, pairs=7):
    from scar.analysis.cpu_coefficient_dispatch import derive_cpu_coefficient_dispatch
    from scar.codegen.native_cpu_coefficient_dispatch import export_native_cpu_coefficient_dispatch
    from scar.optimization.cpu_coefficient_dispatch_plan import derive_cpu_coefficient_dispatch_plan

    _need(type(pairs) is int and 3 <= pairs <= 32, 'three to thirty-two measured pairs required')
    out = Path(out).absolute()
    out.mkdir(parents=True, exist_ok=False)
    with replay_transaction():
        runtime, recipe, case, expected, baseline = comparison._scope(acceptance, baseline_ref)
        proof = derive_cpu_coefficient_dispatch(baseline_ref)
        _write(out/'proof.json', proof)
        proof_ref = _ref(out/'proof.json', proof['schema'])
        plan = derive_cpu_coefficient_dispatch_plan(proof_ref)
        _write(out/'plan.json', plan)
        plan_ref = _ref(out/'plan.json', plan['schema'])
        variant_ref = export_native_cpu_coefficient_dispatch(plan_ref, out/'native')
        variant = read_artifact(variant_ref)
        _need(variant['baseline'] == baseline_ref and variant['plan'] == plan_ref,
              'emitted project differs from the proved baseline or fusion plan')
        commands = comparison._commands(runtime, recipe, baseline, variant, _CANDIDATE)
    measurements = comparison._measure_comparisons(out, commands, runtime, recipe, case, expected, pairs, _CAMPAIGNS)
    return dict(schema=SCHEMA, schema_version=1, status='VALIDATED_SCOPED_COEFFICIENT_DISPATCH',
                acceptance=acceptance, baseline=baseline_ref, proof=proof_ref, plan=plan_ref, variant=variant_ref,
                argv=case['argv'], cpu_ids=runtime['cpu_ids'], **measurements, target_executed=True,
                default_action='KEEP_ORIGINAL', deployment_authorized=False)


def check(path):
    from scar.codegen.native_cpu_coefficient_dispatch import verify_native_cpu_coefficient_dispatch

    report = read_artifact(_ref(path, SCHEMA))
    _need(report['schema_version'] == 1 and report['status'] == 'VALIDATED_SCOPED_COEFFICIENT_DISPATCH'
          and report['target_executed'] is True and report['default_action'] == 'KEEP_ORIGINAL'
          and report['deployment_authorized'] is False, 'unsupported study authority')
    with replay_transaction():
        runtime, recipe, case, expected, baseline = comparison._scope(report['acceptance'], report['baseline'])
        variant = verify_native_cpu_coefficient_dispatch(report['variant'])
        plan = read_artifact(variant['plan'])
        _need(variant['baseline'] == report['baseline'] and variant['plan'] == report['plan']
              and plan['proof'] == report['proof'] and report['argv'] == case['argv']
              and report['cpu_ids'] == runtime['cpu_ids'], 'replayed program or invocation differs')
        commands = comparison._commands(runtime, recipe, baseline, variant, _CANDIDATE)
    comparison._check_comparisons(report, commands, runtime, recipe, case, expected, _CAMPAIGNS)
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
