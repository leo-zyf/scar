"""Build and validate grouped floating-point environment checks."""

import argparse
import json
from pathlib import Path

from scar.analysis.replay_transaction import replay_transaction
from scar.compiler import calibrate_array_task_entry as calibration
from scar.compiler import cpu_task_arena_study as validation
from scar.compiler.calibrate_array_entry import _ref, _write
from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.cpu-fp-guard-study'
_MODES = ('per_step', 'grouped_guards')


def _need(ok, reason):
    if not ok:
        raise ValueError('CPU FP guard study: ' + reason)


def study(acceptance, arena_proof, out, *, native_pairs=7, original_pairs=3):
    from scar.analysis.cpu_fp_guard_groups import derive_cpu_fp_guard_groups
    from scar.codegen.native_cpu_fp_guard_groups import (
        export_native_cpu_fp_guard_groups, verify_native_cpu_fp_guard_groups,
    )
    from scar.optimization.cpu_fp_guard_plan import derive_cpu_fp_guard_plan

    _need(all(type(value) is int and 3 <= value <= 32 for value in (native_pairs, original_pairs)),
          'three to thirty-two measured pairs required per campaign')
    out = Path(out).absolute()
    out.mkdir(parents=True, exist_ok=False)
    with replay_transaction():
        parent, runtime, recipe, case, expected = validation._source(acceptance)
        proof = derive_cpu_fp_guard_groups(arena_proof)
        _need(proof['runtime'] == parent, 'FP guard proof belongs to a different native task provider')
        _write(out/'proof.json', proof)
        proof_ref = _ref(out/'proof.json', proof['schema'])
        plan = derive_cpu_fp_guard_plan(proof_ref)
        _write(out/'plan.json', plan)
        plan_ref = _ref(out/'plan.json', plan['schema'])
        review = dict(schema=SCHEMA, schema_version=1, acceptance=acceptance, proof=proof_ref, plan=plan_ref,
                      default_action='KEEP_ORIGINAL', deployment_authorized=False)
        if plan['status'] == 'REJECTED':
            return dict(review, status='NO_APPLICABLE_FP_GUARD_GROUPS', target_executed=False)
        variant_ref = export_native_cpu_fp_guard_groups(plan_ref, out/'native')
        variant = verify_native_cpu_fp_guard_groups(variant_ref)
        _need(variant['runtime'] == parent and variant['plan'] == plan_ref,
              'emitted FP guard variant differs from its reviewed program')
    measurements = validation._measure_candidate(out, _MODES, runtime, recipe, case, expected,
                                                  variant, native_pairs, original_pairs)
    return dict(review, variant=variant_ref, **measurements,
                status='VALIDATED_SCOPED_FP_GUARD_GROUPS', target_executed=True)


def check(path):
    from scar.codegen.native_cpu_fp_guard_groups import verify_native_cpu_fp_guard_groups
    from scar.optimization.cpu_fp_guard_plan import verify_cpu_fp_guard_plan

    report = read_artifact(_ref(path, SCHEMA))
    _need(report['schema_version'] == 1 and report['default_action'] == 'KEEP_ORIGINAL'
          and report['deployment_authorized'] is False, 'unsupported study authority')
    with replay_transaction():
        plan = verify_cpu_fp_guard_plan(report['plan'])
        _need(plan['proof'] == report['proof'], 'study references a different FP guard proof')
        parent, runtime, recipe, case, expected = validation._source(report['acceptance'])
        _need(plan['runtime'] == parent, 'study references a different original entry')
        if report['status'] == 'NO_APPLICABLE_FP_GUARD_GROUPS':
            _need(plan['status'] == 'REJECTED' and report['target_executed'] is False,
                  'rejected study must execute no target')
            return report
        variant = verify_native_cpu_fp_guard_groups(report['variant'])
        _need(variant['runtime'] == parent and variant['plan'] == report['plan']
              and report['cpu_ids'] == runtime['cpu_ids'] and report['argv'] == case['argv']
              and report['status'] == 'VALIDATED_SCOPED_FP_GUARD_GROUPS' and report['target_executed'] is True,
              'study references a different executable, option or invocation')
        commands = validation._commands(runtime, recipe, variant, _MODES)
    validation._check_measurements(report, _MODES, runtime, recipe, case, expected, commands)
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
        if args.arena_proof is not None or args.out is not None:
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
