"""Resolve a rejected MPI plan to an independently verified local executable."""

import argparse
import json
from pathlib import Path

from scar.compiler import calibrate_array_task_entry as validation
from scar.compiler.calibrate_array_entry import _normalize, _ref, _replay, _write
from scar.ir.execution_recipe import read_artifact
from scar.optimization.paired_entry_selection import compare_entry_pairs


SCHEMA = 'scar.retained-native-array-task-entry'
_PLACEMENT = ('mode', 'worker_count', 'owner_cpu_id', 'worker_cpu_ids', 'launch_cpu_ids', 'window', 'memory')


def _need(condition, detail):
    if not condition:
        raise ValueError('retain native array task entry: ' + detail)


def _same_placement(left, right):
    return all(left[key] == right[key] for key in _PLACEMENT)


def _retained(accepted, incumbent):
    if _same_placement(accepted['selected'], incumbent):
        return accepted['entry'], None, accepted['pairing']
    controls = [(label, row) for label, row in accepted.get('validation_controls', {}).items()
                if _same_placement(row['selected'], incumbent)]
    _need(len(controls) == 1, 'a matched independently tested local incumbent is required')
    (label, control), = controls
    native = read_artifact(accepted['entry'])
    recipe = read_artifact(native['recipe'])
    holdout = read_artifact(accepted['holdout'])
    expected = _normalize(read_artifact(holdout['stdout'], json_document=False), recipe)
    original_pairing = accepted['pairing']
    timing = {key: original_pairing[key] for key in ('warmup_pairs', 'measurement_pairs')}
    control_runs = [_replay(row, recipe, expected) for row in control['clean_pairs']]
    checked = [_replay(row, recipe, expected) for row in accepted['clean_pairs']]
    comparison = compare_entry_pairs([row for row in checked if row['mode'] == 'native'] + control_runs,
        incumbent=label, challenger='native', **timing)
    if comparison['verdict'] == 'ADVANCE_TO_RELEASE_REVIEW':
        return accepted['entry'], comparison, original_pairing
    control_comparison = compare_entry_pairs([row for row in checked if row['mode'] == 'original'] + control_runs,
        incumbent='original', challenger=label, **timing)
    return control['entry'], comparison, control_comparison


def resolve(schedule_path, acceptance_path):
    from scar.optimization import independent_array_task_mpi_schedule as mpi

    schedule_ref = _ref(schedule_path, mpi.SCHEMA)
    plan = mpi.check(schedule_ref['uri'])
    decision = plan['decision']
    _need(decision['all_eligible_mpi_rejected'] is True and decision['selected'] is not None,
          'MPI cost evidence has not rejected every eligible MPI alternative')
    team = read_artifact(plan['team_plan'])
    local = read_artifact(team['process_plan'])
    accepted_ref = _ref(acceptance_path, validation.SCHEMA)
    accepted = validation._check(accepted_ref['uri'], already_replayed_schedule=(plan['team_plan'], team))
    _need(accepted['schedule'] == plan['team_plan'] and _same_placement(accepted['selected'], decision['selected'])
          and accepted['pairing']['verdict'] == 'ADVANCE_TO_RELEASE_REVIEW',
          'independent qualification must match the retained team model and pass its original-entry comparison')
    incumbent = local['decision']['selected']
    _need(incumbent is not None, 'no existing local incumbent in the same task graph')
    release, comparison, original_comparison = _retained(accepted, incumbent)
    _need(original_comparison['verdict'] == 'ADVANCE_TO_RELEASE_REVIEW',
          'preserved local executable has not passed its own original-entry comparison')
    package = read_artifact(release)
    _need(package['scope'] == plan['scope'] and package['schedule'] == plan['team_plan']
          and package['family'] == plan['family'] and package['native'] == plan['native']
          and package['decision']['mode'] != 'ORDERED_LOCAL_MPI' and package['python_runtime_required'] is False,
          'retained binary crosses the graph, proven numerical body, or execution domain')
    matches = [row for row in plan['alternatives'] if _same_placement(row, package['decision']) and row['cost']]
    mpi_rows = [row for row in plan['alternatives']
                if row['mode'] == 'ORDERED_LOCAL_MPI' and not row['capacity_reasons']]
    _need(len(matches) == 1 and mpi_rows and all(row['cost'] for row in mpi_rows),
          'retained implementation and all eligible MPI alternatives require comparable costs')
    local_upper = matches[0]['cost']['modeled_elapsed_ns']['upper_ns']
    mpi_lower = min(row['cost']['modeled_elapsed_ns']['lower_ns'] for row in mpi_rows)
    _need(local_upper < mpi_lower, 'MPI is not separated from the independently qualified local implementation')
    return dict(schema=SCHEMA, schema_version=1, schedule=schedule_ref, existing_acceptance=accepted_ref,
                mpi_decision=decision['status'], retained_release=release, selected=package['decision'],
                comparison_against_local_incumbent=comparison, retained_vs_original=original_comparison,
                modeled_retained_upper_ns=local_upper, smallest_eligible_mpi_lower_ns=mpi_lower,
                reused_binary=package['files'][package['binary']],
                action='RETAIN_VERIFIED_LOCAL_BINARY', executables_generated=0, deployment_authorized=False,
                original_default_retained=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule')
    parser.add_argument('--acceptance')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    if args.check and not any((args.schedule, args.acceptance, args.out)):
        result = read_artifact(_ref(args.check, SCHEMA))
        replay = resolve(result['schedule']['uri'], result['existing_acceptance']['uri'])
        _need(result == replay, 'retained release differs after re-evaluation')
    elif args.schedule and args.acceptance and args.out and args.check is None:
        result = resolve(args.schedule, args.acceptance)
        _write(args.out, result)
    else:
        parser.error('provide --schedule PLAN --acceptance CALIBRATION --out RECEIPT or --check RECEIPT')
    print(json.dumps(dict(action=result['action'], mode=result['selected']['mode'],
                          reused_binary=result['reused_binary']['uri'])))


if __name__ == '__main__':
    main()
