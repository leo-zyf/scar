"""Check an independent in-flight window against a matching diagonal observation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scar.codegen import native_ordered_timing as diagnostics
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.native_ordered_transport_cost import _observed_holdout
from scar.optimization.native_ordered_width_cost import _bind_observed_profiles


SCHEMA = 'scar.native-ordered-independent-window-validation'


def _require(valid, reason):
    if not valid:
        raise ValueError(reason)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _measured_arm(observation, profile):
    memory = profile['resources']['memory']
    return dict(workers=observation['width'], window=observation.get('window', observation['width']),
        owner_cpu_id=observation['placement']['owner_cpu_id'],
        worker_cpu_ids=observation['placement']['worker_cpu_ids'],
        worker_solve_counts={str(row['cpu_start']): row['phases']['solve']['count']
            for row in observation['timing']['workers']}, live_credits=observation['timing'].get('credits'),
        minimum_launch_headroom_bytes=memory['minimum_launch_headroom_bytes'],
        summed_worker_address_space_limit_bytes=memory['worker_address_space_sum_bytes'],
        conservative_owner_pending_bytes=memory['owner_pending_input_bytes']+memory['owner_pending_result_bound_bytes'],
        diagnostic_observation=_observed_holdout(observation),
        one_production_correctness_launch_ns=observation['production']['wall_ns'])


def compare(diagonal_path, independent_path):
    old_ref, new_ref = (_ref(path, diagnostics.SCHEMA) for path in (diagonal_path, independent_path))
    before, after = (diagnostics.check(ref['uri']) for ref in (old_ref, new_ref))
    _require(before == read_artifact(old_ref) and after == read_artifact(new_ref)
        and before['schema_version'] == 1 and after['schema_version'] == 2
        and before['width'] == after['window'] < after['width']
        and after['placement']['worker_cpu_ids'][:before['width']] == before['placement']['worker_cpu_ids'],
        'comparison requires the same typed window and a wider checked cohort')
    before_ref, after_ref = (read_artifact(row['package'])['profile_plan'] for row in (before, after))
    old, new = (read_artifact(ref) for ref in (before_ref, after_ref))
    matched = _bind_observed_profiles([before, after], before_ref, old, allow_independent_windows=True)
    first, second = _measured_arm(before, old), _measured_arm(after, new)
    address_delta = second['summed_worker_address_space_limit_bytes'] - first['summed_worker_address_space_limit_bytes']
    _require(first['conservative_owner_pending_bytes'] == second['conservative_owner_pending_bytes']
        and address_delta == second['minimum_launch_headroom_bytes'] - first['minimum_launch_headroom_bytes']
        and address_delta == (after['width']-before['width'])*new['resources']['memory']['worker_address_space_limit_bytes'],
        'a fixed window must preserve the owner pending budget while accounting for extra workers')
    return dict(schema=SCHEMA, schema_version=1, diagonal_observation=old_ref, independent_observation=new_ref,
        diagonal_profile=before_ref, independent_profile=after_ref, recipe_id=after['recipe_id'],
        count=after['count'], matched_input=matched, exact_stdout_sha256=after['diagnostic']['stdout']['sha256'],
        measurements=dict(diagonal=first, independent=second), extra_worker_address_space_limits_bytes=address_delta,
        cumulative_limits_are_physical_reservation=False, guard_validated=True,
        outcome='INDEPENDENT_CLIENT_VISIBLE_CREDITS_VERIFIED', source_effects_equivalence_proven=False,
        clean_performance_comparison=False, default_selection_changed=False, deployment_authorized=False)


def check(path):
    stored = read_artifact(_ref(path, SCHEMA))
    replay = compare(stored['diagonal_observation']['uri'], stored['independent_observation']['uri'])
    _require(replay == stored, 'independent-window validation must exactly match cold replay')
    return stored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('diagonal', 'independent', 'check', 'out'):
        parser.add_argument('--'+flag, type=Path)
    args = parser.parse_args(argv)
    if args.check and not any((args.diagonal, args.independent, args.out)):
        value = check(args.check)
    elif not args.check and args.diagonal and args.independent and args.out:
        value = compare(args.diagonal, args.independent)
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        parser.error('use --check RESULT or --diagonal RUN --independent RUN --out RESULT')
    print(json.dumps(dict(outcome=value['outcome'], credits=value['measurements']['independent']['live_credits'])))


if __name__ == '__main__':
    main()
