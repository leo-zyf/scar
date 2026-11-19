"""Nominate an ordered native worker width from a checked diagnostic pilot."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
from statistics import median

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import native_ordered_profile as profiles
from scar.optimization.sparse_parallel_cost import project_serial_order


SCHEMA = 'scar.native-ordered-width-cost'
_THREE = ('lower_ns', 'estimate_ns', 'upper_ns')
_OWNER = ('issue_prepare', 'issue_send', 'retire')
_DIAGNOSTIC_OWNER = (*_OWNER, 'ordered_wait_receive_header', 'ordered_receive_payload', 'digest_update')
_DIAGNOSTIC_WORKER = ('solve', 'numeric_provider_solve', 'receive_request', 'send_response')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _interval(groups, axis, count):
    """Pool occurrence extrema and give each complete diagnostic run one mean."""
    values, means = [], []
    for group in groups:
        rows = [row for row in group if row['count']]
        _require(rows and sum(row['count'] for row in rows) == count, 'diagnostic phase has an incomplete request count')
        for row in rows:
            n, lo, total, hi = (row[key] for key in ('count', f'min_{axis}_ns', f'total_{axis}_ns', f'max_{axis}_ns'))
            _require(all(type(value) is int for value in (n, lo, total, hi)) and n > 0 and 0 <= lo <= hi
                and n * lo <= total <= n * hi, 'diagnostic phase has inconsistent occurrence bounds')
        values.extend(rows)
        means.append(Fraction(sum(row[f'total_{axis}_ns'] for row in rows), count))
    return dict(lower_ns=min(row[f'min_{axis}_ns'] for row in values), estimate_ns=round(median(means)),
        upper_ns=max(row[f'max_{axis}_ns'] for row in values), observations=len(groups),
        occurrences_per_observation=count, estimate_statistic='MEDIAN_OF_COMPLETE_RUN_MEANS')


def _bounds(profiles_rows, axis, count, *, name, workers):
    if workers:
        rows = [[worker['phases'][name] for worker in row['timing']['workers']] for row in profiles_rows]
    else:
        rows = [[row['timing']['owner']['phases'][name]] for row in profiles_rows]
    return _interval(rows, axis, count)


def _phase_intervals(rows, count):
    def metrics(name, workers):
        return {axis: _bounds(rows, axis, count, name=name, workers=workers) for axis in ('wall', 'cpu')}

    return dict(owner={name: metrics(name, False) for name in _DIAGNOSTIC_OWNER},
                workers={name: metrics(name, True) for name in _DIAGNOSTIC_WORKER})


def _conditional_floor(terms, width, quota):
    count = terms['count']
    owner_cpu = count * terms['owner_cpu']
    worker_cpu = count * terms['worker_cpu']
    capacity = Fraction(width+1, 1)
    if quota is not None:
        capacity = min(capacity, Fraction(*quota))
    floors = dict(owner_lane_cpu_ns=owner_cpu,
        worker_lane_service_ns=-(-count*terms['worker_elapsed']//width),
        first_issue_to_first_retirement_chain_ns=
            terms['prepare_elapsed'] + terms['worker_elapsed'] + terms['retire_elapsed'],
        average_cpu_capacity_work_floor_ns=_ceil(Fraction(owner_cpu+worker_cpu, 1)/capacity))
    return dict(terms=floors, max_ns=max(floors.values()),
        basis='CONDITIONAL_OBSERVED_MINIMUM_PHASE_SERVICE_AND_AVERAGE_CPU_CAPACITY',
        physical_or_clean_runtime_lower_bound_proven=False)


def _ceil(value):
    return -(-value.numerator // value.denominator)


def _project(resources, intervals, exact_diagnostics, quota):
    placement, case, memory = (resources[key] for key in ('placement', 'case', 'memory'))
    width = resources['dispatch']['window']
    count = case['call_count']
    result = dict(width=width, worker_cpu_ids=placement['worker_cpu_ids'],
        owner_cpu_id=placement['owner_cpu_id'], cpu_numa_node=placement['cpu_numa_node'],
        required_memory_headroom_bytes=memory['minimum_launch_headroom_bytes'],
        owner_and_workers_on_distinct_physical_cores=True, cpu_quota_equivalents=quota,
        exact_width_diagnostic_lifecycle=exact_diagnostics, full_entry_estimate_ns=None)
    if intervals is None:
        return dict(result, model_service_interval_ns=None, estimated_ordered_wait_ns=None,
            conditional_lower_bound=None)
    owner, workers = (intervals[key] for key in ('owner', 'workers'))
    projections = {}
    for side in _THREE:
        # The owner receives the oldest result and publishes it before refill.
        # Observed receive wall includes wait and is not charged as another service.
        # Model worker readiness at solve return; result transport/lifetime remain unknown.
        costs = dict(bootstrap_ns=0, prepare_ns=owner['issue_prepare']['wall'][side],
            submit_ns=owner['issue_send']['wall'][side], worker_ns=workers['solve']['wall'][side],
            replay_before_wait_ns=0, publish_ns=owner['retire']['wall'][side], close_ns=0,
            owner_work_ns=sum(owner[name]['cpu'][side] for name in _OWNER),
            worker_work_ns=workers['solve']['cpu'][side])
        projections[side] = project_serial_order(count, width, width, costs, cpu_quota=quota)
        if side == 'lower_ns':
            floor = _conditional_floor(dict(count=count, owner_cpu=costs['owner_work_ns'],
                worker_cpu=costs['worker_work_ns'], worker_elapsed=costs['worker_ns'],
                prepare_elapsed=costs['prepare_ns'], retire_elapsed=costs['publish_ns']), width, quota)
    return dict(result, model_service_interval_ns={side: projection['elapsed_projection_ns']
            for side, projection in projections.items()},
        estimated_ordered_wait_ns=projections['estimate_ns']['ordered_wait_ns'],
        conditional_lower_bound=floor,
        service_envelope_scope='CONDITIONAL_ORDERED_LOOP_SAMPLE_SCENARIOS', is_empirical_runtime_prediction=False)


def _lifecycle(rows, width):
    return [{name: row['timing']['owner']['phases'][name] for name in
        ('entry', 'ingress', 'dispatch_bootstrap', 'publication_finalize', 'drain_reap')}
        for row in rows if row['width'] == width]


def _projections(profile, rows):
    before = read_artifact(profile['anchor']['plan'])
    hardware_ref = profile['hardware']
    hardware = read_artifact(hardware_ref)
    case = profile['resources']['case']
    count = case['call_count']
    recipe = profile['resources']['recipe_id']
    quota = hardware['cpu_quota_equivalents']
    widths = profiles.admissible_profile_widths(before, hardware, count)
    _require(profile['width'] in widths, 'original pilot width must be within the exact admissible hardware domain')
    intervals = _phase_intervals(rows, count) if rows else None
    result = {}
    for width in widths:
        resources, _ = profiles._resources(before, hardware_ref, hardware, count, width, recipe)
        result[str(width)] = _project(resources, intervals, _lifecycle(rows, width), quota)
    return widths, intervals, result


def _nominate(profile, projections):
    measured = [row for row in projections.values() if row['model_service_interval_ns'] is not None]
    if not measured:
        return profile['width'], [], 'KEEP_PILOT_MODEL_UNRESOLVED'
    best = min(measured, key=lambda row: (row['model_service_interval_ns']['estimate_ns'], row['width']))
    best_interval = best['model_service_interval_ns']
    overlap = sorted(row['width'] for row in measured if row is not best
        and row['model_service_interval_ns']['lower_ns'] <= best_interval['upper_ns']
        and best_interval['lower_ns'] <= row['model_service_interval_ns']['upper_ns'])
    return best['width'], overlap, 'LOWEST_CONDITIONAL_LOOP_ESTIMATE'


def _same_bytes(left, right):
    return all(left[key] == right[key] for key in ('sha256', 'size_bytes'))


def _configuration(profile):
    resources = profile['resources']
    placement = resources['placement']
    return dict(workers=profile['width'], window=resources['dispatch']['window'], **{
        key: placement[key] for key in ('owner_cpu_id', 'worker_cpu_ids', 'launch_cpu_ids', 'cpu_numa_node')})


def _credit_service_quote(credit_ref, observation_ref, profile_ref, arm, stdout_sha256):
    """Attach an audited measured credit chain to one concrete placement."""
    profile = read_artifact(profile_ref)
    observation = read_artifact(observation_ref)
    configuration = _configuration(profile)
    _require(configuration['workers'] == observation['width'] == arm['workers']
        and configuration['window'] == observation.get('window', observation['width']) == arm['window']
        and all(configuration[key] == observation['placement'][key] for key in observation['placement'])
        and observation['diagnostic']['stdout']['sha256'] == stdout_sha256,
        'credit service quote must bind its exact observed configuration and output')
    first, last = arm['owner_loop_envelope_ns']
    solves = arm['heaviest_serial_solve_service_ns']['solve']
    _require(0 <= solves['lower_ns'] <= solves['upper_ns'] <= last and 0 < first <= last,
        'solve-chain work must fit within the measured native loop observation')
    memory = profile['resources']['memory']
    return dict(basis='EXACT_CONFIGURATION_DIAGNOSTIC_OBSERVATION', configuration=configuration,
        credit_cost=credit_ref, native_observation=observation_ref, profile_plan=profile_ref,
        observed_loop_wall_ns=dict(lower_ns=first, upper_ns=last),
        heaviest_serial_solve_ns=solves,
        address_space_limits_bytes=memory['worker_address_space_sum_bytes'],
        minimum_launch_headroom_bytes=memory['minimum_launch_headroom_bytes'],
        exact_stdout_sha256=stdout_sha256, exact_input=observation['input'],
        transferable_to_other_configuration=False, clean_profitability_proven=False)


def _bind_observed_profiles(rows, profile_ref, profile, *, allow_independent_windows=False):
    from scar.codegen.native_ordered_profile import SCHEMA as SOLVER

    matched_input = rows[0]['input']
    case = profile['resources']['case']
    recipe = profile['resources']['recipe_id']
    oracle = profile['oracle']
    scope_keys = ('anchor', 'source_seed', 'argv', 'original_observation', 'hardware',
                  'wire_identity', 'source_stream_input', 'source_stream_header', 'oracle')
    saw_pilot = False
    for row in rows:
        _require(row['package']['schema'] == SOLVER and row['count'] == case['call_count']
            and row['recipe_id'] == recipe and _same_bytes(row['input'], matched_input)
            and row['expected_stdout'] is not None and _same_bytes(row['expected_stdout'], oracle)
            and all(_same_bytes(row[arm]['stdout'], oracle) for arm in ('production', 'diagnostic'))
            and row['timing']['cost_scope'] == 'INSTRUMENTED_NATIVE_DIAGNOSTIC_NOT_PROFITABILITY_EVIDENCE',
            'diagnostic input, count, oracle or clock scope differs from the requested native pilot')
        # The diagnostic verifier has already replayed its exact solver and scoped plan.
        solver = read_artifact(row['package'])
        bound_ref = solver['profile_plan']
        bound = read_artifact(bound_ref)
        saw_pilot |= bound_ref == profile_ref
        configuration = _configuration(bound)
        window = row.get('window', row['width'])
        _require(all(bound[key] == profile[key] for key in scope_keys)
            and bound['resources']['case'] == case and configuration['workers'] == solver['width'] == row['width']
            and window == configuration['window'] == solver.get('window', solver['width'])
            and (allow_independent_windows or window == row['width'])
            and row['placement'] == {key: configuration[key] for key in
                ('owner_cpu_id', 'worker_cpu_ids', 'launch_cpu_ids', 'cpu_numa_node')},
            'diagnostic service is bound to a different registered scope or physical placement')
    _require(saw_pilot, 'at least one diagnostic must measure the exact requested pilot plan')
    return matched_input


def _observations(paths, profile_ref, profile):
    paths = tuple(paths)
    if not paths:
        return [], [], None
    from scar.codegen import native_ordered_timing

    references = [_ref(path, native_ordered_timing.SCHEMA) for path in paths]
    _require(len({reference['sha256'] for reference in references}) == len(references),
        'each native diagnostic must be a distinct recorded observation')
    rows = [native_ordered_timing.check(reference['uri']) for reference in references]
    _require(all(read_artifact(reference) == row for reference,row in zip(references, rows, strict=True)),
        'native observation changed during cold verification')
    matched_input = _bind_observed_profiles(rows, profile_ref, profile)
    return references, rows, matched_input


def propose(profile_plan_path, native_observation_paths):
    """Replay a diagnostic pilot and nominate one admissible width for clean A/B."""
    profile_ref = _ref(profile_plan_path, profiles.SCHEMA)
    profile = profiles.verify_profile_plan(profile_ref['uri'])
    references, rows, matched_input = _observations(native_observation_paths, profile_ref, profile)
    widths, intervals, projections = _projections(profile, rows)
    choice, overlap, reason = _nominate(profile, projections)
    return dict(schema=SCHEMA, schema_version=1, profile_plan=profile_ref, native_observations=references,
        hardware=profile['hardware'], recipe_id=profile['resources']['recipe_id'], matched_input=matched_input,
        case=profile['resources']['case'], admissible_widths=list(widths), phase_intervals=intervals,
        projections=projections, recommended_width=choice, model_envelopes_overlap_with=overlap, decision=reason,
        evidence_scope='SAME_FIXED_INPUT_INSTRUMENTED_NATIVE_ORDERED_LOOP',
        model=dict(owner_lane='ISSUE_PREPARE_THEN_SEND_AND_OLDEST_RETIRE',
            worker_lane='DISJOINT_PINNED_PRIVATE_SOLVE_SPAN',
            cpu_clock='PER_PROCESS_CPU_TIME_ON_CHECKED_PINNED_LANES',
            wait='PROJECT_FROM_WORKER_SOLVE_READINESS_AND_ORDERED_SLOT_CREDITS',
            owner_send='OBSERVED_OWNER_SEND_WALL_MAY_INCLUDE_BLOCKING',
            observed_receive_in_loop_cost=False, nested_digest_added_again=False,
            derived_terms_apply_across_widths_conditionally=True,
            owner_receive_wall_is_pure_ipc=False, worker_receive_and_send_wall_is_pure_ipc=False,
            unknown=dict(pure_request_ipc_service_ns=None, pure_result_ipc_service_ns=None,
                result_ready_after_transfer_ns=None, backpressure_change_by_width_ns=None,
                startup_close_by_unmeasured_width_ns=None, cache_numa_or_memory_contention_ns=None,
                clean_full_entry_ns=None),
            tie_break='NARROWER_WIDTH_FOR_EQUAL_POINT_ESTIMATE'),
        native_full_entry_estimate_ns=None, clean_validation_required=True, deployment_authorized=False,
        global_optimum_proven=False)


def check(path):
    stored = read_artifact(_ref(path, SCHEMA))
    replay = propose(stored['profile_plan']['uri'], [reference['uri'] for reference in stored['native_observations']])
    _require(replay == stored, 'ordered width recommendation differs from dry replay')
    return stored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-plan', type=Path)
    parser.add_argument('--native-observation', action='append', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args(argv)
    if args.check and not any((args.profile_plan, args.native_observation, args.out)):
        result = check(args.check)
    elif not args.check and args.profile_plan and args.out:
        result = propose(args.profile_plan, args.native_observation or ())
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        parser.error('use --check RESULT or --profile-plan PLAN --native-observation DIAGNOSTIC --out RESULT')
    print(json.dumps(dict(recommended_width=result['recommended_width'], decision=result['decision'])))


if __name__ == '__main__':
    main()
