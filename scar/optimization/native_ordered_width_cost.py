"""Plan ordered native worker widths and transport from a checked CPU pilot."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
from statistics import median
import sys

from scar.ir.execution_recipe import artifact_reference, open_artifact, read_artifact
from scar.optimization import native_ordered_profile as profiles
from scar.optimization.sparse_parallel_cost import project_serial_order


SCHEMA = 'scar.native-ordered-width-cost'
TRANSPORT = 'scar.native-ordered-result-transport-cost'
HOLDOUT = TRANSPORT + '-holdout'
_THREE = ('lower_ns', 'estimate_ns', 'upper_ns')
_OWNER = ('issue_prepare', 'issue_send', 'retire')
_DIAGNOSTIC_OWNER = (*_OWNER, 'ordered_wait_receive_header', 'ordered_receive_payload', 'digest_update')
_DIAGNOSTIC_WORKER = ('solve', 'numeric_provider_solve', 'receive_request', 'send_response')
_SCENARIOS = ('early_header_cpu_service', 'late_header_cpu_service', 'late_header_mixed_send_wall')
_READ_PHASES = ('ordered_wait_receive_header', 'ordered_receive_payload')
_ISSUE_PHASES = ('issue_prepare', 'issue_send')
_NONLOOP = ('ingress', 'dispatch_bootstrap', 'drain_reap', 'publication_finalize')


@dataclass(frozen=True, slots=True)
class _WidthReplay:
    proposal: dict
    profile: dict
    observations: list[dict]


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
    floors = dict(owner_lane_cpu_ns=owner_cpu,
        worker_lane_service_ns=-(-count*terms['worker_elapsed']//width),
        first_issue_to_first_retirement_chain_ns=
            terms['prepare_elapsed'] + terms['worker_elapsed'] + terms['retire_elapsed'],
        average_cpu_capacity_work_floor_ns=_cpu_capacity_floor(owner_cpu+worker_cpu, width, quota))
    return dict(terms=floors, max_ns=max(floors.values()),
        basis='CONDITIONAL_OBSERVED_MINIMUM_PHASE_SERVICE_AND_AVERAGE_CPU_CAPACITY',
        physical_or_clean_runtime_lower_bound_proven=False)


def _ceil(value):
    return -(-value.numerator // value.denominator)


def _cpu_capacity_floor(work, width, quota):
    capacity = Fraction(width + 1, 1)
    if quota is not None:
        capacity = min(capacity, Fraction(*quota))
    return _ceil(Fraction(work, 1) / capacity)


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
    admitted = profiles._admissible_profile_resources(before, hardware_ref, hardware, count, recipe)
    widths = tuple(admitted)
    _require(profile['width'] in widths, 'original pilot width must be within the exact admissible hardware domain')
    intervals = _phase_intervals(rows, count) if rows else None
    result = {}
    for width, resources in admitted.items():
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


def _bind_observed_profile(row, profile, matched_input, *, allow_independent_windows):
    from scar.codegen.native_ordered_profile import SCHEMA as SOLVER

    case = profile['resources']['case']
    oracle = profile['oracle']
    scope_keys = ('anchor', 'source_seed', 'argv', 'original_observation', 'hardware',
                  'wire_identity', 'source_stream_input', 'source_stream_header', 'oracle')
    _require(row['package']['schema'] == SOLVER and row['count'] == case['call_count']
        and row['recipe_id'] == profile['resources']['recipe_id'] and _same_bytes(row['input'], matched_input)
        and row['expected_stdout'] is not None and _same_bytes(row['expected_stdout'], oracle)
        and all(_same_bytes(row[arm]['stdout'], oracle) for arm in ('production', 'diagnostic'))
        and row['timing']['cost_scope'] == 'INSTRUMENTED_NATIVE_DIAGNOSTIC_NOT_PROFITABILITY_EVIDENCE',
        'diagnostic input, count, oracle or clock scope differs from the requested native pilot')
    # The diagnostic verifier has already replayed its exact solver and scoped plan.
    solver = read_artifact(row['package'])
    bound_ref = solver['profile_plan']
    bound = read_artifact(bound_ref)
    configuration = _configuration(bound)
    window = row.get('window', row['width'])
    _require(all(bound[key] == profile[key] for key in scope_keys)
        and bound['resources']['case'] == case and configuration['workers'] == solver['width'] == row['width']
        and window == configuration['window'] == solver.get('window', solver['width'])
        and (allow_independent_windows or window == row['width'])
        and row['placement'] == {key: configuration[key] for key in
            ('owner_cpu_id', 'worker_cpu_ids', 'launch_cpu_ids', 'cpu_numa_node')},
        'diagnostic service is bound to a different registered scope or physical placement')
    return bound_ref


def _bind_observed_profiles(rows, profile_ref, profile, *, allow_independent_windows=False):
    matched_input = rows[0]['input']
    bound = [_bind_observed_profile(row, profile, matched_input,
        allow_independent_windows=allow_independent_windows) for row in rows]
    _require(profile_ref in bound, 'at least one diagnostic must measure the exact requested pilot plan')
    return matched_input


def _observations(paths, profile_ref, profile):
    paths = tuple(paths)
    if not paths:
        return [], [], None
    from scar.codegen import native_ordered_timing

    references = [path if type(path) is dict else _ref(path, native_ordered_timing.SCHEMA) for path in paths]
    _require(all(reference.get('schema') == native_ordered_timing.SCHEMA
        and reference['uri'] == str(Path(reference['uri']).resolve(strict=True)) for reference in references),
        'native timing observation references required')
    _require(len({reference['sha256'] for reference in references}) == len(references),
        'each native diagnostic must be a distinct recorded observation')
    rows = [native_ordered_timing.check(reference['uri']) for reference in references]
    _require(all(read_artifact(reference) == row for reference,row in zip(references, rows, strict=True)),
        'native observation changed during cold verification')
    matched_input = _bind_observed_profiles(rows, profile_ref, profile)
    return references, rows, matched_input


def prepare_width_proposal(profile_plan_path, native_observation_paths):
    """Admit one diagnostic pilot for width and transport planning."""
    profile_ref = profile_plan_path if type(profile_plan_path) is dict else _ref(profile_plan_path, profiles.SCHEMA)
    profile = profiles.verify_profile_plan(profile_ref)
    references, rows, matched_input = _observations(native_observation_paths, profile_ref, profile)
    widths, intervals, projections = _projections(profile, rows)
    choice, overlap, reason = _nominate(profile, projections)
    proposal = dict(schema=SCHEMA, schema_version=1, profile_plan=profile_ref, native_observations=references,
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
    return _WidthReplay(proposal, profile, rows)


def propose(profile_plan_path, native_observation_paths):
    """Replay a pilot and nominate one admissible width for clean A/B."""
    return prepare_width_proposal(profile_plan_path, native_observation_paths).proposal


def _read_width_proposal(source):
    if type(source) is dict:
        reference, stored = source, read_artifact(source)
        _require(reference['schema'] == SCHEMA
            and reference['uri'] == str(Path(reference['uri']).resolve(strict=True)),
            'ordered width proposal reference required')
    else:
        reference, stored = open_artifact(Path(source).resolve(strict=True), SCHEMA)
    replay = prepare_width_proposal(stored['profile_plan'], stored['native_observations'])
    _require(replay.proposal == stored, 'ordered width recommendation differs from dry replay')
    read_artifact(reference, json_document=False)
    return replay


def check(path):
    return _read_width_proposal(path).proposal


def _terms(phases, side):
    owner, worker = phases['owner'], phases['workers']
    return dict(prepare=owner['issue_prepare']['wall'][side], request=owner['issue_send']['wall'][side],
        solve=worker['solve']['wall'][side], send_cpu=worker['send_response']['cpu'][side],
        send_wall=worker['send_response']['wall'][side],
        header_cpu=owner[_READ_PHASES[0]]['cpu'][side], payload_cpu=owner[_READ_PHASES[1]]['cpu'][side],
        retire=owner['retire']['wall'][side],
        owner_cpu=sum(owner[name]['cpu'][side] for name in (*_ISSUE_PHASES, *_READ_PHASES, 'retire')),
        worker_cpu=worker['solve']['cpu'][side] + worker['send_response']['cpu'][side])


def _schedule(count, width, terms, scenario):
    """Simulate one owner, a bounded FIFO, and one outstanding request per worker."""
    job_send = terms['send_wall'] if scenario == 'late_header_mixed_send_wall' else terms['send_cpu']
    header_send = 0 if scenario == 'early_header_cpu_service' else job_send
    pending = deque()
    owner, wait_header, wait_payload = 0, 0, 0

    def submit():
        nonlocal owner
        owner += terms['prepare'] + terms['request']
        computed = owner + terms['solve']
        pending.append((computed + header_send, computed + job_send))

    for _ in range(width):
        submit()
    for issue in range(count):
        header, payload = pending.popleft()
        blocked = max(0, header - owner)
        wait_header += blocked
        owner += blocked + terms['header_cpu']
        blocked = max(0, payload - owner)
        wait_payload += blocked
        owner += blocked + terms['payload_cpu'] + terms['retire']
        if issue + width < count:
            submit()
    return dict(loop_ns=owner, header_wait_ns=wait_header, payload_wait_ns=wait_payload,
        owner_receive_cpu_ns=count*(terms['header_cpu'] + terms['payload_cpu']),
        owner_loop_service_proxy_ns=count*sum(
            terms[name] for name in ('prepare', 'request', 'header_cpu', 'payload_cpu', 'retire')),
        owner_receive_elapsed_proxy_ns=wait_header + wait_payload + count*(terms['header_cpu']+terms['payload_cpu']))


def _floor(terms, count, width, quota):
    owner, worker = count*terms['owner_cpu'], count*terms['worker_cpu']
    floors = dict(serial_owner_process_cpu_ns=owner,
        worker_solve_plus_response_service_ns=_ceil(
            Fraction(count*(terms['solve'] + terms['send_cpu']), width)),
        average_cpu_work_rate_ns=_cpu_capacity_floor(owner+worker, width, quota),
        first_result_in_early_header_scenario_ns=terms['prepare'] + terms['solve']
            + max(terms['send_cpu'], terms['header_cpu']) + terms['payload_cpu'] + terms['retire'])
    return dict(max_ns=max(floors.values()), terms=floors,
        basis='CONDITIONAL_PILOT_MINIMA_AND_ASSUMED_ENDPOINT_VISIBILITY',
        physical_or_clean_runtime_lower_bound_proven=False)


def project_transport(pilot_ref, replay):
    """Price transfers using an admitted pilot and its published width receipt."""
    pilot = replay.proposal
    _require(read_artifact(pilot_ref) == pilot and pilot['schema_version'] == 1 and pilot['native_observations']
        and pilot['phase_intervals'] is not None and pilot['matched_input'] is not None,
        'transport modeling needs a checked native diagnostic pilot')
    read_artifact(pilot['profile_plan'], json_document=False)
    for reference in pilot['native_observations']:
        read_artifact(reference, json_document=False)
    profile = replay.profile
    count = pilot['case']['call_count']
    phases = pilot['phase_intervals']
    fitted = {side: _terms(phases, side) for side in _THREE}
    projections = {}
    for name, row in pilot['projections'].items():
        width = row['width']
        simulations = {scenario: {
            side: _schedule(count, width, terms, scenario) for side,terms in fitted.items()}
            for scenario in _SCENARIOS}
        projections[name] = dict(width=width, owner_cpu_id=row['owner_cpu_id'],
            worker_cpu_ids=row['worker_cpu_ids'], cpu_numa_node=row['cpu_numa_node'],
            required_memory_headroom_bytes=row['required_memory_headroom_bytes'],
            cpu_quota_equivalents=row['cpu_quota_equivalents'],
            model=scenarios_summary(simulations),
            conditional_lower_bound=_floor(fitted['lower_ns'], count, width, row['cpu_quota_equivalents']),
            previous_solve_ready_projection_ns=row['model_service_interval_ns']['estimate_ns'])
    primary = 'late_header_cpu_service'
    candidates = sorted(projections.values(), key=lambda row: (row['model'][primary]['estimate_ns'], row['width']))
    best = candidates[0]
    return dict(schema=TRANSPORT, schema_version=1, pilot_proposal=pilot_ref, profile_plan=pilot['profile_plan'],
        native_observations=pilot['native_observations'], matched_input=pilot['matched_input'],
        recipe_id=pilot['recipe_id'], case=pilot['case'], pilot_width=profile['width'], fitted_terms_ns=fitted,
        admissible_widths=pilot['admissible_widths'], projections=projections, recommended_width=best['width'],
        model=dict(primary_scenario=primary, ranking='MINIMUM_SCENARIO_POINT_ESTIMATE_THEN_SMALLER_WIDTH',
            raw_clock_subset='INSTRUMENTED_FIXED_COUNT_NATIVE',
            owner_read_charge='PROCESS_CPU_ONLY_FOR_HEADER_AND_PAYLOAD',
            wait='MAX_OF_MODEL_READY_CLOCK_MINUS_ORDERED_OWNER_CLOCK_ZERO',
            worker_send_cpu='PROXY_FOR_SERIAL_SOCKET_ENDPOINT_WORK_AFTER_SOLVE',
            early_header='HEADER_AT_SOLVE_END_PAYLOAD_AFTER_SENDER_CPU',
            late_header='HEADER_AND_PAYLOAD_AFTER_SENDER_CPU',
            wall_stress='MIXED_SEND_WALL_AS_SENSITIVITY_ONLY_INCLUDES_POSSIBLE_BLOCKING',
            unmeasured=dict(pure_ipc_ns=None, sender_header_payload_split_ns=None, kernel_socket_queues_ns=None,
                partial_payload_overlap_ns=None, request_path_split_ns=None, width_specific_contention_ns=None,
                production_startup_close_ns=None, full_entry_ns=None),
            observed_read_wall_added=False, digest_child_double_counted=False,
            cross_width_projections_conditional=True, clean_benchmark_required=True),
        holdout_used_for_nomination=False, global_optimum_proven=False, deployment_authorized=False)


def propose_transport(pilot_proposal_path):
    pilot_ref = _ref(pilot_proposal_path, SCHEMA)
    return project_transport(pilot_ref, _read_width_proposal(pilot_ref))


def _replay_transport_proposal(reference, stored):
    replay = _read_width_proposal(stored['pilot_proposal'])
    _require(project_transport(stored['pilot_proposal'], replay) == stored,
        'ordered transport cost does not match cold replay')
    read_artifact(reference, json_document=False)
    return replay


def scenarios_summary(simulations):
    return {name: {**{side: sample['loop_ns'] for side,sample in samples.items()},
        'point_breakdown': samples['estimate_ns']} for name,samples in simulations.items()}


def _observed_holdout(observed):
    phases = observed['timing']['owner']['phases']
    loop = [*_ISSUE_PHASES, *_READ_PHASES, 'retire']
    covered = sum(phases[name]['total_wall_ns'] for name in loop)
    envelope = phases['entry']['total_wall_ns'] - sum(phases[name]['total_wall_ns'] for name in _NONLOOP)
    _require(0 < covered <= envelope, 'diagnostic disjoint owner loop cuts exceed the entry envelope')
    workers = observed['timing']['workers']
    solve_total = sum(worker['phases']['solve']['total_wall_ns'] for worker in workers)
    return dict(observed_owner_loop_coverage_ns=covered, owner_loop_wall_upper_envelope_ns=envelope,
        envelope_unassigned_ns=envelope-covered, scope='INSTRUMENTED_SEPARATE_OWNER_CUTS_AND_OUTER_BOUND',
        measured_owner_receive_wall_ns=sum(phases[name]['total_wall_ns'] for name in _READ_PHASES),
        measured_owner_receive_cpu_ns=sum(phases[name]['total_cpu_ns'] for name in _READ_PHASES),
        measured_worker_send_cpu_ns=sum(worker['phases']['send_response']['total_cpu_ns'] for worker in workers),
        measured_worker_send_wall_ns=sum(worker['phases']['send_response']['total_wall_ns'] for worker in workers),
        worker_solve_wall_mean_ns=round(Fraction(solve_total, observed['count'])),
        pure_ipc_ns=None, whole_entry_clean_timing_comparison=False)


def validate_holdout(proposal_path, holdout_path):
    """Check a distinct width after nomination and report residuals without refitting."""
    from scar.codegen import native_ordered_timing as diagnostic

    proposal_ref, proposal = open_artifact(Path(proposal_path).resolve(strict=True), TRANSPORT)
    replay = _replay_transport_proposal(proposal_ref, proposal)
    reference = _ref(holdout_path, diagnostic.SCHEMA)
    held = diagnostic.check(reference['uri'])
    _require(held == read_artifact(reference), 'held-out observation changed during replay')
    pilot = replay.proposal
    width = held['width']
    _require(reference not in proposal['native_observations'] and width in proposal['admissible_widths']
        and all(row['width'] != width for row in replay.observations), 'holdout width already appears in the pilot')
    for calibrated in proposal['native_observations']:
        read_artifact(calibrated, json_document=False)
    _bind_observed_profile(held, replay.profile, proposal['matched_input'], allow_independent_windows=False)
    observed = _observed_holdout(held)
    lower, upper = (observed[key] for key in ('observed_owner_loop_coverage_ns', 'owner_loop_wall_upper_envelope_ns'))
    model = proposal['projections'][str(width)]['model']
    predictions = {name: sample['estimate_ns'] for name,sample in model.items()}
    predictions['previous_solve_ready'] = pilot['projections'][str(width)]['model_service_interval_ns']['estimate_ns']
    residuals = {name: dict(predicted_ns=estimate, observed_minus_prediction_ns=[lower-estimate, upper-estimate],
        prediction_within_instrumented_loop_envelope=lower <= estimate <= upper)
        for name,estimate in predictions.items()}
    receiver = {name: observed['measured_owner_receive_wall_ns']-sample['point_breakdown']['owner_receive_elapsed_proxy_ns']
        for name,sample in model.items()}
    terms = proposal['fitted_terms_ns']['estimate_ns']
    scenario = proposal['model']['primary_scenario']
    counterfactual = _schedule(held['count'], width, dict(terms, solve=observed['worker_solve_wall_mean_ns']), scenario)
    return dict(schema=HOLDOUT, schema_version=1, proposal=proposal_ref, holdout=reference,
        holdout_width=width, proposed_width=proposal['recommended_width'], observation=observed, residuals=residuals,
        observed_receive_wall_minus_model_receive_proxy_ns=receiver,
        after_holdout_solve_only_sensitivity=dict(pilot_solve_mean_ns=terms['solve'],
            holdout_solve_mean_ns=observed['worker_solve_wall_mean_ns'],
            conditional_loop_ns=counterfactual['loop_ns'],
            observed_minus_conditional_loop_ns=[lower-counterfactual['loop_ns'], upper-counterfactual['loop_ns']],
            used_for_nomination=False, change_cause_proven=False),
        holdout_used_for_nomination=False, clean_benchmark_required=True, deployment_authorized=False)


def check_transport(path):
    reference, stored = open_artifact(Path(path).resolve(strict=True))
    schema = reference['schema']
    _require(schema in (TRANSPORT, HOLDOUT), 'unsupported ordered transport result')
    if schema == TRANSPORT:
        _replay_transport_proposal(reference, stored)
    else:
        replay = validate_holdout(stored['proposal']['uri'], stored['holdout']['uri'])
        _require(replay == stored, 'ordered transport cost does not match cold replay')
        read_artifact(reference, json_document=False)
    return stored


def _write_report(path, report):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def _width_cli(argv):
    parser = argparse.ArgumentParser(description=__doc__, epilog='Use the transport subcommand to price result delivery.')
    parser.add_argument('--profile-plan', type=Path)
    parser.add_argument('--native-observation', action='append', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args(argv)
    if args.check and not any((args.profile_plan, args.native_observation, args.out)):
        result = check(args.check)
    elif not args.check and args.profile_plan and args.out:
        result = propose(args.profile_plan, args.native_observation or ())
        _write_report(args.out, result)
    else:
        parser.error('use --check RESULT or --profile-plan PLAN --native-observation DIAGNOSTIC --out RESULT')
    print(json.dumps(dict(recommended_width=result['recommended_width'], decision=result['decision'])))


def _transport_cli(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for flag in ('pilot-proposal', 'proposal', 'check'):
        modes.add_argument('--'+flag, type=Path)
    parser.add_argument('--holdout', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.check and not args.out and not args.holdout:
        result = check_transport(args.check)
    elif args.pilot_proposal and args.out and not args.holdout:
        result = propose_transport(args.pilot_proposal)
    elif args.proposal and args.holdout and args.out:
        result = validate_holdout(args.proposal, args.holdout)
    else:
        parser.error('use --check RESULT, --pilot-proposal PILOT --out RESULT, or --proposal RESULT --holdout RUN --out VALIDATION')
    if args.out:
        _write_report(args.out, result)
    print(json.dumps(dict(schema=result['schema'], recommended_width=
        result.get('recommended_width', result.get('proposed_width')))))


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == 'transport':
        return _transport_cli(args[1:])
    return _width_cli(args)


if __name__ == '__main__':
    main()
