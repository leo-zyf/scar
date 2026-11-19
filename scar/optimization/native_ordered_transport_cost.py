"""Project socket endpoint service and ordered waits from a frozen native pilot."""
from __future__ import annotations

import argparse
from collections import deque
from fractions import Fraction
import json
from pathlib import Path

from scar.codegen import native_ordered_timing as diagnostic
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import native_ordered_width_cost as width_cost


SCHEMA = 'scar.native-ordered-result-transport-cost'
HOLDOUT = SCHEMA + '-holdout'
_SIDES = ('lower_ns', 'estimate_ns', 'upper_ns')
_SCENARIOS = ('early_header_cpu_service', 'late_header_cpu_service', 'late_header_mixed_send_wall')
_READ_PHASES = ('ordered_wait_receive_header', 'ordered_receive_payload')
_ISSUE_PHASES = ('issue_prepare', 'issue_send')
_NONLOOP = ('ingress', 'dispatch_bootstrap', 'drain_reap', 'publication_finalize')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


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
    cpus = Fraction(width+1, 1)
    if quota is not None:
        cpus = min(cpus, Fraction(*quota))
    owner, worker = count*terms['owner_cpu'], count*terms['worker_cpu']
    floors = dict(serial_owner_process_cpu_ns=owner,
        worker_solve_plus_response_service_ns=width_cost._ceil(
            Fraction(count*(terms['solve'] + terms['send_cpu']), width)),
        average_cpu_work_rate_ns=width_cost._ceil(Fraction(owner+worker, 1)/cpus),
        first_result_in_early_header_scenario_ns=terms['prepare'] + terms['solve']
            + max(terms['send_cpu'], terms['header_cpu']) + terms['payload_cpu'] + terms['retire'])
    return dict(max_ns=max(floors.values()), terms=floors,
        basis='CONDITIONAL_PILOT_MINIMA_AND_ASSUMED_ENDPOINT_VISIBILITY',
        physical_or_clean_runtime_lower_bound_proven=False)


def propose(pilot_proposal_path):
    """Freeze an independent transport model from the archived pilot alone."""
    pilot_ref = _ref(pilot_proposal_path, width_cost.SCHEMA)
    pilot = width_cost.check(pilot_ref['uri'])
    _require(read_artifact(pilot_ref) == pilot and pilot['schema_version'] == 1 and pilot['native_observations']
        and pilot['phase_intervals'] is not None and pilot['matched_input'] is not None,
        'transport modeling needs a checked native diagnostic pilot')
    profile = read_artifact(pilot['profile_plan'])
    count = pilot['case']['call_count']
    phases = pilot['phase_intervals']
    fitted = {side: _terms(phases, side) for side in _SIDES}
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
    return dict(schema=SCHEMA, schema_version=1, pilot_proposal=pilot_ref, profile_plan=pilot['profile_plan'],
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
    proposal_ref = _ref(proposal_path, SCHEMA)
    proposal = check(proposal_ref['uri'])
    reference = _ref(holdout_path, diagnostic.SCHEMA)
    held = diagnostic.check(reference['uri'])
    _require(held == read_artifact(reference), 'held-out observation changed during replay')
    pilot_ref = proposal['profile_plan']
    profile = read_artifact(pilot_ref)
    pilot = read_artifact(proposal['pilot_proposal'])
    calibration = [read_artifact(row) for row in proposal['native_observations']]
    width = held['width']
    _require(reference not in proposal['native_observations'] and width in proposal['admissible_widths']
        and all(row['width'] != width for row in calibration), 'holdout width already appears in the pilot')
    width_cost._bind_observed_profiles([*calibration, held], pilot_ref, profile)
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


def check(path):
    schema = json.loads(Path(path).read_bytes())['schema']
    _require(schema in (SCHEMA, HOLDOUT), 'unsupported ordered transport result')
    stored = read_artifact(_ref(path, schema))
    if schema == SCHEMA:
        replay = propose(stored['pilot_proposal']['uri'])
    else:
        replay = validate_holdout(stored['proposal']['uri'], stored['holdout']['uri'])
    _require(replay == stored, 'ordered transport cost does not match cold replay')
    return stored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for flag in ('pilot-proposal', 'proposal', 'check'):
        modes.add_argument('--'+flag, type=Path)
    parser.add_argument('--holdout', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.check and not args.out and not args.holdout:
        result = check(args.check)
    elif args.pilot_proposal and args.out and not args.holdout:
        result = propose(args.pilot_proposal)
    elif args.proposal and args.holdout and args.out:
        result = validate_holdout(args.proposal, args.holdout)
    else:
        parser.error('use --check RESULT, --pilot-proposal PILOT --out RESULT, or --proposal RESULT --holdout RUN --out VALIDATION')
    if args.out:
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    print(json.dumps(dict(schema=result['schema'], recommended_width=
        result.get('recommended_width', result.get('proposed_width')))))


if __name__ == '__main__':
    main()
