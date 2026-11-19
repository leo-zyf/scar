"""Reconstruct credit chains and process-local reuse costs from checked native runs."""
from __future__ import annotations

import argparse
import json
from math import lcm
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import native_ordered_window_validation as windows


SCHEMA = 'scar.native-ordered-credit-cost'
_DISPATCH = 'native_ordered_dispatch_v1.h'
_PHASES = ('solve', 'numeric_provider_solve')
_CLOCKS = ('wall', 'cpu')


def _require(valid, reason):
    if not valid:
        raise ValueError(reason)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _local(package_ref, member):
    return {**member, 'uri': str(Path(package_ref['uri']).parent/member['uri'])}


def _source(observation):
    package_ref = observation['package']
    package = read_artifact(package_ref)
    header, = (member for member in package['headers'] if member['uri'] == _DISPATCH)
    dispatcher, main = _local(package_ref, header), _local(package_ref, package['stream']['source'])
    dispatch = read_artifact(dispatcher, json_document=False).decode()
    worker = read_artifact(main, json_document=False).decode()
    required = (
        'team.slots[sequence % team.slots.size()]',
        'while (next < std::min(count, static_cast<std::uint64_t>(config.window)))',
        'const auto index = sequence % team.slots.size();',
        'retire(result);\n        ++summary.retired;\n        if (next < count) {\n            issue(next++);',
        'auto worker = make_worker(cpu);')
    lifecycle = ('const auto factory = [&](int)', 'provider->create(', 'symbolic = std::move(symbolic)',
        'provider->solve(symbolic.get(),')
    _require(all(dispatch.count(marker) == 1 for marker in required) and
        all(worker.count(marker) == 1 for marker in lifecycle),
        'credit and persistent-worker projection requires the exact admitted dispatcher and solver route')
    positions = [worker.index(marker) for marker in lifecycle]
    _require(positions == sorted(positions), 'worker symbolic handle must be created and retained before use')
    return dict(dispatcher=dispatcher, solver=main, symbolic_handle_lifetime='ONE_PRIVATE_HANDLE_PER_WORKER_FACTORY',
        worker_solve_call='EXACT_REPEATED_NUMERIC_PROVIDER_CALL', pre_run_bootstrap_charged_to_loop=False)


def _incidence(count, width, window):
    """Count each lane's membership in each reusable credit chain."""
    period = lcm(width, window)
    repeats, tail = divmod(count, period)
    rows = [[0] * width for _ in range(window)]
    # The map is periodic even when the last issue fills only part of the cycle.
    for issue in range(min(period, count)):
        rows[issue % window][issue % width] += repeats + int(issue < tail)
    return rows, period


def _subset_clock(row, clock, selected):
    count = row['count']
    total, least, most = (row[f'{kind}_{clock}_ns'] for kind in ('total', 'min', 'max'))
    _require(type(selected) is int and 0 <= selected <= count, 'scope contains more observations than one lane')
    excluded = count-selected
    lower, upper = max(selected*least, total-excluded*most), min(selected*most, total-excluded*least)
    _require(0 <= lower <= upper, 'lane aggregate does not admit its declared subset')
    return lower, upper


def _mean_after_one(rows, clock):
    left = sum(row['count']-1 for row in rows)
    _require(all(row['count'] > 1 for row in rows), 'post-initial service bounds require a subsequent request per lane')
    ranges = [_subset_clock(row, clock, row['count']-1) for row in rows]
    lower, upper = (sum(endpoints) for endpoints in zip(*ranges, strict=True))
    return dict(remaining_requests=left, lower_ns=lower//left, upper_ns=(upper+left-1)//left)


def _chain(incidence, lanes, phase, clock):
    bounds = [_subset_clock(lane['phases'][phase], clock, used)
              for lane,used in zip(lanes, incidence, strict=True)]
    return dict(lower_ns=sum(lower for lower,_ in bounds), upper_ns=sum(upper for _,upper in bounds),
        exactly_reconstructed=all(lower == upper for lower,upper in bounds))


def _arm(observation, matched_arm):
    width, window = (matched_arm[key] for key in ('workers', 'window'))
    count = observation['count']
    lanes = observation['timing']['workers']
    incidence, period = _incidence(count, width, window)
    _require(len(lanes) == width and all(sum(col) == lane['phases']['solve']['count']
        for col,lane in zip(zip(*incidence, strict=True), lanes, strict=True)),
        'source modulo assignment must exactly match measured worker solve counts')
    chains = [dict(credit=credit, lane_requests={str(lane['cpu_start']): requests for lane,requests
        in zip(lanes, row, strict=True) if requests}, measurements={phase: {
            clock: _chain(row, lanes, phase, clock) for clock in _CLOCKS} for phase in _PHASES})
        for credit,row in enumerate(incidence)]
    for chain in chains:
        solve = chain['measurements']['solve']['wall']
        _require(solve['lower_ns'] <= matched_arm['diagnostic_observation']['owner_loop_wall_upper_envelope_ns'],
            'observed credit chain contradicts the owner loop envelope')
    heaviest = {}
    for phase in _PHASES:
        bounds = [row['measurements'][phase]['wall'] for row in chains]
        heaviest[phase] = dict(lower_ns=max(row['lower_ns'] for row in bounds),
            upper_ns=max(row['upper_ns'] for row in bounds))
    by_phase = {}
    for phase in (*_PHASES, 'receive_request'):
        cuts = [lane['phases'][phase] for lane in lanes]
        by_phase[phase] = {clock: dict(lowest_observed_ns=min(row[f'min_{clock}_ns'] for row in cuts),
            highest_observed_ns=max(row[f'max_{clock}_ns'] for row in cuts),
            total_ns=sum(row[f'total_{clock}_ns'] for row in cuts),
            after_one_request_per_worker_mean_bounds=_mean_after_one(cuts, clock)) for clock in _CLOCKS}
    owner_timing = observation['timing']['owner']['phases']
    obs = matched_arm['diagnostic_observation']
    _require(all(lane['phases']['bootstrap']['count'] == 1 for lane in lanes),
        'each measured worker must bootstrap its private solve state once')
    return dict(count=count, workers=width, window=window, assignment_cycle_issues=period,
        live_credit_counts=observation['timing'].get('credits'),
        outstanding_requests_at_most=min(window, count),
        worker_slots_without_outstanding_at_least=max(0,width-window),
        first_requests=width, requests_reusing_worker_process=count-width, same_worker_reissue_stride=width,
        additional_in_order_retirements_before_lane_reissue=width-window,
        source=_source(observation), chains=chains, heaviest_serial_solve_service_ns=heaviest,
        measured_phase_evidence=by_phase,
        worker_bootstrap_cpu_ns_sum=sum(lane['phases']['bootstrap']['total_cpu_ns'] for lane in lanes),
        owner_dispatch_bootstrap_wall_ns=owner_timing['dispatch_bootstrap']['total_wall_ns'],
        owner_loop_envelope_ns=[obs['observed_owner_loop_coverage_ns'],obs['owner_loop_wall_upper_envelope_ns']])


def _delta(before, after):
    return [after[0]-before[1], after[1]-before[0]]


def _qualified_pair(diagonal, independent):
    first = diagonal['heaviest_serial_solve_service_ns']['solve']
    second = independent['heaviest_serial_solve_service_ns']['solve']
    return dict(observed_loop_duration_delta_ns=_delta(diagonal['owner_loop_envelope_ns'],
        independent['owner_loop_envelope_ns']),
        heaviest_serial_solve_chain_delta_ns=_delta([first['lower_ns'], first['upper_ns']],
            [second['lower_ns'], second['upper_ns']]))


def project(validation_path):
    """Explain which changes in the recorded ordered schedule are identifiable."""
    checked_ref = _ref(validation_path, windows.SCHEMA)
    checked = windows.check(checked_ref['uri'])
    _require(checked == read_artifact(checked_ref), 'independent-window comparison changed under replay')
    observations = (read_artifact(checked['diagonal_observation']), read_artifact(checked['independent_observation']))
    before, after = (_arm(raw, checked['measurements'][role])
        for raw,role in zip(observations, ('diagonal', 'independent'), strict=True))
    _require(before['source']['dispatcher']['sha256'] == after['source']['dispatcher']['sha256'],
        'both runs need the same exact dispatcher semantics')
    service = _qualified_pair(before, after)
    service['unassigned_delta_ns'] = _delta(service['heaviest_serial_solve_chain_delta_ns'],
        service['observed_loop_duration_delta_ns'])
    after_one = 'after_one_request_per_worker_mean_bounds'
    b, a = (row['measured_phase_evidence']['solve']['cpu'] for row in (before, after))
    first_only = a[after_one]['lower_ns'] > b[after_one]['upper_ns']
    all_samples = a['lowest_observed_ns'] > b['highest_observed_ns']
    return dict(schema=SCHEMA, schema_version=1, validation=checked_ref, matched_input=checked['matched_input'],
        exact_stdout_sha256=checked['exact_stdout_sha256'], recipe_id=checked['recipe_id'],
        issue_graph=dict(worker_assignment='I_MOD_WORKER_POPULATION', credit='I_MOD_ACTIVE_WINDOW',
            refill=dict(retirement_ordinal='I', successor_issue_ordinal='I_PLUS_WINDOW'), ordered_retirement=True,
            process_reuse_requires_same_worker_modulo_lane=True),
        arms=dict(diagonal=before, independent=after), measured_service_delta=service,
        conclusions=dict(evidence='INSTRUMENTED_EXACT_INPUT_OBSERVATIONS',
            one_first_request_per_worker_shift_excluded=first_only,
            independent_sample_min_cpu_exceeds_diagonal_sample_max=all_samples,
            post_initial_cpu_mean_shift_required=first_only,
            process_local_reuse_scope_visible=True, per_request_clock_or_cache_trace_available=False,
            worker_receive_wall_includes_idle_wait_wake_and_transfer=True,
            measured_cause_of_solve_service_shift=None, pure_ipc_cost_ns=None,
            solver_cross_width_steady_service_transfer_supported=False,
            unassigned_delta_is_causal_attribution=False, clean_performance_comparison=False,
            deployment_authorized=False))


def check(path):
    stored = read_artifact(_ref(path, SCHEMA))
    _require(stored == project(stored['validation']['uri']), 'credit cost result must exactly match cold replay')
    return stored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--validation', type=Path)
    mode.add_argument('--check', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.check and args.out is None:
        value = check(args.check)
    elif args.validation and args.out:
        value = project(args.validation)
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        parser.error('use --check RESULT or --validation VERIFIED_PAIR --out RESULT')
    print(json.dumps(value['measured_service_delta'], sort_keys=True))


if __name__ == '__main__':
    main()
