"""Separate ordered worker placement from credit and payload capacity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import native_ordered_transport_cost as transport
from scar.optimization.sparse_parallel_recipe import _storage_requirements


SCHEMA = 'scar.native-ordered-resource-schedule'
VALIDATION = SCHEMA + '-validation'
_DISPATCH = 'native_ordered_dispatch_v1.h'


def _require(valid, reason):
    if not valid:
        raise ValueError(reason)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _dispatcher(transport_record):
    ranges = set()
    members = []
    for observation_ref in transport_record['native_observations']:
        observation = read_artifact(observation_ref)
        package_ref = observation['package']
        package = read_artifact(package_ref)
        header, = (ref for ref in package['headers'] if ref['uri'] == _DISPATCH)
        absolute = {**header, 'uri': str(Path(package_ref['uri']).parent/_DISPATCH)}
        read_artifact(absolute, json_document=False)
        members.append(absolute)
        ranges.add((header['sha256'], header['size_bytes']))
    _require(len(ranges) == 1, 'all diagnostics must use the same exact ordered dispatcher source')
    return members


def _memory(case, window, workers, memory):
    cap = memory['worker_address_space_limit_bytes']
    guard = _storage_requirements(case['size'], case['nnz'], window, cap, memory['per_worker_symbolic_budget_bytes'])
    fixed = guard['owner_structure_bytes'] + guard['owner_detached_snapshot_bound_bytes']
    pending = guard['owner_pending_input_bytes'] + guard['owner_pending_result_bound_bytes']
    worker_limits = cap * workers
    return dict(owner_fixed_bytes=fixed, owner_conservative_pending_bytes=pending,
        pending_slot_budget_bytes=pending // window, summed_worker_address_space_limits_bytes=worker_limits,
        minimum_launch_headroom_bytes=fixed+pending+worker_limits, physical_reservation=False)


def _credit_chain(count, window, terms):
    generations = (count + window - 1)//window
    per_generation = sum(terms[name] for name in (
        'prepare', 'request', 'solve', 'send_cpu', 'header_cpu', 'payload_cpu', 'retire'))
    return dict(generations=generations, chain_first_issue=0, chain_step=window,
        chain_last_issue=(generations-1)*window, conditional_service_per_generation_ns=per_generation,
        lower_ns=generations*per_generation, physical_or_clean_runtime_bound_proven=False)


def _candidate(n, workers, window, terms, scenario, memory, limit, diagonal):
    _require(1 <= window <= workers <= n, 'ordered dispatch requires a free modulo lane for the next issue')
    point_model = transport._schedule(n, window, terms['estimate_ns'], scenario)
    intervals = {side: transport._schedule(n, window, terms[side], scenario)['loop_ns']
        for side in ('lower_ns', 'upper_ns')}
    intervals['estimate_ns'] = point_model['loop_ns']
    lower = _credit_chain(n, window, terms['lower_ns'])
    _require(intervals['lower_ns'] >= lower['lower_ns'], 'owner schedule violated the modeled slot-credit chain')
    return dict(workers=workers, window=window, memory=memory,
        fits_recorded_headroom=memory['minimum_launch_headroom_bytes'] <= limit,
        current_lowering_admitted=diagonal,
        lowering_status='CHECKED_NATIVE_PROFILE' if diagonal else 'DISPATCH_MODEL_ONLY',
        conditional_cost_interval_ns=intervals, point_model=point_model, conditional_credit_chain=lower)


def _dominance(candidate, competitor):
    """Use disjoint observed service envelopes and the same result contract."""
    return (candidate['configuration']['window'] == competitor['configuration']['window']
        and candidate['exact_stdout_sha256'] == competitor['exact_stdout_sha256']
        and candidate['observed_loop_wall_ns']['upper_ns'] < competitor['observed_loop_wall_ns']['lower_ns']
        and candidate['heaviest_serial_solve_ns']['upper_ns'] < competitor['heaviest_serial_solve_ns']['lower_ns']
        and candidate['address_space_limits_bytes'] <= competitor['address_space_limits_bytes']
        and candidate['minimum_launch_headroom_bytes'] <= competitor['minimum_launch_headroom_bytes'])


def _observed_schedule(source_ref, model, plan, limit, memory_cap, credit_cost_path):
    from scar.optimization import native_ordered_credit_cost as credit
    from scar.optimization import native_ordered_width_cost as width_cost

    credit_ref = _ref(credit_cost_path, credit.SCHEMA)
    measured = credit.check(credit_ref['uri'])
    pair = read_artifact(measured['validation'])
    _require(read_artifact(credit_ref) == measured
        and model['recipe_id'] == measured['recipe_id']
        and model['matched_input'] == measured['matched_input'] == pair['matched_input']
        and pair['diagonal_profile'] == model['profile_plan']
        and pair['diagonal_observation'] in model['native_observations']
        and measured['exact_stdout_sha256'] == pair['exact_stdout_sha256'],
        'credit observations require the same exact pilot, ordered input, recipe and output')

    quotes = {}
    for role in ('diagonal', 'independent'):
        row = width_cost._credit_service_quote(credit_ref, pair[f'{role}_observation'],
            pair[f'{role}_profile'], measured['arms'][role], measured['exact_stdout_sha256'])
        config = row['configuration']
        key = f'{config["workers"]}/{config["window"]}'
        _require(key not in quotes and config['workers'] in model['admissible_widths'],
            'exact observation lies outside the checked resource domain')
        quotes[key] = row

    options = {}
    for workers in model['admissible_widths']:
        cohort = model['projections'][str(workers)]
        for window in range(1, workers+1):
            key = f'{workers}/{window}'
            memory = _memory(model['case'], window, workers, plan['memory'])
            fits = memory['minimum_launch_headroom_bytes'] <= limit
            quote = quotes.get(key)
            worker_cpus, owner = cohort['worker_cpu_ids'], cohort['owner_cpu_id']
            configuration = dict(workers=workers, window=window, owner_cpu_id=owner, worker_cpu_ids=worker_cpus,
                launch_cpu_ids=sorted([owner, *worker_cpus]), cpu_numa_node=cohort['cpu_numa_node'])
            if quote is not None:
                _require(configuration == quote['configuration']
                    and memory['minimum_launch_headroom_bytes'] == quote['minimum_launch_headroom_bytes']
                    and memory['summed_worker_address_space_limits_bytes'] == quote['address_space_limits_bytes'],
                    'cost quote differs from its scheduled lanes or memory contract')
            state = 'MEASURED_ADMISSIBLE' if quote is not None else 'UNKNOWN'
            if not fits:
                state = 'RESOURCE_REJECTED'
            if window == 1:
                state = 'WINDOW_UNAVAILABLE'
            options[key] = dict(configuration=configuration, memory=memory, fits_recorded_headroom=fits,
                cost_quote=quote, cost_status='OBSERVED' if quote is not None else 'UNKNOWN',
                state=state, observed_package_verified=quote is not None)

    candidates = {key: row for key,row in options.items() if row['state'] == 'MEASURED_ADMISSIBLE'}
    cost_order = sorted(candidates, key=lambda key: (
        candidates[key]['cost_quote']['observed_loop_wall_ns']['upper_ns'],
        candidates[key]['configuration']['workers'], candidates[key]['configuration']['window']))
    for loser in cost_order:
        right = candidates[loser]['cost_quote']
        winner = next((key for key in cost_order if key != loser and _dominance(candidates[key]['cost_quote'], right)), None)
        if winner is not None:
            left = candidates[winner]['cost_quote']
            options[loser].update(state='DOMINATED', dominance=dict(by=winner,
                scope='MATCHED_NATIVE_DIAGNOSTIC_OUTPUT_AND_EXACT_PLACEMENT_ONLY',
                loop_wall_separation_ns=right['observed_loop_wall_ns']['lower_ns']-left['observed_loop_wall_ns']['upper_ns'],
                solve_chain_separation_ns=right['heaviest_serial_solve_ns']['lower_ns']-left['heaviest_serial_solve_ns']['upper_ns'],
                address_space_limits_bytes_saved=right['address_space_limits_bytes']-left['address_space_limits_bytes']))
    chosen = next((key for key in cost_order if options[key]['state'] == 'MEASURED_ADMISSIBLE'), None)
    compiler_job = None
    if chosen is not None:
        row = options[chosen]
        config = row['configuration']
        compiler_job = dict(width=config['workers'], window=config['window'], owner_cpu_id=config['owner_cpu_id'],
            worker_cpu_ids=config['worker_cpu_ids'], minimum_launch_headroom_bytes=row['memory']['minimum_launch_headroom_bytes'],
            live_resource_guard_required=True)
    return dict(schema=SCHEMA, schema_version=2, transport=source_ref, credit_cost=credit_ref,
        profile_plan=model['profile_plan'], matched_input=model['matched_input'],
        recorded_headroom_bytes=limit, requested_memory_cap_bytes=memory_cap, options=options,
        cost_policy='EXACT_OBSERVATION_PER_WORKER_POPULATION_WINDOW_AND_PLACEMENT',
        unmatched_configuration_fallback='UNKNOWN', selected_configuration=chosen, compile_candidate=compiler_job,
        decision='NOMINATE_UNDOMINATED_MEASURED_CONFIGURATION' if compiler_job else 'NO_MEASURED_CONFIGURATION_FITS',
        transport_model_costs_used=False, cross_configuration_timing_transfer=False, default_selection_changed=False,
        global_optimum_proven=False, clean_validation_required=True, deployment_authorized=False)


def propose(transport_proposal_path, *, memory_limit_bytes=None, credit_cost_path=None):
    """Nominate one currently lowerable candidate and retain unsupported window projections."""
    source_ref = _ref(transport_proposal_path, transport.SCHEMA)
    model = transport.check(source_ref['uri'])
    _require(model == read_artifact(source_ref)
        and model['model']['primary_scenario'] == 'late_header_cpu_service',
        'expected the checked two-endpoint transport service model')
    profile = read_artifact(model['profile_plan'])
    hardware = read_artifact(profile['hardware'])
    plan = profile['resources']
    node = plan['placement']['cpu_numa_node']
    primary = model['model']['primary_scenario']
    minimum = hardware['node_free_snapshot'][str(node)]['bytes']
    finite = hardware['finite_memory_headroom_snapshot_bytes']
    if finite is not None:
        minimum = min(minimum, finite)
    if memory_limit_bytes is not None:
        _require(type(memory_limit_bytes) is int and 0 < memory_limit_bytes <= minimum,
            'explicit cap must fit the archived physical and cgroup snapshot')
        minimum = memory_limit_bytes
    if credit_cost_path is not None:
        return _observed_schedule(source_ref, model, plan, minimum, memory_limit_bytes, credit_cost_path)

    count = model['case']['call_count']
    terms = model['fitted_terms_ns']
    options = {}
    for workers in model['admissible_widths']:
        original = model['projections'][str(workers)]
        for window in range(1, workers + 1):
            same = window == workers
            memory = _memory(model['case'], window, workers, plan['memory'])
            row = _candidate(count, workers, window, terms, primary, memory, minimum, same)
            if same:
                original_cost = original['model'][primary]
                _require(memory['minimum_launch_headroom_bytes'] == original['required_memory_headroom_bytes']
                    and all(row['conditional_cost_interval_ns'][key] == original_cost[key]
                        for key in row['conditional_cost_interval_ns']),
                    'diagonal projection must match the recorded lowering and pilot model')
            else:
                lower_workers = options.get(f'{window}/{window}')
                if lower_workers is not None:
                    saved = (memory['summed_worker_address_space_limits_bytes']
                             - lower_workers['memory']['summed_worker_address_space_limits_bytes'])
                    _require(saved > 0
                        and row['conditional_cost_interval_ns'] == lower_workers['conditional_cost_interval_ns'],
                        'the homogeneous service model must preserve cost for the same number of credits')
                    row['same_credit_dominance'] = dict(by=f'{window}/{window}',
                        worker_address_space_limit_bytes_saved=saved,
                        scope='HOMOGENEOUS_SERVICE_MODEL_ONLY')
            options[f'{workers}/{window}'] = dict(row, owner_cpu_id=original['owner_cpu_id'],
                worker_cpu_ids=original['worker_cpu_ids'])
    ready = [row for row in options.values() if row['current_lowering_admitted'] and row['fits_recorded_headroom']]
    compiler_job = None
    overlap = []
    if ready:
        choice = min(ready, key=lambda row: (row['conditional_cost_interval_ns']['estimate_ns'], row['workers']))
        expected = choice['conditional_cost_interval_ns']
        overlap = sorted(row['workers'] for row in ready if row is not choice
            and row['conditional_cost_interval_ns']['lower_ns'] <= expected['upper_ns']
            and expected['lower_ns'] <= row['conditional_cost_interval_ns']['upper_ns'])
        compiler_job = dict(width=choice['workers'], window=choice['window'], owner_cpu_id=choice['owner_cpu_id'],
            worker_cpu_ids=choice['worker_cpu_ids'],
            minimum_launch_headroom_bytes=choice['memory']['minimum_launch_headroom_bytes'],
            live_resource_guard_required=True)
    return dict(schema=SCHEMA, schema_version=1, transport=source_ref, profile_plan=model['profile_plan'],
        pilot_observations=model['native_observations'], matched_input=model['matched_input'],
        dispatcher=_dispatcher(model), recorded_headroom_bytes=minimum, requested_memory_cap_bytes=memory_limit_bytes,
        graph=dict(issue_domain={'axis': 'i', 'count': count}, lane='i mod worker_count', constraints=[
                '1 <= window <= worker_count <= call_count',
                'owner_fixed + window * pending_slot_budget + worker_count * worker_address_space_limit <= headroom'],
            dependencies=[dict(issue='i', prerequisite='retire(i-window)', when='i >= window'),
                dict(retire='i', prerequisite='retire(i-1)', when='i > 0'),
                dict(retire='i', prerequisite='response_visible(i)', when='all'),
                dict(retire='0', prerequisite='initial_fill(window)', when='all')],
            slot_reuse_proof='PREVIOUS_SAME_MODULO_LANE_ISSUE_I_MINUS_WORKERS_ALREADY_RETIRED_BY_I_MINUS_WINDOW',
            window_limit_bounds_inflight=True, worker_placement_controls_reserved_lanes=True,
            off_diagonal_dispatch_model_only=True),
        options=options, compile_candidate=compiler_job, overlapping_candidate_widths=overlap,
        decision='NOMINATE_ONE_CHECKED_CONFIGURATION' if compiler_job else 'NO_CHECKED_CONFIGURATION_FITS',
        same_input_and_placement_required_for_pilot_reuse=True,
        heterogeneous_per_lane_service_proven=False, global_optimum_proven=False,
        clean_validation_required=True, deployment_authorized=False)


def validate_holdout(proposal_path, transport_holdout_path):
    """Check the selected diagonal against an already archived independent run."""
    proposal_ref = _ref(proposal_path, SCHEMA)
    proposal = check(proposal_ref['uri'])
    _require(proposal['schema_version'] == 1, 'transport holdout replays the conditional resource model only')
    heldout_ref = _ref(transport_holdout_path, transport.HOLDOUT)
    holdout = transport.check(heldout_ref['uri'])
    width = holdout['holdout_width']
    _require(holdout == read_artifact(heldout_ref) and holdout['proposal'] == proposal['transport'],
        'validation requires the same frozen pilot and independent width observation')
    row = proposal['options'][f'{width}/{width}']
    obs = holdout['observation']
    envelope_top = obs['owner_loop_wall_upper_envelope_ns']
    observed_lower = obs['observed_owner_loop_coverage_ns']
    credits = row['conditional_credit_chain']['lower_ns']
    _require(row['current_lowering_admitted'] and credits <= envelope_top,
        'observed independent loop contradicts the conditional credit lower bound')
    choice = proposal['compile_candidate']
    return dict(schema=VALIDATION, schema_version=1, proposal=proposal_ref, transport_holdout=heldout_ref,
        validated_width=width, validated_window=width, selected_configuration_matches=choice is not None
            and choice['width'] == choice['window'] == width,
        conditional_credit_lower_ns=credits, observed_diagnostic_loop_envelope_ns=[observed_lower,envelope_top],
        diagonal_matches_existing_native_and_model=True, off_diagonal_runtime_checked=False,
        holdout_used_for_nomination=False, global_optimum_proven=False, deployment_authorized=False)


def check(path):
    schema = json.loads(Path(path).read_bytes())['schema']
    _require(schema in (SCHEMA, VALIDATION), 'unsupported ordered resource-schedule result')
    frozen = read_artifact(_ref(path, schema))
    if schema == SCHEMA:
        cost = frozen['credit_cost']['uri'] if frozen['schema_version'] == 2 else None
        replay = propose(frozen['transport']['uri'], memory_limit_bytes=frozen['requested_memory_cap_bytes'],
            credit_cost_path=cost)
    else:
        replay = validate_holdout(frozen['proposal']['uri'], frozen['transport_holdout']['uri'])
    _require(replay == frozen, 'ordered resource schedule must exactly match cold replay')
    return frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for flag in ('transport', 'proposal', 'check'):
        mode.add_argument('--'+flag, type=Path)
    parser.add_argument('--holdout', type=Path)
    parser.add_argument('--credit-cost', type=Path, help='quote only configurations with admitted credit observations')
    parser.add_argument('--memory-limit-bytes', type=int)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.check and not any((args.holdout, args.out, args.credit_cost, args.memory_limit_bytes is not None)):
        value = check(args.check)
    elif args.transport and args.out and not args.holdout:
        value = propose(args.transport, memory_limit_bytes=args.memory_limit_bytes, credit_cost_path=args.credit_cost)
    elif args.proposal and args.holdout and args.out and args.memory_limit_bytes is None and args.credit_cost is None:
        value = validate_holdout(args.proposal, args.holdout)
    else:
        parser.error('use --check RESULT, --transport MODEL --out RESULT, or --proposal RESULT --holdout VALIDATION --out RESULT')
    if args.out:
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(dict(schema=value['schema'], compile_candidate=value.get('compile_candidate'),
        validated_width=value.get('validated_width'))))


if __name__ == '__main__':
    main()
