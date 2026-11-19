"""Fit native solve service at two widths and check a separate larger width."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path

from scar.codegen import native_ordered_timing as diagnostic
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import native_ordered_profile as profiles
from scar.optimization import native_ordered_transport_cost as transport
from scar.optimization import native_ordered_width_cost as width_cost


SCHEMA = 'scar.native-ordered-width-solve-service-fit'
HOLDOUT = SCHEMA + '-holdout'
_PHASES = ('solve', 'numeric_provider_solve')
_AXES = ('wall', 'cpu')
_SIDES = ('lower_ns', 'estimate_ns', 'upper_ns')
_HYPOTHESES = ('flat_pilot', 'affine_two_width', 'plateau_after_intermediate')


def _require(valid, message):
    if not valid:
        raise ValueError(message)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _diagnostic(reference):
    checked = diagnostic.check(reference['uri'])
    _require(checked == read_artifact(reference), 'verified native observation changed during replay')
    return checked


def _live_resources(profile, measured, hardware_ref):
    before = read_artifact(profile['anchor']['plan'])
    archive, live = (read_artifact(ref) for ref in (profile['hardware'], hardware_ref))
    for field in ('cgroup_mode', 'cpu_quota_equivalents', 'allowed_memory_nodes', 'memory_placement_enforced'):
        _require(archive[field] == live[field], 'live resource constraints differ from the calibration context: ' + field)
    count = measured['count']
    resources, _ = profiles._resources(before, hardware_ref, live, count, measured['width'], measured['recipe_id'])
    _require(all(resources['placement'][key] == expected for key,expected in measured['placement'].items()),
        'intermediate observation differs from the compatible live CPU and NUMA placement')
    return dict(cpu_ids=resources['placement']['launch_cpu_ids'], numa_node=resources['placement']['cpu_numa_node'],
        cpu_quota_equivalents=live['cpu_quota_equivalents'],
        memory_headroom_required_bytes=resources['memory']['minimum_launch_headroom_bytes'],
        live_mask=live['launching_affinity'], archived_placement_matches=True)


def _services(observation):
    count = observation['count']
    return {phase: {axis: width_cost._bounds([observation], axis, count, name=phase, workers=True)
                    for axis in _AXES} for phase in _PHASES}


def _per_lane(observation):
    return {str(worker['cpu_start']): {phase: {axis: round(Fraction(
        worker['phases'][phase][f'total_{axis}_ns'], worker['phases'][phase]['count']))
            for axis in _AXES} for phase in _PHASES} for worker in observation['timing']['workers']}


def _fit(origin, anchor, weight):
    """Apply a two-point curve to the point estimate and sample extrema separately."""
    a = 1 - weight
    bounds = (('lower_ns', 'upper_ns') if a >= 0 else ('upper_ns', 'lower_ns'))
    lower = a * origin[bounds[0]] + weight * anchor['lower_ns']
    point = a * origin['estimate_ns'] + weight * anchor['estimate_ns']
    upper = a * origin[bounds[1]] + weight * anchor['upper_ns']
    return dict(lower_ns=max(0, lower.numerator // lower.denominator),
        estimate_ns=max(0, round(point)), upper_ns=max(0, width_cost._ceil(upper)))


def _curve(start, end, width, first, second, hypothesis):
    position = Fraction(width - first, second - first)
    if hypothesis == 'flat_pilot':
        position = Fraction(0)
    elif hypothesis == 'plateau_after_intermediate':
        position = min(Fraction(1), position)
    return {phase: {axis: _fit(start[phase][axis], end[phase][axis], position)
        for axis in _AXES} for phase in _PHASES}


def _project(base, fitted, count, width):
    times = {}
    for side in _SIDES:
        terms = base[side]
        solve = fitted['solve']
        scenario = dict(terms, solve=solve['wall'][side], worker_cpu=solve['cpu'][side]+terms['send_cpu'])
        times[side] = transport._schedule(count, width, scenario, 'late_header_cpu_service')
    return dict(phase_service_ns=fitted,
        loop_scenario_ns={side: row['loop_ns'] for side,row in times.items()},
        point_breakdown=times['estimate_ns'])


def propose(pilot_path, intermediate_path, live_hardware_path):
    """Freeze solve-service hypotheses using only the pilot and intermediate width."""
    pilot_ref = _ref(pilot_path, width_cost.SCHEMA)
    pilot = width_cost.check(pilot_ref['uri'])
    _require(read_artifact(pilot_ref) == pilot and len(pilot['native_observations']) == 1
        and pilot['phase_intervals'] is not None, 'expected one exact checked native pilot recording')
    intermediate_ref = _ref(intermediate_path, diagnostic.SCHEMA)
    middle = _diagnostic(intermediate_ref)
    profile = read_artifact(pilot['profile_plan'])
    start = read_artifact(pilot['native_observations'][0])
    width_cost._bind_observed_profiles([start, middle], pilot['profile_plan'], profile)
    first, second = start['width'], middle['width']
    widths = pilot['admissible_widths']
    _require(widths[0] == first < second < widths[-1] and second in widths
        and middle['placement']['worker_cpu_ids'][:first] == start['placement']['worker_cpu_ids'],
        'expected the minimum pilot width and an intermediate width on its exact CPU prefix')
    live_ref = _ref(live_hardware_path, profiles.HARDWARE)
    resources = _live_resources(profile, middle, live_ref)
    measured = {str(first): _services(start), str(second): _services(middle)}
    per_lane = {str(first): _per_lane(start), str(second): _per_lane(middle)}
    start_cost, end_cost = measured[str(first)], measured[str(second)]
    base = {side: transport._terms(pilot['phase_intervals'], side) for side in _SIDES}
    projections = {str(width): {hypothesis: _project(base,
        _curve(start_cost, end_cost, width, first, second, hypothesis), start['count'], width)
        for hypothesis in _HYPOTHESES} for width in widths}
    choices = {hypothesis: min(widths,
        key=lambda width: (projections[str(width)][hypothesis]['loop_scenario_ns']['estimate_ns'], width))
        for hypothesis in _HYPOTHESES}
    return dict(schema=SCHEMA, schema_version=1, pilot=pilot_ref, intermediate=intermediate_ref,
        live_hardware=live_ref, measured_resources=resources, widths=widths,
        calibrated_widths=[first, second], solve_measurements=measured, per_cpu_measured_occurrence_means_ns=per_lane,
        projections=projections, fastest_width_by_hypothesis=choices, retained_pilot_width=pilot['recommended_width'],
        assumptions=dict(scope='INSTRUMENTED_IDENTICAL_CSR_AND_NUMERICAL_PROVIDER',
            fitted_component='WORKER_SOLVE_ONLY_INCLUDES_NUMERIC_PROVIDER_CHILD',
            nested_numeric_added=False, transport_and_owner_terms='FROZEN_PILOT_ENDPOINT_CPU_PROXY',
            transport_split_or_contention_proven=False, request_ordinals_paired_per_cpu=False,
            extrapolation_scope='TWO_POINTS_CONDITIONAL_SENSITIVITY',
            environment_contention_and_frequency_ns=None, pure_ipc_ns=None, clean_full_entry_ns=None),
        holdout_used_for_fit=False, global_optimum_proven=False, deployment_authorized=False)


def _residual(estimate, lower, upper):
    return dict(prediction_ns=estimate, observed_minus_prediction_ns=[lower-estimate, upper-estimate],
        distance_to_observed_envelope_ns=max(0, lower-estimate, estimate-upper),
        within_observed_envelope=lower <= estimate <= upper)


def validate_holdout(proposal_path, holdout_path):
    """Compare a larger width against the frozen service hypotheses without refitting."""
    proposal_ref = _ref(proposal_path, SCHEMA)
    proposal = check(proposal_ref['uri'])
    pilot = read_artifact(proposal['pilot'])
    profile = read_artifact(pilot['profile_plan'])
    start = read_artifact(pilot['native_observations'][0])
    middle = read_artifact(proposal['intermediate'])
    held_ref = _ref(holdout_path, diagnostic.SCHEMA)
    held = _diagnostic(held_ref)
    width = held['width']
    _require(width in proposal['widths'] and width > proposal['calibrated_widths'][-1],
        'holdout must use an independent uncalibrated larger width')
    width_cost._bind_observed_profiles([start, middle, held], pilot['profile_plan'], profile)
    observed = transport._observed_holdout(held)
    lower, upper = (observed[key] for key in ('observed_owner_loop_coverage_ns', 'owner_loop_wall_upper_envelope_ns'))
    modeled = proposal['projections'][str(width)]
    residuals = {name: _residual(row['loop_scenario_ns']['estimate_ns'], lower, upper) for name,row in modeled.items()}
    error = 'distance_to_observed_envelope_ns'
    better = any(residuals[name][error] < residuals['flat_pilot'][error] for name in _HYPOTHESES[1:])
    return dict(schema=HOLDOUT, schema_version=1, proposal=proposal_ref, holdout=held_ref, holdout_width=width,
        observation=observed, holdout_service=_services(held), residuals=residuals,
        decision='SENSITIVITY_IMPROVED_ONE_HOLDOUT' if better else 'TWO_WIDTH_FIT_DID_NOT_IMPROVE_HOLDOUT',
        retained_pilot_width=proposal['retained_pilot_width'], fitted_model_replaces_pilot=False,
        holdout_used_for_fit=False, deployment_authorized=False)


def check(path):
    schema = json.loads(Path(path).read_bytes())['schema']
    _require(schema in (SCHEMA, HOLDOUT), 'unsupported ordered solve-service result')
    stored = read_artifact(_ref(path, schema))
    if schema == SCHEMA:
        replay = propose(stored['pilot']['uri'], stored['intermediate']['uri'], stored['live_hardware']['uri'])
    else:
        replay = validate_holdout(stored['proposal']['uri'], stored['holdout']['uri'])
    _require(replay == stored, 'ordered solve-service result must exactly match cold replay')
    return stored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for flag in ('pilot', 'proposal', 'check'):
        modes.add_argument('--'+flag, type=Path)
    for flag in ('intermediate', 'live-hardware', 'holdout', 'out'):
        parser.add_argument('--'+flag, type=Path)
    args = parser.parse_args(argv)
    if args.check and not any((args.intermediate, args.live_hardware, args.holdout, args.out)):
        result = check(args.check)
    elif args.pilot and args.intermediate and args.live_hardware and args.out and not args.holdout:
        result = propose(args.pilot, args.intermediate, args.live_hardware)
    elif args.proposal and args.holdout and args.out and not any((args.intermediate, args.live_hardware)):
        result = validate_holdout(args.proposal, args.holdout)
    else:
        parser.error('use --check RECEIPT, --pilot RESULT --intermediate DIAG --live-hardware HARDWARE --out PROPOSAL, or --proposal PROPOSAL --holdout DIAG --out VALIDATION')
    if args.out:
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    print(json.dumps(dict(schema=result['schema'], decision=result.get('decision'), retained_pilot_width=result['retained_pilot_width'])))


if __name__ == '__main__':
    main()
