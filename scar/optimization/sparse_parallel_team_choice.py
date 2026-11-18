"""Freeze a forward team-width model and invite one ordered sparse challenger."""

import argparse
from fractions import Fraction
import json
from pathlib import Path

from scar.analysis.sparse_parallel_scaling_scope import _lanes, _release_lineage
from scar.analysis.target_inventory import _cpu_list
from scar.ir.execution_recipe import artifact_reference, read_artifact, validate_recipe
from . import sparse_parallel_cost as calibration
from .sparse_parallel_recipe import CHOICE, PROFILE, SCOPE
from .sparse_parallel_recipe_cost import project_recipe_window


SCHEMA = 'scar.sparse-parallel-team-frontier'
RELEASE = 'scar.sparse-parallel-entry-selection'
HARDWARE = calibration.HARDWARE
_ESTIMATES = ('estimate_ns', 'lower_ns', 'upper_ns')


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _anchor(release_ref):
    release = read_artifact(release_ref)
    source_ref = release['source']
    _require(source_ref['schema'] == SCOPE, 'published choice requires an exact original sparse scope')
    source = read_artifact(source_ref)
    _release_lineage(source_ref, source, release)
    cost_ref, recipe_ref = release['cost'], release['execution_recipe']
    cost = calibration.check(cost_ref['uri'])
    recipe = validate_recipe(read_artifact(recipe_ref))
    lowering = read_artifact(release['lowering'])
    resources = recipe['resources']
    cpu, dispatch, memory, case = (resources[name] for name in ('selected_cpu', 'dispatch', 'memory', 'case'))
    workers = release['workers']
    _require(cost_ref['schema'] == calibration.SCHEMA and cost == read_artifact(cost_ref)
        and cost['execution_recipe'] == recipe_ref and cost['calibration_is_section_only'] is True
        and cost['all_outputs_bitwise'] is True and cost['automatic_admission'] is False
        and cost['full_entry_cost_ns'] is None and recipe['profile'] == PROFILE
        and recipe['artifacts']['parallel_scope'] == source_ref
        and cpu['worker_cpu_ids'] == workers['cpu_ids'] == cost['worker_cpu_ids']
        and cpu['owner_cpu_id'] == release['owner_cpu'] == cost['owner_cpu_id']
        and cpu['cpu_numa_node'] == release['numa_node'] and cpu['launch_cpu_ids'] == release['launch_cpu_ids']
        and dispatch['window'] == workers['window'] == cost['window'] == len(workers['cpu_ids'])
        and dispatch['one_inflight_per_worker'] is True
        and memory['worker_address_space_limit_bytes'] == workers['address_space_limit_bytes']
        and memory['minimum_launch_headroom_bytes'] == release['memory_floor_bytes']
        and case == dict(call_count=release['count'], **release['matrix'])
        and lowering['recipe_id'] == recipe['recipe_id'] and read_artifact(lowering['recipe']) == recipe
        and lowering['execution'] == 'EXISTING_ORDERED_PROCESS_SPARSE_BACKEND'
        and lowering['native'] == recipe['implementation']['backend_capability_record']['native'] == release['native'],
        'team model must start from a published measured typed recipe, native carrier and exact case')
    return release, recipe, cost


def _hardware_candidates(hardware, release):
    owner = release['owner_cpu']
    numa = release['numa_node']
    baseline = release['workers']['cpu_ids']
    baseline_lanes = _lanes(hardware, [owner, *baseline], numa)
    cores = {tuple(lane['physical_core']) for lane in baseline_lanes}
    siblings = {cpu for lane in baseline_lanes for cpu in lane['observed_smt_siblings']}
    additions, excluded = [], []
    allowed = set(hardware['launching_affinity'])
    for row in sorted(hardware['processors'], key=lambda item: item['cpu_id']):
        cpu = row['cpu_id']
        if cpu == owner or cpu in baseline:
            continue
        if row['permitted'] is not True or cpu not in allowed:
            reason = 'CPU_NOT_PERMITTED_OR_ONLINE'
        elif row['numa_nodes'] != [numa]:
            reason = 'OUTSIDE_OWNER_NUMA_DOMAIN'
        elif (row['core_id'] is None or row['siblings'] is None or cpu not in row['siblings']):
            reason = 'NO_EXACT_PHYSICAL_CORE_OR_SMT_EVIDENCE'
        else:
            lane, = _lanes(hardware, [cpu], numa)
            core = tuple(lane['physical_core'])
            if core not in cores and siblings.isdisjoint(lane['observed_smt_siblings']):
                cores.add(core)
                siblings.update(lane['observed_smt_siblings'])
                additions.append(cpu)
                continue
            reason = 'SMT_SIBLING_OR_PHYSICAL_CORE_ALREADY_USED'
        excluded.append(dict(cpu_id=cpu, reason=reason))
    return additions, excluded


def _capacity(hardware, release, recipe, width):
    resources = recipe['resources']
    memory, case = resources['memory'], resources['case']
    n, nnz = case['size'], case['nnz']
    fixed = memory['owner_detached_snapshot_bound_bytes'] + memory['owner_structure_bytes']
    slot = 8 * (nnz + 2*n)
    cap = memory['worker_address_space_limit_bytes']
    need = fixed + width*(cap + slot)
    _require(width != release['workers']['window'] or need == memory['minimum_launch_headroom_bytes'],
        'model and published recipe disagree on the address-space and owner-payload floor')
    numa = hardware['node_free_snapshot'].get(str(release['numa_node']), {})
    free = numa.get('bytes')
    cgroup = hardware['finite_memory_headroom_snapshot_bytes']
    quota = hardware['cpu_quota_equivalents']
    reasons = []
    if not hardware['hierarchy_observation_complete']:
        reasons.append('CGROUP_HIERARCHY_UNRESOLVED')
    if quota is not None and quota[0] < (width+1)*quota[1]:
        reasons.append('CPU_QUOTA_BELOW_OWNER_AND_WORKER_DEMAND')
    if numa.get('allowed') is not True or not isinstance(free, int):
        reasons.append('NUMA_HEADROOM_UNOBSERVED')
    elif free < need:
        reasons.append('NUMA_HEADROOM_BELOW_DECLARED_FLOOR')
    if cgroup is not None and cgroup < need:
        reasons.append('CGROUP_HEADROOM_BELOW_DECLARED_FLOOR')
    details = dict(minimum_launch_headroom_bytes=need, owner_fixed_payload_bytes=fixed,
        owner_slot_payload_bytes=slot, worker_address_space_limit_bytes=cap,
        worker_address_space_sum_bytes=width*cap, cgroup_memory_headroom_bytes=cgroup,
        numa_free_snapshot_bytes=free, cpu_quota_equivalents=quota, physical_reservation=False)
    return details, reasons


def _project(recipe, cost, width, anchor):
    spans, components = {}, {}
    for key in _ESTIMATES:
        service = {name: row[key] for name, row in cost['ordered'].items()}
        phases = {name: row[key] for name, row in cost['phase_cost_ledger']['measured'].items()}
        observed_team_boot = service['team_spawn_and_analyze_ns']
        non_team_boot = service['bootstrap_ns'] - observed_team_boot
        _require(non_team_boot >= 0 and observed_team_boot > 0 and service['close_ns'] >= 0,
            'bootstrap and close terms must identify a positive measured worker lifecycle')
        team_boot = round(Fraction(observed_team_boot * width, anchor))
        close = round(Fraction(service['close_ns'] * width, anchor))
        profile = dict(service, bootstrap_ns=non_team_boot + team_boot, close_ns=close)
        projection = project_recipe_window(recipe, profile, phases, width=width,
            include_first_wave=width == anchor and key == 'estimate_ns')
        _require(width != anchor or projection == cost['projections'][key],
            'counterfactual projection changed the observed team anchor')
        spans[key] = projection['elapsed_projection_ns']
        components[key] = dict(non_team_bootstrap_ns=non_team_boot, projected_team_bootstrap_ns=team_boot,
            projected_close_ns=close, modeled_ordered_wait_ns=projection['ordered_wait_ns'],
            modeled_loop_ns=spans[key] - profile['bootstrap_ns'] - close)
    return spans, components


def derive_team_frontier(release_ref, hardware_ref):
    """Project every nested width with the original owner and current core evidence."""
    _require((release_ref.get('schema'), hardware_ref.get('schema')) == (RELEASE, HARDWARE),
        'a published ordered sparse selection and original hardware observation are required')
    release, recipe, cost = _anchor(release_ref)
    hardware = read_artifact(hardware_ref)
    _require(hardware['schema_version'] == 1 and hardware['memory_reservation'] is False
        and hardware['memory_placement_enforced'] is False, 'expected a non-reserving target inventory')
    workers = release['workers']['cpu_ids']
    width = len(workers)
    additions, excluded = _hardware_candidates(hardware, release)
    limit = min(release['count'], width + len(additions))
    alternatives, rejected = [], []
    for slots in range(width, limit+1):
        worker_ids = sorted([*workers, *additions[:slots-width]])
        capacity, reasons = _capacity(hardware, release, recipe, slots)
        row = dict(window=slots, worker_cpu_ids=worker_ids, capacity=capacity)
        if reasons:
            rejected.append(dict(row, reasons=reasons))
        else:
            spans, components = _project(recipe, cost, slots, width)
            alternatives.append(dict(row, **spans, model_terms=components, measured_config=slots == width))
    rank = lambda row: (row['estimate_ns'], row['window'])
    expanded = [row for row in alternatives if row['window'] > width]
    selected = min(alternatives, key=rank, default=None)
    challenger = min(expanded, key=rank, default=None)
    return dict(schema=SCHEMA, schema_version=1, released_selection=release_ref, hardware=hardware_ref,
        calibration_seed=release['cost'], parallel_scope=release['source'], source_scope=cost['source_scope'],
        execution_recipe=release['execution_recipe'], native=release['native'],
        case=recipe['resources']['case'], owner_cpu_id=release['owner_cpu'], cpu_numa_node=release['numa_node'],
        baseline=dict(window=width, worker_cpu_ids=workers),
        expansion_cpu_order=additions, excluded_cpus=excluded, physical_worker_limit=limit,
        alternatives=alternatives, rejected=rejected, model_selected=selected, fresh_challenger=challenger,
        recommendation='KEEP_PUBLISHED_RELEASE_PENDING_FRESH_WHOLE_ENTRY_VALIDATION',
        maximum_fresh_configurations=int(challenger is not None), full_entry_cost_ns=None,
        qualification=dict(automatic_admission=False, extrapolated_clock_is_qualification=False,
            layout_space='NESTED_PRESERVE_CURRENT_WORKERS_ADD_LOWEST_INDEPENDENT_LOGICAL_CPU',
            lifecycle_model='FREEZE_NONTEAM_BOOTSTRAP_SCALE_MEASURED_TEAM_BOOTSTRAP_AND_CLOSE_LINEARLY',
            loop_services_frozen_from_anchor=True, service_interval_is_observed_sample_envelope=True,
            fresh_live_resource_preflight_required=True, bandwidth_and_contention_unmodeled=True,
            whole_entry_optimality_proven=False))


def _choice(frontier_ref, frontier):
    row = frontier['fresh_challenger']
    _require(row is not None, 'no wider configuration passed the frozen target capacity checks')
    cost = read_artifact(frontier['calibration_seed'])
    previous_choice = read_artifact(cost['choice'])
    window = row['window']
    base = frontier['baseline']['window']
    capacity = row['capacity']
    finite = [limit for limit in (capacity['numa_free_snapshot_bytes'], capacity['cgroup_memory_headroom_bytes'])
              if limit is not None]
    headroom = min(finite)
    fixed, slot = capacity['owner_fixed_payload_bytes'], capacity['owner_slot_payload_bytes']
    cap = capacity['worker_address_space_limit_bytes']
    owner_service = round(Fraction(cost['projections']['estimate_ns']['serialized_submit_then_publish_owner_service_ns'],
        frontier['case']['call_count']))
    worker_service = round(Fraction(cost['projections']['estimate_ns']['elapsed_worker_service_sum_ns'],
        frontier['case']['call_count']))
    _require(owner_service > 0 and worker_service > 0 and window > base,
        'fresh challenger requires a larger valid measured service anchor')
    saturation = -(-worker_service//owner_service)
    return dict(schema=CHOICE, schema_version=1, hardware=frontier['hardware'], team_frontier=frontier_ref,
        released_selection=frontier['released_selection'], parallel_scope=frontier['parallel_scope'],
        calibration_seed=frontier['calibration_seed'], scope_baseline=previous_choice['scope_baseline'],
        owner_service_seed_ns=owner_service, worker_service_seed_ns=worker_service,
        empirical_resource_reservation=False, worker_set_order='RETAIN_PUBLISHED_TEAM_THEN_ASCENDING_INDEPENDENT_CPU',
        dimensions=dict(call_count=frontier['case']['call_count'], max_ordered_window=frontier['physical_worker_limit']),
        budget=dict(worker_total_limit_bytes=cap, total_cap_bytes=headroom, fixed_owned_bytes=fixed,
            owner_slot_bound_bytes=slot),
        choice=dict(owner_cpu_id=frontier['owner_cpu_id'], worker_cpu_ids=row['worker_cpu_ids'],
            worker_count=window, window=window, cpu_numa_node=frontier['cpu_numa_node'],
            physical_worker_limit=frontier['physical_worker_limit'],
            owner_saturation_estimate_workers=max(window, saturation),
            memory_worker_limit=max(0, (headroom-fixed)//(cap+slot)), memory_bytes_per_inflight_worker=cap+slot,
            modeled_owned_memory_bytes=capacity['minimum_launch_headroom_bytes'], memory_bound_inputs_complete=True,
            uses_smt_siblings_as_extra_cores=False, resource_snapshot_is_reservation=False, automatic_admission=False))


def _reference(value, schema):
    return value if isinstance(value, dict) else artifact_reference(Path(value).resolve(), schema)


def check(value):
    """Recompute the frontier from frozen original admission, hardware and raw W8 clocks."""
    reference = _reference(value, SCHEMA)
    _require(reference['schema'] == SCHEMA, 'expected an ordered sparse team frontier')
    result = read_artifact(reference)
    _require(result == derive_team_frontier(result['released_selection'], result['hardware']),
        'saved team frontier differs from forward hardware and measured anchor replay')
    return result


def check_choice(value):
    """Check one invitation against its full frontier and exact release lineage."""
    reference = _reference(value, CHOICE)
    _require(reference['schema'] == CHOICE, 'expected an ordered sparse preliminary choice')
    choice = read_artifact(reference)
    frontier = check(choice['team_frontier'])
    _require(choice == _choice(choice['team_frontier'], frontier),
        'saved challenger differs from the frozen forward model')
    return choice


def _write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, value['schema'])


def publish(release, hardware, out, *, cpu_ids=None):
    directory = Path(out).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    release = _reference(release, RELEASE)
    _require((hardware is None) != (cpu_ids is None), 'choose either a hardware receipt or a CPU list to collect')
    if hardware is None:
        hardware = _write(directory/'hardware.json', calibration.collect_hardware(cpu_ids))
    else:
        hardware = _reference(hardware, HARDWARE)
    frontier = derive_team_frontier(release, hardware)
    frontier_ref = _write(directory/'frontier.json', frontier)
    choice_ref = None if frontier['fresh_challenger'] is None else _write(directory/'choice.json', _choice(frontier_ref, frontier))
    return dict(frontier=frontier_ref, choice=choice_ref)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release')
    hardware = parser.add_mutually_exclusive_group()
    hardware.add_argument('--hardware')
    hardware.add_argument('--cpu-ids', type=_cpu_list)
    parser.add_argument('--out')
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument('--check')
    checks.add_argument('--check-choice')
    args = parser.parse_args(argv)
    if args.check or args.check_choice:
        if any((args.release, args.out, args.hardware, args.cpu_ids)):
            parser.error('a check accepts only its saved receipt')
        result = check(args.check) if args.check else check_choice(args.check_choice)
    elif args.release and args.out and (args.hardware or args.cpu_ids):
        result = publish(args.release, args.hardware, args.out, cpu_ids=args.cpu_ids)
    else:
        parser.error('use --release with --hardware or --cpu-ids, and --out')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
