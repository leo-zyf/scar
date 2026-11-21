"""Screen a target MPI plan using already measured rank bootstrap costs."""

from scar.ir.execution_recipe import read_artifact
from scar.optimization import independent_array_task_mpi_schedule as mpi
from scar.optimization import independent_array_task_schedule as base


SCHEMA = 'scar.independent-array-task-mpi-preflight'
_PLACEMENT = ('mode', 'worker_count', 'owner_cpu_id', 'worker_cpu_ids', 'launch_cpu_ids', 'window')
_CAPS = ('worker_address_space_limit_bytes', 'coordinator_address_space_limit_bytes',
         'launcher_address_space_limit_bytes', 'thread_stack_bytes')
_BUFFERS = ('retained_result_payload_bytes', 'payload_budget_bytes', 'explicit_owner_payload_bytes',
            'explicit_worker_payload_bytes')
_REJECTED = 'DOMINATED_BY_LOCAL_WITH_BOOTSTRAP_ALONE'


def _need(condition, reason):
    if not condition:
        raise ValueError('MPI preflight: ' + reason)


def _content(reference):
    return reference['schema'], reference['size_bytes'], reference['sha256']


def _platform(target, target_team, donor, donor_team):
    common = ('family', 'native', 'hardware', 'runtime', 'rank_limit_bytes', 'rank_placement', 'common_processes')
    _need(all(target[key] == donor[key] for key in common),
          'hardware snapshot, numerical source, MPI identity, rank binding or launcher caps differ')
    common_team = ('allowed_cpu_lanes', 'dispatch', 'thread_domain', 'thread_contract', 'openmp_runtime')
    _need(all(target_team[key] == donor_team[key] for key in common_team),
          'local CPU or execution contract differs')
    target_cost = read_artifact(target_team['thread_evidence'])
    donor_cost = read_artifact(donor_team['thread_evidence'])
    _need(all(target_cost[key] == donor_cost[key] for key in ('compiler_binary', 'openmp_runtime', 'thread_environment'))
          and _content(target_cost['probe']) == _content(donor_cost['probe']),
          'local execution probe, runtime or launch environment differs')


def _placement(target, donor):
    _need(all(target[key] == donor[key] for key in _PLACEMENT)
          and all(target['memory'].get(key) == donor['memory'].get(key) for key in _CAPS),
          'matching mode and width have different CPU placement or memory caps')


def _local_bounds(team, donor_rows):
    bounds = []
    for row in team['alternatives']:
        prior = donor_rows.get((row['mode'], row['worker_count']))
        if prior is None or prior['cost'] is None or row['capacity_reasons'] or row['cost'] is None:
            continue
        _placement(row, prior)
        # The donor's outside-dispatch interval includes input setup and output handling.
        if any(row['memory'][key] > prior['memory'][key] for key in _BUFFERS):
            continue
        internal = row['cost']['modeled_elapsed_ns']['upper_ns']
        outside = prior['cost']['host_startup_and_return_ns']['upper_ns']
        bounds.append(dict(mode=row['mode'], worker_count=row['worker_count'], worker_cpu_ids=row['worker_cpu_ids'],
                           target_internal_upper_ns=internal, platform_startup_return_upper_ns=outside,
                           modeled_upper_ns=internal + outside))
    return bounds


def _outcome(row, donor_rows, local):
    width = row['worker_count']
    base_row = dict(rank_count=width, rank_cpu_ids=row['worker_cpu_ids'],
                    capacity_reasons=row['capacity_reasons'], mpi_lower_ns=None)
    if row['capacity_reasons']:
        return dict(base_row, outcome='INSUFFICIENT_RESOURCE_CAPACITY')
    prior = donor_rows.get((mpi.MODE, width))
    if prior is None or prior['cost'] is None:
        return dict(base_row, outcome='UNRESOLVED_WITHOUT_MATCHED_BOOTSTRAP')
    _placement(row, prior)
    _need(row['launch_prefix'] == prior['launch_prefix'], 'matching MPI width has a different launcher prefix')
    observed = prior['cost']['mpi_empty_component_observations_ns']['max_rank_bootstrap_ns']
    bootstrap = observed['lower_ns']
    if local is None:
        status = 'UNRESOLVED_WITHOUT_MATCHED_LOCAL_UPPER'
    elif bootstrap > local['modeled_upper_ns']:
        status = _REJECTED
    else:
        status = 'UNRESOLVED_BOOTSTRAP_ENVELOPE_OVERLAPS_LOCAL'
    return dict(base_row, outcome=status, mpi_lower_ns=bootstrap, platform_rank_bootstrap_ns=observed,
                optimistic_unpriced_work_ns=0, local_upper_ns=local['modeled_upper_ns'] if local else None)


def project(graph_path, platform_plan_path):
    """Replay a checked platform receipt without running the target MPI plan."""
    target_ref, target, team, _, _ = mpi._inputs(graph_path)
    donor_ref = base._ref(platform_plan_path, mpi.SCHEMA)
    donor_plan = mpi.check(donor_ref['uri'])
    donor = read_artifact(donor_plan['mpi_graph'])
    donor_team = read_artifact(donor['team_plan'])
    _platform(target, team, donor, donor_team)

    donor_rows = {(row['mode'], row['worker_count']): row for row in donor_plan['alternatives']}
    bounds = _local_bounds(team, donor_rows)
    best = min(bounds, key=lambda row: row['modeled_upper_ns'], default=None)
    outcomes = [_outcome(row, donor_rows, best) for row in target['alternatives']]
    eligible = [row for row in outcomes if not row['capacity_reasons']]
    rejected = bool(eligible) and all(row['outcome'] == _REJECTED for row in eligible)
    if not eligible:
        status = 'NO_MPI_CANDIDATE_FITS_RESOURCE_BOUNDS'
    elif rejected:
        status = 'ALL_ELIGIBLE_MPI_DOMINATED_WITH_REUSED_BOOTSTRAP'
    else:
        status = 'MPI_UNRESOLVED_WITHOUT_TARGET_TRIAL'
    selected = team['decision']['selected'] if rejected else None
    if selected is not None:
        minimum = min(row['mpi_lower_ns'] for row in eligible)
        if not any(row['mode'] == selected['mode'] and row['worker_count'] == selected['worker_count']
                   and row['modeled_upper_ns'] < minimum for row in bounds):
            selected = None
    return dict(schema=SCHEMA, schema_version=1, mpi_graph=target_ref, team_plan=target['team_plan'],
                scope=target['scope'], platform_plan=donor_ref, platform_mpi_graph=donor_plan['mpi_graph'],
                platform_evidence=donor_plan['mpi_evidence'], platform_scope=donor_plan['scope'],
                local_bounds=bounds, best_local_bound=best,
                decision=dict(status=status, all_eligible_mpi_rejected=rejected, mpi_outcomes=outcomes,
                              selected=selected, deployment_authorized=False),
                cost_boundary=dict(basis='REUSED_EMPIRICAL_SAMPLE_ENVELOPES_ON_IDENTICAL_PLATFORM_CONTRACT',
                    mpi_cost='MIN_OBSERVED_MAX_RANK_BOOTSTRAP_BEFORE_FIRST_NUMERICAL_CALL',
                    local_cost='TARGET_TEAM_INTERNAL_UPPER_PLUS_MATCHED_PLATFORM_OUTSIDE_DISPATCH_UPPER',
                    empirical_ranges_are_universal_bounds=False, target_mpi_trial_run=False,
                    target_mpi_executable_generated=False), deployment_authorized=False)


def check(path):
    stored = read_artifact(base._ref(path, SCHEMA))
    actual = project(stored['mpi_graph']['uri'], stored['platform_plan']['uri'])
    _need(stored == actual, 'stored preflight differs during replay')
    return actual
