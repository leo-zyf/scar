"""Replay scoped CPU array lanes and empty dispatch without entry-time bounds."""

from scar.analysis.replay_transaction import replay_transaction
from scar.ir.execution_recipe import read_artifact
from . import independent_array_task_schedule as process, independent_array_task_team_schedule as team
from .cpu_entry_forward_cost import _interval, _replay_team_components, _scheduled_lane_work, _sum

SCHEMA = 'scar.cpu-array-native-route-component-cost'
_ROUTE_PROOF = 'scar.cpu-array-native-route-proof'
_GROUPED_PROBE = 'scar.cpu-array-native-route-body-observation'
_PROVEN = 'PROVEN_SCOPED_CPU_ARRAY_NATIVE_ROUTE'
_SIDES = ('lower_ns', 'estimate_ns', 'upper_ns')
_UNPRICED = ('ENTRY_LAUNCH_AND_LOADER', 'INPUT_INGRESS_AND_RUNTIME_GUARDS',
             'LIVE_DISPATCH_NUMERICAL_INTERACTION', 'FINAL_REPORT_FILE_STDOUT_AND_EXIT')
_DISPATCH = 'include/scar_generated/native_ordered_dispatch.hpp'
_OPENMP = 'include/scar_generated/native_ordered_openmp.hpp'


def _require(condition, detail):
    if not condition:
        raise ValueError('CPU array native route component cost: ' + detail)


def _matching_choice(document, placement):
    matches = [(index, row) for index, row in enumerate(document['alternatives'])
               if all(row[key] == value for key, value in placement.items())]
    _require(len(matches) == 1 and not matches[0][1]['capacity_reasons'] and matches[0][1]['cost'] is not None,
             'component schedule has no unique eligible row for the exact deployment lanes and limits')
    return matches[0]


def _same_bytes(left, right):
    return all(left[key] == right[key] for key in ('size_bytes', 'sha256'))


def _origins(source):
    evidence, scope, resources, workload = (source[name] for name in ('evidence', 'scope', 'resources', 'workload'))
    schedule_ref = evidence['schedule']
    schedule = read_artifact(schedule_ref)
    process_ref, team_ref = schedule['process_plan'], schedule['thread_evidence']
    standalone_plan, team_probe = read_artifact(process_ref), read_artifact(team_ref)
    local, grouped = read_artifact(source['retained_runtime']), read_artifact(source['baseline_grouped'])
    recipe = read_artifact(workload['recipe'])
    _require(schedule_ref['schema'] == team.SCHEMA and local['schedule'] == grouped['schedule'] == schedule_ref
             and schedule['scope'] == standalone_plan['scope'] == local['scope'] == grouped['scope'] == scope
             and schedule['hardware'] == standalone_plan['hardware'] == resources['hardware']
             and schedule['native'] == standalone_plan['native'] == local['native'] == evidence['original_numerical_body']
             and local['recipe'] == grouped['recipe'] == workload['recipe']
             and recipe['environment'] == workload['local_process_environment']
             and team_probe['thread_environment'] == workload['openmp_grouped_environment']
             and team_probe['openmp_runtime'] == evidence['providers']['openmp_runtime']['libgomp.so.1']
             and local['files'][local['binary']] == evidence['retained_executable']
             and grouped['executable'] == evidence['openmp_executable'],
             'component observations do not bind the original numerical package, target or launch contexts')
    provider = read_artifact(schedule['native'])
    profile, thread_environment, _ = _replay_team_components(schedule)
    local_probe_ref = standalone_plan['evidence']
    local_probe = read_artifact(local_probe_ref)
    _require(local['numerical_body_reused'] == profile and provider['recipe'] == workload['recipe']
             and all(_same_bytes(local['files'][name], provider['files'][name])
                     for name in ('src/tape_0.inc', *(f'src/{name}' for name in team._NATIVE_SOURCES)))
             and _same_bytes(local['files'][_DISPATCH], local_probe['dispatch_header'])
             and _same_bytes(grouped['files'][_OPENMP], team_probe['sources']['ordered_openmp.hpp'])
             and thread_environment == workload['openmp_grouped_environment'],
             'the numerical sources, dispatch ABI or OpenMP probe environment differ from the retained routes')
    process_index, process_row = _matching_choice(standalone_plan, resources['local_processes'])
    _, local_row = _matching_choice(schedule, resources['local_processes'])
    team_index, team_row = _matching_choice(schedule, resources['openmp_grouped'])
    _require(local['decision'] == local_row and grouped['decision'] == team_row
             and process_row['mode'] == 'ORDERED_LOCAL_PROCESSES' and team_row['mode'] == team.MODE,
             'native route manifests no longer bind the pinned local process and grouped placements')
    return (process_row, team_row,
            dict(schedule=process_ref, locator=f'#/alternatives/{process_index}', samples=local_probe_ref),
            dict(schedule=schedule_ref, locator=f'#/alternatives/{team_index}', samples=team_ref))


def _lane_counts(scope, choice):
    tasks, cpus = scope['task_count'], choice['worker_cpu_ids']
    _require(type(tasks) is int and type(cpus) is list and 0 < len(cpus) <= tasks
             and choice['worker_count'] == choice['window'] == len(cpus)
             and len(set(cpus)) == len(cpus), 'fixed non-empty ordinal lane width required')
    waves, remaining = divmod(tasks, len(cpus))
    return [dict(cpu_id=cpu, first_ordinal=index, ordinal_stride=len(cpus),
                 task_count=waves + (index < remaining)) for index, cpu in enumerate(cpus)]


def _conditional(scope, choice, cost, *, openmp):
    lanes = _lane_counts(scope, choice)
    services = cost['worker_service_intervals_ns']
    _require(set(services) == {str(row['cpu_id']) for row in lanes}, 'lane observations differ from original placement')
    for row in lanes:
        service = _interval(services[str(row['cpu_id'])])
        row.update(per_task_service_interval_ns=service,
                   numeric_work_interval_ns={side: service[side] * row['task_count'] for side in _SIDES})
    transport = _interval(cost['additive_transport_interval_ns'])
    if openmp:
        model = _scheduled_lane_work(
            dict(lanes=lanes, task_count=scope['task_count'], worker_count=len(lanes)),
            scope, dict(choice, cost=cost))
        numeric = model['critical_path_ns']
    else:
        numeric = {side: process._ordered_time(scope['task_count'],
                   [row['per_task_service_interval_ns'][side] for row in lanes]) for side in _SIDES}
    additive = _sum([numeric, transport])
    _require(numeric == _interval(cost['ordered_numeric_completion_ns'])
             and additive == _interval(cost['modeled_elapsed_ns']),
             'conditional lane order or independent empty transport does not reproduce the pinned model')
    return dict(per_lane=lanes, conditional_numeric_critical_interval_ns=numeric,
                transport_proxy_interval_ns=transport, conditional_additive_component_interval_ns=additive)


def _grouped_cost(reference, proof_ref, source, choice):
    from scar.compiler.calibrate_cpu_array_native_route_body import _verify

    _require(reference['schema'] == _GROUPED_PROBE, 'exact independently measured grouped body observation required')
    probe = _verify(reference, proof_ref, source)
    _require(probe['proof'] == proof_ref and probe['route'] == 'grouped_openmp'
             and probe['executable'] == source['evidence']['openmp_executable']
             and probe['scope'] == source['scope']
             and probe['cpu_ids'] == choice['launch_cpu_ids'] == source['resources']['launch_cpu_ids']
             and probe['worker_cpu_ids'] == choice['worker_cpu_ids'],
             'grouped numerical observations cross package, launch scope or lane placement')
    return _conditional(source['scope'], choice, probe['conditional_model'], openmp=True)


def derive_cpu_array_native_route_component_cost(proof_ref, *, grouped_probe=None):
    """Bind lane service and independent dispatch inputs to the exact two native routes."""
    from scar.analysis.cpu_array_native_route import verify_cpu_array_native_route

    _require(proof_ref['schema'] == _ROUTE_PROOF, 'scoped array native route proof required')
    with replay_transaction():
        source = verify_cpu_array_native_route(proof_ref)
        return _project(proof_ref, source, grouped_probe)


def _project(proof_ref, source, grouped_probe):
    _require(source['status'] == _PROVEN, 'route proof did not establish exact scope')
    local, grouped, local_evidence, team_evidence = _origins(source)
    process_components = _conditional(source['scope'], local, local['cost'], openmp=False)
    exact_grouped = None if grouped_probe is None else _grouped_cost(grouped_probe, proof_ref, source, grouped)
    expected_grouped_transport = _interval(grouped['cost']['additive_transport_interval_ns'])
    grouped_components = (dict(per_lane=_lane_counts(source['scope'], grouped),
        conditional_numeric_critical_interval_ns=None, transport_proxy_interval_ns=expected_grouped_transport,
        conditional_additive_component_interval_ns=None) if exact_grouped is None else exact_grouped)
    routes = dict(
        retained_process=dict(**process_components,
            executable=source['evidence']['retained_executable'], resources='#/resources/local_processes',
            numeric_basis='UNCONTENDED_SCOPED_PROCESS_WORKER_SERVICE_ORDERED_RETIRE_REFILL',
            transport_basis='ISOLATED_SAME_ABI_EMPTY_PROCESS_DISPATCH',
            provenance=dict(numerical=local_evidence, transport=local_evidence),
            assumptions=['UNCONTENDED_SERVICE_TRANSFERS_TO_ORDERED_PROCESS_WAVES',
                         'INDEPENDENT_EMPTY_TRANSPORT_ADDED_AS_COMPONENT_PROXY'],
            unpriced_phases=list(_UNPRICED)),
        grouped_openmp=dict(**grouped_components,
            executable=source['evidence']['openmp_executable'], resources='#/resources/openmp_grouped',
            numeric_basis=('EXACT_GROUPED_BODY_ONE_SIMULTANEOUS_TASK_PER_LANE' if exact_grouped is not None
                           else 'EXACT_GROUPED_BODY_SERVICE_UNOBSERVED'),
            transport_basis='ISOLATED_SAME_ABI_EMPTY_OPENMP_DISPATCH',
            provenance=(dict(grouped_probe=grouped_probe, locator='#/conditional_model')
                        if exact_grouped is not None else dict(transport=team_evidence)),
            historical_ungrouped_numeric_transferred=False,
            assumptions=(['PER_LANE_SERVICE_TRANSFERS_TO_MULTI_WAVE_EXECUTION',
                          'INDEPENDENT_EMPTY_TRANSPORT_ADDED_AS_COMPONENT_PROXY'] if exact_grouped is not None else []),
            unpriced_phases=list(_UNPRICED) + (['EXACT_GROUPED_NUMERICAL_SERVICE'] if exact_grouped is None else [])))
    return dict(schema=SCHEMA, schema_version=1, proof=proof_ref, grouped_probe=grouped_probe,
        retention=source['retention'], grouped=source['baseline_grouped'], scope=source['scope'],
        resources=source['resources'], routes=routes,
        status='SCOPED_COMPONENT_MODELS' if exact_grouped is not None else 'GROUPED_NUMERICAL_COMPONENT_OPEN',
        interval_basis='EMPIRICAL_SAMPLE_CONDITIONED_SERVICE_MODEL',
        full_entry_interval_ns=dict(lower=None, upper=None), physical_duration_bound_proven=False,
        whole_entry_choice_proven=False, forward_cost_proven=False, deployment_authorized=False,
        execution_default='NO_OP', target_executed=False)


def verify_cpu_array_native_route_component_cost(reference):
    """Replay the exact route, conditional cost inputs and report without launching a target."""
    saved = read_artifact(reference)
    _require(reference['schema'] == SCHEMA and type(saved['schema_version']) is int and saved['schema_version'] == 1,
             'unsupported CPU array route component cost report')
    replay = derive_cpu_array_native_route_component_cost(saved['proof'], grouped_probe=saved['grouped_probe'])
    _require(saved == replay, 'saved native route component cost differs at cold replay')
    return replay
