"""Replay conditional CPU component costs without manufacturing entry bounds."""

from hashlib import sha256
import json

from scar.ir.execution_recipe import read_artifact
from .cpu_entry_capacity_contract import _ARRAY, _array_runtime
from .cpu_entry_decision_ledger import verify_cpu_entry_decision_ledger
from .deployment_cost import _pointer
from .whole_entry_choice import LIFECYCLE_PHASES


SCHEMA = 'scar.cpu-entry-forward-cost'
LAUNCH_PROXY = 'scar.cpu-launch-proxy-observation'
TEAM_PROBE = 'scar.independent-array-task-openmp-cost-evidence'
_LIFECYCLE = 'scar.f4-cpu-entry-lifecycle-graph'
_TEAM = 'scar.independent-array-task-team-schedule'
_CATALOG_FAMILY = 'CATALOG_CPU_ENTRY'
_STREAM = 'scar.cpu-recursive-stream-source-provider'
_SIDES = ('lower_ns', 'estimate_ns', 'upper_ns')


def _require(condition, message):
    if not condition:
        raise ValueError('CPU entry forward cost: ' + message)


def _interval(values):
    _require(type(values) is dict and set(values) == set(_SIDES)
             and all(type(values[key]) is int and values[key] >= 0 for key in _SIDES)
             and values['lower_ns'] <= values['estimate_ns'] <= values['upper_ns'],
             'sample-conditioned model terms need ordered integral durations')
    return dict(values)


def _sum(components):
    return {key: sum(values[key] for values in components) for key in _SIDES}


def _team_rank(schedule, repeat_rows=None):
    repeated = {} if repeat_rows is None else {(row['mode'], row['worker_count']): row for row in repeat_rows}
    modeled, alternatives = [], []
    missing = False
    seen = set()
    for option in schedule['alternatives']:
        id_ = f"{option['mode']}:{option['worker_count']}"
        _require(id_ not in seen, 'duplicate team mode and width')
        seen.add(id_)
        reasons = option['capacity_reasons']
        cost = option['cost']
        other = None
        if repeat_rows is not None:
            alternate = repeated.pop((option['mode'], option['worker_count']), None)
            _require(alternate is not None and {key: val for key, val in alternate.items() if key != 'cost'}
                     == {key: val for key, val in option.items() if key != 'cost'},
                     'repeated observations differ from the original placement or capacity')
            other = alternate['cost']
        interval, repeat_interval = None, None
        if not reasons and cost is not None and (repeat_rows is None or other is not None):
            terms = [_interval(cost[key]) for key in ('ordered_numeric_completion_ns', 'additive_transport_interval_ns')]
            interval = _interval(cost['modeled_elapsed_ns'])
            _require(interval == _sum(terms), 'schedule body prediction does not add its isolated components')
            if other is not None:
                repeat_interval = _interval(other['modeled_elapsed_ns'])
                interval = dict(interval, lower_ns=min(interval['lower_ns'], repeat_interval['lower_ns']),
                                upper_ns=max(interval['upper_ns'], repeat_interval['upper_ns']))
            modeled.append((id_, interval))
        elif not reasons:
            missing = True
        alternatives.append(dict(id=id_, mode=option['mode'], worker_count=option['worker_count'],
            owner_cpu_id=option['owner_cpu_id'], worker_cpu_ids=option['worker_cpu_ids'],
            capacity_reasons=reasons, conditional_interval_ns=interval, repeat_interval_ns=repeat_interval))
    _require(not repeated, 'repeat contains extra placements outside the retained schedule')
    alternatives.sort(key=lambda row: row['id'])
    modeled.sort(key=lambda row: (row[1]['estimate_ns'], row[0]))
    nomination, separation = None, []
    status = 'NO_COMPLETE_FINITE_BODY_COMPARISON' if missing or not modeled else 'EMPIRICAL_MODEL_ENVELOPES_OVERLAP'
    if modeled and not missing:
        leader, interval = modeled[0]
        separation = [dict(other_id=id_, margin_ns=other['lower_ns'] - interval['upper_ns'])
                      for id_, other in modeled[1:]]
        if all(row['margin_ns'] > 0 for row in separation):
            nomination = leader
            status = 'STRICTLY_SEPARATED_CONDITIONAL_BODY_MODEL' if separation else 'SINGLE_FEASIBLE_BODY_PLACEMENT'
    return dict(status=status, alternatives=alternatives, provisional_body_nomination_id=nomination,
                separation=separation, calibration_campaigns=1 if repeat_rows is None else 2,
                envelope_policy='RETAINED_ENVELOPE' if repeat_rows is None else 'UNION_OF_REPLAYED_CAMPAIGN_ENVELOPES',
                physical_optimality_proven=False, whole_entry_choice_proven=False)


def _replay_team_components(schedule, repeat_probe=None):
    from . import independent_array_task_schedule as base, independent_array_task_team_schedule as team

    process_ref = schedule['process_plan']
    process = read_artifact(process_ref)
    process_probe = read_artifact(process['evidence'])
    initial = read_artifact(schedule['thread_evidence'])
    native, hardware = map(read_artifact, (process['native'], process['hardware']))
    _require(process_ref['schema'] == base.SCHEMA and schedule['thread_evidence']['schema'] == team.EVIDENCE
             and initial['process_plan'] == process_ref and initial['thread_contract'] == schedule['thread_contract']
             and initial['replayable_kernel_state_qualification'] == schedule['thread_contract']
             and initial['native'] == process['native'] == schedule['native']
             and process['scope'] == schedule['scope'], 'isolated service evidence crosses provider scopes')
    # The ledger already pins the source and provider. Replay the numerical
    # probes and resource arithmetic without rebuilding that original source graph.
    limits = {name: process_probe[name] for name in ('budget_bytes', 'worker_address_space_limit_bytes',
                                                    'coordinator_address_space_limit_bytes')}
    base_rows = base._derive(process['native'], process['hardware'], process['evidence'],
                             owner_cpu=process_probe['owner_cpu_id'], **process['scope'], **limits)['alternatives']
    _require(base_rows == process['alternatives'], 'serial and process probe rows differ at replay')
    team_rows = team._thread_rows(process, native, hardware)
    eligible = team._eligible(process, team_rows)

    def replay(evidence):
        intervals = team._intervals(evidence, process, native, eligible)
        return [base_rows[0], *(dict(row, cost=intervals.get((row['mode'], row['worker_count'])))
                               for row in (*base_rows[1:], *team_rows))]

    _require(replay(initial) == schedule['alternatives'] and initial['openmp_runtime'] == schedule['openmp_runtime'],
             'team timing differs from replayed dispatch and kernel probes')
    repeated = None
    if repeat_probe is not None:
        evidence = read_artifact(repeat_probe)
        keys = ('schema', 'schema_version', 'process_plan', 'thread_contract', 'native', 'thread_environment',
                'boundary', 'replayable_kernel_state_qualification', 'compiler_binary', 'openmp_runtime')
        _require(all(evidence[key] == initial[key] for key in keys),
                 'repeat must bind the same provider proof, toolchain, hardware placement and service boundary')
        repeated = replay(evidence)
        width = process['scope']['task_count']
        _require(all(abs(previous - fresh) >= width for previous in initial['seeds'] for fresh in evidence['seeds']),
                 'repeat requires disjoint calibration seed ranges')
    return native['files']['src/profile_0.cpp'], initial['thread_environment'], repeated


def _scheduled_lane_work(constraints, scope, selected):
    lanes = constraints['lanes']
    cost = selected['cost']
    services = cost['worker_service_intervals_ns']
    _require(constraints['task_count'] == scope['task_count']
             and constraints['worker_count'] == selected['worker_count']
             and [row['cpu_id'] for row in lanes] == selected['worker_cpu_ids']
             and set(services) == {str(row['cpu_id']) for row in lanes},
             'scoped lane allocation differs from calibrated service placement')
    per_lane = []
    for row in lanes:
        service = _interval(services[str(row['cpu_id'])])
        per_lane.append(dict(cpu_id=row['cpu_id'], task_count=row['task_count'],
                             modeled_service_ns={side: service[side] * row['task_count'] for side in _SIDES}))
    numeric = {side: max(row['modeled_service_ns'][side] for row in per_lane) for side in _SIDES}
    elapsed = _sum([numeric, _interval(cost['additive_transport_interval_ns'])])
    _require(numeric == cost['ordered_numeric_completion_ns'] and elapsed == cost['modeled_elapsed_ns'],
             'lane-derived critical path does not reproduce the provider body model')
    return dict(per_lane=per_lane, critical_path_ns=numeric, conditional_elapsed_ns=elapsed,
                cost_basis='MATCHED_PROBE_SERVICE_PER_FIXED_LANE', physical_duration_bound=False)


def _team_body(plan, option, repeat_probe=None, *, schedule_constraints=None):
    from .independent_array_task_team_schedule import MODE

    if plan is None or plan['provider_family'] != _ARRAY or not all(plan['context']['scope_matches'].values()):
        return None
    witness = option['memory']['virtual_address_space_headroom']
    if witness['status'] != 'VIRTUAL_ADDRESS_SPACE_CAPS':
        return None
    compiled, native, selected = _array_runtime(plan, witness)
    ref = native['schedule']
    if ref['schema'] != _TEAM or selected['mode'] != MODE:
        return None
    schedule = read_artifact(ref)
    numerical_body, thread_environment, repeated = _replay_team_components(schedule, repeat_probe)
    boundary = schedule['cost_boundary']
    _require(selected in schedule['alternatives'] and not selected['capacity_reasons'] and selected['cost'] is not None
             and native['scope'] == schedule['scope'] and native['thread_contract'] == schedule['thread_contract']
             and selected['launch_cpu_ids'] == plan['context']['cpu_ids']
             and native['numerical_body_reused'] == numerical_body
             and native['linkage']['qualified_runtime_dependencies']['libgomp.so.1'] == schedule['openmp_runtime']
             and compiled['invocation']['environment'] == thread_environment
             and boundary['additive_empty_dispatch'] is True and boundary['seeded_microbenchmarks_only'] is True
             and boundary['runtime_environment_and_publication_excluded'] is True,
             'team body timing requires exact provider placement, source scope and component boundaries')
    cost = selected['cost']
    body = dict(schedule=ref, provider=plan['authority'], runtime=compiled['runtime'],
        isolated_evidence=schedule['thread_evidence'], numeric_scope=schedule['scope'], model=boundary,
        mode=selected['mode'], worker_count=selected['worker_count'], worker_cpu_ids=selected['worker_cpu_ids'],
        estimated_components_ns=dict(contended_lane_waves=_interval(cost['ordered_numeric_completion_ns']),
                                     isolated_empty_dispatch=_interval(cost['additive_transport_interval_ns'])),
        conditional_interval_ns=_interval(cost['modeled_elapsed_ns']), repeat_probe=repeat_probe,
        placement_comparison=_team_rank(schedule, repeated))
    if schedule_constraints is not None and schedule_constraints['status'] == 'EXACT_ORDINAL_LANE_CONSTRAINTS':
        body['lane_work_model'] = _scheduled_lane_work(schedule_constraints, schedule['scope'], selected)
        body['conditional_interval_ns'] = body['lane_work_model']['conditional_elapsed_ns']
    return body


def _launch_context(scope, plan, original):
    if original:
        return dict(cpu_ids=scope['process_affinity_cpu_ids'], cwd=scope['working_directory'],
            environment_sha256=scope['environment_sha256'], abi='PACED_STDIN' if scope.get('cpu_stream_capture') else 'ARGV')
    _require(plan is not None and all(plan['context']['scope_matches'].values()),
             'launch proxy requires the exact native launch context')
    family = plan['provider_family']
    context = plan['context']
    if family == _ARRAY:
        environment = plan['contract']['profile']['invocation']['environment']
    else:
        _require(family == _STREAM, 'native provider launch environment is not qualified for this proxy')
        environment = context['original_environment']
    digest = sha256(json.dumps(sorted(environment.items()), separators=(',', ':')).encode()).hexdigest()
    return dict(cpu_ids=context['cpu_ids'], cwd=context['cwd'], environment_sha256=digest,
                abi='PACED_STDIN' if family == _STREAM else 'ARGV')


def _launch_probe(reference, scope, plan, original):
    from scar.codegen.cpu_launch_probe import check

    report = read_artifact(reference)
    _require(report == check(reference['uri']), 'isolated launch proxy changed during cold replay')
    context = _launch_context(scope, plan, original)
    _require(all(report[key] == value for key, value in context.items()),
             'launch proxy context differs from the exact scoped option')
    return dict(evidence=reference, context=context, service_observations_ns=report['measurements'],
        coverage='EMPTY_PROCESS_LIFECYCLE_PROXY', transfer_status='TARGET_LOADER_AND_INITIALIZATION_NOT_REPLAYED',
        charged_to_body_model=False, charged_to_full_entry_model=False)


def _phase_order(reference, ledger_ref, scope, option_id):
    if reference is None:
        return None
    from scar.analysis.cpu_entry_lifecycle_graph import verify_cpu_entry_lifecycle_graph

    graph = verify_cpu_entry_lifecycle_graph(reference)
    _require(reference['schema'] == _LIFECYCLE and graph['schema_version'] == 1
             and graph.get('scope_sha256') == scope, 'lifecycle graph differs from exact entry scope')
    graph_ledger = graph['decision_ledger']
    if graph_ledger is None and graph['capacity_contract'] is not None:
        graph_ledger = read_artifact(graph['capacity_contract'])['ledger']
    _require(graph_ledger == ledger_ref, 'lifecycle graph must bind the exact option ledger')
    matches = [row for row in graph['lanes'] if row['option_id'] == option_id]
    _require(len(matches) <= 1, 'ambiguous lifecycle lane')
    if not matches:
        return dict(graph=reference, lane_status='OPTION_LANE_NOT_RETAINED', happens_before=[])
    lane, = matches
    operations = {row['id']: row['phase'] for row in lane['operations']}
    _require(len(operations) == len(lane['operations']), 'duplicate lifecycle operations')
    edges = [row for row in lane['edges'] if row['kind'] == 'HAPPENS_BEFORE']
    _require(all(row['source'] in operations and row['target'] in operations for row in edges),
             'causal lifecycle order has an invalid endpoint')
    return dict(graph=reference, lane_status='EXACT_OPTION_LANE',
        numerical_team_operations=[row['id'] for row in lane['operations'] if row['kind'] == 'SCHEDULED_NUMERICAL_TEAM'],
        happens_before=[dict(source=row['source'], target=row['target'], source_phase=operations[row['source']],
                             target_phase=operations[row['target']], basis=row['basis'], when=row['when']) for row in edges])


def derive_cpu_entry_forward_cost(ledger_ref, option_id, probes=(), *, lifecycle_graph=None,
                                  schedule_constraints=None):
    """Replay the scoped service prediction and retain the unpriced entry lifecycle."""
    ledger = verify_cpu_entry_decision_ledger(ledger_ref)
    scope = read_artifact(ledger['domain'])['scope']
    matches = [row for row in ledger['options'] if row['option_id'] == option_id]
    _require(len(matches) == 1 and type(probes) in (list, tuple), 'one exact option and a list of probe references required')
    option, = matches
    plan = None
    if not option['original'] and option['legality'] != 'INELIGIBLE':
        proof, = option['legality_evidence']
        fact = _pointer(read_artifact(proof['artifact']), proof['locator'])
        generated = fact.get('generated_option', {})
        if generated.get('implementation_family') == _CATALOG_FAMILY:
            plan = generated['proof']['emission_plan']
    grouped = {reference['schema']: reference for reference in probes}
    _require(len(grouped) == len(probes) and set(grouped) <= {LAUNCH_PROXY, TEAM_PROBE},
             'at most one launch proxy and one independent team repeat may bind an option')
    repeat = grouped.get(TEAM_PROBE)
    constraints = None
    if schedule_constraints is not None:
        from .cpu_entry_schedule_constraints import verify_cpu_entry_schedule_constraints

        constraints = verify_cpu_entry_schedule_constraints(schedule_constraints)
        _require(constraints['scope_sha256'] == ledger['scope_sha256']
                 and constraints['lifecycle_graph'] == lifecycle_graph
                 and constraints['option_id'] in (None, option_id),
                 'resource schedule must bind the same entry and option lifecycle')
    body = _team_body(plan, option, repeat, schedule_constraints=constraints)
    _require(repeat is None or body is not None, 'independent team repeat needs an exact scoped body')
    diagnostics = []
    if LAUNCH_PROXY in grouped:
        diagnostics.append(_launch_probe(grouped[LAUNCH_PROXY], scope, plan, option['original']))
    phase_order = _phase_order(lifecycle_graph, ledger_ref, ledger['scope_sha256'], option_id)
    covered = None
    if body is not None:
        covered = dict(group='DISPATCHED_NUMERIC_BODY', phase_overlap=['worker_lifecycle', 'execution', 'synchronization'],
            graph_operations=[] if phase_order is None else phase_order.get('numerical_team_operations', []),
            composition='CONTENDED_LANE_CRITICAL_PATH_PLUS_ISOLATED_EMPTY_DISPATCH',
            conditional_interval_ns=body['conditional_interval_ns'], sampled_interval_is_physical_bound=False)
    result = dict(schema=SCHEMA, schema_version=1, ledger=ledger_ref, option_id=option_id, probes=list(probes),
        lifecycle_graph=lifecycle_graph, scope_sha256=ledger['scope_sha256'], exact_legality=option['legality'],
        phase_order=phase_order, launch_diagnostics=diagnostics, body_model=body, covered_subgraph=covered,
        status='CONDITIONAL_COVERED_SUBGRAPH_PREDICTION' if covered is not None else 'NO_MATCHED_SERVICE_MODEL',
        prediction_basis='ISOLATED_COMPONENTS_IN_FINITE_PROVIDER_MODEL',
        lifecycle=[dict(phase=phase, coverage='PARTIAL_COMPONENT_MODEL' if covered is not None
                       and phase in covered['phase_overlap'] else 'UNRESOLVED') for phase in LIFECYCLE_PHASES],
        missing_lifecycle_terms=list(LIFECYCLE_PHASES), whole_entry_model_bound_ns=dict(lower=None, upper=None),
        complete_entry_modeled=False, whole_entry_ranked=False, physical_optimality_proven=False,
        execution_default='NO_OP', deployment_authorized=False)
    if schedule_constraints is not None:
        result['schedule_constraints'] = schedule_constraints
    return result


def verify_cpu_entry_forward_cost(report_ref):
    saved = read_artifact(report_ref)
    _require(report_ref['schema'] == SCHEMA and type(saved['schema_version']) is int and saved['schema_version'] == 1,
             'unsupported forward cost report')
    replay = derive_cpu_entry_forward_cost(saved['ledger'], saved['option_id'], saved['probes'],
        lifecycle_graph=saved['lifecycle_graph'], schedule_constraints=saved.get('schedule_constraints'))
    _require(replay == saved, 'forward cost differs at cold replay')
    return replay
