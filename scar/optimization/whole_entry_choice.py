"""Replay finite complete-entry choices over exact execution scopes.

A regional bound never closes an entry's lifecycle. An unpriced alternative
can still win unless an independently sourced lower bound rules it out.
"""

from collections import defaultdict
from dataclasses import asdict, replace

from scar.ir.execution_recipe import read_artifact
from scar.optimization.scope_accounting import _pointer
from scar.optimization.entry_mandatory_bound import KIND as MANDATORY_SUBGRAPH, mandatory_entry_lower_bound
from .resource_solver import semantic_sha256


DOMAIN = 'scar.whole-entry-option-domain'
CHOICE = 'scar.whole-entry-choice'
LIFECYCLE_PHASES = (
    'startup', 'imports', 'guards', 'loading', 'worker_lifecycle', 'allocation',
    'file_io', 'transfers', 'synchronization', 'contention', 'execution',
    'publication', 'fallback', 'teardown',
)
_BOUNDARIES = {'PROCESS_LAUNCH_THROUGH_EXIT', 'WARM_SESSION_ENTRY_THROUGH_EXIT'}
_STATUS = {'LEGAL', 'INELIGIBLE', 'UNRESOLVED'}
_MODEL_BASIS = {'DECLARED_MODEL', 'ANALYTICAL_MODEL'}
_TIE = 'ORIGINAL_THEN_CANONICAL_ID_AT_PROVED_EXACT_EQUALITY'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _duration(value):
    return type(value) is int and value >= 0


def _identity(value):
    return type(value) is str and bool(value)


class _Facts:
    def __init__(self):
        self.documents = {}
        self.references = {}

    def read(self, reference):
        key = semantic_sha256(reference)
        if key not in self.documents:
            self.documents[key] = read_artifact(reference)
            self.references[key] = reference
        return self.documents[key]

    def fact(self, reference):
        _need(type(reference) is dict and set(reference) == {'artifact', 'locator'},
              'original ProviderFactReference required')
        try:
            value = _pointer(self.read(reference['artifact']), reference['locator'])
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError('whole-entry evidence has no retained fact at its pointer') from error
        _need(type(value) is dict, 'whole-entry fact must be a structured record')
        return value

    def recheck(self):
        for reference in self.references.values():
            read_artifact(reference, json_document=False)


def _matching(row, option, scope_sha):
    return row.get('option_id') == option and row.get('scope_sha256') == scope_sha


def _phases(cost, scope):
    lifecycle = cost.get('lifecycle_coverage')
    _need(type(lifecycle) is dict and set(lifecycle) == set(LIFECYCLE_PHASES)
          and all(value in ('ACCOUNTED', 'INAPPLICABLE') for value in lifecycle.values())
          and lifecycle['execution'] == 'ACCOUNTED', 'complete entry lifecycle coverage required')
    if scope['boundary'] == 'PROCESS_LAUNCH_THROUGH_EXIT':
        _need(lifecycle['startup'] == lifecycle['teardown'] == 'ACCOUNTED',
              'whole-process evidence must account for process startup and exit')
    return lifecycle


def _boundary_order(problem, cost, scope, phases):
    anchors = cost.get('boundary_anchors')
    _need(type(anchors) is dict and set(anchors) == {'entry', 'exit'}
          and all(_identity(name) for name in anchors.values()), 'exact schedule must name its entry and exit')
    entry, exit_ = anchors['entry'], anchors['exit']
    always = {op.id for op in problem.operations} | {row.id for row in problem.activities if not row.when}
    _need(entry != exit_ and {entry, exit_} <= always,
          'schedule boundaries must be distinct unconditional activities')
    if scope['boundary'] == 'PROCESS_LAUNCH_THROUGH_EXIT':
        _need(entry in phases['startup'] and exit_ in phases['teardown'],
              'process launch and exit must anchor the startup and teardown phases')

    order = [(parent, op.id) for op in problem.operations for parent in op.predecessors]
    order += [(row.before, row.after) for row in problem.timing_constraints
              if not row.when and (row.before_endpoint, row.after_endpoint) == ('END', 'START')]
    for edge in problem.data_edges:
        if edge.producer is not None:
            order.append((edge.producer, edge.consumer))
        if edge.transfers:
            transfer = 'transfer:' + edge.id
            if edge.producer is not None:
                order.append((edge.producer, transfer))
            order.append((transfer, edge.consumer))

    def reachable(root, edges):
        neighbors = defaultdict(list)
        for before, after in edges:
            neighbors[before].append(after)
        reached, pending = {root}, [root]
        while pending:
            for neighbor in neighbors[pending.pop()]:
                if neighbor not in reached:
                    reached.add(neighbor)
                    if neighbor in always:
                        pending.append(neighbor)
        return reached

    inside = reachable(entry, order) & reachable(exit_, [(after, before) for before, after in order])
    _need(inside == {row for rows in phases.values() for row in rows},
          'every accounted activity must be ordered inside unconditional entry/exit boundaries')


def _scheduled_problem(cost, scope, lifecycle, facts):
    from scar.ir.implementation_plan import problem_from_dict

    problem = problem_from_dict(facts.read(cost['problem']))
    target = scope.get('target')
    _need(type(target) is dict and target == asdict(problem.target),
          'entry time problem must retain the exact original scoped target')
    baseline_only = cost.get('baseline_only', False)
    _need(type(baseline_only) is bool and type(cost.get('seconds', 30)) is int
          and 1 <= cost.get('seconds', 30) <= 3600, 'bounded exact schedule policy required')
    activities = {op.id for op in problem.operations} | {row.id for row in problem.activities}
    transfers = {'transfer:' + edge.id for edge in problem.data_edges if edge.transfers}
    _need(not activities & transfers, 'transfer and lifecycle activity identities must be distinct')
    activities |= transfers
    binding = cost.get('phase_bindings')
    included = cost.get('included_phases', {})
    accounted = {name for name, state in lifecycle.items() if state == 'ACCOUNTED'}
    _need(type(binding) is dict and type(included) is dict and not set(binding) & set(included)
          and set(binding) | set(included) == accounted
          and all(type(rows) is list and rows and all(_identity(row) for row in rows) for rows in binding.values()),
          'full-entry phases must bind their accounted activities')
    for phase, owner in included.items():
        _need(type(owner) is dict and set(owner) == {'phase', 'evidence'} and owner['phase'] in binding,
              'included phase must point to a directly charged activity group')
        evidence = facts.fact(owner['evidence'])
        _need(evidence.get('included_phase') == phase and evidence.get('charged_phase') == owner['phase']
              and _matching(evidence, cost['option_id'], cost['scope_sha256']),
              'included lifecycle cost needs original exact-scope grouping evidence')
    if scope['boundary'] == 'PROCESS_LAUNCH_THROUGH_EXIT':
        _need({'startup', 'teardown'} <= set(binding), 'process boundary activities require direct phase bindings')
    sources = [row for values in binding.values() for row in values]
    _need(set(sources) == activities and len(sources) == len(activities),
          'each original activity and transfer must be covered exactly once')
    _boundary_order(problem, cost, scope, binding)
    return problem


def _point_cost(cost, scope, lifecycle, facts):
    from .constructive_planner import construct_implementations
    from .joint_planner import solve_implementations, verify_execution_plan

    problem = _scheduled_problem(cost, scope, lifecycle, facts)
    baseline_only = cost.get('baseline_only', False)
    timed = [row for op in problem.operations for row in op.options
             if not baseline_only or row.id == op.baseline_option]
    transfer_costs = [route.cost for edge in problem.data_edges for route in edge.transfers]
    activity_costs = [row.cost for row in problem.activities]
    estimates = [row.cost for row in timed] + transfer_costs + activity_costs
    if (any(not _duration(row.estimate_ns) or not row.lower_ns == row.estimate_ns == row.upper_ns
            or row.basis not in _MODEL_BASIS or row.missing_terms for row in estimates)
            or any(row.unresolved_resources for row in timed)):
        # Equal sample endpoints do not establish an exact duration model.
        return 0, None, 'EXACT_SCHEDULE_COST_OR_RESOURCE_FRONTIER', None

    constructed = construct_implementations(problem, baseline_only=baseline_only,
                                             max_seconds=cost.get('seconds', 30))
    constructed_upper = constructed['makespan_ns']
    if constructed_upper is not None:
        verify_execution_plan(problem, constructed)
        if (constructed['domain_complete'] and constructed_upper == constructed['model_lower_bound_ns']):
            constructed.pop('elapsed_seconds', None)
            return constructed_upper, constructed_upper, 'EXACT_POINT_SCHEDULE_IN_DECLARED_MODEL', dict(
                problem=cost['problem'], proof='CONSTRUCTIVE_UPPER_MEETS_ANALYTIC_LOWER_BOUND',
                constructive_upper_ns=constructed_upper, plan=constructed)
    try:
        plan = solve_implementations(problem, baseline_only=baseline_only,
                                     max_seconds=cost.get('seconds', 30), workers=1)
    except RuntimeError as error:
        if not isinstance(error.__cause__, ImportError):
            raise
        return 0, None, 'OPTIONAL_EXACT_SOLVER_UNAVAILABLE', None
    if (plan['status'] != 'OPTIMAL_WITHIN_MODEL' or not plan['domain_complete']
            or not plan['model_optimality_proven'] or plan['relative_gap'] != 0
            or plan['makespan_ns'] != plan['model_lower_bound_ns']):
        return 0, None, 'EXACT_SCHEDULE_OPTIMALITY_UNRESOLVED', None
    best = plan['makespan_ns']
    _need(_duration(best) and (constructed_upper is None or best <= constructed_upper),
          'exact model optimum exceeds a verified constructive incumbent')
    verify_execution_plan(problem, plan)
    plan.pop('elapsed_seconds', None)
    return best, best, 'EXACT_POINT_SCHEDULE_IN_DECLARED_MODEL', dict(problem=cost['problem'],
        proof='CP_SAT_CLOSED_DOMAIN_ZERO_GAP', constructive_upper_ns=constructed_upper, plan=plan)


def _endpoint_problem(problem, endpoint):
    def pin(cost):
        duration = getattr(cost, endpoint)
        return replace(cost, estimate_ns=duration, lower_ns=duration, upper_ns=duration)

    operations = tuple(replace(op, options=(replace(op.options[0], cost=pin(op.options[0].cost)),))
                       for op in problem.operations)
    activities = tuple(replace(row, cost=pin(row.cost)) for row in problem.activities)
    data_edges = tuple(replace(edge, transfers=tuple(replace(row, cost=pin(row.cost))
                       for row in edge.transfers)) for edge in problem.data_edges)
    return replace(problem, operations=operations, activities=activities, data_edges=data_edges)


def _monotone_cost(cost, scope, lifecycle, facts):
    from .constructive_planner import construct_implementations
    from .joint_planner import solve_implementations, verify_execution_plan

    problem = _scheduled_problem(cost, scope, lifecycle, facts)
    _need(all(len(op.options) == 1 for op in problem.operations)
          and all(not activity.when and activity.duration_kind == 'FIXED' for activity in problem.activities)
          and all(not relation.when and relation.maximum_ns is None
                  and (relation.before_endpoint, relation.after_endpoint) == ('END', 'START')
                  for relation in problem.timing_constraints),
          'interval replay requires one fixed implementation and unconditional end-to-start order')
    modes = {op.id: op.options[0] for op in problem.operations}
    for edge in problem.data_edges:
        source = edge.external_location if edge.producer is None else modes[edge.producer].output_locations[edge.value]
        destination = modes[edge.consumer].input_locations[edge.value]
        _need((source == destination and not edge.transfers) or (len(edge.transfers) == 1
              and (edge.transfers[0].source_memory, edge.transfers[0].destination_memory) == (source, destination)),
              'interval replay requires an exact materialization route for every edge')
    estimates = [mode.cost for mode in modes.values()] + [row.cost for row in problem.activities]
    estimates.extend(row.cost for edge in problem.data_edges for row in edge.transfers)
    if (any(row.basis not in _MODEL_BASIS or row.missing_terms or not row.source_refs
            or not _duration(row.lower_ns) or not _duration(row.upper_ns) for row in estimates)
            or any(mode.unresolved_resources for mode in modes.values())):
        return 0, None, 'BOUNDED_SCHEDULE_COST_OR_RESOURCE_FRONTIER', None
    for row in estimates:
        for reference in row.source_refs:
            facts.fact(reference)

    baseline = cost.get('baseline_only', False)
    budget = cost.get('seconds', 30)
    lower_problem = _endpoint_problem(problem, 'lower_ns')
    lower = construct_implementations(lower_problem, baseline_only=baseline, max_seconds=budget)
    upper_problem = _endpoint_problem(problem, 'upper_ns')
    upper = construct_implementations(upper_problem, baseline_only=baseline, max_seconds=budget)
    if upper['makespan_ns'] is None:
        try:
            upper = solve_implementations(upper_problem, baseline_only=baseline, max_seconds=budget, workers=1)
        except RuntimeError as error:
            if not isinstance(error.__cause__, ImportError):
                raise
    bounds = [lower.get(key) for key in ('model_lower_bound_ns', 'fixed_selection_lower_bound_ns')]
    lower_bound = max((bound for bound in bounds if bound is not None), default=0)
    if upper['makespan_ns'] is None or not upper['domain_complete']:
        return lower_bound, None, 'BOUNDED_SCHEDULE_FEASIBLE_UPPER_UNRESOLVED', None
    # Keeping the upper plan's starts and shortening individual activities
    # preserves end-to-start order and only shrinks the occupied intervals.
    verify_execution_plan(upper_problem, upper)
    upper.pop('elapsed_seconds', None)
    return lower_bound, upper['makespan_ns'], 'FIXED_SCHEDULE_MODEL_INTERVAL', dict(
        problem=cost['problem'], proof='LOWER_ENDPOINT_RELAXATION_AND_FEASIBLE_UPPER_SCHEDULE',
        relaxed_lower_bounds_ns=bounds, plan=upper)


def _cost(reference, option, scope, scope_sha, facts, *, baseline):
    if reference is None:
        return 0, None, 'COMPLETE_ENTRY_COST_UNKNOWN', None
    cost = facts.fact(reference)
    _need(_matching(cost, option, scope_sha), 'cost fact differs from original option or exact entry scope')
    kind = cost.get('kind')
    if kind == 'LOWER_BOUND_ONLY':
        lower = cost.get('lower_ns')
        _need(_duration(lower) and cost.get('basis') in _MODEL_BASIS
              and _identity(cost.get('required_work')), 'proven mandatory-work lower bound required')
        return lower, None, 'RETAINED_MANDATORY_WORK_LOWER_BOUND', None
    if kind == MANDATORY_SUBGRAPH:
        lower, proof = mandatory_entry_lower_bound(cost, scope, facts)
        return lower, None, 'MANDATORY_SUBGRAPH_MODEL_LOWER_BOUND', proof
    lifecycle = _phases(cost, scope)
    if kind in ('OPTIMIZED_POINT_SCHEDULE', 'BOUNDED_MONOTONE_SCHEDULE'):
        _need(cost.get('baseline_only', False) is baseline,
              'only original implementations may populate the original execution option')
        replay = _point_cost if kind == 'OPTIMIZED_POINT_SCHEDULE' else _monotone_cost
        return replay(cost, scope, lifecycle, facts)
    interval = cost.get('interval_ns')
    basis = cost.get('basis')
    _need(kind == 'COMPLETE_INTERVAL' and (basis in _MODEL_BASIS or basis == 'EMPIRICAL_ENVELOPE')
          and type(interval) is dict and set(interval) == {'lower', 'upper'}
          and _duration(interval['lower']) and _duration(interval['upper'])
          and interval['lower'] <= interval['upper'], 'sourced complete-entry interval required')
    if basis == 'EMPIRICAL_ENVELOPE':
        return 0, None, 'EMPIRICAL_ENVELOPE_NOT_MODEL_BOUND', None
    _need(type(cost.get('model_gap_ns')) is int and cost['model_gap_ns'] == 0,
          'interval must describe a fixed implementation or closed model')
    return interval['lower'], interval['upper'], basis, None


def solve_whole_entry_choice(domain, *, expected_option_ids):
    """Compare all independently enumerated alternatives in one declared scope."""
    _need(type(domain) is dict and set(domain) == {'schema', 'schema_version', 'scope', 'baseline_id',
           'options', 'coverage_frontier'} and domain['schema'] == DOMAIN
          and type(domain['schema_version']) is int and domain['schema_version'] == 1,
          'versioned whole-entry finite domain required')
    scope = domain['scope']
    _need(type(scope) is dict and scope.get('boundary') in _BOUNDARIES,
          'exact whole-entry lifecycle context required')
    _need(type(domain['coverage_frontier']) is list, 'explicit generator coverage frontier required')
    declared = domain['options']
    _need(isinstance(expected_option_ids, (list, tuple)) and expected_option_ids
          and all(_identity(name) for name in expected_option_ids)
          and len(set(expected_option_ids)) == len(expected_option_ids),
          'independently enumerated finite options required')
    _need(type(declared) is list and len(declared) == len(expected_option_ids)
          and all(type(row) is dict and set(row) == {'id', 'legal_status', 'legality_evidence', 'cost'}
                  and _identity(row['id']) for row in declared)
          and {row['id'] for row in declared} == set(expected_option_ids),
          'whole-entry domain differs from the independent enumeration')
    original = domain['baseline_id']
    _need(_identity(original) and original in expected_option_ids, 'original execution must be retained')
    facts = _Facts()
    if type(scope.get('target')) is dict:
        facts.read(scope['target'])
    scope_sha = semantic_sha256(scope)
    rows = []
    for row in sorted(declared, key=lambda row: row['id']):
        option, status, proofs = row['id'], row['legal_status'], row['legality_evidence']
        baseline = option == original
        _need(status in _STATUS and (not baseline or status == 'LEGAL') and type(proofs) is list,
              'legality must be explicit and original execution remains legal')
        legal = [facts.fact(ref) for ref in proofs]
        _need((baseline or status == 'UNRESOLVED' or bool(legal)) and all(
            _matching(fact, option, scope_sha) and fact.get('legal_status') == status for fact in legal),
            'nonoriginal concluded legality requires original exact-scope proof')
        _need(status != 'INELIGIBLE' or row['cost'] is None,
              'ineligible options require a legality reason, with no ranked cost')
        if status == 'INELIGIBLE':
            lower, upper, basis, schedule = None, None, 'PROVED_INELIGIBLE', None
        else:
            lower, upper, basis, schedule = _cost(row['cost'], option, scope, scope_sha, facts,
                                                  baseline=baseline)
        rows.append(dict(option_id=option, legal_status=status, interval_ns=dict(lower=lower, upper=upper),
            cost_basis=basis, source_cost=row['cost'], legality_evidence=proofs, schedule_certificate=schedule))

    ranked = [row for row in rows if row['legal_status'] == 'LEGAL' and row['interval_ns']['upper'] is not None]
    incumbent = min(ranked, key=lambda row: (row['interval_ns']['upper'], row['option_id'] != original,
                                             row['option_id'])) if ranked else None
    excluded, frontier, tied = [], [], []
    for row in rows:
        if row['legal_status'] == 'INELIGIBLE' or row is incumbent:
            continue
        option = row['option_id']
        low, high = row['interval_ns']['lower'], row['interval_ns']['upper']
        if incumbent is not None:
            bound = incumbent['interval_ns']['upper']
            if low > bound:
                excluded.append(dict(option_id=option, lower_ns=low, incumbent_id=incumbent['option_id'],
                                     incumbent_upper_ns=bound, rule='LOWER_STRICTLY_ABOVE_FEASIBLE_UPPER'))
                continue
            if (row['legal_status'] == 'LEGAL' and low == high == bound
                    and incumbent['interval_ns']['lower'] == bound):
                tied.append(option)
                continue
        frontier.append(dict(option_id=option, reason='NO_FEASIBLE_UPPER' if incumbent is None
                             else 'COMPETITOR_COULD_MATCH_OR_IMPROVE_INCUMBENT'))
    optimal = incumbent is not None and not frontier and not domain['coverage_frontier']
    selected = incumbent['option_id'] if optimal else None
    facts.recheck()
    return dict(schema=CHOICE, schema_version=1, domain_sha256=semantic_sha256(domain),
        scope_sha256=scope_sha, status='MODEL_OPTIMAL' if optimal else 'UNRESOLVED',
        selected_option_id=selected, execution_default='NO_OP',
        decision='CANDIDATE_FOR_VALIDATION' if optimal and selected != original else 'NO_OP',
        objective='MINIMIZE_ENTRY_COMPLETION_TIME', model_optimality_proven=optimal,
        selection_gap_ns=0 if optimal else None, selection_gap_basis='ROBUST_RANKING',
        domain_coverage_closed=not domain['coverage_frontier'],
        coverage_frontier=domain['coverage_frontier'], cost_frontier=frontier,
        known_incumbent_id=None if incumbent is None else incumbent['option_id'],
        dominated=excluded, proved_exact_tie_ids=tied, tie_policy=_TIE, alternatives=rows,
        time_unit='NANOSECONDS', all_options_retained=True, unknown_duration_imputed=False,
        physical_optimality_proven=False, deployment_authorized=False)


def verify_whole_entry_choice(domain, report, *, expected_option_ids):
    replay = solve_whole_entry_choice(domain, expected_option_ids=expected_option_ids)
    _need(report == replay, 'whole-entry choice differs from exact independent replay')
    return replay
