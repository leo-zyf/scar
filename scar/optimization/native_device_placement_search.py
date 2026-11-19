"""Compare explicitly materialized CPU/CUDA placements without running them."""

from sys import maxsize

from scar.ir.implementation_plan import PLAN_SCHEMA, digest, problem_from_dict
from scar.optimization.joint_planner import verify_execution_plan


SCHEMA = 'scar.native-device-placement-search'
DEFAULT_PLACEMENT_BUDGET = 128
_DEVICE_OPTIONS = (('CPU',), ('CPU', 'CUDA'))
_FEASIBLE = ('FEASIBLE_WITHIN_MODEL', 'FEASIBLE_PARTIAL_DOMAIN')
_COST_BASES = ('DECLARED_MODEL', 'EMPIRICAL_MODEL')


class PlacementCapacityExceeded(ValueError):
    """A fixed placement exceeds a declared resource capacity before scheduling."""


def _choice_table(choices):
    if not isinstance(choices, dict) or not choices:
        raise ValueError('placement search requires an ordered typed operation domain')
    table = []
    for name, modes in choices.items():
        if (type(name) is not str or not name or type(modes) not in (list, tuple)
                or any(type(mode) is not str for mode in modes)
                or tuple(modes) not in _DEVICE_OPTIONS):
            raise ValueError('each operation needs a CPU incumbent and optionally a qualified CUDA choice')
        table.append((name, tuple(modes)))
    return table, sum(len(modes) == 2 for _, modes in table)


def _placement_for_ordinal(table, switchable, ordinal):
    bits = iter(f'{ordinal:0{switchable}b}')
    return {name: modes[1] if len(modes) == 2 and next(bits) == '1' else modes[0]
            for name, modes in table}


def placement_assignment(choices, ordinal: int) -> dict[str, str]:
    """Resolve an assignment in CPU-first order, with later choices varying faster."""
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError('placement ordinal must be a nonnegative integer')
    table, switchable = _choice_table(choices)
    if ordinal.bit_length() > switchable:
        raise ValueError('placement ordinal is outside the declared placement domain')
    return _placement_for_ordinal(table, switchable, ordinal)


def _cost_frontiers(problem, cost_basis):
    frontiers = []
    empirical = cost_basis == 'EMPIRICAL_MODEL'

    def missing(cost):
        return (cost.basis != cost_basis or cost.estimate_ns is None or cost.missing_terms
                or empirical and (cost.estimate_ns == 0 or not cost.source_refs))

    def add(cost, **scope):
        row = dict(**scope, reason='MISSING_MATCHED_COST', cost_basis=cost.basis)
        if empirical:
            row.update(context=cost.context, missing_terms=list(cost.missing_terms))
        frontiers.append(row)

    for operation in problem.operations:
        mode, = operation.options
        cost = mode.cost
        if (cost.estimate_ns == 0 and cost.basis == 'DECLARED_MODEL' and cost.context == 'LOGICAL_ENDPOINT'
                and not mode.demands and not mode.workspace_bytes
                and (not empirical or not mode.output_locations and not mode.unresolved_resources)):
            continue
        if missing(cost):
            scope = dict(operation=operation.id, lowering=mode.lowering) if empirical else dict(operation=operation.id)
            add(cost, **scope)
    for activity in problem.activities:
        if missing(activity.cost):
            add(activity.cost, activity=activity.id)
    return frontiers


def _evaluate(placement, built, domain, cost_basis):
    problem_wire, schedule, bindings = built
    problem = problem_from_dict(problem_wire)
    if (any(len(op.options) != 1 for op in problem.operations)
            or placement.keys() - {op.id for op in problem.operations}
            or any(edge.transfers for edge in problem.data_edges)):
        raise ValueError('placement callback must build one fixed implementation with explicit shared materializations')
    identity = (problem.target, problem.source_refs, problem.normal_completion_only, problem.objective)
    if domain is not None and identity != domain:
        raise ValueError('placement plans must describe the same source, target and observation contract')
    frontiers = _cost_frontiers(problem, cost_basis)
    if schedule is None:
        if cost_basis != 'EMPIRICAL_MODEL' or not frontiers:
            raise ValueError('only an unpriced empirical placement may defer scheduling')
        row = dict(problem_sha256=digest(problem), schedule_status='UNRESOLVED', makespan_ns=None,
                   status='UNRESOLVED', frontier=frontiers)
        return row, None, identity
    if (schedule.get('schema') != PLAN_SCHEMA or schedule.get('problem_sha256') != digest(problem)
            or schedule.get('execution_authorized') is not False
            or schedule.get('deployment_authorized') is not False
            or schedule.get('target_program_executed') is not False):
        raise ValueError('placement score requires an offline schedule locked to its physical problem')
    status = schedule['status']
    if status not in (*_FEASIBLE, 'UNRESOLVED'):
        raise ValueError('unexpected status from the fixed-placement constructive scheduler')
    proof = verify_execution_plan(problem, schedule) if status in _FEASIBLE else None
    frontiers.extend(schedule['frontiers'])
    frontiers.extend(schedule.get('construction_frontier', ()))
    row = dict(problem_sha256=schedule['problem_sha256'], schedule_status=status, makespan_ns=None)
    if proof is None or frontiers:
        row.update(status='UNRESOLVED', frontier=frontiers or [{'reason': 'NO_VERIFIED_CONSTRUCTIVE_SCHEDULE'}])
        return row, None, identity
    stable_schedule = {key: value for key, value in schedule.items() if key != 'elapsed_seconds'}
    row.update(status='VERIFIED_SCHEDULE', makespan_ns=schedule['makespan_ns'])
    record = dict(placement=placement, problem=problem_wire, schedule=stable_schedule, bindings=bindings)
    return row, record, identity


def search_native_device_placements(choices, build, *, max_placements=DEFAULT_PLACEMENT_BUDGET,
                                    cost_basis='DECLARED_MODEL'):
    """Return the audit summary and one selected fixed-placement record.

    The callback qualifies a placement and constructs its shared-material
    problem, schedule and bindings. It owns evidence replay for matched
    empirical costs and can defer an unpriced empirical schedule. Placement
    ranking compares constructed schedules, so an exhaustive placement domain
    still makes no claim about schedule optimality.
    """
    table, switchable = _choice_table(choices)
    if type(max_placements) is not int or not 1 <= max_placements <= maxsize or cost_basis not in _COST_BASES:
        raise ValueError('placement search requires a positive bound and one explicit cost basis')
    total = 1 << switchable
    checked, unresolved = [], []
    baseline = best = None
    best_key = None
    domain = None
    for ordinal in range(min(total, max_placements)):
        placement = _placement_for_ordinal(table, switchable, ordinal)
        try:
            built = build(placement)
        except PlacementCapacityExceeded as error:
            checked.append(dict(ordinal=ordinal, status='PLACEMENT_CAPACITY_EXCEEDED', detail=str(error)))
            continue
        row, record, domain = _evaluate(placement, built, domain, cost_basis)
        checked.append(dict(ordinal=ordinal, **row))
        if record is None:
            unresolved.append(dict(ordinal=ordinal, reason='PLACEMENT_UNRESOLVED'))
            continue
        if ordinal == 0:
            baseline = record
        score = (row['makespan_ns'], ordinal.bit_count(), ordinal)
        if best_key is None or score < best_key:
            best, best_key = record, score
    enumeration_complete = len(checked) == total
    if not enumeration_complete:
        unresolved.append(dict(reason='PLACEMENT_BUDGET_EXHAUSTED', next_ordinal=len(checked)))
    selected = baseline
    if baseline is None:
        reason = 'CPU_INCUMBENT_UNRESOLVED'
    elif unresolved:
        reason = 'PLACEMENT_DOMAIN_UNRESOLVED'
    elif best is baseline:
        reason = 'NO_STRICT_MODELED_GAIN'
    else:
        selected, reason = best, 'STRICT_MODELED_GAIN'
    known_best = None if best is None else dict(ordinal=best_key[2], placement=best['placement'],
        makespan_ns=best_key[0], problem_sha256=best['schedule']['problem_sha256'])
    summary = dict(schema=SCHEMA, schema_version=1, cost_basis=cost_basis,
        scope='CONSTRUCTED_SCHEDULES_WITH_EXPLICIT_SHARED_MATERIALS',
        search_policy=dict(order='CPU_FIRST_LEXICOGRAPHIC_IN_SOURCE_ORDER', max_placements=max_placements,
                           tie_break='MAKESPAN_THEN_FEWER_CUDA_OPERATIONS_THEN_ORDINAL'),
        choices=[dict(operation=name, devices=list(modes)) for name, modes in table],
        binary_device_choices=switchable, declared_placements=total if switchable < 63 else None,
        evaluated_placements=len(checked), enumeration_complete=enumeration_complete,
        placement_domain_complete=not unresolved, baseline=checked[0], evaluated=checked, frontiers=unresolved,
        best_known=known_best, selected_placement=None if selected is None else selected['placement'],
        decision=reason, default_backend='CPU', model_optimality_proven=False, physical_optimality_proven=False,
        target_program_executed=False, execution_authorized=False, deployment_authorized=False)
    return summary, selected
