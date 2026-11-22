"""Bounded lane placement and order planning over qualified CPU Tensor calls."""

from math import floor

from scar.optimization.aten_live_values import _validate, ready_lifetime_bound
from scar.optimization.aten_live_values import _bounded_sum, _bytes


_LANE_COUNT = 3
_MAX_OPERATIONS = 24
_MAX_MODELED_NS = 1 << 50
_DETERMINISTIC_WORK = 0.25
_SCOPE = 'FIXED_CPU_TEAM_ISOLATED_SERVICE_PLUS_DISPATCH'


def ordered_ready_intervals(dependencies, lanes, costs, operation_order):
    """Retiming for a topological order, with intervals indexed by source ID."""
    intervals = [None] * len(operation_order)
    previous = [None] * _LANE_COUNT
    for op in operation_order:
        lane = lanes[op]
        prior = previous[lane]
        parents = dependencies[op] if prior is None else (*dependencies[op], prior)
        start = max((intervals[parent][1] for parent in parents), default=0)
        intervals[op] = (start, start + costs[op], prior)
        previous[lane] = op
    return intervals


def _duration(value):
    return type(value) is int and 0 < value < 1 << 63


def _service_upper_bound(service, handoffs, count):
    if (type(service) not in (list, tuple) or len(service) != _LANE_COUNT
            or any(type(row) not in (list, tuple) or len(row) != count
                   or not all(map(_duration, row)) for row in service)):
        raise ValueError('joint ready scheduling requires per-operation durations for every CPU lane')
    if type(handoffs) is not dict or handoffs.keys() != {'2', '3'} or not all(map(_duration, handoffs.values())):
        raise ValueError('joint ready scheduling requires both dispatch costs')
    return sum(map(max, zip(*service, strict=True)))


def _add_storage(model, dependencies, sizes, exports, starts, ends, finish, horizon, capacity):
    readers = [[] for _ in sizes]
    for consumer, parents in enumerate(dependencies):
        for parent in parents:
            readers[parent].append(ends[consumer])
    leases, demands = [], []
    for op, size in enumerate(sizes):
        if size == 0:
            continue
        if op in exports:
            release = finish
        elif readers[op]:
            release = model.new_int_var(0, horizon, f'release:{op}')
            model.add_max_equality(release, readers[op])
        else:
            release = ends[op]
        duration = model.new_int_var(1, horizon, f'lease-duration:{op}')
        leases.append(model.new_interval_var(starts[op], duration, release, f'lease:{op}'))
        demands.append(size)
    if leases:
        model.add_cumulative(leases, demands, capacity)


def solve_ready_joint(dependencies, service, handoffs, output_indices, result_bytes, input_bytes, budget):
    """Minimize integer makespan plus dispatch over a fixed three-CPU team.

    Timed storage is a relaxation. Before accepting a plan, require that the
    independent lifetime proof covers every completion prefix of its lane
    queues, including those possible when live operation times differ.
    """
    count = len(dependencies)
    _validate(dependencies, [0] * count, output_indices, result_bytes, input_bytes)
    if not _bytes(budget) or budget == 0:
        raise ValueError('joint ready scheduling requires a positive bounded storage budget')
    horizon = _service_upper_bound(service, handoffs, count)
    full_retention = _bounded_sum((input_bytes, *result_bytes))
    report = dict(status='UNRESOLVED', reason=None, scope=_SCOPE, lanes=None, operation_order=None,
                  estimate_ns=None, lower_bound_ns=None, lifetime_proof=None, launch_width=None,
                  max_operations=_MAX_OPERATIONS, max_deterministic_time=_DETERMINISTIC_WORK,
                  timed_storage_is_relaxation=True, physical_optimality_proven=False)
    if count < 2 or count > _MAX_OPERATIONS:
        return dict(report, reason='READY_JOINT_OPERATION_COUNT_OUTSIDE_DOMAIN')
    if horizon + max(handoffs.values()) >= _MAX_MODELED_NS:
        return dict(report, reason='READY_JOINT_TIME_DOMAIN_TOO_WIDE')
    if input_bytes + max(result_bytes) > budget:
        return dict(report, status='INFEASIBLE_WITHIN_MODEL', reason='READY_JOINT_SINGLE_RESULT_EXCEEDS_BUDGET')

    try:
        from ortools.sat.python import cp_model
    except ImportError:
        return dict(report, reason='READY_JOINT_REQUIRES_PLANNING_EXTRA')

    model = cp_model.CpModel()
    starts = [model.new_int_var(0, horizon, f'start:{op}') for op in range(count)]
    ends = [model.new_int_var(0, horizon, f'end:{op}') for op in range(count)]
    on_lane = [[model.new_bool_var(f'lane:{lane}:{op}') for op in range(count)] for lane in range(_LANE_COUNT)]
    for op, parents in enumerate(dependencies):
        model.add_exactly_one(row[op] for row in on_lane)
        for parent in parents:
            model.add(ends[parent] <= starts[op])

    used = []
    for lane, durations in enumerate(service):
        model.add_no_overlap([model.new_optional_interval_var(starts[op], durations[op], ends[op],
                                on_lane[lane][op], f'task:{lane}:{op}') for op in range(count)])
        active = model.new_bool_var(f'active:{lane}')
        model.add_max_equality(active, on_lane[lane])
        used.append(active)
    model.add(sum(used) >= 2)

    finish = model.new_int_var(0, horizon, 'finish')
    model.add_max_equality(finish, ends)
    if full_retention > budget:
        _add_storage(model, dependencies, result_bytes, set(output_indices), starts, ends, finish,
                     horizon, budget - input_bytes)
    model.minimize(finish + handoffs['2'] + (handoffs['3'] - handoffs['2']) * used[2])

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_deterministic_time = _DETERMINISTIC_WORK
    outcome = solver.solve(model)
    report['solver_status'] = solver.status_name(outcome)
    if outcome == cp_model.INFEASIBLE:
        return dict(report, status='INFEASIBLE_WITHIN_MODEL', reason='NO_TIMED_READY_PLAN_WITHIN_BUDGET')
    if outcome == cp_model.MODEL_INVALID:
        raise ValueError('joint ready model is invalid: ' + model.validate())
    lower = max(0, floor(solver.best_objective_bound))
    report['lower_bound_ns'] = lower
    if outcome not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return dict(report, reason='READY_JOINT_SEARCH_LIMIT')

    lanes = [next(lane for lane, row in enumerate(on_lane) if solver.value(row[op])) for op in range(count)]
    order = sorted(range(count), key=lambda op: (solver.value(starts[op]), op))
    costs = [service[lane][op] for op, lane in enumerate(lanes)]
    intervals = ordered_ready_intervals(dependencies, lanes, costs, order)
    order.sort(key=lambda op: (intervals[op][0], op))
    release = max(end for _, end, _ in intervals)
    if release == sum(costs):
        return dict(report, reason='READY_JOINT_RETIMING_HAS_NO_OVERLAP')
    proof = ready_lifetime_bound(dependencies, lanes, output_indices, result_bytes, input_bytes,
                                 operation_order=order)
    if proof['peak_tensor_storage_bytes'] > budget:
        return dict(report, reason='READY_JOINT_ASYNCHRONOUS_STORAGE_BUDGET_UNRESOLVED', lifetime_proof=proof)

    width = max(lanes) + 1
    estimate = release + handoffs[str(width)]
    if estimate < lower or estimate > round(solver.objective_value):
        raise ValueError('dependency-ready timing disagrees with its relaxed solver bound')
    status = 'OPTIMAL_WITHIN_MODEL' if estimate == lower else 'FEASIBLE_WITHIN_MODEL'
    return dict(report, status=status, lanes=lanes, operation_order=order, estimate_ns=estimate,
                lifetime_proof=proof, launch_width=width)
