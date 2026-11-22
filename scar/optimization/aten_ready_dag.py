"""Schedule and price ready CPU lanes, including lane-local input materialization."""

from dataclasses import dataclass
from fractions import Fraction
from math import floor, prod
from statistics import median_low

from scar.optimization.aten_boundary_trials import ATEN_GAIN_POLICY, _passes_gain
from scar.optimization.aten_live_values import _bounded_sum, _bytes, _operation_order, _validate, ready_lifetime_bound
from scar.optimization.aten_numa_locality import (
    _MAX_PAGES_PER_STORAGE, _bounded_duration, _page_sample, _remote_fraction, _validate_samples,
    model_ready_numa_locality,
)
from scar.optimization.aten_team import READY_SERVICE_SCOPE, _measurements, _storage, _team_groups, data_dependencies

_LANE_DOMAINS = ((0, 1), (0, 2), (0, 1, 2))
_LANE_COUNT = 3
_MAX_OPERATIONS = 24
_MAX_MODELED_NS = 1 << 50
_DETERMINISTIC_WORK = 0.25
_SCOPE = 'FIXED_CPU_TEAM_ISOLATED_SERVICE_PLUS_DISPATCH'

_CALIBRATION_FIELDS = {
    'cpu_ids', 'lanes', 'input_index', 'lane', 'warmup_pairs', 'direct_samples_ns',
    'replicated_samples_ns', 'copy_samples_ns', 'replica_pages', 'all_runs_validated',
}
_CALIBRATION_RENAMES = {'replicated_samples_ns': 'retained_samples_ns', 'copy_samples_ns': 'check_samples_ns'}
_RETAINED_FIELDS = {
    *(_CALIBRATION_RENAMES.get(field, field) for field in _CALIBRATION_FIELDS),
    'cold_samples_ns',
}
_GAIN_PERCENT = 5


@dataclass(frozen=True, slots=True)
class _ReadyOrder:
    operations: tuple[int, ...]
    predecessors: tuple[tuple[int, ...], ...]
    lane_predecessors: tuple[int | None, ...]
    lane_heads: tuple[int | None, ...]

    def intervals(self, costs, *, prologue_lane=None, prologue_ns=0):
        # A copy or retained-content check must finish before the first operation
        # on its lane. Every other head can proceed on its original DATA edges.
        delayed_head = None if prologue_lane is None else self.lane_heads[prologue_lane]
        count = len(self.operations)
        ends = [0] * count
        intervals = [None] * count
        for operation in self.operations:
            start = max((ends[parent] for parent in self.predecessors[operation]), default=0)
            if operation == delayed_head:
                start = max(start, prologue_ns)
            end = start + costs[operation]
            ends[operation] = end
            intervals[operation] = (start, end, self.lane_predecessors[operation])
        return intervals


def _ready_order(dependencies, lanes, operation_order=None):
    """Bind the checked Tensor graph and CPU lane order once for all samples."""
    order = _operation_order(dependencies, operation_order)
    heads = [None] * (max(lanes) + 1)
    previous = heads.copy()
    lane_predecessors = [None] * len(order)
    predecessors = [()] * len(order)
    for operation in order:
        lane = lanes[operation]
        prior = previous[lane]
        parents = tuple(dependencies[operation])
        if prior is None:
            heads[lane] = operation
        else:
            parents = (*parents, prior)
        predecessors[operation] = parents
        lane_predecessors[operation] = prior
        previous[lane] = operation
    return _ReadyOrder(tuple(order), tuple(predecessors), tuple(lane_predecessors), tuple(heads))


def _assign_lanes(dependencies, service, domains):
    ends = []
    available = [0] * len(service)
    lanes = []
    for index, parents in enumerate(dependencies):
        ready = max((ends[parent] for parent in parents), default=0)
        end, lane = min((max(ready, available[lane]) + service[lane][index], lane)
                        for lane in domains if service[lane][index] is not None)
        ends.append(end)
        lanes.append(lane)
        available[lane] = end
    return lanes


def _assigned_costs(rows, lanes):
    return [rows[lane][index] for index, lane in enumerate(lanes)]


def _finish(intervals):
    return max(end for _, end, _ in intervals)


def _overlap(intervals):
    previous_end = 0
    for start, end, _ in sorted(intervals, key=lambda row: row[0]):
        if start < previous_end:
            return True
        previous_end = end
    return False


def _prediction(point, samples):
    return dict(estimate_ns=point, sample_min_ns=min(point, *samples), sample_max_ns=max(point, *samples))


def _column_medians(samples):
    return [median_low(column) for column in zip(*samples, strict=True)]


def _timeline(operations, lanes, cpus, dependencies, intervals):
    return [dict(index=index, operation=operation['id'], lane=lane, cpu_id=cpus[lane],
                 start_ns=start, end_ns=end, data_predecessors=dependencies[index], lane_predecessor=prior)
            for index, (operation, lane, (start, end, prior)) in enumerate(
                zip(operations, lanes, intervals, strict=True))]


def plan_ready_dag(recipe, schedule, calibration, warmup, *, tensor_budget_bytes, ready_dag_abi=2, joint_order=False,
                   gain_policy=ATEN_GAIN_POLICY):
    """Nominate and price a deterministic lane schedule on the measured placement.

    Callers must replay the exact concurrency declaration for every operator.
    Each lane is traversed in original source order, with waits on original
    DATA predecessors only. Across NUMA nodes, price the exact nominee using
    a measured provider replay on its lanes and observed storage pages. Cost
    estimates still require matched concurrent validation.
    """
    dependencies = data_dependencies(recipe)
    operations = recipe['operations']
    windows = _team_groups(recipe, schedule)
    full_graph = schedule['schema_version'] == 2
    if full_graph and ready_dag_abi != 2:
        raise ValueError('whole-graph readiness requires the Tensor lifetime ABI')
    cpus = schedule['cpus']
    nodes = [lane['numa_node'] for lane in schedule['lanes']]
    if len(cpus) != 3 or len(set(cpus)) != 3 or len(nodes) != 3:
        raise ValueError('ready DAG requires a verified three-lane CPU placement')
    report = dict(status='KEEP_ORIGINAL', reason=None, data_dependencies=dependencies, cpu_ids=cpus,
                  lanes=[], timeline=[], predictions={}, tensor_storage=None, launch_width=None,
                  dispatch_join_ns=None, bounds='ISOLATED_CALIBRATION_SPAN_NOT_PHYSICAL_BOUNDS',
                  basis='SOURCE_ORDER_EARLIEST_FINISH_WITH_MEASURED_LANE_SERVICES',
                  contention_modelled=False, per_edge_notification_cost_modelled=False,
                  source_replacement_authorized=False)
    if gain_policy is not None:
        report['gain_policy'] = dict(gain_policy)
    cross_numa = len(set(nodes)) > 1
    if joint_order and (cross_numa or not full_graph):
        raise ValueError('joint ready ordering requires the same-NUMA full-graph contract')
    if None in nodes or (cross_numa and not full_graph):
        return dict(report, reason='READY_DAG_REQUIRES_SAME_NUMA_NODE')
    if tensor_budget_bytes is None:
        return dict(report, reason='READY_DAG_REQUIRES_TENSOR_BUDGET')
    if not _bytes(tensor_budget_bytes) or tensor_budget_bytes == 0:
        raise ValueError('ready DAG Tensor budget must be a positive bounded byte count')

    service, handoffs, lane_samples, handoff_samples = _measurements(
        calibration, warmup, len(operations), windows, full_graph=full_graph)
    if full_graph:
        report['service_scope'] = READY_SERVICE_SCOPE
    original_storage, _ = _storage(calibration, recipe, {}, None)
    input_bytes = original_storage['persistent_input_bytes']
    result_bytes = calibration['result_storage_bytes']
    peak = _bounded_sum((input_bytes, *result_bytes))
    report['tensor_storage'] = dict(scope=original_storage['scope'], accounting='ALL_INPUTS_AND_RESULTS_TO_RETURN',
        persistent_input_bytes=input_bytes, serial_peak_bytes=original_storage['serial_peak_bytes'],
        peak_tensor_storage_bytes=peak, budget_bytes=tensor_budget_bytes, excluded_bytes=original_storage['excluded_bytes'])
    serial = _prediction(sum(service[0]), [sum(row) for row in lane_samples[0]])
    report['predictions']['SERIAL'] = serial
    if ready_dag_abi == 1 and peak > tensor_budget_bytes:
        return dict(report, reason='READY_DAG_TENSOR_BUDGET_EXCEEDED')

    exports = set(recipe['outputs'])
    output_indices = [i for i, op in enumerate(operations) if op['id'] in exports]
    if joint_order:
        report['joint_cpu_plan'] = solve_ready_joint(dependencies, service, handoffs, output_indices,
            result_bytes, input_bytes, tensor_budget_bytes)
    candidates = []
    for domain in _LANE_DOMAINS:
        lanes = _assign_lanes(dependencies, service, domain)
        width = max(lanes) + 1
        lane_order = _ready_order(dependencies, lanes)
        intervals = lane_order.intervals(_assigned_costs(service, lanes))
        if width > 1 and _overlap(intervals):
            candidates.append((_finish(intervals) + handoffs[str(width)], width, lanes, intervals, None, lane_order))
    if joint_order and report['joint_cpu_plan']['lanes'] is not None:
        solution = report['joint_cpu_plan']
        lanes, order = solution['lanes'], solution['operation_order']
        lane_order = _ready_order(dependencies, lanes, order)
        intervals = lane_order.intervals(_assigned_costs(service, lanes))
        width = max(lanes) + 1
        finish = _finish(intervals) + handoffs[str(width)]
        if finish != solution['estimate_ns']:
            raise ValueError('joint placement proof differs from the CPU ready graph')
        if _overlap(intervals):
            candidates.append((finish, width, lanes, intervals, order, lane_order))
    if not candidates:
        return dict(report, reason='READY_DAG_HAS_NO_MODELED_OVERLAP')

    candidates.sort(key=lambda row: (row[0], row[1], row[4] is not None, row[2]))
    if ready_dag_abi == 2:
        checked = {}
        def identity(row):
            return tuple(row[2]), tuple(row[4]) if row[4] is not None else None

        for row in candidates:
            key = identity(row)
            if key not in checked:
                checked[key] = ready_lifetime_bound(dependencies, row[2], output_indices, result_bytes, input_bytes,
                                                    operation_order=row[4])
        eligible = [row for row in candidates if checked[identity(row)]['peak_tensor_storage_bytes'] <= tensor_budget_bytes]
        bound = checked[identity(eligible[0])] if eligible else min(
            checked.values(), key=lambda proof: proof['peak_tensor_storage_bytes'])
        report['tensor_storage'].update(accounting='RELEASE_ON_LAST_COMPLETED_CONSUMER',
            full_retention_bytes=peak, lifetime_proof=bound,
            peak_tensor_storage_bytes=bound['peak_tensor_storage_bytes'])
        if not eligible:
            return dict(report, reason='READY_DAG_TENSOR_BUDGET_EXCEEDED')
        candidates = eligible

    estimate, width, lanes, intervals, order, lane_order = candidates[0]
    if order is not None:
        report.update(operation_order=order, basis='JOINT_PLACEMENT_AND_ORDER_WITH_MEASURED_LANE_SERVICES')
    samples = [_finish(lane_order.intervals(_assigned_costs(measurements, lanes))) + handoffs_ns
               for measurements, handoffs_ns in zip(zip(*lane_samples, strict=True), handoff_samples[str(width)], strict=True)]
    prediction = _prediction(estimate, samples)
    if cross_numa:
        measured = calibration.get('ready_numa')
        if measured is None:
            return dict(report, status='READY_DAG_NUMA_CALIBRATION_REQUIRED',
                        reason='READY_DAG_NUMA_CALIBRATION_REQUIRED', lanes=lanes, launch_width=width)
        locality = model_ready_numa_locality(measured, recipe, schedule, lanes, warmup,
                                             result_storage_bytes=result_bytes)
        report['ready_numa'] = locality
        locality['isolated_prediction'] = prediction
        if locality['status'] != 'MEASURED':
            return dict(report, reason='READY_DAG_NUMA_PAGE_EVIDENCE_INCOMPLETE', lanes=lanes, launch_width=width)
        timings = locality['operation_samples_ns']
        intervals = lane_order.intervals(_column_medians(timings))
        estimate = _finish(intervals) + handoffs[str(width)]
        context_samples = [_finish(lane_order.intervals(measurements)) + handoffs_ns
                           for measurements, handoffs_ns in zip(timings, handoff_samples[str(width)], strict=True)]
        contextual = _prediction(estimate, context_samples)
        locality['contextual_prediction'] = contextual
        prediction = dict(estimate_ns=estimate,
                          sample_min_ns=min(prediction['sample_min_ns'], contextual['sample_min_ns']),
                          sample_max_ns=max(prediction['sample_max_ns'], contextual['sample_max_ns']))
        report.update(basis='SOURCE_ORDER_EARLIEST_FINISH_WITH_FIXED_LANE_NUMA_REPLAY',
                      bounds='ISOLATED_AND_CONTEXTUAL_CALIBRATION_SPAN_NOT_PHYSICAL_BOUNDS')
    report['predictions']['READY_DAG'] = prediction
    if not _passes_gain(serial['estimate_ns'], estimate, gain_policy):
        return dict(report, reason='READY_DAG_PREDICTED_GAIN_BELOW_THRESHOLD')

    return dict(report, status='PLAN_READY_DAG', lanes=lanes,
                timeline=_timeline(operations, lanes, cpus, dependencies, intervals), launch_width=width,
                dispatch_join_ns=handoffs[str(width)], timeline_scope='OPERATORS_EXCLUDING_SINGLE_DISPATCH_JOIN')


def _service_upper_bound(service, handoffs, count):
    if (type(service) not in (list, tuple) or len(service) != _LANE_COUNT
            or any(type(row) not in (list, tuple) or len(row) != count
                   or not all(map(_bounded_duration, row)) for row in service)):
        raise ValueError('joint ready scheduling requires per-operation durations for every CPU lane')
    if type(handoffs) is not dict or handoffs.keys() != {'2', '3'} or not all(map(_bounded_duration, handoffs.values())):
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
    intervals = _ready_order(dependencies, lanes, order).intervals(costs)
    order.sort(key=lambda op: (intervals[op][0], op))
    release = _finish(intervals)
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


def nominate_ready_replica(recipe, schedule, model):
    """Choose one same-version external Tensor with repeated remote reads.

    The caller supplies an admitted, effect-checked graph. The weighted input
    bytes prioritize measurement; they do not estimate memory traffic.
    """
    locality = model.get('ready_numa')
    if model.get('status') != 'PLAN_READY_DAG' or not locality or locality.get('status') != 'MEASURED':
        return None
    if schedule['schema_version'] != 2 or schedule.get('cross_numa') is not True:
        return None
    if model['cpu_ids'] != schedule['cpus'] or model['lanes'] != locality['lanes']:
        raise ValueError('ready replica requires the exact measured CPU lane assignment')
    candidates = []
    storage = model['tensor_storage']
    budget = storage['budget_bytes']
    peak = storage['peak_tensor_storage_bytes']
    if not _bytes(budget) or not _bytes(peak) or peak > budget:
        return None

    positions = {row['id']: i for i, row in enumerate(recipe['inputs'])}
    consumers = {}
    for index, (operation, lane) in enumerate(zip(recipe['operations'], model['lanes'], strict=True)):
        for value in dict.fromkeys(arg.get('value') for arg in operation['arguments']):
            if value in positions:
                consumers.setdefault((value, lane), []).append(index)
    for (value, lane), reads in consumers.items():
        if len(reads) < 2:
            continue
        input_index = positions[value]
        byte_count = 4 * prod(recipe['inputs'][input_index]['shape'])
        if not byte_count:
            continue
        pages = _page_sample(locality['input_pages'][value], byte_count)
        dest_node = schedule['lanes'][lane]['numa_node']
        exposed = _remote_fraction(pages, dest_node)
        if exposed is None or not exposed[0]:
            continue
        page_size = pages['page_size_bytes']
        allocated = ((byte_count + page_size - 1) // page_size) * page_size
        if allocated > budget - peak:
            continue
        score = Fraction(byte_count * len(reads) * exposed[0], exposed[1])
        candidate = dict(value=value, input_index=input_index, lane=lane, dest_node=dest_node,
                         consumers=reads, bytes=byte_count, allocated_bytes=allocated,
                         exposure_score_bytes=int(score))
        candidates.append((-score, input_index, lane, candidate))
    return min(candidates)[-1] if candidates else None


def price_ready_replica(recipe, schedule, model, candidate, raw, warmup):
    """Estimate the scheduled critical path, including one lane-local copy."""
    nominated = nominate_ready_replica(recipe, schedule, model)
    if nominated is None or candidate != nominated:
        raise ValueError('ready replica does not match the admitted graph and storage budget')
    candidate = nominated
    if type(warmup) is not int or not 3 <= warmup <= 100:
        raise ValueError('ready replica calibration requires three to one hundred warmups')
    if (type(raw) is not dict or raw.keys() != _CALIBRATION_FIELDS
            or raw['cpu_ids'] != schedule['cpus'] or raw['lanes'] != model['lanes']
            or any(type(value) is not int for value in (*raw['cpu_ids'], *raw['lanes']))
            or any(type(raw[key]) is not int for key in ('input_index', 'lane', 'warmup_pairs'))
            or raw['input_index'] != candidate['input_index'] or raw['lane'] != candidate['lane']
            or raw['warmup_pairs'] != warmup or raw['all_runs_validated'] is not True):
        raise ValueError('ready replica calibration differs from its admitted source and CPU placement')

    count = len(recipe['operations'])
    direct, copied = raw['direct_samples_ns'], raw['replicated_samples_ns']
    positions = set(range(count))
    _validate_samples(direct, warmup, count, positions)
    _validate_samples(copied, warmup, count, positions)
    copy = raw['copy_samples_ns']
    if type(copy) is not list or len(copy) != warmup or not all(map(_bounded_duration, copy)):
        raise ValueError('ready replica requires one bounded copy duration in each calibration pair')

    pages = _page_sample(raw['replica_pages'], candidate['bytes'])
    original_pages = model['ready_numa']['input_pages'][candidate['value']]
    if (pages['page_size_bytes'] != original_pages['page_size_bytes']
            or pages['storage_pages'] * pages['page_size_bytes'] != candidate['allocated_bytes']
            or pages['sampled_pages'] != min(pages['storage_pages'], _MAX_PAGES_PER_STORAGE)):
        raise ValueError('ready replica page observations differ from its planned mapping geometry')

    lane = candidate['lane']
    cpus, lanes = schedule['cpus'], model['lanes']
    dispatch = model['dispatch_join_ns']
    dependencies = data_dependencies(recipe)
    if dependencies != model['data_dependencies'] or not _bounded_duration(dispatch):
        raise ValueError('ready replica requires the admitted DATA graph and dispatch time')
    report = dict(status='KEEP_DIRECT_READY', reason=None, candidate=candidate, predictions={}, timeline=[],
                  copy_ns=median_low(copy), replica_pages=pages,
                  scope='CONCURRENT_PROVIDER_SAMPLE_CRITICAL_PATH_WITH_LANE_LOCAL_COPY',
                  bounds='CALIBRATION_SAMPLE_SPAN_NOT_PHYSICAL_BOUNDS',
                  source_replacement_authorized=False,
                  peak_tensor_storage_bytes=_bounded_sum((model['tensor_storage']['peak_tensor_storage_bytes'],
                                                          candidate['allocated_bytes'])))
    if pages['status'] != 'OBSERVED':
        return dict(report, reason='READY_REPLICA_PAGE_EVIDENCE_INCOMPLETE')
    if pages['node_pages'] != {str(candidate['dest_node']): pages['sampled_pages']}:
        return dict(report, reason='READY_REPLICA_PAGES_NOT_ON_DESTINATION_NODE')

    def total(intervals):
        return _bounded_sum((_finish(intervals), dispatch))

    lane_order = _ready_order(dependencies, lanes)
    direct_intervals = lane_order.intervals(_column_medians(direct))
    copied_intervals = lane_order.intervals(_column_medians(copied), prologue_lane=lane, prologue_ns=report['copy_ns'])
    direct_ns = total(direct_intervals)
    copied_ns = total(copied_intervals)
    report['predictions'] = {
        'DIRECT': _prediction(direct_ns, [total(lane_order.intervals(row)) for row in direct]),
        'REPLICATED': _prediction(copied_ns, [total(lane_order.intervals(row, prologue_lane=lane, prologue_ns=duration))
                                            for row, duration in zip(copied, copy, strict=True)]),
    }
    report['copy_prologue'] = dict(lane=lane, cpu_id=cpus[lane], start_ns=0, end_ns=report['copy_ns'])
    report['timeline'] = _timeline(recipe['operations'], lanes, cpus, dependencies, copied_intervals)
    if copied_ns * 100 <= direct_ns * (100 - _GAIN_PERCENT):
        return dict(report, status='PLAN_READY_REPLICA')
    return dict(report, reason='READY_REPLICA_PREDICTED_GAIN_BELOW_THRESHOLD')


def price_ready_retained(recipe, schedule, model, candidate, raw, warmup):
    """Compare the matched schedule with a content-checked retained mapping.

    Warm calls use modeled critical paths. Cold calls use measured region
    wall time, including initialization. Amortization assumes one cold call
    followed by uninterrupted hits. Actual workload reuse is unknown.
    """
    if type(warmup) is not int or not 3 <= warmup <= 100:
        raise ValueError('ready retained calibration requires three to one hundred warmups')
    if type(raw) is not dict or raw.keys() != _RETAINED_FIELDS:
        raise ValueError('ready retained calibration requires matched direct, warm retained and cold observations')
    cold = raw['cold_samples_ns']
    if type(cold) is not list or len(cold) != warmup or not all(map(_bounded_duration, cold)):
        raise ValueError('ready retained calibration requires one bounded cold call in each calibration pair')

    # The same nominated mapping, page evidence and lane-local predecessor
    # apply. On a warm retained call the predecessor is an exact content check.
    report = price_ready_replica(recipe, schedule, model, candidate, {
        field: raw[_CALIBRATION_RENAMES.get(field, field)] for field in _CALIBRATION_FIELDS
    }, warmup)
    report.update(status='KEEP_DIRECT_READY', check_ns=report.pop('copy_ns'),
                  scope='CONCURRENT_PROVIDER_SAMPLE_CRITICAL_PATH_WITH_LANE_LOCAL_EXACT_CONTENT_CHECK',
                  cold_first_call=dict(_prediction(median_low(cold), cold),
                                       scope='MEASURED_INITIAL_RETAINED_REGION_WALL_TIME_WITHOUT_OPERATION_PROBES'),
                  break_even_total_calls=None, break_even_gain_percent=_GAIN_PERCENT,
                  break_even_basis='ONE_COLD_CALL_THEN_ALL_HITS', reuse_rate='UNKNOWN',
                  resident_mapping_count=1)
    if report['reason']:
        report['reason'] = report['reason'].replace('READY_REPLICA_', 'READY_RETAINED_', 1)
    if not report['predictions']:
        return report

    predictions = report['predictions']
    predictions['RETAINED'] = predictions.pop('REPLICATED')
    direct_ns = predictions['DIRECT']['estimate_ns']
    hot_ns = predictions['RETAINED']['estimate_ns']
    headroom = (100 - _GAIN_PERCENT) * direct_ns - 100 * hot_ns
    if headroom < 0:
        return dict(report, reason='READY_RETAINED_PREDICTED_GAIN_BELOW_THRESHOLD')

    # Solve C + (N-1)H <= N * D * (1-gain) for the first N with a hit.
    overhead = 100 * (report['cold_first_call']['estimate_ns'] - hot_ns)
    if headroom == 0 and overhead > 0:
        return dict(report, reason='READY_RETAINED_NO_FINITE_BREAK_EVEN')
    calls = max(2, (overhead + headroom - 1) // headroom) if headroom else 2
    return dict(report, status='PLAN_READY_RETAINED', reason=None, break_even_total_calls=calls)
