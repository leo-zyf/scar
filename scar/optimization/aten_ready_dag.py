"""Price a source-ordered CPU lane schedule over original Tensor dependencies."""

from statistics import median_low

from scar.optimization.aten_native_team import READY_SERVICE_SCOPE, _measurements, _storage, _team_groups, data_dependencies
from scar.optimization.aten_ready_joint import ordered_ready_intervals, solve_ready_joint
from scar.optimization.aten_ready_lifetime import ready_lifetime_bound
from scar.optimization.aten_ready_numa import model_ready_numa_locality
from scar.optimization.aten_tensor_lifetime import _bounded_sum, _bytes


_MIN_GAIN = 0.05
_LANE_DOMAINS = ((0, 1), (0, 2), (0, 1, 2))


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


def _intervals(dependencies, lanes, costs):
    intervals = []
    previous = [None] * (max(lanes) + 1)
    for index, (parents, lane) in enumerate(zip(dependencies, lanes, strict=True)):
        prior = previous[lane]
        constraints = parents if prior is None else (*parents, prior)
        start = max((intervals[parent][1] for parent in constraints), default=0)
        intervals.append((start, start + costs[index], prior))
        previous[lane] = index
    return intervals


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


def plan_ready_dag(recipe, schedule, calibration, warmup, *, tensor_budget_bytes, ready_dag_abi=2, joint_order=False):
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
        intervals = _intervals(dependencies, lanes, _assigned_costs(service, lanes))
        if width > 1 and _overlap(intervals):
            candidates.append((_finish(intervals) + handoffs[str(width)], width, lanes, intervals, None))
    if joint_order and report['joint_cpu_plan']['lanes'] is not None:
        solution = report['joint_cpu_plan']
        lanes, order = solution['lanes'], solution['operation_order']
        intervals = ordered_ready_intervals(dependencies, lanes, _assigned_costs(service, lanes), order)
        width = max(lanes) + 1
        finish = _finish(intervals) + handoffs[str(width)]
        if finish != solution['estimate_ns']:
            raise ValueError('joint placement proof differs from the CPU ready graph')
        if _overlap(intervals):
            candidates.append((finish, width, lanes, intervals, order))
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

    estimate, width, lanes, intervals, order = candidates[0]
    if order is None:
        replay = _intervals
    else:
        def replay(dependencies, lanes, costs):
            return ordered_ready_intervals(dependencies, lanes, costs, order)

        report.update(operation_order=order, basis='JOINT_PLACEMENT_AND_ORDER_WITH_MEASURED_LANE_SERVICES')
    samples = [_finish(replay(dependencies, lanes, _assigned_costs(measurements, lanes))) + handoffs_ns
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
        intervals = _intervals(dependencies, lanes, [median_low(samples) for samples in zip(*timings, strict=True)])
        estimate = _finish(intervals) + handoffs[str(width)]
        context_samples = [_finish(_intervals(dependencies, lanes, measurements)) + handoffs_ns
                           for measurements, handoffs_ns in zip(timings, handoff_samples[str(width)], strict=True)]
        contextual = _prediction(estimate, context_samples)
        locality['contextual_prediction'] = contextual
        prediction = dict(estimate_ns=estimate,
                          sample_min_ns=min(prediction['sample_min_ns'], contextual['sample_min_ns']),
                          sample_max_ns=max(prediction['sample_max_ns'], contextual['sample_max_ns']))
        report.update(basis='SOURCE_ORDER_EARLIEST_FINISH_WITH_FIXED_LANE_NUMA_REPLAY',
                      bounds='ISOLATED_AND_CONTEXTUAL_CALIBRATION_SPAN_NOT_PHYSICAL_BOUNDS')
    report['predictions']['READY_DAG'] = prediction
    if (serial['estimate_ns'] - estimate) / serial['estimate_ns'] < _MIN_GAIN:
        return dict(report, reason='READY_DAG_PREDICTED_GAIN_BELOW_THRESHOLD')

    timeline = [dict(index=i, operation=operations[i]['id'], lane=lane, cpu_id=cpus[lane],
                     start_ns=start, end_ns=end, data_predecessors=dependencies[i], lane_predecessor=prior)
                for i, (lane, (start, end, prior)) in enumerate(zip(lanes, intervals, strict=True))]
    return dict(report, status='PLAN_READY_DAG', lanes=lanes, timeline=timeline, launch_width=width,
                dispatch_join_ns=handoffs[str(width)], timeline_scope='OPERATORS_EXCLUDING_SINGLE_DISPATCH_JOIN')
