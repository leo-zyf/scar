"""Price a source-ordered CPU lane schedule over original Tensor dependencies."""

from scar.optimization.aten_native_team import _measurements, _storage, _team_groups
from scar.optimization.aten_ready_lifetime import ready_lifetime_bound
from scar.optimization.aten_tensor_lifetime import _bounded_sum, _bytes, tensor_releases


_MIN_GAIN = 0.05
_LANE_DOMAINS = ((0, 1), (0, 2), (0, 1, 2))


def data_dependencies(recipe):
    """Recover direct Tensor producers, including a single edge for repeated operands."""
    tensor_releases(recipe)
    operations = recipe['operations']
    producers = {operation['id']: index for index, operation in enumerate(operations)}
    return [sorted({producers[arg['value']] for arg in operation['arguments'] if arg.get('value') in producers})
            for operation in operations]


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


def _intervals(dependencies, lanes, service):
    intervals = []
    previous = [None] * len(service)
    for index, (parents, lane) in enumerate(zip(dependencies, lanes, strict=True)):
        prior = previous[lane]
        constraints = parents if prior is None else (*parents, prior)
        start = max((intervals[parent][1] for parent in constraints), default=0)
        intervals.append((start, start + service[lane][index], prior))
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


def plan_ready_dag(recipe, schedule, calibration, warmup, *, tensor_budget_bytes, ready_dag_abi=2):
    """Nominate a deterministic lane schedule from the measured placement alone.

    Callers must replay the exact concurrency declaration for every operator.
    Each lane is traversed in original source order, with waits on original
    DATA predecessors only. Same-NUMA isolated times omit contention and
    notification overhead. The bounded model requires later matched validation.
    """
    dependencies = data_dependencies(recipe)
    operations = recipe['operations']
    windows = _team_groups(recipe, schedule)
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
    if None in nodes or len(set(nodes)) != 1:
        return dict(report, reason='READY_DAG_REQUIRES_SAME_NUMA_NODE')
    if tensor_budget_bytes is None:
        return dict(report, reason='READY_DAG_REQUIRES_TENSOR_BUDGET')
    if not _bytes(tensor_budget_bytes) or tensor_budget_bytes == 0:
        raise ValueError('ready DAG Tensor budget must be a positive bounded byte count')

    service, handoffs, lane_samples, handoff_samples = _measurements(calibration, warmup, len(operations), windows)
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

    candidates = []
    for domain in _LANE_DOMAINS:
        lanes = _assign_lanes(dependencies, service, domain)
        width = max(lanes) + 1
        intervals = _intervals(dependencies, lanes, service)
        if width > 1 and _overlap(intervals):
            candidates.append((_finish(intervals) + handoffs[str(width)], width, lanes, intervals))
    if not candidates:
        return dict(report, reason='READY_DAG_HAS_NO_MODELED_OVERLAP')

    candidates.sort()
    if ready_dag_abi == 2:
        exports = {value for value in recipe['outputs']}
        output_indices = [i for i, op in enumerate(operations) if op['id'] in exports]
        checked = {}
        for _, _, lanes, _ in candidates:
            key = tuple(lanes)
            if key not in checked:
                checked[key] = ready_lifetime_bound(dependencies, lanes, output_indices, result_bytes, input_bytes)
        eligible = [row for row in candidates
                    if checked[tuple(row[2])]['peak_tensor_storage_bytes'] <= tensor_budget_bytes]
        bound = checked[tuple(eligible[0][2])] if eligible else min(
            checked.values(), key=lambda proof: proof['peak_tensor_storage_bytes'])
        report['tensor_storage'].update(accounting='RELEASE_ON_LAST_COMPLETED_CONSUMER',
            full_retention_bytes=peak, lifetime_proof=bound,
            peak_tensor_storage_bytes=bound['peak_tensor_storage_bytes'])
        if not eligible:
            return dict(report, reason='READY_DAG_TENSOR_BUDGET_EXCEEDED')
        candidates = eligible

    estimate, width, lanes, intervals = candidates[0]
    samples = [_finish(_intervals(dependencies, lanes, measurements)) + handoffs_ns
               for measurements, handoffs_ns in zip(zip(*lane_samples, strict=True), handoff_samples[str(width)],
                                                     strict=True)]
    prediction = _prediction(estimate, samples)
    report['predictions']['READY_DAG'] = prediction
    if (serial['estimate_ns'] - estimate) / serial['estimate_ns'] < _MIN_GAIN:
        return dict(report, reason='READY_DAG_PREDICTED_GAIN_BELOW_THRESHOLD')

    timeline = [dict(index=i, operation=operations[i]['id'], lane=lane, cpu_id=cpus[lane],
                     start_ns=start, end_ns=end, data_predecessors=dependencies[i], lane_predecessor=prior)
                for i, (lane, (start, end, prior)) in enumerate(zip(lanes, intervals, strict=True))]
    return dict(report, status='PLAN_READY_DAG', lanes=lanes, timeline=timeline, launch_width=width,
                dispatch_join_ns=handoffs[str(width)], timeline_scope='OPERATORS_EXCLUDING_SINGLE_DISPATCH_JOIN')
