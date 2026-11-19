"""Nominate and price a private input materialization on one ready-DAG lane."""

from fractions import Fraction
from math import prod
from statistics import median_low

from scar.optimization.aten_native_team import _bounded_duration, _validate_samples, data_dependencies
from scar.optimization.aten_numa_locality import _page_sample, _remote_fraction
from scar.optimization.aten_ready_dag import _finish, _intervals, _prediction
from scar.optimization.aten_team_numa_locality import _MAX_PAGES_PER_STORAGE
from scar.optimization.aten_tensor_lifetime import _bounded_sum, _bytes


_CALIBRATION_FIELDS = {
    'cpu_ids', 'lanes', 'input_index', 'lane', 'warmup_pairs', 'direct_samples_ns',
    'replicated_samples_ns', 'copy_samples_ns', 'replica_pages', 'all_runs_validated',
}


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


def _replica_intervals(dependencies, lanes, costs, copy_lane, copy_ns):
    # Add a lane-local predecessor. Other lane heads remain immediately ready.
    shifted = [[]] + [[parent + 1 for parent in parents] for parents in dependencies]
    intervals = _intervals(shifted, [copy_lane, *lanes], [copy_ns, *costs])
    return [(start, end, None if prior in (None, 0) else prior - 1) for start, end, prior in intervals[1:]]


def _column_medians(samples):
    return [median_low(column) for column in zip(*samples, strict=True)]


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
    _validate_samples(direct, warmup, count, set(range(count)))
    _validate_samples(copied, warmup, count, set(range(count)))
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

    direct_intervals = _intervals(dependencies, lanes, _column_medians(direct))
    copied_intervals = _replica_intervals(dependencies, lanes, _column_medians(copied), lane, report['copy_ns'])
    direct_ns = total(direct_intervals)
    copied_ns = total(copied_intervals)
    report['predictions'] = {
        'DIRECT': _prediction(direct_ns, [total(_intervals(dependencies, lanes, row)) for row in direct]),
        'REPLICATED': _prediction(copied_ns, [total(_replica_intervals(dependencies, lanes, row, lane, duration))
                                            for row, duration in zip(copied, copy, strict=True)]),
    }
    report['copy_prologue'] = dict(lane=lane, cpu_id=cpus[lane], start_ns=0, end_ns=report['copy_ns'])
    report['timeline'] = [dict(index=i, operation=op['id'], lane=assigned, cpu_id=cpus[assigned], start_ns=start,
                               end_ns=end, data_predecessors=dependencies[i], lane_predecessor=prior)
                          for i, (op, assigned, (start, end, prior)) in enumerate(
                              zip(recipe['operations'], lanes, copied_intervals, strict=True))]
    if copied_ns * 100 <= direct_ns * 95:
        return dict(report, status='PLAN_READY_REPLICA')
    return dict(report, reason='READY_REPLICA_PREDICTED_GAIN_BELOW_THRESHOLD')
