"""Bind measured CPU Tensor page samples to native graph data dependencies."""

from dataclasses import dataclass
from itertools import combinations
from math import prod

from scar.optimization.aten_live_values import tensor_releases


_FRONTIERS = ('REQUIRES_SINGLE_LEGAL_WINDOW', 'GRAPH_EXCEEDS_LOCALITY_SAMPLE_BUDGET')
_PAGE_KEYS = {'storage_bytes', 'page_size_bytes', 'storage_pages', 'sampled_pages',
              'node_pages', 'unknown_pages', 'query_errno', 'status'}
_MEASUREMENT_KEYS = {'schema_version', 'status', 'window_start', 'input_pages', 'caller_output_pages',
                     'worker_output_pages', 'worker_output_consumer_samples_ns'}
_MAX_CONTEXTS = 8
_MAX_SAMPLE_STORAGES = 256
_MAX_PAGES_PER_STORAGE = 32
_JOINT_BUDGET_FRONTIER = 'JOINT_CONTEXT_BUDGET_EXCEEDED'
_TEAM_BUDGET_FRONTIER = 'TEAM_CONTEXT_BUDGET_EXCEEDED'
_READY_SCOPE = 'SOURCE_ORDER_REPLAY_ON_PLANNED_LANES'
_READY_FIELDS = {'schema_version', 'scope', 'lanes', 'input_pages', 'output_pages',
                 'operation_samples_ns', 'result_storage_bytes'}


def _tensor_operands(operation, *, with_positions=False):
    arguments = operation['arguments']
    if not with_positions:
        return dict.fromkeys(argument['value'] for argument in arguments if 'value' in argument)
    positions = {}
    for index, argument in enumerate(arguments):
        if 'value' in argument:
            positions.setdefault(argument['value'], []).append(index)
    return positions


@dataclass(frozen=True, slots=True)
class _TensorReadIndex:
    operations: list
    inputs: dict[str, int]
    results: dict[str, int]
    operands: list[dict[str, None]]


def _tensor_reads(recipe):
    operations = recipe['operations']
    return _TensorReadIndex(
        operations,
        {item['id']: index for index, item in enumerate(recipe['inputs'])},
        {item['id']: index for index, item in enumerate(operations)},
        [_tensor_operands(operation) for operation in operations],
    )


def _nonnegative_integer(value):
    return type(value) is int and value >= 0


def _page_sample(sample, expected_storage_bytes):
    if type(sample) is not dict or sample.keys() != _PAGE_KEYS:
        raise ValueError('NUMA locality requires a complete Tensor storage page sample')
    storage_bytes, page_size, pages, count, unknown = (
        sample[key] for key in ('storage_bytes', 'page_size_bytes', 'storage_pages', 'sampled_pages', 'unknown_pages'))
    if any(not _nonnegative_integer(value) for value in (storage_bytes, page_size, pages, count, unknown)):
        raise ValueError('NUMA storage byte and page counts must be nonnegative integers')
    if expected_storage_bytes is not None and storage_bytes != expected_storage_bytes:
        raise ValueError('NUMA page sample differs from the Tensor storage geometry')
    nodes, error, status = (sample[key] for key in ('node_pages', 'query_errno', 'status'))
    if type(nodes) is not dict or any(
            type(node) is not str or not node.isascii() or not node.isdecimal() or len(node) > 10
            or str(int(node)) != node or int(node) >= 2**31 or type(value) is not int or value <= 0
            for node, value in nodes.items()):
        raise ValueError('NUMA page samples require kernel node IDs and positive per-node counts')
    if count != unknown + sum(nodes.values()) or count > pages:
        raise ValueError('NUMA node and unknown page counts do not match sampled storage pages')
    if pages:
        if not page_size or not storage_bytes:
            raise ValueError('sampled pages require nonempty storage and a reported page size')
        minimum = (storage_bytes + page_size - 1) // page_size
        if pages not in (minimum, minimum + 1):
            raise ValueError('NUMA storage page span differs from its reported byte size')

    if status == 'EMPTY':
        valid = storage_bytes == pages == count == 0 and page_size > 0 and error is None
    elif status == 'UNAVAILABLE':
        valid = type(error) is int and error > 0 and not nodes
    elif status == 'PARTIAL':
        valid = count > 0 and unknown > 0 and error is None
    elif status == 'OBSERVED':
        valid = count > 0 and unknown == 0 and error is None
    else:
        valid = False
    if not valid:
        raise ValueError('NUMA page sample counts contradict the reported query status')
    return sample


def _ordered_samples(samples, sizes):
    if type(samples) is not list or len(samples) != len(sizes):
        raise ValueError('NUMA page samples do not cover the ordered Tensor storage slots')
    return [_page_sample(sample, size) for sample, size in zip(samples, sizes, strict=True)]


def _baseline_page_samples(locality, recipe, sizes, *, size_error):
    inputs = _ordered_samples(locality['input_pages'], [4 * prod(item['shape']) for item in recipe['inputs']])
    operation_count = len(recipe['operations'])
    sizes = [None] * operation_count if sizes is None else sizes
    if len(sizes) != operation_count:
        raise ValueError(size_error)
    baseline = _ordered_samples(locality['caller_output_pages'], sizes)
    return inputs, baseline, [sample['storage_bytes'] for sample in baseline]


def _validate_page_set(samples, *, size_error, bounds_error=None):
    if bounds_error is not None and any(
        sample['sampled_pages'] != min(sample['storage_pages'], _MAX_PAGES_PER_STORAGE) for sample in samples
    ):
        raise ValueError(bounds_error)
    if len({sample['page_size_bytes'] for sample in samples} - {0}) > 1:
        raise ValueError(size_error)


def _validate_service_times(samples, repetitions, operations, *, error, upper_limit=None):
    if (type(samples) is not list or len(samples) != repetitions or any(
        type(row) is not list or len(row) != operations or any(
            type(value) is not int or value < 1 for value in row
        ) for row in samples
    ) or (upper_limit is not None and any(max(row, default=0) >= upper_limit for row in samples))):
        raise ValueError(error)


def _bounded_duration(value):
    return type(value) is int and 0 < value < 2**63


def _validate_samples(samples, count, operation_count, positions):
    if (type(samples) is not list or len(samples) != count
            or any(type(row) is not list or len(row) != operation_count for row in samples)):
        raise ValueError('native team calibration requires a rectangular operation sample matrix per lane')
    for row in samples:
        for index, duration in enumerate(row):
            if index in positions and not _bounded_duration(duration):
                raise ValueError('a legal native team lane is missing its measured service time')
            if index not in positions and duration is not None:
                raise ValueError('native team calibration measured a disallowed worker position')


def _select_timing_columns(timings, selected):
    return [[value if index in selected else None for index, value in enumerate(row)] for row in timings]


def _remote_fraction(sample, caller_numa_node):
    if sample['status'] != 'OBSERVED' or caller_numa_node is None:
        return None
    total = sample['sampled_pages']
    return [total - sample['node_pages'].get(str(caller_numa_node), 0), total]


def _fraction_changed(before, after):
    return before is not None and after is not None and before[0] * after[1] != after[0] * before[1]


def _input_exposure(inputs, pages, caller_node, worker_node):
    return [dict(value=item['id'], pages=page, caller_remote_pages=_remote_fraction(page, caller_node),
                 worker_remote_pages=_remote_fraction(page, worker_node))
            for item, page in zip(inputs, pages, strict=True)]


def model_aten_numa_locality(locality, recipe, schedule, *, caller_numa_node, worker_numa_node,
                             warmup_pairs, result_storage_bytes=None):
    """Use paired consumer timings only for a measured change in sampled remote exposure.

    The evidence describes the standalone calibration allocations. Page
    distributions are sampled observations, and a timing is an operation cost
    in that measured context rather than a separately identified link penalty.
    """
    if locality is None:
        return None
    if (type(locality) is not dict or type(locality.get('schema_version')) is not int
            or locality['schema_version'] != 1):
        raise ValueError('NUMA locality evidence version is unsupported')
    status = locality.get('status')
    if status in _FRONTIERS:
        if locality.keys() != {'schema_version', 'status'}:
            raise ValueError('unmeasured NUMA locality cannot carry observations')
        return dict(status=status, sampled_node_residency_observed=False,
                    affected_caller_operation_indices=[], replacement_operation_samples_ns=None)
    if status != 'MEASURED' or locality.keys() != _MEASUREMENT_KEYS:
        raise ValueError('incomplete measured NUMA locality evidence')
    if any(node is not None and not _nonnegative_integer(node) for node in (caller_numa_node, worker_numa_node)):
        raise ValueError('NUMA locality requires target-derived CPU node IDs')
    start = locality['window_start']
    operations = recipe['operations']
    if (type(start) is not int or not 0 <= start < len(operations) - 1
            or [window['pair_indices'] for window in schedule['windows']] != [[start, start + 1]]):
        raise ValueError('measured NUMA transition must match the single scheduled legal CPU window')
    producer = start + 1
    timings = locality['worker_output_consumer_samples_ns']
    if (type(timings) is not list or len(timings) != warmup_pairs or any(
            type(row) is not list or len(row) != len(operations) or any(
                (type(value) is not int or value <= 0) if index > producer else value is not None
                for index, value in enumerate(row)) for row in timings)):
        raise ValueError('NUMA timings must align exactly with caller operations after the worker result')

    inputs, baseline, sizes = _baseline_page_samples(locality, recipe, result_storage_bytes,
        size_error='NUMA results require one storage size per operation')
    shifted = _ordered_samples(locality['worker_output_pages'], sizes)
    _validate_page_set((*inputs, *baseline, *shifted),
        size_error='NUMA samples from one calibration disagree on the kernel page size')

    origin_before = _remote_fraction(baseline[producer], caller_numa_node)
    origin_after = _remote_fraction(shifted[producer], caller_numa_node)
    origin_changed = _fraction_changed(origin_before, origin_after)
    cross_node = caller_numa_node is not None and worker_numa_node is not None and caller_numa_node != worker_numa_node
    identifier = operations[producer]['id']
    paths, affected, consumers = {identifier: [identifier]}, [], []
    results = {op['id']: i for i, op in enumerate(operations)}
    for index, operation in enumerate(operations[producer + 1:], start=producer + 1):
        dependencies = [value for value in _tensor_operands(operation) if value in paths]
        if not dependencies:
            continue
        reads = []
        for source in dependencies:
            source_index = results[source]
            before = _remote_fraction(baseline[source_index], caller_numa_node)
            after = _remote_fraction(shifted[source_index], caller_numa_node)
            reads.append(dict(value=source, path=[*paths[source], operation['id']],
                              caller_produced_pages=baseline[source_index], shifted_producer_pages=shifted[source_index],
                              baseline_remote_pages=before, shifted_remote_pages=after,
                              sampled_remote_fraction_changed=_fraction_changed(before, after)))
        paths[operation['id']] = reads[0]['path']
        consumers.append(dict(operation=operation['id'], index=index, affected_reads=reads))
        if cross_node and origin_changed and any(row['sampled_remote_fraction_changed'] for row in reads):
            affected.append(index)

    replacements = _select_timing_columns(timings, set(affected)) if affected else None
    if affected:
        reason = None
    elif caller_numa_node is None or worker_numa_node is None:
        reason = 'TARGET_LANE_NUMA_NODE_UNAVAILABLE'
    elif not cross_node:
        reason = 'SAME_EXECUTION_NUMA_NODE'
    elif origin_before is None or origin_after is None:
        reason = 'PRODUCER_PAGE_SAMPLE_INCOMPLETE'
    elif not origin_changed:
        reason = 'PRODUCER_REMOTE_EXPOSURE_UNCHANGED'
    else:
        reason = 'NO_GRAPH_CALLER_CONSUMER'
    input_exposure = _input_exposure(recipe['inputs'], inputs, caller_numa_node, worker_numa_node)
    return dict(status='MEASURED_CALLER_CONSUMER_CONTEXT' if affected else 'NO_OBSERVED_CALLER_CONSUMER_TRANSITION',
                reason=reason, window_start=start, cpu_ids=schedule['cpus'],
                caller_numa_node=caller_numa_node, worker_numa_node=worker_numa_node,
                sampled_node_residency_observed=any(row['node_pages'] for row in (*inputs, *baseline, *shifted)),
                input_exposure=input_exposure,
                worker_result=dict(value=identifier, caller_produced_pages=baseline[producer],
                                   shifted_producer_pages=shifted[producer], baseline_remote_pages=origin_before,
                                   shifted_remote_pages=origin_after, sampled_remote_fraction_changed=origin_changed),
                affected_caller_operation_indices=affected, consumer_paths=consumers,
                replacement_operation_samples_ns=replacements)


def _joint_windows(schedule, input_count, operation_count):
    starts = []
    for first, second in (window['pair_indices'] for window in schedule['windows']):
        if (type(first) is not int or type(second) is not int or not 0 <= first < operation_count - 1
                or second != first + 1):
            raise ValueError('joint NUMA evidence requires legal adjacent CPU windows')
        starts.append(first)
    if len(starts) < 2 or len(starts) != len(set(starts)):
        raise ValueError('joint NUMA evidence requires distinct legal CPU windows')
    if len(starts) > _MAX_CONTEXTS:
        return None

    contexts = []
    for count in range(1, len(starts) + 1):
        for selection in combinations(sorted(starts), count):
            if any(right == left + 1 for left, right in zip(selection, selection[1:])):
                continue
            contexts.append(selection)
            if (len(contexts) > _MAX_CONTEXTS
                    or input_count + (1 + len(contexts)) * operation_count > _MAX_SAMPLE_STORAGES):
                return None
    return sorted(contexts)


def _context_samples(context, expected_starts, sizes, warmup_pairs):
    if type(context) is not dict or context.keys() != {'window_starts', 'output_pages', 'operation_samples_ns'}:
        raise ValueError('joint NUMA context requires output page and operation observations')
    starts = context['window_starts']
    if type(starts) is not list or any(type(start) is not int for start in starts) or tuple(starts) != expected_starts:
        raise ValueError('joint NUMA contexts must cover the complete ordered legal set')
    output_pages = _ordered_samples(context['output_pages'], sizes)
    timings = context['operation_samples_ns']
    _validate_service_times(timings, warmup_pairs, len(sizes),
        error='joint NUMA timings require a positive service sample per operation and calibration row')
    return output_pages, timings


def _joint_read_context(graph, starts, baseline, pages, timings, caller_node, worker_node):
    operations = graph.operations
    worker_indices = {start + 1 for start in starts}
    moved_results = {}
    for index in sorted(worker_indices):
        before = _remote_fraction(baseline[index], caller_node)
        after = _remote_fraction(pages[index], caller_node)
        moved_results[operations[index]['id']] = dict(
            operation_index=index, caller_produced_pages=baseline[index], shifted_producer_pages=pages[index],
            baseline_remote_pages=before, shifted_remote_pages=after,
            sampled_remote_fraction_changed=_fraction_changed(before, after),
        )

    paths, consumers = {}, []
    affected_caller, affected_worker = [], []
    for index, operation in enumerate(operations):
        name = operation['id']
        runs_on_worker = index in worker_indices
        consumer_node = worker_node if runs_on_worker else caller_node
        next_paths, reads = {}, []
        for source in graph.operands[index]:
            if source not in paths:
                continue
            producer_index = graph.results[source]
            before = _remote_fraction(baseline[producer_index], caller_node)
            after = _remote_fraction(pages[producer_index], consumer_node)
            read_paths = {origin: [*path, name] for origin, path in paths[source].items()}
            reads.append(dict(value=source, paths=list(read_paths.values()),
                              caller_produced_pages=baseline[producer_index], shifted_producer_pages=pages[producer_index],
                              baseline_remote_pages=before, shifted_remote_pages=after,
                              sampled_remote_fraction_changed=_fraction_changed(before, after),
                              moved_producer_exposure_observed=any(
                                  moved_results[origin]['sampled_remote_fraction_changed'] for origin in read_paths)))
            for origin, path in read_paths.items():
                next_paths.setdefault(origin, path)
        if reads:
            consumers.append(dict(operation=name, index=index, lane='worker' if runs_on_worker else 'caller',
                                  numa_node=consumer_node, affected_reads=reads))
            if runs_on_worker:
                affected_worker.append(index)
            elif any(read['sampled_remote_fraction_changed'] and read['moved_producer_exposure_observed']
                     for read in reads):
                affected_caller.append(index)
        if runs_on_worker:
            next_paths[name] = [name]
        if next_paths:
            paths[name] = next_paths

    replaced = worker_indices | set(affected_caller)
    return dict(window_starts=starts, worker_operation_indices=sorted(worker_indices),
                affected_caller_operation_indices=affected_caller, affected_worker_operation_indices=affected_worker,
                worker_results=moved_results, consumer_paths=consumers,
                replacement_operation_samples_ns=_select_timing_columns(timings, replaced))


def model_aten_joint_numa_locality(locality, recipe, schedule, *, caller_numa_node, worker_numa_node,
                                   warmup_pairs, result_storage_bytes=None):
    """Return context-specific service samples linked to their observed Tensor placement.

    Workers use service samples from the precise selected offload set. Caller
    replacements require a dependency on a moved result and an observed remote
    exposure change. Joint traversals price operation service outside concurrency.
    """
    if (type(locality) is not dict or type(locality.get('schema_version')) is not int
            or locality['schema_version'] != 2):
        raise ValueError('joint NUMA locality evidence version is unsupported')
    if any(node is not None and not _nonnegative_integer(node) for node in (caller_numa_node, worker_numa_node)):
        raise ValueError('joint NUMA locality requires target-derived CPU node IDs')

    operations, inputs = recipe['operations'], recipe['inputs']
    expected = _joint_windows(schedule, len(inputs), len(operations))
    status = locality.get('status')
    if status == _JOINT_BUDGET_FRONTIER:
        if locality.keys() != {'schema_version', 'status'} or expected is not None:
            raise ValueError('joint NUMA budget frontier differs from the legal window set')
        return dict(status=status, sampled_node_residency_observed=False, contexts=[])
    if (status != 'MEASURED_JOINT'
            or locality.keys() != {'schema_version', 'status', 'input_pages', 'caller_output_pages', 'contexts'}
            or expected is None or type(locality['contexts']) is not list or len(locality['contexts']) != len(expected)):
        raise ValueError('joint NUMA observations require the complete bounded legal window set')
    if type(warmup_pairs) is not int or warmup_pairs < 1:
        raise ValueError('joint NUMA context timing requires a positive calibration row count')

    input_pages, baseline, sizes = _baseline_page_samples(locality, recipe, result_storage_bytes,
        size_error='joint NUMA results require one storage size per operation')
    graph = _tensor_reads(recipe)
    page_samples, contexts = [*input_pages, *baseline], []
    for starts, raw in zip(expected, locality['contexts'], strict=True):
        pages, timings = _context_samples(raw, starts, sizes, warmup_pairs)
        page_samples.extend(pages)
        contexts.append(_joint_read_context(graph, raw['window_starts'], baseline, pages, timings,
                                      caller_numa_node, worker_numa_node))
    _validate_page_set(page_samples, size_error='joint NUMA samples disagree on the kernel page size',
        bounds_error='joint NUMA samples differ from the bounded per-storage page inventory')

    input_exposure = _input_exposure(inputs, input_pages, caller_numa_node, worker_numa_node)
    return dict(status=status, cpu_ids=schedule['cpus'], caller_numa_node=caller_numa_node,
                worker_numa_node=worker_numa_node, input_exposure=input_exposure, contexts=contexts,
                sampled_node_residency_observed=any(sample['node_pages'] for sample in page_samples))


def _legal_contexts(schedule, recipe):
    operations, input_count = recipe['operations'], len(recipe['inputs'])
    operation_count = len(operations)
    widths = schedule['legal_window_widths']
    if (type(widths) is not list or len(widths) != operation_count
            or any(type(width) is not int or not 1 <= width <= min(3, operation_count - start)
                   for start, width in enumerate(widths))
            or (schedule.get('schema_version') != 2 and 3 not in widths)):
        raise ValueError('team NUMA observations require legal two- and three-lane windows')

    for start, width in enumerate(widths):
        if width == 1:
            continue
        group = operations[start:start + width]
        produced = {operation['id'] for operation in group}
        if any(arg.get('value') in produced for operation in group for arg in operation['arguments']):
            raise ValueError('team NUMA windows cannot contain an intra-window Tensor read')

    contexts = []

    def extend(offset, prefix):
        for start in range(offset, operation_count):
            for width in range(2, widths[start] + 1):
                selection = (*prefix, (start, width))
                contexts.append(selection)
                if (len(contexts) > _MAX_CONTEXTS
                        or input_count + (1 + len(contexts)) * operation_count > _MAX_SAMPLE_STORAGES):
                    return False
                if not extend(start + width, selection):
                    return False
        return True

    return contexts if extend(0, ()) else None


def _execution_nodes(schedule):
    cpus, lanes = schedule['cpus'], schedule['lanes']
    if (type(cpus) is not list or len(cpus) != 3 or not all(map(_nonnegative_integer, cpus))
            or len(set(cpus)) != 3 or type(lanes) is not list or len(lanes) != 3
            or any(type(lane) is not dict or type(lane.get('cpu_id')) is not int or lane['cpu_id'] != cpu
                   or not _nonnegative_integer(lane.get('numa_node')) for cpu, lane in zip(cpus, lanes, strict=True))):
        raise ValueError('team NUMA observations require three ordered lanes with recorded nodes')
    return [lane['numa_node'] for lane in lanes]


def _context_observations(raw, expected, sizes, warmup):
    if type(raw) is not dict or raw.keys() != {'windows', 'output_pages', 'operation_samples_ns'}:
        raise ValueError('team NUMA context requires output page observations and operation samples')
    windows = raw['windows']
    if (type(windows) is not list
            or any(type(window) is not list or len(window) != 2 or any(type(v) is not int for v in window)
                   for window in windows)
            or tuple(map(tuple, windows)) != expected):
        raise ValueError('team NUMA evidence differs from the complete ordered context set')
    timings = raw['operation_samples_ns']
    _validate_service_times(timings, warmup, len(sizes), upper_limit=2**63,
        error='team NUMA context requires one positive service time per operation and row')
    return _ordered_samples(raw['output_pages'], sizes), timings


def _node_distribution_changed(before, after):
    if before['status'] != 'OBSERVED' or after['status'] != 'OBSERVED':
        return False
    left, right = before['node_pages'], after['node_pages']
    return any(left.get(node, 0) * after['sampled_pages'] != right.get(node, 0) * before['sampled_pages']
               for node in left.keys() | right.keys())


def _team_read_context(graph, windows, input_pages, baseline, context, timings, cpus, nodes):
    operations = graph.operations
    caller_node = nodes[0]
    lanes = [0] * len(operations)
    for start, width in windows:
        lanes[start:start + width] = range(width)
    worker_indices = [index for index, lane in enumerate(lanes) if lane]
    worker_results = {}
    for index in worker_indices:
        before = _remote_fraction(baseline[index], caller_node)
        after = _remote_fraction(context[index], caller_node)
        lane = lanes[index]
        worker_results[operations[index]['id']] = dict(
            operation_index=index, lane=lane, cpu_id=cpus[lane], numa_node=nodes[lane],
            caller_produced_pages=baseline[index], shifted_producer_pages=context[index],
            baseline_remote_pages=before, shifted_remote_pages=after,
            sampled_remote_fraction_changed=_fraction_changed(before, after),
            sampled_node_distribution_changed=_node_distribution_changed(baseline[index], context[index]),
        )

    inputs, results = graph.inputs, graph.results
    paths, consumers = {}, []
    changed_callers, linked_workers = [], []
    for index, operation in enumerate(operations):
        name = operation['id']
        lane = lanes[index]
        next_paths, reads = {}, []
        for value in graph.operands[index]:
            read_paths = {origin: [*path, name] for origin, path in paths.get(value, {}).items()}
            if not lane and not read_paths:
                continue
            if value in inputs:
                source = dict(kind='GRAPH_INPUT', input_index=inputs[value])
                before_page = after_page = input_pages[inputs[value]]
            elif value in results and results[value] < index:
                producer = results[value]
                source = dict(kind='OPERATION_RESULT', operation_index=producer, lane=lanes[producer])
                before_page, after_page = baseline[producer], context[producer]
            else:
                raise ValueError('team NUMA operation reads an unordered or missing Tensor result')
            before = _remote_fraction(before_page, caller_node)
            after = _remote_fraction(after_page, nodes[lane])
            changed_origin = any(worker_results[origin]['sampled_node_distribution_changed'] for origin in read_paths)
            reads.append(dict(value=value, producer=source, paths=list(read_paths.values()),
                              baseline_input_pages=before_page, context_input_pages=after_page,
                              baseline_remote_pages=before, context_remote_pages=after,
                              sampled_remote_fraction_changed=_fraction_changed(before, after),
                              sampled_node_distribution_changed=_node_distribution_changed(before_page, after_page),
                              moved_producer_exposure_observed=changed_origin))
            for origin, path in read_paths.items():
                next_paths.setdefault(origin, path)
        if reads:
            consumers.append(dict(operation=name, operation_index=index, lane=lane, cpu_id=cpus[lane],
                                  numa_node=nodes[lane], affected_reads=reads))
        if lane:
            if next_paths:
                linked_workers.append(index)
            next_paths[name] = [name]
        elif any(read['sampled_node_distribution_changed'] and read['moved_producer_exposure_observed'] for read in reads):
            changed_callers.append(index)
        if next_paths:
            paths[name] = next_paths

    replaced = set(worker_indices) | set(changed_callers)
    complete = all(sample['status'] in ('OBSERVED', 'EMPTY') for sample in (*input_pages, *baseline, *context))
    return dict(windows=windows, complete_page_evidence=complete, worker_operation_indices=worker_indices,
                affected_worker_operation_indices=linked_workers, affected_caller_operation_indices=changed_callers,
                worker_results=worker_results, consumer_paths=consumers,
                replacement_operation_samples_ns=_select_timing_columns(timings, replaced))


def model_aten_team_numa_locality(locality, recipe, schedule, warmup, *, result_storage_bytes=None):
    """Price each occupied lane in its measured placement and dataflow context."""
    if locality is None:
        return None
    if (type(locality) is not dict or type(locality.get('schema_version')) is not int
            or locality['schema_version'] != 1):
        raise ValueError('team NUMA locality evidence version is unsupported')

    nodes = _execution_nodes(schedule)
    inputs, operations = recipe['inputs'], recipe['operations']
    tensor_releases(recipe)
    expected = _legal_contexts(schedule, recipe)
    status = locality.get('status')
    if status == _TEAM_BUDGET_FRONTIER:
        if locality.keys() != {'schema_version', 'status'} or expected is not None:
            raise ValueError('team NUMA context budget frontier differs from legal windows')
        return dict(status=status, execution_nodes=nodes, contexts=[], sampled_node_residency_observed=False)
    if (status != 'MEASURED'
            or locality.keys() != {'schema_version', 'status', 'input_pages', 'caller_output_pages', 'contexts'}
            or expected is None or type(locality['contexts']) is not list or len(locality['contexts']) != len(expected)):
        raise ValueError('team NUMA evidence requires the complete bounded legal context set')
    if type(warmup) is not int or not 1 <= warmup <= 100:
        raise ValueError('team NUMA timing requires a positive bounded calibration row count')

    input_pages, baseline, sizes = _baseline_page_samples(locality, recipe, result_storage_bytes,
        size_error='team NUMA requires one result storage size per operation')
    graph = _tensor_reads(recipe)
    page_samples, contexts = [*input_pages, *baseline], []
    for selection, raw in zip(expected, locality['contexts'], strict=True):
        pages, timings = _context_observations(raw, selection, sizes, warmup)
        page_samples.extend(pages)
        contexts.append(_team_read_context(graph, raw['windows'], input_pages, baseline, pages, timings,
                                      schedule['cpus'], nodes))
    _validate_page_set(page_samples, size_error='team NUMA samples disagree on the kernel page size',
        bounds_error='team NUMA samples disagree with the bounded per-storage page inventory')
    return dict(status=status, execution_nodes=nodes, contexts=contexts,
                input_pages={item['id']: sample for item, sample in zip(inputs, input_pages, strict=True)},
                sampled_node_residency_observed=any(page['node_pages'] for page in page_samples))


def _node_id(node):
    return _nonnegative_integer(node) and node < 2**31


def _read_edges(recipe, schedule, lanes, nodes, input_pages, output_pages):
    inputs, operations = recipe['inputs'], recipe['operations']
    sources = {item['id']: dict(kind='GRAPH_INPUT', input_index=index) for index, item in enumerate(inputs)}

    def placement(index):
        lane = lanes[index]
        return dict(operation_index=index, lane=lane, cpu_id=schedule['cpus'][lane], numa_node=nodes[lane])

    pages_by_value = input_pages | output_pages
    reads = []
    for index, operation in enumerate(operations):
        consumer = placement(index)
        for value, operand_indices in _tensor_operands(operation, with_positions=True).items():
            pages = pages_by_value[value]
            reads.append(dict(value=value, operand_indices=operand_indices, producer=sources[value],
                              consumer=consumer, sampled_remote_pages=_remote_fraction(pages, consumer['numa_node']),
                              sampled_page_nodes=sorted(map(int, pages['node_pages']))))
        sources[operation['id']] = dict(kind='OPERATION_RESULT', **consumer)
    return reads


def model_ready_numa_locality(locality, recipe, schedule, lanes, warmup, *, result_storage_bytes):
    """Validate one fixed-lane provider replay and attach its measured locations to DATA edges.

    Page fractions describe the sampled allocation and never price transfers.
    Operation times represent this source-ordered replay on the assigned lanes.
    """
    if (type(locality) is not dict or locality.keys() != _READY_FIELDS
            or type(locality['schema_version']) is not int or locality['schema_version'] != 1
            or locality['scope'] != _READY_SCOPE):
        raise ValueError('ready DAG locality requires a supported fixed-lane replay')
    raw_lanes = locality['lanes']
    if (type(raw_lanes) is not list or any(type(lane) is not int or not 0 <= lane < 3 for lane in raw_lanes)
            or raw_lanes != lanes):
        raise ValueError('ready DAG locality lane assignment differs from its nominated schedule')
    nodes = _execution_nodes(schedule)
    host_nodes = schedule.get('host_numa_node_ids')
    if (type(host_nodes) is not list or not all(map(_node_id, host_nodes))
            or host_nodes != sorted(set(host_nodes)) or not set(nodes).issubset(host_nodes)):
        raise ValueError('ready DAG locality requires recorded host NUMA nodes for every CPU lane')

    storage_bytes = locality['result_storage_bytes']
    if (type(storage_bytes) is not list or any(type(size) is not int or size < 0 for size in storage_bytes)
            or storage_bytes != result_storage_bytes):
        raise ValueError('ready DAG locality result geometry differs from its isolated calibration')
    inputs, operations = recipe['inputs'], recipe['operations']
    input_samples = _ordered_samples(locality['input_pages'], [4 * prod(item['shape']) for item in inputs])
    output_samples = _ordered_samples(locality['output_pages'], storage_bytes)
    samples = (*input_samples, *output_samples)
    _validate_page_set(samples, size_error='ready DAG locality page samples disagree on the kernel page size',
        bounds_error='ready DAG locality page samples disagree with the bounded per-storage inventory')
    recorded = set(host_nodes)
    if any(any(int(node) not in recorded for node in sample['node_pages']) for sample in samples):
        raise ValueError('ready DAG locality page sample names an unrecorded host NUMA node')

    timings = locality['operation_samples_ns']
    _validate_samples(timings, warmup, len(operations), set(range(len(operations))))
    complete = all(sample['status'] in ('OBSERVED', 'EMPTY') for sample in samples)
    input_pages = {item['id']: sample for item, sample in zip(inputs, input_samples, strict=True)}
    output_pages = {op['id']: sample for op, sample in zip(operations, output_samples, strict=True)}
    return dict(status='MEASURED' if complete else 'INCOMPLETE_PAGE_EVIDENCE', scope=_READY_SCOPE,
                lanes=lanes, execution_nodes=nodes, input_pages=input_pages, output_pages=output_pages,
                reads=_read_edges(recipe, schedule, lanes, nodes, input_pages, output_pages),
                operation_samples_ns=timings)
