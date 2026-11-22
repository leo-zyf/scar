"""Bind calibrated team placements to observed Tensor pages and graph reads."""

from math import prod

from scar.optimization.aten_numa_locality import (
    _fraction_changed,
    _nonnegative_integer,
    _ordered_samples,
    _remote_fraction,
)
from scar.optimization.aten_live_values import tensor_releases


_MAX_CONTEXTS = 8
_MAX_SAMPLE_STORAGES = 256
_MAX_PAGES_PER_STORAGE = 32
_BUDGET_FRONTIER = 'TEAM_CONTEXT_BUDGET_EXCEEDED'


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
    if (type(timings) is not list or len(timings) != warmup
            or any(type(row) is not list or len(row) != len(sizes)
                   or any(type(duration) is not int or not 0 < duration < 2**63 for duration in row)
                   for row in timings)):
        raise ValueError('team NUMA context requires one positive service time per operation and row')
    return _ordered_samples(raw['output_pages'], sizes), timings


def _node_distribution_changed(before, after):
    if before['status'] != 'OBSERVED' or after['status'] != 'OBSERVED':
        return False
    left, right = before['node_pages'], after['node_pages']
    return any(left.get(node, 0) * after['sampled_pages'] != right.get(node, 0) * before['sampled_pages']
               for node in left.keys() | right.keys())


def _read_context(recipe, windows, input_pages, baseline, context, timings, cpus, nodes):
    operations = recipe['operations']
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

    inputs = {item['id']: i for i, item in enumerate(recipe['inputs'])}
    results = {item['id']: i for i, item in enumerate(operations)}
    paths, consumers = {}, []
    changed_callers, linked_workers = [], []
    for index, operation in enumerate(operations):
        name = operation['id']
        lane = lanes[index]
        next_paths, reads = {}, []
        for value in dict.fromkeys(arg['value'] for arg in operation['arguments'] if 'value' in arg):
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
                replacement_operation_samples_ns=[[duration if index in replaced else None
                                                   for index, duration in enumerate(row)] for row in timings])


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
    if status == _BUDGET_FRONTIER:
        if locality.keys() != {'schema_version', 'status'} or expected is not None:
            raise ValueError('team NUMA context budget frontier differs from legal windows')
        return dict(status=status, execution_nodes=nodes, contexts=[], sampled_node_residency_observed=False)
    if (status != 'MEASURED'
            or locality.keys() != {'schema_version', 'status', 'input_pages', 'caller_output_pages', 'contexts'}
            or expected is None or type(locality['contexts']) is not list or len(locality['contexts']) != len(expected)):
        raise ValueError('team NUMA evidence requires the complete bounded legal context set')
    if type(warmup) is not int or not 1 <= warmup <= 100:
        raise ValueError('team NUMA timing requires a positive bounded calibration row count')

    input_pages = _ordered_samples(locality['input_pages'], [4 * prod(item['shape']) for item in inputs])
    sizes = [None] * len(operations) if result_storage_bytes is None else result_storage_bytes
    if len(sizes) != len(operations):
        raise ValueError('team NUMA requires one result storage size per operation')
    baseline = _ordered_samples(locality['caller_output_pages'], sizes)
    sizes = [page['storage_bytes'] for page in baseline]
    page_samples, contexts = [*input_pages, *baseline], []
    for selection, raw in zip(expected, locality['contexts'], strict=True):
        pages, timings = _context_observations(raw, selection, sizes, warmup)
        page_samples.extend(pages)
        contexts.append(_read_context(recipe, raw['windows'], input_pages, baseline, pages, timings,
                                      schedule['cpus'], nodes))
    if any(page['sampled_pages'] != min(page['storage_pages'], _MAX_PAGES_PER_STORAGE) for page in page_samples):
        raise ValueError('team NUMA samples disagree with the bounded per-storage page inventory')
    if len({page['page_size_bytes'] for page in page_samples} - {0}) > 1:
        raise ValueError('team NUMA samples disagree on the kernel page size')
    return dict(status=status, execution_nodes=nodes, contexts=contexts,
                input_pages={item['id']: sample for item, sample in zip(inputs, input_pages, strict=True)},
                sampled_node_residency_observed=any(page['node_pages'] for page in page_samples))
