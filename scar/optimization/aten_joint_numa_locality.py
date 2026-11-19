"""Bind measured joint NUMA contexts to legal CPU windows and Tensor dataflow."""

from itertools import combinations
from math import prod

from scar.optimization.aten_numa_locality import (
    _fraction_changed,
    _nonnegative_integer,
    _ordered_samples,
    _remote_fraction,
)


_MAX_CONTEXTS = 8
_MAX_SAMPLE_STORAGES = 256
_MAX_PAGES_PER_STORAGE = 32
_BUDGET_FRONTIER = 'JOINT_CONTEXT_BUDGET_EXCEEDED'


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
    if (type(timings) is not list or len(timings) != warmup_pairs or any(
            type(row) is not list or len(row) != len(sizes)
            or any(type(value) is not int or value < 1 for value in row) for row in timings)):
        raise ValueError('joint NUMA timings require a positive service sample per operation and calibration row')
    return output_pages, timings


def _read_context(operations, starts, baseline, pages, timings, caller_node, worker_node):
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
    results = {op['id']: index for index, op in enumerate(operations)}
    for index, operation in enumerate(operations):
        name = operation['id']
        runs_on_worker = index in worker_indices
        consumer_node = worker_node if runs_on_worker else caller_node
        next_paths, reads = {}, []
        for source in dict.fromkeys(arg['value'] for arg in operation['arguments'] if 'value' in arg):
            if source not in paths:
                continue
            producer_index = results[source]
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
                replacement_operation_samples_ns=[[value if index in replaced else None
                                                   for index, value in enumerate(row)] for row in timings])


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
    if status == _BUDGET_FRONTIER:
        if locality.keys() != {'schema_version', 'status'} or expected is not None:
            raise ValueError('joint NUMA budget frontier differs from the legal window set')
        return dict(status=status, sampled_node_residency_observed=False, contexts=[])
    if (status != 'MEASURED_JOINT'
            or locality.keys() != {'schema_version', 'status', 'input_pages', 'caller_output_pages', 'contexts'}
            or expected is None or type(locality['contexts']) is not list or len(locality['contexts']) != len(expected)):
        raise ValueError('joint NUMA observations require the complete bounded legal window set')
    if type(warmup_pairs) is not int or warmup_pairs < 1:
        raise ValueError('joint NUMA context timing requires a positive calibration row count')

    input_pages = _ordered_samples(locality['input_pages'], [4 * prod(item['shape']) for item in inputs])
    sizes = [None] * len(operations) if result_storage_bytes is None else result_storage_bytes
    if len(sizes) != len(operations):
        raise ValueError('joint NUMA results require one storage size per operation')
    baseline = _ordered_samples(locality['caller_output_pages'], sizes)
    sizes = [sample['storage_bytes'] for sample in baseline]
    page_samples, contexts = [*input_pages, *baseline], []
    for starts, raw in zip(expected, locality['contexts'], strict=True):
        pages, timings = _context_samples(raw, starts, sizes, warmup_pairs)
        page_samples.extend(pages)
        contexts.append(_read_context(operations, raw['window_starts'], baseline, pages, timings,
                                      caller_numa_node, worker_numa_node))
    if any(sample['sampled_pages'] != min(sample['storage_pages'], _MAX_PAGES_PER_STORAGE)
           for sample in page_samples):
        raise ValueError('joint NUMA samples differ from the bounded per-storage page inventory')
    if len({sample['page_size_bytes'] for sample in page_samples} - {0}) > 1:
        raise ValueError('joint NUMA samples disagree on the kernel page size')

    input_exposure = [dict(value=item['id'], pages=pages, caller_remote_pages=_remote_fraction(pages, caller_numa_node),
                           worker_remote_pages=_remote_fraction(pages, worker_numa_node))
                      for item, pages in zip(inputs, input_pages, strict=True)]
    return dict(status=status, cpu_ids=schedule['cpus'], caller_numa_node=caller_numa_node,
                worker_numa_node=worker_numa_node, input_exposure=input_exposure, contexts=contexts,
                sampled_node_residency_observed=any(sample['node_pages'] for sample in page_samples))
