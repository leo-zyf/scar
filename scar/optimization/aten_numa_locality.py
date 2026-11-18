"""Bind measured CPU Tensor page samples to native graph data dependencies."""

from math import prod


_FRONTIERS = ('REQUIRES_SINGLE_LEGAL_WINDOW', 'GRAPH_EXCEEDS_LOCALITY_SAMPLE_BUDGET')
_PAGE_KEYS = {'storage_bytes', 'page_size_bytes', 'storage_pages', 'sampled_pages',
              'node_pages', 'unknown_pages', 'query_errno', 'status'}
_MEASUREMENT_KEYS = {'schema_version', 'status', 'window_start', 'input_pages', 'caller_output_pages',
                     'worker_output_pages', 'worker_output_consumer_samples_ns'}


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


def _remote_fraction(sample, caller_numa_node):
    if sample['status'] != 'OBSERVED' or caller_numa_node is None:
        return None
    total = sample['sampled_pages']
    return [total - sample['node_pages'].get(str(caller_numa_node), 0), total]


def _fraction_changed(before, after):
    return before is not None and after is not None and before[0] * after[1] != after[0] * before[1]


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

    inputs = _ordered_samples(locality['input_pages'], [4 * prod(item['shape']) for item in recipe['inputs']])
    sizes = result_storage_bytes if result_storage_bytes is not None else [None] * len(operations)
    if len(sizes) != len(operations):
        raise ValueError('NUMA results require one storage size per operation')
    baseline = _ordered_samples(locality['caller_output_pages'], sizes)
    shifted = _ordered_samples(locality['worker_output_pages'], [sample['storage_bytes'] for sample in baseline])
    if len({row['page_size_bytes'] for row in (*inputs, *baseline, *shifted)} - {0}) > 1:
        raise ValueError('NUMA samples from one calibration disagree on the kernel page size')

    origin_before = _remote_fraction(baseline[producer], caller_numa_node)
    origin_after = _remote_fraction(shifted[producer], caller_numa_node)
    origin_changed = _fraction_changed(origin_before, origin_after)
    cross_node = caller_numa_node is not None and worker_numa_node is not None and caller_numa_node != worker_numa_node
    identifier = operations[producer]['id']
    paths, affected, consumers = {identifier: [identifier]}, [], []
    results = {op['id']: i for i, op in enumerate(operations)}
    for index in range(producer + 1, len(operations)):
        operation = operations[index]
        dependencies = list(dict.fromkeys(arg['value'] for arg in operation['arguments']
                                          if 'value' in arg and arg['value'] in paths))
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

    changed_indices = set(affected)
    replacements = [[value if index in changed_indices else None for index, value in enumerate(row)]
                    for row in timings] if affected else None
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
    input_exposure = [dict(value=item['id'], pages=pages, caller_remote_pages=_remote_fraction(pages, caller_numa_node),
                           worker_remote_pages=_remote_fraction(pages, worker_numa_node))
                      for item, pages in zip(recipe['inputs'], inputs, strict=True)]
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
