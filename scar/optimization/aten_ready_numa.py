"""Associate observed NUMA pages with original ready-DAG Tensor reads."""

from math import prod

from scar.optimization.aten_native_team import _validate_samples
from scar.optimization.aten_numa_locality import _ordered_samples, _remote_fraction
from scar.optimization.aten_team_numa_locality import _MAX_PAGES_PER_STORAGE, _execution_nodes


_SCOPE = 'SOURCE_ORDER_REPLAY_ON_PLANNED_LANES'
_FIELDS = {'schema_version', 'scope', 'lanes', 'input_pages', 'output_pages',
           'operation_samples_ns', 'result_storage_bytes'}


def _node_id(node):
    return type(node) is int and 0 <= node < 2**31


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
        operands = {}
        for operand_index, argument in enumerate(operation['arguments']):
            if 'value' in argument:
                operands.setdefault(argument['value'], []).append(operand_index)
        for value, operand_indices in operands.items():
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
    if (type(locality) is not dict or locality.keys() != _FIELDS
            or type(locality['schema_version']) is not int or locality['schema_version'] != 1
            or locality['scope'] != _SCOPE):
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
    if any(sample['sampled_pages'] != min(sample['storage_pages'], _MAX_PAGES_PER_STORAGE) for sample in samples):
        raise ValueError('ready DAG locality page samples disagree with the bounded per-storage inventory')
    if len({sample['page_size_bytes'] for sample in samples} - {0}) > 1:
        raise ValueError('ready DAG locality page samples disagree on the kernel page size')
    recorded = set(host_nodes)
    if any(any(int(node) not in recorded for node in sample['node_pages']) for sample in samples):
        raise ValueError('ready DAG locality page sample names an unrecorded host NUMA node')

    timings = locality['operation_samples_ns']
    _validate_samples(timings, warmup, len(operations), set(range(len(operations))))
    complete = all(sample['status'] in ('OBSERVED', 'EMPTY') for sample in samples)
    input_pages = {item['id']: sample for item, sample in zip(inputs, input_samples, strict=True)}
    output_pages = {op['id']: sample for op, sample in zip(operations, output_samples, strict=True)}
    return dict(status='MEASURED' if complete else 'INCOMPLETE_PAGE_EVIDENCE', scope=_SCOPE,
                lanes=lanes, execution_nodes=nodes, input_pages=input_pages, output_pages=output_pages,
                reads=_read_edges(recipe, schedule, lanes, nodes, input_pages, output_pages),
                operation_samples_ns=timings)
