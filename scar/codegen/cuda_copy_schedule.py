"""Lower verified single-device CUDA regions into the C++20 queue executor."""

from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush

from scar.optimization.aten_export_rewrite import FUNCTIONAL_EXPORT_SCHEMAS
from scar.optimization.joint_planner import verify_execution_plan


_DIRECTIONS = {
    ('PinnedHost', 'Device'): 'cudaMemcpyHostToDevice',
    ('Device', 'PinnedHost'): 'cudaMemcpyDeviceToHost',
    ('Device', 'Device'): 'cudaMemcpyDeviceToDevice',
}
_SGEMM = 'CUBLAS_SGEMM_F32_COLUMN_MAJOR_V1'
_HOST_ATEN = 'LIBTORCH_ATEN_CPU_OUT_F32_V1'
_RELU = FUNCTIONAL_EXPORT_SCHEMAS['aten.relu.default']
_ADD = FUNCTIONAL_EXPORT_SCHEMAS['aten.add.Tensor']


@dataclass(frozen=True, slots=True)
class CudaCopySource:
    header: str
    buffer_values: tuple[str, ...]
    stream_channels: tuple[str, ...]
    activity_operations: tuple[str, ...]
    activity_streams: tuple[str, ...]
    activity_waits: tuple[tuple[int, ...], ...]


def _completion_waits(copies, selected, streams, predecessors):
    """Enforce model resource non-overlap despite varying physical durations."""
    by_resource = defaultdict(set)
    for index, (_, mode) in enumerate(copies):
        for resource, demand in mode.demands.items():
            if demand:
                by_resource[resource].add(streams[index])
    shared = {resource for resource, members in by_resource.items() if len(members) > 1}
    pending = {resource: [] for resource in shared}
    finished = {resource: {} for resource in shared}
    positions = {op.id: index for index, (op, _) in enumerate(copies)}
    waited = defaultdict(dict)
    result = []
    for index, (op, mode) in enumerate(copies):
        row = selected[op.id]
        stream = streams[index]
        required = {}

        def add(prior):
            prior_stream = streams[prior]
            if prior_stream != stream:
                required[prior_stream] = max(prior, required.get(prior_stream, -1))

        for name in predecessors[op.id]:
            prior = positions.get(name)
            if prior is None or prior >= index:
                raise ValueError('a CUDA queue dependency is outside its preceding copy region')
            add(prior)
        for resource, units in mode.demands.items():
            if not units or resource not in shared:
                continue
            queue, ready = pending[resource], finished[resource]
            while queue and queue[0][0] <= row['start_ns']:
                _, prior = heappop(queue)
                source_stream = streams[prior]
                ready[source_stream] = max(prior, ready.get(source_stream, -1))
            for prior in ready.values():
                add(prior)
            heappush(queue, (row['end_ns'], index))
        new_waits = {source: prior for source, prior in required.items()
                     if prior > waited[stream].get(source, -1)}
        waited[stream].update(new_waits)
        result.append(tuple(sorted(new_waits.values())))
    return result


def render_cuda_copy_schedule(problem, schedule, *, device, memory_kinds, stream_channels):
    """Emit a copy region from an already qualified provider/resource binding.

    ``memory_kinds`` binds planning pools to ``PinnedHost`` or ``Device`` for
    one CUDA device. ``stream_channels`` names exclusive FIFO copy channels.
    Physical placement at this boundary is explicit, and every material slot
    remains allocated through the host return. That full backing bound is
    checked in addition to the planning certificate's interval leases.
    """
    return _render(problem, schedule, device, memory_kinds, stream_channels, compute_resource=None)


def render_cuda_cublas_schedule(problem, schedule, *, device, memory_kinds, stream_channels, compute_resource):
    """Emit copies and same-stream, out-of-place FP32 cuBLAS matrix products.

    Explicit operand roles refer to PlanningDataEdge IDs, which survive route
    expansion, and the device/stream/compute binding is required from the host.
    Each GEMM stream retains a provider workspace through the region return.
    CUDA and cuBLAS context internals and setup durations are outside the model.
    """
    if not isinstance(compute_resource, str):
        raise ValueError('cuBLAS needs the bound CUDA execution resource ID')
    return _render(problem, schedule, device, memory_kinds, stream_channels, compute_resource)


def render_cuda_hybrid_schedule(problem, schedule, *, device, memory_kinds, stream_channels, compute_resource,
                                host_compute_resource):
    """Emit a cuBLAS and LibTorch schedule with dependency-local host waits."""
    if not isinstance(compute_resource, str) or not isinstance(host_compute_resource, str):
        raise ValueError('hybrid scheduling needs bound device and host execution resources')
    return _render(problem, schedule, device, memory_kinds, stream_channels, compute_resource, host_compute_resource)


def _sgemm_ports(mode, incoming, outgoing, output, destination):
    parameters = mode.parameters
    if (set(parameters) != {'left_edge', 'right_edge', 'rows', 'columns', 'reduction', 'completion'}
            or parameters['completion'] != 'SAME_STREAM'):
        raise ValueError('cuBLAS SGEMM requires typed operand roles and same-stream completion')
    rows, columns, reduction = (parameters[name] for name in ('rows', 'columns', 'reduction'))
    if any(type(extent) is not int or not 0 < extent < 2**31 for extent in (rows, columns, reduction)):
        raise ValueError('cuBLAS SGEMM dimensions must fit positive provider ABI integers')
    inputs = {edge.id: edge for edge in incoming}
    left_id, right_id = parameters['left_edge'], parameters['right_edge']
    if not isinstance(left_id, str) or not isinstance(right_id, str) or left_id == right_id or inputs.keys() != {left_id, right_id}:
        raise ValueError('cuBLAS SGEMM operand roles must match two distinct data edges')
    left, right = inputs[left_id], inputs[right_id]
    if (len({left.value, right.value, output}) != 3 or mode.input_locations != dict.fromkeys((left.value, right.value), destination)
            or (left.nbytes, right.nbytes) != (4 * rows * reduction, 4 * reduction * columns)
            or any(edge.value != output or edge.nbytes != 4 * rows * columns for edge in outgoing)):
        raise ValueError('cuBLAS SGEMM device ports have different matrix byte extents or alias')
    workspace = mode.workspace_bytes.get(destination, 0)
    if mode.workspace_bytes != {destination: workspace} or workspace < 16384:
        raise ValueError('cuBLAS SGEMM requires a qualified device workspace of at least 16 KiB')
    return left, right, (rows, columns, reduction), workspace


def _matrix_geometry(shape):
    if (type(shape) is not list or len(shape) != 2
            or any(type(extent) is not int or not 0 < extent < 2**31 for extent in shape)
            or 4 * shape[0] * shape[1] >= 2**63):
        raise ValueError('ATen host ports require bounded contiguous FP32 matrices')
    return tuple(shape)


def _host_ports(mode, incoming, outgoing, output, destination):
    parameters = mode.parameters
    if (set(parameters) != {'schema', 'arguments', 'shape', 'completion'}
            or parameters['completion'] != 'HOST_SYNCHRONOUS' or mode.workspace_bytes):
        raise ValueError('ATen host operation requires a synchronous out-port contract')
    schema = parameters['schema']
    arguments = parameters['arguments']
    if type(arguments) is not list or not (schema == _RELU and len(arguments) == 1
                                            or schema == _ADD and len(arguments) == 3):
        raise ValueError('ATen host operation has no qualified provider binding')
    count = 1 if schema == _RELU else 2
    inputs = {edge.id: edge for edge in incoming}
    ports, edge_ids = [], set()
    for arg in arguments[:count]:
        if (type(arg) is not dict or set(arg) != {'edge', 'shape'}
                or not isinstance(arg['edge'], str) or arg['edge'] not in inputs):
            raise ValueError('ATen host argument differs from its ordered input edge')
        edge = inputs[arg['edge']]
        shape = _matrix_geometry(arg['shape'])
        if edge.nbytes != 4 * shape[0] * shape[1] or edge.value == output:
            raise ValueError('ATen host input has a different extent or aliases its output')
        edge_ids.add(edge.id)
        ports.append((edge.value, *shape))
    result = _matrix_geometry(parameters['shape'])
    if (edge_ids != inputs.keys() or mode.input_locations != dict.fromkeys((row[0] for row in ports), destination)
            or any(edge.value != output or edge.nbytes != 4 * result[0] * result[1] for edge in outgoing)):
        raise ValueError('ATen host materials do not match the full input and output contract')
    alpha = None
    if schema == _RELU:
        expected = ports[0][1:]
    else:
        left, right = (port[1:] for port in ports)
        if any(a != b and 1 not in (a, b) for a, b in zip(left, right, strict=True)):
            raise ValueError('ATen add host arguments do not broadcast')
        expected = tuple(map(max, zip(left, right, strict=True)))
        scalar = arguments[-1]
        if (type(scalar) is not dict or set(scalar) != {'scalar'} or type(scalar['scalar']) is not int
                or not -(2**63) <= scalar['scalar'] < 2**63):
            raise ValueError('ATen add requires a signed 64-bit integral alpha')
        alpha = scalar['scalar']
    if result != expected:
        raise ValueError('ATen host provider result has a different projected shape')
    return ('HostRelu' if schema == _RELU else 'HostAdd'), (*ports, (output, *result)), alpha


def _render(problem, schedule, device, memory_kinds, stream_channels, compute_resource, host_compute_resource=None):
    verify_execution_plan(problem, schedule)
    if type(device) is not int or not 0 <= device < 2**31:
        raise ValueError('CUDA copy lowering requires an explicit device ordinal')
    if schedule['transfers'] or problem.activities or problem.timing_constraints:
        raise ValueError('CUDA lowering requires explicit operations with direct dependencies')
    resources = {row.id: row for row in problem.resources}
    if (not memory_kinds or len(set(memory_kinds.values())) != len(memory_kinds)
            or any(kind not in ('PinnedHost', 'Device') or name not in resources or resources[name].kind != 'MEMORY'
                   for name, kind in memory_kinds.items())):
        raise ValueError('CUDA copy pools must bind distinct pinned-host and device memory domains')
    channels = set(stream_channels)
    if not channels or any(name not in resources or resources[name].kind != 'CHANNEL' or resources[name].capacity != 1
                           for name in channels):
        raise ValueError('CUDA streams require exclusive planning channels')
    if compute_resource is not None and (compute_resource not in resources or resources[compute_resource].kind != 'EXECUTION'):
        raise ValueError('cuBLAS needs one bound CUDA execution resource')
    if host_compute_resource is not None and (host_compute_resource == compute_resource
            or host_compute_resource not in resources or resources[host_compute_resource].kind != 'EXECUTION'
            or resources[host_compute_resource].capacity != 1):
        raise ValueError('ATen hybrid host requires one exclusive synchronous execution resource')

    selected = {row['operation']: row for row in schedule['selected']}
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    for edge in problem.data_edges:
        incoming[edge.consumer].append(edge)
        outgoing[edge.producer].append(edge)
    operations, returns, stream_keys, operands, slots, workspaces = [], [], {}, {}, {}, {}
    predecessors = {}

    def material(value, pool, size):
        if pool not in memory_kinds or not 0 < size < 2**63:
            raise ValueError('CUDA copy material needs a bound pool and a positive byte extent')
        if slots.setdefault(value, (pool, size)) != (pool, size):
            raise ValueError('CUDA copy material has inconsistent placement or size')

    for op in problem.operations:
        row = selected[op.id]
        mode = next(item for item in op.options if item.id == row['option'])
        predecessors[op.id] = set(op.predecessors) | {e.producer for e in incoming[op.id] if e.producer is not None}
        if mode.lowering == 'CUDA_COPY_RETURN':
            if (outgoing[op.id] or mode.output_locations or mode.demands or mode.workspace_bytes or mode.parameters
                    or mode.cost.estimate_ns != 0 or not incoming[op.id]
                    or any(edge.producer is None for edge in incoming[op.id])
                    or any(memory_kinds.get(pool) != 'PinnedHost' for pool in mode.input_locations.values())
                    or mode.input_locations.keys() != {e.value for e in incoming[op.id]}):
                raise ValueError('CUDA copy return must be a zero-work terminal pinned-host observation')
            returns.append(op.id)
            continue
        is_compute = mode.lowering == _SGEMM and compute_resource is not None
        is_host = mode.lowering == _HOST_ATEN and host_compute_resource is not None
        if ((mode.lowering != 'CUDA_MEMCPY_ASYNC' and not (is_compute or is_host)) or mode.cost.estimate_ns <= 0
                or not outgoing[op.id] or len(mode.output_locations) != 1):
            raise ValueError('CUDA operation requires positive service and a single explicit output material')
        (output, destination), = mode.output_locations.items()
        if is_host:
            if mode.demands != {host_compute_resource: 1} or memory_kinds.get(destination) != 'PinnedHost':
                raise ValueError('ATen host operation requires the host lane and pinned-host output')
            operation, ports, alpha = _host_ports(mode, incoming[op.id], outgoing[op.id], output, destination)
            for name, rows, columns in ports:
                material(name, destination, 4 * rows * columns)
            stream_keys[op.id] = host_compute_resource
            operands[op.id] = operation, ports, alpha
            operations.append((op, mode))
            continue
        active = {name for name, units in mode.demands.items() if units}
        own = active & channels
        required = own | {compute_resource} if is_compute else own
        if (len(own) != 1 or not required <= active
                or any(resources[name].kind != 'BANDWIDTH_CLASS' for name in active - required)):
            raise ValueError('CUDA operations need one selected stream and their qualified device demands')
        stream_keys[op.id], = own
        if is_compute:
            if memory_kinds.get(destination) != 'Device':
                raise ValueError('cuBLAS SGEMM requires bound device memory')
            left, right, shape, workspace = _sgemm_ports(mode, incoming[op.id], outgoing[op.id], output, destination)
            for edge in (left, right):
                material(edge.value, destination, edge.nbytes)
            material(output, destination, outgoing[op.id][0].nbytes)
            stream = stream_keys[op.id]
            workspaces[stream] = max(workspace, workspaces.get(stream, 0))
            operands[op.id] = (left.value, right.value, output), shape
        else:
            if mode.workspace_bytes or len(incoming[op.id]) != 1:
                raise ValueError('CUDA copy operation requires a single explicit byte-preserving transfer')
            edge, = incoming[op.id]
            source = mode.input_locations[edge.value]
            direction = _DIRECTIONS.get((memory_kinds.get(source), memory_kinds.get(destination)))
            if (mode.input_locations.keys() != {edge.value} or not direction or output == edge.value
                    or any(item.value != output or item.nbytes != edge.nbytes for item in outgoing[op.id])
                    or set(mode.parameters) != {'logical_value', 'source_memory', 'destination_memory', 'completion'}
                    or type(mode.parameters['logical_value']) is not str or not mode.parameters['logical_value']
                    or mode.parameters['completion'] != 'DESTINATION_READY'
                    or mode.parameters['source_memory'] != source
                    or mode.parameters['destination_memory'] != destination):
                raise ValueError('CUDA copy byte extent, direction, ports or completion contract differs')
            material(edge.value, source, edge.nbytes)
            material(output, destination, edge.nbytes)
            operands[op.id] = (edge.value, output), (direction,)
        operations.append((op, mode))

    if not operations or len(returns) != 1 or any(returns[0] in predecessors[op.id] for op, _ in operations):
        raise ValueError('CUDA lowering requires a nonempty region and one terminal host return')
    if compute_resource is not None and not workspaces:
        raise ValueError('cuBLAS lowering requires at least one selected SGEMM operation')
    occupied = dict(problem.reserved_memory)
    for pool, size in slots.values():
        occupied[pool] = occupied.get(pool, 0) + size
    if workspaces:
        pool, = (name for name, kind in memory_kinds.items() if kind == 'Device')
        occupied[pool] += sum(workspaces.values())
    if any(size > resources[pool].capacity for pool, size in occupied.items()):
        raise ValueError('CUDA buffers and workspaces cannot all remain allocated within the verified budget')

    operations.sort(key=lambda item: (selected[item[0].id]['start_ns'], item[0].id))
    used_streams = tuple(sorted(set(stream_keys.values()) - {host_compute_resource}))
    stream_indices = {key: index for index, key in enumerate(used_streams)}
    host_index = len(used_streams)
    if host_compute_resource is not None:
        stream_indices[host_compute_resource] = host_index
    streams = [stream_indices[stream_keys[op.id]] for op, _ in operations]
    buffers = tuple(sorted(slots))
    buffer_indices = {key: index for index, key in enumerate(buffers)}
    waits = _completion_waits(operations, selected, streams, predecessors)
    if host_compute_resource is not None:
        # Synchronous caller work has completed before any later GPU submission.
        waits = [tuple(prior for prior in row if streams[prior] != host_index) for row in waits]
    recorded = {prior for row in waits for prior in row}

    provider = 'hybrid' if host_compute_resource is not None else ('cublas' if workspaces else 'copy')
    lines = ['#pragma once', '', f'#include "cuda_{provider}_schedule_v1.h"', '#include <array>', '',
             'namespace scar_cuda_generated {', f'using namespace scar::cuda_{provider}_v1;', '',
             f'inline constexpr int device = {device};',
             f'inline constexpr std::array<BufferSpec, {len(buffers)}> buffers{{{{']
    for value in buffers:
        pool, size = slots[value]
        lines.append(f'    {{Memory::{memory_kinds[pool]}, {size}ULL}},')
    lines.extend(['}};', ''])
    if workspaces:
        sizes = ', '.join(f'{workspaces.get(key, 0)}ULL' for key in used_streams)
        lines.append(f'inline constexpr std::array<std::size_t, {len(used_streams)}> workspace_bytes{{{sizes}}};')
    for index, row in enumerate(waits):
        if row:
            entries = ', '.join(map(str, row))
            lines.append(f'inline constexpr std::array<std::size_t, {len(row)}> wait_{index}{{{entries}}};')
    lines.append(f'inline constexpr std::array<Step, {len(operations)}> steps{{{{')
    for index, (op, mode) in enumerate(operations):
        if mode.lowering == _HOST_ATEN:
            operation, ports, alpha = operands[op.id]
            fields = [f'{{{buffer_indices[name]}, {rows}, {columns}}}' for name, rows, columns in ports]
            if alpha is not None:
                fields.append(f'{alpha}LL' if alpha != -(2**63) else '(-9223372036854775807LL - 1LL)')
            operands_cpp = operation + '{' + ', '.join(fields) + '}'
        else:
            values, parameters = operands[op.id]
            operands_cpp = ', '.join(str(item) for item in (*[buffer_indices[name] for name in values], *parameters))
            if workspaces:
                alternative = 'MatrixMultiply' if mode.lowering == _SGEMM else 'Copy'
                operands_cpp = f'{alternative}{{{operands_cpp}}}'
        dependency = f'wait_{index}' if waits[index] else '{}'
        record = 'true' if index in recorded else 'false'
        lines.append(f'    {{{streams[index]}, {operands_cpp}, {dependency}, {record}}},')
    stream_resource = 'workspace_bytes' if workspaces else len(used_streams)
    lines.extend(['}};', '', 'inline void run(std::span<Buffer* const> materializations) {',
                  f'    detail::execute(device, {stream_resource}, buffers, steps, materializations);', '}', '',
                  '}  // namespace scar_cuda_generated', ''])
    activity_operations = tuple(op.id for op, _ in operations)
    return CudaCopySource('\n'.join(lines), buffers, used_streams, activity_operations,
                          tuple(stream_keys[name] for name in activity_operations), tuple(waits))
