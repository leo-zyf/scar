"""Lower a verified, explicit single-device copy region into C++20."""

from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush

from scar.optimization.joint_planner import verify_execution_plan


_DIRECTIONS = {
    ('PinnedHost', 'Device'): 'cudaMemcpyHostToDevice',
    ('Device', 'PinnedHost'): 'cudaMemcpyDeviceToHost',
    ('Device', 'Device'): 'cudaMemcpyDeviceToDevice',
}


@dataclass(frozen=True, slots=True)
class CudaCopySource:
    header: str
    buffer_values: tuple[str, ...]
    stream_channels: tuple[str, ...]


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
    verify_execution_plan(problem, schedule)
    if type(device) is not int or not 0 <= device < 2**31:
        raise ValueError('CUDA copy lowering requires an explicit device ordinal')
    if schedule['transfers'] or problem.activities or problem.timing_constraints:
        raise ValueError('CUDA copy lowering requires explicit copy operations with direct dependencies')
    resources = {row.id: row for row in problem.resources}
    if (not memory_kinds or len(set(memory_kinds.values())) != len(memory_kinds)
            or any(kind not in ('PinnedHost', 'Device') or name not in resources or resources[name].kind != 'MEMORY'
                   for name, kind in memory_kinds.items())):
        raise ValueError('CUDA copy pools must bind distinct pinned-host and device memory domains')
    channels = set(stream_channels)
    if not channels or any(name not in resources or resources[name].kind != 'CHANNEL' or resources[name].capacity != 1
                           for name in channels):
        raise ValueError('CUDA copy streams require exclusive planning channels')

    selected = {row['operation']: row for row in schedule['selected']}
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    for edge in problem.data_edges:
        incoming[edge.consumer].append(edge)
        outgoing[edge.producer].append(edge)
    copies, returns, stream_keys, operands, slots = [], [], {}, {}, {}
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
        if (mode.lowering != 'CUDA_MEMCPY_ASYNC' or mode.workspace_bytes or mode.cost.estimate_ns <= 0
                or len(incoming[op.id]) != 1 or not outgoing[op.id] or len(mode.output_locations) != 1):
            raise ValueError('CUDA copy operation requires a single explicit byte-preserving transfer')
        edge, = incoming[op.id]
        (output, destination), = mode.output_locations.items()
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
        active = {name for name, units in mode.demands.items() if units}
        own = active & channels
        if len(own) != 1 or any(resources[name].kind not in ('CHANNEL', 'BANDWIDTH_CLASS') for name in active):
            raise ValueError('each CUDA copy needs one selected stream and qualified copy resource demands')
        stream_keys[op.id], = own
        material(edge.value, source, edge.nbytes)
        material(output, destination, edge.nbytes)
        operands[op.id] = edge.value, output, direction
        copies.append((op, mode))

    if not copies or len(returns) != 1 or any(returns[0] in predecessors[op.id] for op, _ in copies):
        raise ValueError('CUDA lowering requires a nonempty copy region and one terminal host return')
    occupied = dict(problem.reserved_memory)
    for pool, size in slots.values():
        occupied[pool] = occupied.get(pool, 0) + size
    if any(size > resources[pool].capacity for pool, size in occupied.items()):
        raise ValueError('CUDA copy buffers cannot all remain allocated within the verified budget')

    copies.sort(key=lambda item: (selected[item[0].id]['start_ns'], item[0].id))
    used_streams = tuple(sorted(set(stream_keys.values())))
    stream_indices = {key: index for index, key in enumerate(used_streams)}
    streams = [stream_indices[stream_keys[op.id]] for op, _ in copies]
    buffers = tuple(sorted(slots))
    buffer_indices = {key: index for index, key in enumerate(buffers)}
    waits = _completion_waits(copies, selected, streams, predecessors)
    recorded = {prior for row in waits for prior in row}

    lines = ['#pragma once', '', '#include "cuda_copy_schedule_v1.h"', '#include <array>', '',
             'namespace scar_cuda_generated {', 'using namespace scar::cuda_copy_v1;', '',
             f'inline constexpr int device = {device};',
             f'inline constexpr std::array<BufferSpec, {len(buffers)}> buffers{{{{']
    for value in buffers:
        pool, size = slots[value]
        lines.append(f'    {{Memory::{memory_kinds[pool]}, {size}ULL}},')
    lines.extend(['}};', ''])
    for index, row in enumerate(waits):
        if row:
            entries = ', '.join(map(str, row))
            lines.append(f'inline constexpr std::array<std::size_t, {len(row)}> wait_{index}{{{entries}}};')
    lines.append(f'inline constexpr std::array<Step, {len(copies)}> steps{{{{')
    for index, (op, _) in enumerate(copies):
        source, destination, direction = operands[op.id]
        dependency = f'wait_{index}' if waits[index] else '{}'
        record = 'true' if index in recorded else 'false'
        lines.append(f'    {{{streams[index]}, {buffer_indices[source]}, {buffer_indices[destination]}, '
                     f'{direction}, {dependency}, {record}}},')
    lines.extend(['}};', '', 'inline void run(std::span<Buffer* const> materializations) {',
                  f'    detail::execute(device, {len(used_streams)}, buffers, steps, materializations);', '}', '',
                  '}  // namespace scar_cuda_generated', ''])
    return CudaCopySource('\n'.join(lines), buffers, used_streams)
