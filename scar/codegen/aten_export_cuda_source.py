"""Render a CUDA schedule as the raw matrix file ABI used by native exports."""

from dataclasses import dataclass
from pathlib import Path

from scar.codegen.cuda_copy_schedule import render_cuda_cublas_schedule


_ROOT = Path(__file__).parent
_TEMPLATE = _ROOT / 'aten_export_cuda_cli_v1.cpp.in'
_CPU_FILE_ABI = _ROOT / 'aten_native_cli_v1.cpp.in'
_RUNTIME_HEADERS = ('cuda_copy_schedule_v1.h', 'cuda_cublas_schedule_v1.h')


@dataclass(frozen=True, slots=True)
class CudaExportSource:
    sources: dict[str, str]
    input_slots: tuple[int, ...]
    output_slots: tuple[int, ...]
    buffer_values: tuple[str, ...]
    stream_channels: tuple[str, ...]


def _file_ports(ports, planned_bytes, indices, kind):
    if not ports:
        raise ValueError(f'CUDA native entry needs at least one file {kind}')
    slots, specifications, seen = [], [], set()
    for port in ports:
        value, shape = port['value'], port['shape']
        if (not isinstance(value, str) or value not in planned_bytes
                or not isinstance(shape, (list, tuple)) or len(shape) != 2
                or any(type(extent) is not int or not 0 < extent < 2**31 for extent in shape)
                or 4 * shape[0] * shape[1] != planned_bytes[value]
                or kind == 'input' and value in seen):
            raise ValueError(f'CUDA file {kind} is inconsistent with its planned FP32 matrix')
        slot = indices[value]
        slots.append(slot)
        specifications.append(f'    {{{slot}, {shape[0]}, {shape[1]}}},')
        seen.add(value)
    if seen != planned_bytes.keys():
        raise ValueError(f'CUDA file {kind} ports do not cover the entire planned host boundary')
    return tuple(slots), '\n'.join((f'constexpr std::array<FilePort, {len(slots)}> {kind}s{{{{',
                                   *specifications, '}};'))


def _host_boundary(problem, schedule, memory_kinds):
    choices = {row['operation']: row['option'] for row in schedule['selected']}
    returns = {task.id: option for task in problem.operations for option in task.options
               if option.id == choices[task.id] and option.lowering == 'CUDA_COPY_RETURN'}
    input_bytes, output_bytes = {}, {}
    for edge in problem.data_edges:
        if edge.producer is None:
            kind = memory_kinds.get(edge.external_location)
            if kind != 'PinnedHost' or input_bytes.setdefault(edge.value, edge.nbytes) != edge.nbytes:
                raise ValueError('CUDA file inputs must be consistent pinned-host materials')
        if edge.consumer in returns:
            kind = memory_kinds.get(returns[edge.consumer].input_locations[edge.value])
            if kind != 'PinnedHost' or output_bytes.setdefault(edge.value, edge.nbytes) != edge.nbytes:
                raise ValueError('CUDA file outputs must be consistent pinned-host materials')
    return input_bytes, output_bytes


def render_exported_cuda_entry(problem, schedule, *, device, memory_kinds, stream_channels, compute_resource,
                               input_ports, output_ports):
    """Emit an explicit-trial CUDA executable and its ordered file-to-buffer bindings."""
    lowered = render_cuda_cublas_schedule(problem, schedule, device=device, memory_kinds=memory_kinds,
                                          stream_channels=stream_channels, compute_resource=compute_resource)
    indices = {value: index for index, value in enumerate(lowered.buffer_values)}
    input_bytes, output_bytes = _host_boundary(problem, schedule, memory_kinds)
    input_slots, inputs = _file_ports(input_ports, input_bytes, indices, 'input')
    output_slots, outputs = _file_ports(output_ports, output_bytes, indices, 'output')

    # Use the CPU ABI's complete-file publisher without bringing in LibTorch.
    _, start, tail = _CPU_FILE_ABI.read_text().partition('struct StagedFile {')
    staging, end, _ = tail.partition('\ntemplate <class Runner>')
    if not start or not end:
        raise ValueError('native file ABI template is missing its staged publisher')
    main = _TEMPLATE.read_text().replace('/* SCAR_FILE_PORTS */', inputs + '\n\n' + outputs)
    main = main.replace('/* SCAR_STAGED_FILE */', start + staging)
    sources = {'src/main.cpp': main, 'include/scar_generated/cuda_schedule.hpp': lowered.header,
               **{f'src/{name}': (_ROOT / name).read_text() for name in _RUNTIME_HEADERS}}
    return CudaExportSource(sources, input_slots, output_slots, lowered.buffer_values, lowered.stream_channels)
