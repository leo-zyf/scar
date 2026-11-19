"""Render a CUDA schedule as the raw matrix file ABI used by native exports."""

from dataclasses import dataclass
from pathlib import Path

from scar.codegen.cuda_copy_schedule import render_cuda_cublas_schedule, render_cuda_hybrid_schedule


_ROOT = Path(__file__).parent
_TEMPLATE = _ROOT / 'aten_export_cuda_cli_v1.cpp.in'
_CPU_FILE_ABI = _ROOT / 'aten_native_cli_v1.cpp.in'
_RUNTIME_HEADERS = ('cuda_copy_schedule_v1.h', 'cuda_cublas_schedule_v1.h')
ACTIVITY_CALIBRATION_SCHEMA = 'scar.hybrid-native-activity-observation'

# Only materialize the header and typed host branch for a plan that selected CPU MM.
_HOST_MM_HOOKS = {
    'INCLUDE': '\n#include <ATen/ops/mm_ops.h>',
    'STRUCT': '''

struct HostMm {
    HostTensor left;
    HostTensor right;
    HostTensor output;
};''',
    'VARIANT': ', HostMm',
    'PREDICATE': ' || std::holds_alternative<HostMm>(operation)',
    'SCHEMA': '''
static_assert(std::string_view(at::_ops::mm_out::schema_str) ==
              "mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)");''',
    'CALL': ''' else if (const auto* mm = std::get_if<HostMm>(&step.operation)) {
        output_slot = mm->output;
        output = host_view(output_slot, buffers);
        at::_ops::mm_out::call(host_view(mm->left, buffers), host_view(mm->right, buffers), output);
    }''',
}

# Empty expansions preserve the established C++ bytes for archived bundles.
_EXECUTOR_HOOKS = {
    'INCLUDE': '\n#include "cuda_hybrid_activity_v1.h"',
    'TEMPLATE': 'template <bool Calibrating = false>\n',
    'ARGUMENT': ', activity::Report* report = nullptr',
    'SETUP_START': """
    using Timing = std::conditional_t<Calibrating, activity::Timing, activity::NoTiming>;
    const auto setup_start = Timing::now();""",
    'INIT': '\n    const Timing timing(steps, report);',
    'SETUP_END': '\n    timing.setup_done(setup_start);',
    'HOST_WAIT_START': '\n            const auto wait_start = Timing::now();',
    'HOST_START': """
            if constexpr (Calibrating) {
                if (!step.wait_for.empty()) timing.wait_done(wait_start);
            }
            const auto host_start = Timing::now();""",
    'HOST_END': '\n            timing.host_done(index, host_start);',
    'DEVICE_START': '\n        timing.begin_device(index, stream);',
    'DEVICE_END': '\n        timing.end_device(index, stream);',
    'JOIN_START': '\n    const auto join_start = Timing::now();',
    'JOIN_END': '\n    timing.join_done(join_start);',
}

_CLI_HOOKS = {
    'NAMESPACE': '\nnamespace activity = scar::cuda_hybrid_v1::activity;',
    'DESCRIPTION': ('\n              << ",\\\"activity_calibration\\\":{\\\"schema\\\":\\\""'
                    f' << "{ACTIVITY_CALIBRATION_SCHEMA}"'
                    ' << "\\\",\\\"schema_version\\\":1,\\\"steps\\\":" << graph::steps.size() << \'}\''),
    'TEMPLATE': 'template <bool Calibrating = false>\n',
    'ARGUMENT': ', activity::Report* report = nullptr',
    'ALLOC_START': """
    using Timing = std::conditional_t<Calibrating, activity::Timing, activity::NoTiming>;
    const auto allocation_start = Timing::now();""",
    'INPUT_START': """
    if constexpr (Calibrating) report->allocation_ns = activity::elapsed(allocation_start);
    const auto input_start = Timing::now();""",
    'RUN': """if constexpr (Calibrating) {
        report->input_ns = activity::elapsed(input_start);
        graph::detail::execute<true>(graph::device, graph::workspace_bytes, graph::buffers, graph::steps,
                                    bindings, report);
    } else {
        graph::run(bindings);
    }
    const auto publish_start = Timing::now();""",
    'PUBLISH_END': '\n    if constexpr (Calibrating) report->publish_ns = activity::elapsed(publish_start);',
    'FLAG': '\n    const bool calibrate = flag == "--calibrate" && static_cast<std::size_t>(argc) == 2 + inputs.size() + outputs.size();',
    'REQUIRED': ' && !calibrate',
    'USAGE': ' | --calibrate',
    'EXECUTE': """
        else if (calibrate) {
            activity::Report report;
            execute<true>(argv + 2, &report);
            report.write(std::cout);
        }""",
}


@dataclass(frozen=True, slots=True)
class CudaExportSource:
    sources: dict[str, str]
    input_slots: tuple[int, ...]
    output_slots: tuple[int, ...]
    buffer_values: tuple[str, ...]
    stream_channels: tuple[str, ...]


def _apply_hooks(source, prefix, hooks, enabled):
    for key, active in hooks.items():
        marker = f'/* SCAR_{prefix}_{key} */'
        if source.count(marker) != 1:
            raise ValueError(f'native template needs exactly one {marker} hook')
        replacement = active if enabled else ('graph::run(bindings);' if prefix == 'PROBE' and key == 'RUN' else '')
        source = source.replace(marker, replacement)
    return source


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
                               input_ports, output_ports, host_compute_resource=None, calibrate=False):
    """Emit an explicit-trial CUDA executable and its ordered file-to-buffer bindings."""
    if type(calibrate) is not bool or calibrate and host_compute_resource is None:
        raise ValueError('native activity calibration requires an opted-in mixed CPU/CUDA plan')
    kwargs = dict(device=device, memory_kinds=memory_kinds, stream_channels=stream_channels,
                  compute_resource=compute_resource)
    runtime_headers = _RUNTIME_HEADERS
    if host_compute_resource is None:
        lowered = render_cuda_cublas_schedule(problem, schedule, **kwargs)
    else:
        lowered = render_cuda_hybrid_schedule(problem, schedule, host_compute_resource=host_compute_resource, **kwargs)
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
    host_metrics = host_description = ''
    count_waits = 'waits += step.wait_for.size();'
    if host_compute_resource is not None:
        host_metrics = '\n    std::size_t host_provider_calls = 0, host_waits = 0;'
        count_waits = ('if (step.host()) {\n            ++host_provider_calls;\n'
                       '            host_waits += step.wait_for.size();\n        } else {\n'
                       '            waits += step.wait_for.size();\n        }')
        host_description = ('\n              << ",\\\"host_provider_calls\\\":" << host_provider_calls'
                            ' << ",\\\"host_waits\\\":" << host_waits')
    main = main.replace('/* SCAR_HOST_METRICS */', host_metrics)
    main = main.replace('/* SCAR_COUNT_WAITS */', count_waits)
    main = main.replace('/* SCAR_HOST_DESCRIPTION */', host_description)
    main = _apply_hooks(main, 'PROBE', _CLI_HOOKS, calibrate)
    sources = {'src/main.cpp': main, 'include/scar_generated/cuda_schedule.hpp': lowered.header,
               **{f'src/{name}': (_ROOT / name).read_text() for name in runtime_headers}}
    if host_compute_resource is not None:
        header = 'cuda_hybrid_schedule_v1.h'
        runtime = _apply_hooks((_ROOT / header).read_text(), 'HOST_MM', _HOST_MM_HOOKS, lowered.requires_host_mm)
        sources[f'src/{header}'] = _apply_hooks(runtime, 'ACTIVITY', _EXECUTOR_HOOKS, calibrate)
    if calibrate:
        header = 'cuda_hybrid_activity_v1.h'
        sources[f'src/{header}'] = (_ROOT / header).read_text()
    return CudaExportSource(sources, input_slots, output_slots, lowered.buffer_values, lowered.stream_channels)
