"""Render an ordered Tensor recurrence using the native ATen file ABI."""

import json
from pathlib import Path


_FILE_CLI = Path(__file__).with_name('aten_native_cli_v1.cpp.in')
_PROBE = Path(__file__).with_name('aten_tensor_probe_v1.h')
_ENTRIES = {
    'ADD': ('add', 'add_Tensor'),
    'SUB': ('sub', 'sub_Tensor'),
    'MUL': ('mul', 'mul_Tensor'),
    'MUL_SCALAR': ('mul', 'mul_Scalar'),
}


def _scalar(value):
    if type(value) is int:
        literal = '(-9223372036854775807LL - 1)' if value == -(1 << 63) else str(value)
        return f'at::Scalar(std::int64_t{{{literal}}})'
    return f'at::Scalar({value.hex()})'


def _placement(rhs_mode, cpus, lanes, operation_order):
    if any(type(cpu) is not int or not 0 <= cpu < 1024 for cpu in cpus) or len(set(cpus)) != len(cpus):
        raise ValueError('recurrence CPU IDs must be distinct Linux CPU IDs')
    if rhs_mode == 'SERIAL':
        if len(cpus) > 1 or lanes is not None or operation_order is not None:
            raise ValueError('serial recurrence needs at most one CPU ID and no worker schedule')
        if not cpus:
            return '', '', 'graph::run(rhs_inputs)'
        return '', f'        check_serial_cpu({cpus[0]});\n', 'graph::run(rhs_inputs)'
    if rhs_mode != 'READY_DAG' or len(cpus) != 3 or not lanes:
        raise ValueError('recurrence requires a serial or admitted three-core ready RHS')
    if any(type(lane) is not int or not 0 <= lane < 3 for lane in lanes) or len(set(lanes)) < 2:
        raise ValueError('ready recurrence requires valid lanes for the RHS graph')
    if operation_order is not None and (any(type(op) is not int for op in operation_order)
                                        or sorted(operation_order) != list(range(len(lanes)))):
        raise ValueError('ready recurrence requires an exact RHS operation permutation')

    cpu_list = ', '.join(map(str, cpus))
    lane_list = ', '.join(map(str, lanes))
    constructor = f'        graph::CpuTeam team(std::array<int, 3>{{{cpu_list}}});\n'
    definitions = (f'static_assert(graph::operation_count == {len(lanes)});\n'
                   f'constexpr graph::ReadyLanes rhs_lanes{{{lane_list}}};\n')
    if operation_order is None:
        return definitions, constructor, 'graph::run_ready(rhs_inputs, team, rhs_lanes)'
    order_list = ', '.join(map(str, operation_order))
    definitions += f'constexpr graph::ReadyOrder rhs_order{{{order_list}}};\n'
    return definitions, constructor, 'graph::run_ready_ordered(rhs_inputs, team, rhs_lanes, rhs_order)'


def render_recurrence(plan, *, rhs_mode, cpus, lanes=None, operation_order=None):
    """Use an admitted recurrence plan to emit one shared loop body and native main."""
    definitions, constructor, call = _placement(rhs_mode, cpus, lanes, operation_order)
    state_index = plan['state_input_index']
    if type(state_index) is not int or state_index < 0:
        raise ValueError('recurrence requires a nonnegative RHS state parameter index')
    # Keep the file ABI's reader, borrowed-storage lifetime and atomic writer.
    file_cli, marker, _ = _FILE_CLI.read_text().rpartition('\nint main(')
    if not marker:
        raise ValueError('native file ABI template is missing its standalone main')

    headers, signatures = set(), {}
    values = {plan['inputs'][0]['id']: 'state', plan['inputs'][1]['id']: 'step_size'}
    temporaries, body = [], []
    for index, (operation, releases) in enumerate(zip(plan['operations'], plan['release_after'], strict=True)):
        kind = operation['op']
        arguments = [values[arg['value']] if 'value' in arg else _scalar(arg['scalar'])
                     for arg in operation['arguments']]
        if kind == 'RHS':
            expression = f'call_rhs({arguments[0]})'
        else:
            header, entry = _ENTRIES[kind]
            headers.add(f'#include <ATen/ops/{header}_ops.h>')
            target = 'at::_ops::' + entry
            schema = operation['native_schema'].removeprefix('aten::')
            if signatures.setdefault(target, schema) != schema:
                raise ValueError('recurrence operator has multiple native declarations')
            expression = f'{target}::call({", ".join(arguments)})'
        name = f'temporary_{index}'
        values[operation['id']] = name
        temporaries.append(name)
        body.append(f'        auto {name} = observe({index}, [&] {{\n'
                    f'            auto result = {expression};')
        body.extend(f'            {temporaries[released]}.reset();' for released in releases if released != index)
        body.extend(('            return result;', '        });'))
        if index in releases:
            body.append(f'        {name}.reset();')
    includes = '\n'.join(sorted(headers))
    checks = '\n'.join(f'static_assert(std::string_view({target}::schema_str) == {json.dumps(schema)});'
                       for target, schema in sorted(signatures.items()))
    step = '\n'.join(body)
    result = values[plan['output']]
    parallel_rhs = rhs_mode == 'READY_DAG'
    team_parameter = ', graph::CpuTeam& team' if parallel_rhs else ''
    team_argument = ', team' if parallel_rhs else ''
    serial_guard = ''
    if rhs_mode == 'SERIAL' and cpus:
        serial_guard = '''
void check_serial_cpu(int cpu) {
    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0 || CPU_COUNT(&mask) != 1 || !CPU_ISSET(cpu, &mask)
            || sched_getcpu() != cpu) {
        throw std::runtime_error("native serial recurrence requires its exact CPU affinity");
    }
}
'''

    region = f'''#pragma once
#include "runtime/file_io.hpp"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <ATen/ops/scalar_tensor.h>
{includes}

#include <cfenv>
#include <charconv>
#include <sched.h>
#include <string_view>

namespace {{
static_assert(graph::output_count == 1);
constexpr std::size_t state_index = {state_index};
static_assert(state_index < graph::input_count);
{definitions}{checks}
{serial_guard}
std::int32_t parse_step_count(std::string_view text) {{
    std::int32_t count = 0;
    const auto end = text.data() + text.size();
    const auto [next, error] = std::from_chars(text.data(), end, count);
    if (next != end || error != std::errc{{}} || count <= 0) {{
        throw std::invalid_argument("recurrence step count must be a positive 32-bit integer");
    }}
    return count;
}}

template <class Observer>
std::array<at::Tensor, graph::output_count> repeat(
        const std::array<at::Tensor, graph::input_count>& inputs, std::int32_t count{team_parameter}, Observer&& observe) {{
    const auto step_size = at::scalar_tensor({float(plan['step_size']).hex()},
                                            at::TensorOptions().device(at::kCPU).dtype(at::kDouble));
    auto rhs_inputs = inputs;
    const auto call_rhs = [&](const at::Tensor& stage) {{
        rhs_inputs[state_index] = stage;
        auto outputs = {call};
        rhs_inputs[state_index].reset();
        return std::move(outputs[0]);
    }};
    auto state = inputs[state_index];
    for (std::int32_t iteration = 0; iteration < count; ++iteration) {{
{step}
        state = std::move({result});
    }}
    return {{std::move(state)}};
}}
}}  // namespace
'''

    main = f'''#include "regions/step.hpp"
#include "runtime/aten_tensor_probe_v1.h"

int main(int argc, char** argv) {{
    const bool probe = argc > 1 && std::string_view(argv[1]) == "--calibrate";
    const int first = probe ? 2 : 1;
    if (static_cast<std::size_t>(argc) != first + 1 + graph::input_count + graph::output_count) {{
        std::fputs("usage: aten-recurrence <count> <input.f32>... <new-output.f32>\\n", stderr);
        return 2;
    }}
    try {{
        ScarTensorBodyProbe<{len(plan['operations'])}> measurement(probe);
        const auto count = parse_step_count(argv[first]);
        if (std::fegetround() != FE_TONEAREST) {{
            throw std::runtime_error("native recurrence requires default floating-point rounding");
        }}
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        at::NoGradGuard no_grad;
{constructor}        if (!probe) {{
            return invoke(argv + first + 1, [&](const auto& inputs) {{
                return repeat(inputs, count{team_argument}, [](auto, auto&& run) {{ return run(); }});
            }});
        }}
        measurement.setup_complete();
        const auto status = invoke(argv + first + 1, [&](const auto& inputs) {{
            measurement.input_ready();
            auto output = repeat(inputs, count{team_argument}, measurement);
            measurement.computed();
            return output;
        }});
        measurement.write();
        return status;
    }} catch (const std::exception& error) {{
        std::fprintf(stderr, "SCAR ATen recurrence failed: %s\\n", error.what());
        return 1;
    }}
}}
'''
    return {'src/main.cpp': main, 'src/regions/step.hpp': region,
            'src/runtime/aten_tensor_probe_v1.h': _PROBE.read_text(),
            'src/runtime/file_io.hpp': '#pragma once\n' + file_cli + '\n'}
