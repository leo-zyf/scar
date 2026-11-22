"""Render an ordered Tensor recurrence using the native ATen file ABI."""

import json
from pathlib import Path, PurePosixPath


_FILE_CLI = Path(__file__).with_name('aten_native_cli_v1.cpp.in')
_PROBE = Path(__file__).with_name('aten_tensor_probe_v1.h')
FRESH_SOURCE_FILE = 'FRESH_EXCLUSIVE_CREATE_MODE_0666'
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


def _source_file_abi(file_cli):
    """Share the raw tensor reader and publisher with the source entry."""
    signature = 'template <class Runner>\nint invoke(char** paths, Runner&& runner) {'
    load = '    for (std::size_t i = 0; i < inputs.size(); ++i) {\n'
    stdout = '    constexpr auto byte_order = std::endian::native'
    if any(file_cli.count(marker) != 1 for marker in (signature, load, stdout)):
        raise ValueError('native file ABI template differs from the source-entry adapter')
    file_cli = file_cli.replace(signature, '''template <class Runner, class Stdout>
int invoke(char** paths, Runner&& runner, std::span<const std::size_t> load_order, Stdout&& stdout_record) {''')
    file_cli = file_cli.replace(load, '''    for (std::size_t ordinal = 0; ordinal < inputs.size(); ++ordinal) {
        const auto i = load_order.empty() ? ordinal : load_order[ordinal];
        if (!load_order.empty() && !fs::is_regular_file(paths[i])) {
            throw std::runtime_error(std::string("expected a regular source input file: ") + paths[i]);
        }
''')
    prefix, _, suffix = file_cli.partition(stdout)
    raw_stdout, terminator, closing = suffix.partition('    return 0;\n}')
    if not terminator:
        raise ValueError('native file ABI template is missing its stdout record')
    # The legacy argv mode keeps the old stdout and the native port read order.
    raw_stdout = '\n'.join('        ' + line for line in (stdout + raw_stdout).splitlines())
    return prefix + '''    stdout_record(outputs, sizes);
    return 0;
}

template <class Runner>
int invoke(char** paths, Runner&& runner) {
    return invoke(paths, std::forward<Runner>(runner), {}, [](const auto& outputs, const auto& sizes) {
''' + raw_stdout + '''
    });
}''' + closing


def _cpp_text(value):
    """Emit a literal with stable UTF-8 bytes independently of source encoding."""
    return '"' + ''.join(chr(byte) if 32 <= byte < 127 and byte not in (34, 92)
                         else f'\\{byte:03o}' for byte in value.encode('utf-8')) + '"'


def _source_entry(plan, abi, team_argument, publication_arg=''):
    if (abi.get('schema'), abi.get('schema_version')) != ('scar.fixed-grid-file-entry', 1):
        raise ValueError('expected an admitted fixed-grid source entry ABI')
    argv, inputs, publication = abi['argv'], abi['inputs'], abi['publication']
    roles = [argv[name] for name in ('input_root', 'steps', 'output')]
    if argv['count'] != 3 or any(type(index) is not int for index in roles) or sorted(roles) != [1, 2, 3]:
        raise ValueError('source entry requires one root, step count and output argv port')
    if (abi['solver']['state_input_index'] != plan['state_input_index']
            or abi['solver']['step_size'] != plan['step_size']
            or abi['domain']['step_count_syntax'] != 'ASCII_DECIMAL_DIGITS_WITHOUT_SIGN_OR_WHITESPACE'
            or (abi['domain']['step_count_min'], abi['domain']['step_count_max']) != (1, 2147483647)):
        raise ValueError('source recurrence count and state contract differs from the plan')
    if (not inputs or not 0 <= plan['state_input_index'] < len(inputs)
            or any(row['port'] != index or row['dtype'] != 'FLOAT32'
                   or len(row['shape']) != 2 or any(type(n) is not int or not 0 < n < 1 << 63 for n in row['shape'])
                   for index, row in enumerate(inputs))):
        raise ValueError('source recurrence requires one ordered CPU float32 input row per native port')
    load_order = sorted(range(len(inputs)), key=lambda index: inputs[index]['load_order'])
    if [inputs[index]['load_order'] for index in load_order] != list(range(len(inputs))):
        raise ValueError('source input load order is not a permutation')
    files = []
    for row in inputs:
        path = PurePosixPath(row['path'])
        if (not path.parts or path.is_absolute() or '..' in path.parts or str(path) != row['path']
                or '\x00' in row['path']):
            raise ValueError('source input requires a normalized relative path')
        files.append(_cpp_text(row['path']))

    fields = publication['stdout']
    if (publication['file'] != 'FINAL_TENSOR_RAW_BYTES' or publication['order'] != ['FILE', 'STDOUT']
            or not fields or [row['key'] for row in fields] != sorted({row['key'] for row in fields})):
        raise ValueError('source recurrence requires file publication followed by a sorted JSON record')
    result_shape = inputs[plan['state_input_index']]['shape']
    pieces = ['{']
    for index, field in enumerate(fields):
        if index:
            pieces.append(', ')
        pieces.append(json.dumps(field['key']) + ': ')
        kind = field['kind']
        if kind == 'STEP_COUNT':
            pieces.append(None)
        elif kind == 'FINAL_SHAPE':
            pieces.append(json.dumps(result_shape))
        elif kind == 'LITERAL':
            pieces.append(json.dumps(field['value'], allow_nan=False))
        else:
            raise ValueError('unsupported source recurrence stdout field')
    pieces.append('}\n')
    stdout = ' << '.join('count' if item is None else _cpp_text(item) for item in pieces)
    lengths = ', '.join(map(str, result_shape))
    checks = [f'        static_assert(std::ranges::equal(graph::input_shapes[{index}], '
              f'std::array<std::int64_t, 2>{{{", ".join(map(str, row["shape"]))}}}));'
              for index, row in enumerate(inputs)]
    checks = '\n'.join(checks)
    return f'''        if (source_entry) {{
            static_assert(graph::input_count == {len(inputs)});
{checks}
            constexpr std::array<const char*, graph::input_count> names{{{', '.join(files)}}};
            constexpr std::array<std::size_t, graph::input_count> load_order{{{', '.join(map(str, load_order))}}};
            const fs::path root(argv[{roles[0] + 1}]);
            std::array<std::string, graph::input_count> paths;
            std::array<char*, graph::input_count + 1> arguments;
            for (std::size_t index = 0; index < paths.size(); ++index) {{
                paths[index] = (root / names[index]).string();
                arguments[index] = paths[index].data();
            }}
            arguments.back() = argv[{roles[2] + 1}];
            return invoke(arguments.data(), [&](const auto& inputs) {{
                auto result = repeat(inputs, count{team_argument}, [](auto, auto&& run) {{ return run(); }});
                constexpr std::array<std::int64_t, 2> expected{{{lengths}}};
                if (!std::ranges::equal(result[0].sizes(), expected)) {{
                    throw std::runtime_error("recurrence result differs from the source output shape");
                }}
                return result;
            }}, load_order, [&](const auto&, const auto&) {{ std::cout << {stdout}; }}{publication_arg});
        }}
'''


def _fresh_source_file(file_cli):
    signature = 'Stdout&& stdout_record) {'
    original = '''    std::array<StagedFile, graph::output_count> staged;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        staged[i].stage(destinations[i], outputs[i].const_data_ptr<float>(), sizes[i]);
    }
    for (std::size_t i = 0; i < staged.size(); ++i) staged[i].publish(destinations[i]);'''
    write_start = '        const auto* next = static_cast<const char*>(data);\n'
    write_end = '        if (close(fd) == -1) throw std::system_error(errno, std::generic_category(), temporary);'
    anchors = (signature, original, write_start, write_end, 'struct StagedFile {', '#include <fstream>')
    if any(file_cli.count(anchor) != 1 for anchor in anchors):
        raise ValueError('fixed-grid source publisher does not match its generated file ABI')
    begin = file_cli.index(write_start)
    end = file_cli.index(write_end, begin) + len(write_end)
    body = file_cli[begin:end]
    writer = '\n'.join(line[4:] for line in body.replace('temporary', 'path').splitlines())
    common_writer = 'void write_and_close(int fd, const fs::path& path, const void* data, std::size_t bytes) {\n'
    helper = common_writer + writer + '''
}

void write_fresh_source_file(const fs::path& path, const void* data, std::size_t bytes) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
    if (fd == -1) throw std::system_error(errno, std::generic_category(), path.string());
    write_and_close(fd, path, data, bytes);
}

'''
    file_cli = file_cli.replace(body, '        write_and_close(fd, temporary, data, bytes);')
    # A path can be rendered as a std::system_error diagnostic only as text.
    helper = helper.replace('std::generic_category(), path);', 'std::generic_category(), path.string());')
    return (file_cli.replace('#include <fstream>', '#include <fcntl.h>\n#include <fstream>')
        .replace('struct StagedFile {', helper + 'struct StagedFile {')
        .replace(signature, 'Stdout&& stdout_record, bool fresh_source_file = false) {')
        .replace(original, '''    if (fresh_source_file) {
        static_assert(graph::output_count == 1);
        write_fresh_source_file(destinations[0], outputs[0].const_data_ptr<float>(), sizes[0]);
    } else {
''' + '\n'.join('    ' + line for line in original.splitlines()) + '\n    }'))


def render_recurrence(plan, *, rhs_mode, cpus, lanes=None, operation_order=None, source_entry_abi=None,
                      source_file_publication=None):
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
    cli_mode = ''
    first = 'probe ? 2 : 1'
    argc_test = 'static_cast<std::size_t>(argc) != first + 1 + graph::input_count + graph::output_count'
    count_arg = 'first'
    source_call = ''
    if source_file_publication not in (None, FRESH_SOURCE_FILE) or (
            source_file_publication is not None and source_entry_abi is None):
        raise ValueError('source file creation requires the admitted normal-completion entry ABI')
    if source_entry_abi is not None:
        file_cli = _source_file_abi(file_cli)
        if source_file_publication == FRESH_SOURCE_FILE:
            file_cli = _fresh_source_file(file_cli)
        publication_arg = ', true' if source_file_publication == FRESH_SOURCE_FILE else ''
        source_call = _source_entry(plan, source_entry_abi, team_argument, publication_arg)
        cli_mode = '    const bool source_entry = argc > 1 && std::string_view(argv[1]) == "--source-entry";\n'
        first = '(probe || source_entry) ? 2 : 1'
        argc_test = f'source_entry ? argc != 5 : {argc_test}'
        count_arg = f"source_entry ? {source_entry_abi['argv']['steps'] + 1} : first"
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
{cli_mode}    const int first = {first};
    if ({argc_test}) {{
        std::fputs("usage: aten-recurrence <count> <input.f32>... <new-output.f32>\\n", stderr);
        return 2;
    }}
    try {{
        ScarTensorBodyProbe<{len(plan['operations'])}> measurement(probe);
        const auto count = parse_step_count(argv[{count_arg}]);
        if (std::fegetround() != FE_TONEAREST) {{
            throw std::runtime_error("native recurrence requires default floating-point rounding");
        }}
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        at::NoGradGuard no_grad;
{constructor}{source_call}        if (!probe) {{
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
