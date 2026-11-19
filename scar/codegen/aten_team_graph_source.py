"""Render admitted two- and three-lane windows using ordinary provider calls."""


def render_team_calls(calls, releases, outputs, legal_widths):
    def observed(position):
        _, expression = calls[position]
        return f'observe({position}, [&] {{ return {expression}; }})'

    serial = []
    parallel = [f'    at::Tensor {value};' for value, _ in calls]
    for index, (value, _) in enumerate(calls):
        serial.append(f'    at::Tensor {value} = {observed(index)};')
        serial.extend(f'    {line}' for line in releases[index])

        covering = [f'windows[{start}] >= {index - start + 1}'
                    for start in range(max(0, index - 2), index)
                    if legal_widths[start] > index - start]
        indent = '    '
        if covering:
            parallel.append(f'{indent}if (!({" || ".join(covering)})) {{')
            indent += '    '
        for width in range(legal_widths[index], 1, -1):
            parallel.append(f'{indent}if (windows[{index}] == {width}) {{')
            indices = range(index, index + width)
            closures = [f'[&] {{ {calls[i][0]} = {observed(i)}; }}' for i in indices]
            parallel.extend((f'{indent}    observe_group({index}, {width}, [&] {{',
                             f'{indent}        team.run({", ".join(closures)});',
                             f'{indent}    }});'))
            parallel.extend(f'{indent}    {line}' for i in indices for line in releases[i])
            parallel.append(f'{indent}}} else {{')
            indent += '    '
        parallel.append(f'{indent}{value} = {observed(index)};')
        parallel.extend(f'{indent}{line}' for line in releases[index])
        for _ in range(1, legal_widths[index]):
            indent = indent[:-4]
            parallel.append(f'{indent}}}')
        if covering:
            parallel.append('    }')

    widths = ', '.join(map(str, legal_widths))
    serial_body, parallel_body = '\n'.join(serial), '\n'.join(parallel)
    return f'''
inline constexpr std::size_t operation_count = {len(calls)};
using WindowSelection = std::array<std::uint8_t, operation_count>;
inline constexpr WindowSelection legal_window_widths{{{widths}}};

template <class Observer>
inline std::array<at::Tensor, output_count> run_observed(std::span<const at::Tensor> inputs, Observer&& observe) {{
    validate_inputs(inputs);
{serial_body}
    return {{{outputs}}};
}}

template <class Observer, class GroupObserver>
inline std::array<at::Tensor, output_count> run_parallel_groups_observed(
        std::span<const at::Tensor> inputs, CpuTeam& team, const WindowSelection& windows,
        Observer&& observe, GroupObserver&& observe_group) {{
    validate_inputs(inputs);
    for (std::size_t first = 0; first < operation_count; ++first) {{
        const auto width = windows[first];
        if (width == 0) continue;
        if (width < 2 || width > legal_window_widths[first]) {{
            throw std::invalid_argument("native team has an unqualified window width");
        }}
        for (std::size_t i = 1; i < width; ++i) {{
            if (windows[first + i] != 0) throw std::invalid_argument("native team windows overlap");
        }}
    }}
{parallel_body}
    return {{{outputs}}};
}}

template <class Observer>
inline std::array<at::Tensor, output_count> run_parallel_observed(
        std::span<const at::Tensor> inputs, CpuTeam& team, const WindowSelection& windows, Observer&& observe) {{
    return run_parallel_groups_observed(inputs, team, windows, std::forward<Observer>(observe),
                                        [](std::size_t, std::size_t, auto&& run) {{ run(); }});
}}

inline std::array<at::Tensor, output_count> run_parallel(
        std::span<const at::Tensor> inputs, CpuTeam& team, const WindowSelection& windows) {{
    return run_parallel_observed(inputs, team, windows, [](std::size_t, auto&& call) {{ return call(); }});
}}
'''
