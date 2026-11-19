"""Render admitted two- and three-lane windows using ordinary provider calls."""


def render_team_calls(calls, releases, outputs, legal_widths, *, dependencies):
    def observed(position, observer='observe'):
        _, expression = calls[position]
        return f'{observer}({position}, [&] {{ return {expression}; }})'

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
        parallel.append(f'{indent}{value} = {observed(index, "observe_serial")};')
        parallel.extend(f'{indent}{line}' for line in releases[index])
        for _ in range(1, legal_widths[index]):
            indent = indent[:-4]
            parallel.append(f'{indent}}}')
        if covering:
            parallel.append('    }')

    widths = ', '.join(map(str, legal_widths))
    edges = sorted(set(dependencies))
    if any(type(producer) is not int or type(consumer) is not int or not 0 <= producer < consumer < len(calls)
           for producer, consumer in edges):
        raise ValueError('team graph dependencies must follow source order')
    edge_values = ', '.join(f'TeamDataEdge{{{producer}, {consumer}}}' for producer, consumer in edges)
    serial_body, parallel_body = '\n'.join(serial), '\n'.join(parallel)
    return f'''
inline constexpr std::size_t operation_count = {len(calls)};
using WindowSelection = std::array<std::uint8_t, operation_count>;
inline constexpr WindowSelection legal_window_widths{{{widths}}};
using TeamDataEdge = std::array<std::size_t, 2>;
inline constexpr std::array<TeamDataEdge, {len(edges)}> team_data_edges{{{edge_values}}};

template <class Observer>
inline std::array<at::Tensor, output_count> run_observed(std::span<const at::Tensor> inputs, Observer&& observe) {{
    validate_inputs(inputs);
{serial_body}
    return {{{outputs}}};
}}

template <class Observer, class GroupObserver, class SerialObserver>
inline std::array<at::Tensor, output_count> run_parallel_groups_observed(
        std::span<const at::Tensor> inputs, CpuTeam& team, const WindowSelection& windows,
        Observer&& observe, GroupObserver&& observe_group, SerialObserver&& observe_serial) {{
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
    return run_parallel_groups_observed(inputs, team, windows, observe,
                                        [](std::size_t, std::size_t, auto&& run) {{ run(); }}, observe);
}}

inline std::array<at::Tensor, output_count> run_parallel(
        std::span<const at::Tensor> inputs, CpuTeam& team, const WindowSelection& windows) {{
    return run_parallel_observed(inputs, team, windows, [](std::size_t, auto&& call) {{ return call(); }});
}}
'''


def render_ready_support(calls, outputs, dependencies):
    """Use the original Tensor edges to drive an opt-in ready DAG."""
    incoming = [set() for _ in calls]
    for producer, consumer in dependencies:
        if type(producer) is not int or type(consumer) is not int or not 0 <= producer < consumer < len(calls):
            raise ValueError('ready DAG dependencies must follow source order')
        incoming[consumer].add(producer)
    offsets = [0]
    predecessors = []
    for parents in incoming:
        predecessors.extend(sorted(parents))
        offsets.append(len(predecessors))

    offset_items = ', '.join(map(str, offsets))
    predecessor_items = ', '.join(map(str, predecessors))
    values = '\n'.join(f'    at::Tensor {value};' for value, _ in calls)
    cases = '\n'.join(f'        case {i}: {value} = observe({i}, [&] {{ return {expression}; }}); return;'
                      for i, (value, expression) in enumerate(calls))
    return f'''
using ReadyLanes = std::array<std::uint8_t, operation_count>;
inline constexpr std::array<std::size_t, operation_count + 1> ready_offsets{{{offset_items}}};
inline constexpr std::array<std::size_t, {len(predecessors)}> ready_predecessors{{{predecessor_items}}};

template <class Observer>
inline std::array<at::Tensor, output_count> run_ready(
        std::span<const at::Tensor> inputs, CpuTeam& team, const ReadyLanes& lanes, Observer&& observe) {{
    validate_inputs(inputs);
{values}
    execute_ready_dag(team, lanes, ready_offsets, ready_predecessors, [&](std::size_t index) {{
        switch (index) {{
{cases}
        default: throw std::logic_error("unknown native DAG operation");
        }}
    }});
    return {{{outputs}}};
}}

inline std::array<at::Tensor, output_count> run_ready(
        std::span<const at::Tensor> inputs, CpuTeam& team, const ReadyLanes& lanes) {{
    return run_ready(inputs, team, lanes, [](std::size_t, auto&& call) {{ return call(); }});
}}
'''
