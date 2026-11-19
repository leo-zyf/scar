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


def render_ready_support(calls, outputs, dependencies, *, exported_indices, input_read_edges=()):
    """Render completion dependencies and last-use handling of original Tensor values."""
    incoming = [set() for _ in calls]
    for producer, consumer in dependencies:
        if type(producer) is not int or type(consumer) is not int or not 0 <= producer < consumer < len(calls):
            raise ValueError('ready DAG dependencies must follow source order')
        incoming[consumer].add(producer)
    offsets = [0]
    predecessors = []
    consumers = [0] * len(calls)
    for parents in incoming:
        predecessors.extend(sorted(parents))
        offsets.append(len(predecessors))
        for parent in parents:
            consumers[parent] += 1

    exported_indices = tuple(exported_indices)
    if not exported_indices or any(type(index) is not int or not 0 <= index < len(calls) for index in exported_indices):
        raise ValueError('ready DAG requires explicit exported result indices')
    exported = set(exported_indices)

    offset_items = ', '.join(map(str, offsets))
    predecessor_items = ', '.join(map(str, predecessors))
    consumer_items = ', '.join(map(str, consumers))
    export_items = ', '.join('true' if i in exported else 'false' for i in range(len(calls)))
    values = '\n'.join(f'    auto& {value} = values[{i}];' for i, (value, _) in enumerate(calls))
    cases = '\n'.join(f'        case {i}: {value} = observe({i}, [&] {{ return {expression}; }}); return;'
                      for i, (value, expression) in enumerate(calls))
    input_edges = sorted(set(input_read_edges))
    if any(type(input_index) is not int or type(consumer) is not int or input_index < 0
           or not 0 <= consumer < len(calls) for input_index, consumer in input_edges):
        raise ValueError('ready replica reads must refer to external inputs and known operations')
    input_edge_values = ', '.join(f'TeamDataEdge{{{input_index}, {consumer}}}' for input_index, consumer in input_edges)
    return f'''
using ReadyLanes = std::array<std::uint8_t, operation_count>;
using ReadyOrder = std::array<std::size_t, operation_count>;
inline constexpr const ReadyOrder& ready_source_order = source_ready_order<operation_count>;
inline constexpr std::array<std::size_t, operation_count + 1> ready_offsets{{{offset_items}}};
inline constexpr std::array<std::size_t, {len(predecessors)}> ready_predecessors{{{predecessor_items}}};
inline constexpr std::array<std::size_t, operation_count> ready_consumer_counts{{{consumer_items}}};
inline constexpr std::array<bool, operation_count> ready_exported{{{export_items}}};

template <class LaneInputs, class Prepare, class Observer, class ReleaseObserver>
inline std::array<at::Tensor, output_count> run_ready_with_lane_inputs(
        CpuTeam& team, const ReadyLanes& lanes, LaneInputs&& select_inputs, Prepare&& prepare,
        Observer&& observe, ReleaseObserver&& observe_release, const ReadyOrder& order = ready_source_order) {{
    std::array<at::Tensor, operation_count> values;
{values}
    const auto invoke = [&](std::size_t index) {{
        const auto inputs = select_inputs(lanes[index]);
        switch (index) {{
{cases}
        default: throw std::logic_error("unknown native DAG operation");
        }}
    }};
    execute_ready_dag(team, lanes, ready_offsets, ready_predecessors, ready_consumer_counts, ready_exported,
                      invoke, [&](std::size_t index) {{
        values[index].reset();
        observe_release(index);
    }}, prepare, order);
    return {{{outputs}}};
}}

template <class Observer, class ReleaseObserver>
inline std::array<at::Tensor, output_count> run_ready_inspected(
        std::span<const at::Tensor> inputs, CpuTeam& team, const ReadyLanes& lanes,
        Observer&& observe, ReleaseObserver&& observe_release, const ReadyOrder& order = ready_source_order) {{
    validate_inputs(inputs);
    return run_ready_with_lane_inputs(team, lanes, [inputs](std::size_t) {{ return inputs; }},
                                      [](std::size_t) {{}}, observe, observe_release, order);
}}

inline std::array<at::Tensor, output_count> run_ready_ordered(
        std::span<const at::Tensor> inputs, CpuTeam& team, const ReadyLanes& lanes, const ReadyOrder& order) {{
    return run_ready_inspected(inputs, team, lanes, [](std::size_t, auto&& call) {{ return call(); }},
                               [](std::size_t) {{}}, order);
}}

template <class Observer>
inline std::array<at::Tensor, output_count> run_ready(
        std::span<const at::Tensor> inputs, CpuTeam& team, const ReadyLanes& lanes, Observer&& observe) {{
    return run_ready_inspected(inputs, team, lanes, observe, [](std::size_t) {{}});
}}

inline std::array<at::Tensor, output_count> run_ready(
        std::span<const at::Tensor> inputs, CpuTeam& team, const ReadyLanes& lanes) {{
    return run_ready(inputs, team, lanes, [](std::size_t, auto&& call) {{ return call(); }});
}}

#ifdef SCAR_ATEN_READY_REPLICA
inline constexpr std::array<TeamDataEdge, {len(input_edges)}> ready_input_read_edges{{{input_edge_values}}};

template <class Observer, class ReleaseObserver, class PrepareObserver, class Materialize>
inline std::array<at::Tensor, output_count> run_ready_replica_inspected(
        std::span<const at::Tensor> inputs, CpuTeam& team, const ReadyLanes& lanes, ReadyReplica replica,
        Observer&& observe, ReleaseObserver&& observe_release, PrepareObserver&& observe_prepare,
        Materialize&& materialize) {{
    validate_inputs(inputs);
    if (replica.input_index >= inputs.size() || replica.lane >= 3 || inputs[replica.input_index].numel() == 0
            || !std::any_of(ready_input_read_edges.begin(), ready_input_read_edges.end(), [&](const auto& edge) {{
                return edge[0] == replica.input_index && lanes[edge[1]] == replica.lane;
            }})) {{
        throw std::invalid_argument("ready replica must identify a nonempty input and a consuming lane");
    }}
    std::array<at::Tensor, input_count> replicated_inputs;
    const auto select_inputs = [&](std::size_t lane) {{
        return lane == replica.lane ? std::span<const at::Tensor>(replicated_inputs) : inputs;
    }};
    const auto prepare = [&](std::size_t lane) {{
        if (lane != replica.lane) return;
        std::copy(inputs.begin(), inputs.end(), replicated_inputs.begin());
        replicated_inputs[replica.input_index] = observe_prepare([&] {{
            return materialize(inputs[replica.input_index]);
        }});
    }};
    return run_ready_with_lane_inputs(team, lanes, select_inputs, prepare, observe, observe_release);
}}

template <class Observer, class ReleaseObserver, class CopyObserver>
inline std::array<at::Tensor, output_count> run_ready_replicated_inspected(
        std::span<const at::Tensor> inputs, CpuTeam& team, const ReadyLanes& lanes, ReadyReplica replica,
        Observer&& observe, ReleaseObserver&& observe_release, CopyObserver&& observe_copy) {{
    return run_ready_replica_inspected(inputs, team, lanes, replica, observe, observe_release, observe_copy,
        materialize_ready_replica);
}}

inline std::array<at::Tensor, output_count> run_ready_replicated(
        std::span<const at::Tensor> inputs, CpuTeam& team, const ReadyLanes& lanes, ReadyReplica replica) {{
    return run_ready_replicated_inspected(inputs, team, lanes, replica,
        [](std::size_t, auto&& call) {{ return call(); }}, [](std::size_t) {{}}, [](auto&& copy) {{ return copy(); }});
}}

template <class Observer, class ReleaseObserver, class PrepareObserver>
inline std::array<at::Tensor, output_count> run_ready_retained_inspected(
        std::span<const at::Tensor> inputs, CpuTeam& team, const ReadyLanes& lanes, ReadyReplica replica,
        ReadyRetainedReplica& cache, Observer&& observe, ReleaseObserver&& observe_release,
        PrepareObserver&& observe_prepare) {{
    return run_ready_replica_inspected(inputs, team, lanes, replica, observe, observe_release, observe_prepare,
        [&](const at::Tensor& input) {{ return cache.acquire(input); }});
}}

inline std::array<at::Tensor, output_count> run_ready_retained(
        std::span<const at::Tensor> inputs, CpuTeam& team, const ReadyLanes& lanes, ReadyReplica replica,
        ReadyRetainedReplica& cache) {{
    return run_ready_retained_inspected(inputs, team, lanes, replica, cache,
        [](std::size_t, auto&& call) {{ return call(); }}, [](std::size_t) {{}}, [](auto&& call) {{ return call(); }});
}}
#endif
'''
