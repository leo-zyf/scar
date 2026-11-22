"""Render native CPU inference from a scheduled functional ATen recipe."""

import re

from scar.codegen.aten_native_graph import _READY_HEADER, _TEAM_HEADER, _render
from scar.codegen.aten_recurrence_source import _FILE_CLI, _PROBE
from scar.optimization.aten_team import data_dependencies


_PROFILE = re.compile(r'[A-Za-z][A-Za-z0-9_]{0,47}')


def valid_profile_name(name):
    return isinstance(name, str) and _PROFILE.fullmatch(name) is not None


def _ready_entry(placement, recipe, cpu):
    cpus = placement['cpu_ids']
    if (len(cpus) != 3 or any(type(item) is not int or not 0 <= item < 1024 for item in cpus)
            or len(set(cpus)) != 3 or cpus[0] != cpu):
        raise ValueError('ready inference requires three distinct CPUs headed by the caller')
    count = len(recipe['operations'])
    lanes = placement['lanes']
    if (len(lanes) != count or any(type(lane) is not int or not 0 <= lane < 3 for lane in lanes)
            or len(set(lanes)) < 2):
        raise ValueError('ready inference requires one valid lane per operation and at least two active lanes')
    order = placement['operation_order']
    if order is None:
        order = list(range(count))
    if any(type(i) is not int for i in order) or sorted(order) != list(range(count)):
        raise ValueError('ready inference requires an exact operation permutation')
    dependencies = data_dependencies(recipe)
    rank = {index: position for position, index in enumerate(order)}
    if any(rank[parent] >= rank[index] for index, parents in enumerate(dependencies) for parent in parents):
        raise ValueError('ready inference order violates a Tensor dependency')

    cpu_list = ', '.join(map(str, cpus))
    lane_list = ', '.join(map(str, lanes))
    setup = (f'        graph::CpuTeam team(std::array<int, 3>{{{cpu_list}}});\n'
             f'        constexpr graph::ReadyLanes lanes{{{lane_list}}};\n')
    call = 'graph::run_ready(inputs, team, lanes)'
    if order != list(range(count)):
        order_list = ', '.join(map(str, order))
        setup += f'        constexpr graph::ReadyOrder order{{{order_list}}};\n'
        call = 'graph::run_ready_ordered(inputs, team, lanes, order)'
    setup += f'        const auto execute = [&](const auto& inputs) {{ return {call}; }};\n'
    return setup, dependencies


def render_exported_entry(recipe, bindings, release_after, cpu, *, profile=None, placement=None):
    """Lower an admitted serial or ready placement to the native file ABI.

    A ready probe measures only elapsed phases. Concurrent operator service
    samples come from the planner's separate lane calibration.
    """
    if type(cpu) is not int or not 0 <= cpu < 1024:
        raise ValueError('native inference requires one Linux CPU ID')
    file_cli, marker, _ = _FILE_CLI.read_text().rpartition('\nint main(')
    if not marker:
        raise ValueError('native file ABI template is missing its entry')
    namespace = 'scar_aten_graph'
    header = 'include/scar_generated/graph.hpp'
    source = 'src/main.cpp'
    entry = 'main'
    if profile is not None:
        if not valid_profile_name(profile):
            raise ValueError('expected an ASCII C++ inference profile name')
        namespace += '_' + profile
        source = f'src/profiles/{profile}.cpp'
        header = f'include/scar_generated/profiles/{profile}.hpp'
        entry = f'scar_profile_{profile}'
        file_cli = file_cli.replace('#include "graph.hpp"', f'#include "profiles/{profile}.hpp"', 1)
        file_cli = file_cli.replace('namespace graph = scar_aten_graph;', f'namespace graph = {namespace};', 1)
    affinity = f'''        cpu_set_t mask;
        if (sched_getaffinity(0, sizeof(mask), &mask) != 0 || CPU_COUNT(&mask) != 1 || !CPU_ISSET({cpu}, &mask)) {{
            throw std::runtime_error("native inference requires the planned CPU");
        }}
'''
    setup = ''
    execute = 'graph::run'
    observe = 'graph::run_observed(inputs, measurement)'
    probe_count = len(recipe['operations'])
    dependencies = None
    files = {}
    if placement is not None:
        setup, dependencies = _ready_entry(placement, recipe, cpu)
        affinity = ''  # CpuTeam checks its complete mask and pins each lane.
        execute = 'execute'
        observe = 'execute(inputs)'
        probe_count = 0
        for path in (_TEAM_HEADER, _READY_HEADER):
            files[f'src/runtime/{path.name}'] = path.read_text()
    cpp = f'''{file_cli}
#include "runtime/aten_tensor_probe_v1.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <sched.h>
#include <string_view>

int {entry}(int argc, char** argv) {{
    const bool probe = argc > 1 && std::string_view(argv[1]) == "--calibrate";
    const int first = probe ? 2 : 1;
    if (static_cast<std::size_t>(argc) != first + graph::input_count + graph::output_count) {{
        std::fputs("usage: aten-inference <input.f32>... <new-output.f32>...\\n", stderr);
        return 2;
    }}
    try {{
        ScarTensorBodyProbe<{probe_count}> measurement(probe);
{affinity}        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        at::NoGradGuard no_grad;
{setup}        if (!probe) return invoke(argv + first, {execute});
        measurement.setup_complete();
        const auto status = invoke(argv + first, [&](const auto& inputs) {{
            measurement.input_ready();
            auto output = {observe};
            measurement.computed();
            return output;
        }});
        measurement.write();
        return status;
    }} catch (const std::exception& error) {{
        std::fprintf(stderr, "SCAR ATen inference failed: %s\\n", error.what());
        return 1;
    }}
}}
'''
    files.update({source: cpp, 'src/runtime/aten_tensor_probe_v1.h': _PROBE.read_text(),
                  header: _render(recipe, bindings, release_after=release_after, observe_serial=placement is None,
                                  namespace=namespace, ready_dependencies=dependencies)})
    return files


def render_profile_dispatch(profiles):
    if len(profiles) < 2 or len(set(profiles)) != len(profiles) or not all(map(valid_profile_name, profiles)):
        raise ValueError('a native dispatch requires two or more distinct profile names')
    declarations = '\n'.join(f'int scar_profile_{name}(int, char**);' for name in profiles)
    cases = '\n'.join(f'    if (profile == "{name}") return scar_profile_{name}(argc - 1, argv + 1);'
                      for name in profiles)
    return f'''#include <cstdio>
#include <string_view>

{declarations}

int main(int argc, char** argv) {{
    if (argc < 2) {{
        std::fputs("usage: aten-inference PROFILE <input.f32>... <new-output.f32>...\\n", stderr);
        return 2;
    }}
    const std::string_view profile = argv[1];
{cases}
    std::fputs("unknown native inference profile\\n", stderr);
    return 2;
}}
'''
