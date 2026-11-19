"""Render native CPU inference from a scheduled functional ATen recipe."""

import re

from scar.codegen.aten_native_graph import _render
from scar.codegen.aten_recurrence_source import _FILE_CLI, _PROBE


_PROFILE = re.compile(r'[A-Za-z][A-Za-z0-9_]{0,47}')


def valid_profile_name(name):
    return isinstance(name, str) and _PROFILE.fullmatch(name) is not None


def render_exported_entry(recipe, bindings, release_after, cpu, *, profile=None):
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
        ScarTensorBodyProbe<{len(recipe['operations'])}> measurement(probe);
        cpu_set_t mask;
        if (sched_getaffinity(0, sizeof(mask), &mask) != 0 || CPU_COUNT(&mask) != 1 || !CPU_ISSET({cpu}, &mask)) {{
            throw std::runtime_error("native inference requires the planned CPU");
        }}
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        at::NoGradGuard no_grad;
        if (!probe) return invoke(argv + first, graph::run);
        measurement.setup_complete();
        const auto status = invoke(argv + first, [&](const auto& inputs) {{
            measurement.input_ready();
            auto output = graph::run_observed(inputs, measurement);
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
    return {source: cpp, 'src/runtime/aten_tensor_probe_v1.h': _PROBE.read_text(),
            header: _render(recipe, bindings, release_after=release_after, observe_serial=True, namespace=namespace)}


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
