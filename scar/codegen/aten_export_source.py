"""Render native CPU inference from a scheduled functional ATen recipe."""

from scar.codegen.aten_native_graph import _render
from scar.codegen.aten_recurrence_source import _FILE_CLI, _PROBE


def render_exported_entry(recipe, bindings, release_after, cpu):
    if type(cpu) is not int or not 0 <= cpu < 1024:
        raise ValueError('native inference requires one Linux CPU ID')
    file_cli, marker, _ = _FILE_CLI.read_text().rpartition('\nint main(')
    if not marker:
        raise ValueError('native file ABI template is missing its entry')
    cpp = f'''{file_cli}
#include "runtime/aten_tensor_probe_v1.h"
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <sched.h>
#include <string_view>

int main(int argc, char** argv) {{
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
    return {'src/main.cpp': cpp, 'src/runtime/aten_tensor_probe_v1.h': _PROBE.read_text(),
            'include/scar_generated/graph.hpp': _render(recipe, bindings, release_after=release_after, observe_serial=True)}
