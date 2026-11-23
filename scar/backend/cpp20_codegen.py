"""B2 renders the executable IR as C++20 project inputs."""

from collections import defaultdict
from pathlib import Path
import sys

from scar.contract import (
    ElementwiseEntry,
    FragmentModule,
    Member,
    ScalarModule,
    read_artifact,
    read_native_material,
    write_json,
)

from .abi import f2c_support_header, render_abi_wrappers
from .scalar_codegen import emit_scalar_module, render_expression
from .region_codegen import emit_region_helpers, plan_region_source_edits


def _statements(rows, slots, symbols, indent, *, loop=False):
    code = []
    for kind, *args in rows:
        if kind == "store":
            name, value = args
            expression = render_expression(value, slots, symbols)
            fresh = name not in slots
            slot = slots.setdefault(name, f"v{len(slots)}")
            code.append(f"{indent}{'double ' if fresh else ''}{slot} = {expression};")
        elif kind == "return":
            code.append(f"{indent}return {render_expression(args[0], slots, symbols)};")
        elif kind == "repeat" and loop:
            code.append(indent + "for (std::int64_t step = 0; step < steps; ++step) {")
            code.extend(_statements(args[0], slots.copy(), symbols, indent + "    "))
            code.append(indent + "}")
        else:
            raise ValueError(f"unsupported portable statement {kind}")
    return code


def _functions(portable, trip_count):
    symbols, definitions = {}, []
    entry = portable["entry"]
    for index, function in enumerate(portable["functions"]):
        name = function["name"]
        if name in symbols:
            raise ValueError("duplicate portable function")
        slots = {key: f"v{pos}" for pos, key in enumerate(function["parameters"])}
        if len(slots) != len(function["parameters"]):
            raise ValueError("duplicate portable parameter")
        symbol = f"function{index}"
        definitions.append(
            f"double {symbol}({', '.join('double ' + value for value in slots.values())}) {{"
        )
        if name == entry:
            definitions.append(f"    constexpr std::int64_t steps = {trip_count};")
        definitions.extend(
            _statements(function["body"], slots, symbols, "    ", loop=name == entry)
        )
        definitions.append("}")
        symbols[name] = (symbol, len(function["parameters"]))
    return "\n".join(definitions), symbols[entry]


def _render_elementwise_source(body, cpus):
    rows, columns = body.shape
    if any(cpu >= 1024 for cpu in cpus):
        raise ValueError("CPU index exceeds the C++ affinity ABI")
    definitions, (entry, inputs) = _functions(body.program, body.count)
    elements = rows * columns
    if elements > sys.maxsize // ((inputs + 1) * 8):
        raise ValueError("array-loop storage exceeds host address limits")
    call = (
        f"{entry}("
        + ", ".join(f"input[{index} * elements + i]" for index in range(inputs))
        + ")"
    )
    source = _SOURCE.replace("@FUNCTIONS@", definitions).replace(
        "@CPUS@", ", ".join(map(str, cpus))
    )
    source = (
        source.replace("@ELEMENTS@", str(elements))
        .replace("@INPUTS@", str(inputs))
        .replace("@CALL@", call)
    )
    return source, inputs


_SOURCE = """#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sched.h>
#include <thread>
#include <vector>

static_assert(sizeof(double) == 8 && std::endian::native == std::endian::little);
constexpr std::array cpus{@CPUS@};
constexpr std::int64_t elements = @ELEMENTS@;
constexpr std::int64_t input_count = @INPUTS@;

@FUNCTIONS@

bool pin_cpu(int cpu) {
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    CPU_SET(cpu, &affinity);
    return sched_setaffinity(0, sizeof(affinity), &affinity) == 0;
}

int main(int argc, char** argv) {
    constexpr std::int64_t bytes = elements * input_count * sizeof(double);
    std::error_code error;
    if (argc != 3 || std::filesystem::file_size(argv[1], error) != bytes || error) {
        std::cerr << "input must contain the exact specialized payload\\n";
        return 65;
    }
    if (!pin_cpu(cpus.front())) return 67;
    std::vector<double> input(elements * input_count), output(elements);
    std::ifstream file(argv[1], std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(input.data()), bytes)) return 65;
    for (double value : input) if (!std::isfinite(value)) return 66;

    std::array<int, cpus.size()> errors{};
    auto execute = [&](std::size_t worker) {
        if (worker != 0 && !pin_cpu(cpus[worker])) {
            errors[worker] = 67;
            return;
        }
        std::feclearexcept(FE_ALL_EXCEPT);
        const std::int64_t size = elements / cpus.size();
        const std::int64_t extra = elements % cpus.size();
        const std::int64_t index = worker;
        const std::int64_t begin = size * index + (index < extra ? index : extra);
        const std::int64_t end = begin + size + (index < extra);
        for (std::int64_t i = begin; i < end; ++i) output[i] = @CALL@;
        if (std::fetestexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW)) errors[worker] = 66;
    };
    {
        std::vector<std::jthread> workers;
        for (std::size_t worker = 1; worker < cpus.size(); ++worker) workers.emplace_back(execute, worker);
        execute(0);
    }
    for (int code : errors) if (code) return code;
    for (double value : output) if (!std::isfinite(value)) return 66;
    std::ofstream result(argv[2], std::ios::binary | std::ios::trunc);
    result.write(reinterpret_cast<const char*>(output.data()), elements * sizeof(double));
    return result ? 0 : 68;
}
"""


def _emit_elementwise_entry(body, cpus, project):
    source, inputs = _render_elementwise_source(body, cpus)
    (project / "src").mkdir(parents=True)
    (project / "src/main.cpp").write_text(source, encoding="utf-8")
    configuration = dict(
        sources=["src/main.cpp"],
        libraries=["pthread"],
        compile_options=[
            "-pthread",
            "-O3",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-frounding-math",
            "-ftrapping-math",
        ],
    )
    invocation = dict(
        inputs=inputs,
        shape=body.shape,
        dtype="little-endian-float64",
        args=["INPUT.bin", "OUTPUT.bin"],
    )
    return configuration, (), invocation


def _runtime_support(configuration, project):
    support = set(configuration.pop("runtime_support", ()))
    f2c = support & {"f2c-v1", "f2c-cfi-v1"}
    if support - (f2c | {"gnu-cfi-v1"}) or len(f2c) > 1:
        raise ValueError("unknown native runtime requirement")
    headers = project / "include/scar_generated"
    descriptors = "f2c-cfi-v1" in f2c
    if (descriptors or "gnu-cfi-v1" in support) and not (
        headers / "ISO_Fortran_binding.h"
    ).is_file():
        raise ValueError("generated GNU CFI runtime requires its sealed ABI header")
    if f2c:
        with (headers / "scar_f2c_call.hpp").open("xb") as stream:
            stream.write(f2c_support_header(descriptors=descriptors))
    if "gnu-cfi-v1" not in support:
        return
    name = "src/fortran-abi/scar_gnu_cfi.cpp"
    destination = project / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(Path(__file__).with_name("gnu_cfi.cpp").read_bytes())
    configuration["sources"] = [*configuration["sources"], name]


def _apply_region_replacements(name, data, replacements):
    if not replacements:
        return data
    chunks, position = [], 0
    for start, stop, replacement in sorted(replacements):
        if not position <= start < stop <= len(data):
            raise ValueError(f"overlapping or invalid region replacement in {name}")
        chunks.extend((data[position:start], replacement))
        position = stop
    chunks.append(data[position:])
    return b"".join(chunks)


def _emit_fragment_module(body, project, bundle):
    material = read_native_material(body.artifact.resolve(bundle), graph=False)
    fragments = material["fragments"]
    abi_units = material["abi_units"]
    regions = defaultdict(list)
    configuration = material["configuration"]
    for row in body.regions:
        regions[row["fragment"]["path"]].append(row)
    sources = set(configuration["sources"])
    if fragments.keys() & abi_units.keys() or not abi_units.keys() <= sources:
        raise ValueError("an ABI declaration must own a separate C++20 unit")
    if not regions.keys() <= sources & (fragments.keys() | abi_units.keys()):
        raise ValueError("a typed native region has no owned C++20 unit")
    project.mkdir(parents=True, exist_ok=True)
    definitions, decisions = [], []
    for name in (*fragments, *abi_units):
        relative = Path(name)
        if (
            relative.as_posix() != name
            or len(relative.parts) < 2
            or ".." in relative.parts
            or relative.parts[0] not in {"src", "include"}
        ):
            raise ValueError(f"invalid C++20 fragment path {name}")
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        declarations, spans = (), {}
        if name in abi_units:
            declarations = abi_units[name]
            data, spans = render_abi_wrappers(declarations)
        else:
            data = read_artifact(Member(**fragments[name]).resolve(bundle))
        replacements = []
        if name in regions:
            guarded, helpers, emitted = plan_region_source_edits(
                regions[name], configuration, name, data, declarations, spans
            )
            replacements.extend(guarded)
            definitions.extend(helpers)
            decisions.extend(emitted)
        destination.write_bytes(_apply_region_replacements(name, data, replacements))
    _runtime_support(configuration, project)
    if decisions:
        emit_region_helpers(project, configuration, definitions)
        write_json(
            project / "native-region-lowering.json",
            dict(
                schema="scar.native-region-lowering",
                schema_version=1,
                regions=decisions,
            ),
        )
    return configuration, tuple(material["exports"]), None


def emit_cpp20_project(ir, project, bundle):
    """Return C++20 build inputs and the ABI of the sealed executable entry."""
    if len(ir.bodies) == 1 and isinstance(ir.bodies[0], ElementwiseEntry):
        return _emit_elementwise_entry(ir.bodies[0], ir.cpu_ids, project)
    fragments = next(
        (body for body in ir.bodies if isinstance(body, FragmentModule)), None
    )
    if fragments is None:
        raise ValueError("standalone scalar IR needs a native entry module")
    configuration, exports, invocation = _emit_fragment_module(
        fragments, project, bundle
    )
    for body in ir.bodies:
        if isinstance(body, ScalarModule):
            emit_scalar_module(body, project, configuration)
            exports = (*exports, *(function["name"] for function in body.functions))
    return configuration, exports, invocation
