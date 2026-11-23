"""B1 renders the executable IR as C++20 project inputs."""

import json
import math
from pathlib import Path
import sys

from scar.contract import ElementwiseEntry, FragmentModule, Member, read_artifact


_BINARY = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


def _expression(row, slots, symbols):
    kind, *args = row
    if kind == "read":
        return slots[args[0]]
    if kind == "constant":
        value = float(args[0])
        if not math.isfinite(value):
            raise ValueError("portable constants must be finite")
        return value.hex()
    if kind == "neg":
        return "(-" + _expression(args[0], slots, symbols) + ")"
    if kind in _BINARY:
        left, right = (_expression(arg, slots, symbols) for arg in args)
        return f"({left} {_BINARY[kind]} {right})"
    if kind == "call":
        name, arity = symbols[args[0]]
        if len(args) != arity + 1:
            raise ValueError("portable call does not match its signature")
        return name + "(" + ", ".join(_expression(arg, slots, symbols) for arg in args[1:]) + ")"
    raise ValueError(f"unsupported portable expression {kind}")


def _statements(rows, slots, symbols, indent, *, loop=False):
    code = []
    for kind, *args in rows:
        if kind == "store":
            name, value = args
            expression = _expression(value, slots, symbols)
            fresh = name not in slots
            slot = slots.setdefault(name, f"v{len(slots)}")
            code.append(f"{indent}{'double ' if fresh else ''}{slot} = {expression};")
        elif kind == "return":
            code.append(f"{indent}return {_expression(args[0], slots, symbols)};")
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
        definitions.append(f"double {symbol}({', '.join('double ' + value for value in slots.values())}) {{")
        if name == entry:
            definitions.append(f"    constexpr std::int64_t steps = {trip_count};")
        definitions.extend(_statements(function["body"], slots, symbols, "    ", loop=name == entry))
        definitions.append("}")
        symbols[name] = (symbol, len(function["parameters"]))
    return "\n".join(definitions), symbols[entry]


def _source(body, cpus):
    rows, columns = body.shape
    if any(cpu >= 1024 for cpu in cpus):
        raise ValueError("CPU index exceeds the C++ affinity ABI")
    definitions, (entry, inputs) = _functions(body.program, body.count)
    elements = rows * columns
    if elements > sys.maxsize // ((inputs + 1) * 8):
        raise ValueError("array-loop storage exceeds host address limits")
    call = f"{entry}(" + ", ".join(f"input[{index} * elements + i]" for index in range(inputs)) + ")"
    source = _SOURCE.replace("@FUNCTIONS@", definitions).replace("@CPUS@", ", ".join(map(str, cpus)))
    source = source.replace("@ELEMENTS@", str(elements)).replace("@INPUTS@", str(inputs)).replace("@CALL@", call)
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

int main(int argc, char** argv) {
    constexpr std::int64_t bytes = elements * input_count * sizeof(double);
    std::error_code error;
    if (argc != 3 || std::filesystem::file_size(argv[1], error) != bytes || error) {
        std::cerr << "input must contain the exact specialized payload\\n";
        return 65;
    }
    std::vector<double> input(elements * input_count), output(elements);
    std::ifstream file(argv[1], std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(input.data()), bytes)) return 65;
    for (double value : input) if (!std::isfinite(value)) return 66;

    std::array<int, cpus.size()> errors{};
    auto execute = [&](std::size_t worker) {
        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        CPU_SET(cpus[worker], &affinity);
        if (sched_setaffinity(0, sizeof(affinity), &affinity) != 0) {
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


def _elementwise(body, cpus, project):
    source, inputs = _source(body, cpus)
    (project / "src").mkdir(parents=True)
    (project / "src/main.cpp").write_text(source, encoding="utf-8")
    configuration = dict(sources=["src/main.cpp"], libraries=["pthread"],
        compile_options=["-pthread", "-O3", "-fno-fast-math", "-ffp-contract=off",
                         "-frounding-math", "-ftrapping-math"])
    invocation = dict(inputs=inputs, shape=body.shape,
                      dtype="little-endian-float64", args=["INPUT.bin", "OUTPUT.bin"])
    return configuration, (), invocation


def _fragments(body, project, bundle):
    material = json.loads(read_artifact(body.artifact.resolve(bundle)))
    if (set(material) != {"schema", "schema_version", "configuration", "fragments", "exports"}
            or material["schema"] != "scar.native-units-ir" or material["schema_version"] != 1):
        raise ValueError("invalid C++20 fragment module")
    for name, reference in material["fragments"].items():
        relative = Path(name)
        if (relative.as_posix() != name or len(relative.parts) < 2 or ".." in relative.parts
                or relative.parts[0] not in {"src", "include"}):
            raise ValueError(f"invalid C++20 fragment path {name}")
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(read_artifact(Member(**reference).resolve(bundle)))
    return material["configuration"], tuple(material["exports"]), None


def lower_project(ir, project, bundle):
    """Return C++20 build inputs and the ABI of one sealed executable body."""
    body, = ir.bodies
    if isinstance(body, ElementwiseEntry):
        return _elementwise(body, ir.cpu_ids, project)
    if isinstance(body, FragmentModule):
        return _fragments(body, project, bundle)
    raise ValueError("unsupported executable body")
