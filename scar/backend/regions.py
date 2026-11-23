"""Render proved counted regions and retain the exact source fallback."""

from hashlib import sha256
import math
from pathlib import Path
import platform
import re

from ..contract import json_digest


_PARAMETERS = {"i32": "int", "f64": "double", "ptr<f64>": "double*", "ptr<const f64>": "const double*"}
_OPERATORS = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_STRICT_FP = ("-fno-fast-math", "-ffp-contract=off", "-frounding-math", "-ftrapping-math")
_UNSAFE_FP = {"-Ofast", "-ffast-math", "-fassociative-math", "-funsafe-math-optimizations",
              "-fno-trapping-math", "-ffinite-math-only", "-fsingle-precision-constant"}


def _identifier(name):
    if type(name) is not str or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"invalid typed native identifier: {name!r}")
    return name


def _expression(row, symbols, index, pointers, locals_):
    kind, dtype, *operands = row
    if kind == "constant" and len(operands) == 1:
        value, = operands
        if dtype == "i32" and type(value) is int and -(2**31) <= value < 2**31:
            return str(value)
        if dtype == "f64" and type(value) in (int, float) and math.isfinite(value):
            return float(value).hex()
    if kind == "parameter" and len(operands) == 1 and symbols.get(operands[0]) == dtype and dtype in {"i32", "f64"}:
        return operands[0]
    if kind == "index" and row == ["index", "i32", index]:
        return index
    if kind == "add" and dtype == "i32" and operands == [["index", "i32", index], ["constant", "i32", 1]]:
        # The loop's last index is below its signed int32 element count.
        return f"({index} + 1)"
    if kind == "local" and dtype == "f64" and len(operands) == 1 and operands[0] in locals_:
        return locals_[operands[0]]
    if kind == "load" and dtype == "f64" and len(operands) == 2:
        pointer, subscript = operands
        if symbols.get(pointer) in {"ptr<f64>", "ptr<const f64>"} and subscript == ["index", "i32", index]:
            pointers.add(pointer)
            return f"{pointer}[{index}]"
    if kind == "cast" and dtype == "f64" and len(operands) == 1 and operands[0][1] == "i32":
        return "static_cast<double>(" + _expression(operands[0], symbols, index, pointers, locals_) + ")"
    if kind in _OPERATORS and dtype == "f64" and len(operands) == 2 and all(operand[1] == "f64" for operand in operands):
        left, right = (_expression(operand, symbols, index, pointers, locals_) for operand in operands)
        return f"({left} {_OPERATORS[kind]} {right})"
    raise ValueError(f"unsupported typed native expression: {row!r}")


def _loops(rows, symbols, length):
    code, pointers = [], set()
    for loop in rows:
        index = _identifier(loop["index"])
        if (loop["kind"] != "for" or loop["index_type"] != "i32" or index in symbols or loop["step"] != 1
                or loop["comparison"] != "lt" or loop["start"] != ["constant", "i32", 0]
                or loop["stop"] != ["parameter", "i32", length] or not loop["body"]):
            raise ValueError("native counted loop does not match its guarded range")
        code.append(f"    for (int {index} = 0; {index} < {length}; ++{index}) {{")
        locals_, names = {}, {*symbols, index}
        for row in loop["body"]:
            if row["kind"] == "bind":
                binding = _identifier(row["binding"])
                if binding in locals_ or row["type"] != "f64" or row["value"][1] != "f64":
                    raise ValueError("invalid typed native local definition")
                local = "scar_local_" + binding
                while local in names:
                    local += "_"
                value = _expression(row["value"], symbols, index, pointers, locals_)
                code.append(f"        const double {local} = {value};")
                locals_[binding] = local
                names.add(local)
                continue
            pointer = row["pointer"]
            if (row["kind"] != "store" or row["type"] != "f64" or symbols.get(pointer) != "ptr<f64>"
                    or row["index"] != ["index", "i32", index] or row["value"][1] != "f64"):
                raise ValueError("native store does not match its guarded range")
            pointers.add(pointer)
            value = _expression(row["value"], symbols, index, pointers, locals_)
            code.append(f"        {pointer}[{index}] = {value};")
        code.append("    }")
    if not code:
        raise ValueError("empty typed native body")
    return code, pointers


def _region(row, symbol, original):
    if (row["schema"], row["schema_version"], row["return_type"]) != ("scar.native-region", 1, "void"):
        raise ValueError("unsupported structured native region")
    if row["effects"]["floating_exceptions"] != "PRESERVE_PROFILE" or row["effects"]["external"] != "NONE":
        raise ValueError("native region has no closed floating-point effect contract")
    parameters = {_identifier(value["name"]): value["type"] for value in row["parameters"]}
    if len(parameters) != len(row["parameters"]) or set(parameters.values()) - _PARAMETERS.keys():
        raise ValueError("invalid typed native signature")
    guard = row["guard"]
    length, pointers = guard["length"], guard["pointers"]
    max_count = guard.get("max_count", 2**31 - 1)
    if (guard["kind"] != "DISJOINT_FP64_RANGES" or parameters.get(length) != "i32" or not pointers
            or type(max_count) is not int or not 0 < max_count < 2**31):
        raise ValueError("unsupported native region guard")
    code, accessed = _loops(row["optimized_body"], parameters, length)
    if len(pointers) != len(set(pointers)) or accessed != set(pointers):
        raise ValueError("native guard does not cover the exact optimized memory access set")
    signature = ", ".join(_PARAMETERS[value] for value in parameters.values())
    signature_named = ", ".join(f"{_PARAMETERS[value]} {key}" for key, value in parameters.items())
    bindings = row["fragment"].get("bindings", dict(zip(parameters, parameters)))
    if set(bindings) != parameters.keys() or len(set(bindings.values())) != len(parameters):
        raise ValueError("native fragment parameter bindings differ from the typed signature")
    arguments = ", ".join(_identifier(bindings[name]) for name in parameters)
    aliases = "scar_ranges_" + symbol.removeprefix("scar_try_")
    if aliases in parameters or symbol in parameters:
        raise ValueError("native helper name collides with its signature")
    helper = [f'extern "C" bool {symbol}({signature_named}) noexcept {{',
              f'    const double* const {aliases}[] = {{{", ".join(pointers)}}};',
              f'    if (!admit({length}, {max_count}, {aliases})) return false;',
              *code, '    return true;', '}']
    prefix = (f'{{\n    extern bool {symbol}({signature}) noexcept __asm__("{symbol}");\n'
              f'    if ({symbol}({arguments})) return;\n    ').encode("ascii")
    return prefix + original + b"\n}", "\n".join(helper)


def _strict_source(configuration, name):
    flags = [*configuration.get("compile_options", ()), *configuration.get("source_settings", {}).get(name, ((), ()))[1]]
    if (_UNSAFE_FP.intersection(flags) or any(flag.startswith("-mfpmath=") and flag != "-mfpmath=sse"
                                             for flag in flags)):
        return False
    contract = next((flag for flag in reversed(flags) if flag.startswith("-ffp-contract=")), None)
    if contract == "-ffp-contract=off":
        return True
    return (contract is None and platform.machine() == "x86_64"
            and not any(flag.startswith("-m") and flag not in {"-m64", "-msse2"} for flag in flags))


def _abi_span(row, declarations, spans):
    witness = row["fragment"]
    symbol = row["symbol"]
    declaration = declarations.get(symbol)
    if (declaration is None or witness["symbol"] != symbol or witness["abi_sha256"] != json_digest(declaration)
            or declaration["result"] is not None):
        raise ValueError(f"native region ABI witness changed: {symbol}")
    actual = []
    for argument in declaration["arguments"]:
        native_type = argument["c_type"]
        if argument.get("descriptor"):
            raise ValueError(f"native region does not admit descriptor arguments: {symbol}")
        if not argument["by_value"]:
            native_type = ("const " if argument["read_only"] else "") + native_type + "*"
        actual.append((argument["wrapper_name"], native_type))
    expected = [(witness["bindings"].get(parameter["name"]), _PARAMETERS.get(parameter["type"]))
                for parameter in row["parameters"]]
    if actual != expected:
        raise ValueError(f"native region parameters differ from the ABI declaration: {symbol}")
    return spans[symbol]


def region_edits(rows, material, name, data, declarations, abi_spans):
    """Return verified source replacements and their guarded helper definitions."""
    spans, definitions, decisions = [], [], []
    exports = {row["symbol"]: row for row in declarations}
    digest = sha256(data).hexdigest()
    for row in rows:
        witness = row["fragment"]
        if witness.get("authority") == "TYPED_ABI_EXPORT":
            start, stop = _abi_span(row, exports, abi_spans)
        else:
            start, stop = witness["span"]
            if witness["sha256"] != digest or sha256(data[start:stop]).hexdigest() != witness["slice_sha256"]:
                raise ValueError(f"native region source witness changed: {name}:{start}")
        original = data[start:stop]
        if (witness["path"] != name or not (0 <= start < stop <= len(data))
                or not original.startswith(b"{") or not original.endswith(b"}")):
            raise ValueError(f"native region source witness changed: {name}:{start}")
        if not _strict_source(material, name):
            decisions.append(dict(region=row["region"], status="ORIGINAL_FLOATING_PROFILE_UNPROVED"))
            continue
        symbol = "scar_try_" + sha256(f"{name}:{digest}:{start}".encode()).hexdigest()[:20]
        replacement, definition = _region(row, symbol, original)
        spans.append((start, stop, replacement))
        definitions.append(definition)
        decisions.append(dict(region=row["region"], status="GUARDED_REGION", helper=symbol))
    return spans, definitions, decisions


def emit_region_helpers(project, configuration, definitions):
    if not definitions:
        return
    path = Path("src/scar_native_regions.cpp")
    with (project / path).open("x", encoding="utf-8") as stream:
        stream.write(_GUARD + "\n\n".join(definitions) + "\n")
    configuration["sources"] = [*configuration["sources"], path.as_posix()]
    configuration.setdefault("source_settings", {})[path.as_posix()] = ((), _STRICT_FP)


_GUARD = """#include <cfloat>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {
template<std::size_t N>
bool admit(int length, int maximum, const double* const (&arrays)[N]) noexcept {
    if (length <= 0 || length > maximum || std::fegetround() != FE_TONEAREST) return false;
#if defined(__GLIBC__) && defined(__USE_GNU)
    if (::fegetexcept() != 0) return false;
#else
    return false;
#endif
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559 && FLT_EVAL_METHOD == 0);
    static_assert(std::numeric_limits<int>::digits == 31);
    constexpr auto limit = std::numeric_limits<std::uintptr_t>::max();
    const auto count = static_cast<std::uintptr_t>(length);
    if (count > limit / sizeof(double)) return false;
    const auto bytes = count * sizeof(double);
    std::uintptr_t starts[N];
    for (std::size_t i = 0; i < N; ++i) {
        const auto start = reinterpret_cast<std::uintptr_t>(arrays[i]);
        if (!arrays[i] || start > limit - bytes) return false;
        for (std::size_t j = 0; j < i; ++j) {
            if (start < starts[j] + bytes && starts[j] < start + bytes) return false;
        }
        starts[i] = start;
    }
    return true;
}
}

"""
