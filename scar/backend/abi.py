"""Render C++20 ABI bridges from typed records."""

import re

_CALL_WRAPPER = """#pragma once

template<class Result, class... Args>
static inline Result scar_f2c_call(Result (*callee)(...), Args... args) {
    return reinterpret_cast<Result (*)(Args...)>(callee)(args...);
}
"""
_CFI_ELEMENT = """
#include <cstdint>

using longint = std::int64_t;

// The companion CFI ABI describes byte addresses of live scalar objects.
template<class T>
static inline T* scar_cfi_element(T* base, longint byte_offset) {
    return reinterpret_cast<T*>(reinterpret_cast<std::uintptr_t>(base) +
                                static_cast<std::uintptr_t>(byte_offset));
}
"""
_CFI_VIEW = """static_assert(CFI_VERSION == 1);
static_assert(std::is_same_v<CFI_index_t, longint>);
static_assert(sizeof(std::uintptr_t) == sizeof(longint));

struct ScarCFIAxis {
    longint extent, stride, origin, lower, upper;
};

template<class T, std::size_t Rank>
struct ScarCFIView {
    T* data;
    std::array<ScarCFIAxis, Rank> axes;
};

template<class T, CFI_type_t Type, std::size_t Rank>
static ScarCFIView<T, Rank> scar_cfi_view(CFI_cdesc_t* descriptor,
                                        std::array<longint, Rank> origins) {
    static_assert(0 < Rank && Rank <= CFI_MAX_RANK);
    if (!descriptor || descriptor->version != CFI_VERSION || descriptor->rank != Rank ||
        descriptor->type != Type || descriptor->elem_len != sizeof(T) ||
        descriptor->attribute != CFI_attribute_other) std::abort();

    ScarCFIView<T, Rank> result{};
    result.data = static_cast<T*>(descriptor->base_addr);
    bool empty = false;
    for (std::size_t i = 0; i < Rank; ++i) {
        const CFI_dim_t& dim = descriptor->dim[i];
        if (dim.lower_bound != 0 || dim.extent < 0 || dim.sm % longint(sizeof(T))) std::abort();
        ScarCFIAxis& axis = result.axes[i];
        axis.extent = dim.extent;
        axis.stride = dim.sm;
        axis.origin = origins[i];
        if (dim.extent == 0) {
            empty = true;
            axis.lower = 1;
            axis.upper = 0;
        } else {
            axis.lower = axis.origin;
            if (__builtin_add_overflow(axis.origin, dim.extent - 1, &axis.upper)) std::abort();
        }
    }

    if (!empty) {
        longint min_offset = 0, max_offset = 0;
        for (const auto& axis : result.axes) {
            longint displacement;
            if ((axis.extent > 1 && axis.stride == 0) ||
                __builtin_mul_overflow(axis.extent - 1, axis.stride, &displacement)) std::abort();
            longint& sum = displacement < 0 ? min_offset : max_offset;
            if (__builtin_add_overflow(sum, displacement, &sum)) std::abort();
        }
        const auto base = reinterpret_cast<std::uintptr_t>(result.data);
        const auto available = (std::numeric_limits<std::uintptr_t>::max)() - base;
        const auto negative = std::uintptr_t(0) - static_cast<std::uintptr_t>(min_offset);
        // The caller establishes accessible backing objects and their lifetime.
        // Check the representable byte envelope before evaluating any element.
        if (!base || base % alignof(T) || negative > base || max_offset == (std::numeric_limits<longint>::max)() ||
            static_cast<std::uintptr_t>(max_offset) > available ||
            sizeof(T) - 1 > available - static_cast<std::uintptr_t>(max_offset)) std::abort();
    }
    return result;
}
"""
_CFI_PACKING = """template<class T, std::size_t Rank>
class ScarCFIPacking {
public:
    explicit ScarCFIPacking(ScarCFIView<T, Rank>& view) {
        for (const auto& axis : view.axes) {
            if (axis.extent == 0) return;
        }

        // GNU requires the exact packed stride at every axis, even singletons.
        std::array<longint, Rank> packed_strides;
        longint bytes = sizeof(T);
        bool packed = true;
        for (std::size_t i = 0; i < Rank; ++i) {
            packed_strides[i] = bytes;
            packed &= view.axes[i].stride == bytes;
            if (__builtin_mul_overflow(bytes, view.axes[i].extent, &bytes)) std::abort();
        }
        if (packed) return;

        count = bytes / sizeof(T);
        buffer.reset(new (std::nothrow) T[count]);
        if (!buffer) std::abort();
        source = view;
        // GNU's BIND(C) prologue copies the bytes in for all INTENT modes.
        transfer<true>();
        view.data = buffer.get();
        for (std::size_t i = 0; i < Rank; ++i) view.axes[i].stride = packed_strides[i];
    }

    void copy_back() {
        if (buffer) transfer<false>();
    }

private:
    template<bool CopyIn>
    void transfer() {
        for (longint i = 0; i < count; ++i) {
            longint linear = i;
            longint offset = 0;
            for (const auto& axis : source.axes) {
                offset += (linear % axis.extent) * axis.stride;
                linear /= axis.extent;
            }
            T* element = scar_cfi_element(source.data, offset);
            T* packed = buffer.get() + i;
            if constexpr (CopyIn) std::memcpy(packed, element, sizeof(T));
            else std::memcpy(element, packed, sizeof(T));
        }
    }

    ScarCFIView<T, Rank> source;
    std::unique_ptr<T[]> buffer;
    longint count = 0;
};
"""
_CFI_AXIS_FIELDS = ("extent", "stride", "origin", "lower", "upper")


_ABI_ARGUMENT_TYPES = {"int": "integer", "float": "real", "double": "doublereal", "char": "char"}
_ABI_RESULT_TYPES = {**_ABI_ARGUMENT_TYPES, "float": "doublereal"}
_ABI_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def f2c_support_header(*, descriptors: bool) -> bytes:
    """Return the C++20 f2c call helpers for the admitted descriptor profile."""
    if type(descriptors) is not bool:
        raise ValueError("explicit Fortran descriptor profile required")
    return (_CALL_WRAPPER + (_CFI_ELEMENT if descriptors else "")).encode()


def _abi_record(record, types):
    if (
        type(record) is not dict
        or type(record.get("c_type")) is not str
        or record["c_type"] not in types
        or record.get("transport_type") != types[record["c_type"]]
    ):
        raise ValueError("unsupported Fortran ABI scalar type")


def _abi_descriptor(argument):
    descriptor = argument.get("descriptor")
    if descriptor is None and "descriptor" not in argument:
        return
    if type(descriptor) is not dict or argument["by_value"]:
        raise ValueError("invalid Fortran descriptor argument")
    rank = descriptor.get("rank")
    bounds = descriptor.get("lower_bounds")
    contiguous = descriptor.get("contiguous", False)
    if (
        type(rank) is not int
        or not 1 <= rank <= 15
        or type(bounds) not in (tuple, list)
        or len(bounds) != rank
        or any(
            type(bound) is not int or not -(2**63) <= bound < 2**63 for bound in bounds
        )
        or descriptor.get("cfi_type") != f"CFI_type_{argument['c_type']}"
        or type(contiguous) is not bool
        or contiguous
        and descriptor.get("intent") not in ("in", "out", "inout")
    ):
        raise ValueError("unsupported Fortran descriptor layout")


def _abi_exports(exports):
    if type(exports) not in (tuple, list):
        raise ValueError("Fortran ABI exports must be ordered records")
    exports = tuple(exports)
    symbols = set()
    for row in exports:
        if type(row) is not dict:
            raise ValueError("invalid Fortran ABI export")
        symbol, transport = row.get("symbol"), row.get("transport")
        if (
            any(
                type(name) is not str or not _ABI_IDENTIFIER.fullmatch(name)
                for name in (symbol, transport)
            )
            or symbol == transport
            or symbol in symbols
        ):
            raise ValueError("invalid or repeated Fortran ABI symbol")
        symbols.add(symbol)
        if "result" not in row or type(row.get("arguments")) not in (tuple, list):
            raise ValueError("missing Fortran ABI signature")
        result = row["result"]
        if result is not None:
            _abi_record(result, _ABI_RESULT_TYPES)
        names, locals_ = set(), set()
        for index, arg in enumerate(row["arguments"]):
            _abi_record(arg, _ABI_ARGUMENT_TYPES)
            name = arg.get("wrapper_name")
            if (
                type(name) is not str
                or not _ABI_IDENTIFIER.fullmatch(name)
                or name in names
                or type(arg.get("by_value")) is not bool
                or type(arg.get("read_only")) is not bool
            ):
                raise ValueError("invalid Fortran ABI argument")
            names.add(name)
            if "character_length" in arg and (
                arg["c_type"] != "char" or arg["character_length"] != "size_t" or arg["by_value"]
            ):
                raise ValueError("invalid character argument length")
            _abi_descriptor(arg)
            if "descriptor" in arg:
                locals_.add(f"view{index}")
                if arg["descriptor"].get("contiguous"):
                    locals_.add(f"packing{index}")
                    if result is not None and arg["descriptor"]["intent"] != "in":
                        locals_.add("result")
        if names & locals_:
            raise ValueError("Fortran ABI argument collides with a generated local")
    if any(row["transport"] in symbols for row in exports):
        raise ValueError("Fortran transport collides with an exported symbol")
    return exports


def render_abi_wrappers(exports):
    """Render generated C ABI exports and return exact function body byte ranges."""
    exports = _abi_exports(exports)
    descriptors = [
        arg["descriptor"]
        for row in exports
        for arg in row["arguments"]
        if "descriptor" in arg
    ]
    has_packing = any(row.get("contiguous") for row in descriptors)
    lines = ["#include <type_traits>", "#include <cstddef>", "#include <limits>", "#include <stdexcept>"]
    if descriptors:
        lines.extend(
            [
                "#include <ISO_Fortran_binding.h>",
                "#include <array>",
                "#include <cstdlib>",
                "#include <limits>",
                '#include "scar_f2c_call.hpp"',
            ]
        )
    if has_packing:
        lines.extend(["#include <cstring>", "#include <memory>", "#include <new>"])
    lines.extend(
        [
            '#include "f2c.h"',
            "static_assert(std::is_same_v<integer, int>);",
            "static_assert(std::is_same_v<doublereal, double>);",
        ]
    )
    if any(
        argument["transport_type"] == "real"
        for row in exports
        for argument in row["arguments"]
    ):
        lines.append("static_assert(std::is_same_v<real, float>);")
    if descriptors:
        lines.append(_CFI_VIEW)
    if has_packing:
        lines.append(_CFI_PACKING)
    header = ("\n".join(lines) + "\n").encode()
    chunks, spans, position = [header], {}, len(header)
    for row in exports:
        result = row["result"]
        native_return = result["c_type"] if result else "void"
        transport_return = result["transport_type"] if result else "int"
        types, parameters, arguments, checks = [], [], [], []
        packings, copy_backs = [], []
        for index, argument in enumerate(row["arguments"]):
            name = argument["wrapper_name"]
            types.append(f'{argument["transport_type"]} *')
            if descriptor := argument.get("descriptor"):
                parameters.append(f"CFI_cdesc_t *{name}")
                view = f"view{index}"
                lower = ", ".join(map(str, descriptor["lower_bounds"]))
                template = f'{argument["c_type"]}, {descriptor["cfi_type"]}, {descriptor["rank"]}'
                checks.append(
                    f"    auto {view} = scar_cfi_view<{template}>({name}, {{{lower}}});"
                )
                if descriptor.get("contiguous"):
                    packing = f"packing{index}"
                    packings.append(f"    ScarCFIPacking {packing}({view});")
                    if descriptor["intent"] != "in":
                        copy_backs.append(f"    {packing}.copy_back();")
                arguments.append(f"{view}.data")
                for axis in range(descriptor["rank"]):
                    for field in _CFI_AXIS_FIELDS:
                        types.append("longint *")
                        arguments.append(f"&{view}.axes[{axis}].{field}")
            elif argument["by_value"]:
                parameters.append(f'{argument["c_type"]} {name}')
                arguments.append("&" + name)
            elif argument["read_only"]:
                parameters.append(f'const {argument["c_type"]} *{name}')
                arguments.append(f'const_cast<{argument["c_type"]} *>({name})')
            else:
                parameters.append(f'{argument["c_type"]} *{name}')
                arguments.append(name)
        for argument in row["arguments"]:
            if "character_length" in argument:
                name = argument["wrapper_name"] + "_length"
                types.append("ftnlen")
                parameters.append(f"std::size_t {name}")
                checks.append(
                    f"    if ({name} > static_cast<std::size_t>((std::numeric_limits<ftnlen>::max)())) "
                    'throw std::length_error("Fortran character length exceeds runtime representation");'
                )
                arguments.append(f"static_cast<ftnlen>({name})")
        prefix = (
            f'extern "C" {transport_return} {row["transport"]}({", ".join(types)});\n'
            f'extern "C" {native_return} {row["symbol"]}({", ".join(parameters)}) '
        ).encode()
        body = ["{", *checks, *packings]
        call = f'{row["transport"]}({", ".join(arguments)});'
        if result and copy_backs:
            body.append("    const auto result = " + call)
            body.extend(copy_backs)
            body.append("    return result;")
        else:
            body.append("    " + ("return " if result else "") + call)
            body.extend(copy_backs)
        compound = "\n".join((*body, "}")).encode()
        start = position + len(prefix)
        spans[row["symbol"]] = (start, start + len(compound))
        chunks.extend((prefix, compound, b"\n"))
        position = start + len(compound) + 1
    return b"".join(chunks), spans
