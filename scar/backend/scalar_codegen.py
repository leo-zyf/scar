"""Emit closed scalar expressions and scalar function bodies as C++20."""

import math
from pathlib import Path
import re

_BINARY = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


def render_expression(row, slots, symbols):
    kind, *args = row
    if kind == "read":
        return slots[args[0]]
    if kind == "constant":
        value = float(args[0])
        if not math.isfinite(value):
            raise ValueError("portable constants must be finite")
        return value.hex()
    if kind == "neg":
        return "(-" + render_expression(args[0], slots, symbols) + ")"
    if kind in _BINARY:
        left, right = (render_expression(arg, slots, symbols) for arg in args)
        return f"({left} {_BINARY[kind]} {right})"
    if kind == "call":
        name, arity = symbols[args[0]]
        if len(args) != arity + 1:
            raise ValueError("portable call does not match its signature")
        return (
            name
            + "("
            + ", ".join(render_expression(arg, slots, symbols) for arg in args[1:])
            + ")"
        )
    raise ValueError(f"unsupported portable expression {kind}")


def emit_scalar_module(body, project, configuration):
    symbols = {}

    def add_symbol(name, arity):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError(f"invalid scalar symbol {name!r}")
        if symbols.setdefault(name, (name, arity)) != (name, arity):
            raise ValueError(f"inconsistent scalar ABI for {name}")

    def visit(expression):
        kind, *args = expression
        if kind == "call":
            add_symbol(args[0], len(args) - 1)
            for arg in args[1:]:
                visit(arg)
        elif kind in {"add", "sub", "mul", "neg"}:
            for arg in args:
                visit(arg)
        elif kind not in {"read", "constant"}:
            raise ValueError(f"unsupported scalar operation {kind}")

    def result(expression, slots, code):
        kind, *args = expression
        if kind in {"read", "constant"}:
            return render_expression(expression, slots, symbols)
        operands = args[1:] if kind == "call" else args
        args_read = []
        for operand in operands:
            value = result(operand, slots, code)
            variable = f"t{len(code)}"
            code.append(f"    const double {variable} = {value};")
            key = (variable,)
            slots[key] = variable
            args_read.append(["read", key])
        head = [kind, args[0]] if kind == "call" else [kind]
        return render_expression([*head, *args_read], slots, symbols)

    for function in body.functions:
        add_symbol(function["name"], len(function["parameters"]))
        visit(function["result"])
    lines = [
        f'extern "C" double {name}({", ".join(["double"] * arity)});'
        for name, arity in symbols.values()
    ]
    for function in body.functions:
        slots = {key: f"v{index}" for index, key in enumerate(function["parameters"])}
        if len(slots) != len(function["parameters"]):
            raise ValueError("duplicate scalar parameter")
        parameters = ", ".join("double " + slot for slot in slots.values())
        local = []
        value = result(function["result"], slots, local)
        lines.extend(
            (
                f'extern "C" double {function["name"]}({parameters}) {{',
                *local,
                f"    return {value};",
                "}",
            )
        )
    source = Path("src/scar_scalar_abi.cpp")
    destination = project / source
    with destination.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    configuration["sources"] = [*configuration["sources"], source.as_posix()]
    configuration.setdefault("source_settings", {})[source.as_posix()] = (
        (),
        ("-fno-fast-math", "-ffp-contract=off", "-frounding-math", "-ftrapping-math"),
    )
