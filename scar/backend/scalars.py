"""Generate C++20 definitions for explicitly bound scalar IR functions."""

from pathlib import Path
import re


def emit_scalar_module(body, project, configuration):
    from .lowering import _expression

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
            return _expression(expression, slots, symbols)
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
        return _expression([*head, *args_read], slots, symbols)

    for function in body.functions:
        add_symbol(function["name"], len(function["parameters"]))
        visit(function["result"])
    lines = [f'extern "C" double {name}({", ".join(["double"] * arity)});'
             for name, arity in symbols.values()]
    for function in body.functions:
        slots = {key: f"v{index}" for index, key in enumerate(function["parameters"])}
        if len(slots) != len(function["parameters"]):
            raise ValueError("duplicate scalar parameter")
        parameters = ", ".join("double " + slot for slot in slots.values())
        local = []
        value = result(function["result"], slots, local)
        lines.extend((f'extern "C" double {function["name"]}({parameters}) {{',
                      *local, f"    return {value};", "}"))
    source = Path("src/scar_scalar_abi.cpp")
    destination = project / source
    with destination.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    configuration["sources"] = [*configuration["sources"], source.as_posix()]
    configuration.setdefault("source_settings", {})[source.as_posix()] = (
        (), ("-fno-fast-math", "-ffp-contract=off", "-frounding-math", "-ftrapping-math")
    )
