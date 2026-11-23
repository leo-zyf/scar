"""Print typed C++ target trees without inspecting program IR or planning work."""

from .model import (
    CASTS,
    FUNCTION_DEFINITIONS,
    TYPE_WRAPPERS,
    Decl,
    Expr,
    Stmt,
    Type,
    validate_node,
    validate_unit,
)


def _quoted(value):
    result = []
    escapes = {34: '\\"', 92: "\\\\", 10: "\\n", 13: "\\r", 9: "\\t"}
    for byte in value.encode("utf-8"):
        result.append(
            escapes.get(byte, chr(byte) if 32 <= byte < 127 else f"\\{byte:03o}")
        )
    return '"' + "".join(result) + '"'


def _type_name(node):
    kind = node.kind
    if kind in {"name", "auto"}:
        name = "auto" if kind == "auto" else node.name
        return name
    if kind == "template":
        return node.name + "<" + ", ".join(_argument(a) for a in node.args) + ">"
    if kind == "member":
        return "typename " + _type_name(node.parent) + "::" + node.name
    if kind == "tagged":
        return "struct " + node.name
    if kind in TYPE_WRAPPERS:
        inner = _type_name(node.element)
        if "auto" in inner.split():
            if kind in {"const", "volatile"}:
                return kind + " " + inner
            return inner + {"ptr": " *", "ref": " &", "rref": " &&"}[kind]
        return f"std::{TYPE_WRAPPERS[kind]}<{inner}>"
    if kind == "array":
        size = _expression(node.extent) if node.extent is not None else ""
        return f"std::type_identity_t<{_type_name(node.element)}[{size}]>"
    if kind == "function":
        return (
            "std::type_identity_t<"
            + _type_name(node.result)
            + "("
            + ", ".join(_type_name(t) for t in node.params)
            + ")>"
        )
    if kind == "decltype":
        return "decltype(" + _expression(node.args[0]) + ")"
    raise ValueError(f"unsupported C++ target type {kind!r}")


def _argument(node):
    return _type_name(node) if isinstance(node, Type) else _expression(node)


def _expression(node):
    kind, args = node.kind, node.args
    if kind == "name":
        return node.name
    if kind == "literal":
        value = node.value
        if value is None:
            return "nullptr"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        return value.hex() if isinstance(value, float) else value
    if kind == "string":
        return _quoted(node.value)
    if kind == "nullptr":
        return "nullptr"
    if kind == "this":
        return "this"
    if kind == "template_name":
        return node.name + "<" + ", ".join(_argument(a) for a in args) + ">"
    if kind == "type_member":
        return _type_name(node.type) + "::" + node.name
    if kind in {"braced", "init_list"}:
        return "{" + ", ".join(_argument(a) for a in args) + "}"
    if kind == "construct":
        opening, closing = (
            ("{", "}") if node.qualifier in {"brace", "list"} else ("(", ")")
        )
        return (
            _type_name(node.type)
            + opening
            + ", ".join(_argument(a) for a in args)
            + closing
        )
    if kind == "call":
        return (
            _expression(args[0]) + "(" + ", ".join(_argument(a) for a in args[1:]) + ")"
        )
    if kind == "member":
        return (
            "(" + _expression(args[0]) + ")" + ("->" if node.arrow else ".") + node.name
        )
    if kind == "subscript":
        return (
            "("
            + _expression(args[0])
            + ")"
            + "".join("[" + _expression(a) + "]" for a in args[1:])
        )
    if kind in {"unary", "binary"}:
        operands = [_expression(a) for a in args]
        if kind == "binary":
            return f"({operands[0]} {node.operator} {operands[1]})"
        return (
            f"({operands[0]}{node.operator})"
            if node.postfix
            else f"({node.operator}{operands[0]})"
        )
    if kind == "cast":
        mode = node.qualifier or node.operator or "static"
        if mode == "explicit_pointer":
            return f"(({_type_name(node.type)})({_expression(args[0])}))"
        return f"{CASTS[mode]}<{_type_name(node.type)}>({_expression(args[0])})"
    if kind == "conditional":
        return f"({_expression(args[0])} ? {_expression(args[1])} : {_expression(args[2])})"
    if kind == "new":
        suffix = "[" + _expression(node.extent) + "]" if node.extent is not None else ""
        init = ""
        if node.qualifier != "default":
            opening, closing = ("{", "}") if node.qualifier == "brace" else ("(", ")")
            init = opening + ", ".join(_argument(a) for a in args) + closing
        return "new " + _type_name(node.type) + suffix + init
    if kind == "delete":
        return (
            "delete" + ("[] " if node.array else " ") + "(" + _expression(args[0]) + ")"
        )
    if kind == "lambda":
        captures = []
        for capture in node.captures:
            if isinstance(capture, tuple):
                captures.append(capture[0] + " = " + _expression(capture[1]))
            elif capture in {"&", "=", "this", "*this"}:
                captures.append(capture)
            else:
                captures.append(
                    ("&" if capture.startswith("&") else "") + capture.removeprefix("&")
                )
        params = ", ".join(_declaration(p, parameter=True) for p in node.params)
        qualifier = " mutable" if node.qualifier == "mutable" else ""
        result = " -> " + _type_name(node.type) if node.type is not None else ""
        return (
            "["
            + ", ".join(captures)
            + "]("
            + params
            + ")"
            + qualifier
            + result
            + " "
            + _block(node.body, 0)
        )
    if kind in {"sizeof_type", "alignof"}:
        return (
            ("sizeof" if kind == "sizeof_type" else "alignof")
            + "("
            + _type_name(node.type)
            + ")"
        )
    if kind == "sizeof_expr":
        return "sizeof(" + _expression(args[0]) + ")"
    if kind == "offsetof":
        return "offsetof(" + _type_name(node.type) + ", " + node.name + ")"
    if kind == "offset_check":
        return (
            "SCAR_OFFSET_MATCH("
            + _type_name(node.type)
            + ", "
            + node.name
            + ", "
            + str(int(node.value))
            + ")"
        )
    raise ValueError(f"unsupported C++ target expression {kind!r}")


def _block(body, depth):
    lines = ["{"]
    lines.extend(_statement(s, depth + 1) for s in body)
    lines.append("  " * depth + "}")
    return "\n".join(lines)


def _statement(node, depth=0):
    indent, kind = "  " * depth, node.kind
    if kind in {"expression", "expr"}:
        text = _expression(node.value) + ";"
    elif kind == "return":
        text = (
            "return"
            + (" " + _expression(node.value) if node.value is not None else "")
            + ";"
        )
    elif kind == "delete":
        text = (
            _expression(node.value)
            if node.value.kind == "delete"
            else "delete " + _expression(node.value)
        ) + ";"
    elif kind == "declaration":
        text = _declaration(node.decl)
    elif kind == "block":
        text = _block(node.body, depth)
    elif kind == "throw":
        text = (
            "throw"
            + (" " + _expression(node.value) if node.value is not None else "")
            + ";"
        )
    elif kind == "try":
        text = "try " + _block(node.body, depth)
        for parameter, body in node.handlers:
            target = (
                _declaration(parameter, parameter=True)
                if parameter is not None
                else "..."
            )
            text += " catch (" + target + ") " + _block(body, depth)
    elif kind == "if":
        text = "if (" + _expression(node.condition) + ") " + _block(node.body, depth)
        if node.else_body:
            text += " else " + _block(node.else_body, depth)
    elif kind == "for":
        initial = (
            _statement(node.init).removesuffix(";")
            if isinstance(node.init, Stmt)
            else _expression(node.init) if node.init is not None else ""
        )
        condition = _expression(node.condition) if node.condition is not None else ""
        step = _expression(node.step) if node.step is not None else ""
        text = f"for ({initial}; {condition}; {step}) " + _block(node.body, depth)
    elif kind == "each":
        text = (
            "for ("
            + _declaration(node.decl, parameter=True)
            + " : "
            + _expression(node.value)
            + ") "
            + _block(node.body, depth)
        )
    elif kind in {"while", "switch"}:
        text = (
            kind + " (" + _expression(node.condition) + ") " + _block(node.body, depth)
        )
    elif kind == "do_while":
        text = (
            "do "
            + _block(node.body, depth)
            + " while ("
            + _expression(node.condition)
            + ");"
        )
    elif kind in {"break", "continue"}:
        text = kind + ";"
    elif kind == "goto":
        text = "goto " + node.name + ";"
    elif kind == "label":
        text = node.name + ":;"
    elif kind == "case":
        text = "case " + _expression(node.value) + ":"
    elif kind == "default":
        text = "default:"
    elif kind == "pragma":
        clauses = []
        for name, args in node.clauses:
            clauses.append(name + "(" + ", ".join(_argument(a) for a in args) + ")")
        text = (
            "#pragma " + node.directive + (" " + " ".join(clauses) if clauses else "")
        )
    else:
        raise ValueError(f"unsupported C++ target statement {kind!r}")
    return indent + text


def _declaration(node, parameter=False):
    kind = node.kind
    if kind == "param" and node.definition == "variadic":
        return "..."
    attributes = ""
    if node.attributes:
        names = []
        for attribute in node.attributes:
            name = attribute.name
            if attribute.args:
                name += "(" + ", ".join(_argument(a) for a in attribute.args) + ")"
            names.append(name)
        attributes = "[[" + ", ".join(names) + "]] "
    storage = (" ".join(node.storage) + " ") if node.storage else ""
    alignment = ""
    if node.alignment is not None:
        alignment = f"alignas({node.alignment}) "
    if kind in {"variable", "field", "param"}:
        text = attributes + alignment + storage + _type_name(node.type)
        if node.name:
            text += " " + node.name
        if node.initializer is not None:
            value = _expression(node.initializer)
            text += (
                "(" + value + ")" if "direct_init" in node.qualifiers else " = " + value
            )
        if parameter or kind == "param":
            return text
        text += ";"
        return 'extern "C" { ' + text + " }" if node.linkage == "C" else text
    if kind == "alias":
        return "using " + node.name + " = " + _type_name(node.type) + ";"
    if kind == "static_assert":
        return "static_assert(" + _expression(node.initializer) + ");"
    if kind in {"forward", "record"}:
        keyword = node.definition or "struct"
        text = keyword + " " + alignment + node.name
        if kind == "forward":
            return text + ";"
        if node.bases:
            text += " : " + ", ".join(
                access + " " + ("virtual " if virtual else "") + _type_name(type_)
                for access, virtual, type_ in node.bases
            )
        return (
            text
            + " {\n"
            + "\n".join(
                "  " + _declaration(d).replace("\n", "\n  ") for d in node.members
            )
            + "\n};"
        )
    if kind == "function":
        prefix = 'extern "C" ' if node.linkage == "C" else ""
        result = _type_name(node.type) + " " if node.type is not None else ""
        text = (
            prefix
            + attributes
            + storage
            + result
            + node.name
            + "("
            + ", ".join(_declaration(p, parameter=True) for p in node.params)
            + ")"
        )
        if node.qualifiers:
            text += " " + " ".join(node.qualifiers)
        if node.definition is not None:
            suffix = FUNCTION_DEFINITIONS[node.definition]
            return text + " = " + suffix + ";"
        if node.initializers:
            text += " : " + ", ".join(
                name + "(" + ", ".join(_argument(a) for a in args) + ")"
                for name, args in node.initializers
            )
        return text + (" " + _block(node.body, 0) if node.body is not None else ";")
    if kind == "namespace":
        return (
            "namespace "
            + node.name
            + " {\n"
            + "\n".join(_declaration(d) for d in node.members)
            + "\n}"
        )
    raise ValueError(f"unsupported C++ target declaration {kind!r}")


def print_unit(unit):
    validate_unit(unit)
    lines = ["#pragma once"] if unit.once else []
    for kind, path in unit.includes:
        lines.append(
            "#include " + ("<" + path + ">" if kind == "system" else '"' + path + '"')
        )
    lines.extend(["", *[_declaration(d) for d in unit.declarations]])
    return "\n\n".join(lines) + "\n"


def type_name(node):
    return _type_name(validate_node(node, Type))


def expression(node):
    return _expression(validate_node(node, Expr))


def statement(node, depth=0):
    return _statement(validate_node(node, Stmt), depth)


def declaration(node, parameter=False):
    if parameter and (
        not isinstance(node, Decl)
        or node.kind not in {"param", "variable"}
        or node.linkage
    ):
        raise ValueError("invalid parameter declaration")
    return _declaration(validate_node(node, Decl), parameter)
