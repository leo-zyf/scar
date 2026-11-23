"""Print typed C++ target trees without inspecting program IR or planning work."""

import math
import re

from .model import Attribute, Decl, Expr, Stmt, Type, Unit

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
_NUMBER = re.compile(
    r"[+-]?(?:0[xX][0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?[pP][+-]?[0-9]+|(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|0[xX][0-9a-fA-F]+|0[bB][01]+)[uUlLfF]*\Z"
)
_OPERATORS = set(
    "+ - * / % & | ^ && || == != < > <= >= << >> = += -= *= /= %= &= |= ^= <<= >>= , ! ~ ++ --".split()
)
_BUILTINS = {
    "void",
    "auto",
    "bool",
    "char",
    "signed char",
    "unsigned char",
    "short",
    "unsigned short",
    "int",
    "unsigned",
    "unsigned int",
    "long",
    "unsigned long",
    "long long",
    "unsigned long long",
    "float",
    "double",
    "long double",
    "wchar_t",
    "char8_t",
    "char16_t",
    "char32_t",
}
_STORAGE = {
    "extern",
    "static",
    "inline",
    "constexpr",
    "consteval",
    "constinit",
    "thread_local",
    "mutable",
    "virtual",
    "explicit",
}
_QUALIFIERS = {
    "const",
    "volatile",
    "noexcept",
    "noexcept(true)",
    "noexcept(false)",
    "override",
    "final",
    "&",
    "&&",
}


def _name(value):
    if not isinstance(value, str) or not value:
        raise ValueError("C++ name must be a nonempty identifier")
    parts = value.removeprefix("::").split("::")
    for part in parts:
        if (
            _IDENTIFIER.fullmatch(part)
            or part.startswith("~")
            and _IDENTIFIER.fullmatch(part[1:])
        ):
            continue
        if part.startswith("operator") and part[8:] in _OPERATORS | {
            "()",
            "[]",
            "->",
            "bool",
        }:
            continue
        raise ValueError(f"invalid target identifier {value!r}")
    return value


def _quoted(value):
    if not isinstance(value, str):
        raise TypeError("C++ string literal requires decoded string data")
    result = []
    escapes = {34: '\\"', 92: "\\\\", 10: "\\n", 13: "\\r", 9: "\\t"}
    for byte in value.encode("utf-8"):
        result.append(
            escapes.get(byte, chr(byte) if 32 <= byte < 127 else f"\\{byte:03o}")
        )
    return '"' + "".join(result) + '"'


def type_name(node):
    if not isinstance(node, Type):
        raise TypeError("C++ type position requires a Type node")
    kind = node.kind
    if kind in {"name", "auto"}:
        name = "auto" if kind == "auto" else node.name
        return name if name in _BUILTINS else _name(name)
    if kind == "template":
        return _name(node.name) + "<" + ", ".join(_argument(a) for a in node.args) + ">"
    if kind == "member":
        return "typename " + type_name(node.parent) + "::" + _name(node.name)
    if kind == "tagged":
        return "struct " + _name(node.name)
    wrappers = {
        "ptr": "add_pointer_t",
        "ref": "add_lvalue_reference_t",
        "rref": "add_rvalue_reference_t",
        "const": "add_const_t",
        "volatile": "add_volatile_t",
    }
    if kind in wrappers:
        inner = type_name(node.element)
        if "auto" in inner.split():
            if kind in {"const", "volatile"}:
                return kind + " " + inner
            return inner + {"ptr": " *", "ref": " &", "rref": " &&"}[kind]
        return f"std::{wrappers[kind]}<{inner}>"
    if kind == "array":
        size = expression(node.extent) if node.extent is not None else ""
        return f"std::type_identity_t<{type_name(node.element)}[{size}]>"
    if kind == "function":
        return (
            "std::type_identity_t<"
            + type_name(node.result)
            + "("
            + ", ".join(type_name(t) for t in node.params)
            + ")>"
        )
    if kind == "decltype":
        return "decltype(" + expression(node.args[0]) + ")"
    raise ValueError(f"unsupported C++ target type {kind!r}")


def _argument(node):
    return type_name(node) if isinstance(node, Type) else expression(node)


def expression(node):
    if not isinstance(node, Expr):
        raise TypeError("C++ expression position requires an Expr node")
    kind, args = node.kind, node.args
    if kind == "name":
        return _name(node.name)
    if kind == "literal":
        value = node.value
        if value is None:
            return "nullptr"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float) and math.isfinite(value):
            return value.hex()
        if isinstance(value, str) and _NUMBER.fullmatch(value):
            return value
        raise ValueError(
            "target literal requires a numeric token or structured bit representation"
        )
    if kind == "string":
        return _quoted(node.value)
    if kind == "nullptr":
        return "nullptr"
    if kind == "this":
        return "this"
    if kind == "template_name":
        return _name(node.name) + "<" + ", ".join(_argument(a) for a in args) + ">"
    if kind == "type_member":
        return type_name(node.type) + "::" + _name(node.name)
    if kind in {"braced", "init_list"}:
        return "{" + ", ".join(_argument(a) for a in args) + "}"
    if kind == "construct":
        opening, closing = (
            ("{", "}") if node.qualifier in {"brace", "list"} else ("(", ")")
        )
        return (
            type_name(node.type)
            + opening
            + ", ".join(_argument(a) for a in args)
            + closing
        )
    if kind == "call":
        return (
            expression(args[0]) + "(" + ", ".join(_argument(a) for a in args[1:]) + ")"
        )
    if kind == "member":
        return (
            "("
            + expression(args[0])
            + ")"
            + ("->" if node.arrow else ".")
            + _name(node.name)
        )
    if kind == "subscript":
        return (
            "("
            + expression(args[0])
            + ")"
            + "".join("[" + expression(a) + "]" for a in args[1:])
        )
    if kind in {"unary", "binary"}:
        if node.operator not in _OPERATORS:
            raise ValueError(f"invalid target operator {node.operator!r}")
        operands = [expression(a) for a in args]
        if kind == "binary" and len(operands) == 2:
            return f"({operands[0]} {node.operator} {operands[1]})"
        if kind == "unary" and len(operands) == 1:
            return (
                f"({operands[0]}{node.operator})"
                if node.postfix
                else f"({node.operator}{operands[0]})"
            )
        raise ValueError("invalid target operator arity")
    if kind == "cast":
        mode = node.qualifier or node.operator or "static"
        if mode == "explicit_pointer":
            return f"(({type_name(node.type)})({expression(args[0])}))"
        target = {
            "static": "static_cast",
            "qualification": "const_cast",
            "const": "const_cast",
            "reinterpret": "reinterpret_cast",
            "dynamic": "dynamic_cast",
        }.get(mode)
        if target is None:
            raise ValueError(f"invalid C++ cast {mode!r}")
        return f"{target}<{type_name(node.type)}>({expression(args[0])})"
    if kind == "conditional":
        return (
            f"({expression(args[0])} ? {expression(args[1])} : {expression(args[2])})"
        )
    if kind == "new":
        suffix = "[" + expression(node.extent) + "]" if node.extent is not None else ""
        init = ""
        if node.qualifier != "default":
            opening, closing = ("{", "}") if node.qualifier == "brace" else ("(", ")")
            init = opening + ", ".join(_argument(a) for a in args) + closing
        return "new " + type_name(node.type) + suffix + init
    if kind == "delete":
        return (
            "delete" + ("[] " if node.array else " ") + "(" + expression(args[0]) + ")"
        )
    if kind == "lambda":
        captures = []
        for capture in node.captures:
            if isinstance(capture, tuple):
                captures.append(_name(capture[0]) + " = " + expression(capture[1]))
            elif capture in {"&", "=", "this", "*this"}:
                captures.append(capture)
            else:
                captures.append(
                    ("&" if capture.startswith("&") else "")
                    + _name(capture.removeprefix("&"))
                )
        params = ", ".join(declaration(p, parameter=True) for p in node.params)
        qualifier = " mutable" if node.qualifier == "mutable" else ""
        result = " -> " + type_name(node.type) if node.type is not None else ""
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
            + type_name(node.type)
            + ")"
        )
    if kind == "sizeof_expr":
        return "sizeof(" + expression(args[0]) + ")"
    if kind == "offsetof":
        return "offsetof(" + type_name(node.type) + ", " + _name(node.name) + ")"
    if kind == "offset_check":
        return (
            "SCAR_OFFSET_MATCH("
            + type_name(node.type)
            + ", "
            + _name(node.name)
            + ", "
            + str(int(node.value))
            + ")"
        )
    raise ValueError(f"unsupported C++ target expression {kind!r}")


def _block(body, depth):
    lines = ["{"]
    lines.extend(statement(s, depth + 1) for s in body)
    lines.append("  " * depth + "}")
    return "\n".join(lines)


def statement(node, depth=0):
    if not isinstance(node, Stmt):
        raise TypeError("C++ statement position requires a Stmt node")
    indent, kind = "  " * depth, node.kind
    if kind in {"expression", "expr"}:
        text = expression(node.value) + ";"
    elif kind == "return":
        text = (
            "return"
            + (" " + expression(node.value) if node.value is not None else "")
            + ";"
        )
    elif kind == "delete":
        text = (
            expression(node.value)
            if node.value.kind == "delete"
            else "delete " + expression(node.value)
        ) + ";"
    elif kind == "declaration":
        text = declaration(node.decl)
    elif kind == "block":
        text = _block(node.body, depth)
    elif kind == "if":
        text = "if (" + expression(node.condition) + ") " + _block(node.body, depth)
        if node.else_body:
            text += " else " + _block(node.else_body, depth)
    elif kind == "for":
        initial = (
            statement(node.init).removesuffix(";")
            if isinstance(node.init, Stmt)
            else expression(node.init) if node.init is not None else ""
        )
        condition = expression(node.condition) if node.condition is not None else ""
        step = expression(node.step) if node.step is not None else ""
        text = f"for ({initial}; {condition}; {step}) " + _block(node.body, depth)
    elif kind == "each":
        text = (
            "for ("
            + declaration(node.decl, parameter=True)
            + " : "
            + expression(node.value)
            + ") "
            + _block(node.body, depth)
        )
    elif kind in {"while", "switch"}:
        text = (
            kind + " (" + expression(node.condition) + ") " + _block(node.body, depth)
        )
    elif kind == "do_while":
        text = (
            "do "
            + _block(node.body, depth)
            + " while ("
            + expression(node.condition)
            + ");"
        )
    elif kind in {"break", "continue"}:
        text = kind + ";"
    elif kind == "goto":
        text = "goto " + _name(node.name) + ";"
    elif kind == "label":
        text = _name(node.name) + ":;"
    elif kind == "case":
        text = "case " + expression(node.value) + ":"
    elif kind == "default":
        text = "default:"
    elif kind == "pragma":
        if node.directive not in {
            "omp parallel for",
            "omp parallel",
            "omp for",
            "omp simd",
            "omp critical",
            "omp single",
            "omp barrier",
        }:
            raise ValueError("unsupported target pragma")
        clauses = []
        for name, args in node.clauses:
            clauses.append(
                _name(name) + "(" + ", ".join(_argument(a) for a in args) + ")"
            )
        text = (
            "#pragma " + node.directive + (" " + " ".join(clauses) if clauses else "")
        )
    else:
        raise ValueError(f"unsupported C++ target statement {kind!r}")
    return indent + text


def declaration(node, parameter=False):
    if not isinstance(node, Decl):
        raise TypeError("C++ declaration position requires a Decl node")
    kind = node.kind
    if kind == "param" and node.definition == "variadic":
        return "..."
    attributes = ""
    if node.attributes:
        names = []
        for attribute in node.attributes:
            if not isinstance(attribute, Attribute):
                raise TypeError("C++ declaration attributes require Attribute nodes")
            name = _name(attribute.name)
            if attribute.args:
                name += "(" + ", ".join(_argument(a) for a in attribute.args) + ")"
            names.append(name)
        attributes = "[[" + ", ".join(names) + "]] "
    if any(s not in _STORAGE for s in node.storage):
        raise ValueError("invalid target storage specifier")
    storage = (" ".join(node.storage) + " ") if node.storage else ""
    alignment = ""
    if node.alignment is not None:
        if node.alignment <= 0 or node.alignment & (node.alignment - 1):
            raise ValueError("invalid target alignment")
        alignment = f"alignas({node.alignment}) "
    if kind in {"variable", "field", "param"}:
        if node.linkage not in {None, "C", "C++"} or (
            kind != "variable" and node.linkage is not None
        ):
            raise ValueError("invalid target variable linkage")
        text = attributes + alignment + storage + type_name(node.type)
        if node.name:
            text += " " + _name(node.name)
        if node.initializer is not None:
            value = expression(node.initializer)
            text += (
                "(" + value + ")" if "direct_init" in node.qualifiers else " = " + value
            )
        if parameter or kind == "param":
            if node.linkage == "C":
                raise ValueError("a parameter cannot declare C linkage")
            return text
        text += ";"
        return 'extern "C" { ' + text + " }" if node.linkage == "C" else text
    if kind == "alias":
        return "using " + _name(node.name) + " = " + type_name(node.type) + ";"
    if kind == "static_assert":
        return "static_assert(" + expression(node.initializer) + ");"
    if kind in {"forward", "record"}:
        keyword = node.definition or "struct"
        if keyword not in {"struct", "union", "class"}:
            raise ValueError("invalid record keyword")
        text = keyword + " " + alignment + _name(node.name)
        if kind == "forward":
            return text + ";"
        if node.bases:
            text += " : " + ", ".join(
                access + " " + ("virtual " if virtual else "") + type_name(type_)
                for access, virtual, type_ in node.bases
            )
        return (
            text
            + " {\n"
            + "\n".join(
                "  " + declaration(d).replace("\n", "\n  ") for d in node.members
            )
            + "\n};"
        )
    if kind == "function":
        if any(q not in _QUALIFIERS for q in node.qualifiers):
            raise ValueError("invalid target function qualifier")
        prefix = 'extern "C" ' if node.linkage == "C" else ""
        if node.linkage not in {None, "C", "C++"}:
            raise ValueError("invalid target linkage")
        result = type_name(node.type) + " " if node.type is not None else ""
        text = (
            prefix
            + attributes
            + storage
            + result
            + _name(node.name)
            + "("
            + ", ".join(declaration(p, parameter=True) for p in node.params)
            + ")"
        )
        if node.qualifiers:
            text += " " + " ".join(node.qualifiers)
        if node.definition is not None:
            suffix = {"default": "default", "delete": "delete", "pure": "0"}.get(
                node.definition
            )
            if suffix is None:
                raise ValueError("invalid target function definition")
            return text + " = " + suffix + ";"
        if node.initializers:
            text += " : " + ", ".join(
                _name(name) + "(" + ", ".join(_argument(a) for a in args) + ")"
                for name, args in node.initializers
            )
        return text + (" " + _block(node.body, 0) if node.body is not None else ";")
    if kind == "namespace":
        return (
            "namespace "
            + _name(node.name)
            + " {\n"
            + "\n".join(declaration(d) for d in node.members)
            + "\n}"
        )
    raise ValueError(f"unsupported C++ target declaration {kind!r}")


def print_unit(unit):
    if not isinstance(unit, Unit):
        raise TypeError("expected a typed C++ translation unit")
    lines = ["#pragma once"] if unit.once else []
    for kind, path in unit.includes:
        if not isinstance(path, str) or any(c in path for c in '\n\r"<>'):
            raise ValueError("invalid target include path")
        if kind not in {"system", "local"}:
            raise ValueError("invalid target include kind")
        lines.append(
            "#include " + ("<" + path + ">" if kind == "system" else '"' + path + '"')
        )
    lines.extend(["", *[declaration(d) for d in unit.declarations]])
    return "\n\n".join(lines) + "\n"
