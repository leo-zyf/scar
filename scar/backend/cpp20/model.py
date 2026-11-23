"""Typed C++ target trees exchanged by lowering and source printing."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from hashlib import sha256
import math
import re


@dataclass(frozen=True)
class Type:
    kind: str
    name: str | None = None
    args: tuple[Type | Expr, ...] = ()
    element: Type | None = None
    extent: Expr | None = None
    params: tuple[Type, ...] = ()
    result: Type | None = None
    parent: Type | None = None


@dataclass(frozen=True)
class Expr:
    kind: str
    args: tuple[Type | Expr, ...] = ()
    name: str | None = None
    value: str | int | float | bool | None = None
    type: Type | None = None
    operator: str | None = None
    body: tuple[Stmt, ...] = ()
    params: tuple[Decl, ...] = ()
    captures: tuple[str | tuple[str, Expr], ...] = ()
    qualifier: str | None = None
    postfix: bool = False
    arrow: bool = False
    array: bool = False
    extent: Expr | None = None


@dataclass(frozen=True)
class Stmt:
    kind: str
    value: Expr | None = None
    body: tuple[Stmt, ...] = ()
    else_body: tuple[Stmt, ...] = ()
    init: Stmt | Expr | None = None
    condition: Expr | None = None
    step: Expr | None = None
    decl: Decl | None = None
    name: str | None = None
    cases: tuple[()] = ()
    directive: str | None = None
    clauses: tuple[tuple[str, tuple[Expr, ...]], ...] = ()
    handlers: tuple[tuple[Decl | None, tuple[Stmt, ...]], ...] = ()


@dataclass(frozen=True)
class Attribute:
    name: str
    args: tuple[Type | Expr, ...] = ()


@dataclass(frozen=True)
class Decl:
    kind: str
    name: str | None = None
    type: Type | None = None
    params: tuple[Decl, ...] = ()
    body: tuple[Stmt, ...] | None = None
    members: tuple[Decl, ...] = ()
    bases: tuple[tuple[str, bool, Type], ...] = ()
    storage: tuple[str, ...] = ()
    initializer: Expr | None = None
    initializers: tuple[tuple[str, tuple[Expr, ...]], ...] = ()
    linkage: str | None = None
    qualifiers: tuple[str, ...] = ()
    definition: str | None = None
    alignment: int | None = None
    attributes: tuple[Attribute, ...] = ()


@dataclass(frozen=True)
class Unit:
    includes: tuple[tuple[str, str], ...] = ()
    declarations: tuple[Decl, ...] = ()
    once: bool = False


_NODE_TYPES = {t.__name__: t for t in (Type, Expr, Stmt, Attribute, Decl, Unit)}
_FIELDS = {cls: {f.name: f for f in fields(cls)} for cls in _NODE_TYPES.values()}
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
_NUMBER = re.compile(
    r"[+-]?(?:(?:0[xX][0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?[pP][+-]?[0-9]+"
    r"|(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+)[fFlL]?"
    r"|(?:0[xX][0-9a-fA-F]+|0[bB][01]+|0[0-7]*|[1-9][0-9]*)"
    r"(?:[uU](?:ll|LL|l|L)?|(?:ll|LL|l|L)[uU]?)?)\Z"
)
_BINARY = frozenset(
    "+ - * / % & | ^ && || == != < > <= >= << >> = += -= *= /= %= &= |= ^= <<= >>= ,".split()
)
_UNARY = frozenset("+ - * & ! ~ ++ --".split())
_OVERLOADS = _BINARY | _UNARY | {"()", "[]", "->", " bool"}
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
TYPE_WRAPPERS = {
    "ptr": "add_pointer_t",
    "ref": "add_lvalue_reference_t",
    "rref": "add_rvalue_reference_t",
    "const": "add_const_t",
    "volatile": "add_volatile_t",
}
CASTS = {
    "static": "static_cast",
    "qualification": "const_cast",
    "const": "const_cast",
    "reinterpret": "reinterpret_cast",
    "dynamic": "dynamic_cast",
}
FUNCTION_DEFINITIONS = {"default": "default", "delete": "delete", "pure": "0"}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def identifier(value):
    _require(isinstance(value, str) and bool(value), "expected a qualified identifier")
    for part in value.removeprefix("::").split("::"):
        if _IDENTIFIER.fullmatch(part.removeprefix("~")):
            continue
        if part.startswith("operator") and part[8:] in _OVERLOADS:
            continue
        raise ValueError(f"invalid C++ identifier {value!r}")
    return value


def _type_identifier(value):
    if not isinstance(value, str) or value not in _BUILTINS:
        identifier(value)


def _literal(value):
    if value is None or type(value) in (int, bool):
        return
    if type(value) is float and math.isfinite(value):
        return
    _require(
        isinstance(value, str) and _NUMBER.fullmatch(value), "invalid numeric literal"
    )


def _choices(*values):
    options = frozenset(values)

    def check(value):
        _require(
            isinstance(value, str) and value in options, f"unexpected token {value!r}"
        )

    return check


def _optional(check):
    def present(value):
        if value is not None:
            check(value)

    return present


def _sequence(check, minimum=0, maximum=None):
    def items(values):
        _require(isinstance(values, tuple), "expected a sequence")
        if len(values) < minimum or maximum is not None and len(values) > maximum:
            raise ValueError(
                f"invalid arity {len(values)} (minimum {minimum}, maximum {maximum})"
            )
        for index, value in enumerate(values):
            try:
                check(value)
            except ValueError as exc:
                raise ValueError(f"[{index}]: {exc}") from None

    return items


def _tuple(*checks):
    def items(value):
        _require(
            isinstance(value, tuple) and len(value) == len(checks),
            "invalid tuple arity",
        )
        for check, member in zip(checks, value):
            check(member)

    return items


def _child(cls, *kinds):
    def check(value):
        _require(type(value) is cls, f"expected {cls.__name__} node")
        if kinds and value.kind not in kinds:
            raise ValueError(f"{cls.__name__}.{value.kind} is not valid here")
        _validate(value)

    return check


_type = _child(Type)
_expr = _child(Expr)
_stmt = _child(Stmt)
_attribute = _child(Attribute)
_global = _child(
    Decl,
    "variable",
    "function",
    "record",
    "forward",
    "alias",
    "static_assert",
    "namespace",
)
_local = _child(
    Decl, "variable", "function", "record", "forward", "alias", "static_assert"
)
_member = _child(
    Decl, "field", "variable", "function", "record", "forward", "alias", "static_assert"
)
_param = _child(Decl, "param")


def _argument(value):
    (_type if isinstance(value, Type) else _expr)(value)


def _call(args):
    _require(isinstance(args, tuple) and bool(args), "call requires a callee")
    _expr(args[0])
    # A function-like macro may take a type token, e.g. va_arg(ap, T).
    _sequence(_argument)(args[1:])


def _parameters(values):
    _sequence(_param)(values)
    _require(
        all(p.definition != "variadic" for p in values[:-1]),
        "variadic parameter must be last",
    )


def _capture(value):
    if isinstance(value, tuple):
        _tuple(identifier, _expr)(value)
    elif isinstance(value, str) and value in {"&", "=", "this", "*this"}:
        return
    elif isinstance(value, str):
        identifier(value.removeprefix("&"))
    else:
        raise ValueError("invalid lambda capture")


def _loop_init(value):
    if isinstance(value, Stmt):
        _child(Stmt, "declaration", "expression", "expr")(value)
    elif value is not None:
        _expr(value)


def _range_var(value):
    _child(Decl, "variable", "param")(value)
    _require(
        value.name
        and value.type
        and value.initializer is None
        and value.linkage is None,
        "range loop requires a named declaration without an initializer",
    )


def _boolean(value):
    _require(type(value) is bool, "expected a boolean")


def _alignment(value):
    _require(
        type(value) is int and value > 0 and value & (value - 1) == 0,
        "invalid alignment",
    )


def _offset(value):
    _require(type(value) is int and value >= 0, "expected a nonnegative integer offset")


def _include_path(value):
    _require(
        isinstance(value, str)
        and bool(value)
        and not any(ord(ch) < 32 or ch in '"<>' for ch in value),
        "invalid include path",
    )


_statements = _sequence(_stmt)
_expressions = _sequence(_expr)
_arguments = _sequence(_argument)
_attrs = _sequence(_attribute)
_align = _optional(_alignment)
_linkage = _optional(_choices("C", "C++"))
_variable_storage = _sequence(
    _choices("extern", "static", "inline", "constexpr", "constinit", "thread_local")
)
_field_storage = _sequence(
    _choices("static", "inline", "constexpr", "constinit", "thread_local", "mutable")
)
_function_storage = _sequence(
    _choices(
        "extern", "static", "inline", "constexpr", "consteval", "virtual", "explicit"
    )
)
_qualifiers = _sequence(
    _choices(
        "const",
        "volatile",
        "noexcept",
        "noexcept(true)",
        "noexcept(false)",
        "override",
        "final",
        "&",
        "&&",
    )
)
_aggregate = _optional(_choices("struct", "union", "class"))
_one = _sequence(_expr, 1, 1)
_two = _sequence(_expr, 2, 2)
_cast_mode = _optional(_choices(*CASTS, "explicit_pointer"))


def _grammar(cls, forms):
    """Compile field productions once, including fields that must retain their default."""
    rules = {}
    for kinds, slots in forms.items():
        unused = tuple(
            (name, field.default)
            for name, field in _FIELDS[cls].items()
            if name != "kind" and name not in slots
        )
        for kind in kinds.split():
            rules[kind] = (tuple(slots.items()), unused)
    return rules


_GRAMMAR = {
    Unit: _grammar(
        Unit,
        {
            "unit": dict(
                includes=_sequence(_tuple(_choices("system", "local"), _include_path)),
                declarations=_sequence(_global),
                once=_boolean,
            )
        },
    ),
    Attribute: _grammar(
        Attribute, {"attribute": dict(name=identifier, args=_arguments)}
    ),
    Type: _grammar(
        Type,
        {
            "name": dict(name=_type_identifier),
            "auto": {},
            "tagged": dict(name=identifier),
            "template": dict(name=identifier, args=_arguments),
            "member": dict(name=identifier, parent=_type),
            " ".join(TYPE_WRAPPERS): dict(element=_type),
            "array": dict(element=_type, extent=_optional(_expr)),
            "function": dict(result=_type, params=_sequence(_type)),
            "decltype": dict(args=_one),
        },
    ),
    Expr: _grammar(
        Expr,
        {
            "name": dict(name=identifier),
            "literal": dict(value=_literal),
            "string": dict(
                value=lambda v: _require(isinstance(v, str), "expected decoded string")
            ),
            "nullptr this": {},
            "template_name": dict(name=identifier, args=_arguments),
            "type_member offsetof": dict(type=_type, name=identifier),
            "braced init_list": dict(args=_expressions),
            "construct": dict(
                type=_type,
                args=_expressions,
                qualifier=_optional(_choices("paren", "brace", "list")),
            ),
            "call": dict(args=_call),
            "member": dict(args=_one, name=identifier, arrow=_boolean),
            "subscript": dict(args=_sequence(_expr, 2)),
            "unary": dict(args=_one, operator=_choices(*_UNARY), postfix=_boolean),
            "binary": dict(args=_two, operator=_choices(*_BINARY)),
            "cast": dict(
                args=_one, type=_type, qualifier=_cast_mode, operator=_cast_mode
            ),
            "conditional": dict(args=_sequence(_expr, 3, 3)),
            "new": dict(
                type=_type,
                extent=_optional(_expr),
                args=_expressions,
                qualifier=_optional(_choices("default", "paren", "brace")),
            ),
            "delete": dict(args=_one, array=_boolean),
            "lambda": dict(
                captures=_sequence(_capture),
                params=_parameters,
                type=_optional(_type),
                qualifier=_optional(_choices("mutable")),
                body=_statements,
            ),
            "sizeof_type alignof": dict(type=_type),
            "sizeof_expr": dict(args=_one),
            "offset_check": dict(type=_type, name=identifier, value=_offset),
        },
    ),
    Stmt: _grammar(
        Stmt,
        {
            "expression expr case delete": dict(value=_expr),
            "return": dict(value=_optional(_expr)),
            "declaration": dict(decl=_local),
            "block": dict(body=_statements),
            "if": dict(condition=_expr, body=_statements, else_body=_statements),
            "for": dict(
                init=_loop_init,
                condition=_optional(_expr),
                step=_optional(_expr),
                body=_statements,
            ),
            "each": dict(decl=_range_var, value=_expr, body=_statements),
            "while do_while switch": dict(condition=_expr, body=_statements),
            "break continue default": {},
            "throw": dict(value=_optional(_expr)),
            "try": dict(
                body=_statements,
                handlers=_sequence(
                    _tuple(_optional(_child(Decl, "param")), _statements), minimum=1
                ),
            ),
            "goto label": dict(name=identifier),
            "pragma": dict(
                directive=_choices(
                    "omp parallel for",
                    "omp parallel",
                    "omp for",
                    "omp simd",
                    "omp critical",
                    "omp barrier",
                    "omp single",
                ),
                clauses=_sequence(_tuple(identifier, _expressions)),
            ),
        },
    ),
    Decl: _grammar(
        Decl,
        {
            "variable": dict(
                name=identifier,
                type=_type,
                initializer=_optional(_expr),
                linkage=_linkage,
                storage=_variable_storage,
                alignment=_align,
                attributes=_attrs,
                qualifiers=_sequence(_choices("direct_init")),
            ),
            "field": dict(
                name=identifier,
                type=_type,
                initializer=_optional(_expr),
                storage=_field_storage,
                alignment=_align,
                attributes=_attrs,
            ),
            "param": dict(
                name=_optional(identifier),
                type=_optional(_type),
                initializer=_optional(_expr),
                attributes=_attrs,
                definition=_optional(_choices("variadic")),
            ),
            "alias": dict(name=identifier, type=_type),
            "static_assert": dict(initializer=_expr),
            "forward": dict(name=identifier, definition=_aggregate, alignment=_align),
            "record": dict(
                name=identifier,
                members=_sequence(_member),
                definition=_aggregate,
                alignment=_align,
                bases=_sequence(
                    _tuple(_choices("public", "private", "protected"), _boolean, _type)
                ),
            ),
            "function": dict(
                name=identifier,
                type=_optional(_type),
                params=_parameters,
                body=_optional(_statements),
                linkage=_linkage,
                storage=_function_storage,
                attributes=_attrs,
                qualifiers=_qualifiers,
                definition=_optional(_choices(*FUNCTION_DEFINITIONS)),
                initializers=_sequence(_tuple(identifier, _expressions)),
            ),
            "namespace": dict(name=identifier, members=_sequence(_global)),
        },
    ),
}


def _check_declaration_scope(declarations, owner=None):
    for decl in declarations:
        if owner is not None:
            _require(decl.linkage is None, "language linkage requires namespace scope")
            if decl.kind == "variable":
                _require(
                    "static" in decl.storage,
                    "instance member requires a field declaration",
                )
        elif "virtual" in decl.storage or "explicit" in decl.storage:
            raise ValueError("virtual and explicit apply to member declarations")
        if decl.kind != "function" or decl.type is not None:
            continue
        parts = decl.name.removeprefix("::").split("::")
        record = parts[-2] if len(parts) > 1 else owner
        spelling = parts[-1]
        special = record and (
            spelling == record
            or spelling == "~" + record
            or spelling.startswith("operator ")
        )
        _require(
            special and decl.linkage != "C",
            f"function {decl.name!r} requires a result type",
        )
        if decl.initializers:
            _require(spelling == record, "only a constructor takes member initializers")


def _validate(node):
    cls = type(node)
    kind = "unit" if cls is Unit else "attribute" if cls is Attribute else node.kind
    form = _GRAMMAR[cls].get(kind) if isinstance(kind, str) else None
    if form is None:
        raise ValueError(f"unknown {cls.__name__} kind {kind!r}")
    slots, unused = form
    for field, default in unused:
        value = getattr(node, field)
        if value != default or type(value) is not type(default):
            raise ValueError(f"{cls.__name__}.{kind} does not accept {field}")
    for field, check in slots:
        try:
            check(getattr(node, field))
        except ValueError as exc:
            raise ValueError(f"{cls.__name__}.{kind}.{field}: {exc}") from None
    if cls is Expr:
        if kind == "unary" and node.postfix and node.operator not in ("++", "--"):
            raise ValueError("only ++ and -- can be postfix operators")
        if kind == "cast" and node.operator and node.qualifier:
            raise ValueError("cast has two conversion modes")
        if kind == "new" and node.qualifier == "default" and node.args:
            raise ValueError("default initialization has no explicit arguments")
    elif cls is Decl:
        if kind == "param":
            if node.definition == "variadic":
                _require(
                    not any((node.type, node.name, node.initializer, node.attributes)),
                    "variadic marker cannot carry a parameter declaration",
                )
            else:
                _require(node.type is not None, "parameter requires a type")
        if kind == "function":
            if node.definition:
                _require(
                    node.body is None and not node.initializers,
                    "special function definition cannot carry a body",
                )
            if node.initializers:
                _require(
                    node.body is not None and node.type is None,
                    "member initializers require a constructor definition",
                )
        if kind in {"record", "namespace"}:
            _check_declaration_scope(
                node.members, node.name if kind == "record" else None
            )
    elif cls is Unit:
        _check_declaration_scope(node.declarations)
    elif cls is Stmt and kind == "try":
        for index, (parameter, _) in enumerate(node.handlers):
            if parameter is None:
                _require(
                    index == len(node.handlers) - 1, "a catch-all handler must be last"
                )
            else:
                _require(
                    parameter.type is not None
                    and not parameter.definition
                    and parameter.initializer is None,
                    "invalid catch parameter",
                )
    elif cls is Stmt and kind == "declaration":
        decl = node.decl
        _require(
            decl.linkage is None
            and not (
                decl.kind == "function" and (decl.body is not None or decl.type is None)
            ),
            "invalid block-scope declaration",
        )


def validate_node(node, expected):
    """Validate one standalone target syntax tree."""
    _child(expected)(node)
    return node


def validate_unit(unit):
    """Check structural C++ syntax before publishing or consuming a translation unit."""
    return validate_node(unit, Unit)


def to_data(value):
    """Serialize every target node as structure, preserving empty definitions."""
    if type(value) in _FIELDS:
        result = {"node": type(value).__name__}
        for field in _FIELDS[type(value)].values():
            member = getattr(value, field.name)
            if member != field.default:
                result[field.name] = to_data(member)
        return result
    if isinstance(value, (tuple, list)):
        return [to_data(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported target tree value {type(value).__name__}")


def from_data(value):
    if isinstance(value, list):
        return tuple(from_data(v) for v in value)
    if not isinstance(value, dict):
        return value
    tag = value.get("node")
    cls = _NODE_TYPES.get(tag) if isinstance(tag, str) else None
    if cls is None:
        raise ValueError(f"unknown C++ target node {tag!r}")
    allowed = _FIELDS[cls]
    if set(value) - {"node"} - allowed.keys():
        raise ValueError(f"unknown fields in {cls.__name__}")
    if any(f.default is MISSING and f.name not in value for f in allowed.values()):
        raise ValueError(f"missing required field in {cls.__name__}")
    return cls(**{k: from_data(v) for k, v in value.items() if k != "node"})


def encode_unit(unit):
    return to_data(validate_unit(unit))


def unit_from_data(data):
    """Decode wire records; the consuming print entry validates the complete tree."""
    unit = from_data(data)
    _require(type(unit) is Unit, "target product must contain a translation unit")
    return unit


def name(value):
    return Expr("name", name=value)


def named(value):
    return Type("name", name=value)


def integer(value):
    return Expr("literal", value=value)


def call(callee, *args):
    return Expr(
        "call", args=((name(callee) if isinstance(callee, str) else callee), *args)
    )


def binary(operator, left, right):
    return Expr("binary", args=(left, right), operator=operator)


def member(base, field, arrow=False):
    return Expr("member", args=(base,), name=field, arrow=arrow)


def construct(type_, *args, form="paren"):
    return Expr("construct", type=type_, args=args, qualifier=form)


def template(value, *args):
    return Expr("template_name", name=value, args=args)


def symbol_name(name, category):
    if not isinstance(name, str) or not name:
        raise ValueError(f"empty {category} identifier")
    if _IDENTIFIER.fullmatch(name) and not name.startswith("_") and "__" not in name:
        return f"{category}_{name}"
    label = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[:48] or "symbol"
    key = sha256(name.encode()).hexdigest()[:20]
    return f"{category}x_{label}_{key}"


def unary(operator, value, postfix=False):
    return Expr("unary", operator=operator, args=(value,), postfix=postfix)


def utf8_literal(value, target):
    if not isinstance(value, str):
        raise ValueError("text literal requires a UTF-8 string")
    return Expr(
        "construct",
        type=named(target),
        args=(Expr("string", value=value), integer(len(value.encode("utf-8")))),
        qualifier="paren",
    )


def variable(symbol, ty, init=None, **attributes):
    return Decl("variable", name=symbol, type=ty, initializer=init, **attributes)


def declaration(decl):
    return Stmt("declaration", decl=decl)


def expression_stmt(value):
    return Stmt("expression", value=value)


def cast(ty, operand, operator="static"):
    return Expr("cast", operator=operator, type=ty, args=(operand,))
