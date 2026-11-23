"""Typed C++ target trees exchanged by lowering and source printing."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Type:
    kind: str
    name: str | None = None
    args: tuple = ()
    element: "Type | None" = None
    extent: "Expr | None" = None
    params: tuple = ()
    result: "Type | None" = None
    parent: "Type | None" = None


@dataclass(frozen=True)
class Expr:
    kind: str
    args: tuple = ()
    name: str | None = None
    value: object = None
    type: Type | None = None
    operator: str | None = None
    body: tuple = ()
    params: tuple = ()
    captures: tuple = ()
    qualifier: str | None = None
    postfix: bool = False
    arrow: bool = False
    array: bool = False
    extent: "Expr | None" = None


@dataclass(frozen=True)
class Stmt:
    kind: str
    value: Expr | None = None
    body: tuple = ()
    else_body: tuple = ()
    init: "Stmt | Expr | None" = None
    condition: Expr | None = None
    step: Expr | None = None
    decl: "Decl | None" = None
    name: str | None = None
    cases: tuple = ()
    directive: str | None = None
    clauses: tuple = ()


@dataclass(frozen=True)
class Attribute:
    name: str
    args: tuple = ()


@dataclass(frozen=True)
class Decl:
    kind: str
    name: str | None = None
    type: Type | None = None
    params: tuple = ()
    body: tuple | None = None
    members: tuple = ()
    bases: tuple = ()
    storage: tuple = ()
    initializer: Expr | None = None
    initializers: tuple = ()
    linkage: str | None = None
    qualifiers: tuple = ()
    definition: str | None = None
    alignment: int | None = None
    attributes: tuple = ()


@dataclass(frozen=True)
class Unit:
    includes: tuple = ()
    declarations: tuple = ()
    once: bool = False


_NODE_TYPES = {t.__name__: t for t in (Type, Expr, Stmt, Attribute, Decl, Unit)}


def to_data(value):
    """Serialize every target node as structure, preserving empty definitions."""
    if type(value).__name__ in _NODE_TYPES and type(value) in _NODE_TYPES.values():
        result = {"node": type(value).__name__}
        for field in fields(value):
            member = getattr(value, field.name)
            if member != field.default:
                result[field.name] = to_data(member)
        return result
    if isinstance(value, (tuple, list)):
        return [to_data(v) for v in value]
    if isinstance(value, dict):
        return {k: to_data(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported target tree value {type(value).__name__}")


def from_data(value):
    if isinstance(value, list):
        return tuple(from_data(v) for v in value)
    if not isinstance(value, dict):
        return value
    if "node" not in value:
        return {k: from_data(v) for k, v in value.items()}
    cls = _NODE_TYPES.get(value["node"])
    if cls is None:
        raise ValueError(f"unknown C++ target node {value['node']!r}")
    allowed = {f.name for f in fields(cls)}
    if set(value) - {"node"} - allowed:
        raise ValueError(f"unknown fields in {cls.__name__}")
    return cls(**{k: from_data(v) for k, v in value.items() if k != "node"})


def encode_unit(unit):
    if not isinstance(unit, Unit):
        raise TypeError("expected a C++ translation unit")
    return to_data(unit)


def unit_from_data(data):
    unit = from_data(data)
    if not isinstance(unit, Unit):
        raise TypeError("target product must contain a C++ translation unit")
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
