"""Normalize compiler-resolved object construction, lifetime and dispatch."""

from hashlib import sha256
import json

from .facts import cursor_kind as _kind, type_kind, is_expression


class BindingError(ValueError):
    def __init__(self, cursor, detail):
        super().__init__(detail)
        self.cursor = cursor


def _record_type(ty, cursor, unit):
    spec = unit.type_info(ty, cursor)
    while isinstance(spec, dict) and spec.get("kind") in {"const", "volatile"}:
        spec = spec["of"]
    return spec if isinstance(spec, dict) and spec.get("kind") == "record" else None


def method(cursor, unit):
    kind = _kind(cursor)
    if kind not in {"CONSTRUCTOR", "DESTRUCTOR", "CXX_METHOD"}:
        return None
    special = {"CONSTRUCTOR": "constructor", "DESTRUCTOR": "destructor"}
    if kind == "CXX_METHOD" and (
        cursor.is_static_method() or not cursor.is_virtual_method()
    ):
        return None
    params = [unit.param_info(p) for p in cursor.get_arguments()]
    const = kind == "CXX_METHOD" and cursor.is_const_method()
    qualifier = cursor.type.get_ref_qualifier().name
    ref = {"LVALUE": "&", "RVALUE": "&&"}.get(qualifier)
    slot_params = []
    for param in params:
        spec = param["type"]
        while isinstance(spec, dict) and spec.get("kind") in {"const", "volatile"}:
            spec = spec["of"]
        slot_params.append(spec)
    slot_key = [cursor.spelling, slot_params, const, ref]
    result = dict(
        kind=special.get(kind, "virtual_method"),
        owner=unit.record_name(cursor.semantic_parent),
        symbol=unit.symbol(cursor),
        params=params,
        const=const,
        order=cursor.canonical.location.offset,
        virtual=kind != "CONSTRUCTOR" and cursor.is_virtual_method(),
        pure=kind != "CONSTRUCTOR" and cursor.is_pure_virtual_method(),
        defaulted=cursor.is_default_method(),
        deleted=cursor.is_deleted_method(),
        slot=sha256(json.dumps(slot_key, sort_keys=True).encode()).hexdigest()[:24],
        result=(
            "void" if kind in special else unit.type_info(cursor.result_type, cursor)
        ),
    )
    owner_location = cursor.semantic_parent.location
    if (
        kind in special
        and result["defaulted"]
        and cursor.location.file
        and owner_location.file
        and cursor.location.file.name == owner_location.file.name
        and cursor.location.offset == owner_location.offset
    ):
        result["implicit"] = True
    if ref:
        result["ref_qualifier"] = ref
    exception = cursor.exception_specification_kind.name
    tokens = [t.spelling for t in unit.facts.tokens(cursor)]
    # An omitted destructor exception specification is inferred from subobjects
    # by the target language. The compiler may resolve it later in another TU.
    if kind == "DESTRUCTOR" and "noexcept" not in tokens and "throw" not in tokens:
        return result
    if exception == "BASIC_NOEXCEPT":
        result["noexcept"] = True
    elif exception == "COMPUTED_NOEXCEPT":
        if "noexcept" in tokens:
            tail = tokens[tokens.index("noexcept") + 1 :]
            if tail[:3] == ["(", "true", ")"]:
                result["noexcept"] = True
            elif tail[:3] == ["(", "false", ")"]:
                result["noexcept"] = False
            else:
                raise BindingError(
                    cursor, "conditional exception contract needs a resolved constant"
                )
    return result


def record(cursor, unit):
    bases, methods = [], []
    for child in cursor.get_children():
        if _kind(child) == "CXX_BASE_SPECIFIER":
            ty = _record_type(child.type, child, unit)
            if ty is None:
                raise BindingError(child, "base class has no concrete object type")
            virtual = any(t.spelling == "virtual" for t in unit.facts.tokens(child))
            bases.append(dict(name=ty["name"], virtual=virtual))
        else:
            info = method(child, unit)
            if info:
                methods.append(info)
    return dict(bases=bases, methods=methods)


def _require(cursor, reader):
    symbol = reader.unit.declare_call(cursor)
    reader.unit.require(cursor)
    reader.calls.add(symbol)
    return symbol


def _arguments(cursor, reader):
    return reader.arguments(cursor, cursor.referenced)


def _parameters(declaration, reader):
    return [reader.unit.param_info(p)["type"] for p in declaration.get_arguments()]


def invoke(cursor, reader):
    kind = _kind(cursor)
    if kind != "CALL_EXPR" or not cursor.referenced:
        return None
    declaration = cursor.referenced
    declared = _kind(declaration)
    if declared == "CONSTRUCTOR":
        ty = _record_type(declaration.semantic_parent.type, cursor, reader.unit)
        if ty is None:
            return None
        symbol = _require(declaration, reader)
        return dict(
            op="object_construct",
            type=ty,
            args=_arguments(cursor, reader),
            parameter_types=_parameters(declaration, reader),
            symbol=symbol,
        )
    info = method(declaration, reader.unit)
    if not info:
        return None
    children = list(cursor.get_children())
    member = next((c for c in children if _kind(c) == "MEMBER_REF_EXPR"), None)
    if member is None:
        raise BindingError(cursor, "object call requires a resolved receiver")
    operands = [c for c in member.get_children() if is_expression(c)]
    if (
        not operands
        and reader.owner
        and _kind(reader.owner) in {"CXX_METHOD", "CONSTRUCTOR", "DESTRUCTOR"}
    ):
        receiver = dict(op="var", name="this")
    elif len(operands) != 1:
        raise BindingError(cursor, "object call receiver is ambiguous")
    else:
        receiver = reader.expression(operands[0])
        if type_kind(operands[0].type.get_canonical()) != "POINTER":
            receiver = dict(op="unary", operator="addr", value=receiver)
    symbol = _require(declaration, reader)
    if info["kind"] == "destructor":
        return dict(
            op="object_destroy",
            value=receiver,
            owner=info["owner"],
            explicit=True,
            destructor=symbol,
        )
    tokens = [t.spelling for t in reader.unit.facts.tokens(member)]
    return dict(
        op="object_dispatch",
        receiver=receiver,
        args=_arguments(cursor, reader),
        parameter_types=[p["type"] for p in info["params"]],
        owner=info["owner"],
        slot=info["slot"],
        symbol=symbol,
        qualified="::" in tokens,
    )


def allocation(cursor, reader):
    ty = cursor.type.get_pointee().get_canonical()
    spec = _record_type(ty, cursor, reader.unit)
    if spec is None:
        return None
    tokens = [t.spelling for t in reader.unit.facts.tokens(cursor)]
    if "new" not in tokens or tokens[tokens.index("new") + 1 :][:1] == ["("]:
        raise BindingError(
            cursor, "placement allocation requires an explicit storage binding"
        )
    operands = [c for c in cursor.get_children() if is_expression(c)]
    call = next(
        (
            c
            for c in operands
            if _kind(c) == "CALL_EXPR"
            and c.referenced
            and _kind(c.referenced) == "CONSTRUCTOR"
        ),
        None,
    )
    result = dict(
        op="object_new", type=spec, args=[], parameter_types=[], init_mode="default"
    )
    if call:
        construction = invoke(call, reader)
        result.update(
            args=construction["args"],
            parameter_types=construction["parameter_types"],
            symbol=construction["symbol"],
            init_mode="value",
        )
    end = min((i for i, t in enumerate(tokens) if t in {"(", "{"}), default=len(tokens))
    if "[" in tokens[:end]:
        bounds = [c for c in operands if c != call]
        if len(bounds) != 1:
            raise BindingError(
                cursor, "array object allocation requires one resolved bound"
            )
        result["extent"] = reader.expression(bounds[0])
    return result


def deletion(cursor, reader):
    children = [c for c in cursor.get_children() if is_expression(c)]
    if len(children) != 1:
        return None
    pointer = children[0].type.get_canonical()
    if type_kind(pointer) != "POINTER":
        return None
    ty = pointer.get_pointee().get_canonical()
    spec = _record_type(ty, cursor, reader.unit)
    if spec is None:
        return None
    tokens = [t.spelling for t in reader.unit.facts.tokens(cursor)]
    out = dict(
        op="object_destroy",
        value=reader.expression(children[0]),
        owner=spec["name"],
        array="[" in tokens,
    )
    for member in ty.get_declaration().get_children():
        if _kind(member) == "DESTRUCTOR":
            out["destructor"] = _require(member, reader)
    return out


def lifetime(element, reader):
    ty = _record_type(element, element.get_declaration(), reader.unit)
    if ty is None:
        return None
    result = dict(type=ty["name"])
    for member in element.get_declaration().get_children():
        if _kind(member) == "DESTRUCTOR":
            result["destructor"] = _require(member, reader)
    return result


def prepare(function, cursor, reader):
    info = function.get("method")
    if not info:
        return
    info["params"] = function["params"][1:]
    if info["kind"] != "constructor":
        return
    children = list(cursor.get_children())
    initializers, pending = [], None
    for child in children:
        kind = _kind(child)
        if kind == "COMPOUND_STMT":
            break
        if child.location.offset < cursor.location.offset or kind == "PARM_DECL":
            continue
        if kind == "MEMBER_REF":
            pending = dict(kind="member", name=child.spelling)
            field = child.referenced
            if field and _kind(field) == "FIELD_DECL":
                pending["parameter_types"] = [reader.unit.type_info(field.type, field)]
        elif kind == "TYPE_REF":
            ty = _record_type(child.type, child, reader.unit)
            if ty:
                pending = dict(
                    kind="delegate" if ty["name"] == info["owner"] else "base",
                    name=ty["name"],
                )
        elif is_expression(child) and pending:
            if (
                kind == "CALL_EXPR"
                and child.referenced
                and _kind(child.referenced) == "CONSTRUCTOR"
            ):
                _require(child.referenced, reader)
                args = _arguments(child, reader)
                pending["parameter_types"] = _parameters(child.referenced, reader)
            elif kind == "INIT_LIST_EXPR":
                args = [
                    reader.expression(c)
                    for c in child.get_children()
                    if is_expression(c)
                ]
            else:
                args = [reader.expression(child)]
            initializers.append({**pending, "args": args})
            pending = None
    function["initializers"] = initializers
