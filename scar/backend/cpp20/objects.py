"""Map object lifetime, native class layout and dispatch to C++ target declarations."""

from .model import (
    Decl,
    Expr,
    Stmt,
    Type,
    binary,
    call,
    cast,
    construct,
    member,
    named,
    template,
    unary,
    utf8_literal,
)


class ObjectModel:
    def __init__(self, names, records):
        self.names = names
        self._fields = {}
        for record in records:
            if "object_state" not in record:
                continue
            if (
                record["object_state"]
                != dict(
                    layout="fixed_fields", identity="shared", initial_fields="unbound"
                )
                or record.get("kind", "struct") not in ("record", "struct", "class")
                or any(
                    record.get(k)
                    for k in ("bases", "methods", "opaque", "provider", "packed")
                )
                or any(k in record for k in ("align_bytes", "size_bytes"))
                or not isinstance(record.get("fields"), list)
            ):
                raise ValueError("unsupported shared object record layout")
            fields = {}
            for field in record["fields"]:
                name = field.get("name")
                ty = field.get("type")
                missing = field.get("missing")
                if (
                    not isinstance(name, str)
                    or not name
                    or name in fields
                    or set(field) != {"name", "type", "missing"}
                    or not (
                        isinstance(ty, str)
                        and ty != "void"
                        or isinstance(ty, dict)
                        and ty.get("kind") == "ndarray"
                    )
                    or not isinstance(missing, dict)
                    or set(missing) != {"category", "message"}
                    or missing["category"] != "attribute_error"
                    or type(missing["message"]) is not str
                ):
                    raise ValueError("invalid shared object field descriptor")
                fields[name] = field
            self._fields[record["name"]] = fields

    def shared_record(self, name):
        return name in self._fields

    def reference(self, spec, lower):
        if (
            not isinstance(spec, dict)
            or set(spec) != {"kind", "record"}
            or spec["kind"] != "object_ref"
            or not isinstance(spec["record"], str)
            or not self.shared_record(spec["record"])
        ):
            raise ValueError("object reference requires a closed shared record")
        lower.headers.add("memory")
        return Type(
            "template", name="std::shared_ptr", args=(named(self.tag(spec["record"])),)
        )

    def shared_members(self, record, lower):
        lower.headers.add("optional")
        lower.use_sdk("exceptions.hpp")
        result = []
        for field in self._fields[record["name"]].values():
            key = field["name"]
            ty = lower.type_name(field["type"])
            slot = self.names(key, "m")
            result.append(
                Decl(
                    "field",
                    name=slot,
                    type=Type("template", name="std::optional", args=(ty,)),
                )
            )
            stored = member(Expr("this"), slot, arrow=True)
            missing = field["missing"]
            error = construct(
                named("scar::program_error"),
                utf8_literal(missing["category"], "std::string"),
                utf8_literal(missing["message"], "std::string"),
            )
            body = (
                Stmt(
                    "if",
                    condition=unary("!", stored),
                    body=(Stmt("throw", value=error),),
                ),
                Stmt("return", value=unary("*", stored)),
            )
            result.append(
                Decl(
                    "function",
                    name=self.names(key, "get"),
                    type=ty,
                    qualifiers=("const",),
                    body=body,
                )
            )
        return tuple(result)

    def shared_operation(self, expr, lower):
        op = expr["op"]
        alloc = op == "object_allocate"
        expected = {"MEM", "STATE"} if alloc else {"STATE"}
        if (
            set(expr.get("effects", ())) != expected
            or expr.get("may_raise") is not True
        ):
            raise ValueError(f"{op} requires a closed shared object effect contract")
        if alloc:
            spec = expr["type"]
            self.reference(spec, lower)
            if expr.get("args"):
                raise ValueError("unbound object allocation cannot run an initializer")
            return call(template("std::make_shared", named(self.tag(spec["record"]))))
        base = expr["base"]
        spec = base.get("type")
        self.reference(spec, lower)
        key = expr.get("field")
        fields = self._fields[spec["record"]]
        if not isinstance(key, str) or key not in fields:
            raise ValueError(f"undeclared fixed object field {key!r}")
        ty = fields[key]["type"]
        receiver = lower.expression(base)
        if op == "object_get":
            if expr.get("type") != ty:
                raise ValueError("shared object read result differs from field type")
            return call(member(receiver, self.names(key, "get"), arrow=True))
        if (
            expr.get("type") != "unit"
            or expr.get("operand_order") != ["value", "base"]
            or expr["value"].get("type") != ty
        ):
            raise ValueError("invalid shared object assignment type or operand order")
        # In C++20 assignment sequences its RHS before its LHS, including an
        # overloaded operator written as assignment. Materialize the value so
        # receiver evaluation cannot change an earlier source binding.
        value = cast(lower.type_name(ty), lower.expression(expr["value"]))
        write = binary("=", member(receiver, self.names(key, "m"), arrow=True), value)
        return binary(",", write, Expr("nullptr"))

    def tag(self, owner):
        return self.names(owner, "rec")

    def bases(self, record):
        return tuple(
            (
                "public",
                base.get("virtual", False),
                Type("name", name=self.tag(base["name"])),
            )
            for base in record.get("bases", ())
        )

    def method(self, method, lower, definition=False):
        owner, kind = self.tag(method["owner"]), method["kind"]
        if kind in {"constructor", "destructor"}:
            name = ("~" if kind == "destructor" else "") + owner
            result = None
        else:
            name = self.names(method["slot"], "method")
            result = lower.type_name(method["result"])
        parameters = tuple(
            Decl(
                "param",
                name=self.names(p["name"], "v"),
                type=lower.type_name(p["type"]),
            )
            for p in method["params"]
        )
        if method.get("variadic"):
            parameters += (Decl("param", definition="variadic"),)
        storage = tuple(
            flag
            for flag in ("explicit", "virtual")
            if not definition and method.get(flag)
        )
        qualifiers = []
        if kind != "constructor":
            qualifiers.extend(
                flag for flag in ("const", "volatile") if method.get(flag)
            )
            if method.get("ref_qualifier"):
                qualifiers.append(method["ref_qualifier"])
        if "noexcept" in method:
            qualifiers.append("noexcept" if method["noexcept"] else "noexcept(false)")
        special = (
            None
            if definition
            else next(
                (
                    value
                    for key, value in (
                        ("deleted", "delete"),
                        ("defaulted", "default"),
                        ("pure", "pure"),
                    )
                    if method.get(key)
                ),
                None,
            )
        )
        return Decl(
            "function",
            name=owner + "::" + name if definition else name,
            type=result,
            params=parameters,
            qualifiers=tuple(qualifiers),
            storage=storage,
            definition=special,
        )

    def declarations(self, record, lower):
        methods = {}
        for method in record.get("methods", ()):
            if not method.get("implicit") and method["symbol"] not in methods:
                methods[method["symbol"]] = self.method(method, lower)
        return tuple(methods.values())

    def initializers(self, function, lower):
        return tuple(
            (
                (
                    self.tag(i["name"])
                    if i["kind"] in {"base", "delegate"}
                    else self.names(i["name"], "m")
                ),
                tuple(map(lower.expression, i["args"])),
            )
            for i in function.get("initializers", ())
        )

    def expression(self, expr, lower):
        op = expr["op"]
        E = lower.expression
        if op in {"object_allocate", "object_get", "object_set"}:
            return self.shared_operation(expr, lower)
        if op in {"object_construct", "object_new"}:
            ty = lower.type_name(expr["type"])
            args = tuple(map(E, expr.get("args", ())))
            if op == "object_construct":
                return Expr("construct", type=ty, args=args, qualifier="paren")
            extent = E(expr["extent"]) if "extent" in expr else None
            return Expr(
                "new",
                type=ty,
                args=args,
                extent=extent,
                qualifier=(
                    "paren" if expr.get("init_mode", "value") == "value" else "default"
                ),
            )
        if op == "object_dispatch":
            name = self.names(expr["slot"], "method")
            if expr.get("qualified"):
                name = self.tag(expr["owner"]) + "::" + name
            method = Expr("member", args=(E(expr["receiver"]),), name=name, arrow=True)
            return Expr("call", args=(method, *map(E, expr.get("args", ()))))
        if op == "object_destroy":
            value = E(expr["value"])
            if expr.get("explicit"):
                return Expr(
                    "call",
                    args=(
                        Expr(
                            "member",
                            args=(value,),
                            name="~" + self.tag(expr["owner"]),
                            arrow=True,
                        ),
                    ),
                )
            return Expr("delete", args=(value,), array=bool(expr.get("array")))
        if op == "base_cast":
            return Expr(
                "cast",
                operator="static",
                type=lower.type_name(expr["type"]),
                args=(E(expr["value"]),),
            )
        raise ValueError(f"unsupported object operation {op!r}")
