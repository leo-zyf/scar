"""Lower identity-bearing aggregate values and native class lifetimes to C++."""

from scar.contract import digest, json_bytes, product_items

from .model import (
    Decl,
    Expr,
    Stmt,
    Type,
    binary,
    call,
    cast,
    construct,
    declaration,
    integer,
    member,
    name,
    named,
    template,
    unary,
    utf8_literal,
    variable,
)


class ProductModel:
    """Immutable, heterogeneous aggregates with allocation identity."""

    def __init__(self):
        self.constants = {}

    @staticmethod
    def contains(spec):
        if isinstance(spec, dict):
            return spec.get("kind") == "product_ref" or any(
                ProductModel.contains(v) for v in spec.values()
            )
        return isinstance(spec, list) and any(ProductModel.contains(v) for v in spec)

    @classmethod
    def in_signature(cls, fn):
        return cls.contains(fn.get("result")) or any(
            cls.contains(p.get("type")) for p in fn.get("params", ())
        )

    @staticmethod
    def _tuple(spec, lower):
        lower.headers.update(("memory", "tuple"))
        return Type(
            "template",
            name="std::tuple",
            args=tuple(lower.type_name(t) for t in product_items(spec)),
        )

    def reference(self, spec, lower):
        value = self._tuple(spec, lower)
        return Type(
            "template", name="std::shared_ptr", args=(Type("const", element=value),)
        )

    def make(self, node, lower):
        spec = node["type"]
        fields = product_items(spec)
        values = node.get("items")
        if (
            not isinstance(values, list)
            or len(values) != len(fields)
            or any(
                not isinstance(v, dict) or v.get("type") != t
                for v, t in zip(values, fields)
            )
            or set(node.get("effects", ())) != {"MEM", "STATE"}
            or node.get("may_raise") is not True
        ):
            raise ValueError("invalid product construction type, arity or effects")
        identity = node.get("identity")
        key = None
        if identity != {"kind": "fresh"}:
            if (
                not isinstance(identity, dict)
                or set(identity) != {"kind", "pool", "slot"}
                or identity["kind"] != "constant_pool"
                or not isinstance(identity["pool"], str)
                or not identity["pool"]
                or type(identity["slot"]) is not int
                or identity["slot"] < 0
            ):
                raise ValueError(
                    "product requires an explicit allocation or constant pool identity"
                )
            key = (identity["pool"], identity["slot"])
            if not all(
                v.get("op") == "literal"
                or (
                    v.get("op") == "product_make"
                    and isinstance(v.get("identity"), dict)
                    and v["identity"].get("kind") == "constant_pool"
                )
                for v in values
            ):
                raise ValueError(
                    "product constant pool requires literal or pooled constant members"
                )

        storage = self._tuple(spec, lower)
        # Braces sequence explicit value captures before the allocation call.
        members = (cast(t, lower.expression(v)) for t, v in zip(storage.args, values))
        payload = construct(storage, *members, form="brace")
        init = call(
            template("std::make_shared", Type("const", element=storage)), payload
        )
        if key is None:
            return init
        symbol = "slot_" + digest(json_bytes(key))
        if key in self.constants:
            wire_type, registered, _ = self.constants[key]
            if wire_type != spec or registered != init:
                raise ValueError(
                    "conflicting type or contents for product constant pool slot"
                )
        else:
            # An inline function's local static is one object across translation units.
            handle = self.reference(spec, lower)
            local = variable(
                "value", Type("const", element=handle), init, storage=("static",)
            )
            accessor = Decl(
                "function",
                name=symbol,
                type=handle,
                storage=("inline",),
                body=(declaration(local), Stmt("return", value=name("value"))),
            )
            self.constants[key] = (spec, init, accessor)
        return call("::scar::generated_products::" + symbol)

    def expression(self, node, lower):
        op = node["op"]
        if op == "product_make":
            return self.make(node, lower)
        if set(node.get("effects", ())) != {"STATE"}:
            raise ValueError("invalid product read effects")
        if op == "product_get":
            base = node["base"]
            members = product_items(base.get("type"))
            index = node.get("index")
            if (
                type(index) is not int
                or not 0 <= index < len(members)
                or members[index] != node.get("type")
            ):
                raise ValueError(
                    "product projection requires an in-range index and matching result type"
                )
            self.reference(base["type"], lower)
            return call(
                template("std::get", integer(index)), unary("*", lower.expression(base))
            )
        if (
            op != "product_identity"
            or node.get("type") != "bool"
            or node.get("operator") not in {"eq", "ne"}
        ):
            raise ValueError("invalid product identity comparison")
        operands = (node["left"], node["right"])
        types = tuple(self.reference(v.get("type"), lower) for v in operands)
        # Keep both owners alive until address comparison, even if RHS rebinds the LHS.
        held = construct(
            Type("template", name="std::tuple", args=types),
            *(cast(t, lower.expression(v)) for t, v in zip(types, operands)),
            form="brace",
        )
        address = Type("ptr", element=Type("const", element=named("void")))
        borrowed = Type("ref", element=Type("const", element=Type("auto")))
        params = tuple(Decl("param", name=n, type=borrowed) for n in ("left", "right"))
        addresses = (cast(address, call(member(name(p.name), "get"))) for p in params)
        result = binary("==" if node["operator"] == "eq" else "!=", *addresses)
        compare = Expr("lambda", params=params, body=(Stmt("return", value=result),))
        return call("std::apply", compare, held)

    def declarations(self):
        if not self.constants:
            return ()
        return (
            Decl(
                "namespace",
                name="scar::generated_products",
                members=tuple(decl for _, _, decl in self.constants.values()),
            ),
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
