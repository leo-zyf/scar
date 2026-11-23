"""Map object lifetime, native class layout and dispatch to C++ target declarations."""

from .model import Decl, Expr, Type


class ObjectModel:
    def __init__(self, names):
        self.names = names

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
