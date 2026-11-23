"""Lower typed object lifetime and dispatch operations into C++20 classes."""


class ObjectModel:
    def __init__(self, module, names):
        self.names = names
        self.records = {r["name"]: r for r in module.get("records", ())}

    def tag(self, owner):
        return self.names(owner, "rec")

    def bases(self, record):
        bases = [
            "public " + ("virtual " if b.get("virtual") else "") + self.tag(b["name"])
            for b in record.get("bases", ())
        ]
        return " : " + ", ".join(bases) if bases else ""

    def _signature(self, method, emitter, *, definition=False):
        owner = self.tag(method["owner"])
        kind = method["kind"]
        prefix = owner + "::" if definition else ""
        if kind in {"constructor", "destructor"}:
            name = ("~" if kind == "destructor" else "") + owner
            result = ""
        else:
            name = self.names(method["slot"], "method")
            result = emitter.type_name(method["result"]) + " "
        args = [
            emitter.type_name(p["type"]) + " " + self.names(p["name"], "v")
            for p in method["params"]
        ]
        if method.get("variadic"):
            args.append("...")
        qualifiers = ""
        if kind != "constructor":
            qualifiers += " const" if method.get("const") else ""
            qualifiers += " volatile" if method.get("volatile") else ""
            qualifiers += (
                " " + method["ref_qualifier"] if method.get("ref_qualifier") else ""
            )
        if "noexcept" in method:
            qualifiers += " noexcept" if method["noexcept"] else " noexcept(false)"
        virtual = "virtual " if method.get("virtual") and not definition else ""
        explicit = "explicit " if method.get("explicit") and not definition else ""
        return (
            f"{explicit}{virtual}{result}{prefix}{name}({', '.join(args)}){qualifiers}"
        )

    def declarations(self, record, emitter):
        declarations, seen = [], set()
        for method in record.get("methods", ()):
            if method.get("implicit"):
                continue
            key = method["symbol"]
            if key in seen:
                continue
            seen.add(key)
            signature = self._signature(method, emitter)
            if method.get("deleted"):
                signature += " = delete"
            elif method.get("defaulted"):
                signature += " = default"
            elif method.get("pure"):
                signature += " = 0"
            declarations.append(signature + ";")
        return declarations

    def signature(self, function, emitter):
        if not function.get("method"):
            return None
        return self._signature(function["method"], emitter, definition=True)

    def initializers(self, function, emitter):
        entries = []
        for item in function.get("initializers", ()):
            target = (
                self.tag(item["name"])
                if item["kind"] in {"base", "delegate"}
                else self.names(item["name"], "m")
            )
            arguments = ", ".join(emitter.expression(a) for a in item["args"])
            entries.append(f"{target}({arguments})")
        return " : " + ", ".join(entries) if entries else ""

    def expression(self, expr, emitter):
        op = expr["op"]
        if op not in {
            "object_construct",
            "object_new",
            "object_dispatch",
            "object_destroy",
            "base_cast",
        }:
            return None
        value = emitter.expression
        if op in {"object_construct", "object_new"}:
            target = emitter.type_name(expr["type"])
            args = ", ".join(map(value, expr.get("args", ())))
            if op == "object_construct":
                return f"{target}({args})"
            extent = "[" + value(expr["extent"]) + "]" if "extent" in expr else ""
            init = f"({args})" if expr.get("init_mode", "value") == "value" else ""
            return f"new {target}{extent}{init}"
        if op == "object_dispatch":
            method = self.names(expr["slot"], "method")
            if expr.get("qualified"):
                method = self.tag(expr["owner"]) + "::" + method
            args = ", ".join(map(value, expr.get("args", ())))
            return f"({value(expr['receiver'])})->{method}({args})"
        if op == "object_destroy":
            if expr.get("explicit"):
                return f"({value(expr['value'])})->~{self.tag(expr['owner'])}()"
            return (
                "delete"
                + ("[] " if expr.get("array") else " ")
                + f"({value(expr['value'])})"
            )
        return f"static_cast<{emitter.type_name(expr['type'])}>({value(expr['value'])})"
