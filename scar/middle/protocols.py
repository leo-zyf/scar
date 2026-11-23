"""Close lexical Python protocols from the bound project operation graph."""

from dataclasses import dataclass

from scar.contract import walk_operations

_SCALARS = {"bool", "i64", "text", "unit"}


@dataclass(frozen=True)
class _Value:
    kind: str
    value: object

    def at(self, name, **attributes):
        return _Value(
            self.kind, dict(op="var", name=name, type=self.kind, **attributes)
        )


def _literal(value):
    kind = {bool: "bool", int: "i64", str: "text", type(None): "unit"}[type(value)]
    return _Value(kind, dict(op="literal", type=kind, value=value))


def _slot(module, name):
    return f"py:{module}:{name}"


class _Unclosed(Exception):
    def __init__(self, name, node, field, detail):
        self.issue = dict(
            code="python." + field,
            symbol=name,
            location=node.get("location"),
            detail=detail,
        )
        super().__init__(detail)


class _Project:
    def __init__(self, functions):
        self.functions = functions
        self.initializers = {
            scope["module"]: name
            for name, f in functions.items()
            if (scope := f.get("python_scope", {})).get("kind") == "module"
        }
        self.names = {}
        self.globals = {}
        self.types = {}
        self.closed = {}
        self.active = []
        self.shadow = set()
        for function in functions.values():
            for node in walk_operations(function["body"]):
                if node["op"] in {
                    "assign",
                    "update",
                    "python_define",
                    "python_define_class",
                    "python_import",
                }:
                    target = node.get("target", node.get("binding", {}))
                    if target.get("scope") == "global":
                        self.shadow.add(target["name"])

    def compile(self, name, types):
        fn = self.functions[name]
        if name in self.active:
            raise _Unclosed(
                name,
                fn,
                "cycle",
                "recursive calls or partially initialized modules need a recursive contract",
            )
        if name in self.closed:
            if self.types[name] != types:
                raise _Unclosed(
                    name,
                    fn,
                    "argument_types",
                    f"call types {types} conflict with earlier {self.types[name]}",
                )
            return self.closed[name]
        scope = fn.get("python_scope", {})
        kind = scope.get("kind")
        if (
            kind not in {"module", "function"}
            or kind == "function"
            and fn.get("call_protocol", {}).get("kind") != "regular"
        ):
            raise _Unclosed(
                name,
                fn,
                "activation",
                "a regular function or module initialization contract is required",
            )
        if len(types) != len(fn["params"]):
            raise _Unclosed(
                name,
                fn,
                "arity",
                "the invocation must bind every formal parameter once",
            )
        self.active.append(name)
        try:
            frame = _Frame(self, fn, types)
            result = frame.lower()
            self.types[name] = types
            self.closed[name] = result
            return result
        finally:
            self.active.pop()


class _Frame:
    def __init__(self, project, fn, types):
        self.project, self.fn = project, fn
        self.name, self.scope = fn["name"], fn["python_scope"]
        self.module = self.scope["module"]
        self.initializing = self.scope["kind"] == "module"
        self.env = {}
        self.locals = {}
        self.returns = set()
        self.next_temp = 0
        self.params = []
        for formal, kind in zip(fn["params"], types):
            if formal.get("cell") or formal.get("binding") not in {
                "positional",
                "posonly",
                "keywordonly",
            }:
                self.fail(
                    formal,
                    "formal",
                    "captured or variadic parameter needs a binding contract",
                )
            name = formal["name"]
            self.params.append(dict(name=name, type=kind))
            self.env[name] = _Value(kind, None).at(name)
            self.locals[name] = kind

    def fail(self, node, field, message):
        raise _Unclosed(self.name, node, field, message)

    def scalar(self, value, node):
        if value.kind not in _SCALARS:
            self.fail(
                node, "value_contract", "symbolic binding escapes its closed namespace"
            )
        return value.value

    def effect(self, node, kind, steps, discard=False):
        if discard:
            steps.append(dict(op="expr", value=node))
            return _literal(None)
        self.next_temp += 1
        name = f"@evaluation/{self.next_temp}"
        steps.append(dict(op="decl", name=name, type=kind, init=node))
        return _Value(kind, None).at(name)

    def ref(self, node):
        name = node["name"]
        if node.get("cell"):
            self.fail(node, "cell", "a lexical cell needs a closed activation lifetime")
        if node.get("scope") == "global":
            result = self.project.names.get(name)
        elif node["op"] == "python_builtin":
            slot = _slot(self.module, name)
            result = self.env.get(name, self.project.names.get(slot))
            if result is None and slot not in self.project.shadow:
                if name == "print":
                    result = _Value("builtin", name)
                else:
                    self.fail(
                        node,
                        "provider_contract",
                        f"builtins.{name} needs a typed provider contract",
                    )
        else:
            result = self.env.get(name)
        if result is None:
            self.fail(
                node,
                "name_binding",
                f"{name} has no definitely initialized binding at this point",
            )
        return result

    def store(self, target, value, steps):
        if target.get("op") != "var" or target.get("cell"):
            self.fail(
                target,
                "store",
                "attribute, subscript or closure assignment needs a state contract",
            )
        name = target["name"]
        global_ = target.get("scope") == "global"
        if global_ and not self.initializing:
            self.fail(target, "global_write", "a function writes a module namespace")
        if global_ and name in {
            _slot(self.module, field)
            for field in (
                "__builtins__",
                "__name__",
                "__package__",
                "__path__",
                "__spec__",
                "__loader__",
            )
        }:
            self.fail(
                target, "rebind", "assignment changes the module or import environment"
            )
        scope = self.project.names if global_ else self.env
        previous = scope.get(name)
        if value.kind not in _SCALARS:
            if previous is not None and previous != value:
                self.fail(target, "rebind", f"symbolic identity of {name} changes")
            scope[name] = value
            return
        if previous is not None and previous.kind != value.kind:
            self.fail(
                target,
                "rebind",
                f"{name} changes type from {previous.kind} to {value.kind}",
            )
        types = self.project.globals if global_ else self.locals
        if name in types and types[name] != value.kind:
            self.fail(
                target, "local_type", f"{name} has incompatible reaching value types"
            )
        types[name] = value.kind
        variable = value.at(name, **({"scope": "global"} if global_ else {}))
        scope[name] = variable
        steps.append(dict(op="assign", target=variable.value, value=value.value))

    def imported(self, node, steps):
        req = node.get("request", {})
        mod = node["module"]
        if (
            not self.initializing
            or node.get("provider") != "project"
            or req.get("form") not in {"import", "from"}
            or node["member"] == "*"
        ):
            self.fail(
                node,
                "import_contract",
                f"{mod} requires a closed project module initialized from module scope",
            )

        def initialize(symbol):
            target = self.project.functions.get(symbol, {})
            scope = target.get("python_scope", {})
            if scope.get("kind") != "module":
                self.fail(
                    node, "import_initializer", f"missing module initializer {symbol}"
                )
            if symbol not in self.project.closed:
                self.project.compile(symbol, ())
                steps.append(
                    dict(op="expr", value=dict(op="call", callee=symbol, args=[]))
                )
            parent, _, leaf = scope["module"].rpartition(".")
            if parent:
                slot = _slot(parent, leaf)
                value = _Value("module", scope["module"])
                if slot in self.project.names and self.project.names[slot] != value:
                    self.fail(
                        node,
                        "rebind",
                        f"import replaces existing package member {slot}",
                    )
                self.project.names[slot] = value

        for symbol in node["initializers"]:
            initialize(symbol)
        if self.project.initializers.get(mod) not in self.project.closed:
            self.fail(
                node, "import_initializer", f"incomplete or circular import of {mod}"
            )
        for optional in node.get("conditional_initializers", ()):
            when = optional["when"]
            if when.get("kind") != "package_attribute_missing":
                self.fail(
                    node, "import_guard", "unsupported conditional submodule binding"
                )
            if _slot(when["module"], when["name"]) not in self.project.names:
                if _slot(when["module"], "__getattr__") in self.project.names:
                    self.fail(
                        node,
                        "import_guard",
                        "package implements dynamic attribute lookup",
                    )
                initialize(optional["initializer"])
        if req["form"] == "import":
            value = _Value("module", mod if node.get("alias") else mod.split(".")[0])
        else:
            value = self.project.names.get(_slot(mod, node["member"]))
            if value is None:
                self.fail(
                    node,
                    "import_binding",
                    f"{mod} has no bound member {node['member']}",
                )
        self.store(node["binding"], value, steps)

    def evaluate(self, node, steps, discard=False):
        op = node["op"]
        if op == "literal":
            value = node["value"]
            if type(value) not in (bool, int, str, type(None)) or (
                type(value) is int and not -(1 << 63) <= value < 1 << 63
            ):
                self.fail(
                    node,
                    "literal_contract",
                    "literal requires another numeric or object representation",
                )
            if type(value) is str:
                try:
                    value.encode("utf-8")
                except UnicodeEncodeError:
                    self.fail(node, "encoding", "text contains an unpaired surrogate")
            return _literal(value)
        if op in {"var", "python_builtin"}:
            return self.ref(node)
        if op == "member":
            base = self.evaluate(node["base"], steps)
            if base.kind == "module":
                value = self.project.names.get(_slot(base.value, node["field"]))
                if value is not None:
                    return value
            self.fail(
                node,
                "attribute",
                f"unbound module or dynamic attribute {node['field']}",
            )
        if op in {"call", "library_call", "python_invoke"}:
            return self.call(node, steps, discard)
        if op == "unary" and node["operator"] == "not":
            value = self.truth(self.evaluate(node["value"], steps), node)
            if value.value["op"] == "literal":
                return _literal(not value.value["value"])
            return _Value(
                "bool", dict(op="unary", operator="not", type="bool", value=value.value)
            )
        if op in {"binary", "compare_chain"}:
            if op == "compare_chain" and len(node["operators"]) != 1:
                self.fail(
                    node,
                    "comparison_chain",
                    "multiple comparisons require short-circuit evaluation",
                )
            operator = node["operator"] if op == "binary" else node["operators"][0]
            lhs = self.evaluate(node["left"], steps)
            rhs = self.evaluate(
                node["right"] if op == "binary" else node["operands"][0], steps
            )
            left, right = self.scalar(lhs, node), self.scalar(rhs, node)
            equal = operator in {"eq", "noteq", "ne"}
            if lhs.kind == rhs.kind and (
                equal or operator == "add" and lhs.kind == "text"
            ):
                if left["op"] == right["op"] == "literal":
                    a, b = left["value"], right["value"]
                    return _literal(
                        (a == b if operator == "eq" else a != b) if equal else a + b
                    )
                return _Value(
                    "bool" if equal else "text",
                    dict(
                        op="binary",
                        operator="ne" if operator == "noteq" else operator,
                        left=left,
                        right=right,
                        type="bool" if equal else "text",
                    ),
                )
            self.fail(
                node,
                "arithmetic_contract",
                f"{operator} on {lhs.kind} and {rhs.kind} needs a closed operator and range",
            )
        self.fail(node, "operation_contract", f"unclosed expression {op}")

    def call(self, node, steps, discard):
        target = self.evaluate(node["target"], steps)

        def argument(expr):
            value = self.evaluate(expr, steps)
            if value.kind in _SCALARS and value.value["op"] not in {"var", "literal"}:
                return self.effect(value.value, value.kind, steps)
            return value

        args = [argument(v) for v in node["args"]]
        kwargs = {}
        for item in node.get("keywords", ()):
            key = item["name"]
            if key is None or key in kwargs:
                self.fail(node, "keyword_binding", "dynamic or repeated keyword")
            kwargs[key] = argument(item["value"])
        if target.kind == "builtin" and target.value == "print":
            if set(kwargs) - {"sep", "end", "flush", "file"}:
                self.fail(node, "print_contract", "unrecognized print keyword")
            sep, end = (kwargs.get(k, _literal(None)) for k in ("sep", "end"))
            sep = _literal(" ") if sep.kind == "unit" else sep
            end = _literal("\n") if end.kind == "unit" else end
            flush = kwargs.get("flush", _literal(False))
            if (
                sep.kind != "text"
                or end.kind != "text"
                or flush.kind != "bool"
                or kwargs.get("file", _literal(None)).kind != "unit"
            ):
                self.fail(
                    node,
                    "print_contract",
                    "print requires the default text output, text separators and a Boolean flush flag",
                )
            call = dict(
                op="text_print",
                args=[self.scalar(a, node) for a in args],
                sep=sep.value,
                end=end.value,
                flush=flush.value,
                type="unit",
                effects=["IO", "STATE"],
                encoding="utf-8",
                scalar_format=dict(
                    boolean_true="True",
                    boolean_false="False",
                    unit="None",
                    integer_base=10,
                ),
                keyword_order=[k for k in kwargs if k != "file"],
            )
            return self.effect(call, "unit", steps, discard)
        if target.kind != "function" or target.value not in self.project.functions:
            self.fail(
                node,
                "dispatch_contract",
                f"a callable has no reachable project body ({target.kind}: {target.value})",
            )
        name = target.value
        formals = self.project.functions[name]["params"]
        if len(args) > len(formals):
            self.fail(
                node, "argument_binding", f"too many positional arguments for {name}"
            )
        bound = []
        for i, formal in enumerate(formals):
            param, mode = formal["name"], formal.get("binding")
            if (
                i < len(args)
                and mode in {"posonly", "positional"}
                and param not in kwargs
            ):
                bound.append(args[i])
            elif (
                i >= len(args)
                and mode in {"positional", "keywordonly"}
                and param in kwargs
            ):
                bound.append(kwargs.pop(param))
            else:
                self.fail(
                    node,
                    "argument_binding",
                    f"missing, repeated or unsupported binding of {name}.{param}",
                )
        if kwargs:
            self.fail(node, "argument_binding", f"unconsumed keywords for {name}")
        values = [self.scalar(v, node) for v in bound]
        function = self.project.compile(name, tuple(v.kind for v in bound))
        call = dict(
            op="call",
            callee=name,
            args=values,
            type=function["result"],
            parameter_types=[v.kind for v in bound],
        )
        return self.effect(call, function["result"], steps, discard)

    def truth(self, value, node):
        expr = self.scalar(value, node)
        if expr["op"] == "literal":
            return _literal(bool(expr["value"]))
        if value.kind == "bool":
            return value
        if value.kind == "unit":
            return _literal(False)
        return _Value(
            "bool",
            dict(
                op="binary",
                operator="ne",
                left=expr,
                right=_literal(0 if value.kind == "i64" else "").value,
                type="bool",
            ),
        )

    def sequence(self, body, steps):
        for node in body:
            op = node["op"]
            if (
                op == "decl"
                and node.get("initial_binding") == "unbound"
                or op == "pass"
            ):
                continue
            if op == "expr":
                self.evaluate(node["value"], steps, discard=True)
            elif op == "assign":
                self.store(node["target"], self.evaluate(node["value"], steps), steps)
            elif op == "python_import":
                self.imported(node, steps)
            elif op == "python_define":
                if any(
                    node.get(k)
                    for k in ("defaults", "decorators", "annotations", "type_params")
                ) or any(x is not None for x in node.get("keyword_defaults", ())):
                    self.fail(
                        node,
                        "function_creation",
                        "definition requires default, annotation, type parameter or decorator evaluation",
                    )
                self.store(node["target"], _Value("function", node["symbol"]), steps)
            elif op == "return":
                value = (
                    self.evaluate(node["value"], steps)
                    if "value" in node
                    else _literal(None)
                )
                steps.append(dict(op="return", value=self.scalar(value, node)))
                self.returns.add(value.kind)
                return False
            elif op == "if":
                check = self.truth(self.evaluate(node["condition"], steps), node)
                if check.value["op"] == "literal":
                    branch = node["then"] if check.value["value"] else node["else"]
                    if not self.sequence(branch, steps):
                        return False
                    continue
                if self.initializing and any(
                    n["op"] in {"assign", "python_import", "python_define"}
                    for n in walk_operations([node["then"], node["else"]])
                ):
                    self.fail(
                        node,
                        "conditional_binding",
                        "runtime-dependent module binding or initialization",
                    )
                incoming = self.env
                arms, exits = [], []
                for side in ("then", "else"):
                    self.env = dict(incoming)
                    statements = []
                    if self.sequence(node[side], statements):
                        exits.append(self.env)
                    arms.append(statements)
                steps.append(
                    dict(
                        op="if",
                        condition=check.value,
                        then=arms[0],
                        **{"else": arms[1]},
                    )
                )
                if not exits:
                    return False
                self.env = {
                    key: v
                    for key, v in exits[0].items()
                    if all(out.get(key) == v for out in exits[1:])
                }
            else:
                self.fail(node, "statement_contract", f"unclosed statement {op}")
        return True

    def lower(self):
        if self.initializing:
            self.project.names[_slot(self.module, "__name__")] = _literal(
                "__main__" if self.scope["main"] else self.module
            )
        body = []
        falls = self.sequence(self.fn["body"], body)
        if falls and not self.initializing:
            body.append(dict(op="return", value=_literal(None).value))
            self.returns.add("unit")
        if len(self.returns) > 1 or self.initializing and self.returns:
            self.fail(
                self.fn,
                "result_contract",
                f"incompatible return paths {sorted(self.returns)}",
            )
        result = "void" if self.initializing else next(iter(self.returns))
        parameters = {p["name"] for p in self.params}
        declarations = [
            dict(op="decl", name=n, type=t)
            for n, t in self.locals.items()
            if n not in parameters
        ]
        return dict(
            name=self.name,
            params=self.params,
            result=result,
            location=self.fn.get("location"),
            body=[*declarations, *body],
        )


def normalize_protocols(module):
    """Certify each reachable protocol, retaining closed signatures on a partial result."""
    functions = {
        f["name"]: f
        for f in module["functions"]
        if f.get("execution_kind") == "python_protocol"
    }
    if not functions:
        return module, [], []
    project = _Project(functions)
    issue = None
    try:
        entry = module["entry"]
        if (
            entry not in functions
            or functions[entry].get("python_scope", {}).get("kind") != "module"
        ):
            raise _Unclosed(
                entry,
                functions.get(entry, {}),
                "project_entry",
                "module entry required for closed Python name binding",
            )
        project.compile(entry, ())
        for name, fn in functions.items():
            if name not in project.closed:
                raise _Unclosed(
                    name,
                    fn,
                    "reachable_protocol",
                    "no proven activation or call signature for this reachable body",
                )
        for obligation in module.get("protocol_obligations", ()):
            if obligation.get("field") not in {
                "dispatch",
                "native_binding",
                "provider_binding",
            }:
                raise _Unclosed(
                    obligation["symbol"],
                    {
                        "location": {
                            k: obligation[k]
                            for k in ("file", "line", "column")
                            if k in obligation
                        }
                    },
                    "frontend_obligation",
                    obligation["detail"],
                )
    except _Unclosed as error:
        issue = error.issue
    contracts = [
        dict(
            symbol=name,
            params=[p["type"] for p in fn["params"]],
            result=fn["result"],
        )
        for name, fn in project.closed.items()
    ]
    module["protocol_analysis"] = dict(
        entry=module["entry"],
        closed=contracts,
        status="INCOMPLETE" if issue else "READY",
        **({"first_gap": issue} if issue else {}),
    )
    if issue:
        return module, [], [issue]
    changes = [
        dict(
            **contract,
            code="closed_python_protocol",
            proof="ordered module initialization, reaching lexical bindings and consistent call types",
        )
        for contract in contracts
    ]
    module["functions"] = [
        project.closed.get(fn["name"], fn) for fn in module["functions"]
    ]
    module["globals"].extend(
        dict(op="decl", name=n, type=t) for n, t in project.globals.items()
    )
    module["externs"] = [
        e for e in module["externs"] if e.get("calling_convention") != "python"
    ]
    declarations = module["declarations"]
    for name, fn in project.closed.items():
        # Keep independent native declarations for the downstream ABI comparison.
        # Only graph-level Python object placeholders acquire the inferred contract.
        variants = [
            d
            for d in declarations.get(name, ())
            if all(
                not (isinstance(t, dict) and t.get("kind") == "python_object")
                for t in (d["result"], *(p["type"] for p in d["params"]))
            )
        ]
        declarations[name] = [
            *variants,
            {k: fn[k] for k in ("name", "params", "result")},
        ]
    module.pop("protocol_obligations", None)
    return module, changes, []
