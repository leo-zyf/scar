"""Close lexical Python protocols from the bound project operation graph."""

from scar.contract import walk_operations
from . import arrays, exceptions, host_protocols, scalars, values
from .values import Value as _Value, literal as _literal


def _slot(module, name):
    return f"py:{module}:{name}"


def _join(paths):
    """Only bindings present on every control predecessor are definitely bound."""
    result = {}
    for name, first in paths[0].items():
        incoming = [path.get(name) for path in paths[1:]]
        if all(values.identical(v, first) for v in incoming):
            result[name] = first
        elif all(
            v is not None
            and v.kind == first.kind
            and values.identical(v.value, first.value)
            for v in incoming
        ):
            facts = {}
            if first.kind == "i64":
                bounds = [v.bounds for v in [first, *incoming]]
                facts["range"] = (
                    min(lo for lo, _ in bounds),
                    max(hi for _, hi in bounds),
                )
            result[name] = _Value(first.kind, first.value, facts)
    return result


def _global_bindings(functions):
    """Prove single module bindings without loop/reentrant or alias writes."""
    writes, repeated = {}, set()
    targets = {
        "assign",
        "update",
        "for",
        "python_define",
        "python_define_class",
        "python_import",
        "exception_handler",
    }
    for function in functions.values():
        pending = [(function["body"], function["python_scope"]["kind"] != "module")]
        while pending:
            node, multiple = pending.pop()
            if isinstance(node, list):
                pending.extend((item, multiple) for item in node)
                continue
            if not isinstance(node, dict):
                continue
            op = node.get("op")
            bindings = []
            if op in targets:
                bindings.append(node.get("target") or node.get("binding") or {})
            if op == "with_context":
                bindings.extend(item.get("binding") or {} for item in node["resources"])
            multiple = multiple or op in {"for", "while", "exception_handler"}
            for target in bindings:
                if target.get("op") == "var" and target.get("scope") == "global":
                    name = target["name"]
                    writes[name] = writes.get(name, 0) + 1
                    if multiple:
                        repeated.add(name)
            pending.extend((child, multiple) for child in node.values())
    return set(writes), {
        name for name, count in writes.items() if count == 1 and name not in repeated
    }


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
        self.result_kinds = {}
        self.providers = {}
        self.array_ops = set()
        self.active = []
        self.global_reads = {}
        self.shadow, self.single_bind_globals = _global_bindings(functions)

    def global_read(self, name, value):
        for active in self.active:
            if self.functions[active]["python_scope"]["kind"] == "function":
                self.global_reads[active][name] = (value.kind, value.value)

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
            for slot, required in self.global_reads.get(name, {}).items():
                current = self.names.get(slot)
                if current is None or (current.kind, current.value) != required:
                    raise _Unclosed(
                        name,
                        fn,
                        "name_binding",
                        f"cached function requires initialized binding {slot} at this call",
                    )
                self.global_read(slot, current)
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
        self.global_reads[name] = {}
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
        self.loops = []
        self.handler_depth = 0
        self.dynamic = 0
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
            self.params.append(dict(name=name, type=values.wire(kind)))
            self.env[name] = _Value(kind, None).at(name)
            self.locals[name] = kind

    def fail(self, node, field, message):
        raise _Unclosed(self.name, node, field, message)

    def scalar(self, value, node):
        if not values.runtime(value.kind):
            self.fail(
                node, "value_contract", "symbolic binding escapes its closed namespace"
            )
        return value.value

    def temporary(self):
        self.next_temp += 1
        return f"@evaluation/{self.next_temp}"

    def effect(self, node, kind, steps, discard=False, *, facts=None):
        if discard:
            steps.append(dict(op="expr", value=node))
            return _literal(None)
        name = self.temporary()
        steps.append(dict(op="decl", name=name, type=values.wire(kind), init=node))
        return _Value(kind, None, facts or {}).at(name)

    def global_value(self, name, value):
        self.project.global_read(name, value)
        if (
            self.initializing
            or not values.runtime(value.kind)
            or value.value["op"] != "var"
        ):
            return value
        facts = {}
        if name in self.project.single_bind_globals:
            if value.kind in values.SCALARS | {"numpy_f64"}:
                facts = value.facts
            elif values.rank(value.kind) is not None and "shape" in value.facts:
                # A fixed handle and immutable region layout do not prove content.
                facts = {"shape": value.facts["shape"]}
        return _Value(value.kind, value.value, facts)

    def ref(self, node):
        name = node["name"]
        if node.get("cell"):
            self.fail(node, "cell", "a lexical cell needs a closed activation lifetime")
        if node.get("scope") == "global":
            result = self.project.names.get(name)
            if result is not None:
                result = self.global_value(name, result)
        elif node["op"] == "python_builtin":
            slot = _slot(self.module, name)
            result = self.env.get(name, self.project.names.get(slot))
            if result is None and slot not in self.project.shadow:
                if name in {"print", "range"}:
                    result = _Value("builtin", name)
                elif name == "float":
                    result = _Value("dtype", "f64")
                else:
                    result = exceptions.builtin(name) or host_protocols.builtin(
                        self, node, name
                    )
                    if result is None:
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
        if target.get("op") == "index":
            arrays.index(self, target, steps, rhs=value)
            return
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
        if not values.runtime(value.kind):
            if self.dynamic or value.kind == "range":
                self.fail(
                    target,
                    "conditional_binding",
                    f"{value.kind} needs a persistent runtime binding here",
                )
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
        if self.dynamic:
            self.fail(
                node,
                "conditional_import",
                "runtime-dependent module initialization requires an import-state contract",
            )
        if (
            self.initializing
            and node.get("source_provider", {}).get("status") == "verified_stdlib"
        ):
            self.store(
                node["binding"], host_protocols.import_provider(self, node), steps
            )
            return
        if (
            self.initializing
            and mod == "numpy"
            and node.get("provider") != "project"
            and req.get("form") in {"import", "from"}
            and not req.get("level")
        ):
            self.store(node["binding"], arrays.import_provider(self, node), steps)
            return
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
            if type(value) not in (bool, int, float, str, type(None)) or (
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
        if op == "aggregate":
            return arrays.capture_sequence(self, node, steps)
        if op == "index":
            return arrays.index(self, node, steps, discard=discard)
        if op == "member":
            base = self.evaluate(node["base"], steps)
            if base.kind == "handle:text_stream":
                base = self.effect(base.value, base.kind, steps, facts=base.facts)
            host_member = host_protocols.member(self, node, base, node["field"])
            if host_member is not None:
                return host_member
            if base.kind == "provider_namespace":
                return arrays.provider_member(self, node, node["field"])
            if values.rank(base.kind) is not None and node["field"] == "copy":
                return _Value("array_copy_method", base)
            if base.kind == "module":
                slot = _slot(base.value, node["field"])
                value = self.project.names.get(slot)
                if value is not None:
                    return self.global_value(slot, value)
            self.fail(
                node,
                "attribute",
                f"unbound module or dynamic attribute {node['field']}",
            )
        if op in {"call", "library_call", "python_invoke"}:
            return self.call(node, steps, discard)
        if op == "unary" and node["operator"] != "not":
            return arrays.scalar_unary(
                self, node, self.evaluate(node["value"], steps), steps, discard
            )
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
            right_node = node["right"] if op == "binary" else node["operands"][0]
            lhs = self.evaluate(node["left"], steps)
            if (
                values.runtime(lhs.kind)
                and lhs.value["op"] != "literal"
                and right_node["op"] != "literal"
            ):
                lhs = self.effect(lhs.value, lhs.kind, steps, facts=lhs.facts)
            rhs = self.evaluate(right_node, steps)
            numeric = arrays.binary(self, node, operator, lhs, rhs, steps, discard)
            if numeric is None:
                numeric = scalars.binary(self, node, operator, lhs, rhs)
            if numeric is not None:
                if numeric.value["op"] == "scalar_power":
                    return self.effect(
                        numeric.value, numeric.kind, steps, discard, facts=numeric.facts
                    )
                return numeric
            if values.rank(lhs.kind) is not None or values.rank(rhs.kind) is not None:
                self.fail(
                    node,
                    "array_operator",
                    f"unclosed elementwise array operator {operator}",
                )
            left, right = self.scalar(lhs, node), self.scalar(rhs, node)
            equal = operator in {"eq", "noteq", "ne"}
            if lhs.kind == rhs.kind and (
                equal
                and lhs.kind in {"bool", "text", "unit"}
                or operator == "add"
                and lhs.kind == "text"
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
            if values.runtime(value.kind) and value.value["op"] != "literal":
                return self.effect(value.value, value.kind, steps, facts=value.facts)
            return value

        args = [argument(v) for v in node["args"]]
        kwargs = {}
        for item in node.get("keywords", ()):
            key = item["name"]
            if key is None or key in kwargs:
                self.fail(node, "keyword_binding", "dynamic or repeated keyword")
            kwargs[key] = argument(item["value"])
        if target.kind in {"array_constructor", "array_copy_method"}:
            return arrays.construct(self, node, target, args, kwargs, steps, discard)
        if target.kind == "exception_class":
            return exceptions.construct(
                self, node, target, args, kwargs, steps, discard
            )
        if target.kind == "builtin" and target.value == "range":
            if kwargs or not 1 <= len(args) <= 3 or any(v.kind != "i64" for v in args):
                self.fail(
                    node,
                    "range_contract",
                    "range requires one to three index-width integers",
                )
            if len(args) == 3 and (
                type(args[2].exact) is not int or args[2].exact == 0
            ):
                self.fail(node, "range_step", "a nonzero constant step is required")
            return _Value("range", tuple(args))
        if target.kind == "builtin" and target.value == "print":
            args = [self.display(value, node, steps) for value in args]
            if any(
                a.kind not in {"bool", "i64", "f64", "numpy_f64", "text", "unit"}
                for a in args
            ):
                self.fail(
                    node,
                    "print_contract",
                    "no text formatting contract for this scalar or aggregate",
                )
            if set(kwargs) - {"sep", "end", "flush", "file"}:
                self.fail(node, "print_contract", "unrecognized print keyword")
            sep, end = (kwargs.get(k, _literal(None)) for k in ("sep", "end"))
            sep = _literal(" ") if sep.kind == "unit" else sep
            end = _literal("\n") if end.kind == "unit" else end
            flush = kwargs.get("flush", _literal(False))
            stream = kwargs.get("file", _literal(None))
            if (
                sep.kind != "text"
                or end.kind != "text"
                or flush.kind != "bool"
                or stream.kind not in {"unit", "handle:text_stream"}
            ):
                self.fail(
                    node,
                    "print_contract",
                    "print requires a text stream, text separators and a Boolean flush flag",
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
                keyword_order=[
                    "stream" if k == "file" else k
                    for k in kwargs
                    if k != "file" or stream.kind != "unit"
                ],
            )
            if stream.kind != "unit":
                call["stream"] = stream.value
            if any(a.kind in {"f64", "numpy_f64"} for a in args):
                call["scalar_format"]["float"] = "shortest_binary64"
            return self.effect(call, "unit", steps, discard)
        host_call = host_protocols.invoke(
            self, node, target, args, kwargs, steps, discard
        )
        if host_call is not None:
            return host_call
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
        operands = [self.scalar(v, node) for v in bound]
        function = self.project.compile(name, tuple(v.kind for v in bound))
        call = dict(
            op="call",
            callee=name,
            args=operands,
            type=function["result"],
            parameter_types=[values.wire(v.kind) for v in bound],
            **({"noreturn": True} if function.get("noreturn") else {}),
        )
        return self.effect(call, self.project.result_kinds[name], steps, discard)

    def display(self, value, node, steps, *, representation=False):
        if value.kind in {"list", "tuple"}:
            parts = [
                self.display(item, node, steps, representation=True)
                for item in value.value
            ]
            opening, closing = ("[", "]") if value.kind == "list" else ("(", ")")
            text = [_literal(opening).value]
            for index, part in enumerate(parts):
                if index:
                    text.append(_literal(", ").value)
                text.append(part.value)
            if value.kind == "tuple" and len(parts) == 1:
                closing = ",)"
            text.append(_literal(closing).value)
            return self.effect(
                dict(op="text_concat", parts=text, type="text", effects=["VAL", "MEM"]),
                "text",
                steps,
            )
        if representation:
            if value.kind == "text":
                if type(value.exact) is not str:
                    self.fail(
                        node,
                        "sequence_repr",
                        "dynamic text in a sequence requires a Unicode repr contract",
                    )
                return _literal(repr(value.exact))
            if value.kind in {"bool", "i64", "f64", "numpy_f64", "unit"}:
                result = host_protocols.invoke(
                    self,
                    node,
                    _Value("host_callable", "builtins.str"),
                    [value],
                    {},
                    steps,
                    False,
                )
                if (
                    value.kind == "numpy_f64"
                    and self.project.providers["numpy"]["version"].split(".")[0] == "2"
                ):
                    result = _Value(
                        "text",
                        dict(
                            op="binary",
                            operator="add",
                            type="text",
                            left=_literal("np.float64(").value,
                            right=result.value,
                        ),
                    )
                    result = _Value(
                        "text",
                        dict(
                            op="binary",
                            operator="add",
                            type="text",
                            left=result.value,
                            right=_literal(")").value,
                        ),
                    )
                return result
            self.fail(
                node,
                "sequence_repr",
                f"no closed container representation for {value.kind}",
            )
        if values.rank(value.kind) is None:
            return value
        self.project.array_ops.add("array_format")
        return self.effect(
            dict(
                op="array_format",
                value=self.scalar(value, node),
                type="text",
                options=dict(
                    precision=8,
                    threshold=1000,
                    edgeitems=3,
                    linewidth=75,
                    suppress=False,
                    sign="-",
                    floatmode="maxprec",
                    legacy=False,
                ),
                effects=["VAL", "MEM", "STATE"],
            ),
            "text",
            steps,
        )

    def truth(self, value, node):
        if values.rank(value.kind) is not None:
            self.fail(
                node,
                "array_truth",
                "an array requires an explicit scalar truth or reduction contract",
            )
        expr = self.scalar(value, node)
        if "exact" in value.facts:
            return _literal(bool(value.exact))
        if value.kind == "bool":
            return value
        if value.kind == "unit":
            return _literal(False)
        if value.kind not in {"i64", "f64", "text"}:
            self.fail(
                node, "truth_contract", f"{value.kind} has no closed truth conversion"
            )
        return _Value(
            "bool",
            dict(
                op="binary",
                operator="ne",
                left=expr,
                right=_literal(
                    {"i64": 0, "f64": 0.0, "numpy_f64": 0.0, "text": ""}[value.kind]
                ).value,
                type="bool",
            ),
        )

    def snapshot(self):
        return self.env.copy(), self.project.names.copy()

    def restore(self, state):
        self.env, self.project.names = (scope.copy() for scope in state)

    def merge(self, states):
        self.env, self.project.names = (_join(group) for group in zip(*states))

    def conditional(self, node, steps):
        check = self.truth(self.evaluate(node["condition"], steps), node)
        if check.value["op"] == "literal":
            return self.sequence(
                node["then"] if check.value["value"] else node["else"], steps
            )
        start = self.snapshot()
        arms, exits = [], []
        self.dynamic += 1
        for side in ("then", "else"):
            self.restore(start)
            out = []
            if self.sequence(node[side], out):
                exits.append(self.snapshot())
            arms.append(out)
        self.dynamic -= 1
        steps.append(
            dict(op="if", condition=check.value, then=arms[0], **{"else": arms[1]})
        )
        if exits:
            self.merge(exits)
        return bool(exits)

    def range_loop(self, node, steps):
        domain = self.evaluate(node["range"], steps)
        if (
            node.get("protocol") != "iterator"
            or domain.kind != "range"
            or node["target"].get("op") != "var"
        ):
            self.fail(
                node,
                "iterator_contract",
                "counted iteration requires a closed range and one lexical target",
            )
        args = domain.value
        start, stop = (_literal(0), args[0]) if len(args) == 1 else args[:2]
        step = args[2] if len(args) == 3 else _literal(1)
        stride = step.exact
        if type(stride) is not int or not stride:
            self.fail(node, "range_step", "a nonzero constant step is required")
        # Range operands are already captured in source order by call().
        positive = stride > 0
        al, ah = start.bounds
        bl, bh = stop.bounds
        empty = al >= bh if positive else ah <= bl
        nonempty = ah < bl if positive else al > bh
        distance = bh - al if positive else ah - bl
        count = max(0, (distance + abs(stride) - 1) // abs(stride))
        low, high = (al, bh - 1) if positive else (bl + 1, ah)
        cursor = self.temporary()
        iv = _Value("i64", None, {"range": (low, high)}).at(cursor)
        condition = dict(
            op="binary",
            operator="lt" if positive else "gt",
            type="bool",
            left=iv.value,
            right=stop.value,
        )
        advance = dict(
            op="binary", operator="add", type="i64", left=iv.value, right=step.value
        )
        if not empty and (high + stride > (1 << 63) - 1 or low + stride < -(1 << 63)):
            # The terminal mathematical index may exceed int64. Substituting
            # the exclusive end when that happens leaves precisely the same visits.
            limit = (1 << 63) - 1 - stride if positive else -(1 << 63) - stride
            guard = dict(
                op="binary",
                operator="gt" if positive else "lt",
                left=iv.value,
                right=_literal(limit).value,
                type="bool",
            )
            advance = dict(
                op="select",
                condition=guard,
                then=stop.value,
                **{"else": advance},
                type="i64",
            )
        loop = dict(
            op="for",
            init=[dict(op="decl", name=cursor, type="i64", init=start.value)],
            condition=condition,
            step=dict(
                op="update", operator="set", target=iv.value, value=advance, type="i64"
            ),
            body=[],
        )
        return loop, iv, count, empty, nonempty

    def widen_loop(self, node, iv, count):
        body = node["body"]
        writes = {}
        nested = False
        for op in walk_operations(body):
            if op["op"] == "exception_handler" and (count is None or count > 1):
                alias = op.get("binding")
                if alias and alias.get("op") == "var":
                    env = (
                        self.project.names
                        if alias.get("scope") == "global"
                        else self.env
                    )
                    env.pop(alias["name"], None)
            if op["op"] in {"for", "while"}:
                nested = True
            if op["op"] in {
                "for",
                "assign",
                "update",
                "python_define",
                "python_import",
            }:
                dest = op.get("target", op.get("binding", {}))
                if dest.get("op") == "var":
                    writes.setdefault(
                        (dest.get("scope") == "global", dest["name"]), []
                    ).append(op)
        if iv is not None:
            target = node["target"]
            index = (target.get("scope") == "global", target["name"])
            writes.setdefault(index, [])
        else:
            index = None

        def delta(port):
            if port.get("op") == "literal" and type(port.get("value")) is int:
                return (port["value"],) * 2
            if port.get("op") == "var":
                key = port.get("scope") == "global", port["name"]
                if key == index and not writes[key]:
                    return iv.bounds
                binding = (self.project.names if key[0] else self.env).get(key[1])
                if key not in writes and binding and binding.kind == "i64":
                    return binding.bounds
            return None

        for (global_, name), sites in writes.items():
            env = self.project.names if global_ else self.env
            value = env.get(name)
            if value is None:
                continue
            if not values.runtime(value.kind):
                self.fail(
                    node,
                    "loop_binding",
                    f"loop-carried symbolic binding {name} needs a runtime representation",
                )
            facts = {}
            if value.kind == "i64" and count and not nested and sites:
                down = up = 0
                for site in sites:
                    rhs = site.get("value", {})
                    op = site.get("operator")
                    if (
                        site["op"] == "assign"
                        and rhs.get("op") == "binary"
                        and rhs["left"].get("op") == "var"
                        and rhs["left"]["name"] == name
                        and (rhs["left"].get("scope") == "global") == global_
                    ):
                        op, rhs = rhs["operator"], rhs["right"]
                    elif site["op"] != "update":
                        break
                    interval = delta(rhs)
                    if op not in {"add", "sub"} or interval is None:
                        break
                    a, b = interval
                    if op == "sub":
                        a, b = -b, -a
                    down += min(a, 0)
                    up += max(b, 0)
                else:
                    lower = value.bounds[0] + (count - 1) * down
                    upper = value.bounds[1] + (count - 1) * up
                    if -(1 << 63) <= lower <= upper < 1 << 63:
                        facts["range"] = (lower, upper)
            env[name] = _Value(value.kind, value.value, facts)

    def refine_loop_test(self, test):
        if test.get("op") != "binary":
            return
        op = test["operator"]
        left, right = test["left"], test["right"]
        if right.get("op") == "var" and left.get("op") == "literal":
            left, right = right, left
            op = {"gt": "lt", "ge": "le", "lt": "gt", "le": "ge", "eq": "eq"}.get(op)
        if (
            left.get("op") != "var"
            or right.get("op") != "literal"
            or type(right["value"]) is not int
        ):
            return
        env = self.project.names if left.get("scope") == "global" else self.env
        v = env.get(left["name"])
        if v is None or v.kind != "i64":
            return
        low, high = v.bounds
        b = right["value"]
        if op in {"le", "lt", "eq"}:
            high = min(high, b - (op == "lt"))
        if op in {"ge", "gt", "eq"}:
            low = max(low, b + (op == "gt"))
        if low <= high:
            env[left["name"]] = _Value("i64", v.value, {"range": (low, high)})

    def loop(self, node, steps):
        counted = node["op"] == "for"
        start = self.snapshot()
        if counted:
            loop, iv, count, empty, nonempty = self.range_loop(node, steps)
        else:
            iv, count = None, None
            loop = dict(op="while", body=[])
        self.widen_loop(node, iv, count)
        header = self.snapshot()
        test_setup = []
        flag = self.temporary() if node.get("else") else None
        if counted:
            condition = loop["condition"]
        else:
            test = self.truth(self.evaluate(node["condition"], test_setup), node)
            empty, nonempty = test.exact is False, test.exact is True
            condition = test.value
            self.refine_loop_test(condition)
        loop["condition"] = condition
        if test_setup or flag:
            # Mark exhaustion at the condition edge. A pending break can be
            # replaced by a continue or a return inside source-level finally.
            finish = []
            if flag:
                steps.append(
                    dict(op="decl", name=flag, type="bool", init=_literal(False).value)
                )
                finish.append(
                    dict(
                        op="assign",
                        target=_Value("bool", None).at(flag).value,
                        value=_literal(True).value,
                    )
                )
            loop["condition"] = _literal(True).value
            loop["body"].extend(test_setup)
            loop["body"].append(
                dict(
                    op="if",
                    condition=dict(
                        op="unary", operator="not", value=condition, type="bool"
                    ),
                    then=[*finish, dict(op="break")],
                    **{"else": []},
                )
            )
        flow = {"break": [], "continue": []}
        self.loops.append(flow)
        self.dynamic += 1
        if not empty:
            if counted:
                self.store(node["target"], iv, loop["body"])
            if self.sequence(node["body"], loop["body"]):
                flow["continue"].append(self.snapshot())
        self.dynamic -= 1
        self.loops.pop()
        broke = flow["break"]
        steps.append(loop)
        if empty:
            normal = [start]
        elif counted:
            normal = ([start] if not nonempty else []) + flow["continue"]
        else:
            normal = [] if nonempty else [header, *flow["continue"]]
        exits = list(broke)
        if normal:
            self.merge(normal)
            else_body = []
            self.dynamic += bool(broke)
            if self.sequence(node.get("else", []), else_body):
                exits.append(self.snapshot())
            self.dynamic -= bool(broke)
            if else_body:
                if broke:
                    steps.append(
                        dict(
                            op="if",
                            condition=dict(op="var", type="bool", name=flag),
                            then=else_body,
                            **{"else": []},
                        )
                    )
                else:
                    steps.extend(else_body)
        if exits:
            self.merge(exits)
        return bool(exits)

    def sequence(self, body, steps):
        for node in body:
            op = node["op"]
            first = len(steps)
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
            elif op == "raise":
                exceptions.throw(self, node, steps)
                return False
            elif op == "try":
                if not exceptions.protected(self, node, steps):
                    return False
            elif op == "with_context":
                if not exceptions.resources(self, node, steps):
                    return False
            elif op == "return":
                value = (
                    self.evaluate(node["value"], steps)
                    if "value" in node
                    else _literal(None)
                )
                steps.append(dict(op="return", value=self.scalar(value, node)))
                self.returns.add(value.kind)
                return False
            elif op == "update" and node["target"].get("op") == "var":
                target = node["target"]
                left = self.ref(target)
                left = self.effect(left.value, left.kind, steps, facts=left.facts)
                right = self.evaluate(node["value"], steps)
                if arrays.update(self, node, left, right, steps):
                    self.store(target, left, steps)
                    continue
                result = scalars.binary(self, node, node["operator"], left, right)
                if result is None:
                    result = arrays.binary(
                        self, node, node["operator"], left, right, steps
                    )
                if result is None:
                    self.fail(
                        node,
                        "update_contract",
                        "in-place operation needs a closed scalar contract",
                    )
                self.store(target, result, steps)
            elif op in {"break", "continue"} and self.loops:
                flow = self.loops[-1]
                flow[op].append(self.snapshot())
                steps.append(dict(op=op))
                return False
            elif op == "if":
                if not self.conditional(node, steps):
                    return False
            elif op in {"for", "while"}:
                if not self.loop(node, steps):
                    return False
            else:
                self.fail(node, "statement_contract", f"unclosed statement {op}")
            for statement in steps[first:]:
                if statement["op"] in {"decl", "expr", "assign"}:
                    value = (
                        statement.get("init")
                        if statement["op"] == "decl"
                        else statement.get("value")
                    )
                    if value and value.get("noreturn"):
                        return False
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
        result = "void" if self.initializing else next(iter(self.returns), "unit")
        self.project.result_kinds[self.name] = result
        parameters = {p["name"] for p in self.params}
        declarations = [
            dict(op="decl", name=n, type=values.wire(t))
            for n, t in self.locals.items()
            if n not in parameters
        ]
        return dict(
            name=self.name,
            params=self.params,
            result=values.wire(result),
            location=self.fn.get("location"),
            body=[*declarations, *body],
            **({"noreturn": True} if not falls and not self.returns else {}),
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
        single_bind_globals=sorted(project.single_bind_globals),
        status="INCOMPLETE" if issue else "READY",
        **({"first_gap": issue} if issue else {}),
        **(
            {
                "providers": list(project.providers.values()),
                "array_operations": sorted(project.array_ops),
            }
            if project.providers
            else {}
        ),
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
        dict(op="decl", name=n, type=values.wire(t)) for n, t in project.globals.items()
    )
    if project.array_ops:
        module["operator_requirements"] = {
            "ndarray": dict(
                device="cpu", element="f64", operations=sorted(project.array_ops)
            )
        }
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
