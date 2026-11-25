"""M2 list allocation, homogeneous element constraints and live iteration.

Type variables describe only an element type, never a runtime object identity.
They are private to one project normalization and must disappear before sealing.
"""

from copy import deepcopy

from scar.contract import walk_operations
from . import values
from .values import Value, literal


def _slot(function, node):
    return ("global" if node.get("scope") == "global" else function, node["name"])


class Domains:
    def __init__(self, functions, single_bind_globals):
        self.bindings, self.parents, self.sources, self.next_state, self.states = (
            {},
            {},
            {},
            0,
            {},
        )
        self.functions, self.single_bind_globals = functions, single_bind_globals
        self.namespaces, self.callables = self._imports(functions, single_bind_globals)
        self.static_lists = self._static_uses(functions, single_bind_globals)

    def _imports(self, functions, single_bind_globals):
        namespaces, callables = {}, {}
        for function in functions.values():
            for node in walk_operations(function["body"]):
                if (
                    node["op"] != "python_import"
                    or node.get("binding", {}).get("op") != "var"
                ):
                    continue
                binding = node["binding"]
                if (
                    binding.get("scope") != "global"
                    or binding["name"] not in single_bind_globals
                ):
                    continue
                key = _slot(function["name"], binding)
                provider, module = node.get("source_provider", {}), node["module"]
                resolved = node.get("import_resolution", {})
                binding_value = node.get("bound_value", {})
                if (
                    node.get("provider") == "project"
                    and resolved.get("status") == "project"
                    and resolved.get("module") == module
                    and node["request"]["form"] == "import"
                    and binding_value.get("class") == "module"
                    and any(
                        function["python_scope"].get("kind") == "module"
                        and function["python_scope"].get("module")
                        == binding_value.get("module")
                        for function in functions.values()
                    )
                ):
                    namespaces[key] = "project:" + binding_value["module"]
                verified = (
                    provider.get("module") == module
                    and isinstance(provider.get("sha256"), str)
                    and len(provider["sha256"]) == 64
                    and provider.get("origin")
                )
                verified = verified and (
                    module == "numpy"
                    and provider.get("status") == "verified_distribution"
                    and provider.get("distribution", "").lower() == "numpy"
                    or module == "subprocess"
                    and provider.get("status") == "verified_stdlib"
                )
                if verified:
                    if node["request"]["form"] == "import":
                        namespaces[key] = node["module"]
                    elif node.get("member") in {
                        "array",
                        "zeros",
                        "meshgrid",
                        "linspace",
                        "Popen",
                    }:
                        callables[key] = module + "." + node["member"]
        return namespaces, callables

    def bind_frame(self, function, types):
        arrays = {
            _slot(function["name"], dict(name=p["name"]))
            for p, kind in zip(function.get("params", ()), types)
            if values.is_array(kind)
        }
        if arrays:
            self.static_lists.update(
                self._static_uses(
                    {function["name"]: function},
                    self.single_bind_globals,
                    array_parameters=arrays,
                )
            )

    def _static_uses(
        self, functions, single_bind_globals, *, array_parameters=frozenset()
    ):
        """Prove exclusive static-consumer uses before choosing representation."""
        incoming, lists, used, runtime = {}, set(), set(), set()
        namespaces, callables = self.namespaces, self.callables
        array_slots, assignments = set(array_parameters), {}
        bound = {
            _slot(function["name"], target)
            for function in self.functions.values()
            for operation in walk_operations(function["body"])
            for target in values.written_targets(operation)
            if target.get("op") == "var"
        }
        bound.update(
            (function["name"], parameter["name"])
            for function in self.functions.values()
            for parameter in function.get("params", ())
        )

        def link(source, target):
            if source is not None and target is not None:
                incoming.setdefault(target, set()).add(source)

        def origin(function, node):
            if not isinstance(node, dict):
                return None
            if node.get("op") == "var":
                return _slot(function, node)
            if node.get("op") == "aggregate" and node.get("kind") == "list":
                token = ("list", id(node))
                lists.add(token)
                return token
            if node.get("op") == "member":
                namespace = namespaces.get(origin(function, node.get("base")), "")
                if namespace.startswith("project:"):
                    return ("global", f"py:{namespace[8:]}:{node['field']}")
            if node.get("op") == "index":
                token = ("projection", id(node))
                link(origin(function, node.get("base")), token)
                return token
            return None

        def static_call(function, target):
            if target.get("op") == "python_builtin" and target["name"] in {
                "print",
                "str",
                "repr",
                "ascii",
                "format",
                "len",
            }:
                module = self.functions[function]["python_scope"]["module"]
                return (function, target["name"]) not in bound and (
                    "global",
                    f"py:{module}:{target['name']}",
                ) not in bound
            if target.get("op") in {"var", "python_builtin"}:
                return _slot(function, target) in callables
            if target.get("op") == "member":
                return (
                    target["field"] == "reshape"
                    and origin(function, target.get("base")) in array_slots
                    or (
                        namespaces.get(origin(function, target.get("base"))) == "numpy"
                        and target["field"]
                        in {"array", "zeros", "meshgrid", "linspace"}
                    )
                    or namespaces.get(origin(function, target.get("base")))
                    == "subprocess"
                    and target["field"] == "Popen"
                )
            return False

        def array_source(function, source):
            if source.get("op") == "var":
                return origin(function, source) in array_slots
            if source.get("op") not in {
                "python_invoke",
                "library_call",
                "call",
            } or not isinstance(source.get("target"), dict):
                return False
            target = source["target"]
            known = callables.get(origin(function, target)) in {
                "numpy.array",
                "numpy.zeros",
                "numpy.linspace",
            }
            known |= target.get("op") == "member" and (
                namespaces.get(origin(function, target.get("base"))) == "numpy"
                and target["field"] in {"array", "zeros", "linspace"}
                or target["field"] == "reshape"
                and origin(function, target.get("base")) in array_slots
            )
            return known and all(
                keyword["name"] != "retstep"
                or keyword["value"].get("op") == "literal"
                and keyword["value"].get("value") is False
                for keyword in source.get("keywords", ())
            )

        for function in functions.values():
            for operation in walk_operations(function["body"]):
                if operation["op"] in {"assign", "decl"}:
                    if operation["op"] == "decl" and "init" not in operation:
                        continue
                    target = (
                        operation if operation["op"] == "decl" else operation["target"]
                    )
                    if target.get("op", "var") in {"var", "decl"}:
                        assignments.setdefault(
                            _slot(function["name"], target), []
                        ).append(
                            (
                                function["name"],
                                operation.get(
                                    "value" if operation["op"] == "assign" else "init",
                                    {},
                                ),
                            )
                        )
        while True:
            proven = {
                key
                for key, sources in assignments.items()
                if sources
                and all(array_source(function, source) for function, source in sources)
            }
            if proven <= array_slots:
                break
            array_slots.update(proven)

        def contained(function, node):
            ref = origin(function, node)
            if ref is not None:
                return {ref}
            if isinstance(node, dict) and node.get("op") == "aggregate":
                return set().union(
                    *(contained(function, item) for item in node["items"])
                )
            return set()

        def scan(function, node, role="runtime"):
            if isinstance(node, list):
                for child in node:
                    scan(function, child, role)
                return
            if not isinstance(node, dict):
                return
            op, ref = node.get("op"), origin(function, node)
            if ref is not None:
                if role == "runtime":
                    runtime.add(ref)
                elif role == "static":
                    used.add(ref)
            if op == "aggregate" and node.get("kind") == "list":
                for item in node["items"]:
                    for source in contained(function, item):
                        link(source, ref)
                    scan(function, item, "alias")
                return
            if op in {"assign", "decl"}:
                target = node if op == "decl" else node["target"]
                rhs = node.get("init" if op == "decl" else "value", {})
                if target.get("op", "var") in {"var", "decl"}:
                    for source in contained(function, rhs):
                        link(source, _slot(function, target))
                    scan(function, rhs, "alias")
                else:
                    scan(function, target)
                    # A proven ndarray receiver consumes an indexed-store
                    # sequence immediately; it does not retain the list object.
                    # Other consumers and aliases still seed runtime escape.
                    array_store = (
                        target.get("op") == "index"
                        and origin(function, target.get("base")) in array_slots
                    )
                    scan(function, rhs, "static" if array_store else "runtime")
                return
            if op == "python_import" and node.get("provider") == "project":
                resolved = node.get("import_resolution", {})
                if (
                    node.get("member")
                    and resolved.get("status") == "project"
                    and resolved.get("module") == node["module"]
                    and node.get("binding", {}).get("op") == "var"
                ):
                    link(
                        ("global", f"py:{node['module']}:{node['member']}"),
                        origin(function, node["binding"]),
                    )
                return
            if op == "format_field":
                scan(function, node["value"], "static")
                scan(function, node.get("format"))
                return
            if op == "binary" and node["operator"] in {
                "add",
                "sub",
                "mult",
                "mul",
                "div",
                "floordiv",
                "mod",
                "pow",
                "bitand",
                "bitor",
                "bitxor",
                "lshift",
                "rshift",
            }:
                scan(function, node["left"], "static")
                scan(function, node["right"], "static")
                return
            if op == "unary" and node["operator"] in {
                "usub",
                "uadd",
                "neg",
                "pos",
                "not",
                "invert",
            }:
                scan(function, node["value"], "static")
                return
            if op in {"if", "while"}:
                scan(function, node["condition"], "static")
                for field in ("then", "body", "else"):
                    scan(function, node.get(field, ()))
                return
            if op == "index":
                # A selector is a dataflow vertex, not an unconditional escape.
                # Its consumers decide whether the selected object can remain
                # static; alias and mutation uses propagate back to its base.
                indices = node.get("indices", ())
                constant = lambda item: (
                    item.get("op") == "literal"
                    and type(item.get("value")) is int
                    or item.get("op") == "unary"
                    and item.get("operator") in {"usub", "uadd", "neg", "pos"}
                    and item.get("value", {}).get("op") == "literal"
                    and type(item["value"].get("value")) is int
                )
                if len(indices) != 1 or not constant(indices[0]):
                    runtime.add(ref)
                scan(function, node["base"], "alias")
                scan(function, indices, "static")
                return
            if op in {
                "python_invoke",
                "library_call",
                "call",
                "process_launch",
            } and isinstance(node.get("target"), dict):
                scan(function, node["target"])
                use = "static" if static_call(function, node["target"]) else "runtime"
                scan(function, node.get("args", []), use)
                for keyword in node.get("keywords", ()):
                    scan(function, keyword["value"], use)
                return
            for key, child in node.items():
                if key not in {
                    "type",
                    "source_evidence",
                    "bound_value",
                    "binding",
                    "location",
                }:
                    scan(function, child, role)

        for function in functions.values():
            scan(function["name"], function["body"])

        def ancestors(seeds):
            found, pending = set(), list(seeds)
            while pending:
                item = pending.pop()
                if item not in found:
                    found.add(item)
                    pending.extend(incoming.get(item, ()))
            return found

        return {token[1] for token in lists & ancestors(used) - ancestors(runtime)}

    def new(self, frame, node):
        index = len(self.bindings)
        self.bindings[index] = None
        self.parents[index] = index
        self.sources[index] = frame.name, values.source_site(node)
        return values.sequence_variable(index)

    def _root(self, identity):
        while self.parents[identity] != identity:
            self.parents[identity] = self.parents[self.parents[identity]]
            identity = self.parents[identity]
        return identity

    def resolve_kind(self, kind, *, strict=False, active=()):
        variable = values.sequence_variable_id(kind)
        if variable is not None:
            variable = self._root(variable)
            if variable in active:
                raise ValueError(
                    "recursive sequence element constraints have no finite type"
                )
            element = self.bindings[variable]
            if element is None:
                if strict:
                    raise ValueError(
                        f"sequence element remains unresolved at {self.sources[variable]}"
                    )
                return values.sequence_variable(variable)
            return values.sequence_kind(
                self.resolve_kind(element, strict=strict, active=(*active, variable))
            )
        if values.is_sequence(kind):
            return values.sequence_kind(
                self.resolve_kind(
                    values.sequence_element(kind), strict=strict, active=active
                )
            )
        return kind

    def constrain(self, kind, element):
        variable = values.sequence_variable_id(kind)
        if variable is not None:
            variable = self._root(variable)
            previous = self.bindings[variable]
            if previous is None:
                if self._occurs(variable, element):
                    raise ValueError("recursive list element types are unsupported")
                self.bindings[variable] = element
            else:
                self.unify(previous, element)
        else:
            self.unify(values.sequence_element(kind), element)

    def _occurs(self, variable, kind, seen=frozenset()):
        other = values.sequence_variable_id(kind)
        if other is not None:
            other = self._root(other)
            return (
                other == variable
                or other not in seen
                and self.bindings[other] is not None
                and self._occurs(variable, self.bindings[other], seen | {other})
            )
        return values.is_sequence(kind) and self._occurs(
            variable, values.sequence_element(kind), seen
        )

    def unify(self, left, right):
        if left == right:
            return
        if values.is_sequence(left) and values.is_sequence(right):
            a, b = values.sequence_variable_id(left), values.sequence_variable_id(right)
            a = self._root(a) if a is not None else None
            b = self._root(b) if b is not None else None
            if a is not None and self.bindings[a] is None:
                if b is not None and self.bindings[b] is None:
                    # Two element cells share a constraint; their lists do not
                    # thereby share an allocation or runtime identity.
                    self.parents[a] = b
                else:
                    self.constrain(left, self.element(right))
            elif b is not None and self.bindings[b] is None:
                self.constrain(right, self.element(left))
            else:
                self.unify(self.element(left), self.element(right))
            return
        raise ValueError(f"heterogeneous sequence elements {left} and {right}")

    def element(self, kind):
        variable = values.sequence_variable_id(kind)
        if variable is not None:
            variable = self._root(variable)
            element = self.bindings[variable]
            if element is None:
                raise ValueError("a sequence consumer needs a concrete element type")
            return self.resolve_kind(element)
        return values.sequence_element(kind)

    def compatible(self, left, right):
        if values.is_sequence(left) or values.is_sequence(right):
            if not values.is_sequence(left) or not values.is_sequence(right):
                return False
            try:
                self.unify(left, right)
                return True
            except ValueError:
                return False
        return values.storage_compatible(left, right)

    def resolve(self, node, *, strict=True):
        if isinstance(node, list):
            return [self.resolve(child, strict=strict) for child in node]
        if isinstance(node, dict):
            if node.get("kind") == "m2_typevar":
                variable = self._root(node["id"])
                if self.bindings[variable] is None and not strict:
                    return dict(kind="m2_typevar", id=variable)
                return values.wire(
                    self.resolve_kind(
                        self.element(values.sequence_variable(variable)), strict=strict
                    )
                )
            return {
                key: self.resolve(child, strict=strict) for key, child in node.items()
            }
        return node

    def freeze(self, project, *, partial=False):
        closed = {}
        for name, function in project.closed.items():
            try:
                closed[name] = self.resolve(function)
            except ValueError:
                if not partial:
                    raise
        project.closed = closed
        prefixes = []
        for fragment in project.prefixes:
            try:
                prefixes.append(self.resolve(fragment))
            except ValueError:
                continue
        project.prefixes = prefixes
        if not partial:
            project.globals = {
                name: self.resolve_kind(kind, strict=True)
                for name, kind in project.globals.items()
            }
            project.types = {
                name: tuple(self.resolve_kind(kind, strict=True) for kind in kinds)
                for name, kinds in project.types.items()
            }
            project.result_kinds = {
                name: self.resolve_kind(kind, strict=True)
                for name, kind in project.result_kinds.items()
            }

    def register(self, safe):
        self.next_state += 1
        self.states[self.next_state] = bool(safe)
        return dict(sequence_state=self.next_state)

    def safe(self, value):
        return self.states.get(value.facts.get("sequence_state"), False)

    def invalidate(self):
        for token in self.states:
            self.states[token] = False


def _checked(frame, node, operation):
    try:
        return operation()
    except ValueError as error:
        frame.fail(node, "sequence_element_type", str(error))


def text_order_atom(value):
    return value.kind == "text" and (
        value.facts.get("text_order_atom") is True
        or type(value.exact) is str
        and not any(0xD800 <= ord(c) <= 0xDFFF for c in value.exact)
    )


def produced(
    frame, node, element, operation, steps, *, text_order=False, discard=False
):
    kind = values.sequence_kind(element)
    return frame.effect(
        operation,
        kind,
        steps,
        discard,
        facts=frame.project.sequences.register(text_order),
    )


def capture(frame, node, items, steps):
    domain = frame.project.sequences
    kind = domain.new(frame, node)
    for item in items:
        if not values.runtime(item.kind):
            frame.fail(
                node,
                "sequence_element_type",
                "runtime list members require managed values",
            )
        _checked(frame, node, lambda: domain.constrain(kind, item.kind))
    operation = dict(
        op="sequence_make",
        type=values.wire(kind),
        items=[item.value for item in items],
        effects=["MEM", "STATE"],
        may_raise=True,
        **values.source_site(node),
    )
    return frame.effect(
        operation,
        kind,
        steps,
        facts=domain.register(all(text_order_atom(item) for item in items)),
    )


def member(frame, node, base, name):
    if not values.is_sequence(base.kind):
        return None
    if name != "append":
        frame.fail(node, "sequence_member", f"no mutable sequence protocol for {name}")
    return Value("sequence_method", (base, name))


def comprehension(frame, node, steps, lower_generators):
    """Build one shared list with ordered, scoped generator/filter regions."""
    if node.get("kind") != "list":
        frame.fail(
            node, "sequence_comprehension", "only eager list construction is closed"
        )

    def make_result(body):
        return capture(frame, node, [], body)

    def put_entry(result, active, body):
        value = active.evaluate(node["element"], body)
        invoke(
            active,
            node,
            Value("sequence_method", (result, "append")),
            [value],
            {},
            body,
            True,
        )

    return lower_generators(node, steps, make_result, put_entry)


def persist_method(frame, target, method, steps):
    receiver, name = method.value
    if target.get("scope") == "global":
        slot = "@sequence-method/" + target["name"]
        frame.project.globals[slot] = receiver.kind
        captured = receiver.at(slot, scope="global")
        steps.append(dict(op="assign", target=captured.value, value=receiver.value))
    else:
        captured = frame.effect(
            receiver.value, receiver.kind, steps, facts=receiver.facts
        )
    return Value("sequence_method", (captured, name))


def identity(node, left, right):
    if not any(values.is_sequence(item.kind) for item in (left, right)):
        return None
    if not all(values.is_sequence(item.kind) for item in (left, right)):
        return literal(node["operator"] == "isnot")
    return Value(
        "bool",
        dict(
            op="sequence_identity",
            type="bool",
            left=left.value,
            right=right.value,
            operator="eq" if node["operator"] == "is" else "ne",
            effects=["STATE"],
            **values.source_site(node),
        ),
    )


def invoke(frame, node, target, args, keywords, steps, discard):
    domain = frame.project.sequences
    if target.kind == "sequence_method":
        if keywords or len(args) != 1:
            frame.fail(node, "sequence_append", "append requires exactly one element")
        base = target.value[0]
        _checked(frame, node, lambda: domain.constrain(base.kind, args[0].kind))
        token = base.facts.get("sequence_state")
        if token in domain.states and not text_order_atom(args[0]):
            domain.states[token] = False
        frame.effect(
            dict(
                op="sequence_append",
                type="unit",
                base=base.value,
                value=frame.scalar(args[0], node),
                effects=["MEM", "STATE"],
                may_raise=True,
                reads=["sequence.state"],
                writes=["sequence.state"],
                **values.source_site(node),
            ),
            "unit",
            steps,
            True,
        )
        return literal(None)
    if target.kind == "builtin" and target.value == "sorted":
        if keywords or len(args) != 1 or not values.is_sequence(args[0].kind):
            frame.fail(
                node,
                "sequence_sort",
                "sorted requires one homogeneous sequence with default options",
            )
        source = args[0]
        element = _checked(frame, node, lambda: domain.element(source.kind))
        if (
            element not in {"i64", "text"}
            or element == "text"
            and not domain.safe(source)
        ):
            frame.fail(
                node,
                "sequence_sort",
                "sorting requires signed integers or proven codepoint-comparable text atoms",
            )
        operation = dict(
            op="sequence_sorted_copy",
            type=values.wire(source.kind),
            base=source.value,
            comparison="signed_integer" if element == "i64" else "unicode_codepoint",
            effects=["MEM", "STATE", "VAL"],
            may_raise=True,
            reads=["sequence.state"],
            writes=["new.sequence"],
            **values.source_site(node),
        )
        return frame.effect(
            operation,
            source.kind,
            steps,
            discard,
            facts=domain.register(element == "text" and domain.safe(source)),
        )
    return None


def size(value):
    return Value(
        "i64",
        dict(
            op="sequence_size",
            type="i64",
            base=value.value,
            effects=["STATE"],
            reads=["sequence.state"],
        ),
    )


def get(frame, node, base, index):
    element = _checked(frame, node, lambda: frame.project.sequences.element(base.kind))
    if index.kind != "i64":
        frame.fail(node, "sequence_index", "a sequence index requires an i64 value")
    return Value(
        element,
        dict(
            op="sequence_get",
            type=values.wire(element),
            base=base.value,
            index=index.value,
            source_index_profile="python_list_v1",
            effects=["STATE"],
            may_raise=True,
            reads=["sequence.state"],
            **values.source_site(node),
        ),
        (
            dict(text_order_atom=True)
            if element == "text" and frame.project.sequences.safe(base)
            else {}
        ),
    )


def iteration(frame, node, domain, steps):
    _checked(frame, node, lambda: frame.project.sequences.element(domain.kind))
    captured = frame.effect(domain.value, domain.kind, steps, facts=domain.facts)
    cursor = Value("i64", None).at(frame.temporary())
    loop = dict(
        op="for",
        init=[
            dict(
                op="decl", name=cursor.value["name"], type="i64", init=literal(0).value
            )
        ],
        condition=dict(
            op="binary",
            operator="lt",
            type="bool",
            left=cursor.value,
            right=size(captured).value,
        ),
        step=dict(
            op="update",
            operator="add",
            type="i64",
            target=cursor.value,
            value=literal(1).value,
        ),
        body=[],
    )
    return loop, get(frame, node, captured, cursor), None, False, False, []
