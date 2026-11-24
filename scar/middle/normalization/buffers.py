"""M2 reaching byte-buffer versions and command requests over canonical operations."""

from dataclasses import dataclass, field, replace
import re

from scar.contract import (
    byte_value_id,
    control_regions,
    digest,
    json_bytes,
    walk_operations,
)

# F2 supplies the actual selected C ABI, role positions and its byte semantics.
# Only these public bindings have a transfer function at this layer.
_LIBC = {
    "sprintf": ("stdio.h", "bytes.format_into"),
    "snprintf": ("stdio.h", "bytes.format_into"),
    "strcpy": ("string.h", "bytes.copy"),
    "strncpy": ("string.h", "bytes.copy"),
    "memcpy": ("string.h", "bytes.copy"),
    "memmove": ("string.h", "bytes.copy"),
    "strcat": ("string.h", "bytes.concat"),
    "system": ("stdlib.h", "process.system"),
}
_BYTES = {"char", "i8", "u8"}
_CALLS = {
    "call",
    "indirect_call",
    "library_call",
    "library_construct",
    "object_construct",
    "object_dispatch",
    "object_new",
    "object_destroy",
    "process_spawn",
}
_CONVERSION = re.compile(
    r"%(?:[1-9][0-9]*\$)?[-+ #0']*(?:\*(?:[1-9][0-9]*\$)?|[0-9]+)?"
    r"(?:\.(?:\*(?:[1-9][0-9]*\$)?|[0-9]*))?(?:hh|ll|[hljztL])?([diouxXfFeEgGaAcspn])"
)


def _format_facts(literal):
    """Read literal bytes and conversion boundaries without evaluating formatting."""
    if literal is None:
        return "", None, True
    prefix = []
    converted = False
    additional_write = False
    position = 0
    while (at := literal.find("%", position)) >= 0:
        if not converted:
            prefix.append(literal[position:at])
        if literal.startswith("%%", at):
            if not converted:
                prefix.append("%")
            position = at + 2
        else:
            converted = True
            conversion = _CONVERSION.match(literal, at)
            if conversion is None:
                return "".join(prefix), None, True
            additional_write |= conversion[1] == "n"
            position = conversion.end()
    if not converted:
        prefix.append(literal[position:])
    prefix = "".join(prefix)
    return prefix, None if converted else prefix, additional_write


def _plain(t):
    while isinstance(t, dict) and t.get("kind") in {"const", "volatile"}:
        t = t["of"]
    return t


def _storage(t):
    t = _plain(t)
    if not isinstance(t, dict):
        return None
    kind = t.get("kind")
    item = _plain(t.get("of") if kind == "array" else t.get("to"))
    if kind == "array" and isinstance(item, str) and item in _BYTES:
        return "array"
    if kind in {"ptr", "ref"} and isinstance(item, str) and item in _BYTES | {"void"}:
        return "pointer"
    return None


@dataclass(frozen=True)
class _Value:
    ref: str | None = None
    locator: str | None = None
    text: str | None = None  # ASCII bytes before the first proven NUL
    prefix: str = ""
    terminated: str = "unknown"
    integer: int | None = None
    null: bool = False
    expression: str | None = None
    format: str | None = None


_UNKNOWN = _Value()


@dataclass
class _State:
    names: dict = field(default_factory=dict)
    versions: dict = field(default_factory=dict)
    escaped: set = field(default_factory=set)
    arrays: set = field(default_factory=set)
    exposed_names: set = field(default_factory=set)

    def copy(self):
        return _State(
            self.names.copy(),
            self.versions.copy(),
            self.escaped.copy(),
            self.arrays.copy(),
            self.exposed_names.copy(),
        )


class _Graph:
    def __init__(self, flow, image):
        self.original, self.image = flow, image
        self.nodes = {n["id"]: n for n in flow["nodes"]}
        self.edges = {json_bytes(e): e for e in flow["edges"]}
        self.ports = {
            f"{n['graph_id']}/{n['port']}": n["id"]
            for n in self.nodes.values()
            if n["kind"] == "SOURCE_OPERATION"
        }
        self.requests = {}
        self.calls = 0

    def node(self, kind, key, **fields):
        identity = (
            "artifact:" + kind.lower() + ":" + digest(json_bytes([self.image, key]))
        )
        record = dict(id=identity, kind=kind, **fields)
        if self.nodes.setdefault(identity, record) != record:
            raise ValueError("conflicting buffer flow node " + identity)
        return identity

    def edge(self, kind, source, target, **fields):
        if source is None:
            return
        row = dict(kind=kind, source=source, target=target, **fields)
        self.edges[json_bytes(row)] = row

    def bytes(self, text):
        sha = digest(text.encode("ascii"))
        ident = byte_value_id(sha)
        self.nodes.setdefault(
            ident, dict(kind="BYTE_VALUE", id=ident, sha256=sha, literal_text=text)
        )
        return ident

    def result(self):
        counts = {}
        for node in self.nodes.values():
            counts[node["kind"]] = counts.get(node["kind"], 0) + 1
        return dict(
            graph=dict(
                self.original,
                nodes=list(self.nodes.values()),
                edges=list(self.edges.values()),
            ),
            command_requests=list(self.requests.values()),
            coverage=dict(
                qualified_calls=self.calls,
                command_requests=len(self.requests),
                buffer_locators=counts.get("BUFFER_LOCATOR", 0),
                buffer_versions=counts.get("BUFFER_VERSION", 0),
                byte_expressions=counts.get("BYTE_EXPRESSION", 0),
            ),
        )


def _effect(node, externs):
    if node.get("op") != "call":
        return None
    name = node.get("callee")
    effect = node.get("artifact_effect")
    expected = _LIBC.get(name)
    ext = externs.get(name, {})
    if (
        not expected
        or not isinstance(effect, dict)
        or effect.get("semantics_version") != 1
        or ext.get("provider") != "libc"
        or ext.get("header") != expected[0]
        or effect.get("interface") != expected[1]
        or not isinstance(effect.get("contract"), dict)
        or effect["contract"].get("source_interface") != name
        or effect["contract"].get("semantics_version") != 1
        or effect["contract"].get("count_unit") != "bytes"
    ):
        return None
    params = [p.get("type") for p in ext.get("params", ())]
    variadic = name in {"sprintf", "snprintf"}
    if (
        bool(ext.get("variadic")) != variadic
        or node.get("parameter_types", params) != params
        or len(params)
        != (
            3
            if name in {"snprintf", "strncpy", "memcpy", "memmove"}
            else 1 if name == "system" else 2
        )
    ):
        return None
    roles = effect.get("operands")
    arity = len(node.get("args", ()))
    if name == "system":
        wanted = ["command"]
    elif name in {"sprintf", "snprintf"}:
        wanted = ["dst", *(["capacity"] if name == "snprintf" else []), "format"]
        wanted += ["value_" + str(i) for i in range(arity - len(wanted))]
    else:
        wanted = [
            "dst",
            "src",
            *(["count"] if name in {"strncpy", "memcpy", "memmove"} else []),
        ]
    if arity != len(wanted) or roles != {
        key: ["args", i] for i, key in enumerate(wanted)
    }:
        return None
    return effect


def _common_prefix(a, b):
    length = next(
        (i for i, (left, right) in enumerate(zip(a, b)) if left != right),
        min(len(a), len(b)),
    )
    return a[:length]


class _Function:
    def __init__(self, graph, fn, externs, module_globals):
        self.g, self.fn, self.symbol, self.externs = graph, fn, fn["name"], externs
        self.external = set()
        self.controls = []
        state = _State()
        for i, p in enumerate(fn["params"]):
            if _storage(p["type"]):
                self.declare(p, state, ("params", i), "parameter")
        usedglobals = {
            n["name"]
            for n in walk_operations(fn["body"])
            if n["op"] == "var" and n.get("scope") == "global"
        }
        for name in sorted(usedglobals & module_globals.keys()):
            row = module_globals[name]
            if _storage(row["type"]):
                self.declare(row, state, ("global", name), "global")
        self.sequence(fn["body"], state, ("body",), None, None)

    def op(self, node, path):
        site = node.get("site")
        old = self.g.ports.get(site)
        ident = old or self.g.node(
            "CANONICAL_OPERATION",
            [self.symbol, site or path],
            symbol=self.symbol,
            site=site,
            operation=node.get("op", "parameter"),
            instance_scope="per_dynamic_invocation_and_iteration",
        )
        if self.controls and self.controls[-1][0] != ident:
            parent, arm = self.controls[-1]
            self.g.edge("CONTROL_REGION", parent, ident, role=arm)
        return ident

    def region(self, node, path, port, body, state, breaks, continues):
        self.controls.append((self.op(node, path), port))
        try:
            return self.sequence(body, state, (*path, port), breaks, continues, True)
        finally:
            self.controls.pop()

    def version(
        self, state, locator, producer, form, content=_UNKNOWN, inputs=(), port=None
    ):
        ident = self.g.node(
            "BUFFER_VERSION",
            [locator, producer, form, port],
            locator=locator,
            producer=producer,
            form=form,
            expression=content.expression,
            termination=content.terminated,
            instance_scope="per_dynamic_allocation_and_control_visit",
        )
        self.g.edge("VERSION_OF_BUFFER", ident, locator)
        self.g.edge("PRODUCES_BUFFER_VERSION", producer, ident)
        self.g.edge("DERIVES_BUFFER_VERSION", content.expression, ident)
        for predecessor in inputs:
            if predecessor != ident:
                self.g.edge("REACHES_BUFFER_VERSION", predecessor, ident)
        if content.text is not None:
            self.g.edge("HAS_C_STRING_PAYLOAD", ident, self.g.bytes(content.text))
        state.versions[locator] = replace(content, ref=ident, locator=locator)
        return state.versions[locator]

    def declare(self, node, state, path, duration):
        spec = _plain(node["type"])
        key = ("global" if duration == "global" else None, node["name"])
        bound = spec.get("shape", []) if isinstance(spec, dict) else []
        capacity = (
            bound[0]
            if len(bound) == 1 and type(bound[0]) is int and bound[0] >= 0
            else None
        )
        locator = self.g.node(
            "BUFFER_LOCATOR",
            [
                ("shared" if duration == "global" else self.symbol),
                node.get("site") or path,
            ],
            symbol=None if duration == "global" else self.symbol,
            name=node["name"],
            site=node.get("site"),
            declaration_path=list(path),
            storage=duration,
            capacity_bytes=capacity,
            address_identity=(
                "caller_region_may_alias"
                if duration == "parameter"
                else "allocation_template"
            ),
        )
        state.names[key] = _Value(locator=locator)
        if _storage(spec) == "array" and duration != "parameter":
            state.arrays.add(key)
        if duration in {"global", "parameter", "static", "thread_local"}:
            self.external.add(locator)
        self.version(state, locator, self.op(node, path), "entry")
        return locator

    def unknown(self, state, producer, targets, reason):
        for loc in sorted(set(targets) & state.versions.keys()):
            self.version(
                state,
                loc,
                producer,
                "may_write:" + reason,
                inputs=[state.versions[loc].ref],
            )

    def merge(self, states, node, path):
        states = [s for s in states if s is not None]
        if not states:
            return None
        result = states[0].copy()
        if len(states) == 1:
            return result
        result.escaped = set().union(*(s.escaped for s in states))
        result.exposed_names = set().union(*(s.exposed_names for s in states))
        result.arrays = set.intersection(*(s.arrays for s in states))
        for key in set().union(*(s.names for s in states)):
            first = result.names.get(key, _UNKNOWN)
            result.names[key] = (
                first
                if all(s.names.get(key, _UNKNOWN) == first for s in states[1:])
                else _UNKNOWN
            )
        common = set.intersection(*(set(s.versions) for s in states))
        result.versions = {
            loc: v for loc, v in result.versions.items() if loc in common
        }
        for key, v in result.names.items():
            if v.locator and v.locator not in common:
                result.names[key] = _UNKNOWN
        for loc in sorted(common):
            pred = [s.versions[loc] for s in states]
            first = pred[0]
            if all(v.ref == first.ref for v in pred):
                continue
            prefix = first.prefix
            for v in pred[1:]:
                prefix = _common_prefix(prefix, v.prefix)
            text = first.text if all(v.text == first.text for v in pred) else None
            terminate = (
                first.terminated
                if all(v.terminated == first.terminated for v in pred)
                else "unknown"
            )
            self.version(
                result,
                loc,
                self.op(node, path),
                "phi",
                _Value(text=text, prefix=prefix, terminated=terminate),
                [v.ref for v in pred],
                port=path,
            )
        return result

    def known_ref(self, node, state, value, path):
        if value.locator in state.versions:
            return state.versions[value.locator]
        if value.ref:
            return value
        return replace(value, ref=self.op(node, path))

    @staticmethod
    def forget_bindings(state, keys=None):
        for key in state.names.keys() if keys is None else keys:
            if key not in state.arrays:
                state.names[key] = _UNKNOWN

    def unordered(self, nodes, state, node, path, need=False):
        """C sibling expressions without a sequencing edge may visit memory in either order."""

        def effects(arg):
            return [
                part
                for part in walk_operations(arg)
                if part["op"] in _CALLS | {"assign", "update"}
                or (
                    part["op"] == "unary"
                    and part.get("operator")
                    in {"preinc", "postinc", "predec", "postdec"}
                )
            ]

        modified = [effects(arg) for port, arg in nodes]
        conflicting = sum(bool(parts) for parts in modified) > 1
        writes = set(state.exposed_names) | {k for k in state.names if k[0] == "global"}
        for parts in modified:
            for part in parts:
                target = part.get("target") or (
                    part.get("value") if part["op"] == "unary" else None
                )
                if isinstance(target, dict) and target.get("op") == "var":
                    writes.add((target.get("scope"), target["name"]))
        result = []
        for i, (port, arg) in enumerate(nodes):
            if conflicting:
                self.unknown(
                    state,
                    self.op(node, path),
                    state.versions,
                    "unordered_operand_" + str(i),
                )
            if any(parts for j, parts in enumerate(modified) if i != j):
                self.forget_bindings(state, writes)
            result.append(self.expr(arg, state, (*path, *port), need))
        if conflicting:
            self.unknown(
                state, self.op(node, path), state.versions, "unordered_operands_out"
            )
            self.forget_bindings(state, writes)
        return result

    def expr(self, node, state, path, need=False):
        if not isinstance(node, dict) or "op" not in node:
            return _UNKNOWN
        op = node["op"]
        value = _UNKNOWN
        if op == "var":
            value = state.names.get((node.get("scope"), node["name"]), _UNKNOWN)
        elif op == "literal":
            val = node.get("value")
            value = _Value(integer=val if type(val) is int else None, null=val is None)
        elif op == "string":
            val = node.get("value")
            form = val.split("\0", 1)[0] if isinstance(val, str) else None
            if form is not None and (not form or form.isascii()):
                value = (
                    _Value(
                        ref=self.g.bytes(form),
                        text=form,
                        prefix=form,
                        terminated="proven",
                    )
                    if need
                    else _Value(text=form, prefix=form, terminated="proven")
                )
        elif op == "array_decay":
            key = "value" if "value" in node else "base"
            value = self.expr(node.get(key), state, (*path, key), need)
        elif op == "cast":
            inner = self.expr(node["value"], state, (*path, "value"), need)
            target = node.get("type")
            target = _plain(target)
            if _storage(target) == "pointer" and (
                inner.locator
                or inner.text is not None
                or inner.null
                or inner.integer == 0
            ):
                value = replace(inner, null=inner.null or inner.integer == 0)
            elif (
                isinstance(target, str)
                and re.fullmatch("[ui](8|16|32|64)", target)
                and type(inner.integer) is int
            ):
                bits = int(target[1:])
                low = 0 if target[0] == "u" else -(1 << (bits - 1))
                high = low + (1 << bits) - 1
                if low <= inner.integer <= high:
                    value = _Value(integer=inner.integer)
        elif op == "unary" and node.get("operator") == "addr":
            base = node.get("value", {})
            inner = self.expr(base, state, (*path, "value"), False)
            if base.get("op") == "var":
                key = base.get("scope"), base["name"]
                if key in state.arrays:
                    value = inner
                else:
                    state.exposed_names.add(key)
            elif (
                base.get("op") == "index"
                and len(base.get("indices", ())) == 1
                and base["indices"][0].get("op") == "literal"
                and base["indices"][0].get("value") == 0
            ):
                value = self.simple(base["base"], state)
        elif op == "select" or (
            op == "binary" and node.get("operator") in {"land", "lor"}
        ):
            first = "condition" if op == "select" else "left"
            self.expr(node[first], state, (*path, first))
            a, b = state.copy(), state.copy()
            if op == "select":
                v = self.expr(node["then"], a, (*path, "then"), need)
                w = self.expr(node["else"], b, (*path, "else"), need)
                if v == w:
                    value = v
            else:
                self.expr(node["right"], a, (*path, "right"))
            merged = self.merge([a, b], node, path)
            state.names, state.versions, state.escaped = (
                merged.names,
                merged.versions,
                merged.escaped,
            )
            state.exposed_names = merged.exposed_names
        elif op in _CALLS:
            args = node.get("args", [])
            effect = _effect(node, self.externs)
            if "receiver" in node:
                self.expr(node["receiver"], state, (*path, "receiver"))
            if "target" in node:
                self.expr(node["target"], state, (*path, "target"))
            vals = self.unordered(
                [(("args", i), arg) for i, arg in enumerate(args)],
                state,
                node,
                path,
                bool(effect),
            )
            if effect:
                vals = [
                    self.known_ref(arg, state, x, (*path, "args", i))
                    for i, (arg, x) in enumerate(zip(args, vals))
                ]
                value = self.transfer(node, effect, vals, state, path)
            else:
                if not (
                    op == "library_call"
                    and node.get("provider") == "cxx_standard"
                    and node.get("name") == "c_str"
                    and not args
                ):
                    passed = {x.locator for x in vals if x.locator}
                    for part in walk_operations(
                        [args, node.get("receiver"), node.get("target")]
                    ):
                        if part["op"] == "var" and (
                            loc := self.simple(part, state).locator
                        ):
                            passed.add(loc)
                    state.escaped.update(passed)
                    self.unknown(
                        state,
                        self.op(node, path),
                        state.escaped | self.external,
                        "opaque_call",
                    )
                    self.forget_bindings(
                        state,
                        state.exposed_names
                        | {k for k in state.names if k[0] == "global"},
                    )
        elif op == "update" and "target" in node:
            rhs = self.expr(node["value"], state, (*path, "value"))
            self.store(
                node["target"], rhs, state, node, path, node.get("operator") == "set"
            )
        elif op == "binary" and "left" in node and "right" in node:
            if node.get("operator") == "comma":
                self.expr(node["left"], state, (*path, "left"))
                value = self.expr(node["right"], state, (*path, "right"), need)
            else:
                self.unordered(
                    [((k,), node[k]) for k in ("left", "right")], state, node, path
                )
        else:
            for key, part in node.items():
                if isinstance(part, dict) and "op" in part:
                    self.expr(part, state, (*path, key))
                elif isinstance(part, list):
                    for i, arg in enumerate(part):
                        if isinstance(arg, dict) and "op" in arg:
                            self.expr(arg, state, (*path, key, i))
            if op in {"store_bytes", "release"} or (
                op == "unary"
                and node.get("operator") in {"preinc", "postinc", "predec", "postdec"}
            ):
                self.unknown(
                    state, self.op(node, path), state.versions, "unresolved_write"
                )
                self.forget_bindings(state, state.exposed_names)
                if op == "unary":
                    self.store(node["value"], _UNKNOWN, state, node, path, False)
        if need:
            return self.known_ref(node, state, value, path)
        if value.locator in state.versions:
            return state.versions[value.locator]
        return value

    @staticmethod
    def simple(node, state):
        while node.get("op") == "cast" and _storage(node.get("type")) == "pointer":
            node = node["value"]
        if node.get("op") == "var":
            return state.names.get((node.get("scope"), node["name"]), _UNKNOWN)
        return _UNKNOWN

    def store(self, target, value, state, node, path, plain=True):
        if target.get("op") == "var":
            key = (target.get("scope"), target["name"])
            state.names[key] = (
                (_Value(locator=value.locator) if value.locator else value)
                if plain
                else _UNKNOWN
            )
            if key[0] == "global" and value.locator:
                state.escaped.add(value.locator)
            return
        addr = self.simple(target.get("base", target.get("value", {})), state)
        possible = {addr.locator} if addr.locator else set(state.versions)
        if addr.locator in self.external:
            possible |= state.escaped | self.external
        self.unknown(state, self.op(node, path), possible, "address_write")
        if addr.locator is None or addr.locator in self.external:
            self.forget_bindings(state, state.exposed_names)
        if value.locator:
            state.escaped.add(value.locator)

    def transfer(self, node, effect, values, state, path):
        g = self.g
        g.calls += 1
        site = self.op(node, path)
        api = node["callee"]
        roles = {key: values[v[1]] for key, v in effect["operands"].items()}
        contract = effect["contract"]
        obligations = list(effect.get("obligations", ()))
        if effect["interface"] == "process.system":
            v = roles["command"]
            source = v.ref if v.locator else None
            exact = g.bytes(v.text) if v.text is not None else None
            query = v.null or v.integer == 0
            unresolved = (
                []
                if query
                else [
                    "command_processor_identity",
                    "external_tool_binding",
                    "runtime_environment_and_cwd",
                ]
            )
            if not query and v.text is None:
                unresolved += (
                    ["dynamic_command_bytes", "c_string_termination"]
                    if v.terminated != "proven"
                    else ["dynamic_command_bytes"]
                )
            ident = g.node(
                "COMMAND_REQUEST",
                [self.symbol, site],
                symbol=self.symbol,
                site=node.get("site"),
                operation=site,
                source_version=source,
                source_value=exact,
                expression=v.expression,
                literal_format=v.format,
                literal_prefix=v.prefix,
                command=v.text,
                request_kind=(
                    "processor_availability_query" if query else "conditional_command"
                ),
                unresolved=unresolved,
                obligations=obligations
                + (
                    ["producer_succeeded_and_source_preconditions_hold"]
                    if source
                    else []
                ),
                execution="not_executed",
            )
            g.edge("REQUESTED_BY", ident, site)
            g.edge("SUPPLIES_COMMAND_BYTES", v.ref, ident)
            g.edge("COMMAND_PAYLOAD", exact, ident)
            g.requests[ident] = g.nodes[ident]
            return _UNKNOWN
        dest = roles["dst"]
        cap = roles.get("capacity")
        count = roles.get("count")
        limit = cap.integer if cap else None
        length = count.integer if count else None
        src = roles.get("src", _UNKNOWN)
        fmt = roles.get("format", _UNKNOWN)
        prefix = ""
        text = None
        term = "unknown"
        additional = False
        if effect["interface"] == "bytes.format_into":
            prefix, text, additional = _format_facts(fmt.text)
            if cap and limit is None:
                prefix = ""
                text = None
            if cap and limit is not None and limit > 0:
                prefix = prefix[: limit - 1]
                text = text[: limit - 1] if text is not None else None
            term = (
                "on_success"
                if not cap or limit is not None and limit > 0
                else "unknown"
            )
            obligations.append("formatting_succeeds")
        elif api in {"strcpy", "strcat"}:
            first = dest if api == "strcat" else _Value(text="")
            text = (
                first.text + src.text
                if first.text is not None and src.text is not None
                else None
            )
            prefix = first.text + src.prefix if first.text is not None else first.prefix
            term = "on_defined_copy"
        elif length is not None and length > 0:
            prefix = src.prefix[:length]
            text = src.text if src.text is not None and len(src.text) < length else None
            term = "proven" if text is not None else "not_guaranteed"
        expression = g.node(
            "BYTE_EXPRESSION",
            [self.symbol, site],
            operation=site,
            interface=effect["interface"],
            source_interface=api,
            inputs={key: v.ref for key, v in roles.items()},
            integer_inputs={
                key: v.integer for key, v in roles.items() if v.integer is not None
            },
            defaults=effect.get("defaults", {}),
            literal_format=fmt.text,
            literal_prefix=prefix,
            contract=contract,
            obligations=obligations,
        )
        g.edge("EVALUATES_BYTE_EXPRESSION", site, expression)
        for key, v in roles.items():
            g.edge(
                "BUFFER_OPERAND" if key == "dst" else "BYTE_INPUT",
                v.ref,
                expression,
                role=key,
            )
        if additional:
            indirect = {
                v.locator
                for key, v in roles.items()
                if key.startswith("value_") and v.locator
            }
            if any(
                v.locator is None
                for key, v in roles.items()
                if key.startswith("value_")
            ):
                indirect.update(state.versions)
            self.unknown(state, site, indirect, "possible_percent_n")
            self.forget_bindings(state, state.exposed_names)
        if (cap and limit == 0) or (count and length == 0):
            return dest if api in {"strncpy", "memcpy", "memmove"} else _UNKNOWN
        loc = dest.locator
        self.unknown(
            state,
            site,
            (
                (state.escaped | self.external) - {loc}
                if loc in self.external | state.escaped
                else (set(state.versions) if not loc else ())
            ),
            "destination_alias",
        )
        if loc is None:
            self.forget_bindings(state, state.exposed_names)
            loc = g.node(
                "BUFFER_LOCATOR",
                [self.symbol, site, "unresolved_destination"],
                symbol=self.symbol,
                site=node.get("site"),
                storage="unresolved_region",
                capacity_bytes=None,
                address_identity="runtime_pointer_may_alias",
            )
            self.external.add(loc)
        capacity = g.nodes[loc].get("capacity_bytes")
        impossible = type(capacity) is int and (
            (length is not None and length > capacity)
            or (text is not None and len(text) + 1 > capacity)
            or (
                effect["interface"] == "bytes.format_into"
                and len(prefix) + 1 > capacity
            )
        )
        if impossible or additional:
            text = None
            prefix = ""
            term = "unknown"
        content = _Value(
            text=text,
            prefix=prefix,
            terminated="proven" if text is not None else term,
            format=(
                fmt.text
                if effect["interface"] == "bytes.format_into"
                else (src.format if api == "strcpy" else None)
            ),
            expression=expression,
        )
        self.version(
            state,
            loc,
            site,
            "write",
            content,
            [state.versions[loc].ref] if loc in state.versions else [],
        )
        return (
            state.versions.get(loc, _UNKNOWN)
            if effect["interface"] != "bytes.format_into"
            else _UNKNOWN
        )

    def sequence(self, body, state, path, breaks, continues, scoped=False):
        shadows = {}
        local_allocations = set()
        former_arrays = state.arrays.copy()
        start_breaks = len(breaks) if breaks is not None else 0
        start_continues = len(continues) if continues is not None else 0
        for i, stmt in enumerate(body):
            if state is None:
                break
            here = (*path, i)
            op = stmt["op"]
            if op == "decl":
                key = stmt.get("scope"), stmt["name"]
                if scoped:
                    shadows.setdefault(key, state.names.get(key, _UNKNOWN))
                state.arrays.discard(key)
                kind = _storage(stmt["type"])
                initial = stmt.get("init")
                storage = stmt.get("storage", "automatic")
                persistent = storage in {"static", "thread_local"}
                if persistent and initial:
                    initialization = state.copy()
                    self.controls.append((self.op(stmt, here), "first_initialization"))
                    try:
                        init = self.expr(
                            initial, initialization, (*here, "init"), kind == "array"
                        )
                    finally:
                        self.controls.pop()
                    state = self.merge(
                        [initialization, state], stmt, (*here, "persistent_entry")
                    )
                else:
                    init = self.expr(initial, state, (*here, "init"), kind == "array")
                if kind == "array":
                    loc = self.declare(stmt, state, here, storage)
                    if scoped and stmt.get("storage", "auto") in {
                        "auto",
                        "automatic",
                        "register",
                    }:
                        local_allocations.add(loc)
                    if init.text is not None:
                        previous = state.versions[loc]
                        size = self.g.nodes[loc]["capacity_bytes"]
                        valid = size is not None and len(init.text) < size
                        val = _Value(
                            text=init.text if valid else None,
                            prefix=init.prefix[:size] if size is not None else "",
                            terminated="proven" if valid else "unknown",
                        )
                        self.version(
                            state,
                            loc,
                            self.op(stmt, here),
                            "initialize",
                            val,
                            [state.versions[loc].ref],
                        )
                        self.g.edge(
                            "INITIALIZES_BUFFER", init.ref, state.versions[loc].ref
                        )
                        if persistent:
                            self.version(
                                state,
                                loc,
                                self.op(stmt, here),
                                "persistent_entry",
                                inputs=[previous.ref, state.versions[loc].ref],
                            )
                else:
                    state.names[key] = (
                        _UNKNOWN
                        if persistent
                        else (_Value(locator=init.locator) if init.locator else init)
                    )
            elif op == "assign":
                value = self.expr(stmt["value"], state, (*here, "value"))
                self.expr(stmt["target"], state, (*here, "target"))
                self.store(stmt["target"], value, state, stmt, here)
            elif op == "expr":
                self.expr(stmt["value"], state, (*here, "value"))
            elif op == "return":
                self.expr(stmt.get("value"), state, (*here, "value"))
                state = None
            elif op in {"break", "continue"}:
                target = breaks if op == "break" else continues
                if target is not None:
                    target.append(state)
                state = None
            elif op == "block":
                state = self.sequence(
                    stmt["body"], state, (*here, "body"), breaks, continues, True
                )
            elif op == "if":
                self.expr(stmt["condition"], state, (*here, "condition"))
                paths = [
                    self.region(
                        stmt, here, k, stmt.get(k, ()), state.copy(), breaks, continues
                    )
                    for k in ("then", "else")
                ]
                state = self.merge(paths, stmt, (*here, "exit"))
            elif op in {"for", "while", "do_while", "each", "map"}:
                state = self.loop(stmt, state, here)
            elif op == "switch":
                self.expr(stmt["condition"], state, (*here, "condition"))
                entry = state.copy()
                exits = []
                active = None
                default = False
                for j, part in enumerate(stmt["body"]):
                    if part["op"] in {"case", "default"}:
                        default |= part["op"] == "default"
                        active = self.merge(
                            [active, entry.copy()], part, (*here, "label", j)
                        )
                    elif active is not None:
                        active = self.sequence(
                            [part], active, (*here, "switch", j), exits, continues
                        )
                state = self.merge(
                    [*exits, active, *([] if default else [entry])],
                    stmt,
                    (*here, "exit"),
                )
            elif op in {"try", "resource_scope"}:
                # Handlers and cleanup can begin after a partial source body. They
                # may also rewrite state during a nonlocal break/continue transfer.
                scopes = [(owner[key], key) for owner, key, _ in control_regions(stmt)]
                exits = []
                b = []
                c = []
                for j, (stmts, key) in enumerate(scopes):
                    arm = state.copy()
                    self.unknown(
                        arm,
                        self.op(stmt, here),
                        arm.versions,
                        "exceptional_entry_" + str(j),
                    )
                    self.forget_bindings(arm)
                    arm = self.sequence(stmts, arm, (*here, key, j), b, c, True)
                    exits.append(arm)
                for route, outer in ((b, breaks), (c, continues)):
                    for s in route:
                        self.unknown(
                            s, self.op(stmt, here), s.versions, "cleanup_transfer"
                        )
                        self.forget_bindings(s)
                        if outer is not None:
                            outer.append(s)
                state = self.merge([state, *exits], stmt, (*here, "exit"))
                self.unknown(
                    state, self.op(stmt, here), state.versions, "unmodeled_cleanup"
                )
                self.forget_bindings(state)
            elif op in {"goto", "label"}:
                self.unknown(
                    state, self.op(stmt, here), state.versions, "unstructured_control"
                )
                self.forget_bindings(state)
            else:
                self.expr(stmt, state, here)
        if scoped:
            for s in [
                state,
                *(breaks[start_breaks:] if breaks is not None else []),
                *(continues[start_continues:] if continues is not None else []),
            ]:
                if s is not None:
                    s.names.update(shadows)
                    s.arrays = (s.arrays - shadows.keys()) | (
                        former_arrays & shadows.keys()
                    )
                    for loc in local_allocations:
                        s.versions.pop(loc, None)
                    for key, val in s.names.items():
                        if val.locator in local_allocations:
                            s.names[key] = _UNKNOWN
        return state

    def loop(self, node, state, path):
        entry = state.copy()
        loop_breaks = []
        next_iterations = []
        if node["op"] == "for":
            state = self.sequence(
                node.get("init", ()), state, (*path, "init"), None, None
            )
            if state is None:
                return None
        head = state.copy()
        marker = self.op(node, path)
        # An address can escape on a previous visit to this header.
        head.escaped.update(head.versions)
        names = {
            v["target"]["name"]
            for v in walk_operations([node.get("body", []), node.get("step", {})])
            if v["op"] in {"assign", "update"}
            and v.get("target", {}).get("op") == "var"
        }
        for key in head.names:
            if key[1] in names:
                head.names[key] = _UNKNOWN
        for loc, last in state.versions.items():
            self.version(head, loc, marker, "loop_header", inputs=[last.ref])
        headrefs = {loc: v.ref for loc, v in head.versions.items()}
        if node["op"] != "do_while":
            for key in ("condition", "range", "extent"):
                self.expr(node.get(key), head, (*path, key))
        body = self.region(
            node,
            path,
            "body",
            node.get("body", ()),
            head.copy(),
            loop_breaks,
            next_iterations,
        )
        back = self.merge([body, *next_iterations], node, (*path, "latch"))
        if back is not None:
            self.expr(node.get("step"), back, (*path, "step"))
            if node["op"] == "do_while":
                self.expr(node.get("condition"), back, (*path, "condition"))
            for loc, key in headrefs.items():
                if loc in back.versions:
                    self.g.edge("LOOP_BACKEDGE", back.versions[loc].ref, key)
        exits = (
            [*loop_breaks, back]
            if node["op"] == "do_while"
            else [*loop_breaks, head, back]
        )
        tail = self.merge(exits, node, (*path, "exit"))
        if tail is not None and node["op"] == "for":
            for init in node.get("init", ()):
                if init["op"] == "decl":
                    key = init.get("scope"), init["name"]
                    tail.names[key] = entry.names.get(key, _UNKNOWN)
        return tail


def analyze_buffers(module, flow):
    """Add buffer allocation/version templates; no source code or command is executed."""
    graph = _Graph(flow, module.get("current_image"))
    functions = module.get("functions", [])
    definitions = {fn["name"] for fn in functions}
    externs = {
        row["name"]: row
        for row in module.get("externs", ())
        if row["name"] not in definitions
    }
    globals_ = {
        row["name"]: row for row in module.get("globals", ()) if _storage(row["type"])
    }
    for fn in functions:
        if fn.get("execution_kind") == "python_protocol":
            continue
        if any(_effect(op, externs) for op in walk_operations(fn["body"])):
            _Function(graph, fn, externs, globals_)
    return graph.result()
