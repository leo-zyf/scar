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
from .file_state import calls_libc_exit

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
    "process_request",
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


def _stream_kind(t, names):
    """Track an object (including its reference) or a single pointer to it."""
    t = _plain(t)
    kind = "object"
    if isinstance(t, dict) and t.get("kind") in {"ptr", "ref"}:
        kind = "pointer" if t["kind"] == "ptr" else "object"
        t = _plain(t.get("to"))
    if (
        isinstance(t, dict)
        and t.get("kind") == "library"
        and t.get("provider") == "cxx_standard"
        and t.get("name") in names
        and t.get("args", [])[:1] == ["char"]
    ):
        return kind
    return None


def _stream_effect(node):
    effect = node.get("artifact_effect")
    if not isinstance(effect, dict) or effect.get("semantics_version") != 1:
        return None
    contract = effect.get("contract", {})
    if contract.get("source_profile") != "cxx_standard_ifstream_char_v1":
        return None
    op, interface = node["op"], effect.get("interface")
    argc = len(node.get("args", []))
    roles, defaults = {}, {}
    if op == "library_construct":
        if (
            interface != "stream.open"
            or _stream_kind(node.get("type"), {"basic_ifstream"}) != "object"
            or node.get("provider", "cxx_standard") != "cxx_standard"
        ):
            return None
        name, handle, result = (
            "std::basic_ifstream::basic_ifstream",
            "result",
            "stream_object",
        )
    elif (
        op == "library_call"
        and node.get("provider") == "cxx_standard"
        and isinstance(node.get("receiver"), dict)
    ):
        roles["stream"] = ["receiver"]
        handle = "receiver"
        method = node.get("name")
        if (method, interface) == ("getline", "stream.getline_into"):
            name, result = "std::basic_istream::getline", "receiver_reference"
        elif (method, interface) in {
            ("open", "stream.open"),
            ("close", "stream.close"),
        }:
            name, result = "std::basic_ifstream::" + method, "void"
        else:
            return None
    else:
        return None
    if interface == "stream.open" and argc in {1, 2}:
        roles["path"] = ["args", 0]
        if argc == 2:
            roles["mode"] = ["args", 1]
        else:
            defaults["mode"] = dict(kind="ios_openmode", flag="in")
    elif interface == "stream.getline_into" and argc in {2, 3}:
        roles.update(dst=["args", 0], count=["args", 1])
        if argc == 3:
            roles["delimiter"] = ["args", 2]
        else:
            defaults["delimiter"] = dict(kind="widen_newline")
    elif interface != "stream.close" or argc:
        return None
    if (
        effect.get("operands") != roles
        or effect.get("defaults", {}) != defaults
        or contract.get("source_interface") != name
        or contract.get("handle") != handle
        or contract.get("result") != result
    ):
        return None
    return effect


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
    stream: str | None = None
    stream_pointer: bool = False


_UNKNOWN = _Value()


@dataclass(frozen=True)
class _Stream:
    """Possible file associations and the preceding stream state operations."""

    locators: frozenset = frozenset()
    states: frozenset = frozenset()
    unresolved: bool = True

    @classmethod
    def join(cls, values):
        return cls(
            frozenset.union(*(item.locators for item in values)),
            frozenset.union(*(item.states for item in values)),
            any(item.unresolved for item in values),
        )


@dataclass
class _State:
    names: dict = field(default_factory=dict)
    versions: dict = field(default_factory=dict)
    escaped: set = field(default_factory=set)
    arrays: set = field(default_factory=set)
    exposed_names: set = field(default_factory=set)
    streams: dict = field(default_factory=dict)
    escaped_streams: set = field(default_factory=set)
    map_origins: dict = field(default_factory=dict)

    def copy(self):
        return _State(
            self.names.copy(),
            self.versions.copy(),
            self.escaped.copy(),
            self.arrays.copy(),
            self.exposed_names.copy(),
            self.streams.copy(),
            self.escaped_streams.copy(),
            self.map_origins.copy(),
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
        return _stream_effect(node)
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


def _cstring_copy(node, effect, roles, declaration, capacity):
    """Identify an unadorned narrow %s conversion at the actual call."""
    api = node["callee"]
    if api not in {"sprintf", "snprintf"} or set(roles) != (
        {"dst", "format", "value_0"} | ({"capacity"} if api == "snprintf" else set())
    ):
        return None
    parameters = [p["type"] for p in declaration["params"]]
    if (
        declaration.get("result") != "i32"
        or parameters[0] != {"kind": "ptr", "to": "char"}
        or parameters[-1] != {"kind": "ptr", "to": {"kind": "const", "of": "char"}}
        or api == "snprintf"
        and parameters[1] not in ("u32", "u64")
    ):
        return None
    literal = node["args"][effect["operands"]["format"][1]]
    if literal.get("op") == "cast" and literal.get("conversion") == "array_decay":
        literal = literal["value"]
    if literal.get("op") == "array_decay":
        literal = literal["value"] if "value" in literal else literal["base"]
    source = roles["value_0"]
    if (
        literal.get("op") != "string"
        or literal.get("value") != "%s"
        or roles["format"].text != "%s"
        or not source.ref
        or source.null
        or source.integer is not None
        or effect["contract"].get("format_family") != "c_printf"
        or effect["contract"].get("overlap") != "forbidden_for_formatted_copy_sources"
        or roles["dst"].locator is not None
        and roles["dst"].locator == source.locator
    ):
        return None
    limit = roles.get("capacity")
    if any(
        bound is not None
        and (bound <= 0 or source.text is not None and len(source.text) >= bound)
        for bound in (capacity, limit.integer if limit else None)
    ):
        return None
    return dict(
        source_operand=effect["operands"]["value_0"],
        destination_operand=effect["operands"]["dst"],
        source_value=source.ref,
        extent="through_first_nul_inclusive",
        requires=[
            "selected_libc_call_returns_successfully",
            "variadic_source_is_a_live_readable_narrow_c_string_at_conversion",
            "destination_is_writable_for_source_length_plus_nul",
            "source_and_destination_do_not_overlap",
            *(["capacity_exceeds_source_length"] if limit else []),
        ],
    )


def _common_prefix(a, b):
    length = next(
        (i for i, (left, right) in enumerate(zip(a, b)) if left != right),
        min(len(a), len(b)),
    )
    return a[:length]


def _system_boundary(node, declaration, operation, request):
    """A source-selected C runtime call is an ordered, externally effectful leaf."""
    params = [p["type"] for p in declaration["params"]]
    contract = node["artifact_effect"]["contract"]
    if (
        not node.get("site")
        or params != [{"kind": "ptr", "to": {"kind": "const", "of": "char"}}]
        or node.get("parameter_types") != params
        or declaration.get("result") != "i32"
        or contract.get("null_command") != "query_command_processor_availability"
        or contract.get("command_read") != "through_first_nul_when_nonnull"
        or contract.get("process_effect") != "external_command_processor_when_nonnull"
        or contract.get("return_value") != "implementation_status_not_direct_exit_code"
    ):
        return
    node["runtime_boundary"] = dict(
        semantics_version=1,
        profile="posix_libc_system_v1",
        site=node["site"],
        source_abi=dict(
            symbol=declaration["name"],
            provider=declaration["provider"],
            header=declaration["header"],
            parameters=params,
            result=declaration["result"],
            variadic=False,
        ),
        source_operation=operation,
        command_request=request,
        command_operand=["args", 0],
        execution="synchronous",
        result="c_system_status",
        descendant_closure="open",
    )
    for key, additions in (
        ("effects", {"CTRL", "IO", "OPAQUE", "ORDER", "STATE"}),
        (
            "reads",
            {
                "command.c_string",
                "process.cwd",
                "process.environment",
                "process.signals",
                "external.state",
            },
        ),
        (
            "writes",
            {"process.children", "process.signals", "thread.errno", "external.state"},
        ),
    ):
        node[key] = sorted(set(node.get(key, ())) | additions)
    node["external_effect"] = "UNKNOWN"


def _standard(name, *args):
    return dict(kind="library", provider="cxx_standard", name=name, args=list(args))


_STRING = _standard(
    "basic_string",
    "char",
    _standard("char_traits", "char"),
    _standard("allocator", "char"),
)
_MAP_PAIR = _standard("pair", {"kind": "const", "of": _STRING}, _STRING)
_STRING_MAP = _standard(
    "map",
    _STRING,
    _STRING,
    _standard("less", _STRING),
    _standard("allocator", _MAP_PAIR),
)
_MAP_ITER = _standard("map_iterator", _STRING, _STRING)


def _library_call(node, name, modes):
    return (
        node.get("op") == "library_call"
        and node.get("provider") == "cxx_standard"
        and node.get("name") == name
        and node.get("parameter_modes") == modes
        and len(node.get("args", ())) == len(modes)
    )


def _library_construct(node, type_, modes):
    return (
        node.get("op") == "library_construct"
        and node.get("provider") == "cxx_standard"
        and node.get("type") == type_
        and node.get("parameter_modes") == modes
        and len(node.get("args", ())) == len(modes)
    )


def _unqualify(node):
    if (
        node.get("op") == "library_value"
        and node.get("provider") == "cxx_standard"
        and node.get("mode") == "qualify"
    ):
        return node["value"]
    return node


def _string_key(node):
    if _library_construct(node, _STRING, ["pointer"]):
        node = node["args"][0]
        if node.get("op") == "cast" and node.get("conversion") == "array_decay":
            node = node["value"]
    if (
        node.get("op") == "string"
        and isinstance(node.get("value"), str)
        and node["value"].isascii()
        and "\0" not in node["value"]
    ):
        return node["value"]
    return None


class _StringMap:
    """Conditional definitions for standard string copies and single map insertions."""

    def __init__(self, frame, declarations):
        self.frame = frame
        self.types = declarations

    def slot(self, node, type_):
        node = _unqualify(node)
        key = node.get("scope"), node.get("name")
        return key if node.get("op") == "var" and self.types.get(key) == type_ else None

    def edges(self, sources, target, kind, *conditions, **facts):
        for source in sorted(sources):
            self.frame.g.edge(
                kind,
                source,
                target,
                conditional=True,
                requires=list(conditions),
                **facts,
            )

    def statement(self, stmt, facts, path):
        op = stmt["op"]
        field = {"decl": "init", "expr": "value", "if": "condition"}.get(op)
        node = stmt.get(field)
        if not node:
            if op not in {"block", "decl", "expr", "if", "return"}:
                facts.clear()
            return
        port = (*path, field)
        args = node.get("args", [])
        recv = node.get("receiver", {})
        if (
            op == "decl"
            and stmt["type"] == _STRING
            and _library_construct(node, _STRING, ["pointer"])
        ):
            # Only ordinary address/index/load operands can retain earlier map candidates.
            if not any(
                p["op"] in _CALLS or p["op"] == "update" for p in walk_operations(args)
            ):
                target = self.frame.op(node, port)
                arg = self.frame.op(args[0], (*port, "args", 0))
                self.edges(
                    [arg],
                    target,
                    "CONSTRUCTS_STRING_FROM_C_STRING",
                    "normal_construction_from_live_terminated_character_array",
                    operand_path=[*port, "args", 0],
                )
                facts[("string", (stmt.get("scope"), stmt["name"]))] = frozenset(
                    {target}
                )
                return
        map_slot = self.slot(recv, _STRING_MAP)
        if _library_call(node, "insert", ["reference"]) and map_slot:
            pair = args[0]
            if _library_construct(pair, _MAP_PAIR, ["reference", "reference"]):
                key = _string_key(pair["args"][0])
                source = self.slot(pair["args"][1], _STRING)
                if key is not None and source:
                    site = self.frame.op(node, port)
                    operand = self.frame.op(
                        pair["args"][1], (*port, "args", 0, "args", 1)
                    )
                    self.edges(
                        [operand],
                        site,
                        "PROPOSES_MAPPED_VALUE",
                        "value_type_constructed_from_current_mapped_operand",
                        key=key,
                    )
                    self.edges(
                        facts.get(("string", source), ()),
                        operand,
                        "MAY_SUPPLY_STRING_OPERAND",
                        "same_string_instance_and_contents_reach_this_read",
                    )
                    entry = ("insert", map_slot, key)
                    facts[entry] = facts.get(entry, frozenset()) | {site}
                    return
        if _library_call(node, "operator=", ["reference"]) and self.assignment(
            node, facts, port
        ):
            return
        if op == "if" and _library_call(node, "operator!=", ["reference", "reference"]):
            end = _unqualify(args[1])
            if (
                self.slot(args[0], _MAP_ITER)
                and _library_call(end, "end", [])
                and self.slot(end.get("receiver", {}), _STRING_MAP)
            ):
                return
        if any(
            n["op"] in _CALLS
            or n["op"] in {"update", "store_bytes", "release", "opaque", "unsupported"}
            or n["op"] == "unary"
            and n.get("operator") in {"preinc", "postinc", "predec", "postdec"}
            for n in walk_operations(node)
        ):
            facts.clear()

    def assignment(self, node, facts, port):
        rhs = _unqualify(node["args"][0])
        recv = node["receiver"]
        iterator = self.slot(recv, _MAP_ITER)
        if (
            iterator
            and _library_call(rhs, "find", ["reference"])
            and (map_slot := self.slot(rhs.get("receiver", {}), _STRING_MAP))
        ):
            key = _string_key(rhs["args"][0])
            if key is not None:
                lookup = self.frame.op(rhs, (*port, "args", 0))
                self.edges(
                    facts.get(("insert", map_slot, key), ()),
                    lookup,
                    "MAY_SUPPLY_MAP_LOOKUP",
                    "same_dynamic_map_and_comparator_equivalent_key",
                    "insert_returned_second_true",
                    "same_entry_survives_without_intervening_replacement_or_mapped_value_write",
                    key=key,
                    other_sources="preexisting_entry_or_other_write_or_no_match",
                )
                target = self.frame.op(node, port)
                self.edges(
                    [lookup],
                    target,
                    "ASSIGNS_MAP_ITERATOR",
                    "find_and_iterator_assignment_return_normally",
                )
                facts[("iterator", iterator)] = frozenset({target})
                return True
        string = self.slot(recv, _STRING)
        if (
            string
            and rhs.get("op") == "library_member"
            and rhs.get("provider") == "cxx_standard"
            and rhs.get("name") == "second"
        ):
            deref = rhs.get("receiver", {})
            pointer = deref.get("value", {})
            if (
                deref.get("operator") == "deref"
                and deref.get("op") == "unary"
                and _library_call(pointer, "operator->", [])
                and (iterator := self.slot(pointer.get("receiver", {}), _MAP_ITER))
            ):
                target = self.frame.op(node, port)
                self.edges(
                    facts.get(("iterator", iterator), ()),
                    target,
                    "COPIES_MAPPED_STRING",
                    "same_live_iterator_reaches_dereference",
                    "iterator_denotes_an_element_at_the_read",
                    "mapped_value_unchanged_between_find_and_copy",
                    "string_copy_assignment_returns_normally",
                )
                facts[("string", string)] = frozenset({target})
                return True
        return False

    def c_string(self, node, facts, source):
        if _library_call(node, "c_str", []) and (
            slot := self.slot(node.get("receiver", {}), _STRING)
        ):
            self.edges(
                facts.get(("string", slot), ()),
                source,
                "EXPOSES_STRING_C_STR",
                "same_live_string_contents_reach_c_str",
                "returned_pointer_remains_valid_until_conversion",
            )


class _Function:
    def __init__(self, graph, fn, externs, module_globals):
        self.g, self.fn, self.symbol, self.externs = graph, fn, fn["name"], externs
        self.external = set()
        self.controls = []
        state = _State()
        for i, p in enumerate(fn["params"]):
            if _storage(p["type"]):
                self.declare(p, state, ("params", i), "parameter")
            elif kind := _stream_kind(p["type"], {"basic_ifstream", "basic_istream"}):
                identity = self.op(p, ("params", i))
                state.names[(None, p["name"])] = _Value(
                    stream=identity, stream_pointer=kind == "pointer"
                )
                state.escaped_streams.add(identity)
        types = {(None, p["name"]): p["type"] for p in fn["params"]}
        usedglobals = set()
        self.structured = True
        for operation in walk_operations(fn["body"]):
            if operation["op"] == "decl":
                key = operation.get("scope"), operation["name"]
                types[key] = None if key in types else operation["type"]
            if operation["op"] in {"goto", "label"}:
                self.structured = False
            elif operation["op"] == "var" and operation.get("scope") == "global":
                usedglobals.add(operation["name"])
        for name in sorted(usedglobals & module_globals.keys()):
            row = module_globals[name]
            types["global", name] = row["type"]
            if _storage(row["type"]):
                self.declare(row, state, ("global", name), "global")
        self.types = types
        self.map_strings = (
            _StringMap(self, types)
            if self.structured and _STRING_MAP in types.values()
            else None
        )
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
        if self.map_strings and len(states) > 1:
            for other in states[1:]:
                for key, refs in other.map_origins.items():
                    result.map_origins[key] = (
                        result.map_origins.get(key, frozenset()) | refs
                    )
        if len(states) == 1:
            return result
        result.escaped = set().union(*(s.escaped for s in states))
        result.escaped_streams = set().union(*(s.escaped_streams for s in states))
        result.streams = {
            key: _Stream.join([s.streams.get(key, _Stream()) for s in states])
            for key in set().union(*(s.streams for s in states))
        }
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
        elif op == "library_value" and node.get("mode") == "qualify":
            inner = self.expr(node["value"], state, (*path, "value"), need)
            if inner.stream and node.get("provider") == "cxx_standard":
                value = inner
        elif op == "cast":
            inner = self.expr(node["value"], state, (*path, "value"), need)
            target = node.get("type")
            target = _plain(target)
            if inner.stream and _stream_kind(
                target, {"basic_ifstream", "basic_istream", "basic_ios"}
            ) == ("pointer" if inner.stream_pointer else "object"):
                value = inner
            elif _storage(target) == "pointer" and (
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
                    if inner.stream_pointer:
                        state.names[key] = _UNKNOWN
                    elif inner.stream:
                        value = _Value(stream=inner.stream, stream_pointer=True)
            elif (
                base.get("op") == "index"
                and len(base.get("indices", ())) == 1
                and base["indices"][0].get("op") == "literal"
                and base["indices"][0].get("value") == 0
            ):
                value = self.simple(base["base"], state)
        elif op == "unary" and node.get("operator") == "deref":
            inner = self.expr(node["value"], state, (*path, "value"))
            if inner.stream and inner.stream_pointer:
                value = _Value(stream=inner.stream)
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
            state.streams, state.escaped_streams = (
                merged.streams,
                merged.escaped_streams,
            )
        elif op in _CALLS:
            args = node.get("args", [])
            effect = _effect(node, self.externs)
            receiver = _UNKNOWN
            if "receiver" in node:
                receiver = self.expr(
                    node["receiver"], state, (*path, "receiver"), bool(effect)
                )
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
                if "source_profile" in effect["contract"] and op != "call":
                    value = self.stream_transfer(
                        node, effect, receiver, vals, state, path
                    )
                else:
                    value = self.transfer(node, effect, vals, state, path)
            else:
                if not (
                    op == "library_call"
                    and node.get("provider") == "cxx_standard"
                    and (
                        node.get("name") == "c_str"
                        or receiver.stream
                        and node.get("name")
                        in {"good", "eof", "bad", "fail", "is_open"}
                    )
                    and not args
                ):
                    passed_streams = {v.stream for v in [receiver, *vals] if v.stream}
                    for part in walk_operations(
                        [args, node.get("receiver"), node.get("target")]
                    ):
                        if part["op"] == "var" and (
                            stream := self.simple(part, state).stream
                        ):
                            passed_streams.add(stream)
                    state.escaped_streams.update(passed_streams)
                    for stream in state.escaped_streams:
                        state.streams[stream] = _Stream(
                            states=frozenset({self.op(node, path)})
                        )
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
                if plain and not (value.stream_pointer and key in state.exposed_names)
                else _UNKNOWN
            )
            if key[0] == "global" and value.locator:
                state.escaped.add(value.locator)
            if key[0] == "global" and value.stream:
                state.escaped_streams.add(value.stream)
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
        if value.stream:
            state.escaped_streams.add(value.stream)

    def destination(self, dest, state, node, site):
        loc = dest.locator
        aliases = self.external | state.escaped
        self.unknown(
            state,
            site,
            aliases - {loc} if loc in aliases else (() if loc else state.versions),
            "destination_alias",
        )
        if loc is None:
            self.forget_bindings(state, state.exposed_names)
            loc = self.g.node(
                "BUFFER_LOCATOR",
                [self.symbol, site, "unresolved_destination"],
                symbol=self.symbol,
                site=node.get("site"),
                storage="unresolved_region",
                capacity_bytes=None,
                address_identity="runtime_pointer_may_alias",
            )
            self.external.add(loc)
        return loc

    def stream_transfer(self, node, effect, receiver, values, state, path):
        g, site = self.g, self.op(node, path)
        g.calls += 1
        interface, contract = effect["interface"], effect["contract"]
        ports, roles = {}, {}
        for role, port in effect["operands"].items():
            operand = node[port[0]]
            if port == ["receiver"]:
                roles[role] = receiver
            else:
                operand = operand[port[1]]
                roles[role] = values[port[1]]
            ports[role] = self.op(operand, (*path, *port))
        constructed = contract["handle"] == "result"
        handle = (
            site
            if constructed
            else (None if receiver.stream_pointer else receiver.stream)
        )
        before = (
            _Stream(unresolved=False)
            if constructed
            else state.streams.get(handle, _Stream())
        )
        g.edge("USES_STREAM_OBJECT", handle if not constructed else None, site)
        for previous in sorted(before.states):
            g.edge("REACHES_STREAM_STATE", previous, site, conditional=True)
        after = frozenset({site})
        if handle is None:
            for possible, incoming in state.streams.items():
                for previous in sorted(incoming.states):
                    g.edge(
                        "REACHES_STREAM_STATE",
                        previous,
                        site,
                        role="possible_receiver",
                        stream_object=possible,
                        conditional=True,
                    )
                states = incoming.states | after
                if interface in {"stream.open", "stream.close"}:
                    state.streams[possible] = _Stream(states=states)
                else:
                    state.streams[possible] = replace(incoming, states=states)
        if interface == "stream.open":
            locator = g.node(
                "FILE_LOCATOR",
                [self.symbol, site, "open_path"],
                identity_scope="path_operand_at_open",
                symbol=self.symbol,
                operation=site,
                operand=effect["operands"]["path"],
                operand_port=ports["path"],
                operand_value=roles["path"].ref,
                path=roles["path"].text,
                cwd="process_cwd_at_open",
                inode_aliases="unresolved",
                stream_object=handle,
                defaults=effect.get("defaults", {}),
                contract=contract,
                obligations=effect.get("obligations", []),
            )
            for role, v in roles.items():
                g.edge("OPEN_OPERAND", ports[role], locator, role=role)
                g.edge("OPEN_OPERAND_VALUE", v.ref, locator, role=role)
            g.edge(
                "MAY_ASSOCIATE_FILE",
                site,
                locator,
                conditional=True,
                requires=["filebuf_open_succeeds"],
            )
            if handle:
                # Reopening an already open filebuf can fail and leave the old file associated.
                state.streams[handle] = _Stream(
                    before.locators | {locator}, after, before.unresolved
                )
            return _Value(stream=handle) if constructed else _UNKNOWN
        if interface == "stream.close":
            for locator in sorted(before.locators):
                g.edge(
                    "MAY_CLOSE_FILE",
                    site,
                    locator,
                    conditional=True,
                    requires=["filebuf_close_succeeds"],
                )
            if handle:
                state.streams[handle] = _Stream(states=after)
            return _UNKNOWN

        locators = set(before.locators)
        if before.unresolved:
            locators.add(
                g.node(
                    "FILE_LOCATOR",
                    [self.symbol, site, "stream_association"],
                    identity_scope="unresolved_stream_association",
                    symbol=self.symbol,
                    operation=site,
                    operand=effect["operands"]["stream"],
                    operand_port=ports["stream"],
                    stream_object=handle,
                    path=None,
                    cwd=None,
                    inode_aliases="unresolved",
                )
            )
        expression = g.node(
            "BYTE_EXPRESSION",
            [self.symbol, site, interface],
            symbol=self.symbol,
            operation=site,
            interface=interface,
            source_interface=contract["source_interface"],
            stream_object=handle,
            operand_ports=ports,
            inputs={role: v.ref for role, v in roles.items()},
            integer_inputs={
                role: v.integer for role, v in roles.items() if v.integer is not None
            },
            defaults=effect.get("defaults", {}),
            contract=contract,
            obligations=effect.get("obligations", []),
            result_extent="depends_on_count_delimiter_eof_and_stream_state",
        )
        g.edge("EVALUATES_BYTE_EXPRESSION", site, expression)
        for role, v in roles.items():
            g.edge("OPERAND_PORT", ports[role], expression, role=role)
            g.edge(
                "BUFFER_OPERAND" if role == "dst" else "BYTE_INPUT",
                v.ref,
                expression,
                role=role,
            )
        for locator in sorted(locators):
            requirement = g.node(
                "FILE_VERSION_REQUIREMENT",
                [self.symbol, site, locator, "getline"],
                locator=locator,
                consumer=site,
                value=None,
                role="runtime_stream_read",
                operand=effect["operands"]["stream"],
                stream_object=handle,
                instance_scope="per_dynamic_extraction",
                historical_instance="not_established",
                version_status="backing_bytes_required",
                acquisition="filebuf_fill_or_prior_read_ahead",
                requires=[
                    "file_still_associated",
                    "sentry_allows_input",
                    "file_bytes_consulted",
                ],
            )
            g.edge("REQUIRES_RUNTIME_FILE_VERSION", requirement, site, conditional=True)
            g.edge("SUPPLIES_STREAM_BYTES", requirement, expression, conditional=True)
        if handle:
            state.streams[handle] = replace(before, states=after)
        count = roles["count"].integer
        if count is None or count > 0:
            loc = self.destination(roles["dst"], state, node, site)
            self.version(
                state,
                loc,
                site,
                "conditional_stream_write",
                _Value(expression=expression),
                [state.versions[loc].ref] if loc in state.versions else [],
            )
        return _Value(stream=handle)

    def line_consumer(self, body, i, state, path):
        """Connect an extracted buffer to an immediately copied by-value call operand."""
        read = body[i]["value"]
        if not self.structured or i + 2 >= len(body):
            return
        assign = body[i + 1].get("value")
        if (
            body[i + 1]["op"] != "expr"
            or not isinstance(assign, dict)
            or not _library_call(assign, "operator=", ["pointer"])
        ):
            return
        variable = assign.get("receiver", {})
        slot = variable.get("scope"), variable.get("name")
        if variable.get("op") != "var" or self.types.get(slot) != _STRING:
            return
        # Restrict both pointers to the same tracked storage. An offset or open
        # pointer computation cannot identify the first character of this write.
        dest = self.simple(read["args"][0], state).locator if read.get("args") else None
        src = self.simple(assign["args"][0], state).locator
        reaching = state.versions.get(dest) if dest and dest == src else None
        version = self.g.nodes.get(reaching.ref, {}) if reaching else {}
        if version.get("form") != "conditional_stream_write" or version[
            "producer"
        ] != self.op(read, (*path, i, "value")):
            return
        following = body[i + 2]
        field = {"decl": "init", "expr": "value", "return": "value"}.get(
            following["op"]
        )
        call = following.get(field) or {}
        if (
            not isinstance(call, dict)
            or call.get("op") != "call"
            or not call.get("callee")
        ):
            return
        args = call["args"]
        params = call.get("parameter_types", ())
        if len(args) != len(params):
            return
        copies = []
        for index, (arg, param) in enumerate(zip(args, params)):
            if param == _STRING and _library_construct(arg, _STRING, ["reference"]):
                copied = _unqualify(arg["args"][0])
                if (
                    copied.get("op") == "var"
                    and (copied.get("scope"), copied.get("name")) == slot
                ):
                    copies.append(index)
                    continue
            if any(
                n["op"] not in {"var", "literal", "cast", "library_value"}
                for n in walk_operations(arg)
            ):
                return
        if copies:
            assignment = self.op(assign, (*path, i + 1, "value"))
            self.g.edge(
                "COPIES_EXTRACTED_C_STRING",
                reaching.ref,
                assignment,
                operand=["args", 0],
                extent="characters_before_first_nul",
                conditional=True,
                requires=[
                    "getline_wrote_a_terminated_sequence_at_this_address",
                    "same_buffer_contents_reach_normal_string_assignment",
                ],
            )
            consumer = self.op(call, (*path, i + 2, field))
            for index in copies:
                self.g.edge(
                    "SUPPLIES_STRING_CALL_ARGUMENT",
                    assignment,
                    consumer,
                    operand=["args", index],
                    callee=call["callee"],
                    passing="value",
                    conditional=True,
                    requires=[
                        "same_live_string_contents_reach_copy_construction",
                        "argument_construction_and_call_entry_succeed",
                    ],
                )

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
            _system_boundary(node, self.externs[api], site, ident)
            self.unknown(state, site, state.escaped | self.external, "external_runtime")
            self.forget_bindings(
                state,
                state.exposed_names | {k for k in state.names if k[0] == "global"},
            )
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
        copy = _cstring_copy(
            node,
            effect,
            roles,
            self.externs[api],
            g.nodes.get(dest.locator, {}).get("capacity_bytes"),
        )
        if copy and self.map_strings:
            self.map_strings.c_string(
                node["args"][copy["source_operand"][1]],
                state.map_origins,
                copy["source_value"],
            )
        expression = g.node(
            "BYTE_EXPRESSION",
            [self.symbol, site],
            operation=site,
            interface=effect["interface"],
            **({"c_string_copy": copy} if copy else {}),
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
        loc = self.destination(dest, state, node, site)
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
        local_streams = set()
        former_arrays = state.arrays.copy()
        start_breaks = len(breaks) if breaks is not None else 0
        start_continues = len(continues) if continues is not None else 0
        for i, stmt in enumerate(body):
            if state is None:
                break
            here = (*path, i)
            op = stmt["op"]
            if self.map_strings and op != "expr":
                self.map_strings.statement(stmt, state.map_origins, here)
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
                    declared = _plain(stmt["type"])
                    if (
                        isinstance(declared, dict)
                        and declared.get("kind") in {"ref", "rref"}
                        and _stream_kind(
                            declared["to"],
                            {"basic_ifstream", "basic_istream", "basic_ios"},
                        )
                        == "pointer"
                    ):
                        # A reference to a mutable pointer can rebind its target.
                        state.exposed_names.add(key)
                        for part in walk_operations(initial):
                            if part["op"] == "var":
                                source = (part.get("scope"), part["name"])
                                state.exposed_names.add(source)
                                if state.names.get(source, _UNKNOWN).stream_pointer:
                                    state.names[source] = _UNKNOWN
                        init = _UNKNOWN
                    if (
                        not persistent
                        and _stream_kind(stmt["type"], {"basic_ifstream"}) == "object"
                        and _plain(stmt["type"]).get("kind") == "library"
                    ):
                        if not init.stream:
                            identity = self.op(stmt, here)
                            default_constructed = not initial or (
                                initial.get("op") == "library_construct"
                                and not initial.get("args")
                            )
                            init = _Value(stream=identity)
                            state.streams[identity] = _Stream(
                                states=frozenset({identity}),
                                unresolved=not default_constructed,
                            )
                        if scoped:
                            local_streams.add(init.stream)
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
                if (
                    stmt["value"].get("artifact_effect", {}).get("interface")
                    == "stream.getline_into"
                ):
                    self.line_consumer(body, i, state, path)
                if self.map_strings:
                    self.map_strings.statement(stmt, state.map_origins, here)
                if self.structured and calls_libc_exit(stmt["value"], self.externs):
                    state = None
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
                    for identity in local_streams:
                        s.streams.pop(identity, None)
                    s.escaped_streams.difference_update(local_streams)
                    for key, val in s.names.items():
                        if (
                            val.locator in local_allocations
                            or val.stream in local_streams
                        ):
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
        for handle, incoming in state.streams.items():
            for previous in sorted(incoming.states):
                self.g.edge(
                    "REACHES_STREAM_STATE",
                    previous,
                    marker,
                    role="loop_entry",
                    stream_object=handle,
                    conditional=True,
                )
            # An earlier visit can refill, close, or rebind this stream.
            head.streams[handle] = replace(
                incoming, states=frozenset({marker}), unresolved=True
            )
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
            for handle in state.streams.keys() & back.streams.keys():
                for previous in sorted(back.streams[handle].states - {marker}):
                    self.g.edge(
                        "REACHES_STREAM_STATE",
                        previous,
                        marker,
                        role="loop_backedge",
                        stream_object=handle,
                        conditional=True,
                    )
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
        row["name"]: row
        for row in module.get("globals", ())
        if _storage(row["type"]) or row["type"] == _STRING_MAP
    }
    for fn in functions:
        if fn.get("execution_kind") == "python_protocol":
            continue
        if any(_effect(op, externs) for op in walk_operations(fn["body"])):
            _Function(graph, fn, externs, globals_)
    return graph.result()
