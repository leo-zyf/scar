"""M2 value identities, physical type keys and conservative scalar facts."""

from dataclasses import dataclass, field
import math
import posixpath
import struct

from scar.contract import array_rank_domain

_NUMPY_SCALAR_TYPES = {"numpy_bool": "bool", "numpy_f64": "f64"}
# These are compile-time sets of possible protocols, never runtime tags. A
# consumer may erase the distinction only after proving one common operation.
_SCALAR_CHOICES = {
    "binary64_choice": frozenset({"f64", "numpy_f64"}),
    "boolean_choice": frozenset({"bool", "numpy_bool"}),
}
SCALARS = {"bool", "i64", "f64", "text", "unit", *_NUMPY_SCALAR_TYPES, *_SCALAR_CHOICES}
_UNKNOWN = object()
_MIN, _MAX = -(1 << 63), (1 << 63) - 1
_ARRAY_PREFIX = "ndarray:f64:"
_HANDLE_PREFIX = "handle:"
_OBJECT_PREFIX = "object_ref:"


def is_array(kind):
    return kind.startswith(_ARRAY_PREFIX)


def rank_domain(kind):
    """Possible ranks: empty for a non-array, None for an unbounded runtime rank."""
    if not is_array(kind):
        return ()
    suffix = kind[len(_ARRAY_PREFIX) :]
    return None if suffix == "*" else tuple(map(int, suffix.split("|")))


def rank(kind):
    """Return a proven rank. Use is_array to distinguish arrays from scalars."""
    domain = rank_domain(kind)
    return domain[0] if domain is not None and len(domain) == 1 else None


def array_kind(n):
    if n is None:
        return _ARRAY_PREFIX + "*"
    domain = (n,) if type(n) is int else tuple(sorted(set(n)))
    if not domain or any(type(k) is not int or k < 0 for k in domain):
        raise ValueError("array rank domain requires nonnegative integers")
    return _ARRAY_PREFIX + "|".join(map(str, domain))


def join_arrays(kinds):
    domains = [rank_domain(kind) for kind in kinds]
    return array_kind(
        None
        if any(domain is None for domain in domains)
        else tuple(n for domain in domains for n in domain)
    )


def storage_compatible(left, right):
    return wire(left) == wire(right) or is_array(left) and is_array(right)


def object_kind(name):
    return _OBJECT_PREFIX + name


def object_name(kind):
    return kind[len(_OBJECT_PREFIX) :] if kind.startswith(_OBJECT_PREFIX) else None


def protocols(kind):
    return _SCALAR_CHOICES.get(kind, frozenset({kind}))


def choice(kinds):
    """Join behavior domains without introducing a conversion or a tagged value."""
    possible = frozenset(p for kind in kinds for p in protocols(kind))
    if len(possible) == 1:
        return next(iter(possible))
    return next(
        (k for k, members in _SCALAR_CHOICES.items() if members == possible), None
    )


def wire(kind):
    if kind.startswith("handle:process:"):
        return dict(kind="handle", resource="process")
    if kind in _SCALAR_CHOICES:
        return "f64" if kind == "binary64_choice" else "bool"
    if kind in _NUMPY_SCALAR_TYPES:
        return _NUMPY_SCALAR_TYPES[kind]
    if is_array(kind):
        n = rank(kind)
        domain = rank_domain(kind)
        return dict(
            kind="ndarray",
            element="f64",
            rank=(
                n
                if n is not None
                else {"values": list(domain)} if domain is not None else {"minimum": 0}
            ),
        )
    if (name := object_name(kind)) is not None:
        return dict(kind="object_ref", record=name)
    if kind.startswith(_HANDLE_PREFIX):
        return dict(kind="handle", resource=kind[len(_HANDLE_PREFIX) :])
    return kind


def from_wire(spec):
    if not isinstance(spec, dict):
        return spec
    if spec.get("kind") == "ndarray":
        if spec.get("element") != "f64":
            raise ValueError("unregistered array element protocol")
        return array_kind(array_rank_domain(spec))
    if spec.get("kind") == "handle":
        return _HANDLE_PREFIX + spec["resource"]
    if spec.get("kind") == "object_ref":
        return object_kind(spec["record"])
    raise ValueError(f"unrecognized protocol value type: {spec}")


def runtime(kind):
    return (
        kind in SCALARS
        or kind.startswith("handle:process:")
        or is_array(kind)
        or object_name(kind) is not None
        or kind
        in {
            "handle:text_stream",
            "handle:process",
            "handle:error",
            "handle:graphics_context",
            "handle:graphics_lines",
        }
    )


def binding_targets(target):
    """Yield assignment destinations in source order, keeping each index or member whole."""
    if not isinstance(target, dict):
        return
    if target.get("op") == "unpack":
        for item in target["items"]:
            yield from binding_targets(item)
    else:
        yield target


def written_targets(node):
    """Bound destinations of a source operation; receiver/index reads are separate."""
    op = node.get("op")
    if op in {"assign", "update", "for", "python_define", "python_define_class"}:
        yield from binding_targets(node.get("target"))
    elif op in {"python_import", "exception_handler"}:
        yield from binding_targets(node.get("binding"))
    elif op == "with_context":
        for resource in node["resources"]:
            yield from binding_targets(resource.get("binding"))
    elif op == "unbind":
        for target in node["targets"]:
            yield from binding_targets(target)


@dataclass(frozen=True)
class Value:
    kind: str
    value: object
    facts: dict = field(default_factory=dict)

    @property
    def exact(self):
        return self.facts.get("exact", _UNKNOWN)

    @property
    def bounds(self):
        return self.facts.get("range", (_MIN, _MAX))

    def at(self, name, **attributes):
        return Value(
            self.kind,
            dict(op="var", name=name, type=wire(self.kind), **attributes),
            self.facts,
        )


def numeric_facts(value):
    if type(value) is int:
        return {"exact": value, "range": (value, value)}
    return {"exact": value} if type(value) is not float or math.isfinite(value) else {}


def literal(value):
    kind = {bool: "bool", int: "i64", float: "f64", str: "text", type(None): "unit"}[
        type(value)
    ]
    return Value(kind, dict(op="literal", type=kind, value=value), numeric_facts(value))


def source_site(node):
    return {key: node[key] for key in ("site", "location") if node.get(key) is not None}


def scalar_evidence(value):
    """Return only exact JSON scalars representable at the stated wire type."""
    if value.kind not in SCALARS:
        return None
    item, kind = value.exact, wire(value.kind)
    if kind == "f64" and type(item) is int and -(1 << 53) <= item <= 1 << 53:
        item = float(item)
    if (
        type(item)
        is not {
            "text": str,
            "i64": int,
            "f64": float,
            "bool": bool,
            "unit": type(None),
        }[kind]
    ):
        return None
    if kind == "f64" and not math.isfinite(item):
        return None
    if kind == "i64" and not _MIN <= item <= _MAX:
        return None
    return dict(type=kind, exact=item)


def argument_evidence(args, **options):
    positions = [scalar_evidence(value) for value in args]
    known_options = {
        key: fact
        for key, value in options.items()
        if (fact := scalar_evidence(value)) is not None
    }
    return (
        dict(argument_facts=dict(args=positions, **known_options))
        if any(fact is not None for fact in positions) or known_options
        else {}
    )


def identical(left, right):
    """Compare analysis records without merging distinct floating bit patterns."""
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        return struct.pack("!d", left) == struct.pack("!d", right)
    if isinstance(left, Value):
        return (
            left.kind == right.kind
            and identical(left.value, right.value)
            and identical(left.facts, right.facts)
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            identical(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            identical(a, b) for a, b in zip(left, right)
        )
    return left == right


def scalar_spelling(value, profile):
    """Evaluate the declared primitive text profile, never an object's protocol."""
    if type(value) is str:
        return value
    if type(value) is bool:
        return profile.get("boolean_true" if value else "boolean_false", _UNKNOWN)
    if value is None:
        return profile.get("unit", _UNKNOWN)
    if (
        type(value) is int
        and profile.get("integer_base") == 10
        and _MIN <= value <= _MAX
    ):
        return str(value)
    if (
        type(value) is float
        and math.isfinite(value)
        and profile.get("float") == "shortest_binary64"
    ):
        # Python's primitive binary64 spelling follows this IR profile: shortest
        # round trip, fixed notation for decimal exponents -4 through 15.
        return str(value)
    return _UNKNOWN


def exact_scalar(value, kind):
    """Keep a scalar fact only when its value is exact at the destination type."""
    if not isinstance(kind, str):
        return _UNKNOWN
    if kind == "text" and type(value) is str:
        return value
    if kind == "bool" and type(value) is bool:
        return value
    if kind == "unit" and value is None:
        return value
    if kind == "f64" and type(value) is float and math.isfinite(value):
        return value
    if kind in {"i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"}:
        bits = int(kind[1:])
        lower, upper = (
            (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
            if kind[0] == "i"
            else (0, (1 << bits) - 1)
        )
        if type(value) is int and lower <= value <= upper:
            return value
    return _UNKNOWN


def evidence_scalar(fact, kind):
    if not isinstance(fact, dict) or fact.get("type") != kind:
        return _UNKNOWN
    return exact_scalar(fact.get("exact", _UNKNOWN), kind)


_SUMMARY_SCALARS = {
    "text",
    "bool",
    "unit",
    "i8",
    "i16",
    "i32",
    "i64",
    "u8",
    "u16",
    "u32",
    "u64",
    "f64",
}


def result_summaries(functions):
    """Summarize closed scalar value paths, without deleting their calls."""
    summaries = {}

    def function(name):
        if name in summaries:
            return summaries[name]
        summaries[name] = None  # Recursive result dependencies stay unresolved.
        fn = functions[name]
        if (
            fn.get("noreturn")
            or not isinstance(fn.get("result"), str)
            or fn["result"] not in _SUMMARY_SCALARS
        ):
            return None

        def typed(summary, kind):
            return (
                dict(kind="scalar", type=kind, value=summary)
                if summary is not None
                else None
            )

        scope = {
            parameter["name"]: typed(
                dict(kind="parameter", index=index), parameter["type"]
            )
            for index, parameter in enumerate(fn.get("params", ()))
            if isinstance(parameter.get("type"), str)
            and parameter["type"] in _SUMMARY_SCALARS
        }
        types = {
            parameter["name"]: parameter["type"] for parameter in fn.get("params", ())
        }

        def value(node):
            if not isinstance(node, dict):
                return None
            op = node["op"]
            if op == "var" and node.get("scope") != "global":
                return scope.get(node["name"])
            if op == "literal" and type(node.get("value")) in {
                str,
                int,
                float,
                bool,
                type(None),
            }:
                constant = exact_scalar(node["value"], node.get("type"))
                return (
                    dict(kind="literal", value=constant)
                    if constant is not _UNKNOWN
                    else None
                )
            if op == "text_concat":
                parts = [value(part) for part in node["parts"]]
                return (
                    dict(kind="concat", parts=parts)
                    if all(part is not None for part in parts)
                    else None
                )
            if op == "host_call" and node["interface"] == "path.join":
                parts = [value(part) for part in node["args"]]
                return (
                    dict(kind="path_join", parts=parts)
                    if all(part is not None for part in parts)
                    else None
                )
            if (
                op == "call"
                and node.get("callee") in functions
                and function(node["callee"]) is not None
            ):
                parts = [value(part) for part in node["args"]]
                if len(parts) == len(
                    functions[node["callee"]].get("params", ())
                ) and all(part is not None for part in parts):
                    return dict(kind="call", target=node["callee"], args=parts)
            return None

        for statement in fn["body"]:
            op = statement["op"]
            if op == "return":
                summaries[name] = typed(value(statement.get("value")), fn["result"])
                return summaries[name]
            if (
                op == "decl"
                and isinstance(statement.get("type"), str)
                and statement["type"] in _SUMMARY_SCALARS
                and statement.get("scope") != "global"
                and statement.get("storage") not in {"static", "thread_local", "extern"}
            ):
                result = value(statement.get("init")) if "init" in statement else None
                if "init" in statement and result is None:
                    return None
                types[statement["name"]] = statement["type"]
                scope[statement["name"]] = typed(result, statement["type"])
            elif (
                op == "assign"
                and statement["target"]["op"] == "var"
                and statement["target"].get("scope") != "global"
                and statement["target"]["name"] in scope
            ):
                result = value(statement["value"])
                if result is None:
                    return None
                target = statement["target"]["name"]
                scope[target] = typed(result, types[target])
            else:
                return None
        return None

    for name in functions:
        function(name)
    return summaries


def apply_result(summary, arguments, summaries):
    if summary is None:
        return _UNKNOWN
    kind = summary["kind"]
    if kind == "scalar":
        return exact_scalar(
            apply_result(summary["value"], arguments, summaries), summary["type"]
        )
    if kind == "parameter":
        return arguments[summary["index"]]
    if kind == "literal":
        return summary["value"]
    parts = [
        apply_result(part, arguments, summaries)
        for part in summary.get("parts", summary.get("args", ()))
    ]
    if kind == "call":
        return apply_result(summaries[summary["target"]], parts, summaries)
    if not all(type(part) is str for part in parts):
        return _UNKNOWN
    if kind == "concat":
        return "".join(parts)
    if kind == "path_join" and parts:
        return posixpath.join(*parts)
    return _UNKNOWN
