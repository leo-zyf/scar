"""M2 value identities, physical type keys and conservative scalar facts."""

from dataclasses import dataclass, field
import math
import struct

_NUMPY_SCALAR_TYPES = {"numpy_bool": "bool", "numpy_f64": "f64"}
SCALARS = {"bool", "i64", "f64", "text", "unit", *_NUMPY_SCALAR_TYPES}
_UNKNOWN = object()
_MIN, _MAX = -(1 << 63), (1 << 63) - 1
_ARRAY_PREFIX = "ndarray:f64:"
_HANDLE_PREFIX = "handle:"
_OBJECT_PREFIX = "object_ref:"


def rank(kind):
    return int(kind[len(_ARRAY_PREFIX) :]) if kind.startswith(_ARRAY_PREFIX) else None


def array_kind(n):
    return _ARRAY_PREFIX + str(n)


def object_kind(name):
    return _OBJECT_PREFIX + name


def object_name(kind):
    return kind[len(_OBJECT_PREFIX) :] if kind.startswith(_OBJECT_PREFIX) else None


def wire(kind):
    if kind in _NUMPY_SCALAR_TYPES:
        return _NUMPY_SCALAR_TYPES[kind]
    n = rank(kind)
    if n is not None:
        return dict(kind="ndarray", element="f64", rank=n)
    if (name := object_name(kind)) is not None:
        return dict(kind="object_ref", record=name)
    if kind.startswith(_HANDLE_PREFIX):
        return dict(kind="handle", resource=kind[len(_HANDLE_PREFIX) :])
    return kind


def from_wire(spec):
    if not isinstance(spec, dict):
        return spec
    if spec.get("kind") == "ndarray":
        return array_kind(spec["rank"])
    if spec.get("kind") == "handle":
        return _HANDLE_PREFIX + spec["resource"]
    if spec.get("kind") == "object_ref":
        return object_kind(spec["record"])
    raise ValueError(f"unrecognized protocol value type: {spec}")


def runtime(kind):
    return (
        kind in SCALARS
        or rank(kind) is not None
        or object_name(kind) is not None
        or kind
        in {
            "handle:text_stream",
            "handle:error",
            "handle:graphics_context",
            "handle:graphics_lines",
        }
    )


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
