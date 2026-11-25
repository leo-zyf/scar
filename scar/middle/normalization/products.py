"""M2 immutable product construction, projections and activation-safe facts."""

from copy import deepcopy

from . import sequences, values
from .values import Value


def is_sequence(value):
    return (
        value.kind in {"tuple", "list"}
        or values.is_product(value.kind)
        or values.is_sequence(value.kind)
    )


def get(frame, node, value, index, steps):
    """Project one statically selected field; never reconstruct its container."""
    if not values.is_product(value.kind):
        return value.value[index]
    members = values.product_members(value.kind)
    index = index + len(members) if index < 0 else index
    if not 0 <= index < len(members):
        frame.fail(node, "product_index", "a product index is outside its field domain")
    facts = value.facts.get("product_members", ())
    field_facts = facts[index] if len(facts) == len(members) else {}
    return Value(
        members[index],
        dict(
            op="product_get",
            type=values.wire(members[index]),
            base=value.value,
            index=index,
            effects=["STATE"],
            **values.source_site(node),
        ),
        deepcopy(field_facts),
    )


def elements(frame, node, value, steps):
    if values.is_sequence(value.kind):
        return sequences.elements(frame, node, value, steps)
    if value.kind in {"tuple", "list"}:
        return value.value
    if values.is_product(value.kind):
        return tuple(
            get(frame, node, value, index, steps)
            for index in range(len(values.product_members(value.kind)))
        )
    return None


def _source_constant(node):
    op = node.get("op")
    if op == "literal":
        return True
    if op == "unary" and node.get("operator") in {
        "pos",
        "neg",
        "uadd",
        "usub",
        "not",
        "bnot",
        "invert",
    }:
        return _source_constant(node["value"])
    return (
        op == "binary"
        and _source_constant(node["left"])
        and _source_constant(node["right"])
    )


def _constant_items(node, items):
    source_items = node.get("items", ())
    if len(source_items) != len(items):
        return None
    result = []
    for source, item in zip(source_items, items):
        if _source_constant(source) and item.value.get("op") == "literal":
            result.append(deepcopy(item.value))
        elif (
            source.get("op") == "aggregate"
            and source.get("kind") == "tuple"
            and source.get("identity", {}).get("kind") == "constant_pool"
        ):
            expression = item.facts.get("constant_expr")
            if (
                not isinstance(expression, dict)
                or expression.get("op") != "product_make"
            ):
                return None
            result.append(deepcopy(expression))
        else:
            return None
    return result


def make(frame, node, items, steps):
    """Box a qualified source tuple exactly at its original construction site."""
    identity = node.get("identity", {"kind": "unresolved"})
    if not isinstance(identity, dict):
        frame.fail(
            node, "product_identity", "tuple identity requires a qualified record"
        )
    if identity.get("kind") == "unresolved" or not all(
        values.runtime(item.kind) for item in items
    ):
        return Value("tuple", tuple(items))
    qualified = identity == {"kind": "fresh"} or (
        set(identity) == {"kind", "pool", "slot"}
        and identity.get("kind") == "constant_pool"
        and isinstance(identity.get("pool"), str)
        and bool(identity["pool"])
        and type(identity.get("slot")) is int
        and identity["slot"] >= 0
    )
    if not qualified:
        frame.fail(
            node, "product_identity", "tuple allocation identity is not qualified"
        )
    kind = values.product_kind(tuple(item.kind for item in items))
    expression = dict(
        op="product_make",
        type=values.wire(kind),
        items=[item.value for item in items],
        identity=deepcopy(identity),
        effects=["MEM", "STATE"],
        may_raise=True,
        **values.source_site(node),
    )
    facts = dict(product_members=[deepcopy(item.facts) for item in items])
    if identity["kind"] == "constant_pool":
        constant = _constant_items(node, items)
        if constant is None:
            frame.fail(
                node,
                "product_constant_pool",
                "a pooled tuple requires pure source constants",
            )
        expression["items"] = constant
        facts["constant_expr"] = deepcopy(expression)
    return frame.effect(expression, kind, steps, facts=facts)


def returned_facts(value):
    """Only immutable layout/value facts may cross activation boundaries."""
    facts = {}
    if value.kind in values.SCALARS and value.exact is not values._UNKNOWN:
        facts.update(values.numeric_facts(value.exact))
    if values.is_array(value.kind) and "shape" in value.facts:
        facts["shape"] = deepcopy(value.facts["shape"])
    if "stream_origin" in value.facts:
        facts["stream_origin"] = deepcopy(value.facts["stream_origin"])
    if values.is_product(value.kind):
        kinds = values.product_members(value.kind)
        members = value.facts.get("product_members", ())
        if len(members) == len(kinds):
            facts["product_members"] = [
                returned_facts(Value(kind, None, member))
                for kind, member in zip(kinds, members)
            ]
        if "constant_expr" in value.facts:
            facts["constant_expr"] = deepcopy(value.facts["constant_expr"])
    return facts
