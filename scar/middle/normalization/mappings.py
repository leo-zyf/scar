"""M2 ordered builtin mappings and tagged scalar values, without content caching.

Only text keys and builtin unit/bool/i64/f64/text or shared acyclic mappings
belong to this profile. Views and iterators retain identity and observe live
mapping state. Source programs, custom hashing and Python are never executed.
"""

from . import values
from .values import Value, literal as _literal

PROFILE = "builtin-scalar-map-v1"
MAPPING = "mapping_ref"
STRUCTURED = "structured_value"
ITEMS = "mapping_items_ref"
ITERATOR = "mapping_iterator_ref"
_SCALARS = frozenset({"unit", "bool", "i64", "f64", "text"})


def _mapping_facts(value):
    facts = {}
    for name in ("mapping_origin", "mapping_descendant_of"):
        origin = value.facts.get(name)
        if (
            isinstance(origin, dict)
            and set(origin) == {"owner", "parameter"}
            and all(isinstance(part, str) and part for part in origin.values())
        ):
            facts[name] = dict(origin)
    return facts


def _op(frame, node, opcode, kind, *, effects, may_raise=False, **operands):
    frame.project.require_operator("builtin_scalar_mapping", opcode, profile=PROFILE)
    return dict(
        op=opcode,
        type=values.wire(kind),
        profile=PROFILE,
        semantics_version=1,
        operator_family="builtin_scalar_mapping",
        effects=effects,
        may_raise=may_raise,
        **operands,
        **values.source_site(node),
    )


def pack(frame, node, value):
    if value.kind == STRUCTURED:
        return value
    if value.kind not in _SCALARS | {MAPPING}:
        frame.fail(
            node,
            "mapping_value",
            "mapping values require builtin scalars or acyclic shared mappings",
        )
    return Value(
        STRUCTURED,
        _op(
            frame,
            node,
            "structured_pack",
            STRUCTURED,
            value=value.value,
            tag=value.kind,
            effects=["VAL", "STATE"],
            may_raise=True,
        ),
        _mapping_facts(value) if value.kind == MAPPING else {},
    )


def as_mapping(frame, node, value, steps, *, access):
    if access not in {"items", "subscript", "unpack"}:
        frame.fail(node, "mapping_access", "unsupported mapping receiver protocol")
    if value.kind == MAPPING:
        return value
    if value.kind != STRUCTURED:
        frame.fail(
            node,
            "mapping_receiver",
            "a mapping receiver needs its closed tagged value protocol",
        )
    return frame.effect(
        _op(
            frame,
            node,
            "structured_as_mapping",
            MAPPING,
            value=value.value,
            access=access,
            effects=["STATE"],
            may_raise=True,
        ),
        MAPPING,
        steps,
        facts=_mapping_facts(value),
    )


def is_mapping(frame, node, value):
    if value.kind == MAPPING:
        return _literal(True)
    if value.kind != STRUCTURED:
        if value.kind in _SCALARS:
            return _literal(False)
        frame.fail(
            node,
            "mapping_isinstance",
            "isinstance needs a closed builtin scalar/mapping value",
        )
    return Value(
        "bool",
        _op(
            frame,
            node,
            "structured_is_mapping",
            "bool",
            value=value.value,
            effects=["STATE"],
        ),
    )


def _make(frame, node, entries, steps):
    return frame.effect(
        _op(
            frame,
            node,
            "mapping_make",
            MAPPING,
            entries=entries,
            effects=["MEM", "STATE"],
            may_raise=True,
        ),
        MAPPING,
        steps,
        facts={"mapping_builder": True},
    )


def _key(frame, node, value):
    if value.kind != "text":
        frame.fail(
            node, "mapping_key", "this mapping profile requires a proven text key"
        )
    return value


def put(frame, node, base, key, value, steps, *, construction=False):
    base = as_mapping(frame, node, base, steps, access="subscript")
    key = _key(frame, node, key)
    packed = pack(frame, node, value)
    if construction:
        if base.facts.get("mapping_builder") is not True:
            frame.fail(
                node,
                "mapping_acyclicity",
                "construction writes require an unpublished fresh builder",
            )
        acyclicity = dict(semantics_version=1, rule="unpublished_fresh_builder")
    else:
        expression = packed.value
        tag = (
            expression.get("tag") if expression.get("op") == "structured_pack" else None
        )
        if tag not in _SCALARS:
            frame.fail(
                node,
                "mapping_acyclicity",
                "a possibly mapping-valued mutation needs a noncyclic alias proof",
            )
        acyclicity = dict(semantics_version=1, rule="nonmapping_scalar", tag=tag)
    frame.effect(
        _op(
            frame,
            node,
            "mapping_put",
            "unit",
            base=base.value,
            key=key.value,
            value=packed.value,
            acyclicity=acyclicity,
            effects=["MEM", "STATE"],
            may_raise=True,
            reads=["mapping.state"],
            writes=["mapping.state"],
        ),
        "unit",
        steps,
        True,
    )
    return _literal(None)


def _merge(frame, node, base, source, steps):
    if base.facts.get("mapping_builder") is not True:
        frame.fail(
            node,
            "mapping_acyclicity",
            "unpacking requires an unpublished fresh builder",
        )
    source = as_mapping(frame, node, source, steps, access="unpack")
    frame.effect(
        _op(
            frame,
            node,
            "mapping_merge",
            "unit",
            base=base.value,
            source=source.value,
            acyclicity=dict(semantics_version=1, rule="unpublished_fresh_builder"),
            effects=["MEM", "STATE"],
            may_raise=True,
            reads=["mapping.state"],
            writes=["mapping.state"],
        ),
        "unit",
        steps,
        True,
    )


def _plan(frame, node):
    construction = node.get("construction", {})
    if construction.get("status") != "captured":
        frame.fail(
            node,
            "mapping_insertion_schedule",
            "dictionary construction needs its captured source insertion schedule",
        )
    entries, cursor, depth = node.get("entries", []), 0, 0
    plan = construction.get("steps")
    if (
        not isinstance(entries, list)
        or not isinstance(plan, list)
        or any(not isinstance(step, dict) for step in plan)
    ):
        frame.fail(
            node,
            "mapping_insertion_schedule",
            "dictionary construction needs ordered entry and step tables",
        )
    for step in plan:
        kind = step.get("kind")
        if kind == "build_pairs":
            first, count = step.get("first"), step.get("count")
            if (
                type(first) is not int
                or type(count) is not int
                or first != cursor
                or count < 0
            ):
                frame.fail(
                    node,
                    "mapping_insertion_schedule",
                    "invalid pair-batch source positions",
                )
            indexes = range(first, first + count)
            depth += 1
        elif kind in {"insert_pair", "unpack_mapping"}:
            if (
                type(step.get("entry")) is not int
                or step["entry"] != cursor
                or depth < 1
            ):
                frame.fail(
                    node,
                    "mapping_insertion_schedule",
                    "invalid insertion source position",
                )
            indexes = (cursor,)
        elif kind == "merge_batch" and depth >= 2:
            depth -= 1
            continue
        else:
            frame.fail(
                node,
                "mapping_insertion_schedule",
                "unsupported dictionary insertion step",
            )
        for index in indexes:
            expected = "unpack" if kind == "unpack_mapping" else "pair"
            if index >= len(entries) or entries[index].get("kind") != expected:
                frame.fail(
                    node,
                    "mapping_insertion_schedule",
                    "insertion step does not match its source entry",
                )
            cursor += 1
    if cursor != len(entries) or depth != 1:
        frame.fail(
            node,
            "mapping_insertion_schedule",
            "dictionary insertion schedule is incomplete",
        )
    return construction["steps"]


def literal(frame, node, steps):
    plan = _plan(frame, node)
    entries, stack = node["entries"], []

    def pair(index):
        entry = entries[index]
        key = _key(frame, entry["key"], frame.evaluate(entry["key"], steps))
        key = frame.effect(key.value, key.kind, steps)
        value = pack(frame, entry["value"], frame.evaluate(entry["value"], steps))
        value = frame.effect(value.value, value.kind, steps)
        return key, value

    for step in plan:
        kind = step["kind"]
        if kind == "build_pairs":
            items = [
                pair(i) for i in range(step["first"], step["first"] + step["count"])
            ]
            stack.append(
                _make(
                    frame,
                    node,
                    [dict(key=k.value, value=v.value) for k, v in items],
                    steps,
                )
            )
        elif kind == "insert_pair":
            key, value = pair(step["entry"])
            put(frame, node, stack[-1], key, value, steps, construction=True)
        elif kind == "unpack_mapping":
            entry = entries[step["entry"]]
            source = as_mapping(
                frame,
                node,
                frame.evaluate(entry["mapping"], steps),
                steps,
                access="unpack",
            )
            source = frame.effect(source.value, source.kind, steps, facts=source.facts)
            _merge(frame, node, stack[-1], source, steps)
        else:
            right = stack.pop()
            _merge(frame, node, stack[-1], right, steps)
    return stack[0]


def get(frame, node, base, key, steps):
    base = as_mapping(frame, node, base, steps, access="subscript")
    key = _key(frame, node, key)
    profile = frame.scope.get("text_profile")
    if profile not in frame.project.text_profiles:
        frame.fail(
            node,
            "mapping_key_repr",
            "missing-key errors require the captured Unicode text profile",
        )
    return frame.effect(
        _op(
            frame,
            node,
            "mapping_get",
            STRUCTURED,
            base=base.value,
            key=key.value,
            text_profile=profile,
            effects=["STATE"],
            may_raise=True,
            reads=["mapping.state"],
        ),
        STRUCTURED,
        steps,
    )


def member(frame, node, base, name, steps):
    if base.kind not in {MAPPING, STRUCTURED}:
        return None
    if name != "items":
        frame.fail(
            node, "mapping_member", "only the builtin mapping items view is closed"
        )
    base = as_mapping(frame, node, base, steps, access="items")
    base = frame.effect(base.value, base.kind, steps, facts=base.facts)
    return Value("mapping_method", (base, name))


def invoke(frame, node, target, args, keywords, steps, discard=False):
    if target.kind != "mapping_method":
        return None
    if keywords or args:
        frame.fail(node, "mapping_items", "dict.items accepts no arguments")
    base, name = target.value
    if name != "items":
        frame.fail(node, "mapping_member", "unsupported mapping method")
    return frame.effect(
        _op(
            frame,
            node,
            "mapping_items",
            ITEMS,
            base=base.value,
            effects=["MEM", "STATE"],
            may_raise=True,
        ),
        ITEMS,
        steps,
        discard,
        facts=_mapping_facts(base),
    )


def size(frame, node, value):
    if value.kind != MAPPING:
        frame.fail(node, "mapping_size", "len requires a proven builtin mapping")
    return Value(
        "i64",
        _op(
            frame,
            node,
            "mapping_size",
            "i64",
            base=value.value,
            effects=["STATE"],
            reads=["mapping.state"],
        ),
    )


def truth(frame, node, value):
    if value.kind == STRUCTURED:
        return Value(
            "bool",
            _op(
                frame,
                node,
                "structured_truth",
                "bool",
                value=value.value,
                effects=["STATE"],
                reads=["mapping.state"],
            ),
        )
    if value.kind == MAPPING:
        return Value(
            "bool",
            dict(
                op="binary",
                operator="ne",
                type="bool",
                left=size(frame, node, value).value,
                right=_literal(0).value,
            ),
        )
    return None


def identity(frame, node, left, right):
    if node.get("operator") not in {"is", "isnot"}:
        frame.fail(node, "mapping_identity", "mapping identity requires is or isnot")
    if left.kind != MAPPING or right.kind != MAPPING:
        frame.fail(
            node,
            "mapping_identity",
            "mapping identity needs two proven mapping references",
        )
    return Value(
        "bool",
        _op(
            frame,
            node,
            "mapping_identity",
            "bool",
            left=left.value,
            right=right.value,
            operator=node["operator"],
            effects=["STATE"],
        ),
    )


def compare(frame, node, left, right, steps):
    if left.kind not in {MAPPING, STRUCTURED} and right.kind not in {
        MAPPING,
        STRUCTURED,
    }:
        return None
    if (left.kind, right.kind) not in {
        (STRUCTURED, "text"),
        ("text", STRUCTURED),
    } or node["operator"] not in {"eq", "ne", "noteq"}:
        frame.fail(
            node,
            "mapping_compare",
            "only tagged-value/text equality is closed; mapping deep equality needs scalar identity semantics",
        )
    return Value(
        "bool",
        _op(
            frame,
            node,
            "structured_compare",
            "bool",
            left=left.value,
            right=right.value,
            operator="eq" if node["operator"] == "eq" else "ne",
            effects=["VAL", "STATE"],
        ),
    )


def iteration(frame, node, view, steps):
    if view.kind != ITEMS:
        frame.fail(
            node, "mapping_iteration", "mapping iteration requires its live items view"
        )
    facts = _mapping_facts(view)
    captured = frame.effect(view.value, view.kind, steps, facts=facts)
    cursor = frame.effect(
        _op(
            frame,
            node,
            "mapping_iterator",
            ITERATOR,
            value=captured.value,
            effects=["MEM", "STATE"],
            may_raise=True,
        ),
        ITERATOR,
        steps,
        facts=facts,
    )
    condition = _op(
        frame,
        node,
        "mapping_next",
        "bool",
        base=cursor.value,
        effects=["MEM", "STATE"],
        may_raise=True,
        reads=["mapping.state", "mapping_iterator.state"],
        writes=["mapping_iterator.state"],
    )
    kind = values.product_kind(("text", STRUCTURED))
    origin = facts.get("mapping_origin") or facts.get("mapping_descendant_of")
    current = Value(
        kind,
        _op(
            frame,
            node,
            "mapping_current",
            kind,
            base=cursor.value,
            effects=["STATE"],
            requires=["successful_mapping_next_for_this_iterator"],
        ),
        (
            dict(product_members=[{}, {"mapping_descendant_of": dict(origin)}])
            if origin is not None
            else {}
        ),
    )
    return (
        dict(op="for", init=[], condition=condition, step=None, body=[]),
        current,
        None,
        False,
        False,
        [],
    )


def comprehension(frame, node, steps, lower_generators):
    def make_result(out):
        return _make(frame, node, [], out)

    def put_entry(result, active_frame, out):
        key = _key(active_frame, node, active_frame.evaluate(node["key"], out))
        key = active_frame.effect(key.value, key.kind, out)
        value = pack(active_frame, node, active_frame.evaluate(node["value"], out))
        value = active_frame.effect(value.value, value.kind, out)
        put(active_frame, node, result, key, value, out, construction=True)

    return lower_generators(node, steps, make_result, put_entry)
