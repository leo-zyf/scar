"""Nominal identities and guarded primitive projections of finite source enums."""

import math

from scar.contract import digest, json_bytes, python_provider_id
from .values import Value, literal, source_site

PROFILE = "cpython-3.12.3-ordinary-enum-v1"
_SOURCE_SHA = "c8ead615c159598370295649eb296819ad4b40d50b200c4fec2d4269bf7af9ae"
_FAMILY = "nominal_enum"


def qualify_source(frame, node, source):
    identity = python_provider_id(source)
    if not (
        source.get("status") == "verified_stdlib"
        and source.get("module") == "enum"
        and source.get("interpreter") == "cpython 3.12.3"
        and source.get("loader") == "path"
        and source.get("hash_basis") == "provider_file"
        and source.get("sha256") == _SOURCE_SHA
        and source.get("origin")
        and source.get("provider_id", identity) == identity
        and frame.provider_sources.get("enum", {}).get("provider_id") == identity
    ):
        frame.fail(
            node,
            "enum_source",
            "the enum class protocol requires its sealed CPython implementation",
        )


def _capture(frame, node, role):
    return digest(json_bytes([frame.name, source_site(node), frame.temporary(), role]))


def _require(frame, opcode):
    frame.project.require_operator(_FAMILY, opcode, profile=PROFILE)


def _guard(frame, node, steps, operation, *, continuation, resume):
    family = operation["operator_family"]
    if family == _FAMILY:
        _require(frame, operation["op"])
        _require(frame, continuation)
    matched = frame.effect(
        dict(**operation, type="bool", **source_site(node)), "bool", steps
    )
    fallback = dict(
        op=continuation,
        type="void",
        operator_family=family,
        capture=operation["capture"],
        owner=frame.name,
        resume=resume,
        noreturn=True,
        reads=["captured_source_state", "source.continuation"],
        writes=["source.continuation"],
        effects=["CTRL", "OPAQUE"],
        external_effect="UNKNOWN",
        **source_site(node),
    )
    steps.append(
        {
            "op": "if",
            "condition": matched.value,
            "then": [],
            "else": [dict(op="expr", value=fallback)],
        }
    )


def base(frame, node, namespace, steps):
    """Capture the actual import/attribute result before selecting its class protocol."""
    capture = _capture(frame, node, "enum_base")
    bound = {
        key: namespace[key]
        for key in ("module", "source_provider_id", "source_sha256", "occurrence")
    }
    _guard(
        frame,
        node,
        steps,
        dict(
            op="module_attribute_resolve",
            operator_family="module_registry",
            semantics_version=1,
            capture=capture,
            owner=frame.name,
            member="Enum",
            namespace_binding=bound,
            lookup="IMPORT_FROM" if node.get("op") == "python_import" else "attribute",
            instance_scope="per_dynamic_lookup_in_current_activation",
            candidate=dict(
                kind="source_class",
                module="enum",
                name="Enum",
                metaclass="EnumType",
                profile=PROFILE,
            ),
            checks=[
                "capture_actual_lookup_once_including_attribute_hooks",
                "match_exact_source_class_object_and_metaclass",
            ],
            reads=["source.import_binding", "module.namespace"],
            writes=["lookup.capture", "import.external_effects"],
            effects=["CTRL", "STATE", "MEM", "IO", "OPAQUE"],
            may_raise=True,
            external_effect="UNKNOWN",
        ),
        continuation="module_export_continuation_required",
        resume="continue_source_with_captured_actual_attribute_without_another_lookup",
    )
    return Value(
        "enum_base",
        "enum.Enum",
        dict(base_binding=dict(**bound, capture=capture, owner=frame.name)),
    )


def _primitive(frame, action, expression):
    # Unary signs are syntax nodes in Python, including the i64 lower bound.
    sign = None
    if expression.get("op") == "unary" and expression.get("operator") in {
        "uadd",
        "usub",
    }:
        sign = expression["operator"]
        expression = expression.get("value", {})
    value = expression.get("value")
    source_type = expression.get("type")
    protocol = source_type.get("class") if isinstance(source_type, dict) else None
    if (
        expression.get("op") != "literal"
        or type(value) not in {type(None), bool, int, str, float}
        or protocol is not None
        and protocol != type(value).__name__
    ):
        frame.fail(
            action,
            "enum_member_value",
            "a member requires one immutable primitive source literal",
        )
    if sign:
        if type(value) not in {int, float}:
            frame.fail(
                action,
                "enum_member_value",
                "a signed member needs an ordinary numeric literal",
            )
        if sign == "usub":
            value = -value
    if (
        type(value) is int
        and not -(1 << 63) <= value < 1 << 63
        or type(value) is float
        and not math.isfinite(value)
        or type(value) is str
        and any(0xD800 <= ord(c) <= 0xDFFF for c in value)
    ):
        frame.fail(
            action,
            "enum_member_value",
            "the literal needs a different primitive representation",
        )
    return value


def _members(frame, node, body):
    by_value = {}
    members = {}
    doc = node["creation_protocol"].get("implicit_bindings", {}).get("__doc__", {})
    meaningful = 0
    for action in body:
        op = action["op"]
        if (
            op == "decl"
            and action.get("initial_binding") == "unbound"
            and "init" not in action
        ):
            continue
        leading = meaningful == 0
        meaningful += 1
        if op == "pass" or (
            leading
            and op == "expr"
            and doc.get("source") == "leading_docstring"
            and action.get("value", {}).get("op") == "literal"
            and type(action["value"].get("value")) is str
        ):
            continue
        target = action.get("target", {})
        name = target.get("name", "")
        if (
            op != "assign"
            or target.get("op") != "var"
            or target.get("scope")
            or not name.isidentifier()
            or name.startswith("_")
            or name == "mro"
            or name in members
        ):
            frame.fail(
                action,
                "enum_namespace",
                "the finite enum requires ordinary unique member assignments in source order",
            )
        raw = _primitive(frame, action, action["value"])
        value = literal(raw)
        # CPython's ordinary enum uses primitive dictionary equality and retains
        # the first member, including aliases across True, 1 and 1.0.
        canonical = by_value.setdefault(raw, name)
        members[name] = dict(
            name=name,
            canonical=canonical,
            ordinal=(
                len(by_value) - 1
                if canonical == name
                else members[canonical]["ordinal"]
            ),
            value=dict(type=value.kind, value=raw),
            **source_site(action),
        )
    return members


def define(frame, node, base_value, definition, steps):
    creation = node.get("creation_protocol", {})
    if (
        creation.get("bases") != "resolve_supplied"
        or creation.get("metaclass") != "from_bases"
        or creation.get("namespace") != "prepare_then_execute"
        or set(creation.get("implicit_bindings", {}))
        - {"__module__", "__qualname__", "__doc__"}
    ):
        frame.fail(
            node,
            "enum_class_protocol",
            "the finite enum requires an ordinary class namespace and inherited metaclass",
        )
    members = _members(frame, node, definition["body"])
    symbol = node["symbol"]
    capture = _capture(frame, node, "enum_class")
    normal = dict(
        nominal_id=symbol, definition_capture=capture, definition_owner=frame.name
    )
    _guard(
        frame,
        node,
        steps,
        dict(
            op="enum_creation_matches",
            operator_family=_FAMILY,
            capture=capture,
            owner=frame.name,
            **normal,
            base_binding=base_value.facts["base_binding"],
            checks=[
                "current___build_class___and_enumtype_construction_protocol",
                "current_class_namespace_and_inherited_hooks",
            ],
            reads=["lookup.capture", "class.creation_protocol"],
            writes=[],
            effects=["CTRL", "STATE"],
            may_raise=False,
        ),
        continuation="enum_source_continuation_required",
        resume="execute_original_class_statement_with_captured_base",
    )
    _require(frame, "enum_define")
    frame.effect(
        dict(
            op="enum_define",
            type="unit",
            operator_family=_FAMILY,
            owner=frame.name,
            **normal,
            activation="per_successful_module_class_statement",
            base_binding=base_value.facts["base_binding"],
            source_name=node.get("source_name"),
            qualname=node.get("qualname"),
            creation_protocol=creation,
            members=list(members.values()),
            effects=["CTRL", "STATE", "MEM"],
            may_raise=True,
            reads=["lookup.capture", "module.namespace", "class.creation_protocol"],
            writes=["enum.class.capture", "enum.member.identities"],
            **source_site(node),
        ),
        "unit",
        steps,
        True,
    )
    return Value("enum_class", symbol, dict(identity=normal, members=members))


def member(frame, node, base_value, name, steps):
    if base_value.kind not in {"enum_class", "enum_member"}:
        return None
    owner = base_value.facts["identity"]
    capture = _capture(frame, node, "enum_attribute")
    common = dict(
        operator_family=_FAMILY, capture=capture, owner=frame.name, **owner, member=name
    )
    if base_value.kind == "enum_class":
        row = base_value.facts["members"].get(name)
        if row is None:
            frame.fail(
                node, "enum_class_attribute", f"no finite member binding for {name!r}"
            )
        first = base_value.facts["members"][row["canonical"]]
        _guard(
            frame,
            node,
            steps,
            dict(
                op="enum_member_read",
                **common,
                canonical=first["name"],
                checks=[
                    "capture_actual_attribute_once",
                    "match_canonical_identity_in_this_class_instance",
                ],
                reads=[
                    "enum.class.capture",
                    "class.namespace",
                    "enum.member.identities",
                ],
                writes=["enum.member.capture", "import.external_effects"],
                effects=["CTRL", "STATE", "MEM", "IO", "OPAQUE"],
                may_raise=True,
                external_effect="UNKNOWN",
            ),
            continuation="enum_source_continuation_required",
            resume="continue_source_with_captured_actual_class_attribute",
        )
        return Value(
            "enum_member",
            first["name"],
            dict(
                identity=owner, member_capture=capture, canonical_value=first["value"]
            ),
        )
    if name not in {"name", "value"}:
        frame.fail(
            node,
            "enum_member_attribute",
            f"{name!r} needs a further enum member protocol",
        )
    expected = (
        dict(type="text", value=base_value.value)
        if name == "name"
        else base_value.facts["canonical_value"]
    )
    common.update(
        canonical=base_value.value, member_capture=base_value.facts["member_capture"]
    )
    _guard(
        frame,
        node,
        steps,
        dict(
            op="enum_projection_matches",
            **common,
            expected=expected,
            checks=[
                "capture_actual_property_once",
                "exact_source_type",
                (
                    "binary64_bits"
                    if expected["type"] == "f64"
                    else "exact_primitive_value"
                ),
            ],
            reads=["enum.member.capture", "enum.member.state", "class.namespace"],
            writes=["enum.projection.capture", "import.external_effects"],
            effects=["CTRL", "STATE", "MEM", "IO", "OPAQUE"],
            may_raise=True,
            external_effect="UNKNOWN",
        ),
        continuation="enum_source_continuation_required",
        resume="continue_source_with_captured_actual_projection",
    )
    _require(frame, "enum_project")
    return frame.effect(
        dict(
            op="enum_project",
            type=expected["type"],
            **common,
            projection="actual_value_of_matching_capture",
            reads=["enum.projection.capture"],
            writes=[],
            effects=["STATE"],
            **source_site(node),
        ),
        expected["type"],
        steps,
        facts=literal(expected["value"]).facts,
    )
