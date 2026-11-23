"""M2 class definitions, ordinary method binding and shared instance fields."""

from dataclasses import dataclass, field

from scar.contract import walk_operations
from . import values
from .values import Value, literal


def plain_definition(node):
    return not any(
        node.get(key)
        for key in ("defaults", "decorators", "annotations", "type_params")
    ) and not any(value is not None for value in node.get("keyword_defaults", ()))


def _site(name, location):
    return name, location.get("file"), location.get("line"), location.get("column")


@dataclass
class InstanceLayout:
    name: str
    display_name: str
    body_symbol: str
    methods: dict = field(default_factory=dict)
    fields: dict = field(default_factory=dict)
    literal_writes: dict | None = None

    def record(self):
        # CPython renders at most 100 UTF-8 bytes of the instance type name.
        error_name = self.display_name.encode("utf-8")[:100].decode("utf-8", "replace")
        return dict(
            name=self.name,
            fields=[
                dict(
                    name=name,
                    type=values.wire(kind),
                    missing=dict(
                        category="attribute_error",
                        message=f"'{error_name}' object has no attribute '{name}'",
                    ),
                )
                for name, kind in self.fields.items()
            ],
            bases=[],
            methods=[],
            object_state=dict(
                layout="fixed_fields", identity="shared", initial_fields="unbound"
            ),
        )

    def observe_literal_writes(self, function):
        # An optional type hint for reads before a write. It never binds the field.
        params = function["params"]
        if not params:
            return
        receiver = params[0]["name"]
        for op in walk_operations(function["body"]):
            if op["op"] != "assign":
                continue
            target, value = op["target"], op["value"]
            base = target.get("base", {})
            if (
                target.get("op") == "member"
                and base.get("op") == "var"
                and base.get("name") == receiver
                and not base.get("scope")
                and value.get("op") == "literal"
            ):
                kinds = {
                    bool: "bool",
                    int: "i64",
                    float: "f64",
                    str: "text",
                    type(None): "unit",
                }
                kind = kinds.get(type(value.get("value")))
                if kind:
                    self.literal_writes.setdefault(target["field"], set()).add(kind)

    def possible_kinds(self, name, functions):
        if self.literal_writes is None:
            self.literal_writes = {}
            for method in self.methods.values():
                self.observe_literal_writes(functions[method])
        return self.literal_writes.get(name, set())


class ObjectModel:
    def __init__(self):
        self.layouts = {}
        self.class_bodies = set()
        self.proofs = set()

    def discharged(self, obligation):
        return (
            obligation.get("field") == "class_protocol"
            and _site(obligation.get("symbol"), obligation) in self.proofs
        )

    def define(self, frame, node):
        if not frame.initializing or frame.dynamic:
            frame.fail(
                node,
                "class_activation",
                "local or conditional class creation requires runtime class identity",
            )
        if any(
            node.get(key)
            for key in ("bases", "keywords", "decorators", "type_params", "annotations")
        ):
            frame.fail(
                node,
                "class_protocol",
                "bases, metaclass keywords, decorators or type parameters require a class creation contract",
            )
        creation = node.get("creation_protocol")
        if creation and (
            creation.get("bases") != "implicit_object"
            or creation.get("metaclass") != "default_type"
            or creation.get("namespace") != "prepare_then_execute"
            or "__classcell__" in creation.get("implicit_bindings", {})
        ):
            frame.fail(
                node,
                "class_namespace_protocol",
                "class definition requires a metaclass, namespace or lexical class-cell contract",
            )
        symbol, body = node["symbol"], node["body_symbol"]
        definition = frame.project.functions.get(body, {})
        if (
            definition.get("python_scope", {}).get("kind") != "class"
            or definition.get("owner_class") != symbol
        ):
            frame.fail(node, "class_body", f"missing class namespace body for {symbol}")
        if symbol in self.layouts:
            frame.fail(
                node, "class_rebind", f"creation of another class identity for {symbol}"
            )
        display_name = (
            node.get("source_name") or node["target"]["name"].rsplit(":", 1)[-1]
        )
        layout = InstanceLayout(symbol, display_name, body)
        doc = (creation or {}).get("implicit_bindings", {}).get("__doc__", {})
        namespace = (
            action
            for action in definition["body"]
            if not (
                action["op"] == "decl" and action.get("initial_binding") == "unbound"
            )
        )
        for index, action in enumerate(namespace):
            value = action.get("value", {})
            if (
                index == 0
                and doc.get("source") == "leading_docstring"
                and action["op"] == "expr"
                and value.get("op") == "literal"
                and type(value.get("value")) is str
            ):
                continue
            if action["op"] == "pass":
                continue
            if action["op"] != "python_define" or not plain_definition(action):
                frame.fail(
                    action,
                    "class_namespace",
                    f"{symbol} requires executable class-body effects or a definition-time protocol",
                )
            name = action["target"].get("name", "")
            method = action["symbol"]
            if (
                not name
                or name in layout.methods
                or name.startswith("__")
                and name != "__init__"
            ):
                frame.fail(
                    action,
                    "class_method_protocol",
                    f"special, private or rebound method {name!r} requires its own lookup contract",
                )
            implementation = frame.project.functions.get(method, {})
            if (
                implementation.get("owner_class") != symbol
                or implementation.get("call_protocol", {}).get("kind") != "regular"
            ):
                frame.fail(
                    action,
                    "class_method",
                    f"missing ordinary method body for {symbol}.{name}",
                )
            layout.methods[name] = method
        self.layouts[symbol] = layout
        self.class_bodies.add(body)
        self.proofs.add(_site(frame.name, node.get("location", {})))
        return Value("project_class", symbol)

    def _layout(self, frame, node, receiver, name):
        layout = self.layouts.get(values.object_name(receiver.kind))
        if layout is None:
            frame.fail(
                node,
                "object_receiver",
                "attribute access requires a closed project instance",
            )
        if name.startswith("__"):
            frame.fail(
                node,
                "object_attribute_protocol",
                f"private or protocol attribute {name!r} requires an explicit lookup contract",
            )
        return layout

    def member(self, frame, node, receiver, name, steps):
        layout = self._layout(frame, node, receiver, name)
        if name in layout.methods:
            # Attribute lookup retains this instance even if evaluating arguments
            # or a later statement rebinds the source variable.
            receiver = frame.effect(receiver.value, receiver.kind, steps)
            return Value("project_method", (layout.methods[name], receiver))
        kind = layout.fields.get(name)
        if kind is None:
            hints = layout.possible_kinds(name, frame.project.functions)
            if len(hints) != 1:
                frame.fail(
                    node,
                    "object_field_type",
                    f"no unique value type for field {layout.name}.{name}",
                )
            kind = layout.fields[name] = next(iter(hints))
        return frame.effect(
            dict(
                op="object_get",
                base=receiver.value,
                field=name,
                type=values.wire(kind),
                effects=["STATE"],
                may_raise=True,
            ),
            kind,
            steps,
        )

    def write(self, frame, node, receiver, name, value, steps):
        layout = self._layout(frame, node, receiver, name)
        if name in layout.methods:
            frame.fail(
                node,
                "object_method_shadow",
                f"instance assignment can shadow method {layout.name}.{name}",
            )
        if (
            value.kind not in values.SCALARS | {"numpy_f64"}
            and values.rank(value.kind) is None
        ):
            frame.fail(
                node,
                "object_field_type",
                f"field {layout.name}.{name} requires storage for {value.kind}",
            )
        previous = layout.fields.setdefault(name, value.kind)
        if previous != value.kind:
            frame.fail(
                node,
                "object_field_type",
                f"field {layout.name}.{name} changes type from {previous} to {value.kind}",
            )
        frame.effect(
            dict(
                op="object_set",
                base=receiver.value,
                field=name,
                value=value.value,
                type="unit",
                operand_order=["value", "base"],
                effects=["STATE"],
                may_raise=True,
            ),
            "unit",
            steps,
            discard=True,
        )

    def construct(self, frame, node, target, args, kwargs, steps):
        layout = self.layouts[target.value]
        kind = values.object_kind(layout.name)
        instance = frame.effect(
            dict(
                op="object_allocate",
                type=values.wire(kind),
                effects=["MEM", "STATE"],
                may_raise=True,
            ),
            kind,
            steps,
        )
        if init := layout.methods.get("__init__"):
            result = frame.invoke_project(
                node, init, [instance, *args], kwargs, steps, False
            )
            if result.kind != "unit":
                frame.fail(
                    node, "object_initializer_result", f"{init} must return None"
                )
        elif args or kwargs:
            frame.fail(
                node,
                "object_initializer_arguments",
                f"{layout.name} has no initializer accepting arguments",
            )
        return instance


def persist_method(frame, target, method, steps):
    symbol, receiver = method.value
    slot = "@bound_receiver/" + target["name"]
    previous = frame.project.globals.setdefault(slot, receiver.kind)
    if previous != receiver.kind:
        frame.fail(
            target,
            "object_method_binding",
            "a retained receiver changes its record type",
        )
    saved = receiver.at(slot, scope="global")
    steps.append(dict(op="assign", target=saved.value, value=receiver.value))
    return Value(method.kind, (symbol, saved))


def compare(operator, lhs, rhs):
    """Exact default equality and identity for a class with closed lookup."""
    first, second = values.object_name(lhs.kind), values.object_name(rhs.kind)
    if first is None and second is None:
        return None
    identity = operator in {"is", "isnot"}
    if not identity and (
        first is None or second is None or operator not in {"eq", "noteq"}
    ):
        return None
    equal = operator in {"eq", "is"}
    if first != second:
        return literal(not equal)
    return Value(
        "bool",
        dict(
            op="binary",
            operator="eq" if equal else "ne",
            left=lhs.value,
            right=rhs.value,
            type="bool",
        ),
    )
