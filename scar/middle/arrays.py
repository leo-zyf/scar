"""M2 contracts for dense binary64 arrays, element operations and shared views."""

import math
import operator
import re

from scar.contract import walk_operations

from . import scalars
from .scalars import as_float as _as_float
from .values import Value, _UNKNOWN, array_kind, literal, rank, runtime, wire

_PROVIDED = {"array", "zeros", "float64"}
_OPERATORS = {
    "add": ("add", "add"),
    "sub": ("sub", "subtract"),
    "mult": ("mul", "multiply"),
    "div": ("div", "divide"),
    "pow": ("pow", "power"),
}
_COMPARISONS = {
    "eq": "eq",
    "noteq": "ne",
    "lt": "lt",
    "lte": "le",
    "gt": "gt",
    "gte": "ge",
}


class LiteralState:
    """Current reaching facts from source literals, never runtime tensor samples.

    Allocation tokens are unique within one normalization. Views share a token;
    copies get a fresh one. Invalidation is monotone across control alternatives,
    so restoring name bindings cannot revive a possibly overwritten value.
    """

    def __init__(self):
        self.next_token = 0
        self.contents = {}

    def create(self, shape, items=None):
        self.next_token += 1
        facts = dict(
            shape=tuple(shape), array_allocation=self.next_token, literal_prefix=()
        )
        if items is not None and all(
            type(v) in {int, float}
            and math.isfinite(v)
            and (type(v) is not int or int(float(v)) == v)
            for v in items
        ):
            self.contents[self.next_token] = (
                tuple(shape),
                tuple(float(v) for v in items),
            )
        return facts

    def region(self, value):
        token = value.facts.get("array_allocation")
        stored = self.contents.get(token)
        if stored is None:
            return None
        shape, items = stored
        prefix = value.facts.get("literal_prefix", ())
        if prefix is None:
            return None
        offset = 0
        for index, size in zip(prefix, shape):
            offset = offset * size + index
        remaining = shape[len(prefix) :]
        count = math.prod(remaining)
        return remaining, items[offset * count : (offset + 1) * count]

    def copied(self, source, *, negate=False):
        region = self.region(source)
        if region is None:
            return self.create(_shape(source))
        shape, items = region
        return self.create(shape, tuple(-v for v in items) if negate else items)

    def selected(self, source, indices):
        region = self.region(source)
        if region is None or any(type(v.exact) is not int for v in indices):
            if len(indices) == rank(source.kind):
                return {}
            token = source.facts.get("array_allocation")
            return (
                dict(array_allocation=token, literal_prefix=None)
                if token is not None
                else {}
            )
        shape, _ = region
        prefix = []
        for index, extent in zip(indices, shape):
            if not -extent <= index.exact < extent:
                return {}
            prefix.append(index.exact % extent)
        facts = dict(
            array_allocation=source.facts["array_allocation"],
            literal_prefix=(*source.facts["literal_prefix"], *prefix),
        )
        selected = Value(source.kind, source.value, facts)
        shape, items = self.region(selected)
        return dict(facts, shape=shape) if shape else {"exact": items[0]}

    def invalidate(self, source=None):
        token = source.facts.get("array_allocation") if source else None
        if token is None:
            self.contents.clear()
        else:
            self.contents.pop(token, None)

    def enter_loop(self, frame, body):
        """Kill potential back-edge writes before reasoning about any iteration."""
        operations = list(walk_operations(body))
        rebound = {
            node["target"]["name"]
            for node in operations
            if node["op"] in {"assign", "for"}
            and node.get("target", {}).get("op") == "var"
        }
        for node in operations:
            op = node["op"]
            if op in {
                "call",
                "library_call",
                "python_invoke",
            } and not frame.array_readonly_call(node, rebound):
                self.invalidate()
                break
            if op not in {"assign", "update"}:
                continue
            target = node["target"]
            if target.get("op") == "index":
                target = target["base"]
            elif op == "assign":
                continue
            if target.get("op") != "var":
                self.invalidate()
                break
            if target["name"] in rebound:
                self.invalidate()
                break
            scope = (
                frame.project.names if target.get("scope") == "global" else frame.env
            )
            value = scope.get(target["name"])
            if value is None or rank(value.kind) is not None:
                self.invalidate(value)
        return self.next_token

    def leave_loop(self, first):
        # Each execution of an allocation in the loop creates a distinct object.
        # A single symbolic pass cannot export its first iteration's contents.
        self.contents = {key: val for key, val in self.contents.items() if key <= first}


def may_write_external(function, functions, active=()):
    """Distinguish writes to fresh local arrays from caller-visible writes."""
    name = function["name"]
    if name in active:
        return True
    assignments = {}
    operations = list(walk_operations(function["body"]))
    for node in operations:
        if node["op"] == "decl" and "init" in node:
            assignments.setdefault(node["name"], []).append(node["init"])
        elif node["op"] == "assign" and node["target"].get("op") == "var":
            target = node["target"]
            if target.get("scope") != "global":
                assignments.setdefault(target["name"], []).append(node["value"])
    fresh = set()
    parameters = {p["name"] for p in function["params"]}

    def private(expr):
        op = expr["op"]
        if op in {
            "array_create",
            "array_literal",
            "array_copy",
            "array_unary",
            "array_binary",
        }:
            return True
        if op == "array_view":
            return private(expr["base"])
        return op == "var" and expr.get("scope") != "global" and expr["name"] in fresh

    while True:
        proven = {
            slot
            for slot, sources in assignments.items()
            if slot not in parameters and all(private(source) for source in sources)
        }
        if proven <= fresh:
            break
        fresh.update(proven)
    for node in operations:
        if node["op"] in {"array_store", "array_update"}:
            target = node["base"] if node["op"] == "array_store" else node["left"]
            if not private(target):
                return True
        if node["op"] == "call":
            callee = functions.get(node.get("callee"))
            if callee is None or may_write_external(callee, functions, (*active, name)):
                return True
    return False


def bind_copy(frame, node, source, steps):
    """A bound copy method captures this handle before later name rebinding."""
    captured = frame.effect(source.value, source.kind, steps, facts=source.facts)
    return Value("array_copy_method", captured)


def persist_copy(frame, target, method, steps):
    """Only a stored module binding requires a receiver beyond its initializer."""
    source = method.value
    slot = "@array-method/" + target["name"]
    frame.project.globals[slot] = source.kind
    captured = source.at(slot, scope="global")
    steps.append(dict(op="assign", target=captured.value, value=source.value))
    return Value(method.kind, captured)


def _scalar_facts(op, lhs, rhs):
    if lhs.exact is _UNKNOWN or rhs.exact is _UNKNOWN:
        return {}
    if any(
        item.kind == "i64" and int(float(item.exact)) != item.exact
        for item in (lhs, rhs)
    ):
        return {}
    evaluate = {
        "add": operator.add,
        "sub": operator.sub,
        "mult": operator.mul,
        "div": operator.truediv,
    }.get(op)
    if evaluate is None:
        return {}
    try:
        result = evaluate(float(lhs.exact), float(rhs.exact))
    except (OverflowError, ZeroDivisionError):
        return {}
    return {"exact": result} if math.isfinite(result) else {}


def _output(frame, node, kind, steps, *, facts=None, discard=False, **fields):
    frame.project.array_ops.add(node)
    if (
        rank(kind) is not None
        and node
        in {
            "array_create",
            "array_literal",
            "array_copy",
            "array_unary",
            "array_binary",
        }
        and "array_allocation" not in (facts or {})
    ):
        facts = frame.project.array_state.create(
            (facts or {}).get("shape", (None,) * rank(kind))
        )
    return frame.effect(
        dict(op=node, type=wire(kind), **fields), kind, steps, discard, facts=facts
    )


def import_provider(frame, node):
    source = node.get("source_provider", {})
    release = re.fullmatch(
        r"(\d+)\.(\d+)\.(\d+)(?:\.post\d+)?(?:\+[A-Za-z0-9.]+)?",
        str(source.get("version", "")),
    )
    if not (
        source.get("status") == "verified_distribution"
        and source.get("distribution", "").lower() == "numpy"
        and release
        and (1, 24) <= tuple(map(int, release.group(1, 2))) < (3, 0)
        and source.get("sha256")
    ):
        frame.fail(
            node,
            "provider_identity",
            "the imported module requires a verified NumPy 1.24–2.x distribution",
        )
    prior = frame.project.providers.get("numpy")
    if prior and any(source[key] != prior[key] for key in ("version", "sha256")):
        frame.fail(
            node,
            "provider_identity",
            "one program resolves conflicting numerical provider revisions",
        )
    frame.project.providers["numpy"] = {
        "distribution": source["distribution"],
        "version": source["version"],
        "sha256": source["sha256"],
        "import_effects": node["effects"],
        "contract": "closed public numerical operations with unobserved module registry and initialization hooks",
    }
    if node["request"]["form"] == "import":
        return Value("provider_namespace", "numpy")
    return provider_member(frame, node, node["member"])


def provider_member(frame, node, name):
    if name not in _PROVIDED:
        frame.fail(
            node, "provider_member", f"no closed numerical contract for numpy.{name}"
        )
    return (
        Value("dtype", "f64") if name == "float64" else Value("array_constructor", name)
    )


def _shape(item):
    dimensions = rank(item.kind)
    return () if dimensions is None else item.facts.get("shape", (None,) * dimensions)


def _broadcast(frame, node, lhs, rhs, *, inplace):
    left, right = _shape(lhs), _shape(rhs)
    if inplace and len(right) > len(left):
        frame.fail(
            node,
            "array_broadcast",
            "the RHS rank exceeds the in-place destination rank",
        )
    dimensions = max(len(left), len(right))
    left = (1,) * (dimensions - len(left)) + left
    right = (1,) * (dimensions - len(right)) + right
    result = []
    for axis, (a, b) in enumerate(zip(left, right)):
        if inplace:
            if a is not None and b not in (None, 1, a):
                frame.fail(
                    node,
                    "array_broadcast",
                    f"cannot write RHS extent {b} into extent {a} on axis {axis}",
                )
            extent = a
        elif a == b or b == 1:
            extent = a
        elif a == 1 or a is None:
            extent = b
        elif b is None:
            extent = a
        else:
            frame.fail(
                node,
                "array_broadcast",
                f"incompatible extents {a} and {b} on axis {axis}",
            )
        result.append(extent)
    return tuple(result)


def _power_path(frame, node, lhs, rhs):
    """The source numeric slot chooses a ufunc before broadcasting or conversion."""
    if rank(lhs.kind) is None or rank(rhs.kind) not in (None, 0):
        return "power"
    revision = frame.project.providers.get("numpy", {}).get("version", "")
    match = re.match(r"^(\d+)\.(\d+)", revision)
    version = tuple(map(int, match.groups())) if match else ()
    if (1, 24) <= version < (2, 3):
        choices = {
            -1: "reciprocal",
            0: "ones_like",
            0.5: "sqrt",
            1: "positive",
            2: "square",
        }
    elif (2, 3) <= version < (2, 6):
        choices = {
            "i64": {-1: "reciprocal", 2: "square"},
            "f64": {0.5: "sqrt"},
        }.get(rhs.kind, {})
    else:
        frame.fail(
            node,
            "array_power_dispatch",
            f"no scalar power dispatch contract for NumPy {revision!r}",
        )
    if rhs.exact is not _UNKNOWN:
        return choices.get(rhs.exact, "power")
    if rhs.kind == "i64":
        lo, hi = rhs.bounds
        choices = {
            v: path for v, path in choices.items() if int(v) == v and lo <= v <= hi
        }
        if lo == hi:
            return choices.get(lo, "power")
    if choices:
        frame.fail(
            node,
            "array_power_dispatch",
            "runtime exponent could require a different numerical kernel and warning operation",
        )
    return "power"


def _operand(frame, node, value):
    return (
        value.value if rank(value.kind) is not None else _as_float(frame, node, value)
    )


def _operation(frame, node, op, lhs, rhs, steps, *, inplace=False, discard=False):
    if op not in _OPERATORS:
        frame.fail(node, "array_operator", f"no binary64 array contract for {op}")
    left, right = (_operand(frame, node, item) for item in (lhs, rhs))
    shape = _broadcast(frame, node, lhs, rhs, inplace=inplace)
    operator, ufunc = _OPERATORS[op]
    dispatch = {}
    if operator == "pow":
        ufunc = _power_path(frame, node, lhs, rhs)
        dispatch["power_path"] = ufunc
    if rank(lhs.kind) is None and rank(rhs.kind) is None:
        ufunc = "scalar " + ufunc
    location = dict(node.get("location", {}))
    warning = {"operation": ufunc}
    if frame.module:
        warning["module"] = "__main__" if frame.scope.get("main") else frame.module
    if "source_line" in location:
        warning["source_line"] = location["source_line"]
    contract = dict(
        operator=operator,
        left=left,
        right=right,
        broadcasting="trailing",
        numeric_policy="numpy_default",
        warning=warning,
        location=location,
        **dispatch,
    )
    if inplace:
        frame.project.array_state.invalidate(lhs)
        frame.project.array_ops.add("array_update")
        steps.append(
            dict(op="array_update", type="unit", effects=["STATE", "IO"], **contract)
        )
        return
    return _output(
        frame,
        "array_binary",
        array_kind(len(shape)) if shape else "numpy_f64",
        steps,
        facts={"shape": shape} if shape else _scalar_facts(op, lhs, rhs),
        discard=discard,
        effects=["MEM", "VAL", "STATE", "IO"],
        **contract,
    )


def binary(frame, node, op, lhs, rhs, steps, discard=False):
    """Select NumPy arithmetic from source operand identities, including scalar loads."""
    if all(rank(v.kind) is None and v.kind != "numpy_f64" for v in (lhs, rhs)):
        return None
    if op in _COMPARISONS:
        if any(rank(v.kind) not in (None, 0) for v in (lhs, rhs)):
            frame.fail(
                node,
                "array_comparison",
                "a non-scalar comparison requires a boolean array contract",
            )
        operands = []
        for v in (lhs, rhs):
            if v.kind == "i64" and max(map(abs, v.bounds)) > 1 << 53:
                frame.fail(
                    node,
                    "integer_precision",
                    "mixed scalar comparison requires an exact integer conversion",
                )
            if rank(v.kind) == 0:
                v = _output(
                    frame,
                    "array_load",
                    "numpy_f64",
                    steps,
                    base=v.value,
                    indices=[],
                    effects=["STATE"],
                )
            operands.append(_as_float(frame, node, v))
        return Value(
            "bool",
            dict(
                op="binary",
                operator=_COMPARISONS[op],
                left=operands[0],
                right=operands[1],
                type="bool",
            ),
        )
    return _operation(frame, node, op, lhs, rhs, steps, discard=discard)


def update(frame, node, lhs, rhs, steps):
    """Update the existing storage so live views observe the write."""
    if rank(lhs.kind) is None:
        return False
    _operation(frame, node, node["operator"], lhs, rhs, steps, inplace=True)
    return True


def scalar_unary(frame, node, item, steps, discard):
    op = {"uadd": "pos", "usub": "neg"}.get(node["operator"])
    n = rank(item.kind)
    if op is None or n is None and item.kind not in {"i64", "f64", "numpy_f64"}:
        frame.fail(
            node, "unary_contract", f"unclosed {node['operator']} on {item.kind}"
        )
    if n is not None:
        if n:
            return _output(
                frame,
                "array_unary",
                item.kind,
                steps,
                facts=frame.project.array_state.copied(item, negate=op == "neg"),
                discard=discard,
                operator=op,
                value=item.value,
                effects=["MEM", "VAL"],
            )
        item = _output(
            frame,
            "array_load",
            "numpy_f64",
            steps,
            facts=frame.project.array_state.selected(item, []),
            base=item.value,
            indices=[],
            effects=["STATE"],
        )
    if item.kind == "numpy_f64":
        return Value(
            "numpy_f64",
            dict(op="unary", operator=op, value=item.value, type="f64"),
            (
                {}
                if item.exact is _UNKNOWN
                else {"exact": -item.exact if op == "neg" else +item.exact}
            ),
        )
    return scalars.unary(frame, node, item)


def capture_sequence(frame, node, steps):
    kind = node["kind"]
    if kind not in {"tuple", "list"}:
        frame.fail(
            node, "sequence_contract", f"{kind} has no ordered array or shape contract"
        )
    items = []
    for child in node["items"]:
        item = frame.evaluate(child, steps)
        if runtime(item.kind):
            # A sequence captures the current value, including the current array handle.
            # A module-level sequence may outlive its initializer's local temporaries.
            if item.exact is not _UNKNOWN:
                item = Value(item.kind, literal(item.exact).value, item.facts)
            elif frame.initializing:
                name = frame.temporary()
                slot = f"@sequence/{frame.name}/{name}"
                frame.project.globals[slot] = item.kind
                captured = item.at(slot, scope="global")
                steps.append(dict(op="assign", target=captured.value, value=item.value))
                item = captured
            else:
                item = frame.effect(item.value, item.kind, steps, facts=item.facts)
        elif item.kind not in {"list", "tuple"}:
            frame.fail(
                node,
                "sequence_escape",
                "dynamic sequence elements have no closed value representation",
            )
        items.append(item)
    return Value(kind, tuple(items))


def _tree(frame, node, item):
    if item.kind not in {"list", "tuple"}:
        return (), [item]
    branches = [_tree(frame, node, part) for part in item.value]
    shape = branches[0][0] if branches else ()
    if any(dims != shape for dims, _ in branches):
        frame.fail(
            node,
            "array_shape",
            "ragged sequences require a different element or layout contract",
        )
    return (len(branches), *shape), [v for _, items in branches for v in items]


def _literal_array(frame, node, source, steps, *, force, discard=False):
    shape, items = _tree(frame, node, source)
    if not force and items and all(v.kind == "i64" for v in items):
        frame.fail(
            node,
            "array_dtype",
            "default integer arrays cannot be lowered to a binary64 array",
        )
    values = [_as_float(frame, node, v) for v in items]
    return _output(
        frame,
        "array_literal",
        array_kind(len(shape)),
        steps,
        facts=(
            frame.project.array_state.create(shape, [v.exact for v in items])
            if all(v.exact is not _UNKNOWN for v in items)
            else {"shape": shape}
        ),
        discard=discard,
        shape=[literal(d).value for d in shape],
        values=values,
        effects=["MEM", "VAL"],
    )


def construct(frame, node, target, args, keywords, steps, discard):
    options = dict(keywords)
    if target.kind == "array_copy_method":
        if args or options:
            frame.fail(
                node,
                "array_copy_contract",
                "copy supports the default contiguous order",
            )
        source = target.value
        return _output(
            frame,
            "array_copy",
            source.kind,
            steps,
            discard=discard,
            facts=frame.project.array_state.copied(source),
            value=source.value,
            effects=["MEM", "STATE", "XFER"],
        )
    name = target.value
    formal = "shape" if name == "zeros" else "object"
    if not args and formal in options:
        args = [options.pop(formal)]
    if (
        len(args) not in {1, 2}
        or formal in options
        or len(args) == 2
        and "dtype" in options
    ):
        frame.fail(
            node, "array_arguments", f"{name} requires one {formal} and optional dtype"
        )
    source = args[0]
    dtype = args[1] if len(args) == 2 else options.pop("dtype", literal(None))
    explicit = dtype.kind != "unit"
    if explicit and not (
        dtype.kind == "dtype"
        and dtype.value == "f64"
        or dtype.kind == "builtin"
        and dtype.value == "float"
        or dtype.kind == "text"
        and dtype.exact in {"float64", "f8", "d"}
    ):
        frame.fail(
            node,
            "array_dtype",
            "only an explicitly resolved binary64 dtype is supported",
        )
    if name == "array" and "copy" in options:
        copy = options.pop("copy")
        if copy.kind != "bool" or copy.exact is not True:
            frame.fail(
                node,
                "array_copy_contract",
                "a requested noncopy or runtime copy mode needs a separate ownership contract",
            )
    if options:
        frame.fail(
            node, "array_arguments", f"unsupported {name} options: {', '.join(options)}"
        )
    if name == "zeros":
        dims = source.value if source.kind in {"list", "tuple"} else (source,)
        if any(
            v.kind != "i64" or v.exact is not _UNKNOWN and v.exact < 0 for v in dims
        ):
            frame.fail(
                node,
                "array_shape",
                "dimensions must be nonnegative index-width integers",
            )
        extents = tuple(v.exact if v.exact is not _UNKNOWN else None for v in dims)
        return _output(
            frame,
            "array_create",
            array_kind(len(dims)),
            steps,
            discard=discard,
            facts={"shape": extents},
            shape=[v.value for v in dims],
            initialization="zero",
            effects=["MEM", "VAL"],
        )
    if rank(source.kind) is not None:
        return _output(
            frame,
            "array_copy",
            source.kind,
            steps,
            discard=discard,
            facts=frame.project.array_state.copied(source),
            value=source.value,
            effects=["MEM", "STATE", "XFER"],
        )
    return _literal_array(frame, node, source, steps, force=explicit, discard=discard)


def index(frame, node, steps, rhs=None, discard=False):
    base = frame.evaluate(node["base"], steps)
    axes = []
    for port in node["indices"]:
        item = frame.evaluate(port, steps)
        axes.extend(item.value if item.kind == "tuple" else (item,))
    if base.kind in {"tuple", "list"}:
        if rhs is not None:
            frame.fail(
                node,
                "sequence_mutation",
                "a mutable Python sequence cannot be eliminated when any alias writes it",
            )
        if (
            len(axes) == 1
            and axes[0].kind == "i64"
            and type(axes[0].exact) is int
            and -len(base.value) <= axes[0].exact < len(base.value)
        ):
            return base.value[axes[0].exact]
        frame.fail(
            node,
            "sequence_index",
            "a captured sequence requires one proven in-bounds integer index",
        )
    n = rank(base.kind)
    if n is None or len(axes) > n or any(v.kind != "i64" for v in axes):
        frame.fail(
            node,
            "array_index",
            "only integer prefix indexing within the declared array rank is supported",
        )
    shape = base.facts.get("shape", (None,) * n)
    if any(
        d is not None and v.exact is not _UNKNOWN and not -d <= v.exact < d
        for v, d in zip(axes, shape)
    ):
        frame.fail(node, "array_bounds", "a constant index is outside the current axis")
    indices = [v.value for v in axes]
    result = shape[len(axes) :]
    if rhs is not None:
        if rhs.kind in {"tuple", "list"}:
            rhs = _literal_array(frame, node, rhs, steps, force=True)
        source_rank = rank(rhs.kind)
        if source_rank is not None:
            actual = rhs.facts.get("shape", (None,) * source_rank)
            extra = max(0, source_rank - len(result))
            if (
                (not result and source_rank)
                or any(src not in (None, 1) for src in actual[:extra])
                or any(
                    src not in (None, 1) and dst is not None and src != dst
                    for src, dst in zip(reversed(actual), reversed(result))
                )
            ):
                frame.fail(
                    node,
                    "array_broadcast",
                    "RHS cannot broadcast into the selected destination shape",
                )
            value = rhs.value
        else:
            value = _as_float(frame, node, rhs)
        frame.project.array_state.invalidate(base)
        frame.project.array_ops.add("array_store")
        steps.append(
            dict(
                op="array_store",
                base=base.value,
                indices=indices,
                value=value,
                effects=["STATE"],
                broadcasting="trailing",
            )
        )
        return
    if not result:
        return _output(
            frame,
            "array_load",
            "numpy_f64",
            steps,
            facts=frame.project.array_state.selected(base, axes),
            discard=discard,
            base=base.value,
            indices=indices,
            effects=["STATE"],
        )
    return _output(
        frame,
        "array_view",
        array_kind(len(result)),
        steps,
        discard=discard,
        facts={"shape": result, **frame.project.array_state.selected(base, axes)},
        base=base.value,
        indices=indices,
        effects=["STATE", "MEM"],
    )
