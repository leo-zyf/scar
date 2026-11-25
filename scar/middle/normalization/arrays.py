"""M2 contracts for dense binary64 arrays, element operations and shared views."""

import math
import operator
import re

from scar.contract import python_source_boundary, walk_operations

from . import exceptions, products, scalars, sequences, text
from .scalars import as_float as _as_float
from .values import (
    Value,
    _UNKNOWN,
    array_kind,
    is_array,
    is_product,
    is_sequence,
    product_members,
    rank_domain,
    binding_targets,
    literal,
    protocols,
    rank,
    runtime,
    wire,
)

_PROVIDED = {"array", "zeros", "float64", "linspace", "meshgrid", "loadtxt"}
_TEXT_ARRAY_SOURCE = {
    "numpy": "09295a80660f17925ae23765ce8cbd7ff7ceae968d5f2f89349f1cb74c0b9e11",
    "numpy.lib._npyio_impl": "51eb4572baad1e8198f71b3488a3cbcdb81ac85aa3c15cc2708cd907198a1fb5",
}
_LINEAR_SPACE_SOURCE = {
    "numpy": "09295a80660f17925ae23765ce8cbd7ff7ceae968d5f2f89349f1cb74c0b9e11",
    "numpy._core": "837ce8aec8693095d2e1c7c306d385d1785a50be97a009935c64cae04e3555d2",
    "numpy._core.function_base": "97925f5f2a271088cbff838a149a6312335d8dac80ac0314f0c85a31b442c1b5",
}
_COORDINATE_GRID_SOURCE = {
    "numpy": _LINEAR_SPACE_SOURCE["numpy"],
    "numpy.lib._function_base_impl": "6b892b2733cdd036443c360cbf5ff702f62e2b37affe7a541c424236d8f18a27",
}
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
        if shape is None:
            return dict(array_allocation=self.next_token, literal_region=None)
        facts = dict(
            shape=tuple(shape),
            array_allocation=self.next_token,
            literal_region=(0, tuple(shape)),
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
        _, items = stored
        layout = value.facts.get("literal_region")
        if layout is None:
            return None
        offset, shape = layout
        count = math.prod(shape)
        return shape, items[offset : offset + count]

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
                dict(array_allocation=token, literal_region=None)
                if token is not None
                else {}
            )
        shape, _ = region
        offset = source.facts["literal_region"][0]
        for axis, (index, extent) in enumerate(zip(indices, shape)):
            if not -extent <= index.exact < extent:
                return {}
            offset += index.exact % extent * math.prod(shape[axis + 1 :])
        remaining = shape[len(indices) :]
        facts = dict(
            array_allocation=source.facts["array_allocation"],
            literal_region=(offset, remaining),
        )
        selected = Value(source.kind, source.value, facts)
        shape, items = self.region(selected)
        return dict(facts, shape=shape) if shape else {"exact": items[0]}

    def reshaped(self, source, shape):
        facts = dict(shape=tuple(shape))
        token = source.facts.get("array_allocation")
        if token is not None:
            facts.update(array_allocation=token, literal_region=None)
            if self.region(source) is not None and all(type(n) is int for n in shape):
                facts["literal_region"] = (
                    source.facts["literal_region"][0],
                    tuple(shape),
                )
        return facts

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
            target["name"]
            for node in operations
            if node["op"] in {"assign", "for"}
            for target in binding_targets(node.get("target", {}))
            if target.get("op") == "var"
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
            for target in binding_targets(node["target"]):
                if target.get("op") == "index":
                    target = target["base"]
                elif op == "assign":
                    continue
                if target.get("op") != "var" or target["name"] in rebound:
                    self.invalidate()
                    break
                scope = (
                    frame.project.names
                    if target.get("scope") == "global"
                    else frame.env
                )
                value = scope.get(target["name"])
                if value is None or is_array(value.kind):
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
            "array_iota",
            "array_literal",
            "array_copy",
            "array_broadcast_copy",
            "array_unary",
            "array_binary",
        }:
            return True
        if op in {"array_view", "array_reshape"}:
            return private(expr["base"] if op == "array_view" else expr["value"])
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


def persist_method(frame, target, method, steps):
    """Only a stored module binding requires a receiver beyond its initializer."""
    source = method.value
    slot = "@array-method/" + target["name"]
    frame.project.globals[slot] = source.kind
    captured = source.at(slot, scope="global")
    steps.append(dict(op="assign", target=captured.value, value=source.value))
    return Value(method.kind, captured)


def member(frame, node, source, name, steps):
    """Separate array identity from the currently known rank and extents."""
    if not is_array(source.kind):
        return None
    if name in {"copy", "reshape"}:
        captured = frame.effect(source.value, source.kind, steps, facts=source.facts)
        return Value(
            "array_copy_method" if name == "copy" else "array_reshape_method", captured
        )
    dimensions = rank(source.kind)
    if name == "ndim":
        if dimensions is not None:
            return literal(dimensions)
        domain = rank_domain(source.kind)
        return _output(
            frame,
            "array_rank",
            "i64",
            steps,
            value=source.value,
            effects=["STATE"],
            facts=(
                {"range": (min(domain), max(domain))}
                if domain is not None
                else {"range": (0, (1 << 63) - 1)}
            ),
        )
    if name == "size":
        shape = _shape(source)
        if shape is not None and all(type(n) is int for n in shape):
            return literal(math.prod(shape))
        return _output(
            frame,
            "array_size",
            "i64",
            steps,
            value=source.value,
            effects=["STATE"],
            facts={"range": (0, (1 << 63) - 1)},
        )
    if name == "shape":
        if dimensions is None:
            frame.fail(
                node,
                "array_shape_rank",
                "a runtime-length shape tuple needs a sequence contract",
            )
        shape = _shape(source)
        return Value(
            "tuple",
            tuple(
                (
                    literal(extent)
                    if extent is not None
                    else _output(
                        frame,
                        "array_extent",
                        "i64",
                        steps,
                        value=source.value,
                        axis=literal(axis).value,
                        effects=["STATE"],
                        facts={"range": (0, (1 << 63) - 1)},
                    )
                )
                for axis, extent in enumerate(shape)
            ),
        )
    frame.fail(node, "array_member", f"no closed array member contract for {name}")


def _reshape(frame, node, source, args, options, steps, discard):
    if frame.project.providers.get("numpy", {}).get("version") != "2.5.5":
        frame.fail(
            node,
            "array_reshape_profile",
            "reshape requires its registered source dimension and error profile",
        )
    order, copy = options.pop("order", literal("C")), options.pop("copy", literal(None))
    if (
        options
        or order.kind != "text"
        or order.exact != "C"
        or copy.kind != "unit"
        or not args
    ):
        frame.fail(
            node,
            "array_reshape",
            "reshape requires positional dimensions, C order and the default copy policy",
        )
    dims = (
        products.elements(frame, node, args[0], steps)
        if len(args) == 1 and products.is_sequence(args[0])
        else args
    )
    if any(value.kind != "i64" for value in dims):
        frame.fail(node, "array_shape", "reshape dimensions need index-width integers")
    shape = tuple(
        value.exact if type(value.exact) is int and value.exact >= 0 else None
        for value in dims
    )
    known = _shape(source)
    negative = [
        i
        for i, value in enumerate(dims)
        if type(value.exact) is int and value.exact < 0
    ]
    if (
        known is not None
        and all(type(n) is int for n in known)
        and len(negative) == 1
        and all(type(v.exact) is int for v in dims)
    ):
        product = math.prod(v.exact for v in dims if v.exact >= 0)
        size = math.prod(known)
        if product and size % product == 0:
            shape = tuple(
                size // product if i == negative[0] else value.exact
                for i, value in enumerate(dims)
            )
    return _output(
        frame,
        "array_reshape",
        array_kind(len(dims)),
        steps,
        facts=frame.project.array_state.reshaped(source, shape),
        discard=discard,
        value=source.value,
        shape=[value.value for value in dims],
        layout="C",
        inference="numpy_single_negative_dimension",
        effects=["STATE", "MEM"],
        may_raise=True,
    )


def _load_text(frame, node, args, options, steps, discard):
    profile = _implementation(frame, node, "loadtxt", _TEXT_ARRAY_SOURCE)
    if not args and "fname" in options:
        args = [options.pop("fname")]
    if len(args) != 1 or "fname" in options:
        frame.fail(
            node,
            "array_text_arguments",
            "loadtxt requires one source and closed default parsing options",
        )
    defaults = {
        "delimiter": None,
        "comments": "#",
        "converters": None,
        "skiprows": 0,
        "usecols": None,
        "unpack": False,
        "ndmin": 0,
        "encoding": None,
        "max_rows": None,
        "quotechar": None,
        "like": None,
    }
    dtype = options.pop("dtype", literal(None))
    if not (
        dtype.kind == "unit"
        or dtype.kind == "dtype"
        and dtype.value == "f64"
        or dtype.kind == "builtin"
        and dtype.value == "float"
        or dtype.kind == "text"
        and dtype.exact in {"float64", "f8", "d"}
    ):
        frame.fail(
            node,
            "array_dtype",
            "the registered text reader requires source binary64 elements",
        )
    if set(options) - defaults.keys() or any(
        value.exact is _UNKNOWN
        or type(value.exact) is not type(defaults[key])
        or value.exact != defaults[key]
        for key, value in options.items()
    ):
        frame.fail(
            node,
            "array_text_options",
            "nondefault text parsing needs its own source contract",
        )
    source = args[0]
    owned = source.kind == "text"
    if not owned and source.kind != "handle:text_stream":
        frame.fail(
            node,
            "array_text_source",
            "text arrays require a path or a qualified text stream",
        )
    if owned:
        frame.project.require_operator(
            "text_array_source",
            "array_source_open",
            profile="numpy-2.5.5-datasource-v1",
        )
        source = frame.effect(
            dict(
                op="array_source_open",
                operator_family="text_array_source",
                source=source.value,
                resolution=dict(
                    profile="numpy-2.5.5-datasource-v1",
                    mode="rt",
                    encoding="source_default",
                    errors="strict",
                    compressed_fallback=True,
                    url_cache=True,
                ),
                type=wire("handle:text_stream"),
                effects=["IO", "STATE", "MEM", "XFER"],
                reads=["filesystem", "network", "process.cwd"],
                writes=["filesystem", "network", "new.stream"],
                may_raise=True,
                **{key: node[key] for key in ("site", "location") if key in node},
            ),
            "handle:text_stream",
            steps,
        )
    frame.project.require_operator(
        "text_array_io", "array_text_read", profile="numpy-2.5.5-whitespace-f64-v1"
    )
    facts = frame.project.array_state.create(None)
    kind = array_kind((0, 1, 2))
    warning = dict(
        operation="loadtxt",
        module="__main__" if frame.scope.get("main") else frame.module,
        source_line=node.get("location", {}).get("source_line", ""),
    )
    read = dict(
        op="array_text_read",
        operator_family="text_array_io",
        source=source.value,
        source_origin=source.facts.get("stream_origin", {"kind": "unresolved"}),
        parser=dict(
            profile="numpy-2.5.5-whitespace-f64-v1",
            encoding="unicode",
            delimiter="whitespace_runs",
            comments="#",
            ndmin=0,
            source_implementation=profile["sha256"],
        ),
        type=wire(kind),
        effects=["IO", "STATE", "VAL", "MEM"],
        reads=["filesystem", "stream.state"],
        writes=["new.array_storage", "stream.state"],
        may_raise=True,
        warning=warning,
        text_profile=frame.scope.get("text_profile"),
        **{key: node[key] for key in ("site", "location") if key in node},
    )
    if not owned:
        return frame.effect(read, kind, steps, discard, facts=facts)
    name = frame.temporary()
    result = Value(kind, None, facts).at(name)
    steps.append(dict(op="decl", name=name, type=wire(kind)))
    steps.append(
        dict(
            op="resource_scope",
            resource=source.value,
            body=[dict(op="assign", target=result.value, value=read)],
        )
    )
    return literal(None) if discard else result


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
        is_array(kind)
        and node
        in {
            "array_create",
            "array_iota",
            "array_literal",
            "array_copy",
            "array_broadcast_copy",
            "array_unary",
            "array_binary",
        }
        and "array_allocation" not in (facts or {})
    ):
        facts = frame.project.array_state.create(
            (facts or {}).get(
                "shape", None if rank(kind) is None else (None,) * rank(kind)
            )
        )
    return frame.effect(
        dict(op=node, type=wire(kind), **fields), kind, steps, discard, facts=facts
    )


def import_provider(frame, node):
    source = node.get("source_provider", {})
    if python_source_boundary(source) != "numpy-public-v1":
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
        "exports": {**(prior or {}).get("exports", {}), **source.get("exports", {})},
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
    if is_array(item.kind) and dimensions is None:
        return None
    return () if dimensions is None else item.facts.get("shape", (None,) * dimensions)


def _broadcast(frame, node, lhs, rhs, *, inplace):
    left, right = _shape(lhs), _shape(rhs)
    if left is None or right is None:
        frame.fail(node, "array_rank", "broadcast needs resolved operand ranks")
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
    possible = protocols(rhs.kind)
    if len(possible) > 1:
        paths = {
            _power_path(frame, node, lhs, Value(kind, rhs.value, rhs.facts))
            for kind in possible
        }
        if len(paths) == 1:
            return paths.pop()
        frame.fail(
            node,
            "array_power_dispatch",
            "selected exponent protocols require different numerical kernels or warnings",
        )
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
    return value.value if is_array(value.kind) else _as_float(frame, node, value)


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
    warning.update(node.get("diagnostic_context", {}))
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
    if all(
        not is_array(v.kind) and v.kind not in {"numpy_f64", "numpy_bool"}
        for v in (lhs, rhs)
    ):
        return None
    if any(is_array(v.kind) and rank(v.kind) is None for v in (lhs, rhs)):
        frame.fail(
            node,
            "array_rank",
            "array arithmetic requires resolved ranks or an explicit scalar/array result contract",
        )
    if op in _COMPARISONS:
        if any(rank(v.kind) not in (None, 0) for v in (lhs, rhs)):
            frame.fail(
                node,
                "array_comparison",
                "a non-scalar comparison requires a boolean array contract",
            )
        operands = []
        integral = all(v.kind in {"bool", "numpy_bool", "i64"} for v in (lhs, rhs))
        for v in (lhs, rhs):
            if not integral and v.kind == "i64" and max(map(abs, v.bounds)) > 1 << 53:
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
                    facts=frame.project.array_state.selected(v, []),
                    base=v.value,
                    indices=[],
                    effects=["STATE"],
                )
            if v.kind in {"bool", "numpy_bool"}:
                expr = dict(op="cast", type="i64" if integral else "f64", value=v.value)
            else:
                expr = v.value if integral else _as_float(frame, node, v)
            operands.append((expr, v.exact))
        symbol = _COMPARISONS[op]
        (left, a), (right, b) = operands
        facts = {}
        if a is not _UNKNOWN and b is not _UNKNOWN:
            facts["exact"] = getattr(operator, symbol)(a, b)
        return Value(
            "numpy_bool",
            dict(
                op="binary",
                operator=symbol,
                left=left,
                right=right,
                type="bool",
            ),
            facts,
        )
    if any(v.kind == "numpy_bool" for v in (lhs, rhs)):
        frame.fail(
            node,
            "array_operator_domain",
            f"{op} needs a NumPy Boolean promotion contract",
        )
    return _operation(frame, node, op, lhs, rhs, steps, discard=discard)


def update(frame, node, lhs, rhs, steps):
    """Update the existing storage so live views observe the write."""
    if not is_array(lhs.kind):
        return False
    _operation(frame, node, node["operator"], lhs, rhs, steps, inplace=True)
    return True


def scalar_unary(frame, node, item, steps, discard):
    op = {"uadd": "pos", "usub": "neg"}.get(node["operator"])
    n = rank(item.kind)
    if is_array(item.kind) and n is None:
        frame.fail(
            node,
            "array_rank",
            "unary array results need a resolved zero-dimensional scalar protocol",
        )
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
    result = _capture_values(
        frame,
        node,
        kind,
        (frame.evaluate(child, steps) for child in node["items"]),
        steps,
    )
    if kind == "list" and id(node) not in frame.project.sequences.static_lists:
        return sequences.capture(frame, node, result.value, steps)
    if kind == "tuple" and any(is_sequence(value.kind) for value in result.value):
        resolved = []
        for value in result.value:
            if is_sequence(value.kind):
                sequences._checked(
                    frame, node, lambda: frame.project.sequences.element(value.kind)
                )
                value = Value(
                    frame.project.sequences.resolve_kind(value.kind, strict=True),
                    frame.project.sequences.resolve(value.value),
                    value.facts,
                )
            resolved.append(value)
        result = Value(kind, tuple(resolved))
    return (
        products.make(frame, node, result.value, steps) if kind == "tuple" else result
    )


def _capture_values(frame, node, kind, source, steps):
    items = []
    for item in source:
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


def _tree(frame, node, item, steps):
    if is_sequence(item.kind):
        frame.fail(
            node,
            "array_sequence_input",
            "a mutable runtime sequence needs a sequence-to-array contract",
        )
    elements = products.elements(frame, node, item, steps)
    if elements is None:
        return (), [item]
    branches = [_tree(frame, node, part, steps) for part in elements]
    shape = branches[0][0] if branches else ()
    if any(dims != shape for dims, _ in branches):
        frame.fail(
            node,
            "array_shape",
            "ragged sequences require a different element or layout contract",
        )
    return (len(branches), *shape), [v for _, items in branches for v in items]


def _literal_array(frame, node, source, steps, *, force, discard=False):
    shape, items = _tree(frame, node, source, steps)
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


def _implementation(frame, node, name, sources):
    """Bind a registered composite contract to its sealed public export evidence."""
    provider = frame.project.providers.get("numpy", {})
    exported = provider.get("exports", {}).get(name, {})
    candidates = exported.get("candidates", ())
    implementation = next(reversed(sources))
    matches = [
        p
        for p in candidates
        if p.get("sha256") == sources[implementation] and p.get("name") == name
    ]
    if provider.get("version") != "2.5.5" or len(matches) != 1:
        frame.fail(
            node,
            "provider_implementation",
            f"{name} needs captured source evidence for its registered implementation profile",
        )
    profile = matches[0]
    route = {p["module"]: p.get("sha256") for p in profile.get("via", ())}
    if provider.get("sha256") != sources["numpy"] or any(
        route.get(m) != sha for m, sha in sources.items()
    ):
        frame.fail(
            node,
            "provider_implementation",
            "the public API export path differs from the registered source profile",
        )
    return profile


def _linspace(frame, node, args, keywords, steps, discard):
    """Expand a versioned public sequence contract into neutral scalar/array ops."""
    first_operation = len(steps)
    profile = _implementation(frame, node, "linspace", _LINEAR_SPACE_SOURCE)
    names = ("start", "stop", "num", "endpoint", "retstep", "dtype", "axis")
    defaults = (
        None,
        None,
        literal(50),
        literal(True),
        literal(False),
        literal(None),
        literal(0),
    )
    if len(args) > len(names) or set(keywords) - {*names, "device"}:
        frame.fail(
            node, "array_arguments", "linspace has unbound or excessive arguments"
        )
    bound = []
    for index, (name, default) in enumerate(zip(names, defaults)):
        if index < len(args):
            if name in keywords:
                frame.fail(node, "array_arguments", f"linspace binds {name} twice")
            bound.append(args[index])
        else:
            value = keywords.get(name, default)
            if value is None:
                frame.fail(node, "array_arguments", f"linspace requires {name}")
            bound.append(value)
    start, stop, count, endpoint, retstep, dtype, axis = bound
    if count.kind == "bool":
        count = scalars.construct(frame, node, "int", [count], {})
    if count.kind != "i64" or count.bounds[1] > 1 << 53:
        frame.fail(
            node,
            "array_count",
            "the exact coordinate sequence requires a proven index-width count no larger than 2**53",
        )
    if count.bounds[1] < 0:
        frame.fail(
            node,
            "array_count",
            "this source invocation raises ValueError for a negative sample count",
        )
    if not (
        dtype.kind == "unit"
        or dtype.kind == "dtype"
        and dtype.value == "f64"
        or dtype.kind == "builtin"
        and dtype.value == "float"
        or dtype.kind == "text"
        and dtype.exact in {"float64", "f8", "d"}
    ):
        frame.fail(
            node, "array_dtype", "linspace requires a binary64/default output dtype"
        )
    if (
        axis.kind != "i64"
        or axis.exact not in {0, -1}
        or keywords.get("device", literal(None)).kind != "unit"
    ):
        frame.fail(
            node,
            "array_layout",
            "scalar endpoints require axis 0/-1 and the default CPU device",
        )
    endpoint, retstep = frame.truth(endpoint, node), frame.truth(retstep, node)
    if type(retstep.exact) is not bool:
        frame.fail(
            node, "array_return", "retstep must determine one source return layout"
        )
    start, stop = [
        scalars.construct(frame, node, "float", [v], {}) for v in (start, stop)
    ]

    def scalar(op, a, b, kind="i64"):
        return dict(op="binary", type=kind, operator=op, left=a, right=b)

    if count.bounds[0] < 0:
        bad = []
        rendered = text.render(frame, node, count, "str", literal(""), bad)
        message = text._concat(
            frame,
            [
                literal("Number of samples, "),
                rendered,
                literal(", must be non-negative."),
            ],
            bad,
        )
        error = exceptions.construct(
            frame, node, exceptions.builtin("ValueError"), [message], {}, bad
        )
        bad.append(dict(op="throw", value=error.value))
        steps.append(
            dict(
                op="if",
                condition=scalar("lt", count.value, literal(0).value, "bool"),
                then=bad,
                **{"else": []},
            )
        )
        count = Value("i64", count.value, dict(range=(0, count.bounds[1])))
    lower, upper = count.bounds
    div = (
        count
        if endpoint.exact is False
        else Value(
            "i64",
            scalar("sub", count.value, literal(1).value),
            dict(range=(lower - 1, upper - 1)),
        )
    )
    if endpoint.exact is _UNKNOWN:
        div = Value(
            "i64",
            dict(
                op="select",
                type="i64",
                condition=endpoint.value,
                then=div.value,
                **{"else": count.value},
            ),
            dict(range=(lower - 1, upper)),
        )
    div = frame.effect(div.value, div.kind, steps, facts=div.facts)
    positive = True if div.bounds[0] > 0 else False if div.bounds[1] <= 0 else None
    if retstep.exact and positive is None:
        frame.fail(
            node,
            "array_return",
            "the returned step can have either a Python float or NumPy scalar source type",
        )

    def site(line, operation):
        source_line = profile.get("locations", {}).get(str(line))
        if source_line is None:
            frame.fail(
                node,
                "provider_implementation",
                "captured sequence profile lacks a required operation site",
            )
        return dict(
            location=dict(
                file=profile["origin"],
                diagnostic_file=profile["origin"],
                line=line,
                source_line=source_line,
            ),
            diagnostic_context=dict(module=profile["module"], operation=operation),
        )

    def arithmetic(line, op, a, b, out, *, inplace=False, operation=None):
        return _operation(
            frame,
            site(line, operation or _OPERATORS[op][1]),
            op,
            a,
            b,
            out,
            inplace=inplace,
        )

    delta = arithmetic(141, "sub", stop, start, steps)
    extent = count.exact if count.exact is not _UNKNOWN else None
    ramp = _output(
        frame,
        "array_iota",
        array_kind(1),
        steps,
        count=count.value,
        max_count=1 << 53,
        facts={"shape": (extent,)},
        effects=["MEM", "VAL"],
    )
    denominator = Value("f64", _as_float(frame, node, div))

    def regular(out):
        step = arithmetic(
            152, "div", delta, denominator, out, operation="scalar divide"
        )
        zero = binary(frame, node, "eq", step, literal(0.0), out)
        zero = frame.truth(zero, node)
        tiny, normal = [], []
        if zero.exact is not False:
            arithmetic(157, "div", ramp, denominator, tiny, inplace=True)
            arithmetic(159, "mult", ramp, delta, tiny, inplace=True)
        if zero.exact is not True:
            arithmetic(163, "mult", ramp, step, normal, inplace=True)
        if zero.exact is not _UNKNOWN:
            out.extend(tiny if zero.exact else normal)
        else:
            out.append(
                dict(op="if", condition=zero.value, then=tiny, **{"else": normal})
            )
        return ramp, step

    def degenerate(out):
        return arithmetic(171, "mult", ramp, delta, out), Value(
            "f64", dict(op="literal", type="f64", value="nan")
        )

    if positive is not None:
        result, step = (regular if positive else degenerate)(steps)
    else:
        yes, no = [], []
        first, _ = regular(yes)
        last, _ = degenerate(no)
        result = Value(
            array_kind(1), None, frame.project.array_state.create((extent,))
        ).at(frame.temporary())
        steps.append(dict(op="decl", name=result.value["name"], type=wire(result.kind)))
        for body, selected in ((yes, first), (no, last)):
            body.append(dict(op="assign", target=result.value, value=selected.value))
        steps.append(
            dict(
                op="if",
                condition=scalar("gt", div.value, literal(0).value, "bool"),
                then=yes,
                **{"else": no},
            )
        )
        step = None
    arithmetic(173, "add", result, start, steps, inplace=True)
    if endpoint.exact is not False and upper > 1:
        store = dict(
            op="array_store",
            base=result.value,
            indices=[literal(-1).value],
            value=stop.value,
            effects=["STATE"],
            broadcasting="trailing",
        )
        frame.project.array_ops.add("array_store")
        if endpoint.exact is True and lower > 1:
            steps.append(store)
        else:
            guard = scalar(
                "land",
                endpoint.value,
                scalar("gt", count.value, literal(1).value, "bool"),
                "bool",
            )
            steps.append(dict(op="if", condition=guard, then=[store], **{"else": []}))
    region = "sequence/" + node.get("site", f"{frame.name}/{first_operation}")
    selected = (
        literal(None)
        if discard
        else (
            _capture_values(frame, node, "tuple", (result, step), steps)
            if retstep.exact
            else result
        )
    )
    pending = list(steps[first_operation:])
    while pending:
        generated = pending.pop()
        generated.setdefault("origin_region", region)
        for branch in ("then", "else", "body"):
            pending.extend(generated.get(branch, ()))
    frame.project.decisions.append(
        dict(
            code="expanded_sequence_region",
            region=region,
            source_site=node.get("site"),
            symbol=frame.name,
            location=node.get("location"),
            provider=profile["module"],
            implementation_sha256=profile["sha256"],
            leaf="array_iota",
            binding_contract="registered release public API with pristine exported namespace and numeric inputs; static export candidates alone do not establish runtime identity",
            proof="ordered scalar-f64 source implementation with underflow branch, fresh private storage and endpoint overwrite; base/flags do not escape",
        )
    )
    return selected


def _meshgrid(frame, node, args, keywords, steps, discard):
    """Factor a coordinate region into shared layouts and explicit materializations."""
    profile = _implementation(frame, node, "meshgrid", _COORDINATE_GRID_SOURCE)
    if set(keywords) - {"copy", "sparse", "indexing"}:
        frame.fail(node, "array_arguments", "meshgrid has unbound keyword arguments")
    indexing = keywords.get("indexing", literal("xy"))
    if indexing.kind != "text" or indexing.exact not in {"xy", "ij"}:
        frame.fail(
            node, "array_layout", "meshgrid needs a proven xy or ij indexing mode"
        )
    if len(args) > 64:
        frame.fail(
            node,
            "array_rank",
            "the registered numerical provider supports at most 64 axes",
        )
    flags = [
        frame.truth(keywords.get(k, literal(v)), node)
        for k, v in (("copy", True), ("sparse", False))
    ]
    if any(type(v.exact) is not bool for v in flags):
        frame.fail(
            node, "array_layout", "meshgrid ownership and sparse layout must be proven"
        )
    copy, sparse = (v.exact for v in flags)
    if args and not copy and not sparse:
        frame.fail(
            node,
            "array_alias",
            "dense noncopy grids require overlapping writable views and their source warning contract",
        )
    first = len(steps)
    vectors, sizes, extents = [], [], []
    for item in args:
        if not is_array(item.kind):
            item = _literal_array(frame, node, item, steps, force=False)
        shape = _shape(item)
        if shape is not None and all(type(n) is int for n in shape):
            count = math.prod(shape)
            size = literal(count)
        else:
            count = None
            size = _output(
                frame,
                "array_size",
                "i64",
                steps,
                value=item.value,
                effects=["STATE"],
                facts={"range": (0, (1 << 63) - 1)},
            )
        vectors.append(item)
        sizes.append(size)
        extents.append(count)
    order = list(range(len(args)))
    if indexing.exact == "xy" and len(order) > 1:
        order[:2] = (1, 0)
    dense_shape = tuple(extents[i] for i in order)
    result = []
    for i, item in enumerate(vectors):
        axis = order.index(i)
        shape = tuple(extents[i] if j == axis else 1 for j in range(len(args)))
        dimensions = [
            sizes[i].value if j == axis else literal(1).value for j in range(len(args))
        ]
        view = _output(
            frame,
            "array_reshape",
            array_kind(len(args)),
            steps,
            value=item.value,
            shape=dimensions,
            layout="C",
            effects=["STATE", "MEM"],
            facts=frame.project.array_state.reshaped(item, shape),
        )
        if sparse:
            output = (
                _output(
                    frame,
                    "array_copy",
                    view.kind,
                    steps,
                    value=view.value,
                    effects=["MEM", "STATE", "XFER"],
                    facts=frame.project.array_state.copied(view),
                )
                if copy
                else view
            )
        else:
            output = _output(
                frame,
                "array_broadcast_copy",
                view.kind,
                steps,
                value=view.value,
                shape=[sizes[j].value for j in order],
                broadcasting="trailing",
                effects=["MEM", "STATE", "XFER"],
                facts={"shape": dense_shape},
            )
        result.append(output)
    region = "coordinates/" + (node.get("site") or frame.temporary())
    pending = list(steps[first:])
    while pending:
        stmt = pending.pop()
        stmt["origin_region"] = region
        for key in ("body", "then", "else"):
            pending.extend(stmt.get(key, ()))
    frame.project.decisions.append(
        dict(
            code="expanded_coordinate_region",
            symbol=frame.name,
            region=region,
            source_site=node.get("site"),
            location=node.get("location"),
            provider=profile["module"],
            implementation_sha256=profile["sha256"],
            axes=order,
            sparse=sparse,
            copy=copy,
            proof="C-order flatten and singleton reshape retain source storage; explicit broadcast/copy produces independent outputs; xy permutes the first two coordinate axes only",
            binding_contract="registered release public API with pristine exported namespace and numeric inputs; static export candidates alone do not establish runtime identity",
        )
    )
    return (
        literal(None)
        if discard
        else _capture_values(frame, node, "tuple", result, steps)
    )


def construct(frame, node, target, args, keywords, steps, discard):
    options = dict(keywords)
    if target.kind == "array_reshape_method":
        return _reshape(frame, node, target.value, args, options, steps, discard)
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
    if name == "loadtxt":
        return _load_text(frame, node, args, options, steps, discard)
    if name == "linspace":
        return _linspace(frame, node, args, options, steps, discard)
    if name == "meshgrid":
        return _meshgrid(frame, node, args, options, steps, discard)
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
        dims = (
            products.elements(frame, node, source, steps)
            if products.is_sequence(source)
            else (source,)
        )
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
    if is_array(source.kind):
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
    if is_sequence(base.kind):
        base = frame.effect(base.value, base.kind, steps, facts=base.facts)
    axes = []
    for port in node["indices"]:
        item = frame.evaluate(port, steps)
        axes.extend(
            products.elements(frame, node, item, steps)
            if item.kind == "tuple" or is_product(item.kind)
            else (item,)
        )
    if is_sequence(base.kind):
        if rhs is not None or len(axes) != 1:
            frame.fail(
                node,
                "sequence_index",
                "one list index is supported; indexed mutation needs a separate protocol",
            )
        return sequences.get(frame, node, base, axes[0])
    if products.is_sequence(base):
        if rhs is not None:
            frame.fail(
                node,
                "sequence_mutation",
                "a mutable Python sequence cannot be eliminated when any alias writes it",
            )
        length = (
            len(product_members(base.kind))
            if is_product(base.kind)
            else len(base.value)
        )
        if (
            len(axes) == 1
            and axes[0].kind == "i64"
            and type(axes[0].exact) is int
            and -length <= axes[0].exact < length
        ):
            return products.get(frame, node, base, axes[0].exact, steps)
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
        if products.is_sequence(rhs):
            rhs = _literal_array(frame, node, rhs, steps, force=True)
        source_rank = rank(rhs.kind)
        if is_array(rhs.kind) and source_rank is None:
            frame.fail(node, "array_rank", "array assignment needs a resolved RHS rank")
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
