"""M2 contracts for exact-width scalar arithmetic, dense arrays and their views."""

import math
import operator
import re

from .values import (
    SCALARS,
    Value,
    _MIN,
    _MAX,
    _UNKNOWN,
    array_kind,
    from_wire,
    literal,
    numeric_facts as _numeric_facts,
    rank,
    runtime,
    wire,
)

_NUMERIC = {"i64", "f64"}
_PROVIDED = {"array", "zeros", "float64"}
_OPERATORS = {
    "add": ("add", operator.add),
    "sub": ("sub", operator.sub),
    "mult": ("mul", operator.mul),
    "div": ("div", operator.truediv),
    "eq": ("eq", operator.eq),
    "noteq": ("ne", operator.ne),
    "lt": ("lt", operator.lt),
    "lte": ("le", operator.le),
    "gt": ("gt", operator.gt),
    "gte": ("ge", operator.ge),
}


def _output(frame, node, kind, steps, *, facts=None, discard=False, **fields):
    frame.project.array_ops.add(node)
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


def _as_float(frame, node, item):
    if item.kind == "f64":
        return item.value
    if item.kind == "i64":
        return dict(op="cast", type="f64", value=item.value)
    frame.fail(node, "array_element", f"cannot convert {item.kind} to binary64")


def scalar_unary(frame, node, item, steps, discard):
    op = {"uadd": "pos", "usub": "neg"}.get(node["operator"])
    n = rank(item.kind)
    if op is None or n is None and item.kind not in _NUMERIC:
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
                facts=item.facts,
                discard=discard,
                operator=op,
                value=item.value,
                effects=["MEM", "VAL"],
            )
        item = _output(
            frame,
            "array_load",
            "f64",
            steps,
            base=item.value,
            indices=[],
            effects=["STATE"],
        )
    exact = item.exact
    if item.kind == "i64" and op == "neg" and item.bounds[0] == _MIN:
        frame.fail(
            node,
            "integer_range",
            "integer negation needs a representable result on all paths",
        )
    facts = (
        {} if exact is _UNKNOWN else _numeric_facts(+exact if op == "pos" else -exact)
    )
    if item.kind == "i64":
        low, high = item.bounds
        facts["range"] = (low, high) if op == "pos" else (-high, -low)
    return Value(
        item.kind,
        dict(op="unary", type=item.kind, operator=op, value=item.value),
        facts,
    )


def scalar_binary(frame, node, op, lhs, rhs):
    """Return a typed numeric expression, or None for the nonnumeric protocol."""
    if lhs.kind not in _NUMERIC or rhs.kind not in _NUMERIC:
        return None
    if op not in _OPERATORS:
        frame.fail(node, "arithmetic_contract", f"unclosed numeric operator {op}")
    symbol, evaluate = _OPERATORS[op]
    a, b = lhs.exact, rhs.exact
    known = a is not _UNKNOWN and b is not _UNKNOWN
    compare = symbol in {"eq", "ne", "lt", "le", "gt", "ge"}
    floating = "f64" in (lhs.kind, rhs.kind) or symbol == "div"
    nonzero = (
        (b is not _UNKNOWN and b != 0)
        or rhs.kind == "i64"
        and not rhs.bounds[0] <= 0 <= rhs.bounds[1]
    )
    if symbol == "div" and not nonzero:
        frame.fail(
            node, "division_contract", "the divisor must be proven nonzero at this use"
        )
    result_type = "bool" if compare else ("f64" if floating else "i64")
    facts = _numeric_facts(evaluate(a, b)) if known else {}
    if result_type == "i64":
        corners = [evaluate(x, y) for x in lhs.bounds for y in rhs.bounds]
        low, high = min(corners), max(corners)
        if not _MIN <= low <= high <= _MAX:
            frame.fail(
                node,
                "integer_range",
                "integer operation needs a proven signed 64-bit result range",
            )
        facts["range"] = (low, high)
    if floating and (compare or symbol == "div" and lhs.kind == rhs.kind == "i64"):
        if any(
            v.kind == "i64" and max(map(abs, v.bounds)) > 1 << 53 for v in (lhs, rhs)
        ):
            if known:
                return literal(evaluate(a, b))
            frame.fail(
                node,
                "integer_precision",
                "integer/floating comparison or true division requires exact integer conversion",
            )
    left, right = (
        _as_float(frame, node, v) if floating else v.value for v in (lhs, rhs)
    )
    return Value(
        result_type,
        dict(op="binary", type=result_type, operator=symbol, left=left, right=right),
        facts,
    )


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
                item = literal(item.exact)
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
        facts={"shape": shape},
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
            facts=source.facts,
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
            facts=source.facts,
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
            "f64",
            steps,
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
        facts={"shape": result},
        base=base.value,
        indices=indices,
        effects=["STATE", "MEM"],
    )
