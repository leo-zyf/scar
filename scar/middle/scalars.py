"""M2 contracts for scalar domains, exact integer ranges and binary64 arithmetic."""

import math
import operator
from .values import (
    Value,
    literal,
    numeric_facts as _numeric_facts,
    wire,
    _MIN,
    _MAX,
    _UNKNOWN,
)

_NUMERIC = {"i64", "f64"}
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


def _binary64_bounds(item):
    """Enclose finite values after binary64 conversion. An unknown float has no bounds."""
    if item.exact is not _UNKNOWN:
        interval = (item.exact, item.exact)
    elif item.kind in {"bool", "numpy_bool"}:
        interval = (0, 1)
    elif item.kind == "i64":
        interval = item.bounds
    else:
        interval = item.facts.get("range")
    if interval is None:
        return None
    low, high = map(float, interval)
    if item.kind == "i64":
        # A nonrepresentable integer can land on either adjacent binary64
        # value under a native integral cast. Retain both possibilities.
        if low > interval[0]:
            low = math.nextafter(low, -math.inf)
        if high < interval[1]:
            high = math.nextafter(high, math.inf)
    if math.isfinite(low) and math.isfinite(high) and low <= high:
        return low, high
    return None


def _absolute(frame, node, item):
    if item.kind == "numpy_bool":
        return item
    exact = item.exact
    if item.kind == "bool":
        facts = {"range": (0, 1)} if exact is _UNKNOWN else _numeric_facts(int(exact))
        return Value("i64", dict(op="cast", type="i64", value=item.value), facts)
    if item.kind == "i64":
        interval = item.bounds if exact is _UNKNOWN else (exact, exact)
        if interval[0] <= _MIN:
            frame.fail(
                node,
                "integer_range",
                "abs() requires a signed 64-bit operand above INT64_MIN",
            )
    elif item.kind in {"f64", "numpy_f64"}:
        interval = _binary64_bounds(item)
    else:
        frame.fail(
            node,
            "absolute_domain",
            f"abs() requires a scalar magnitude contract for {item.kind}",
        )

    facts = {} if exact is _UNKNOWN else _numeric_facts(abs(exact))
    if interval is not None:
        low, high = interval
        lower = min(abs(low), abs(high))
        if low <= 0 <= high:
            lower = 0 if item.kind == "i64" else 0.0
        facts["range"] = (lower, max(abs(low), abs(high)))
    return Value(
        item.kind,
        dict(op="intrinsic", name="abs", type=wire(item.kind), args=[item.value]),
        facts,
    )


def construct(frame, node, name, args, keywords):
    """Close a proven numeric builtin using its already evaluated arguments."""
    if name == "abs":
        if len(args) != 1 or keywords:
            frame.fail(
                node,
                "absolute_arguments",
                "abs() requires exactly one positional argument",
            )
        return _absolute(frame, node, args[0])
    if name not in {"int", "float"}:
        return None
    if keywords or len(args) > 1:
        frame.fail(
            node,
            "scalar_constructor_arguments",
            f"{name}() accepts zero or one positional numeric argument in this contract"
            + (
                "; parsing an integer with a base requires a text contract"
                if name == "int"
                else ""
            ),
        )
    if not args:
        return literal(0 if name == "int" else 0.0)
    operand = args[0]
    if operand.kind not in {"bool", "numpy_bool", "i64", "f64", "numpy_f64"}:
        detail = (
            "text parsing and its exceptions require a parsing contract"
            if operand.kind == "text"
            else f"the conversion protocol of {operand.kind} is unclosed"
        )
        frame.fail(node, "scalar_conversion_domain", f"{name}(): {detail}")

    if name == "int":
        if operand.kind == "i64":
            return operand
        bounds = _binary64_bounds(operand)
        if bounds is None:
            known = (
                operand.value["value"]
                if operand.value.get("op") == "literal"
                else operand.exact
            )
            if type(known) is float and not math.isfinite(known):
                error = "ValueError" if math.isnan(known) else "OverflowError"
                frame.fail(
                    node, "integer_conversion_nonfinite", f"int({known}) raises {error}"
                )
            frame.fail(
                node,
                "integer_conversion_range",
                "int() needs a finite binary64 interval whose truncation fits signed 64 bits",
            )
        # Comparing the mathematical truncations also handles +2**63: the
        # largest i64 rounds up to this *out-of-range* value in binary64.
        low, high = map(math.trunc, bounds)
        if not _MIN <= low <= high <= _MAX:
            frame.fail(
                node,
                "integer_conversion_range",
                f"int() can produce [{low}, {high}], outside signed 64 bits",
            )
        facts = _numeric_facts(low) if low == high else {"range": (low, high)}
        return Value("i64", dict(op="cast", type="i64", value=operand.value), facts)

    if operand.kind in {"f64", "numpy_f64"}:
        return Value("f64", operand.value, operand.facts)
    if operand.kind == "i64" and max(map(abs, operand.bounds)) > 1 << 53:
        exact = operand.exact
        if exact is not _UNKNOWN and operand.value.get("op") in {"literal", "var"}:
            return literal(float(exact))
        frame.fail(
            node,
            "integer_precision",
            "float() of an integer outside the exact binary64 interval needs a proven rounding contract",
        )
    facts = (
        _numeric_facts(float(operand.exact))
        if operand.exact is not _UNKNOWN
        else {"range": _binary64_bounds(operand)}
    )
    return Value("f64", dict(op="cast", type="f64", value=operand.value), facts)


def as_float(frame, node, item):
    if item.kind in {"f64", "numpy_f64"}:
        return item.value
    if item.kind == "i64":
        return dict(op="cast", type="f64", value=item.value)
    frame.fail(node, "array_element", f"cannot convert {item.kind} to binary64")


def unary(frame, node, item):
    op = {"uadd": "pos", "usub": "neg"}.get(node["operator"])
    if op is None or item.kind not in _NUMERIC:
        frame.fail(
            node, "unary_contract", f"unclosed {node['operator']} on {item.kind}"
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
    elif exact is _UNKNOWN and (bounds := _binary64_bounds(item)) is not None:
        low, high = bounds
        facts["range"] = bounds if op == "pos" else (-high, -low)
    return Value(
        item.kind,
        dict(op="unary", type=item.kind, operator=op, value=item.value),
        facts,
    )


def _integer_power_range(frame, node, base, exponent):
    low, high = base.bounds
    first, last = exponent.bounds
    if last > 63 and max(abs(low), abs(high)) >= 2:
        frame.fail(node, "integer_range", "integer power can exceed signed 64 bits")
    # For |base| >= 1 the magnitude increases with the exponent. The first
    # and last exponent of each parity suffice; zero has its own 0**0 case.
    powers = {first, last}
    if first < last:
        powers.update((first + 1, last - 1))
    bases = {low, high}
    if low <= 0 <= high:
        bases.add(0)
    extrema = [a**b for a in bases for b in powers]
    lower, upper = min(extrema), max(extrema)
    if lower < _MIN or upper > _MAX:
        frame.fail(node, "integer_range", "integer power can exceed signed 64 bits")
    return lower, upper


def _power(frame, node, base, exponent):
    a, b = base.exact, exponent.exact
    snapshot = all(v.value.get("op") in {"literal", "var"} for v in (base, exponent))
    known = a is not _UNKNOWN and b is not _UNKNOWN
    if base.kind == exponent.kind == "i64":
        first, last = exponent.bounds
        if first >= 0:
            bounds = _integer_power_range(frame, node, base, exponent)
            if known and snapshot:
                return literal(a**b)
            facts = dict(range=bounds)
            if bounds[0] == bounds[1]:
                facts["exact"] = bounds[0]
            return Value(
                "i64",
                dict(
                    op="intrinsic",
                    name="powi",
                    type="i64",
                    args=[base.value, exponent.value],
                ),
                facts,
            )
        if last >= 0:
            frame.fail(
                node,
                "power_type",
                "integer exponent can select either an integer or a float result",
            )

    nonnegative = (a is not _UNKNOWN and a >= 0) or (
        "range" in base.facts and base.facts["range"][0] >= 0
    )
    integral_exponent = exponent.kind == "i64" or (
        b is not _UNKNOWN and math.isfinite(b) and float(b).is_integer()
    )
    if not (nonnegative or integral_exponent):
        frame.fail(
            node,
            "power_domain",
            "a negative base with a fractional exponent requires a complex result",
        )

    # Finite binary64 constants can use this host interpreter's libm. Leave
    # exceptions at the original expression, including in a discarded result.
    # A joined f64 zero can carry either sign; use the actual runtime operand.
    signed_zero = base.kind == "f64" and a == 0.0 and base.value.get("op") != "literal"
    if known and snapshot and not signed_zero:
        try:
            result = a**b
        except (OverflowError, ZeroDivisionError):
            pass
        else:
            if math.isfinite(result):
                return literal(result)

    def floating(value):
        if value.kind == "i64" and max(map(abs, value.bounds)) > 1 << 53:
            if value.exact is not _UNKNOWN and value.value.get("op") in {
                "literal",
                "var",
            }:
                return literal(float(value.exact)).value
            frame.fail(
                node,
                "integer_precision",
                "power operand needs a proven binary64 integer conversion",
            )
        return as_float(frame, node, value)

    return Value(
        "f64",
        dict(
            op="scalar_power",
            base=floating(base),
            exponent=floating(exponent),
            type="f64",
            domain="real",
            effects=["VAL"],
            may_raise=True,
        ),
    )


def _integer_division(frame, node, op, lhs, rhs, steps):
    if lhs.kind != "i64" or rhs.kind != "i64":
        frame.fail(
            node,
            "integer_division_domain",
            f"{op} requires two signed integers in this contract",
        )
    a, b = (v.bounds if v.exact is _UNKNOWN else (v.exact, v.exact) for v in (lhs, rhs))
    if b[0] <= 0 <= b[1]:
        frame.fail(node, "division_contract", f"{op} needs a proven nonzero divisor")
    positive = b[0] > 0
    if op == "floordiv":
        # With a fixed divisor sign, extrema of floor(x/y) occur at corners.
        corners = [x // y for x in a for y in b]
        lo, hi = min(corners), max(corners)
        if hi > _MAX:
            frame.fail(node, "integer_range", "INT64_MIN // -1 exceeds signed 64 bits")
    elif a[0] == a[1] and b[0] == b[1]:
        lo = hi = a[0] % b[0]
    elif positive:
        lo, hi = 0, b[1] - 1
        if a[0] >= 0:
            hi = min(hi, a[1])
    else:
        lo, hi = b[0] + 1, 0
        if a[1] <= 0:
            lo = max(lo, a[0])

    def capture(v):
        if v.value["op"] in {"literal", "var"}:
            return v.value
        if steps is None:
            frame.fail(
                node,
                "integer_evaluation_order",
                f"{op} needs an operand evaluation site",
            )
        return frame.effect(v.value, v.kind, steps, facts=v.facts).value

    left, right = capture(lhs), capture(rhs)
    if lo == hi:
        return literal(lo)

    def expr(operator, left, right, kind="i64"):
        return dict(op="binary", operator=operator, left=left, right=right, type=kind)

    same_sign = a[0] >= 0 if positive else a[1] <= 0
    zero = literal(0).value
    if op == "mod":
        result = (
            expr("mod", left, right)
            if same_sign
            else dict(op="intrinsic", name="modulo", type="i64", args=[left, right])
        )
        if a[0] == _MIN and b[1] == -1:
            # Both native % and modulo's internal % would overflow on this pair.
            result = {
                "op": "select",
                "type": "i64",
                "condition": expr("eq", right, literal(-1).value, "bool"),
                "then": zero,
                "else": result,
            }
    else:
        result = expr("div", left, right)
        if not same_sign:
            # Truncating quotient minus one iff signs differ and remainder is nonzero.
            adjust = expr("ne", expr("mod", left, right), zero, "bool")
            if a[0] < 0 < a[1]:
                sign = expr("lt" if positive else "gt", left, zero, "bool")
                adjust = expr("land", sign, adjust, "bool")
            result = expr("sub", result, dict(op="cast", type="i64", value=adjust))
    return Value("i64", result, {"range": (lo, hi)})


def binary(frame, node, op, lhs, rhs, steps=None):
    """Return a typed numeric expression, or None for the nonnumeric protocol."""
    if lhs.kind not in _NUMERIC or rhs.kind not in _NUMERIC:
        return None
    if op in {"floordiv", "mod"}:
        return _integer_division(frame, node, op, lhs, rhs, steps)
    if op == "pow":
        return _power(frame, node, lhs, rhs)
    if op not in _OPERATORS:
        frame.fail(node, "arithmetic_contract", f"unclosed numeric operator {op}")
    symbol, evaluate = _OPERATORS[op]
    a, b = lhs.exact, rhs.exact
    known = a is not _UNKNOWN and b is not _UNKNOWN
    compare = symbol in {"eq", "ne", "lt", "le", "gt", "ge"}
    floating = "f64" in (lhs.kind, rhs.kind) or symbol == "div"
    right_bounds = _binary64_bounds(rhs) if symbol == "div" else None
    nonzero = (b is not _UNKNOWN and b != 0) or (
        right_bounds is not None and not right_bounds[0] <= 0 <= right_bounds[1]
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
    elif result_type == "f64" and not known:
        left_bounds = _binary64_bounds(lhs)
        right_bounds = right_bounds or _binary64_bounds(rhs)
        if left_bounds is not None and right_bounds is not None:
            # These four operators attain their extrema at rectangle corners
            # (division has already excluded intervals containing zero).
            corners = [evaluate(x, y) for x in left_bounds for y in right_bounds]
            if all(map(math.isfinite, corners)):
                facts["range"] = (min(corners), max(corners))
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
        as_float(frame, node, v) if floating else v.value for v in (lhs, rhs)
    )
    return Value(
        result_type,
        dict(op="binary", type=result_type, operator=symbol, left=left, right=right),
        facts,
    )
