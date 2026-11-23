"""M2 contracts for scalar domains, exact integer ranges and binary64 arithmetic."""

import math
import operator
from .values import (
    Value,
    literal,
    numeric_facts as _numeric_facts,
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


def binary(frame, node, op, lhs, rhs):
    """Return a typed numeric expression, or None for the nonnumeric protocol."""
    if lhs.kind not in _NUMERIC or rhs.kind not in _NUMERIC:
        return None
    if op == "pow":
        return _power(frame, node, lhs, rhs)
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
        as_float(frame, node, v) if floating else v.value for v in (lhs, rhs)
    )
    return Value(
        result_type,
        dict(op="binary", type=result_type, operator=symbol, left=left, right=right),
        facts,
    )
