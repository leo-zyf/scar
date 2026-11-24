"""M2 contracts for scalar domains, exact integer ranges and binary64 arithmetic."""

import math
import operator
import re
from .values import (
    Value,
    literal,
    numeric_facts as _numeric_facts,
    wire,
    _MIN,
    _MAX,
    _UNKNOWN,
    choice,
    protocols,
    source_site,
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

# (native intrinsic, finite-input overflow possible, total on IEEE binary64).
# Integer-valued rounding, multi-value functions and custom numeric protocols
# need their own result/conversion contracts and do not enter this family.
MATH_FUNCTIONS = {
    **{
        name: (name, False, False)
        for name in (
            "acos",
            "acosh",
            "asin",
            "atanh",
            "cos",
            "sin",
            "sqrt",
            "tan",
            "log",
            "log1p",
            "log2",
            "log10",
        )
    },
    **{name: (name, True, False) for name in ("cosh", "sinh", "exp", "exp2", "expm1")},
    **{
        name: (name, False, True)
        for name in ("asinh", "atan", "tanh", "cbrt", "erf", "erfc")
    },
    "fabs": ("abs", False, True),
}
MATH_CLASSIFICATIONS = frozenset({"isfinite", "isinf", "isnan"})
_MATH_DOMAIN_PREFIXES = {
    "acos": "expected a number in range from -1 up to 1, got ",
    "asin": "expected a number in range from -1 up to 1, got ",
    "acosh": "expected argument value not less than 1, got ",
    "atanh": "expected a number between -1 and 1, got ",
    "cos": "expected a finite input, got ",
    "sin": "expected a finite input, got ",
    "tan": "expected a finite input, got ",
    "sqrt": "expected a nonnegative input, got ",
    "log1p": "expected argument value > -1, got ",
}


def _finite_math_domain(name, operand):
    """Discharge error edges from an already-proven finite input interval."""
    bounds = _binary64_bounds(operand)
    if bounds is None:
        return False
    low, high = bounds
    if name == "sqrt":
        return low >= 0.0
    if name in {"log", "log2", "log10"}:
        return low > 0.0
    if name == "log1p":
        return low > -1.0
    if name in {"acos", "asin"}:
        return -1.0 <= low <= high <= 1.0
    if name == "acosh":
        return low >= 1.0
    if name == "atanh":
        return -1.0 < low <= high < 1.0
    return name in {"sin", "cos", "tan"}


def math_call(frame, node, name, args, keywords, steps, discard=False):
    """Close scalar libm calls using ordinary values and explicit error regions.

    The result classification rules follow CPython's scalar math contract.
    This pass neither implements libm nor hides source errors in a C++ wrapper.
    """
    if keywords or len(args) not in ({1, 2} if name == "log" else {1}):
        frame.fail(
            node, "math_arguments", f"{name} needs its positional scalar signature"
        )
    captured = node.get("source_runtime", {}).get(
        "interpreter"
    ) or frame.project.text_profiles.get(frame.scope.get("text_profile"), {}).get(
        "interpreter", ""
    )
    match = re.match(r"cpython (\d+)\.(\d+)\.", captured)
    version = tuple(map(int, match.groups())) if match else ()
    if not (3, 10) <= version < (3, 15):
        frame.fail(
            node,
            "math_source_version",
            "scalar math needs a registered source-library contract",
        )
    if name in {"cbrt", "exp2"} and version < (3, 11):
        frame.fail(
            node, "math_source_version", f"{name} is absent from this source library"
        )
    source_args = args
    args = [construct(frame, node, "float", [value], {}) for value in args]

    def intrinsic(function, value, kind):
        return dict(
            op="intrinsic",
            name=function,
            type=kind,
            args=[value],
            effects=["VAL"],
            may_raise=False,
        )

    def combine(operator, left, right):
        return dict(op="binary", type="bool", operator=operator, left=left, right=right)

    def reject(condition, category, message, operand=None):
        # Use the shared exception protocol at this same layer. The error
        # object is allocated only along the exceptional edge.
        from . import exceptions

        body = []
        message = literal(message)
        if operand is not None:
            from . import text

            rendered = text.render(frame, node, operand, "repr", literal(""), body)
            message = text._concat(frame, [message, rendered], body)
        error = exceptions.construct(
            frame, node, exceptions.builtin(category), [message], {}, body
        )
        body.append(dict(op="throw", value=error.value))
        steps.append(
            dict(
                op="if",
                condition=condition,
                then=body,
                **{"else": []},
                **source_site(node),
            )
        )

    def domain(function, operand, source):
        if version < (3, 14):
            return "math domain error", None
        if function in {"log", "log2", "log10"}:
            if source.kind in {"i64", "bool"}:
                return "expected a positive input", None
            if source.kind == "boolean_choice":
                frame.fail(
                    node,
                    "math_error_protocol",
                    "logarithm error presentation observes the source boolean protocol",
                )
            return "expected a positive input, got ", operand
        prefix = _MATH_DOMAIN_PREFIXES.get(function)
        return (prefix, operand) if prefix else ("math domain error", None)

    def evaluate(function, operand, source):
        if function in MATH_CLASSIFICATIONS:
            value = intrinsic(function, operand.value, "bool")
            facts = (
                _numeric_facts(getattr(math, function)(operand.exact))
                if operand.exact is not _UNKNOWN
                else {}
            )
            return frame.effect(value, "bool", steps, facts=facts)
        native, can_overflow, total = MATH_FUNCTIONS[function]
        facts = {}
        if operand.exact is not _UNKNOWN:
            try:
                facts = _numeric_facts(getattr(math, function)(operand.exact))
            except (ValueError, OverflowError):
                # Keep the operation and exceptional continuation at runtime.
                pass
        value = dict(**intrinsic(native, operand.value, "f64"), **source_site(node))
        result = frame.effect(value, "f64", steps, facts=facts)
        if (
            not total
            and "exact" not in facts
            and not _finite_math_domain(function, operand)
        ):
            invalid = combine(
                "land",
                intrinsic("isnan", result.value, "bool"),
                dict(
                    op="unary",
                    operator="not",
                    type="bool",
                    value=intrinsic("isnan", operand.value, "bool"),
                ),
            )
            message, displayed = domain(function, operand, source)
            reject(invalid, "ValueError", message, displayed)
            infinite = combine(
                "land",
                intrinsic("isinf", result.value, "bool"),
                intrinsic("isfinite", operand.value, "bool"),
            )
            reject(
                infinite,
                "OverflowError" if can_overflow else "ValueError",
                "math range error" if can_overflow else message,
                None if can_overflow else displayed,
            )
        return result

    result = evaluate(name, args[0], source_args[0])
    if len(args) == 2:
        base = evaluate("log", args[1], source_args[1])
        if base.exact is _UNKNOWN or base.exact == 0.0:
            reject(
                combine("eq", base.value, literal(0.0).value),
                "ZeroDivisionError",
                "division by zero" if version >= (3, 14) else "float division by zero",
            )
        facts = {}
        if (
            result.exact is not _UNKNOWN
            and base.exact is not _UNKNOWN
            and base.exact != 0.0
        ):
            facts = _numeric_facts(result.exact / base.exact)
        result = frame.effect(
            dict(
                op="binary",
                operator="div",
                type="f64",
                left=result.value,
                right=base.value,
            ),
            "f64",
            steps,
            facts=facts,
        )
    frame.project.decisions.append(
        dict(
            code="closed_scalar_math",
            symbol=frame.name,
            function=name,
            input_types=[value.kind for value in source_args],
            source_interpreter=captured,
            result=result.kind,
            **source_site(node),
            proof="verified scalar library identity; binary64 argument conversion and explicit domain/range error continuations; no source-language execution",
        )
    )
    return literal(None) if discard else result


def _binary64_bounds(item):
    """Enclose finite values after binary64 conversion. An unknown float has no bounds."""
    if item.exact is not _UNKNOWN:
        interval = (item.exact, item.exact)
    elif item.kind in {"bool", "numpy_bool", "boolean_choice"}:
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
    elif item.kind in {"f64", "numpy_f64", "binary64_choice"}:
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
    if operand.kind not in {
        "bool",
        "numpy_bool",
        "boolean_choice",
        "i64",
        "f64",
        "numpy_f64",
        "binary64_choice",
    }:
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

    if operand.kind in {"f64", "numpy_f64", "binary64_choice"}:
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
    if item.kind in {"f64", "numpy_f64", "binary64_choice"}:
        return item.value
    if item.kind == "i64":
        return dict(op="cast", type="f64", value=item.value)
    frame.fail(node, "array_element", f"cannot convert {item.kind} to binary64")


def comparison(frame, node, op, lhs, rhs):
    """Compare shared scalar representations; keep the result's protocol set."""
    possible = [protocols(v.kind) for v in (lhs, rhs)]
    if op not in {"eq", "noteq", "lt", "lte", "gt", "gte"} or any(
        not kinds <= {"bool", "numpy_bool", "i64", "f64", "numpy_f64"}
        for kinds in possible
    ):
        frame.fail(
            node,
            "scalar_observation",
            f"{op} has no common behavior contract for {lhs.kind} and {rhs.kind}",
        )
    if any("numpy_f64" in kinds for kinds in possible) and any(
        item.kind == "i64" and max(map(abs, item.bounds)) > 1 << 53
        for item in (lhs, rhs)
    ):
        # NumPy promotes this integer before comparing; Python's mixed
        # comparison can instead preserve its exact mathematical value. Even
        # two exact facts cannot be folded with the Python rule for both.
        frame.fail(
            node,
            "integer_precision",
            "a possible array-scalar comparison requires exact integer conversion",
        )
    operands = []
    for item in (lhs, rhs):
        spec = wire(item.kind)
        if spec == "bool":
            item = Value(
                "i64",
                dict(op="cast", type="i64", value=item.value),
                (
                    {"range": (0, 1)}
                    if item.exact is _UNKNOWN
                    else _numeric_facts(int(item.exact))
                ),
            )
        else:
            item = Value(spec, item.value, item.facts)
        operands.append(item)
    result = binary(frame, node, op, *operands)
    result_kinds = {
        "numpy_bool" if a.startswith("numpy_") or b.startswith("numpy_") else "bool"
        for a in possible[0]
        for b in possible[1]
    }
    return Value(choice(result_kinds), result.value, result.facts)


def extreme(frame, node, name, args, keywords, steps):
    """Select an existing argument in source order, without numeric promotion."""
    if keywords or len(args) < 2:
        frame.fail(
            node,
            "extreme_arguments",
            f"{name} needs two or more scalar arguments; iterable/key/default protocols are not closed",
        )
    winner = args[0]
    for item in args[1:]:
        check = comparison(frame, node, "gt" if name == "max" else "lt", item, winner)
        if type(check.exact) is bool:
            if check.exact:
                winner = item
            continue
        kind = choice((winner.kind, item.kind))
        if kind is None:
            frame.fail(
                node,
                "extreme_representation",
                f"{name} selection needs a common lossless representation for {winner.kind} and {item.kind}",
            )
        facts = {}
        if winner.kind == item.kind == "i64":
            facts["range"] = (
                min(winner.bounds[0], item.bounds[0]),
                max(winner.bounds[1], item.bounds[1]),
            )
        winner = frame.effect(
            dict(
                op="select",
                type=wire(kind),
                condition=check.value,
                then=item.value,
                **{"else": winner.value},
            ),
            kind,
            steps,
            facts=facts,
        )
    frame.project.decisions.append(
        dict(
            code="scalar_selection",
            symbol=frame.name,
            location=node.get("location"),
            operation=name,
            source_types=[v.kind for v in args],
            result=winner.kind,
            representation=wire(winner.kind),
            proof="arguments already evaluated in source order; strict comparison preserves the first winner on ties or unordered comparisons; select copies existing bits without promotion or a runtime source tag",
        )
    )
    return winner


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
