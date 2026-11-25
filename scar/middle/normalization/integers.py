"""M2 mathematical signed integers and their explicit scalar conversions."""

import math
import operator

from .values import (
    Value,
    literal as scalar_literal,
    numeric_facts,
    source_site,
    wire,
)

# This follows the binary64 wire format, not the analysis host's float limits.
_FLOAT_MAX_INTEGER = ((1 << 53) - 1) << 971
_BINARY = {
    "add": ("add", operator.add),
    "sub": ("sub", operator.sub),
    "mult": ("mul", operator.mul),
    "mul": ("mul", operator.mul),
    "floordiv": ("floordiv", operator.floordiv),
    "mod": ("mod", operator.mod),
    "eq": ("eq", operator.eq),
    "noteq": ("ne", operator.ne),
    "ne": ("ne", operator.ne),
    "lt": ("lt", operator.lt),
    "lte": ("le", operator.le),
    "le": ("le", operator.le),
    "gt": ("gt", operator.gt),
    "gte": ("ge", operator.ge),
    "ge": ("ge", operator.ge),
}
_COMPARISONS = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})


def apply(
    frame,
    node,
    operation,
    args,
    result_kind="integer",
    *,
    facts=None,
    steps=None,
    exception_messages=None,
    source_interpreter=None,
    may_raise=None,
    **contract,
):
    """Construct one logical interface; target libraries are chosen downstream."""
    frame.project.require_operator("integer", operation, semantics_version=1)
    value = dict(
        op="integer_op",
        operator_family="integer",
        semantics_version=1,
        operation=operation,
        args=[arg.value for arg in args],
        type=wire(result_kind),
        effects=(["VAL", "MEM"] if result_kind in {"integer", "text"} else ["VAL"]),
        reads=[],
        writes=[],
        may_raise=(operation not in _COMPARISONS if may_raise is None else may_raise),
        **source_site(node),
    )
    if exception_messages is not None:
        value["exception_messages"] = exception_messages
    if source_interpreter is not None:
        value["source_interpreter"] = source_interpreter
    if value.keys() & contract.keys():
        frame.fail(
            node,
            "integer_contract",
            "an integer contract cannot replace its operation fields",
        )
    value.update(contract)
    if steps is not None:
        return frame.effect(value, result_kind, steps, facts=facts)
    return Value(result_kind, value, facts or {})


def _source_contract(frame, node):
    from .scalars import _source_interpreter

    captured, version = _source_interpreter(frame, node)
    if not (3, 10) <= version < (3, 15):
        frame.fail(
            node,
            "integer_source_version",
            "integer conversions and division errors need a sealed CPython 3.10–3.14 profile",
        )
    return captured, version


def literal(frame, node, value, *, steps=None):
    if type(value) is not int:
        frame.fail(node, "integer_literal", "an integer literal requires an exact int")
    try:
        spelling = str(value)
    except ValueError:
        frame.fail(
            node,
            "integer_literal_size",
            "the decimal spelling exceeds this source-capture integer limit",
        )
    return apply(
        frame,
        node,
        "decimal_literal",
        [scalar_literal(spelling)],
        facts=numeric_facts(value),
        steps=steps,
    )


def promote(frame, node, value, *, steps=None):
    if value.kind == "integer":
        if steps is not None and value.value["op"] not in {"literal", "var"}:
            return frame.effect(value.value, value.kind, steps, facts=value.facts)
        return value
    if value.kind != "i64":
        frame.fail(node, "integer_operand", "integer arithmetic requires an integer")
    facts = dict(value.facts)
    facts.setdefault("range", value.bounds)
    return apply(frame, node, "from_i64", [value], facts=facts, steps=steps)


def from_f64(frame, node, value, *, steps=None):
    if wire(value.kind) != "f64":
        frame.fail(
            node, "integer_operand", "float-to-integer requires a binary64 operand"
        )
    captured, _ = _source_contract(frame, node)
    facts = {"range": (-_FLOAT_MAX_INTEGER, _FLOAT_MAX_INTEGER)}
    exact = value.exact
    if type(exact) is float and math.isfinite(exact):
        facts = numeric_facts(math.trunc(exact))
    elif (bounds := value.facts.get("range")) is not None and all(
        type(endpoint) in {int, float} and math.isfinite(endpoint)
        for endpoint in bounds
    ):
        facts["range"] = tuple(map(math.trunc, bounds))
    return apply(
        frame,
        node,
        "from_f64",
        [value],
        facts=facts,
        steps=steps,
        source_interpreter=captured,
        exception_messages={
            "value_error": "cannot convert float NaN to integer",
            "overflow_error": "cannot convert float infinity to integer",
        },
    )


def to_binary64(frame, node, value, *, steps=None):
    if value.kind != "integer":
        frame.fail(
            node, "integer_operand", "integer-to-float requires an integer operand"
        )
    captured, _ = _source_contract(frame, node)
    facts = {}
    if type(value.exact) is int:
        try:
            facts = numeric_facts(float(value.exact))
        except OverflowError:
            pass
    return apply(
        frame,
        node,
        "to_binary64",
        [value],
        "f64",
        facts=facts,
        steps=steps,
        source_interpreter=captured,
        exception_messages={"overflow_error": "int too large to convert to float"},
    )


def _ranges(operation, left, right):
    a, b = left.bounds, right.bounds
    if operation == "mod" and b is not None:
        # On the normal edge, the remainder shares the nonzero divisor's sign
        # and has smaller magnitude, including when the interval contains zero.
        return {"range": (min(0, b[0] + 1), max(0, b[1] - 1))}
    if operation == "floordiv" and a is not None:
        if b is None or b[0] <= 0 <= b[1]:
            # A nonzero mathematical integer divisor has magnitude >= 1.
            magnitude = max(map(abs, a))
            return {"range": (-magnitude, magnitude)}
        corners = [x // y for x in a for y in b]
        return {"range": (min(corners), max(corners))}
    if operation in {"add", "sub", "mul"} and a is not None and b is not None:
        evaluate = {"add": operator.add, "sub": operator.sub, "mul": operator.mul}[
            operation
        ]
        corners = [evaluate(x, y) for x in a for y in b]
        return {"range": (min(corners), max(corners))}
    return {}


def binary(frame, node, op, lhs, rhs, steps=None):
    if lhs.kind not in {"integer", "i64"} or rhs.kind not in {"integer", "i64"}:
        frame.fail(node, "integer_operand", "mixed integer arithmetic is not closed")
    if op not in _BINARY:
        frame.fail(node, "integer_operator", f"unclosed mathematical integer {op}")
    operation, evaluate = _BINARY[op]
    if steps is None:
        frame.fail(
            node,
            "integer_evaluation_order",
            "integer operations require an explicit ordered evaluation site",
        )
    args = [promote(frame, node, value, steps=steps) for value in (lhs, rhs)]
    facts = _ranges(operation, *args)
    if type(lhs.exact) is int and type(rhs.exact) is int:
        if operation not in {"floordiv", "mod"} or rhs.exact:
            facts = numeric_facts(evaluate(lhs.exact, rhs.exact))
    metadata = {}
    if operation in {"floordiv", "mod"}:
        captured, version = _source_contract(frame, node)
        message = (
            "division by zero"
            if version >= (3, 14)
            else (
                "integer modulo by zero"
                if operation == "mod"
                else "integer division or modulo by zero"
            )
        )
        metadata = dict(
            source_interpreter=captured,
            exception_messages={"zero_division_error": message},
        )
    return apply(
        frame,
        node,
        operation,
        args,
        "bool" if operation in _COMPARISONS else "integer",
        facts=facts,
        steps=steps,
        **metadata,
    )


def unary(frame, node, value, operation, *, steps=None):
    if operation not in {"negative", "absolute"} or value.kind != "integer":
        frame.fail(node, "integer_operator", "unclosed integer unary operation")
    facts = {}
    if type(value.exact) is int:
        facts = numeric_facts(
            -value.exact if operation == "negative" else abs(value.exact)
        )
    elif value.bounds is not None:
        low, high = value.bounds
        facts["range"] = (
            (-high, -low)
            if operation == "negative"
            else (
                0 if low <= 0 <= high else min(abs(low), abs(high)),
                max(abs(low), abs(high)),
            )
        )
    return apply(frame, node, operation, [value], facts=facts, steps=steps)
