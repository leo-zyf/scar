"""M2 ordered operand evaluation and control-flow joins for source operators."""

from . import arrays, objects, scalars, values
from .values import Value, literal


def _capture(frame, item, steps):
    if values.runtime(item.kind) and item.value["op"] != "literal":
        return frame.effect(item.value, item.kind, steps, facts=item.facts)
    return item


def apply_binary(frame, node, operator, lhs, rhs, steps, discard=False):
    """Bind one operation after both operands have been evaluated in source order."""
    if any(values.is_product(item.kind) for item in (lhs, rhs)):
        if operator in {"is", "isnot"}:
            if all(values.is_product(item.kind) for item in (lhs, rhs)):
                return frame.effect(
                    dict(
                        op="product_identity",
                        type="bool",
                        left=frame.scalar(lhs, node),
                        right=frame.scalar(rhs, node),
                        operator="eq" if operator == "is" else "ne",
                        effects=["STATE"],
                    ),
                    "bool",
                    steps,
                    discard,
                )
            if any(item.kind == "tuple" for item in (lhs, rhs)):
                frame.fail(
                    node,
                    "product_identity_source",
                    "tuple construction identity is unresolved",
                )
            # The closed product protocol denotes an exact immutable tuple.
            return literal(operator == "isnot")
        frame.fail(
            node,
            "product_comparison",
            "tuple value operators require elementwise source comparison contracts",
        )
    if operator in {"is", "isnot"} and any(
        item.kind in {"numpy_bool", "numpy_f64", "binary64_choice", "boolean_choice"}
        or values.is_array(item.kind)
        for item in (lhs, rhs)
    ):
        frame.fail(
            node,
            "comparison_identity",
            "array storage and scalar bits do not establish source object identity",
        )
    identity = objects.compare(operator, lhs, rhs)
    if identity is not None:
        return identity
    if any(len(values.protocols(item.kind)) > 1 for item in (lhs, rhs)):
        return scalars.comparison(frame, node, operator, lhs, rhs)
    result = arrays.binary(frame, node, operator, lhs, rhs, steps, discard)
    if result is not None:
        return result
    result = scalars.binary(frame, node, operator, lhs, rhs, steps)
    if result is not None:
        if result.value["op"] == "scalar_power":
            return frame.effect(
                result.value, result.kind, steps, discard, facts=result.facts
            )
        return result

    left, right = frame.scalar(lhs, node), frame.scalar(rhs, node)
    equal = operator in {"eq", "noteq", "ne"}
    if lhs.kind == rhs.kind and (
        equal
        and lhs.kind in {"bool", "text", "unit"}
        or operator == "add"
        and lhs.kind == "text"
    ):
        if lhs.exact is not values._UNKNOWN and rhs.exact is not values._UNKNOWN:
            if left["op"] != "literal" or right["op"] != "literal":
                frame.project.decisions.append(
                    dict(
                        code="exact_primitive_fact",
                        symbol=frame.name,
                        location=node.get("location"),
                        operator=operator,
                        operand_types=[lhs.kind, rhs.kind],
                        proof="current reaching scalar facts determine the primitive result; operand effects remain at their evaluation sites",
                    )
                )
            a, b = lhs.exact, rhs.exact
            return literal((a == b if operator == "eq" else a != b) if equal else a + b)
        kind = "bool" if equal else "text"
        return Value(
            kind,
            dict(
                op="binary",
                operator="ne" if operator == "noteq" else operator,
                left=left,
                right=right,
                type=kind,
            ),
        )
    frame.fail(
        node,
        "arithmetic_contract",
        f"{operator} on {lhs.kind} and {rhs.kind} needs a closed operator and range",
    )


def _join_facts(items):
    first = items[0]
    facts = {
        key: fact
        for key, fact in first.facts.items()
        if all(
            key in item.facts and values.identical(item.facts[key], fact)
            for item in items[1:]
        )
    }
    if first.kind == "i64":
        facts["range"] = min(item.bounds[0] for item in items), max(
            item.bounds[1] for item in items
        )
    if "array_allocation" in facts and "literal_region" not in facts:
        # Different views may still share their owner's invalidation token.
        facts["literal_region"] = None
    return facts


def _guarded(
    frame, node, steps, operands, evaluate_operand, stop_truth, discard, truth_context
):
    """Emit each continuation once. Only the selected exit assigns the result."""
    body = steps
    states, exits = [], []
    depth = frame.dynamic
    try:
        for index, operand in enumerate(operands):
            last = index + 1 == len(operands)
            test_only = truth_context or (discard and not last)
            value = evaluate_operand(index, operand, body, test_only, discard and last)
            if last:
                break
            if test_only:
                check = value
            else:
                value = _capture(frame, value, body)
                check = frame.truth(value, node)
            if type(check.exact) is bool:
                if check.exact != stop_truth:
                    continue
                proof = dict(
                    rule="short_circuit",
                    guard_site=node.get("site"),
                    operand_index=index,
                    truth=check.exact,
                    operator=node.get("operator", "compare_chain"),
                )
                for skipped in operands[index + 1 :]:
                    frame.skip_expression(skipped, proof)
                break

            if not states:
                declaration_index = len(steps)
            state = frame.snapshot()
            states.append(state)
            stop, resume = [], []
            exits.append((stop, literal(stop_truth) if truth_context else value))
            then, otherwise = (stop, resume) if stop_truth else (resume, stop)
            body.append(
                dict(op="if", condition=check.value, then=then, **{"else": otherwise})
            )
            # Restore lexical bindings only. Writes in the continuation
            # monotonically invalidate shared allocation facts across paths.
            frame.restore(state)
            frame.dynamic += 1
            body = resume
    finally:
        frame.dynamic = depth
    if states:
        frame.merge([*states, frame.snapshot()])
    if discard:
        return literal(None)
    if not states:
        return value
    exits.append((body, value))
    items = [item for _, item in exits]
    kinds = {item.kind for item in items}
    if len(kinds) != 1 or not values.runtime(value.kind):
        frame.fail(
            node,
            "short_circuit_union",
            f"selected expression paths require a common source value type; got {sorted(kinds)}",
        )
    result = Value(value.kind, None, _join_facts(items)).at(frame.temporary())
    steps.insert(
        declaration_index,
        dict(op="decl", name=result.value["name"], type=values.wire(result.kind)),
    )
    for arm, item in exits:
        arm.append(
            dict(op="assign", target=result.value, value=frame.scalar(item, node))
        )
    return result


def evaluate(frame, node, steps, discard=False, truth_context=False):
    """Normalize binary operators, chained comparisons and operand selection."""
    op = node["op"]
    if op == "short_circuit":
        operands = node["operands"]
        operator = node["operator"]
        if len(operands) < 2 or operator not in {"and", "or"}:
            frame.fail(
                node,
                "short_circuit",
                "operand selection requires and/or and two or more operands",
            )

        def operand(index, term, body, as_truth, drop):
            return frame.evaluate(term, body, discard=drop, truth_context=as_truth)

        return _guarded(
            frame,
            node,
            steps,
            operands,
            operand,
            operator == "or",
            discard,
            truth_context,
        )

    if op == "binary":
        operators, operands = [node["operator"]], [node["right"]]
    elif op == "compare_chain":
        operators, operands = node["operators"], node["operands"]
        if not operators or len(operators) != len(operands):
            frame.fail(
                node,
                "comparison_chain",
                "each comparison requires exactly one right operand",
            )
    else:
        frame.fail(node, "operation_contract", f"unexpected expression {op}")

    left = frame.evaluate(node["left"], steps)
    if operands[0]["op"] != "literal":
        left = _capture(frame, left, steps)

    def comparison(index, term, body, as_truth, drop):
        nonlocal left
        right = frame.evaluate(term, body)
        if index + 1 < len(operands):
            # Both adjacent comparisons use the same value, even if evaluating
            # the following operand changes the original name or storage.
            right = _capture(frame, right, body)
        result = apply_binary(frame, node, operators[index], left, right, body, drop)
        left = right
        return frame.truth(result, node) if as_truth else result

    if len(operands) == 1:
        return comparison(0, operands[0], steps, truth_context, discard)
    return _guarded(
        frame, node, steps, operands, comparison, False, discard, truth_context
    )
