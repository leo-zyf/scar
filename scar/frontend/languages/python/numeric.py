"""F2 numeric expressions and array iteration over explicitly bound Python inputs."""

import math
from scar.contract import walk_operations


class Unsupported(ValueError):
    def __init__(self, node, detail):
        self.line = (
            node.get("location", {}).get("line", 1)
            if isinstance(node, dict)
            else getattr(node, "lineno", 1)
        )
        self.kind = (
            node.get("op", "protocol")
            if isinstance(node, dict)
            else type(node).__name__
        )
        super().__init__(detail)


def var(name):
    return dict(op="var", name=name)


def literal(value):
    if isinstance(value, str):
        return dict(op="string", value=value)
    return dict(op="literal", type="f64", value=float(value))


def _float_literal(node, value):
    if (
        type(value) not in (int, float)
        or type(value) is int
        and abs(value) > 2**53
        or type(value) is float
        and not math.isfinite(value)
    ):
        raise Unsupported(
            node, "constant exceeds the declared float64 arithmetic domain"
        )
    return literal(value)


def _names(fn):
    args = fn.args
    if (
        fn.decorator_list
        or args.defaults
        or args.posonlyargs
        or args.kwonlyargs
        or args.vararg
        or args.kwarg
    ):
        raise Unsupported(
            fn, "Python callable requires runtime argument binding or decorators"
        )
    return [a.arg for a in args.args]


_DYNAMIC = object()
_ARITHMETIC = {"add": "add", "sub": "sub", "mult": "mul", "div": "div"}
_COMPARE = {"eq": "eq", "noteq": "ne", "lt": "lt", "lte": "le", "gt": "gt", "gte": "ge"}


def _integer_constant(node, cache):
    """Keep Python integer arithmetic exact before a declared float conversion."""
    identity = id(node)
    if identity in cache:
        return cache[identity]
    op = node.get("op")
    result = _DYNAMIC
    if op == "literal" and type(node.get("value")) is int:
        result = node["value"]
    elif op == "unary" and node["operator"] in {"usub", "uadd"}:
        value = _integer_constant(node["value"], cache)
        if value is not _DYNAMIC:
            result = -value if node["operator"] == "usub" else value
    elif op == "binary" and node["operator"] in {"add", "sub", "mult"}:
        left, right = (_integer_constant(node[k], cache) for k in ("left", "right"))
        if left is not _DYNAMIC and right is not _DYNAMIC:
            result = {
                "add": lambda: left + right,
                "sub": lambda: left - right,
                "mult": lambda: left * right,
            }[node["operator"]]()
    cache[identity] = result
    return result


def _scalar_function(function):
    """Qualify a Python protocol body using an explicit by-value float64 ABI."""
    env = {p["name"]: "f64" for p in function["params"]}
    locals_, serial = {}, 0
    loop_depth = 0
    constants = {}

    def fail(node, detail):
        raise Unsupported(node, detail)

    def fresh():
        nonlocal serial
        serial += 1
        return "$value" + str(serial)

    def number(node, value):
        if type(value) is bool:
            return dict(op="literal", type="bool", value=value), "bool"
        return _float_literal(node, value), "f64"

    def expr(node, prefix):
        exact = _integer_constant(node, constants)
        if exact is not _DYNAMIC:
            return number(node, exact)
        op = node["op"]
        if op == "literal":
            return number(node, node.get("value"))
        if op == "var" and node.get("scope") != "global":
            if node["name"] not in env:
                fail(node, "local may be read before its assignment")
            return var(node["name"]), env[node["name"]]
        if op == "unary" and node["operator"] in {"usub", "uadd", "not"}:
            value, type_ = expr(node["value"], prefix)
            if node["operator"] == "not":
                return dict(op="unary", operator="not", value=value), "bool"
            if type_ != "f64":
                fail(node, "integer or boolean promotion needs a precise value domain")
            return (
                dict(
                    op="unary",
                    operator="neg" if node["operator"] == "usub" else "pos",
                    value=value,
                ),
                type_,
            )
        if op == "binary" and node["operator"] in _ARITHMETIC:
            left, lt = expr(node["left"], prefix)
            right, rt = expr(node["right"], prefix)
            if lt != "f64" or rt != "f64":
                fail(node, "mixed Python numeric types require explicit promotion")
            if node["operator"] == "div" and not (
                right["op"] == "literal" and right["value"] != 0
            ):
                fail(node, "division needs the Python zero-divisor exception contract")
            return (
                dict(
                    op="binary",
                    operator=_ARITHMETIC[node["operator"]],
                    left=left,
                    right=right,
                ),
                "f64",
            )
        if op == "compare_chain" and len(node["operators"]) == 1:
            left, lt = expr(node["left"], prefix)
            right, rt = expr(node["operands"][0], prefix)
            operator = _COMPARE.get(node["operators"][0])
            if operator is None or lt != rt:
                fail(node, "comparison requires one resolved numeric protocol")
            return dict(op="binary", operator=operator, left=left, right=right), "bool"
        if (
            op == "call"
            and node.get("binding") in {"python_lexical", "python_abi"}
            and not node.get("keywords")
        ):
            args = []
            for arg in node["args"]:
                value, type_ = expr(arg, prefix)
                if type_ != "f64":
                    fail(arg, "native scalar call requires float64 by-value arguments")
                args.append(value)
            name = fresh()
            prefix.append(
                dict(
                    op="decl",
                    name=name,
                    type="f64",
                    init=dict(
                        op="call",
                        callee=node["callee"],
                        args=args,
                        parameter_types=["f64"] * len(args),
                    ),
                )
            )
            return var(name), "f64"
        fail(node, "operation is outside the resolved float64 scalar protocol")

    def assign(node, target, value, type_):
        if (
            target.get("op") != "var"
            or target.get("scope") == "global"
            or target.get("cell")
        ):
            fail(node, "assignment requires an owned scalar local")
        name = target["name"]
        previous = env.get(name, locals_.get(name))
        if previous is not None and previous != type_:
            fail(node, "the scalar binding changes its runtime type")
        if name not in {p["name"] for p in function["params"]}:
            locals_[name] = type_
        env[name] = type_
        return dict(op="assign", target=var(name), value=value)

    def sequence(items):
        nonlocal env, loop_depth
        result = []
        for node in items:
            op = node["op"]
            if op == "decl" and node.get("initial_binding") == "unbound":
                continue
            if op == "assign":
                if type(_integer_constant(node["value"], constants)) is int:
                    fail(
                        node,
                        "an integer local needs bounded or arbitrary-precision state",
                    )
                value, type_ = expr(node["value"], result)
                result.append(assign(node, node["target"], value, type_))
            elif op == "update":
                target, type_ = expr(node["target"], result)
                value, vt = expr(node["value"], result)
                operator = _ARITHMETIC.get(node["operator"])
                if (
                    type_ != "f64"
                    or vt != "f64"
                    or operator not in {"add", "sub", "mul"}
                ):
                    fail(
                        node, "update requires a resolved float64 arithmetic operation"
                    )
                result.append(
                    assign(
                        node,
                        node["target"],
                        dict(op="binary", operator=operator, left=target, right=value),
                        type_,
                    )
                )
            elif op == "return":
                value, type_ = expr(node["value"], result)
                if type_ == "bool":
                    value = dict(op="cast", type="f64", value=value)
                result.append(dict(op="return", value=value))
                break
            elif op == "expr":
                if (
                    node["value"]["op"] == "literal"
                    and type(node["value"].get("value")) is str
                ):
                    continue
                value, _ = expr(node["value"], result)
                result.append(
                    dict(op="expr", value=dict(op="cast", type="void", value=value))
                )
            elif op == "if":
                condition, _ = expr(node["condition"], result)
                before = env.copy()
                yes = sequence(node["then"])
                left = env
                env = before.copy()
                no = sequence(node.get("else", []))
                right = env
                returns = lambda body: bool(body) and (
                    body[-1]["op"] == "return"
                    or body[-1]["op"] == "if"
                    and total(body[-1]["then"])
                    and total(body[-1]["else"])
                )
                env = (
                    right
                    if returns(yes)
                    else (
                        left
                        if returns(no)
                        else {
                            name: type_
                            for name, type_ in left.items()
                            if right.get(name) == type_
                        }
                    )
                )
                result.append(
                    dict(op="if", condition=condition, then=yes, **{"else": no})
                )
            elif op == "while" and not node.get("else"):
                before = env.copy()
                condition_prefix = []
                condition, _ = expr(node["condition"], condition_prefix)
                loop_depth += 1
                body = sequence(node["body"])
                loop_depth -= 1
                env = before
                if condition_prefix:
                    body = [
                        *condition_prefix,
                        dict(
                            op="if",
                            condition=dict(op="unary", operator="not", value=condition),
                            then=[dict(op="break")],
                            **{"else": []},
                        ),
                        *body,
                    ]
                    condition = dict(op="literal", type="bool", value=True)
                result.append(dict(op="while", condition=condition, body=body))
            elif op in {"break", "continue"} and loop_depth:
                result.append(dict(op=op))
                break
            elif op == "pass":
                continue
            else:
                fail(node, "statement requires an unresolved Python execution protocol")
        return result

    def total(body):
        return bool(body) and (
            body[-1]["op"] == "return"
            or body[-1]["op"] == "if"
            and total(body[-1]["then"])
            and total(body[-1]["else"])
        )

    body = sequence(function["partial_body"])
    if not total(body):
        fail(function, "scalar function has a reachable implicit None return")
    return dict(
        name=function["name"],
        result="f64",
        params=[dict(name=p["name"], type="f64") for p in function["params"]],
        body=[
            *[
                dict(op="decl", name=name, type=type_)
                for name, type_ in locals_.items()
            ],
            *body,
        ],
        location=function["location"],
    )


def _array_function(function, entry):
    names = [p["name"] for p in function["params"]]
    array_names = names[:-1] if entry else names
    if not array_names:
        raise Unsupported(function, "array entry needs at least one numerical input")
    element, count, output = "$index", "$count", "$output"
    n = var("$elements")
    ptr = {"kind": "ptr", "to": {"kind": "const", "of": "f64"}}
    internal = {}
    prefix = []
    serial = 0
    constants = {}

    def at(base):
        return dict(op="index", base=base, indices=[var(element)])

    def copy(target, expr):
        return dict(
            op="map",
            index=element,
            extent=n,
            independent=True,
            proof="fresh private result or owned destination; same-index accesses",
            body=[dict(op="assign", target=at(var(target)), value=expr)],
        )

    def scratch(statements):
        nonlocal serial
        serial += 1
        name = f"$temporary{serial}"
        statements.append(dict(op="buffer", name=name, type="f64", extent=n))
        return name

    def evaluate(node, statements, bound):
        op = node["op"]
        exact = _integer_constant(node, constants)
        if exact is not _DYNAMIC:
            return _float_literal(node, exact), False
        if op == "var" and node.get("scope") != "global" and node["name"] in bound:
            return var(internal.get(node["name"], node["name"])), True
        if op == "literal":
            return _float_literal(node, node["value"]), False
        if op == "unary" and node["operator"] == "usub":
            value, array = evaluate(node["value"], statements, bound)
            if not array:
                return dict(op="unary", operator="neg", value=value), False
            dest = scratch(statements)
            statements.append(
                copy(dest, dict(op="unary", operator="neg", value=at(value)))
            )
            return var(dest), True
        if op == "binary" and node["operator"] in _ARITHMETIC:
            a, aa = evaluate(node["left"], statements, bound)
            b, ba = evaluate(node["right"], statements, bound)
            if not (aa or ba):
                raise Unsupported(
                    node, "scalar-only subtree needs Python scalar promotion"
                )
            if node["operator"] == "div":
                raise Unsupported(
                    node, "array divide requires NumPy error-mode effects"
                )
            dest = scratch(statements)
            expr = dict(
                op="binary",
                operator=_ARITHMETIC[node["operator"]],
                left=at(a) if aa else a,
                right=at(b) if ba else b,
            )
            statements.append(copy(dest, expr))
            return var(dest), True
        if (
            op == "call"
            and node.get("binding") in {"python_lexical", "python_abi"}
            and not node.get("keywords")
        ):
            args = [evaluate(a, statements, bound) for a in node["args"]]
            if not all(is_array for _, is_array in args):
                raise Unsupported(node, "array call has a nonarray argument")
            dest = scratch(statements)
            statements.append(
                dict(
                    op="expr",
                    value=dict(
                        op="call",
                        callee=node["callee"],
                        args=[n, *[v for v, _ in args], var(dest)],
                    ),
                )
            )
            return var(dest), True
        raise Unsupported(
            node, "dynamic Python expression outside the declared array ABI"
        )

    for node in walk_operations(function["partial_body"]):
        if (
            node["op"] == "assign"
            and node["target"]["op"] == "var"
            and node["target"].get("scope") != "global"
        ):
            name = node["target"]["name"]
            if name not in internal:
                internal[name] = "$local_" + name
                prefix.append(
                    dict(op="buffer", name=internal[name], type="f64", extent=n)
                )
                if name in array_names:
                    prefix.append(copy(internal[name], at(var(name))))

    def statements(nodes, bound, inner=False):
        result = []
        for node in nodes:
            op = node["op"]
            if op == "decl" and node.get("initial_binding") == "unbound":
                continue
            if (
                op == "expr"
                and node["value"]["op"] == "literal"
                and type(node["value"].get("value")) is str
            ):
                continue
            if (
                op == "assign"
                and node["target"]["op"] == "var"
                and node["target"].get("scope") != "global"
                and not node["target"].get("cell")
            ):
                name = node["target"]["name"]
                v, is_array = evaluate(node["value"], result, bound)
                if not is_array:
                    raise Unsupported(node, "array variable assigned a scalar")
                result.append(copy(internal[name], at(v)))
                bound.add(name)
            elif op == "return" and not inner:
                v, is_array = evaluate(node["value"], result, bound)
                if not is_array:
                    raise Unsupported(node, "array entry returned a scalar")
                result.extend([copy(output, at(v)), dict(op="return")])
                break
            elif (
                entry
                and op == "for"
                and node.get("protocol") == "iterator"
                and node["target"]["op"] == "var"
                and not node.get("else")
                and node["range"].get("op") == "library_call"
                and node["range"].get("symbol") == "builtins.range"
                and not node["range"].get("keywords")
                and len(node["range"]["args"]) == 1
                and node["range"]["args"][0].get("op") == "var"
                and node["range"]["args"][0]["name"] == names[-1]
                and node["target"]["name"] not in bound
                and node["target"]["name"] != names[-1]
            ):
                i = "$iteration"
                result.append(
                    dict(
                        op="for",
                        init=[
                            dict(
                                op="decl",
                                name=i,
                                type="i64",
                                init=dict(op="literal", type="i64", value=0),
                            )
                        ],
                        condition=dict(
                            op="binary", operator="lt", left=var(i), right=var(count)
                        ),
                        step=dict(op="unary", operator="preinc", value=var(i)),
                        body=statements(node["body"], bound.copy(), True),
                    )
                )
            else:
                raise Unsupported(node, "statement outside the closed array profile")
        return result

    body = function["partial_body"]
    if not body or body[-1]["op"] != "return":
        raise Unsupported(function, "array function needs a total return")
    params = [dict(name="$elements", type="i64")]
    if entry:
        params.append(dict(name=count, type="i64"))
    params.extend(dict(name=name, type=ptr) for name in array_names)
    params.append(dict(name=output, type={"kind": "ptr", "to": "f64"}))
    return dict(
        name=function["name"],
        result="void",
        params=params,
        body=prefix + statements(body, set(array_names)),
    )
