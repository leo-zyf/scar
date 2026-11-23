"""F2 numeric expressions and array iteration over explicitly bound Python inputs."""

import ast
import math

_OPS = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}


class Unsupported(ValueError):
    def __init__(self, node, detail):
        self.line = getattr(node, "lineno", 1)
        self.kind = type(node).__name__
        super().__init__(detail)


def var(name):
    return dict(op="var", name=name)


def literal(value):
    if isinstance(value, str):
        return dict(op="string", value=value)
    return dict(op="literal", type="f64", value=float(value))


def _coefficient(node):
    if (
        type(node.value) not in (int, float)
        or not math.isfinite(node.value)
        or abs(node.value) > 2**53
    ):
        raise Unsupported(
            node, "constant exceeds the declared float64 arithmetic domain"
        )
    return literal(node.value)


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


def _scalar(node, bound, symbols):
    if isinstance(node, ast.Name) and node.id in bound:
        return var(node.id)
    if isinstance(node, ast.Constant):
        return _coefficient(node)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return dict(
            op="unary",
            operator="neg" if isinstance(node.op, ast.USub) else "pos",
            value=_scalar(node.operand, bound, symbols),
        )
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Div) and not (
            isinstance(node.right, ast.Constant) and node.right.value != 0
        ):
            raise Unsupported(
                node, "division needs the Python zero-divisor exception contract"
            )
        return dict(
            op="binary",
            operator=_OPS[type(node.op)],
            left=_scalar(node.left, bound, symbols),
            right=_scalar(node.right, bound, symbols),
        )
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in symbols
        and not node.keywords
    ):
        return dict(
            op="call",
            callee=symbols[node.func.id],
            args=[_scalar(x, bound, symbols) for x in node.args],
        )
    raise Unsupported(node, "unresolved Python operation or dynamic dispatch")


def _array_function(fn, names, symbols, entry):
    array_names = names[:-1] if entry else names
    if not array_names:
        raise Unsupported(fn, "array entry needs at least one numerical input")
    element, count, output = "$index", "$count", "$output"
    n = var("$elements")
    ptr = {"kind": "ptr", "to": {"kind": "const", "of": "f64"}}
    internal = {}
    prefix = []
    serial = 0

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
        if isinstance(node, ast.Name) and node.id in bound:
            return var(internal.get(node.id, node.id)), True
        if isinstance(node, ast.Constant):
            return _coefficient(node), False
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value, array = evaluate(node.operand, statements, bound)
            if not array:
                return dict(op="unary", operator="neg", value=value), False
            dest = scratch(statements)
            statements.append(
                copy(dest, dict(op="unary", operator="neg", value=at(value)))
            )
            return var(dest), True
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            a, aa = evaluate(node.left, statements, bound)
            b, ba = evaluate(node.right, statements, bound)
            if not (aa or ba):
                raise Unsupported(
                    node, "scalar-only subtree needs Python scalar promotion"
                )
            if isinstance(node.op, ast.Div):
                raise Unsupported(
                    node, "array divide requires NumPy error-mode effects"
                )
            dest = scratch(statements)
            expr = dict(
                op="binary",
                operator=_OPS[type(node.op)],
                left=at(a) if aa else a,
                right=at(b) if ba else b,
            )
            statements.append(copy(dest, expr))
            return var(dest), True
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in symbols
            and not node.keywords
        ):
            args = [evaluate(a, statements, bound) for a in node.args]
            if not all(is_array for _, is_array in args):
                raise Unsupported(node, "array call has a nonarray argument")
            dest = scratch(statements)
            statements.append(
                dict(
                    op="expr",
                    value=dict(
                        op="call",
                        callee=symbols[node.func.id],
                        args=[n, *[v for v, _ in args], var(dest)],
                    ),
                )
            )
            return var(dest), True
        raise Unsupported(
            node, "dynamic Python expression outside the declared array ABI"
        )

    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
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
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                name = node.targets[0].id
                v, is_array = evaluate(node.value, result, bound)
                if not is_array:
                    raise Unsupported(node, "array variable assigned a scalar")
                result.append(copy(internal[name], at(v)))
                bound.add(name)
            elif isinstance(node, ast.Return) and not inner:
                v, is_array = evaluate(node.value, result, bound)
                if not is_array:
                    raise Unsupported(node, "array entry returned a scalar")
                result.extend([copy(output, at(v)), dict(op="return")])
            elif (
                entry
                and isinstance(node, ast.For)
                and isinstance(node.target, ast.Name)
                and not node.orelse
                and isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
                and not node.iter.keywords
                and len(node.iter.args) == 1
                and isinstance(node.iter.args[0], ast.Name)
                and node.iter.args[0].id == names[-1]
                and node.target.id not in bound
                and node.target.id != names[-1]
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
                        body=statements(node.body, bound.copy(), True),
                    )
                )
            else:
                raise Unsupported(node, "statement outside the closed array profile")
        return result

    if not isinstance(fn.body[-1], ast.Return):
        raise Unsupported(fn, "array function needs a total return")
    params = [dict(name="$elements", type="i64")]
    if entry:
        params.append(dict(name=count, type="i64"))
    params.extend(dict(name=name, type=ptr) for name in array_names)
    params.append(dict(name=output, type={"kind": "ptr", "to": "f64"}))
    return dict(
        name=symbols[fn.name],
        result="void",
        params=params,
        body=prefix + statements(fn.body, set(array_names)),
    )
