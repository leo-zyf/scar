"""Project a bounded, elementwise Python callable graph into portable IR."""

import ast
import math


_BINARY = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}


class _Unsupported(ValueError):
    pass


def _expression(node, bound, functions, callees):
    if isinstance(node, ast.Name) and node.id in bound:
        return ["read", node.id], True
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = node.value
        if (type(value) is int and abs(value) > 2**53) or not math.isfinite(value):
            raise _Unsupported()
        return ["constant", value], False
    if isinstance(node, ast.UnaryOp) and type(node.op) is ast.USub:
        operand, is_array = _expression(node.operand, bound, functions, callees)
        return ["neg", operand], is_array
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, left_array = _expression(node.left, bound, functions, callees)
        right, right_array = _expression(node.right, bound, functions, callees)
        if not (left_array or right_array):
            raise _Unsupported()
        return [_BINARY[type(node.op)], left, right], True
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords
            and node.func.id in functions and len(node.args) == len(functions[node.func.id].args.args)):
        name = node.func.id
        callees.add(name)
        arguments = [_array_expression(arg, bound, functions, callees) for arg in node.args]
        return ["call", name, *arguments], True
    raise _Unsupported()


def _array_expression(node, bound, functions, callees):
    expression, is_array = _expression(node, bound, functions, callees)
    if not is_array:
        raise _Unsupported()
    return expression


def _statements(nodes, bound, functions, callees, *, count=None, allow_return=True):
    body = []
    for index, node in enumerate(nodes):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == count:
                raise _Unsupported()
            body.append(["store", name, _array_expression(node.value, bound, functions, callees)])
            bound.add(name)
        elif (isinstance(node, ast.Return) and allow_return and index == len(nodes) - 1
              and node.value is not None):
            body.append(["return", _array_expression(node.value, bound, functions, callees)])
        elif isinstance(node, ast.For) and count is not None and not node.orelse:
            call = node.iter
            if (not isinstance(node.target, ast.Name) or node.target.id in bound or node.target.id == count
                    or not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name)
                    or call.func.id != "range" or call.keywords or len(call.args) != 1
                    or not isinstance(call.args[0], ast.Name) or call.args[0].id != count):
                raise _Unsupported()
            nested = _statements(node.body, set(bound), functions, callees, allow_return=False)
            body.append(["repeat", nested])
        else:
            raise _Unsupported()
    if allow_return and (not body or body[-1][0] != "return"):
        raise _Unsupported()
    return body


def _portable_function(function, functions, *, entry):
    args = function.args
    if (function.decorator_list or args.posonlyargs or args.vararg is not None or args.kwarg is not None
            or args.kwonlyargs or args.defaults or not args.args):
        raise _Unsupported()
    parameters = [arg.arg for arg in args.args]
    count = parameters[-1] if entry else None
    inputs = parameters[:-1] if entry else parameters
    if not inputs:
        raise _Unsupported()
    callees = set()
    nodes = function.body
    if (nodes and isinstance(nodes[0], ast.Expr) and isinstance(nodes[0].value, ast.Constant)
            and isinstance(nodes[0].value.value, str)):
        nodes = nodes[1:]
    body = _statements(nodes, set(inputs), functions, callees, count=count)
    if entry and not any(row[0] == "repeat" for row in body):
        raise _Unsupported()
    return dict(name=function.name, parameters=inputs, body=body), callees


def lower_portable(source_text, entry_name=None, *, tree=None):
    """Return callable and counted-loop IR when the closed arithmetic profile applies."""
    if tree is None:
        tree = ast.parse(source_text)
    definitions = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name in definitions or node.name == "range":
                return None
            definitions[node.name] = node
        elif not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)):
            # This profile admits a closed module with stable global call bindings.
            return None
    if entry_name is None:
        entries = [name for name, node in definitions.items() if any(isinstance(row, ast.For) for row in node.body)]
        if len(entries) != 1:
            return None
        entry_name = entries[0]
    if entry_name not in definitions:
        return None
    active, prepared = set(), {}

    def visit(name):
        if name in active:
            raise _Unsupported()
        if name in prepared:
            return
        active.add(name)
        function, callees = _portable_function(definitions[name], definitions, entry=name == entry_name)
        for callee in sorted(callees):
            visit(callee)
        active.remove(name)
        prepared[name] = function

    try:
        visit(entry_name)
    except _Unsupported:
        return None
    return dict(entry=entry_name, functions=list(prepared.values()))
