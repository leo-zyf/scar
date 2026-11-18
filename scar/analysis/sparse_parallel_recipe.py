"""Recognize detached arithmetic for a sealed repeated CSR update/solve case."""

import ast
import math
from operator import add, mul, sub, truediv
from types import CodeType

from scar.trace.code_snapshot import code_key

_OPERATORS = {kind.__name__: operation for kind, operation in (
    (ast.Add, add), (ast.Sub, sub), (ast.Mult, mul), (ast.Div, truediv))}


def _span(node):
    return [node.lineno, node.col_offset, node.end_lineno, node.end_col_offset]


def _formula(node, variables):
    if type(node) is ast.Name and node.id in variables:
        return ['input', variables[node.id]]
    if type(node) is ast.Constant and type(node.value) in (int, float) and math.isfinite(node.value):
        return ['constant', node.value]
    if type(node) is ast.BinOp and type(node.op).__name__ in _OPERATORS:
        return [type(node.op).__name__, _formula(node.left, variables), _formula(node.right, variables)]
    raise ValueError('speculative scale needs only exact scalar arithmetic and the source loop inputs')


def compute_scale(expression, index, count):
    """Replay a sealed arithmetic tree with built-in integer/float semantics."""
    kind, a, *rest = expression
    if kind == 'constant':
        return a
    if kind == 'input':
        return (index, count)[a]
    return _OPERATORS[kind](compute_scale(a, index, count), compute_scale(rest[0], index, count))


def _name(node, expected):
    return type(node) is ast.Name and node.id == expected


def derive_recipe(raw, body, preparation, *, optimize):
    path = body['path']
    module = compile(raw, path, 'exec', dont_inherit=True, optimize=optimize)
    matches = [code for code in module.co_consts if type(code) is CodeType and code_key(code) == body['code_key']]
    if len(matches) != 1:
        raise ValueError('observed outer function CodeID has no single original definition')
    code, = matches
    definitions = [node for node in ast.parse(raw, filename=path).body
                   if type(node) is ast.FunctionDef and node.name == code.co_qualname]
    if len(definitions) != 1:
        raise ValueError('speculative solve requires an ordinary outer source function')
    roles = body['parameter_roles']
    function = definitions[0]
    names = {arg.arg for arg in (*function.args.posonlyargs, *function.args.args)}
    if set(roles) != {'matrix', 'load', 'boundary', 'count'} or set(roles.values()) != names:
        raise ValueError('live inputs do not match the observed source parameter roles')
    loops = [node for node in function.body if type(node) is ast.For]
    if len(loops) != 1:
        raise ValueError('one outer source iteration is required')
    loop, = loops
    iterator = loop.iter
    i = preparation['iteration_variable']
    scale = preparation['scale_binding']
    if (not _name(loop.target, i) or type(iterator) is not ast.Call or not _name(iterator.func, 'range')
            or iterator.keywords or len(iterator.args) != 1 or not _name(iterator.args[0], roles['count'])
            or len(loop.body) < 2):
        raise ValueError('iteration head differs from the admitted source range')
    scale_stmt, update_stmt = loop.body[:2]
    if (type(scale_stmt) is not ast.Assign or len(scale_stmt.targets) != 1
            or not _name(scale_stmt.targets[0], scale) or _span(scale_stmt.value) != preparation['scale_expression_span']
            or type(update_stmt) is not ast.Assign or type(update_stmt.value) is not ast.Call
            or _span(update_stmt.value) != preparation['row_edit_call_span']):
        raise ValueError('candidate preparation differs from the original consecutive statements')
    update = update_stmt.value
    boundary = {kw.arg: kw.value for kw in update.keywords}
    rhs = update.args[1] if len(update.args) == 2 else None
    if (rhs is None or not _name(update.args[0], roles['matrix']) or type(rhs) is not ast.BinOp
            or type(rhs.op) is not ast.Mult or not _name(rhs.left, roles['load']) or not _name(rhs.right, scale)
            or len(update.keywords) != 2 or set(boundary) != {'D', 'diag'}
            or not _name(boundary['D'], roles['boundary']) or not _name(boundary['diag'], scale)):
        raise ValueError('original update no longer has the standard private CSR preparation')
    return dict(body_code_key=body['code_key'], roles=roles,
        scale=_formula(scale_stmt.value, {i:0, roles['count']:1}),
        source_spans=dict(scale=preparation['scale_expression_span'], update=preparation['row_edit_call_span']))
