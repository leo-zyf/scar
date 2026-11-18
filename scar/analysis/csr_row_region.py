"""Recognize a closed CSR row edit inside an admitted Python source body."""

import ast
from types import CodeType

from scar.trace.code_snapshot import code_key

FAMILY = 'CSR_ZERO_ROWS_SET_DIAGONAL'
_COPIED = ast.parse('C = A if O else A.copy()').body[0]
_REGION = ast.parse('''
S = C.indptr[R]
N = C.indptr[R + 1] - S
if N.sum() > 0:
    F = P.arange(N.sum()) - P.repeat(P.cumsum(N) - N, N)
    J = P.repeat(S, N) + F
    C.data[J] = 0.
V = C.diagonal()
V[R] = D
C.setdiag(V)
''').body
_INPUTS = ('A', 'C', 'R', 'D', 'O', 'P')
_LOCALS = ('S', 'N', 'F', 'J', 'V')


def _match(pattern, node, names):
    if type(pattern) is not type(node):
        return False
    if isinstance(pattern, ast.Name):
        key = pattern.id
        if key not in names:
            if node.id in names.values():
                return False
            names[key] = node.id
        return names[key] == node.id and type(pattern.ctx) is type(node.ctx)
    if isinstance(pattern, ast.AST):
        return all(_match(value, getattr(node, field), names)
                   for field, value in ast.iter_fields(pattern))
    if isinstance(pattern, list):
        return len(pattern) == len(node) and all(_match(a, b, names) for a, b in zip(pattern, node))
    return pattern == node


def _span(node):
    return [node.lineno, node.col_offset, node.end_lineno, node.end_col_offset]


def _code(module_code, name):
    return next(code for code in module_code.co_consts
                if type(code) is CodeType and code.co_name == name)


def recognize(source, filename, expected_code, *, optimize=0, helper=None):
    """Return a region description and optional replacement code, without importing the source."""
    module = ast.parse(source, filename=filename)
    original = compile(module, filename, 'exec', dont_inherit=True, optimize=optimize)
    matching = [code for code in original.co_consts
                if type(code) is CodeType and code_key(code) == expected_code]
    if len(matching) != 1:
        raise ValueError('expected one admitted ordinary top-level source body')
    code, = matching
    definitions = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == code.co_name]
    if len(definitions) != 1 or code.co_freevars or code.co_cellvars:
        raise ValueError('expected one ordinary source definition without closure cells')
    function, = definitions
    body = function.body
    length = len(_REGION)
    candidates = []
    for first in range(1, len(body) - length + 1):
        names = {}
        before = body[first - 1]
        nodes = body[first:first + length]
        if _match(_COPIED, before, names) and _match(_REGION, nodes, names):
            candidates.append((first, names, before, nodes))
    if len(candidates) != 1:
        raise ValueError('expected one structurally matched CSR region after a conditional copy')
    first, names, before, nodes = candidates[0]
    parameters = {arg.arg for arg in ast.walk(function.args) if isinstance(arg, ast.arg)}
    if any(names[key] not in parameters for key in ('A', 'R', 'D', 'O')):
        raise ValueError('CSR source region bindings must enter at original argument ports')
    if names['P'] not in code.co_names or names['P'] in code.co_varnames:
        raise ValueError('NumPy operations must come from an original global binding')
    temporaries = {names[key] for key in _LOCALS}
    if temporaries & parameters or any(isinstance(child, ast.Name) and child.id in temporaries
            for node in body[:first] + body[first + length:] for child in ast.walk(node)):
        raise ValueError('removed temporary has another source definition or use')
    if any(isinstance(node, (ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom)) for node in ast.walk(function)):
        raise ValueError('CSR source body contains suspension or redirected bindings')

    description = dict(family=FAMILY, qualname=code.co_qualname, code_key=expected_code,
        inputs={key: names[key] for key in _INPUTS}, internal_locals=sorted(temporaries),
        copy=_span(before), statements=[_span(node) for node in nodes],
        control='SINGLE_ENTRY_NORMAL_COMPLETION_CONTIGUOUS_SUITE',
        effects=['WRITE_PRIVATE_CSR_DATA', 'KEEP_CSR_INDEX_STORAGE',
                 'PRESERVE_PROVIDER_DTYPE_MEMO_UPDATE_ON_FIRST_ACCESS'],
        conditions=['STANDARD_BOUND_NUMPY_AND_SCIPY_OPERATIONS', 'EXCLUSIVE_NORMAL_EXECUTION',
                    'NO_FRAME_REFLECTION_OR_ALLOCATION_FAILURE_OBSERVATION',
                    'LOWERING_GUARDS_BEFORE_ANY_WRITE'])
    if helper is None:
        return description, None
    if helper in code.co_names or helper in code.co_varnames or not helper.isidentifier():
        raise ValueError('dispatcher name conflicts with original bindings')
    call = ast.Call(func=ast.Name(id=helper, ctx=ast.Load()), args=[
        ast.Name(id=names[key], ctx=ast.Load()) for key in _INPUTS], keywords=[])
    branch = ast.If(test=ast.UnaryOp(op=ast.Not(), operand=call), body=nodes, orelse=[])
    body[first:first + length] = [ast.copy_location(branch, nodes[0])]
    revised = compile(ast.fix_missing_locations(module), filename, 'exec', dont_inherit=True, optimize=optimize)
    return description, _code(revised, code.co_name)
