"""Seal one original sparse-solver call without replacing its Python driver."""

import ast
from types import CodeType

from scar.trace.code_snapshot import code_key

OFFER = '__scar_sparse_solve_offer'
_CARRIER = '__scar_sparse_solve_carrier'
_ARGS = '__scar_sparse_solve_args'
_OPTIONS = '__scar_sparse_solve_options'
_RESULT = '__scar_sparse_solve_result'
_NAMES = (OFFER, _CARRIER, _ARGS, _OPTIONS, _RESULT)


def _span(node):
    return [node.lineno, node.col_offset, node.end_lineno, node.end_col_offset]


def _code(module, name):
    matches = [body for body in module.co_consts if type(body) is CodeType and body.co_qualname == name]
    if len(matches) != 1:
        raise ValueError('one ordinary module function code object required')
    return matches[0]


def seal_callsite(source, filename, code_id, span, carrier, *, optimize=0):
    """Preserve callable/argument evaluation and keep the fallback call in its original frame."""
    tree = ast.parse(source, filename=filename)
    initial = compile(tree, filename, 'exec', dont_inherit=True, optimize=optimize)
    match = [body for body in initial.co_consts if type(body) is CodeType and code_key(body) == code_id]
    if len(match) != 1:
        raise ValueError('a source-matched solver CodeID is required')
    original, = match
    definitions = [node for node in tree.body if type(node) is ast.FunctionDef and node.name == original.co_qualname]
    if len(definitions) != 1 or original.co_freevars or set(_NAMES) & (set(original.co_names) | set(original.co_varnames)):
        raise ValueError('unsupported solver body or reserved name collision')
    function, = definitions
    candidates = []
    for owner in ast.walk(function):
        for _, nodes in ast.iter_fields(owner):
            if isinstance(nodes, list):
                candidates.extend((nodes, index, node) for index, node in enumerate(nodes)
                    if type(node) is ast.Assign and type(node.value) is ast.Call and _span(node.value) == span)
    if len(candidates) != 1:
        raise ValueError('the admitted call position must identify one direct assignment')
    suite, index, assign = candidates[0]
    call = assign.value
    if (type(call.func) is not ast.Attribute or type(call.func.value) is not ast.Name or call.func.attr != carrier
            or len(call.args) != 7 or any(type(arg) is ast.Starred for arg in call.args)
            or [kw.arg for kw in call.keywords] != ['options']
            or len(assign.targets) != 1 or type(assign.targets[0]) is not ast.Tuple
            or len(assign.targets[0].elts) != 2 or any(type(name) is not ast.Name for name in assign.targets[0].elts)):
        raise ValueError('the admitted solve call does not have the raw CSR driver ABI')

    replacement = ast.parse(f'''
{_CARRIER} = None
{_ARGS} = None
{_OPTIONS} = None
{_RESULT} = {OFFER}({_CARRIER}, {_ARGS}, {_OPTIONS})
if {_RESULT} is None:
    __a, __b = {_CARRIER}(*{_ARGS}, options={_OPTIONS})
else:
    __a, __b = {_RESULT}
''').body
    replacement[0].value = call.func
    replacement[1].value = ast.Tuple(elts=call.args, ctx=ast.Load())
    replacement[2].value = call.keywords[0].value
    branch = replacement[-1]
    branch.body[0].targets = assign.targets
    branch.orelse[0].targets = assign.targets
    for node in replacement:
        ast.copy_location(node, assign)
    for node in ast.walk(branch):
        if hasattr(node, 'lineno'):
            ast.copy_location(node, call)
    for node in ast.walk(replacement[3]):
        if hasattr(node, 'lineno'):
            ast.copy_location(node, call)
    # Parse-generated stores have synthetic coordinates. Anchor those stores,
    # keeping the original callable and argument expressions' locations.
    for node in replacement[:3]:
        ast.copy_location(node.targets[0], assign)
    suite[index:index+1] = replacement
    revised = compile(ast.fix_missing_locations(tree), filename, 'exec', dont_inherit=True, optimize=optimize)
    return dict(qualname=original.co_qualname, provider_binding=call.func.value.id,
                carrier_name=carrier, source_span=span), _code(revised, original.co_qualname)
