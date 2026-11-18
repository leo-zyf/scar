"""Recognize a copy-initialized partition update without running its source."""

import ast
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import sys
from types import CodeType

from scar.analysis.source_bytes import source_bytes_document
from scar.analysis.source_syntax import read_source_syntax, source_syntax_document
from scar.analysis.source_tokens import source_tokens_document
from scar.trace.code_snapshot import code_key


SCHEMA = 'scar.partition-copy-source'


def _span(node):
    return [node.lineno, node.col_offset, node.end_lineno, node.end_col_offset]


def _same(node, source):
    expected, = ast.parse(source).body
    return ast.dump(node) == ast.dump(expected)


def _enumeration(node, selectors):
    match node:
        case ast.For(target=ast.Tuple(elts=[ast.Name(id=index), ast.Name(id=view)]),
                     iter=ast.Call(func=ast.Name(id='enumerate'),
                                   args=[ast.Name(id=iterable)], keywords=[]), orelse=[]):
            if iterable == selectors:
                return index, view
    return None


@dataclass(frozen=True, slots=True)
class _Window:
    inputs: tuple[str, ...]
    prefix: tuple[ast.If, ...]
    initializer: ast.stmt
    copy: ast.Assign
    outer: ast.For
    diagonal: ast.AugAssign
    inner: ast.For
    off_diagonal: ast.AugAssign
    result: ast.Return


def _window(function):
    args = function.args
    if (function.decorator_list or args.vararg or args.kwarg or args.posonlyargs
            or len(args.args) != 3 or len(args.kwonlyargs) != 1 or args.defaults):
        return None
    names = tuple(arg.arg for arg in (*args.args, *args.kwonlyargs))
    source, weights, selectors, output = names
    statements = function.body[1:] if ast.get_docstring(function, clean=False) is not None else function.body
    if len(statements) < 3:
        return None
    *prefix, initializer, outer, result = statements
    # Only admit source checks that are implied by the optimized path's guards.
    guards = [f'{output} is {source}', f'{weights}.shape != (len({selectors}), len({selectors}))']
    for statement in prefix:
        match statement:
            case ast.If(test=test, body=[ast.Raise(exc=ast.Call(func=ast.Name(id='ValueError'),
                        args=[ast.Constant(value=message)], keywords=[]), cause=None)], orelse=[]):
                guard = next((form for form in guards if _same(ast.Expr(test), form)), None)
                if guard is not None and type(message) is str:
                    guards.remove(guard)
                    continue
        return None
    copy = initializer
    if isinstance(initializer, ast.If):
        if (not _same(ast.Expr(initializer.test), f'{output} is None')
                or len(initializer.body) != 1 or len(initializer.orelse) != 1):
            return None
        match initializer.body[0]:
            case ast.Assign(targets=[ast.Name(id=dest)], value=ast.Call(
                    func=ast.Attribute(value=ast.Name(), attr='copy'),
                    args=[ast.Name(id=operand)], keywords=[])) if dest == output and operand == source:
                copy, = initializer.orelse
            case _:
                return None
    if not _same(copy, f'{output}[...] = {source}[...]') or not _same(result, f'return {output}'):
        return None
    outer_index = _enumeration(outer, selectors)
    if outer_index is None or len(outer.body) != 2:
        return None
    i, src_i = outer_index
    diagonal, inner = outer.body
    inner_index = _enumeration(inner, selectors)
    if inner_index is None:
        return None
    j, src_j = inner_index
    if len(set((*names, i, src_i, j, src_j))) != 8:
        return None
    if not _same(diagonal, f'{output}[{src_i}] *= {weights}[{i}, {i}]'):
        return None
    match inner.body:
        case [ast.If(test=test, body=[off_diagonal], orelse=[])]:
            if _same(ast.Expr(test), f'{i} != {j}') and _same(
                    off_diagonal, f'{output}[{src_i}] += {source}[{src_j}] * {weights}[{i}, {j}]'):
                return _Window(names, tuple(prefix), initializer, copy, outer, diagonal, inner, off_diagonal, result)
    return None


def discover(source):
    """Return source-bound windows selected by operations and data routes."""
    path = Path(source).resolve(strict=True)
    raw = path.read_bytes()
    f0 = source_bytes_document(raw, path=str(path), namespace_root=str(path.parent))
    f1 = source_tokens_document(f0)
    f2 = source_syntax_document(f0, f1)
    tree = read_source_syntax(f2, f0, f1)
    if not isinstance(tree, ast.Module):
        raise ValueError('partition discovery requires a parsed source module')
    counts = Counter(node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
    windows = [(node, _window(node)) for node in tree.body
               if type(node) is ast.FunctionDef and counts[node.name] == 1]
    if not any(window is not None for _, window in windows):
        return []
    compiled = compile(tree, str(path), 'exec', dont_inherit=True)
    codes = {body.co_name: body for body in compiled.co_consts if type(body) is CodeType
             and body.co_qualname == body.co_name}
    results = []
    for definition, window in windows:
        if window is None:
            continue
        code = codes[definition.name]
        results.append(dict(schema=SCHEMA, schema_version=1, path=str(path),
            source_sha256=sha256(raw).hexdigest(), code_id=code_key(code),
            qualname=code.co_qualname, inputs=list(window.inputs),
            parser=dict(api=f2['profile']['api'], ast_sha256=f2['profile']['ast_sha256'],
                        parser_sha256=f2['profile']['parser_sha256']),
            spans={role: _span(getattr(window, role)) for role in
                   ('initializer', 'copy', 'outer', 'diagonal', 'inner', 'off_diagonal', 'result')},
            entry_checks=[_span(node) for node in window.prefix],
            conditional_output_allocation=window.initializer is not window.copy,
            optimize=sys.flags.optimize))
    return results
