"""Seal one original scalar table expression and its untouched fallback."""

import argparse
import ast
from copy import deepcopy
import dis
import json
import marshal
from pathlib import Path
from types import CodeType

from scar.analysis.scalar_table_cse import check
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.runtime.csr_region_entry import interpreter_context
from scar.runtime.scalar_table_contract import ENTRY_CODE, MISS, OFFER, PROOF, RESULT, SCHEMA, compile_original
from scar.trace.code_snapshot import code_key


_RESERVED = {OFFER, MISS, RESULT}
_FAST = {'LOAD_FAST', 'LOAD_FAST_BORROW'}


def _require(condition, detail):
    if not condition:
        raise ValueError(detail)


def code_tree(code):
    yield code
    for child in code.co_consts:
        if type(child) is CodeType:
            yield from code_tree(child)


def _span(node):
    return (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)


def _position(record):
    return tuple(record[field] for field in ('line', 'column', 'end_line', 'end_column'))


def _definition(tree, qualname):
    parts = qualname.split('.')
    _require(all(part.isidentifier() for part in parts), 'one ordinary module or class function is required')
    owner = tree
    for part in parts[:-1]:
        choices = [node for node in owner.body if type(node) is ast.ClassDef and node.name == part]
        _require(len(choices) == 1, 'original function owner is ambiguous')
        owner, = choices
    choices = [node for node in owner.body if type(node) is ast.FunctionDef and node.name == parts[-1]]
    _require(len(choices) == 1 and not choices[0].decorator_list, 'one plain original function is required')
    return choices[0]


def _loads(code, nodes, mappings):
    """Do not speculate a name resolution that can change exception order."""
    instructions = list(dis.get_instructions(code))
    loads = []
    for node, evidence in zip(nodes, mappings, strict=True):
        if type(node) is ast.Constant:
            _require(type(node.value) is int and 'namespace' not in evidence,
                'the retained index is not an exact integer literal')
            loads.append(dict(span=list(_span(node)), opcode='SOURCE_INT_LITERAL'))
            continue
        _require(type(node) is ast.Name and type(node.ctx) is ast.Load
            and node.id == evidence['namespace']['name'], 'compiled operand is not the retained source binding')
        matched = [op for op in instructions if op.positions is not None and op.argval == node.id
            and (op.positions.lineno, op.positions.col_offset, op.positions.end_lineno,
                 op.positions.end_col_offset) == _span(node)]
        namespace = evidence['namespace']['kind']
        _require(len(matched) == 1 and ((namespace == 'LEXICAL' and matched[0].opname in _FAST)
            or (namespace == 'MODULE' and matched[0].opname == 'LOAD_GLOBAL')),
            'operand requires a potentially failing local load, callable binding, or custom namespace access')
        loads.append(dict(span=list(_span(node)), opcode=matched[0].opname, namespace=namespace, name=node.id))
    return loads


def _operands(case, expression, original):
    _require(type(expression) is ast.BinOp and type(expression.op) is ast.Sub and case['operation'] == 'Sub'
        and all(type(node) is ast.Subscript and type(node.value) is ast.Subscript
            and type(node.ctx) is ast.Load and type(node.value.ctx) is ast.Load
            for node in (expression.left, expression.right)), 'source differs from the admitted scalar subtraction')
    left, right = expression.left, expression.right
    first, second = left.value, right.value
    _require(_span(first) == _position(case['first_span']) and _span(second) == _position(case['repeated_span'])
        and all(type(node) is ast.Name for node in (first.value, first.slice, second.value, second.slice))
        and (first.value.id, first.slice.id) == (second.value.id, second.slice.id),
        'source no longer reads the same original table name and row key')
    nodes = [first.value, first.slice, left.slice, second.value, second.slice, right.slice]
    names = case['bindings']
    evidence = [names['table'][0], names['key'][0], names['index'][0],
                names['table'][1], names['key'][1], names['index'][1]]
    loads = _loads(original, nodes, evidence)
    _require(all(row['opcode'] in _FAST or row['opcode'] == 'SOURCE_INT_LITERAL'
                 for row in (loads[1], loads[2], loads[4], loads[5])),
        'speculative key and index reads must be original initialized fast locals or integer literals')
    return [first.value, first.slice, left.slice, right.slice], loads


def _value(expression, operands):
    offer = ast.Call(func=ast.Name(id=OFFER, ctx=ast.Load()), args=deepcopy(operands), keywords=[])
    attempt = ast.NamedExpr(target=ast.Name(id=RESULT, ctx=ast.Store()), value=offer)
    matched = ast.Compare(left=attempt, ops=[ast.IsNot()], comparators=[ast.Name(id=MISS, ctx=ast.Load())])
    return ast.copy_location(ast.IfExp(test=matched, body=ast.Name(id=RESULT, ctx=ast.Load()),
        orelse=expression), expression)


def seal_expression(raw, case):
    """Replay authenticated spans and compile just one same-frame replacement."""
    recipe = case['compile_recipe']
    filename = case['source']['uri']
    _require(recipe['filename'] == filename, 'compile recipe differs from the original source path')
    root = compile_original(raw, recipe)
    codes = tuple(code_tree(root))
    _require(not any(_RESERVED.intersection((*code.co_names, *code.co_varnames, *code.co_cellvars, *code.co_freevars))
        for code in codes), 'reserved scalar table splice names occur in original code')
    originals = [code for code in codes if code_key(code) == case['code_key']]
    _require(len(originals) == 1 and originals[0].co_qualname == case['qualname']
        and not originals[0].co_freevars and not originals[0].co_cellvars,
        'one exact original function without closure cells is required')
    original, = originals
    tree = ast.parse(raw, filename=filename)
    function = _definition(tree, original.co_qualname)
    assignments = [node for node in function.body if type(node) is ast.Assign
        and len(node.targets) == 1 and type(node.targets[0]) is ast.Name
        and _span(node.value) == _position(case['span'])]
    _require(len(assignments) == 1, 'scalar table expression needs one direct original function-body assignment')
    assignment, = assignments
    expression = assignment.value
    operands, loads = _operands(case, expression, original)
    assignment.value = _value(expression, operands)
    revised = compile(ast.fix_missing_locations(tree), filename, 'exec', dont_inherit=True, optimize=recipe['optimize'])
    leaves = [code for code in code_tree(revised) if code.co_qualname == original.co_qualname
        and code.co_firstlineno == original.co_firstlineno]
    _require(len(leaves) == 1, 'sealed scalar table code lost the original function owner')
    leaf, = leaves
    fields = ('co_qualname','co_filename','co_firstlineno','co_argcount','co_posonlyargcount',
              'co_kwonlyargcount','co_flags','co_cellvars','co_freevars')
    _require(all(getattr(leaf, field) == getattr(original, field) for field in fields)
        and set(leaf.co_varnames) == set(original.co_varnames) | {RESULT}
        and set(leaf.co_names) == set(original.co_names) | {OFFER, MISS}, 'scalar splice changed the original callable ABI')
    return dict(source=case['source'], compile_recipe=recipe, qualname=case['qualname'], span=case['span'],
        preserved_loads=loads, original_code_key=case['code_key'], revised_code_key=code_key(leaf),
        operation=case['operation'], miss='RETURN_NOT_IMPLEMENTED_BEFORE_ORIGINAL_EXPRESSION',
        source_obligation=list(case['obligations'])), leaf


def prepare(proof, out, *, case_index=0):
    """Cold-check the F4 rewrite and publish the verified source and code bytes."""
    ref = artifact_reference(Path(proof).resolve(), PROOF)
    report = check(ref)
    _require(type(case_index) is int and 0 <= case_index < len(report['cases']), 'an admitted original expression is required')
    case = report['cases'][case_index]
    raw = read_artifact(case['source'], json_document=False)
    region, leaf = seal_expression(raw, case)
    context = interpreter_context()
    _require(context['optimize'] == region['compile_recipe']['optimize'], 'compiler and source interpreter policies differ')
    directory = Path(out).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    path = directory/'leaf.marshal'
    with path.open('xb') as stream:
        marshal.dump(leaf, stream)
    result = dict(schema=SCHEMA, schema_version=1, proof=ref, proof_case=case_index, region=region,
        code=artifact_reference(path, ENTRY_CODE), interpreter=context,
        execution_scope='EXPLICIT_EXCLUSIVE_EXPRESSION_STUDY', deployment_authorized=False)
    path = directory/'region.json'
    with path.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, SCHEMA)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proof', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--case', type=int, default=0)
    parser.add_argument('--build-native', action='store_true', help='also build the C++20 trial provider')
    args = parser.parse_args(argv)
    region = prepare(args.proof, args.out, case_index=args.case)
    if args.build_native:
        from scar.codegen.scalar_table_native import build_scalar_table

        native = build_scalar_table(Path(args.out) / 'native')
        print(json.dumps(dict(region=region, native=native)))
    else:
        print(json.dumps(region))


if __name__ == '__main__':
    main()
