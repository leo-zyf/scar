"""Normalize a scoped, observed scalar function and reuse immutable row reads."""

import ast
import operator
from pathlib import Path

from scar.compiler.scalar_table_region import _definition, code_tree
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.runtime.scalar_table_contract import compile_original
from scar.trace.code_snapshot import code_key


SCHEMA = 'scar.scalar-score-plan'
_EVIDENCE = 'scar.scalar-score-evidence'
_ARITHMETIC = {ast.Add: 'add', ast.Sub: 'sub', ast.FloorDiv: 'floordiv'}
_COMPARE = {ast.Eq: 'eq', ast.Lt: 'lt', ast.LtE: 'le', ast.In: 'in'}
_LEAF = {'int', 'str', 'input', 'get'}
_LAZY = {'and', 'select'}
_INTEGER_OPS = {'abs': abs, 'add': operator.add, 'sub': operator.sub, 'floordiv': operator.floordiv}
_COMPARISONS = {'eq': operator.eq, 'lt': operator.lt, 'le': operator.le}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _int64(value):
    return type(value) is int and -(1 << 63) <= value < (1 << 63)


def _type(value):
    if _int64(value):
        return 'int'
    if type(value) is str and value.isascii() and '\0' not in value:
        return 'str'
    raise ValueError('scalar inputs require signed-64 integers or ASCII strings without NUL')


def _node(op, kind, **fields):
    return dict(op=op, type=kind, **fields)


def _args(op, result, operands, kinds):
    _require(tuple(row['type'] for row in operands) == kinds, 'unsupported operand types for ' + op)
    return _node(op, result, args=operands)


class _Normalize:
    def __init__(self, function, code, table, fields, positionals, globals_):
        _require(len(function.args.posonlyargs + function.args.args) == 2
            and not (function.args.kwonlyargs or function.args.vararg or function.args.kwarg
                     or function.args.defaults or function.decorator_list or code.co_freevars or code.co_cellvars),
            'scalar boundary requires a plain receiver and one tuple argument')
        self.receiver, self.argument = (arg.arg for arg in function.args.posonlyargs + function.args.args)
        self.table = table
        self.fields = fields
        self.positionals = positionals
        self.globals = globals_
        self.locals = set(code.co_varnames)
        self.serial = 0

    def expr(self, node, bound):
        if type(node) is ast.Constant:
            kind = _type(node.value)
            return _node(kind, kind, value=node.value)
        if type(node) is ast.UnaryOp and type(node.op) is ast.USub and type(node.operand) is ast.Constant:
            value = node.operand.value
            _require(type(value) is int and _int64(-value), 'signed-64 literal required')
            return _node('int', 'int', value=-value)
        if type(node) is ast.Name:
            if node.id in bound:
                return _node('get', bound[node.id], name=node.id)
            _require(node.id not in self.locals and node.id in self.globals and _int64(self.globals[node.id]),
                'unresolved original scalar binding: ' + node.id)
            return _node('int', 'int', value=self.globals[node.id])
        if type(node) is ast.Attribute and type(node.value) is ast.Name and node.value.id == self.receiver:
            _require(node.attr in self.fields, 'receiver field has no observed exact type')
            return _node('input', self.fields[node.attr], name='field_' + node.attr)
        if type(node) is ast.Subscript:
            if type(node.value) is ast.Name and node.value.id == self.table and self.table not in self.locals:
                key = self.expr(node.slice, bound)
                _require(key['type'] == 'str', 'table row needs an exact string key')
                return _node('table_row', 'row', key=key)
            value, index = self.expr(node.value, bound), self.expr(node.slice, bound)
            _require(value['type'] in ('row', 'str') and index['type'] == 'int', 'unsupported scalar item access')
            return _node('item', 'int' if value['type'] == 'row' else 'str', value=value, index=index)
        if type(node) is ast.BinOp and type(node.op) in _ARITHMETIC:
            return _args(_ARITHMETIC[type(node.op)], 'int',
                         [self.expr(node.left, bound), self.expr(node.right, bound)], ('int', 'int'))
        if type(node) is ast.Call and not node.keywords:
            if type(node.func) is ast.Name and node.func.id == 'abs' and 'abs' not in self.locals:
                return _args('abs', 'int', [self.expr(arg, bound) for arg in node.args], ('int',))
            if type(node.func) is ast.Attribute and node.func.attr == 'upper' and not node.args:
                return _args('upper', 'str', [self.expr(node.func.value, bound)], ('str',))
        if type(node) is ast.BoolOp and type(node.op) is ast.And:
            arms = [self.expr(part, bound) for part in node.values]
            return _args('and', 'bool', arms, ('bool',) * len(arms))
        if type(node) is ast.IfExp:
            arms = [self.expr(part, bound) for part in (node.test, node.body, node.orelse)]
            kind = arms[1]['type']
            return _args('select', kind, arms, ('bool', kind, kind))
        if type(node) is ast.Compare:
            args = [self.expr(part, bound) for part in (node.left, *node.comparators)]
            _require(all(part['op'] in _LEAF for part in args[1:-1]),
                'chained comparison needs a safe shared scalar operand')
            parts = []
            for op, left, right in zip(node.ops, args[:-1], args[1:], strict=True):
                kind = _COMPARE.get(type(op))
                _require(kind is not None, 'unsupported scalar comparison')
                operand_type = {'in': 'str', 'eq': left['type']}.get(kind, 'int')
                _require(operand_type in ('int', 'str'), 'only exact scalar equality can be lowered')
                parts.append(_args(kind, 'bool', [left, right], (operand_type, operand_type)))
            return parts[0] if len(parts) == 1 else _node('and', 'bool', args=parts)
        raise ValueError('unsupported scalar expression: ' + type(node).__name__)

    def assign(self, target, value, bound, nested):
        _require(type(target) is ast.Name and target.id not in (self.receiver, self.argument)
            and value['type'] in ('int', 'str', 'bool')
            and (target.id in bound or not nested)
            and bound.get(target.id, value['type']) == value['type'],
            'scalar assignment must preserve its original local type and conditional lifetime')
        bound[target.id] = value['type']
        return dict(op='set', name=target.id, value=value)

    def unpack(self, target, value, bound, nested):
        if type(value) is ast.Name and value.id == self.argument:
            values = [_node('input', kind, name=f'arg_{i}') for i, kind in enumerate(self.positionals)]
        else:
            _require(type(value) is ast.Tuple, 'scalar unpack requires the exact input tuple or a tuple expression')
            values = [self.expr(part, bound) for part in value.elts]
        _require(len(target.elts) == len(values)
            and all(type(item) is ast.Name for item in target.elts), 'scalar tuple unpack extent differs')
        if any(item.id in bound for item in target.elts):
            # Evaluate all original operands before updating any existing binding.
            temps = []
            for operand in values:
                name = f'@unpack_{self.serial}'
                self.serial += 1
                temps.append(dict(op='set', name=name, value=operand))
            values = [_node('get', row['value']['type'], name=row['name']) for row in temps]
        else:
            temps = []
        return temps + [self.assign(item, operand, bound, nested)
                        for item, operand in zip(target.elts, values, strict=True)]

    def block(self, statements, bound, *, nested=False):
        result = []
        for i, node in enumerate(statements):
            if type(node) is ast.Assign and len(node.targets) == 1:
                target, = node.targets
                if type(target) is ast.Tuple:
                    result.extend(self.unpack(target, node.value, bound, nested))
                else:
                    result.append(self.assign(target, self.expr(node.value, bound), bound, nested))
            elif type(node) is ast.AugAssign and type(node.op) in _ARITHMETIC and type(node.target) is ast.Name:
                name = node.target.id
                _require(bound.get(name) == 'int', 'augmented scalar assignment must read an initialized exact int')
                value = _args(_ARITHMETIC[type(node.op)], 'int',
                              [_node('get', 'int', name=name), self.expr(node.value, bound)], ('int', 'int'))
                result.append(self.assign(node.target, value, bound, nested))
            elif type(node) is ast.If:
                test = self.expr(node.test, bound)
                _require(test['type'] == 'bool', 'scalar conditional needs an exact bool')
                result.append({'op':'if', 'site':f'{node.lineno}:{node.col_offset}', 'test':test,
                               'body':self.block(node.body, dict(bound), nested=True),
                               'else':self.block(node.orelse, dict(bound), nested=True)})
            elif type(node) is ast.Return and not nested and i == len(statements)-1:
                value = self.expr(node.value, bound)
                _require(value['type'] == 'int', 'detached scalar result must be an exact integer')
                result.append(dict(op='return', value=value))
            else:
                raise ValueError('unsupported scalar statement: ' + type(node).__name__)
        return result


def _children(expr):
    if expr['op'] == 'table_row':
        return (expr['key'],)
    if expr['op'] == 'item':
        return (expr['value'], expr['index'])
    return expr.get('args', ())


def _first_evaluation(expr):
    if expr['op'] in _LEAF:
        return None
    if expr['op'] not in _LAZY:
        for child in _children(expr):
            first = _first_evaluation(child)
            if first is not None:
                return first
    return expr


def _replace_row(expr, row, replacement):
    if expr == row:
        return replacement, 1
    if expr['op'] in _LAZY:
        return expr, 0
    values, count = [], 0
    for child in _children(expr):
        value, uses = _replace_row(child, row, replacement)
        values.append(value)
        count += uses
    if not count:
        return expr, 0
    if expr['op'] == 'item':
        return dict(expr, value=values[0], index=values[1]), count
    if expr['op'] == 'table_row':
        return dict(expr, key=values[0]), count
    return dict(expr, args=values), count


def _writes_key(stmt, row):
    key = row['key']
    return key['op'] == 'get' and stmt['op'] == 'set' and stmt['name'] == key['name']


def _later_read(block, index, row):
    if _writes_key(block[index], row):
        return False
    for stmt in block[index+1:]:
        if stmt['op'] == 'if':
            break
        if _replace_row(stmt['value'], row, row)[1]:
            return True
        if _writes_key(stmt, row):
            break
    return False


def _simplify(block, replacements):
    result, available = [], []
    for index, stmt in enumerate(block):
        if stmt['op'] == 'if':
            result.append({**stmt, 'body':_simplify(stmt['body'], replacements),
                           'else':_simplify(stmt['else'], replacements)})
            available.clear()
            continue
        value = stmt['value']
        for previous, cached, record in available:
            value, uses = _replace_row(value, previous, cached)
            record['proven_uses'] += uses
            record['removed_row_reads'] += uses
        stmt = dict(stmt, value=value)
        first = _first_evaluation(stmt['value'])
        if first is not None and first['op'] == 'table_row':
            name = f'@row_{len(replacements)}'
            cached = _node('get', 'row', name=name)
            value, uses = _replace_row(stmt['value'], first, cached)
            later = _later_read(block, index, first)
            if uses > 1 or later:
                result.append(dict(op='set', name=name, value=first))
                record = dict(row=name, proven_uses=uses, removed_row_reads=uses-1)
                replacements.append(record)
                if later:
                    available.append((first, cached, record))
                stmt = dict(stmt, value=value)
        result.append(stmt)
        available = [entry for entry in available if not _writes_key(stmt, entry[0])]
    return result


def _evaluate(expr, inputs, values, tables):
    op = expr['op']
    if op == 'input':
        return inputs[expr['name']]
    if op == 'get':
        return values[expr['name']]
    if op in ('int', 'str'):
        return expr['value']
    if op == 'table_row':
        return tables[_evaluate(expr['key'], inputs, values, tables)]
    if op == 'item':
        return _evaluate(expr['value'], inputs, values, tables)[_evaluate(expr['index'], inputs, values, tables)]
    args = expr['args']
    if op == 'and':
        return all(_evaluate(arg, inputs, values, tables) for arg in args)
    if op == 'select':
        return _evaluate(args[1] if _evaluate(args[0], inputs, values, tables) else args[2], inputs, values, tables)
    data = [_evaluate(arg, inputs, values, tables) for arg in args]
    if op == 'upper':
        return data[0].upper()
    if op == 'in':
        return data[0] in data[1]
    if op in _COMPARISONS:
        return _COMPARISONS[op](*data)
    value = _INTEGER_OPS[op](*data)
    _require(_int64(value), 'captured arithmetic exceeds the detached signed-64 contract')
    return value


def _run(block, inputs, table, values, branches=None):
    for stmt in block:
        op = stmt['op']
        if op == 'if':
            arm = 'body' if _evaluate(stmt['test'], inputs, values, table) else 'else'
            if branches is not None:
                branches[stmt['site']] = 'THEN' if arm == 'body' else 'ELSE' if stmt['else'] else 'FALSE'
            _run(stmt[arm], inputs, table, values, branches)
        elif op == 'set':
            values[stmt['name']] = _evaluate(stmt['value'], inputs, values, table)
        else:
            return _evaluate(stmt['value'], inputs, values, table)
    return None


def plan(evidence_path):
    from scar.analysis.scalar_score_evidence import check as check_evidence

    evidence_ref = artifact_reference(Path(evidence_path).resolve(strict=True), _EVIDENCE)
    evidence = check_evidence(evidence_ref['uri'])
    _require(evidence == read_artifact(evidence_ref), 'source function evidence changed during normalization')
    capture_ref = evidence['capture']
    capture = read_artifact(capture_ref)
    proof = read_artifact(evidence['proof'])
    case = proof['cases'][capture['case_index']]
    source = case['source']
    _require(capture['source'] == source and capture['case'] == evidence['proof']
        and capture['builtin_abs_verified'] is True, 'score functions require their exact observed source and builtin binding')
    raw = read_artifact(source, json_document=False)
    codes = [code for code in code_tree(compile_original(raw, case['compile_recipe'])) if code_key(code) == case['code_key']]
    _require(len(codes) == 1, 'score function is ambiguous in original source')
    code, = codes
    function = _definition(ast.parse(raw, filename=source['uri']), code.co_qualname)
    table = case['bindings']['table'][0]['namespace']['name']
    samples = capture['samples']
    tables = capture['table_versions']
    _require(samples and len(tables) >= 1, 'score function needs original exact input and table witnesses')
    fields = {field:_type(value) for field, value in sorted(samples[0]['receiver_fields'].items())}
    positionals = tuple(map(_type, samples[0]['argument']))
    inputs = [dict(name='field_'+name, type=kind) for name,kind in fields.items()]
    inputs += [dict(name=f'arg_{i}', type=kind) for i,kind in enumerate(positionals)]
    original = _Normalize(function, code, table, fields, positionals, capture['globals']).block(function.body, {})
    _require(original and original[-1]['op'] == 'return', 'score function must end in a normal exact integer result')
    reuse = []
    simplified = _simplify(original, reuse)
    observations = []
    for row in samples:
        receiver, args, version = row['receiver_fields'], row['argument'], row['table_version']
        _require({field:_type(value) for field,value in receiver.items()} == fields
            and tuple(map(_type, args)) == positionals and version in tables and _int64(row['result']),
            'score sample disagrees with the original input types or table version')
        values = {'field_'+name: value for name,value in receiver.items()}
        values.update({f'arg_{i}': value for i,value in enumerate(args)})
        rows = tables[version]['rows']
        original_arms, simplified_arms = {}, {}
        original_output = _run(original, values, rows, {}, original_arms)
        simplified_output = _run(simplified, values, rows, {}, simplified_arms)
        observed_arms = {arm['id']:arm['arm'] for arm in row['branch_arms'] if arm['arm'] != 'NOT_REACHED'}
        _require(original_output == simplified_output == row['result'],
            'original output disagrees with the original or simplified scalar IR')
        _require(original_arms == simplified_arms == observed_arms,
            'conditional choices disagree with the original execution')
        observations.append(dict(table_version=version, inputs=values, result=row['result']))
    return dict(schema=SCHEMA, schema_version=1, evidence=evidence_ref, capture=capture_ref,
        source=source, code_key=case['code_key'], inputs=inputs, original_body=original, body=simplified,
        table_versions=tables, samples=observations, row_reuse=reuse,
        scope='DETACHED_EXACT_SCALAR_ON_EXPLICIT_TABLE_VERSION', deployment_authorized=False)


def check(path):
    reference = artifact_reference(Path(path).resolve(strict=True), SCHEMA)
    expected = read_artifact(reference)
    _require(expected == plan(expected['evidence']['uri']), 'scalar lowering differs from cold graph replay')
    return expected
