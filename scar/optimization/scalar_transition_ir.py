"""Compose a scalar function with a checked immutable-record transition."""

import ast
import operator
from pathlib import Path

from scar.compiler.scalar_table_region import _definition, code_tree
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import scalar_score_ir as scalar
from scar.runtime.scalar_table_contract import compile_original
from scar.trace.code_snapshot import describe_code


SCHEMA = 'scar.scalar-transition-plan'
_EVIDENCE = 'scar.scalar-transition-evidence'
_KINDS = {'EXACT_ASCII':'str', 'EXACT_I64':'int', 'EXACT_BOOL':'bool', 'EXACT_BOOL_TUPLE_2':'bool_pair'}
_PAIR = 'bool_pair'
_node = scalar._node
_args = scalar._args
_require = scalar._require


def _type(value):
    if type(value) is bool:
        return 'bool'
    if type(value) is list and len(value) == 2 and all(type(bit) is bool for bit in value):
        return _PAIR
    return scalar._type(value)


def _names(field, kind):
    name = 'field_' + field
    return (name+'__0', name+'__1') if kind == _PAIR else (name,)


def _interface(fields):
    return [dict(name=name, type='bool' if kind == _PAIR else kind)
            for field,kind in fields.items() for name in _names(field, kind)]


def _flatten(fields, values):
    _require(fields == {name:_type(value) for name,value in values.items()}, 'observed state field type or name changed')
    result = {}
    for field, kind in fields.items():
        value = values[field]
        result.update(zip(_names(field, kind), value if kind == _PAIR else (value,), strict=True))
    return result


def _input_fields(fields):
    result = {}
    for field, kind in fields.items():
        names = _names(field, kind)
        if kind == _PAIR:
            result[field] = _node('pair', _PAIR, args=[_node('input', 'bool', name=name) for name in names])
        else:
            result[field] = _node('input', kind, name=names[0])
    return result


def _int(value):
    return _node('int', 'int', value=value)


def _slice(value, start, stop):
    _require(value['type'] == 'str' and all(bound is None or bound['type'] == 'int' for bound in (start, stop)),
        'string slices require an exact ASCII operand and signed-64 bounds')
    return _node('slice', 'str', value=value, start=start, stop=stop)


class _Normalize(scalar._Normalize):
    assignment_types = (*scalar._Normalize.assignment_types, _PAIR)

    def __init__(self, function, code, field_values, positionals, globals_, *, defaults=False):
        super().__init__(function, code, None, {}, positionals, globals_, allow_defaults=defaults)
        self.field_values = field_values
        self.aliases = {}
        self.helpers = {}
        self.score = None

    def test(self, node, bound):
        if type(node) is ast.BoolOp and type(node.op) is ast.And:
            return _node('and', 'bool', args=[self.test(child, bound) for child in node.values])
        value = self.expr(node, bound)
        if value['type'] == 'int':
            return _args('ne', 'bool', [value, _int(0)], ('int','int'))
        _require(value['type'] == 'bool', 'transition condition needs an exact int or bool')
        return value

    def assign(self, target, value, bound, nested):
        _require(not isinstance(target, ast.Name) or target.id not in self.helpers,
            'original lexical helper must keep its binding throughout the transition')
        return super().assign(target, value, bound, nested)

    def _inline(self, node, bound):
        helper = self.helpers[node.func.id]
        params = [arg.arg for arg in helper.args.posonlyargs + helper.args.args]
        _require(len(params) == len(node.args), 'lexical helper argument extent differs')
        values = [self.expr(operand, bound) for operand in node.args]
        inside = {param:value['type'] for param,value in zip(params, values, strict=True)}
        result = self.expr(helper.body, inside)
        bindings = [dict(name=name, value=value) for name,value in zip(params, values, strict=True)]
        return _node('let', result['type'], bindings=bindings, result=result)

    def expr(self, node, bound):
        if type(node) is ast.Constant and type(node.value) is bool:
            return _node('bool', 'bool', value=node.value)
        if type(node) is ast.Name and node.id in self.aliases and node.id not in bound:
            return self.aliases[node.id]
        if (type(node) is ast.Attribute and type(node.value) is ast.Name
                and node.value.id == self.receiver and self.receiver not in bound):
            _require(node.attr in self.field_values, 'transition receiver field has no admitted tuple slot')
            return self.field_values[node.attr]
        if type(node) is ast.Tuple:
            return _args('pair', _PAIR, [self.expr(item, bound) for item in node.elts], ('bool','bool'))
        if type(node) is ast.Subscript:
            value = self.expr(node.value, bound)
            if type(node.slice) is ast.Slice:
                axis = node.slice
                if axis.step is not None:
                    _require(axis.lower is None and axis.upper is None
                        and self.expr(axis.step, bound) == _int(-1), 'only full ASCII reverse supports a slice step')
                    return _args('reverse', 'str', [value], ('str',))
                lower = None if axis.lower is None else self.expr(axis.lower, bound)
                upper = None if axis.upper is None else self.expr(axis.upper, bound)
                return _slice(value, lower, upper)
            index = self.expr(node.slice, bound)
            _require(value['type'] in ('str', _PAIR) and index['type'] == 'int',
                'transition item read needs an exact string or bool pair and an int index')
            return _node('item', 'str' if value['type'] == 'str' else 'bool', value=value, index=index)
        if type(node) is ast.UnaryOp:
            if type(node.op) is ast.Not:
                return _args('not', 'bool', [self.test(node.operand, bound)], ('bool',))
            if type(node.op) is ast.USub and type(node.operand) is not ast.Constant:
                return _args('neg', 'int', [self.expr(node.operand, bound)], ('int',))
        if type(node) is ast.BinOp and type(node.op) in (ast.Add, ast.Mult):
            values = [self.expr(node.left, bound), self.expr(node.right, bound)]
            if type(node.op) is ast.Add and all(value['type'] == 'str' for value in values):
                return _args('concat', 'str', values, ('str','str'))
            return _args('add' if type(node.op) is ast.Add else 'mul', 'int', values, ('int','int'))
        if type(node) is ast.Compare and len(node.ops) == 1 and type(node.ops[0]) is ast.NotEq:
            left, right = self.expr(node.left, bound), self.expr(node.comparators[0], bound)
            _require(left['type'] in ('int','bool'), 'transition inequality must use exact ints or bools')
            return _args('ne', 'bool', [left,right], (left['type'],left['type']))
        if type(node) is ast.Call and not node.keywords:
            if type(node.func) is ast.Name and node.func.id in self.helpers:
                return self._inline(node, bound)
            if type(node.func) is ast.Attribute:
                target = node.func
                if target.attr == 'swapcase' and not node.args:
                    return _args('swapcase', 'str', [self.expr(target.value, bound)], ('str',))
                if (self.score is not None and target.attr == self.score[0]
                        and type(target.value) is ast.Name and target.value.id == self.receiver
                        and len(node.args) == 1 and type(node.args[0]) is ast.Name and node.args[0].id == self.argument):
                    return _node('score_call', 'int', args=self.score[1])
        return super().expr(node, bound)


def _original_body(function, normalizer):
    _require(function.body and type(function.body[-1]) is ast.Return, 'transition needs its original final return')
    statements = []
    for index, stmt in enumerate(function.body[:-1]):
        if (index == 0 and type(stmt) is ast.Expr and type(stmt.value) is ast.Constant
                and type(stmt.value.value) is str):
            continue
        if (type(stmt) is ast.Assign and type(stmt.value) is ast.Lambda
                and len(stmt.targets) == 1 and type(stmt.targets[0]) is ast.Name):
            name = stmt.targets[0].id
            helper = stmt.value
            params = {param.arg for param in helper.args.posonlyargs + helper.args.args}
            _require(not (helper.args.kwonlyargs or helper.args.defaults or helper.args.kwarg or helper.args.vararg)
                and name not in (*normalizer.helpers, normalizer.receiver, normalizer.argument)
                and all(node.id in params for node in ast.walk(helper.body) if type(node) is ast.Name)
                and all(node.lineno >= stmt.lineno for node in ast.walk(function) if type(node) is ast.Name
                        and type(node.ctx) is ast.Load and node.id == name),
                'lexical transition helper must be capture-free and available before use')
            normalizer.helpers[name] = helper
        else:
            statements.append(stmt)
    bound = {}
    return normalizer.block(statements, bound), bound, function.body[-1].value


def _constructor(node, name, fields, normalizer, bound):
    _require(type(node) is ast.Call and type(node.func) is ast.Name and node.func.id == name
        and not node.keywords and len(node.args) == len(fields), 'exact original positional tuple constructor required')
    values = [normalizer.expr(arg,bound) for arg in node.args]
    _require(tuple(fields.values()) == tuple(value['type'] for value in values),
        'transition constructor must preserve its original field types and order')
    return values


def _materialize(values, label):
    statements, references = [], []
    for i,value in enumerate(values):
        name = f'@{label}_{i}'
        statements.append(dict(op='set', name=name, value=value))
        references.append(_node('get', value['type'], name=name))
    return statements, references


def _compose(function, rotate, code, rotate_code, constructor, fields, positionals, globals_, score, default):
    transition = _Normalize(function, code, _input_fields(fields), positionals, globals_)
    transition.score = score
    body, bound, final = _original_body(function, transition)
    _require(type(final) is ast.Call and type(final.func) is ast.Attribute and not (final.args or final.keywords)
        and final.func.attr == rotate.name, 'transition must return its selected exact method on the constructed record')
    intermediate, receiver = _materialize(_constructor(final.func.value, constructor, fields, transition, bound), 'state')
    values = dict(zip(fields, receiver, strict=True))
    rotation = _Normalize(rotate, rotate_code, values, (), globals_, defaults=True)
    defaults = rotate.args.defaults
    _require(len(defaults) == 1 and type(defaults[0]) is ast.Constant
        and type(defaults[0].value) is bool and defaults[0].value is default,
        'exact original rotation default must match its observed argument')
    rotation.aliases[rotation.argument] = _node('bool','bool',value=default)
    steps, rotate_bound, returned = _original_body(rotate, rotation)
    _require(not steps, 'rotation must directly return its original tuple fields')
    result, output = _materialize(_constructor(returned, constructor, fields, rotation, rotate_bound), 'return')
    leaves = []
    for kind,value in zip(fields.values(), output, strict=True):
        if kind == _PAIR:
            leaves.extend(_node('item','bool',value=value,index=_int(i)) for i in (0,1))
        else:
            leaves.append(value)
    return [*body, *intermediate, *result, dict(op='return', values=leaves)]


def _splice(expr):
    """Find a bounded lexical concat-of-slices over one already-bound string."""
    node = expr.get('result')
    if expr['op'] != 'let' or node['op'] != 'concat' or node['args'][0]['op'] != 'concat':
        return None
    prefix,replacement = node['args'][0]['args']
    suffix = node['args'][1]
    if (prefix['op'] != 'slice' or suffix['op'] != 'slice' or prefix['start'] is not None
            or suffix['stop'] is not None or prefix['value'] != suffix['value']
            or prefix['stop'] is None or suffix['start'] is None):
        return None
    text,index = prefix['value'],prefix['stop']
    if (any(leaf['op'] != 'get' for leaf in (text,index,replacement))
            or suffix['start'] != _args('add','int',[index,_int(1)],('int','int'))):
        return None
    return dict(expr, result=_args('splice','str',[text,index,replacement],('str','int','str')))


def _simplify(value, rewrites):
    if type(value) is list:
        return [_simplify(row,rewrites) for row in value]
    if type(value) is not dict:
        return value
    if value.get('op') == 'let':
        replacement = _splice(value)
        if replacement is not None:
            rewrites.append(dict(rule='FUSE_ASCII_PREFIX_INSERT_SUFFIX', retained_order='BOUND_OPERANDS_EVALUATED_ONCE'))
            value = replacement
    return {key:_simplify(row,rewrites) for key,row in value.items()}


_INTEGER = {**scalar._INTEGER_OPS, 'neg':operator.neg, 'mul':operator.mul}
_COMPARE = {**scalar._COMPARISONS, 'ne':operator.ne}


def _evaluate(expr, inputs, values, table, score_plan, deltas):
    def evaluate(node, scope=values):
        return _evaluate(node,inputs,scope,table,score_plan,deltas)

    op = expr['op']
    if op in ('int','str','bool'):
        return expr['value']
    if op == 'input':
        return inputs[expr['name']]
    if op == 'get':
        return values[expr['name']]
    if op == 'item':
        return evaluate(expr['value'])[evaluate(expr['index'])]
    if op == 'slice':
        text = evaluate(expr['value'])
        lower = None if expr['start'] is None else evaluate(expr['start'])
        upper = None if expr['stop'] is None else evaluate(expr['stop'])
        return text[lower:upper]
    if op == 'let':
        fields = {binding['name']:evaluate(binding['value']) for binding in expr['bindings']}
        return evaluate(expr['result'], {**values, **fields})
    args = expr['args']
    if op == 'and':
        return all(evaluate(arg) for arg in args)
    if op == 'select':
        return evaluate(args[1] if evaluate(args[0]) else args[2])
    operands = [evaluate(arg) for arg in args]
    if op in _COMPARE:
        return _COMPARE[op](*operands)
    if op in _INTEGER:
        result = _INTEGER[op](*operands)
        _require(scalar._int64(result), 'transition arithmetic exceeds the signed-64 contract')
        return result
    if op == 'score_call':
        arguments = {row['name']:operand for row,operand in zip(score_plan['inputs'],operands,strict=True)}
        result = scalar._run(score_plan['body'], arguments, table, {})
        deltas.append(result)
        return result
    if op == 'splice':
        text,index,replacement = operands
        _require(scalar._int64(index+1), 'transition slice increment exceeds the signed-64 contract')
        return text[:index] + replacement + text[index+1:]
    if op == 'in':
        return operands[0] in operands[1]
    if op == 'pair':
        return operands
    return {'not':operator.not_, 'upper':str.upper, 'swapcase':str.swapcase,
            'reverse':lambda text:text[::-1], 'concat':operator.add}[op](*operands)


def _run(block, inputs, table, score, values, deltas):
    def evaluate(node):
        return _evaluate(node,inputs,values,table,score,deltas)

    for stmt in block:
        if stmt['op'] == 'if':
            arm = 'body' if evaluate(stmt['test']) else 'else'
            _run(stmt[arm],inputs,table,score,values,deltas)
        elif stmt['op'] == 'set':
            values[stmt['name']] = evaluate(stmt['value'])
        else:
            return [evaluate(node) for node in stmt['values']]
    return None


def _ref(path,schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def plan(evidence_path, score_path):
    from scar.analysis.scalar_transition_evidence import check as check_evidence

    evidence_ref = _ref(evidence_path,_EVIDENCE)
    evidence = check_evidence(evidence_ref['uri'])
    score_ref = _ref(score_path,scalar.SCHEMA)
    score_plan = scalar.check(score_ref['uri'])
    _require(evidence == read_artifact(evidence_ref) and evidence['score_evidence'] == score_plan['evidence'],
        'transition and score must share their admitted source proof')
    capture_ref = evidence['capture']
    capture = read_artifact(capture_ref)
    score_evidence = read_artifact(score_plan['evidence'])
    score_capture = read_artifact(score_evidence['capture'])
    case = read_artifact(score_evidence['proof'])['cases'][score_capture['case_index']]
    source = evidence['source']
    _require(source == score_plan['source'] and capture['builtin_abs_verified']
        and all(capture['globals'].get(name)==value for name,value in score_capture['globals'].items()),
        'transition needs the exact original source and score scalar context')
    raw = read_artifact(source,json_document=False)
    codes = {describe_code(code)['key']:code for code in code_tree(compile_original(raw,case['compile_recipe']))}
    tree = ast.parse(raw,filename=source['uri'])
    code,rotation = (codes[evidence[role]['code']['key']] for role in ('callee','rotation'))
    _require(describe_code(code)==evidence['callee']['code'] and describe_code(rotation)==evidence['rotation']['code'],
        'transition or rotation source identity changed')
    function,rotate = (_definition(tree, body.co_qualname) for body in (code,rotation))
    fields = {row['name']:_KINDS[row['kind']] for row in capture['state_fields']}
    _require(list(fields)==evidence['constructor']['field_order'], 'original constructor field order differs')
    samples = capture['samples']
    positionals = tuple(map(_type,samples[0]['argument']))
    _require(all(kind in ('int','bool','str') for kind in positionals), 'transition tuple argument must have primitive leaves')
    inputs = _interface(fields)+[dict(name=f'arg_{i}',type=kind) for i,kind in enumerate(positionals)]
    outputs = _interface(fields)
    input_types = {row['name']:row['type'] for row in inputs}
    _require(len(input_types)==len(inputs)
        and all(input_types.get(row['name'])==row['type'] for row in score_plan['inputs']),
        'score operands must belong to the exact selected transition input')
    score = (score_evidence['callee']['code']['qualname'].rsplit('.',1)[-1],
             [_node('input',row['type'],name=row['name']) for row in score_plan['inputs']])
    default = samples[0]['rotation_arg']
    _require(type(default) is bool, 'exact bool rotation default required')
    original = _compose(function,rotate,code,rotation,evidence['constructor']['name'],fields,
                        positionals,capture['globals'],score,default)
    rewrites = []
    body = _simplify(original,rewrites)
    tables = capture['table_versions']
    _require(tables.keys()==score_plan['table_versions'].keys()
        and all(value['rows']==score_plan['table_versions'][version]['rows'] for version,value in tables.items()),
        'transition table contents must match both admitted score versions')
    verified = []
    for sample in samples:
        version = sample['table_version']
        args = sample['argument']
        _require(tuple(map(_type,args))==positionals and sample['rotation_arg'] is default,
            'transition sample input types or rotation default changed')
        arguments = _flatten(fields,sample['receiver_fields'])
        arguments.update({f'arg_{i}':arg for i,arg in enumerate(args)})
        expected = _flatten(fields,sample['returned_fields'])
        for graph in (original,body):
            state,deltas = {},[]
            result = _run(graph,arguments,tables[version]['rows'],score_plan,state,deltas)
            _require(deltas==[sample['score_delta']]
                and {field:state[f'@state_{i}'] for i,field in enumerate(fields)}==sample['intermediate_fields']
                and result==list(expected.values()), 'original transition disagrees with typed score, intermediate, or return')
        verified.append(dict(table_version=version,inputs=arguments,outputs=expected))
    return dict(schema=SCHEMA,schema_version=1,evidence=evidence_ref,capture=capture_ref,score_plan=score_ref,
        source=source,inputs=inputs,outputs=outputs,original_body=original,body=body,table_versions=tables,samples=verified,
        rewrites=rewrites,scope='DETACHED_IMMUTABLE_RECORD_TRANSITION_WITH_EXPLICIT_TABLE_VERSION',deployment_authorized=False)


def check(path):
    restored = read_artifact(_ref(path,SCHEMA))
    _require(restored==plan(restored['evidence']['uri'],restored['score_plan']['uri']),
        'transition differs from cold source and typed state replay')
    return restored
