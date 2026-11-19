"""Form an eager typed region from a sorted scalar generator and its scorer."""

import ast
from pathlib import Path

from scar.analysis import scalar_score_reuse_evidence as evidence
from scar.compiler.scalar_table_region import _definition
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import scalar_score_ir as scalar
from scar.optimization import scalar_transition_ir as transition
from scar.trace.code_snapshot import describe_code


SCHEMA = 'scar.scalar-sorted-region'
CAPTURE = 'scar.scalar-sorted-source-capture'
_node = scalar._node
_args = scalar._args
_require = scalar._require


def _literal(node, constants):
    if type(node) is ast.Name and node.id in constants:
        return constants[node.id]
    if type(node) is ast.Tuple:
        return tuple(_literal(value, constants) for value in node.elts)
    if type(node) is ast.Dict:
        keys = [_literal(key, constants) for key in node.keys]
        _require(len(keys) == len(set(keys)), 'literal map keys must be unique')
        return dict(zip(keys, (_literal(value, constants) for value in node.values), strict=True))
    if type(node) is ast.BinOp and type(node.op) is ast.Add:
        a, b = (_literal(child, constants) for child in (node.left, node.right))
        _require(scalar._int64(a) and scalar._int64(b) and scalar._int64(a + b),
                 'literal folds require signed-64 addition')
        return a + b
    return ast.literal_eval(node)


def _defaults(module):
    constants = {}
    for stmt in module.body:
        if type(stmt) is not ast.Assign or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        words = target.elts if type(target) is ast.Tuple else [target]
        if any(type(word) is not ast.Name for word in words):
            continue
        try:
            value = _literal(stmt.value, constants)
        except (ValueError, TypeError):
            for word in words:
                constants.pop(word.id, None)
            continue
        values = value if type(target) is ast.Tuple else (value,)
        if type(values) is tuple and len(values) == len(words):
            constants.update((word.id, item) for word, item in zip(words, values, strict=True))
    return constants


class _Generator(transition._Normalize):
    def __init__(self, function, code, fields, defaults, move_types, imports, constructors):
        parameters = function.args
        arguments = parameters.posonlyargs + parameters.args
        _require(len(arguments) == 1 and not (parameters.vararg or parameters.kwarg
                 or parameters.kwonlyargs or parameters.defaults or function.decorator_list
                 or code.co_freevars), 'generator requires an exact receiver-only binding')
        self.receiver = arguments[0].arg
        self.argument = None
        self.field_values = transition._input_fields(fields)
        self.defaults = defaults
        self.globals = {}
        self.mappings = {}
        self.fields = {}
        self.aliases = {}
        self.helpers = {}
        self.locals = set(code.co_varnames) | set(code.co_cellvars)
        self.table = None
        self.score = None
        self.move_types = move_types
        self.imports = imports
        self.constructors = constructors
        self.constructor = None
        self.serial = 0

    def expr(self, node, bound):
        if type(node) is ast.Name and node.id not in bound and node.id not in self.locals:
            _require(scalar._int64(self.defaults.get(node.id)), 'generator global needs an exact signed-64 binding')
            self.globals[node.id] = self.defaults[node.id]
        if type(node) is ast.BoolOp and type(node.op) is ast.Or:
            return _node('or', 'bool', args=[self.test(child, bound) for child in node.values])
        if type(node) is ast.Compare and len(node.ops) == 1:
            op, right = node.ops[0], node.comparators[0]
            if type(op) is ast.Gt:
                return _args('gt', 'bool', [self.expr(child, bound) for child in (node.left, right)], ('int', 'int'))
            if type(op) is ast.NotEq:
                operands = [self.expr(child, bound) for child in (node.left, right)]
                kind = operands[0]['type']
                _require(kind in ('int', 'str', 'bool'), 'unsupported generator inequality')
                return _args('ne', 'bool', operands, (kind, kind))
            if type(op) in (ast.In, ast.NotIn):
                left = self.expr(node.left, bound)
                if type(right) is ast.Tuple:
                    operands = [left, *(self.expr(part, bound) for part in right.elts)]
                    _require(len(operands) > 1 and all(part['type'] == 'int' for part in operands),
                             'generator tuple membership needs exact integers')
                    predicate = _node('contains_int', 'bool', args=operands)
                else:
                    predicate = _args('in', 'bool', [left, self.expr(right, bound)], ('str', 'str'))
                return _args('not', 'bool', [predicate], ('bool',)) if type(op) is ast.NotIn else predicate
        return super().expr(node, bound)

    def _yield(self, node, bound):
        _require(type(node) is ast.Call and type(node.func) is ast.Name and not node.keywords
                 and node.func.id in self.constructors and node.func.id not in self.locals
                 and (self.constructor is None or node.func.id == self.constructor),
                 'generator emit needs one declared exact tuple constructor')
        self.constructor = node.func.id
        values = [self.expr(arg, bound) for arg in node.args]
        _require(tuple(value['type'] for value in values) == self.move_types, 'emitted record types differ from scorer ports')
        return dict(op='emit', values=values)

    def _loop(self, target, iterator, bound):
        if (type(iterator) is ast.Call and type(iterator.func) is ast.Name
                and iterator.func.id not in self.locals and not iterator.keywords):
            if iterator.func.id == 'enumerate' and len(iterator.args) == 1 and type(target) is ast.Tuple:
                names = target.elts
                _require(len(names) == 2 and all(type(word) is ast.Name for word in names),
                         'enumerate requires index and value binders')
                value = self.expr(iterator.args[0], bound)
                _require(value['type'] == 'str', 'enumerate expects an exact ASCII flow')
                return dict(iterator='enumerate_ascii', names=[word.id for word in names],
                            args=[value]), ('int', 'str')
            if self.imports.get(iterator.func.id) == ('itertools', 'count') and len(iterator.args) == 2:
                args = [self.expr(child, bound) for child in iterator.args]
                _require(type(target) is ast.Name and all(row['type'] == 'int' for row in args),
                         'count requires signed-64 start, step and a single binder')
                return dict(iterator='integer_count', names=[target.id], args=args), ('int',)
        _require(type(target) is ast.Name, 'expected one scalar iterator binder')
        if type(iterator) is ast.Subscript and type(iterator.value) is ast.Name:
            name = iterator.value.id
            values = self.defaults.get(name)
            key = self.expr(iterator.slice, bound)
            _require(name not in self.locals and type(values) is dict and key['type'] == 'str'
                     and all(type(k) is str and k.isascii() and '\0' not in k
                         and type(v) is tuple and all(scalar._int64(item) for item in v)
                         for k, v in values.items()), 'iterator map needs exact ASCII keys and immutable integer rows')
            self.mappings[name] = values
            return dict(iterator='integer_row', names=[target.id], args=[key], mapping=name), ('int',)
        value = self.expr(iterator, bound)
        _require(value['type'] == 'str', 'scalar iterator requires exact ASCII text')
        return dict(iterator='ascii', names=[target.id], args=[value]), ('str',)

    def _for(self, target, iterator, body, bound, depth, *, expression=False):
        desc, types = self._loop(target, iterator, bound)
        names = desc['names']
        _require(len(names) == len(set(names)) and all(name not in bound and name != self.receiver for name in names),
                 'scalar loop binder overlaps an existing value')
        scope = {**bound, **dict(zip(names, types, strict=True))}
        desc['body'] = [self._yield(body, scope)] if expression else self.block(body, scope, depth + 1)
        return dict(op='for', **desc)

    def block(self, statements, bound, depth=0):
        result = []
        for stmt in statements:
            if type(stmt) is ast.For:
                _require(not stmt.orelse, 'loop else has no admitted lowering')
                row = self._for(stmt.target, stmt.iter, stmt.body, bound, depth)
            elif type(stmt) is ast.Assign and len(stmt.targets) == 1:
                row = self.assign(stmt.targets[0], self.expr(stmt.value, bound), bound, False)
            elif type(stmt) is ast.If:
                row = dict(op='if', test=self.test(stmt.test, bound),
                           body=self.block(stmt.body, dict(bound), depth),
                           **{'else': self.block(stmt.orelse, dict(bound), depth)})
            elif type(stmt) in (ast.Break, ast.Continue):
                _require(depth > 0, 'loop transfer requires an enclosing original loop')
                row = dict(op=type(stmt).__name__.lower())
            elif type(stmt) is ast.Expr and type(stmt.value) is ast.Yield:
                row = self._yield(stmt.value.value, bound)
            elif type(stmt) is ast.Expr and type(stmt.value) is ast.YieldFrom:
                comp = stmt.value.value
                _require(type(comp) is ast.GeneratorExp and len(comp.generators) == 1,
                         'yield delegation requires a single generator expression')
                part, = comp.generators
                _require(not part.is_async and not part.ifs, 'yield delegation must be an unfiltered synchronous iterator')
                row = self._for(part.target, part.iter, comp.elt, bound, depth, expression=True)
            else:
                raise ValueError('unsupported scalar generator statement: ' + type(stmt).__name__)
            result.append(row)
        return result


def _calls(module, code_map, selection, arm):
    caller = _definition(module, selection['caller']['qualname'])
    gen_name = selection['generator']['qualname'].rsplit('.', 1)[-1]
    gen = evidence._one((node for node in caller.body if type(node) is ast.FunctionDef and node.name == gen_name),
                        'original generator declaration changed')
    sorted_call = evidence._one((node for node in ast.walk(gen) if type(node) is ast.Call
                                and evidence._span(node) == arm['sorted_call']['position']), 'original sort call changed')
    comp, = sorted_call.args
    source, = comp.generators
    bind, = source.ifs
    binding = bind.values[0] if type(bind) is ast.BoolOp and type(bind.op) is ast.Or else bind
    _require(type(binding) is ast.Compare and len(binding.ops) == 1 and type(binding.ops[0]) is ast.GtE
             and type(binding.left) is ast.NamedExpr and type(binding.left.target) is ast.Name
             and type(binding.comparators[0]) is ast.Name and not source.is_async,
             'expected score binding followed by signed-64 filter')
    pair = evidence._tuple_names(comp.elt)
    _require(pair == (binding.left.target.id, evidence._name(source.target))
             and evidence._span(binding.left.value) == arm['score_call']['position'] and
             len(sorted_call.keywords) == 1 and sorted_call.keywords[0].arg == 'reverse'
             and type(sorted_call.keywords[0].value) is ast.Constant
             and type(sorted_call.keywords[0].value.value) is bool and
             type(source.iter) is ast.Call and type(source.iter.func) is ast.Attribute
             and not (source.iter.args or source.iter.keywords), 'unsupported sorted record flow')
    fallback = None
    if binding is not bind:
        _require(len(bind.values) == 2 and type(bind.values[1]) is ast.Name, 'filter alternative must be one scalar capture')
        fallback = bind.values[1].id
        _require(fallback in code_map[selection['generator']['qualname']].co_freevars,
                 'sort filter alternative is not scoped to the original activation')
    receiver = evidence._name(source.iter.func.value)
    _require(receiver == selection['loop_binding']['receiver'], 'source factory must use the exact scoring receiver')
    return sorted_call, source.iter, binding.comparators[0].id, fallback


def _read_ports(node):
    if type(node) is list:
        for child in node:
            yield from _read_ports(child)
    elif type(node) is dict:
        if node.get('op') == 'input':
            yield node['name']
        else:
            for child in node.values():
                yield from _read_ports(child)


def project(evidence_path, transition_path):
    evidence_ref = artifact_reference(Path(evidence_path).resolve(strict=True), evidence.SCHEMA)
    transition_ref = artifact_reference(Path(transition_path).resolve(strict=True), transition.SCHEMA)
    source = read_artifact(evidence_ref)
    old = read_artifact(transition_ref)
    _require(source['transition_evidence'] == old['evidence'], 'region evidence and transition belong to different sources')
    capture = read_artifact(source['capture'])
    selection = evidence.prepare(old['evidence']['uri'])
    module, codes = evidence._compile(source['source'])
    arm = evidence._one((row for row in selection['arms'] if row['arm'] == 'SORTED_DELEGATION'),
                        'expected one exact sorted producer')
    sort_call, factory, threshold, alternate = _calls(module, codes, selection, arm)
    transition_evidence = read_artifact(old['evidence'])
    score_code = transition_evidence['score']['code']
    owner, _, _ = score_code['qualname'].rpartition('.')
    qualname = owner + '.' + factory.func.attr
    generator = _definition(module, qualname)
    generator_code = codes[qualname]
    factory_site = evidence._event(codes[selection['generator']['qualname']], factory, 'CALL')
    imports = {item.asname or item.name:(row.module,item.name) for row in module.body
               if type(row) is ast.ImportFrom and row.level == 0 for item in row.names}
    constructors = {}
    for stmt in module.body:
        if type(stmt) is not ast.Assign or len(stmt.targets) != 1 or type(stmt.targets[0]) is not ast.Name:
            continue
        value = stmt.value
        if (type(value) is ast.Call and type(value.func) is ast.Name
                and imports.get(value.func.id) == ('collections', 'namedtuple') and len(value.args) == 2
                and not value.keywords and all(type(arg) is ast.Constant and type(arg.value) is str for arg in value.args)):
            constructors[stmt.targets[0].id] = value.args[1].value.replace(',', ' ').split()
    fields = {row['name']: transition._KINDS[row['kind']] for row in capture['state_fields']}
    scorer = read_artifact(old['score_plan'])
    score_ports = scorer['inputs']
    moves = tuple(row['type'] for row in score_ports if row['name'].startswith('arg_'))
    _require(moves and all(kind in ('int', 'str') for kind in moves), 'sort needs lexicographic exact scalar records')
    lowering = _Generator(generator, generator_code, fields, _defaults(module), moves, imports, constructors)
    body = lowering.block(generator.body, {})
    _require(len(constructors[lowering.constructor]) == len(moves), 'constructor fields differ from scoring record ports')
    ports = transition._interface(fields)
    reads = set(_read_ports(body)) | {row['name'] for row in score_ports if row['name'].startswith('field_')}
    _require(reads <= {port['name'] for port in ports}, 'ordered region contains unbound state reads')
    return dict(schema=SCHEMA, schema_version=1, source=source['source'], evidence=evidence_ref,
                transition=transition_ref, scorer=old['score_plan'],
                table=read_artifact(selection['transition_selection'])['table'],
                sorted_owner=selection['generator'], sorted_site=arm['sorted_call'], factory_site=factory_site,
                record_source=arm['code'],
                receiver=selection['loop_binding']['receiver'], factory=describe_code(generator_code), score=score_code,
                constructor=lowering.constructor, record_fields=constructors[lowering.constructor],
                builtin_calls={name:list(binding) for name,binding in imports.items() if binding == ('itertools', 'count')},
                state_fields=fields, state_field_order=list(fields),
                inputs=[port for port in ports if port['name'] in reads], record_types=list(moves),
                scalar_globals=lowering.globals,
                integer_rows={name:{key:list(values) for key,values in mapping.items()}
                              for name,mapping in lowering.mappings.items()},
                predicate=dict(threshold=threshold, nonzero_capture=alternate),
                descending=sort_call.keywords[0].value.value, body=body,
                rewrite=dict(rule='FUSE_EAGER_SCALAR_GENERATOR_CONSUMER',
                    chain=['READ_STATE', 'GENERATE_RECORD', 'SCORE_RECORD', 'FILTER_RECORD', 'STABLE_SORT'],
                    retained_effects=['ORDERED_SCORER_EVALUATION', 'FILTER_AFTER_SCORE', 'EXACT_COMPARISON_ORDER']),
                contract=dict(output='ORDERED_EXACT_SCALAR_RECORD_VALUES', captures='EXPLICIT_RUNTIME_PARAMETERS',
                    analysis_scope='DETACHED_EAGER_SORTED_PATH', deployment_authorized=False))
