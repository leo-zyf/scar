"""Lower checked, exact scalar values and state transitions to C++20."""

from pathlib import Path

from scar.analysis.scalar_table_handoff import table_version
from scar.codegen.scalar_table_native import MAX_LIVE_KEYS


_TABLE_TEMPLATE = Path(__file__).with_name('scalar_table_standalone_v1.cpp.in')
_SCORE_TEMPLATE = Path(__file__).with_name('scalar_score_standalone_v1.cpp.in')
_TRANSITION_TEMPLATE = Path(__file__).with_name('scalar_transition_standalone_v1.cpp.in')
_I64_MIN, _I64_MAX = -(1 << 63), (1 << 63) - 1
_CPP_TYPE = {'int': 'Integer', 'str': 'std::string', 'bool': 'bool', 'row': 'Cells'}
_ARITH = {'add': 'add(a, b)', 'sub': 'sub(a, b)', 'floordiv': 'floor_div(a, b)'}
_COMPARE = {'eq': '==', 'lt': '<', 'le': '<='}
_HEADER = 'SCAR_STATE_V1\n'
_PARSERS = {'bool': 'parse_bool', 'int': 'parse_integer', 'str': 'parse_string'}
_UNARY = {'neg': ('int', 'sub(Integer{0}, {})'), 'not': ('bool', '!({})'),
          'reverse': ('str', 'reversed({})'), 'swapcase': ('str', 'swapcase({})')}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _i64(value):
    return type(value) is int and _I64_MIN <= value <= _I64_MAX


def _table_rows(rows, version):
    _require(type(rows) is dict and 0 < len(rows) <= MAX_LIVE_KEYS, 'bounded frozen scalar table required')
    _require(table_version(rows) == version, 'frozen scalar table content version differs')
    for key, row in rows.items():
        _require(type(key) is str and '\0' not in key and type(row) is list and all(map(_i64, row)),
                 'frozen table requires UTF-8 keys and exact signed-64 tuple cells')
        key.encode('utf-8')
    return {key: tuple(row) for key, row in rows.items()}


def _cells(name, cells):
    blocks = [', '.join(map(_integer, cells[index:index + 12])) for index in range(0, len(cells), 12)]
    values = ',\n'.join('    ' + block for block in blocks)
    return f'constexpr std::array<Integer, {len(cells)}> {name}{{\n{values}\n}};'


def _emit(template, fields):
    source = template.read_text()
    for name, value in fields.items():
        source = source.replace('@SCAR_' + name + '@', value)
    return source.encode()


def score_prelude(versions):
    shared, _ = _SCORE_TEMPLATE.read_text().split('@SCAR_EVALUATOR@', 1)
    return shared.replace('@SCAR_TABLE_DEFINITION@', render_tables(versions))


def transition_support():
    template = _TRANSITION_TEMPLATE.read_text()
    return template.split('@SCAR_SHARED_RUNTIME@', 1)[1].split('@SCAR_SCORE@', 1)[0]


def _integer(value):
    return 'std::numeric_limits<Integer>::min()' if value == _I64_MIN else str(value)


def render_table(table):
    definitions, entries = [], []
    for i, key in enumerate(sorted(table)):
        cells = table[key]
        literal = ''.join(f'\\{byte:03o}' for byte in key.encode('utf-8'))
        definitions.append(_cells(f'values_{i}', cells))
        entries.append(f'    Row{{"{literal}", values_{i}}}')
    joined = ',\n'.join(entries)
    definitions.append(f'constexpr std::array<Row, {len(table)}> rows{{\n{joined}\n}};')
    return _emit(_TABLE_TEMPLATE, {'TABLE_DEFINITION': '\n\n'.join(definitions)})


def _ascii(text):
    return type(text) is str and text.isascii() and '\0' not in text


def _literal(text):
    _require(_ascii(text), 'score string literals and ABI inputs must be ASCII without NUL')
    payload = ''.join(f'\\{byte:03o}' for byte in text.encode('ascii'))
    return f'"{payload}"'


def _types(node, inferred):
    _require(node.get('type', inferred) == inferred, 'score expression type differs from its typed operands')
    return inferred


def _fields(node, expected):
    _require(type(node) is dict and node.keys() - {'type'} == {'op', *expected}, 'unexpected score IR expression fields')


def _ordered(left, right, task):
    # The checked expressions can decline. Finish the original left operand
    # before entering the right, independently of C++ argument ordering.
    return f'([&] {{ const auto& a = {left}; const auto& b = {right}; return {task}; }}())'


class ScoreEmitter:
    cpp_types = _CPP_TYPE

    def __init__(self, inputs):
        self.inputs = {row['name']: (row['type'], f'in_{i}') for i, row in enumerate(inputs)}
        _require(len(self.inputs) == len(inputs) and self.inputs
                 and all(type(row['name']) is str and row['name'] and row['type'] in ('str', 'int')
                         and row.keys() == {'name', 'type'} for row in inputs),
                 'score inputs need a unique sequence of typed ASCII or signed-64 formals')
        self.serial = 0

    def expression(self, node, local):
        op = node['op']
        if op in ('int', 'str'):
            _fields(node, ('value',))
            value = node['value']
            _require(_i64(value) if op == 'int' else _ascii(value), 'invalid typed score literal')
            text = _integer(value) if op == 'int' else f'std::string({_literal(value)})'
            return _types(node, op), text
        if op in ('input', 'get'):
            _fields(node, ('name',))
            names = self.inputs if op == 'input' else local
            _require(node['name'] in names, 'score operand has no prior typed definition')
            kind, text = names[node['name']]
            return _types(node, kind), text
        if op == 'table_row':
            _fields(node, ('key',))
            kind, key = self.expression(node['key'], local)
            _require(kind == 'str', 'score table row requires one ASCII key')
            return _types(node, 'row'), f'table_row(table, {key})'
        if op == 'item':
            _fields(node, ('value', 'index'))
            kind, value = self.expression(node['value'], local)
            index_type, index = self.expression(node['index'], local)
            _require(kind in ('str', 'row') and index_type == 'int', 'score index requires a row or ASCII text')
            return _types(node, 'str' if kind == 'str' else 'int'), _ordered(value, index, 'item(a, b)')

        _fields(node, ('args',))
        arguments = [self.expression(arg, local) for arg in node['args']]
        types = tuple(row[0] for row in arguments)
        args = tuple(row[1] for row in arguments)
        if op == 'and':
            _require(len(args) >= 2 and all(item == 'bool' for item in types), 'score boolean operands must be bool')
            return _types(node, 'bool'), '(' + ' && '.join(args) + ')'
        if op == 'select':
            _require(len(args) == 3 and types[0] == 'bool' and types[1] == types[2], 'score choice types differ')
            return _types(node, types[1]), f'({args[0]} ? {args[1]} : {args[2]})'
        if op in ('abs', 'upper'):
            kind, operation = ('int', 'absolute') if op == 'abs' else ('str', 'upper_ascii')
            _require(types == (kind,), 'score unary operand type differs')
            return _types(node, kind), f'{operation}({args[0]})'
        if op in _ARITH:
            _require(types == ('int', 'int'), 'score arithmetic requires two signed-64 integers')
            return _types(node, 'int'), _ordered(*args, _ARITH[op])
        if op == 'in':
            _require(types == ('str', 'str'), 'score string membership requires ASCII inputs')
            return _types(node, 'bool'), _ordered(*args, 'contains(a, b)')
        if op in _COMPARE:
            _require(types == ('int', 'int') or op == 'eq' and types == ('str', 'str'),
                     'score comparison has unsupported operand types')
            return _types(node, 'bool'), _ordered(*args, f'a {_COMPARE[op]} b')
        raise ValueError('unsupported normalized score expression: ' + str(op))

    def block(self, body, local, depth=1):
        prefix = '    ' * depth
        lines = []
        for i, statement in enumerate(body):
            op = statement['op']
            if op == 'set':
                _require(statement.keys() == {'op', 'name', 'value'} and type(statement['name']) is str,
                         'invalid score local assignment')
                name = statement['name']
                kind, code = self.expression(statement['value'], local)
                if name in local:
                    original, cpp = local[name]
                    _require(original == kind, 'score local cannot change its scalar type')
                    lines.append(f'{prefix}{cpp} = {code};')
                else:
                    cpp = f'local_{self.serial}'
                    self.serial += 1
                    local[name] = (kind, cpp)
                    lines.append(f'{prefix}[[maybe_unused]] {self.cpp_types[kind]} {cpp} = {code};')
            elif op == 'if':
                _require(statement.keys() <= {'op', 'test', 'body', 'else', 'site'}
                         and type(statement.get('site', '')) is str, 'invalid score conditional')
                kind, test = self.expression(statement['test'], local)
                _require(kind == 'bool', 'score condition must be bool')
                lines.append(f'{prefix}if ({test}) {{')
                lines.extend(self.block(statement['body'], dict(local), depth + 1))
                if statement.get('else'):
                    lines.append(f'{prefix}}} else {{')
                    lines.extend(self.block(statement['else'], dict(local), depth + 1))
                lines.append(f'{prefix}}}')
            elif op == 'return' and depth == 1 and i == len(body) - 1:
                _require(statement.keys() == {'op', 'value'}, 'invalid score return statement')
                kind, value = self.expression(statement['value'], local)
                _require(kind == 'int', 'score return must be signed-64')
                lines.append(f'{prefix}return {value};')
            else:
                raise ValueError('unsupported normalized score statement: ' + str(op))
        return lines

    def evaluate(self, body):
        _require(type(body) is list and body and body[-1]['op'] == 'return', 'score body needs one final scalar return')
        formals = []
        for kind, cpp in self.inputs.values():
            dtype = 'const std::string&' if kind == 'str' else _CPP_TYPE[kind]
            formals.append(f'[[maybe_unused]] {dtype} {cpp}')
        prototype = 'Integer evaluate([[maybe_unused]] const Table& table, ' + ', '.join(formals) + ') {'
        return '\n'.join((prototype, *self.block(body, {}), '}'))


def render_tables(versions):
    _require(type(versions) is dict and versions, 'at least one observed table version required')
    definitions, tables, symbols = [], [], {}
    for version_index, (version, snapshot) in enumerate(sorted(versions.items())):
        table = _table_rows(snapshot['rows'], version)
        rows = []
        for key, cells in sorted(table.items()):
            if cells not in symbols:
                symbol = f'cells_{len(symbols)}'
                symbols[cells] = symbol
                definitions.append(_cells(symbol, cells))
            symbol = symbols[cells]
            rows.append(f'    Row{{{_literal(key)}, {symbol}}}')
        contents = ',\n'.join(rows)
        definitions.append(f'constexpr std::array<Row, {len(rows)}> rows_{version_index}{{\n{contents}\n}};')
        tables.append(f'    Table{{"{version}", rows_{version_index}}}')
    contents = ',\n'.join(tables)
    definitions.append(f'constexpr std::array<Table, {len(tables)}> tables{{\n{contents}\n}};')
    return '\n\n'.join(definitions)


def render_score(plan):
    emitter = ScoreEmitter(plan['inputs'])
    reads, arguments = [], []
    for i, (kind, cpp) in enumerate(emitter.inputs.values(), 2):
        parse = 'parse_integer' if kind == 'int' else 'parse_string'
        reads.append(f'                const auto {cpp} = {parse}(argv[{i}]);')
        arguments.append(cpp)
    fields = dict(TABLE_DEFINITION=render_tables(plan['table_versions']), EVALUATOR=emitter.evaluate(plan['body']),
                  ARGC=str(len(arguments) + 2), INPUT_SETUP='\n'.join(reads), INPUTS=', '.join(arguments))
    return _emit(_SCORE_TEMPLATE, fields)


def score_samples(plan):
    result = []
    schema = {row['name']: row['type'] for row in plan['inputs']}
    for row in plan['samples']:
        entries = row['inputs']
        _require(entries.keys() == schema.keys() and row['table_version'] in plan['table_versions']
                 and _i64(row['result']), 'invalid score sample boundary')
        args = [row['table_version']]
        for name, kind in schema.items():
            item = entries[name]
            _require(_i64(item) if kind == 'int' else _ascii(item), 'score sample differs from its typed ABI')
            args.append(str(item))
        result.append(dict(arguments=args, result=row['result']))
    _require(result and {row['arguments'][0] for row in result} == set(plan['table_versions']),
             'each embedded table version needs an original result sample')
    return result


def _schema(ports):
    _require(type(ports) is list and ports and all(type(row) is dict and set(row) == {'name', 'type'}
        and type(row['name']) is str and row['name'] and row['type'] in _PARSERS for row in ports),
        'transition boundary requires flat typed scalar ports')
    values = {row['name']: row['type'] for row in ports}
    _require(len(values) == len(ports), 'transition port names must be unique')
    return values


def _valid(value, kind):
    return {'bool': lambda item: type(item) is bool, 'str': _ascii, 'int': _i64}[kind](value)


def encode_outputs(ports, values):
    """Encode one fully computed transition return in its exact flat port order."""
    fields = _schema(ports)
    _require(type(values) is dict and fields.keys() == values.keys(), 'transition output fields differ from their ABI')
    output = [_HEADER]
    for name, kind in fields.items():
        value = values[name]
        _require(_valid(value, kind), 'transition output has an unsupported exact type')
        if kind == 'str':
            output.append('s:' + value.encode('ascii').hex() + '\n')
        elif kind == 'bool':
            output.append(f'b:{int(value)}\n')
        else:
            output.append(f'i:{value}\n')
    return ''.join(output).encode('ascii')


def transition_arguments(plan, sample):
    fields = _schema(plan['inputs'])
    version = sample['table_version']
    values = sample['inputs']
    _require(version in plan['table_versions'] and type(values) is dict and values.keys() == fields.keys(),
             'transition sample inputs differ from their exact table or ports')
    result = [version]
    for name, kind in fields.items():
        value = values[name]
        _require(_valid(value, kind), 'transition sample has an unsupported exact input type')
        result.append(str(int(value)) if kind == 'bool' else str(value))
    return result


def _lambda(definitions, result):
    assignments = ''.join(f'const auto& {name} = {value}; ' for name, value in definitions)
    return f'([&] {{ {assignments}return {result}; }}())'


class TransitionEmitter(ScoreEmitter):
    cpp_types = {**_CPP_TYPE, 'bool_pair': 'BoolPair'}

    def __init__(self, plan, scorer):
        self.inputs = {name: (kind, f'in_{i}') for i, (name, kind) in enumerate(_schema(plan['inputs']).items())}
        self.outputs = tuple(_schema(plan['outputs']).values())
        self.score_types = tuple(_schema(scorer['inputs']).values())
        self.serial = 0

    def expression(self, node, local):
        op = node['op']
        if op == 'bool':
            _fields(node, ('value',))
            _require(type(node['value']) is bool, 'transition bool literal must be exact')
            return _types(node, 'bool'), 'true' if node['value'] else 'false'
        if op == 'let':
            return self.let(node, local)
        if op == 'slice':
            _fields(node, ('value', 'start', 'stop'))
            kind, text = self.expression(node['value'], local)
            _require(kind == 'str', 'transition slice requires ASCII text')
            bindings = [('text', text)]
            for name in ('start', 'stop'):
                operand = node[name]
                if operand is None:
                    bindings.append((name, 'std::nullopt'))
                else:
                    bound, value = self.expression(operand, local)
                    _require(bound == 'int', 'transition slice bounds must be signed-64 integers')
                    bindings.append((name, f'std::optional<Integer>{{{value}}}'))
            return _types(node, 'str'), _lambda(bindings, 'slice(text, start, stop)')
        if op == 'item':
            _fields(node, ('value', 'index'))
            kind, base = self.expression(node['value'], local)
            idx_type, idx = self.expression(node['index'], local)
            _require(kind in ('bool_pair', 'row', 'str') and idx_type == 'int', 'unsupported transition index types')
            value_type = {'bool_pair': 'bool', 'row': 'int', 'str': 'str'}[kind]
            return _types(node, value_type), _ordered(base, idx, 'item(a, b)')
        if op not in {*_UNARY, 'score_call', 'pair', 'concat', 'mul', 'eq', 'ne', 'splice'}:
            return super().expression(node, local)

        _fields(node, ('args',))
        operands = [self.expression(arg, local) for arg in node['args']]
        types = tuple(row[0] for row in operands)
        args = tuple(row[1] for row in operands)
        if op == 'score_call':
            _require(types == self.score_types, 'transition score call differs from the exact admitted input ports')
            names = [f'arg_{i}' for i in range(len(args))]
            call = f'evaluate_score(table, {", ".join(names)})'
            return _types(node, 'int'), _lambda(zip(names, args, strict=True), call)
        if op == 'splice':
            _require(types == ('str', 'int', 'str'), 'transition splice operand types differ')
            names = ('text', 'index', 'replacement')
            return _types(node, 'str'), _lambda(zip(names, args, strict=True), 'splice(text, index, replacement)')
        if op == 'pair':
            _require(types == ('bool', 'bool'), 'transition pair requires two exact bools')
            return _types(node, 'bool_pair'), 'BoolPair{' + ', '.join(args) + '}'
        if op in _UNARY:
            kind, render = _UNARY[op]
            _require(types == (kind,), 'transition unary operand type differs')
            return _types(node, kind), render.replace('{}', args[0])
        if op in ('eq', 'ne'):
            _require(types in (('int', 'int'), ('str', 'str'), ('bool', 'bool')), 'unsupported transition equality types')
            relation = 'a == b' if op == 'eq' else 'a != b'
            return _types(node, 'bool'), _ordered(*args, relation)
        kind, render = ('str', 'a + b') if op == 'concat' else ('int', 'multiply(a, b)')
        _require(types == (kind, kind), 'transition binary operand types differ')
        return _types(node, kind), _ordered(*args, render)

    def let(self, node, local):
        _fields(node, ('bindings', 'result'))
        nested, assigned, definitions = dict(local), set(), []
        for binding in node['bindings']:
            name = binding['name']
            _require(type(name) is str and name not in assigned and binding.keys() == {'name', 'value'},
                     'transition lexical parameter bound more than once')
            kind, code = self.expression(binding['value'], local)
            word = f'bound_{self.serial}'
            self.serial += 1
            assigned.add(name)
            nested[name] = kind, word
            definitions.append((word, code))
        kind, code = self.expression(node['result'], nested)
        return _types(node, kind), _lambda(definitions, code)

    def evaluate(self, body):
        _require(type(body) is list and body and body[-1].keys() == {'op', 'values'} and body[-1]['op'] == 'return',
                 'transition body requires one typed state return')
        local = {}
        lines = self.block(body[:-1], local)
        returns = [self.expression(arg, local) for arg in body[-1]['values']]
        _require(tuple(row[0] for row in returns) == self.outputs, 'transition return differs from the exact output ABI')
        types = ', '.join(self.cpp_types[kind] for kind in self.outputs)
        formals = []
        for kind, word in self.inputs.values():
            cpp = 'const std::string&' if kind == 'str' else self.cpp_types[kind]
            formals.append(f'[[maybe_unused]] {cpp} {word}')
        prototype = 'State evaluate_transition([[maybe_unused]] const Table& table, ' + ', '.join(formals) + ') {'
        result = '    return State{' + ', '.join(row[1] for row in returns) + '};'
        return '\n'.join((f'using State = std::tuple<{types}>;', '', prototype, *lines, result, '}'))


def render_transition(plan, scorer):
    shared = score_prelude(plan['table_versions'])
    scored = ScoreEmitter(scorer['inputs']).evaluate(scorer['body']).replace('Integer evaluate(',
                                                                             '[[maybe_unused]] Integer evaluate_score(', 1)
    emitter = TransitionEmitter(plan, scorer)
    formals = list(emitter.inputs.values())
    setup = '\n'.join(f'                const auto {word} = {_PARSERS[kind]}(argv[{i}]);'
                      for i, (kind, word) in enumerate(formals, 2))
    fields = dict(SHARED_RUNTIME=shared, SCORE=scored, TRANSITION=emitter.evaluate(plan['body']),
                  ARGC=str(len(formals) + 2), INPUT_SETUP=setup, INPUTS=', '.join(row[1] for row in formals))
    return _emit(_TRANSITION_TEMPLATE, fields)
