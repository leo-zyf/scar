"""Render a replayed source controller into a typed C++20 execution component."""

import math
from pathlib import Path

from scar.ir.literals import PythonLiteral
from scar.optimization.resource_solver import semantic_sha256


SCHEMA = 'scar.native-controller-plan'
SOURCE_SCHEMA = 'scar.source-native-controller'
SOURCE = 'src/controller.cpp'
_TEMPLATE = Path(__file__).with_name('native_controller_v1.cpp.in')
_ARITY = {**dict.fromkeys(('Add', 'Sub', 'Mult', 'Div', 'Mod', 'Lt', 'LtE', 'Gt', 'GtE', 'Eq', 'NotEq'), 2),
          **dict.fromkeys(('USub', 'UAdd', 'Not'), 1)}
_SOURCE_OPERATORS = dict(ADD='Add', SUBTRACT='Sub', MULTIPLY='Mult', TRUE_DIVIDE='Div', MODULO='Mod',
                         LESS_THAN='Lt', LESS_THAN_OR_EQUAL='LtE', GREATER_THAN='Gt', GREATER_THAN_OR_EQUAL='GtE',
                         EQUAL='Eq', NOT_EQUAL='NotEq')


def _need(condition, detail):
    if not condition:
        raise ValueError('native controller: ' + detail)


def _blocks(rows, depth=0):
    _need(depth <= 128 and type(rows) is list, 'controller control nesting exceeds profile')
    for row in rows:
        yield row
        if row['kind'] == 'IF':
            yield from _blocks(row['then'], depth + 1)
            yield from _blocks(row['otherwise'], depth + 1)
        elif row['kind'] == 'WHILE':
            yield from _blocks(row['condition'], depth + 1)
            yield from _blocks(row['body'], depth + 1)
            yield from _blocks(row['otherwise'], depth + 1)


def _literal(wire):
    value = PythonLiteral.from_dict(wire).to_python()
    if value is None:
        return 'Value{None{}}'
    if type(value) is bool:
        return 'Value{true}' if value else 'Value{false}'
    if type(value) is int:
        _need(-(1 << 63) <= value < 1 << 63, 'integer literal exceeds the i64 profile')
        return f'Value{{std::bit_cast<std::int64_t>(UINT64_C({value % (1 << 64)}))}}'
    if type(value) is float:
        _need(math.isfinite(value), 'nonfinite scalar literal outside profile')
        return 'Value{' + value.hex() + '}'
    if type(value) is str:
        raw = value.encode('utf-8')
        escaped = ''.join(f'\\x{unit:02x}' for unit in raw)
        return f'Value{{std::string("{escaped}", {len(raw)})}}'
    raise ValueError('native controller: unsupported original literal kind')


def _value_id(operand):
    return operand['id'] if type(operand) is dict and operand.get('kind') == 'VALUE' else None


class _Controller:
    def __init__(self, source):
        self.source = source
        self.rows = list(_blocks(source['program']))
        _need(0 < len(self.rows) <= 4096, 'controller operation count exceeds profile')
        self.lexical = {row['id']: row for row in source['slots']}
        self.params = {row['value']: row for row in source['parameters']}
        self.values = {row['id']: row['expression'] for row in self.rows if row['kind'] == 'EVAL'}
        _need(len(self.lexical) == len(source['slots']) and len(self.params) == len(source['parameters'])
              and len(self.values) == sum(row['kind'] == 'EVAL' for row in self.rows),
              'source slot or result identities are ambiguous')
        self.calls = [row for row in self.rows if row['kind'] == 'CALL']
        self.effects = []
        self.updates = []
        self.local_writes = {row['slot'] for row in self.rows if row['kind'] == 'STORE_SLOT'
                             and not self._parameter_wiring(row)}
        self.field_writes = {(*self._path(row['receiver']), row['field']) for row in self.rows
                             if row['kind'] == 'STORE_FIELD'}
        receiver_nodes = [row['receiver'] for row in self.rows if row['kind'] == 'STORE_FIELD']
        receiver_nodes.extend(row['receiver'] for row in self.values.values() if row['kind'] == 'FIELD')
        receiver_roots = {self._path(value)[0] for value in receiver_nodes}
        self.handles = {_value_id(value) for value in receiver_nodes}
        self.handles.update(_value_id(row['callee']) for row in self.calls)
        self.handles.discard(None)
        self.slot_registers, self.fields, self.slots = {}, {}, []
        for row in source['parameters']:
            if row['name'] not in receiver_roots:
                self._slot(row['slot'], 'PARAMETER', (row['name'],), True,
                           row['slot'] in self.local_writes, row.get('default'))
        for key, expr in self.values.items():
            if expr['kind'] == 'FIELD' and key not in self.handles:
                self._field((*self._path(expr['receiver']), expr['field']))
        for row in self.rows:
            if row['kind'] == 'STORE_FIELD':
                self._field((*self._path(row['receiver']), row['field']))
        for row in source['slots']:
            if row['parameter'] is None:
                self._slot(row['id'], 'LOCAL', (row['namespace']['name'],), False,
                           row['id'] in self.local_writes)
        result_ids = [row['id'] if row['kind'] == 'EVAL' else row['result'] for row in self.rows
                      if row['kind'] in ('EVAL', 'CALL')]
        _need(len(set(result_ids)) == len(result_ids), 'source results are not unique')
        self.registers = {key: order for order, key in enumerate(value for value in result_ids if value not in self.handles)}
        for call in self.calls:
            if call['role'] == 'ORIGINAL_EFFECT':
                _need(not self._consumed(call['result']), 'opaque effect result has an unclosed value consumer')

    def _parameter_wiring(self, row):
        origin = self.params.get(_value_id(row['value']))
        return origin is not None and origin['slot'] == row['slot']

    def _path(self, operand):
        target = _value_id(operand)
        if target in self.params:
            return (self.params[target]['name'],)
        expr = self.values.get(target, {})
        if expr.get('kind') == 'SLOT':
            declared = self.lexical[expr['slot']]
            _need(declared['parameter'] is not None, 'opaque receiver requires an original parameter root')
            return (declared['parameter'],)
        if expr.get('kind') == 'FIELD':
            return (*self._path(expr['receiver']), expr['field'])
        if expr.get('kind') == 'NAMESPACE':
            names = {row['name'] for row in expr['namespace']}
            _need(len(names) == 1, 'opaque callable namespace name is ambiguous')
            return (next(iter(names)),)
        raise ValueError('native controller: opaque callee or receiver requires a direct original path')

    def _slot(self, key, kind, path, ingress, output, default=None):
        _need(key not in self.slot_registers, 'duplicate scalar wire slot')
        ordinal = len(self.slots)
        self.slot_registers[key] = ordinal
        row = dict(ordinal=ordinal, id=key, kind=kind, path=list(path), input=ingress, output=output)
        if default is not None:
            row['default'] = default
        self.slots.append(row)
        return ordinal

    def _field(self, path):
        if path not in self.fields:
            key = 'field:' + semantic_sha256(path)
            self.fields[path] = self._slot(key, 'FIELD', path, True, path in self.field_writes)
        return self.fields[path]

    def _consumed(self, identity):
        def inspect(value):
            if type(value) is list:
                return any(inspect(item) for item in value)
            if type(value) is dict:
                return _value_id(value) == identity or any(inspect(item) for item in value.values())
            return False

        # Each nested statement is inspected independently so release-only
        # uses never masquerade as value consumers through a parent block.
        return any(inspect({key:value for key, value in row.items()
                            if key not in ('body','condition','otherwise','then','origin')})
                   for row in self.rows if row['kind'] != 'RELEASE')

    def _operand(self, operand):
        kind = operand['kind']
        if kind == 'LITERAL':
            return _literal(operand['literal'])
        key = _value_id(operand)
        _need(key is not None and key not in self.handles, 'opaque reference used as scalar')
        if key in self.params:
            return f'bound(ctx.slots[{self.slot_registers[self.params[key]["slot"]]}])'
        return f'bound(ctx.registers[{self.registers[key]}])'

    @staticmethod
    def _array(values):
        return f'std::array<Value, {len(values)}>{{{{' + ', '.join(values) + '}}'

    def _expression(self, expr):
        kind = expr['kind']
        if kind == 'SLOT':
            return f'bound(ctx.slots[{self.slot_registers[expr["slot"]]}])'
        if kind == 'FIELD':
            field = self.fields[(*self._path(expr['receiver']), expr['field'])]
            return f'bound(ctx.slots[{field}])'
        if kind == 'TRUTH':
            return f'Value{{truth({self._operand(expr["value"])})}}'
        args = [self._operand(row) for row in expr['operands']]
        if kind == 'OPERATOR':
            op = _SOURCE_OPERATORS.get(expr['operator'], expr['operator'])
            _need(_ARITY.get(op) == len(args), 'unsupported exact scalar operator: ' + expr['operator'])
            return ('unary' if len(args) == 1 else 'binary') + f'(Op::{op}, ' + ', '.join(args) + ')'
        if kind == 'FORMAT':
            _need(len(args) == 2, 'original format requires value and specifier')
            return f'format({args[0]}, {args[1]})'
        if kind == 'STRING_JOIN':
            return f'join({self._array(args)})'
        if kind == 'STRING_CONVERT':
            _need(len(args) == 2 and expr['operands'][1].get('kind') == 'LITERAL'
                  and PythonLiteral.from_dict(expr['operands'][1]['literal']).to_python() == 115,
                  'only exact builtin string conversion is covered')
            return f'format({args[0]}, Value{{None{{}}}})'
        raise ValueError('native controller: unsupported original expression ' + kind)

    def _call(self, row):
        role = row['role']
        _need(role in ('NATIVE_UPDATE', 'ORIGINAL_EFFECT'), 'unresolved original call')
        sites = self.updates if role == 'NATIVE_UPDATE' else self.effects
        ordinal = len(sites)
        keywords = [item['name'] for item in row['keywords']]
        _need(len(keywords) == len(set(keywords)), 'original call has duplicate keywords')
        site = dict(ordinal=ordinal, site=row['site'], source_fact=row['source_fact'], callee=row['callee'],
                    callee_path=list(self._path(row['callee'])),
                    positional_count=len(row['positional']), keywords=keywords)
        if role == 'ORIGINAL_EFFECT':
            effect = self.source['effects'][ordinal]
            _need(row['effect'] == f'effect:{ordinal}' and effect['site'] == row['site']
                  and effect['result_policy'] == 'ORIGINAL_RELEASE_ONLY',
                  'effect ordinal differs from original source identity')
        else:
            site['formal_routes'] = row['formal_routes']
        sites.append(site)
        args = [self._operand(value) for value in (*row['positional'], *(arg['value'] for arg in row['keywords']))]
        function = 'invoke_update' if role == 'NATIVE_UPDATE' else 'invoke_effect'
        slot = self.registers[row['result']]
        return f'ctx.registers[{slot}] = ctx.{function}({ordinal}, {self._array(args)});'

    def render(self, rows, depth):
        lines = []
        for row in rows:
            def emit(text):
                lines.append('    ' * depth + text)

            kind = row['kind']
            if kind == 'EVAL':
                if row['id'] not in self.handles:
                    slot = self.registers[row['id']]
                    emit(f'ctx.registers[{slot}] = {self._expression(row["expression"])};')
            elif kind == 'STORE_SLOT' or kind == 'STORE_FIELD':
                if kind == 'STORE_SLOT' and self._parameter_wiring(row):
                    continue
                index = (self.slot_registers[row['slot']] if kind == 'STORE_SLOT'
                         else self.fields[(*self._path(row['receiver']), row['field'])])
                emit(f'ctx.slots[{index}] = {self._operand(row["value"])};')
                emit(f'ctx.dirty[{index}] = true;')
            elif kind == 'IF':
                emit(f'if (truth({self._operand(row["test"])})) {{')
                lines.extend(self.render(row['then'], depth + 1))
                emit('}')
                if row['otherwise']:
                    emit('else {')
                    lines.extend(self.render(row['otherwise'], depth + 1))
                    emit('}')
            elif kind == 'WHILE':
                emit('while (true) {')
                lines.extend(self.render(row['condition'], depth + 1))
                lines.append('    ' * (depth + 1) + f'if (!truth({self._operand(row["test"])})) break;')
                lines.append('    ' * (depth + 1) + 'ctx.tick();')
                lines.extend(self.render(row['body'], depth + 1))
                emit('}')
                lines.extend(self.render(row['otherwise'], depth))
            elif kind == 'CALL':
                emit(self._call(row))
            elif kind == 'RELEASE':
                value = _value_id(row['value'])
                if value in self.registers:
                    emit(f'ctx.registers[{self.registers[value]}] = Unbound{{}};')
            elif kind == 'RETURN':
                emit(f'return ctx.finish({self._operand(row["value"])});')
            else:
                raise ValueError('native controller: unsupported original statement ' + kind)
        return lines


def prepare(source, *, array_ids):
    """Lower checked source control and retain the exact synchronous call sites."""
    _need((source['schema'], source['schema_version']) == (SOURCE_SCHEMA, 1), 'versioned source controller required')
    array_ids = tuple(array_ids)
    _need(len(set(array_ids)) == len(array_ids) <= 256, 'array root order is ambiguous')
    lowerer = _Controller(source)
    program = '\n'.join(lowerer.render(source['program'], 1))
    _need(len(lowerer.effects) == len(source['effects']) and lowerer.updates,
          'source controller call denominator differs')
    plan = dict(schema=SCHEMA, schema_version=1, cxx_standard=20, callback_abi_version=1,
                source_sha256=semantic_sha256(source), slots=lowerer.slots,
                update_sites=lowerer.updates, effect_sites=lowerer.effects, array_ids=list(array_ids),
                callback_order='SYNCHRONOUS_ORIGINAL_CONTROL_ORDER', external_effects_closed=False)
    plan['plan_sha256'] = semantic_sha256(plan)
    substitutions = dict(SLOT_COUNT=len(lowerer.slots),
                         REGISTER_COUNT=len(lowerer.registers), ARRAY_COUNT=len(array_ids),
                         ENTRY_ARGUMENTS=', '.join(str(row['input']).lower() for row in lowerer.slots),
                         PROGRAM=program, PLAN_SHA256=plan['plan_sha256'])
    source = _TEMPLATE.read_text()
    for name, value in substitutions.items():
        source = source.replace(f'@{name}@', str(value))
    return plan, source.encode()
