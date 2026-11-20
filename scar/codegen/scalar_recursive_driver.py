"""Generate a persistent native generator from a checked recursive kernel and its source driver."""

import argparse
import ast
import ctypes as ffi
import json
import os
from pathlib import Path
import re
import shlex
from shutil import copy2
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from scar.analysis.scalar_score_reuse_evidence import _compile, _one
from scar.codegen.aten_provider import _dependencies
from scar.codegen.native_cpu_elf import cpu_executable_dynamic
from scar.codegen import scalar_recursive_lowering as core
from scar.codegen import scalar_recursive_standalone as kernel
from scar.codegen import scalar_sorted_library as sorted_library
from scar.codegen.scalar_score_standalone import _literal
from scar.codegen.scalar_table_standalone import _FLAGS, _HOST, _LINK_FLAGS, _local, _ref, _require
from scar.ir.execution_recipe import read_artifact
from scar.trace.code_snapshot import describe_code


SCHEMA = 'scar.scalar-recursive-driver-library'
_NAME = 'scar_recursive_driver'
_SYMBOLS = tuple(f'{_NAME}_{operation}_v1' for operation in (
    'create', 'advance', 'get_move', 'snapshot', 'set_deadline_reader', 'close'))
_SONAME = 'libscar_recursive_driver_v1.so'
_LIBRARY = 'lib/' + _SONAME
_SOURCE = 'src/driver.cpp'
_HEADER = 'include/scar_recursive_driver_v1.h'
_MAP = 'exports.map'
_NATIVE = ('-std=c++20', *_FLAGS, '-shared', '-fPIC', '-fvisibility=hidden', '-fvisibility-inlines-hidden',
           *_LINK_FLAGS, '-Wl,-soname,' + _SONAME, '-Wl,--version-script,exports.map')
_STATUS = dict(yielded=0, exhausted=1, declined=2, internal=3, source_stop=4)
_WORLD_HEADER = b'SCAR_DRIVER_WORLD_V1\n'
_TYPE = {'int': 'Integer', 'bool': 'bool', 'str': 'std::string', 'float': 'double', 'state': 'State'}


def _member(node, name):
    return core._member(node, name)


def _call(node, name, count):
    return type(node) is ast.Call and core._name(node.func) == name and len(node.args) == count and not node.keywords


def _plain_arguments(function):
    args = function.args
    return [] if (args.kwonlyargs or args.vararg or args.kwarg or args.defaults or function.decorator_list) else (
        args.posonlyargs + args.args)


def _body(function):
    steps = function.body
    if (steps and type(steps[0]) is ast.Expr and type(steps[0].value) is ast.Constant
            and type(steps[0].value.value) is str):
        return steps[1:]
    return steps


def _find_driver(module, owner, recursive):
    classes = owner.split('.')
    scope = module
    for name in classes:
        scope = _one((node for node in scope.body if type(node) is ast.ClassDef and node.name == name),
                     'driver and recursion must share one exact owner')
    choices = []
    for function in scope.body:
        if type(function) is not ast.FunctionDef or function.name == recursive:
            continue
        args = _plain_arguments(function)
        if len(args) != 2:
            continue
        nodes = tuple(ast.walk(function))
        if (any(type(node) is ast.Yield for node in nodes) and any(type(node) is ast.Call
                and _member(node.func, args[0].arg) == recursive for node in nodes)):
            choices.append((function, args[0].arg, args[1].arg))
    return _one(choices, 'one exact source driver for the recursive owner is required')


def _prefix(body, receiver, argument, world, table):
    _require(len(body) >= 4 and all(type(node) is ast.Assign for node in body[:3]),
             'driver must begin with its original initialization block')
    reset, root, select = body[:3]
    _require(len(reset.targets) == 1 and type(reset.targets[0]) is ast.Tuple and type(reset.value) is ast.Tuple,
             'first driver initialization must retain its parallel assignments')
    init, roles = [], {}
    for lhs, rhs in zip(reset.targets[0].elts, reset.value.elts, strict=True):
        name = _member(lhs, receiver)
        kind, field = (world[name][item] for item in ('type', 'field'))
        _require(kind not in roles, 'ambiguous driver initialization role')
        if kind == 'int' and type(rhs) is ast.Constant and type(rhs.value) is int and rhs.value == 0:
            init.append(f'        handle->world.{field} = 0;')
        elif kind == 'entry_map' and type(rhs) is ast.Dict and not rhs.keys:
            init.append(f'        handle->world.{field}.clear();')
        elif kind == 'set' and _call(rhs, 'set', 1) and core._name(rhs.args[0]) == argument:
            init.append(f'        if (!handle->world.{field}.contains(position)) return SCAR_RECURSIVE_DRIVER_DECLINED_V1;')
        else:
            raise ValueError('driver initializer differs from its checked explicit world inputs')
        roles[kind] = field
    _require(set(roles) == {'int', 'entry_map', 'set'} and len(root.targets) == 2
             and type(root.value) is ast.Subscript and core._name(root.value.value) == argument
             and ast.literal_eval(root.value.slice) == -1, 'driver root must be its original last sequence item')
    names = [node.id for node in root.targets if type(node) is ast.Name]
    owners = [_member(node, receiver) for node in root.targets if type(node) is ast.Attribute]
    _require(len(names) == len(owners) == 1 and world[owners[0]]['type'] == 'state', 'driver root receiver changed')
    field = world[owners[0]]['field']
    init.append(f'        handle->world.{field} = position;')
    _require(len(select.targets) == 1 and type(select.targets[0]) is ast.Subscript
             and core._name(select.targets[0].value) == table and type(select.targets[0].slice) is ast.Constant
             and type(select.targets[0].slice.value) is str and type(select.value) is ast.IfExp
             and all(core._name(node) for node in (select.value.body, select.value.orelse)),
             'driver must select one observed frozen table row through its original conditional')
    return names[0], init, roles, select.targets[0].slice.value, select.value


class _Emitter:
    def __init__(self, receiver, recursive, recursive_formals, world, constants, fields, ports, clocks):
        self.receiver = receiver
        self.recursive = recursive
        self.root_keyword = recursive_formals[-1]
        self.world = world
        self.constants = {name: f'world.global_{i}' for i, name in enumerate(constants)}
        self.clocks = clocks
        self.fields = fields
        self.port_index = {row['name']: i for i, row in enumerate(ports)}
        self.soft_members = set()
        self.moves = set()
        self.serial = 0

    def allocate(self):
        name = f'driver_{self.serial}'
        self.serial += 1
        return name

    def expr(self, node, local):
        if type(node) is ast.Constant:
            if core._int64(node.value):
                return 'int', f'Integer{{{core._integer(node.value)}}}'
            if type(node.value) is bool:
                return 'bool', 'true' if node.value else 'false'
            if type(node.value) is str:
                return 'str', f'std::string({_literal(node.value)})'
        if type(node) is ast.Name:
            if node.id in local:
                return local[node.id]
            if node.id in self.constants:
                return 'int', self.constants[node.id]
        if type(node) is ast.Attribute:
            owner = core._name(node.value)
            if local.get(owner, (None,))[0] == 'state':
                field = self.fields[node.attr]
                _require(field in ('int', 'str', 'bool'), 'driver field needs one exact scalar state slot')
                return field, f'std::get<{self.port_index["field_" + node.attr]}>({local[owner][1]})'
            if owner == self.receiver and node.attr not in self.world:
                self.soft_members.add(node.attr)
                return 'float', 'soft'
        if type(node) is ast.UnaryOp and type(node.op) is ast.USub:
            kind, text = self.expr(node.operand, local)
            _require(kind == 'int', 'driver negation needs a signed-64 operand')
            return 'int', f'sub(Integer{{0}}, {text})'
        if type(node) is ast.BinOp and type(node.op) in core._ARITHMETIC:
            kinds, values = self.operands((node.left, node.right), local)
            _require(kinds == ('int', 'int'), 'driver arithmetic requires bounded integers')
            return 'int', core._sequenced(values, f'{core._ARITHMETIC[type(node.op)]}(a0, a1)')
        if type(node) is ast.Compare and len(node.ops) == len(node.comparators) == 1:
            kinds, values = self.operands((node.left, node.comparators[0]), local)
            op = type(node.ops[0])
            if kinds == ('str', 'str') and op is ast.In:
                return 'bool', core._sequenced(values, 'contains(a0, a1)')
            _require(kinds in (('int', 'int'), ('float', 'float')) and op in core._COMPARE,
                     'driver comparison needs like exact scalar types')
            return 'bool', core._sequenced(values, f'a0 {core._COMPARE[op]} a1')
        if type(node) is ast.BoolOp and type(node.op) in (ast.And, ast.Or):
            kinds, values = self.operands(node.values, local)
            _require(all(kind == 'bool' for kind in kinds), 'driver short circuit requires exact predicates')
            return 'bool', '(' + (' && ' if type(node.op) is ast.And else ' || ').join(values) + ')'
        if type(node) is ast.Call and type(node.func) is ast.Attribute:
            return self.call(node, local)
        raise ValueError('unsupported scoped source driver expression: ' + type(node).__name__)

    def operands(self, nodes, local):
        kinds, text = zip(*(self.expr(node, local) for node in nodes), strict=True)
        return tuple(kinds), tuple(text)

    def call(self, node, local):
        member = node.func
        if (_member(member, self.receiver) == self.recursive and len(node.args) == 3 and len(node.keywords) == 1
                and node.keywords[0].arg == self.root_keyword):
            kinds, values = self.operands((*node.args, node.keywords[0].value), local)
            _require(kinds == ('state', 'int', 'int', 'bool'), 'source driver changed the recursive typed ports')
            return 'int', core._sequenced(values, 'recurse(world, table, a0, a1, a2, a3)')
        name = _member(member.value, self.receiver)
        if member.attr == 'get' and name in self.world and len(node.args) == 1 and not node.keywords:
            field = self.world[name]
            kind, value = self.expr(node.args[0], local)
            _require(kind == 'state' and field['type'] == 'move_map', 'driver lookup must use its original move map')
            self.moves.add(name)
            return 'maybe_move', f'find_move(world.{field["field"]}, {value})'
        if member.attr == 'time' and core._name(member.value) in self.clocks and not (node.args or node.keywords):
            return 'float', 'exact_clock()'
        raise ValueError('driver call is outside the source clock, probe and state lookup protocol')

    def predicate(self, node, local):
        kind, text = self.expr(node, local)
        _require(kind == 'bool', 'scoped source driver condition must be an exact bool')
        return text

    def assign(self, target, kind, expression, local):
        name = core._name(target)
        _require(name is not None and kind in _TYPE, 'driver assignments must bind a fixed lexical type')
        if name in local:
            original, symbol = local[name]
            _require(original == kind, 'driver local changes its source type')
            return f'{symbol} = {expression};'
        symbol = self.allocate()
        local[name] = kind, symbol
        return f'{_TYPE[kind]} {symbol} = {expression};'

    def block(self, steps, local, depth=1):
        padding = '    ' * depth
        code = []
        for step in steps:
            if type(step) is ast.Assign:
                target, = step.targets
                if type(target) is ast.Tuple and type(step.value) is ast.Tuple:
                    values = []
                    for expression in step.value.elts:
                        kind, text = self.expr(expression, local)
                        staged = self.allocate()
                        code.append(f'{padding}const auto {staged} = {text};')
                        values.append((kind, staged))
                    code += [padding + self.assign(name, kind, text, local)
                             for name, (kind, text) in zip(target.elts, values, strict=True)]
                else:
                    code.append(padding + self.assign(target, *self.expr(step.value, local), local))
            elif type(step) is ast.For:
                name = core._name(step.target)
                _require(name is not None and name not in local and not step.orelse and _call(step.iter, 'range', 2),
                         'driver outer loop must retain its original two-bound range')
                kinds, bounds = self.operands(step.iter.args, local)
                _require(kinds == ('int', 'int'), 'driver range needs exact signed bounds')
                low, high, index = (self.allocate() for _ in range(3))
                code += [f'{padding}const Integer {low} = {bounds[0]};', f'{padding}const Integer {high} = {bounds[1]};',
                         f'{padding}for (Integer {index} = {low}; {index} < {high}; {index} = add({index}, 1)) {{']
                code += self.block(step.body, {**local, name: ('int', index)}, depth + 1) + [padding + '}']
            elif type(step) in (ast.While, ast.If):
                body = self.block(step.body, dict(local), depth + 1)
                test = self.predicate(step.test, local)
                clause = 'while' if type(step) is ast.While else 'if'
                code += [f'{padding}{clause} ({test}) {{', *body, padding + '}']
                if step.orelse:
                    _require(type(step) is ast.If, 'driver does not admit while-else effects')
                    code += [padding + 'else {', *self.block(step.orelse, dict(local), depth + 1), padding + '}']
            elif type(step) is ast.Expr and type(step.value) is ast.Yield and type(step.value.value) is ast.Tuple:
                kinds, row = self.operands(step.value.value.elts, local)
                _require(kinds == ('int', 'int', 'int', 'maybe_move'), 'source driver yield shape changed')
                code.append(padding + 'co_yield DriverRow{' + ', '.join(row) + '};')
            elif type(step) is ast.Return and step.value is None:
                code.append(padding + 'co_return;')
            else:
                raise ValueError('unsupported scoped source driver statement: ' + type(step).__name__)
        return code


def _selector(branch, position, versions, tables, observed, root_formal):
    _require(len(versions) == 2, 'driver needs exactly two original table variants')
    test = compile(ast.Expression(branch.test), '<typed source table choice>', 'eval')
    slots = {}
    for sample in observed:
        if sample['oracle_kind'] != 'CAPTURED_DRIVER_ACTIVATION':
            continue
        raw = sample['actuals'][root_formal]
        value = eval(test, {'__builtins__': {}}, {position: SimpleNamespace(**raw)})
        slot = versions.index(sample['table_version'])
        _require(type(value) is bool and slots.get(value, slot) == slot, 'table selection contradicts an original root activation')
        slots[value] = slot
    _require(set(slots) == {False, True} and set(slots.values()) == set(range(len(versions)))
             and all(tables[name]['rows'] for name in versions), 'both source table choice arms need original root evidence')
    return slots


def _plan(original, region, sorted_plan, state_plan, cases, abi):
    module, codes = _compile(region['source'])
    original_recursive = region['region']['owner']
    owner, _, name = original_recursive['qualname'].rpartition('.')
    function, receiver, history_arg = _find_driver(module, owner, name)
    body = _body(function)
    world = original['world']
    position, initial, roles, changed, branch = _prefix(body, receiver, history_arg, world, sorted_plan['table'])
    types = sorted_plan['state_fields']
    clocks = {part.asname or part.name for row in module.body if type(row) is ast.Import
              for part in row.names if part.name == 'time'}
    generator = _Emitter(receiver, name, region['region']['formals'], world, original['configuration_ports'],
                         types, state_plan['outputs'], clocks)
    state = {position: ('state', 'position')}
    choose = generator.predicate(branch.test, state)
    choices = _selector(branch, position, abi['table_slots'], state_plan['table_versions'], cases['cases'],
                        region['region']['formals'][1])
    tables = [state_plan['table_versions'][key]['rows'] for key in abi['table_slots']]
    _require(all(changed in table for table in tables) and tables[0][changed] != tables[1][changed]
             and {key: row for key, row in tables[0].items() if key != changed} == {
                 key: row for key, row in tables[1].items() if key != changed},
             'driver table alternatives must differ exactly at the original written row')
    driver = generator.block(body[3:], dict(state))
    move, = generator.moves
    soft, = generator.soft_members
    deadline, = [field for field in world if world[field]['type'] == 'float']
    code = describe_code(codes[owner + '.' + function.name])
    fields = dict(INITIALIZE='\n'.join(initial), SELECT=choose, DRIVER='\n'.join(driver), NODES=roles['int'],
                  DEADLINE=world[deadline]['field'], MOVE_MAP=world[move]['field'],
                  TABLE_TRUE=str(choices[True]), TABLE_FALSE=str(choices[False]))
    metadata = dict(code=code, world=world, world_order=list(world), configuration_ports=original['configuration_ports'],
                    table_slots=abi['table_slots'], table_write=dict(key=changed, branch=dict(
                        then=core._name(branch.body), otherwise=core._name(branch.orelse), true_slot=choices[True],
                        false_slot=choices[False])), clock_fields=dict(soft=soft, deadline=deadline),
                    probe_formals=region['region']['formals'][1:],
                    original_builtins=['range', 'set'], same_world_across_probes=True)
    return fields, metadata


def _move(site):
    outputs = []
    for i, port in enumerate(site['ports'][1:]):
        slot = port['slot']
        if port['type'] == 'str':
            outputs.append(f'    write_text(output.strings + {port["offset"]}, output.lengths[{slot}], {port["capacity"]}, std::get<{i}>(move));')
        else:
            outputs.append(f'    output.scalars[{slot}] = std::get<{i}>(move);')
    return '\n'.join(outputs)


def _kernel(original, deadline):
    end = 'int execute(std::string_view payload, Integer repetitions, bool display) {'
    _require(original.count(end) == 1, 'original recursive executable shell changed')
    text = original.split(end, 1)[0]
    start, stop = 'void publish(bool value)', 'struct Query {'
    _require(text.count(start) == text.count(stop) == 1, 'original recursive publisher framing changed')
    prefix, rest = text.split(start, 1)
    serialization, suffix = rest.split(stop, 1)
    serialization = (start + serialization).replace('std::cout', 'stream').replace('[](const auto&', '[&](const auto&')
    serialization = serialization.replace('void publish(', 'void publish(std::ostream& stream, ')
    serialization = serialization.replace('void publish_state(', 'void publish_state(std::ostream& stream, ')
    serialization = re.sub(r'(?<!void )\bpublish\(', 'publish(stream, ', serialization)
    clock = '    double clock() const { return exact_clock(); }'
    _require(prefix.count(clock) == 1, 'recursive kernel has no unique original deadline observation')
    clock_with_reader = ('    scar_recursive_deadline_reader_v1 deadline_reader = nullptr;\n'
                         '    void* deadline_context = nullptr;\n'
                         '    double clock() {\n'
                         '        const auto now = exact_clock();\n'
                         '        if (deadline_reader) {\n'
                         '            const double value = deadline_reader(deadline_context);\n'
                         '            if (!std::isfinite(value)) throw Decline{};\n'
                         f'            {deadline} = value;\n'
                         '        }\n        return now;\n    }')
    return prefix.replace(clock, clock_with_reader) + serialization + stop + suffix


def _basis(path):
    native_ref = _ref(path, kernel.SCHEMA)
    original = read_artifact(native_ref)
    root = Path(native_ref['uri']).parent
    refs, members, _, lower = kernel.prepare(*(original[name]['uri'] for name in ('region', 'fusion', 'cases')))
    _require(refs == tuple(original[name] for name in ('region', 'fusion', 'cases'))
             and original['world'] == lower['world'] and original['configuration_ports'] == list(lower['constant_ports'])
             and original['source'] == _local(root, kernel._SOURCE, 'scar.cpp-source')
             and all((root / name).read_bytes() == data for name, data in members.items()),
             'driver requires an exact checked native recursive source')
    region, fused, cases = map(read_artifact, refs)
    ordered = read_artifact(fused['sorted'])
    sorted_plan = read_artifact(ordered['region'])
    state_plan = read_artifact(sorted_plan['transition'])
    abi = ordered['abi']
    fields, metadata = _plan(original, region, sorted_plan, state_plan, cases, abi)
    fields.update(KERNEL=_kernel(members[kernel._SOURCE].decode(), fields['DEADLINE']), MOVE_OUTPUT=_move(abi['record']))
    generated = Path(__file__).with_name('scalar_recursive_driver_v1.cpp.in').read_text()
    for name, content in fields.items():
        generated = generated.replace('@SCAR_' + name + '@', content)
    _require('@SCAR_' not in generated, 'native driver source material is incomplete')
    support = {name: data for name, data in members.items() if name.startswith('include/')}
    support.update({_SOURCE: generated.encode(), _HEADER: Path(__file__).with_name('scalar_recursive_driver_v1.h.in').read_bytes(),
                    _MAP: ('{ global: ' + '; '.join(_SYMBOLS) + '; local: *; };\n').encode()})
    inputs = [(index, row) for index, row in enumerate(cases['cases']) if row['oracle_kind'] == 'CAPTURED_DRIVER_ACTIVATION']
    seeds = [(row, (root / 'samples' / f'{index}.stdin').read_bytes()) for index, row in inputs]
    _require(all(raw == kernel._case(row, lower, sorted_plan, abi['table_slots'])[0] for row, raw in seeds),
             'native driver seeds differ from original root activation ports')
    return native_ref, metadata, abi, support, seeds, sorted_plan


def _attach(lib, operation, result, *arguments):
    call = getattr(lib, f'{_NAME}_{operation}_v1')
    call.restype, call.argtypes = result, arguments
    return call


def _verify(path, metadata, abi, seeds, sorted_plan):
    _, record = sorted_library.pod_types(abi)
    class Row(ffi.Structure):
        _fields_ = [(port, ffi.c_int64) for port in ('depth', 'threshold', 'score', 'nodes')] + [
            ('has_move', ffi.c_uint32), ('table_slot', ffi.c_uint32), ('move', record)]
    lib = ffi.CDLL(str(path))
    ptr, count, status = ffi.c_void_p, ffi.c_size_t, ffi.c_int
    create = _attach(lib, 'create', status, ffi.c_char_p, count, ffi.POINTER(ptr))
    advance = _attach(lib, 'advance', status, ptr, ffi.c_double, ffi.c_double, ffi.POINTER(Row))
    get = _attach(lib, 'get_move', status, ptr, ffi.c_char_p, count, ffi.POINTER(record))
    snapshot = _attach(lib, 'snapshot', status, ptr, ptr, count, ffi.POINTER(count))
    close = _attach(lib, 'close', None, ptr)
    records = []
    for oracle, raw in seeds:
        handle = ptr()
        _require(create(raw, len(raw), ffi.byref(handle)) == 0 and handle.value, 'driver rejected exact original seed')
        try:
            result = Row()
            world = oracle['before']
            deadline_member = metadata['clock_fields']['deadline']
            deadline = float.fromhex(world[deadline_member]['float_hex'])
            _require(advance(handle, 1e100, deadline, ffi.byref(result)) == 0, 'driver failed to yield the original first probe')
            actual = oracle['actuals']
            position, threshold, depth, _ = metadata['probe_formals']
            nodes_role = next(name for name, role in metadata['world'].items() if role['type'] == 'int')
            _require((result.depth, result.threshold) == (actual[depth], actual[threshold]) and result.score == oracle['result']
                     and result.nodes == oracle['after'][nodes_role]
                     and metadata['table_slots'][result.table_slot] == oracle['table_version'],
                     'driver yield differs from first original root result or effects')
            root_value = actual[position]
            root_wire = kernel._encoded(kernel._state(root_value, sorted_plan)).encode('ascii')
            found = record()
            code = get(handle, root_wire, len(root_wire), ffi.byref(found))
            _require(code == (0 if result.has_move else 1) and (code == 1 or bytes(found) == bytes(result.move)),
                     'driver point move lookup differs from its current exact state')
            length = count()
            _require(snapshot(handle, None, 0, ffi.byref(length)) == 1 and length.value, 'driver world length query failed')
            output = ffi.create_string_buffer(length.value)
            _require(snapshot(handle, output, len(output), ffi.byref(length)) == 0, 'driver world snapshot failed')
            expected = _WORLD_HEADER + ''.join((kernel._encoded(len(value)) if kind in ('entry_map', 'move_map', 'set') else '')
                       + kernel._encoded(value) for _, kind, value in kernel._world(oracle['after'], metadata['world'], sorted_plan)).encode('ascii')
            _require(output.raw[:length.value] == expected, 'driver complete first-probe world differs from the original')
            row = dict(table_slot=result.table_slot, depth=result.depth, threshold=result.threshold,
                       score=result.score, nodes=result.nodes, snapshot_bytes=length.value)
            _require(advance(handle, 1e100, 1e100, ffi.byref(result)) == 0 and result.nodes >= row['nodes'],
                     'persistent driver failed to resume into its second probe')
            row.update(resume_depth=result.depth, resume_threshold=result.threshold, resume_nodes=result.nodes)
            records.append(row)
        finally:
            close(handle)
    return records


def _links(path):
    elf = cpu_executable_dynamic(path)
    linked = _dependencies(subprocess.check_output(['/usr/bin/ldd', str(path)], env={}, text=True))
    sonames = {Path(name).name for name in linked}
    _require(elf['runpath'] is None and elf['rpath'] is None and set(elf['needed']) <= sonames <= _HOST
             and len(sonames) == len(linked),
             f'native driver has unexpected host ELF linkage: {elf!r} {linked!r}')
    symbols = subprocess.check_output(['/usr/bin/nm', '-D', '--defined-only', '--format=posix', str(path)], env={}, text=True)
    _require(sorted(line.split()[0] for line in symbols.splitlines()) == sorted(_SYMBOLS), 'native driver exports a different ABI')
    return linked


def _abi(ordered):
    return dict(symbols=list(_SYMBOLS), seed_protocol='SCAR_RECURSIVE_INPUT_V1', world_protocol='SCAR_DRIVER_WORLD_V1',
                lookup_protocol='OLD_STATE_LEAF_LINES_WITHOUT_HEADER', move=ordered['record'], statuses=_STATUS,
                clock='SOURCE_EPOCH_WITH_OPTIONAL_DEADLINE_CALLBACK_AT_RECURSIVE_CHECK',
                control='ONE_WORLD_PER_SOURCE_GENERATOR')


def export(native_path, out):
    """Produce a persistent driver DSO from the exact old recursive kernel scope."""
    native, metadata, ordered, members, seeds, sorted_plan = _basis(native_path)
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    _require(compiler, 'an available C++20 compiler is required')
    environment = {name: value for name, value in os.environ.items() if not name.startswith('LD_')}
    version = subprocess.check_output([*compiler, '--version'], env=environment, text=True).splitlines()[0]
    with TemporaryDirectory(prefix='.scar-recursive-driver-', dir=out.parent) as temporary:
        root = Path(temporary) / 'package'
        for name, content in members.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (root / _LIBRARY).parent.mkdir()
        subprocess.run([*compiler, *_NATIVE, '-Iinclude', _SOURCE, '-o', _LIBRARY], cwd=root, env=environment, check=True)
        moved = Path(temporary) / 'relocated recursive driver.so'
        copy2(root / _LIBRARY, moved)
        verified = _verify(moved, metadata, ordered, seeds, sorted_plan)
        artifact = dict(schema=SCHEMA, schema_version=1, cxx_standard=20, native=native, abi=_abi(ordered),
                        **metadata, validation=verified, linkage=_links(moved),
                        compiler=dict(command=compiler, version=version, flags=list(_NATIVE)),
                        library=_local(root, _LIBRARY, 'scar.native-library-file'), header=_local(root, _HEADER, 'scar.c-header'),
                        members={name: _local(root, name, 'application/octet-stream') for name in members},
                        whole_driver_cpp=True, deployment_authorized=False)
        (root / 'artifact.json').write_text(json.dumps(artifact, indent=2, sort_keys=True) + '\n')
        if out.exists():
            raise FileExistsError(out)
        root.rename(out)
    return _ref(out / 'artifact.json', SCHEMA)


def check(path):
    ref = _ref(path, SCHEMA)
    document = read_artifact(ref)
    root = Path(ref['uri']).parent
    native, metadata, ordered, members, seeds, sorted_plan = _basis(document['native']['uri'])
    executable = root / _LIBRARY
    _require(document['schema_version'] == 1 and document['cxx_standard'] == 20 and document['native'] == native
             and document['whole_driver_cpp'] is True and document['deployment_authorized'] is False
             and all(document[key] == value for key, value in metadata.items()) and document['abi'] == _abi(ordered)
             and document['compiler']['flags'] == list(_NATIVE) and document['linkage'] == _links(executable)
             and document['library'] == _local(root, _LIBRARY, 'scar.native-library-file')
             and document['header'] == _local(root, _HEADER, 'scar.c-header')
             and document['members'] == {name: _local(root, name, 'application/octet-stream') for name in members}
             and all((root / name).read_bytes() == value for name, value in members.items()), 'native driver package differs')
    _require(document['validation'] == _verify(executable, metadata, ordered, seeds, sorted_plan),
             'native driver persistent sample results differ from fresh verification')
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native')
    parser.add_argument('--out')
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    if args.check and not (args.native or args.out):
        result = check(args.check)
    elif args.native and args.out and not args.check:
        result = export(args.native, args.out)
    else:
        parser.error('provide --native and --out or --check ARTIFACT')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
