"""Lower the checked public path projection of a recursive scalar entry."""

import ast
from pathlib import Path

from scar.analysis.scalar_score_reuse_evidence import _compile, _one, _span
from scar.codegen.scalar_recursive_driver import SCHEMA as NATIVE
from scar.codegen.scalar_source import _literal
from scar.codegen.scalar_standalone import _ref, _require
from scar.compiler.scalar_table_region import _definition
from scar.ir.execution_recipe import read_artifact
from scar.trace.code_snapshot import describe_code


_UI = 'scar.scalar-recursive-ui-contract'
_ARITH = {ast.Add: 'add', ast.Sub: 'sub', ast.Mod: 'remainder', ast.FloorDiv: 'floor_div'}
_COMPARE = {ast.Eq: '==', ast.NotEq: '!=', ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>='}


def _name(node):
    return node.id if type(node) is ast.Name else None


def _named(node, name):
    return _name(node) == name


def _const(node, value):
    return type(node) is ast.Constant and type(node.value) is type(value) and node.value == value


def _method(node, base, name, size):
    return (type(node) is ast.Call and type(node.func) is ast.Attribute and _named(node.func.value, base)
            and node.func.attr == name and len(node.args) == size and not node.keywords)


def _body(node):
    body = node.body
    if body and type(body[0]) is ast.Expr and type(body[0].value) is ast.Constant and type(body[0].value.value) is str:
        return body[1:]
    return body


def _formals(function, count):
    args = function.args
    names = [row.arg for row in (*args.posonlyargs, *args.args)]
    _require(len(names) == count and not args.vararg and not args.kwarg and not args.kwonlyargs
             and not function.decorator_list, 'unsupported public projection formals')
    return names


def _returned(function):
    statements = _body(function)
    _require(len(statements) == 1 and type(statements[0]) is ast.Return, 'public scalar helper must return one value')
    return statements[0].value


def _assign(node):
    _require(type(node) is ast.Assign and len(node.targets) == 1, 'expected an exact public state store')
    return node.targets[0], node.value


def _tuple(node):
    _require(type(node) is ast.Tuple and len(node.elts) == 2, 'expected two exact public scalar ports')
    return node.elts


def _equal(a, b):
    return ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)


def _read_globals(module):
    constants = {}
    for node in module.body:
        if type(node) is not ast.Assign or len(node.targets) != 1:
            continue
        target, value = node.targets[0], node.value
        if type(target) is ast.Name and type(value) is ast.Constant and type(value.value) is int:
            constants[target.id] = value.value
        elif type(target) is ast.Tuple and all(type(name) is ast.Name for name in target.elts):
            if type(value) is ast.Tuple and all(type(item) is ast.Constant and type(item.value) is int for item in value.elts):
                values = tuple(item.value for item in value.elts)
            elif (type(value) is ast.Call and _named(value.func, 'range') and len(value.args) == 1
                  and not value.keywords and _const(value.args[0], len(target.elts))):
                values = range(len(target.elts))
            else:
                continue
            constants.update((name.id, item) for name, item in zip(target.elts, values, strict=True))
    return constants


class _Scalar:
    def __init__(self, fields, moves, constants, calls=None):
        self.ports = {'State': fields, 'Move': moves}
        self.constants = constants
        self.calls = calls or {}

    def expect(self, node, local, kind):
        got, text = self.expression(node, local)
        _require(got == kind, 'public scalar expression changed its typed ports')
        return text

    def expression(self, node, local):
        recurse = lambda value: self.expression(value, local)
        if type(node) is ast.Constant:
            if type(node.value) is int and -(1 << 63) < node.value < (1 << 63):
                return 'int', f'Integer{{{node.value}}}'
            if type(node.value) is str:
                return 'str', f'std::string({_literal(node.value)})'
        if type(node) is ast.Name:
            if node.id in local:
                return local[node.id]
            if node.id in self.constants:
                return self.expression(ast.Constant(self.constants[node.id]), local)
        if type(node) is ast.Attribute:
            kind, symbol = recurse(node.value)
            field = self.ports.get(kind, {}).get(node.attr)
            _require(field is not None, 'public scalar expression crosses an unresolved source property')
            index, result = field
            return result, f'std::get<{index}>({symbol})'
        if type(node) is ast.Subscript:
            a = self.expect(node.value, local, 'str')
            b = self.expect(node.slice, local, 'int')
            return 'str', f'item({a}, {b})'
        if type(node) is ast.IfExp:
            test = self.expect(node.test, local, 'bool')
            kind, yes = recurse(node.body)
            return kind, f'({test} ? {yes} : {self.expect(node.orelse, local, kind)})'
        if type(node) is ast.BoolOp and type(node.op) in (ast.And, ast.Or):
            join = ' && ' if type(node.op) is ast.And else ' || '
            return 'bool', '(' + join.join(self.expect(item, local, 'bool') for item in node.values) + ')'
        if type(node) is ast.BinOp:
            kind, a = recurse(node.left)
            b = self.expect(node.right, local, kind)
            if type(node.op) is ast.Add and kind == 'str':
                return 'str', f'({a} + {b})'
            if kind == 'int' and type(node.op) in _ARITH:
                return 'int', f'{_ARITH[type(node.op)]}({a}, {b})'
        if type(node) is ast.Compare and len(node.ops) == 1 and type(node.ops[0]) in _COMPARE:
            kind, a = recurse(node.left)
            b = self.expect(node.comparators[0], local, kind)
            _require(kind in ('str', 'int'), 'unproved public comparison type')
            return 'bool', f'({a} {_COMPARE[type(node.ops[0])]} {b})'
        if type(node) is ast.Call and not node.keywords:
            name = _name(node.func)
            signatures = {'chr': ('str', 'scar_pv_ascii', ('int',)),
                          'str': ('str', 'std::to_string', ('int',)), 'abs': ('int', 'absolute', ('int',)),
                          **self.calls}
            if name == 'ord' and len(node.args) == 1 and type(node.args[0]) is ast.Constant:
                value = node.args[0].value
                _require(type(value) is str and len(value) == 1 and value.isascii(), 'ordinal must be literal ASCII')
                return 'int', f'Integer{{{ord(value)}}}'
            if name in signatures:
                kind, native, args = signatures[name]
                values = (self.expect(arg, local, port) for arg, port in zip(node.args, args, strict=True))
                return kind, f'{native}({", ".join(values)})'
            if type(node.func) is ast.Attribute:
                base = self.expect(node.func.value, local, 'str')
                if node.func.attr == 'lower' and not node.args:
                    return 'str', f'scar_pv_lower({base})'
                if node.func.attr == 'startswith' and len(node.args) == 1:
                    return 'bool', f'{base}.starts_with({self.expect(node.args[0], local, "str")})'
        raise ValueError('unresolved public scalar node: ' + ast.dump(node, include_attributes=False)[:160])


def _presence(node, layouts):
    _require(type(node) is ast.Call and _named(node.func, 'hasattr') and len(node.args) == 2 and not node.keywords
             and _name(node.args[0]) in layouts and type(node.args[1]) is ast.Constant
             and type(node.args[1].value) is str, 'unproved public representation branch')
    return node.args[1].value in layouts[_name(node.args[0])]


def _class_attrs(module, owner, fields, *, record):
    cls = _one(
        (node for node in module.body if type(node) is ast.ClassDef and node.name == owner), 'public class source differs')
    if record:
        _require(len(cls.bases) == 1 and type(cls.bases[0]) is ast.Call and _named(cls.bases[0].func, 'namedtuple')
                 and len(cls.bases[0].args) == 2 and not cls.bases[0].keywords and not cls.keywords
                 and _const(cls.bases[0].args[0], owner) and type(cls.bases[0].args[1]) is ast.Constant
                 and type(cls.bases[0].args[1].value) is str
                 and cls.bases[0].args[1].value.split() == list(fields), 'public state carrier differs from its typed tuple')
    else:
        _require(not cls.bases and not cls.keywords, 'public owner has unresolved inherited ports')
    members = {node.name for node in cls.body if isinstance(node, ast.FunctionDef)}
    for step in cls.body:
        if type(step) is ast.Assign:
            members.update(_name(target) for target in step.targets)
    _require(not cls.decorator_list and not members.intersection({'__getattr__', '__getattribute__'}),
             'public class presence depends on a custom attribute protocol')
    return set(fields) | members | set(dir(tuple if record else object))


def _render_record(ui, engine, name, fields, moves):
    function = _definition(ui, name)
    move, white = _formals(function, 2)
    missing, unpack, flip, alias, final = _body(function)
    _require(type(missing) is ast.If and type(missing.test) is ast.Compare and _named(missing.test.left, move)
             and len(missing.test.ops) == 1 and type(missing.test.ops[0]) is ast.Is
             and _const(missing.test.comparators[0], None) and not missing.orelse and len(missing.body) == 1
             and type(missing.body[0]) is ast.Return and type(missing.body[0].value) is ast.Constant
             and type(missing.body[0].value.value) is str and not function.args.defaults,
             'move absence must have its original public spelling')
    pair, inputs = _assign(unpack)
    names = [_name(node) for node in _tuple(pair)]
    _require(all(names) and type(flip) is ast.If and type(flip.test) is ast.UnaryOp
             and type(flip.test.op) is ast.Not and _named(flip.test.operand, white) and len(flip.body) == 1
             and not flip.orelse, 'public orientation operation differs')
    swapped, values = _assign(flip.body[0])
    _require(_equal(pair, swapped), 'public orientation must update its input pair')
    local = {move: ('Move', 'move')}
    code = _Scalar(fields, moves, _read_globals(ui))
    initial = [code.expect(node, local, 'int') for node in _tuple(inputs)]
    local.update(zip(names, (('int', 'first'), ('int', 'second')), strict=True))
    alternate = [code.expect(node, local, 'int') for node in _tuple(values)]
    target, rhs = _assign(alias)
    _require(_name(target) and type(rhs) is ast.Attribute and _name(rhs.value) and type(final) is ast.Return,
             'public render call must bind one original engine helper')
    ascii_fn = _definition(engine, rhs.attr)
    formal, = _formals(ascii_fn, 1)
    _require(not ascii_fn.args.defaults, 'public engine render needs an exact integer input')
    ascii_text = _Scalar(fields, moves, _read_globals(engine)).expect(_returned(ascii_fn), {formal: ('int', 'value')}, 'str')
    code.calls[target.id] = ('str', 'scar_pv_render_point', ('int',))
    rendered = code.expect(final.value, local, 'str')
    return dict(wrapper=function, engine=ascii_fn, engine_export=rhs.attr, engine_alias=_name(rhs.value),
                none=missing.body[0].value.value, render_ascii=ascii_text, initial=initial, flipped=alternate, rendered=rendered)


def _path_calls(module, codes, ui):
    consumer = _definition(module, ui['consumer']['qualname'])
    _require(describe_code(codes[consumer.name]) == ui['consumer'], 'public consumer descriptor differs from source')
    calls = []
    for node in ast.walk(consumer):
        if (type(node) is ast.Call and _name(node.func) and len(node.args) == 2
                and _named(node.args[0], ui['receiver']) and type(node.args[1]) is ast.Subscript
                and _named(node.args[1].value, ui['history']) and ast.literal_eval(node.args[1].slice) == -1):
            calls.append(node)
    name = _one({node.func.id for node in calls}, 'public consumer requires one source path routine')
    function = _definition(module, name)
    owner, position, scores, show_loop, clock = _formals(function, 5)
    _require(len(function.args.defaults) == 3 and _const(function.args.defaults[1], False)
             and all(len(node.keywords) == 2
                and any(port.arg == scores and _const(port.value, False) for port in node.keywords)
                and any(port.arg == clock and _name(port.value) for port in node.keywords) for node in calls),
             'public consumers must explicitly disable score formatting and retain the original clock operand')
    return function, calls, (owner, position, scores, show_loop, clock)


def _break(node):
    _require(type(node) is ast.If and len(node.body) == 1 and type(node.body[0]) is ast.Break and not node.orelse,
             'public path stop does not have its original control effect')
    return node.test


def _legality(module, function, factory, fields, moves):
    receiver, = _formals(function, 1)
    test = _returned(function)
    _require(not function.args.defaults and type(test) is ast.Call and _named(test.func, 'any')
             and len(test.args) == 1 and not test.keywords and type(test.args[0]) is ast.GeneratorExp,
             'public legality must use the original short-circuit visitor')
    comp = test.args[0]
    step, = comp.generators
    variable = _name(step.target)
    _require(variable and not step.ifs and not step.is_async and _method(step.iter, receiver, factory, 0),
             'public legality iterator differs from the checked native provider')
    return _Scalar(fields, moves, _read_globals(module)).expect(
        comp.elt, {receiver: ('State', 'position'), variable: ('Move', 'move')}, 'bool')


def _loop(module, function, formals, layouts, field, transition, factory, fields, moves):
    owner, position, scores, show_loop, clock = formals
    result_init, seen_init, color_init, initial_color, initial_scores, loop, done = _body(function)
    result, empty = _assign(result_init)
    seen, set_call = _assign(seen_init)
    color, color_call = _assign(color_init)
    original_color, copied = _assign(initial_color)
    _require(all(map(_name, (result, seen, color, original_color))) and _named(copied, color.id)
             and type(empty) is ast.List and not empty.elts and type(set_call) is ast.Call and _named(set_call.func, 'set')
             and not set_call.args and not set_call.keywords and type(color_call) is ast.Call and _name(color_call.func)
             and len(color_call.args) == 1 and _named(color_call.args[0], position) and not color_call.keywords
             and type(initial_scores) is ast.If and _named(initial_scores.test, scores) and not initial_scores.orelse
             and type(loop) is ast.While and _const(loop.test, True) and not loop.orelse
             and type(done) is ast.Return and _named(done.value, result.id), 'public loop prefix/return shape differs')

    lookup, reject, emit, tick, step, cap, seen_branch, emit_score = loop.body
    while type(lookup) is ast.If:
        selected = lookup.body if _presence(lookup.test, layouts) else lookup.orelse
        _require(len(selected) == 1, 'public lookup needs one shape-qualified source branch')
        lookup = selected[0]
    key, query = _assign(lookup)
    _require(_name(key) and type(query) is ast.Call and type(query.func) is ast.Attribute and query.func.attr == 'get'
             and type(query.func.value) is ast.Attribute and _named(query.func.value.value, owner)
             and query.func.value.attr == field and len(query.args) == 1 and _named(query.args[0], position)
             and not query.keywords, 'public path must read the checked ordered map by the full immutable state')
    reject = _break(reject)
    _require(type(reject) is ast.BoolOp and type(reject.op) is ast.Or and len(reject.values) == 2,
             'public path uses an unsupported missing-move guard')
    absent, illegal = reject.values
    _require(type(absent) is ast.Compare and _named(absent.left, key.id) and len(absent.ops) == 1
             and type(absent.ops[0]) is ast.Is and _const(absent.comparators[0], None)
             and type(illegal) is ast.Call and _name(illegal.func) and len(illegal.args) == 1 and not illegal.keywords
             and _method(illegal.args[0], position, transition, 1) and _named(illegal.args[0].args[0], key.id),
             'public legality does not use the proved native move transition')

    _require(type(emit) is ast.Expr and _method(emit.value, result.id, 'append', 1), 'public path cannot publish an unmodeled record')
    publish = emit.value.args[0]
    _require(type(publish) is ast.Call and _name(publish.func) and len(publish.args) == 2
             and _named(publish.args[0], key.id) and not publish.keywords, 'public move emission needs a known formatter')
    target, advance = _assign(tick)
    _require(_named(target, clock) and type(advance) is ast.IfExp and type(advance.test) is ast.Call
             and _name(advance.test.func) and [_name(node) for node in advance.test.args] == [position, key.id]
             and not advance.test.keywords, 'public clock update lost a required source operand')
    target, values = _assign(step)
    _require([_name(node) for node in _tuple(target)] == [position, color.id] and _equal(_tuple(values)[0], illegal.args[0]),
             'public legality and iteration must share the same pure transition inputs')
    native_cap = _break(cap)
    _require(type(seen_branch) is ast.If and not _presence(seen_branch.test, layouts),
             'public path deduplication must use the full immutable state')
    repeat, added = seen_branch.orelse
    _require(type(repeat) is ast.If and type(repeat.test) is ast.Compare and _named(repeat.test.left, position)
             and len(repeat.test.ops) == 1 and type(repeat.test.ops[0]) is ast.In and _named(repeat.test.comparators[0], seen.id)
             and len(repeat.body) == 2 and type(repeat.body[-1]) is ast.Break and not repeat.orelse
             and type(repeat.body[0]) is ast.If and _named(repeat.body[0].test, show_loop)
             and not repeat.body[0].orelse and type(added) is ast.Expr and _method(added.value, seen.id, 'add', 1)
             and _named(added.value.args[0], position) and type(emit_score) is ast.If
             and _named(emit_score.test, scores) and not emit_score.orelse,
             'public path retains original membership, cycle stop and disabled optional output')
    color_helper = _definition(module, color_call.func.id)
    quiet_helper = _definition(module, advance.test.func.id)
    legality = _definition(module, illegal.func.id)
    row, = _formals(color_helper, 1)
    q_position, q_move = _formals(quiet_helper, 2)
    _require(not color_helper.args.defaults and not quiet_helper.args.defaults, 'public helper inputs must be explicit')
    constants = _read_globals(module)
    code = _Scalar(fields, moves, constants, calls={color_helper.name: ('int', 'scar_pv_color', ('State',)),
                                                  quiet_helper.name: ('bool', 'scar_pv_quiet', ('State', 'Move'))})
    # Verify the unused color carry is a pure scalar expression. Unselected score formatting is never evaluated.
    code.expect(_tuple(values)[1], {color.id: ('int', 'previous_color')}, 'int')
    local = {position: ('State', 'position'), key.id: ('Move', 'move'), clock: ('int', 'hclock')}
    return dict(formatter=publish.func.id, color=color_helper, quiet=quiet_helper, legal=legality,
                color_value=code.expect(_returned(color_helper), {row: ('State', 'position')}, 'int'),
                quiet_value=code.expect(_returned(quiet_helper), {q_position: ('State', 'position'), q_move: ('Move', 'move')}, 'bool'),
                legal_value=_legality(module, legality, factory, fields, moves),
                orientation=code.expect(publish.args[1], local, 'bool'), counter=code.expect(advance, local, 'int'),
                cap=code.expect(native_cap, local, 'bool'))


def render(ui_path, native_path):
    """Return the C++ projection to append to the exact checked driver translation unit."""
    ui_ref, native_ref = _ref(ui_path, _UI), _ref(native_path, NATIVE)
    ui, native = read_artifact(ui_ref), read_artifact(native_ref)
    driver = read_artifact(ui['source_driver'])
    recursive = read_artifact(native['native'])
    fused = read_artifact(recursive['fusion'])
    provider = read_artifact(fused['sorted'])
    sorted_plan = read_artifact(provider['region'])
    transition = read_artifact(sorted_plan['transition'])
    effects = read_artifact(transition['evidence'])
    _require(native['schema_version'] == 2 and native['code'] == driver['driver']['code']
             and recursive['region'] == driver['recursive_region'] and driver['source'] == transition['source'] == sorted_plan['source']
             and native['abi']['move'] == provider['abi']['record'] and native['world'] == recursive['world'],
             'public projection and native providers belong to different source')
    engine, engine_codes = _compile(driver['source'])
    module, codes = _compile(ui['ui_source'])
    state_owner, _, factory = sorted_plan['factory']['qualname'].rpartition('.')
    move_owner, _, move_method = effects['callee']['code']['qualname'].rpartition('.')
    owner_class, _, _ = native['code']['qualname'].rpartition('.')
    _require(state_owner == move_owner and describe_code(engine_codes[sorted_plan['factory']['qualname']]) == sorted_plan['factory']
             and describe_code(engine_codes[effects['callee']['code']['qualname']]) == effects['callee']['code'],
             'native projection did not bind the exact move generator and transition')
    state_fields = sorted_plan['state_field_order']
    record_fields = sorted_plan['record_fields']
    position_fields = {}
    i = 0
    for name in state_fields:
        kind = sorted_plan['state_fields'][name]
        position_fields[name] = (i, kind)
        i += 2 if kind == 'bool_pair' else 1
    move_fields = {name: (i, kind) for i, (name, kind) in enumerate(zip(record_fields, sorted_plan['record_types'], strict=True))}
    function, sites, formals = _path_calls(module, codes, ui)
    owner, position = formals[:2]
    layouts = {owner: _class_attrs(engine, owner_class, native['world'], record=False),
               position: _class_attrs(engine, state_owner, state_fields, record=True)}
    move_map = driver['yield']['move_lookup']['field']
    _require(native['world'][move_map]['type'] == 'move_map', 'public lookup has no physically retained map')
    pv = _loop(module, function, formals, layouts, move_map, move_method, factory, position_fields, move_fields)
    fmt = _render_record(module, engine, pv['formatter'], position_fields, move_fields)
    template = Path(__file__).with_name('scalar_recursive_source_pv.cpp.in').read_text()
    fields = dict(POINT=fmt['render_ascii'], INITIAL_FIRST=fmt['initial'][0], INITIAL_SECOND=fmt['initial'][1],
                  FLIP_FIRST=fmt['flipped'][0], FLIP_SECOND=fmt['flipped'][1], RECORD=fmt['rendered'],
                  COLOR=pv['color_value'], QUIET=pv['quiet_value'], LEGAL=pv['legal_value'],
                  ORIENTATION=pv['orientation'], COUNTER=pv['counter'], STOP=pv['cap'], MAP=native['world'][move_map]['field'])
    for name, fragment in fields.items():
        template = template.replace('@SCAR_' + name + '@', fragment)
    _require('@SCAR_' not in template, 'native public path projection is incomplete')
    ports = dict(path=describe_code(codes[function.name]), calls=[_span(node) for node in sites],
                 record=describe_code(codes[fmt['wrapper'].name]), point=describe_code(engine_codes[fmt['engine'].name]),
                 bound_helpers={key: describe_code(codes[pv[key].name]) for key in ('color', 'quiet', 'legal')},
                 source_engine_export=[fmt['engine_alias'], fmt['engine_export']], none_move=fmt['none'],
                 score_output_enabled=False, loop_marker_enabled=False,
                 transition=effects['callee']['code'], move_factory=sorted_plan['factory'], format_scope='ASCII_NORMAL_DOMAIN')
    return dict(source=template, source_ports=ports, ui=ui_ref, native=native_ref)
