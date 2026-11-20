"""Lower typed position ingress from the original scalar entry source."""

import ast
from pathlib import Path

from scar.analysis.scalar_score_reuse_evidence import _compile, _one, _span
from scar.codegen.scalar_recursive_driver import SCHEMA as NATIVE
from scar.codegen.scalar_recursive_source_pv import (
    _UI, _Scalar, _assign, _body, _const, _equal, _formals, _method, _name, _named,
    _read_globals, _returned, _tuple,
)
from scar.codegen.scalar_score_standalone import _literal
from scar.codegen.scalar_table_standalone import _ref, _require
from scar.compiler.scalar_table_region import _definition
from scar.ir.execution_recipe import read_artifact
from scar.trace.code_snapshot import describe_code


def _value(node, expected):
    result = ast.literal_eval(node)
    _require(type(result) is expected, 'position source constant differs from its typed input')
    return result


def _call(node, function, size):
    return type(node) is ast.Call and _named(node.func, function) and len(node.args) == size and not node.keywords


def _slice(node, owner):
    _require(type(node) is ast.Subscript and _named(node.value, owner) and type(node.slice) is ast.Slice,
             'position source requires an exact array slice')
    return tuple(None if item is None else _value(item, int) for item in (node.slice.lower, node.slice.upper, node.slice.step))


def _tokens(node):
    if type(node) is not ast.Compare or len(node.ops) != 1 or type(node.ops[0]) is not ast.Eq:
        return None
    target, = node.comparators
    if (type(node.left) is ast.Subscript and _name(node.left.value) and type(node.left.slice) is ast.Slice
            and type(target) is ast.List and len(target.elts) == 2
            and all(type(item) is ast.Constant and type(item.value) is str for item in target.elts)
            and _slice(node.left, node.left.value.id) == (None, 2, None)):
        return node.left.value.id, tuple(item.value for item in target.elts)
    return None


def _last(node, owner):
    return type(node) is ast.Subscript and _named(node.value, owner) and _value(node.slice, int) == -1


def _orientation(node, value, module, color):
    _require(type(node) is ast.Compare and len(node.ops) == 1 and type(node.ops[0]) is ast.Eq
             and type(node.left) is ast.Call and _name(node.left.func) and len(node.left.args) == 1
             and _named(node.left.args[0], value) and not node.left.keywords,
             'position orientation must use its original representation test')
    callee = _definition(module, node.left.func.id)
    _require(callee.name == color['qualname'], 'position orientation uses a different checked provider')
    rhs, = node.comparators
    _require(_name(rhs) in _read_globals(module), 'position orientation needs one stable source color')
    return _read_globals(module)[rhs.id]


def _parity(node):
    _require(type(node) is ast.Compare and len(node.ops) == 1 and type(node.ops[0]) is ast.Eq
             and type(node.left) is ast.BinOp and type(node.left.op) is ast.Mod and _const(node.left.right, 2),
             'position input orientation requires its exact source parity')
    result = _value(node.comparators[0], int)
    _require(result in (0, 1), 'position input parity differs from its boolean domain')
    return node.left.left, result


def _tail(loop, *, history, clock, command, start, input_index, transition, quiet):
    _require(type(loop) is ast.For and not loop.orelse, 'position move suffix requires one complete loop')
    if input_index:
        index, token = map(_name, _tuple(loop.target))
        _require(index and token and _call(loop.iter, 'enumerate', 1) and _slice(loop.iter.args[0], command) == (start, None, None),
                 'starting-position move route differs')
    else:
        index, token = None, _name(loop.target)
        _require(token and _slice(loop.iter, command) == (start, None, None), 'FEN move route differs')
    conversion, tick, transition_call = loop.body
    result, parse = _assign(conversion)
    _require(_name(result) and type(parse) is ast.Call and _name(parse.func) and len(parse.args) == 2
             and _named(parse.args[0], token) and not parse.keywords, 'position moves lost their parse source')
    orientation, parity = _parity(parse.args[1])
    _require((_named(orientation, index) if input_index else _call(orientation, 'len', 1)
              and _named(orientation.args[0], history)), 'position move orientation lost its live operand')
    counter, update = _assign(tick)
    _require(_named(counter, clock) and type(update) is ast.IfExp and _call(update.test, quiet, 2)
             and _last(update.test.args[0], history) and _named(update.test.args[1], result.id)
             and type(update.body) is ast.BinOp and type(update.body.op) is ast.Add and _named(update.body.left, clock),
             'position quiet clock changed its original dependency')
    _require(type(transition_call) is ast.Expr and _method(transition_call.value, history, 'append', 1),
             'position suffix must append its full next immutable state')
    move_call = transition_call.value.args[0]
    _require(type(move_call) is ast.Call and type(move_call.func) is ast.Attribute and move_call.func.attr == transition
             and _last(move_call.func.value, history) and len(move_call.args) == 1
             and _named(move_call.args[0], result.id) and not move_call.keywords,
             'position suffix must use the checked native transition')
    return (parse.func.id, _value(update.body.right, int), _value(update.orelse, int), parity)


def _controller(ui, engine, ui_codes, quiet, color, transition, rotation, history):
    options = []
    for function in ui.body:
        if type(function) is not ast.FunctionDef:
            continue
        for row in ast.walk(function):
            if type(row) is ast.If and (match := _tokens(row.test)) is not None:
                options.append((function, row, *match))
    start_control, start_branch, command, start_tokens = _one(
        (row for row in options if len(row[1].body) == 2 and type(row[1].body[-1]) is ast.For
         and type(row[1].body[0]) is ast.Assign and type(row[1].body[0].targets[0]) is ast.Tuple
         and _named(row[1].body[0].targets[0].elts[0], history)), 'expected one original literal initial position route')
    _, fen_branch, fen_command, fen_tokens = _one(
        (row for row in options if row[0] is start_control and len(row[1].body) == 4
         and type(row[1].body[0]) is ast.Assign and type(row[1].body[0].value) is ast.Call
         and len(row[1].body[0].value.args) == 1 and type(row[1].body[0].value.args[0]) is ast.Starred),
        'expected one original structured position route')
    _require(command == fen_command and start_tokens[0] == fen_tokens[0] and start_tokens[1] != fen_tokens[1],
             'position modes do not share their original controller port')
    targets, values = _assign(start_branch.body[0])
    _, clock = map(_name, _tuple(targets))
    startup, initial_clock = _tuple(values)
    _require(clock and type(startup) is ast.List and len(startup.elts) == 1 and _name(startup.elts[0]),
             'starting position is not one original immutable state')
    start_seed_formal = startup.elts[0].id
    original_position, prepare = _assign(fen_branch.body[0])
    field_start, field_stop, step = _slice(prepare.args[0].value, command)
    _require(_name(original_position) and _name(prepare.func) and not prepare.keywords
             and (field_start, field_stop, step) == (2, 8, None), 'structured position requires six original inputs')
    target, positions = _assign(fen_branch.body[1])
    _require(_named(target, history) and type(positions) is ast.IfExp and type(positions.body) is ast.List
             and len(positions.body.elts) == 1 and _named(positions.body.elts[0], original_position.id)
             and type(positions.orelse) is ast.List and len(positions.orelse.elts) == 2
             and _method(positions.orelse.elts[0], original_position.id, rotation, 0)
             and _named(positions.orelse.elts[1], original_position.id), 'FEN history orientation lost its rotated predecessor')
    white = _orientation(positions.test, original_position.id, ui, color)
    target, counter = _assign(fen_branch.body[2])
    _require(_named(target, clock) and _call(counter, 'int', 1) and type(counter.args[0]) is ast.Subscript
             and _named(counter.args[0].value, command) and _value(counter.args[0].slice, int) == field_start + 4,
             'structured position halfmove input differs')
    tail = fen_branch.body[-1]
    _require(type(tail) is ast.If and not tail.orelse and type(tail.test) is ast.Compare and len(tail.test.ops) == 1
             and type(tail.test.ops[0]) is ast.Gt and _call(tail.test.left, 'len', 1) and _named(tail.test.left.args[0], command)
             and _const(tail.test.comparators[0], field_stop) and len(tail.body) == 2 and type(tail.body[0]) is ast.Assert,
             'structured position trailing moves have an unresolved source guard')
    suffix = tail.body[0].test
    _require(type(suffix) is ast.Compare and len(suffix.ops) == 1 and type(suffix.ops[0]) is ast.Eq
             and type(suffix.left) is ast.Subscript and _named(suffix.left.value, command)
             and _const(suffix.left.slice, field_stop), 'structured position suffix marker is not source-derived')
    marker = _value(suffix.comparators[0], str)
    starting = _tail(start_branch.body[-1], history=history, clock=clock, command=command, start=3,
                     input_index=True, transition=transition, quiet=quiet['qualname'])
    derived = _tail(tail.body[1], history=history, clock=clock, command=command, start=field_stop + 1,
                    input_index=False, transition=transition, quiet=quiet['qualname'])
    _require(starting[:3] == derived[:3] and starting[3] != derived[3], 'position routes disagree on move decoding or clocks')
    formals = start_control.args.posonlyargs + start_control.args.args
    ordinal = _one((i for i, node in enumerate(formals) if node.arg == start_seed_formal), 'missing original start state formal')
    actual = _one((node.args[ordinal] for fn in engine.body if type(fn) is ast.FunctionDef for node in ast.walk(fn)
                   if type(node) is ast.Call and type(node.func) is ast.Attribute and node.func.attr == start_control.name
                   and len(node.args) == len(formals) and not node.keywords), 'missing original engine-to-controller input')
    _require(type(actual) is ast.Subscript and _name(actual.value) and _last(actual, actual.value.id),
             'initial controller state must be the original stored history')
    initial = _one((node.value for node in engine.body if type(node) is ast.Assign and len(node.targets) == 1
                    and _named(node.targets[0], actual.value.id)), 'missing source initial history construction')
    _require(type(initial) is ast.List and len(initial.elts) == 1, 'initial history has unlowered construction')
    return dict(code=describe_code(ui_codes[start_control.name]), start=_span(start_branch.test), fen=_span(fen_branch.test),
                field_start=field_start, field_stop=field_stop, clock=clock, initial_clock=_value(initial_clock, int),
                init=initial.elts[0], commands=(*start_tokens, fen_tokens[1], marker),
                parser=starting[0], fen_helper=prepare.func.id, advance=starting[1], reset=starting[2],
                start_parity=starting[3], history_parity=derived[3], white=white)


class _PositionScalar(_Scalar):
    def expression(self, node, local):
        if type(node) is ast.Constant and type(node.value) is bool:
            return 'bool', 'true' if node.value else 'false'
        if type(node) is ast.BinOp and type(node.op) is ast.Mult:
            a = self.expect(node.left, local, 'int')
            b = self.expect(node.right, local, 'int')
            return 'int', f'multiply({a}, {b})'
        if _call(node, 'ord', 1) and type(node.args[0]) is not ast.Constant:
            return 'int', f'scar_pos_ord({self.expect(node.args[0], local, "str")})'
        if _call(node, 'int', 1):
            return 'int', f'parse_integer({self.expect(node.args[0], local, "str")})'
        if type(node) is ast.Call and type(node.func) is ast.Attribute and not node.args and not node.keywords:
            if node.func.attr == 'upper':
                return 'str', f'upper_ascii({self.expect(node.func.value, local, "str")})'
        if type(node) is ast.Subscript and type(node.value) is ast.Subscript and type(node.value.value) is ast.Attribute:
            table = node.value.value
            _require((_name(table.value), table.attr) == self.table, 'position score reads an unbound native table')
            key = self.expect(node.value.slice, local, 'str')
            index = self.expect(node.slice, local, 'int')
            return 'int', f'item(table_row(table, {key}), {index})'
        return super().expression(node, local)


def _move_parser(module, engine, fn):
    value, white = _formals(fn, 2)
    alias_stmt, unpack, flip, result = _body(fn)
    alias, entry = _assign(alias_stmt)
    _require(_name(alias) and type(entry) is ast.Attribute and _name(entry.value) and not fn.args.defaults,
             'typed coordinate route requires one original engine export')
    coordinates, values = _assign(unpack)
    _require(type(coordinates) is ast.Tuple and type(values) is ast.Tuple and len(coordinates.elts) == len(values.elts) == 3,
             'move parser has unmodeled output ports')
    first, second, promotion = map(_name, coordinates.elts)
    for item, bounds in zip(values.elts[:2], ((None, 2, None), (2, 4, None)), strict=True):
        _require(_call(item, alias.id, 1) and _slice(item.args[0], value) == bounds,
                 'move coordinate slices differ from original exact ports')
    up = values.elts[2]
    _require(type(up) is ast.Call and type(up.func) is ast.Attribute and up.func.attr == 'upper' and not up.args
             and not up.keywords and _slice(up.func.value, value) == (4, None, None), 'move promotion slice differs')
    _require(first and second and promotion and type(flip) is ast.If and type(flip.test) is ast.UnaryOp
             and type(flip.test.op) is ast.Not and _named(flip.test.operand, white) and not flip.orelse and len(flip.body) == 1,
             'move parser changed source orientation shape')
    output, flip_value = _assign(flip.body[0])
    _require(list(map(_name, _tuple(output))) == [first, second] and type(result) is ast.Return
             and type(result.value) is ast.Call and type(result.value.func) is ast.Attribute
             and _named(result.value.func.value, entry.value.id) and not result.value.keywords
             and list(map(_name, result.value.args)) == [first, second, promotion], 'move construction or flipped targets differ')
    exported = _definition(engine, entry.attr)
    point, = _formals(exported, 1)
    _require(not exported.args.defaults, 'the exact source coordinate requires a typed text port')
    code = _PositionScalar({}, {}, _read_globals(engine))
    formula = code.expect(_returned(exported), {point: ('str', 'square')}, 'int')
    local = {first: ('int', 'first'), second: ('int', 'second')}
    flips = [code.expect(item, local, 'int') for item in _tuple(flip_value)]
    return dict(module=entry.value.id, point=entry.attr, move=result.value.func.attr, formula=formula, flipped=flips)


def _padding(board, body):
    first, expanded = _assign(body[0])
    _require(_named(first, board) and type(expanded) is ast.Call and type(expanded.func) is ast.Attribute
             and expanded.func.attr == 'sub' and len(expanded.args) == 3 and not expanded.keywords
             and _const(expanded.args[0], r'\d') and _named(expanded.args[2], board), 'FEN digit expansion differs')
    transform = expanded.args[1]
    _require(type(transform) is ast.Lambda and len(transform.args.args) == 1
             and type(transform.body) is ast.BinOp and type(transform.body.op) is ast.Mult
             and _call(transform.body.right, 'int', 1)
             and _method(transform.body.right.args[0], transform.args.args[0].arg, 'group', 1)
             and _const(transform.body.right.args[0].args[0], 0), 'FEN fill expression has a different input')
    fill = _value(transform.body.left, str)
    target, chars = _assign(body[1])
    _require(_named(target, board) and _call(chars, 'list', 1), 'FEN padding must use an owned character sequence')
    padded = chars.args[0]
    _require(type(padded) is ast.BinOp and type(padded.op) is ast.Add and type(padded.left) is ast.BinOp
             and type(padded.left.op) is ast.Add, 'FEN padding construction differs')
    left, sep, right = padded.left.left, padded.left.right, padded.right
    _require(all(type(item) is ast.BinOp and type(item.op) is ast.Mult for item in (left, right))
             and type(sep) is ast.Call and type(sep.func) is ast.Attribute and sep.func.attr == 'join'
             and len(sep.args) == 1 and not sep.keywords and _method(sep.args[0], board, 'split', 1),
             'FEN ranked input split or padding changed')
    sides = [_value(item.left, int) for item in (left, right)]
    border = _value(left.right, str)
    separator = _value(sep.func.value, str)
    delimiter = _value(sep.args[0].args[0], str)
    target, overwrite = _assign(body[2])
    offset, stop, stride = _slice(target, board)
    _require(type(overwrite) is ast.BinOp and type(overwrite.op) is ast.Mult and type(overwrite.left) is ast.List
             and len(overwrite.left.elts) == 1, 'FEN board row boundary write differs')
    row_mark = _value(overwrite.left.elts[0], str)
    row_count = _value(overwrite.right, int)
    target, joined = _assign(body[3])
    _require(_named(target, board) and type(joined) is ast.Call and type(joined.func) is ast.Attribute
             and _const(joined.func.value, '') and joined.func.attr == 'join' and len(joined.args) == 1
             and _named(joined.args[0], board) and not joined.keywords, 'FEN board does not return its exact text layout')
    _require(stop is None and offset is not None and stride is not None and 0 <= offset < stride and 0 < row_count < 64
             and _value(right.right, str) == border and all(side >= 0 for side in sides)
             and all(len(item) == 1 for item in (border, row_mark, delimiter, fill))
             and 0 < len(separator) < stride, 'unsupported FEN board representation')
    size = stride * row_count
    ranks, remainder = divmod(size - sum(sides) + len(separator), stride)
    files = stride - len(separator)
    _require(remainder == 0 and 0 < ranks < 10 and 0 < files < 10, 'FEN board exceeds exact decimal cells')
    return dict(sides=sides, separator=separator, delimiter=delimiter, border=border, row_mark=row_mark,
                fill=fill, offset=offset, stride=stride, ranks=ranks, files=files, size=size)


def _casts(node, casting):
    target, value = _assign(node)
    _require(_name(target), 'castling side lost its original state field')
    keys = []
    for condition in _tuple(value):
        _require(type(condition) is ast.Compare and len(condition.ops) == 1 and type(condition.ops[0]) is ast.In
                 and _named(condition.comparators[0], casting), 'castling source requires its exact membership ports')
        keys.append(_value(condition.left, str))
    _require(all(len(key) == 1 for key in keys), 'castling port requires an exact ASCII flag')
    return target.id, keys


def _score_sum(node, board, classification):
    _require(_call(node, 'sum', 1) and type(node.args[0]) is ast.GeneratorExp, 'FEN score is not a bounded source reduction')
    gen = node.args[0]
    step, = gen.generators
    index, piece = map(_name, _tuple(step.target))
    _require(index and piece and _call(step.iter, 'enumerate', 1) and _named(step.iter.args[0], board)
             and not step.is_async and len(step.ifs) == 1 and _method(step.ifs[0], piece, classification, 0),
             'FEN input table scan has an unmodeled predicate')
    return gen.elt, {index: ('int', 'offset'), piece: ('str', 'piece')}


def _state_constructor(node, carrier, ordered_fields, scalar, bindings, alias=None):
    if alias is None:
        _require(_call(node, carrier, len(ordered_fields)), 'startup constructs a different native state')
    else:
        _require(_method(node, alias, carrier, len(ordered_fields)), 'structured source constructs a different native state')
    pieces = []
    for arg, kind in zip(node.args, ordered_fields, strict=True):
        if kind == 'bool_pair':
            if type(arg) is ast.Tuple:
                pieces.extend(scalar.expect(part, bindings, 'bool') for part in _tuple(arg))
            else:
                pair = scalar.expect(arg, bindings, 'bool_pair')
                pieces.extend(f'{pair}[{i}]' for i in range(2))
        else:
            pieces.append(scalar.expect(arg, bindings, kind))
    return 'State{' + ', '.join(pieces) + '}'


def _fen(module, engine, fn, parser, carrier, rotation, fields):
    board, side, castle, passant, _, _ = _formals(fn, 6)
    _require(not fn.args.defaults, 'structured position has a different source call convention')
    body = _body(fn)
    _require(len(body) == 10, 'structured position includes unsupported actions')
    geometry = _padding(board, body)
    white, white_letters = _casts(body[4], castle)
    black, black_letters = _casts(body[5], castle)
    ep, convert = _assign(body[6])
    alias = parser['module']
    _require(_name(ep) and type(convert) is ast.IfExp and type(convert.test) is ast.Compare
             and len(convert.test.ops) == 1 and type(convert.test.ops[0]) is ast.NotEq and _named(convert.test.left, passant)
             and _method(convert.body, alias, parser['point'], 1) and _named(convert.body.args[0], passant),
             'en passant field does not use its checked source coordinate parser')
    none_ep = _value(convert.test.comparators[0], str)
    zero_ep = _value(convert.orelse, int)

    members = {node.name for node in engine.body if type(node) in (ast.ClassDef, ast.FunctionDef)}
    members.update(item.id for row in engine.body if type(row) in (ast.Assign, ast.AnnAssign, ast.AugAssign)
                   for item in ast.walk(row) if type(item) is ast.Name and type(item.ctx) is ast.Store)
    members.update(item.asname or item.name.split('.', 1)[0] for row in engine.body
                   if type(row) in (ast.Import, ast.ImportFrom) for item in row.names)
    omitted = []
    for branch in body[7:9]:
        _require(type(branch) is ast.If and _call(branch.test, 'hasattr', 2) and _named(branch.test.args[0], alias),
                 'unresolved external position provider selection')
        feature = _value(branch.test.args[1], str)
        _require(feature not in members, 'selected engine has an unsupported alternate position provider')
        omitted.append(feature)

    low, high, build = body[8].orelse
    weight, positive = _assign(low)
    _require(_name(weight) and type(high) is ast.AugAssign and _named(high.target, weight.id)
             and type(high.op) is ast.Sub, 'FEN score accumulation differs')
    part, pos = _score_sum(positive, board, 'isupper')
    subtract, neg = _score_sum(high.value, board, 'islower')
    _require(type(part) is ast.Subscript and type(part.value) is ast.Subscript and type(part.value.value) is ast.Attribute
             and _named(part.value.value.value, alias), 'structured position must read its active source table')
    table = part.value.value.attr
    codec = _PositionScalar({}, {}, _read_globals(module))
    codec.table = alias, table
    position, state = _assign(build)
    _require(_name(position), 'structured source did not bind its prepared position')
    prepared = _state_constructor(state, carrier, fields, codec, {
        board: ('str', 'board'), weight.id: ('int', 'score'), white: ('bool_pair', 'white_rights'),
        black: ('bool_pair', 'black_rights'), ep.id: ('int', 'ep')}, alias=alias)
    ret = body[9]
    _require(type(ret) is ast.Return and type(ret.value) is ast.IfExp and type(ret.value.test) is ast.Compare
             and len(ret.value.test.ops) == 1 and type(ret.value.test.ops[0]) is ast.Eq
             and _named(ret.value.test.left, side) and _named(ret.value.body, position.id)
             and _method(ret.value.orelse, position.id, rotation, 0), 'structured source orientation is not its original native rotation')
    white_token = _value(ret.value.test.comparators[0], str)
    _require(white_token == 'w' and len(none_ep) == 1, 'unsupported standard ASCII FEN side notation')
    return dict(geometry=geometry, rights=[white_letters, black_letters], no_rights='-', no_ep=none_ep, ep_zero=zero_ep,
                score_pos=codec.expect(part, pos, 'int'), score_neg=codec.expect(subtract, neg, 'int'), state=prepared,
                table=table, excluded_provider_attributes=omitted, white_token=white_token, black_token='b')


def _initial_table(engine, native, table):
    change = native['table_write']
    alternatives = {change['branch']['then']: change['branch']['true_slot'],
                    change['branch']['otherwise']: change['branch']['false_slot']}
    rows = []
    for node in engine.body:
        if type(node) is not ast.Assign or len(node.targets) != 1:
            continue
        target, value = node.targets[0], node.value
        pairs = zip(target.elts, value.elts, strict=True) if type(target) is type(value) is ast.Tuple else ((target, value),)
        for name, part in pairs:
            if (_name(name) in alternatives and type(part) is ast.Subscript and _named(part.value, table)
                    and _const(part.slice, change['key'])):
                rows.append((name.id, _span(part)))
    original, span = _one(rows, 'initial scoring table does not alias one checked source alternative')
    return alternatives[original], span


def render(ui_path, native_path):
    """Return the position fragment, using the same driver and PV translation unit."""
    from scar.codegen.scalar_recursive_source_pv import render as pv_projection

    ui_ref, native_ref = _ref(ui_path, _UI), _ref(native_path, NATIVE)
    ui, native = read_artifact(ui_ref), read_artifact(native_ref)
    driver = read_artifact(ui['source_driver'])
    previous = pv_projection(ui_ref['uri'], native_ref['uri'])['source_ports']
    old_kernel = read_artifact(native['native'])
    sorting = read_artifact(read_artifact(read_artifact(old_kernel['fusion'])['sorted'])['region'])
    transition = read_artifact(sorting['transition'])
    effects = read_artifact(transition['evidence'])
    engine, engine_codes = _compile(driver['source'])
    module, codes = _compile(ui['ui_source'])
    rotation_code = effects['rotation']['code']
    rotation_name = rotation_code['qualname']
    _, _, rotation = rotation_name.rpartition('.')
    _, _, operation = effects['callee']['code']['qualname'].rpartition('.')
    rotating = _definition(engine, rotation_name)
    _require(describe_code(engine_codes[rotation_name]) == rotation_code
             and len(rotating.args.defaults) == 1, 'original implicit rotation mode differs')
    rotation_default = _value(rotating.args.defaults[0], bool)
    helper = previous['bound_helpers']
    controller = _controller(module, engine, codes, helper['quiet'], helper['color'], operation, rotation, ui['history'])
    decode_move = _definition(module, controller['parser'])
    parser = _move_parser(module, engine, decode_move)
    _require(parser['move'] == sorting['constructor'], 'position suffix constructs a different provider record')
    carrier = effects['constructor']['name']
    shape = [sorting['state_fields'][name] for name in sorting['state_field_order']]
    start = _state_constructor(controller['init'], carrier, shape, _PositionScalar({}, {}, _read_globals(engine)), {
        node.targets[0].id: ('str', f'std::string({_literal(node.value.value)})') for node in engine.body
        if type(node) is ast.Assign and len(node.targets) == 1 and _name(node.targets[0])
        and type(node.value) is ast.Constant and type(node.value.value) is str})
    prepare = _definition(module, controller['fen_helper'])
    fen = _fen(module, engine, prepare, parser, carrier, rotation, shape)
    _require(fen['table'] == driver['initialization']['table']['symbol'], 'FEN reads a different module scoring state')
    initial_slot, initial_source = _initial_table(engine, native, fen['table'])
    versions = [transition['table_versions'][label]['rows'] for label in native['table_slots']]
    pieces = ''.join(sorted(versions[0]))
    _require(all(row.keys() == versions[0].keys() and all(len(cells) == fen['geometry']['size'] for cells in row.values())
                 for row in versions) and pieces.isascii() and pieces.isalpha() and pieces.isupper()
             and len(pieces) == len(versions[0]), 'FEN input pieces need complete single-character native scoring rows')
    ready = fen['geometry']
    source_fields = dict(POINT=parser['formula'], FLIP_FIRST=parser['flipped'][0], FLIP_SECOND=parser['flipped'][1],
          RANKS=str(ready['ranks']), FILES=str(ready['files']), SIZE=str(ready['size']),
          DELIMITER=_literal(ready['delimiter']), FILL=_literal(ready['fill']), BORDER=_literal(ready['border']),
          SEP=_literal(ready['separator']), MARK=_literal(ready['row_mark']), OFFSET=str(ready['offset']),
          STRIDE=str(ready['stride']), LEFT=str(ready['sides'][0]), RIGHT=str(ready['sides'][1]),
          PIECES=_literal(pieces), CASTLING=_literal(''.join(letter for side in fen['rights'] for letter in side)),
          SCORE_POS=fen['score_pos'], SCORE_NEG=fen['score_neg'], FEN_STATE=fen['state'], START_STATE=start,
          WHITE_RIGHTS=', '.join(f'has_flag(rights, {_literal(letter)}[0])' for letter in fen['rights'][0]),
          BLACK_RIGHTS=', '.join(f'has_flag(rights, {_literal(letter)}[0])' for letter in fen['rights'][1]),
          NO_CASTLING=_literal(fen['no_rights']), NO_EP=_literal(fen['no_ep']), ZERO_EP=str(fen['ep_zero']),
          INITIAL_SLOT=str(initial_slot), INITIAL_CLOCK=str(controller['initial_clock']),
          CLOCK_ADVANCE=str(controller['advance']), CLOCK_RESET=str(controller['reset']),
          ROTATION_DEFAULT=str(rotation_default).lower(),
          WHITE_COLOR=str(controller['white']), START_PARITY=str(controller['start_parity']),
          HISTORY_PARITY=str(controller['history_parity']))
    for name, word in zip(('COMMAND', 'START', 'FEN', 'MOVES'), controller['commands'], strict=True):
        source_fields[name] = _literal(word)
    source_fields['WHITE_TOKEN'] = _literal(fen['white_token'])
    source_fields['BLACK_TOKEN'] = _literal(fen['black_token'])
    text = Path(__file__).with_name('scalar_recursive_source_position.cpp.in').read_text()
    for name, material in source_fields.items():
        text = text.replace('@SCAR_' + name + '@', material)
    _require('@SCAR_' not in text, 'position native ingress is missing a typed source port')
    ports = dict(controller=controller['code'], start_branch=controller['start'], fen_branch=controller['fen'],
                 initial_state=_span(controller['init']), initial_table_slot=initial_slot, initial_table_source=initial_source,
                 table_symbol=fen['table'], fen_geometry=ready, unsupported_provider_attributes=fen['excluded_provider_attributes'],
                 move_parser=describe_code(codes[decode_move.name]), field_projection=describe_code(codes[prepare.name]),
                 coordinate_parser=describe_code(engine_codes[parser['point']]), transition=effects['callee']['code'],
                 rotation=effects['rotation']['code'], quiet=helper['quiet'], color=helper['color'],
                 ownership='FRESH_ORDERED_HISTORY_ONLY', table_policy='CALLER_SUPPLIES_PRECEDING_MODULE_TABLE')
    return dict(source=text, source_ports=ports, ui=ui_ref, native=native_ref)
