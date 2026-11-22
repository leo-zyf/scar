"""Capture selected original scalar calls with one scoped monitoring lifecycle."""

import argparse
import builtins
from collections import Counter, defaultdict
from _collections import _tuplegetter
from contextlib import contextmanager
import os
from pathlib import Path
import sys
from threading import get_ident
from types import CodeType

from scar.analysis.scalar_score_evidence import CAPTURE as SCORE_CAPTURE, DESIGN as SCORE_DESIGN, _need as _need_score
from scar.analysis.scalar_table_handoff import CAPTURE as TABLE_CAPTURE, DESIGN as TABLE_DESIGN, _need, _write, table_version
from scar.analysis.scalar_transition_evidence import (
    CAPTURE as TRANSITION_CAPTURE, DESIGN as TRANSITION_DESIGN, _need as _need_transition, field_kind,
)
from scar.ir.execution_recipe import open_artifact, read_artifact
from scar.runtime.entry_runner import run_original_script
from scar.runtime.source_call_handoff_observe import SCHEMA as HANDOFF, _observe as observe_handoff
from scar.runtime.source_value_observe import _codes, _coordinate, _target_coordinate
from scar.trace.code_snapshot import describe_code, site


def _int(word):
    _need(type(word) is int and -(1 << 63) <= word < (1 << 63), 'exact i64 value required')
    return word


def _field(value, name):
    kind = type(value)
    _need(tuple in type.__getattribute__(kind, '__mro__') and
          type.__getattribute__(kind, '__getattribute__') is tuple.__getattribute__,
          'position requires an original tuple descriptor')
    mro = type.__getattribute__(kind, '__mro__')
    descriptor = next((type.__getattribute__(base, '__dict__')[name] for base in mro
                      if name in type.__getattribute__(base, '__dict__')), None)
    _need(type(descriptor) is _tuplegetter, 'selected original position field is not a tuple slot')
    return _tuplegetter.__get__(descriptor, value, kind)


def _position(value, names):
    result = {name: _field(value, name) for name in names}
    _need(all(type(cell) is int or type(cell) is str for cell in result.values()),
          'selected position fields must be immutable exact primitive values')
    return result


def _operand(frame, selector):
    return _int(selector['integer'] if 'integer' in selector else frame.f_locals[selector['local']])


def _bindings(frame, names):
    constants, carriers = {}, {}
    for name in names:
        value = frame.f_globals[name] if name in frame.f_globals else frame.f_builtins[name]
        if value is builtins.abs:
            carriers[name] = 'CPYTHON_EXACT_BUILTIN_ABS'
        else:
            constants[name] = _int(value)
    return constants, carriers


def _argument(value):
    kind = type(value)
    _need_score(tuple in type.__getattribute__(kind, '__mro__')
          and type.__getattribute__(kind, '__iter__') is tuple.__iter__
          and tuple.__len__(value) == 3, 'original argument requires a transparent three-slot tuple')
    first, second, third = (tuple.__getitem__(value, index) for index in range(3))
    _need_score(type(third) is str and third.isascii(), 'original tuple string field outside the scalar domain')
    return [_int(first), _int(second), third]


def _state(value, expected_type, fields):
    _need_transition(type(value) is expected_type and tuple.__len__(value) == len(fields),
          'original transition object requires the selected exact tuple class')
    result = {}
    for index, name in enumerate(fields):
        cell = _field(value, name)
        _need_transition(cell is tuple.__getitem__(value,index), 'named tuple descriptor does not match original field ordinal')
        serialized = list(cell) if type(cell) is tuple else cell
        field_kind(serialized)
        result[name] = serialized
    return result


def _class_metadata(cls):
    return dict(module=type.__getattribute__(cls,'__module__'), qualname=type.__getattribute__(cls,'__qualname__'))


def _unchanged(root, anchors):
    return (len(root) == len(anchors) and all(type(key) is str for key in root)
            and all(root.get(key) is row for key,row in anchors))


def _snapshot(roots, versions, table):
    _need(type(table) is dict and 0 < len(table) <= 64, 'exact dictionary with exact string keys required')
    # Compare one coherent shallow snapshot before sorting new versions.
    current = table.copy()
    _need(0 < len(current) <= 64 and all(type(key) is str for key in current),
          'exact dictionary with exact string keys required')
    token = None
    for prior, root, anchors, version in roots:
        if root is table:
            token = prior
            if len(current) == len(anchors) and all(current.get(key) is row for key, row in anchors):
                return token, anchors, version
    anchors = tuple(sorted(current.items()))
    _need(len(roots) < 32 and all(type(row) is tuple and row for _, row in anchors)
          and sum(len(row) for _, row in anchors) <= 4096, 'selected table exceeds scalar capture contract')
    rows = {key: [_int(item) for item in row] for key, row in anchors}
    version = table_version(rows)
    if token is None:
        token = len(roots) + 1
    if version in versions:
        _need(versions[version]['root_token'] == token, 'same content selected under different table roots')
    else:
        versions[version] = dict(rows=rows, root_token=token)
    roots.append((token, table, anchors, version))
    return token, anchors, version


@contextmanager
def _record(observer, schema, handlers, require, scope):
    monitor = observer.monitor = sys.monitoring
    observer.events = monitor.events
    slot = observer.slot = next(index for index in range(5, -1, -1) if monitor.get_tool(index) is None)
    callbacks = [(getattr(observer.events, event), getattr(observer, method)) for event, method in handlers]
    monitor.use_tool_id(slot, schema)
    observer.active = True
    try:
        require(monitor.get_events(slot) == 0, scope)
        for event, callback in callbacks:
            monitor.register_callback(slot, event, callback)
        sys.addaudithook(observer.attach)
        yield observer
    finally:
        try:
            observer.close()
        finally:
            for event, _ in callbacks:
                monitor.register_callback(slot, event, None)
            monitor.free_tool_id(slot)


class _Scores:
    def __init__(self, design, handoff):
        self.design = design
        self.handoff = handoff
        self.wanted = {_target_coordinate(design[name]): role
                       for name, role in (('value_code', 'value'), ('producer_code', 'move'))}
        self.codes = {}
        self.samples = {}
        self.values = {}
        self.moves = {}
        self.roots = []
        self.versions = {}
        self.frontiers = Counter()

    def _mask(self, code, *, instruction=False):
        mask = self.events.PY_START | self.events.PY_RETURN
        if instruction:
            mask |= self.events.INSTRUCTION
        self.monitor.set_local_events(self.slot, code, mask)

    def attach(self, event, args):
        if not self.active or event != 'exec' or not args or type(args[0]) is not CodeType:
            return
        root = args[0]
        if root.co_filename != self.design['source']['uri']:
            return
        for code in _codes(root):
            selected = self.wanted.get(_coordinate(code))
            if selected is None:
                continue
            name = 'value_code' if selected == 'value' else 'producer_code'
            _need(describe_code(code) == self.design[name], 'original scalar body changed on load')
            self.codes[code] = selected
            self._mask(code)

    def enter(self, code, _offset):
        if self.codes[code] != 'value' or not self.handoff.active or get_ident() != self.handoff.thread:
            return
        frame = sys._getframe(1)
        parent = frame.f_back
        if parent not in self.handoff.producers:
            return
        design = self.design
        if site(parent.f_code, parent.f_lasti) != design['value_call']:
            return
        try:
            parent_local, value_local = parent.f_locals, frame.f_locals
            receiver = value_local[design['value_receiver']]
            _need(receiver is parent_local[design['producer_receiver']]
                  and value_local[design['value_formal']] is parent_local[design['producer_formal']],
                  'value receiver and argument must be the same original producer inputs')
            origin = self.handoff.producers[parent][2]
            _need(frame not in self.values and parent not in self.moves, 'one selected accumulator call expected')
            self.values[frame] = (parent, origin, _position(receiver, design['position_fields']))
            self._mask(code, instruction=True)
        except (ValueError, KeyError):
            self.frontiers['ORIGINAL_SCALAR_CALL_INPUTS_OPEN'] += 1

    def _before(self, frame, parent, origin, position):
        design = self.design
        table = frame.f_globals[design['table']]
        token, anchors, version = _snapshot(self.roots, self.versions, table)
        key = frame.f_locals[design['key']]
        _need(type(key) is str and key in table, 'exact selected string key required')
        row = table[key]
        _need(type(row) is tuple, 'exact selected tuple row required')
        left, right = (_operand(frame, design[operand]) for operand in ('left', 'right'))
        _need(-len(row) <= left < len(row) and -len(row) <= right < len(row), 'selected index is outside the row')
        record = dict(producer=origin, table_version=version, root_token=token, key=key,
            left=left, right=right, left_value=_int(row[left]), right_value=_int(row[right]),
            row_size=len(row), input_position=position)
        self.moves[parent] = (record, table, anchors)

    def _after(self, frame, parent):
        design = self.design
        record, table, anchors = self.moves[parent]
        original = frame.f_globals[design['table']]
        _need(original is table and _unchanged(table, anchors), 'table root or rows changed at window boundaries')
        result = _int(frame.f_locals[design['base_local']])
        _need(result == record['left_value'] - record['right_value'], 'original scalar subtraction differs')
        record['result'] = result

    def instruction(self, code, offset):
        role = self.codes[code]
        frame = sys._getframe(1)
        design = self.design
        try:
            if role == 'value' and frame in self.values:
                parent, origin, position = self.values[frame]
                if offset == design['value_before']['offset']:
                    self._before(frame, parent, origin, position)
                elif offset == design['value_after']['offset'] and parent in self.moves:
                    self._after(frame, parent)
                    self._mask(code)
            elif role == 'move' and frame in self.moves and offset == design['producer_store_after']['offset']:
                record, _, _ = self.moves[frame]
                total = _int(frame.f_locals[design['producer_total']])
                start = _int(record['input_position'][design['score_field']])
                _need(total == start + record['value_delta'], 'original caller accumulation differs')
                record['producer_total'] = total
                self._mask(code)
        except (ValueError, KeyError):
            self.frontiers['ORIGINAL_SCALAR_WINDOW_OR_ACCUMULATOR_OPEN'] += 1
            selected_parent = self.values[frame][0] if role == 'value' else frame
            self.moves.pop(selected_parent, None)
            self._mask(code)

    def finish(self, code, offset, value):
        frame = sys._getframe(1)
        role = self.codes[code]
        if role == 'value':
            selected = self.values.pop(frame, None)
            if selected is None:
                return
            self._mask(code)
            parent, _, _ = selected
            found = self.moves.get(parent)
            if found is None or 'result' not in found[0]:
                self.moves.pop(parent, None)
                return
            try:
                found[0]['value_delta'] = _int(value)
                found[0]['value_return'] = site(code, offset)
            except ValueError:
                self.frontiers['NORMAL_SCALAR_VALUE_RETURN_OPEN'] += 1
                self.moves.pop(parent, None)
                return
            self._mask(parent.f_code, instruction=True)
            return
        selected = self.moves.pop(frame, None)
        if selected is None or 'producer_total' not in selected[0]:
            return
        record, _, _ = selected
        try:
            record['returned_position'] = _position(value, self.design['position_fields'])
        except ValueError:
            self.frontiers['NORMAL_PRODUCER_POSITION_FIELDS_OPEN'] += 1
            return
        record['producer_return'] = site(code, offset)
        self.samples[record['producer']] = record

    def close(self):
        self.active = False
        for code in self.codes:
            self.monitor.set_local_events(self.slot, code, 0)


class _Probe:
    def __init__(self, design):
        self.design = design
        self.code = None
        self.window = {}
        self.roots = []
        self.versions = {}
        self.recent = None
        self.attempted = Counter()
        self.frontiers = Counter()
        self.samples = defaultdict(list)
        self.globals = self.builtins = None
        self.ordinal = 0
        self.offsets = {row[key] for row in design['branches'] for key in
                        ('test_offset', 'then_offset', 'else_offset') if row[key] is not None}

    def _mask(self, instruction=False):
        mask = self.events.PY_START | self.events.PY_RETURN
        if instruction:
            mask |= self.events.INSTRUCTION
        self.monitor.set_local_events(self.slot, self.code, mask)

    def attach(self, event, args):
        if not self.active or event != 'exec' or not args or type(args[0]) is not CodeType:
            return
        root = args[0]
        if root.co_filename != self.design['source']['uri']:
            return
        wanted = _target_coordinate(self.design['code'])
        for code in _codes(root):
            if _coordinate(code) == wanted:
                _need_score(self.code is None and describe_code(code) == self.design['code'], 'original score code changed on load')
                self.code = code
                self._mask()

    def _table(self, root):
        if self.recent is not None:
            token, known, members, digest = self.recent
            if (root is known and _unchanged(root, members)):
                return self.recent
        # Use the same exact dict/tuple/i64 snapshot contract as the retained
        # source handoff. Content versions are never reconstructed from id().
        token, members, digest = _snapshot(self.roots, self.versions, root)
        _need_score(len(self.versions) <= self.design['limits']['max_versions'], 'table version capture budget exceeded')
        self.recent = token, root, members, digest
        return self.recent

    def begin(self, _code, _offset):
        frame = sys._getframe(1)
        parent = frame.f_back
        if parent is None or self.window:
            return
        spec = self.design
        try:
            selected = self._table(frame.f_globals[spec['table']])
            token, _, _, version = selected
            if self.attempted[version] >= spec['limits']['attempts_per_version']:
                return
            entry = site(parent.f_code, parent.f_lasti)
            _need_score(entry['opname'] in ('CALL', 'CALL_KW'), 'original score caller protocol not selected')
            locals_ = frame.f_locals
            fields = _position(locals_[spec['receiver']], spec['receiver_fields'])
            _need_score(all(type(value) is int or value.isascii() for value in fields.values()),
                  'exact ASCII source field required')
            arg = _argument(locals_[spec['argument']])
            constants, carriers = _bindings(frame, spec['global_loads'])
            if self.globals is None:
                self.globals, self.builtins = constants, carriers
            _need_score(self.globals == constants and self.builtins == carriers,
                  'global scalar or builtin changed between sampled frames')
            self.attempted[version] += 1
            self.ordinal += 1
            self.window[frame] = (selected, set(), dict(table_version=version, root_token=token,
                frame_ordinal=self.ordinal, thread_id=get_ident(), entry_site=entry,
                receiver_fields=fields, argument=arg, move=arg))
            self._mask(instruction=True)
        except (ValueError, KeyError):
            self.frontiers['SCORE_INPUT_BINDINGS_OPEN'] += 1

    def instruction(self, _code, offset):
        if offset not in self.offsets:
            return
        row = self.window.get(sys._getframe(1))
        if row is not None:
            row[1].add(offset)

    @staticmethod
    def _signature(row):
        return tuple(branch['arm'] for branch in row['branch_arms'])

    def _retain(self, sample):
        rows = self.samples[sample['table_version']]
        if len(rows) < self.design['limits']['per_version']:
            rows.append(sample)
            return
        signatures = Counter(map(self._signature,rows))
        current = self._signature(sample)
        if current in signatures:
            return
        slot = next((i for i,row in enumerate(rows) if signatures[self._signature(row)] > 1), None)
        if slot is not None:
            rows[slot] = sample

    def finish(self, code, offset, value):
        frame = sys._getframe(1)
        sample = self.window.pop(frame, None)
        if sample is None:
            return
        self._mask()
        (token, root, members, version), visited, row = sample
        try:
            current = frame.f_globals[self.design['table']]
            _need_score(current is root and len(current) == len(members)
                  and all(type(key) is str for key in current)
                  and all(current.get(name) is member for name, member in members),
                  'table changed across selected score invocation boundaries')
            _need_score(_bindings(frame, self.design['global_loads']) == (self.globals, self.builtins) and
                  get_ident() == row['thread_id'], 'sampled score namespace or thread changed')
            returned = site(code, offset)
            _need_score(returned['opname'] in ('RETURN_VALUE', 'RETURN_CONST'), 'normal scalar return did not occur')
            _need_score(row['receiver_fields'] == _position(frame.f_locals[self.design['receiver']],
                  self.design['receiver_fields']) and row['argument'] == _argument(frame.f_locals[self.design['argument']]),
                  'receiver/argument fields changed within selected score')
            arms = []
            for branch in self.design['branches']:
                if branch['test_offset'] not in visited:
                    arm = 'NOT_REACHED'
                elif branch['then_offset'] in visited:
                    arm = 'THEN'
                elif branch['else_offset'] is None:
                    arm = 'FALSE'
                else:
                    _need_score(branch['else_offset'] in visited, 'selected ordinary branch did not enter either arm')
                    arm = 'ELSE'
                arms.append(dict(id=branch['id'], arm=arm))
            row.update(result=_int(value), return_site=returned, branch_arms=arms)
            self._retain(row)
            if (len(self.attempted) == self.design['limits']['max_versions'] and
                    all(count >= self.design['limits']['attempts_per_version'] for count in self.attempted.values())):
                self.active = False
                self.monitor.set_local_events(self.slot, code, 0)
        except (ValueError, KeyError):
            self.frontiers['NORMAL_SCALAR_RESULT_OR_BOUNDARY_OPEN'] += 1

    def close(self):
        self.active = False
        if self.code is not None:
            self.monitor.set_local_events(self.slot, self.code, 0)


class _Transitions:
    def __init__(self, design):
        self.design = design
        self.descriptors = {'score':design['score'], **{role:design[role]['code']
            for role in ('producer','rotation','consumer')}}
        self.wanted = {_target_coordinate(code):role for role,code in self.descriptors.items()}
        self.codes = {}
        self.frames = {}
        self.active_moves = {}
        self.children = {}
        self.pending = {}
        self.roots = []
        self.versions = {}
        self.recent = None
        self.frontiers = Counter()
        self.counts = Counter()
        self.samples = []
        self.ordinal = 0
        self.globals = self.builtins = None
        self.state_fields = None
        self.constructor = None

    def attach(self, event, args):
        if not self.active or event != 'exec' or not args or type(args[0]) is not CodeType:
            return
        root = args[0]
        if root.co_filename != self.design['source']['uri']:
            return
        for code in _codes(root):
            role = self.wanted.get(_coordinate(code))
            if role is None:
                continue
            _need_transition(describe_code(code) == self.descriptors[role], 'original transition code differs')
            self.codes[code] = role
            mask = self.events.PY_START | self.events.PY_RETURN
            if role in ('producer','rotation'):
                mask |= self.events.CALL
            self.monitor.set_local_events(self.slot, code, mask)

    def _identity(self, frame):
        if frame not in self.frames:
            self.ordinal += 1
            self.frames[frame] = self.ordinal
        return self.frames[frame]

    def _table(self, root):
        if self.recent is not None:
            _, old, anchors, _ = self.recent
            if root is old and _unchanged(root, anchors):
                return self.recent
        token, anchors, version = _snapshot(self.roots, self.versions, root)
        _need_transition(len(self.versions) <= self.design['max_versions'], 'transition table version budget exceeded')
        self.recent = token, root, anchors, version
        return self.recent

    def _begin_move(self, frame, parent, selected):
        source = self.design
        self.pending.pop(parent, None)
        table = self._table(frame.f_globals[source['table']])
        token, _, _, version = table
        if self.counts[version] >= source['per_version']:
            return
        _need_transition(len(self.frames) < 256, 'live transition observation budget exceeded')
        locals_ = frame.f_locals
        carrier = frame.f_globals[source['constructor']['name']]
        instance = locals_[source['producer_receiver']]
        _need_transition(type(carrier) is type and type(instance) is carrier, 'original field owner differs from exact constructor')
        if self.constructor is None:
            self.constructor = carrier
        _need_transition(self.constructor is carrier, 'transition substituted the original tuple class')
        fields = _state(instance, carrier, source['constructor']['fields'])
        schema = [dict(name=name,kind=field_kind(value)) for name,value in fields.items()]
        if self.state_fields is None:
            self.state_fields = schema
        _need_transition(self.state_fields == schema, 'source state field ABI changed')
        constants, carriers = _bindings(frame, source['globals'])
        if self.globals is None:
            self.globals, self.builtins = constants, carriers
        _need_transition((self.globals,self.builtins) == (constants,carriers), 'transition source globals changed')
        caller, producer = self._identity(parent), self._identity(frame)
        move = locals_[source['producer_argument']]
        data = dict(table_version=version, root_token=token, thread_id=get_ident(),
            caller_frame=caller, producer_frame=producer, producer_site=selected, receiver_fields=fields,
            argument=_argument(move), constructors={})
        self.active_moves[frame] = dict(parent=parent, receiver=instance, argument=move, table=table,
                                       data=data, rotated=None)

    def begin(self, code, _offset):
        frame = sys._getframe(1)
        parent = frame.f_back
        if parent is None:
            return
        role = self.codes[code]
        source = self.design
        selected = (source['score_call'] if role == 'score' else source[role]['caller_site'])
        if parent.f_lasti != selected['offset'] or site(parent.f_code,parent.f_lasti) != selected:
            return
        try:
            if role == 'producer':
                self._begin_move(frame, parent, selected)
                return
            if role == 'consumer':
                entry = self.pending.pop(parent,None)
                if entry is None:
                    return
                data, returned = entry
                _need_transition(get_ident() == data['thread_id'] and frame.f_locals[source['consumer']['formal']] is returned,
                      'recursive consumer did not enter with the selected normal result')
                data.update(consumer_frame=self._identity(frame), consumer_site=selected, consumer_is_return=True)
                self.samples.append(data)
                self.counts[data['table_version']] += 1
                if (len(self.counts) == source['max_versions'] and
                    all(count >= source['per_version'] for count in self.counts.values())):
                    self.close()
                return

            state = self.active_moves.get(parent)
            if state is None:
                return
            data, loc = state['data'], frame.f_locals
            _need_transition(data['thread_id'] == get_ident(), 'transition crossed capture threads')
            if role == 'score':
                token, root, anchors, version = state['table']
                _need_transition('score_frame' not in data and loc[source['score_receiver']] is state['receiver'] and
                      loc[source['score_argument']] is state['argument'] and frame.f_globals[source['table']] is root
                      and _unchanged(root,anchors) and _bindings(frame,source['globals']) == (self.globals,self.builtins),
                      'scalar call did not bind the same exact producer inputs')
                data.update(score_frame=self._identity(frame),score_site=selected)
            else:
                _need_transition('producer' in data['constructors'] and 'score_delta' in data and 'rotation_frame' not in data,
                      'post-construction method lacks its selected shape and original score')
                formal, = source['rotation']['parameters']
                _need_transition(loc[formal] is False, 'expected actual false original default in the rotation frame')
                receiver = loc[source['rotation']['receiver']]
                data.update(rotation_frame=self._identity(frame),rotation_site=selected,rotation_arg=False,
                            intermediate_fields=_state(receiver,self.constructor,source['constructor']['fields']))
            self.children[frame] = parent
        except (ValueError,KeyError,TypeError):
            self.frontiers['ORIGINAL_TRANSITION_ENTRY_OPEN'] += 1

    def call(self, code, offset, target, _arg0):
        role = self.codes[code]
        options = self.design['constructor']['sites'][role]
        selected = next((row for row in options if row['offset'] == offset), None)
        if selected is None:
            return
        frame = sys._getframe(1)
        parent = self.children.get(frame) if role == 'rotation' else frame
        state = self.active_moves.get(parent)
        if state is None:
            return
        try:
            seen = state['data']['constructors']
            _need_transition(role not in seen and selected == site(code,offset) and
                  target is self.constructor is frame.f_globals[self.design['constructor']['name']] and type(target) is type,
                  'runtime constructor did not select the original exact type carrier')
            seen[role] = dict(site=selected, target=_class_metadata(target), type_token=1,
                protocol='OBSERVED_EXACT_TYPE_CARRIER')
        except (ValueError,KeyError):
            self.frontiers['ORIGINAL_CONSTRUCTOR_PROTOCOL_OPEN'] += 1

    def finish(self, code, offset, result):
        frame = sys._getframe(1)
        self.frames.pop(frame,None)
        self.pending.pop(frame,None)
        role = self.codes[code]
        owner = self.children.pop(frame,None)
        state = self.active_moves.pop(frame,None) if role == 'producer' else self.active_moves.get(owner)
        if state is None:
            return
        try:
            data = state['data']
            returned = site(code,offset)
            _need_transition(data['thread_id'] == get_ident() and returned['opname'] in ('RETURN_VALUE','RETURN_CONST'),
                  'expected selected ordinary normal return')
            if role == 'score':
                _, root, anchors, _ = state['table']
                _need_transition(frame.f_globals[self.design['table']] is root and _unchanged(root,anchors),
                      'selected score window changed table root or row identities')
                data.update(score_delta=_int(result),score_return=returned)
            elif role == 'rotation':
                _need_transition('rotation' in data['constructors'], 'recognized rotation constructor was not observed')
                data['rotation_return'] = returned
                state['rotated'] = result
            elif role == 'producer':
                _need_transition(result is state['rotated'] and {'producer','rotation'} == data['constructors'].keys()
                      and 'score_delta' in data and 'rotation_return' in data,
                      'final return does not carry the normal rotation result')
                fields = _state(result,self.constructor,self.design['constructor']['fields'])
                data.update(producer_return=returned,returned_fields=fields,producer_is_rotation=True)
                self.pending[state['parent']] = data,result
        except (ValueError,KeyError,TypeError):
            self.frontiers['NORMAL_TRANSITION_RETURN_OPEN'] += 1

    def close(self):
        self.active = False
        for code in self.codes:
            self.monitor.set_local_events(self.slot,code,0)
        self.frames.clear()
        self.children.clear()
        self.active_moves.clear()
        self.pending.clear()


def _capture_handoff(design_ref, design, out):
    handoff_ref = design['handoff_selection']
    handoff = read_artifact(handoff_ref)
    _need(design['schema_version'] == 1 and handoff['source'] == design['source']
          and handoff['producer']['code'] == design['producer_code'], 'score and handoff launches differ')
    handlers = (('PY_START', 'enter'), ('PY_RETURN', 'finish'), ('INSTRUCTION', 'instruction'))
    with observe_handoff(handoff) as flow:
        with _record(_Scores(design, flow), TABLE_CAPTURE, handlers, _need,
                     'local monitoring scope required') as scalars:
            launch = run_original_script(design['source']['uri'], handoff['argv'])
    out = Path(out)
    raw_flow = _write(out/'handoff-capture.json', dict(schema=HANDOFF, schema_version=1, selection=handoff_ref,
        process_id=flow.process, thread_id=flow.thread, launch=launch, matched=flow.rows,
        observation_frontiers=flow.frontiers, reference_test='LIVE_RETURN_OBJECT_IS_SELECTED_ENTRY_FORMAL',
        values_saved=False, original_execution=True, performance_measurement=False))
    samples = [dict(scalars.samples[row['producer']], handoff_index=index)
        for index, row in enumerate(flow.rows) if row['producer'] in scalars.samples]
    _write(out/'capture.json', dict(schema=TABLE_CAPTURE, schema_version=1, selection=design_ref,
        source=design['source'], case=design['case'], case_index=design['case_index'], handoff_capture=raw_flow,
        process_id=os.getpid(), thread_id=flow.thread, table_versions=scalars.versions,
        samples=samples, observation_frontiers=dict(scalars.frontiers),
        table_scope='EXACT_BUILTIN_ROW_SNAPSHOTS_AT_SELECTED_EXPRESSION_BOUNDARIES',
        original_execution=True, performance_measurement=False))


def _capture_score(reference, design, out):
    handlers = (('PY_START', 'begin'), ('PY_RETURN', 'finish'), ('INSTRUCTION', 'instruction'))
    with _record(_Probe(design), SCORE_CAPTURE, handlers, _need_score,
                 'global monitoring mask unavailable for scoped capture') as observer:
        launch = run_original_script(design['source']['uri'], [])
    kept = {version: observer.versions[version] for version in observer.samples}
    samples = sorted((row for group in observer.samples.values() for row in group), key=lambda row: row['frame_ordinal'])
    return _write(out, dict(schema=SCORE_CAPTURE, schema_version=1, selection=reference, source=design['source'],
        case=design['case'], case_index=design['case_index'], launch=launch, process_id=os.getpid(),
        table_versions=kept, globals=observer.globals, builtins=observer.builtins,
        builtin_abs_verified=bool(observer.builtins),
        attempts_per_version=dict(observer.attempted), samples=samples, observation_frontiers=dict(observer.frontiers),
        original_execution=True, performance_measurement=False))


def _capture_transition(reference, design, out):
    handlers = (('PY_START', 'begin'), ('PY_RETURN', 'finish'), ('CALL', 'call'))
    with _record(_Transitions(design), TRANSITION_CAPTURE, handlers, _need_transition,
                 'original transition requires local monitor events only') as observer:
        launch = run_original_script(design['source']['uri'], [])
    return _write(out,dict(schema=TRANSITION_CAPTURE,schema_version=1,selection=reference,source=design['source'],
        score_evidence=design['score_evidence'],launch=launch,process_id=os.getpid(),
        state_fields=observer.state_fields,globals=observer.globals,builtins=observer.builtins,
        builtin_abs_verified=bool(observer.builtins),table_versions={version:observer.versions[version]
            for version in observer.counts}, samples=observer.samples, observation_frontiers=dict(observer.frontiers),
        original_execution=True,performance_measurement=False))


_CAPTURES = {
    TABLE_DESIGN: _capture_handoff,
    SCORE_DESIGN: _capture_score,
    TRANSITION_DESIGN: _capture_transition,
}


def capture(selection, out):
    reference, design = open_artifact(Path(selection).resolve(strict=True))
    observe = _CAPTURES.get(reference['schema'])
    if observe is None:
        raise ValueError('expected a scalar handoff, score, or transition selection')
    read_artifact(design['source'], json_document=False)
    return observe(reference, design, out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    capture(args.selection, args.out)


if __name__ == '__main__':
    main()
