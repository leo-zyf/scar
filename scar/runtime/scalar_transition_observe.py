"""Observe a bounded original scoring, tuple construction and consumption chain."""

import argparse
from collections import Counter
from pathlib import Path
import os
import sys
from threading import get_ident
from types import CodeType

from scar.analysis.scalar_table_handoff import _ref, _write
from scar.analysis.scalar_transition_evidence import CAPTURE, DESIGN, _need, field_kind
from scar.ir.execution_recipe import read_artifact
from scar.runtime.entry_runner import run_original_script
from scar.runtime.scalar_score_observe import _argument, _bindings
from scar.runtime.scalar_table_handoff_observe import _Scores, _field, _int
from scar.runtime.source_value_observe import _codes, _coordinate, _target_coordinate
from scar.trace.code_snapshot import describe_code, site


def _state(value, expected_type, fields):
    _need(type(value) is expected_type and tuple.__len__(value) == len(fields),
          'original transition object requires the selected exact tuple class')
    result = {}
    for index, name in enumerate(fields):
        cell = _field(value, name)
        _need(cell is tuple.__getitem__(value,index), 'named tuple descriptor does not match original field ordinal')
        serialized = list(cell) if type(cell) is tuple else cell
        field_kind(serialized)
        result[name] = serialized
    return result


def _class_metadata(cls):
    return dict(module=type.__getattribute__(cls,'__module__'), qualname=type.__getattribute__(cls,'__qualname__'))


def _unchanged(root, anchors):
    return (len(root) == len(anchors) and all(type(key) is str for key in root)
            and all(root.get(key) is row for key,row in anchors))


class _Transitions:
    def __init__(self, design):
        self.design = design
        self.monitor = sys.monitoring
        self.events = self.monitor.events
        self.slot = next(index for index in range(5,-1,-1) if self.monitor.get_tool(index) is None)
        self.descriptors = {'score':design['score'], **{role:design[role]['code']
            for role in ('producer','rotation','consumer')}}
        self.wanted = {_target_coordinate(code):role for role,code in self.descriptors.items()}
        self.active = True
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
            _need(describe_code(code) == self.descriptors[role], 'original transition code differs')
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
        token, anchors, version = _Scores._table(self, root)
        _need(len(self.versions) <= self.design['max_versions'], 'transition table version budget exceeded')
        self.recent = token, root, anchors, version
        return self.recent

    def _begin_move(self, frame, parent, selected):
        source = self.design
        self.pending.pop(parent, None)
        table = self._table(frame.f_globals[source['table']])
        token, _, _, version = table
        if self.counts[version] >= source['per_version']:
            return
        _need(len(self.frames) < 256, 'live transition observation budget exceeded')
        locals_ = frame.f_locals
        carrier = frame.f_globals[source['constructor']['name']]
        instance = locals_[source['producer_receiver']]
        _need(type(carrier) is type and type(instance) is carrier, 'original field owner differs from exact constructor')
        if self.constructor is None:
            self.constructor = carrier
        _need(self.constructor is carrier, 'transition substituted the original tuple class')
        fields = _state(instance, carrier, source['constructor']['fields'])
        schema = [dict(name=name,kind=field_kind(value)) for name,value in fields.items()]
        if self.state_fields is None:
            self.state_fields = schema
        _need(self.state_fields == schema, 'source state field ABI changed')
        constants, carriers = _bindings(frame, source['globals'])
        if self.globals is None:
            self.globals, self.builtins = constants, carriers
        _need((self.globals,self.builtins) == (constants,carriers), 'transition source globals changed')
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
                _need(get_ident() == data['thread_id'] and frame.f_locals[source['consumer']['formal']] is returned,
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
            _need(data['thread_id'] == get_ident(), 'transition crossed capture threads')
            if role == 'score':
                token, root, anchors, version = state['table']
                _need('score_frame' not in data and loc[source['score_receiver']] is state['receiver'] and
                      loc[source['score_argument']] is state['argument'] and frame.f_globals[source['table']] is root
                      and _unchanged(root,anchors) and _bindings(frame,source['globals']) == (self.globals,self.builtins),
                      'scalar call did not bind the same exact producer inputs')
                data.update(score_frame=self._identity(frame),score_site=selected)
            else:
                _need('producer' in data['constructors'] and 'score_delta' in data and 'rotation_frame' not in data,
                      'post-construction method lacks its selected shape and original score')
                formal, = source['rotation']['parameters']
                _need(loc[formal] is False, 'expected actual false original default in the rotation frame')
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
            _need(role not in seen and selected == site(code,offset) and
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
            _need(data['thread_id'] == get_ident() and returned['opname'] in ('RETURN_VALUE','RETURN_CONST'),
                  'expected selected ordinary normal return')
            if role == 'score':
                _, root, anchors, _ = state['table']
                _need(frame.f_globals[self.design['table']] is root and _unchanged(root,anchors),
                      'selected score window changed table root or row identities')
                data.update(score_delta=_int(result),score_return=returned)
            elif role == 'rotation':
                _need('rotation' in data['constructors'], 'recognized rotation constructor was not observed')
                data['rotation_return'] = returned
                state['rotated'] = result
            elif role == 'producer':
                _need(result is state['rotated'] and {'producer','rotation'} == data['constructors'].keys()
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


def capture(selection,out):
    reference = _ref(selection,DESIGN)
    design = read_artifact(reference)
    read_artifact(design['source'],json_document=False)
    observer = _Transitions(design)
    monitor,events,slot = observer.monitor,observer.events,observer.slot
    callbacks = ((events.PY_START,observer.begin), (events.PY_RETURN,observer.finish), (events.CALL,observer.call))
    monitor.use_tool_id(slot,CAPTURE)
    try:
        _need(monitor.get_events(slot) == 0,'original transition requires local monitor events only')
        for event,callback in callbacks:
            monitor.register_callback(slot,event,callback)
        sys.addaudithook(observer.attach)
        launch = run_original_script(design['source']['uri'],[])
    finally:
        observer.close()
        for event,_ in callbacks:
            monitor.register_callback(slot,event,None)
        monitor.free_tool_id(slot)
    return _write(out,dict(schema=CAPTURE,schema_version=1,selection=reference,source=design['source'],
        score_evidence=design['score_evidence'],launch=launch,process_id=os.getpid(),
        state_fields=observer.state_fields,globals=observer.globals,builtins=observer.builtins,
        builtin_abs_verified=bool(observer.builtins),table_versions={version:observer.versions[version]
            for version in observer.counts}, samples=observer.samples, observation_frontiers=dict(observer.frontiers),
        original_execution=True,performance_measurement=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    capture(args.selection,args.out)


if __name__ == '__main__':
    main()
