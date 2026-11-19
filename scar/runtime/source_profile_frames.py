"""Bounded local frame metadata selected by exact retained code coordinates."""

from contextlib import contextmanager
import os
import sys
from threading import get_ident
from types import CodeType
from weakref import ref

from scar.ir.execution_recipe import read_artifact
from scar.runtime.source_value_observe import _codes, _coordinate, _instance_field, _target_coordinate
from scar.trace import call_binding
from scar.trace.code_snapshot import code_key, describe_code


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


class _Frames:
    def __init__(self, selectors):
        self.selectors = {}
        for role, selector in selectors.items():
            code = selector['code']
            coordinate = _target_coordinate(code)
            read_artifact(selector['source'], json_document=False)
            _require(selector['source']['uri'] == coordinate[0] and coordinate not in self.selectors,
                     'profile frame selector must name a unique exact source body')
            self.selectors[coordinate] = role, selector
        self.paths = {coordinate[0] for coordinate in self.selectors}
        self.monitor = sys.monitoring
        self.events = self.monitor.events
        self.slot = next(slot for slot in range(5, -1, -1) if self.monitor.get_tool(slot) is None)
        self.current = {}
        self.frames = {}
        self.roles = {}
        self.records = []
        self.attempts = set()
        self.objects = {}
        self.object_ordinal = 0
        self.active = False
        self.thread = get_ident()
        self.pid = os.getpid()

    def attach(self, event, args):
        if not self.active or event != 'exec' or not args or type(args[0]) is not CodeType:
            return
        root = args[0]
        if root.co_filename not in self.paths:
            return
        for code in _codes(root):
            selected = self.selectors.get(_coordinate(code))
            if selected is None:
                continue
            _require(describe_code(code) == selected[1]['code'], 'profile frame loaded different bytecode')
            self.current[code] = selected
            events = self.events.PY_START
            if selected[1]['phase'] == 'NORMAL_RETURN':
                events |= self.events.PY_RETURN
            self.monitor.set_local_events(self.slot, code, events)

    def _metadata(self, value, operand):
        kind = operand['kind']
        if kind == 'COUNT':
            _require(type(value) is int and value > 0, 'selected count must be a positive exact integer')
            return {'formal': operand['formal'], 'value': value, 'type': ['builtins', 'int']}
        identity = id(value)
        pair = self.objects.get(identity)
        if pair is None or pair[0]() is not value:
            try:
                weak = ref(value)
            except TypeError:
                weak = None
            pair = (None, None)
            if weak is not None:
                self.object_ordinal += 1
                pair = (weak, self.object_ordinal)
                self.objects[identity] = pair
        descriptor = call_binding._type_metadata(type(value))
        data = {'type': [descriptor['module'], descriptor['qualname']], 'object_token': pair[1]}
        if kind == 'ARRAY':
            numpy = sys.modules['numpy']
            _require(type(value) is numpy.ndarray, 'profile metadata requires an exact ndarray')
            data.update(shape=list(value.shape), dtype=value.dtype.str)
        return data

    def capture(self, frame, row, selector, returned=None):
        for name, operand in selector['operands'].items():
            value = returned if operand['origin'] == 'RETURN' else frame.f_locals[operand['formal']]
            for field in operand.get('path', ()):
                value = _instance_field(value, field)
            row['fields'][name] = self._metadata(value, operand)

    def start(self, code, _offset):
        frame = sys._getframe(1)
        caller = frame.f_back
        role, selector = self.current[code]
        site = selector['caller_site']
        if caller is None or caller.f_lasti != site['offset'] or code_key(caller.f_code) != site['code']:
            return
        _require(get_ident() == self.thread and os.getpid() == self.pid, 'selected frame escaped the original owner')
        ordinals = self.roles.setdefault(role, [])
        _require(len(ordinals) < selector['max_activations'], 'selected frame exceeded its finite activation budget')
        collector = call_binding._installed
        _require(collector is not None and collector.records, 'missing bounded CALL attempt for selected frame')
        latest = collector.records[-1]
        matches = (latest['caller_code_key'], latest['offset'], latest['thread_id']) == (
            site['code'], site['offset'], self.thread)
        attempt = None
        if matches and latest['index'] not in self.attempts:
            attempt = latest['index']
            self.attempts.add(attempt)
        row = dict(activation=len(self.records)+1, code=selector['code'], caller_code=site['code'],
                   caller_offset=site['offset'], process_id=self.pid, thread_id=self.thread,
                   raw_call_attempt=attempt, phase='ENTERED', normal_return_offset=None, fields={})
        ordinals.append(len(self.records))
        self.records.append(row)
        if selector['phase'] == 'ENTERED':
            self.capture(frame, row, selector)
        else:
            self.frames[frame] = row

    def finish(self, code, offset, result):
        frame = sys._getframe(1)
        row = self.frames.pop(frame, None)
        if row is None:
            return
        _, selector = self.current[code]
        _require(offset in selector['normal_return_offsets'], 'selected frame returned outside its admitted sites')
        self.capture(frame, row, selector, result)
        row.update(phase='NORMAL_RETURN', normal_return_offset=offset)


@contextmanager
def observe_frames(selectors):
    """Run alongside other local capture tools in one dedicated subprocess."""
    observer = _Frames(selectors)
    monitor, slot, events = observer.monitor, observer.slot, observer.events
    monitor.use_tool_id(slot, 'scar.source-profile-frames')
    observer.active = True
    try:
        monitor.register_callback(slot, events.PY_START, observer.start)
        monitor.register_callback(slot, events.PY_RETURN, observer.finish)
        sys.addaudithook(observer.attach)
        yield observer
        _require(not observer.frames, 'selected frame did not return normally')
    finally:
        observer.active = False
        for code in observer.current:
            monitor.set_local_events(slot, code, 0)
        monitor.register_callback(slot, events.PY_START, None)
        monitor.register_callback(slot, events.PY_RETURN, None)
        monitor.free_tool_id(slot)
