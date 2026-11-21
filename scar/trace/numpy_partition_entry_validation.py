"""Observe complete state digests at natural Python result read boundaries."""

from contextlib import contextmanager
from functools import cached_property
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import sys
from types import FunctionType, ModuleType
import weakref

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.trace.code_snapshot import code_key


SCHEMA = 'scar.numpy-partition-entry-trajectory'
RETURN_RELATION_OBSERVATION = 'NATURAL_GETTER_CALLER_AND_WEAK_PREVIOUS_VALUE'
WEAK_RETURN_IDENTITY = 'PYTHON_IS_WITH_WEAK_PREVIOUS_PROPERTY_VALUE'
_SOURCE = 'scar.python-source'
_SUSPENDED = inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR


def _member(cls, name):
    member = inspect.getattr_static(cls, name)
    if type(member) is FunctionType:
        function, kind = member, 'METHOD'
    elif type(member) is property:
        function, kind = member.fget, 'PROPERTY_GETTER'
    elif type(member) is cached_property:
        function, kind = member.func, 'CACHED_PROPERTY_GETTER'
    else:
        raise ValueError('expected an exact Python method or property getter')
    if type(function) is not FunctionType or function.__code__.co_flags & _SUSPENDED:
        raise ValueError('expected an ordinary synchronous Python endpoint')
    return function, kind


def method_binding(cls, member):
    """Describe a trusted loaded class member without invoking its descriptor."""
    function, kind = _member(cls, member)
    code = function.__code__
    return dict(module=cls.__module__, class_name=cls.__qualname__, method=member,
                kind=kind, code_id=code_key(code),
                source=artifact_reference(Path(code.co_filename).resolve(strict=True), _SOURCE))


def _loaded_class(binding):
    module = sys.modules.get(binding['module'])
    if type(module) is not ModuleType:
        raise ValueError('configured endpoint module is not loaded')
    cls = module
    for part in binding['class_name'].split('.'):
        cls = inspect.getattr_static(cls, part)
    if not isinstance(cls, type):
        raise ValueError('configured endpoint receiver is not a class')
    return cls


def _get_binding(binding):
    cls = _loaded_class(binding)
    if method_binding(cls, binding['method']) != binding:
        raise ValueError('live source or endpoint binding differs')
    return cls, _member(cls, binding['method'])[0]


def _state(value, max_bytes, *, allow_scalar=False):
    if allow_scalar and type(value) is int:
        return dict(kind='EXACT_INT', value=value)
    numpy = sys.modules.get('numpy')
    if (type(numpy) is not ModuleType or type(value) is not vars(numpy).get('ndarray')
            or value.dtype.kind not in 'biufc' or value.dtype.hasobject or value.nbytes > max_bytes):
        raise ValueError('bounded exact NumPy state required at the observed endpoint')
    raw = memoryview(value).cast('B') if value.flags.c_contiguous else value.tobytes(order='C')
    return dict(kind='NDARRAY', dtype=value.dtype.str, shape=list(value.shape), size_bytes=value.nbytes,
                order='C', sha256=sha256(raw).hexdigest())


class _Trajectory:
    def __init__(self, spec, arguments, max_rounds, max_bytes, retain_return_relation):
        self.spec, self.arguments = spec, arguments
        self.max_rounds, self.max_bytes = max_rounds, max_bytes
        self.retain_return_relation = retain_return_relation
        self.call, self.result = spec['call'], spec['result']
        if self.call['kind'] != 'METHOD' or self.result['kind'] not in ('PROPERTY_GETTER', 'CACHED_PROPERTY_GETTER'):
            raise ValueError('expected a call and a naturally read result property')
        for reference in (self.call['source'], self.result['source'], spec['caller_source']):
            if reference['schema'] != _SOURCE:
                raise ValueError('endpoint source artifact type differs')
            read_artifact(reference, json_document=False)
        self.monitor = sys.monitoring
        self.events = self.monitor.events
        self.tool = next((i for i in range(6) if self.monitor.get_tool(i) is None), None)
        if self.tool is None:
            raise RuntimeError('no free CPython monitoring slot')
        self.call_function = self.result_function = self.call_code = self.result_code = None
        self.result_class = None
        self.active, self.completed, self.callers = {}, {}, {}
        self.previous_outputs = {}
        self.records = []
        self.calls, self.normal_returns = 0, 0
        self.cpu_ids = sorted(os.sched_getaffinity(0))
        self.error = None
        self.closed = False
        self.unresolved = False

    def _bind(self, code, frame):
        if (code.co_name != self.call['method']
                or os.path.abspath(code.co_filename) != self.call['source']['uri']):
            return False
        cls = _loaded_class(self.call)
        if type(frame.f_locals.get(code.co_varnames[0])) is not cls:
            return False
        _, self.call_function = _get_binding(self.call)
        self.result_class, self.result_function = _get_binding(self.result)
        self.call_code = self.call_function.__code__
        self.result_code = self.result_function.__code__
        if self.call_code is not code or self.result_code is code:
            raise ValueError('actual endpoint code differs')
        self.monitor.set_local_events(self.tool, code, self.events.PY_START | self.events.PY_RETURN)
        self.monitor.set_local_events(self.tool, self.result_code, self.events.PY_RETURN)
        self.monitor.set_events(self.tool, 0)
        return True

    def start(self, code, offset):
        if self.error is not None or (self.call_code is None and code.co_name != self.call['method']):
            return
        frame = sys._getframe(1)
        try:
            caller = frame.f_back
            if caller is None or os.path.abspath(caller.f_code.co_filename) != self.spec['caller_source']['uri']:
                return
            if self.call_code is None and not self._bind(code, frame):
                return
            if code is not self.call_code or type(frame.f_locals[code.co_varnames[0]]) is not _loaded_class(self.call):
                return
            if self.calls >= self.max_rounds:
                raise ValueError('trajectory round budget exceeded')
            values = frame.f_locals
            input_value = values[self.spec['input_parameter']]
            state = _state(input_value, self.max_bytes, allow_scalar=True)
            args = {name: project(values[name]) for name, project in self.arguments.items()}
            args = json.loads(json.dumps(args, sort_keys=True, allow_nan=False))
            key = id(caller)
            activation, round_index, _ = self.callers.get(key, (len(self.callers), 0, caller))
            self.callers[key] = activation, round_index + 1, caller
            row = dict(index=self.calls, activation=activation, round_index=round_index, input=state,
                       caller=dict(code_id=code_key(caller.f_code), offset=caller.f_lasti), arguments=args)
            previous = self.previous_outputs.get(activation)
            if self.retain_return_relation and previous is not None and previous[1] == round_index - 1:
                previous_value = previous[2]()
                row['input_object_relation'] = dict(method=WEAK_RETURN_IDENTITY, producer_index=previous[0],
                                                    previous_alive=previous_value is not None,
                                                    same_object=previous_value is input_value)
            self.active[id(frame)] = frame, caller, row
            self.calls += 1
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            self.error = str(error)

    def finish(self, code, offset, value):
        if self.error is not None:
            return
        frame = sys._getframe(1)
        try:
            if code is self.call_code:
                matched = self.active.pop(id(frame), None)
                if matched is None:
                    return
                if matched[0] is not frame or type(value) is not self.result_class or id(value) in self.completed:
                    raise ValueError('normal return does not bind one fresh result token')
                if self.retain_return_relation:
                    matched[2]['returned_state'] = dict(method_return_offset=offset)
                self.completed[id(value)] = value, matched[1], matched[2]
                self.normal_returns += 1
            elif code is self.result_code:
                self._read_result(frame, offset, value)
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            self.error = str(error)

    def _read_result(self, frame, offset, value):
        receiver = frame.f_locals[frame.f_code.co_varnames[0]]
        match = self.completed.get(id(receiver))
        if match is None or match[0] is not receiver:
            return
        caller = frame.f_back
        while caller is not None and caller is not match[1]:
            caller = caller.f_back
        if caller is None:
            raise ValueError('result property was read outside the observed caller activation')
        row = match[2]
        row['output'] = _state(value, self.max_bytes)
        if self.retain_return_relation:
            row['returned_state'].update(result_return_offset=offset,
                                         result_caller=dict(code_id=code_key(caller.f_code), offset=caller.f_lasti))
            self.previous_outputs[row['activation']] = (row['index'], row['round_index'], weakref.ref(value))
        self.records.append(row)
        del self.completed[id(receiver)]

    def close(self):
        self.monitor.clear_tool_id(self.tool)
        self.monitor.free_tool_id(self.tool)
        self.closed = True
        self.unresolved = bool(self.active or self.completed)
        self.active.clear()
        self.completed.clear()
        self.callers.clear()
        self.previous_outputs.clear()

    def report(self):
        """Return ordered detached read evidence after the natural endpoint completes."""
        if (not self.closed or self.error is not None or not self.records or self.unresolved
                or len(self.records) != self.calls or self.normal_returns != self.calls):
            raise ValueError('trajectory observation incomplete' + (': ' + self.error if self.error else ''))
        if (_get_binding(self.call)[1] is not self.call_function
                or _get_binding(self.result)[1] is not self.result_function
                or sorted(os.sched_getaffinity(0)) != self.cpu_ids):
            raise ValueError('endpoint binding or visible CPU affinity differs')
        read_artifact(self.spec['caller_source'], json_document=False)
        report = dict(schema=SCHEMA, schema_version=1, endpoints=self.spec, cpu_ids=self.cpu_ids,
                      records=sorted(self.records, key=lambda row: row['index']), calls=self.calls,
                      natural_result_property_reads=self.normal_returns,
                      performance_measured=False, capture_is_diagnostic=True)
        if self.retain_return_relation:
            report['return_relation_observation'] = RETURN_RELATION_OBSERVATION
        return json.loads(_canonical(report))


@contextmanager
def observe_states(spec, *, arguments=None, max_rounds=128, max_bytes=256 << 20, retain_return_relation=False):
    """Hash original method inputs and exact arrays returned by natural property reads.

    Optional argument projectors must be read-only and return JSON values.
    The opt-in relation records the natural getter site and weak previous value.
    """
    if (type(max_rounds) is not int or max_rounds < 1 or type(max_bytes) is not int or max_bytes < 1):
        raise ValueError('positive trajectory bounds required')
    if type(retain_return_relation) is not bool:
        raise ValueError('return relation observation requires an exact bool')
    spec = json.loads(json.dumps(spec, sort_keys=True, allow_nan=False))
    observer = _Trajectory(spec, dict(arguments or {}), max_rounds, max_bytes, retain_return_relation)
    observer.monitor.use_tool_id(observer.tool, 'scar-numpy-partition-entry-validation')
    try:
        observer.monitor.register_callback(observer.tool, observer.events.PY_START, observer.start)
        observer.monitor.register_callback(observer.tool, observer.events.PY_RETURN, observer.finish)
        observer.monitor.set_events(observer.tool, observer.events.PY_START)
        yield observer
    finally:
        observer.close()


def normalize_report(report, *, omit_paths=()):
    """Remove only configured report fields; ``*`` selects every list item."""
    value = json.loads(_canonical(report))

    def remove(node, path):
        part, *tail = path
        if part == '*':
            if type(node) is not list or not tail:
                raise ValueError('report wildcard must traverse a list to a named field')
            for item in node:
                remove(item, tail)
        elif type(node) is not dict or part not in node:
            raise ValueError('configured report field is absent: ' + part)
        elif tail:
            remove(node[part], tail)
        else:
            del node[part]

    for path in omit_paths:
        if type(path) not in (list, tuple) or not path or any(type(part) is not str for part in path):
            raise ValueError('report omissions must be explicit nonempty field paths')
        remove(value, path)
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def compare_trajectories(original, candidate, *, require_recurrence=False, expected_rounds=None):
    """Check full state digests and optionally the exact loop-carried input link."""
    if expected_rounds is not None and (type(expected_rounds) is not int or expected_rounds <= 0):
        raise ValueError('expected a positive exact round count')
    rows = original['records']
    for report in (original, candidate):
        if (report['schema'] != SCHEMA or report['schema_version'] != 1
                or report['capture_is_diagnostic'] is not True or report['performance_measured'] is not False
                or not rows or report['calls'] != len(report['records'])
                or report['natural_result_property_reads'] != report['calls']):
            raise ValueError('incomplete diagnostic trajectory report')
    if (original['endpoints'] != candidate['endpoints'] or original['cpu_ids'] != candidate['cpu_ids']
            or len(rows) != len(candidate['records'])):
        raise ValueError('trajectory source, CPU or call count differs')
    by_activation = {}
    for index, (before, after) in enumerate(zip(rows, candidate['records'], strict=True)):
        if before['index'] != index or _canonical(before) != _canonical(after):
            raise ValueError(f'trajectory state, producer or arguments differ at call {index}')
        previous = by_activation.setdefault(before['activation'], [])
        if (before['round_index'] != len(previous)
                or (require_recurrence and previous and previous[-1]['output'] != before['input'])):
            raise ValueError(f'loop-carried state differs at call {index}')
        previous.append(before)
    lengths = [len(activation) for activation in by_activation.values()]
    if expected_rounds is not None and any(length != expected_rounds for length in lengths):
        raise ValueError('observed trajectory round count differs')
    return dict(matched=True, full_state_digests_equal=True, matched_calls=len(rows),
                calls_per_activation=lengths, recurrence_checked=require_recurrence,
                cpu_ids=original['cpu_ids'], performance_measured=False)
