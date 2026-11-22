"""Recover an observed positional receiver from two adjacent call boundaries.

A method-form attribute load leaves the self-or-NULL slot unresolved. Here a
retained normal producer return identifies the last explicit actual, which
separates the zero-prefix and one-prefix layouts of the observed Python call.
This compiler-local proof keeps descriptor and future dispatch open.
"""

from bisect import bisect_left
from collections import defaultdict
import dis
import inspect
import re
import sys

from scar.ir.program.call_edges import FrameCompletion
from scar.ir.program.call_operands import CallableKind, OperandCallAttempt, OperandEntryHandshake, OperandGap, OperandScopedGap
from scar.ir.program.python_call_targets import PythonTargetKind
from scar.ir.record_codec import encode
from scar.trace.code_snapshot import code_parameters, site
from scar.trace.python_call_targets import load_slots


_UNSUPPORTED = (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS | inspect.CO_GENERATOR
    | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR)
_WEAK = re.compile(r'weak:[1-9][0-9]*:[1-9][0-9]*')


def _weak(reference, process):
    return (type(reference) is str and _WEAK.fullmatch(reference) is not None
        and reference.startswith(f'weak:{process}:'))


def _attempt_key(attempt):
    return attempt.process_id, attempt.index


def _index_unique(rows, key):
    result = {}
    for row in rows:
        identity = key(row)
        result[identity] = None if identity in result else row
    return result


class _Replay:
    def __init__(self, trace):
        self.trace = trace
        self.entries = _index_unique(trace.entries, lambda frame: frame.invocation)
        self.exits = _index_unique(trace.exits, lambda frame: frame.invocation)
        self.handshakes = _index_unique((row for row in trace.events if type(row) is OperandEntryHandshake),
            lambda row: (row.process_id, row.attempt_index))
        self.code = {}
        self.previous = {}
        last = {}
        gaps = defaultdict(list)
        for row in trace.events:
            if type(row) is OperandCallAttempt:
                prior = last.get(row.invocation)
                if prior is not None:
                    self.previous[_attempt_key(row)] = prior
                last[row.invocation] = row
            elif type(row) in (OperandGap, OperandScopedGap):
                key = row.process_id, None if row.invocation is None else row.thread_id
                gaps[key].append((row.index, row.invocation))
        self.gaps = {key: (tuple(i for i, _ in rows), tuple(rows)) for key, rows in gaps.items()}

    def instructions(self, code):
        cached = self.code.get(id(code))
        if cached is None:
            instructions = tuple(dis.get_instructions(code))
            positions = {row.offset: i for i, row in enumerate(instructions)}
            handlers = frozenset(row.target for row in dis.Bytecode(code).exception_entries)
            cached = (code, instructions, positions, handlers)
            self.code[id(code)] = cached
        return cached[1:]

    def uninterrupted(self, first, last, invocations):
        for key in ((first.process_id, None), (first.process_id, first.thread_id)):
            positions, rows = self.gaps.get(key, ((), ()))
            start = bisect_left(positions, first.index)
            stop = bisect_left(positions, last.index + 1)
            if any(owner is None or owner in invocations for _, owner in rows[start:stop]):
                return False
        return True


def _replay(index):
    trace = index.codes.trace
    cached = getattr(index, '_positional_receiver_replay', None)
    if cached is None or cached.trace is not trace:
        cached = _Replay(trace)
        index._positional_receiver_replay = cached
    return cached


def _direct_entry(replay, attempt):
    handshake = replay.handshakes.get(_attempt_key(attempt))
    if (handshake is None or handshake.thread_id != attempt.thread_id
            or handshake.hook_epoch != attempt.hook_epoch or handshake.index <= attempt.index
            or handshake.entry_index <= attempt.frame_index):
        return None, None
    frame = replay.entries.get(handshake.invocation)
    if (frame is None or frame.anchor.index != handshake.entry_index or frame.code != attempt.dispatch.code
            or frame.suspended or not frame.path_complete or frame.path != (attempt.site,)
            or frame.caller != attempt.site or frame.caller_invocation != attempt.invocation):
        return None, None
    return handshake, frame


def _exact_layout(attempt, frame):
    layout = attempt.dispatch.layout
    return (attempt.dispatch.kind is CallableKind.EXACT_FUNCTION and layout is not None
        and not layout.flags & _UNSUPPORTED and not layout.kwonlyargcount
        and frame.parameters_complete and tuple(row.name for row in frame.parameters) == layout.parameters)


def _writes_fast(instruction, name):
    opname = instruction.opname
    if 'STORE_FAST' not in opname and opname not in ('DELETE_FAST', 'LOAD_FAST_AND_CLEAR'):
        return False
    value = instruction.argval
    if type(value) is str:
        return value == name
    if type(value) is tuple and all(type(item) is str for item in value):
        return name in value
    return True


def _shape(replay, code, call_offset):
    instructions, positions, handlers = replay.instructions(code)
    call = positions.get(call_offset)
    if call is None:
        return None
    item = instructions[call]
    count = item.arg
    if item.opname != 'CALL' or type(count) is not int or count < 1:
        return None

    pos = call - 1
    explicit = []
    while pos >= 0 and len(explicit) < count:
        slots = load_slots(code, instructions[pos])
        if not slots or any(kind != 'FAST_LOCAL' for _, kind, _, _ in slots):
            return None
        explicit[0:0] = [(instructions[pos], slot) for _, _, _, slot in slots]
        pos -= 1
    if len(explicit) != count or pos < 3:
        return None
    attribute, receiver, store, producer = instructions[pos], instructions[pos - 1], instructions[pos - 2], instructions[pos - 3]
    receiver_slots = load_slots(code, receiver)
    if (attribute.opname != 'LOAD_ATTR' or type(attribute.arg) is not int or not attribute.arg & 1
            or not receiver_slots or len(receiver_slots) != 1 or receiver_slots[0][1] != 'FAST_LOCAL'
            or store.opname != 'STORE_FAST' or store.arg != explicit[-1][1]
            or producer.opname not in ('CALL', 'CALL_KW')):
        return None
    _, _, receiver_name, receiver_slot = receiver_slots[0]
    if (receiver_slot >= code.co_argcount or receiver_slot == store.arg or receiver_name in code.co_cellvars
            or any(_writes_fast(row, receiver_name) for row in instructions)
            or any(row.is_jump_target or row.offset in handlers for row in instructions[pos - 2:call + 1])):
        return None
    return count, receiver_slot, producer, store, receiver, explicit[-1][0]


def recover_positional_receiver(index, observation, frame):
    """Return a replayable witness for this observed call, if the layout closes."""
    if sys.implementation.name != 'cpython' or sys.version_info[:3] != (3, 14, 7):
        return None
    attempt, target = observation.attempt, observation.target
    if (observation.rule != 'EXPLICIT_SOURCE_CALL' or target.kind is not PythonTargetKind.FUNCTION
            or target.receiver is not None or target.caller != attempt.invocation
            or target.attempt_index != attempt.index or target.layout != attempt.dispatch.layout
            or target.code != attempt.dispatch.code
            or target.callable_reference != attempt.dispatch.reference or target.function_reference != target.callable_reference
            or attempt.arg0_missing or not _weak(attempt.arg0, attempt.process_id)
            or not _exact_layout(attempt, frame)):
        return None
    replay = _replay(index)
    caller_code = index.caller_for(observation).code
    shape = _shape(replay, caller_code, attempt.site.offset)
    if (shape is None or caller_code.co_flags & _UNSUPPORTED
            or encode(attempt.site) != site(caller_code, attempt.site.offset)):
        return None
    count, receiver_slot, producer, store, receiver, load = shape
    if target.layout.argcount != count + 1:
        return None
    caller = replay.entries.get(attempt.invocation)
    handshake, actual_entry = _direct_entry(replay, attempt)
    if (actual_entry != frame or caller is None or caller.suspended or caller.code != attempt.site.code
            or not caller.parameters_complete or tuple(row.name for row in caller.parameters) != code_parameters(caller_code)
            or caller.parameters[receiver_slot].ref != attempt.arg0 or frame.parameters[0].ref != attempt.arg0):
        return None

    preceding = replay.previous.get(_attempt_key(attempt))
    if (preceding is None or preceding.dispatch.kind is not CallableKind.EXACT_FUNCTION
            or preceding.hook_epoch != attempt.hook_epoch or preceding.invocation != attempt.invocation
            or preceding.index >= attempt.index
            or (preceding.site.offset, preceding.site.opname) != (producer.offset, producer.opname)
            or preceding.site.code != attempt.site.code
            or encode(preceding.site) != site(caller_code, producer.offset)):
        return None
    producer_handshake, producer_entry = _direct_entry(replay, preceding)
    if producer_entry is None or not _exact_layout(preceding, producer_entry):
        return None
    returned = replay.exits.get(producer_entry.invocation)
    if (returned is None or returned.code != producer_entry.code or returned.return_site.code != producer_entry.code
            or returned.completion is not FrameCompletion.NORMAL
            or returned.return_site.opname != 'RETURN_VALUE'
            or not producer_entry.anchor.index < returned.anchor.index == attempt.frame_index
            or not producer_handshake.index < attempt.index
            or not _weak(returned.return_ref, attempt.process_id)
            or returned.return_ref != frame.parameters[count].ref
            or returned.return_ref == frame.parameters[count - 1].ref
            or not replay.uninterrupted(preceding, handshake, {attempt.invocation, frame.invocation, producer_entry.invocation})):
        return None
    from scar.ir.program.source_observations import ObservedPositionalReceiver

    return ObservedPositionalReceiver(producer_attempt=preceding.anchor, producer_handshake=producer_handshake.anchor,
        producer_return=returned.anchor, caller_entry=caller.anchor, callee_entry=frame.anchor,
        explicit_position=count - 1, producer_store_offset=store.offset, explicit_load_offset=load.offset,
        receiver_load_offset=receiver.offset, receiver_local=caller_code.co_varnames[receiver_slot],
        explicit_local=caller_code.co_varnames[store.arg], receiver_reference=attempt.arg0,
        explicit_reference=returned.return_ref)
