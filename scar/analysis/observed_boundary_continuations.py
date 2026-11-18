"""Prove normal continuation of one recorded Python-side CALL occurrence."""

from bisect import bisect_left
from collections import defaultdict
import dis
import inspect
import sys
from typing import TYPE_CHECKING

from scar.ir.program.call_operands import OperandCallAttempt, OperandGap, OperandScopedGap
from scar.ir.program.source_observed_calls import ObservedSourceCallsite
from scar.ir.record_codec import encode
from scar.trace.code_snapshot import site

if TYPE_CHECKING:
    from scar.analysis.source_observed_calls import ObservedSourceCallsiteIndex
    from scar.ir.program.source_observed_boundaries import ObservedBoundaryContinuation


CONTINUATION_CONDITIONS = (
    'RECORDED_OCCURRENCE_ONLY',
    'ORDINARY_CPYTHON_3_14_7_FRAME_CONTROL_FLOW',
    'NO_DEBUGGER_FRAME_JUMP_OR_OUT_OF_PROTOCOL_FRAME_REENTRY',
)

_CALLS = frozenset(('CALL', 'CALL_KW', 'CALL_FUNCTION_EX'))
_SUSPENDING = inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR
_UNPLACED_GAPS = frozenset(('COLLECTION_GAP', 'FRAME_COLLECTION_GAP'))
# These instructions can have Python or native effects, but they have no
# normal successor other than fall-through in the current frame. The
# interval must additionally have no local exception handler or entry edge.
_FALL_THROUGH = frozenset((
    'BINARY_OP', 'BINARY_SLICE', 'BUILD_LIST', 'BUILD_MAP', 'BUILD_SET',
    'BUILD_SLICE', 'BUILD_STRING', 'BUILD_TUPLE', 'COMPARE_OP', 'CONTAINS_OP',
    'COPY', 'EXTENDED_ARG', 'IS_OP', 'LOAD_ATTR', 'LOAD_CONST', 'LOAD_DEREF',
    'LOAD_FAST', 'LOAD_FAST_BORROW', 'LOAD_FAST_BORROW_LOAD_FAST_BORROW',
    'LOAD_FAST_CHECK', 'LOAD_FAST_LOAD_FAST', 'LOAD_GLOBAL', 'LOAD_NAME',
    'LOAD_SMALL_INT', 'POP_TOP', 'PUSH_NULL', 'STORE_ATTR', 'STORE_DEREF',
    'STORE_FAST', 'STORE_FAST_LOAD_FAST', 'STORE_FAST_STORE_FAST', 'STORE_NAME',
    'STORE_SLICE', 'STORE_SUBSCR', 'SWAP', 'TO_BOOL', 'UNPACK_EX', 'UNPACK_SEQUENCE',
))


class _Replay:
    def __init__(self, trace):
        self.trace = trace
        self.pairs = {}
        self.code = {}
        self.compatible_processes = frozenset(meta.process_id for meta in trace.metadata
            if meta.implementation == sys.implementation.name and meta.python == sys.version)
        self.unplaced_gaps = frozenset(issue.process_id for issue in trace.issues
            if issue.kind in _UNPLACED_GAPS)
        previous = {}
        gaps = defaultdict(list)
        code_gaps = defaultdict(list)
        for event in trace.events:
            if type(event) is OperandCallAttempt:
                prior = previous.get(event.invocation)
                if prior is not None:
                    self.pairs[prior.key] = prior, event
                previous[event.invocation] = event
            elif type(event) is OperandScopedGap and event.invocation is None:
                code_gaps[event.process_id, event.thread_id, event.code].append(event.index)
            elif type(event) in (OperandGap, OperandScopedGap):
                gaps[event.process_id, event.invocation].append(event.index)
        self.gaps = dict(gaps)
        self.code_gaps = dict(code_gaps)

    def has_gap(self, first, last):
        if first.process_id in self.unplaced_gaps:
            return True
        process = first.process_id
        for positions in (self.gaps.get((process, None), ()),
                self.gaps.get((process, first.invocation), ()),
                self.code_gaps.get((process, first.thread_id, first.site.code), ())):
            position = bisect_left(positions, first.index)
            if position < len(positions) and positions[position] <= last.index:
                return True
        return False

    def instructions(self, code):
        cached = self.code.get(id(code))
        if cached is None:
            instructions = tuple(dis.get_instructions(code, adaptive=False))
            positions = {row.offset: i for i, row in enumerate(instructions)}
            handlers = tuple(dis.Bytecode(code).exception_entries)
            targets = frozenset(handler.target for handler in handlers)
            cached = code, instructions, positions, handlers, targets
            self.code[id(code)] = cached
        return cached[1:]


def _replay(index):
    trace = index.codes.trace
    cached = getattr(index, '_source_boundary_continuation_replay', None)
    if cached is None or cached.trace is not trace:
        cached = _Replay(trace)
        index._source_boundary_continuation_replay = cached
    return cached


def _straight_line(replay, code, first, last):
    if code.co_flags & _SUSPENDING:
        return False
    instructions, positions, handlers, targets = replay.instructions(code)
    start, stop = positions.get(first.offset), positions.get(last.offset)
    if start is None or stop is None or start >= stop:
        return False
    if instructions[start].opname not in _CALLS or instructions[stop].opname not in _CALLS:
        return False
    if encode(first) != site(code, first.offset) or encode(last) != site(code, last.offset):
        return False
    if any(handler.start < last.offset and first.offset < handler.end for handler in handlers):
        return False
    if any(item.is_jump_target or item.offset in targets for item in instructions[start + 1:stop + 1]):
        return False
    return all(item.opname in _FALL_THROUGH for item in instructions[start + 1:stop])


def normal_continuation(
    index: 'ObservedSourceCallsiteIndex', observation: ObservedSourceCallsite,
) -> 'ObservedBoundaryContinuation | None':
    """Return the next verified same-frame CALL witness under the stated conditions.

    This result establishes that the earlier CALL returned to its caller.
    It carries no return value, callee effects, or device-completion claim.
    """
    if sys.implementation.name != 'cpython' or sys.version_info[:3] != (3, 14, 7):
        return None
    if type(observation) is not ObservedSourceCallsite:
        return None
    attempt = observation.attempt
    replay = _replay(index)
    if attempt.process_id not in replay.compatible_processes:
        return None
    pair = replay.pairs.get(attempt.key)
    if pair is None:
        return None
    previous, successor = pair
    if (attempt != previous or successor.invocation != attempt.invocation
            or successor.hook_epoch != attempt.hook_epoch or successor.index <= attempt.index
            or successor.frame_index < attempt.frame_index or replay.has_gap(attempt, successor)):
        return None
    if not _straight_line(replay, index.caller_for(observation).code, attempt.site, successor.site):
        return None

    from scar.ir.program.source_observed_boundaries import ObservedBoundaryContinuation
    return ObservedBoundaryContinuation(successor.anchor, successor.site, successor.hook_epoch)


__all__ = ['CONTINUATION_CONDITIONS', 'normal_continuation']
