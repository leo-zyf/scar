"""Replay the original outer-frame route between two observed loop calls."""
from collections import deque
import dis
from pathlib import Path

from scar.analysis.source_observed_calls import _span
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.trace.code_snapshot import site
from scar.trace.control_transfers import MAX_TRANSFERS, SCHEMA


_CALL = frozenset(('CALL', 'CALL_KW', 'CALL_FUNCTION_EX'))
_JUMP = frozenset(('JUMP', 'JUMP_FORWARD', 'JUMP_BACKWARD', 'JUMP_BACKWARD_NO_INTERRUPT'))
_STOP = frozenset(('RETURN_VALUE', 'RETURN_CONST', 'RAISE_VARARGS', 'RERAISE', 'YIELD_VALUE'))


def _within(position, span):
    line, end, column, end_column = position
    first, last, start, finish = span
    return (None not in (*position, *span) and (first, start) <= (line, column)
            and (end, end_column) <= (last, finish))


def _compiler_latch(index, caller, loop, call_offset):
    """Find one supported normal `while` latch and its original test entry."""
    backedges = loop['control_backedges']
    if (len(backedges) != 1 or backedges[0]['source']['completion'] != 'FALLTHROUGH'
            or loop['source_loop'] is None or dis.Bytecode(caller.code).exception_entries):
        return None
    syntax = index._source_syntax(caller.unit)
    ordinal = loop['source_loop']['occurrence']
    node = syntax.nodes[ordinal]
    if node.constructor != 'While':
        return None
    fields = next(row.fields for row in syntax.profile.ast_schema if row.name == node.constructor)
    test = node.fields[fields.index('test')]
    span = _span(syntax, ordinal)
    condition = _span(syntax, test.reference)
    instructions = tuple(dis.get_instructions(caller.code, adaptive=False, show_caches=False))
    positions = {row.offset: row for row in instructions}
    candidates = tuple((row.offset, row.jump_target) for row in instructions
        if row.opname in _JUMP and row.jump_target in positions and row.jump_target <= call_offset < row.offset
        and _within(tuple(row.positions), span)
        and _within(tuple(positions[row.jump_target].positions), condition))
    if len(candidates) != 1:
        return None
    return candidates[0], instructions


def _replay_path(instructions, call_offset, interval):
    """Replay sequential spans plus all recorded transfers. Never execute bytecode."""
    positions = {row.offset: ordinal for ordinal, row in enumerate(instructions)}
    if call_offset not in positions:
        return None
    reads = deque(interval)
    ordinal = positions[call_offset] + 1
    visited = []
    while 0 <= ordinal < len(instructions) and len(visited) < MAX_TRANSFERS:
        instruction = instructions[ordinal]
        if instruction.offset == call_offset:
            return tuple(visited) if not reads else None
        if instruction.opname in _STOP:
            return None
        visited.append(instruction)
        if instruction.jump_target is None:
            ordinal += 1
            continue
        if not reads:
            return None
        _, row = reads.popleft()
        _, _, _, start, end, event = row
        unconditional = instruction.opname in _JUMP
        legal = {instruction.jump_target}
        if not unconditional and ordinal + 1 < len(instructions):
            following = instructions[ordinal + 1]
            legal.add(following.offset)
            # Monitoring reports the fallthrough past CPython's branch marker.
            if following.opname == 'NOT_TAKEN' and ordinal + 2 < len(instructions):
                legal.add(instructions[ordinal + 2].offset)
        if (start != instruction.offset or end not in legal or end not in positions
                or event != ('JUMP' if unconditional else 'BRANCH')):
            return None
        ordinal = positions[end]
    return None


def _transfer_rows(document):
    rows = document['transfers']
    if (type(rows) is not list or len(rows) > MAX_TRANSFERS
            or any(type(row) is not list or len(row) != 6 or any(type(v) is not int or v < 0 for v in row[:5])
                   or row[5] not in ('JUMP', 'BRANCH') for row in rows)
            or any(a[2] > b[2] for a, b in zip(rows, rows[1:]))):
        raise ValueError('invalid original control-transfer order')
    return rows


def observed_loop_control(index, parent_link, loop, windows, lifetime):
    """Tie an observed bytecode return-to-header to the original F4 backedge."""
    endpoint = lifetime['endpoints'][0]['invocation']
    process = endpoint['process_id'] if endpoint is not None else None
    trace = index.codes.trace
    path = Path(trace.directory) / f'control-transfers.{process}.json'
    if not path.is_file():
        return None
    if path.stat().st_size > 3_000_000:
        raise ValueError('original control-transfer receipt exceeds its bounded size')
    reference = artifact_reference(path, SCHEMA)
    document = read_artifact(reference)
    if (type(document['schema_version']) is not int or document['schema_version'] != 1
            or document['process_id'] != process):
        raise ValueError('original control-transfer process or version differs')
    result = dict(receipt=reference, exact_source_backedges=None, source_control_backedge=None,
                  same_outer_activation=False, direct_field_write_sites=[], intervening_call_sites=[],
                  actual_instruction_path_replayed=False, effect_closure_proven=False)
    if document['failure'] is not None or document['open_frames'] != 0 or len(windows) != 2:
        return dict(result, status='OPEN_CAPTURE_OR_CALL_WINDOW')
    first, second = [window['parent'] for window in windows]
    attempts = [tuple(row for row in index.bindings if row.result == parent_link.result
        and row.attempt.anchor.path == parent['attempt']['path']
        and row.attempt.index == parent['attempt']['index']) for parent in (first, second)]
    if any(len(rows) != 1 for rows in attempts):
        return dict(result, status='OPEN_ORIGINAL_CALL_CONTAINMENT')
    left, right = [rows[0] for rows in attempts]
    actor = left.attempt.invocation
    if actor != right.attempt.invocation or left.attempt.site != right.attempt.site:
        return dict(result, status='OPEN_DIFFERENT_OUTER_ACTORS_OR_CALLS')
    source = index.caller_for(left)
    if source != index.caller_for(right):
        return dict(result, status='OPEN_DIFFERENT_COMPILER_BODIES')
    entry, = [row for row in trace.entries if row.invocation == actor]
    exit_, = [row for row in trace.exits if row.invocation == actor]
    relative_event_file = Path(entry.anchor.path)
    frame_file, = [row for row in trace.frame_scan.files if Path(row.path) == relative_event_file]
    if frame_file.raw_sha256 != document['frame_events_sha256']:
        raise ValueError('control transfers name another original frame stream')
    frame = [actor.thread_id, actor.invocation_id, entry.anchor.index, exit_.anchor.index]
    if document['frames'].count(frame) != 1 or entry.code not in document['selected']:
        return dict(result, status='OPEN_OUTER_ACTIVATION_MONITORING_COVERAGE')
    mapped = _compiler_latch(index, source, loop, left.attempt.site.offset)
    if mapped is None:
        return dict(result, status='OPEN_ORIGINAL_SOURCE_LATCH_MAPPING')
    latch, instructions = mapped
    interval = tuple((i, row) for i, row in enumerate(_transfer_rows(document))
        if row[:2] == frame[:2] and first['exit']['index'] <= row[2] < second['entry']['index'])
    route = _replay_path(instructions, left.attempt.site.offset, interval)
    if route is None:
        return dict(result, status='OPEN_DISCONTINUOUS_OR_UNSUPPORTED_CONTROL_ROUTE')
    crossings = [f'#/transfers/{i}' for i, row in interval if tuple(row[3:5]) == latch]
    return dict(result, status='OBSERVED_EXACT_SOURCE_BACKEDGE_COUNT', exact_source_backedges=len(crossings),
                source_control_backedge=loop['control_backedges'][0],
                same_outer_activation=True, compiler_edge=list(latch), original_transfer_rows=crossings,
                observed_interval_transfer_rows=[f'#/transfers/{i}' for i, _ in interval],
                direct_field_write_sites=[site(source.code, row.offset) for row in route
                                          if row.opname in ('STORE_ATTR', 'DELETE_ATTR')],
                intervening_call_sites=[site(source.code, row.offset) for row in route if row.opname in _CALL],
                actual_instruction_path_replayed=True)
