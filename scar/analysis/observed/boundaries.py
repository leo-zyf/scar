"""Observed boundaries."""

import ast
import json
import dis
import inspect
import sys
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from scar.ir.program.call_edges import CallEdgeBudget, CapturedCode, FrameCompletion
from scar.ir.record_codec import decode, encode
from bisect import bisect_left
from typing import TYPE_CHECKING
from scar.ir.program.call_operands import OperandCallAttempt, OperandGap, OperandScopedGap
from scar.ir.program.source_observations import ObservedSourceCallsite
from scar.trace.code_snapshot import site
from scar.analysis.source_resource_scopes import ResourceScopeStore
from scar.ir.program.resource_observations import ObservedPythonResourceScope
from scar.trace.os_resources import _WRITE


# Select canonical bodies by exact observed code; retain every source template.
#
# This schedules offline lowering, not program reachability or transformation.
# Unobserved declarations and unresolved compiler mappings stay in the ledger.
class _DiscardFrames:
    def entry(self, value): pass
    def exit(self, value): pass
    def abort(self): pass


def read_observed_scope(trace):
    from scar.analysis.call_trace_reader import scan_call_trace
    trace = Path(trace).resolve()
    metadata = sorted(trace.glob('call-meta.*.json'))
    if not metadata or any(p.stat().st_size > 2_000_000 for p in metadata):
        raise ValueError('observed body scope requires original bounded call metadata')
    roots = {json.loads(p.read_bytes())['project_root'] for p in metadata}
    if len(roots) != 1:
        raise ValueError('observed body scope requires one capture namespace')
    budget = CallEdgeBudget(max_events=300000, max_raw_bytes=256_000_000,
        max_lines=600000, max_codes=8192, max_source_files=4096)
    return scan_call_trace(trace, roots.pop(), budget, sink=_DiscardFrames())


def select_observed_bodies(model, trace, entry, *, scope=None):
    from scar.analysis.static.compilation import _scope_nodes
    from scar.analysis.static_program import _read_fragment
    from scar.trace.code_snapshot import describe_code
    model, trace, entry = Path(model), Path(trace).resolve(), Path(entry)
    scan = read_observed_scope(trace) if scope is None else scope
    if Path(scan.directory) != trace:
        raise ValueError('body selection scope belongs to a different trace')
    captured = defaultdict(set)
    for capture in scan.captures:
        captured[capture.code.filename].add(capture.code)
    report = json.loads((model / 'manifest.json').read_bytes())
    selected, matched, mapping_issues = set(), set(), []
    for unit in report['units']:
        observations = captured.get(unit['path'], ())
        if not observations and unit['path'] != str(entry):
            continue
        source = _read_fragment(model, unit['artifacts']['source'])
        raw = Path(unit['path']).read_bytes()
        if sha256(raw).hexdigest() != unit['raw_sha256']:
            raise ValueError('source changed before observed body selection')
        code = compile(raw, unit['path'], 'exec', flags=0, dont_inherit=True, optimize=0)
        mappings, issues = {}, []
        _scope_nodes(ast.parse(source['text'], filename=unit['path']), code, (), ast.Module, mappings, issues)
        mapping_issues.extend(dict(file=unit['path'], detail=str(issue)) for issue in issues)
        catalog = _read_fragment(model, unit['artifacts']['lexical'])
        for declaration in catalog['entries']:
            if declaration['kind'] == 'MODULE':
                candidates = (code,)
            else:
                ref = declaration.get('source_ref')
                if ref is None:
                    continue
                span = (ref['start_line'], ref['end_line'], ref['start_column'], ref['end_column'])
                candidates = tuple(c for _, c, _ in mappings.get(span, ()))
            for candidate in candidates:
                descriptor = decode(CapturedCode, describe_code(candidate))
                if descriptor in observations:
                    selected.add(declaration['entry_key'])
                    matched.add(descriptor.key)
            if declaration['kind'] == 'MODULE' and unit['path'] == str(entry):
                selected.add(declaration['entry_key'])
    if not selected:
        raise ValueError('observed body scope selected no original source entries')
    admitted_paths = {u['path'] for u in report['units']}
    unmatched = [dict(file=code.filename, qualname=code.qualname, code=code.key)
        for codes in captured.values() for code in codes
        if code.filename in admitted_paths and code.key not in matched]
    return tuple(sorted(selected)), dict(schema='scar.observed-body-selection', schema_version=1,
        selected_entries=sorted(selected), captured_codes=sum(map(len, captured.values())),
        matched_codes=len(matched), unmatched_admitted_codes=unmatched, mapping_issues=mapping_issues,
        trace_issues=[str(issue) for issue in scan.issues],
        source_templates_dropped=0, unobserved_paths_proven_dead=False,
        loop_iterations_expanded=0, target_executed_by_selector=False)


# Prove normal continuation of one recorded Python-side CALL occurrence.
if TYPE_CHECKING:
    from scar.analysis.source_observed_calls import ObservedSourceCallsiteIndex
    from scar.ir.program.source_observations import ObservedBoundaryContinuation


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

    from scar.ir.program.source_observations import ObservedBoundaryContinuation
    return ObservedBoundaryContinuation(successor.anchor, successor.site, successor.hook_epoch)


# Retain the observed order of enclosing source calls and external writes.
def _parents(entries, invocation):
    while invocation in entries:
        child = entries[invocation]
        parent = child.caller_invocation
        if parent is None:
            return
        ancestor = entries.get(parent)
        if (ancestor is None or ancestor.anchor.path != child.anchor.path
                or ancestor.anchor.index >= child.anchor.index
                or (parent.process_id, parent.thread_id) != (invocation.process_id, invocation.thread_id)):
            return
        yield parent
        invocation = parent


def project_observed_effect_order(scope_report, source):
    """Join a source-replayed recipe to one capture, without scheduling its I/O."""
    store = ResourceScopeStore(scope_report)
    if store.index is None:
        raise ValueError('observed effect order requires a source-bound scope report')
    trace = store.index.codes.trace
    entries = {row.invocation: row for row in trace.entries}
    exits = defaultdict(list)
    for row in trace.exits:
        exits[row.invocation].append(row)
    bindings = {row.scope: row.anchor for row in store.bindings}
    scopes = {row.id: row for row in store.rows['scopes']}
    activations = {row.callee: row for row in scopes.values() if isinstance(row, ObservedPythonResourceScope)}

    def is_source_bound_normal(scope):
        ends = exits[scope.callee]
        return (scope.id in bindings and len(ends) == 1 and ends[0].anchor == scope.frame_exit
                and ends[0].completion is FrameCompletion.NORMAL)

    process = store.coordinates.target_process.pid
    codes = {row.code.key for row in trace.frame_scan.captures if row.process_id == process
             and row.code.filename == source['path'] and row.code.key == source['code_key']}
    regions = sorted((row for row in trace.entries if row.invocation.process_id == process and row.code in codes),
                     key=lambda row: row.anchor.index)
    if not regions or any(row.invocation not in activations or not is_source_bound_normal(activations[row.invocation])
                          for row in regions):
        raise ValueError('recipe source needs matching ordinary activations and original caller ports in this capture')
    first, last = (activations[row.invocation] for row in (regions[0], regions[-1]))
    common_parents = set(_parents(entries, first.callee))
    for row in regions[1:]:
        common_parents.intersection_update(_parents(entries, row.invocation))
        if not common_parents:
            break
    writes = defaultdict(list)
    for row in store.rows['work']:
        if row.syscall in _WRITE and (row.result or 0) > 0:
            writes[row.scope].append(row)

    relations, open_scopes = [], []
    for identity, work in writes.items():
        scope = scopes[identity]
        if not isinstance(scope, ObservedPythonResourceScope) or not is_source_bound_normal(scope):
            open_scopes.append({'scope': identity, 'reason': 'SOURCE_BOUND_NORMAL_PYTHON_EXIT_OPEN'})
            continue
        owner = next((parent for parent in _parents(entries, scope.callee) if parent in common_parents), None)
        if owner is None or scope.frame_entry.path != first.frame_entry.path:
            open_scopes.append({'scope': identity, 'reason': 'COMMON_OBSERVED_CALLER_OPEN'})
            continue
        if scope.frame_exit.index < first.frame_entry.index:
            order, endpoint = 'WRITE_SCOPE_EXIT_BEFORE_FIRST_REGION_ENTRY', first.frame_entry
        elif last.frame_exit.index < scope.frame_entry.index:
            order, endpoint = 'LAST_REGION_EXIT_BEFORE_WRITE_SCOPE_ENTRY', last.frame_exit
        else:
            open_scopes.append({'scope': identity, 'reason': 'WITHIN_OR_BETWEEN_REGION_OCCURRENCES'})
            continue
        relations.append(dict(order=order, common_observed_caller=entries[owner].anchor,
            original_source_port=bindings[identity].operation, call_attempt=scope.call_attempt,
            frame_entry=scope.frame_entry, frame_exit=scope.frame_exit, region_endpoint=endpoint,
            writes=tuple(work), service_duration_ns=None))
    store.check()
    return encode(dict(related_writes=relations, open_write_scopes=open_scopes,
        region_source=dict(path=source['path'], code_key=source['code_key'], sha256=source['sha256']),
        region_occurrences=len(regions), unassigned_resource_occurrences=len(store.rows['unassigned']),
        order_authority='OBSERVED_NORMAL_OCCURRENCES_ONLY',
        complete_effects_proven=False, deployment_authorized=False))


__all__ = [
    'CONTINUATION_CONDITIONS',
    'normal_continuation',
    'read_observed_scope',
    'select_observed_bodies',
    'project_observed_effect_order',
]
