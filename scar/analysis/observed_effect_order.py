"""Retain the observed order of enclosing source calls and external writes."""
from collections import defaultdict

from scar.analysis.source_resource_scopes import ResourceScopeStore
from scar.ir.program.call_edges import FrameCompletion
from scar.ir.program.resource_scopes import ObservedPythonResourceScope
from scar.ir.record_codec import encode
from scar.trace.os_resources import _WRITE


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
