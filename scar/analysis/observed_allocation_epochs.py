"""Replay a bounded NumPy allocation-epoch witness on original frame entries."""

from collections import defaultdict
from pathlib import Path

from scar.analysis.call_operand_reader import CallOperandTrace
from scar.analysis.stateful_storage_accesses import observed_storage_bindings
from scar.compiler.stateful_model import validate_stateful_region
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program.call_edges import FrameCompletion, InvocationKey
from scar.ir.record_codec import decode, encode
from scar.trace.numpy_allocation_epochs import read_tracker


SCHEMA = 'scar.observed-allocation-epochs'
_METADATA = 'scar.region-input-metadata'
_TICKET_FIELDS = {'epoch', 'cursor'}
_INVOCATION_FIELDS = {'trace_directory', 'code_key', 'process_id', 'thread_id', 'invocation_id'}


def _need(condition, detail):
    if not condition:
        raise ValueError(detail)


def _ticket(row):
    ticket = row.get('allocator_epoch')
    if ticket is not None:
        _need(type(ticket) is dict and set(ticket) == _TICKET_FIELDS
              and all(type(ticket[key]) is int for key in _TICKET_FIELDS)
              and 0 < ticket['epoch'] <= ticket['cursor'] < 2**64,
              'invalid allocator-epoch checkpoint')
    return ticket


def _endpoint(row, ordinal, source, trace, entries, descriptors):
    coordinate = row.get('boundary_observation')
    _need(type(coordinate) is dict and set(coordinate) == {'run', 'process_id', 'ordinal'}
          and type(coordinate['run']) is str and bool(coordinate['run'])
          and type(coordinate['process_id']) is int and coordinate['process_id'] > 0
          and type(coordinate['ordinal']) is int and coordinate['ordinal'] == ordinal + 1,
          'original boundary observation has invalid coordinates')
    binding = row.get('trace_invocation')
    if binding is None:
        return dict(observation=coordinate, record='#/records/' + str(ordinal),
                    invocation=None, frame_entry=None)
    _need(type(binding) is dict and set(binding) == _INVOCATION_FIELDS
          and binding['trace_directory'] == trace.directory and binding['code_key'] == source.code_key,
          'allocation checkpoint names another original trace or code')
    identity = InvocationKey(binding['process_id'], binding['thread_id'], binding['invocation_id'])
    _need(identity.process_id == coordinate['process_id'],
          'allocation checkpoint process differs from its boundary observation')
    frame = entries.get(identity)
    code = descriptors.get((identity.process_id, source.code_key))
    _need(frame is not None and frame.code == source.code_key and not frame.suspended
          and frame.anchor.index is not None and frame.parameters_complete and code is not None
          and code.filename == source.path and code.qualname == source.qualname
          and code.parameters == tuple(parameter.name for parameter in frame.parameters),
          'allocation checkpoint lacks its exact original ordinary frame entry')
    return dict(observation=coordinate, record='#/records/' + str(ordinal),
                invocation=encode(identity), frame_entry=encode(frame.anchor))


def _same_interval(first, second):
    if first['invocation'] is None or second['invocation'] is None:
        return False
    a, b = first['invocation'], second['invocation']
    if (first['observation']['run'] != second['observation']['run']
            or (a['process_id'], a['thread_id']) != (b['process_id'], b['thread_id'])):
        return False
    left, right = first['frame_entry'], second['frame_entry']
    _need(a != b and left['path'] == right['path'] and left['index'] < right['index'],
          'allocation endpoints must be distinct, ordered original frame entries')
    return True


def _parents(entry, entries, exits):
    while entry.caller_invocation is not None:
        identity = entry.caller_invocation
        parent = entries.get(identity)
        completions = exits.get(identity, ())
        if parent is None or parent.suspended or entry.caller is None or len(completions) != 1:
            return
        end, = completions
        if (entry.caller.code != parent.code or end.code != parent.code
                or end.completion is FrameCompletion.SUSPENDED
                or not parent.anchor.path == entry.anchor.path == end.anchor.path
                or parent.anchor.index is None or end.anchor.index is None
                or not parent.anchor.index < entry.anchor.index < end.anchor.index):
            return
        yield parent, end
        entry = parent


def _common_ancestor(trace, entries, endpoints):
    """Require one ordinary activation to bracket both exact caller chains."""
    left, right = (entries[decode(InvocationKey, row['invocation'])] for row in endpoints)
    exits = defaultdict(list)
    for event in trace.exits:
        exits[event.invocation].append(event)
    left_parents = {parent.invocation for parent, _ in _parents(left, entries, exits)}
    for parent, end in _parents(right, entries, exits):
        if parent.invocation in left_parents and right.anchor.index < end.anchor.index:
            return dict(invocation=encode(parent.invocation), frame_entry=encode(parent.anchor),
                        frame_exit=encode(end.anchor))
    return None


def observe_allocation_epochs(region, metadata, trace, *, observation=0, expected_sha256=None):
    """Compare adjacent selected frame checkpoints of one admitted input profile.

    The same live epoch at both checkpoints proves survival of one allocation
    issued by the observed handler over that measured interval. Field binding,
    array contents, and loop-control coverage are separate obligations.
    """
    _need(type(trace) is CallOperandTrace, 'exact original CallOperandTrace required')
    validate_stateful_region(region)
    source = region.source
    reference = artifact_reference(Path(metadata).resolve(strict=True), _METADATA)
    _need(expected_sha256 is None or reference['sha256'] == expected_sha256,
          'allocation metadata differs from its original receipt')
    document = read_artifact(reference)
    # Validate the original storage geometry once for each selected boundary.
    # The observer issues the root ticket only for the already-qualified owner.
    _, evidence = observed_storage_bindings(region, reference['uri'], observation=observation)
    _need(evidence['artifact'] == reference, 'allocation metadata changed during storage admission')
    matched = [(ordinal, row) for ordinal, row in enumerate(document['records'])
               if (row['source']['path'], row['source']['sha256'], row['source']['code_key'])
               == (source.path, source.sha256, source.code_key)]

    tracker = document.get('allocation_epoch_tracker')
    if tracker is not None:
        collector = document.get('collector')
        _need(type(collector) is dict and collector.get('local_event') == 'ACCEPTED_FRAME_ENTRY'
              and document.get('errors') == [],
              'allocator checkpoints require the original shared frame-entry collector')
        read_artifact(tracker)
        admitted, _ = read_tracker(tracker['uri'])
        _need(admitted == tracker, 'allocation tracker differs from its original metadata reference')
    entries = {frame.invocation: frame for frame in trace.entries}
    descriptors = {(row.process_id, row.code.key): row.code for row in trace.frame_scan.captures}
    selected = matched[observation:observation + 2]
    endpoints = [_endpoint(row, ordinal, source, trace, entries, descriptors) for ordinal, row in selected]
    same_interval = len(endpoints) == 2 and _same_interval(*endpoints)
    ancestor = _common_ancestor(trace, entries, endpoints) if same_interval else None
    roots = {item.id: item for item in region.inputs if item.kind == 'array'}

    comparisons = evidence.get('next_observed_boundary', {}).get('arrays', {})
    if len(selected) == 2:
        _, next_evidence = observed_storage_bindings(region, reference['uri'], observation=observation + 1)
        _need(next_evidence['artifact'] == reference, 'allocation metadata changed during storage admission')

    tickets = []
    for _, row in selected:
        batch = {name: _ticket(row['inputs'][name]) for name in roots}
        present = [ticket for ticket in batch.values() if ticket is not None]
        _need(not present or tracker is not None, 'allocator tickets lack the original tracker reference')
        _need(len({ticket['epoch'] for ticket in present}) == len(present),
              'distinct admitted storage owners share one allocator epoch')
        tickets.append(batch)
    if same_interval:
        cursors = [ticket['cursor'] for batch in tickets for ticket in batch.values() if ticket is not None]
        _need(all(a <= b for a, b in zip(cursors, cursors[1:])),
              'allocation checkpoints are not in monotonic tracker order')

    arrays = {}
    for name, spec in roots.items():
        active = [batch[name] for batch in tickets]
        match = comparisons.get(name)
        status = 'OPEN'
        if tracker is None:
            detail = 'NO_TRACKED_ALLOCATOR'
        elif not same_interval:
            detail = 'NO_TWO_ORDERED_ENTRIES_IN_THE_SAME_TRACE_ACTOR'
        elif ancestor is None:
            detail = 'COMMON_ACTIVE_ORDINARY_ANCESTOR_UNOBSERVED'
        elif None in active:
            detail = 'ALLOCATOR_EPOCH_UNOBSERVED_AT_BOUNDARY'
        elif active[0]['epoch'] != active[1]['epoch']:
            status = 'OBSERVED_DIFFERENT_ALLOCATION_EPOCHS'
            detail = 'DISTINCT_PROVIDER_ALLOCATION_GENERATIONS'
        elif match != {'same_owner': True, 'same_geometry': True}:
            detail = 'ORIGINAL_OWNER_OR_REGION_GEOMETRY_NOT_MATCHED'
        else:
            status = 'OBSERVED_CONTINUOUS_ALLOCATION'
            detail = 'SAME_LIVE_EPOCH_AT_ORDERED_CHECKPOINTS_OF_ITS_OWNER_HANDLER'
        arrays[name] = dict(input_path=list(spec.path), status=status, relation=detail,
                            checkpoints=active, endpoint_storage_comparison=match)
    return dict(schema=SCHEMA, schema_version=1, metadata=reference, allocation_epoch_tracker=tracker,
                source_code_key=source.code_key, observation=observation, endpoints=endpoints, arrays=arrays,
                original_entry_interval_identified=same_interval, shared_active_ancestor=ancestor,
                covered_observed_interval=bool(arrays) and all(
                    item['status'] == 'OBSERVED_CONTINUOUS_ALLOCATION' for item in arrays.values()),
                slot_binding_continuity_verified=False,
                source_loop_backedge_verified=False, content_continuity_verified=False,
                transformation_authorized=False)
