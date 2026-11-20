"""Bind original shared loop call templates to selected real call activations."""
from collections import defaultdict

from scar.ir.program.call_edges import FrameCompletion, InvocationKey
from scar.ir.record_codec import decode, encode


def observed_loop_calls(index, parent, children, selected, checkpoints):
    """Replay the two observed parent windows through the original F4 ports."""
    codes = index.codes
    entries = {row.invocation: row for row in codes.trace.entries}
    exits = defaultdict(list)
    for row in codes.trace.exits:
        exits[row.invocation].append(row)
    source_codes = defaultdict(set)
    for row in codes.body_links():
        for capture in row.captures:
            source_codes[row.body].add((capture.process_id, capture.code.key))
    targets = {(row.caller, row.attempt_index): row for row in codes.targets.entries}
    callsites = defaultdict(list)
    for row in index.bindings:
        callsites[row.attempt.invocation, row.result].append(row)

    def activation(caller, link, expected=None):
        matches = []
        for site in callsites.get((caller, link.result), ()):
            call = targets.get((caller, site.attempt.index))
            if call is None or (expected is not None and call.callee != expected):
                continue
            frame = entries[call.callee]
            if (frame.invocation.process_id, frame.code) in source_codes[link.body]:
                matches.append((site, frame))
        if len(matches) != 1:
            return None
        callsite, start = matches[0]
        ends = exits[start.invocation]
        if (len(ends) != 1 or ends[0].completion is not FrameCompletion.NORMAL
                or ends[0].anchor.index <= start.anchor.index):
            return None
        return dict(invocation=start.invocation, source_result=link.result,
                    attempt=callsite.attempt.anchor, entry=start.anchor, exit=ends[0].anchor)

    windows = []
    for checkpoint in checkpoints:
        if checkpoint['invocation'] is None:
            break
        identity = decode(InvocationKey, checkpoint['invocation'])
        region = entries[identity]
        owner = entries.get(region.caller_invocation)
        if owner is None:
            break
        call = activation(owner.caller_invocation, parent, owner.invocation)
        members = [activation(owner.invocation, link, identity if offset == selected else None)
                   for offset, link in enumerate(children)]
        if call is None or any(row is None for row in members):
            break
        if (not call['entry'].index < members[0]['entry'].index
                or any(a['exit'].index >= b['entry'].index for a, b in zip(members, members[1:]))
                or members[-1]['exit'].index >= call['exit'].index):
            break
        windows.append(dict(parent=call, children=members))
    index.check()
    return encode(dict(windows=windows, all_selected_windows_bound=len(windows)==2,
                       unseen_control_routes_proven=False, all_intervening_effects_closed=False))
