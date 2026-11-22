"""Source-body effect closure for the native GCC SSA profile."""

from collections import Counter, defaultdict, deque
import re

from scar.ir.program.model import AccessMode, TerminatorKind, UseRole
from scar.ir.program.ids import OperationID

from ..contracts import native_bound_targets


_SCALAR_ASSIGN = re.compile(
    r'gimple_assign <(integer_cst|bit_and_expr|bit_ior_expr|bit_xor_expr|bit_not_expr),')
_INTEGER_PREDICATE = re.compile(
    r'gimple_cond <(?:eq_expr|ne_expr|lt_expr|le_expr|gt_expr|ge_expr), '
    r'[\w.$]*_\d+(?:\(D\))?, [+-]?\d+, NULL, NULL>$')
_SSA_RETURN = re.compile(r'gimple_return <(?:NULL|[+-]?\d+|[\w.$]*_\d+(?:\(D\))?)>$')
_NORMAL_TERMINATORS = {TerminatorKind.JUMP, TerminatorKind.BRANCH, TerminatorKind.RETURN}


def is_ssa_copy(operation, source):
    """The GCC value-forwarding statement used by the copy revision."""
    return (operation.label == 'gimple_assign' and len(operation.results) == 1
            and len(operation.uses) == 1 and operation.uses[0].role is UseRole.DATA
            and not operation.accesses and 'gimple_assign <ssa_name,' in source)


def _has_cycle(edges):
    incoming = dict.fromkeys(edges, 0)
    for successors in edges.values():
        for successor in successors:
            incoming[successor] += 1
    ready = deque(node for node, count in incoming.items() if count == 0)
    remaining = len(edges)
    while ready:
        remaining -= 1
        for successor in edges[ready.popleft()]:
            incoming[successor] -= 1
            if incoming[successor] == 0:
                ready.append(successor)
    return remaining != 0


def _local_rule(operation, source, function_addresses):
    if is_ssa_copy(operation, source):
        return 'SSA_VALUE_FORWARD'
    if operation.terminator is not None:
        if operation.terminator.kind in _NORMAL_TERMINATORS:
            return 'GRAPH_CONTROL'
        return None
    if (operation.label == 'gimple_return' and _SSA_RETURN.search(source)
            and all(access.mode is AccessMode.READ for access in operation.accesses)):
        # The operand has already been evaluated in SSA. GCC attaches vuse
        # for the outgoing memory state even to a scalar return.
        return 'NORMAL_RETURN_STATE_TRANSFER'
    if not operation.accesses:
        if operation.label == 'gimple_label':
            return 'GRAPH_CONTROL'
        if operation.label == 'gimple_cond' and _INTEGER_PREDICATE.search(source):
            return 'INTEGER_PREDICATE'
    if operation.label != 'gimple_assign' or operation.accesses or len(operation.results) != 1:
        return None
    if _SCALAR_ASSIGN.search(source):
        return 'INTEGER_CONSTANT_OR_BITWISE'
    if operation.id.wire in function_addresses and 'gimple_assign <addr_expr,' in source:
        return 'BOUND_FUNCTION_ADDRESS'
    return None


def native_entry_effects(graph, evidence, calls):
    """Summarize all reachable source regions, retaining unproved effect sites.

    A closed body inventory covers modeled source operations and normal
    control only. Process startup, shutdown and compiler/runtime behavior
    outside the retained GCC SSA boundary are separate obligations.
    """
    root = graph.root.wire
    if root != evidence['entry'] or (calls.get('entry') or {}).get('region') != root:
        raise ValueError('effect proof requires the same original program and call entry')
    regions = {region.id.wire: region for region in graph.regions}
    operations = {operation.id: operation for operation in graph.operations}
    anchors = {anchor['operation']: anchor for anchor in evidence['anchors']}
    omitted = {row['region'] for row in evidence['excluded_control']}
    sites, function_addresses = {}, set()
    for row in calls['calls']:
        if (operation := row['operation']) is None:
            continue
        anchor = anchors.get(operation)
        caller = row['caller']
        if (operation in sites or anchor is None or caller['region'] not in regions
                or (anchor['unit'], anchor['symbol'], anchor['region'], anchor['call'])
                != (caller['unit'], caller['symbol'], caller['region'], row['spelling'])):
            raise ValueError('effect proof call does not match its original operation anchor')
        sites[operation] = row
        region_value = regions[caller['region']].id.value
        for definition in row.get('target_flow', {}).get('definitions', {}).values():
            if definition['kind'] == 'ADDRESS':
                function_addresses.add(
                    OperationID(f'{region_value}:{definition["block"]}:{definition["ordinal"]}').wire)

    pending, summaries, dependencies = [root], {}, {}
    blocked, reverse_edges = set(), defaultdict(set)
    while pending:
        region_id = pending.pop()
        if region_id in summaries:
            continue
        region = regions[region_id]
        blocks = {block.id: block for block in region.blocks}
        reachable_blocks, callees = {}, set()
        reasons, counts = {}, Counter()
        reads, writes = set(), set()
        queue = [region.entry]

        def defer(reason, operation=None):
            if reason not in reasons:
                reasons[reason] = dict(count=0, first_operation=operation)
            reasons[reason]['count'] += 1

        if region_id in omitted:
            defer('EXCLUDED_EXCEPTIONAL_CONTROL')
        while queue:
            block_id = queue.pop()
            if block_id in reachable_blocks:
                continue
            block = blocks[block_id]
            terminator = operations[block.operations[-1]].terminator
            successors = tuple(edge.block for edge in terminator.successors)
            reachable_blocks[block_id] = successors
            queue.extend(reversed(successors))
            for identity in block.operations:
                op = operations[identity]
                for access in op.accesses:
                    if access.mode in (AccessMode.READ, AccessMode.READ_WRITE):
                        reads.add(access.location.wire)
                    if access.mode in (AccessMode.WRITE, AccessMode.READ_WRITE):
                        writes.add(access.location.wire)
                if op.label == 'gimple_call':
                    row = sites.get(identity.wire)
                    if row is None:
                        defer('CALL_WITHOUT_EFFECT_BINDING', identity.wire)
                        continue
                    bound = native_bound_targets(row)
                    expected = len(row['targets']) if row['status'] == 'FINITE_TARGETS' else 1
                    targets = {item['callee']['region'] for item in bound}
                    known = targets & regions.keys()
                    callees.update(known)
                    if not bound or len(bound) != expected or known != targets:
                        defer('CALL_TARGET_EFFECT_OPEN', identity.wire)
                    else:
                        counts['BOUND_CALL'] += 1
                    continue
                source = anchors.get(identity.wire, {}).get('source') or ''
                if rule := _local_rule(op, source, function_addresses):
                    counts[rule] += 1
                else:
                    reason = ('MEMORY_FOOTPRINT_OPEN' if op.accesses else
                              'NONNORMAL_EXIT_OPEN' if op.terminator is not None else
                              'SSA_OPERATION_EFFECT_OPEN')
                    defer(reason, identity.wire)
        cycle = _has_cycle(reachable_blocks)
        if reasons:
            blocked.add(region_id)
        summaries[region_id] = dict(
            blocks=len(reachable_blocks), admitted=dict(sorted(counts.items())),
            virtual_reads=sorted(reads), virtual_writes=sorted(writes),
            local_frontiers=dict(sorted(reasons.items())), cfg_cycle=cycle)
        dependencies[region_id] = callees
        for target in sorted(callees, reverse=True):
            reverse_edges[target].add(region_id)
            pending.append(target)

    pending = list(blocked)
    while pending:
        for caller in reverse_edges[pending.pop()] - blocked:
            blocked.add(caller)
            pending.append(caller)
    for region_id, summary in summaries.items():
        summary['reachable_effects_closed'] = region_id not in blocked
    call_cycle = _has_cycle(dependencies)
    normal_return = root not in blocked and not call_cycle and not any(
        summary['cfg_cycle'] for summary in summaries.values())
    return dict(schema='scar.native-entry-effects', schema_version=1,
                scope='REACHABLE_GCC_SSA_SOURCE_BODY', entry=root,
                status='CLOSED_BODY' if root not in blocked else 'OPEN_BODY',
                normal_return_in_retained_graph=normal_return, call_cycle=call_cycle,
                process_startup_and_shutdown_covered=False,
                regions=dict(sorted(summaries.items())))
