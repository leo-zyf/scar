"""Await outcomes."""

from scar.ir.program.lifecycle import DriveOutcomeRef
from scar.ir.program.control import ProtocolValueKind
from scar.ir.program.model import (
    ExitKind,
    OperationResult,
    ProtocolValue,
    TerminatorKind,
    CompletionResult,
    InvokeLiteralResult,
    StateVersionValue,
    ValueKind,
)
from scar.ir.program.protocol_connections import ProtocolOperationRef
from scar.ir.program.protocols import (
    ContractResult,
    DriveResult,
    HostTerminal,
    OutcomeAdapterRule,
    OwnedUse,
    StaticProtocolGap,
    OutcomeAdapter,
    ProtocolClause,
    ProtocolProfile,
)
from scar.ir.program.structured import ControlEdgeReference
from scar.ir.program.suspension_interfaces import packets as _packets, structured_region
from scar.ir.program.data_materials import flow, required_choices, selected, inventory_flow
from scar.ir.program.operation_results import result_legs
from scar.ir.program.protocol_await_records import AwaitAcquisition, ExpandedAwaitCallee


# Bounded original-port inventories shared by await outcome/failure views.
#
# May-copy enumerates possible consumers. Only a later must binding establishes
# an exact connection. No semantic projection is inserted into ordinary copies.
def flags():
    return {'source_runtime_verified': False, 'runtime_identity_verified': False,
            'value_identity_verified': False, 'transformation_authorized': False}


def answer(match=None, *, gaps=(), **extra):
    return {'match': match, 'routes': (), 'gaps': tuple(gaps), **extra}


def combine(reports):
    checks = tuple(row['match'] for row in reports)
    return answer(False if False in checks else None if not checks or None in checks else True,
                  gaps=tuple(gap for row in reports for gap in row.get('gaps', ())),
                  bindings=tuple(reports))


def prepare(ctx):
    if hasattr(ctx, '_await_outcome_raw'):
        return ctx._await_outcome_raw
    by_input, by_operation = {}, {}
    for row in ctx.graph.protocol_operations:
        if type(row.contract) is OutcomeAdapter:
            by_input.setdefault(row.contract.input, []).append(row)
            by_operation.setdefault(row.operation, []).append(row)
    successor_total = sum(len(ctx.operations[row.operation].terminator.successors)
                          for row in ctx.graph.body_drives)
    ctx._await_outcome_raw = by_input, by_operation, successor_total
    ctx.preparation['await_outcome_raw_adapter_rows'] = sum(map(len, by_input.values()))
    ctx.preparation['await_original_drive_successors'] = successor_total
    ctx.preparation['await_prepared_value_use_index_reused'] = True
    return ctx._await_outcome_raw


def terminal_inventory(ctx, source, choices=()):
    """Original endpoints after this exact edge, not a semantic adapter search."""
    port = ctx.port(source)
    if port is None or port.operation is None or port.gap is not None:
        return (), ('await_terminal_source_unresolved',)
    operation = ctx.operations[port.operation]
    start = (operation.terminator.successors[port.edge].block if port.phase == 'edge'
             else ctx.block_of[port.operation])
    _, outgoing, _, _, _ = ctx._edges(ctx.region_of[port.operation])
    guards = ctx.selected_edges(choices, region=ctx.region_of[port.operation])
    own_choices = {edge.operation: edge.successor_index for edge in guards}
    incoming = {ctx.operations[edge.operation].terminator.successors[edge.successor_index].block:
                (edge.operation, edge.successor_index) for edge in guards}
    pending, seen, found, gaps = [start], set(), [], []
    while pending:
        if not ctx.spend('await_outcome_terminal_blocks'):
            gaps.append('await_outcome_terminal_budget')
            break
        block_id = pending.pop()
        if block_id in seen:
            continue
        seen.add(block_id)
        block = ctx.blocks[block_id]
        last = ctx.operations[block.operations[-1]]
        if last.id in ctx.drives:
            gaps.append('await_outcome_next_drive_boundary')
            continue
        kind = last.terminator.kind
        if kind in (TerminatorKind.RETURN, TerminatorKind.RAISE):
            found.append(HostTerminal(last.id))
            continue
        if kind not in (TerminatorKind.INVOKE, TerminatorKind.JUMP, TerminatorKind.BRANCH):
            gaps.append('await_outcome_structured_continuation_unresolved')
            continue
        for identity, number, target, _ in reversed(outgoing[block_id]):
            if not ctx.spend('await_outcome_terminal_edges'):
                gaps.append('await_outcome_terminal_budget')
                break
            if identity in own_choices and own_choices[identity] != number:
                continue
            if target in incoming and incoming[target] != (identity, number):
                continue
            pending.append(target)
    if not found and not gaps:
        gaps.append('await_outcome_terminal_absent')
    return tuple(found), tuple(dict.fromkeys(gaps))


def terminal_payload(ctx, terminal):
    return tuple(OwnedUse(terminal.operation, identity)
        for identity in ctx.operations[terminal.operation].terminator.values
        if not isinstance(ctx.values[ctx.owned_use(terminal.operation, identity).value], StateVersionValue))


def packet_use(ctx, terminal):
    payloads = terminal_payload(ctx, terminal)
    return next((use for use in payloads if isinstance(
        value := ctx.values[ctx.owned_use(use.operation, use.use).value], ProtocolValue)
        and value.protocol is ProtocolValueKind.EXCEPTION_PACKET), None)


def bind_target(ctx, source, target, choices=()):
    if type(target) is HostTerminal:
        reports = tuple(case_binding(ctx, source, use, choices)
                        for use in terminal_payload(ctx, target))
        return {**combine(reports), 'target': target}
    return {**case_binding(ctx, source, target, choices), 'target': target}


def case_binding(ctx, source, actual, choices):
    """A possible multi-visit source is not a false payload-copy claim.

    Preserve the ordinary binding as evidence of its flat static context.
    Only a candidate in the conditional source-evaluated inventory may turn
    that context into an explicit unmet obligation, never a copy True.
    """
    binding = ctx.bind(source, actual, guard=choices or None)
    membership = inventory_flow(ctx, source, choices)
    before = membership['before']
    use = ctx.owned_use(actual.operation, actual.use)
    if (membership['conditional'] and (membership.get('pending', False)
            or before is not None and use.value in before.get(actual.operation, ()))):
        return {**answer(gaps=(*binding.get('gaps', ()), *membership['gaps'])),
                'guarded_binding': binding}
    return binding


def consumers(ctx, source, choices=()):
    """Indexed position-sensitive may inventory, with explicit call boundaries."""
    from scar.ir.program.protocol_connection_validation import _consumer_role
    choices = ctx.selected_edges(choices)
    if not hasattr(ctx, '_await_outcome_consumers'):
        ctx._await_outcome_consumers = {}
    key = source, choices, ctx._returns.guards
    if key in ctx._await_outcome_consumers:
        return ctx._await_outcome_consumers[key]
    membership = inventory_flow(ctx, source, choices)
    before = membership['before']
    targets, support, gaps, seen = [], [], [], set()
    gaps.extend(membership['gaps'])
    if key in getattr(ctx, '_await_flow_choice_conflicts', ()):
        gaps.append('await_source_choice_conflict')
    if before is None:
        gaps.append('await_consumer_inventory_unresolved')
    else:
        for operation, values in before.items():
            if not ctx.spend('await_consumer_positions'):
                break
            if not ctx.spend('await_consumer_values', len(values)):
                break
            for value in sorted(values, key=lambda item: item.wire):
                for site in ctx.index._consumers[value]:
                    if not ctx.spend('await_consumer_indexed_uses'):
                        break
                    if site.operation != operation or site.use.id in seen:
                        continue
                    seen.add(site.use.id)
                    consumer = ctx.operations[site.operation]
                    if isinstance(ctx.values[site.use.value], StateVersionValue) or not _consumer_role(consumer, site.use.id):
                        continue
                    actual = OwnedUse(site.operation, site.use.id)
                    target = (HostTerminal(site.operation) if consumer.terminator is not None
                        and consumer.terminator.kind in (TerminatorKind.RETURN, TerminatorKind.RAISE) else actual)
                    if target not in targets:
                        targets.append(target)
                        binding = bind_target(ctx, source, target, choices)
                        if membership['conditional']:
                            binding = {**answer(False if binding['match'] is False else None,
                                gaps=(*binding.get('gaps', ()), *membership['gaps'])),
                                'guarded_binding': binding}
                        support.append({'actual': target, 'binding': binding})
                    if consumer.callee is not None:
                        # Keep the actual call Use; a return-copy summary does
                        # not inventory that callee's internal consumers.
                        gaps.append('await_consumer_callee_expansion_pending')
    if ctx.budget.truncated:
        gaps.append('await_consumer_inventory_budget')
    from scar.ir.program.data_materials import consumer_graph
    graph = consumer_graph(ctx, source, choices)
    if graph['inventory']['known_inventory_complete']:
        gaps = [gap for gap in gaps if gap != 'await_consumer_callee_expansion_pending']
    result = {'source': source, 'targets': tuple(targets), 'support': tuple(support),
        'consumer_graph': graph, 'inventory_membership': membership,
        'gaps': tuple(dict.fromkeys(gaps)), 'known_inventory_complete': before is not None
        and not membership['conditional'] and not ctx.budget.truncated
        and 'await_consumer_callee_expansion_pending' not in gaps}
    ctx._await_outcome_consumers[key] = result
    return result


def state_links(ctx, terminal, edge, choices=(), *, outward=False):
    """Exact signature ordinals and locations; never landing-formal positions."""
    from .protocol_await import _result_copy
    operation = ctx.operations[edge.operation]
    completion = operation.terminator.successors[edge.successor_index]
    signature = next(row for row in operation.completions if row.kind is completion.kind)
    actuals = tuple(OwnedUse(terminal.operation, identity)
                    for identity in ctx.operations[terminal.operation].terminator.values)
    rows = []
    for slot, value_type in enumerate(signature.types):
        if value_type.kind is not ValueKind.STATE:
            continue
        if not ctx.spend('await_outcome_state_slots'):
            break
        matches = tuple(use for use in actuals if isinstance(
            value := ctx.values[ctx.owned_use(use.operation, use.use).value], StateVersionValue)
            and value.location == value_type.location)
        post = InvokeLiteralResult(edge, CompletionResult(slot))
        actual = matches[0] if len(matches) == 1 else None
        report = (answer(gaps=('await_terminal_state_unresolved',)) if actual is None else
                  case_binding(ctx, post, actual, choices) if outward else
                  _result_copy(ctx, actual, post, choices))
        rows.append({'slot': slot, 'source': post if outward else actual,
                     'target': actual if outward else post, 'binding': report, 'forwarded': False})
    landing = ctx.blocks[completion.block]
    for position, (argument, formal) in enumerate(zip(completion.arguments, landing.arguments)):
        if type(argument) is CompletionResult:
            continue
        original = ctx.owned_use(operation.id, argument)
        value = ctx.values[original.value]
        if not isinstance(value, StateVersionValue):
            continue
        if not ctx.spend('await_outcome_forwarded_states'):
            break
        matches = tuple(use for use in actuals if isinstance(
            current := ctx.values[ctx.owned_use(use.operation, use.use).value], StateVersionValue)
            and current.location == value.location)
        target = matches[0] if len(matches) == 1 else None
        source = OwnedUse(operation.id, argument)
        binding = (answer(gaps=('await_forwarded_terminal_state_unresolved',)) if target is None else
                   case_binding(ctx, source, target, choices))
        rows.append({'slot': None, 'position': position, 'source': source, 'target': target,
                     'landing_formal': formal, 'binding': binding, 'forwarded': True})
    return tuple(rows)


def inventory(rows, checked, complete, *, cursor=None, **extra):
    total = len(rows) if complete else None
    return {'enumerated_count': len(rows), 'checked_count': checked,
            'pending_count': None if total is None else total - checked,
            'jobs_total': total, 'known_inventory_complete': complete,
            'cursor': cursor, **extra}


# Original body outcomes connected to physical packets and caller consumers.
#
# The original inventory precedes adapters. Declared semantic conversions stay
# conditional, and never become copy/alias edges in the canonical graph.
COMPLETION_RULES = (OutcomeAdapterRule.GENERATOR_RETURN_TO_STOP_ITERATION,
                    OutcomeAdapterRule.COROUTINE_RETURN_TO_STOP_ITERATION)


def _ref(row):
    return ProtocolOperationRef(row.scope, row.clause, row.operation, row.slot)


def _own_legs(ctx, operation, rows, choices, issues, *, required):
    """Raw N/R inventory first, then optional own contract/consumer receipts."""
    legs, gaps = [], []
    for original in result_legs(ctx, operation):
        if not ctx.spend('await_final_completion_legs'):
            legs.append({**original, 'targets': (), 'bindings': (), 'states': (),
                         'status': 'NOT_CHECKED', 'gaps': ()})
            continue
        own = original['own_result']
        local = []
        if own is None:
            legs.append({**original, 'targets': (), 'bindings': (), 'states': (),
                         'status': 'PARTIAL', 'gaps': ('await_final_payload_slot_unresolved',)})
            gaps.append('await_final_payload_slot_unresolved')
            continue
        # The own result exists only after this particular call was reached.
        # Two mutually exclusive calls can join one terminal; the incoming
        # advance's conditions alone do not select the call that produced it.
        # Derive the prefix from the source, never from a successful consumer.
        required_prefix, prefix_complete = required_choices(ctx, own)
        if not prefix_complete:
            local.append('await_final_source_prefix_unresolved')
        conditions = selected(ctx, choices, required_prefix, original['edge'])
        ledger = consumers(ctx, own, conditions)
        targets = list(ledger['targets'])
        matching = tuple(row for row in rows if row.contract.output == own)
        if required and not matching:
            local.append('await_final_adapter_leg_missing')
        for row in matching:
            receipt = ctx.result_consumers.get(_ref(row))
            if receipt is None:
                local.append('protocol_result_consumers_missing')
                continue
            for target in receipt.targets:
                if not ctx.spend('await_final_receipt_targets'):
                    break
                if type(target) is StaticProtocolGap:
                    local.append('await_final_consumer_attachment_gap')
                elif target not in targets:
                    targets.append(target)
        bindings = tuple(bind_target(ctx, own, target, conditions) for target in targets)
        for binding in bindings:
            if binding['match'] is False:
                local.append('await_final_consumer_binding_contradicted')
                issues.append(('protocol_await_final_copy', str(operation.id),
                    'final consumer does not use this exact own completion payload'))
            elif binding['match'] is None:
                local.append('await_final_consumer_binding_unproved')
        if not targets:
            local.append('await_final_output_consumer_unresolved')
        states = []
        if original['edge'] is not None:
            for target in targets:
                if type(target) is HostTerminal:
                    states.extend(state_links(ctx, target, original['edge'], conditions, outward=True))
        for state in states:
            if state['binding']['match'] is False:
                local.append('await_final_state_contradicted')
                issues.append(('protocol_await_final_state', str(operation.id),
                               'final consumer uses another completion/state slot'))
            elif state['binding']['match'] is None:
                local.append('await_final_state_unproved')
        local.extend(ledger['gaps'])
        contradicted = any(binding['match'] is False for binding in bindings) or any(
                          state['binding']['match'] is False for state in states)
        legs.append({**original, 'choices': conditions,
            'targets': tuple(targets), 'bindings': bindings, 'states': tuple(states),
            'status': 'CONTRADICTED' if contradicted else 'PARTIAL' if local or ctx.budget.truncated else 'DECLARED',
            'gaps': tuple(dict.fromkeys(local))})
        gaps.extend(local)
    return tuple(legs), tuple(dict.fromkeys(gaps))


def _caller_consumers(ctx, advance, parent, choices, role, requires, issues):
    _, by_operation, _ = prepare(ctx)
    source = parent['own_result']
    if source is None:
        return (), ('await_advance_payload_unresolved',)
    ledger = consumers(ctx, source, choices)
    views, gaps = [], list(ledger['gaps'])
    for support in ledger['support']:
        if not ctx.spend('await_outcome_consumer_jobs'):
            break
        actual, binding = support['actual'], support['binding']
        local = []
        operation = ctx.operations[actual.operation]
        final_rows = tuple(row for row in by_operation.get(operation.id, ())
            if row.contract.rule is OutcomeAdapterRule.AWAIT_STOPITER_TO_EXPRESSION
            and type(row.contract.input) is ContractResult
            and row.contract.input.operation == advance.id
            and row.contract.input.result_port == source
            and row.contract.input_use == getattr(actual, 'use', None))
        if requires and not final_rows:
            local.append('await_final_adapter_missing')
        legs, leg_gaps = ((), ()) if type(actual) is HostTerminal else _own_legs(
            ctx, operation, final_rows, choices, issues, required=bool(requires))
        local.extend(leg_gaps)
        if binding['match'] is not True:
            local.append('await_consumer_must_binding_unproved')
        from .region_implementation import region_implementation
        implementation = region_implementation(ctx, source, operation, choices,
            graph=ledger['consumer_graph'], parent_legs=legs)
        views.append({'actual': actual, 'advance_result': source, 'choices': choices,
            'consumer_graph': ledger['consumer_graph'],
            'implementation': implementation,
            'binding': binding, 'final_rows': final_rows, 'legs': legs, 'role': role,
            'requires_final_extraction': requires,
            'status': 'CONTRADICTED' if any(leg['status'] == 'CONTRADICTED' for leg in legs) else
                      'PARTIAL' if local or ctx.budget.truncated else 'DECLARED',
            'gaps': tuple(dict.fromkeys(local)), **flags()})
        gaps.extend(local)
    return tuple(views), tuple(dict.fromkeys(gaps))


def outcome_connections(ctx, original_inventory, *, derive=True):
    from .protocol_await import _result_copy
    by_input, _, successor_total = prepare(ctx)
    seeds, seen = [], set()
    enumeration_complete = original_inventory['known_inventory_complete']
    cursor = original_inventory['cursor']
    for job in original_inventory['jobs']:
        if job.get('drive') is None or job.get('advance_operation') is None:
            continue
        drive = ctx.operations[job['drive']]
        for number, edge in enumerate(drive.terminator.successors):
            if derive and not ctx.spend('await_original_outcome_edges'):
                enumeration_complete = False
                cursor = cursor or {'phase': 'await_outcome_inventory', 'drive': drive.id, 'successor': number}
                # Retain this and the raw declared suffix keys. Exhaustion
                # prevents detail proofs, never erases original anchors.
            body = ctx.operations[edge.source]
            outcome = DriveOutcomeRef(drive.id, edge.source)
            key = (job['scope'], job['source_operation'], job['advance_operation'],
                   job.get('input_guard'), job['choices'], outcome)
            if key in seen:
                continue
            seen.add(key)
            seeds.append((key, job, number, body, outcome))
    if not derive:
        ctx.preparation['await_deferred_original_outcome_keys'] = len(seeds)
    rows, issues, gaps, checked = [], [], [], 0
    for key, job, number, body, outcome in seeds:
        payload = DriveResult(outcome, 0)
        kind = ('RETURN_COMPLETION' if body.terminator.kind is TerminatorKind.RETURN else
                'EXCEPTION' if body.terminator.kind is TerminatorKind.RAISE else
                'HANDOFF' if body.terminator.kind is TerminatorKind.SUSPEND else 'UNRESOLVED')
        row = {'key': key, 'scope': job['scope'], 'source': job['source'],
            'source_operation': job['source_operation'], 'advance_operation': job['advance_operation'],
            'input_guard': job.get('input_guard'), 'choices': job['choices'],
            'drive_outcome': outcome, 'payload': payload, 'kind': kind}
        if not derive or ctx.budget.truncated or not ctx.spend('await_outcome_jobs'):
            rows.append({**row, 'conversion_legs': (), 'advance_legs': (), 'consumers': (), 'status': 'NOT_CHECKED',
                         'gaps': ('await_static_details_deferred_without_capture',) if not derive else
                                 ('await_outcome_budget',), **flags()})
            if derive:
                cursor = cursor or {'phase': 'await_outcome_jobs', 'key': key}
            continue
        local, converted, advance_legs, caller_views = list(job['gaps']), [], [], []
        required, complete = required_choices(ctx, payload)
        choices = selected(ctx, job['choices'], required, ControlEdgeReference(outcome.drive, number))
        if not complete:
            local.append('await_outcome_source_prefix_unresolved')
        advance = ctx.operations[job['advance_operation']]
        parents = tuple(result_legs(ctx, advance))
        conversion_rows = tuple(candidate for candidate in by_input.get(payload, ())
                                if candidate.contract.rule in COMPLETION_RULES)
        producer_ids = tuple(dict.fromkeys(candidate.operation for candidate in conversion_rows))
        if kind == 'RETURN_COMPLETION' and not producer_ids:
            local.append('await_completion_adapter_missing')
        for producer in producer_ids:
            operation = ctx.operations[producer]
            candidates = tuple(candidate for candidate in conversion_rows if candidate.operation == producer)
            input_binding = ctx.bind(payload, OwnedUse(producer, candidates[0].contract.input_use), guard=choices)
            for leg in result_legs(ctx, operation):
                if not ctx.spend('await_conversion_legs'):
                    break
                leg_choices = selected(ctx, choices, leg['edge'])
                own = leg['own_result']
                role = 'COMPLETION' if leg['kind'] is ExitKind.NORMAL else 'CONSTRUCTION_FAILURE'
                own_rows = tuple(candidate for candidate in candidates if candidate.contract.output == own)
                packets, packet_gaps = (_packets(ctx, own, leg_choices) if own is not None
                    and leg['kind'] is ExitKind.NORMAL else
                    (({'source': own, 'definition': None, 'exception_binding': None},), ()) if own is not None
                    else ((), ('await_conversion_payload_unresolved',)))
                if own is not None and structured_region(ctx, own):
                    from scar.ir.program.await_inventory_originals import structured_completion
                    joined = structured_completion(ctx, own, packets, leg_choices,
                        parents, advance, input_binding, own_rows, role, _caller_consumers, issues)
                    leg_gaps = [*packet_gaps, *joined['gaps']]
                    if not own_rows:
                        leg_gaps.append('await_completion_adapter_leg_missing')
                    if input_binding['match'] is not True:
                        leg_gaps.append('await_conversion_input_unproved')
                    converted.append({'operation': producer, **leg, 'role': role,
                        'choices': leg_choices, 'rows': own_rows, 'input_binding': input_binding,
                        'packets': packets, 'continuations': joined['continuations'],
                        'terminals': joined['terminals'], 'states': joined['states'],
                        'status': 'CONTRADICTED' if input_binding['match'] is False or any(
                            terminal['status'] == 'CONTRADICTED' for terminal in joined['terminals']) else
                            'PARTIAL' if leg_gaps or ctx.budget.truncated else 'DECLARED',
                        'gaps': tuple(dict.fromkeys(leg_gaps))})
                    caller_views.extend(joined['consumers'])
                    local.extend(leg_gaps)
                    continue
                terminals, terminal_gaps = (((), ('await_conversion_payload_unresolved',)) if own is None
                                           else terminal_inventory(ctx, own, leg_choices))
                terminal_views, states, leg_gaps = [], [], list((*packet_gaps, *terminal_gaps))
                if not own_rows:
                    leg_gaps.append('await_completion_adapter_leg_missing')
                for terminal in terminals:
                    if not ctx.spend('await_conversion_terminals'):
                        break
                    target = packet_use(ctx, terminal)
                    packet_reports = tuple(ctx.bind(packet['source'], target, guard=leg_choices)
                        for packet in packets if packet['source'] is not None and target is not None)
                    copied = (answer(True, bindings=packet_reports) if any(
                        report['match'] is True for report in packet_reports) else
                        answer(False, bindings=packet_reports,
                            gaps=('await_conversion_packet_terminal_contradicted',))
                        if packet_reports and all(report['match'] is False for report in packet_reports) else
                        answer(gaps=('await_conversion_packet_terminal_unproved',)))
                    if copied['match'] is not True:
                        leg_gaps.extend(copied['gaps'])
                    if copied['match'] is False:
                        issues.append(('protocol_await_conversion_packet', str(producer),
                            'driver terminal is not a packet of this own conversion result'))
                    terminal_views.append({'terminal': terminal, 'packet': target, 'binding': copied})
                    if leg['edge'] is not None:
                        states.extend(state_links(ctx, terminal, leg['edge'], leg_choices, outward=True))
                    for parent in parents:
                        if parent['kind'] is not ExitKind.RAISE or parent['own_result'] is None:
                            continue
                        conditions = selected(ctx, leg_choices, parent['edge'])
                        delivery = (answer(gaps=('await_conversion_terminal_packet_missing',)) if target is None
                                    else _result_copy(ctx, target, parent['own_result'], conditions))
                        requires = (role == 'COMPLETION' and copied['match'] is True
                                    and delivery['match'] is True and input_binding['match'] is True
                                    and bool(own_rows))
                        if delivery['match'] is not True:
                            leg_gaps.append('await_conversion_caller_delivery_unproved')
                        views, consumer_gaps = _caller_consumers(ctx, advance, parent, conditions,
                            role, requires if role != 'COMPLETION' or requires else None, issues)
                        if delivery['match'] is not True:
                            views = tuple({**view, 'delivery_binding': delivery, 'status': 'PARTIAL',
                                'gaps': (*view['gaps'], 'await_conversion_caller_delivery_unproved')}
                                for view in views)
                        else:
                            views = tuple({**view, 'delivery_binding': delivery} for view in views)
                        caller_views.extend(views)
                        leg_gaps.extend(consumer_gaps)
                        terminal_views[-1].update(advance_result=parent['own_result'], delivery_binding=delivery)
                for state in states:
                    if state['binding']['match'] is None:
                        leg_gaps.append('await_conversion_state_unproved')
                    elif state['binding']['match'] is False:
                        leg_gaps.append('await_conversion_state_contradicted')
                        issues.append(('protocol_await_conversion_state', str(producer),
                            'conversion terminal uses another state completion'))
                if input_binding['match'] is None:
                    leg_gaps.append('await_conversion_input_unproved')
                elif input_binding['match'] is False:
                    leg_gaps.append('await_conversion_input_contradicted')
                    issues.append(('protocol_await_conversion_input', str(producer),
                        'conversion input is not this exact body outcome payload'))
                converted.append({'operation': producer, **leg, 'role': role, 'choices': leg_choices,
                    'rows': own_rows, 'input_binding': input_binding, 'packets': packets,
                    'terminals': tuple(terminal_views),
                    'states': tuple(states), 'status': 'CONTRADICTED' if input_binding['match'] is False
                        or any(state['binding']['match'] is False for state in states)
                        or any(terminal['binding']['match'] is False for terminal in terminal_views) else
                        'PARTIAL' if leg_gaps or input_binding['match'] is None or ctx.budget.truncated else 'DECLARED',
                    'gaps': tuple(dict.fromkeys(leg_gaps))})
                local.extend(leg_gaps)
        continued_raise = None
        if not producer_ids and kind == 'EXCEPTION' and structured_region(ctx, payload):
            from scar.ir.program.await_inventory_originals import structured_completion
            packets, packet_gaps = _packets(ctx, payload, choices)
            continued_raise = structured_completion(ctx, payload, packets, choices,
                parents, advance, None, (), 'EXCEPTION', _caller_consumers, issues)
            local.extend((*packet_gaps, *continued_raise['gaps']))
            caller_views.extend(continued_raise['consumers'])
            complete = not ctx.budget.truncated and bool(continued_raise['continuations']) and all(
                item['continuation']['known_inventory_complete']
                for item in continued_raise['continuations'])
            for parent in parents:
                alternatives = tuple(terminal for terminal in continued_raise['terminals']
                    if terminal['advance_result'] == parent['own_result'])
                checks = tuple(terminal['delivery_binding'] for terminal in alternatives)
                statuses = tuple(terminal['status'] for terminal in alternatives)
                unknown = not alternatives and not complete
                advance_legs.append({**parent, 'alternatives': alternatives,
                    'delivery_bindings': checks,
                    'consumers': tuple(view for terminal in alternatives for view in terminal['consumers']),
                    'status': 'CONTRADICTED' if 'CONTRADICTED' in statuses else
                        'PARTIAL' if ctx.budget.truncated or unknown
                            or any(item != 'DECLARED' for item in statuses) else
                        'DECLARED' if alternatives else 'NOT_APPLICABLE',
                    'gaps': (*(('await_body_raise_terminal_budget',) if ctx.budget.truncated else ()),
                        *(('await_body_raise_continuation_unproved',) if unknown else
                          tuple(dict.fromkeys(gap for terminal in alternatives for gap in terminal['gaps']))))})
        if not producer_ids and continued_raise is None:
            # Even total deletion of semantic adapters cannot delete the
            # caller's actual consumers or the original BodyReturn obligation.
            for parent in parents:
                if parent['own_result'] is None:
                    continue
                conditions = selected(ctx, choices, parent['edge'])
                delivery = (_result_copy(ctx, payload, parent['own_result'], conditions)
                            if kind in ('HANDOFF', 'EXCEPTION') else
                            answer(gaps=('await_completion_conversion_unproved',)))
                if delivery['match'] is False:
                    advance_legs.append({**parent, 'delivery_binding': delivery, 'consumers': (),
                        'status': 'NOT_APPLICABLE', 'gaps': ()})
                    continue
                views, consumer_gaps = _caller_consumers(ctx, advance, parent, conditions,
                    'HANDOFF' if kind == 'HANDOFF' else 'EXCEPTION' if kind == 'EXCEPTION' else 'UNRESOLVED',
                    False if kind in ('HANDOFF', 'EXCEPTION') else None, issues)
                caller_views.extend(views)
                local.extend(consumer_gaps)
                if delivery['match'] is None:
                    local.append('await_outcome_caller_delivery_unproved')
                    views = tuple({**view, 'status': 'PARTIAL', 'delivery_binding': delivery,
                        'gaps': (*view['gaps'], 'await_outcome_caller_delivery_unproved')} for view in views)
                    # Replace the same original consumer projections, keeping
                    # the unresolved delivery visible at every child level.
                    if views:
                        caller_views[-len(views):] = views
                advance_legs.append({**parent, 'delivery_binding': delivery, 'consumers': views,
                    'status': 'PARTIAL' if delivery['match'] is None or consumer_gaps else 'DECLARED',
                    'gaps': consumer_gaps})
        if producer_ids:
            for parent in parents:
                alternatives = tuple(terminal for leg in converted for terminal in leg['terminals']
                    if terminal.get('advance_result') == parent['own_result'])
                checks = tuple(terminal['delivery_binding'] for terminal in alternatives)
                advance_legs.append({**parent, 'alternatives': alternatives,
                    'delivery_bindings': checks,
                    'status': 'PARTIAL' if not checks or any(check['match'] is None for check in checks)
                        else 'DECLARED' if all(check['match'] is True for check in checks) else 'CONTRADICTED',
                    'gaps': () if checks else ('await_advance_outcome_leg_unproved',)})
        local.extend(gap for leg in advance_legs for gap in leg['gaps'])
        if any(leg['status'] == 'PARTIAL' for leg in advance_legs):
            local.append('await_advance_outcome_leg_unproved')
        checked += int(not ctx.budget.truncated)
        rows.append({**row, **({} if continued_raise is None else {
                'continuations': continued_raise['continuations'], 'terminals': continued_raise['terminals']}),
            'conversion_legs': tuple(converted), 'advance_legs': tuple(advance_legs),
            'consumers': tuple(caller_views),
            'status': 'CONTRADICTED' if any(leg['status'] == 'CONTRADICTED' for leg in converted)
                or any(leg['status'] == 'CONTRADICTED' for leg in advance_legs)
                or any(consumer['status'] == 'CONTRADICTED' for consumer in caller_views) else
                'PARTIAL' if local or ctx.budget.truncated else 'DECLARED',
            'gaps': tuple(dict.fromkeys(local)), **flags()})
        gaps.extend(local)
        if ctx.budget.truncated:
            cursor = cursor or {'phase': 'await_outcome_connections', 'key': key}
    complete = enumeration_complete and not ctx.budget.truncated
    return {'outcomes': tuple(rows), 'outcome_inventory': inventory(rows, checked, complete,
        cursor=cursor, drive_successors_total=successor_total), 'issues': tuple(issues), 'gaps': tuple(gaps)}


# Source-owned and expanded await failures on original control/result ports.
def _caller_cleanup(ctx, source, choices, issues):
    """Current caller completion, without retyping the callee's incoming R."""
    from scar.ir.program.suspension_interfaces import continuation_rows, structured_region
    from .model import StateVersionValue
    if not structured_region(ctx, source):
        return None
    continuations, missing = continuation_rows(ctx, source, choices)
    terminals, gaps = [], list(missing)
    for row in continuations:
        for receipt in row['continuation']['terminals']:
            if not ctx.spend('await_failure_caller_terminals'):
                gaps.append('await_failure_caller_budget')
                break
            operation = ctx.operations[receipt['terminal'].operation]
            state_uses = []
            if ctx.spend('await_failure_caller_state_uses', len(operation.terminator.values) + 1):
                for ordinal, identity in enumerate(operation.terminator.values):
                    actual = OwnedUse(operation.id, identity)
                    value = ctx.values[ctx.owned_use(operation.id, identity).value]
                    if type(value) is StateVersionValue:
                        state_uses.append({'ordinal': ordinal, 'actual': actual,
                            'value': value, 'location': value.location})
            local = list(receipt['gaps'])
            if ctx.budget.truncated:
                local.append('await_failure_caller_state_budget')
            if receipt['binding']['match'] is False:
                issues.append(('protocol_await_failure_current_source', str(operation.id),
                    'caller terminal is not its current cleanup completion source'))
            terminals.append({**receipt, 'state_uses': tuple(state_uses),
                'requires_final_extraction': False,
                'status': 'CONTRADICTED' if receipt['binding']['match'] is False else
                    'PARTIAL' if local or ctx.budget.truncated else receipt['status'],
                'gaps': tuple(dict.fromkeys(local))})
            gaps.extend(local)
    if not terminals:
        gaps.append('await_failure_caller_terminal_unproved')
    if ctx.budget.truncated:
        gaps.append('await_failure_caller_budget')
    return {'caller_continuations': continuations, 'caller_terminals': tuple(terminals),
            'gaps': tuple(dict.fromkeys(gaps))}




def _seeds(ctx, scope_id=None):
    """Original scope subjects survive deletion of all own RAISE rows."""
    seeds, seen, missing = [], set(), []
    for scope in ctx.graph.protocol_scopes:
        if scope_id is not None and scope.id != scope_id:
            continue
        if scope.profile not in (ProtocolProfile.AWAIT_ACQUISITION_V1,
                                  ProtocolProfile.COROUTINE_WRAPPER_CONSTRUCTION_V1):
            continue
        operation = ctx.operations[scope.subject.operation]
        clause = (ProtocolClause.AWAIT_ACQUISITION
                  if scope.profile is ProtocolProfile.AWAIT_ACQUISITION_V1
                  else ProtocolClause.WRAPPER_CONSTRUCTION)
        rows = tuple(row for row in ctx.rows_by_clause.get((scope.id, clause), ())
                     if row.operation == operation.id)
        for parent in result_legs(ctx, operation):
            if parent['kind'] is not ExitKind.RAISE:
                continue
            own_row = next((row for row in rows if getattr(row.contract, 'result', None)
                            == parent['own_result']), None)
            common = {'scope': scope.id, 'source_operation': operation.id,
                'parent_result': parent['own_result'], 'parent_edge': parent['edge'],
                'acquisition': _ref(own_row) if own_row is not None
                    and type(own_row.contract) is AwaitAcquisition else None,
                'source_row': own_row}

            def append(stage_operation, edge, stage, choices, *, own=False):
                key = scope.id, operation.id, stage_operation, edge
                if key not in seen:
                    seen.add(key)
                    seeds.append({**common, 'key': key, 'stage_operation': stage_operation,
                        'edge': edge, 'stage': stage, 'choices': choices, 'own': own})

            append(operation.id, parent['edge'], 'ACQUISITION_RAISE'
                if clause is ProtocolClause.AWAIT_ACQUISITION else 'WRAPPER_CONSTRUCTION_RAISE', (), own=True)
            for row in rows:
                expansion = getattr(row.contract, 'expansion', None)
                if type(expansion) is not ExpandedAwaitCallee:
                    continue
                prefix = ()
                for ref, stage_kind in ((expansion.lookup, 'LOOKUP'),
                                        (expansion.call, 'METHOD_CALL'),
                                        (expansion.qualification, 'QUALIFICATION_RAISE')):
                    if type(ref) is StaticProtocolGap:
                        missing.append('await_failure_expansion_stage_missing')
                        break
                    local = ctx.row(ref)
                    stage_operation = ctx.operations[local.operation]
                    branch = getattr(local.contract, 'branch', None)
                    if branch is not None:
                        reject = ControlEdgeReference(branch, 1 if local.contract.accepts_true else 0)
                        append(branch, reject, 'QUALIFICATION_REJECT', prefix)
                        continue
                    for leg in result_legs(ctx, stage_operation):
                        if leg['kind'] is ExitKind.RAISE:
                            append(stage_operation.id, leg['edge'], stage_kind, prefix)
                    normal = tuple(leg for leg in result_legs(ctx, stage_operation)
                                   if leg['kind'] is ExitKind.NORMAL)
                    if len(normal) == 1 and normal[0]['edge'] is not None:
                        prefix = selected(ctx, prefix, normal[0]['edge'])
                    else:
                        missing.append('await_failure_normal_prefix_unresolved')
                        break
    ctx.preparation['await_original_failure_seeds'] = len(seeds)
    return tuple(seeds), tuple(dict.fromkeys(missing))


def failure_paths(ctx, *, scope=None, derive=True):
    from .protocol_await import _result_copy
    seeds, preparation_gaps = _seeds(ctx, scope)
    paths, issues, gaps, checked, cursor = [], [], list(preparation_gaps), 0, None
    for seed in seeds:
        path = {key: value for key, value in seed.items()
                if key not in ('source_row', 'own', 'parent_result', 'parent_edge')}
        path.update(source_result=seed['parent_result'], source_edge=seed['parent_edge'])
        local = []
        if not derive or not ctx.spend('await_failure_paths'):
            paths.append({**path, 'terminals': (), 'status': 'NOT_CHECKED',
                'gaps': ('await_static_details_deferred_without_capture',) if not derive else (), **flags()})
            if derive:
                cursor = cursor or {'phase': 'await_failure_paths', 'key': seed['key']}
            continue
        # A BRANCH supplies only an original control edge. Use its original
        # first landing formal as a traversal anchor, never a fake result.
        stage = ctx.operations[seed['stage_operation']]
        if stage.terminator.kind is TerminatorKind.BRANCH:
            landing = ctx.blocks[stage.terminator.successors[seed['edge'].successor_index].block]
            anchor = ctx.values[landing.arguments[0]].origin if landing.arguments else None
        else:
            anchor = next((leg['own_result'] for leg in result_legs(ctx, stage)
                           if leg['edge'] == seed['edge']), None)
        whole_choices = selected(ctx, seed['choices'], seed['edge'], seed['parent_edge'])
        caller = _caller_cleanup(ctx, seed['parent_result'], whole_choices, issues)
        endpoints, endpoint_gaps = (((), ()) if caller is not None and seed['own'] else
            ((), ('await_failure_anchor_missing',)) if anchor is None else
            terminal_inventory(ctx, anchor, whole_choices))
        if caller is not None:
            local.extend(caller['gaps'])
        local.extend(endpoint_gaps)
        if seed['source_row'] is None:
            local.append('await_failure_source_row_missing')
        receiver_ledger = consumers(ctx, seed['parent_result'], whole_choices)
        receipts = (None if seed['source_row'] is None else
                    ctx.result_consumers.get(_ref(seed['source_row'])))
        targets = list(receiver_ledger['targets'])
        if receipts is not None:
            targets.extend(target for target in receipts.targets
                           if type(target) is not StaticProtocolGap and target not in targets)
        receiver_views = tuple({'actual': target,
            'binding': bind_target(ctx, seed['parent_result'], target, whole_choices)} for target in targets)
        if receipts is None:
            local.append('protocol_result_consumers_missing')
        elif any(target not in receipts.targets for target in receiver_ledger['targets']):
            local.append('await_failure_consumer_receipt_incomplete')
        local.extend(receiver_ledger['gaps'])
        source_bindings = () if caller is None else tuple(view['binding'] for view in receiver_views)
        for binding in source_bindings:
            if binding['match'] is False:
                local.append('await_failure_source_consumer_contradicted')
                issues.append(('protocol_await_failure_source_consumer', str(seed['key']),
                    'original incoming failure does not reach its declared source consumer'))
            elif binding['match'] is None:
                local.append('await_failure_source_consumer_unproved')
        terminals = []
        for terminal in endpoints:
            if not ctx.spend('await_failure_terminals'):
                break
            packet = packet_use(ctx, terminal)
            raised = ctx.operations[terminal.operation].terminator.kind is TerminatorKind.RAISE
            report = (answer(False, gaps=('await_failure_whole_packet_missing',)) if packet is None else
                ctx.bind(seed['parent_result'], packet, guard=whole_choices) if seed['own'] else
                _result_copy(ctx, packet, seed['parent_result'], whole_choices))
            states = state_links(ctx, terminal, seed['parent_edge'], whole_choices, outward=seed['own'])
            terminal_gaps = list(local)
            bindings = (report, *(state['binding'] for state in states),
                        *(() if caller is not None else (consumer['binding'] for consumer in receiver_views)))
            stage_fields = {}
            if caller is not None and not seed['own']:
                stage_source = None if stage.terminator.kind is TerminatorKind.BRANCH else anchor
                stage_binding = None if stage_source is None else (
                    answer(gaps=('await_failure_stage_packet_missing',)) if packet is None else
                    ctx.bind(stage_source, packet, guard=whole_choices))
                origins = (None if stage_source is not None or packet is None else
                           ctx.origins(packet, guard=whole_choices))
                stage_fields = {'stage_source': stage_source, 'stage_binding': stage_binding,
                    'control_anchor': seed['edge'] if stage_source is None else None,
                    'stage_origins': origins}
                if stage_binding is not None:
                    bindings = (*bindings, stage_binding)
                elif origins is None or not origins['source_inventory_complete'] or not origins['recipes']:
                    terminal_gaps.append('await_failure_stage_origins_unproved')
            if not raised:
                terminal_gaps.append('await_failure_terminal_not_raise')
            for binding in bindings:
                if binding['match'] is None:
                    terminal_gaps.append('await_failure_binding_unproved')
                elif binding['match'] is False:
                    terminal_gaps.append('await_failure_binding_contradicted')
                    issues.append(('protocol_await_failure_copy', str(seed['key']),
                        'failure packet/state/consumer is not the same exact original completion'))
            terminals.append({'terminal': terminal, 'packet': packet, 'binding': report,
                'states': states, 'consumers': receiver_views, **stage_fields,
                'status': 'CONTRADICTED' if any(binding['match'] is False for binding in bindings) else
                          'PARTIAL' if terminal_gaps or not raised or ctx.budget.truncated else 'DECLARED',
                'gaps': tuple(dict.fromkeys(terminal_gaps)), **flags()})
            gaps.extend(terminal_gaps)
        checked += int(not ctx.budget.truncated)
        local.extend(gap for terminal in terminals for gap in terminal['gaps'])
        children = (*terminals, *(() if caller is None else caller['caller_terminals']))
        paths.append({**path, 'terminals': tuple(terminals), **({} if caller is None else {
                'caller_continuations': caller['caller_continuations'],
                'caller_terminals': caller['caller_terminals'], 'source_consumers': receiver_views}),
            'status': 'CONTRADICTED' if any(binding['match'] is False for binding in source_bindings)
                or any(terminal['status'] == 'CONTRADICTED' for terminal in children) else
                      'PARTIAL' if local or ctx.budget.truncated or not children else 'DECLARED',
            'gaps': tuple(dict.fromkeys(local)), **flags()})
        gaps.extend(local)
        if ctx.budget.truncated:
            cursor = cursor or {'phase': 'await_failure_terminals', 'key': seed['key']}
    complete = not preparation_gaps and not ctx.budget.truncated
    return {'failure_paths': tuple(paths), 'failure_path_inventory': inventory(paths, checked, complete,
        cursor=cursor), 'issues': tuple(issues), 'gaps': tuple(gaps)}


__all__ = [
    'outcome_connections',
    'COMPLETION_RULES',
    'flags',
    'answer',
    'combine',
    'prepare',
    'terminal_inventory',
    'terminal_payload',
    'packet_use',
    'bind_target',
    'case_binding',
    'consumers',
    'state_links',
    'inventory',
    'failure_paths',
]
