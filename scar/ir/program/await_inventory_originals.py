"""Await inventory originals."""

from scar.ir.program.await_outcomes import selected, state_links
from scar.ir.program.suspension_interfaces import continuation_rows
from scar.ir.program.model import ExitKind, TerminatorKind, InvokeLiteralResult
from scar.ir.program.operation_results import evaluated_literal, evaluated_sources
from scar.ir.program.protocol_await_records import (
    AwaitAcquisition,
    AwaitIteratorQualification,
    ExpandedAwaitCallee,
)
from scar.ir.program.protocol_connections import ProtocolOperationRef
from scar.ir.program.protocols import OwnedUse, StaticProtocolGap
from scar.ir.program.resumption import ResumableKind
from scar.ir.program.structured import ControlEdgeReference


# Original await anchors retained independently from budgeted associations.
#
# Explicit link/case FKs supply declared, unchecked candidates only. Independent
# source, advance and Drive inventories never become a Cartesian ownership join.
def _key(job):
    return (job['source'], job['advance'], job.get('input_guard'),
            job['choices'], job['drive'])


def prepare(ctx, advances, links):
    from .protocol_await import _site, success_source
    by_advance = {_site(row): row for row in advances}
    grouped, templates = {}, []
    for link in links:
        grouped.setdefault((link.source, link.advance, link.drive), []).append(link)
    for (source, advance, drive), originals in grouped.items():
        row = ctx.rows.get((source.scope, source.clause, source.operation, source.slot))
        step = by_advance.get(advance)
        if row is None or step is None:
            continue
        port, forced = success_source(ctx, row)
        attachment = ctx.call_inputs.get(advance)
        cases = (None,) if attachment is None or not attachment.cases else attachment.cases
        for case in cases:
            guard = None if case is None else case.guard
            choices = forced if guard is None or guard in forced else (*forced, guard)
            conflict = any(a.operation == b.operation and a != b
                           for number, a in enumerate(choices) for b in choices[number + 1:])
            templates.append({'scope': source.scope, 'source': source,
                'source_operation': source.operation, 'source_success': port,
                'accepted_edges': forced, 'choices': choices, 'case': case,
                'input_guard': guard, 'input_case_missing': case is None,
                'advance': advance, 'advance_operation': advance.operation,
                'iterator_use': None, 'drive': drive, 'links': tuple(originals),
                'binding': None, 'status': 'NOT_CHECKED',
                'gaps': ('await_declared_association_unchecked',),
                'cursor': {'phase': 'await_declared_association', 'advance': advance},
                'choice_conflict': conflict, 'declared_anchor_only': True})
    ctx._await_declared_job_templates = tuple(templates)
    ctx._await_original_anchor_inventory = {
        'advances': tuple(by_advance), 'links': tuple(links),
        'drives': tuple((drive.operation, tuple(edge.source for edge in
            ctx.operations[drive.operation].terminator.successors)) for drive in ctx.graph.body_drives),
        'associations_total': None,
        'meaning': 'independent original anchors and declared FKs; not reached ownership or complete pairing'}
    ctx.preparation['await_declared_link_case_templates'] = len(templates)
    ctx.preparation['await_original_advance_anchors'] = len(advances)
    ctx.preparation['await_ownership_joins_precomputed'] = 0


def unchecked_suffix(ctx, jobs):
    """Present raw original tuples without deriving an unpaid suffix."""
    present = {_key(job) for job in jobs}
    result = []
    for original in ctx._await_declared_job_templates:
        if _key(original) not in present:
            result.append(original)
            present.add(_key(original))
    ctx.preparation['await_unchecked_declared_jobs_presented'] = len(result)
    return tuple(result)


# Connect typed same-region completion receipts to their exact caller ports.
def structured_completion(ctx, source, packets, choices, parents, advance,
                          input_binding, own_rows, role, consumers, issues):
    from .protocol_await import _result_copy
    rows, missing = continuation_rows(ctx, source, choices, candidates=packets)
    terminals, states, caller_views, gaps = [], [], [], list(missing)
    for row in rows:
        for receipt in row['continuation']['terminals']:
            if not ctx.spend('await_structured_terminals'):
                gaps.append('await_structured_budget')
                break
            terminal, actual = receipt['terminal'], receipt['actual']
            copied, root = receipt['binding'], receipt['root_binding']
            if copied['match'] is False:
                gaps.append('await_conversion_packet_terminal_contradicted')
                issues.append(('protocol_await_conversion_packet', str(terminal.operation),
                    'cleanup terminal is not its current completion source'))
            elif copied['match'] is None:
                gaps.append('await_conversion_packet_terminal_unproved')
            kind = (ExitKind.RAISE if ctx.operations[terminal.operation].terminator.kind
                    is TerminatorKind.RAISE else ExitKind.NORMAL)
            for parent in parents:
                if parent['kind'] is not kind or parent['own_result'] is None:
                    continue
                conditions = selected(ctx, receipt['choices'], parent['edge'])
                delivery = _result_copy(ctx, actual, parent['own_result'], conditions)
                current_states = state_links(ctx, terminal, parent['edge'], conditions)
                states.extend(current_states)
                for state in current_states:
                    if state['binding']['match'] is False:
                        gaps.append('await_conversion_state_contradicted')
                        issues.append(('protocol_await_conversion_state', str(terminal.operation),
                            'current cleanup terminal state does not reach its own caller result'))
                    elif state['binding']['match'] is None:
                        gaps.append('await_conversion_state_unproved')
                preserved = root['match'] is True and receipt['role'] == 'packet'
                overridden = (root['match'] is False and copied['match'] is True
                    and receipt['current_source'] != receipt['root_source'])
                receiver_role = 'OVERRIDE' if overridden else role
                requires = (False if overridden or role != 'COMPLETION' else
                    True if preserved and copied['match'] is True and delivery['match'] is True
                    and input_binding['match'] is True and own_rows else None)
                views, consumer_gaps = ((), ()) if delivery['match'] is False else consumers(
                    ctx, advance, parent, conditions, receiver_role, requires, issues)
                views = tuple({**view, 'delivery_binding': delivery,
                    **({'status': 'PARTIAL', 'gaps': tuple(dict.fromkeys((*view.get('gaps', ()),
                        'await_conversion_caller_delivery_unproved')))}
                       if delivery['match'] is None else {})} for view in views)
                caller_views.extend(views)
                gaps.extend(consumer_gaps)
                terminal_gaps = list(receipt['gaps'])
                if not preserved and not overridden and role == 'COMPLETION':
                    terminal_gaps.append('await_completion_preservation_unproved')
                if delivery['match'] is not True:
                    terminal_gaps.append('await_conversion_caller_delivery_contradicted'
                        if delivery['match'] is False else 'await_conversion_caller_delivery_unproved')
                terminal_gaps.extend(gap for state in current_states for gap in state.get('gaps', ()))
                terminal_gaps.extend('await_conversion_state_contradicted'
                    if state['binding']['match'] is False else 'await_conversion_state_unproved'
                    for state in current_states if state['binding']['match'] is not True)
                if ctx.budget.truncated:
                    terminal_gaps.append('await_completion_terminal_budget')
                gaps.extend(terminal_gaps)
                matches = (copied['match'], delivery['match'],
                           *(state['binding']['match'] for state in current_states))
                terminals.append({**receipt, 'packet': actual, 'choices': conditions,
                    'advance_result': parent['own_result'], 'delivery_binding': delivery,
                    'states': current_states, 'consumers': views,
                    'status': 'CONTRADICTED' if False in matches else
                        'PARTIAL' if None in matches or terminal_gaps else receipt['status'],
                    'gaps': tuple(dict.fromkeys(terminal_gaps))})
    return {'continuations': rows, 'terminals': tuple(terminals), 'states': tuple(states),
            'consumers': tuple(caller_views), 'gaps': tuple(dict.fromkeys(gaps))}


# Conditional reachability from an exact original iterator qualification.
#
# These are bounded query views, not new CFG edges or native identity facts.
# Syntactically present but infeasible accepting legs stay in the inventory.
BAD_ITERATORS = (ResumableKind.COROUTINE, ResumableKind.GENERATOR_COROUTINE,
                ResumableKind.ASYNC_GENERATOR, 'NON_ITERATOR')


def _ref(row):
    return ProtocolOperationRef(row.scope, row.clause, row.operation, row.slot)


def _flags():
    return {'source_runtime_verified': False, 'runtime_identity_verified': False,
            'value_identity_verified': False, 'transformation_authorized': False}


def qualification_view(ctx, row, choices=()):
    """Interpret all source cases without certifying a native implementation."""
    choices = ctx.selected_edges(choices)
    if not hasattr(ctx, '_await_qualification_views'):
        ctx._await_qualification_views = {}
    key = _ref(row), choices
    if key in ctx._await_qualification_views:
        return ctx._await_qualification_views[key]
    contract = row.contract
    source = evaluated_sources(ctx, contract.expected,
        OwnedUse(row.operation, contract.input_use), choices, role='GENERAL_RETURN')
    if contract.branch is not None:
        branch = ctx.operations[contract.branch]
        edge = ControlEdgeReference(branch.id, 0 if contract.accepts_true else 1)
        boolean = ctx.bind(contract.result, OwnedUse(branch.id, branch.terminator.condition),
                           guard=choices or None)
    else:
        boolean, edge = None, None
        for sibling in ctx.rows_by_clause.get((row.scope, row.clause), ()):
            if sibling.operation != row.operation:
                continue
            port = sibling.contract.result
            if (type(port) is InvokeLiteralResult and ctx.operations[row.operation].terminator.
                    successors[port.edge.successor_index].kind is ExitKind.NORMAL):
                edge = port.edge
                break
    valid = (source['binding']['match'] is True
             and (boolean is None or boolean['match'] is True))
    cases = []
    for case in source['cases']:
        accepts = None
        if valid and case['current'] is True:
            accepts = (False if case['kind'] in BAD_ITERATORS else
                       True if case['kind'] is ResumableKind.GENERATOR else None)
        cases.append({'source_case': case, 'accepts': accepts,
            'reason': 'known_non_iterator_for_general_await' if accepts is False else
                      'declared_generator_iterator' if accepts is True else 'qualification_unproved'})
    complete = source['classification_complete'] and valid and not ctx.budget.truncated
    values = {case['accepts'] for case in cases}
    acceptance = next(iter(values)) if complete and len(values) == 1 else None
    declared_boolean, predicate_conflict = None, False
    if contract.branch is not None:
        from scar.ir.literals import LiteralKind, PythonLiteral
        literal = evaluated_literal(ctx, contract.result)
        if (literal is not None and type(literal.literal) is PythonLiteral
                and literal.literal.kind is LiteralKind.BOOL):
            declared_boolean = literal.literal.payload
            expected_boolean = acceptance if contract.accepts_true else (
                None if acceptance is None else not acceptance)
            predicate_conflict = (expected_boolean is not None
                                  and declared_boolean is not expected_boolean)
    result = {'record': row, 'source': source, 'accepting_edge': edge,
        'bool_binding': boolean, 'cases': tuple(cases), 'acceptance': acceptance,
        'declared_boolean': declared_boolean, 'predicate_conflict': predicate_conflict,
        'known_inventory_complete': complete, 'conditions': (
            'qualification native semantics and source correspondence remain conditional',), **_flags()}
    ctx._await_qualification_views[key] = result
    return result


def _controlled(ctx, edge, targets):
    """An original accepting edge controls every stated success target.

    Remove the edge from the original graph, without assuming its selection.
    A reachable bypass is a counterexample; budget exhaustion is unknown.
    """
    key = edge, tuple(targets)
    if not hasattr(ctx, '_await_qualification_control'):
        ctx._await_qualification_control = {}
    if key in ctx._await_qualification_control:
        return ctx._await_qualification_control[key]
    region = ctx.region_of[edge.operation]
    _, outgoing, _, reached, _ = ctx._edges(region)
    wanted = {ctx.block_of[target] for target in targets}
    if not wanted or not wanted.issubset(reached):
        return None
    if ctx.block_of[edge.operation] not in reached:
        return False
    pending, seen = [ctx.regions[region].entry], set()
    while pending:
        if not ctx.spend('await_qualification_control_blocks'):
            return None
        block = pending.pop()
        if block in seen:
            continue
        seen.add(block)
        if block in wanted:
            ctx._await_qualification_control[key] = False
            return False
        for operation, number, target, _ in outgoing[block]:
            if not ctx.spend('await_qualification_control_edges'):
                return None
            if (operation, number) != (edge.operation, edge.successor_index):
                pending.append(target)
    ctx._await_qualification_control[key] = True
    return True


def protected_advance(ctx, candidate_source, iterator, choices=()):
    """An exact check covers this candidate and this original success route.

    The native predicate can exclude a bad case only after current copies and
    control coverage are established. A nearby or bypassed check is insufficient.
    """
    from .protocol_await import _kind, _prepare, _result_copy, _selected, success_source
    rows = _prepare(ctx)[0][AwaitIteratorQualification]
    if not rows or iterator is None:
        return {'match': False, 'guards': (), 'gaps': ()}
    choices = ctx.selected_edges(choices)
    if not hasattr(ctx, '_await_protected_advances'):
        ctx._await_protected_advances = {}
    key = candidate_source, iterator, choices
    if key in ctx._await_protected_advances:
        return ctx._await_protected_advances[key]
    guarded, unknown = [], []
    for row in rows:
        if not ctx.spend('await_qualification_guards'):
            unknown.append('await_qualification_guard_budget')
            break
        accepted, forced = success_source(ctx, row)
        edge = (ControlEdgeReference(row.contract.branch, 0 if row.contract.accepts_true else 1)
                if row.contract.branch is not None else
                row.contract.result.edge if type(row.contract.result) is InvokeLiteralResult else None)
        if accepted is None or edge is None:
            continue
        try:
            selected = _selected(ctx, choices, forced, edge)
        except ValueError:
            # This request selects the rejecting leg, not this protection.
            continue
        region = ctx.region_of[row.operation]
        paths = []
        if region == ctx.region_of[iterator.operation]:
            copied = ctx.bind(accepted, iterator, guard=selected or None)
            if copied['match'] is not False:
                paths.append({'bindings': (copied,),
                    'controlled': _controlled(ctx, edge, (iterator.operation,))})
        else:
            for parent in ctx._await_qualification_parents.get(_ref(row), ()):
                if not ctx.spend('await_qualification_parent_sites'):
                    unknown.append('await_qualification_parent_budget')
                    break
                if (type(parent.contract) is not AwaitAcquisition
                        or type(parent.contract.expansion) is not ExpandedAwaitCallee
                        or parent.contract.expansion.qualification != _ref(row)
                        or ctx.operations[parent.operation].callee != region
                        or _kind(ctx, parent.contract.result) is not ExitKind.NORMAL):
                    continue
                delivered = ctx.bind(parent.contract.result, iterator, guard=selected or None)
                if delivered['match'] is False:
                    continue
                terminals = ctx._await_normal_terminals.get(region, ())
                if not ctx.spend('await_qualification_success_terminals', len(terminals)):
                    unknown.append('await_qualification_terminal_budget')
                    break
                paths.append({'parent': parent, 'bindings': (
                    _result_copy(ctx, accepted, parent.contract.result, selected),
                    delivered),
                    'controlled': _controlled(ctx, edge, terminals)})
        # Unrelated predicates have no path skeleton to this iterator. Their
        # unknown external input cannot weaken a known unqualified rejection.
        paths = [path for path in paths if path['controlled'] is not False
                 and all(binding['match'] is not False for binding in path['bindings'])]
        if not paths:
            continue
        view = qualification_view(ctx, row, choices)
        if view['predicate_conflict']:
            continue
        input_report = ctx.bind(candidate_source, OwnedUse(row.operation, row.contract.input_use),
                                guard=choices or None)
        if input_report['match'] is False:
            continue
        if (input_report['match'] is None or view['source']['binding']['match'] is not True
                or view['bool_binding'] is not None and view['bool_binding']['match'] is not True):
            unknown.append('await_qualification_input_unproved')
            continue
        for path in paths:
            bindings = (input_report, *path['bindings'])
            checks = (path['controlled'], *(binding['match'] for binding in bindings))
            if all(check is True for check in checks):
                guarded.append({'qualification': row, 'feasibility': view, **path, 'bindings': bindings})
            elif False not in checks:
                unknown.append('await_qualification_path_unproved')
    result = {'match': None if unknown else True if guarded else False,
              'guards': tuple(guarded), 'gaps': tuple(dict.fromkeys(unknown))}
    ctx._await_protected_advances[key] = result
    return result


__all__ = [
    'BAD_ITERATORS',
    'qualification_view',
    'protected_advance',
    'prepare',
    'unchecked_suffix',
    'structured_completion',
]
