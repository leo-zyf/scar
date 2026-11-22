"""Protocol connection validation."""

from scar.ir.program.model import (
    BlockArgument,
    StateVersionValue,
    TerminatorKind,
    InvokeLiteralResult,
    OperationResult,
    ReferenceValue,
)
from scar.ir.program.structured import ControlEdgeReference
from scar.ir.program.protocol_connections import (
    InitialAdvance,
    InjectedPacketTrigger,
    ProtocolOperationRef,
    RawRequestTrigger,
)
from scar.ir.program.protocols import (
    AdmissionPredicate,
    HostTerminal,
    NativeExit,
    OutcomeAdapter,
    OwnedUse,
    ProtocolCall,
    RequestProcessing,
    StaticProtocolGap,
    ContractResult,
)
from scar.ir.program.resumption import SuspendTerminator
from scar.ir.execution.validity_budget import ValidityBudget
from scar.ir.program.protocol_flow import ProtocolFlow, ProtocolPort, ProtocolReferenceError
from scar.ir.program.state_addresses import ExistingAccess, InvokeStateOperand


# Independent input and consumer inventories over canonical program ports.
#
# Missing attachments remain obligations. Supplied targets or successful input
# subsets cannot choose the denominator or grant transformation permission.
def _site(row):
    return ProtocolOperationRef(row.scope, row.clause, row.operation, row.slot)


def _joint_routes(ctx, reports):
    """Join whole tuples on caller and CFG choices, retaining route support."""
    joined = [(None, {}, ())]
    for report in reports:
        next_rows = []
        for route in report["routes"]:
            if route.get("match") is not True:
                continue
            context = route.get("context")
            choices = dict(route.get("choices", {}))
            for old_context, old_choices, support in joined:
                if not ctx.spend("connection_tuple_joins"):
                    return (), ("protocol_connection_budget_exhausted",)
                if old_context is not None and context is not None and old_context != context:
                    continue
                if any(key in old_choices and old_choices[key] != value for key, value in choices.items()):
                    continue
                next_rows.append((old_context if context is None else context,
                                  {**old_choices, **choices}, (*support, route)))
        joined = next_rows
        if not joined:
            return (), ("whole_input_tuple_context_unresolved",)
    return tuple({"context": context, "choices": choices, "support": support}
                 for context, choices, support in joined), ()


def _guard_inventory(ctx, row, attachment):
    cases, issues, gaps = attachment.cases, [], []
    if len(cases) == 1 and cases[0].guard is None:
        return (), (), ()
    if any(case.guard is None for case in cases):
        return (("protocol_input_guard", str(row.operation),
                 "multiple cases require exact nonempty incoming guards"),), (), ()
    guards = tuple(case.guard for case in cases)
    targets = []
    for guard in guards:
        operation = ctx.operations.get(guard.operation)
        if (operation is None or operation.terminator is None or
                guard.successor_index >= len(operation.terminator.successors) or
                ctx.region_of[operation.id] != ctx.region_of[row.operation]):
            issues.append(("protocol_input_guard", str(row.operation),
                           "guard must select an original incoming edge in this region"))
        else:
            targets.append(operation.terminator.successors[guard.successor_index].block)
    if issues:
        return tuple(issues), (), ()
    if len(set(targets)) != 1 or len(set(guards)) != len(guards):
        return (("protocol_input_guard", str(row.operation),
                 "case guards must be unique incoming edges of the same canonical merge"),), (), ()
    merge = targets[0]
    _, _, incoming, reached, _ = ctx._edges(ctx.region_of[row.operation])
    expected = {(edge[0], edge[1]) for parent, edge in incoming[merge] if parent in reached}
    declared = {(guard.operation, guard.successor_index) for guard in guards}
    if declared - expected:
        issues.append(("protocol_input_guard", str(row.operation),
                       "case includes a nonapplicable merge predecessor"))
    missing = tuple(sorted(expected - declared, key=lambda item: (item[0].wire, item[1])))
    if missing:
        gaps.append({"code": "protocol_input_case_missing", "merge": merge, "incoming": missing})
    # A shared early branch is insufficient. Some actual role must be a copy
    # of a canonical formal from this merge, possibly through later JUMPs.
    from .model import BlockArgument
    actuals = tuple(OwnedUse(row.operation, use) for use in
                    (row.contract.receiver_use, *row.contract.argument_uses))
    carries, unknown = False, False
    for index in range(len(ctx.blocks[merge].arguments)):
        for actual in actuals:
            if isinstance(ctx.values[ctx.owned_use(actual.operation, actual.use).value], StateVersionValue):
                continue
            result = ctx.bind(BlockArgument(merge, index), actual)
            carries |= result["match"] is True
            unknown |= result["match"] is None
    if not carries:
        if unknown:
            gaps.append("protocol_input_merge_support_unresolved")
        else:
            issues.append(("protocol_input_guard", str(row.operation),
                           "guard merge does not carry any actual input role"))
    return tuple(issues), tuple(gaps), missing


def _call_inputs(ctx, row, attachment):
    site, issues, gaps, cases = _site(row), [], [], []
    if attachment is None:
        return {"site": site, "status": "MISSING", "cases": (),
                "gaps": ("protocol_call_inputs_missing",), "issues": ()}
    guard_issues, guard_gaps, missing = _guard_inventory(ctx, row, attachment)
    issues.extend(guard_issues)
    gaps.extend(guard_gaps)
    for case in attachment.cases:
        if not ctx.spend("connection_input_cases"):
            gaps.append("protocol_connection_budget_exhausted")
            break
        if len(case.arguments) != len(row.contract.argument_uses):
            issues.append(("protocol_input_arity", str(row.operation),
                           "expected argument tuple must match the actual ordered call arguments"))
            continue
        try:
            reports = tuple(ctx.bind(expected, OwnedUse(row.operation, actual),
                                    guard=case.guard, path="protocol_call_inputs")
                            for expected, actual in zip((case.receiver, *case.arguments),
                                (row.contract.receiver_use, *row.contract.argument_uses)))
            trigger = case.trigger
            if type(trigger) is RawRequestTrigger:
                ctx.owned_use(trigger.raw.operation, trigger.raw.use)
                reports += (ctx.available(trigger.raw, row.operation, guard=case.guard),)
            elif type(trigger) is InjectedPacketTrigger:
                reports += (ctx.available(trigger.packet, row.operation, guard=case.guard),)
            elif type(trigger) is InitialAdvance:
                reports += (ctx.available(trigger.iterator, row.operation, guard=case.guard),)
            elif type(trigger) is StaticProtocolGap:
                gaps.append(trigger)
            for report in reports:
                gaps.extend(report["gaps"] if report["match"] is None else ())
                if report["match"] is False:
                    issues.append(("protocol_input_source", str(row.operation),
                                   "actual input or trigger does not come from its independent expected source"))
            correlation = ctx.correlate(*reports)
            gaps.extend(correlation["gaps"])
            if correlation["match"] is False and all(report["match"] is True for report in reports):
                issues.append(("protocol_input_context", str(row.operation),
                               "input roles belong to incompatible caller/CFG contexts"))
            cases.append({"case": case, "inputs": reports, "joint": correlation["routes"],
                          "correlation": correlation})
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(("protocol_input_reference", str(row.operation), str(exc)))
    return {"site": site, "status": "REJECTED" if issues else "PARTIAL" if gaps else "DECLARED",
            "cases": tuple(cases), "missing_incoming": missing,
            "issues": tuple(issues), "gaps": tuple(gaps)}


def _consumer_role(operation, use):
    """Keep actual consumers, excluding mere original edge forwarding Uses."""
    term = operation.terminator
    if term is None:
        return operation.callee is None or use in operation.call_arguments
    if term.kind is TerminatorKind.INVOKE:
        return use in operation.invoke_arguments
    if type(term) is SuspendTerminator:
        return use == term.exported or use in term.handoff or any(use == slot.use for slot in term.saved)
    if term.kind in (TerminatorKind.RETURN, TerminatorKind.RAISE):
        return use in term.values
    if term.kind is TerminatorKind.BRANCH:
        return use == term.condition
    # Special structured terminators have their own canonical payload Uses;
    # retain these rather than silently classifying them as dead forwarding.
    return term.kind is not TerminatorKind.JUMP


def consumer_ledger(ctx, row=None, *, source=None, site=None, source_operation=None):
    """Derive known exact targets before inspecting any targets attachment."""
    if row is not None:
        site, source_operation = _site(row), row.operation
    targets, support, gaps = [], [], []
    try:
        if row is not None:
            source = row.contract.output if type(row.contract) is OutcomeAdapter else ctx.row_result(row)
        if type(source) in (HostTerminal, NativeExit):
            return {"site": site, "source": source, "targets": (source,),
                    "support": (), "direct_terminal": True, "gaps": (),
                    "known_inventory_complete": True, "source_inventory_complete": False}
        port = ctx.port(source)
    except (KeyError, TypeError, ValueError) as exc:
        return {"site": site, "source": None, "targets": (), "support": (),
                "gaps": ({"code": "consumer_source_unresolved", "message": str(exc)},),
                "known_inventory_complete": False, "source_inventory_complete": False}
    if port is None or port.gap is not None:
        return {"site": site, "source": source, "targets": (), "support": (),
                "gaps": ("consumer_source_unresolved",), "known_inventory_complete": False,
                "source_inventory_complete": False}
    for operation in ctx.graph.operations:
        for use in operation.uses:
            if not ctx.spend("connection_consumer_inventory"):
                gaps.append("protocol_connection_budget_exhausted")
                break
            if (isinstance(ctx.values[use.value], StateVersionValue)
                    or not _consumer_role(operation, use.id)):
                continue
            actual = OwnedUse(operation.id, use.id)
            report = ctx.bind(source, actual)
            conditional = []
            if report["match"] is not True and type(ctx.values[use.value].origin) is BlockArgument:
                original = ctx.origins(actual)
                guards = []
                for recipe in original["recipes"]:
                    if recipe["leaf"].id not in port.values:
                        continue
                    for step in recipe["path"]:
                        if step.get("kind") == "CFG_COPY" and ctx.region_of[step["operation"].id] == ctx.region_of[operation.id]:
                            guard = ControlEdgeReference(step["operation"].id, step["successor_index"])
                            if guard not in guards:
                                guards.append(guard)
                for guard in guards:
                    binding = ctx.bind(source, actual, guard=guard)
                    conditional.append({"guard": guard, "binding": binding})
            copied = report["match"] is True or any(item["binding"]["match"] is True for item in conditional)
            possible_route = any(route.get("match") is True for route in report["routes"])
            possible = True if copied or possible_route else ctx.may_forwards(source, actual)
            if copied or possible is True:
                terminal = operation.terminator
                target = (HostTerminal(operation.id) if terminal is not None and terminal.kind
                          in (TerminatorKind.RETURN, TerminatorKind.RAISE) else actual)
                if target not in targets:
                    targets.append(target)
                support.append({"target": target, "use": actual, "binding": report,
                                "conditional": tuple(conditional), "possible_copy": possible})
                if not copied:
                    gaps.append({"code": "consumer_condition_correlation_unresolved", "target": target})
                if not operation.effects_complete:
                    gaps.append({"code": "consumer_escape_or_effects_unresolved", "target": target})
            elif report["match"] is None and ctx.region_of[operation.id] != ctx.region_of[source_operation]:
                # A possible cross-region route must not be silently erased.
                gaps.append({"code": "consumer_cross_region_inventory_unresolved", "use": actual})
        if ctx.budget.truncated:
            break
    if not targets:
        gaps.append("consumer_absence_is_not_dead_work")
    return {"site": site, "source": source, "targets": tuple(targets), "support": tuple(support),
            "gaps": tuple(gaps), "direct_terminal": False,
            "known_inventory_complete": not ctx.budget.truncated,
            "source_inventory_complete": False}


def _result_consumers(ctx, row, attachment):
    ledger = consumer_ledger(ctx, row)
    known = ledger["targets"]
    supplied = () if attachment is None else attachment.targets
    gaps, issues = list(ledger["gaps"]), []
    missing = tuple(target for target in known if target not in supplied)
    if ledger.get("direct_terminal"):
        missing = ()
    if attachment is None and not ledger.get("direct_terminal"):
        gaps.append("protocol_result_consumers_missing")
    for target in supplied:
        if type(target) is StaticProtocolGap:
            gaps.append(target)
        elif target not in known:
            # A legal FK without a proven connection cannot fill this ledger.
            try:
                if type(target) is OwnedUse:
                    report = ctx.bind(ledger["source"], target)
                    if report["match"] is False:
                        issues.append(("protocol_result_consumer", str(row.operation),
                                       "listed target does not consume this exact result"))
                    else:
                        gaps.append({"code": "consumer_target_role_unresolved", "target": target})
                else:
                    ctx.port(target)
                    matches = (tuple(ctx.bind(ledger["source"], actual)["match"]
                               for actual in ctx.terminal_uses(target))
                               if type(target) is HostTerminal else ())
                    if matches and all(match is False for match in matches):
                        issues.append(("protocol_result_consumer", str(row.operation),
                                       "listed host target does not consume this exact result"))
                    else:
                        gaps.append({"code": "consumer_target_route_unresolved", "target": target})
            except (KeyError, TypeError, ValueError) as exc:
                issues.append(("protocol_result_reference", str(row.operation), str(exc)))
    if missing:
        gaps.append({"code": "protocol_result_consumer_missing", "targets": missing})
    return {**ledger, "declared": supplied, "missing": missing, "issues": tuple(issues),
            "gaps": tuple(gaps), "status": "REJECTED" if issues else "PARTIAL" if gaps else "DECLARED"}


def common_connections(ctx):
    if "common" in ctx._connection_reports:
        return ctx._connection_reports["common"]
    from .protocol_await_records import (AwaitAcquisition, AwaitIteratorQualification,
                                         CoroutineAwaitWrapper)
    result_contracts = (AdmissionPredicate, RequestProcessing, ProtocolCall, OutcomeAdapter,
                        AwaitAcquisition, AwaitIteratorQualification, CoroutineAwaitWrapper)
    issues, gaps, inputs, consumers = [], [], [], []
    for site, attachment in (*ctx.call_inputs.items(), *ctx.result_consumers.items()):
        try:
            row = ctx.row(site)
            if site in ctx.call_inputs and type(row.contract) is not ProtocolCall:
                raise ValueError("call input attachment requires an existing ProtocolCall row")
            if site in ctx.result_consumers and type(row.contract) not in result_contracts:
                raise ValueError("result consumer attachment requires a row with its own exact result/output")
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(("protocol_connection_site", str(site), str(exc)))
    for row in ctx.graph.protocol_operations:
        if type(row.contract) is ProtocolCall:
            report = _call_inputs(ctx, row, ctx.call_inputs.get(_site(row)))
            inputs.append(report)
            issues.extend(report["issues"])
            gaps.extend(report["gaps"])
        if type(row.contract) in result_contracts:
            report = _result_consumers(ctx, row, ctx.result_consumers.get(_site(row)))
            consumers.append(report)
            issues.extend(report["issues"])
            gaps.extend(report["gaps"])
    result = {"issues": tuple(issues), "gaps": tuple(gaps), "inputs": tuple(inputs),
              "consumers": tuple(consumers), "obligations": (*inputs, *consumers),
              "cursor": {"phase": "connections"} if ctx.budget.truncated else None,
              "source_runtime_verified": False, "transformation_authorized": False}
    ctx._connection_reports["common"] = result
    return result


def connection_reports(ctx):
    """All fixed families share one immutable index, flow and accounting."""
    from .protocol_async import async_connections, request_state_errors
    from .protocol_continuations import continuation_connections
    from .protocol_await import await_connections
    from .protocol_connections import RequestStateCondition, SessionOccupancyCondition
    from .protocols import (AdmissionPredicate, BodyStateCondition, ProtocolProfile,
                            ReferenceIsNoneCondition)
    reports = {"common": common_connections(ctx),
               "continuations": continuation_connections(ctx),
               "async": async_connections(ctx),
               "await": await_connections(ctx)}
    if ctx.graph.exception_matches:
        from .exception_match_query import exception_match_reports
        reports['exception_matches'] = exception_match_reports(ctx)
    predicate_issues = []
    for row in ctx.graph.protocol_operations:
        if type(row.contract) is AdmissionPredicate and type(row.contract.subject) is RequestStateCondition:
            predicate_issues.extend(request_state_errors(ctx, row))
        elif type(row.contract) is AdmissionPredicate and type(row.contract.subject) is SessionOccupancyCondition:
            from scar.ir.program.session_flow import occupancy_errors
            predicate_issues.extend(occupancy_errors(ctx, row))
        elif (type(row.contract) is AdmissionPredicate
              and type(row.contract.subject) in (BodyStateCondition, ReferenceIsNoneCondition)
              and ctx.scopes[row.scope].profile is ProtocolProfile.ASYNC_GENERATOR_BODY_V1):
            from .async_initial import async_predicate_errors
            predicate_issues.extend(async_predicate_errors(ctx, row))
    return {"families": reports,
            "issues": (*predicate_issues, *(issue for report in reports.values() for issue in report["issues"])),
            "gaps": tuple(gap for report in reports.values() for gap in report["gaps"]),
            "cursor": tuple(report["cursor"] for report in reports.values() if report["cursor"]),
            "source_runtime_verified": False, "transformation_authorized": False}


# Shared original-port queries for the Program19 connection obligations.
#
# The context is prepared only after canonical validation. Expected ports are
# independent declarations; resolving the actual producer never selects one.
# No result here grants runtime identity, source applicability or a rewrite.
def _answer(match=None, *, routes=(), gaps=(), cursor=None, conditions=()):
    return {"match": match, "routes": tuple(routes), "gaps": tuple(gaps),
            "cursor": cursor, "conditions": (
                "supplied-model ports only; source/native applicability unproved",
                "dynamic visit and value identity unproved", *conditions),
            "source_runtime_verified": False, "runtime_identity_verified": False,
            "transformation_authorized": False}


class ConnectionContext(ProtocolFlow):
    """One validated-core index, original flow and shared query budget."""

    def __init__(self, graph, *, index, budget=None):
        from .queries import ProgramIndex
        if type(index) is not ProgramIndex or index.graph is not graph:
            raise TypeError("connections require the exact prepared ProgramIndex")
        accounting = ValidityBudget(1000000) if budget is None else budget
        if type(accounting) is not ValidityBudget:
            raise TypeError("connections require one exact shared ValidityBudget")
        super().__init__(graph, budget=accounting, core=index._core, driver_facts=index._driver_facts)
        self.index = index
        self.call_inputs = {row.site: row for row in graph.protocol_call_inputs}
        self.result_consumers = {row.site: row for row in graph.protocol_result_consumers}
        self._connection_origins = {}
        self._connection_bindings = {}
        self._connection_reports = {}
        self._connection_guard_entries = {}
        self._connection_depth = 0
        from scar.ir.program.call_returns import CallReturns
        self._returns = CallReturns(self)
        self.preparation["prepared_program_index_reused"] = True

    def _copy_context_key(self):
        if getattr(self, '_copy_consumer_bounds', False):
            return self._returns.guards, 'copy-consumer-incomplete-bounds'
        return self._returns.guards

    def _copy_call_transfer(self, operation, current, *, mode):
        return self._returns.transfer(operation, current, mode=mode)

    def _fresh_copies(self, port, *, guard=None, mode="must"):
        ambiguity = self._guard_entry_ambiguity(port, guard)
        if ambiguity is not None:
            self._returns.gap(ambiguity)
            return None
        return self._returns.flow(port, guard=guard, mode=mode)

    def _guard_entry_ambiguity(self, port, guard):
        """A selected later incoming cannot erase a loop's initial entry.

        Diagnose the loss using the original CFG and outgoing choices. Never
        feed a relaxed graph to the copy solver or grant a new copy True.
        Acyclic exclusions and explicit own-completion conflicts stay False.
        """
        guards = self.selected_edges(guard)
        if not guards or self.selection_ambiguity(guards) is not None:
            return None
        if port.phase == 'edge' and any(edge.operation == port.operation
                and edge.successor_index != port.edge for edge in guards):
            return None
        region = self.region_of[port.operation]
        guards = self.selected_edges(guards, region=region)
        if not guards:
            return None
        key = region, guards
        if key not in self._connection_guard_entries:
            declaration, outgoing, _, _, _ = self._edges(region)
            choices = {edge.operation: edge.successor_index for edge in guards}
            targets = {self.operations[edge.operation].terminator.successors[
                edge.successor_index].block: (edge.operation, edge.successor_index) for edge in guards}

            def reachable(start, *, released=frozenset(), stop=None, outgoing_choices=True):
                seen, pending = set(), [start]
                while pending:
                    if not self.spend('connection_guard_entry_blocks'):
                        return None
                    block = pending.pop()
                    if block in seen:
                        continue
                    seen.add(block)
                    if block == stop:
                        return seen
                    for edge in outgoing[block]:
                        if not self.spend('connection_guard_entry_edges'):
                            return None
                        if outgoing_choices and edge[0] in choices and choices[edge[0]] != edge[1]:
                            continue
                        if (edge[2] not in released and edge[2] in targets
                                and targets[edge[2]] != edge[:2]):
                            continue
                        pending.append(edge[2])
                return seen

            strict = reachable(declaration.entry)
            if strict is None:
                return 'protocol_connection_budget_exhausted'
            cyclic = set()
            for target, (operation, _) in targets.items():
                parent = self.block_of[operation]
                if target in strict or parent in strict:
                    continue
                # The incoming can belong to an earlier visit, before this
                # source's own completion choices. This is only a cycle
                # witness; the actual source reachability keeps all outgoing
                # choices and every unrelated incoming constraint.
                cycle = reachable(target, released={target}, stop=parent, outgoing_choices=False)
                if cycle is None:
                    return 'protocol_connection_budget_exhausted'
                if parent in cycle:
                    cyclic.add(target)
            possible = reachable(declaration.entry, released=cyclic) if cyclic else strict
            if possible is None:
                return 'protocol_connection_budget_exhausted'
            self._connection_guard_entries[key] = frozenset(strict), frozenset(possible)
        strict, possible = self._connection_guard_entries[key]
        block = self.block_of[port.operation]
        if block in possible and block not in strict:
            return 'selected_loop_entry_requires_occurrence_context'
        return None

    def _copy_cache_begin(self, key):
        if getattr(self, '_copy_consumer_bounds', False):
            if not hasattr(self, '_copy_consumer_bound_flows'):
                self._copy_consumer_bound_flows = {}
            return len(self._copy_consumer_bound_events)
        return None

    def _copy_cache_replay(self, key):
        if getattr(self, '_copy_consumer_bounds', False):
            gaps = self._copy_consumer_bound_flows[key]
            self._copy_consumer_bound_events.extend(gaps)
            for _, gap in gaps:
                self._returns.gap(gap)

    def _copy_cache_commit(self, key, token):
        if token is not None:
            # Applies to solver subjobs too, not just the public flow wrapper.
            self._copy_consumer_bound_flows[key] = tuple(
                dict.fromkeys(self._copy_consumer_bound_events[token:]))

    def row(self, site):
        from .protocol_connections import ProtocolOperationRef
        if type(site) is not ProtocolOperationRef:
            raise TypeError("connection site requires exact ProtocolOperationRef")
        site.__post_init__()
        row = self.rows.get((site.scope, site.clause, site.operation, site.slot))
        if row is None:
            raise ProtocolReferenceError("connection site must select an exact existing protocol row")
        return row

    def port(self, source, *, referring=None):
        if type(source) is not BlockArgument:
            return super().port(source, referring=referring)
        source.__post_init__()
        if not self.spend("ports"):
            return None
        if source not in self._ports:
            block = self.blocks.get(source.block)
            if block is None or source.index >= len(block.arguments):
                raise ProtocolReferenceError("expected formal is outside its original block")
            value = block.arguments[source.index]
            if self.values[value].origin != source:
                raise ProtocolReferenceError("expected formal must retain its original producer")
            self._ports[source] = ProtocolPort(source, block.operations[0], (value,), "before")
        return self._ports[source]

    def origins(self, actual, *, guard=None):
        """Original input recipes; caches are scoped to this shared budget."""
        from scar.ir.program.request_inputs import _input_origins_with_flow
        self.owned_use(actual.operation, actual.use)
        key = actual, guard
        if key not in self._connection_origins:
            self._connection_origins[key] = _input_origins_with_flow(
                self.index, actual.operation, actual.use,
                accounting=self.budget, _copy_flow=self, _guard=guard)
        return self._connection_origins[key]

    def carried_source(self, expected, carrier, actual, *, guard=None):
        """Keep a declared historical reference separate from currentness.

        A current-copy False is still retained. An independently named formal
        and an exact original historical recipe may make the source possible,
        but cannot choose a dynamic evaluation or certify it True. State and
        fresh method-lookup obligations keep their ordinary binding rules.
        """
        current = self.bind(expected, actual, guard=guard)
        if current['match'] is not False or type(carrier) is not BlockArgument:
            return current
        port = expected.result_port if type(expected) is ContractResult else expected
        if type(port) not in (OperationResult, InvokeLiteralResult):
            return current

        def unfinished(phase, copied=None):
            return {**_answer(gaps=('carried_source_origin_inventory_unresolved',),
                cursor={'phase': phase}), 'current_binding': current,
                'carrier_binding': copied, 'historical_origins': None,
                'matching_historical_recipes': ()}

        formal, source = self.port(carrier), self.port(expected)
        if (formal is None or source is None or formal.gap is not None
                or source.gap is not None or not formal.values or not source.values
                or any(not isinstance(self.values[value], ReferenceValue)
                       for value in (*formal.values, *source.values))):
            return unfinished('carried_source_ports') if self.budget.truncated else current
        copied = self.bind(carrier, actual, guard=guard)
        if copied['match'] is not True:
            return unfinished('carried_source_carrier', copied) if self.budget.truncated else current
        history = self.origins(actual, guard=guard)
        matching = []
        for recipe in history['recipes']:
            if not self.spend('connection_carried_source_recipes'):
                break
            if (recipe['origin'] != port or recipe['kind'] != (
                    'INVOKE_COMPLETION' if type(port) is InvokeLiteralResult else
                    'OPERATION_RESULT' if type(port) is OperationResult else None)):
                continue
            for step in recipe['path']:
                if not self.spend('connection_carried_source_steps'):
                    break
                if (step.get('kind') == 'CFG_COPY'
                        and getattr(step.get('formal'), 'id', step.get('formal')) in formal.values):
                    matching.append(recipe)
                    break
        incomplete = self.budget.truncated or not history['source_inventory_complete']
        if not matching and not incomplete:
            return current
        answer = _answer(gaps=('carried_source_evaluation_unselected',
            *(('carried_source_origin_inventory_unresolved',) if incomplete else ())),
            cursor=history['cursor'] or ({'phase': 'carried_source'} if self.budget.truncated else None))
        return {**answer, 'current_binding': current, 'carrier_binding': copied,
                'historical_origins': history, 'matching_historical_recipes': tuple(matching)}

    def request_binding(self, raw, input, ordinal=0, *, guard=None):
        from scar.ir.program.request_inputs import request_input_binding
        return request_input_binding(self.index, raw, input, ordinal,
                                     accounting=self.budget, _flow=self, _guard=guard)

    def recipe_context(self, recipe):
        """Project original caller/CFG context without adding a copy relation."""
        from scar.ir.program.request_inputs import _context
        calls, choices, rooted = _context(self.index, recipe, self.budget)
        complete = recipe.get('origin_inventory_complete', True)
        return {"match": True if rooted and complete and not self.budget.truncated else None,
                "context": calls, "choices": choices, "recipe": recipe,
                "rooted": rooted, "runtime_identity_verified": False,
                "gaps": (("source_origin_inventory_unresolved",) if not complete else
                         () if rooted else ("static_caller_or_cfg_visit_context_unresolved",))}

    def correlate(self, *reports):
        """Join declared semantic projections on the same static context.

        Family resolvers provide independently established port/projection
        reports. This method checks context coherence, never invents a value
        equality or turns a semantic projection into an ordinary copy edge.
        """
        from .protocol_connection_validation import _joint_routes
        joined, gaps = _joint_routes(self, reports)
        complete = bool(reports) and all(report["match"] is True for report in reports)
        covered = {id(route) for row in joined for route in row["support"]}
        all_routes = tuple(route for report in reports for route in report["routes"]
                           if route.get("match") is True)
        full_coverage = all(id(route) in covered for route in all_routes)
        match = (False if any(report["match"] is False for report in reports) else
                 True if joined and complete and full_coverage else
                 False if not joined and complete and not self.budget.truncated else None)
        if not full_coverage:
            gaps += ("whole_input_tuple_context_coverage_unresolved",)
        return _answer(match, routes=joined, gaps=gaps,
                       cursor={"phase": "context_correlation"} if self.budget.truncated else None,
                       conditions=("context coherence does not establish projection semantics",))

    def extend_context(self, recipe, anchor, consumer_recipe):
        """Extend a source recipe along an independently selected caller path.

        Used for a declared request's latent/body projections. The anchor is
        an actual original Use; no RAW formal or semantic-copy fact is added.
        """
        from scar.ir.program.request_inputs import _context, _helper_context_extension
        if type(anchor) is not OwnedUse:
            raise TypeError("context extension requires an exact original Use anchor")
        self.owned_use(anchor.operation, anchor.use)
        source_calls, source_choices, source_rooted = _context(self.index, recipe, self.budget)
        target_calls, target_choices, target_rooted = _context(self.index, consumer_recipe, self.budget)
        choices = {**source_choices, **target_choices}
        match, extension, gaps = None, None, ()
        complete = (recipe.get('origin_inventory_complete', True)
                    and consumer_recipe.get('origin_inventory_complete', True))
        if not complete:
            gaps = ('source_origin_inventory_unresolved',)
        elif any(key in source_choices and source_choices[key] != value for key, value in target_choices.items()):
            match = False
        elif source_calls == target_calls and source_rooted and target_rooted:
            match = True
        elif source_rooted and target_rooted:
            extension = _helper_context_extension(self.index, self, anchor,
                source_calls, target_calls, self.budget)
            match = True if type(extension) is dict else extension
        if match is None and not gaps:
            gaps = ("projection_caller_prefix_unresolved",)
        return {"match": match, "context": target_calls, "choices": choices,
                "recipe": recipe, "consumer_recipe": consumer_recipe,
                "anchor": anchor, "extension": extension, "gaps": gaps,
                "runtime_identity_verified": False, "transformation_authorized": False}

    def bind(self, expected, actual, *, guard=None, choices=(), path=""):
        """Keep complete guards scoped to this query, including nested helpers."""
        if type(choices) is not tuple:
            raise TypeError("additional original choices require an exact tuple")
        guards = self.selected_edges((*self.selected_edges(guard), *choices))
        previous = self._returns.guards
        if self._connection_depth == 0:
            self._returns.gaps = []
            self._returns.gap_events = []
        self._connection_depth += 1
        self._returns.guards = guards
        try:
            return self._bind(expected, actual, guard=guards or None, path=path)
        finally:
            self._returns.guards = previous
            self._connection_depth -= 1

    def _bind(self, expected, actual, *, guard=None, choices=(), path=""):
        """Tri-state exact copy/source check, preserving every known route.

        ``True`` is an original-port relation, not equality of dynamic objects.
        Mixed known routes stay partial, never an independent ANY success.
        """
        from .protocol_connections import RawOperandSource
        if type(choices) is not tuple:
            raise TypeError("additional original choices require an exact tuple")
        guard = self.selected_edges((*self.selected_edges(guard), *choices)) or None
        ambiguity = self.selection_ambiguity(guard)
        if ambiguity is not None:
            return _answer(gaps=(ambiguity["code"],))
        if type(actual) is not OwnedUse:
            raise TypeError("actual connection must be exact OwnedUse")
        destination = self.owned_use(actual.operation, actual.use)
        key = expected, actual, guard
        if key in self._connection_bindings:
            return self._connection_bindings[key]
        if not self.spend("connection_bindings"):
            return _answer(gaps=("protocol_connection_budget_exhausted",),
                           cursor={"path": path, "expected": expected, "actual": actual})
        if type(expected) is RawOperandSource:
            expected.__post_init__()
            from .protocol_validation import raw_definition
            definition = raw_definition(self, expected.raw)
            origin = None if definition is None else self.values[definition.value].origin
            if type(origin) is OperationResult and self.region_of[origin.operation] == self.region_of[actual.operation]:
                if expected.ordinal >= len(definition.operands):
                    answer = _answer(False, gaps=("request_ordinal_absent",))
                else:
                    operand = OwnedUse(origin.operation, definition.operands[expected.ordinal])
                    copied = self.bind(operand, actual, guard=guard, path=path)
                    current = self.available(expected.raw, actual.operation, guard=guard, path=path)
                    checks = (copied["match"], current["match"])
                    match = False if False in checks else None if None in checks else True
                    answer = _answer(match, routes=copied["routes"],
                        gaps=(*copied["gaps"], *current["gaps"]),
                        cursor=copied["cursor"] or current["cursor"])
            else:
                report = self.request_binding(expected.raw, actual, expected.ordinal, guard=guard)
                routes = tuple({"request_binding": row, "match": True,
                                "context": row["caller_context"],
                                "choices": dict(row["cfg_choices"])}
                               for row in report["bindings"])
                covered = {id(row["input_recipe"]) for row in report["bindings"]}
                complete = (report["cursor"] is None and report['source_inventory_complete']
                            and report["input"]["recipes"]
                            and all(id(recipe) in covered for recipe in report["input"]["recipes"])
                            and all(recipe["freshness"] is True or recipe["freshness_target"] is None
                                for row in report["bindings"] for recipe in
                                (row["raw_recipe"], row["operand_recipe"], row["input_recipe"])))
                answer = _answer(True if routes and complete else None, routes=routes,
                                 gaps=report["gaps"], cursor=report["cursor"])
        else:
            port = self.port(expected)
            if port is None or port.gap is not None or port.operation is None:
                answer = _answer(gaps=(("expected_source_unresolved" if port is None
                                        else port.gap),))
            elif self.region_of[port.operation] == self.region_of[actual.operation]:
                future = (self.block_of[port.operation] == self.block_of[actual.operation]
                    and (self.positions[port.operation] > self.positions[actual.operation]
                         or (port.operation == actual.operation and port.phase != "before")))
                relevant = self.selected_edges(guard, region=self.region_of[port.operation]) or None
                before = {} if future else self._fresh_copies(port, guard=relevant)
                match = None if before is None else destination.value in before.get(actual.operation, ())
                answer = _answer(match, routes=({"source": expected, "actual": actual,
                    "guard": guard, "context": None, "choices": {}, "match": match},),
                    gaps=(("protocol_connection_budget_exhausted",) if self.budget.truncated else
                          tuple(self._returns.gaps) or ("call_return_flow_unresolved",))
                         if before is None else (),
                    cursor={"phase": "connection_copies", "actual": actual}
                           if self.budget.truncated else None)
            else:
                answer = self._bind_cross_region(expected, port, actual, guard=guard)
        self._connection_bindings[key] = answer
        return answer

    def available(self, expected, consumer, *, guard=None, path=""):
        """Source evaluation is available before an operation, without a fake operand.

        This is used for zero-argument request triggers. Cross-region support
        comes from real caller/actual/formal recipes of this consumer's inputs.
        Availability alone does not prove that a native call uses the request.
        """
        from scar.ir.program.request_inputs import _context
        operation = self.operations.get(consumer)
        if operation is None:
            raise ProtocolReferenceError("availability consumer must be an original operation")
        ambiguity = self.selection_ambiguity(guard)
        if ambiguity is not None:
            return _answer(gaps=(ambiguity["code"],))
        port = self.port(expected)
        if port is None or port.operation is None or port.gap is not None:
            return _answer(gaps=("source_availability_unresolved",))
        source_region, target_region = self.region_of[port.operation], self.region_of[consumer]
        if source_region == target_region:
            future = (self.block_of[port.operation] == self.block_of[consumer]
                and (self.positions[port.operation] > self.positions[consumer]
                     or port.operation == consumer and port.phase != "before"))
            relevant = self.selected_edges(guard, region=source_region) or None
            flow = {} if future else self._fresh_copies(port, guard=relevant)
            match = None if flow is None else bool(flow.get(consumer, ()))
            return _answer(match, routes=({"source": expected, "consumer": consumer,
                "guard": guard, "context": None, "choices": {}, "match": match},),
                gaps=("source_availability_unresolved",) if match is None else (),
                conditions=("availability does not prove native request selection",))
        routes, gaps, cursor = [], [], None
        inventory_complete = True
        arguments = (operation.invoke_arguments if operation.terminator is not None
                     and operation.terminator.kind is TerminatorKind.INVOKE else
                     operation.call_arguments if operation.callee is not None else
                     tuple(use.id for use in operation.uses))
        for identity in arguments:
            value = self.values[self.owned_use(consumer, identity).value]
            if isinstance(value, StateVersionValue):
                continue
            report = self.origins(OwnedUse(consumer, identity), guard=guard)
            inventory_complete &= report['source_inventory_complete']
            if not report['source_inventory_complete']:
                gaps.extend(report['missing_origin_routes'])
                gaps.append('source_availability_origin_inventory_unresolved')
            for recipe in report["recipes"]:
                if not self.spend("connection_available_routes"):
                    cursor = {"phase": "source_availability", "consumer": consumer, "path": path}
                    break
                calls, choices, _ = _context(self.index, recipe, self.budget)
                first = next((caller for caller, region, _ in calls if region == source_region), None)
                if first is None:
                    continue
                relevant = self.selected_edges(guard, region=source_region) or None
                flow = self._fresh_copies(port, guard=relevant)
                match = None if flow is None else bool(flow.get(first, ()))
                route = {"source": expected, "consumer": consumer, "context": calls,
                         "choices": choices, "match": match, "entry_call": first,
                         "recipe": recipe, "source_current_before_call": match}
                if not any(old["context"] == calls and old["choices"] == choices for old in routes):
                    routes.append(route)
            if report["cursor"] is not None:
                cursor = report["cursor"]
            if self.budget.truncated:
                break
        matches = tuple(route["match"] for route in routes)
        if matches and all(match is True for match in matches) and cursor is None and inventory_complete:
            match = True
        elif matches and all(match is False for match in matches) and cursor is None and inventory_complete:
            match = False
        else:
            match = None
            gaps.append("cross_region_source_availability_unresolved")
        return _answer(match, routes=routes, gaps=gaps, cursor=cursor,
                       conditions=("availability does not prove native request selection",))

    def _bind_cross_region(self, expected, port, actual, *, guard):
        """Follow original call/formal paths; never equate local reachability with global reachability."""
        from scar.ir.program.request_inputs import _context
        from scar.ir.program.call_composition import compose_route
        export = self._returns.export_applicable(self.region_of[port.operation],
                                                self.region_of[actual.operation])
        if export is None:
            return _answer(gaps=("call_return_budget_exhausted",),
                           cursor={"phase": "call_return_ancestors", "actual": actual})
        if export:
            return self._returns.export(expected, port, actual, guard=guard)
        report = self.origins(actual, guard=guard)
        routes, gaps = [], list(report["gaps"])
        for recipe in report["recipes"]:
            if not self.spend("connection_source_routes"):
                return _answer(routes=routes, gaps=(*gaps, "protocol_connection_budget_exhausted"),
                               cursor={"phase": "connection_source_routes", "actual": actual})
            calls, choices, rooted = _context(self.index, recipe, self.budget)
            for selected in self.selected_edges(guard):
                applicable = [number for (_, operation), number in choices.items()
                              if operation == selected.operation]
                if applicable and selected.successor_index not in applicable:
                    break
            else:
                applicable = None
            if applicable is not None:
                continue
            formal_steps = tuple(step for step in recipe["path"]
                if step.get("kind") in ("CALL_ACTUAL_TO_FORMAL", "CFG_COPY")
                and getattr(step.get("formal"), "id", step.get("formal")) in port.values)
            composition = compose_route(self, expected, port, actual, recipe,
                guard=guard, calls=calls, choices=choices, rooted=rooted)
            match = composition['match']
            gaps.extend(composition['gaps'])
            routes.append({"source": expected, "actual": actual, "recipe": recipe,
                           "context": calls, "choices": choices, "match": match,
                           "formal_support": formal_steps, "composition": composition})
        matches = tuple(route["match"] for route in routes)
        complete = report['source_inventory_complete']
        if not complete:
            gaps.append('call_composition_origin_inventory_unresolved')
        if matches and all(item is True for item in matches) and report["cursor"] is None and complete:
            match = True
        elif matches and all(item is False for item in matches) and report["cursor"] is None and complete:
            match = False
        else:
            match = None
            gaps.append("cross_region_source_routes_unresolved")
        cursor = report['cursor']
        if self.budget.truncated:
            match = None
            cursor = cursor or {'phase': 'call_composition', 'actual': actual}
        return {**_answer(match, routes=routes, gaps=gaps, cursor=cursor),
                'source_inventory_complete': complete, 'missing_origin_routes': report['missing_origin_routes']}

    def current(self, site, consumer, *, after=False, path=""):
        """Check the original state version at this consumer operation.

        The consumer may be a reference Use; its operation must still expose
        the corresponding canonical state input. No empty-support success.
        """
        self.owned_use(consumer.operation, consumer.use)
        operation = self.operations.get(site.operation)
        if operation is None:
            raise ProtocolReferenceError("state site operation does not exist")
        if type(site) is ExistingAccess:
            if site.index >= len(operation.accesses):
                raise ProtocolReferenceError("state site access does not exist")
            access = operation.accesses[site.index]
            before = self.owned_use(operation.id, access.before)
            location = access.location
            if after:
                if access.after is None:
                    return _answer(gaps=("state_site_has_no_after",))
                source = self.values[access.after].origin
            else:
                source = OwnedUse(operation.id, before.id)
        elif type(site) is InvokeStateOperand:
            before = self.owned_use(operation.id, site.state_use)
            if site.state_use not in operation.invoke_arguments:
                raise ProtocolReferenceError("state site must select actual invoke input")
            value = self.values[before.value]
            if not isinstance(value, StateVersionValue):
                raise ProtocolReferenceError("state site must carry canonical state")
            location, source = value.location, OwnedUse(operation.id, before.id)
            if after:
                return _answer(gaps=("invoke_after_requires_exact_completion_state_port",))
        else:
            raise TypeError("current state requires an exact state address site")
        target = self.operations[consumer.operation]
        uses = target.invoke_arguments if (target.terminator is not None and
            target.terminator.kind is TerminatorKind.INVOKE) else tuple(a.before for a in target.accesses)
        targets = tuple(OwnedUse(target.id, identity) for identity in uses
                       if isinstance(self.values[self.owned_use(target.id, identity).value], StateVersionValue)
                       and self.values[self.owned_use(target.id, identity).value].location == location)
        if not targets:
            return _answer(gaps=("consumer_current_state_input_unresolved",))
        reports = tuple(self.bind(source, actual, path=path) for actual in targets)
        matches = tuple(row["match"] for row in reports)
        match = False if False in matches else None if None in matches else True
        return _answer(match, routes=tuple(route for row in reports for route in row["routes"]),
                       gaps=tuple(gap for row in reports for gap in row["gaps"]))


__all__ = [
    'common_connections',
    'consumer_ledger',
    'connection_reports',
    'ConnectionContext',
]
