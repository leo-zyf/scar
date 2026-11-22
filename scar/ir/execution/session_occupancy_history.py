"""Session occupancy history."""

from scar.ir.program.model import (
    CompletionResult,
    ExitKind,
    InvokeLiteralResult,
    AccessMode,
    OperationResult,
    StateVersionValue,
    TerminatorKind,
)
from scar.ir.program.protocol_connections import (
    RequestAssociation,
    RequestStateCondition,
    RequestStateRule,
)
from scar.ir.program.protocols import (
    DriverSubject,
    AdmissionPredicate,
    BodyStateCondition,
    BodyStatePredicateRule,
)
from scar.ir.program.structured import ControlEdgeReference
from scar.ir.execution.await_outcomes import pair
from scar.ir.execution.await_links import _result
from scar.ir.execution.common import Unresolved
from scar.ir.execution.coordinates import OccurrencePhase, OccurrencePoint
from scar.ir.execution.order_validation import host_path
from scar.ir.execution.resumable_flow import value_for_use
from scar.ir.program.queries import ProgramIndex
from scar.ir.execution.state_preservation import state_preservation
from scar.ir.program.lifecycle import LifecyclePostBinding, SessionPost


# Conditional post-to-read receipts over original captured program ports.
#
# An established post, reference copy, state route and content preservation are
# independent relations. Nothing here creates a state, a field address, a native
# phase observation or permission to replace execution.
def _flags():
    return dict(source_runtime_verified=False, runtime_identity_verified=False,
                value_identity_verified=False, transformation_authorized=False)


def _empty_preservation():
    return {'preserved': None, 'reached': None, 'operations': (), 'calls': (),
            'transfers': (), 'state_bindings': (), 'blockers': (), 'gaps': (), 'cursor': None}


def _unknown_pair():
    return {'match': None, 'support': (), 'gaps': (), **_flags()}


class _History:
    def __init__(self, ctx, shared):
        self.ctx, self.shared = ctx, shared
        self.gaps, self.cursor = [], None
        self.sites = getattr(ctx, 'session_occurrences_by_visit_site', None)
        if self.sites is None:
            self.sites = {(row.visit, row.operation): row for row in ctx.occurrences.values()}

    def gap(self, code):
        if code not in self.gaps:
            self.gaps.append(code)

    def spend(self, kind, path):
        if self.shared.truncated or not self.shared.spend('state_history_' + kind, path=str(path)):
            self.gap('state_history_budget_exhausted')
            if self.cursor is None:
                self.cursor = {'phase': kind, 'path': str(path)}
            return False
        return True

    def use_value(self, occurrence, use):
        return value_for_use(self.ctx, self.ctx.uses.get((occurrence.id, use)))

    def observed_branch(self, read, row, boolean):
        """Follow this READ evaluation, never a later visit of the same site."""
        ctx = self.ctx
        visit = ctx.visits[read.visit]
        position = ctx.blocks[visit.block].operations.index(read.operation) + 1
        segment, seen = visit.segment, set()
        while visit.id not in seen:
            if not self.spend('branch_visits', visit.id):
                return None
            seen.add(visit.id)
            block = ctx.blocks[visit.block]
            for site in block.operations[position:]:
                if not self.spend('branch_operations', site):
                    return None
                occurrence = self.sites.get((visit.id, site))
                if occurrence is None:
                    self.gap('state_history_branch_occurrence_missing')
                    return None
                if site == read.operation:
                    self.gap('state_history_read_reevaluated_before_branch')
                    return None
                if site == row.contract.branch:
                    operation = ctx.core_operations[site]
                    condition = self.use_value(occurrence, operation.terminator.condition)
                    binding = pair(ctx, boolean, condition, self.shared)
                    transfer = ctx.outgoing.get(occurrence.id)
                    if binding['match'] is not True or transfer is None or self.shared.truncated:
                        self.gap('state_history_branch_binding_unproved')
                        return None
                    return transfer.successor_index == 0
            terminal = self.sites.get((visit.id, block.operations[-1]))
            operation = ctx.core_operations[block.operations[-1]]
            if operation.terminator.kind in (TerminatorKind.RETURN, TerminatorKind.RAISE,
                                             TerminatorKind.SUSPEND, TerminatorKind.BODY_DRIVE):
                return None
            transfer = None if terminal is None else ctx.outgoing.get(terminal.id)
            if transfer is None:
                self.gap('state_history_branch_transfer_missing')
                return None
            following = ctx.visits[transfer.target]
            if following.segment != segment:
                self.gap('state_history_branch_segment_unproved')
                return None
            visit, position = following, 0
        self.gap('state_history_branch_route_unproved')
        return None


def request_state_history(ctx, read_id, *, shared):
    return _post_history(ctx, read_id, shared=shared)


def _post_history(ctx, read_id, *, shared, occupancy=False, bind_subject=None):
    """One original post/port/preservation engine, with closed implications."""
    from scar.ir.program.protocol_connections import SessionOccupancyCondition
    subject_type = SessionOccupancyCondition if occupancy else RequestStateCondition
    read_role = 'body_use' if occupancy else 'request_use'
    analysis = _History(ctx, shared)
    read = ctx.occurrences[read_id]
    operation = ctx.core_operations[read.operation]
    declarations = tuple(row for row in ctx.program.protocol_operations
        if row.operation == read.operation and type(row.contract) is AdmissionPredicate
        and type(row.contract.subject) is subject_type)
    inventory = tuple(ctx.program.session_updates)
    occurrences = {}
    for occurrence in ctx.occurrences.values():
        occurrences.setdefault(occurrence.operation, []).append(occurrence)
    # Original declarations and actual occurrences survive even budget=1.
    candidates = [dict(update=update, source=source, post_established=None,
        subject_binding=_unknown_pair(), locations=(), support=(), source_point=None,
        state_input_inventory=(), state_inputs=(),
        preservation=_empty_preservation(), expected_bool=None, gaps=(), status='NOT_CHECKED')
        for update in inventory for source in occurrences.get(update.operation, ())]
    missing = tuple(update for update in inventory if not occurrences.get(update.operation))
    if missing:
        analysis.gap('state_history_post_sources_unobserved')
    if not inventory:
        analysis.gap('state_history_post_source_missing')
    publication_index = getattr(ctx, 'body_program_index', None)
    prepared = publication_index is not None
    if publication_index is None and inventory:
        publication_index = ProgramIndex(ctx.program)
    predicate = declarations[0] if len(declarations) == 1 else None
    # Dependency inventory is original input, not a successful proof prefix.
    read_locations = {access.location for access in operation.accesses}
    read_supported, read_subject, boolean = False, None, None
    observed = None
    if analysis.spend('read', read.id) and predicate is not None:
        read_supported = operation.effects_complete
        if not operation.effects_complete:
            analysis.gap('state_history_read_effects_incomplete')
        read_subject = analysis.use_value(read, getattr(predicate.contract.subject, read_role))
        current = ctx.current(OccurrencePoint(read.id, OccurrencePhase.BEFORE)) or {}
        for access in operation.accesses:
            if not analysis.spend('read_accesses', read.id):
                read_supported = False
                break
            value = analysis.use_value(read, access.before)
            if (access.mode is not AccessMode.READ or value is None or type(value) is Unresolved
                    or current.get(access.location) != value.id):
                read_supported = False
                analysis.gap('state_history_read_state_unproved')
        result = predicate.contract.result
        if (type(result) is OperationResult and result.operation == read.operation
                and result.index < len(operation.results)):
            boolean = ctx.results.get((read.id, operation.results[result.index]))
        if boolean is None:
            read_supported = False
            analysis.gap('state_history_read_bool_unobserved')
        if not read_locations:
            read_supported = False
            analysis.gap('state_history_read_dependencies_unresolved')
        observed = analysis.observed_branch(read, predicate, boolean)
    else:
        analysis.gap('state_history_predicate_unresolved')
    pending = []
    for ordinal, candidate in enumerate(candidates):
        source, update = candidate['source'], candidate['update']
        if not analysis.spend('sources', source.id):
            pending.extend(range(ordinal, len(candidates)))
            break
        gaps = []
        try:
            publication = publication_index.session_publication(update.operation)
        except (KeyError, ValueError, StopIteration, AttributeError):
            publication = None
        support = () if publication is None else tuple(publication.support)
        if not support:
            gaps.append('state_history_post_support_unresolved')
        locations = tuple(sorted({value.location for value in support} | read_locations,
                                 key=lambda value: value.wire))
        candidate.update(support=support, locations=locations)
        input_inventory = () if publication is None else publication.state_inputs
        candidate['state_input_inventory'] = input_inventory
        transfer = ctx.outgoing.get(source.id)
        source_operation = ctx.core_operations[source.operation]
        if update.binding is LifecyclePostBinding.AFTER_OPERATION:
            point, established = OccurrencePoint(source.id, OccurrencePhase.AFTER), True
        elif transfer is None:
            point, established = None, None
            gaps.append('state_history_post_transfer_missing')
        elif source_operation.terminator.successors[transfer.successor_index].kind is not ExitKind.NORMAL:
            point, established = None, False
            gaps.append('state_history_post_normal_not_selected')
        else:
            point = OccurrencePoint(source.id, OccurrencePhase.EDGE, transfer.successor_index)
            established = True
        states = {} if point is None else ctx.current(point) or {}
        source_request = analysis.use_value(source, update.request)
        source_body = analysis.use_value(source, update.body)
        if not support or any(states.get(value.location) is None for value in support):
            established = None if established is not False else False
            gaps.append('state_history_post_state_unobserved')
        if any(value is None or type(value) is Unresolved for value in (source_request, source_body)):
            established = None if established is not False else False
            gaps.append('state_history_post_subject_unobserved')
        # A captured output cannot fill an absent original input.  The generic
        # interval intentionally starts after the post, so this source boundary
        # must be checked here, using the original publication's entire inputs.
        before = ctx.before.get(source.id, {})
        input_bindings = []
        for site in input_inventory:
            if not analysis.spend('post_input_states', source.id):
                established = None if established is not False else False
                break
            definition = ctx.core_values[site.use.value]
            actual = ctx.uses.get((source.id, site.use.id))
            value = value_for_use(ctx, actual)
            current = before.get(definition.location)
            input_bindings.append({'definition': site, 'use': actual, 'value': value,
                'location': definition.location, 'current': current})
            if (value is None or type(value) is Unresolved or current != value.id):
                established = None if established is not False else False
                gaps.append('state_history_post_input_state_unobserved')
        candidate['state_inputs'] = tuple(input_bindings)
        for definition in support:
            if not analysis.spend('post_states', source.id):
                established = None if established is not False else False
                break
            value = ctx.values.get(states.get(definition.location))
            if value is None or value.value != definition.id:
                established = None if established is not False else False
                gaps.append('state_history_post_state_port_unproved')
        subject = source_body if occupancy else source_request
        binding = (pair(ctx, subject, read_subject, shared) if bind_subject is None
                   else bind_subject(ctx, subject, read_subject, shared))
        candidate.update(post_established=established, source_point=point, subject_binding=binding)
        if established is True and locations and not shared.truncated:
            preservation = state_preservation(ctx, point,
                OccurrencePoint(read.id, OccurrencePhase.BEFORE), locations, shared=shared)
            candidate['preservation'] = preservation
            if preservation['preserved'] is not True:
                gaps.append('state_history_post_preservation_unproved')
        else:
            gaps.append('state_history_post_not_established')
        if binding['match'] is not True:
            gaps.append('state_history_subject_unproved')
        supported_rule = (predicate is not None and (
            update.post is SessionPost.HOLD if occupancy else
            predicate.contract.subject.rule is RequestStateRule.IS_COMPLETE
            and update.post is SessionPost.COMPLETE))
        if not supported_rule:
            gaps.append('state_history_post_implication_unresolved')
        if (supported_rule and read_supported and established is True
                and binding['match'] is True and candidate['preservation']['preserved'] is True
                and not shared.truncated):
            candidate['expected_bool'] = True
        candidate['gaps'] = tuple(dict.fromkeys(gaps))
        candidate['status'] = 'DECLARED' if candidate['expected_bool'] is True else 'PARTIAL'
    expected = (True if not shared.truncated and any(row['expected_bool'] is True
                                                    for row in candidates) else None)
    if shared.truncated:
        analysis.gap('state_history_budget_exhausted')
        if analysis.cursor is None:
            analysis.cursor = {'phase': 'state_history', 'read': read.id}
    return {'read': read, 'predicate': predicate, 'predicate_inventory': declarations,
        'source_inventory': inventory, 'missing_sources': missing,
        'candidates': tuple(candidates), 'locations': tuple(sorted(read_locations, key=lambda value: value.wire)),
        'expected_bool': expected, 'observed_branch': observed,
        'consistent': None if expected is None or observed is None else expected is observed,
        'status': 'NOT_CHECKED' if shared.truncated else 'DECLARED' if expected is True else 'PARTIAL',
        'gaps': tuple(analysis.gaps), 'pending_candidates': tuple(pending), 'cursor': analysis.cursor,
        'budget': {'limit': shared.limit, 'used': shared.used, 'costs': dict(shared.costs),
                   'truncated': shared.truncated},
        'preparation': {'source_rows': len(inventory), 'occurrence_rows': len(ctx.occurrences),
            'program_index_reused': prepared, 'new_program_indexes': int(not prepared and bool(inventory)),
            'derived_paths_precomputed': 0},
        'conditions': ('SessionUpdate implementation and source/native applicability remain conditional',
                       'location IDs and address labels do not establish isolation',
                       'expected truth is independent of the captured branch choice'), **_flags()}


# Conditional body activity and terminal history over original occurrences.
#
# Activity has an exact Drive interval. Terminal history uses the existing
# explicit native/no-reset lifecycle scope, independently of mutable state
# contents. Neither relation establishes native applicability or rewrite rights.
class _BodyHistory:
    def __init__(self, ctx, shared):
        self.ctx, self.shared = ctx, shared
        self.read_helper = _History(ctx, shared)
        self.gaps, self.cursor = [], None
        self.native_scopes, self.native_by_drive = set(), {}
        self.parents = {}
        for call in ctx.resolved_call_activations.values():
            self.parents.setdefault(call['activation'].id, []).append(call)

    def gap(self, code):
        if code not in self.gaps:
            self.gaps.append(code)

    def spend(self, kind, identity):
        if self.shared.truncated or not self.shared.spend('body_state_' + kind, path=str(identity)):
            self.gap('body_state_budget_exhausted')
            if self.cursor is None:
                self.cursor = {'phase': kind, 'identity': identity}
            return False
        return True

    def value(self, identity):
        return None if identity is None or type(identity) is Unresolved else self.ctx.values.get(identity)

    def use(self, occurrence, identity):
        return value_for_use(self.ctx, self.ctx.uses.get((occurrence.id, identity)))

    def operation_ports(self, occurrence):
        """All original owned ports, including raw scheduler/control Uses."""
        definition = self.ctx.core_operations[occurrence.operation]
        inputs, outputs, gaps = [], [], []
        before = self.ctx.before.get(occurrence.id, {})
        for use in definition.uses:
            if not self.spend('owned_uses', occurrence.id):
                break
            actual = self.ctx.uses.get((occurrence.id, use.id))
            value = value_for_use(self.ctx, actual)
            original = self.ctx.core_values[use.value]
            complete = value is not None and type(value) is not Unresolved
            if complete and isinstance(original, StateVersionValue):
                complete = before.get(original.location) == value.id
            inputs.append({'definition': use, 'actual': actual, 'value': value, 'supported': complete})
            if not complete:
                gaps.append('body_state_owned_use_unobserved')
        for identity in definition.results:
            if not self.spend('owned_results', occurrence.id):
                break
            value = self.ctx.results.get((occurrence.id, identity))
            outputs.append({'definition': self.ctx.core_values[identity], 'value': value})
            if value is None:
                gaps.append('body_state_owned_result_unobserved')
        transfer = self.ctx.outgoing.get(occurrence.id)
        if definition.terminator is not None and definition.terminator.successors:
            if transfer is None:
                gaps.append('body_state_owned_transfer_unobserved')
            else:
                block = self.ctx.blocks[self.ctx.visits[transfer.target].block]
                for identity in block.arguments:
                    if not self.spend('owned_landing_ports', occurrence.id):
                        break
                    value = self.ctx.results.get((transfer.target, identity))
                    outputs.append({'definition': self.ctx.core_values[identity], 'value': value})
                    if value is None:
                        gaps.append('body_state_owned_landing_unobserved')
        return {'input_inventory': definition.uses, 'inputs': tuple(inputs),
                'outputs': tuple(outputs), 'transfer': transfer,
                'supported': not gaps and not self.shared.truncated,
                'gaps': tuple(dict.fromkeys(gaps))}

    def native_projection(self, view):
        """Original source-first native self projection, not an identity union."""
        from scar.ir.program.protocol_await_records import AwaitAcquisition, AwaitBodyLink, ExpandedAwaitCallee
        from .await_links import project_await_links, await_identity_errors
        links = tuple(row for row in self.ctx.program.protocol_bindings
                      if type(row) is AwaitBodyLink and row.drive == view['drive'].operation
                      and row.projection is None)
        for scope in dict.fromkeys(row.scope for row in links):
            if scope in self.native_scopes:
                continue
            if not self.spend('native_source_scopes', scope):
                break
            self.native_scopes.add(scope)
            project_await_links(self.ctx, scope, self.shared)
            await_identity_errors(self.ctx, self.shared)
            for job in self.ctx.await_jobs:
                row = job.get('source_row')
                if (row is None or type(row.contract) is not AwaitAcquisition
                        or type(row.contract.expansion) is ExpandedAwaitCallee):
                    continue
                for occurrence in job['occurrences']:
                    if not self.spend('native_source_occurrences', occurrence['advance'].id):
                        break
                    sources = occurrence.get('matched_sources', ())
                    if (not occurrence['derived_checked'] or occurrence['advance_selected'] is not True
                            or occurrence['case_selected'] is not True or len(sources) != 1
                            or not occurrence['bindings']
                            or any(item['original_copy'] is not True for item in occurrence['bindings'])):
                        continue
                    source, = sources
                    if (source['accepted'] is not True or source['required_choices'] is not True
                            or source['body_source_gaps']
                            or any(getattr(gap, 'code', str(gap)).startswith('await_')
                                   for gap in occurrence['gaps'])):
                        continue
                    for route in occurrence['routes']:
                        for projection in route.get('projections', ()):
                            if (projection['projection'] is not None or not projection['bindings']
                                    or any(item['original_copy'] is not True
                                           or item['compatibility']['status'] == 'CONFLICT'
                                           for item in projection['bindings'])):
                                continue
                            identity = route['drive_occurrence'].id
                            actual = self.ctx.resolved_body_drives[identity]
                            # The native projection does not exempt ordinary
                            # delivery from the selected advance to this Drive.
                            delivery = pair(self.ctx, occurrence['iterator_value'],
                                            actual['instance']['value'], self.shared)
                            source_ports = self.operation_ports(source['occurrence'])
                            if (delivery['match'] is not True or not source_ports['supported']
                                    or self.shared.truncated):
                                continue
                            self.native_by_drive.setdefault(identity, []).append({
                                'source_row': row, 'source': source, 'advance': occurrence['advance'],
                                'iterator_bindings': occurrence['bindings'], 'route': route,
                                'projection': projection, 'delivery': delivery,
                                'source_ports': source_ports,
                                'scope': scope, **_flags()})
        receipts = tuple(self.native_by_drive.get(view['drive'].id, ()))
        supported = len(receipts) == 1 and not self.shared.truncated and not self.ctx.issues
        return {'inventory': links, 'receipts': receipts, 'supported': supported,
                'scope': 'conditional original native self acquisition; no ordinary copy edge is added',
                **_flags()}

    def body_pair(self, view, source, target):
        # Comparing another candidate's frame is not an asserted copy edge.
        # Select the local identity comparison before pair() could register a
        # global copy conflict for an ordinary, expected distinct object.
        from .content import ObjectIdentity
        from .content_validation import artifact_key
        identities = []
        for value in (source, target):
            row = (None if value is None or type(value) is Unresolved else
                   self.ctx.resolved_denotations.get(value.id))
            identities.append(None if row is None else row['denotation'].identity)
        left, right = identities
        if (type(left) is ObjectIdentity and type(right) is ObjectIdentity
                and artifact_key(left.object.artifact) == artifact_key(right.object.artifact)
                and left.object.object != right.object.object):
            # These declarations choose a comparison algorithm only. The
            # bounded helper still checks the fresh catalogs and actual scope.
            return self.object_binding(source, target)
        ordinary = pair(self.ctx, source, target, self.shared)
        if ordinary['match'] is not None or self.shared.truncated:
            return ordinary
        stored = None
        if self.ctx.program.stored_value_reads:
            from scar.ir.execution.stored_values import source_pair
            stored = source_pair(self.ctx, source, target, self.shared)
            if stored['match'] is False or self.shared.truncated:
                return {**stored, 'ordinary_binding': ordinary, 'stored_source_binding': stored}
            if stored['match'] is True and stored['conditional_stored_source']:
                return {**stored, 'ordinary_binding': ordinary, 'stored_source_binding': stored,
                        'binding_kind': 'CONDITIONAL_STORED_VALUE_SOURCE'}
        identity = self.object_binding(source, target, ordinary=ordinary)
        if stored is not None:
            identity = {**identity, 'stored_source_binding': stored}
        if identity['match'] is not None or self.shared.truncated:
            return identity
        if view is None:
            return identity
        projection = self.native_projection(view)
        if not projection['supported']:
            return {**identity, 'native_projection': projection}
        handle, receiver = view['instance']['creation_handle'], view['instance']['value']
        routes = []
        for left, right in ((handle, receiver), (receiver, handle)):
            a = pair(self.ctx, source, left, self.shared)
            b = pair(self.ctx, target, right, self.shared)
            routes.append({'source': a, 'target': b})
            if a['match'] is True and b['match'] is True and not self.shared.truncated:
                support = (*a['support'], *b['support'], {
                    'source': source.id, 'target': target.id,
                    'rule': 'CONDITIONAL_NATIVE_BODY_PROJECTION', 'support': projection['receipts']})
                return {**ordinary, 'match': True, 'ordinary_binding': ordinary,
                        'conditional_native_projection': True, 'native_projection': projection,
                        'projection_bindings': tuple(routes), 'support': support}
        return {**identity, 'native_projection': projection, 'projection_bindings': tuple(routes)}

    def object_binding(self, source, target, *, ordinary=None):
        """A supplied current object identity, never a new original copy."""
        from scar.ir.execution.source_validation import object_identity
        guard = object_identity(self.ctx, source, target, shared=self.shared)
        match = guard['match']
        if match is True and self.ctx.issues:
            match = None
        if self.shared.truncated:
            match = None
        support = (() if match is not True else ({
            'rule': 'DECLARED_SCOPED_OBJECT_IDENTITY',
            'source': source.id, 'target': target.id, 'identity_guard': guard},))
        return {**_unknown_pair(), 'source_value': source, 'target_value': target,
            'match': match, 'support': support, 'ordinary_binding': ordinary,
            'identity_guard': guard,
            'binding_kind': 'DECLARED_SCOPED_OBJECT_IDENTITY' if match is True else
                            'DECLARED_SCOPED_DISTINCT_OBJECTS' if match is False else
                            'UNPROVED_OBJECT_IDENTITY',
            'conditional_object_identity': match is True,
            'gaps': guard['gaps'],
            'conditions': ('current same-object declarations are conditional; original copy is separate',
                'no stored request field preservation or data-content equality is inferred')}

    def chain(self, occurrence, *, stop=None, segment=None):
        """Only exact synchronous ancestry; no temporal/callee instance guesses."""
        ctx, rows, gaps, seen = self.ctx, [], [], set()
        anchor, activation = occurrence, ctx.activation_of(occurrence.id)
        supported = True
        while activation not in seen:
            if not self.spend('call_ancestry', activation):
                supported = False
                break
            seen.add(activation)
            if activation == stop:
                if segment is not None and ctx.visits[anchor.visit].segment != segment:
                    supported = False
                    gaps.append('body_state_other_body_segment')
                return {'calls': tuple(rows), 'root': activation, 'anchor': anchor,
                        'supported': supported and not self.shared.truncated, 'gaps': tuple(gaps)}
            candidates = self.parents.get(activation, ())
            if not candidates and stop is None and ctx.activations[activation].region == ctx.program.root:
                return {'calls': tuple(rows), 'root': activation, 'anchor': anchor,
                        'supported': supported and not self.shared.truncated, 'gaps': tuple(gaps)}
            if len(candidates) != 1:
                supported = False
                gaps.append('body_state_call_association_missing')
                break
            call, = candidates
            ports = []
            structural_gaps = tuple(gap for gap in call['gaps']
                                    if gap.code != 'call_source_runtime_unverified')
            if structural_gaps:
                supported = False
                gaps.extend(gap.code for gap in structural_gaps)
            if call['entry'] is None or call['completion'] is None:
                supported = False
                gaps.append('body_state_call_endpoint_unobserved')
            call_path = None
            if call['entry'] is not None and call['completion'] is not None:
                block = ctx.blocks[call['entry'].block]
                first = self.read_helper.sites.get((call['entry'].id, block.operations[0]))
                if first is not None:
                    call_path = host_path(ctx, OccurrencePoint(first.id, OccurrencePhase.BEFORE),
                        OccurrencePoint(call['completion'].id, OccurrencePhase.BEFORE), budget=self.shared)
            if call_path is None:
                supported = False
                gaps.append('body_state_call_path_unproved')
            for route in call['inputs']:
                if not self.spend('call_input_ports', call['caller'].id):
                    supported = False
                    break
                binding = pair(ctx, route['actual_value'], route['formal'], self.shared)
                ports.append({'kind': 'input', 'route': route, 'binding': binding})
                if binding['match'] is not True:
                    supported = False
                    gaps.append('body_state_call_input_unproved')
            for route in call['outputs']:
                if not self.spend('call_output_ports', call['caller'].id):
                    supported = False
                    break
                actual = route['callee_use']
                source = None if actual is None else self.value(actual.value)
                target = route['caller_formal'] or route['caller_result']
                binding = pair(ctx, source, target, self.shared)
                ports.append({'kind': 'output', 'route': route, 'binding': binding})
                if binding['match'] is not True:
                    supported = False
                    gaps.append('body_state_call_output_unproved')
            rows.append({'association': call['association'], 'caller': call['caller'],
                'entry': call['entry'], 'completion': call['completion'], 'ports': tuple(ports),
                'entry_to_completion': call_path, 'validation_gaps': call['gaps']})
            anchor, activation = call['caller'], ctx.activation_of(call['caller'].id)
        if activation in seen and not gaps:
            gaps.append('body_state_call_ancestry_cycle')
        return {'calls': tuple(rows), 'root': activation, 'anchor': anchor,
                'supported': False, 'gaps': tuple(dict.fromkeys(gaps))}

    def drive_support(self, view):
        ports, gaps = [], []
        supported = (view['association'] is not None and view['entry_bridge']
                     and view['outcome_bridge'] and view['path_complete'])
        if not supported:
            gaps.append('body_state_drive_interval_unproved')
        owned_ports = self.operation_ports(view['drive'])
        driver_chain = self.chain(view['drive'])
        if not owned_ports['supported'] or not driver_chain['supported']:
            supported = False
            gaps.extend((*owned_ports['gaps'], *driver_chain['gaps']))
        origin = view['origin']
        creation_ports, creation_chain = None, None
        if (origin is None or origin['association'] is None or origin['creation'] is None
                or view['instance']['creation_handle'] is None):
            supported = False
            gaps.append('body_state_creation_unobserved')
        if origin is not None and origin['creation'] is not None:
            creation_ports = self.operation_ports(origin['creation'])
            creation_chain = self.chain(origin['creation'])
            if not creation_ports['supported'] or not creation_chain['supported']:
                supported = False
                gaps.extend((*creation_ports['gaps'], *creation_chain['gaps']))
        creation_binding = self.body_pair(view, view['instance']['value'],
                                         view['instance']['creation_handle'])
        if creation_binding['match'] is not True:
            supported = False
            gaps.append('body_state_drive_owner_unproved')
        for field in ('inputs', 'outputs', 'forwards'):
            for row in view[field]:
                if not self.spend('drive_ports', view['drive'].id):
                    supported = False
                    break
                binding = pair(self.ctx, row['source_value'], row['formal'], self.shared)
                ports.append({'kind': field, 'route': row, 'binding': binding})
                if row['source_use'] is None or binding['match'] is not True:
                    supported = False
                    gaps.append('body_state_drive_port_unproved')
        return {'supported': supported and not self.shared.truncated,
                'creation_binding': creation_binding, 'ports': tuple(ports),
                'owned_ports': owned_ports, 'driver_chain': driver_chain,
                'creation_ports': creation_ports, 'creation_chain': creation_chain,
                'gaps': tuple(dict.fromkeys(gaps))}


def body_state_history(ctx, read_id, *, shared):
    analysis = _BodyHistory(ctx, shared)
    read, program = ctx.occurrences[read_id], ctx.program
    operation = ctx.core_operations[read.operation]
    declarations = tuple(row for row in program.protocol_operations
        if row.operation == read.operation and type(row.contract) is AdmissionPredicate
        and type(row.contract.subject) is BodyStateCondition)
    predicate = declarations[0] if len(declarations) == 1 else None
    drive_inventory, commit_inventory = tuple(program.body_drives), tuple(program.body_outcome_commits)
    by_site = {}
    for row in ctx.occurrences.values():
        by_site.setdefault(row.operation, []).append(row)
    active = [dict(drive=view['drive'], association=view['association'],
        subject_binding=_unknown_pair(), drive_support=None, call_chain=None,
        entry_order=None, outcome_order=None, expected_bool=None, gaps=(), status='NOT_CHECKED')
        for view in ctx.resolved_body_drives.values()]
    terminal = [dict(commit=commit, source=source, drive=None, association=None,
        post_established=None, subject_binding=_unknown_pair(), terminal_scope=None,
        terminal_route=None, source_chain=None, read_chain=None, drive_support=None,
        source_state_bindings=(), content_preservation=_empty_preservation(),
        expected_bool=None, gaps=(), status='NOT_CHECKED')
        for commit in commit_inventory for source in by_site.get(commit.operation, ())]
    missing = tuple(('drive', row) for row in drive_inventory if not by_site.get(row.operation)) + tuple(
        ('commit', row) for row in commit_inventory if not by_site.get(row.operation))
    if missing:
        analysis.gap('body_state_original_sources_unobserved')
    index = getattr(ctx, 'body_program_index', None)
    prepared = index is not None
    if index is None:
        index = ProgramIndex(program)
    read_region = ctx.block_regions[ctx.visits[read.visit].block]
    read_scope = index.lifecycle_scope(read_region) if read_region in {
        row.region for row in program.driver_lifecycles} else None
    locations = tuple(sorted({access.location for access in operation.accesses}, key=lambda item: item.wire))
    read_supported, subject, boolean, observed = False, None, None, None
    if analysis.spend('read', read.id) and predicate is not None:
        read_supported = operation.effects_complete
        subject = analysis.use(read, predicate.contract.subject.instance_use)
        current = ctx.current(OccurrencePoint(read.id, OccurrencePhase.BEFORE)) or {}
        for access in operation.accesses:
            if not analysis.spend('read_accesses', read.id):
                read_supported = False
                break
            value = analysis.use(read, access.before)
            if (access.mode is not AccessMode.READ or value is None or type(value) is Unresolved
                    or current.get(access.location) != value.id):
                read_supported = False
                analysis.gap('body_state_read_input_unproved')
        result = predicate.contract.result
        if type(result) is OperationResult and result.operation == read.operation:
            boolean = ctx.results.get((read.id, operation.results[result.index]))
        if boolean is None or not locations:
            read_supported = False
            analysis.gap('body_state_read_result_or_support_unobserved')
        observed = analysis.read_helper.observed_branch(read, predicate, boolean)
    else:
        analysis.gap('body_state_predicate_unresolved')
    rule = None if predicate is None else predicate.contract.subject.rule
    if rule not in (BodyStatePredicateRule.IS_RUNNING, BodyStatePredicateRule.IS_CLOSED):
        analysis.gap('body_state_rule_inventory_only')
    if rule is BodyStatePredicateRule.IS_RUNNING and not active:
        analysis.gap('body_state_active_source_missing')
    if rule is BodyStatePredicateRule.IS_CLOSED and not terminal:
        analysis.gap('body_state_terminal_source_missing')
    before_read = OccurrencePoint(read.id, OccurrencePhase.BEFORE)
    for candidate in active:
        drive = candidate['drive']
        if not analysis.spend('active_sources', drive.id):
            break
        if rule is not BodyStatePredicateRule.IS_RUNNING:
            candidate['status'] = 'NOT_APPLICABLE'
            continue
        view = ctx.resolved_body_drives[drive.id]
        support = analysis.drive_support(view)
        binding = analysis.body_pair(view, view['instance']['creation_handle'], subject)
        gaps = list(support['gaps'])
        candidate.update(drive_support=support, subject_binding=binding)
        if binding['match'] is not True:
            gaps.append('body_state_subject_unproved')
        if view['association'] is not None and view['segment'] is not None:
            chain = analysis.chain(read, stop=view['association'].activation, segment=view['segment'].id)
            candidate['call_chain'] = chain
            gaps.extend(chain['gaps'])
        else:
            chain = None
            gaps.append('body_state_active_association_missing')
        entry, outcome = view['entry_first'], view['outcome']
        if entry is not None and outcome is not None:
            candidate['entry_order'] = host_path(ctx,
                OccurrencePoint(entry.id, OccurrencePhase.BEFORE), before_read, budget=shared)
            candidate['outcome_order'] = host_path(ctx, before_read,
                OccurrencePoint(outcome.id, OccurrencePhase.BEFORE), budget=shared)
        if not candidate['entry_order'] or not candidate['outcome_order']:
            gaps.append('body_state_active_interval_unproved')
        if (read_supported and support['supported'] and binding['match'] is True
                and chain is not None and chain['supported']
                and candidate['entry_order'] and candidate['outcome_order'] and not shared.truncated):
            candidate['expected_bool'] = True
        candidate['gaps'] = tuple(dict.fromkeys(gaps))
        candidate['status'] = 'DECLARED' if candidate['expected_bool'] is True else 'PARTIAL'
    for candidate in terminal:
        source, commit = candidate['source'], candidate['commit']
        if not analysis.spend('terminal_sources', source.id):
            break
        if rule is not BodyStatePredicateRule.IS_CLOSED:
            candidate['status'] = 'NOT_APPLICABLE'
            continue
        publication = index.body_publication(commit.operation)
        point = index.lifecycle_after(commit.operation)
        scope = point.scope if point is not None else None
        source_region = ctx.block_regions[ctx.visits[source.visit].block]
        transfer = ctx.incoming.get(source.visit)
        view = None if transfer is None else ctx.resolved_body_drives.get(transfer.operation)
        source_subject = analysis.use(source, commit.instance)
        binding = analysis.body_pair(view, source_subject, subject)
        gaps = []
        established = (view is not None and view['drive'].operation == commit.outcome.drive
            and view['outcome'] is not None and view['outcome'].operation == commit.outcome.source
            and publication.source.kind in (TerminatorKind.RETURN, TerminatorKind.RAISE)
            and scope is not None and scope.declaration.region == source_region
            and point.terminal.is_finished(publication.instance.use.value)
            and commit.operation in point.terminal.source_operations(publication.instance.use.value))
        candidate.update(drive=None if view is None else view['drive'],
            association=None if view is None else view['association'], terminal_scope=scope,
            subject_binding=binding)
        if view is not None:
            support = analysis.drive_support(view)
            candidate['drive_support'] = support
            drive_binding = analysis.body_pair(view, view['instance']['creation_handle'], source_subject)
            candidate['source_subject_binding'] = drive_binding
            established = established and support['supported'] and drive_binding['match'] is True
            gaps.extend(support['gaps'])
        else:
            gaps.append('body_state_terminal_drive_missing')
        source_operation = ctx.core_operations[source.operation]
        current = ctx.current(OccurrencePoint(source.id, OccurrencePhase.BEFORE)) or {}
        after = ctx.current(OccurrencePoint(source.id, OccurrencePhase.AFTER)) or {}
        ports = []
        for access in source_operation.accesses:
            if not analysis.spend('commit_state_ports', source.id):
                established = False
                break
            value = analysis.use(source, access.before)
            output = None if access.after is None else ctx.results.get((source.id, access.after))
            ports.append({'access': access, 'input': value, 'output': output,
                          'before': current.get(access.location), 'after': after.get(access.location)})
            if (value is None or type(value) is Unresolved or current.get(access.location) != value.id
                    or access.after is not None and (output is None or after.get(access.location) != output.id)):
                established = False
                gaps.append('body_state_terminal_state_port_unobserved')
        candidate['source_state_bindings'] = tuple(ports)
        candidate['post_established'] = True if established and not shared.truncated else None
        source_chain, read_chain = analysis.chain(source), analysis.chain(read)
        candidate.update(source_chain=source_chain, read_chain=read_chain)
        gaps.extend((*source_chain['gaps'], *read_chain['gaps']))
        route = host_path(ctx, OccurrencePoint(source.id, OccurrencePhase.AFTER), before_read, budget=shared)
        candidate['terminal_route'] = route
        same_scope = (scope is not None and read_scope is not None
            and scope.interface.body == read_scope.interface.body
            and scope.rule == read_scope.rule)
        if not same_scope:
            gaps.append('body_state_terminal_scope_unproved')
        if binding['match'] is not True:
            gaps.append('body_state_subject_unproved')
        if not route:
            gaps.append('body_state_terminal_route_unproved')
        domains = tuple(sorted(set(locations) | (set(scope.interface.lifecycle) if scope else set()),
                               key=lambda item: item.wire))
        if established and domains and not shared.truncated:
            candidate['content_preservation'] = state_preservation(ctx,
                OccurrencePoint(source.id, OccurrencePhase.AFTER), before_read, domains, shared=shared)
        # F is historical under the original explicit no-reset scope; P/field
        # contents may change across these same calls. Keep that receipt apart.
        if (read_supported and established and same_scope and binding['match'] is True and route
                and source_chain['supported'] and read_chain['supported']
                and source_chain['root'] == read_chain['root'] and not shared.truncated):
            candidate['expected_bool'] = True
        if not established:
            gaps.append('body_state_terminal_post_unproved')
        candidate['gaps'] = tuple(dict.fromkeys(gaps))
        candidate['status'] = 'DECLARED' if candidate['expected_bool'] is True else 'PARTIAL'
    expected = True if not shared.truncated and any(row['expected_bool'] is True
                                                   for row in (*active, *terminal)) else None
    analysis.gaps.extend(gap for gap in analysis.read_helper.gaps if gap not in analysis.gaps)
    if shared.truncated:
        analysis.gap('body_state_budget_exhausted')
        if analysis.cursor is None:
            analysis.cursor = {'phase': 'nested_query', 'read': read.id}
    return {'read': read, 'predicate': predicate, 'predicate_inventory': declarations,
        'drive_inventory': drive_inventory, 'commit_inventory': commit_inventory,
        'no_body_finish_inventory': tuple(program.no_body_finishes),
        'identity_inventory': tuple(ctx.model.denotations),
        'missing_sources': missing, 'read_scope': read_scope, 'locations': locations,
        'active_candidates': tuple(active), 'terminal_candidates': tuple(terminal),
        'source_keys': tuple(('active', row['drive'].id) for row in active)
                       + tuple(('terminal', row['source'].id) for row in terminal),
        'expected_bool': expected, 'observed_branch': observed,
        'consistent': None if expected is None or observed is None else expected is observed,
        'status': 'NOT_CHECKED' if shared.truncated else 'DECLARED' if expected is True else 'PARTIAL',
        'gaps': tuple(dict.fromkeys(analysis.gaps)), 'cursor': analysis.cursor,
        'budget': {'limit': shared.limit, 'used': shared.used, 'costs': dict(shared.costs),
                   'truncated': shared.truncated},
        'preparation': {'original_occurrences': len(ctx.occurrences),
            'call_associations': len(ctx.resolved_call_activations), 'program_index_reused': prepared},
        'conditions': ('original BodyStateCondition/native profile semantics remain conditional',
            'terminal F uses the original explicit same-native-lifetime/no-reset scope',
            'F history does not prove mutable state content, current P or physical lifetime',
            'same-object identity may use independent scoped denotations; unresolved basis remains conditional',
            'object identity does not prove request-field provenance or a stable data version',
            'created-close NoBodyFinish and other predicate rules are inventory-only here'),
        'native_phase_verified': False, **_flags()}


# Current HOLD support, with conditional request/body provenance receipts.
#
# The original association is evaluated even when the checking request never
# enters a BodyDrive. Ordinary identity/copy remains distinct from this fixed
# protocol projection, and mutable contents still require interval preservation.
class _Projections:
    def __init__(self, ctx, shared):
        self.ctx, self.shared = ctx, shared
        self.structure = _BodyHistory(ctx, shared)
        self.creates = tuple(ctx.program.async_request_creates)
        self.associations = tuple(ctx.program.request_association_reads)
        self.occurrences = {}
        for row in ctx.occurrences.values():
            self.occurrences.setdefault(row.operation, []).append(row)
        self.inventory = tuple((row, occurrence) for row in self.associations
                               for occurrence in self.occurrences.get(row.operation, ()))
        self.rows = []
        self.prepared = False

    def use(self, operation, identity):
        return value_for_use(self.ctx, self.ctx.uses.get((operation.id, identity)))

    def own_normal(self, operation, port):
        transfer = self.ctx.outgoing.get(operation.id)
        if transfer is None:
            return None
        definition = self.ctx.core_operations[operation.operation]
        edge = definition.terminator.successors[transfer.successor_index]
        if edge.kind is not ExitKind.NORMAL:
            return None
        return _result(self.ctx, operation, InvokeLiteralResult(
            ControlEdgeReference(definition.id, transfer.successor_index), CompletionResult(port.index)),
            shared=self.shared)

    def prepare(self):
        if self.prepared:
            return
        self.prepared = True
        ctx, shared = self.ctx, self.shared
        scopes = {row.id: row for row in ctx.program.protocol_scopes}
        interfaces = {row.region: row for row in ctx.program.driver_interfaces}
        for marker, association in self.inventory:
            if not shared.spend('session_occupancy_associations', path=str(association.id)):
                return
            region = ctx.block_regions[ctx.visits[association.visit].block]
            interface = interfaces.get(region)
            contracts = []
            for row in ctx.program.protocol_operations:
                if not shared.spend('session_occupancy_contract_rows', path=str(association.id)):
                    return
                scope = scopes.get(row.scope)
                if (row.operation == marker.operation and type(row.contract) is RequestAssociation
                        and scope is not None and type(scope.subject) is DriverSubject
                        and scope.subject.interface_region == region):
                    contracts.append(row)
            if interface is None or len(contracts) != 1:
                continue
            declaration, = contracts
            contract = declaration.contract
            selected = ctx.outgoing.get(association.id)
            if selected is None:
                continue
            body_port = InvokeLiteralResult(ControlEdgeReference(marker.operation, selected.successor_index),
                                            CompletionResult(marker.body_result.index))
            latent_port = InvokeLiteralResult(ControlEdgeReference(marker.operation, selected.successor_index),
                                              CompletionResult(marker.latent_result.index))
            if (contract.request_use != marker.request or contract.body != body_port
                    or contract.latent != latent_port):
                continue
            request = self.use(association, marker.request)
            parents = self.structure.parents.get(ctx.activation_of(association.id), ())
            receiver_routes = tuple(route for call in parents for route in call['inputs']
                if route['formal_definition'].id == interface.receiver)
            receiver = receiver_routes[0]['formal'] if len(receiver_routes) == 1 else None
            receiver_binding = pair(ctx, receiver, request, shared)
            body, latent = self.own_normal(association, marker.body_result), self.own_normal(association, marker.latent_result)
            ports = self.structure.operation_ports(association)
            chain = self.structure.chain(association)
            for creation in self.creates:
                for occurrence in self.occurrences.get(creation.operation, ()):
                    if not shared.spend('session_occupancy_create_sources', path=str(occurrence.id)):
                        return
                    result = self.own_normal(occurrence, creation.request_result)
                    request_binding = pair(ctx, result, request, shared)
                    if request_binding['match'] is not True:
                        continue
                    created_body = self.use(occurrence, creation.body)
                    created_latent = self.use(occurrence, creation.latent)
                    creation_ports = self.structure.operation_ports(occurrence)
                    creation_chain = self.structure.chain(occurrence)
                    body_compatibility = pair(ctx, created_body, body, shared)
                    latent_compatibility = pair(ctx, created_latent, latent, shared)
                    transfer = ctx.outgoing.get(occurrence.id)
                    route = None if transfer is None else host_path(ctx,
                        OccurrencePoint(occurrence.id, OccurrencePhase.EDGE, transfer.successor_index),
                        OccurrencePoint(association.id, OccurrencePhase.BEFORE), budget=shared)
                    supported = (all(value is not None and type(value) is not Unresolved for value in
                        (created_body, created_latent, body, latent))
                        and ports['supported'] and chain['supported']
                        and receiver_binding['match'] is True
                        and creation_ports['supported'] and creation_chain['supported']
                        and chain['root'] == creation_chain['root']
                        and body_compatibility['match'] is not False
                        and latent_compatibility['match'] is not False
                        and route is not None and not shared.truncated)
                    self.rows.append({'declaration': declaration, 'association': marker,
                        'association_occurrence': association, 'creation': creation,
                        'creation_occurrence': occurrence, 'request_binding': request_binding,
                        'receiver_binding': receiver_binding, 'receiver_routes': receiver_routes,
                        'body': body, 'latent': latent, 'created_body': created_body,
                        'created_latent': created_latent, 'ordinary_body_binding': body_compatibility,
                        'ordinary_latent_binding': latent_compatibility, 'association_ports': ports,
                        'association_chain': chain, 'creation_ports': creation_ports,
                        'creation_chain': creation_chain, 'source_route': route,
                        'supported': supported, 'conditional': True})

    def bind(self, ctx, source, target, shared):
        ordinary = pair(ctx, source, target, shared)
        if ordinary['match'] is not None or shared.truncated:
            return ordinary
        self.prepare()
        receipts = []
        # Each endpoint may use its own association of a different request.
        # Join only their independently recovered body inputs, never R1/R2 IDs.
        for left in self.rows:
            if not shared.spend('session_occupancy_left_projections', path='session.occupancy.body'):
                break
            if not left['supported']:
                continue
            source_binding = pair(ctx, source, left['body'], shared)
            if source_binding['match'] is not True:
                continue
            for right in self.rows:
                if not shared.spend('session_occupancy_right_projections', path='session.occupancy.body'):
                    break
                if not right['supported']:
                    continue
                target_binding = pair(ctx, right['body'], target, shared)
                root_binding = pair(ctx, left['created_body'], right['created_body'], shared)
                if target_binding['match'] is True and root_binding['match'] is True:
                    receipts.append({'source': left, 'target': right,
                        'source_binding': source_binding, 'target_binding': target_binding,
                        'original_body_binding': root_binding})
        if len(receipts) != 1 or shared.truncated or ctx.issues:
            return {**ordinary, 'projection_inventory': tuple(self.rows),
                    'conditional_projection': False}
        return {**ordinary, 'match': True, 'ordinary_binding': ordinary,
            'conditional_projection': True, 'projections': tuple(receipts),
            'support': (*ordinary.get('support', ()),
                        {'rule': 'CONDITIONAL_REQUEST_BODY_PROJECTION', 'receipt': receipts[0]}),
            'conditions': ('original fixed Create/RequestAssociation field semantics remain conditional',
                           'projection does not union identities or prove mutable state preservation')}


def session_occupancy_history(ctx, read_id, *, shared):
    projections = _Projections(ctx, shared)
    result = _post_history(ctx, read_id, shared=shared, occupancy=True,
                           bind_subject=projections.bind)
    return {**result, 'request_create_inventory': projections.creates,
        'association_inventory': projections.associations,
        'actual_association_inventory': projections.inventory,
        'body_projections': tuple(projections.rows),
        'conditions': (*result['conditions'],
            'HOLD may support occupied True; COMPLETE/ABANDON never establish globally empty',
            'occupied alone is not request rejection: the initial-step control is separate')}
