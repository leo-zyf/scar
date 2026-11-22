"""Session flow."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
from scar.ir.program.drivers import BodyDriveTerminator, DriverReceiverKind
from scar.ir.program.ids import OperationID, RegionID, ValueID
from scar.ir.program.lifecycle import LifecyclePostBinding, SessionPost
from scar.ir.program.lifecycle_flow import reference_assignments
from scar.ir.program.model import (
    AccessMode,
    ExitKind,
    ReferenceValue,
    StateVersionValue,
    TerminatorKind,
    CompletionResult,
    InvokeLiteralResult,
    Behavior,
    FootprintKind,
    UseRole,
)
from scar.ir.program.resumption import SuspendTerminator
from scar.ir.program.admission import admission_routes
from scar.ir.program.data_materials import required_choices, selected
from scar.ir.program.protocol_async import _Analysis
from scar.ir.program.protocol_connections import RequestAssociation, SessionOccupancyCondition
from scar.ir.program.protocols import AdmissionPredicate, DriverSubject, OwnedUse, ProtocolProfile
from scar.ir.program.state_addresses import ExistingAccess, InvokeStateOperand
from scar.ir.program.structured import ControlEdgeReference


# Current session predicates over fixed owner aliases and canonical state SSA.
#
# P is a finite must relation of ordered (R, B, predicate) triples. Supporting
# state versions are derived from the current program point, never a second
# mutable world. Boolean facts stabilize before explanatory source sets grow.
@dataclass(frozen=True, slots=True)
class SessionPostFact:
    request: ValueID
    body: ValueID
    post: SessionPost
    support: tuple[StateVersionValue, ...]
    sources: frozenset[OperationID]


@dataclass(frozen=True, slots=True)
class SessionPointFacts:
    reachable: bool
    posts: tuple[SessionPostFact, ...]

    def for_pair(self, request, body):
        return tuple(fact for fact in self.posts if fact.request == request and fact.body == body)


@dataclass(frozen=True, slots=True)
class SessionFacts:
    region: RegionID
    before: Mapping[OperationID, SessionPointFacts]
    after: Mapping[OperationID, SessionPointFacts]
    edges: Mapping[tuple[OperationID, int], SessionPointFacts]

    def __post_init__(self):
        for name in ("before", "after", "edges"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


def _post_closure(post):
    # These are current predicates, not mutually exclusive machine phases.
    return (post, SessionPost.ABANDON) if post is SessionPost.COMPLETE else (post,)


def _reference_targets(assignments, carriers):
    """Preindex all parallel edge assignments once, including unchanged names."""
    targets = {}
    for target in carriers:
        source = assignments.get(target, target)
        if source is not None:
            targets.setdefault(source, []).append(target)
    return targets


def _rename_relation(relation, targets):
    """Simultaneously rename closed alias carriers from the pre-edge snapshot."""
    return frozenset((request, body, post) for old_request, old_body, post in relation
                     for request in targets.get(old_request, ()) for body in targets.get(old_body, ()))


def derive_session_facts(graph, interface, terminal, *, core=None) -> SessionFacts:
    """Call only after shape validation and fixed E/F/T in this enabled driver."""
    if core is None:
        from .validation import ProgramCoordinates
        core = ProgramCoordinates.from_graph(graph)
    region = core.regions[interface.region]
    blocks = {block.id: block for block in region.blocks}
    all_ops = core.operations
    operations = {identity: all_ops[identity] for block in region.blocks for identity in block.operations}
    values = core.values
    updates = {row.operation: row for row in graph.session_updates if row.operation in operations}
    if not updates:
        # Empty-entry must analysis with no post sources cannot create a P.
        def empty(points):
            return {key: SessionPointFacts(point.reachable, ()) for key, point in points.items()}
        return SessionFacts(region.id, empty(terminal.before), empty(terminal.after), empty(terminal.edges))
    body_only = {row.operation for row in (*graph.body_outcome_commits, *graph.no_body_finishes)}
    lifecycle = tuple(interface.lifecycle)
    carriers = tuple(terminal.before[blocks[region.entry].operations[0]].aliases)
    incoming, outgoing = {block: [] for block in blocks}, {block: [] for block in blocks}
    edge_maps, edge_targets, edge_seed = {}, {}, {}
    support_before, support_after, support_edge = {}, {}, {}
    seeds, forget, kills = {}, {}, {}

    def seed(row, point):
        own = {use.id: use.value for use in operations[row.operation].uses}
        return frozenset((request, body, post)
            for request in point.aliases[own[row.request]]
            for body in point.aliases[own[row.body]] for post in _post_closure(row.post))

    for block in region.blocks:
        current = {values[value].location: value for value in block.arguments
                   if isinstance(values[value], StateVersionValue)}
        for identity in block.operations:
            op = operations[identity]
            support_before[identity] = tuple(current[location] for location in lifecycle)
            for result in op.results:
                value = values[result]
                if isinstance(value, StateVersionValue):
                    current[value.location] = result
            support_after[identity] = tuple(current[location] for location in lifecycle)
            forget[identity] = frozenset(result for result in op.results if isinstance(values[result], ReferenceValue))
            row = updates.get(identity)
            if row is not None and row.binding is LifecyclePostBinding.AFTER_OPERATION:
                seeds[identity] = seed(row, terminal.after[identity])
            kills[identity] = (identity not in body_only and (
                not op.effects_complete or op.callee is not None
                or isinstance(op.terminator, (BodyDriveTerminator, SuspendTerminator))
                or (op.terminator is not None and op.terminator.kind is TerminatorKind.INVOKE)
                or any(access.location in lifecycle and access.mode is not AccessMode.READ for access in op.accesses)
                or any(isinstance(values[result], StateVersionValue) and values[result].location in lifecycle
                       for result in op.results)))
        op = operations[block.operations[-1]]
        row = updates.get(op.id)
        for index, edge in enumerate(op.terminator.successors):
            key = (op.id, index)
            edge_targets[key] = edge.block
            outgoing[block.id].append(key)
            incoming[edge.block].append((block.id, key))
            edge_maps[key] = _reference_targets(reference_assignments(op, edge, values=values, blocks=blocks), carriers)
            states = {values[formal].location: formal for formal in blocks[edge.block].arguments
                      if isinstance(values[formal], StateVersionValue)}
            support_edge[key] = tuple(states[location] for location in lifecycle)
            if row is not None and row.binding is LifecyclePostBinding.NORMAL_COMPLETION and edge.kind is ExitKind.NORMAL:
                edge_seed[key] = seed(row, terminal.edges[key])

    def transfer(state, identity):
        if state is None:
            return None
        if identity in seeds:
            return seeds[identity]
        if kills[identity]:
            return frozenset()
        return frozenset(key for key in state if key[0] not in forget[identity] and key[1] not in forget[identity])

    def edge_transfer(state, key):
        if state is None:
            return None
        return edge_seed[key] if key in edge_seed else _rename_relation(state, edge_maps[key])

    before_blocks, after_blocks = {block: None for block in blocks}, {block: None for block in blocks}
    work, queued = deque((region.entry,)), {region.entry}
    while work:
        block_id = work.popleft()
        queued.remove(block_id)
        arriving = [edge_transfer(after_blocks[parent], key) for parent, key in incoming[block_id]
                    if after_blocks[parent] is not None]
        state = frozenset() if block_id == region.entry else (
            frozenset.intersection(*arriving) if arriving else None)
        before_blocks[block_id] = state
        for identity in blocks[block_id].operations:
            state = transfer(state, identity)
        if state != after_blocks[block_id]:
            after_blocks[block_id] = state
            for key in outgoing[block_id]:
                target = edge_targets[key]
                if target not in queued:
                    work.append(target)
                    queued.add(target)
    before, after, edges = {}, {}, {}
    for block in region.blocks:
        state = before_blocks[block.id]
        for identity in block.operations:
            before[identity] = state
            state = transfer(state, identity)
            after[identity] = state
        for key in outgoing[block.id]:
            edges[key] = edge_transfer(state, key)

    def mask(sources, allowed):
        return {key: origins for key, origins in sources.items() if allowed is not None and key in allowed and origins}

    def source_transfer(sources, identity):
        if identity in seeds:
            return {key: frozenset((identity,)) for key in seeds[identity]}
        if kills[identity]:
            return {}
        return mask(sources, after[identity])

    def source_edge(sources, key):
        if key in edge_seed:
            return {fact: frozenset((key[0],)) for fact in edge_seed[key]}
        result = {}
        for fact, origins in sources.items():
            for renamed in _rename_relation((fact,), edge_maps[key]):
                result[renamed] = result.get(renamed, frozenset()) | origins
        return mask(result, edges[key])

    source_before, source_after = {block: {} for block in blocks}, {block: {} for block in blocks}
    work = deque(block.id for block in region.blocks if before_blocks[block.id] is not None)
    queued = set(work)
    while work:
        block_id = work.popleft()
        queued.remove(block_id)
        sources = {}
        if block_id != region.entry:
            for parent, key in incoming[block_id]:
                if after_blocks[parent] is not None:
                    for fact, origins in source_edge(source_after[parent], key).items():
                        sources[fact] = sources.get(fact, frozenset()) | origins
            sources = mask(sources, before_blocks[block_id])
        source_before[block_id] = sources
        for identity in blocks[block_id].operations:
            sources = source_transfer(sources, identity)
        if sources != source_after[block_id]:
            source_after[block_id] = sources
            for key in outgoing[block_id]:
                target = edge_targets[key]
                if target not in queued:
                    work.append(target)
                    queued.add(target)

    def point(facts, sources, support):
        if facts is None:
            return SessionPointFacts(False, ())
        if any(not sources.get(key) for key in facts):
            raise AssertionError("a current session predicate requires a declared post source")
        states = tuple(values[identity] for identity in support)
        return SessionPointFacts(True, tuple(SessionPostFact(*key, states, sources[key])
            for key in sorted(facts, key=lambda key: (key[0].wire, key[1].wire, key[2].value))))

    public_before, public_after, public_edges = {}, {}, {}
    for block in region.blocks:
        sources = source_before[block.id]
        for identity in block.operations:
            public_before[identity] = point(before[identity], sources, support_before[identity])
            sources = source_transfer(sources, identity)
            public_after[identity] = point(after[identity], sources, support_after[identity])
        for key in outgoing[block.id]:
            public_edges[key] = point(edges[key], source_edge(sources, key), support_edge[key])
    return SessionFacts(region.id, public_before, public_after, public_edges)


# Original async-session occupancy reads, separate from body-frame phase.
#
# The body source is the checking request's declared association projection.
# Neither an observed bool nor a future BodyDrive selects that source.  These
# views describe supplied ports and conditions; HOLD history is a separate query.
def _answer(match=None, *, routes=(), gaps=(), cursor=None):
    return {'match': match, 'routes': tuple(routes), 'gaps': tuple(gaps), 'cursor': cursor,
            'source_runtime_verified': False, 'runtime_identity_verified': False,
            'value_identity_verified': False, 'transformation_authorized': False}


def _aggregate(ctx, reports, *, complete):
    """Every applicable original route is required, never an ANY success."""
    reports = tuple(reports)
    values = tuple(report['match'] for report in reports)
    match = (None if ctx.budget.truncated or not complete or not values else
             True if all(value is True for value in values) else
             False if all(value is False for value in values) else None)
    return _answer(match, routes=tuple(
        route if 'match' in route else {**route, 'match': report['match']}
        for report in reports for route in report['routes']),
                   gaps=tuple(gap for report in reports for gap in report['gaps']),
                   cursor=next((report.get('cursor') for report in reports if report.get('cursor')), None))


def _prepare(ctx):
    """Index only original records before any bounded candidate traversal."""
    cache = getattr(ctx, '_session_occupancy_inventory', None)
    if cache is not None:
        return cache
    predicates, rows, markers = [], {}, {}
    for row in ctx.graph.protocol_operations:
        if type(row.contract) is RequestAssociation:
            rows.setdefault((row.scope, ctx.region_of[row.operation]), []).append(row)
        elif (type(row.contract) is AdmissionPredicate
              and type(row.contract.subject) is SessionOccupancyCondition):
            predicates.append(row)
    for marker in ctx.graph.request_association_reads:
        markers.setdefault(ctx.region_of[marker.operation], []).append(marker)
    cache = {'predicates': tuple(predicates),
             'markers': tuple(ctx.graph.request_association_reads),
             'by_region': {key: tuple(value) for key, value in markers.items()},
             'rows': {key: tuple(value) for key, value in rows.items()},
             'creates': {row.operation: row for row in ctx.graph.async_request_creates}}
    ctx.preparation['session_occupancy_predicate_rows'] = len(predicates)
    ctx.preparation['session_occupancy_association_markers'] = len(cache['markers'])
    ctx.preparation['session_occupancy_association_rows'] = sum(len(value) for value in rows.values())
    ctx.preparation['session_occupancy_request_creates'] = len(cache['creates'])
    ctx._session_occupancy_inventory = cache
    return cache


def _declarations(ctx, row, inventory):
    region = ctx.region_of[row.operation]
    rows = inventory['rows'].get((row.scope, region), ())
    marked, declarations = set(), []
    for marker in inventory['by_region'].get(region, ()):
        operation = ctx.operations[marker.operation]
        normal = next(number for number, edge in enumerate(operation.terminator.successors)
                      if edge.kind is ExitKind.NORMAL)
        edge = ControlEdgeReference(operation.id, normal)
        body = InvokeLiteralResult(edge, CompletionResult(marker.body_result.index))
        latent = InvokeLiteralResult(edge, CompletionResult(marker.latent_result.index))
        sites = tuple(InvokeStateOperand(operation.id, use.id) for use in operation.uses
                      if use.id in operation.invoke_arguments
                      and isinstance(ctx.values[use.value], StateVersionValue))
        declarations.append((marker, RequestAssociation(marker.request, body, latent, sites)))
        marked.add(operation.id)
    declarations.extend((item, item.contract) for item in rows if item.operation not in marked)
    return tuple(declarations)


def _context():
    return {'scope': 'SUPPLIED_STATIC_CFG', 'dynamic_path_selected': False,
            'native_gate_verified': False, 'transformation_authorized': False,
            'frontiers': (), 'gaps': ()}


def _read_branch_support(ctx, analysis, operation, branch):
    """READ dependencies reach real branch transfers, not invented accesses."""
    support = []
    for number, access in enumerate(operation.accesses):
        if not analysis.spend('occupancy_read_branch_sites', operation.id):
            break
        actuals = []
        for use in branch.uses:
            if not analysis.spend('occupancy_read_branch_inputs', operation.id):
                break
            value = ctx.values[use.value]
            if isinstance(value, StateVersionValue) and value.location == access.location:
                actuals.append(OwnedUse(branch.id, use.id))
        bindings = tuple(ctx.bind(OwnedUse(operation.id, access.before), actual,
                                 path='async.occupancy.read_branch') for actual in actuals)
        binding = _aggregate(ctx, bindings, complete=bool(actuals) and not ctx.budget.truncated)
        if binding['match'] is not True:
            analysis.gap('async_session_occupancy_read_branch_unresolved', operation.id,
                         site=ExistingAccess(operation.id, number), binding=binding)
        support.append({'site': ExistingAccess(operation.id, number), 'access': access,
                        'actuals': tuple(actuals), 'bindings': bindings, 'binding': binding})
    return tuple(support)


def occupancy_view(ctx, row):
    """Resolve the original B/READ/bool tuple, without deciding gate admission."""
    if (type(row.contract) is not AdmissionPredicate
            or type(row.contract.subject) is not SessionOccupancyCondition):
        raise TypeError('occupancy views require exact SessionOccupancyCondition')
    cache = getattr(ctx, '_session_occupancy_views', None)
    if cache is None:
        cache = ctx._session_occupancy_views = {}
    key = row.scope, row.clause, row.operation, row.slot
    if key in cache:
        return cache[key]
    inventory = _prepare(ctx)
    operation = ctx.operations[row.operation]
    subject = row.contract.subject
    actual = OwnedUse(operation.id, subject.body_use)
    branch = ctx.operations[row.contract.branch]
    condition = OwnedUse(branch.id, branch.terminator.condition)
    scope = ctx.scopes.get(row.scope)
    interface = ctx.interfaces.get(ctx.region_of[operation.id])
    declarations = _declarations(ctx, row, inventory)
    analysis = _Analysis(ctx, prepare=False)
    analysis.creates = inventory['creates']
    gaps, associations = [], []
    body_checks, tuple_checks = [], []
    cursor = None
    complete = True
    correct_profile = (scope is not None
        and scope.profile is ProtocolProfile.ASYNC_GENERATOR_BODY_V1
        and type(scope.subject) is DriverSubject and interface is not None
        and interface.receiver_kind is DriverReceiverKind.ASYNC_REQUEST
        and scope.subject.interface_region == interface.region)
    result = ctx.port(row.contract.result)
    boolean = ctx.bind(row.contract.result, condition, path='async.occupancy.bool')
    read_sites = tuple(ExistingAccess(operation.id, number) for number in range(len(operation.accesses)))
    support = analysis.support(read_sites, operation, 'async.occupancy.read')
    read_branch_support = _read_branch_support(ctx, analysis, operation, branch)
    if boolean['match'] is not True:
        gaps.append('async_session_occupancy_bool_unresolved')
    if not correct_profile:
        complete = False
        gaps.append('async_session_occupancy_profile_unresolved')
    else:
        expected_request = analysis.formal(interface.region, interface.receiver)
        for ordinal, (declaration, contract) in enumerate(declarations):
            if not analysis.spend('occupancy_associations', operation.id):
                complete = False
                cursor = {'phase': 'occupancy_association', 'operation': operation.id, 'ordinal': ordinal}
                break
            source = contract.body
            entry = {'declaration': declaration, 'contract': contract,
                     'operation': ctx.operations[declaration.operation], 'body': source,
                     'latent': contract.latent, 'actual': actual, 'projections': (),
                     'applicable': None, 'choices': (), 'receiver': _answer(),
                     'body_binding': _answer(), 'state_support': (), 'projection_bindings': (),
                     'association_state_consumer': branch,
                     'binding': _answer(), 'gaps': ()}
            associations.append(entry)
            try:
                # Fixed original association inventory chooses the expected B.
                # The actual predicate input is only checked against that B.
                port = analysis.port(source, owner=declaration.operation, normal=True, reference=True)
                if port is None or port.gap is not None:
                    complete = False
                    entry['gaps'] = ('async_session_occupancy_association_port_unresolved',)
                    continue
                if ctx.reaches(source, operation.id) is False:
                    entry['applicable'] = False
                    continue
                choices, choices_complete = required_choices(ctx, source)
                if not choices_complete or ctx.budget.truncated:
                    complete = False
                    entry['gaps'] = ('async_session_occupancy_association_context_unresolved',)
                    continue
                choices = selected(ctx, choices)
                entry['choices'], entry['applicable'] = choices, True
                request = OwnedUse(declaration.operation, contract.request_use)
                receiver = ctx.bind(expected_request, request, choices=choices,
                                    path='async.occupancy.request')
                body = ctx.bind(source, actual, choices=choices, path='async.occupancy.body')
                origins = ctx.origins(request)
                projections = analysis.request_projection(request, 'async.occupancy.projection')
                completion = source.edge if type(source) is InvokeLiteralResult else None
                # The READ exposes only its genuine dependencies. The branch
                # independently forwards every original D domain, so it is the
                # full association-state consumer, including unread domains.
                current = analysis.support(contract.state_sites, branch, 'async.occupancy.association_state',
                                           after=True, completion=completion)
                source_complete = (origins['source_inventory_complete'] and bool(origins['recipes'])
                    and len(projections) == len(origins['recipes'])
                    and all(projection['source_inventory_complete'] for projection in projections))
                if not source_complete:
                    gaps.append('async_session_occupancy_request_origin_unresolved')
                projection_checks = []
                for projection in projections:
                    if not analysis.spend('occupancy_projection_tuples', operation.id):
                        source_complete = False
                        break
                    context = projection['context']
                    projection_check = _answer(context['match'], routes=(context,), gaps=context.get('gaps', ()))
                    projection_checks.append(projection_check)
                state_complete = (len(current) == len(contract.state_sites)
                                  and len(read_branch_support) == len(read_sites))
                # All caller recipes are one independent source inventory.
                # Joining each recipe separately against an all-caller copy
                # report would wrongly require every caller to be the same
                # caller. Joining the complete context set preserves coverage.
                source_context = _aggregate(ctx, projection_checks, complete=source_complete)
                source_binding = _aggregate(ctx,
                    (ctx.correlate(receiver, body, source_context),), complete=source_complete)
                combined = _aggregate(ctx, (ctx.correlate(source_binding,
                    *(item['binding'] for item in current),
                    *(item['binding'] for item in read_branch_support)),),
                    complete=source_complete and state_complete)
                entry.update(receiver=receiver, body_binding=body, projections=projections,
                             request_origins=origins, state_support=current,
                             source_binding=source_binding,
                             projection_bindings=tuple(projection_checks), binding=combined)
                # A marker on a different R is not an alternative owner for B.
                # Its known wrong receiver remains visible and cannot grant a
                # successful source route through another association.
                body_checks.append(entry['source_binding'] if receiver['match'] is True else _answer(
                    None, gaps=('async_session_occupancy_receiver_unresolved',)))
                tuple_checks.append(combined)
                if receiver['match'] is not True:
                    gaps.append('async_session_occupancy_receiver_unresolved')
                if combined['match'] is not True:
                    gaps.append('async_session_occupancy_source_tuple_unresolved')
            except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
                complete = False
                entry['gaps'] = ('async_session_occupancy_association_reference_unresolved',)
                entry['reason'] = str(exc)
    if not body_checks:
        gaps.append('async_session_occupancy_association_missing')
    body_binding = _aggregate(ctx, body_checks, complete=complete)
    binding = _aggregate(ctx, tuple_checks, complete=complete)
    routes = admission_routes(ctx, row) if not ctx.budget.truncated else _context()
    routes = {**routes, 'frontiers': tuple({**frontier,
        'target_compatible_at_snapshot': None, 'session_target_compatible_at_snapshot': None,
        'native_gate_verified': False} for frontier in routes['frontiers'])}
    gaps.extend(routes['gaps'])
    gaps.extend(item['code'] if type(item) is dict else item for item in analysis.gaps)
    gaps.extend(gap for entry in associations for gap in entry['gaps'])
    if binding['match'] is not True:
        gaps.append('async_session_occupancy_body_source_unresolved')
    if ctx.budget.truncated:
        gaps.append('protocol_budget_exhausted')
        complete = False
        cursor = cursor or analysis.cursor or {'phase': 'occupancy_tuple', 'operation': operation.id}
        body_binding = {**body_binding, 'match': None, 'cursor': cursor}
        binding = {**binding, 'match': None, 'cursor': cursor}
    wrong = body_binding['match'] is False or boolean['match'] is False
    view = {'predicate': row, 'operation': operation, 'subject': subject,
        'input': ctx.owned_use(actual.operation, actual.use), 'actual': actual, 'result': result,
        'branch': branch, 'condition': ctx.owned_use(condition.operation, condition.use),
        'bool_must_copy': boolean['match'], 'boolean_binding': boolean,
        'state_support': support, 'read_branch_support': read_branch_support,
        'associations': tuple(associations),
        'association_inventory': inventory['markers'], 'association_declarations': declarations,
        'predicate_inventory': inventory['predicates'], 'body_binding': body_binding, 'binding': binding,
        'successors': tuple(enumerate(branch.terminator.successors)), 'control_routes': routes,
        'expected_bool': None, 'target_compatible_at_snapshot': None,
        'status': 'CONTRADICTED' if wrong else 'PARTIAL' if gaps else 'DECLARED',
        'gaps': tuple(dict.fromkeys(gaps)), 'cursor': cursor,
        'inventory': {'associations_total': len(declarations), 'associations_checked': len(associations),
                      'known_inventory_complete': complete and not ctx.budget.truncated},
        'budget': {'limit': ctx.budget.limit, 'used': ctx.budget.used,
                   'costs': dict(ctx.budget.costs), 'truncated': ctx.budget.truncated},
        'conditions': ('occupancy is independent of frame phase and request initial/complete state',
                       'the bool alone does not authorize or reject the complete admission tuple',
                       'association projection and native implementation applicability remain conditional'),
        'native_phase_verified': False, 'native_admission_verified': False,
        'source_runtime_verified': False, 'runtime_identity_verified': False,
        'value_identity_verified': False, 'transformation_authorized': False}
    cache[key] = view
    return view


def occupancy_errors(ctx, row):
    """Ordinary predicate shape plus exact async B ownership, never R proxying."""
    from .admission import predicate_shape_errors
    if (type(row.contract) is not AdmissionPredicate
            or type(row.contract.subject) is not SessionOccupancyCondition):
        raise TypeError('occupancy checks require exact SessionOccupancyCondition')
    subject = row.contract.subject
    issues = list(predicate_shape_errors(ctx, row, identity=subject.body_use,
                                        state_sites=subject.state_sites))
    operation = ctx.operations[row.operation]
    expected = {subject.body_use, *(access.before for access in operation.accesses)}
    if {use.id for use in operation.uses} != expected:
        issues.append(('async_session_occupancy_inputs', str(row.operation),
                       'occupancy predicate has only its exact body and every original READ input'))
    scope = ctx.scopes.get(row.scope)
    interface = ctx.interfaces.get(ctx.region_of[operation.id])
    if (scope is None or scope.profile is not ProtocolProfile.ASYNC_GENERATOR_BODY_V1
            or type(scope.subject) is not DriverSubject or interface is None
            or interface.receiver_kind is not DriverReceiverKind.ASYNC_REQUEST
            or scope.subject.interface_region != interface.region):
        issues.append(('async_session_occupancy_profile', str(row.operation),
                       'occupancy requires this original async request driver scope'))
    if issues:
        return tuple(issues)
    view = occupancy_view(ctx, row)
    if view['body_binding']['match'] is False:
        issues.append(('async_session_occupancy_body', str(row.operation),
                       'occupancy input differs from every applicable independent request-body projection'))
    return tuple(issues)


# Session post and terminal frame-release shape over canonical endpoints.
#
# The declared native conditions are not proved here. Historical terminal
# eligibility is checked only after these local shapes and the core have passed.
def session_release_errors(graph, *, core, region_domains):
    if not (graph.session_updates or graph.terminal_frame_releases):
        return ()
    errors = []

    def error(code, path, message):
        errors.append((code, str(path), message))

    values, operations, blocks = core.values, core.operations, core.blocks
    interfaces = {row.region: row for row in graph.driver_interfaces}
    scopes = {row.region for row in graph.driver_lifecycles}

    def operation_scope(row):
        op = operations.get(row.operation)
        if op is None:
            error("session_release_operation", row.operation, "declaration must name its real operation")
            return None, None
        interface = interfaces.get(core.operation_regions[op.id])
        if interface is None or interface.region not in scopes:
            error("session_release_scope", row.operation, "declaration requires its owning enabled driver scope")
            return op, None
        return op, interface

    def reference(op, use_id):
        use = next((use for use in op.uses if use.id == use_id), None)
        return (use is not None and use.role is UseRole.REFERENCE
                and isinstance(values[use.value], ReferenceValue))

    def ordinary(op, expected, complete, prefix):
        if (op.terminator is not None or op.callee is not None or op.call_arguments
                or op.invoke_arguments or op.completions or op.effects_complete is not complete
                or set(op.behaviors) != expected):
            error(prefix + "_shape", op.id, "operation must satisfy its exact ordinary intrinsic effect shape")

    def accesses(op, domains, role_uses, *, release=False):
        prefix = "terminal_release" if release else "session_post"
        own = {use.id: use for use in op.uses}
        locations, before_uses, writes = [], set(), set()
        for access in op.accesses:
            locations.append(access.location)
            before_uses.add(access.before)
            before = own.get(access.before)
            if (access.mode not in ((AccessMode.READ_WRITE,) if release
                                    else (AccessMode.READ, AccessMode.READ_WRITE))
                    or access.footprint.kind is not FootprintKind.WHOLE
                    or access.full_overwrite or access.reference not in (None, *role_uses)
                    or before is None or before.role is not UseRole.STATE_CONTENT
                    or not isinstance(values[before.value], StateVersionValue)):
                error(prefix + "_access", op.id, "intrinsic uses whole current states with its fixed read/write shape")
            if access.mode is AccessMode.READ_WRITE:
                writes.add(access.after)
        if set(locations) != domains or len(locations) != len(domains):
            error(prefix + "_domains", op.id, "accesses cover each required domain exactly once")
        if not writes:
            error(prefix + "_writes", op.id, "intrinsic requires an actual nonempty written state set")
        if set(op.results) != writes:
            error(prefix + "_results", op.id, "results are exactly the actual written after states")
        if set(own) != before_uses | set(role_uses):
            error(prefix + "_uses", op.id, "only actual roles and state-access inputs belong to the intrinsic")

    for row in graph.session_updates:
        op, interface = operation_scope(row)
        if interface is None:
            continue
        if not reference(op, row.request) or not reference(op, row.body):
            error("session_post_roles", op.id, "request and body are independent owned Reference Uses")
        if row.binding is LifecyclePostBinding.AFTER_OPERATION:
            ordinary(op, {Behavior.CONTROL, Behavior.READ, Behavior.WRITE}, True, "session_post")
            accesses(op, set(interface.lifecycle), (row.request, row.body))
        else:
            if (op.terminator is None or op.terminator.kind is not TerminatorKind.INVOKE
                    or op.effects_complete or row.request not in op.invoke_arguments
                    or row.body not in op.invoke_arguments):
                error("session_post_invoke", op.id, "normal post requires a real unknown helper with both actual roles")
                continue
            own = {use.id: use for use in op.uses}
            domains = region_domains[interface.region]
            state_inputs = [values[own[identity].value].location for identity in op.invoke_arguments
                            if isinstance(values[own[identity].value], StateVersionValue)]
            if set(state_inputs) != domains or len(state_inputs) != len(domains):
                error("session_post_invoke_states", op.id, "helper consumes every current driver state exactly once")
            for edge in op.terminator.successors:
                produced = [values[formal].location for actual, formal in
                            zip(edge.arguments, blocks[edge.block].arguments)
                            if isinstance(values[formal], StateVersionValue)
                            and isinstance(actual, CompletionResult)]
                if set(produced) != domains or len(produced) != len(domains):
                    error("session_post_completion", op.id, "normal and raise retain every actual produced driver post-state")
            if not any(edge.kind is ExitKind.NORMAL for edge in op.terminator.successors):
                error("session_post_completion", op.id, "post attaches only to its helper's actual NORMAL successor")

    for row in graph.terminal_frame_releases:
        op, interface = operation_scope(row)
        if interface is None:
            continue
        if not reference(op, row.owner):
            error("terminal_release_owner", op.id, "release uses its owned original Reference owner")
        ordinary(op, {Behavior.READ, Behavior.WRITE, Behavior.MEMORY}, False, "terminal_release")
        accesses(op, region_domains[interface.region], (row.owner,), release=True)
    return tuple(errors)


__all__ = [
    'SessionPostFact',
    'SessionPointFacts',
    'SessionFacts',
    'derive_session_facts',
    'occupancy_errors',
    'occupancy_view',
    'session_release_errors',
]
