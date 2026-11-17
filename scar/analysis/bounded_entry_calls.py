"""Entry-scoped queries over admitted bounded CALL/source interfaces.

This is a projection of the existing ProgramBuild providers, not a frontend or
an invocation graph. A selected source body exposes its static CALL templates;
observations select guarded ABI alternatives without closing future dispatch.
"""
from collections import Counter, defaultdict, deque
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from hashlib import sha256
import hashlib
import json
import os
from pathlib import Path
from threading import get_ident

from scar.analysis.source_call_view import Flow, source_call_site
from scar.analysis.source_reference_slice import source_port
from scar.ir.program.dependencies import context_push, DependencyKind
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.record_codec import decode, encode


SCHEMA = 'scar.bounded-entry-calls'
_ACTIVE_ADMISSIONS = ContextVar('scar_active_entry_admissions', default=())


class EntryAdmission:
    """Nonserializable capability owned by one live entry replay interval."""
    __slots__ = ('_session', '_owner', '_view', '_catalog', '_pid', '_thread', '_transaction',
        '_registry', '_nonce', '_projection', '_digest', '_versions', '_references',
        '_state', '_closed', '_invalid', '_fact_resolver', '_fact_routes', '_fact_producer')

    def __new__(cls, *args, **kwargs):
        raise TypeError('entry admission requires an actual entry replay factory')

    def __setattr__(self, name, value):
        raise TypeError('entry admission handles are immutable')

    def __reduce_ex__(self, protocol):
        raise TypeError('entry admission cannot be serialized')

    def __copy__(self):
        raise TypeError('entry admission cannot be copied')

    def __deepcopy__(self, memo):
        raise TypeError('entry admission cannot be copied')


def _metadata_json(value):
    """Detach readonly JSON containers without dropping/coercing metadata."""
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError('entry metadata fingerprint requires original JSON string keys')
        return {key:_metadata_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_json(item) for item in value]
    return value


def _resolve_fact_artifact(catalog, reference, *, resolver=None, locators=()):
    """Verify an exact logical/physical receipt through the original resolver.

    The unit-field anchor validates bare artifacts without replacing any
    published FactRef. Additional original locators are verified in the same
    physical decode. Unknown receipts never fall back to filename guessing.
    """
    from scar.analysis.source_fact_materials import FactMaterialResolver, FACTS

    artifact = reference if type(reference) is ProviderArtifact else decode(ProviderArtifact, reference)
    if artifact.schema != FACTS:
        raise ValueError('original source fact artifact required')
    if resolver is None:
        resolver = FactMaterialResolver(catalog)
    elif resolver.catalog is not catalog:
        raise ValueError('source fact material resolver belongs to another catalog owner')
    route = resolver.route(ProviderFactReference(artifact, '#/unit'))
    anchor = '#/records/0/unit' if route.format == 'COMPILE_CONTEXT_F4_JSONL' else '#/unit'
    references = tuple(ProviderFactReference(artifact, locator)
        for locator in dict.fromkeys((anchor, *locators)))
    if resolver.verify(references) != (route,):
        raise ValueError('source fact artifact differs from its original catalog route')
    return resolver, route


def _entry_reference_materials(admission, reference):
    if reference['schema'] != 'scar.source-facts-unit':
        return (reference,)
    key = _canonical(reference)
    route = admission._fact_routes.get(key)
    if route is None:
        from scar.analysis import source_fact_materials
        resolver, route = _resolve_fact_artifact(admission._catalog, reference,
            resolver=admission._fact_resolver)
        object.__setattr__(admission, '_fact_resolver', resolver)
        admission._fact_routes[key] = route
        if admission._fact_producer is None:
            object.__setattr__(admission, '_fact_producer',
                _artifact(source_fact_materials.__file__, 'scar.python-producer'))
    else:
        resolver = admission._fact_resolver
        resolver.validate_catalog()
        artifact = decode(ProviderArtifact, reference)
        if resolver.route(ProviderFactReference(artifact, '#/unit')) != route:
            raise ValueError('source fact artifact original route changed')
    # These are the real files selected by SourceFactStore.unit_parent, not
    # the collection-relative logical URI retained in original evidence.
    return (*tuple(encode(item) for item in (*route.catalogs, route.physical)),
        admission._fact_producer)


def _entry_parent_materials(owner):
    """Check existing guards and enumerate their original file dependencies."""
    from scar.analysis import build

    if json.loads((owner.path/'manifest.json').read_bytes()) != owner.report:
        raise ValueError('entry admission ProgramBuild manifest changed')
    owner.catalog.fact_store.validate_parents()
    owner.catalog.interface_store.validate_parents()
    paths = {owner.path/'manifest.json', Path(__file__).resolve()}
    expected = {}
    for name, digest in owner.report['rules'].items():
        path = (Path(build.__file__).parent/name).resolve()
        paths.add(path); expected[str(path)] = digest, None
    for receipt in owner._admitted.values():
        path = Path(receipt['path']).absolute()
        paths.add(path); expected[str(path)] = receipt['sha256'], receipt['bytes']
    for source in owner.view.units.values():
        source.check()
        paths.update(Path(path).absolute() for path in source._guard)
        for path, raw in source.material._parents.items():
            path = Path(path).absolute()
            paths.add(path); expected[str(path)] = sha256(raw).hexdigest(), len(raw)
    for name, provider in owner.providers.items():
        if not name.startswith('bounded_call_bindings:'):
            continue
        provider.check()
        for path, raw in provider.materials.items():
            path = Path(path).absolute()
            paths.add(path); expected[str(path)] = sha256(raw).hexdigest(), len(raw)
        paths.update(Path(path).absolute() for path in provider.body_index._guard)
    state = _canonical(_metadata_json(dict(report=owner.report, catalog=owner.catalog.report,
        models=owner.catalog.fact_store.model_units, fact_rows=owner.catalog.fact_store.rows)))
    return paths, expected, state


def _check_entry_admission(session, admission):
    from scar.analysis.replay_transaction import current_transaction, _file_version

    if (type(admission) is not EntryAdmission or admission._session is not session
            or getattr(session, '_f4', None) is not admission._owner
            or admission._owner.closed or admission._closed
            or getattr(session, '_scar_entry_admissions', None) is not admission._registry
            or admission._registry.get(admission._nonce) is not admission):
        raise ValueError('entry admission needs its exact live owner session')
    if (os.getpid() != admission._pid or get_ident() != admission._thread
            or current_transaction() is not admission._transaction
            or not any(value is admission for value in _ACTIVE_ADMISSIONS.get())):
        raise ValueError('entry admission left its owner thread or active replay context')
    if admission._invalid:
        raise ValueError('entry admission was invalidated by changed parent material')
    try:
        owner = admission._owner
        if sha256(admission._projection.encode()).hexdigest() != admission._digest:
            raise ValueError('entry admission projection digest changed')
        if owner.view is not admission._view or owner.catalog is not admission._catalog:
            raise ValueError('entry admission original view/catalog owner changed')
        paths, expected, state = _entry_parent_materials(owner)
        if admission._state is not None and state != admission._state:
            raise ValueError('entry admission original parent metadata changed in memory')
        object.__setattr__(admission, '_state', state)
        for reference in admission._references:
            for material in _entry_reference_materials(admission, reference):
                path = Path(material['uri']).absolute()
                paths.add(path); expected[str(path)] = material['sha256'], material['size_bytes']
        for path in sorted(paths, key=str):
            version = _file_version(path)
            prior = admission._versions.get(version[0])
            wanted = expected.get(version[0])
            if (prior is not None and prior != version or wanted is not None and
                    (version[1] != wanted[0] or wanted[1] is not None and version[2] != wanted[1])):
                raise ValueError('entry admission dependency version changed: '+version[0])
            admission._versions[version[0]] = version
            if admission._transaction is not None:
                admission._transaction.watch(version[0], version[1], version[2])
    except BaseException:
        object.__setattr__(admission, '_invalid', True)
        raise


def _entry_projection(session, admission):
    _check_entry_admission(session, admission)
    return json.loads(admission._projection)


def _entry_admission_dependencies(session, admission, references):
    """Extend a live interval with producers/materials verified by its consumer."""
    _check_entry_admission(session, admission)
    for reference in references:
        if reference not in admission._references:
            admission._references.append(dict(reference))
    _check_entry_admission(session, admission)


@contextmanager
def _admit_entry_calls(session, report=None):
    """Replay once, then lend only the genuinely registered owner capability."""
    from scar.analysis.replay_transaction import current_transaction

    entry = derive_bounded_entry_calls(session)
    if report is not None and _canonical(entry) != _canonical(report):
        raise ValueError('bounded entry query differs from original ProgramBuild providers')
    registry = getattr(session, '_scar_entry_admissions', None)
    if registry is None:
        registry = {}; session._scar_entry_admissions = registry
    admission = object.__new__(EntryAdmission)
    values = dict(_session=session, _owner=session._f4, _view=session._f4.view,
        _catalog=session._f4.catalog, _pid=os.getpid(), _thread=get_ident(), _transaction=current_transaction(),
        _registry=registry, _nonce=object(), _projection=_canonical(entry),
        _digest=sha256(_canonical(entry).encode()).hexdigest(), _versions={},
        _references=[dict(entry['program']), dict(entry['producer'])], _state=None,
        _closed=False, _invalid=False, _fact_resolver=None, _fact_routes={}, _fact_producer=None)
    for name, value in values.items():
        object.__setattr__(admission, name, value)
    registry[admission._nonce] = admission
    token = _ACTIVE_ADMISSIONS.set((*_ACTIVE_ADMISSIONS.get(), admission))
    failed = False
    try:
        _check_entry_admission(session, admission)
        yield admission
    except BaseException:
        failed = True
        raise
    finally:
        try:
            if not (failed and admission._invalid):
                _check_entry_admission(session, admission)
        except BaseException:
            if not failed:
                raise
        finally:
            registry.pop(admission._nonce, None)
            object.__setattr__(admission, '_closed', True)
            _ACTIVE_ADMISSIONS.reset(token)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _artifact(path, schema):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return encode(ProviderArtifact(str(path), sha256(raw).hexdigest(), len(raw), schema))


def _route_query(provider, relation, caller_context):
    """Ask the existing frame-aware provider for this exact typed relation."""
    callee_context = context_push(caller_context, relation.frame)
    entering = relation.flow is Flow.ENTER
    source = replace(relation.source, context=caller_context if entering else callee_context)
    target = replace(relation.target, context=callee_context if entering else caller_context)
    neighborhood = provider.outgoing(source)
    edges = tuple(edge for edge in neighborhood.edges
        if edge.source == source and edge.target == target
        and edge.kind is relation.kind and edge.meaning is relation.meaning
        and any(condition.domain == 'scar.source-call.precondition'
            and condition.index == relation.interface[1] for condition in edge.conditions)
        and any(condition.domain == 'scar.source-call.route'
            and condition.name == relation.flow.value and condition.index == relation.route
            for condition in edge.conditions))
    if not edges:
        raise ValueError('bounded entry CALL lost its original frame-aware source route')
    return edges, neighborhood.frontiers


def derive_bounded_entry_calls(session):
    """Project conditional static entry reachability from an open BuildSession.

    Source ``query.owners`` and original function body scopes select the domain.
    Each body and CALL template appears once, including loop/recursive sites.
    Query contexts remain open for nested scopes: no captured invocation or
    exact receiver/content version is manufactured by this static projection.
    """
    from scar.analysis.build import BuildSession

    if not isinstance(session, BuildSession) or session._f4 is None:
        raise TypeError('an open original F4 BuildSession is required')
    owner = session._f4
    if owner.closed:
        raise ValueError('bounded entry queries require an open original session')
    if json.loads((owner.path/'manifest.json').read_bytes()) != owner.report:
        raise ValueError('original ProgramBuild manifest changed')
    providers = []
    for name, targets in owner.providers.items():
        if not name.startswith('bounded_call_bindings:'):
            continue
        targets.check()
        provider = targets.function_interfaces
        if provider is None:
            raise ValueError('bounded CALL source interfaces are not attached')
        providers.append((name, targets, provider))
    if not providers:
        raise ValueError('ProgramBuild has no admitted bounded CALL observation channel')

    view = owner.view
    root_unit = owner.report['f4']['entry_unit']
    root_facts = owner.catalog.fact_store.load(root_unit)
    root_scope = root_facts.root_scope.owner
    scopes = defaultdict(list)
    graphs = {}
    observations = []
    observed_kinds = Counter()
    observed_frontiers = []
    source_interface_count = 0
    for name, targets, provider in providers:
        store = provider.store
        for unit in store.report['shards']:
            graph = store.read_unit(unit)
            graphs[name, unit] = graph
            source_interface_count += len(graph.interfaces)
            for query in graph.queries:
                for scope in query.owners:
                    scopes[unit, scope].append((name, targets, provider, graph, query))
        for site in targets.plan['sites']:
            operation = targets.operations[site['id']]
            for point in targets.by_call.get((operation.namespace, operation.node), ()):
                record = targets.original(point)
                observed_kinds[record.target_kind] += 1
                body = targets.body_targets[point.namespace, point.node]
                row = dict(provider=name, site_id=site['id'], unit=site['unit'], query=site['query'],
                    operation=encode(operation), observation=encode(record), source_body=encode(body))
                observations.append(row)
                if body is None:
                    reason = targets.body_frontiers.get((point.namespace, point.node))
                    observed_frontiers.append(dict(provider=name, site_id=site['id'],
                        unit=site['unit'], query=site['query'], target_kind=record.target_kind,
                        evidence=encode(record.evidence), reason=reason or
                        'OBSERVED_CALLABLE_REQUIRES_OTHER_CALL_PROTOCOL'))

    pending = deque(((root_unit, root_scope),))
    visited = set()
    bodies = {}
    calls = {}
    selected_queries = set()
    interfaces = []
    frontiers = []
    route_edges = {}
    route_frontiers = {}
    while pending:
        scope_key = pending.popleft()
        if scope_key in visited:
            continue
        visited.add(scope_key)
        unit, scope = scope_key
        if not any(candidate_unit == unit for _, candidate_unit in graphs):
            frontiers.append(dict(unit=unit, scope=encode(scope),
                reason='REACHED_SOURCE_UNIT_HAS_NO_BOUNDED_CALL_PLAN'))
        # All nested callers retain an unknown context prefix. The scope
        # selection itself is the original query owner, not an invented frame.
        caller_context = () if scope_key == (root_unit, root_scope) else None
        for name, targets, provider, graph, query in scopes.get(scope_key, ()):
            call_key = name, unit, query.query
            original_call_key = unit, query.query
            if call_key in selected_queries:
                owners = calls[original_call_key]['entry_selected_owners']
                if encode(scope) not in owners:
                    owners.append(encode(scope))
                continue
            selected_queries.add(call_key)
            provider.attach(unit, queries=(query.query,))
            selected = provider.store._sites[unit].get(query.query)
            if selected is None:
                from types import SimpleNamespace
                facts = owner.catalog.fact_store.load(unit)
                call = facts.calls[query.query]
                operation, _, _ = source_call_site(view, SimpleNamespace(
                    result=source_port(facts, call.result), callee=source_port(facts, call.callee)))
                site_id, selection = None, query.call
                frontiers.append(dict(provider=name, unit=unit, query=query.query,
                    call=encode(query.call), reason='SOURCE_CALL_NOT_SELECTED_BY_BOUNDED_PLAN'))
            else:
                site, selection = selected
                site_id = site['id']
                operation = targets.operations[site_id]
            selection_row = dict(provider=name, site_id=site_id, selection=encode(selection),
                observed_attempts=len(targets.by_call.get((operation.namespace, operation.node), ())),
                interface_ordinals=list(query.interfaces))
            row = calls.setdefault(original_call_key, dict(unit=unit, query=query.query,
                operation=encode(operation), call=encode(query.call),
                entry_selected_owners=[], selections=[]))
            if row['operation'] != encode(operation) or row['call'] != encode(query.call):
                raise ValueError('bounded observations disagree on the original source CALL')
            if encode(scope) not in row['entry_selected_owners']:
                row['entry_selected_owners'].append(encode(scope))
            row['selections'].append(selection_row)
            for frontier in graph.frontiers:
                if frontier.query == query.query:
                    frontiers.append(dict(provider=name, site_id=site_id, frontier=encode(frontier)))
            for ordinal in query.interfaces:
                interface = graph.interfaces[ordinal]
                if interface.query != query.query:
                    raise ValueError('bounded entry CALL changed original interface query index')
                key = unit, ordinal
                frame = provider.frames.get(key)
                if frame is None:
                    # Deferred creation has a retained interface but does not
                    # enter an ordinary executing function body.
                    interfaces.append(dict(provider=name, site_id=site_id, ordinal=ordinal,
                        interface=encode(interface), frame=None, ordinary_body_entry=False))
                    continue
                body = interface.link.body
                source = view.unit(body.unit)
                matches = [(index, function) for index, function in enumerate(source.facts.functions)
                    if function.template == body.template and function.body == body.region]
                if len(matches) != 1 or matches[0][1].body_scope is None:
                    raise ValueError('bounded entry target lacks its unique original function body scope')
                index, function = matches[0]
                target_key = body.unit, function.body_scope.owner
                body_wire = _canonical(encode(body))
                body_row = dict(body=encode(body), scope=encode(function.body_scope),
                    definition=encode(provider.store.slicer.fact(source.facts, 'functions', index)),
                    point=encode(view.source_region(body.unit, body.template, body.region)))
                if body_wire in bodies and bodies[body_wire] != body_row:
                    raise ValueError('shared bounded entry body has conflicting original evidence')
                bodies[body_wire] = body_row
                pending.append(target_key)
                relations = tuple(relation for relation in provider.relations if relation.interface == key)
                edges = {}
                for relation in relations:
                    rows, gaps = _route_query(provider, relation, caller_context)
                    for edge in rows:
                        wire = encode(edge)
                        edges[_canonical(wire)] = edge
                        route_edges[_canonical(wire)] = edge
                    for frontier in gaps:
                        wire = encode(frontier)
                        route_frontiers[_canonical(wire)] = frontier
                interfaces.append(dict(provider=name, site_id=site_id, ordinal=ordinal,
                    interface=encode(interface), frame=encode(frame), ordinary_body_entry=True,
                    caller_scope=encode(scope), callee_scope=encode(function.body_scope.owner),
                    _typed_route_edge_keys=sorted(edges)))

    active = selected_queries
    for row in observations:
        row['entry_scope_reachable'] = (row['provider'], row['unit'], row['query']) in active
    for row in observed_frontiers:
        row['entry_scope_reachable'] = (row['provider'], row['unit'], row['query']) in active
    body_units = {root_unit, *(row['body']['unit'] for row in bodies.values())}
    for _, targets, provider in providers:
        targets.check()
        provider.store.validate()
    if json.loads((owner.path/'manifest.json').read_bytes()) != owner.report:
        raise ValueError('original ProgramBuild manifest changed during entry query')
    # Reuse the existing normalized dependency-query representation, so shared
    # namespaces, contexts and evidence are retained once instead of repeating
    # their full wire records at every route endpoint.
    from scar.analysis.program_dependencies import DependencyWalk
    from scar.analysis.program_dependency_export import export_dependency_walk
    edge_keys = sorted(route_edges)
    edge_index = {key: index for index, key in enumerate(edge_keys)}
    edges = tuple(route_edges[key] for key in edge_keys)
    points = {point for edge in edges for point in (edge.source, edge.target)}
    roots = {edge.source for edge in edges if edge.kind is DependencyKind.CALL_TARGET}
    point_order = lambda point: _canonical(encode(point))
    route_query = export_dependency_walk(DependencyWalk(
        tuple(sorted(roots, key=point_order)), tuple(sorted(points, key=point_order)), edges,
        tuple(route_frontiers[key] for key in sorted(route_frontiers)), 'outgoing', False,
        tuple(sorted({edge.kind for edge in edges}, key=lambda kind: kind.value))))
    for interface in interfaces:
        if '_typed_route_edge_keys' in interface:
            interface['typed_route_edge_refs'] = ['#/source_route_query/edges/'+str(edge_index[key])
                for key in interface.pop('_typed_route_edge_keys')]
    return dict(schema=SCHEMA, schema_version=1,
        scope='CONDITIONAL_STATIC_SOURCE_CALL_REACHABILITY_FROM_ORIGINAL_ENTRY',
        program=_artifact(owner.path/'manifest.json', owner.report['schema']),
        producer=_artifact(__file__, 'scar.python-producer'),
        entry=encode(owner.entry), entry_scope=encode(root_scope),
        source_bodies=[bodies[k] for k in sorted(bodies)],
        calls=[calls[k] for k in sorted(calls)],
        interfaces=interfaces, observed_targets=observations,
        frontiers=frontiers, observed_target_frontiers=observed_frontiers,
        source_route_query=route_query,
        summary=dict(catalog_units=len(owner.catalog.fact_store.rows),
            preparation_loaded_units=len(view.units), entry_projection_units=len(body_units),
            entry_static_scopes=len(visited), entry_static_call_templates=len(calls),
            entry_conditional_interfaces=len(interfaces), shared_source_bodies=len(bodies),
            all_bounded_source_interfaces=source_interface_count,
            observed_attempts=len(observations), observed_target_kinds=dict(observed_kinds),
            entry_reachable_observed_attempts=sum(row['entry_scope_reachable'] for row in observations),
            observed_target_frontiers=dict(Counter(row['reason'] for row in observed_frontiers)),
            typed_route_edges=len(route_edges), typed_route_kinds=dict(Counter(
                edge.kind.value for edge in route_edges.values()))),
        qualification=dict(source_graphs_and_bodies_copied=False, loops_unrolled=False,
            recursion_unrolled=False, runtime_invocations_created=False,
            caller_invocation_or_callee_entry_completion_proven=False,
            unique_future_dispatch_proven=False, receiver_or_content_versions_proven=False,
            effects_closed=False, target_executed_by_projection=False,
            first_stage_complete=False, transformation_authorized=False,
            shape_or_physical_resource_facts_inferred=False,
            body_membership='ORIGINAL_SOURCE_QUERY_OWNERS_NOT_PREPARATION_LOADS',
            unknown_context_prefix='PRESERVED_FOR_NESTED_STATIC_QUERIES',
            observations='BOUNDED_ORIGINAL_ATTEMPTS_NOT_FUTURE_TARGET_EXHAUSTIVENESS'))


def verify_bounded_entry_calls(session, report):
    """Replay original providers in the caller's independently admitted session."""
    expected = derive_bounded_entry_calls(session)
    if _canonical(report) != _canonical(expected):
        raise ValueError('bounded entry query differs from original ProgramBuild providers')
    return dict(schema=SCHEMA+'.verification', verified=True,
        report_sha256=sha256(_canonical(report).encode()).hexdigest(),
        summary=expected['summary'], target_executed=False,
        runtime_invocations_created=False, first_stage_complete=False)


def derive_entry_source_witness(session, report, *, path, sha256, code_key):
    """Independently admit the entry before producing its source witness."""
    with _admit_entry_calls(session, report) as admission:
        return _derive_entry_source_witness_admitted(session, admission,
            path=path, sha256=sha256, code_key=code_key)


def _derive_entry_source_witness_admitted(session, admission, *, path, sha256, code_key):
    """Reuse only an actual live admission; no serialized verification flag."""
    report = _entry_projection(session, admission)
    result = _source_witness(session, report, path=path, sha256=sha256, code_key=code_key)
    _entry_admission_dependencies(session, admission, (result['source'],))
    return result


def _source_witness(session, report, *, path, sha256, code_key):
    """Bind an exact source CodeID to an admitted conditional entry interface.

    A matching name is never sufficient. Original attempt membership, the
    provider's selection fact, its ordinary frame and its source Body must all
    agree. This proves static source correspondence, not an invocation or any
    runtime argument/content, effect or future dispatch property.
    """
    source_path = Path(path).absolute()
    source_bytes = source_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != sha256:
        raise ValueError('entry source witness differs from the current source bytes')
    if type(code_key) is not str or not code_key:
        raise ValueError('entry source witness requires an exact CodeID key')
    owner = session._f4
    matches, rejected = [], []
    code_matches = 0
    for observation_index, row in enumerate(report['observed_targets']):
        record = row['observation']
        code = record['target_code']
        if code is None or (code['key'], code['filename']) != (code_key, str(source_path)):
            continue
        code_matches += 1
        observation_pointer = '#/observed_targets/'+str(observation_index)
        if not row['entry_scope_reachable']:
            rejected.append(dict(observation=observation_pointer,
                reason='OBSERVED_SOURCE_CODE_OUTSIDE_ENTRY_CONDITIONAL_SCOPE_CHAIN'))
            continue
        targets = owner.providers[row['provider']]
        provider = targets.function_interfaces
        linked = False
        for interface_index, selected in enumerate(report['interfaces']):
            if ((selected['provider'], selected['site_id']) != (row['provider'], row['site_id'])
                    or not selected['ordinary_body_entry']):
                continue
            ordinal = selected['ordinal']
            interface = provider.links[row['unit'], ordinal]
            frame = provider.frames[row['unit'], ordinal]
            attempts = provider.store.interface_attempts(row['unit'], ordinal)
            binding_evidence = provider.store.interface_evidence(row['unit'], ordinal)
            selection = provider.store.interface_fact(row['unit'], ordinal)
            # A grouped alternative can have several original attempts. A
            # descriptor's frame selection is its original GET fact, whereas
            # the observation remains a separate original attempt fact.
            if record['evidence'] not in tuple(encode(attempt) for attempt in attempts):
                continue
            if (frame.selection != selection or selection not in binding_evidence
                    or interface.query != row['query']
                    or encode(interface) != selected['interface'] or encode(frame) != selected['frame']
                    or encode(frame.caller) != row['operation']['namespace']
                    or encode(frame.operation) != row['operation']['node']):
                raise ValueError('entry source witness lost its original interface/frame selection')
            body = interface.link.body
            body_point = owner.view.source_region(body.unit, body.template, body.region)
            if (encode(body_point) != row['source_body'] or frame.target != body_point.namespace
                    or frame.entry != body_point.node):
                raise ValueError('entry source witness changed the observed original source body')
            model = owner.catalog.fact_store.model_units[body.unit]
            if (model['path'], model['raw_sha256']) != (str(source_path), sha256):
                raise ValueError('entry source witness differs from the admitted original F4 source')
            bodies = [(index, original) for index, original in enumerate(report['source_bodies'])
                if original['body'] == encode(body) and original['point'] == encode(body_point)]
            if len(bodies) != 1 or bodies[0][1]['definition'] != encode(interface.definition):
                raise ValueError('entry source witness lacks one shared original body definition')
            body_index, original = bodies[0]
            matches.append(dict(provider=row['provider'], site_id=row['site_id'],
                observation=observation_pointer, interface='#/interfaces/'+str(interface_index),
                source_body='#/source_bodies/'+str(body_index),
                original_body=encode(body), original_body_point=encode(body_point),
                original_function_definition=original['definition'],
                original_call=encode(interface.call), attempt_evidence=record['evidence'],
                frame_selection=encode(selection), binding_evidence=encode(binding_evidence),
                conditions=list(interface.link.conditions), remaining_fields=record['remaining_fields']))
            linked = True
        if not linked:
            rejected.append(dict(observation=observation_pointer,
                reason='OBSERVED_SOURCE_CODE_HAS_NO_MATCHING_ORDINARY_ENTRY_INTERFACE'))
        targets.check()
        provider.store.validate()
    if source_path.read_bytes() != source_bytes:
        raise ValueError('source bytes changed during entry source witness derivation')
    return dict(schema=SCHEMA+'.source-witness', schema_version=1,
        requested_source=dict(path=str(source_path), sha256=sha256, code_key=code_key),
        source=encode(ProviderArtifact(str(source_path.resolve()), sha256,
            len(source_bytes), 'scar.python-source')), program=report['program'],
        entry_projection=dict(schema=report['schema'], schema_version=report['schema_version'],
            canonical_sha256=hashlib.sha256(_canonical(report).encode()).hexdigest()),
        producer=_artifact(__file__, 'scar.python-producer'), witnessed=bool(matches),
        observed_code_matches=code_matches, matches=matches, rejected_observations=rejected,
        frontier=[] if matches else ['SOURCE_CODE_NOT_IN_ORDINARY_ENTRY_CONDITIONAL_CALL_CHAIN'],
        qualification=dict(proof=('EXACT_SOURCE_BYTES_AND_CODE_ID_IN_CONDITIONAL_STATIC_ENTRY_CHAIN'
            if matches else 'NO_ORDINARY_ENTRY_SOURCE_WITNESS'),
            parameters_or_value_contents_proven=False, runtime_invocations_created=False,
            caller_invocation_or_callee_entry_completion_proven=False,
            unique_future_dispatch_proven=False, effects_closed=False,
            native_implementation_equivalence_or_deployment_authorized=False,
            target_executed=False, first_stage_complete=False))


__all__ = ['derive_bounded_entry_calls', 'verify_bounded_entry_calls', 'derive_entry_source_witness']
