"""Contextual module bindings."""

from __future__ import annotations

import ast
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass, replace
from scar.analysis.attribute_selector import (
    derive_attribute_selector,
    prepare_attribute_selector_source,
)
from scar.analysis.contextual_receiver_sources import walk_reference_sources
from scar.analysis.source_attribute_lookup import _source_lookup
from scar.analysis.static.namespace import NamespaceBindingWitness, normal_reference
from scar.ir.program.attribute_lookups import StaticAttributeSelector
from scar.ir.program.dependencies import (
    DependencyCondition as Condition,
    DependencyEdge as Edge,
    DependencyEvidence as Evidence,
    DependencyFrontier as Frontier,
    DependencyKind as K,
    DependencyMeaning as M,
    DependencyNeighborhood as Neighborhood,
    DependencyPoint as Point,
    DependencyStrength as Strength,
    GraphNamespace,
)
from scar.ir.program.ids import OperationID, UseID, ValueID
from scar.ir.program.model import LiteralOrigin, ReferenceValue, StateVersionValue, TerminatorKind
from scar.ir.program.source_call_bindings import SourceBodyNamespace
from scar.ir.program.static_calls import (
    StaticAttributeAccess,
    StaticImportFromRead,
    StaticImportFromRequest,
)
from scar.ir.program.structured import ControlEdgeReference
from scar.analysis.static.imports import _unit_mappings, _importer_contexts, _resolve_from
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.v2.common import SourceReference


# Conditional source modules for original contextual import-result ports.
#
# The requested spelling/level/fromlist are original source/compiler operands.
# A separately declared package candidate can resolve a relative request, but
# never becomes the runtime __package__, an executed import or a closed source
# set. This adapter does not modify historical import routes or wire records.
@dataclass(frozen=True, slots=True)
class ContextualModuleImport:
    origin: Point
    operation: OperationID
    source: SourceReference
    requested_spelling: str
    relative_level: int
    resolved_module: str
    module: SourceBodyNamespace
    locals_namespace: ValueID
    globals_namespace: ValueID
    entry_world: ValueID
    world: UseID
    normal: ControlEdgeReference
    conditions: tuple[Condition, ...]
    evidence: tuple[Evidence, ...]
    request: StaticImportFromRequest | None = None
    package_context: str | None = None

    def __post_init__(self):
        for value, kind in ((self.origin, Point), (self.operation, OperationID),
                (self.source, SourceReference), (self.module, SourceBodyNamespace),
                (self.locals_namespace, ValueID), (self.globals_namespace, ValueID),
                (self.entry_world, ValueID), (self.world, UseID),
                (self.normal, ControlEdgeReference)):
            if type(value) is not kind:
                raise TypeError('contextual import needs exact ' + kind.__name__)
            value.__post_init__()
        if (type(self.origin.node) is not ValueID or self.normal.operation != self.operation
                or type(self.requested_spelling) is not str
                or type(self.relative_level) is not int or self.relative_level < 0
                or type(self.resolved_module) is not str or not self.resolved_module
                or self.module.source is not None):
            raise ValueError('contextual import changed its original request/module/normal ports')
        for rows, kind in ((self.conditions, Condition), (self.evidence, Evidence)):
            if type(rows) is not tuple or not rows or any(type(row) is not kind for row in rows):
                raise TypeError('contextual import needs nonempty typed conditions/evidence')
        if self.request is not None:
            if type(self.request) is not StaticImportFromRequest:
                raise TypeError('FROM import must retain its exact original request')
            self.request.__post_init__()
            if self.request.fromlist_constant.source != self.source:
                raise ValueError('FROM import changed its original fromlist source')
        elif self.relative_level:
            raise ValueError('relative import requires its original FROM request')
        if self.relative_level:
            if type(self.package_context) is not str or not self.package_context:
                raise ValueError('relative import needs its declared package candidate')
            if not any(condition.domain == 'scar.contextual-import'
                    and condition.name == 'CURRENT_IMPORT_PACKAGE_CONTEXT_MATCHES_DECLARED_CANDIDATE'
                    and condition.subject == self.origin and condition.operands
                    and type(condition.index) is int for condition in self.conditions):
                raise ValueError('relative import cannot drop its declared-package predicate')
        elif self.package_context is not None:
            raise ValueError('an absolute request has no inferred package-context requirement')

    @property
    def module_context(self):
        return None


class ContextualImportSources:
    """One bounded source query, sharing the resolver's original input owner.

    Only the last point's diagnostics are retained. Derived routes are freshly
    recomputed; original module/source indices remain owned by the resolver.
    """
    def __init__(self, resolver):
        self.resolver = resolver
        self._point = None
        self._frontiers = {}

    def _note(self, point, kind, detail):
        self._frontiers[Frontier(point, kind, detail)] = None

    def frontiers_for(self, point):
        return tuple(self._frontiers) if self._point == point else ()

    def _unit(self, body):
        number = body.context.number
        unit = self.resolver.owner.units[number]
        if unit['path'] != body.metadata.source.path:
            raise ValueError('contextual importer unit differs from its original source body')
        return number, unit

    def _declaration(self, number, suffix=''):
        session = self.resolver.session
        artifact = session.manifest_artifact
        locator = SourceArtifactLocator(artifact, artifact.sha256, artifact.size_bytes,
            'canonical-json', '#/units/' + str(number) + suffix)
        session.read_artifact(locator)
        return locator

    def _mapping(self, number, unit, module_name, source_root):
        rows = unit.get('module_mappings')
        if type(rows) is list:
            for ordinal, row in enumerate(rows):
                if type(row) is dict and row.get('name') == module_name and row.get('source_root') == source_root:
                    return ordinal, self._declaration(number, '/module_mappings/' + str(ordinal))
        names = unit.get('module_names')
        if type(names) is list and unit.get('namespace_root') == source_root:
            for ordinal, name in enumerate(names):
                if name == module_name:
                    return ordinal, self._declaration(number, '/module_names/' + str(ordinal))
        raise ValueError('declared package candidate has no exact original unit-mapping provenance')

    def _module_ports(self, body):
        facts = self.resolver.modules._index(body)
        if facts.module is None or len(body.graph.frame_roots) != 1 or body.namespace.source is not None:
            return None
        frame = body.graph.frame_roots[0].entry
        for value in (frame.locals_namespace, frame.globals_namespace):
            if type(body.index.values.get(value)) is not ReferenceValue:
                raise ValueError('source module namespace is not its original reference')
        if type(body.index.values.get(frame.world)) is not StateVersionValue:
            raise ValueError('source module entry world is not its original state value')
        return frame

    def _record(self, *, body, origin, operation, source, spelling, level,
            resolved, module, world, normal, namespaces, conditions,
            request=None, package=None, package_mapping=None):
        frame = self._module_ports(module)
        if frame is None:
            self._note(origin, 'CONTEXTUAL_IMPORT_SOURCE_MODULE_INTERFACE_UNAVAILABLE',
                'A declared module candidate has no retained original L/G/world module initializer interface.')
            return None
        source_number, _ = self._unit(body)
        module_number, _ = self._unit(module)
        module_ns = GraphNamespace.from_source(module.namespace)
        module_g = Point(module_ns, frame.globals_namespace, None)
        module_world = Point(module_ns, frame.world, None)
        at = lambda node: Point(origin.namespace, node, origin.context)
        operands = tuple(at(use) for use in namespaces)
        if request is not None:
            operands += tuple(at(use) for use in (request.requested_use, request.level_use, request.fromlist_use))
        operands += (at(world), module_g, module_world)
        selected = [replace(Condition.from_enum(condition), subject=origin, operands=operands)
            for condition in conditions]
        selected.append(Condition('scar.contextual-import',
            'IMPORT_NORMAL_RESULT_DENOTES_DECLARED_SOURCE_MODULE', origin,
            operands=(at(operation), *operands)))
        sources = [origin.namespace.graph, replace(origin.namespace.graph, pointer='#'),
            module.namespace.graph,
            self._declaration(source_number), self._declaration(module_number)]
        if request is not None:
            selected.append(Condition('scar.contextual-import',
                'FROM_IMPORT_NORMAL_RETURNS_RESOLVED_MODULE', origin,
                operands=(at(operation), *operands)))
        if package_mapping is not None:
            ordinal, mapping_source = package_mapping
            selected.append(Condition('scar.contextual-import',
                'CURRENT_IMPORT_PACKAGE_CONTEXT_MATCHES_DECLARED_CANDIDATE', origin,
                ordinal, operands))
            sources.append(mapping_source)
        evidence = Evidence('CONTEXTUAL_IMPORT_DECLARED_MODULE_SOURCE_V1',
            tuple(dict.fromkeys(sources)), tuple(dict.fromkeys((origin, at(operation), *operands))))
        return ContextualModuleImport(origin, operation, source, spelling, level, resolved,
            module.namespace, frame.locals_namespace, frame.globals_namespace, frame.world,
            world, normal, tuple(dict.fromkeys(selected)), (evidence,), request, package)

    def sources(self, origin):
        if type(origin) is not Point or type(origin.node) is not ValueID or origin.namespace.source_body is None:
            raise TypeError('contextual import origin must be an original source IMPORT_NAME normal Value point')
        origin.__post_init__()
        self._point, self._frontiers = origin, {}
        resolver = self.resolver
        records = {}
        request_source = None

        def admit(record, maximum):
            if record not in records and len(records) >= maximum:
                raise OverflowError('contextual import source candidate budget exhausted')
            records[record] = None
        try:
            resolver.owner._check()
            body = resolver._body(origin.namespace)
            facts = resolver.modules._index(body)
            operation = facts.normal_producers.get(origin.node)
            if operation is None or facts.protocols.get(operation) not in ('IMPORT_NAME_314', 'FUNCTION_IMPORT_NAME'):
                raise ValueError('contextual import origin is not an original IMPORT_NAME normal result')
            source_links = [row for row in facts.links if row.operation == operation]
            if len(source_links) != 1:
                raise ValueError('import request lacks its unique original source link')
            request_source = source_links[0].source
            actual = body.index.operations[operation]
            worlds = [use for use in actual.uses if type(body.index.values[use.value]) is StateVersionValue]
            if len(worlds) != 1:
                raise ValueError('import request lost its unique original world Use')
            world = worlds[0].id
            normal = normal_reference(body.index, operation, blocks=resolver.owner._namespace_for(body).blocks)
            if normal is None or normal[1] != origin.node:
                raise ValueError('import origin changed its original normal payload0 port')
            count = 3 if facts.protocols[operation] == 'IMPORT_NAME_314' else 2
            namespaces = tuple(use.id for use in actual.uses[:count])
            maximum = resolver.session.budget.max_entry_projections
            from_selected = facts.from_requests.get(operation)
            if from_selected is None:
                # A template's empty requirements mean no CALL support; they
                # never replace origin.context with a closed root selection.
                for route in resolver.modules.imports_for_result(body, origin.node, requirements=()):
                    resolver.owner._check()
                    module = resolver._body(GraphNamespace.from_source(route.module_body))
                    record = self._record(body=body, origin=origin, operation=operation,
                        source=route.source, spelling=route.requested, level=0,
                        resolved=route.returned_module, module=module, world=world,
                        normal=normal[0], namespaces=namespaces, conditions=route.conditions)
                    if record is not None:
                        admit(record, maximum)
            else:
                node, source, request = from_selected
                if type(node) is not ast.ImportFrom or type(request) is not StaticImportFromRequest:
                    raise ValueError('FROM source index changed its original request types')
                spelling, level = node.module or '', node.level
                for use, expected in ((request.requested_use, spelling), (request.level_use, level)):
                    site = body.index.use_site(use)
                    value = body.index.values[site.use.value]
                    if (site.operation != operation or type(value.origin) is not LiteralOrigin
                            or type(value.origin.value) is not type(expected) or value.origin.value != expected):
                        raise ValueError('FROM request spelling/level differs from original source/compiler operands')
                fromlist = body.index.use_site(request.fromlist_use)
                if fromlist.operation != operation or request.fromlist_constant.source != source:
                    raise ValueError('FROM request changed its original code-constant fromlist endpoint')
                importer_number, unit = self._unit(body)
                choices = []
                if level == 0:
                    resolved, status = _resolve_from(spelling, 0, None)
                    if status == 'OK':
                        choices.append((resolved, None, None))
                    else:
                        self._note(origin, 'CONTEXTUAL_IMPORT_ABSOLUTE_REQUEST_UNSUPPORTED', status)
                else:
                    issues = []
                    mappings = _unit_mappings(unit, issues)
                    for issue in issues:
                        self._note(origin, 'CONTEXTUAL_IMPORT_DECLARED_PACKAGE_MAPPING_OPEN',
                            issue['kind'] + ': ' + issue.get('detail', 'Original source mapping is incomplete.'))
                    contexts = _importer_contexts(unit, mappings)
                    if not contexts:
                        self._note(origin, 'CONTEXTUAL_IMPORT_PACKAGE_CANDIDATE_UNAVAILABLE',
                            'No supplied source-unit mapping provides a declared package candidate; runtime __package__ is not guessed.')
                    for context in contexts:
                        resolver.owner._check()
                        package = context['package_context']
                        resolved, status = _resolve_from(spelling, level, package)
                        if status != 'OK':
                            self._note(origin, 'CONTEXTUAL_IMPORT_RELATIVE_CANDIDATE_' + status,
                                'Declared package ' + repr(package) + ' cannot resolve original level ' + str(level)
                                + '; this does not establish the runtime package context.')
                            continue
                        mapping = self._mapping(importer_number, unit,
                            context['module_name'], context['source_root'])
                        choices.append((resolved, package, mapping))
                from scar.ir.program.static_calls import MODULE_FLOW_BASE_CONDITIONS
                for resolved, package, mapping in tuple(dict.fromkeys(choices)):
                    candidates = 0
                    for module in resolver._module_bodies(resolved):
                        resolver.owner._check()
                        candidates += 1
                        record = self._record(body=body, origin=origin, operation=operation,
                            source=source, spelling=spelling, level=level, resolved=resolved,
                            module=module, world=world, normal=normal[0], namespaces=namespaces,
                            conditions=MODULE_FLOW_BASE_CONDITIONS[:4], request=request,
                            package=package, package_mapping=mapping)
                        if record is not None:
                            admit(record, maximum)
                    if not candidates:
                        self._note(origin, 'CONTEXTUAL_IMPORT_SOURCE_MODULE_UNAVAILABLE',
                            'No supplied canonical source module matches declared candidate ' + resolved + '.')
            if not records:
                self._note(origin, 'CONTEXTUAL_IMPORT_SOURCE_SET_UNCONNECTED',
                    'No supported declared source-module candidate supplies this exact original import result/context.')
            # Retain a missing source/compiler request adapter as that exact
            # gap, rather than replacing it with a generic absent candidate.
            for frontier in resolver.modules._permanent_frontiers.get(body.namespace, ()):
                if frontier.source is None or frontier.source == request_source:
                    self._note(origin, frontier.kind, frontier.detail)
        except (OverflowError, TimeoutError) as error:
            self._note(origin, 'CONTEXTUAL_IMPORT_SOURCE_BUDGET', type(error).__name__ + ': ' + str(error))
        self._note(origin, 'CONTEXTUAL_IMPORT_LOADER_AND_STATE_OPEN',
            'Import hooks/cache, package selection, source initialization and mutable module state remain conditional; known candidates do not close the source set.')
        return tuple(records)

    def validate(self, record):
        if type(record) is not ContextualModuleImport:
            raise TypeError('expected ContextualModuleImport')
        record.__post_init__()
        replay = self.sources(record.origin)
        if any(frontier.kind == 'CONTEXTUAL_IMPORT_SOURCE_BUDGET' for frontier in self.frontiers_for(record.origin)):
            raise OverflowError('fresh contextual import replay did not complete its bounded source demand')
        if record not in replay:
            raise ValueError('contextual module import differs from fresh original request/declaration/ports')
        return True


def module_import_sources(resolver, origin):
    """Convenience API retaining one resolver-owned diagnostics collector."""
    collector = getattr(resolver, '_contextual_import_sources', None)
    if collector is None:
        collector = ContextualImportSources(resolver)
        resolver._contextual_import_sources = collector
    elif type(collector) is not ContextualImportSources or collector.resolver is not resolver:
        raise ValueError('resolver contextual import collector belongs to another input owner')
    return collector.sources(origin)


# Conditional module bindings reached through the shared receiver graph.
#
# An import result supplies a guarded source-module interpretation. The snapshot
# of its original normal initializer return is not the contents at a later read;
# that preservation premise retains all participating worlds and known writers.
def _need(ok, detail):
    if not ok:
        raise ValueError('contextual module binding: ' + detail)


@dataclass(frozen=True, slots=True)
class ContextualModuleBindingProjection:
    read: StaticAttributeAccess | StaticImportFromRead
    key: str
    selector: StaticAttributeSelector | None
    import_source: ContextualModuleImport
    receiver_path: tuple[Edge, ...]
    source_module: SourceBodyNamespace
    snapshot_operation: OperationID
    snapshot_world: UseID
    namespace_binding: NamespaceBindingWitness
    consumer_path: tuple[ControlEdgeReference, ...]
    consumer_operations: tuple[OperationID, ...]
    known_writes: tuple[Point, ...]
    source: Point
    target: Point
    conditions: tuple[Condition, ...]
    evidence: tuple[Evidence, ...]

    def __post_init__(self):
        _need(type(self.read) in (StaticAttributeAccess, StaticImportFromRead), 'original field-read required')
        for value, kind in ((self.import_source, ContextualModuleImport),
                (self.source_module, SourceBodyNamespace), (self.snapshot_operation, OperationID),
                (self.snapshot_world, UseID), (self.namespace_binding, NamespaceBindingWitness),
                (self.source, Point), (self.target, Point)):
            _need(type(value) is kind, 'original typed ' + kind.__name__ + ' required')
            value.__post_init__()
        _need(type(self.key) is str and bool(self.key), 'known actual lookup key required')
        if type(self.read) is StaticAttributeAccess:
            _need(type(self.selector) is StaticAttributeSelector and self.selector.lookup_name == self.key,
                'Attribute lookup key must come from its original lexical selector')
        else:
            _need(self.selector is None and self.key == self.read.field, 'FROM key changed its literal selector')
        for rows, kind in ((self.receiver_path, Edge), (self.consumer_path, ControlEdgeReference),
                (self.consumer_operations, OperationID), (self.known_writes, Point),
                (self.conditions, Condition), (self.evidence, Evidence)):
            _need(type(rows) is tuple and all(type(row) is kind for row in rows),
                'immutable original path/condition/evidence records required')
        binding = self.namespace_binding
        _need(self.source_module == self.import_source.module
            and self.source.namespace == GraphNamespace.from_source(self.source_module)
            and self.source.context is None and self.source.node == binding.write.payload
            and self.target.namespace == GraphNamespace.from_source(self.read.body)
            and self.target.node == self.read.result
            and (binding.query_operation, binding.query_world, binding.query_owner, binding.key) ==
                (self.snapshot_operation, self.snapshot_world, self.import_source.globals_namespace, self.key)
            and binding.equivalent_owners == tuple(dict.fromkeys((self.import_source.locals_namespace,
                self.import_source.globals_namespace))), 'snapshot/payload/read changed original endpoints')
        receiver = Point(self.target.namespace, self.read.receiver, self.target.context)
        _need(bool(self.receiver_path) and self.receiver_path[0].source == self.import_source.origin
            and self.receiver_path[-1].target == receiver, 'import origin must reach the actual receiver Use')
        cursor, seen = self.import_source.origin, {self.import_source.origin}
        for edge in self.receiver_path:
            _need(edge.source == cursor and edge.target not in seen, 'receiver path is disconnected or self-seeding')
            _need(all(condition in self.conditions for condition in edge.conditions)
                and all(item in self.evidence for item in edge.evidence), 'receiver path evidence was dropped')
            cursor = edge.target; seen.add(cursor)
        _need(bool(self.evidence) and all(condition in self.conditions for condition in self.import_source.conditions)
            and all(item in self.evidence for item in self.import_source.evidence), 'import conditions/evidence were dropped')

    def dependency_edge(self):
        return Edge(self.source, self.target, K.STATE_VALUE, M.REFERENCE,
            self.evidence, self.conditions, Strength.MAY)


class ContextualModuleBindingProvider:
    relation_kinds = frozenset((K.STATE_VALUE,))

    def __init__(self, resolver):
        self.resolver = resolver
        self.imports = ContextualImportSources(resolver)
        self._active = set()
        self._known = OrderedDict()
        self._counts = Counter()
        self._last = None

    def _read(self, point):
        if (type(point) is not Point or type(point.node) is not ValueID
                or point.namespace.source_body is None):
            return None
        body = self.resolver._body(point.namespace)
        if type(body.index.values.get(point.node)) is not ReferenceValue:
            return None
        facts = self.resolver.modules._index(body)
        read = facts.reads.get(point.node)
        return (body, facts, read) if type(read) in (StaticAttributeAccess, StaticImportFromRead) else None

    def _key(self, body, read):
        if type(read) is StaticImportFromRead:
            value = body.index.use_site(read.key).use.value
            origin = body.index.values[value].origin
            _need(type(origin) is LiteralOrigin and origin.value == read.field,
                'FROM member key is not its actual original string literal')
            return read.field, None
        context = body.context
        saved = getattr(context, '_contextual_module_selector_inputs', None)
        if saved is None or saved[0] is not context.tree or saved[1] is not context.semantic:
            text = context.docs['source']['text']
            original = _source_lookup(context.semantic, body.metadata.source.path, body.metadata.source.fingerprint)
            facts = prepare_attribute_selector_source(context.tree, text, body.metadata.source.path)
            nodes = {}
            for node in ast.walk(context.tree):
                if type(node) is ast.Attribute:
                    source = original(node)
                    if source is not None:
                        _need(source not in nodes, 'ambiguous original Attribute source')
                        nodes[source] = node
            saved = context.tree, context.semantic, text, original, facts, nodes
            context._contextual_module_selector_inputs = saved
        node = saved[5].get(read.source)
        _need(node is not None, 'Attribute read lost its exact original source node')
        selector = derive_attribute_selector(node, source_tree=context.tree, source=read.source,
            source_text=saved[2], source_for_node=saved[3], source_facts=saved[4])
        return selector.lookup_name, selector

    def _receiver_paths(self, walk, origin):
        by_target = defaultdict(list)
        for edge in walk.edges:
            by_target[edge.target].append(edge)
        root, = walk.roots
        pending = deque(((root, (), frozenset((root,))),))
        steps = 0
        while pending:
            self.resolver.owner._check()
            point, reverse, ancestors = pending.popleft(); steps += 1
            if steps > self.resolver.owner.budget.max_resolution_steps:
                raise OverflowError('module receiver-path reconstruction budget exhausted')
            if point == origin:
                yield tuple(reversed(reverse))
                continue
            for edge in by_target.get(point, ()):
                if edge.source not in ancestors:
                    pending.append((edge.source, (*reverse, edge), ancestors | {edge.source}))

    def _snapshots(self, module):
        overlay = module.module
        _need(overlay is not None, 'import source must retain its actual MODULE overlay')
        partial = (getattr(overlay, 'remaining', None) is not None
            or getattr(overlay, 'unexpanded_body', None) is not None
            or any('REQUIRED_EXPANSION' in str(getattr(row.kind, 'value', row.kind))
                   for row in overlay.frontiers))
        if partial:
            return None
        dependencies = self.resolver.owner._namespace_for(module)
        snapshots = []
        for operation in module.graph.operations:
            if (operation.terminator is not None and operation.terminator.kind is TerminatorKind.RETURN
                    and dependencies.owners[operation.id][0] == module.graph.root):
                worlds = [use.id for use in operation.uses
                    if type(module.index.values[use.value]) is StateVersionValue]
                _need(len(worlds) == 1, 'module RETURN snapshot lost its unique original world')
                snapshots.append((operation.id, worlds[0]))
        return dependencies, tuple(snapshots)

    def _consumer_corridor(self, body, facts, read, imported, key, context):
        start = body.index.regions[body.graph.root].entry
        if (imported.origin.namespace == GraphNamespace.from_source(body.namespace)
                and imported.origin.context == context):
            # This CFG corridor is usable only within the same selected context.
            # Otherwise the receiver path carries the interprocedural ordering.
            operation = body.index.operations[imported.operation]
            start = operation.terminator.successors[imported.normal.successor_index].block
        return facts.paths.find_path(start, read.operation, key, ())

    def _projection(self, body, facts, read, query, key, selector, imported, path, module,
                    dependencies, snapshot, binding, consumer):
        dependencies.validate_binding(binding)
        source_ns = GraphNamespace.from_source(imported.module)
        source = Point(source_ns, binding.write.payload, None)
        payload = module.index.use_site(binding.write.payload).use.value
        _need(type(module.index.values[payload]) is ReferenceValue, 'snapshot binding has no original reference payload')
        snapshot_op, snapshot_world = snapshot
        consumer_path, consumer_operations = consumer
        conditions = dict.fromkeys((*imported.conditions,
            *(condition for edge in path for condition in edge.conditions)))
        anchors = dict.fromkeys((source, query, imported.origin))
        sources = dict.fromkeys((query.namespace.graph, source_ns.graph,
            replace(query.namespace.graph, pointer='#'), replace(source_ns.graph, pointer='#')))
        known_writes = {}

        def point(namespace, node, context):
            result = Point(namespace, node, context); anchors[result] = None
            return result

        def ports(original, namespace, identity, context, *, full=True):
            operation = original.index.operations[identity]
            point(namespace, identity, context)
            for use in operation.uses:
                if full or type(original.index.values[use.value]) is StateVersionValue:
                    point(namespace, use.id, context)
            for value in operation.results:
                if full or type(original.index.values[value]) is StateVersionValue:
                    point(namespace, value, context)
            if operation.terminator is not None:
                for edge in operation.terminator.successors:
                    for value in original.index._block_records[edge.block].arguments:
                        if type(original.index.values[value]) is StateVersionValue:
                            point(namespace, value, context)

        point(query.namespace, read.receiver, query.context)
        point(query.namespace, read.world, query.context)
        if type(read) is StaticImportFromRead:
            point(query.namespace, read.key, query.context)
        for node in (imported.locals_namespace, imported.globals_namespace, imported.entry_world):
            point(source_ns, node, None)
        for node in (binding.query_owner, binding.write.owners[0], binding.write.keys[0],
                binding.write.world, snapshot_world):
            point(source_ns, node, None)
        ports(module, source_ns, binding.write.operation, None)
        ports(module, source_ns, snapshot_op, None)
        ports(body, query.namespace, read.operation, query.context)
        origin_body = self.resolver._body(imported.origin.namespace)
        ports(origin_body, imported.origin.namespace, imported.operation, imported.origin.context)
        if imported.request is not None:
            for node in (imported.request.requested_use, imported.request.level_use,
                    imported.request.fromlist_use, imported.request.fromlist_constant.operation):
                point(imported.origin.namespace, node, imported.origin.context)
        for edge in path:
            anchors[edge.source] = None; anchors[edge.target] = None
            sources[edge.source.namespace.graph] = None; sources[edge.target.namespace.graph] = None
        for identity in binding.intervening_operations:
            ports(module, source_ns, identity, None, full=False)
        for identity in consumer_operations:
            ports(body, query.namespace, identity, query.context, full=False)
        # Retain known write interfaces in every body/context participating in
        # this receiver path. Their aliases/commit/effects are not inferred.
        contexts = dict.fromkeys(((query.namespace, query.context),
            *((endpoint.namespace, endpoint.context) for edge in path for endpoint in (edge.source, edge.target))))
        for namespace, context in contexts:
            self.resolver.owner._check()
            if namespace.source_body is None:
                continue
            original = self.resolver._body(namespace)
            original_facts = self.resolver.modules._index(original)
            local = self.resolver.owner._namespace_for(original)
            writes = dict.fromkeys((*local.writes,
                *(write.operation for write in original_facts.writes)))
            for identity in writes:
                writer = point(namespace, identity, context); known_writes[writer] = None
                ports(original, namespace, identity, context)
        operands = tuple(anchors)

        def condition(name, subject=query):
            conditions[Condition('scar.contextual-module-binding', name, subject, operands=operands)] = None

        condition('STANDARD_IMPORT_INITIALIZER_USES_MODULE_DICT')
        condition('SOURCE_MODULE_INITIALIZER_COMPLETED_NORMALLY')
        if type(read) is StaticImportFromRead:
            condition('IMPORT_FROM_LOOKUP_USES_NAMESPACE_BINDING')
            condition('IMPORT_FROM_FALLBACKS_NOT_SELECTED')
        else:
            condition('MODULE_LOOKUP_USES_SOURCE_MODULE_DICT')
            condition('MODULE_LOOKUP_HOOKS_NOT_SELECTED')
        condition('SOURCE_SNAPSHOT_BINDING_PRESERVED_AT_CURRENT_READ')
        for original in binding.conditions:
            conditions[replace(Condition.from_enum(original), subject=query, operands=operands)] = None
        for namespace, context, controls in ((source_ns, None, (binding.write.normal_edge, *binding.path)),
                (query.namespace, query.context, (*consumer_path, read.normal)),
                (imported.origin.namespace, imported.origin.context, (imported.normal,))):
            for control in controls:
                conditions[Condition('canonical', 'SUCCESSOR_SELECTED',
                    Point(namespace, control.operation, context), control.successor_index)] = None
        evidence = tuple(dict.fromkeys((*imported.evidence,
            *(item for edge in path for item in edge.evidence),
            Evidence('CONTEXTUAL_MODULE_BINDING_FROM_SHARED_REFERENCE_SOURCE_V1', tuple(sources), operands))))
        return ContextualModuleBindingProjection(read, key, selector, imported, path, imported.module,
            snapshot_op, snapshot_world, binding, consumer_path, consumer_operations,
            tuple(known_writes), source, query, tuple(conditions), evidence)

    def _derive(self, query):
        if (type(query) is not Point or type(query.node) is not ValueID
                or query.namespace.source_body is None):
            return (), (), True
        try:
            self.resolver.owner._check()
            selected = self._read(query)
        except (OverflowError, TimeoutError) as error:
            return (), (Frontier(query, 'CONTEXTUAL_MODULE_BINDING_BUDGET',
                type(error).__name__ + ': ' + str(error)),), False
        if selected is None:
            return (), (), True
        if query in self._active:
            return (), (Frontier(query, 'CONTEXTUAL_MODULE_BINDING_SOURCE_CYCLE_OPEN',
                'An active field read cannot establish its own receiver module or snapshot binding.'),), False
        self._active.add(query)
        rows, notes = {}, {}
        def note(kind, detail):
            notes[Frontier(query, kind, detail)] = None
        note('CONTEXTUAL_MODULE_BINDING_SCOPE_OPEN',
            'Import selection/cache/hooks, conditional initializer L/G alias, lookup and preservation across all original worlds remain open.')
        try:
            body, facts, read = selected
            key, selector = self._key(body, read)
            if key is None:
                note('CONTEXTUAL_MODULE_LOOKUP_NAME_UNRESOLVED', selector.unresolved_reason)
                return (), tuple(notes), False
            root = Point(query.namespace, read.receiver, query.context)
            walk = walk_reference_sources(self.resolver, root)
            notes.update((item, None) for item in walk.frontiers)
            for origin in walk.points:
                if type(origin.node) is not ValueID or origin.namespace.source_body is None:
                    continue
                self.resolver.owner._check()
                origin_body = self.resolver._body(origin.namespace)
                origin_facts = self.resolver.modules._index(origin_body)
                producer = origin_facts.normal_producers.get(origin.node)
                if origin_facts.protocols.get(producer) not in ('IMPORT_NAME_314', 'FUNCTION_IMPORT_NAME'):
                    continue
                for imported in self.imports.sources(origin):
                    module = self.resolver._body(GraphNamespace.from_source(imported.module))
                    snapshots = self._snapshots(module)
                    if snapshots is None:
                        note('CONTEXTUAL_MODULE_INITIALIZER_PARTIAL',
                            'The selected source module retains an unexpanded suffix; no complete initialized snapshot is invented.')
                        continue
                    dependencies, terminals = snapshots
                    if not terminals:
                        note('CONTEXTUAL_MODULE_INITIALIZER_NORMAL_RETURN_UNAVAILABLE',
                            'The selected source MODULE has no original root normal RETURN snapshot.')
                    consumer = self._consumer_corridor(body, facts, read, imported, key, query.context)
                    if consumer is None:
                        note('CONTEXTUAL_MODULE_READ_CONTROL_PATH_UNAVAILABLE',
                            'No original conditional CFG corridor reaches this field read from its entry/import boundary.')
                        continue
                    for path in self._receiver_paths(walk, origin):
                        for snapshot_op, world in terminals:
                            bindings = dependencies.bindings_at(imported.globals_namespace, key, snapshot_op, world,
                                equivalent_owners=tuple(dict.fromkeys((imported.locals_namespace, imported.globals_namespace))))
                            for binding in bindings:
                                if len(rows) >= self.resolver.owner.budget.max_candidates_per_call:
                                    raise OverflowError('contextual module binding projection budget exhausted')
                                projection = self._projection(body, facts, read, query, key, selector,
                                    imported, path, module, dependencies, (snapshot_op, world), binding, consumer)
                                rows[projection] = None
                notes.update((item, None) for item in self.imports.frontiers_for(origin))
        except (OverflowError, TimeoutError) as error:
            note('CONTEXTUAL_MODULE_BINDING_BUDGET', type(error).__name__ + ': ' + str(error))
        finally:
            self._active.remove(query)
        for projection in rows:
            self._known[projection] = None; self._known.move_to_end(projection)
        while len(self._known) > self.resolver.owner.budget.max_candidates_per_call:
            self._known.popitem(last=False)
        self._counts['read_queries'] += 1; self._counts['projections'] += len(rows)
        return tuple(rows), tuple(notes), False

    def projections_for(self, point):
        rows, notes, enumerated = self._derive(point)
        self._last = point, rows, notes
        return rows

    def incoming(self, point):
        rows, notes, enumerated = self._derive(point)
        self._last = point, rows, notes
        return Neighborhood(tuple(row.dependency_edge() for row in rows), notes, enumerated)

    def outgoing(self, point):
        if point.namespace.source_body is None or type(point.node) is not UseID:
            return Neighborhood()
        body = self.resolver._body(point.namespace)
        if body.module is None:
            return Neighborhood()
        dependencies = self.resolver.owner._namespace_for(body)
        if not any(write.payload == point.node for write in dependencies.writes.values()):
            return Neighborhood()
        rows = [row for row in self._known if row.source == point]
        for row in rows:
            self.validate_projection(row)
        return Neighborhood(tuple(row.dependency_edge() for row in rows),
            (Frontier(point, 'CONTEXTUAL_MODULE_BINDING_CONSUMERS_NOT_ENUMERATED',
                'Only bounded already discovered projections are mirrored; other contexts and consumers were not enumerated.'),), False)

    def validate_projection(self, projection):
        _need(type(projection) is ContextualModuleBindingProjection, 'typed module projection required')
        projection.__post_init__()
        selected = self._read(projection.target)
        _need(selected is not None and selected[2] == projection.read, 'read differs from original source/ports')
        body, facts, read = selected
        key, selector = self._key(body, read)
        _need((key, selector) == (projection.key, projection.selector), 'lookup name differs from fresh lexical source')
        self.imports.validate(projection.import_source)
        for edge in projection.receiver_path:
            _need(self.resolver._transport(edge), 'receiver path is not legal original reference transport')
            fresh = self.resolver.session.graph().incoming(edge.target, kinds=(edge.kind,))
            _need(edge in fresh.edges, 'receiver path differs from fresh shared-graph evidence')
        module = self.resolver._body(GraphNamespace.from_source(projection.source_module))
        snapshots = self._snapshots(module)
        _need(snapshots is not None, 'partial module cannot supply a completed initializer snapshot')
        dependencies, terminals = snapshots
        snapshot = projection.snapshot_operation, projection.snapshot_world
        _need(snapshot in terminals, 'snapshot is not the original root MODULE RETURN/world')
        dependencies.validate_binding(projection.namespace_binding)
        consumer = self._consumer_corridor(body, facts, read, projection.import_source, key,
            projection.target.context)
        _need(consumer == (projection.consumer_path, projection.consumer_operations),
            'consumer control corridor differs from original CFG')
        expected = self._projection(body, facts, read, projection.target, key, selector,
            projection.import_source, projection.receiver_path, module, dependencies, snapshot,
            projection.namespace_binding, consumer)
        _need(expected == projection, 'projection payload/owners/worlds/guards/evidence differ from original inputs')
        self._counts['fresh_validations'] += 1
        return True

    def clear(self):
        self._active.clear(); self._known.clear(); self._last = None

    def measurements(self):
        return {**dict(self._counts), 'retained_projection_witnesses': len(self._known),
            'projection_retention_ceiling': self.resolver.owner.budget.max_candidates_per_call,
            'full_consumer_scope_enumerated': False}


__all__ = [
    'ContextualModuleBindingProjection',
    'ContextualModuleBindingProvider',
    'ContextualModuleImport',
    'ContextualImportSources',
    'module_import_sources',
]
