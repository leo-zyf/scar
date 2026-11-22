"""Class bindings."""

from collections import defaultdict
from scar.analysis.source_address_paths import AddressResolver
from scar.analysis.source_reference_slice import ReferenceSlicer, source_port, DECLARED
from scar.analysis.source_namespace_references import NamespaceReferenceResolver
from scar.ir.program import (
    normal_links as F,
    normal_targets as T,
    content_routes as A,
    class_ports as B,
    declared_contexts as D,
    class_ports as X,
    class_ports as C,
    source_productions as P,
    normal_graph as G,
)
from scar.ir.program.namespace_routes import NamespaceOriginKind
from scar.ir.program.control import LanguageCompletionKind as Completion
from scar.ir.program.model import ValueKind, ValueType
from scar.ir.program.class_ports import ClassTypeParameterProjection
from scar.ir.program.ids import UseID
from scar.ir.program.dependencies import (
    DependencyPoint as Point,
    DependencyCondition as Condition,
    DependencyKind as Kind,
    DependencyMeaning as Meaning,
)


# Project original class base-expression operands and their source paths.
#
# UnitFacts.Class.bases retains the evaluated source operands of the class-build
# call after its hidden prefix, before runtime __mro_entries__/metaclass/MRO
# selection. A bases_pack is the whole packed class-build positional ABI, which
# also contains hidden body/name operands; it cannot be enumerated as bases.
_CONDITIONS = (*DECLARED,
    'SOURCE_CLASS_CREATION_SCOPE_AND_BASE_INPUT_VERSIONS_SELECTED',
    'SOURCE_CLASS_RESULT_AND_NAMESPACE_VERSIONS_SELECTED',
    'BASE_OPERANDS_ARE_EVALUATED_SOURCE_EXPRESSIONS_NOT_RESOLVED_RUNTIME_BASES',
    'METACLASS_MRO_ENTRIES_AND_RUNTIME_MRO_SELECTION_NOT_ESTABLISHED',
)
_SLOT_CONDITIONS = ('BASE_CLASS_CANDIDATE_REQUIRES_SELECTED_ADDRESS_AND_NAMESPACE_PATH_GUARDS',
    'NO_CLASS_IDENTITY_FROM_SELECTOR_CALL_IMPORT_OR_IDENTIFIER_SPELLING')


def _port(unit, local):
    if type(local) is not int or not 0 <= local < len(unit.ports):
        raise ValueError('class base operand is outside its original source port table')
    return source_port(unit, local)


def project_class_bases(snapshot, source_classes):
    """Project requested classes only, preserving ordered original operands.

    Definitions are sorted by original unit/port coordinates for stable output
    even when the caller supplies a set. Class-definition candidates reference
    original ports; they do not imply resolved runtime inheritance. Namespace
    writes/exports add conditional sources with their own evidence; current
    slot versions and imported provider selection stay open. No class is
    selected by identifier spelling and no relation table is scanned.
    """
    points = tuple(source_classes)
    if any(type(point) is not T.GlobalPort for point in points):
        raise TypeError('class base projection requires original source-class GlobalPorts')
    namespace_resolver = NamespaceReferenceResolver(snapshot)
    slicer = namespace_resolver.slicer; requested = []
    for point in dict.fromkeys(points):
        unit, local = slicer.local(point)
        matches = [(ordinal, cls) for ordinal, cls in enumerate(unit.classes) if cls.value == local]
        if len(matches) != 1:
            raise ValueError('class base source is not one original class declaration')
        ordinal, cls = matches[0]
        requested.append((unit.unit, local, point, unit, ordinal, cls))
    requested.sort(key=lambda row: (row[0], row[1]))
    owners = {}; resolvers = {}; declarations = []; slots = []; frontiers = []; frontier_ids = set()

    def frontier(owner, slot, reason):
        if (owner, slot, reason) not in frontier_ids:
            frontier_ids.add((owner, slot, reason)); frontiers.append(B.ClassBaseFrontier(owner, slot, reason))

    def paths_for(resolver, local):
        # Source transport often has several copy wires per expression. The
        # ordinary resolver preserves its own state/depth/cycle frontiers.
        return resolver.paths(local, max_states=1000, max_depth=128)

    for _, _, point, unit, ordinal, cls in requested:
        if unit.unit not in owners:
            covered = defaultdict(set)
            for group in unit.scope_groups:
                for span in group.tables:
                    if span.table is F.FactTable.CLASSES:
                        for index in range(span.start, span.stop): covered[index].add(group.owner)
            owners[unit.unit] = covered
        creators = owners[unit.unit][ordinal]
        if len(creators) != 1:
            raise ValueError('class declaration has no unique original creation code scope')
        creator = next(iter(creators))
        if type(creator) is not F.CodeScopeRef or creator.unit != unit.unit:
            raise ValueError('class creation code scope belongs to a different source unit')
        # cls.body_scope selects the class suite. Base expressions belong to
        # the surrounding scope that owns this CLASSES fact, not that suite.
        if creator not in resolvers:
            resolvers[creator] = AddressResolver(snapshot.fact_store, unit, creator)
        resolver = resolvers[creator]; owner = len(declarations)
        bases = tuple(_port(unit, base) for base in cls.bases)
        pack = None if cls.bases_pack is None else _port(unit, cls.bases_pack)
        pack_paths = () if cls.bases_pack is None else paths_for(resolver, cls.bases_pack)
        no_explicit = not bases and pack is None
        declarations.append(B.ClassBaseDeclaration(point, slicer.fact(unit, 'classes', ordinal), creator,
            bases, pack, pack_paths, no_explicit, _CONDITIONS))
        if pack is not None:
            frontier(owner, None, 'PACKED_CLASS_BUILD_ABI_INCLUDES_HIDDEN_OPERANDS_BASE_ITEMS_UNRESOLVED')
            for path in pack_paths:
                if path.frontier is not None: frontier(owner, None, path.frontier)
        elif no_explicit:
            frontier(owner, None, 'NO_EXPLICIT_BASE_ARGUMENTS_DOES_NOT_ESTABLISH_RUNTIME_OBJECT_BASE_OR_MRO')
        for index, expression in enumerate(bases):
            paths = paths_for(resolver, cls.bases[index]); candidates = []
            namespace = namespace_resolver.resolve(expression)
            for path in paths:
                if (path.root_kind is A.AddressRootKind.CLASS_DEFINITION
                        and not path.selectors and path.frontier is None):
                    if path.root not in candidates: candidates.append(path.root)
            for origin in namespace.origins:
                state = namespace.states[origin.state]
                if origin.kind is NamespaceOriginKind.CLASS_DEFINITION and not state.selectors:
                    if state.port not in candidates: candidates.append(state.port)
            slot = len(slots)
            slots.append(B.ClassBaseSlot(owner, index, expression, paths, tuple(candidates), _SLOT_CONDITIONS, namespace))
            if not candidates:
                frontier(owner, slot, 'BASE_EXPRESSION_DECLARED_CLASS_SOURCE_UNRESOLVED')
            # Original address boundaries remain in paths. This pass reports
            # the additional namespace model's still-open fields separately.
            for value in namespace.frontiers: frontier(owner, slot, value.reason)
    return B.ClassBaseGraph(tuple(declarations), tuple(slots), tuple(frontiers))


# Project constructor class alternatives without rebinding or entering bodies.
#
# A compatible init receiver may belong to a subclass of the requested class.
# Original entry/step references and connection guards remain the authority for
# each alternative; shared source definitions do not identify runtime objects.
_GUARDS = (
    'PARENT_CONSTRUCTOR_ENTRY_AND_STAGE_ALTERNATIVE_SELECTED',
    'MRO_DESCRIPTOR_AND_OVERRIDE_SELECTION_NOT_ESTABLISHED',
    'CLASS_INPUT_IS_NOT_EXACT_RUNTIME_CLASS_OR_OBJECT_IDENTITY_PROOF',
    'SHARED_SOURCE_DEFINITION_NOT_SELECTED_RUNTIME_INVOCATION',
    'SOURCE_NEW_DOES_NOT_ESTABLISH_ALLOCATION_OR_INITIALIZATION',
)


def _index(rows, ordinal):
    if type(ordinal) is not int or not 0 <= ordinal < len(rows):
        raise ValueError('constructor context index is outside its original parent table')
    return rows[ordinal]


def constructor_context_inputs(graph):
    """Retain every normal source entry and its provable single-argument shape.

    Root.entry and frontier.entry select the original parent entries table,
    whose step retains source_class, evidence, original site and connection.
    No original stage, object reference, world or parameter route is replaced.
    """
    if type(graph) is not C.ConstructorInterfaceGraph:
        raise TypeError('constructor class inputs require the original typed interface graph')
    roots = []; frontiers = []; frontier_ids = set()

    def frontier(entry, reason):
        if (entry, reason) not in frontier_ids:
            frontier_ids.add((entry, reason)); frontiers.append(X.ConstructorClassInputFrontier(entry, reason))

    for ordinal, entry in enumerate(graph.entries):
        step = _index(graph.steps, entry.step)
        _index(graph.sites, step.site)
        link = step.link; connection = entry.connection
        if link.stage not in (T.ConstructionStage.SOURCE_NEW, T.ConstructionStage.SOURCE_INIT):
            frontier(ordinal, 'NON_SOURCE_CONSTRUCTOR_STAGE_HAS_NO_SOURCE_CLASS_BODY_ENTRY')
            continue
        if link.connection != connection:
            raise ValueError('constructor class entry changed its original stage function connection')
        if connection.activation is not T.FunctionActivation.NORMAL:
            frontier(ordinal, 'DEFERRED_BODY_CREATION_IS_NOT_BODY_ENTRY')
            continue
        if (type(link.source_class) is not T.GlobalPort
                or connection.body.unit != connection.source_callable.unit):
            raise ValueError('constructor class or source body lacks its original qualified identity')
        new = link.stage is T.ConstructionStage.SOURCE_NEW
        required = 'NEW_RECEIVES_SOURCE_CLASS' if new else 'NEW_RESULT_IS_REQUESTED_CLASS_INSTANCE'
        if required not in link.conditions or required not in connection.conditions:
            raise ValueError('constructor class compatibility lost its original stage guard')
        source = link.source_class if new else T.ConstructionPort(
            link.result, link.source_class, T.ConstructionPortKind.NEW_RESULT)
        kind = D.ClassValueKind.CLASS_OBJECT if new else D.ClassValueKind.INSTANCE
        inputs = []
        for route in connection.parameters:
            if route.kind in (T.RouteKind.PACK_POSITIONALS, T.RouteKind.PACK_KEYWORDS):
                # Unrelated argument packs remain in the parent interface;
                # they are not missing evidence for this known class source.
                if source in route.actuals:
                    frontier(ordinal, route.kind.value+'_IS_PACK_DEPENDENCY_NOT_SINGLE_CLASS_REFERENCE')
                continue
            if route.kind is T.RouteKind.CLASS_RECEIVER:
                if source in route.actuals:
                    frontier(ordinal, 'CLASS_RECEIVER_IS_TYPE_PROJECTION_NOT_ARGUMENT_REFERENCE')
                continue
            if route.kind is not T.RouteKind.ARGUMENT:
                if source in route.actuals:
                    frontier(ordinal, 'CLASS_SOURCE_REQUIRES_ARGUMENT_ROUTE_NOT_'+route.kind.value)
                continue
            if len(route.actuals) > 1:
                if source in route.actuals:
                    frontier(ordinal, 'MULTI_ACTUAL_ARGUMENT_ROUTE_IS_NOT_SINGLE_CLASS_REFERENCE')
                continue
            if route.actuals != (source,):
                continue
            if type(route.formal) is not T.GlobalPort or route.formal.unit != connection.body.unit:
                raise ValueError('constructor class input formal belongs to a different original body')
            value = D.ClassInput(route.formal, link.source_class, kind)
            if value not in inputs:
                inputs.append(value)
        if not inputs:
            frontier(ordinal, 'CONSTRUCTOR_CLASS_SOURCE_NOT_BOUND_BY_SINGLE_ARGUMENT_ROUTE')
        context = D.DeclaredContext(connection.source_callable, connection.body, tuple(inputs))
        extra = ('REQUESTED_CLASS_COMPATIBILITY_DOES_NOT_ESTABLISH_EXACT_RETURNED_CLASS',) if not new else (
            'SOURCE_CLASS_OBJECT_ARGUMENT_IS_NOT_ALLOCATED_INSTANCE',)
        roots.append(X.ConstructorClassInputRoot(ordinal, context,
            (*link.conditions, *connection.conditions, *_GUARDS, *extra)))
    return X.ConstructorClassInputGraph(tuple(roots), tuple(frontiers))


# Project retained values using original callback interfaces and ports.
def identify_class_type_parameters(source, context, step, operation):
    if type(step.instruction) is not P.ReadClassTypeParameters:
        return None
    interface=step.instruction.interface
    if (context.steps.get(step.id)!=step or interface.type_parameters is None
            or context.owner(step.region).source_scope!=interface.scope):
        raise ValueError('class type tuple projection lost its original callback selector/scope')
    reference=ValueType(ValueKind.REFERENCE)
    arguments=tuple(context.emitter.value(v)[1] for v in step.arguments)
    if (len(arguments)!=2 or arguments[0]!=reference
            or type(arguments[1]) is not ValueType or arguments[1].kind is not ValueKind.STATE
            or len(step.completions)!=1 or step.completions[0].kind is not Completion.FALLTHROUGH
            or step.completions[0].types!=(reference,arguments[1])):
        raise ValueError('class type tuple environment/result/world ABI differs')
    use=source.port(context.emitter.port(step.id,G.PortKind.ARGUMENT,0))
    if type(use.node) is not UseID:
        raise TypeError('class callback environment needs its original owned Use')
    before,after=context.worlds(step)
    return ClassTypeParameterProjection(operation,interface,context.value(step.arguments[0]),
        use.node,context.result(step,0),before,after)


def attach_class_type_parameters(state, projection):
    """Attach dependence only; no physical slot, alias or purity is inferred."""
    operation=Point(state.namespace,projection.operation,None)
    environment=state.view.source_port(projection.environment)
    result=state.view.source_port(projection.result)
    owned=Point(state.namespace,projection.environment_use,None)
    selected=Condition('scar.source-environment.value',
        'CURRENT_ACTIVATION_USES_SELECTED_CLASS_CALLBACK_ENVIRONMENT',operation,None,(environment,owned))
    state._edge(owned,result,Kind.PROJECTION,Meaning.DEPENDENCE,(selected,),
        'scar.source-environment.class-type-parameter-tuple.v1')
