"""Loop graph view."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from scar.ir.program.dependencies import GraphNamespace, DependencyFrontier, DependencyPoint
from scar.ir.program.ids import BlockID, ValueID, UseID, OperationID
from scar.ir.program.model import (
    BlockArgument,
    CompletionResult,
    LocalBindingValue,
    StateVersionValue,
    value_type,
)
from scar.ir.program.source_call_bindings import SourceBodyNamespace
from scar.ir.program.structured import (
    ControlEdgeReference,
    LoopBindingPort,
    LoopInterface,
    StructuredBodyModel,
)
from scar.ir.record_codec import decode
from scar.ir.v2.common import SourceReference
from scar.ir.v2.ids import OperationDefinitionID
from scar.analysis.program_dependencies import ProgramDependencyGraph
from scar.ir.program.source import LoweringMetadata
from scar.ir.program.source_references import SourceArtifactLocator


# Composable views of retained loop ports, without CFG or iteration copies.
#
# Header arguments are phi-like static ports, not runtime values or iteration
# identities. Forwarding a binding/reference does not preserve referent contents;
# a different SSA ID does not prove a different numerical value. Source nesting
# is lexical only. The owning region is the original body region, not a newly
# constructed LoopRegion or a complete loop-membership/exit closure.
class LoopViewUnavailable(ValueError):
    """A retained source body has no recorded structured-loop interface."""


def _need(ok, detail):
    if not ok:
        raise ValueError('loop template view: ' + detail)


@dataclass(frozen=True, slots=True)
class LoopSlotRoute:
    index: int
    formal: ValueID
    actual: UseID | CompletionResult
    actual_value: ValueID | None
    is_world: bool
    ssa_relation: str
    content_invariance: str = 'UNPROVED'

    def __post_init__(self):
        _need(type(self.index) is int and self.index >= 0 and type(self.formal) is ValueID
            and type(self.actual) in (UseID, CompletionResult) and type(self.is_world) is bool
            and type(self.ssa_relation) is str,
            'slot needs exact original index, formal, actual and world classification')
        if type(self.actual) is UseID:
            _need(type(self.actual_value) is ValueID and self.ssa_relation in
                ('INITIAL', 'FORWARDED', 'REDEFINED'), 'Use route lost its original value')
        else:
            _need(self.actual_value is None and self.ssa_relation == 'COMPLETION_OUTPUT',
                'a completion slot cannot be replaced by an invented SSA value')
        _need(self.content_invariance == 'UNPROVED', 'port structure cannot prove content invariance')


@dataclass(frozen=True, slots=True)
class LoopPortRoute:
    edge: ControlEdgeReference
    origin: BlockID
    target: BlockID
    slots: tuple[LoopSlotRoute, ...]

    def __post_init__(self):
        _need(type(self.edge) is ControlEdgeReference and type(self.origin) is BlockID
            and type(self.target) is BlockID and type(self.slots) is tuple
            and all(type(slot) is LoopSlotRoute for slot in self.slots), 'typed original control route required')
        self.edge.__post_init__()
        for slot in self.slots:
            slot.__post_init__()
        _need(tuple(slot.index for slot in self.slots) == tuple(range(len(self.slots)))
            and len({slot.formal for slot in self.slots}) == len(self.slots), 'header slot order changed')


@dataclass(frozen=True, slots=True)
class LoopTemplateView:
    namespace: GraphNamespace
    definition: OperationDefinitionID
    loop: LoopInterface
    initial_routes: tuple[LoopPortRoute, ...]
    backedge_routes: tuple[LoopPortRoute, ...]
    lexical_parent: SourceReference | None
    nested_sources: tuple[SourceReference, ...]
    content_invariance: str = 'UNPROVED'
    scope_complete: bool = False
    independent_loop_region: bool = False
    iterations_unrolled: bool = False

    def __post_init__(self):
        _need(type(self.namespace) is GraphNamespace and type(self.definition) is OperationDefinitionID
            and type(self.loop) is LoopInterface, 'original source namespace/definition/loop required')
        self.namespace.__post_init__(); self.loop.__post_init__()
        _need(self.namespace.source_body is not None
            and self.definition == self.namespace.source_body.definition
            and self.loop.region == self.namespace.root, 'loop changed its original owning body')
        for routes in (self.initial_routes, self.backedge_routes):
            _need(type(routes) is tuple and all(type(route) is LoopPortRoute for route in routes),
                'loop routes must be immutable original ports')
            for route in routes:
                route.__post_init__()
                _need(route.target == self.loop.header, 'route does not land at the original header')
        _need(tuple(route.edge for route in self.backedge_routes) == self.loop.backedges
            and not set(route.edge for route in self.initial_routes).intersection(self.loop.backedges),
            'initial and backedge routes changed their original partition')
        _need((self.lexical_parent is None or type(self.lexical_parent) is SourceReference)
            and type(self.nested_sources) is tuple
            and all(type(source) is SourceReference for source in self.nested_sources), 'typed lexical source references required')
        _need(self.content_invariance == 'UNPROVED' and self.scope_complete is False
            and self.independent_loop_region is False and self.iterations_unrolled is False,
            'loop interface view cannot assert content, coverage or runtime iteration proofs')


def _contains(outer, inner):
    start = lambda source: (source.start_line, source.start_column)
    end = lambda source: (source.end_line, source.end_column)
    return ((outer.path, outer.fingerprint) == (inner.path, inner.fingerprint)
        and start(outer) <= start(inner) and end(inner) <= end(outer)
        and (start(outer), end(outer)) != (start(inner), end(inner)))


def loop_template_views(body):
    """Expose every recorded loop and all its original header incoming slots.

    ``body`` is a normal session-loaded, receipt-checked source body. No source
    parsing, new ProgramIndex, CFG construction, target execution or caching is
    performed here. Missing structured evidence raises instead of meaning zero
    loops; even an empty recorded collection is not a whole-source coverage proof.
    """
    if type(body.namespace) is not SourceBodyNamespace:
        raise LoopViewUnavailable('source-body namespace is not recorded for this graph')
    if body.document.get('structured') is None:
        raise LoopViewUnavailable('structured loop interfaces are NOT_RECORDED for this body')
    model = decode(StructuredBodyModel, body.document['structured'])
    namespace = GraphNamespace.from_source(body.namespace)
    index = body.index
    _need(index.graph is body.graph and model.definition == body.namespace.definition
        and model.region == body.namespace.region == body.graph.root,
        'structured definition, region and index differ from the original body')
    _need(body.namespace.source is None or model.source == body.namespace.source,
        'structured source differs from the original body declaration')
    _need(len({loop.header for loop in model.loops}) == len(model.loops)
        and len({loop.source for loop in model.loops}) == len(model.loops), 'duplicate loop interface')
    headers = {loop.header for loop in model.loops}
    incoming = defaultdict(dict)
    for operation in body.graph.operations:
        if operation.terminator is None:
            continue
        for ordinal, successor in enumerate(operation.terminator.successors):
            if successor.block in headers:
                incoming[successor.block][ControlEdgeReference(operation.id, ordinal)] = (operation, successor)

    def route(loop, edge, operation, successor, backedge):
        origin = index._operation_blocks[operation.id]
        _need(index._blocks[origin] == index._blocks.get(loop.header) == loop.region
            and successor.block == loop.header, 'header incoming edge crosses another region')
        formals = index._block_records[loop.header].arguments
        _need(len(formals) == len(successor.arguments), 'header incoming arity differs')
        slots = []
        for position, (formal, actual) in enumerate(zip(formals, successor.arguments)):
            value = index.values[formal]
            _need(type(value.origin) is BlockArgument and value.origin.block == loop.header
                and value.origin.index == position, 'formal is not its original ordered header argument')
            actual_value = None
            if type(actual) is UseID:
                use = index.use_site(actual)
                _need(use.operation == operation.id, 'incoming Use belongs to another operation')
                actual_value = use.use.value
                _need(value_type(index.values[actual_value]) == value_type(value), 'incoming Use type differs')
                relation = ('FORWARDED' if actual_value == formal else 'REDEFINED') if backedge else 'INITIAL'
            else:
                _need(type(actual) is CompletionResult, 'unsupported non-Use header input')
                signatures = [item for item in operation.completions if item.kind == successor.kind]
                _need(len(signatures) == 1 and actual.index < len(signatures[0].types)
                    and signatures[0].types[actual.index] == value_type(value), 'completion slot/type differs')
                relation = 'COMPLETION_OUTPUT'
            slots.append(LoopSlotRoute(position, formal, actual, actual_value,
                type(value) is StateVersionValue, relation))
        return LoopPortRoute(edge, origin, successor.block, tuple(slots))

    result = []
    for loop in model.loops:
        _need(loop.region == model.region and index._blocks.get(loop.header) == loop.region
            and all(index._blocks.get(block) == loop.region for block in
                (loop.body_entry, loop.natural_exit, *((loop.after,) if loop.after is not None else ()))),
            'loop control ports are not in their original region')
        _need((loop.source.path, loop.source.fingerprint) == (model.source.path, model.source.fingerprint)
            and _contains(model.source, loop.source), 'loop source is outside its original body')
        arcs = incoming[loop.header]
        formals = tuple(index.values[value] for value in index._block_records[loop.header].arguments)
        _need(loop.bindings == tuple(LoopBindingPort(value.slot, value.id) for value in formals
            if type(value) is LocalBindingValue), 'declared local binding ports differ from the header')
        _need(set(loop.backedges) <= arcs.keys(), 'declared backedge is not an original header incoming edge')
        initial = tuple(route(loop, edge, *ports, False) for edge, ports in arcs.items() if edge not in loop.backedges)
        carried = tuple(route(loop, edge, *arcs[edge], True) for edge in loop.backedges)
        parents = [other.source for other in model.loops if _contains(other.source, loop.source)]
        parent = min(parents, key=lambda source: (-source.start_line, -source.start_column,
            source.end_line, source.end_column)) if parents else None
        result.append((loop, initial, carried, parent))
    return tuple(LoopTemplateView(namespace, model.definition, loop, initial, carried, parent,
        tuple(child.source for child, _, _, owner in result if owner == loop.source))
        for loop, initial, carried, parent in result)


# Source-loop construct membership over original operations and ports.
#
# This projection includes initializer, test, body, else and source-attributed
# cleanup operations. It neither takes a CFG reachability closure nor claims
# that source containment is a control/effect proof. External handlers/cleanup
# stay outside, with their original boundary interfaces available to cut queries.
def _require_loop_region(ok, detail):
    if not ok:
        raise ValueError('source loop construct: ' + detail)


def _inside(outer, inner):
    return ((outer.path, outer.fingerprint) == (inner.path, inner.fingerprint)
        and (outer.start_line, outer.start_column) <= (inner.start_line, inner.start_column)
        and (inner.end_line, inner.end_column) <= (outer.end_line, outer.end_column))


def _overlaps(left, right):
    return ((left.path, left.fingerprint) == (right.path, right.fingerprint)
        and (left.start_line, left.start_column) < (right.end_line, right.end_column)
        and (right.start_line, right.start_column) < (left.end_line, left.end_column))


@dataclass(frozen=True, slots=True)
class LoopConstructView:
    template: LoopTemplateView
    members: tuple[OperationID, ...]
    direct_members: tuple[OperationID, ...]
    internal_block_arguments: tuple[ValueID, ...]
    source_mapping: SourceArtifactLocator | None
    source_gap_locators: tuple[SourceArtifactLocator, ...]
    frontiers: tuple[DependencyFrontier, ...]
    member_provenance_knowledge: str
    kind: str = 'SOURCE_LOOP_CONSTRUCT'
    scope_complete: bool = False
    iteration_knowledge: str = 'TEMPLATE_ONLY'
    content_invariance: str = 'UNPROVED'

    def __post_init__(self):
        _require_loop_region(type(self.template) is LoopTemplateView, 'original loop template required')
        self.template.__post_init__()
        for rows, kind in ((self.members, OperationID), (self.direct_members, OperationID),
                (self.internal_block_arguments, ValueID), (self.source_gap_locators, SourceArtifactLocator),
                (self.frontiers, DependencyFrontier)):
            _require_loop_region(type(rows) is tuple and all(type(row) is kind for row in rows)
                and len(set(rows)) == len(rows), 'immutable unique original IDs/locators/frontiers required')
        _require_loop_region(set(self.direct_members) <= set(self.members), 'direct members escape construct membership')
        if self.source_mapping is None:
            _require_loop_region(any(frontier.kind == 'LOOP_SOURCE_METADATA_NOT_RECORDED' for frontier in self.frontiers),
                'missing metadata must retain its original coverage frontier')
        else:
            _require_loop_region(type(self.source_mapping) is SourceArtifactLocator, 'original metadata source locator required')
            self.source_mapping.__post_init__()
            _require_loop_region(self.source_mapping.artifact == self.template.namespace.graph.artifact
                and self.source_mapping.pointer == '#/metadata/operation_links', 'source mapping differs from original body')
        _require_loop_region(self.member_provenance_knowledge in ('COMPLETE_FOR_RECORDED_OPERATIONS', 'PARTIAL')
            and self.kind == 'SOURCE_LOOP_CONSTRUCT' and self.scope_complete is False
            and self.iteration_knowledge == 'TEMPLATE_ONLY' and self.content_invariance == 'UNPROVED',
            'source membership cannot assert control/effect closure, runtime iteration or invariance')

    def expand(self, graph):
        """Load original operations through the existing receipt-owned graph."""
        _require_loop_region(type(graph) is ProgramDependencyGraph, 'expand requires the existing ProgramDependencyGraph')
        self.__post_init__()
        index = graph.load_index(self.template.namespace)
        _require_loop_region(index.graph.root == self.template.namespace.root
            and all(identity in index.operations and
                index._blocks[index._operation_blocks[identity]] == self.template.loop.region
                for identity in self.members), 'members do not belong to the original source graph/region')
        return tuple(index.operations[identity] for identity in self.members)


def loop_construct_views(body, *, templates=None):
    """Select source-attributed operations, retaining every unresolved member.

    The input is a normal session-loaded source body. Indices and source links
    are shared; no operation objects, graph, block, iteration or CFG are copied.
    ``templates`` may reuse that body's complete already checked template tuple.
    A query can select one returned construct without losing its child context.
    """
    original_templates = loop_template_views(body)
    if templates is not None:
        _require_loop_region(templates == original_templates,
            'supplied templates differ from freshly checked original loop ports')
    templates = original_templates
    _require_loop_region(type(templates) is tuple and all(type(item) is LoopTemplateView for item in templates),
        'templates must be the original complete immutable tuple')
    _require_loop_region(type(body.namespace) is SourceBodyNamespace and body.document.get('structured') is not None,
        'original structured body evidence is not recorded')
    original = decode(StructuredBodyModel, body.document['structured'])
    namespace = GraphNamespace.from_source(body.namespace)
    _require_loop_region(tuple(item.loop for item in templates) == original.loops
        and all(item.namespace == namespace and item.definition == original.definition for item in templates),
        'templates differ from the original body/recorded loop collection')
    index = body.index
    _require_loop_region(index.graph is body.graph and original.region == namespace.root,
        'source body index/region differs from its original graph')
    metadata = getattr(body, 'metadata', None)
    if metadata is None and body.document.get('metadata') is not None:
        metadata = decode(LoweringMetadata, body.document['metadata'])
    _require_loop_region(metadata is None or type(metadata) is LoweringMetadata, 'invalid original source metadata')
    local = tuple(operation.id for operation in body.graph.operations
        if index._blocks[index._operation_blocks[operation.id]] == original.region)
    sources = defaultdict(list)
    if metadata is not None:
        for ordinal, link in enumerate(metadata.operation_links):
            _require_loop_region(link.operation in index.operations, 'source link refers outside the original graph')
            sources[link.operation].append((ordinal, link.source))
    mapping = replace(namespace.graph, pointer='#/metadata/operation_links') if metadata is not None else None
    selected, diagnostics, gaps = {}, {}, {}
    for template in templates:
        template.__post_init__()
        source, members, notes, gap_sources = template.loop.source, set(), {}, []
        at = lambda identity: DependencyPoint(namespace, identity, None)
        def note(identity, kind, detail):
            notes[DependencyFrontier(at(identity), kind, detail)] = None
        for identity in local:
            rows = sources.get(identity, ())
            if not rows:
                note(identity, 'LOOP_MEMBER_SOURCE_NOT_RECORDED',
                    'This original operation has no recorded source mapping; its loop membership is unresolved.')
                continue
            included = tuple(_inside(source, reference) for _, reference in rows)
            if all(included):
                members.add(identity)
            elif any(included):
                note(identity, 'LOOP_MEMBER_SOURCE_AMBIGUOUS',
                    'Original operation_links rows ' + ','.join(str(number) for number, _ in rows)
                    + ' have both inside and outside sources; the operation was not arbitrarily included.')
            elif any((reference.path, reference.fingerprint) != (source.path, source.fingerprint)
                     for _, reference in rows):
                note(identity, 'LOOP_MEMBER_SOURCE_VERSION_OPEN',
                    'Original operation provenance includes another source version/file; construct ownership is not inferred.')
        if metadata is None:
            note(template.loop.header, 'LOOP_SOURCE_METADATA_NOT_RECORDED',
                'Loop ports are recorded, but original operation source/coverage metadata is absent.')
        else:
            for number, gap in enumerate(metadata.gaps):
                if _overlaps(source, gap.source):
                    locator = replace(namespace.graph, pointer='#/metadata/gaps/' + str(number))
                    gap_sources.append(locator)
                    note(template.loop.header, 'LOOP_RETAINED_SOURCE_GAP',
                        locator.pointer + ': ' + gap.kind.value + ': ' + gap.detail)
        selected[source] = members
        diagnostics[source] = tuple(notes)
        gaps[source] = tuple(gap_sources)
    result = []
    root_entry = index.regions[original.region].entry
    for template in templates:
        source = template.loop.source
        nested = set()
        for child in template.nested_sources:
            _require_loop_region(child in selected, 'direct child construct is absent from the original template collection')
            nested.update(selected[child])
        members = tuple(identity for identity in local if identity in selected[source])
        internal = tuple(value for block in index.regions[original.region].blocks
            if block.id != root_entry and block.operations
            and all(identity in selected[source] for identity in block.operations)
            for value in block.arguments)
        # A source correspondence gap is separate from effects/coverage gaps.
        partial = any(frontier.kind.startswith('LOOP_MEMBER_SOURCE_')
            or frontier.kind == 'LOOP_SOURCE_METADATA_NOT_RECORDED' for frontier in diagnostics[source])
        result.append(LoopConstructView(template, members,
            tuple(identity for identity in members if identity not in nested), internal,
            mapping, gaps[source], diagnostics[source],
            'PARTIAL' if partial else 'COMPLETE_FOR_RECORDED_OPERATIONS'))
    return tuple(result)


__all__ = [
    'LoopViewUnavailable',
    'LoopSlotRoute',
    'LoopPortRoute',
    'LoopTemplateView',
    'loop_template_views',
    'LoopConstructView',
    'loop_construct_views',
]
