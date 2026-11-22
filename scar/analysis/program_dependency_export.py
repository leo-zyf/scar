"""Program dependency export."""

from __future__ import annotations

from collections import Counter
from dataclasses import fields, is_dataclass, dataclass
from enum import Enum
from scar.analysis.program_dependencies import DependencyWalk
from scar.ir.program.dependencies import (
    GraphNamespace,
    DependencyFrame,
    DependencyPoint,
    DependencyCondition,
    DependencyEvidence,
    DependencyEdge,
    DependencyFrontier,
    OpenDependencyContext,
    context_frames,
    context_is_closed,
)
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.literals import PythonLiteral
from scar.ir.record_codec import encode, decode
from scar.ir.v2.ids import Identifier
from scar.analysis.program_dependency_session import ProgramDependencySession
import json
from scar.ir.program.source_call_bindings import SourceBodyNamespace


# Program dependency export.
# Normalized query records: original namespaces, contexts and evidence once.
class _Tables:
    def __init__(self, visited=()):
        self.tables = {name: [] for name in ('locators', 'namespaces', 'frames', 'contexts',
            'points', 'conditions', 'evidence', 'frontier_reasons', 'edges')}
        self.indices = {name: {} for name in self.tables}
        self.visited = set(visited)

    def put(self, name, value, record):
        found = self.indices[name].get(value)
        if found is not None:
            return found
        index = len(self.tables[name])
        self.indices[name][value] = index
        self.tables[name].append(record())
        return index

    def locator(self, value):
        return self.put('locators', value, lambda: encode(value))

    def namespace(self, value):
        def record():
            body = value.source_body
            return {'graph': self.locator(value.graph), 'representation': value.representation,
                'root': value.root.wire, 'source_body': None if body is None else {
                    'definition': body.definition.wire, 'entry': body.entry, 'source': encode(body.source)}}
        return self.put('namespaces', value, record)

    def frame(self, value):
        return self.put('frames', value, lambda: {'caller': self.namespace(value.caller),
            'operation': value.operation.wire,
            'callee_use': None if value.callee_use is None else value.callee_use.wire,
            'callee_value': None if value.callee_value is None else value.callee_value.wire,
            'target': self.namespace(value.target),
            'definition': None if value.definition is None else value.definition.wire,
            'entry': None if value.entry is None else value.entry.wire})

    def context(self, value):
        return None if value is None else self.put('contexts', value,
            lambda: {'frames': [self.frame(item) for item in context_frames(value)],
                'unknown_prefix': not context_is_closed(value)})

    def point(self, value):
        return self.put('points', value, lambda: {'namespace': self.namespace(value.namespace),
            'node': value.node.wire, 'node_type': type(value.node).__name__,
            'context': self.context(value.context), 'visited': value in self.visited})

    def condition(self, value):
        return self.put('conditions', value, lambda: {'domain': value.domain, 'name': value.name,
            'subject': None if value.subject is None else self.point(value.subject), 'index': value.index,
            'operands': [self.point(operand) for operand in value.operands]})

    def evidence(self, value):
        return self.put('evidence', value, lambda: {'rule': value.rule,
            'sources': [self.locator(item) for item in value.sources],
            'anchors': [self.point(item) for item in value.anchors]})

    def edge(self, value):
        return self.put('edges', value, lambda: {'source': self.point(value.source),
            'target': self.point(value.target), 'kind': value.kind.value, 'meaning': value.meaning.value,
            'strength': value.strength.value, 'evidence': [self.evidence(item) for item in value.evidence],
            'conditions': [self.condition(item) for item in value.conditions]})

    def frontier(self, value):
        reason = self.put('frontier_reasons', (value.kind, value.detail),
            lambda: {'kind': value.kind, 'detail': value.detail})
        return {'point': self.point(value.point), 'reason': reason}

    def pack(self, value):
        references = ((GraphNamespace, 'namespace', self.namespace), (DependencyFrame, 'frame', self.frame),
            (OpenDependencyContext, 'context', self.context),
            (DependencyPoint, 'point', self.point), (DependencyCondition, 'condition', self.condition),
            (DependencyEvidence, 'evidence', self.evidence), (DependencyEdge, 'edge', self.edge),
            (SourceArtifactLocator, 'locator', self.locator))
        for kind, name, method in references:
            if type(value) is kind:
                return {name: method(value)}
        if type(value) is DependencyFrontier:
            return self.frontier(value)
        if type(value) is PythonLiteral:
            return {'literal': value.as_dict()}
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Identifier):
            return {'original_id': value.wire}
        if is_dataclass(value):
            return {'record_type': type(value).__name__,
                **{field.name: self.pack(getattr(value, field.name)) for field in fields(value)}}
        if isinstance(value, (tuple, list)):
            return [self.pack(item) for item in value]
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise TypeError('query records need string field names')
            return {key: self.pack(item) for key, item in value.items()}
        if value is None or type(value) in (str, int, float, bool):
            return value
        raise TypeError('unsupported query field: ' + type(value).__name__)


def export_dependency_walk(walk):
    """Export a bounded dependency query, retaining original entity references."""
    if type(walk) is not DependencyWalk:
        raise TypeError('expected a typed original dependency traversal')
    table = _Tables(walk.points)
    roots = [table.point(root) for root in walk.roots]
    for point in walk.points:
        table.point(point)
    for edge in walk.edges:
        table.edge(edge)
    frontiers = [table.frontier(item) for item in walk.frontiers]
    return {'schema': 'scar.program-dependency-query', 'schema_version': 3,
        'direction': walk.direction, 'roots': roots, **table.tables, 'frontiers': frontiers,
        'relation_scope': [kind.value for kind in walk.kinds],
        'summary': {'visited_points': len(walk.points), 'referenced_points': len(table.tables['points']),
            'namespaces': len(table.tables['namespaces']), 'edges': len(walk.edges),
            'edge_kinds': dict(Counter(edge.kind.value for edge in walk.edges)),
            'frontier_kinds': dict(Counter(item.kind for item in walk.frontiers))},
        'traversal_exhausted': walk.traversal_exhausted, 'dependence_closed': walk.dependence_closed,
        'whole_program_complete': False, 'transformation_authorized': False,
        'qualification': 'Original typed endpoints and conditional dependencies over shared source fragments. '
            'Context null is wholly unknown; unknown_prefix retains only a known suffix, while a closed empty frame list selects no caller. '
            'CFG/sequence/containment are structural, not required ordering or equality. '
            'Query completion is not whole-program closure or transformation permission.'}


def export_value_analysis(result):
    from scar.analysis.program_value_analysis import ValueAnalysisResult
    if type(result) is not ValueAnalysisResult:
        raise TypeError('expected typed original value facts')
    table = _Tables(fact.point for fact in result.facts)
    roots = [table.point(point) for point in result.roots]
    for edge in result.edges:
        table.edge(edge)
    def field(value):
        return {'precision': value.precision.value, 'applicable': value.applicable,
            'alternatives': [table.pack(item) for item in value.alternatives],
            'unknown_paths': [table.frontier(item) for item in value.unknown_paths]}
    def fields_of(value):
        return {name: field(getattr(value, name)) for name in
                ('reference_sources', 'scalar_constant', 'builtin_type', 'immutable_tuple')}
    facts = [{'point': table.point(fact.point), 'role': fact.role, 'feasible': fact.feasible,
        'fields': fields_of(fact.fields),
        'binding_payload': None if fact.binding_payload is None else fields_of(fact.binding_payload),
        'binding_state': field(fact.binding_state),
        'occurrence': table.pack(fact.occurrence),
        'infeasibility': table.pack(fact.infeasibility),
        'binding_states': [item.value for item in fact.binding_states]} for fact in result.facts]
    frontiers = [table.frontier(item) for item in result.frontiers]
    branch_proofs = [table.pack(item) for item in result.branch_proofs]
    binding_proofs = [table.pack(item) for item in result.binding_proofs]
    route_refutations = [table.pack(item) for item in result.route_refutations]
    return {'schema': 'scar.program-value-analysis', 'schema_version': 4, 'roots': roots,
        **table.tables, 'facts': facts, 'frontiers': frontiers, 'branch_proofs': branch_proofs,
        'binding_proofs': binding_proofs,
        'route_refutations': route_refutations,
        'usage': result.usage,
        'fixed_point_reached': result.worklist_stable, 'query_scope_complete': result.exhausted,
        'whole_program_complete': False,
        'transformation_authorized': False,
        'qualification': 'Field-wise definition provenance and supported immutable builtin scalar/tuple content facts. '
            'Binding state knowledge is separate from payload; binding_states is only a convenience projection. '
            'Fixed-point convergence describes represented field constraints, separately from original query scope completeness. '
            'An open context retains known fields without authorizing closed entry, guard or normal-route proofs. '
            'A source Point is not an ObjectID or content identity; conditions, proofs and unknown paths remain explicit.'}


def export_boundary_analysis(result):
    from scar.analysis.program_regions import BoundaryAnalysis
    if type(result) is not BoundaryAnalysis:
        raise TypeError('expected typed original region boundary')
    table = _Tables(result.cut.members)
    records = table.pack(result)
    # Derived API views reference records already present in the normalized
    # export. Keep read/write membership explicit without duplicating state.
    reads, writes = result.reads, result.writes
    records['reads'] = [number for number, state in enumerate(result.state) if state in reads]
    records['writes'] = [number for number, state in enumerate(result.state) if state in writes]
    records['control_dependencies'] = [table.edge(edge) for edge in result.control_dependencies]
    return {'schema': 'scar.program-region-boundary', 'schema_version': 3,
        **table.tables, 'boundary': records, 'whole_program_complete': False,
        'transformation_authorized': False,
        'qualification': 'A cut of original operations and field-wise interface/effect knowledge. '
            'Implementation transfers and outside consumers do not prove observable escape; missing contracts stay open.'}


# Static field and boundary queries using the one dependency session/view.
def query_program_values(directory, *, file=None, entry=None, kind=None, node=None, roots=None,
                         context=(), budget=None, value_budget=None):
    from scar.analysis.program_value_analysis import ProgramValueAnalysis
    with ProgramDependencySession(directory, budget=budget) as session:
        if roots is None:
            if node is None:
                raise ValueError("value inference needs an original typed root")
            roots = (session.point(file=file, entry=entry, kind=kind, node=node, context=context),)
        return ProgramValueAnalysis(session, budget=value_budget).analyze(roots)


def query_program_boundary(directory, *, file=None, entry=None, kind=None, nodes=(), operations=None,
                           context=(), budget=None, max_members=16384):
    from scar.analysis.program_regions import analyze_boundary
    with ProgramDependencySession(directory, budget=budget) as session:
        if operations is None:
            operations = tuple(session.point(file=file, entry=entry, kind=kind, node=node, context=context)
                               for node in nodes)
        return analyze_boundary(session.graph(), operations, max_members=max_members)


# Receipt-checked raw routing, without claiming typed relation validation.
#
# Every serialized source-body namespace contributes its original shard to an
# upper bound.  This includes foreign producers, creators and context witnesses,
# not just caller/callee headers.  Only a selected shard is subsequently decoded
# and replayed by the session's normal fresh verifier.  A complete routing index
# therefore closes the *serialized lookup*, not target sets or program effects.
_BODY_FIELDS = frozenset(("graph", "graph_representation", "definition", "source", "entry", "region"))
_RELATION_FIELDS = frozenset(("source_model", "caller", "calls", "import_routes",
    "module_attribute_flows", "module_frontiers", "module_globals_flows"))


@dataclass(frozen=True, slots=True)
class RelationRoutingResult:
    uris: tuple[str, ...]
    complete: bool
    issue: str | None = None


def _namespace_wire_key(value, check):
    """An exact JSON key; no tuple/key coercions may borrow a valid record."""
    pending, containers = [value], set()
    visits = 0
    while pending:
        current = pending.pop()
        visits += 1
        if visits % 256 == 0:
            check()
        kind = type(current)
        if kind in (dict, list):
            if kind is dict and any(type(key) is not str for key in current):
                raise TypeError('raw namespace wire requires exact JSON string keys')
            identity = id(current)
            if identity not in containers:
                containers.add(identity)
                pending.extend(current.values() if kind is dict else current)
        elif kind not in (str, int, float, bool, type(None)):
            raise TypeError('raw namespace wire requires exact JSON value types')
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _namespace_identities(document, check):
    """Visit all wire fields, retaining no document or second program graph."""
    pending = [document]
    found = set()
    # Only this shard's exact wire records are reused. A namespace identity
    # omits declaration/source fields, so it cannot be the validation key.
    identities_by_wire = {}
    visits = 0
    while pending:
        value = pending.pop()
        visits += 1
        if visits % 256 == 0:
            check()
        if type(value) is dict:
            if "graph_representation" in value:
                if set(value) != _BODY_FIELDS:
                    raise ValueError("raw relationship routing found a malformed source-body namespace")
                wire_key = _namespace_wire_key(value, check)
                identity = identities_by_wire.get(wire_key)
                if identity is None:
                    # Decode the whole record, never its referenced graph.
                    # Distinct declarations/source fields still get checked.
                    namespace = GraphNamespace.from_source(decode(SourceBodyNamespace, value))
                    identity = namespace.identity
                    identities_by_wire[wire_key] = identity
                found.add(identity)
                continue
            pending.extend(value.values())
        elif type(value) is list:
            pending.extend(value)
    return found


class RawCallRelationRouting:
    """One immutable-session, bounded namespace -> original-shard upper bound."""
    def __init__(self, *, items, source_model, version, read, check,
                 max_shards, max_references):
        self.items = items
        self.source_model, self.version = source_model, version
        self.read, self.check = read, check
        self.max_shards, self.max_references = max_shards, max_references
        self.routes = {}
        self.shards = self.references = 0
        self.complete = False
        self.issue = None
        self._remaining = iter(items)
        self._pending = None

    def ensure(self):
        if self.complete:
            return
        if self.version not in (8, 9):
            self.issue = "Raw routing does not assert completeness for historical call wire profiles."
            return
        try:
            while True:
                self.check()
                if self._pending is None:
                    self._pending = next(self._remaining, None)
                if self._pending is None:
                    self.complete, self.issue = True, None
                    return
                if self.shards >= self.max_shards:
                    raise OverflowError("dependency raw routing shard budget exhausted")
                uri = self._pending
                document = self.read(uri)
                if (type(document) is not dict or set(document) != {"schema", "schema_version", "relations"}
                        or document["schema"] != "scar.static-calls"
                        or type(document["schema_version"]) is not int or document["schema_version"] != self.version):
                    raise ValueError("raw routing shard differs from its exact published call wire profile")
                relations = document["relations"]
                if (type(relations) is not dict or set(relations) != _RELATION_FIELDS
                        or relations["source_model"] != self.source_model
                        or relations["caller"] != self.items[uri]["caller"]
                        or any(type(relations[name]) is not list for name in _RELATION_FIELDS - {"source_model", "caller"})):
                    raise ValueError("raw routing shard differs from its original source basis or owner")
                identities = _namespace_identities(relations, self.check)
                if self.references + len(identities) > self.max_references:
                    raise OverflowError("dependency raw routing reference budget exhausted")
                # Commit a whole receipt-checked shard atomically.  A partial
                # prefix is still only an upper bound for that known prefix.
                for identity in identities:
                    self.routes.setdefault(identity, []).append(uri)
                self.references += len(identities)
                self.shards += 1
                self._pending = None
        except (OverflowError, TimeoutError) as error:
            self.issue = type(error).__name__ + ": " + str(error)

    def lookup(self, namespace):
        if type(namespace) is not GraphNamespace:
            raise TypeError("raw call routing needs an original graph namespace")
        self.ensure()
        return RelationRoutingResult(tuple(self.routes.get(namespace.identity, ())), self.complete, self.issue)


__all__ = [
    'RawCallRelationRouting',
    'RelationRoutingResult',
    'export_dependency_walk',
    'export_value_analysis',
    'export_boundary_analysis',
    'query_program_values',
    'query_program_boundary',
]
