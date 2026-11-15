"""Incremental conditional bridges from original values to content state.

Read/write intent comes from the admitted original content protocol record.
Callbacks, aliases, exception effects and actual object contents stay open.
Every new edge is a MAY dependence; no reference/equality edge or logical
version is inferred from a GET, a stable receiver, a shape, or a world token.
"""
from collections import Counter, defaultdict
from dataclasses import replace

from scar.ir.program import source_content as C, normal_graph as G
from scar.ir.program import source_content_lineage as L
from scar.ir.program.ids import OperationID
from scar.ir.program.dependencies import (DependencyPoint as Point, DependencyEdge as Edge,
    DependencyCondition as Condition, DependencyEvidence as Evidence,
    DependencyFrontier as Frontier, DependencyNeighborhood as Neighborhood,
    DependencyKind as K, DependencyMeaning as Meaning, DependencyStrength as Strength)
from scar.ir.program.normal_targets import GlobalPort
from scar.ir.program.source_productions import ProductionStepID
from scar.ir.record_codec import encode


SCHEMA = 'scar.source-content-value-lineage'
_ROLE_CONDITIONS = {
    L.ContentLineageRole.GET_RESULT: 'SELECTED_GET_PROTOCOL_MAY_READ_REQUESTED_CONTENT',
    L.ContentLineageRole.SET_PAYLOAD: 'SELECTED_SET_PROTOCOL_MAY_STORE_OR_DERIVE_CONTENT_FROM_PAYLOAD',
    L.ContentLineageRole.UPDATE_INPUT: 'UPDATE_INPUT_MAY_CONTRIBUTE_TO_RESULTING_CONTENT_NOT_EQUAL_WHOLE_CONTENT',
    L.ContentLineageRole.UPDATE_PRIOR: 'UPDATE_MAY_DEPEND_ON_PREVIOUS_CONTENT_NOT_REFERENCE_FORWARDING',
    L.ContentLineageRole.DELETE_PRIOR: 'DELETE_PROTOCOL_MAY_DEPEND_ON_PREVIOUS_CONTENT',
}
_FORWARDING_CONTRACTS = (
    'SAME_EXECUTION_ACTIVATION_AND_MATCHED_CALL_CONTEXT',
    'SAME_ACTUAL_RECEIVER_AND_SELECTOR_INSTANCE_OR_PROVEN_ADDRESS_ALIAS',
    'WRITE_PROTOCOL_IS_REFERENCE_PRESERVING_SLOT_ASSIGNMENT',
    'READ_PROTOCOL_RETURNS_THAT_STORED_REFERENCE',
    'WRITE_NORMAL_COMPLETION_DOMINATES_READ_WITH_REQUIRED_HAPPENS_BEFORE',
    'NO_INTERVENING_MAY_ALIAS_WRITE_DELETE_OR_UPDATE',
    'DESCRIPTOR_ITEM_CALLBACK_HOOK_RNG_EXTERNAL_AND_ORDERING_EFFECTS_CLOSED',
    'REQUIRED_EXCEPTION_AND_ESCAPE_BEHAVIOR_PRESERVED',
    'CONTENT_MUTATION_COVERAGE_REQUIRED_SEPARATELY_FOR_LOGICAL_VERSION_EQUALITY',
)


class SourceContentLineageProvider:
    """A relation-only provider over the existing source/content endpoints.

    ``records`` refer to parent access ordinals rather than copying access,
    selector and world records. New parent accesses are consumed once. Calling
    ``ensure_operation`` never requests all unit operations; ``ensure_all`` is
    the separate, explicit complete projection path.
    """
    relation_kinds = frozenset((K.STATE_VALUE,))

    def __init__(self, content, *, register=True):
        self.content = content
        self.source = content.source
        self.view = content.view
        self.namespace = content.namespace
        self.records = []
        self.by_access = defaultdict(list)
        self.endpoint_records = defaultdict(list)
        self.in_edges = defaultdict(list)
        self.out_edges = defaultdict(list)
        self._cursor = 0
        self._syncing = False
        self._sync()
        if register:
            self.view.register(self)

    @property
    def complete(self):
        return self.content.complete and self._cursor == len(self.content.accesses)

    def check(self, point=None):
        self.content.check(point)
        if len(self.content.accesses) < self._cursor:
            raise ValueError('original content access domain shrank after lineage admission')

    def access_ref(self, ordinal):
        self.check()
        if type(ordinal) is not int or not 0 <= ordinal < len(self.content.accesses):
            raise ValueError('content access ordinal is outside its original provider')
        return L.ContentAccessRef(self.source.unit, ordinal, self.content.accesses[ordinal].operation)

    def access(self, reference):
        self.check()
        if type(reference) is not L.ContentAccessRef:
            raise TypeError('original typed content access reference required')
        reference.__post_init__()
        if (reference.unit != self.source.unit or reference.ordinal >= len(self.content.accesses)
                or self.content.accesses[reference.ordinal].operation != reference.operation):
            raise ValueError('content lineage reference differs from its original access')
        return self.content.accesses[reference.ordinal]

    def _point(self, endpoint):
        return self.view.source_port(endpoint) if type(endpoint) is GlobalPort else Point(self.namespace, endpoint, None)

    def _derive(self, ordinal, access):
        address = self.content.addresses[access.address]
        before = self.content.projections[access.state_before]
        after = self.content.projections[access.state_after]
        if ((before.address, before.world) != (address.id, access.world_before)
                or (after.address, after.world) != (address.id, access.world_after)):
            raise ValueError('content lineage state/address/world differs from original access')
        reference = self.access_ref(ordinal)
        if access.kind is C.ContentRequestKind.GET:
            if access.result is None:
                raise ValueError('GET lineage lacks its original result')
            links = ((L.ContentLineageRole.GET_RESULT, access.state_before, access.result),)
        elif access.kind is C.ContentRequestKind.SET:
            if access.payload is None:
                raise ValueError('SET lineage lacks its original payload')
            links = ((L.ContentLineageRole.SET_PAYLOAD, access.payload, access.state_after),)
        elif access.kind is C.ContentRequestKind.UPDATE:
            links = ((L.ContentLineageRole.UPDATE_PRIOR, access.state_before, access.state_after),)
            if access.payload is not None:
                links += ((L.ContentLineageRole.UPDATE_INPUT, access.payload, access.state_after),)
        elif access.kind is C.ContentRequestKind.DELETE:
            links = ((L.ContentLineageRole.DELETE_PRIOR, access.state_before, access.state_after),)
        else:
            raise ValueError('content lineage has an undeclared original request kind')
        operation = Point(self.namespace, access.operation, None)
        location = Point(self.namespace, address.id, None)
        receiver = Point(self.namespace, access.receiver_use, None)
        normal = self.view.source_port(access.world_after)
        staged = []
        for role, source_endpoint, target_endpoint in links:
            record = L.ContentValueLineage(reference, role, source_endpoint, target_endpoint)
            source, target = self._point(source_endpoint), self._point(target_endpoint)
            # Parent access ordinals cite a demand-ordered table; they are
            # not semantic branch coordinates. The current source protocol
            # has one request per original operation, already selected by
            # the operation/address endpoints and finite role predicate.
            guards = (
                Condition(SCHEMA, 'SAME_SELECTED_SOURCE_ACTIVATION_AND_REQUEST_ADDRESS_INSTANCE',
                    operation, None, (receiver, location)),
                Condition(SCHEMA, 'ORIGINAL_NORMAL_COMPLETION_SELECTED',
                    normal, access.world_after.port.ordinal, (operation,)),
                Condition(SCHEMA, _ROLE_CONDITIONS[role], operation, None, (source, target)),
            )
            edge = Edge(source, target, K.STATE_VALUE, Meaning.DEPENDENCE,
                (Evidence(SCHEMA+'.'+role.value, (self.source.locator,), (operation, source, target)),),
                guards, Strength.MAY)
            staged.append((record, edge))
        return staged

    def _sync(self):
        self.check()
        if self._syncing:
            return
        self._syncing = True
        try:
            while self._cursor < len(self.content.accesses):
                ordinal = self._cursor
                staged = self._derive(ordinal, self.content.accesses[ordinal])
                self.check()
                for record, edge in staged:
                    index = len(self.records)
                    self.by_access[ordinal].append(index)
                    for endpoint in dict.fromkeys((edge.source.node, edge.target.node)):
                        self.endpoint_records[endpoint].append(index)
                    self.records.append(record)
                    self.out_edges[edge.source.node].append(edge)
                    self.in_edges[edge.target.node].append(edge)
                self._cursor += 1
        finally:
            self._syncing = False

    def ensure_operation(self, operation):
        self.content.ensure_operation(operation)
        self._sync()
        identity = operation.node if type(operation) is Point else operation
        return tuple(self.records[index]
            for ordinal in self.content.operation_accesses.get(identity, ())
            for index in self.by_access[ordinal])

    def ensure_all(self):
        self.content.ensure_all()
        self._sync()
        return self

    def _event(self, point):
        node = self.source.nodes.get(point.node)
        if node is None:
            return None
        if type(point.node) is OperationID:
            return point
        original = node.anchor.original
        if type(original) is G.Port and type(original.owner) is ProductionStepID:
            return self.source._node(original.template, original.owner)
        return None

    def _neighbors(self, point, incoming):
        if point.namespace != self.namespace:
            return Neighborhood()
        self.check(point)
        event = self._event(point)
        if event is not None:
            self.ensure_operation(event)
        else:
            self._sync()
        rows = (self.in_edges if incoming else self.out_edges).get(point.node, ())
        edges = tuple(replace(edge, source=replace(edge.source, context=point.context),
            target=replace(edge.target, context=point.context), conditions=tuple(replace(condition,
                subject=None if condition.subject is None else replace(condition.subject, context=point.context),
                operands=tuple(replace(operand, context=point.context) for operand in condition.operands))
                for condition in edge.conditions)) for edge in rows)
        known = (point.node in self.source.nodes or point.node in self.content.addresses
            or point.node in self.content.projections)
        frontiers = []
        if known and not self.complete:
            frontiers.append(Frontier(point, 'SOURCE_CONTENT_LINEAGE_ADJACENCY_PARTIAL',
                'Unrequested content accesses can use this value/address/world; only ensure_all explicitly closes the original unit domain.'))
        if rows or point.node in self.content.addresses or point.node in self.content.projections:
            frontiers.append(Frontier(point, 'CONTENT_LINEAGE_IS_NOT_LOGICAL_VERSION_OR_PROTOCOL_EFFECT_PROOF',
                'Descriptor/item callbacks, aliasing, mutation, exceptions and reference-preserving forwarding require separate contracts.'))
        return Neighborhood(edges, tuple(frontiers))

    def incoming(self, point):
        return self._neighbors(point, True)

    def outgoing(self, point):
        return self._neighbors(point, False)

    def forwarding_obligations(self, write, read):
        """Inspect one caller-selected SET/GET pair; never all-pairs discovery."""
        self._sync()
        written, loaded = self.access(write), self.access(read)
        if (written.kind is not C.ContentRequestKind.SET or written.payload is None
                or loaded.kind is not C.ContentRequestKind.GET or loaded.result is None):
            raise ValueError('reference forwarding inquiry requires original SET payload and GET result')
        known = ['ORIGINAL_SET_PAYLOAD_AND_GET_RESULT_PORTS_RETAINED']
        if written.address == loaded.address:
            known.append('SAME_STATIC_REQUEST_ADDRESS_TEMPLATE_NOT_RUNTIME_INSTANCE_PROOF')
        if written.scope == loaded.scope:
            known.append('SAME_ORIGINAL_SOURCE_SCOPE_NOT_ACTIVATION_PROOF')
        if written.world_after == loaded.world_before:
            known.append('WRITE_POST_WORLD_IS_READ_PRE_WORLD_NOT_CONTENT_VERSION_EQUALITY')
        return L.ContentForwardingObligations(write, read, tuple(known), _FORWARDING_CONTRACTS)

    def summary(self):
        self._sync()
        return dict(schema=SCHEMA, access_records=self._cursor, lineage_records=len(self.records),
            roles=dict(Counter(record.role.value for record in self.records)), complete=self.complete,
            logical_values_created=0, logical_versions_created=0, equality_claims_created=0,
            forwarding_edges_created=0, target_executed=False)

    def query(self, point):
        """Explain relation evidence while parent providers own every node."""
        neighborhood = self.incoming(point)
        records = tuple(self.records[index] for index in self.endpoint_records.get(point.node, ()))
        return dict(summary=self.summary(), records=encode(records), neighborhood=encode(neighborhood),
            world_snapshot_is_logical_value_version=False, logical_version_equality_proven=False)


__all__ = ['SourceContentLineageProvider']
