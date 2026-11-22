"""Portable source-interface fixture; no retained artifacts or target runtime."""
from collections import defaultdict
from types import SimpleNamespace as NS

from scar.analysis.program_view import ProgramView,SOURCE_REPRESENTATION,_digest
from scar.analysis.source_content_lineage import SourceContentLineageProvider
from scar.ir.program import data_materials as C, data_materials as L, normal_graph as G
from scar.ir.program import normal_graph as T, source_productions as P
from scar.ir.program.control import LanguageCompletionKind as Completion
from scar.ir.program.dependencies import (GraphNamespace,DependencyPoint as Point,
    DependencyKind as K,DependencyMeaning as Meaning,DependencyStrength as Strength)
from scar.ir.program.ids import RegionID,OperationID,ValueID,UseID,LocationID,ControlPortID
from scar.ir.program.normal_graph import CodeScopeRef
from scar.ir.program.providers import ProviderArtifact
from scar.ir.program.data_materials import FieldSelector
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.program.source_template_links import SourceTemplateKey
from scar.ir.record_codec import encode,decode
from scar.ir.v2.common import ProofStatus
from scar.ir.v2.values import ProvenanceRelation,EquivalenceClaim


def test_content_lineage_preserves_four_protocols_without_fabricating_versions(monkeypatch):
    unit='portable-content';key=SourceTemplateKey(0,P.ProductionPurpose.VALUE)
    region=P.ProductionRegionID('body')
    locator=SourceArtifactLocator(ProviderArtifact('/fixture/original.json','0'*64,1,'fixture'),
        '0'*64,1,'canonical-json')
    namespace=GraphNamespace(locator,SOURCE_REPRESENTATION,RegionID('root'))
    gp=lambda owner,kind,index=0:T.GlobalPort(unit,G.Port(key,owner,kind,index,
        Completion.FALLTHROUGH if kind is G.PortKind.RESULT else None))
    receiver=gp(region,G.PortKind.PARAMETER,0)
    payload=gp(region,G.PortKind.PARAMETER,1)
    world=gp(region,G.PortKind.PARAMETER,2)
    address=C.ContentAddressTemplate(LocationID('slot'),receiver,FieldSelector('field'),
        (C.ContentInstanceParameter.ACTIVATION,C.ContentInstanceParameter.OWNER_REFERENCE_OCCURRENCE))
    accesses=[];projections={}
    state=ValueID('before')
    projections[state]=C.ContentStateProjection(state,address.id,world)
    for index,kind in enumerate((C.ContentRequestKind.SET,C.ContentRequestKind.GET,
            C.ContentRequestKind.UPDATE,C.ContentRequestKind.DELETE)):
        step=P.ProductionStepID(kind.value.lower());operation=OperationID(kind.value.lower())
        after=gp(step,G.PortKind.RESULT,1 if kind is C.ContentRequestKind.GET else 0)
        next_state=ValueID('after-'+str(index))
        projections[next_state]=C.ContentStateProjection(next_state,address.id,after)
        accesses.append(C.ContentAccessTemplate(operation,gp(step,G.PortKind.ENTER),CodeScopeRef(unit,0),
            kind,address.id,UseID('receiver-'+str(index)),
            payload if kind in (C.ContentRequestKind.SET,C.ContentRequestKind.UPDATE) else None,
            gp(step,G.PortKind.RESULT) if kind is C.ContentRequestKind.GET else None,
            world,after,state,next_state,None))
        world,state=after,next_state
    view=ProgramView(None)

    def source_point(port):
        cls=ControlPortID if port.port.kind is G.PortKind.ENTER else ValueID
        return Point(namespace,cls(_digest(port)),None)

    monkeypatch.setattr(view,'source_port',source_point)
    source=NS(unit=unit,locator=locator,nodes={row.operation:NS() for row in accesses})

    class ContentFixture:
        def __init__(self):
            self.view=view;self.source=source;self.namespace=namespace
            self.addresses={address.id:address};self.projections=projections
            self.accesses=[];self.operation_accesses=defaultdict(list);self.complete=False

        def check(self,point=None):
            assert point is None or point.namespace==namespace

        def ensure_operation(self,operation):
            identity=operation.node if type(operation)is Point else operation
            if identity not in self.operation_accesses:
                row=next(row for row in accesses if row.operation==identity)
                self.operation_accesses[identity].append(len(self.accesses));self.accesses.append(row)

        def ensure_all(self):
            for row in accesses:self.ensure_operation(row.operation)
            self.complete=True

    content=ContentFixture();lineage=SourceContentLineageProvider(content)
    metadata=Point(namespace,accesses[0].state_before,None)
    empty=view.dependencies.incoming(metadata,kinds=(K.STATE_VALUE,))
    assert not lineage.records
    assert any(f.kind=='SOURCE_CONTENT_LINEAGE_ADJACENCY_PARTIAL' for f in empty.frontiers)
    lineage.ensure_operation(accesses[0].operation)
    assert len(lineage.records)==1 and not lineage.complete
    # A parent-admitted GET is synchronized by querying stored metadata,
    # without a new operation event or an implicit complete-unit request.
    content.ensure_operation(accesses[1].operation)
    assert len(lineage.records)==1
    view.dependencies.outgoing(metadata,kinds=(K.STATE_VALUE,))
    assert len(lineage.records)==2 and not lineage.complete
    lineage.ensure_all()
    assert lineage.complete and len(lineage.records)==5
    roles={row.role:row for row in lineage.records}
    expected={
        L.ContentLineageRole.SET_PAYLOAD:(accesses[0].payload,accesses[0].state_after),
        L.ContentLineageRole.GET_RESULT:(accesses[1].state_before,accesses[1].result),
        L.ContentLineageRole.UPDATE_PRIOR:(accesses[2].state_before,accesses[2].state_after),
        L.ContentLineageRole.UPDATE_INPUT:(accesses[2].payload,accesses[2].state_after),
        L.ContentLineageRole.DELETE_PRIOR:(accesses[3].state_before,accesses[3].state_after),
    }
    assert set(roles)==set(expected)
    for role,endpoints in expected.items():
        row=roles[role]
        assert (row.source,row.target)==endpoints
        assert lineage.access(row.access) is content.accesses[row.access.ordinal]
        assert row.provenance_relation is ProvenanceRelation.UNKNOWN
        assert row.equivalence is EquivalenceClaim.UNKNOWN
        assert row.logical_version_equality is ProofStatus.UNKNOWN
        assert decode(L.ContentValueLineage,encode(row))==row
    edges=[edge for rows in lineage.out_edges.values() for edge in rows]
    assert len(edges)==5
    assert all(edge.kind is K.STATE_VALUE and edge.meaning is Meaning.DEPENDENCE
        and edge.strength is Strength.MAY and edge.conditions for edge in edges)
    assert not any(edge.kind is K.ALIAS or edge.meaning is Meaning.REFERENCE for edge in edges)
    loaded=view.dependencies.incoming(source_point(accesses[1].result),kinds=(K.STATE_VALUE,))
    assert any(edge.source.node==accesses[1].state_before for edge in loaded.edges)
    assert not any(f.kind=='SOURCE_CONTENT_LINEAGE_ADJACENCY_PARTIAL' for f in loaded.frontiers)
    obligation=lineage.forwarding_obligations(lineage.access_ref(0),lineage.access_ref(1))
    assert obligation.status is ProofStatus.UNKNOWN and obligation.equivalence is EquivalenceClaim.UNKNOWN
    assert obligation.required_contracts
    summary=lineage.summary();lineage.ensure_all()
    assert lineage.summary()==summary
    assert all(summary[name]==0 for name in ('logical_values_created','logical_versions_created',
        'equality_claims_created','forwarding_edges_created'))
    # Parent citation order is session-local. Opposite incremental demand
    # order must preserve the semantic edge set and still cite the correct
    # parent access row, rather than leaking its ordinal into predicates.
    reversed_parent=ContentFixture()
    reversed_lineage=SourceContentLineageProvider(reversed_parent,register=False)
    for access in reversed(accesses):reversed_lineage.ensure_operation(access.operation)
    reversed_lineage.ensure_all()
    reversed_edges={edge for rows in reversed_lineage.out_edges.values() for edge in rows}
    assert reversed_edges==set(edges)
    assert reversed_parent.accesses==list(reversed(content.accesses))
    for record in reversed_lineage.records:
        assert reversed_lineage.access(record.access) is reversed_parent.accesses[record.access.ordinal]
        assert record.access.operation==reversed_parent.accesses[record.access.ordinal].operation
        original=next(row for row in lineage.records
            if row.access.operation==record.access.operation and row.role is record.role)
        assert original.access.ordinal!=record.access.ordinal
