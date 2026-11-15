"""Portable balanced-control positive and intervening-callback counterexample."""
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from scar.analysis.program_view import SOURCE_REPRESENTATION,_digest
from scar.analysis.source_content_forwarding import SourceContentForwardingProvider,_control_window,_Refuse
from scar.ir.program import normal_graph as G,normal_targets as T,source_productions as P
from scar.ir.program import source_content_forwarding as F
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.dependencies import (GraphNamespace,DependencyPoint as Point,
    DependencyMeaning,DependencyStrength)
from scar.ir.program.ids import RegionID,ValueID,OperationID
from scar.ir.program.providers import ProviderArtifact,ProviderFactReference
from scar.ir.program.source_content_lineage import ContentAccessRef
from scar.ir.program.source_reference_slices import ReferenceSlice
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.program.source_template_links import SourceTemplateKey
from scar.ir.program.source_value_flow import SourceBodyValueFlow,SourceOperation
from scar.ir.literals import PythonLiteral
from scar.ir.v2.common import ProofStatus
from scar.ir.v2.values import EquivalenceClaim


def test_guarded_builtin_rule_balances_returns_and_refuses_intervening_call():
    unit='portable-forwarding';key=SourceTemplateKey(0,P.ProductionPurpose.VALUE)
    rid,sid=P.ProductionRegionID,P.ProductionStepID
    port=lambda owner,kind:G.Port(key,owner,kind,completion=C.FALLTHROUGH
        if kind in (G.PortKind.EXIT,G.PortKind.RESULT,G.PortKind.CALL_RETURN)else None)
    gp=lambda p:T.GlobalPort(unit,p)
    enter=port(rid('body'),G.PortKind.ENTER)
    write=gp(port(sid('write'),G.PortKind.ENTER));read=gp(port(sid('read'),G.PortKind.ENTER))
    literal=port(sid('literal'),G.PortKind.ENTER)
    site=G.CallSite(key,sid('invoke'));wrong=G.CallSite(key,sid('another-caller'))
    edge=lambda a,b,transition=G.Transition.DIRECT,call=None:G.Edge(a,b,G.EdgeKind.CONTROL,
        site=call,transition=transition)
    edges=(edge(enter,write.port),edge(write.port,port(sid('write'),G.PortKind.EXIT)),
        edge(port(sid('write'),G.PortKind.EXIT),port(sid('invoke'),G.PortKind.ENTER)),
        edge(port(sid('invoke'),G.PortKind.ENTER),port(rid('shared'),G.PortKind.ENTER),G.Transition.CALL,site),
        edge(port(rid('shared'),G.PortKind.ENTER),literal),
        edge(literal,port(sid('literal'),G.PortKind.EXIT)),
        edge(port(sid('literal'),G.PortKind.EXIT),port(rid('shared'),G.PortKind.EXIT)),
        edge(port(rid('shared'),G.PortKind.EXIT),port(sid('another-caller'),G.PortKind.CALL_RETURN),G.Transition.RETURN,wrong),
        edge(port(rid('shared'),G.PortKind.EXIT),port(sid('invoke'),G.PortKind.CALL_RETURN),G.Transition.RETURN,site),
        edge(port(sid('invoke'),G.PortKind.CALL_RETURN),read.port))
    artifact=ProviderArtifact('/fixture/original.json','0'*64,1,'fixture')
    function=gp(G.Port(key,rid('body'),G.PortKind.PARAMETER))
    operation=SourceOperation(literal,P.DefineLiteral(PythonLiteral.from_python(None)),(),())
    flow=SourceBodyValueFlow(function,T.Body(unit,key,rid('body')),artifact,edges,(operation,),(),())
    path,start,reads,cleanups=_control_window(flow,write,read,())
    assert start==2 and not reads and not cleanups
    assert len(path)==9 and not any(e.site==wrong for e in path)
    callback=replace(operation,instruction=P.ApplyLanguageAction(P.LanguageActionKind.CALL))
    with pytest.raises(_Refuse)as caught:
        _control_window(replace(flow,operations=(callback,)),write,read,())
    assert caught.value.reason=='INTERVENING_OPERATION_NOT_CERTIFIED_REFERENCE_READ_OR_LITERAL'
    assert caught.value.site==gp(literal)

    producer=F.SourceContainerProducer(gp(port(sid('build'),G.PortKind.RESULT)),
        gp(port(sid('build'),G.PortKind.ENTER)),ProviderFactReference(artifact,'#/containers/0'),
        'BUILD_MAP',F.BuiltinSlotKind.DICT)
    selector=PythonLiteral.from_python('fixed')
    guard=F.BuiltinSlotForwardingGuard(F.BuiltinSlotKind.DICT,producer.value,producer.value,
        producer.value,selector,('Q_EXACT_BUILTIN_PROTOCOL_AND_ISOLATED_PRODUCER_INSTANCE',))
    control=F.SourceForwardingControlWindow(function,ProviderFactReference(artifact,'#/functions/0'),
        path,start,(),(),(),())
    proof=F.ContentReferenceForwardingProof(ContentAccessRef(unit,0,OperationID('write')),
        ContentAccessRef(unit,1,OperationID('read')),function,gp(port(sid('read'),G.PortKind.RESULT)),
        producer,selector,(ReferenceSlice((),(),(),()),),(),control,guard)
    assert proof.unconditional_status is ProofStatus.UNKNOWN
    assert proof.logical_equivalence is EquivalenceClaim.UNKNOWN
    assert proof.transformation_authorized is False
    locator=SourceArtifactLocator(artifact,artifact.sha256,artifact.size_bytes,'canonical-json')
    namespace=GraphNamespace(locator,SOURCE_REPRESENTATION,RegionID('root'))
    provider=SourceContentForwardingProvider.__new__(SourceContentForwardingProvider)
    provider.namespace=namespace;provider.source=NS(locator=locator)
    provider.view=NS(source_port=lambda value:Point(namespace,ValueID(_digest(value)),None))
    provider.in_edges=defaultdict(list);provider.out_edges=defaultdict(list)
    provider._attach(proof)
    emitted=next(iter(provider.out_edges.values()))[0]
    assert emitted.meaning is DependencyMeaning.REFERENCE
    assert emitted.strength is DependencyStrength.MAY and emitted.conditions
