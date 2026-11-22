"""Invocation-qualified queries retain one shared, conditional F4 body."""
from collections import defaultdict,deque
from types import SimpleNamespace as NS

import pytest

from scar.analysis.program_view import ProgramView,SOURCE_REPRESENTATION
from scar.analysis.source_invocation_view import SourceInvocationProvider
from scar.ir.program import source_value_routes as I, source_productions as P, normal_graph as G, normal_targets as T, view as V
from scar.ir.program.call_edges import InvocationKey,CallRecordAnchor
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.dependencies import (DependencyPoint as Point,DependencyEdge as Edge,
    DependencyCondition as Condition,DependencyEvidence as Evidence,DependencyKind as K,
    DependencyMeaning as M,DependencyStrength as S,GraphNamespace)
from scar.ir.program.ids import ProviderID,RegionID,OperationID,ValueID,UseID,LocationID
from scar.ir.program.providers import ProviderArtifact
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.program.model import ValueType,ValueKind


def _fixture():
    view=ProgramView(None);view.catalog=object();key=G.SourceTemplateKey(1,P.ProductionPurpose.STATEMENT)
    root,loop,nested,caller=(P.ProductionRegionID(x) for x in ('body','loop','other-code','caller'))
    calculate,repeat,foreign=(P.ProductionStepID(x) for x in ('calculate','repeat','foreign'))
    ref=ValueType(ValueKind.REFERENCE);world=ValueType(ValueKind.STATE,location=LocationID('world'))
    result=P.ProductionCompletion(C.RETURN,(ref,world))
    regions=tuple(P.ProductionRegion(r,None,P.ProductionRegionKind.EXECUTION,None,None,
        (ref,world),(result,),steps) for r,steps in ((root,(calculate,)),(loop,(repeat,)),(nested,(foreign,)),(caller,())))
    steps=tuple(P.ProductionStep(s,r,None,P.TransferCompletion(C.RETURN),(),(result,))
        for s,r in ((calculate,root),(repeat,loop),(foreign,nested)))
    ports=tuple(P.CompilationContextPort(r,scope,scope) for r,scope in ((root,1),(loop,1),(nested,2),(caller,3)))
    plan=NS(source=NS(occurrence=1),purpose=key.purpose,regions=regions,steps=steps,
        compilation_flow=NS(ports=ports))
    artifact=ProviderArtifact('/sealed/source.json','a'*64,1,'test.f4')
    locator=SourceArtifactLocator(artifact,artifact.sha256,1,'canonical-json')
    namespace=GraphNamespace(locator,SOURCE_REPRESENTATION,RegionID('unit'))
    source=NS(unit='unit',namespace=namespace,locator=locator,nodes={},records={},plans={key:plan},
        incoming_edges=defaultdict(list),outgoing_edges=defaultdict(list),check=lambda *a:None,ensure=lambda *a:None,
        material=NS(interfaces=dict(nodes={key:NS()},links={})),_expand=lambda keys:None)
    identities={}
    def node(original,kind=None,owner=None,type_=None):
        if original in identities:return identities[original]
        cls=RegionID if type(original)is P.ProductionRegionID else OperationID if type(original)is P.ProductionStepID else UseID if original.kind is G.PortKind.ARGUMENT else ValueID
        point=Point(namespace,cls(str(len(identities))),None);identities[original]=point
        source.nodes[point.node]=V.SourceNodeProjection(point,V.SourceAnchor('unit',key,original),kind,
            V.ProgramNodeRole.PROVIDER,owner,type_,artifact)
        source.records[point.node]=original
        return point
    for region in regions:
        point=node(region.id,V.ProgramEntityKind.REGION);source.records[point.node]=region
    for step in steps:
        point=node(step.id,V.ProgramEntityKind.OPERATION,node(step.region));source.records[point.node]=step
    gp=lambda owner,kind,i=0,completion=None:T.GlobalPort('unit',G.Port(key,owner,kind,i,completion))
    formal=gp(root,G.PortKind.PARAMETER);formal_world=gp(root,G.PortKind.PARAMETER,1)
    returned=gp(root,G.PortKind.RESULT,completion=C.RETURN);returned_world=gp(root,G.PortKind.RESULT,1,C.RETURN)
    arguments=(gp(calculate,G.PortKind.ARGUMENT),gp(calculate,G.PortKind.ARGUMENT,1))
    outputs=(gp(calculate,G.PortKind.RESULT,completion=C.RETURN),gp(calculate,G.PortKind.RESULT,1,C.RETURN))
    caller_input=gp(caller,G.PortKind.PARAMETER);caller_output=gp(caller,G.PortKind.RESULT,completion=C.RETURN)
    for port in (formal,formal_world,returned,returned_world,*arguments,*outputs,caller_input,caller_output):
        node(port.port,owner=node(port.port.owner),type_=world if port.port.ordinal==1 else ref)
    source.port=lambda port,context=None:Point(namespace,node(port).node,context)
    source._identity=lambda cls,template,original:node(original).node
    source.expand=lambda point:tuple(node(step) for step in source.records[point.node].steps) if type(source.records[point.node])is P.ProductionRegion else ()
    source.effects=lambda point:NS(unresolved_fields=('writes',),missing_rules=(),
        rule=NS(resolution=NS(value='DYNAMIC_TARGET_OR_PROTOCOL'),instruction='ORIGINAL_PROTOCOL'))
    view.units['unit']=source;view.namespaces[namespace]=source
    def edge(left,right,kind,meaning=M.DEPENDENCE,conditions=()):
        edge=Edge(left,right,kind,meaning,(Evidence('original.f4',(locator,),(left,right)),),conditions)
        source.outgoing_edges[left.node].append(edge);source.incoming_edges[right.node].append(edge)
    for before,arg,out,after,kind in ((formal,arguments[0],outputs[0],returned,K.VALUE_USE),
            (formal_world,arguments[1],outputs[1],returned_world,K.STATE_INTERFACE)):
        edge(node(before.port),node(arg.port),kind)
        edge(node(arg.port),node(calculate),K.STATE_INPUT if kind is K.STATE_INTERFACE else K.OPERAND)
        edge(node(calculate),node(out.port),K.STATE_OUTPUT if kind is K.STATE_INTERFACE else K.RESULT,
            conditions=(Condition('original.completion','RETURN',node(out.port),operands=(node(out.port),)),))
        edge(node(out.port),node(after.port),kind)
    edge(node(repeat),node(repeat),K.CONTROL_FLOW,M.STRUCTURE,
        (Condition('original.loop','BACKEDGE',node(formal.port),operands=(node(formal.port),)),))
    edge(node(calculate),node(nested),K.IMPLEMENTATION,M.STRUCTURE)
    anchor=lambda i:CallRecordAnchor('/trace/events',i,'b'*64,i)
    bindings=[]
    for ordinal in (1,2):
        scope=I.SourceCallScope('c'*64,InvocationKey(1,1,ordinal+1))
        caller_scope=I.SourceCallScope('c'*64,InvocationKey(1,1,1))
        routes=(I.SourceInvocationRoute(I.SourceInvocationPort(caller_scope,caller_input,anchor(ordinal)),
            I.SourceInvocationPort(scope,formal),'argument',(),()),
            I.SourceInvocationRoute(I.SourceInvocationPort(scope,returned),
                I.SourceInvocationPort(caller_scope,caller_output,anchor(ordinal)),'data_return',(),()))
        bindings.append(I.SourceInvocationBinding(ProviderID(str(ordinal)),scope,caller_scope,ProviderID('activation'),
            ProviderID('call'),T.Body('unit',key,root),anchor(10+ordinal),anchor(20+ordinal),routes,()))
    provider=SourceInvocationProvider(view,bindings)
    return view,provider,bindings,NS(formal=formal,world=formal_world,returned=returned,returned_world=returned_world,
        calculate=node(calculate),loop=node(repeat),foreign=node(nested),source=source,plan=plan)


def _reachable(view,start,target):
    pending=deque((start,));seen=set();edges=[]
    while pending:
        point=pending.popleft()
        if point==target:return edges,seen
        if point in seen:continue
        seen.add(point)
        for edge in view.consumers(point).edges:
            edges.append(edge);pending.append(edge.target)
    raise AssertionError('original local dependency did not reach its scoped return')


def test_actual_input_crosses_shared_body_to_same_scope_return_and_original_caller_boundary():
    view,provider,bindings,original=_fixture()
    first,second=bindings
    start=provider.port(first.routes[0].source);target=provider.port(first.routes[-1].target)
    edges,seen=_reachable(view,start,target)
    assert any(edge.kind is K.OPERAND for edge in edges)
    assert any(edge.kind is K.RESULT for edge in edges)
    local=provider.scoped(first.call_scope,original.calculate)
    assert all(edge.strength is S.MAY for edge in view.consumers(local).edges)
    assert any(f.kind=='SCOPED_SOURCE_EFFECT_FIELD_OPEN' and f.detail=='writes' for f in view.consumers(local).frontiers)
    assert any(f.kind=='INTERNAL_CALL_PROTOCOL_SELECTION_OPEN' for f in view.consumers(local).frontiers)
    other=provider.scoped(second.call_scope,original.calculate)
    assert local!=other and other not in seen
    assert provider.shared(local)==provider.shared(other)==original.calculate
    assert view.explain(local)['occurrence_evidence']==(first,)
    assert provider.original(view.get(local).owner).call_scope==first.call_scope
    assert provider._body(first.call_scope)is provider._body(second.call_scope)
    assert original.source.plans[next(iter(original.source.plans))]is original.plan
    expanded=view.expand(provider.body(first.call_scope))
    assert local in expanded and all(provider.original(p).call_scope==first.call_scope for p in expanded)
    assert not view.expand(target)  # attempt boundary does not merge into caller's static values
    assert view.get(target).owner is None


def test_world_and_loop_edges_keep_scope_conditions_and_foreign_code_remains_frontier():
    view,provider,bindings,original=_fixture();scope=bindings[0].call_scope
    _reachable(view,provider.scoped(scope,original.world),provider.scoped(scope,original.returned_world))
    loop=provider.scoped(scope,original.loop);edges=view.consumers(loop).edges
    assert len(edges)==1 and edges[0].source==edges[0].target==loop
    assert edges[0].meaning is M.STRUCTURE and edges[0].strength is S.MAY
    condition=next(c for c in edges[0].conditions if c.name=='BACKEDGE')
    assert provider.original(condition.subject).call_scope==scope
    assert all(provider.original(p).call_scope==scope for p in condition.operands)
    with pytest.raises(ValueError,match='outside'):
        provider.scoped(scope,original.foreign)
    local=provider.scoped(scope,original.calculate)
    assert any(f.kind=='ORIGINAL_RELATION_OUTSIDE_INVOCATION_BODY' for f in view.consumers(local).frontiers)
