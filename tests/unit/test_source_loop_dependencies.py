"""Shared F4 regions cannot splice an in-loop entry to an outside return."""
from types import SimpleNamespace as NS

from scar.analysis.source_control import loop_dependencies
from scar.analysis.source_productions import _digest
from scar.ir.program import source_productions as P, normal_graph as G, normal_graph as F
from scar.ir.program.source_template_links import SourceTemplateKey
from scar.ir.program.source_contexts import ContextState, OwnerRef
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.model import ValueKind, ValueType
from scar.ir.program.ids import LocationID
from scar.ir.program.providers import ProviderArtifact
from scar.ir.record_codec import encode


def test_nested_loop_body_membership_balances_shared_region_calls_and_stops_at_exit():
    reference = ValueType(ValueKind.REFERENCE)
    boolean = ValueType(ValueKind.DATA, 'builtins.bool')
    world = ValueType(ValueKind.STATE, location=LocationID('world'))
    source = P.ProductionOccurrence(0); key = SourceTemplateKey(0, P.ProductionPurpose.VALUE)
    context = ContextState(OwnerRef(0,'ROOT'),0,0,0)
    rid = P.ProductionRegionID; sid = P.ProductionStepID
    param = lambda region,i: P.ProductionParameter(rid(region),i)
    result = lambda name,i: P.ProductionResult(sid(name),i)
    normal = lambda *types: P.ProductionCompletion(C.FALLTHROUGH,types)
    regions=[]; steps=[]
    def region(name,parent,kind,parameters,types,names):
        regions.append(P.ProductionRegion(rid(name),None if parent is None else rid(parent),kind,
            source,context,parameters,(normal(*types),),tuple(sid(n) for n in names)))
    def step(name,owner,instruction,args,types):
        steps.append(P.ProductionStep(sid(name),rid(owner),source,instruction,args,(normal(*types),)))
    def invoke(name,owner,target,world_value):
        step(name,owner,P.ApplyRegion(P.ProductionRegionCall(rid(target),(param(owner,0),world_value))),
            (), (world,))
    region('root',None,P.ProductionRegionKind.EXECUTION,(reference,world),(world,),
        ('outer_entry','outside_shared','outside_call','root_return'))
    region('outer','root',P.ProductionRegionKind.LOOP,(reference,world),(world,),('outer_drive',))
    region('outer_cond','outer',P.ProductionRegionKind.LOOP_CONDITION,(reference,world),(boolean,world),('outer_truth','outer_tested'))
    region('outer_body','outer',P.ProductionRegionKind.LOOP_BODY,(reference,world),(world,),
        ('inside_shared','inner_entry','outer_return'))
    region('inner','outer_body',P.ProductionRegionKind.LOOP,(reference,world),(world,),('inner_drive',))
    region('inner_cond','inner',P.ProductionRegionKind.LOOP_CONDITION,(reference,world),(boolean,world),('inner_truth','inner_tested'))
    region('inner_body','inner',P.ProductionRegionKind.LOOP_BODY,(reference,world),(world,),('inner_call','inner_return'))
    region('shared','root',P.ProductionRegionKind.EXECUTION,(reference,world),(world,),('shared_call','shared_return'))
    invoke('outer_entry','root','outer',param('root',1))
    invoke('outside_shared','root','shared',result('outer_entry',0))
    step('outside_call','root',P.ApplyLanguageAction(P.LanguageActionKind.CALL),
        (param('root',0),result('outside_shared',0)),(reference,world))
    step('root_return','root',P.TransferCompletion(C.FALLTHROUGH),(result('outside_call',1),),(world,))
    for name in ('outer','inner'):
        step(name+'_drive',name,P.LoopControl(rid(name+'_cond'),rid(name+'_body'),invariant_count=1),
            (param(name,0),param(name,1)),(world,))
        step(name+'_truth',name+'_cond',P.ApplyLanguageAction(P.LanguageActionKind.TRUTH),
            (param(name+'_cond',0),param(name+'_cond',1)),(boolean,world))
        step(name+'_tested',name+'_cond',P.TransferCompletion(C.FALLTHROUGH),
            (result(name+'_truth',0),result(name+'_truth',1)),(boolean,world))
    invoke('inside_shared','outer_body','shared',param('outer_body',1))
    invoke('inner_entry','outer_body','inner',result('inside_shared',0))
    step('outer_return','outer_body',P.TransferCompletion(C.FALLTHROUGH),(result('inner_entry',0),),(world,))
    for owner,name in (('inner_body','inner'),('shared','shared')):
        step(name+'_call',owner,P.ApplyLanguageAction(P.LanguageActionKind.CALL),
            (param(owner,0),param(owner,1)),(reference,world))
        step(name+'_return',owner,P.TransferCompletion(C.FALLTHROUGH),(result(name+'_call',1),),(world,))
    basis=P.ProductionInput('/source/nested.py','/source',*('a'*64 for _ in range(8)))
    plan=P.SourceProductionPlan(basis,source,P.ProductionPurpose.VALUE,rid('root'),tuple(regions),tuple(steps),())
    port=lambda owner,kind,i=0,completion=None:G.Port(key,owner,kind,i,completion)
    ports=[port(rid('root'),G.PortKind.PARAMETER,0)]
    calls=[]
    for owner,name in (('root','outside'),('inner_body','inner'),('shared','shared')):
        start=len(ports)
        ports.extend((port(rid(owner),G.PortKind.PARAMETER,0),port(rid(owner),G.PortKind.PARAMETER,1),
            port(sid(name+'_call'),G.PortKind.RESULT,0,C.FALLTHROUGH),
            port(sid(name+'_call'),G.PortKind.RESULT,1,C.FALLTHROUGH)))
        calls.append(F.Call(start,(),(),start+2,start+1,start+3))
    scope=F.CodeScopeRef('unit',0)
    function=NS(value=0,template=key,body=rid('root'),body_scope=NS(owner=scope))
    unit=NS(unit='unit',ports=tuple(ports),functions=(function,),calls=tuple(calls),names=(),
        scope_groups=(F.FactScopeGroup(scope,(F.FactRange(F.FactTable.CALLS,0,len(calls)),)),))
    sealed=NS(origin=NS(plan_sha256=_digest(encode(plan,record_tags=P.PRODUCTION_RECORD_TAGS))))
    artifact=ProviderArtifact('/source/result.json','0'*64,0,'fixture')
    material=NS(unit='unit',closure=lambda *_:{key:plan},interfaces=dict(nodes={key:sealed},links={}),
        result_artifact=artifact,validate_parents=lambda:None,
        fact_store=NS(load=lambda _:unit,rows={'unit':dict(facts='/source/facts.json',sha256='0'*64,bytes=0)}))
    report=loop_dependencies(material,function)
    assert len(report['loops'])==2
    outer,inner=report['loops']
    assert outer['parent'] is None and inner['parent']==outer['header']
    # The shared region has two original RETURN routes; only the matching one
    # can participate in this loop's body. The other leads to outside_call.
    call_owners=lambda loop:{row['result']['port']['owner']['wire'] for row in loop['body_calls']}
    assert call_owners(outer)=={str(sid('shared_call')),str(sid('inner_call'))}
    assert call_owners(inner)=={str(sid('inner_call'))}
    assert len(outer['control_backedges'])==len(inner['control_backedges'])==1
    for loop in (outer,inner):
        assert [slot['role'] for slot in loop['slots']]==['INVARIANT_PREFIX','CARRIED_SUFFIX']
        assert not loop['slots'][0]['backedges'] and loop['slots'][1]['backedges']
        assert all(slot['content_invariance']=='UNPROVED' for slot in loop['slots'])
    assert report['iterations_unrolled'] is False and report['contents_proven_invariant'] is False
