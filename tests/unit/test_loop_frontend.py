"""An original structured loop is shared, including its lexical carry ABI."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from scar.analysis.source_namespace_contents import NamespacePlanView
from scar.compiler.graph_frontend import GraphRegionRejected
from scar.compiler.loop_admission import _LoopLowerer
from scar.compiler.loop_admission import ArrayLoopPlan, document, loop_from_dict, loop_digest
from scar.compiler.model import SourceRef,ArrayRegion,Expr
from scar.compiler.pipeline_frontend import _Callee
from scar.compiler.pipeline_frontend import CalleeGuard, RegionCall, compose_pipeline
from scar.ir.program import source_productions as P, normal_graph as G, normal_graph as F
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.ids import LocationID
from scar.ir.program.model import ValueType,ValueKind
from scar.ir.program.structured import PythonProtocolKind
from scar.optimization.optimizer import optimize_region


def _original_loop():
    unit='unit';key=G.SourceTemplateKey(4,P.ProductionPurpose.STATEMENT)
    ref=ValueType(ValueKind.REFERENCE);world=ValueType(ValueKind.STATE,location=LocationID('world'))
    ids={name:P.ProductionRegionID(name) for name in ('entry','loop','next','exhausted','body','else')}
    p=lambda reg,i:P.ProductionParameter(ids[reg],i)
    r=lambda name,i=0:P.ProductionResult(P.ProductionStepID(name),i)
    rows=[];groups={name:[] for name in ids}
    def step(reg,name,ins,args,types,kind=C.FALLTHROUGH):
        row=P.ProductionStep(P.ProductionStepID(name),ids[reg],P.ProductionOccurrence(4),ins,args,
            (P.ProductionCompletion(kind,types),));rows.append(row);groups[reg].append(row.id)
    read=lambda reg,name,after:step(reg,name,P.AccessBinding(P.BindingActionKind.READ,0),(after,),(ref,world))
    call=lambda reg,name,fn,arg,after:step(reg,name,P.ApplyLanguageAction(P.LanguageActionKind.CALL),
        (fn,arg,after),(ref,world))
    protocol=lambda reg,name,kind,args,types:step(reg,name,P.ApplyPythonProtocol(PythonProtocolKind(kind),()),args,types)
    transfer=lambda reg,name,args,types,kind=C.FALLTHROUGH:step(reg,name,P.TransferCompletion(kind),args,types,kind)
    read('entry','range-name',p('entry',3));read('entry','trip',r('range-name',1))
    call('entry','range-call',r('range-name'),r('trip'),r('trip',1))
    protocol('entry','iterator','GET_ITERATOR',(r('range-call'),r('range-call',1)),(ref,world))
    step('entry','invoke-loop',P.ApplyRegion(P.ProductionRegionCall(ids['loop'],(r('iterator'),r('iterator',1)))),(),(world,))
    read('entry','final-carry',r('invoke-loop'))
    transfer('entry','return',(r('final-carry'),r('final-carry',1)),(ref,world),C.RETURN)
    condition=P.IterationCondition(P.ProductionRegionCall(ids['next'],(p('loop',0),p('loop',1))),ids['exhausted'],1)
    step('loop','control',P.LoopControl(condition,ids['body'],ids['else'],1,P.SourceLoopTarget(4)),
        (p('loop',0),p('loop',1)),(world,))
    protocol('next','next-item','COMPREHENSION_NEXT',(p('next',0),p('next',1)),(ref,world))
    transfer('next','next-done',(r('next-item'),r('next-item',1)),(ref,world))
    protocol('exhausted','match','MATCH_STOP_ITERATION',(p('exhausted',0),p('exhausted',1)),(ref,))
    transfer('exhausted','match-done',(r('match'),),(ref,))
    step('body','bind-item',P.AccessBinding(P.BindingActionKind.WRITE,0),(p('body',2),p('body',1)),(world,))
    read('body','callee',r('bind-item'));read('body','carry',r('callee',1))
    call('body','body-call',r('callee'),r('carry'),r('carry',1))
    step('body','bind-carry',P.AccessBinding(P.BindingActionKind.WRITE,0),(r('body-call'),r('body-call',1)),(world,))
    transfer('body','body-done',(r('bind-carry'),),(world,))
    transfer('else','else-done',(p('else',1),),(world,))
    specifications=(('entry',P.ProductionRegionKind.DEFERRED,(ref,ref,ref,world),(ref,world),C.RETURN),
        ('loop',P.ProductionRegionKind.LOOP,(ref,world),(world,),C.FALLTHROUGH),
        ('next',P.ProductionRegionKind.LOOP_CONDITION,(ref,world),(ref,world),C.FALLTHROUGH),
        ('exhausted',P.ProductionRegionKind.HANDLER,(ref,world),(ref,),C.FALLTHROUGH),
        ('body',P.ProductionRegionKind.LOOP_BODY,(ref,world,ref),(world,),C.FALLTHROUGH),
        ('else',P.ProductionRegionKind.ALTERNATIVE,(ref,world),(world,),C.FALLTHROUGH))
    regions=tuple(P.ProductionRegion(ids[name],None,kind,P.ProductionOccurrence(4),None,params,
        (P.ProductionCompletion(completion,types),),tuple(groups[name])) for name,kind,params,types,completion in specifications)
    plan=P.SourceProductionPlan(None,P.ProductionOccurrence(4),key.purpose,ids['entry'],regions,tuple(rows),(),
        compilation_flow=P.CompilationContextFlow(tuple(P.CompilationContextPort(reg.id,1,1) for reg in regions),(),()))
    context=NamespacePlanView(unit,plan);ports=[]
    def value(v):
        port=context.value(v).port
        if port not in ports:ports.append(port)
        return ports.index(port)
    formals=tuple(value(p('entry',i)) for i in range(4))
    clock=F.Namespace(F.NamespaceKind.LEXICAL,1,'clock');payload=F.Namespace(F.NamespaceKind.LEXICAL,2,'payload')
    ignored=F.Namespace(F.NamespaceKind.LEXICAL,3,'ordinary_iteration_name')
    names=[F.NameAccess(clock,formals[1],True),F.NameAccess(payload,formals[2],True)]
    for ns,v,write in ((clock,r('trip'),False),(payload,r('carry'),False),(payload,r('final-carry'),False),
            (payload,r('body-call'),True),(ignored,p('body',2),True)):
        names.append(F.NameAccess(ns,value(v),write))
    for name,global_name in (('range-name','range'),('callee','arbitrary_body')):
        for kind in (F.NamespaceKind.MODULE,F.NamespaceKind.BUILTINS):
            names.append(F.NameAccess(F.Namespace(kind,0,global_name),value(r(name)),False))
    calls=tuple(F.Call(value(r(fn)),(value(r(arg)),),(),value(r(result)),value(r(arg,1)),value(r(result,1)))
        for result,fn,arg in (('range-call','range-name','trip'),('body-call','callee','carry')))
    environment=NS(kind=P.DeferredBodyKind.FUNCTION,captures=(),owned_cells=(),
        parameters=tuple(NS(kind=P.DeferredParameterKind.POSITIONAL_OR_KEYWORD) for _ in range(2)))
    function=NS(template=key,body=ids['entry'],body_scope=context.owner(ids['entry']),environment=environment,
        parameter_names=('clock','payload'),formals=formals)
    facts=NS(unit=unit,ports=tuple(ports),aliases=(),functions=(),names=tuple(names),calls=calls,operations=(),constants=())
    source=NS(unit=unit,facts=facts,material=NS(interfaces={'links':{}}))
    reference=SourceRef('/original/module.py','a'*64,'leaf',1,0,'pycode-v2:'+'b'*64)
    leaf=optimize_region(ArrayRegion('leaf',reference,('x',),(Expr('x','input',attrs={'name':'x'}),
        Expr('c','const',attrs={'value':2}),Expr('a','add',('x','c')),Expr('y','mul',('a','c'))),'y'))
    body_source=replace(reference,qualname='shared_body',code_key='pycode-v2:'+'c'*64)
    body=compose_pipeline(body_source,('u',),(RegionCall('one',leaf,('arg:0',),'out',CalleeGuard(reference,('opaque_leaf',))),),'out')
    globals={'arbitrary_body':(_Callee('arbitrary_body',NS(),body),)}
    return source,function,plan,globals,body


def test_original_loop_terminal_carry_roles_and_rejected_completions():
    source,function,plan,globals,body=_original_loop()
    def lower(changed=plan):
        context=NamespacePlanView(source.unit,changed)
        return _LoopLowerer(source,function,NS(plan=lambda *args:context),globals).lower()
    state,trip,shared,guard,origins=lower()
    assert (state,trip)==('payload','clock') and shared is body
    loop=ArrayLoopPlan('loop',replace(body.source,qualname='entry'),function.parameter_names,state,trip,shared,guard,origins)
    wire=document(loop);restored=loop_from_dict(wire)
    assert restored.parameters==('clock','payload') and restored.body_callee.source is restored.body.source
    assert 'source' not in wire['body_callee'] and loop_digest(restored)==loop_digest(loop)
    backedge=next(row for row in origins if row.get('role')=='single_shared_loop')
    assert backedge['body_materializations']==1 and not backedge['loop_unrolled']
    assert backedge['backedge']['explicit_array_phi'] is False
    for name,kind,code in (('body-done',C.BREAK,'LOOP_COMPLETION'),('else-done',C.BREAK,'LOOP_ELSE')):
        changed=replace(plan,steps=tuple(replace(row,instruction=P.TransferCompletion(kind))
            if row.id.value==name else row for row in plan.steps))
        with pytest.raises(GraphRegionRejected) as caught:lower(changed)
        assert caught.value.code==code
    # The exhaustion predicate must consume and classify the original exception.
    changed=replace(plan,steps=tuple(replace(row,arguments=tuple(reversed(row.arguments)))
        if row.id.value=='match' else row for row in plan.steps))
    with pytest.raises(GraphRegionRejected) as caught:lower(changed)
    assert caught.value.code=='LOOP_EXHAUSTION_HANDLER'
