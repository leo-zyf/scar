"""Copy lowering retains the actual F4 method/CALL ABI and fresh-value Q."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from scar.analysis.source_namespace_interfaces import NamespacePlanView
from scar.compiler.array_copy_contract import COPY_ATTRS,COPY_REQUIRED_GUARDS
from scar.compiler.graph_frontend import _GraphLowerer,GraphRegionRejected
from scar.ir.literals import PythonLiteral
from scar.ir.program import source_productions as P,normal_graph as G,normal_links as F
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.ids import LocationID
from scar.ir.program.model import ValueType,ValueKind
from scar.ir.record_codec import encode


def _copy_graph(form):
    unit='original-copy-unit';root=P.ProductionRegionID('body')
    site=P.ProductionOccurrence(41);key=G.SourceTemplateKey(41,P.ProductionPurpose.STATEMENT)
    ref=ValueType(ValueKind.REFERENCE);world=ValueType(ValueKind.STATE,location=LocationID('world'))
    parameter=lambda i:P.ProductionParameter(root,i)
    result=lambda name,i=0:P.ProductionResult(P.ProductionStepID(name),i)
    steps=[]
    def step(name,instruction,args,types,kind=C.FALLTHROUGH):
        row=P.ProductionStep(P.ProductionStepID(name),root,site,instruction,args,
            (P.ProductionCompletion(kind,types),));steps.append(row);return row
    read=step('read',P.AccessBinding(P.BindingActionKind.READ,7),(parameter(2),),(ref,world))
    method=step('method',P.ApplyLanguageAction(P.LanguageActionKind.ATTRIBUTE_GET,(P.SlotRef(41,'FIELD',0),)),
        (result('read'),result('read',1)),(ref,world))
    literal=step('order',P.DefineLiteral(PythonLiteral.from_python('C',exact=True)),(),(ref,))if form else None
    arguments=(result('method'),*((result('order'),)if form else()),result('method',1))
    call=step('call',P.ApplyLanguageAction(P.LanguageActionKind.CALL),arguments,(ref,world))
    step('return',P.TransferCompletion(C.RETURN),(result('call'),result('call',1)),(ref,world),C.RETURN)
    body=P.ProductionRegion(root,None,P.ProductionRegionKind.DEFERRED,site,None,(ref,ref,world),
        (P.ProductionCompletion(C.RETURN,(ref,world)),),tuple(s.id for s in steps))
    plan=P.SourceProductionPlan(None,site,P.ProductionPurpose.STATEMENT,root,(body,),tuple(steps),(),
        compilation_flow=P.CompilationContextFlow((P.CompilationContextPort(root,2,2),),(),()))
    context=NamespacePlanView(unit,plan);ports=[]
    def index(value):
        port=context.value(value).port
        if port not in ports:ports.append(port)
        return ports.index(port)
    formals=tuple(index(parameter(i))for i in range(3))
    read_value=index(result('read'));read_world=index(result('read',1))
    method_value=index(result('method'));method_world=index(result('method',1))
    pos=(index(result('order')),)if form=='POSITIONAL'else()
    kw=(('order',index(result('order'))),)if form=='KEYWORD'else()
    call_fact=F.Call(method_value,pos,kw,index(result('call')),method_world,index(result('call',1)))
    namespace=F.Namespace(F.NamespaceKind.LEXICAL,2,'argument')
    facts=NS(unit=unit,ports=ports,constants=(),operations=(),
        names=(F.NameAccess(namespace,formals[1],True),F.NameAccess(namespace,read_value,False)),
        attributes=(F.Attribute(read_value,'copy',method_value,F.AttributeAction.GET,read_world,method_world),),
        calls=(call_fact,))
    store=NS(rows={unit:dict(facts='/original/facts.json',sha256='0'*64,bytes=1)})
    source=NS(unit=unit,facts=facts,material=NS(interfaces={'links':{}},fact_store=store))
    function=NS(template=key,body=root,body_scope=context.owner(root),parameter_names=('argument',),formals=formals,
        environment=NS(kind=P.DeferredBodyKind.FUNCTION,captures=(),owned_cells=(),parameters=()))
    return _GraphLowerer(source,function,NS(plan=lambda *args:context)),facts,context


def test_copy_uses_original_method_call_world_facts_not_storage_or_pure_tokens():
    for form in(None,'POSITIONAL','KEYWORD'):
        lowerer,facts,context=_copy_graph(form)
        nodes,output,origins=lowerer.lower()
        assert nodes[-1].op=='copy'and nodes[-1].inputs==('n0',)and nodes[-1].attrs==COPY_ATTRS
        assert output==nodes[-1].id
        witness=next(o['copy_contract']for o in origins if o.get('expression')==output)
        assert witness['call_fact']['locator']=='#/calls/0'
        assert witness['attribute_fact']['locator']=='#/attributes/0'
        assert witness['receiver']['port']==encode(facts.ports[3])
        assert witness['world_after']['port']['owner']['wire']==P.ProductionStepID('call').wire
        assert witness['attribute_world_after']['port']['owner']['wire']==P.ProductionStepID('method').wire
        assert set(COPY_REQUIRED_GUARDS)<=set(witness['required_guards'])
        assert witness['materialization']=='FRESH_INDEPENDENT_C_ORDER_ALLOCATION'
        assert witness['may_raise']!='NONE'and not witness['runtime_value_or_storage_identity_proven']
    lowerer,facts,context=_copy_graph('KEYWORD')
    facts.calls=(replace(facts.calls[0],result_world=facts.calls[0].world),)
    lowerer.array_call_facts.clear()
    lowerer.array_call_facts[context.result(context.steps[P.ProductionStepID('call')]).port].append((0,facts.calls[0]))
    with pytest.raises(GraphRegionRejected)as caught:lowerer.lower()
    assert caught.value.code=='ARRAY_COPY_CALL_ABI'
