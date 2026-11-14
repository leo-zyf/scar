"""Shared leaf plans and original F4 call/binding fields form a pipeline."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from scar.analysis.source_namespace_interfaces import NamespacePlanView
from scar.compiler.model import SourceRef,ArrayRegion,Expr
from scar.compiler.pipeline_model import (CalleeGuard,RegionCall,compose_pipeline,document,pipeline_from_dict,
    prepare_pipeline_leaf,pipeline_digest)
from scar.compiler.pipeline_frontend import _PipelineLowerer,_global_functions
from scar.compiler.graph_frontend import GraphRegionRejected
from scar.optimization.optimizer import optimize_region
from scar.ir.program import source_productions as P,normal_graph as G,normal_links as F
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.ids import LocationID
from scar.ir.program.model import ValueType,ValueKind


def _leaf():
    source=SourceRef('/original/arrays.py','a'*64,'leaf',1,0,'pycode-v2:'+'b'*64,
        ({'kind':'original_f4','unit':'unit','step':'original-step'},))
    region=ArrayRegion('leaf',source,('x',),(Expr('x','input',attrs={'name':'x'}),
        Expr('c','const',attrs={'value':2}),Expr('y','mul',('x','c'))),'y')
    return optimize_region(region)


def test_shared_wire_leaf_table_and_explicit_nonprofitable_component_context():
    original=_leaf();assert original.proof['lowering_eligible'] is False
    leaf=prepare_pipeline_leaf(original)
    assert leaf.region is original.region and leaf.cost is original.cost
    assert leaf.proof['standalone_lowering_eligible'] is False and not leaf.proof['selected']
    assert leaf.proof['lowering_context']=='PIPELINE_NATIVE_HOST_COMPONENT'
    assert leaf.cost['decision']=='not_profitable' and not original.proof['lowering_eligible']
    guard=CalleeGuard(leaf.region.source,('arbitrary_global',))
    pipeline=compose_pipeline(leaf.region.source,('x',),(
        RegionCall('a',leaf,('arg:0',),'v:a',guard),RegionCall('b',leaf,('v:a',),'v:b',guard)),'v:b')
    wire=document(pipeline);assert len(wire['leaf_plans'])==1
    assert all('source' not in call['callee'] for call in wire['calls'])
    restored=pipeline_from_dict(wire)
    assert restored.calls[0].plan is restored.calls[1].plan
    assert restored.calls[0].callee.source is restored.calls[0].plan.region.source
    assert pipeline_digest(restored)==pipeline_digest(pipeline)
    with pytest.raises(ValueError):prepare_pipeline_leaf(optimize_region(original.region,memory_budget_bytes=0))
    with pytest.raises(ValueError,match='reasons'):
        prepare_pipeline_leaf(replace(original,cost={**original.cost,'reason':['unknown_target']}))
    conflicting=replace(leaf,region=replace(leaf.region,source=replace(leaf.region.source,code_key='pycode-v2:'+'c'*64)))
    with pytest.raises(ValueError,match='global lookup'):
        compose_pipeline(leaf.region.source,('x',),(pipeline.calls[0],RegionCall('b',conflicting,('v:a',),'v:b',
            CalleeGuard(conflicting.region.source,guard.lookup_path))),'v:b')


def test_original_f4_calls_and_local_aliases_supply_call_order_and_value_routes():
    leaf=prepare_pipeline_leaf(_leaf());unit='unit';key=G.SourceTemplateKey(4,P.ProductionPurpose.STATEMENT)
    leaf_key=G.SourceTemplateKey(9,P.ProductionPurpose.STATEMENT);body=P.ProductionRegionID('body')
    ref=ValueType(ValueKind.REFERENCE);world=ValueType(ValueKind.STATE,location=LocationID('world'))
    p=lambda i:P.ProductionParameter(body,i);r=lambda name,i=0:P.ProductionResult(P.ProductionStepID(name),i)
    rows=[]
    def step(name,instruction,args,types,kind=C.FALLTHROUGH):
        row=P.ProductionStep(P.ProductionStepID(name),body,P.ProductionOccurrence(4),instruction,args,
            (P.ProductionCompletion(kind,types),));rows.append(row);return row
    read_fn=step('read-global',P.AccessBinding(P.BindingActionKind.READ,0),(p(2),),(ref,world))
    read_x=step('read-x',P.AccessBinding(P.BindingActionKind.READ,1),(r('read-global',1),),(ref,world))
    first=step('first',P.ApplyLanguageAction(P.LanguageActionKind.CALL),
        (r('read-global'),r('read-x'),r('read-x',1)),(ref,world))
    step('bind',P.AccessBinding(P.BindingActionKind.WRITE,2),(r('first'),r('first',1)),(world,))
    read_alias=step('read-alias',P.AccessBinding(P.BindingActionKind.READ,3),(r('bind'),),(ref,world))
    second=step('second',P.ApplyLanguageAction(P.LanguageActionKind.CALL),
        (r('read-global'),r('read-alias'),r('read-alias',1)),(ref,world))
    step('return',P.TransferCompletion(C.RETURN),(r('second'),r('second',1)),(ref,world),C.RETURN)
    region=P.ProductionRegion(body,None,P.ProductionRegionKind.DEFERRED,P.ProductionOccurrence(4),None,
        (ref,ref,world),(P.ProductionCompletion(C.RETURN,(ref,world)),),tuple(row.id for row in rows))
    plan=P.SourceProductionPlan(None,P.ProductionOccurrence(4),key.purpose,body,(region,),tuple(rows),(),
        compilation_flow=P.CompilationContextFlow((P.CompilationContextPort(body,1,1),),(),()))
    context=NamespacePlanView(unit,plan);ports=[]
    def index(port):
        if port not in ports:ports.append(port)
        return ports.index(port)
    value=lambda x:index(context.value(x).port)
    formal=tuple(value(p(i)) for i in range(3))
    global_ns=F.Namespace(F.NamespaceKind.MODULE,0,'arbitrary_global')
    local=F.Namespace(F.NamespaceKind.LEXICAL,1,'x');alias=F.Namespace(F.NamespaceKind.LEXICAL,2,'alias')
    leaf_value=index(G.Port(leaf_key,P.ProductionStepID('definition'),G.PortKind.RESULT,0,C.FALLTHROUGH))
    environment=NS(kind=P.DeferredBodyKind.FUNCTION,captures=(),owned_cells=(),
        parameters=(NS(kind=P.DeferredParameterKind.POSITIONAL_OR_KEYWORD),))
    function=NS(template=key,body=body,body_scope=context.owner(body),environment=environment,
        parameter_names=('x',),formals=formal)
    declared=NS(template=leaf_key,value=leaf_value,environment=environment)
    names=(F.NameAccess(local,formal[1],True),F.NameAccess(global_ns,value(r('read-global')),False),
        F.NameAccess(F.Namespace(F.NamespaceKind.BUILTINS,0,global_ns.name),value(r('read-global')),False),
        F.NameAccess(global_ns,leaf_value,True),F.NameAccess(local,value(r('read-x')),False),
        F.NameAccess(alias,value(r('first')),True),F.NameAccess(alias,value(r('read-alias')),False))
    calls=tuple(F.Call(value(r('read-global')),(value(arg),),(),value(r(name)),value(before),value(r(name,1)))
        for name,arg,before in (('first',r('read-x'),r('read-x',1)),('second',r('read-alias'),r('read-alias',1))))
    facts=NS(unit=unit,ports=tuple(ports),aliases=(),functions=(declared,),names=names,calls=calls,operations=(),constants=())
    source=NS(unit=unit,facts=facts,material=NS(interfaces={'links':{}}))
    lowerer=_PipelineLowerer(source,function,NS(plan=lambda *args:context),_global_functions(facts,{leaf_key:leaf}))
    lowered,output,origins=lowerer.lower()
    assert [(c.inputs,c.output) for c in lowered]==[(('arg:0',),'call:0:output'),(('call:0:output',),'call:1:output')]
    assert output=='call:1:output' and lowered[0].plan is lowered[1].plan
    assert all(c.callee.lookup_path==('arbitrary_global',) for c in lowered)
    assert all(c.origins[0]['step'].endswith(name) for c,name in zip(lowered,('first','second')))
    with pytest.raises(GraphRegionRejected,match='unique admitted'):
        _PipelineLowerer(source,function,NS(plan=lambda *args:context),{}).lower()
    # A later unused call remains in source order; its warnings/exceptions
    # cannot disappear just because the returned value comes from call one.
    changed=replace(plan,steps=(*plan.steps[:-1],replace(plan.steps[-1],arguments=(r('first'),r('second',1)))))
    second_context=NamespacePlanView(unit,changed)
    calls,output,_=_PipelineLowerer(source,function,NS(plan=lambda *args:second_context),
        _global_functions(facts,{leaf_key:leaf})).lower()
    assert len(calls)==2 and output=='call:0:output'
    alias_output=NamespacePlanView(unit,replace(plan,steps=(*plan.steps[:-1],
        replace(plan.steps[-1],arguments=(p(1),r('second',1))))))
    with pytest.raises(GraphRegionRejected,match='fresh-output'):
        _PipelineLowerer(source,function,NS(plan=lambda *args:alias_output),
            _global_functions(facts,{leaf_key:leaf})).lower()
