"""Actual repeated calls qualify shared ports without cloning their body."""
from dataclasses import replace
from types import SimpleNamespace as NS

from scar.analysis.source_invocation_view import derive_invocation_bindings,SourceInvocationProvider
from scar.ir.program import source_productions as P, normal_graph as G, normal_graph as F, normal_graph as T
from scar.ir.program.call_edges import (InvocationKey,CallRecordAnchor,CapturedSite,CapturedCode,CodeCapture,
    FrameEntry,FrameExit,FrameCompletion)
from scar.ir.program.call_operands import (OperandCallAttempt,OperandEventKind,CallableShape,CallableKind,FunctionLayout)
from scar.ir.program.python_call_targets import PythonCallTarget,PythonTargetKind,PythonCodeScope,PythonEntryBinding
from scar.ir.program.body_observations import SourceBodyActivation, SourceBodyActivationObservation
from scar.ir.program.ids import ProviderID,RegionID,ValueID
from scar.ir.program.providers import ProviderArtifact,ProviderFactReference
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.program.dependencies import GraphNamespace,DependencyPoint
from scar.ir.program.control import LanguageCompletionKind as C


def _evidence(layout_proven=True,completion=FrameCompletion.NORMAL):
    run='a'*64;key='pycode-v2:'+'b'*64;caller_key='pycode-v2:'+'c'*64
    anchor=lambda i:CallRecordAnchor('/trace/events.jsonl',i,'d'*64,i)
    template=G.SourceTemplateKey(1,P.ProductionPurpose.STATEMENT);body=P.ProductionRegionID('body')
    callstep=P.ProductionStepID('call');producer=P.ProductionStepID('producer')
    port=lambda owner,kind,i,label=None:G.Port(template,owner,kind,i,label)
    ports=tuple(port(body,G.PortKind.PARAMETER,i) for i in range(3))
    ports+=tuple(port(body,G.PortKind.RESULT,i,C.RETURN) for i in range(2))
    ports+=tuple(port(producer,G.PortKind.RESULT,i,C.FALLTHROUGH) for i in range(4))
    ports+=tuple(port(callstep,G.PortKind.RESULT,i,C.FALLTHROUGH) for i in range(2))
    gp=lambda i:T.GlobalPort('unit',ports[i])
    scope=F.CodeScopeWitness(F.CodeScopeRef('unit',2),template,body,2)
    owner=F.CodeScopeWitness(F.CodeScopeRef('unit',1),template,P.ProductionRegionID('caller'),1)
    function=NS(value=5,formals=(0,1,2),returns=3,return_world=4,template=template,body=body,
        body_scope=scope,parameter_names=('x',),environment=NS(kind=P.DeferredBodyKind.FUNCTION))
    call=F.Call(6,(7,),(),9,8,10)
    facts=NS(unit='unit',ports=ports,functions=(function,),calls=(call,),
        scope_groups=(F.FactScopeGroup(owner.owner,(F.FactRange(F.FactTable.CALLS,0,1),)),))
    artifact=ProviderArtifact('/source/result.json','e'*64,1,'test')
    locator=SourceArtifactLocator(artifact,artifact.sha256,1,'canonical-json')
    namespace=GraphNamespace(locator,'test',RegionID('shared-source'))
    source=NS(facts=facts,locator=locator,_expand=lambda keys:None)
    view=NS(unit=lambda unit:source,source_port=lambda gp:DependencyPoint(namespace,ValueID(str(gp.port))),
        source_region=lambda *args:DependencyPoint(namespace,RegionID('shared-body')),
        get=lambda point:NS(value_type=None),register=lambda provider:provider)
    captured=CapturedCode(key,'/source/function.py','function',1,0,('x',))
    code=NS(id=ProviderID('code'),body=T.Body('unit',template,body),scope=scope,function=gp(5),
        function_fact=ProviderFactReference(artifact,'#/functions/0'),captures=(CodeCapture(captured,anchor(1),1),))
    connection=T.FunctionConnection(gp(5),code.body,T.FunctionActivation.NORMAL,
        (T.Route(gp(0),T.RouteKind.FUNCTION_ENVIRONMENT,(gp(6),)),
         T.Route(gp(1),T.RouteKind.ARGUMENT,(gp(7),)),T.Route(gp(2),T.RouteKind.STATE,(gp(8),))),
        ((gp(3),gp(9)),(gp(4),gp(10))),('ORIGINAL_ROUTE_CONDITION',))
    caller=InvocationKey(1,2,1);site=CapturedSite(caller_key,10,'CALL',(1,1,0,4))
    layout=FunctionLayout(1,0,0,0,('x',));observations=[];frames=[];exits=[];bindings=[];attempts=[];targets=[];entries=[]
    for ordinal in (1,2):
        i=10*ordinal;invocation=InvocationKey(1,2,ordinal+1)
        attempt=OperandCallAttempt(anchor(ordinal),ordinal,OperandEventKind.CALL_ATTEMPT,1,2,1,i,1,site,
            CallableShape(CallableKind.EXACT_FUNCTION,'weak:1:1',key,layout),None,True)
        target=PythonCallTarget(ordinal,caller,PythonTargetKind.FUNCTION,'weak:1:1','weak:1:1',key,layout,None,PythonCodeScope.PROJECT)
        frame=FrameEntry(anchor(i+1),invocation,key,(),False,site,caller,None,(site,),True,(),False,False)
        exit=FrameExit(anchor(i+2),invocation,key,CapturedSite(key,20,'RETURN_VALUE',(2,2,0,4)),completion,'weak:1:2',(),False)
        observation=NS(id=ProviderID('callsite-'+str(ordinal)),unit='unit',query=0,callee=gp(6),result=gp(9),
            owner=owner,attempt=attempt,target=target,target_anchor=anchor(i+3),call=ProviderFactReference(artifact,'#/calls/0'))
        observations.append(observation);frames.append(frame);exits.append(exit);attempts.append(attempt);targets.append(target)
        entries.append(PythonEntryBinding(ordinal,caller,invocation,i+1,key,None,None))
        bindings.append(SourceBodyActivationObservation(ProviderID('activation'),frame,exit))
    activation=SourceBodyActivation(ProviderID('activation'),code.id,ProviderID('interface'),captured,(site,),False,
        (ProviderID('caller-code'),),0,())
    group=NS(code=code,connection=connection,query=0,observations=tuple(observations),
        body_entries=tuple(frame.anchor for frame in frames),argument_layout_proven=layout_proven)
    trace=NS(entries=tuple(frames),exits=tuple(exits),attempts=tuple(attempts))
    raw=NS(source_sha256=run,targets=tuple(targets),target_anchors=tuple(row.target_anchor for row in observations),entries=tuple(entries))
    args=(view,(group,),tuple(bindings),{activation.id:activation},
        {activation.interface:NS(body=code.body,scope=scope)},trace,raw)
    return args,dict(group_facts=(ProviderFactReference(artifact,'#/groups/0'),))


def test_repeated_same_callsite_keeps_invocation_and_caller_boundary_occurrences_distinct(tmp_path,monkeypatch):
    args,kwargs=_evidence();bindings=derive_invocation_bindings(*args,**kwargs)
    assert len(bindings)==2 and bindings[0].body==bindings[1].body
    assert bindings[0].activation==bindings[1].activation
    assert bindings[0].call_scope!=bindings[1].call_scope
    assert {route.field for route in bindings[0].routes}=={'argument','environment','world_input','data_return','world_return'}
    left=next(route for route in bindings[0].routes if route.field=='data_return')
    right=next(route for route in bindings[1].routes if route.field=='data_return')
    assert left.target.original==right.target.original and left.target!=right.target
    provider=SourceInvocationProvider(args[0],bindings)
    point=provider.port(left.source);edges=provider.outgoing(point).edges
    assert len(edges)==1 and edges[0].target==provider.port(left.target)
    assert provider.original(point).original==left.source.original
    assert not provider.explain(point)['content_preservation_proven']
    # The real F4 session hook installs discovery without opening every group
    # shard. Its first source query binds that unit; another query reuses it.
    from scar.analysis.build_f4 import F4BuildSession
    from scar.analysis.source_body_activations import SourceBodyActivationProvider
    from scar.ir.program.dependencies import DependencyKind,DependencyMeaning
    from scar.analysis import source_invocation_view as invocation_module
    view=args[0];source=view.unit('unit');source.unit='unit'
    view.catalog=NS(path=tmp_path/'catalog',report={'schema':'test'})
    source.records={view.source_port(port).node:port for binding in bindings
        for route in binding.routes for port in (route.source.original,route.target.original)}
    view.namespaces={point.namespace:source};registered=[];view.register=lambda p:registered.append(p)
    activation_path=tmp_path/'activations';activation_path.mkdir();(activation_path/'result.json').write_text('{}')
    observed_path=tmp_path/'observed';observed_path.mkdir();(observed_path/'result.json').write_text('{}')
    body=NS(view=view,store=NS(path=activation_path,check=lambda:None,
        report={'schema':'scar.source-body-activations','basis':{'run_scope':args[-1].source_sha256}},
        index=NS(codes=NS(trace=args[-2],targets=args[-1]))),
        observations={'activation':args[2]},activations=args[3],interfaces=args[4])
    body.bind_invocations=lambda *a,**kw:SourceBodyActivationProvider.bind_invocations(body,*a,**kw)
    reads=[];decodes=[]
    store=NS(path=observed_path,catalog=view.catalog,validate=lambda:None,read_unit=lambda unit:reads.append(unit),
        report={'schema':'scar.source-observed-function-interfaces','run_scope':args[-1].source_sha256,
            'shards':{'unit':{'path':'units/unit','tables':{'groups':{'path':'groups.jsonl.gz'}}}}})
    def read_groups(actual_store,unit):
        assert actual_store is store
        actual_store.read_unit(unit);decodes.append(unit)
        return args[1],kwargs['group_facts']
    monkeypatch.setattr(invocation_module,'_original_invocation_groups',read_groups)
    session=object.__new__(F4BuildSession);session.view=view;session.gaps=[];session._checkpoint=lambda:None
    session.providers={'body_activations':body,'functions:observed':NS(store=store,available_units={'unit','untouched'})}
    session._invocations();lazy=session.providers['invocations']
    assert registered[-1] is lazy and not reads and not decodes
    assert lazy.publication()['materialized_units']==[]
    original=view.source_port(left.target.original)
    discovered=lazy.outgoing(original).edges
    assert len(discovered)==2
    assert all(edge.kind is DependencyKind.IMPLEMENTATION and edge.meaning is DependencyMeaning.STRUCTURE
        and edge.conditions[0].name=='INSTANCE_OF_SHARED_PORT' for edge in discovered)
    import json
    lazy.outgoing(original);lazy.restore(json.loads(json.dumps(lazy.publication())))
    assert reads==decodes==['unit'] and len(lazy.providers['unit'].bindings)==2


def test_unknown_layout_and_exit_leave_field_frontiers_without_normal_data_or_world_return():
    args,kwargs=_evidence(False,FrameCompletion.UNKNOWN)
    bindings=derive_invocation_bindings(*args,**kwargs)
    assert all({route.field for route in binding.routes}=={'environment','world_input'} for binding in bindings)
    assert all({'parameter','data_return','world_return'}<=set(frontier.field for frontier in binding.frontiers)
        for binding in bindings)
    provider=SourceInvocationProvider(args[0],bindings)
    assert all({'parameter','data_return','world_return'}<=
        {frontier.detail for frontier in provider.outgoing(provider.binding_points[binding.id]).frontiers}
        for binding in bindings)
