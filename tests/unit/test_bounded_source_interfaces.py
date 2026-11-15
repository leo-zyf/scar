"""Targeted G3 checks; finite fixtures are not whole-program completeness.

The first check exercises the shared binder/default-state interface without
compiling or running a target. The retained-capture check below uses an already
admitted frontend and a bounded run, never rebuilds that frontend.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from scar.analysis.source_call_defaults import FunctionDefaultStateProvider
from scar.analysis.source_function_connections import Invocation,connect
from scar.ir.program import normal_graph as G,normal_targets as T,source_productions as P,model as M
from scar.ir.program.dependencies import (GraphNamespace,DependencyPoint,DependencyFrame,context_push,context_pop,
    context_frames,context_is_closed,OpenDependencyContext,DependencyKind as K,
    DependencyMeaning as Meaning,DependencyStrength as Strength)
from scar.ir.program.ids import LocationID,OperationID,RegionID,UseID,ValueID
from scar.ir.program.providers import ProviderArtifact
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.program.source_template_links import SourceTemplateKey


def _signature_fixture():
    key=SourceTemplateKey(0,P.ProductionPurpose.VALUE);region=P.ProductionRegionID('minimal-signature')
    ports=tuple(G.Port(key,region,G.PortKind.PARAMETER,index) for index in range(20))
    gp=lambda index:T.GlobalPort('unit',ports[index])
    source=P.SlotRef(0,'FIELD',0)
    parameters=(P.DeferredParameter(P.DeferredParameterKind.POSITIONAL_OR_KEYWORD,source,0),
        P.DeferredParameter(P.DeferredParameterKind.POSITIONAL_OR_KEYWORD,source,1,source))
    function=NS(value=10,template=key,body=region,parameter_names=('left','right'),defaults=(None,12),
        formals=(0,1,2,3),returns=13,return_world=14,
        environment=NS(parameters=parameters,kind=P.DeferredBodyKind.FUNCTION,captures=()))
    unit=NS(unit='unit',ports=ports,functions=(function,))
    mini=NS(functions={10:(unit,function)},offsets={'unit':0},gp=gp,edge=lambda *_:None,add=lambda *_:None)
    invocation=Invocation(10,(5,),(),6,7,8)
    locator=SourceArtifactLocator(ProviderArtifact('/fixture/original.json','0'*64,1,'fixture'),
        '0'*64,1,'canonical-json')
    namespace=GraphNamespace(locator,'fixture-original-f4',RegionID('unit'))
    return unit,function,mini,invocation,gp,namespace,locator


def test_shared_binder_preserves_current_default_world_and_deferred_boundary():
    unit,function,mini,invocation,gp,namespace,locator=_signature_fixture()
    connection=connect(mini,10,invocation,conditions=('OBSERVED_CODE_IS_ONLY_CONDITIONAL_SELECTION',))
    assert connection.activation is T.FunctionActivation.NORMAL
    assert connection.returns==((gp(13),gp(6)),(gp(14),gp(8)))
    assert next(route for route in connection.parameters if route.kind is T.RouteKind.STATE).actuals==(gp(7),)
    route=next(route for route in connection.parameters if route.kind is T.RouteKind.INITIAL_DEFAULT)
    link=connection.to_link(gp(6),gp(10),'SOURCE_FUNCTION')
    source=NS(facts=unit,namespace=namespace,basis='minimal-signature',locator=locator,check=lambda *_:None)
    def source_point(point):return DependencyPoint(namespace,ValueID('port-'+str(point.port.ordinal)),None)
    view=NS(register=lambda _:None,unit=lambda _:source,namespaces={namespace:source},
        source_port=source_point,get=lambda _:NS(value_type=M.ValueType(M.ValueKind.STATE,location=LocationID('world'))))
    frame=NS(caller=namespace,operation=OperationID('call'))
    defaults=FunctionDefaultStateProvider(view)
    current=defaults.read(NS(link=link),route,frame,gp(7))
    assert defaults.reads[0].initial_source==gp(12)
    assert current.node!=source_point(gp(12)).node
    assert defaults.states[namespace,current.node].world==gp(7)
    assert defaults.explain(current)['current_equals_initial_proven'] is False
    assert all(edge.source!=source_point(gp(12))for edge in defaults.incoming(current).edges)
    assert any(f.kind=='FUNCTION_DEFAULT_CREATION_INSTANCE_AND_MUTATION_HISTORY_NOT_SELECTED'
        for f in defaults.incoming(current).frontiers)
    for kind in (P.DeferredBodyKind.GENERATOR,P.DeferredBodyKind.COROUTINE,P.DeferredBodyKind.ASYNC_GENERATOR):
        function.environment.kind=kind
        deferred=connect(mini,10,invocation)
        assert deferred.activation is T.FunctionActivation.CREATE_DEFERRED_BODY
        assert not deferred.returns


def test_shared_callee_requires_exact_caller_frame_and_preserves_open_prefix():
    _,_,_,_,_,namespace,_=_signature_fixture()
    first=DependencyFrame(namespace,OperationID('caller-a'),UseID('callee-use-a'),
        ValueID('callee-value-a'),namespace,entry=RegionID('same-callee'))
    second=replace(first,operation=OperationID('caller-b'),callee_use=UseID('callee-use-b'),
        callee_value=ValueID('callee-value-b'))
    selected=context_push((),first)
    assert context_pop(selected,(first,)).remaining==()
    assert context_pop(selected,(first,)).matched
    assert not context_pop(selected,(second,)).matched
    opened=context_push(None,first)
    assert type(opened)is OpenDependencyContext and not context_is_closed(opened)
    assert context_frames(opened)==(first,)
    popped=context_pop(opened,(first,))
    assert popped.matched and popped.remaining is None
    assert not context_is_closed(popped.remaining)


def test_retained_bounded_interfaces_are_balanced_conditional_source_routes(monkeypatch):
    from scar.analysis.program_view import ProgramView
    from scar.analysis.source_call_binding_view import SourceCallBindingProvider
    from scar.analysis.bounded_function_interfaces import BoundedFunctionInterfaceStore
    from scar.analysis.source_call_view import Flow
    root=Path(__file__).resolve().parents[2]/'artifacts/current/experiments/cross-unit-composite-v1'
    raw=tuple(sorted((root/'call-bindings').glob('call-bindings.*.json')))
    if not raw or not (root/'frontend/catalog/result.json').is_file():
        pytest.skip('retained bounded G3 capture is an explicit prerequisite; no frontend rebuilt by this test')
    view=ProgramView(root/'frontend/catalog')
    owner=SourceCallBindingProvider(view,root/'call-bindings/plan.json',raw)
    provider=owner.attach_function_interfaces()
    for unit in provider.store.report['shards']:provider.attach(unit)
    assert provider.frames and provider.relations
    assert any(len(provider.store.interface_attempts(*key))>1 for key in provider.frames)
    records=tuple(value[2]for value in owner.nodes.values())
    assert records and all('callee_entry'in row.remaining_fields and 'callee_completion'in row.remaining_fields
        and 'array_content_versions'in row.remaining_fields for row in records)
    assert all(owner.explain(value[0])['entry_completion_or_effects_proven']is False
        for value in owner.nodes.values())
    assert provider.summary()['runtime_dispatch_proven']is False
    assert provider.summary()['loops_unrolled']is False
    raw_names={str(path)for path in raw}
    for key,frame in provider.frames.items():
        assert frame.selection==provider.store.interface_fact(*key)
        assert frame.selection.artifact.uri in raw_names
        assert frame.selection.locator.startswith('#/records/')
    observed_kinds=set()
    for relation in provider.relations:
        if relation.kind not in (K.ARGUMENT,K.STATE_INTERFACE,K.COMPLETION):continue
        observed_kinds.add(relation.kind)
        if relation.kind is K.STATE_INTERFACE:assert relation.meaning is Meaning.STATE_INTERFACE
        else:assert relation.meaning is Meaning.REFERENCE
        # An unknown outer caller remains open; a selected static frame is
        # neither a root activation nor a dynamic invocation identifier.
        start=replace(relation.source,context=None if relation.flow is Flow.ENTER else
            context_push(None,relation.frame))
        expected_target=replace(relation.target,context=context_push(None,relation.frame)
            if relation.flow is Flow.ENTER else None)
        matches=[edge for edge in provider.outgoing(start).edges if edge.target==expected_target
            and edge.kind is relation.kind]
        assert matches
        edge=matches[0]
        assert edge.strength is Strength.MAY
        names={condition.name for condition in edge.conditions}
        assert 'CALL_ATTEMPT_DOES_NOT_PROVE_CALLER_INVOCATION_OR_CALLEE_ENTRY'in names
        assert 'FUNCTION_CODE_CAPTURE_DEFAULT_AND_NAMESPACE_VERSIONS_MATCH_DECLARATION'in names
        assert edge in provider.incoming(expected_target).edges
        actual_evidence=edge.evidence[0].sources[-1]
        assert actual_evidence.artifact==relation.frame.selection.artifact
        assert actual_evidence.pointer==relation.frame.selection.locator
        # No selected frame means no borrowed actual/return route. A different
        # original evidence alternative cannot pop this frame either.
        body_endpoint=edge.target if relation.flow is Flow.ENTER else edge.source
        bare=replace(body_endpoint,context=None)
        query=provider.incoming if relation.flow is Flow.ENTER else provider.outgoing
        assert any(f.kind=='SOURCE_CALL_CONTEXT_FRAME_NOT_SELECTED'for f in query(bare).frontiers)
        other=next(frame for frame in provider.frames.values()if frame.selection!=relation.frame.selection)
        wrong=replace(relation.frame,selection=other.selection)
        wrong_point=replace(body_endpoint,context=context_push(None,wrong))
        assert not any(candidate.kind is relation.kind
            and replace(candidate.source,context=None)==replace(relation.source,context=None)
            and replace(candidate.target,context=None)==replace(relation.target,context=None)
            for candidate in query(wrong_point).edges)
    assert observed_kinds=={K.ARGUMENT,K.STATE_INTERFACE,K.COMPLETION}
    # This local protocol fixture does not claim a bound method was observed.
    # It asks whether the adapter conservatively rejects that metadata kind.
    store=BoundedFunctionInterfaceStore(owner)
    original_groups=store._attempt_groups
    site=next(site for site in owner.plan['sites']if any(kind=='PYTHON_FUNCTION'and body is not None
        for (kind,_,body,_),_ in original_groups(site)))
    def bound_method_groups(selected):
        groups=original_groups(selected)
        if selected['id']!=site['id']:return groups
        return tuple((('BOUND_PYTHON_METHOD',code,body,metadata),attempts)
            for (_,code,body,metadata),attempts in groups)
    monkeypatch.setattr(store,'_attempt_groups',bound_method_groups)
    graph=store.read_unit(site['unit'])
    assert not any(interface.query==site['query']for interface in graph.interfaces)
    assert any(frontier.query==site['query']and frontier.reason==
        'BOUND_METHOD_RECEIVER_AND_DESCRIPTOR_ABI_NOT_CAPTURED'for frontier in graph.frontiers)
