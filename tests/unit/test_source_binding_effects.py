from dataclasses import replace
from types import SimpleNamespace

import pytest

from scar.analysis import source_binding_effects as E, source_name_definitions as solver
from scar.analysis import source_frame_binding_effects as I
from scar.ir.program import normal_graph as G, normal_graph as F, normal_graph as T
from scar.ir.program import source_bindings as B, name_flow as A
from scar.ir.program import name_flow as N, source_productions as P, source_value_routes as V
from scar.ir.program import name_flow as D
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.model import ValueType, ValueKind
from scar.ir.program.ids import LocationID
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.program.source_declarations import SourceTemplateKey
from scar.ir.program.protocol_signatures import ExceptionABI


def test_conditional_local_read_keeps_q_and_unknown_call_release_and_capture(monkeypatch):
    artifact = ProviderArtifact('/original/f4', '1' * 64, 1, 'original-f4')
    fact = lambda table, i: ProviderFactReference(artifact, '#/' + table + '/' + str(i))
    key = SourceTemplateKey(0, P.ProductionPurpose.STATEMENT)
    region = P.ProductionRegionID('fixture:body')
    scope = F.CodeScopeRef('unit', 0)
    owner = F.CodeScopeWitness(scope, key, region, 0)
    name = B.NameRef('SOURCE', 0)
    namespace = F.Namespace(F.NamespaceKind.LEXICAL, 0, 'value')
    port = lambda step, kind, ordinal=0: G.Port(key, step, kind, ordinal,
        C.FALLTHROUGH if kind in (G.PortKind.EXIT, G.PortKind.RESULT) else None)
    gp = lambda p: T.GlobalPort('unit', p)
    root = port(region, G.PortKind.ENTER)
    formal = gp(port(region, G.PortKind.PARAMETER, 1))
    function_ref = gp(port(P.ProductionStepID('fixture:function'), G.PortKind.RESULT))
    function = SimpleNamespace(value=0, template=key, body=region, body_scope=owner,
        environment=SimpleNamespace(kind=P.DeferredBodyKind.FUNCTION, scope=0))
    world = ValueType(ValueKind.STATE, location=LocationID('world'))
    reference = ValueType(ValueKind.REFERENCE)
    bindings = B.SourceBindingsMaterial(
        B.SourceBindingsInput(*(['0' * 64] * 6), '/original.py', 'exec', 'DECLARED', 1, 1),
        B.SourceBindingsProfile('rule', 'projection', *(['0' * 64] * 5)),
        1, 1, 1, 1, 1, 0, 0, 0, 'RESOLVED',
        (B.BindingSlot('LOCAL_BINDING', 0, name, False, ()),),
        (B.EnvironmentInterface('LEXICAL_BINDING', 0, 0),),
        (B.LookupPlan((B.LookupStep(0),), 'RAISE_UNBOUND_LOCAL'),),
        (B.PortResolution(0, 'READ', 0, 0, name, None, 0, 0, None, None, 'FUNCTION', 'VISIBLE', ()),),
        (B.ScopeNameResolution(0, name, 'LOCAL', 0, 'original', ()),),
        (B.ScopeBindings(0, (0,), ()),), (), (), (), (), (), (), 0)
    active_bindings = [bindings]
    body_bindings = [bindings]
    monkeypatch.setattr(E, '_bindings', lambda material, remembered: (body_bindings[0], None, (artifact,)))

    class Slicer:
        def __init__(self, snapshot): pass
        def fact(self, unit, table, i): return fact(table, i)

    monkeypatch.setattr(E, 'ReferenceSlicer', Slicer)
    monkeypatch.setattr(solver, 'ReferenceSlicer', Slicer)
    source_port = lambda unit, local: unit.port_map[local]
    monkeypatch.setattr(E, 'source_port', source_port)
    monkeypatch.setattr(I, 'source_port', source_port)
    monkeypatch.setattr(solver, 'source_port', source_port)

    def analyze(middle=None, cycle=False):
        body_bindings[0] = active_bindings[0]
        if type(middle) is P.AccessBinding and middle.action is P.BindingActionKind.WRITE:
            middle = replace(middle, port=1)
            body_bindings[0] = replace(body_bindings[0], input=replace(bindings.input, port_count=2),
                ports=(*bindings.ports, replace(bindings.ports[0], port=1, read_plan=None, write_target=0)))
        instructions = [P.AccessBinding(P.BindingActionKind.READ, 0)]
        if middle is not None: instructions.append(middle)
        instructions.append(P.AccessBinding(P.BindingActionKind.READ, 0))
        operations = []; accesses = []; anchors = []
        names = [F.NameAccess(namespace, 1, True)]; port_map = {0: function_ref, 1: formal}
        edges = []; previous = root
        for i, ins in enumerate(instructions):
            step = P.ProductionStepID('fixture:op' + str(i)); enter = port(step, G.PortKind.ENTER)
            exit = port(step, G.PortKind.EXIT)
            is_store = type(ins) is P.AccessBinding and ins.action is P.BindingActionKind.WRITE
            inputs = ((V.TypedValuePort(port(step, G.PortKind.ARGUMENT), reference),
                V.TypedValuePort(port(step, G.PortKind.ARGUMENT, 1), world)) if is_store else
                (V.TypedValuePort(port(step, G.PortKind.ARGUMENT), world),))
            operations.append(V.SourceOperation(enter, ins, inputs,
                (V.TypedValuePort(port(step, G.PortKind.RESULT), reference),
                 V.TypedValuePort(port(step, G.PortKind.RESULT, 1), world))))
            edges.extend((G.Edge(previous, enter, G.EdgeKind.CONTROL), G.Edge(enter, exit, G.EdgeKind.CONTROL)))
            previous = exit
            if type(ins) is P.AccessBinding:
                is_write = ins.action is P.BindingActionKind.WRITE
                local = len(names) + 1
                value = gp(port(step, G.PortKind.ARGUMENT)) if is_write else gp(port(step, G.PortKind.RESULT))
                port_map[local] = value; record = F.NameAccess(namespace, local, is_write)
                ordinal = len(names); names.append(record); access_index = len(accesses)
                accesses.append(N.NamespaceAccess(gp(enter), owner, ins.port, ins.action, name, 'value',
                    None if is_write else 0,
                    () if is_write else (N.NamespaceLookupStep(0, N.NamespaceInterfaceRef('unit', 0), 'RETURN_VALUE', 'NEXT', 'PROPAGATE'),),
                    None if is_write else 'RAISE_UNBOUND_LOCAL', N.NamespaceInterfaceRef('unit', 0) if is_write else None,
                    value, gp(port(step, G.PortKind.ARGUMENT)), gp(port(step, G.PortKind.RESULT, 1))))
                anchors.append(A.NameAccessAnchor(fact('names', ordinal), record, scope, access_index,
                    namespace, None if is_write else 0, N.NamespaceInterfaceRef('unit', 0) if is_write else None))
        if cycle: edges.append(G.Edge(previous, operations[0].entry, G.EdgeKind.CONTROL, transition=G.Transition.BACKEDGE))
        unit = SimpleNamespace(unit='unit', functions=(function,), names=tuple(names), port_map=port_map)
        store = SimpleNamespace(load=lambda identity: unit)
        material = SimpleNamespace(unit='unit', fact_store=store, result_artifact=artifact, validate_parents=lambda: None)
        snapshot = SimpleNamespace(fact_store=store)
        admission = A.FormalNameAdmission(fact('names', 0), names[0], fact('functions', 0), function_ref,
            owner, 0, 0, formal, gp(root))
        graph = A.NameAccessAnchorGraph('unit', (scope,), (artifact,), tuple(accesses), tuple(anchors), (admission,), (), ())
        flow = V.SourceBodyValueFlow(function_ref, T.Body('unit', key, region), artifact,
            tuple(edges), tuple(operations), (), ())
        kwargs = dict(material=material, anchor_graph=graph, flow=flow)
        strict = solver.reaching_name_definitions(snapshot, function_ref, namespace, **kwargs)
        conditional = E.conditional_reaching_name_definitions(snapshot, function_ref, namespace, **kwargs)
        isolated = I.frame_isolated_reaching_name_definitions(snapshot, function_ref, namespace, **kwargs)
        return strict, conditional, isolated

    for cycle in (False, True):
        strict, conditional, isolated = analyze(cycle=cycle)
        assert any(d.kind is D.NameDefinitionKind.OPAQUE_EFFECT for d in strict.definitions)
        assert len(conditional.witnesses) == 2
        assert all(d.kind is D.NameDefinitionKind.FORMAL for d in conditional.graph.definitions)
        assert conditional.graph.traversal_complete and conditional.graph.dataflow_complete
        assert conditional.graph.semantics != strict.semantics
        assert not conditional.obligations_discharged
        assert set(E.REQUIRED_Q) <= set(conditional.graph.conditions)
        assert isolated.witness is not None
        assert all(d.kind is D.NameDefinitionKind.FORMAL for d in isolated.graph.definitions)
        assert not isolated.obligations_discharged
        assert isolated.heap_rng_external_and_array_content_effects == 'UNCHANGED_UNKNOWN'
        assert set(I.FRAME_REQUIRED_Q) <= set(isolated.graph.conditions)
    for unknown in (P.ApplyLanguageAction(P.LanguageActionKind.CALL), P.AccessBinding(P.BindingActionKind.WRITE, 0)):
        _, conditional, isolated = analyze(unknown)
        assert any(d.kind is D.NameDefinitionKind.OPAQUE_EFFECT for d in conditional.graph.definitions)
        last = conditional.graph.reads[-1]
        assert any(conditional.graph.definitions[d].kind is D.NameDefinitionKind.OPAQUE_EFFECT for d in last.definitions)
        assert isolated.witness is not None
        assert not any(d.kind is D.NameDefinitionKind.OPAQUE_EFFECT for d in isolated.graph.definitions)
        assert isolated.graph.operations == conditional.graph.operations
        if type(unknown) is P.AccessBinding:assert len(isolated.witness.writers) == 1
    for implicit in (P.DeleteExceptionBinding(0, ExceptionABI.EXCEPTION_WORLD),
            P.ApplyNamespaceAction(P.NamespaceActionKind.STORE_NAME, ExceptionABI.EXCEPTION_WORLD)):
        strict, _, isolated = analyze(implicit)
        assert isolated.witness is None and isolated.graph.definitions == strict.definitions
        assert any(r.startswith('UNMODELED_DIRECT_NAMESPACE_WRITER:') for r in isolated.refusal_reasons)
    active_bindings[0] = replace(bindings, captures=(B.CaptureDemand(1, 0, 'READ', (), 'FUNCTION', None),))
    strict, conditional, isolated = analyze()
    assert conditional.witnesses == () and conditional.graph.definitions == strict.definitions
    assert isolated.witness is None and isolated.graph.definitions == strict.definitions
    with pytest.raises(TypeError):
        E.ConditionalBindingProjection(conditional.graph, (), conditional.analysis_sources, obligations_discharged=True)
