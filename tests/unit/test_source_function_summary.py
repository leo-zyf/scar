"""One metadata-only local summary checks cut calls, selectors and channels."""
from dataclasses import replace
from hashlib import sha256
import json
from types import SimpleNamespace as NS

from scar.analysis import source_function_summary as S
from scar.ir.program import normal_graph as G, normal_links as F, normal_targets as T
from scar.ir.program import source_productions as P, source_value_flow as V
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.ids import LocationID
from scar.ir.program.model import ValueKind, ValueType
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.program.source_template_links import SourceTemplateKey


def test_local_summary_keeps_call_return_cut_and_control_selector(tmp_path, monkeypatch):
    key = SourceTemplateKey(0, P.ProductionPurpose.STATEMENT)
    region = P.ProductionRegionID('function')
    port = lambda owner, kind, index=0, completion=None: G.Port(key, owner, kind, index, completion)
    formal = [port(region, G.PortKind.PARAMETER, i) for i in range(4)]
    ret = port(region, G.PortKind.RESULT, 0, C.FALLTHROUGH)
    ret_world = port(region, G.PortKind.RESULT, 1, C.FALLTHROUGH)
    call = P.ProductionStepID('call')
    call_result = port(call, G.PortKind.RESULT, 0, C.FALLTHROUGH)
    call_world = port(call, G.PortKind.RESULT, 1, C.FALLTHROUGH)
    local = port(P.ProductionStepID('local'), G.PortKind.RESULT, 0, C.FALLTHROUGH)
    root_enter = port(region, G.PortKind.ENTER)
    branch_enter = port(P.ProductionRegionID('branch'), G.PortKind.ENTER)
    source_path = tmp_path/'source.py'; source_path.write_text('metadata fixture, never compiled\n')
    facts_path = tmp_path/'facts.json'; facts_path.write_text('{}')
    catalog_path = tmp_path/'result.json'; catalog_path.write_text('{}')
    artifact = ProviderArtifact(str(facts_path), sha256(facts_path.read_bytes()).hexdigest(), 2, 'scar.source-facts-unit')
    ref = lambda locator: ProviderFactReference(artifact, locator)
    scope = F.CodeScopeWitness(F.CodeScopeRef('unit', 1), key, region, 1)
    function = F.Function(0, key, region, NS(kind=P.DeferredBodyKind.FUNCTION), ('a', 'flag'),
        (0, 1, 2, 3), (None, None), 4, 5, scope)
    facts = NS(unit='unit', ports=(*formal, ret, ret_world, call_result, call_world, local),
        functions=(function,), calls=(F.Call(0, (8,), (), 6, 3, 7),), scope_groups=())
    reference = ValueType(ValueKind.REFERENCE)
    state = ValueType(ValueKind.STATE, location=LocationID('world'))
    typed = V.TypedValuePort
    activation = V.SourceOperation(port(P.ProductionStepID('activation'), G.PortKind.ENTER),
        P.EnterDeferredBody(NS()), tuple(typed(p, state if i==3 else reference)
            for i, p in enumerate(formal)), (typed(port(P.ProductionStepID('activation'),
                G.PortKind.RESULT, 0, C.FALLTHROUGH), state),))
    call_op = V.SourceOperation(port(call, G.PortKind.ENTER),
        P.ApplyLanguageAction(P.LanguageActionKind.CALL),
        (typed(formal[0], reference), typed(local, reference), typed(formal[3], state)),
        (typed(call_result, reference), typed(call_world, state)))
    flow = V.SourceBodyValueFlow(T.GlobalPort('unit', formal[0]), T.Body('unit', key, region), artifact,
        (G.Edge(call_result, ret, G.EdgeKind.DATA, reference),
         G.Edge(call_world, ret_world, G.EdgeKind.STATE, state),
         G.Edge(root_enter, branch_enter, G.EdgeKind.CONTROL, condition=formal[2], when=True)),
        (activation, call_op), (V.LexicalLocation(scope.owner,
            F.Namespace(F.NamespaceKind.LEXICAL, 1, 'slot'),
            (V.LocationAccess(formal[1], ref('#/names/0')),),
            (V.LocationAccess(local, ref('#/names/1')),)),),
        (V.SourceBodyCall(ref('#/calls/0'), facts.calls[0]),))
    store = NS(rows={'unit':dict(facts=str(facts_path), sha256=artifact.sha256, bytes=2)},
        model_units={'unit':dict(path=str(source_path), raw_sha256=sha256(source_path.read_bytes()).hexdigest())})
    source = NS(facts=facts, material=NS(validate_parents=lambda:None), check=lambda:None)
    view = NS(unit=lambda unit:source, catalog=NS(fact_store=store,path=tmp_path,report={'schema':'fixture'}),
        source_region=lambda *args:None)
    monkeypatch.setattr(S, 'project_value_flow', lambda *args:flow)
    report = S.derive_source_function_summary(view, 'unit', function)
    by_kind = {row['kind']:row['id'] for row in report['tokens'] if row['kind'].startswith('CALL_')}
    formal_tokens = [row['token'] for row in report['ports']['formals']]
    assert report['normal_return']['without_control']['data'] == [by_kind['CALL_RESULT']]
    assert set(report['normal_return']['data']) == {by_kind['CALL_RESULT'], formal_tokens[2]}
    assert formal_tokens[1] not in report['normal_return']['data']
    assert report['normal_return_world']['without_control']['state'] == [by_kind['CALL_WORLD']]
    assert not report['normal_return']['state']
    local_row = report['port_dependencies'][report['port_index'][
        json.dumps(S.encode(T.GlobalPort('unit', local)), sort_keys=True, separators=(',',':'))]]
    assert local_row['without_control']['data'] == [formal_tokens[1]]
    assert report['control_selector_dependencies']['data'] == [formal_tokens[2]]
    assert all(row['status']=='UNKNOWN' for row in report['effects'].values())
    assert S.verify_source_function_summary(report, view, 'unit', function) == report
    assert not report['purity_proven'] and not report['transformation_authorized']

    # Constant branch results still depend on the original selector, even
    # with no DATA edge from the selector into the returned literal.
    flow = replace(flow, edges=tuple(edge for edge in flow.edges if edge.kind is not G.EdgeKind.DATA))
    constant = S.derive_source_function_summary(view, 'unit', function)
    assert constant['normal_return']['data'] == [formal_tokens[2]]
    assert constant['normal_return']['without_control']['data'] == []
    partial = S.derive_source_function_summary(view, 'unit', function, max_work=1)
    assert not partial['propagation']['local_fixed_point_reached']
    assert 'LOCAL_DEPENDENCY_BUDGET_EXHAUSTED' in partial['normal_return']['unknown']
