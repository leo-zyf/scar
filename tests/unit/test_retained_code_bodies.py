"""One retained-catalog check; never rebuild a frontend or execute a target."""
from dataclasses import replace
from pathlib import Path
from types import CodeType

import pytest

from scar.analysis.program_view import ProgramView
from scar.analysis.retained_code_bodies import RetainedCodeBodyIndex
from scar.ir.program.call_edges import CapturedCode
from scar.ir.record_codec import decode
from scar.trace.code_snapshot import describe_code


def test_compiler_scope_body_join_does_not_cache_on_fingerprint_alone():
    repo = Path(__file__).resolve().parents[2]
    catalog = repo/'artifacts/current/experiments/cross-unit-composite-v1/frontend/catalog'
    source = repo/'tests/workloads/cross_unit_composite/scalar_ops.py'
    if not (catalog/'result.json').is_file() or not source.is_file():
        pytest.skip('retained four-file catalog is an explicit prerequisite')
    # Compilation yields reference descriptors only; neither source imports
    # nor the selected function body execute during this check.
    root = compile(source.read_bytes(), str(source), 'exec', dont_inherit=True, optimize=0)
    child = next(code for code in root.co_consts
        if type(code) is CodeType and code.co_name == 'prepare')
    code = decode(CapturedCode, describe_code(child))
    index = RetainedCodeBodyIndex(ProgramView(catalog))
    conflict = replace(code, qualname=code.qualname+'.conflicting-descriptor')
    assert index.match(conflict)[0] is None
    body, frontier = index.match(code)
    assert body is not None and frontier is None
    assert index.match(conflict)[0] is None
    assert index.match(code) == (body, None)
    proof = index.explain(code)
    assert len(proof['candidate_units']) == len(index.sources) == 1
    assert len(proof['matches']) == 1
    match = proof['matches'][0]
    assert match['function_fact']['locator'].endswith('/functions/0')
    assert match['compiler_path']['source_scope'] == match['body_scope']['source_scope']
    assert match['constant_path'] == match['compiler_path']['constant_path']
    assert match['compiled_root'] == describe_code(root)
    assert proof['callable_environment_or_invocation_proven'] is False


def test_call_query_demand_keeps_original_ordinals_and_does_not_attach_unrelated_calls():
    from scar.analysis.source_call_binding_view import SourceCallBindingProvider
    from scar.ir.program.normal_graph import FunctionActivation
    repo=Path(__file__).resolve().parents[2]
    root=repo/'artifacts/current/experiments/cross-unit-composite-v1'
    raw=tuple(sorted((root/'call-bindings').glob('call-bindings.*.json')))
    if not raw or not(root/'frontend/catalog/result.json').is_file():
        pytest.skip('retained four-file capture is an explicit prerequisite')
    owner=SourceCallBindingProvider(ProgramView(root/'frontend/catalog'),root/'call-bindings/plan.json',raw)
    provider=owner.attach_function_interfaces()
    graphs=[provider.store.read_unit(unit)for unit in provider.available_units]
    graph,query=next((graph,row.query)for graph in graphs
        if len({row.query for row in graph.interfaces if row.activation is FunctionActivation.NORMAL})>1
        for row in graph.interfaces if row.activation is FunctionActivation.NORMAL
        and owner.view.unit(graph.unit).facts.root_scope.owner not in graph.queries[row.query].owners)
    site=next(row for row in owner.plan['sites']if(row['unit'],row['query'])==(graph.unit,query))
    operation=owner.operations[site['id']]
    # A known module scope cannot activate a call owned by a function. Do not
    # normalize its unrelated candidate bodies before checking that scope.
    wrong=provider.outgoing(replace(operation,context=()))
    assert not provider.links and not provider.relations
    assert any(item.kind=='SOURCE_CALLER_CODE_SCOPE_DOES_NOT_OWN_CALLSITE'for item in wrong.frontiers)
    selected=provider.outgoing(replace(operation,context=None))
    assert selected.edges and set(provider.attached_queries[graph.unit])=={query}
    assert {interface.query for interface in provider.links.values()}=={query}
    count=len(provider.relations)
    provider.outgoing(replace(operation,context=None))
    assert len(provider.relations)==count
    provider.attach(graph.unit)
    assert provider.attached_queries[graph.unit]==set(range(graph.calls))
    assert {interface.query for interface in provider.links.values()}=={row.query for row in graph.interfaces}
