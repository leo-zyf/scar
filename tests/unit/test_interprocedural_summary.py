"""Finite original-port summaries retain conditions without recursive bodies."""
from copy import deepcopy

import pytest

from scar.analysis import interprocedural_summary as summary
from scar.analysis.interprocedural_summary import (_compose, expand_interprocedural_dependencies,
    resolve_dependency_set, verify_interprocedural_summary)
from scar.ir.program import normal_graph as G, normal_graph as T, source_productions as P
from scar.ir.record_codec import encode
from scar.optimization.resource_solver import semantic_sha256


def test_recursive_union_selected_targets_and_open_dispatch_keep_field_dependencies(monkeypatch):
    channels = ('data', 'environment', 'state', 'unknown')

    def dep(**values):
        return {key: list(values.get(key, ())) for key in channels}

    def function(name, function_index):
        template = G.SourceTemplateKey(function_index, P.ProductionPurpose.STATEMENT)
        region = P.ProductionRegionID(name)

        def port(index):
            return encode(T.GlobalPort('unit', G.Port(template, region, G.PortKind.PARAMETER, index)))

        tokens, dependencies, formals = [], [], []
        for index, role in enumerate(('ENVIRONMENT', 'DATA', 'DATA', 'STATE')):
            token = name+':'+str(index)
            tokens.append(dict(id=token, port=port(index), kind='FORMAL_'+role))
            channel = 'environment' if role=='ENVIRONMENT' else 'state' if role=='STATE' else 'data'
            dependencies.append(dict(port=port(index), **dep(**{channel:[token]})))
            formals.append(dict(index=index, role=role, port=port(index), token=token))
        ports = dict(formals=formals, environment=port(0), parameters=formals[1:3],
            state=port(3), **{'return':port(4), 'return_world':port(5)})
        local = dict(tokens=tokens, port_dependencies=dependencies, ports=ports,
            call_sites=[], reads=[], writes=[], normal_return=dep(),
            normal_return_world=dep(state=[name+':3']), effects={'writes':{'status':'UNKNOWN'}})
        row = dict(function_id=name, body=encode(T.Body('unit', template, region)),
            body_scope={'unit':'unit', 'ordinal':function_index}, local=local, ports=ports)
        return row, port

    a, ap = function('a', 0)
    constant, cp = function('constant', 1)
    second, sp = function('second', 2)
    recursive, rp = function('recursive', 3)
    open_call, op = function('open', 4)
    second['local']['normal_return'] = dep(data=['second:2'])
    second['local']['reads'] = [dict(dependencies=dep(environment=['second:0']))]
    for row in (a, constant, second, recursive, open_call):
        local = row['local']
        local['port_dependencies'].extend([
            dict(port=local['ports']['return'], **local['normal_return']),
            dict(port=local['ports']['return_world'], **local['normal_return_world'])])

    def call(row, port, query, *, callee_is_argument=False):
        local = row['local']; name = row['function_id']
        call = dict(call_id=name+':call', query=query, callee=port(6), result=port(7),
            world=port(3), result_world=port(8), positional=[{'port':port(1)}, {'port':port(2)}],
            keywords=[], positional_pack=None, keyword_pack=None)
        local['call_sites'].append(call)
        local['port_dependencies'].append(dict(port=port(6),
            **(dep(data=[name+':1']) if callee_is_argument else dep(environment=[name+':0']))))
        for key, kind in (('result','CALL_RESULT'), ('result_world','CALL_WORLD')):
            token = name+':'+kind
            local['tokens'].append(dict(id=token, kind=kind, port=call[key]))
            channel = 'state' if key=='result_world' else 'data'
            local['port_dependencies'].append(dict(port=call[key], **dep(**{channel:[token]}, unknown=[token])))
        local['normal_return'] = dep(data=[name+':CALL_RESULT'], unknown=[name+':CALL_RESULT'])
        local['normal_return_world'] = dep(state=[name+':CALL_WORLD'], unknown=[name+':CALL_WORLD'])
        for value in local['port_dependencies']:
            if value['port']==local['ports']['return']:
                value.update(local['normal_return'])
            if value['port']==local['ports']['return_world']:
                value.update(local['normal_return_world'])
        return call

    ac = call(a, ap, 0)
    rc = call(recursive, rp, 1)
    call(open_call, op, 2, callee_is_argument=True)
    recursive['local']['normal_return']['data'].append('recursive:1')
    for value in recursive['local']['port_dependencies']:
        if value['port']==recursive['ports']['return']:
            value.update(recursive['local']['normal_return'])

    def interface(caller, call, callee, actuals):
        parameters = []
        for formal, actual in zip(callee['ports']['formals'], actuals):
            parameters.append(dict(formal=formal['port'], actuals=[actual],
                kind='FUNCTION_ENVIRONMENT' if formal['role']=='ENVIRONMENT' else
                    'STATE' if formal['role']=='STATE' else 'ARGUMENT'))
        return dict(caller_scope=caller['body_scope'], ordinary_body_entry=True,
            interface=dict(query=call['query'], link=dict(result=call['result'], callee=call['callee'],
                body=callee['body'], parameters=parameters,
                returns=[[callee['ports']['return'],call['result']],
                    [callee['ports']['return_world'],call['result_world']]],
                projections=[], input_envelope=None, conditions=['ORIGINAL_SELECTED_TARGET_GUARD'], binding='DIRECT')))

    entry = {'interfaces':[
        interface(a, ac, constant, [ap(0), ap(1), ap(2), ap(3)]),
        interface(a, ac, second, [ap(0), ap(1), ap(2), ap(3)]),
        interface(recursive, rc, recursive, [rp(0), rp(2), rp(1), rp(3)])]}
    material = [a, constant, second, recursive, open_call]
    original = deepcopy(material)
    normalized = _compose(material, entry)
    result = expand_interprocedural_dependencies(normalized)
    assert material==original
    rows = {row['function_id']:row for row in result['functions']}
    selected = rows['a']['conditional_selected_target']
    assert selected['normal_return']==dep(data=['a:2'], environment=['a:0'])
    assert selected['normal_return_world']==dep(environment=['a:0'], state=['a:3'])
    assert selected['reads'][0]['dependencies']==dep(environment=['a:0'])
    assert rows['a']['local']['effects']['writes']['status']=='UNKNOWN'
    recursive_result = rows['recursive']['conditional_selected_target']['normal_return']
    assert recursive_result==dep(data=['recursive:1','recursive:2'], environment=['recursive:0'])
    scc = next(row for row in result['sccs'] if row['function_ids']==['recursive'])
    assert scc['converged'] and scc['recursive'] and scc['iterations'] < 10
    opened = rows['open']['conditional_selected_target']['normal_return']
    assert opened['data']==['open:1','open:2'] and opened['state']==['open:3'] and opened['unknown']
    assert all(not row['transformation_authorized'] and not row['termination_proven'] for row in rows.values())
    assert all(row['future_dispatch']['status']=='UNKNOWN' for row in result['calls'])
    assert len(result['functions'])==5 and len(result['calls'])==3
    changed = deepcopy(material)
    changed[2]['local']['normal_return'] = dep(data=['second:1'])
    for value in changed[2]['local']['port_dependencies']:
        if value['port']==changed[2]['ports']['return']:
            value.update(changed[2]['local']['normal_return'])
    assert semantic_sha256(_compose(changed, entry)) != semantic_sha256(normalized)
    normalized_rows = {row['function_id']:row for row in normalized['functions']}
    for index, row in enumerate(normalized['functions']):
        assert row['local']==material[index]['local']
        for field in ('normal_return','normal_return_world'):
            reference = row['conditional_selected_target'][field]
            assert resolve_dependency_set(normalized, reference)==rows[row['function_id']]['conditional_selected_target'][field]
    a_call = next(row for row in normalized['calls'] if row['caller_function_id']=='a')
    a_read = normalized_rows['a']['conditional_selected_target']['reads'][0]
    assert a_call['dispatch_environment_dependencies']['dependency_ref']==a_read['dependency_ref']
    assert len(normalized['unknown_facts'])==1
    references = [value for row in normalized['dependency_sets'].values() for value in row['unknown_refs']]
    assert references and set(references)==set(normalized['unknown_facts'])
    table = summary._DependencyTable()
    original_dependency = (frozenset(['value']), frozenset(), frozenset(),
        frozenset([summary._canonical({'reason':'UNRESOLVED', 'source':{'pointer':'#/original/fact'}})]))
    same_unknown = (frozenset(['another']), *original_dependency[1:])
    first_ref, second_ref = table.intern(original_dependency), table.intern(same_unknown)
    ledger = {'dependency_sets':table.sets, 'unknown_facts':table.unknowns}
    assert first_ref != second_ref and len(table.unknowns)==1
    assert resolve_dependency_set(ledger, first_ref)==summary._pack(original_dependency)
    assert resolve_dependency_set(ledger, second_ref)==summary._pack(same_unknown)
    reference = normalized_rows['a']['conditional_selected_target']['normal_return']['dependency_ref']
    detached = resolve_dependency_set(normalized, reference)
    detached['data'].clear()
    assert resolve_dependency_set(normalized, reference)['data']==['a:2']
    monkeypatch.setattr(summary, 'derive_interprocedural_summary', lambda *_: _compose(material, entry))
    assert verify_interprocedural_summary(None, normalized)==normalized
    forged = deepcopy(normalized)
    forged['functions'][0]['conditional_selected_target']['normal_return'] = deepcopy(
        forged['functions'][0]['conditional_selected_target']['normal_return_world'])
    with pytest.raises(ValueError, match='source-route replay'):
        verify_interprocedural_summary(None, forged)
    forged = deepcopy(normalized)
    forged['functions'][0]['conditional_selected_target']['normal_return']['dependency_ref']='dependency-set:missing'
    with pytest.raises(ValueError, match='missing|content ID'):
        resolve_dependency_set(forged, forged['functions'][0]['conditional_selected_target']['normal_return'])
    with pytest.raises(ValueError, match='source-route replay'):
        verify_interprocedural_summary(None, forged)
    forged = deepcopy(normalized)
    forged['dependency_sets'][reference]['data'].append('forged-source')
    with pytest.raises(ValueError, match='content ID'):
        resolve_dependency_set(forged, reference)
    with pytest.raises(ValueError, match='source-route replay'):
        verify_interprocedural_summary(None, forged)
    forged = deepcopy(normalized)
    unknown_ref = next(iter(forged['unknown_facts']))
    forged['unknown_facts'][unknown_ref]['reason']='forged-reason'
    open_reference = normalized_rows['open']['conditional_selected_target']['normal_return']
    with pytest.raises(ValueError, match='content ID'):
        resolve_dependency_set(forged, open_reference)
    with pytest.raises(ValueError, match='source-route replay'):
        verify_interprocedural_summary(None, forged)
