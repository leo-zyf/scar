"""Motion facts bind the body and retain origin/content/execution obligations."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import pytest

from scar.analysis.source_motion_window import derive_array_motion, verify_array_motion
from scar.compiler.model import ArrayRegion, Expr, SourceRef


def test_motion_dependency_closure_and_unproved_content_stability(tmp_path):
    path = tmp_path/'source.py'
    path.write_text('retained source identity; metadata fixture, never executed\n')
    source = SourceRef(str(path), sha256(path.read_bytes()).hexdigest(), 'label_only',
        1, 0, 'pycode-v2:'+'7'*64, ({'original_fact': 'retained-by-digest'},))
    body = ArrayRegion('motion:fixture', source, ('carry', 'fixed', 'other'), (
        Expr('c', 'input', attrs={'name': 'carry'}),
        Expr('x', 'input', attrs={'name': 'fixed'}),
        Expr('y', 'input', attrs={'name': 'other'}),
        Expr('one', 'const', attrs={'value': 1}),
        Expr('xx', 'mul', ('x', 'x')),
        Expr('sum', 'add', ('xx', 'y')),
        Expr('inv', 'neg', ('sum',)),
        Expr('result', 'add', ('c', 'inv')),
    ), 'result')
    interface = dict(carry_parameter='carry', readonly_parameters=['fixed', 'other'],
        source_F4_replayed=True, runtime_guards_discharged=True)
    report = derive_array_motion(body, carry_parameter='carry', loop_interface=interface)
    assert verify_array_motion(report, body, carry_parameter='carry', loop_interface=interface) == report
    assert report['dependencies']['inv'] == ['fixed', 'other']
    assert report['dependencies']['one'] == []
    assert report['invariant_nodes'] == ['xx', 'sum', 'inv']
    assert report['invariant_roots'] == ['inv']
    assert report['placements']['c']['temporal_placement'] == 'ITERATION_CARRY'
    assert report['placements']['x']['temporal_placement'] == 'PRE_TEMPORAL_LOOP'
    assert report['placements']['inv']['temporal_placement'] == 'PRE_TEMPORAL_LOOP'
    assert report['placements']['x']['origin_binding_stability']['status'] == 'CONDITIONAL'
    assert report['placements']['x']['referent_content_stability']['status'] == 'CONDITIONAL'
    assert report['placements']['inv']['effects']['input_aliasing'] == 'MAY_ALIAS_NOT_INFERRED_DISJOINT'
    assert 'origins' not in report['source'] and report['source_origins']['count'] == 1
    assert all(row['status'] == 'UNRESOLVED'
        for rows in report['conditional_obligations'].values() for row in rows)
    assert not report['source_replacement_authorized'] and not report['runtime_guards_discharged']
    assert report['zero_trip'].startswith('RETURN_ORIGINAL_CARRY_ALIAS_BEFORE_')

    # No scalar operation is evaluated here, including a possible exception.
    blocked = replace(body, nodes=(*body.nodes[:4],
        Expr('zero', 'const', attrs={'value': 0}),
        Expr('scalar', 'div', ('one', 'zero')),
        Expr('inv', 'mul', ('x', 'scalar')),
        Expr('result', 'add', ('c', 'inv'))))
    analysis = derive_array_motion(blocked, carry_parameter='carry', loop_interface=interface)
    assert analysis['dependencies']['scalar'] == []
    assert analysis['placements']['scalar']['temporal_placement'] == 'ITERATION_BODY'
    assert analysis['placements']['inv']['temporal_placement'] == 'ITERATION_BODY'
    assert analysis['placements']['inv']['blocking_exprs'] == ['scalar']
    assert analysis['status'] == 'UNRESOLVED'
    assert analysis['placements']['scalar']['effects']['may_raise'].startswith('UNKNOWN_')
    with pytest.raises(ValueError, match='replay'):
        verify_array_motion(report, blocked, carry_parameter='carry', loop_interface=interface)
    with pytest.raises(ValueError, match='replay'):
        verify_array_motion(report, body, carry_parameter='carry',
            loop_interface={**interface, 'unproved_extra_marker': True})
    changed = deepcopy(report)
    changed['placements']['result']['temporal_placement'] = 'PRE_TEMPORAL_LOOP'
    with pytest.raises(ValueError, match='replay'):
        verify_array_motion(changed, body, carry_parameter='carry', loop_interface=interface)
