"""LICM from dependency roots, with real unchanged C++ math kernels."""
from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

from scar.compiler.model import ArrayRegion,Expr,SourceRef
from scar.compiler.invariant_loop import build_invariant_loop,plan_invariant_loop,load_invariant_loop


def test_hoist_once_zero_trip_alias_and_whole_loop_fallback(tmp_path):
    path=tmp_path/'original.py';path.write_text('def step(state, fixed):\n    return state + fixed * fixed\n')
    source=SourceRef(str(path),sha256(path.read_bytes()).hexdigest(),'step',1,0,'pycode-v2:'+'5'*64)
    body=ArrayRegion('test:licm',source,('state','fixed'),
        (Expr('s','input',attrs=dict(name='state')),Expr('x','input',attrs=dict(name='fixed')),
         Expr('xx','mul',('x','x')),Expr('result','add',('s','xx'))),'result')
    interface=dict(carry_parameter='state',readonly_parameters=['fixed'],
        authority='UNIT_FIXTURE_CONDITIONAL_LOOP_INTERFACE_NOT_SOURCE_PROOF')
    plan=plan_invariant_loop(body,carry_parameter='state',loop_interface=interface,memory_budget_bytes=4096)
    assert plan['hoisted_roots']==['xx'] and plan['dependencies']['xx']==['fixed']
    artifact=build_invariant_loop(body,tmp_path/'licm',carry_parameter='state',loop_interface=interface,memory_budget_bytes=4096)
    state=np.ones((3,5));fixed=np.full((3,5),0.125);expected=state.copy()
    for _ in range(7):expected=expected+fixed*fixed
    actual=artifact.run(dict(state=state,fixed=fixed),7)
    assert np.array_equal(actual,expected) and np.array_equal(state,np.ones((3,5)))
    assert not np.shares_memory(actual,state)
    assert artifact.report['last_execution']==dict(hoisted_calls=1,residual_calls=7,status=0)
    assert artifact.run(dict(state=state),0) is state
    assert artifact.report['last_execution']['hoisted_calls']==0
    assert artifact.run(dict(state=state),-3) is state
    assert not artifact.report['source_replacement_authorized']
    control=build_invariant_loop(body,tmp_path/'without-hoist',carry_parameter='state',
        loop_interface=interface,memory_budget_bytes=4096,hoist_invariants=False)
    assert control.plan['hoisted_roots']==[] and control.plan['inferred_invariant_roots']==['xx']
    assert np.array_equal(control.run(dict(state=state,fixed=fixed),7),expected)
    assert control.report['last_execution']==dict(hoisted_calls=0,residual_calls=7,status=0)
    fixed[:]=0.25
    artifact=load_invariant_loop(tmp_path/'licm')
    changed=artifact.run(dict(state=state,fixed=fixed),1)
    assert np.array_equal(changed,state+fixed*fixed)
    fixed[:]=1e200;fallbacks=[]
    def original():
        fallbacks.append(1)
        with np.errstate(over='ignore'):return state+fixed*fixed
    actual=artifact.run(dict(state=state,fixed=fixed),3,original_loop=original)
    assert len(fallbacks)==1 and np.isinf(actual).all()
    assert artifact.report['last_execution']['residual_calls']==0
    assert np.array_equal(state,np.ones((3,5)))
    with pytest.raises(ValueError,match='missing guarded'):
        plan_invariant_loop(replace(body,assumptions=('FRESH_OUTPUT',)),carry_parameter='state',
            loop_interface=interface,memory_budget_bytes=4096)
    with pytest.raises(ValueError,match='no nonconstant'):
        plan_invariant_loop(replace(body,nodes=(*body.nodes[:2],Expr('mixed','mul',('s','x')),
            Expr('result','add',('s','mixed')))),carry_parameter='state',loop_interface=interface,memory_budget_bytes=4096)
