"""Temporal LICM reuses scalar values without exposing failed private output."""
from hashlib import sha256

import numpy as np

from scar.compiler.model import ArrayRegion,Expr,SourceRef
from scar.compiler.temporal_invariant_loop import build_temporal_invariant_loop,load_temporal_invariant_loop


def test_temporal_scalar_preheader_exact_results_alias_and_fp_fallback(tmp_path):
    source=tmp_path/'original.py';source.write_text('def step(state, fixed):\n    return state + fixed * fixed\n')
    ref=SourceRef(str(source),sha256(source.read_bytes()).hexdigest(),'step',1,0,'pycode-v2:'+'4'*64)
    body=ArrayRegion('test:temporal-licm',ref,('state','fixed'),
        (Expr('s','input',attrs=dict(name='state')),Expr('x','input',attrs=dict(name='fixed')),
         Expr('xx','mul',('x','x')),Expr('next','add',('s','xx'))),'next')
    interface=dict(carry_parameter='state',readonly_parameters=['fixed'],authority='CONDITIONAL_UNIT_INTERFACE')
    artifact=build_temporal_invariant_loop(body,tmp_path/'native',carry_parameter='state',
        loop_interface=interface,memory_budget_bytes=4096)
    state=np.ones((5,3)).T;fixed=np.full((5,3),0.125).T
    expected=state.copy()
    for _ in range(13):expected=expected+fixed*fixed
    result=artifact.run(dict(state=state,fixed=fixed),13)
    assert np.array_equal(result,expected) and np.array_equal(state,np.ones((3,5)))
    assert not np.shares_memory(result,state)
    assert artifact.report['last_execution']==dict(native_calls=1,element_preheaders=15,
        element_iterations=195,logical_iterations=13,status=0)
    assert artifact.report['full_invariant_buffers']==artifact.report['carry_pingpong_buffers']==0
    assert artifact.report['final_output_buffers']==1 and not artifact.report['physical_register_allocation_proven']
    assert artifact.run(dict(state=state),0) is state
    assert artifact.report['last_execution']['element_iterations']==0
    assert artifact.run(dict(state=state),-1) is state
    artifact=load_temporal_invariant_loop(tmp_path/'native')
    fixed[:]=0.25
    assert np.array_equal(artifact.run(dict(state=state,fixed=fixed),1),state+fixed*fixed)
    fixed[:]=1e200;calls=[]
    def fallback():
        calls.append(1)
        with np.errstate(over='ignore'):return state+fixed*fixed
    result=artifact.run(dict(state=state,fixed=fixed),3,original_loop=fallback)
    assert len(calls)==1 and np.isinf(result).all()
    assert artifact.report['last_execution']['fallback'] is True
    assert np.array_equal(state,np.ones((3,5)))
