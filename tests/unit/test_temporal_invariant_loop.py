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


def test_temporal_emitter_consumes_exact_optimized_motion_without_native_execution(tmp_path, monkeypatch):
    from copy import deepcopy
    from dataclasses import replace
    import ctypes
    import subprocess
    import pytest
    from scar.analysis.array_motion import verify_array_motion
    from scar.compiler.temporal_invariant_loop import _plan, _source

    def forbidden(*args, **kwargs):
        raise AssertionError('motion/source metadata test must not compile or load native code')

    monkeypatch.setattr(subprocess,'Popen',forbidden)
    monkeypatch.setattr(ctypes,'CDLL',forbidden)
    source=tmp_path/'original.py';source.write_text('def step(state, fixed):\n    return state + fixed*fixed + fixed*fixed\n')
    ref=SourceRef(str(source),sha256(source.read_bytes()).hexdigest(),'step',1,0,'pycode-v2:'+'5'*64)
    body=ArrayRegion('test:temporal-motion',ref,('state','fixed'),
        (Expr('s','input',attrs=dict(name='state')),Expr('x','input',attrs=dict(name='fixed')),
         Expr('xx','mul',('x','x')),Expr('duplicate','mul',('x','x')),
         Expr('sum','add',('xx','duplicate')),Expr('next','add',('s','sum'))),'next')
    interface=dict(carry_parameter='state',readonly_parameters=['fixed'])
    plan,execution=_plan(body,'state',interface,4096,None)
    assert {node.id for node in execution.region.nodes}!={node.id for node in body.nodes}
    assert plan['motion_analysis']['body_sha256']!=plan['execution_motion_analysis']['body_sha256']
    motion=verify_array_motion(plan['execution_motion_analysis'],execution.region,
        carry_parameter='state',loop_interface=interface)
    raw,counts=_source(plan,execution)
    before,inside=raw.decode().split('for (uint64_t iteration=0; iteration<steps; ++iteration)',1)
    assert counts==dict(invariant_ops=2,variant_ops=1)
    for index,node in enumerate(execution.region.nodes):
        placement=motion['placements'][node.id]['temporal_placement']
        if node.op=='input' and node.attrs['name']=='state':continue
        declaration='const double value'+str(index)+' = '
        assert declaration in (before if placement=='PRE_TEMPORAL_LOOP' else inside)
    forged=deepcopy(plan)
    invariant=next(node.id for node in execution.region.nodes if node.op=='mul')
    forged['execution_motion_analysis']['placements'][invariant]['temporal_placement']='ITERATION_BODY'
    with pytest.raises(ValueError):_source(forged,execution)
    wrong=deepcopy(plan);wrong['execution_motion_analysis']=plan['motion_analysis']
    with pytest.raises(ValueError):_source(wrong,execution)
    names={node.id:'changed:'+node.id for node in execution.region.nodes}
    changed=replace(execution,region=replace(execution.region,
        nodes=tuple(replace(node,id=names[node.id],inputs=tuple(names[key] for key in node.inputs))
            for node in execution.region.nodes),output=names[execution.region.output]))
    with pytest.raises(ValueError):_source(plan,changed)
    assert plan['source_replacement_authorized'] is False and plan['selected_for_use'] is False


def test_source_movement_gate_binds_real_legality_and_emitted_placements(tmp_path, monkeypatch):
    """Synthetic original records test consumption, without granting F4 authority."""
    from copy import deepcopy
    from dataclasses import asdict
    import ctypes
    import json
    import subprocess
    import pytest
    from test_array_motion_legality import _strict_window_case
    from scar.analysis.array_motion import derive_array_motion
    from scar.analysis.array_motion_legality import derive_array_motion_legality
    from scar.compiler.temporal_invariant_loop import _plan, _source
    from scar.ir.execution_recipe import artifact_reference
    from scar.optimization.resource_solver import semantic_sha256

    def forbidden(*args, **kwargs):
        raise AssertionError('source gate metadata test must not compile or load native code')

    monkeypatch.setattr(subprocess,'Popen',forbidden)
    monkeypatch.setattr(ctypes,'CDLL',forbidden)
    body,canonical,window=_strict_window_case()
    canonical['original_f4_proof']['id']='test:original-loop-proof'
    proof_path=tmp_path/'proof.json'
    proof_path.write_text(json.dumps(dict(schema='test.original-loop-proof',schema_version=1,
        **canonical['original_f4_proof'])))
    proof_ref=artifact_reference(proof_path,'test.original-loop-proof')
    sequence=0

    def source_interface(query, *, change=None):
        nonlocal sequence
        query=deepcopy(query)
        query['id']=semantic_sha256({key:value for key,value in query.items()if key!='id'})
        motion=derive_array_motion(body,carry_parameter='state',loop_interface=canonical)
        legality=derive_array_motion_legality(body,carry_parameter='state',loop_interface=canonical,source_window=query)
        report=dict(schema='scar.source-loop-motion',schema_version=2,proof=proof_ref,
            proof_id=canonical['original_f4_proof']['id'],proof_sha256=semantic_sha256(canonical['original_f4_proof']),
            body_sha256=semantic_sha256(asdict(body)),canonical_interface_sha256=semantic_sha256(canonical),
            array_motion=motion,source_window=query,movement_legality=legality)
        report['id']='source-loop-motion:'+semantic_sha256(report)
        if change is not None:change(report)
        path=tmp_path/('motion-'+str(sequence)+'.json');sequence+=1
        path.write_text(json.dumps(report))
        return dict(canonical,source_motion=artifact_reference(path,report['schema'])),report

    interface,report=source_interface(window)
    plan,execution=_plan(body,'state',interface,4096,None)
    binding=plan['execution_movement_binding']
    assert binding['source_motion']==interface['source_motion']
    assert binding['source_motion_sha256']==semantic_sha256(report)
    assert binding['source_window_replay_required'] and not binding['source_replacement_authorized']
    assert not binding['source_F4_replayed_here'] and binding['runtime_guards_are_not_must_execute_proof']
    assert binding['optimized_body_sha256']==semantic_sha256(asdict(execution.region))
    raw,counts=_source(plan,execution)
    assert counts==dict(invariant_ops=1,variant_ops=1)
    assert b'for (uint64_t iteration=0; iteration<steps; ++iteration)'in raw
    forged=deepcopy(plan)
    selected=next(key for key,row in forged['execution_movement_binding']['node_placements'].items()
        if row['placement']=='PRE_TEMPORAL_LOOP')
    forged['execution_movement_binding']['node_placements'][selected]['placement']='ITERATION_BODY'
    with pytest.raises(ValueError,match='placement'):_source(forged,execution)
    wrong,_=source_interface(window,change=lambda row:row.update(id='source-loop-motion:'+'0'*64))
    with pytest.raises(ValueError,match='content identity'):_plan(body,'state',wrong,4096,None)
    blocked=deepcopy(window);blocked['must_execute']['status']='UNRESOLVED'
    blocked['must_execute']['missing']=['SELECTED_CALL_CAN_BE_AVOIDED']
    wrong,blocked_report=source_interface(blocked)
    assert blocked_report['movement_legality']['source_window_effects']['status']=='CONDITIONAL_NO_INPUT_CLOBBERS'
    assert blocked_report['movement_legality']['status']=='UNRESOLVED'
    with pytest.raises(ValueError,match='unresolved'):_plan(body,'state',wrong,4096,None)
    def wrong_canonical(row):
        row['canonical_interface_sha256']='0'*64
        row['id']='source-loop-motion:'+semantic_sha256({key:value for key,value in row.items()if key!='id'})

    malformed,_=source_interface(window,change=wrong_canonical)
    with pytest.raises(ValueError,match='canonical interface'):_plan(body,'state',malformed,4096,None)
    conditional,_=_plan(body,'state',canonical,4096,None)
    assert conditional['execution_movement_binding']is None and not conditional['source_replacement_authorized']
