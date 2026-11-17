"""Structural composite reads never replace original source admission."""
from dataclasses import asdict,replace
from hashlib import sha256
import json

import pytest

from scar.compiler.model import ArrayRegion,Expr,SourceRef
from scar.compiler import pipeline_loop_frontend as frontend
from scar.compiler.pipeline_model import CalleeGuard,RegionCall,compose_pipeline,prepare_pipeline_leaf
from scar.optimization.optimizer import optimize_region


def test_composite_material_decode_binds_fusion_and_keeps_full_source_gate(tmp_path,monkeypatch):
    source_path=tmp_path/'original.py';source_path.write_bytes(b'# retained source bytes\n')
    digest=sha256(source_path.read_bytes()).hexdigest()
    source=SourceRef(str(source_path),digest,'label',1,0,'source-code:root')
    pre_source=replace(source,code_key='source-code:prepare')
    add_source=replace(source,code_key='source-code:combine')
    pre=ArrayRegion('prepare',pre_source,('fixed',),(
        Expr('f','input',attrs={'name':'fixed'}),Expr('pre','mul',('f','f'))),'pre')
    add=ArrayRegion('combine',add_source,('payload','prepared'),(
        Expr('x','input',attrs={'name':'payload'}),Expr('p','input',attrs={'name':'prepared'}),
        Expr('out','add',('x','p'))),'out')
    preplan=prepare_pipeline_leaf(optimize_region(pre));addplan=prepare_pipeline_leaf(optimize_region(add))
    pipeline=compose_pipeline(source,('payload','fixed'),(
        RegionCall('pre',preplan,('arg:1',),'prepared',CalleeGuard(pre_source,('first',))),
        RegionCall('add',addplan,('arg:0','prepared'),'output',CalleeGuard(add_source,('second',)))),
        'output')
    component=frontend._component(pipeline,{(pre_source.path,pre_source.code_key):frontend._Component(preplan),
        (add_source.path,add_source.code_key):frontend._Component(addplan)})
    owner=replace(source,code_key='source-code:loop')
    proof=frontend.CompositeInvariantLoopProof('proof:test',owner,('initial','fixed','count'),
        component.fused_body,'payload','count',{'payload':'initial','fixed':'fixed'},source,'root',
        component.guards,component.compositions,())
    path=tmp_path/'proof.json';reference=frontend.write_composite_invariant_loop_proof(proof,path)
    def no_source_authority(*args,**kwargs):
        raise ValueError('full original source replay still required')
    monkeypatch.setattr(frontend,'validate_composite_invariant_loop_proof',no_source_authority)
    decoded,retained=frontend._read_composite_invariant_loop_materials(path)
    assert retained==reference and asdict(decoded)==asdict(proof)
    assert decoded.transformation_authorized is False
    interface=frontend.canonical_composite_invariant_motion_interface(decoded)
    assert set(interface)=={'carry_parameter','readonly_parameters','original_composite_f4_proof'}
    assert interface['readonly_parameters']==['fixed']
    with pytest.raises(ValueError,match='full original source replay'):
        frontend.read_composite_invariant_loop_proof(path,catalog=object())
    document=json.loads(path.read_bytes())
    document['body']['nodes'][-1]['op']='sub'
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError,match='original root composition fusion'):
        frontend._read_composite_invariant_loop_materials(path)
    path.write_text(json.dumps(dict(document,body=asdict(proof.body))))
    source_path.write_bytes(b'# changed original bytes\n')
    with pytest.raises(ValueError,match='source bytes changed'):
        frontend._read_composite_invariant_loop_materials(path)
