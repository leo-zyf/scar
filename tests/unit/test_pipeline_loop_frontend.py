"""Fusion keeps invariant input roots and every function-global guard hop."""
from dataclasses import replace
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import builtins
import runpy

import pytest

from scar.compiler.graph_frontend import GraphRegionRejected
from scar.compiler.model import ArrayRegion,Expr,SourceRef
from scar.compiler.pipeline_loop_frontend import _Component,_component
from scar.compiler.pipeline_loop_frontend import CompositeInvariantLoopProof,FunctionGlobalGuard
from scar.compiler.pipeline_frontend import CalleeGuard, RegionCall, compose_pipeline, prepare_pipeline_leaf
from scar.optimization.optimizer import optimize_region
from scar.optimization.resource_solver import semantic_sha256
from scar.trace.code_snapshot import code_key


def test_composed_invariant_dependencies_keep_nested_code_and_global_routes():
    source=SourceRef('/original/arrays.py','a'*64,'prepare',1,0,'pycode-v2:'+'b'*64)
    prepare=ArrayRegion('prepare',source,('fixed',),(Expr('f','input',attrs={'name':'fixed'}),
        Expr('p','mul',('f','f'))),'p')
    combine_source=replace(source,qualname='combine',code_key='pycode-v2:'+'c'*64)
    combine=ArrayRegion('combine',combine_source,('payload','prepared'),
        (Expr('x','input',attrs={'name':'payload'}),Expr('p','input',attrs={'name':'prepared'}),
         Expr('out','add',('x','p'))),'out')
    prepared=prepare_pipeline_leaf(optimize_region(prepare));combined=prepare_pipeline_leaf(optimize_region(combine))
    wrapper_source=replace(source,qualname='wrapper',code_key='pycode-v2:'+'d'*64)
    pipeline=compose_pipeline(wrapper_source,('payload','fixed'),(
        RegionCall('pre',prepared,('arg:1',),'preprocessed',CalleeGuard(source,('prepare',))),
        RegionCall('final',combined,('arg:0','preprocessed'),'output',CalleeGuard(combine_source,('combine',)))),
        'output')
    known={(source.path,source.code_key):_Component(prepared),
        (combine_source.path,combine_source.code_key):_Component(combined)}
    wrapper=_component(pipeline,known)
    known[wrapper_source.path,wrapper_source.code_key]=wrapper
    relay_source=replace(source,qualname='relay',code_key='pycode-v2:'+'e'*64)
    relay_pipeline=compose_pipeline(relay_source,('payload','fixed'),(
        RegionCall('relay',wrapper.execution,('arg:0','arg:1'),'out',CalleeGuard(wrapper_source,('wrapper',))),),'out')
    relay=_component(relay_pipeline,known)
    assert {guard.global_path for guard in relay.guards}=={('wrapper',),('wrapper','prepare'),('wrapper','combine')}
    nested=next(guard for guard in relay.guards if guard.global_path==('wrapper','prepare'))
    assert nested.owner_source==wrapper_source and nested.target_source==source
    assert len(relay.compositions)==2
    deps={}
    for node in relay.fused_body.nodes:
        deps[node.id]=({node.attrs['name']} if node.op=='input' else set().union(*(deps[i] for i in node.inputs)))
    invariant=[node for node in relay.fused_body.nodes if node.op=='mul']
    assert len(invariant)==1 and deps[invariant[0].id]=={'fixed'}
    assert deps[relay.fused_body.output]=={'payload','fixed'}
    unused=RegionCall('unused',prepared,('arg:1',),'unused',CalleeGuard(source,('prepare',)))
    bad=compose_pipeline(wrapper_source,pipeline.parameters,(*pipeline.calls,unused),pipeline.output)
    with pytest.raises(GraphRegionRejected,match='ORIGINAL_CALL_ARITHMETIC_OUTSIDE_FINAL_DEPENDENCY'):
        _component(bad,known)


def test_composite_boundary_checks_nested_globals_and_zero_trip_alias(tmp_path):
    """The stub observes guard admission; real C++ runs are separate artifacts."""
    import numpy as np
    from scar.compiler.composite_invariant_loop import guarded_composite_invariant_loop
    from scar.compiler.runtime import SourceMismatch
    path=tmp_path/'program.py'
    path.write_bytes((Path(__file__).parents[1]/'workloads/pipeline_licm_program.py').read_bytes())
    module=runpy.run_path(str(path));original=module['simulate'];namespace=original.__globals__
    digest=sha256(path.read_bytes()).hexdigest()
    def source(name):
        function=namespace[name]
        return SourceRef(str(path),digest,function.__qualname__,function.__code__.co_firstlineno,0,code_key(function.__code__))
    body=ArrayRegion('derived:body',source('relay'),('payload','fixed'),
        (Expr('p','input',attrs={'name':'payload'}),Expr('f','input',attrs={'name':'fixed'}),
         Expr('pre','mul',('f','f')),Expr('out','add',('p','pre'))),'out')
    routes=(('simulate','relay',('relay',)),('relay','wrapper',('relay','wrapper')),
        ('wrapper','prepare',('relay','wrapper','prepare')),('wrapper','combine',('relay','wrapper','combine')))
    guards=tuple(FunctionGlobalGuard(source(owner),name,source(name),route) for owner,name,route in routes)
    proof=CompositeInvariantLoopProof('composite:boundary',source('simulate'),('initial','fixed','count'),body,
        'payload','count',{'payload':'initial','fixed':'fixed'},source('relay'),'relay',guards,(),())
    class Artifact:
        calls=0
        plan=dict(body_sha256=semantic_sha256(asdict(body)),carry_parameter='payload',loop_interface=dict(
            authority='ORIGINAL_F4_COMPOSITE_FUSION_AND_FUNCTION_GLOBAL_ROUTE_REPLAY',proof_kind=proof.profile,
            runtime_route_admission_required=True,synthetic_body_is_original_callable=False,
            original_composite_f4_proof=asdict(proof)))
        def run(self,inputs,count,original_loop):
            if count<=0:return inputs['payload']
            self.calls+=1
            return original(inputs['payload'],inputs['fixed'],count)
    artifact=Artifact()
    with pytest.raises(SourceMismatch,match='consumed source movement'):
        guarded_composite_invariant_loop(original,artifact,proof,exclusive_inputs=True,trial=True)
    # A structural loader-output fixture tests only the live callable guards.
    # These synthetic fields do not establish source-window/F4 admission.
    movement_ref=dict(uri=str(tmp_path/'synthetic-movement.json'),sha256='0'*64,size_bytes=0,
        schema='scar.source-composite-loop-motion')
    artifact.plan['loop_interface']['source_motion']=movement_ref
    artifact.plan['execution_movement_binding']=dict(schema='scar.temporal-source-movement-binding',
        schema_version=1,source_motion=movement_ref,source_body_sha256=artifact.plan['body_sha256'])
    call=guarded_composite_invariant_loop(original,artifact,proof,exclusive_inputs=True,trial=True)
    initial=np.zeros((2,2));fixed=np.full((2,2),.5)
    assert np.array_equal(call(initial,fixed,10),original(initial,fixed,10)) and artifact.calls==1
    namespace['prepare']=lambda value:value*3
    assert np.array_equal(call(initial,fixed,10),original(initial,fixed,10)) and artifact.calls==1
    assert call.scar_stats['fallback_calls']==1
    namespace['prepare']=module['prepare']
    class Opaque:
        @property
        def shape(self):raise AssertionError('zero trip inspected an unused array')
    assert call(initial,Opaque(),0) is initial and call(initial,Opaque(),-2) is initial
    assert call.scar_stats['alias_returns']==2 and artifact.calls==1
    ordinary_range=builtins.range
    try:
        builtins.range=lambda count:ordinary_range(0)
        assert call(initial,fixed,10) is initial
    finally:builtins.range=ordinary_range
    assert call.scar_stats['fallback_calls']==2 and artifact.calls==1


def test_composite_movement_consumes_pinned_materials_effects_and_optimized_placements(tmp_path,monkeypatch):
    """Synthetic source-query records test the consumer, never F4 authority."""
    from copy import deepcopy
    import json
    from types import SimpleNamespace
    from test_array_call_effects import _composite_case
    from test_array_motion_legality import _strict_window_case
    from scar.analysis.array_call_effects import derive_composite_call_effects
    from scar.analysis.array_motion import derive_array_motion
    from scar.analysis.array_motion_legality import derive_array_motion_legality
    from scar.compiler.composite_invariant_loop import _loop_interface
    from scar.compiler.invariant_loop_runtime import _require_source_movement_binding
    from scar.compiler.runtime import SourceMismatch
    from scar.compiler.pipeline_loop_frontend import (canonical_composite_invariant_motion_interface,
        write_composite_invariant_loop_proof)
    from scar.compiler.temporal_invariant_loop import _plan,_source
    from scar.ir.execution_recipe import artifact_reference

    def forbidden(*args,**kwargs):
        raise AssertionError('metadata consumer must not compile or load native code')
    monkeypatch.setattr('subprocess.Popen',forbidden)
    monkeypatch.setattr('ctypes.CDLL',forbidden)
    proof=_composite_case(tmp_path)
    _,direct,window=_strict_window_case()
    source=replace(proof.source,origins=tuple(direct['original_f4_proof']['source']['origins']))
    selected=window['selected_call']
    discarded={selected['positional'][2]['port']['owner']['wire'],
        window['lexical_bindings'][1]['reads'][1]['value']['port']['owner']['wire']}
    selected['positional']=selected['positional'][:2]
    origins=[deepcopy(row) for row in direct['original_f4_proof']['origins']if row['step']not in discarded]
    original_call=next(row for row in origins if row.get('role')=='invariant_array_leaf_call')
    original_call.update(leaf_region_id=proof.body.id,outer_parameter_indices=[0,1],
        call_inputs=selected['positional'],arguments=[selected['callee'],*selected['positional'],selected['world_before']])
    proof=replace(proof,source=source,origins=tuple(origins),nested_callee_guards=tuple(
        replace(guard,owner_source=source)if guard.owner_source==proof.source else guard
        for guard in proof.nested_callee_guards))
    window['source_version']=dict(path=source.path,sha256=source.sha256)
    window['operations']=[row for row in window['operations']if row['entry']['port']['owner']['wire']not in discarded]
    for index,row in enumerate(window['operations']):
        row['index']=index
        if row['entry']==selected['entry']:
            row['inputs']=[dict(port=row['inputs'][i]['port'],value=value)
                for i,value in enumerate(original_call['arguments'])]
    window['reference_transports']=[row for row in window['reference_transports']
        if not {row['source']['port']['owner']['wire'],row['target']['port']['owner']['wire']}&discarded]
    window['lexical_bindings'][1]['reads']=window['lexical_bindings'][1]['reads'][:1]
    window['reference_transport_slice'].update(seeds=[selected['callee'],*selected['positional']],
        edges=len(window['reference_transports']))
    canonical=canonical_composite_invariant_motion_interface(proof)
    effects=derive_composite_call_effects(proof)
    # This identity-bound synthetic query is not an original ProgramView proof.
    # The controller/exporter independently verifies the real query and F4.
    callee_binding=dict(schema='scar.composite-call-effect-binding',schema_version=1,
        proof_sha256=semantic_sha256(asdict(proof)),body_sha256=semantic_sha256(asdict(proof.body)),
        callee_effects_sha256=semantic_sha256(effects),root_source=asdict(proof.root_callee_source),source=asdict(source),
        outer_call=dict(entry=selected['entry'],callee_port=selected['callee'],positional=selected['positional'],
            keywords=selected['keywords'],result=selected['result'],world_before=selected['world_before'],
            world_after=selected['world_after'],source_origin=original_call),status='CONDITIONAL',frontier=[])
    callee_binding['id']='composite-call-effect-binding:'+semantic_sha256(callee_binding)
    proof_ref=write_composite_invariant_loop_proof(proof,tmp_path/'proof.json')
    sequence=0

    def source_interface(query,*,change=None):
        nonlocal sequence
        query=deepcopy(query);query['id']=semantic_sha256({key:value for key,value in query.items()if key!='id'})
        motion=derive_array_motion(proof.body,carry_parameter=proof.carry_parameter,loop_interface=canonical)
        legality=derive_array_motion_legality(proof.body,carry_parameter=proof.carry_parameter,
            loop_interface=canonical,source_window=query,callee_effects=effects,callee_binding=callee_binding)
        report=dict(schema='scar.source-composite-loop-motion',schema_version=1,proof=proof_ref,
            proof_id=proof.id,proof_sha256=semantic_sha256(asdict(proof)),body_sha256=motion['body_sha256'],
            canonical_interface_sha256=semantic_sha256(canonical),array_motion=motion,
            source_window=query,callee_effects=deepcopy(effects),callee_binding=callee_binding,movement_legality=legality)
        if change is not None:change(report)
        report['id']='source-composite-loop-motion:'+semantic_sha256(report)
        path=tmp_path/('motion-'+str(sequence)+'.json');sequence+=1;path.write_text(json.dumps(report))
        return _loop_interface(proof,source_motion=artifact_reference(path,report['schema'])),report

    interface,report=source_interface(window)
    assert report['movement_legality']['status']=='CONDITIONAL'
    assert report['movement_legality']['source_proof_kind']=='COMPOSITE'
    plan,execution=_plan(proof.body,proof.carry_parameter,interface,4096,None)
    binding=plan['execution_movement_binding']
    assert binding['source_motion']==interface['source_motion']
    assert binding['callee_effects_sha256']==semantic_sha256(effects)
    assert binding['callee_binding_sha256']==semantic_sha256(callee_binding)
    assert not binding['source_F4_replayed_here'] and not binding['source_replacement_authorized']
    assert binding['optimized_body_sha256']==semantic_sha256(asdict(execution.region))
    raw,counts=_source(plan,execution)
    assert counts['invariant_ops']>0 and counts['variant_ops']>0 and b'iteration<steps'in raw
    _require_source_movement_binding(SimpleNamespace(plan=plan),'scar.source-composite-loop-motion')
    wrong=deepcopy(interface);wrong['original_f4_proof']=wrong.pop('original_composite_f4_proof')
    with pytest.raises(ValueError,match='distinct'):_plan(proof.body,proof.carry_parameter,wrong,4096,None)
    forged=deepcopy(plan)
    next(iter(forged['execution_movement_binding']['node_placements'].values()))['placement']='ITERATION_BODY'
    with pytest.raises(ValueError,match='placement'):_source(forged,execution)
    def changed_effects(row):row['callee_effects']['calls'][0]['multiplicity_per_root_call']=123
    wrong,_=source_interface(window,change=changed_effects)
    with pytest.raises(ValueError,match='composition/ABI replay'):_plan(proof.body,proof.carry_parameter,wrong,4096,None)
    blocked=deepcopy(window);blocked['must_execute']['status']='UNRESOLVED'
    blocked['must_execute']['missing']=['SELECTED_CALL_CAN_BE_AVOIDED']
    wrong,unresolved=source_interface(blocked)
    assert unresolved['movement_legality']['status']=='UNRESOLVED'
    with pytest.raises(ValueError,match='unresolved'):_plan(proof.body,proof.carry_parameter,wrong,4096,None)
    micro,_=_plan(proof.body,proof.carry_parameter,canonical,4096,None)
    with pytest.raises(SourceMismatch,match='consumed source movement'):
        _require_source_movement_binding(SimpleNamespace(plan=micro),'scar.source-composite-loop-motion')
    # Exercise the existing exact-copy scalar rule as a conditional micro view;
    # no caller replacement or source admission is obtained from this view.
    from scar.compiler.array_copy_contract import COPY_REQUIRED_GUARDS, copy_attributes
    copied=replace(proof.body,nodes=(*proof.body.nodes,Expr('final-copy','copy',(proof.body.output,),copy_attributes())),
        output='final-copy',assumptions=tuple(dict.fromkeys((*proof.body.assumptions,*COPY_REQUIRED_GUARDS))))
    copy_interface=dict(carry_parameter=proof.carry_parameter,
        readonly_parameters=[name for name in copied.parameters if name!=proof.carry_parameter])
    copy_plan,copy_execution=_plan(copied,proof.carry_parameter,copy_interface,4096,None)
    copy_source,_=_source(copy_plan,copy_execution)
    node=next(row for row in copy_execution.region.nodes if row.op=='copy')
    indices={row.id:index for index,row in enumerate(copy_execution.region.nodes)}
    assert ('const double value'+str(indices[node.id])+' = value'+str(indices[node.inputs[0]])+';').encode()in copy_source
    assert copy_plan['execution_movement_binding']is None
    (tmp_path/'proof.json').write_text((tmp_path/'proof.json').read_text()+'\n')
    with pytest.raises(ValueError,match='byte identity'):_plan(proof.body,proof.carry_parameter,interface,4096,None)
