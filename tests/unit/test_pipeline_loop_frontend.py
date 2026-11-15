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
from scar.compiler.pipeline_model import CalleeGuard,RegionCall,compose_pipeline,prepare_pipeline_leaf
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
    artifact=Artifact();call=guarded_composite_invariant_loop(original,artifact,proof,exclusive_inputs=True,trial=True)
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
