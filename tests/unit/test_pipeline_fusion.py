"""Whole-pipeline fusion retains original arithmetic and guarded contracts."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from scar.compiler.model import SourceRef,ArrayRegion,Expr
from scar.compiler.pipeline_model import (CalleeGuard,RegionCall,compose_pipeline,document,
    pipeline_digest,prepare_pipeline_leaf)
from scar.optimization.optimizer import optimize_region
from scar.optimization.pipeline_fusion import fuse_pipeline_region,verify_pipeline_fusion,fusion_region_digest


def _leaf(index,binary=False,slice=False):
    names=('a','b') if binary else ('a',)
    source=SourceRef('/original/stages.py','a'*64,'stage'+str(index),1+index,0,
        'pycode-v2:'+str(index)*64,({'kind':'original_f4','unit':'unit','expression':'out','step':'original-step'},))
    nodes=[Expr(name,'input',attrs={'name':name}) for name in names]
    if slice:
        nodes.extend((Expr('view','slice',('a',),{'bounds':[[0,None],[0,None]]}),
            Expr('inner','neg',('view',)),Expr('out','neg',('inner',))))
    elif binary:
        nodes.extend((Expr('inner','sub',('a','b')),Expr('out','div',('inner','b'))))
    else:
        nodes.extend((Expr('constant','const',attrs={'value':0.25}),Expr('inner','mul',('a','constant')),
            Expr('out','sub',('inner','a'))))
    return prepare_pipeline_leaf(optimize_region(ArrayRegion('leaf'+str(index),source,names,tuple(nodes),'out')))


def _call(name,plan,inputs,output):
    return RegionCall(name,plan,tuple(inputs),output,CalleeGuard(plan.region.source,(plan.region.source.qualname,)))


def _pipeline(calls,output,parameters=('first','second')):
    source=SourceRef('/original/entry.py','e'*64,'entry',1,0,'pycode-v2:'+'f'*64)
    return compose_pipeline(source,parameters,calls,output)


def test_exact_operand_order_all_expression_provenance_and_contract_replay():
    first,second=_leaf(1),_leaf(2,binary=True)
    pipeline=_pipeline((_call('source-call-a',first,('arg:0',),'a'),
        _call('source-call-b',second,('a','arg:1'),'b')),'b')
    original=deepcopy(document(pipeline));region,certificate=fuse_pipeline_region(pipeline)
    assert region is not None and region.source is pipeline.source
    assert region.parameters==pipeline.parameters
    assert [node.attrs['name'] for node in region.nodes if node.op=='input']==['first','second']
    assert certificate['pipeline_sha256']==pipeline_digest(pipeline)
    assert certificate['region_sha256']==fusion_region_digest(region)
    assert certificate['shape_contract']['external_parameter_shapes_assumed_equal']is False
    by_id={node.id:node for node in region.nodes}
    for call,mapping in zip(pipeline.calls,certificate['calls']):
        assert set(mapping['expr_map'])=={node.id for node in call.plan.region.nodes}
        for node in call.plan.region.nodes:
            if node.op=='input':continue
            fused=by_id[mapping['expr_map'][node.id]]
            assert (fused.op,fused.attrs,fused.inputs)==(node.op,node.attrs,tuple(mapping['expr_map'][v] for v in node.inputs))
    assert certificate['calls'][1]['expr_map']['a']==certificate['calls'][0]['output_expr']
    assert document(pipeline)==original
    assert verify_pipeline_fusion(pipeline,region,json.loads(json.dumps(certificate)))is True
    altered=deepcopy(certificate);altered['calls'][1]['expr_map']['out']='forged'
    with pytest.raises(ValueError,match='replay'):verify_pipeline_fusion(pipeline,region,altered)
    changed=replace(region,nodes=(*region.nodes[:-1],replace(region.nodes[-1],inputs=tuple(reversed(region.nodes[-1].inputs)))))
    with pytest.raises(ValueError,match='operand-order'):verify_pipeline_fusion(pipeline,changed,certificate)
    replacement=replace(second,region=replace(second.region,source=replace(second.region.source,code_key='pycode-v2:'+'c'*64)))
    changed_pipeline=replace(pipeline,calls=(pipeline.calls[0],_call('source-call-b',replacement,('a','arg:1'),'b')))
    with pytest.raises(ValueError,match='replay'):verify_pipeline_fusion(changed_pipeline,region,certificate)


def test_slice_dead_call_and_escape_are_frontiers_without_changing_original_pipeline():
    first=_leaf(1);view=_leaf(2,slice=True)
    sliced=_pipeline((_call('a',first,('arg:0',),'a'),_call('b',view,('a',),'b')),'b')
    region,certificate=fuse_pipeline_region(sliced)
    assert region is None and any(f['reason']=='SLICE_OR_NON_ELEMENTWISE_OPERATION' for f in certificate['frontiers'])
    dead=_pipeline((_call('dead',first,('arg:0',),'unused'),_call('last',first,('arg:0',),'out')),'out')
    before=deepcopy(document(dead));region,certificate=fuse_pipeline_region(dead)
    assert region is None and any(f.get('call_id')=='dead' and f['reason']=='ORIGINAL_CALL_ARITHMETIC_OUTSIDE_FINAL_DEPENDENCY'
        for f in certificate['frontiers'])
    assert verify_pipeline_fusion(dead,None,certificate)is True and document(dead)==before
    escaped=replace(first,proof={**first.proof,'effects':{**first.proof['effects'],'escapes':True}})
    escaping=_pipeline((_call('a',escaped,('arg:0',),'a'),_call('b',first,('a',),'b')),'b')
    region,certificate=fuse_pipeline_region(escaping)
    assert region is None and any(f['reason']=='INTERMEDIATE_ESCAPE_OR_EFFECT_OPEN' for f in certificate['frontiers'])
