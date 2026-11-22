"""No scalar invariance by callee-only reasoning; loop frames stay matched."""
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace as NS

from scar.analysis.ordered_loop_transfer import OrderedLoopTransferProvider,_input_role
from scar.ir.program import ordered_regions as R, normal_graph as G, normal_targets as T, source_productions as P
from scar.ir.program.dependencies import (GraphNamespace,DependencyPoint as Point,DependencyFrame as Frame,
    DependencyKind as K,DependencyStrength,context_push)
from scar.ir.program.ids import RegionID,ValueID,ControlPortID,OperationID,UseID
from scar.ir.program.normal_links import CodeScopeRef
from scar.ir.program.providers import ProviderArtifact
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.program.source_template_links import SourceTemplateKey


def test_caller_store_and_effect_barriers_prevent_false_scalar_invariance_and_frame_splicing():
    assert _input_role('scalar',(),(0,),())is R.OrderedLoopInputRole.CALLER_REWRITTEN_FIELD
    assert _input_role('scalar',(),(),(0,))is R.OrderedLoopInputRole.LOOP_LIVE_IN_ROOT_SELECTED_VALUE_UNPROVEN
    assert _input_role('scalar',(),(),())is R.OrderedLoopInputRole.CONDITIONAL_LOOP_LIVE_IN_SCALAR
    assert _input_role('array',(3,),(),(0,))is R.OrderedLoopInputRole.ORDERED_ARRAY_CONTENT_CARRY
    artifact=ProviderArtifact('/fixture/original.json','0'*64,1,'fixture')
    locator=SourceArtifactLocator(artifact,artifact.sha256,artifact.size_bytes,'canonical-json')
    source_ns=GraphNamespace(locator,'source',RegionID('source'))
    ordered_ns=GraphNamespace(locator,'ordered',RegionID('ordered'))
    first=Frame(source_ns,OperationID('loop-call'),UseID('loop-use'),ValueID('loop-value'),
        source_ns,entry=RegionID('wrapper'))
    final=Frame(source_ns,OperationID('region-call'),UseID('region-use'),ValueID('region-value'),
        ordered_ns,entry=ordered_ns.root)
    key=SourceTemplateKey(3,P.ProductionPurpose.STATEMENT)
    header=T.GlobalPort('unit',G.Port(key,P.ProductionStepID('loop'),G.PortKind.ENTER))
    header_point=Point(source_ns,ControlPortID('loop-header'),None)
    formal=Point(source_ns,ValueID('loop-caller-formal'),None)
    scalar=Point(ordered_ns,ValueID('scalar-input'),None)
    epoch0=Point(ordered_ns,ValueID('entry-epoch'),None)
    epoch1=Point(ordered_ns,ValueID('exit-epoch'),None)
    p=OrderedLoopTransferProvider.__new__(OrderedLoopTransferProvider)
    p.view=NS(namespaces={source_ns:NS()},source_port=lambda _:header_point)
    p.ordered=NS(locator=locator)
    p.activation=NS(bindings=(NS(ordered_frame=final),))
    p.providers={'fixture':NS(manifest=locator)}
    p.loops=(NS(binding=0,header=header,call_path=(NS(frame=first,provider='fixture',
        caller_scope=CodeScopeRef('unit',1)),NS(frame=final,provider='fixture'))),)
    p._edges=[];p.in_index=defaultdict(list);p.out_index=defaultdict(list);p.check=lambda:None
    p._edge(0,formal,0,scalar,-1,K.STATE_INPUT,'selected-field',('CALLER_EFFECTS_PRESERVE_FIELD',))
    p._edge(0,epoch1,-1,epoch0,-1,K.STATE_INTERFACE,'conditional-recurrence',('CALLER_EFFECTS_PRESERVE_CONTENT',))
    context=context_push(None,first,final)
    selected=replace(scalar,context=context)
    edge=p.incoming(selected).edges[0]
    assert edge.source==formal and edge in p.outgoing(formal).edges
    assert edge.conditions and edge.strength is DependencyStrength.MAY
    wrong=replace(first,operation=OperationID('another-loop-call'))
    assert not p.incoming(replace(scalar,context=context_push(None,wrong,final))).edges
    assert not p.incoming(replace(scalar,context=context_push(None,final))).edges
    carried=p.incoming(replace(epoch0,context=context)).edges[0]
    assert carried.source==replace(epoch1,context=context)
    assert carried in p.outgoing(carried.source).edges
    assert carried.source in carried.evidence[0].anchors
