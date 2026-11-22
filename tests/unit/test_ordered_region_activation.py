"""Context transport only; real original-ABI admission has a CPU integration."""
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace as NS

from scar.analysis.ordered_region_activation import OrderedRegionActivationProvider
from scar.ir.program import ordered_regions as A
from scar.ir.program.dependencies import (GraphNamespace,DependencyPoint as Point,
    DependencyFrame as Frame,DependencyKind as K,DependencyMeaning as M,
    DependencyEdge as Edge,DependencyEvidence as Evidence,DependencyStrength as S,
    context_push)
from scar.ir.program.ids import RegionID,OperationID,UseID,ValueID
from scar.ir.program.providers import ProviderArtifact,ProviderFactReference
from scar.ir.program.source_references import SourceArtifactLocator


def test_alternative_frames_preserve_two_callers_and_bidirectional_evidence():
    def namespace(name):
        raw=ProviderArtifact('/'+name+'.json','0'*64,1,'fixture')
        locator=SourceArtifactLocator(raw,'0'*64,1,'canonical-json')
        return GraphNamespace(locator,'fixture',RegionID(name))
    caller,source,ordered=map(namespace,('caller','source','ordered'))
    form=Point(source,ValueID('formal'),None);input_=Point(ordered,ValueID('input'),None)
    entry=Point(ordered,ordered.root,None)
    frames=[Frame(caller,OperationID('call'+str(i)),UseID('callee'+str(i)),ValueID('callee'+str(i)),source,entry=RegionID('function'))for i in range(2)]
    alternatives=[replace(frame,target=ordered,entry=ordered.root)for frame in frames]
    # Synthetic fixture isolates transport from the separately verified source
    # interface reader and field/ABI proof. No source-purity authority is made.
    p=object.__new__(OrderedRegionActivationProvider)
    p.ordered=NS(namespace=ordered,entry=entry,locator=ordered.graph,
        in_edges=defaultdict(list),out_edges=defaultdict(list))
    p.providers={'fixture':NS(manifest=source.graph)};p.function=NS(body_scope=NS(owner='outer'))
    p.bindings=[A.OrderedActivationBinding('fixture','unit',i,i,
        ProviderFactReference(source.graph.artifact,'#/interfaces/'+str(i)),frame,alternate,('GUARDED_SOURCE_INTERFACE',))
        for i,(frame,alternate)in enumerate(zip(frames,alternatives))]
    p.source_frames=defaultdict(list);p.ordered_frames={}
    p.routes=[];p.out_index=defaultdict(list);p.in_index=defaultdict(list)
    p.check=lambda:None;p._caller_matches=lambda binding,base:True
    p._source_scope=lambda point:'outer'
    for index,(frame,alternate)in enumerate(zip(frames,alternatives)):
        p.source_frames[frame].append(index);p.ordered_frames[alternate]=index
        p._route(index,form,A.OrderedContextSpace.SOURCE,input_,A.OrderedContextSpace.ORDERED,
            K.ARGUMENT,M.REFERENCE,'input')
    first=replace(input_,context=context_push(None,alternatives[0]))
    second=replace(input_,context=context_push(None,alternatives[1]))
    a=p.incoming(first).edges;b=p.incoming(second).edges
    assert len(a)==len(b)==1 and a[0]!=b[0]
    assert a[0].source.context==context_push(None,frames[0])
    assert b[0].source.context==context_push(None,frames[1])
    for edge in(*a,*b):
        assert edge in p.outgoing(edge.source).edges and edge in p.incoming(edge.target).edges
        assert edge.source in edge.evidence[0].anchors and edge.target in edge.evidence[0].anchors
        assert edge.conditions and edge.strength is S.MAY
    foreign=replace(alternatives[0],entry=RegionID('different-function'))
    assert not p.incoming(replace(input_,context=context_push(None,foreign))).edges
    # Local definition provenance must transport its original source namespace
    # under the SOURCE frame, not copy an ORDERED frame onto that endpoint.
    original=Edge(form,input_,K.IMPLEMENTATION,M.STRUCTURE,
        (Evidence('fixture.original',(source.graph,),(form,input_)),),(),S.MAY)
    p.ordered.in_edges[(ordered,input_.node)].append(original)
    p.ordered.out_edges[(source,form.node)].append(original)
    local,_=p._local_definition(first,True,p.bindings[0],None)
    assert len(local)==1 and local[0].source.context==context_push(None,frames[0])
    assert local[0].source in local[0].evidence[0].anchors
    assert local[0]in p.outgoing(local[0].source).edges
