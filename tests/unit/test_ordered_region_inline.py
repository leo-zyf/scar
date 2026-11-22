"""Two CALLs to a shared leaf retain distinct frames and caller scopes."""
from dataclasses import replace
from types import SimpleNamespace as NS

from scar.analysis.ordered_region_inline import OrderedRegionInlineIndex
from scar.ir.program import normal_graph as F, source_productions as P
from scar.ir.program.dependencies import (DependencyFrame as Frame,DependencyPoint as Point,
    GraphNamespace,context_push,context_frames)
from scar.ir.program.ids import OperationID,RegionID,ValueID,UseID
from scar.ir.program.ordered_regions import OrderedInlineRoute
from scar.ir.program.providers import ProviderArtifact,ProviderFactReference
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.program.source_declarations import SourceTemplateKey
from scar.ir.record_codec import encode,decode


def test_two_inline_calls_do_not_share_context_or_another_same_unit_caller():
    artifact=ProviderArtifact('/fixture/source.json','0'*64,1,'fixture')
    locator=SourceArtifactLocator(artifact,artifact.sha256,artifact.size_bytes,'canonical-json')
    namespace=lambda root:GraphNamespace(locator,'fixture',RegionID(root))
    entry_ns,source_ns,ordered_ns=map(namespace,('entry','source','ordered'))
    outer_entry,other_entry,leaf_entry=map(RegionID,('outer','other','leaf'))
    outer_key=SourceTemplateKey(1,P.ProductionPurpose.STATEMENT)
    other_key=SourceTemplateKey(2,P.ProductionPurpose.STATEMENT)
    prod_outer,prod_other=map(P.ProductionRegionID,('outer','other'))
    source=NS(unit='unit',nodes={
        outer_entry:NS(anchor=NS(template=outer_key,original=prod_outer)),
        other_entry:NS(anchor=NS(template=other_key,original=prod_other))},plans={
        outer_key:NS(compilation_flow=NS(ports=(NS(region=prod_outer,code_scope=1),))),
        other_key:NS(compilation_flow=NS(ports=(NS(region=prod_other,code_scope=2),)))})
    source_op=Point(source_ns,OperationID('shared-leaf-operation'),None)
    first=Point(ordered_ns,ValueID('first-inline-value'),None)
    second=Point(ordered_ns,ValueID('second-inline-value'),None)
    outer=Frame(entry_ns,OperationID('outer-call'),UseID('outer-use'),ValueID('outer-value'),
        source_ns,entry=outer_entry)
    other=replace(outer,operation=OperationID('other-call'),entry=other_entry)
    first_frame=Frame(source_ns,OperationID('call-A'),UseID('use-A'),ValueID('value-A'),
        source_ns,entry=leaf_entry,selection=ProviderFactReference(artifact,'#/interfaces/0'))
    second_frame=replace(first_frame,operation=OperationID('call-B'),
        callee_use=UseID('use-B'),selection=ProviderFactReference(artifact,'#/interfaces/1'))
    evidence=ProviderFactReference(artifact,'#/origin/0')
    route=OrderedInlineRoute('#/nodes/0',source_op,evidence,evidence,'provider','unit',0,40,
        first_frame.selection,first_frame,F.CodeScopeRef('unit',1),F.CodeScopeRef('unit',3),('CONDITIONAL',))
    another=replace(route,ordered_key='#/nodes/1',frame=second_frame,query=41,
        interface_ordinal=1,interface=second_frame.selection)
    index=OrderedRegionInlineIndex.__new__(OrderedRegionInlineIndex)
    index.view=NS(namespaces={source_ns:source})
    index.routes=(route,another)
    index.by_edge={(index._key(source_op),index._key(first)):(0,),
        (index._key(source_op),index._key(second)):(1,)}
    outer_context=context_push((),outer)
    first_context=index.contexts_for(source_op,first,outer_context)[0][1]
    second_context=index.contexts_for(source_op,second,outer_context)[0][1]
    assert context_frames(first_context)==(outer,first_frame)
    assert context_frames(second_context)==(outer,second_frame)
    assert index.pop_source_context(source_op,first,first_context)==((route,outer_context),)
    assert not index.pop_source_context(source_op,first,second_context)
    assert not index.contexts_for(source_op,first,context_push((),other))
    assert decode(OrderedInlineRoute,encode(route))==route
