"""Outer formals survive original lexical/alias transport; inline ones do not."""
from types import SimpleNamespace as NS

from scar.analysis.ordered_region_inputs import derive_ordered_inputs
from scar.ir.program import normal_graph as G,normal_links as F,normal_targets as T
from scar.ir.program import ordered_regions as R, source_productions as P
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.source_reference_slices import ReferenceTransitionKind
from scar.ir.program.source_template_links import SourceTemplateKey
from scar.ir.record_codec import decode,encode


def test_original_alias_read_route_does_not_rebind_an_inline_formal():
    unit='portable-ordered-inputs';outer=SourceTemplateKey(4,P.ProductionPurpose.STATEMENT)
    inner=SourceTemplateKey(8,P.ProductionPurpose.STATEMENT)
    rid,sid=P.ProductionRegionID,P.ProductionStepID
    ports=(
        G.Port(outer,rid('outer'),G.PortKind.PARAMETER,ordinal=0),
        G.Port(outer,rid('outer'),G.PortKind.PARAMETER,ordinal=1),
        G.Port(outer,rid('outer'),G.PortKind.PARAMETER,ordinal=2),
        G.Port(outer,sid('local-read'),G.PortKind.RESULT,completion=C.FALLTHROUGH),
        G.Port(outer,sid('alias'),G.PortKind.RESULT,completion=C.FALLTHROUGH),
        G.Port(inner,rid('inner'),G.PortKind.PARAMETER,ordinal=0),
        G.Port(inner,rid('inner'),G.PortKind.PARAMETER,ordinal=1),
        G.Port(inner,rid('inner'),G.PortKind.PARAMETER,ordinal=2),
        G.Port(outer,sid('definition'),G.PortKind.RESULT,completion=C.FALLTHROUGH),
        G.Port(inner,sid('definition'),G.PortKind.RESULT,completion=C.FALLTHROUGH),
    )
    # The identical display name is deliberately insufficient to join frames.
    function=lambda key,body,formals,value: NS(template=key,body=rid(body),
        parameter_names=('payload',),formals=formals,value=value,
        body_scope=NS(owner=F.CodeScopeRef(unit,key.occurrence)))
    scope=F.Namespace(F.NamespaceKind.LEXICAL,outer.occurrence,'payload')
    facts=NS(unit=unit,ports=ports,aliases=((3,4),),
        names=(F.NameAccess(scope,1,True),F.NameAccess(scope,3,False)),
        functions=(function(outer,'outer',(0,1,2),8),function(inner,'inner',(5,6,7),9)),
        constants=(),classes=(),attributes=())
    checked=[]
    store=NS(load=lambda identity:facts if identity==unit else None,
        rows={unit:{'facts':'/fixture/original-facts.json','sha256':'0'*64,'bytes':1}})
    source=NS(unit=unit,facts=facts,check=lambda:checked.append('source'),
        material=NS(fact_store=store,validate_parents=lambda:checked.append('parents')))
    origin=lambda index:{'kind':'original_f4','unit':unit,
        'value':encode(T.GlobalPort(unit,ports[index]))}
    region=NS(parameter_names=('payload',),
        source=NS(origins=({'kind':'source_identity','unit':unit,'occurrence':outer.occurrence},)),
        inputs=(NS(path=('payload',),origins=(origin(4),)),
            NS(path=('payload',),origins=(origin(6),))))
    ordered=NS(check=lambda:checked.append('ordered'),region=region,
        view=NS(unit=lambda identity:source if identity==unit else None),
        reference={'uri':'/fixture/replayed-region.json','sha256':'1'*64,
            'size_bytes':1,'schema':'scar.stateful-region'})

    routes,boundaries=derive_ordered_inputs(ordered)
    assert len(routes)==len(boundaries)==1
    route=routes[0]
    assert route.input_key=='#/inputs/0' and not route.field_path and not route.read_chain
    assert route.root_formal==T.GlobalPort(unit,ports[1])
    assert route.original_read_result==T.GlobalPort(unit,ports[4])
    assert {edge.kind for edge in route.reference_path.transitions}=={
        ReferenceTransitionKind.DECLARED_ALIAS,ReferenceTransitionKind.LEXICAL_MAY_BINDING}
    assert route.guards and not route.logical_content_version_equality_proven
    assert route.source_origin.locator=='#/inputs/0/origins/0'
    assert boundaries[0].input_key=='#/inputs/1'
    assert boundaries[0].reason=='REFERENCE_PATH_DOES_NOT_SELECT_THIS_OUTER_FORMAL_OR_REQUIRES_INLINE_CALLER_BINDING'
    assert decode(R.OrderedInputRoute,encode(route))==route
    assert checked.count('ordered')==checked.count('source')==2
