"""Local direct uses, immutable snapshots and exit content remain distinct."""
from types import SimpleNamespace as NS

from scar.analysis.ordered_region_lifetime import OrderedRegionLifetimeProvider
from scar.analysis.stateful_dependencies import project_stateful_dependencies
from scar.compiler.model import Expr,SourceRef
from scar.compiler.stateful_model import StatefulInput,StatefulAction,StatefulRegion,REQUIRED_GUARDS
from scar.ir.program import ordered_region_activation as A,ordered_region_lifetime as L
from scar.ir.program.dependencies import GraphNamespace,DependencyPoint as Point,DependencyStrength,DependencyMeaning
from scar.ir.program.ids import RegionID,ValueID
from scar.ir.program.providers import ProviderArtifact
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.record_codec import decode,encode


def test_read_lifetime_does_not_confuse_full_overwrite_or_cross_call_identity():
    source=SourceRef('/fixture/original.py','0'*64,'step',1,0,'pycode-v2:'+'1'*64)
    inputs=tuple(StatefulInput(name,('owner',name),kind)for name,kind in
        (('written','array'),('cleared','array'),('fixed','array'),('scale','scalar')))
    nodes=(Expr('a','input',attrs=dict(name='written',kind='array',epoch=0)),
        Expr('b','input',attrs=dict(name='fixed',kind='array',epoch=0)),
        Expr('s','input',attrs=dict(name='scale',kind='scalar',epoch=None)),
        Expr('sum','add',('a','b')),Expr('saved','mul',('sum','s')),
        Expr('after','input',attrs=dict(name='written',kind='array',epoch=1)),
        Expr('next','mul',('after','s')),Expr('zero','const',attrs=dict(value=0)))
    actions=(StatefulAction('sum','evaluate','sum'),StatefulAction('saved','evaluate','saved'),
        StatefulAction('update','write','saved','written',((1,-1),(1,-1)),0,1,'add'),
        StatefulAction('next','evaluate','next'),
        StatefulAction('clear','write','zero','cleared',None,0,1,'assign'))
    region=StatefulRegion('region',source,('owner',),inputs,nodes,actions,guards=REQUIRED_GUARDS)
    projection=project_stateful_dependencies(region)
    artifact=ProviderArtifact('/fixture/region.json','2'*64,1,'scar.stateful-array-region')
    locator=SourceArtifactLocator(artifact,artifact.sha256,artifact.size_bytes,'canonical-json')
    namespace=GraphNamespace(locator,'fixture',RegionID('region'))
    points={row['key']:Point(namespace,ValueID(row['key']),None)for row in projection['nodes']}
    ordered=NS(_region=region,_projection=projection,reference=dict(uri=artifact.uri,
        sha256=artifact.sha256,size_bytes=artifact.size_bytes,schema=artifact.schema),point=points.__getitem__,
        namespace=namespace,locator=locator,entry=Point(namespace,namespace.root,None))
    activation=NS(check=lambda:None,ordered=ordered,view=NS(),
        inputs=tuple(NS(input_key='#/inputs/'+str(i))for i in range(len(inputs))),
        ports={A.OrderedBoundaryRole.WORLD_EXIT:Point(namespace,ValueID('world-exit'),None)})
    provider=OrderedRegionLifetimeProvider(activation,register=False)
    written,cleared,fixed,scalar=provider.inputs
    assert written.entry_content_reaches_exit is True
    assert cleared.entry_content_reaches_exit is False and cleared.first_read_action is None
    assert fixed.stability is L.OrderedInputStability.ARRAY_CONTENT_PRESERVED
    assert len(fixed.epoch_ordinals)==1 and fixed.entry_content_reaches_exit is True
    assert scalar.stability is L.OrderedInputStability.SELECTED_SCALAR_VALUE_FIXED
    assert scalar.first_read_action==1 and scalar.last_read_action==3 and not scalar.write_actions
    before,after=[provider.epochs[i]for i in written.epoch_ordinals]
    assert before.last_read_action==before.superseded_at_action==2
    assert after.producer_action==2 and after.last_read_action==3 and after.required_at_normal_exit
    assert provider.epochs[cleared.epoch_ordinals[-1]].required_at_normal_exit
    assert all(use.action.artifact==artifact for use in provider.uses)
    assert decode(L.OrderedInputLifetime,encode(written))==written
    assert not provider.summary()['cross_call_content_identity_proven']
    assert not provider.summary()['external_storage_release_authorized']
    for source,target,_,_ in provider._edges:
        assert source!=target
        for edge in provider.outgoing(source).edges:
            assert edge in provider.incoming(edge.target).edges
            assert edge.strength is DependencyStrength.MAY and edge.conditions
            assert edge.meaning is DependencyMeaning.DEPENDENCE
