"""Public dependency precision; real source replay is exercised by the CPU case."""
from dataclasses import replace
from types import SimpleNamespace as NS
import json

import pytest

from scar.analysis.program_view import ProgramView
from scar.analysis.stateful_dependency_view import StatefulDependencyProvider,OrderedContentEpoch
from scar.compiler.model import SourceRef,Expr
from scar.compiler.stateful_model import StatefulInput,StatefulAction,StatefulRegion,REQUIRED_GUARDS,document
from scar.ir.program.dependencies import GraphNamespace,DependencyPoint as Point,DependencyKind as K,DependencyStrength as S
from scar.ir.program.ids import RegionID
from scar.ir.program.providers import ProviderArtifact
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.program.source_template_links import SourceTemplateKey
from scar.ir.program.source_productions import ProductionPurpose,ProductionRegionID


def test_ordered_epochs_stay_distinct_from_descriptors_and_require_original_replay(tmp_path,monkeypatch):
    source=SourceRef('/original.py','0'*64,'step',1,0,'pycode-v2:'+'1'*64,
        origins=({'kind':'source_identity','unit':'original','occurrence':0},))
    nodes=(Expr('x0','input',attrs=dict(name='x',kind='array',epoch=0)),
        Expr('v','slice',('x0',),dict(bounds=((1,-1),(1,-1)),semantics='ALIAS_DESCRIPTOR')),
        Expr('one','const',attrs=dict(value=1)),Expr('sum','add',('v','one')),
        Expr('x1','input',attrs=dict(name='x',kind='array',epoch=1)),Expr('next','add',('x1','one')))
    actions=(StatefulAction('a','evaluate','sum'),
        StatefulAction('w','write','sum','x',((1,-1),(1,-1)),0,1,'assign'),
        StatefulAction('b','evaluate','next'),StatefulAction('z','write','next','x',None,1,2,'assign'))
    region=StatefulRegion('region',source,('x',),(StatefulInput('x',('x',),'array'),),nodes,actions,guards=REQUIRED_GUARDS)
    path=tmp_path/'region.json';path.write_text(json.dumps(document(region)))
    monkeypatch.setattr('scar.compiler.stateful_frontend.discover_stateful_regions',lambda view,source:([region],[]))
    view=ProgramView(None)
    artifact=ProviderArtifact('/original.json','0'*64,1,'fixture')
    locator=SourceArtifactLocator(artifact,'0'*64,1,'canonical-json')
    namespace=GraphNamespace(locator,'fixture',RegionID('root'))
    function=NS(template=SourceTemplateKey(0,ProductionPurpose.STATEMENT),body=ProductionRegionID('body'))
    original=NS(unit='original',facts=NS(functions=[function]),check=lambda *args:None)
    monkeypatch.setattr(view,'unit',lambda unit:original)
    monkeypatch.setattr(view,'source_region_ref',lambda *args:Point(namespace,RegionID('body'),None))
    provider=StatefulDependencyProvider(view,path)
    epochs={r.epoch:provider.point(row['key'])for row in provider._projection['nodes']
        if row['kind']=='ORDERED_ARRAY_CONTENT_VERSION'
        for r in (provider.original(provider.point(row['key'])),)}
    assert len(epochs)==3 and all(type(view.original(p))is OrderedContentEpoch for p in epochs.values())
    partial=provider.point('#/actions/1');full=provider.point('#/actions/3')
    partial_reads=view.dependencies.incoming(partial,kinds=(K.STATE_INPUT,)).edges
    full_reads=view.dependencies.incoming(full,kinds=(K.STATE_INPUT,)).edges
    assert any(e.source==epochs[0]for e in partial_reads)
    assert not any(e.source==epochs[1]for e in full_reads)
    next_reads=view.dependencies.incoming(provider.point('#/actions/2'),kinds=(K.STATE_INPUT,)).edges
    assert any(e.source==epochs[1]for e in next_reads)
    assert all(e.strength is S.MAY and e.conditions for e in(*partial_reads,*next_reads))
    assert len(view.expand(provider.entry))==4
    selected=provider.outgoing(replace(provider.source_body,context=()))
    assert not selected.edges and any(f.kind=='ORDERED_REGION_ACTIVATION_INTERFACE_NOT_ATTACHED'for f in selected.frontiers)
    definition=provider.outgoing(provider.source_body).edges
    assert definition and all(e in provider.incoming(e.target).edges for e in definition)
    copied=view.original(provider.point('#/nodes/2'));copied.attrs['value']=99
    assert view.original(provider.point('#/nodes/2')).attrs['value']==1
    assert not provider.summary()['global_logical_versions_proven']
    altered=document(region);altered['nodes'][2]['attrs']['value']=2
    other=tmp_path/'changed.json';other.write_text(json.dumps(altered))
    before=len(view.dependencies.providers)
    with pytest.raises(ValueError,match='original static source derivation'):
        StatefulDependencyProvider(view,other)
    assert len(view.dependencies.providers)==before
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='changed during the query session'):provider.check()
