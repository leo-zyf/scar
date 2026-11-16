"""Expose a source-replayed ordered array region in the public program graph.

The existing StatefulRegion and dependency projection remain the authorities.
An epoch is local to one guarded region activation, not a physical allocation
or a globally established LogicalVersion. No arrays are captured or executed.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
import argparse
import json
import time

from scar.analysis.stateful_dependencies import project_stateful_dependencies
from scar.analysis.source_consumer_parents import _stamp
from scar.compiler.stateful_model import SCHEMA, stateful_from_dict, stateful_digest
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program import view as V, source_productions as P
from scar.ir.program.ids import RegionID, OperationID, ValueID
from scar.ir.program.providers import ProviderArtifact
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.program.source_template_links import SourceTemplateKey
from scar.ir.program.dependencies import (GraphNamespace, DependencyPoint as Point,
    DependencyEdge as Edge, DependencyEvidence as Evidence, DependencyCondition as Condition,
    DependencyKind as K, DependencyMeaning as Meaning, DependencyStrength as Strength,
    DependencyFrontier as Frontier, DependencyNeighborhood as Neighborhood)
from scar.ir.record_codec import decode


REPRESENTATION='scar.stateful-ordered-content.v1'
_KINDS={
    'INPUT_PORT':(ValueID,V.ProgramEntityKind.VALUE,V.ProgramNodeRole.PARAMETER),
    'ALIAS_DESCRIPTOR':(ValueID,V.ProgramEntityKind.VALUE,V.ProgramNodeRole.CORE_VALUE),
    'LITERAL':(ValueID,V.ProgramEntityKind.VALUE,V.ProgramNodeRole.CORE_VALUE),
    'IMMUTABLE_EVALUATED_VALUE':(ValueID,V.ProgramEntityKind.VALUE,V.ProgramNodeRole.RESULT),
    'ORDERED_ARRAY_CONTENT_VERSION':(ValueID,V.ProgramEntityKind.VALUE,V.ProgramNodeRole.CORE_VALUE),
    'EVALUATE':(OperationID,V.ProgramEntityKind.OPERATION,V.ProgramNodeRole.OPERATION),
    'ORDERED_ARRAY_WRITE':(OperationID,V.ProgramEntityKind.OPERATION,V.ProgramNodeRole.OPERATION),
}
_EDGES={
    'STORAGE_REFERENCE':(K.STATE_LOCATION,Meaning.STATE_INTERFACE),
    'SCALAR_INPUT_VALUE':(K.VALUE_USE,Meaning.VALUE),
    'ALIAS_DESCRIPTOR_SOURCE':(K.ALIAS,Meaning.REFERENCE),
    'OPERAND_VALUE':(K.OPERAND,Meaning.DEPENDENCE),
    'OPERAND_DESCRIPTOR':(K.OPERAND,Meaning.DEPENDENCE),
    'ARRAY_CONTENT_READ':(K.STATE_INPUT,Meaning.DEPENDENCE),
    'ORIGINAL_ACTION_ORDER':(K.ORDER,Meaning.ORDER),
    'PRODUCES_IMMUTABLE_VALUE':(K.RESULT,Meaning.VALUE),
    'WRITE_DESTINATION_REFERENCE':(K.STATE_LOCATION,Meaning.STATE_INTERFACE),
    'READ_MODIFY_WRITE':(K.STATE_INPUT,Meaning.DEPENDENCE),
    'PRESERVES_UNWRITTEN_REGION':(K.STATE_INPUT,Meaning.DEPENDENCE),
    'PRODUCES_ORDERED_CONTENT_VERSION':(K.STATE_OUTPUT,Meaning.DEPENDENCE),
}


@dataclass(frozen=True,slots=True)
class OrderedContentEpoch:
    """Input/epoch coordinates, qualified by the provider's original region."""
    input_id:str
    epoch:int
    region_entry:bool


def _digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class StatefulDependencyProvider:
    """Replay before registering; query original records without a second IR.

    Creation uses the existing source extractor on this same ProgramView and
    compares the complete region, including all provenance. This is static
    compile-only analysis. Profile guards and caller recurrence stay open.
    """
    relation_kinds=frozenset((*[kind for kind,_ in _EDGES.values()],K.CONTAINS,K.IMPLEMENTATION))

    def __init__(self,view,region,*,register=True):
        from scar.compiler.stateful_frontend import discover_stateful_regions
        self.view=view
        self._reference=artifact_reference(region,SCHEMA)
        self._region=stateful_from_dict(read_artifact(self._reference))
        self.digest=stateful_digest(self._region)
        matches,_=discover_stateful_regions(view,source=self._region.source.path)
        if len([r for r in matches if stateful_digest(r)==self.digest])!=1:
            raise ValueError('ordered content region differs from the original static source derivation')
        self._projection=project_stateful_dependencies(self._region)
        # Pin the input before and after source replay. The retained original
        # source is compile-only; current runtime source identity is a guard.
        read_artifact(self._reference,json_document=False)
        self._region_path=Path(self._reference['uri'])
        self._region_stamp=_stamp(self._region_path)
        artifact=ProviderArtifact(self._reference['uri'],self._reference['sha256'],
            self._reference['size_bytes'],self._reference['schema'])
        self.locator=SourceArtifactLocator(artifact,artifact.sha256,artifact.size_bytes,'canonical-json')
        root=RegionID(_digest([REPRESENTATION,self.digest,'region']))
        self.namespace=GraphNamespace(self.locator,REPRESENTATION,root)
        self.owned_namespaces=(self.namespace,)
        self.entry=Point(self.namespace,root,None)
        self.points={};self._records={root:self._region};self.rows={};self.nodes={}
        self.in_edges=defaultdict(list);self.out_edges=defaultdict(list)
        self._original_edges={};self._sources={}
        self.nodes[root]=V.ProgramNode(self.entry,V.ProgramEntityKind.REGION,V.ProgramNodeRole.REGION,None,None)
        self._assemble()
        self.check()
        if register:view.register(self)

    def point(self,key):return self.points[key]

    @property
    def region(self):return deepcopy(self._region)

    @property
    def reference(self):return dict(self._reference)

    @property
    def projection(self):return deepcopy(self._projection)

    @staticmethod
    def _key(point):return point.namespace,point.node

    def _edge(self,source,target,kind,meaning,rule,*,strength=Strength.MAY,original=None):
        conditions=()if kind is K.CONTAINS else (Condition(REPRESENTATION,
            'ORIGINAL_REGION_PROFILE_GUARDS_AND_SAME_ACTIVATION_SELECTED',self.entry,None,(source,target)),)
        evidence=Evidence(REPRESENTATION+'.'+rule,(self.locator,),(source,target))
        edge=Edge(source,target,kind,meaning,(evidence,),conditions,strength)
        self.out_edges[self._key(source)].append(edge);self.in_edges[self._key(target)].append(edge)
        if original is not None:self._original_edges[edge]=original
        return edge

    def _assemble(self):
        for row in self._projection['nodes']:
            cls,kind,role=_KINDS[row['kind']]
            point=Point(self.namespace,cls(_digest([self.digest,row['key']])),None)
            self.points[row['key']]=point;self.rows[point.node]=row
            self.nodes[point.node]=V.ProgramNode(point,kind,role,self.entry,None)
            key=row['key']
            if key.startswith('#/inputs/'):record=self._region.inputs[int(key.rsplit('/',1)[1])]
            elif key.startswith('#/nodes/'):record=self._region.nodes[int(key.rsplit('/',1)[1])]
            elif key.startswith('#/actions/'):record=self._region.actions[int(key.rsplit('/',1)[1])]
            else:
                record=OrderedContentEpoch(row['input_id'],row['epoch'],row['region_entry'])
            self._records[point.node]=record
            self._edge(self.entry,point,K.CONTAINS,Meaning.STRUCTURE,'original-record',strength=Strength.EXACT)
        for index,row in enumerate(self._projection['edges']):
            kind,meaning=_EDGES[row['kind']]
            self._edge(self.points[row['source']],self.points[row['target']],kind,meaning,
                'dependency/'+str(index),original=row)
        # F4 operations and value ports stay in their original namespace.
        # IMPLEMENTATION denotes a checked lowering, not reference equality
        # or proof that the currently executing instance meets its guards.
        source_identity=next(row for row in self._region.source.origins if row.get('kind')=='source_identity')
        source=self.view.unit(source_identity['unit'])
        self._sources[source.unit]=source
        functions=[f for f in source.facts.functions if f.template.occurrence==source_identity['occurrence']]
        if len(functions)!=1:raise ValueError('ordered region lacks its one original function body')
        function=functions[0]
        body=self.view.source_region_ref(source.unit,function.template,function.body)
        self.source_body=body
        self._edge(body,self.entry,K.IMPLEMENTATION,Meaning.STRUCTURE,'original-source-body')
        origins=defaultdict(list)
        for origin in self._region.source.origins:
            if origin.get('kind')=='original_f4' and origin.get('expression'):
                origins[origin['expression']].append(origin)
        for index,node in enumerate(self._region.nodes):
            for origin in origins[node.id]:
                # A fully replayed expression can inline another leaf. Its
                # operation origin is retained, but not mislabeled as an
                # equal reference to an arbitrary normal-completion port.
                self._origin_edge(origin,self.points['#/nodes/'+str(index)])
        for index,action in enumerate(self._region.actions):
            for origin in action.origins:self._origin_edge(origin,self.points['#/actions/'+str(index)])

    def _origin_edge(self,origin,target):
        if origin.get('kind')!='original_f4' or 'step'not in origin:return
        template=decode(SourceTemplateKey,origin['template'])
        prefix=P.ProductionStepID.prefix+':'
        wire=origin['step']
        if not wire.startswith(prefix):raise ValueError('original stateful step has another ID domain')
        source=self.view.unit(origin['unit'])
        self._sources[source.unit]=source
        point=source._node(template,P.ProductionStepID(wire[len(prefix):]))
        source.ensure(point)
        self._edge(point,target,K.IMPLEMENTATION,Meaning.STRUCTURE,'original-source-operation')

    def check(self,point=None):
        if _stamp(self._region_path)!=self._region_stamp:
            raise ValueError('admitted ordered region changed during the query session')
        for source in self._sources.values():source.check()

    def describe(self,point):
        if point.namespace!=self.namespace or point.node not in self.nodes:return None
        self.check(point);node=self.nodes[point.node]
        return replace(node,point=point,owner=None if node.owner is None else replace(node.owner,context=point.context))

    def original(self,point):
        if self.describe(point)is None:raise ValueError('not an original ordered-region endpoint')
        return deepcopy(self._records[point.node])

    def expand(self,point):
        self.original(point)
        if point.node==self.entry.node:
            return tuple(replace(self.points['#/actions/'+str(i)],context=point.context)for i in range(len(self._region.actions)))
        return tuple(replace(edge.target,context=point.context)for edge in self.out_edges[self._key(point)]
            if edge.kind is K.RESULT)

    def explain(self,point):
        return dict(node=self.describe(point),original=self.original(point),
            projection=deepcopy(self.rows.get(point.node)),required_profile_guards=self._region.guards,
            source_replayed=True,epochs_scope='ONE_GUARDED_REGION_ACTIVATION',
            global_logical_versions_proven=False,physical_storage_identity_proven=False,
            caller_loop_recurrence_proven=False,transformation_authorized=False)

    def _neighbors(self,point,incoming):
        rows=(self.in_edges if incoming else self.out_edges).get(self._key(point),())
        if not rows and point.namespace!=self.namespace:return Neighborhood()
        self.check(point)
        if point.context is not None:
            activation=getattr(self,'activation_provider',None)
            if activation is not None and activation.accepts_context(point):return Neighborhood()
            # No interface maps a selected F4 call stack to this abstract
            # ordered activation yet. Contextualizing just one endpoint
            # would create a different edge in the reverse neighborhood.
            return Neighborhood((),(Frontier(point,'ORDERED_REGION_ACTIVATION_INTERFACE_NOT_ATTACHED',
                'A caller selection cannot be erased or copied into a distinct ordered-region namespace; only unselected definition links are available.'),))
        edges=tuple(rows)
        frontiers=(Frontier(point,'ORDERED_REGION_PROFILE_AND_CALLER_ACTIVATION_NOT_DISCHARGED',
            'Static source replay proves the local derivation; alias/scalar/provider guards and caller recurrence remain separate.'),)
        return Neighborhood(edges,frontiers)

    def incoming(self,point):return self._neighbors(point,True)
    def outgoing(self,point):return self._neighbors(point,False)

    def summary(self):
        return dict(schema=REPRESENTATION,original_region=self.reference,nodes=len(self.nodes),
            relations=sum(map(len,self.out_edges.values())),
            kinds=dict(Counter(row['kind']for row in self._projection['nodes'])),
            dependency_kinds=dict(Counter(row['kind']for row in self._projection['edges'])),
            source_derivation_replayed=True,global_logical_versions_proven=False,
            caller_loop_recurrence_proven=False,target_executed=False,transformation_authorized=False)


def _pins():
    from scar.analysis import stateful_dependencies
    from scar.analysis import program_view,normal_output
    from scar.compiler import stateful_model,stateful_frontend,graph_frontend
    from scar.ir.program import dependencies
    from scar.ir import record_codec
    return {name:sha256(Path(module.__file__).read_bytes()).hexdigest()for name,module in (
        ('adapter',__import__(__name__,fromlist=['__file__'])),('projection',stateful_dependencies),
        ('region_model',stateful_model),('extractor',stateful_frontend),('source_lowering',graph_frontend),
        ('public_view',program_view),('public_types',V),('relations',dependencies),
        ('wire_codec',record_codec),('table_writer',normal_output))}


def publish_stateful_graph(provider,out):
    """Store the typed projection and original references, not body snapshots."""
    from scar.analysis.normal_output import write_table
    from scar.ir.record_codec import encode
    out=Path(out).resolve()
    if out.exists():raise ValueError('fresh ordered-content graph output required')
    provider.check();pins=_pins();out.mkdir(parents=True)
    nodes=tuple(provider.nodes.values())
    edges=tuple(edge for rows in provider.out_edges.values()for edge in rows)
    tables=dict(nodes=write_table(out/'nodes.jsonl.gz',nodes),
        edges=write_table(out/'edges.jsonl.gz',edges))
    # Typed nodes carry one original pointer/epoch coordinate; no source
    # bodies, tensor contents, repeated expressions or iterations are copied.
    refs=[dict(node=encode(provider.points[row['key']].node),original_key=row['key'])
        for row in provider._projection['nodes']]
    tables['references']=write_table(out/'references.jsonl.gz',refs)
    report=dict(schema='scar.stateful-public-graph',schema_version=1,pins=pins,
        original_region=provider.reference,entry=encode(provider.entry),
        source_body=encode(provider.source_body),summary=provider.summary(),tables=tables,
        target_executed=False,original_body_bytes_copied=0,loops_unrolled=False,
        runtime_guards_discharged=False,transformation_authorized=False)
    provider.check()
    if pins!=_pins():raise ValueError('ordered-content graph producer changed during publication')
    (out/'result.json').write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return report


def read_stateful_graph(path,view,*,_provider=None):
    """Check stored typed tables against the source-replayed public adapter."""
    from itertools import zip_longest
    from scar.analysis.source_symbol_variants import _table
    path=Path(path).resolve();raw=(path/'result.json').read_bytes();report=json.loads(raw)
    if (report.get('schema'),report.get('schema_version'))!=('scar.stateful-public-graph',1)or report['pins']!=_pins():
        raise ValueError('ordered-content graph producer/schema changed')
    provider=_provider if _provider is not None else StatefulDependencyProvider(view,report['original_region']['uri'],register=False)
    if (type(provider)is not StatefulDependencyProvider or provider.view is not view
            or provider.reference!=report['original_region'] or provider.summary()!=report['summary']):
        raise ValueError('ordered-content graph differs from original source projection')
    from scar.ir.record_codec import encode
    if (encode(provider.entry)!=report['entry'] or encode(provider.source_body)!=report['source_body']
            or any(report[k]is not False for k in('target_executed','loops_unrolled','runtime_guards_discharged','transformation_authorized'))
            or report['original_body_bytes_copied']!=0 or set(report['tables'])!={'nodes','edges','references'}):
        raise ValueError('ordered-content graph authority or table domain changed')
    expected=dict(nodes=tuple(provider.nodes.values()),edges=tuple(edge for rows in provider.out_edges.values()for edge in rows),
        references=tuple(dict(node=encode(provider.points[row['key']].node),original_key=row['key'])
            for row in provider._projection['nodes']))
    sentinel=object()
    for name,values in expected.items():
        cls=V.ProgramNode if name=='nodes'else Edge if name=='edges'else None
        stored=(decode(cls,row)if cls else row for row in _table(path,report['tables'][name]))
        if any(a!=b for a,b in zip_longest(stored,values,fillvalue=sentinel)):
            raise ValueError('stored ordered-content '+name+' differs from the original derivation')
    provider.check()
    if (path/'result.json').read_bytes()!=raw:raise ValueError('ordered-content graph changed during read')
    return report,provider


def main(argv=None):
    from scar.analysis.build import BuildSession
    from scar.compiler.stateful_frontend import discover_stateful_regions
    from scar.compiler.stateful_model import write_stateful
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',required=True)
    select=parser.add_mutually_exclusive_group()
    select.add_argument('--region',help='replay one retained original region')
    select.add_argument('--source',help='discover supported regions in this admitted source; default is the whole declared catalog')
    parser.add_argument('--out',required=True)
    args=parser.parse_args(argv);started=time.perf_counter()
    with BuildSession(args.build)as session:
        if args.region:
            provider=StatefulDependencyProvider(session.view,args.region)
            report=publish_stateful_graph(provider,args.out)
            read_stateful_graph(args.out,session.view,_provider=provider)
            result=dict(nodes=report['summary']['nodes'],relations=report['summary']['relations'],
                source_derivation_replayed=True,typed_tables_read_back=True,
                stored_table_bytes=sum(row['bytes']for row in report['tables'].values()),
                seconds=time.perf_counter()-started,target_executed=False)
        else:
            out=Path(args.out).resolve()
            if out.exists():raise ValueError('fresh ordered-content graph catalog output required')
            models,rejected=discover_stateful_regions(session.view,source=args.source)
            out.mkdir(parents=True);graphs=[]
            for index,region in enumerate(models):
                folder=out/('region-'+str(index));folder.mkdir()
                write_stateful(region,folder/'region.json')
                provider=StatefulDependencyProvider(session.view,folder/'region.json')
                report=publish_stateful_graph(provider,folder/'graph')
                read_stateful_graph(folder/'graph',session.view,_provider=provider)
                graphs.append(dict(graph=artifact_reference(folder/'graph/result.json',report['schema']),
                    summary=report['summary'],typed_tables_read_back=True,
                    stored_table_bytes=sum(row['bytes']for row in report['tables'].values())))
            result=dict(schema='scar.source-ordered-content-graphs',schema_version=1,
                graphs=graphs,rejected_profiles=rejected,source_filter=args.source,
                program=artifact_reference(session.path/'manifest.json','scar.program-build'),
                automatic_source_discovery=True,target_executed=False,loops_unrolled=False,
                first_stage_complete=False,seconds=time.perf_counter()-started)
            (out/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result))
    return 0


__all__=['StatefulDependencyProvider','OrderedContentEpoch','publish_stateful_graph','read_stateful_graph']


if __name__=='__main__':raise SystemExit(main())
