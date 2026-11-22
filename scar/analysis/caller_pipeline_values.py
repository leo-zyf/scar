"""Macro dataflow/content templates over one replayed caller PipelinePlan.

The existing PipelinePlan and pipeline-memory metadata own connectivity and
lifetimes. Shared optimized leaves remain in the original proof; this view
substitutes their formal content roots without cloning a source call tree.
"""
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from itertools import zip_longest
from pathlib import Path
import json

from scar.compiler.caller_pipeline import CallerPipelineProof,document as proof_document
from scar.compiler.pipeline_frontend import document as pipeline_document, validate_pipeline, pipeline_digest
from scar.compiler.array_copy_contract import COPY_REQUIRED_GUARDS, validate_copy_attributes
from scar.optimization.graph import ordered_nodes,canonical_value
from scar.optimization.pipeline_memory import _metadata
from scar.analysis.source_consumer_parents import _stamp
from scar.ir.program import caller_pipeline_values as R, layout_templates as A, view as V
from scar.ir.program import normal_graph as T, normal_graph as G
from scar.ir.program.ids import RegionID,OperationID,ValueID,UseID
from scar.ir.program.providers import ProviderFactReference,ProviderArtifact
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.program.dependencies import (GraphNamespace,DependencyPoint as Point,
    DependencyFrame,DependencyEdge as Edge,DependencyEvidence as Evidence,DependencyCondition as Condition,
    DependencyKind as K,DependencyMeaning as Meaning,DependencyStrength as Strength,
    DependencyNeighborhood as Neighborhood,DependencyFrontier as Frontier)
from scar.ir.execution_recipe import artifact_reference,read_artifact
from scar.ir.record_codec import encode,decode


SCHEMA='scar.caller-pipeline-values'
_CLOSED_OPS=frozenset(('input','const','slice','copy','neg','add','sub','mul','div'))


def _canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def _digest(value):return sha256(_canonical(value)).hexdigest()


def _artifact_stamps(document):
    """Guard only byte-validated source/proof receipts used by this graph."""
    receipts={};pending=[document]
    while pending:
        value=pending.pop()
        if type(value)is dict:
            if set(value)=={'uri','sha256','size_bytes','schema'}:
                path=Path(value['uri']);receipts[path]=_stamp(path)
            else:pending.extend(value.values())
        elif type(value)in(list,tuple):pending.extend(value)
    return receipts


def derive_caller_values(proof,*,input_shapes=None):
    """Project a typed proof structurally; publication still replays its source.

    Contents are functions of this caller's input-content variables. Different
    output value/materialization records remain distinct even when those
    functions are equal. World/environment routes never enter content hashes.
    """
    if type(proof)is not CallerPipelineProof:raise TypeError('typed caller pipeline proof required')
    pipeline=validate_pipeline(proof.pipeline);wire=pipeline_document(pipeline)
    metadata,_,declared,issues=_metadata(pipeline,{}if input_shapes is None else input_shapes)
    values={row['id']:row for row in metadata}
    scope=['one-caller-parametric-content',proof.id]
    content={value:_digest([scope,'input-content-variable',index])for index,value in enumerate(pipeline.input_values)}
    evidence=defaultdict(list)
    for index,row in enumerate(proof.call_evidence):evidence[row['pipeline_call']].append((index,row))
    calls=[];guards=list((*proof.required_runtime_guards,*pipeline.assumptions,
        'INPUT_CONTENTS_STABLE_THROUGH_ENTIRE_SELECTED_CLOSED_CALLER_ACTIVATION'))
    for index,call in enumerate(pipeline.calls):
        if len(evidence[call.id])!=1:raise ValueError('each macro CALL requires one original caller interface witness')
        ei,original=evidence[call.id][0];body=decode(T.Body,original['body'])
        parameters=dict(zip(call.plan.region.parameters,call.inputs,strict=True));classes={}
        for node in ordered_nodes(call.plan.region):
            if node.op not in _CLOSED_OPS:raise ValueError('macro content substitution requires closed original array operators')
            if node.op=='input':classes[node.id]=content[parameters[node.attrs['name']]]
            elif node.op=='copy':
                validate_copy_attributes(node.attrs)
                if not set(COPY_REQUIRED_GUARDS)<=set(call.plan.region.assumptions):
                    raise ValueError('copy content substitution lacks exact descriptor/source stability guards')
                classes[node.id]=classes[node.inputs[0]]
            else:classes[node.id]=_digest([scope,'ordered-primitive',node.op,
                [classes[value]for value in node.inputs],canonical_value(node.attrs)])
        content[call.output]=classes[call.plan.region.output]
        routes=decode(tuple[T.Route,...],original['parameters']);substitutions=[]
        for pi,(parameter,value)in enumerate(zip(call.plan.region.parameters,call.inputs,strict=True)):
            formal=T.GlobalPort(body.unit,G.Port(body.template,body.region,G.PortKind.PARAMETER,pi+1))
            choices=[r for r in routes if r.formal==formal and r.kind is T.RouteKind.ARGUMENT
                and len(r.actuals)==1 and type(r.actuals[0])is T.GlobalPort]
            if len(choices)!=1:raise ValueError('macro parameter lacks its exact original direct formal route')
            substitutions.append(R.CallerParameterContent(parameter,value,formal,choices[0].actuals[0],content[value]))
        calls.append(R.CallerDataCall(call.id,index,call.inputs,call.output,
            tuple(sorted({values[value]['producer_index']for value in call.inputs if values[value]['producer_index']>=0})),
            tuple(substitutions),decode(DependencyFrame,original['frame']),
            decode(ProviderFactReference,original['original_call']),decode(ProviderFactReference,original['interface_fact']),
            body,decode(T.GlobalPort,original['result']),decode(T.GlobalPort,original['world_before']),
            decode(T.GlobalPort,original['world_after']),'#/pipeline/leaf_plans/'+wire['calls'][index]['plan_id'],
            '#/call_evidence/'+str(ei)))
        guards.extend(call.plan.region.assumptions);guards.extend(call.plan.proof['runtime_guards'])
        guards.extend(call.plan.proof.get('legality',{}).get('external_assumptions',()))
    result=[]
    for row in metadata:
        initial=row['producer_index']<0
        pointer=('#/pipeline/input_values/'+str(pipeline.input_values.index(row['id']))if initial
            else '#/pipeline/calls/'+str(row['producer_index'])+'/output')
        result.append(R.CallerDataValue(row['id'],content[row['id']],
            A.ArrayMaterializationRole.INPUT_REFERENCE if initial else A.ArrayMaterializationRole.FRESH_ARRAY_RESULT,
            row['producer_index'],tuple(row['consumer_indices']),row['last_use'],row['ownership'],row['escape'],
            None if row['shape']is None else tuple(row['shape']),row['nbytes'],pointer))
    return R.CallerValueProjection(pipeline_digest(pipeline),tuple(result),tuple(calls),
        tuple(R.CallerValueFieldFrontier(row['field'],row['reason'])for row in issues),
        tuple(dict.fromkeys(guards)),declared)


class CallerPipelineValueProvider:
    """Definition-time macro ports; original CALL frames remain evidence only."""
    relation_kinds=frozenset((K.CONTAINS,K.ARGUMENT,K.OPERAND,K.RESULT,K.PROJECTION,K.IMPLEMENTATION,
        K.STATE_INPUT,K.STATE_OUTPUT,K.COMPLETION))

    def __init__(self,proof,reference,*,view,provider,input_shapes=None,register=True):
        self.proof=proof;self.reference=reference;self.view=view;self.source_calls=provider
        self._parents=_artifact_stamps([reference,proof_document(proof)])
        self.projection=derive_caller_values(proof,input_shapes=input_shapes)
        artifact=ProviderArtifact(**reference)
        self.locator=SourceArtifactLocator(artifact,artifact.sha256,artifact.size_bytes,'canonical-json')
        self.basis=_digest([reference,proof.id]);root=RegionID(self.basis)
        self.namespace=GraphNamespace(self.locator,SCHEMA,root);self.owned_namespaces=(self.namespace,)
        self.entry=Point(self.namespace,root,None);self.nodes={};self.records={};self.values={};self.calls={};self.shared={}
        self.in_edges=defaultdict(list);self.out_edges=defaultdict(list);self._sources={}
        self.nodes[root]=V.ProgramNode(self.entry,V.ProgramEntityKind.REGION,V.ProgramNodeRole.REGION,None,None)
        self.records[root]=self.projection;self._assemble();self.check()
        if register:self.view.register(self)

    @staticmethod
    def _key(point):return point.namespace,point.node

    def _edge(self,source,target,kind,meaning,rule,index=None):
        conditions=()if kind is K.CONTAINS else(Condition(SCHEMA,
            'ALL_CLOSED_CALLER_GUARDS_AND_SELECTED_ACTIVATION_INPUT_VERSIONS',self.entry),)
        if index is not None:conditions+=(Condition(SCHEMA,'ORIGINAL_PARAMETER_ORDINAL',target,index,(source,target)),)
        edge=Edge(source,target,kind,meaning,(Evidence(SCHEMA+'.'+rule,(self.locator,),(source,target)),),
            conditions,Strength.MAY if conditions else Strength.EXACT)
        self.out_edges[self._key(source)].append(edge);self.in_edges[self._key(target)].append(edge)

    def _source_port(self,point):
        self._sources[point.unit]=self.view.unit(point.unit)
        return self.view.source_port(point)

    def _assemble(self):
        for row in self.projection.values:
            point=Point(self.namespace,ValueID(_digest([self.basis,'value',row.id])),None);self.values[row.id]=point
            self.nodes[point.node]=V.ProgramNode(point,V.ProgramEntityKind.VALUE,
                V.ProgramNodeRole.PARAMETER if row.producer_index<0 else V.ProgramNodeRole.RESULT,self.entry,None)
            self.records[point.node]=row;self._edge(self.entry,point,K.CONTAINS,Meaning.STRUCTURE,'macro-value')
        for row in self.projection.calls:
            point=Point(self.namespace,OperationID(_digest([self.basis,'call',row.id])),None);self.calls[row.id]=point
            self.nodes[point.node]=V.ProgramNode(point,V.ProgramEntityKind.OPERATION,V.ProgramNodeRole.OPERATION,self.entry,None)
            self.records[point.node]=row;self._edge(self.entry,point,K.CONTAINS,Meaning.STRUCTURE,'macro-call')
            self._edge(Point(row.frame.caller,row.frame.operation,None),point,K.IMPLEMENTATION,Meaning.STRUCTURE,'original-call')
            body=self.view.source_region_ref(row.shared_body.unit,row.shared_body.template,row.shared_body.region)
            self.shared[row.id]=body;self._sources[row.shared_body.unit]=self.view.unit(row.shared_body.unit)
            self._edge(body,point,K.IMPLEMENTATION,Meaning.STRUCTURE,'shared-original-leaf-definition')
            for index,binding in enumerate(row.substitutions):
                use=Point(self.namespace,UseID(_digest([self.basis,'argument',row.id,index])),None)
                self.nodes[use.node]=V.ProgramNode(use,None,V.ProgramNodeRole.ARGUMENT,point,None);self.records[use.node]=binding
                self._edge(self.values[binding.value],use,K.ARGUMENT,Meaning.REFERENCE,'macro-actual',index)
                self._edge(self._source_port(binding.actual),use,K.PROJECTION,Meaning.REFERENCE,'original-actual',index)
                self._edge(use,point,K.OPERAND,Meaning.DEPENDENCE,'bound-parameter',index)
            self._edge(point,self.values[row.output],K.RESULT,Meaning.VALUE,'fresh-call-result')
            self._edge(self._source_port(row.source_result),self.values[row.output],K.PROJECTION,
                Meaning.REFERENCE,'original-normal-call-result')
            self._edge(self._source_port(row.world_before),point,K.STATE_INPUT,Meaning.STATE_INTERFACE,'original-pre-call-world')
            self._edge(point,self._source_port(row.world_after),K.STATE_OUTPUT,Meaning.STATE_INTERFACE,'original-normal-post-call-world')
        parameters=self.proof.caller_origin['caller_parameters'][1:-1]
        if len(parameters)!=len(self.proof.pipeline.input_values):raise ValueError('caller data formal/input arity differs')
        for formal,value in zip(parameters,self.proof.pipeline.input_values,strict=True):
            self._edge(self._source_port(decode(T.GlobalPort,formal)),self.values[value],K.ARGUMENT,Meaning.REFERENCE,'original-caller-parameter')
        self._edge(self.values[self.proof.pipeline.output],self._source_port(decode(T.GlobalPort,
            self.proof.caller_origin['caller_return'])),K.COMPLETION,Meaning.REFERENCE,'original-caller-return')

    def check(self):
        if any(_stamp(path)!=stamp for path,stamp in self._parents.items()):
            raise ValueError('original caller proof/interface source receipt changed during query')
        for source in self._sources.values():source.check()

    def describe(self,point):
        if point.namespace!=self.namespace or point.node not in self.nodes:return None
        self.check()
        return self.nodes[point.node]if point.context is None else None

    def original(self,point):
        if self.describe(point)is None:raise ValueError('not an unselected caller macro template endpoint')
        return deepcopy(self.records[point.node])

    def expand(self,point):
        self.original(point)
        if point==self.entry:return tuple(self.calls.values())
        if point.node in(self.calls[key].node for key in self.calls):
            call=self.records[point.node];return(self.shared[call.id],self.values[call.output])
        return ()

    def explain(self,point):
        return dict(node=self.describe(point),original=self.original(point),required_guards=self.projection.required_guards,
            scope=self.projection.scope,leaf_representation=self.projection.leaf_representation,
            runtime_logical_or_storage_identity_proven=False,cross_call_cache_authorized=False,
            transformation_authorized=False)

    def _neighbors(self,point,incoming):
        rows=(self.in_edges if incoming else self.out_edges).get(self._key(point),())
        if not rows:return Neighborhood()
        self.check()
        if point.context is not None:return Neighborhood(frontiers=(Frontier(point,
            'CALLER_VALUE_TEMPLATE_ACTIVATION_NOT_SELECTED',
            'Original F4 frames cannot be copied onto a distinct macro-template namespace.'),))
        return Neighborhood(tuple(rows))

    def incoming(self,point):return self._neighbors(point,True)
    def outgoing(self,point):return self._neighbors(point,False)

    def summary(self):
        return dict(calls=len(self.projection.calls),values=len(self.projection.values),nodes=len(self.nodes),
            relations=sum(map(len,self.out_edges.values())),
            content_templates=len({v.content_template for v in self.projection.values}),
            fresh_call_results=sum(v.producer_index>=0 for v in self.projection.values),
            field_frontiers=len(self.projection.frontiers),scope=self.projection.scope,
            leaf_representation=self.projection.leaf_representation,source_proof_replayed=True,
            runtime_logical_versions_created=0,physical_allocations_observed=0,source_bodies_copied=0,
            input_shapes=self.projection.input_shapes,target_executed=False,transformation_authorized=False)


def _pins():
    from scar.compiler import caller_pipeline, pipeline_frontend as pipeline_model, array_copy_contract as array_copy_contract
    from scar.optimization import graph, pipeline_memory as pipeline_memory, optimizer
    return {name:sha256(Path(module.__file__).read_bytes()).hexdigest()for name,module in(
        ('projection',__import__(__name__,fromlist=['__file__'])),('types',R),('source_proof',caller_pipeline),
        ('pipeline',pipeline_model),('copy_contract',array_copy_contract),('ordered_values',graph),
        ('optimizer',optimizer),('memory_metadata',pipeline_memory))}


def _rows(provider):
    return dict(nodes=(V.ProgramNode,tuple(provider.nodes.values())),values=(R.CallerDataValue,provider.projection.values),
        calls=(R.CallerDataCall,provider.projection.calls),frontiers=(R.CallerValueFieldFrontier,provider.projection.frontiers),
        edges=(Edge,tuple(edge for rows in provider.out_edges.values()for edge in rows)))


def _tables(path,report,provider):
    from scar.analysis.source_symbol_variants import _table
    expected=_rows(provider)
    if set(report['tables'])!=set(expected):raise ValueError('caller value table domain differs')
    absent=object();counts={}
    for name,(cls,rows)in expected.items():
        if report['tables'][name].get('path')!=name+'.jsonl.gz':raise ValueError('caller value table local path differs')
        count=0
        for wire,original in zip_longest(_table(path,report['tables'][name]),rows,fillvalue=absent):
            if wire is absent or original is absent or decode(cls,wire)!=original:
                raise ValueError('caller value '+name+' differs from original proof/dataflow replay')
            count+=1
        counts[name]=count
    return counts


def _graph(provider):
    for rows in provider.out_edges.values():
        for edge in rows:
            if(edge not in provider.view.dependencies.incoming(edge.target,kinds=(edge.kind,)).edges
                    or edge not in provider.view.dependencies.outgoing(edge.source,kinds=(edge.kind,)).edges):
                raise ValueError('caller macro value connection missing from public graph')
    return dict(bidirectional_missing=0)


def _replay(reference,view,provider,original=None):
    from scar.compiler.caller_pipeline import read_caller_pipeline_proof,SCHEMA as PROOF_SCHEMA
    if reference.get('schema')!=PROOF_SCHEMA:raise ValueError('original caller pipeline proof reference required')
    read_artifact(reference)
    replayed=read_caller_pipeline_proof(reference['uri'],view=view,provider=provider)
    if original is not None and _canonical(proof_document(original))!=_canonical(proof_document(replayed)):
        raise ValueError('provided caller proof differs from independently replayed original')
    return replayed


def publish_caller_values(proof_reference,out,*,view,provider,proof=None,input_shapes=None):
    """Return (report, replayed_proof); perform one original proof replay."""
    from scar.analysis.normal_output import write_table
    replayed=_replay(proof_reference,view,provider,proof);pins=_pins()
    values=CallerPipelineValueProvider(replayed,proof_reference,view=view,provider=provider,input_shapes=input_shapes)
    out=Path(out).resolve()
    if out.exists():raise ValueError('fresh caller value publication required')
    out.mkdir(parents=True)
    tables={name:write_table(out/(name+'.jsonl.gz'),rows)for name,(_,rows)in _rows(values).items()}
    report=dict(schema=SCHEMA,schema_version=1,original_proof=proof_reference,pins=pins,
        pipeline_sha256=values.projection.pipeline_sha256,input_shapes=values.projection.input_shapes,
        required_guards=list(values.projection.required_guards),tables=tables,summary=values.summary(),graph_validation=_graph(values),
        runtime_guards_discharged=False,transformation_authorized=False,source_bodies_copied=0,target_executed=False)
    _tables(out,report,values);values.check();provider.store.validate()
    read_artifact(proof_reference,json_document=False)
    if pins!=_pins():raise ValueError('caller value rules changed during publication')
    (out/'result.json').write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return report,replayed


def read_caller_values(path,*,view,provider):
    path=Path(path).resolve();path=path.parent if path.is_file()else path
    original=(path/'result.json').read_bytes();report=json.loads(original)
    if(report.get('schema'),report.get('schema_version'))!=(SCHEMA,1)or report['pins']!=_pins():
        raise ValueError('caller values schema/producing rules changed')
    proof=_replay(report['original_proof'],view,provider)
    values=CallerPipelineValueProvider(proof,report['original_proof'],view=view,provider=provider,
        input_shapes=report['input_shapes'],register=False)
    if(report['pipeline_sha256']!=values.projection.pipeline_sha256 or report['summary']!=values.summary()
            or report['required_guards']!=list(values.projection.required_guards)
            or report['runtime_guards_discharged']is not False or report['transformation_authorized']is not False
            or report['target_executed']is not False or type(report['source_bodies_copied'])is not int
            or report['source_bodies_copied']!=0):raise ValueError('caller values scope/guards/authority differs')
    _tables(path,report,values);view.register(values)
    if _graph(values)!=report['graph_validation']:raise ValueError('caller values public graph differs')
    values.check();provider.store.validate();read_artifact(report['original_proof'],json_document=False)
    if(path/'result.json').read_bytes()!=original or report['pins']!=_pins():raise ValueError('caller value publication changed during replay')
    return values


__all__=['derive_caller_values','publish_caller_values','read_caller_values','CallerPipelineValueProvider']
