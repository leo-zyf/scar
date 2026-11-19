"""Bounded observed-target metadata on original F4 CALL operations.

Provider handles identify retained attempt records, never callable instances,
new library bodies, native implementations or completed Python invocations.
"""
from collections import defaultdict
from dataclasses import dataclass,replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace as NS

from scar.analysis.program_view import _digest
from scar.analysis.source_call_view import source_call_site,prepare_source_ports
from scar.analysis.source_reference_slice import source_port
from scar.compiler.call_binding import read_call_binding_plan,plan_call_bindings,join_call_bindings,plan_digest
from scar.ir.program import view as V
from scar.ir.program.call_edges import CapturedSite,CapturedCode
from scar.ir.program.normal_targets import GlobalPort
from scar.ir.program.ids import ProviderID
from scar.ir.program.providers import ProviderArtifact,ProviderFactReference
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.program.dependencies import (DependencyPoint as Point,DependencyEdge as Edge,
    DependencyKind as K,DependencyMeaning as Meaning,DependencyStrength as Strength,
    DependencyCondition as Condition,DependencyEvidence as Evidence,DependencyFrontier as Frontier,
    DependencyNeighborhood as Neighborhood)
from scar.ir.record_codec import decode,encode


SCHEMA='scar.source-bounded-call-targets'
_OPEN=('caller_invocation','callee_entry','callee_completion','argument_layout_and_defaults',
    'creator_environment_and_receiver_contents','array_content_versions','native_schema_and_effects',
    'unobserved_paths_and_future_targets')


@dataclass(frozen=True,slots=True)
class SourceObservedCallTarget:
    id: ProviderID
    source_call: GlobalPort
    caller_site: CapturedSite
    run: str
    process_id: int
    thread_id: int
    attempt_index: int
    target_kind: str
    target_code: CapturedCode | None
    metadata_json: str
    evidence: ProviderFactReference
    remaining_fields: tuple[str,...]=_OPEN

    def __post_init__(self):
        if (type(self.id)is not ProviderID or type(self.source_call)is not GlobalPort
                or type(self.caller_site)is not CapturedSite
                or type(self.run)is not str or len(self.run)!=64
                or type(self.process_id)is not int or self.process_id<1
                or type(self.thread_id)is not int or self.thread_id<0
                or type(self.attempt_index)is not int or self.attempt_index<1
                or self.target_kind not in ('PYTHON_FUNCTION','BOUND_PYTHON_METHOD','NATIVE_CARRIER','TYPE_OBJECT','OTHER_CALLABLE')
                or type(self.evidence)is not ProviderFactReference):
            raise ValueError('original bounded CALL attempt metadata required')
        self.caller_site.__post_init__();self.evidence.__post_init__()
        if (self.target_kind in ('PYTHON_FUNCTION','BOUND_PYTHON_METHOD'))!=(self.target_code is not None):
            raise ValueError('only observed Python function carriers have a CodeID')
        if self.target_code is not None:
            if type(self.target_code)is not CapturedCode:raise ValueError('exact target CodeID descriptor required')
            self.target_code.__post_init__()


class SourceCallBindingProvider:
    relation_kinds=frozenset((K.CALL_TARGET,K.IMPLEMENTATION))

    def __init__(self,view,plan_path,observation_paths):
        self.view=view;self.materials={};self.nodes={};self.by_call=defaultdict(list)
        self.out_edges=defaultdict(list);self.in_edges=defaultdict(list);self.operations={};self.sources={}
        self.body_targets={};self.body_frontiers={};self.body_keys=set()
        from scar.analysis.retained_code_bodies import RetainedCodeBodyIndex
        self.body_index=RetainedCodeBodyIndex(view)
        self.function_interfaces=None
        self.plan_path=Path(plan_path).absolute();self.plan=read_call_binding_plan(self.plan_path)
        self._read(self.plan_path)
        catalog=self.plan['catalog'];catalog_path=view.catalog.path/'result.json'
        if (catalog.get('path')!=str(catalog_path) or catalog.get('sha256')!=sha256(catalog_path.read_bytes()).hexdigest()
                or catalog.get('bytes')!=catalog_path.stat().st_size):
            raise ValueError('bounded CALL plan uses another original source catalog')
        if isinstance(observation_paths,(str,Path)):observation_paths=(observation_paths,)
        documents=[];anchors={}
        for path in observation_paths:
            path=Path(path).absolute();raw=self._read(path);doc=json.loads(raw);documents.append(doc)
            artifact=ProviderArtifact(str(path),sha256(raw).hexdigest(),len(raw),doc['schema'])
            for index,row in enumerate(doc['records']):
                key=doc['process_id'],row['index']
                if key in anchors:raise ValueError('same bounded attempt occurs in multiple raw documents')
                anchors[key]=(artifact,index)
        joined=join_call_bindings(self.plan,documents)
        # Independently replay the existing F4/bytecode mapper. Observed JSON
        # is not permission to redirect an original call to another offset.
        expected={}
        bodies=() if self.plan['selector']is None else self.plan['selector']['bodies']
        path_by_code={body['code']['key']:body['source']['path'] for body in bodies}
        by_path=defaultdict(set)
        for row in self.plan['sites']:
            by_path[path_by_code[row['code_key']]].add(row['query'])
        budget=self.plan['max_targets_per_site']
        for path,queries in by_path.items():
            replay=plan_call_bindings(view,path,call_indices=queries,max_targets_per_site=budget)
            for row in replay['sites']:expected[row['id'],row['code_key'],row['offset']]=row
            if any(row['source']['path']==path and row['code']['qualname']=='<module>' for row in bodies):
                replay=plan_call_bindings(view,path,entry='<module>',call_indices=queries,max_targets_per_site=budget)
                for row in replay['sites']:expected[row['id'],row['code_key'],row['offset']]=row
        for row in self.plan['sites']:
            if expected.get((row['id'],row['code_key'],row['offset']))!=row:
                raise ValueError('bounded CALL plan differs from current original F4/source correspondence')
        sites={row['id']:row for row in self.plan['sites']}
        ports=[]
        for row in sites.values():
            facts=view.unit(row['unit']).facts;call=facts.calls[row['query']]
            if (call.callee,call.result,call.world,call.result_world)!=(row['callee_port'],row['result_port'],row['world_port'],row['result_world_port']):
                raise ValueError('bounded target changed original callee/result/world ports')
            ports.extend(source_port(facts,i) for i in (call.callee,call.result,call.world,call.result_world))
        prepare_source_ports(view,ports)
        for row in sites.values():
            source=view.unit(row['unit']);self.sources[source.unit]=source;facts=source.facts;call=facts.calls[row['query']]
            operation,_,_=source_call_site(view,NS(result=source_port(facts,call.result),callee=source_port(facts,call.callee)))
            if view.original(operation).id.wire!=row['original_step']:raise ValueError('bounded CALL operation differs')
            self.operations[row['id']]=operation
            self.by_call.setdefault((operation.namespace,operation.node),[])
        for binding in joined['bindings']:
            selected=sites[binding['site_id']];operation=self.operations[binding['site_id']]
            artifact,index=anchors[binding['process_id'],binding['attempt_index']]
            evidence=ProviderFactReference(artifact,'#/records/'+str(index))
            locator=SourceArtifactLocator(artifact,artifact.sha256,artifact.size_bytes,'canonical-json',evidence.locator)
            identity=ProviderID(_digest((SCHEMA,plan_digest(self.plan),artifact,evidence.locator)))
            metadata=binding['target'];code=None if metadata.get('code')is None else decode(CapturedCode,metadata['code'])
            source=self.sources[selected['unit']];call=source.facts.calls[selected['query']]
            record=SourceObservedCallTarget(identity,source_port(source.facts,call.result),decode(CapturedSite,selected['site']),
                artifact.sha256,binding['process_id'],binding['thread_id'],binding['attempt_index'],metadata['kind'],code,
                json.dumps(metadata,sort_keys=True,separators=(',',':'),allow_nan=False),evidence)
            point=Point(operation.namespace,identity,None);self.nodes[point.namespace,identity]=(point,operation,record)
            self.by_call[operation.namespace,operation.node].append(point)
            guards=('RETAINED_CALL_ATTEMPT_METADATA_SELECTED','RUN_SHA256_'+record.run,
                'RECORDED_PROCESS_'+str(record.process_id),'RECORDED_THREAD_'+str(record.thread_id),
                'NO_CALLER_INVOCATION_ENTRY_COMPLETION_OR_FUTURE_DISPATCH_PROOF')
            edge=Edge(operation,point,K.CALL_TARGET,Meaning.DEPENDENCE,
                (Evidence(SCHEMA+'.attempt',(locator,source.locator),(operation,point)),),
                tuple(Condition(SCHEMA,guard,point,index=record.attempt_index) for guard in guards),Strength.MAY)
            self.out_edges[operation.namespace,operation.node].append(edge);self.in_edges[point.namespace,point.node].append(edge)
            body,reason=self._target_body(record.target_code)
            self.body_targets[point.namespace,point.node]=body
            if reason is not None:self.body_frontiers[point.namespace,point.node]=reason
            if body is not None:
                # This expands the observed implementation's static body. It
                # does not invent an invocation, formal binding or return edge.
                original_source=self.view.namespaces[body.namespace]
                locators=(locator,original_source.locator)
                implementation=Edge(point,body,K.IMPLEMENTATION,Meaning.STRUCTURE,
                    (Evidence(SCHEMA+'.original-code-body',locators,(point,body)),),
                    tuple(Condition(SCHEMA,guard,point,index=record.attempt_index) for guard in
                        ('EXACT_RETAINED_CODE_DESCRIPTOR_MATCH','SOURCE_BODY_IS_NOT_ACTIVATION',
                         'CREATOR_ENVIRONMENT_ARGUMENT_BINDING_AND_NATIVE_EFFECTS_REMAIN_OPEN')),Strength.MAY)
                self.out_edges[point.namespace,point.node].append(implementation)
                self.in_edges[body.namespace,body.node].append(implementation)
                self.body_keys.add((body.namespace,body.node))
        self.summary=joined;self.check();view.register(self)

    def attach_function_interfaces(self):
        """Project conditional source ABIs; an attempt is never an invocation.

        The existing function provider owns signature/default/capture and
        balanced ENTER/RETURN semantics. This metadata provider continues to
        retain the complete, bounded attempts independently of that view.
        """
        if self.function_interfaces is None:
            from scar.analysis.bounded_function_interfaces import BoundedFunctionInterfaceStore
            from scar.analysis.source_call_view import SourceFunctionCallProvider
            self.function_interfaces=SourceFunctionCallProvider(self.view,BoundedFunctionInterfaceStore(self))
        return self.function_interfaces

    def _target_body(self,code):
        """Join original compiler paths, including retained replay filenames."""
        if code is None:return None,None
        result=self.body_index.match(code)
        self.sources.update(self.body_index.sources)
        return result

    def _read(self,path):
        if path.is_symlink() or not path.is_file() or not 0<path.stat().st_size<=2_000_000:
            raise ValueError('bounded regular original observation/plan required')
        raw=path.read_bytes();self.materials[path]=raw;return raw

    def check(self):
        self.body_index.check()
        if any(path.read_bytes()!=raw for path,raw in self.materials.items()):
            raise ValueError('original bounded CALL materials changed')
        if json.loads(self.materials[self.plan_path])!=self.plan:
            raise ValueError('original bounded CALL plan changed in memory')
        for source in self.sources.values():source.check()

    def describe(self,point):
        value=self.nodes.get((point.namespace,point.node))
        if value is None:return None
        self.check();_,owner,_=value
        return V.ProgramNode(point,None,V.ProgramNodeRole.PROVIDER,replace(owner,context=point.context),None)

    def original(self,point):
        value=self.nodes.get((point.namespace,point.node))
        if value is None:return None
        self.check();return value[2]

    def expand(self,point):return ()

    def _neighbors(self,point,incoming):
        key=point.namespace,point.node
        if key not in self.nodes and key not in self.by_call and key not in self.body_keys:return Neighborhood()
        self.check()
        if point.context is not None:
            return Neighborhood((),(Frontier(point,'BOUNDED_CALL_CONTEXT_NOT_BOUND_TO_ATTEMPT',
                'A static DependencyFrame is not the recorded caller InvocationKey.'),))
        edges=(self.in_edges if incoming else self.out_edges).get(key,())
        if key in self.body_keys:return Neighborhood(tuple(edges))
        frontiers=tuple(Frontier(point,'BOUNDED_CALL_FIELD_OPEN',field) for field in _OPEN)
        if key in self.body_frontiers:
            frontiers+=(Frontier(point,'BOUNDED_CALL_BODY_OPEN',self.body_frontiers[key]),)
        if key in self.by_call and not self.by_call[key]:
            frontiers+=(Frontier(point,'SELECTED_CALL_TARGET_NOT_OBSERVED','No retained attempt supplies this selected site.'),)
        return Neighborhood(tuple(edges),frontiers)

    def incoming(self,point):return self._neighbors(point,True)
    def outgoing(self,point):return self._neighbors(point,False)

    def query(self,call):
        self.check();key=call.namespace,call.node
        return dict(schema=SCHEMA,call=encode(call),targets=[dict(point=encode(p),record=encode(self.original(p)),
            metadata=json.loads(self.original(p).metadata_json),
            source_body=encode(self.body_targets.get((p.namespace,p.node))),
            source_body_frontier=self.body_frontiers.get((p.namespace,p.node)))
            for p in self.by_call.get(key,())] if call.context is None else [],
            remaining_fields=list(_OPEN),static_context_bound_to_recorded_attempt=False,
            conditional_source_interfaces_attached=self.function_interfaces is not None,first_stage_complete=False)

    def explain(self,point):
        record=self.original(point)
        return dict(original=record,remaining_fields=list(_OPEN),
            code_correspondence=None if record.target_code is None else self.body_index.explain(record.target_code),
            entry_completion_or_effects_proven=False)


__all__=['SourceObservedCallTarget','SourceCallBindingProvider']
