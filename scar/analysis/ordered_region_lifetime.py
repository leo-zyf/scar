"""Derive local input lifetime/transfer facts from replayed ordered dataflow.

Last direct read, final destination-reference use and observable normal exit
are separate facts. None grants ownership, deallocation, an unguarded alias,
physical commit timing or equality with the next function invocation.
"""
from collections import Counter,defaultdict
from dataclasses import replace
from hashlib import sha256
from itertools import zip_longest
from pathlib import Path
import argparse
import json
import time

from scar.analysis.stateful_dependencies import trace_dependencies
from scar.ir.program import ordered_regions as L, ordered_regions as A
from scar.ir.program.providers import ProviderArtifact,ProviderFactReference
from scar.ir.program.dependencies import (DependencyPoint as Point,DependencyEdge as Edge,
    DependencyEvidence as Evidence,DependencyCondition as Condition,DependencyKind as K,
    DependencyMeaning as Meaning,DependencyStrength as Strength,
    DependencyNeighborhood as Neighborhood,DependencyFrontier as Frontier,
    context_frames,context_pop)


SCHEMA='scar.ordered-input-lifetime'
_GUARDS=('SAME_MATCHED_ORDERED_REGION_ACTIVATION_AND_PROFILE_SELECTED',
    'LOCAL_ORDERED_EPOCHS_ARE_NOT_GLOBAL_LOGICAL_CONTENT_VERSIONS',
    'LOCAL_LAST_USE_DOES_NOT_AUTHORIZE_EXTERNAL_STORAGE_RELEASE')
_CONTENT_ROLES=frozenset((L.OrderedInputUseRole.ARRAY_CONTENT_READ,
    L.OrderedInputUseRole.READ_MODIFY_WRITE,L.OrderedInputUseRole.PRESERVES_UNWRITTEN_REGION))


def _span(values):return (min(values),max(values))if values else(None,None)


class OrderedRegionLifetimeProvider:
    """Query summaries and conditional edges over existing ordered endpoints."""
    relation_kinds=frozenset((K.STATE_INPUT,K.STATE_OUTPUT,K.STATE_INTERFACE,K.VALUE_USE))

    def __init__(self,activation,*,register=False):
        activation.check();self.activation=activation;self.ordered=activation.ordered;self.view=activation.view
        self.uses=[];self.epochs=[];self.inputs=[];self.in_index=defaultdict(list);self.out_index=defaultdict(list)
        self._edges=[];self._derive();self.uses=tuple(self.uses);self.epochs=tuple(self.epochs);self.inputs=tuple(self.inputs)
        self.by_input={row.input_key:row for row in self.inputs}
        self.points={self._key(self.ordered.point(key)):key for key in self.by_input}
        self.check();self._registered=False
        if register:self.view.register(self);self._registered=True

    @staticmethod
    def _key(point):return point.namespace,point.node

    def _edge(self,source,target,kind,rule):
        if source==target:return
        ordinal=len(self._edges);self._edges.append((source,target,kind,rule))
        self.out_index[self._key(source)].append(ordinal);self.in_index[self._key(target)].append(ordinal)

    def _derive(self):
        ordered=self.ordered;region=ordered._region;projection=ordered._projection
        ref=ordered.reference;artifact=ProviderArtifact(ref['uri'],ref['sha256'],ref['size_bytes'],ref['schema'])
        input_keys={row.id:'#/inputs/'+str(i)for i,row in enumerate(region.inputs)}
        action_indices={'#/actions/'+str(i):i for i in range(len(region.actions))}
        records={row['key']:row for row in projection['nodes']}
        scalar_expressions={row['key']:input_keys[row['attributes']['name']]for row in projection['nodes']
            if row.get('op')=='input'and row['attributes']['kind']=='scalar'}
        by_input=defaultdict(list);by_epoch=defaultdict(list);write_actions=defaultdict(list)
        producers={};superseded={}
        for ordinal,edge in enumerate(projection['edges']):
            source,target,kind=edge['source'],edge['target'],edge['kind'];input_key=None;epoch=None;expression=None
            if kind=='OPERAND_VALUE'and source in scalar_expressions:
                input_key=scalar_expressions[source];role=L.OrderedInputUseRole.SCALAR_VALUE_READ;expression=source
            elif kind=='ARRAY_CONTENT_READ':
                epoch_row=records[source];input_key=epoch_row['input_port'];epoch=epoch_row['epoch']
                role=L.OrderedInputUseRole.ARRAY_CONTENT_READ;expression=edge['descriptor']
            elif kind in ('READ_MODIFY_WRITE','PRESERVES_UNWRITTEN_REGION'):
                epoch_row=records[source];input_key=epoch_row['input_port'];epoch=epoch_row['epoch']
                role=L.OrderedInputUseRole(kind)
            elif kind=='WRITE_DESTINATION_REFERENCE':
                input_key=source;role=L.OrderedInputUseRole.WRITE_DESTINATION_REFERENCE
            if input_key is None:continue
            action_index=action_indices[target];action=region.actions[action_index]
            use=L.OrderedInputUse(input_key,action_index,role,epoch,expression,ordinal,
                ProviderFactReference(artifact,target),tuple(ProviderFactReference(artifact,
                    target+'/origins/'+str(i))for i in range(len(action.origins))))
            use_ordinal=len(self.uses);self.uses.append(use);by_input[input_key].append(use_ordinal)
            if role in _CONTENT_ROLES:by_epoch[(input_key,epoch)].append(use_ordinal)
            if role is L.OrderedInputUseRole.SCALAR_VALUE_READ:
                self._edge(ordered.point(input_key),ordered.point(target),K.VALUE_USE,'direct-selected-scalar-use')
        for i,action in enumerate(region.actions):
            if action.kind!='write':continue
            key=input_keys[action.target];write_actions[key].append(i)
            producers[(key,action.epoch_after)]=i;superseded[(key,action.epoch_before)]=i
        epoch_ordinals=defaultdict(list)
        for row in projection['nodes']:
            if row['kind']!='ORDERED_ARRAY_CONTENT_VERSION':continue
            key,epoch=row['input_port'],row['epoch'];uses=tuple(by_epoch[(key,epoch)])
            first,last=_span([self.uses[i].action_index for i in uses])
            at_exit=projection['exit_states'][row['input_id']]==row['key']
            epoch_ordinals[key].append(len(self.epochs))
            self.epochs.append(L.OrderedEpochLifetime(key,row['key'],epoch,producers.get((key,epoch)),uses,
                first,last,superseded.get((key,epoch)),at_exit))
        for i,input in enumerate(region.inputs):
            key=input_keys[input.id];uses=tuple(by_input[key]);writes=tuple(write_actions[key])
            reads=[self.uses[j].action_index for j in uses
                if self.uses[j].role in _CONTENT_ROLES or self.uses[j].role is L.OrderedInputUseRole.SCALAR_VALUE_READ]
            first,last=_span(reads)
            reference_uses=[self.uses[j].action_index for j in uses]
            exit_key=projection['exit_states'].get(input.id);reaches=None
            if input.kind=='array':
                entry_key=projection['entry_states'][input.id]
                reaches=entry_key in trace_dependencies(projection,exit_key)['visited_keys']
                stability=(L.OrderedInputStability.ARRAY_CONTENT_HAS_ORDERED_WRITES if writes
                    else L.OrderedInputStability.ARRAY_CONTENT_PRESERVED)
                self._edge(ordered.point(key),ordered.point(entry_key),K.STATE_INPUT,'input-reference-selects-entry-content')
                if reaches:self._edge(ordered.point(entry_key),ordered.point(exit_key),K.STATE_INTERFACE,'entry-content-dependency-survives-to-exit')
                self._edge(ordered.point(exit_key),self.activation.ports[A.OrderedBoundaryRole.WORLD_EXIT],
                    K.STATE_OUTPUT,'final-local-content-required-at-normal-exit')
            else:stability=L.OrderedInputStability.SELECTED_SCALAR_VALUE_FIXED
            self.inputs.append(L.OrderedInputLifetime(key,ProviderFactReference(artifact,key),input.kind,
                tuple(j for j,route in enumerate(self.activation.inputs)if route.input_key==key),
                tuple('#/nodes/'+str(j)for j,node in enumerate(region.nodes)
                    if node.op=='input'and node.attrs['name']==input.id),uses,writes,tuple(epoch_ordinals[key]),
                first,last,max(reference_uses)if reference_uses else None,exit_key,reaches,stability))

    def check(self):self.activation.check()

    def lifetime(self,input_point):
        self.check()
        key=self.points.get(self._key(input_point))
        if key is None:raise ValueError('ordered lifetime query requires an original input port')
        return self.by_input[key]

    def _neighbors(self,point,incoming):
        rows=(self.in_index if incoming else self.out_index).get(self._key(point),())
        if not rows:return Neighborhood()
        self.check()
        if point.context is not None:
            frames=context_frames(point.context)
            binding_index=self.activation.ordered_frames.get(frames[-1])if frames else None
            if binding_index is None:
                return Neighborhood(frontiers=(Frontier(point,'ORDERED_LIFETIME_ACTIVATION_NOT_SELECTED',
                    'Region-local lifetime requires the original matched ordered activation.'),))
            binding=self.activation.bindings[binding_index]
            popped=context_pop(point.context,(binding.ordered_frame,))
            if not popped.matched or not self.activation._caller_matches(binding,popped.remaining):return Neighborhood()
        entry=replace(self.ordered.entry,context=point.context);edges=[]
        guards=tuple(dict.fromkeys((*self.ordered._region.guards,*_GUARDS)))
        for i in rows:
            source,target,kind,rule=self._edges[i]
            source=replace(source,context=point.context);target=replace(target,context=point.context)
            conditions=tuple(Condition(SCHEMA,g,entry,None,(source,target))for g in guards)
            evidence=Evidence(SCHEMA+'.'+rule,(self.ordered.locator,),(entry,source,target))
            edges.append(Edge(source,target,kind,Meaning.DEPENDENCE,(evidence,),conditions,Strength.MAY))
        return Neighborhood(tuple(dict.fromkeys(edges)))

    def incoming(self,point):return self._neighbors(point,True)
    def outgoing(self,point):return self._neighbors(point,False)

    def summary(self):
        return dict(inputs=len(self.inputs),uses=len(self.uses),epochs=len(self.epochs),
            stability=dict(Counter(row.stability.value for row in self.inputs)),
            direct_use_roles=dict(Counter(row.role.value for row in self.uses)),
            array_write_actions=sum(len(row.write_actions)for row in self.inputs),
            conditional_relations=len(self._edges),
            arrays_required_at_normal_exit=sum(row.required_at_normal_exit for row in self.epochs),
            scope='ONE_SELECTED_GUARDED_REGION_ACTIVATION',
            cross_call_content_identity_proven=False,loop_recurrence_proven=False,
            physical_allocation_or_commit_timing_proven=False,external_storage_release_authorized=False,
            global_logical_versions_created=0,target_executed=False,transformation_authorized=False)


def _pins():
    from scar.analysis import stateful_dependencies,inspect_ordered_activations
    from scar.ir import record_codec
    return dict(producer=sha256(Path(__file__).read_bytes()).hexdigest(),
        types=sha256(Path(L.__file__).read_bytes()).hexdigest(),
        projection=sha256(Path(stateful_dependencies.__file__).read_bytes()).hexdigest(),
        codec=sha256(Path(record_codec.__file__).read_bytes()).hexdigest(),
        original_activation=inspect_ordered_activations._pins())


def publish_ordered_lifetimes(provider,out,*,activation_publication):
    from scar.artifacts import write_table
    from scar.artifacts import read_record_table as _table
    from scar.ir.execution_recipe import artifact_reference,read_artifact
    from scar.ir.record_codec import decode
    from scar.analysis.ordered_loop_transfer_storage import _activation_header,_activation_tables
    from scar.analysis.inspect_ordered_activations import validate_public_interfaces
    out=Path(out).resolve()
    if out.exists():raise ValueError('fresh ordered lifetime output required')
    parent=Path(activation_publication).resolve()
    if parent.is_dir():parent=parent/'result.json'
    reference=artifact_reference(parent,'scar.ordered-activation-interfaces');report=read_artifact(reference)
    _activation_header(report)
    if report['original_region']!=provider.ordered.reference or report['summary']!=provider.activation.summary():
        raise ValueError('ordered lifetime parent differs from the replayed activation')
    if provider._registered:
        raise ValueError('publish original activation validation before attaching lifetime extension')
    _activation_tables(parent.parent,report,provider.activation)
    if validate_public_interfaces(provider.activation)!=report['validation']:
        raise ValueError('ordered lifetime parent public graph validation differs')
    pins=_pins();provider.check();out.mkdir(parents=True)
    provider.view.register(provider);provider._registered=True
    definition_edges=tuple(dict.fromkeys(edge for source,_,_,_ in provider._edges
        for edge in provider.outgoing(source).edges))
    groups=dict(inputs=(L.OrderedInputLifetime,provider.inputs),
        epochs=(L.OrderedEpochLifetime,provider.epochs),uses=(L.OrderedInputUse,provider.uses),
        edges=(Edge,definition_edges))
    tables={name:write_table(out/(name+'.jsonl.gz'),rows)for name,(_,rows)in groups.items()}
    absent=object()
    for name,(cls,rows)in groups.items():
        if any(w is absent or r is absent or decode(cls,w)!=r
                for w,r in zip_longest(_table(out,tables[name]),rows,fillvalue=absent)):
            raise ValueError('ordered lifetime typed table differs: '+name)
    observed=set()
    for source,target,_,_ in provider._edges:
        for point in(source,target):
            observed.update(provider.incoming(point).edges);observed.update(provider.outgoing(point).edges)
    for edge in observed:
        if edge not in provider.view.dependencies.incoming(edge.target,kinds=(edge.kind,)).edges:
            raise ValueError('ordered lifetime incoming edge missing in public graph')
        if edge not in provider.view.dependencies.outgoing(edge.source,kinds=(edge.kind,)).edges:
            raise ValueError('ordered lifetime outgoing edge missing in public graph')
    provider.check();read_artifact(reference,json_document=False)
    if pins!=_pins():raise ValueError('ordered lifetime producing rules changed')
    result=dict(schema=SCHEMA,schema_version=1,activation_publication=reference,
        original_region=provider.ordered.reference,pins=pins,tables=tables,summary=provider.summary(),
        graph_validation=dict(conditional_edges=len(observed),bidirectional_missing=0),
        original_body_bytes_copied=0,iterations_unrolled=False,target_executed=False,
        transformation_authorized=False)
    (out/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return result


def main(argv=None):
    from scar.analysis.build import BuildSession
    from scar.analysis.ordered_loop_transfer_storage import _activation_header
    from scar.analysis.stateful_dependency_view import StatefulDependencyProvider,read_stateful_graph
    from scar.analysis.ordered_region_activation import OrderedRegionActivationProvider
    from scar.analysis.source_call_view import SourceFunctionCallProvider
    from scar.ir.execution_recipe import artifact_reference,read_artifact
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interfaces',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args(argv);started=time.perf_counter();path=Path(args.interfaces).resolve()
    if path.is_dir():path=path/'result.json'
    parent=read_artifact(artifact_reference(path,'scar.ordered-activation-interfaces'));_activation_header(parent)
    with BuildSession(Path(parent['program']['uri']).parent)as session:
        ordered=StatefulDependencyProvider(session.view,parent['original_region']['uri'])
        read_stateful_graph(Path(parent['shared_body']['uri']).parent,session.view,_provider=ordered)
        calls={name:p for name,p in session.session.providers.items()if type(p)is SourceFunctionCallProvider}
        activation=OrderedRegionActivationProvider(ordered,calls)
        provider=OrderedRegionLifetimeProvider(activation)
        result=publish_ordered_lifetimes(provider,args.out,activation_publication=path)
        acceptance=dict(summary=result['summary'],graph_validation=result['graph_validation'],
            seconds=time.perf_counter()-started,target_executed=False)
        (Path(args.out)/'acceptance.json').write_text(json.dumps(acceptance,sort_keys=True,indent=2)+'\n')
        print(json.dumps(acceptance,sort_keys=True))
    return 0


__all__=['OrderedRegionLifetimeProvider','publish_ordered_lifetimes']


if __name__=='__main__':raise SystemExit(main())
