"""Connect local ordered content transfers to original shared loop backedges.

This is a state-path interface over existing source/ordered ports, not a new
SSA value identity. Exact CALLs and formal routes establish structural loop
membership; separate caller effect obligations qualify content recurrence.
"""
from collections import defaultdict,Counter
from dataclasses import dataclass,replace
from pathlib import Path
from types import SimpleNamespace as NS

from scar.analysis.source_call_view import function_interface_fact,source_call_site,source_caller_scope
from scar.analysis.source_control import loop_dependencies, _enclosed
from scar.analysis.source_namespace_interfaces import NamespacePlanView
from scar.analysis.source_reference_slice import ReferenceSlicer,source_port
from scar.analysis.source_value_flow import project_value_flow
from scar.analysis.source_selected_call_window import derive_selected_call_window
from scar.ir.program import ordered_regions as R, normal_graph as F, normal_graph as G
from scar.ir.program import normal_graph as T, source_productions as P
from scar.ir.program.model import ValueKind,ValueType
from scar.ir.program.providers import ProviderFactReference
from scar.ir.program.call_returns import OrderedWrapperCallWindow
from scar.ir.program.dependencies import (DependencyEdge as Edge,DependencyEvidence as Evidence,
    DependencyCondition as Condition,DependencyKind as K,DependencyMeaning as Meaning,
    DependencyStrength as Strength,DependencyNeighborhood as Neighborhood,DependencyFrontier as Frontier,
    context_push,context_pop)
from scar.ir.record_codec import decode


SCHEMA='scar.ordered-loop-state-transfer'
_GUARDS=('SAME_LOOP_CALLER_ACTIVATION_AND_RECORDED_CALL_PATH_SELECTED',
    'RECORDED_CALL_PATH_EXECUTES_AND_COMPLETES_NORMALLY_ON_SELECTED_ITERATIONS',
    'CURRENT_FORMAL_AND_ORDINARY_FIELD_PATH_SELECT_SAME_RECEIVER_INSTANCES',
    'CALLER_EFFECTS_BETWEEN_SELECTED_CALLS_PRESERVE_DECLARED_STATE_PATH_DEPENDENCE',
    'NORMAL_BACKEDGE_SELECTED_NOT_BREAK_EXCEPTION_OR_SUSPENSION')


def _leaves(path):
    return {path.states[i].port for i in set(range(len(path.states)))-{e.downstream for e in path.transitions}}


def _input_role(kind,written,caller_writes,frontiers):
    if kind=='array'and written:return R.OrderedLoopInputRole.ORDERED_ARRAY_CONTENT_CARRY
    if caller_writes:return R.OrderedLoopInputRole.CALLER_REWRITTEN_FIELD
    if kind=='scalar'and not frontiers:return R.OrderedLoopInputRole.CONDITIONAL_LOOP_LIVE_IN_SCALAR
    return R.OrderedLoopInputRole.LOOP_LIVE_IN_ROOT_SELECTED_VALUE_UNPROVEN


@dataclass(frozen=True,slots=True)
class _Edge:
    loop:int
    source:object
    source_depth:int
    target:object
    target_depth:int
    kind:K
    rule:str
    guards:tuple[str,...]


class OrderedLoopTransferProvider:
    relation_kinds=frozenset((K.CONTROL_FLOW,K.STATE_INTERFACE,K.STATE_INPUT))

    def __init__(self,lifetime,*,register=True,max_call_depth=8):
        lifetime.check();self.lifetime=lifetime;self.activation=lifetime.activation
        self.ordered=lifetime.ordered;self.view=lifetime.view;self.providers=self.activation.providers
        self.loops=[];self.root_routes=[];self.caller_writes=[];self.frontiers=[];self.transfers=[];self.selected_windows=[]
        self.in_index=defaultdict(list);self.out_index=defaultdict(list);self._edges=[]
        self._graphs={};self._slicers={};self._slices={};self._flows={};self._loop_data={};self._scope_functions={}
        self._roots={};self._contexts={};self._limits=max_call_depth
        self._by_body=defaultdict(list)
        for name,provider in self.providers.items():
            for unit in sorted(provider.available_units):
                graph=provider.store.read_unit(unit);self._graphs[name,unit]=graph
                for ordinal,interface in enumerate(graph.interfaces):
                    body=interface.link.body
                    if interface.activation is T.FunctionActivation.NORMAL and body is not None:
                        self._by_body[body].append((name,unit,ordinal))
        for bi,binding in enumerate(self.activation.bindings):
            self._walk(bi,((binding.provider,binding.unit,binding.ordinal),),frozenset())
        self.loops=tuple(self.loops);self.root_routes=tuple(self.root_routes)
        self.caller_writes=tuple(self.caller_writes);self.frontiers=tuple(self.frontiers);self.transfers=tuple(self.transfers)
        self.selected_windows=tuple(self.selected_windows)
        self.check()
        if register:self.view.register(self)

    @staticmethod
    def _key(point):return point.namespace,point.node

    def _slicer(self,unit):
        if unit not in self._slicers:
            source=self.view.unit(unit);material=source.material
            self._slicers[unit]=ReferenceSlicer(NS(fact_store=material.fact_store,
                rows=lambda *a,**k:(),validate_parents=material.validate_parents))
        return self._slicers[unit]

    def _fact(self,unit,table,index):return self._slicer(unit).fact(self.view.unit(unit).facts,table,index)

    def _slice(self,point):
        if point not in self._slices:self._slices[point]=self._slicer(point.unit).slice(point)
        return self._slices[point]

    def _hop(self,key):
        name,unit,ordinal=key;provider=self.providers[name];graph=self._graphs[name,unit]
        interface=graph.interfaces[ordinal];source=self.view.unit(unit)
        operation,_,_=source_call_site(self.view,interface.link)
        step=source.records[operation.node]
        context=NamespacePlanView(unit,source.plans[interface.link.result.port.template])
        owner=context.owner(step.region).owner
        if (unit,owner)not in self._scope_functions:
            functions=[(i,f)for i,f in enumerate(source.facts.functions)
                if f.body_scope is not None and f.body_scope.owner==owner]
            self._scope_functions[unit,owner]=functions
        functions=self._scope_functions[unit,owner]
        if len(functions)!=1:return None
        provider.attach(unit,queries=(interface.query,));frame=provider.frames[unit,ordinal]
        query=next(q for q in graph.queries if q.query==interface.query)
        if owner not in query.owners:raise ValueError('loop path CALL scope differs from original interface owners')
        hop=R.OrderedLoopCallHop(name,unit,ordinal,interface.query,
            function_interface_fact(provider.store,unit,ordinal),frame,owner,tuple(interface.link.conditions))
        return hop,functions[0],interface

    def _body(self,unit,function):
        body=T.Body(unit,function.template,function.body)
        if body not in self._flows:
            source=self.view.unit(unit)
            flow=project_value_flow(source.material,source_port(source.facts,function.value),body)
            self._flows[body]=flow;self._loop_data[body]=loop_dependencies(source.material,function,flow=flow)
        return self._flows[body],self._loop_data[body]

    def _walk(self,binding,path,seen):
        selected=self._hop(path[0])
        if selected is None:
            self.frontiers.append(R.OrderedLoopEffectFrontier(None,None,'CALLER_IS_NOT_ONE_ORIGINAL_FUNCTION_SCOPE'));return
        hop,(fi,function),interface=selected;body=T.Body(hop.unit,function.template,function.body)
        if body in seen:
            self.frontiers.append(R.OrderedLoopEffectFrontier(None,None,'RECURSIVE_CALLER_PATH_NEEDS_RECURSION_INTERFACE'));return
        flow,report=self._body(hop.unit,function)
        matches=[row for row in report['loops']if any(decode(T.GlobalPort,c['result'])==interface.link.result
            for c in row['body_calls'])]
        # Prefer the innermost enclosing shared loop; its parent relation stays
        # in original F4. A loop elsewhere in this function is never enough.
        if matches:
            parents={str(row['parent'])for row in matches if row['parent']is not None}
            selected_loops=[row for row in matches if str(row['header'])not in parents]
            for row in selected_loops:self._loop(binding,path,fi,function,flow,row)
            return
        if len(path)>=self._limits:
            self.frontiers.append(R.OrderedLoopEffectFrontier(None,None,'CALLER_PATH_DEPTH_FRONTIER'));return
        target=source_port(self.view.unit(hop.unit).facts,function.value)
        parents=[key for key in self._by_body.get(body,())
            if self._graphs[key[0],key[1]].interfaces[key[2]].link.source_callable==target]
        if not parents:
            self.frontiers.append(R.OrderedLoopEffectFrontier(None,None,'NO_MATCHED_PARENT_CALLER_LOOP_INTERFACE'));return
        for parent in parents:self._walk(binding,(parent,*path),seen|{body})

    def _membership(self,flow,row):
        header=decode(T.GlobalPort,row['header']);source=self.view.unit(header.unit)
        source._ensure_original(header.port.template,header.port)
        operation=source._node(header.port.template,header.port.owner);step=source.records[operation.node]
        if type(step.instruction)is not P.LoopControl:raise ValueError('loop header is not original LoopControl')
        stops={header.port,*(G.Port(header.port.template,step.id,G.PortKind.EXIT,
            completion=c.kind,target=c.target)for c in step.completions)}
        adjacency=defaultdict(list)
        for edge in flow.edges:
            if edge.kind is G.EdgeKind.CONTROL:adjacency[edge.source].append(edge)
        inside=_enclosed(G.Port(header.port.template,step.instruction.body,G.PortKind.ENTER),adjacency,stops)
        if type(step.instruction.condition)is P.ProductionRegionID:
            inside|=_enclosed(G.Port(header.port.template,step.instruction.condition,G.PortKind.ENTER),adjacency,stops)
        return inside

    def _loop(self,binding,path,fi,function,flow,row):
        hops=tuple(self._hop(key)[0]for key in path);header=decode(T.GlobalPort,row['header'])
        carries=[]
        for slot in row['slots']:
            kind=decode(ValueType,slot['value_type'])
            if slot['role']=='CARRIED_SUFFIX'and kind.kind is ValueKind.STATE:
                carries.append(R.OrderedLoopWorldCarry(slot['index'],decode(T.GlobalPort,slot['formal']),
                    decode(tuple[G.Edge,...],slot['backedges'])))
        backedges=decode(tuple[G.Edge,...],row['control_backedges'])
        if not carries or not backedges:
            self.frontiers.append(R.OrderedLoopEffectFrontier(None,None,'ORIGINAL_LOOP_WORLD_OR_CONTROL_BACKEDGE_MISSING',header));return
        call=next(c for c in row['body_calls']if decode(T.GlobalPort,c['result'])==self._hop(path[0])[2].link.result)
        li=len(self.loops)
        loop=R.OrderedLoopInterface(binding,self._fact(header.unit,'functions',fi),header,
            decode(T.Body,row['body']),decode(ProviderFactReference,call['fact']),hops,tuple(carries),backedges)
        self.loops.append(loop);inside=self._membership(flow,row)
        rootmap={}
        for index,input in enumerate(self.lifetime.inputs):
            sources=[self.activation.inputs[j]for j in input.source_route_ordinals]
            if not sources:
                self.frontiers.append(R.OrderedLoopEffectFrontier(li,input.input_key,'ORDERED_INPUT_HAS_NO_ORIGINAL_FORMAL_ROUTE'));continue
            roots={source.root_formal for source in sources}
            fields={source.field_path for source in sources}
            if len(roots)!=1 or len(fields)!=1:
                self.frontiers.append(R.OrderedLoopEffectFrontier(li,input.input_key,'INPUT_ORIGINAL_ROOT_OR_FIELD_PATH_IS_AMBIGUOUS'));continue
            root=next(iter(roots));field_path=next(iter(fields))
            if root not in rootmap:
                rootmap[root]=self._root(li,path,root)
            ri=rootmap[root]
            if ri is None:continue
            if len(field_path)>1:
                self.frontiers.append(R.OrderedLoopEffectFrontier(li,input.input_key,'NESTED_FIELD_EFFECT_PATH_NOT_MODELED'));continue
            if li not in self._contexts:self._contexts[li]=self._effects(li,path,inside)
            writes,barriers=self._contexts[li]
            rr=self.root_routes[ri]
            stage_roots={len(path)-1-i:hop.caller_formal for i,hop in enumerate(rr.hops)}
            matching=[];uncertain=[]
            for wi in writes:
                write=self.caller_writes[wi]
                if not field_path or write.field!=field_path[0]:continue
                expected=stage_roots[write.caller_depth]
                if set(write.receiver_roots)=={expected}:matching.append(wi)
                else:uncertain.append(wi)
            effects=list(barriers)
            for wi in uncertain:
                write=self.caller_writes[wi]
                effects.append(len(self.frontiers));self.frontiers.append(R.OrderedLoopEffectFrontier(li,input.input_key,
                    'CALLER_FIELD_STORE_RECEIVER_ALIAS_OR_CURRENT_BINDING_UNRESOLVED',write.receiver,write.fact))
            role=_input_role(input.input_kind,input.write_actions,matching,effects)
            guards=tuple(dict.fromkeys((*_GUARDS,*self.ordered._region.guards,
                *(g for hop in hops for g in hop.guards),
                *(g for source in sources for g in source.guards))))
            entry=self.ordered._projection['entry_states'].get(self.ordered._region.inputs[index].id)
            transfer=R.OrderedLoopInputTransfer(li,input.input_key,index,ri,field_path,role,entry,
                input.exit_epoch_key,tuple(matching),tuple(effects),guards)
            self.transfers.append(transfer)
            if input.input_kind=='array':
                self._edge(li,self.ordered.point(input.exit_epoch_key),-1,self.ordered.point(entry),-1,
                    K.STATE_INTERFACE,'conditional-next-selected-iteration-content-carry',guards)
            else:
                self._edge(li,self.view.source_port(rr.loop_caller_formal),0,self.ordered.point(input.input_key),-1,
                    K.STATE_INPUT,'loop-live-in-root-selects-current-scalar-field',guards)
            for wi in matching:
                write=self.caller_writes[wi]
                if write.payload is not None:
                    self._edge(li,self.view.source_port(write.payload),write.caller_depth,
                        self.ordered.point(input.input_key),-1,K.STATE_INPUT,
                        'caller-store-may-feed-next-selected-iteration-field',
                        (*guards,'THIS_ORIGINAL_STORE_IS_SELECTED_REACHING_DEFINITION_AT_NEXT_CALL'))
        self._edge(li,self.view.source_port(header),0,self.ordered.entry,-1,K.CONTROL_FLOW,
            'shared-loop-contains-selected-original-call-path',_GUARDS)

    def _root(self,loop,path,root):
        current=root;hops=[]
        for key in reversed(path):
            hop,(_,function),interface=self._hop(key);link=interface.link
            routes=[r for r in link.parameters if r.formal==current and r.kind is T.RouteKind.ARGUMENT
                and len(r.actuals)==1 and type(r.actuals[0])is T.GlobalPort]
            if len(routes)!=1:
                self.frontiers.append(R.OrderedLoopEffectFrontier(loop,None,'ROOT_HAS_NO_SINGLE_ORIGINAL_REFERENCE_ARGUMENT_ROUTE',current,hop.interface));return None
            actual=routes[0].actuals[0];reference=self._slice(actual)
            formals={source_port(self.view.unit(hop.unit).facts,p)for p in function.formals[1:-1]}
            roots=_leaves(reference)
            if len(roots)!=1 or not roots<=formals:
                self.frontiers.append(R.OrderedLoopEffectFrontier(loop,None,
                    'CALLER_ACTUAL_REFERENCE_DOES_NOT_SELECT_ONE_CURRENT_CALLER_FORMAL',actual,hop.interface));return None
            caller_formal=next(iter(roots))
            hops.append(R.OrderedLoopFormalHop(hop.interface,current,actual,caller_formal,reference));current=caller_formal
        result=len(self.root_routes);self.root_routes.append(R.OrderedLoopRootRoute(loop,root,current,tuple(hops)))
        return result

    def _effects(self,loop,path,inside):
        writes=[];frontiers=[]
        selected={self._hop(key)[0].frame.operation for key in path}
        for depth,key in enumerate(path):
            hop,(_,function),_=self._hop(key);source=self.view.unit(hop.unit)
            flow,_=self._body(hop.unit,function)
            operations=[op for op in flow.operations if depth or op.entry in inside]
            if depth>0:
                window=derive_selected_call_window(self.providers[hop.provider],hop.unit,hop.ordinal)
                self.selected_windows.append(OrderedWrapperCallWindow(loop,depth,window))
                excluded={row.operation.port for row in window.exclusions}
                operations=[op for op in operations if op.entry not in excluded]
                for boundary in window.boundaries:
                    frontiers.append(len(self.frontiers));self.frontiers.append(R.OrderedLoopEffectFrontier(loop,None,
                        'SELECTED_WRAPPER_WINDOW_'+boundary.reason,boundary.operation,window.original_call))
            sites={op.entry for op in operations}
            for i,attribute in enumerate(source.facts.attributes):
                if attribute.action not in(F.AttributeAction.SET,F.AttributeAction.DELETE):continue
                world=source_port(source.facts,attribute.result_world)
                entry=G.Port(world.port.template,world.port.owner,G.PortKind.ENTER)
                if entry not in sites:continue
                receiver=source_port(source.facts,attribute.receiver)
                writes.append(len(self.caller_writes))
                self.caller_writes.append(R.OrderedLoopCallerWrite(loop,depth,self._fact(source.unit,'attributes',i),
                    attribute.action.value,receiver,attribute.name,
                    None if attribute.value is None else source_port(source.facts,attribute.value),
                    tuple(sorted(_leaves(self._slice(receiver)),key=str))))
            for op in operations:
                ins=op.instruction;site=T.GlobalPort(source.unit,op.entry)
                operation=source._node(op.entry.template,op.entry.owner)
                if operation.node in selected:continue
                if type(ins)is P.DefineLiteral:continue
                if type(ins)is P.AccessBinding and ins.action is P.BindingActionKind.READ:continue
                reason=('OTHER_CALL_EFFECT_SUMMARY_NOT_BOUND_TO_SELECTED_LOOP_STATE_PATH'
                    if type(ins)is P.ApplyLanguageAction and ins.action is P.LanguageActionKind.CALL
                    else 'CALLER_PROTOCOL_OR_REFERENCE_CLEANUP_EFFECTS_REQUIRE_SELECTED_STATE_CONTRACT')
                frontiers.append(len(self.frontiers));self.frontiers.append(R.OrderedLoopEffectFrontier(loop,None,reason,site))
        return tuple(writes),tuple(frontiers)

    def _edge(self,loop,source,sd,target,td,kind,rule,guards):
        ordinal=len(self._edges);self._edges.append(_Edge(loop,source,sd,target,td,kind,rule,tuple(guards)))
        self.out_index[self._key(source)].append(ordinal);self.in_index[self._key(target)].append(ordinal)

    def _context(self,row,point,incoming):
        if point.context is None:return None
        loop=self.loops[row.loop];frames=tuple(h.frame for h in loop.call_path)
        ordered=(*frames[:-1],self.activation.bindings[loop.binding].ordered_frame)
        depth=row.target_depth if incoming else row.source_depth
        expected=ordered if depth<0 else frames[:depth]
        popped=context_pop(point.context,expected)
        if not popped.matched:return False
        active=source_caller_scope(self.view,frames[0],popped.remaining)
        if active is not None and active!=loop.call_path[0].caller_scope:return False
        return popped.remaining

    def _project(self,row,base):
        loop=self.loops[row.loop];frames=tuple(h.frame for h in loop.call_path)
        def endpoint(point,depth):
            chain=(*frames[:-1],self.activation.bindings[loop.binding].ordered_frame)if depth<0 else frames[:depth]
            return replace(point,context=context_push(base,*chain))
        return endpoint(row.source,row.source_depth),endpoint(row.target,row.target_depth)

    def _neighbors(self,point,incoming):
        indices=(self.in_index if incoming else self.out_index).get(self._key(point),())
        if not indices:return Neighborhood()
        self.check();edges=[];unmatched=False
        for i in indices:
            row=self._edges[i];base=self._context(row,point,incoming)
            if base is False:unmatched=True;continue
            loop=self.loops[row.loop]
            choices=[(row.source,row.target,None)]if point.context is None else[(*self._project(row,base),base)]
            depth=row.target_depth if incoming else row.source_depth
            # A zero-depth source endpoint has an unknown caller prefix and
            # therefore remains context=None even when its peer enters the
            # known selected call chain. Emit that paired interface as well
            # as the unselected definition, preserving reverse queries.
            if point.context is None and depth==0:choices.append((*self._project(row,None),None))
            for source,target,caller in choices:
                header=replace(self.view.source_port(loop.header),context=caller)
                evidence=Evidence(SCHEMA+'.'+row.rule,
                    (self.ordered.locator,*(self.providers[h.provider].manifest for h in loop.call_path)),(header,source,target))
                conditions=tuple(Condition(SCHEMA,g,header,None,(source,target))for g in row.guards)
                edges.append(Edge(source,target,row.kind,Meaning.DEPENDENCE,(evidence,),conditions,Strength.MAY))
        return Neighborhood(tuple(dict.fromkeys(edges)),()if not unmatched else(Frontier(point,
            'LOOP_STATE_TRANSFER_REQUIRES_EXACT_ORIGINAL_CALL_PATH_SUFFIX',
            'Another caller or a partial frame suffix cannot select this shared loop recurrence.'),))

    def incoming(self,point):return self._neighbors(point,True)
    def outgoing(self,point):return self._neighbors(point,False)

    def check(self):
        self.lifetime.check()
        for provider in self.providers.values():provider.store.validate()

    def summary(self):
        return dict(shared_loops=len(self.loops),input_transfers=len(self.transfers),root_routes=len(self.root_routes),
            original_call_hops=sum(len(l.call_path)for l in self.loops),
            original_world_carries=sum(len(l.world_carries)for l in self.loops),
            original_control_backedges=sum(len(l.control_backedges)for l in self.loops),
            roles=dict(Counter(t.role.value for t in self.transfers)),caller_attribute_writes=len(self.caller_writes),
            effect_frontiers=len(self.frontiers),conditional_relations=len(self._edges),
            selected_wrapper_windows=len(self.selected_windows),
            excluded_wrapper_operations=sum(len(w.window.exclusions)for w in self.selected_windows),
            selected_window_boundaries=sum(len(w.window.boundaries)for w in self.selected_windows),
            structural_call_loop_membership_rederived=True,unconditional_content_recurrence_proven=False,
            global_logical_versions_created=0,iterations_unrolled=False,target_executed=False,
            transformation_authorized=False)


__all__=['OrderedLoopTransferProvider']
