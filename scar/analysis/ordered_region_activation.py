"""Attach caller-matched source ABIs to a source-replayed ordered profile.

Original and ordered contexts are alternatives for the same CALL. Replacing
the final frame preserves its caller, callee Use/value and original selection;
copying that frame into another namespace would mix activations. These are
conditional implementation interfaces, not runtime invocations or new proofs
of global logical content equality.
"""
from collections import defaultdict,Counter
from dataclasses import replace
from hashlib import sha256

from scar.analysis.ordered_region_inputs import derive_ordered_inputs
from scar.analysis.ordered_region_inline import OrderedRegionInlineIndex
from scar.analysis.source_call_view import (SourceFunctionCallProvider,function_interface_fact,
    source_caller_scope)
from scar.analysis.source_namespace_interfaces import NamespacePlanView
from scar.analysis.source_reference_slice import source_port
from scar.ir.program import ordered_regions as A, view as V, normal_graph as G, normal_graph as T
from scar.ir.program.ids import ValueID,ControlPortID
from scar.ir.program.dependencies import (DependencyPoint as Point,
    DependencyCondition as Condition,DependencyEvidence as Evidence,DependencyEdge as Edge,
    DependencyKind as K,DependencyMeaning as Meaning,DependencyStrength as Strength,
    DependencyNeighborhood as Neighborhood,DependencyFrontier as Frontier,
    context_push,context_pop,context_frames)


SCHEMA='scar.ordered-region-matched-activation'
_GUARDS=('SAME_ORIGINAL_CALL_AND_SELECTED_SOURCE_FUNCTION_INSTANCE',
    'ORDERED_PROFILE_IMPLEMENTATION_SELECTED_FOR_THIS_CALL',
    'SOURCE_AND_ORDERED_CONTEXTS_SELECT_THE_SAME_ACTIVATION',
    'ALL_ORDERED_REGION_INPUT_STATE_ALIAS_PROVIDER_AND_NORMAL_COMPLETION_CONTRACTS_HOLD')


class OrderedRegionActivationProvider:
    relation_kinds=frozenset(K)

    def __init__(self,ordered,providers,*,register=True):
        self.ordered=ordered;self.view=ordered.view;self.providers=dict(providers)
        if any(type(p)is not SourceFunctionCallProvider or p.view is not self.view for p in self.providers.values()):
            raise TypeError('original same-view source-call providers required')
        self.inputs,self.input_boundaries=derive_ordered_inputs(ordered)
        source=self.view.namespaces[ordered.source_body.namespace];self.source=source
        identity=next(o for o in ordered._region.source.origins if o.get('kind')=='source_identity')
        functions=[f for f in source.facts.functions if f.template.occurrence==identity['occurrence']]
        if len(functions)!=1:raise ValueError('ordered activation lacks its one original function')
        self.function=functions[0];self.bindings=[];self.routes=[]
        self.nodes={};self.records={};self.ports={};self._scope_cache={}
        self.out_index=defaultdict(list);self.in_index=defaultdict(list)
        self.source_frames=defaultdict(list);self.ordered_frames={};self._seen=set()
        self._boundaries=[];self._ports()
        self.attach()
        self.inline=OrderedRegionInlineIndex(ordered,self.providers)
        if register:
            if getattr(ordered,'activation_provider',None)is not None:
                raise ValueError('ordered region already has a matched activation adapter')
            ordered.activation_provider=self
            self.view.register(self)

    @staticmethod
    def _key(point):return point.namespace,point.node

    def _port(self,role,original=None):
        control=role in (A.OrderedBoundaryRole.CONTROL_ENTRY,A.OrderedBoundaryRole.CONTROL_EXIT)
        cls=ControlPortID if control else ValueID
        identity=cls(sha256((self.ordered.digest+'\0'+role.value).encode()).hexdigest())
        point=Point(self.ordered.namespace,identity,None)
        node_role=(V.ProgramNodeRole.ENTER if role is A.OrderedBoundaryRole.CONTROL_ENTRY else
            V.ProgramNodeRole.EXIT if role is A.OrderedBoundaryRole.CONTROL_EXIT else
            V.ProgramNodeRole.PARAMETER if role in (A.OrderedBoundaryRole.WORLD_ENTRY,A.OrderedBoundaryRole.ENVIRONMENT)
            else V.ProgramNodeRole.RESULT)
        value_type=None if original is None else self.view.get(self.view.source_port(original)).value_type
        self.nodes[identity]=V.ProgramNode(point,None if control else V.ProgramEntityKind.VALUE,
            node_role,self.ordered.entry,value_type)
        self.records[identity]=A.OrderedBoundaryPort(role,original);self.ports[role]=point

    def _ports(self):
        f=self.function;facts=self.source.facts
        self._port(A.OrderedBoundaryRole.ENVIRONMENT,source_port(facts,f.formals[0]))
        self._port(A.OrderedBoundaryRole.WORLD_ENTRY,source_port(facts,f.formals[-1]))
        self._port(A.OrderedBoundaryRole.WORLD_EXIT,None if f.return_world is None else source_port(facts,f.return_world))
        self._port(A.OrderedBoundaryRole.RETURN_NONE,None if f.returns is None else source_port(facts,f.returns))
        self._port(A.OrderedBoundaryRole.CONTROL_ENTRY);self._port(A.OrderedBoundaryRole.CONTROL_EXIT)

    def attach(self):
        """Select exact original bodies in retained interfaces, not source names."""
        for name,provider in self.providers.items():
            for unit in sorted(provider.available_units):
                graph=provider.store.read_unit(unit)
                for ordinal,interface in enumerate(graph.interfaces):
                    body=interface.link.body
                    if (interface.activation is not T.FunctionActivation.NORMAL or body is None
                            or body.unit!=self.source.unit or body.template!=self.function.template
                            or body.region!=self.function.body):continue
                    key=name,unit,ordinal
                    if key in self._seen:continue
                    provider.attach(unit,queries=(interface.query,))
                    source_frame=provider.frames[(unit,ordinal)]
                    if source_frame.target!=self.ordered.source_body.namespace or source_frame.entry!=self.ordered.source_body.node:
                        raise ValueError('original call selection differs from the replayed source body')
                    alternative=replace(source_frame,target=self.ordered.namespace,entry=self.ordered.entry.node,definition=None)
                    index=len(self.bindings)
                    binding=A.OrderedActivationBinding(name,unit,ordinal,interface.query,
                        function_interface_fact(provider.store,unit,ordinal),source_frame,alternative,
                        (*interface.link.conditions,*_GUARDS))
                    self.bindings.append(binding);self.source_frames[source_frame].append(index)
                    self.ordered_frames[alternative]=index;self._seen.add(key)
                    self._bind_routes(index,interface)
            provider.store.validate()
        return self

    def _route(self,index,source,source_space,target,target_space,kind,meaning,rule,guards=()):
        row=A.OrderedActivationRoute(index,source,source_space,target,target_space,kind,meaning,rule,tuple(guards))
        ordinal=len(self.routes);self.routes.append(row)
        self.out_index[self._key(source)].append(ordinal);self.in_index[self._key(target)].append(ordinal)

    def _bind_routes(self,index,interface):
        C,S,O=A.OrderedContextSpace.CALLER,A.OrderedContextSpace.SOURCE,A.OrderedContextSpace.ORDERED
        B=A.OrderedBoundaryRole;b=self.bindings[index];call=Point(b.source_frame.caller,b.source_frame.operation,None)
        add=lambda s,ss,t,ts,k,m,r,g=():self._route(index,s,ss,t,ts,k,m,r,g)
        add(call,C,self.ordered.entry,O,K.IMPLEMENTATION,Meaning.STRUCTURE,'call-implementation')
        add(self.ordered.source_body,S,self.ordered.entry,O,K.IMPLEMENTATION,Meaning.STRUCTURE,'body-implementation')
        add(self.ports[B.ENVIRONMENT],O,self.ordered.entry,O,K.ENVIRONMENT,Meaning.DEPENDENCE,'implementation-environment')
        for role in (B.ENVIRONMENT,B.WORLD_ENTRY):
            port=self.ports[role];original=self.records[port.node].original
            add(self.view.source_port(original),S,port,O,
                K.ENVIRONMENT if role is B.ENVIRONMENT else K.STATE_INTERFACE,
                Meaning.DEPENDENCE if role is B.ENVIRONMENT else Meaning.STATE_INTERFACE,role.value.lower())
        for row in self.inputs:
            source=self.view.source_port(row.original_read_result)
            add(source,S,self.ordered.point(row.input_key),O,K.PROJECTION if row.field_path else K.ARGUMENT,
                Meaning.DEPENDENCE if row.field_path else Meaning.REFERENCE,'ordered-input',row.guards)
            previous=row.root_formal
            for read in row.read_chain:
                if previous!=read.receiver:
                    add(self.view.source_port(previous),S,self.view.source_port(read.receiver),S,
                        K.PROJECTION,Meaning.REFERENCE,'selected-field-receiver-binding',row.guards)
                content=self.view.state(read.content_access.unit).content(complete=False)
                access=content.accesses[read.content_access.ordinal]
                if access.operation!=read.content_access.operation or access.result!=read.result:
                    raise ValueError('field input cites another original content request')
                selected_state=Point(self.source.namespace,access.state_before,None)
                add(selected_state,S,self.ordered.point(row.input_key),O,K.STATE_INPUT,
                    Meaning.DEPENDENCE,'selected-field-slot-state-not-array-content-equality',row.guards)
                previous=read.result
            if not row.field_path and row.root_formal!=row.original_read_result:
                add(self.view.source_port(row.root_formal),S,source,S,K.PROJECTION,Meaning.REFERENCE,
                    'selected-parameter-binding',row.guards)
        for role in (B.WORLD_EXIT,B.RETURN_NONE):
            port=self.ports[role];original=self.records[port.node].original
            if original is not None:
                add(port,O,self.view.source_port(original),S,
                    K.STATE_INTERFACE if role is B.WORLD_EXIT else K.COMPLETION,
                    Meaning.STATE_INTERFACE if role is B.WORLD_EXIT else Meaning.VALUE,role.value.lower())
        # Control and world are interfaces, not array contents or extra calls.
        actions=[self.ordered.point('#/actions/'+str(i))for i in range(len(self.ordered._region.actions))]
        add(self.ordered.entry,O,self.ports[B.CONTROL_ENTRY],O,K.CONTROL_FLOW,Meaning.ORDER,'enter')
        if actions:
            add(self.ports[B.CONTROL_ENTRY],O,actions[0],O,K.CONTROL_FLOW,Meaning.ORDER,'first-action')
            add(actions[-1],O,self.ports[B.CONTROL_EXIT],O,K.CONTROL_FLOW,Meaning.ORDER,'normal-exit')
        add(self.ports[B.CONTROL_EXIT],O,self.ports[B.RETURN_NONE],O,K.COMPLETION,Meaning.DEPENDENCE,'normal-none-result')
        add(self.ports[B.WORLD_ENTRY],O,self.ports[B.WORLD_EXIT],O,K.STATE_INTERFACE,Meaning.STATE_INTERFACE,'world-frame-not-version-equality')
        for key in self.ordered._projection['entry_states'].values():
            add(self.ports[B.WORLD_ENTRY],O,self.ordered.point(key),O,K.STATE_INPUT,Meaning.DEPENDENCE,'entry-content-projection')
        for key in self.ordered._projection['exit_states'].values():
            add(self.ordered.point(key),O,self.ports[B.WORLD_EXIT],O,K.STATE_OUTPUT,Meaning.DEPENDENCE,'exit-content-projection')
        for port in self.ports.values():
            add(self.ordered.entry,O,port,O,K.CONTAINS,Meaning.STRUCTURE,'interface-port')

    def check(self):
        self.ordered.check()
        for provider in self.providers.values():provider.store.validate()

    def _source_scope(self,point):
        key=self._key(point)
        if key in self._scope_cache:return self._scope_cache[key]
        source=self.view.namespaces.get(point.namespace);scope=None
        if source is not None and point.node in source.nodes:
            source.ensure(point);record=source.records[point.node];anchor=source.nodes[point.node].anchor
            if anchor.template is not None:
                context=NamespacePlanView(source.unit,source.plans[anchor.template])
                region=(record.region if hasattr(record,'region') else
                    context.steps[record.owner].region if type(record)is G.Port and record.owner in context.steps else
                    record.owner if type(record)is G.Port else getattr(record,'id',None))
                if region in context.contexts:scope=context.owner(region).owner
        self._scope_cache[key]=scope;return scope

    def _selection(self,point,binding,space):
        if space is A.OrderedContextSpace.CALLER:return True,point.context
        frame=binding.source_frame if space is A.OrderedContextSpace.SOURCE else binding.ordered_frame
        result=context_pop(point.context,(frame,))
        return result.matched,result.remaining

    def _contexts(self,binding,base):
        return {A.OrderedContextSpace.CALLER:base,
            A.OrderedContextSpace.SOURCE:context_push(base,binding.source_frame),
            A.OrderedContextSpace.ORDERED:context_push(base,binding.ordered_frame)}

    def _caller_matches(self,binding,base):
        provider=self.providers[binding.provider]
        owners=provider.graphs[binding.unit].queries[binding.query].owners
        active=source_caller_scope(self.view,binding.source_frame,base)
        return active is None or active in owners

    def accepts_context(self,point):
        frames=context_frames(point.context)
        if not frames:return False
        if point.namespace==self.ordered.namespace:return frames[-1]in self.ordered_frames
        if frames[-1]in self.source_frames and self._source_scope(point)==self.function.body_scope.owner:return True
        return bool(self._inline_selections(point))

    def _inline_selections(self,point):
        """Recover outer selections from exact leaf frame suffixes."""
        rows=[];inline=getattr(self,'inline',None)
        if inline is None:return ()
        for index in inline.by_source.get(self._key(point),()):
            route=inline.routes[index];target=self.ordered.point(route.ordered_key)
            for selected,outer in inline.pop_source_context(point,target,point.context):
                frames=context_frames(outer)
                for binding_index in self.source_frames.get(frames[-1],())if frames else ():
                    binding=self.bindings[binding_index]
                    popped=context_pop(outer,(binding.source_frame,))
                    if popped.matched and self._caller_matches(binding,popped.remaining):
                        item=(binding_index,popped.remaining)
                        if item not in rows:rows.append(item)
        return tuple(rows)

    def _project_definition_edge(self,edge,binding,base):
        O=A.OrderedContextSpace.ORDERED;S=A.OrderedContextSpace.SOURCE
        contexts=self._contexts(binding,base)
        foreign=[p for p in(edge.source,edge.target)if p.namespace!=self.ordered.namespace
            and self._source_scope(p)!=self.function.body_scope.owner]
        inline=getattr(self,'inline',None)
        alternatives=((None,None),)
        if foreign:
            if (len(foreign)!=1 or edge.target.namespace!=self.ordered.namespace or inline is None):return ()
            alternatives=inline.contexts_for(foreign[0],edge.target,contexts[S])
        result=[]
        for route,leaf_context in alternatives:
            def mapped(p):
                if p.namespace==self.ordered.namespace:return replace(p,context=contexts[O])
                if self._source_scope(p)==self.function.body_scope.owner:return replace(p,context=contexts[S])
                if route is not None and self._source_scope(p)==route.body_scope:return replace(p,context=leaf_context)
                raise ValueError('definition provenance has no matched original source scope')
            evidence=tuple(replace(e,anchors=tuple(mapped(p)for p in e.anchors))for e in edge.evidence)
            conditions=tuple(replace(c,subject=None if c.subject is None else mapped(c.subject),
                operands=tuple(mapped(p)for p in c.operands))for c in edge.conditions)
            if route is not None:
                call=Point(route.frame.caller,route.frame.operation,contexts[S])
                evidence+=(Evidence(SCHEMA+'.matched-inline-origin',
                    (self.ordered.locator,self.providers[route.provider].manifest),
                    (call,mapped(edge.source),mapped(edge.target))),)
                conditions+=tuple(Condition(SCHEMA,name,call,None,(mapped(edge.source),mapped(edge.target)))for name in route.guards)
            result.append(replace(edge,source=mapped(edge.source),target=mapped(edge.target),
                evidence=evidence,conditions=conditions))
        return tuple(result)

    def _edge(self,row,contexts):
        binding=self.bindings[row.binding]
        source=replace(row.source,context=contexts[row.source_space]);target=replace(row.target,context=contexts[row.target_space])
        call=Point(binding.source_frame.caller,binding.source_frame.operation,contexts[A.OrderedContextSpace.CALLER])
        guards=tuple(Condition(SCHEMA,name,call,None,(source,target))for name in (*binding.guards,*row.guards))
        evidence=Evidence(SCHEMA+'.'+row.rule,(self.ordered.locator,binding.source_frame.target.graph,
            self.providers[binding.provider].manifest),(call,source,target))
        return Edge(source,target,row.kind,row.meaning,(evidence,),guards,Strength.MAY)

    def _local_definition(self,point,incoming,binding,base):
        rows=(self.ordered.in_edges if incoming else self.ordered.out_edges).get(self._key(point),())
        edges=[];unmapped=False
        for edge in rows:
            projected=self._project_definition_edge(edge,binding,base)
            if not projected:unmapped=True
            edges.extend(value for value in projected if (value.target if incoming else value.source)==point)
        return tuple(edges),unmapped

    def _neighbors(self,point,incoming):
        key=self._key(point)
        index=self.in_index if incoming else self.out_index
        definition=self.ordered.in_edges if incoming else self.ordered.out_edges
        if key not in index and key not in definition and point.namespace!=self.ordered.namespace:return Neighborhood()
        self.check();edges=[];reasons=set()
        for index in (self.in_index if incoming else self.out_index).get(self._key(point),()):
            row=self.routes[index];binding=self.bindings[row.binding]
            space=row.target_space if incoming else row.source_space
            selected,base=self._selection(point,binding,space)
            if not selected:continue
            if not self._caller_matches(binding,base):reasons.add('ORDERED_CALLER_SCOPE_DOES_NOT_OWN_CALLSITE');continue
            edges.append(self._edge(row,self._contexts(binding,base)))
        frames=context_frames(point.context)
        choices=([self.ordered_frames[frames[-1]]]if frames and point.namespace==self.ordered.namespace and frames[-1]in self.ordered_frames
            else self.source_frames.get(frames[-1],())if frames else ())
        for index in choices:
            binding=self.bindings[index]
            space=A.OrderedContextSpace.ORDERED if point.namespace==self.ordered.namespace else A.OrderedContextSpace.SOURCE
            selected,base=self._selection(point,binding,space)
            if not selected or not self._caller_matches(binding,base):continue
            if space is A.OrderedContextSpace.SOURCE and self._source_scope(point)!=self.function.body_scope.owner:continue
            local,unmapped=self._local_definition(point,incoming,binding,base);edges.extend(local)
            if unmapped:reasons.add('INLINED_SOURCE_PROVENANCE_NEEDS_ITS_OWN_MATCHED_ACTIVATION')
        for index,base in self._inline_selections(point):
            local,unmapped=self._local_definition(point,incoming,self.bindings[index],base);edges.extend(local)
            if unmapped:reasons.add('INLINED_SOURCE_PROVENANCE_NEEDS_ITS_OWN_MATCHED_ACTIVATION')
        if edges:reasons.add('ORDERED_ACTIVATION_IS_CONDITIONAL_PROFILE_INTERFACE_NOT_OBSERVED_INVOCATION')
        elif point.namespace==self.ordered.namespace and point.context is not None:reasons.add('ORDERED_ACTIVATION_CONTEXT_NOT_SELECTED')
        return Neighborhood(tuple(dict.fromkeys(edges)),tuple(Frontier(point,r,
            'The original call selection and ordered implementation guards remain explicit; contexts do not establish runtime instances or global content versions.')for r in sorted(reasons)))

    def incoming(self,point):return self._neighbors(point,True)
    def outgoing(self,point):return self._neighbors(point,False)

    def describe(self,point):
        if point.namespace!=self.ordered.namespace or point.node not in self.nodes:return None
        self.check();row=self.nodes[point.node]
        return replace(row,point=point,owner=replace(row.owner,context=point.context))

    def original(self,point):
        if self.describe(point)is None:raise ValueError('not an ordered boundary port')
        return self.records[point.node]

    def expand(self,point):self.original(point);return ()

    def explain(self,point):
        return dict(node=self.describe(point),original=self.original(point),matched_interfaces=tuple(self.bindings),
            input_boundaries=self.input_boundaries,runtime_invocation_proven=False,
            logical_versions_equal=False,transformation_authorized=False)

    def summary(self):
        return dict(schema=SCHEMA,bindings=len(self.bindings),routes=len(self.routes),
            boundary_ports=len(self.ports),input_routes=len(self.inputs),input_boundaries=len(self.input_boundaries),
            inline=None if getattr(self,'inline',None)is None else self.inline.summary(),
            route_kinds=dict(Counter(r.kind.value for r in self.routes)),
            call_contexts_are_alternative_implementations=True,runtime_invocations_created=0,
            logical_content_version_equality_proven=False,target_executed=False,transformation_authorized=False)


__all__=['OrderedRegionActivationProvider']
