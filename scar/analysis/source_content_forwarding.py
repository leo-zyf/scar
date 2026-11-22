"""Prove a bounded builtin-slot reference rule under explicit runtime guards.

Only original function control, reference paths, constant selectors and typed
container producers are used. Calls, writes, branches, loops and resumptions
inside the supported normal window are refused. Ordinary local reads reuse
the existing original C2/F4 effect witnesses, including their unresolved Q.
The resulting graph edge stays MAY and conditional, never logical equality.
"""
from collections import Counter,defaultdict
from dataclasses import replace
from types import SimpleNamespace as NS

from scar.analysis.source_frame_binding_effects import LocalReadEffectContext
from scar.analysis.source_name_access_anchors import anchor_name_accesses
from scar.analysis.source_references import ReferenceSlicer, source_port
from scar.analysis.source_value_flow import project_value_flow
from scar.analysis.source_templates import _json
from scar.ir.program import normal_graph as G, normal_graph as T, source_productions as P
from scar.ir.program import data_materials as C, data_materials as F
from scar.ir.program.control import LanguageCompletionKind as Completion
from scar.ir.program.structured import PythonProtocolKind as Protocol
from scar.ir.program.source_value_routes import ReferenceOriginKind
from scar.ir.program.dependencies import (DependencyPoint as Point,DependencyEdge as Edge,
    DependencyCondition as Condition,DependencyEvidence as Evidence,DependencyFrontier as Frontier,
    DependencyNeighborhood as Neighborhood,DependencyKind as K,DependencyMeaning as Meaning,
    DependencyStrength as Strength)
from scar.ir.literals import PythonLiteral
from scar.ir.record_codec import encode


SCHEMA='scar.source-content-reference-forwarding'
_GUARDS=(
    'Q_ORIGINAL_SOURCE_FUNCTION_AND_SELECTED_LITERAL_PROTOCOL_CODE_CURRENT',
    'Q_WRITE_AND_READ_SELECT_SAME_PRODUCER_INSTANCE_IN_SAME_ACTIVATION',
    'Q_RECEIVER_AND_SELECTED_SLOT_HAVE_NO_ESCAPED_OR_CONCURRENT_WRITER_IN_WINDOW',
    'Q_SELECTED_REFERENCE_PATHS_ARE_CURRENT_BINDINGS_NOT_MAY_ALTERNATIVES',
    'Q_STANDARD_CPYTHON_BUILTIN_SLOT_SEMANTICS_NOT_CUSTOM_DESCRIPTOR_OR_ITEM_CALLBACK',
    'Q_SET_RELEASE_OF_REPLACED_VALUE_HAS_NO_REENTRANT_FINALIZER_MUTATION',
    'Q_NO_TRACE_PROFILE_MONITOR_SIGNAL_ASYNC_REENTRY_MUTATES_RECEIVER_OR_SLOT',
    'Q_REQUIRED_NORMAL_COMPLETIONS_AND_EXCEPTION_BEHAVIOR_PRESERVED',
)


class _Refuse(ValueError):
    def __init__(self,reason,site=None):
        self.reason,self.site=reason,site
        super().__init__(reason)


def _normal_route(flow,*,max_states=20000,max_stack=128):
    """Yield original control states without choosing branches or unrolling."""
    controls=defaultdict(list)
    for edge in flow.edges:
        if edge.kind is G.EdgeKind.CONTROL:controls[edge.source].append(edge)
    port=G.Port(flow.body.template,flow.body.region,G.PortKind.ENTER)
    stack=();seen=set();incoming=None
    while (port,stack) not in seen:
        if len(seen)>=max_states:raise _Refuse('CONTROL_WINDOW_STATE_BUDGET_EXCEEDED')
        seen.add((port,stack))
        yield port,stack,incoming
        following=[]
        for edge in controls.get(port,()):
            if edge.transition is G.Transition.RETURN and (not stack or stack[-1]!=edge.site):continue
            following.append(edge)
        if len(following)!=1:raise _Refuse('NORMAL_CONTROL_ROUTE_NOT_UNIQUE_OR_READ_UNREACHABLE',
            T.GlobalPort(flow.source_function.unit,port))
        edge=following[0]
        if edge.condition is not None or edge.transition in (G.Transition.BACKEDGE,G.Transition.RESUME):
            raise _Refuse('BRANCH_LOOP_OR_RESUMPTION_OUTSIDE_FORWARDING_PROFILE',
                T.GlobalPort(flow.source_function.unit,port))
        if edge.transition is G.Transition.CALL:
            if edge.site is None or len(stack)>=max_stack:raise _Refuse('ORIGINAL_RECIPE_CALL_STACK_NOT_BOUNDED')
            stack=(*stack,edge.site)
        elif edge.transition is G.Transition.RETURN:stack=stack[:-1]
        incoming=edge;port=edge.target
    raise _Refuse('CYCLIC_NORMAL_CONTROL_ROUTE_OUTSIDE_FORWARDING_PROFILE')


def _control_window(flow,write,read,read_witnesses,*,max_states=20000,max_stack=128):
    """Certify one finite normal route using existing local-READ witnesses."""
    operations={row.entry:row for row in flow.operations}
    qualified={w.site.port:w for w in read_witnesses}
    write_exit=G.Port(write.port.template,write.port.owner,G.PortKind.EXIT,
        completion=Completion.FALLTHROUGH)
    path=[];start=None;reads=[];cleanups=[];ports=set()
    for port,_,incoming in _normal_route(flow,max_states=max_states,max_stack=max_stack):
        if port in ports:raise _Refuse('REPEATED_SOURCE_PORT_REQUIRES_INVOCATION_SELECTION',
            T.GlobalPort(flow.source_function.unit,port))
        ports.add(port)
        if incoming is not None:path.append(incoming)
        if port==write_exit:
            if start is not None:raise _Refuse('WRITE_OCCURS_MORE_THAN_ONCE_IN_SUPPORTED_ROUTE',write)
            start=len(path)
        if port==read.port:
            if start is None:raise _Refuse('READ_NOT_DOMINATED_BY_SELECTED_WRITE_IN_SUPPORTED_ROUTE',read)
            return tuple(path),start,tuple(reads),tuple(cleanups)
        if start is not None and port in operations:
            instruction=operations[port].instruction
            if type(instruction)in (P.DefineLiteral,P.ReadCommonConstant):pass
            elif type(instruction)is P.ConstructValue and instruction.construction is P.ConstructionKind.LITERAL:pass
            elif (type(instruction)is P.ApplyPythonProtocol
                    and instruction.protocol is Protocol.FRAME_RELEASE_TEMPORARY):
                cleanups.append(T.GlobalPort(flow.source_function.unit,port))
            elif port in qualified:reads.append(qualified[port])
            else:raise _Refuse('INTERVENING_OPERATION_NOT_CERTIFIED_REFERENCE_READ_OR_LITERAL',
                T.GlobalPort(flow.source_function.unit,port))


class SourceContentForwardingProvider:
    """Explicit SET/GET proof queries; no automatic all-pairs graph expansion."""
    relation_kinds=frozenset((K.PROJECTION,))

    def __init__(self,lineage,*,register=True):
        self.lineage=lineage;self.content=lineage.content;self.source=lineage.source
        self.view=lineage.view;self.namespace=lineage.namespace
        self.proofs={};self.refusals={};self.in_edges=defaultdict(list);self.out_edges=defaultdict(list)
        self._flows={};self._anchors=None;self._effects=None
        self.discovery_boundaries={};self.discovery_complete=False
        self._producer_cache={};self._selector_cache={}
        snapshot=NS(fact_store=self.source.material.fact_store,rows=lambda *args,**kwargs:(),
            validate_parents=self.source.material.validate_parents)
        # Local declaration paths only: no unselected Python CALL target is
        # silently admitted by the reference slicer.
        self._slicer=ReferenceSlicer(snapshot)
        facts=self.source.facts
        self._containers={source_port(facts,row.value):(i,row) for i,row in enumerate(facts.containers)}
        self._constants={source_port(facts,row.value):(i,row) for i,row in enumerate(facts.constants)}
        if register:self.view.register(self)

    def check(self,point=None):
        self.lineage.check(point)
        if self._effects is not None:self._effects.check()

    def _slice(self,port):
        return self._slicer.slice(port,max_states=20000,max_call_depth=128)

    @staticmethod
    def _leaves(path):
        return set(range(len(path.states)))-{edge.downstream for edge in path.transitions}

    def _selector(self,selector):
        if selector in self._selector_cache:return self._selector_cache[selector]
        if type(selector)is not C.ItemSelector:raise _Refuse('ONLY_SCALAR_ITEM_SLOTS_SUPPORTED')
        if type(selector.key)is PythonLiteral:return selector.key,None
        path=self._slice(selector.key)
        leaves=self._leaves(path);values=[]
        for index in leaves:
            endpoint=path.states[index].port
            if endpoint not in self._constants:raise _Refuse('SELECTOR_HAS_NO_EXACT_ORIGINAL_LITERAL_PRODUCER')
            values.append(self._constants[endpoint][1].literal)
        if not values or any(value!=values[0] for value in values):
            raise _Refuse('SELECTOR_LITERAL_ORIGIN_NOT_UNIQUE')
        result=values[0],path;self._selector_cache[selector]=result
        return result

    def _producer(self,receiver):
        if receiver in self._producer_cache:return self._producer_cache[receiver]
        path=self._slice(receiver);leaves=self._leaves(path)
        origins={path.states[index].port for index in leaves}
        if len(origins)!=1 or not origins<=self._containers.keys():
            raise _Refuse('RECEIVER_HAS_NO_UNIQUE_ORIGINAL_CONTAINER_PRODUCER')
        value=next(iter(origins));ordinal,_=self._containers[value]
        self.view.source_port(value)
        operation=self.source._node(value.port.template,value.port.owner)
        self.source.ensure(operation);step=self.source.records[operation.node]
        instruction=step.instruction
        if type(instruction)is not P.ApplyPythonProtocol:raise _Refuse('CONTAINER_PRODUCER_NOT_EXACT_BUILTIN_DISPLAY_PROTOCOL')
        if instruction.protocol is Protocol.BUILD_MAP:kind=F.BuiltinSlotKind.DICT
        elif instruction.protocol is Protocol.BUILD_DISPLAY_LIST:kind=F.BuiltinSlotKind.LIST
        else:raise _Refuse('CONTAINER_PRODUCER_NOT_EXACT_DICT_OR_LIST_DISPLAY')
        site=T.GlobalPort(self.source.unit,G.Port(value.port.template,step.id,G.PortKind.ENTER))
        result=F.SourceContainerProducer(value,site,self._slicer.fact(self.source.facts,'containers',ordinal),
            instruction.protocol.value,kind),path
        self._producer_cache[receiver]=result
        return result

    def _source_discharges(self,producer,selector,flow,path,write,read_witnesses,cleanups):
        """Discharge only explicit literal/ownership facts, preserving Q.

        The source-selected producer and current binding guards still apply.
        This does not upgrade conditional name paths to runtime identities.
        """
        facts=['EXACT_BUILTIN_DISPLAY_PRODUCER_RECORDED',
            'SELECTOR_HAS_EXACT_SAFE_PRIMITIVE_LITERAL_ORIGIN',
            'UNIQUE_BALANCED_NORMAL_ROUTE_WITH_SET_BEFORE_GET',
            'NO_INTERVENING_CALL_OTHER_WRITE_BRANCH_LOOP_OR_RESUMPTION']
        remaining=list(cleanups);positions={edge.source:i for i,edge in enumerate(path)}
        operations={op.entry:op for op in flow.operations}
        producer_exit=G.Port(producer.site.port.template,producer.site.port.owner,G.PortKind.EXIT,
            completion=Completion.FALLTHROUGH)
        begin,end=positions.get(producer_exit),positions.get(write.site.port)
        qualified={w.site.port for w in read_witnesses};prefix=True;prefix_cleanups=[]
        function=next(f for f in self.source.facts.functions
            if source_port(self.source.facts,f.value)==flow.source_function)
        captured={c.binding for c in self._effects.bindings.captures}|{c.binding for c in self._effects.bindings.relays}
        for entry,op in operations.items():
            index=positions.get(entry)
            if begin is None or end is None:prefix=False;break
            if index is None or not begin<index<end:continue
            ins=op.instruction
            if entry in qualified or type(ins)in(P.DefineLiteral,P.ReadCommonConstant):continue
            if type(ins)is P.ConstructValue and ins.construction is P.ConstructionKind.LITERAL:continue
            if type(ins)is P.ApplyPythonProtocol and ins.protocol in (
                    Protocol.FRAME_HOLD_TEMPORARY,Protocol.FRAME_RELEASE_TEMPORARY):
                if ins.protocol is Protocol.FRAME_RELEASE_TEMPORARY:
                    prefix_cleanups.append(T.GlobalPort(self.source.unit,entry))
                continue
            if type(ins)is P.AccessBinding and ins.action is P.BindingActionKind.WRITE:
                port=self._effects.bindings.ports[ins.port]
                target=None if port.write_target is None else self._effects.bindings.interfaces[port.write_target]
                binding=None if target is None or target.binding is None else self._effects.bindings.bindings[target.binding]
                if (target is not None and target.kind=='LEXICAL_BINDING' and binding is not None
                        and target.owner==function.environment.scope
                        and binding.domain=='LOCAL_BINDING' and not binding.cell_required
                        and target.binding not in captured):continue
            prefix=False;break
        remove_old_release=False;remove_key_guard=False
        if prefix and begin is not None and end is not None:
            facts.append('FRESH_PRODUCER_PREFIX_HAS_NO_CALL_OR_HEAP_NONLOCAL_OR_CAPTURE_STORE')
            remaining.extend(prefix_cleanups)
            _,container=self._containers[producer.value]
            previous=None;absent=False
            if producer.kind is F.BuiltinSlotKind.DICT:
                keys=[]
                try:
                    keys=[(self._selector(C.ItemSelector(source_port(self.source.facts,key)))[0],
                        source_port(self.source.facts,value))for key,value in container.pairs]
                    remove_key_guard=all(type(key.to_python())in(str,int)for key,_ in keys)
                    hits=[value for key,value in keys if key==selector]
                    absent=not hits;previous=None if absent else hits[-1]
                except _Refuse:pass
            else:
                index=selector.to_python();index=index if index>=0 else len(container.elements)+index
                if not 0<=index<len(container.elements):raise _Refuse('FRESH_LIST_INDEX_OUT_OF_INITIAL_BOUNDS')
                previous=source_port(self.source.facts,container.elements[index])
                facts.append('NO_SOURCE_OPERATION_CHANGES_LIST_LENGTH_BEFORE_SELECTED_GET')
            if absent:
                remove_old_release=True;facts.append('SELECTED_FRESH_DICT_KEY_ABSENT_NO_REPLACED_VALUE_TO_FINALIZE')
            elif previous is not None:
                try:
                    literal,_=self._selector(C.ItemSelector(previous));literal.to_python()
                    remove_old_release=True;facts.append('REPLACED_INITIAL_SLOT_VALUE_IS_EXACT_NONFINALIZING_BUILTIN_LITERAL')
                except _Refuse:pass
            if remove_key_guard:facts.append('ALL_RESIDENT_DICT_KEYS_HAVE_EXACT_PRIMITIVE_LITERAL_ORIGINS')
        # A released reference still owned by the selected builtin SET cannot
        # reach refcount zero. Its contents and external aliases remain open.
        payload_path=self._slice(write.payload)
        payload_roots={payload_path.states[i].port for i in self._leaves(payload_path)}
        write_exit=G.Port(write.site.port.template,write.site.port.owner,G.PortKind.EXIT,
            completion=Completion.FALLTHROUGH)
        completed=positions.get(write_exit)
        unresolved=[]
        for site in dict.fromkeys(remaining):
            op=operations[site.port]
            endpoint=T.GlobalPort(self.source.unit,op.inputs[0].port)
            released=self._slice(endpoint);roots={released.states[i].port for i in self._leaves(released)}
            literal=bool(roots) and all(root in self._constants for root in roots)
            if literal:facts.append('FRAME_RELEASE_VALUE_IS_EXACT_NONFINALIZING_BUILTIN_LITERAL')
            elif (roots and roots==payload_roots and completed is not None
                    and positions.get(site.port,-1)>completed):
                facts.append('FRAME_RELEASE_PAYLOAD_REFERENCE_RETAINED_BY_SELECTED_BUILTIN_SET')
            else:unresolved.append(site)
        return tuple(dict.fromkeys(facts)),tuple(unresolved),remove_old_release,remove_key_guard

    def _function(self,scope):
        matches=[(i,f) for i,f in enumerate(self.source.facts.functions)
            if f.body_scope is not None and f.body_scope.owner==scope]
        if len(matches)!=1:raise _Refuse('ONE_ORIGINAL_FUNCTION_ACTIVATION_SCOPE_REQUIRED')
        ordinal,function=matches[0]
        if function.environment.kind is not P.DeferredBodyKind.FUNCTION:
            raise _Refuse('DEFERRED_FUNCTION_PROTOCOL_OUTSIDE_FORWARDING_PROFILE')
        value=source_port(self.source.facts,function.value)
        if value not in self._flows:
            material=self.source.material
            flow=project_value_flow(material,value,T.Body(self.source.unit,function.template,function.body))
            if self._anchors is None:self._anchors=anchor_name_accesses(self.view.catalog,self.source.unit,material=material)
            if self._effects is None:self._effects=LocalReadEffectContext(material)
            witnesses=self._effects.witnesses(material,self._anchors,function,scope,flow)
            self._flows[value]=(flow,witnesses)
        return self._slicer.fact(self.source.facts,'functions',ordinal),self._flows[value]

    def prove(self,write_ref,read_ref):
        self.check();write=self.lineage.access(write_ref);read=self.lineage.access(read_ref)
        key=write.operation,read.operation
        if key in self.proofs:return self.proofs[key],None
        if key in self.refusals:return None,self.refusals[key]
        try:
            if write.kind is not C.ContentRequestKind.SET or read.kind is not C.ContentRequestKind.GET:
                raise _Refuse('ORIGINAL_SET_AND_GET_REQUESTS_REQUIRED')
            if write.payload is None or read.result is None:raise _Refuse('ORIGINAL_PAYLOAD_AND_RESULT_REQUIRED')
            if write.scope!=read.scope:raise _Refuse('CROSS_FUNCTION_ACTIVATION_FORWARDING_OUTSIDE_PROFILE')
            wa,ra=self.content.addresses[write.address],self.content.addresses[read.address]
            wk,ws=self._selector(wa.selector);rk,rs=self._selector(ra.selector)
            value=wk.to_python()
            if wk!=rk or type(value)not in (str,int):raise _Refuse('SAME_EXACT_SAFE_STR_OR_INT_SELECTOR_REQUIRED')
            wp,wpath=self._producer(wa.receiver);rp,rpath=self._producer(ra.receiver)
            if wp!=rp:raise _Refuse('WRITE_AND_READ_LACK_SAME_ORIGINAL_CONTAINER_PRODUCER')
            if wp.kind is F.BuiltinSlotKind.LIST and type(value)is not int:
                raise _Refuse('EXACT_INTEGER_LIST_INDEX_REQUIRED')
            function,(flow,witnesses)=self._function(write.scope)
            path,start,used,cleanups=_control_window(flow,write.site,read.site,witnesses)
            static_facts,cleanups,old_release_safe,keys_safe=self._source_discharges(
                wp,wk,flow,path,write,witnesses,cleanups)
            source_guards=tuple(dict.fromkeys(q for witness in used for q in witness.conditions))
            runtime=(*(q for q in _GUARDS if not old_release_safe
                    or q!='Q_SET_RELEASE_OF_REPLACED_VALUE_HAS_NO_REENTRANT_FINALIZER_MUTATION'),
                *(('Q_DICT_RESIDENT_KEYS_HAVE_NO_CUSTOM_HASH_OR_EQUALITY_CALLBACKS',)
                    if wp.kind is F.BuiltinSlotKind.DICT and not keys_safe else ()),
                *(('Q_ORIGINAL_FRAME_RELEASE_TEMPORARY_HAS_NO_FINALIZER_REENTRY_OR_SLOT_MUTATION',)
                    if cleanups else ()),
                *source_guards)
            guard=F.BuiltinSlotForwardingGuard(wp.kind,wp.value,wa.receiver,ra.receiver,wk,runtime)
            control=F.SourceForwardingControlWindow(flow.source_function,function,path,start,
                tuple(w.site for w in used),tuple(w.name_fact for w in used),
                tuple(dict.fromkeys(m for w in used for m in w.materials)),source_guards,cleanups)
            proof=F.ContentReferenceForwardingProof(write_ref,read_ref,write.payload,read.result,wp,wk,
                (wpath,rpath),tuple(p for p in (ws,rs)if p is not None),control,guard,static_facts,
                tuple(q for q in runtime if q in (
                    'Q_ORIGINAL_SOURCE_FUNCTION_AND_SELECTED_LITERAL_PROTOCOL_CODE_CURRENT',
                    'Q_WRITE_AND_READ_SELECT_SAME_PRODUCER_INSTANCE_IN_SAME_ACTIVATION',
                    'Q_SELECTED_REFERENCE_PATHS_ARE_CURRENT_BINDINGS_NOT_MAY_ALTERNATIVES',
                    'Q_DICT_RESIDENT_KEYS_HAVE_NO_CUSTOM_HASH_OR_EQUALITY_CALLBACKS')),
                tuple(q for q in runtime if q not in (
                    'Q_ORIGINAL_SOURCE_FUNCTION_AND_SELECTED_LITERAL_PROTOCOL_CODE_CURRENT',
                    'Q_WRITE_AND_READ_SELECT_SAME_PRODUCER_INSTANCE_IN_SAME_ACTIVATION',
                    'Q_SELECTED_REFERENCE_PATHS_ARE_CURRENT_BINDINGS_NOT_MAY_ALTERNATIVES',
                    'Q_DICT_RESIDENT_KEYS_HAVE_NO_CUSTOM_HASH_OR_EQUALITY_CALLBACKS')))
            self.check();self.proofs[key]=proof;self._attach(proof)
            return proof,None
        except _Refuse as reason:
            refused=F.ContentForwardingRefusal(write_ref,read_ref,reason.reason,reason.site)
            self.refusals[key]=refused
            return None,refused

    def _attach(self,proof):
        source=self.view.source_port(proof.payload);target=self.view.source_port(proof.result)
        read=Point(self.namespace,proof.read.operation,None)
        ports=(proof.producer.value,proof.guard.write_receiver,proof.guard.read_receiver,
            proof.payload,proof.result)
        operands=tuple(self.view.source_port(port) for port in ports)
        guards=tuple(Condition(SCHEMA,name,read,None,operands) for name in proof.guard.contracts)
        edge=Edge(source,target,K.PROJECTION,Meaning.REFERENCE,
            (Evidence(SCHEMA+'.builtin-slot-theorem',(self.source.locator,),(read,source,target)),),guards,Strength.MAY)
        self.out_edges[source.node].append(edge);self.in_edges[target.node].append(edge)

    def discover(self):
        """Group original producers/selectors and match writes in control order.

        One balanced route per source scope supplies the order, independent
        of the parent's demand-ordered access table. Only the most recent SET
        for each producer/key is considered; there is no global pair product.
        Every unsupported access retains its concrete original boundary.
        """
        self.lineage.ensure_all()
        if self.discovery_complete:return self
        scopes=defaultdict(dict)
        for ordinal,access in enumerate(self.content.accesses):
            reference=self.lineage.access_ref(ordinal)
            try:
                if access.kind not in (C.ContentRequestKind.SET,C.ContentRequestKind.GET):
                    raise _Refuse('DISCOVERY_PROFILE_REQUIRES_SCALAR_SET_OR_GET')
                address=self.content.addresses[access.address]
                selector,_=self._selector(address.selector)
                if type(selector.to_python())not in(str,int):raise _Refuse('SAFE_PRIMITIVE_SELECTOR_REQUIRED')
                producer,_=self._producer(address.receiver)
                signature=producer.value,_json(encode(selector))
                scopes[access.scope][access.site.port]=(reference,access,signature)
            except _Refuse as error:
                self.discovery_boundaries[access.operation]=F.ContentForwardingDiscoveryBoundary(
                    reference,error.reason,error.site)
        for scope,targets in scopes.items():
            visited=set();last_write={}
            try:
                _,(flow,_)=self._function(scope)
                for port,_,_ in _normal_route(flow):
                    selected=targets.get(port)
                    if selected is None:continue
                    reference,access,signature=selected;visited.add(port)
                    if access.kind is C.ContentRequestKind.SET:last_write[signature]=reference
                    elif signature in last_write:self.prove(last_write[signature],reference)
                    else:self.discovery_boundaries[access.operation]=F.ContentForwardingDiscoveryBoundary(
                        reference,'NO_PRECEDING_SUPPORTED_SET_ON_ORIGINAL_NORMAL_ROUTE',access.site)
                    if len(visited)==len(targets):break
            except _Refuse as error:
                for port,(reference,access,_) in targets.items():
                    if port not in visited:
                        self.discovery_boundaries[access.operation]=F.ContentForwardingDiscoveryBoundary(
                            reference,error.reason,error.site)
        self.discovery_complete=True
        self.check()
        return self

    def ensure_all(self):
        return self.discover()

    def _neighbors(self,point,incoming):
        if point.namespace!=self.namespace:return Neighborhood()
        self.check(point)
        rows=(self.in_edges if incoming else self.out_edges).get(point.node,())
        edges=tuple(replace(edge,source=replace(edge.source,context=point.context),
            target=replace(edge.target,context=point.context),conditions=tuple(replace(c,
                subject=replace(c.subject,context=point.context),
                operands=tuple(replace(p,context=point.context)for p in c.operands))for c in edge.conditions))for edge in rows)
        frontiers=()if not rows else (Frontier(point,'BUILTIN_SLOT_FORWARDING_RUNTIME_CONTRACTS_NOT_DISCHARGED',
            'Checked source/control theorem requires exact builtin protocol, producer-instance/alias isolation and callback/exception guards; no unconditional rewrite or logical version equality follows.'),)
        return Neighborhood(edges,frontiers)

    def incoming(self,point):return self._neighbors(point,True)
    def outgoing(self,point):return self._neighbors(point,False)

    def summary(self):
        return dict(schema=SCHEMA,conditional_reference_proofs=len(self.proofs),refusals=len(self.refusals),
            automatic_discovery_run=self.discovery_complete,
            discovery_boundaries=dict(Counter(row.reason for row in self.discovery_boundaries.values())),
            runtime_contracts_discharged=False,unconditional_forwarding_authorized=False,
            logical_version_equality_proven=False,target_executed=False,first_stage_complete=False)


__all__=['SourceContentForwardingProvider']
