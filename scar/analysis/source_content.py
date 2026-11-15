"""Common-graph projections of original field/index protocol requests.

The owned receiver Use, exact selectors and coarse world remain linked.
No default descriptor, alias, physical write or content equality is inferred.
"""
from collections import ChainMap, defaultdict
from copy import copy
from dataclasses import replace

from scar.analysis.program_view import _digest
from scar.analysis.source_namespace_interfaces import NamespacePlanView
from scar.ir.program import source_content as C, view as V, model as M, normal_graph as G
from scar.ir.program.dependencies import (DependencyPoint as Point, DependencyEdge,
    DependencyCondition, DependencyEvidence, DependencyFrontier, DependencyNeighborhood,
    DependencyKind as K, DependencyMeaning as Meaning)
from scar.ir.program.ids import LocationID, ValueID
from scar.ir.program.normal_targets import GlobalPort


class SourceContentProvider:
    def __init__(self, state, *, register=True):
        self.bindings=state;self.view=state.view;self.source=state.source;self.namespace=state.namespace
        self.addresses={};self.projections={};self.accesses=[];self.operation_accesses=defaultdict(list)
        self.address_accesses=defaultdict(list)
        self.in_edges=defaultdict(list);self.out_edges=defaultdict(list)
        self.analyzed_operations=set();self._building=set();self._failures={}
        self._interfaces=None;self.complete=False
        if register:self.view.register(self)

    def _identify(self, context, step, interface):
        operation=self.source._node(context.key,step.id)
        address=self._address(interface.receiver,interface.selector)
        before,after=context.worlds(step)
        previous=self._project(address,before);following=self._project(address,after)
        use=self.source.port(context.emitter.port(step.id,G.PortKind.ARGUMENT,interface.receiver_argument))
        record=C.ContentAccessTemplate(operation.node,context.site(step),context.owner(step.region).owner,
            C.ContentRequestKind(interface.kind),address.id,use.node,interface.payload,interface.result,
            before,after,previous.id,following.id,interface.attribute_fact)
        ordinal=len(self.accesses);self.accesses.append(record);self.operation_accesses[operation.node].append(ordinal)
        self.address_accesses[address.id].append(ordinal)
        location=Point(self.namespace,address.id)
        selected=DependencyCondition('scar.source-content.request','SELECTED_PROTOCOL_ADDRESS_INSTANCE',
            operation,None,(use,location))
        self._edge(use,location,K.ENVIRONMENT,Meaning.STATE_INTERFACE,(selected,))
        if type(address.selector) is C.ItemSelector:keys=(address.selector.key,)
        elif type(address.selector) is C.SliceBoundsSelector:keys=(address.selector.lower,address.selector.upper)
        else:keys=()
        for key_port in keys:
            if type(key_port) is GlobalPort:
                self._edge(self.view.source_port(key_port),location,K.PROJECTION,Meaning.DEPENDENCE,(selected,))
        self._edge(Point(self.namespace,previous.id),operation,K.STATE_INPUT,Meaning.STATE_INTERFACE,(selected,))
        completed=DependencyCondition('scar.source-content.request','ORIGINAL_NORMAL_COMPLETION_SELECTED',
            self.view.source_port(after),after.port.ordinal,(operation,Point(self.namespace,following.id)))
        self._edge(operation,Point(self.namespace,following.id),K.STATE_OUTPUT,Meaning.STATE_INTERFACE,(selected,completed))

    def ensure_operation(self, operation):
        """Publish contents only after the operation's binding request exists."""
        from scar.analysis.source_content_interfaces import ContentInterfaceIndex
        from scar.analysis.source_state import _AppendBuffer,_publish_edges
        point,context,step=self.bindings._operation_context(operation);identity=point.node
        self.bindings.ensure_operation(point,include_content=False)
        if identity in self.bindings._building or identity in self._building:return ()
        if identity in self._failures:raise ValueError(self._failures[identity])
        if identity in self.analyzed_operations:return tuple(self.operation_accesses.get(identity,()))
        self._building.add(identity)
        try:
            if self._interfaces is None:self._interfaces=ContentInterfaceIndex(self.source)
            interface=self._interfaces.identify(context,step)
            staged=copy(self);staged.addresses=ChainMap({},self.addresses)
            staged.projections=ChainMap({},self.projections);staged.accesses=_AppendBuffer(self.accesses)
            staged.operation_accesses=defaultdict(list);staged.address_accesses=defaultdict(list)
            staged.in_edges=defaultdict(list);staged.out_edges=defaultdict(list)
            if interface is not None:staged._identify(context,step,interface)
            self.check(point)
            self.addresses.update(staged.addresses.maps[0]);self.projections.update(staged.projections.maps[0])
            self.accesses.extend(staged.accesses.added)
            for key,rows in staged.operation_accesses.items():self.operation_accesses[key].extend(rows)
            for key,rows in staged.address_accesses.items():self.address_accesses[key].extend(rows)
            _publish_edges(self,staged);self.analyzed_operations.add(identity)
            request=self.bindings.requests[identity]
            missing=tuple(rule for rule in request.missing_rules if rule not in
                ('SOURCE_CONTENT_EFFECT_REQUEST_NOT_ANALYZED','STATE_ADDRESS_PROJECTION_NOT_IMPLEMENTED'))
            if interface is None and 'SOURCE_CONTENT_EFFECT_REQUEST_NOT_ANALYZED'in request.missing_rules:
                missing=(*missing,'STATE_ADDRESS_PROJECTION_NOT_IMPLEMENTED')
            self.bindings.requests[identity]=replace(request,missing_rules=missing)
        except Exception as error:
            self._failures[identity]='content operation identification failed: '+str(error)
            request=self.bindings.requests[identity]
            self.bindings.requests[identity]=replace(request,missing_rules=(*request.missing_rules,'SOURCE_CONTENT_EFFECT_ANALYSIS_FAILED'))
            raise
        finally:self._building.remove(identity)
        return tuple(self.operation_accesses.get(identity,()))

    def ensure_regions(self, key, regions):
        return self.bindings.ensure_regions(key,regions,include_content=True)

    def ensure_all(self):
        self.bindings.ensure_all(include_content=True)
        return self

    def _address(self,receiver,selector):
        identity=LocationID(_digest(('scar.source-content-address.v1',self.source.basis,receiver,selector)))
        if identity not in self.addresses:
            parameters=(C.ContentInstanceParameter.ACTIVATION,C.ContentInstanceParameter.OWNER_REFERENCE_OCCURRENCE)
            if type(selector) in (C.ItemSelector,C.SliceBoundsSelector):
                parameters+= (C.ContentInstanceParameter.SELECTOR_REFERENCE_OCCURRENCES,)
            self.addresses[identity]=C.ContentAddressTemplate(identity,receiver,selector,parameters)
        return self.addresses[identity]

    def _project(self,address,world):
        source=self.view.source_port(world)
        if self.view.get(source).value_type.kind is not M.ValueKind.STATE:
            raise ValueError('content projection requires the original state world')
        identity=ValueID(_digest(('scar.source-content-state.v1',self.source.basis,address.id,world)))
        if identity not in self.projections:
            record=C.ContentStateProjection(identity,address.id,world);self.projections[identity]=record
            value=Point(self.namespace,identity);location=Point(self.namespace,address.id)
            selected=DependencyCondition('scar.source-content.projection','SAME_SELECTED_ADDRESS_INSTANCE',source,None,(location,value))
            self._edge(location,value,K.STATE_LOCATION,Meaning.STRUCTURE)
            self._edge(source,value,K.PROJECTION,Meaning.STATE_INTERFACE,(selected,))
        return self.projections[identity]

    def _edge(self,source,target,kind,meaning,conditions=()):
        edge=DependencyEdge(source,target,kind,meaning,
            (DependencyEvidence('scar.source-content.interface.v1',(self.source.locator,),(source,target)),),tuple(conditions))
        if edge not in self.out_edges[source.node]:
            self.out_edges[source.node].append(edge);self.in_edges[target.node].append(edge)

    def check(self,point=None):self.bindings.check(point)

    def describe(self,point):
        if point.namespace!=self.namespace:return None
        self.check(point)
        if point.node in self.addresses:
            return V.ProgramNode(point,V.ProgramEntityKind.STATE_LOCATION,V.ProgramNodeRole.STATE_LOCATION,None,None)
        if point.node in self.projections:
            record=self.projections[point.node]
            return V.ProgramNode(point,V.ProgramEntityKind.VALUE,V.ProgramNodeRole.CORE_VALUE,None,
                M.ValueType(M.ValueKind.STATE,location=record.address))
        return None

    def original(self,point):
        if self.describe(point) is None:raise ValueError('content endpoint is not admitted')
        return self.addresses[point.node] if point.node in self.addresses else self.projections[point.node]

    def expand(self,point):
        self.original(point)
        return tuple(Point(self.namespace,p.id,point.context) for p in self.projections.values() if p.address==point.node)

    def explain(self,point):
        record=self.original(point)
        address=point.node if point.node in self.addresses else record.address
        accesses=tuple(self.accesses[i] for i in self.address_accesses[address])
        return dict(node=self.describe(point),original=self.original(point),source_material=self.source.material.result_artifact,
            accesses=accesses,attribute_facts=tuple((a.attribute_fact,self.source.facts.attributes[a.attribute_fact])
                for a in accesses if a.attribute_fact is not None),
            protocol_effects_closed=False,physical_address_selected=False,logical_version_equality_proven=False)

    def _neighbors(self,point,incoming):
        if point.namespace!=self.namespace:return DependencyNeighborhood()
        self.check(point)
        from scar.ir.program.ids import OperationID
        if type(point.node)is OperationID and point.node in self.source.nodes:
            self.ensure_operation(point)
        rows=self.in_edges[point.node] if incoming else self.out_edges[point.node]
        edges=tuple(replace(edge,source=replace(edge.source,context=point.context),target=replace(edge.target,context=point.context),
            conditions=tuple(replace(c,subject=None if c.subject is None else replace(c.subject,context=point.context),
                operands=tuple(replace(p,context=point.context) for p in c.operands)) for c in edge.conditions)) for edge in rows)
        frontiers=[]
        if not self.complete and (type(point.node)is LocationID or point.node in self.projections
                or point.node in self.source.nodes and type(point.node)is not OperationID):
            frontiers.append(DependencyFrontier(point,'SOURCE_CONTENT_ADJACENCY_PARTIAL',
                'Unrequested operations may share this address, selector or world; ensure_all() explicitly projects the original unit.'))
        if point.node in self.addresses or point.node in self.projections or point.node in self.operation_accesses:
            frontiers.append(DependencyFrontier(point,'CONTENT_PROTOCOL_AND_INSTANCE_OPEN',
                'Request/selector/world are retained; descriptor, item method, alias and actual content versions need their contracts.'))
        return DependencyNeighborhood(edges,tuple(frontiers))

    def incoming(self,point):return self._neighbors(point,True)
    def outgoing(self,point):return self._neighbors(point,False)
