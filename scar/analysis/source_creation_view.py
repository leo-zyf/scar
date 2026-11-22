"""Source creation view."""

from collections import Counter, defaultdict
from dataclasses import replace, dataclass
from threading import Lock
from scar.analysis.source_function_creations import SourceFunctionCreationStore
from scar.analysis.source_value_flow import _artifact
from scar.ir.program import normal_graph as G
from scar.ir.program.ids import OperationID, UseID, ValueID
from scar.ir.program.dependencies import (
    DependencyCondition as Condition,
    DependencyEdge as Edge,
    DependencyEvidence as Evidence,
    DependencyKind as Kind,
    DependencyMeaning as Meaning,
    DependencyStrength as Strength,
    DependencyNeighborhood as Neighborhood,
    DependencyPoint as Point,
    DependencyFrame as Frame,
    DependencyKind as K,
    DependencyFrontier as Frontier,
    context_push,
    context_pop,
    context_is_closed,
)
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.program.function_sites import ReturnedCallableProvenance
from scar.analysis.source_call_view import source_call_site, source_active_scope


# Initial factory operand roles over existing F4 Uses and operations.
class SourceFunctionCreationProvider:
    """Role edges are creation dependencies, never current-content aliases.

    Default expression-to-factory forwarding remains in original F4 edges.
    The retained operand Use is labeled here, without entering the body or
    assigning a creation activation to a later caller's context.
    """
    relation_kinds=frozenset((Kind.FIELD_INITIALIZER,Kind.ENVIRONMENT))

    def __init__(self,view,creations,*,units=()):
        self.view=view
        self.store=creations if isinstance(creations,SourceFunctionCreationStore) else SourceFunctionCreationStore(creations,catalog=view.catalog)
        if (self.store.catalog.path,self.store.catalog.report)!=(view.catalog.path,view.catalog.report):
            raise ValueError('creation roles require the same original source catalog')
        self.graphs={};self.records={};self.in_edges=defaultdict(list);self.out_edges=defaultdict(list)
        self._attachment_lock = Lock()
        self.roles=Counter();artifact=_artifact(self.store.path/'result.json',self.store.report['schema'])
        self.manifest=SourceArtifactLocator(artifact,artifact.sha256,artifact.size_bytes,'canonical-json')
        for unit in units:self.attach(unit)
        view.register(self)

    def attach(self,unit):
        with self._attachment_lock:
            if unit in self.graphs:
                return self.graphs[unit]
            return self._attach(unit)

    def _attach(self, unit):
        source = self.view.unit(unit)
        graph, contexts = self.store._read_unit_with_views(unit, source=source)
        source._expand(graph.templates)
        records = {}; incoming = defaultdict(list); outgoing = defaultdict(list); counts = Counter()
        locator = replace(self.manifest, pointer='#/shards/'+unit+'/tables/creations')
        for record in graph.creations:
            key = record.site.port.template
            operation = source._node(key, record.site.port.owner)
            step = self.view.original(operation)
            context = contexts[key]
            if context.plan is not source.plans[key]:
                raise ValueError('creation replay and graph projection retained different factory plans')
            if context.result(step)!=record.function:raise ValueError('creation role has a different factory result')
            result=self.view.source_port(record.function);records[source.namespace,result.node]=record
            roles=[('CODE',0,Kind.ENVIRONMENT),('GLOBALS',1,Kind.ENVIRONMENT)]
            roles.extend(('INITIAL_DEFAULT',d.operand_ordinal,Kind.FIELD_INITIALIZER) for d in record.defaults)
            roles.extend(('CELL_HANDLE_NOT_CONTENT',c.operand_ordinal,Kind.ENVIRONMENT) for c in record.captures)
            if record.annotation is not None:
                found=[i for i,v in enumerate(step.arguments[:-1]) if context.value(v)==record.annotation]
                if len(found)!=1:raise ValueError('annotation role lacks its exact original owned operand')
                roles.append(('DEFERRED_ANNOTATION_OPERAND',found[0],Kind.ENVIRONMENT))
            for role,index,kind in roles:
                use=source.port(context.emitter.port(step.id,G.PortKind.ARGUMENT,index))
                condition=Condition('scar.source-creation.operand',role,operation,index,(use,result))
                evidence=Evidence('scar.source-creation.initial-operand-role.v1',(source.locator,locator),(use,operation))
                edge=Edge(use,operation,kind,Meaning.DEPENDENCE,(evidence,),(condition,),Strength.MAY)
                outgoing[source.namespace,use.node].append(edge)
                incoming[source.namespace,operation.node].append(edge)
                counts[role] += 1
        self.store.validate()
        self.records.update(records)
        for key, edges in incoming.items():
            self.in_edges[key].extend(edges)
        for key, edges in outgoing.items():
            self.out_edges[key].extend(edges)
        self.roles.update(counts)
        self.graphs[unit] = graph
        return graph

    def _demand(self,point):
        if type(point.node) not in (OperationID,UseID,ValueID):return
        source=self.view.namespaces.get(point.namespace)
        unit=getattr(source,'unit',None)
        if unit in self.store.report['shards'] and unit not in self.graphs:self.attach(unit)

    def creation(self,point):
        self._demand(point);self.view.get(point)
        return self.records.get((point.namespace,point.node))

    def _neighbors(self,point,incoming):
        self._demand(point);rows=(self.in_edges if incoming else self.out_edges).get((point.namespace,point.node),())
        if not rows:return Neighborhood()
        self.store.validate();self.view.get(point)
        return Neighborhood(tuple(replace(e,source=replace(e.source,context=point.context),
            target=replace(e.target,context=point.context),conditions=tuple(replace(c,
                subject=None if c.subject is None else replace(c.subject,context=point.context),
                operands=tuple(replace(p,context=point.context) for p in c.operands)) for c in e.conditions)) for e in rows))

    def incoming(self,point):return self._neighbors(point,True)
    def outgoing(self,point):return self._neighbors(point,False)

    def summary(self):
        return dict(available_units=len(self.store.report['shards']),units=len(self.graphs),
            creations=len(self.records),operand_roles=dict(self.roles),
            current_defaults_or_cells_proven=False,creation_instances_selected=False)


# Selected function origins keep past creator and current caller separate.
#
# A returned function may outlive its creator. Creator frames describe a
# reference provenance, not the live caller stack or a snapshot of cell contents.
@dataclass(frozen=True,slots=True)
class ReturnedCallableBinding:
    key: tuple[str,int]
    origin: Point
    callee: Point
    current_frame: Frame
    creator_frames: tuple[Frame,...]
    provenance: ReturnedCallableProvenance


class ReturnedCallableProvider:
    relation_kinds=frozenset((K.PROJECTION,))

    def __init__(self,owner):
        self.owner=owner;self.view=owner.view;self.bindings={}
        self.by_origin=defaultdict(list);self.by_callee=defaultdict(list)
        self.view.register(self)

    @staticmethod
    def _key(point):return point.namespace,point.node

    def bind(self,key,interface,frame):
        provenance=self.owner.store.callable_provenance(*key)
        if provenance is None:return
        if provenance.creation.function!=interface.link.source_callable:
            raise ValueError('returned callable origin differs from its selected source definition')
        creator=[]
        for record in provenance.creator_interfaces:
            operation,use,callee=source_call_site(self.view,record.link)
            body=record.link.body
            target=self.view.source_region(body.unit,body.template,body.region)
            creator.append(Frame(operation.namespace,operation.node,use.node,callee.node,target.namespace,
                entry=target.node,selection=record.evidence))
        origin=self.view.source_port(provenance.creation.function)
        callee=self.view.source_port(interface.link.callee)
        binding=ReturnedCallableBinding(key,origin,callee,frame,tuple(creator),provenance)
        self.bindings[key]=binding;self.by_origin[self._key(origin)].append(key);self.by_callee[self._key(callee)].append(key)

    def creator_context(self,key,caller_context):
        binding=self.bindings[key]
        active=source_active_scope(self.view,binding.callee.namespace,caller_context)
        if active is not None and active not in binding.provenance.base_scopes:return None
        context=caller_context
        for frame in binding.creator_frames:context=context_push(context,frame)
        frames=binding.creator_frames
        namespace=frames[-1].target if frames else binding.callee.namespace
        owner=source_active_scope(self.view,namespace,context)
        if owner is not None and owner!=binding.provenance.creation.creator.owner:return None
        return context

    def _neighbors(self,point,incoming):
        self.owner._demand(point)
        keys=(self.by_callee if incoming else self.by_origin).get(self._key(point),())
        if not keys:return Neighborhood()
        self.owner.store.validate();self.view.get(point);edges=[];reasons=set()
        for key in keys:
            binding=self.bindings[key]
            if incoming:caller_context=point.context
            else:
                popped=context_pop(point.context,binding.creator_frames)
                if not popped.matched:
                    reasons.add('CALLABLE_CREATOR_REFERENCE_CONTEXT_NOT_SELECTED');continue
                caller_context=popped.remaining
            creator_context=self.creator_context(key,caller_context)
            # None is a legitimate unknown base for a zero-frame origin.
            if creator_context is None and (binding.creator_frames or caller_context is not None):
                reasons.add('CALLABLE_CREATOR_BASE_SCOPE_NOT_SELECTED');continue
            source=replace(binding.origin,context=creator_context);target=replace(binding.callee,context=caller_context)
            region=self.owner.frames[key]
            call=Point(region.caller,region.operation,caller_context)
            guards=(*binding.provenance.conditions,
                'CURRENT_CALLABLE_IS_SELECTED_RETURNED_FUNCTION_OCCURRENCE',
                'CREATOR_REFERENCE_CONTEXT_IS_NOT_CURRENT_CALLER_STACK',
                'CAPTURE_HANDLES_NOT_CURRENT_CELL_CONTENTS')
            conditions=tuple(Condition('scar.returned-callable.provenance',g,call,key[1],(source,target)) for g in guards)
            locator=self.owner.store.provenance_locator(*key)
            if locator is None:raise ValueError('returned callable lacks its original provenance receipt')
            edges.append(Edge(source,target,K.PROJECTION,Meaning.REFERENCE,
                (Evidence('scar.returned-callable.selected-creation.v1',
                    (locator,source.namespace.graph,target.namespace.graph),(call,source,target)),),conditions,Strength.MAY))
            reasons.add('RETURNED_CALLABLE_AND_CREATOR_CELL_VERSIONS_NOT_PROVEN')
        if edges and not context_is_closed(point.context):reasons.add('SOURCE_CALL_CONTEXT_PREFIX_OPEN')
        return Neighborhood(tuple(edges),tuple(Frontier(point,r,
            'Past creator context and current caller are separately selected; callable/cell identity and current versions remain conditional.')
            for r in sorted(reasons)))

    def incoming(self,point):return self._neighbors(point,True)
    def outgoing(self,point):return self._neighbors(point,False)
