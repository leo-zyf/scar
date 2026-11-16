"""Match inlined source origins to existing, exact caller-specific frames.

The leaf body is shared. Only the original retained CALL query selects a leaf
activation; source names, same-unit membership and the leaf CodeID alone do
not. These remain conditional source interfaces, not observed invocations.
"""
from collections import defaultdict
from types import SimpleNamespace as NS

from scar.analysis.source_call_view import (SourceFunctionCallProvider,
    function_interface_fact,source_call_site,source_caller_scope)
from scar.analysis.source_namespace_interfaces import NamespacePlanView
from scar.analysis.source_reference_slice import ReferenceSlicer,source_port
from scar.ir.program import normal_targets as T,source_productions as P
from scar.ir.program import ordered_region_inline as R
from scar.ir.program.providers import ProviderArtifact,ProviderFactReference
from scar.ir.program.source_template_links import SourceTemplateKey
from scar.ir.program.dependencies import context_push,context_pop
from scar.ir.record_codec import decode


_GUARDS=(
    'ORIGINAL_OUTER_SOURCE_ACTIVATION_OWNS_RECORDED_INLINE_CALLSITE',
    'RECORDED_CALLSITE_SELECTS_THIS_ORIGINAL_LEAF_INTERFACE_AND_NORMAL_COMPLETION',
    'SOURCE_REPLAYED_ORDERED_EXPRESSION_USES_THIS_INLINE_CALL_RESULT',
    'INLINE_FRAME_IS_STATIC_CONDITIONAL_SELECTION_NOT_OBSERVED_INVOCATION',
)


class _Boundary(ValueError):pass


def _operation(view,origin):
    if origin.get('kind')!='original_f4' or 'step'not in origin:
        raise _Boundary('INLINE_ORIGIN_HAS_NO_ORIGINAL_F4_OPERATION')
    prefix=P.ProductionStepID.prefix+':'
    if not origin['step'].startswith(prefix):raise _Boundary('INLINE_ORIGIN_STEP_ID_DOMAIN_DIFFERS')
    source=view.unit(origin['unit']);template=decode(SourceTemplateKey,origin['template'])
    point=source._node(template,P.ProductionStepID(origin['step'][len(prefix):]))
    source.ensure(point);step=source.records[point.node]
    context=NamespacePlanView(source.unit,source.plans[template])
    return source,point,step,context,context.owner(step.region).owner


class OrderedRegionInlineIndex:
    def __init__(self,ordered,providers):
        ordered.check();self.ordered=ordered;self.view=ordered.view;self.providers=dict(providers)
        if any(type(p)is not SourceFunctionCallProvider or p.view is not self.view for p in self.providers.values()):
            raise TypeError('inline recovery requires original same-view source call providers')
        reference=ordered.reference
        self.artifact=ProviderArtifact(reference['uri'],reference['sha256'],reference['size_bytes'],reference['schema'])
        identity=next(o for o in ordered._region.source.origins if o.get('kind')=='source_identity')
        self.source=self.view.unit(identity['unit'])
        functions=[f for f in self.source.facts.functions if f.template.occurrence==identity['occurrence']]
        if len(functions)!=1 or functions[0].body_scope is None:
            raise ValueError('inline recovery requires the replayed original outer function scope')
        self.outer_scope=functions[0].body_scope.owner
        self.routes=[];self.boundaries=[];self.by_edge=defaultdict(list);self.by_source=defaultdict(list)
        self._calls={};self._sources={};self._slicers={}
        for ordinal,origin in enumerate(ordered._region.source.origins):
            if not origin.get('callsite'):continue
            expression=origin.get('expression')
            indices=[i for i,node in enumerate(ordered._region.nodes)if node.id==expression]
            if len(indices)!=1:
                self.boundaries.append(R.OrderedInlineBoundary('#/source/origins/'+str(ordinal),
                    ProviderFactReference(self.artifact,'#/source/origins/'+str(ordinal)),
                    'INLINE_ORIGIN_HAS_NO_UNIQUE_ORDERED_EXPRESSION'));continue
            self._derive('#/nodes/'+str(indices[0]),origin,'#/source/origins/'+str(ordinal))
        for index,action in enumerate(ordered._region.actions):
            for ordinal,origin in enumerate(action.origins):
                if origin.get('callsite'):
                    self._derive('#/actions/'+str(index),origin,'#/actions/'+str(index)+'/origins/'+str(ordinal))
        self.routes=tuple(self.routes);self.boundaries=tuple(self.boundaries);self.check()

    @staticmethod
    def _key(point):return point.namespace,point.node

    def _fact(self,source,table,ordinal):
        if source.unit not in self._slicers:
            material=source.material
            self._slicers[source.unit]=ReferenceSlicer(NS(fact_store=material.fact_store,
                rows=lambda *args,**kwargs:(),validate_parents=material.validate_parents))
        return self._slicers[source.unit].fact(source.facts,table,ordinal)

    def _selections(self,origin,scope):
        caller,call,step,context,owner=_operation(self.view,origin)
        self._sources[caller.unit]=caller
        if owner!=self.outer_scope:raise _Boundary('INLINE_CALLSITE_NOT_OWNED_BY_REPLAYED_OUTER_FUNCTION')
        if origin.get('callsite'):raise _Boundary('NESTED_INLINE_CALLSITE_REQUIRES_AN_EXPLICIT_INTERMEDIATE_FRAME')
        if type(step.instruction)is not P.ApplyLanguageAction or step.instruction.action is not P.LanguageActionKind.CALL:
            raise _Boundary('INLINE_CALLSITE_IS_NOT_ORIGINAL_CALL_OPERATION')
        result=context.result(step)
        queries=[i for i,fact in enumerate(caller.facts.calls)if source_port(caller.facts,fact.result)==result]
        if len(queries)!=1:raise _Boundary('INLINE_CALLSITE_HAS_NO_UNIQUE_ORIGINAL_CALL_QUERY')
        query=queries[0];key=caller.unit,query,scope
        if key in self._calls:return self._calls[key]
        selections=[]
        for name,provider in self.providers.items():
            if caller.unit not in provider.available_units:continue
            graph=provider.store.read_unit(caller.unit)
            for ordinal,interface in enumerate(graph.interfaces):
                link=interface.link
                if interface.query!=query or interface.activation is not T.FunctionActivation.NORMAL or link.body is None:continue
                target=self.view.unit(link.body.unit);self._sources[target.unit]=target
                functions=[fn for fn in target.facts.functions if fn.template==link.body.template
                    and fn.body==link.body.region and fn.body_scope is not None
                    and fn.body_scope.owner==scope
                    and source_port(target.facts,fn.value)==link.source_callable]
                if len(functions)!=1:continue
                if source_call_site(self.view,link)[0]!=call:
                    raise ValueError('inline interface selects another original CALL operation')
                # Attachment reuses the existing binder, world/return routes
                # and the exact interface selection; no new frame is invented.
                provider.attach(caller.unit,queries=(query,));frame=provider.frames[(caller.unit,ordinal)]
                body=self.view.source_region(link.body.unit,link.body.template,link.body.region)
                if frame.target!=body.namespace or frame.entry!=body.node:
                    raise ValueError('inline frame target differs from its original body')
                query_row=next(row for row in graph.queries if row.query==query)
                if owner not in query_row.owners:
                    raise ValueError('inline interface CALL ownership differs from the original scope')
                selections.append((name,caller.unit,ordinal,query,
                    function_interface_fact(provider.store,caller.unit,ordinal),frame,owner,
                    self._fact(caller,'calls',query),tuple(link.conditions)))
        self._calls[key]=tuple(selections)
        return self._calls[key]

    def _derive(self,key,origin,locator):
        fact=ProviderFactReference(self.artifact,locator)
        try:
            source,operation,_,_,scope=_operation(self.view,origin);self._sources[source.unit]=source
            matches=self._selections(origin['callsite'],scope)
            if not matches:raise _Boundary('RECORDED_INLINE_CALLSITE_HAS_NO_MATCHED_NORMAL_INTERFACE_TO_ORIGINAL_LEAF_SCOPE')
            for name,unit,ordinal,query,interface,frame,owner,call,guards in matches:
                row=R.OrderedInlineRoute(key,operation,fact,call,name,unit,ordinal,query,
                    interface,frame,owner,scope,(*guards,*_GUARDS))
                index=len(self.routes);self.routes.append(row)
                self.by_edge[(self._key(operation),self._key(self.ordered.point(key)))].append(index)
                self.by_source[self._key(operation)].append(index)
        except _Boundary as reason:self.boundaries.append(R.OrderedInlineBoundary(key,fact,str(reason)))

    def routes_for(self,source_operation,ordered_point):
        return tuple(self.routes[i]for i in self.by_edge.get((self._key(source_operation),self._key(ordered_point)),()))

    def contexts_for(self,source_operation,ordered_point,outer_source_context):
        """Return each leaf context only under its real original caller scope."""
        rows=[]
        for route in self.routes_for(source_operation,ordered_point):
            active=source_caller_scope(self.view,route.frame,outer_source_context)
            if active is not None and active!=route.caller_scope:continue
            rows.append((route,context_push(outer_source_context,route.frame)))
        return tuple(rows)

    def pop_source_context(self,source_operation,ordered_point,leaf_context):
        rows=[]
        for route in self.routes_for(source_operation,ordered_point):
            popped=context_pop(leaf_context,(route.frame,))
            if not popped.matched:continue
            active=source_caller_scope(self.view,route.frame,popped.remaining)
            if active is not None and active!=route.caller_scope:continue
            rows.append((route,popped.remaining))
        return tuple(rows)

    def check(self):
        self.ordered.check()
        for source in self._sources.values():source.check()
        for provider in self.providers.values():provider.store.validate()

    def summary(self):
        return dict(inline_origins=len(self.routes),boundaries=len(self.boundaries),
            distinct_frames=len({r.frame for r in self.routes}),
            distinct_queries=len({(r.unit,r.query)for r in self.routes}),
            source_contexts_are_conditional=True,runtime_invocations_created=0,
            target_executed=False,transformation_authorized=False)


__all__=['OrderedRegionInlineIndex']
