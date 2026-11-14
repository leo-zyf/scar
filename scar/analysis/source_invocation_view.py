"""Attach recorded invocations to original parameter/return/world interfaces.

Only direct ordinary Python body entries are joined. Shared F4 bodies are not
copied; occurrence handles qualify original ports by run and InvocationKey.
Static DependencyFrame remains a static interface selection.
"""
from collections import defaultdict

from scar.analysis.program_view import _digest
from scar.analysis.source_reference_slice import source_port
from scar.ir.program import source_invocations as I, normal_targets as T, source_productions as P, view as V
from scar.ir.program.call_edges import FrameCompletion
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.ids import ProviderID
from scar.ir.program.python_call_targets import PythonTargetKind
from scar.ir.program.dependencies import (DependencyPoint as Point, DependencyEdge as Edge,
    DependencyCondition as Condition, DependencyEvidence as Evidence, DependencyFrontier as Frontier,
    DependencyNeighborhood as Neighborhood, DependencyKind as K, DependencyMeaning as Meaning,
    DependencyStrength as Strength)


SCHEMA='scar.source-invocation-ports'
_GUARDS=('RECORDED_DIRECT_INVOCATION_SELECTED','ORIGINAL_SOURCE_INTERFACE_CONDITIONS_RETAINED',
    'SOURCE_WORLD_BINDING_IS_NOT_CONTENT_PRESERVATION','NO_CONTENT_VERSION_EQUALITY_OR_EFFECT_PURITY_PROVEN')


def derive_invocation_bindings(view,groups,observations,activations,interfaces,trace,targets,
                               *,group_facts,check=lambda:None):
    """Join a selected original record set; admission remains with its reader.

    ``check`` revalidates the selected material receipts. Raw frame/target/attempt
    records and current original F4 function/call ports are checked separately;
    this does not certify an old whole report under a changed producer pin.
    """
    from types import SimpleNamespace
    from scar.analysis.source_observed_interfaces import _body_entry
    from scar.analysis.source_declared_calls import _owners

    check();groups=tuple(groups);observations=tuple(observations);group_facts=tuple(group_facts)
    if len(groups)!=len(group_facts):raise ValueError('every original group requires its own evidence reference')
    run=targets.source_sha256
    raw_frames={frame.anchor:frame for frame in trace.entries}
    raw_exits={exit.invocation:exit for exit in trace.exits}
    attempts={(attempt.process_id,attempt.index):attempt for attempt in trace.attempts}
    target_rows={(row.caller,row.attempt_index):(row,anchor)
        for row,anchor in zip(targets.targets,targets.target_anchors)}
    entries={(row.caller,row.attempt_index):row for row in targets.entries}
    witnessed=defaultdict(list)
    for row in observations:
        if raw_frames.get(row.entry.anchor)!=row.entry or row.exit is not None and raw_exits.get(row.entry.invocation)!=row.exit:
            raise ValueError('body observation differs from original retained frame evidence')
        witnessed[row.entry.anchor].append(row)
    index=SimpleNamespace(codes=SimpleNamespace(trace=trace))
    candidates=defaultdict(set)
    for group in groups:
        if len(group.observations)!=len(group.body_entries):
            raise ValueError('observed group lacks entry evidence for every occurrence')
        for observation,entry in zip(group.observations,group.body_entries):
            candidates[observation.id,entry].add((group.code.body,group.code.function_fact))
    result=[]
    for group,fact in zip(groups,group_facts):
        code=group.code;connection=group.connection
        callee=view.unit(code.body.unit).facts
        functions=[fn for fn in callee.functions if source_port(callee,fn.value)==code.function]
        if len(functions)!=1:raise ValueError('code correspondence has no unique original function fact')
        fn=functions[0]
        if (T.Body(callee.unit,fn.template,fn.body)!=code.body or fn.body_scope!=code.scope
                or connection.body!=code.body or connection.source_callable!=code.function):
            raise ValueError('observed connection changed its original function body or code scope')
        for observation,entry_anchor in zip(group.observations,group.body_entries):
            caller=view.unit(observation.unit).facts;call=caller.calls[observation.query]
            if (observation.query!=group.query or source_port(caller,call.result)!=observation.result
                    or source_port(caller,call.callee)!=observation.callee
                    or attempts.get((observation.attempt.process_id,observation.attempt.index))!=observation.attempt
                    or target_rows.get((observation.target.caller,observation.target.attempt_index))!=(observation.target,observation.target_anchor)):
                raise ValueError('observed call differs from original CALL, attempt or actual target record')
            if observation.owner.owner not in _owners(caller)[observation.query]:
                raise ValueError('actual caller code scope does not own the original source CALL')
            frame=raw_frames.get(entry_anchor)
            if frame is None:raise ValueError('group body entry is absent from original frame records')
            retained=[row for row in witnessed[entry_anchor]
                if activations[row.activation].code_link==code.id
                and interfaces[activations[row.activation].interface].body==code.body]
            if len(retained)!=1:raise ValueError('body entry has no unique retained activation/interface alternative')
            body_observation=retained[0]
            site=activations[body_observation.activation]
            if interfaces[site.interface].scope!=code.scope or site.code.key!=frame.code:
                raise ValueError('actual activation differs from original code/body interface')
            call_scope=I.SourceCallScope(run,frame.invocation)
            caller_scope=I.SourceCallScope(run,observation.target.caller)
            frontiers=[];routes=[]
            def missing(field,reason,port=None):frontiers.append(I.SourceInvocationFrontier(field,reason,port))
            actual_entry=entries.get((observation.target.caller,observation.target.attempt_index))
            matched=(actual_entry is not None and actual_entry.callee==frame.invocation
                and actual_entry.entry_index==frame.anchor.index and actual_entry.code==frame.code)
            if actual_entry is None:
                matched=_body_entry(index,observation)==frame
            direct=(not frame.suspended and frame.path_complete and len(frame.path)==1
                and frame.caller==observation.attempt.site and frame.path[0]==frame.caller
                and frame.caller_invocation==observation.target.caller
                and frame.code==observation.target.code
                and any(c.code.key==frame.code and c.process_id==frame.invocation.process_id for c in code.captures))
            ordinary=(connection.activation is T.FunctionActivation.NORMAL
                and fn.environment.kind is P.DeferredBodyKind.FUNCTION and observation.target.kind in (
                PythonTargetKind.FUNCTION,PythonTargetKind.BOUND_METHOD,PythonTargetKind.NORMALIZED_RECEIVER_FUNCTION)
                )
            selected=matched and direct and ordinary and len(candidates[observation.id,entry_anchor])==1
            if not selected:missing('target','DIRECT_INVOCATION_AND_SOURCE_TARGET_NOT_UNIQUELY_SELECTED')
            def route(left,right,field,enter):
                if selected:
                    routes.append(I.SourceInvocationRoute(
                        I.SourceInvocationPort(caller_scope if enter else call_scope,left,
                            observation.attempt.anchor if enter else None),
                        I.SourceInvocationPort(call_scope if enter else caller_scope,right,
                            None if enter else observation.attempt.anchor),field,(fact,observation.call),
                        (*connection.conditions,*_GUARDS)))
            formals=tuple(source_port(callee,index) for index in fn.formals)
            allowed={source_port(caller,index) for index in (*call.positional,*(point for _,point in call.keywords))}
            covered=set()
            for item in connection.parameters:
                if item.formal not in formals:raise ValueError('parameter route is outside original function formals')
                if item.kind is T.RouteKind.STATE:
                    if item.formal!=formals[-1] or item.actuals!=(source_port(caller,call.world),):
                        raise ValueError('state argument route differs from original pre-call world')
                    route(item.actuals[0],item.formal,'world_input',True);covered.add(item.formal)
                elif item.kind is T.RouteKind.FUNCTION_ENVIRONMENT:
                    if item.formal!=formals[0] or item.actuals!=(observation.callee,):
                        raise ValueError('environment route differs from actual original callee reference')
                    route(item.actuals[0],item.formal,'environment',True);covered.add(item.formal)
                    missing('environment.contents','CREATOR_GLOBALS_AND_CAPTURE_INSTANCES_NOT_BOUND',item.formal)
                elif (item.kind is T.RouteKind.ARGUMENT and group.argument_layout_proven
                        and connection.binding is T.ArgumentBinding.DIRECT
                        and observation.target.layout is not None
                        and observation.target.layout.parameters==fn.parameter_names):
                    if len(item.actuals)==1 and item.actuals[0] in allowed:
                        route(item.actuals[0],item.formal,'argument',True);covered.add(item.formal)
                    else:missing('parameter','RECEIVER_OR_NONCALLER_ARGUMENT_SOURCE_OPEN',item.formal)
                else:missing('parameter','LAYOUT_DEFAULT_SLOT_OR_PACK_BINDING_OPEN',item.formal)
            for formal in formals:
                if formal not in covered and not any(frontier.port==formal for frontier in frontiers):
                    missing('parameter','ORIGINAL_FORMAL_ROUTE_NOT_ESTABLISHED',formal)
            exit=body_observation.exit
            if exit is None or exit.completion is not FrameCompletion.NORMAL:
                missing('data_return','ACTUAL_NORMAL_EXIT_NOT_RECORDED')
                missing('world_return','ACTUAL_NORMAL_EXIT_NOT_RECORDED')
            else:
                expected=[]
                if fn.returns is not None:expected.append((source_port(callee,fn.returns),observation.result,'data_return'))
                if fn.return_world is not None:expected.append((source_port(callee,fn.return_world),source_port(caller,call.result_world),'world_return'))
                for left,right,field in expected:
                    if (left,right) not in connection.returns:missing(field,'ORIGINAL_NORMAL_RETURN_ROUTE_MISSING',left)
                    elif left.port.completion is not C.RETURN or right.port.completion is not C.FALLTHROUGH:
                        missing(field,'NORMAL_RETURN_COMPLETION_LABEL_UNRESOLVED',left)
                    else:route(left,right,field,False)
            identity=ProviderID(_digest((SCHEMA,call_scope,caller_scope,observation.id,body_observation.activation)))
            result.append(I.SourceInvocationBinding(identity,call_scope,caller_scope,body_observation.activation,
                observation.id,code.body,frame.anchor,None if exit is None else exit.anchor,
                tuple(routes),tuple(dict.fromkeys(frontiers))))
    check();return tuple(result)


class SourceInvocationProvider:
    """Occurrence-qualified boundary relations on the existing ProgramView."""
    relation_kinds=frozenset((K.ARGUMENT,K.ENVIRONMENT,K.STATE_INTERFACE,K.COMPLETION))

    def __init__(self,view,bindings,*,check=lambda:None):
        from scar.analysis.source_call_view import prepare_source_ports
        self.view,self.bindings,self.check=view,tuple(bindings),check
        prepare_source_ports(view,tuple(port.original for binding in self.bindings for route in binding.routes
            for port in (route.source,route.target)))
        self.nodes={};self.points={};self.in_edges=defaultdict(list);self.out_edges=defaultdict(list)
        self.frontiers=defaultdict(list);self.port_bindings=defaultdict(list)
        self.binding_points={}
        for binding in self.bindings:
            body=view.source_region(binding.body.unit,binding.body.template,binding.body.region)
            occurrence=Point(body.namespace,binding.id)
            self.nodes[occurrence.node]=(binding,body);self.binding_points[binding.id]=occurrence
            self.port_bindings[occurrence.node].append(binding)
            for route in binding.routes:
                source,target=self.port(route.source),self.port(route.target)
                kind=(K.STATE_INTERFACE if route.field.startswith('world') else K.ENVIRONMENT if route.field=='environment'
                    else K.COMPLETION if route.field=='data_return' else K.ARGUMENT)
                meaning=Meaning.STATE_INTERFACE if kind is K.STATE_INTERFACE else Meaning.DEPENDENCE
                sources=tuple(dict.fromkeys(self.view.unit(port.original.unit).locator
                    for port in (route.source,route.target)))
                edge=Edge(source,target,kind,meaning,
                    (Evidence(SCHEMA,sources,(source,target)),),
                    tuple(Condition(SCHEMA,guard,source,operands=(source,target)) for guard in route.conditions),Strength.MAY)
                self.out_edges[source.node].append(edge);self.in_edges[target.node].append(edge)
                self.port_bindings[source.node].append(binding);self.port_bindings[target.node].append(binding)
            for frontier in binding.frontiers:
                self.frontiers[occurrence.node].append(Frontier(occurrence,frontier.reason,frontier.field))
                if frontier.port is not None:
                    point=self.port(I.SourceInvocationPort(binding.call_scope,frontier.port))
                    self.frontiers[point.node].append(Frontier(point,frontier.reason,frontier.field))
                    self.port_bindings[point.node].append(binding)
        view.register(self)

    def port(self,record):
        if record not in self.points:
            original=self.view.source_port(record.original)
            point=Point(original.namespace,ProviderID(_digest((SCHEMA,record))))
            self.points[record]=point;self.nodes[point.node]=(record,original)
        return self.points[record]

    def describe(self,point):
        if point.node not in self.nodes:return None
        self.check();record,original=self.nodes[point.node]
        if point.namespace!=original.namespace:return None
        return V.ProgramNode(point,None,V.ProgramNodeRole.PROVIDER,original,self.view.get(original).value_type)

    def original(self,point):return self.nodes[point.node][0]
    def expand(self,point):return (self.nodes[point.node][1],)
    def explain(self,point):
        record=self.original(point)
        return dict(original=record,call_scope=record.call_scope,shared_source=self.expand(point)[0],
            occurrence_evidence=tuple(dict.fromkeys(self.port_bindings[point.node])),
            content_preservation_proven=False,logical_content_version_proven=False,source_body_copied=False)

    def _neighbors(self,point,incoming):
        self.check()
        if self.describe(point) is None:return Neighborhood()
        if point.context!=():
            return Neighborhood(frontiers=(Frontier(point,'STATIC_FRAME_IS_NOT_INVOCATION_SCOPE',
                'Use the original run/InvocationKey occurrence handle without a static frame context.'),))
        return Neighborhood(tuple((self.in_edges if incoming else self.out_edges)[point.node]),tuple(self.frontiers[point.node]))

    def incoming(self,point):return self._neighbors(point,True)
    def outgoing(self,point):return self._neighbors(point,False)


def _original_invocation_groups(store,unit):
    """Reuse the admitted unit replay; decode its original group table once."""
    from scar.analysis.source_symbol_variants import _table
    from scar.analysis.source_world_returns import _table_ref
    from scar.ir.program.source_observed_calls import ObservedSourceFunctionGroup
    from scar.ir.record_codec import decode
    store.read_unit(unit)
    shard=store.report['shards'][unit];folder=store.path/shard['path'];receipt=shard['tables']['groups']
    groups=tuple(decode(ObservedSourceFunctionGroup,row,record_tags=P.PRODUCTION_RECORD_TAGS)
        for row in _table(folder,receipt))
    if len(groups)!=shard['counts']['observed_groups']:
        raise ValueError('original observed group denominator differs')
    store.validate()
    return groups,tuple(_table_ref(folder,receipt,'scar.observed-source-function-group',i)
        for i in range(len(groups)))


class LazySourceInvocationProvider:
    """Demand same-run original group units, without rescanning source or trace.

    Discovery is structural INSTANCE_OF_SHARED_PORT, never data production,
    aliasing or content equality. Actual routes remain on occurrence ports.
    """
    relation_kinds=frozenset((K.IMPLEMENTATION,))

    def __init__(self,view,activations,functions,*,gaps=None,checkpoint=lambda:None):
        from pathlib import Path
        from scar.analysis.source_value_flow import _artifact
        from scar.ir.record_codec import encode
        from scar.ir.program import source_invocations
        self.view,self.activations,self.checkpoint=view,activations,checkpoint
        self.gaps=[] if gaps is None else gaps;self.materials={};self.providers={};self.failed={}
        self.out_edges=defaultdict(list);self.in_edges=defaultdict(list);self.group_tables={}
        self.run_scope=activations.store.report['basis']['run_scope']
        self.receipts=[]
        for provider in functions:
            store=provider.store
            if store.report.get('schema')!='scar.source-observed-function-interfaces':continue
            if (store.report['run_scope']!=self.run_scope
                    or (store.catalog.path,store.catalog.report)!=(view.catalog.path,view.catalog.report)):
                self.gaps.append(dict(layer='invocations',kind='OBSERVED_GROUP_RUN_OR_CATALOG_DIFFERS',path=str(store.path)))
                continue
            self.receipts.append(encode(_artifact(store.path/'result.json',store.report['schema'])))
            for unit in sorted(provider.available_units):self.materials.setdefault(unit,store)
        self.rules=tuple(_artifact(path,'scar.source-invocation-provider-rules') for path in (
            Path(__file__),Path(source_invocations.__file__),Path(__import__(
                'scar.analysis.source_body_activation_view',fromlist=['__file__']).__file__)))
        self.activation_receipt=encode(_artifact(activations.store.path/'result.json',activations.store.report['schema']))
        if not self.materials:
            self.gaps.append(dict(layer='invocations',kind='SAME_RUN_OBSERVED_FUNCTION_GROUPS_NOT_AVAILABLE',run_scope=self.run_scope))
        view.register(self)

    def check(self):
        from pathlib import Path
        from scar.analysis.source_value_flow import _artifact
        self.activations.store.check()
        if any(_artifact(Path(rule.uri),rule.schema)!=rule for rule in self.rules):
            raise ValueError('invocation scope provider rules changed')
        stores={id(self.materials[unit]):self.materials[unit] for unit in self.providers}
        for store in stores.values():store.validate()

    def ensure(self,unit):
        self.check()
        if unit in self.providers:return self.providers[unit]
        if unit in self.failed:return None
        store=self.materials.get(unit)
        if store is None:
            self.failed[unit]='SAME_RUN_OBSERVED_GROUP_UNIT_NOT_AVAILABLE'
            self.gaps.append(dict(layer='invocations',kind=self.failed[unit],unit=unit,run_scope=self.run_scope))
            return None
        try:
            groups,facts=_original_invocation_groups(store,unit)
            def guard():
                self.check();store.validate()
            provider=self.activations.bind_invocations(groups,group_facts=facts,check=guard)
            self.providers[unit]=provider
            shard=store.report['shards'][unit]
            self.group_tables[unit]=dict(publication=str(store.path),unit=unit,
                table=shard['tables']['groups'],path=str(store.path/shard['path']/shard['tables']['groups']['path']),
                groups=len(groups))
            def discovery(original,instance,rule):
                edge=Edge(original,instance,K.IMPLEMENTATION,Meaning.STRUCTURE,
                    (Evidence(SCHEMA+'.instances',tuple(dict.fromkeys((original.namespace.graph,
                        self.view.unit(unit).locator))),(original,instance)),),
                    (Condition(SCHEMA,rule,original,operands=(original,instance)),),Strength.MAY)
                self.out_edges[original.namespace,original.node].append(edge)
                self.in_edges[instance.namespace,instance.node].append(edge)
            for record,point in provider.points.items():
                discovery(self.view.source_port(record.original),point,'INSTANCE_OF_SHARED_PORT')
            for binding in provider.bindings:
                discovery(self.view.source_region(binding.body.unit,binding.body.template,binding.body.region),
                    provider.binding_points[binding.id],'INSTANCE_OF_SHARED_BODY')
            self.checkpoint();return provider
        except (OSError,ValueError) as error:
            self.failed[unit]=str(error)
            self.gaps.append(dict(layer='invocations',kind='OBSERVED_GROUP_UNIT_NOT_BOUND',unit=unit,
                path=str(store.path),detail=str(error)))
            return None

    def _query(self,point,incoming):
        from dataclasses import replace
        self.check()
        source=self.view.namespaces.get(point.namespace)
        original=source is not None and point.node in source.records
        if original:self.ensure(source.unit)
        key=point.namespace,point.node
        if incoming:
            # Instance handles have their own InvocationKey scope, never a
            # static DependencyFrame selected by a source navigation path.
            if point.context!=():return Neighborhood()
            edges=tuple(self.in_edges[key])
        else:
            edges=tuple(replace(edge,source=point) for edge in self.out_edges[key])
        frontiers=()
        if original and source.unit in self.failed:
            frontiers=(Frontier(point,'INVOCATION_SOURCE_GROUPS_UNAVAILABLE',self.failed[source.unit]),)
        return Neighborhood(edges,frontiers)

    def incoming(self,point):return self._query(point,True)
    def outgoing(self,point):return self._query(point,False)

    def publication(self):
        """Receipts and demanded-domain counts; no event or body copies."""
        from scar.ir.record_codec import encode
        self.check()
        bindings=tuple(binding for provider in self.providers.values() for binding in provider.bindings)
        return dict(schema=SCHEMA+'.lazy',schema_version=1,run_scope=self.run_scope,
            activation_publication=self.activation_receipt,observed_publications=self.receipts,
            provider_rules=encode(self.rules),group_tables=[self.group_tables[u] for u in sorted(self.group_tables)],
            available_units=len(self.materials),materialized_units=sorted(self.providers),
            unavailable_units=dict(sorted(self.failed.items())),actual_call_handles=len(bindings),
            routes=sum(len(binding.routes) for binding in bindings),
            field_frontiers=sum(len(binding.frontiers) for binding in bindings),
            undemanded_units_not_joined=True,shared_bodies_copied=False,events_copied=False,
            source_content_versions_inferred=False)

    def restore(self,record):
        """Replay only a publication's demanded units against original stores."""
        import json
        current=self.publication()
        dynamic={'group_tables','materialized_units','unavailable_units','actual_call_handles','routes','field_frontiers'}
        if (type(record) is not dict or set(record)!=set(current)
                or any(record[key]!=current[key] for key in current.keys()-dynamic)):
            raise ValueError('stored invocation scope provider receipts or run differ')
        units=record['materialized_units'];unavailable=record['unavailable_units']
        if (type(units) is not list or any(type(unit) is not str for unit in units)
                or units!=sorted(set(units)) or type(unavailable) is not dict
                or any(type(unit) is not str or type(reason) is not str for unit,reason in unavailable.items())
                or set(units)&set(unavailable)):
            raise ValueError('stored invocation demanded-unit domain differs')
        for unit in sorted(set(units)|set(unavailable)):self.ensure(unit)
        if json.loads(json.dumps(self.publication()))!=record:
            raise ValueError('stored invocation scope counts or original group receipts differ')


__all__=['derive_invocation_bindings','SourceInvocationProvider','LazySourceInvocationProvider']
