"""Solve actual script entries through compiled contexts, with typed receipts."""
import argparse
from collections import Counter, defaultdict, deque
from contextlib import ExitStack
import faulthandler
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

from scar.analysis import source_context_fact_store, source_context_interfaces
from scar.analysis import namespace_flow as source_namespace_flow, source_relations as source_relation_replay, source_class_body as source_class_queries
from scar.analysis import normal_targets,source_launch,source_namespace_graph
from scar.analysis import normal_entry, normal_facts, namespace_values as source_namespace_builtins
from scar.analysis import source_frame_captures,source_frame_globals
from scar.analysis import source_references as source_reference_transport
from scar.ir.program import source_paths as source_builtin_namespaces
from scar.ir.program import normal_graph as G, source_frame_bindings, source_productions as P
from scar.ir.record_codec import decode, encode
from scar.analysis.normal_facts import SourceFactStore
from scar.artifacts import write_table
from scar.artifacts import _durable_bytes, _durable_mkdir
from scar.artifacts import _json


class _ContextEntrySolver(normal_entry.EntrySolver):
    def __init__(self, base, namespace_graph, entries, *, interfaces, jobs=None,
                 supplemental_sources=(), supplemental_interfaces=(), entry_units=None):
        self.namespace_graph = namespace_graph
        self.reference_transport=source_reference_transport.ActiveReferenceTransport(self)
        self.frames = None
        self.namespace_flow = None; self.namespace_pending = deque(); self.namespace_scopes = set()
        self.exec_by_result = defaultdict(list); self.call_owners = {}
        for connection in namespace_graph.exec_graph.connections():
            self.exec_by_result[connection.site.result].append(connection)
        store = source_context_fact_store.ContextFactStore(base,namespace_graph.exec_graph,supplemental_sources=supplemental_sources)
        summaries = source_context_interfaces.ContextInterfaceStore(interfaces,store,namespace_graph.exec_graph,
            supplemental_interfaces=supplemental_interfaces)
        if entry_units is not None:
            if entries:raise ValueError('select script entries by paths or qualified units, not both')
            entry_units=tuple(entry_units)
            if not entry_units or not set(entry_units)<=store.importable_units:
                raise ValueError('declared entry units must belong to the source catalog')
        super().__init__(store,entries,jobs=jobs,interface_store=summaries)
        # A compile-context body is not an importable module instance.
        for module in tuple(self.module_units):
            self.module_units[module] = [u for u in self.module_units[module] if u in store.importable_units]
            if not self.module_units[module]:
                del self.module_units[module]
        self.containers = source_relation_replay.SharedContainerFlow(self,[])
        self.constructions = source_relation_replay.SharedConstructionFlow(self)
        self.namespace_flow = source_namespace_flow.NamespaceFlow(self,namespace_graph)
        self.frames = source_frame_globals.FrameGlobalsFlow(self)
        if entry_units is not None:
            for identity in entry_units:
                unit=self.load_unit(identity)
                self.activate(unit.root_scope.owner,None,G.EntryReason.PROGRAM_ENTRY)

    def activate_function(self, fid, callee):
        super().activate_function(fid,callee)
        if self.frames is not None:
            self.frames.activate_function(fid,callee)

    def activate_generator(self, fid, driver, callee):
        super().activate_generator(fid,driver,callee)
        if self.frames is not None:
            self.frames.activate_function(fid,callee)

    def load_unit(self, identity):
        fresh = identity not in self._metadata_loaded
        result = super().load_unit(identity)
        if not fresh:
            return result
        for group in result.scope_groups:
            for span in group.tables:
                if span.table is G.FactTable.CALLS:
                    for index in range(span.start,span.stop):
                        self.call_owners[self.local_calls[identity,index]]=group.owner
        return result

    def add(self, point, value):
        normalized = ('external_value',None) if value[0]=='external_value' else (
            ('external',value[1]) if value[0]=='external' else value)
        fresh = normalized not in self.values.get(point,())
        super().add(point,value)
        if fresh:self.reference_transport.enqueue(point,normalized)
        flow = self.namespace_flow
        if fresh and flow is not None and (point in flow.watchers or point in flow.builtins.watchers):
            self.namespace_pending.append((point,normalized))

    def edge(self,source,target):
        super().edge(source,target)
        self.reference_transport.install(source,target)

    def admit(self, unit, table, index):
        if (table=='names' and unit.unit in self.store.original_source_units
                and unit.names[index].namespace.kind in (G.NamespaceKind.MODULE,G.NamespaceKind.BUILTINS)):
            # These lossy MODULE rows are replaced by ordered C2 interfaces.
            self.admitted.add((unit.unit,table,index))
            return
        result=super().admit(unit,table,index)
        if table=='aliases':self.reference_transport.admit(unit,index)
        if table=='names' and self.frames is not None:
            access=unit.names[index]
            if access.write and access.namespace.kind is G.NamespaceKind.MODULE:
                self.frames.name_write(unit.unit,access.namespace.name,self.offsets[unit.unit]+access.value)
            elif not access.write and access.namespace.kind is G.NamespaceKind.MODULE:
                self.frames.name_read(unit.unit,access.namespace.name,self.offsets[unit.unit]+access.value)
        return result

    def attribute(self, point, value, name, target):
        if value[0]=='source_root_namespace':
            self.frames.views.attribute(value,name,target)
            return
        return super().attribute(point,value,name,target)

    def prepare(self):
        super().prepare()
        if self.namespace_flow is not None:
            for owner in self.active_scopes-self.namespace_scopes:
                self.namespace_scopes.add(owner)
                self.namespace_flow.admit_scope(owner)
        self.reference_transport.drain()

    def call(self, index, value):
        unit,call = self.call_rows[index]; offset = self.offsets[unit.unit]
        if value[:2]==('external','builtins.exec'):
            connections = self.exec_by_result.get(self.gp(offset+call.result),())
            if connections:
                for connection in connections:
                    self.load_unit(connection.root.owner.unit)
                    self.namespace_flow.admit_connection(connection)
                    self.activate(connection.root.owner,connection.site.result,G.EntryReason.EAGER_REGION)
                self.add(offset+call.result,('constant','NoneType',None))
                self.resolved_interface_calls.add(index)
                return
        if (value[:2]==('external','builtins.globals')
                and not call.positional and not call.keywords
                and call.positional_pack is None and call.keyword_pack is None):
            owner = self.call_owners[index]
            if not self.frames.read_globals(owner,self.gp(offset+call.result),self.gp(offset+call.world)):
                self.add(offset+call.result,('external_value',None))
            self.resolved_interface_calls.add(index)
            return
        return super().call(index,value)

    def solve(self, max_events=20_000_000):
        self.prepare()
        while (self.queue or self.pending or self.namespace_pending or self.reference_transport.queue) and self.events<max_events:
            while self.namespace_pending:
                point,value = self.namespace_pending.popleft()
                self.namespace_flow.consume(point,value)
            self.prepare()
            if self.queue:
                super().solve(min(max_events,self.events+65_536))
        return not self.queue and not self.pending and not self.namespace_pending and not self.reference_transport.queue


_SAMPLE = 4
_TEXT = 768


def _repr(value):
    text = repr(value)
    return dict(text=text[:_TEXT],truncated=len(text)>_TEXT,characters=len(text))


def _snapshot_ports(solver, requests):
    """Observe selected original ``(unit, local)`` ports at this work frontier.

    Counts cover complete selected value sets; each kind and subscription list
    has at most four displayed entries. The queues are each traversed once, with
    no copy of their complete contents. Call this while propagation is stopped.
    An empty selected pending count never grants whole-program fixed point.
    """
    requests = tuple(dict.fromkeys(requests))
    if len(requests)>64:
        raise ValueError('at most 64 original source ports may be observed')
    store = getattr(solver,'store',None)
    units = {unit.unit:unit for unit in solver.units}
    references = {}

    def ref(point):
        if point in references:return references[point]
        result = dict(point=point if type(point) is int else None,gp=None)
        if type(point) is not int:
            result.update(kind='INTERNAL_LOOKUP_KEY',diagnostic=_repr(point))
        elif point<0:
            endpoint = getattr(solver,'interface_ports',{}).get(point)
            result.update(kind='DECLARED_INTERFACE_POINT',endpoint=None if endpoint is None else
                encode(endpoint,record_tags=P.PRODUCTION_RECORD_TAGS),unknown=endpoint is None)
        else:
            if store is not None:
                identity,local = store.locate(point)
                unit = store.loaded.get(identity)
            else:
                identity = solver.owner[point];local = point-solver.offsets[identity]
                unit = units.get(identity)
            result.update(kind='ORIGINAL_SOURCE_POINT',unit=identity,local=local)
            if unit is None:
                result['unknown']='UNIT_NOT_LOADED_NO_SOURCE_READ_PERFORMED'
            else:
                raw = unit.ports[local]
                wire = encode(raw,record_tags=P.PRODUCTION_RECORD_TAGS)
                result['gp']=dict(unit=identity,port=wire)
        references[point]=result
        return result

    def values(items):
        counts=Counter();samples=defaultdict(list)
        for value in items:
            kind=value[0] if type(value) is tuple and value and type(value[0]) is str else 'UNRECOGNIZED_TOKEN'
            counts[kind]+=1
            if len(samples[kind])>=_SAMPLE:continue
            row=dict(repr=_repr(value))
            positions={'function':(1,),'class':(1,),'instance':(1,2),'bound':(1,2),
                'classbound':(1,2,3),'staticmethod':(1,),'classmethod':(1,),'container':(1,)}.get(kind,())
            row['point_references']=[dict(token_ordinal=i,reference=ref(value[i]))
                for i in positions if len(value)>i and type(value[i]) is int]
            samples[kind].append(row)
        return dict(count=sum(counts.values()),by_kind={kind:dict(count=count,samples=samples[kind],
            truncated=count>len(samples[kind])) for kind,count in sorted(counts.items())})

    def points(items):
        total=0;sample=[]
        for point in items:
            total+=1
            if len(sample)<_SAMPLE:sample.append(ref(point))
        return dict(count=total,samples=sample,truncated=total>len(sample))

    descriptors=getattr(solver,'descriptors',None)
    selected={};namespace_rows={}
    for request in requests:
        if (type(request) is not tuple or len(request)!=2 or type(request[0]) is not str
                or type(request[1]) is not int or request[1]<0):
            raise ValueError('observation needs an exact source unit and nonnegative local ordinal')
        identity,local=request
        if identity not in solver.offsets:raise ValueError('observed source unit is outside the solver catalog')
        size=(store.rows[identity]['counts']['ports'] if store is not None else len(units[identity].ports))
        if local>=size:raise ValueError('observed ordinal is outside its original source port table')
        point=solver.offsets[identity]+local;endpoint=ref(point)
        if endpoint['gp'] is not None:
            # Decode only the explicitly requested ports, never every token GP.
            decode(G.Port,endpoint['gp']['port'],record_tags=P.PRODUCTION_RECORD_TAGS)
        accesses=() if descriptors is None else descriptors.accesses.get(point,())
        gets=solver.gets.get(point,())
        members={access.member for access in accesses}
        members.update(name for name,_ in gets)
        access_rows=[];read_rows=[]
        for access in accesses[:_SAMPLE]:
            access_rows.append(dict(key=_repr(access.key),action=access.action.value,member=access.member,
                receiver=ref(access.receiver),result=None if access.result is None else ref(access.result),
                assigned=None if access.assigned is None else ref(access.assigned),world=ref(access.world),
                result_world=ref(access.result_world)))
        if descriptors is not None:
            pairs={(a.member,a.result) for a in accesses if a.result is not None}
            pairs.update(gets)
            for member,target in list(pairs)[:_SAMPLE]:
                registered=descriptors.reads.get((point,member,target),())
                read_rows.append(dict(member=member,target=ref(target),count=len(registered),
                    accesses=[_repr(a.key) for a in registered[:_SAMPLE]],truncated=len(registered)>_SAMPLE))
        call_rows=[];watch=solver.call_watch.get(point,())
        for index in watch[:_SAMPLE]:
            unit,call=solver.call_rows[index];offset=solver.offsets[unit.unit]
            call_rows.append(dict(index=index,unit=unit.unit,callee=ref(offset+call.callee),
                result=ref(offset+call.result),world=ref(offset+call.world),result_world=ref(offset+call.result_world)))
        tokens=solver.values.get(point,())
        classes={value[1] for value in tokens if type(value) is tuple and len(value)>1
            and value[0] in ('class','instance')}
        class_rows=[]
        for cls in list(classes)[:_SAMPLE]:
            row=dict(source_class=ref(cls));declaration=solver.classes.get(cls)
            if declaration is None:
                row['unknown']='SOURCE_CLASS_METADATA_NOT_AVAILABLE'
            else:
                unit,descriptor=declaration
                row.update(namespace_unit=unit.unit,namespace_class_scope=descriptor.scope,members=[])
                for member in sorted(members)[:_SAMPLE]:
                    key=(unit.unit,'CLASS',descriptor.scope,member)
                    if key not in namespace_rows:
                        namespace_rows[key]=dict(unit=unit.unit,scope=descriptor.scope,member=member,
                            values=values(solver.ns_values.get(key,())),
                            readers=points(solver.reads.get(key,())),
                            member_watch=dict(count=len(solver.member_watch.get(key,())),
                                samples=[_repr(v) for v in list(solver.member_watch.get(key,()))[:_SAMPLE]],
                                truncated=len(solver.member_watch.get(key,()))>_SAMPLE))
                    row['members'].append(dict(namespace_key=list(key)))
            class_rows.append(row)
        selected[point]=dict(request=dict(unit=identity,local=local),source=endpoint,
            original_port_validated=endpoint['gp'] is not None,values=values(tokens),
            outgoing=points(solver.adj.get(point,())),
            descriptor_accesses=dict(count=len(accesses),samples=access_rows,truncated=len(accesses)>_SAMPLE),
            descriptor_read_subscriptions=read_rows,
            gets=dict(count=len(gets),samples=[dict(member=name,target=ref(target)) for name,target in list(gets)[:_SAMPLE]],
                truncated=len(gets)>_SAMPLE),
            call_watch=dict(count=len(watch),samples=call_rows,truncated=len(watch)>_SAMPLE),
            classes=dict(count=len(classes),samples=class_rows,truncated=len(classes)>_SAMPLE),
            subscription_scope='EXISTING_RECEIVER_INDEXES_ONLY_NO_GLOBAL_REVERSE_SCAN')

    pending=Counter();pending_kinds=defaultdict(Counter);namespace_pending=Counter();namespace_kinds=defaultdict(Counter)
    queue=getattr(solver,'queue',())
    for point,value in queue:
        if point in selected:
            pending[point]+=1;pending_kinds[point][value[0]]+=1
        for key in solver.writes.get(point,()):
            if key in namespace_rows:
                namespace_pending[key]+=1;namespace_kinds[key][value[0]]+=1
    other_pending=getattr(solver,'namespace_pending',())
    other_counts=Counter()
    for point,value in other_pending:
        if point in selected:other_counts[point]+=1
    for point,row in selected.items():
        row['pending']=dict(value_queue=pending[point],by_kind=dict(pending_kinds[point]),
            namespace_protocol_queue=other_counts[point],zero_does_not_imply_fixed_point=True)
    for key,row in namespace_rows.items():
        row['pending_source_write_events']=namespace_pending[key]
        row['pending_by_kind']=dict(namespace_kinds[key])
        row['pending_scope']='VALUE_QUEUE_EVENTS_WITH_EXISTING_WRITES_SUBSCRIPTION_TO_THIS_NAMESPACE'
    raw=None if store is None else store.raw
    return dict(schema='scar.source-solver-observations',schema_version=1,event_count=solver.events,
        frontier=len(queue),namespace_frontier=len(other_pending),
        pending_scopes=len(getattr(solver,'pending',())),unfixed=bool(queue or other_pending or getattr(solver,'pending',())),
        whole_program_fixed_point_claimed=False,source_catalog_sha256=None if raw is None else sha256(raw).hexdigest(),
        limits=dict(samples_per_kind_or_subscription=_SAMPLE,repr_characters=_TEXT),
        value_semantics='BOUNDED_ANALYZER_ABSTRACT_TOKEN_REPRS_NOT_LOGICAL_VALUES_OR_RUNTIME_OBJECT_CONTENTS',
        ports=list(selected.values()),class_namespaces=list(namespace_rows.values()),
        unknown_fields=('RUNTIME_DISPATCH_AND_OBJECT_CONTENTS_NOT_OBSERVED',
            'UNLOADED_SOURCE_PORT_BYTES_NOT_READ','UNSUBSCRIBED_MEMBERS_NOT_QUERIED',
            'NAMESPACE_PROTOCOL_QUEUE_HAS_POINT_EVENTS_NOT_DIRECT_CLASS_NAMESPACE_KEYS'),
        solver_mutated=False,source_units_loaded=False,propagation_performed=False,target_executed=False)


def pins():
    return {name:sha256(Path(module.__file__).read_bytes()).hexdigest() for name,module in (
        ('facts',source_context_fact_store),
        ('interfaces',source_context_interfaces),('namespaces',source_namespace_flow),
        ('relations',source_relation_replay),('class_queries',source_class_queries),
        ('original_publisher',normal_targets),('entry_contract',source_launch),
        ('namespace_graph',source_namespace_graph),('entry_solver',normal_entry),
        ('fact_reader',normal_facts),('builtin_flow',source_namespace_builtins),
        ('builtin_types',source_builtin_namespaces),('frame_index',source_frame_captures),
        ('frame_flow',source_frame_globals),('frame_types',source_frame_bindings),
        ('root_views',source_namespace_flow),
        ('reference_transport',source_reference_transport))}


def build_context_targets(facts,namespaces,entry,interfaces,out,*,max_events=2_000_000,jobs=4,supplements=(),observe_ports=()):
    with ExitStack() as resources:
        return _build_context_targets(facts,namespaces,entry,interfaces,out,max_events=max_events,jobs=jobs,
            resources=resources,supplements=tuple(supplements),observe_ports=tuple(observe_ports))


def _build_context_targets(facts,namespaces,entry,interfaces,out,*,max_events,jobs,resources,supplements,observe_ports):
    started=time.monotonic(); out=Path(out).absolute()
    if out.exists():raise ValueError('fresh context target output required')
    base=SourceFactStore(facts); graph=source_namespace_graph.NamespaceGraph(namespaces)
    contract=source_launch.read_script_entry(entry,base)
    initial=pins(); producer=sha256(Path(__file__).read_bytes()).hexdigest()
    supplemental_sources=tuple(SourceFactStore(pair[0]) for pair in supplements)
    solver=_ContextEntrySolver(base,graph,(),interfaces=interfaces,jobs=jobs,
        supplemental_sources=supplemental_sources,supplemental_interfaces=tuple(pair[1] for pair in supplements),
        entry_units=(contract.source.unit,))
    resources.callback(solver.frames.close)
    resources.callback(solver.generators.close)
    prepared=time.monotonic()
    solver.frames.configure_entry(contract)
    for binding in contract.globals:
        value=binding.literal.to_python()
        solver.namespace_value(solver.ns(contract.source.unit,binding.namespace),('constant',type(value).__name__,value))
    history=deque(maxlen=32);observation_count=0
    while (solver.queue or solver.pending or solver.namespace_pending or solver.reference_transport.queue) and (max_events is None or solver.events<max_events):
        solver.solve(solver.events+200_000 if max_events is None else min(max_events,solver.events+200_000))
        if observe_ports:
            history.append(_snapshot_ports(solver,observe_ports));observation_count+=1
        print(json.dumps(dict(phase='CONTEXT_ENTRY_PROPAGATION',events=solver.events,frontier=len(solver.queue),
            namespace_frontier=len(solver.namespace_pending),active_scopes=len(solver.active_scopes),
            exec_connections=len(solver.namespace_flow.connections),loaded_units=len(solver.store.loaded),
            seconds=time.monotonic()-started)),file=sys.stderr,flush=True)
    fixed=not solver.queue and not solver.pending and not solver.namespace_pending and not solver.reference_transport.queue
    solver.frames.validate_parents()
    solver.complete_call_demands(); _durable_mkdir(out)
    _durable_bytes(out/'context-facts.json',solver.store.raw,replace=False)
    observations=None;observation_history=None
    if observe_ports:
        raw=_json(_snapshot_ports(solver,observe_ports))
        _durable_bytes(out/'solver-observations.json',raw,replace=False)
        observations=dict(path='solver-observations.json',bytes=len(raw),sha256=sha256(raw).hexdigest())
        observation_history=write_table(out/'solver-observation-history.jsonl.gz',history)
    extra=dict(analysis_mode='ENTRY_COMPILED_CONTEXT_MAY_ANALYSIS',entry_sources=[contract.source.path],
        active_code_scopes=len(solver.active_scopes),loaded_sources=len(solver.units),
        active_source_calls=len(solver.active_call_indices),pending_code_scopes=len(solver.pending),
        namespace_frontier=len(solver.namespace_pending),
        active_compiled_namespaces=sorted({o.unit for o in solver.active_scopes if o.unit in solver.store.original_source_units}),
        admitted_exec_connections=len(solver.namespace_flow.connections),
        entry_activations=write_table(out/'entry-activations.jsonl.gz',solver.activations.values()),
        entry_module_demands=write_table(out/'entry-module-demands.jsonl.gz',solver.module_demands.values()),
        entry_body_demands=write_table(out/'entry-body-demands.jsonl.gz',solver.body_demands.values()),
        entry_call_demands=write_table(out/'entry-call-demands.jsonl.gz',solver.call_demands.values()),
        exec_connections=write_table(out/'exec-connections.jsonl.gz',solver.namespace_flow.connections.values()),
        builtin_candidates=write_table(out/'builtin-candidates.jsonl.gz',solver.namespace_flow.builtins.fallbacks),
        builtin_selections=write_table(out/'builtin-selections.jsonl.gz',solver.namespace_flow.builtins.selections),
        root_namespace_bindings=write_table(out/'root-namespace-bindings.jsonl.gz',solver.frames.roots.values()),
        frame_globals_bindings=write_table(out/'frame-globals-bindings.jsonl.gz',solver.frames.bindings.values()),
        frame_globals_reads=write_table(out/'frame-globals-reads.jsonl.gz',solver.frames.reads.values()),
        frame_globals_obligations=write_table(out/'frame-globals-obligations.jsonl.gz',solver.frames.obligations.values()),
        frame_capture_obligations=write_table(out/'frame-capture-obligations.jsonl.gz',
            (o for material in solver.frames.materials.values() for o in material.obligations)),
        static_input_bindings=write_table(out/'static-input-bindings.jsonl.gz',solver.frames.views.input_bindings.values()),
        frame_source_stats=solver.frames.index.stats,
        reference_transport=solver.reference_transport.stats,
        reference_alias_facts=write_table(out/'reference-alias-facts.jsonl.gz',
            (dict(unit=unit,alias=index) for unit,index in sorted(solver.reference_transport.facts))),
        construction_queries=dict(solver.constructions.query_stats),
        namespace_requests=len(solver.namespace_flow.mapping_requests),
        analysis_phase_seconds=dict(preparation=prepared-started,propagation=time.monotonic()-prepared))
    report = normal_targets._publish(solver, base.path, base.raw, solver.store.report, out, started,
                                     max_events, fixed, solver.stars, extra, write_report=False)
    solver.frames.validate_parents(); solver.frames.close()
    if (pins()!=initial or sha256(Path(__file__).read_bytes()).hexdigest()!=producer
            or (base.path/'result.json').read_bytes()!=base.raw):
        raise ValueError('context source rules changed during propagation')
    solver.store.validate_parents()
    def reference(path):
        path=Path(path).absolute()
        return dict(path=str(path),sha256=sha256((path/'result.json').read_bytes()).hexdigest())
    report.update(schema='scar.context-source-targets',schema_version=1,
        status='CONTEXT_TARGET_FIXED_POINT' if fixed else 'CONTEXT_TARGET_BUDGET_FRONTIER',
        source_only_publisher_sha256=report.pop('implementation_sha256'),
        implementation_sha256=producer,extension_pins=initial,
        context_facts=dict(path='context-facts.json',bytes=len(solver.store.raw),sha256=sha256(solver.store.raw).hexdigest()),
        namespace_graph=reference(namespaces),script_entry=reference(entry),
        supplemental_sources=[dict(facts=reference(source.path),interfaces=reference(pair[1]))
            for source,pair in zip(supplemental_sources,supplements)],
        entry_unit=contract.source.unit,importable_compiled_namespaces=False,
        namespace_lookup_semantics='ORDERED_SOURCE_GRAPH_CONDITIONAL_MAY_TRANSFER',
        solver_observations=observations,
        solver_observation_history=observation_history,observation_snapshots_seen=observation_count,
        observation_snapshots_dropped=max(0,observation_count-len(history)),
        fixed_point_scope='ADMITTED_SOURCE_AND_EXEC_CONTEXT_CONSTRAINTS',
        target_executed=False,first_stage_complete=False,seconds=time.monotonic()-started)
    _durable_bytes(out/'result.json',_json(report),replace=False)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('facts','namespaces','entry','interfaces','out'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--max-events',type=int,default=2_000_000)
    parser.add_argument('--until-fixed-point',action='store_true')
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--supplement',nargs=2,action='append',default=[],metavar=('FACTS','INTERFACES'),
        help='append a pinned source catalog and its local interfaces without changing existing ports')
    parser.add_argument('--observe-port',nargs=2,action='append',default=[],metavar=('UNIT','LOCAL'),
        help='snapshot selected analyzer port values and subscriptions at this event frontier')
    parser.add_argument('--stack-log')
    args=parser.parse_args()
    with ExitStack() as diagnostics:
        if args.stack_log:
            stream=diagnostics.enter_context(Path(args.stack_log).open('x'))
            faulthandler.dump_traceback_later(30,repeat=True,file=stream)
            diagnostics.callback(faulthandler.cancel_dump_traceback_later)
        report=build_context_targets(args.facts,args.namespaces,args.entry,args.interfaces,args.out,
            max_events=None if args.until_fixed_point else args.max_events,jobs=args.jobs,supplements=args.supplement,
            observe_ports=tuple((unit,int(local)) for unit,local in args.observe_port))
    print(json.dumps({name:report[name] for name in ('status','value_events','frontier','admitted_exec_connections',
        'active_compiled_namespaces','seconds')}),flush=True)


if __name__=='__main__':main()
