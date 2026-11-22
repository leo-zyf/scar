"""Source body activations."""

from collections import Counter, defaultdict
from hashlib import sha256
import argparse
import json
import time
from pathlib import Path
from typing import get_args, get_origin, get_type_hints
from scar.analysis.source_body_interfaces import BodyInterfaceResolver, _pins as body_pins
from scar.analysis.source_observed_codes import ObservedSourceCodeProvider
from scar.analysis.source_observed_calls import ObservedSourceCallsiteIndex, _span
from scar.analysis.source_namespace_contents import NamespacePlanView
from scar.analysis.program_view import ProgramView, _digest
from scar.analysis.source_consumer_parents import _stamp
from scar.artifacts import (
    _durable_bytes,
    _durable_mkdir,
    _json,
    read_record_table as _table,
    write_table,
)
from scar.analysis.source_value_flow import _artifact
from scar.ir.program import (
    body_observations as R,
    body_observations as B,
    source_observations as C,
    source_productions as P,
    normal_graph as G,
    view as V,
)
from scar.ir.program.ids import ProviderID, OperationID, ValueID
from scar.ir.program.model import ValueKind, ValueType
from scar.ir.program.call_edges import CallEdgeBudget
from scar.ir.record_codec import encode, decode
from scar.trace.code_snapshot import site as code_site
from dataclasses import fields, is_dataclass, replace
from scar.analysis.source_world_returns import _table_ref
from scar.ir.program.normal_graph import GlobalPort
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.program.dependencies import (
    DependencyPoint as Point,
    DependencyFrame as Frame,
    DependencyEdge as Edge,
    DependencyKind as K,
    DependencyMeaning as Meaning,
    DependencyStrength as Strength,
    DependencyCondition as Condition,
    DependencyEvidence as Evidence,
    DependencyNeighborhood as Neighborhood,
    context_push,
    context_pop,
)


# Join all recorded source-body entries through shared protocol interfaces.
#
# All correspondence variants enter the same join. A captured caller location
# selects existing source operations; unresolved choices remain conditional.
# Occurrences stay in evidence tables rather than unrolling the source graph.
SCHEMA = 'scar.source-body-activations'
TABLES = dict(interfaces=B.SourceBodyInterface, activations=R.SourceBodyActivation,
    observations=R.SourceBodyActivationObservation, parents=R.SourceBodyActivationParent,
    frontiers=R.SourceBodyActivationFrontier)
_LINK_TABLES = {C.ObservedSourceCodeLink:'records', C.ObservedSourceScopeLink:'scope_records',
    C.ObservedSourceRequestCodeLink:'request_records', C.ObservedSourceCompilerCodeLink:'compiler_records'}


def _links(codes):
    if set(_LINK_TABLES) != set(get_args(C.ObservedSourceBodyLink)):
        raise ValueError('activation correspondence adapters do not cover the closed source-body link domain')
    result = tuple(row for kind, table in _LINK_TABLES.items()
        for row in getattr(codes,table).values() if type(row) is kind)
    if (len(result) != sum(len(getattr(codes,name)) for name in _LINK_TABLES.values())
            or len({row.id for row in result}) != len(result)):
        raise ValueError('source-body correspondence domain contains unclassified or duplicate rows')
    # Correspondence alternatives are a set. Their source catalog enumeration
    # must not change activation identity when another admitted index is reused.
    return tuple(sorted(result,key=lambda row:row.id.value))


def _completions(context, step):
    return tuple(B.SourceBodyCompletion(completion.kind,completion.target,completion.types,
        tuple(context.gp(context.emitter.port(step.id,G.PortKind.RESULT,i,completion))
            for i in range(len(completion.types))),
        context.gp(context.emitter.port(step.id,G.PortKind.EXIT,completion=completion)))
        for completion in step.completions)


class _SourceAnchors:
    """Location abstraction over original typed operations, for every opcode.

    Exact existing CALL mappings take precedence. Otherwise the innermost
    enclosing F2 location supplies conditional operation alternatives in the
    original code scope. This neither infers a callable by name nor asserts
    that a source span uniquely identifies an operation.
    """
    def __init__(self,index,resolver):
        self.index=index; self.resolver=resolver; self.observed=defaultdict(list)
        self.cache={}; self.locations={}
        for row in (*index.bindings,*index.protocol_bindings):
            self.observed[row.caller.process_id,row.attempt.site].append(row)

    def _anchor(self,site,unit,context,step):
        return R.SourceActivationAnchor(site,context.site(step),context.owner(step.region),
            tuple(context.value(value) for value in step.arguments),
            tuple(context.emitter.value(value)[1] for value in step.arguments),_completions(context,step))

    @staticmethod
    def _contains(outer,inner):
        return (outer[0],outer[2]) <= (inner[0],inner[2]) and (inner[1],inner[3]) <= (outer[1],outer[3])

    def select(self,process,site):
        key=process,site
        if key in self.cache:return self.cache[key]
        callers=self.index._callers.get((process,site.code),())
        for caller in callers:
            if code_site(caller.code,site.offset) != encode(site):
                raise ValueError('activation caller site differs from its original full compiler code')
        result=[]
        for row in self.observed.get(key,()):
            port=row.operation if hasattr(row,'operation') else row.result
            context=self.resolver.plan(row.unit,port.port.template)
            step=context.steps[port.port.owner]
            if context.owner(step.region) != row.owner:
                raise ValueError('activation anchor differs from the original observed source scope')
            result.append(self._anchor(site,row.unit,context,step))
        if not result and site.position is not None:
            for caller in callers:
                unit=caller.unit; source=self.index.view.unit(unit)
                if unit not in self.locations:
                    syntax=self.index._source_syntax(unit); locations=[]
                    for template in source.material.interfaces['nodes']:
                        try:span=_span(syntax,template.occurrence)
                        except (KeyError,TypeError):continue  # Module roots have no PEP657 coordinates.
                        if span is not None and all(type(v) is int for v in span):locations.append((span,template))
                    self.locations[unit]=tuple(locations)
                candidates=[(span,template) for span,template in self.locations[unit]
                    if self._contains(span,site.position)]
                if not candidates:continue
                # Select code ownership before taking the innermost source
                # location. A compiler instruction may carry the location of
                # syntax evaluated in another scope (e.g. a generator's first
                # iterable); pruning by syntax alone hides its body protocol.
                located=[]
                for _,template in candidates:
                    context=self.resolver.plan(unit,template)
                    for step in context.plan.steps:
                        if context.owner(step.region).owner != caller.scope.owner:continue
                        try:span=_span(self.index._source_syntax(unit),step.source.occurrence)
                        except (KeyError,TypeError):continue
                        if (span is None or any(type(value) is not int for value in span)
                                or not self._contains(span,site.position)):continue
                        types=tuple(context.emitter.value(value)[1] for value in step.arguments)
                        if any(type(t) is ValueType and t.kind is ValueKind.STATE for t in types):
                            located.append((span,self._anchor(site,unit,context,step)))
                result.extend(anchor for span,anchor in located if not any(
                    other!=span and self._contains(span,other) for other,_ in located))
        answer=tuple(sorted(dict.fromkeys(result),key=_digest)); self.cache[key]=answer
        return answer


def derive_body_activations(index):
    """Account for every retained frame entry; no direct-CALL-only filter."""
    if type(index) is not ObservedSourceCallsiteIndex:
        raise TypeError('body activation join requires its original checked call/source index')
    index.check(); links=_links(index.codes); by_code=defaultdict(list); by_site=defaultdict(list)
    registered=set()
    for link in links:
        for capture in link.captures:
            identity=capture.process_id,capture.code.key,link.id
            if identity in registered:continue
            registered.add(identity)
            by_code[capture.process_id,capture.code.key].append((link,capture))
            by_site[capture.process_id,capture.code.key].append(link)
    endings=defaultdict(list)
    for exit_ in index.codes.trace.exits:endings[exit_.invocation].append(exit_)
    resolver=BodyInterfaceResolver(index.view)
    resolver.prepare_many(link.body for entry in index.codes.trace.entries
        for link,_ in by_code.get((entry.invocation.process_id,entry.code),()))
    anchors=_SourceAnchors(index,resolver)
    interfaces={}; activations={}; observations=[]; frontiers=[]; by_invocation=defaultdict(list)
    for entry in index.codes.trace.entries:
        matches=by_code.get((entry.invocation.process_id,entry.code),())
        if not matches:
            frontiers.append(R.SourceBodyActivationFrontier(entry,'EXECUTED_CODE_HAS_NO_SOURCE_BODY_CORRESPONDENCE'))
        for link,capture in matches:
            interface=resolver.resolve(link.body)
            if interface.scope != link.scope:raise ValueError('activation source scope differs from its original body interface')
            if entry.suspended != bool(capture.code.flags & (0x20|0x80|0x200)):
                raise ValueError('activation suspension capability differs from its captured code')
            interface_id=interface.id; interfaces[interface_id]=interface
            path_index=None; caller_links=(); selected=()
            for ordinal,site in enumerate(entry.path):
                parents=by_site.get((entry.invocation.process_id,site.code),())
                if parents:
                    path_index=ordinal; caller_links=tuple(dict.fromkeys(row.id for row in parents))
                    selected=anchors.select(entry.invocation.process_id,site)
                    break
            identity=ProviderID(_digest((SCHEMA,link.id,interface_id,capture.code,entry.path,
                entry.suspended,caller_links,path_index,selected)))
            activation=R.SourceBodyActivation(identity,link.id,interface_id,capture.code,entry.path,
                entry.suspended,caller_links,path_index,selected)
            if activations.setdefault(identity,activation) != activation:raise ValueError('activation identity collision')
            exits=endings.get(entry.invocation,())
            exit_=exits[0] if len(exits)==1 and exits[0].code==entry.code else None
            observations.append(R.SourceBodyActivationObservation(identity,entry,exit_))
            by_invocation[entry.invocation].append(identity)
            if not entry.path_complete:
                frontiers.append(R.SourceBodyActivationFrontier(entry,'RECORDED_CALLER_PATH_INCOMPLETE'))
            if path_index is not None and not selected:
                frontiers.append(R.SourceBodyActivationFrontier(entry,'SOURCE_CALLER_KNOWN_OPERATION_ANCHOR_UNRESOLVED'))
    parents=[]
    for observation in observations:
        entry=observation.entry; caller=entry.caller_invocation or entry.ancestor_invocation
        for parent in by_invocation.get(caller,()):
            parents.append(R.SourceBodyActivationParent(parent,observation.activation,entry.anchor,
                entry.caller_invocation is not None))
    units=defaultdict(lambda:{name:[] for name in TABLES})
    for interface in interfaces.values():units[interface.body.unit]['interfaces'].append(interface)
    by_id={row.id:row for row in interfaces.values()}
    unit_of=lambda identity:by_id[activations[identity].interface].body.unit
    for row in activations.values():units[unit_of(row.id)]['activations'].append(row)
    for row in observations:units[unit_of(row.activation)]['observations'].append(row)
    for row in dict.fromkeys(parents):units[unit_of(row.child)]['parents'].append(row)
    for row in dict.fromkeys(frontiers):
        for identity in by_invocation.get(row.entry.invocation,()):units[unit_of(identity)]['frontiers'].append(row)
    index.check()
    return {unit:{name:tuple(rows[name]) for name in TABLES} for unit,rows in sorted(units.items())},tuple(
        row for row in frontiers if row.entry.invocation not in by_invocation)


def _pins():
    from scar.ir.program import source_observations as source_observed_codes
    from scar.analysis import source_observed_codes as code_reader,source_observed_calls as call_reader
    return {**body_pins(), **{'correspondence_'+name:digest
        for name,digest in code_reader.request_code_pins().items()},
        'activation_producer':sha256(Path(__file__).read_bytes()).hexdigest(),
        'activation_types':sha256(Path(R.__file__).read_bytes()).hexdigest(),
        'correspondence_types':sha256(Path(source_observed_codes.__file__).read_bytes()).hexdigest(),
        'correspondence_reader':sha256(Path(code_reader.__file__).read_bytes()).hexdigest(),
        'call_reader':sha256(Path(call_reader.__file__).read_bytes()).hexdigest()}


def _basis(index):
    return dict(catalog=encode(_artifact(index.view.catalog.path/'result.json',index.view.catalog.report['schema'])),
        trace=index.codes.trace.directory,project_root=index.codes.project_root,code_units=sorted(index.codes.selected_units),
        run_scope=index.codes.targets.source_sha256,target_reader_budget=index.codes.target_reader_budget,
        frame_budget=encode(index.codes.trace.frame_budget))


def _coverage(index,rows,unmatched):
    observations=[row for values in rows.values() for row in values['observations']]
    actual={row.anchor for row in index.codes.trace.entries}; covered={row.entry.anchor for row in observations}
    return dict(recorded_entries=len(actual),covered_entries=len(covered),missing_entries=len(actual-covered),
        shared_activations=sum(len(values['activations']) for values in rows.values()),
        shared_interfaces=sum(len(values['interfaces']) for values in rows.values()),
        unmatched_entries=len(unmatched),occurrences=len(observations),
        source_anchor_frontiers=sum(len(values['frontiers']) for values in rows.values()),
        entry_coverage_complete=covered==actual,
        body_protocols=dict(Counter(type(row.instruction).__name__ for values in rows.values() for row in values['interfaces'])))


def build_body_activations(index,out):
    started=time.monotonic(); out=Path(out).absolute()
    if out.resolve()!=out or out.exists():raise ValueError('activation graph requires fresh canonical output')
    pins=_pins(); basis=_basis(index); rows,unmatched=derive_body_activations(index); _durable_mkdir(out); shards={}
    for unit,values in rows.items():
        folder=out/'units'/unit; _durable_mkdir(folder)
        shards[unit]=dict(path='units/'+unit,tables={name:write_table(folder/(name+'.jsonl.gz'),values[name]) for name in TABLES},
            counts={name:len(values[name]) for name in TABLES})
    missing=write_table(out/'unmatched.jsonl.gz',unmatched)
    report=dict(schema=SCHEMA,schema_version=1,basis=basis,pins=pins,shards=shards,unmatched=missing,
        coverage=_coverage(index,rows,unmatched),target_executed=False,source_bodies_copied=False,
        argument_aliases_proven=False,namespace_instances_bound=False,first_stage_complete=False,
        seconds=time.monotonic()-started)
    if pins!=_pins() or basis!=_basis(index):raise ValueError('activation input or implementation changed during join')
    _durable_bytes(out/'result.json',_json(report),replace=False)
    return report


class SourceBodyActivationStore:
    def __init__(self,path,*,index=None):
        self.path=Path(path).resolve(); self.raw=(self.path/'result.json').read_bytes(); self.report=r=json.loads(self.raw)
        if (r['schema'],r['schema_version'],r['pins'])!=(SCHEMA,1,_pins()):
            raise ValueError('activation schema or implementation differs')
        if any(r[name] is not False for name in ('target_executed','source_bodies_copied',
                'argument_aliases_proven','namespace_instances_bound','first_stage_complete')):
            raise ValueError('activation metadata claims unproved identities or transformations')
        b=r['basis']
        if index is None:
            view=ProgramView(Path(b['catalog']['uri']).parent)
            codes=ObservedSourceCodeProvider(view,b['trace'],b['project_root'],units=b['code_units'],
                frame_budget=decode(CallEdgeBudget,b['frame_budget']),**b['target_reader_budget'])
            index=ObservedSourceCallsiteIndex(codes)
        self.index=index; self.view=index.view; self.rows={}; self.guards={}
        if _basis(index)!=b:raise ValueError('activation source or trace identity differs')
        expected,unmatched=derive_body_activations(index)
        if tuple(expected)!=tuple(r['shards']):raise ValueError('activation unit denominator differs')
        for unit,values in expected.items():
            shard=r['shards'][unit]; folder=self.path/shard['path']; loaded={}
            if shard['path']!='units/'+unit or set(shard['tables'])!=set(TABLES):raise ValueError('activation shard schema differs')
            for name,kind in TABLES.items():
                receipt=shard['tables'][name]; path=folder/receipt['path']; stamp=_stamp(path)
                if receipt['path']!=name+'.jsonl.gz':raise ValueError('activation table location differs')
                loaded[name]=tuple(decode(kind,row,record_tags=P.PRODUCTION_RECORD_TAGS) for row in _table(folder,receipt))
                if _stamp(path)!=stamp:raise ValueError('activation table changed during read')
                self.guards[path]=stamp
            if loaded!=values or shard['counts']!={name:len(values[name]) for name in TABLES}:
                raise ValueError('activation table differs from original frame, source scope or typed body ports')
            self.rows[unit]=loaded
        missing=tuple(decode(R.SourceBodyActivationFrontier,row) for row in _table(self.path,r['unmatched']))
        if missing!=unmatched or r['coverage']!=_coverage(index,expected,unmatched):raise ValueError('activation entry coverage differs')
        self.guards[self.path/'result.json']=_stamp(self.path/'result.json')
        self.guards[self.path/r['unmatched']['path']]=_stamp(self.path/r['unmatched']['path'])
        self.guards.update((Path(module.__file__),_stamp(Path(module.__file__))) for module in (
            __import__(__name__,fromlist=['__file__']),R,B,C,P))
        self.check()

    def check(self):
        if self.raw!=(self.path/'result.json').read_bytes() or any(_stamp(path)!=stamp for path,stamp in self.guards.items()):
            raise ValueError('admitted activation material changed')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog'); parser.add_argument('--trace'); parser.add_argument('--project-root')
    parser.add_argument('--out'); parser.add_argument('--read'); parser.add_argument('--graph-check',action='store_true')
    parser.add_argument('--graph-check-out',help='read back graph using the same admitted source/trace index')
    args=parser.parse_args(argv)
    if args.graph_check:
        if not args.read or not args.out:parser.error('--graph-check requires --read and --out')
        from scar.analysis.source_body_activations import verify_body_activations
        report=verify_body_activations(args.read,args.out)
        print(json.dumps(report,indent=2));return 0
    if args.read:
        store=SourceBodyActivationStore(args.read); report=store.report
    else:
        if not all((args.catalog,args.trace,args.project_root,args.out)):parser.error('--catalog --trace --project-root --out required')
        view=ProgramView(args.catalog); codes=ObservedSourceCodeProvider(view,args.trace,args.project_root)
        index=ObservedSourceCallsiteIndex(codes)
        report=build_body_activations(index,args.out)
        if args.graph_check_out:
            from scar.analysis.source_body_activations import verify_body_activations
            verify_body_activations(args.out,args.graph_check_out,index=index)
    print(json.dumps(report['coverage'],indent=2))
    return 0


# Balanced entry/exit relations for every shared source body protocol.
#
# Unknown input origins remain projections of an explicit dispatch boundary.
# They are never reference aliases, pure calls, or observed normal returns.
_PORT_RULES = {
    ('ingresses','*','parameters','*'):'INPUT',
    ('ingresses','*','saved','*',0):'SAVED_INPUT',
    ('ingresses','*','saved','*',1):'SAVED_SOURCE',
    ('ingresses','*','suspension'):'SUSPENSION_SITE',
    ('completions','*','ports','*'):'OUTPUT',
    ('completions','*','exit'):'EXIT',
    ('suspensions','*','arguments','*'):'HANDOFF',
    ('suspensions','*','step'):'SUSPENSION_SITE',
}


def _port_fields(kind,path=(),active=()):
    if kind is GlobalPort:return {path}
    if kind in active:return set()
    args=get_args(kind)
    if args:
        if get_origin(kind) is tuple:
            if args[-1] is Ellipsis:return _port_fields(args[0],path+('*',),active+(kind,))
            return set().union(*(_port_fields(arg,path+(i,),active+(kind,)) for i,arg in enumerate(args)))
        return set().union(*(_port_fields(arg,path,active+(kind,)) for arg in args))
    if isinstance(kind,type) and is_dataclass(kind):
        return set().union(*(_port_fields(t,path+(name,),active+(kind,)) for name,t in get_type_hints(kind).items()))
    return set()


def body_port_contract():
    """An independent denominator: every GlobalPort field needs a rule."""
    expected=_port_fields(B.SourceBodyInterface)
    if expected!=set(_PORT_RULES):
        raise ValueError('body protocol port rules differ from the complete declared schema')
    return tuple((path,role) for path,role in _PORT_RULES.items())


def _walk_ports(value,path,indices=()):
    if not path:
        yield indices,value
    elif path[0]=='*':
        for index,item in enumerate(value):yield from _walk_ports(item,path[1:],indices+(index,))
    else:
        child=value[path[0]] if type(path[0]) is int else getattr(value,path[0])
        yield from _walk_ports(child,path[1:],indices)


def _state(kind):return type(kind) is ValueType and kind.kind is ValueKind.STATE


class SourceBodyActivationProvider:
    relation_kinds=frozenset((K.CONTAINS,K.IMPLEMENTATION,K.CALL_TARGET,K.CONTROL_FLOW,
        K.OPERAND,K.PROJECTION,K.STATE_INPUT,K.STATE_OUTPUT,K.STATE_INTERFACE,K.COMPLETION,K.RESULT))

    def __init__(self,view,store):
        self.port_rules=body_port_contract()
        self.view=view; self.store=store if type(store) is SourceBodyActivationStore else SourceBodyActivationStore(store)
        if (view.catalog.path,view.catalog.report)!=(self.store.view.catalog.path,self.store.view.catalog.report):
            raise ValueError('activation provider uses another original source catalog')
        self.nodes={}; self.out_edges=defaultdict(list); self.in_edges=defaultdict(list)
        self.interfaces={}; self.activations={}; self.observations=defaultdict(list); self.points={}
        self.frames={}; self.frame_keys={}; self.origins={}; self.code_points={}; self.covered=set()
        self.frame_rows=defaultdict(list)
        for table in self.store.index.codes.body_link_tables():
            for key,row in table.items():self.code_points[row.id]=self.store.index.codes.points[key]
        artifact=_artifact(self.store.path/'result.json',SCHEMA)
        self.locator=SourceArtifactLocator(artifact,artifact.sha256,artifact.size_bytes,'canonical-json')
        for values in self.store.rows.values():
            self.interfaces.update((row.id,row) for row in values['interfaces'])
            self.activations.update((row.id,row) for row in values['activations'])
            for row in values['observations']:self.observations[row.activation].append(row)
        for unit,values in self.store.rows.items():
            for ordinal,row in enumerate(values['activations']):self._attach(unit,ordinal,row)
        # Repeated invocations remain evidence for one shared parent relation.
        parents={(row.parent,row.child,row.direct) for values in self.store.rows.values() for row in values['parents']}
        for parent,child,direct in parents:
            for left in self.points[parent]:
                for right in self.points[child]:
                    self._add(left,right,None,'STATIC',K.CONTROL_FLOW,Meaning.ORDER,
                        ('RECORDED_DIRECT_PARENT' if direct else 'RECORDED_ANCESTOR_WITH_RETAINED_PATH',),None)
        self._check_ports()
        view.register(self)

    @staticmethod
    def _key(point):return point.namespace,point.node

    def _port(self,port):
        point=self.view.source_port(port); self.origins[self._key(point)]=port
        return point

    def _add(self,left,right,frame,flow,kind,meaning,guards,fact):
        row=(left,right,frame,flow,kind,meaning,tuple(guards),fact)
        if row not in self.out_edges[self._key(left)]:
            self.out_edges[self._key(left)].append(row); self.in_edges[self._key(right)].append(row)
            if frame is not None:self.frame_rows[frame].append(row)

    def _node(self,site,namespace,phase,index,owner,kind=None,value_type=None):
        cls=OperationID if kind is V.ProgramEntityKind.OPERATION else ValueID if kind is V.ProgramEntityKind.VALUE else ProviderID
        identity=cls(_digest((SCHEMA+'.port',site.id,phase,index)))
        point=Point(namespace,identity,None)
        self.nodes[self._key(point)]=(site,phase,index,owner,kind,value_type)
        return point

    def _attach(self,unit,ordinal,site):
        interface=self.interfaces[site.interface]
        code=self.code_points[site.code_link]
        namespaces=tuple(dict.fromkeys(self.view.unit(anchor.operation.unit).namespace for anchor in site.anchors))
        if not namespaces:
            namespaces=tuple(dict.fromkeys(self.code_points[key].namespace for key in site.caller_links)) or (code.namespace,)
        shard=self.store.report['shards'][unit]
        fact=_table_ref(self.store.path/shard['path'],shard['tables']['activations'],SCHEMA+'.site',ordinal)
        points=[]
        for namespace in namespaces:
            point=Point(namespace,site.id,None); points.append(point)
            self.nodes[self._key(point)]=(site,'ACTIVATION',None,code,None,None)
            self._add(code,point,None,'STATIC',K.IMPLEMENTATION,Meaning.STRUCTURE,('RECORDED_BODY_CODE',),fact)
            for parent in site.caller_links:
                self._add(self.code_points[parent],point,None,'STATIC',K.CONTROL_FLOW,Meaning.ORDER,('RECORDED_CALLER_PATH',),fact)
            anchors=tuple(anchor for anchor in site.anchors if self.view.unit(anchor.operation.unit).namespace==namespace)
            for ingress_index,ingress in enumerate(interface.ingresses if site.suspended else interface.ingresses[:1]):
                driver=self._node(site,namespace,'INGRESS',ingress_index,point,V.ProgramEntityKind.OPERATION)
                self._add(point,driver,None,None,K.CONTAINS,Meaning.STRUCTURE,(),fact)
                target=self.view.source_region(ingress.region.unit,ingress.region.template,ingress.region.region)
                frame=Frame(namespace,driver.node,None,None,target.namespace,entry=target.node,selection=fact)
                self.frames[site.id,namespace,ingress_index]=frame
                self.frame_keys[frame]=(site.id,namespace,ingress_index)
                guards=('ORIGINAL_BODY_PROTOCOL_SELECTED','PARAMETER_AND_NAMESPACE_INSTANCES_NOT_BOUND')
                if site.suspended:guards+=('START_OR_ACTIVE_RESUME_POSITION_MUST_BE_SELECTED',)
                for anchor in anchors:
                    self._add(self._port(anchor.operation),driver,None,None,K.CONTROL_FLOW,Meaning.ORDER,
                        (*guards,'SOURCE_OPERATION_ALTERNATIVE_SELECTED'),fact)
                    for port,kind in zip(anchor.arguments,anchor.types):
                        self._add(self._port(port),driver,None,None,K.STATE_INPUT if _state(kind) else K.OPERAND,
                            Meaning.STATE_INTERFACE if _state(kind) else Meaning.DEPENDENCE,
                            (*guards,'HIDDEN_PATH_INPUT_DEPENDENCE_NOT_ALIAS'),fact)
                world_type=ingress.types[ingress.world_index]
                world=self._node(site,namespace,'CURRENT_WORLD',ingress_index,driver,V.ProgramEntityKind.VALUE,world_type)
                self._add(driver,world,None,None,K.STATE_OUTPUT,Meaning.STATE_INTERFACE,
                    (*guards,'PREFIX_EFFECTS_UNKNOWN_FRESH_CURRENT_WORLD'),fact)
                self._add(driver,target,frame,'ENTER',K.CALL_TARGET,Meaning.DEPENDENCE,guards,fact)
                saved=dict(ingress.saved)
                for port,kind in zip(ingress.parameters,ingress.types):
                    formal=self._port(port)
                    if port in saved:
                        self._add(self._port(saved[port]),formal,frame,'LOCAL',K.PROJECTION,Meaning.REFERENCE,
                            (*guards,'SAVED_REFERENCE_SAME_ACTIVATION_NOT_CONTENT_SNAPSHOT'),fact)
                    else:
                        self._add(world if _state(kind) else driver,formal,frame,'ENTER',
                            K.STATE_INTERFACE if _state(kind) else K.PROJECTION,
                            Meaning.STATE_INTERFACE if _state(kind) else Meaning.DEPENDENCE,guards,fact)
                for completion_index,completion in enumerate(interface.completions):
                    post=self._node(site,namespace,'COMPLETE',(ingress_index,completion_index),point,V.ProgramEntityKind.OPERATION)
                    selected=(*guards,'BODY_COMPLETION_'+completion.kind.value+'_SELECTED')
                    self._add(self._port(completion.exit),post,frame,'RETURN',K.COMPLETION,Meaning.ORDER,selected,fact)
                    for port,kind in zip(completion.ports,completion.types):
                        self._add(self._port(port),post,frame,'RETURN',K.STATE_INPUT if _state(kind) else K.COMPLETION,
                            Meaning.STATE_INTERFACE if _state(kind) else Meaning.DEPENDENCE,selected,fact)
                    self._outer(post,anchors,selected,fact)
                for suspension_index,suspension in enumerate(interface.suspensions if site.suspended else ()):
                    post=self._node(site,namespace,'HANDOFF',(ingress_index,suspension_index),point,V.ProgramEntityKind.OPERATION)
                    selected=(*guards,'ACTIVE_SUSPENSION_'+_digest(suspension.step)+'_SELECTED')
                    self._add(self._port(suspension.step),post,frame,'RETURN',K.COMPLETION,Meaning.ORDER,selected,fact)
                    for port,kind in zip(suspension.arguments,suspension.types):
                        self._add(self._port(port),post,frame,'RETURN',K.STATE_INPUT if _state(kind) else K.COMPLETION,
                            Meaning.STATE_INTERFACE if _state(kind) else Meaning.DEPENDENCE,selected,fact)
                    self._outer(post,anchors,selected,fact)
        self.points[site.id]=tuple(points)

    def _outer(self,post,anchors,guards,fact):
        for anchor in anchors:
            for completion in anchor.completions:
                selected=(*guards,'WRAPPER_CONTINUATION_AND_OUTER_'+completion.kind.value+'_SELECTED')
                self._add(post,self._port(completion.exit),None,None,K.COMPLETION,Meaning.ORDER,selected,fact)
                for port,kind in zip(completion.ports,completion.types):
                    self._add(post,self._port(port),None,None,K.STATE_OUTPUT if _state(kind) else K.RESULT,
                        Meaning.STATE_INTERFACE if _state(kind) else Meaning.DEPENDENCE,
                        (*selected,'WRAPPER_RESULT_DEPENDENCE_NOT_CALLBACK_RESULT_ALIAS'),fact)

    def _check_ports(self):
        """Check requirements from IR fields against the constructed relation index."""
        for key,frame in self.frames.items():
            site_id,namespace,ingress_index=key; site=self.activations[site_id]; interface=self.interfaces[site.interface]
            ingress=interface.ingresses[ingress_index]
            rows=self.frame_rows[frame]
            inputs={self.origins.get(self._key(row[1])) for row in rows if row[3] in ('ENTER','LOCAL')}
            sources={self.origins.get(self._key(row[0])) for row in rows if row[3] in ('LOCAL','RETURN')}
            required_inputs=set(); required_sources=set()
            for path,role in self.port_rules:
                for indices,port in _walk_ports(interface,path):
                    if port is None:continue
                    if path[0]=='ingresses' and indices[0]!=ingress_index:continue
                    if role=='HANDOFF' and not site.suspended:continue
                    (required_inputs if role in ('INPUT','SAVED_INPUT') else required_sources).add(port)
            target=self.view.source_region(ingress.region.unit,ingress.region.template,ingress.region.region)
            control=any(row[1]==target and row[3]=='ENTER' and row[4] is K.CALL_TARGET for row in rows)
            if not control or not required_inputs<=inputs or not required_sources<=sources:
                raise ValueError('activation graph misses a required original ingress/completion/saved/handoff port')
            self.covered.update((site_id,namespace,ingress_index,port) for port in required_inputs|required_sources)

    def describe(self,point):
        row=self.nodes.get(self._key(point))
        if row is None:return None
        self.store.check(); _,_,_,owner,kind,value_type=row
        return V.ProgramNode(point,kind,V.ProgramNodeRole.PROVIDER,replace(owner,context=point.context),value_type)

    def original(self,point):return self.nodes[self._key(point)][0]
    def expand(self,point):
        site=self.original(point)
        return tuple(edge.target for edge in self.outgoing(point).edges if edge.kind in (K.CONTAINS,K.CALL_TARGET))
    def explain(self,point):
        site=self.original(point)
        return dict(original=site,interface=self.interfaces[site.interface],observations=tuple(self.observations[site.id]),
            argument_aliases_proven=False,namespace_instances_bound=False,resume_position_selected=False,
            callback_result_is_outer_result=False,first_stage_complete=False)

    def _neighbors(self,point,incoming):
        self.store.check(); rows=(self.in_edges if incoming else self.out_edges).get(self._key(point),()); edges=[]
        for left,right,frame,flow,kind,meaning,guards,fact in rows:
            context=point.context
            if flow=='STATIC':
                if context is not None:continue
                source,target=left,right
            elif flow=='ENTER':
                if incoming:
                    popped=context_pop(context,(frame,))
                    if not popped.matched:continue
                    source,target=replace(left,context=popped.remaining),replace(right,context=context)
                else:source,target=replace(left,context=context),replace(right,context=context_push(context,frame))
            elif flow=='RETURN':
                if incoming:source,target=replace(left,context=context_push(context,frame)),replace(right,context=context)
                else:
                    popped=context_pop(context,(frame,))
                    if not popped.matched:continue
                    source,target=replace(left,context=context),replace(right,context=popped.remaining)
            elif flow=='LOCAL':
                if not context_pop(context,(frame,)).matched:continue
                source,target=replace(left,context=context),replace(right,context=context)
            else:source,target=replace(left,context=context),replace(right,context=context)
            conditions=tuple(Condition(SCHEMA,guard,source,n,(source,target)) for n,guard in enumerate(guards))
            evidence=Evidence(SCHEMA+'.original-body-protocol', (self.locator,), (source,target))
            edges.append(Edge(source,target,kind,meaning,(evidence,),conditions,Strength.MAY))
        return Neighborhood(tuple(edges),())

    def outgoing(self,point):return self._neighbors(point,False)
    def incoming(self,point):return self._neighbors(point,True)

    def bind_invocations(self,groups,*,group_facts,check=lambda:None):
        """Attach actual run/InvocationKey routes without changing static frames.

        Groups and their original facts must come from the same admitted
        observed-interface reader/session as this activation store's index.
        """
        from scar.analysis.source_invocation_view import derive_invocation_bindings,SourceInvocationProvider
        def guard():
            self.store.check();check()
        codes=self.store.index.codes
        bindings=derive_invocation_bindings(self.view,groups,
            tuple(row for rows in self.observations.values() for row in rows),self.activations,self.interfaces,
            codes.trace,codes.targets,group_facts=group_facts,check=guard)
        return SourceInvocationProvider(self.view,bindings,check=guard)


def verify_body_activations(path,out,*,index=None):
    """Read stored tables and every balanced relation; reuse an admitted index."""
    from scar.artifacts import _durable_bytes
    from scar.artifacts import _json
    started=time.monotonic(); store=SourceBodyActivationStore(path,index=index)
    provider=SourceBodyActivationProvider(store.view,store)
    checked=0
    for rows in provider.out_edges.values():
        for left,right,frame,flow,*_ in rows:
            source=left
            if flow in ('RETURN','LOCAL'):source=replace(left,context=context_push(None,frame))
            expected_context=context_push(None,frame) if flow in ('ENTER','LOCAL') else None
            matches=[edge for edge in provider.outgoing(source).edges
                if provider._key(edge.target)==provider._key(right) and edge.target.context==expected_context]
            if not matches:raise ValueError('required activation relation is not queryable in its declared frame')
            for edge in matches:
                if edge not in provider.incoming(edge.target).edges:
                    raise ValueError('activation incoming/outgoing queries disagree')
                checked+=1
    # A return cannot escape through an unrelated activation selection.
    checked_frames=0
    for frame,rows in provider.frame_rows.items():
        returns=[row for row in rows if row[3]=='RETURN']
        if not returns:continue
        row=returns[0]; wrong=replace(frame,operation=OperationID(_digest(('unrelated-frame',frame.operation))))
        query=replace(row[0],context=context_push(None,wrong))
        if any(provider._key(edge.target)==provider._key(row[1]) and edge.target.context is None
                for edge in provider.outgoing(query).edges):
            raise ValueError('activation return accepted an unrelated frame')
        checked_frames+=1
    store.check()
    report=dict(schema=SCHEMA+'.graph-check',schema_version=1,status='STRUCTURE_VERIFIED',
        coverage=store.report['coverage'],shared_activations=len(provider.activations),
        ingress_frames=len(provider.frames),required_ports_checked=len(provider.covered),
        bidirectional_edges_checked=checked,unrelated_return_frames_rejected=checked_frames,
        first_stage_complete=False,target_executed=False,source_bodies_copied=False,
        seconds=time.monotonic()-started)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    _durable_bytes(out,_json(report),replace=False)
    return report


__all__ = [
    'SourceBodyActivationProvider',
    'body_port_contract',
    'verify_body_activations',
    'SCHEMA',
    'TABLES',
    'derive_body_activations',
    'build_body_activations',
    'SourceBodyActivationStore',
    'main',
]


if __name__=='__main__':raise SystemExit(main())
