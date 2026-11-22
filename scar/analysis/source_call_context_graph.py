"""Source call context graph."""

import argparse
import gzip
import json
import time
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from scar.analysis.source_body_expansion import ExpansionParents, read_body_expansion
from scar.analysis.call_inputs import ContextReferenceEngine
from scar.analysis.source_class_base_graph import project_constructor_interfaces
from scar.analysis.source_field_interfaces import _InterfaceOnly
from scar.analysis.source_function_connections import Invocation, connect
from scar.analysis.source_references import source_port
from scar.analysis.source_world_returns import _table_ref
from scar.analysis.source_value_flow import _artifact
from scar.artifacts import (
    write_table,
    _durable_bytes,
    _durable_mkdir,
    read_record_table as _table,
    _json,
)
from scar.ir.program import (
    call_inputs as X,
    class_ports as C,
    source_productions as P,
    source_value_routes as S,
    binding_ports as R,
    declared_contexts as D,
    normal_graph as T,
    view as V,
)
from scar.ir.record_codec import encode, decode
from dataclasses import dataclass, replace
from types import SimpleNamespace as NS
from scar.analysis.program_view import _digest
from scar.analysis.source_call_view import source_call_site, prepare_source_ports
from scar.compiler.call_binding import (
    read_call_binding_plan,
    plan_call_bindings,
    join_call_bindings,
    plan_digest,
)
from scar.ir.program.call_edges import CapturedSite, CapturedCode
from scar.ir.program.normal_graph import GlobalPort
from scar.ir.program.ids import ProviderID
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.program.dependencies import (
    DependencyPoint as Point,
    DependencyEdge as Edge,
    DependencyKind as K,
    DependencyMeaning as Meaning,
    DependencyStrength as Strength,
    DependencyCondition as Condition,
    DependencyEvidence as Evidence,
    DependencyFrontier as Frontier,
    DependencyNeighborhood as Neighborhood,
)


# Keep selected caller contexts and constructor stages over shared bodies.
class CallContextParents:
    def __init__(self, path):
        started = time.monotonic(); self.root = Path(path).absolute()
        header = json.loads((self.root/'result.json').read_bytes())
        self.base = ExpansionParents(header['interfaces'])
        self.report, self.graph = read_body_expansion(self.root, parents=self.base)
        self.raw = (self.root/'result.json').read_bytes()
        if json.loads(self.raw) != self.report: raise ValueError('call context source body parent changed')
        self.artifact = _artifact(self.root/'result.json', self.report['schema'])
        self.bodies = (*self.base.values.bodies, *self.graph.bodies)
        consumer = Path(self.base.values.report['consumers'])
        self.consumer_report = json.loads((consumer/'result.json').read_bytes()); self.consumer = consumer
        if encode(_artifact(consumer/'result.json', 'scar.reference-consumer-graph')) != self.base.values.report['parent']:
            raise ValueError('call context original consumer parent differs')
        self.bindings = tuple(decode(R.SourceConsumerBinding, row, record_tags=P.PRODUCTION_RECORD_TAGS)
            for row in _table(consumer, self.consumer_report['tables']['bindings']))
        self.entries = tuple(decode(R.ConsumerBodyEntry, row, record_tags=P.PRODUCTION_RECORD_TAGS)
            for row in _table(consumer, self.consumer_report['tables']['entries']))
        self.validate(); self.certification_seconds = time.monotonic()-started

    def validate(self):
        from scar.analysis.source_body_expansion import _pins
        if self.raw != (self.root/'result.json').read_bytes() or self.report['pins'] != _pins():
            raise ValueError('call context certified body report or rules changed')
        for root, tables in ((self.root, self.report['tables']), (self.consumer, self.consumer_report['tables'])):
            for receipt in tables.values():
                raw = (root/receipt['path']).read_bytes()
                if len(raw) != receipt['bytes'] or sha256(raw).hexdigest() != receipt['sha256']:
                    raise ValueError('call context certified source table changed')
        self.base.validate()


def _function_connections(engine, queries, sites):
    result = []; mini_cache = {}
    for qi, (query, (identity, record)) in enumerate(zip(queries, sites)):
        caller = engine.slicer.unit(identity)[0]; call = record.call
        offset = engine.slicer.store.offsets[identity]
        invocation = Invocation(offset+call.callee, tuple(offset+p for p in call.positional),
            tuple((name, offset+p) for name, p in call.keywords), offset+call.result, offset+call.world,
            offset+call.result_world, None if call.positional_pack is None else offset+call.positional_pack,
            None if call.keyword_pack is None else offset+call.keyword_pack)
        for oi, origin in enumerate(query.reference.origins):
            if origin.kind is not S.ReferenceOriginKind.FUNCTION_DEFINITION: continue
            point = query.reference.states[origin.state].port
            provider, local = engine.slicer.local(point)
            fn = next(f for f in provider.functions if f.value == local)
            if point not in mini_cache: mini_cache[point] = _InterfaceOnly(engine.slicer, provider, fn)
            mini = mini_cache[point]
            connection = connect(mini, mini.offsets[provider.unit]+fn.value, invocation,
                conditions=('SELECTED_INCOMING_CALL_CONTEXT_AND_REFERENCE_ORIGIN_PATH',
                    'SOURCE_FUNCTION_CREATION_AND_DEFAULT_VERSION_MATCH_DECLARATION',
                    'PARENT_BODY_ENTRY_CONDITIONS_SELECTED_NOT_RUNTIME_ACTIVATION'))
            if connection is not None: result.append(X.ContextFunctionConnection(qi, oi, connection))
    return tuple(result)


def _declared_incoming(engine, binding):
    """Adapt the original declared edge without masquerading as a history row."""
    evidence = binding.interface
    if evidence.artifact.schema != 'scar.declared-context-edges':
        raise ValueError('incoming interface uses an unsupported original evidence schema')
    raw = Path(evidence.artifact.uri).read_bytes()
    if len(raw) != evidence.artifact.size_bytes or sha256(raw).hexdigest() != evidence.artifact.sha256:
        raise ValueError('declared incoming interface artifact changed')
    prefix = '#/records/'; tail = evidence.locator.removeprefix(prefix)
    if not evidence.locator.startswith(prefix) or not tail.isdecimal() or str(int(tail)) != tail:
        raise ValueError('declared incoming interface requires its original canonical record ordinal')
    rows = gzip.decompress(raw).splitlines(); ordinal = int(tail)
    if not 0 <= ordinal < len(rows): raise ValueError('declared incoming interface record is absent')
    edge = decode(D.DeclaredContextEdge, json.loads(rows[ordinal]), record_tags=P.PRODUCTION_RECORD_TAGS)
    unit = engine.slicer.unit(binding.state.actuals[0].unit)[0]
    call_ordinal = int(edge.call.locator.rsplit('/', 1)[1]); call = unit.calls[call_ordinal]
    if edge.call != engine.slicer.fact(unit, 'calls', call_ordinal):
        raise ValueError('declared incoming source call evidence belongs to a different unit')
    connection = edge.connection
    if (connection.source_callable != binding.source_callable or connection.body != binding.body
            or binding.argument not in connection.parameters or binding.state not in connection.parameters):
        raise ValueError('declared incoming body or original argument/state route differs')
    mode = 'SOURCE_CLASS_METHOD' if edge.descriptor.kind is D.DescriptorKind.CLASSMETHOD else (
        'SOURCE_BOUND_METHOD' if edge.descriptor.kind is D.DescriptorKind.FUNCTION
        and edge.receiver.kind is D.ClassValueKind.INSTANCE else 'SOURCE_FUNCTION')
    link = T.CallLink(source_port(unit, call.result), source_port(unit, call.callee),
        connection.source_callable, connection.body, mode, None, connection.parameters, connection.returns,
        tuple(dict.fromkeys((*connection.conditions, *edge.conditions))), argument_packs=connection.argument_packs,
        projections=connection.projections, binding=connection.binding)
    return engine.admit(evidence, link)


def derive_contexts(parents):
    snapshot = parents.base.values.snapshot; engine = ContextReferenceEngine(snapshot)
    inputs = []
    for ordinal, entry in enumerate(parents.entries):
        binding = parents.bindings[entry.binding]
        inputs.append((entry.body, binding, _table_ref(parents.consumer,
            parents.consumer_report['tables']['entries'], 'scar.consumer-body-entries', ordinal)))
    for ordinal, entry in enumerate(parents.graph.entries):
        binding = parents.base.graph.bindings[entry.binding]
        inputs.append((entry.body, binding, _table_ref(parents.root, parents.report['tables']['entries'],
            'scar.expanded-body-entries', ordinal)))
    units = {b.body.unit for b in parents.bodies}
    units.update(actual.unit for _, binding, _ in inputs for actual in binding.state.actuals)
    engine.index(units)
    grouped = {}; roots = []
    for body_index, binding, entry in inputs:
        if binding.interface not in engine.ids: _declared_incoming(engine, binding)
        incoming = engine.ids[binding.interface]; link = engine.interfaces[incoming].link
        body = parents.bodies[body_index]
        if link.source_callable != body.source_function or link.body != body.body:
            raise ValueError('call context incoming interface belongs to a different shared body')
        grouped.setdefault((body_index, incoming), []).append(entry)
    for (body, incoming), entries in grouped.items():
        roots.append(X.CallContextRoot(body, incoming, tuple(entries)))
    requested = []; metadata = []; sites = []
    for root_id, root in enumerate(roots):
        body = parents.bodies[root.body]; unit = engine.slicer.unit(body.body.unit)[0]
        for record in body.calls:
            point = source_port(unit, record.call.callee)
            requested.append((root.incoming, point)); metadata.append((root_id, root.incoming, record.fact, point))
            sites.append((body.body.unit, record))
    references = engine.solve(requested)
    queries = tuple(X.CallContextQuery(incoming, fact, point, reference, root)
        for (root, incoming, fact, point), reference in zip(metadata, references))
    assessments = []
    for qi, (query, (identity, record)) in enumerate(zip(queries, sites)):
        known = {query.reference.states[o.state].port for o in query.reference.origins
            if o.kind in (S.ReferenceOriginKind.FUNCTION_DEFINITION, S.ReferenceOriginKind.CLASS_DEFINITION)}
        result = source_port(engine.slicer.unit(identity)[0], record.call.result)
        for candidate in engine.by_result.get(result, ()):
            link = engine.interfaces[candidate].link
            if link.callee != query.callee: continue
            relation = 'ORIGIN_MATCH_UNDER_PATH_CONDITIONS' if link.source_callable in known else (
                'CONTEXT_OPEN' if query.reference.frontiers else 'NOT_IN_CLOSED_SOURCE_ORIGINS')
            assessments.append(X.ContextTargetEvidence(qi, candidate, relation))
    functions = _function_connections(engine, queries, sites)
    # Constructor protocols cover every distinct original body call, including
    # those not selected as a callee query in a particular incoming context.
    unique = {}; constructor_sites = []
    for body in parents.bodies:
        for record in body.calls:
            key = body.body.unit, record.fact
            if key not in unique:
                unique[key] = True; constructor_sites.append((body.body.unit, record))
    constructors = project_constructor_interfaces(snapshot, tuple(constructor_sites))
    parents.validate()
    return dict(interfaces=tuple(engine.interfaces), roots=tuple(roots), queries=queries,
        assessments=tuple(assessments), functions=functions,
        constructor_sites=constructors.sites, constructor_steps=constructors.steps,
        constructor_entries=constructors.entries, constructor_frontiers=constructors.frontiers)


CLASSES = dict(interfaces=X.CallInterfaceRecord, roots=X.CallContextRoot, queries=X.CallContextQuery,
    assessments=X.ContextTargetEvidence, functions=X.ContextFunctionConnection,
    constructor_sites=C.ConstructorInterfaceSite, constructor_steps=C.ConstructorInterfaceStep,
    constructor_entries=C.ConstructorBodyEntry, constructor_frontiers=C.ConstructorInterfaceFrontier)


def _counts(graph):
    return dict(interfaces=len(graph['interfaces']), roots=len(graph['roots']), queries=len(graph['queries']),
        reference_states=sum(len(q.reference.states) for q in graph['queries']),
        reference_edges=sum(len(q.reference.edges) for q in graph['queries']),
        reference_origins=sum(len(q.reference.origins) for q in graph['queries']),
        context_frontiers=sum(len(q.reference.frontiers) for q in graph['queries']),
        frontier_reasons=dict(Counter(f.reason for q in graph['queries'] for f in q.reference.frontiers)),
        assessments=dict(Counter(a.relation for a in graph['assessments'])),
        source_function_connections=len(graph['functions']),
        constructor_sites=len(graph['constructor_sites']), constructor_steps=len(graph['constructor_steps']),
        constructor_entries=len(graph['constructor_entries']), constructor_frontiers=len(graph['constructor_frontiers']),
        constructor_stages=dict(Counter(s.link.stage.value for s in graph['constructor_steps'])))


def _pins():
    from scar.analysis import call_inputs as source_call_contexts, source_class_base_graph as source_constructor_interfaces, source_function_connections
    return {name: sha256(Path(path).read_bytes()).hexdigest() for name, path in (
        ('producer', __file__), ('types', X.__file__), ('constructor_types', C.__file__),
        ('declared_types', D.__file__),
        ('contexts', source_call_contexts.__file__), ('constructors', source_constructor_interfaces.__file__),
        ('function_connections', source_function_connections.__file__))}


def _parent(path, parents):
    parents = parents if parents is not None else CallContextParents(path)
    if parents.root != Path(path).absolute(): raise ValueError('call context parent path differs')
    parents.validate(); return parents


def build_call_context_graph(path, out, *, parents=None):
    started = time.monotonic(); pins = _pins(); out = Path(out).absolute()
    if out.exists(): raise ValueError('fresh call context output required')
    parents = _parent(path, parents); graph = derive_contexts(parents)
    if pins != _pins(): raise ValueError('call context rules changed')
    _durable_mkdir(out)
    tables = {name: write_table(out/(name+'.jsonl.gz'), rows) for name, rows in graph.items()}
    report = dict(schema='scar.source-call-context-graph', schema_version=1,
        status='SELECTED_SOURCE_CALL_CONTEXTS_AND_CONSTRUCTOR_INTERFACES', body_expansion=str(parents.root),
        parent=encode(parents.artifact), tables=tables, pins=pins, counts=_counts(graph),
        all_historical_candidates_retained=True, runtime_dispatch_proven=False,
        content_versions_proven=False, target_executed=False, first_stage_complete=False,
        seconds=time.monotonic()-started)
    _durable_bytes(out/'result.json', _json(report), replace=False); return report


def read_call_context_graph(path, *, parents=None):
    path = Path(path).absolute(); raw = (path/'result.json').read_bytes(); report = json.loads(raw)
    if (report['schema'], report['schema_version']) != ('scar.source-call-context-graph', 1) or report['pins'] != _pins():
        raise ValueError('call context producer differs')
    parents = _parent(report['body_expansion'], parents)
    if report['parent'] != encode(parents.artifact): raise ValueError('call context original parent differs')
    graph = {name: tuple(decode(cls, row, record_tags=P.PRODUCTION_RECORD_TAGS)
        for row in _table(path, report['tables'][name])) for name, cls in CLASSES.items()}
    if graph != derive_contexts(parents) or _counts(graph) != report['counts']:
        raise ValueError('selected caller references or constructor interfaces differ from original evidence')
    if (report['all_historical_candidates_retained'] is not True or any(report[k] is not False for k in (
            'runtime_dispatch_proven', 'content_versions_proven', 'target_executed', 'first_stage_complete'))):
        raise ValueError('call contexts cannot prove runtime dispatch or state contents')
    parents.validate()
    if (path/'result.json').read_bytes() != raw: raise ValueError('call context graph changed during verification')
    return report, graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bodies'); parser.add_argument('--out'); parser.add_argument('--read')
    args = parser.parse_args()
    if args.read: report, _ = read_call_context_graph(args.read)
    else:
        if not args.bodies or not args.out: parser.error('--bodies and --out required')
        report = build_call_context_graph(args.bodies, args.out)
    print(json.dumps(dict(status=report['status'], counts=report['counts'], seconds=report['seconds'])), flush=True)


# Bounded observed-target metadata on original F4 CALL operations.
#
# Provider handles identify retained attempt records, never callable instances,
# new library bodies, native implementations or completed Python invocations.
SCHEMA='scar.source-bounded-call-targets'
_OPEN=('caller_invocation','callee_entry','callee_completion','argument_layout_and_defaults',
    'creator_environment_and_receiver_contents','array_content_versions','native_schema_and_effects',
    'unobserved_paths_and_future_targets')


@dataclass(frozen=True,slots=True)
class SourceObservedCallTarget:
    id: ProviderID
    source_call: GlobalPort
    caller_site: CapturedSite
    run: str
    process_id: int
    thread_id: int
    attempt_index: int
    target_kind: str
    target_code: CapturedCode | None
    metadata_json: str
    evidence: ProviderFactReference
    remaining_fields: tuple[str,...]=_OPEN

    def __post_init__(self):
        if (type(self.id)is not ProviderID or type(self.source_call)is not GlobalPort
                or type(self.caller_site)is not CapturedSite
                or type(self.run)is not str or len(self.run)!=64
                or type(self.process_id)is not int or self.process_id<1
                or type(self.thread_id)is not int or self.thread_id<0
                or type(self.attempt_index)is not int or self.attempt_index<1
                or self.target_kind not in ('PYTHON_FUNCTION','BOUND_PYTHON_METHOD','NATIVE_CARRIER','TYPE_OBJECT','OTHER_CALLABLE')
                or type(self.evidence)is not ProviderFactReference):
            raise ValueError('original bounded CALL attempt metadata required')
        self.caller_site.__post_init__();self.evidence.__post_init__()
        if (self.target_kind in ('PYTHON_FUNCTION','BOUND_PYTHON_METHOD'))!=(self.target_code is not None):
            raise ValueError('only observed Python function carriers have a CodeID')
        if self.target_code is not None:
            if type(self.target_code)is not CapturedCode:raise ValueError('exact target CodeID descriptor required')
            self.target_code.__post_init__()


class SourceCallBindingProvider:
    relation_kinds=frozenset((K.CALL_TARGET,K.IMPLEMENTATION))

    def __init__(self,view,plan_path,observation_paths):
        self.view=view;self.materials={};self.nodes={};self.by_call=defaultdict(list)
        self.out_edges=defaultdict(list);self.in_edges=defaultdict(list);self.operations={};self.sources={}
        self.body_targets={};self.body_frontiers={};self.body_keys=set()
        from scar.analysis.function_interface_observations import RetainedCodeBodyIndex
        self.body_index=RetainedCodeBodyIndex(view)
        self.function_interfaces=None
        self.plan_path=Path(plan_path).absolute();self.plan=read_call_binding_plan(self.plan_path)
        self._read(self.plan_path)
        catalog=self.plan['catalog'];catalog_path=view.catalog.path/'result.json'
        if (catalog.get('path')!=str(catalog_path) or catalog.get('sha256')!=sha256(catalog_path.read_bytes()).hexdigest()
                or catalog.get('bytes')!=catalog_path.stat().st_size):
            raise ValueError('bounded CALL plan uses another original source catalog')
        if isinstance(observation_paths,(str,Path)):observation_paths=(observation_paths,)
        documents=[];anchors={}
        for path in observation_paths:
            path=Path(path).absolute();raw=self._read(path);doc=json.loads(raw);documents.append(doc)
            artifact=ProviderArtifact(str(path),sha256(raw).hexdigest(),len(raw),doc['schema'])
            for index,row in enumerate(doc['records']):
                key=doc['process_id'],row['index']
                if key in anchors:raise ValueError('same bounded attempt occurs in multiple raw documents')
                anchors[key]=(artifact,index)
        joined=join_call_bindings(self.plan,documents)
        # Independently replay the existing F4/bytecode mapper. Observed JSON
        # is not permission to redirect an original call to another offset.
        expected={}
        bodies=() if self.plan['selector']is None else self.plan['selector']['bodies']
        path_by_code={body['code']['key']:body['source']['path'] for body in bodies}
        by_path=defaultdict(set)
        for row in self.plan['sites']:
            by_path[path_by_code[row['code_key']]].add(row['query'])
        budget=self.plan['max_targets_per_site']
        for path,queries in by_path.items():
            replay=plan_call_bindings(view,path,call_indices=queries,max_targets_per_site=budget)
            for row in replay['sites']:expected[row['id'],row['code_key'],row['offset']]=row
            if any(row['source']['path']==path and row['code']['qualname']=='<module>' for row in bodies):
                replay=plan_call_bindings(view,path,entry='<module>',call_indices=queries,max_targets_per_site=budget)
                for row in replay['sites']:expected[row['id'],row['code_key'],row['offset']]=row
        for row in self.plan['sites']:
            if expected.get((row['id'],row['code_key'],row['offset']))!=row:
                raise ValueError('bounded CALL plan differs from current original F4/source correspondence')
        sites={row['id']:row for row in self.plan['sites']}
        ports=[]
        for row in sites.values():
            facts=view.unit(row['unit']).facts;call=facts.calls[row['query']]
            if (call.callee,call.result,call.world,call.result_world)!=(row['callee_port'],row['result_port'],row['world_port'],row['result_world_port']):
                raise ValueError('bounded target changed original callee/result/world ports')
            ports.extend(source_port(facts,i) for i in (call.callee,call.result,call.world,call.result_world))
        prepare_source_ports(view,ports)
        for row in sites.values():
            source=view.unit(row['unit']);self.sources[source.unit]=source;facts=source.facts;call=facts.calls[row['query']]
            operation,_,_=source_call_site(view,NS(result=source_port(facts,call.result),callee=source_port(facts,call.callee)))
            if view.original(operation).id.wire!=row['original_step']:raise ValueError('bounded CALL operation differs')
            self.operations[row['id']]=operation
            self.by_call.setdefault((operation.namespace,operation.node),[])
        for binding in joined['bindings']:
            selected=sites[binding['site_id']];operation=self.operations[binding['site_id']]
            artifact,index=anchors[binding['process_id'],binding['attempt_index']]
            evidence=ProviderFactReference(artifact,'#/records/'+str(index))
            locator=SourceArtifactLocator(artifact,artifact.sha256,artifact.size_bytes,'canonical-json',evidence.locator)
            identity=ProviderID(_digest((SCHEMA,plan_digest(self.plan),artifact,evidence.locator)))
            metadata=binding['target'];code=None if metadata.get('code')is None else decode(CapturedCode,metadata['code'])
            source=self.sources[selected['unit']];call=source.facts.calls[selected['query']]
            record=SourceObservedCallTarget(identity,source_port(source.facts,call.result),decode(CapturedSite,selected['site']),
                artifact.sha256,binding['process_id'],binding['thread_id'],binding['attempt_index'],metadata['kind'],code,
                json.dumps(metadata,sort_keys=True,separators=(',',':'),allow_nan=False),evidence)
            point=Point(operation.namespace,identity,None);self.nodes[point.namespace,identity]=(point,operation,record)
            self.by_call[operation.namespace,operation.node].append(point)
            guards=('RETAINED_CALL_ATTEMPT_METADATA_SELECTED','RUN_SHA256_'+record.run,
                'RECORDED_PROCESS_'+str(record.process_id),'RECORDED_THREAD_'+str(record.thread_id),
                'NO_CALLER_INVOCATION_ENTRY_COMPLETION_OR_FUTURE_DISPATCH_PROOF')
            edge=Edge(operation,point,K.CALL_TARGET,Meaning.DEPENDENCE,
                (Evidence(SCHEMA+'.attempt',(locator,source.locator),(operation,point)),),
                tuple(Condition(SCHEMA,guard,point,index=record.attempt_index) for guard in guards),Strength.MAY)
            self.out_edges[operation.namespace,operation.node].append(edge);self.in_edges[point.namespace,point.node].append(edge)
            body,reason=self._target_body(record.target_code)
            self.body_targets[point.namespace,point.node]=body
            if reason is not None:self.body_frontiers[point.namespace,point.node]=reason
            if body is not None:
                # This expands the observed implementation's static body. It
                # does not invent an invocation, formal binding or return edge.
                original_source=self.view.namespaces[body.namespace]
                locators=(locator,original_source.locator)
                implementation=Edge(point,body,K.IMPLEMENTATION,Meaning.STRUCTURE,
                    (Evidence(SCHEMA+'.original-code-body',locators,(point,body)),),
                    tuple(Condition(SCHEMA,guard,point,index=record.attempt_index) for guard in
                        ('EXACT_RETAINED_CODE_DESCRIPTOR_MATCH','SOURCE_BODY_IS_NOT_ACTIVATION',
                         'CREATOR_ENVIRONMENT_ARGUMENT_BINDING_AND_NATIVE_EFFECTS_REMAIN_OPEN')),Strength.MAY)
                self.out_edges[point.namespace,point.node].append(implementation)
                self.in_edges[body.namespace,body.node].append(implementation)
                self.body_keys.add((body.namespace,body.node))
        self.summary=joined;self.check();view.register(self)

    def attach_function_interfaces(self):
        """Project conditional source ABIs; an attempt is never an invocation.

        The existing function provider owns signature/default/capture and
        balanced ENTER/RETURN semantics. This metadata provider continues to
        retain the complete, bounded attempts independently of that view.
        """
        if self.function_interfaces is None:
            from scar.analysis.function_interface_observations import BoundedFunctionInterfaceStore
            from scar.analysis.source_call_view import SourceFunctionCallProvider
            self.function_interfaces=SourceFunctionCallProvider(self.view,BoundedFunctionInterfaceStore(self))
        return self.function_interfaces

    def _target_body(self,code):
        """Join original compiler paths, including retained replay filenames."""
        if code is None:return None,None
        result=self.body_index.match(code)
        self.sources.update(self.body_index.sources)
        return result

    def _read(self,path):
        if path.is_symlink() or not path.is_file() or not 0<path.stat().st_size<=2_000_000:
            raise ValueError('bounded regular original observation/plan required')
        raw=path.read_bytes();self.materials[path]=raw;return raw

    def check(self):
        self.body_index.check()
        if any(path.read_bytes()!=raw for path,raw in self.materials.items()):
            raise ValueError('original bounded CALL materials changed')
        if json.loads(self.materials[self.plan_path])!=self.plan:
            raise ValueError('original bounded CALL plan changed in memory')
        for source in self.sources.values():source.check()

    def describe(self,point):
        value=self.nodes.get((point.namespace,point.node))
        if value is None:return None
        self.check();_,owner,_=value
        return V.ProgramNode(point,None,V.ProgramNodeRole.PROVIDER,replace(owner,context=point.context),None)

    def original(self,point):
        value=self.nodes.get((point.namespace,point.node))
        if value is None:return None
        self.check();return value[2]

    def expand(self,point):return ()

    def _neighbors(self,point,incoming):
        key=point.namespace,point.node
        if key not in self.nodes and key not in self.by_call and key not in self.body_keys:return Neighborhood()
        self.check()
        if point.context is not None:
            return Neighborhood((),(Frontier(point,'BOUNDED_CALL_CONTEXT_NOT_BOUND_TO_ATTEMPT',
                'A static DependencyFrame is not the recorded caller InvocationKey.'),))
        edges=(self.in_edges if incoming else self.out_edges).get(key,())
        if key in self.body_keys:return Neighborhood(tuple(edges))
        frontiers=tuple(Frontier(point,'BOUNDED_CALL_FIELD_OPEN',field) for field in _OPEN)
        if key in self.body_frontiers:
            frontiers+=(Frontier(point,'BOUNDED_CALL_BODY_OPEN',self.body_frontiers[key]),)
        if key in self.by_call and not self.by_call[key]:
            frontiers+=(Frontier(point,'SELECTED_CALL_TARGET_NOT_OBSERVED','No retained attempt supplies this selected site.'),)
        return Neighborhood(tuple(edges),frontiers)

    def incoming(self,point):return self._neighbors(point,True)
    def outgoing(self,point):return self._neighbors(point,False)

    def query(self,call):
        self.check();key=call.namespace,call.node
        return dict(schema=SCHEMA,call=encode(call),targets=[dict(point=encode(p),record=encode(self.original(p)),
            metadata=json.loads(self.original(p).metadata_json),
            source_body=encode(self.body_targets.get((p.namespace,p.node))),
            source_body_frontier=self.body_frontiers.get((p.namespace,p.node)))
            for p in self.by_call.get(key,())] if call.context is None else [],
            remaining_fields=list(_OPEN),static_context_bound_to_recorded_attempt=False,
            conditional_source_interfaces_attached=self.function_interfaces is not None,first_stage_complete=False)

    def explain(self,point):
        record=self.original(point)
        return dict(original=record,remaining_fields=list(_OPEN),
            code_correspondence=None if record.target_code is None else self.body_index.explain(record.target_code),
            entry_completion_or_effects_proven=False)


__all__ = [
    'SourceObservedCallTarget',
    'SourceCallBindingProvider',
    'CallContextParents',
    'derive_contexts',
    'CLASSES',
    'build_call_context_graph',
    'read_call_context_graph',
    'main',
    'SCHEMA',
]


if __name__ == '__main__': main()
