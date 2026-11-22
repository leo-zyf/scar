"""Source control."""

from scar.analysis.source_expression_rules import _REF, _BOOL, _require
from scar.ir.program import (
    source_productions as P,
    normal_graph as G,
    normal_graph as T,
    name_flow as R,
    name_flow as A,
    name_flow as D,
    name_flow as N,
    normal_graph as F,
)
from scar.ir.program.structured import PythonProtocolKind as K
from collections import defaultdict, deque, Counter
from pathlib import Path
from scar.analysis.normal_graph import _Emitter
from scar.analysis.source_references import source_port
from scar.analysis.source_value_flow import project_value_flow, _artifact
from scar.ir.record_codec import encode, decode
import argparse
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from hashlib import sha256
from scar.analysis.source_declared_calls import (
    DeclaredCallStore,
    _NoTargets,
    _check_targets,
    _folder,
    _flat_tables,
)
from scar.analysis.source_graph_material import source_graph_material
from scar.analysis.source_stack_routes import anchor_name_accesses, _bindings
from scar.analysis.source_name_definitions import reaching_name_definitions
from scar.analysis.source_namespace_contents import NamespaceReferenceResolver, _Query
from scar.artifacts import (
    read_record_table as _table,
    write_table,
    _durable_bytes,
    _durable_mkdir,
    _json,
)


# Source control.
# Control-boolean recipes distinct from value-producing Python expressions.
#
# Matches codegen_jump_if's context propagation. In particular, a short-circuit
# operand/result is never returned as an object and then truth-tested again.
def _return_decision(b,parent,source):
    done=b.region(source,(_BOOL,b.world),parent)
    b.finish(done,(b.parameter(done,0),b.parameter(done,1)),(_BOOL,b.world))
    return done


def _branch(b,region,decision,yes,yes_args,no,no_args):
    return b.control(region,P.BranchControl(P.ProductionRegionCall(yes.id,yes_args),
        P.ProductionRegionCall(no.id,no_args)),(decision,),b.union(b.completions(yes),b.completions(no)))


def emit_condition(b,region,world):
    kind=b.node.constructor
    if kind=='BoolOp':
        _boolean(b,region,world)
    elif kind=='IfExp':
        _conditional(b,region,world)
    elif kind=='UnaryOp' and b.selector('UnaryOp',b.site('op'))=='Not':
        b.mark('op',P.ProductionDispositionKind.MARKER)
        b.mark('operand',P.ProductionDispositionKind.EVALUATED_CHILD)
        decision,world=b.condition(region,b.site('operand'),world)
        inverted,=b.emit(region,P.InvertCondition(),(decision,),(_BOOL,),may_raise=False)
        b.finish(region,(inverted,world),(_BOOL,b.world))
    elif kind=='Compare' and len(b.value(b.site('ops')).items)>1:
        _comparison(b,region,world)
    else:
        site=b.role.parents.get(b.occurrence)
        _require(site is not None,'SOURCE_REFERENCE',str(b.occurrence),'condition fallback requires its original value-use slot')
        # Same occurrence, different purpose: VALUE is a distinct child template.
        for name in b.role.fields[kind]:b.mark(name,P.ProductionDispositionKind.DELEGATED_VALUE_RULE)
        b.slots[site]=P.ProductionDispositionKind.EVALUATED_CHILD
        value,world=b.evaluate(region,site,world)
        decision,world=b.protocol(region,K.TRUTH,(value,),world)
        world,=b.protocol(region,K.FRAME_RELEASE_TEMPORARY,(value,),world)
        b.finish(region,(decision,world),(_BOOL,b.world))


def _boolean(b,region,world):
    b.mark('op',P.ProductionDispositionKind.MARKER)
    b.mark('values',P.ProductionDispositionKind.EVALUATED_CHILD)
    selector=b.selector('BoolOp',b.site('op'))
    values=b.value(b.site('values'))
    _require(values.kind=='LIST' and len(values.items)>=2,'RULE_SHAPE',str(b.occurrence),'boolean condition requires at least two operands')
    for ordinal in range(len(values.items)):
        site=b.site('values',ordinal)
        decision,world=b.condition(region,site,world)
        if ordinal==len(values.items)-1:
            b.finish(region,(decision,world),(_BOOL,b.world))
            return
        done=_return_decision(b,region,site)
        follow=b.region(b.site('values',ordinal+1),(b.world,),region)
        # Each continuation requests a condition child with BOOL/world + raise.
        calls=((follow.id,(world,)),(done.id,(decision,world))) if selector=='And' else (
            (done.id,(decision,world)),(follow.id,(world,)))
        b.emit(region,P.BranchControl(*(P.ProductionRegionCall(*call) for call in calls)),(decision,),(_BOOL,b.world))
        region,world=follow,b.parameter(follow,0)


def _conditional(b,region,world):
    for name in ('test','body','orelse'):b.mark(name,P.ProductionDispositionKind.EVALUATED_CHILD)
    decision,world=b.condition(region,b.site('test'),world)
    branches=[]
    for name in ('body','orelse'):
        child=b.region(b.site(name),(b.world,),region)
        result=b.condition(child,b.site(name),b.parameter(child,0))
        b.finish(child,result,(_BOOL,b.world));branches.append(child)
    result=_branch(b,region,decision,branches[0],(world,),branches[1],(world,))
    b.finish(region,result,(_BOOL,b.world))


def _comparison(b,region,world):
    b.mark('left',P.ProductionDispositionKind.EVALUATED_CHILD)
    b.mark('ops',P.ProductionDispositionKind.MARKER)
    b.mark('comparators',P.ProductionDispositionKind.EVALUATED_CHILD)
    operators,comparators=b.value(b.site('ops')),b.value(b.site('comparators'))
    _require(operators.kind==comparators.kind=='LIST' and len(operators.items)==len(comparators.items),
        'RULE_SHAPE',str(b.occurrence),'comparison condition operand/selector lengths differ')
    left,world=b.evaluate(region,b.site('left'),world)
    for ordinal in range(len(operators.items)):
        operator,site=b.site('ops',ordinal),b.site('comparators',ordinal)
        b.selector('Compare',operator)
        right,world=b.evaluate(region,site,world)
        value,world=b.action(region,P.LanguageActionKind.COMPARE,(operator,),(left,right,world))
        decision,world=b.protocol(region,K.TRUTH,(value,),world)
        world,=b.protocol(region,K.FRAME_RELEASE_TEMPORARY,(value,),world)
        if ordinal==len(operators.items)-1:
            b.finish(region,(decision,world),(_BOOL,b.world))
            return
        done=_return_decision(b,region,operator)
        follow=b.region(b.site('comparators',ordinal+1),(_REF,b.world),region)
        b.emit(region,P.BranchControl(P.ProductionRegionCall(follow.id,(right,world)),
            P.ProductionRegionCall(done.id,(decision,world))),(decision,),(_BOOL,b.world))
        region=follow
        left,world=b.parameter(region,0),b.parameter(region,1)


# Loop interfaces and enclosed calls projected from original F4 control.
#
# This is a view of shared recipes, not an unrolled CFG or a content-invariance
# proof. Call target observations remain observations of attempts, not completed
# invocations. No AST or target execution is needed.
def _enclosed(start, adjacency, stops, *, max_states=20000, max_stack=128):
    return {port for port, _ in _control_states(((start, ()),), adjacency, stops,
                                               max_states=max_states, max_stack=max_stack)}


def _control_states(seeds, adjacency, stops, *, max_states=20000, max_stack=128):
    # Shared child templates can return to several distinct callers. A plain
    # port BFS would splice one caller's entry to another caller's return and
    # leak membership past the selected loop. Keep original CallSite stacks;
    # they are structural recipe contexts, not runtime invocation identities.
    seen=set(); pending=deque(seeds)
    while pending:
        node,stack=pending.popleft(); coordinate=node,stack
        if node in stops or coordinate in seen:continue
        if len(seen)>=max_states:raise ValueError('loop membership control-state budget exhausted')
        seen.add(coordinate)
        for edge in adjacency.get(node,()):
            following=stack
            if edge.transition is G.Transition.RESUME:
                # Selecting the same suspended activation needs separate
                # evidence; this query describes normal structural control.
                continue
            if edge.transition is G.Transition.CALL:
                if edge.site is None:raise ValueError('loop membership CALL has no original CallSite')
                if len(stack)>=max_stack:raise ValueError('loop membership recipe-stack budget exhausted')
                following=(*stack,edge.site)
            elif edge.transition is G.Transition.RETURN:
                if not stack or stack[-1]!=edge.site:continue
                following=stack[:-1]
            pending.append((edge.target,following))
    return seen


def loop_dependencies(material, function, *, flow=None):
    """Return original invariant-prefix, carry routes and body CALL ports."""
    unit=material.fact_store.load(material.unit)
    endpoint=source_port(unit,function.value)
    body=T.Body(unit.unit,function.template,function.body)
    flow=project_value_flow(material,endpoint,body) if flow is None else flow
    if flow.source_function!=endpoint or flow.body!=body:
        raise ValueError('loop view requires its original source function and body')
    adjacency=defaultdict(list)
    for edge in flow.edges:
        if edge.kind is G.EdgeKind.CONTROL:adjacency[edge.source].append(edge)
    reachable=_enclosed(G.Port(body.template,body.region,G.PortKind.ENTER),adjacency,set())
    loops=[];members=[]
    for key,plan in material.closure(body.template).items():
        emitter=_Emitter(plan,key,material.interfaces)
        for step in plan.steps:
            ins=step.instruction;header=emitter.port(step.id,G.PortKind.ENTER)
            if type(ins) is not P.LoopControl or header not in reachable:continue
            stops={header,*(emitter.port(step.id,G.PortKind.EXIT,completion=c) for c in step.completions)}
            inside=_enclosed(emitter.port(ins.body,G.PortKind.ENTER),adjacency,stops)
            slots=[]
            for index,value in enumerate(step.arguments):
                formal=emitter.port(step.region,G.PortKind.PARAMETER,index)
                carried=tuple(e for e in flow.edges if e.target==formal and e.transition is G.Transition.BACKEDGE)
                slots.append(dict(index=index,role='INVARIANT_PREFIX' if index<ins.invariant_count else 'CARRIED_SUFFIX',
                    argument=encode(T.GlobalPort(unit.unit,emitter.value(value)[0])),
                    formal=encode(T.GlobalPort(unit.unit,formal)),
                    value_type=encode(emitter.value(value)[1]),backedges=encode(carried),
                    content_invariance='UNPROVED'))
            calls=[]
            for record in flow.calls:
                call=record.call;result=source_port(unit,call.result)
                entry=G.Port(result.port.template,result.port.owner,G.PortKind.ENTER)
                if entry not in inside:continue
                calls.append(dict(fact=encode(record.fact),entry=encode(T.GlobalPort(unit.unit,entry)),
                    callee=encode(source_port(unit,call.callee)),result=encode(result),
                    positional=[encode(source_port(unit,i)) for i in call.positional],
                    keywords=[dict(name=name,value=encode(source_port(unit,i))) for name,i in call.keywords],
                    positional_pack=None if call.positional_pack is None else encode(source_port(unit,call.positional_pack)),
                    keyword_pack=None if call.keyword_pack is None else encode(source_port(unit,call.keyword_pack)),
                    world_before=encode(source_port(unit,call.world)),world_after=encode(source_port(unit,call.result_world))))
            bindings=[]
            def body_access(access):
                port=access.value
                return G.Port(port.template,port.owner,G.PortKind.ENTER) in inside
            for location in flow.locations:
                writes=[a for a in location.writes if body_access(a)]
                reads=[a for a in location.reads if body_access(a)]
                if not writes and not reads:continue
                bindings.append(dict(namespace=encode(location.namespace),reads=encode(reads),writes=encode(writes),
                    direct_binding_role='BODY_WRITTEN' if writes else 'NO_EXPLICIT_BODY_WRITE',
                    referent_content_invariance='UNPROVED',
                    scope='ORIGINAL_LEXICAL_ACCESSES_NOT_SELECTED_REACHING_VERSIONS'))
            loops.append(dict(header=encode(T.GlobalPort(unit.unit,header)),source_loop=encode(ins.source_loop),
                condition=encode(ins.condition,record_tags=P.PRODUCTION_RECORD_TAGS),
                body=encode(T.Body(unit.unit,key,ins.body)),slots=slots,body_calls=calls,lexical_bindings=bindings,
                control_backedges=encode(tuple(e for e in flow.edges if e.kind is G.EdgeKind.CONTROL
                    and e.target==header and e.transition is G.Transition.BACKEDGE)),
                body_operation_count=sum(o.entry in inside for o in flow.operations),parent=None))
            members.append((header,inside))
    for index,(header,_) in enumerate(members):
        parents=[j for j,(_,inside) in enumerate(members) if j!=index and header in inside]
        if parents:loops[index]['parent']=loops[min(parents,key=lambda j:len(members[j][1]))]['header']
    material.validate_parents()
    return dict(source_function=encode(endpoint),body=encode(body),source_material=encode(material.result_artifact),
        loops=loops,iterations_unrolled=False,scope='ORIGINAL_NORMAL_CONTROL_BALANCED_CALL_MAY_MEMBERSHIP',
        contents_proven_invariant=False,call_dispatch_proven=False,
        remaining=['BINDING_CONTENT_VERSIONS_ACROSS_CALLS','EXCEPTIONAL_OR_SUSPENDED_EXECUTION_COVERAGE'])


def source_loop_dependencies(view, source):
    """One source-unit pass, reusing an admitted ProgramView's material."""
    units=sorted(view.catalog.units_for_source(str(Path(source).absolute())) or ())
    rows=[]
    for unit in units:
        provider=view.unit(unit)
        for function in provider.facts.functions:
            rows.append(loop_dependencies(provider.material,function))
    return dict(schema='scar.source-loop-dependencies',schema_version=1,source=str(Path(source).absolute()),
        functions=rows,loop_count=sum(len(r['loops']) for r in rows),
        loop_body_call_count=sum(len(l['body_calls']) for r in rows for l in r['loops']),
        loops_unrolled=False,first_stage_complete=False)


def attach_loop_call_observations(view, report, providers):
    """Attach admitted attempt targets at original loop CALL coordinates.

    A known code body is expandable. Its receiver/argument contents, effects
    and runtime completion are not established by this target observation.
    """
    from scar.ir.record_codec import decode
    attached=0
    for function in report['functions']:
        for loop in function['loops']:
            for call in loop['body_calls']:
                entry=decode(T.GlobalPort,call['entry'])
                operation=view.unit(entry.unit)._node(entry.port.template,entry.port.owner)
                targets=[]
                for provider in providers:
                    if not hasattr(provider,'by_call'):continue
                    for target in provider.query(operation)['targets']:
                        targets.append(target)
                call['observed_target_attempts']=targets
                call['binding_and_effects_proven']=False
                attached+=len(targets)
    report['observed_loop_call_attempts']=attached
    return report


# Qualify declared lexical reference edges with matched control GEN/KILL.
#
# This is a separate NORMAL_SUCCESS branch, not an overwrite of the source
# declarations. Missing exception/resume paths and opaque reference sources
# remain explicit frontiers. Shared loops are solved to a may fixed point;
# neither source-table order nor a runtime trace selects the binding version.
SCHEMA = 'scar.source-control-namespaces'
NO_PROOF = ('source_scopes_activated', 'runtime_dispatch_proven', 'current_binding_versions_proven',
    'tensor_content_versions_proven', 'all_exception_paths_connected', 'source_relations_rederived',
    'loops_unrolled', 'target_executed', 'first_stage_complete')
_WORKER = None


class ControlNamespaceResolver(NamespaceReferenceResolver):
    def __init__(self, snapshot, *, catalog=None, **limits):
        super().__init__(snapshot)
        self.catalog = catalog if catalog is not None else getattr(snapshot, 'catalog', snapshot)
        self.limits = limits
        self.materials = {}; self.anchors = []; self.anchor_ids = {}; self.bindings = {}
        self.tracks = []; self.track_ids = {}; self.flows = {}; self.filters = []; self.filter_ids = {}

    def _material(self, unit):
        if unit not in self.materials:
            material = source_graph_material(self.catalog, unit)
            self.materials[unit] = material
            self.anchor_ids[unit] = len(self.anchors)
            self.anchors.append(anchor_name_accesses(self.snapshot, unit, material=material))
            self.bindings[unit] = _bindings(material, {})[0]
        return self.materials[unit], self.anchors[self.anchor_ids[unit]], self.bindings[unit]

    def filter(self, unit, ordinal):
        fact = self.slicer.fact(unit, 'names', ordinal)
        if fact in self.filter_ids: return self.filter_ids[fact]
        record = unit.names[ordinal]; owners = self.unit(unit.unit)[6][ordinal]
        scope = next(iter(owners)) if len(owners) == 1 else None
        track = None; accesses = (); states = (); writers = (); open_kinds = (); applied = False
        reason = 'LEXICAL_READ_SOURCE_SCOPE_NOT_UNIQUE'
        if scope is not None:
            functions = [f for f in unit.functions if f.body_scope is not None and f.body_scope.owner == scope]
            reason = 'LEXICAL_READ_SOURCE_FUNCTION_NOT_UNIQUE'
            if len(functions) == 1:
                function = functions[0]; material, anchors, bindings = self._material(unit.unit)
                slot = bindings.bindings[record.namespace.owner]
                # Namespace.owner is a C2 BindingSlot index, not CodeScope.ordinal.
                reason = 'CAPTURE_OR_NONLOCAL_SLOT_REQUIRES_SELECTED_CREATION_FRAME'
                if slot.domain == 'LOCAL_BINDING' and slot.owner == function.environment.scope:
                    key = source_port(unit, function.value), record.namespace
                    if key not in self.track_ids:
                        if key[0] not in self.flows:
                            self.flows[key[0]] = project_value_flow(material, key[0],
                                T.Body(unit.unit, function.template, function.body))
                        self.track_ids[key] = len(self.tracks)
                        self.tracks.append(reaching_name_definitions(self.snapshot, *key,
                            material=material, anchor_graph=anchors, flow=self.flows[key[0]], **self.limits))
                    track = self.track_ids[key]; graph = self.tracks[track]
                    accesses = tuple(sorted({a.access for a in anchors.anchors if a.fact == fact and a.scope == scope}))
                    observed = [r for r in graph.reads if r.access in accesses]
                    states = tuple(sorted({r.state for r in observed}))
                    ids = sorted({d for r in observed for d in r.definitions})
                    selected = [graph.definitions[d] for d in ids]
                    writers = tuple(dict.fromkeys(e for d in selected
                        if d.kind in (D.NameDefinitionKind.FORMAL, D.NameDefinitionKind.WRITE)
                        for e in (d.evidence[:1])))
                    open_kinds = tuple(sorted({d.kind for d in selected if d.kind not in
                        (D.NameDefinitionKind.FORMAL, D.NameDefinitionKind.WRITE)}, key=lambda kind: kind.value))
                    anchored = not any(f.fact == fact for f in anchors.frontiers)
                    covered = accesses and {r.access for r in observed} == set(accesses)
                    # A missing WRITE/FORMAL anchor can omit a reaching source
                    # even when this READ and the retained CFG are complete.
                    # Only these two explicitly scoped unknown branches may
                    # coexist with exclusion in the normal control view.
                    covered_track = all(f.reason in {
                        'APPLICATION_EXCEPTION_PATHS_OUTSIDE_NORMAL_SUCCESS_VIEW',
                        'OPAQUE_CALLBACK_AND_CAPTURE_SLOT_IDENTITY_REMAIN_OPEN',
                    } for f in graph.frontiers)
                    applied = bool(anchored and covered and covered_track and ids
                        and graph.traversal_complete and graph.dataflow_complete)
                    reason = ('NORMAL_CONTROL_MAY_DEFINITIONS_JOINED' if applied else
                        'READ_ANCHOR_OR_RETAINED_CONTROL_DATAFLOW_COVERAGE_OPEN')
        value = R.NameReadFilter(fact, scope, track, accesses, states, writers, open_kinds, applied, reason)
        index = len(self.filters); self.filter_ids[fact] = index; self.filters.append(value)
        return index

    def resolve(self, point, *, max_states=4000, max_selectors=32, query=0):
        worker = _ControlQuery(self, point, max_states, max_selectors)
        reference = worker.run()
        return R.ControlNamespaceQuery(query, point, reference, tuple(sorted(worker.used_filters)), tuple(worker.excluded))


class _ControlQuery(_Query):
    def __init__(self, *args):
        super().__init__(*args); self.used_filters = set(); self.excluded = []

    def _filter(self, state, unit, ordinal):
        index = self.resolver.filter(unit, ordinal); value = self.resolver.filters[index]
        self.used_filters.add(index)
        if not value.applied:
            self.frontier(state, 'LEXICAL_NORMAL_CONTROL_FILTER_OPEN_'+value.reason, (value.read,))
        else:
            for kind in value.open_kinds:
                self.frontier(state, 'LEXICAL_REACHING_'+kind.value+'_SOURCE_OPEN', (value.read,))
            self.frontier(state, 'LEXICAL_EXCEPTION_AND_RESUME_SOURCE_BRANCH_OPEN', (value.read,))
        return index, value

    def edge(self, state, point, selectors, kind, evidence, conditions):
        if kind is N.NamespaceEdgeKind.LEXICAL_WRITE:
            writer, read = evidence
            unit, local = self.resolver.slicer.local(self.states[state].port)
            candidates = [(ordinal, namespace) for namespace, ordinal in self.resolver.unit(unit.unit)[2].get(local, ())
                if self.resolver.slicer.fact(unit, 'names', ordinal) == read]
            if len(candidates) != 1: raise ValueError('lexical edge lost its original read fact')
            index, value = self._filter(state, unit, candidates[0][0])
            if value.applied:
                if writer not in value.writers:
                    self.excluded.append(R.ExcludedNameWriter(state, writer, read, index)); return
                conditions = (*conditions, *R.NORMAL_VIEW)
        super().edge(state, point, selectors, kind, evidence, conditions)

    def run(self):
        result = super().run()
        # A lexical read without a declared writer still needs its unknown
        # entry/delete/opaque branch; do not depend on edge() being called.
        for index, state in enumerate(self.states):
            unit, local = self.resolver.slicer.local(state.port)
            for namespace, ordinal in self.resolver.unit(unit.unit)[2].get(local, ()):
                if namespace.kind is F.NamespaceKind.LEXICAL:
                    self._filter(index, unit, ordinal)
        return replace(result, frontiers=tuple(self.frontiers))


def project_control_namespaces(store, unit, *, frontier=None, **limits):
    original = store.load_unit(unit)
    resolver = ControlNamespaceResolver(store.snapshot, catalog=store.catalog, **limits)
    selected = [i for i, q in enumerate(original.queries)
        if frontier is None or any(f.reason == frontier for f in q.reference.frontiers)]
    queries = tuple(resolver.resolve(original.queries[i].callee, query=i, **store.report['limits']) for i in selected)
    store.validate()
    return _unit(unit, tuple(resolver.anchors), tuple(resolver.tracks), tuple(resolver.filters), queries)


def _unit(unit, anchors, tracks, filters, queries):
    """Intern original function control/operation records once per unit shard."""
    bodies = []; indexes = {}; shared = []
    for track in tracks:
        body = R.NameControlBody(track.source_function, track.function, track.body, track.scope,
            track.control_edges, track.operations)
        if body.source_function not in indexes:
            indexes[body.source_function] = len(bodies); bodies.append(body)
        common = bodies[indexes[body.source_function]]
        if common != body: raise ValueError('one source function has incompatible control body records')
        shared.append(replace(track, control_edges=common.control_edges, operations=common.operations))
    return R.ControlNamespaceUnit(unit, anchors, tuple(bodies), tuple(shared), filters, queries)


def _pins():
    from scar.analysis import source_stack_routes as source_name_access_anchors, source_name_definitions, source_graph_material, source_value_flow
    from scar.analysis import source_namespace_contents as source_namespace_interfaces, source_world_flow as source_world_flow, source_productions
    return dict(producer=sha256(Path(__file__).read_bytes()).hexdigest(), **{
        name: sha256(Path(module.__file__).read_bytes()).hexdigest() for name, module in (
            ('types', R), ('anchors', source_name_access_anchors), ('anchor_types', A),
            ('definitions', source_name_definitions), ('definition_types', D),
            ('materials', source_graph_material), ('value_flow', source_value_flow),
            ('namespace_interfaces', source_namespace_interfaces), ('world_flow', source_world_flow),
            ('productions', source_productions))})


def _counts(graph):
    return dict(queries=len(graph.queries), anchors=len(graph.anchors), tracks=len(graph.tracks),
        shared_bodies=len(graph.bodies), shared_body_operations=sum(len(b.operations) for b in graph.bodies),
        shared_body_control_edges=sum(len(b.control_edges) for b in graph.bodies),
        filters=len(graph.filters), applied_filters=sum(f.applied for f in graph.filters),
        excluded_writer_edges=sum(len(q.excluded) for q in graph.queries),
        namespace_states=sum(len(q.reference.states) for q in graph.queries),
        namespace_edges=sum(len(q.reference.edges) for q in graph.queries),
        control_states=sum(len(g.states) for g in graph.tracks),
        control_transitions=sum(len(g.transitions) for g in graph.tracks),
        definitions=sum(len(g.definitions) for g in graph.tracks),
        backedges=sum(sum(e.transition.value == 'BACKEDGE' for e in g.control_edges) for g in graph.tracks),
        frontier_reasons=dict(Counter(f.reason for q in graph.queries for f in q.reference.frontiers)))


def _track_rows(graph):
    for track in graph.tracks:
        anchor = next(i for i, value in enumerate(graph.anchors) if value is track.anchor_graph)
        body = next(i for i, value in enumerate(graph.bodies) if value.source_function == track.source_function)
        yield R.StoredNameDefinitionTrack(anchor, body, track.namespace, track.definitions, track.states,
            track.transitions, track.reads, track.frontiers, track.traversal_complete, track.dataflow_complete,
            track.conditions, track.loops_unrolled, track.semantics)


def _write_unit(store, unit, out, frontier, limits, pins):
    started = time.monotonic(); graph = project_control_namespaces(store, unit, frontier=frontier, **limits)
    folder = Path(out)/_folder(unit); _durable_mkdir(folder)
    tables = {name: write_table(folder/(name+'.jsonl.gz'), rows) for name, rows in (
        ('anchors', graph.anchors), ('bodies', graph.bodies), ('tracks', _track_rows(graph)),
        ('filters', graph.filters), ('queries', graph.queries))}
    if pins != _pins(): raise ValueError('control name projection rules changed during build')
    store.runtime.clear(); _check_targets()
    return dict(path=_folder(unit), tables=tables, counts=_counts(graph), seconds=time.monotonic()-started)


def _initialize(declared, out, frontier, limits, pins):
    global _WORKER
    sys.meta_path.insert(0, _NoTargets()); _check_targets()
    _WORKER = DeclaredCallStore(declared), out, frontier, limits, pins


def _worker(unit):
    store, out, frontier, limits, pins = _WORKER
    return unit, _write_unit(store, unit, out, frontier, limits, pins)


def _selected(report, units, frontier):
    selected = set(report['shards']) if not units else set(units)
    if not selected <= set(report['shards']): raise ValueError('control names unit is outside declared source selection')
    return tuple(sorted(u for u in selected if frontier is None or
        report['shards'][u]['counts']['frontier_reasons'].get(frontier, 0)))


def _totals(shards):
    counts = Counter(); reasons = Counter()
    for shard in shards.values():
        for key, value in shard['counts'].items():
            if key == 'frontier_reasons': reasons.update(value)
            else: counts[key] += value
    return dict(units=len(shards), **counts, frontier_reasons=dict(reasons))


def build_control_namespaces(declared, out, *, units=(), frontier=None, jobs=4,
        max_states=20000, max_stack=128, max_updates=1000000):
    started = time.monotonic(); pins = _pins(); out = Path(out).absolute()
    if out.exists(): raise ValueError('fresh control namespace output required')
    if type(jobs) is not int or jobs < 1: raise ValueError('control namespace jobs must be positive')
    limits = dict(max_states=max_states, max_stack=max_stack, max_updates=max_updates)
    if any(type(n) is not int or n < 1 for n in limits.values()): raise ValueError('positive control name budgets required')
    store = DeclaredCallStore(declared); selection = _selected(store.report, units, frontier)
    parent = _artifact(store.path/'result.json', store.report['schema']); _durable_mkdir(out)
    if jobs == 1:
        shards = {u: _write_unit(store, u, out, frontier, limits, pins) for u in selection}
    else:
        with ProcessPoolExecutor(max_workers=min(jobs, len(selection) or 1), mp_context=multiprocessing.get_context('spawn'),
                initializer=_initialize, initargs=(str(store.path), str(out), frontier, limits, pins)) as pool:
            shards = dict(pool.map(_worker, selection))
    store.validate()
    if pins != _pins(): raise ValueError('control namespace projection rules changed')
    report = dict(schema=SCHEMA, schema_version=2, declared=str(store.path), parent=encode(parent), pins=pins,
        selection=dict(units=list(units), frontier=frontier), limits=limits, shards=shards, tables=_flat_tables(shards),
        counts=_totals(shards), jobs=jobs, seconds=time.monotonic()-started,
        scope='RECORDED_NORMAL_CONTROL_BRANCH_ONLY_ORIGINAL_DECLARATIONS_RETAINED_IN_PARENT',
        **{name: False for name in NO_PROOF})
    _durable_bytes(out/'result.json', _json(report), replace=False)
    return report


class ControlNamespaceStore:
    def __init__(self, path):
        self.path = Path(path).absolute(); self.raw = (self.path/'result.json').read_bytes(); self.report = r = json.loads(self.raw)
        if (r['schema'], r['schema_version']) != (SCHEMA, 2) or r['pins'] != _pins():
            raise ValueError('control namespace schema or rules differ')
        self.parent = DeclaredCallStore(r['declared'])
        if encode(_artifact(self.parent.path/'result.json', self.parent.report['schema'])) != r['parent']:
            raise ValueError('control namespace declared parent differs')
        if tuple(r['shards']) != _selected(self.parent.report, **r['selection']):
            raise ValueError('control namespace source selection differs')
        if (r['tables'] != _flat_tables(r['shards']) or r['counts'] != _totals(r['shards'])
                or any(r[n] is not False for n in NO_PROOF)
                or r['scope'] != 'RECORDED_NORMAL_CONTROL_BRANCH_ONLY_ORIGINAL_DECLARATIONS_RETAINED_IN_PARENT'):
            raise ValueError('control namespace receipt index, scope or proof status differs')
        for unit, shard in r['shards'].items():
            if shard['path'] != _folder(unit) or set(shard['tables']) != {'anchors', 'bodies', 'tracks', 'filters', 'queries'}:
                raise ValueError('control namespace shard paths or tables differ')
            if any(row['path'] != name+'.jsonl.gz' for name, row in shard['tables'].items()):
                raise ValueError('control namespace table path differs')

    def read_unit(self, unit):
        if self.raw != (self.path/'result.json').read_bytes() or self.report['pins'] != _pins():
            raise ValueError('control namespace report or rules changed')
        shard = self.report['shards'][unit]; folder = self.path/shard['path']; tables = shard['tables']
        def rows(name, cls):
            return tuple(decode(cls, wire, record_tags=P.PRODUCTION_RECORD_TAGS) for wire in _table(folder, tables[name]))
        anchors = rows('anchors', A.NameAccessAnchorGraph); bodies = rows('bodies', R.NameControlBody); tracks = []
        for value in rows('tracks', R.StoredNameDefinitionTrack):
            if not 0 <= value.anchor < len(anchors) or not 0 <= value.body < len(bodies):
                raise ValueError('factored name track body/anchor reference differs')
            body = bodies[value.body]
            tracks.append(D.NameDefinitionGraph(body.source_function, body.function, body.body, body.scope,
                value.namespace, anchors[value.anchor], body.control_edges, body.operations, value.definitions,
                value.states, value.transitions, value.reads, value.frontiers, value.traversal_complete,
                value.dataflow_complete, value.conditions, value.loops_unrolled, value.semantics))
        graph = R.ControlNamespaceUnit(unit, anchors, bodies, tuple(tracks),
            rows('filters', R.NameReadFilter), rows('queries', R.ControlNamespaceQuery))
        expected = project_control_namespaces(self.parent, unit, frontier=self.report['selection']['frontier'], **self.report['limits'])
        if graph != expected or _counts(graph) != shard['counts']:
            raise ValueError('control-qualified source names differ from original control/namespace facts')
        if self.raw != (self.path/'result.json').read_bytes(): raise ValueError('control namespace report changed during read')
        return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declared'); parser.add_argument('--out'); parser.add_argument('--read')
    parser.add_argument('--unit', action='append', default=[]); parser.add_argument('--frontier')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if args.read:
        if len(args.unit) != 1: parser.error('--read requires one --unit to rederive')
        graph = ControlNamespaceStore(args.read).read_unit(args.unit[0])
        print(json.dumps(dict(status='CONTROL_QUALIFIED_NAMES_REDERIVED', counts=_counts(graph))), flush=True)
    else:
        if not args.declared or not args.out: parser.error('--declared and --out required')
        report = build_control_namespaces(args.declared, args.out, units=args.unit, frontier=args.frontier, jobs=args.jobs)
        print(json.dumps(dict(status='CONTROL_QUALIFIED_NAMES_BUILT', counts=report['counts'], seconds=report['seconds'])), flush=True)


if __name__ == '__main__': main()
