"""Loop interfaces and enclosed calls projected from original F4 control.

This is a view of shared recipes, not an unrolled CFG or a content-invariance
proof. Call target observations remain observations of attempts, not completed
invocations. No AST or target execution is needed.
"""
from collections import defaultdict, deque
from pathlib import Path

from scar.analysis.normal_graph import _Emitter
from scar.analysis.source_reference_slice import source_port
from scar.analysis.source_value_flow import project_value_flow
from scar.ir.program import normal_graph as G, normal_targets as T, source_productions as P
from scar.ir.record_codec import encode


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
