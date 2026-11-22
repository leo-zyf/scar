"""Loop motion facts over typed array bodies and original F4 control windows.

Traversal states use existing recipe CallSites solely to match CALL/RETURN.
They are not new source blocks, runtime invocations or unrolled iterations.
All-path results concern the retained normal-control model and preserve the
separate obligations for exceptions, atomic termination, effects and guards.
"""
from collections import defaultdict, deque
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

from scar.analysis.normal_graph import _Emitter
from scar.analysis.source_control import loop_dependencies
from scar.analysis.source_references import ReferenceSlicer, source_port
from scar.analysis.source_value_flow import project_value_flow
from scar.compiler.model import validate_region
from scar.ir.program import normal_graph as G, normal_graph as F, normal_graph as T
from scar.ir.program import source_productions as P
from scar.ir.program.providers import ProviderArtifact
from scar.ir.record_codec import decode, encode
from scar.optimization.graph import value_kinds
from scar.optimization.resource_solver import semantic_sha256


SCHEMA = 'scar.source-motion-window'


def _port(value):
    return value if type(value) is T.GlobalPort else decode(T.GlobalPort, value)


def _artifact(path, schema):
    path = Path(path).resolve(); raw = path.read_bytes()
    return encode(ProviderArtifact(str(path), sha256(raw).hexdigest(), len(raw), schema))


def _avoiding_cycle(states, successors):
    zeros = {state for state in states if state[2] == 0}
    indegree = {state:0 for state in zeros}
    for state in zeros:
        for target in successors.get(state, ()):
            if target in zeros:indegree[target] += 1
    pending = deque(state for state in zeros if not indegree[state]); removed = 0
    while pending:
        state = pending.popleft(); removed += 1
        for target in successors.get(state, ()):
            if target not in zeros:continue
            indegree[target] -= 1
            if not indegree[target]:pending.append(target)
    return removed != len(zeros)


def _transport_slice(flow, seeds, *, max_states):
    """Select original DATA edges only; balanced witnesses are a later query."""
    incoming = defaultdict(list)
    for index, edge in enumerate(flow.edges):
        if edge.kind is G.EdgeKind.DATA:incoming[edge.target].append((index, edge))
    pending = deque(seeds); seen = set(); selected = set(); complete = True
    while pending:
        port = pending.popleft()
        if port in seen:continue
        if len(seen) >= max_states:
            complete = False; break
        seen.add(port)
        for index, edge in incoming.get(port, ()):
            selected.add(index); pending.append(edge.source)
    return tuple(flow.edges[index] for index in sorted(selected)), len(seen), complete


def _control_walk(start, adjacency, *, boundary, header, selected=None,
                  max_states=20000, max_stack=128):
    """Query original edges, tracking no more than two selected CALL visits."""
    def coordinate(node, stack, hits):
        return node, stack, min(2, hits + int(selected is not None and node == selected))
    initial = coordinate(start, (), 0)
    pending = deque((initial,)); seen = set(); ports = set(); used = set()
    successors = defaultdict(set); terminals = []; frontier = []
    repeated = False; selected_seen = False
    while pending:
        state = pending.popleft()
        if state in seen:continue
        if len(seen) >= max_states:
            frontier.append(dict(reason='CONTROL_STATE_BUDGET_EXHAUSTED', max_states=max_states)); break
        seen.add(state); node, stack, hits = state; ports.add(node)
        selected_seen |= hits > 0
        if hits >= 2:
            repeated = True; continue
        if not stack and boundary(node):
            terminals.append(dict(port=node, hits=hits, kind='ORIGINAL_ITERATION_OR_LOOP_EXIT')); continue
        followed = False
        for edge in adjacency.get(node, ()):
            if edge.transition is G.Transition.RESUME:
                frontier.append(dict(reason='SUSPENDED_ACTIVATION_NOT_SELECTED', edge=edge)); continue
            following = stack
            if edge.transition is G.Transition.CALL:
                if edge.site is None:
                    frontier.append(dict(reason='RECIPE_CALL_SITE_UNAVAILABLE', edge=edge)); continue
                if len(stack) >= max_stack:
                    frontier.append(dict(reason='RECIPE_STACK_BUDGET_EXHAUSTED', edge=edge)); continue
                following = (*stack, edge.site)
            elif edge.transition is G.Transition.RETURN:
                if not stack or stack[-1] != edge.site:continue
                following = stack[:-1]
            followed = True; used.add(edge)
            if edge.transition is G.Transition.BACKEDGE and edge.target == header:
                terminals.append(dict(port=edge.source, hits=hits, kind='SELECTED_LOOP_BACKEDGE')); continue
            target = coordinate(edge.target, following, hits)
            successors[state].add(target); pending.append(target)
        if not followed:
            frontier.append(dict(reason='NO_MATCHED_NORMAL_CONTROL_SUCCESSOR', port=node))
    return dict(ports=ports, edges=used, states=seen, terminals=terminals, frontier=frontier,
        complete=not frontier, selected_seen=selected_seen, repeated=repeated,
        avoiding_cycle=_avoiding_cycle(seen, successors))


def derive_source_motion_window(view, *, unit, function_ordinal, loop_header, call_entry,
                                max_states=20000, max_stack=128):
    """Project one original loop; source admission remains the parent's duty."""
    if (type(max_states) is not int or max_states < 1 or type(max_stack) is not int or max_stack < 1):
        raise ValueError('positive explicit structural control-query budgets required')
    source = view.unit(unit); facts = source.facts
    if type(function_ordinal) is not int or not 0 <= function_ordinal < len(facts.functions):
        raise ValueError('one original function ordinal is required')
    function = facts.functions[function_ordinal]
    if function.body_scope is None:raise ValueError('original function body scope is unavailable')
    header, selected = _port(loop_header), _port(call_entry)
    if (header.unit != unit or selected.unit != unit or header.port.kind is not G.PortKind.ENTER
            or selected.port.kind is not G.PortKind.ENTER):
        raise ValueError('loop and CALL must use original ENTER ports in the selected unit')
    function_body = T.Body(unit, function.template, function.body)
    source_function = source_port(facts, function.value)
    flow = project_value_flow(source.material, source_function, function_body)
    plans = source.material.closure(function_body.template)
    original = []
    for key, plan in plans.items():
        emitter = _Emitter(plan, key, source.material.interfaces)
        original.extend((key, emitter, step) for step in plan.steps)
    matched = [(key, emitter, step) for key, emitter, step in original
        if emitter.port(step.id, G.PortKind.ENTER) == header.port
        and type(step.instruction) is P.LoopControl]
    if len(matched) != 1:raise ValueError('loop header is not one original LoopControl')
    key, emitter, step = matched[0]; instruction = step.instruction
    body = T.Body(unit, key, instruction.body)
    body_entry = G.Port(key, instruction.body, G.PortKind.ENTER)
    selected_records = [record for record in flow.calls if G.Port(
        source_port(facts, record.call.result).port.template,
        source_port(facts, record.call.result).port.owner, G.PortKind.ENTER) == selected.port]
    if len(selected_records) != 1:raise ValueError('selected CALL is not an original CALL of this function')
    frontier = []
    try:
        dependencies = loop_dependencies(source.material, function, flow=flow)
        rows = [row for row in dependencies['loops'] if row['header'] == encode(header)]
        if len(rows) != 1 or not any(row['entry'] == encode(selected) for row in rows[0]['body_calls']):
            raise ValueError('selected CALL is not in the original selected loop body')
    except ValueError as error:
        if 'budget' not in str(error):raise
        frontier.append(dict(reason='ORIGINAL_LOOP_MEMBERSHIP_BUDGET_EXHAUSTED', detail=str(error)))
    adjacency = defaultdict(list)
    for edge in flow.edges:
        if edge.kind is G.EdgeKind.CONTROL:adjacency[edge.source].append(edge)
    exits = tuple(emitter.port(step.id, G.PortKind.EXIT, completion=c) for c in step.completions)
    window = _control_walk(header.port, adjacency, boundary=lambda p:p in exits,
        header=header.port, max_states=max_states, max_stack=max_stack)
    entered = _control_walk(body_entry, adjacency,
        boundary=lambda p:p.template == key and p.owner == body.region and p.kind is G.PortKind.EXIT,
        header=header.port, selected=selected.port, max_states=max_states, max_stack=max_stack)
    frontier.extend(window['frontier']); frontier.extend(entered['frontier'])
    if entered['complete'] and selected.port not in entered['ports']:
        raise ValueError('selected CALL is not reached in the original loop body')
    gp = lambda port:T.GlobalPort(unit, port)
    active_ports = window['ports'] | entered['ports']
    active = {(port.template, port.owner) for port in active_ports}
    body_members = {(port.template, port.owner) for port in entered['ports']}
    operations = []; excluded_exceptional = []
    for template, context, operation in original:
        enter = context.port(operation.id, G.PortKind.ENTER)
        if (template, operation.id) not in active:continue
        inputs = [dict(port=encode(gp(context.port(operation.id, G.PortKind.ARGUMENT, i))),
            value=encode(gp(context.value(value)[0])),
            value_type=encode(context.value(value)[1])) for i, value in enumerate(operation.arguments)]
        outputs = [dict(port=encode(gp(context.port(operation.id, G.PortKind.RESULT, i, completion))),
            value_type=encode(value_type)) for completion in operation.completions
            for i, value_type in enumerate(completion.types)]
        for completion in operation.completions:
            if not context.allowed(completion, operation.region):
                excluded_exceptional.append(dict(entry=encode(gp(enter)),
                    completion=encode(completion, record_tags=P.PRODUCTION_RECORD_TAGS),
                    reason='ORIGINAL_NORMAL_EMITTER_OMITS_NON_ITERATOR_EXHAUSTION_RAISE'))
        operations.append(dict(index=len(operations), entry=encode(gp(enter)),
            step=encode(operation.id), region=encode(operation.region),
            arguments=encode(operation.arguments, record_tags=P.PRODUCTION_RECORD_TAGS),
            completions=encode(operation.completions, record_tags=P.PRODUCTION_RECORD_TAGS),
            instruction=encode(operation.instruction, record_tags=P.PRODUCTION_RECORD_TAGS),
            inputs=inputs, outputs=outputs,
            membership=['BODY' if (template, operation.id) in body_members else 'CONDITION_OR_EXIT']))
    slicer = ReferenceSlicer(SimpleNamespace(fact_store=view.catalog.fact_store))
    call_ordinals = {slicer.fact(facts, 'calls', i):i for i in range(len(facts.calls))}
    def call_row(record):
        call = record.call; result = source_port(facts, call.result)
        return dict(fact=encode(record.fact), query=call_ordinals[record.fact],
            entry=encode(gp(G.Port(result.port.template, result.port.owner, G.PortKind.ENTER))),
            callee=encode(source_port(facts, call.callee)), result=encode(result),
            positional=[encode(source_port(facts, i)) for i in call.positional],
            keywords=[dict(name=name, value=encode(source_port(facts, i))) for name, i in call.keywords],
            positional_pack=None if call.positional_pack is None else encode(source_port(facts, call.positional_pack)),
            keyword_pack=None if call.keyword_pack is None else encode(source_port(facts, call.keyword_pack)),
            world_before=encode(source_port(facts, call.world)), world_after=encode(source_port(facts, call.result_world)))
    calls = [call_row(record) for record in flow.calls if (
        source_port(facts, record.call.result).port.template, source_port(facts, record.call.result).port.owner) in active]
    bindings = []
    for location in flow.locations:
        reads = [dict(value=encode(gp(access.value)), fact=encode(access.fact)) for access in location.reads
            if (access.value.template, access.value.owner) in active]
        writes = [dict(value=encode(gp(access.value)), fact=encode(access.fact)) for access in location.writes
            if (access.value.template, access.value.owner) in active]
        if reads or writes:
            bindings.append(dict(namespace=encode(location.namespace), scope=encode(location.scope),
                reads=reads, writes=writes, reference_binding_is_not_referent_content=True))
    accesses = []
    for group in facts.scope_groups:
        if group.owner != function.body_scope.owner:continue
        for span in group.tables:
            if span.table not in (F.FactTable.NAMES,F.FactTable.ATTRIBUTES,F.FactTable.ITEMS,F.FactTable.CALLS):continue
            for ordinal in range(span.start, span.stop):
                record = getattr(facts, span.table.value)[ordinal]
                indices = ([record.value] if span.table is F.FactTable.NAMES else
                    [record.world, record.result_world])
                if not any(index is not None and (source_port(facts,index).port.template,
                        source_port(facts,index).port.owner) in active for index in indices):continue
                accesses.append(dict(table=span.table.value, ordinal=ordinal,
                    fact=encode(slicer.fact(facts, span.table.value, ordinal)), record=encode(record)))
    selected_record = selected_records[0].call
    seed_indices = (selected_record.callee, *selected_record.positional,
        *(value for _, value in selected_record.keywords),
        *(value for value in (selected_record.positional_pack, selected_record.keyword_pack) if value is not None))
    transport_seeds = tuple(dict.fromkeys(source_port(facts,index).port for index in seed_indices))
    transports, transport_ports, transport_complete = _transport_slice(flow, transport_seeds, max_states=max_states)
    transport_frontier = ([] if transport_complete else
        [dict(reason='REFERENCE_TRANSPORT_SLICE_BUDGET_EXHAUSTED',max_states=max_states)])
    missing = []
    avoids = any(row['hits']==0 for row in entered['terminals'])
    if avoids:missing.append('NORMAL_ENTERED_BODY_PATH_AVOIDS_SELECTED_CALL')
    if entered['avoiding_cycle']:missing.append('NORMAL_CYCLE_CAN_AVOID_SELECTED_CALL')
    if entered['repeated']:missing.append('SELECTED_CALL_CAN_REPEAT_IN_ONE_ENTERED_BODY')
    if frontier:missing.append('NORMAL_CONTROL_WINDOW_OR_MEMBERSHIP_INCOMPLETE')
    if not entered['selected_seen']:missing.append('SELECTED_CALL_NOT_REACHED_WITHIN_QUERY')
    proven = not missing
    control_edges = window['edges'] | entered['edges']
    backedges = [edge for edge in flow.edges if edge.kind is G.EdgeKind.CONTROL
        and edge.transition is G.Transition.BACKEDGE and edge.target == header.port]
    def frontier_wire(row):
        return {name:encode(gp(value)) if type(value) is G.Port else encode(value)
            for name, value in row.items()}
    source.check(); source.material.validate_parents()
    model = view.catalog.fact_store.model_units[unit]
    result = dict(schema=SCHEMA, schema_version=1,
        request=dict(unit=unit, function_ordinal=function_ordinal, loop_header=encode(header),
            call_entry=encode(selected), max_states=max_states, max_stack=max_stack),
        unit=unit, function_ordinal=function_ordinal,
        function_fact=encode(slicer.fact(facts,'functions',function_ordinal)), source_function=encode(source_function),
        function=dict(ordinal=function_ordinal, template=encode(function.template), body=encode(function_body),
            scope=encode(function.body_scope.owner)), source_version=dict(path=model['path'],sha256=model['raw_sha256']),
        source_material=encode(flow.source_templates), loop_header=encode(header),
        loop=dict(header=encode(header), body=encode(body), body_entry=encode(gp(body_entry)),
            header_instruction=encode(instruction,record_tags=P.PRODUCTION_RECORD_TAGS),
            otherwise=encode(instruction.otherwise),
            condition=encode(instruction.condition,record_tags=P.PRODUCTION_RECORD_TAGS),
            control_backedges=encode(backedges), latches=[encode(gp(edge.source)) for edge in backedges],
            exits=[encode(gp(port)) for port in exits], zero_trip='PRESERVED_NOT_BODY_ENTRY'),
        selected_call=call_row(selected_records[0]), operations=operations,
        lexical_bindings=bindings, accesses=accesses, call_sites=calls,
        reference_transports=[dict(source=encode(gp(edge.source)),target=encode(gp(edge.target)),
            kind=edge.kind.value,value_type=encode(edge.value_type),site=encode(edge.site),
            transition=edge.transition.value,
            condition=None if edge.condition is None else encode(gp(edge.condition)),when=edge.when)
            for edge in transports],
        reference_transport_scope='SELECTED_CALL_VALUES_BACKWARD_ORIGINAL_DATA_SLICE_NOT_PRIMITIVE_OR_CONTENT_EQUIVALENCE',
        reference_transport_slice=dict(complete=transport_complete,seeds=[encode(gp(port)) for port in transport_seeds],
            visited_ports=transport_ports,edges=len(transports),max_states=max_states,
            balanced_path_selection_required=True,frontier=transport_frontier),
        control_edges=[encode(edge) for edge in sorted(control_edges,key=lambda edge:semantic_sha256(encode(edge)))],
        excluded_exceptional_completions=excluded_exceptional,
        entered_body_exits=[dict(port=encode(gp(row['port'])),selected_visits=row['hits'],kind=row['kind'])
            for row in entered['terminals']],
        coverage=dict(window_complete=window['complete'] and not frontier,
            body_complete=entered['complete'] and not frontier,
            control_scope='ORIGINAL_NORMAL_CONTROL_BALANCED_RECIPE_CALL_RETURN',
            exceptional_complete=False,resume_complete=False,preheader_operations_included=False),
        must_execute=dict(status='CONDITIONAL' if proven else 'UNRESOLVED',
            fact='ONE_PER_ENTERED_BODY' if proven else None, missing=missing,
            normal_paths_avoid_selected=avoids, selected_repeat_reachable=entered['repeated'],
            avoiding_cycle_reachable=entered['avoiding_cycle'], termination_proven=False),
        conditional_obligations=['ORIGINAL_BODY_ITERATION_HAS_BEEN_ENTERED',
            'PRECEDING_ATOMIC_OPERATIONS_COMPLETE_NORMALLY',
            'EXCEPTIONAL_FP_AND_FAILURE_ORDERING_REQUIRES_SEPARATE_LEGALITY',
            'NORMAL_CONTROL_GUARDS_AND_CALL_CONTEXT_SELECTED',
            'NO_CONTENT_STABILITY_ALIAS_EFFECT_OR_RUNTIME_INVOCATION_PROOF'],
        frontier=[frontier_wire(row) for row in frontier],
        source_refs=[encode(flow.source_templates),encode(slicer.fact(facts,'functions',function_ordinal).artifact)],
        producer=_artifact(__file__,'scar.python-producer'),
        source_bodies_copied=False,loops_unrolled=False,new_cfg_or_ssa_created=False,
        runtime_invocations_created=False,effects_closed=False,source_replacement_authorized=False,
        runtime_guards_discharged=False,target_executed=False,first_stage_complete=False)
    result['id'] = semantic_sha256(result)
    return result


def verify_source_motion_window(report, view):
    expected = derive_source_motion_window(view, **report['request'])
    if semantic_sha256(expected) != semantic_sha256(report):
        raise ValueError('source motion window differs from original F4 control/material replay')
    return expected


def _motion_origin_entry(origin):
    """Recover the original ENTER from its retained typed result coordinate."""
    results=[decode(T.GlobalPort, value) for value in origin['results']]
    candidates=[value for value in results if value.unit==origin['unit']
        and value.port.owner.wire==origin['step']]
    if not candidates or len({(value.port.template,value.port.owner) for value in candidates})!=1:
        raise ValueError('motion origin has no unique original step coordinate')
    point=candidates[0]
    return T.GlobalPort(point.unit,G.Port(point.port.template,point.port.owner,G.PortKind.ENTER))


def project_loop_source_motion(view, proof, loop_origin, call_origin):
    """Project the original control window and argument routes after F4 admission."""
    identity=next(row for row in proof.source.origins if row.get('kind')=='source_identity')
    source=view.unit(identity['unit'])
    functions=[index for index,function in enumerate(source.facts.functions)
        if function.template.occurrence==identity['occurrence']]
    if len(functions)!=1:
        raise ValueError('motion source identity lacks one original F4 function')
    window=derive_source_motion_window(view,unit=identity['unit'],function_ordinal=functions[0],
        loop_header=_motion_origin_entry(loop_origin),call_entry=_motion_origin_entry(call_origin))
    routes=[dict(leaf_formal=name,outer_parameter=proof.parameter_map[name],
        original_actual=call_origin['call_inputs'][index],
        binding_role='CARRIED' if name==proof.carry_parameter else 'OUTER_LIVE_IN',
        referent_content_invariance='CONDITIONAL_ON_EXACT_ARRAY_AND_NO_FOREIGN_WRITERS')
        for index,name in enumerate(proof.body.parameters)]
    return window,routes


_ARRAY_MOTION_SCHEMA = 'scar.array-motion'

_OBLIGATIONS = {
    'source': (
        ('ORIGINAL_F4_LOOP_AND_ARGUMENT_MAPPING',
         'Replay the original loop/call/carry/formal mapping and current callable environment.'),
        ('ORIGINAL_LOOP_HAS_NO_OTHER_REFERENT_WRITERS',
         'The complete original loop must preserve external input origins and contain no intervening input-content writes.'),
        ('NO_INTERMEDIATE_ESCAPE_OR_IDENTITY_OBSERVERS',
         'No intermediate identity, allocation lifetime, release callback or per-iteration result observer may distinguish motion.'),
    ),
    'runtime': (
        ('EXACT_ARRAY_AND_CALLABLE_RUNTIME_GUARDS',
         'Admit exact ndarray dtype/rank/layout/extents and the original callable; shape declarations are not live guards.'),
        ('POSITIVE_TRIP_ONLY',
         'Return the original carry alias for nonpositive counts before hoisted work, input scans or extra allocation.'),
        ('FP_AND_FAILURE_FALLBACK',
         'Retain original operand order and guarded FP/finite/failure handling; withhold private results before original-loop fallback.'),
        ('FRESH_PRIVATE_RESULTS_AND_BUDGET',
         'Results and carry destinations must be fresh private payloads within the separately verified allocation budget.'),
    ),
    'external': (
        ('NO_FOREIGN_CONCURRENT_INPUT_WRITERS',
         'Array contents remain stable throughout the loop; origin identity or exact ndarray type cannot prove this.'),
        ('NO_INTERSTEP_CALLBACKS_OR_HOOKS',
         'No callbacks, hooks or concurrent observers may modify or observe intervening contents or evaluation order.'),
        ('FP_STATUS_FLAGS_ARE_NOT_PROGRAM_OUTPUTS',
         'Moving evaluation requires the retained floating-status observation contract.'),
    ),
}


def _condition(status, reason, obligations=()):
    return dict(status=status, reason=reason, obligations=list(obligations))


def derive_array_motion(body, *, carry_parameter, loop_interface):
    """Return replayable facts and shared proposed placements, without running Q.

    ``readonly_parameters`` declares the requested interface roles. Its names,
    boolean credentials, and embedded proof documents do not discharge source
    or runtime obligations. Constant-only arithmetic is not evaluated/folded
    here; consumers depending on it cannot move ahead of that unmoved work.
    """
    validation = validate_region(body)
    if carry_parameter not in body.parameters:
        raise ValueError('motion carry must be an original body parameter')
    if type(loop_interface) is not dict or loop_interface.get('carry_parameter') != carry_parameter:
        raise ValueError('original caller carry mapping is required for array motion')
    readonly = loop_interface.get('readonly_parameters')
    if (type(readonly) not in (list, tuple) or len(set(readonly)) != len(readonly)
            or set(readonly) != set(body.parameters)-{carry_parameter}):
        raise ValueError('all non-carry original parameters require explicit requested readonly roles')
    # Canonical JSON both binds every supplied interface field and detaches the
    # result from mutable caller metadata. No field is a proof credential.
    interface = json.loads(json.dumps(loop_interface, sort_keys=True, allow_nan=False))
    document = asdict(body)
    source = dict(document['source'])
    origins = source.pop('origins')
    kinds = value_kinds(body.nodes)
    dependencies, consumers, blockers = {}, {node.id: [] for node in body.nodes}, {}
    for node in body.nodes:
        dependencies[node.id] = ({node.attrs['name']} if node.op == 'input' else
            set().union(*(dependencies[key] for key in node.inputs)))
        blockers[node.id] = set().union(*(blockers[key] for key in node.inputs))
        if node.op == 'slice' or (node.op not in ('input', 'const') and not dependencies[node.id]):
            blockers[node.id].add(node.id)
        for key in node.inputs:
            consumers[key].append(node.id)
    invariant = {node.id for node in body.nodes if node.op not in ('input', 'const')
        and dependencies[node.id] and carry_parameter not in dependencies[node.id]}
    invariant_nodes = [node.id for node in body.nodes if node.id in invariant]
    roots = [node.id for node in body.nodes if node.id in invariant and
        (node.id == body.output or any(key not in invariant for key in consumers[node.id]))]
    fully_invariant_output = body.output in invariant
    frontier = []
    if fully_invariant_output:
        frontier.append(dict(reason='FULLY_INVARIANT_OUTPUT_REQUIRES_FRESH_RESULT_LOOP_CONTRACT',
            expr=body.output))
    placements = {}
    for node in body.nodes:
        deps = dependencies[node.id]
        varying = carry_parameter in deps
        if node.op == 'input':
            classification = 'INPUT_CARRY' if varying else 'INPUT_INVARIANT_ORIGIN'
            placement = 'ITERATION_CARRY' if varying else 'PRE_TEMPORAL_LOOP'
            status = 'NOT_INVARIANT' if varying else 'CONDITIONAL'
        elif node.op == 'const':
            classification, placement, status = 'CONSTANT_LITERAL', 'PRE_TEMPORAL_LOOP', 'CONDITIONAL'
        elif not deps:
            classification, placement, status = 'CONSTANT_ONLY_COMPUTATION', 'ITERATION_BODY', 'NOT_SELECTED'
        elif node.op == 'slice':
            classification, placement, status = 'DESCRIPTOR', 'ITERATION_BODY', 'UNRESOLVED'
        elif varying:
            classification, placement, status = 'CARRY_DEPENDENT_COMPUTATION', 'ITERATION_BODY', 'NOT_INVARIANT'
        else:
            classification = 'INVARIANT_COMPUTATION'
            placement, status = ('ITERATION_BODY', 'UNRESOLVED') if (
                blockers[node.id] or fully_invariant_output) else ('PRE_TEMPORAL_LOOP', 'CONDITIONAL')
        if node.id in roots and blockers[node.id]:
            frontier.append(dict(reason='INVARIANT_DEPENDS_ON_UNMOVED_SCALAR_OR_VIEW_WORK',
                expr=node.id, blocking_exprs=sorted(blockers[node.id])))
        origin = (_condition('NOT_APPLICABLE', 'No formal input origins are read.') if not deps else
            _condition('NOT_INVARIANT' if varying else 'CONDITIONAL',
                'Carry reference rebinding makes this expression variant; it does not prove an input-content write.'
                if varying else 'Fixed formal origins require the original complete loop mapping, not their readonly labels.',
                ('ORIGINAL_F4_LOOP_AND_ARGUMENT_MAPPING',)))
        content = (_condition('NOT_APPLICABLE', 'Literal/scalar-only expression has no array referent reads.') if not deps else
            _condition('NOT_INVARIANT' if varying else 'CONDITIONAL',
                'Carry-dependent values change across iterations even when input objects are not mutated.'
                if varying else 'The same input origin may still have changing contents; stability is a separate obligation.',
                ('ORIGINAL_LOOP_HAS_NO_OTHER_REFERENT_WRITERS', 'NO_FOREIGN_CONCURRENT_INPUT_WRITERS',
                 'NO_INTERSTEP_CALLBACKS_OR_HOOKS', 'FRESH_PRIVATE_RESULTS_AND_BUDGET')))
        placements[node.id] = dict(op=node.op, value_kind=kinds[node.id],
            classification=classification, status=status, temporal_placement=placement,
            formal_dependencies=sorted(deps), carry_dependent=varying,
            blocking_exprs=sorted(blockers[node.id]),
            array_motion_candidate=node.id in invariant,
            origin_binding_stability=origin, referent_content_stability=content,
            data=_condition('NOT_SATISFIED' if varying else 'PROVEN',
                'Transitive operand closure contains the original carry formal.' if varying else
                'Transitive operand closure excludes the carry; this is a syntactic data fact, not content stability.'),
            no_clobber=_condition('CONDITIONAL' if deps else 'NOT_APPLICABLE',
                'The closed Expr profile has no input-write operation. Source-loop writes, aliases and foreign writers remain unproved.'
                if deps else 'No array referent is read by this scalar/literal closure.',
                ('ORIGINAL_LOOP_HAS_NO_OTHER_REFERENT_WRITERS', 'NO_FOREIGN_CONCURRENT_INPUT_WRITERS',
                 'FRESH_PRIVATE_RESULTS_AND_BUDGET') if deps else ()),
            execution=_condition('UNRESOLVED' if blockers[node.id] else 'CONDITIONAL',
                'Unmoved scalar-only computation or unsupported view must not be crossed by its consumers.'
                if blockers[node.id] else 'Pre-loop placement is guarded by positive count and original loop/normal-completion admission.',
                ('POSITIVE_TRIP_ONLY', 'ORIGINAL_F4_LOOP_AND_ARGUMENT_MAPPING')),
            effects=dict(status='CONDITIONAL',
                basis='VALIDATED_CLOSED_ARRAY_EXPR_PROFILE_NOT_ORIGINAL_SOURCE_EFFECT_CLOSURE',
                may_raise='UNKNOWN_UNTIL_GUARDED_EXECUTION',
                input_aliasing='MAY_ALIAS_NOT_INFERRED_DISJOINT',
                obligations=['EXACT_ARRAY_AND_CALLABLE_RUNTIME_GUARDS', 'FP_AND_FAILURE_FALLBACK',
                    'NO_INTERMEDIATE_ESCAPE_OR_IDENTITY_OBSERVERS', 'FP_STATUS_FLAGS_ARE_NOT_PROGRAM_OUTPUTS']))
    status = 'UNRESOLVED' if frontier else 'CONDITIONAL' if roots else 'NO_MOVABLE_INVARIANT'
    result = dict(schema=_ARRAY_MOTION_SCHEMA, schema_version=1, body_id=body.id,
        body_sha256=semantic_sha256(document), source=source,
        source_ref_sha256=semantic_sha256(document['source']),
        source_origins=dict(count=len(origins), sha256=semantic_sha256(origins)),
        loop_interface_sha256=semantic_sha256(interface), carry_parameter=carry_parameter,
        requested_readonly_parameters=[name for name in body.parameters if name != carry_parameter],
        dependencies={node.id: sorted(dependencies[node.id]) for node in body.nodes},
        invariant_nodes=invariant_nodes, invariant_roots=roots, placements=placements,
        status=status, conditional_legality=dict(status=status,
            authority='CLOSED_ARRAY_MODEL_WITH_UNDISCHARGED_SOURCE_RUNTIME_AND_EXTERNAL_Q',
            readonly_role_is_not_source_proof=True, general_cfg_or_memoryssa_legality_proven=False),
        conditional_obligations={scope:[dict(id=identifier, status='UNRESOLVED', description=description)
            for identifier, description in rows] for scope, rows in _OBLIGATIONS.items()},
        alias_contract='INPUTS_MAY_ALIAS_NO_CLOBBER_REQUIRES_NO_INPUT_WRITES_AND_FRESH_PRIVATE_RESULTS',
        zero_trip='RETURN_ORIGINAL_CARRY_ALIAS_BEFORE_HOIST_INPUT_GUARDS_SCANS_OR_ALLOCATION',
        constant_policy='FINITE_LITERALS_NOT_EVALUATED_CONSTANT_ONLY_OPERATIONS_NOT_FOLDED_OR_MOVED',
        preserved_assumptions=list(body.assumptions), validation=validation, frontier=frontier,
        original_f4_replayed=False, source_bytes_reopened=False, runtime_guards_discharged=False,
        memory_or_physical_storage_proven=False, cost_or_timing_inferred=False,
        source_replacement_authorized=False, selected_for_use=False)
    result['id'] = semantic_sha256(result)
    return result


def verify_array_motion(report, body, *, carry_parameter, loop_interface):
    """Recompute for this exact body/interface; stale or altered facts reject."""
    expected = derive_array_motion(body, carry_parameter=carry_parameter, loop_interface=loop_interface)
    if json.dumps(report, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError('array motion differs from the exact body/interface replay')
    return expected


__all__ = ['derive_source_motion_window', 'verify_source_motion_window', 'project_loop_source_motion',
           'derive_array_motion', 'verify_array_motion']
