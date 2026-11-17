"""Conditional movement queries bound to an original source-window report.

This module checks the supplied records and their correspondence, not original
F4 admission. The controller must independently replay the complete window.
No result authorizes source replacement or discharges live/external guards.
"""
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import asdict
import json

from scar.analysis.array_motion import derive_array_motion
from scar.analysis.source_effects import classify_instruction
from scar.ir.execution_recipe import artifact_reference
from scar.ir.program import normal_graph as G, normal_targets as T, source_productions as P
from scar.ir.record_codec import decode, encode
from scar.optimization.resource_solver import semantic_sha256


SCHEMA = 'scar.array-motion-legality'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _coordinate(port):
    value = decode(T.GlobalPort, port)
    return value.unit, _canonical(encode(value.port.template)), value.port.owner.wire


def _origin_coordinate(origin):
    return origin['unit'], _canonical(origin['template']), origin['step']


def _reference_binding_trace(actual, window, reference):
    """Walk only original forwarding edges with balanced recipe contexts."""
    incoming = defaultdict(list)
    for index, edge in enumerate(window['reference_transports']):
        if edge['kind']!='DATA':
            raise ValueError('reference transports must contain only original DATA edges')
        incoming[_canonical(edge['target'])].append((index,edge))
    reads = defaultdict(list)
    for ordinal,row in enumerate(window['lexical_bindings']):
        for index,read in enumerate(row['reads']):
            reads[_canonical(read['value'])].append((ordinal,index,row))
    budget = window.get('request', {})
    max_states, max_stack = budget.get('max_states',20000), budget.get('max_stack',128)
    if any(type(value) is not int or value<1 for value in (max_states,max_stack)):
        raise ValueError('positive exact reference-query state and stack budgets required')
    start = _canonical(actual), ()
    pending = deque((start,)); parents = {start:None}; frontier = []; origins = []; traversed = set()
    while pending:
        state = pending.popleft(); point, stack = state
        if point in reads:
            path = []; cursor = state
            while parents[cursor] is not None:
                previous, edge = parents[cursor]
                path.append(edge); cursor = previous
            for ordinal,index,row in reads[point]:
                origins.append(dict(namespace=deepcopy(row['namespace']),
                    lexical_binding_ref=reference('#/lexical_bindings/'+str(ordinal)),
                    read_ref=reference('#/lexical_bindings/'+str(ordinal)+'/reads/'+str(index)),
                    value=json.loads(point), context=[json.loads(site) for site in stack],
                    transport_refs=[reference('#/reference_transports/'+str(edge)) for edge in reversed(path)]))
            continue  # an atomic READ output must never be crossed to inputs
        matched = False
        for index,edge in incoming.get(point, ()):
            transition = edge['transition']; following = stack
            if transition=='RETURN':
                if edge['site'] is None or len(stack)>=max_stack:
                    frontier.append(dict(reason='REFERENCE_CONTEXT_STACK_UNRESOLVED',
                        source_ref=reference('#/reference_transports/'+str(index))))
                    continue
                following = (*stack,_canonical(encode(decode(G.CallSite,edge['site']))))
            elif transition=='CALL':
                if not stack or edge['site'] is None or stack[-1]!=_canonical(edge['site']):
                    continue  # never splice another caller's shared recipe
                following = stack[:-1]
            elif transition in ('BACKEDGE','RESUME'):
                frontier.append(dict(reason='REFERENCE_'+transition+'_VERSION_OR_ACTIVATION_UNRESOLVED',
                    source_ref=reference('#/reference_transports/'+str(index))))
                continue
            elif transition!='DIRECT':
                raise ValueError('unsupported original reference transport transition')
            if edge.get('condition') is not None:
                frontier.append(dict(reason='REFERENCE_CONTROL_GUARD_NOT_SELECTED',
                    source_ref=reference('#/reference_transports/'+str(index))))
            matched = True; traversed.add(index)
            upstream = _canonical(edge['source']), following
            if upstream not in parents:
                if len(parents)>=max_states:
                    frontier.append(dict(reason='REFERENCE_STATE_BUDGET_EXHAUSTED',max_states=max_states,
                        source_ref=reference('#/reference_transports/'+str(index))))
                    continue
                parents[upstream] = state,index; pending.append(upstream)
        if not matched:
            frontier.append(dict(reason='NO_BALANCED_REFERENCE_SOURCE_OR_LEXICAL_READ',
                value=json.loads(point), context=[json.loads(site) for site in stack]))
    if len(origins)!=1:
        frontier.append(dict(reason='REFERENCE_ORIGIN_NOT_UNIQUE',origins=len(origins)))
    return dict(status='UNRESOLVED' if frontier else 'CONDITIONAL_UNIQUE_LEXICAL_ORIGIN',
        origins=origins, frontier=frontier, visited_states=len(parents), max_states=max_states,max_stack=max_stack,
        traversed_transport_refs=[reference('#/reference_transports/'+str(index)) for index in sorted(traversed)],
        witness_scope='ONE_BALANCED_PATH_PER_READ_ORIGIN_PLUS_ALL_TRAVERSED_EDGES',
        primitive_operations_crossed=False, content_identity_or_path_sensitive_equality_proven=False)


def _strict_boundary_operations(window):
    """Replay the exhaustion/empty-else ABI checked directly by the frontend.

    Those regions are checked without the lowerer's ordinary ``run`` and
    consequently need not occur in its per-operation origin ledger.
    """
    header = decode(T.GlobalPort, window['loop_header'])
    instruction = decode(P.LoopControl, window['loop']['header_instruction'], record_tags=P.PRODUCTION_RECORD_TAGS)
    if (type(instruction.condition) is not P.IterationCondition
            or encode(instruction.condition,record_tags=P.PRODUCTION_RECORD_TAGS)!=window['loop']['condition']
            or encode(instruction.body)!=window['loop']['body']['region']
            or encode(instruction.otherwise)!=window['loop']['otherwise'] or instruction.invariant_count!=1):
        raise ValueError('source window changed the original loop-header instruction')

    def parameter(region, ordinal):
        return encode(T.GlobalPort(header.unit, G.Port(header.port.template,region,G.PortKind.PARAMETER,ordinal)))

    def region_operations(region):
        return [row for row in window['operations'] if row['entry']['unit']==header.unit
            and row['entry']['port']['template']==encode(header.port.template) and row['region']==encode(region)]

    admitted = set()
    exhausted = region_operations(instruction.condition.exhausted)
    if len(exhausted)==2:
        first, last = exhausted
        protocol = decode(P.ProductionInstruction,first['instruction'],record_tags=P.PRODUCTION_RECORD_TAGS)
        completion = decode(P.ProductionInstruction,last['instruction'],record_tags=P.PRODUCTION_RECORD_TAGS)
        results = [row['port'] for row in first['outputs'] if row['port']['port']['completion']=='FALLTHROUGH'
            and row['port']['port']['ordinal']==0]
        if (type(protocol) is P.ApplyPythonProtocol and protocol.protocol.value=='MATCH_STOP_ITERATION'
                and [row['value'] for row in first['inputs']]==[parameter(instruction.condition.exhausted,i) for i in range(2)]
                and type(completion) is P.TransferCompletion and completion.kind.value=='FALLTHROUGH'
                and completion.target is None and len(results)==1
                and [row['value'] for row in last['inputs']]==results):
            admitted.update((_coordinate(first['entry']),_coordinate(last['entry'])))
    if instruction.otherwise is not None:
        otherwise = region_operations(instruction.otherwise)
        if len(otherwise)==1:
            row = otherwise[0]
            completion = decode(P.ProductionInstruction,row['instruction'],record_tags=P.PRODUCTION_RECORD_TAGS)
            if (type(completion) is P.TransferCompletion and completion.kind.value=='FALLTHROUGH'
                    and completion.target is None and [value['value'] for value in row['inputs']]==
                    [parameter(instruction.otherwise,1)]):
                admitted.add(_coordinate(row['entry']))
    return admitted


def _bind(body, carry, interface, window, motion):
    if (type(window) is not dict or window.get('schema')!='scar.source-motion-window'
            or type(window.get('schema_version')) is not int or window['schema_version']!=1):
        raise ValueError('complete original source-motion-window query required')
    payload = {key:value for key,value in window.items() if key!='id'}
    if window.get('id') != semantic_sha256(payload):
        raise ValueError('source motion window content identity differs')
    proof_keys = [key for key in ('original_f4_proof','original_composite_f4_proof') if key in interface]
    if len(proof_keys)!=1:
        raise ValueError('one distinct original direct or composite source proof required')
    proof_kind = 'DIRECT' if proof_keys[0]=='original_f4_proof' else 'COMPOSITE'
    proof = interface[proof_keys[0]]
    if (type(proof) is not dict or semantic_sha256(proof.get('body'))!=motion['body_sha256']
            or proof.get('carry_parameter')!=carry
            or type(proof.get('parameter_map')) is not dict
            or set(proof['parameter_map'])!=set(body.parameters)
            or any(name not in proof.get('parameters', ()) for name in proof['parameter_map'].values())):
        raise ValueError('original loop proof must bind this exact body, carry and parameter map')
    if proof_kind=='COMPOSITE' and (
            proof.get('profile')!='ORIGINAL_F4_COMPOSITE_ARRAY_LOOP_V1'
            or proof.get('body_identity')!='DERIVED_FUSED_DAG_NOT_A_NEW_PYTHON_CALLABLE'
            or proof.get('transformation_authorized') is not False):
        raise ValueError('separate original composite body/composition contract required')
    origins = proof.get('origins', [])
    identities = [row for row in proof['source'].get('origins', []) if row.get('kind')=='source_identity']
    if (len(identities)!=1 or identities[0].get('unit')!=window['unit']
            or identities[0].get('occurrence')!=window['function']['template']['occurrence']
            or window['function']['body']['unit']!=window['unit']
            or window['function']['body']['template']!=window['function']['template']):
        raise ValueError('source window function differs from the original source CodeID/template identity')

    def role(name):
        matches = [(index,row) for index,row in enumerate(origins)
            if row.get('kind')=='original_f4' and row.get('role')==name]
        if len(matches)!=1:
            raise ValueError('one original loop witness required: '+name)
        return matches[0]

    call_index, call = role('invariant_array_leaf_call')
    loop_index, loop = role('single_shared_invariant_loop')
    carry_index, rebinding = role('lexical_carry_write')
    selected = window['selected_call']; header = window['loop_header']
    parameters = proof['parameters']
    actual_indices = [parameters.index(proof['parameter_map'][name]) for name in body.parameters]
    if (header!=window['loop']['header'] or header['port']['kind']!='ENTER'
            or _coordinate(header)!=_origin_coordinate(loop)
            or selected['entry']['port']['kind']!='ENTER'
            or _coordinate(selected['entry'])!=_origin_coordinate(call)
            or selected['callee']!=call['call_callee'] or selected['result']!=call['call_result']
            or selected['positional']!=call['call_inputs'] or selected['keywords']
            or selected['positional_pack'] is not None or selected['keyword_pack'] is not None
            or len(selected['positional'])!=len(body.parameters)
            or call['outer_parameter_indices']!=actual_indices or call['leaf_region_id']!=body.id
            or window['unit']!=loop['unit'] or window['unit']!=call['unit']
            or window['source_version']!=dict(path=proof['source']['path'],sha256=proof['source']['sha256'])
            or window['loop']['body']['unit']!=loop['unit'] or window['loop']['body']['template']!=loop['template']
            or window['loop']['body']['region']['wire']!=loop['body_region']
            or rebinding['binding']!=loop['lexical_carry_binding']
            or rebinding['leaf_formal_index']!=body.parameters.index(carry)
            or rebinding['outer_parameter_index']!=parameters.index(proof['parameter_map'][carry])):
        raise ValueError('source window differs from original loop/CALL/carry routes')
    entry = decode(T.GlobalPort, window['loop']['body_entry'])
    if (entry.unit!=loop['unit'] or encode(entry.port.template)!=loop['template']
            or entry.port.owner.wire!=loop['body_region'] or entry.port.kind.value!='ENTER'):
        raise ValueError('source window changed the original loop-body entry')
    condition = decode(P.IterationCondition, window['loop']['condition'], record_tags=P.PRODUCTION_RECORD_TAGS)
    if encode(condition)!=loop['iteration_condition']:
        raise ValueError('source window changed the original iteration condition')
    worlds = [value for value in call['results'] if value['port']['completion']=='FALLTHROUGH'
        and value['port']['ordinal']==1]
    if (not call.get('arguments') or selected['world_before']!=call['arguments'][-1]
            or len(worlds)!=1 or selected['world_after']!=worlds[0]):
        raise ValueError('source window changed the original selected CALL world ABI')
    if not window['loop']['control_backedges'] or any(edge['target']!=header['port']
            or edge['transition']!='BACKEDGE' or edge['kind']!='CONTROL'
            for edge in window['loop']['control_backedges']):
        raise ValueError('original loop backedges are required in the motion window')
    return proof, dict(call=call_index, loop=loop_index, carry=carry_index), proof_kind


def _composite_callee_contract(proof, window, effects, binding):
    """Join conditional callee facts; source authority stays with cold replay.

    The typed-material effect verifier belongs to the lowering/controller.
    This query checks its supplied identity and original CALL coordinates,
    never treating a status flag as permission to replace original source.
    """
    if (type(effects) is not dict or effects.get('schema')!='scar.composite-call-effects'
            or type(effects.get('schema_version')) is not int or effects['schema_version']!=1
            or type(binding) is not dict or binding.get('schema')!='scar.composite-call-effect-binding'
            or type(binding.get('schema_version')) is not int or binding['schema_version']!=1):
        raise ValueError('distinct composite effect summary and original source binding required')
    proof_sha=semantic_sha256(proof);body_sha=semantic_sha256(proof['body'])
    if (effects.get('proof_sha256')!=proof_sha or binding.get('proof_sha256')!=proof_sha
            or effects.get('body_sha256')!=body_sha or binding.get('body_sha256')!=body_sha
            or binding.get('callee_effects_sha256')!=semantic_sha256(effects)
            or _canonical(effects.get('root_source'))!=_canonical(proof['root_callee_source'])
            or _canonical(binding.get('root_source'))!=_canonical(proof['root_callee_source'])
            or _canonical(binding.get('source'))!=_canonical(proof['source'])):
        raise ValueError('composite effects, source binding and exact original proof differ')
    wanted='composite-call-effect-binding:'+semantic_sha256({key:value for key,value in binding.items() if key!='id'})
    if binding.get('id')!=wanted:
        raise ValueError('composite source effect binding content identity differs')
    outer=binding['outer_call'];selected=window['selected_call']
    fields=(('entry','entry'),('callee_port','callee'),('positional','positional'),
        ('keywords','keywords'),('result','result'),('world_before','world_before'),('world_after','world_after'))
    if any(_canonical(outer.get(left))!=_canonical(selected.get(right)) for left,right in fields):
        raise ValueError('composite outer CALL does not match the original movement window')
    original=[row for row in proof['origins'] if row.get('kind')=='original_f4'
        and row.get('role')=='invariant_array_leaf_call']
    if len(original)!=1 or _origin_coordinate(outer['source_origin'])!=_origin_coordinate(original[0]):
        raise ValueError('composite outer CALL source occurrence differs from its original proof')
    blockers=[]
    def missing(reason, document, locator):
        blockers.append(dict(reason=reason, source_refs=[dict(
            document_sha256=semantic_sha256(document),locator=locator)]))
    if effects['status']!='CONDITIONAL':
        missing('COMPOSITE_CALLEE_EFFECT_SUMMARY_UNRESOLVED',effects,'#/status')
    if binding['status']!='CONDITIONAL' or binding['frontier']:
        missing('COMPOSITE_CALLEE_SOURCE_COVERAGE_UNRESOLVED',binding,'#/frontier')
    writes=effects['input_writes'];fresh=effects['fresh_result']
    if writes['status']!='CONDITIONAL_NONE' or writes['blockers']:
        missing('COMPOSITE_CALLEE_INPUT_WRITES_UNRESOLVED',effects,'#/input_writes')
    if fresh['status']!='CONDITIONAL_FRESH' or fresh['blockers']:
        missing('COMPOSITE_CALLEE_FRESH_RESULT_UNRESOLVED',effects,'#/fresh_result')
    return dict(status='UNRESOLVED' if blockers else 'CONDITIONAL',
        proof_sha256=proof_sha,body_sha256=body_sha,callee_effects_sha256=semantic_sha256(effects),
        callee_binding_sha256=semantic_sha256(binding),source_window_sha256=semantic_sha256(window),
        input_writes=deepcopy(writes),fresh_result=deepcopy(fresh),blockers=blockers,
        caller_exposed_effects=deepcopy(effects['caller_exposed_effects']),
        shared_summaries_do_not_omit_call_occurrences=True,
        full_python_effects_closed=False,original_source_replay_required=True,
        source_replacement_authorized=False)


def derive_array_motion_legality(body, *, carry_parameter, loop_interface, source_window,
                                callee_effects=None, callee_binding=None):
    """Check source correspondence and expose precise remaining obligations."""
    motion = derive_array_motion(body, carry_parameter=carry_parameter, loop_interface=loop_interface)
    window = json.loads(_canonical(source_window))
    proof, witness_indices, proof_kind = _bind(body, carry_parameter, loop_interface, window, motion)
    if proof_kind=='DIRECT' and (callee_effects is not None or callee_binding is not None):
        raise ValueError('direct source proof cannot borrow composite effect/source bindings')
    callee_contract=(_composite_callee_contract(proof,window,callee_effects,callee_binding)
        if proof_kind=='COMPOSITE' else None)
    window_sha = semantic_sha256(window)
    proof_sha = semantic_sha256(proof)

    def window_ref(pointer):
        return dict(source_window_sha256=window_sha, locator=pointer)

    def proof_ref(index):
        return dict(proof_sha256=proof_sha, locator='#/origins/'+str(index))

    def blocker(reason, refs=(), **fields):
        return dict(reason=reason, source_refs=list(refs), **fields)

    coverage = window['coverage']
    coverage_blockers = []
    if coverage.get('window_complete') is not True or coverage.get('body_complete') is not True:
        coverage_blockers.append(blocker('SOURCE_WINDOW_NORMAL_COVERAGE_INCOMPLETE', [window_ref('#/coverage')]))
    if coverage.get('control_scope')!='ORIGINAL_NORMAL_CONTROL_BALANCED_RECIPE_CALL_RETURN':
        coverage_blockers.append(blocker('SOURCE_WINDOW_CONTROL_SCOPE_NOT_SUPPORTED', [window_ref('#/coverage')]))
    coverage_blockers.extend(blocker('SOURCE_WINDOW_FRONTIER', [window_ref('#/frontier/'+str(index))], frontier=value)
        for index,value in enumerate(window['frontier']))
    sliced = window['reference_transport_slice']
    expected_seeds = [window['selected_call']['callee'],*window['selected_call']['positional']]
    if (window['reference_transport_scope']!=
            'SELECTED_CALL_VALUES_BACKWARD_ORIGINAL_DATA_SLICE_NOT_PRIMITIVE_OR_CONTENT_EQUIVALENCE'
            or {_canonical(value) for value in sliced['seeds']}!={_canonical(value) for value in expected_seeds}
            or sliced['edges']!=len(window['reference_transports'])
            or sliced['balanced_path_selection_required'] is not True):
        raise ValueError('reference transport slice differs from original selected CALL inputs')
    reference_blockers = []
    if sliced['complete'] is not True or sliced['frontier']:
        reference_blockers.append(blocker('REFERENCE_TRANSPORT_SLICE_INCOMPLETE',
            [window_ref('#/reference_transport_slice')], frontier=deepcopy(sliced['frontier'])))
    lexical = window['lexical_bindings']
    stable, input_bindings = [], []
    for index, parameter in enumerate(body.parameters):
        actual = window['selected_call']['positional'][index]
        trace = (None if parameter==carry_parameter else _reference_binding_trace(actual,window,window_ref))
        matched_refs = set() if trace is None else {_canonical(row['lexical_binding_ref']) for row in trace['origins']}
        matches = [(ordinal,row) for ordinal,row in enumerate(lexical)
            if _canonical(window_ref('#/lexical_bindings/'+str(ordinal))) in matched_refs]
        status = ('CARRIED_BINDING' if parameter==carry_parameter else
            'PROVEN_IN_SUPPLIED_QUERY' if len(matches)==1 and matches[0][1]['namespace']['kind']=='LEXICAL'
                and trace['status']=='CONDITIONAL_UNIQUE_LEXICAL_ORIGIN'
                and not matches[0][1]['writes'] and not coverage_blockers and not reference_blockers else 'UNRESOLVED')
        if status=='PROVEN_IN_SUPPLIED_QUERY':
            stable.append(parameter)
        input_bindings.append(dict(parameter=parameter, original_actual=actual,
            outer_parameter=proof['parameter_map'][parameter], status=status,
            source_refs=[window_ref('#/lexical_bindings/'+str(ordinal)) for ordinal,_ in matches],
            namespaces=[deepcopy(row['namespace']) for _,row in matches],
            reference_trace=trace,
            binding_identity_is_content_stability=False))

    operations, clobber_blockers, execution_blockers = [], list(coverage_blockers), list(coverage_blockers)
    if callee_contract is not None:
        clobber_blockers.extend(deepcopy(callee_contract['blockers']))
    original_ops = {_origin_coordinate(row):row for row in proof['origins']
        if row.get('kind')=='original_f4' and all(key in row for key in ('unit','template','step'))}
    selected_entry = window['selected_call']['entry']
    boundary_operations = _strict_boundary_operations(window)
    entries = set(); selected_count = 0
    for index, operation in enumerate(window['operations']):
        reference = window_ref('#/operations/'+str(index))
        key = _coordinate(operation['entry'])
        if operation['index']!=index or key in entries or operation['entry']['port']['kind']!='ENTER':
            raise ValueError('source window requires unique indexed original ENTER operations')
        entries.add(key)
        instruction = decode(P.ProductionInstruction, operation['instruction'], record_tags=P.PRODUCTION_RECORD_TAGS)
        classification = encode(classify_instruction(instruction))
        reason = None; role = 'STRUCTURAL_SOURCE_OPERATION'
        if key not in original_ops and key not in boundary_operations:
            reason = 'OPERATION_NOT_IN_RETAINED_STRICT_LOOP_PROOF'
        if type(instruction) is P.ApplyLanguageAction:
            if instruction.action is P.LanguageActionKind.CALL and operation['entry']==selected_entry:
                selected_count += 1
                role = ('SELECTED_COMPOSED_ARRAY_CALL_UNDER_ORIGINAL_CALLEE_EFFECT_AND_RUNTIME_Q'
                    if callee_contract is not None else 'SELECTED_CLOSED_ARRAY_CALL_UNDER_RUNTIME_IDENTITY_GUARDS')
                ports = [value['value'] for value in operation['inputs']]
                selected = window['selected_call']
                if ports != [selected['callee'], *selected['positional'], selected['world_before']]:
                    raise ValueError('selected source operation changed its original argument/world ports')
            else:
                reason = 'OPAQUE_CALL_OR_LANGUAGE_PROTOCOL_IN_MOVEMENT_WINDOW'
        elif type(instruction) is P.AccessBinding:
            values = ([value['value'] for value in operation['inputs']] if instruction.action is P.BindingActionKind.WRITE else
                [value['port'] for value in operation['outputs']])
            table = 'writes' if instruction.action is P.BindingActionKind.WRITE else 'reads'
            matches = [row for row in lexical if any(access['value'] in values for access in row[table])]
            if instruction.action is P.BindingActionKind.DELETE:
                reason = 'BINDING_DELETION_NOT_ADMITTED_BY_STRICT_LOOP_PROFILE'
            elif instruction.action is P.BindingActionKind.WRITE:
                if len(matches)!=1 or matches[0]['namespace']['kind']!='LEXICAL':
                    reason = 'UNRESOLVED_OR_NONLEXICAL_NAMESPACE_WRITE'
                else:
                    role = ('CARRY_LEXICAL_REBIND_NOT_INPUT_MUTATION' if matches[0]['namespace']==
                        proof['origins'][witness_indices['carry']]['binding'] else
                        'LOCAL_LEXICAL_REBIND_NOT_INPUT_MUTATION')
            else:
                role = 'SOURCE_BINDING_READ_RUNTIME_NAMESPACE_IDENTITY_REMAINS'
            if key in original_ops and original_ops[key].get('binding_port')!=instruction.port:
                reason = 'BINDING_OPERATION_DIFFERS_FROM_ORIGINAL_PROOF'
        elif type(instruction) is P.ApplyPythonProtocol:
            allowed = instruction.protocol.value=='FRAME_RELEASE_TEMPORARY' or (
                instruction.protocol.value in ('COMPREHENSION_NEXT','MATCH_STOP_ITERATION')
                and 'CONDITION_OR_EXIT' in operation['membership'])
            if not allowed:
                reason = 'OPAQUE_PYTHON_PROTOCOL_IN_MOVEMENT_WINDOW'
            role = 'ORIGINAL_RANGE_OR_RELEASE_PROTOCOL_UNDER_RETAINED_RUNTIME_Q'
        elif type(instruction) is P.TransferCompletion:
            if instruction.kind.value!='FALLTHROUGH':
                execution_blockers.append(blocker('NON_FALLTHROUGH_SOURCE_COMPLETION', [reference]))
        elif type(instruction) is P.LoopControl:
            if operation['entry']!=window['loop_header']:
                execution_blockers.append(blocker('ADDITIONAL_LOOP_CONTROL_IN_WINDOW', [reference]))
        elif type(instruction) not in (P.EvaluateChild, P.ApplyRegion, P.EnterDeferredBody):
            reason = 'UNMODELED_SOURCE_INSTRUCTION_IN_MOVEMENT_WINDOW'
        if reason is not None:
            clobber_blockers.append(blocker(reason, [reference], operation=deepcopy(operation['entry']),
                instruction=type(instruction).__name__))
        operations.append(dict(source_ref=reference, entry=deepcopy(operation['entry']),
            instruction=type(instruction).__name__, classification=classification, role=role,
            may_clobber_inputs=reason is not None))
    if selected_count!=1:
        raise ValueError('motion window must enumerate the selected ordinary source CALL exactly once')
    for index, call in enumerate(window['call_sites']):
        if call!=window['selected_call']:
            clobber_blockers.append(blocker('ADDITIONAL_OPAQUE_SOURCE_CALL', [window_ref('#/call_sites/'+str(index))],
                operation=deepcopy(call['entry'])))
    for index, access in enumerate(window['accesses']):
        if access['table'] in ('attributes','items','updates'):
            clobber_blockers.append(blocker('UNMODELED_REFERENT_ACCESS_OR_PROTOCOL',
                [window_ref('#/accesses/'+str(index))], fact=deepcopy(access['fact'])))

    must = window['must_execute']
    if (must.get('status')!='CONDITIONAL' or must.get('fact')!='ONE_PER_ENTERED_BODY'
            or must.get('missing') or must.get('normal_paths_avoid_selected') is not False
            or must.get('selected_repeat_reachable') is not False or must.get('avoiding_cycle_reachable') is not False):
        execution_blockers.append(blocker('SELECTED_CALL_NOT_MUST_EXECUTE_ON_ENTERED_NORMAL_BODY',
            [window_ref('#/must_execute')]))
    obligations = deepcopy(motion['conditional_obligations'])
    runtime_ids = [row['id'] for row in obligations['runtime']]
    external_ids = [row['id'] for row in obligations['external']]
    roots = []
    nodes = {node.id:node for node in body.nodes}
    for root in motion['invariant_roots']:
        closure = set(); pending = [root]
        while pending:
            key = pending.pop()
            if key in closure:
                continue
            closure.add(key); pending.extend(nodes[key].inputs)
        reads = [dict(parameter=node.attrs['name'], expression=node.id,
            source_ref=dict(body_sha256=motion['body_sha256'], locator='#/nodes/'+str(index)),
            kind='CALL_BOUNDARY_ARRAY_REFERENT_READ', physical_alias='MAY_ALIAS_OTHER_INPUTS',
            content_identity=None) for index,node in enumerate(body.nodes) if node.id in closure and node.op=='input']
        parameters = sorted({row['parameter'] for row in reads})
        data_blockers = [blocker('INPUT_BINDING_NOT_PROVEN_STABLE_IN_SOURCE_WINDOW',
            binding['source_refs'], parameter=binding['parameter'],
            reference_frontier=[] if binding['reference_trace'] is None else deepcopy(binding['reference_trace']['frontier']))
            for binding in input_bindings if binding['parameter'] in parameters and binding['parameter'] not in stable]
        placement = motion['placements'][root]
        placement_blockers = [blocker('ARRAY_MOTION_PLACEMENT_UNRESOLVED', expression=root,
            blocking_exprs=deepcopy(placement['blocking_exprs']))] if placement['status']!='CONDITIONAL' else []

        def condition(blockers, source_refs, pending_obligations):
            return dict(status='UNRESOLVED' if blockers else 'CONDITIONAL',
                source_status='UNRESOLVED' if blockers else 'PROVEN_IN_SUPPLIED_WINDOW_UNDER_ARRAY_PROFILE',
                source_refs=source_refs, obligations=pending_obligations, blockers=deepcopy(blockers))

        source_refs = [window_ref('#/selected_call'), proof_ref(witness_indices['call'])]
        properties = dict(
            data_available=condition([*reference_blockers,*data_blockers,*placement_blockers], source_refs,
                ['EXACT_ARRAY_AND_CALLABLE_RUNTIME_GUARDS']),
            no_clobber=condition(clobber_blockers, [window_ref('#/operations'),window_ref('#/loop/control_backedges')],
                ['EXACT_ARRAY_AND_CALLABLE_RUNTIME_GUARDS','FRESH_PRIVATE_RESULTS_AND_BUDGET',*external_ids]),
            execution=condition([*execution_blockers,*placement_blockers], [window_ref('#/must_execute')],
                ['POSITIVE_TRIP_ONLY','FP_AND_FAILURE_FALLBACK']),
            effects=condition(clobber_blockers, source_refs,
                ['NO_INTERMEDIATE_ESCAPE_OR_IDENTITY_OBSERVERS',*runtime_ids,*external_ids]))
        roots.append(dict(expression=root, read_parameters=parameters, input_reads=reads,
            materialized_exprs=[node.id for node in body.nodes if node.id in closure and node.op not in ('input','const','slice')],
            materialized_value_provenance_is_repeat_input_load=False,
            alias_requirements=dict(status='CONDITIONAL', read_read_aliasing_allowed=True,
                pairwise_read_input_disjointness_required=False,
                obligations=['FRESH_PRIVATE_RESULTS_AND_BUDGET','NO_FOREIGN_CONCURRENT_INPUT_WRITERS'],
                physical_disjointness_proven=False),
            status='CONDITIONAL' if all(value['status']=='CONDITIONAL' for value in properties.values()) else 'UNRESOLVED',
            **properties))
    result = dict(schema=SCHEMA, schema_version=1, body_sha256=motion['body_sha256'],
        producer=artifact_reference(__file__, 'scar.python-producer'),
        loop_interface_sha256=motion['loop_interface_sha256'], source_window_sha256=window_sha,
        source_window_id=window['id'], source_proof_sha256=proof_sha,
        source_proof_kind=proof_kind,composed_callee_contract=callee_contract,
        array_motion_sha256=semantic_sha256(motion), carry_parameter=carry_parameter,
        stable_formal_inputs=stable, input_bindings=input_bindings, roots=roots, data_scope_blockers=reference_blockers,
        source_window_effects=dict(status='UNRESOLVED' if clobber_blockers else 'CONDITIONAL_NO_INPUT_CLOBBERS',
            blockers=clobber_blockers, operations=operations,
            read_read_input_aliasing='ALLOWED_WHEN_NO_INPUT_REFERENT_IS_WRITTEN',
            carry_rebinding_is_input_mutation=False, source_order_and_backedges_included=True),
        conditional_obligations=obligations,
        source_obligation_dispositions=[
            dict(id='ORIGINAL_F4_LOOP_AND_ARGUMENT_MAPPING', status='MATCHED_SUPPLIED_ORIGINAL_QUERY_RECORDS',
                independent_source_replay_required=True, source_refs=[proof_ref(index) for index in witness_indices.values()]),
            dict(id='ORIGINAL_LOOP_HAS_NO_OTHER_REFERENT_WRITERS',
                status='UNRESOLVED' if clobber_blockers else 'PROVEN_IN_SUPPLIED_WINDOW_UNDER_CLOSED_ARRAY_PROFILE',
                independent_source_replay_required=True, runtime_identity_and_fresh_output_guards_required=True),
            dict(id='NO_INTERMEDIATE_ESCAPE_OR_IDENTITY_OBSERVERS', status='UNRESOLVED',
                reason='Local source correspondence is not allocation/release/observer closure.')],
        status='CONDITIONAL' if roots and all(row['status']=='CONDITIONAL' for row in roots) else 'UNRESOLVED',
        source_window_replay_required=True, source_f4_replayed=False, runtime_guards_discharged=False,
        source_replacement_authorized=False, whole_python_effects_closed=False,
        physical_alias_disjointness_proven=False, cost_or_timing_inferred=False, target_executed=False)
    result['id'] = 'array-motion-legality:'+semantic_sha256(result)
    return result


def verify_array_motion_legality(report, body, *, carry_parameter, loop_interface, source_window,
                                callee_effects=None, callee_binding=None):
    expected = derive_array_motion_legality(body, carry_parameter=carry_parameter,
        loop_interface=loop_interface, source_window=source_window,
        callee_effects=callee_effects,callee_binding=callee_binding)
    if _canonical(report)!=_canonical(expected):
        raise ValueError('array motion legality differs from exact body/interface/source-window replay')
    return deepcopy(expected)


__all__ = ['derive_array_motion_legality', 'verify_array_motion_legality']
