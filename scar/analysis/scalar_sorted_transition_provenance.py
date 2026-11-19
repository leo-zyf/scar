"""Join an admitted sorted carrier to one original lazy transition and return."""

import argparse
import ast
from collections import Counter
from dis import get_instructions
import json
from pathlib import Path

from scar.analysis import scalar_loop_score_provenance as loop
from scar.analysis import scalar_score_reuse_evidence as observed
from scar.analysis import scalar_sorted_source_guard as sorted_guard
from scar.analysis import scalar_table_handoff
from scar.compiler.scalar_table_region import _definition
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.record_codec import encode
from scar.optimization import scalar_score_reuse_plan as transport_plan
from scar.optimization import scalar_transition_ir as transition_ir
from scar.trace.code_snapshot import describe_code, site

SCHEMA = 'scar.scalar-sorted-transition-provenance'


def _require(condition, explanation):
    if not condition:
        raise ValueError('scalar sorted transition provenance: ' + explanation)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True),schema)


def _delegation(code, sort_call, yielded):
    instructions = list(get_instructions(code))
    ordinal = next(index for index,op in enumerate(instructions) if op.offset==sort_call['offset'])
    _require(site(code,instructions[ordinal].offset)==sort_call,'original sort boundary changed')
    operations = instructions[ordinal+1:ordinal+8]
    names = tuple(op.opname for op in operations)
    _require(names[:5]==('GET_YIELD_FROM_ITER','LOAD_CONST','SEND','YIELD_VALUE','RESUME')
        and names[5:] in (('JUMP_BACKWARD_NO_INTERRUPT','END_SEND'),('JUMP_BACKWARD','END_SEND'))
        and operations[1].argval is None and operations[2].jump_target==operations[6].offset
        and operations[5].jump_target==operations[2].offset and site(code,operations[3].offset)==yielded,
        'sorted list is not consumed directly by original delegation')
    names = ('iter','first_send','next','yield','resume','loop_back','exhausted')
    return {role:site(code,op.offset) for role,op in zip(names,operations,strict=True)}


def _graph(guard, provenance, evidence, transition, handoff, transport, region):
    arm = observed._one((row for row in provenance['paths']['scored'] if row['role']=='SORTED_DELEGATION'),
                        'one original sorted producer path required')
    candidate = provenance['candidate_input']
    tree,codes = observed._compile(guard['source'])
    generator = codes[guard['creation']['sorted_owner']['qualname']]
    _require(describe_code(generator)==guard['creation']['sorted_owner']
        and arm['carrier']==guard['creation']['original_output_pair']
        and arm['score_call_fact']['fact']==guard['source_calls']['scorer']['fact'],
        'same original sorted producer and score carrier required')
    source_owner = _definition(tree,provenance['record_scope']['owner']['qualname'])
    loop_ast = observed._one((row for row in ast.walk(source_owner) if type(row) is ast.For
        and observed._span(row)==provenance['record_scope']['loop_span']), 'original owner loop missing')
    _require(observed._span(loop_ast.iter)==candidate['original_generator_call']['position']
        and provenance['obligations']['current_iteration_consumer_uses_unpacked_move'] is True
        and provenance['obligations']['successful_score_definition_reaches_carried_slot'] is True,
        'original loop or normal def/use receipt differs')
    delegation = _delegation(generator,guard['creation']['sort_call'],arm['yield_site'])
    facts = loop._OriginalFacts(evidence)
    _,factory_call = facts.coordinate('calls',loop_ast.iter,lambda row:not row.positional and not row.keywords)
    factory_result = facts.row('calls',factory_call).result
    starters = []
    for ordinal,row in enumerate(facts.facts.iterations):
        _,span,_ = loop._metadata(facts.syntax,facts.port(row.value).port.template.occurrence)
        if [span[key] for key in ('line','end_line','column','end_column')]==provenance['record_scope']['loop_span']:
            starters.append((ordinal,row))
    construction, iterator = observed._one((item for item in starters if not item[1].advance),'original loop iterator construction boundary changed')
    drive, next_item = observed._one((item for item in starters if item[1].advance),'original loop advance boundary changed')
    slots = candidate['record_slots']
    _require(len(slots)==2 and [slot['slot'] for slot in slots]==[0,1],'original two-slot unpack ordering changed')
    items = [facts.row('items',slot['item']) for slot in slots]
    _require(items[0].receiver==items[1].receiver and all(slot['record']==encode(facts.port(item.receiver))
        for slot,item in zip(slots,items,strict=True)), 'score and move must come from one unpacked record')
    generator_to_iterator = facts.aliases_between(factory_result,iterator.source)
    iterator_to_next = facts.aliases_between(iterator.value,next_item.source)
    next_to_record = facts.aliases_between(next_item.value,items[0].receiver)
    producer, consumer = (transition['original_calls'][role] for role in ('producer','consumer'))
    _require(producer==handoff['original_sites']['producer'] and consumer==handoff['original_sites']['consumer']
        and candidate['transition_site']==producer['site'] and handoff['catalog']==guard['catalog']
        and {row['formal'] for row in handoff['observed_edges']}=={transition['consumer']['actual_formal']['name']},
        'original transition to recursive consumer handoff differs')
    producer_row,consumer_row = (facts.row('calls',dict(fact=call['fact_reference'])) for call in (producer,consumer))
    move_actual = facts.row('names',candidate['move_at_transition'])
    _require(len(producer_row.positional)==1 and move_actual.namespace.name==slots[1]['symbol']
        and candidate['move_at_transition']['binding']==slots[1]['definition']['binding'],
        'original move call actual is not the selected loop slot')
    move_to_actual = facts.aliases_between(move_actual.value,producer_row.positional[0])
    actual_index = handoff['caller_actual_ordinal']
    _require(0 <= actual_index < len(consumer_row.positional) and producer['site']['code']==consumer['site']['code']
        and producer['site']['offset'] < consumer['site']['offset'], 'original recursive call ordering differs')
    return_flow = facts.aliases_between(producer_row.result,consumer_row.positional[actual_index])
    make_iter = encode(facts.slicer.fact(facts.facts,'iterations',construction))
    next_iter = encode(facts.slicer.fact(facts.facts,'iterations',drive))
    facts.material.check()
    return dict(nodes=dict(score=dict(kind='CALL',source=guard['source_calls']['scorer']),
        pack=dict(kind='CONSTRUCT_TWO_SLOT_RECORD',source=arm['carrier']),
        sort=dict(kind='EAGER_STABLE_ORDER',source=guard['source_calls']['sort'],descending=region['descending']),
        delegate=dict(kind='DELEGATE_AND_SUSPEND',source=delegation),
        drive=dict(kind='NEXT_ORIGINAL_OWNER_GENERATOR',generator_call=factory_call,iterator=make_iter,
                   next=next_iter,site=provenance['record_scope']['next_record']),
        unpack=dict(kind='UNPACK_CURRENT_RECORD',site=provenance['record_scope']['unpack'],slots=slots),
        transition=dict(kind='ORIGINAL_NORMAL_MOVE_CALL',source=producer,
            eliminated_nested_call_if_admitted=transition['original_calls']['score'],native_input=transport['inputs'][-1]),
        recurse=dict(kind='ORIGINAL_RECURSIVE_CALL',source=consumer,formal=transition['consumer']['actual_formal'])),
        edges=[dict(source='score',target='pack',kind='DATA',positional_slot=0,
                    definition=arm['definition'],result_aliases=arm['score_result_to_definition'],slot_aliases=arm['carried_slots_from_uses'][0]),
               dict(source='pack',target='sort',kind='DATA_STREAM',protocol='ORIGINAL_GENERATOR_ARGUMENT_RETURNS_SCALAR_PAIR'),
               dict(source='sort',target='delegate',kind='DATA_SEQUENCE',protocol='COMPILED_SORT_RESULT_STACK_TO_YIELD_FROM'),
               dict(source='delegate',target='drive',kind='DATA',protocol='SAME_GENERATOR_ORIGINAL_NEXT',
                    generator_to_iterator=generator_to_iterator,iterator_to_advance=iterator_to_next),
               dict(source='drive',target='unpack',kind='DATA',aliases=next_to_record),
               dict(source='unpack',target='transition',kind='DATA',slot=1,aliases=move_to_actual,
                    conditions=['ORIGINAL_NORMAL_ITERATION_PATH','EXCLUDES_NULL_AND_CUTOFF']),
               dict(source='unpack',target='transition',kind='CONDITIONAL_REWRITE_INPUT',slot=0,
                    conditions=['CURRENT_SORT_TOKEN','CONSUMER_GUARDS_PASSED']),
               dict(source='transition',target='recurse',kind='DATA',actual_ordinal=actual_index,
                    caller_aliases=return_flow,callee_normal_handoff=handoff['f4_data_route'])])


def _samples(transport, original, evidence, raw, arm):
    _require(transport['capture']==evidence['capture'] and isinstance(raw['samples'],list),
             'score transport must use the same actual transition capture')
    removed = []
    revised = transport_plan._replace(original['body'],removed)
    slot = dict(name=transport_plan._INPUT,type='int')
    _require(len(removed)==transport['removed_score_calls']==1 and revised==transport['body']
        and transport['inputs']==[*original['inputs'],slot], 'score transport does not replace exactly one original scorer')
    scorer = read_artifact(original['score_plan'])
    fields = {row['name']:transition_ir._KINDS[row['kind']] for row in raw['state_fields']}
    seen,counts,rows = set(),Counter(),[]
    for edge in evidence['samples']:
        ordinal = edge['sample']
        _require(type(ordinal) is int and 0<=ordinal<len(raw['samples']) and ordinal not in seen,
                 'duplicate or missing original sample ordinal')
        seen.add(ordinal)
        row = raw['samples'][ordinal]
        paired = row['score_reuse']
        version = row['table_version']
        name = paired['arm']
        from_sort = name==arm['role']
        qualified = from_sort and paired['first_score_site']==arm['score_call'] and paired['yield_site']==arm['yield_site']
        _require((qualified or name=='DIRECT_YIELD') and row['consumer_is_return'] is True
            and all(paired[key] is True for key in ('same_original_loop_receiver','same_original_move_object','same_table_root_and_content'))
            and paired['first_result']==edge['first_score']==edge['second_score']==row['score_delta']
            and paired['table_version']==edge['table_version']==version and row['caller_frame']==edge['owner_frame'],
            'observed original loop/recursive result does not match the score handoff')
        table = original['table_versions'][version]['rows']
        _require(table==raw['table_versions'][version]['rows'] and scalar_table_handoff.table_version(table)==version,
                 'source sample table content has a different logical version')
        src = transition_ir._flatten(fields,row['receiver_fields'])
        inputs = {**src, **{f'arg_{i}':value for i,value in enumerate(row['argument'])}}
        expected = list(transition_ir._flatten(fields,row['returned_fields']).values())
        original_scores,reused_scores = [],[]
        _require(transition_ir._run(original['body'],inputs,table,scorer,{},original_scores)==expected
            and transition_ir._run(revised,{**inputs,slot['name']:paired['first_result']},table,scorer,{},reused_scores)==expected
            and original_scores==[paired['first_result']] and not reused_scores,
            'source score replay did not preserve the complete recursive actual')
        counts[name] += 1
        rows.append(dict(sample=ordinal,disposition='SORTED_TOKEN_SCOPE' if qualified else 'KEEP_ORIGINAL_DIRECT_YIELD',
            table_version=version,table_root_occurrence=row['root_token'],actual_thread=row['thread_id'],
            owner_frame=row['caller_frame'],transition_frame=row['producer_frame'],recursive_frame=row['consumer_frame'],
            receiver=row['receiver_fields'],move=row['argument'],producer_score=paired['first_result'],
            original_nested_score=row['score_delta'],recursive_actual=row['returned_fields'],
            observed_sequence={key:paired[key] for key in ('first_return_sequence','yield_sequence','transition_enter_sequence')},
            sorted_rank_observed=False,table_context_replay='CAPTURE_VERSION_ONLY'))
    _require(counts==transport['verified_sample_counts'] and counts[arm['role']]>0,
             'sorted bridge must preserve every previously verified original state sample')
    return rows,dict(counts)


def project(source_guard_path, transport_path):
    guard_ref = _ref(source_guard_path,sorted_guard.SCHEMA)
    transport_ref = _ref(transport_path,transport_plan.SCHEMA)
    guard = sorted_guard.check(guard_ref['uri'])
    provenance = read_artifact(guard['provenance'])
    checked_transport = read_artifact(transport_ref)
    region = read_artifact(guard['plan'])
    _require(read_artifact(guard_ref)==guard and guard['source']==provenance['source']==region['source']
        and provenance['evidence']==checked_transport['evidence']==region['evidence'],
        'sorted source, lexical provenance and score transport must share one original graph')
    original = read_artifact(checked_transport['transition_plan'])
    _require(all(checked_transport['transition_plan'][key]==region['transition'][key]
        for key in ('schema','sha256','size_bytes')) and read_artifact(region['transition'])==original,
        'sorted source and score transport must name identical transition plan bytes')
    evidence = read_artifact(checked_transport['evidence'])
    _require(original['evidence']==evidence['transition_evidence'] and evidence['source']==guard['source']
        and evidence['catalog']==guard['catalog'], 'native transition may not borrow another original caller or table')
    transition = read_artifact(original['evidence'])
    handoff = read_artifact(transition['original_handoff'])
    raw = read_artifact(evidence['capture'])
    arm = observed._one((row for row in provenance['paths']['scored'] if row['role']=='SORTED_DELEGATION'),
                        'one original sorted arm must transport scores')
    graph = _graph(guard,provenance,evidence,transition,handoff,checked_transport,region)
    samples,counts = _samples(checked_transport,original,evidence,raw,arm)
    return dict(schema=SCHEMA,schema_version=1,source_guard=guard_ref,provenance=guard['provenance'],
        transport=transport_ref,source=guard['source'],catalog=guard['catalog'],original_capture=evidence['capture'],
        graph=graph,samples=samples,sample_counts=counts,
        token=dict(scope='PRIVATE_TO_ONE_CHECKED_SORT_OWNER_ACTIVATION_AND_THREAD',
            mint='AFTER_COMMITTED_SORT_RESULTS_WITH_DISTINCT_ACTUAL_MOVE_OBJECTS',
            cookie=['EXACT_ORIGINAL_OWNER_FRAME','EXACT_OWNING_GENERATOR','SORT_OCCURRENCE','ONE_RESULT_ROW'],
            payload=['EXACT_RECEIVER_OBJECT','ACTUAL_MOVE_OBJECT','SIGNED_64_SCORE','SCORE_TABLE_CONTENT_VERSION',
                'MUTATION_EPOCH_FOR_SCORER_CONTEXT','EXACT_SCORER_AND_TRANSITION_BINDINGS'],
            use='ONLY_AT_SELECTED_ORIGINAL_TRANSITION_AFTER_CURRENT_TWO_SLOT_UNPACK',
            compare=['SAME_OWNER_AND_THREAD','SAME_MOVE_AND_RECEIVER_OBJECTS','SAME_CURRENT_UNPACKED_SCORE_VALUE',
                'SAME_LIVE_SCORER_CONTEXT_AND_RELEVANT_TRANSITION_BINDINGS'],
            consume='AT_MOST_ONCE_BEFORE_ENTERING_THE_ORIGINAL_RECURSIVE_CONSUMER',
            invalidators=['ANY_RELEVANT_TABLE_OR_SCORER_CONTEXT_MUTATION','OWNER_EXHAUSTION_BREAK_RETURN_UNWIND_OR_CLOSE',
                'RECURSIVE_CONTEXT_OR_CURRENT_ROW_MISMATCH'],
            lifetime='BOUNDED_PRIVATE_ROWS_WITH_EXACT_WEAK_OWNER_OR_EQUIVALENT_CLEANUP',
            fallback='CALL_ORIGINAL_TRANSITION_WITH_ORIGINAL_MOVE_AND_KEEP_ORIGINAL_RECURSION'),
        effects=dict(source_order_preserved=['ORIGINAL_SORT_THEN_DELEGATION','ORIGINAL_CURRENT_ITERATION_BRANCHES',
            'LAZY_TRANSITION_ONLY_IF_ORIGINAL_CALL_REACHED','ORIGINAL_RECURSIVE_CALL_AND_ARGUMENT_ORDER',
            'ORIGINAL_RECURSIVE_STATE_WRITES_AND_DEADLINE_CHECK_SITES'],
            runtime_guards=['TABLE_IDENTITY_AND_CONTENT_VERSION_AT_PRODUCTION_AND_CONSUMPTION',
                'NO_INTERVENING_RELEVANT_EPOCH_CHANGE_EVEN_IF_CONTENT_RESTORED',
                'EXACT_SCORER_TRANSITION_ROTATION_CONSTRUCTOR_AND_BUILTINS_AT_USE',
                'NO_OBSERVABLE_COMPILED_FRAME_OR_DELEGATION_INTERCEPTION'],
            direct_table_writes=provenance['frontiers']['direct_source_writes'],
            otherwise_open=['RECURSION_AND_OTHER_RECIPIENTS_WHILE_OWNER_SUSPENDED','ALIAS_AND_EXTERNAL_EFFECTS',
                'WALL_CLOCK_DEADLINE_OUTCOMES',
                'TYPE_OR_CALLABLE_MUTATION_OUTSIDE_MODULE_DICT_EPOCH','CLOSE_THROW_AND_HOOKS']),
        qualification=dict(sort_and_transition_coobserved_in_original_worker=True,complete_sampled_recursive_value_replayed=True,
            historical_sorted_row_rank_known=False,observed_transition_occurrences_have_continuous_effect_proof=False,
            whole_recursive_effects_closed=False,live_token_qualified=False,deployment_authorized=False,
            target_executed_by_projection=False))


def check(path):
    result = read_artifact(_ref(path,SCHEMA))
    read_artifact(result['source_guard'])
    read_artifact(result['transport'])
    _require(result==project(result['source_guard']['uri'],result['transport']['uri']),
             'stored sorted transition flow differs from independent source replay')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-guard', type=Path)
    parser.add_argument('--transport', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args(argv)
    inputs = (args.source_guard,args.transport,args.out)
    if args.check and not any(inputs):
        result = check(args.check)
    elif not args.check and all(inputs):
        result = project(args.source_guard,args.transport)
        with args.out.open('x',encoding='utf-8') as stream:
            json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
            stream.write('\n')
    else:
        parser.error('provide --source-guard, --transport and --out, or --check')
    print(json.dumps(dict(schema=SCHEMA,sample_counts=result['sample_counts'],
                          graph_edges=len(result['graph']['edges']))))


if __name__ == '__main__':
    main()
