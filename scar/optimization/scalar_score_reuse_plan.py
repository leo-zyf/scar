"""Prepare a guarded score transport rewrite from original producer and consumer evidence."""

import argparse
from collections import Counter
import json
from pathlib import Path

from scar.analysis import scalar_score_reuse_evidence as observed
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import scalar_transition_ir as transition


SCHEMA = 'scar.scalar-score-transport-plan'
_INPUT = 'producer_score'


def _replace(value, removed):
    if type(value) is list:
        return [_replace(row, removed) for row in value]
    if type(value) is not dict:
        return value
    if value.get('op') == 'score_call':
        removed.append(value)
        return dict(op='input', type='int', name=_INPUT)
    return {key:_replace(child, removed) for key,child in value.items()}


def propose(evidence_path, transition_path):
    evidence_ref = transition._ref(evidence_path, observed.SCHEMA)
    native_ref = transition._ref(transition_path, transition.SCHEMA)
    evidence = observed.check(evidence_ref['uri'])
    original = transition.check(native_ref['uri'])
    transition._require(evidence == read_artifact(evidence_ref) and original == read_artifact(native_ref)
        and evidence['transition_evidence'] == original['evidence'],
        'score transport must bind the exact original transition graph')
    capture_ref = evidence['capture']
    capture = read_artifact(capture_ref)
    fields = {row['name']:transition._KINDS[row['kind']] for row in capture['state_fields']}
    score = read_artifact(original['score_plan'])
    removed = []
    body = _replace(original['body'], removed)
    transition._require(len(removed)==1 and _INPUT not in {row['name'] for row in original['inputs']},
        'guarded transport needs one original signed-64 score call')
    counts = Counter()
    for evidence_sample in evidence['samples']:
        row = capture['samples'][evidence_sample['sample']]
        witness = row['score_reuse']
        version = row['table_version']
        cells = original['table_versions'][version]['rows']
        transition._require(cells==capture['table_versions'][version]['rows']
            and witness['table_version']==version and witness['same_original_loop_receiver']
            and witness['same_original_move_object'] and witness['same_table_root_and_content']
            and witness['first_result']==row['score_delta'],
            'source score witness differs from its original transition')
        inputs = transition._flatten(fields, row['receiver_fields'])
        inputs.update({f'arg_{i}':value for i,value in enumerate(row['argument'])})
        expected = transition._flatten(fields,row['returned_fields'])
        original_deltas, simplified_deltas = [], []
        source = transition._run(original['body'],inputs,cells,score,{},original_deltas)
        revised = transition._run(body,{**inputs,_INPUT:witness['first_result']},cells,score,{},simplified_deltas)
        transition._require(source==revised==list(expected.values())
            and original_deltas==[witness['first_result']] and not simplified_deltas,
            'effect-free score transport did not preserve the complete sampled return')
        counts[witness['arm']] += 1
    transition._require(counts, 'score transport needs at least one original witnessed producer-to-consumer route')
    return dict(schema=SCHEMA,schema_version=1,evidence=evidence_ref,transition_plan=native_ref,capture=capture_ref,
        inputs=[*original['inputs'],dict(name=_INPUT,type='int')],body=body,removed_score_calls=1,
        verified_sample_counts=dict(counts),disposition='WAIT_FOR_LIVE_VALUE_PROVENANCE',
        guards=dict(producer='TOKEN_FROM_CURRENT_SUCCESSFUL_SOURCE_SCORE_RETURN',
            exact_receiver_and_argument='TOKEN_IDENTITIES_MATCH_CURRENT_CONSUMER',
            scalar_context='EXACT_SCORE_BODY_CONSTANTS_AND_BUILTINS',
            table='TOKEN_VERSION_EQUALS_CURRENT_EXACT_TABLE_VERSION',
            invalidation='DISCARD_ON_MODEL_OR_SCORE_CONTEXT_CHANGE',
            fallback='EVALUATE_ORIGINAL_SCORE_ON_ANY_MISS',
            live_producer_token_implemented=False),
        unmodeled=dict(whole_search_effects_closed=False,continuous_no_writer_proven=False),
        native_score_bypass_qualified=False, deployment_authorized=False)


def check(path):
    result = read_artifact(transition._ref(path,SCHEMA))
    transition._require(result==propose(result['evidence']['uri'],result['transition_plan']['uri']),
        'score transport differs from cold source replay')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path)
    parser.add_argument('--transition-plan',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--check',type=Path)
    args = parser.parse_args()
    inputs = (args.evidence,args.transition_plan,args.out)
    if args.check and not any(inputs):
        result = check(args.check)
    elif not args.check and all(inputs):
        result = propose(args.evidence,args.transition_plan)
        with args.out.open('x',encoding='utf-8') as stream:
            json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
            stream.write('\n')
    else:
        parser.error('provide --evidence, --transition-plan and --out, or --check RESULT')
    print(json.dumps(dict(schema=SCHEMA,disposition=result['disposition'],
                          verified_sample_counts=result['verified_sample_counts'])))
