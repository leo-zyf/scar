"""Bind a source loop's original ports to normal returned-value observations."""

import argparse
import json
from pathlib import Path

from scar.analysis import source_returned_loop_carry as static
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.trace import returned_state_recurrence as dynamic


SCHEMA = 'scar.returned-loop-binding'


def _loc(reference, pointer):
    return dict(artifact=reference, locator=pointer)


def _single(rows, description):
    if len(rows) != 1:
        raise ValueError(f'expected one {description}, found {len(rows)}')
    return rows[0]


def _source_candidates(document, code_id, position, keyword):
    for i, function in enumerate(document['functions']):
        if function['code_id'] != code_id:
            continue
        for j, loop in enumerate(function['loops']):
            for k, carry in enumerate(loop['carries']):
                path = f'#/functions/{i}/loops/{j}/carries/{k}'
                for l, consumer in enumerate(carry['consumers']):
                    arg = consumer['argument']
                    if (consumer['source_span'] == position and arg['kind'] == 'KEYWORD'
                            and arg['name'] == keyword):
                        yield (path, carry, f'{path}/consumers/{l}', consumer)


def _sources(carry, path, phase):
    for index, route in enumerate(carry[phase]):
        for ordinal, definition in enumerate(route['known_definitions']):
            producer = definition.get('producer')
            if producer is not None:
                yield f'{path}/{phase}/{index}/known_definitions/{ordinal}/producer', producer


def _match(document, source, witness, evidence):
    call = witness['endpoints']['call']
    getter = witness['endpoints']['result']
    formal = witness['endpoints']['input_parameter']
    observed = []
    for index, activation in enumerate(witness['activations']):
        rounds = activation['iterations']
        sites = [row['call']['site'] for row in rounds]
        call_site = sites[0]
        if (len(rounds) < 2 or any(site != call_site for site in sites)
                or call_site['source'] != document['source']
                or any(row['result']['receiver_relation'] != 'SAME_OBJECT_AS_METHOD_RESULT_OBSERVED'
                       for row in rounds)):
            raise ValueError('requires one directly observed recurrent call and its actual result receiver')
        path, carry, use_path, use = _single(
            list(_source_candidates(document, call_site['code_id'], call_site['position'], formal)),
            'original caller, loop and keyword use')
        getter_sites = [row['result'].get('site') for row in rounds]
        getter_site = getter_sites[0]
        if any(site != getter_site for site in getter_sites):
            raise ValueError('getter executes at different caller coordinates')
        if getter_site is not None and (getter_site['source'] != document['source']
                or getter_site['code_id'] != call_site['code_id']):
            raise ValueError('observed getter lies outside the original caller')
        candidates = {}
        for loc, producer in _sources(carry, path, 'backedges'):
            if (producer['kind'] == 'ATTRIBUTE_GET' and producer['attribute'] == getter['method']
                    and (getter_site is None or producer['source_span'] == getter_site['position'])):
                candidates.setdefault(json.dumps(producer['result'], sort_keys=True), (loc, producer))
        output_path, output = _single(list(candidates.values()), 'original result-producing port')
        if not all(carry[phase] for phase in ('initial', 'backedges', 'normal_exits')):
            raise ValueError('original entry, carry and normal-exit routes are required')
        receiver_definitions = output.get('receiver_definitions', {})
        receiver_producers = [definition['producer'] for definition in
                              receiver_definitions.get('known_definitions', ())]
        if 'receiver_producer' in output:
            receiver_producers.append(output['receiver_producer'])
        linked_receiver = bool(receiver_producers) and all(
            producer['kind'] == 'CALL' and producer['result'] == use['result']
            for producer in receiver_producers)
        transitions = activation['consumptions']
        if (len(transitions) != len(rounds) - 1
                or any(not row['same_call_site']
                       or row['digest_comparison'] != 'EQUAL_DIGEST_OBSERVED' for row in transitions)):
            raise ValueError('this profile requires observed preservation across each selected backedge')
        path_evidence = _loc(evidence, f'#/activations/{index}')
        observed.append(dict(source_carry=_loc(source, path), consumer=_loc(source, use_path),
            producer=_loc(source, output_path), observed_activation=path_evidence,
            observed_call_site=call_site, observed_getter_site=getter_site,
            receiver_port=output.get('receiver'), call_result_port=use['result'],
            source_receiver_call_linked=linked_receiver,
            source_receiver_open_alternatives=receiver_definitions.get('open_definition_kinds', []),
            loop_count=len(rounds), observation_cpu_ids=witness['cpu_ids'],
            state_parameter=formal, call_code_id=call['code_id'], getter_code_id=getter['code_id'],
            input_size_bytes=sorted({row['call']['input']['value']['size_bytes'] for row in rounds
                                     if row['call']['input']['value']['kind'] == 'NDARRAY'}),
            output_size_bytes=sorted({row['result']['output']['value']['size_bytes'] for row in rounds}),
            state_identity_observations=[row['object_relation']['kind'] for row in transitions],
            loop_exit_evidence=witness['application_completion']))
    return observed


def project(source_path, witness_paths):
    source_ref = artifact_reference(Path(source_path).resolve(strict=True), static.SCHEMA)
    document = static.check(source_ref['uri'])
    if read_artifact(source_ref) != document:
        raise ValueError('source projection changed during replay')
    witnesses = []
    matches = []
    for path in witness_paths:
        ref = artifact_reference(Path(path).resolve(strict=True), dynamic.SCHEMA)
        witness = dynamic.check(ref['uri'])
        if ref in witnesses or read_artifact(ref) != witness:
            raise ValueError('duplicate or changed dynamic evidence')
        witnesses.append(ref)
        matches.extend(_match(document, source_ref, witness, ref))
    if not matches:
        raise ValueError('returned-value binding requires an observed activation')
    return dict(schema=SCHEMA, schema_version=1, source=source_ref, witnesses=witnesses,
        matches=matches, iterations_unrolled=False, storage_binding_known=False,
        deployment_authorized=False)


def check(path):
    report = read_artifact(artifact_reference(Path(path).resolve(strict=True), SCHEMA))
    actual = project(report['source']['uri'], [row['uri'] for row in report['witnesses']])
    if actual != report:
        raise ValueError('retained returned-value bindings differ from source/trace replay')
    return actual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source')
    parser.add_argument('--witness', action='append')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    if args.check and not (args.source or args.witness or args.out):
        result = check(args.check)
    elif args.source and args.witness and args.out and args.check is None:
        result = project(args.source, args.witness)
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        parser.error('use --source FILE --witness FILE [--witness FILE] --out FILE or --check FILE')
    print(json.dumps(dict(matches=len(result['matches']), iterations_unrolled=False)))


if __name__ == '__main__':
    main()
