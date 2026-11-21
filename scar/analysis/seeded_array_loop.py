"""Recover a repeated ndarray entry region and its original trip-count parameter."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path

from scar.analysis import seeded_array_tape_binding as seeded, source_array_entry_contract as entry
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.seeded-array-runtime-loop'
_LOCAL_COORDINATES = {'original_ordinal', 'address_mod_64', 'sha256', 'selected'}


def _need(condition, reason):
    if not condition:
        raise ValueError('seeded array loop: ' + reason)


def _one(values, reason):
    rows = tuple(values)
    _need(len(rows) == 1, reason)
    return rows[0]


def _loc(value):
    document = read_artifact(value['artifact'])
    pointer = value['locator']
    _need(pointer.startswith('#/'), 'source route locator is invalid')
    for token in pointer[2:].split('/'):
        document = document[int(token)] if type(document) is list else document[token]
    return document


def _source_guard(contract, tree, option):
    outer = entry._entry_function(tree)
    scope = contract['entry_function']['code_id']
    dotted = contract['argv']['namespace'] + '.' + option['id']
    for statement in outer.body:
        if type(statement) is not ast.If or statement.orelse:
            continue
        condition = entry._fact(statement.test, scope)
        if not any(row['condition'] == condition for row in contract['argv']['guards']):
            continue
        fail_cases = statement.test.values if type(statement.test) is ast.BoolOp and type(statement.test.op) is ast.Or else (statement.test,)
        for case in fail_cases:
            if (type(case) is not ast.Compare or len(case.ops) != 1 or type(case.ops[0]) not in (ast.Lt, ast.LtE)
                    or type(case.comparators[0]) is not ast.Constant or type(case.comparators[0].value) is not int):
                continue
            operands = case.left.args if entry._call(case.left, 'min') and not case.left.keywords else (case.left,)
            if not any(entry._name(operand) == dotted for operand in operands):
                continue
            minimum = case.comparators[0].value + (type(case.ops[0]) is ast.LtE)
            _need(minimum > 0, 'original guard does not imply a positive count')
            return minimum, condition
    raise ValueError('seeded array loop: no original positive integer parser guard')


def _source_loop(contract, witness):
    reference, worker = contract['entry'], contract['worker']
    tree, raw = seeded._ast(reference)
    scope = worker['code_id']
    body = seeded._body(tree, raw, reference, scope)
    call_site, getter_site = (witness[key] for key in ('observed_call_site', 'observed_getter_site'))
    for site in (call_site, getter_site):
        _need(site is not None and site['source'] == reference and site['code_id'] == scope,
              'observed loop invocation is outside the entry worker')
    carry = _loc(witness['source_carry'])
    consumer, producer = (_loc(witness[key]) for key in ('consumer', 'producer'))
    _need(carry['control_and_dataflow_traversed'] and witness['source_receiver_call_linked']
          and consumer['source_span'] == call_site['position'] and producer['source_span'] == getter_site['position'],
          'the original normal call and getter ports do not close the carry')
    state_name = carry['namespace']['name']
    loops = []
    for node in body.body:
        if type(node) is not ast.For or len(node.body) != 2 or node.orelse:
            continue
        first, second = (entry._assignment(row) for row in node.body)
        if first is None or second is None:
            continue
        result, call = first
        state, getter = second
        if (seeded._position(call) == call_site['position'] and seeded._position(getter) == getter_site['position']):
            loops.append((node, result, call, state, getter))
    loop, result, call, state, getter = _one(loops, 'exact original normal method-then-getter loop required')
    _need(type(loop.target) is ast.Name and type(call) is ast.Call and type(call.func) is ast.Attribute
          and type(call.func.value) is ast.Name and type(getter) is ast.Attribute and seeded._as_name(getter.value) == result
          and state == state_name and len({result, state, loop.target.id}) == 3
          and entry._call(loop.iter, 'range') and len(loop.iter.args) == 1 and not loop.iter.keywords,
          'loop side work, nonlocal operand or unqualified iterator')
    actuals = [*call.args, *(argument.value for argument in call.keywords)]
    _need(all(type(value) is ast.Name and value.id not in (result, loop.target.id) for value in actuals)
          and not any(arg.arg is None for arg in call.keywords)
          and [arg.arg for arg in call.keywords if seeded._as_name(arg.value) == state] == [witness['state_parameter']]
          and sum(value.id == state for value in actuals) == 1 and call.func.value.id not in (result, state, loop.target.id),
          'runtime call operands are not invariant lexical values plus the captured state')
    _need(not any(type(node) is ast.Name and isinstance(node.ctx, ast.Load) and node.id in (result, loop.target.id)
                  and node.lineno > loop.end_lineno for node in ast.walk(body)),
          'an extra induction or method-result value escapes the repeat region')

    formal = seeded._as_name(loop.iter.args[0])
    option = _one((row for row in contract['argv']['options'] if any(
        binding['formal'] == formal and binding['source']['expression'] == entry._term(ast.parse(
            contract['argv']['namespace'] + '.' + row['id'], mode='eval').body)
        for binding in worker['actuals'])), 'range bound is not one original parser option')
    _need(option['kind'] == 'int' and option['action'] == 'store' and not option['choices'],
          'range bound must be an unrestricted original integer parser option')
    bound_writes = [node for node in ast.walk(body) if type(node) is ast.Name and node.id == formal
                    and isinstance(node.ctx, (ast.Store, ast.Del))]
    _need(len(bound_writes) == 1 and bound_writes[0].lineno < loop.lineno, 'range parameter is rebound')
    report_source = {tuple(row['source']['position']) for row in contract['report']['worker_fields']
                     if row['source']['expression'] == entry._term(loop.iter.args[0])}
    _need(all(node is loop.iter.args[0] or tuple(seeded._position(node)) in report_source
              for node in ast.walk(body)
              if type(node) is ast.Name and isinstance(node.ctx, ast.Load) and node.id == formal),
          'range option has additional effects through other source uses')
    _need(not any((type(node) is ast.Name and node.id == 'range' and isinstance(node.ctx, (ast.Store, ast.Del)))
                  or (type(node) in (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef) and node.name == 'range')
                  or (type(node) is ast.alias and (node.asname or node.name) == 'range')
                  or (type(node) is ast.arg and node.arg == 'range') for node in ast.walk(tree)),
          'shadowed range producer')
    minimum, positive_guard = _source_guard(contract, tree, option)
    initial = _one((pair[1] for node in body.body if node.lineno < loop.lineno
                    and (pair := entry._assignment(node)) is not None and pair[0] == state),
                   'one exact initializer for the loop state is required')
    init_definitions = [item.get('producer', {}) for row in carry['initial'] for item in row['known_definitions']]
    back_definitions = [item.get('producer', {}) for row in carry['backedges'] for item in row['known_definitions']]
    exit_definitions = [item.get('producer', {}) for row in carry['normal_exits'] for item in row['known_definitions']]
    _need(type(initial) is ast.Constant and type(initial.value) is int and initial.value >= 0
          and len(init_definitions) == 1 and init_definitions[0]['kind'] == 'LITERAL'
          and init_definitions[0]['source_span'] == seeded._position(initial)
          and back_definitions and all(row['source_span'] == producer['source_span'] for row in back_definitions)
          and any(row['source_span'] == producer['source_span'] for row in exit_definitions)
          and all(row['source_span'] in (producer['source_span'], seeded._position(initial)) for row in exit_definitions),
          'integer entry and getter backedge/exit do not match the original F4 definition graph')
    return dict(id=option['id'], flags=option['flags'], minimum=minimum, default=option['default']['value'],
                parser_type='int', source_guard=positive_guard), dict(
        source=reference, code_id=scope, iterator=entry._fact(loop.iter, scope), source_span=seeded._position(loop),
        carrier=carry['namespace'], initial_value=initial.value, normal_call=call_site, normal_getter=getter_site,
        call_code_id=witness['call_code_id'], getter_code_id=witness['getter_code_id'],
        ports={key: witness[key] for key in ('source_carry', 'consumer', 'producer')},
        normal_body='CALL_THEN_GETTER_THEN_REBIND', additional_loop_statements=False,
        provider_effects='AS_REQUIRED_BY_PINNED_TEMPLATE', evidence_frontiers=carry['definition_frontiers'])


def _canonical(value, storage_ids, storages):
    if type(value) is dict:
        result = {}
        for key, child in value.items():
            if key in _LOCAL_COORDINATES:
                continue
            if key == 'storage':
                result[key] = [storage_ids.setdefault(child, len(storage_ids)), storages[child]['size_bytes']]
            else:
                result[key] = _canonical(child, storage_ids, storages)
        return result
    if type(value) is list:
        return [_canonical(child, storage_ids, storages) for child in value]
    return value


def _layout(value):
    return {key: value[key] for key in ('dtype', 'shape', 'strides_bytes', 'offset_bytes')}


def _round(tape, raw, extent, sites, previous, initial_value):
    begin, end = extent['begin'], extent['end']
    events = tape['events'][begin:end]
    _need(len(events) >= 3 and [row['kind'] for row in events[-2:]] == ['NORMALIZE', 'CHECKPOINT'],
          'one original normalization and result boundary are required')
    normalize, checkpoint = events[-2:]
    returned = dict(raw=normalize['input'], normalized=normalize['normalized'])
    chosen = returned['normalized' if normalize['selected'] else 'raw']
    _need(checkpoint['input'] == chosen and returned['raw']['storage'] != returned['normalized']['storage']
          and _layout(returned['raw']) == _layout(returned['normalized']) and raw['output']['sha256'] == checkpoint['sha256'],
          'original result selection no longer matches the available branch outputs')
    ingress = None
    if previous is None:
        _need(raw['input'] == dict(kind='EXACT_INT', value=initial_value), 'observed initial basis differs from its source value')
    else:
        first, copied = events[:2]
        _need(first['kind'] == 'VIEW' and copied['kind'] == 'COPY' and first['output'] == copied['input']
              and first['input'] == previous and _layout(first['input']) == _layout(chosen)
              and first['input']['storage'] == first['output']['storage'] != copied['output']['storage'],
              'continuation does not copy from the actual selected previous result into private storage')
        ingress = first['input']
    storage_ids, operations = {}, []
    dynamic_indices = {site['event_index'] for site in sites}
    for index, event in enumerate(events[:-1], begin):
        shape = _canonical(event, storage_ids, tape['storages'])
        if index in dynamic_indices:
            field = 'coefficients' if shape['kind'] == 'PARTITION' else 'coefficient'
            _need(field in shape, 'bound coefficient is outside its source operation')
            shape[field] = 'RUNTIME_SOURCE_BINDING'
        operations.append(shape)
    if ingress is not None:
        _need(all(row['storage'] != ingress['storage'] for event in events[2:] for row in event.values()
                  if type(row) is dict and 'storage' in row), 'private recurring input storage escapes the entry copy')
    source_geometry = [{key: event[key] for key in ('kind', 'source', 'reason') if key in event}
                       for event in raw['events']]
    index_geometry = [{key: value for key, value in site.items() if key not in ('event_index', 'round', 'original_ordinal')}
                      | {'index_in_round': site['event_index'] - begin} for site in sites]
    signature = dict(operations=operations, source=source_geometry, coefficients=index_geometry)
    return dict(round=extent['index'], begin=begin, end=end, input=ingress, output=returned), chosen, signature


def _tape(row, contract, source_sites):
    tape = read_artifact(row['tape'])
    observed = read_artifact(tape['observation'])
    binding = read_artifact(tape['source_binding'])
    _need(len(tape['rounds']) == len(observed['rounds']) >= 3 and not tape['frontiers'],
          'two consecutive original continuation occurrences with no tape frontier are required')
    witness = _one((match for match in binding['matches'] if match['observed_activation']['artifact'] == tape['witness']),
                   'exact original recurrence witness for this tape is required')
    source = _source_loop(contract, witness)
    if source_sites is not None:
        _need(source == source_sites, 'profiles bind different original recurrence sites')
    selected = None
    initial, continuation, signature = None, None, None
    continuation_rounds = []
    for number, (extent, raw) in enumerate(zip(tape['rounds'], observed['rounds'], strict=True)):
        part, selected, shape = _round(tape, raw, extent, [site for site in row['sites'] if site['round'] == number],
                                       selected, source[1]['initial_value'])
        if number == 0:
            initial = part
        elif number == 1:
            continuation, signature = part, shape
        else:
            _need(shape == signature, 'continuation alias/order/layout or source coefficient routes differ')
            continuation_rounds.append(number)
    digest = sha256(json.dumps(signature, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(tape=row['tape'], seed=row['seed'], initial=initial, continuation=continuation,
                confirmed_rounds=continuation_rounds, continuation_signature_sha256=digest), source


def project(seed_binding_path):
    """Replay an existing seed binding and recover its finite repeated region."""
    reference = artifact_reference(Path(seed_binding_path).resolve(strict=True), seeded.SCHEMA)
    bound = seeded.check(reference['uri'])
    contract = read_artifact(bound['source_contract'])
    reports, source = [], None
    for tape in bound['tapes']:
        report, source = _tape(tape, contract, source)
        reports.append(report)
    option, source_loop = source
    _need(option['id'] in bound['fixed_options'] and option['id'] != bound['construction']['seed_option'],
          'trip-count option changes the selected RNG construction')
    _need(all(row['continuation_signature_sha256'] == reports[0]['continuation_signature_sha256'] for row in reports),
          'normalized continuation differs across original seeds')
    return dict(schema=SCHEMA, schema_version=1, seed_binding=reference, option=option, source_loop=source_loop,
                tapes=reports, storage_aliases='EXACT_LOCAL_RENAMING_WITH_PRIVATE_CONTINUATION_INPUT_COPY',
                selection='RUN_ORIGINAL_NORMALIZATION_BRANCH_AND_FORWARD_SELECTED_EXACT_ARRAY',
                continuation_rules='FIRST_OCCURRENCE_THEN_REPEAT_CONTINUATION_FOR_REMAINING_COUNT',
                guards=['PINNED_PROVIDERS_AND_NONLOOP_PROFILE', 'SELECTED_ARRAY_RESULT_GEOMETRY', 'DYNAMIC_SEED_GUARDS'],
                unobserved_counts_require_validation=True, deployment_authorized=False)


def check(path):
    stored = read_artifact(artifact_reference(Path(path).resolve(strict=True), SCHEMA))
    _need(stored == project(stored['seed_binding']['uri']), 'stored loop binding differs from cold replay')
    return stored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('seed-binding', 'check', 'out'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args(argv)
    if args.check and not (args.seed_binding or args.out):
        result = check(args.check)
    elif args.seed_binding and args.out and not args.check:
        result = project(args.seed_binding)
        with args.out.open('x', encoding='utf-8') as output:
            json.dump(result, output, allow_nan=False, indent=2, sort_keys=True)
            output.write('\n')
    else:
        parser.error('supply --seed-binding BINDING --out RESULT or --check RESULT')
    print(json.dumps(dict(option=result['option']['id'], minimum=result['option']['minimum'],
                          confirmed_rounds=[row['confirmed_rounds'] for row in result['tapes']])))


if __name__ == '__main__':
    main()
