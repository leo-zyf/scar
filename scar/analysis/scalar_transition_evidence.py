"""Bind a sampled scalar call and complete tuple transition to their F4 ports."""

import argparse
import ast
from collections import Counter, defaultdict
from dis import get_instructions
import json
from pathlib import Path

from scar.analysis import scalar_score_evidence, scalar_table_handoff
from scar.analysis.scalar_score_evidence import _compile_source, _open, _pinned_document
from scar.analysis.program_view import ProgramView
from scar.analysis.function_interface_observations import RetainedCodeBodyIndex
from scar.analysis.static.lowering.source_call_lowering import _body
from scar.analysis.source_references import source_port
from scar.compiler.call_binding import plan_call_bindings
from scar.ir.execution_recipe import read_artifact
from scar.ir.record_codec import encode
from scar.trace.code_snapshot import describe_code, site

SCHEMA = 'scar.scalar-transition-evidence'
DESIGN = 'scar.scalar-transition-selection'
CAPTURE = 'scar.scalar-transition-capture'
KINDS = ('EXACT_ASCII', 'EXACT_I64', 'EXACT_BOOL_TUPLE_2')


def _need(valid, reason):
    if not valid:
        raise ValueError('scalar transition: ' + reason)


def _one(rows, reason):
    rows = tuple(rows)
    _need(len(rows) == 1, reason)
    return rows[0]


def _span(node):
    return [node.lineno, node.end_lineno, node.col_offset, node.end_col_offset]


def _sites(code, node):
    sites = [site(code, op.offset) for op in get_instructions(code) if op.opname == 'CALL'
             and site(code, op.offset)['position'] == _span(node)]
    _need(sites, 'original nested call lacks a compiled CALL site')
    return sites


def _call(code, node):
    return _one(_sites(code, node), 'ordinary nested operation requires one compiled CALL site')


def _constructor(node, width):
    _need(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and len(node.args) == width
          and not node.keywords and all(not isinstance(value, ast.Starred) for value in node.args),
          'tuple construction requires one direct named positional carrier')
    return node.func.id


def _design(score, stored_score, per_version, max_versions):
    observed, score_raw, score_design = scalar_score_evidence._replay(stored_score)
    supplier = read_artifact(observed['handoff'])
    selection = read_artifact(read_artifact(supplier['capture'])['selection'])
    ordinary = read_artifact(supplier['handoff'])
    _need(ordinary['catalog'] == observed['catalog'] and selection['source'] == observed['source'],
          'scalar score and outer handoff require the same original source')
    selected = read_artifact(selection['handoff_selection'])
    _need(len({row['formal'] for row in ordinary['observed_edges']}) == 1,
          'outer result does not identify one original consumer formal')
    consumer_formal = ordinary['observed_edges'][0]['formal']
    source = observed['source']
    raw, compiled = _compile_source(source)
    tree = ast.parse(raw, filename=source['uri'])
    codes = {code.co_qualname: code for code in compiled}
    from scar.compiler.scalar_table_region import _definition

    producer_code = codes[selected['producer']['code']['qualname']]
    scalar_code = codes[observed['callee']['code']['qualname']]
    _need(describe_code(producer_code) == selected['producer']['code'] == selection['producer_code']
          and describe_code(scalar_code) == observed['callee']['code'] == selection['value_code'],
          'move and score compiler identities differ')
    producer = _definition(tree, producer_code.co_qualname)
    owner_name, method_name = producer_code.co_qualname.split('.')
    owner = _one((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner_name),
                 'one source class must own the selected transition')
    _need(len(owner.bases) == 1 and not owner.keywords and not owner.decorator_list,
          'tuple transition needs one ordinary inherited source class')
    named = owner.bases[0]
    _need(isinstance(named, ast.Call) and isinstance(named.func, ast.Name) and len(named.args) == 2
          and not named.keywords and all(isinstance(arg, ast.Constant) and type(arg.value) is str for arg in named.args)
          and any(isinstance(node, ast.ImportFrom) and node.module == 'collections' and node.level == 0
              and any(alias.name == 'namedtuple' and (alias.asname or alias.name) == named.func.id
                      for alias in node.names) for node in tree.body),
          'tuple field order requires explicit original collections declaration')
    names = named.args[1].value.split()
    _need(0 < len(names) <= 12 and all(name.isidentifier() for name in names) and len(set(names)) == len(names),
          'tuple declaration needs unique finite ordered source fields')

    producer_return = _one((node for node in producer.body if isinstance(node, ast.Return)),
                           'transition requires one explicit final source return')
    dispatch = producer_return.value
    _need(isinstance(dispatch, ast.Call) and isinstance(dispatch.func, ast.Attribute)
          and isinstance(dispatch.func.value, ast.Call) and not dispatch.args and not dispatch.keywords,
          'producer must return one ordinary no-argument method on its constructed value')
    rotation_name = dispatch.func.attr
    _need(rotation_name != method_name, 'post-construction method must be a separate source body')
    rotation_code = codes[f'{owner.name}.{rotation_name}']
    rotation = _definition(tree, rotation_code.co_qualname)
    wrapped = _one((node.value for node in rotation.body if isinstance(node, ast.Return)),
                   'post-construction method requires one explicit original return value')
    inner = dispatch.func.value
    ctor = _constructor(inner, len(names))
    _need(ctor == _constructor(wrapped, len(names)) == owner.name,
          'intermediate and final constructor must select the same original declaration')

    globals_ = sorted({op.argval for code in (producer_code, scalar_code, rotation_code)
                       for op in get_instructions(code) if op.opname == 'LOAD_GLOBAL'} - {ctor, score_design['table']})
    _need(type(per_version) is int and 12 <= per_version <= 24
          and type(max_versions) is int and 2 <= max_versions <= 4,
          'small bounded two-or-more table-version study expected')
    design = dict(schema=DESIGN, schema_version=1, source=source, score_evidence=score, catalog=observed['catalog'],
        old_handoff=supplier['handoff'], caller=selected['caller'], producer=selected['producer'],
        consumer={**selected['consumer'], 'formal': consumer_formal}, score=observed['callee']['code'],
        score_call=selection['value_call'], score_receiver=selection['value_receiver'],
        score_argument=selection['value_formal'], producer_receiver=selection['producer_receiver'],
        producer_argument=selection['producer_formal'], score_field=selection['score_field'],
        rotation=dict(code=describe_code(rotation_code), caller_site=_call(producer_code, dispatch),
                      receiver=describe_code(rotation_code)['parameters'][0],
                      parameters=describe_code(rotation_code)['parameters'][1:]),
        constructor=dict(name=ctor, declaration_span=_span(owner), fields=names,
            sites=dict(producer=_sites(producer_code, inner), rotation=_sites(rotation_code, wrapped))),
        table=score_design['table'], globals=globals_, per_version=per_version, max_versions=max_versions)
    return design, score_raw, compiled


def prepare(score, *, per_version=16, max_versions=2):
    reference, stored = _open(score, scalar_score_evidence.SCHEMA)
    design, _, _ = _design(reference, stored, per_version, max_versions)
    return design


def field_kind(value):
    if type(value) is str and value.isascii():
        return KINDS[0]
    if scalar_table_handoff._int64(value):
        return KINDS[1]
    if type(value) is list and len(value) == 2 and all(type(item) is bool for item in value):
        return KINDS[2]
    raise ValueError('unsupported exact transition state field')


def _f4(design):
    from scar.analysis.source_declared_calls import _owners
    from scar.analysis.source_references import ReferenceSlicer
    from scar.compiler.graph_frontend import _metadata, _source_identity

    view = ProgramView(Path(design['catalog']['uri']).parent)
    read_artifact(design['catalog'])
    index = RetainedCodeBodyIndex(view)
    source = design['source']['uri']
    descriptors = dict(caller=design['caller'], score=design['score'],
                       **{role: design[role]['code'] for role in ('producer', 'rotation', 'consumer')})
    bodies, contexts = {}, {}
    for role, code in descriptors.items():
        body, authority, facts, function = _body(index, code)
        contexts[role] = facts, function
        bodies[role] = dict(body=encode(body), code=code, function_fact=authority,
            formals=[dict(name=name, port=encode(source_port(facts, ordinal)))
                     for name,ordinal in zip(function.parameter_names,function.formals[1:-1],strict=True)],
            normal_return=None if function.returns is None else encode(source_port(facts, function.returns)))
    calls = dict(producer=[design['producer']['caller_site']], consumer=[design['consumer']['caller_site']],
        score=[design['score_call']], rotation=[design['rotation']['caller_site']],
        **{f'{name}_constructor': value for name,value in design['constructor']['sites'].items()})
    outputs = {}
    for caller in ('caller', 'producer', 'rotation'):
        needed = [name for name, selected in calls.items() if selected[0]['code'] == descriptors[caller]['key']]
        if not needed:
            continue
        plan = plan_call_bindings(view, source, entry=descriptors[caller]['qualname'])
        for role in needed:
            sites = calls[role]
            if len(sites) == 1:
                outputs[role] = _one((row for row in plan['sites'] if row['site'] == sites[0]),
                                     'retained runtime site has no exact original F4 invocation: ' + role)
                continue
            facts, function = contexts[caller]
            _, syntax, _, _, _ = _source_identity(view.catalog, facts.unit)
            owners = _owners(facts)
            expected = sites[0]['position']
            _need(all(candidate['position'] == expected for candidate in sites),
                  'compiled constructor variants do not share a source expression')
            candidates = []
            for query, record in enumerate(facts.calls):
                if function.body_scope.owner not in owners[query]:
                    continue
                port = source_port(facts, record.result)
                _, span, _ = _metadata(syntax, port.port.template.occurrence)
                if [span[key] for key in ('line', 'end_line', 'column', 'end_column')] == expected:
                    candidates.append((query, port))
            query, port = _one(candidates, 'constructor variants lack one original CALL result')
            outputs[role] = dict(unit=facts.unit, query=query, original_result=encode(port),
                sites=sites, mapping_rule='COMPILED_CALL_VARIANTS_SHARE_ORIGINAL_F4_SOURCE_EXPRESSION',
                fact_reference=encode(ReferenceSlicer(view.catalog).fact(facts, 'calls', query)))
    _need(outputs.keys() == calls.keys()
          and all(bodies[role]['normal_return'] is not None for role in ('producer', 'rotation', 'score')),
          'expected original source normal returns and six call ports')
    index.check()
    return bodies, outputs


def _check_capture(raw, design, *, prior=None, compiled=None):
    if prior is None:
        prior = read_artifact(read_artifact(design['score_evidence'])['capture'])
    versions = raw['table_versions']
    schema = raw['state_fields']
    _need(list(versions) == sorted(prior['table_versions']) and
          all(row['rows'] == prior['table_versions'][digest]['rows'] and
              scalar_table_handoff.table_version(row['rows']) == digest and
              type(row['root_token']) is int and row['root_token'] > 0 for digest,row in versions.items()),
          'full transition and score evidence need the same observed table contents')
    globals_ = raw['globals']
    i64 = scalar_table_handoff._int64
    _need(all(globals_.get(name) == value for name,value in prior['globals'].items()) and
          all(i64(value) for value in globals_.values()) and raw['builtins'] == prior['builtins']
          and raw['builtin_abs_verified'] is prior['builtin_abs_verified'] is True and
          set(globals_) | set(raw['builtins']) == set(design['globals']),
          'original transition module constants or builtin identities differ')
    declared = {row['name']:row['kind'] for row in schema}
    _need(len(declared) == len(schema) and list(declared) == design['constructor']['fields']
          and set(declared.values()) <= set(KINDS), 'original exact state tuple ordinals/types differ')
    if compiled is None:
        _, compiled = _compile_source(design['source'])
    codes = {describe_code(code)['key']:code for code in compiled}
    seen = set()
    counts = Counter()
    constructors = set()
    for sample in raw['samples']:
        version = sample['table_version']
        ids = [sample[name+'_frame'] for name in ('caller','producer','score','rotation','consumer')]
        _need(all(type(value) is int and value > 0 for value in ids)
              and ids == sorted(set(ids)) and sample['producer_frame'] not in seen
              and type(sample['thread_id']) is int and sample['thread_id'] > 0
              and sample['root_token'] == versions[version]['root_token'],
              'scoring, construction and selected consumption must share ordered occurrence evidence')
        seen.add(sample['producer_frame'])
        _need(sample['producer_site'] == design['producer']['caller_site']
              and sample['consumer_site'] == design['consumer']['caller_site']
              and sample['score_site'] == design['score_call']
              and sample['rotation_site'] == design['rotation']['caller_site']
              and sample['rotation_arg'] is False and sample['producer_is_rotation'] is True
              and sample['consumer_is_return'] is True,
              'recorded transition did not take the original nested normal path')
        for role in ('score','rotation','producer'):
            expected = design['score'] if role == 'score' else design[role]['code']
            returned = sample[role+'_return']
            _need(returned['code'] == expected['key'] and returned['opname'] in ('RETURN_VALUE','RETURN_CONST')
                  and site(codes[expected['key']],returned['offset']) == returned,
                  'sampled return site is not the original body: '+role)
        _need(set(sample['constructors']) == {'producer','rotation'}, 'both original typed CALL carriers required')
        for role,target in sample['constructors'].items():
            _need(target['site'] in design['constructor']['sites'][role]
                  and target['type_token'] == 1 and target['protocol'] == 'OBSERVED_EXACT_TYPE_CARRIER'
                  and target['target']['qualname'] == design['constructor']['name']
                  and type(target['target']['module']) is str, 'exact selected constructor differs')
            constructors.add((target['target']['module'],target['target']['qualname']))
        states = [sample[key] for key in ('receiver_fields','intermediate_fields','returned_fields')]
        _need(all({name:field_kind(value) for name,value in state.items()} == declared for state in states),
              'transition field shape or primitive type differs')
        move = sample['argument']
        _need(type(move) is list and len(move) == 3 and i64(move[0]) and i64(move[1]) and
              type(move[2]) is str and move[2].isascii(), 'selected original argument layout differs')
        score = design['score_field']
        _need(i64(states[0][score]) and i64(sample['score_delta']) and i64(states[1][score]) and
              states[1][score] == states[0][score] + sample['score_delta'],
              'observed intermediate score did not consume the same-frame original scalar return')
        counts[version] += 1
    _need(len(constructors) == 1 and len(counts) == design['max_versions']
          and all(count == design['per_version'] for count in counts.values())
          and set(counts) == set(versions), 'two complete bounded original transition scopes required')
    return counts, next(iter(constructors))


def _project(reference, raw):
    selection = read_artifact(raw['selection'])
    score = raw['score_evidence']
    _, stored_score = _pinned_document(score, scalar_score_evidence.SCHEMA)
    design, prior, compiled = _design(score, stored_score, selection['per_version'], selection['max_versions'])
    launch = raw['launch']
    _need(raw['schema_version'] == 1 and selection == design and raw['source'] == design['source']
          and type(raw['process_id']) is int and raw['process_id'] > 0 and raw['original_execution'] is True
          and launch['entry_path'] == design['source']['uri'] and launch['decision'] == 'NO_OP' and launch['exit_code'] == 0,
          'complete actual original state transition capture required')
    counts, constructor = _check_capture(raw, design, prior=prior, compiled=compiled)
    bodies, calls = _f4(design)
    original = read_artifact(design['old_handoff'])
    _need(original['original_sites']['producer'] == calls['producer']
          and original['original_sites']['consumer'] == calls['consumer']
          and {row['formal'] for row in original['observed_edges']} == {design['consumer']['formal']},
          'new return and recursive consumer do not bind the original source ports')
    typed_formal = _one((row for row in bodies['consumer']['formals'] if row['name'] == design['consumer']['formal']),
                        'one original recursive argument port required')
    return dict(schema=SCHEMA,schema_version=1,source=design['source'],score_evidence=design['score_evidence'],
        capture=reference,catalog=design['catalog'],original_handoff=design['old_handoff'],
        callee=bodies['producer'],score=bodies['score'],rotation=bodies['rotation'],
        caller=bodies['caller'],consumer={**bodies['consumer'],'actual_formal':typed_formal},
        constructor=dict(name=design['constructor']['name'], field_order=design['constructor']['fields'],
            declaration_span=design['constructor']['declaration_span'], observed_type=list(constructor),
            original_calls={role:calls[role+'_constructor'] for role in ('producer','rotation')}),
        original_calls=calls,table_versions=sorted(counts),sample_counts=dict(counts),
        state_fields=raw['state_fields'],capture_frontiers=raw['observation_frontiers'],
        qualification=dict(same_original_process=True, nested_score_actual_return_observed=True,
            constructor_type_carrier_observed=True, constructor_implicit_init_and_effects_closed=False,
            rotation_default_false_observed=True, normal_return_object_enters_original_consumer=True,
            type_and_value_evidence_for_sampled_inputs=True, continuous_table_no_writer_proven=False,
            future_dispatch_proven=False, external_effects_closed=False, transformation_authorized=False,
            target_executed_by_projection=False))


def project(capture_path):
    return _project(*_open(capture_path, CAPTURE))


def check(path):
    _, stored = _open(path, SCHEMA)
    rebuilt = _project(*_pinned_document(stored['capture'], CAPTURE))
    _need(stored == rebuilt,'stored complete transition differs from independent replay')
    return rebuilt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare')
    group.add_argument('--capture')
    group.add_argument('--check')
    parser.add_argument('--out')
    args = parser.parse_args()
    if args.check:
        value = check(args.check)
        print(json.dumps(dict(samples=value['sample_counts'],fields=value['state_fields'])))
    else:
        if args.out is None:
            parser.error('publication requires --out')
        value = prepare(args.prepare) if args.prepare else project(args.capture)
        print(json.dumps(scalar_table_handoff._write(args.out,value)))


if __name__ == '__main__':
    main()
