"""Join a guarded scalar read window to a sampled caller accumulator and return."""

import argparse
import ast
from dis import get_instructions
from hashlib import sha256
import json
from pathlib import Path

from scar.analysis import scalar_table_cse, source_call_handoff
from scar.analysis.program_view import ProgramView
from scar.analysis.source_reference_slice import ReferenceSlicer, source_port
from scar.compiler.call_binding import plan_call_bindings
from scar.compiler.graph_frontend import _metadata, _source_identity
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program.normal_links import LanguageOperationKind
from scar.ir.record_codec import encode
from scar.runtime.source_call_handoff_observe import DESIGN as HANDOFF
from scar.runtime.source_value_observe import _codes
from scar.trace.code_snapshot import describe_code, site

SCHEMA = 'scar.scalar-table-handoff'
DESIGN = 'scar.scalar-table-handoff-selection'
CAPTURE = 'scar.scalar-table-handoff-capture'


def _need(valid, reason):
    if not valid:
        raise ValueError('scalar table handoff: ' + reason)


def _one(rows, reason):
    rows = tuple(rows)
    _need(len(rows) == 1, reason)
    return rows[0]


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write('\n')
    return _ref(path, value['schema'])


def _span(node):
    return [node.lineno, node.end_lineno, node.col_offset, node.end_col_offset]


def _same_source_facts(value):
    if isinstance(value, dict):
        if set(value) == {'uri', 'sha256', 'size_bytes', 'schema'}:
            return {key: value[key] for key in ('sha256', 'size_bytes', 'schema')}
        return {key: _same_source_facts(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_same_source_facts(child) for child in value]
    return value


def table_version(rows):
    wire = json.dumps(rows, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
    return sha256(wire.encode('ascii')).hexdigest()


def _literal_or_local(node):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return {'integer': node.value}
    _need(isinstance(node, ast.Name), 'table index must be a source integer or exact initialized local')
    return {'local': node.id}


def _store_end(code, assignment):
    target, = assignment.targets
    opcodes = list(get_instructions(code))
    ordinal = _one((n for n, op in enumerate(opcodes) if op.opname == 'STORE_FAST' and op.argval == target.id
                    and site(code, op.offset)['position'] == _span(target)), 'assignment has no exact post-store instruction')
    return site(code, opcodes[ordinal + 1].offset)


def _write_frontiers(tree, name):
    writes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del)):
            if isinstance(node.value, ast.Name) and node.value.id == name:
                literal = node.slice
                key = literal.value if isinstance(literal, ast.Constant) and type(literal.value) is str else None
                writes.append(dict(role='ITEM_WRITE_OR_DELETE', key=key, span=_span(node)))
        elif isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, (ast.Store, ast.Del)):
            writes.append(dict(role='POSSIBLE_BINDING_WRITE_OR_DELETE', key=None, span=_span(node)))
    return sorted(writes, key=lambda row: row['span'])


def _design(proof, proof_case, historical, handoff_ref):
    report = scalar_table_cse.check(proof)
    _need(type(proof_case) is int and 0 <= proof_case < len(report['cases']), 'one current F4 scalar case required')
    current = report['cases'][proof_case]
    earlier = read_artifact(historical)
    old = _one((case for case in earlier['cases'] if (case['source'], case['code_key'], case['span'])
               == (current['source'], current['code_key'], current['span'])), 'historical proof must name the same source expression')
    _need(all(_same_source_facts(value) == _same_source_facts(current.get(key))
              for key, value in old.items() if key != 'obligations')
          and set(old['obligations']) <= set(current['obligations']),
          'retained expression/effect order differs from current F4 proof')

    handoff = read_artifact(handoff_ref)
    capture = read_artifact(handoff['capture'])
    selection_ref = capture['selection']
    selection = read_artifact(selection_ref)
    source = current['source']
    _need(selection_ref['schema'] == HANDOFF and selection['source'] == source
          and report['catalog'] == handoff['catalog'], 'selected producer and table case require one current source graph')
    raw = read_artifact(source, json_document=False)
    tree = ast.parse(raw, filename=source['uri'])
    compiled = tuple(_codes(compile(raw, source['uri'], 'exec', dont_inherit=True, optimize=0)))
    value_code = _one((code for code in compiled if describe_code(code)['key'] == current['code_key']),
                     'table expression lacks exact original code')
    producer_code = _one((code for code in compiled if describe_code(code) == selection['producer']['code']),
                        'handoff producer lacks exact original code')
    from scar.compiler.scalar_table_region import _definition

    value = _definition(tree, value_code.co_qualname)
    expected = current['span']
    span = [expected[key] for key in ('line', 'end_line', 'column', 'end_column')]
    selected = _one((node for node in value.body if isinstance(node, ast.Assign)
                     and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                     and _span(node.value) == span), 'scalar case must assign its original accumulator')
    expr = selected.value
    _need(isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Sub) and current['operation'] == 'Sub',
          'supported sampled scalar window is an exact subtraction')
    table, row_key = expr.left.value.value.id, expr.left.value.slice.id
    base_local = selected.targets[0].id
    suffix = value.body[value.body.index(selected)+1:]
    returns = [node for node in ast.walk(ast.Module(body=suffix, type_ignores=[])) if isinstance(node, ast.Return)]
    _need(len(returns) == 1 and isinstance(returns[0].value, ast.Name)
          and returns[0].value.id == base_local, 'value return must read the selected accumulator')
    first = _one((op for op in get_instructions(value_code) if op.opname == 'LOAD_GLOBAL'
                  and op.argval == table and site(value_code, op.offset)['position'] == _span(expr.left.value.value)),
                 'first global table read is not uniquely present')

    producer = _definition(tree, producer_code.co_qualname)
    receiver, move = describe_code(producer_code)['parameters'][:2]
    choices = []
    for candidate in producer.body:
        if not isinstance(candidate, ast.Assign) or len(candidate.targets) != 1 or not isinstance(candidate.targets[0], ast.Name):
            continue
        rhs = candidate.value
        if (isinstance(rhs, ast.BinOp) and isinstance(rhs.op, ast.Add)
            and isinstance(rhs.left, ast.Attribute) and isinstance(rhs.left.value, ast.Name)
            and rhs.left.value.id == receiver and isinstance(rhs.right, ast.Call) and not rhs.right.keywords
            and isinstance(rhs.right.func, ast.Attribute) and isinstance(rhs.right.func.value, ast.Name)
            and rhs.right.func.value.id == receiver and rhs.right.func.attr == value.name
            and len(rhs.right.args) == 1 and isinstance(rhs.right.args[0], ast.Name) and rhs.right.args[0].id == move):
            choices.append(candidate)
    total = _one(choices, 'producer must add the same-frame value return to its original receiver field')
    call = total.value.right
    call_site = _one((site(producer_code, op.offset) for op in get_instructions(producer_code)
                      if op.opname == 'CALL' and site(producer_code, op.offset)['position'] == _span(call)),
                     'score delta does not have one retained original CALL')
    self_name, input_name = describe_code(value_code)['parameters'][:2]
    fields = sorted({node.attr for node in ast.walk(value) if isinstance(node, ast.Attribute)
                     and isinstance(node.value, ast.Name) and node.value.id == self_name} | {total.value.left.attr})
    return dict(schema=DESIGN, schema_version=1, source=source, case=proof, case_index=proof_case,
        historical_proof=historical, original_handoff=handoff_ref, handoff_selection=selection_ref,
        value_code=describe_code(value_code), value_call=call_site, value_receiver=self_name, value_formal=input_name,
        value_before=site(value_code, first.offset), value_after=_store_end(value_code, selected),
        table=table, key=row_key, left=_literal_or_local(expr.left.slice), right=_literal_or_local(expr.right.slice),
        base_local=base_local, accumulator_return=_span(returns[0]),
        producer_code=describe_code(producer_code), producer_receiver=receiver, producer_formal=move,
        producer_store_after=_store_end(producer_code, total), producer_total=total.targets[0].id,
        add_span=_span(total.value), score_field=total.value.left.attr, position_fields=fields,
        static_table_write_frontiers=_write_frontiers(tree, table))


def prepare(proof, historical, handoff, *, case=0):
    return _design(_ref(proof, scalar_table_cse.SCHEMA), case, _ref(historical, scalar_table_cse.SCHEMA),
                   _ref(handoff, source_call_handoff.SCHEMA))


def _f4_total(design, graph):
    from scar.analysis.source_call_handoff import _body, _original_route
    from scar.analysis.retained_code_bodies import RetainedCodeBodyIndex

    view = ProgramView(Path(graph['catalog']['uri']).parent)
    source = design['source']['uri']
    plan = plan_call_bindings(view, source, entry=design['producer_code']['qualname'])
    call = _one((row for row in plan['sites'] if row['site'] == design['value_call']),
                'nested value CALL has no unique original F4 port')
    unit = view.unit(call['unit'])
    _, syntax, _, _, _ = _source_identity(view.catalog, unit.unit)
    operations = []
    for ordinal, operation in enumerate(unit.facts.operations):
        if operation.action is LanguageOperationKind.BINARY and operation.selector.value == 'Add':
            result = source_port(unit.facts, operation.result)
            _, position, _ = _metadata(syntax, result.port.template.occurrence)
            if [position[key] for key in ('line', 'end_line', 'column', 'end_column')] == design['add_span']:
                operations.append((ordinal, operation, result))
    ordinal, operation, result = _one(operations, 'producer total does not have one exact original F4 addition')
    body_index = RetainedCodeBodyIndex(view)
    body, function, _, source_fn = _body(body_index, design['producer_code'])
    callee, callee_fact, callee_facts, callee_fn = _body(body_index, design['value_code'])
    delta = source_port(unit.facts, call['result_port'])
    carried = _original_route(unit, source_fn, delta, source_port(unit.facts, operation.operands[1]))
    body_index.check()
    return dict(producer=encode(body), producer_fact=function, value_body=encode(callee), value_fact=callee_fact,
        value_return=encode(source_port(callee_facts, callee_fn.returns)),
        delta_call=call, addition_fact=encode(ReferenceSlicer(view.catalog).fact(unit.facts, 'operations', ordinal)),
        result=encode(result), delta_transport=carried,
        original_score_operand=encode(source_port(unit.facts, operation.operands[0])))


def _int64(value):
    return type(value) is int and -(1 << 63) <= value < (1 << 63)


def _samples(design, raw, flow, graph):
    versions = raw['table_versions']
    _need(0 < len(versions) <= 32, 'expected bounded original table versions')
    for digest, version in versions.items():
        rows = version['rows']
        _need(type(rows) is dict and 0 < len(rows) <= 64 and
              all(type(key) is str and type(row) is list and row and all(_int64(value) for value in row)
                  for key, row in rows.items()) and sum(map(len, rows.values())) <= 4096 and
              digest == table_version(rows) and type(version['root_token']) is int and version['root_token'] > 0,
              'table version differs from exact observed scalar rows')
    _need(raw['handoff_capture'] == graph['capture'] and len(flow['matched']) == len(graph['observed_edges'])
          and raw['process_id'] == flow['process_id'] and raw['thread_id'] == flow['thread_id']
          and flow['selection'] == design['handoff_selection'] and raw['original_execution'] is True
          and raw['table_scope'] == 'EXACT_BUILTIN_ROW_SNAPSHOTS_AT_SELECTED_EXPRESSION_BOUNDARIES',
          'scalar captures and object flow must come from one original run')
    selected = set()
    fields = set(design['position_fields'])
    raw_code = read_artifact(design['source'], json_document=False)
    compiled = {describe_code(code)['key']: code for code in _codes(
        compile(raw_code, design['source']['uri'], 'exec', dont_inherit=True, optimize=0))}
    result = []
    for index, sample in enumerate(raw['samples']):
        ordinal = sample['handoff_index']
        _need(type(ordinal) is int and 0 <= ordinal < len(flow['matched']) and ordinal not in selected,
              'one unique original handoff pointer per scalar sample required')
        selected.add(ordinal)
        occurrence = flow['matched'][ordinal]
        edge = graph['observed_edges'][ordinal]
        _need(sample['producer'] == occurrence['producer'] and
              sample['producer_return'] == occurrence['producer_return'] == edge['producer_return'],
              'scalar result attached to another original producer frame')
        returned = sample['value_return']
        _need(returned['code'] == design['value_code']['key'] and
              returned['opname'] in ('RETURN_VALUE', 'RETURN_CONST') and
              returned['position'] == design['accumulator_return'] and
              site(compiled[returned['code']], returned['offset']) == returned,
              'delta sample does not return the original accumulator')
        row = versions[sample['table_version']]['rows'][sample['key']]
        left, right = sample['left'], sample['right']
        _need(type(sample['key']) is str and _int64(left) and _int64(right) and
              -len(row) <= left < len(row) and -len(row) <= right < len(row) and sample['row_size'] == len(row)
              and sample['left_value'] == row[left] and sample['right_value'] == row[right]
              and sample['result'] == row[left] - row[right] and
              sample['root_token'] == versions[sample['table_version']]['root_token'],
              'sampled original subtraction differs from its exact table version')
        start, end = sample['input_position'], sample['returned_position']
        field = design['score_field']
        _need(set(start) == set(end) == fields and all(type(value) in (str, int)
              for value in (*start.values(), *end.values())) and
              all(_int64(value) for value in (sample['result'], sample['value_delta'], start[field],
                  sample['producer_total'], end[field])) and
              sample['producer_total'] == start[field] + sample['value_delta'],
              'sampled sum and source position field values differ')
        result.append(dict(scalar_sample=f'#/samples/{index}', source_handoff=f'#/observed_edges/{ordinal}',
            table_version=sample['table_version'], caller=edge['caller'], produced=sample['producer'],
            score=dict(initial=start[field], expression=sample['result'], delta=sample['value_delta'],
                accumulated=sample['producer_total'], output_field=end[field]),
            returned_fields_same_reference_at_consumer=True))
    _need(bool(result), 'no same-run scalar route was completed')
    return result


def project(handoff_path, capture_path):
    handoff = _ref(handoff_path, source_call_handoff.SCHEMA)
    graph = source_call_handoff.check(handoff['uri'])
    capture = _ref(capture_path, CAPTURE)
    raw = read_artifact(capture)
    selection = read_artifact(raw['selection'])
    design = _design(raw['case'], raw['case_index'], selection['historical_proof'], selection['original_handoff'])
    _need(raw['schema_version'] == 1 and selection == design and raw['source'] == design['source'] and
          graph['catalog'] == read_artifact(design['case'])['catalog'],
          'capture does not use the exact current source, effect proof and graph')
    flow = read_artifact(raw['handoff_capture'])
    samples = _samples(design, raw, flow, graph)
    proof = read_artifact(design['case'])['cases'][design['case_index']]
    total = _f4_total(design, graph)
    return dict(schema=SCHEMA, schema_version=1, proof=design['case'], proof_case=design['case_index'],
        capture=capture, handoff=handoff, historical_proof=design['historical_proof'],
        original_effect_window=dict(result=proof['result'], reads=proof['original_normal_read_order'],
            effects=proof['normal_window_effects'], required_guards=proof['obligations']),
        f4_total=total, samples=samples, table_versions=sorted(raw['table_versions']),
        table_write_frontiers=design['static_table_write_frontiers'],
        capture_frontiers=raw['observation_frontiers'],
        qualification=dict(table_scope=raw['table_scope'], logical_compacted_snapshot_versions=True,
            original_table_binding_and_row_reference_tested_at_selected_boundaries=True,
            continuous_no_writer_proven=False, full_table_process_constant=False,
            whole_value_body_effects_closed=False, original_tuple_slot_fields_immutable=True,
            future_type_dispatch_proven=False, concurrent_or_external_effects_closed=False,
            transformation_authorized=False, full_entry_complete=False))


def check(path):
    stored = read_artifact(_ref(path, SCHEMA))
    rebuilt = project(stored['handoff']['uri'], stored['capture']['uri'])
    _need(stored == rebuilt, 'published scalar table handoff differs from independent replay')
    return rebuilt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff')
    parser.add_argument('--capture')
    parser.add_argument('--out')
    parser.add_argument('--check')
    args = parser.parse_args()
    if args.check:
        result = check(args.check)
    else:
        if not all((args.handoff, args.capture, args.out)):
            parser.error('supply --handoff, --capture and --out, or --check')
        result = project(args.handoff, args.capture)
        _write(Path(args.out), result)
    print(json.dumps(dict(samples=len(result['samples']), table_versions=len(result['table_versions']),
                          direct_write_frontiers=len(result['table_write_frontiers']))))


if __name__ == '__main__':
    main()
