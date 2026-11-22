"""Bind bounded full scalar returns and selected table versions to their F4 body."""

import argparse
import ast
from collections import Counter
from dis import get_instructions
import json
from pathlib import Path

from scar.analysis import scalar_table_cse, scalar_table_handoff
from scar.analysis.program_view import ProgramView
from scar.analysis.retained_code_bodies import RetainedCodeBodyIndex
from scar.analysis.static.lowering.source_call_lowering import _body
from scar.analysis.source_references import source_port
from scar.runtime.source_value_observe import _codes
from scar.ir.execution_recipe import read_artifact
from scar.ir.record_codec import encode
from scar.trace.code_snapshot import describe_code, site

SCHEMA = 'scar.scalar-score-evidence'
CAPTURE = 'scar.scalar-score-capture'
DESIGN = 'scar.scalar-score-selection'


def _need(valid, reason):
    if not valid:
        raise ValueError('scalar score: ' + reason)


def _one(rows, reason):
    rows = tuple(rows)
    _need(len(rows) == 1, reason)
    return rows[0]


def _position(node):
    return [node.lineno, node.end_lineno, node.col_offset, node.end_col_offset]


def _within(position, span):
    return position is not None and (span[0], span[2]) <= (position[0], position[2]) \
        and (position[1], position[3]) <= (span[1], span[3])


def _entry(instructions, node):
    span = _position(node)
    offset = next((offset for offset, position in instructions if _within(position, span)), None)
    _need(offset is not None, 'source/test arm lacks a current executable position')
    return offset


def _exact_code(source, descriptor):
    raw = read_artifact(source, json_document=False)
    code = _one((candidate for candidate in _codes(compile(raw, source['uri'], 'exec', dont_inherit=True, optimize=0))
                if describe_code(candidate) == descriptor), 'retained source code descriptor must be exact')
    return raw, code


def _design(handoff, limits):
    previous = read_artifact(handoff)
    scope = read_artifact(previous['capture'])
    selected = read_artifact(scope['selection'])
    proof, ordinal = previous['proof'], previous['proof_case']
    current = scalar_table_cse.check(proof)
    case = current['cases'][ordinal]
    source, descriptor, table = case['source'], selected['value_code'], selected['table']
    _need(source == scope['source'] == selected['source'] and descriptor['key'] == case['code_key'],
          'full body and retained scalar read case belong to different code')
    raw, code = _exact_code(source, descriptor)
    from scar.compiler.scalar_table_region import _definition

    definition = _definition(ast.parse(raw, filename=source['uri']), code.co_qualname)
    _need(code.co_argcount == 2 and code.co_kwonlyargcount == 0 and len(descriptor['parameters']) == 2,
          'scope requires exact ordinary receiver and tuple argument')
    receiver, argument = descriptor['parameters']
    instructions = [(op.offset, site(code, op.offset)['position']) for op in get_instructions(code)]
    branches = []
    for node in sorted((node for node in ast.walk(definition) if isinstance(node, ast.If)), key=_position):
        line, _, column, _ = _position(node)
        branches.append(dict(id=f'{line}:{column}', span=_position(node), test_span=_position(node.test),
            test_offset=_entry(instructions, node.test), then_offset=_entry(instructions, node.body[0]),
            else_offset=None if not node.orelse else _entry(instructions, node.orelse[0])))
    _need(branches and len({row['id'] for row in branches}) == len(branches),
          'bounded ordinary conditional arms expected')
    fields = sorted({node.attr for node in ast.walk(definition) if isinstance(node, ast.Attribute)
                     and isinstance(node.value, ast.Name) and node.value.id == receiver})
    _need(fields, 'selected receiver supplies no original data fields')
    globals_ = sorted({op.argval for op in get_instructions(code) if op.opname == 'LOAD_GLOBAL'} - {table})
    per_version, attempts, max_versions = (limits[key] for key in ('per_version', 'attempts_per_version', 'max_versions'))
    _need(all(type(limit) is int for limit in (per_version, attempts, max_versions))
          and 1 <= per_version <= 64 and per_version <= attempts <= 1024 and 1 <= max_versions <= 4,
          'bounded score observation budget required')
    return dict(schema=DESIGN, schema_version=1, source=source, case=proof, case_index=ordinal, handoff=handoff,
        code=descriptor, receiver=receiver, argument=argument, receiver_fields=fields, table=table,
        global_loads=globals_, branches=branches, limits=limits,
        original_effect_frontiers=previous['table_write_frontiers'])


def prepare(handoff, *, per_version=24, attempts_per_version=192, max_versions=2):
    """Prepare a local score-body selector from existing F4 and source inputs."""
    limits = dict(per_version=per_version, attempts_per_version=attempts_per_version, max_versions=max_versions)
    return _design(scalar_table_handoff._ref(handoff, scalar_table_handoff.SCHEMA), limits)


def _validate_samples(raw, design):
    versions = raw['table_versions']
    _need(0 < len(versions) <= design['limits']['max_versions'], 'expected bounded exact table versions')
    i64 = scalar_table_handoff._int64
    for digest, row in versions.items():
        rows = row['rows']
        _need(type(rows) is dict and 0 < len(rows) <= 64 and
              all(type(key) is str and key.isascii() and type(values) is list and values and all(i64(v) for v in values)
                  for key, values in rows.items()) and sum(map(len, rows.values())) <= 4096 and
              scalar_table_handoff.table_version(rows) == digest and type(row['root_token']) is int and row['root_token'] > 0,
              'table version bytes or scalar ABI differ')
    global_names, builtins_ = raw['globals'], raw['builtins']
    _need(type(global_names) is dict and all(i64(value) for value in global_names.values()) and
          type(builtins_) is dict and all(role == 'CPYTHON_EXACT_BUILTIN_ABS' for role in builtins_.values()) and
          bool(builtins_) is raw['builtin_abs_verified'] and not global_names.keys() & builtins_.keys() and
          set(global_names) | set(builtins_) == set(design['global_loads']),
          'selected global bindings or builtin identities are unavailable')
    _, code = _exact_code(design['source'], design['code'])
    branches = {row['id']: row for row in design['branches']}
    counts, arms = Counter(), {version: Counter() for version in versions}
    keys = set()
    for sample in raw['samples']:
        version = sample['table_version']
        thread, ordinal = sample['thread_id'], sample['frame_ordinal']
        returned = sample['return_site']
        _need(version in versions and sample['root_token'] == versions[version]['root_token'] and
              type(thread) is int and thread > 0 and type(ordinal) is int and ordinal > 0 and (thread, ordinal) not in keys
              and type(sample['entry_site']['code']) is str and sample['entry_site']['opname'] in ('CALL', 'CALL_KW')
              and i64(sample['result']) and returned['opname'] in ('RETURN_VALUE', 'RETURN_CONST')
              and returned == site(code, returned['offset']), 'source scope or normal return site differs')
        keys.add((thread, ordinal))
        values = sample['receiver_fields']
        argument = sample['argument']
        _need(set(values) == set(design['receiver_fields']) and all(i64(v) or type(v) is str and v.isascii()
              for v in values.values()) and type(argument) is list and len(argument) == 3 and
              i64(argument[0]) and i64(argument[1]) and type(argument[2]) is str and argument[2].isascii() and
              sample['move'] == argument, 'exact source receiver/tuple argument fields required')
        outcomes = sample['branch_arms']
        _need([row['id'] for row in outcomes] == list(branches) and all(
              row['arm'] in ('THEN', 'ELSE', 'FALSE', 'NOT_REACHED') and
              (row['arm'] != 'ELSE' or branches[row['id']]['else_offset'] is not None) and
              (row['arm'] != 'FALSE' or branches[row['id']]['else_offset'] is None)
              for row in outcomes), 'source conditional arms differ from the selector')
        counts[version] += 1
        arms[version].update((row['id'], row['arm']) for row in outcomes)
    _need(set(counts) == set(versions) and all(0 < count <= design['limits']['per_version'] for count in counts.values()),
          'sampled versions require bounded normal returns')
    return counts, {version: [
        dict(id=identity, arm=arm, count=count) for (identity, arm), count in sorted(outcomes.items())]
        for version, outcomes in arms.items()}


def project(capture_path):
    capture = scalar_table_handoff._ref(capture_path, CAPTURE)
    raw = read_artifact(capture)
    selection = read_artifact(raw['selection'])
    design = _design(selection['handoff'], selection['limits'])
    launch = raw['launch']
    _need(raw['schema_version'] == 1 and selection == design and raw['source'] == design['source'] and
          raw['case'] == design['case'] and raw['case_index'] == design['case_index'] and
          type(raw['process_id']) is int and raw['process_id'] > 0 and raw['original_execution'] is True
          and launch['decision'] == 'NO_OP' and launch['entry_path'] == design['source']['uri'] and launch['exit_code'] == 0,
          'complete original score capture and exact source proof required')
    counts, branches = _validate_samples(raw, design)
    previous = read_artifact(design['handoff'])
    prior_graph = read_artifact(previous['handoff'])
    catalog = prior_graph['catalog']
    read_artifact(catalog)
    view = ProgramView(Path(catalog['uri']).parent)
    correspondence = RetainedCodeBodyIndex(view)
    body, function_fact, facts, function = _body(correspondence, design['code'])
    formals = [encode(source_port(facts, ordinal)) for ordinal in function.formals[1:-1]]
    returned = encode(source_port(facts, function.returns))
    _need(encode(body) == previous['f4_total']['value_body']
          and function_fact == previous['f4_total']['value_fact']
          and returned == previous['f4_total']['value_return'] and len(formals) == 2,
          'selected normal return or formals differ from the original F4 body')
    correspondence.check()
    return dict(schema=SCHEMA, schema_version=1, source=design['source'], proof=design['case'],
        proof_case=design['case_index'], handoff=design['handoff'], capture=capture, catalog=catalog,
        callee=dict(code=design['code'], body=encode(body), function_fact=function_fact,
            receiver=formals[0], argument=formals[1], normal_return=returned),
        tables=sorted(counts), sample_counts=dict(counts), branch_outcomes=branches,
        state_frontiers=design['original_effect_frontiers'], capture_frontiers=raw['observation_frontiers'],
        qualification=dict(same_process_table_versions=True, full_original_normal_return_observed=True,
            selected_receiver_fields_and_builtin_roles_observed=True, effects_closed=False,
            continuous_no_writer_proven=False, future_dispatch_proven=False,
            transformation_authorized=False, target_executed_by_projection=False))


def check(path):
    stored = read_artifact(scalar_table_handoff._ref(path, SCHEMA))
    read_artifact(stored['capture'])
    rebuilt = project(stored['capture']['uri'])
    _need(rebuilt == stored, 'published score evidence differs from cold replay')
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
    elif args.prepare:
        value = prepare(args.prepare)
    else:
        value = project(args.capture)
    if args.check is None:
        if args.out is None:
            parser.error('publication requires --out')
        print(json.dumps(scalar_table_handoff._write(args.out, value)))
    else:
        print(json.dumps(dict(samples=value['sample_counts'], tables=value['tables'])))


if __name__ == '__main__':
    main()
