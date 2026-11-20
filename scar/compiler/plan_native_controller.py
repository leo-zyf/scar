"""Keep a shared source controller around a replayed CPU update and opaque calls."""
import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path

from scar.analysis.replay_transaction import replay_transaction
from scar.compiler.plan_entry_effects import SCHEMA as EFFECTS, read_entry_effect_plan
from scar.compiler.plan_native_update import SCHEMA as UPDATE, read_native_update_plan
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import SCHEMA as PROBLEM, digest, problem_document, problem_from_dict, validate_problem
from scar.optimization.constructive_planner import construct_implementations
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.native-controller-planning-preparation'
_SOURCE = 'scar.source-native-controller'
_PREFIX = 'WRITE_SCOPE_EXIT_BEFORE_FIRST_REGION_ENTRY'
_SUFFIX = 'LAST_REGION_EXIT_BEFORE_WRITE_SCOPE_ENTRY'


def _require(condition, detail):
    if not condition:
        raise ValueError(detail)


def _reference(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _at(reference, locator='#'):
    return dict(artifact=reference, locator=locator)


def _port_key(port):
    original = port['port']
    return semantic_sha256((port['unit'], original['owner'], original['template']))


def _program_records(block, locator='#/program', *, inside_loop=False):
    """Walk the original nested controller without repeating an iteration."""
    _require(type(block) is list, 'ordered original controller block required')
    for index, row in enumerate(block):
        path = f'{locator}/{index}'
        kind = row['kind']
        yield row, path, inside_loop
        if kind == 'WHILE':
            for branch in ('condition', 'body', 'otherwise'):
                nested = inside_loop or branch == 'body'
                yield from _program_records(row[branch], f'{path}/{branch}', inside_loop=nested)
        elif kind == 'IF':
            for branch in ('then', 'otherwise'):
                yield from _program_records(row[branch], f'{path}/{branch}', inside_loop=inside_loop)


def _field_addresses(controller, records):
    expressions = {row['id']: row for row, _, _ in records if row['kind'] == 'EVAL'}
    parameters = {row['value']: row['name'] for row in controller['parameters']}
    parameter_slots = {row['slot']: row['value'] for row in controller['parameters']}
    reassigned = {row['slot'] for row, _, _ in records if row['kind'] == 'STORE_SLOT'
                  and row['value'] != dict(kind='VALUE', id=parameter_slots.get(row['slot']))}
    definitions = {row['result']: row['result_kinds'] for row, _, _ in records if row['kind'] == 'CALL'}
    definitions.update((row['id'], row['value_kinds']) for row in expressions.values())

    def path(value):
        if value['kind'] != 'VALUE':
            return None
        identity = value['id']
        if identity in parameters:
            return [parameters[identity]]
        expression = expressions.get(identity, {}).get('expression', {})
        if (expression.get('kind') == 'SLOT' and expression['slot'] in parameter_slots
                and expression['slot'] not in reassigned):
            return [parameters[parameter_slots[expression['slot']]]]
        if expression.get('kind') == 'FIELD':
            parent = path(expression['receiver'])
            return None if parent is None else [*parent, expression['field']]
        return None

    fields = []
    for row, locator, inside in records:
        if row['kind'] != 'STORE_FIELD':
            continue
        _require(row['requires_ordinary_field_protocol'] is True and row['source_fact'],
                 'mutable field needs its exact original ordinary write and scalar operand')
        base = path(row['receiver'])
        kinds = definitions.get(row['value'].get('id'), ['RUNTIME_VALUE_KIND_REQUIRED'])
        fields.append(dict(locator=locator, path=None if base is None else [*base, row['field']],
                           inside_loop=inside, stored_value_kinds=kinds))
    return fields, expressions


def _inspect_program(controller, update, reference):
    records = list(_program_records(controller['program']))
    loops = [(row, locator) for row, locator, _ in records if row['kind'] == 'WHILE']
    native = [(row, locator, inside) for row, locator, inside in records
              if row['kind'] == 'CALL' and row['role'] == 'NATIVE_UPDATE']
    _require(len(loops) == len(native) == 1 and native[0][2]
             and native[0][0]['site'] == update['parent']['source_result']
             and native[0][0]['result_kinds'] == ['NUMBER'],
             'controller must retain one shared loop and the original whole update in its body')
    calls = [row for row, _, _ in records if row['kind'] == 'CALL']
    _require(all(row['role'] in ('ORIGINAL_EFFECT', 'NATIVE_UPDATE') for row in calls),
             'every controller call must preserve its synchronous source role')
    effects = [dict(id=row['effect'], site=row['site'], source=_at(reference, locator), inside_loop=inside,
                    disposition='ORIGINAL_SYNCHRONOUS_CALL_AT_ORIGINAL_CONDITIONAL_SITE')
               for row, locator, inside in records if row['kind'] == 'CALL' and row['role'] == 'ORIGINAL_EFFECT']
    identities = {(row['id'], _port_key(row['site'])) for row in effects}
    _require(len(identities) == len(effects)
             and identities == {(row['id'], _port_key(row['site'])) for row in controller['effects']}
             and all(row['effects_closed'] is False for row in controller['effects']),
             'the original effect ledger differs from the retained controller calls')
    field_rows, expressions = _field_addresses(controller, records)
    test = loops[0][0]['test']
    _require(test['kind'] == 'VALUE' and expressions[test['id']]['expression']['kind'] == 'TRUTH',
             'shared loop must preserve its original scalar truth test')
    fields = [dict(row, definition=_at(reference, row['locator'])) for row in field_rows]
    scalar_paths = {tuple(row['parent_path']) for row in update['scalar_inputs']}
    carried_fields = [row for row in fields
                      if row['path'] is not None and tuple(row['path']) in scalar_paths and row['inside_loop']]
    return effects, dict(
        shared_loop=_at(reference, loops[0][1]), shared_body_copies=1, iterations_unrolled=0,
        update_call=_at(reference, native[0][1]), slots=_at(reference, '#/slots'),
        local_scalar_stores=[_at(reference, locator) for row, locator, _ in records if row['kind'] == 'STORE_SLOT'],
        mutable_update_scalar_fields=carried_fields, original_field_stores=fields,
        original_scalar_qualification=_at(reference, '#/qualification'),
        scalar_domain='SOURCE_NUMERIC_EXPRESSIONS_WITH_EXACT_RUNTIME_TYPES_REQUIRED',
        scalar_writes_preserved_before_backedge=True,
    )


def _contains(outer, inner):
    start, end = outer['begin'], outer['end']
    lower, upper = inner['begin'], inner['end']
    return (start['caller']['process_id'] == lower['caller']['process_id']
            and start['native_tid'] == lower['native_tid']
            and start['sequence'] <= lower['sequence'] < upper['sequence'] <= end['sequence'])


def _observed_effects(effects, update, time_ref, time):
    from scar.analysis.source_symbol_variants import _table

    order = time['observed_effects']['order']
    scope_ref = time['observed_effects']['resource_scopes']
    scope = read_artifact(scope_ref)
    directory = Path(scope_ref['uri']).parent
    scopes = {row['id']['wire']: row for row in _table(directory, scope['tables']['scopes'])}
    by_operation = {_port_key(row['site']): row for row in effects}
    update_key = _port_key(update['parent']['source_result'])
    candidates = defaultdict(list)
    for shard in scope['shards'].values():
        for binding in _table(directory / shard['path'], shard['bindings']):
            key = _port_key(binding['anchor']['operation'])
            if key in by_operation or key == update_key:
                candidates[key].append(scopes[binding['scope']['wire']])
    update_scopes = candidates.pop(update_key, [])
    whole_updates = sorted((row for row in update_scopes if 'frame_entry' in row), key=lambda row: row['begin']['sequence'])
    if not whole_updates or whole_updates[0]['frame_entry'] != update['parent']['observations'][0]['entry']:
        raise ValueError('original update scope and child update proof differ at the first observed call')
    first_update = whole_updates[0]
    observed = []
    for index, write in enumerate(order['related_writes']):
        owners = {row['scope']['wire'] for row in write['writes']}
        _require(len(owners) == 1, 'one verified original enclosure required for each write group')
        inner = scopes[owners.pop()]
        _require(inner['frame_entry'] == write['frame_entry'] and inner['frame_exit'] == write['frame_exit']
                 and write['order'] in (_PREFIX, _SUFFIX),
                 'write order differs from the replayed original scope')
        related = [(key, outer) for key, rows in candidates.items() for outer in rows if _contains(outer, inner)]
        if related:
            key, enclosing = max(related, key=lambda pair: pair[1]['begin']['sequence'])
            effect = by_operation[key]['source']
            enclosure = enclosing['id']
        else:
            effect = enclosure = None
        observed.append(dict(
            write_order=_at(time_ref, f'#/observed_effects/order/related_writes/{index}'),
            observed_side=write['order'], original_controller_call=effect, original_call_enclosure=enclosure,
            publication_before_first_whole_update=(write['order'] == _PREFIX
                and inner['frame_exit']['path'] == first_update['frame_entry']['path']
                and inner['frame_exit']['index'] < first_update['frame_entry']['index']),
            unchanged_nested_call_order_required=True,
        ))
    read_artifact(scope_ref)
    return observed, len(whole_updates)


@replay_transaction()
def _derive(source_ref, update_ref, effects_ref):
    from scar.analysis.source_native_controller import check

    controller = check(source_ref['uri'])
    update_report, update_problem = read_native_update_plan(update_ref['uri'])
    effect_report, effect_problem = read_entry_effect_plan(effects_ref['uri'])
    _require(controller == read_artifact(source_ref) and update_report == read_artifact(update_ref)
             and effect_report == read_artifact(effects_ref), 'controller inputs changed while replaying')
    _require(controller['update_source'] == update_report['source'],
             'controller and selection need the same original update source')
    update = read_artifact(update_report['source'])
    time_ref = read_artifact(effect_report['resources'])['time_schedule']
    time = read_artifact(time_ref)
    _require(update['ordered_graph'] == effect_report.get('loop_content')
             and update_report['numeric_recipe'] == time['source']
             and update_problem.target == effect_problem.target,
             'update, loop effects and controller must share source, capture and hardware')
    _require(controller['effect_closure_proven'] is False
             and time['observed_effects']['order']['complete_effects_proven'] is False,
             'controller envelope expects the observed opaque effect contract')
    effects, loop = _inspect_program(controller, update, source_ref)
    observed, update_scope_count = _observed_effects(effects, update, time_ref, time)
    scope = _at(source_ref, '#/program')
    run = _at(update_ref, '#/problem')
    order = _at(effects_ref, '#/problem')
    mode, = (op.options[0] for op in update_problem.operations)
    region = effect_report['replay']['region_operation']
    operations = []
    for row in effect_problem.operations:
        if row.id == region:
            original, = row.options
            row = replace(row, source_refs=(*row.source_refs, scope, run), options=(replace(
                original, proof_refs=(*original.proof_refs, scope, run),
                demands=mode.demands, workspace_bytes=mode.workspace_bytes,
                parameters=dict(original.parameters, shared_controller=scope, child_update=run),
            ),))
        operations.append(row)
    problem = validate_problem(replace(
        effect_problem, source_refs=(*effect_problem.source_refs, scope, run, order),
        operations=tuple(operations), resources=effect_problem.resources + update_problem.resources,
        reserved_memory=effect_problem.reserved_memory | update_problem.reserved_memory,
        assumptions=effect_problem.assumptions + (
            'OPAQUE_SOURCE_CALLS_KEEP_ORIGINAL_SYNCHRONOUS_CONTROL_AND_STATE_BOUNDARIES',
            'ONE_SHARED_LOOP_TEMPLATE_WITH_MUTABLE_SCALAR_AND_ARRAY_STATE',
            'SCOPED_ARRAY_CAPACITY_NOT_OPAQUE_EFFECT_MEMORY_OR_PROCESS_RSS',
        ),
    ))
    plan = construct_implementations(problem, baseline_only=True)
    return problem, dict(
        controller=scope, update=run, original_opaque_calls=effects,
        runtime_effect_mode='ORIGINAL_SYNCHRONOUS_EFFECT_PROVIDER_REQUIRED',
        loop=loop, observed_publications=observed,
        observed_whole_update_scopes=update_scope_count,
        observed_numeric_occurrences=effect_report['replay']['observed_region_occurrences'],
        original_timing_constraints=len(problem.timing_constraints),
        array_lifetime=dict(
            observation=_at(update_report['source'], '#/allocation_lifetime'),
            scoped_payload=update_report['replay']['original_array_payload'],
            numeric_cpu_ids=update_report['replay']['numeric_cpu_ids'],
            opaque_effects_extend_no_array_identity_claim=False,
            complete_outer_lifetime_proven=False,
        ),
        first_snapshot_replay=dict(
            disposition='BLOCKED_OPEN_ORIGINAL_EFFECTS',
            original_call_retained=True, byte_oracle_confers_rewrite_authority=False,
            preceding_update_publications=[row['write_order'] for row in observed if row['observed_side'] == _PREFIX],
            reasons=['NO_CLOSED_ORIGINAL_PUBLICATION_READ_WRITE_AND_ESCAPE_CONTRACT',
                     'OBSERVED_NORMAL_ORDER_DOES_NOT_QUALIFY_REPLAY_OF_OPAQUE_WORLD_EFFECTS'],
        ),
        original_effect_frontiers=effect_report['replay']['order_frontiers'],
        source_frontiers=controller['remaining'], planner_status=plan['status'],
        default_backend=plan['default_backend'], makespan_ns=plan['makespan_ns'],
        complete_entry_cost='UNKNOWN', complete_entry_authority=False, deployment_authorized=False,
    )


def _write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return _reference(path, value['schema'])


def _report(source, update_plan, effect_plan, problem_ref, problem, replay):
    return dict(schema=SCHEMA, schema_version=1, source=source, update_plan=update_plan,
                effect_plan=effect_plan, problem=problem_ref, problem_sha256=digest(problem),
                replay=replay, target_program_executed=False, performance_measured=False,
                deployment_authorized=False)


def prepare_native_controller_plan(source, update_plan, effect_plan, out):
    refs = (_reference(source, _SOURCE), _reference(update_plan, UPDATE), _reference(effect_plan, EFFECTS))
    problem, replay = _derive(*refs)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    problem_ref = _write(out / 'problem.json', problem_document(problem))
    report = _report(*refs, problem_ref, problem, replay)
    _write(out / 'result.json', report)
    return report


def read_native_controller_plan(path):
    reference = _reference(path, SCHEMA)
    report = read_artifact(reference)
    refs = tuple(report[key] for key in ('source', 'update_plan', 'effect_plan'))
    problem, replay = _derive(*refs)
    problem_ref = _reference(Path(reference['uri']).with_name('problem.json'), PROBLEM)
    if (report != _report(*refs, problem_ref, problem, replay)
            or problem_from_dict(read_artifact(problem_ref)) != problem):
        raise ValueError('controller planning differs from original source/update/effect replay')
    read_artifact(reference)
    return report, problem


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--source')
    inputs.add_argument('--check')
    parser.add_argument('--update-plan')
    parser.add_argument('--effect-plan')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check:
        if args.update_plan or args.effect_plan or args.out:
            parser.error('--check only consumes its preserved planning receipt')
        report, _ = read_native_controller_plan(args.check)
    else:
        if not args.update_plan or not args.effect_plan or not args.out:
            parser.error('plan preparation requires both source-matched child plans and --out')
        report = prepare_native_controller_plan(args.source, args.update_plan, args.effect_plan, args.out)
    row = report['replay']
    print(json.dumps(dict(
        planner_status=row['planner_status'], default_backend=row['default_backend'],
        original_opaque_calls=len(row['original_opaque_calls']),
        observed_publications=len(row['observed_publications']),
        mutable_scalar_fields=len(row['loop']['mutable_update_scalar_fields']),
        first_snapshot_replay=row['first_snapshot_replay']['disposition'],
        deployment_authorized=False,
    ), sort_keys=True))


if __name__ == '__main__':
    main()
