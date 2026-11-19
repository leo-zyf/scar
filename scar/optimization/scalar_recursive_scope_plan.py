"""Keep stateful recursive calls on an owner lane and price its open cuts."""

import argparse
import ast
import json
from pathlib import Path

from scar.analysis import scalar_recursive_region
from scar.compiler.scalar_table_region import _definition
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import scalar_sorted_transition_cost as entry_cost


SCHEMA = 'scar.scalar-recursive-scope-plan'


def _require(condition, reason):
    if not condition:
        raise ValueError('recursive placement: ' + reason)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _span(node):
    return [node.lineno, node.end_lineno, node.col_offset, node.end_col_offset]


def _same_expression(left, right):
    return ast.unparse(left) == ast.unparse(right)


def _evictions(body, receiver):
    """Recognize the first-key deletion forms, without presuming builtin types."""
    result = []
    for node in ast.walk(body):
        match node:
            case ast.If(test=ast.Compare(left=ast.Call(func=ast.Name(id='len'), args=[sized], keywords=[]),
                                         ops=[ast.Gt()], comparators=[capacity]),
                        body=[ast.Delete(targets=[ast.Subscript(value=table, slice=ast.Call(
                            func=ast.Name(id='next'), args=[selection], keywords=[]))])], orelse=[]):
                if not _same_expression(sized, table):
                    continue
            case _:
                continue
        if not isinstance(table, ast.Attribute) or ast.unparse(table.value) != receiver:
            continue
        excluded = None
        match selection:
            case ast.Call(func=ast.Name(id='iter'), args=[source], keywords=[]):
                pass
            case ast.GeneratorExp(elt=ast.Name(id=yielded), generators=[ast.comprehension(
                    target=ast.Name(id=key), iter=source, ifs=[ast.Compare(left=ast.Name(id=checked),
                    ops=[ast.NotEq()], comparators=[ast.Attribute(value=ast.Name(id=owner), attr=excluded)])],
                    is_async=0)]) if yielded == key == checked and owner == receiver:
                pass
            case _:
                continue
        if _same_expression(source, table):
            result.append(dict(field=table.attr, guard_span=_span(node.test), delete_span=_span(node.body[0].targets[0]),
                capacity_expression=ast.unparse(capacity), condition='SIZE_STRICTLY_GREATER_THAN_CAPACITY',
                selection='FIRST_ELIGIBLE_KEY_IN_INSERTION_ORDER', excluded_receiver_field=excluded,
                existing_key_update='RETAIN_ORIGINAL_INSERTION_POSITION', empty_selection='PRESERVE_STOP_ITERATION',
                requires=['EXACT_BUILTIN_LENGTH_AND_ITERATION', 'EXACT_KEY_EQUALITY_AND_ORDER', 'VERSIONED_CAPACITY']))
    return sorted(result, key=lambda row: row['field'])


def _clock_contract(body, receiver, effects):
    for node in body.body:
        match node:
            case ast.If(test=ast.BoolOp(op=ast.And(), values=[
                    ast.Compare(left=ast.BinOp(left=ast.Attribute(value=ast.Name(id=counter_owner), attr=counter),
                        op=ast.Mod(), right=ast.Constant(value=period)), ops=[ast.Eq()],
                        comparators=[ast.Constant(value=0 as zero)]),
                    ast.Compare(left=ast.Call() as query, ops=[ast.Gt()], comparators=[
                        ast.Attribute(value=ast.Name(id=limit_owner), attr=limit)])]),
                    body=[ast.Raise(exc=ast.Name(id=stop), cause=None)], orelse=[]) \
                    if counter_owner == limit_owner == receiver and type(zero) is type(period) is int and period > 0:
                pass
            case _:
                continue
        # The first two statements below the docstring must be the counter increment and clock check.
        active = body.body[1:] if ast.get_docstring(body) is not None else body.body
        if len(active) < 2 or active[1] is not node:
            continue
        match active[0]:
            case ast.AugAssign(target=ast.Attribute(value=ast.Name(id=owner), attr=field),
                               op=ast.Add(), value=ast.Constant(value=1 as one)) \
                    if (owner, field) == (receiver, counter) and type(one) is int:
                pass
            case _:
                continue
        effect = next((row for row in effects if row['source_span'] == _span(query)
                       and (row.get('import_module'), row.get('member')) == ('time', 'time')), None)
        if effect is not None:
            return dict(source=effect['original_call'], counter_field=counter, increment_before_check=1,
                increment_span=_span(active[0]), period=period, deadline_field=limit, comparison='STRICTLY_GREATER',
                source_clock='PYTHON_TIME_TIME', exception=stop, exception_span=_span(node.body[0]),
                timing_semantics='LIVE_REALTIME_AND_EXCEPTION_PREFIX', observed_deadline_traces=None)
    return None


def _calls(graph, owner, formals, receiver):
    calls = []
    for row in graph['recursive_back_edges']:
        source = row['original_call']
        actuals = row['actuals']
        defaults = row['omitted_defaults']
        _require(row['target_candidate'] == owner and source['compiled']['code'] == owner['key']
            and row['receiver'] == dict(symbol=receiver, binding='SAME_LEXICAL_RECEIVER')
            and [x['formal'] for x in actuals + defaults] == formals[1:]
            and row['continuation'] == 'RETURN_OR_EXCEPTION_TO_THIS_CALL_SITE'
            and row['world_preservation'] == 'CHILD_EFFECT_WORLD_RETURNS_BEFORE_OWNER_RESUMES',
            'recursive call cannot be mapped to a balanced same-owner activation')
        calls.append(dict(site=source['compiled'], source_fact=source['fact'], condition=row['control'], actuals=actuals,
            defaults=defaults, effect_ports=dict(before=source['world_in'], after=source['world_out']),
            dispatch=row['dynamic_target']))
    _require(calls and len({row['site']['offset'] for row in calls}) == len(calls), 'missing distinct recursive sites')
    return calls


def _scope(region):
    graph, spec = region['graph'], region['region']
    source = read_artifact(region['source'], json_document=False)
    body = _definition(ast.parse(source, filename=region['source']['uri']), spec['owner']['qualname'])
    calls = _calls(graph, spec['owner'], spec['formals'], spec['receiver'])
    evictions = _evictions(body, spec['receiver'])
    state = {row['field']: row['effects'] for row in graph['receiver_state']}
    deletions = {(name, tuple(effect['source_span'])) for name, effects in state.items()
                 for effect in effects if effect['effect'] == 'ITEM_DELETE'}
    _require(deletions == {(row['field'], tuple(row['delete_span'])) for row in evictions},
        'state deletion frontier needs an explicit ordered container contract')
    clock = _clock_contract(body, spec['receiver'], graph['external_or_unclosed_effects'])
    fields = set(state)
    fields.update(row['excluded_receiver_field'] for row in evictions if row['excluded_receiver_field'] is not None)
    # Nested operations were deliberately left open by the source projector.
    nested = [row for row in graph['structured_regions'] if row['kind'] in
              ('FunctionDef', 'AsyncFunctionDef', 'Lambda', 'GeneratorExp', 'ListComp', 'SetComp', 'DictComp')]
    return dict(owner=spec['owner'], recursive_calls=calls, ordered_state_reads_and_writes=state,
        ordered_container_evictions=evictions, clock=clock, exits=graph['exits'],
        external_effects=graph['external_or_unclosed_effects'], nested_callable_frontier=nested,
        abi=dict(language='C++20', positional_formals=spec['formals'], return_ports=['VALUE', 'EXCEPTION'],
            receiver=spec['receiver'], receiver_fields_observed_or_referenced=sorted(fields),
            receiver_ownership='ONE_EXCLUSIVE_EPOCH_SHARED_BY_ALL_ACTIVATIONS',
            activation='PRIVATE_FRAME_WITH_BALANCED_RETURN_AND_EXCEPTION_CONTINUATIONS',
            publish='PRESERVE_SOURCE_ORDERED_STATE_AT_EACH_OBSERVABLE_EXIT',
            fallback='ORIGINAL_BEFORE_FIRST_EFFECT_OR_FROM_EXACT_COMMITTED_CONTINUATION',
            layout_numeric_ranges_and_allocator_capacity='UNRESOLVED'))


def _unknown(*components):
    return dict(status='UNKNOWN', estimate_ns=None, components=list(components))


def propose(recursive_region_path, selection_path, historical_path):
    """Produce a same-owner compile experiment with measured, distinct alternatives."""
    refs = dict(recursive_region=_ref(recursive_region_path, scalar_recursive_region.SCHEMA),
        entry_selection=_ref(selection_path, entry_cost.SCHEMA), historical=_ref(historical_path, entry_cost.HISTORY))
    region = scalar_recursive_region.check(refs['recursive_region']['uri'])
    matched = entry_cost.check(refs['entry_selection']['uri'])
    historical = entry_cost.check(refs['historical']['uri'])
    _require(region['source'] == matched['source'] == historical['source'], 'entry evidence belongs to different source')
    _require(region['contract']['recursive_shared_state_versions_require_order'], 'missing recursive state order')
    scope = _scope(region)
    closed_clock = scope['clock'] is not None and all(row.get('original_call') == scope['clock']['source']
        for row in scope['external_effects'])
    obligations = [
        dict(phase='FRONTEND', status='UNRESOLVED', subject='DYNAMIC_CALL_AND_FIELD_DISPATCH',
            acceptance='Guard every recursive target, receiver field, nested callable and global binding for the epoch.'),
        dict(phase='FRONTEND', status='UNRESOLVED', subject='EXCLUSIVE_RECEIVER_AND_ALIASES',
            acceptance='Prove exclusive ownership or retain each external alias, escape and observer at its access order.'),
        dict(phase='IR', status='PROVEN_ORDER_REQUIREMENT', subject='SERIAL_STATE_AND_BALANCED_CONTINUATIONS',
            acceptance='Pass the child effect state before resuming its parent and keep local values in each activation.'),
        dict(phase='IR', status='UNRESOLVED', subject='REALTIME_AND_EXCEPTION_FRONTIER',
            acceptance='Keep the source clock check and matching committed prefix on Stop or other reachable exceptions.'),
        dict(phase='BACKEND', status='LOWERING_CHECK', subject='VALUE_AND_CONTAINER_REPRESENTATION',
            acceptance='Check numeric ranges, equality and insertion order, overwrite, deletion and receiver capacity.'),
        dict(phase='BACKEND', status='LOWERING_CHECK', subject='ORDERED_CONTROL_AND_EFFECTS',
            acceptance='Prove ordered yields, short circuit, early returns, unwind, state publication and fallback.'),
        dict(phase='VALIDATION', status='UNRESOLVED', subject='SAME_BOUNDARY_CORRECTNESS_AND_COST',
            acceptance='Compare value, state and effect traces before matched clean whole-entry samples including all bridge costs.'),
    ]
    measured = {arm: dict(status='MATCHED_CLEAN_ENTRY_ENVELOPE', **values)
                for arm, values in matched['measurements'].items()}
    measured['recursive_region_cpp'] = _unknown('outer_entry_and_value_marshalling', 'guards_and_fallback',
        'receiver_epoch_enter_and_exit', 'native_recursive_service', 'remaining_clock_and_callback_cuts',
        'unwind_and_state_publication', 'outside_region_execution')
    observed = region['evidence']
    return dict(schema=SCHEMA, schema_version=1, artifacts=refs, source=region['source'], scope=scope,
        placement=dict(cpu_ids=matched['cpu_ids'], worker_count=1, scope='MATCHED_OBSERVATION_CPU_MASK_ONLY',
            recursive_frame_storage='OWNER_STACK_OR_EQUIVALENT_PRIVATE_FRAME_STORE',
            shared_receiver_storage='OWNER_MEMORY_IF_EXCLUSIVE_EPOCH_PROVEN',
            independent_recursive_tasks=False, maximum_live_frames=None, receiver_storage_bytes=None),
        costs=dict(coverage='MATCHED_WHOLE_ENTRY_OBSERVATIONS_ONLY', arms=measured,
            matched_launch=dict(stdin=matched['stdin'], depth=matched['depth'], positions=matched['positions'],
                argv=matched['argv'], exact_canonical_stdout_sha256=matched['exact_canonical_stdout_sha256'],
                clean_rounds=matched['measured_pairs'], warmup_rounds=matched['warmup_pairs']),
            boundary_work=dict(outer_entries=None, native_recursive_visits=None, external_calls=None,
                cardinality_guard='COUNT_AT_SAME_RUNTIME_SCOPE_BEFORE_PRICING',
                same_owner_recursive_crossing='INTERNAL_IF_ALL_DYNAMIC_DISPATCH_OBLIGATIONS_CLOSE',
                remaining_effects='CHARGE_AT_ACTUAL_CLOCK_AND_CALLBACK_FRONTIERS'),
            prior_boundary_evidence=dict(recursive_transfer_samples=observed['recursive_actual_samples'],
                dictionary_read_write_pairs=observed['observed_original_dictionary_pairs'],
                measured_small_region_choice=matched['selected'],
                latest_fused_instrumentation=matched['instrumented_boundaries'].get('fused', []),
                historical_calls={name:row['diagnostics']['native_calls'] for name, row in
                                  historical['historical_search'].items()},
                can_transfer_to_recursive_cost=False)),
        obligations=obligations,
        decision=dict(selected='RETAIN_ORIGINAL_SOURCE', experimental_nomination='COLOCATE_RECURSIVE_OWNER_REGION_IN_CPP',
            status='COMPILE_EXPERIMENT_ONLY', implementation_verified=False, deployment_authorized=False,
            clock_contract_identified=closed_clock, clock_replay_discharged=False,
            reason='Shared serial state supports the same-owner scope. Runtime effects and full-boundary native cost remain open.'))


def check(path):
    stored = read_artifact(_ref(path, SCHEMA))
    refs = stored['artifacts']
    replay = propose(refs['recursive_region']['uri'], refs['entry_selection']['uri'], refs['historical']['uri'])
    _require(json.dumps(stored, sort_keys=True, allow_nan=False) == json.dumps(replay, sort_keys=True, allow_nan=False),
        'recursive placement differs from cold replay')
    return replay


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--region', type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--history', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args(argv)
    if args.check is not None and not any((args.region, args.selection, args.history, args.out)):
        result = check(args.check)
    elif args.check is None and all((args.region, args.selection, args.history, args.out)):
        result = propose(args.region, args.selection, args.history)
        with args.out.open('x') as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    else:
        parser.error('supply --check, or --region, --selection, --history and --out')
    print(json.dumps(result['decision'], sort_keys=True))


if __name__ == '__main__':
    main()
