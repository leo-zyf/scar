"""Retain an observed ATen region in the common planning IR."""
from collections import defaultdict
from copy import deepcopy

from scar.analysis.aten_wrapper_contracts import SCHEMA as WRAPPER_CONTRACTS
from scar.ir.implementation_plan import (
    ImplementationOption, PlanningOperation, PlanningProblem, ResourceObligation,
    validate_problem,
)
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.record_codec import decode, encode
from scar.optimization.implementation_costs import CostQuery, resolve_cost


SCHEMA = 'scar.observed-aten-planning-projection'
_SCOPE = 'ORIGINAL_ATEN_CALL_BOUNDARY'
_FRONTIERS = (
    'GENERATED_WRAPPER_TO_EXECUTABLE_BUILD_RELATION_UNQUALIFIED',
    'WRAPPER_PRE_AND_POST_DISPATCH_EFFECTS_UNQUALIFIED',
    'LOGICAL_CONTENT_AND_READINESS_UNQUALIFIED',
    'NO_COMPARABLE_SOURCE_AND_REPLACEMENT_COSTS',
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _references(rows):
    _require(type(rows) in (list, tuple) and rows
             and all(type(row) is dict and row for row in rows),
             'nonempty original source references required')
    return tuple(deepcopy(rows))


def build_observed_aten_problem(program_refs, target_ref, source, native, *, contracts=None):
    """Project checked F4 and native facts without inferring content or placement.

    The producer must cold-replay source matching, world paths and capture
    membership. This projection checks their shape and retains witnessed order.
    """
    references = _references(program_refs)
    _require(type(target_ref) is ProviderArtifact, 'original typed hardware receipt required')
    _require(native.get('schema') == 'scar.aten-scoped-ports'
             and native.get('schema_version') == 1
             and native.get('transformation_authorized') is False,
             'original observed ATen port facts required')
    calls = source['calls']
    attempts = [row['attempt'] for row in calls]
    _require(attempts and all(type(attempt) is int and attempt > 0 for attempt in attempts)
             and attempts == sorted(set(attempts)),
             'distinct original attempts ordered within an invocation required')
    order = source['world_order']
    _require([(row['before'], row['after']) for row in order] == list(zip(attempts, attempts[1:])),
             'consecutive original F4 world routes must cover the selected region')
    world_refs = {row['after']: _references(row['source_refs']) for row in order}
    by_attempt = defaultdict(list)
    for native_op in native['operators']:
        by_attempt[native_op['call_attempt']].append(native_op)
    _require(set(by_attempt) == set(attempts),
             'selected source attempts and retained provider scopes must match exactly')
    invocation = by_attempt[attempts[0]][0]['invocation']
    _require(invocation and all(op['invocation'] == invocation for op in native['operators'])
             and source.get('invocation', invocation) == invocation,
             'source and native calls need one original invocation')
    if contracts is not None:
        _require(contracts['schema'] == WRAPPER_CONTRACTS and contracts['schema_version'] == 1
                 and (contracts['run'], contracts['process'], contracts['invocation'])
                 == (native['run'], native['process'], invocation)
                 and [row['attempt'] for row in contracts['calls']] == attempts,
                 'wrapper contracts differ from original captured invocation')
    routes = source.get('data_routes', [])
    routes_by_consumer = defaultdict(list)
    for index, route in enumerate(routes):
        before, after = route['before'], route['after']
        _require(type(before) is int and type(after) is int and before < after
                 and before in attempts and after in attempts,
                 'observed route must connect ordered calls in the selected region')
        routes_by_consumer[after].append('#/data_routes/' + str(index))
    coordinates = decode(ProviderArtifact, native['coordinate_binding'])
    references += (encode(ProviderFactReference(coordinates, '#')),)
    operations = []
    for index, call in enumerate(calls):
        attempt = call['attempt']
        children = by_attempt[attempt]
        native_op = children[0]
        _require(all(all(row[key] == native_op[key] for key in ('call_site', 'operand', 'scope'))
                     for row in children), 'provider children need one original attempt scope')
        _require(call.get('site', native_op['call_site']) == native_op['call_site'],
                 'source call and native observation have different original bytecode sites')
        observations = tuple({key: row[key] for key in ('schema', 'eto', 'kineto')} for row in children)
        refs = references + (encode(decode(ProviderFactReference, call['source_call'])),
                             *_references([native_op['operand'], native_op['scope'],
                                           *(row[key] for row in observations for key in ('eto', 'kineto'))]),
                             *world_refs.get(attempt, ()))
        op_id = 'aten-call-attempt:' + str(attempt)
        context = dict(target=encode(target_ref), native_run=native['run'],
                       native_process=native['process'], invocation=invocation,
                       attempt=attempt, provider_observations=observations,
                       provider_revision=native['provider_revision'], source_call=call['source_call'])
        cost = resolve_cost(CostQuery(_SCOPE, context, refs))
        obligations = (
            ResourceObligation('cpu_participation_and_parallelism',
                'Observed host thread identity does not establish call-scoped affinity or library worker demand.', refs),
            ResourceObligation('memory_placement_and_transfers',
                'Observed provider ports do not establish guarded residency, materialization or workspace bounds.', refs),
        )
        parameters = dict(attempt=attempt, provider_observations=observations,
            source_call=call['source_call'], positional=call['positional'], keywords=call['keywords'],
            result=call['result'], world_before=call['world_before'], world_after=call['world_after'],
            observed_route_locators=routes_by_consumer.get(attempt, []),
            effects_admitted=False, replacement_authorized=False)
        if contracts is not None:
            wrapper = contracts['calls'][index]
            _require((wrapper['source_call'], wrapper['site']) == (call['source_call'], native_op['call_site'])
                     and wrapper['effects_complete'] is False and wrapper['replacement_authorized'] is False,
                     'wrapper contract changed original site or replacement authority')
            parameters['wrapper_contract'] = dict(
                locator='#/calls/' + str(index), image_status=wrapper['image_status'],
                argument_status=wrapper['argument_status'], entry_name_agreement=wrapper['entry_name_agreement'],
                effect_requirements=sorted({effect['kind'] for effect in wrapper['effects']}))
        baseline = ImplementationOption('KEEP', 'ORIGINAL_CPYTHON_PROVIDER_CALL', refs,
            {}, {}, {}, {}, cost, parameters, obligations)
        predecessors = () if index == 0 else (operations[-1].id,)
        operations.append(PlanningOperation(op_id, refs, predecessors, (baseline,), baseline.id))
    problem = validate_problem(PlanningProblem(target_ref, references, (), tuple(operations), (),
        assumptions=('ORIGINAL_F4_WORLD_ORDER_RETAINED_ON_OBSERVED_NORMAL_COMPLETION_PATH',
                     'PROVIDER_REFERENCE_EVIDENCE_DOES_NOT_ESTABLISH_LOGICAL_CONTENT_OR_STORAGE_LEASES',
                     'NO_ADMITTED_EXTERNAL_REPLACEMENT_OPTION')))
    report = dict(schema=SCHEMA, schema_version=1, invocation=deepcopy(invocation),
        world_order=[dict(before=row['before'], after=row['after'], source_locator='#/world_order/' + str(index))
                     for index, row in enumerate(order)],
        data_routes=[dict(before=row['before'], after=row['after'], source_locator='#/data_routes/' + str(index))
                     for index, row in enumerate(routes)],
        physical_data_edges_admitted=0,
        operation_ids=[op.id for op in operations], frontiers=list(_FRONTIERS),
        decision='KEEP_ORIGINAL', replacement_authorized=False)
    if contracts is not None:
        report['wrapper_contracts'] = {'artifact_name': 'wrapper-contracts.json',
            'calls': [dict(attempt=row['attempt'], **op.options[0].parameters['wrapper_contract'])
                      for row, op in zip(contracts['calls'], operations, strict=True)]}
    return problem, report


__all__ = ['build_observed_aten_problem']
