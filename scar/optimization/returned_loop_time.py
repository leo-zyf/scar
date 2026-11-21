"""Project source-bound returned-value recurrence into the existing time graph."""

import argparse
import json
from pathlib import Path

from scar.analysis import returned_loop_binding as binding
from scar.analysis.temporal_resources import derive_temporal_resources
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.temporal_resources import projection_document
from scar.ir.versioned_schedule import (
    TemporalActivity, TemporalConstraint, TemporalConstraintKind as Constraint,
    TemporalEvent, TemporalEventKind as Event, TemporalIterationSelection,
    TemporalIterationSelector as Select, TemporalMode, TemporalModel,
    TemporalOccupancy, TemporalOccupancyKind, TemporalPathCondition,
    TemporalScope, TemporalScopeKind, VersionedResourceSchedule,
    VersionedTask, VersionedTaskKind, schedule_document, validate_versioned_schedule,
)


SCHEMA = 'scar.returned-loop-time'
_ROOT = '$CALL_INSTANCE'
_LOOP = 'returned-value-loop'
_RESOURCE = 'original-caller-cpu-domain'


def _loc(reference, pointer):
    return dict(artifact=reference, locator=pointer)


def _unique(rows, fields):
    key = tuple(rows[0][field] for field in fields)
    if any(tuple(row[field] for field in fields) != key for row in rows):
        raise ValueError('selected activations do not share their original boundary, count and CPU domain')
    return rows[0]


def _schedule(bound, reference):
    matches = bound['matches']
    match = _unique(matches, ('source_carry', 'consumer', 'producer', 'loop_count',
                               'observation_cpu_ids', 'call_code_id', 'getter_code_id'))
    if any(row['observed_getter_site'] is None or not row['source_receiver_call_linked']
           for row in matches):
        raise ValueError('resource projection requires the actual getter coordinates and original receiver route')
    if any(row['loop_count'] < 2 for row in matches):
        raise ValueError('positive returned-value recurrence required')
    ports = (match['consumer'], match['producer'])
    refs = tuple(_loc(reference, f'#/matches/{index}') for index in range(len(matches)))
    tasks = (
        VersionedTask('invoke', VersionedTaskKind.SOURCE_OPERATION, source_refs=(ports[0], *refs),
                      attrs={'code_id': match['call_code_id'], 'effects': 'UNKNOWN'}),
        VersionedTask('read-returned-value', VersionedTaskKind.SOURCE_OPERATION,
                      depends_on=('invoke',), source_refs=(ports[1], *refs),
                      attrs={'code_id': match['getter_code_id'], 'effects': 'UNKNOWN'}),
    )
    events = [
        TemporalEvent('entry', Event.SCOPE_ENTRY, _ROOT),
        TemporalEvent('exit', Event.SCOPE_EXIT, _ROOT),
        TemporalEvent('iteration-entry', Event.SCOPE_ENTRY, _LOOP),
        TemporalEvent('iteration-exit', Event.SCOPE_EXIT, _LOOP),
        TemporalEvent('returned-value-ready', Event.READY, _LOOP, 'read-returned-value', source_refs=refs),
    ]
    activities = []
    occupancies = []
    for task in tasks:
        start = task.id + ':start'
        end = task.id + ':complete'
        ready_events = ('returned-value-ready',) if task.id == 'read-returned-value' else ()
        events.extend((
            TemporalEvent(start, Event.START, _LOOP, task.id, source_refs=task.source_refs),
            TemporalEvent(end, Event.COMPLETION, _LOOP, task.id, source_refs=task.source_refs),
        ))
        activities.append(TemporalActivity(task.id, task.id, _LOOP, start, end,
                                           ready_events=ready_events, source_refs=task.source_refs))
        occupancies.append(TemporalOccupancy(task.id + ':cpu', TemporalOccupancyKind.EXECUTION_DEMAND,
                                              _RESOURCE, start, end, _LOOP,
                                              activity=task.id, source_refs=task.source_refs))
    original = match['source_carry']
    initial = dict(original, locator=original['locator'] + '/initial')
    backedge = dict(original, locator=original['locator'] + '/backedges')
    exit_ = dict(original, locator=original['locator'] + '/normal_exits')
    constraints = (
        TemporalConstraint('initial-control', 'entry', 'invoke:start', Constraint.CAUSAL,
                           after_iteration=TemporalIterationSelection(_LOOP, Select.FIRST),
                           source_refs=(initial, *refs)),
        TemporalConstraint('returned-value-backedge', 'returned-value-ready', 'invoke:start',
                           Constraint.LOOP_CARRIED, iteration_distance=1, loop_scope=_LOOP,
                           source_refs=(backedge, *ports, *refs)),
        TemporalConstraint('normal-exit-control', 'returned-value-ready', 'exit', Constraint.CAUSAL,
                           before_iteration=TemporalIterationSelection(_LOOP, Select.LAST_NORMAL),
                           source_refs=(exit_, *refs)),
    )
    temporal = TemporalModel(TemporalMode.SEMANTIC, tuple(events), tuple(activities), tuple(occupancies),
        constraints, (TemporalScope(_ROOT, TemporalScopeKind.ROOT, 'entry', 'exit'),
                      TemporalScope(_LOOP, TemporalScopeKind.LOOP, 'iteration-entry', 'iteration-exit',
                                    parent=_ROOT, iteration_symbol='i', trip_count=match['loop_count'],
                                    source_refs=(original, *refs))),
        resource_capacities={_RESOURCE: None}, path_condition=TemporalPathCondition.NORMAL_COMPLETE,
        source_refs=refs)
    schedule = VersionedResourceSchedule('returned-state:' + reference['sha256'], (), (), (), tasks,
        temporal=temporal, proof=dict(recurrence='RETURNED_VALUE_TO_NEXT_SOURCE_ARGUMENT',
            trip_count_authority='PROFILED_CALL_CHAIN', logical_value_ready_event='returned-value-ready',
            storage_identity_proven=False, provided_target_capacity_is_available_budget=False),
        unresolved=tuple(dict(field=field, reason=reason) for field, reason in (
            ('operator_effects', 'UNDERLYING_CALL_AND_GETTER_EFFECTS_NOT_CLOSED'),
            ('duration', 'NO_MATCHED_OPERATOR_SERVICE_MEASUREMENT'),
            ('physical_slots', 'NO_COMPLETE_ALIAS_ALLOCATOR_AND_LAST_USE_PROOF'),
        )))
    return schedule, match


def project(binding_path, target_path):
    binding_ref = artifact_reference(Path(binding_path).resolve(strict=True), binding.SCHEMA)
    target_path = Path(target_path).resolve(strict=True)
    if target_path.is_dir():
        target_path /= 'result.json'
    target_ref = artifact_reference(target_path, 'scar.target-hardware-graph')
    joined = binding.check(binding_ref['uri'])
    if joined != read_artifact(binding_ref):
        raise ValueError('returned-value binding changed during time projection')
    schedule, match = _schedule(joined, binding_ref)
    summary = validate_versioned_schedule(schedule)
    resources = derive_temporal_resources(schedule, target_path,
        source_context=dict(target_reference=target_ref,
                            caller_affinity_cpu_ids=match['observation_cpu_ids'],
                            source_refs=(_loc(binding_ref, '#/matches'),)))
    sizes = sorted({size for row in joined['matches'] for size in row['output_size_bytes']})
    identity = [relation for row in joined['matches'] for relation in row['state_identity_observations']]
    return dict(schema=SCHEMA, schema_version=1, binding=binding_ref, target=target_ref,
        time_graph=schedule_document(schedule), time_graph_summary=summary,
        resources=projection_document(resources), causal_selection=dict(
            may_parallelize_rounds_in_one_activation=False,
            reason='NEXT_CALL_CONSUMES_PREVIOUS_PROPERTY_RESULT',
            cross_activation_independence='UNKNOWN'),
        value_lifetime=dict(observed_logical_payload_bytes=sizes,
            minimum_live_until='NEXT_CONSUMER_ENTRY_ON_EVERY_OBSERVED_BACKEDGE',
            same_object_transfers=sum(relation == 'SAME_OBJECT_OBSERVED' for relation in identity),
            backing_capacity_bytes=None, actual_last_use=None, memory_ranking='UNKNOWN'),
        cost_model='UNKNOWN', decision='NO_OP', deployment_authorized=False)


def check(path):
    ref = artifact_reference(Path(path).resolve(strict=True), SCHEMA)
    stored = read_artifact(ref)
    actual = project(stored['binding']['uri'], stored['target']['uri'])
    if actual != stored:
        raise ValueError('retained time projection differs from source and hardware replay')
    return actual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding')
    parser.add_argument('--target')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    if args.check and not (args.binding or args.target or args.out):
        result = check(args.check)
    elif args.binding and args.target and args.out and not args.check:
        result = project(args.binding, args.target)
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        parser.error('use --binding FILE --target TARGET --out FILE or --check FILE')
    print(json.dumps({key: result[key] for key in ('causal_selection', 'value_lifetime', 'decision')}))


if __name__ == '__main__':
    main()
