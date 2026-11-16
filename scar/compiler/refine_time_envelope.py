"""Model an existing CPU implementation envelope in the same task/time IR."""
import argparse
import json
from pathlib import Path

from scar.analysis.schedule_time import attach_task_time
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.versioned_schedule import TemporalOccupancyKind, TemporalScopeKind
from scar.optimization.schedule import schedule_document, schedule_from_dict

SCHEMA = 'scar.implementation-time-preparation'
SERIAL_AFFINITY = 'SOURCE_PROVEN_SERIAL_AFFINITY'
SERIAL_SCOPE = 'SOURCE_PROVEN_SERIAL_SCOPE_DEMANDS'
SCOPE_CLEANUP = 'SOURCE_PROVEN_SCOPE_CLEANUP'


def _derive(reference, *, serial_demands=False, scope_cleanup=False):
    from scar.compiler.time_schedule import read_time_schedule, _summary, SCHEMA as BASE_SCHEMA
    from scar.analysis.implementation_envelope import derive_stateful_envelope
    base = read_time_schedule(reference['uri'])
    if base != read_artifact(reference) or base['schema'] != BASE_SCHEMA:
        raise ValueError('implementation refinement requires the original unrefined time receipt')
    if base['source_kind'] != 'stateful_recipe':
        raise ValueError('current implementation envelope contract is the existing stateful CPU backend')
    recipe = read_artifact(base['source'])
    original = schedule_from_dict(read_artifact(base['schedule']))
    if any(scope.kind is TemporalScopeKind.LOOP for scope in original.temporal.scopes):
        raise ValueError('standalone method envelope cannot model a repeat adapter; preserve the original shared loop')
    if scope_cleanup:
        from scar.analysis.implementation_scope import derive_stateful_scope_envelope
        refined = derive_stateful_scope_envelope(original, recipe)
    else:
        refined = derive_stateful_envelope(original, recipe)
    activities = {row.id: row for row in original.temporal.activities}
    demands = {}
    for row in original.temporal.occupancies:
        if row.kind is TemporalOccupancyKind.EXECUTION_DEMAND:
            task = activities[row.activity].task
            demands.setdefault(task, {})[row.resource] = row.units
    schedule = attach_task_time(refined, execution_demands=demands,
        resource_capacities=original.temporal.resource_capacities,
        source_refs=original.temporal.source_refs)
    if scope_cleanup:
        from scar.analysis.control_milestone import derive_scope_control_milestone
        schedule = derive_scope_control_milestone(schedule, recipe)
    if serial_demands:
        target = recipe['artifacts']['target_snapshot']
        read_artifact(target)
        if scope_cleanup:
            from scar.analysis.scope_demands import derive_serial_scope_demands
            schedule = derive_serial_scope_demands(schedule, recipe, Path(target['uri']).parent)
        else:
            from scar.analysis.affinity_demands import derive_serial_affinity_demands
            schedule = derive_serial_affinity_demands(schedule, recipe, Path(target['uri']).parent)
    read_artifact(reference)
    return base, schedule, _summary(schedule)


def _write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, value['schema'])


def _report(base_reference, base, schedule_ref, summary, *, serial_demands=False, scope_cleanup=False):
    result = dict(schema=SCHEMA, schema_version=1, base_time=base_reference,
        source_kind=base['source_kind'], source=base['source'], loop_source=base['loop_source'],
        schedule=schedule_ref, summary=summary,
        layer='MODEL_EXISTING_IMPLEMENTATION_ENVELOPE',
        target_program_executed=False, solver_invoked=False, native_compiled=False,
        performance_measured=False, deployment_authorized=False)
    if serial_demands:
        result['resource_refinement'] = SERIAL_SCOPE if scope_cleanup else SERIAL_AFFINITY
    if scope_cleanup:
        result['implementation_refinement'] = SCOPE_CLEANUP
    return result


def prepare_implementation_time(time_schedule, out, *, serial_demands=False, scope_cleanup=False):
    from scar.compiler.time_schedule import time_schedule_reference
    reference = time_schedule_reference(time_schedule)
    base, schedule, summary = _derive(reference, serial_demands=serial_demands, scope_cleanup=scope_cleanup)
    directory = Path(out).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    result = _report(reference, base,
        _write(directory/'schedule.json', schedule_document(schedule)), summary,
        serial_demands=serial_demands, scope_cleanup=scope_cleanup)
    _write(directory/'result.json', result)
    return result


def read_implementation_time(path):
    reference = artifact_reference(Path(path).resolve(), SCHEMA)
    result = read_artifact(reference)
    if (result.get('schema'), result.get('schema_version')) != (SCHEMA, 1):
        raise ValueError('implementation time preparation receipt required')
    if result.get('resource_refinement') not in (None, SERIAL_AFFINITY, SERIAL_SCOPE):
        raise ValueError('unknown implementation resource refinement')
    if result.get('implementation_refinement') not in (None, SCOPE_CLEANUP):
        raise ValueError('unknown source implementation scope refinement')
    scope_cleanup = result.get('implementation_refinement') == SCOPE_CLEANUP
    serial_demands = result.get('resource_refinement') in (SERIAL_AFFINITY, SERIAL_SCOPE)
    if serial_demands and scope_cleanup != (result['resource_refinement'] == SERIAL_SCOPE):
        raise ValueError('scope and legacy resource refinements cannot be mixed')
    base, replay, summary = _derive(result['base_time'], serial_demands=serial_demands, scope_cleanup=scope_cleanup)
    saved = schedule_from_dict(read_artifact(result['schedule']))
    if saved != replay or result != _report(result['base_time'], base, result['schedule'], summary,
            serial_demands=serial_demands, scope_cleanup=scope_cleanup):
        raise ValueError('implementation envelope differs from original source/backend/phase replay')
    read_artifact(reference)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--time-schedule')
    source.add_argument('--check')
    parser.add_argument('--out')
    parser.add_argument('--bind-serial-demands', action='store_true')
    parser.add_argument('--scope-cleanup', action='store_true')
    args = parser.parse_args(argv)
    if args.check:
        if args.out or args.bind_serial_demands or args.scope_cleanup:
            parser.error('check consumes only the saved implementation-time receipt')
        result = read_implementation_time(args.check)
    else:
        if not args.out:
            parser.error('implementation-time preparation requires --out')
        result = prepare_implementation_time(args.time_schedule, args.out,
            serial_demands=args.bind_serial_demands, scope_cleanup=args.scope_cleanup)
    print(json.dumps(dict(summary=result['summary'], schedule=result['schedule'],
        deployment_authorized=False), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
