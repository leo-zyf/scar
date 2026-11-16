"""Project replayed activity component coordinates without solving or executing."""
import argparse
import json
from pathlib import Path

from scar.compiler.cost_activities import replay_activity_costs, SCHEMA as COST_SCHEMA
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.record_codec import encode
from scar.ir.versioned_schedule import schedule_document
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.activity-time-preparation'


def _derive(reference):
    from scar.optimization.activity_time_projection import project_activity_time
    report, table, derivation, schedule, resources = replay_activity_costs(reference['uri'])
    if report != read_artifact(reference):
        raise ValueError('original activity-cost preparation changed during time projection')
    accounting = derivation.get('scope_accounting')
    if type(accounting) is not dict:
        raise ValueError('activity time requires the original source-replayed scope accounting')
    # These are cheap retained-record reads. The complete resource/time/source
    # admission has already run inside the single cost replay above.
    resource_receipt = read_artifact(report['resources'])
    time_reference = resource_receipt['time_schedule']
    time_receipt = read_artifact(time_reference)
    scope_reference = time_receipt['schedule']
    if (accounting.get('scope_schedule') != scope_reference
            or read_artifact(scope_reference) != schedule_document(schedule)):
        raise ValueError('activity time must retain the exact derived scope schedule reference')
    inputs = dict(costs=reference, cost_table=report['table'], cost_derivation=report['derivation'],
        resource_preparation=report['resources'], resource_projection=resource_receipt['projection'],
        time_preparation=time_reference, scope_schedule=scope_reference,
        source=time_receipt['source'], target=encode(resources.target))
    if report['domain'] is not None:
        inputs['domain'] = report['domain']
    for name in ('calibrations', 'boundary_calibrations', 'dag_calibrations'):
        for index, calibration in enumerate(report.get(name, ())):
            inputs[name + ':' + str(index)] = calibration
    refs = tuple(dict(role=role, artifact=value, locator='#') for role, value in inputs.items())
    projection = project_activity_time(schedule, resources, table, accounting, source_refs=refs)
    if (projection.get('schema'), projection.get('schema_version')) != ('scar.activity-time-projection', 1):
        raise ValueError('versioned activity-time projection required')
    for value in inputs.values():
        read_artifact(value)
    return projection, inputs, (report, table, derivation, schedule, resources)


def _write(path, document):
    with path.open('x') as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, document['schema'])


def _report(inputs, projection_reference, projection):
    return dict(schema=SCHEMA, schema_version=1, inputs=inputs, projection=projection_reference,
        summary=projection['summary'],
        coordinate_scope='CONDITIONAL_COMPONENT_MODELS_NOT_OBSERVED_PHYSICAL_TIMELINE',
        original_graph_modified=False, target_program_executed=False, solver_invoked=False,
        native_compiled=False, performance_measured=False, physical_timeline_proven=False,
        deployment_authorized=False)


def prepare_activity_time(costs, out):
    """Publish one projection from a complete cold cost/source replay."""
    directory = Path(out).resolve()
    if directory.exists():
        raise ValueError('fresh activity-time output required')
    reference = artifact_reference(Path(costs).resolve(), COST_SCHEMA)
    projection, inputs, _ = _derive(reference)
    directory.mkdir(parents=True, exist_ok=False)
    projection_reference = _write(directory/'projection.json', projection)
    report = _report(inputs, projection_reference, projection)
    _write(directory/'result.json', report)
    return report


def replay_activity_time(path):
    """Cold-replay once and retain qualified inputs for a downstream layer."""
    reference = artifact_reference(Path(path).resolve(), SCHEMA)
    report = read_artifact(reference)
    if (report.get('schema'), report.get('schema_version')) != (SCHEMA, 1):
        raise ValueError('versioned activity-time preparation required')
    projection, inputs, context = _derive(report['inputs']['costs'])
    retained = read_artifact(report['projection'])
    expected = _report(inputs, report['projection'], projection)
    if (report != expected or semantic_sha256(retained) != semantic_sha256(projection)):
        raise ValueError('activity time differs from complete source/cost/resource projection replay')
    read_artifact(reference)
    return report, projection, *context


def read_activity_time(path):
    return replay_activity_time(path)[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--costs')
    source.add_argument('--check')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check:
        if args.out:
            parser.error('check consumes only the saved activity-time receipt')
        result = read_activity_time(args.check)
    else:
        if not args.out:
            parser.error('activity time projection requires --out')
        result = prepare_activity_time(args.costs, args.out)
    print(json.dumps(dict(summary=result['summary'], projection=result['projection'],
        physical_timeline_proven=False, deployment_authorized=False), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
