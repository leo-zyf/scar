"""Replay component evidence into conditional costs of original activity groups."""
import argparse
import json
from pathlib import Path

from scar.compiler.bind_time_resources import read_resource_bindings, SCHEMA as RESOURCE_SCHEMA
from scar.ir.activity_costs import (activity_cost_document, activity_cost_from_dict,
    validate_activity_cost_table)
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.temporal_resources import projection_from_dict
from scar.optimization.schedule import schedule_from_dict
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.temporal_activity_costs import (project_stateful_activity_costs,
    project_unknown_activity_costs)

SCHEMA = 'scar.activity-cost-preparation'


def _derive(resource_reference, domain_reference, candidate_id, calibration_refs, boundary_refs=()):
    resource_receipt = read_resource_bindings(resource_reference['uri'])
    if resource_receipt != read_artifact(resource_reference):
        raise ValueError('original resource preparation changed during cost replay')
    resources = projection_from_dict(read_artifact(resource_receipt['projection']))
    time = read_artifact(resource_receipt['time_schedule'])
    schedule = schedule_from_dict(read_artifact(time['schedule']))
    refs = ({'artifact': resource_reference, 'locator': '#/time_schedule'},)
    if domain_reference is None:
        if candidate_id is not None or calibration_refs or boundary_refs:
            raise ValueError('canonical facts require an explicit original domain and candidate')
        return (*project_unknown_activity_costs(schedule, resources, source_refs=refs,
            reason='NO_ADMITTED_ACTIVITY_COMPONENT_SOURCE'), schedule, resources)
    if not candidate_id or time['source_kind'] != 'stateful_recipe':
        raise ValueError('this component protocol requires a source-replayed stateful CPU recipe and candidate')
    domain = read_artifact(domain_reference)
    for ref in (*calibration_refs, *boundary_refs):
        read_artifact(ref)
    recipe = read_artifact(time['source'])
    source_refs = (*refs, {'artifact': domain_reference, 'locator': '#/candidates', 'candidate_id': candidate_id},
        *({'artifact': ref, 'locator': '#/components'} for ref in calibration_refs))
    table, derivation = project_stateful_activity_costs(schedule, resources, recipe, domain,
        candidate_id, [ref['uri'] for ref in calibration_refs], source_refs=source_refs,
        boundary_calibrations=boundary_refs)
    for ref in (resource_reference, domain_reference, *calibration_refs, *boundary_refs):
        read_artifact(ref)
    return table, derivation, schedule, resources


def _write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, value['schema'])


def prepare_activity_costs(resources, out, *, domain=None, candidate_id=None, calibrations=(),
                           boundary_calibrations=()):
    reference = artifact_reference(Path(resources).resolve(), RESOURCE_SCHEMA)
    domain_ref = (artifact_reference(Path(domain).resolve(), 'scar.stateful-cpu-domain')
                  if domain is not None else None)
    calibration_refs = [artifact_reference(Path(path).resolve(), 'scar.stateful-resource-calibration')
                        for path in calibrations]
    boundary_refs = [artifact_reference(Path(path).resolve(), 'scar.stateful-boundary-cut-calibration')
                     for path in boundary_calibrations]
    table, derivation, schedule, projection = _derive(reference, domain_ref, candidate_id, calibration_refs, boundary_refs)
    directory = Path(out).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(schema=SCHEMA, schema_version=1, resources=reference, domain=domain_ref,
        candidate_id=candidate_id, calibrations=calibration_refs,
        table=_write(directory/'costs.json', activity_cost_document(table)),
        derivation=_write(directory/'derivation.json', dict(
            schema='scar.activity-cost-derivation', schema_version=1, **derivation)),
        summary=validate_activity_cost_table(table, schedule, projection),
        component_facts_source_replayed=bool(calibration_refs or boundary_refs), target_program_executed=False,
        variants_ranked=False, solver_invoked=False, native_compiled=False,
        complete_call_cost=False, performance_measured=False, deployment_authorized=False)
    if boundary_refs:
        report['boundary_calibrations'] = boundary_refs
    _write(directory/'result.json', report)
    return report


def read_activity_costs(path):
    reference = artifact_reference(Path(path).resolve(), SCHEMA)
    report = read_artifact(reference)
    if (report.get('schema'), report.get('schema_version')) != (SCHEMA, 1):
        raise ValueError('source-replayed activity cost receipt required')
    table = activity_cost_from_dict(read_artifact(report['table']))
    boundary_refs = report.get('boundary_calibrations', ())
    replay, derivation, schedule, resources = _derive(report['resources'], report['domain'],
        report['candidate_id'], report['calibrations'], boundary_refs)
    expected = dict(schema=SCHEMA, schema_version=1, resources=report['resources'], domain=report['domain'],
        candidate_id=report['candidate_id'], calibrations=report['calibrations'],
        table=report['table'], derivation=report['derivation'],
        summary=validate_activity_cost_table(replay, schedule, resources),
        component_facts_source_replayed=bool(report['calibrations'] or boundary_refs), target_program_executed=False,
        variants_ranked=False, solver_invoked=False, native_compiled=False,
        complete_call_cost=False, performance_measured=False, deployment_authorized=False)
    if boundary_refs:
        expected['boundary_calibrations'] = boundary_refs
    if (table != replay or report != expected or semantic_sha256(read_artifact(report['derivation'])) != semantic_sha256(dict(
            schema='scar.activity-cost-derivation', schema_version=1, **derivation))):
        raise ValueError('activity costs differ from original source/resource/component replay')
    read_artifact(reference)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--resources')
    source.add_argument('--check')
    parser.add_argument('--domain')
    parser.add_argument('--candidate')
    parser.add_argument('--calibration', action='append', default=[])
    parser.add_argument('--boundary-calibration', action='append', default=[])
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check:
        if any((args.out, args.domain, args.candidate, args.calibration, args.boundary_calibration)):
            parser.error('check consumes only the saved activity-cost receipt')
        result = read_activity_costs(args.check)
    else:
        if not args.out:
            parser.error('activity cost preparation requires --out')
        result = prepare_activity_costs(args.resources, args.out, domain=args.domain,
            candidate_id=args.candidate, calibrations=args.calibration, boundary_calibrations=args.boundary_calibration)
    print(json.dumps(dict(summary=result['summary'], table=result['table'],
        complete_call_cost=False, deployment_authorized=False), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
