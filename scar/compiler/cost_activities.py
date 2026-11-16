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
from scar.optimization.boundary_activity_costs import boundary_calibration_reference
from scar.optimization.temporal_activity_costs import (project_stateful_activity_costs,
    project_unknown_activity_costs)

SCHEMA = 'scar.activity-cost-preparation'


def _workspace_costs(schedule, resources, time, time_reference, resource_reference, domain_reference,
                     candidate_id, calibration_refs, boundary_refs, dag_refs):
    from scar.optimization.workspace_costs import derive_workspace_costs
    from scar.optimization.stateful_cpu_choices import _domain
    from scar.optimization.stateful_cpu_model import stateful_resource_context, component_key
    from scar.optimization.temporal_activity_costs import _samples
    from scar.optimization.stateful_recipe import stateful_backend_capability
    if boundary_refs or dag_refs:
        raise ValueError('standalone boundary/DAG clocks cannot be assigned to shared-loop phases')
    preparation = read_artifact(time['source'])
    plan_reference = preparation['step_plan']
    plan = read_artifact(plan_reference)
    # The admitted base may retain a byte-identical plan at another URI.
    # Keep both provenance paths, using the CPU proof's actual base reference
    # only after rechecking its complete content against the preparation.
    original_plan_reference = plan_reference
    domains = [row for row in schedule.temporal.source_refs
               if row.get('schema') == 'scar.workspace-cpu-demand-source-proof']
    if domains:
        if len(domains) != 1 or read_artifact(domains[0]['stage_plan']) != plan:
            raise ValueError('source workspace CPU proof and preparation step plans differ')
        plan_reference = domains[0]['stage_plan']
    refs = ({'artifact':resource_reference,'locator':'#/time_schedule'},
            {'artifact':original_plan_reference,'locator':'#'},
            {'artifact':plan_reference,'locator':'#'},
            {'artifact':artifact_reference(Path(__file__).resolve(),'scar.python-source'),'locator':'#'},
            {'artifact':artifact_reference(Path(__file__).resolve().parents[1]/'optimization/workspace_costs.py',
                 'scar.python-source'),'locator':'#'})
    primitive = None
    if domain_reference is None:
        if candidate_id is not None or calibration_refs:
            raise ValueError('canonical workspace components require an original domain and candidate')
    else:
        if not candidate_id:
            raise ValueError('canonical workspace component context requires its original candidate')
        domain = _domain(read_artifact(domain_reference))
        context = stateful_resource_context(domain, candidate_id)
        documents = []
        session = dict(sources={}, validated_probes=set(), statistics={})
        for reference in calibration_refs:
            document = read_artifact(reference)
            # Qualify exact original samples and compiled-probe bytes once per
            # source. Foreign contexts remain diagnostics, never usable facts.
            if document['resource_context'] == context:
                for index, row in enumerate(document['components']):
                    if row['component'] == 'ROW_COPY_STAGE':
                        term = dict(component=row['component'],parameters=row['parameters'],
                            key=component_key(row['component'],row['parameters']))
                        _samples(dict(artifact=reference,locator='#/components/'+str(index)),term,session)
            documents.append(dict(reference=reference,document=document))
        primitive = dict(domain=domain,candidate_id=candidate_id,actual_step_plan=plan,
            actual_step_plan_reference=plan_reference,actual_backend=stateful_backend_capability(plan),
            calibrations=documents)
        refs = (*refs,{'artifact':domain_reference,'locator':'#/backend_capabilities/serial'},
                *({'artifact':reference,'locator':'#/components'} for reference in calibration_refs))
    table, accounting = derive_workspace_costs(schedule, resources, primitive_evidence=primitive,
        workspace_time_ref=time_reference,
        workspace_plan_ref=time['workspace_plan'], source_refs=refs)
    for reference in (resource_reference,*([] if domain_reference is None else [domain_reference]),*calibration_refs):
        read_artifact(reference)
    derivation = dict(protocol='SOURCE_QUALIFIED_WORKSPACE_ONCE_AND_INDEXED_PRIMITIVE_COSTS',
        workspace_accounting=accounting, summary=validate_activity_cost_table(table,schedule,resources))
    return table, derivation, schedule, resources


def _derive(resource_reference, domain_reference, candidate_id, calibration_refs, boundary_refs=(), dag_refs=()):
    resource_receipt = read_resource_bindings(resource_reference['uri'])
    if resource_receipt != read_artifact(resource_reference):
        raise ValueError('original resource preparation changed during cost replay')
    resources = projection_from_dict(read_artifact(resource_receipt['projection']))
    time = read_artifact(resource_receipt['time_schedule'])
    schedule = schedule_from_dict(read_artifact(time['schedule']))
    refs = ({'artifact': resource_reference, 'locator': '#/time_schedule'},)
    if time['source_kind'] == 'stateful_workspace':
        return _workspace_costs(schedule, resources, time, resource_receipt['time_schedule'], resource_reference, domain_reference,
            candidate_id, calibration_refs, boundary_refs, dag_refs)
    if domain_reference is None:
        if candidate_id is not None or calibration_refs or boundary_refs or dag_refs:
            raise ValueError('canonical facts require an explicit original domain and candidate')
        return (*project_unknown_activity_costs(schedule, resources, source_refs=refs,
            reason='NO_ADMITTED_ACTIVITY_COMPONENT_SOURCE'), schedule, resources)
    if not candidate_id or time['source_kind'] != 'stateful_recipe':
        raise ValueError('this component protocol requires a source-replayed stateful CPU recipe and candidate')
    domain = read_artifact(domain_reference)
    for ref in (*calibration_refs, *boundary_refs, *dag_refs):
        read_artifact(ref)
    recipe = read_artifact(time['source'])
    source_refs = (*refs, {'artifact': domain_reference, 'locator': '#/candidates', 'candidate_id': candidate_id},
        *({'artifact': ref, 'locator': '#/components'} for ref in calibration_refs))
    table, derivation = project_stateful_activity_costs(schedule, resources, recipe, domain,
        candidate_id, [ref['uri'] for ref in calibration_refs], source_refs=source_refs,
        boundary_calibrations=boundary_refs, dag_calibrations=dag_refs, scope_schedule_ref=time['schedule'])
    for ref in (resource_reference, domain_reference, *calibration_refs, *boundary_refs, *dag_refs):
        read_artifact(ref)
    return table, derivation, schedule, resources


def _write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, value['schema'])


def prepare_activity_costs(resources, out, *, domain=None, candidate_id=None, calibrations=(),
                           boundary_calibrations=(), dag_calibrations=()):
    reference = artifact_reference(Path(resources).resolve(), RESOURCE_SCHEMA)
    domain_ref = (artifact_reference(Path(domain).resolve(), 'scar.stateful-cpu-domain')
                  if domain is not None else None)
    calibration_refs = [artifact_reference(Path(path).resolve(), 'scar.stateful-resource-calibration')
                        for path in calibrations]
    boundary_refs = [boundary_calibration_reference(path) for path in boundary_calibrations]
    dag_refs = [artifact_reference(Path(path).resolve(), 'scar.stateful-stage-dag-calibration')
                for path in dag_calibrations]
    table, derivation, schedule, projection = _derive(reference, domain_ref, candidate_id,
        calibration_refs, boundary_refs, dag_refs)
    directory = Path(out).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(schema=SCHEMA, schema_version=1, resources=reference, domain=domain_ref,
        candidate_id=candidate_id, calibrations=calibration_refs,
        table=_write(directory/'costs.json', activity_cost_document(table)),
        derivation=_write(directory/'derivation.json', dict(
            schema='scar.activity-cost-derivation', schema_version=1, **derivation)),
        summary=validate_activity_cost_table(table, schedule, projection),
        component_facts_source_replayed=bool(calibration_refs or boundary_refs or dag_refs), target_program_executed=False,
        variants_ranked=False, solver_invoked=False, native_compiled=False,
        complete_call_cost=False, performance_measured=False, deployment_authorized=False)
    if boundary_refs:
        report['boundary_calibrations'] = boundary_refs
    if dag_refs:
        report['dag_calibrations'] = dag_refs
    _write(directory/'result.json', report)
    return report


def replay_activity_costs(path):
    """Cold-replay once and return the retained table plus its source inputs."""
    reference = artifact_reference(Path(path).resolve(), SCHEMA)
    report = read_artifact(reference)
    if (report.get('schema'), report.get('schema_version')) != (SCHEMA, 1):
        raise ValueError('source-replayed activity cost receipt required')
    table = activity_cost_from_dict(read_artifact(report['table']))
    boundary_refs = report.get('boundary_calibrations', ())
    dag_refs = report.get('dag_calibrations', ())
    replay, derivation, schedule, resources = _derive(report['resources'], report['domain'],
        report['candidate_id'], report['calibrations'], boundary_refs, dag_refs)
    expected = dict(schema=SCHEMA, schema_version=1, resources=report['resources'], domain=report['domain'],
        candidate_id=report['candidate_id'], calibrations=report['calibrations'],
        table=report['table'], derivation=report['derivation'],
        summary=validate_activity_cost_table(replay, schedule, resources),
        component_facts_source_replayed=bool(report['calibrations'] or boundary_refs or dag_refs), target_program_executed=False,
        variants_ranked=False, solver_invoked=False, native_compiled=False,
        complete_call_cost=False, performance_measured=False, deployment_authorized=False)
    if boundary_refs:
        expected['boundary_calibrations'] = boundary_refs
    if dag_refs:
        expected['dag_calibrations'] = dag_refs
    saved_derivation = read_artifact(report['derivation'])
    if (table != replay or report != expected or semantic_sha256(saved_derivation) != semantic_sha256(dict(
            schema='scar.activity-cost-derivation', schema_version=1, **derivation))):
        raise ValueError('activity costs differ from original source/resource/component replay')
    read_artifact(reference)
    return report, table, saved_derivation, schedule, resources


def read_activity_costs(path):
    return replay_activity_costs(path)[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--resources')
    source.add_argument('--check')
    parser.add_argument('--domain')
    parser.add_argument('--candidate')
    parser.add_argument('--calibration', action='append', default=[])
    parser.add_argument('--boundary-calibration', action='append', default=[])
    parser.add_argument('--dag-calibration', action='append', default=[])
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check:
        if any((args.out, args.domain, args.candidate, args.calibration, args.boundary_calibration, args.dag_calibration)):
            parser.error('check consumes only the saved activity-cost receipt')
        result = read_activity_costs(args.check)
    else:
        if not args.out:
            parser.error('activity cost preparation requires --out')
        result = prepare_activity_costs(args.resources, args.out, domain=args.domain,
            candidate_id=args.candidate, calibrations=args.calibration,
            boundary_calibrations=args.boundary_calibration, dag_calibrations=args.dag_calibration)
    print(json.dumps(dict(summary=result['summary'], table=result['table'],
        complete_call_cost=False, deployment_authorized=False), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
