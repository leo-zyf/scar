"""Connect source-replayed workspace time/resource evidence to trial lowering.

The certificate preserves the admitted implementation's order and transactions.
It does not turn component estimates into a numeric timeline or a deployment
decision. Compilation is a separate consumer of this retained certificate.
"""
import argparse
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.versioned_schedule import schedule_document, schedule_from_dict
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.workspace-execution-refinement-publication'
CERTIFICATE = 'scar.workspace-execution-refinement'


def _derive(reference):
    from scar.compiler.cost_activities import replay_activity_costs
    from scar.optimization.workspace_execution_refinement import derive_workspace_execution_refinement
    costs, table, derivation, schedule, resources = replay_activity_costs(reference['uri'])
    if costs != read_artifact(reference):
        raise ValueError('workspace cost publication changed during source replay')
    resource_receipt = read_artifact(costs['resources'])
    time_reference = resource_receipt['time_schedule']
    time = read_artifact(time_reference)
    if ((time.get('schema'), time.get('schema_version')) != ('scar.workspace-demand-time-preparation', 1)
            or time.get('source_kind') != 'stateful_workspace'
            or 'workspace_accounting' not in derivation):
        raise ValueError('current source-qualified workspace CPU/resource/cost evidence required')
    parent = read_artifact(time['base_time'])
    baseline = schedule_from_dict(read_artifact(parent['schedule']))
    metadata = read_artifact(time['native_manifest'])
    preparation = read_artifact(time['source'])
    workspace_plan = read_artifact(time['workspace_plan'])
    proofs = [row for row in schedule.temporal.source_refs
              if row.get('schema') == 'scar.workspace-cpu-demand-source-proof']
    if (len(proofs) != 1 or preparation.get('base_native') is None
            or metadata['preparation'] != time['source']
            or preparation['workspace_plan'] != time['workspace_plan']
            or preparation['base_native'] != metadata['base_native_reference']
            or parent['native_manifest'] != time['native_manifest']):
        raise ValueError('exact source preparation, fixed parent native and CPU proof required')
    step_plan = read_artifact(proofs[0]['stage_plan'])
    if semantic_sha256(step_plan) != semantic_sha256(read_artifact(preparation['step_plan'])):
        raise ValueError('workspace and admitted source CPU proof have different step plans')
    refs = tuple(dict(artifact=row, locator='#')
                 for row in (reference, costs['resources'], time_reference))
    certificate = derive_workspace_execution_refinement(schedule, resources, table,
        derivation['workspace_accounting'], workspace_plan, metadata,
        step_plan=step_plan, baseline_schedule=baseline, source_refs=refs)
    # Recheck inputs after derivation; no target library is loaded in this path.
    for row in (reference, costs['resources'], time_reference, time['source'], time['native_manifest']):
        read_artifact(row)
    context = dict(preparation=time['source'], native_manifest=time['native_manifest'],
        workspace_plan=workspace_plan, native_metadata=metadata, step_plan=step_plan,
        baseline_schedule=baseline, schedule=schedule, resources=resources,
        costs=table, accounting=derivation['workspace_accounting'], source_refs=refs)
    return certificate, context


def _report(reference, certificate_reference, certificate, context):
    root = Path(__file__).resolve().parents[1]
    return dict(schema=SCHEMA, schema_version=1, costs=reference,
        certificate=certificate_reference, preparation=context['preparation'],
        native_blueprint=context['native_manifest'],
        summary=dict(activities=len(context['schedule'].temporal.activities),
            owned_slots=len(context['workspace_plan']['slots']),
            trip_count=context['workspace_plan']['trip_count'], shared_body_count=int(
                context['workspace_plan']['trip_count'] > 0), iterations_unrolled=0),
        implementations={name:artifact_reference(root/name, 'scar.python-source') for name in
            ('compiler/refine_workspace_execution.py', 'optimization/workspace_execution_refinement.py')},
        implementation_scope='WORKSPACE_LIFETIME_AND_EXISTING_CONTROL_ORDER_ONLY',
        source_evidence_replayed=True, source_graph_sha256=semantic_sha256(schedule_document(context['schedule'])),
        full_numeric_timeline_planned=False, physical_optimality_proven=False,
        target_program_executed=False, native_libraries_loaded=False, native_compiled=False,
        performance_measured=False, trial_only=True, default_backend='NO_OP',
        automatic_deployment_authorized=False)


def _write(path, document):
    with path.open('x') as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, document['schema'])


def refine_workspace_execution(costs, out):
    reference = artifact_reference(Path(costs).resolve(), 'scar.activity-cost-preparation')
    output = Path(out).resolve()
    if output.exists() or output.is_relative_to(Path(reference['uri']).parent):
        raise ValueError('fresh independent workspace refinement output required')
    certificate, context = _derive(reference)
    output.mkdir(parents=True)
    cert_reference = _write(output/'certificate.json', certificate)
    report = _report(reference, cert_reference, certificate, context)
    _write(output/'result.json', report)
    return report


def replay_workspace_execution(path):
    """Cold-replay all source evidence once before admitting this publication."""
    reference = artifact_reference(Path(path).resolve(), SCHEMA)
    report = read_artifact(reference)
    if (report.get('schema'), report.get('schema_version')) != (SCHEMA, 1):
        raise ValueError('workspace execution refinement publication required')
    if report.get('certificate', {}).get('schema') != CERTIFICATE:
        raise ValueError('typed workspace execution refinement certificate required')
    saved = read_artifact(report['certificate'])
    expected, context = _derive(report['costs'])
    if (semantic_sha256(saved) != semantic_sha256(expected)
            or report != _report(report['costs'], report['certificate'], expected, context)):
        raise ValueError('workspace execution certificate differs from complete source/graph/cost replay')
    read_artifact(reference)
    return report, expected, context


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--costs'); source.add_argument('--check')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check:
        if args.out:
            parser.error('check only consumes an existing refinement')
        report = replay_workspace_execution(args.check)[0]
    else:
        if not args.out:
            parser.error('refinement publication requires --out')
        report = refine_workspace_execution(args.costs, args.out)
    print(json.dumps({key:report[key] for key in ('certificate', 'summary', 'implementation_scope',
        'full_numeric_timeline_planned', 'default_backend', 'automatic_deployment_authorized')}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
