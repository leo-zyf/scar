"""Publish a replayable replacement graph, hardware bindings and cost lineage."""
import argparse
import json
from pathlib import Path

from scar.analysis.target_graph import TargetHardwareGraph
from scar.compiler.plan_closed_windows import replay_closed_window_plans, SCHEMA as WINDOW_SCHEMA
from scar.ir.activity_costs import activity_cost_document, validate_activity_cost_table
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.temporal_resources import projection_document, validate_resource_projection
from scar.ir.versioned_schedule import schedule_document, validate_versioned_schedule
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.window-rewrite-preparation'
KIND = 'FUSE_IDENTICAL_INDEPENDENT_RAW_COPY_LOOPS'


def _derive(reference, window_id):
    from scar.optimization.fused_copy_graph import derive_fused_copy_graph
    from scar.optimization.fused_copy_costs import project_fused_copy_resources, project_fused_copy_costs
    original_report, windows, plans, parent = replay_closed_window_plans(reference['uri'])
    if original_report != read_artifact(reference):
        raise ValueError('original window preparation changed during rewrite')
    choices = [(row['window_id'], candidate) for row in plans['windows']
        for candidate in row['candidate_transforms'] if candidate['kind'] == KIND
        and (window_id is None or row['window_id'] == window_id)]
    if len(choices) != 1:
        raise ValueError('one graph-derived fusion proposal required; specify --window if ambiguous')
    identity, proposal = choices[0]
    matches = [row for row in windows['windows'] if row['id'] == identity]
    if len(matches) != 1:
        raise ValueError('original complete window required')
    original = parent['schedule']
    parent_table_reference = parent['costs']['table']
    if read_artifact(parent_table_reference) != activity_cost_document(parent['table']):
        raise ValueError('original component table must retain its real saved artifact reference')
    schedule, certificate = derive_fused_copy_graph(original, matches[0], proposal)
    resources = project_fused_copy_resources(original, parent['resources'], schedule, certificate)
    costs, lineage = project_fused_copy_costs(original, parent['resources'], parent['table'],
        schedule, resources, certificate)
    inventory = TargetHardwareGraph(Path(resources.target.uri).parent)
    retained_resources = inventory.resources()
    resource_summary = validate_resource_projection(resources, schedule, resources=retained_resources)
    cost_summary = validate_activity_cost_table(costs, schedule, resources)
    context = dict(parent_window_preparation=reference['uri'], window_id=identity,
        proposal=proposal, original_schedule=original, original_resources=parent['resources'],
        original_table=parent['table'], original_cost_table_reference=parent_table_reference,
        recipe=parent['recipe'], qualified_parent=(original_report, windows, plans, parent))
    selected_tasks = set(proposal['task_ids'])
    copy_inputs = {key for task in original.tasks if task.id in selected_tasks for key in task.reads}
    summary = dict(original_tasks=len(original.tasks), rewritten_tasks=len(schedule.tasks),
        original_activities=len(original.temporal.activities), rewritten_activities=len(schedule.temporal.activities),
        original_values=len(original.values), rewritten_values=len(schedule.values),
        original_materializations=len(original.materializations), rewritten_materializations=len(schedule.materializations),
        original_slots=len(original.storage_slots), rewritten_slots=len(schedule.storage_slots),
        retained_hardware_inventory_records=len(retained_resources), hardware_inventory_pruned=False,
        removed_copy_loop_nests=len(proposal['task_ids'])-1,
        raw_copy_assignments_preserved=len(proposal['task_ids']),
        copied_payload_bytes=sum(material.region.nbytes for material in original.materializations
            if material.id in copy_inputs),
        data_copies_eliminated=0,
        resources=resource_summary, costs=cost_summary,
        replacement_duration=None, profitability_proven=False, automatic_selected=False,
        native_compiled=False, target_program_executed=False, deployment_authorized=False)
    read_artifact(reference)
    return schedule, resources, costs, certificate, lineage, context, summary


def _implementations():
    root = Path(__file__).resolve().parents[1]
    return {name: artifact_reference(root/name, 'scar.python-source') for name in
        ('compiler/prepare_window_rewrite.py', 'optimization/fused_copy_graph.py',
         'optimization/fused_copy_costs.py', 'ir/versioned_schedule.py')}


def _write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, value['schema'])


def _report(reference, window_id, artifacts, summary, context):
    return dict(schema=SCHEMA, schema_version=1, original_windows=reference, window_id=window_id,
        parent_cost_table=context['original_cost_table_reference'],
        **artifacts, implementations=_implementations(), summary=summary,
        original_graph_modified=False, replacement_graph_materialized=True,
        source_window_boundary_policy='SAME_CALLER_SAME_CPU_PRIVATE_COPY_OUTPUTS_READY_AT_COMPLETE_FUSION',
        physical_optimality_proven=False, automatic_transformation_authorized=False,
        execution_authorized=False, deployment_authorized=False)


def prepare_window_rewrite(windows, out, *, window=None):
    directory = Path(out).resolve()
    if directory.exists():
        raise ValueError('fresh replacement graph output required')
    reference = artifact_reference(Path(windows).resolve(), WINDOW_SCHEMA)
    schedule, resources, costs, certificate, lineage, context, summary = _derive(reference, window)
    directory.mkdir(parents=True)
    documents = dict(schedule=schedule_document(schedule), resources=projection_document(resources),
        costs=activity_cost_document(costs), certificate=certificate, lineage=lineage)
    artifacts = {key: _write(directory/(key+'.json'), value) for key, value in documents.items()}
    report = _report(reference, context['window_id'], artifacts, summary, context)
    _write(directory/'result.json', report)
    return report


def replay_window_rewrite(path):
    reference = artifact_reference(Path(path).resolve(), SCHEMA)
    report = read_artifact(reference)
    if (report.get('schema'), report.get('schema_version')) != (SCHEMA, 1):
        raise ValueError('versioned replacement graph preparation required')
    schedule, resources, costs, certificate, lineage, context, summary = _derive(
        report['original_windows'], report['window_id'])
    expected = _report(report['original_windows'], context['window_id'],
        {key: report[key] for key in ('schedule', 'resources', 'costs', 'certificate', 'lineage')}, summary, context)
    documents = dict(schedule=schedule_document(schedule), resources=projection_document(resources),
        costs=activity_cost_document(costs), certificate=certificate, lineage=lineage)
    if report != expected or any(semantic_sha256(read_artifact(report[key])) != semantic_sha256(value)
            for key, value in documents.items()):
        raise ValueError('replacement graph differs from original graph/source/resource/cost replay')
    validate_versioned_schedule(schedule)
    read_artifact(reference)
    return report, schedule, resources, costs, certificate, context


def read_window_rewrite(path):
    return replay_window_rewrite(path)[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--windows'); source.add_argument('--check')
    parser.add_argument('--out'); parser.add_argument('--window')
    args = parser.parse_args(argv)
    if args.check:
        if args.out or args.window:
            parser.error('check only consumes an existing replacement receipt')
        report = read_window_rewrite(args.check)
    else:
        if not args.out:
            parser.error('rewrite preparation requires --out')
        report = prepare_window_rewrite(args.windows, args.out, window=args.window)
    print(json.dumps(report['summary'], sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
