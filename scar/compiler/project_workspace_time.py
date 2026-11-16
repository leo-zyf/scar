"""Project a current workspace implementation into indexed time/resource IR."""
import argparse
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.versioned_schedule import schedule_document, validate_versioned_schedule
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.workspace-time-preparation'


def _derive(reference):
    from scar.codegen.stateful_workspace import inspect_workspace_repeat
    from scar.analysis.workspace_time import derive_workspace_time
    metadata, qualified = inspect_workspace_repeat(Path(reference['uri']).parent)
    if metadata != read_artifact(reference):
        raise ValueError('original workspace native publication changed during source replay')
    preparation, region, step_plan, workspace_plan, context = qualified
    source = metadata['preparation']
    if read_artifact(source) != preparation:
        raise ValueError('original workspace preparation reference differs')
    directory = Path(reference['uri']).parent
    cpp = artifact_reference(directory/metadata['files']['source'], 'scar.cpp-source')
    if cpp['sha256'] != metadata['sha256']['source']:
        raise ValueError('admitted workspace C++ source differs')
    target = next(row['artifact'] for row in context['source_refs'] if row['role'] == 'target')
    read_artifact(target)
    schedule = derive_workspace_time(workspace_plan, native_manifest=reference,
        native_metadata=metadata, cpp_source=cpp,
        workspace_reference=preparation['workspace_plan'])
    read_artifact(reference)
    return schedule, source, preparation['workspace_plan'], target


def _report(reference, source, workspace_plan, target, schedule_reference, schedule):
    root = Path(__file__).resolve().parents[1]
    summary = validate_versioned_schedule(schedule)
    summary['source_task_graph_executable'] = summary.pop('executable')
    certificate = schedule.proof['workspace_time']
    summary.update(trip_count=certificate['trip_count'], shared_body_count=int(certificate['trip_count'] > 0),
        iterations_unrolled=0, owned_slot_physical_count=len(certificate['storage_occurrences']),
        normal_completion_only=True, failure_path_timing_modeled=False,
        python_packing_lock_and_ffi_modeled=False, costs_estimated=False)
    return dict(schema=SCHEMA, schema_version=1, source_kind='stateful_workspace',
        source=source, native_manifest=reference, workspace_plan=workspace_plan, target=target,
        schedule=schedule_reference, summary=summary,
        implementations={name:artifact_reference(root/name, 'scar.python-source') for name in
            ('compiler/project_workspace_time.py', 'analysis/workspace_time.py', 'ir/versioned_schedule.py')},
        source_authority='CURRENT_WORKSPACE_PREPARATION_AND_EXACT_NATIVE_SOURCE_REPLAY',
        target_program_executed=False, native_libraries_loaded=False, native_compiled=False,
        solver_invoked=False, performance_measured=False, deployment_authorized=False)


def _write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, value['schema'])


def prepare_workspace_time(native, out):
    directory = Path(native).resolve()
    if directory.is_dir():
        directory /= 'artifact.json'
    reference = artifact_reference(directory, 'scar.native-stateful-workspace-repeat')
    output = Path(out).resolve()
    if output.exists() or output.is_relative_to(directory.parent):
        raise ValueError('fresh independent workspace time output required')
    schedule, source, plan, target = _derive(reference)
    output.mkdir(parents=True)
    schedule_reference = _write(output/'schedule.json', schedule_document(schedule))
    report = _report(reference, source, plan, target, schedule_reference, schedule)
    _write(output/'result.json', report)
    return report


def read_workspace_time(path):
    reference = artifact_reference(Path(path).resolve(), SCHEMA)
    report = read_artifact(reference)
    if (report.get('schema'), report.get('schema_version')) != (SCHEMA, 1):
        raise ValueError('source-replayed workspace time preparation required')
    schedule, source, plan, target = _derive(report['native_manifest'])
    expected = _report(report['native_manifest'], source, plan, target, report['schedule'], schedule)
    if report != expected or semantic_sha256(read_artifact(report['schedule'])) != semantic_sha256(schedule_document(schedule)):
        raise ValueError('workspace time graph differs from complete source/implementation replay')
    read_artifact(reference)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--native'); inputs.add_argument('--check')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check:
        if args.out:
            parser.error('check only reads an existing time preparation')
        report = read_workspace_time(args.check)
    else:
        if not args.out:
            parser.error('native source projection requires --out')
        report = prepare_workspace_time(args.native, args.out)
    print(json.dumps(dict(schedule=report['schedule']['uri'], summary=report['summary'],
        target_program_executed=False, deployment_authorized=False), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
