"""Refine shared-loop CPU domains from admitted native control and effects."""
import argparse
import json
from pathlib import Path

from scar.analysis.scope_demands import WORKSPACE_PROOF, demand_proof, derive_workspace_demands
from scar.compiler.project_workspace_time import read_workspace_time, SCHEMA as BASE_SCHEMA
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.versioned_schedule import schedule_document, schedule_from_dict, validate_versioned_schedule
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.workspace-demand-time-preparation'


def _derive(reference, *, source_rule=None):
    parent = read_workspace_time(reference['uri'])
    if parent != read_artifact(reference):
        raise ValueError('original workspace time publication changed during demand replay')
    baseline = schedule_from_dict(read_artifact(parent['schedule']))
    preparation = read_artifact(parent['source'])
    plan = read_artifact(preparation['step_plan'])
    metadata = read_artifact(parent['native_manifest'])
    cpp = artifact_reference(Path(parent['native_manifest']['uri']).parent/metadata['files']['source'], 'scar.cpp-source')
    refined = derive_workspace_demands(baseline, plan, metadata,
        native_reference=parent['native_manifest'], cpp_reference=cpp,
        workspace_plan_reference=parent['workspace_plan'], target=Path(parent['target']['uri']).parent,
        parent_schedule_reference=parent['schedule'], source_rule=source_rule)
    read_artifact(reference)
    return parent, refined


def _report(reference, parent, schedule_reference, schedule):
    summary = validate_versioned_schedule(schedule)
    summary['source_task_graph_executable'] = summary.pop('executable')
    _, proof = demand_proof(schedule, WORKSPACE_PROOF)
    summary.update(exact_thread_demands=sum(row['exact_cpu_id'] is not None for row in proof['occupancies']),
        source_serial_demands=sum(row['units'] == 1 for row in proof['occupancies']),
        unknown_cpu_domains=sum(row['cpu_ids'] is None for row in proof['occupancies']),
        observations_claimed=False)
    root = Path(__file__).resolve().parents[1]
    wrapper = 'compiler/project_workspace_demands.py'
    source_rule = proof['source_rule']
    implementations = {
        wrapper: artifact_reference(root / wrapper, 'scar.python-source'),
        'analysis/workspace_source.py': artifact_reference(root / 'analysis/workspace_source.py', 'scar.python-source'),
        'analysis/scope_demands.py': {**source_rule, 'schema': 'scar.python-source'},
    }
    if Path(source_rule['uri']).name == 'workspace_demands.py':
        # A complete replay already verified this exact predecessor rule.
        implementations['analysis/workspace_demands.py'] = implementations.pop('analysis/scope_demands.py')
        implementations[wrapper] = dict(uri=str(root / wrapper), schema='scar.python-source',
            sha256='5019492315e9966f0082f65f0fa50c6a74beb4395357f71155efe64f75ab0f84', size_bytes=5229)
    return dict(schema=SCHEMA, schema_version=1, base_time=reference,
        **{key:parent[key] for key in ('source_kind','source','native_manifest','workspace_plan','target')},
        schedule=schedule_reference, summary=summary,
        implementations=implementations,
        source_authority='PARENT_SOURCE_REPLAY_AND_FORWARD_NATIVE_AFFINITY_EFFECTS',
        target_program_executed=False, native_libraries_loaded=False, native_compiled=False,
        solver_invoked=False, performance_measured=False, deployment_authorized=False)


def _write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, value['schema'])


def prepare_workspace_demands(time, out):
    reference = artifact_reference(Path(time).resolve(), BASE_SCHEMA)
    output = Path(out).resolve()
    if output.exists():
        raise ValueError('fresh workspace demand output required')
    parent, schedule = _derive(reference)
    output.mkdir(parents=True)
    schedule_reference = _write(output/'schedule.json', schedule_document(schedule))
    report = _report(reference, parent, schedule_reference, schedule)
    _write(output/'result.json', report)
    return report


def read_workspace_demands(path):
    reference = artifact_reference(Path(path).resolve(), SCHEMA)
    report = read_artifact(reference)
    if (report.get('schema'), report.get('schema_version')) != (SCHEMA, 1):
        raise ValueError('source-replayed workspace demand preparation required')
    saved = read_artifact(report['schedule'])
    proof = demand_proof(schedule_from_dict(saved), WORKSPACE_PROOF)
    parent, schedule = _derive(report['base_time'], source_rule=None if proof is None else proof[1]['source_rule'])
    expected = _report(report['base_time'], parent, report['schedule'], schedule)
    if report != expected or semantic_sha256(saved) != semantic_sha256(schedule_document(schedule)):
        raise ValueError('workspace demands differ from original source/control/effect replay')
    read_artifact(reference)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--time-schedule'); source.add_argument('--check')
    parser.add_argument('--out'); args = parser.parse_args(argv)
    if args.check:
        if args.out:
            parser.error('check only reads an existing demand preparation')
        report = read_workspace_demands(args.check)
    else:
        if not args.out:
            parser.error('demand refinement requires --out')
        report = prepare_workspace_demands(args.time_schedule, args.out)
    print(json.dumps(dict(schedule=report['schedule'], summary=report['summary'],
        actual_cpu_observation=False, deployment_authorized=False), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
