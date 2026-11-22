"""Publish and replay workspace timing and its source-qualified CPU demands."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, open_artifact, read_artifact
from scar.ir.versioned_schedule import (
    VersionedResourceSchedule, schedule_document, schedule_from_dict, validate_versioned_schedule,
)
from scar.optimization.resource_solver import semantic_sha256


TIME_SCHEMA = 'scar.workspace-time-preparation'
DEMAND_SCHEMA = 'scar.workspace-demand-time-preparation'
NATIVE_SCHEMA = 'scar.native-stateful-workspace-repeat'
_ROOT = Path(__file__).resolve().parents[1]
_OWNER = 'compiler/workspace_schedule.py'
_PRIOR_PRODUCERS = {
    TIME_SCHEMA: ('compiler/project_workspace_time.py',
                  '04adcde2913dc753748022045dedbb957e205c10d3c853120632fc8b5b71747c', 5589),
    DEMAND_SCHEMA: ('compiler/project_workspace_demands.py',
                    '2bda16e673c72a53b57aadf5f9739cfb56890819a3a9115c6873c3f9c1e99ab8', 6025),
}


@dataclass(frozen=True, slots=True)
class _WorkspaceImplementation:
    reference: dict
    metadata: dict
    step_plan: dict
    workspace_plan: dict
    workspace_reference: dict
    cpp: dict
    target: dict


def _inspect_native(reference):
    from scar.codegen.stateful_workspace import inspect_workspace_repeat

    directory = Path(reference['uri']).parent
    metadata, qualified = inspect_workspace_repeat(directory)
    if metadata != read_artifact(reference):
        raise ValueError('original workspace native publication changed during source replay')
    preparation, _, step_plan, workspace_plan, context = qualified
    if read_artifact(metadata['preparation']) != preparation:
        raise ValueError('original workspace preparation reference differs')
    cpp = artifact_reference(directory / metadata['files']['source'], 'scar.cpp-source')
    if cpp['sha256'] != metadata['sha256']['source']:
        raise ValueError('admitted workspace C++ source differs')
    target = next(row['artifact'] for row in context['source_refs'] if row['role'] == 'target')
    read_artifact(target)
    return _WorkspaceImplementation(reference, metadata, step_plan, workspace_plan,
        preparation['workspace_plan'], cpp, target)


def _project_time(native):
    from scar.analysis.workspace_time import derive_workspace_time

    return derive_workspace_time(native.workspace_plan, native_manifest=native.reference,
        native_metadata=native.metadata, cpp_source=native.cpp,
        workspace_reference=native.workspace_reference)


def _report(schema, schedule_reference, summary, implementations, **fields):
    return dict(schema=schema, schema_version=1, source_kind='stateful_workspace',
        schedule=schedule_reference, summary=summary, implementations=implementations, **fields,
        target_program_executed=False, native_libraries_loaded=False, native_compiled=False,
        solver_invoked=False, performance_measured=False, deployment_authorized=False)


def _summary(schedule):
    summary = validate_versioned_schedule(schedule)
    summary['source_task_graph_executable'] = summary.pop('executable')
    return summary


def _time_report(native, schedule_reference, schedule):
    summary = _summary(schedule)
    certificate = schedule.proof['workspace_time']
    summary.update(trip_count=certificate['trip_count'], shared_body_count=int(certificate['trip_count'] > 0),
        iterations_unrolled=0, owned_slot_physical_count=len(certificate['storage_occurrences']),
        normal_completion_only=True, failure_path_timing_modeled=False,
        python_packing_lock_and_ffi_modeled=False, costs_estimated=False)
    implementations = {name: artifact_reference(_ROOT / name, 'scar.python-source') for name in
        (_OWNER, 'analysis/workspace_time.py', 'ir/versioned_schedule.py')}
    return _report(TIME_SCHEMA, schedule_reference, summary, implementations,
        source=native.metadata['preparation'], native_manifest=native.reference,
        workspace_plan=native.workspace_reference, target=native.target,
        source_authority='CURRENT_WORKSPACE_PREPARATION_AND_EXACT_NATIVE_SOURCE_REPLAY')


def _historical_producer(name, digest, size):
    return dict(uri=str(_ROOT / name), schema='scar.python-source', sha256=digest, size_bytes=size)


def _matches(report, expected):
    if report == expected:
        return True
    implementations = expected['implementations']
    if _OWNER not in implementations:
        return False
    name, digest, size = _PRIOR_PRODUCERS[expected['schema']]
    predecessor = {key: value for key, value in implementations.items() if key != _OWNER}
    predecessor[name] = _historical_producer(name, digest, size)
    return report == {**expected, 'implementations': predecessor}


def _read_time(reference, report, *, for_demands=False):
    if (report.get('schema'), report.get('schema_version')) != (TIME_SCHEMA, 1):
        raise ValueError('source-replayed workspace time preparation required')
    native = _inspect_native(report['native_manifest'])
    saved = read_artifact(report['schedule'])
    # Demand refinement already rederives and compares the original graph.
    schedule = schedule_from_dict(saved) if for_demands else _project_time(native)
    expected = _time_report(native, report['schedule'], schedule)
    if (not _matches(report, expected)
            or semantic_sha256(saved) != semantic_sha256(schedule_document(schedule))):
        raise ValueError('workspace time graph differs from complete source/implementation replay')
    read_artifact(native.reference, json_document=False)
    read_artifact(report['schedule'], json_document=False)
    read_artifact(reference, json_document=False)
    return schedule, native


def _project_demands(reference, *, source_rule=None):
    from scar.analysis.scope_demands import derive_workspace_demands

    parent = read_artifact(reference)
    baseline, native = _read_time(reference, parent, for_demands=True)
    schedule = derive_workspace_demands(baseline, native.step_plan, native.metadata,
        native_reference=native.reference, cpp_reference=native.cpp,
        workspace_plan_reference=native.workspace_reference, target=Path(native.target['uri']).parent,
        parent_schedule_reference=parent['schedule'], source_rule=source_rule)
    read_artifact(reference, json_document=False)
    return parent, schedule


def _demand_report(reference, parent, schedule_reference, schedule):
    from scar.analysis.scope_demands import WORKSPACE_PROOF, demand_proof

    summary = _summary(schedule)
    _, proof = demand_proof(schedule, WORKSPACE_PROOF)
    summary.update(exact_thread_demands=sum(row['exact_cpu_id'] is not None for row in proof['occupancies']),
        source_serial_demands=sum(row['units'] == 1 for row in proof['occupancies']),
        unknown_cpu_domains=sum(row['cpu_ids'] is None for row in proof['occupancies']),
        observations_claimed=False)
    rule = proof['source_rule']
    implementations = {'analysis/workspace_source.py': artifact_reference(
        _ROOT / 'analysis/workspace_source.py', 'scar.python-source')}
    if Path(rule['uri']).name == 'workspace_demands.py':
        name = 'compiler/project_workspace_demands.py'
        implementations[name] = _historical_producer(name,
            '5019492315e9966f0082f65f0fa50c6a74beb4395357f71155efe64f75ab0f84', 5229)
        implementations['analysis/workspace_demands.py'] = {**rule, 'schema': 'scar.python-source'}
    else:
        implementations[_OWNER] = artifact_reference(_ROOT / _OWNER, 'scar.python-source')
        implementations['analysis/scope_demands.py'] = {**rule, 'schema': 'scar.python-source'}
    return _report(DEMAND_SCHEMA, schedule_reference, summary, implementations, base_time=reference,
        **{key: parent[key] for key in ('source', 'native_manifest', 'workspace_plan', 'target')},
        source_authority='PARENT_SOURCE_REPLAY_AND_FORWARD_NATIVE_AFFINITY_EFFECTS')


def _read_demands(reference, report):
    from scar.analysis.scope_demands import WORKSPACE_PROOF, demand_proof

    if (report.get('schema'), report.get('schema_version')) != (DEMAND_SCHEMA, 1):
        raise ValueError('source-replayed workspace demand preparation required')
    saved = read_artifact(report['schedule'])
    proof = demand_proof(schedule_from_dict(saved), WORKSPACE_PROOF)
    parent, schedule = _project_demands(report['base_time'], source_rule=None if proof is None else proof[1]['source_rule'])
    expected = _demand_report(report['base_time'], parent, report['schedule'], schedule)
    if not _matches(report, expected) or semantic_sha256(saved) != semantic_sha256(schedule_document(schedule)):
        raise ValueError('workspace demands differ from original source/control/effect replay')
    read_artifact(report['schedule'], json_document=False)
    read_artifact(reference, json_document=False)
    return schedule


def _write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, value['schema'])


def prepare_workspace_time(native, out):
    path = Path(native).resolve()
    if path.is_dir():
        path /= 'artifact.json'
    output = Path(out).resolve()
    if output.exists() or output.is_relative_to(path.parent):
        raise ValueError('fresh independent workspace time output required')
    native = _inspect_native(artifact_reference(path, NATIVE_SCHEMA))
    schedule = _project_time(native)
    read_artifact(native.reference, json_document=False)
    output.mkdir(parents=True)
    schedule_reference = _write(output / 'schedule.json', schedule_document(schedule))
    report = _time_report(native, schedule_reference, schedule)
    _write(output / 'result.json', report)
    return report


def prepare_workspace_demands(time, out):
    reference = artifact_reference(Path(time).resolve(), TIME_SCHEMA)
    output = Path(out).resolve()
    if output.exists():
        raise ValueError('fresh workspace demand output required')
    parent, schedule = _project_demands(reference)
    output.mkdir(parents=True)
    schedule_reference = _write(output / 'schedule.json', schedule_document(schedule))
    report = _demand_report(reference, parent, schedule_reference, schedule)
    _write(output / 'result.json', report)
    return report


def _replay(reference, report):
    if reference['schema'] == TIME_SCHEMA:
        schedule, _ = _read_time(reference, report)
    elif reference['schema'] == DEMAND_SCHEMA:
        schedule = _read_demands(reference, report)
    else:
        raise ValueError('workspace time or demand preparation required')
    return report, schedule


def replay_workspace_schedule(source) -> tuple[dict, VersionedResourceSchedule]:
    if isinstance(source, dict):
        return _replay(source, read_artifact(source))
    return _replay(*open_artifact(Path(source).resolve()))


def read_workspace_time(path):
    return _replay(*open_artifact(Path(path).resolve(), TIME_SCHEMA))[0]


def read_workspace_demands(path):
    return _replay(*open_artifact(Path(path).resolve(), DEMAND_SCHEMA))[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--native')
    source.add_argument('--time-schedule')
    source.add_argument('--check')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check:
        if args.out:
            parser.error('check reads an existing workspace preparation')
        report, _ = replay_workspace_schedule(args.check)
    else:
        if not args.out:
            parser.error('workspace projection requires --out')
        report = (prepare_workspace_time(args.native, args.out) if args.native
                  else prepare_workspace_demands(args.time_schedule, args.out))
    if report['schema'] == TIME_SCHEMA:
        result = dict(schedule=report['schedule']['uri'], target_program_executed=False)
    else:
        result = dict(schedule=report['schedule'], actual_cpu_observation=False)
    print(json.dumps(dict(**result, summary=report['summary'], deployment_authorized=False), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
