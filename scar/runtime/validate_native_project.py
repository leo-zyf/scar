"""Verify a native project and compare its outputs with explicit reference files."""

import argparse
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen.aten_provider import _dependencies
from scar.codegen.native_provider_bundle import _content
from scar.compiler.tensor_execution import (
    replay_native_comparison, replay_operation_costs, replay_recurrence_placement, scheduled_tensor_body,
)
from scar.ir.implementation_plan import problem_from_dict
from scar.ir.program import codec


def _replay_recurrence_selection(manifest, member, execution, observation, schedule):
    def read(relative):
        return json.loads(member(relative).read_bytes())

    alternatives = manifest['alternatives']
    admitted = {key for key, item in alternatives.items() if item['within_budget']}
    mode = execution['placement']['mode']
    if (mode not in admitted or alternatives[mode]['body_estimate_ns'] != schedule['makespan_ns']
            or alternatives[mode]['peak_tensor_storage_bytes'] != execution['peak_tensor_storage_bytes']
            or any(item['within_budget'] != (item['peak_tensor_storage_bytes'] <= manifest['tensor_budget_bytes'])
                   for item in alternatives.values())):
        raise ValueError('recurrence selection disagrees with the admitted storage or body cost')
    expected_mode = mode
    sampled_comparison = False
    if admitted == {'SERIAL', 'READY_DAG'}:
        alternate_mode, = admitted - {mode}
        archive = 'validation/alternative/'
        other_obs = read(archive + 'calibration.json')
        if ((other_obs['inputs'], other_obs['repeats'], other_obs['outputs']) !=
                (observation['inputs'], observation['repeats'], observation['outputs'])
                or other_obs['placement']['mode'] != alternate_mode):
            raise ValueError('alternative calibration has a different input or execution scope')
        other_problem = read(archive + 'problem.json')
        other_schedule = read(archive + 'schedule.json')
        replay_operation_costs(problem_from_dict(other_problem), execution, other_obs,
            dict(path='validation/calibration.json', **_content(member(archive + 'calibration.json'))))
        scheduled_tensor_body(other_problem, other_schedule, execution)
        if alternatives[alternate_mode]['body_estimate_ns'] != other_schedule['makespan_ns']:
            raise ValueError('alternative estimate differs from its physical schedule')
        if alternatives['READY_DAG']['body_estimate_ns'] < alternatives['SERIAL']['body_estimate_ns']:
            comparison = read('validation/comparison.json')
            expected_mode = replay_native_comparison(comparison, observation['outputs'])
            sampled_comparison = True
        else:
            expected_mode = 'SERIAL'
    if expected_mode != mode or sampled_comparison != manifest['full_entry_performance_measured']:
        raise ValueError('published recurrence does not implement the replayed native selection')


def verify_native_project(directory, manifest_sha256):
    root = Path(directory).resolve(strict=True)
    path = root / 'artifact.json'
    if _content(path)['sha256'] != manifest_sha256:
        raise ValueError('native project manifest differs from its declared digest')
    manifest = json.loads(path.read_bytes())
    files = manifest['files']

    def member(relative):
        path = root / relative
        if relative not in files or path.resolve(strict=True) != path or not path.is_relative_to(root):
            raise ValueError(f'not a local native project member: {relative}')
        return path

    for name, expected in files.items():
        if _content(member(name)) != expected:
            raise ValueError(f'native project member differs: {name}')
    graph = codec.loads(member('plan/program.json').read_text())
    execution = json.loads(member('plan/execution.json').read_text())
    problem = problem_from_dict(json.loads(member('plan/problem.json').read_text()))
    reference = execution['program_graph']
    if reference != dict(path='plan/program.json', **files['plan/program.json']):
        raise ValueError('execution plan does not reference its original program graph')
    known = {operation.id.wire for operation in graph.operations}
    if any(ref.get('operation') not in known for row in problem.operations if row.id != '$return'
           for ref in row.source_refs):
        raise ValueError('physical operation lacks a source program graph node')
    if problem.source_refs != (reference,) or execution['target']['sha256'] != problem.target.sha256:
        raise ValueError('shared physical problem differs from its graph or target binding')
    replayed = False
    if 'plan/schedule.json' in files:
        schedule = json.loads(member('plan/schedule.json').read_bytes())
        calibration_path = 'validation/calibration.json'
        observation = json.loads(member(calibration_path).read_bytes())
        recurrence = execution['schema'] == 'scar.aten-recurrence-plan'
        body = execution if recurrence else json.loads(member('plan/recipe.json').read_bytes())
        replay_operation_costs(problem, body, observation, dict(path=calibration_path, **files[calibration_path]))
        lowered = scheduled_tensor_body(json.loads(member('plan/problem.json').read_bytes()), schedule, body)
        if recurrence:
            from scar.codegen.aten_recurrence_source import render_recurrence

            placement = execution['placement']
            if manifest.get('source_entry_inferred'):
                boundary = json.loads(member('plan/source-entry.json').read_bytes())
                if (boundary['method'] != manifest['method_provider']['method']
                        or boundary['step_size'] != execution['step_size']
                        or boundary['state_input_index'] != execution['state_input_index']):
                    raise ValueError('recurrence differs from its admitted source-entry binding')
            rhs_recipe = json.loads(member('plan/rhs-recipe.json').read_bytes())
            if placement != observation['placement'] or (placement['mode'], placement['cpus']) != (
                    manifest['rhs_mode'], manifest['cpus']):
                raise ValueError('recurrence calibration and emitted placement differ')
            rhs_problem, rhs_schedule = ((json.loads(member(name).read_bytes()) for name in
                ('plan/rhs-problem.json', 'plan/rhs-schedule.json')) if placement['mode'] == 'READY_DAG' else (None, None))
            if rhs_schedule is not None and execution['rhs_schedule']['sha256'] != files['plan/rhs-schedule.json']['sha256']:
                raise ValueError('parallel RHS schedule differs from the archived selection')
            replay_recurrence_placement(problem, placement, rhs_recipe, rhs_problem, rhs_schedule)
            _replay_recurrence_selection(manifest, member, execution, observation, schedule)
            generated = render_recurrence(lowered, rhs_mode=placement['mode'], cpus=placement['cpus'],
                lanes=placement['lanes'], operation_order=placement['operation_order'])
        elif execution['schema'] == 'scar.native-tensor-entry-plan':
            from scar.codegen.aten_export_source import render_exported_entry

            cpu, = execution['cpu_ids']
            if execution['cpu_ids'] != observation['cpu_ids'] or any(option.parameters['cpu_ids'] != [cpu]
                    for task in problem.operations if task.id != '$return' for option in task.options):
                raise ValueError('inference calibration and emitted placement differ')
            generated = render_exported_entry(lowered, execution['bindings'], lowered['release_after'], cpu)
        else:
            raise ValueError('native schedule is missing a source replay implementation')
        if any(member(name).read_text() != source for name, source in generated.items()):
            raise ValueError('generated C++ differs from the verified native schedule lowering')
        replayed = True
    executable = member(manifest['executable'])
    current = _dependencies(subprocess.check_output(['/usr/bin/ldd', str(executable)], text=True, env={}))
    if current != manifest['dependencies'] or any(name.startswith(('libpython', 'libtorch_python')) for name in current):
        raise ValueError('runtime dependency resolution differs from the checked native project')
    locks = json.loads(member('plan/dependencies.json').read_text())
    for lock in locks['libraries'] if type(locks) is dict else locks:
        library = Path(lock['path'])
        if current.get(library.name) != str(library.resolve()) or _content(library) != {
                key: lock[key] for key in ('sha256', 'size_bytes')}:
            raise ValueError(f'native operator library differs from the build lock: {library.name}')
    return root, dict(graph_regions=len(graph.regions), graph_operations=len(graph.operations),
                      physical_activities=len(problem.operations), native_dependencies_checked=True,
                      executed_schedule_replayed=replayed)


def validate_native_project(directory, manifest_sha256, *, arguments=(), references=()):
    root, receipt = verify_native_project(directory, manifest_sha256)
    report = dict(schema='scar.native-project-validation', schema_version=1,
                  manifest_sha256=manifest_sha256, **receipt)
    if references:
        expected = [_content(Path(path).resolve(strict=True)) for path in references]
        with TemporaryDirectory(prefix='scar-native-validation-') as temporary:
            outputs = [Path(temporary) / str(i) for i in range(len(references))]
            subprocess.run([str(root / 'launch.sh'), *arguments, *map(str, outputs)], env={},
                           check=True, capture_output=True)
            observed = [_content(path) for path in outputs]
        if observed != expected:
            raise ValueError('native program output differs from its reference bytes')
        report['bitwise_output_receipts'] = observed
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar validate-native-project', description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--reference', action='append', default=[])
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    receipt = validate_native_project(args.project, args.manifest_sha256, arguments=arguments,
                                      references=args.reference)
    print(json.dumps(receipt, sort_keys=True))
    return 0
