"""Observe bounded short-count lifecycle windows on the admitted mapped C++20 route."""

import os
from pathlib import Path
import shutil

from scar.codegen.cpu_native_route_package import (
    _fixed_grid_route_replay, _output_guards, verify_cpu_fixed_grid_native_route_package,
)
from scar.compiler.calibrate_array_entry import _write
from scar.compiler.cpu_fixed_grid_mapped_input_study import _input_files
from scar.compiler.observe_cpu_fixed_grid_input_counters import SCHEMA as COUNTERS
from scar.compiler.observe_cpu_fixed_grid_native_lifecycle import _capture, _project, _raw, _ref, _sample
from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.cpu-fixed-grid-short-native-probe'
_ORACLE = 'scar.cpu-fixed-grid-short-python-probe'
_ROUTE = 'PROVIDER_MAP'
COUNTS = (2, 8)


def _need(condition, reason):
    if not condition:
        raise ValueError('fixed-grid short native probe: ' + reason)


def _binding(package_ref, oracle_ref):
    from scar.compiler.cpu_fixed_grid_short_python_probe import verify_short_python

    _need(oracle_ref['schema'] == _ORACLE, 'exact independent short-count Python oracle required')
    with _fixed_grid_route_replay():
        package = verify_cpu_fixed_grid_native_route_package(package_ref)
        oracle = verify_short_python(oracle_ref)
    _need(package['selected_runtime'] == _ROUTE and package['inputs_must_remain_stable'] is True
          and package['deployment_authorized'] is False and oracle['package'] == package_ref
          and oracle['source'] == package['source'] and oracle['target'] == package['hardware']
          and oracle['cpu_ids'] == package['cpu_ids'] and oracle['working_directory'] == package['working_directory']
          and oracle['environment_sha256'] == package['environment_sha256']
          and oracle['step_counts'] == list(COUNTS) and set(oracle['cases']) == {str(count) for count in COUNTS},
          'mapped native route and Python oracle scopes differ')
    plan = read_artifact(package['plan'])
    proof_ref = plan['proof']
    abi = read_artifact(read_artifact(proof_ref)['abi'])
    argv = abi['argv']
    step, output = argv['steps'] - 1, argv['output'] - 1
    _need(argv['count'] == len(package['argv']) and output == package['publication']['output_index']
          and step != output and abi['domain']['normal_completion_only'] is True
          and abi['domain']['step_count_min'] <= min(COUNTS) <= max(COUNTS) <= abi['domain']['step_count_max'],
          'short counts or publication slot are outside the exact source ABI')
    original = read_artifact(read_artifact(package['owned_copy_delivery'])['observation'])['commands']['ORIGINAL_PYTHON']
    prefix = original[:-argv['count']]
    for count in COUNTS:
        row = oracle['cases'][str(count)]
        value = row['oracle']
        command = value['command']
        args = command[len(prefix):]
        expected = list(package['argv'])
        expected[step] = str(count)
        expected[output] = args[output]
        _need(command[:len(prefix)] == prefix and args == expected == row['argv'] and Path(args[output]).is_absolute()
              and args[output] != package['publication']['path']
              and value['mode'] == package['publication']['expected_mode']
              and value['file']['size_bytes'] == package['publication']['expected_file']['size_bytes']
              and not _raw(value['stderr']), 'short-count original argv, mode or final file oracle differs')
    return package, oracle, proof_ref, step, output


def _sibling(package, proof_ref, root, compiler, counters_ref, *, build):
    route = package['routes'][_ROUTE]
    material = dict(project=route['project'], native_executable=route['executable'])
    if counters_ref is None:
        sibling = _project(root/'native-probe', material, compiler, build=build, mapped_proof=proof_ref)
        return sibling, False
    _need(counters_ref['schema'] == COUNTERS and Path(counters_ref['uri']).name == 'result.json',
          'exact retained counter sibling authority required')
    counters = read_artifact(counters_ref)
    retained = counters['routes'][_ROUTE]
    _need(counters['mapped_plan'] == package['plan'] and counters['observation'] == package['observation']
          and counters['control_delivery'] == package['owned_copy_delivery']
          and counters['source'] == package['source'] and counters['target'] == package['hardware']
          and counters['cpu_ids'] == package['cpu_ids'] and counters['working_directory'] == package['working_directory']
          and counters['argv'] == package['argv'] and counters['environment_sha256'] == package['environment_sha256']
          and counters['compiler'] == compiler and counters['clean_benchmark'] is False
          and retained['original_project'] == route['project'] and retained['original_executable'] == route['executable'],
          'retained marked C++ executable does not cover the same mapped route and context')
    sibling = _project(Path(counters_ref['uri']).parent/_ROUTE, material, compiler, build=False,
                       usage=True, mapped_proof=proof_ref)
    _need(sibling == retained['sibling'], 'retained diagnostic C++ source, binary or loader changed')
    return sibling, True


def _report(package_ref, oracle_ref, root, count, compiler, counters_ref, *, capture=False):
    package, oracle, proof_ref, step, output = _binding(package_ref, oracle_ref)
    _need(type(count) is int and 1 <= count <= 12 and root.resolve() == root
          and not root.is_relative_to(Path(package_ref['uri']).parent)
          and not root.is_relative_to(Path(oracle_ref['uri']).parent)
          and all(not root.is_relative_to(route['project_root']) for route in package['routes'].values())
          and all(not root.is_relative_to(Path(row['path']).parent) for row in package['file_inputs']),
          'bounded samples and independent output directory required')
    _raw(compiler)
    if capture:
        _need(set(package['cpu_ids']) <= os.sched_getaffinity(0), 'the exact launch CPU cohort is unavailable')
        root.mkdir(parents=True, mode=0o700, exist_ok=False)
    sibling, usage = _sibling(package, proof_ref, root, compiler, counters_ref, build=capture)
    owner = dict(working_directory=package['working_directory'], python_provider_environment=package['source_environment'])
    expected_mode = package['publication']['expected_mode']
    input_snapshots = _input_files(package)
    cases, ordered = {}, []
    outputs = root/'private-output-slots'
    if capture:
        outputs.mkdir(mode=0o700)
    for steps in COUNTS:
        expected = oracle['cases'][str(steps)]['oracle']
        check = dict(validated_source_stdout=expected['stdout'], validated_final_tensor=expected['file'])
        rows = []
        for index in range(count):
            name = f'{steps:02d}-sample-{index:02d}'
            publish = outputs/f'{name}.f32'
            args = list(package['argv'])
            args[step], args[output] = str(steps), str(publish)
            command = [sibling['diagnostic_launcher']['uri'], *args]
            folder = root/name
            if capture:
                _need(not _output_guards(publish, expected_mode), 'private output is occupied or its mode premise changed')
                _capture(folder, command, owner, publish)
            row = _sample(folder, command, check, index, expected_mode, usage=usage)
            markers = row['markers_ns']
            first, last = markers['RECURRENCE_BEGIN'], markers['RECURRENCE_RETURN']
            row['windows_ns'] = dict(before_recurrence=first - row['launcher_start_ns'],
                recurrence=last - first, after_recurrence=row['process_reaped_ns'] - last)
            _need(all(value > 0 for value in row['windows_ns'].values())
                  and _raw(row['stderr']) == _raw(expected['stderr']), 'phase clock or original stderr differs')
            rows.append(row)
            ordered.append(row)
        cases[str(steps)] = dict(steps=steps, oracle=expected, samples=rows)
    _need(_input_files(package) == input_snapshots and not any(outputs.iterdir())
          and all(left['process_reaped_ns'] <= right['launcher_start_ns'] for left, right in zip(ordered, ordered[1:])),
          'input identity, privately staged output or sample order changed')
    return dict(schema=SCHEMA, schema_version=1, status='EXACT_SHORT_NATIVE_LIFECYCLE_OBSERVED',
        package=package_ref, python_probe=oracle_ref, project=package['selected_project'],
        native_executable=package['routes'][_ROUTE]['executable'], mapped_proof=proof_ref,
        source=package['source'], target=package['hardware'], cpu_ids=package['cpu_ids'],
        working_directory=package['working_directory'], environment_sha256=package['environment_sha256'],
        argv=package['argv'], file_inputs=package['file_inputs'], input_snapshots=input_snapshots,
        step_counts=list(COUNTS), cases=cases, sample_count=count, compiler=compiler,
        counters_ref=counters_ref, sibling=sibling, clock='LINUX_CLOCK_MONOTONIC_NS',
        exact_exit_streams_file_and_mode_verified=True, diagnostic_only=True, clean_benchmark=False,
        instrumented_rusage_at_four_cuts=usage, physical_duration_bound_proven=False,
        deployment_authorized=False, target_executed=True,
        limitations=['BEFORE_RECURRENCE_INCLUDES_LOADER_AND_INPUT_INGRESS',
                     'AFTER_RECURRENCE_INCLUDES_PUBLICATION_AND_PROBE_RECORDER_BEFORE_REAP',
                     'PER_KERNEL_AND_HARDWARE_EVENTS_NOT_MEASURED'])


def observe_short_native(package_ref, oracle_ref, out, *, samples=3, counters_ref=None):
    """Measure three independent native lifecycle windows per declared short count."""
    out = Path(out).absolute()
    _need(out.resolve() == out and not out.exists() and not out.is_symlink(), 'fresh canonical output required')
    if counters_ref is not None:
        compiler = read_artifact(counters_ref)['compiler']
    else:
        tool = shutil.which('g++')
        _need(tool is not None, 'C++20 compiler required')
        compiler = _ref(Path(tool).resolve(strict=True), 'application/x-executable')
    report = _report(package_ref, oracle_ref, out, samples, compiler, counters_ref, capture=True)
    _write(out/'result.json', report)
    return report


def verify_short_native(reference):
    """Replay exact original outputs, source modifications and native clock records cold."""
    saved = read_artifact(reference)
    path = Path(reference['uri'])
    _need(reference['schema'] == SCHEMA and path.name == 'result.json' and path.resolve(strict=True) == path,
          'exact short native lifecycle receipt required')
    expected = _report(saved['package'], saved['python_probe'], path.parent, saved['sample_count'],
                       saved['compiler'], saved['counters_ref'])
    _need(saved == expected, 'saved native observations differ from cold replay')
    return saved
