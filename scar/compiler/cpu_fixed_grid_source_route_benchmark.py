"""Measure the exact original fixed-grid CLI against its admitted mapped C++20 route."""

import argparse
import json
import os
from pathlib import Path
from statistics import pvariance

from scar.codegen.cpu_native_route_package import (
    FIXED_GRID_SCHEMA as PACKAGE, _output_guards, verify_cpu_fixed_grid_native_route_package,
)
from scar.compiler.calibrate_array_entry import _write
from scar.compiler.cpu_fixed_grid_mapped_input_study import _capture, _input_files, _rows
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.paired_entry_selection import compare_entry_pairs, replay_entry_pairs


SCHEMA = 'scar.cpu-fixed-grid-source-route-benchmark'
_ORIGINAL, _MAPPED = 'ORIGINAL_PYTHON', 'PROVIDER_MAP'
_WARMUP, _PAIRS = 1, 3


def _need(condition, reason):
    if not condition:
        raise ValueError('fixed-grid source route benchmark: ' + reason)


def _raw(reference):
    return read_artifact(reference, json_document=False)


def _scope(package_ref):
    package = verify_cpu_fixed_grid_native_route_package(package_ref)
    _need(package['selected_runtime'] == _MAPPED and package['deployment_authorized'] is False
          and package['inputs_must_remain_stable'] is True,
          'the mapped C++20 route must be explicitly selected under the stable-input premise')
    delivery = read_artifact(package['owned_copy_delivery'])
    original = read_artifact(delivery['observation'])
    command = original['commands'][_ORIGINAL]
    argv, cpus = package['argv'], package['cpu_ids']
    _need(command[:3] == ['/usr/bin/taskset', '-c', ','.join(map(str, cpus))]
          and command[4:] == ['-B', package['source']['uri'], *argv]
          and original['argv'] == argv and original['cpu_ids'] == original['original_cpu_ids'] == cpus
          and original['working_directory'] == package['working_directory']
          and original['python_provider_environment'] == package['source_environment']
          and delivery['validated_source_stdout'] == package['stdout']
          and delivery['validated_final_tensor'] == package['publication']['expected_file'],
          'retained original command or published oracle crosses the package scope')
    route = package['routes'][_MAPPED]
    commands = {_ORIGINAL: command, _MAPPED: [route['launcher']['uri'], *argv]}
    return package, commands


def _report(package_ref, package, commands, out, *, capture=False, archived=None):
    output = Path(package['publication']['path'])
    expected_mode = package['publication']['expected_mode']
    expected_stdout = _raw(package['stdout'])
    expected_output = _raw(package['publication']['expected_file'])
    inputs = _input_files(package)
    runs = []
    for pair in range(_WARMUP + _PAIRS):
        for mode in ((_ORIGINAL, _MAPPED) if pair % 2 == 0 else (_MAPPED, _ORIGINAL)):
            folder = out / f'{pair:02d}-{mode}'
            if capture:
                _need(not _output_guards(output, expected_mode), 'exact output is occupied or its permissions changed')
                _capture(folder, commands[mode], dict(working_directory=package['working_directory'],
                         environment=package['source_environment']), output)
            row = _rows(folder, command=commands[mode], mode=mode, pair=pair, expected_mode=expected_mode, timed=True)
            _need(_raw(row['stdout']) == expected_stdout and _raw(row['published']) == expected_output,
                  'source stdout or complete FP32 output differs from the retained original')
            runs.append(row)
    _need(inputs == _input_files(package), 'source input file identity or bytes changed during capture')
    rows = [dict(pair=row['pair'], mode=row['mode'], wall_ns=row['wall_ns'], returncode=row['returncode'],
                 stdout_sha256=row['stdout']['sha256'], stderr_sha256=row['stderr']['sha256']) for row in runs]
    context = dict(incumbent=_ORIGINAL, challenger=_MAPPED, warmup_pairs=_WARMUP, measurement_pairs=_PAIRS)
    selection = compare_entry_pairs(rows, **context) if archived is None else replay_entry_pairs(rows, archived, **context)
    return dict(schema=SCHEMA, schema_version=1, status='DIRECT_SOURCE_AND_MAPPED_NATIVE_COMPARED',
        package=package_ref, original_delivery=package['owned_copy_delivery'], source=package['source'],
        target=package['hardware'], native_project=package['selected_project'],
        native_executable=package['routes'][_MAPPED]['executable'], commands=commands,
        argv=package['argv'], cpu_ids=package['cpu_ids'], steps=package['steps'],
        working_directory=package['working_directory'], environment_sha256=package['environment_sha256'],
        publication=package['publication'], file_inputs=package['file_inputs'], input_snapshots=inputs,
        runs=runs, selection=selection,
        variance_ns2={name: pvariance(stats['samples_ns']) for name, stats in selection['metrics'].items()},
        timing_scope='DIRECT_FRESH_PROCESSES_FROM_LAUNCH_THROUGH_EXIT_INCLUDING_FILE_IO',
        exact_exit_streams_file_and_mode_verified=True, input_stability_is_a_launch_premise=True,
        path_state_at_start='ABSENT_EXACT_PUBLICATION_PATH', physical_time_bound=False,
        whole_entry_equivalence_proven=False, deployment_authorized=False, target_executed=True)


def measure(source_package_ref, out):
    """Run four alternating pairs on the exact package cohort and output path."""
    package, commands = _scope(source_package_ref)
    out = Path(out).absolute()
    output = Path(package['publication']['path'])
    protected = [output.parent, Path(source_package_ref['uri']).parent,
                 *(Path(route['project_root']) for route in package['routes'].values())]
    _need(out.resolve() == out and not any(out.is_relative_to(path) for path in protected)
          and set(package['cpu_ids']) <= os.sched_getaffinity(0)
          and not _output_guards(output, package['publication']['expected_mode']),
          'available exact CPU cohort, fresh publication and independent archive directory required')
    out.mkdir(parents=True, exist_ok=False)
    report = _report(source_package_ref, package, commands, out, capture=True)
    _write(out/'result.json', report)
    return report


def verify_cpu_fixed_grid_source_route_benchmark(reference):
    """Cold-replay the commands, scope, saved output bytes and recorded comparison."""
    path = Path(reference['uri'])
    _need(reference['schema'] == SCHEMA and path.name == 'result.json' and path.resolve(strict=True) == path,
          'exact direct source route benchmark receipt required')
    saved = read_artifact(reference)
    package, commands = _scope(saved['package'])
    _need(saved == _report(saved['package'], package, commands, path.parent, archived=saved['selection']),
          'saved benchmark differs from its retained native route, outputs or timing rows')
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--package', type=Path)
    mode.add_argument('--check', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.check is not None:
        report = verify_cpu_fixed_grid_source_route_benchmark(artifact_reference(args.check.resolve(strict=True), SCHEMA))
    else:
        if args.out is None:
            parser.error('--out is required with --package')
        report = measure(artifact_reference(args.package.resolve(strict=True), PACKAGE), args.out)
    print(json.dumps(report['selection'], sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
