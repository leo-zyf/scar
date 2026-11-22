"""Measure native file and stdout publication using exact retained C++ writers."""

import argparse
import json
import math
import os
from pathlib import Path
from struct import Struct
import subprocess
from time import CLOCK_MONOTONIC, clock_gettime_ns

from scar.analysis.replay_transaction import replay_transaction
from scar.codegen.cpu_catalog_runtime_package import _loader, _signature
from scar.codegen.native_cpu_arena_reuse import _path
from scar.codegen.native_project import build_native_project
from scar.compiler.calibrate_array_entry import _write
from scar.compiler.cpu_array_native_route_trial import SCHEMA as COMPARISON, verify_cpu_array_native_route_observation
from scar.compiler.cpu_entry_replay_adapters import _match_json_report
from scar.ir.execution_recipe import _canonical, artifact_reference, read_artifact


SCHEMA = 'scar.cpu-array-native-route-boundary-calibration'
_SOURCE = 'src/main.cpp'
_HEADER = 'include/scar_generated/scar_boundary_calibration.hpp'
_ROUTES = ('retained_process', 'grouped_openmp')
_MARKERS = ('PUBLICATION_BEGIN', 'FILE_OPEN_BEGIN', 'FILE_CLOSE_COMPLETE', 'STDOUT_BEGIN', 'STDOUT_SUBMIT_END')
_TIMES = Struct('=5Q')
_TIMESTAMPS = 'publication-times.bin'
_WORKER_ROLES = ('norm', 'split_difference', 'norm_error', 'worker_wall', 'worker_cpu', 'pid', 'peak_rss')
_OWNER_ROLES = ('elapsed', 'worker_cpu', 'norm_error')
_RAW = 'application/octet-stream'

_CLOCK_SOURCE = b'''#pragma once
#include <array>
#include <cstdint>
#include <cstdio>
#include <time.h>

namespace scar::boundary_calibration {
enum Marker { PUBLICATION_BEGIN, FILE_OPEN_BEGIN, FILE_CLOSE_COMPLETE, STDOUT_BEGIN, STDOUT_SUBMIT_END };
static std::array<std::uint64_t, 5> samples{};
static_assert(sizeof(samples) == 40);

inline void stamp(Marker marker) {
    timespec timestamp{};
    if (clock_gettime(CLOCK_MONOTONIC, &timestamp) == 0) {
        samples[marker] = std::uint64_t(timestamp.tv_sec) * 1000000000ULL + std::uint64_t(timestamp.tv_nsec);
    }
}

inline bool save() {
    auto* stream = std::fopen("publication-times.bin", "wbx");
    if (!stream) return false;
    const auto written = std::fwrite(samples.data(), sizeof(samples), 1, stream);
    const auto closed = std::fclose(stream);
    return written == 1 && closed == 0;
}
}
'''


def _require(ok, detail):
    if not ok:
        raise ValueError('CPU array native route boundary calibration: ' + detail)


def _raw(reference):
    return read_artifact(reference, json_document=False)


def _ref(path, schema=_RAW):
    _require(path.resolve(strict=True) == path and path.is_file(), 'expected a local probe file')
    return artifact_reference(path, schema)


def _double(number):
    _require(type(number) in (int, float) and math.isfinite(number), 'expected finite binary64 fixture')
    return float(number).hex()


def _integer(number):
    _require(type(number) is int and 0 <= number < 1 << 63, 'expected nonnegative signed 64-bit fixture')
    return str(number)


def _fixture(source, recipe, selected, count):
    report = json.loads(_raw(source))
    bindings = recipe['report_bindings']
    by_owner, by_worker = {}, {}
    for binding in bindings:
        path, role = binding['path'], binding['role']
        if len(path) == 1 and role in (*_OWNER_ROLES, 'pid_list'):
            _require(role not in by_owner, 'duplicate typed owner report role')
            by_owner[role] = path[0]
        if len(path) == 3 and path[:2] == [selected['results'], 0] and role in _WORKER_ROLES:
            _require(role not in by_worker, 'duplicate typed task report role')
            by_worker[role] = path[2]
    _require(by_owner.keys() == {*_OWNER_ROLES, 'pid_list'} and by_worker.keys() == set(_WORKER_ROLES)
             and by_worker['pid'] == selected['worker_pid'] and by_owner['pid_list'] == selected['pid_list'],
             'report does not cover the generated Result and WorkerResult fixture slots')
    tasks = report[selected['results']]
    pids = report[by_owner['pid_list']]
    _require(len(tasks) == count and all(task[selected['ordinal']] == index for index, task in enumerate(tasks))
             and pids == sorted({task[by_worker['pid']] for task in tasks}),
             'retained result is not the exact ordered task fixture')
    # Preserve the exact linked team runtime under ELF --as-needed without invoking its work.
    output = ['[[gnu::used]] static const auto scar_boundary_retained_execute = &execute;']
    output.extend(f'result.{role} = {_double(report[by_owner[role]])};' for role in _OWNER_ROLES)
    output.append('result.worker_pids = {' + ', '.join(map(_integer, pids)) + '};')
    output.append(f'result.records.resize({count});')
    for index, task in enumerate(tasks):
        for role in _WORKER_ROLES:
            value = task[by_worker[role]]
            payload = _integer(value) if role in ('pid', 'peak_rss') else _double(value)
            output.append(f'result.records[{index}].{role} = {payload};')
    return output


def _render(source, route_sites, assignments):
    ranges = route_sites['source_ranges']
    replaced = ranges['replace_execute']
    begin, end = replaced['byte_start'], replaced['byte_end']
    _require(0 < begin < end < len(source)
             and source[begin:end].lstrip().startswith(b'const auto status = execute(result);')
             and source[end:].lstrip().startswith(b'const auto& path ='),
             'exact generated execute and publication boundaries required')

    def line_start(offset):
        start = source.rfind(b'\n', 0, offset) + 1
        _require(source[start:offset].strip(b' ') == b'', 'publication marker is not at a statement line')
        return start

    begin, end = line_start(begin), line_start(end)
    fixture = ''.join(f'        {assignment}\n' for assignment in assignments).encode('ascii')
    points = {}
    for name in _MARKERS:
        offset = line_start(route_sites['sites'][name]['byte_offset'])
        statement = f'        scar::boundary_calibration::stamp(scar::boundary_calibration::{name});\n'
        points.setdefault(offset, []).append(statement)
    last = line_start(route_sites['sites']['STDOUT_SUBMIT_END']['byte_offset'])
    _require(end <= min(points) < last < len(source), 'marker crosses the publication suffix')
    points[last].append('        if (!scar::boundary_calibration::save()) return 73;\n')
    rendered = [b'#include "scar_boundary_calibration.hpp"\n', source[:begin], fixture]
    previous = end
    for offset, statements in sorted(points.items()):
        rendered.extend((source[previous:offset], ''.join(statements).encode('ascii')))
        previous = offset
    rendered.append(source[previous:])
    return b''.join(rendered)


def _project(directory, route_sites, runtime_ref, fixture, recipe, count, compiler, environment, *, build):
    runtime = read_artifact(runtime_ref)
    binary = runtime['binary']
    members = runtime['files']
    _require(members[_SOURCE] == route_sites['source'] and members[binary] == route_sites['executable']
             and _HEADER not in members and runtime['cxx_standard'] == 20,
             'component source or baseline binary differs from exact route')
    chart = _fixture(fixture['stdout'], recipe, route_sites['output_fields'], count)
    inputs = {name: _raw(reference) for name, reference in members.items() if name != binary}
    inputs[_SOURCE] = _render(inputs[_SOURCE], route_sites, chart)
    inputs[_HEADER] = _CLOCK_SOURCE
    if build:
        for name, content in inputs.items():
            path = _path(directory, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        libraries = [_path(directory, name) for name in inputs if name.startswith('provider/objects/')]
        build_native_project(directory, Path(binary).name, (compiler['uri'],), expected_libraries=libraries)
    refs = {name: _ref(_path(directory, name), 'scar.native-project-member') for name in inputs}
    _require(all(_raw(refs[name]) == content for name, content in inputs.items()), 'C++ component source changed')
    exe = _ref(_path(directory, binary), 'scar.native-project-member')
    _require(exe['sha256'] != members[binary]['sha256'] and _raw(exe).startswith(b'\x7fELF')
             and os.access(exe['uri'], os.X_OK), 'expected distinct, ordinary C++20 component executable')
    files = {'project/' + name: _signature(reference) for name, reference in refs.items()}
    loader = _loader(Path(exe['uri']), directory, files, str(directory), environment)
    _require(loader == route_sites['expected_loader'], 'probe changed the retained route loader closure')
    return dict(executable=exe, files=refs, loader=loader, original_executable=members[binary], original_source=members[_SOURCE])


def _capture(directory, launch):
    directory.mkdir()
    start = clock_gettime_ns(CLOCK_MONOTONIC)
    result = subprocess.run(launch['argv'], cwd=launch['cwd'], env=launch['environment'], stdin=subprocess.DEVNULL,
                            capture_output=True, timeout=120)
    end = clock_gettime_ns(CLOCK_MONOTONIC)
    (directory/'stdout.json').write_bytes(result.stdout)
    (directory/'stderr.log').write_bytes(result.stderr)
    _write(directory/'capture.json', dict(schema='scar.cpu-array-native-publication-clocks',
        returncode=result.returncode, launcher_start_ns=start, process_reaped_ns=end))


def _sample(directory, fixture, output, launch, recipe, oracle):
    stdout, stderr, published, timestamps = (_ref(directory/name) for name in
        ('stdout.json', 'stderr.log', output, _TIMESTAMPS))
    clocks = _ref(directory/'capture.json', 'scar.cpu-array-native-publication-clocks')
    retained = read_artifact(clocks)
    _require(retained['returncode'] == 0 and not _raw(stderr)
             and _raw(stdout) == _raw(fixture['stdout']) and _raw(published) == _raw(fixture['published'])
             and _match_json_report(_raw(stdout), _raw(published), recipe, oracle),
             'the native writer changed bytes or the normalized publication oracle')
    _require({path.name for path in directory.iterdir()} ==
             {'stdout.json', 'stderr.log', output, _TIMESTAMPS, 'capture.json'}, 'unexpected publication side effect')
    values = _TIMES.unpack(_raw(timestamps))
    bounds = (retained['launcher_start_ns'], *values, retained['process_reaped_ns'])
    _require(all(type(value) is int and value > 0 for value in bounds) and tuple(sorted(bounds)) == bounds,
             'component markers are outside the scoped process clock')
    return dict(markers_ns=dict(zip(_MARKERS, values, strict=True)), launch=launch, timestamps=timestamps,
                process_clocks=clocks, stdout=stdout, stderr=stderr, publication=published)


def _inputs(probe_sites_ref, comparison_ref, checked_lifecycle=None):
    from scar.analysis import cpu_array_native_route_probe_sites as sites

    with replay_transaction():
        if checked_lifecycle is None:
            boundaries = sites.verify_cpu_array_native_route_probe_sites(probe_sites_ref)
        else:
            boundaries = read_artifact(probe_sites_ref)
            graph_ref = boundaries['lifecycle']
            source = read_artifact(checked_lifecycle['proof'])
            _require(probe_sites_ref['schema'] == sites.SCHEMA
                     and _canonical(read_artifact(graph_ref)) == _canonical(checked_lifecycle)
                     and _canonical(boundaries) == _canonical(sites._project(graph_ref, checked_lifecycle, source)),
                     'publication sites differ from the verified lifecycle')
        comparison = verify_cpu_array_native_route_observation(comparison_ref)
    source = read_artifact(boundaries['proof'])
    workload = boundaries['workload']
    _require(comparison['retention'] == source['retention'] and comparison['grouped'] == source['baseline_grouped']
             and all(comparison['case'][key] == workload[key] for key in ('argv', 'recipe', 'holdout'))
             and comparison['case']['cpu_ids'] == source['resources']['launch_cpu_ids'],
             'native report fixtures do not use the admitted argv, routes, CPU cohort and oracle')
    references = {'retained_process': source['retained_runtime'], 'grouped_openmp': source['baseline_grouped']}
    compiler = read_artifact(references['grouped_openmp'])['compiler']
    _raw(compiler)
    fixtures = {}
    environments = {'retained_process': source['workload']['local_process_environment'],
                    'grouped_openmp': source['workload']['openmp_grouped_environment']}
    index = comparison['pairs']['warmup_pairs']
    for route in _ROUTES:
        row, = [row for row in comparison['runs'] if row['pair'] == index and row['mode'] == route]
        _require(row['diagnostics_enabled'] is False and row['returncode'] == 0,
                 'sealed clean native publication fixture required')
        fixtures[route] = {key: row[key] for key in ('stdout', 'published')}
    return boundaries, source['resources']['launch_cpu_ids'], references, compiler, fixtures, environments


def _repeat(probe_sites_ref, comparison_ref, root, count, *, build=False, checked_lifecycle=None):
    boundaries, cpus, references, compiler, fixtures, environments = _inputs(
        probe_sites_ref, comparison_ref, checked_lifecycle)
    _require(type(count) is int and 1 <= count <= 24, 'expected one to twenty-four diagnostics per native route')
    workload = boundaries['workload']
    argv, output_index = workload['argv'], workload['output_index']
    output = argv[output_index]
    recipe = read_artifact(workload['recipe'])
    oracle = _raw(workload['normalized_oracle'])
    _require(output_index == len(argv) - 1 and Path(output).name == output and output not in ('.', '..', _TIMESTAMPS),
             'isolated terminal publication filename required')
    if build:
        _require(set(cpus) <= os.sched_getaffinity(0), 'sealed CPU cohort is unavailable')
        root.mkdir(parents=True, exist_ok=False)
    routes = {}
    for route in _ROUTES:
        project = _project(root/route/'project', boundaries['routes'][route], references[route], fixtures[route],
                           recipe, boundaries['scope']['task_count'], compiler, environments[route], build=build)
        routes[route] = dict(**project, fixture=fixtures[route], samples=[])
    acquisition = []
    for index in range(count):
        for route in (_ROUTES if index % 2 == 0 else _ROUTES[::-1]):
            directory = root/route/f'sample-{index:02d}'
            exe = routes[route]['executable']
            launch = dict(argv=['/usr/bin/taskset', '-c', ','.join(map(str, cpus)), exe['uri'], *argv],
                          cwd=str(directory), environment=environments[route])
            if build:
                _capture(directory, launch)
            sample = _sample(directory, fixtures[route], output, launch, recipe, oracle)
            acquisition.append(read_artifact(sample['process_clocks']))
            routes[route]['samples'].append(sample)
    _require(all(left['process_reaped_ns'] <= right['launcher_start_ns']
                 for left, right in zip(acquisition, acquisition[1:])), 'publication sample acquisition order differs')
    return dict(schema=SCHEMA, schema_version=1, probe_sites=probe_sites_ref, comparison=comparison_ref,
                proof=boundaries['proof'], cpu_ids=cpus, argv=argv, compiler=compiler, cxx_standard=20,
                sample_count=count, routes=routes, clock='LINUX_CLOCK_MONOTONIC_NS',
                observation='REPLAYED_OUTPUT_VALUES_THROUGH_EXACT_NATIVE_WRITER', normalized_oracle=workload['normalized_oracle'],
                exact_stdout_and_file_bytes=True, original_executable_executed=False,
                original_numerical_tasks_executed=False, probe_executed=True, clean_benchmark=False,
                physical_duration_bound_proven=False, forward_cost_proven=False, deployment_authorized=False,
                windows=dict(file_write_close=['FILE_OPEN_BEGIN', 'FILE_CLOSE_COMPLETE'],
                    stdout_submit=['STDOUT_BEGIN', 'STDOUT_SUBMIT_END'],
                    publication_statements=['PUBLICATION_BEGIN', 'STDOUT_SUBMIT_END']),
                limits=dict(file_fsync_not_called=True, stdout_flush_boundary_unobserved=True,
                    done_after_probe_write_and_process_exit=True, producer_team_history_omitted=True,
                    launcher_start_and_reap_are_diagnostic_only=True))


def measure(probe_sites_ref, comparison_ref, out, *, samples=5):
    """Replay exact output fixtures through separate C++20 publication probes."""
    out = Path(out).absolute()
    _require(out.resolve() == out and not out.exists() and not out.is_symlink(), 'fresh canonical probe root required')
    report = _repeat(probe_sites_ref, comparison_ref, out, samples, build=True)
    _write(out/'result.json', report)
    return report


def verify_cpu_array_native_route_boundary_calibration(reference):
    """Reconstruct source, loader and exact output proofs without invoking either probe."""
    return _verify(reference)


def _verify(reference, checked_lifecycle=None):
    saved = read_artifact(reference)
    path = Path(reference['uri'])
    _require(reference['schema'] == SCHEMA and path.name == 'result.json' and path.resolve(strict=True) == path,
             'exact boundary calibration receipt required')
    _require(saved == _repeat(saved['probe_sites'], saved['comparison'], path.parent, saved['sample_count'],
                              checked_lifecycle=checked_lifecycle),
             'retained calibration differs at cold replay')
    return saved


def main(argv=None):
    from scar.analysis.cpu_array_native_route_probe_sites import SCHEMA as SITES

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--sites', type=Path)
    mode.add_argument('--check', type=Path)
    parser.add_argument('--comparison', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--samples', type=int, default=5)
    args = parser.parse_args(argv)
    if args.check is not None:
        if args.comparison is not None or args.out is not None:
            parser.error('--check reads an existing publication calibration')
        report = verify_cpu_array_native_route_boundary_calibration(_ref(args.check, SCHEMA))
        result = args.check
    else:
        if args.comparison is None or args.out is None:
            parser.error('--sites requires --comparison and --out')
        report = measure(_ref(args.sites, SITES), _ref(args.comparison, COMPARISON), args.out, samples=args.samples)
        result = args.out/'result.json'
    print(json.dumps(dict(result=str(result), samples_per_route=report['sample_count'])))


if __name__ == '__main__':
    raise SystemExit(main())
