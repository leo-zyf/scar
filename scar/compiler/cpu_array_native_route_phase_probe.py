"""Source-bound CPU route diagnostics in separately compiled C++20 projects."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from struct import Struct
import subprocess
from time import CLOCK_MONOTONIC, clock_gettime_ns

from scar.codegen.cpu_array_native_route_package import SCHEMA as PACKAGE, verify_cpu_array_native_route_package
from scar.codegen.cpu_catalog_runtime_package import _loader, _signature
from scar.codegen.native_cpu_arena_reuse import _path
from scar.codegen.native_project import build_native_project
from scar.compiler.calibrate_array_entry import _write
from scar.compiler.cpu_entry_replay_adapters import _match_json_report
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.cpu-array-native-route-phase-probe'
_ROUTES = ('retained_process', 'grouped_openmp')
_SOURCE = 'src/main.cpp'
_HEADER = 'include/scar_generated/scar_route_phase_probe.hpp'
_MARKERS = ('MAIN_ENTER', 'DISPATCH_BEGIN', 'TEAM_ENTER', 'TEAM_RETURN',
            'DISPATCH_RETURN', 'PUBLICATION_BEGIN', 'PUBLICATION_END')
_OWNER_RECORD = Struct('=8Q')
_TASK_RECORD = Struct('=8Q')
_SIDECAR = 'phase-record.bin'
_LIMIT = 64
_RAW = 'application/octet-stream'

_CPP = b'''#pragma once
#include <array>
#include <cstdint>
#include <cstdio>
#include <new>
#include <sched.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

namespace scar::route_phase {
enum Marker { MAIN_ENTER, DISPATCH_BEGIN, TEAM_ENTER, TEAM_RETURN, DISPATCH_RETURN,
              PUBLICATION_BEGIN, PUBLICATION_END, MARKER_COUNT };
using Word = std::uint64_t;
struct alignas(64) Row {
    Word start, end, pid, tid, cpu, reserved[3];
};
static_assert(sizeof(Row) == 64);
using Rows = std::array<Row, 64>;
static Rows* slots{};
static std::array<Word, MARKER_COUNT + 1> times{};
static_assert(sizeof(times) == 64);

inline Word now() {
    timespec timestamp{};
    if (clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0) return 0;
    return Word(timestamp.tv_sec) * 1000000000ULL + Word(timestamp.tv_nsec);
}

inline void mark(Marker index) { times[index] = now(); }

inline bool prepare(Word count) {
    if (!count || count > 64) return false;
    void* storage = mmap(nullptr, sizeof(Rows), PROT_READ | PROT_WRITE,
                        MAP_ANONYMOUS | MAP_SHARED, -1, 0);
    if (storage == MAP_FAILED) return false;
    slots = ::new (storage) Rows{};
    return true;
}

inline void record(Word ordinal, Word begin) {
    const auto end = now();
    (*slots)[ordinal] = {begin, end, Word(getpid()), Word(gettid()), Word(sched_getcpu()), {0, 0, 0}};
}

inline bool write(Word count) {
    times[MARKER_COUNT] = getpid();
    auto* stream = std::fopen("phase-record.bin", "wbx");
    if (!stream) return false;
    const auto owner = std::fwrite(times.data(), sizeof(times), 1, stream);
    const auto task = std::fwrite(slots->data(), sizeof(Row), count, stream);
    const auto closed = std::fclose(stream);
    return owner == 1 && task == count && closed == 0;
}
}
'''


def _need(condition, detail):
    if not condition:
        raise ValueError('CPU array native route phase probe: ' + detail)


def _raw(ref):
    return read_artifact(ref, json_document=False)


def _ref(path, schema=_RAW):
    _need(path == path.resolve(strict=True), 'expected a local file: ' + str(path))
    return artifact_reference(path, schema)


def _render(source, route):
    marks = {}

    def insert(site, anchor, statement, *, after=False):
        _need(source.count(anchor) == 1, 'unexpected generated source at ' + site)
        if after:
            _need(anchor.endswith(b'\n'), 'after anchor must end at the original next line')
        start = source.index(anchor)
        offset = start + (len(anchor) if after else 0)
        _need(offset > 0 and source[offset-1:offset] == b'\n' and offset not in marks, 'invalid insertion point')
        marks[offset] = dict(id=site, byte_offset=offset, anchor_byte_start=start,
                             anchor=anchor.decode('ascii'), position='AFTER' if after else 'BEFORE',
                             statement=statement.decode('ascii'))

    def mark(site, anchor, *, after=False, indent=b'        '):
        insert(site, anchor, indent + b'scar::route_phase::mark(scar::route_phase::' + site.encode() + b');\n',
               after=after)

    mark('MAIN_ENTER', b'int main(int argc, char** argv) {\n', after=True, indent=b'    ')
    insert('RESERVE_PRIVATE_WITNESS', b'        if (!enforce_coordinator_limit()) return 2;\n',
           b'        if (!scar::route_phase::prepare(result.task_count)) return 73;\n')
    mark('DISPATCH_BEGIN', b'        const auto status = execute(result);\n')
    mark('DISPATCH_RETURN', b'        const auto status = execute(result);\n', after=True)
    if route == 'grouped_openmp':
        mark('TEAM_ENTER', b'    if (!scar::openmp_v1::dispatch(worker_cpus, result.task_count,', indent=b'    ')
        mark('TEAM_RETURN', b'                                 result.records[task], diagnostic, task);\n'
                            b'    })) return 3;\n', after=True, indent=b'    ')
    else:
        mark('TEAM_ENTER', b'        const auto summary = scar::ordered_v1::dispatch(plan, result.task_count,')
        mark('TEAM_RETURN', b'        if (status != 0) return status;\n'
                            b'        if (summary.succeeded != result.task_count) return 3;')
    numeric = (b'    if (scar_array_entry_profile_0(&result.norm, &result.split_difference, '
               b'diagnostic ? checkpoint.c_str() : nullptr,\n'
               b'                                  request.seed, request.rounds) != 0) return 3;\n')
    insert('TASK_NUMERICAL_BEGIN', numeric, b'    const auto probe_begin = scar::route_phase::now();\n')
    insert('TASK_NUMERICAL_RETURN', numeric, b'    scar::route_phase::record(ordinal, probe_begin);\n', after=True)
    mark('PUBLICATION_BEGIN', b'        const auto& path = values[8].text;\n')
    insert('PUBLICATION_END_AND_DIAGNOSTIC_WRITE', b'        return std::cout ? 0 : 1;\n',
           b'        scar::route_phase::mark(scar::route_phase::PUBLICATION_END);\n'
           b'        if (!scar::route_phase::write(result.task_count)) return 73;\n')
    result = [b'#include "scar_route_phase_probe.hpp"\n']
    previous = 0
    sites = []
    for offset, row in sorted(marks.items()):
        result.extend((source[previous:offset], row['statement'].encode('ascii')))
        sites.append(row)
        previous = offset
    result.append(source[previous:])
    return b''.join(result), sites


def _project(folder, arm, route, compiler, *, build=False):
    runtime = read_artifact(arm['package'])
    baseline = runtime['files']
    source = baseline[_SOURCE]
    edited, sites = _render(_raw(source), route)
    binary = runtime['binary']
    _need(_HEADER not in baseline and runtime['cxx_standard'] == 20, 'existing runtime already has probe or wrong C++ ABI')
    members = {name: _raw(reference) for name, reference in baseline.items() if name != binary and name != _SOURCE}
    members.update({_SOURCE: edited, _HEADER: _CPP})
    if build:
        for name, payload in members.items():
            file = _path(folder, name)
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(payload)
        libraries = [_path(folder, key) for key in members if key.startswith('provider/objects/')]
        build_native_project(folder, Path(binary).name, (compiler['uri'],), expected_libraries=libraries)
    receipts = {name: _ref(_path(folder, name), 'scar.native-project-member') for name in members}
    _need(all(_raw(receipts[name]) == payload for name, payload in members.items()),
          'prepared source, original member or witness header differs')
    sibling = _ref(_path(folder, binary), 'scar.native-project-member')
    _need(sibling['sha256'] != arm['executable']['sha256'] and _raw(sibling).startswith(b'\x7fELF')
          and os.access(sibling['uri'], os.X_OK), 'diagnostic must be a separate executable ELF')
    environment = arm['launch']['environment']
    files = {'project/' + name: _signature(receipt) for name, receipt in receipts.items()}
    loader = _loader(Path(sibling['uri']), folder, files, str(folder), environment)
    _need(loader == arm['loader'], 'instrumented route changed its effective loader dependencies')
    return dict(original_executable=arm['executable'], original_source=source, sites=sites,
                diagnostic_executable=sibling, diagnostic_members=receipts, loader=loader)


def _launch(arm, project, package, folder):
    command = arm['launch']['command']
    _need(command[3] == arm['executable']['uri'] and command[4:] == package['argv'],
          'retained arm launcher or arguments differ')
    return dict(argv=[*command[:3], project['diagnostic_executable']['uri'], *package['argv']],
                cwd=str(folder), environment=arm['launch']['environment'], timeout=package['timeout_seconds'])


def _capture(arm, project, package, folder):
    folder.mkdir()
    launch = _launch(arm, project, package, folder)
    start = clock_gettime_ns(CLOCK_MONOTONIC)
    result = subprocess.run(launch['argv'], cwd=launch['cwd'], env=launch['environment'], stdin=subprocess.DEVNULL,
                            capture_output=True, timeout=launch['timeout'])
    end = clock_gettime_ns(CLOCK_MONOTONIC)
    (folder/'stdout.json').write_bytes(result.stdout)
    (folder/'stderr.log').write_bytes(result.stderr)
    _write(folder/'clocks.json', dict(schema='scar.cpu-array-native-route-phase-clocks', returncode=result.returncode,
                                     runner_job_start_ns=start, runner_reaped_ns=end))


def _sample(folder, launch, package, runtime, placement):
    clocks = _ref(folder/'clocks.json', 'scar.cpu-array-native-route-phase-clocks')
    clock = read_artifact(clocks)
    raw, stdout, stderr, publication = (_ref(folder/name) for name in
            (_SIDECAR, 'stdout.json', 'stderr.log', package['output_leaf']))
    stdout_bytes = _raw(stdout)
    recipe = read_artifact(package['recipe'])
    _need(clock['returncode'] == 0 and not _raw(stderr)
          and _match_json_report(stdout_bytes, _raw(publication), recipe, _raw(package['oracle'])),
          'diagnostic changed exit status, stderr or normalized output')
    _need({path.name for path in folder.iterdir()} == {'clocks.json', _SIDECAR, 'stdout.json', 'stderr.log',
                                                     package['output_leaf']},
          'unexpected diagnostic sample files')
    rows = json.loads(stdout_bytes)[runtime['output_fields']['results']]
    count = package['scope']['task_count']
    payload = _raw(raw)
    _need(len(payload) == _OWNER_RECORD.size + count * _TASK_RECORD.size and len(rows) == count,
          'missing or excess owner/task phase rows')
    *owner, pid = _OWNER_RECORD.unpack_from(payload)
    bounds = clock['runner_job_start_ns'], *owner, clock['runner_reaped_ns']
    _need(pid > 0 and all(type(stamp) is int and stamp > 0 for stamp in bounds)
          and list(bounds) == sorted(bounds), 'invalid or unordered owner clock')
    marks = dict(zip(_MARKERS, owner, strict=True))
    task_rows = []
    lanes = placement['worker_cpu_ids']
    per_lane = defaultdict(list)
    grouped = placement['mode'] == 'ORDERED_OPENMP'
    for ordinal, output in enumerate(rows):
        begin, end, task_pid, tid, cpu, *reserved = _TASK_RECORD.unpack_from(
            payload, _OWNER_RECORD.size + ordinal * _TASK_RECORD.size)
        lane = ordinal % len(lanes)
        _need(not any(reserved) and marks['TEAM_ENTER'] <= begin < end <= marks['TEAM_RETURN']
              and cpu == lanes[lane] and task_pid == output[runtime['output_fields']['worker_pid']]
              and task_pid > 0 and tid > 0 and (task_pid == pid if grouped else tid == task_pid != pid),
              'worker identity, CPU, chronological or native report evidence differs')
        row = dict(ordinal=ordinal, lane=lane, pid=task_pid, tid=tid, cpu_id=cpu, begin_ns=begin, end_ns=end)
        task_rows.append(row)
        per_lane[lane].append(row)
    identities = []
    for events in per_lane.values():
        ids = {(event['pid'], event['tid']) for event in events}
        _need(len(ids) == 1 and all(left['end_ns'] <= right['begin_ns'] for left, right in zip(events, events[1:])),
              'one lane changed task identity or overlapped sequential work')
        identities.extend(ids)
    _need(len(identities) == len(set(identities)) == len(lanes), 'worker lanes do not own distinct worker identities')
    return dict(launch=launch, owner_pid=pid, owner_markers_ns=marks, task_numerical_intervals=task_rows,
                clocks=clocks, raw_markers=raw, stdout=stdout, stderr=stderr, published=publication,
                normalized_output_matches=True)


def _report(reference, root, sample_count, *, build=False, checked_package=None):
    package = verify_cpu_array_native_route_package(reference) if checked_package is None else checked_package
    plan = read_artifact(package['plan'])
    _need(type(sample_count) is int and 1 <= sample_count <= 12 and 1 <= package['scope']['task_count'] <= _LIMIT,
          'bounded sample and task counts required')
    selected = read_artifact(package['routes']['grouped_openmp']['package'])
    compiler = selected['compiler']
    _raw(compiler)
    resources = plan['resources']
    placement_keys = dict(zip(_ROUTES, ('local_processes', 'openmp_grouped'), strict=True))
    if build:
        _need(set(package['cpu_ids']) <= os.sched_getaffinity(0), 'sealed CPU cohort is unavailable')
        root.mkdir(parents=True, exist_ok=False)
    projects = {}
    for route in _ROUTES:
        arm = package['routes'][route]
        projects[route] = _project(root/route/'project', arm, route, compiler, build=build)
    if build:
        for index in range(sample_count):
            for route in (_ROUTES if index % 2 == 0 else _ROUTES[::-1]):
                _capture(package['routes'][route], projects[route], package, root/route/f'sample-{index:02d}')
    measurements = {}
    for route in _ROUTES:
        arm, project = package['routes'][route], projects[route]
        runtime = read_artifact(arm['package'])
        if route == 'grouped_openmp':
            runtime = read_artifact(runtime['runtime'])
        rows = []
        for index in range(sample_count):
            folder = root/route/f'sample-{index:02d}'
            rows.append(_sample(folder, _launch(arm, project, package, folder), package, runtime,
                                resources[placement_keys[route]]))
        measurements[route] = dict(**project, samples=rows)
    acquisition = [
        read_artifact(measurements[route]['samples'][index]['clocks'])
        for index in range(sample_count) for route in (_ROUTES if index % 2 == 0 else _ROUTES[::-1])
    ]
    _need(all(left['runner_reaped_ns'] <= right['runner_job_start_ns']
              for left, right in zip(acquisition, acquisition[1:])), 'saved probes did not run in alternating order')
    return dict(schema=SCHEMA, schema_version=1, route_package=reference, plan=package['plan'],
                arg_scope=package['argv'], cpu_ids=package['cpu_ids'], target=package['hardware'],
                compiler=compiler, cxx_standard=20, clock='LINUX_CLOCK_MONOTONIC_NS',
                practice='SOURCE_BOUND_INSTRUMENTED_SIBLINGS', sample_count=sample_count, routes=measurements,
                oracle=package['oracle'], normalized_output_matches=True, target_executed=True,
                original_executable_executed=False, python_workload_executed=False, clean_benchmark=False,
                forward_cost_proven=False, deployment_authorized=False, limits=dict(
                    per_task_stamps='PROFILE_ENTRY_THROUGH_SUCCESSFUL_RETURN',
                    team_window='DISPATCH_INCLUDES_WORKER_LIFETIME_AND_JOIN',
                    task_intervals_overlap_between_lanes=True,
                    witness_setup='MAPPED_BEFORE_NATIVE_COORDINATOR_LIMIT',
                    witness_serialization='AFTER_PUBLICATION_STATEMENTS_BEFORE_RETURN',
                    stdout_flush_may_be_after_last_source_marker=True,
                    runner_interval_includes_launcher_and_pipe_collection=True))


def measure(route_package_ref, out, *, samples=3):
    """Build and run separate native diagnostic projects on the pinned CPU scope."""
    out = Path(out).absolute()
    _need(out.resolve() == out and not out.exists() and not out.is_symlink(), 'fresh canonical diagnosis root required')
    report = _report(route_package_ref, out, samples, build=True)
    _write(out/'result.json', report)
    return report


def verify_cpu_array_native_route_phase_probe(reference):
    """Replay exact source boundaries, sibling linkage and saved observations without target runs."""
    return _verify(reference)


def _verify(reference, checked_package=None):
    report = read_artifact(reference)
    path = Path(reference['uri'])
    _need(reference['schema'] == SCHEMA and path == path.resolve(strict=True) and path.name == 'result.json',
          'expected canonical phase probe result reference')
    package = None
    if checked_package is not None:
        package_ref, package = checked_package
        _need(package_ref == report['route_package'], 'phase samples differ from the checked package')
    _need(report == _report(report['route_package'], path.parent, report['sample_count'], checked_package=package),
          'retained probe evidence differs from cold replay')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--package', type=Path)
    mode.add_argument('--check', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--samples', type=int, default=3)
    args = parser.parse_args(argv)
    if args.check is not None:
        if args.out is not None:
            parser.error('--check reads an existing diagnostic')
        report = verify_cpu_array_native_route_phase_probe(_ref(args.check, SCHEMA))
        result = args.check
    else:
        if args.out is None:
            parser.error('--package requires a fresh --out path')
        report = measure(_ref(args.package, PACKAGE), args.out, samples=args.samples)
        result = args.out/'result.json'
    print(json.dumps(dict(result=str(result), samples_per_route=report['sample_count'],
                          clean_benchmark=report['clean_benchmark'])))


if __name__ == '__main__':
    raise SystemExit(main())
