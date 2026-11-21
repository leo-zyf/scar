"""Compare serial, local-process and OpenMP schedules for one proved task family."""

import argparse
import json
from pathlib import Path
import subprocess

from scar.ir.execution_recipe import read_artifact
from scar.optimization import independent_array_task_schedule as base


SCHEMA = 'scar.independent-array-task-team-schedule'
EVIDENCE = 'scar.independent-array-task-openmp-cost-evidence'
MODE = 'ORDERED_OPENMP'
STACK_BYTES = 8 * 1024**2
_CODEGEN = Path(__file__).resolve().parents[1] / 'codegen'
_OPENMP_HEADER = _CODEGEN / 'native_ordered_openmp_v1.h.in'
_PROCESS_HEADER = _CODEGEN / 'native_ordered_dispatch_v1.h.in'
_NATIVE_SOURCES = ('profile_0.cpp', 'numpy_vector_normalize.cpp', 'numpy_observables.cpp',
                   'numpy_complex.cpp', 'numpy_random.cpp', 'numpy_random_coefficients.cpp')
_SAMPLE_BOUNDARY = 'FRESH_EXACT_DISPATCH_EMPTY_OR_ONE_SIMULTANEOUS_KERNEL_CALL_PER_LANE'


def _need(ok, reason):
    if not ok:
        raise ValueError('independent array task team schedule: ' + reason)


_PROBE = r'''#include "ordered_openmp.hpp"
#include "ordered_processes.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <xmmintrin.h>

extern "C" int scar_array_entry_profile_0(double*, double*, const char*, std::uint64_t, std::uint64_t);
extern "C" int scipy_openblas_get_parallel64_();
extern "C" int scipy_openblas_get_num_threads64_();
extern "C" const char* scipy_openblas_get_corename64_();

namespace {
using Clock = std::chrono::steady_clock;
struct Request { std::uint64_t seed, rounds; };
struct Result { std::int64_t duration{}; std::uint64_t ordinal{}; double fields[5]{}; };
static_assert(sizeof(Request) == 16 && sizeof(Result) == 56);

std::int32_t run(Request request, Result& result, bool numeric) {
    if (!numeric) return 0;
    const auto start = Clock::now();
    if (scar_array_entry_profile_0(result.fields, result.fields + 1, nullptr, request.seed, request.rounds)) return 1;
    result.duration = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
    return std::isfinite(result.fields[0]) && std::isfinite(result.fields[1]) && result.duration > 0 ? 0 : 1;
}
}

int main(int argc, char** argv) {
    if (argc < 10) return 2;
    const bool threaded = std::string_view(argv[1]) == "openmp";
    const bool numeric = std::string_view(argv[2]) == "kernel";
    const auto count = std::stoull(argv[3]);
    const Request request{std::stoull(argv[4]), std::stoull(argv[5])};
    const int owner = std::stoi(argv[6]);
    const auto owner_limit = std::stoull(argv[7]);
    const auto worker_limit = std::stoull(argv[8]);
    const rlimit bounds{owner_limit, owner_limit};
    if (setrlimit(RLIMIT_AS, &bounds)) return 3;
    if (threaded && (std::fegetround() != FE_TONEAREST || fegetexcept() != 0
            || (_mm_getcsr() & 0xe040u) != 0 || (_mm_getcsr() & 0x1f80u) != 0x1f80u
            || scipy_openblas_get_parallel64_() != 1 || scipy_openblas_get_num_threads64_() != 1
            || std::string_view(scipy_openblas_get_corename64_()) != SCAR_PROBE_BLAS_CORE)) return 3;
    std::vector<int> cpus;
    for (int i = 9; i < argc; ++i) cpus.push_back(std::stoi(argv[i]));
    if (numeric && count != cpus.size()) return 2;
    std::vector<Result> results(count);
    std::vector<std::int32_t> statuses(threaded ? count : 0);

    const auto start = Clock::now();
    if (threaded) {
        auto row = [&](std::uint64_t ordinal, std::size_t) {
            auto& result = results[ordinal];
            result.ordinal = ordinal;
            statuses[ordinal] = run({request.seed + ordinal, request.rounds}, result, numeric);
        };
        if (!scar::openmp_v1::dispatch(cpus, count, row)) return 4;
    } else {
        const scar::ordered_v1::Config cfg{owner, cpus, cpus.size(), sizeof(Request), sizeof(Result), worker_limit,
                                          cpus.size() * (sizeof(Request) + sizeof(Result))};
        const auto factory = [numeric](int) {
            return [numeric](std::uint64_t ordinal, std::span<const std::byte> in, std::span<std::byte> out) {
                Request input{};
                Result result{};
                result.ordinal = ordinal;
                std::memcpy(&input, in.data(), sizeof(input));
                const auto error = run(input, result, numeric);
                if (!error) std::memcpy(out.data(), &result, sizeof(result));
                return error;
            };
        };
        auto summary = scar::ordered_v1::dispatch(cfg, count, factory,
            [&](std::uint64_t ordinal, std::span<std::byte> out) {
                const Request input{request.seed + ordinal, request.rounds};
                std::memcpy(out.data(), &input, sizeof(input));
            }, [&](const scar::ordered_v1::Retirement& row) {
                if (row.kind == scar::ordered_v1::Kind::success) {
                    std::memcpy(&results[row.sequence], row.result.data(), sizeof(Result));
                }
            });
        if (summary.succeeded != count) return 5;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
    std::cout << elapsed;
    for (std::size_t index = 0; index < count; ++index) {
        if (results[index].ordinal != index || results[index].duration < 0 || (threaded && statuses[index])) return 6;
        std::cout << ' ' << results[index].duration;
    }
    std::cout << '\n';
}
'''


def _inputs(process_path, contract_path):
    from scar.analysis import native_array_thread_safety as legal

    process_ref, contract_ref = base._ref(process_path, base.SCHEMA), base._ref(contract_path, legal.SCHEMA)
    process, contract = base.check(process_ref['uri']), legal.check(contract_ref['uri'])
    _need(process['family'] == contract['family'] and process['native'] == contract['native'],
          'thread-safety evidence does not close over the same task family and numerical body')
    return process_ref, process, contract_ref, contract


def _thread_rows(process, native, hardware):
    serial = process['alternatives'][0]
    cpus = [row['cpu_id'] for row in process['allowed_cpu_lanes']]
    memory = serial['memory']
    limit = memory['coordinator_address_space_limit_bytes']
    arena, retained = native['arena_bytes'], memory['retained_result_payload_bytes']
    max_width = min(len(cpus), *[process['scope'][key] for key in ('task_count', 'requested_workers')])
    rows = []
    for width in range(2, max_width + 1):
        statuses = 4 * process['scope']['task_count']
        explicit = width * arena + (width - 1) * STACK_BYTES + retained + statuses
        budgets = dict(memory, request_frame_bytes=0, result_frame_bytes=0, payload_budget_bytes=0,
            explicit_owner_payload_bytes=explicit, explicit_worker_payload_bytes=arena,
            required_headroom_bytes=limit, thread_stack_bytes=STACK_BYTES, retained_status_bytes=statuses,
            bound_scope='SINGLE_PROCESS_VAS_CAP_INCLUDING_PRIVATE_ARRAYS_AND_OPENMP_CHILD_STACKS')
        reasons = [reason for reason in serial['capacity_reasons'] if reason != 'CPU_QUOTA_BELOW_CONFIGURED_LANES']
        quota = hardware['cpu_quota_equivalents']
        if quota is not None and quota[0] < width * quota[1]:
            reasons.append('CPU_QUOTA_BELOW_CONFIGURED_LANES')
        if explicit >= limit:
            reasons.append('EXCEEDS_DECLARED_TEAM_ADDRESS_SPACE_LIMIT')
        rows.append(dict(mode=MODE, owner_cpu_id=serial['owner_cpu_id'], worker_count=width, worker_cpu_ids=cpus[:width],
            window=width, launch_cpu_ids=serial['launch_cpu_ids'], memory=budgets, capacity_reasons=reasons,
            scheduled_pod_result_lifetime=serial['scheduled_pod_result_lifetime']))
    return rows


def _compile_command(out, native):
    files = native['files']
    library, = (key for key in files if key.startswith('provider/objects/') and 'openblas' in key)
    objects = Path(files[library]['uri']).parent
    root = Path(files['CMakeLists.txt']['uri']).parent
    sources = [files[f'src/{name}']['uri'] for name in _NATIVE_SOURCES]
    archive, = (value['uri'] for key, value in files.items() if key.endswith('/libnpyrandom.a'))
    normal = read_artifact(read_artifact(native['recipe'])['normalize'])['provider']
    _need(normal['blas_threads'] == 1 and normal['blas']['sha256'] == files[library]['sha256'],
          'exact single-thread BLAS object and core required')
    return ['/usr/bin/g++', '-std=c++20', '-O3', '-DNDEBUG', '-fopenmp', '-fno-fast-math', '-ffp-contract=off',
            '-DSCAR_ARRAY_TAPE_NORMALIZE', '-DSCAR_PROBE_BLAS_CORE=' + json.dumps(normal['blas_core']),
            '-I', str(root / 'include/scar_generated'), '-I', str(root / 'src'),
            str(out / 'probe.cpp'), *sources, archive, '-L', str(objects), '-l:' + Path(library).name,
            '-Wl,--disable-new-dtags,-rpath,' + str(objects), '-Wl,-rpath-link,' + str(objects), '-lm',
            '-o', str(out / 'probe')]


def _verify_native(native):
    for member in native['files'].values():
        read_artifact(member, json_document=False)


def _environment(recipe):
    env = dict(recipe['environment'], OMP_STACKSIZE='8M')
    env.pop('GOMP_STACKSIZE', None)
    return env


def _runtime_identity(probe, compiler, native, env):
    from scar.codegen.aten_provider import _dependencies

    def inspect(arguments):
        return subprocess.check_output(arguments, env=env, text=True, timeout=10).strip()

    openmp = inspect([compiler, '-print-file-name=libgomp.so.1'])
    runtime = base._ref(openmp, 'scar.native-provider-library')
    deps = _dependencies(inspect(['/usr/bin/ldd', probe]))
    providers = [value['uri'] for key, value in native['files'].items() if key.startswith('provider/objects/')]
    _need('libgomp.so.1' in deps and Path(deps['libgomp.so.1']).resolve() == Path(runtime['uri'])
          and all(deps.get(Path(path).name) == path for path in providers),
          'probe did not load the exact OpenMP runtime and numerical providers')
    return base._ref(compiler, 'scar.native-compiler-binary'), runtime


def _command(probe, choice, scope, seed, part):
    memory = choice['memory']
    threaded = choice['mode'] == MODE
    count = choice['worker_count'] if part == 'kernel' else scope['task_count']
    return ['/usr/bin/taskset', '-c', ','.join(map(str, choice['launch_cpu_ids'])), probe,
            'openmp' if threaded else 'process', part, str(count), str(seed), str(scope['round_count']),
            str(choice['owner_cpu_id']), str(memory['coordinator_address_space_limit_bytes']),
            str(memory['worker_address_space_limit_bytes']), *map(str, choice['worker_cpu_ids'])]


def _eligible(process, thread_rows):
    return [row for row in (*process['alternatives'][1:], *thread_rows) if not row['capacity_reasons']]


def _seeds(seeds, process, native):
    prior = read_artifact(process['evidence'])['seeds']
    width = process['scope']['task_count']
    observed = [row['parameters'][native['seed_option']] for row in read_artifact(native['recipe'])['profiles']]
    _need(len(seeds) >= 4 and len(set(seeds)) == len(seeds) and all(type(seed) is int and
          native['seed_range']['minimum'] <= seed <= native['seed_range']['maximum'] - width + 1 for seed in seeds)
          and all(not any(seed <= old < seed + width for old in (*prior, *observed)) for seed in seeds)
          and all(right - left >= width for left, right in zip(sorted(seeds), sorted(seeds)[1:])),
          'distinct new in-range calibration seeds required, including one warmup')


def measure(process_plan_path, thread_contract_path, out, *, seeds):
    """Measure empty dispatch and contended kernel service independently of application ingress."""
    process_ref, process, contract_ref, _ = _inputs(process_plan_path, thread_contract_path)
    native, hardware = map(read_artifact, (process['native'], process['hardware']))
    _verify_native(native)
    _seeds(seeds, process, native)
    rows = _eligible(process, _thread_rows(process, native, hardware))
    out = Path(out).resolve()
    out.mkdir(parents=True)
    (out / 'probe.cpp').write_text(_PROBE)
    (out / 'ordered_openmp.hpp').write_bytes(_OPENMP_HEADER.read_bytes())
    (out / 'ordered_processes.hpp').write_bytes(_PROCESS_HEADER.read_bytes())
    environment = _environment(read_artifact(native['recipe']))
    compiler = _compile_command(out, native)
    subprocess.run(compiler, env=environment, check=True, timeout=120)
    compiler_binary, openmp_runtime = _runtime_identity(str(out / 'probe'), compiler[0], native, environment)
    samples = []
    for index, seed in enumerate(seeds):
        order = rows if index % 2 == 0 else reversed(rows)
        for row in order:
            mode, width = row['mode'], row['worker_count']
            for part in ('dispatch', 'kernel'):
                command = _command(str(out / 'probe'), row, process['scope'], seed, part)
                stdout = base._capture(command, env=environment, stdout=out / f'{mode}-{width}-{part}-{index}.txt')
                samples.append(dict(mode=mode, width=width, seed=seed, part=part, command=command, stdout=stdout))
    return base._write(out / 'result.json', dict(schema=EVIDENCE, schema_version=1, process_plan=process_ref,
        thread_contract=contract_ref, native=process['native'], thread_environment=environment, seeds=seeds,
        boundary=_SAMPLE_BOUNDARY, replayable_kernel_state_qualification=contract_ref, compiler=compiler,
        compiler_binary=compiler_binary, openmp_runtime=openmp_runtime,
        probe=base._ref(out / 'probe', 'application/octet-stream'),
        sources={name: base._ref(out / name, 'text/x-c++') for name in ('probe.cpp', 'ordered_openmp.hpp',
                                                                        'ordered_processes.hpp')}, samples=samples))


def _intervals(evidence, process, native, rows):
    sources = evidence['sources']
    expected = {'probe.cpp': _PROBE.encode(), 'ordered_openmp.hpp': _OPENMP_HEADER.read_bytes(),
                'ordered_processes.hpp': _PROCESS_HEADER.read_bytes()}
    _need(sources.keys() == expected.keys() and all(read_artifact(ref, json_document=False) == expected[name]
          for name, ref in sources.items()), 'probe source or reserved dispatch header changed')
    read_artifact(evidence['probe'], json_document=False)
    out = Path(evidence['probe']['uri']).parent
    _need(evidence['schema_version'] == 1 and evidence['boundary'] == _SAMPLE_BOUNDARY
          and evidence['thread_environment'] == _environment(read_artifact(native['recipe']))
          and evidence['compiler'] == _compile_command(out, native), 'probe scope or compiler changed')
    compiler_binary, openmp_runtime = _runtime_identity(evidence['probe']['uri'], evidence['compiler'][0], native,
                                                       evidence['thread_environment'])
    _need(compiler_binary == evidence['compiler_binary'] and openmp_runtime == evidence['openmp_runtime'],
          'compiler or loaded OpenMP runtime changed')
    seeds = evidence['seeds']
    _seeds(seeds, process, native)
    choices = {(row['mode'], row['worker_count']): row for row in rows}
    spans, seen = {}, set()
    for sample in evidence['samples']:
        mode, width, seed, part = (sample[key] for key in ('mode', 'width', 'seed', 'part'))
        key = (mode, width)
        _need(key in choices and seed in seeds and part in ('dispatch', 'kernel')
              and (key, seed, part) not in seen and sample['command'] == _command(evidence['probe']['uri'],
              choices[key], process['scope'], seed, part), 'duplicate or out-of-scope service observation')
        seen.add((key, seed, part))
        elapsed, *lanes = map(int, read_artifact(sample['stdout'], json_document=False).split())
        kernel = part == 'kernel'
        _need(elapsed > 0 and len(lanes) == (width if kernel else process['scope']['task_count'])
              and all(0 < span <= elapsed if kernel else span == 0 for span in lanes), 'invalid probe spans')
        if seed != seeds[0]:
            spans.setdefault((key, part), []).append(lanes if kernel else elapsed)
    estimates = {}
    for key, row in choices.items():
        dispatch = spans.get((key, 'dispatch'), [])
        kernel = spans.get((key, 'kernel'), [])
        if len(dispatch) == len(kernel) == len(seeds) - 1:
            services = {cpu: base._summary(list(series)) for cpu, series in
                        zip(row['worker_cpu_ids'], zip(*kernel, strict=True), strict=True)}
            overhead = base._summary(dispatch)
            if row['mode'] != MODE:
                estimates[key] = base._cost(process['scope']['task_count'], row['worker_cpu_ids'], services, overhead)
                continue
            waves, spill = divmod(process['scope']['task_count'], row['worker_count'])
            numeric = {side: max(services[cpu][side] * (waves + (lane < spill))
                                 for lane, cpu in enumerate(row['worker_cpu_ids'])) for side in base._SIDES}
            estimates[key] = dict(worker_service_intervals_ns={str(cpu): values for cpu, values in services.items()},
                ordered_numeric_completion_ns=numeric, additive_transport_interval_ns=overhead,
                modeled_elapsed_ns={key: numeric[key] + overhead[key] for key in base._SIDES})
    return estimates


def _decision(alternatives, incumbent):
    viable = [row for row in alternatives if not row['capacity_reasons']]
    measured = [row for row in viable if row['cost'] is not None]

    def elapsed(row, side):
        return row['cost']['modeled_elapsed_ns'][side]

    serial = alternatives[0] if alternatives[0] in viable and alternatives[0]['cost'] is not None else None
    incumbent = next((row for row in measured if incumbent is not None
                      and (row['mode'], row['worker_count']) == (incumbent['mode'], incumbent['worker_count'])), None)
    if serial is None:
        status, selected = 'NO_ADMISSIBLE_NATIVE_SERIAL_BASELINE', None
    elif len(viable) != len(measured):
        status, selected = 'KEEP_SERIAL_UNRESOLVED_ALTERNATIVE', serial
    else:
        point = min(measured, key=lambda row: (elapsed(row, 'estimate_ns'), row['worker_count']))
        separated = all(elapsed(point, 'upper_ns') < elapsed(row, 'lower_ns') for row in measured if row is not point)
        if separated and point is not serial:
            status, selected = 'NOMINATE_FINITE_TEAM_FOR_VALIDATION', point
        elif incumbent is not None and incumbent is not serial and elapsed(incumbent, 'upper_ns') < elapsed(serial, 'lower_ns'):
            status, selected = 'KEEP_PROCESS_WHEN_TEAM_ENVELOPES_OVERLAP', incumbent
        else:
            status, selected = 'KEEP_SERIAL_WITHOUT_SEPARATED_GAIN', serial
    return dict(status=status, selected=selected, point_cost_optimum_is_finite_model_only=True,
                deployment_authorized=False)


def project(process_plan_path, thread_cost_path, thread_contract_path):
    """Nominate a finite candidate using certified shared state and independent component evidence."""
    process_ref, process, contract_ref, _ = _inputs(process_plan_path, thread_contract_path)
    evidence_ref = base._ref(thread_cost_path, EVIDENCE)
    evidence = read_artifact(evidence_ref)
    _need(evidence['process_plan'] == process_ref and evidence['thread_contract'] == contract_ref
          and evidence['replayable_kernel_state_qualification'] == contract_ref
          and evidence['native'] == process['native'], 'OpenMP service receipt crosses plans or legality scope')
    native, hardware = map(read_artifact, (process['native'], process['hardware']))
    _verify_native(native)
    rows = _thread_rows(process, native, hardware)
    estimates = _intervals(evidence, process, native, _eligible(process, rows))
    alternatives = [process['alternatives'][0], *(dict(row, cost=estimates.get((row['mode'], row['worker_count'])))
                   for row in (*process['alternatives'][1:], *rows))]
    return dict(process, schema=SCHEMA, schema_version=1, process_plan=process_ref, thread_evidence=evidence_ref,
        thread_contract=contract_ref, alternatives=alternatives, openmp_runtime=evidence['openmp_runtime'],
        decision=_decision(alternatives, process['decision']['selected']),
        thread_domain=dict(assignment='ORDINAL_MOD_WORKER_COUNT', owner_participates=True,
                           publication='ORDERED_AFTER_ALL_LANES_JOIN', shared_result_slots_disjoint=True,
                           thread_environment={'OMP_STACKSIZE': '8M'}, forbidden_environment=['GOMP_STACKSIZE']),
        cost_boundary=dict(basis=_SAMPLE_BOUNDARY, numeric_model='SAMPLE_ENVELOPES_OF_CONTENDED_LANE_SERVICE',
                           additive_empty_dispatch=True, seeded_microbenchmarks_only=True,
                           runtime_environment_and_publication_excluded=True,
                           multi_wave_contention_extrapolated=True, standalone_global_optimum_proven=False))


def check(path):
    expected = read_artifact(base._ref(path, SCHEMA))
    actual = project(*(expected[key]['uri'] for key in ('process_plan', 'thread_evidence', 'thread_contract')))
    _need(expected == actual, 'stored joint decision differs during replay')
    return actual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    verifier = commands.add_parser('check')
    verifier.add_argument('receipt')
    for mode in ('measure', 'project'):
        operation = commands.add_parser(mode)
        for flag in ('process-plan', 'thread-contract', 'out', 'seeds' if mode == 'measure' else 'thread-cost'):
            operation.add_argument('--' + flag, required=True)
    args = parser.parse_args(argv)
    if args.command == 'check':
        result = check(args.receipt)['decision']
    elif args.command == 'measure':
        result = measure(args.process_plan, args.thread_contract, args.out, seeds=list(map(int, args.seeds.split(','))))
    else:
        report = project(args.process_plan, args.thread_cost, args.thread_contract)
        base._write(args.out, report)
        result = report['decision']
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
