"""Measure and rank local MPI candidates before lowering an application."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
from time import perf_counter_ns

from scar.ir.execution_recipe import read_artifact
from scar.optimization import independent_array_task_schedule as base
from scar.optimization import independent_array_task_team_schedule as team


SCHEMA = 'scar.independent-array-task-mpi-schedule'
EVIDENCE = 'scar.independent-array-task-mpi-cost-evidence'
MODE = 'ORDERED_LOCAL_MPI'
_BOUNDARY = 'FRESH_COMPONENT_EXEC_THROUGH_EXIT_EXCLUDING_APPLICATION_PUBLICATION'
_TIMEOUT_SECONDS = 30


_PROBE = r'''#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

extern "C" int scar_array_entry_profile_0(double*, double*, const char*, std::uint64_t, std::uint64_t);

namespace {
using Clock = std::chrono::steady_clock;

std::int64_t since(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}

struct Row {
    std::int64_t ordinal{}, service_ns{}, bootstrap_ns{}, cpu{}, pid{};
    double norm{}, difference{};
};
static_assert(sizeof(Row) == 56);

bool bind(int cpu) {
    cpu_set_t allowed;
    if (cpu < 0 || cpu >= CPU_SETSIZE || sched_getaffinity(0, sizeof(allowed), &allowed) || !CPU_ISSET(cpu, &allowed)) {
        return false;
    }
    CPU_ZERO(&allowed);
    CPU_SET(cpu, &allowed);
    return sched_setaffinity(0, sizeof(allowed), &allowed) == 0 && sched_getcpu() == cpu;
}
}

int main(int argc, char** argv) {
    const auto start = Clock::now();
    if (argc < 7) return 2;
    const bool numeric = std::string_view(argv[1]) == "kernel";
    if (!numeric && std::string_view(argv[1]) != "exchange") return 2;
    const int count = std::stoi(argv[2]);
    const auto seed = std::stoull(argv[3]);
    const auto rounds = std::stoull(argv[4]);
    const int width = argc - 5;
    if (count < width || count > INT32_MAX / sizeof(Row) || (numeric && count != width)) return 2;
    const char* launcher_rank = std::getenv("OMPI_COMM_WORLD_RANK");
    const int observed = launcher_rank ? std::stoi(launcher_rank) : -1;
    const int cpu = 0 <= observed && observed < width ? std::stoi(argv[5 + observed]) : -1;
    const bool pinned = bind(cpu);
    int provided = -1;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided) != MPI_SUCCESS) return 3;
    int rank = -1, world = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    const auto bootstrap = since(start);
    int local_status = pinned && rank == observed && world == width && sched_getcpu() == cpu
                       && provided == MPI_THREAD_SINGLE ? 0 : 1;
    const auto row_count = [&](int owner) { return (count - 1 - owner) / width + 1; };
    std::vector<Row> rows;
    if (!local_status) {
        rows.resize(row_count(rank));
        for (std::size_t index = 0; index < rows.size(); ++index) {
            auto& row = rows[index];
            row.ordinal = rank + static_cast<int>(index) * width;
            row.bootstrap_ns = bootstrap;
            row.cpu = cpu;
            row.pid = getpid();
            if (!numeric) continue;
            const auto begin = Clock::now();
            const int status = scar_array_entry_profile_0(&row.norm, &row.difference, nullptr,
                                                           seed + row.ordinal, rounds);
            row.service_ns = since(begin);
            if (status || !std::isfinite(row.norm) || !std::isfinite(row.difference) || sched_getcpu() != cpu) {
                local_status = 1;
                break;
            }
        }
    }
    const auto collect_begin = Clock::now();
    int status = 1;
    MPI_Allreduce(&local_status, &status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (status) {
        MPI_Finalize();
        return 4;
    }
    std::vector<int> sizes, displacements;
    std::vector<Row> gathered, ordered;
    if (rank == 0) {
        gathered.resize(count);
        ordered.resize(count);
        for (int owner = 0, offset = 0; owner < width; ++owner) {
            displacements.push_back(offset);
            const int actual = row_count(owner) * sizeof(Row);
            sizes.push_back(actual);
            offset += actual;
        }
    }
    MPI_Gatherv(rows.data(), rows.size() * sizeof(Row), MPI_BYTE, gathered.data(), sizes.data(),
                displacements.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (int owner = 0; owner < width; ++owner) {
            for (int index = 0; index < row_count(owner); ++index) {
                const auto& row = gathered[displacements[owner] / sizeof(Row) + index];
                if (row.ordinal != owner + index * width || row.cpu != std::stoi(argv[5 + owner])) status = 1;
                ordered[owner + index * width] = row;
            }
        }
    }
    const auto collected = since(collect_begin);
    const auto finalize_begin = Clock::now();
    MPI_Finalize();
    if (status) return 5;
    if (rank == 0) {
        const auto finalized = since(finalize_begin);
        std::cout << since(start) << ' ' << collected << ' ' << finalized << '\n';
        for (const auto& row : ordered) {
            std::cout << row.ordinal << ' ' << row.service_ns << ' ' << row.bootstrap_ns << ' '
                      << row.cpu << ' ' << row.pid << '\n';
        }
    }
}
'''


def _need(condition, detail):
    if not condition:
        raise ValueError('independent array task MPI schedule: ' + detail)


def _inputs(graph_path, *, replayed_graph=None):
    from scar.analysis import native_array_mpi_graph as analysis

    graph_ref = base._ref(graph_path, analysis.SCHEMA)
    if replayed_graph is None:
        graph = analysis.check(graph_ref['uri'])
    else:
        _need(graph_ref == replayed_graph, 'preflight does not identify the target graph')
        graph = read_artifact(graph_ref)
    prior = read_artifact(graph['team_plan'])
    native = read_artifact(graph['native'])
    _need(all(graph[key] == prior[key] for key in ('scope', 'family', 'native', 'hardware')),
          'MPI graph must preserve the proved task family, parameters and hardware')
    _need(all(row['mode'] == MODE for row in graph['alternatives']), 'MPI realization domain contains another backend')
    team._verify_native(native)
    return graph_ref, graph, prior, native, read_artifact(native['recipe'])


def _compiler(out, graph, native):
    flags = [flag for flag in team._compile_command(out, native)[1:]
             if flag != '-fopenmp' and not flag.startswith('-DSCAR_PROBE_BLAS_CORE=')]
    return [graph['runtime']['compiler_path'], *flags]


def _runtime_identity(probe, graph, native, environment):
    from scar.codegen.native_provider_bundle import _dependencies

    runtime = graph['runtime']
    dependencies = _dependencies(subprocess.check_output(['/usr/bin/ldd', str(probe)],
                                                          env=environment, text=True, timeout=10))
    libmpi = [Path(path).resolve() for name, path in dependencies.items() if name.startswith('libmpi.so.')]
    _need(len(libmpi) == 1 and str(libmpi[0]) == runtime['libmpi']['uri'], 'probe MPI library differs from graph')
    _need(all(dependencies.get(Path(ref['uri']).name) == ref['uri'] for name, ref in native['files'].items()
              if name.startswith('provider/objects/')), 'probe does not use the same numerical libraries')
    return base._ref(libmpi[0], runtime['libmpi']['schema'])


def _seeds(seeds, prior, native, screened=None):
    team._seeds(seeds, prior, native)
    width = prior['scope']['task_count']
    previous = [(seed, width) for seed in read_artifact(prior['thread_evidence'])['seeds']]
    if screened:
        platform_width = screened['platform_scope']['task_count']
        previous.extend((seed, platform_width) for seed in read_artifact(screened['platform_evidence'])['seeds'])
        if 'local_acceptance' in screened:
            local = read_artifact(screened['local_acceptance'])
            heldout = read_artifact(local['holdout'])
            seed_argument = read_artifact(prior['family'])['input_options']['seed']
            previous.append((heldout['parameters'][seed_argument], width))
    _need(not any(max(seed, old) < min(seed + width, old + old_width)
                  for seed in seeds for old, old_width in previous),
          'MPI component samples must be disjoint from prior evidence and accepted inputs')


def _mpi_command(graph, probe, row, seed, part):
    from scar.analysis.native_array_mpi_graph import command

    width = row['worker_count']
    scope = graph['scope']
    count = width if part == 'kernel' else scope['task_count']
    return command(row, probe, [part, str(count), str(seed), str(scope['round_count']), *map(str, row['worker_cpu_ids'])])


def _local_command(prior, choice, native, recipe, team_probe, seed):
    if choice['mode'] == 'IN_PROCESS_SERIAL':
        _, arguments = base._parameters(native, recipe, seed, prior['scope']['round_count'])
        return base._worker_command(choice['owner_cpu_id'], native['files'][native['binary']],
                                    choice['memory']['coordinator_address_space_limit_bytes'], arguments)
    return team._command(team_probe, choice, prior['scope'], seed, 'dispatch')


def _capture(command, environment, path):
    start = perf_counter_ns()
    with subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          start_new_session=True) as child:
        try:
            stdout, stderr = child.communicate(timeout=_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.communicate(timeout=5)
            raise
    elapsed = perf_counter_ns() - start
    _need(child.returncode == 0 and not stderr,
          f'component probe failed ({child.returncode}): {stderr.decode(errors="replace").strip()}')
    path.write_bytes(stdout)
    return dict(command=command, wall_ns=elapsed, stdout=base._ref(path, 'application/octet-stream'))


def _choices(rows):
    return {(row['mode'], row['worker_count']): row for row in rows if not row['capacity_reasons']}


def _screening(reference):
    if reference is None:
        return None, {}
    from scar.optimization import independent_array_task_mpi_preflight as preflight

    checked = preflight.check(reference['uri'])
    _need(read_artifact(reference) == checked, 'preflight differs from the retained evidence')
    pruned = {row['rank_count']: row for row in checked['decision']['mpi_outcomes']
              if row['outcome'] == preflight._REJECTED}
    return checked, pruned


def measure(graph_path, out, *, seeds=None, plan_out=None, preflight_path=None):
    """Sample only exact-launch empty exchange and isolated contended kernel calls."""
    preflight_ref = None
    if preflight_path is not None:
        from scar.optimization import independent_array_task_mpi_preflight as preflight

        preflight_ref = base._ref(preflight_path, preflight.SCHEMA)
    screened, pruned = _screening(preflight_ref)
    inputs = _inputs(graph_path, replayed_graph=screened['mpi_graph'] if screened else None)
    graph_ref, graph, prior, native, recipe = inputs
    mpi_rows = {key: row for key, row in _choices(graph['alternatives']).items() if key[1] not in pruned}
    if screened and not mpi_rows:
        if plan_out is not None:
            base._write(plan_out, screened)
        return dict(screened['decision'], action='SKIP_TARGET_MPI_PROBE')
    _need(seeds is not None, 'measure requires --seeds for unresolved MPI rank plans')
    _seeds(seeds, prior, native, screened)
    local_rows = _choices(prior['alternatives'])
    out = Path(out).resolve()
    out.mkdir(parents=True)
    (out / 'probe.cpp').write_text(_PROBE)
    environment = recipe['environment']
    _need(not (set(environment) & {'SCAR_ARRAY_ENTRY_STATES', 'SCAR_ARRAY_ENTRY_LANE_REPORT'})
          and not any(name.startswith(('OMPI_', 'PMIX_', 'OPAL_', 'LD_')) for name in environment),
          'MPI measurements require the sealed original environment without launcher overrides')
    compiler = _compiler(out, graph, native)
    subprocess.run(compiler, env=environment, check=True, timeout=120)
    probe = base._ref(out / 'probe', 'application/octet-stream')
    provider = _runtime_identity(probe['uri'], graph, native, environment)
    old = read_artifact(prior['thread_evidence'])
    team_probe = old['probe']
    read_artifact(team_probe, json_document=False)
    samples = []
    for index, seed in enumerate(seeds):
        order = list(mpi_rows.items()) if index % 2 == 0 else list(reversed(mpi_rows.items()))
        for (mode, width), choice in order:
            for part in ('exchange', 'kernel'):
                command = _mpi_command(graph, probe['uri'], choice, seed, part)
                raw = _capture(command, environment, out / f'mpi-{width}-{part}-{index}.txt')
                samples.append(dict(mode=mode, width=width, seed=seed, part=part, **raw))
        for (mode, width), choice in local_rows.items():
            command = _local_command(prior, choice, native, recipe, team_probe['uri'], seed)
            env = environment if mode == 'IN_PROCESS_SERIAL' else old['thread_environment']
            raw = _capture(command, env, out / f'{mode}-{width}-startup-{index}.txt')
            samples.append(dict(mode=mode, width=width, seed=seed, part='startup', **raw))
    evidence = dict(schema=EVIDENCE, schema_version=1, mpi_graph=graph_ref,
        team_plan=graph['team_plan'], native=graph['native'], boundary=_BOUNDARY, seeds=seeds, samples=samples,
        timeout_seconds=_TIMEOUT_SECONDS, environment=environment, team_probe=team_probe, compiler=compiler,
        probe=probe, probe_source=base._ref(out / 'probe.cpp', 'text/x-c++'), loaded_libmpi=provider)
    if preflight_ref is not None:
        evidence.update(preflight=preflight_ref, measured_mpi_rank_counts=sorted(width for _, width in mpi_rows))
    cost_ref = base._write(out / 'result.json', evidence)
    if plan_out is not None:
        plan = _project(inputs, cost_ref, preflight_ref=preflight_ref, pruned=pruned)
        base._write(plan_out, plan)
        return plan['decision']
    return cost_ref


def _series(raw, wall_ns, cpus, count, numeric):
    spans, *lines = [list(map(int, line.split())) for line in raw.splitlines()]
    entry, collected, finalized = spans
    _need(0 < collected + finalized <= entry <= wall_ns and collected > 0 and finalized > 0 and len(lines) == count,
          'malformed MPI timing boundary')
    width = len(cpus)
    owners, bootstrap, kernels = {}, {}, []
    for position, row in enumerate(lines):
        ordinal, service, pre, cpu, pid = row
        lane = position % width
        _need(ordinal == position and cpu == cpus[lane] and pid > 0 and 0 < pre <= wall_ns
              and (0 < service <= wall_ns if numeric else service == 0), 'MPI rank, affinity or service receipt differs')
        _need(owners.setdefault(lane, pid) == pid and bootstrap.setdefault(lane, pre) == pre,
              'ordinal moved out of its static rank assignment')
        kernels.append(service)
    _need(len(owners) == len(set(owners.values())) == width, 'one actual process per MPI rank is required')
    return dict(kernels=kernels, empty_external_ns=wall_ns, rank0_entry_ns=entry,
                launcher_and_return_tail_ns=wall_ns - entry, max_rank_bootstrap_ns=max(bootstrap.values()),
                status_and_gather_ns=collected, finalize_ns=finalized)


def _comparable(prior, graph, evidence, native, recipe, *, screened, pruned):
    team_probe = read_artifact(prior['thread_evidence'])['probe']
    probe = evidence['probe']
    directory = Path(probe['uri']).parent
    _need(evidence['schema_version'] == 1 and evidence['boundary'] == _BOUNDARY
          and evidence['environment'] == recipe['environment'] and evidence['timeout_seconds'] == _TIMEOUT_SECONDS
          and evidence['compiler'] == _compiler(directory, graph, native) and evidence['team_probe'] == team_probe
          and read_artifact(evidence['probe_source'], json_document=False) == _PROBE.encode(),
          'cost probe source, environment or timing boundary changed')
    read_artifact(probe, json_document=False)
    read_artifact(team_probe, json_document=False)
    _need(_runtime_identity(probe['uri'], graph, native, evidence['environment']) == evidence['loaded_libmpi'],
          'loaded MPI library changed')
    seeds = evidence['seeds']
    _seeds(seeds, prior, native, screened)
    mpi = {key: row for key, row in _choices(graph['alternatives']).items() if key[1] not in pruned}
    local = _choices(prior['alternatives'])
    if 'preflight' in evidence:
        _need(evidence['measured_mpi_rank_counts'] == sorted(width for _, width in mpi),
              'measurement domain differs from unresolved eligible MPI rank plans')
    parameter_field = read_artifact(recipe['source_contract'])['report']['parameters']['field']
    spans, seen = {}, set()
    for sample in evidence['samples']:
        key = sample['mode'], sample['width']
        part, seed, wall = (sample[name] for name in ('part', 'seed', 'wall_ns'))
        identity = key, part, seed
        _need(identity not in seen and seed in seeds and type(wall) is int and 0 < wall < _TIMEOUT_SECONDS * 10**9,
              'invalid or duplicate probe measurement')
        seen.add(identity)
        raw = read_artifact(sample['stdout'], json_document=False)
        if key in mpi:
            choice = mpi[key]
            _need(part in ('exchange', 'kernel') and sample['command'] == _mpi_command(graph, probe['uri'], choice,
                                                                                      seed, part), 'MPI launch changed')
            numeric = part == 'kernel'
            count = choice['worker_count'] if numeric else graph['scope']['task_count']
            value = _series(raw, wall, choice['worker_cpu_ids'], count, numeric)
        else:
            _need(key in local and part == 'startup' and sample['command'] ==
                  _local_command(prior, local[key], native, recipe, team_probe['uri'], seed), 'local launch changed')
            if key[0] == 'IN_PROCESS_SERIAL':
                inner = base._worker_seconds(native, recipe, raw, seed, graph['scope']['round_count'], parameter_field)
            else:
                inner, *zeros = map(int, raw.split())
                _need(len(zeros) == graph['scope']['task_count'] and not any(zeros), 'old dispatcher ran a numeric task')
            _need(0 < inner < wall, 'local component did not measure startup outside the modeled span')
            value = wall - inner
        if seed != seeds[0]:
            spans.setdefault((key, part), []).append(value)
    _need(len(seen) == len(seeds) * (2 * len(mpi) + len(local)), 'incomplete alternatives or seeds')
    return spans


def _estimate_mpi(choice, count, samples):
    key = choice['mode'], choice['worker_count']
    cpus = choice['worker_cpu_ids']
    exchange = samples[(key, 'exchange')]
    kernel = samples[(key, 'kernel')]
    overhead = base._summary([sample['empty_external_ns'] for sample in exchange])
    lane_services = {str(cpu): base._summary(list(values)) for cpu, values in
                     zip(cpus, zip(*(sample['kernels'] for sample in kernel), strict=True), strict=True)}
    quotient, spill = divmod(count, len(cpus))
    numeric = {side: max(lane_services[str(cpu)][side] * (quotient + (index < spill))
                         for index, cpu in enumerate(cpus)) for side in base._SIDES}
    stages = {name: base._summary([sample[name] for sample in exchange]) for name in (
        'rank0_entry_ns', 'launcher_and_return_tail_ns', 'max_rank_bootstrap_ns', 'status_and_gather_ns', 'finalize_ns')}
    return dict(worker_service_intervals_ns=lane_services, ordered_numeric_completion_ns=numeric,
                additive_transport_interval_ns=overhead, mpi_empty_component_observations_ns=stages,
                modeled_elapsed_ns={side: overhead[side] + numeric[side] for side in base._SIDES})


def _estimate_local(choice, samples):
    key = choice['mode'], choice['worker_count']
    internal = choice['cost']
    if internal is None or choice['capacity_reasons']:
        return None
    startup = base._summary(samples[(key, 'startup')])
    return dict(internal, host_startup_and_return_ns=startup,
                modeled_elapsed_ns={side: internal['modeled_elapsed_ns'][side] + startup[side] for side in base._SIDES})


def _decide(previous, locals_, mpi, *, screened=None):
    selectable = [row for row in locals_ if not row['capacity_reasons'] and row['cost'] is not None]
    finite = [row for row in mpi if not row['capacity_reasons']]
    screened = {} if screened is None else screened
    _need(type(screened) is dict and all(type(width) is int for width in screened)
          and screened.keys() <= {row['worker_count'] for row in finite if row['cost'] is None},
          'bootstrap screening may only price unmeasured feasible ranks')
    incumbent = next((row for row in selectable if previous and (row['mode'], row['worker_count']) ==
                      (previous['mode'], previous['worker_count'])), None)

    def estimate(choice, side):
        return choice['cost']['modeled_elapsed_ns'][side]

    best_local = min(selectable, key=lambda row: estimate(row, 'upper_ns'), default=None)
    local_upper = estimate(best_local, 'upper_ns') if best_local else None
    outcomes = []
    for row in mpi:
        width = row['worker_count']
        screening = screened.get(width)
        mpi_lower = estimate(row, 'lower_ns') if row['cost'] else None
        provenance = {}
        if screening is not None:
            mpi_lower = screening['mpi_lower_ns']
            screened_upper = screening['local_upper_ns']
            _need(screening['rank_count'] == width and screening['outcome'] == 'DOMINATED_BY_LOCAL_WITH_BOOTSTRAP_ALONE'
                  and type(mpi_lower) is int and type(screened_upper) is int and mpi_lower > screened_upper >= 0,
                  'invalid bootstrap screening bound')
            provenance = dict(screening=dict(preflight_local_upper_ns=screened_upper, mpi_probe_measured=False))
        if row['capacity_reasons']:
            outcome = 'INSUFFICIENT_RESOURCE_CAPACITY'
        elif screening is not None:
            outcome = ('DOMINATED_BY_LOCAL_WITH_BOOTSTRAP_ALONE' if local_upper is not None and mpi_lower > local_upper
                       else 'UNRESOLVED_SCREENED_BOOTSTRAP_OVERLAPS_LOCAL')
        elif local_upper is None or mpi_lower is None:
            outcome = 'UNRESOLVED_WITHOUT_COMPARABLE_INTERVALS'
        elif mpi_lower > local_upper:
            outcome = 'DOMINATED_BY_EXISTING_LOCAL_BACKEND'
        else:
            outcome = 'NO_DEFINITE_DOMINATION'
        local_bounds = [dict(mode=local['mode'], worker_count=local['worker_count'],
                             upper_ns=estimate(local, 'upper_ns'),
                             dominates_mpi=(mpi_lower > estimate(local, 'upper_ns') if mpi_lower is not None else None))
                        for local in selectable]
        outcomes.append(dict(rank_count=width, outcome=outcome, mpi_lower_ns=mpi_lower, local_upper_ns=local_upper,
                             local_mode=best_local['mode'] if best_local else None,
                             local_worker_count=best_local['worker_count'] if best_local else None,
                             competing_local_bounds=local_bounds, **provenance))
    confirmed_screening = [row for row in outcomes if row['outcome'] == 'DOMINATED_BY_LOCAL_WITH_BOOTSTRAP_ALONE']
    dominated = bool(finite) and all(row['outcome'] in ('INSUFFICIENT_RESOURCE_CAPACITY',
                                                      'DOMINATED_BY_LOCAL_WITH_BOOTSTRAP_ALONE',
                                                      'DOMINATED_BY_EXISTING_LOCAL_BACKEND') for row in outcomes)
    measured = [row for row in (*selectable, *finite) if row['cost'] is not None]
    proposal = min((row for row in finite if row['cost'] is not None),
                   key=lambda row: estimate(row, 'estimate_ns'), default=None)
    separated = proposal is not None and len(measured) + len(confirmed_screening) == len(finite) + len(
        [row for row in locals_ if not row['capacity_reasons']]) and all(
            estimate(proposal, 'upper_ns') < estimate(row, 'lower_ns') for row in measured if row is not proposal) and all(
            estimate(proposal, 'upper_ns') < row['mpi_lower_ns'] for row in confirmed_screening)
    if dominated:
        status = 'ALL_ELIGIBLE_MPI_DOMINATED'
    elif not finite:
        status = 'NO_MPI_CANDIDATE_FITS_RESOURCE_BOUNDS'
    elif best_local is not None and separated:
        status = 'NOMINATE_MPI_FOR_INDEPENDENT_VALIDATION'
    else:
        status = 'KEEP_LOCAL_BACKEND_WITHOUT_SEPARATED_MPI_GAIN'
    selected = proposal if status == 'NOMINATE_MPI_FOR_INDEPENDENT_VALIDATION' else incumbent
    return dict(status=status, selected=selected, all_eligible_mpi_rejected=dominated,
                mpi_outcomes=outcomes, deployment_authorized=False)


def _project(inputs, cost_ref, *, preflight_ref=None, pruned=None):
    graph_ref, graph, prior, native, recipe = inputs
    evidence = read_artifact(cost_ref)
    pruned = pruned or {}
    screened = read_artifact(preflight_ref) if preflight_ref is not None else None
    _need(evidence['mpi_graph'] == graph_ref and evidence['team_plan'] == graph['team_plan']
          and evidence['native'] == graph['native'] and evidence.get('preflight') == preflight_ref
          and ('measured_mpi_rank_counts' in evidence) == (preflight_ref is not None),
          'MPI cost evidence crosses task graph or screening context')
    samples = _comparable(prior, graph, evidence, native, recipe, screened=screened, pruned=pruned)
    locals_ = [dict(row, cost=_estimate_local(row, samples)) for row in prior['alternatives']]
    mpi = [dict(row, cost=None if row['capacity_reasons'] or row['worker_count'] in pruned
                else _estimate_mpi(row, graph['scope']['task_count'], samples))
           for row in graph['alternatives']]
    report = dict(schema=SCHEMA, schema_version=1, mpi_graph=graph_ref, mpi_evidence=cost_ref,
        **{key: graph[key] for key in ('scope', 'family', 'native', 'hardware', 'team_plan')},
        alternatives=[*locals_, *mpi], decision=_decide(prior['decision']['selected'], locals_, mpi, screened=pruned),
        cost_boundary=dict(basis=_BOUNDARY, kernel_service='ONE_NUMERIC_TASK_PER_RANK_MEASURED_CONCURRENTLY',
                           multi_wave_service='EXTRAPOLATE_PER_RANK_ORDINAL_COUNT',
                           mpi_launch_is_charged_once=True, publication_cost_common_and_excluded=True,
                           components_within_mpi_empty_launch_not_added_again=True,
                           envelopes_are_sample_extrema=True, input_holdouts_used=False,
                           executable_validation_required_after_nomination=True))
    if preflight_ref is not None:
        report.update(mpi_preflight=preflight_ref, measured_mpi_rank_counts=evidence['measured_mpi_rank_counts'],
                      unmeasured_mpi_rank_counts=sorted(pruned))
        report['cost_boundary']['input_holdouts_used'] = 'local_acceptance' in screened
    return report


def project(graph_path, cost_path):
    """Replay source, resources and independent components to reject or nominate MPI."""
    cost_ref = base._ref(cost_path, EVIDENCE)
    preflight_ref = read_artifact(cost_ref).get('preflight')
    screened, pruned = _screening(preflight_ref)
    inputs = _inputs(graph_path, replayed_graph=screened['mpi_graph'] if screened else None)
    return _project(inputs, cost_ref, preflight_ref=preflight_ref, pruned=pruned)


def check(path):
    recorded = read_artifact(base._ref(path, SCHEMA))
    checked = project(recorded['mpi_graph']['uri'], recorded['mpi_evidence']['uri'])
    _need(recorded == checked, 'stored MPI decision differs during replay')
    return checked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for mode in ('measure', 'project'):
        child = commands.add_parser(mode)
        child.add_argument('--mpi-graph', required=True)
        child.add_argument('--out', required=True)
        if mode == 'measure':
            child.add_argument('--seeds')
            child.add_argument('--preflight', help='apply a prior screening receipt before launching probes')
            child.add_argument('--plan-out', help='derive a plan immediately from these verified component receipts')
        else:
            child.add_argument('--mpi-cost', required=True)
    preflight = commands.add_parser('preflight', help='price optimistic MPI completion without a target run')
    preflight.add_argument('--mpi-graph', required=True)
    preflight.add_argument('--platform-plan', required=True)
    preflight.add_argument('--local-acceptance', help='use a previously accepted local entry as an upper estimate')
    preflight.add_argument('--out', required=True)
    commands.add_parser('check').add_argument('receipt')
    args = parser.parse_args(argv)
    if args.command == 'measure':
        seeds = list(map(int, args.seeds.split(','))) if args.seeds is not None else None
        result = measure(args.mpi_graph, args.out, seeds=seeds, plan_out=args.plan_out, preflight_path=args.preflight)
    elif args.command == 'project':
        result = project(args.mpi_graph, args.mpi_cost)
        base._write(args.out, result)
        result = result['decision']
    elif args.command == 'preflight':
        from scar.optimization import independent_array_task_mpi_preflight

        report = independent_array_task_mpi_preflight.project(
            args.mpi_graph, args.platform_plan, local_acceptance_path=args.local_acceptance)
        base._write(args.out, report)
        result = report['decision']
    else:
        schema = json.loads(Path(args.receipt).read_bytes())['schema']
        if schema == SCHEMA:
            result = check(args.receipt)['decision']
        else:
            from scar.optimization import independent_array_task_mpi_preflight

            result = independent_array_task_mpi_preflight.check(args.receipt)['decision']
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
