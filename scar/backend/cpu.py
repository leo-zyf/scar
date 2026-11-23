"""Price a statically bounded entry against the available host CPU resources."""

from collections import Counter, defaultdict
from dataclasses import asdict
from functools import lru_cache
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
from statistics import median
import subprocess

from ..contract import ElementwiseEntry, artifact


def _text(path):
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _limits():
    memory = [int(row.split()[1]) * 1024 for row in _text(Path("/proc/meminfo")).splitlines()
              if row.startswith("MemAvailable:")]
    quotas = []
    root = Path("/sys/fs/cgroup")
    for row in _text(Path("/proc/self/cgroup")).splitlines():
        if not row.startswith("0::/"):
            continue
        group = root / row.removeprefix("0::/")
        for directory in (group, *group.parents):
            if not directory.is_relative_to(root):
                break
            quota = _text(directory / "cpu.max").split()
            limit = _text(directory / "memory.max")
            current = _text(directory / "memory.current")
            if len(quota) == 2 and all(value.isdigit() for value in quota) and int(quota[1]):
                quotas.append(int(quota[0]) / int(quota[1]))
            if limit.isdigit() and current.isdigit():
                memory.append(max(0, int(limit) - int(current)))
    return min(memory, default=None), min(quotas, default=None)


def _cohort(resources, available):
    domains = defaultdict(dict)
    for resource in resources:
        if resource.kind != "CPU":
            continue
        spec = resource.attributes
        cpu = spec.get("logical_cpu")
        key = (spec.get("package_id"), spec.get("core_id"))
        if cpu in available and None not in key:
            node = spec.get("numa_node")
            # Co-locate within a known memory domain; package bounds missing NUMA facts.
            domain = ("NUMA", node) if node is not None else ("PACKAGE", key[0])
            domains[domain][key] = min(domains[domain].get(key, cpu), cpu)
    largest = min((tuple(sorted(domain.values())) for domain in domains.values()),
                  key=lambda cpus: (-len(cpus), cpus), default=())
    return largest or available[:1]


def _work(body):
    functions = {function["name"]: function for function in body.program["functions"]}

    @lru_cache(maxsize=None)
    def function_cost(name):
        return statements(functions[name]["body"])

    def expression(row):
        kind, *terms = row
        if kind in {"read", "constant"}:
            return Counter()
        if kind == "call":
            result = function_cost(terms[0]).copy()
            terms = terms[1:]
        else:
            result = Counter({kind: 1})
        for operand in terms:
            result.update(expression(operand))
        return result

    def statements(rows):
        result = Counter()
        for kind, *args in rows:
            if kind == "repeat":
                nested = statements(args[0])
                result.update({key: amount * body.count for key, amount in nested.items()})
            else:
                result.update(expression(args[-1]))
        return result

    entry = body.program["entry"]
    elements = math.prod(body.shape)
    inputs = len(functions[entry]["parameters"])
    bytes_ = elements * (inputs + 1) * 8
    return dict(elements=elements, repetitions=body.count, input_arrays=inputs, independent_axis="element",
                loop_carried_axis="per_element_recurrence", peak_resident_bytes=bytes_, stream_bytes=bytes_,
                memory_scope="OWNED_INPUT_AND_OUTPUT_ARRAYS",
                arithmetic={key: amount * elements for key, amount in sorted(function_cost(entry).items())})


def _widths(limit):
    widths = [1]
    while widths[-1] * 2 < limit:
        widths.append(widths[-1] * 2)
    if widths[-1] != limit:
        widths.append(limit)
    return widths


def _probe(directory, cpus, widths, memory, compiler, env):
    source = directory / "cpu-probe.cpp"
    binary = directory / "cpu-probe"
    source.write_text(_PROBE, encoding="utf-8")
    flags = ["-std=c++20", "-O3", "-pthread", "-ffp-contract=off", "-fno-fast-math"]
    subprocess.run([compiler, *flags, str(source), "-o", str(binary)], env=env,
                   capture_output=True, check=True, timeout=20)
    # Each arm has its own bounded vector; the sample need not exceed the LLC.
    elements = min(2**22, memory // 128) if memory is not None else 2**20
    if elements < 2**15:
        raise ValueError("insufficient free memory for a CPU bandwidth probe")
    iterations = 2**20
    command = [str(binary), str(elements), str(iterations), ",".join(map(str, cpus)), ",".join(map(str, widths))]
    result = subprocess.run(command, env=env, capture_output=True,
                            text=True, check=True, timeout=15)
    samples = defaultdict(list)
    for line in result.stdout.splitlines():
        width, setup, copy, arithmetic, divide = map(int, line.split())
        if width not in widths or min(setup, copy, arithmetic, divide) <= 0:
            raise ValueError("unusable CPU probe sample")
        samples[width].append(dict(team_setup_ns=setup, copy_ns=copy, arithmetic_ns=arithmetic, divide_ns=divide))
    if any(len(samples[width]) != 3 for width in widths):
        raise ValueError("incomplete CPU probe")
    return dict(compiler=asdict(artifact(Path(compiler))), flags=flags, probe_sha256=sha256(_PROBE.encode()).hexdigest(),
                copy_sample_bytes=2 * elements * 8, arithmetic_sample_ops=2 * iterations, divide_sample_ops=iterations,
                bandwidth_scope="BOUNDED_HOST_COPY", samples=dict(samples))


def _estimates(work, cpus, probe):
    division = work["arithmetic"].get("div", 0)
    other = sum(work["arithmetic"].values()) - division
    estimates = []
    for width, samples in probe["samples"].items():
        times = []
        for sample in samples:
            setup = sample["team_setup_ns"]
            copies = work["stream_bytes"] * sample["copy_ns"] / probe["copy_sample_bytes"]
            arithmetic = other * sample["arithmetic_ns"] / probe["arithmetic_sample_ops"]
            arithmetic += division * sample["divide_ns"] / probe["divide_sample_ops"]
            times.append(math.ceil(setup + max(copies, arithmetic)))
        estimates.append(dict(workers=width, cpu_ids=list(cpus[:width]), min_ns=min(times),
                              median_ns=median(times), max_ns=max(times)))
    return estimates


def physical_plan(plan, directory, cpu_threads, compiler, environment):
    """Return selected CPU ids and evidence, with a serial fallback on weak evidence."""
    available = tuple(sorted(cpu for cpu in os.sched_getaffinity(0) if 0 <= cpu < 1024))
    explicit = plan.lowering.cpu_ids
    if explicit and (set(explicit) - set(available) or cpu_threads is not None and len(explicit) > cpu_threads):
        raise ValueError("the recorded CPU cohort is unavailable under the current affinity or worker limit")
    body = next((body for body in plan.lowering.bodies if isinstance(body, ElementwiseEntry)), None)
    memory, quota = _limits()
    cohort = _cohort(plan.resources, available)
    count = min(len(cohort), cpu_threads or len(cohort), max(1, int(quota)) if quota is not None else len(cohort))
    cohort = cohort[:count]
    capacity = next((resource.capacity for resource in plan.resources
                     if resource.kind in {"NUMA_MEMORY", "DRAM"} and resource.capacity is not None
                     and (resource.kind == "DRAM" or cohort
                          and cohort[0] in resource.attributes.get("logical_cpus", ()))), None)
    if capacity is not None:
        memory = min(memory, capacity) if memory is not None else capacity
    host = dict(machine=platform.machine(), kernel=platform.release(), affinity=list(available),
                cohort=list(cohort), cpu_quota=quota, available_memory_bytes=memory)
    host["fingerprint"] = sha256(json.dumps({key: value for key, value in host.items()
                                             if key != "available_memory_bytes"}, sort_keys=True).encode()).hexdigest()
    result = dict(schema="scar.physical-plan", schema_version=1, host=host, worker_limit=cpu_threads)
    if body is None:
        reason = "NO_BOUNDED_PARALLEL_BODY" if plan.lowering.bodies else "NO_EXECUTABLE_BODY"
        decision = "PRESERVE_NATIVE_PLACEMENT" if plan.lowering.bodies else "KEEP_ORIGINAL"
        result.update(reason=reason, cpu_ids=list(explicit), decision=decision)
        return explicit, result
    elif not cohort and not explicit:
        raise ValueError("no CPU is available for the C++20 affinity ABI")
    else:
        work = result["work"] = _work(body)
        if memory is not None and work["peak_resident_bytes"] > memory:
            result.update(reason="INSUFFICIENT_AVAILABLE_MEMORY", cpu_ids=[], decision="KEEP_ORIGINAL")
            return (), result
        cpus = explicit or cohort[:1]
        reason = "LEGACY_EXPLICIT_COHORT" if explicit else "ONE_ELIGIBLE_WORKER"
        limit = min(len(cohort), work["elements"])
        if not explicit and limit > 1:
            try:
                probe = _probe(directory, cohort[:limit], _widths(limit), memory, compiler, environment)
                estimates = _estimates(work, cohort, probe)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                reason = "CPU_PROBE_UNAVAILABLE"
                result["probe_error"] = str(error)
            else:
                result.update(probe=probe, estimates=estimates, time_model="TEAM_SETUP_PLUS_MAX_OF_COPY_AND_ARITHMETIC")
                serial = estimates[0]
                # Require a strict benefit across the observed sample ranges.
                eligible = [row for row in estimates[1:] if row["max_ns"] < serial["min_ns"]]
                if eligible:
                    best = min(eligible, key=lambda row: (row["median_ns"], row["workers"]))
                    cpus = tuple(best["cpu_ids"])
                    reason = "PREDICTED_PARALLEL_BENEFIT"
                else:
                    reason = "NO_STABLE_PREDICTED_PARALLEL_BENEFIT"
    result.update(reason=reason, cpu_ids=list(cpus), decision="CPU_TEAM" if len(cpus) > 1 else "CPU_SERIAL")
    chosen = next((row for row in result.get("estimates", ()) if row["cpu_ids"] == list(cpus)), None)
    bounds = {key: chosen[f"{key}_ns"] for key in ("min", "median", "max")} if chosen else None
    result["time_schedule"] = dict(
        time_origin="INPUTS_READY", unit="ns", peak_owned_memory_bytes=work["peak_resident_bytes"],
        windows=[dict(id="input", resource="HOST_CPU_AND_IO", cpu_ids=list(cpus[:1]),
                      depends_on=[], start_ns=None, duration_ns=None),
                 dict(id="compute", resource="HOST_CPU", cpu_ids=list(cpus), depends_on=["input"],
                      start_ns=0 if chosen else None, duration_ns=chosen["median_ns"] if chosen else None,
                      duration_bounds_ns=bounds),
                 dict(id="output", resource="HOST_CPU_AND_IO", cpu_ids=list(cpus[:1]),
                      depends_on=["compute"], start_ns=None, duration_ns=None)])
    result["unmodeled_choices"] = ["ISA_AND_SIMD", "CROSS_NUMA", "OPERATOR_LIBRARY_VARIANTS"]
    return tuple(cpus), result


_PROBE = r"""#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sched.h>
#include <sstream>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
std::vector<int> numbers(const char* arg) {
    std::istringstream input(arg);
    std::vector<int> values;
    for (std::string value; std::getline(input, value, ',');) values.push_back(std::stoi(value));
    return values;
}

bool pin(int cpu) {
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    CPU_SET(cpu, &affinity);
    return sched_setaffinity(0, sizeof(affinity), &affinity) == 0;
}

template<class Task> auto sample(const std::vector<int>& cpus, int count, Task task, bool measure_work = true) {
    std::atomic<bool> pinned{true};
    std::barrier ready{count};
    std::vector<Clock::time_point> begins(count), ends(count);
    const auto start = Clock::now();
    auto run = [&](int worker) {
        if (!pin(cpus[worker])) pinned.store(false);
        if (measure_work) {
            ready.arrive_and_wait();
            begins[worker] = Clock::now();
        }
        task(worker);
        if (measure_work) ends[worker] = Clock::now();
    };
    {
        std::vector<std::jthread> threads;
        for (int k = 1; k < count; ++k) threads.emplace_back(run, k);
        run(0);
    }
    if (!pinned.load()) std::exit(2);
    const auto elapsed = measure_work ? *std::max_element(ends.begin(), ends.end())
                                       - *std::min_element(begins.begin(), begins.end())
                                     : Clock::now() - start;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
}

int main(int argc, char** argv) {
    if (argc != 5) return 1;
    const long elements = std::stol(argv[1]), iterations = std::stol(argv[2]);
    const auto cpus = numbers(argv[3]), widths = numbers(argv[4]);
    if (!pin(cpus.front())) return 2;
    std::vector<double> source(elements, 1.001), target(elements), results(cpus.size());
    double observable = 0;
    for (int width : widths) {
        constexpr int samples = 3;
        for (int k = 0; k < samples; ++k) {
            long setup[5];
            for (auto& value : setup) value = sample(cpus, width, [](int) {}, false);
            std::sort(std::begin(setup), std::end(setup));
            const auto copy = sample(cpus, width, [&](int worker) {
                const auto start = elements * worker / width, stop = elements * (worker + 1) / width;
                std::memcpy(target.data() + start, source.data() + start, (stop - start) * sizeof(double));
            });
            const auto math = sample(cpus, width, [&](int worker) {
                const auto count = iterations * (worker + 1) / width - iterations * worker / width;
                double x = worker + 1.01;
                for (long i = 0; i < count; ++i) x = x * 0.99999997 + 0.00000002;
                results[worker] = x;
            });
            for (int worker = 0; worker < width; ++worker) observable += results[worker];
            const auto div = sample(cpus, width, [&](int worker) {
                const auto count = iterations * (worker + 1) / width - iterations * worker / width;
                double x = worker + 1.01;
                for (long i = 0; i < count; ++i) x = 2.010001 / (x + 1.001);
                results[worker] = x;
            });
            for (int worker = 0; worker < width; ++worker) observable += results[worker];
            observable += target[k];
            std::cout << width << ' ' << setup[2] << ' ' << copy << ' ' << math << ' ' << div << '\n';
        }
    }
    return observable > 0 ? 0 : 2;
}
"""
