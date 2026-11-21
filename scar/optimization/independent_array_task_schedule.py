"""Schedule a proved independent task family on one bounded local CPU team."""

import argparse
import json
from math import ceil, isfinite
import os
from pathlib import Path
from statistics import median
import subprocess

from scar.analysis.sparse_parallel_scaling_scope import _lanes
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.sparse_parallel_cost import HARDWARE, collect_hardware


SCHEMA = 'scar.independent-array-task-schedule'
EVIDENCE = 'scar.independent-array-task-cost-evidence'
NATIVE = 'scar.native-seeded-array-loop-entry'
REQUEST_BYTES = 16
RESULT_BYTES = 56
_ORDINAL_WIRE_BYTES = 8
_RESPONSE_WIRE_BYTES = 16
_DISPATCH_HEADER = Path(__file__).resolve().parents[1] / 'codegen/native_ordered_dispatch_v1.h.in'
_SIDES = ('lower_ns', 'estimate_ns', 'upper_ns')
_MAX_I64 = 2**63 - 1


def _need(condition, detail):
    if not condition:
        raise ValueError('independent array task schedule: ' + detail)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _write(path, document):
    with Path(path).open('x', encoding='utf-8') as output:
        json.dump(document, output, sort_keys=True, indent=2, allow_nan=False)
        output.write('\n')
    return _ref(path, document['schema'])


def collect(path, cpu_ids):
    """Capture permitted CPU cores, NUMA topology and process headroom."""
    _need(cpu_ids and len(set(cpu_ids)) == len(cpu_ids) and set(cpu_ids) <= os.sched_getaffinity(0),
          'explicit, unique and currently permitted CPU cohort required')
    return _write(path, collect_hardware(cpu_ids))


def _scope(task_count, requested_workers, round_count):
    values = dict(task_count=task_count, requested_workers=requested_workers, round_count=round_count)
    _need(all(type(value) is int and 1 <= value <= _MAX_I64 for value in values.values()),
          'positive finite task, original worker and inner repetition counts required')
    return values


def _cohort(hardware, owner):
    processors = {row['cpu_id']: row for row in hardware['processors']}
    _need(owner in processors and type(owner) is int, 'owner must belong to the observed CPU cohort')
    row = processors[owner]
    _need(len(row['numa_nodes']) == 1, 'owner needs exactly one known NUMA node')
    node, = row['numa_nodes']
    cpus = [owner, *sorted(cpu for cpu in processors if cpu != owner)]
    return node, _lanes(hardware, cpus, node)


def _summary(samples):
    if not samples:
        return None
    _need(len(samples) >= 3 and all(type(value) is int and value > 0 for value in samples),
          'service needs at least three positive observations')
    return dict(lower_ns=min(samples), estimate_ns=ceil(median(samples)), upper_ns=max(samples))


def _parameters(native, recipe, seed, rounds):
    values = dict(recipe['profiles'][0]['parameters'])
    values.update({native['seed_option']: seed, native['round_option']: rounds})
    flags = [recipe['options'][name]['flags'] for name in values]
    _need(all(len(group) == 1 for group in flags), 'service probe requires one flag per specialized input')
    return values, [word for name, value in values.items() for word in (recipe['options'][name]['flags'][0], str(value))]


def _worker_seconds(native, recipe, raw, seed, rounds, parameter_field):
    data = json.loads(raw)
    expected, _ = _parameters(native, recipe, seed, rounds)
    expected[recipe['tolerance_option']] = None
    bindings = recipe['report_bindings']
    path, = (row['path'] for row in bindings if row['role'] == 'worker_wall')
    value = data
    for key in path:
        value = value[key]
    _need(data[parameter_field] == expected and type(value) is float and isfinite(value) and value > 0,
          'worker service report does not match the configured original ingress or timed span')
    return ceil(value * 1_000_000_000)


def _worker_command(cpu, binary, limit, argv):
    return ['/usr/bin/taskset', '-c', str(cpu), '/usr/bin/prlimit', f'--as={limit}:{limit}',
            binary['uri'], *argv]


def _transport_command(probe, launch_cpus, slot_cpus, count, worker_limit, owner_limit):
    return ['/usr/bin/taskset', '-c', ','.join(map(str, launch_cpus)), str(probe),
            str(count), str(worker_limit), str(owner_limit), *map(str, slot_cpus)]


_PROBE_SOURCE = f'''#include "ordered_dispatch.hpp"
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {{
    if (argc < 7) return 2;
    const auto count = std::stoull(argv[1]);
    const auto child_limit = std::stoull(argv[2]);
    const auto owner_limit = std::stoull(argv[3]);
    const int owner = std::stoi(argv[4]);
    const rlimit governor{{owner_limit, owner_limit}};
    if (setrlimit(RLIMIT_AS, &governor) != 0) return 3;
    std::vector<int> workers;
    for (int i = 5; i < argc; ++i) workers.push_back(std::stoi(argv[i]));
    constexpr std::size_t request_bytes = {REQUEST_BYTES};
    constexpr std::size_t result_bytes = {RESULT_BYTES};
    const scar::ordered_v1::Config config{{owner, workers, workers.size(), request_bytes, result_bytes,
        child_limit, workers.size() * (request_bytes + result_bytes)}};
    std::vector<std::byte> retained(count * result_bytes);
    const auto factory = [](int) {{
        return [](std::uint64_t, std::span<const std::byte> input, std::span<std::byte> output) -> std::int32_t {{
            std::copy(input.begin(), input.end(), output.begin());
            return 0;
        }};
    }};
    const auto fill = [](std::uint64_t sequence, std::span<std::byte> input) {{
        input[0] = static_cast<std::byte>(sequence % 256);
    }};
    const auto retire = [&](const scar::ordered_v1::Retirement& event) {{
        if (event.kind != scar::ordered_v1::Kind::success ||
            event.result.front() != static_cast<std::byte>(event.sequence % 256)) {{
            throw std::runtime_error("transport service returned a wrong sequence");
        }}
        std::copy(event.result.begin(), event.result.end(), retained.data() + event.sequence * result_bytes);
    }};
    constexpr auto reps = 4;
    for (int i = 0; i < reps; ++i) {{
        const auto start = std::chrono::steady_clock::now();
        const auto summary = scar::ordered_v1::dispatch(config, count, factory, fill, retire);
        const auto stop = std::chrono::steady_clock::now();
        if (summary.succeeded != count || summary.retired != count) return 4;
        std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() << '\\n';
    }}
}}
'''


def _capture(command, *, env, stdout):
    result = subprocess.run(command, env=env, capture_output=True, timeout=120, check=True)
    _need(not result.stderr, 'calibration wrote stderr')
    stdout.write_bytes(result.stdout)
    return _ref(stdout, 'application/octet-stream')


def _budgets(native, count, width, worker_limit, owner_limit):
    arena = native['arena_bytes']
    retained = count * RESULT_BYTES
    payload = width * (REQUEST_BYTES + RESULT_BYTES)
    # Pending-frame budget is passed to the dispatcher. Its actual owner data
    # buffers use one request and one receive buffer. The explicit VAS cap also
    # includes dylibs, allocator, stack and any common JSON/report allocations.
    owner_minimum = retained + (payload if width else arena)
    worker_minimum = arena + REQUEST_BYTES + RESULT_BYTES
    _need(owner_minimum < owner_limit and (width == 0 or worker_minimum < worker_limit),
          'address-space cap cannot contain explicit array and result data')
    return dict(private_arena_bytes_per_numerical_worker=arena, request_bytes=REQUEST_BYTES,
                result_bytes=RESULT_BYTES, request_frame_bytes=REQUEST_BYTES + _ORDINAL_WIRE_BYTES,
                result_frame_bytes=RESULT_BYTES + _RESPONSE_WIRE_BYTES, payload_budget_bytes=payload,
                retained_result_payload_bytes=retained, explicit_owner_payload_bytes=owner_minimum,
                explicit_worker_payload_bytes=worker_minimum,
                worker_address_space_limit_bytes=worker_limit, coordinator_address_space_limit_bytes=owner_limit,
                required_headroom_bytes=owner_limit + width * worker_limit,
                bound_scope='AGGREGATE_PROCESS_VAS_CAPS_PLUS_EXPLICIT_PAYLOAD_SUBACCOUNT',
                physical_reservation=False, socket_kernel_buffer_capacity_proven=False)


def _widths(hardware, node, maximum, native, scope, budget_bytes, worker_limit, owner_limit):
    _need(all(type(value) is int and 0 < value <= _MAX_I64
              for value in (budget_bytes, worker_limit, owner_limit)) and worker_limit <= owner_limit,
          'positive address-space caps required, with coordinator cap inheritable by the workers')
    numa = hardware['node_free_snapshot'].get(str(node), {})
    cgroup = hardware['finite_memory_headroom_snapshot_bytes']
    quota = hardware['cpu_quota_equivalents']
    alternatives = []
    for width in (0, *range(2, maximum + 1)):
        memory = _budgets(native, scope['task_count'], width, worker_limit, owner_limit)
        reasons = []
        if not hardware['hierarchy_observation_complete']:
            reasons.append('UNRESOLVED_CGROUP_HIERARCHY')
        if quota is not None and quota[0] < (width + 1) * quota[1]:
            reasons.append('CPU_QUOTA_BELOW_CONFIGURED_LANES')
        if memory['required_headroom_bytes'] > budget_bytes:
            reasons.append('EXCEEDS_DECLARED_PROCESS_MEMORY_BUDGET')
        available = numa.get('bytes')
        if numa.get('allowed') is not True or type(available) is not int:
            reasons.append('UNRESOLVED_NUMA_HEADROOM')
        elif memory['required_headroom_bytes'] > available:
            reasons.append('EXCEEDS_OBSERVED_NUMA_FREE_MEMORY')
        if cgroup is not None and memory['required_headroom_bytes'] > cgroup:
            reasons.append('EXCEEDS_OBSERVED_CGROUP_HEADROOM')
        alternatives.append((width, memory, reasons))
    return alternatives


def measure(native_path, hardware_path, out, *, owner_cpu, task_count, requested_workers, round_count,
            budget_bytes, worker_address_space_limit_bytes, coordinator_address_space_limit_bytes, seeds):
    """Sample numerical worker service and same-ABI no-op transport independently."""
    native_ref, hardware_ref = _ref(native_path, NATIVE), _ref(hardware_path, HARDWARE)
    native, hardware = map(read_artifact, (native_ref, hardware_ref))
    recipe = read_artifact(native['recipe'])
    binary = native['files'][native['binary']]
    read_artifact(binary, json_document=False)
    scope = _scope(task_count, requested_workers, round_count)
    node, lanes = _cohort(hardware, owner_cpu)
    max_width = min(len(lanes) - 1, task_count, requested_workers)
    alternatives = _widths(hardware, node, max_width, native, scope, budget_bytes,
                           worker_address_space_limit_bytes, coordinator_address_space_limit_bytes)
    _need(len(seeds) >= 4 and len(set(seeds)) == len(seeds) and all(type(seed) is int and
          native['seed_range']['minimum'] <= seed <= native['seed_range']['maximum'] for seed in seeds),
          'four distinct in-range calibration-only seeds are required, including warmup')
    out = Path(out).absolute()
    out.mkdir(parents=True)
    widths = [width for width, _, reasons in alternatives if width and not reasons]
    cpus = [row['cpu_id'] for row in lanes[:max(widths, default=0) + 1]]
    launch_cpus = sorted(row['cpu_id'] for row in lanes)
    _need(set(launch_cpus) <= os.sched_getaffinity(0), 'calibration CPU mask is currently unavailable')
    environment = recipe['environment']
    worker_samples = []
    # Standalone native worker spans exclude entry startup and all report I/O.
    for cpu in cpus:
        for index, seed in enumerate(seeds):
            _, args = _parameters(native, recipe, seed, round_count)
            limit = coordinator_address_space_limit_bytes if cpu == owner_cpu else worker_address_space_limit_bytes
            command = _worker_command(cpu, binary, limit, args)
            raw = _capture(command, env=environment, stdout=out / f'worker-{cpu}-{index}.json')
            worker_samples.append(dict(cpu_id=cpu, seed=seed, warmup=index == 0, stdout=raw, command=command))
    source_path, header_path, probe_path = (out / name for name in ('transport.cpp', 'ordered_dispatch.hpp', 'transport'))
    source_path.write_text(_PROBE_SOURCE)
    header_path.write_bytes(_DISPATCH_HEADER.read_bytes())
    compiler = ['/usr/bin/g++', '-std=c++20', '-O3', str(source_path), '-o', str(probe_path)]
    subprocess.run(compiler, env=environment, capture_output=True, check=True, timeout=120)
    transport_samples = []
    for width in widths:
        command = _transport_command(probe_path, launch_cpus, cpus[:width + 1], task_count,
                                     worker_address_space_limit_bytes, coordinator_address_space_limit_bytes)
        raw = _capture(command, env=environment, stdout=out / f'transport-{width}.txt')
        transport_samples.append(dict(width=width, command=command, stdout=raw))
    return _write(out / 'result.json', dict(schema=EVIDENCE, schema_version=1, native=native_ref,
        hardware=hardware_ref, native_binary=binary, scope=scope, owner_cpu_id=owner_cpu,
        budget_bytes=budget_bytes, worker_address_space_limit_bytes=worker_address_space_limit_bytes,
        coordinator_address_space_limit_bytes=coordinator_address_space_limit_bytes,
        seeds=seeds, worker_samples=worker_samples, transport_samples=transport_samples,
        probe=_ref(probe_path, 'application/octet-stream'), probe_source=_ref(source_path, 'text/x-c++'),
        dispatch_header=_ref(header_path, 'text/x-c++'), compiler=compiler,
        experiment_boundary='CALIBRATION_ONLY_NO_ORIGINAL_HOLDOUT_OR_TARGET_PARALLEL_NUMERICAL_RUNS'))


def _service_intervals(evidence, native, recipe, cpus, launch_cpus, widths):
    binary = native['files'][native['binary']]
    scope = evidence['scope']
    seeds = evidence['seeds']
    _need(evidence['schema_version'] == 1 and evidence['native_binary'] == binary
          and evidence['experiment_boundary'] == 'CALIBRATION_ONLY_NO_ORIGINAL_HOLDOUT_OR_TARGET_PARALLEL_NUMERICAL_RUNS'
          and len(seeds) >= 4 and len(seeds) == len(set(seeds)) and all(type(seed) is int and
          native['seed_range']['minimum'] <= seed <= native['seed_range']['maximum'] for seed in seeds),
          'calibration dependency changed or lacks an independent service scope')
    read_artifact(binary, json_document=False)
    for key in ('probe', 'probe_source', 'dispatch_header'):
        read_artifact(evidence[key], json_document=False)
    _need(read_artifact(evidence['probe_source'], json_document=False) == _PROBE_SOURCE.encode()
          and read_artifact(evidence['dispatch_header'], json_document=False) == _DISPATCH_HEADER.read_bytes()
          and evidence['compiler'] == ['/usr/bin/g++', '-std=c++20', '-O3', evidence['probe_source']['uri'],
                                      '-o', evidence['probe']['uri']],
          'transport did not use the current same-ABI dispatcher')

    parameter_field = read_artifact(recipe['source_contract'])['report']['parameters']['field']
    services = {}
    seen = set()
    per_cpu = {cpu: [] for cpu in cpus}
    for row in evidence['worker_samples']:
        cpu, seed = row['cpu_id'], row['seed']
        _, args = _parameters(native, recipe, seed, scope['round_count'])
        cap_kind = 'coordinator' if cpu == evidence['owner_cpu_id'] else 'worker'
        limit = evidence[cap_kind + '_address_space_limit_bytes']
        _need((cpu, seed) not in seen and cpu in cpus and seed in seeds and row['warmup'] is (seed == seeds[0])
              and row['command'] == _worker_command(cpu, binary, limit, args),
              'duplicate or out-of-scope numerical service sample')
        seen.add((cpu, seed))
        raw = read_artifact(row['stdout'], json_document=False)
        elapsed = _worker_seconds(native, recipe, raw, seed, scope['round_count'], parameter_field)
        if not row['warmup']:
            per_cpu[cpu].append(elapsed)
    for cpu, values in per_cpu.items():
        if len(values) == len(seeds) - 1:
            services[cpu] = _summary(values)

    transports = {}
    for row in evidence['transport_samples']:
        width = row['width']
        mask = cpus[:width + 1]
        command = _transport_command(evidence['probe']['uri'], launch_cpus, mask, scope['task_count'],
            evidence['worker_address_space_limit_bytes'], evidence['coordinator_address_space_limit_bytes'])
        _need(width in widths and width not in transports and row['command'] == command,
              'transport sample crosses its exact cohort or request dimensions')
        samples = list(map(int, read_artifact(row['stdout'], json_document=False).splitlines()))
        _need(len(samples) == 4, 'transport lifecycle requires one warmup and three samples')
        transports[width] = _summary(samples[1:])
    return services, transports


def _ordered_time(count, durations):
    """Replay exact issue/refill dependencies with CPU-service times only."""
    lane_count = len(durations)
    ready = list(durations)
    owner = 0
    for sequence in range(count):
        lane = sequence % lane_count
        owner = max(owner, ready[lane])
        if sequence + lane_count < count:
            ready[lane] = owner + durations[lane]
    return owner


def _cost(count, cpus, services, transport):
    if transport is None or not all(cpu in services for cpu in cpus):
        return None
    numeric = {key: _ordered_time(count, [services[cpu][key] for cpu in cpus]) for key in _SIDES}
    return dict(worker_service_intervals_ns={str(cpu): services[cpu] for cpu in cpus},
                ordered_numeric_completion_ns=numeric, additive_transport_interval_ns=transport,
                modeled_elapsed_ns={key: numeric[key] + transport[key] for key in _SIDES})


def _decision(alternatives):
    serial = alternatives[0]
    _need(serial['mode'] == 'IN_PROCESS_SERIAL', 'serial option must remain in the plan')
    viable = [row for row in alternatives if not row['capacity_reasons']]
    measured = [row for row in viable if row['cost'] is not None]
    shadowed = {'UNRESOLVED_CGROUP_HIERARCHY', 'UNRESOLVED_NUMA_HEADROOM'}
    unresolved = any(row['cost'] is None and not (set(row['capacity_reasons']) - shadowed)
                     for row in alternatives)
    if serial not in viable:
        status, selected = 'NO_ADMISSIBLE_NATIVE_SERIAL_BASELINE', None
    elif unresolved:
        status, selected = 'KEEP_SERIAL_UNRESOLVED_ALTERNATIVE', serial
    else:
        def elapsed(row, field):
            return row['cost']['modeled_elapsed_ns'][field]

        point_choice = min(measured, key=lambda row: (elapsed(row, 'estimate_ns'), row['worker_count']))
        separated = all(elapsed(point_choice, 'upper_ns') < elapsed(row, 'lower_ns')
                        for row in measured if row is not point_choice)
        selected = point_choice if separated else serial
        if selected is not serial:
            status = 'NOMINATE_ORDERED_PROCESSES_FOR_VALIDATION'
        elif len(measured) == 1:
            status = 'ONLY_SERIAL_IN_FINITE_DOMAIN'
        else:
            status = 'KEEP_SERIAL_NO_SEPARATED_PARALLEL_GAIN'
    return dict(status=status, selected=selected, point_cost_optimum_is_finite_model_only=True,
                deployment_authorized=False)


def _derive(native_ref, hardware_ref, evidence_ref, *, owner_cpu, task_count, requested_workers, round_count,
            budget_bytes, worker_address_space_limit_bytes, coordinator_address_space_limit_bytes):
    native, hardware, evidence = map(read_artifact, (native_ref, hardware_ref, evidence_ref))
    recipe = read_artifact(native['recipe'])
    scope = _scope(task_count, requested_workers, round_count)
    node, lanes = _cohort(hardware, owner_cpu)
    _need(evidence['native'] == native_ref and evidence['hardware'] == hardware_ref and evidence['scope'] == scope
          and evidence['owner_cpu_id'] == owner_cpu and evidence['budget_bytes'] == budget_bytes
          and evidence['worker_address_space_limit_bytes'] == worker_address_space_limit_bytes
          and evidence['coordinator_address_space_limit_bytes'] == coordinator_address_space_limit_bytes
          and native['python_runtime_required'] is False, 'service evidence or backend differs from scheduling scope')
    max_width = min(len(lanes) - 1, task_count, requested_workers)
    domain = _widths(hardware, node, max_width, native, scope, budget_bytes,
                     worker_address_space_limit_bytes, coordinator_address_space_limit_bytes)
    widths = [width for width, _, reasons in domain if width and not reasons]
    cpus = [row['cpu_id'] for row in lanes[:max(widths, default=0) + 1]]
    launch_cpus = sorted(row['cpu_id'] for row in lanes)
    services, transports = _service_intervals(evidence, native, recipe, cpus, launch_cpus, widths)
    alternatives = []
    for width, memory, reasons in domain:
        if width == 0:
            worker_ids = []
            cost = _cost(task_count, [owner_cpu], services, dict.fromkeys(_SIDES, 0))
        else:
            worker_ids = [row['cpu_id'] for row in lanes[1:width + 1]]
            cost = _cost(task_count, worker_ids, services, transports.get(width))
        alternatives.append(dict(mode='ORDERED_LOCAL_PROCESSES' if width else 'IN_PROCESS_SERIAL',
            owner_cpu_id=owner_cpu, worker_count=width or 1, worker_cpu_ids=worker_ids, window=width,
            launch_cpu_ids=launch_cpus, memory=memory, capacity_reasons=reasons, cost=cost,
            scheduled_pod_result_lifetime='RETAIN_IN_OWNER_UNTIL_ALL_TASKS_HAVE_SUCCEEDED'))
    return dict(schema=SCHEMA, schema_version=1, native=native_ref, hardware=hardware_ref, evidence=evidence_ref,
        scope=scope, cpu_numa_node=node, allowed_cpu_lanes=lanes,
        base_seed_range=dict(minimum=native['seed_range']['minimum'],
            maximum=native['seed_range']['maximum'] - task_count + 1),
        budget_bytes=budget_bytes, alternatives=alternatives, decision=_decision(alternatives),
        dispatch=dict(task_order='ASCENDING_ZERO_BASED', assignment='ORDINAL_MOD_WORKER_COUNT',
            refill='AFTER_OLDEST_ORDERED_RETIREMENT', terminal_publication='AFTER_ALL_SUCCEEDED_IN_ORDINAL_ORDER',
            worker_provider_threads=1, worker_task_timeout_ms=30000),
        domain=dict(execution='LOCAL_SINGLE_NUMA_OWNER_AND_ORDERED_PROCESSES',
            enumeration='SERIAL_AND_NESTED_PHYSICAL_CORE_PREFIXES_WITHIN_ORIGINAL_CAP',
            original_requested_workers_is_concurrency_cap=True, other_placement_search=False,
            max_parallel_width=max_width),
        cost_boundary=dict(kernel='NATIVE_WORKER_WALL_AFTER_ENTRY_INGRESS_BEFORE_REPORT',
            transport='WHOLE_EMPTY_DISPATCH_WITH_SAME_COUNT_WIRE_ABI_RETAINED_POD_AND_ADDRESS_SPACE_CAPS',
            summary='EMPIRICAL_SAMPLE_ENVELOPE_INDEPENDENT_LANE_RUNTIME_AND_ADDITIVE_TRANSPORT',
            simultaneous_numerical_contention_observed=False, launch_and_final_publication_shared_unpriced=True,
            process_space_bound_requires_runtime_enforcement=True, exact_wall_clock_or_global_optimum_proven=False,
            fresh_real_output_and_clean_timing_required=True), deployment_authorized=False)


def project(family_path, native_path, hardware_path, evidence_path, *, owner_cpu, task_count, requested_workers,
            round_count, budget_bytes, worker_address_space_limit_bytes, coordinator_address_space_limit_bytes):
    """Bind the frozen finite decision to the original family and verified numerical entry."""
    from scar.analysis import seeded_array_task_family

    family_ref = _ref(family_path, seeded_array_task_family.SCHEMA)
    family = seeded_array_task_family.check(family_ref['uri'])
    native_ref = _ref(native_path, NATIVE)
    native = read_artifact(native_ref)
    recipe = read_artifact(native['recipe'])
    options = family['input_options']
    worker_choices = recipe['options'][options['worker_limit']]['choices']
    _need(family == read_artifact(family_ref) and family['loop'] == native['loop']
          and family['source_contract'] == recipe['source_contract']
          and family['state']['kind'] == 'PER_TASK_PRIVATE' and family['state']['loop'] == native['loop']
          and family['state']['no_application_level_shared_writes'] is True
          and family['publication']['after_all_workers'] is True and family['publication']['pid_order'] == 'SORTED_UNIQUE'
          and family['publication']['result_order'] == 'ASCENDING_ORDINAL'
          and family['ordinal_range'] == dict(start=0, step=1)
          and family['effective_seed']['operation'] == 'ADD'
          and family['effective_seed']['domain'] == native['seed_range']
          and options['seed'] == native['seed_option'] and options['round_count'] == native['round_option']
          and requested_workers in worker_choices, 'native plan requires proved independence and original worker domain')
    decision = _derive(native_ref, _ref(hardware_path, HARDWARE), _ref(evidence_path, EVIDENCE),
        owner_cpu=owner_cpu, task_count=task_count, requested_workers=requested_workers, round_count=round_count,
        budget_bytes=budget_bytes, worker_address_space_limit_bytes=worker_address_space_limit_bytes,
        coordinator_address_space_limit_bytes=coordinator_address_space_limit_bytes)
    _need(decision['base_seed_range']['maximum'] >= decision['base_seed_range']['minimum'],
          'family task count exceeds the original native seed domain')
    return dict(decision, family=family_ref, input_options=options)


def check(path):
    """Replay source binding, hardware scope, service evidence and finite ranking."""
    result = read_artifact(_ref(path, SCHEMA))
    evidence = read_artifact(result['evidence'])
    kwargs = {key: evidence[key] for key in ('budget_bytes', 'worker_address_space_limit_bytes',
                                             'coordinator_address_space_limit_bytes')}
    expected = project(*(result[key]['uri'] for key in ('family', 'native', 'hardware', 'evidence')),
                       owner_cpu=evidence['owner_cpu_id'], **result['scope'], **kwargs)
    _need(result == expected, 'saved decision differs from its retained evidence')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    capture = commands.add_parser('inventory')
    capture.add_argument('--cpu-ids', required=True)
    capture.add_argument('--out', required=True)
    replay = commands.add_parser('check')
    replay.add_argument('receipt')
    for name in ('measure', 'project'):
        action = commands.add_parser(name)
        for flag in ('native', 'hardware', 'out'):
            action.add_argument('--' + flag, required=True)
        for flag in ('owner-cpu', 'task-count', 'requested-workers', 'round-count', 'budget-bytes',
                     'worker-address-space-limit-bytes', 'coordinator-address-space-limit-bytes'):
            action.add_argument('--' + flag, type=int, required=True)
        if name == 'measure':
            action.add_argument('--seeds', required=True)
        else:
            action.add_argument('--family', required=True)
            action.add_argument('--evidence', required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    if command == 'check':
        result = check(args['receipt'])['decision']
    else:
        out = args.pop('out')
        if command == 'inventory':
            result = collect(out, list(map(int, args['cpu_ids'].split(','))))
        else:
            native, hardware = (args.pop(key) for key in ('native', 'hardware'))
            if command == 'measure':
                args['seeds'] = list(map(int, args['seeds'].split(',')))
                result = measure(native, hardware, out, **args)
            else:
                family, evidence = (args.pop(key) for key in ('family', 'evidence'))
                report = project(family, native, hardware, evidence, **args)
                _write(out, report)
                result = report['decision']
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
