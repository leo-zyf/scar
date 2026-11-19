"""Qualify and price adjacent ATen windows on a three-core CPU team."""

from collections import Counter, defaultdict
from itertools import product
from math import prod
from statistics import median_low

from scar.analysis.target_graph import TargetHardwareGraph
from scar.ir.record_codec import encode
from scar.optimization.aten_cpu_placement_domain import _cpu_memberships, _lane, _signature, _topology
from scar.optimization.aten_native_schedule import (
    SCOPE,
    AtenCpuWindowFrontier,
    _declared_operators,
    _live_inputs,
    recorded_cpu_domain,
)
from scar.optimization.aten_tensor_lifetime import _bounded_sum, tensor_footprint, tensor_releases


TEAM_SCHEMA = 'scar.aten-native-cpu-team'
TEAM_DOMAIN_SCHEMA = 'scar.aten-cpu-team-placement-domain'
_MAX_REPRESENTATIVES = 64
_MODE_NAMES = ('SERIAL', 'TWO_CPU', 'THREE_CPU')


def _groups(recipe, contract):
    tensor_releases(recipe)
    declared = _declared_operators(contract)
    operations = recipe['operations']
    reads = [_live_inputs(op) for op in operations]
    groups = []
    for start in range(len(operations)):
        for width in (2, 3):
            region = operations[start:start + width]
            if len(region) != width or any(op['schema'] not in declared for op in region):
                break
            produced = {op['id'] for op in region}
            region_reads = reads[start:start + width]
            if any(produced & inputs for inputs in region_reads):
                break
            multiplicities = Counter(value for inputs in region_reads for value in inputs)
            groups.append(dict(start=start, width=width, operations=[op['id'] for op in region],
                               statically_shared_reads=sorted(value for value, count in multiplicities.items()
                                                              if count > 1)))
    return groups


def _widths(groups, operation_count):
    widths = [1] * operation_count
    for group in groups:
        widths[group['start']] = max(widths[group['start']], group['width'])
    return widths


def team_window_widths(recipe, contract):
    """Return the largest legal adjacent width at each normalized operation."""
    return _widths(_groups(recipe, contract), len(recipe['operations']))


def _valid_lane(lane):
    thread, core, executor, cache = lane
    return (thread is not None and thread.online is True and core is not None and executor is not None
            and executor.numa_node is not None and cache is not None and thread.isa_flags is not None
            and thread.isa_evidence is not None)


def _team_context(lane):
    return lane[2].numa_node, tuple(sorted(lane[0].isa_flags))


def _cpu_number(value):
    return type(value) is int and 0 <= value < 1024


def _representatives(lanes):
    # Teams stay inside an observed NUMA and ISA domain. Within it, retain one
    # ordered sample per combination of exact topology and LLC memberships.
    by_context = defaultdict(lambda: defaultdict(list))
    for cpu, lane in lanes.items():
        by_context[_team_context(lane)][_signature(lane), lane[3].id].append(cpu)
    representatives = []
    for classes in by_context.values():
        if sum(map(len, classes.values())) < 3:
            continue
        for ordered in product(classes, repeat=3):
            remaining = Counter(ordered)
            if any(len(classes[group]) < count for group, count in remaining.items()):
                continue
            used = Counter()
            cpus, combinations = [], 1
            for group in ordered:
                available = classes[group]
                combinations *= len(available) - used[group]
                cpus.append(available[used[group]])
                used[group] += 1
            if len(representatives) >= _MAX_REPRESENTATIVES:
                return [], 'CPU_TEAM_TOPOLOGY_CLASS_BUDGET_EXCEEDED'
            representatives.append(dict(cpus=cpus, represented_ordered_teams=combinations,
                                        topology=[_topology(lanes[cpu]) for cpu in cpus]))
    if not representatives:
        return [], 'NO_THREE_DISTINCT_COMPATIBLE_CORES'
    return representatives, None


def aten_cpu_team_domain(target, *, allowed_cpu_ids):
    """Nominate recorded same-NUMA, ISA-compatible team placements for calibration."""
    allowed = tuple(allowed_cpu_ids)
    if any(type(cpu) is not int or cpu < 0 for cpu in allowed):
        raise ValueError('launch affinity requires nonnegative Linux CPU IDs')
    launch = set(allowed)
    graph = TargetHardwareGraph(target)
    constraint, recorded = recorded_cpu_domain(graph)
    membership = _cpu_memberships(graph.resources())
    lanes, used_cores, excluded = {}, set(), Counter()
    for cpu in recorded:
        lane = _lane(cpu, *membership)
        if cpu not in launch:
            excluded['OUTSIDE_LAUNCH_AFFINITY'] += 1
        elif not _cpu_number(cpu) or not _valid_lane(lane):
            excluded['UNAVAILABLE_TEAM_TOPOLOGY'] += 1
        elif lane[1].id in used_cores:
            excluded['SMT_SIBLING_NOT_NOMINATED'] += 1
        else:
            lanes[cpu] = lane
            used_cores.add(lane[1].id)
    representatives, reason = _representatives(lanes)
    return dict(schema=TEAM_DOMAIN_SCHEMA, schema_version=1, target=encode(graph._snapshot),
                constraint_fact=encode(constraint.evidence), recorded_cpu_ids=list(recorded),
                launch_cpu_ids=sorted(launch), eligible_cpus=list(lanes),
                policy='ONE_ORDERED_REPRESENTATIVE_PER_SAME_NUMA_ISA_TOPOLOGY_CLASS',
                class_limit=_MAX_REPRESENTATIVES, representatives=representatives, reason=reason,
                excluded_cpu_counts=dict(excluded))


def propose_cpu_team(recipe, *, contract, target, cpu_ids):
    """Replay a concrete team against recorded hardware and graph independence."""
    groups = _groups(recipe, contract)
    if not any(group['width'] == 3 for group in groups):
        raise AtenCpuWindowFrontier('NO_THREE_READY_ATEN_OPERATIONS',
                                   'assessed recipe has no independent adjacent triple')
    cpus = list(cpu_ids)
    if len(cpus) != 3 or not all(map(_cpu_number, cpus)) or len(set(cpus)) != 3:
        raise ValueError('three distinct ordered Linux CPU IDs required')
    graph = TargetHardwareGraph(target)
    constraint, domain = recorded_cpu_domain(graph)
    if not set(cpus).issubset(domain):
        raise AtenCpuWindowFrontier('CPU_TEAM_OUTSIDE_TARGET_CPU_DOMAIN',
                                   'team lanes must belong to the recorded target availability domain')
    membership = _cpu_memberships(graph.resources())
    lanes = [_lane(cpu, *membership) for cpu in cpus]
    if not all(map(_valid_lane, lanes)) or len({lane[1].id for lane in lanes}) != 3:
        raise AtenCpuWindowFrontier('CPU_TEAM_DISTINCT_CORES_UNAVAILABLE',
                                   'team lanes require distinct online cores with known topology')
    if len(set(map(_team_context, lanes))) != 1:
        raise AtenCpuWindowFrontier('CPU_TEAM_REQUIRES_SAME_NUMA_ISA_CONTEXT',
                                   'team lanes require one recorded NUMA node and compatible ISA flags')
    return dict(schema=TEAM_SCHEMA, schema_version=1, scope=SCOPE, target=encode(graph._snapshot),
                cpus=cpus, availability_mask=sorted(cpus), lanes=[_topology(lane) for lane in lanes],
                constraint_fact=encode(constraint.evidence),
                target_cpu_domain=dict(affinity_cpu_ids=list(constraint.affinity_cpu_ids),
                                       allowed_cpu_ids=list(constraint.allowed_cpu_ids), eligible_cpu_ids=list(domain)),
                legal_window_widths=_widths(groups, len(recipe['operations'])), groups=groups,
                effect_basis='EXPLICIT_STANDALONE_CALLER_CONTRACT',
                source_replacement_authorized=False, performance_measured=False)


def _bounded_duration(value):
    return type(value) is int and 0 < value < 2**63


def _validate_samples(samples, count, operation_count, positions):
    if (type(samples) is not list or len(samples) != count
            or any(type(row) is not list or len(row) != operation_count for row in samples)):
        raise ValueError('native team calibration requires a rectangular operation sample matrix per lane')
    for row in samples:
        for index, duration in enumerate(row):
            if index in positions and not _bounded_duration(duration):
                raise ValueError('a legal native team lane is missing its measured service time')
            if index not in positions and duration is not None:
                raise ValueError('native team calibration measured a disallowed worker position')


def _team_groups(recipe, schedule):
    if (type(schedule) is not dict or schedule.get('schema') != TEAM_SCHEMA
            or type(schedule.get('schema_version')) is not int or schedule['schema_version'] != 1):
        raise ValueError('native CPU team schedule required')
    operations = recipe['operations']
    windows = {}
    for group in schedule['groups']:
        start, width = group['start'], group['width']
        if (type(start) is not int or type(width) is not int or width not in (2, 3)
                or not 0 <= start <= len(operations) - width or (start, width) in windows):
            raise ValueError('native team requires distinct adjacent windows at widths two or three')
        region = operations[start:start + width]
        names = [op['id'] for op in region]
        if group['operations'] != names or any(set(names) & _live_inputs(op) for op in region):
            raise ValueError('native team window crosses an original Tensor dependency')
        windows[start, width] = group
    if (not any(width == 3 for _, width in windows)
            or any((start, 2) not in windows for start, width in windows if width == 3)
            or schedule['legal_window_widths'] != _widths(windows.values(), len(operations))):
        raise ValueError('native team windows disagree with the admitted widths')
    return windows


def _storage(calibration, recipe, windows, budget):
    input_bytes, output_bytes = calibration['input_storage_bytes'], calibration['result_storage_bytes']
    if input_bytes != [4 * prod(spec['shape']) for spec in recipe['inputs']]:
        raise ValueError('standalone input storage bytes differ from the float32 file ABI')
    footprint = tensor_footprint(recipe, input_bytes=input_bytes, output_bytes=output_bytes,
                                 candidates=[], budget_bytes=budget)
    group_peaks, possible = {}, set()
    for start, width in windows:
        entry_live = footprint['operation_peak_bytes'][start] - output_bytes[start]
        peak = _bounded_sum((entry_live, *output_bytes[start:start + width]))
        group_peaks[f'{start}:{width}'] = peak
        if budget is None or peak <= budget:
            possible.add((start, width))
    footprint['group_peak_bytes'] = group_peaks
    footprint['candidate_groups_within_budget'] = [list(group) for group in sorted(possible)]
    return footprint, possible


def _measurements(calibration, warmup, operation_count, windows):
    if type(warmup) is not int or not 3 <= warmup <= 100:
        raise ValueError('native team calibration requires three to one hundred warmups')
    all_lanes = calibration['worker_operation_samples_ns']
    handoffs = calibration['empty_dispatch_samples_ns']
    if (type(all_lanes) is not list or len(all_lanes) != 2 or type(handoffs) is not dict
            or handoffs.keys() != {'2', '3'} or any(type(samples) is not list or len(samples) != warmup
            or not all(map(_bounded_duration, samples)) for samples in handoffs.values())):
        raise ValueError('native team needs exactly two worker matrices and separate two- and three-lane handoff costs')
    measured_positions = [set(range(operation_count)), {start + 1 for start, width in windows if width >= 2},
                          {start + 2 for start, width in windows if width == 3}]
    lanes = [calibration['operation_samples_ns'], *all_lanes]
    for samples, positions in zip(lanes, measured_positions, strict=True):
        _validate_samples(samples, warmup, operation_count, positions)
    service = [[median_low(observations) if index in positions else None
                for index, observations in enumerate(zip(*samples, strict=True))]
               for samples, positions in zip(lanes, measured_positions, strict=True)]
    dispatch = {width: median_low(samples) for width, samples in handoffs.items()}
    return service, dispatch, lanes, handoffs


def _segment_time(service, handoffs, start, width):
    if width == 1:
        return service[0][start]
    return max(service[lane][start + lane] for lane in range(width)) + handoffs[str(width)]


def _best_tilings(service, dispatch, possible):
    # Suffixes are indexed by the largest width actually used, preserving the
    # best comparable plan at every executable width without enumerating tilings.
    count = len(service[0])
    suffixes = [{} for _ in range(count + 1)]
    suffixes[count][1] = (0, 0, 1, 1)
    for start in range(count - 1, -1, -1):
        for width in (1, 2, 3):
            if width != 1 and (start, width) not in possible:
                continue
            duration = _segment_time(service, dispatch, start, width)
            for tail_width, tail in suffixes[start + width].items():
                largest = max(width, tail_width)
                estimate = (duration + tail[0], (width > 1) + tail[1], -width, tail_width)
                previous = suffixes[start].get(largest)
                if previous is None or estimate < previous:
                    suffixes[start][largest] = estimate

    results = {}
    for largest in sorted(suffixes[0]):
        start, remaining_width, choices = 0, largest, []
        while start < count:
            _, _, reversed_width, remaining_width = suffixes[start][remaining_width]
            width = -reversed_width
            if width > 1:
                choices.append([start, width])
            start += width
        results[_MODE_NAMES[largest - 1]] = choices
    return results


def _timeline(operations, cpus, service, handoffs, choices, storage):
    widths = dict(choices)
    start, offset, timeline = 0, 0, []
    while start < len(operations):
        width = widths.get(start, 1)
        duration = _segment_time(service, handoffs, start, width)
        peak = (storage['operation_peak_bytes'][start] if width == 1
                else storage['group_peak_bytes'][f'{start}:{width}'])
        timeline.append(dict(start_ns=offset, end_ns=offset + duration, start=start, width=width,
                             operations=[op['id'] for op in operations[start:start + width]],
                             cpu_ids=cpus[:width], peak_tensor_storage_bytes=peak,
                             dispatch_join_ns=0 if width == 1 else handoffs[str(width)]))
        start += width
        offset += duration
    return timeline


def model_cpu_team(calibration, recipe, schedule, warmup, *, tensor_budget_bytes=None):
    """Select a resource-legal ordered tiling from separately measured primitives."""
    windows = _team_groups(recipe, schedule)
    operations = recipe['operations']
    service, handoffs, lane_samples, handoff_samples = _measurements(calibration, warmup, len(operations), windows)
    storage, possible = _storage(calibration, recipe, windows, tensor_budget_bytes)
    by_mode = _best_tilings(service, handoffs, possible)
    timelines = {mode: _timeline(operations, schedule['cpus'], service, handoffs, choices, storage)
                 for mode, choices in by_mode.items()}
    predictions = {}
    for mode, timeline in timelines.items():
        estimate = timeline[-1]['end_ns']
        totals = []
        for row in range(warmup):
            measured_service = [lane[row] for lane in lane_samples]
            measured_handoffs = {width: values[row] for width, values in handoff_samples.items()}
            totals.append(sum(_segment_time(measured_service, measured_handoffs, segment['start'], segment['width'])
                              for segment in timeline))
        predictions[mode] = dict(estimate_ns=estimate, sample_min_ns=min(estimate, *totals),
                                 sample_max_ns=max(estimate, *totals))

    selected = min(by_mode, key=lambda mode: (predictions[mode]['estimate_ns'], _MODE_NAMES.index(mode)))
    choices = {tuple(group) for group in by_mode[selected]}
    candidates = [dict(group, peak_tensor_storage_bytes=storage['group_peak_bytes'][f'{start}:{width}'],
                       within_tensor_budget=(start, width) in possible, selected=(start, width) in choices)
                  for (start, width), group in windows.items()]
    return dict(basis='MINIMUM_ADJACENT_ATEN_WINDOW_TILING_PER_EXECUTION_WIDTH', selected=selected,
                selected_windows=by_mode[selected], mode_windows=by_mode, timeline=timelines[selected],
                timelines=timelines, predictions=predictions, candidates=candidates, tensor_storage=storage,
                operations=[dict(id=op['id'], service_ns=duration) for op, duration in zip(operations, service[0], strict=True)],
                lane_service_ns=service, dispatch_join_ns=handoffs, bounds='CALIBRATION_SPAN_NOT_PHYSICAL_BOUNDS',
                shared_memory_contention_modelled=False, numa_page_placement_modelled=False)
