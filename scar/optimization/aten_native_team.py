"""Qualify a three-core ATen team and price its adjacent windows."""

from collections import Counter, defaultdict
from itertools import product
from math import prod
from statistics import median_low

from scar.analysis.target_graph import TargetHardwareGraph
from scar.ir.program.target_hardware import MemoryDomain, MemoryKind
from scar.ir.record_codec import encode
from scar.optimization.aten_cpu_placement_domain import _cpu_memberships, _lane, _signature, _topology
from scar.optimization.aten_native_schedule import (
    SCOPE,
    AtenCpuWindowFrontier,
    _declared_operators,
    _live_inputs,
    recorded_cpu_domain,
)
from scar.optimization.aten_post_join_evidence import read_aten_post_join_evidence
from scar.optimization.aten_live_values import _bounded_sum, tensor_footprint, tensor_releases
from scar.optimization.aten_team_group_evidence import read_aten_team_group_evidence
from scar.optimization.aten_team_numa_locality import model_aten_team_numa_locality


TEAM_SCHEMA = 'scar.aten-native-cpu-team'
TEAM_DOMAIN_SCHEMA = 'scar.aten-cpu-team-placement-domain'
TEAM_PLACEMENT_POLICY = 'ISA_COMPATIBLE_ORDERED_TOPOLOGY'
READY_SERVICE_SCOPE = 'ALL_QUALIFIED_RECIPE_OPERATIONS'
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


def data_dependencies(recipe):
    """Recover direct Tensor producers, including a single edge for repeated operands."""
    tensor_releases(recipe)
    operations = recipe['operations']
    producers = {operation['id']: index for index, operation in enumerate(operations)}
    return [sorted({producers[arg['value']] for arg in operation['arguments'] if arg.get('value') in producers})
            for operation in operations]


def _ready_graph(recipe):
    dependencies = data_dependencies(recipe)
    releases = [[] for _ in dependencies]
    for operation, parents in enumerate(dependencies):
        releases[max(parents, default=-1) + 1].append(operation)
    ready = set()
    for prefix, released in enumerate(releases):
        ready.update(released)
        if len(ready) > 1:
            return dict(data_dependencies=dependencies, first_ready_prefix=prefix, ready_operations=sorted(ready))
        ready.remove(prefix)
    return None


def ready_team_available(recipe, contract):
    """Check whether exact whole-graph effects and DATA permit a ready team."""
    declared = _declared_operators(contract)
    return all(op['schema'] in declared for op in recipe['operations']) and _ready_graph(recipe) is not None


def _valid_lane(lane):
    thread, core, executor, cache = lane
    return (thread is not None and thread.online is True and core is not None and executor is not None
            and executor.numa_node is not None and cache is not None and thread.isa_flags is not None
            and thread.isa_evidence is not None)


def _team_context(lane, allow_cross_numa):
    numa = None if allow_cross_numa else lane[2].numa_node
    return numa, tuple(sorted(lane[0].isa_flags))


def _crosses_numa(lanes):
    return len({lane[2].numa_node for lane in lanes}) > 1


def _cpu_number(value):
    return type(value) is int and 0 <= value < 1024


def _representatives(lanes, allow_cross_numa):
    # Keep NUMA and LLC in the ordered signature even when lanes can span nodes.
    by_context = defaultdict(lambda: defaultdict(list))
    for cpu, lane in lanes.items():
        by_context[_team_context(lane, allow_cross_numa)][_signature(lane), lane[3].id].append(cpu)
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
            team_lanes = [lanes[cpu] for cpu in cpus]
            row = dict(cpus=cpus, represented_ordered_teams=combinations,
                       topology=[_topology(lane) for lane in team_lanes])
            if allow_cross_numa:
                row['cross_numa'] = _crosses_numa(team_lanes)
            representatives.append(row)
    if not representatives:
        return [], 'NO_THREE_DISTINCT_COMPATIBLE_CORES'
    return representatives, None


def aten_cpu_team_domain(target, *, allowed_cpu_ids, allow_cross_numa=False):
    """Nominate ordered, ISA-compatible team placements for calibration."""
    if type(allow_cross_numa) is not bool:
        raise ValueError('team NUMA policy must be an explicit boolean')
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
    representatives, reason = _representatives(lanes, allow_cross_numa)
    policy = ('ONE_ORDERED_REPRESENTATIVE_PER_ISA_TOPOLOGY_CLASS' if allow_cross_numa
              else 'ONE_ORDERED_REPRESENTATIVE_PER_SAME_NUMA_ISA_TOPOLOGY_CLASS')
    return dict(schema=TEAM_DOMAIN_SCHEMA, schema_version=1, target=encode(graph._snapshot),
                constraint_fact=encode(constraint.evidence), recorded_cpu_ids=list(recorded),
                launch_cpu_ids=sorted(launch), eligible_cpus=list(lanes),
                policy=policy,
                class_limit=_MAX_REPRESENTATIVES, representatives=representatives, reason=reason,
                excluded_cpu_counts=dict(excluded))


def propose_cpu_team(recipe, *, contract, target, cpu_ids, allow_cross_numa=False, ready_dag=False):
    """Replay a concrete team against recorded hardware and graph independence."""
    if type(allow_cross_numa) is not bool or type(ready_dag) is not bool:
        raise ValueError('team NUMA and ready-DAG policies must be explicit booleans')
    groups = _groups(recipe, contract)
    ready_graph = None
    if ready_dag:
        declared = _declared_operators(contract)
        if any(op['schema'] not in declared for op in recipe['operations']):
            raise AtenCpuWindowFrontier('READY_DAG_EFFECT_CONTRACT_INCOMPLETE',
                                       'ready DAG requires an exact effect declaration for every operation')
        ready_graph = _ready_graph(recipe)
        if ready_graph is None:
            raise AtenCpuWindowFrontier('READY_DAG_HAS_NO_INDEPENDENT_OPERATIONS',
                                       'no source prefix has two DATA-ready operations')
    elif not any(group['width'] == 3 for group in groups):
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
    inventory = graph.resources()
    membership = _cpu_memberships(inventory)
    lanes = [_lane(cpu, *membership) for cpu in cpus]
    if not all(map(_valid_lane, lanes)) or len({lane[1].id for lane in lanes}) != 3:
        raise AtenCpuWindowFrontier('CPU_TEAM_DISTINCT_CORES_UNAVAILABLE',
                                   'team lanes require distinct online cores with known topology')
    if len({_team_context(lane, allow_cross_numa) for lane in lanes}) != 1:
        if allow_cross_numa:
            raise AtenCpuWindowFrontier('CPU_TEAM_REQUIRES_COMPATIBLE_ISA_CONTEXT',
                                       'team lanes require compatible ISA flags')
        raise AtenCpuWindowFrontier('CPU_TEAM_REQUIRES_SAME_NUMA_ISA_CONTEXT',
                                   'team lanes require one recorded NUMA node and compatible ISA flags')
    schedule = dict(schema=TEAM_SCHEMA, schema_version=2 if ready_dag else 1, scope=SCOPE, target=encode(graph._snapshot),
                    cpus=cpus, availability_mask=sorted(cpus), lanes=[_topology(lane) for lane in lanes],
                    constraint_fact=encode(constraint.evidence),
                    target_cpu_domain=dict(affinity_cpu_ids=list(constraint.affinity_cpu_ids),
                                           allowed_cpu_ids=list(constraint.allowed_cpu_ids), eligible_cpu_ids=list(domain)),
                    legal_window_widths=_widths(groups, len(recipe['operations'])), groups=groups,
                    effect_basis='EXPLICIT_STANDALONE_CALLER_CONTRACT',
                    source_replacement_authorized=False, performance_measured=False)
    if ready_dag:
        schedule['ready_graph'] = ready_graph
    if allow_cross_numa:
        schedule.update(placement_policy=TEAM_PLACEMENT_POLICY, cross_numa=_crosses_numa(lanes))
        if ready_dag and schedule['cross_numa']:
            schedule['host_numa_node_ids'] = sorted({row.numa_node for row in inventory
                if type(row) is MemoryDomain and row.kind is MemoryKind.HOST and row.numa_node is not None})
    return schedule


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
            or type(schedule.get('schema_version')) is not int or schedule['schema_version'] not in (1, 2)):
        raise ValueError('native CPU team schedule required')
    full_graph = schedule['schema_version'] == 2
    if full_graph:
        graph = _ready_graph(recipe)
        if graph is None or schedule.get('ready_graph') != graph:
            raise ValueError('ready team admission differs from its original Tensor dependency graph')
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
    if ((not full_graph and not any(width == 3 for _, width in windows))
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


def _measurements(calibration, warmup, operation_count, windows, *, full_graph=False):
    if type(warmup) is not int or not 3 <= warmup <= 100:
        raise ValueError('native team calibration requires three to one hundred warmups')
    all_lanes = calibration['worker_operation_samples_ns']
    handoffs = calibration['empty_dispatch_samples_ns']
    if (type(all_lanes) is not list or len(all_lanes) != 2 or type(handoffs) is not dict
            or handoffs.keys() != {'2', '3'} or any(type(samples) is not list or len(samples) != warmup
            or not all(map(_bounded_duration, samples)) for samples in handoffs.values())):
        raise ValueError('native team needs exactly two worker matrices and separate two- and three-lane handoff costs')
    all_positions = set(range(operation_count))
    scope = READY_SERVICE_SCOPE if full_graph else None
    if calibration.get('service_scope') != scope:
        raise ValueError('ATen team service evidence scope differs from its schedule')
    measured_positions = ([all_positions] * 3 if full_graph else
        [all_positions, {start + 1 for start, width in windows if width >= 2},
                        {start + 2 for start, width in windows if width == 3}])
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


def _timeline(operations, cpus, service, handoffs, choices, storage, *, group_medians=None, post_join_indices=()):
    widths = dict(choices)
    start, offset, timeline = 0, 0, []
    while start < len(operations):
        width = widths.get(start, 1)
        duration = _segment_time(service, handoffs, start, width)
        measured = None if group_medians is None else group_medians.get(start)
        if measured is not None:
            duration = measured
        peak = (storage['operation_peak_bytes'][start] if width == 1
                else storage['group_peak_bytes'][f'{start}:{width}'])
        segment = dict(start_ns=offset, end_ns=offset + duration, start=start, width=width,
                       operations=[op['id'] for op in operations[start:start + width]],
                       cpu_ids=cpus[:width], peak_tensor_storage_bytes=peak,
                       dispatch_join_ns=0 if width == 1 else handoffs[str(width)])
        if measured is not None:
            segment['duration_basis'] = 'MEASURED_CONCURRENT_FORK_JOIN'
        elif start in post_join_indices:
            segment['duration_basis'] = 'MEASURED_POST_JOIN_CONSUMER'
        timeline.append(segment)
        start += width
        offset += duration
    return timeline


def _timeline_prediction(timeline, lane_samples, handoff_samples, *, groups=None):
    estimate = timeline[-1]['end_ns']
    totals = []
    for row, measurements in enumerate(zip(*lane_samples, strict=True)):
        handoffs = {width: values[row] for width, values in handoff_samples.items()}
        observed = {} if groups is None else groups[row]
        totals.append(sum(observed[segment['start']] if segment['start'] in observed else
                          _segment_time(measurements, handoffs, segment['start'], segment['width'])
                          for segment in timeline))
    return dict(estimate_ns=estimate, sample_min_ns=min(estimate, *totals), sample_max_ns=max(estimate, *totals))


def _price_numa_context(context, operations, cpus, service, handoffs, lane_samples, handoff_samples, storage,
                        *, group_evidence=None, post_join_evidence=None):
    choices = context['windows']
    lane_by_op = [0] * len(operations)
    for start, width in choices:
        lane_by_op[start:start + width] = range(width)
    contextual_service = [list(row) for row in service]
    contextual_samples = [[list(row) for row in samples] for samples in lane_samples]
    post_join_samples = {}
    if post_join_evidence is not None:
        post_join_samples = dict(zip(post_join_evidence['operation_indices'],
            zip(*post_join_evidence['consumer_samples_ns'], strict=True), strict=True))
    for i, samples in enumerate(zip(*context['replacement_operation_samples_ns'], strict=True)):
        if i in post_join_samples:
            samples = post_join_samples[i]
        if samples[0] is None:
            continue
        lane = lane_by_op[i]
        contextual_service[lane][i] = median_low(samples)
        for row, duration in zip(contextual_samples[lane], samples, strict=True):
            row[i] = duration
    starts = [start for start, _ in choices]
    medians, groups = None, None
    if group_evidence is not None:
        medians = dict(zip(starts, group_evidence['group_median_ns'], strict=True))
        groups = [dict(zip(starts, row, strict=True)) for row in group_evidence['group_samples_ns']]
    timeline = _timeline(operations, cpus, contextual_service, handoffs, choices, storage,
                         group_medians=medians, post_join_indices=post_join_samples)
    return timeline, _timeline_prediction(timeline, contextual_samples, handoff_samples, groups=groups)


def model_cpu_team(calibration, recipe, schedule, warmup, *, tensor_budget_bytes=None):
    """Select a resource-legal ordered tiling from separately measured primitives."""
    windows = _team_groups(recipe, schedule)
    operations = recipe['operations']
    service, handoffs, lane_samples, handoff_samples = _measurements(
        calibration, warmup, len(operations), windows, full_graph=schedule['schema_version'] == 2)
    storage, possible = _storage(calibration, recipe, windows, tensor_budget_bytes)
    numa = model_aten_team_numa_locality(calibration.get('numa_team'), recipe, schedule, warmup,
                                        result_storage_bytes=calibration['result_storage_bytes'])
    concurrent = read_aten_team_group_evidence(calibration.get('concurrent_team'), recipe, schedule, warmup)
    post_join = read_aten_post_join_evidence(calibration.get('post_join_team'), recipe, schedule, warmup)
    cross_numa = schedule.get('cross_numa') is True
    contexts = [] if numa is None else numa['contexts']
    group_contexts = {} if concurrent is None else {
        tuple(map(tuple, context['windows'])): context for context in concurrent['contexts']}
    context_keys = [tuple(map(tuple, context['windows'])) for context in contexts]
    if concurrent is not None and (numa is None or concurrent['status'] != numa['status'] or
                                   list(group_contexts) != context_keys):
        raise ValueError('team simultaneous timings and NUMA contexts describe different window sets')
    post_join_contexts = {} if post_join is None else {
        tuple(map(tuple, context['windows'])): context for context in post_join['contexts']}
    if post_join is not None and (concurrent is None or post_join['status'] != concurrent['status'] or
                                  list(post_join_contexts) != context_keys):
        raise ValueError('team serial consumer timings and simultaneous timings describe different window sets')
    nodes = [lane['numa_node'] for lane in schedule['lanes']]
    if contexts:
        by_mode = {'SERIAL': []}
    else:
        without_pages = {window for window in possible if len(set(nodes[:window[1]])) == 1}
        by_mode = _best_tilings(service, handoffs, without_pages if cross_numa else possible)
    timelines = {mode: _timeline(operations, schedule['cpus'], service, handoffs, choices, storage)
                 for mode, choices in by_mode.items()}
    predictions = {mode: _timeline_prediction(timeline, lane_samples, handoff_samples)
                   for mode, timeline in timelines.items()}
    context_costs = []
    for context in contexts:
        choices = context['windows']
        within_budget = all(tuple(window) in possible for window in choices)
        cost = dict(windows=choices, within_tensor_budget=within_budget,
                    complete_page_evidence=context['complete_page_evidence'])
        context_costs.append(cost)
        if not within_budget:
            cost['reason'] = 'TENSOR_BUDGET_EXCEEDED'
            continue
        widest = max(width for _, width in choices)
        if len(set(nodes[:widest])) > 1 and not context['complete_page_evidence']:
            cost['reason'] = 'CROSS_NUMA_PAGE_EVIDENCE_INCOMPLETE'
            continue
        context_key = tuple(map(tuple, choices))
        groups = group_contexts.get(context_key)
        consumers = post_join_contexts.get(context_key)
        timeline, prediction = _price_numa_context(context, operations, schedule['cpus'], service, handoffs,
            lane_samples, handoff_samples, storage, group_evidence=groups, post_join_evidence=consumers)
        if consumers is not None and consumers['operation_indices']:
            _, independent = _price_numa_context(context, operations, schedule['cpus'], service, handoffs,
                lane_samples, handoff_samples, storage, group_evidence=groups)
            cost.update(post_join_prediction=prediction, independent_consumer_prediction=independent)
            prediction = dict(prediction, sample_min_ns=min(prediction['sample_min_ns'], independent['sample_min_ns']),
                              sample_max_ns=max(prediction['sample_max_ns'], independent['sample_max_ns']))
        cost['prediction'] = prediction
        mode = _MODE_NAMES[widest - 1]
        score = (prediction['estimate_ns'], len(choices), choices)
        if mode not in predictions or score < (predictions[mode]['estimate_ns'], len(by_mode[mode]), by_mode[mode]):
            by_mode[mode], timelines[mode], predictions[mode] = choices, timeline, prediction

    selected = min(by_mode, key=lambda mode: (predictions[mode]['estimate_ns'], _MODE_NAMES.index(mode)))
    choices = {tuple(group) for group in by_mode[selected]}
    candidates = [dict(group, peak_tensor_storage_bytes=storage['group_peak_bytes'][f'{start}:{width}'],
                       within_tensor_budget=(start, width) in possible, selected=(start, width) in choices)
                  for (start, width), group in windows.items()]
    model = dict(basis='MINIMUM_ADJACENT_ATEN_WINDOW_TILING_PER_EXECUTION_WIDTH', selected=selected,
                selected_windows=by_mode[selected], mode_windows=by_mode, timeline=timelines[selected],
                timelines=timelines, predictions=predictions, candidates=candidates, tensor_storage=storage,
                operations=[dict(id=op['id'], service_ns=duration) for op, duration in zip(operations, service[0], strict=True)],
                lane_service_ns=service, dispatch_join_ns=handoffs, bounds='CALIBRATION_SPAN_NOT_PHYSICAL_BOUNDS',
                shared_memory_contention_modelled=False, numa_page_placement_modelled=False)
    if numa is not None:
        model.update(numa_locality=numa, numa_context_costs=context_costs,
                     numa_page_placement_modelled=bool(contexts and numa['sampled_node_residency_observed']))
    if concurrent is not None:
        model.update(concurrent_team=concurrent, concurrent_group_interaction_measured=bool(group_contexts))
    if post_join is not None:
        model['post_join_team'] = post_join
        if any('post_join_prediction' in cost for cost in context_costs):
            model['bounds'] = 'UNION_OF_CONSUMER_CALIBRATION_SPANS_NOT_PHYSICAL_BOUNDS'
    if cross_numa and 'THREE_CPU' not in predictions and (not contexts or any(
            cost.get('reason') == 'CROSS_NUMA_PAGE_EVIDENCE_INCOMPLETE' for cost in context_costs)):
        model['parallel_frontier'] = 'CROSS_NUMA_PAGE_CONTEXT_UNAVAILABLE'
    return model
