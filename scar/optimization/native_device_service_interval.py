"""Project observed service ranges through one replayed native schedule."""

from collections import defaultdict
from graphlib import CycleError, TopologicalSorter

from scar.ir.implementation_plan import CostEstimate, PlanningProblem


_SCOPE = 'FIXED_RESOURCE_ORDER_SERVICE_SAMPLE_ENVELOPE'
_LOGICAL_ZERO = CostEstimate(0, 0, 0, 'DECLARED_MODEL', 'LOGICAL_ENDPOINT')
_PERSISTENT_ABI = 'ALL_FILE_ABI_BUFFERS_AND_PROVIDER_WORKSPACE_RETAINED_THROUGH_RETURN'


def _duration_range(mode):
    cost = mode.cost
    if (mode.lowering in ('REGION_VALUE_HANDOFF', 'CUDA_COPY_RETURN') and cost == _LOGICAL_ZERO
            and not (mode.demands or mode.output_locations or mode.workspace_bytes or mode.unresolved_resources)):
        return 0, 0, 0
    lower, point, upper = cost.lower_ns, cost.estimate_ns, cost.upper_ns
    if (cost.basis == 'EMPIRICAL_MODEL' and cost.source_refs and not cost.missing_terms
            and all(type(value) is int for value in (lower, point, upper)) and 0 < lower <= point <= upper):
        return lower, point, upper
    return None


def _entailed(before, after, predecessors, position):
    pending, seen = [after], {after}
    while pending:
        for parent in predecessors[pending.pop()]:
            if parent == before:
                return True
            if position[parent] > position[before] and parent not in seen:
                seen.add(parent)
                pending.append(parent)
    return False


def _project(order, predecessors, durations, bound):
    points = {}
    for name in order:
        start = max((points[parent][1] + delay for parent, delay in predecessors[name].items()), default=0)
        points[name] = start, start + durations[name][bound]
    return points


def _serial_memory(problem, modes, position, values):
    last_use = {}
    for edge in problem.data_edges:
        last_use[edge.value] = max(last_use.get(edge.value, 0), position[edge.consumer])
    events = defaultdict(list)

    def lease(pool, nbytes, first, last):
        if nbytes:
            events[pool].extend(((2 * first, nbytes), (2 * last + 1, -nbytes)))

    for name, mode in modes.items():
        for pool, nbytes in mode.workspace_bytes.items():
            lease(pool, nbytes, position[name], position[name])
    for value, edge in values.items():
        if edge.producer is not None:
            lease(modes[edge.producer].output_locations[value], edge.nbytes,
                  position[edge.producer], last_use[value])
    peaks = dict(problem.reserved_memory)
    for pool, changes in events.items():
        current = peak = peaks.get(pool, 0)
        for _, change in sorted(changes):
            current += change
            peak = max(peak, current)
        peaks[pool] = peak
    return peaks


def _resident_memory(problem, modes, lane_for, values):
    peaks = defaultdict(int, problem.reserved_memory)
    for value, edge in values.items():
        pool = edge.external_location if edge.producer is None else modes[edge.producer].output_locations[value]
        peaks[pool] += edge.nbytes
    workspace = defaultdict(int)
    for name, mode in modes.items():
        for pool, nbytes in mode.workspace_bytes.items():
            key = pool, lane_for[name]
            workspace[key] = max(workspace[key], nbytes)
    for (pool, _), nbytes in workspace.items():
        peaks[pool] += nbytes
    return dict(peaks)


def native_device_service_interval(problem: PlanningProblem, schedule: dict) -> dict:
    """Bound service time over replayed sample extrema for a verified schedule.

    Exclusive resource order must follow from existing dependencies. The
    envelope models scoped services under the declared capacities. Sample
    extrema do not give a confidence interval or an end-to-end time bound.
    """
    estimate = schedule['makespan_ns']

    def unresolved(reason, **details):
        return dict(lower_ns=0, estimate_ns=estimate, upper_ns=None, status='UNRESOLVED', reason=reason, **details)

    if (type(estimate) is not int or problem.activities or schedule['transfers'] or schedule.get('activities')
            or any(len(op.options) != 1 for op in problem.operations) or any(e.transfers for e in problem.data_edges)):
        return unresolved('UNSUPPORTED_FIXED_NATIVE_SCHEDULE')
    modes = {op.id: op.options[0] for op in problem.operations}
    predecessors = {op.id: dict.fromkeys(op.predecessors, 0) for op in problem.operations}
    for edge in problem.data_edges:
        if edge.producer is not None:
            predecessors[edge.consumer][edge.producer] = 0
    for constraint in problem.timing_constraints:
        if (constraint.before_endpoint != 'END' or constraint.after_endpoint != 'START' or constraint.when
                or constraint.minimum_ns < 0 or constraint.maximum_ns is not None):
            return unresolved('UNSUPPORTED_SERVICE_TIMING_CONSTRAINT', constraint=constraint.id)
        incoming = predecessors[constraint.after]
        incoming[constraint.before] = max(incoming.get(constraint.before, 0), constraint.minimum_ns)
    try:
        order = tuple(TopologicalSorter(predecessors).static_order())
    except CycleError:
        return unresolved('FIXED_ORDER_DEPENDENCY_CYCLE')
    position = {name: index for index, name in enumerate(order)}
    resources = {row.id: row for row in problem.resources}
    durations, lane_for = {}, {}
    lanes = defaultdict(list)
    for row in sorted(schedule['selected'], key=lambda item: (item['start_ns'], item['end_ns'], item['operation'])):
        name = row['operation']
        mode = modes[name]
        extent = _duration_range(mode)
        if extent is None:
            return unresolved('MISSING_EMPIRICAL_SERVICE_ENVELOPE', operation=name)
        durations[name] = extent
        demand = [resource for resource, units in mode.demands.items() if units]
        if extent[1] and not demand:
            return unresolved('SERVICE_EXECUTION_LANE_MISSING', operation=name)
        for resource in demand:
            lane = resources[resource]
            if lane.kind not in ('EXECUTION', 'CHANNEL') or lane.capacity != 1 or mode.demands[resource] != 1:
                return unresolved('UNSUPPORTED_EXECUTION_LANE', operation=name, resource=resource)
            lanes[resource].append(name)
        if demand:
            lane_for[name] = min(demand)
    for resource, names in lanes.items():
        for before, after in zip(names, names[1:]):
            if not _entailed(before, after, predecessors, position):
                return unresolved('RESOURCE_LANE_ORDER_NOT_ENTAILED', resource=resource, before=before, after=after)

    points = _project(order, predecessors, durations, 1)
    for row in schedule['selected']:
        if points[row['operation']] != (row['start_ns'], row['end_ns']):
            return unresolved('POINT_TIMELINE_NOT_REPRODUCED', operation=row['operation'])
    lower = max(end for _, end in _project(order, predecessors, durations, 0).values())
    upper = max(end for _, end in _project(order, predecessors, durations, 2).values())
    if estimate != max(end for _, end in points.values()) or not lower <= estimate <= upper:
        return unresolved('POINT_MAKESPAN_NOT_REPRODUCED')

    values = {edge.value: edge for edge in problem.data_edges}
    if (_PERSISTENT_ABI not in problem.assumptions
            and all(before in predecessors[after] for before, after in zip(order, order[1:]))):
        memory = _serial_memory(problem, modes, position, values)
        memory_proof = 'TOTAL_ORDER_VALUE_AND_WORKSPACE_LIFETIMES'
    else:
        memory = _resident_memory(problem, modes, lane_for, values)
        memory_proof = 'SIMULTANEOUS_MATERIALS_AND_PER_LANE_MAX_WORKSPACE'
    for pool, nbytes in sorted(memory.items()):
        if nbytes > resources[pool].capacity:
            return unresolved('MEMORY_CAPACITY_NOT_PROVEN_ACROSS_SERVICE_ENVELOPE', memory=pool,
                              required_bytes=nbytes, capacity_bytes=resources[pool].capacity)
    return dict(lower_ns=lower, estimate_ns=estimate, upper_ns=upper, status='BOUNDED', scope=_SCOPE,
                memory_proof=memory_proof, memory_upper_bytes=memory, sample_envelope_is_physical_bound=False)
