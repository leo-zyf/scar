"""Native device activity costs."""

from dataclasses import replace
import re
from statistics import median_low
from scar.ir.implementation_plan import (
    CostEstimate,
    PlanningTimingConstraint,
    digest,
    problem_document,
    PlanningProblem,
)
from scar.optimization.constructive_planner import construct_implementations
from scar.optimization.implementation_costs import CostQuery, query_key
from scar.optimization.native_device_comparison import native_device_probe_reason
from collections import defaultdict
from graphlib import CycleError, TopologicalSorter


# Reprice a fixed hybrid treatment using scoped native service observations.
SCHEMA = 'scar.native-device-activity-pricing'
OBSERVATION_SCHEMA = 'scar.native-device-activity-observations'
SAMPLE_SCHEMA = 'scar.hybrid-native-activity-observation'
_BINDING_SCHEMA = 'scar.native-device-activity-binding'
_CLOCKS = {'LIBTORCH_ATEN_CPU_OUT_F32_V1': 'HOST_STEADY',
           'CUBLAS_SGEMM_F32_COLUMN_MAJOR_V1': 'CUDA_EVENT', 'CUDA_MEMCPY_ASYNC': 'CUDA_EVENT'}
_BOUNDARY = ('allocation_ns', 'input_ns', 'setup_ns', 'host_wait_ns', 'join_ns', 'publish_ns')
_PREFLIGHT_REJECTIONS = {'DEVICE_UNAVAILABLE': 'CUDA_DEVICE_UNAVAILABLE', 'PROBE_FAILED': 'CUDA_PROBE_FAILED',
                         'DEVICE_SCOPE_UNVERIFIED': None}
_HASH = re.compile('[0-9a-f]{64}')


def _payload(row):
    if (type(row) is not dict or type(row.get('sha256')) is not str or not _HASH.fullmatch(row['sha256'])
            or type(row.get('size_bytes')) is not int or row['size_bytes'] < 0):
        raise ValueError('native activity evidence requires an exact byte identity')
    return dict(sha256=row['sha256'], size_bytes=row['size_bytes'])


def _reference(row):
    if type(row) is not dict or type(row.get('path')) is not str or not row['path']:
        raise ValueError('native activity evidence requires an archived file reference')
    return dict(path=row['path'], **_payload(row))


def _ns(value):
    return type(value) is int and 0 <= value < 2**60


def verify_native_activity_sample(sample, steps):
    """Check one native timing wire against its ordered step binding."""
    if (type(sample) is not dict or set(sample) != {'schema', 'schema_version', 'activities', 'boundary'}
            or sample['schema'] != SAMPLE_SCHEMA or type(sample['schema_version']) is not int
            or sample['schema_version'] != 1 or type(sample['activities']) is not list
            or len(sample['activities']) != len(steps) or type(sample['boundary']) is not dict
            or set(sample['boundary']) != set(_BOUNDARY) or not all(map(_ns, sample['boundary'].values()))):
        raise ValueError('native activity sample is missing its complete timing wire')
    for stage, row in zip(steps, sample['activities'], strict=True):
        if (type(row) is not dict or set(row) != {'step_index', 'clock', 'service_ns'}
                or type(row['step_index']) is not int or row['step_index'] != stage['index']
                or row['clock'] != stage['clock'] or not _ns(row['service_ns'])):
            raise ValueError('native sample does not preserve activity index, clock or nanosecond extent')


def _bound_steps(problem, binding):
    if (binding['schema'] != _BINDING_SCHEMA or type(binding['schema_version']) is not int
            or binding['schema_version'] != 1 or binding['problem_sha256'] != digest(problem)
            or problem.activities or problem.timing_constraints or any(edge.transfers for edge in problem.data_edges)):
        raise ValueError('native activity repricing requires the bound original fixed placement')
    modes, returns = {}, []
    for operation in problem.operations:
        if len(operation.options) != 1 or operation.options[0].id != operation.baseline_option:
            raise ValueError('native activity repricing requires one fixed choice per operation')
        mode, = operation.options
        if mode.lowering == 'CUDA_COPY_RETURN':
            if mode.cost.estimate_ns != 0 or mode.cost.context != 'LOGICAL_ENDPOINT':
                raise ValueError('native activity return endpoint must remain logical')
            returns.append(operation.id)
        elif mode.lowering in _CLOCKS:
            modes[operation.id] = mode
        else:
            raise ValueError('native activity has no registered clock binding')
    steps = binding['activities']
    if (len(returns) != 1 or not modes or len(steps) != len(modes)
            or {row['operation'] for row in steps} != modes.keys()):
        raise ValueError('activity binding must cover every executable step exactly once')
    for index, row in enumerate(steps):
        mode = modes[row['operation']]
        if (type(row['index']) is not int or row['index'] != index or row['option'] != mode.id
                or row['lowering'] != mode.lowering or row['clock'] != _CLOCKS[mode.lowering]
                or row['execution_lane'] not in mode.demands or type(row['wait_for']) is not list
                or any(type(wait) is not int or not 0 <= wait < index for wait in row['wait_for'])):
            raise ValueError('activity identity or timing clock differs from the bound implementation')
    return steps


def _verify_runs(observation, binding, binding_ref, steps):
    if (observation['schema'] != OBSERVATION_SCHEMA or type(observation['schema_version']) is not int
            or observation['schema_version'] != 1 or observation['binding_sha256'] != binding_ref['sha256']):
        raise ValueError('native activity calibration differs from the archived binding')
    warmup, count = observation['warmup_runs'], observation['measurement_runs']
    if (type(warmup) is not int or warmup < 0 or type(count) is not int or count < 3
            or type(observation['runs']) is not list or len(observation['runs']) > warmup + count):
        raise ValueError('native activity calibration requires at least three measured runs')
    inputs = [_reference(row) for row in observation['inputs']]
    references = [_reference(row) for row in observation['references']]
    if not references:
        raise ValueError('native activity calibration requires ordered reference outputs')
    context = observation['context']
    identity = observation['package_identity']['CUDA']
    if (context['device_ordinal'] != binding['device']
            or any(_payload(identity[name]) != _payload(binding['references'][name])
                   for name in ('executable', 'launcher'))):
        raise ValueError('activity calibration names a different device or treatment binary')
    phase = observation['phase']
    runs, failure = observation['runs'], observation['failure']
    probe = native_device_probe_reason(observation['probe'], device=binding['device'])
    device = context['cuda_identity']
    qualified = type(device) is dict and bool(device) and context['cuda_identity_error'] is None
    if phase in _PREFLIGHT_REJECTIONS:
        scope_error = context['cuda_identity_error']
        if (runs or failure is not None or probe != _PREFLIGHT_REJECTIONS[phase] or device is not None
                or phase == 'DEVICE_SCOPE_UNVERIFIED' and (type(scope_error) is not str or not scope_error)
                or phase != 'DEVICE_SCOPE_UNVERIFIED' and scope_error is not None):
            raise ValueError('rejected CUDA preflight cannot support activity costs')
    elif (phase not in ('COMPLETE', 'CALIBRATION_FAILED') or probe is not None or not qualified
          or phase == 'COMPLETE' and (len(runs) != warmup + count or failure is not None)
          or phase == 'CALIBRATION_FAILED' and (type(failure) is not dict or not failure)):
        raise ValueError('native activity suite does not match its calibrated device or completion state')
    expected_outputs = [_payload(row) for row in references]
    for index, run in enumerate(runs):
        if (type(run) is not dict or set(run) != {'ordinal', 'sample', 'outputs'}
                or type(run['ordinal']) is not int or run['ordinal'] != index or run['outputs'] != expected_outputs):
            raise ValueError('native activity suite contains a mismatched or incomplete reference observation')
        verify_native_activity_sample(run['sample'], steps)
    cost_context = dict(runtime=context, package_identity=identity, input_identities=[_payload(row) for row in inputs],
                        reference_identities=expected_outputs)
    return runs[warmup:] if phase == 'COMPLETE' else [], cost_context


def _stats(values):
    return dict(estimate_ns=median_low(values), lower_ns=min(values), upper_ns=max(values))


def _queue_order(steps, binding_ref, *, host_dispatch=False):
    # Preserve emitted FIFO order and cross-lane wait relations in the diagnostic model.
    host_lane = (next((step['execution_lane'] for step in steps if step['clock'] == 'HOST_STEADY'), None)
                 if host_dispatch else None)
    last_host_call = None
    previous, constraints = {}, []
    for step in steps:
        name, lane = (step[key] for key in ('operation', 'execution_lane'))
        dependencies = {steps[index]['operation'] for index in step['wait_for']}
        if lane in previous:
            dependencies.add(previous[lane])
        if host_dispatch and lane == host_lane:
            last_host_call = name
        elif last_host_call is not None:
            # The native host loop cannot enqueue this GPU step until the synchronous call returns.
            dependencies.add(last_host_call)
        for before in sorted(dependencies):
            constraints.append(PlanningTimingConstraint(f'native-service-order/{len(constraints)}',
                before, 'END', name, 'START', source_refs=(binding_ref,)))
        previous[lane] = name
    return tuple(constraints)


def reprice_native_device_activities(problem, binding, observation, *, binding_ref, observation_ref):
    """Return a typed problem document, a deterministic plan and a diagnostic summary.

    The caller replays the native binding, verifies the pinned files and
    qualifies device identity. This step verifies their observation joins.
    """
    binding_ref, observation_ref = _reference(binding_ref), _reference(observation_ref)
    steps = _bound_steps(problem, binding)
    runs, context = _verify_runs(observation, binding, binding_ref, steps)
    estimates = {}
    for step in steps:
        index = step['index']
        measured = [run['sample']['activities'][index]['service_ns'] for run in runs]
        scope = ('CUDA_STREAM_EVENT_SERVICE_AS_OBSERVED' if step['clock'] == 'CUDA_EVENT'
                 else 'HOST_STEADY_SERVICE_EXCLUDING_DEPENDENCY_WAIT')
        key = query_key(CostQuery(scope, dict(context, binding_sha256=binding_ref['sha256'], activity=step)))
        refs = (dict(binding_ref, locator=f'#/activities/{index}'), dict(observation_ref, locator='#/runs'))
        if not measured or min(measured) == 0:
            reason = observation['phase'] if not measured else 'ZERO_NATIVE_TIMER_SAMPLE'
            estimates[step['operation']] = CostEstimate(None, None, None, 'UNKNOWN', key, refs, (reason,))
        else:
            stats = _stats(measured)
            estimates[step['operation']] = CostEstimate(**stats, basis='EMPIRICAL_MODEL', context=key, source_refs=refs)
    options = tuple(replace(row, options=(replace(row.options[0], cost=estimates[row.id]),))
                    if row.id in estimates else row for row in problem.operations)
    replaced = {'OFFLINE_SYNTHETIC_ONLY', 'ACTUAL_DEVICE_TIMING_UNKNOWN'}
    assumptions = (*[item for item in problem.assumptions if item not in replaced],
        'DECLARED_RESOURCE_CAPACITIES_NOT_DEVICE_MEASUREMENTS', 'NATIVE_SERVICE_COSTS_EXACT_OBSERVED_TREATMENT_ONLY',
        'OBSERVED_RESOURCE_ORDER_RETAINED', 'HOST_BOUNDARY_COSTS_OUTSIDE_ACTIVITY_MODEL',
        'CUDA_EVENT_AND_HOST_STEADY_INTERVALS_DO_NOT_SHARE_A_TIMESTAMP_ORIGIN')
    repriced = replace(problem, operations=options, assumptions=assumptions,
                       timing_constraints=_queue_order(steps, binding_ref))
    schedule = construct_implementations(repriced, baseline_only=True)
    schedule.pop('elapsed_seconds', None)
    unknown = [name for name, cost in estimates.items() if cost.basis == 'UNKNOWN']
    summary = dict(schema=SCHEMA, schema_version=1, phase=observation['phase'], measured_runs=len(runs),
        known_activity_count=len(estimates) - len(unknown), unknown_activities=unknown,
        estimate='LOWER_MEDIAN_INTEGER_NS', range='OBSERVED_SAMPLE_EXTREMA_NOT_GUARANTEED_BOUNDS',
        boundary_host_steady_ns={key: _stats([row['sample']['boundary'][key] for row in runs]) for key in _BOUNDARY}
            if runs else None,
        schedule_status=schedule['status'], service_model_makespan_ns=schedule['makespan_ns'],
        interpretation='FIXED_PLACEMENT_AS_OBSERVED_DIAGNOSTIC_NOT_END_TO_END_PERFORMANCE',
        scheduling_optimality_claimed=False, execution_authorized=False, deployment_authorized=False)
    return problem_document(repriced), schedule, summary


# Project observed service ranges through one replayed native schedule.
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
