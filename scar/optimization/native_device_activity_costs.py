"""Reprice a fixed hybrid treatment using scoped native service observations."""

from dataclasses import replace
import re
from statistics import median_low

from scar.ir.implementation_plan import CostEstimate, PlanningTimingConstraint, digest, problem_document
from scar.optimization.constructive_planner import construct_implementations
from scar.optimization.implementation_costs import CostQuery, query_key
from scar.optimization.native_device_comparison import native_device_probe_reason


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
