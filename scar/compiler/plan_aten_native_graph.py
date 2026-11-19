"""Predict and verify a standalone ATen CPU graph's caller/worker schedule."""

import argparse
import json
from math import prod
from pathlib import Path
import shlex
from statistics import median_low, variance
import subprocess
from tempfile import TemporaryDirectory

from scar.analysis.target_graph import TargetHardwareGraph
from scar.codegen.aten_native_graph import _read_declared
from scar.codegen.native_provider_bundle import _content
from scar.compiler.plan_aten_boundary import _checked, _target, _write
from scar.ir.execution_recipe import artifact_reference
from scar.ir.implementation_plan import (
    CostEstimate, ImplementationOption, PlanningOperation, PlanningProblem, PlanningResource, problem_document,
)
from scar.ir.program.target_hardware import CpuExecutor, MemoryDomain, MemoryKind
from scar.ir.record_codec import encode, loads
from scar.optimization.aten_boundary_trials import _measurements
from scar.optimization.aten_joint_numa_locality import model_aten_joint_numa_locality
from scar.optimization.aten_native_schedule import propose_cpu_pair, propose_cpu_windows, select_cpu_windows
from scar.optimization.aten_numa_locality import model_aten_numa_locality
from scar.optimization.aten_tensor_lifetime import tensor_footprint, tensor_releases
from scar.optimization.constructive_planner import construct_implementations
from scar.optimization.joint_planner import verify_execution_plan
from scar.optimization.paired_entry_selection import _metrics


_EXE = 'aten-native-graph'
_ENV = {'PATH': '/usr/bin:/bin', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
_GAIN = 0.05
_MEMORY_PROFILE = 'LAST_CONSUMER_OR_EXPORT_V1'
_SCHEDULE_KEYS = {
    'ORDERED_DEFAULT_WITH_DECLARED_CPU_PAIR_TRIAL': 'cpu_pair',
    'ORDERED_DEFAULT_WITH_DECLARED_CPU_WINDOWS_TRIAL': 'cpu_windows',
    'ORDERED_DEFAULT_WITH_DECLARED_CPU_TEAM_TRIAL': 'cpu_team',
}


def _receipt(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), **_content(path))


def _bundle(path, digest):
    path, artifact, receipt = _read_declared(path, digest)
    key = _SCHEDULE_KEYS.get(artifact['implementation'])
    if (artifact['schema'], artifact['schema_version']) != ('scar.aten-native-graph-delivery', 1) or key is None:
        raise ValueError('generated graph bundle with a CPU concurrency contract required')
    recipe_path = path.with_name('recipe.json')
    _, recipe, _ = _read_declared(recipe_path, artifact['files']['recipe.json']['sha256'])
    lifetime = artifact.get('tensor_lifetime')
    if lifetime is not None and lifetime != dict(profile=_MEMORY_PROFILE, release_after=tensor_releases(recipe)):
        raise ValueError('generated result lifetime table differs from the recipe')
    executable = path.with_name(_EXE)
    if _content(executable) != artifact['files'][_EXE]:
        raise ValueError('generated native graph executable differs from the bundle receipt')
    contract = artifact['concurrency_contract']
    _, requirements, _ = _read_declared(contract['path'], contract['sha256'])
    schedule = artifact[key]
    hardware = Path(schedule['target']['uri']).parent
    if key == 'cpu_team':
        from scar.optimization.aten_native_team import TEAM_PLACEMENT_POLICY, propose_cpu_team

        replayed = propose_cpu_team(recipe, contract=requirements, target=hardware, cpu_ids=schedule['cpus'],
                                    allow_cross_numa=schedule.get('placement_policy') == TEAM_PLACEMENT_POLICY,
                                    ready_dag=schedule['schema_version'] == 2)
    else:
        qualify = propose_cpu_windows if key == 'cpu_windows' else propose_cpu_pair
        replayed = qualify(recipe, contract=requirements, target=hardware, cpu_ids=schedule['cpus'])
    if replayed != schedule:
        raise ValueError('retained CPU placement differs from the recipe, contract or target hardware graph')
    for library in artifact['provider']['libraries']:
        _checked(library)
    origin = dict(uri=receipt['path'], sha256=receipt['sha256'], size_bytes=receipt['size_bytes'],
                  schema=artifact['schema'])
    return artifact, origin, recipe, hardware, executable, key


def _services(calibration, recipe, warmup):
    op_samples = calibration['operation_samples_ns']
    dispatch_samples = calibration['empty_dispatch_samples_ns']
    count = len(recipe['operations'])
    if (len(op_samples) != warmup or len(dispatch_samples) != warmup
            or any(len(row) != count for row in op_samples)
            or any(type(n) is not int or n <= 0 for row in [*op_samples, dispatch_samples] for n in row)):
        raise ValueError('one exact service sample vector and dispatch observation per calibration row required')

    service = [median_low(samples) for samples in zip(*op_samples, strict=True)]
    dispatch = median_low(dispatch_samples)
    return service, dispatch, op_samples, dispatch_samples


def _prediction(estimate, values):
    return dict(estimate_ns=estimate, sample_min_ns=min(estimate, *values), sample_max_ns=max(estimate, *values))


def _component_model(calibration, recipe, schedule, warmup):
    service, dispatch, op_samples, dispatch_samples = _services(calibration, recipe, warmup)
    first, second = schedule['pair_indices']

    def totals(row, cost):
        serial = sum(row)
        return serial, serial - min(row[first], row[second]) + cost

    estimates = totals(service, dispatch)
    by_mode = {}
    sample_totals = [totals(row, cost) for row, cost in zip(op_samples, dispatch_samples, strict=True)]
    samples = zip(*sample_totals, strict=True)
    for mode, estimate, values in zip(('SERIAL', 'TWO_CPU'), estimates, samples, strict=True):
        by_mode[mode] = _prediction(estimate, values)
    return dict(basis='SUM_SERIAL_PREFIX_AND_SUFFIX_WITH_MAX_OF_READY_PARALLEL_PAIR',
                operations=[dict(id=op['id'], service_ns=ns) for op, ns in zip(recipe['operations'], service, strict=True)],
                pair=schedule['pair'], empty_dispatch_join_ns=dispatch, predictions=by_mode,
                bounds='CALIBRATION_SPAN_NOT_PHYSICAL_BOUNDS', shared_memory_contention_modelled=False)


def _timeline(names, service, choices, cpus, dispatch, worker_service=None):
    timeline = []
    cursor = 0
    i = 0
    while i < len(names):
        window = i in choices
        indices = [i, i + 1] if window else [i]
        overhead = dispatch if window else 0
        duration = service[i]
        if window:
            right = service if worker_service is None else worker_service
            duration = max(duration, right[i + 1])
        finish = cursor + duration + overhead
        timeline.append(dict(start_ns=cursor, end_ns=finish, operations=[names[j] for j in indices],
            cpu_ids=cpus if window else cpus[:1], dispatch_join_ns=overhead))
        cursor = finish
        i += len(indices)
    return timeline


def _numa_context(calibration, recipe, schedule, warmup):
    locality = calibration.get('numa_locality')
    if locality is None:
        return None
    graph = TargetHardwareGraph(Path(schedule['target']['uri']).parent)
    if encode(graph._snapshot) != schedule['target']:
        raise ValueError('NUMA evidence requires the original scheduled target graph')
    executors = [row for row in graph.resources() if type(row) is CpuExecutor]
    lanes = []
    for cpu in schedule['cpus']:
        owners = [row for row in executors if cpu in row.cpu_ids]
        owner = owners[0] if len(owners) == 1 else None
        lanes.append(dict(cpu=cpu, numa_node=None if owner is None else owner.numa_node,
                          fact=None if owner is None else encode(owner.evidence)))
    model = model_aten_joint_numa_locality if locality.get('schema_version') == 2 else model_aten_numa_locality
    context = model(locality, recipe, schedule, caller_numa_node=lanes[0]['numa_node'],
        worker_numa_node=lanes[1]['numa_node'], warmup_pairs=warmup,
        result_storage_bytes=calibration.get('result_storage_bytes'))
    return dict(context, lanes=lanes)


def _caller_service(service, replacements, choices):
    if replacements is None or replacements['window_start'] not in choices:
        return service
    return [duration if changed is None else changed
            for duration, changed in zip(service, replacements['caller_service_ns'], strict=True)]


def _joint_window_model(recipe, schedule, numa, storage, service, dispatch, op_samples, dispatch_samples):
    names = [op['id'] for op in recipe['operations']]
    serial = _prediction(sum(service), [sum(row) for row in op_samples])
    allowed_starts = ({row['pair_indices'][0] for row in schedule['windows']} if storage is None
                      else set(storage['candidate_starts_within_budget']))
    contexts = []
    for context in numa['contexts']:
        starts = context['window_starts']
        observations = context['replacement_operation_samples_ns']
        lane_services = [old if column[0] is None else median_low(column)
                         for old, column in zip(service, zip(*observations, strict=True), strict=True)]
        estimate = sum(lane_services) + sum(dispatch - min(lane_services[i], lane_services[i + 1]) for i in starts)
        totals = []
        for baseline, replacements, handoff in zip(op_samples, observations, dispatch_samples, strict=True):
            row = [old if new is None else new for old, new in zip(baseline, replacements, strict=True)]
            totals.append(sum(row) + sum(handoff - min(row[i], row[i + 1]) for i in starts))
        contexts.append(dict(window_starts=starts, lane_service_ns=lane_services,
                             within_tensor_budget=set(starts) <= allowed_starts, prediction=_prediction(estimate, totals)))

    affordable = (row for row in contexts if row['within_tensor_budget'])
    best = min(affordable, key=lambda row: (row['prediction']['estimate_ns'], len(row['window_starts']),
                                          row['window_starts']), default=None)
    selected = best is not None and best['prediction']['estimate_ns'] < serial['estimate_ns']
    starts = best['window_starts'] if selected else []
    timeline = _timeline(names, best['lane_service_ns'] if selected else service, set(starts), schedule['cpus'], dispatch)
    predictions = {'SERIAL': serial}
    if selected:
        predictions['TWO_CPU'] = best['prediction']
    by_window = {row['window_starts'][0]: row for row in contexts if len(row['window_starts']) == 1}
    candidates = []
    for window in schedule['windows']:
        first = window['pair_indices'][0]
        gain = serial['estimate_ns'] - by_window[first]['prediction']['estimate_ns']
        candidate = dict(window, gain_ns=gain, selected=first in starts)
        if storage is not None:
            candidate.update(peak_tensor_storage_bytes=storage['window_peak_bytes'][str(first)],
                             within_tensor_budget=first in allowed_starts)
        candidates.append(candidate)
    model = dict(basis='MINIMUM_NONOVERLAPPING_WINDOW_CONTEXT_WITH_MEASURED_OPERATION_COSTS',
        operations=[dict(id=name, service_ns=ns) for name, ns in zip(names, service, strict=True)],
        candidates=candidates, selected_starts=starts, timeline=timeline, empty_dispatch_join_ns=dispatch,
        predictions=predictions, bounds='CALIBRATION_SPAN_NOT_PHYSICAL_BOUNDS',
        shared_memory_contention_modelled=False, numa_locality=numa, joint_numa_context_costs=contexts)
    if storage is not None:
        model['tensor_storage'] = storage
    return model


def _window_model(calibration, recipe, schedule, warmup, *, observe_storage=False, tensor_budget_bytes=None):
    service, dispatch, op_samples, dispatch_samples = _services(calibration, recipe, warmup)
    worker_samples = calibration.get('worker_operation_samples_ns')
    worker_service = None
    if worker_samples is not None:
        positions = {row['pair_indices'][1] for row in schedule['windows']}
        if (type(worker_samples) is not list or len(worker_samples) != warmup or any(
                type(row) is not list or len(row) != len(service) or any(
                    not (type(value) is int and value > 0) if i in positions else value is not None
                    for i, value in enumerate(row)) for row in worker_samples)):
            raise ValueError('exact worker samples required at each legally dispatched position')
        worker_service = [median_low(column) if i in positions else None
                          for i, column in enumerate(zip(*worker_samples, strict=True))]
    elif 'target_cpu_domain' in schedule:
        raise ValueError('broad target planning requires measured worker operation service times')
    eligible = schedule
    storage = None
    if observe_storage:
        input_bytes = calibration['input_storage_bytes']
        if input_bytes != [4 * prod(row['shape']) for row in recipe['inputs']]:
            raise ValueError('standalone input storage bytes differ from the float32 file ABI')
        storage = tensor_footprint(recipe, input_bytes=input_bytes, output_bytes=calibration['result_storage_bytes'],
                                   candidates=schedule['windows'], budget_bytes=tensor_budget_bytes)
        allowed = set(storage['candidate_starts_within_budget'])
        eligible = dict(schedule, windows=[row for row in schedule['windows'] if row['pair_indices'][0] in allowed])

    numa = _numa_context(calibration, recipe, schedule, warmup)
    if numa is not None and numa.get('contexts'):
        return _joint_window_model(recipe, schedule, numa, storage, service, dispatch, op_samples, dispatch_samples)
    replacement_samples = None if numa is None else numa.get('replacement_operation_samples_ns')
    replacements, adjustments = None, {}
    if replacement_samples is not None:
        start = calibration['numa_locality']['window_start']
        if any(window['pair_indices'][0] == start for window in eligible['windows']):
            caller_service = [None if column[0] is None else median_low(column)
                              for column in zip(*replacement_samples, strict=True)]
            adjustment = sum(changed - old for old, changed in zip(service, caller_service, strict=True)
                             if changed is not None)
            adjustments[start] = adjustment
            replacements = dict(window_start=start, caller_service_ns=caller_service, total_adjustment_ns=adjustment)
    starts = select_cpu_windows(eligible, service, dispatch, worker_services=worker_service,
                                pair_cost_adjustments=adjustments)
    choices = set(starts)
    names = [op['id'] for op in recipe['operations']]
    predictions = {'SERIAL': _prediction(sum(service), [sum(row) for row in op_samples])}
    candidates = []
    for row in schedule['windows']:
        first, second = row['pair_indices']
        right = service[second] if worker_service is None else worker_service[second]
        gain = service[first] + service[second] - max(service[first], right) - dispatch - adjustments.get(first, 0)
        candidate = dict(row, gain_ns=gain, selected=first in choices)
        if storage is not None:
            candidate.update(peak_tensor_storage_bytes=storage['window_peak_bytes'][str(first)],
                             within_tensor_budget=first in allowed)
        candidates.append(candidate)

    def serial_or_windows(row, worker, cost):
        return sum(row) + sum(cost + max(row[i], worker[i + 1]) - row[i] - row[i + 1] for i in starts)

    timeline = _timeline(names, _caller_service(service, replacements, choices), choices,
                         schedule['cpus'], dispatch, worker_service)
    if starts:
        sample_workers = op_samples if worker_samples is None else worker_samples
        if replacements is not None and replacements['window_start'] in choices:
            op_samples = [[old if changed is None else changed for old, changed in zip(row, matched, strict=True)]
                          for row, matched in zip(op_samples, replacement_samples, strict=True)]
        estimates = [serial_or_windows(row, worker, cost) for row, worker, cost in
                     zip(op_samples, sample_workers, dispatch_samples, strict=True)]
        predictions['TWO_CPU'] = _prediction(timeline[-1]['end_ns'], estimates)
    model = dict(basis='MINIMUM_ADJACENT_WINDOW_TILING_WITH_ADDITIVE_CALIBRATED_COSTS',
        operations=[dict(id=name, service_ns=ns) for name, ns in zip(names, service, strict=True)],
        candidates=candidates, selected_starts=list(starts), timeline=timeline,
        empty_dispatch_join_ns=dispatch, predictions=predictions,
        bounds='CALIBRATION_SPAN_NOT_PHYSICAL_BOUNDS', shared_memory_contention_modelled=False)
    if storage is not None:
        model['tensor_storage'] = storage
    if worker_service is not None:
        model['worker_service_ns'] = worker_service
    if numa is not None:
        model['numa_locality'] = numa
    if replacements is not None:
        model['numa_consumer_cost'] = replacements
    return model


def _problem(hardware, schedule, origin, evidence, model, *, modes):
    cpus = tuple(schedule['cpus'])
    target, domains = _target(hardware, tuple(schedule['availability_mask']),
                              allow_cpu_subset='target_cpu_domain' in schedule)
    windows = 'windows' in schedule
    key = 'cpu_windows' if windows else 'cpu_pair'
    refs = (dict(artifact=origin, locator='#/' + key), dict(artifact=evidence, locator='#/calibration'))
    options = []
    for mode in modes:
        cost = model['predictions'][mode]
        used = cpus[:1] if mode == 'SERIAL' else cpus
        estimate = CostEstimate(cost['estimate_ns'], cost['sample_min_ns'], cost['sample_max_ns'],
                                'EMPIRICAL_MODEL', 'SCAR_GENERATED_ATEN_CPU_GRAPH_PRIMITIVE_SERVICE', refs)
        if windows:
            profile = dict(window_starts=model['selected_starts'] if mode == 'TWO_CPU' else [])
        else:
            profile = dict(pair=schedule['pair'])
        options.append(ImplementationOption(mode, 'EXISTING_LIBTORCH_ORDERED_GRAPH', refs,
            {r.id: 1 for cpu in used for r in domains[cpu]}, {}, {}, {}, estimate,
            dict(profile, cpus=list(used), source_replacement_authorized=False)))
    problem = PlanningProblem(target, refs, tuple(r for rows in domains.values() for r in rows),
        (PlanningOperation('standalone-aten-graph-call', refs, (), tuple(options), 'SERIAL'),), (),
        assumptions=('SYNCHRONOUS_CPU_ONLY', 'DECLARED_CONCURRENT_OPERATOR_EFFECTS',
                     'RESERVE_CPU_LANES_FOR_GRAPH_CALL', 'MEMORY_CONTENTION_IS_NOT_PREDICTED'))
    plan = construct_implementations(problem)
    verify_execution_plan(problem, plan)
    return problem, plan


def _host_tensor_budget(hardware, budget_bytes, refs):
    memory = [row for row in TargetHardwareGraph(hardware).resources()
              if type(row) is MemoryDomain and row.kind is MemoryKind.HOST]
    if not memory or any(row.capacity_bytes is None for row in memory):
        raise ValueError('target host memory capacity is required for a Tensor storage budget')
    if budget_bytes > sum(row.capacity_bytes for row in memory):
        raise ValueError('Tensor storage budget exceeds recorded total host memory capacity')
    return PlanningResource('host-tensor-storage-budget', 'MEMORY', budget_bytes,
                            (*refs, *(encode(row.evidence) for row in memory),
                             dict(source='EXPLICIT_TENSOR_STORAGE_BUDGET', capacity_bytes=budget_bytes)))


def _segment_plan(hardware, schedule, origin, evidence, model, selected_starts):
    """Reserve the worker core only over selected fork/join intervals."""
    names = [op['id'] for op in model['operations']]
    service = [op['service_ns'] for op in model['operations']]
    choices = set(selected_starts)
    worker_service = model.get('worker_service_ns')
    if choices and 'joint_numa_context_costs' in model:
        context, = (row for row in model['joint_numa_context_costs'] if set(row['window_starts']) == choices)
        service = context['lane_service_ns']
    else:
        service = _caller_service(service, model.get('numa_consumer_cost'), choices)
    timeline = _timeline(names, service, choices, schedule['cpus'], model['empty_dispatch_join_ns'], worker_service)
    storage = model.get('tensor_storage')
    if storage is not None:
        name_to_index = {name: i for i, name in enumerate(names)}
        for segment in timeline:
            first = name_to_index[segment['operations'][0]]
            segment['peak_tensor_storage_bytes'] = (storage['window_peak_bytes'][str(first)]
                if len(segment['operations']) == 2 else storage['operation_peak_bytes'][first])
    sampled = model.get('numa_locality', {}).get('sampled_node_residency_observed')
    return _verify_resource_segments(hardware, schedule, origin, evidence, timeline, storage,
                                     key='cpu_windows', numa_sampled=sampled)


def _verify_resource_segments(hardware, schedule, origin, evidence, timeline, storage, *, key, numa_sampled=False):
    target, domains = _target(hardware, tuple(schedule['availability_mask']),
                              allow_cpu_subset='target_cpu_domain' in schedule)
    refs = (dict(artifact=origin, locator='#/' + key), dict(artifact=evidence, locator='#/calibration'))
    pool = None
    if storage is not None and storage['budget_bytes'] is not None:
        pool = _host_tensor_budget(hardware, storage['budget_bytes'], refs)
    tasks = []
    for i, segment in enumerate(timeline):
        mode = 'FORK_JOIN' if len(segment['operations']) > 1 else 'ORDERED'
        duration = segment['end_ns'] - segment['start_ns']
        measurement = segment.get('duration_basis')
        sources = {'MEASURED_CONCURRENT_FORK_JOIN': 'concurrent_team',
                   'MEASURED_POST_JOIN_CONSUMER': 'post_join_team'}
        if measurement in sources:
            cost_refs = (*refs, dict(artifact=evidence, locator='#/calibration/' + sources[measurement]))
            basis = 'ATEN_' + measurement
        else:
            cost_refs, basis = refs, 'ATEN_SEGMENT_SERVICE_AND_HANDOFF'
        cost = CostEstimate(duration, None, None, 'EMPIRICAL_MODEL', basis, cost_refs)
        demands = {resource.id: 1 for cpu in segment['cpu_ids'] for resource in domains[cpu]}
        workspace = {}
        if pool is not None:
            workspace[pool.id] = segment['peak_tensor_storage_bytes'] - storage['persistent_input_bytes']
        identifier = 'aten-segment-' + str(i)
        option = ImplementationOption(mode, 'EXISTING_LIBTORCH_FIXED_CPU_LANES', refs, demands,
            workspace, {}, {}, cost, dict(operations=segment['operations'], cpu_ids=segment['cpu_ids']))
        tasks.append(PlanningOperation(identifier, refs, (tasks[-1].id,) if tasks else (), (option,), mode))
    resources = tuple(r for rows in domains.values() for r in rows)
    barrier = 'SEGMENT_BARRIER_MAKES_ALL_OUTPUTS_READY' if key == 'cpu_team' else 'SEGMENT_BARRIER_MAKES_BOTH_OUTPUTS_READY'
    assumptions = (barrier, 'DECLARED_CONCURRENT_OPERATOR_EFFECTS',
                   'INTERNAL_FORK_JOIN_TIME_NOT_SPLIT', 'MEMORY_CONTENTION_IS_NOT_PREDICTED')
    reserved = {}
    if pool is not None:
        resources = (*resources, pool)
        reserved[pool.id] = storage['persistent_input_bytes']
        residency = 'NUMA_PAGE_RESIDENCY_SAMPLED_IN_CALIBRATION' if numa_sampled else 'NUMA_PAGE_PLACEMENT_UNOBSERVED'
        assumptions = (*assumptions, 'TENSOR_HANDLE_STORAGE_BUDGET_EXCLUDES_PROVIDER_INTERNAL_WORKSPACE', residency)
    problem = PlanningProblem(target, refs, resources, tuple(tasks), (), assumptions=assumptions, reserved_memory=reserved)
    plan = construct_implementations(problem)
    verify_execution_plan(problem, plan)
    actual = {row['operation']: (row['start_ns'], row['end_ns']) for row in plan['selected']}
    if actual != {task.id: (segment['start_ns'], segment['end_ns']) for task, segment in zip(tasks, timeline, strict=True)}:
        raise ValueError('verified resource plan differs from generated CPU window order')
    return problem, plan, timeline


def _statistics(raw, warmup, repetitions, cpus):
    if ([raw['caller_cpu'], raw['worker_cpu']] != cpus or raw['warmup_pairs'] != warmup
            or raw['measured_pairs'] != repetitions or raw['all_pairs_validated'] is not True):
        raise ValueError('native observations differ from the requested graph trial')
    metrics, wins, by_order, ratio, faster = _compare_measured_pairs(raw['samples'], repetitions)
    return dict(metrics=metrics, wins_two_cpu=wins, paired_order_differences=by_order,
                speedup_two_cpu=ratio, profitable_two_cpu=faster)


def _compare_measured_pairs(rows, repetitions):
    samples, differences, directional = _measurements(rows, repetitions)
    metrics = {mode: dict(_metrics(values), variance_ns2=variance(values)) for mode, values in samples.items()}
    serial, dual = (metrics[m]['median_ns'] for m in ('SERIAL', 'TWO_CPU'))
    wins = sum(value > 0 for value in differences)
    by_order = {order[0]: _metrics(values) for order, values in directional.items()}
    faster = ((serial - dual) / serial >= _GAIN and 4 * wins >= 3 * repetitions
              and all(row['median_ns_twice'] > 0 for row in by_order.values()))
    return metrics, wins, by_order, serial / dual, faster


def _prefix(cpus):
    return ['env', '-i', *(f'{k}={v}' for k, v in _ENV.items()), 'taskset', '-c', ','.join(map(str, cpus))]


def _launcher(stage, executable, selection, schedule, inputs, guarded, choice, *, replica=None, operation_order=None):
    # The result is qualified for exactly the tested input bytes and linked primary images.
    checks = ['test "$(sha256sum < ' + shlex.quote(row['path']) + ' | cut -d " " -f 1)" = '
              + row['sha256'] + ' || { echo "SCAR native graph member digest differs" >&2; exit 78; }'
              for row in guarded]
    dual = selection != 'SERIAL'
    windowed = 'windows' in schedule
    team = 'legal_window_widths' in schedule
    mode = '--run-serial'
    if dual:
        mode = '--run-team' if team else '--run-windows' if windowed else '--run-pair'
    cpus = schedule['cpus'] if dual else schedule['cpus'][:1]
    spec = []
    if selection in ('READY_DAG', 'READY_DAG_REPLICA'):
        mode = '--run-ready' if replica is None else '--run-ready-replica'
        spec = [''.join(map(str, choice))]
        if replica is not None:
            spec.extend(str(replica[k]) for k in ('input_index', 'lane'))
        if operation_order is not None:
            if replica is not None:
                raise ValueError('native ordered launch does not admit a replica')
            mode = '--run-ready-ordered'
            spec.append(','.join(map(str, operation_order)))
    elif dual and team:
        from scar.compiler.plan_aten_native_team import team_window_spec

        spec = [team_window_spec(choice)]
    elif dual and windowed:
        spec = [','.join(map(str, choice))]
    command = [*_prefix(cpus), str(executable), mode, *map(str, cpus), *spec, *(row['path'] for row in inputs)]
    launch = stage / 'launch.sh'
    launch.write_text('#!/bin/sh\nset -eu\n' + '\n'.join(checks) + '\nexec ' + shlex.join(command) + ' "$@"\n')
    launch.chmod(0o755)


def _invoke(executable, schedule, mode, args, inputs):
    command = [*_prefix(schedule['availability_mask']), str(executable), mode, *map(str, schedule['cpus']),
               *map(str, args), *(row['path'] for row in inputs)]
    proc = subprocess.run(command, capture_output=True, check=True)
    if proc.stderr:
        raise ValueError('generated graph trial emitted diagnostics: ' + proc.stderr.decode(errors='replace').strip())
    return loads(proc.stdout)


def plan_aten_native_graph(bundle, *, bundle_sha256, input_paths, out, warmup_pairs=7, measured_pairs=31,
                          tensor_budget_bytes=None, primitive_observation=None):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    if not 3 <= warmup_pairs <= 100 or not 3 <= measured_pairs <= 501 or measured_pairs % 2 == 0:
        raise ValueError('CPU graph trial requires 3–100 warmups and 3–501 odd measurement pairs')
    artifact, origin, recipe, hardware, executable, key = _bundle(bundle, bundle_sha256)
    schedule = artifact[key]
    windowed = key == 'cpu_windows'
    observe_storage = windowed and artifact.get('tensor_lifetime') is not None
    if tensor_budget_bytes is not None and (type(tensor_budget_bytes) is not int or not 0 < tensor_budget_bytes < 2**63):
        raise ValueError('a positive bounded Tensor storage budget is required')
    if key == 'cpu_team':
        from scar.compiler.plan_aten_native_team import plan_aten_native_team

        return plan_aten_native_team((artifact, origin, recipe, hardware, executable, key), input_paths=input_paths,
            out=out, warmup_pairs=warmup_pairs, measured_pairs=measured_pairs, tensor_budget_bytes=tensor_budget_bytes,
            primitive_observation=primitive_observation)
    if tensor_budget_bytes is not None and not observe_storage:
        raise ValueError('a graph rebuilt with the Tensor lifetime and storage calibration ABI is required')
    inputs = [_receipt(p) for p in input_paths]
    if len(inputs) != len(recipe['inputs']):
        raise ValueError('provide the native input files in their recipe order')
    binary = _receipt(executable)
    guarded = [binary, *artifact['provider']['libraries'], *inputs]
    if windowed:
        if primitive_observation is None:
            raw = _invoke(executable, schedule, '--calibrate-windows', [warmup_pairs], inputs)
        else:
            if primitive_observation['inputs'] != inputs:
                raise ValueError('CPU domain calibration input bytes differ from the selected delivery')
            raw = primitive_observation['raw']
        if ([raw['caller_cpu'], raw['worker_cpu']] != schedule['cpus'] or raw['warmup_pairs'] != warmup_pairs):
            raise ValueError('native calibration differs from the requested graph and placement')
        model = _window_model(raw['calibration'], recipe, schedule, warmup_pairs,
                              observe_storage=observe_storage, tensor_budget_bytes=tensor_budget_bytes)
        stats, matched = None, None
    else:
        if primitive_observation is not None:
            raise ValueError('imported primitive observations require a generated CPU window graph')
        raw = _invoke(executable, schedule, '--benchmark-pair', [warmup_pairs, measured_pairs], inputs)
        model = _component_model(raw['calibration'], recipe, schedule, warmup_pairs)
        stats, matched = _statistics(raw, warmup_pairs, measured_pairs, schedule['cpus']), None

    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-aten-schedule-', dir=out.parent) as temp:
        stage = Path(temp)
        observations = stage / 'observations.json'
        _write(observations, dict(raw, schema='scar.aten-generated-graph-observations', schema_version=1))
        evidence = artifact_reference(observations, 'scar.aten-generated-graph-observations')
        evidence['uri'] = str(out / observations.name)
        options = tuple(model['predictions'])
        problem, plan = _problem(hardware, schedule, origin, evidence, model, modes=options)
        predicted, = (row['option'] for row in plan['selected'])
        reasons = []
        costs = model['predictions']
        starts = model['selected_starts'] if windowed else [schedule['pair_indices'][0]]
        dual_cost = costs.get('TWO_CPU')
        modeled_gain = 0.0 if dual_cost is None else 1 - dual_cost['estimate_ns'] / costs['SERIAL']['estimate_ns']
        if predicted == 'TWO_CPU' and modeled_gain < _GAIN:
            reasons.append('PREDICTED_GAIN_BELOW_THRESHOLD')
        if predicted == 'TWO_CPU' and not reasons:
            if windowed:
                spec = ','.join(map(str, starts))
                matched = _invoke(executable, schedule, '--benchmark-windows',
                                  [spec, warmup_pairs, measured_pairs], inputs)
                if matched.get('windows') != starts:
                    raise ValueError('native paired validation did not run the selected windows')
                stats = _statistics(matched, warmup_pairs, measured_pairs, schedule['cpus'])
                if observe_storage and matched.get('result_storage_bytes') != raw['calibration']['result_storage_bytes']:
                    reasons.append('TWO_CPU_STORAGE_BYTES_DIFFER_FROM_SERIAL_CALIBRATION')
            if not stats['profitable_two_cpu']:
                reasons.append('TWO_CPU_FAILED_MATCHED_VALIDATION')
        selected = 'SERIAL' if reasons else predicted
        if not windowed and selected != predicted:
            problem, plan = _problem(hardware, schedule, origin, evidence, model, modes=('SERIAL',))
        if any(_receipt(row['path']) != row for row in guarded):
            raise ValueError('native graph members changed during the trial')
        matched_ref = None
        if matched is not None:
            path = stage / 'paired-validation.json'
            _write(path, dict(matched, schema='scar.aten-generated-graph-paired-validation', schema_version=1))
            matched_ref = artifact_reference(path, 'scar.aten-generated-graph-paired-validation')
            matched_ref['uri'] = str(out / path.name)
        selected_starts = starts if selected == 'TWO_CPU' else []
        ids = ([recipe['operations'][j]['id'] for i in selected_starts for j in (i, i + 1)]
               if windowed else schedule['pair'])
        report = dict(schema='scar.aten-native-graph-schedule', schema_version=1,
            boundary='GENERATED_WARM_ATEN_GRAPH_CALL', bundle=origin, executable=binary,
            inputs=inputs, observations=evidence, model=model, original_target=schedule['target'],
            operation_ids=ids, cpu_ids=schedule['cpus'], model_preferred=predicted,
            selected=selected, reasons=reasons, predicted_gain_two_cpu=modeled_gain,
            validation=stats, warmup_pairs=warmup_pairs, measured_pairs=0 if stats is None else measured_pairs,
            runtime_environment=_ENV, source_replacement_authorized=False,
            cold_launcher_performance_measured=False,
            planning_domain=('NONOVERLAPPING_DECLARED_ADJACENT_PAIRS_ON_TWO_RECORDED_CORES' if windowed
                             else 'ONE_DECLARED_ADJACENT_PAIR_ON_TWO_RECORDED_CORES'))
        if windowed:
            problem, plan, timeline = _segment_plan(hardware, schedule, origin, evidence, model, selected_starts)
            report.update(selected_window_starts=selected_starts, paired_validation=matched_ref,
                          whole_call_calibration_used=False, execution_timeline=timeline)
        _write(stage / 'selection.json', report)
        _write(stage / 'problem.json', problem_document(problem))
        _write(stage / 'plan.json', plan)
        _launcher(stage, executable, selected, schedule, inputs, guarded, selected_starts)
        stage.rename(out)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar plan-aten-graph', description=__doc__)
    parser.add_argument('--bundle', required=True, help='CPU concurrency graph artifact.json')
    parser.add_argument('--bundle-sha256', required=True)
    parser.add_argument('--input', action='append', required=True, help='input float32 files in recipe order')
    parser.add_argument('--warmup-pairs', type=int, default=7)
    parser.add_argument('--measured-pairs', type=int, default=31)
    parser.add_argument('--tensor-budget-bytes', type=int,
                        help='capacity quota for graph-held CPU Tensor storage, excluding provider-internal workspace')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    report = plan_aten_native_graph(args.bundle, bundle_sha256=args.bundle_sha256, input_paths=args.input,
                                    out=args.out, warmup_pairs=args.warmup_pairs, measured_pairs=args.measured_pairs,
                                    tensor_budget_bytes=args.tensor_budget_bytes)
    speedup_key = 'speedup_parallel' if 'selected_windows' in report else 'speedup_two_cpu'
    print(json.dumps(dict(model_preferred=report['model_preferred'], selected=report['selected'],
                          launcher=str(Path(args.out) / 'launch.sh'),
                          **{'validation_' + speedup_key: (report['validation'] or {}).get(speedup_key)})))
    return 0
