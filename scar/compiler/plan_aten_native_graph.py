"""Predict and verify a standalone ATen CPU graph's caller/worker schedule."""

import argparse
import json
from pathlib import Path
import shlex
from statistics import median_low, variance
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen.aten_native_graph import _read_declared
from scar.codegen.native_provider_bundle import _content
from scar.compiler.plan_aten_boundary import _checked, _target, _write
from scar.ir.execution_recipe import artifact_reference
from scar.ir.implementation_plan import (
    CostEstimate, ImplementationOption, PlanningOperation, PlanningProblem, problem_document,
)
from scar.ir.record_codec import loads
from scar.optimization.aten_boundary_trials import _measurements
from scar.optimization.aten_native_schedule import propose_cpu_pair
from scar.optimization.constructive_planner import construct_implementations
from scar.optimization.joint_planner import verify_execution_plan
from scar.optimization.paired_entry_selection import _metrics


_EXE = 'aten-native-graph'
_ENV = {'PATH': '/usr/bin:/bin', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
_GAIN = 0.05


def _receipt(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), **_content(path))


def _bundle(path, digest):
    path, artifact, receipt = _read_declared(path, digest)
    if (artifact['schema'], artifact['schema_version'], artifact['implementation']) != (
            'scar.aten-native-graph-delivery', 1, 'ORDERED_DEFAULT_WITH_DECLARED_CPU_PAIR_TRIAL'):
        raise ValueError('generated CPU-pair graph bundle required')
    recipe_path = path.with_name('recipe.json')
    _, recipe, _ = _read_declared(recipe_path, artifact['files']['recipe.json']['sha256'])
    executable = path.with_name(_EXE)
    if _content(executable) != artifact['files'][_EXE]:
        raise ValueError('generated native graph executable differs from the bundle receipt')
    contract = artifact['concurrency_contract']
    _, requirements, _ = _read_declared(contract['path'], contract['sha256'])
    schedule = artifact['cpu_pair']
    hardware = Path(schedule['target']['uri']).parent
    replayed = propose_cpu_pair(recipe, contract=requirements, target=hardware, cpu_ids=schedule['cpus'])
    if replayed != schedule:
        raise ValueError('retained CPU placement differs from the recipe, contract or target hardware graph')
    for library in artifact['provider']['libraries']:
        _checked(library)
    origin = dict(uri=receipt['path'], sha256=receipt['sha256'], size_bytes=receipt['size_bytes'],
                  schema=artifact['schema'])
    return artifact, origin, recipe, hardware, executable


def _component_model(calibration, recipe, schedule, warmup):
    op_samples = calibration['operation_samples_ns']
    dispatch_samples = calibration['empty_dispatch_samples_ns']
    count = len(recipe['operations'])
    first, second = schedule['pair_indices']
    if (len(op_samples) != warmup or len(dispatch_samples) != warmup
            or any(len(row) != count for row in op_samples)
            or any(type(n) is not int or n <= 0 for row in [*op_samples, dispatch_samples] for n in row)):
        raise ValueError('one exact service sample vector and dispatch observation per calibration row required')

    def totals(row, dispatch):
        serial = sum(row)
        return serial, serial - min(row[first], row[second]) + dispatch

    service = [median_low(samples) for samples in zip(*op_samples, strict=True)]
    dispatch = median_low(dispatch_samples)
    estimates = totals(service, dispatch)
    by_mode = {}
    sample_totals = [totals(row, cost) for row, cost in zip(op_samples, dispatch_samples, strict=True)]
    samples = zip(*sample_totals, strict=True)
    for mode, estimate, values in zip(('SERIAL', 'TWO_CPU'), estimates, samples, strict=True):
        by_mode[mode] = dict(estimate_ns=estimate, sample_min_ns=min(estimate, *values),
                             sample_max_ns=max(estimate, *values))
    return dict(basis='SUM_SERIAL_PREFIX_AND_SUFFIX_WITH_MAX_OF_READY_PARALLEL_PAIR',
                operations=[dict(id=op['id'], service_ns=ns) for op, ns in zip(recipe['operations'], service, strict=True)],
                pair=schedule['pair'], empty_dispatch_join_ns=dispatch, predictions=by_mode,
                bounds='CALIBRATION_SPAN_NOT_PHYSICAL_BOUNDS', shared_memory_contention_modelled=False)


def _problem(hardware, schedule, origin, evidence, model, *, modes):
    cpus = tuple(schedule['cpus'])
    target, domains = _target(hardware, tuple(schedule['availability_mask']))
    refs = (dict(artifact=origin, locator='#/cpu_pair'), dict(artifact=evidence, locator='#/calibration'))
    options = []
    for mode in modes:
        cost = model['predictions'][mode]
        used = cpus[:1] if mode == 'SERIAL' else cpus
        estimate = CostEstimate(cost['estimate_ns'], cost['sample_min_ns'], cost['sample_max_ns'],
                                'EMPIRICAL_MODEL', 'SCAR_GENERATED_ATEN_CPU_GRAPH_PRIMITIVE_SERVICE', refs)
        options.append(ImplementationOption(mode, 'EXISTING_LIBTORCH_ORDERED_GRAPH', refs,
            {r.id: 1 for cpu in used for r in domains[cpu]}, {}, {}, {}, estimate,
            dict(cpus=list(used), pair=schedule['pair'], source_replacement_authorized=False)))
    problem = PlanningProblem(target, refs, tuple(r for rows in domains.values() for r in rows),
        (PlanningOperation('standalone-aten-graph-call', refs, (), tuple(options), 'SERIAL'),), (),
        assumptions=('SYNCHRONOUS_CPU_ONLY', 'DECLARED_CONCURRENT_OPERATOR_EFFECTS',
                     'RESERVE_CPU_LANES_FOR_GRAPH_CALL', 'MEMORY_CONTENTION_IS_NOT_PREDICTED'))
    plan = construct_implementations(problem)
    verify_execution_plan(problem, plan)
    return problem, plan


def _statistics(raw, warmup, repetitions, cpus):
    if ([raw['caller_cpu'], raw['worker_cpu']] != cpus or raw['warmup_pairs'] != warmup
            or raw['measured_pairs'] != repetitions or raw['all_pairs_validated'] is not True):
        raise ValueError('native observations differ from the requested graph trial')
    samples, differences, directional = _measurements(raw['samples'], repetitions)
    metrics = {mode: dict(_metrics(values), variance_ns2=variance(values)) for mode, values in samples.items()}
    serial, dual = (metrics[m]['median_ns'] for m in ('SERIAL', 'TWO_CPU'))
    wins = sum(value > 0 for value in differences)
    by_order = {order[0]: _metrics(values) for order, values in directional.items()}
    faster = ((serial - dual) / serial >= _GAIN and 4 * wins >= 3 * repetitions
              and all(row['median_ns_twice'] > 0 for row in by_order.values()))
    return dict(metrics=metrics, wins_two_cpu=wins, paired_order_differences=by_order,
                speedup_two_cpu=serial / dual, profitable_two_cpu=faster)


def _prefix(cpus):
    return ['env', '-i', *(f'{k}={v}' for k, v in _ENV.items()), 'taskset', '-c', ','.join(map(str, cpus))]


def _launcher(stage, executable, selection, schedule, inputs, guarded):
    # The result is qualified for exactly the tested input bytes and linked primary images.
    checks = ['test "$(sha256sum < ' + shlex.quote(row['path']) + ' | cut -d " " -f 1)" = '
              + row['sha256'] + ' || { echo "SCAR native graph member digest differs" >&2; exit 78; }'
              for row in guarded]
    dual = selection == 'TWO_CPU'
    mode = '--run-pair' if dual else '--run-serial'
    cpus = schedule['cpus'] if dual else schedule['cpus'][:1]
    command = [*_prefix(cpus), str(executable), mode, *map(str, cpus), *(row['path'] for row in inputs)]
    launch = stage / 'launch.sh'
    launch.write_text('#!/bin/sh\nset -eu\n' + '\n'.join(checks) + '\nexec ' + shlex.join(command) + ' "$@"\n')
    launch.chmod(0o755)


def plan_aten_native_graph(bundle, *, bundle_sha256, input_paths, out, warmup_pairs=7, measured_pairs=31):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    if not 3 <= warmup_pairs <= 100 or not 3 <= measured_pairs <= 501 or measured_pairs % 2 == 0:
        raise ValueError('CPU graph trial requires 3–100 warmups and 3–501 odd measurement pairs')
    artifact, origin, recipe, hardware, executable = _bundle(bundle, bundle_sha256)
    schedule = artifact['cpu_pair']
    inputs = [_receipt(p) for p in input_paths]
    if len(inputs) != len(recipe['inputs']):
        raise ValueError('provide the native input files in their recipe order')
    binary = _receipt(executable)
    guarded = [binary, *artifact['provider']['libraries'], *inputs]
    prefix = _prefix(schedule['availability_mask'])
    command = [*prefix, str(executable), '--benchmark-pair', *map(str, schedule['cpus']),
               str(warmup_pairs), str(measured_pairs), *(row['path'] for row in inputs)]
    process = subprocess.run(command, capture_output=True, check=True)
    if process.stderr or any(_receipt(row['path']) != row for row in guarded):
        raise ValueError('diagnostics or changed graph members during the native trial')
    raw = loads(process.stdout)
    model = _component_model(raw['calibration'], recipe, schedule, warmup_pairs)
    statistics = _statistics(raw, warmup_pairs, measured_pairs, schedule['cpus'])

    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-aten-schedule-', dir=out.parent) as temp:
        stage = Path(temp)
        observations = stage / 'observations.json'
        _write(observations, dict(raw, schema='scar.aten-generated-graph-observations', schema_version=1))
        evidence = artifact_reference(observations, 'scar.aten-generated-graph-observations')
        evidence['uri'] = str(out / observations.name)
        options = ('SERIAL', 'TWO_CPU')
        problem, plan = _problem(hardware, schedule, origin, evidence, model, modes=options)
        predicted, = (row['option'] for row in plan['selected'])
        reasons = []
        costs = model['predictions']
        modeled_gain = 1 - costs['TWO_CPU']['estimate_ns'] / costs['SERIAL']['estimate_ns']
        if predicted == 'TWO_CPU' and modeled_gain < _GAIN:
            reasons.append('PREDICTED_GAIN_BELOW_THRESHOLD')
        if predicted == 'TWO_CPU' and not statistics['profitable_two_cpu']:
            reasons.append('TWO_CPU_FAILED_MATCHED_VALIDATION')
        selected = 'SERIAL' if reasons else predicted
        if selected != predicted:
            problem, plan = _problem(hardware, schedule, origin, evidence, model, modes=('SERIAL',))
        report = dict(schema='scar.aten-native-graph-schedule', schema_version=1,
            boundary='GENERATED_WARM_ATEN_GRAPH_CALL', bundle=origin, executable=binary,
            inputs=inputs, observations=evidence, model=model, original_target=schedule['target'],
            operation_ids=schedule['pair'], cpu_ids=schedule['cpus'], model_preferred=predicted,
            selected=selected, reasons=reasons, predicted_gain_two_cpu=modeled_gain,
            validation=statistics, warmup_pairs=warmup_pairs, measured_pairs=measured_pairs,
            runtime_environment=_ENV, source_replacement_authorized=False,
            cold_launcher_performance_measured=False,
            planning_domain='ONE_DECLARED_ADJACENT_PAIR_ON_TWO_RECORDED_CORES')
        _write(stage / 'selection.json', report)
        _write(stage / 'problem.json', problem_document(problem))
        _write(stage / 'plan.json', plan)
        _launcher(stage, executable, selected, schedule, inputs, guarded)
        stage.rename(out)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar plan-aten-graph', description=__doc__)
    parser.add_argument('--bundle', required=True, help='CPU-pair graph artifact.json')
    parser.add_argument('--bundle-sha256', required=True)
    parser.add_argument('--input', action='append', required=True, help='input float32 files in recipe order')
    parser.add_argument('--warmup-pairs', type=int, default=7)
    parser.add_argument('--measured-pairs', type=int, default=31)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    report = plan_aten_native_graph(args.bundle, bundle_sha256=args.bundle_sha256, input_paths=args.input,
                                    out=args.out, warmup_pairs=args.warmup_pairs, measured_pairs=args.measured_pairs)
    print(json.dumps(dict(model_preferred=report['model_preferred'], selected=report['selected'],
                          validation_speedup_two_cpu=report['validation']['speedup_two_cpu'],
                          launcher=str(Path(args.out) / 'launch.sh'))))
    return 0
