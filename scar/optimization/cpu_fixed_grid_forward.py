"""Attach calibrated owner-window service to the counted fixed-grid graph."""

import argparse
from fractions import Fraction
from pathlib import Path

from scar.analysis import cpu_fixed_grid_timing_graph as timing_graph
from scar.codegen.cpu_native_route_package import _fixed_grid_route_replay
from scar.compiler import observe_cpu_fixed_grid_native_lifecycle as native_probe
from scar.compiler import observe_cpu_fixed_grid_python_lifecycle as python_probe
from scar.compiler import cpu_fixed_grid_source_route_benchmark as benchmark
from scar.compiler.calibrate_array_entry import _ref, _write
from scar.ir.execution_recipe import _canonical, read_artifact
from scar.optimization.array_live_regions import WINDOWS, fit_affine_lifecycle
from scar.optimization.paired_entry_selection import DEFAULT_MINIMUM_GAIN


SCHEMA = 'scar.cpu-fixed-grid-forward'
_ORIGINAL, _MAPPED = 'ORIGINAL_PYTHON', 'PROVIDER_MAP'


def _need(condition, reason):
    if not condition:
        raise ValueError('fixed-grid forward model: ' + reason)


def _fraction(value):
    return Fraction(value['numerator'], value['denominator'])


def _wire(value):
    return dict(numerator=value.numerator, denominator=value.denominator)


def _training(observation):
    return {int(count): {phase: [row['windows_ns'][phase] for row in entry['samples']] for phase in WINDOWS}
            for count, entry in observation['cases'].items()}


def _schedule(route, model, steps):
    prediction = model['prediction_ns']
    clocks = dict.fromkeys(('lower', 'point', 'upper'), Fraction(0))
    scheduled = []
    for name in WINDOWS:
        group = route['phases'][name]
        start = {role: None if value is None else _wire(value) for role, value in clocks.items()}
        duration = prediction[name]
        for role in clocks:
            bound = duration[role]
            clocks[role] = None if clocks[role] is None or bound is None else clocks[role] + _fraction(bound)
        end = {role: None if value is None else _wire(value) for role, value in clocks.items()}
        scheduled.append(dict(phase=name, operations=group['operations'], resources=group['resource_ids'],
                              repetitions=steps if name == 'recurrence' else 1, start_ns=start, end_ns=end,
                              duration_ns=duration))
    _need(scheduled[-1]['end_ns'] == prediction['total'], 'phase schedule disagrees with the fitted entry duration')
    return scheduled


def _compare(model, measured):
    total = model['prediction_ns']['total']
    point = None if total['point'] is None else _fraction(total['point'])
    median = Fraction(measured['median_ns_twice'], 2)
    low = None if total['lower'] is None else _fraction(total['lower'])
    high = None if total['upper'] is None else _fraction(total['upper'])
    return dict(measured=measured, prediction_ns=total,
                median_inside_prediction=low is not None and low <= median <= high,
                observed_range_inside_prediction=low is not None and low <= measured['min_ns'] <= measured['max_ns'] <= high,
                observed_median_minus_point_ns=None if point is None else _wire(median - point))


@_fixed_grid_route_replay()
def derive_cpu_fixed_grid_forward(graph_ref, python_ref, native_ref, holdout_ref):
    graph = timing_graph.verify_cpu_fixed_grid_timing_graph(graph_ref)
    native = native_probe.verify_short_native(native_ref)
    _need(native['python_probe'] == python_ref, 'mapped calibration must verify this exact Python oracle')
    original = read_artifact(python_ref)
    package = read_artifact(original['package'])
    scope = graph['scope']
    steps = scope['steps']
    _need(graph['routes'].keys() == {_ORIGINAL, _MAPPED} and native['package'] == original['package']
          and all(scope[key] == original[key] == native[key] == package[key]
                  for key in ('cpu_ids', 'working_directory', 'environment_sha256'))
          and scope['argv'] == package['argv'] and steps == package['steps']
          and graph['source'] == original['source'] == native['source'] == package['source']
          and graph['target'] == original['target'] == native['target'] == package['hardware']
          and native['step_counts'] == original['step_counts'] == list(python_probe.STEP_COUNTS)
          and steps not in original['step_counts'] and all(observation['clean_benchmark'] is False
                                                         for observation in (original, native)),
          'short probes, admitted graph and mapped package have different execution domains')
    input_graph = read_artifact(graph['input_graph'])
    lifecycle = read_artifact(graph['lifecycle'])
    _need(package['selected_runtime'] == _MAPPED and input_graph['mapped_plan'] == package['plan']
          and package['owned_copy_delivery'] == lifecycle['delivery']
          and package['selected_project'] == input_graph['mapped_project'], 'timing graph must name this exact mapped route')

    routes = {}
    for name, observation in ((_ORIGINAL, original), (_MAPPED, native)):
        model = fit_affine_lifecycle(_training(observation), predict_steps=steps)
        routes[name] = dict(model=model, schedule=_schedule(graph['routes'][name], model, steps))

    # The clean count is read only after fitting both routes from short windows.
    holdout = benchmark.verify_cpu_fixed_grid_source_route_benchmark(holdout_ref)
    _need(holdout['package'] == original['package'] and holdout['steps'] == steps
          and holdout['native_project'] == package['selected_project'] and holdout['physical_time_bound'] is False,
          'clean holdout differs from the planned full-process scope')
    metrics = holdout['selection']['metrics']
    for name, row in routes.items():
        row['holdout'] = _compare(row['model'], metrics[name])
    base = routes[_ORIGINAL]['model']['prediction_ns']['total']['lower']
    mapped = routes[_MAPPED]['model']['prediction_ns']['total']['upper']
    gain = (None if base is None or mapped is None or _fraction(base) <= 0 else
            1 - _fraction(mapped) / _fraction(base))
    threshold = Fraction(str(DEFAULT_MINIMUM_GAIN))
    coverage = all(row['holdout']['observed_range_inside_prediction'] for row in routes.values())
    advance = coverage and gain is not None and gain > threshold
    return dict(schema=SCHEMA, schema_version=1, graph=graph_ref, python_probe=python_ref, native_probe=native_ref,
                clean_holdout=holdout_ref, package=original['package'], source=graph['source'], target=graph['target'],
                scope=scope, routes=routes, modeled_steps=steps, phase_order=list(WINDOWS),
                model_kind='SERIAL_OWNER_BOUNDARY_WINDOWS_WITH_COUNTED_AFFINE_RECURRENCE',
                status=('CONDITIONAL_ADVANTAGE_HOLDOUT_COVERED' if advance else
                        'CONDITIONAL_MODEL_NEEDS_RECALIBRATION' if not coverage else 'NO_SEPARATED_CONDITIONAL_GAIN'),
                conditional_lower_gain=None if gain is None else _wire(gain),
                strict_minimum_gain=DEFAULT_MINIMUM_GAIN, clean_holdout_used_for_fit=False,
                holdout_is_separate_run_set=True,
                physical_entry_duration_bound_proven=False, optimization_or_deployment_authority=False,
                execution_default='KEEP_ORIGINAL', target_executed=False)


def verify_cpu_fixed_grid_forward(reference):
    saved = read_artifact(reference)
    _need(reference['schema'] == SCHEMA and type(saved['schema_version']) is int and saved['schema_version'] == 1,
          'versioned fixed-grid forward model required')
    expected = derive_cpu_fixed_grid_forward(saved['graph'], saved['python_probe'], saved['native_probe'], saved['clean_holdout'])
    _need(_canonical(saved) == _canonical(expected), 'fixed-grid forward model differs at cold replay')
    return expected


def main(argv=None):
    from scar.analysis.cpu_fixed_grid_input_route_graph import SCHEMA as INPUT_GRAPH
    from scar.analysis.cpu_fixed_grid_lifecycle import SCHEMA as LIFECYCLE
    from scar.codegen.cpu_native_route_package import FIXED_GRID_SCHEMA as PACKAGE

    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_subparsers(dest='action', required=True)
    graph = action.add_parser('graph')
    for field in ('lifecycle', 'input-graph', 'out'):
        graph.add_argument('--' + field, type=Path, required=True)
    for name in ('probe-python', 'probe-native'):
        probe = action.add_parser(name)
        probe.add_argument('--package', type=Path, required=True)
        probe.add_argument('--out', type=Path, required=True)
        probe.add_argument('--samples', type=int, default=3)
        if name == 'probe-native':
            probe.add_argument('--python', type=Path, required=True)
            probe.add_argument('--counters', type=Path)
    plan = action.add_parser('plan')
    for field in ('graph', 'python', 'native', 'holdout', 'out'):
        plan.add_argument('--' + field, type=Path, required=True)
    check = action.add_parser('check')
    check.add_argument('model', type=Path)
    args = parser.parse_args(argv)
    if args.action == 'check':
        print(verify_cpu_fixed_grid_forward(_ref(args.model, SCHEMA))['status'])
    else:
        if args.action == 'graph':
            result = timing_graph.derive_cpu_fixed_grid_timing_graph(_ref(args.lifecycle, LIFECYCLE),
                                                                      _ref(args.input_graph, INPUT_GRAPH))
        elif args.action == 'probe-python':
            python_probe.observe_short_python(_ref(args.package, PACKAGE), args.out, samples=args.samples)
            print(args.out / 'result.json')
            return 0
        elif args.action == 'probe-native':
            counters = None if args.counters is None else _ref(args.counters, native_probe.COUNTERS_SCHEMA)
            native_probe.observe_short_native(_ref(args.package, PACKAGE), _ref(args.python, python_probe.SHORT_SCHEMA),
                                              args.out, samples=args.samples, counters_ref=counters)
            print(args.out / 'result.json')
            return 0
        else:
            result = derive_cpu_fixed_grid_forward(_ref(args.graph, timing_graph.SCHEMA), _ref(args.python, python_probe.SHORT_SCHEMA),
                        _ref(args.native, native_probe.SHORT_SCHEMA), _ref(args.holdout, benchmark.SCHEMA))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        _write(args.out, result)
        print(args.out)
    return 0
