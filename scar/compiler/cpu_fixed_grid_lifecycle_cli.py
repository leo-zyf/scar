"""Build and replay the source-bound CPU fixed-grid lifecycle receipts."""

import argparse
import json
from pathlib import Path

from scar.compiler.calibrate_array_entry import _ref, _write


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar fixed-grid-lifecycle', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    graph = commands.add_parser('graph', help='connect both exact source entry lifecycles')
    for option in ('delivery', 'effects', 'resources', 'out'):
        graph.add_argument('--' + option, type=Path, required=True)
    native = commands.add_parser('native', help='observe phases in a separate C++20 build')
    native.add_argument('--delivery', type=Path, required=True)
    native.add_argument('--out', type=Path, required=True)
    native.add_argument('--samples', type=int, default=3)
    native.add_argument('--compiler', default='g++')
    plan = commands.add_parser('plan', help='join verified cost evidence to phase and operation IDs')
    for option in ('review', 'lifecycle', 'out'):
        plan.add_argument('--' + option, type=Path, required=True)
    for option in ('python', 'native', 'loader'):
        plan.add_argument('--' + option, type=Path)
    check = commands.add_parser('check', help='replay a saved cost receipt without running the entry')
    check.add_argument('cost', type=Path)
    input_graph = commands.add_parser('input-graph', help='project copy and mapped input subgraphs')
    for option in ('lifecycle', 'mapped-plan', 'out'):
        input_graph.add_argument('--' + option, type=Path, required=True)
    counters = commands.add_parser('input-counters', help='diagnose native process faults at source phase boundaries')
    counters.add_argument('--mapped-plan', type=Path, required=True)
    counters.add_argument('--out', type=Path, required=True)
    counters.add_argument('--pairs', type=int, default=3)
    counters.add_argument('--compiler', default='g++')
    export = commands.add_parser('export-native-route', help='publish the selected C++20 source route package')
    export.add_argument('--mapped-plan', type=Path, required=True)
    export.add_argument('--out', type=Path, required=True)
    run = commands.add_parser('run-native-route', help='validate one explicitly requested native launch')
    run.add_argument('--package', type=Path, required=True)
    run.add_argument('--out', type=Path, required=True, help='fresh validation directory for stdout, stderr and the generated file')
    run.add_argument('--stable-inputs', action='store_true', help='commit to stable input files through process exit')
    review = commands.add_parser('review-native-route', help='join source graph, counters and C++20 delivery evidence')
    for option in ('graph', 'package', 'out'):
        review.add_argument('--' + option, type=Path, required=True)
    review.add_argument('--counters', type=Path)
    startup = commands.add_parser('startup-graph', help='bind source-level import requests to observed module origins')
    for option in ('lifecycle', 'initializers', 'out'):
        startup.add_argument('--' + option, type=Path, required=True)
    startup.add_argument('--observation', type=Path)
    for name, help_text in (
        ('publication-effects', 'observe file and stream publication from the original and C++20 routes'),
        ('publication-guard', 'verify the occupied-file difference on a private publication path'),
    ):
        publication = commands.add_parser(name, help=help_text)
        publication.add_argument('--package', type=Path, required=True)
        publication.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)

    from scar.analysis.cpu_fixed_grid_lifecycle import SCHEMA as GRAPH, derive_cpu_fixed_grid_lifecycle
    from scar.compiler.cpu_fixed_grid_cli_delivery import SCHEMA as DELIVERY
    from scar.optimization.cpu_fixed_grid_lifecycle_cost import (
        SCHEMA as COST, derive_cpu_fixed_grid_lifecycle_cost, verify_cpu_fixed_grid_lifecycle_cost,
    )

    if args.command == 'graph':
        from scar.analysis.cpu_fixed_grid_cli_effect_bridge import SCHEMA as EFFECTS
        from scar.analysis.cpu_fixed_grid_cli_resources import SCHEMA as RESOURCES

        result = derive_cpu_fixed_grid_lifecycle(_ref(args.delivery, DELIVERY), _ref(args.effects, EFFECTS),
                                                 _ref(args.resources, RESOURCES))
        path = args.out
    elif args.command == 'native':
        from scar.compiler.observe_cpu_fixed_grid_native_lifecycle import measure

        measure(_ref(args.delivery, DELIVERY), args.out, samples=args.samples, compiler=args.compiler)
        print(args.out / 'result.json')
        return 0
    elif args.command == 'plan':
        from scar.compiler.calibrate_cpu_fixed_grid_loader import SCHEMA as LOADER
        from scar.compiler.observe_cpu_fixed_grid_native_lifecycle import SCHEMA as NATIVE
        from scar.compiler.observe_cpu_fixed_grid_python_lifecycle import SCHEMA as PYTHON
        from scar.optimization.cpu_fixed_grid_cli_review import SCHEMA as REVIEW

        result = derive_cpu_fixed_grid_lifecycle_cost(
            _ref(args.review, REVIEW), lifecycle_ref=_ref(args.lifecycle, GRAPH),
            python_diagnostic=_ref(args.python, PYTHON) if args.python is not None else None,
            native_diagnostic=_ref(args.native, NATIVE) if args.native is not None else None,
            loader_probe=_ref(args.loader, LOADER) if args.loader is not None else None)
        path = args.out
    elif args.command == 'check':
        reference = _ref(args.cost, COST)
        result = verify_cpu_fixed_grid_lifecycle_cost(reference)
        print(json.dumps(dict(cost=reference, status=result['status'], execution_default=result['execution_default'])))
        return 0
    elif args.command == 'startup-graph':
        from scar.analysis.cpu_entry_import_initializers import SCHEMA as INITIALIZERS
        from scar.analysis.cpu_fixed_grid_startup_graph import derive_cpu_fixed_grid_startup_graph
        from scar.compiler.cpu_import_binding_observation import SCHEMA as IMPORTS

        observed = None if args.observation is None else _ref(args.observation, IMPORTS)
        result = derive_cpu_fixed_grid_startup_graph(_ref(args.lifecycle, GRAPH), _ref(args.initializers, INITIALIZERS),
                                                     observation_ref=observed)
        path = args.out
    elif args.command in ('publication-effects', 'publication-guard'):
        from scar.codegen.cpu_fixed_grid_native_route_package import SCHEMA as PACKAGE
        from scar.compiler.cpu_fixed_grid_publication_guard_study import study
        from scar.compiler.observe_cpu_fixed_grid_publication import measure_cpu_fixed_grid_publication

        measure = measure_cpu_fixed_grid_publication if args.command == 'publication-effects' else study
        measure(_ref(args.package, PACKAGE), args.out)
        print(args.out / 'result.json')
        return 0
    else:
        from scar.analysis.cpu_fixed_grid_input_route_graph import SCHEMA as INPUT_GRAPH, derive_cpu_fixed_grid_input_route_graph
        from scar.codegen.cpu_fixed_grid_native_route_package import SCHEMA as PACKAGE, export_cpu_fixed_grid_native_route_package
        from scar.compiler.observe_cpu_fixed_grid_input_counters import SCHEMA as COUNTERS, measure as measure_counters
        from scar.compiler.run_cpu_fixed_grid_native_route import execute_cpu_fixed_grid_native_route
        from scar.optimization.cpu_fixed_grid_input_route_review import derive_cpu_fixed_grid_input_route_review
        from scar.optimization.cpu_fixed_grid_mapped_input_plan import SCHEMA as MAPPED

        path = args.out
        if args.command == 'input-graph':
            result = derive_cpu_fixed_grid_input_route_graph(_ref(args.lifecycle, GRAPH), _ref(args.mapped_plan, MAPPED))
        elif args.command == 'input-counters':
            measure_counters(_ref(args.mapped_plan, MAPPED), path, pairs=args.pairs, compiler=args.compiler)
            print(path / 'result.json')
            return 0
        elif args.command == 'export-native-route':
            print(export_cpu_fixed_grid_native_route_package(_ref(args.mapped_plan, MAPPED), path)['uri'])
            return 0
        elif args.command == 'run-native-route':
            result = execute_cpu_fixed_grid_native_route(_ref(args.package, PACKAGE), path,
                        stable_inputs_for_process_lifetime=args.stable_inputs)
            print(json.dumps({key: result.get(key) for key in ('status', 'route', 'reasons', 'target_executed')}))
            return 0 if result['target_executed'] else 1
        else:
            counter_ref = None if args.counters is None else _ref(args.counters, COUNTERS)
            result = derive_cpu_fixed_grid_input_route_review(_ref(args.graph, INPUT_GRAPH), _ref(args.package, PACKAGE),
                                                              counters_ref=counter_ref)

    _write(path, result)
    print(path)
    return 0
