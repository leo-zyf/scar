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
    else:
        reference = _ref(args.cost, COST)
        result = verify_cpu_fixed_grid_lifecycle_cost(reference)
        print(json.dumps(dict(cost=reference, status=result['status'], execution_default=result['execution_default'])))
        return 0

    _write(path, result)
    print(path)
    return 0
