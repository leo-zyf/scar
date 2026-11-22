"""Build and replay the scoped hardware timeline for an admitted native array route."""

import argparse
import json
from pathlib import Path

from scar.analysis import cpu_array_native_route_lifecycle as lifecycle
from scar.analysis.cpu_array_native_route import SCHEMA as PROOF
from scar.analysis.replay_transaction import replay_transaction
from scar.codegen.cpu_array_native_route_package import SCHEMA as PACKAGE
from scar.compiler.calibrate_array_entry import _ref, _write
from scar.compiler.calibrate_cpu_array_native_route_body import SCHEMA as BODY
from scar.compiler.calibrate_cpu_array_native_route_boundary import SCHEMA as PUBLICATION
from scar.compiler.calibrate_cpu_array_native_route_matched import SCHEMA as MATCHED
from scar.compiler.cpu_array_native_route_phase_probe import SCHEMA as PHASES
from scar.optimization import cpu_array_native_route_component_cost as components
from scar.optimization import cpu_array_native_route_forward as forward


def plan(proof, out, *, grouped_probe=None, matched_probe=None, package=None, diagnostics=None, publication=None):
    directory = Path(out).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    with replay_transaction():
        source, checked = forward._read_proof(proof, package)
        graph_path = directory/'lifecycle.json'
        graph = lifecycle._project(proof, source)
        _write(graph_path, graph)
        costs_path = directory/'components.json'
        costs = components._project(proof, source, grouped_probe, matched_probe=matched_probe, checked_package=checked)
        _write(costs_path, costs)
        report = forward._project(_ref(graph_path, lifecycle.SCHEMA), _ref(costs_path, components.SCHEMA),
                                  graph, costs, package, diagnostics, checked, publication=publication)
    _write(directory/'result.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--proof', type=Path)
    mode.add_argument('--check', type=Path)
    parser.add_argument('--out', type=Path)
    probes = parser.add_mutually_exclusive_group()
    probes.add_argument('--grouped-body', type=Path)
    probes.add_argument('--matched-dispatch', type=Path)
    parser.add_argument('--package', type=Path)
    parser.add_argument('--diagnostics', type=Path)
    parser.add_argument('--publication', type=Path)
    args = parser.parse_args(argv)
    if args.check is not None:
        if any((args.out, args.grouped_body, args.matched_dispatch, args.package, args.diagnostics, args.publication)):
            parser.error('--check reads an existing forward timeline')
        report = forward.verify_cpu_array_native_route_forward(_ref(args.check, forward.SCHEMA))
        result = args.check
    else:
        if args.out is None:
            parser.error('--proof requires a fresh --out path')
        if args.diagnostics is not None and args.package is None:
            parser.error('--diagnostics requires its original --package')
        report = plan(_ref(args.proof, PROOF), args.out,
            grouped_probe=None if args.grouped_body is None else _ref(args.grouped_body, BODY),
            matched_probe=None if args.matched_dispatch is None else _ref(args.matched_dispatch, MATCHED),
            package=None if args.package is None else _ref(args.package, PACKAGE),
            diagnostics=None if args.diagnostics is None else _ref(args.diagnostics, PHASES),
            publication=None if args.publication is None else _ref(args.publication, PUBLICATION))
        result = args.out/'result.json'
    print(json.dumps(dict(result=str(result), status=report['status'], whole_entry_ranked=report['whole_entry_ranked'])))


if __name__ == '__main__':
    raise SystemExit(main())
