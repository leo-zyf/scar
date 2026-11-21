"""Measure source-bound CPU native lifecycle spans outside clean benchmarks."""

import argparse
import json
from pathlib import Path

from scar.analysis.cpu_entry_lifecycle_graph import SCHEMA as LIFECYCLE
from scar.analysis.cpu_native_phase_sites import derive_cpu_native_phase_sites
from scar.codegen import native_array_lifecycle
from scar.ir.execution_recipe import artifact_reference
from scar.optimization.cpu_entry_forward_cost import SCHEMA as FORWARD
from scar.optimization.cpu_native_lifecycle_cost import derive_cpu_native_lifecycle_cost


def _write(path, document):
    with path.open('x') as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return artifact_reference(path, document['schema'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--lifecycle-graph', type=Path)
    mode.add_argument('--check', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--forward-cost', type=Path)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=0)
    args = parser.parse_args(argv)
    if args.check is not None:
        checked = native_array_lifecycle.check(args.check)
        print(json.dumps(dict(observation=str(args.check.resolve()), schema=checked['schema'], replayed=True)))
        return
    if args.out is None:
        parser.error('--lifecycle-graph requires --out')
    graph = artifact_reference(args.lifecycle_graph.resolve(strict=True), LIFECYCLE)
    forward = (artifact_reference(args.forward_cost.resolve(strict=True), FORWARD)
               if args.forward_cost is not None else None)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sites = _write(out/'phase-sites.json', derive_cpu_native_phase_sites(graph))
    samples = out/'samples'
    observation = native_array_lifecycle.measure(sites, samples, samples=args.samples, warmups=args.warmups)
    reference = artifact_reference(samples/'result.json', observation['schema'])
    cost = _write(out/'phase-cost.json', derive_cpu_native_lifecycle_cost(graph, reference, forward_cost=forward))
    print(json.dumps(dict(observation=reference, phase_cost=cost), sort_keys=True))
