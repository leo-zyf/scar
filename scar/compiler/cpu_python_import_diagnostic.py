"""Observe original CPython imports in one admitted CPU entry invocation."""

import argparse
import json
from pathlib import Path

from scar.compiler.cpu_entry_launch_scope import verify_paired_original_scope
from scar.compiler.cpu_entry_replay import SCHEMA as PAIRED
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.cpu_entry_decision_ledger import SCHEMA as LEDGER, verify_cpu_entry_decision_ledger


def _write(path, document):
    encoded = json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError('existing import projection differs from recovered evidence: ' + str(path))
    else:
        with path.open('x') as stream:
            stream.write(encoded)
    return artifact_reference(path, document['schema'])


def main(argv=None):
    from scar.analysis.cpu_python_import_graph import derive_cpu_python_import_graph
    from scar.compiler import cpu_original_importtime
    from scar.optimization.cpu_python_import_cost import derive_cpu_python_import_cost

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--decision-ledger', type=Path)
    mode.add_argument('--check', type=Path)
    parser.add_argument('--paired', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--recover', action='store_true', help='replay existing raw samples without launching an entry')
    parser.add_argument('--initializations-only', action='store_true',
                        help='omit cached import trace rows to reduce capture volume')
    args = parser.parse_args(argv)
    if args.check is not None:
        observation = cpu_original_importtime.check(args.check)
        print(json.dumps(dict(observation=str(args.check.resolve()), schema=observation['schema'], replayed=True)))
        return
    if args.out is None or args.paired is None:
        parser.error('--decision-ledger requires --paired and --out')
    if args.recover and args.initializations_only:
        parser.error('--recover uses the original recorded profiling mode')
    ledger_ref = artifact_reference(args.decision_ledger.resolve(strict=True), LEDGER)
    ledger = verify_cpu_entry_decision_ledger(ledger_ref)
    obligations = read_artifact(ledger['effect_frontier'])['entry_obligations']
    paired = artifact_reference(args.paired.resolve(strict=True), PAIRED)
    verify_paired_original_scope(read_artifact(ledger['domain'])['scope'], paired)
    out = args.out.resolve()
    samples = out/'samples'
    if args.recover:
        observation = cpu_original_importtime.recover(paired, samples)
    else:
        out.mkdir(parents=True, exist_ok=False)
        observation = cpu_original_importtime.measure(paired, samples, runs=args.runs,
            profile_mode=1 if args.initializations_only else 2)
    if observation['paired'] != paired:
        raise ValueError('recovered import observation belongs to another paired launch')
    observed = artifact_reference(samples/'result.json', observation['schema'])
    graph = _write(out/'import-graph.json', derive_cpu_python_import_graph(obligations, observed, decision_ledger=ledger_ref))
    cost = _write(out/'import-cost.json', derive_cpu_python_import_cost(graph, observed))
    print(json.dumps(dict(observation=observed, import_graph=graph, import_cost=cost), sort_keys=True))
