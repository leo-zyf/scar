"""Compile a graph-derived raw-copy fusion as an explicit C++20 trial."""
import argparse
import json
from pathlib import Path

from scar.codegen.window_order import build_window_fusion_trial, load_window_order_trial
from scar.compiler.plan_closed_windows import SCHEMA
from scar.ir.execution_recipe import artifact_reference, read_artifact


def compile_window_trial(preparation, out, *, window=None):
    # Inspect the hashed saved proposal only to choose its ID. The builder then
    # independently replays the entire original graph/cost/source derivation.
    ref = artifact_reference(Path(preparation).resolve(), SCHEMA)
    report = read_artifact(ref)
    plans = read_artifact(report['plans'])
    candidates = [row['window_id'] for row in plans['windows']
        if any(candidate['kind'] == 'FUSE_IDENTICAL_INDEPENDENT_RAW_COPY_LOOPS'
            for candidate in row['candidate_transforms'])]
    selected = [key for key in candidates if window is None or key == window]
    if len(selected) != 1:
        raise ValueError('one automatically proposed fusion window required; specify --window if ambiguous')
    artifact = build_window_fusion_trial(ref['uri'], selected[0], out)
    read_artifact(ref)
    return artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--windows'); source.add_argument('--rewrite'); source.add_argument('--check')
    parser.add_argument('--out'); parser.add_argument('--window')
    args = parser.parse_args(argv)
    if args.check:
        if args.out or args.window:
            parser.error('check only consumes an existing trial artifact')
        path = Path(args.check).resolve()
        metadata = path/'artifact.json' if path.is_dir() else path
        schema = json.loads(metadata.read_bytes()).get('schema')
        if schema == 'scar.native-certified-window-fusion':
            from scar.codegen.window_order import load_certified_window_fusion
            artifact = load_certified_window_fusion(args.check)
        else:
            artifact = load_window_order_trial(args.check)
    else:
        if not args.out:
            parser.error('compilation requires --out')
        if args.rewrite:
            if args.window:
                parser.error('a replacement graph already identifies its exact source window')
            from scar.codegen.window_order import build_certified_window_fusion
            artifact = build_certified_window_fusion(args.rewrite, args.out)
        else:
            artifact = compile_window_trial(args.windows, args.out, window=args.window)
    report = artifact.report
    print(json.dumps(dict(source=str(artifact.directory/report['files']['source']),
        library=str(artifact.directory/report['files']['library']), control_kind=report['control_kind'],
        trial_only=True, cost_status=report['cost_status'], default_decision='NO_OP',
        deployment_authorized=False), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
