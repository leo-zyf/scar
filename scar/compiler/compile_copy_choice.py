"""Compile the graph variant selected within a canonical component model."""
import argparse
import json

from scar.codegen.copy_choice import build_copy_choice_trial, load_copy_choice_trial


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--choice'); source.add_argument('--check')
    parser.add_argument('--out'); args = parser.parse_args(argv)
    if args.check:
        if args.out: parser.error('check only consumes a saved choice trial')
        artifact = load_copy_choice_trial(args.check)
    else:
        if not args.out: parser.error('compilation requires --out')
        artifact = build_copy_choice_trial(args.choice,args.out)
    report = artifact.report
    print(json.dumps(dict(source=str(artifact.directory/report['files']['source']),
        library=str(artifact.directory/report['files']['library']),
        copy_choice_receipt=report['copy_choice_receipt'], default_runtime_decision='NO_OP',
        deployment_authorized=False),sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
