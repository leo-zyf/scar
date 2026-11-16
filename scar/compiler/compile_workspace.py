"""Compile a source-qualified workspace-lifetime plan as an explicit trial."""
import argparse
import json

from scar.codegen.stateful_workspace import build_workspace_repeat, load_workspace_repeat


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--preparation'); source.add_argument('--check')
    parser.add_argument('--out'); args = parser.parse_args(argv)
    if args.check:
        if args.out: parser.error('check only consumes an existing native artifact')
        artifact = load_workspace_repeat(args.check)
    else:
        if not args.out: parser.error('compilation requires --out')
        artifact = build_workspace_repeat(args.preparation,args.out)
    report = artifact.report
    print(json.dumps(dict(source=str(artifact.directory/report['files']['source']),
        library=str(artifact.directory/report['files']['library']),
        cost_status='UNKNOWN',default_runtime_decision='NO_OP',deployment_authorized=False),sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
