"""Select the verified sparse-solver study for its exact launch context."""

import argparse
import json
from pathlib import Path

from scar.ir.execution_recipe import read_artifact
from scar.runtime.csr_region_entry import launch_context
from scar.runtime.entry_runner import _arguments
from scar.runtime.sparse_solve_entry import SELECTION, run as run_study

LAUNCH = 'scar.sparse-direct-solve-launch'


def _select(entry, argv, launch, expected_sha256, exclusive):
    if not launch or not expected_sha256 or exclusive is not True:
        return None, 'ORIGINAL_EXECUTION_CONDITIONS'
    try:
        path = Path(launch).resolve()
        selection = read_artifact(dict(uri=str(path), schema=LAUNCH, sha256=expected_sha256,
            size_bytes=path.stat().st_size))
        if selection['schema_version'] != 1 or selection['decision'] != 'LOAD_NATIVE_CANDIDATE':
            return None, 'KEEP_FROM_PLAN'
        if selection['entry']['uri'] != str(entry) or selection['context'] != launch_context(argv):
            return None, 'SPARSE_SOLVE_LAUNCH_CONTEXT_CHANGED'
        study = selection['study']
        if study['schema'] != SELECTION:
            raise ValueError('symbolic launch requires a sealed validated study')
        return study, 'QUALIFIED_SPARSE_SYMBOLIC_SCOPE'
    except (OSError, ValueError, KeyError, TypeError):
        return None, 'SPARSE_SOLVE_LAUNCH_ARTIFACT_CHANGED'


def run(entry, argv, *, launch=None, sha256=None, exclusive=False):
    entry, argv = _arguments(entry, argv)
    study, reason = _select(entry, argv, launch, sha256, exclusive)
    report = run_study(entry, argv, selection=None if study is None else study['uri'],
        sha256=None if study is None else study['sha256'], study=study is not None, exclusive=exclusive)
    report['launch'] = dict(reason=reason, sha256=sha256,
        applied='LOAD_NATIVE_CANDIDATE' if report['statistics']['native_calls'] else 'KEEP')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch')
    parser.add_argument('--launch-sha256')
    parser.add_argument('--exclusive-inputs', action='store_true')
    parser.add_argument('--report', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('original Python script required after --')
    if args.report and args.report.exists():
        parser.error('fresh report path required')
    report = run(command[0], command[1:], launch=args.launch, sha256=args.launch_sha256,
        exclusive=args.exclusive_inputs)
    if args.report:
        with args.report.open('x') as stream:
            json.dump(report, stream, sort_keys=True, indent=2)
            stream.write('\n')


if __name__ == '__main__':
    main()
