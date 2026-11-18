"""Select a qualified multi-region sparse pipeline for the sealed entry."""

import argparse
import json
from pathlib import Path

from scar.ir.execution_recipe import read_artifact
from scar.runtime.csr_region_entry import launch_context
from scar.runtime.entry_runner import _arguments
from scar.runtime.sparse_pipeline_entry import run as run_pipeline

SCHEMA = 'scar.sparse-pipeline-launch'


def _select(entry, argv, path, fingerprint, *, exclusive, trial):
    if exclusive is not True or not path or not fingerprint:
        return 'KEEP', None, None, 'ORIGINAL_EXECUTION_CONDITIONS'
    try:
        path = Path(path).resolve()
        launch = read_artifact(dict(uri=str(path), sha256=fingerprint, schema=SCHEMA, size_bytes=path.stat().st_size))
        if (launch['schema_version'] != 1 or launch['entry']['uri'] != str(entry)
                or launch['context'] != launch_context(argv)):
            return 'KEEP', None, None, 'SPARSE_PIPELINE_CONTEXT_CHANGED'
        mode = launch['decision']
        if trial is True and launch['study_joint'] is True:
            mode = 'JOINT'
        if mode == 'KEEP':
            return mode, None, None, 'KEEP_FROM_PLAN'
        if mode not in ('SOLVE_ONLY', 'JOINT'):
            raise ValueError('unknown sparse pipeline mode')
        for reference in launch['runtime_members']:
            read_artifact(reference, json_document=False)
        row = launch['row_selection'] if mode == 'JOINT' else None
        return mode, row, launch['solve_selection'], 'EXPLICIT_JOINT_TRIAL' if trial else 'QUALIFIED_PIPELINE_SELECTION'
    except (OSError, KeyError, TypeError, ValueError):
        return 'KEEP', None, None, 'SPARSE_PIPELINE_SELECTION_CHANGED'


def run(entry, argv, *, selection=None, sha256=None, exclusive=False, study_joint=False):
    entry, argv = _arguments(entry, argv)
    mode, row, solver, reason = _select(entry, argv, selection, sha256, exclusive=exclusive, trial=study_joint)
    report = run_pipeline(entry, argv, row_selection=row, solve_selection=solver, exclusive=exclusive,
        study=row is not None)
    report['launch'] = dict(requested=mode, reason=reason, selection_sha256=sha256)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection')
    parser.add_argument('--selection-sha256')
    parser.add_argument('--exclusive-inputs', action='store_true')
    parser.add_argument('--study-joint', action='store_true')
    parser.add_argument('--report', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('original Python script required after --')
    if args.report and args.report.exists():
        parser.error('fresh report path required')
    report = run(command[0], command[1:], selection=args.selection, sha256=args.selection_sha256,
        exclusive=args.exclusive_inputs, study_joint=args.study_joint)
    if args.report:
        with args.report.open('x') as stream:
            json.dump(report, stream, sort_keys=True, indent=2)
            stream.write('\n')


if __name__ == '__main__':
    main()
