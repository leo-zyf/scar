"""Select an admitted sparse worker team from a published capacity comparison."""

import argparse
import json
import os
from pathlib import Path

from scar.ir.execution_recipe import read_artifact
from scar.runtime.csr_region_entry import launch_context
from scar.runtime.entry_runner import _arguments
from scar.runtime.sparse_parallel_entry import _admit, _affinity, _resources_fit, run as run_team

SCHEMA = 'scar.sparse-parallel-scaling-selection'
EXECUTION = 'scar.sparse-parallel-scaling-execution'


def _selection(entry, argv, selection, fingerprint, exclusive):
    if exclusive is not True or not selection or not fingerprint:
        return None
    try:
        path = Path(selection).resolve()
        row = read_artifact(dict(uri=str(path), schema=SCHEMA, size_bytes=path.stat().st_size, sha256=fingerprint))
        if (row['schema_version'] == 1 and row['entry']['uri'] == str(entry)
                and {**launch_context(argv), 'cpu_ids':[row['owner_cpu']]} == row['context']):
            read_artifact(row['runtime'], json_document=False)
            return row
    except (KeyError, ValueError, OSError, TypeError):
        pass
    return None


def _try_team(entry, argv, reference, *, study):
    try:
        child = read_artifact(reference)
        cpus = set(child['launch_cpu_ids'])
    except (KeyError, OSError, TypeError, ValueError):
        return None
    if not cpus <= os.sched_getaffinity(0):
        return None
    with _affinity(cpus):
        admitted = _admit(entry, argv, reference['uri'], reference['sha256'], True)
        if admitted is None or not _resources_fit(admitted):
            return None
        return run_team(entry, argv, selection=reference['uri'], sha256=reference['sha256'],
                        exclusive=True, study_parallel=study)


def run(entry, argv, *, selection=None, sha256=None, exclusive=False, study_expanded=False):
    entry, argv = _arguments(entry, argv)
    chosen = _selection(entry, argv, selection, sha256, exclusive)
    tried = []
    report = None
    mode = 'KEEP'
    if chosen is not None:
        requested = chosen['decision']
        if study_expanded is True and chosen['study_expanded'] is True:
            requested = 'EXPANDED'
        candidates = ('EXPANDED', 'ESTABLISHED') if requested == 'EXPANDED' else ('ESTABLISHED',)
        for name in candidates:
            tried.append(name)
            report = _try_team(entry, argv, chosen['teams'][name], study=name == 'EXPANDED')
            if report is not None:
                mode = name if report['applied'] == 'PARALLEL' else report['applied']
                break
    if report is None:
        report = run_team(entry, argv)
    report['schema'] = EXECUTION
    report['scaling'] = dict(fingerprint=sha256, applied=mode, attempted_teams=tried)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection')
    parser.add_argument('--selection-sha256')
    parser.add_argument('--exclusive-inputs', action='store_true')
    parser.add_argument('--study-expanded', action='store_true')
    parser.add_argument('--report', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('original Python script required after --')
    if args.report and args.report.exists():
        parser.error('fresh report path required')
    report = run(command[0], command[1:], selection=args.selection, sha256=args.selection_sha256,
        exclusive=args.exclusive_inputs, study_expanded=args.study_expanded)
    if args.report:
        with args.report.open('x') as stream:
            json.dump(report, stream, sort_keys=True, indent=2)
            stream.write('\n')


if __name__ == '__main__':
    main()
