"""Launch the independently admitted incumbent named by a pinned entry review."""

import argparse
import json
from pathlib import Path

from scar.ir.execution_recipe import read_artifact
from scar.runtime.csr_region_entry import launch_context
from scar.runtime.entry_runner import _arguments, run_original_script
from scar.runtime.sparse_parallel_entry import SCHEMA as INCUMBENT
from scar.runtime.sparse_parallel_scaling import _try_team


SCHEMA = 'scar.sparse-entry-review-decision'
EXECUTION = 'scar.sparse-entry-review-execution'
_DECISIONS = ('KEEP_INCUMBENT', 'RELEASE_REVIEW_REQUIRED')


def _select(entry, context, decision, fingerprint, exclusive):
    if not decision or not fingerprint or exclusive is not True:
        return None, 'ORIGINAL_EXECUTION_CONDITIONS'
    try:
        path = Path(decision).resolve()
        review = read_artifact(dict(uri=str(path), sha256=fingerprint,
            size_bytes=path.stat().st_size, schema=SCHEMA))
        if (type(review['schema_version']) is not int or review['schema_version'] != 1
                or review['decision'] not in _DECISIONS or review['challengers_auto_authorized'] is not False):
            return None, 'ENTRY_REVIEW_POLICY_NOT_ADMITTED'
        if (review['entry']['uri'] != str(entry)
                or {**context, 'cpu_ids': review['context']['cpu_ids']} != review['context']):
            return None, 'ENTRY_REVIEW_CONTEXT_CHANGED'
        read_artifact(review['entry'], json_document=False)
        reference = review['incumbent']
        incumbent = read_artifact(reference)
        if (reference['schema'] != INCUMBENT or incumbent['schema_version'] != 1
                or incumbent['decision'] != 'PARALLEL' or incumbent['study_parallel'] is not False
                or incumbent['entry'] != review['entry'] or incumbent['context'] != review['context']
                or [incumbent['owner_cpu']] != review['context']['cpu_ids']):
            return None, 'ENTRY_REVIEW_INCUMBENT_NOT_RELEASED'
        return review, 'ENTRY_REVIEW_ADMITTED'
    except (KeyError, OSError, TypeError, ValueError):
        return None, 'ENTRY_REVIEW_ARTIFACT_CHANGED'


def run(entry, argv, *, decision=None, sha256=None, exclusive=False):
    entry, argv = _arguments(entry, argv)
    context = launch_context(argv)
    review, reason = _select(entry, context, decision, sha256, exclusive)
    report = None
    applied = 'KEEP'
    if review is not None:
        # A non-None child report means the entry has run, including when
        # the child had to use its own fallback. Never run the entry twice.
        report = _try_team(entry, argv, review['incumbent'], study=False)
        if report is None:
            reason = 'ENTRY_REVIEW_INCUMBENT_CONTEXT_OR_RESOURCES_CHANGED'
        elif report['applied'] == 'PARALLEL':
            applied, reason = 'INCUMBENT', 'ENTRY_REVIEW_INCUMBENT_APPLIED'
        else:
            applied, reason = report['applied'], 'ENTRY_REVIEW_INCUMBENT_FALLBACK'
    if report is None:
        report = run_original_script(entry, argv)
    report['schema'] = EXECUTION
    report['applied'] = applied
    report['review'] = dict(decision_sha256=sha256, policy_decision=None if review is None else review['decision'],
        attempted_incumbent=review is not None, applied=applied, reason=reason, context=context)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decision')
    parser.add_argument('--decision-sha256')
    parser.add_argument('--exclusive-inputs', action='store_true')
    parser.add_argument('--report', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('original Python script required after --')
    if args.report and args.report.exists():
        parser.error('fresh report path required')
    report = run(command[0], command[1:], decision=args.decision, sha256=args.decision_sha256,
        exclusive=args.exclusive_inputs)
    if args.report:
        with args.report.open('x') as stream:
            json.dump(report, stream, sort_keys=True, indent=2)
            stream.write('\n')


if __name__ == '__main__':
    main()
