"""Run one source-bound snapshot selection with the existing scoped C++ runtime."""

import argparse
import json
import os
from pathlib import Path
import sys

from scar.ir.execution_recipe import artifact_reference, read_artifact

SCHEMA = 'scar.snapshot-launch-selection'
REPORT = 'scar.snapshot-selection-execution'
NATIVE = 'LOAD_NATIVE_CANDIDATE'
NO_OP = 'NO_OP'
BUILD = 'scar.compiled-workspace-borrowed-roots'


class ContextMismatch(ValueError):
    """A frozen launch fact differs from its current value."""


def _need(condition, reason):
    if not condition:
        raise ContextMismatch(reason)


def _reference(selection, expected_sha256):
    path = Path(selection).resolve()
    return dict(uri=str(path), sha256=expected_sha256, size_bytes=path.stat().st_size, schema=SCHEMA)


def _context_matches(context, entry, argv):
    from scar.compiler.snapshot_entry_calibration import environment, OUTPUT

    _need(context['entry']['uri'] == str(entry), 'ENTRY_PATH_DIFFERS')
    read_artifact(context['entry'], json_document=False)
    _need(context['environment'] == environment(), 'ENVIRONMENT_DIFFERS')
    _need(context['caller_cpu_ids'] == sorted(os.sched_getaffinity(0)), 'CALLER_CPU_AFFINITY_DIFFERS')
    template, token = context['argv'], context['output_token']
    _need(token == OUTPUT and template.count(token) == 1 and len(template) == len(argv),
          'ENTRY_ARGUMENTS_DIFFER')
    slot = template.index(token)
    _need(argv[:slot] == template[:slot] and argv[slot+1:] == template[slot+1:],
          'ENTRY_ARGUMENTS_DIFFER')
    output = Path(argv[slot])
    _need(output.is_absolute() and not output.exists() and not output.is_symlink()
          and output.parent.is_dir(), 'FRESH_ABSOLUTE_OUTPUT_DIRECTORY_REQUIRED')
    _need(output.parent.stat().st_dev == context['output_parent_device'], 'OUTPUT_DEVICE_DIFFERS')
    monitor = getattr(sys, 'monitoring', None)
    _need(sys.getprofile() is None and sys.gettrace() is None
          and (monitor is None or all(monitor.get_tool(index) is None for index in range(6))),
          'SOURCE_OBSERVER_ACTIVE')


def _candidate(selection):
    from scar.runtime.session_scope_bundle import ADMISSION

    reference = selection['candidate']
    build = read_artifact(reference)
    _need(reference['schema'] == BUILD and build['schema_version'] == 1
          and build['explicit_trial_only'] is True
          and reference == selection['context']['candidate']
          and build['bundle'] == selection['bundle']
          and build['scope_admission'] == selection['scope_admission'] == selection['context']['scope_admission'],
          'SELECTION_CANDIDATE_DIFFERS')
    bundle = Path(build['bundle']).resolve()
    admission = build['scope_admission']
    _need(admission['schema'] == ADMISSION and admission['uri'] == str(bundle/'scope-admission.json'),
          'SCOPE_ADMISSION_PATH_DIFFERS')
    admitted = read_artifact(admission)
    _need(admitted['snapshot_rewrite'] == build['native'], 'SCOPE_NATIVE_IDENTITY_DIFFERS')
    for field, member in (('scope', 'scope.json'), ('runtime', 'runtime.json')):
        _need(build[field]['uri'] == str(bundle/member), 'COMPACT_MEMBER_PATH_DIFFERS')
    scope, runtime = (read_artifact(build[name]) for name in ('scope', 'runtime'))
    step = runtime['step_contract']
    sources = [row['source'] for row in (*scope['boundaries'], *step['callee_guards'])]
    sources.append(step['source'])
    hashes = {}
    for source in sources:
        path = source['path']
        if path not in hashes:
            hashes[path] = artifact_reference(path, 'scar.python-source')['sha256']
        _need(hashes[path] == source['sha256'], 'ORIGINAL_BOUNDARY_SOURCE_DIFFERS')
    return build


def _admit(entry, argv, selection, expected_sha256, *, exclusive_inputs, study):
    if selection is None or expected_sha256 is None:
        return None, None, 'PINNED_SELECTION_REQUIRED'
    if exclusive_inputs is not True:
        return None, None, 'EXPLICIT_EXCLUSIVE_INPUTS_REQUIRED'
    reference = None
    try:
        reference = _reference(selection, expected_sha256)
        saved = read_artifact(reference)
        _need(saved['schema_version'] == 1 and saved['decision'] in (NATIVE, NO_OP)
              and saved['requires_exclusive_inputs'] is True,
              'UNSUPPORTED_SELECTION')
        if saved['decision'] == NO_OP and study is not True:
            return reference, None, 'PLANNER_RETAINED_ORIGINAL'
        _need(saved['original_payload_budget_fits'] is True, 'NATIVE_RESOURCE_FIT_NOT_PROVEN')
        _context_matches(saved['context'], entry, argv)
        build = _candidate(saved)
        return reference, build, 'EXPLICIT_STUDY' if study is True else 'PINNED_SELECTION_MATCHED'
    except ContextMismatch as error:
        return reference, None, str(error)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
        return reference, None, 'SELECTION_OR_SOURCE_ARTIFACT_DIFFERS'


def run(entry, argv, *, selection=None, expected_sha256=None, exclusive_inputs=False, study=False):
    """Use a cold-published checksum to admit an exact launch without replaying F4."""
    from scar.runtime.entry_runner import _arguments, run_original_script

    entry, argv = _arguments(entry, argv)
    reference, build, reason = _admit(entry, argv, selection, expected_sha256,
        exclusive_inputs=exclusive_inputs, study=study)
    scope = None
    if build is not None:
        from scar.runtime.native_loop import GuardFailure
        from scar.runtime.session_scope_bundle import load_scope

        try:
            scope = load_scope(build['bundle'], expected_admission_sha256=build['scope_admission']['sha256'],
                trial=True, exclusive_inputs=exclusive_inputs)
        except (GuardFailure, OSError, ImportError, ValueError, TypeError, KeyError, AttributeError):
            reason = 'NATIVE_LOAD_DISALLOWED'
    if scope is None:
        execution = run_original_script(entry, argv)
        return dict(schema=REPORT, schema_version=1, selection=reference, decision=NO_OP,
            reason=reason, native_load_attempted=build is not None,
            native_loaded=False if build is None else None, native_calls=0, kernel_fallback_calls=0,
            execution=execution, scope=None, session_statistics=None, automatic_deployment=False)

    with scope.install():
        execution = run_original_script(entry, argv)
    guards = [binding['stats'] for binding in scope.bindings
              if scope.rows[binding['id']]['role'] == 'KERNEL']
    return dict(schema=REPORT, schema_version=1, selection=reference,
        decision='NATIVE_STUDY' if study is True else NATIVE, reason=reason,
        native_load_attempted=True, native_loaded=True,
        native_calls=scope.session.stats['native_calls'],
        kernel_fallback_calls=sum(row['fallback_calls'] for row in guards), execution=execution,
        scope=scope.report, session_statistics=dict(scope.session.stats), automatic_deployment=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection')
    parser.add_argument('--selection-sha256')
    parser.add_argument('--exclusive-inputs', action='store_true')
    parser.add_argument('--study', action='store_true')
    parser.add_argument('--report')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a Python entry script is required after --')
    if args.report and Path(args.report).exists():
        parser.error('a fresh report path is required')
    result = run(command[0], command[1:], selection=args.selection, expected_sha256=args.selection_sha256,
        exclusive_inputs=args.exclusive_inputs, study=args.study)
    if args.report:
        with Path(args.report).open('x') as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
