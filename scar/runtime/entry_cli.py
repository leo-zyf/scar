"""Run an unchanged Python entry with a scoped, admitted native boundary."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path


def entry_scope(entry, argv):
    from .entry_runner import runner_implementation_sha256
    path = Path(entry).resolve()
    return dict(kind='AUTOMATIC_RUNNER', entry_path=str(path),
                entry_sha256=sha256(path.read_bytes()).hexdigest(), argv=list(argv),
                runner_implementation_sha256=runner_implementation_sha256())


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar run', description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--admission-sha256', required=True)
    parser.add_argument('--exclusive-inputs', action='store_true')
    parser.add_argument('--memory-budget-bytes', type=int)
    parser.add_argument('--reuse-count', type=int, default=1)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--deployment-plan')
    choice.add_argument('--measure-candidate', action='store_true',
                        help='measure an admitted candidate; does not assert deployment profitability')
    parser.add_argument('--report', help='fresh output path; application stdout is preserved')
    parser.add_argument('command', nargs=argparse.REMAINDER, help='-- ENTRY.py [arguments]')
    args = parser.parse_args(argv)
    command = args.command
    if command[:1] == ['--']: command = command[1:]
    if not command: parser.error('a Python script entry is required after --')
    entry = Path(command[0]).resolve()
    if not entry.is_file() or entry.suffix != '.py': parser.error('a Python .py entry file is required')
    report_path = Path(args.report).absolute() if args.report else None
    if report_path is not None and (report_path.exists() or report_path.is_symlink()):
        parser.error('--report requires a fresh path')
    from scar.runtime.array_copy_guard import inspect_bundle
    from .entry_runner import run_script
    metadata = inspect_bundle(args.bundle, expected_admission_sha256=args.admission_sha256)
    contract = metadata['contract']
    stateful = contract.get('schema') == 'scar.stateful-runtime-contract'
    required_bytes = (contract['memory']['total_region_array_payload_bytes'] if stateful
                      else contract['array']['final_bytes'])
    memory_scope = (contract['memory']['scope'] if stateful else
                   'REGION_INCREMENTAL_FRESH_FINAL_PAYLOAD_EXCLUDES_EXTERNAL_INPUTS_AND_PROCESS_RSS')
    scope = entry_scope(entry, command[1:])
    gate = dict(decision='NO_OP', selected=False, reasons=['deployment_cost_plan_not_provided'],
                deployment_authorized=False)
    if args.deployment_plan:
        from scar.optimization.deployment_gate import deployment_gate
        gate = deployment_gate(args.deployment_plan, bundle_ids=contract['native']['ids'],
            admission_sha256=args.admission_sha256, entry_scope=scope,
            reuse_count=args.reuse_count,
            available_cpu_ids=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            memory_budget_bytes=args.memory_budget_bytes,
            required_cpu_ids=contract['cpu']['ids'],
            required_memory_bytes=required_bytes, memory_scope=memory_scope)
    elif args.measure_candidate:
        gate = dict(decision='MEASURE_ADMITTED_CANDIDATE', selected=True,
                    deployment_authorized=False,
                    reasons=['explicit_candidate_measurement_is_not_a_cost_approval'])
    enabled = gate['decision'] in ('LOAD_NATIVE_CANDIDATE', 'MEASURE_ADMITTED_CANDIDATE')
    if args.memory_budget_bytes is not None and args.memory_budget_bytes < required_bytes:
        enabled = False
        gate = dict(decision='NO_OP', selected=False, deployment_authorized=False,
                    reasons=['explicit_payload_memory_budget_insufficient'])
    execution = run_script(entry, command[1:], bundle=args.bundle,
        expected_admission_sha256=args.admission_sha256,
        exclusive_inputs=args.exclusive_inputs, native_enabled=enabled)
    statistics = execution.get('statistics', {})
    counts = [statistics.get(key) for key in ('native_calls', 'fallback_calls')]
    observed = sum(counts) if all(type(value) is int for value in counts) else None
    matches = observed is not None and observed == args.reuse_count
    complete = execution.get('boundary_calls_observation_complete') is True
    report = dict(schema='scar.automatic-entry-execution', schema_version=1,
        entry_scope=scope, deployment_gate=gate, execution=execution,
        expected_reuse_count=args.reuse_count,
        reuse_count_source='CLI_DECLARATION', observed_installed_boundary_calls=observed,
        reuse_count_matches_observed_installed_calls=matches,
        cost_scope_verified=(gate['decision'] == 'LOAD_NATIVE_CANDIDATE' and complete
                             and matches and statistics.get('fallback_calls') == 0),
        memory_budget_bytes=args.memory_budget_bytes,
        required_memory_bytes=required_bytes, memory_scope=memory_scope,
        mode='MEASUREMENT_ONLY' if args.measure_candidate else 'COST_GATED_DEPLOYMENT',
        full_program_cpp=False, global_optimality_proven=False)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open('x') as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
