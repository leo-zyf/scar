"""Measure one existing guarded caller implementation against its unchanged entry.

This adapter reuses the complete-process validation utilities. It compares only
the original and the one compiled implementation; region time is not inferred.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .validate_build import _reference, _save, _equal_files, _statistics
from scar.runtime.caller_pipeline_deployment import (SCHEMA, DESIGN, SAMPLE, TRUST,
    entry_context, runner_identity, environment_sha256, provider_identity,
    source_references, bundle_reference, execution_counts, reference)
from scar.runtime.compiled_deployment import _raw


def validate_caller_pipeline(bundle, entry, argv, out, *, expected_admission_sha256,
        source_paths=(), warmup=1, repetitions=3, minimum_gain=.05):
    """Freeze one exact invocation, validate one candidate, and retain raw pairs.

    ``source_paths`` is the already-admitted source catalog's path set. Paths are
    frozen here; deployment never loads the graph or scans source directories.
    Inherited environment values remain parent-local, with only a digest saved.
    """
    from scar.runtime.entry_runner import _arguments
    entry, argv = _arguments(entry, argv); out = Path(out).resolve()
    if (type(warmup) is not int or warmup < 1 or type(repetitions) is not int or repetitions < 3
            or type(minimum_gain) not in (int, float) or not math.isfinite(minimum_gain)
            or not 0 <= minimum_gain < 1):
        raise ValueError('at least one warmup, three measurements and finite gain in [0,1) required')
    if out.exists(): raise ValueError('fresh caller validation output required')
    frozen_bundle, metadata = bundle_reference(bundle, expected_admission_sha256)
    if Path(metadata['contract']['source']['path']).resolve() == entry:
        raise ValueError('entry-defined caller attachment is not supported')
    context = entry_context(entry, argv); identity = runner_identity(); environment_hash = environment_sha256()
    environment = dict(os.environ); provider = provider_identity()
    sources = source_references(entry, metadata['contract'], source_paths)
    policy = dict(warmup=warmup, repetitions=repetitions, minimum_gain=minimum_gain,
        frozen_before_measurement=True,
        admission='EXACT_ENTRY_STDOUT_STDERR_SUCCESS_NATIVE_ONLY_AND_DECLARED_MEDIAN_GAIN')
    validator = _reference(Path(__file__), 'scar.validation-raw')
    design = dict(schema=DESIGN, schema_version=1, bundle=frozen_bundle, entry_context=context,
        environment_sha256=environment_hash, runner_identity=identity, provider_identity=provider,
        source_refs=sources, source_scope='FROZEN_EXPLICIT_RETAINED_SOURCE_SET_NOT_TRANSITIVE_IMPORT_PROOF',
        policy=policy, validator_implementation=validator,
        pair_order='ALTERNATE_ORIGINAL_FIRST_AND_CANDIDATE_FIRST',
        process_timing_scope='PARENT_MONOTONIC_LAUNCH_THROUGH_EXIT_INCLUDES_IMPORTS_ADMISSION_GUARDS_APPLICATION_AND_REPORT_IO',
        validation_scope='EXACT_STDOUT_STDERR_EXIT_PLUS_ORIGINAL_C1_AND_C2_LIVE_BOUNDARY_GUARDS',
        arbitrary_external_effect_validation=False, regional_cost_profile_produced=False,
        deployment_receipt_replay_in_candidate_clock=False,
        selected_implementations_compared=1, target_variants_ranked=False,
        native_backend_recompiled=False, tensor_values_recorded=False)
    out.mkdir(parents=True); (out/'samples').mkdir()
    design_ref = _save(out/'design.json', design)
    evidence = [design_ref, validator]; samples = []; reasons = []

    def stable():
        if (entry_context(entry, argv) != context or runner_identity() != identity
                or environment_sha256() != environment_hash or provider_identity() != provider
                or bundle_reference(bundle, expected_admission_sha256)[0] != frozen_bundle): return False
        for source in sources: _raw(source)
        return True

    def process(command, folder, label):
        stdout = folder/(label+'.stdout'); stderr = folder/(label+'.stderr')
        with stdout.open('xb') as output, stderr.open('xb') as errors:
            start = time.perf_counter_ns()
            child = subprocess.run(command, stdout=output, stderr=errors,
                env=environment, cwd=context['working_directory'])
            elapsed = time.perf_counter_ns()-start
        row = dict(command=command, wall_ns=elapsed, returncode=child.returncode,
            stdout=_reference(stdout, 'scar.validation-raw'), stderr=_reference(stderr, 'scar.validation-raw'))
        evidence.extend((row['stdout'], row['stderr']))
        return row

    pairs = [('WARMUP', i) for i in range(warmup)] + [('MEASUREMENT', i) for i in range(repetitions)]
    for ordinal, (phase, index) in enumerate(pairs):
        folder = out/'samples'/(phase.lower()+'-'+str(index)); folder.mkdir()
        report_path = folder/'candidate-execution.json'
        commands = dict(baseline=[sys.executable, str(entry), *argv], optimized=[sys.executable,
            '-m', 'scar.runtime.caller_pipeline_delivery', '--bundle', frozen_bundle['directory'],
            '--admission-sha256', expected_admission_sha256, '--exclusive-inputs', '--measure-candidate',
            '--validation-design', design_ref['uri'], '--design-sha256', design_ref['sha256'],
            '--report', str(report_path), '--', str(entry), *argv])
        records = {}
        for label in (('baseline', 'optimized') if ordinal % 2 == 0 else ('optimized', 'baseline')):
            records[label] = process(commands[label], folder, label)
        baseline, optimized = records['baseline'], records['optimized']
        native = fallback = 0; native_only = False
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_bytes())
                report_ref = _reference(report_path, report['schema']); evidence.append(report_ref)
                optimized['execution_report'] = report_ref
                native, fallback, native_only = execution_counts(report, bundle=frozen_bundle,
                    context=context, identity=identity, environment=environment_hash, design=design_ref)
            except (OSError, ValueError, KeyError, TypeError): optimized['execution_report_invalid'] = True
        stdout_match = _equal_files(baseline['stdout']['uri'], optimized['stdout']['uri'])
        stderr_match = _equal_files(baseline['stderr']['uri'], optimized['stderr']['uri'])
        returncodes = baseline['returncode'] == optimized['returncode'] == 0
        try: unchanged = stable()
        except (OSError, ValueError, KeyError, TypeError, ImportError): unchanged = False
        sample = dict(schema=SAMPLE, schema_version=1, phase=phase, index=index,
            baseline=baseline, optimized=optimized, stdout_match=stdout_match, stderr_match=stderr_match,
            successful_returncodes=returncodes, native_calls=native, fallback_calls=fallback,
            native_only=native_only, frozen_context_unchanged=unchanged,
            valid=stdout_match and stderr_match and returncodes and native_only and unchanged)
        sample_ref = _save(folder/'sample.json', sample); evidence.append(sample_ref)
        samples.append(dict(**sample, sample_ref=sample_ref))
        if not sample['valid']:
            reasons.append('ENTRY_OUTPUT_EXIT_NATIVE_COVERAGE_OR_FROZEN_CONTEXT_INVALID'); break
    measured = [row for row in samples if row['phase'] == 'MEASUREMENT']
    original = _statistics([row['baseline']['wall_ns'] for row in measured])
    optimized = _statistics([row['optimized']['wall_ns'] for row in measured])
    a, b = original['median_ns'], optimized['median_ns']
    gain = 1-b/a if a and b else None; speedup = a/b if a and b else None
    complete = len(samples) == warmup+repetitions and len(measured) == repetitions
    valid = complete and all(row['valid'] for row in samples)
    profitable = valid and gain is not None and gain >= minimum_gain
    if not complete: reasons.append('PREDECLARED_MEASUREMENTS_INCOMPLETE')
    if valid and not profitable: reasons.append('NOT_PROFITABLE_AT_FROZEN_WHOLE_ENTRY_GAIN')
    if profitable: reasons.append('FINAL_DELIVERY_ADMISSION_REPLAY_COST_NOT_COVERED')
    receipt = dict(schema=SCHEMA, schema_version=1, trust_policy=TRUST,
        decision='PRELIMINARY_CANDIDATE_PROFITABLE' if profitable else 'NO_OP',
        deployment_authorized=False, final_delivery_cost_admission_supported=False,
        **{key: design[key] for key in ('bundle', 'entry_context', 'environment_sha256', 'runner_identity',
            'source_refs', 'source_scope', 'provider_identity', 'policy', 'validator_implementation')},
        samples=samples, evidence=evidence, reasons=reasons,
        performance=dict(baseline=original, optimized=optimized, speedup=speedup, gain_fraction=gain,
            clock='PARENT_MONOTONIC_FULL_PROCESS_LIFETIME', warmup_pairs_excluded_from_statistics=warmup),
        validation=dict(validated=valid, complete=complete, measurement_pairs=len(measured),
            stdout_stderr_exit_match=all(row['stdout_match'] and row['stderr_match'] and row['successful_returncodes'] for row in samples),
            native_calls=sum(row['native_calls'] for row in samples), fallback_calls=sum(row['fallback_calls'] for row in samples),
            scope=design['validation_scope'], arbitrary_external_effect_validation=False),
        regional_cost_profile_produced=False, full_program_cpp=False,
        deployment_receipt_replay_in_candidate_clock=False,
        selected_implementations_compared=1, target_variants_ranked=False,
        physical_optimality_proven=False, future_profitability_guaranteed=False)
    deployment_ref = _save(out/'deployment.json', receipt)
    return dict(receipt=receipt, deployment=deployment_ref, expected_deployment_sha256=deployment_ref['sha256'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True); parser.add_argument('--admission-sha256', required=True)
    parser.add_argument('--out', required=True); parser.add_argument('--source', action='append', default=[])
    parser.add_argument('--warmup', type=int, default=1); parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--minimum-gain', type=float, default=.05)
    parser.add_argument('command', nargs=argparse.REMAINDER); args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command: parser.error('original Python entry required after --')
    result = validate_caller_pipeline(args.bundle, command[0], command[1:], args.out,
        expected_admission_sha256=args.admission_sha256, source_paths=args.source,
        warmup=args.warmup, repetitions=args.repetitions, minimum_gain=args.minimum_gain)
    print(json.dumps(dict(deployment=result['deployment'], expected_deployment_sha256=result['expected_deployment_sha256'],
        decision=result['receipt']['decision'], performance=result['receipt']['performance'],
        validation=result['receipt']['validation'], reasons=result['receipt']['reasons']), sort_keys=True))
    return 0


if __name__ == '__main__': raise SystemExit(main())
