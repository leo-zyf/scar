"""Retain NO-OP until final delivery cost is covered, or run an explicit trial.

The ordinary path loads no native library and installs no import hook. The
existing C2 measurement adapter remains the only implementation of live guards.
"""
import argparse
import json
from pathlib import Path

from .caller_pipeline_deployment import (EXECUTION, DESIGN, deployment_gate,
    entry_context, runner_identity, environment_sha256, reference, _bundle_identity,
    provider_identity)
from .compiled_deployment import _raw


def _context_matches(design, entry, argv, bundle, identity, environment):
    if any(design[key] != actual for key, actual in (
            ('entry_context', entry_context(entry, argv)), ('bundle', bundle),
            ('runner_identity', identity), ('environment_sha256', environment),
            ('provider_identity', provider_identity()))):
        return False
    for source in design['source_refs']: _raw(source)
    return True


def run_caller_entry(entry, argv, *, bundle, expected_admission_sha256,
        deployment_receipt=None, expected_deployment_sha256=None,
        exclusive_inputs=False, measure_candidate=False,
        validation_design=None, expected_design_sha256=None):
    """Positive candidate receipts remain preliminary, never deployment authority."""
    from .entry_runner import _arguments, run_original_script
    entry, argv = _arguments(entry, argv)
    context = entry_context(entry, argv); identity = runner_identity(); environment = environment_sha256()
    gate = dict(decision='NO_OP', deployment_authorized=False, reason='entry_performance_receipt_not_provided')
    selected = False; frozen_bundle = None; frozen_design = None; design = None; admitted = None
    if measure_candidate is True:
        gate = dict(decision='MEASURE_ADMITTED_CANDIDATE', deployment_authorized=False,
            reason='explicit_trial_is_not_deployment_authority')
        try:
            from .caller_pipeline_guard import _admit
            admitted = _admit(bundle, expected_admission_sha256)
            frozen_bundle = _bundle_identity(admitted.directory, admitted.pin, admitted.contract, admitted.admission)
            if validation_design is not None:
                frozen_design = reference(validation_design, DESIGN)
                if frozen_design['sha256'] != expected_design_sha256:
                    raise ValueError('frozen validation design pin differs')
                design = json.loads(_raw(frozen_design))
                if ((design.get('schema'), design.get('schema_version')) != (DESIGN, 1)
                        or not _context_matches(design, entry, argv, frozen_bundle, identity, environment)):
                    raise ValueError('frozen entry design differs from live context')
            selected = exclusive_inputs is True
            if not selected: gate['reason'] = 'exclusive_input_contract_not_explicitly_true'
        except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
            gate = dict(decision='NO_OP', deployment_authorized=False, reason='candidate_context_not_admitted:' + str(error))
    elif exclusive_inputs is True:
        gate = deployment_gate(deployment_receipt, expected_deployment_sha256=expected_deployment_sha256,
            bundle=bundle, expected_admission_sha256=expected_admission_sha256, entry=entry, argv=argv)
        selected = gate['deployment_authorized'] is True
        if selected:
            from .caller_pipeline_guard import _admit
            admitted = _admit(bundle, expected_admission_sha256)
            frozen_bundle = _bundle_identity(admitted.directory, admitted.pin, admitted.contract, admitted.admission)
    if selected:
        from .caller_pipeline_cli import _run_with_admission
        result = _run_with_admission(entry, argv, admitted,
            exclusive_inputs=True, measure_candidate=True)
        execution = result['execution']; loaded = result['native_libraries_loaded']
    else:
        execution = run_original_script(entry, argv); loaded = False
    stable = (entry_context(entry, argv) == context and runner_identity() == identity
        and environment_sha256() == environment)
    if design is not None:
        try: stable = stable and _context_matches(design, entry, argv, frozen_bundle, identity, environment)
        except (OSError, ValueError, KeyError, TypeError, ImportError): stable = False
    mode = ('MEASURE_CANDIDATE' if measure_candidate is True else
        'ADMITTED_EXACT_ENTRY_DEPLOYMENT' if selected else 'ORIGINAL_NO_OP')
    return dict(schema=EXECUTION, schema_version=1, bundle=frozen_bundle,
        entry_context=context, runner_identity=identity, environment_sha256=environment,
        validation_design=frozen_design, context_unchanged=stable, deployment_gate=gate,
        execution=execution, mode=mode, native_libraries_loaded=loaded,
        import_attachment_enabled=loaded, full_program_cpp=False,
        default_backend='NO_OP', target_variants_ranked=False,
        final_delivery_cost_admission_supported=False,
        future_profitability_guaranteed=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True); parser.add_argument('--admission-sha256', required=True)
    parser.add_argument('--deployment'); parser.add_argument('--deployment-sha256')
    parser.add_argument('--exclusive-inputs', action='store_true')
    parser.add_argument('--measure-candidate', action='store_true')
    parser.add_argument('--validation-design'); parser.add_argument('--design-sha256')
    parser.add_argument('--report'); parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv); command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command: parser.error('original Python entry required after --')
    report = Path(args.report).resolve() if args.report else None
    if report is not None and report.exists(): parser.error('fresh report required')
    result = run_caller_entry(command[0], command[1:], bundle=args.bundle,
        expected_admission_sha256=args.admission_sha256, deployment_receipt=args.deployment,
        expected_deployment_sha256=args.deployment_sha256,
        exclusive_inputs=args.exclusive_inputs, measure_candidate=args.measure_candidate,
        validation_design=args.validation_design, expected_design_sha256=args.design_sha256)
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open('x') as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    return 0


if __name__ == '__main__': raise SystemExit(main())
