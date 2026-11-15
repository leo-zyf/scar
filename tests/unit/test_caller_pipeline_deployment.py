"""Protocol test with hypothetical samples, never a real speedup claim."""
from hashlib import sha256
import json

from scar.runtime import caller_pipeline_deployment as admission


def test_exact_pinned_entry_and_missing_cost_default_no_op(tmp_path, monkeypatch):
    def save(path, value, schema):
        path.write_text(json.dumps(value, sort_keys=True))
        return admission.reference(path, schema)
    entry = tmp_path/'ordinary.py'; entry.write_text('print(1)\n')
    context = admission.entry_context(entry, ['x']); bundle = {'hypothetical_test_bundle': True}
    monkeypatch.setattr(admission, 'bundle_reference', lambda *args: (bundle, {}))
    monkeypatch.setattr(admission, 'provider_identity', lambda: {'hypothetical_test_provider': True})
    policy = dict(warmup=1, repetitions=3, minimum_gain=.05, frozen_before_measurement=True)
    validator = admission.reference(__file__, 'scar.validation-raw')
    common = dict(bundle=bundle, entry_context=context, environment_sha256=admission.environment_sha256(),
        runner_identity=admission.runner_identity(), source_refs=[admission.reference(entry, 'source')],
        provider_identity=admission.provider_identity(), policy=policy, validator_implementation=validator)
    design = save(tmp_path/'design.json', dict(schema=admission.DESIGN, schema_version=1, **common), admission.DESIGN)
    samples = []; evidence = [design, validator]
    for index, phase in enumerate(('WARMUP', 'MEASUREMENT', 'MEASUREMENT', 'MEASUREMENT')):
        folder = tmp_path/str(index); folder.mkdir()
        output = folder/'output'; output.write_bytes(b'1\n')
        error = folder/'error'; error.write_bytes(b'')
        stdout = admission.reference(output, 'scar.validation-raw'); stderr = admission.reference(error, 'scar.validation-raw')
        execution = dict(schema=admission.EXECUTION, schema_version=1, mode='MEASURE_CANDIDATE',
            bundle=bundle, entry_context=context, runner_identity=common['runner_identity'],
            environment_sha256=common['environment_sha256'], context_unchanged=True, validation_design=design,
            execution=dict(schema='scar.entry-runner-execution', schema_version=1, entry_path=str(entry), argv=['x'],
                statistics=dict(native_calls=1, fallback_calls=0), exit_code=0, installed_bindings=1,
                frontier=[], decision='GUARDED_ATTACHMENT', observed_boundary_calls=1))
        report = save(folder/'execution.json', execution, admission.EXECUTION)
        baseline = dict(wall_ns=100, returncode=0, stdout=stdout, stderr=stderr)
        optimized = dict(wall_ns=80, returncode=0, stdout=stdout, stderr=stderr, execution_report=report)
        sample = dict(schema=admission.SAMPLE, schema_version=1, phase=phase,
            index=0 if phase == 'WARMUP' else index-1, baseline=baseline, optimized=optimized,
            valid=True, native_calls=1, fallback_calls=0)
        ref = save(folder/'sample.json', sample, admission.SAMPLE); samples.append(dict(**sample, sample_ref=ref))
        evidence.extend((stdout, stderr, report, ref))
    receipt = dict(schema=admission.SCHEMA, schema_version=1, trust_policy=admission.TRUST,
        decision='LOAD_VALIDATED_CALLER_PIPELINE', **common, samples=samples, evidence=evidence)
    path = tmp_path/'deployment.json'; ref = save(path, receipt, admission.SCHEMA)
    def gate(**overrides):
        values = dict(expected_deployment_sha256=ref['sha256'], bundle='fixture',
            expected_admission_sha256='unused_stub_pin', entry=entry, argv=['x'])
        values.update(overrides)
        return admission.deployment_gate(path, **values)
    # Positive candidate process timing cannot authorize the final delivery,
    # whose receipt-replay cost was absent from these hypothetical clocks.
    positive = gate()
    assert positive['deployment_authorized'] is False
    assert positive['reason'] == 'positive_candidate_gain_only_final_delivery_admission_cost_not_covered'
    assert positive['evidence_status'] == 'PRELIMINARY_CANDIDATE_ONLY'
    assert positive['final_delivery_cost_admission_supported'] is False
    assert gate(argv=['changed'])['deployment_authorized'] is False
    assert gate(expected_deployment_sha256='0'*64)['deployment_authorized'] is False
    monkeypatch.setenv('SCAR_TEST_ONLY_CONTEXT_CHANGE', 'yes')
    assert gate()['deployment_authorized'] is False
    monkeypatch.delenv('SCAR_TEST_ONLY_CONTEXT_CHANGE')
    entry.write_text('print(2)\n')
    assert gate()['deployment_authorized'] is False

    # No cost authority: no admission, library load or import attachment.
    from scar.runtime import caller_pipeline_delivery as delivery, entry_runner
    monkeypatch.setattr(entry_runner, 'run_original_script', lambda *args: {'original_calls': 1})
    monkeypatch.setattr(delivery, 'bundle_reference', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('unexpected bundle access')))
    result = delivery.run_caller_entry(entry, [], bundle='unavailable', expected_admission_sha256='0'*64,
        exclusive_inputs=True)
    assert result['execution'] == {'original_calls': 1}
    assert result['deployment_gate']['deployment_authorized'] is False
    assert result['native_libraries_loaded'] is False and result['import_attachment_enabled'] is False
