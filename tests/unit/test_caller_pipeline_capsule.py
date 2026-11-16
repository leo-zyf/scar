"""Hypothetical protocol inputs test authority, never claim measured speedup."""
from hashlib import sha256
import json

import pytest

from scar.runtime import caller_pipeline_capsule as runtime
from scar.compiler.caller_pipeline_capsule import _slot


def test_fixed_slot_shared_trial_release_path_and_strict_scope(tmp_path, monkeypatch):
    entry = tmp_path/'entry.py'; entry.write_text('pass\n'); capsule = tmp_path/'capsule'; capsule.mkdir()
    payload = dict(schema=runtime.PAYLOAD, schema_version=1, trust_policy=runtime.TRUST,
        capsule_directory=str(capsule), slot_bytes=runtime.SLOT_BYTES,
        entry_context=runtime.entry_context(entry, []), environment_sha256=runtime.environment_sha256(),
        runtime_identity=runtime.implementation_identity(), source_refs=[], provider_identity={'fixture': True},
        bundle={'fixture_bundle': True, 'directory': 'fixture', 'admission_sha256': '1'*64},
        validator_implementation={'sha256': '2'*64}, policy=dict(warmup=1, repetitions=3, minimum_gain=.05,
            frozen_before_measurement=True), reporting=dict(required=True, allowed_root=str(tmp_path)))
    raw = json.dumps(payload, sort_keys=True).encode(); (capsule/'payload.json').write_bytes(raw); digest = sha256(raw).hexdigest()
    monkeypatch.setattr(runtime, 'provider_identity', lambda: {'fixture': True})
    monkeypatch.setattr(runtime, 'bundle_reference', lambda *args: (payload['bundle'], None))
    unknown = _slot(payload, digest, runtime.TRIAL)
    assert runtime.evaluate_slot(unknown, payload, digest)['complete'] is False
    # Synthetic 100/80 coordinates exercise the release parser only. No real
    # exporter is given these inputs and no public performance artifact exists.
    samples = [dict(phase=phase, index=index, baseline={'wall_ns': 100}, optimized={'wall_ns': 80},
        valid=True, sample_ref={'sha256': '3'*64}) for phase, index in
        [('WARMUP', 0), ('MEASUREMENT', 0), ('MEASUREMENT', 1), ('MEASUREMENT', 2)]]
    hypothetical = _slot(payload, digest, runtime.RELEASE, samples, '4'*64)
    assert runtime.evaluate_slot(hypothetical, payload, digest)['approved'] is True
    assert len(runtime.pack_slot(unknown)) == len(runtime.pack_slot(hypothetical)) == 4096
    from scar.runtime import caller_pipeline_cli, entry_runner
    calls = []
    monkeypatch.setattr(caller_pipeline_cli, 'run_caller_entry', lambda *args, **kwargs:
        calls.append('same_C2_runner') or dict(execution={'statistics': {'native_calls': 1, 'fallback_calls': 0}},
            native_libraries_loaded=True))
    monkeypatch.setattr(entry_runner, 'run_original_script', lambda *args: calls.append('whole_original_once') or {'original': 1})
    reports = []
    for index, (value, trial) in enumerate(((unknown, True), (hypothetical, False))):
        path = capsule/('slot-'+str(index)); encoded = runtime.pack_slot(value); path.write_bytes(encoded)
        report = tmp_path/('execution-'+str(index)+'.json')
        result = runtime.run_capsule_entry(entry, [], capsule=capsule, slot=path,
            expected_payload_sha256=digest, expected_slot_sha256=sha256(encoded).hexdigest(), report=report,
            exclusive_inputs=True, measure_candidate=trial)
        assert result['selected'] is True and result['context_unchanged'] is True
        assert result['deployment_authorized'] == int(not trial)
        assert result['raw_benchmark_evidence_replayed'] is False
        reports.append(report.read_bytes())
    assert calls == ['same_C2_runner', 'same_C2_runner']
    assert len(reports[0]) == len(reports[1])
    path = capsule/'slot-1'
    result = runtime.run_capsule_entry(entry, ['changed'], capsule=capsule, slot=path,
        expected_payload_sha256=digest, expected_slot_sha256=sha256(path.read_bytes()).hexdigest(),
        report=tmp_path/'changed.json', exclusive_inputs=True)
    assert result['selected'] is False and calls[-1] == 'whole_original_once'
    with pytest.raises(ValueError, match='explicitly unmeasured'):
        runtime.evaluate_slot(dict(hypothetical, role=runtime.TRIAL), payload, digest)
