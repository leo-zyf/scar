"""Declared outputs must survive capture and subsequent release evidence replay."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from scar.compiler import caller_pipeline_capsule as capsule


def test_final_validation_preserves_and_replays_both_publication_trees(tmp_path, monkeypatch):
    bundle = tmp_path / 'capsule'
    bundle.mkdir()
    output = tmp_path / 'application-output'
    entry = tmp_path / 'entry.py'
    argv = ['--output', str(output)]
    validator = tmp_path / 'validator.py'
    validator.write_text('# validator fixture\n')
    payload = dict(schema=capsule.PAYLOAD, entry_context=dict(
        entry_path=str(entry), argv=argv, working_directory=str(tmp_path)),
        validator_implementation=capsule._reference(validator, 'scar.validation-raw'),
        policy=dict(warmup=1, repetitions=3, minimum_gain=.05),
        publication=capsule._publication_contract(output, ['nested/result'], ['elapsed'], argv))
    payload_ref = capsule._save(bundle / 'payload.json', payload)
    (bundle / 'trial.slot').write_bytes(b'fake trial data')
    trial_ref = capsule._reference(bundle / 'trial.slot', capsule.SLOT)
    # The receipt's immutable native admission is covered separately. Simulate
    # two processes that publish identical one-file payloads and distinct clocks.
    monkeypatch.setattr(capsule, '_prepared', lambda *args: payload)
    monkeypatch.setattr(capsule, '_counts', lambda *args: (64, 0, True))
    elapsed = iter(tick for duration in [100, 80, 80, 100] * 2 for tick in (0, duration))
    monkeypatch.setattr(capsule.time, 'perf_counter_ns', lambda: next(elapsed))

    def publish(command, *, stdout, stderr, env, cwd):
        assert command[-3:] == [str(entry), *argv]
        assert not output.exists()
        (output / 'nested').mkdir(parents=True)
        (output / 'nested/result').write_bytes(b'exact output')
        candidate = '--report' in command
        (output / 'elapsed').write_text('0.1\n' if candidate else '0.2\n')
        if candidate:
            Path(command[command.index('--report') + 1]).write_text('{"schema": "observed"}')
        stdout.write(b'equal stdout\n')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(capsule.subprocess, 'run', publish)
    validated = capsule.validate_capsule(bundle, tmp_path / 'validation',
        expected_payload_sha256=payload_ref['sha256'], expected_trial_slot_sha256=trial_ref['sha256'])
    result = validated['result']
    assert result['validated'] and result['declared_publications_validated']
    assert result['decision'] == 'EXPORT_PROFITABLE_RELEASE_SLOT'
    assert not output.exists()
    assert all(sample['publications_match'] for sample in result['samples'])
    sample = result['samples'][0]
    retained = Path(sample['optimized']['publications']['nested/result']['file']['uri'])
    retained.write_bytes(b'changed after validation')
    with pytest.raises(ValueError, match='retained publication'):
        capsule.export_cost_slot(bundle, validated['validation']['uri'],
            expected_payload_sha256=payload_ref['sha256'],
            expected_validation_sha256=validated['validation']['sha256'])
    assert not (bundle / 'decision.slot').exists()
    retained.write_bytes(b'exact output')
    exported = capsule.export_cost_slot(bundle, validated['validation']['uri'],
        expected_payload_sha256=payload_ref['sha256'],
        expected_validation_sha256=validated['validation']['sha256'])
    assert exported['decision'] == 'RELEASE' and exported['declared_publication_evidence_replayed']


def test_publication_contract_rejects_escape_and_nonfinite_clock(tmp_path):
    argv = ['--output', str(tmp_path)]
    with pytest.raises(ValueError, match='canonical relative'):
        capsule._publication_contract(tmp_path, ['../outside'], [], argv)
    contract = capsule._publication_contract(tmp_path, [], ['elapsed'], argv)
    (tmp_path / 'elapsed').write_text('NaN\n')
    with pytest.raises(ValueError, match='finite nonnegative'):
        capsule._publication_snapshot(tmp_path, contract)
