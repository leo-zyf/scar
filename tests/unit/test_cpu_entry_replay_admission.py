"""The CPU replay must admit its exact invocation before either workload runs."""

import pytest

from scar.compiler import cpu_entry_replay


def test_invocation_gate_precedes_output_directory(tmp_path, monkeypatch):
    spec = {'original': 'checked launch'}
    monkeypatch.setattr(cpu_entry_replay, '_ref', lambda path, schema: {'uri': path, 'schema': schema})
    monkeypatch.setattr(cpu_entry_replay, '_spec', lambda reference, capture, cpu: spec)

    def reject(invocation):
        assert invocation is spec
        raise ValueError('different launch scope')

    out = tmp_path/'replay'
    with pytest.raises(ValueError, match='different launch scope'):
        cpu_entry_replay.measure('compilation.json', out, _admit_invocation=reject)
    assert not out.exists()
