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


@pytest.mark.skipif(not cpu_entry_replay._WAIT4_AVAILABLE, reason='Linux wait4 required')
def test_wait4_isolated_from_compiler_resident_memory(tmp_path):
    compiler_memory = bytearray(64 * 1024 * 1024)
    for offset in range(0, len(compiler_memory), 4096):
        compiler_memory[offset] = 1
    layout = [
        (dict(pair=0, mode=mode, argv=['/bin/true'], cwd=str(tmp_path), environment={}), tmp_path/mode)
        for mode in ('original', 'candidate')
    ]
    rows, runner = cpu_entry_replay._run_rows(tmp_path, layout, [], True)
    observation = cpu_entry_replay._resource_observation(rows, 0, runner=runner)
    assert observation['method'] == 'LINUX_WAIT4_ISOLATED_RUNNER'
    assert all(row['status'] == 0 and 0 < row['resources']['peak_rss_bytes'] < len(compiler_memory) for row in rows)
    assert observation['target_only_peak_rss_observed'] is False
