"""The CPU replay must admit its exact invocation before either workload runs."""

import json
import os
from shutil import which

import pytest

from scar.analysis.replay_transaction import replay_transaction
from scar.compiler import cpu_entry_replay
from scar.ir.execution_recipe import artifact_reference


def test_shared_pair_is_bound_to_a_reference_and_live_transaction(tmp_path, monkeypatch):
    report = dict(schema=cpu_entry_replay.SCHEMA, invocation={})
    original = tmp_path/'paired.json'
    another = tmp_path/'another.json'
    original.write_text(json.dumps(report))
    another.write_bytes(original.read_bytes())
    pair = artifact_reference(original, cpu_entry_replay.SCHEMA)
    unrelated = artifact_reference(another, cpu_entry_replay.SCHEMA)
    cold_checks = []

    def cold(path):
        cold_checks.append(path)
        return report

    monkeypatch.setattr(cpu_entry_replay, 'check', cold)
    monkeypatch.setattr(cpu_entry_replay, '_pins', lambda invocation: None)
    with replay_transaction():
        checked = cpu_entry_replay.CheckedCpuPair(pair)
        assert checked.report(pair) == checked.report(pair) == report
        with pytest.raises(ValueError, match='another reference'):
            checked.report(unrelated)
    assert cold_checks == [str(original)]
    with pytest.raises(ValueError, match='metadata transaction'):
        checked.report(pair)


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


@pytest.mark.skipif(not cpu_entry_replay._WAIT4_AVAILABLE or which('c++') is None,
                    reason='Linux wait4 and C++20 compiler required')
def test_address_space_applied_to_candidate_only(tmp_path):
    cpu = str(min(os.sched_getaffinity(0)))
    argv = ['/usr/bin/taskset', '-c', cpu, '/usr/bin/timeout', '-k', '2', '5', '/bin/true']
    layout = [
        (dict(pair=0, mode=mode, argv=argv, cwd=str(tmp_path), environment={}), tmp_path/mode)
        for mode in ('original', 'candidate')
    ]
    limit = 256 * 1024 * 1024
    rows, runner = cpu_entry_replay._run_rows(tmp_path, layout, [], True,
                                              candidate_address_space_limit_bytes=limit)
    assert all(row['status'] == 0 for row in rows)
    assert 'capacity_guard' not in rows[0]
    assert rows[1]['capacity_guard']['implementation'] == 'NATIVE_CXX20'
    cpu_entry_replay._check_capacity_samples(rows, runner, limit)
    rows[1]['capacity_guard']['soft_bytes'] -= 4096
    with pytest.raises(ValueError, match='pre-exec readback'):
        cpu_entry_replay._check_capacity_samples(rows, runner, limit)
