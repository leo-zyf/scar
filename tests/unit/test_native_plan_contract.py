import os

import pytest

from scar.artifacts import file_reference
from scar.compiler.contracts import ExecutionPlanRef
from scar.compiler.middle.plan import (
    NativeRevision, entry_cpus, publish_plan, read_native_binding, read_plan_contract, schedule_native,
)
from scar.analysis.replay_transaction import replay_transaction

from test_cpu_hardware_placement import hardware


def test_native_baseline_observation_cannot_apply_an_unmeasured_rewrite(tmp_path, hardware):
    cpu = min(os.sched_getaffinity(0))
    cases = [{'id': 'case'}]
    request = publish_plan(tmp_path / 'request.json', dict(schema='scar.native-entry', cases=cases))
    artifacts = dict(request=request,
                     hardware=file_reference(hardware / 'result.json', 'scar.target-hardware-graph'))
    for role in ('program', 'graph_evidence', 'calls', 'simplified', 'rewrites'):
        artifacts[role] = publish_plan(tmp_path / f'{role}.json', {'schema': f'test.{role}'})
    revision = NativeRevision('main', artifacts, (), ({'operation': 'proposed-copy'},))
    observation = dict(schema='scar.native-entry-cost', schema_version=1, request=request,
                       scope='ONE_BASELINE_LAUNCH', cpu=cpu, elapsed_ns=100, exit=0)

    result = schedule_native(revision, observation, cases, tmp_path)
    problem, plan, verified = read_plan_contract(result)
    assert verified['feasible']
    assert plan['baseline_only']
    assert [option.id for option in problem.operations[0].options] == ['original']
    assert read_native_binding(result)[1]['rewrite_operations'] == []

    incompatible = publish_plan(tmp_path / 'tampered-plan.json', dict(plan, baseline_only=False))
    replay = ExecutionPlanRef('main', dict(result.artifacts, execution_plan=incompatible), {}, result.scope)
    with pytest.raises(ValueError, match='baseline scope'):
        read_plan_contract(replay)


def test_reused_cpu_inventory_still_checks_affinity_and_file_identity(hardware, monkeypatch):
    inventory_path = hardware / 'inventory.json'
    target = file_reference(hardware / 'result.json', 'scar.target-hardware-graph')
    live_cpus = {0}
    monkeypatch.setattr(os, 'sched_getaffinity', lambda process: set(live_cpus))
    with pytest.raises(ValueError, match='identity differs'):
        with replay_transaction() as transaction:
            assert entry_cpus(target) == {0}
            live_cpus.clear()
            live_cpus.update((1, 20))
            assert entry_cpus(target) == {1}
            assert transaction.by_namespace['scar.native-entry.cpu-admission/v1'] == {
                'hits': 1, 'misses': 1}
            executors = hardware / 'CpuExecutor.jsonl.gz'
            original = executors.read_bytes()
            try:
                executors.write_bytes(original + b'changed')
                entry_cpus(target)
            finally:
                executors.write_bytes(original)
    with pytest.raises(ValueError, match='identity differs'):
        with replay_transaction():
            assert entry_cpus(target) == {1}
            inventory_path.write_text('{}')
            entry_cpus(target)
