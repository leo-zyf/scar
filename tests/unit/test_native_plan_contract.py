import os

import pytest

from scar.compiler.contracts import ExecutionPlanRef
from scar.compiler.middle.plan import (
    NativeRevision, entry_cpus, publish_plan, read_native_binding, read_plan_contract, schedule_native,
)
from scar.analysis.replay_transaction import replay_transaction


def test_native_baseline_observation_cannot_apply_an_unmeasured_rewrite(tmp_path):
    cpu = min(os.sched_getaffinity(0))
    cases = [{'id': 'case'}]
    request = publish_plan(tmp_path / 'request.json', dict(schema='scar.native-entry', cases=cases))
    inventory = publish_plan(tmp_path / 'inventory.json', dict(
        schema='scar.target-inventory', cpu={'online_cpu_ids': [cpu]},
        execution_constraints={'allowed_cpu_ids': [cpu]},
    ))
    hardware = publish_plan(tmp_path / 'hardware.json', dict(
        schema='scar.target-hardware-graph', inventory=inventory,
    ))
    artifacts = dict(request=request, hardware=hardware)
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


def test_reused_cpu_inventory_still_checks_affinity_and_file_identity(tmp_path, monkeypatch):
    inventory_path = tmp_path / 'inventory.json'
    inventory = publish_plan(inventory_path, dict(schema='scar.target-inventory',
        cpu={'online_cpu_ids': [0, 1]}, execution_constraints={'allowed_cpu_ids': [0, 1]}))
    hardware = publish_plan(tmp_path / 'hardware.json', dict(
        schema='scar.target-hardware-graph', inventory=inventory,
    ))
    live_cpus = {0}
    monkeypatch.setattr(os, 'sched_getaffinity', lambda process: set(live_cpus))
    with pytest.raises(ValueError, match='identity differs'):
        with replay_transaction():
            assert entry_cpus(hardware) == {0}
            live_cpus.clear()
            live_cpus.add(1)
            assert entry_cpus(hardware) == {1}
            inventory_path.write_text('{}')
            entry_cpus(hardware)
