import os

import pytest

from scar.compiler.contracts import ExecutionPlanRef
from scar.compiler.middle.plan import (
    NativeRevision, publish_plan, read_native_binding, read_plan_contract, schedule_native,
)


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
