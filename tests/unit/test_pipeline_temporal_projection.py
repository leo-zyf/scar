from copy import deepcopy
import json
import pytest

from test_pipeline_schedule import _pipeline
from scar.analysis.target_graph import build_target_graph
from scar.ir.execution_recipe import artifact_reference
from scar.optimization.pipeline_memory import plan_pipeline_memory
from scar.optimization.pipeline_schedule import plan_pipeline_ready_schedule
from scar.optimization.pipeline_temporal_model import (
    COST_SCHEMA, CONTROL_RESOURCE, EXAMPLE, ROLE, project_pipeline_temporal_model,
    solve_pipeline_temporal_projection, verify_pipeline_temporal_projection,
)


@pytest.fixture
def hardware(tmp_path, monkeypatch):
    monkeypatch.setattr('os.sched_getaffinity', lambda pid: {0, 1})
    inventory=dict(schema='scar.target-inventory',schema_version=1,sources=[],diagnostics=[],
        cpu=dict(online_cpu_ids=[0,1],model_names=['typed fixture'],
            processors=[dict(cpu_id=i,package_id=0,core_id=i) for i in (0,1)],
            numa_nodes=[dict(node_id=0,cpu_ids=[0,1],memory_total_bytes=2**20)]),
        dram=dict(memory_total_bytes=2**20),gpus=[],storage=dict(disks=[],block_devices=[]),paths=[],
        cpu_topology=dict(processors=[dict(cpu_id=i,online=True,package_id=0,core_id=i,
            die_id=0,thread_siblings=[i]) for i in (0,1)],isa_groups=[],caches=[]))
    path=tmp_path/'inventory.json';path.write_text(json.dumps(inventory))
    build_target_graph(tmp_path/'target',inventory=path)
    return tmp_path/'target'


def _inputs(target):
    pipeline = _pipeline([('a', ('arg:0',)), ('b', ('arg:0',)), ('c', ('a', 'b'))])
    memory = plan_pipeline_memory(pipeline, input_shapes=[2, 3], memory_budget_bytes=4096)
    ready = plan_pipeline_ready_schedule(pipeline, memory, target, cpu_ids=[0, 1])
    return pipeline, memory, ready


def test_unknown_projection_never_invokes_solver_or_invents_free_control(hardware, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('L2 must not invoke L3')
    monkeypatch.setattr('scar.optimization.resource_solver.solve_resource_schedule', forbidden)
    pipeline, memory, ready = _inputs(hardware)
    report = project_pipeline_temporal_model(pipeline, memory, ready, hardware)
    assert report['status']=='UNRESOLVED' and report['solver_invoked'] is False
    assert all(row=={'status':'UNKNOWN', 'duration_ticks':None}
               for row in report['projection']['task_costs'].values())
    assert not set(report['control_task_ids']) & set(report['projection']['task_placements'])
    assert report['projection']['memory_capacities']=={
        'explicit-incremental-host-payload-budget':memory['memory_budget_bytes']}
    assert sum(report['projection']['materialization_sizes'].values())==memory['peak_allocation_bytes']
    assert all(not key.startswith('formal-input:') for key in report['projection']['materialization_sizes'])
    assert report['projection']['memory_capacities'] != {
        'explicit-incremental-host-payload-budget':ready['encoded_graph_budget_bytes']}
    assert verify_pipeline_temporal_projection(pipeline, memory, ready, hardware, report)==report


def test_retained_l2_projection_replays_memory_once_before_solving(hardware, monkeypatch):
    from scar.optimization import pipeline_schedule, resource_solver

    pipeline, memory, ready = _inputs(hardware)
    projection = project_pipeline_temporal_model(pipeline, memory, ready, hardware)
    original = pipeline_schedule.verify_pipeline_memory
    replays = []

    def count(*args):
        replays.append(True)
        return original(*args)

    monkeypatch.setattr(pipeline_schedule, 'verify_pipeline_memory', count)
    plan = solve_pipeline_temporal_projection(pipeline, memory, ready, hardware, projection)
    assert replays == [True]
    assert plan['projection_report'] == projection
    assert plan['solver_report']['status'] == 'UNRESOLVED'

    def forbidden(*args, **kwargs):
        raise AssertionError('modified L2 evidence must be rejected before solving')

    monkeypatch.setattr(resource_solver, 'solve_resource_schedule', forbidden)
    first = next(iter(projection['projection']['task_costs'].values()))
    first['duration_ticks'] = 0
    with pytest.raises(ValueError, match='differs from original input/cost/resource replay'):
        solve_pipeline_temporal_projection(pipeline, memory, ready, hardware, projection)


def test_complete_declared_example_is_not_measurement_and_wrong_binding_or_partial_cost_is_unknown(hardware, tmp_path):
    pipeline, memory, ready = _inputs(hardware)
    unknown = project_pipeline_temporal_model(pipeline, memory, ready, hardware)
    tick = {'numerator':1, 'denominator':10**9}
    context = deepcopy(unknown['cost_context']); context['tick_seconds']=tick
    example = dict(schema=COST_SCHEMA, schema_version=1, basis=EXAMPLE, role=ROLE,
        context=context, tick_seconds=tick,
        task_costs={key:{'duration_ticks':1} for key in unknown['task_kinds']},
        control_resource_demands={key:{CONTROL_RESOURCE:1} for key in unknown['control_task_ids']})
    def project(value, name):
        path=tmp_path/name;path.write_text(json.dumps(value))
        return project_pipeline_temporal_model(pipeline, memory, ready, hardware,
            cost_model=artifact_reference(path,COST_SCHEMA))
    accepted = project(example, 'example.json')
    assert accepted['status']=='COMPLETE_DECLARED_MODEL_EXAMPLE'
    assert accepted['cost_authority']['measured'] is False
    assert accepted['deployment_authorized'] is False and accepted['solver_invoked'] is False
    mismatch=deepcopy(example); mismatch['context']['binding_sha256']='0'*64
    rejected=project(mismatch, 'wrong-binding.json')
    assert rejected['status']=='UNRESOLVED'
    assert all(row['duration_ticks'] is None for row in rejected['projection']['task_costs'].values())
    partial=deepcopy(example);partial['task_costs'].pop(next(iter(partial['task_costs'])))
    assert project(partial,'partial.json')['status']=='UNRESOLVED'
