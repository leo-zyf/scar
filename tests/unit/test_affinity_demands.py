"""Static affinity propagation over synthetic metadata, never native execution."""
from dataclasses import replace
import json

import pytest

from test_temporal_resources import inputs
from test_versioned_schedule import ordered_state_schedule
from scar.analysis.scope_demands import derive_serial_affinity_demands
from scar.analysis.implementation_envelope import derive_stateful_envelope
from scar.analysis.schedule_time import attach_task_time
from scar.analysis.target_graph import build_target_graph
from scar.codegen.stateful import implementation_sha256
from scar.ir.execution_recipe import artifact_reference, make_recipe
from scar.ir.program import target_hardware as H
from scar.ir.versioned_schedule import TemporalOccupancyKind as OK, VersionedTaskKind as K, schedule_document


@pytest.fixture
def cases(inputs,tmp_path):
    _,target,resources,threads,_=inputs
    target_ref=artifact_reference(target/'result.json','scar.target-hardware-graph')
    def make(cpu=0, *, provider=False, parallel=False):
        original=ordered_state_schedule()
        original=replace(original,values=tuple(replace(value,attrs={'input_id':'input:0'})
            if value.id=='initial' else value for value in original.values))
        if provider:
            original=replace(original,tasks=tuple(replace(task,capability='CPU_NUMPY_UFUNC_PROVIDER',
                attrs={'stage_kind':'PROVIDER'}) if task.id=='compute' else task for task in original.tasks))
        selected=[0,1] if parallel else [cpu]
        executor=next(row for row in resources if type(row) is H.CpuExecutor and set(selected)<=set(row.cpu_ids))
        suffix=f'{cpu}-{provider}-{parallel}'
        path=tmp_path/('plan-'+suffix+'.json')
        path.write_text(json.dumps(dict(schema='scar.stateful-stage-plan',schema_version=1,
            cpu_threads=len(selected),resource_binding=dict(cpu_ids=selected,executor_id=str(executor.id),
                target_path=str(target),target_sha256=target_ref['sha256']))))
        plan_ref=artifact_reference(path,'scar.stateful-stage-plan')
        region=tmp_path/('region-'+suffix+'.json');region.write_text(json.dumps({'schema':'scar.test-region'}))
        recipe=make_recipe('cpu-stateful-array-region-v1',
            semantic_boundary={'id':original.region_id,'inputs':[{'id':'input:0','kind':'array','shape':[1,2]}]},
            control={'fixed_source_order':True,'spatial_partition':'OPENMP_STATIC_DISJOINT_ROWS' if parallel else 'SERIAL_ROWS'},
            task_graph=schedule_document(original),resources={'fixture':'STRUCTURE_NOT_SOURCE_QUALIFICATION'},
            implementation={'family':'CPP_SYNCHRONOUS_CPU_TRANSACTIONAL_STATEFUL_REGION',
                'backend_inputs':{'region':'region','plan':'stage_plan'},
                'backend_capability_record':{'implementation_sha256':implementation_sha256()}},
            artifacts={'region':artifact_reference(region,'scar.test-region'),
                       'stage_plan':plan_ref,'target_snapshot':target_ref})
        return recipe,attach_task_time(derive_stateful_envelope(original,recipe))
    return make,target,threads


def _execution_by_task(schedule):
    tasks={activity.id:activity.task for activity in schedule.temporal.activities}
    return {tasks[row.activity]:row for row in schedule.temporal.occupancies if row.kind is OK.EXECUTION_DEMAND}


def test_only_closed_bound_window_refines_demands_and_opaque_provider_invalidates_following_work(cases):
    make,target,threads=cases
    recipe,before=make()
    after=derive_serial_affinity_demands(before,recipe,target)
    assert (after.tasks,after.values,after.materializations,after.storage_slots,after.proof)==(
        before.tasks,before.values,before.materializations,before.storage_slots,before.proof)
    assert (after.temporal.events,after.temporal.activities,after.temporal.constraints,after.temporal.scopes)==(
        before.temporal.events,before.temporal.activities,before.temporal.constraints,before.temporal.scopes)
    old,new=_execution_by_task(before),_execution_by_task(after)
    exit_task=next(t.id for t in after.tasks if t.attrs.get('phase_kind')=='WORKER_EXIT')
    expected={'snapshot','compute','write','reuse',exit_task}
    assert {task for task,row in new.items() if row.units==1}==expected
    assert all(new[task].resource==threads[0].id.wire for task in expected)
    assert all(new[task]==row for task,row in old.items() if task not in expected)
    assert all(row.units is None for task,row in new.items() if task not in expected)
    assert [o for o in after.temporal.occupancies if o.kind is OK.PHYSICAL_SLOT_BYTES]==[
        o for o in before.temporal.occupancies if o.kind is OK.PHYSICAL_SLOT_BYTES]
    proof=after.temporal.source_refs[-1]
    assert proof['actual_cpu_observation'] is False and proof['global_reservation'] is False
    assert proof['native_execution_authorized'] is False and proof['cost_inferred'] is False
    recipe,opaque=make(provider=True)
    contaminated=derive_serial_affinity_demands(opaque,recipe,target)
    rows=_execution_by_task(contaminated)
    assert {task for task,row in rows.items() if row.units==1}=={'snapshot'}
    assert all(rows[key].units is None for key in ('compute','write','reuse',exit_task))


def test_original_cpu_identity_is_generic_and_conflicts_bad_anchor_or_parallel_are_not_upgraded(cases):
    make,target,threads=cases
    recipe,timed=make(cpu=2)
    different=derive_serial_affinity_demands(timed,recipe,target)
    assert {row.resource for row in _execution_by_task(different).values() if row.units==1}=={threads[2].id.wire}
    recipe,timed=make()
    # Identical hardware IDs from another snapshot are not this source target.
    foreign=target.parent/'foreign-target'
    build_target_graph(foreign,inventory=target/'inventory.json')
    with pytest.raises(ValueError,match='target snapshot'):
        derive_serial_affinity_demands(timed,recipe,foreign)
    activity=next(a for a in timed.temporal.activities if a.task=='compute')
    old=next(o for o in timed.temporal.occupancies if o.activity==activity.id)
    conflict=replace(timed,temporal=replace(timed.temporal,
        occupancies=tuple(replace(o,resource=threads[2].id.wire,units=1) if o.id==old.id else o
                          for o in timed.temporal.occupancies),
        resource_capacities={**timed.temporal.resource_capacities,threads[2].id.wire:1}))
    with pytest.raises(ValueError,match='different execution resource'):
        derive_serial_affinity_demands(conflict,recipe,target)
    bad=replace(timed,tasks=tuple(replace(task,attrs={**task.attrs,'phase_kind':'UNPROVEN_BIND_ANCHOR'})
        if task.attrs.get('phase_kind')=='TEAM_ENTER' else task for task in timed.tasks))
    with pytest.raises(ValueError,match='original replay'):
        derive_serial_affinity_demands(bad,recipe,target)
    recipe,parallel=make(parallel=True)
    with pytest.raises(ValueError,match='single-worker serial'):
        derive_serial_affinity_demands(parallel,recipe,target)
