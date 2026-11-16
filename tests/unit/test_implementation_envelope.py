"""Envelope structure only; synthetic recipes do not establish source admission."""
from dataclasses import replace
import json

import pytest

from test_temporal_schedule import LEGACY_V2_SHA256
from test_versioned_schedule import ordered_state_schedule
from scar.analysis.implementation_envelope import derive_stateful_envelope
from scar.codegen.stateful import implementation_sha256
from scar.ir.execution_recipe import artifact_reference, make_recipe
from scar.ir.versioned_schedule import VersionedTaskKind as K, schedule_document, validate_versioned_schedule
from scar.optimization.resource_solver import semantic_sha256


@pytest.fixture
def original_and_envelope(tmp_path):
    original=ordered_state_schedule()
    original=replace(original,values=tuple(replace(value,attrs={**value.attrs,'input_id':'input:0'})
        if value.id=='initial' else value for value in original.values))
    path=tmp_path/'component.json';path.write_text(json.dumps({'schema':'scar.test-source-component'}))
    artifact=artifact_reference(path,'scar.test-source-component')
    recipe=make_recipe('cpu-stateful-array-region-v1',
        semantic_boundary={'id':original.region_id,'inputs':[{'id':'input:0','kind':'array','shape':[1,2]}]},
        control={'fixed_source_order':True,'spatial_partition':'SERIAL_ROWS'},
        task_graph=schedule_document(original),resources={'fixture':'NO_RUNTIME_HARDWARE_CLAIM'},
        implementation={'family':'CPP_SYNCHRONOUS_CPU_TRANSACTIONAL_STATEFUL_REGION',
            'backend_inputs':{'region':'region','plan':'stage_plan'},
            'backend_capability_record':{'implementation_sha256':implementation_sha256()}},
        artifacts={'region':artifact,'stage_plan':artifact})
    return original,derive_stateful_envelope(original,recipe)


def _ancestors(schedule):
    tasks={task.id:task for task in schedule.tasks};result={}
    for name in validate_versioned_schedule(schedule)['topological_order']:
        result[name]=set(tasks[name].depends_on)
        for before in tasks[name].depends_on:result[name].update(result[before])
    return result


def test_refiner_preserves_data_storage_and_original_hb_while_scan_and_precommit_phases_reach_fence(original_and_envelope):
    original,derived=original_and_envelope
    assert derived.values==original.values and derived.materializations==original.materializations
    assert derived.storage_slots==original.storage_slots and derived.order_constraints==original.order_constraints
    assert derived.temporal is None and derived.executable is False
    before,after=_ancestors(original),_ancestors(derived)
    assert all(ancestors<=after[name] for name,ancestors in before.items())
    tasks={task.id:task for task in derived.tasks}
    for old in original.tasks:
        new=tasks[old.id]
        assert (new.reads,new.writes,new.allocates,new.releases,new.attrs,new.source_refs)==(
            old.reads,old.writes,old.allocates,old.releases,old.attrs,old.source_refs)
    phases=[task for task in derived.tasks if task.kind is K.IMPLEMENTATION_PHASE]
    scan=next(task for task in phases if task.attrs['phase_kind']=='INPUT_FINITE_SCAN_SERIAL')
    assert scan.reads==('external0',) and scan.attrs['external_entry_materialization']=='external0'
    assert scan.id in after['commit']
    fence=next(task for task in derived.tasks if task.kind is K.SUCCESS_FENCE)
    pre={task.id for task in phases if task.attrs['failure_domain']=='PRECOMMIT'}
    post=[task for task in phases if task.attrs['failure_domain']=='POSTCOMMIT']
    assert pre<=set(fence.validates) and pre<=after[fence.id]
    assert len(post)==2 and all(task.attrs['may_fail'] is False and task.id not in fence.validates for task in post)
    assert all(task.attrs['phase_effects'] and task.source_refs for task in phases)
    envelope=derived.proof['implementation_envelope']
    assert envelope['backend_source_pins'] and envelope['normal_completion_only']
    assert envelope['native_lowering_authorized'] is False and envelope['repeat_adapter_modeled'] is False
    assert semantic_sha256(schedule_document(ordered_state_schedule()))==LEGACY_V2_SHA256


def test_phase_cannot_write_allocate_free_validate_fail_after_commit_or_enable_native(original_and_envelope):
    _,derived=original_and_envelope
    phase=next(task for task in derived.tasks if task.kind is K.IMPLEMENTATION_PHASE
               and task.attrs['failure_domain']=='POSTCOMMIT')
    def changed(task):
        return replace(derived,tasks=tuple(task if row.id==phase.id else row for row in derived.tasks))
    invalid=(replace(phase,writes=('rhs0',)),replace(phase,allocates=('temp',)),
        replace(phase,releases=('temp',)),replace(phase,validates=('compute',)),
        replace(phase,attrs={}),replace(phase,attrs={**phase.attrs,'phase_effects':[]}),
        replace(phase,source_refs=()),replace(phase,attrs={**phase.attrs,'may_fail':True}),
        replace(phase,attrs={**phase.attrs,'may_fail':'UNKNOWN'}))
    for task in invalid:
        with pytest.raises(ValueError):validate_versioned_schedule(changed(task))
    with pytest.raises(ValueError,match='non-executable'):
        validate_versioned_schedule(replace(derived,executable=True))
    # Adding a postcommit normal handoff to the rollback fence is invalid.
    bad=replace(derived,tasks=tuple(replace(task,validates=(*task.validates,phase.id))
        if task.kind is K.SUCCESS_FENCE else task for task in derived.tasks))
    with pytest.raises(ValueError,match='precommit'):
        validate_versioned_schedule(bad)
