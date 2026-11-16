"""Conditional group coordinates, not target latency or measurement admission."""
from dataclasses import replace
import json
import subprocess

import pytest

from test_implementation_envelope import original_and_envelope
from test_temporal_resources import inputs
from scar.analysis.control_milestone import derive_scope_control_milestone
from scar.analysis.implementation_scope import derive_stateful_scope_envelope
from scar.analysis.schedule_time import attach_task_time
from scar.analysis.temporal_resources import derive_temporal_resources
from scar.codegen.stateful import implementation_sha256
from scar.ir.activity_costs import (ActivityCost, ActivityCostTable, CostBasis as B,
    CostInterval, validate_activity_cost_table)
from scar.ir.execution_recipe import artifact_reference, make_recipe
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.temporal_resources import projection_document
from scar.ir.versioned_schedule import VersionedTaskKind as K, schedule_document
from scar.optimization.activity_time_projection import project_activity_time
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.scope_accounting import project_scope_accounting


@pytest.fixture
def cases(original_and_envelope, inputs, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('activity-time projection must not run a target, compiler or solver tool')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    original, _ = original_and_envelope
    first = replace(original.tasks[0], id='allocate:shadow', allocates=('shadow',))
    last = replace(original.tasks[0], allocates=('temp',), depends_on=(first.id,))
    free_temp = replace(original.tasks[-1], id='release:temp', releases=('temp',))
    free_shadow = replace(original.tasks[-1], releases=('shadow',), depends_on=(free_temp.id,))
    order = 'CPP_STD_ARRAY_REVERSE_DESTRUCTION'
    body = tuple(replace(t, validates=(first.id, *t.validates)) if t.kind is K.SUCCESS_FENCE else t
                 for t in original.tasks[1:-1])
    original = replace(original, tasks=(first, last, *body, free_temp, free_shadow),
        storage_slots=tuple(replace(s, allocated_by=first.id) if s.id == 'shadow' else
            replace(s, released_by=free_temp.id) if s.id == 'temp' else s for s in original.storage_slots),
        budget_scope='ALL_REGION_ARRAY_PAYLOAD_EXTERNAL_STATE_PLUS_FIXED_WORKSPACE',
        proof={**original.proof, 'pool_release_order': order,
            'physical_release_slot_order': ['temp', 'shadow']})
    dummy = tmp_path/'component.json'
    dummy.write_text(json.dumps({'schema': 'scar.test-component', 'median': 5}))
    component = artifact_reference(dummy, 'scar.test-component')
    recipe = make_recipe('cpu-stateful-array-region-v1',
        semantic_boundary={'id': original.region_id,
            'inputs': [{'id': 'input:0', 'kind': 'array', 'shape': [1, 2]}]},
        control={'fixed_source_order': True, 'spatial_partition': 'SERIAL_ROWS', 'pool_release_order': order},
        task_graph=schedule_document(original), resources={'fixture': 'STRUCTURAL_ONLY'},
        implementation={'family': 'CPP_SYNCHRONOUS_CPU_TRANSACTIONAL_STATEFUL_REGION',
            'backend_inputs': {'region': 'region', 'plan': 'stage_plan'},
            'backend_capability_record': {'implementation_sha256': implementation_sha256()}},
        artifacts={'region': component, 'stage_plan': component})
    schedule = derive_scope_control_milestone(attach_task_time(
        derive_stateful_scope_envelope(original, recipe)), recipe)
    _, target, _, _, context = inputs
    resources = derive_temporal_resources(schedule, target, source_context=context)
    saved = tmp_path/'scope.json'; saved.write_text(json.dumps(schedule_document(schedule)))
    scope_ref = artifact_reference(saved, 'scar.resource-schedule')
    fact = ProviderFactReference(ProviderArtifact(**component), '#/median')
    activity = {a.task: a.id for a in schedule.temporal.activities}
    pool_groups = {'pool:allocate': tuple(activity[t.id] for t in schedule.tasks if t.kind is K.ALLOCATE),
                   'pool:release': tuple(activity[t.id] for t in schedule.tasks if t.kind is K.RELEASE)}
    def build(unknown_task=None):
        grouped = [ActivityCost(name, members, B.DECLARED_COMPONENT_MODEL, 5, CostInterval(3, 8),
            component_refs=(fact,), charge_keys=('fixture:'+name,),
            conditions=('STRUCTURAL_FIXTURE_NOT_SOURCE_MEASUREMENT_ADMISSION',),
            missing_terms=('UNVALIDATED_IMPLEMENTATION_DELTA',)) for name, members in pool_groups.items()]
        covered = {a for row in grouped for a in row.activities}
        for task in schedule.tasks:
            if activity[task.id] in covered:
                continue
            if task.kind is K.SUCCESS_FENCE or task.id == unknown_task:
                row = ActivityCost('cost:'+task.id, (activity[task.id],), B.UNKNOWN,
                    missing_terms=('NO_QUALIFIED_COMPONENT',))
            else:
                row = replace(grouped[0], subject='cost:'+task.id, activities=(activity[task.id],),
                    charge_keys=('fixture:cost:'+task.id,))
            grouped.append(row)
        table = ActivityCostTable(semantic_sha256(schedule_document(schedule)),
            semantic_sha256(projection_document(resources)), resources.target, tuple(grouped))
        table, accounting = project_scope_accounting(table, schedule, resources, recipe,
            {'terms': []}, mapping=[], scope_schedule_ref=scope_ref)
        return table, accounting
    return schedule, resources, build, activity, pool_groups, scope_ref


def test_pool_subjects_stay_whole_residuals_stay_signed_and_original_ids_budget_survive(cases):
    schedule, resources, build, _, pools, scope_ref = cases
    table, accounting = build()
    projection = project_activity_time(schedule, resources, table, accounting,
        source_refs=({'role': 'scope_schedule', 'artifact': scope_ref, 'locator': '#'},))
    assert projection['group_costs_divided'] is False and projection['residuals_set_to_zero'] is False
    assert projection['physical_scope_latency'] is None and projection['summary']['physical_scope_latency'] is None
    assert projection['summary']['component_model_sum_ticks'] is not None
    assert not projection['solver_ready'] and not projection['deployment_authorized']
    groups = projection['groups']
    for subject, members in pools.items():
        group = next(g for g in groups if g['subject'] == subject)
        assert len(members) == 2 and group['activities'] == list(members)
        assert group['cost']['estimate_ticks'] == 5 and len(group['cost']['charge_keys']) == 1
    residuals = [g['residual'] for g in groups if g['residual'] is not None]
    assert residuals and all(r['signed_unknown'] and not r['set_to_zero']
        and not r['cancellation_assumed'] and not r['additional_work']
        and r['lower_ticks'] is None and r['upper_ticks'] is None for r in residuals)
    reserve = next(g for g in groups if g['subject'] == 'pool:allocate')
    release = next(g for g in groups if g['subject'] == 'pool:release')
    assert set(reserve['storage']['reserve_slots']) == set(release['storage']['release_slots']) == {'shadow', 'temp'}
    boundaries = {row['group']: row for row in projection['memory']['group_boundaries']}
    assert boundaries[reserve['id']]['owned_before_bytes'] == 0
    assert boundaries[reserve['id']]['owned_during_bytes'] == 32
    assert boundaries[release['id']]['owned_during_bytes'] == 32
    assert boundaries[release['id']]['owned_after_bytes'] == 0
    assert projection['memory']['peak_owned_bytes'] == 32
    assert projection['memory']['external_payload_bytes'] == 16
    assert projection['memory']['peak_total_payload_bytes'] == 48
    assert projection['memory']['budget_bytes'] == 48 and projection['memory']['budget_satisfied'] is True
    assert projection['references']['events'] == [e.id for e in schedule.temporal.events]
    assert projection['references']['constraints'] == [c.id for c in schedule.temporal.constraints]
    assert projection['references']['values'] == [v.id for v in schedule.values]
    assert projection['references']['materializations'] == [m.id for m in schedule.materializations]
    assert projection['references']['slots'] == [s.id for s in schedule.storage_slots]
    assert projection['references']['occupancies'] == [o.id for o in schedule.temporal.occupancies]
    assert {t for g in groups for t in g['task_ids']} == {t.id for t in schedule.tasks}
    for group in groups:
        for ready in group['produced_ready']:
            assert ready['conservatively_ready_at_event'] == group['exit_event']
            assert ready['original_ready_time_equality_claimed'] is False


def test_unknown_cost_blocks_following_coordinates_and_noncontiguous_groups_are_rejected(cases):
    schedule, resources, build, activity, _, _ = cases
    table, accounting = build(unknown_task='compute')
    projection = project_activity_time(schedule, resources, table, accounting)
    groups = projection['groups']
    index = next(i for i, g in enumerate(groups) if g['task_ids'] == ['compute'])
    unknown = groups[index]
    assert unknown['cost']['basis'] == 'UNKNOWN' and unknown['cost']['estimate_ticks'] is None
    assert unknown['component_model_coordinate']['start_ticks'] is not None
    assert unknown['component_model_coordinate']['end_ticks'] is None
    assert all(g['component_model_coordinate'] == {'start_ticks': None, 'end_ticks': None}
        for g in groups[index+1:])
    assert projection['summary']['component_model_sum_ticks'] is None
    assert projection['summary']['known_component_model_subtotal_ticks'] > 0
    zeros = [g for g in groups if g['cost']['estimate_ticks'] == 0]
    assert len(zeros) == 1 and zeros[0]['cost']['basis'] == 'SOURCE_CONTROL_MILESTONE'
    first, last = schedule.tasks[0].id, schedule.tasks[-1].id
    selected = {activity[first], activity[last]}
    old = [r for r in table.costs if selected.intersection(r.activities)]
    crossing = ActivityCost('noncontiguous', tuple(sorted(selected)), B.UNKNOWN,
        missing_terms=('STRUCTURAL_NONCONTIGUOUS_GROUP_NEGATIVE_CASE',))
    bad = replace(table, costs=(crossing, *(r for r in table.costs if r not in old)))
    # Preserve valid structural coverage/accounting so the targeted rejection
    # is the group split, rather than an unrelated stale summary.
    ledger = {**accounting, 'active_subjects': [r.subject for r in bad.costs],
        'numeric_unknown_subjects': [r.subject for r in bad.costs if r.basis is B.UNKNOWN],
        'summary': validate_activity_cost_table(bad, schedule, resources)}
    with pytest.raises(ValueError, match='noncontiguous collective'):
        project_activity_time(schedule, resources, bad, ledger)
