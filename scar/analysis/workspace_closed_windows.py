"""Use the existing closed-window core within one admitted shared iteration.

Outer allocation/release, symbolic carries and FIRST/LAST boundaries remain
parent graph facts. No iteration is expanded and no recipe is manufactured.
"""
from copy import deepcopy
from pathlib import Path

from scar.analysis.closed_windows import project_closed_windows, _source_contracts
from scar.analysis.workspace_time import derive_workspace_time
from scar.ir.execution_recipe import artifact_reference
from scar.ir.record_codec import encode
from scar.ir.temporal_resources import OriginalBudget, projection_document, projection_from_dict
from scar.ir.versioned_schedule import (TemporalMode, TemporalPathCondition,
    TemporalScopeKind as S, TemporalOccupancyKind as O, VersionedTaskKind as K,
    schedule_document, validate_versioned_schedule)
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.workspace_execution_refinement import _replay_domains


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def derive_workspace_closed_windows(schedule, resources, workspace_plan, step_plan,
        native_metadata, execution_certificate, *, source_refs=()):
    """Project same-i windows after the caller's full source/target/cost replay."""
    validate_versioned_schedule(schedule)
    wire = schedule_document(schedule); temporal = schedule.temporal
    _need(wire['schema_version'] == 4 and not schedule.executable
          and temporal.mode is TemporalMode.SEMANTIC
          and temporal.path_condition is TemporalPathCondition.NORMAL_COMPLETE,
          'one original normal-complete semantic v4 workspace graph required')
    resources = projection_from_dict(projection_document(resources))
    digest = semantic_sha256(wire); resource_sha = semantic_sha256(projection_document(resources))
    _need(execution_certificate.get('schema') == 'scar.workspace-execution-refinement'
          and execution_certificate.get('schema_version') == 1
          and execution_certificate['inputs']['source_schedule_sha256'] == digest
          and execution_certificate['inputs']['resource_projection_sha256'] == resource_sha
          and execution_certificate['inputs']['workspace_plan_sha256'] == semantic_sha256(workspace_plan)
          and execution_certificate['inputs']['step_plan_sha256'] == semantic_sha256(step_plan),
          'exact current execution refinement/source/resource/step identities required')
    proofs = [r for r in temporal.source_refs if r.get('schema') == 'scar.workspace-cpu-demand-source-proof']
    _need(len(proofs) == 1, 'one source-replayed workspace CPU domain proof required')
    proof = proofs[0]; blueprint = execution_certificate['native_blueprint']
    _need(blueprint['manifest'] == proof['native_manifest'] and blueprint['cpp_source'] == proof['cpp_source']
          and blueprint['workspace_plan_sha256'] == native_metadata['workspace_plan_sha256']
          and blueprint['execution_plan_sha256'] == native_metadata['execution_plan_sha256']
          and blueprint['base_ids'] == native_metadata['base_ids']
          and blueprint['workspace_plan_sha256'] == semantic_sha256(workspace_plan)
          and blueprint['cpp_source']['sha256'] == native_metadata['sha256']['source']
          and blueprint['base_ids']['plan_sha256'] == semantic_sha256(step_plan)
          and blueprint['base_ids']['region_sha256'] == workspace_plan['step_region_sha256']
          and blueprint['source_mapping'] == native_metadata['source_mapping']
          and proof['target'] == encode(resources.target) == step_plan['resource_context']['target_snapshot']
          and resources.source_schedule_sha256 == digest
          and resources.budget == OriginalBudget(schedule.budget_bytes, schedule.budget_scope, schedule.budget_refs),
          'original workspace backend/target/budget differs')
    baseline = derive_workspace_time(workspace_plan, native_manifest=proof['native_manifest'],
        native_metadata=native_metadata, cpp_source=proof['cpp_source'], workspace_reference=proof['workspace_plan'])
    _replay_domains(baseline, schedule, proof, step_plan)
    loops = [s for s in temporal.scopes if s.kind is S.LOOP]
    roots = [s for s in temporal.scopes if s.kind is S.ROOT]
    _need(len(loops) == len(roots) == 1 and loops[0].parent == roots[0].id
          and loops[0].trip_count == workspace_plan['trip_count'] > 0,
          'one positive fixed-count shared loop inside its original root required')
    loop, root = loops[0], roots[0]
    activities = {a.task: a for a in temporal.activities}
    _need(len(activities) == len(temporal.activities) == len(schedule.tasks)
          and set(activities) == {t.id for t in schedule.tasks}, 'complete original activity/task interface required')
    certificate = schedule.proof['workspace_time']
    body_ids = [t.id for t in schedule.tasks if activities[t.id].scope == loop.id]
    outer_ids = [t.id for t in schedule.tasks if activities[t.id].scope == root.id]
    _need(body_ids == certificate['body_task_ids'] == execution_certificate['body']['task_ids']
          and outer_ids == certificate['outer_task_ids'] == execution_certificate['outer']['task_ids']
          and set(body_ids) | set(outer_ids) == set(activities)
          and loop.iteration_local_values == tuple(v.id for v in schedule.values)
          and loop.iteration_local_materializations == tuple(m.id for m in schedule.materializations)
          and encode(loop.content_carries) == execution_certificate['body']['content_carries'],
          'shared body, local content templates or carry ports differ')
    allocations = [t for t in schedule.tasks if t.kind is K.ALLOCATE]
    releases = [t for t in schedule.tasks if t.kind is K.RELEASE]
    _need(all(activities[t.id].scope == root.id for t in (*allocations, *releases))
          and [s for t in allocations for s in t.allocates] == workspace_plan['allocation_order']
          and [s for t in releases for s in t.releases] == workspace_plan['release_order']
              == list(reversed(workspace_plan['allocation_order'])),
          'workspace physical allocation/reverse release must stay outside the loop')
    domains = {r['occupancy']: r for r in proof['occupancies']}
    _need(len(resources.executions) == len(domains)
          and {r.occupancy for r in resources.executions} == set(domains), 'complete source CPU domains required')
    for row in resources.executions:
        domain = domains[row.occupancy]
        _need((row.resource, row.units) == (domain['resource'], domain['units'])
              and (encode(row.exact_thread) == proof['selected_thread'] and row.candidate_threads == (row.exact_thread,)
                   if domain['exact_cpu_id'] is not None
                   else row.exact_thread is None and not row.candidate_threads),
              'hardware execution binding differs from its source scope')
    physical = {o.storage_slot: o for o in temporal.occupancies if o.kind is O.PHYSICAL_SLOT_BYTES}
    _need({s.id for s in schedule.storage_slots if s.ownership == 'SCHEDULE'} == set(physical)
          and len(resources.storages) == len(physical)
          and {s.slot for s in resources.storages} == set(physical)
          and all(s.occupancy == physical[s.slot].id and s.payload_bytes == physical[s.slot].units
                  and s.exact_memory is None for s in resources.storages),
          'one unbound original physical lifetime per private slot required')
    contracts = _source_contracts()
    contracts['workspace_serial'] = deepcopy(proof['source_contract'])
    contracts['workspace_adapter_rule'] = artifact_reference(Path(__file__).resolve(), 'scar.closed-window-source-rule')
    enter = [t for t in schedule.tasks if t.attrs.get('phase_kind') == proof['source_contract']['enter_phase_kind']]
    _need(len(enter) == 1 and activities[enter[0].id].scope == loop.id,
          'one source-qualified per-iteration bind-completion anchor required')
    contracts['anchor_proofs'] = {enter[0].id: [dict(
        condition='SCAR_SYNCHRONOUS_CALLING_TID_BIND_COMPLETED_ON_NORMAL_RETURN',
        source_ref='#/source_contracts/workspace_serial', authority='EXACT_SOURCE_STATEMENT_AND_HELPER',
        iteration_scope=loop.id, iteration_symbol=loop.iteration_symbol)]}
    result = project_closed_windows(schedule, resources, plan=step_plan, activities=activities,
        contracts=contracts, external_assumptions=workspace_plan['external_assumptions'],
        assumption_source=dict(parent_document='workspace_plan', source_ref='#/external_assumptions'),
        source_source=dict(workspace_plan_id=workspace_plan['id'],
            execution_certificate_sha256=semantic_sha256(execution_certificate),
            artifacts=dict(workspace_plan=proof['workspace_plan'], native_manifest=proof['native_manifest'],
                cpp_source=proof['cpp_source'], stage_plan=proof['stage_plan'], target_snapshot=proof['target']),
            source_scope=dict(kind='SHARED_LOOP_TEMPLATE', scope=loop.id,
                workspace_time_proof_ref='#/proof/workspace_time')),
        source_documents=dict(workspace_plan=dict(id=workspace_plan['id'], artifact=proof['workspace_plan']),
            native_manifest=proof['native_manifest']), scope=loop.id, source_refs=source_refs)
    scope = dict(id=loop.id, kind='LOOP', iteration_symbol=loop.iteration_symbol,
        trip_count=loop.trip_count, occurrence_count=loop.trip_count, symbolic_same_i=True)
    lifetime_rows = []
    for slot in schedule.storage_slots:
        if slot.id not in physical:
            continue
        occupancy = physical[slot.id]; allocate, release = activities[slot.allocated_by], activities[slot.released_by]
        _need(occupancy.scope == root.id, 'shared workspace lifetime must remain owned by the root')
        lifetime_rows.append(dict(slot=slot.id, owner_scope=root.id, physical_count=1,
            allocation=dict(task=slot.allocated_by, start=allocate.start, completion=allocate.completion),
            release=dict(task=slot.released_by, start=release.start, completion=release.completion),
            occupancy=dict(id=occupancy.id, start=occupancy.start, end=occupancy.end),
            capacity_bytes=slot.capacity_bytes, unchanged=True))
    carries = encode(loop.content_carries)
    result['scope'] = scope
    result['shared_loop'] = dict(**scope, entry=loop.entry, exit=loop.exit,
        iteration_local_values=list(loop.iteration_local_values),
        iteration_local_materializations=list(loop.iteration_local_materializations), content_carries=carries,
        iteration_constraints=[dict(ordinal=i, id=e.id, before=e.before, after=e.after,
            iteration_distance=e.iteration_distance, source_refs=encode(e.source_refs))
            for i, e in enumerate(temporal.constraints) if e.loop_scope == loop.id],
        selected_boundary_constraint_ordinals=[i for i,e in enumerate(temporal.constraints)
            if e.before_iteration is not None or e.after_iteration is not None],
        iterations_unrolled=False, cross_iteration_optimization_authorized=False)
    result['shared_workspace_lifetimes'] = lifetime_rows
    by_slot = {r['slot']: i for i, r in enumerate(lifetime_rows)}
    for window in result['windows']:
        window['scope'] = deepcopy(scope)
        touched = set()
        for role in ('live_inputs', 'live_outputs', 'preserved_unconsumed_outputs'):
            for port in window[role]:
                touched.add(port['materialization'])
                port['iteration_coordinate'] = dict(scope=loop.id, iteration_symbol=loop.iteration_symbol,
                    local_materialization_version=port['version'],
                    version_kind='ORIGINAL_STEP_SLOT_VERSION_NOT_GLOBAL_LOGICAL_VERSION')
                port['content_carry_refs'] = ['#/shared_loop/content_carries/'+str(i)
                    for i,c in enumerate(carries) if port['materialization'] in (c['before'],c['after'])]
        for borrowed in window['borrowed_slots']:
            slot = borrowed['slot']['id']
            borrowed['root_lifetime_ref'] = '#/shared_workspace_lifetimes/'+str(by_slot[slot]) if slot in by_slot else None
        window['shared_loop_ref'] = '#/shared_loop'
        window['relevant_content_carry_ordinals'] = [i for i,c in enumerate(carries) if touched.intersection((c['before'],c['after']))]
        window['frontiers'].append(dict(reason='PARENT_ITERATION_CARRY_AND_FIRST_LAST_BOUNDARIES_RETAINED',
            anchor=loop.id, cross_iteration_movement_authorized=False))
    result['summary'].update(shared_body_count=1, occurrence_count=loop.trip_count, iterations_unrolled=0,
        selected_scope_tasks=len(body_ids), root_allocation_tasks=len(allocations), root_release_tasks=len(releases))
    result['hazard_iteration_interpretation'] = 'SAME_SYMBOLIC_i_ONLY_NOT_CROSS_ITERATION_ORDER_ELISION'
    result['current_source_target_and_execution_admission_required_by_caller'] = True
    return result


__all__ = ['derive_workspace_closed_windows']
