"""Retain only the committed private-shadow ROI between synchronous steps.

COMMIT supplies bitwise correspondence, not finiteness or invariant values.
Provider silence between calls is an explicit existing synchronous-profile Q,
never inferred from return status, NoPyAPI, or visible-IR escape analysis.
"""
from scar.compiler.stateful_model import validate_stateful_region, stateful_digest
from scar.ir.record_codec import encode
from scar.ir.versioned_schedule import VersionedTaskKind as K
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.stateful_schedule import _normalized
from scar.optimization.workspace_guard_properties import _initialization

SCHEMA = 'scar.workspace-shadow-residency-plan'


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _area(rectangle):
    return max(0, rectangle[0][1]-rectangle[0][0])*max(0, rectangle[1][1]-rectangle[1][0])


def _complement(shape, rectangle):
    rows, cols = shape; (r0, r1), (c0, c1) = rectangle
    _need(0 <= r0 <= r1 <= rows and 0 <= c0 <= c1 <= cols, 'committed ROI exceeds the original root')
    candidates = [[[0, r0], [0, cols]], [[r1, rows], [0, cols]],
                  [[r0, r1], [0, c0]], [[r0, r1], [c1, cols]]]
    return [r for r in candidates if _area(r)]


def derive_workspace_shadow_residency(region, step_plan, workspace_plan, execution_certificate,
        native_metadata, *, source_refs=()):
    """Derive one-rectangle retention plans; unsupported roots keep full copies."""
    region = validate_stateful_region(region)
    _need(type(source_refs) is tuple and all(type(r) is dict for r in source_refs), 'original source reference tuple required')
    region_sha = stateful_digest(region); step_sha = semantic_sha256(step_plan); workspace_sha = semantic_sha256(workspace_plan)
    blueprint = execution_certificate['native_blueprint']
    _need((execution_certificate.get('schema'), execution_certificate.get('schema_version'))
              == ('scar.workspace-execution-refinement', 1)
          and execution_certificate['execution_scope'] == 'WORKSPACE_LIFETIME_ONLY'
          and execution_certificate['inputs']['workspace_plan_sha256'] == workspace_sha
          and execution_certificate['inputs']['step_plan_sha256'] == step_sha
          and blueprint['workspace_plan_sha256'] == native_metadata['workspace_plan_sha256'] == workspace_sha
          and blueprint['execution_plan_sha256'] == native_metadata['execution_plan_sha256']
          and blueprint['base_ids'] == native_metadata['base_ids']
          and blueprint['base_ids']['region_sha256'] == workspace_plan['step_region_sha256'] == region_sha
          and blueprint['base_ids']['plan_sha256'] == workspace_plan['step_plan_sha256'] == step_sha
          and blueprint['cpp_source']['sha256'] == native_metadata['sha256']['source']
          and blueprint['source_mapping'] == native_metadata['source_mapping']
          and native_metadata['schema'] == 'scar.native-stateful-workspace-repeat',
          'exact original baseline workspace source/region/plan/execution identities required')
    count = workspace_plan['trip_count']
    _need(type(count) is int and 0 <= count < 2**64 and count == native_metadata['trip_count']
          == execution_certificate['trip_count'], 'exact source sequence count differs')
    rejections = []
    def reject(code, detail):
        rejections.append(dict(code=code, detail=detail))
    if count <= 1:
        reject('NO_REPEATED_SNAPSHOT', 'zero/one step has no repeated snapshot to elide')
    if step_plan['cpu_threads'] != 1 or native_metadata['cpu_threads'] != 1:
        reject('NOT_SERIAL_PROFILE', 'only the original serial workspace execution is covered')
    required = {'SYNCHRONOUS_CPU_EXECUTION', 'DISTINCT_ARRAY_INPUTS_HAVE_DISJOINT_STORAGE',
        'ORDERED_TRANSACTIONAL_SHADOW_WRITES_BEFORE_COMMIT',
        'ALL_SHAPES_ALIAS_BOUNDS_AND_WORKSPACE_CHECKED_BEFORE_EXTERNAL_WRITE'}
    loop_required = {'IDENTICAL_ARRAY_IDENTITIES_DESCRIPTORS_SCALARS_AND_PROVIDERS_EACH_STEP',
                    'NO_INTERSTEP_PYTHON_CODE_HOOKS_OBSERVERS_OR_CONCURRENT_STATE_ACCESS'}
    if (not required <= set(step_plan['guard_contract'])
            or not loop_required <= set(workspace_plan['required_live_guards'])
            or not {'SYNCHRONOUS_CPU_EXECUTION', 'NO_FOREIGN_CONCURRENT_STATE_READERS_OR_WRITERS'}
                <= set(workspace_plan['external_assumptions'])
            or step_plan['alias_contract'] != 'PAIRWISE_DISJOINT_ARRAY_INPUT_STORAGE_AND_PRIVATE_WORKSPACE'):
        reject('ISOLATION_OR_MEMORY_CONTRACT_MISSING', 'stable bounded roots and a synchronous exclusive sequence are required')
    transaction = step_plan['transaction']
    if (transaction['external_writes_before_success'] is not False
            or transaction['commit'] != 'ORIGINAL_WRITE_ROIS_IN_SOURCE_ORDER_FROM_FINAL_PRIVATE_SHADOWS'
            or transaction['no_failing_operations_or_GuardFailure_after_first_external_commit'] is not True):
        reject('COMMIT_TRANSACTION_CONTRACT_MISSING', 'original successful ROI copies and failed-step transaction are required')
    stages = step_plan['stages']; seen_commit = False
    for stage in stages:
        if stage['kind'] == 'COMMIT':
            seen_commit = True
        elif seen_commit:
            reject('PAYLOAD_WORK_AFTER_COMMIT', 'all source computational payload work must precede the commit tail')
            break
    if any(s['kind'] not in {'SNAPSHOT_ROOT','SCALAR','COMPUTE','PROVIDER','WRITE_SHADOW','COMMIT'} for s in stages):
        reject('OPAQUE_STAGE_FOOTPRINT', 'an unsupported stage may access retained private data')
    graph, initialized_reads = _initialization(workspace_plan)
    materials = {m.id: m for m in graph.materializations}; tasks = {t.id: t for t in graph.tasks}
    positions = {t.id: i for i,t in enumerate(graph.tasks)}; slots = {s.id: s for s in graph.storage_slots}
    arrays = [r.id for r in region.inputs if r.kind == 'array']
    slot_order = [s['id'] for s in step_plan['workspace_slots']]
    value_rows = {r['id']: r for r in step_plan['values']}
    slot_lifetimes = {r['slot']: r for r in execution_certificate['slot_map']}
    actions = {a.id: a for a in region.actions if a.kind == 'write'}
    original_values = {v.id: v for v in graph.values}
    loop_inputs = {r['input_id']: r for r in workspace_plan['loop_model']['inputs']}
    carries = execution_certificate['body']['content_carries']
    snapshots = []
    for stage in stages:
        if stage['kind'] != 'SNAPSHOT_ROOT':
            continue
        target, shape, payload = stage['target'], list(stage['shape']), stage['output_payload']
        slot = step_plan['assignment'][payload]['slot_id']; full = [[0,shape[0]],[0,shape[1]]]
        _need(target in arrays and shape == step_plan['input_shapes'][target]
              and payload == step_plan['shadows'][target] and slot in slot_order,
              'snapshot must retain its exact root input and physical shadow assignment')
        commits = [c for c in step_plan['commit_rows'] if c['target'] == target]
        commit_ids = [c['stage_id'] for c in commits]
        reasons = []; normalized = []
        for commit in commits:
            action = actions.get(commit['action_id'])
            _need(action is not None and action.target == target
                  and _normalized(action.bounds, shape)[0] == commit['normalized_bounds'],
                  'commit ROI must come from its original typed write action')
            actual = next(s for s in stages if s['id'] == commit['stage_id'])
            _need(actual['kind'] == 'COMMIT' and actual['target'] == target
                  and actual['normalized_bounds'] == commit['normalized_bounds']
                  and actual['input_payloads'] == [payload] and actual['source_action_ids'] == [action.id],
                  'commit must bit-copy the same original final shadow ROI')
            rectangle = [list(axis) for axis in commit['normalized_bounds']]
            if rectangle not in normalized:
                normalized.append(rectangle)
        if len(normalized) != 1:
            reasons.append('MULTIPLE_DISTINCT_OR_MISSING_COMMIT_RECTANGLES')
        roi = normalized[0] if len(normalized) == 1 else None
        complement = _complement(shape, roi) if roi is not None else [full]
        full_bytes = _area(full)*8; later_bytes = sum(_area(r)*8 for r in complement)
        if later_bytes == full_bytes:
            reasons.append('NO_POSITIVE_COMMITTED_REGION')
        shadow_materials = [m for m in graph.materializations if m.ready_task == stage['id']]
        _need(len(shadow_materials) == 1 and shadow_materials[0].region.slot == slot
              and shadow_materials[0].region.offset_bytes == 0
              and shadow_materials[0].region.nbytes == full_bytes
              and slots[slot].ownership == 'SCHEDULE' and slots[slot].capacity_bytes >= full_bytes,
              'snapshot full-root material and private capacity differ')
        lifetime = slot_lifetimes.get(slot)
        if lifetime is None or lifetime['physical_count'] != 1:
            reasons.append('SHADOW_ALLOCATION_NOT_HELD_FOR_THE_SEQUENCE')
        clobbers = []
        if commit_ids:
            last_commit = max(positions[c] for c in commit_ids)
            for task in graph.tasks:
                phase = ('AFTER_LAST_ROOT_COMMIT' if positions[task.id] > last_commit else
                         'NEXT_ITERATION_BEFORE_ROOT_SNAPSHOT' if positions[task.id] < positions[stage['id']] else None)
                if phase is not None:
                    for name in task.writes:
                        region_ = materials[name].region
                        if region_.slot == slot and region_.offset_bytes < full_bytes and region_.nbytes:
                            clobbers.append(dict(task=task.id, materialization=name, phase=phase))
            # A different value cannot overwrite a live shadow in the current
            # body and still supply the claimed source bytes at COMMIT.
            for task in graph.tasks[positions[stage['id']]+1:last_commit+1]:
                for name in task.writes:
                    region_ = materials[name].region
                    source_stage = next((s for s in stages if s['id'] == task.id), None)
                    own_state_write = (task.kind is K.STATE_WRITE and source_stage is not None
                                       and source_stage['target'] == target)
                    if region_.slot == slot and region_.nbytes and not own_state_write:
                        clobbers.append(dict(task=task.id, materialization=name, phase='SHADOW_SLOT_REPURPOSED_BEFORE_COMMIT'))
        if clobbers:
            reasons.append('SHADOW_SLOT_CLOBBERED_BETWEEN_CORRESPONDENCE_AND_REUSE')
        incoming = [m for m in graph.materializations if m.external_ready
                    and original_values[m.value].attrs.get('input_id') == target]
        matching_carry = [c for c in carries if len(incoming) == 1 and c['after'] == incoming[0].id]
        if (len(matching_carry) != 1 or matching_carry[0]['iteration_distance'] != 1
                or materials[matching_carry[0]['before']].ready_task not in commit_ids
                or materials[matching_carry[0]['before']].region != incoming[0].region
                or loop_inputs[target]['role'] != 'ORDERED_ARRAY_CONTENT_CARRY'):
            reasons.append('ORIGINAL_COMMITTED_CONTENT_CARRY_MISSING')
        mode = ('KEEP_FULL_SNAPSHOT' if reasons or rejections else
                'SKIP_AFTER_FIRST' if not complement else 'REFRESH_COMPLEMENT')
        snapshots.append(dict(snapshot_stage_id=stage['id'], input_id=target, array_index=arrays.index(target),
            shape=shape, slot_id=slot, slot_index=slot_order.index(slot), output_payload=payload,
            commit_stage_ids=commit_ids, committed_roi=roi, complement_rectangles=complement,
            mode=mode, first_iteration_nbytes=full_bytes,
            later_iteration_nbytes=full_bytes if mode == 'KEEP_FULL_SNAPSHOT' else later_bytes,
            proof=dict(property='COMMITTED_ROI_BITWISE_CORRESPONDENCE_NOT_CONSTANT_VALUE',
                commit_source='SAME_PRIVATE_SHADOW_AND_EXACT_ORIGINAL_MEMCPY_ROI',
                content_carry_ref=None if not matching_carry else dict(artifact_role='execution_certificate',
                    pointer='#/body/content_carries/'+str(carries.index(matching_carry[0]))),
                current_iteration_initialization_ref=dict(artifact_role='workspace_plan', pointer='#/initialization_proof'),
                shadow_value_record=dict(artifact_role='step_plan', pointer='#/values/'+str(list(value_rows).index(payload))),
                slot_tail_and_next_prefix_clobbers=clobbers, complement_correspondence_claimed=False,
                later_refresh_establishes_full_root_snapshot_equivalence=True,
                no_finiteness_or_provider_purity_inferred=True), rejections=reasons))
    applicable = count > 1 and any(s['mode'] != 'KEEP_FULL_SNAPSHOT' for s in snapshots)
    if not applicable and not rejections:
        reject('NO_PROVED_POSITIVE_RESIDENCY_REGION', 'all roots retain their original full snapshots')
    provider_q = []
    for i, provider in enumerate(region.providers):
        provider_q.append(dict(provider_id=provider.id, source_ref=dict(artifact_role='region', pointer='#/providers/'+str(i)),
            condition='ALL_BORROWED_PORT_ACCESSES_COMPLETE_BEFORE_PROVIDER_RETURN_AND_NO_DELAYED_OR_ASYNC_PAYLOAD_ACCESS_BEFORE_NEXT_ORIGINAL_PROVIDER_CALL',
            authority='EXTERNAL_ASSUMPTION', parent_condition='SYNCHRONOUS_CPU_EXECUTION',
            source_proven=False, runtime_checkable=False, no_pyapi_or_return_status_is_not_evidence=True,
            borrowed_ports=[dict(task=t.id, reads=list(t.reads), writes=list(t.writes),
                regions={name:encode(materials[name].region) for name in (*t.reads,*t.writes)},
                source_refs=encode(t.source_refs),
                declared_read_write_roles_are_not_a_provider_purity_proof=True)
                for t in graph.tasks if t.attrs.get('provider_id') == provider.id],
            pointer_retention_without_interval_access_is_not_prohibited=True))
    source_mode = workspace_plan['source_f4_authority']['mode']
    _need(source_mode in ('ORIGINAL_SOURCE_LOOP_PLAN_BOUND','EXPLICIT_FIXED_INPUT_COMPOSITION'), 'explicit original source authority required')
    original = count*sum(s['first_iteration_nbytes'] for s in snapshots)
    proposed = sum(s['first_iteration_nbytes'] for s in snapshots)+(count-1)*sum(s['later_iteration_nbytes'] for s in snapshots) if count else 0
    return encode(dict(schema=SCHEMA, schema_version=1, status='PROVEN_CONDITIONAL' if applicable else 'REJECTED',
        native_blueprint=blueprint, trip_count=count, snapshots=snapshots,
        inputs=dict(region_sha256=region_sha, step_plan_sha256=step_sha, workspace_plan_sha256=workspace_sha,
            execution_certificate_sha256=semantic_sha256(execution_certificate), source_refs=source_refs),
        provider_boundary_Q=provider_q,
        conditions=dict(required_guards=list(workspace_plan['required_live_guards']),
            external_assumptions=list(workspace_plan['external_assumptions']),
            source_qualified_normal_success=True, private_control_and_completed_storage_disjoint_from_payload_required=True,
            synchronous_profile_Q_includes_provider_borrowed_memory=True,
            no_interstep_or_foreign_payload_writer=True, no_provider_purity_claimed=True,
            provider_access_after_return_counterexample_invalidates_profile=True),
        initialization=dict(private_reads_replayed=initialized_reads, first_iteration='ORIGINAL_FULL_SNAPSHOTS',
            later_iteration='COMMITTED_ROI_PLUS_REFRESHED_COMPLEMENT_REESTABLISHES_ORIGINAL_FULL_SNAPSHOT',
            stale_uninitialized_private_data_never_consumed=True),
        preserved=['ALL_INPUT_FINITE_SCANS_EVERY_ITERATION','ALL_SCALAR_ABI_SHAPE_ALIAS_MEMORY_AND_PROVIDER_GUARDS',
            'ORIGINAL_FP_CPU_CONTEXT_AND_RESTORATION','ORIGINAL_PROVIDER_AND_ARITHMETIC_CALLS',
            'ORIGINAL_SHADOW_WRITE_ORDER_AND_EVERY_COMMIT','EACH_SUCCESSFUL_STEP_CURSOR_AND_UNCOMMITTED_FAILURE_SUFFIX'],
        source_authority=dict(mode=source_mode, automatic_caller_replacement_authorized=False,
            original_loop_guard_and_export_required=source_mode == 'ORIGINAL_SOURCE_LOOP_PLAN_BOUND',
            fixed_input_composition_does_not_authorize_caller_replacement=source_mode == 'EXPLICIT_FIXED_INPUT_COMPOSITION'),
        work_counts=dict(basis='SOURCE_COPY_GEOMETRY_NOT_RUNTIME_COUNTERS', original_snapshot_bytes=original,
            proposed_snapshot_bytes=proposed, avoided_snapshot_bytes=original-proposed),
        rejections=rejections, content_is_loop_invariant=False, finiteness_inferred=False,
        current_source_admission_required=True, source_admission_performed_by_this_function=False,
        trial_only=True, selected_for_use=False, cost_status='UNKNOWN', default_runtime_decision='NO_OP',
        profitability_proven=False, deployment_authorized=False, target_program_executed=False))


def verify_workspace_shadow_residency(region, step_plan, workspace_plan, execution_certificate,
        native_metadata, residency_plan, *, source_refs=()):
    expected = derive_workspace_shadow_residency(region, step_plan, workspace_plan,
        execution_certificate, native_metadata, source_refs=source_refs)
    _need(residency_plan == expected, 'shadow residency plan differs from complete ROI/slot/carry derivation')
    return expected


__all__ = ['SCHEMA','derive_workspace_shadow_residency','verify_workspace_shadow_residency']
