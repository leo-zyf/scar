"""Conditional replica lifetime across explicit step/invalidation operations."""
from copy import deepcopy

from scar.ir.workspace_session import SESSION_PROTOCOL, SESSION_SYMBOLS
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.workspace-session-plan'


def derive_workspace_session(residency_plan, step_plan, native_metadata, *, source_refs=()):
    if (residency_plan.get('schema') != 'scar.workspace-shadow-residency-plan'
            or residency_plan.get('schema_version') != 1 or residency_plan.get('status') != 'PROVEN_CONDITIONAL'
            or residency_plan['native_blueprint']['execution_plan_sha256'] != native_metadata['execution_plan_sha256']
            or residency_plan['inputs']['step_plan_sha256'] != semantic_sha256(step_plan)
            or residency_plan['inputs']['workspace_plan_sha256'] != native_metadata['workspace_plan_sha256']):
        raise ValueError('source-qualified per-step COMMIT/ROI correspondence required')
    snapshots = [dict(input_id=row['input_id'], array_index=row['array_index'], slot_id=row['slot_id'],
        shape=deepcopy(row['shape']), committed_roi=deepcopy(row['committed_roi']),
        complement_rectangles=deepcopy(row['complement_rectangles']), mode=row['mode'])
        for row in residency_plan['snapshots']]
    geometry = dict(snapshots=snapshots, slot_capacities_bytes=[s['capacity_bytes'] for s in step_plan['workspace_slots']],
        memory=deepcopy(native_metadata['memory']), original_array_count=len(step_plan['input_shapes']))
    return dict(schema=SCHEMA, schema_version=1, status='PROVEN_CONDITIONAL',
        parent_native_manifest=residency_plan['native_blueprint']['manifest'],
        native_blueprint=deepcopy(residency_plan['native_blueprint']),
        residency_plan_sha256=semantic_sha256(residency_plan), geometry=geometry,
        protocol=dict(SESSION_PROTOCOL), symbols=dict(SESSION_SYMBOLS),
        validity_invariant='EVERY_RETAINED_COMMIT_ROI_EQUALS_CURRENT_EXTERNAL_BYTES_AT_STEP_ENTRY',
        transitions=dict(create='UNBOUND_DIRTY', successful_step='ESTABLISH_COMMIT_ROI_REPLICA_CORRESPONDENCE',
            known_write='KEEP_ONLY_IF_QUALIFIED_ACTUAL_ROOT_WRITE_SET_DISJOINT_FROM_ALL_RETAINED_ROIS',
            unknown_effect='INVALIDATE_OR_DESTROY_BEFORE_EFFECT', binding_change='DIRTY_FULL_SNAPSHOT',
            failed_step='DIRTY_AND_PYTHON_FALLBACK_CURRENT_CALL_ONLY', destroy='NO_REPLICA_FACT_REMAINS'),
        array_binding_identity_and_owner_lifetime_required=True,
        descriptor_equality_is_not_content_equality=True, scalar_values_reloaded_each_call=True,
        provider_boundary_Q=deepcopy(residency_plan['provider_boundary_Q']),
        source_refs=deepcopy(list(source_refs)), original_fixed_trip_count_is_not_session_call_count=True,
        original_caller_replacement_authorized=False, source_caller_effect_coverage_proven=False,
        local_call_events_are_not_all_memory_writes=True, runtime_protocol_required=True,
        trial_only=True, selected_for_use=False, default_backend='NO_OP', deployment_authorized=False)


def copy_effect_preserves_session(session_plan, summary, *, formal_index, input_id):
    """Geometric implication; actual/formal and live object binding stay guards."""
    from scar.analysis.array_copy_effects import prove_copy_writes_disjoint
    roots = [r for r in session_plan['geometry']['snapshots'] if r['input_id'] == input_id
             and r['mode'] != 'KEEP_FULL_SNAPSHOT']
    if len(roots) != 1:
        raise ValueError('one retained root required for write-effect reasoning')
    proof = prove_copy_writes_disjoint(summary, formal_index=formal_index, rectangles=[roots[0]['committed_roi']])
    return dict(schema='scar.session-write-compatibility', schema_version=1,
        session_plan_sha256=semantic_sha256(session_plan), input_id=input_id, copy_summary_id=summary['id'],
        formal_index=formal_index, geometry_proof=proof,
        decision='PRESERVE_UNDER_ACTUAL_ROOT_GUARDS' if proof['status'] == 'DISJOINT_UNDER_ROOT_GUARDS' else 'INVALIDATE',
        actual_formal_and_array_binding_required=True, caller_replacement_authorized=False)
