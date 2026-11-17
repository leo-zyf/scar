"""Private capacity epochs over original dynamic source activations.

Capacity survives ordinary host intervals; array owners, ABI bindings and
content correspondence do not.  This is a conditional lifecycle overlay, not
a time schedule, solver result or authority to run an unimplemented protocol.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.record_codec import decode, encode
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.session_activation import verify_session_activation
from scar.optimization.session_lifetime import session_lifetime_from_dict

SCHEMA = 'scar.session-capacity-plan'
_INPUTS = ('session_plan', 'step_plan', 'workspace_plan',
           'parent_native_metadata', 'session_native_metadata')


@dataclass(frozen=True, slots=True)
class CapacityAction:
    id: str
    purpose: str
    source_action: ProviderFactReference | None = None
    duration_ticks: int | None = None
    execution_units: int | None = None
    exact_cpu: None = None


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _reference(value, schema):
    ref = value if type(value) is ProviderArtifact else decode(ProviderArtifact, value)
    _need(ref.schema == schema and Path(ref.uri).is_absolute(), 'original absolute '+schema+' reference required')
    return ref


def _fact(reference, locator):
    return ProviderFactReference(reference, locator)


def derive_session_capacity(outer_profile, activation_projection, lifetime_projection, capacity_proof, *,
        outer_profile_ref, activation_ref, lifetime_ref, capacity_proof_ref,
        capacity_inputs, source_refs=()):
    """Purely replay source/lease and typed initialization parents, then overlay.

    The publisher must cold-admit the actual source/native/resource receipts.
    The five original documents are required to rederive the capacity proof;
    a status, numeric estimate, or self-consistent proof hash is insufficient.
    """
    from scar.analysis.session_capacity_proof import derive_capacity_proof
    outer_ref = _reference(outer_profile_ref, 'scar.guarded-session-outer')
    active_ref = _reference(activation_ref, 'scar.workspace-session-activation')
    lease_ref = _reference(lifetime_ref, 'scar.workspace-session-lifetime')
    proof_ref = _reference(capacity_proof_ref, 'scar.session-capacity-proof')
    _need(type(capacity_inputs) is dict and set(capacity_inputs) == set(_INPUTS)
          and all(type(capacity_inputs[key]) is dict for key in _INPUTS),
          'exact five original capacity proof documents required')
    inputs = deepcopy(capacity_inputs)
    proof = deepcopy(capacity_proof)
    _need((proof.get('schema'), proof.get('schema_version')) == ('scar.session-capacity-proof', 1),
          'versioned source capacity proof required')
    expected_proof = derive_capacity_proof(*(inputs[key] for key in _INPUTS),
        source_refs=tuple(proof.get('source_refs', ())))
    _need(semantic_sha256(proof) == semantic_sha256(expected_proof),
          'capacity proof differs from complete typed source/native initialization replay')
    _need(proof.get('status') in ('PROVEN_CONDITIONAL', 'FRONTIER'),
          'capacity proof must retain its original applicability status')
    verify_session_activation(outer_profile, lifetime_projection, activation_projection,
        outer_profile_ref=encode(outer_ref), lifetime_ref=encode(lease_ref),
        source_refs=activation_projection['source_refs'])
    lifetime = session_lifetime_from_dict(lifetime_projection)
    plan = inputs['session_plan']
    parent = inputs['parent_native_metadata']
    native = inputs['session_native_metadata']
    _need(semantic_sha256(plan) == lifetime.parents['session_plan_sha256']
          and native['session_plan'] == lifetime.parents['session_plan']
          and plan['parent_native_manifest'] == lifetime.parents['parent_native']
          and semantic_sha256(inputs['workspace_plan']) == parent['workspace_plan_sha256']
          and native['execution_plan_sha256'] == native['source_mapping']['execution_plan_sha256']
          and native['source_mapping']['parent_execution_plan_sha256'] == parent['execution_plan_sha256']
          and inputs['step_plan']['resource_binding']['target_sha256'] == lifetime.target.sha256,
          'capacity inputs belong to a different original activation/lease/kernel/target')
    _need(activation_projection['parents']['outer_profile'] == encode(outer_ref)
          and activation_projection['parents']['session_lifetime'] == encode(lease_ref),
          'capacity overlay must use the exact verified source activation and lease')
    slots = lifetime.slot_lifetimes
    supported = proof['status'] == 'PROVEN_CONDITIONAL'
    if supported:
        _need([(row['id'], row['capacity_bytes']) for row in proof['slots']]
              == [(slot.slot, slot.capacity_bytes) for slot in slots],
              'capacity proof private slots differ from original resource lifetimes')
    required = proof['required_capacity_protocol']
    for key in ('invalidate_before_every_step', 'invalidate_after_every_step',
                'full_snapshot_each_step', 'no_cross_step_content_fact',
                'no_fixed_trip_count_authority',
                'forget_all_per_step_array_and_data_owner_references_before_host_return',
                'retain_immutable_provider_capsule_code_dtype_and_base_library_owners',
                'per_step_binding_forget_does_not_unload_native_or_provider_implementation',
                'raw_abi_descriptor_bytes_need_not_be_zeroed',
                'have_bindings_false_before_and_after_step',
                'never_compare_or_read_old_binding_descriptors'):
        _need(required.get(key) is True, 'required capacity-only protocol is missing: '+key)

    action_index = {action.id: index for index, action in enumerate(lifetime.actions)}
    actions = (
        CapacityAction('qualify_leaf', 'ALL_ORIGINAL_INDEPENDENT_LEAF_GUARDS'),
        CapacityAction('first_needed_create', 'ALLOCATE_FIXED_PRIVATE_ARRAY_SLOTS_ONCE_PER_EPOCH',
            _fact(lease_ref, '#/actions/'+str(action_index['create']))),
        CapacityAction('forget_before_step', 'INVALIDATE_AND_FORGET_ACTUAL_ARRAY_DATA_BINDINGS_NEVER_CONSULT_OLD_DESCRIPTORS'),
        CapacityAction('full_snapshot_step', 'ONE_ORIGINAL_GUARDED_KERNEL_WITH_FULL_CURRENT_INPUT_SNAPSHOTS',
            _fact(lease_ref, '#/actions/'+str(action_index['step']))),
        CapacityAction('forget_before_host_return', 'INVALIDATE_AND_FORGET_ACTUAL_ARRAY_AND_DATA_OWNER_BINDINGS'),
        CapacityAction('destroy_epoch', 'OWNER_IDLE_REVERSE_PRIVATE_SLOT_RELEASE',
            _fact(lease_ref, '#/actions/'+str(action_index['destroy']))))
    overlay = dict(schema=SCHEMA, schema_version=1,
        status='PROVEN_CONDITIONAL_CAPACITY_POLICY' if supported else 'UNRESOLVED',
        decision='TRIAL_PRIVATE_CAPACITY_ONLY_DIRTY' if supported else 'KEEP_ISOLATED_LEAF_POLICY',
        representation='DYNAMIC_ACTIVATION_CAPACITY_LIFECYCLE_OVERLAY_NOT_VRS',
        capacity_mode='CAPACITY_ONLY_DIRTY',
        parents=dict(outer_profile=outer_ref, activation=active_ref, session_lifetime=lease_ref,
            capacity_proof=proof_ref, outer_profile_sha256=semantic_sha256(outer_profile),
            activation_sha256=semantic_sha256(activation_projection),
            session_lifetime_sha256=semantic_sha256(lifetime_projection),
            capacity_proof_sha256=semantic_sha256(proof),
            capacity_input_sha256={key:semantic_sha256(inputs[key]) for key in _INPUTS}),
        original_source_overlay=_fact(active_ref, '#/activation'),
        original_calls=_fact(active_ref, '#/source_calls'),
        original_qualification_points=_fact(active_ref, '#/isolated_kernel_points'),
        original_source_control_lookups_and_field_writes_retained=True,
        target=lifetime.target, actions=actions,
        epochs=dict(instance_parameter='(outer_activation_instance, capacity_epoch_ordinal)',
            epoch_count=None, step_count_per_epoch=None, total_kernel_step_count=None,
            first_needed_allocation=True, zero_qualified_steps_allocate_no_capacity=True,
            create_count_per_successful_epoch=1, destroy_count_per_successfully_closed_epoch=1,
            maximum_live_epochs_in_normal_single_owner_policy=1,
            normal_host_interval_keeps_only_private_capacity=True,
            fallback_or_wrapped_barrier_ends_epoch=True,
            next_qualified_step_may_create_a_fresh_epoch=True,
            outer_exit_finally_destroys_remaining_epoch=True,
            original_fixed_parent_trip_count_is_not_any_epoch_or_step_count=True),
        slot_lifetimes=[dict(slot=slot.slot, capacity_bytes=slot.capacity_bytes,
            instance_parameter='(outer_activation_instance, capacity_epoch_ordinal, original_slot)',
            original=_fact(lease_ref, '#/slot_lifetimes/'+str(index)),
            allocation_action='first_needed_create', release_action='destroy_epoch',
            invalidate_releases_capacity=False)
            for index, slot in enumerate(slots)],
        protocol_relations=[
            dict(before='qualify_leaf', after='first_needed_create',
                 when='ALL_LEAF_GUARDS_SUCCEED_AND_NO_LIVE_CAPACITY_EPOCH'),
            dict(before='first_needed_create', after='forget_before_step', when='CREATE_SUCCEEDS'),
            dict(before='forget_before_step', after='full_snapshot_step', when='SAME_DYNAMIC_STEP'),
            dict(before='full_snapshot_step', after='forget_before_host_return',
                 when='SUCCESS_OR_ADMITTED_FAILURE_BEFORE_RETURN_TO_ANY_HOST_CODE'),
            dict(before='forget_before_host_return', after='forget_before_step',
                 when='NEXT_QUALIFIED_STEP_IN_SAME_EPOCH_NOT_A_STATIC_DAG_EDGE'),
            dict(before='forget_before_host_return', after='destroy_epoch',
                 when='OUTER_EXIT_OR_FALLBACK_OR_WRAPPED_BARRIER_OR_ERROR')],
        content_policy=dict(private_bytes_may_contain_previous_values=True,
            old_bytes_are_not_available_as_next_step_input_facts=True,
            mode_between_steps='CAPACITY_ONLY_DIRTY', full_snapshot_before_every_step=True,
            forget_scope='PER_STEP_DYNAMIC_BINDINGS_NOT_BASE_ARTIFACT_STATIC_CODE_OR_LIBRARY_IDENTITY',
            per_step_array_and_data_owner_references_across_host_intervals=False,
            immutable_provider_implementation_owners_retained=True,
            old_raw_descriptors_can_remain_but_are_not_read=True,
            old_binding_descriptors_compared_or_read=False, retained_roi_correspondence=False,
            content_reuse_or_finite_scan_elimination_authorized=False,
            initializations=_fact(proof_ref, '#/current_step_initialization') if supported else None,
            existing_invalidate_alone_is_not_python_reference_forgetting=True),
        failures=dict(guard_failure='KEEP_ORIGINAL_CURRENT_CALL_ONCE_NO_NATIVE_REUSE',
            fallback='DESTROY_CURRENT_EPOCH_THEN_ORIGINAL_CURRENT_KERNEL_ONCE',
            wrapped_barrier='DESTROY_BEFORE_ORIGINAL_BARRIER_NEXT_EPOCH_IS_FRESH',
            ordinary_unwrapped_host='NO_NEW_WRAPPER_PRIVATE_CAPACITY_ONLY_WITH_NO_OWNER_OR_CONTENT_FACT',
            cleanup_failure='QUARANTINE_POSSIBLE_CAPACITY_AND_DISABLE_FURTHER_NATIVE_CREATION',
            exception='FORGET_REFERENCES_AND_OWNER_IDLE_DESTROY_DURING_UNWIND'),
        resource_scope=dict(target=lifetime.target, budget=lifetime.budget,
            known_private_array_payload_bytes=sum(slot.capacity_bytes for slot in slots),
            original_external_payload_accounting=_fact(lease_ref, '#/memory'),
            cumulative_allocation_bytes=None, epoch_count=None,
            host_internal_memory_bytes=None, session_metadata_bytes=None,
            whole_activation_rss_bound=None, outer_cpu_domain=None, outer_execution_units=None,
            allocation_release_and_forget_durations=None, total_activation_duration=None,
            kernel_resource_evidence_only=_fact(lease_ref,
                '#/shared_step_template/original_execution_bindings')),
        obligations=dict(required_new_runtime_protocol=_fact(proof_ref, '#/required_capacity_protocol'),
            initialization=_fact(proof_ref, '#/current_step_initialization') if supported else None,
            initialization_frontiers=None if supported else _fact(proof_ref, '#/frontiers'),
            provider_external_assumptions=_fact(proof_ref, '#/provider_boundary_Q') if supported
                else _fact(lease_ref, '#/shared_step_template/provider_boundary_Q'),
            original_invalidate_not_sufficient_for_owner_forgetting=True,
            runtime_implementation_and_source_admission_still_required=True),
        frontiers=deepcopy(proof.get('frontiers', ())), source_refs=tuple(deepcopy(source_refs)),
        runtime_protocol_checked=False, source_effect_coverage_complete=False,
        unknown_host_quiescence_added=False, content_retention_authorized=False,
        physical_resource_feasibility_proven=False, physical_timeline_proven=False,
        solver_invoked=False, native_compiled=False, target_program_executed=False,
        execution_authorized=False, default_backend='NO_OP', deployment_authorized=False)
    result = encode(overlay)
    result['id'] = 'session-capacity-plan:'+semantic_sha256(result)
    return result


def verify_session_capacity(outer_profile, activation_projection, lifetime_projection,
        capacity_proof, plan, **kwargs):
    """Recompute initialization and all original source/lifetime/overlay links."""
    expected = derive_session_capacity(outer_profile, activation_projection,
        lifetime_projection, capacity_proof, **kwargs)
    _need(semantic_sha256(plan) == semantic_sha256(expected),
          'capacity policy differs from typed initialization/source/lifetime replay')
    return plan


__all__ = ['SCHEMA', 'CapacityAction', 'derive_session_capacity', 'verify_session_capacity']
