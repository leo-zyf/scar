"""Compose a shared source activation with isolated dynamic workspace leases.

This source overlay preserves original control/writes and references the typed
lease template.  It is neither an observed activation history nor a VRS, and
does not infer that unknown host effects become quiescent after returning.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.record_codec import decode, encode
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.session_lifetime import session_lifetime_from_dict

SCHEMA = 'scar.workspace-session-activation'
OUTER_SCHEMA = 'scar.guarded-session-outer'
_GUARDS = 'INDEPENDENT_LEAF_SOURCE_ABI_ROOT_PROVIDER_METADATA_GUARDS'


@dataclass(frozen=True, slots=True)
class ActivationLeaseAction:
    id: str
    role: str
    original_action: ProviderFactReference | None
    duration_ticks: int | None
    execution_units: int | None
    exact_cpu: None


@dataclass(frozen=True, slots=True)
class ActivationLeaseRelation:
    before: str
    after: str
    instance: str
    condition: str
    authority: str


@dataclass(frozen=True, slots=True)
class IsolatedKernelPoint:
    operation: int
    qualification: ProviderFactReference
    original_call: ProviderFactReference
    nested_caller_id: str
    required_guards: str
    retention_allowed: bool
    lease_parameter: str
    maximum_native_step_attempts_per_lease: int
    committed_steps_per_attempt: tuple[int, int]


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _reference(value, schema=None):
    value = value if type(value) is ProviderArtifact else decode(ProviderArtifact, value)
    _need(Path(value.uri).is_absolute() and (schema is None or value.schema == schema),
          'original absolute artifact reference with the expected schema required')
    return value


def _fact(reference, pointer):
    return ProviderFactReference(reference, pointer)


def derive_session_activation(outer_profile, lifetime_projection, *,
        outer_profile_ref, lifetime_ref, source_refs=()):
    """Project qualified parent records without disk I/O or runtime execution.

    The publisher must independently replay the source outer profile and lease
    parents.  A scalar source model is not a complete effect proof: this first
    adapter deliberately supports only isolated leaf delivery without retention.
    """
    profile = deepcopy(outer_profile)
    outer = _reference(outer_profile_ref, OUTER_SCHEMA)
    lifetime_reference = _reference(lifetime_ref, 'scar.workspace-session-lifetime')
    lifetime = session_lifetime_from_dict(lifetime_projection)
    _need((profile.get('schema'), profile.get('schema_version'), profile.get('status'))
          == (OUTER_SCHEMA, 1, 'CONDITIONAL_SCALAR_MODEL_WITH_ISOLATED_KERNEL_DELIVERY')
          and profile.get('effect_profile_closed') is False
          and profile.get('cross_host_residency_authorized') is False
          and profile.get('opaque_call_quiescence_assumed') is False
          and profile.get('host_calls_dynamically_intercepted') is False
          and profile.get('unqualified_kernel_policy') == 'FRESH_ISOLATED_NATIVE',
          'conditional outer model must explicitly select isolated independent kernel delivery')
    _need(type(profile.get('id')) is str and profile['id']
          and type(profile.get('unit')) is str and profile['unit']
          and type(profile.get('function_ordinal')) is int and profile['function_ordinal'] >= 0
          and type(profile.get('source')) is dict and profile['source']
          and type(profile.get('body')) is dict and profile['body']
          and (type(profile.get('ledger_id')) is str and profile['ledger_id']
               or type(profile.get('ledger_reference')) is dict),
          'original source function/body and effect ledger identity required')
    for key in ('calls', 'guards', 'nested_caller_ids', 'shared_loops',
                'field_writes', 'qualification_points', 'original_control_refs'):
        _need(type(profile.get(key)) is list, 'original outer model field required: ' + key)
    _need(bool(profile['original_control_refs']), 'original control edges must remain referenced')
    control = []
    for row in profile['original_control_refs']:
        _need(type(row) is dict and set(row) in (
            {'artifact', 'locator'}, {'artifact', 'locator', 'original_edge_ordinals'}),
            'original control fact and optional exact edge ordinals required')
        if 'original_edge_ordinals' in row:
            ordinals = row['original_edge_ordinals']
            _need(type(ordinals) is list and all(type(i) is int and i >= 0 for i in ordinals)
                  and len(set(ordinals)) == len(ordinals), 'original control edge ordinals differ')
        control.append(decode(ProviderFactReference, {key: row[key] for key in ('artifact', 'locator')}))
    for row in control:
        _reference(row.artifact)
        _need(row.locator.startswith('#'), 'original control locator required')
    nested = profile['nested_caller_ids']
    _need(all(type(value) is str and value for value in nested) and len(set(nested)) == len(nested),
          'unique original nested caller identities required')
    for key in ('guards', 'shared_loops', 'field_writes'):
        _need(all(type(row) is dict and row for row in profile[key]),
              'nonempty original source records required: ' + key)

    calls = {}
    source_calls = []
    host_exclusions = []
    for index, call in enumerate(profile['calls']):
        _need(type(call) is dict and type(call.get('operation')) is int and call['operation'] >= 0
              and type(call.get('query_ordinal')) is int and call['query_ordinal'] >= 0
              and call['operation'] not in calls and 'lookup_route' in call
              and call.get('role') in ('NESTED_CALLER', 'ORIGINAL_HOST_CALL'),
              'unique original nested/host CALL records and lookups required')
        if call['role'] == 'NESTED_CALLER':
            boundary = call.get('boundary')
            _need(type(boundary) is dict and boundary.get('id') in nested,
                  'nested call must retain its original qualified caller identity')
        else:
            # No wrapper or observed barrier is fabricated for a print/dump,
            # source field operation, or another unclosed host effect.
            host_exclusions.append(dict(operation=call['operation'],
                original=_fact(outer, '#/calls/'+str(index)),
                policy='UNWRAPPED_ORIGINAL_WITH_NO_LEASE_ACROSS_HOST',
                lookup_and_effects_remain_original=True, observed_barrier_event=False))
        calls[call['operation']] = (index, call)
        source_calls.append(dict(operation=call['operation'], query_ordinal=call['query_ordinal'],
            role=call['role'], original=_fact(outer, '#/calls/'+str(index))))

    points, point_operations = [], set()
    for index, point in enumerate(profile['qualification_points']):
        _need(type(point) is dict and type(point.get('operation')) is int
              and point['operation'] in calls and point['operation'] not in point_operations
              and point.get('retention_allowed') is False
              and point.get('kind') == 'ORIGINAL_KERNEL_SOURCE_AND_INPUT_GUARDS'
              and point.get('isolated_lease_required') is True
              and point.get('required_guards') == _GUARDS,
              'qualification must require independent guards and prohibit retained content')
        call_index, call = calls[point['operation']]
        _need(call['role'] == 'NESTED_CALLER' and point.get('nested_caller_id') == call['boundary']['id'],
              'qualification must refer to its exact original nested caller')
        points.append(IsolatedKernelPoint(point['operation'],
            _fact(outer, '#/qualification_points/'+str(index)),
            _fact(outer, '#/calls/'+str(call_index)), call['boundary']['id'], _GUARDS,
            False, '(outer_activation_instance, dynamic_kernel_call_ordinal)', 1, (0, 1)))
        point_operations.add(point['operation'])
    _need(point_operations == {operation for operation, (_, call) in calls.items()
                              if call['role'] == 'NESTED_CALLER'},
          'each nested caller needs its own original-site isolated-delivery qualification')

    actions = [ActivationLeaseAction('qualify_leaf', 'INDEPENDENT_LEAF_GUARDS', None, None, None, None)]
    for index, action in enumerate(lifetime.actions):
        if action.id in ('create', 'step', 'destroy'):
            _need(action.duration_ticks is None and action.execution_units is None and not action.eligible_threads,
                  'outer actions cannot inherit a kernel CPU or invented service time')
            actions.append(ActivationLeaseAction(action.id, action.kind.value,
                _fact(lifetime_reference, '#/actions/'+str(index)), None, None, None))
    _need({a.id for a in actions} == {'qualify_leaf', 'create', 'step', 'destroy'},
          'original create/step/destroy lease action templates required')
    relations = tuple(ActivationLeaseRelation(before, after, 'SAME_ISOLATED_LEASE_INSTANCE',
        condition, 'CONDITIONAL_DELIVERY_REFINEMENT_NOT_OBSERVED_SOURCE_ORDER')
        for before, after, condition in (
            ('qualify_leaf', 'create', 'ALL_INDEPENDENT_LEAF_GUARDS_SUCCEED'),
            ('create', 'step', 'CREATE_SUCCEEDS_AND_LEAF_CALL_REMAINS_QUALIFIED'),
            ('step', 'destroy', 'SUCCESS_OR_ADMITTED_FAILURE_BEFORE_RETURN_TO_ORIGINAL_HOST')))
    slots = [dict(slot=row.slot, capacity_bytes=row.capacity_bytes,
        original=_fact(lifetime_reference, '#/slot_lifetimes/'+str(index)),
        instance_parameter='(outer_activation_instance, dynamic_kernel_call_ordinal, original_slot)',
        allocation_action='create', release_action='destroy')
        for index, row in enumerate(lifetime.slot_lifetimes)]
    result = dict(schema=SCHEMA, schema_version=1,
        representation='SOURCE_ACTIVATION_WITH_ISOLATED_DYNAMIC_LEASES_NOT_VRS',
        parents=dict(outer_profile=outer, outer_profile_sha256=semantic_sha256(profile),
            session_lifetime=lifetime_reference, session_lifetime_sha256=semantic_sha256(lifetime_projection),
            target=lifetime.target),
        activation=dict(instance_parameter='outer_activation_instance', original_id=profile['id'],
            original_source=_fact(outer, '#/source'), shared_body=_fact(outer, '#/body'),
            original_control_refs=deepcopy(profile['original_control_refs']),
            original_control_facts=control,
            shared_loop_regions=[_fact(outer, '#/shared_loops/'+str(i)) for i in range(len(profile['shared_loops']))],
            original_field_writes=[_fact(outer, '#/field_writes/'+str(i)) for i in range(len(profile['field_writes']))],
            original_guards=_fact(outer, '#/guards'), original_ledger=_fact(outer, '#/ledger_id'
                if 'ledger_id' in profile else '#/ledger_reference'),
            activation_count=1, source_loop_iteration_counts=None,
            nested_callers_share_outer_instance=True, source_regions_unrolled=False,
            original_control_field_writes_and_lookups_retained=True,
            source_world_is_not_runtime_content_version=True),
        source_calls=source_calls, isolated_kernel_points=points, host_exclusions=host_exclusions,
        actions=actions, relations=relations,
        leases=dict(instance_parameter='(outer_activation_instance, dynamic_kernel_call_ordinal)',
            lease_count=None, dynamic_kernel_call_count=None,
            nested_source_caller_not_collapsed_to_one_kernel_step=True,
            each_independently_guarded_leaf_kernel_owns_its_own_lease=True,
            maximum_native_step_attempts_per_lease=1,
            maximum_live_leases_in_normal_owner_policy=1,
            shared_step_template=_fact(lifetime_reference, '#/shared_step_template'),
            owned_slots=slots, kernel_cpu_evidence=_fact(lifetime_reference,
                '#/shared_step_template/original_execution_bindings'),
            slots_are_parameters_not_observed_physical_allocations=True,
            no_lease_across_original_host_operations=True, cross_activation_retention=False,
            first_snapshot='FULL_FOR_EVERY_ISOLATED_LEASE_NO_ROI_RESIDENCY_REUSE',
            absence_at_outer_entry_and_exit='REQUIRED_NORMAL_PROTOCOL_NOT_OBSERVED_HISTORY'),
        content_and_failure=dict(outer_effect_profile_closed=False,
            content_retention_authorized=False, taint_monotone_until_outer_exit=True,
            qualification_does_not_clear_taint=True, revalidation_restores_prior_replica=False,
            guard_failure='ORIGINAL_CURRENT_KERNEL_PATH_ONCE_WITHOUT_NATIVE_REUSE',
            failed_step='DISCARD_CORRESPONDENCE_DESTROY_BEFORE_CURRENT_CALL_FALLBACK',
            destroy_failure='QUARANTINE_POSSIBLY_OWNED_CAPACITY_DISABLE_FURTHER_NATIVE_LEASE_CREATION',
            unknown_io_does_not_acquire_a_quiescence_assumption=True,
            external_input_changes_require_all_original_leaf_guards=True),
        resource_scope=dict(target=lifetime.target, budget=lifetime.budget,
            fixed_payload_bytes_per_lease=sum(row.capacity_bytes for row in lifetime.slot_lifetimes),
            cumulative_allocation_bytes=None, activation_total_service_ticks=None,
            outer_cpu_domain=None, outer_execution_units=None, session_metadata_bytes=None,
            host_internal_resource_demands=None, whole_activation_memory_bound_proven=False,
            existing_leaf_external_conditions=_fact(lifetime_reference,
                '#/shared_step_template/provider_boundary_Q')),
        source_refs=tuple(deepcopy(source_refs)),
        source_effect_coverage_complete=False, runtime_guards_checked=False,
        dynamic_activation_observed=False, solver_invoked=False,
        physical_resource_feasibility_proven=False, physical_timeline_proven=False,
        execution_authorized=False, original_caller_replacement_authorized=False,
        default_backend='NO_OP', deployment_authorized=False)
    return encode(result)


def verify_session_activation(outer_profile, lifetime_projection, projection, **kwargs):
    """Fully rederive the conditional source/lease relation; no disk or execution."""
    expected = derive_session_activation(outer_profile, lifetime_projection, **kwargs)
    _need(semantic_sha256(projection) == semantic_sha256(expected),
          'activation differs from original source/lease model replay')
    return projection


__all__ = ['SCHEMA', 'ActivationLeaseAction', 'ActivationLeaseRelation', 'IsolatedKernelPoint',
           'derive_session_activation', 'verify_session_activation']
