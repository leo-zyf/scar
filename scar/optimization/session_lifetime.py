"""Dynamic workspace leases, referencing the original kernel/resource graph.

This is not a VRS: a lease admits any finite sequence of step/invalidate calls,
whereas the retained v4 source loop has a fixed trip count.  Source/native and
hardware publication readers admit the parents; this pure projection checks
their links and preserves their evidence without loading a library.
"""
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.program.target_hardware import TargetResourceRef
from scar.ir.record_codec import decode, encode
from scar.ir.temporal_resources import OriginalBudget, projection_document
from scar.ir.versioned_schedule import (TemporalOccupancyKind, TemporalScopeKind,
    schedule_document, validate_versioned_schedule)
from scar.ir.workspace_session import SESSION_PROTOCOL, SESSION_SYMBOLS
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.workspace-session-lifetime'


class SessionActionKind(str, Enum):
    CREATE = 'CREATE'
    STEP = 'STEP'
    INVALIDATE = 'INVALIDATE'
    DESTROY = 'DESTROY'


@dataclass(frozen=True, slots=True)
class SessionAction:
    id: str
    kind: SessionActionKind
    native_symbol: str
    minimum_occurrences: int
    maximum_occurrences: int | None
    duration_ticks: int | None
    execution_units: int | None
    eligible_threads: tuple[TargetResourceRef, ...]
    allocation_slots: tuple[str, ...]
    release_slots: tuple[str, ...]
    content_transition: str
    source_refs: tuple[ProviderFactReference, ...]


@dataclass(frozen=True, slots=True)
class SessionSlotLifetime:
    slot: str
    capacity_bytes: int
    slot_instance_parameter: str
    physical_count_per_successful_lease: int
    allocation_action: str
    release_action: str
    allocation_boundary: str
    release_boundary: str
    candidate_memories: tuple[TargetResourceRef, ...]
    exact_memory: TargetResourceRef | None
    source_slot: ProviderFactReference
    source_resource_binding: ProviderFactReference


@dataclass(frozen=True, slots=True)
class SessionLifetime:
    instance_parameter: str
    target: ProviderArtifact
    parents: dict
    actions: tuple[SessionAction, ...]
    slot_lifetimes: tuple[SessionSlotLifetime, ...]
    auxiliary_storage: dict
    shared_step_template: dict
    protocol: dict
    memory: dict
    budget: OriginalBudget
    unresolved: tuple[dict, ...]
    source_refs: tuple[dict, ...]


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _ref(value, schema=None):
    ref = value if type(value) is ProviderArtifact else decode(ProviderArtifact, value)
    _need(Path(ref.uri).is_absolute() and (schema is None or ref.schema == schema),
          'original absolute artifact reference with the expected schema required')
    return ref


def _fact(reference, pointer):
    return ProviderFactReference(reference, pointer)


def session_lifetime_from_dict(document):
    """Strict typed decoding; parent qualification requires the replay API."""
    payload = dict(document)
    _need(payload.pop('schema', None) == SCHEMA and type(payload.get('schema_version')) is int
          and payload.pop('schema_version') == 1, 'unsupported session lifetime schema')
    for key, expected in _CLAIMS.items():
        actual = payload.pop(key, None)
        _need(type(actual) is type(expected) and actual == expected,
              'session lifetime claim differs: ' + key)
    return decode(SessionLifetime, payload)


_CLAIMS = dict(representation='TYPED_DYNAMIC_SESSION_LIFECYCLE_NOT_VRS',
    fixed_trip_count_authorizes_session_count=False, source_caller_effect_coverage_proven=False,
    allocation_ids_are_runtime_observations=False, physical_resource_feasibility_proven=False,
    duration_predicted=False, solver_invoked=False, execution_authorized=False,
    caller_replacement_authorized=False, deployment_authorized=False)


def derive_session_lifetime(session_plan, kernel_schedule, kernel_resources, *,
        session_plan_ref, kernel_schedule_ref, kernel_resources_ref,
        native_reference, native_metadata, parent_native_metadata, source_refs=()):
    """Project one successful dynamic lease; no disk reads, clocks or execution.

    Callers must cold-admit the session preparation/native and original resource
    preparation before calling.  Real ArtifactRefs, exact source identities and
    conditional body evidence remain visible, rather than an authority boolean.
    """
    plan = deepcopy(session_plan)
    _need((plan.get('schema'), plan.get('schema_version'), plan.get('status'))
          == ('scar.workspace-session-plan', 1, 'PROVEN_CONDITIONAL'),
          'source-qualified conditional workspace session plan required')
    _need(plan['protocol'] == dict(SESSION_PROTOCOL) and plan['symbols'] == dict(SESSION_SYMBOLS)
          and plan['original_fixed_trip_count_is_not_session_call_count'] is True
          and plan['original_caller_replacement_authorized'] is False
          and plan['source_caller_effect_coverage_proven'] is False
          and plan['deployment_authorized'] is False, 'dynamic session protocol/authority differs')
    p = _ref(session_plan_ref, 'scar.workspace-session-plan')
    s = _ref(kernel_schedule_ref)
    r = _ref(kernel_resources_ref, 'scar.temporal-resource-projection')
    n = _ref(native_reference, 'scar.native-workspace-session')
    parent = _ref(plan['parent_native_manifest'], 'scar.native-stateful-workspace-repeat')
    original_cpp = _ref(plan['native_blueprint']['cpp_source'], 'scar.cpp-source')
    native = deepcopy(native_metadata)
    old = deepcopy(parent_native_metadata)
    _need((native.get('schema'), native.get('schema_version')) == (n.schema, 1)
          and native['session_plan'] == encode(p), 'native manifest binds a different session plan')
    mapping = native['source_mapping']
    session_cpp = _ref(native['source'], 'scar.cpp-source')
    session_library = _ref(native['library'], 'scar.native-library')
    expected_execution = dict(schema='scar.workspace-session-execution-plan', schema_version=1,
        parent_execution_plan_sha256=old['execution_plan_sha256'],
        residency_plan_sha256=plan['residency_plan_sha256'], session_plan_sha256=semantic_sha256(plan))
    _need((old.get('schema'), old.get('schema_version')) == (parent.schema, 1)
          and mapping['execution_plan'] == expected_execution
          and mapping['execution_plan_sha256'] == native['execution_plan_sha256']
          == semantic_sha256(expected_execution)
          and plan['native_blueprint']['manifest'] == encode(parent)
          and plan['native_blueprint']['execution_plan_sha256'] == old['execution_plan_sha256']
          and plan['native_blueprint']['workspace_plan_sha256'] == old['workspace_plan_sha256']
          and plan['native_blueprint']['base_ids'] == old['base_ids'],
          'session/source parent native identities differ')
    _need(mapping['protocol'] == dict(SESSION_PROTOCOL)
          and mapping['dynamic_call_count_fixed_N_authority'] is False
          and mapping['helper_called_once_per_step'] is True
          and mapping['helper_math_checks_and_commits_unchanged'] is True
          and mapping['create_original_preflight_before_payload_allocation'] is True
          and mapping['create_has_no_math_or_payload_reads'] is True
          and mapping['source_inverse_to_residency_byte_equal'] is True
          and mapping['original_source_sha256'] == original_cpp.sha256
          and mapping['source_sha256'] == session_cpp.sha256
          and mapping['local_to_session_transfer'] == 'UNIQUE_PTR_METADATA_MOVE_NO_PAYLOAD_COPY'
          and mapping['destroy_slot_order'] == 'CPP_STD_ARRAY_REVERSE_ELEMENT_DESTRUCTION'
          and mapping['session_metadata_allocation'] == 'ONE_ADDITIONAL_NOTHROW_NEW_NOT_ARRAY_PAYLOAD',
          'source-mapped session lifetime/step/allocation contract differs')

    validate_versioned_schedule(kernel_schedule)
    doc = schedule_document(kernel_schedule)
    _need(doc['schema'] == s.schema and kernel_schedule.temporal is not None,
          'original typed temporal source schedule required')
    resource_doc = projection_document(kernel_resources)
    _need(kernel_resources.source_schedule_sha256 == semantic_sha256(doc),
          'resource projection refers to another kernel graph')
    certificate = kernel_schedule.proof['workspace_time']
    _need(certificate['native_manifest'] == encode(parent)
          and certificate['source_workspace_plan_sha256'] == old['workspace_plan_sha256'],
          'original kernel template belongs to another native/workspace parent')
    temporal = kernel_schedule.temporal
    loops = [scope for scope in temporal.scopes if scope.kind is TemporalScopeKind.LOOP]
    _need(len(loops) == 1 and loops[0].parent == kernel_schedule.call_scope
          and loops[0].trip_count == old['trip_count'], 'one original fixed source loop template required')
    loop = loops[0]
    activities = {row.id: row for row in temporal.activities}
    occupancy = {row.id: row for row in temporal.occupancies}
    expected_execution = {key for key, row in occupancy.items()
                          if row.kind is TemporalOccupancyKind.EXECUTION_DEMAND}
    _need(len(kernel_resources.executions) == len(expected_execution)
          and {row.occupancy for row in kernel_resources.executions} == expected_execution,
          'original execution occupancy bindings must be complete and unique')
    for binding in kernel_resources.executions:
        o = occupancy[binding.occupancy]
        _need(binding.resource == o.resource and binding.units == o.units
              and (binding.units is None or binding.units > 0),
              'original execution demand cannot be replaced by zero/free work')
        for ref in (*binding.core_refs, *binding.candidate_threads,
                    *((binding.exact_thread,) if binding.exact_thread else ())):
            _need(ref.snapshot == kernel_resources.target, 'body binding belongs to another target snapshot')
    _need(kernel_resources.budget == OriginalBudget(kernel_schedule.budget_bytes,
          kernel_schedule.budget_scope, kernel_schedule.budget_refs), 'original payload budget differs')

    slots = [slot for slot in kernel_schedule.storage_slots if slot.ownership == 'SCHEDULE']
    geometry = plan['geometry']
    _need([slot.capacity_bytes for slot in slots] == geometry['slot_capacities_bytes']
          and len(kernel_resources.storages) == len(slots)
          and {binding.slot for binding in kernel_resources.storages} == {slot.id for slot in slots},
          'fixed session array slots differ from original capacity bindings')
    task_scope = {row.task: row.scope for row in activities.values()}
    slot_rows = []
    for slot in slots:
        matches = [(index, binding) for index, binding in enumerate(kernel_resources.storages)
                   if binding.slot == slot.id]
        _need(len(matches) == 1, 'one original resource binding per owned slot required')
        index, binding = matches[0]
        o = occupancy[binding.occupancy]
        _need(o.kind is TemporalOccupancyKind.PHYSICAL_SLOT_BYTES and o.storage_slot == slot.id
              and o.units == slot.capacity_bytes == binding.payload_bytes
              and task_scope[slot.allocated_by] == task_scope[slot.released_by] == kernel_schedule.call_scope
              and binding.exact_memory is None
              and all(ref.snapshot == kernel_resources.target for ref in binding.candidate_memories),
              'slot ownership/capacity/outer lifetime/possible location differs')
        slot_rows.append(SessionSlotLifetime(slot.id, slot.capacity_bytes, 'session_instance', 1,
            'create', 'destroy', 'ORIGINAL_ARRAY_ALLOCATION_WITHIN_CREATE',
            'REVERSE_UNIQUE_PTR_DESTRUCTION_WITHIN_OWNER_IDLE_DESTROY',
            binding.candidate_memories, binding.exact_memory,
            _fact(s, '#/storage_slots/'+str(kernel_schedule.storage_slots.index(slot))),
            _fact(r, '#/storages/'+str(index))))
    external = sum(slot.capacity_bytes for slot in kernel_schedule.storage_slots if slot.ownership == 'EXTERNAL')
    owned = sum(slot.capacity_bytes for slot in slots)
    memory = geometry['memory']
    _need(memory == old['memory'] and memory['workspace_payload_bytes'] == owned
          and memory['external_array_bytes'] == external
          and memory['total_region_array_payload_bytes'] == owned + external
          and memory['memory_budget_bytes'] == kernel_resources.budget.bytes
          and memory['budget_scope'] == kernel_resources.budget.scope,
          'original declared array-payload memory accounting differs')
    for snapshot in geometry['snapshots']:
        _need(snapshot['slot_id'] in {slot.id for slot in slots}
              and snapshot['mode'] in ('KEEP_FULL_SNAPSHOT', 'REFRESH_COMPLEMENT', 'SKIP_AFTER_FIRST'),
              'snapshot policy references an unknown session slot')

    common = (_fact(p, '#/protocol'), _fact(n, '#/source_mapping'))
    actions = tuple(SessionAction(name, kind, SESSION_SYMBOLS[name], minimum, maximum,
        None, None, (), tuple(slot.id for slot in slots) if name == 'create' else (),
        tuple(slot.id for slot in reversed(slots)) if name == 'destroy' else (), transition,
        (*common, _fact(p, '#/transitions/'+key)))
        for name, kind, minimum, maximum, key, transition in (
            ('create', SessionActionKind.CREATE, 1, 1, 'create', 'UNBOUND_DIRTY'),
            ('step', SessionActionKind.STEP, 0, None, 'successful_step',
             'SUCCESS_ESTABLISHES_ROI_CORRESPONDENCE_ADMITTED_FAILURE_MAKES_DIRTY'),
            ('invalidate', SessionActionKind.INVALIDATE, 0, None, 'unknown_effect',
             'DISCARD_CORRESPONDENCE_ONLY_KEEP_ALLOCATED_CAPACITY_AND_BYTES'),
            ('destroy', SessionActionKind.DESTROY, 1, 1, 'destroy', 'RELEASE_CAPACITY_AND_ALL_CORRESPONDENCE')))
    body_activities = [row for row in temporal.activities if row.scope == loop.id]
    body = dict(original_loop_scope=_fact(s, '#/temporal/scopes/'+str(temporal.scopes.index(loop))),
        original_fixed_trip_count=loop.trip_count, step_occurrence_count=None,
        fixed_count_is_ancestry_only=True, template_is_not_a_new_executable_schedule=True,
        original_activities=[_fact(s, '#/temporal/activities/'+str(temporal.activities.index(a))) for a in body_activities],
        original_execution_bindings=[_fact(r, '#/executions/'+str(index))
            for index, binding in enumerate(kernel_resources.executions)
            if activities[occupancy[binding.occupancy].activity].scope == loop.id],
        snapshot_policies=deepcopy(geometry['snapshots']), scalar_bindings='RELOAD_EVERY_STEP',
        exact_cpu_evidence_scope='ORIGINAL_CONDITIONAL_BODY_FRAGMENT_NOT_OUTER_ACTION_OR_FULL_STEP',
        correspondence_is_not_logical_value_or_content_immutability=True,
        changed_services=['CONDITIONAL_FULL_OR_COMPLEMENT_SNAPSHOT', 'OWNER_BUSY_LEASE_GUARD',
            'ARRAY_PROVIDER_BINDING_COMPARISON', 'DIRTY_AND_COMMITTED_ONE_PUBLICATION'],
        original_repeat_progress_phase_is_not_session_progress=True,
        reused_scope_data_is_reference_evidence_not_unconditional_session_carry=True,
        boundary_and_changed_fragment_durations='UNKNOWN_NOT_COPIED_OR_MULTIPLIED',
        provider_boundary_Q=deepcopy(plan['provider_boundary_Q']))
    unresolved = (
        dict(kind='DYNAMIC_ACTION_COUNTS', step_count=None, invalidate_count=None, finite_per_lease=True),
        dict(kind='ACTION_SERVICE_COST_AND_OUTER_CPU_DOMAIN', duration_ticks=None, execution_units=None,
             eligible_threads=[], selected_kernel_cpu_domain_is_not_outer_domain=True),
        dict(kind='SESSION_METADATA_STORAGE_SPAN', bytes=None, outside_original_array_payload_budget=True),
        dict(kind='ORIGINAL_BODY_RESOURCE_FRONTIERS', parent=r, locator='#/unresolved'),
        dict(kind='SOURCE_CALLER_EFFECT_COVERAGE', proven=False,
             obligation='INVALIDATE_OR_DESTROY_BEFORE_ANY_UNQUALIFIED_EFFECT_CALL_EVENTS_DO_NOT_COVER_ALL_WRITES'),
        dict(kind='BORROWED_PROVIDER_MEMORY', conditions=deepcopy(plan['provider_boundary_Q']),
             source_purity_or_noescape_inferred=False))
    projection = SessionLifetime('session_instance', kernel_resources.target,
        dict(session_plan=p, kernel_schedule=s, kernel_schedule_sha256=semantic_sha256(doc),
             kernel_resources=r, kernel_resources_sha256=semantic_sha256(resource_doc),
             session_native=n, session_cpp=session_cpp, session_library=session_library,
             parent_native=parent, original_cpp=original_cpp,
             session_plan_sha256=semantic_sha256(plan)), actions, tuple(slot_rows),
        dict(kind='SESSION_OBJECT_AND_DESCRIPTOR_PROVIDER_METADATA', count_per_lease=1,
             bytes=None, allocation_action='create', release_action='destroy',
             original_array_payload_budget_covers_this=False,
             source=_fact(n, '#/source_mapping/session_metadata_allocation')), body,
        dict(language='CREATE_SUCCESS (STEP_ATTEMPT | INVALIDATE_SUCCESS)* DESTROY_SUCCESS',
             instance_parameter='session_instance', step_count=None, invalidate_count=None,
             all_dynamic_counts_finite=True, accepted_live_actions_single_owner_nonoverlapping=True,
             successful_lease_normal_protocol_only=True, failed_create_cleanup_timing_modeled=False,
             create_precedes_all_live_actions=True, destroy_follows_all_live_actions=True,
             invalidate_releases_capacity=False, zero_steps_still_allocates_on_successful_create=True,
             source_write_authority='CALLER_MUST_INVALIDATE_BEFORE_UNQUALIFIED_EFFECT',
             invalid_owner_or_busy='REJECT_WITHOUT_PAYLOAD_OR_NONATOMIC_STATE_ACCESS'),
        dict(fixed_owned_array_payload_bytes=owned,
             declared_external_payload_bytes_per_admitted_binding=external,
             declared_external_payload_is_not_owned_or_permanently_retained=True,
             original_array_payload_budget_satisfied=kernel_resources.budget.bytes is not None
                 and owned + external <= kernel_resources.budget.bytes,
             complete_session_storage_bytes=None, physical_rss_bound_proven=False,
             candidate_memory_is_not_numa_residency_or_available_capacity=True),
        kernel_resources.budget, unresolved, tuple(deepcopy(source_refs)))
    result = dict(schema=SCHEMA, schema_version=1, **_CLAIMS, **encode(projection))
    session_lifetime_from_dict(result)
    return result


def verify_session_lifetime(session_plan, kernel_schedule, kernel_resources, projection, **kwargs):
    """Pure full rederivation; returns the original persistent projection."""
    session_lifetime_from_dict(projection)
    expected = derive_session_lifetime(session_plan, kernel_schedule, kernel_resources, **kwargs)
    _need(semantic_sha256(expected) == semantic_sha256(projection),
          'session lifetime differs from original parents/protocol/resource replay')
    return projection


__all__ = ['SCHEMA', 'SessionActionKind', 'SessionAction', 'SessionSlotLifetime', 'SessionLifetime',
           'session_lifetime_from_dict', 'derive_session_lifetime', 'verify_session_lifetime']
