"""Derive a scoped lazy lane-arena replacement from proved task storage."""

from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.cpu-task-arena-plan'
ELIGIBLE = 'ELIGIBLE_FOR_EXPLICIT_VALIDATION'
TRANSFORMATION = 'LAZY_PER_LANE_PRIVATE_ARENA_REUSE'
_RUNTIME = 'scar.native-seeded-array-task-entry'
_DIAGNOSTICS = ('SCAR_ARRAY_ENTRY_STATES', 'SCAR_ARRAY_ENTRY_LANE_REPORT')
_SUCCESS_PROOFS = ('every_arena_read_current_task_initialized', 'arena_pointer_does_not_escape_task',
                   'lane_serial_and_disjoint', 'per_task_rng_report_diagnostics_replayed',
                   'first_allocation_failure_remains_at_original_task_boundary')


def _require(ok, detail):
    if not ok:
        raise ValueError('CPU task arena plan: ' + detail)


def _lane_partition(scope, decision, cpus):
    count = scope['task_count']
    width = decision['worker_count']
    _require(type(count) is type(width) is int and 1 < width <= count < 2**63
             and len(cpus) == len(set(cpus)) == width and all(type(cpu) is int and cpu >= 0 for cpu in cpus)
             and cpus == decision['worker_cpu_ids'] and cpus[0] == decision['owner_cpu_id']
             and set(cpus) <= set(decision['launch_cpu_ids']) and width <= scope['requested_workers'],
             'fixed OpenMP task count and worker cohort required')
    quotient, remainder = divmod(count, width)
    return [dict(lane_index=rank, cpu_id=cpu, first_ordinal=rank, stride=width,
                 task_count=quotient + int(rank < remainder)) for rank, cpu in enumerate(cpus)]


def _resource_plan(runtime, scope, decision, arena_bytes, lane_count):
    from .independent_array_task_schedule import RESULT_BYTES
    from .independent_array_task_team_schedule import STACK_BYTES

    binary = runtime['files'][runtime['binary']]
    elf = read_artifact(binary, json_document=False)
    # The emitter must also assert the owner-slot ABI. Do not reuse ELF class
    # alone as a C++ pointer-size fact on another architecture.
    if not (elf[:7] == b'\x7fELF\x02\x01\x01' and elf[18:20] == b'\x3e\x00'):
        return None
    pointer_bytes = 8
    memory = decision['memory']
    count = scope['task_count']
    payload = dict(arena_bytes=arena_bytes * lane_count, result_bytes=RESULT_BYTES * count,
                   status_bytes=4 * count, configured_child_stack_bytes=STACK_BYTES * (lane_count - 1))
    old_subaccount = sum(payload.values())
    owner_metadata = pointer_bytes * lane_count
    new_subaccount = old_subaccount + owner_metadata
    limit = memory['coordinator_address_space_limit_bytes']
    _require(memory['bound_scope'] == 'SINGLE_PROCESS_VAS_CAP_INCLUDING_PRIVATE_ARRAYS_AND_OPENMP_CHILD_STACKS'
             and memory['physical_reservation'] is False and type(limit) is int and limit > 0
             and memory['required_headroom_bytes'] == limit
             and memory['private_arena_bytes_per_numerical_worker'] == arena_bytes
             and memory['explicit_worker_payload_bytes'] == arena_bytes and memory['result_bytes'] == RESULT_BYTES
             and memory['retained_result_payload_bytes'] == payload['result_bytes']
             and memory['retained_status_bytes'] == payload['status_bytes'] and memory['thread_stack_bytes'] == STACK_BYTES
             and memory['explicit_owner_payload_bytes'] == old_subaccount and old_subaccount < limit,
             'arena, result, status and configured stack subaccounts differ from the admitted source')
    return dict(payload, binary_abi_evidence=binary, owner_slot_size_bytes=pointer_bytes,
        lane_owner_metadata_pointer_words=lane_count, lane_owner_metadata_bytes=owner_metadata,
        original_declared_vas_subaccount_bytes=old_subaccount,
        candidate_declared_vas_subaccount_bytes=new_subaccount, coordinator_soft_vas_limit_bytes=limit,
        candidate_tracked_subaccount_fits_vas_limit=new_subaccount < limit,
        whole_process_vas_usage_bound_bytes=None, physical_residency_bytes=None,
        candidate_arena_lifetime_bytes=dict(before_first_task=0, active_tasks_upper=payload['arena_bytes'],
            after_join_before_bulk_release=payload['arena_bytes'], at_publication=0))


def derive_cpu_task_arena_plan(proof_ref):
    """Return exact allocation-count savings and a replacement for explicit validation."""
    from scar.analysis.cpu_task_arena_reuse import verify_cpu_task_arena_reuse

    source = verify_cpu_task_arena_reuse(proof_ref)
    runtime_ref = source['runtime']
    report = dict(schema=SCHEMA, schema_version=1, proof=proof_ref, runtime=runtime_ref,
        status='REJECTED', reason='OPEN_ARENA_REUSE_PROOF', transformation=None, scope=source.get('scope'),
        worker_cpu_ids=source.get('worker_cpu_ids'), arena=None, placement=None, allocation_counts=None,
        resource_plan=None, failure_contract=None, elapsed_gain_ns=None, profitable=None,
        execution_default='NO_OP', whole_entry_equivalence_proven=False, deployment_authorized=False)
    if source['status'] != 'PROVEN_SCOPED_NORMAL_SUCCESS':
        return report
    proof = source['proof']
    if (proof.get('scope') != 'GENERATED_CPP20_FIXED_OPENMP_NORMAL_SUCCESS'
            or not all(proof.get(name) is True for name in _SUCCESS_PROOFS)):
        return report
    if (proof.get('diagnostic_environment_variables') != list(_DIAGNOSTICS)
            or proof.get('diagnostic_contract') != 'SITES_AND_PAYLOADS_PRESERVED_ON_NORMAL_SUCCESS'):
        return dict(report, reason='OPEN_DIAGNOSTIC_EFFECT_CONTRACT')
    if (proof.get('first_allocation_failure_scope') != 'REQUIRED_EMITTER_OBLIGATION'
            or type(proof.get('removed_repeated_allocator_failure_equivalent')) is not bool
            or proof.get('measurement_metadata_semantics') != 'SCOPED_SCHEMA_ROLE_PRESERVED_VALUES_MAY_CHANGE'):
        return dict(report, reason='OPEN_FAILURE_OR_REPORT_CONTRACT')
    runtime = read_artifact(runtime_ref)
    scope, decision, cpus, arena = (source[key] for key in ('scope', 'decision', 'worker_cpu_ids', 'arena'))
    size, alignment = arena['bytes'], arena['alignment_bytes']
    _require(runtime_ref['schema'] == _RUNTIME and runtime['scope'] == scope and runtime['decision'] == decision
             and runtime['original_default_retained'] is True and runtime['deployment_authorized'] is False
             and runtime['cpu_ids'] == decision['launch_cpu_ids'] and decision['mode'] == 'ORDERED_OPENMP'
             and not decision['capacity_reasons'] and type(size) is type(alignment) is int
             and size > 0 and alignment > 0 and alignment & (alignment - 1) == 0 and size % alignment == 0,
             'arithmetic must use the exact admitted native provider and alignment')
    environment = dict(read_artifact(runtime['recipe'])['environment'], **runtime['launch_environment'])
    if (any(name in environment for name in _DIAGNOSTICS) or environment.get('OMP_STACKSIZE') != '8M'
            or 'GOMP_STACKSIZE' in environment):
        return dict(report, reason='DIAGNOSTICS_OR_UNQUALIFIED_STACK_ENVIRONMENT_ACTIVE')
    lanes = _lane_partition(scope, decision, cpus)
    count = scope['task_count']
    width = len(lanes)
    _require(sum(row['task_count'] for row in lanes) == count, 'closed-form lane partition differs from task count')
    counts = dict(original=count, candidate=width, eliminated=count - width, scope='ALL_TASKS_COMPLETE_SUCCESSFULLY')
    report['allocation_counts'] = counts
    if counts['eliminated'] == 0:
        return dict(report, reason='NO_REPEATED_TASKS_ON_ANY_LANE')
    resources = _resource_plan(runtime, scope, decision, size, width)
    if resources is None:
        return dict(report, reason='OWNER_SLOT_ABI_UNRESOLVED')
    if not resources['candidate_tracked_subaccount_fits_vas_limit']:
        return dict(report, reason='CANDIDATE_TRACKED_VAS_SUBACCOUNT_EXCEEDS_LIMIT', resource_plan=resources)
    return dict(report, status=ELIGIBLE, reason='FEWER_ALLOCATIONS_WITH_PROVED_SCOPED_PRIVATE_ARENA',
        transformation=TRANSFORMATION, resource_plan=resources,
        arena=dict(bytes=size, alignment_bytes=alignment, clear_on_reuse=False,
            acquire_at='ORIGINAL_TASK_ARENA_ALLOCATION_SITE_AFTER_PER_TASK_PREFIX',
            release_at='AFTER_SYNCHRONOUS_TEAM_JOIN_BEFORE_PUBLICATION'),
        placement=dict(lanes=lanes, assignment='ORDINAL_MOD_WORKER_COUNT', per_lane_task_order='ASCENDING_ORDINAL',
            seed_rule='ORIGINAL_BASE_SEED_PLUS_TASK_ORDINAL', publish_after_join=True,
            require_owner_slot_size_bytes=resources['owner_slot_size_bytes'], require_environment_unset=list(_DIAGNOSTICS)),
        failure_contract=dict(scope='NORMAL_SUCCESS_SCOPED_VALIDATION',
            attempted_allocation_failure='ORIGINAL_TASK_STATUS_AT_THE_ORIGINAL_ALLOCATION_SITE',
            leave_slot_empty_on_failed_allocation=True, preserve_ordered_task_status_handling=True,
            removed_repeated_allocator_failure_equivalent=proof['removed_repeated_allocator_failure_equivalent'],
            measurement_metadata=proof['measurement_metadata_semantics']))


def verify_cpu_task_arena_plan(reference):
    saved = read_artifact(reference)
    _require(reference['schema'] == SCHEMA and type(saved['schema_version']) is int and saved['schema_version'] == 1,
             'unsupported CPU task arena plan')
    replay = derive_cpu_task_arena_plan(saved['proof'])
    _require(replay == saved, 'saved arena placement differs at cold replay')
    return replay
