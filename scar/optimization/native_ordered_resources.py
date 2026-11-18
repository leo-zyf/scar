"""Project an admitted ordered sparse recipe onto a raw native caller boundary."""

from scar.ir.execution_recipe import read_artifact, validate_recipe

from .resource_solver import semantic_sha256


SCHEMA = 'scar.native-ordered-resources'
_PROFILE = 'cpu-ordered-sparse-solve-v1'
_CHOICE = 'scar.sparse-parallel-preliminary-choice'
_HARDWARE = 'scar.sparse-parallel-hardware'


def project_native_ordered_resources(recipe):
    """Derive a launch policy after the caller has replayed the source recipe.

    The result is a projection of retained evidence. Live admission and the
    runtime's ABI and process isolation require separate lowering checks.
    """
    recipe = validate_recipe(recipe, verify_artifacts=False)
    if recipe['profile'] != _PROFILE:
        raise ValueError('the native projection requires an ordered sparse recipe')
    resources = recipe['resources']
    case, cpu, dispatch, memory = (resources[key] for key in ('case', 'selected_cpu', 'dispatch', 'memory'))
    control = recipe['control']
    count, n, nnz = (case[key] for key in ('call_count', 'size', 'nnz'))
    window = dispatch['window']
    if (any(type(value) is not int or not 0 < value < 2**31 - 1 for value in (count, n, nnz, window))
            or not 1 < window <= count or dispatch['one_inflight_per_worker'] is not True
            or type(dispatch['worker_threads']) is not int or dispatch['worker_threads'] != 1
            or control['window'] != window or control['iteration'] != dict(axis='i', count=count,
                unrolled=False, shared_body_instances=1) or control.get('process_isolated_worker_offers') is not True):
        raise ValueError('one bounded process slot per worker and one original loop body required')

    choice_ref, hardware_ref = recipe['artifacts']['choice'], resources['hardware']
    choice, hardware = read_artifact(choice_ref), read_artifact(hardware_ref)
    if ((choice_ref['schema'], hardware_ref['schema']) != (_CHOICE, _HARDWARE)
            or choice['schema_version'] != 1 or hardware['schema_version'] != 1
            or choice['hardware'] != hardware_ref):
        raise ValueError('the native policy requires its original team and hardware receipts')
    owner, worker_ids, node = (cpu[key] for key in ('owner_cpu_id', 'worker_cpu_ids', 'cpu_numa_node'))
    cohort = [owner, *worker_ids]
    team = choice['choice']
    if (any(type(value) is not int or value < 0 for value in (*cohort, node))
            or len(worker_ids) != window or len(set(cohort)) != len(cohort)
            or cpu['launch_cpu_ids'] != sorted(cohort)
            or (owner, worker_ids, node, window) != (team['owner_cpu_id'], team['worker_cpu_ids'],
                team['cpu_numa_node'], team['window'])
            or not set(cohort) <= set(hardware['launching_affinity'])
            or node not in hardware['allowed_memory_nodes']):
        raise ValueError('worker placement differs from the retained permitted CPU cohort')

    processors = {row['cpu_id']: row for row in hardware['processors']}
    if len(processors) != len(hardware['processors']) or not set(cohort) <= processors.keys():
        raise ValueError('unique retained processor evidence required for every selected CPU')
    lanes = []
    for cpu_id in cohort:
        row = processors[cpu_id]
        if (row['permitted'] is not True or row['numa_nodes'] != [node]
                or row['package_id'] is None or row['core_id'] is None
                or not row['siblings'] or set(row['siblings']).intersection(cohort) != {cpu_id}):
            raise ValueError('selected CPUs must be permitted, same-NUMA and physically distinct')
        lanes.append({key: row[key] for key in ('cpu_id', 'package_id', 'die_id', 'core_id', 'siblings')})
    physical = {(row['package_id'], row['die_id'], row['core_id']) for row in lanes}
    if len(physical) != len(cohort):
        raise ValueError('selected CPUs share a physical core')

    limit, symbolic, floor = (memory[key] for key in ('worker_address_space_limit_bytes',
        'per_worker_symbolic_budget_bytes', 'minimum_launch_headroom_bytes'))
    structure_bytes = 4 * (nnz + n + 1)
    pending_input_bytes, pending_result_bytes = window * 8 * (nnz + n), window * 8 * n
    workers_bytes = window * limit if type(limit) is int else None
    if (any(type(value) is not int or value <= 0 for value in (limit, symbolic, floor))
            or symbolic > limit or limit != choice['budget']['worker_total_limit_bytes']
            or memory['worker_address_space_sum_bytes'] != workers_bytes
            or memory['owner_structure_bytes'] != structure_bytes
            or memory['owner_pending_input_bytes'] != pending_input_bytes
            or memory['owner_pending_result_bound_bytes'] != pending_result_bytes
            or floor < workers_bytes + structure_bytes + pending_input_bytes + pending_result_bytes):
        raise ValueError('the retained structure and pending-frame budgets differ from the recipe')

    return dict(schema=SCHEMA, schema_version=1, recipe_id=recipe['recipe_id'],
        source_control_sha256=semantic_sha256(control), case=case,
        placement=dict(hardware=hardware_ref, choice=choice_ref, launch_cpu_ids=cpu['launch_cpu_ids'],
            owner_cpu_id=owner, worker_cpu_ids=worker_ids, cpu_numa_node=node, owner=lanes[0],
            workers=[dict(slot=slot, **lane) for slot, lane in enumerate(lanes[1:])]),
        dispatch=dict(window=window, worker_threads=1, one_inflight_per_worker=True,
            worker_isolation='PROCESS_PER_CPU_SLOT', issue_order='ASCENDING_ZERO_BASED',
            retirement_order='OLDEST_OUTSTANDING_ONLY', publish_on='ZERO_STATUS_OF_EXPECTED_ISSUE',
            refill='AFTER_ORDERED_SUCCESS_PUBLICATION', failed_issue='TERMINATE_WITHOUT_PUBLISHING_SUFFIX',
            input='CALLER_IMMUTABLE_CSR_STRUCTURE_AND_ORDERED_RAW_NUMERIC_FRAMES',
            output='PRIVATE_UNTIL_ORDERED_SUCCESS'),
        memory=dict(worker_address_space_limit_bytes=limit, per_worker_symbolic_budget_bytes=symbolic,
            worker_address_space_sum_bytes=workers_bytes, owner_structure_bytes=structure_bytes,
            owner_pending_input_bytes=pending_input_bytes, owner_pending_result_bound_bytes=pending_result_bytes,
            minimum_launch_headroom_bytes=floor, headroom_basis='INHERITED_CONSERVATIVE_RECIPE_GATE',
            worker_native_scratch_bound_bytes=None, owner_transient_scratch_bound_bytes=None,
            physical_reservation=False, physical_numa_residency_proven=False),
        launch=dict(affinity='LIVE_MASK_CONTAINS_SELECTED_COHORT_THEN_PIN_OWNER_AND_WORKER_LANES',
            matching_live_topology_required=True, live_cpu_quota_and_memory_headroom_required=True,
            per_worker_limit='RLIMIT_AS', blas='LIVE_SINGLE_THREAD_PROVIDER_CHECK', automatic_admission=False),
        cost=dict(basis='UNKNOWN', startup_ns=None, per_call_ns=None, close_ns=None, whole_scope_ns=None))
