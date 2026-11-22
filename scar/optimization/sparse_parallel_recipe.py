"""Sparse parallel recipe."""

from contextlib import nullcontext
from pathlib import Path
from scar.analysis.replay_transaction import current_transaction, replay_transaction
from scar.analysis.sparse_parallel_recipe import derive_recipe
from scar.analysis.sparse_parallel_scope import SCHEMA as SCOPE, check_sparse_parallel_scope
from scar.compiler.model import LoopAxis
from scar.ir.execution_recipe import (
    make_recipe,
    read_artifact,
    read_recipe,
    validate_recipe,
    SCHEMA,
)
from scar.optimization.resource_solver import semantic_sha256
from scar.optimization.schedule import (
    LogicalValue,
    Materialization,
    ResourceSchedule,
    Task,
    TaskKind,
)
from scar.optimization.sparse_solve_recipe import PROFILE as SERIAL_PROFILE
from collections import deque
from scar.optimization.sparse_solve_cost import _duration, _summary


# Lower a guarded original sparse loop to an ordered, bounded worker graph.
PROFILE = 'cpu-ordered-sparse-solve-v1'
FAMILY = 'CPP20_ORDERED_PROCESS_ISOLATED_SPARSE_SOLVE'
SERIAL_LAUNCH = 'scar.sparse-direct-solve-launch'
CHOICE = 'scar.sparse-parallel-preliminary-choice'
DETACHED = 'scar.sparse-parallel-detached-recipe'
HARDWARE = 'scar.sparse-parallel-hardware'
_SCHEMAS = dict(parallel_scope=SCOPE, serial_launch=SERIAL_LAUNCH, choice=CHOICE, detached=DETACHED)
_SCOPE = '$ORDERED_SPARSE_ACTIVATION'
_ONCE = 'ONCE_PER_SCOPE'
_CALL = 'PER_CALL'
_EXIT = 'PER_SCOPE_EXIT'
_ENTRY = 'scar.runtime.sparse_parallel_entry.'
_POOL = 'scar.runtime.sparse_parallel_lu.'
_PREDICTION = 'scar.runtime.sparse_launch.DetachedCSR.'
_ORIGINAL = 'ORIGINAL_SOURCE_ROW_EDIT_AND_NORMALIZATION'


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _serial_evidence(reference, scope):
    launch = read_artifact(reference)
    study = read_artifact(launch['study'])
    recipe = validate_recipe(read_artifact(launch['execution_recipe']))
    lowering = read_artifact(launch['lowering'])
    context = launch['context']
    entry = scope['entry']
    _require(launch['schema_version'] == 1 and launch['decision'] == 'LOAD_NATIVE_CANDIDATE'
        and recipe['profile'] == SERIAL_PROFILE
        and launch['entry'] == study['entry'] == entry['source']
        and context['argv'] == entry['argv'] and context['cpu_ids'] == entry['affinity']
        and context == study['launch_context'] and all(context['environment'].get(key) == value
            for key, value in entry['thread_environment'].items())
        and study['source'] == recipe['artifacts']['source_scope'] == scope['solve_scope']
        and study['execution_recipe'] == launch['execution_recipe'] == lowering['recipe']
        and study['lowering'] == launch['lowering'] and study['native'] == lowering['native']
        and study['budget_bytes'] == recipe['resources']['memory']['explicit_budget_bytes'],
        'the released serial path does not match the original sparse loop and its live context')
    capability = recipe['implementation']['backend_capability_record']
    native = read_artifact(study['native'])
    _require(lowering['schema_version'] == 1 and lowering['recipe_replayed'] is True
        and lowering['recipe_id'] == recipe['recipe_id']
        and lowering['recipe_sha256'] == launch['execution_recipe']['sha256']
        and lowering['input_artifacts'] == recipe['artifacts']
        and lowering['backend_capability_record'] == capability
        and all(lowering[field + '_sha256'] == semantic_sha256(recipe[field])
            for field in ('semantic_boundary', 'control', 'task_graph', 'resources'))
        and native['abi_version'] == capability['abi_version'] == lowering['native_abi_version']
        and native['provider_version'] == capability['provider_version'] and native['language'] == 'c++20'
        and dict(native['wheel'], schema='scar.native-library-file') == capability['original_carrier']
        and native['blas'] == capability['blas'] and native['library'] == lowering['native_library'],
        'the released serial solver requires its exact typed lowering and native ABI')
    read_artifact(native['library'], json_document=False)
    return launch, study, recipe


def _storage_requirements(size, nnz, window, cap, symbolic_budget):
    structure = 4 * (nnz + size + 1)
    per_window_input, per_window_output = window * 8 * (nnz + size), window * 8 * size
    detached = 12 * nnz + 28 * size + 4
    total_cap = window * cap
    return dict(worker_address_space_limit_bytes=cap, per_worker_symbolic_budget_bytes=symbolic_budget,
        worker_address_space_sum_bytes=total_cap, owner_pending_input_bytes=per_window_input,
        owner_pending_result_bound_bytes=per_window_output, owner_structure_bytes=structure,
        owner_detached_snapshot_bound_bytes=detached,
        minimum_launch_headroom_bytes=total_cap + per_window_input + per_window_output + structure + detached,
        worker_native_scratch_bound_bytes=None, owner_transient_scratch_bound_bytes=None,
        physical_reservation=False, physical_numa_residency_proven=False)


def _resources(artifacts, scope, count, symbolic_budget):
    choice = read_artifact(artifacts['choice'])
    from .sparse_parallel_trial_choice import BASIS, check_trial_choice

    if choice.get('basis') == BASIS:
        _require(choice['parallel_scope'] == artifacts['parallel_scope']
            and choice['serial_launch'] == artifacts['serial_launch'], 'calibration seed belongs to another sparse recipe')
        check_trial_choice(artifacts['choice'])
    hardware = read_artifact(choice['hardware'])
    team = choice['choice']
    owner, workers, numa = team['owner_cpu_id'], team['worker_cpu_ids'], team['cpu_numa_node']
    window = team['window']
    _require(choice['schema_version'] == hardware['schema_version'] == 1
        and choice['hardware']['schema'] == HARDWARE and choice['empirical_resource_reservation'] is False
        and team['resource_snapshot_is_reservation'] is team['automatic_admission'] is False
        and all(type(cpu) is int and cpu >= 0 for cpu in [owner, *workers])
        and owner not in workers and len(set(workers)) == len(workers) == team['worker_count'] == window
        and type(window) is int and 1 < window <= min(count, choice['dimensions']['max_ordered_window'])
        and choice['dimensions']['call_count'] == count and scope['entry']['affinity'] == [owner],
        'worker choice must retain the original owner and use a finite ordered window')
    cpus = sorted([owner, *workers])
    processors = [row for row in hardware['processors'] if row['cpu_id'] in cpus]
    physical = {(row['package_id'], row['die_id'], row['core_id']) for row in processors}
    _require(sorted(row['cpu_id'] for row in processors) == cpus and len(physical) == len(cpus)
        and all(row['permitted'] is True and row['core_id'] is not None
            and row['numa_nodes'] == [numa] and row['cpu_id'] in row['siblings'] for row in processors)
        and set(cpus) <= set(hardware['launching_affinity']) and numa in hardware['allowed_memory_nodes']
        and all(value is False for value in (hardware['memory_reservation'], hardware['memory_placement_enforced'],
            team['uses_smt_siblings_as_extra_cores'])),
        'selected team no longer identifies distinct same-NUMA physical cores')
    cap = choice['budget']['worker_total_limit_bytes']
    _require(type(cap) is int and type(symbolic_budget) is int and 0 < symbolic_budget <= cap,
        'an enforceable per-worker address-space cap and separate symbolic budget are required')
    structure = read_artifact(scope['solve_scope'])['sampled_values']['first_structure']
    indices, indptr = structure['indices'], structure['indptr']
    n, nnz = indptr['shape'][0] - 1, indices['shape'][0]
    _require(0 < n < 2**31 - 1 and 0 < nnz < 2**31 - 1
        and indices['shape'] == [nnz] and indptr['shape'] == [n + 1]
        and indices['dtype'] == indptr['dtype'] == '<i4'
        and indices['nbytes'] == 4 * nnz and indptr['nbytes'] == 4 * (n + 1),
        'the scaled CSR case must preserve the original 32-bit shape and exact structure extent')
    return dict(case=dict(call_count=count, size=n, nnz=nnz), hardware=choice['hardware'],
        selected_cpu=dict(owner_cpu_id=owner, worker_cpu_ids=workers, launch_cpu_ids=cpus, cpu_numa_node=numa),
        dispatch=dict(window=window, one_inflight_per_worker=True,
            courier_cpu_policy='INHERITED_OWNER_AFFINITY', worker_threads=1),
        memory=_storage_requirements(n, nnz, window, cap, symbolic_budget))


def _schedule(operation, resources, symbols):
    cpu = resources['selected_cpu']
    owner = dict(kind='OWNER_CPU', cpu_id=cpu['owner_cpu_id'])
    courier = dict(kind='IPC_COURIER_SHARED_OWNER_CPU', cpu_id=cpu['owner_cpu_id'])
    worker = dict(kind='ISOLATED_WORKER', cpu_ids=cpu['worker_cpu_ids'], assignment='RETIRED_IDLE_SLOT')
    host = 'owner-address-space:$CALL_PROCESS'
    child = 'worker-address-space:$SELECTED_SLOT'
    values, materials, tasks = [], [], []

    def value(name, producer=None, allocated_by=None, *, location=host, lifetime=_CALL, bytes_expr=None,
              escape=False, **attrs):
        values.append(LogicalValue(name, producer=producer, escape=escape, call_scope=_SCOPE,
            attrs=dict(lifetime=lifetime, coordinate='i' if lifetime == _CALL else 'activation', **attrs)))
        place = worker if location == child else owner
        materials.append(Materialization(name, name, location, (place,), ready_task=producer,
            external_ready=producer is None, allocated_by=allocated_by,
            ownership='EXTERNAL' if producer is None else 'SCHEDULE', bytes_expr=bytes_expr))

    def task(name, kind, lifetime, entrypoint, *, reads=(), writes=(), deps=(), place=(owner,), **attrs):
        tasks.append(Task(name, kind, tuple(reads), tuple(writes), tuple(deps), place,
            dict(lifetime=lifetime, entrypoint=entrypoint, duration_ns=None, **attrs)))

    value('source-live-ins', lifetime=_ONCE)
    value('detached-live-ins', 'capture-live-ins', 'allocate-live-ins', lifetime=_ONCE,
        bytes_expr=str(resources['memory']['owner_detached_snapshot_bound_bytes']), payload_is_upper_bound=True)
    value('owner-structure', 'bootstrap-team', 'allocate-team', lifetime=_ONCE,
        bytes_expr='4 * (nnz + n + 1)')
    value('worker-structure', 'bootstrap-team', 'allocate-team', location=child, lifetime=_ONCE,
        bytes_expr='4 * (nnz + n + 1)', replicated_per_worker=True)
    value('worker-symbolic-plan', 'bootstrap-team', 'allocate-team', location=child, lifetime=_ONCE,
        replicated_per_worker=True, contains_only_owned_structure=True)
    value('original-arguments', 'prepare-original', 'allocate-original')
    call_values = (
        ('private-arguments', 'prepare-private', host, '8 * (nnz + n)'),
        ('owner-input-packet', 'submit', host, '8 * (nnz + n)'),
        ('worker-input-packet', 'deliver-request', child, '8 * (nnz + n)'),
        ('worker-result-packet', 'solve-worker', child, '8 * n'),
        ('owner-result', 'deliver-result', host, '8 * n'),
        ('matched-offer', 'compare-request', host, None),
        ('publishable-result', 'retire-matched', host, None),
    )
    for name, producer, location, size in call_values:
        aliases = dict(storage_alias='owner-result', shared_backing=True) if name == 'publishable-result' else {}
        value(name, producer, 'allocate-call', location=location, bytes_expr=size,
            escape=name == 'publishable-result', **aliases)

    compute, alloc = TaskKind.COMPUTE, TaskKind.ALLOCATION
    team_storage = ('owner-structure', 'worker-structure', 'worker-symbolic-plan')
    task('admit-entry', compute, _ONCE, _ENTRY + '_admit',
        conditions='FROZEN_ENTRY_CONTEXT_EXCLUSIVE_AND_LIVE_RESOURCE_GUARDS')
    task('allocate-live-ins', alloc, _ONCE, _PREDICTION + '__init__',
        writes=('detached-live-ins',), deps=('admit-entry',))
    task('capture-live-ins', compute, _ONCE, _ENTRY + '_OrderedOffers.capture', reads=('source-live-ins',),
        writes=('detached-live-ins',), deps=('allocate-live-ins',), observer='CPYTHON_ONE_SHOT_PY_START')
    task('allocate-team', alloc, _ONCE, _POOL + '_ParallelSolver.prepare', writes=team_storage,
        deps=('capture-live-ins',), place=(owner, courier, worker), storage_allocation_only=True)
    task('bootstrap-team', compute, _ONCE, _POOL + '_ParallelSolver.prepare', reads=('detached-live-ins',),
        writes=team_storage, deps=('allocate-team',), place=(owner, courier, worker), native_abi=symbols['create'],
        requires_first_original_call=True, validates_serial_first_result_bitwise=True,
        sends_structure_one_time_to_each_worker=True, workers_require_one_thread_and_default_fp=True)
    task('allocate-original', alloc, _CALL, _ORIGINAL, writes=('original-arguments',), deps=('capture-live-ins',))
    task('prepare-original', compute, _CALL, _ORIGINAL, reads=('source-live-ins',),
        writes=('original-arguments',), deps=('allocate-original',), original_effects_preserved=True)
    task('allocate-call', alloc, _CALL, _POOL + '_ParallelSolver.submit',
        writes=tuple(row[0] for row in call_values), deps=('bootstrap-team',), place=(owner, courier, worker),
        storage_allocation_only=True, native_scratch_bound_bytes=None)
    task('prepare-private', compute, _CALL, _PREDICTION + 'args', reads=('detached-live-ins',),
        writes=('private-arguments',), deps=('allocate-call',), original_preparation_preserved=True)
    task('submit', compute, _CALL, _POOL + '_ParallelSolver.submit', reads=('private-arguments', 'owner-structure'),
        writes=('owner-input-packet',), deps=('prepare-private',), bounded_slot=True, exact_structure_guard=True)
    task('deliver-request', compute, _CALL, _POOL + '_Worker.exchange', reads=('owner-input-packet',),
        writes=('worker-input-packet',), deps=('submit',), place=(courier, worker), transport='LOCAL_SOCKET')
    task('solve-worker', compute, _CALL, symbols['apply'],
        reads=('worker-structure', 'worker-symbolic-plan', 'worker-input-packet'), writes=('worker-result-packet',),
        deps=('deliver-request',), place=(worker,), native_scratch_bound_bytes=None,
        recomputes=('current_column_binding', 'numeric_factor', 'row_pivots', 'triangular_solve'),
        forwards_result_only_on_zero_status=True)
    task('deliver-result', compute, _CALL, _POOL + '_ParallelSolver._execute', reads=('worker-result-packet',),
        writes=('owner-result',), deps=('solve-worker',), place=(courier, worker), transport='LOCAL_SOCKET')
    task('dequeue-oldest', compute, _CALL, _POOL + '_ParallelSolver._retire',
        reads=('original-arguments', 'owner-input-packet'), deps=('prepare-original', 'submit'),
        returns_pending_credit=False, captures_current_arguments_before_future_wait=True)
    task('compare-request', compute, _CALL, _POOL + '_ParallelSolver.consume',
        reads=('original-arguments', 'owner-structure', 'owner-input-packet', 'owner-result'),
        writes=('matched-offer',), deps=('dequeue-oldest', 'deliver-result'),
        guard='EXACT_LIVE_ABI_OPTIONS_STRUCTURE_VALUES_RHS_AND_NATIVE_SUCCESS')
    task('retire-matched', compute, _CALL, _ENTRY + '_OrderedOffers.__call__',
        reads=('matched-offer', 'owner-result'), writes=('publishable-result',), deps=('compare-request',),
        grants_refill_credit=True, only_on_success=True)
    task('release-call', TaskKind.RELEASE, _CALL, _POOL + '_ParallelSolver.consume',
        reads=('private-arguments', 'owner-input-packet', 'worker-input-packet', 'worker-result-packet', 'matched-offer'),
        deps=('retire-matched',), place=(owner, courier, worker), service_includes_no_extra_deallocation_call=True)
    task('publish-original', TaskKind.EXPORT, _CALL, 'ORIGINAL_SOURCE_STATUS_BRANCH', reads=('publishable-result',),
        deps=('release-call',), retains_original_status_warnings_and_accumulator=True)
    task('close-team', TaskKind.RELEASE, _EXIT, _ENTRY + '_OrderedOffers.close',
        reads=('detached-live-ins', *team_storage), deps=('publish-original',), place=(owner, courier, worker),
        native_abi=symbols['destroy'], joins_workers_and_couriers=True, can_discard_pending_on_fallback=True)
    return ResourceSchedule(operation, tuple(values), tuple(materials), tuple(tasks),
        loop_axes=(LoopAxis('i', 'call_count'),), call_scope=_SCOPE, executable=True,
        proof=dict(profile=PROFILE, normal_success_path_only=True, shared_body=True,
            owner_credit_released_after_exact_match=True, unknown_duration_does_not_authorize_publication=True))


def _control(count, window):
    def edge(source, src_index, target, dst_index, kind, when):
        return {'from':dict(task=source, index=src_index), 'to':dict(task=target, index=dst_index),
            'kind':kind, 'when':when}

    edges = [
        edge('prepare-original', '0', 'allocate-team', 'activation', 'ORIGINAL_ACTIVATION', 'FIRST_ELIGIBLE_CALL'),
        edge('submit', 'j', 'dequeue-oldest', '0', 'INITIAL_FILL', '0 <= j < W'),
        edge('retire-matched', 'i', 'prepare-private', 'i+W', 'SLOT_CREDIT', 'i+W < count'),
        edge('submit', 'i+W', 'publish-original', 'i', 'REFILL_BEFORE_ORIGINAL_RETURN', 'i+W < count'),
        edge('publish-original', 'i', 'prepare-original', 'i+1', 'ORIGINAL_WORLD_BACKEDGE', 'i+1 < count'),
        edge('dequeue-oldest', 'i', 'dequeue-oldest', 'i+1', 'ISSUE_ORDER', 'i+1 < count'),
        edge('publish-original', 'count-1', 'close-team', 'activation', 'DRAIN_THEN_CLOSE', 'NORMAL_COMPLETION'),
    ]
    return dict(iteration=dict(axis='i', count=count, unrolled=False, shared_body_instances=1), window=window,
        cross_instance_dependencies=edges,
        activation='CAPTURE_ONCE_AT_ORIGINAL_FUNCTION_ENTRY_THEN_MATCH_FIRST_RAW_CALL',
        retirement='OLDEST_ONLY_CURRENT_OPERAND_COMPARISON_THEN_REFILL_BEFORE_ORIGINAL_RETURN',
        fallback='STOP_SPECULATION_THEN_TRY_GUARDED_SERIAL_THEN_ORIGINAL_CARRIER',
        publication='ORIGINAL_PYTHON_VISIBLE_EFFECTS_ONLY', process_isolated_worker_offers=True)


def _assemble(artifacts):
    _require(set(artifacts) == set(_SCHEMAS) and all(artifacts[role].get('schema') == schema
        for role, schema in _SCHEMAS.items()), 'parallel recipe requires exact source, released serial, choice and detached references')
    scope = check_sparse_parallel_scope(artifacts['parallel_scope'])
    serial, study, serial_recipe = _serial_evidence(artifacts['serial_launch'], scope)
    contract = scope['speculation_contract']
    entry = scope['entry']['source']
    exact = scope['loop']['exact_count_under_source_conditions']
    count = exact['repetitions']
    _require(exact['normal_scope_completion'] is True and exact['executions_per_normal_iteration'] == 1
        and count == read_artifact(scope['source_scope'])['observed']['consumer_completions']
        and count == serial_recipe['control']['iteration']['count']
        and contract['raw_inputs_always_revalidated'] is True and contract['advanced_work_has_no_visible_publication'] is True,
        'parallel recipe requires one original call per normally completed iteration and exact live matching')
    detached = derive_recipe(read_artifact(entry, json_document=False), scope['source_body'], scope['preparation'],
        optimize=serial['context']['interpreter']['optimize'])
    _require(read_artifact(artifacts['detached']) == dict(detached, schema=DETACHED, schema_version=1,
        source=artifacts['parallel_scope']), 'detached preparation differs from original source replay')
    resources = _resources(artifacts, scope, count, study['budget_bytes'])
    prior = serial_recipe['implementation']['backend_capability_record']
    symbols = {key:prior['symbols'][key] for key in ('create', 'apply', 'destroy')}
    cap = dict(schema='scar.sparse-ordered-backend-capability', schema_version=1, family=FAMILY,
        serial_recipe=serial['execution_recipe'], serial_lowering=serial['lowering'], native=study['native'],
        symbols=symbols, original_carrier=prior['original_carrier'], blas=prior['blas'],
        abi_version=prior['abi_version'], cxx_standard=prior['cxx_standard'])
    call = contract['original_raw_dispatch']
    operation = 'ordered-sparse-solve:' + artifacts['parallel_scope']['sha256']
    boundary = dict(id=operation, original_entry=entry, source_scope=scope['source_scope'],
        original_call=call['call_point'], original_code_key=call['code_key'],
        original_python_status_boundary=call['original_python_boundary'], original_loop=scope['loop']['original_operation'],
        original_body_code_key=detached['body_code_key'], original_preparation_spans=detached['source_spans'],
        original_effects=contract['original_effects_preserved'], entry_context=serial['context'],
        Q=dict(acceptance_key=contract['acceptance_key'], deterministic_private_native_only=True,
            workers_publish_no_effects=True, effects_after_original_exact_match_only=True))
    implementation = dict(family=FAMILY, backend_inputs={role:role for role in _SCHEMAS},
        backend_capability_record=cap, implementation_option='PARALLEL', fallback_option='SERIAL',
        status='EXPLICIT_STUDY_ONLY', program_body_copies=1)
    obligations = [
        dict(id='source_alignment', status='PROVEN', detail='Original F4 loop, raw carrier and detached arithmetic are replayed.'),
        dict(id='live_resource_fit', status='RUNTIME_GUARD', detail='Require the same context, distinct permitted CPU lanes, and live cgroup and NUMA headroom.'),
        dict(id='live_request_match', status='RUNTIME_GUARD', detail='Original arguments, topology, numeric values and RHS must match before an ordered success is published.'),
        dict(id='isolation', status='LOWERING_CHECK', detail='Use independent pinned workers, the exact serial C++ ABI and a shared owner-affinity courier pool.'),
        dict(id='native_scratch_and_transport_cost', status='UNRESOLVED', detail='Scratch memory, wire framing, service times, contention and physical placement require independent evidence.'),
    ]
    return make_recipe(PROFILE, semantic_boundary=boundary, control=_control(count, resources['dispatch']['window']),
        task_graph=_schedule(operation, resources, symbols), resources=resources, implementation=implementation,
        artifacts=artifacts, obligations=obligations)


def make_sparse_parallel_recipe(scope_ref, serial_launch_ref, choice_ref, detached_ref):
    """Link a released serial solve to the current bounded ordered process pool."""
    with nullcontext() if current_transaction() is not None else replay_transaction():
        return _assemble(dict(parallel_scope=scope_ref, serial_launch=serial_launch_ref, choice=choice_ref,
            detached=detached_ref))


def verify_sparse_parallel_recipe(value):
    """Rebuild source provenance, resource placement and each retirement dependency."""
    recipe = read_recipe(value) if isinstance(value, (str, Path)) else validate_recipe(value)
    _require(recipe['profile'] == PROFILE, 'unsupported ordered sparse recipe profile')
    with nullcontext() if current_transaction() is not None else replay_transaction():
        _require(recipe == _assemble(recipe['artifacts']),
            'ordered sparse recipe differs from original source, resource policy or task replay')
    return recipe


# Bind ordered process service measurements to the typed sparse recipe.
_CUTS = {
    'submit_offer_offset_ns': ('submit', 'Predictor ready to the actual submission stamp.'),
    'queue_ns': ('deliver-request', 'Submission stamp to courier exchange start.'),
    'request_exchange_child_setup_ns': ('deliver-request', 'Courier exchange start to child native entry.'),
    'native_solve_ns': ('solve-worker', 'Child native entry to native return.'),
    'result_exchange_unpack_ns': ('deliver-result', 'Child native return to courier exchange completion.'),
    'publish_digest_proxy_ns': ('publish-original', 'Diagnostic output digest after retirement and refill.'),
}
_SUBPHASES = {
    'numeric_factorization_ns': 'Contained in solve-worker native wall time.',
    'triangular_solve_ns': 'Contained in solve-worker native wall time.',
    'worker_column_order_ns': 'Contained in worker symbolic construction during bootstrap.',
    'worker_preorder_ns': 'Contained in worker symbolic construction during bootstrap.',
}


def bind_calibration(design, source, parallel, choice, native):
    """Match the measured launch and its recipe without re-running the source compiler."""
    from scar.optimization.sparse_parallel_recipe import PROFILE

    reference = design['execution_recipe']
    recipe = validate_recipe(read_artifact(reference))
    artifacts, resources = recipe['artifacts'], recipe['resources']
    cpu, dispatch, memory, case = (resources[key] for key in ('selected_cpu','dispatch','memory','case'))
    detached = read_artifact(artifacts['detached'])
    body = {key: value for key,value in detached.items() if key not in ('schema','schema_version','source')}
    capability = recipe['implementation']['backend_capability_record']
    serial = read_artifact(artifacts['serial_launch'])
    study = read_artifact(serial['study'])
    if (reference['schema'] != SCHEMA or recipe['profile'] != PROFILE
            or artifacts['parallel_scope'] != design['parallel_scope']
            or artifacts['choice'] != design['choice'] or detached['source'] != design['parallel_scope']
            or body != design['recipe'] or parallel['source_scope'] != design['source_scope']
            or capability['native'] != study['native'] or study['native'] != design['native']
            or capability['blas'] != native['blas'] or study['source'] != parallel['solve_scope']
            or {**design['context'], 'cpu_ids':[cpu['owner_cpu_id']]} != serial['context']
            or cpu['launch_cpu_ids'] != design['context']['cpu_ids']
            or cpu['worker_cpu_ids'] != choice['worker_cpu_ids'] or cpu['owner_cpu_id'] != choice['owner_cpu_id']
            or cpu['cpu_numa_node'] != choice['cpu_numa_node'] or dispatch['window'] != choice['window']
            or dispatch['one_inflight_per_worker'] is not True or dispatch['worker_threads'] != 1
            or dispatch['courier_cpu_policy'] != 'INHERITED_OWNER_AFFINITY'
            or memory['worker_address_space_limit_bytes'] != design['worker_address_space_bytes']
            or memory['per_worker_symbolic_budget_bytes'] != design['symbolic_budget_bytes']
            or case['call_count'] != source['entry']['argv_repetitions']
            or {key:case[key] for key in ('size','nnz')} != design['matrix']):
        raise ValueError('ordered sparse clocks do not match the recipe, CSR shape or worker lease')
    return recipe


def new_phase_samples():
    return {key: [] for key in (*_CUTS, *_SUBPHASES)}


def append_task_phases(terms, task, sample):
    submitted = task['submitted_ns']
    exchange_start = task['exchange_started_ns']
    start, end = (task[name] for name in ('started_ns','ended_ns'))
    exchanged = task['exchange_complete_ns']
    publish = _duration(sample['publish_clock_ns'])
    terms['submit_offer_offset_ns'].append(submitted - sample['submit_clock_ns'][1])
    terms['queue_ns'].append(exchange_start-submitted)
    terms['request_exchange_child_setup_ns'].append(start-exchange_start)
    terms['native_solve_ns'].append(end-start)
    terms['result_exchange_unpack_ns'].append(exchanged-end)
    terms['publish_digest_proxy_ns'].append(publish)
    _append_nested(terms, task['phases'], end-start,
                   (('factor_s','numeric_factorization_ns'), ('solve_s','triangular_solve_ns')))
    return publish


def _append_nested(terms, record, container_ns, keys):
    values = [round(record[source]*1e9) for source, _ in keys]
    if any(value < 0 for value in values) or sum(values) > container_ns:
        raise ValueError('nested native clocks exceed their observed containing interval')
    for value, (_, name) in zip(values, keys):
        terms[name].append(value)


def append_bootstrap_phases(terms, workers):
    for worker in workers:
        _append_nested(terms, worker['phases'], worker['symbolic_build_ns'],
                       (('ordering_s','worker_column_order_ns'), ('etree_s','worker_preorder_ns')))


def check_group_order(samples, window):
    """Read owner cuts proving that refill occurred before publication."""
    if len(samples) <= window:
        raise ValueError('phase observations must exercise a real refill')
    for index, row in enumerate(samples):
        entered, _, ready, succeeded = row['consume_clock_ns']
        publish_start, publish_end = row['publish_clock_ns']
        if (not ready <= publish_start < publish_end == succeeded
                or index and entered < samples[index-1]['publish_clock_ns'][1]
                or index < window and row['submit_clock_ns'][-1] > samples[0]['consume_clock_ns'][0]):
            raise ValueError('observed publication or initial window differs from the ordered recipe')
        if index+window < len(samples):
            refill_start, _, refill_end = samples[index+window]['submit_clock_ns']
            if not ready <= refill_start < refill_end <= publish_start:
                raise ValueError('observed refill did not precede the original publication proxy')


def check_group_resources(recipe, observed):
    memory = recipe['resources']['memory']
    measured = (
        ('owner_peak_pending_input_bytes','owner_pending_input_bytes'),
        ('owner_pending_result_bound_bytes','owner_pending_result_bound_bytes'),
        ('owner_structure_bytes','owner_structure_bytes'),
        ('worker_virtual_address_caps_bytes','worker_address_space_sum_bytes'),
    )
    if any(observed[field] != memory[bound] for field, bound in measured):
        raise ValueError('ordered sparse observations do not match the recipe payload or worker caps')
    budget = memory['per_worker_symbolic_budget_bytes'] * observed['distinct_workers']
    if not 0 < observed['worker_symbolic_payload_bytes'] <= budget:
        raise ValueError('observed symbolic plan bytes exceed the retained budget')


def phase_cost_ledger(terms):
    return dict(measured={name: dict(recipe_task=task, interval=detail, **_summary(terms[name]))
                         for name, (task,detail) in _CUTS.items()},
                nested_diagnostics={name: dict(coverage=detail, **_summary(terms[name]))
                                    for name, detail in _SUBPHASES.items()},
                unresolved=['owner_result_array_materialization_and_future_ready',
                            'retire_exact_match_cpu_separate_from_await',
                            'numeric_scratch_peak', 'physical_ipc_bandwidth',
                            'original_publish_and_other_entry_effects'],
                bootstrap_first_original_prepare_already_counted=True)


def project_recipe_window(recipe, costs, phases, *, include_first_wave=False, width=None):
    """Project the frozen credit rules, optionally at a counterfactual team width.

    A width override changes only this analytic projection. The recipe still
    describes the measured team's exact runtime placement.
    """
    ids = {item['id'] for item in recipe['task_graph']['tasks']}
    if not {task for task, _ in _CUTS.values()} <= ids:
        raise ValueError('ordered phase model does not cover the recipe task graph')
    control = recipe['control']
    relations = {(edge['from']['task'], edge['from']['index'], edge['to']['task'], edge['to']['index'],
                  edge['kind'], edge['when']) for edge in control['cross_instance_dependencies']}
    required = {
        ('prepare-original','0','allocate-team','activation','ORIGINAL_ACTIVATION','FIRST_ELIGIBLE_CALL'),
        ('submit','j','dequeue-oldest','0','INITIAL_FILL','0 <= j < W'),
        ('retire-matched','i','prepare-private','i+W','SLOT_CREDIT','i+W < count'),
        ('submit','i+W','publish-original','i','REFILL_BEFORE_ORIGINAL_RETURN','i+W < count'),
        ('publish-original','i','prepare-original','i+1','ORIGINAL_WORLD_BACKEDGE','i+1 < count'),
        ('publish-original','count-1','close-team','activation','DRAIN_THEN_CLOSE','NORMAL_COMPLETION'),
    }
    if not required <= relations:
        raise ValueError('ordered phase model requires source publication and refill credits')
    count = control['iteration']['count']
    cpu, dispatch = (recipe['resources'][key] for key in ('selected_cpu','dispatch'))
    workers, window = len(cpu['worker_cpu_ids']), dispatch['window']
    if control['window'] != window or not 0 < window <= workers <= count:
        raise ValueError('ordered capacity cannot support the selected case')
    if width is not None:
        if type(width) is not int or not 0 < width <= count or workers != window:
            raise ValueError('width projection requires one ordered slot per worker')
        workers = window = width

    owner = costs['bootstrap_ns']
    pending = deque()
    issue_offset = phases['submit_offer_offset_ns']
    queue, request, native, send = (phases[name] for name in ('queue_ns','request_exchange_child_setup_ns',
                                                             'native_solve_ns','result_exchange_unpack_ns'))
    exchange_service = queue + request + native + send
    if not 0 <= issue_offset <= costs['submit_ns']:
        raise ValueError('asynchronous offer stamp must fall inside the owner submission')
    ready_max = wait = peak_window = 0
    first_wave = []

    def issue(item):
        nonlocal owner, ready_max, peak_window
        owner += costs['prepare_ns']
        submitted = owner + issue_offset
        ready = submitted + exchange_service
        pending.append((item, ready))
        peak_window = max(peak_window, len(pending))
        ready_max = max(ready_max, ready)
        owner += costs['submit_ns']
        if include_first_wave and item < window:
            first_wave.append(dict(iteration=item, owner_submit_ns=submitted,
                courier_start_ns=submitted+queue, worker_start_ns=submitted+queue+request,
                worker_end_ns=submitted+queue+request+native, result_exchange_complete_ns=ready))

    for index in range(window):
        issue(index)
    for index in range(count):
        ordinal, reply = pending.popleft()
        if ordinal != index:
            raise AssertionError('phase model violated in-order retirement')
        owner += costs['first_replay_before_wait_ns' if index == 0 else 'replay_before_wait_ns']
        wait += max(0, reply-owner)
        owner = max(owner, reply)
        if include_first_wave and index < window:
            first_wave[index]['matched_retirement_ns'] = owner
        if index+window < count:
            issue(index+window)
        if include_first_wave and index < window:
            first_wave[index]['publication_proxy_start_ns'] = owner
        owner += phases['publish_digest_proxy_ns']
        if include_first_wave and index < window:
            first_wave[index]['publication_proxy_end_ns'] = owner

    owner_ns = (count*(costs['prepare_ns'] + costs['submit_ns'] + phases['publish_digest_proxy_ns'])
                + costs['first_replay_before_wait_ns'] + (count-1)*costs['replay_before_wait_ns'])
    worker_ns = count*exchange_service
    floor = costs['bootstrap_ns'] + costs['close_ns'] + max(owner_ns, -(-worker_ns//workers))
    detail = dict(first_wave=dict(clock='MODELED_RELATIVE_SCOPE_NS', iterations=first_wave)) if include_first_wave else {}
    return dict(calls=count, workers=workers, window=window, maximum_inflight=peak_window,
                serialized_submit_then_publish_owner_service_ns=owner_ns,
                elapsed_worker_service_sum_ns=worker_ns, elapsed_service_lower_model_ns=floor,
                cpu_quota_work_floor_ns=None, worker_last_completion_ns=ready_max, ordered_wait_ns=wait,
                elapsed_projection_ns=owner+costs['close_ns'],
                control='RECIPE_RETIRE_MATCHED_REFILL_THEN_PUBLISH',
                future_ready_boundary_unmeasured=True,
                memory_contention_and_smt_interference_predicted=False, full_entry_cost_ns=None, **detail)


__all__ = [
    'PROFILE',
    'make_sparse_parallel_recipe',
    'verify_sparse_parallel_recipe',
    'FAMILY',
    'SERIAL_LAUNCH',
    'CHOICE',
    'DETACHED',
    'HARDWARE',
    'bind_calibration',
    'new_phase_samples',
    'append_task_phases',
    'append_bootstrap_phases',
    'check_group_order',
    'check_group_resources',
    'phase_cost_ledger',
    'project_recipe_window',
]
