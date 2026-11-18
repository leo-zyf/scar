"""Bind ordered process service measurements to the typed sparse recipe."""

from collections import deque

from scar.ir.execution_recipe import SCHEMA, read_artifact, validate_recipe
from scar.optimization.sparse_solve_cost import _duration, _summary


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
