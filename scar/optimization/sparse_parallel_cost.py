"""CPU and memory accounting for a bounded ordered sparse solve pipeline."""

from collections import deque
from fractions import Fraction
import os
from pathlib import Path
import re

from scar.analysis.platform_inventory import _Probe, _constraints
from scar.analysis.target_inventory import _Sources, _cpu_list
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.sparse_solve_cost import _duration, _summary

HARDWARE = 'scar.sparse-parallel-hardware'
SCHEMA = 'scar.sparse-parallel-cost'
CALIBRATION = SCHEMA + '-calibration'
OBSERVATION = SCHEMA + '-observation'
THREAD_ENV = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def collect_hardware(cpu_ids):
    """Capture the selected host lanes and the actual querying task's limits."""
    sources = _Sources()
    probe = _Probe(sources)
    context = _constraints(probe)
    cpu_root = Path('/sys/devices/system/cpu')
    allowed = set(context['affinity_cpu_ids'] or ())
    allowed_nodes = set(context['allowed_memory_nodes'] or ())
    online = set(_cpu_list(probe.file(cpu_root/'online')[0]) or ())
    processors = []
    for cpu in cpu_ids:
        directory = cpu_root/f'cpu{cpu}'
        row, refs = probe.fields(directory/'topology',
                                 ('physical_package_id', 'die_id', 'core_id', 'thread_siblings_list'))
        nodes = sorted(int(p.name[4:]) for p in directory.glob('node[0-9]*'))
        processors.append(dict(cpu_id=cpu, package_id=row['physical_package_id'],
                               die_id=row['die_id'], core_id=row['core_id'],
                               siblings=_cpu_list(row['thread_siblings_list']),
                               numa_nodes=nodes, source_ids=refs,
                               permitted=cpu in allowed and cpu in online))

    mount = next((source for source in sources.rows if source.get('path') == '/proc/self/mountinfo'), {})
    mount_lines = mount.get('text') or ''
    # A delegated cgroup namespace can hide ancestor restrictions.
    complete = any(line.split()[3] == '/' and ' - cgroup2 ' in line
                   for line in mount_lines.splitlines())
    limits = context['cgroup_limits']
    quotas = []
    memory_headroom = []
    for index, row in enumerate(limits):
        current, current_ref = probe.file(Path(row['path'])/'memory.current')
        row.update(memory_current_bytes=int(current) if current and current.isdigit() else None,
                   memory_current_source=current_ref)
        parsed = row['parsed']
        quota = parsed['cpu_max']
        if quota['mode'] == 'QUOTA' and quota['period_us'] and quota['quota_us']:
            quotas.append(Fraction(quota['quota_us'], quota['period_us']))
        # These controls are defined on non-root cgroups.
        is_root = complete and index == len(limits)-1
        if not is_root:
            complete &= quota['mode'] != 'UNKNOWN' and parsed['memory_max']['mode'] != 'UNKNOWN'
        hard_bytes = parsed['memory_max']['bytes']
        if hard_bytes is not None and row['memory_current_bytes'] is not None:
            memory_headroom.append(max(0, hard_bytes-row['memory_current_bytes']))
    quota = min(quotas, default=None)
    node_free = {}
    for node in sorted({node for row in processors for node in row['numa_nodes']}):
        text, ref = probe.file(Path('/sys/devices/system/node')/f'node{node}'/'meminfo')
        match = re.search(r'MemFree:\s+(\d+)\s+kB', text or '')
        node_free[str(node)] = dict(bytes=None if match is None else int(match[1])*1024,
                                    source_id=ref, allowed=node in allowed_nodes)
    return dict(schema=HARDWARE, schema_version=1, processors=processors, limits=limits,
                launching_affinity=sorted(allowed), allowed_memory_nodes=sorted(allowed_nodes),
                cgroup_mode=context['cgroup_mode'], hierarchy_observation_complete=complete and bool(limits),
                cpu_quota_equivalents=None if quota is None else [quota.numerator, quota.denominator],
                finite_memory_headroom_snapshot_bytes=min(memory_headroom, default=None),
                node_free_snapshot=node_free, memory_reservation=False,
                memory_placement_enforced=False, sibling_contention_observed=False,
                thread_environment={name: os.environ.get(name) for name in THREAD_ENV},
                sources=sources.rows)


def choose_trial(hardware, *, owner_cpu, call_count, max_ordered_window, budget,
                 owner_service_ns, worker_service_ns):
    """Choose one study team from capacity and an analytic saturation point."""
    demands = (call_count, max_ordered_window, owner_service_ns, worker_service_ns)
    if any(type(value) is not int or value <= 0 for value in demands):
        raise ValueError('positive scope, window and service demands are required')
    processors = {row['cpu_id']: row for row in hardware['processors']}
    owner = processors[owner_cpu]
    if not owner['permitted'] or len(owner['numa_nodes']) != 1 or owner['core_id'] is None:
        raise ValueError('owner CPU needs observed affinity, core and NUMA placement')
    lane = lambda row: (row['package_id'], row['die_id'], row['core_id'])
    seen = {lane(owner)}
    nearby = []
    for cpu, row in sorted(processors.items()):
        core = lane(row)
        if (row['permitted'] and row['core_id'] is not None and core not in seen
                and row['numa_nodes'] == owner['numa_nodes']):
            nearby.append(cpu)
            seen.add(core)
    saturation = (worker_service_ns + owner_service_ns-1)//owner_service_ns
    physical_limit = min(len(nearby), call_count, max_ordered_window)
    worker_vas = budget.get('worker_total_limit_bytes')
    if worker_vas is None:
        baseline = budget.get('worker_startup_vas_bytes')
        extra = budget.get('worker_extra_limit_bytes')
        if type(baseline) is int and type(extra) is int and baseline >= 0 and extra > 0:
            worker_vas = baseline + extra
    total = budget.get('total_cap_bytes')
    fixed = budget.get('fixed_owned_bytes')
    slot = budget.get('owner_slot_bound_bytes')
    complete_memory = all(type(value) is int and value >= 0 for value in (total, fixed, worker_vas, slot))
    memory_limit = None
    per_worker = None
    if complete_memory:
        per_worker = worker_vas + slot
        if per_worker == 0:
            raise ValueError('positive per-worker bound required')
        memory_limit = max(0, (total-fixed)//per_worker)
    worker_limit = physical_limit if memory_limit is None else min(physical_limit, memory_limit)
    workers = min(saturation, worker_limit)
    return dict(owner_cpu_id=owner_cpu, worker_cpu_ids=nearby[:workers], worker_count=workers,
                window=workers, cpu_numa_node=owner['numa_nodes'][0], physical_worker_limit=physical_limit,
                owner_saturation_estimate_workers=saturation, memory_worker_limit=memory_limit,
                memory_bytes_per_inflight_worker=per_worker,
                modeled_owned_memory_bytes=None if not complete_memory else
                    fixed + workers*per_worker,
                memory_bound_inputs_complete=complete_memory,
                uses_smt_siblings_as_extra_cores=False,
                resource_snapshot_is_reservation=False, automatic_admission=False)


def project_serial_order(call_count, worker_count, window, costs, *, cpu_quota=None):
    """Project a serial submit/publish owner with isolated one-slot workers.

    Each slot remains occupied until its in-order publish. Calls within a slot
    are independent once submitted. Costs describe elapsed service, with
    actual CPU-work demand supplied separately if observed.
    """
    if not 0 < window <= worker_count <= call_count:
        raise ValueError('one or fewer in-flight calls per worker required')
    required = ('bootstrap_ns', 'prepare_ns', 'submit_ns', 'worker_ns', 'replay_before_wait_ns',
                'publish_ns', 'close_ns')
    if any(type(costs[key]) is not int or costs[key] < 0 for key in required):
        raise ValueError('nonnegative matched stage service durations required')
    host = costs['bootstrap_ns']
    pending = deque()
    free_workers = deque(range(worker_count))
    wait = 0
    worker_span = 0
    replay = costs['replay_before_wait_ns']
    owner_service = costs['prepare_ns'] + costs['submit_ns'] + replay + costs['publish_ns']

    def consume(first=False):
        nonlocal host, wait
        lane, done = pending.popleft()
        host += costs.get('first_replay_before_wait_ns', replay) if first else replay
        wait += max(0, done-host)
        host = max(host, done) + costs['publish_ns']
        free_workers.append(lane)

    retired = 0
    for _ in range(call_count):
        if len(pending) == window:
            consume(retired == 0)
            retired += 1
        host += costs['prepare_ns'] + costs['submit_ns']
        worker_span = host + costs['worker_ns']
        pending.append((free_workers.popleft(), worker_span))
    while pending:
        consume(retired == 0)
        retired += 1

    cpu_floor = None
    if (cpu_quota is not None and costs.get('owner_work_ns') is not None
            and costs.get('worker_work_ns') is not None):
        work = costs['owner_work_ns'] + costs['worker_work_ns']
        bound = Fraction(call_count*work, 1)/Fraction(*cpu_quota)
        cpu_floor = -(-bound.numerator//bound.denominator)
    owner_total = call_count*owner_service + costs.get('first_replay_before_wait_ns', replay) - replay
    first_span = owner_service + costs['worker_ns'] + costs.get('first_replay_before_wait_ns', replay) - replay
    floor = max(owner_total, (call_count*costs['worker_ns']+worker_count-1)//worker_count,
                first_span, cpu_floor or 0)
    fixed = costs['bootstrap_ns'] + costs['close_ns']
    return dict(calls=call_count, workers=worker_count, window=window,
                serialized_submit_then_publish_owner_service_ns=owner_total,
                elapsed_worker_service_sum_ns=call_count*costs['worker_ns'],
                elapsed_service_lower_model_ns=fixed+floor, cpu_quota_work_floor_ns=cpu_floor,
                worker_last_completion_ns=worker_span, ordered_wait_ns=wait,
                elapsed_projection_ns=max(host+costs['close_ns'], fixed+floor),
                memory_contention_and_smt_interference_predicted=False,
                full_entry_cost_ns=None)


def _overlap(tasks):
    events = [(task['started_ns'], 1) for task in tasks] + [(task['ended_ns'], -1) for task in tasks]
    live = peak = 0
    for _, delta in sorted(events):
        live += delta
        peak = max(live, peak)
    return peak


def _measure_ordered(raw, choice, address_space, terms, *, phase_samples=None):
    runtime = raw['runtime']
    tasks = sorted(runtime['tasks'], key=lambda task: task['issue'])
    published = runtime['publications']
    samples = raw['samples']
    expected = list(range(len(samples)))
    if ([task['issue'] for task in tasks] != expected
            or [item['issue'] for item in published] != expected
            or runtime['native_calls'] != len(samples) or runtime['fallback_calls']
            or not raw['released']['closed'] or raw['released']['pending_snapshot_bytes'] != 0):
        raise ValueError('ordered workload did not publish every measured task without fallback')
    workers = {item['pid']: item for item in runtime['workers']}
    if (sorted(worker['cpu'] for worker in workers.values()) != choice['worker_cpu_ids']
            or any(worker['address_space_limit_bytes'] != address_space for worker in workers.values())):
        raise ValueError('worker CPU and address-space contract differ from the study')
    if phase_samples is not None:
        from scar.optimization import sparse_parallel_recipe_cost as typed

        typed.check_group_order(samples, choice['window'])
        typed.append_bootstrap_phases(phase_samples, workers.values())
    for sample, task, item in zip(samples, tasks, published):
        start, prepared, issued = sample['submit_clock_ns']
        entered, actual_ready, result_ready, succeeded = sample['consume_clock_ns']
        wait = item['consume_wait_ns']
        if (not item['native'] or task['status'] != 0 or sample['fallback']
                or task['cpu'] != workers[task['pid']]['cpu']
                or not start < prepared <= task['submitted_ns'] <= issued < entered
                or not task['submitted_ns'] <= task['exchange_started_ns'] < task['started_ns']
                       < task['ended_ns'] <= task['exchange_complete_ns'] <= result_ready
                or wait < 0 or wait > result_ready-actual_ready):
            raise ValueError('sparse submission, execution or publication cuts differ')
        terms['prepare_ns'].append(_duration((start, prepared)))
        terms['submit_ns'].append(_duration((prepared, issued)))
        terms['worker_ns'].append(_duration((task['submitted_ns'], task['exchange_complete_ns'])))
        terms['native_thread_cpu_ns'].append(task['native_thread_cpu_ns'])
        terms['native_wall_ns'].append(task['ended_ns']-task['started_ns'])
        terms['wait_ns'].append(wait)
        publication = (_duration((result_ready, succeeded)) if phase_samples is None
                       else typed.append_task_phases(phase_samples, task, sample))
        terms['publish_ns'].append(publication)
        replay = result_ready-entered-wait
        terms['first_replay_before_wait_ns' if task['issue'] == 0 else 'replay_before_wait_ns'].append(replay)
    terms['bootstrap_ns'].append(_duration(raw['bootstrap_clock_ns']))
    terms['close_ns'].append(_duration(raw['close_clock_ns']))
    loader = (raw['loader_cut_ns']-raw['bootstrap_clock_ns'][0] if phase_samples is None
              else _duration(raw['loader_clock_ns']))
    terms['loader_ns'].append(loader)
    terms['team_spawn_and_analyze_ns'].append(runtime['startup_ns'])
    terms['owner_exact_validation_ns'].append(runtime['owner_validation_ns'])
    return dict(group=raw['group'], distinct_workers=len(workers), maximum_native_wall_overlap=_overlap(tasks),
                peak_worker_rss_bytes=max(task['worker_max_rss_bytes'] for task in tasks),
                worker_virtual_address_caps_bytes=sum(worker['address_space_limit_bytes'] for worker in workers.values()),
                worker_symbolic_payload_bytes=sum(worker['symbolic_bytes'] for worker in workers.values()),
                owner_structure_bytes=runtime['structure_bytes'],
                owner_peak_pending_input_bytes=runtime['peak_snapshot_bytes'],
                owner_pending_result_bound_bytes=runtime['pending_result_bound_bytes'])


def project_calibration(design_path, observations):
    design_ref = artifact_reference(design_path, CALIBRATION)
    design = read_artifact(design_ref)
    source = read_artifact(design['source_scope'])
    parallel = read_artifact(design['parallel_scope'])
    native = read_artifact(design['native'])
    for ref in (design['harness'], design['fixture'], design['entry'], native['library'],
                native['wheel'], native['blas'], *design['runtime'].values()):
        read_artifact(ref, json_document=False)
    choice_doc = read_artifact(design['choice'])
    hardware = read_artifact(choice_doc['hardware'])
    choice = choice_doc['choice']
    if (design['schema_version'] != 1 or design['entry'] != source['entry']['source']
            or parallel['source_scope'] != design['source_scope']
            or parallel['shadow_inputs']['caller_code']['key'] != design['recipe']['body_code_key']
            or design['context']['argv'] != source['entry']['argv']
            or design['context']['cpu_ids'] != sorted([choice['owner_cpu_id'], *choice['worker_cpu_ids']])
            or any(design['context']['environment'][key] != value
                   for key, value in source['entry']['thread_environment'].items())):
        raise ValueError('parallel calibration does not match admitted original input or CPU team')
    recipe = phase_samples = None
    if 'execution_recipe' in design:
        from scar.optimization import sparse_parallel_recipe_cost as typed

        recipe = typed.bind_calibration(design, source, parallel, choice, native)
        phase_samples = typed.new_phase_samples()
    fields = ('bootstrap_ns', 'prepare_ns', 'submit_ns', 'worker_ns',
              'replay_before_wait_ns', 'first_replay_before_wait_ns', 'publish_ns', 'close_ns',
              'loader_ns', 'owner_exact_validation_ns', 'team_spawn_and_analyze_ns',
              'native_wall_ns', 'native_thread_cpu_ns', 'wait_ns')
    terms = {field: [] for field in fields}
    serial = dict(fixed_lifecycle_ns=[], warm_ns=[])
    facts = []
    refs = [artifact_reference(path, OBSERVATION) for path in observations]
    if len(refs) != 2*len(design['groups']) or len({ref['uri'] for ref in refs}) != len(refs):
        raise ValueError('unique paired calibration processes are required')
    it = iter(refs)
    count = source['entry']['argv_repetitions']
    for group_id, group in enumerate(design['groups']):
        digests = []
        for mode in group['order']:
            raw = read_artifact(next(it))
            if (raw['schema_version'] != 1 or raw['design'] != design_ref or raw['mode'] != mode
                    or raw['group'] != group_id or raw['context'] != design['context']
                    or raw['source_iterations'] != group['iterations']):
                raise ValueError('observed sparse calibration worker differs from frozen task list')
            digests.append((raw['output_digests'], raw['published_digest']))
            if mode == 'ordered':
                observed = _measure_ordered(raw, choice, design['worker_address_space_bytes'], terms,
                                            phase_samples=phase_samples)
                if recipe is not None:
                    typed.check_group_resources(recipe, observed)
                facts.append(observed)
            else:
                rows = raw['samples']
                times = [_duration((row['consume_clock_ns'][0], row['consume_clock_ns'][-1]))
                         for row in rows]
                serial['fixed_lifecycle_ns'].append(times[0]+_duration(raw['bootstrap_clock_ns'])
                                                   +_duration(raw['close_clock_ns']))
                serial['warm_ns'].extend(times[1:])
        if len(digests[0][0]) != len(group['iterations']) or digests[0] != digests[1]:
            raise ValueError('ordered sparse observations did not preserve source solve bytes')
    ordered = {key: _summary(series) for key, series in terms.items()}
    serial = {key: _summary(series) for key, series in serial.items()}
    keys = ('estimate_ns', 'lower_ns', 'upper_ns')
    projections = {}
    ledger = None if phase_samples is None else typed.phase_cost_ledger(phase_samples)
    for key in keys:
        profile = {field: value[key] for field, value in ordered.items()}
        if ledger is None:
            projections[key] = project_serial_order(count, choice['worker_count'], choice['window'], profile,
                                                   cpu_quota=hardware['cpu_quota_equivalents'])
        else:
            cuts = {name: measure[key] for name, measure in ledger['measured'].items()}
            projections[key] = typed.project_recipe_window(recipe, profile, cuts,
                                                          include_first_wave=key == 'estimate_ns')
    binding = {} if recipe is None else dict(execution_recipe=design['execution_recipe'], phase_cost_ledger=ledger)
    return dict(schema=SCHEMA, schema_version=1, design=design_ref, observations=refs,
                choice=design['choice'], source_scope=design['source_scope'],
                parallel_scope=design['parallel_scope'],
                native=design['native'], context=design['context'],
                owner_cpu_id=choice['owner_cpu_id'], worker_cpu_ids=choice['worker_cpu_ids'],
                window=choice['window'], ordered=ordered, serial=serial, resources=facts,
                projections=projections, all_outputs_bitwise=True,
                calibration_is_section_only=True, full_entry_cost_ns=None,
                automatic_admission=False, **binding)


def check(path):
    result = read_artifact(artifact_reference(Path(path), SCHEMA))
    rebuilt = project_calibration(result['design']['uri'], [item['uri'] for item in result['observations']])
    if result != rebuilt:
        raise ValueError('ordered sparse calibration no longer cold replays')
    return result
