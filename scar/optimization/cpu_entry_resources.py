"""Cpu entry resources."""

from hashlib import sha256
import json
from pathlib import Path
from scar.ir.execution_recipe import read_artifact, artifact_reference
from scar.optimization.cpu_entry_decision_ledger import verify_cpu_entry_decision_ledger
from scar.optimization.scope_accounting import _pointer
from dataclasses import asdict
from scar.analysis.target_graph import TargetHardwareGraph
from scar.ir.program.target_hardware import (
    CpuCore,
    HardwareThread,
    ConnectionKind,
    CpuExecutor,
    MemoryDomain,
    MemoryKind,
)
from scar.analysis.native_call_materials import read_static_cpp_source
from scar.ir.record_codec import encode


# Derive a candidate-only CPU address-space guard from a replayed planner ledger.
CAPACITY_SCHEMA = 'scar.cpu-entry-capacity-contract'
ADMISSIBLE = 'CANDIDATE_GUARD_ADMISSIBLE'
_FAMILY = 'CATALOG_CPU_ENTRY'
_ARRAY = 'scar.cpu-array-task-source-provider'
_RUNTIME = 'scar.native-seeded-array-task-entry'
_TEAM_PLAN = 'scar.independent-array-task-team-schedule'
_THREAD_PROOF = 'scar.native-array-thread-safety'
_SINGLE = 'SINGLE_PROCESS_VAS_CAP_INCLUDING_PRIVATE_ARRAYS_AND_OPENMP_CHILD_STACKS'
_AGGREGATE = 'AGGREGATE_PROCESS_VAS_CAPS_PLUS_EXPLICIT_PAYLOAD_SUBACCOUNT'


def _require_capacity(condition, message):
    if not condition:
        raise ValueError('CPU entry capacity contract: ' + message)


def _array_runtime(plan, witness):
    authority = read_artifact(plan['authority'])
    compiled = plan['contract']['profile']
    native_ref = compiled['runtime']
    retained = read_artifact(compiled['retention'])
    native = read_artifact(native_ref)
    decision = native['decision']
    memory = decision['memory']
    profiles = [row for row in authority['profiles'] if row['argv'] == plan['argv']]
    _require_capacity(len(profiles) == 1 and authority['entry'] == compiled['entry'] == plan['entry']
             and all(compiled[key] == value for key, value in profiles[0].items())
             and compiled['provider'] == plan['authority'] and native_ref['schema'] == _RUNTIME
             and retained['action'] == 'RETAIN_VERIFIED_LOCAL_BINARY' and retained['retained_release'] == native_ref
             and retained['selected'] == decision and native['schema_version'] == 1
             and native['deployment_authorized'] is False and native['python_runtime_required'] is False
             and native['cpu_ids'] == decision['launch_cpu_ids'] == plan['context']['cpu_ids']
             and witness['evidence'] == dict(artifact=native_ref, locator='#/decision/memory/required_headroom_bytes')
             and witness['bytes'] == memory['required_headroom_bytes'] and memory['physical_reservation'] is False,
             'provider runtime does not reproduce the exact candidate capacity witness')
    return compiled, native, decision


def _accounting(mode, memory, workers, aggregate_bytes):
    owner = memory['coordinator_address_space_limit_bytes']
    worker = memory['worker_address_space_limit_bytes']
    _require_capacity(type(owner) is int and 0 < owner < 2**63 and type(worker) is int and 0 < worker < 2**63
             and type(workers) is int and workers >= 1, 'positive exact per-role limits and worker count required')
    roles = [dict(role='COORDINATOR', count=1, per_process_limit_bytes=owner)]
    if memory['bound_scope'] == _SINGLE:
        _require_capacity(mode == 'ORDERED_OPENMP' and workers > 1 and aggregate_bytes == owner,
                 'OpenMP threads must share exactly one address space')
        return dict(kind='SINGLE_PROCESS_INCLUDING_THREADS', roles=roles, configured_threads=workers,
                    process_count=1, aggregate_limit_bytes=owner)
    _require_capacity(memory['bound_scope'] == _AGGREGATE and worker <= owner, 'unknown or non-inheritable address-space caps')
    if mode == 'ORDERED_LOCAL_PROCESSES':
        _require_capacity(workers > 1, 'a process team requires separate worker processes')
        roles.append(dict(role='WORKER', count=workers, per_process_limit_bytes=worker))
    else:
        _require_capacity(mode == 'IN_PROCESS_SERIAL' and workers == 1, 'unsupported aggregate process layout')
    _require_capacity(aggregate_bytes == sum(row['count'] * row['per_process_limit_bytes'] for row in roles),
             'aggregate cap must sum the address-space limits of every process role')
    return dict(kind='AGGREGATE_PER_PROCESS_CAP_ACCOUNTING', roles=roles, configured_threads=None,
                process_count=sum(row['count'] for row in roles), aggregate_limit_bytes=aggregate_bytes)


def derive_cpu_entry_capacity_contract(ledger_ref, option_id):
    """Return the exact guard admission or an unresolved reason for one candidate."""
    ledger = verify_cpu_entry_decision_ledger(ledger_ref)
    matches = [row for row in ledger['options'] if row['option_id'] == option_id and row['original'] is False]
    _require_capacity(len(matches) == 1, 'one exact native candidate option ID required')
    option, = matches
    report = dict(schema=CAPACITY_SCHEMA, schema_version=1, ledger=ledger_ref, option_id=option_id,
        scope_sha256=ledger['scope_sha256'], legality=option['legality'], status='UNRESOLVED',
        reason='CANDIDATE_HAS_NO_SCOPED_VIRTUAL_ADDRESS_SPACE_CAP', native_launch=None,
        candidate_address_space_limit_bytes=None, provider_process_count=None, guard_scope=None,
        guard=None, accounting=None, evidence=None, whole_process_dram_peak_bytes=None,
        physical_dram_reservation_proven=False, execution_default='NO_OP', deployment_authorized=False)
    witness = option['memory']['virtual_address_space_headroom']
    if option['legality'] == 'INELIGIBLE':
        return dict(report, reason='CANDIDATE_LEGALITY_INELIGIBLE')
    if witness['status'] != 'VIRTUAL_ADDRESS_SPACE_CAPS':
        return report
    fact_ref, = option['legality_evidence']
    legality = _pointer(read_artifact(fact_ref['artifact']), fact_ref['locator'])
    generated = legality.get('generated_option', {})
    if generated.get('implementation_family') != _FAMILY:
        return dict(report, reason='NO_SCOPED_NATIVE_PROVIDER')
    plan = generated['proof']['emission_plan']
    if plan['provider_family'] != _ARRAY:
        return dict(report, reason='NATIVE_CAPACITY_INTERFACE_UNSUPPORTED')
    compiled, native, decision = _array_runtime(plan, witness)
    runtime_ref = compiled['runtime']
    memory = decision['memory']
    cap_bytes = witness['bytes']
    accounting = _accounting(decision['mode'], memory, decision['worker_count'], cap_bytes)
    report.update(accounting=accounting, provider_process_count=accounting['process_count'],
                  evidence=dict(provider=plan['authority'], runtime=runtime_ref, memory=witness['evidence']))
    if accounting['kind'] != 'SINGLE_PROCESS_INCLUDING_THREADS':
        return dict(report, reason='PROVIDER_LAYOUT_REQUIRES_A_SEPARATE_PROCESS_ROLE_GUARD')

    context = plan['context']
    if not all(context['scope_matches'].values()):
        return dict(report, reason='PROVIDER_LAUNCH_CONTEXT_DIFFERS')
    target = read_artifact(plan['target'])
    inventory_ref = dict(target['inventory'], schema='scar.target-inventory')
    if read_artifact(inventory_ref)['platform']['system'] != 'linux':
        return dict(report, reason='LINUX_RLIMIT_AS_TARGET_REQUIRED')
    invocation = compiled['invocation']
    binary = native['files'][native['binary']]
    environment = invocation['environment']
    _require_capacity(invocation['executable'] == binary and invocation['cwd'] == context['cwd']
             and invocation['cpu_ids'] == context['cpu_ids'] and invocation['original_executable'] == context['interpreter']
             and invocation['original_environment'] == context['original_environment']
             and environment == dict(context['original_environment'], **native['launch_environment'])
             and invocation['output_index'] == plan['contract']['publication']['output_index'] == len(plan['argv']) - 1
             and Path(binary['uri']).resolve(strict=True) == Path(runtime_ref['uri']).parent / native['binary']
             and read_artifact(binary, json_document=False).startswith(b'\x7fELF'),
             'address-space guard must bind the exact original scope, native ELF and public argv')
    schedule_ref = native['schedule']
    schedule = read_artifact(schedule_ref)
    thread_ref = native['thread_contract']
    threads = read_artifact(thread_ref)
    _require_capacity(schedule_ref['schema'] == _TEAM_PLAN and thread_ref['schema'] == _THREAD_PROOF
             and schedule['thread_contract'] == thread_ref and decision in schedule['alternatives']
             and native['scope'] == schedule['scope'] and native['family'] == schedule['family'] == threads['family']
             and native['native'] == schedule['native'] == threads['native']
             and 0 < cap_bytes <= option['memory']['request_budget_bytes'],
             'single-process cap must bind the admitted OpenMP schedule and request budget')
    report['evidence'].update(schedule=schedule_ref, thread_contract=thread_ref, target_inventory=inventory_ref)
    environment_sha256 = sha256(json.dumps(sorted(environment.items()), separators=(',', ':')).encode()).hexdigest()
    return dict(report, status=ADMISSIBLE, reason='EXACT_SINGLE_PROCESS_PROVIDER_AND_DECLARED_ADDRESS_SPACE_CAP',
        native_launch=dict(executable=binary, argv=[binary['uri'], *plan['argv']],
                           cwd=invocation['cwd'], cpu_ids=invocation['cpu_ids'],
                           environment_sha256=environment_sha256, output_index=invocation['output_index']),
        candidate_address_space_limit_bytes=cap_bytes, guard_scope='PER_PROCESS_VIRTUAL_ADDRESS_SPACE',
        guard=dict(interface='LINUX_RLIMIT_AS', soft_limit_bytes=cap_bytes, hard_limit_bytes=cap_bytes,
                   applies_before='CANDIDATE_ELF_EXEC', candidate_only=True, may_raise_inherited_limits=False,
                   pre_exec_enforcement_observed=False, guarded_candidate_execution_verified=False))


def verify_cpu_entry_capacity_contract(reference):
    saved = read_artifact(reference)
    _require_capacity(reference['schema'] == CAPACITY_SCHEMA and type(saved['schema_version']) is int and saved['schema_version'] == 1,
             'unsupported capacity contract')
    replay = derive_cpu_entry_capacity_contract(saved['ledger'], saved['option_id'])
    _require_capacity(replay == saved, 'retained capacity contract differs at cold replay')
    return replay


# Project the fixed resource and order constraints of a qualified CPU entry.
SCHEDULE_SCHEMA = 'scar.cpu-entry-schedule-constraints'
SCHEDULE_EXACT = 'EXACT_ORDINAL_LANE_CONSTRAINTS'
_NORMAL = 'SELECTED_GUARDED_CANDIDATE_NORMAL_COMPLETION'
_CONDITION = 'QUALIFIED_OPENMP_PLACEMENT_AND_NORMAL_COMPLETION'
_HEADER = 'include/scar_generated/native_ordered_openmp.hpp'


def _require_schedule(condition, message):
    if not condition:
        raise ValueError('CPU entry schedule constraints: ' + message)


def _fact(artifact, locator):
    return dict(artifact=artifact, locator=locator)


def _operation(lane, kind):
    matches = [row for row in lane['operations'] if row['kind'] == kind]
    _require_schedule(len(matches) == 1, 'expected one scoped lifecycle operation: ' + kind)
    return matches[0]


def _provider(capacity, work):
    from scar.analysis.cpu_native_phase_sites import _retained_runtime
    from scar.codegen import native_seeded_array_task_entry as emitter
    from .independent_array_task_team_schedule import MODE, SCHEDULE_SCHEMA as SCHEDULE

    runtime_ref = capacity['evidence']['runtime']
    schedule_ref = capacity['evidence']['schedule']
    runtime, schedule = map(read_artifact, (runtime_ref, schedule_ref))
    decision = runtime['decision']
    cpus, width = decision['worker_cpu_ids'], decision['worker_count']
    count = schedule['scope']['task_count']
    threads = schedule['thread_domain']
    _require_schedule(runtime_ref['schema'] == emitter.SCHEDULE_SCHEMA and schedule_ref['schema'] == SCHEDULE
             and runtime['schedule'] == schedule_ref and runtime['scope'] == schedule['scope']
             and runtime['thread_contract'] == schedule['thread_contract'] == capacity['evidence']['thread_contract']
             and decision in schedule['alternatives'] and not decision['capacity_reasons'] and decision['mode'] == MODE
             and threads['assignment'] == 'ORDINAL_MOD_WORKER_COUNT' and threads['owner_participates'] is True
             and threads['publication'] == 'ORDERED_AFTER_ALL_LANES_JOIN'
             and threads['shared_result_slots_disjoint'] is True and runtime['no_worker_visible_output'] is True
             and schedule['dispatch']['terminal_publication'] == 'AFTER_ALL_SUCCEEDED_IN_ORDINAL_ORDER',
             'the native entry lacks a fixed ordinal-to-lane dispatch and owner join')
    _require_schedule(type(count) is type(width) is int and 1 < width <= count < 2**63
             and len(cpus) == len(set(cpus)) == width and all(type(cpu) is int and cpu >= 0 for cpu in cpus)
             and cpus[0] == decision['owner_cpu_id'] == work['owner_cpu']
             and cpus == work['worker_cpu_ids'] and count == work['task_count']
             and width == work['configured_threads'] == capacity['accounting']['configured_threads']
             and runtime['cpu_ids'] == decision['launch_cpu_ids'] == capacity['native_launch']['cpu_ids']
             and set(cpus) <= set(runtime['cpu_ids']), 'compiled lane counts or CPU placement differ from the entry graph')
    # Re-render the admitted main so a provider decision cannot be paired with
    # an unrelated executable shape. The dispatcher is a frozen source member.
    source, _, _, _, _, _ = _retained_runtime(capacity)
    header = runtime['files'][_HEADER]
    _require_schedule(Path(header['uri']) == Path(runtime_ref['uri']).parent / _HEADER
             and read_artifact(header, json_document=False) == emitter._OPENMP.read_bytes(),
             'retained OpenMP dispatcher differs from the admitted lane protocol')
    return cpus, count, dict(runtime=runtime_ref, schedule=schedule_ref, main_source=source, dispatcher=header,
        dispatcher_template=artifact_reference(emitter._OPENMP, 'scar.native-cpp-source-template'),
        provider_decision=_fact(runtime_ref, '#/decision'), assignment=_fact(schedule_ref, '#/thread_domain/assignment'))


def _lanes(target_ref, placement, cpus, count):
    target = TargetHardwareGraph(Path(target_ref['uri']).parent)
    _require_schedule(asdict(target._snapshot) == target_ref, 'CPU resources must belong to the exact retained target')
    inventory = target.resources()
    threads = {row.cpu_id: row.id.wire for row in inventory if type(row) is HardwareThread and row.online is True}
    cores = {cpu: row.id.wire for row in inventory if type(row) is CpuCore for cpu in row.cpu_ids}
    endpoints = placement['endpoints']
    _require_schedule(all(cpu in threads and cpu in cores and endpoints.get(str(cpu)) == [threads[cpu], cores[cpu]] for cpu in cpus)
             and len({cores[cpu] for cpu in cpus}) == len(cpus), 'planned lanes need distinct original physical cores')
    width = len(cpus)
    quotient, remainder = divmod(count, width)
    return [dict(lane_index=index, cpu_id=cpu, target_thread=threads[cpu], target_core=cores[cpu],
                 first_ordinal=index, stride=width, task_count=quotient + int(index < remainder))
            for index, cpu in enumerate(cpus)]


def _normal_order(graph_ref, lane_index, lane, work):
    guard = _operation(lane, 'PRE_EXEC_CAPACITY_GUARD')
    join = _operation(lane, 'WORKER_TEAM_JOIN')
    publication = _operation(lane, 'FINAL_OWNER_PUBLICATION')
    expected = [(guard['id'], work['id']), (work['id'], join['id']), (join['id'], publication['id'])]
    edges = {}
    for index, row in enumerate(lane['edges']):
        relation = row['source'], row['target']
        if row['kind'] == 'HAPPENS_BEFORE' and row['when'] == _NORMAL and relation in expected:
            _require_schedule(relation not in edges, 'duplicate normal-entry ordering proof')
            edges[relation] = dict(source=row['source'], target=row['target'], when=row['when'],
                                   evidence=_fact(graph_ref, f'#/lanes/{lane_index}/edges/{index}'))
    _require_schedule(set(edges) == set(expected), 'normal guard, work, join and publication order required')
    return [edges[key] for key in expected]


def derive_cpu_entry_schedule_constraints(lifecycle_graph_ref):
    """Build a schedule constraint receipt without unrolling tasks or timing them."""
    from scar.analysis.cpu_entry_lifecycle_graph import verify_cpu_entry_lifecycle_graph

    graph = verify_cpu_entry_lifecycle_graph(lifecycle_graph_ref)
    capacity_ref = graph['capacity_contract']
    capacity = None if capacity_ref is None else read_artifact(capacity_ref)
    option_id = None if capacity is None else capacity['option_id']
    report = dict(schema=SCHEDULE_SCHEMA, schema_version=1, lifecycle_graph=lifecycle_graph_ref,
        scope_sha256=graph['scope_sha256'], option_id=option_id, status='UNRESOLVED',
        reason='NO_ADMITTED_FIXED_OPENMP_CANDIDATE', task_count=None, worker_count=None, lanes=[],
        evidence=None, normal_path_edges=[], logical_order=None, resource_occupancy=None, body_work_lower_bound=None,
        whole_entry_time_bound_ns=None, complete_lifecycle_proven=False, deployment_authorized=False)
    if capacity is None or capacity['status'] != ADMISSIBLE:
        return report
    candidates = [(index, lane) for index, lane in enumerate(graph['lanes'])
                  if lane['option_id'] == option_id and lane['role'] == 'CONDITIONAL_NATIVE_CANDIDATE']
    _require_schedule(len(candidates) == 1 and graph['scope_sha256'] == capacity['scope_sha256']
             and capacity['provider_process_count'] == 1, 'schedule constraints require one exact candidate lane')
    (lane_index, lane), = candidates
    work = _operation(lane, 'SCHEDULED_NUMERICAL_TEAM')
    cpus, count, evidence = _provider(capacity, work)
    ledger_ref = capacity['ledger']
    ledger = read_artifact(ledger_ref)
    option, = (row for row in ledger['options'] if row['option_id'] == option_id)
    placement = option['cpu_placement']
    _require_schedule(placement['cpu_ids'] == capacity['native_launch']['cpu_ids'] and ledger['target']['graph'] == graph['target'],
             'ledger placement differs from the admitted launch and target')
    lanes = _lanes(graph['target'], placement, cpus, count)
    _require_schedule(sum(row['task_count'] for row in lanes) == count, 'closed-form lane partition must cover every ordinal')
    evidence.update(capacity_contract=capacity_ref, decision_ledger=ledger_ref, target=graph['target'])
    return dict(report, status=SCHEDULE_EXACT, reason='FIXED_ORDINAL_DISPATCH_ON_QUALIFIED_CPU_LANES',
        task_count=count, worker_count=len(cpus), lanes=lanes, evidence=evidence,
        normal_path_edges=_normal_order(lifecycle_graph_ref, lane_index, lane, work),
        logical_order=dict(condition=_CONDITION, lane_order='ASCENDING_ORDINAL_ON_EACH_LANE',
                           assignment='ORDINAL_MOD_WORKER_COUNT', cross_lane_order='UNSPECIFIED',
                           join_requires_all_lane_tails=True),
        resource_occupancy=dict(condition=_CONDITION, task_slots_per_lane=1, maximum_active_task_lanes=len(cpus),
                                physical_cores_distinct=True, simultaneous_execution_required=False,
                                uncontended_service_proven=False),
        body_work_lower_bound=dict(kind='MAX_SERIAL_LANE_SERVICE_SUM',
            max_tasks_on_lane=max(row['task_count'] for row in lanes), service_times_ns=None, lower_ns=None))


def verify_cpu_entry_schedule_constraints(reference):
    saved = read_artifact(reference)
    _require_schedule(reference['schema'] == SCHEDULE_SCHEMA and type(saved['schema_version']) is int and saved['schema_version'] == 1,
             'unsupported CPU entry schedule receipt')
    replay = derive_cpu_entry_schedule_constraints(saved['lifecycle_graph'])
    _require_schedule(replay == saved, 'retained CPU schedule constraints differ at cold replay')
    return replay


# Replay tracked payload lifetimes on a fixed CPU task schedule.
EXACT_LANES = SCHEDULE_EXACT
MEMORY_SCHEMA = 'scar.cpu-entry-memory-lifetimes'
MEMORY_EXACT = 'EXACT_TRACKED_PAYLOAD_LIFETIMES'
_MAIN = 'src/main.cpp'
_PROFILE = 'src/profile_0.cpp'
_TAPE = 'src/tape_0.inc'


def _require_memory(condition, detail):
    if not condition:
        raise ValueError('CPU entry memory lifetimes: ' + detail)


def _one(rows, detail):
    rows = tuple(rows)
    _require_memory(len(rows) == 1, detail)
    return rows[0]




class _CppSource:
    def __init__(self, reference):
        self.reference = reference
        self.raw = read_artifact(reference, json_document=False)
        self.model = read_static_cpp_source(reference['uri'])
        stamp = self.model.source
        _require_memory((stamp.path, stamp.sha256, stamp.size_bytes)
                 == (reference['uri'], reference['sha256'], reference['size_bytes']), 'C++ source bytes differ')
        self.index = self.model.index()

    def _body(self, function):
        body = _one((row.body for row in self.model.functions if row.symbol == function),
                    'expected a unique C++ function for ' + function)
        owner = self.index.occurrences[body]
        _require_memory(owner.kind == 'compound_statement' and not owner.has_parse_error, 'C++ lifetime scope has a parse error')
        return owner

    def point(self, function, kind, spelling, *, prefix=False):
        owner = self._body(function)
        def matches(point):
            raw = self.raw[point.byte_start:point.byte_end]
            return raw.startswith(spelling) if prefix else raw == spelling
        node = _one((row for row in self.model.occurrences if row.kind == kind
                     and owner.byte_start <= row.byte_start < row.byte_end <= owner.byte_end and matches(row)),
                    'missing typed C++ lifetime site in ' + function)
        _require_memory(not node.has_parse_error, 'C++ lifetime site has a parse error')
        return dict(artifact=self.reference, source_occurrence=encode(node.source),
                    byte_start=node.byte_start, byte_end=node.byte_end)

    def declaration(self, function, spelling, indentation):
        # The source projection records compound scopes and calls. For local
        # declarations retain an exact statement line within the checked scope.
        owner = self._body(function)
        fragment = b' ' * indentation + spelling
        cursor, matches = owner.byte_start, []
        for line in self.raw[owner.byte_start:owner.byte_end].splitlines(keepends=True):
            if line.removesuffix(b'\n') == fragment:
                matches.append(cursor + indentation)
            cursor += len(line)
        start = _one(matches, 'missing generated declaration in ' + function)
        return dict(artifact=self.reference, scoped_to=encode(owner.source),
                    byte_start=start, byte_end=start + len(spelling))


def _sites(main_ref, profile_ref):
    main, profile = map(_CppSource, (main_ref, profile_ref))
    result = main.declaration('main', b'Result result{};', 8)
    execute = main.point('main', 'call_expression', b'execute(result)')
    publish = main.point('main', 'if_statement', b'if (!path.empty()) {', prefix=True)
    close = main.point('main', 'return_statement', b'return std::cout ? 0 : 1;')
    records = main.point('execute', 'call_expression', b'result.records.resize(result.task_count)')
    team = main.point('execute', 'call_expression', b'execute_threads(result, diagnostic)')
    status = main.declaration('execute_threads', b'std::vector<std::int32_t> failures(result.task_count);', 4)
    dispatch = main.point('execute_threads', 'call_expression', b'scar::openmp_v1::dispatch(', prefix=True)
    release_status = main.point('execute_threads', 'return_statement', b'return 0;')
    compute = main.point('compute', 'call_expression', b'scar_array_entry_profile_0(', prefix=True)
    arena = profile.declaration('scar_array_entry_profile_0',
        b'std::unique_ptr<void, decltype(&std::free)> arena(std::aligned_alloc(64, arena_bytes), &std::free);', 4)
    release_arena = profile.point('scar_array_entry_profile_0', 'return_statement', b'return 0;')
    _require_memory(result['byte_end'] <= execute['byte_start'] < publish['byte_start'] < close['byte_start']
             and records['byte_end'] <= team['byte_start']
             and status['byte_end'] <= dispatch['byte_start'] < dispatch['byte_end'] <= release_status['byte_start']
             and arena['byte_end'] <= release_arena['byte_start'], 'compiled lifetime sites are out of causal order')
    return dict(owner_result_object=result, owner_results_resize=records, owner_team_call=team,
        status_vector=status, dispatch_call=dispatch, status_scope_return=release_status, task_profile_call=compute,
        task_arena_allocation=arena, task_arena_scope_return=release_arena,
        publication_begin=publish, owner_scope_return=close)


def _scoped_launch(constraints, capacity, runtime):
    ledger = read_artifact(capacity['ledger'])
    alternative = _one((row for row in ledger['options'] if row['option_id'] == constraints['option_id']),
                       'missing exact CPU option')
    proof, = alternative['legality_evidence']
    plan = _pointer(read_artifact(proof['artifact']), proof['locator'])['generated_option']['proof']['emission_plan']
    invocation = plan['contract']['profile']['invocation']
    environment = invocation['environment']
    digest = sha256(json.dumps(sorted(environment.items()), separators=(',', ':')).encode()).hexdigest()
    launch = capacity['native_launch']
    _require_memory(plan['contract']['profile']['runtime'] == constraints['evidence']['runtime']
             and invocation['executable'] == launch['executable'] and digest == launch['environment_sha256']
             and launch['cpu_ids'] == invocation['cpu_ids'] == runtime['cpu_ids']
             and launch['argv'] == [launch['executable']['uri'], *plan['argv']]
             and capacity['scope_sha256'] == constraints['scope_sha256'], 'memory lifetimes require the exact native launch')
    _require_memory(environment.get('OMP_STACKSIZE') == '8M' and 'GOMP_STACKSIZE' not in environment
             and 'SCAR_ARRAY_ENTRY_STATES' not in environment and 'SCAR_ARRAY_ENTRY_LANE_REPORT' not in environment,
             'tracked payload model requires the qualified stack setting and disabled diagnostic storage')


def _memory_values(constraints, capacity, runtime):
    from .independent_array_task_schedule import RESULT_BYTES
    from .independent_array_task_team_schedule import STACK_BYTES

    numerical_ref = runtime['native']
    numerical = read_artifact(numerical_ref)
    thread_ref = runtime['thread_contract']
    thread = read_artifact(thread_ref)
    periodic_ref = numerical['periodic_memory']
    periodic = read_artifact(periodic_ref)
    arena = numerical['arena_bytes']
    _require_memory(thread['native'] == numerical_ref and thread['effects']['allocation'] == 'PER_TASK_PRIVATE_ALIGNED_ARENA'
             and type(arena) is int and arena > 0 and arena % 64 == 0 and arena == periodic['arena_bytes'],
             'private arena size lacks an exact loop/storage witness')
    for name in (_PROFILE, _TAPE):
        admitted = numerical['files'][name]
        emitted = runtime['files'][name]
        _require_memory(thread['body'][name] == admitted and all(admitted[key] == emitted[key] for key in ('sha256', 'size_bytes'))
                 and read_artifact(emitted, json_document=False) == read_artifact(admitted, json_document=False),
                 'compiled task member differs from the audited private numerical loop')
    tape_ref = runtime['files'][_TAPE]
    literal = f'constexpr std::size_t arena_bytes = {arena};'.encode('ascii')
    _require_memory(read_artifact(tape_ref, json_document=False).splitlines().count(literal) == 1,
             'generated arena declaration differs from the retained byte accounting')

    count, width = constraints['task_count'], constraints['worker_count']
    records, status, requested_stacks = RESULT_BYTES * count, 4 * count, (width - 1) * STACK_BYTES
    tracked_upper = arena * width + records + status
    declared = runtime['decision']['memory']
    cap = capacity['candidate_address_space_limit_bytes']
    _require_memory(declared['private_arena_bytes_per_numerical_worker'] == declared['explicit_worker_payload_bytes'] == arena
             and declared['retained_result_payload_bytes'] == records and declared['retained_status_bytes'] == status
             and declared['result_bytes'] == RESULT_BYTES and declared['thread_stack_bytes'] == STACK_BYTES
             and declared['explicit_owner_payload_bytes'] == tracked_upper + requested_stacks
             and declared['coordinator_address_space_limit_bytes'] == declared['required_headroom_bytes'] == cap
             and tracked_upper + requested_stacks < cap and declared['physical_reservation'] is False,
             'lifetime-derived sizes must reproduce the admitted virtual subaccount')
    return arena, records, status, STACK_BYTES, dict(audited_thread_contract=thread_ref, numerical_body=numerical_ref,
        periodic_arena=_fact(periodic_ref, '#/arena_bytes'), compiled_tape=tape_ref,
        configured_memory=_fact(constraints['evidence']['runtime'], '#/decision/memory'))


def _numa(target_ref, lanes):
    hardware = TargetHardwareGraph(Path(target_ref['uri']).parent)
    _require_memory(asdict(hardware._snapshot) == target_ref, 'NUMA topology must belong to the original CPU target')
    resources = hardware.resources()
    executors = [node for node in resources if type(node) is CpuExecutor]
    host_memory = {node.id: node for node in resources if type(node) is MemoryDomain and node.kind is MemoryKind.HOST}
    local_links = {(row.source, row.target): row for row in hardware.connections()
                   if row.kind is ConnectionKind.MEMORY_ACCESS and row.locality == 'local' and row.target in host_memory}
    result = []
    for lane in lanes:
        executor = _one((row for row in executors if lane['cpu_id'] in row.cpu_ids), 'CPU lane NUMA executor is ambiguous')
        memory = _one((row for row in host_memory.values() if row.owner == executor.id
                       and row.numa_node == executor.numa_node and (executor.id, row.id) in local_links),
                      'local CPU memory endpoint is unknown')
        link = local_links[executor.id, memory.id]
        result.append(dict(lane_index=lane['lane_index'], cpu_id=lane['cpu_id'], executor=executor.id.wire,
            numa_node=memory.numa_node, local_memory_endpoint=memory.id.wire,
            reported_capacity_bytes=memory.capacity_bytes, actual_residency_bytes=None,
            access_evidence=encode(link.evidence), domain_evidence=encode(memory.evidence)))
    return result


def derive_cpu_entry_memory_lifetimes(schedule_constraints_ref):
    """Retain phase live sets of explicit payloads and separate configured capacity."""
    constraints = verify_cpu_entry_schedule_constraints(schedule_constraints_ref)
    report = dict(schema=MEMORY_SCHEMA, schema_version=1, schedule_constraints=schedule_constraints_ref,
        lifecycle_graph=constraints['lifecycle_graph'], scope_sha256=constraints['scope_sha256'],
        option_id=constraints['option_id'], status='UNRESOLVED', reason='EXACT_FIXED_LANE_SCHEDULE_REQUIRED',
        accounting_scope='TRACKED_EXPLICIT_LOGICAL_PAYLOAD_AND_DECLARED_STACK_REQUEST',
        lifetimes=[], phase_live_counts={}, phase_payload_upper_bytes={}, evidence=None,
        virtual_address_space=None, lane_memory_endpoints=[], physical_dram_residency_proven=False,
        physical_dram_budget_bytes=None, whole_process_payload_upper_bytes=None, whole_entry_time_bound_ns=None,
        deployment_authorized=False)
    if constraints['status'] != EXACT_LANES:
        return report
    proof = constraints['evidence']
    capacity, runtime = map(read_artifact, (proof['capacity_contract'], proof['runtime']))
    _scoped_launch(constraints, capacity, runtime)
    arena, records, status, stack, evidence = _memory_values(constraints, capacity, runtime)
    _require_memory(proof['main_source'] == runtime['files'][_MAIN], 'entry source differs from the exact lane plan')
    sites = _sites(proof['main_source'], runtime['files'][_PROFILE])
    width = constraints['worker_count']
    objects = [
        dict(id='owner_results', bytes_per_instance=records, instances_upper=1, availability='SHARED_DISJOINT_ORDINAL_SLOTS',
             begin='OWNER_RESULT_VECTOR_RESIZE', end='NORMAL_MAIN_SCOPE_EXIT',
             begin_site=sites['owner_results_resize'], end_site=sites['owner_scope_return']),
        dict(id='dispatch_status', bytes_per_instance=status, instances_upper=1, availability='SHARED_DISJOINT_ORDINAL_SLOTS',
             begin='TEAM_STATUS_VECTOR_CONSTRUCTION', end='NORMAL_TEAM_FUNCTION_SCOPE_EXIT',
             begin_site=sites['status_vector'], end_site=sites['status_scope_return']),
        dict(id='task_arena', bytes_per_instance=arena, instances_upper=width, availability='PRIVATE_ONE_AT_A_TIME_PER_LANE',
             begin='EACH_TASK_ARENA_ALLOCATION', end='EACH_NORMAL_TASK_SCOPE_EXIT',
             begin_site=sites['task_arena_allocation'], end_site=sites['task_arena_scope_return']),
    ]
    live = dict(before_result_resize={}, result_records_before_team=dict(owner_results=1),
        before_first_task=dict(owner_results=1, dispatch_status=1),
        running_tasks=dict(owner_results=1, dispatch_status=1, task_arena=width),
        joined_before_status_release=dict(owner_results=1, dispatch_status=1),
        publication=dict(owner_results=1), after_owner_scope_exit={})
    sizes = {row['id']: row['bytes_per_instance'] for row in objects}
    payload = {phase: sum(sizes[name] * count for name, count in counts.items()) for phase, counts in live.items()}
    requested_stacks = (width - 1) * stack
    virtual = dict(rlimit_as_bytes=capacity['candidate_address_space_limit_bytes'],
        configured_stack_bytes_per_additional_lane=stack, configured_additional_lane_count=width - 1,
        configured_team_stack_bytes=requested_stacks, stack_release_event=None,
        declared_address_space_subaccount_bytes=max(payload.values()) + requested_stacks,
        owner_stack_allocator_runtime_and_provider_storage_unaccounted=True)
    evidence.update(capacity_contract=proof['capacity_contract'], source_sites=sites, target=proof['target'])
    return dict(report, status=MEMORY_EXACT, reason='FIXED_SOURCE_OWNERSHIP_AND_LANE_LIVENESS', evidence=evidence,
        lifetimes=objects, phase_live_counts=live, phase_payload_upper_bytes=payload, virtual_address_space=virtual,
        lane_memory_endpoints=_numa(proof['target'], constraints['lanes']))


def verify_cpu_entry_memory_lifetimes(reference):
    saved = read_artifact(reference)
    _require_memory(reference['schema'] == MEMORY_SCHEMA and type(saved['schema_version']) is int and saved['schema_version'] == 1,
             'unsupported CPU entry memory lifetime receipt')
    replay = derive_cpu_entry_memory_lifetimes(saved['schedule_constraints'])
    _require_memory(replay == saved, 'retained memory lifetimes differ at cold replay')
    return replay
