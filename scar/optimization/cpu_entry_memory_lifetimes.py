"""Replay tracked payload lifetimes on a fixed CPU task schedule."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from scar.analysis.native_call_materials import read_static_cpp_source
from scar.analysis.target_graph import TargetHardwareGraph
from scar.ir.execution_recipe import read_artifact
from scar.ir.program.target_hardware import ConnectionKind, CpuExecutor, MemoryDomain, MemoryKind
from scar.ir.record_codec import encode
from .cpu_entry_schedule_constraints import EXACT as EXACT_LANES, verify_cpu_entry_schedule_constraints
from .deployment_cost import _pointer


SCHEMA = 'scar.cpu-entry-memory-lifetimes'
EXACT = 'EXACT_TRACKED_PAYLOAD_LIFETIMES'
_MAIN = 'src/main.cpp'
_PROFILE = 'src/profile_0.cpp'
_TAPE = 'src/tape_0.inc'


def _require(condition, detail):
    if not condition:
        raise ValueError('CPU entry memory lifetimes: ' + detail)


def _one(rows, detail):
    rows = tuple(rows)
    _require(len(rows) == 1, detail)
    return rows[0]


def _fact(reference, locator):
    return dict(artifact=reference, locator=locator)


class _CppSource:
    def __init__(self, reference):
        self.reference = reference
        self.raw = read_artifact(reference, json_document=False)
        self.model = read_static_cpp_source(reference['uri'])
        stamp = self.model.source
        _require((stamp.path, stamp.sha256, stamp.size_bytes)
                 == (reference['uri'], reference['sha256'], reference['size_bytes']), 'C++ source bytes differ')
        self.index = self.model.index()

    def _body(self, function):
        body = _one((row.body for row in self.model.functions if row.symbol == function),
                    'expected a unique C++ function for ' + function)
        owner = self.index.occurrences[body]
        _require(owner.kind == 'compound_statement' and not owner.has_parse_error, 'C++ lifetime scope has a parse error')
        return owner

    def point(self, function, kind, spelling, *, prefix=False):
        owner = self._body(function)
        def matches(point):
            raw = self.raw[point.byte_start:point.byte_end]
            return raw.startswith(spelling) if prefix else raw == spelling
        node = _one((row for row in self.model.occurrences if row.kind == kind
                     and owner.byte_start <= row.byte_start < row.byte_end <= owner.byte_end and matches(row)),
                    'missing typed C++ lifetime site in ' + function)
        _require(not node.has_parse_error, 'C++ lifetime site has a parse error')
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
    _require(result['byte_end'] <= execute['byte_start'] < publish['byte_start'] < close['byte_start']
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
    _require(plan['contract']['profile']['runtime'] == constraints['evidence']['runtime']
             and invocation['executable'] == launch['executable'] and digest == launch['environment_sha256']
             and launch['cpu_ids'] == invocation['cpu_ids'] == runtime['cpu_ids']
             and launch['argv'] == [launch['executable']['uri'], *plan['argv']]
             and capacity['scope_sha256'] == constraints['scope_sha256'], 'memory lifetimes require the exact native launch')
    _require(environment.get('OMP_STACKSIZE') == '8M' and 'GOMP_STACKSIZE' not in environment
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
    _require(thread['native'] == numerical_ref and thread['effects']['allocation'] == 'PER_TASK_PRIVATE_ALIGNED_ARENA'
             and type(arena) is int and arena > 0 and arena % 64 == 0 and arena == periodic['arena_bytes'],
             'private arena size lacks an exact loop/storage witness')
    for name in (_PROFILE, _TAPE):
        admitted = numerical['files'][name]
        emitted = runtime['files'][name]
        _require(thread['body'][name] == admitted and all(admitted[key] == emitted[key] for key in ('sha256', 'size_bytes'))
                 and read_artifact(emitted, json_document=False) == read_artifact(admitted, json_document=False),
                 'compiled task member differs from the audited private numerical loop')
    tape_ref = runtime['files'][_TAPE]
    literal = f'constexpr std::size_t arena_bytes = {arena};'.encode('ascii')
    _require(read_artifact(tape_ref, json_document=False).splitlines().count(literal) == 1,
             'generated arena declaration differs from the retained byte accounting')

    count, width = constraints['task_count'], constraints['worker_count']
    records, status, requested_stacks = RESULT_BYTES * count, 4 * count, (width - 1) * STACK_BYTES
    tracked_upper = arena * width + records + status
    declared = runtime['decision']['memory']
    cap = capacity['candidate_address_space_limit_bytes']
    _require(declared['private_arena_bytes_per_numerical_worker'] == declared['explicit_worker_payload_bytes'] == arena
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
    _require(asdict(hardware._snapshot) == target_ref, 'NUMA topology must belong to the original CPU target')
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
    report = dict(schema=SCHEMA, schema_version=1, schedule_constraints=schedule_constraints_ref,
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
    _require(proof['main_source'] == runtime['files'][_MAIN], 'entry source differs from the exact lane plan')
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
    return dict(report, status=EXACT, reason='FIXED_SOURCE_OWNERSHIP_AND_LANE_LIVENESS', evidence=evidence,
        lifetimes=objects, phase_live_counts=live, phase_payload_upper_bytes=payload, virtual_address_space=virtual,
        lane_memory_endpoints=_numa(proof['target'], constraints['lanes']))


def verify_cpu_entry_memory_lifetimes(reference):
    saved = read_artifact(reference)
    _require(reference['schema'] == SCHEMA and type(saved['schema_version']) is int and saved['schema_version'] == 1,
             'unsupported CPU entry memory lifetime receipt')
    replay = derive_cpu_entry_memory_lifetimes(saved['schedule_constraints'])
    _require(replay == saved, 'retained memory lifetimes differ at cold replay')
    return replay
