"""Project a bounded CPU entry lifecycle from original effect and resource proofs."""

from collections import defaultdict, deque
from enum import Enum

from scar.analysis.cpu_launch_resources import verify_cpu_launch_resources
from scar.ir.execution_recipe import read_artifact
from scar.optimization.cpu_entry_capacity_contract import ADMISSIBLE, verify_cpu_entry_capacity_contract
from scar.optimization.cpu_entry_decision_ledger import verify_cpu_entry_decision_ledger
from scar.optimization.whole_entry_choice import LIFECYCLE_PHASES

SCHEMA = 'scar.f4-cpu-entry-lifecycle-graph'


class _Relation(str, Enum):
    CONTAINS = 'CONTAINS'
    HAPPENS_BEFORE = 'HAPPENS_BEFORE'
    DATA = 'DATA'
    BYTE_ORDER = 'BYTE_ORDER'
    OBSERVABLE_ASSOCIATION = 'OBSERVABLE_ASSOCIATION'


_EFFECTS = {
    'STDIN_BYTES': ('file_io', 'STREAM_INPUT_OBSERVATION'),
    'STDIN_RANGE': ('file_io', 'STREAM_INPUT_RANGE'),
    'STDERR': ('file_io', 'STDERR_OBSERVATION'),
    'STDOUT_BYTES': ('publication', 'STREAM_OUTPUT_OBSERVATION'),
    'STDOUT_ACKNOWLEDGEMENT': ('publication', 'ACKNOWLEDGEMENT_OBSERVATION'),
    'FILE_PUBLICATION': ('publication', 'FILE_PUBLICATION_OBSERVATION'),
    'INVOCATION_CHAIN': ('execution', 'ORIGINAL_INVOCATION'),
}
_STREAM_EDGES = {
    'ADJACENT_STDIN_RANGES': _Relation.BYTE_ORDER,
    'TYPED_INPUT_TO_OBSERVED_OUTPUT': _Relation.OBSERVABLE_ASSOCIATION,
    'TYPED_COMMAND_AND_CAPTURED_SEARCH_ORDINAL': _Relation.OBSERVABLE_ASSOCIATION,
    'CAPTURED_SEARCH_AND_OUTPUT_ORDINAL': _Relation.OBSERVABLE_ASSOCIATION,
    'CAPTURE_ACK_BEFORE_NEXT_SEND': _Relation.HAPPENS_BEFORE,
    'PAIRED_HARNESS_ACK_BEFORE_NEXT_SEND': _Relation.HAPPENS_BEFORE,
}
_STREAM_ROWS = {'commands': '/stdin', 'acknowledgements': '/stdout', 'search_invocations': ''}


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU entry lifecycle: ' + detail)


def _fact(artifact, locator):
    return dict(artifact=artifact, locator=locator)


class _Lane:
    def __init__(self, option, role):
        self.option = option
        self.role = role
        self.operations = {}
        self.edges = []

    def operation(self, identity, kind, phase, status, *evidence, **fields):
        _need(identity not in self.operations and (phase is None or phase in LIFECYCLE_PHASES),
              'invalid or duplicate lifecycle operation')
        self.operations[identity] = dict(id=identity, kind=kind, phase=phase, status=status,
                                         evidence=list(evidence), **fields)
        return identity

    def edge(self, source, target, kind, basis, *evidence, when=None):
        _need(source in self.operations and target in self.operations and source != target and evidence,
              'lifecycle dependence endpoints or evidence differ')
        self.edges.append(dict(source=source, target=target, kind=kind.value, basis=basis,
                               evidence=list(evidence), when=when))

    def finish(self):
        incoming = dict.fromkeys(self.operations, 0)
        outgoing = defaultdict(list)
        for edge in self.edges:
            if edge['kind'] == _Relation.HAPPENS_BEFORE.value:
                outgoing[edge['source']].append(edge['target'])
                incoming[edge['target']] += 1
        pending = deque(key for key, count in incoming.items() if not count)
        visited = 0
        while pending:
            source = pending.popleft()
            visited += 1
            for target in outgoing[source]:
                incoming[target] -= 1
                if not incoming[target]:
                    pending.append(target)
        _need(visited == len(self.operations), 'required lifecycle relations contain a cycle')
        phases = {phase: [] for phase in LIFECYCLE_PHASES}
        for row in self.operations.values():
            if row['phase'] is not None:
                phases[row['phase']].append(row['id'])
        return dict(option_id=self.option, role=self.role, operations=list(self.operations.values()),
            edges=self.edges, phase_coverage={phase: dict(operations=ops,
                status='PARTIAL_SCOPED_EVIDENCE' if ops else 'UNREPRESENTED')
                for phase, ops in phases.items()}, phases_without_operations=[phase for phase, ops in phases.items()
                    if not ops], relative_overlap_outside_required_edges='UNKNOWN', lifecycle_costs_closed=False)


def _stream_edges(lane, graph, prefix, nodes, graph_ref):
    observed = graph['observable_graph']
    base = prefix + '/observable_graph'
    ids = {}
    for table, suffix in _STREAM_ROWS.items():
        for index, row in enumerate(observed.get(table, ())):
            original = f'{base}/{table}/{index}{suffix}'
            _need(original in nodes and row['id'] not in ids, 'exact scoped stream event is absent or ambiguous')
            ids[row['id']] = nodes[original]
    for index, row in enumerate(observed.get('edges', ())):
        kind = _STREAM_EDGES.get(row['kind'])
        if kind is not None:
            _need(row['source'] in ids and row['target'] in ids, 'stream dependency endpoint is outside named evidence')
            # Only the launcher waiting for a reply before its next send
            # carries a retained ordering claim. Ordinal matches stay weaker.
            lane.edge(ids[row['source']], ids[row['target']], kind, row['kind'],
                      _fact(graph_ref, f'{base}/edges/{index}'))


def _original_lane(resources, resources_ref, effects, option):
    effects_ref = resources['entry_effect_frontier']
    lane = _Lane(option, 'ORIGINAL')
    envelope = lane.operation('boundary', 'PROCESS_ENTRY_LIFECYCLE', None, 'F4_ENTRY_BOUNDARY',
        _fact(resources['program'], '#/entry'))
    named = effects['observed']['named_effects']
    exit_facts = [_fact(effects_ref, f'#/observed/named_effects/{index}') for index, row in enumerate(named)
                  if row['kind'] == 'PROCESS_EXIT']
    start = lane.operation('start', 'PROCESS_LAUNCH', 'startup',
        'RETAINED_SCOPED_LAUNCH' if resources['launch'] is not None else 'UNOBSERVED_BOUNDARY',
        _fact(resources_ref, '#/launch'), target_thread_endpoints=resources['launch_owner_cpu_endpoints'])
    work = lane.operation('work', 'SOURCE_EXECUTION_REGION', 'execution', 'STATIC_MAY_REGION',
        _fact(effects['entry_obligations'], '#/entry_body'))
    end = lane.operation('exit', 'PROCESS_EXIT', 'teardown',
        'RETAINED_SCOPED_EXIT' if exit_facts else 'UNOBSERVED_BOUNDARY',
        *(exit_facts or [_fact(resources['program'], '#/entry')]))
    for operation in (start, work, end):
        lane.edge(envelope, operation, _Relation.CONTAINS, 'NORMAL_ENTRY_SCOPE', _fact(effects_ref, '#'))
    lane.edge(start, end, _Relation.HAPPENS_BEFORE, 'SAME_PROCESS_LAUNCH_EXIT_BOUNDARY',
              _fact(resources['program'], '#/entry'), when='NORMAL_ENTRY_COMPLETION')
    if effects['static']['imports']:
        fact = _fact(effects_ref, '#/static/imports')
        imports = lane.operation('imports', 'SOURCE_IMPORT_REGION', 'imports', 'STATIC_MAY_REGION', fact)
        lane.edge(work, imports, _Relation.CONTAINS, 'MAY_INCLUDE_MODULE_RESOLUTION_AND_INITIALIZATION', fact)
    nodes = {}
    for index, row in enumerate(named):
        shape = _EFFECTS.get(row['kind'])
        if shape is None:
            continue
        phase, kind = shape
        fact = _fact(effects_ref, f'#/observed/named_effects/{index}')
        identity = lane.operation(f'observed:{index}', kind, phase, 'RETAINED_NAMED_OBSERVATION',
                                  fact, original_operation_attribution=row['source_attribution'])
        nodes[row['original']] = identity
        lane.edge(envelope, identity, _Relation.CONTAINS, 'BOUNDED_ENTRY_OBSERVABLE', fact)
    graph_ref = resources['cpu_graph']
    if graph_ref is not None:
        graph = read_artifact(graph_ref)
        if graph['schema'] == 'scar.f4-cpu-stream-capture':
            _stream_edges(lane, graph, '#', nodes, graph_ref)
        elif graph.get('stream_capture') is not None:
            _stream_edges(lane, graph['stream_capture'], '#/stream_capture', nodes, graph_ref)
    return lane.finish()


def _candidate_lane(capacity, capacity_ref):
    lane = _Lane(capacity['option_id'], 'CONDITIONAL_NATIVE_CANDIDATE')
    ref = capacity['evidence']['runtime']
    runtime = read_artifact(ref)
    decision = runtime['decision']
    schedule_ref = capacity['evidence']['schedule']
    schedule = read_artifact(schedule_ref)
    thread_ref = capacity['evidence']['thread_contract']
    threads = read_artifact(thread_ref)
    _need(capacity['accounting']['kind'] == 'SINGLE_PROCESS_INCLUDING_THREADS'
          and schedule['thread_domain']['publication'] == 'ORDERED_AFTER_ALL_LANES_JOIN'
          and threads['effects']['publication'] == 'OWNER_ONLY_IN_ORDINAL_ORDER_AFTER_JOIN'
          and schedule['dispatch']['terminal_publication'] == 'AFTER_ALL_SUCCEEDED_IN_ORDINAL_ORDER'
          and decision['scheduled_pod_result_lifetime'] == 'RETAIN_IN_OWNER_UNTIL_ALL_TASKS_HAVE_SUCCEEDED'
          and runtime['no_worker_visible_output'] is True and runtime['schedule'] == schedule_ref,
          'candidate lacks the scoped successful team publication contract')
    publication_fact = _fact(schedule_ref, '#/dispatch/terminal_publication')
    join_fact = _fact(thread_ref, '#/effects/publication')
    team_fact = _fact(ref, '#/decision')
    guard_fact = _fact(capacity_ref, '#/guard')
    entry_fact = _fact(capacity_ref, '#/native_launch')
    envelope = lane.operation('boundary', 'PROCESS_ENTRY_LIFECYCLE', None, 'CONDITIONAL_PROVIDER_PLAN', entry_fact)
    start = lane.operation('start', 'PROCESS_LAUNCH', 'startup', 'CONDITIONAL_PROVIDER_PLAN', entry_fact)
    guard = lane.operation('guard', 'PRE_EXEC_CAPACITY_GUARD', 'guards', 'CONDITIONAL_PROVIDER_PLAN', guard_fact)
    team = lane.operation('team', 'WORKER_TEAM_SCOPE', 'worker_lifecycle', 'CONDITIONAL_PROVIDER_PLAN',
                          _fact(schedule_ref, '#/thread_domain'), scope_only=True)
    work = lane.operation('work', 'SCHEDULED_NUMERICAL_TEAM', 'execution', 'CONDITIONAL_PROVIDER_PLAN', team_fact,
        owner_cpu=decision['owner_cpu_id'], worker_cpu_ids=decision['worker_cpu_ids'],
        task_count=schedule['scope']['task_count'], configured_threads=capacity['accounting']['configured_threads'],
        thread_placement_observed=False)
    join = lane.operation('join', 'WORKER_TEAM_JOIN', 'synchronization', 'CONDITIONAL_PROVIDER_PLAN', join_fact)
    publication = lane.operation('publication', 'FINAL_OWNER_PUBLICATION', 'publication',
                                 'CONDITIONAL_PROVIDER_PLAN', publication_fact, join_fact)
    exit_ = lane.operation('exit', 'PROCESS_EXIT', 'teardown', 'CONDITIONAL_PROVIDER_PLAN', entry_fact)
    for operation in (start, guard, team, publication, exit_):
        lane.edge(envelope, operation, _Relation.CONTAINS, 'CANDIDATE_ENTRY_SCOPE', entry_fact)
    for operation in (work, join):
        lane.edge(team, operation, _Relation.CONTAINS, 'SCHEDULED_TEAM_SCOPE', team_fact)
    condition = 'SELECTED_GUARDED_CANDIDATE_NORMAL_COMPLETION'
    for left, right, basis, fact in (
        (start, guard, 'GUARD_IS_PART_OF_CANDIDATE_LAUNCH', guard_fact),
        (guard, work, 'GUARD_BEFORE_CANDIDATE_ELF_EXEC', guard_fact),
        (work, join, 'TEAM_COMPLETES_BEFORE_OWNER_JOIN_RETURNS', join_fact),
        (join, publication, 'TERMINAL_PUBLICATION_AFTER_SUCCESSFUL_JOIN', publication_fact),
        (publication, exit_, 'SUCCESSFUL_PUBLICATION_INSIDE_ENTRY', publication_fact),
    ):
        lane.edge(left, right, _Relation.HAPPENS_BEFORE, basis, fact, when=condition)
    lane.edge(work, publication, _Relation.DATA, 'ORDERED_RESULTS_RETAINED_IN_OWNER',
              _fact(ref, '#/decision/scheduled_pod_result_lifetime'), publication_fact, when=condition)
    return lane.finish()


def derive_cpu_entry_lifecycle_graph(launch_resources, *, decision_ledger=None, capacity_contract=None):
    """Cold-build source and optional candidate operation lanes without measuring time."""
    resources = verify_cpu_launch_resources(launch_resources)
    effects_ref = resources['entry_effect_frontier']
    effects = read_artifact(effects_ref)
    capacity = None if capacity_contract is None else verify_cpu_entry_capacity_contract(capacity_contract)
    ledger_ref = decision_ledger
    if capacity is not None:
        _need(ledger_ref in (None, capacity['ledger']), 'capacity contract and decision ledger differ')
        ledger_ref = capacity['ledger']
        ledger = read_artifact(ledger_ref)  # already verified by the capacity replay
    else:
        ledger = None if ledger_ref is None else verify_cpu_entry_decision_ledger(ledger_ref)
    baseline, scope = 'original', None
    if ledger is not None:
        _need(ledger['effect_frontier'] == effects_ref and ledger['target']['graph'] == resources['target']
              and ledger['provider_graph'] == resources['cpu_graph'], 'decision ledger belongs to another entry scope')
        baseline, scope = ledger['baseline_id'], ledger['scope_sha256']
    lanes = [_original_lane(resources, launch_resources, effects, baseline)]
    frontier = [
        dict(kind='ORIGINAL_EFFECT_REQUIREMENTS_OPEN', evidence=[_fact(effects_ref, '#/static')]),
        dict(kind='RESOURCE_ENTRY_AND_DESCENDANT_LIMITS_OPEN',
             evidence=[_fact(launch_resources, '#/coverage_frontier')]),
        dict(kind='LIFECYCLE_TIME_AND_UNOBSERVED_OVERLAP_OPEN', evidence=[_fact(launch_resources, '#')]),
    ]
    if capacity is not None and capacity['status'] == ADMISSIBLE:
        lanes.append(_candidate_lane(capacity, capacity_contract))
    else:
        frontier.append(dict(kind='EXACT_CANDIDATE_SCHEDULE_NOT_PROJECTED', evidence=[] if capacity is None
                             else [_fact(capacity_contract, '#/reason')]))
    return dict(schema=SCHEMA, schema_version=1, launch_resources=launch_resources, decision_ledger=decision_ledger,
        capacity_contract=capacity_contract, program=resources['program'], target=resources['target'],
        effect_frontier=effects_ref, scope_sha256=scope, lanes=lanes, coverage_frontier=frontier,
        time_bounds_proven=False, whole_entry_equivalence_proven=False, transformation_authorized=False)


def verify_cpu_entry_lifecycle_graph(reference):
    report = read_artifact(reference)
    _need(reference['schema'] == SCHEMA and report == derive_cpu_entry_lifecycle_graph(
        report['launch_resources'], decision_ledger=report['decision_ledger'],
        capacity_contract=report['capacity_contract']), 'saved lifecycle differs at replay')
    return report
