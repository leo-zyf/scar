"""Qualify and select parallel windows in a normalized standalone ATen recipe."""

from scar.analysis.target_graph import TargetHardwareGraph
from scar.ir.program.target_hardware import CpuCore, HardwareThread
from scar.ir.record_codec import encode


CONTRACT_SCHEMA = 'scar.aten-native-concurrency-contract'
SCHEDULE_SCHEMA = 'scar.aten-native-cpu-pair'
WINDOWS_SCHEMA = 'scar.aten-native-cpu-windows'
SCOPE = 'STANDALONE_CPU_ATEN'
_ATTESTATIONS = ('synchronous_cpu_completion', 'read_only_operands',
                 'no_shared_mutable_or_rng_state', 'no_externally_observable_side_effects')
_NO_WINDOWS = 'no independent adjacent pair has a declared standalone concurrency contract'


class AtenCpuWindowFrontier(ValueError):
    """The current CPU window planner cannot admit this placement."""

    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


def _declared_operators(contract):
    fields = {'schema', 'schema_version', 'scope', 'operator_schemas', *_ATTESTATIONS}
    if (type(contract) is not dict or contract.keys() != fields
            or contract['schema'] != CONTRACT_SCHEMA or type(contract['schema_version']) is not int
            or contract['schema_version'] != 1 or contract['scope'] != SCOPE
            or any(contract[name] is not True for name in _ATTESTATIONS)):
        raise ValueError('explicit standalone ATen CPU concurrency contract required')
    operators = contract['operator_schemas']
    if (type(operators) is not list or not operators
            or any(type(schema) is not str or not schema.startswith('aten::') for schema in operators)
            or len(operators) != len(set(operators))):
        raise ValueError('concurrency contract requires unique exact ATen declarations')
    return frozenset(operators)


def _live_inputs(operation):
    return {arg['value'] for arg in operation['arguments'] if 'value' in arg}


def recorded_cpu_domain(graph):
    """Recover the CPUs admitted by both masks of a single target context."""
    constraints = graph.constraints()
    if len(constraints) == 1:
        constraint, = constraints
        affinity, allowed = constraint.affinity_cpu_ids, constraint.allowed_cpu_ids
        if affinity is not None and allowed is not None:
            domain = tuple(sorted(set(affinity).intersection(allowed)))
            if domain:
                return constraint, domain
    raise AtenCpuWindowFrontier('TARGET_CPU_DOMAIN_UNAVAILABLE',
        'the target requires one execution context with intersecting affinity and allowed CPU masks')


def _exact_cpu_constraint(graph):
    constraints = graph.constraints()
    if len(constraints) == 1:
        constraint, = constraints
        mask = constraint.affinity_cpu_ids
        if mask is not None and len(mask) == 2 and mask == constraint.allowed_cpu_ids:
            return constraint
    raise AtenCpuWindowFrontier('TARGET_CPU_DOMAIN_NOT_EXACT_PAIR',
        'the recorded target requires matching two-CPU affinity and allowed masks')


def _placement(graph, constraint, cpu_ids, *, domain=None):
    if (type(cpu_ids) not in (tuple, list) or len(cpu_ids) != 2
            or any(type(cpu) is not int or not 0 <= cpu < 1024 for cpu in cpu_ids)
            or cpu_ids[0] == cpu_ids[1]):
        raise ValueError('two distinct ordered Linux CPU IDs required')
    mask = tuple(sorted(cpu_ids))
    exact = constraint.affinity_cpu_ids == constraint.allowed_cpu_ids == mask
    if domain is None and not exact:
        raise ValueError('proposed CPU mask differs from the recorded target availability domain')
    if domain is not None and not set(mask).issubset(domain):
        raise AtenCpuWindowFrontier('CPU_PAIR_OUTSIDE_TARGET_CPU_DOMAIN',
            'both CPU lanes must be in the recorded affinity and allowed CPU intersection')
    resources = graph.resources()
    threads = {row.cpu_id: row for row in resources if type(row) is HardwareThread and row.online is True}
    cores = {cpu: row for row in resources if type(row) is CpuCore for cpu in row.cpu_ids}
    if (any(cpu not in threads or cpu not in cores for cpu in cpu_ids)
            or cores[cpu_ids[0]].id == cores[cpu_ids[1]].id):
        raise AtenCpuWindowFrontier('CPU_PAIR_DISTINCT_CORES_UNAVAILABLE',
            'parallel lanes require online threads on distinct physical cores')
    lanes = [dict(cpu=cpu, thread=encode(threads[cpu].id), core=encode(cores[cpu].id),
                  thread_fact=encode(threads[cpu].evidence), core_fact=encode(cores[cpu].evidence))
             for cpu in cpu_ids]
    placement = dict(target=encode(graph._snapshot), availability_mask=list(mask), lanes=lanes,
                     constraint_fact=encode(constraint.evidence), cpus=list(cpu_ids),
                     effect_basis='EXPLICIT_STANDALONE_CALLER_CONTRACT',
                     source_replacement_authorized=False, performance_measured=False)
    if not exact:
        placement['target_cpu_domain'] = dict(
            affinity_cpu_ids=list(constraint.affinity_cpu_ids), allowed_cpu_ids=list(constraint.allowed_cpu_ids),
            eligible_cpu_ids=list(domain))
    return placement


def _independent_windows(recipe, contract):
    if (type(recipe) is not dict or recipe.get('schema') != 'scar.aten-native-recipe'
            or type(recipe.get('schema_version')) is not int or recipe['schema_version'] != 1):
        raise ValueError('normalized ATen graph recipe required')
    admitted = _declared_operators(contract)
    operations = recipe['operations']
    for index, (left, right) in enumerate(zip(operations, operations[1:])):
        if left['schema'] not in admitted or right['schema'] not in admitted:
            continue
        left_reads, right_reads = _live_inputs(left), _live_inputs(right)
        if left['id'] in right_reads or right['id'] in left_reads:
            continue
        yield dict(pair=[left['id'], right['id']], pair_indices=[index, index + 1],
                   statically_shared_reads=sorted(left_reads & right_reads))


def _required_windows(recipe, contract):
    windows = list(_independent_windows(recipe, contract))
    if not windows:
        raise AtenCpuWindowFrontier('NO_PARALLEL_WINDOWS', _NO_WINDOWS)
    return windows


def propose_cpu_pair(recipe, *, contract, target, cpu_ids):
    """Find the first independent adjacent pair with an explicit effect contract.

    The caller normalizes the recipe against the pinned provider before planning.
    Both results become ready before the next operation starts. Availability is a
    placement constraint, while cost selection and runtime affinity are separate.
    """
    window = next(_independent_windows(recipe, contract), None)
    if window is None:
        raise AtenCpuWindowFrontier('NO_PARALLEL_WINDOWS', _NO_WINDOWS)
    graph = TargetHardwareGraph(target)
    placement = _placement(graph, _exact_cpu_constraint(graph), cpu_ids)
    return dict(schema=SCHEDULE_SCHEMA, schema_version=1, scope=SCOPE, **window, **placement)


def propose_cpu_windows(recipe, *, contract, target, cpu_ids):
    """Retain all legal windows on an explicit pair in the recorded target."""
    windows = _required_windows(recipe, contract)
    graph = TargetHardwareGraph(target)
    constraint, domain = recorded_cpu_domain(graph)
    placement = _placement(graph, constraint, cpu_ids, domain=domain)
    return dict(schema=WINDOWS_SCHEMA, schema_version=1, scope=SCOPE,
                windows=windows, **placement)


def resolve_aten_cpu_windows(recipe, *, contract, target, allowed_cpu_ids):
    """Admit CPU windows using the exact recorded lane domain and launch affinity."""
    windows = _required_windows(recipe, contract)
    graph = TargetHardwareGraph(target)
    constraint = _exact_cpu_constraint(graph)
    cpu_ids = tuple(sorted(constraint.affinity_cpu_ids))
    placement = _placement(graph, constraint, cpu_ids)
    if not set(cpu_ids).issubset(allowed_cpu_ids):
        raise AtenCpuWindowFrontier('CPU_PAIR_OUTSIDE_LAUNCH_AFFINITY',
            'the launch environment cannot use both recorded CPUs')
    return dict(schema=WINDOWS_SCHEMA, schema_version=1, scope=SCOPE,
                windows=windows, **placement)


def select_cpu_windows(schedule, services, dispatch, *, worker_services=None):
    """Return non-overlapping first indices that minimize modeled completion time.

    Each position either stays serial, or starts a legal pair and skips its
    second operation. Pair duration uses the slower lane plus one handoff.
    A suffix dynamic program maximizes the total savings, breaking ties by
    fewer handoffs and then the earliest first index.
    """
    if (type(schedule) is not dict or schedule.get('schema') != WINDOWS_SCHEMA
            or type(schedule.get('schema_version')) is not int or schedule['schema_version'] != 1):
        raise ValueError('native CPU window proposal required')
    if (type(services) not in (tuple, list) or len(services) < 2
            or any(type(value) is not int or value <= 0 for value in services)
            or type(dispatch) is not int or dispatch < 0):
        raise ValueError('positive integer service times and nonnegative dispatch time required')
    if worker_services is None:
        worker_services = services
    elif (type(worker_services) not in (tuple, list) or len(worker_services) != len(services)
            or any(value is not None and (type(value) is not int or value <= 0) for value in worker_services)):
        raise ValueError('worker services must align with caller operations and give positive integer durations')

    eligible = set()
    for window in schedule['windows']:
        first, second = window['pair_indices']
        if (type(first) is not int or type(second) is not int or not 0 <= first < len(services) - 1
                or second != first + 1 or first in eligible):
            raise ValueError('invalid or duplicate adjacent window in CPU proposal')
        if worker_services[second] is None:
            raise ValueError('worker service is missing for an eligible CPU window')
        eligible.add(first)

    # Scores compare savings first and negative handoff count second.
    scores = [(0, 0)] * (len(services) + 1)
    take = [False] * len(services)
    for index in range(len(services) - 2, -1, -1):
        scores[index] = scores[index + 1]
        if index in eligible:
            paired_duration = max(services[index], worker_services[index + 1]) + dispatch
            saving = services[index] + services[index + 1] - paired_duration
            tail_saving, tail_handoffs = scores[index + 2]
            paired = (saving + tail_saving, tail_handoffs - 1)
            if saving > 0 and paired >= scores[index]:
                scores[index], take[index] = paired, True

    selected = []
    index = 0
    while index < len(services):
        if take[index]:
            selected.append(index)
            index += 2
        else:
            index += 1
    return tuple(selected)
