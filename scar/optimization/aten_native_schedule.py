"""Qualify one parallel window in a normalized standalone ATen recipe."""

from scar.analysis.target_graph import TargetHardwareGraph
from scar.ir.program.target_hardware import CpuCore, HardwareThread
from scar.ir.record_codec import encode


CONTRACT_SCHEMA = 'scar.aten-native-concurrency-contract'
SCHEDULE_SCHEMA = 'scar.aten-native-cpu-pair'
SCOPE = 'STANDALONE_CPU_ATEN'
_ATTESTATIONS = ('synchronous_cpu_completion', 'read_only_operands',
                 'no_shared_mutable_or_rng_state', 'no_externally_observable_side_effects')


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


def _lanes(target, cpu_ids):
    if (type(cpu_ids) not in (tuple, list) or len(cpu_ids) != 2
            or any(type(cpu) is not int or not 0 <= cpu < 1024 for cpu in cpu_ids)
            or cpu_ids[0] == cpu_ids[1]):
        raise ValueError('two distinct ordered Linux CPU IDs required')
    graph = TargetHardwareGraph(target)
    constraints = graph.constraints()
    mask = tuple(sorted(cpu_ids))
    if (len(constraints) != 1
            or constraints[0].affinity_cpu_ids != mask or constraints[0].allowed_cpu_ids != mask):
        raise ValueError('proposed CPU mask differs from the recorded target availability domain')
    resources = graph.resources()
    threads = {row.cpu_id: row for row in resources if type(row) is HardwareThread and row.online is True}
    cores = {cpu: row for row in resources if type(row) is CpuCore for cpu in row.cpu_ids}
    if (any(cpu not in threads or cpu not in cores for cpu in cpu_ids)
            or cores[cpu_ids[0]].id == cores[cpu_ids[1]].id):
        raise ValueError('parallel lanes require online threads on distinct physical cores')
    lanes = [dict(cpu=cpu, thread=encode(threads[cpu].id), core=encode(cores[cpu].id),
                  thread_fact=encode(threads[cpu].evidence), core_fact=encode(cores[cpu].evidence))
             for cpu in cpu_ids]
    return encode(graph._snapshot), list(mask), lanes, encode(constraints[0].evidence)


def propose_cpu_pair(recipe, *, contract, target, cpu_ids):
    """Find the first independent adjacent pair with an explicit effect contract.

    The caller normalizes the recipe against the pinned provider before planning.
    The unchanged prefix and suffix run on the caller. Both results become ready
    before the suffix starts. Availability is a placement constraint, while cost
    selection and runtime affinity checks are separate.
    """
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
        snapshot, mask, lanes, constraint_fact = _lanes(target, cpu_ids)
        return dict(schema=SCHEDULE_SCHEMA, schema_version=1, scope=SCOPE,
                    pair=[left['id'], right['id']], pair_indices=[index, index + 1], cpus=list(cpu_ids),
                    statically_shared_reads=sorted(left_reads & right_reads),
                    target=snapshot, availability_mask=mask, lanes=lanes, constraint_fact=constraint_fact,
                    effect_basis='EXPLICIT_STANDALONE_CALLER_CONTRACT',
                    source_replacement_authorized=False, performance_measured=False)
    raise ValueError('no independent adjacent pair has a declared standalone concurrency contract')
