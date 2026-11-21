"""Lower bound on whole-entry completion from a declared mandatory subgraph."""

from collections import defaultdict
from dataclasses import asdict
from graphlib import CycleError, TopologicalSorter

from scar.ir.implementation_plan import problem_from_dict


KIND = 'MANDATORY_SUBGRAPH_LOWER_BOUND'
_MODEL = ('ANALYTICAL_MODEL', 'DECLARED_MODEL')


def _matches(fact, cost):
    return all(fact.get(name) == cost[name] for name in ('option_id', 'scope_sha256'))


def _scoped_facts(references, cost, facts):
    for reference in references:
        if set(reference) == {'artifact', 'locator'}:
            proof = facts.fact(reference)
            if _matches(proof, cost):
                yield proof


def _duration(activity, estimate, cost, facts):
    lower = estimate.lower_ns
    if estimate.basis not in _MODEL or not lower:
        return 0
    for proof in _scoped_facts(estimate.source_refs, cost, facts):
        if (proof.get('kind') == 'MANDATORY_ACTIVITY_LOWER_BOUND'
                and proof.get('activity_id') == activity and type(proof.get('lower_ns')) is int
                and proof['lower_ns'] >= lower):
            return lower
    raise ValueError('positive mandatory duration needs a matched model fact: ' + activity)


def _capacity(resource, cost, facts):
    for proof in _scoped_facts(resource.source_refs, cost, facts):
        if (proof.get('kind') == 'MANDATORY_RESOURCE_CAPACITY'
                and proof.get('resource_id') == resource.id and type(proof.get('upper_capacity')) is int
                and proof['upper_capacity'] == resource.capacity):
            return resource.capacity
    return None


def _delay(relation, cost, facts):
    delay = relation.minimum_ns
    if delay and not any(proof.get('kind') == 'MANDATORY_ORDER_MIN_DELAY'
            and proof.get('before') == relation.before and proof.get('after') == relation.after
            and type(proof.get('lower_ns')) is int and proof['lower_ns'] >= delay
            for proof in _scoped_facts(relation.source_refs, cost, facts)):
        return 0
    return delay


def mandatory_entry_lower_bound(cost, scope, facts):
    problem = problem_from_dict(facts.read(cost['problem']))
    if (cost.get('basis') not in _MODEL or scope.get('target') != asdict(problem.target)
            or any(len(op.options) != 1 for op in problem.operations)):
        raise ValueError('mandatory subgraph needs a sourced fixed-implementation model on the exact target')

    durations, energy = {}, defaultdict(int)
    predecessors = {}
    for op in problem.operations:
        mode, = op.options
        duration = durations[op.id] = _duration(op.id, mode.cost, cost, facts)
        predecessors[op.id] = dict.fromkeys(op.predecessors, 0)
        if duration and not mode.unresolved_resources:
            for resource, units in {**mode.demands, **mode.workspace_bytes}.items():
                energy[resource] += units * duration
    for activity in problem.activities:
        if activity.when:
            continue
        duration = durations[activity.id] = _duration(activity.id, activity.cost, cost, facts)
        predecessors[activity.id] = {}
        for resource, units in activity.demands.items():
            energy[resource] += units * duration
    for edge in problem.data_edges:
        if edge.producer is not None:
            predecessors[edge.consumer][edge.producer] = 0
    for relation in problem.timing_constraints:
        if (not relation.when and relation.before in durations and relation.after in durations
                and (relation.before_endpoint, relation.after_endpoint) == ('END', 'START')):
            parents = predecessors[relation.after]
            parents[relation.before] = max(parents.get(relation.before, 0), _delay(relation, cost, facts))

    ends = {}
    try:
        order = TopologicalSorter(predecessors).static_order()
        for child in order:
            parents = predecessors[child]
            ends[child] = durations[child] + max((ends[parent] + delay for parent, delay in parents.items()), default=0)
    except CycleError as error:
        raise ValueError('mandatory end-to-start subgraph must be acyclic') from error
    critical_path = max(ends.values())
    resource_bounds = {}
    for resource in problem.resources:
        amount = energy[resource.id]
        if amount:
            capacity = _capacity(resource, cost, facts)
            if capacity is not None:
                resource_bounds[resource.id] = (amount + capacity - 1) // capacity
    lower = max(critical_path, max(resource_bounds.values(), default=0))
    return lower, dict(problem=cost['problem'], proof='MANDATORY_CRITICAL_PATH_AND_RESOURCE_CAPACITY',
        critical_path_ns=critical_path, resource_lower_bounds_ns=resource_bounds,
        activities_without_positive_bound=sorted(name for name, duration in durations.items() if not duration),
        complete_lifecycle_covered=False, bound_ns=lower)
