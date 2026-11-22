"""Entry mandatory bound."""

from collections import defaultdict
from dataclasses import asdict, dataclass
from graphlib import CycleError, TopologicalSorter
from scar.ir.implementation_plan import problem_from_dict, CostEstimate
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program.providers import ProviderArtifact
from scar.ir.record_codec import encode


# Lower bound on whole-entry completion from a declared mandatory subgraph.
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


# Exact-context implementation cost lookup, without extrapolation or timing.
#
# An empirical range is a sample range, not a physical guarantee. A declared
# model is a mathematical input. Neither changes a component's query scope.
SCHEMA = 'scar.implementation-cost-profile'


@dataclass(frozen=True, slots=True)
class CostQuery:
    scope: str
    context: dict
    source_refs: tuple[dict, ...] = ()


def _need(condition, message):
    if not condition: raise ValueError(message)


def _document(query):
    _need(type(query) is CostQuery and type(query.scope) is str and bool(query.scope)
          and type(query.context) is dict, 'typed scope and complete context required')
    _need(type(query.source_refs) is tuple and all(type(r) is dict for r in query.source_refs),
          'query source references must be a tuple of records')
    value = dict(scope=query.scope, context=encode(query.context)); pending = [value]
    while pending:
        row = pending.pop()
        if type(row) is dict:
            _need(all(type(k) is str for k in row), 'context object keys must be strings')
            pending.extend(row.values())
        elif type(row) in (tuple, list): pending.extend(row)
    # Reject non-JSON/nonfinite context data; normalize tuple/list wire forms.
    return json.loads(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False))


def _key(document):
    return 'implementation-cost:'+sha256(json.dumps(document, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def query_key(query):
    """Identity includes every scope/context field, excluding evidence refs."""
    return _key(_document(query))


def _strings(value, name):
    _need(type(value) is list and all(type(s) is str and s for s in value), name+' must be a string list')
    return tuple(value)


def resolve_cost(query, profile_ref=None):
    document = _document(query); key = _key(document); refs = deepcopy(query.source_refs)
    def unknown(reason, extra=()):
        return CostEstimate(None, None, None, 'UNKNOWN', key, refs+extra, (reason,))
    if profile_ref is None: return unknown('NO_EXACT_CONTEXT_COST_PROFILE')
    if isinstance(profile_ref, (str, Path)):
        reference = artifact_reference(Path(profile_ref).resolve(), SCHEMA)
    else:
        reference = encode(profile_ref) if type(profile_ref) is ProviderArtifact else deepcopy(profile_ref)
    _need(type(reference) is dict and reference.get('schema') == SCHEMA,
          'original implementation cost profile artifact required')
    profile = read_artifact(reference)
    _need(type(profile.get('schema_version')) is int and profile['schema_version'] == 1
          and type(profile.get('entries')) is dict, 'versioned exact-context cost entries required')
    if key not in profile['entries']:
        return unknown('EXACT_CONTEXT_COST_ENTRY_MISSING',
            (dict(artifact=reference, locator='#/entries', exact_context_match=False),))
    row = profile['entries'][key]
    _need(type(row) is dict and row.get('query') == document,
          'profile entry scope/context differs from the complete query')
    _need({'basis','estimate_ns','lower_ns','upper_ns','missing_terms','conditions'} <= set(row),
          'complete cost statistics, conditions and missing terms required')
    _need(_key(row['query']) == key, 'cost entry key differs from its query identity')
    basis = row['basis']; point, lower, upper = (row[k] for k in ('estimate_ns', 'lower_ns', 'upper_ns'))
    _need(basis in ('DECLARED_MODEL', 'EMPIRICAL_MODEL', 'ANALYTICAL_MODEL', 'UNKNOWN'), 'unsupported cost basis')
    _need(all(v is None or type(v) is int and 0 <= v < 2**60 for v in (point, lower, upper)),
          'integer nonnegative bounded nanosecond costs required')
    _need(lower is None or upper is None or lower <= upper, 'cost interval is inverted')
    _need(point is None or (lower is None or lower <= point) and (upper is None or point <= upper),
          'cost estimate lies outside the stated interval')
    _need(basis != 'UNKNOWN' or point is None, 'UNKNOWN cannot supply a point estimate')
    missing = _strings(row['missing_terms'], 'missing_terms')
    conditions = _strings(row['conditions'], 'conditions')
    semantics = ('DECLARED_MATHEMATICAL_MODEL_NOT_MEASUREMENT' if basis == 'DECLARED_MODEL'
                 else 'CONDITIONAL_CONTEXT_COST_NOT_PHYSICAL_GUARANTEE')
    if basis == 'EMPIRICAL_MODEL':
        raw = row.get('observations')
        _need(type(raw) is dict and type(raw.get('warmup')) is int and raw['warmup'] >= 0
              and type(raw.get('repetitions')) is int and raw['repetitions'] > 0,
              'empirical cost requires explicit warmup and repetitions')
        samples = raw.get('samples_ns')
        _need(type(samples) is list and len(samples) == raw['repetitions']
              and all(type(v) is int and 0 <= v < 2**60 for v in samples),
              'all retained post-warmup integer nanosecond samples are required')
        ordered = sorted(samples); count = len(ordered)
        middle = 2*ordered[count//2] if count % 2 else ordered[count//2-1]+ordered[count//2]
        _need(middle % 2 == 0, 'fractional nanosecond median cannot fit the integer cost contract')
        _need((point, lower, upper) == (middle//2, ordered[0], ordered[-1]),
              'empirical median/min/max differ from retained raw samples')
        semantics = 'POST_WARMUP_EMPIRICAL_MEDIAN_AND_SAMPLE_RANGE_NOT_GUARANTEED_BOUNDS'
    pointer = '#/entries/'+key.replace('~', '~0').replace('/', '~1')
    evidence = dict(artifact=reference, locator=pointer, query_scope=document['scope'],
        conditions=list(conditions), cost_interpretation=semantics,
        component_cost_is_not_promoted_to_whole_call=True)
    return CostEstimate(point, lower, upper, basis, key, refs+(evidence,), missing)


__all__ = [
    'SCHEMA',
    'CostQuery',
    'query_key',
    'resolve_cost',
    'KIND',
    'mandatory_entry_lower_bound',
]
