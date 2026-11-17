"""Implementation alternatives over source regions, with explicit physical costs.

This is a planning overlay: references name the authoritative program/proofs;
regions are not duplicated and repeated bodies need not be unrolled. Durations
are estimates, while capacities and causal dependencies are hard constraints.
"""
from dataclasses import dataclass, field
from copy import deepcopy
from hashlib import sha256
import json

from scar.ir.program.providers import ProviderArtifact
from scar.ir.record_codec import decode, encode

SCHEMA = 'scar.implementation-planning-problem'
PLAN_SCHEMA = 'scar.implementation-execution-plan'


@dataclass(frozen=True, slots=True)
class CostEstimate:
    estimate_ns: int | None
    lower_ns: int | None
    upper_ns: int | None
    basis: str
    context: str
    source_refs: tuple[dict, ...] = ()
    missing_terms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PlanningResource:
    id: str
    kind: str
    capacity: int
    source_refs: tuple[dict, ...] = ()


@dataclass(frozen=True, slots=True)
class ResourceObligation:
    """Unresolved physical/resource qualification, separate from known time."""
    field: str
    reason: str
    source_refs: tuple[dict, ...]


@dataclass(frozen=True, slots=True)
class ImplementationOption:
    id: str
    lowering: str
    proof_refs: tuple[dict, ...]
    demands: dict[str, int]
    workspace_bytes: dict[str, int]
    input_locations: dict[str, str]
    output_locations: dict[str, str]
    cost: CostEstimate
    parameters: dict
    unresolved_resources: tuple[ResourceObligation, ...] = ()


@dataclass(frozen=True, slots=True)
class PlanningOperation:
    id: str
    source_refs: tuple[dict, ...]
    predecessors: tuple[str, ...]
    options: tuple[ImplementationOption, ...]
    baseline_option: str


@dataclass(frozen=True, slots=True)
class TransferOption:
    id: str
    source_memory: str
    destination_memory: str
    demands: dict[str, int]
    cost: CostEstimate
    lowering: str
    proof_refs: tuple[dict, ...] = ()


@dataclass(frozen=True, slots=True)
class PlanningDataEdge:
    id: str
    value: str
    producer: str | None
    consumer: str
    nbytes: int
    external_location: str | None
    transfers: tuple[TransferOption, ...] = ()
    source_refs: tuple[dict, ...] = ()


@dataclass(frozen=True, slots=True)
class ImplementationChoice:
    operation: str
    option: str


@dataclass(frozen=True, slots=True)
class PlanningActivity:
    """Conditional service; WAIT_TAIL may include waiting without CPU demand.

    Its cost predicts the tail after readiness, not the whole wait interval.
    Endpoint constraints express readiness; UNKNOWN never implies a zero tail.
    """
    id: str
    cost: CostEstimate
    demands: dict[str, int]
    source_refs: tuple[dict, ...]
    when: tuple[ImplementationChoice, ...] = ()
    duration_kind: str = 'FIXED'


@dataclass(frozen=True, slots=True)
class PlanningTimingConstraint:
    """Endpoint delay active when its OR condition and both endpoints exist."""
    id: str
    before: str
    before_endpoint: str
    after: str
    after_endpoint: str
    minimum_ns: int = 0
    maximum_ns: int | None = None
    when: tuple[ImplementationChoice, ...] = ()
    source_refs: tuple[dict, ...] = ()


@dataclass(frozen=True, slots=True)
class PlanningProblem:
    target: ProviderArtifact
    source_refs: tuple[dict, ...]
    resources: tuple[PlanningResource, ...]
    operations: tuple[PlanningOperation, ...]
    data_edges: tuple[PlanningDataEdge, ...]
    objective: str = 'MINIMIZE_COMPLETION_TIME'
    normal_completion_only: bool = True
    assumptions: tuple[str, ...] = ()
    reserved_memory: dict[str, int] = field(default_factory=dict)
    activities: tuple[PlanningActivity, ...] = ()
    timing_constraints: tuple[PlanningTimingConstraint, ...] = ()


def _problem_payload(problem):
    value = encode(problem)
    for operation in value['operations']:
        for option in operation['options']:
            if not option['unresolved_resources']: option.pop('unresolved_resources')
    for name in ('activities', 'timing_constraints'):
        if not value[name]: value.pop(name)
    return value


def digest(value):
    payload = _problem_payload(value) if type(value) is PlanningProblem else encode(value)
    if type(value) is ImplementationOption and not payload['unresolved_resources']:
        payload.pop('unresolved_resources')
    if type(value) is PlanningOperation:
        for option in payload['options']:
            if not option['unresolved_resources']: option.pop('unresolved_resources')
    return sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _need(condition, message):
    if not condition: raise ValueError(message)


def validate_problem(problem):
    _need(type(problem) is PlanningProblem, 'typed planning problem required')
    p=decode(PlanningProblem,encode(problem))
    _need(p.target.schema=='scar.target-hardware-graph' and bool(p.source_refs),
          'original target and source references required')
    _need(p.objective=='MINIMIZE_COMPLETION_TIME', 'unsupported planning objective')
    _need(p.normal_completion_only is True, 'failure/rollback scheduling is outside this normal-completion profile')
    resources={r.id:r for r in p.resources};ops={o.id:o for o in p.operations}
    _need(len(resources)==len(p.resources) and len(ops)==len(p.operations) and bool(ops), 'unique resources and nonempty operations required')
    _need(all(r.id and r.kind in ('EXECUTION','CHANNEL','BANDWIDTH_CLASS','MEMORY')
              and type(r.capacity)is int and r.capacity>0 for r in p.resources), 'positive explicit resource capacities required')
    pools={r.id for r in p.resources if r.kind=='MEMORY'}
    def demands(values, memory=False):
        _need(all(k in resources and (k in pools)==memory and type(v)is int and v>=0
                  for k,v in values.items()), 'demands must name resources of the correct kind with integer units')
    def cost(c):
        _need(c.basis in ('DECLARED_MODEL','EMPIRICAL_MODEL','ANALYTICAL_MODEL','UNKNOWN') and bool(c.context), 'cost basis/context required')
        _need(all(v is None or type(v)is int and 0<=v<2**60 for v in (c.estimate_ns,c.lower_ns,c.upper_ns)), 'costs must be nonnegative bounded integer nanoseconds')
        _need(c.lower_ns is None or c.upper_ns is None or c.lower_ns<=c.upper_ns, 'cost interval is inverted')
        _need(c.estimate_ns is None or (c.lower_ns is None or c.lower_ns<=c.estimate_ns)
              and (c.upper_ns is None or c.estimate_ns<=c.upper_ns), 'estimate outside its stated range')
        _need(c.basis!='UNKNOWN' or c.estimate_ns is None, 'UNKNOWN is not a point estimate')
    def references(refs):
        _need(refs and all(type(ref) is dict and bool(ref) for ref in refs),
              'complete nonempty source reference dictionaries required')
    def condition(choices):
        _need(len(set(choices)) == len(choices) and all(choice.operation in ops
            and choice.option in {m.id for m in ops[choice.operation].options} for choice in choices),
            'conditions must name distinct original implementation choices')
    demands(p.reserved_memory,True)
    for op in p.operations:
        _need(op.id and op.source_refs and len(set(op.predecessors))==len(op.predecessors)
              and set(op.predecessors)<=set(ops)-{op.id}, 'original region references and valid predecessors required')
        choices={m.id for m in op.options}
        _need(op.options and len(choices)==len(op.options) and op.baseline_option in choices, 'unique implementation choices and baseline required')
        for m in op.options:
            _need(m.id and m.lowering and m.proof_refs, 'implementation lowering and legality references required')
            demands(m.demands);demands(m.workspace_bytes,True);cost(m.cost)
            _need(set(m.input_locations.values())|set(m.output_locations.values())<=pools, 'material locations must name explicit memory pools')
            for obligation in m.unresolved_resources:
                _need(bool(obligation.field) and bool(obligation.reason), 'resource obligation field and reason required')
                references(obligation.source_refs)
    activities={a.id:a for a in p.activities}
    _need(len(activities)==len(p.activities) and not set(activities)&set(ops),
          'auxiliary activity IDs must be unique and distinct from operation IDs')
    for activity in p.activities:
        _need(bool(activity.id), 'nonempty auxiliary activity ID required')
        references(activity.source_refs); demands(activity.demands); cost(activity.cost); condition(activity.when)
        _need(activity.duration_kind in ('FIXED','WAIT_TAIL')
            and (activity.duration_kind!='WAIT_TAIL' or not activity.demands),
            'WAIT_TAIL intervals may stretch and cannot claim whole-interval resource demand')
    constraints={c.id:c for c in p.timing_constraints}
    _need(len(constraints)==len(p.timing_constraints), 'unique timing constraint IDs required')
    nodes=set(ops)|set(activities)
    for constraint in p.timing_constraints:
        _need(constraint.id and constraint.before in nodes and constraint.after in nodes
            and constraint.before_endpoint in ('START','END') and constraint.after_endpoint in ('START','END'),
            'timing endpoints must name original operations or auxiliary activities and START/END')
        _need(type(constraint.minimum_ns) is int and constraint.minimum_ns>=0
            and (constraint.maximum_ns is None or type(constraint.maximum_ns) is int
                and constraint.maximum_ns>=constraint.minimum_ns), 'nonnegative ordered timing bounds required')
        references(constraint.source_refs); condition(constraint.when)
    edges={e.id:e for e in p.data_edges};values={}
    _need(len(edges)==len(p.data_edges), 'unique data edges required')
    _need(len({(e.value,e.consumer)for e in p.data_edges})==len(p.data_edges), 'one value port per consumer is required')
    for e in p.data_edges:
        _need(e.id and e.value and e.consumer in ops and e.producer in (None,*ops)
              and e.producer!=e.consumer and type(e.nbytes)is int and e.nbytes>=0 and e.source_refs,
              'data edge must retain its producer/consumer, size and source')
        identity=(e.producer,e.nbytes,e.external_location)
        _need(e.value not in values or values[e.value]==identity, 'one value cannot have different producers/extents/locations')
        values[e.value]=identity
        _need((e.producer is None and e.external_location in pools)or(e.producer is not None and e.external_location is None), 'external location only applies to external roots')
        _need(all(e.value in m.input_locations for m in ops[e.consumer].options), 'every consumer mode must declare the input location')
        if e.producer is not None:
            _need(all(e.value in m.output_locations for m in ops[e.producer].options), 'every producer mode must declare the output location')
        _need(len({t.id for t in e.transfers})==len(e.transfers), 'unique transfer choices required per edge')
        for t in e.transfers:
            _need(t.source_memory in pools and t.destination_memory in pools and t.source_memory!=t.destination_memory
                  and t.lowering and t.proof_refs, 'transfer endpoints, implementation and proof required')
            demands(t.demands);cost(t.cost)
    predecessors={o.id:set(o.predecessors) for o in p.operations}
    for op in p.operations:
        inputs={e.value for e in p.data_edges if e.consumer==op.id}
        outputs={e.value for e in p.data_edges if e.producer==op.id}
        _need(all(set(m.input_locations)==inputs and set(m.output_locations)==outputs for m in op.options),
              'all declared material ports need corresponding data edges; escaping storage belongs in the scope reservation')
    for e in p.data_edges:
        if e.producer is not None:predecessors[e.consumer].add(e.producer)
    completed=set()
    while len(completed)<len(ops):
        ready={o for o,deps in predecessors.items()if o not in completed and deps<=completed}
        _need(bool(ready), 'region-level dependency graph must be acyclic; loops belong in shared region bodies')
        completed.update(ready)
    return p


def problem_document(problem):
    problem=validate_problem(problem)
    obligations=any(mode.unresolved_resources for op in problem.operations for mode in op.options)
    return dict(schema=SCHEMA,schema_version=3 if obligations else 2 if problem.activities or problem.timing_constraints else 1,
                **_problem_payload(problem))


def problem_from_dict(document):
    value=deepcopy(document)
    version=value.pop('schema_version',None)
    _need(value.pop('schema',None)==SCHEMA and type(version) is int and version in (1,2,3),
          'unsupported planning problem schema')
    extension=('activities','timing_constraints')
    _need(version!=1 or not set(extension)&set(value), 'v1 planning problems cannot carry conditional activities or timing')
    _need(version!=2 or any(value.get(name) for name in extension), 'v2 planning problem requires a nonempty extension')
    options=[mode for op in value['operations'] for mode in op['options']]
    _need(version==3 or not any('unresolved_resources' in mode for mode in options),
          'resource obligations require planning problem schema v3')
    _need(version!=3 or any(mode.get('unresolved_resources') for mode in options),
          'v3 planning problem requires nonempty resource obligations')
    for mode in options: mode.setdefault('unresolved_resources',[])
    for name in extension: value.setdefault(name,[])
    return validate_problem(decode(PlanningProblem,value))


__all__=['CostEstimate','PlanningResource','ResourceObligation','ImplementationOption','PlanningOperation','TransferOption',
         'PlanningDataEdge','ImplementationChoice','PlanningActivity','PlanningTimingConstraint',
         'PlanningProblem','validate_problem','problem_document','problem_from_dict','digest','SCHEMA','PLAN_SCHEMA']
