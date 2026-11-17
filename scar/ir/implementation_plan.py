"""Implementation alternatives over source regions, with explicit physical costs.

This is a planning overlay: references name the authoritative program/proofs;
regions are not duplicated and repeated bodies need not be unrolled. Durations
are estimates, while capacities and causal dependencies are hard constraints.
"""
from dataclasses import dataclass, field
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


def digest(value):
    return sha256(json.dumps(encode(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


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
    return dict(schema=SCHEMA,schema_version=1,**encode(validate_problem(problem)))


def problem_from_dict(document):
    value=dict(document)
    _need(value.pop('schema',None)==SCHEMA and value.pop('schema_version',None)==1, 'unsupported planning problem schema')
    return validate_problem(decode(PlanningProblem,value))


__all__=['CostEstimate','PlanningResource','ImplementationOption','PlanningOperation','TransferOption',
         'PlanningDataEdge','PlanningProblem','validate_problem','problem_document','problem_from_dict','digest','SCHEMA','PLAN_SCHEMA']
