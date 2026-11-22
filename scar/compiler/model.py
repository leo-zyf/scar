"""Types, serialization and admission for the bounded CPU array IR."""

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
from types import MappingProxyType

SCHEMA = 'scar.execution-plan'
PROFILE = 'cpu-array-region-v1'
_ARRAY_ASSUMPTIONS = ('EXACT_NUMPY_NDARRAY_FLOAT64_CPU_2D', 'FRESH_OUTPUT', 'NO_INPUT_WRITES')
_ROW_MAJOR = (('row', 'output.shape[0]'), ('col', 'output.shape[1]'))
_ARITIES = {'input': 0, 'const': 0, 'slice': 1, 'neg': 1, 'copy': 1,
            'add': 2, 'sub': 2, 'mul': 2, 'div': 2}
_FIELDS = {'input': {'name'}, 'const': {'value'}, 'slice': {'bounds'}, 'copy': {'order', 'contract'}}
OPS = frozenset(_ARITIES)

COPY_CONTRACT = 'numpy.ndarray.copy.content-fresh.v1'
COPY_ATTRS = MappingProxyType({'order': 'C', 'contract': COPY_CONTRACT})
COPY_REQUIRED_GUARDS = (
    'EXACT_NUMPY_NDARRAY_COPY_METHOD_DESCRIPTOR',
    'ARRAY_INPUT_CONTENT_STABLE_DURING_REGION',
)


@dataclass(frozen=True)
class SourceRef:
    path: str
    sha256: str
    qualname: str
    line: int
    column: int
    code_key: str
    origins: tuple[dict, ...] = ()


@dataclass(frozen=True)
class Expr:
    id: str
    op: str
    inputs: tuple[str, ...] = ()
    attrs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ArrayRegion:
    id: str
    source: SourceRef
    parameters: tuple[str, ...]
    nodes: tuple[Expr, ...]
    output: str
    assumptions: tuple[str, ...] = _ARRAY_ASSUMPTIONS
    frontiers: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoopAxis:
    variable: str
    extent: str


@dataclass(frozen=True)
class CpuPlacement:
    """A CPU allocation tied to a hardware snapshot with optional memory bounds."""

    target_path: str
    target_sha256: str
    executor_id: str
    cpu_ids: tuple[int, ...]
    memory_domain_id: str | None = None
    memory_budget_bytes: int | None = None
    numa_node: int | None = None
    memory_policy: str = 'UNBOUND'
    evidence: tuple[dict, ...] = ()


@dataclass(frozen=True)
class ExecutionPlan:
    region: ArrayRegion
    schema_version: int = 1
    profile: str = PROFILE
    operator_abi_version: int = 1
    schedule: tuple[LoopAxis, ...] = tuple(LoopAxis(*axis) for axis in _ROW_MAJOR)
    device: str = 'cpu'
    buffer_policy: str = 'fresh_output'
    passes: tuple[str, ...] = ()
    proof: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    cpu_threads: int = 1
    resource_binding: CpuPlacement | None = None


def region_from_dict(value):
    source = dict(value['source'])
    source['origins'] = tuple(source.get('origins', ()))
    nodes = tuple(Expr(node['id'], node['op'], tuple(node.get('inputs', ())), node.get('attrs', {}))
                  for node in value['nodes'])
    return ArrayRegion(value['id'], SourceRef(**source), tuple(value['parameters']), nodes,
                       value['output'], tuple(value.get('assumptions', ())),
                       tuple(value.get('frontiers', ())))


def plan_from_dict(value):
    plan = dict(value)
    if plan.pop('schema', None) != SCHEMA:
        raise ValueError('unsupported execution plan schema')
    plan['region'] = region_from_dict(plan['region'])
    plan['schedule'] = tuple(LoopAxis(**axis) for axis in plan['schedule'])
    plan['passes'] = tuple(plan.get('passes', ()))
    if plan.get('resource_binding') is not None:
        binding = dict(plan['resource_binding'])
        binding['cpu_ids'] = tuple(binding['cpu_ids'])
        binding['evidence'] = tuple(binding.get('evidence', ()))
        plan['resource_binding'] = CpuPlacement(**binding)
    return ExecutionPlan(**plan)


def document(plan):
    result = dict(schema=SCHEMA, **asdict(plan))
    if plan.resource_binding is None:
        result.pop('resource_binding')
    return result


def write_plan(plan, path):
    Path(path).write_text(json.dumps(document(plan), indent=2, allow_nan=False) + '\n')


def read_plan(path):
    return plan_from_dict(json.loads(Path(path).read_text()))


def copy_attributes():
    return dict(COPY_ATTRS)


def validate_copy_attributes(attrs):
    if type(attrs) is not dict or attrs != COPY_ATTRS:
        raise ValueError('copy requires exact ndarray C-order content/fresh-materialization attributes')
    return copy_attributes()


def copy_semantics():
    """Describe independent output storage and equal content at completion."""
    return dict(
        contract=COPY_CONTRACT, relation='copy', equivalence='EXACT_CONTENT_AT_SELECTED_COPY',
        source_content='READ_UNDER_SOURCE_STABILITY', result_content='SAME_CONTENT_ON_NORMAL_COMPLETION',
        materialization='FRESH_INDEPENDENT_C_ORDER_ALLOCATION', aliases_source=False,
        writes='NEW_OUTPUT_ONLY', allocates='FRESH_OUTPUT', may_raise='NUMPY_OR_ALLOCATION_FAILURE',
        equivalence_scope='NORMAL_COMPLETION_NOT_FULL_EXCEPTION_OR_INTERNAL_ALLOCATION_SEQUENCE',
        required_guards=list(COPY_REQUIRED_GUARDS), runtime_value_or_storage_identity_proven=False,
        transformation_authorized=False,
    )


class InvalidPlan(ValueError):
    """The supplied region or plan falls outside the supported profile."""


def validate_region(region):
    if not isinstance(region, ArrayRegion):
        raise InvalidPlan('ArrayRegion required')
    if region.frontiers:
        raise InvalidPlan('unresolved region frontiers: ' + ', '.join(region.frontiers))
    if (not region.parameters or any(type(name) is not str or not name for name in region.parameters)
            or len(set(region.parameters)) != len(region.parameters)):
        raise InvalidPlan('unique array parameters required')
    source = region.source
    if not Path(source.path).is_absolute() or len(source.sha256) != 64 or not source.code_key:
        raise InvalidPlan('versioned source identity required')
    assumptions = set(region.assumptions)
    if not set(_ARRAY_ASSUMPTIONS) <= assumptions:
        raise InvalidPlan('missing guarded array/effect contract')

    by_id = {}
    scalar = {}
    parameters = set()
    for node in region.nodes:
        if node.op not in OPS or not node.id or node.id in by_id:
            raise InvalidPlan('unknown operation or duplicate ID')
        if not isinstance(node.attrs, dict):
            raise InvalidPlan('unsupported operation attributes')
        fields = set(node.attrs)
        if (fields != _FIELDS.get(node.op, set())
                and not (node.op == 'input' and fields == {'name', 'index'})):
            raise InvalidPlan('unsupported operation attributes')
        if len(node.inputs) != _ARITIES[node.op] or any(item not in by_id for item in node.inputs):
            raise InvalidPlan('operation arity, topological order or dependency invalid')

        is_scalar = False
        if node.op == 'input':
            name = node.attrs['name']
            if name not in region.parameters or name in parameters:
                raise InvalidPlan('input does not name one original parameter')
            if 'index' in node.attrs:
                index = node.attrs['index']
                if type(index) is not int or index != region.parameters.index(name):
                    raise InvalidPlan('input ordinal mismatch')
            parameters.add(name)
        elif node.op == 'const':
            value = node.attrs['value']
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or type(value) is int and abs(value) > 2**53):
                raise InvalidPlan('unsupported numeric constant')
            is_scalar = True
        elif node.op == 'slice':
            bounds = node.attrs['bounds']
            if (by_id[node.inputs[0]].op not in ('input', 'slice')
                    or not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                    or any(not isinstance(pair, (list, tuple)) or len(pair) != 2
                           or any(value is not None and (type(value) is not int or abs(value) > 2**63 - 1)
                                  for value in pair) for pair in bounds)):
                raise InvalidPlan('only unit-step 2D input/view slices supported')
        elif node.op == 'copy':
            try:
                validate_copy_attributes(node.attrs)
            except ValueError as error:
                raise InvalidPlan(str(error)) from error
            if scalar[node.inputs[0]] or not set(COPY_REQUIRED_GUARDS) <= assumptions:
                raise InvalidPlan('ndarray copy requires an array and exact descriptor/source-stability guards')
        else:
            is_scalar = all(scalar[item] for item in node.inputs)
        scalar[node.id] = is_scalar
        by_id[node.id] = node

    if parameters != set(region.parameters):
        raise InvalidPlan('every parameter needs an input node')
    if region.output not in by_id or scalar[region.output] or by_id[region.output].op in ('input', 'slice'):
        raise InvalidPlan('output must be a fresh array computation')
    live = set()
    pending = [region.output]
    while pending:
        item = pending.pop()
        if item not in live:
            live.add(item)
            pending.extend(by_id[item].inputs)
    if any(node.id not in live and node.op not in ('input', 'const') for node in region.nodes):
        raise InvalidPlan('dead arithmetic may carry NumPy warning/exception effects')
    return dict(nodes=len(by_id), live_nodes=len(live), array_inputs=len(parameters),
                source_identity=source.sha256)


def validate_resource_binding(binding):
    if type(binding) is not CpuPlacement:
        raise InvalidPlan('typed CPU placement required')
    if (not Path(binding.target_path).is_absolute() or len(binding.target_sha256) != 64
            or any(ch not in '0123456789abcdef' for ch in binding.target_sha256)):
        raise InvalidPlan('versioned target snapshot required')
    if (not binding.executor_id or type(binding.cpu_ids) is not tuple or not binding.cpu_ids
            or any(type(cpu) is not int or cpu < 0 for cpu in binding.cpu_ids)
            or len(set(binding.cpu_ids)) != len(binding.cpu_ids)):
        raise InvalidPlan('unique CPU allocation and executor identity required')
    domain = binding.memory_domain_id
    if domain is not None and (type(domain) is not str or not domain):
        raise InvalidPlan('invalid memory domain reference')
    if binding.memory_policy != 'UNBOUND':
        raise InvalidPlan('NUMA memory placement is not implemented')
    budget = binding.memory_budget_bytes
    if budget is not None and (type(budget) is not int or budget < 8):
        raise InvalidPlan('invalid placement memory budget')
    if binding.numa_node is not None and (type(binding.numa_node) is not int or binding.numa_node < 0):
        raise InvalidPlan('invalid NUMA node')
    if type(binding.evidence) is not tuple or any(type(fact) is not dict for fact in binding.evidence):
        raise InvalidPlan('placement evidence must reference original records')
    return binding


def validate_plan(plan):
    if (not isinstance(plan, ExecutionPlan) or plan.schema_version not in (1, 2)
            or (plan.profile, plan.operator_abi_version) != (PROFILE, 1)):
        raise InvalidPlan('unsupported execution plan schema/profile/ABI')
    if (plan.schema_version == 2) != (plan.resource_binding is not None):
        raise InvalidPlan('resource-bound plans require schema 2; legacy plans require schema 1')
    budget = plan.cost.get('memory_budget_bytes')
    if plan.resource_binding is not None:
        binding = validate_resource_binding(plan.resource_binding)
        if len(binding.cpu_ids) != plan.cpu_threads:
            raise InvalidPlan('CPU allocation and worker count differ')
        if binding.memory_budget_bytes != budget:
            raise InvalidPlan('placement and output memory budgets differ')
    if (plan.device, plan.buffer_policy) != ('cpu', 'fresh_output'):
        raise InvalidPlan('unsupported resource placement or ownership')
    capacity = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    if type(plan.cpu_threads) is not int or not 1 <= plan.cpu_threads <= capacity:
        raise InvalidPlan('CPU thread allocation exceeds available affinity')
    if tuple((axis.variable, axis.extent) for axis in plan.schedule) != _ROW_MAJOR:
        raise InvalidPlan('unsupported loop schedule')
    if plan.proof.get('supported') is False or plan.proof.get('retained_constant_operations'):
        raise InvalidPlan('plan proof retains unsupported operations')
    if budget is not None and (type(budget) is not int or budget < 8):
        raise InvalidPlan('invalid output memory budget')
    return validate_region(plan.region)
