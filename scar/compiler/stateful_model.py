"""Ordered stateful array-region profile, derived from shared original F4.

Expressions are immutable values. Slice nodes are alias descriptors; input
epochs denote this region's ordered writes, never global content identities.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from .model import Expr, SourceRef

SCHEMA = 'scar.stateful-array-region'
PROFILE = 'cpu-stateful-array-region-v1'
POWER2_SEMANTICS = 'NUMPY_UFUNC_POWER_FLOAT64_EXACT_PYTHON_INT_TWO_NOT_MULTIPLICATION_REWRITE'
REQUIRED_GUARDS = (
    'EXACT_NDARRAY_FLOAT64_CPU_RANK2', 'EXACT_FINITE_PYTHON_SCALARS',
    'ORDINARY_RECEIVER_DICTIONARY_FIELDS_NO_DESCRIPTORS', 'DISTINCT_ARRAY_INPUTS_HAVE_DISJOINT_STORAGE',
    'EXACT_SAME_SOURCE_STATICMETHOD_CODE_IDENTITIES', 'EXACT_NUMPY_PROVIDER_IDENTITIES',
    'NO_FOREIGN_CONCURRENT_INPUT_WRITERS', 'NO_FOREIGN_CONCURRENT_STATE_READERS_OR_WRITERS',
    'SYNCHRONOUS_CPU_EXECUTION', 'ORDERED_TRANSACTIONAL_SHADOW_WRITES_BEFORE_COMMIT',
    'ORIGINAL_FLOATING_ENVIRONMENT_AND_FALLBACK_CONTRACT')


@dataclass(frozen=True)
class StatefulInput:
    id: str
    path: tuple[str, ...]
    kind: str
    origins: tuple[dict, ...] = ()
    kind_authority: str = 'RUNTIME_GUARDED_CANDIDATE'


@dataclass(frozen=True)
class StatefulCallee:
    source: SourceRef
    path: tuple[str, ...]
    kind: str = 'STATICMETHOD'
    origins: tuple[dict, ...] = ()


@dataclass(frozen=True)
class StatefulProvider:
    id: str
    module: str
    symbol: str
    lookup_path: tuple[str, ...]
    semantics: str
    origins: tuple[dict, ...] = ()


@dataclass(frozen=True)
class StatefulAction:
    id: str
    kind: str
    value: str
    target: str | None = None
    bounds: tuple[tuple[int | None, int | None], ...] | None = None
    epoch_before: int | None = None
    epoch_after: int | None = None
    op: str | None = None
    world_before: dict | None = None
    world_after: dict | None = None
    origins: tuple[dict, ...] = ()


@dataclass(frozen=True)
class StatefulRegion:
    id: str
    source: SourceRef
    parameter_names: tuple[str, ...]
    inputs: tuple[StatefulInput, ...]
    nodes: tuple[Expr, ...]
    actions: tuple[StatefulAction, ...]
    callees: tuple[StatefulCallee, ...] = ()
    providers: tuple[StatefulProvider, ...] = ()
    guards: tuple[str, ...] = ()
    frontiers: tuple[dict, ...] = ()
    return_kind: str = 'NONE'
    profile: str = PROFILE
    schema_version: int = 1


def _bounds(bounds):
    return (isinstance(bounds, (tuple, list)) and len(bounds) == 2
        and all(isinstance(axis, (tuple, list)) and len(axis) == 2
                and all(v is None or type(v) is int and abs(v) <= 2**63 - 1 for v in axis) for axis in bounds))


def validate_stateful_region(region):
    if (type(region) is not StatefulRegion or region.schema_version != 1 or region.profile != PROFILE
            or region.return_kind != 'NONE' or not region.id or region.frontiers):
        raise ValueError('closed versioned stateful array region returning None required')
    def check_source(source):
        if (type(source) is not SourceRef or type(source.path) is not str or not Path(source.path).is_absolute()
                or type(source.sha256) is not str or re.fullmatch('[0-9a-f]{64}', source.sha256) is None
                or type(source.code_key) is not str or re.fullmatch('pycode-v2:[0-9a-f]{64}', source.code_key) is None
                or type(source.qualname) is not str or not source.qualname
                or type(source.line) is not int or source.line < 1 or type(source.column) is not int or source.column < 0):
            raise ValueError('original complete source/code identity required')
    source = region.source; check_source(source)
    if not set(REQUIRED_GUARDS).issubset(region.guards) or any(type(g) is not str for g in region.guards):
        raise ValueError('complete guarded receiver/array/state/provider boundary required')
    if not region.parameter_names or len(set(region.parameter_names)) != len(region.parameter_names):
        raise ValueError('original ordered parameter names required')
    inputs = {row.id: row for row in region.inputs}
    if len(inputs) != len(region.inputs) or len({row.path for row in region.inputs}) != len(region.inputs):
        raise ValueError('unique state input IDs and paths required')
    for row in region.inputs:
        if (type(row) is not StatefulInput or not row.id or not row.path or row.path[0] not in region.parameter_names
                or any(type(v) is not str or not v.isidentifier() for v in row.path) or row.kind not in ('array', 'scalar')
                or row.kind_authority != 'RUNTIME_GUARDED_CANDIDATE'):
            raise ValueError('input needs original parameter/field path and guarded array/scalar kind')
    providers = {p.id: p for p in region.providers}
    if len(providers) != len(region.providers):
        raise ValueError('unique semantic provider IDs required')
    for provider in region.providers:
        if (type(provider) is not StatefulProvider or type(provider.id) is not str or not provider.id
                or (provider.module, provider.symbol, provider.semantics) != ('numpy', 'power', POWER2_SEMANTICS)
                or type(provider.lookup_path) is not tuple or len(provider.lookup_path) != 2
                or any(type(v) is not str or not v.isidentifier() for v in provider.lookup_path)
                or provider.lookup_path[-1] != provider.symbol):
            raise ValueError('provider requires the independent exact NumPy power2 semantic contract and global lookup path')
    paths = set()
    for callee in region.callees:
        if (type(callee) is not StatefulCallee or callee.kind != 'STATICMETHOD' or type(callee.path) is not tuple
                or len(callee.path) != 2 or callee.path[0] not in region.parameter_names
                or any(type(v) is not str or not v.isidentifier() for v in callee.path) or callee.path in paths):
            raise ValueError('callee needs a unique direct receiver staticmethod guard path')
        check_source(callee.source)
        if (callee.source.path, callee.source.sha256) != (source.path, source.sha256):
            raise ValueError('callee must retain the same original source snapshot')
        paths.add(callee.path)
    nodes, kinds = {}, {}
    arities = dict(input=0, const=0, slice=1, add=2, sub=2, mul=2, div=2, neg=1, power2=1)
    for node in region.nodes:
        if (type(node) is not Expr or not node.id or node.id in nodes or node.op not in arities
                or len(node.inputs) != arities[node.op] or any(i not in nodes for i in node.inputs)):
            raise ValueError('supported topologically ordered immutable expressions required')
        if node.op == 'input':
            if set(node.attrs) != {'name', 'kind', 'epoch'} or node.attrs['name'] not in inputs:
                raise ValueError('input expression needs its input kind and ordered epoch')
            row = inputs[node.attrs['name']]; epoch = node.attrs['epoch']
            if node.attrs['kind'] != row.kind or (row.kind == 'array' and (type(epoch) is not int or epoch < 0)) or (row.kind == 'scalar' and epoch is not None):
                raise ValueError('input expression kind/epoch differs')
            kinds[node.id] = row.kind
        elif node.op == 'const':
            value = node.attrs.get('value')
            if (set(node.attrs) != {'value'} or type(value) not in (int, float) or not math.isfinite(value)
                    or type(value) is int and abs(value) > 2**53):
                raise ValueError('finite exactly representable numeric constant required')
            kinds[node.id] = 'scalar'
        elif node.op == 'slice':
            if (set(node.attrs) != {'bounds', 'semantics'} or node.attrs['semantics'] != 'ALIAS_DESCRIPTOR'
                    or not _bounds(node.attrs['bounds']) or nodes[node.inputs[0]].op not in ('input', 'slice')
                    or kinds[node.inputs[0]] != 'array'):
                raise ValueError('only constant unit-step array alias descriptors supported')
            kinds[node.id] = 'array'
        elif node.op == 'power2':
            if set(node.attrs) != {'provider'} or node.attrs['provider'] not in providers or kinds[node.inputs[0]] != 'array':
                raise ValueError('power2 needs its independent provider and array operand')
            kinds[node.id] = 'array'
        else:
            if node.attrs:
                raise ValueError('arithmetic expression attributes unsupported')
            kinds[node.id] = 'array' if any(kinds[i] == 'array' for i in node.inputs) else 'scalar'
        nodes[node.id] = node
    epochs = {key: 0 for key, row in inputs.items() if row.kind == 'array'}
    ready, evaluated, action_ids = set(), set(), set()
    def require(value):
        if value in ready:
            return
        node = nodes[value]
        if node.op == 'input':
            if node.attrs['kind'] == 'array' and node.attrs['epoch'] != epochs[node.attrs['name']]:
                raise ValueError('array alias read does not use the current ordered storage epoch')
        elif node.op == 'slice':
            require(node.inputs[0])
        elif node.op != 'const':
            raise ValueError('arithmetic value consumed before its original evaluation action')
    for action in region.actions:
        if (type(action) is not StatefulAction or not action.id or action.id in action_ids
                or action.value not in nodes or action.kind not in ('evaluate', 'write')):
            raise ValueError('ordered unique evaluation/write actions required')
        action_ids.add(action.id)
        if action.kind == 'evaluate':
            node = nodes[action.value]
            if node.op in ('input', 'const', 'slice') or action.value in evaluated or action.target is not None:
                raise ValueError('evaluation actions materialize arithmetic values exactly once')
            for value in node.inputs: require(value)
            ready.add(node.id); evaluated.add(node.id)
        else:
            if (action.target not in epochs or action.op not in ('assign', 'add', 'sub', 'mul', 'div')
                    or action.bounds is not None and not _bounds(action.bounds)
                    or action.epoch_before != epochs[action.target] or action.epoch_after != action.epoch_before + 1):
                raise ValueError('write requires original target/slice and consecutive storage epochs')
            require(action.value)
            epochs[action.target] += 1
    if not any(action.kind == 'write' for action in region.actions):
        raise ValueError('stateful region needs an explicit array write')
    if evaluated != {n.id for n in region.nodes if n.op not in ('input', 'const', 'slice')}:
        raise ValueError('every arithmetic operation must retain its ordered evaluation')
    return region


def document(region):
    return dict(schema=SCHEMA, **asdict(validate_stateful_region(region)))


def stateful_from_dict(value):
    value = dict(value)
    if value.pop('schema', None) != SCHEMA: raise ValueError('unsupported stateful region schema')
    def source(row):
        row = dict(row); row['origins'] = tuple(row.get('origins', ())); return SourceRef(**row)
    value['source'] = source(value['source']); value['parameter_names'] = tuple(value['parameter_names'])
    value['inputs'] = tuple(StatefulInput(**{**r, 'path': tuple(r['path']), 'origins': tuple(r.get('origins', ()))}) for r in value['inputs'])
    value['nodes'] = tuple(Expr(r['id'], r['op'], tuple(r['inputs']), r['attrs']) for r in value['nodes'])
    value['actions'] = tuple(StatefulAction(**{**r, 'bounds': None if r.get('bounds') is None else tuple(tuple(b) for b in r['bounds']),
        'origins': tuple(r.get('origins', ()))}) for r in value['actions'])
    value['callees'] = tuple(StatefulCallee(source(r['source']), tuple(r['path']), r['kind'], tuple(r.get('origins', ()))) for r in value.get('callees', ()))
    value['providers'] = tuple(StatefulProvider(**{**r, 'lookup_path': tuple(r['lookup_path']), 'origins': tuple(r.get('origins', ()))}) for r in value.get('providers', ()))
    for field in ('guards', 'frontiers'): value[field] = tuple(value.get(field, ()))
    return validate_stateful_region(StatefulRegion(**value))


def stateful_digest(region):
    return sha256(json.dumps(document(region), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_stateful(path): return stateful_from_dict(json.loads(Path(path).read_bytes()))
def write_stateful(region, path): Path(path).write_text(json.dumps(document(region), indent=2, allow_nan=False) + '\n')


__all__ = ['StatefulRegion', 'StatefulInput', 'StatefulAction', 'StatefulCallee', 'StatefulProvider',
    'validate_stateful_region', 'document', 'stateful_from_dict', 'stateful_digest', 'read_stateful', 'write_stateful',
    'REQUIRED_GUARDS', 'POWER2_SEMANTICS']
