"""Suspension interfaces."""

from dataclasses import dataclass
from enum import Enum
import re
from scar.ir.program import source_productions as P
from scar.ir.program.control import LanguageCompletionKind, ProtocolValueKind
from scar.ir.program.normal_graph import CodeScopeRef, CodeScopeWitness, GlobalPort
from scar.ir.program.protocol_signatures import ExceptionABI
from scar.ir.program.source_declarations import SourceTemplateKey
from scar.ir.program.structured import PythonProtocolKind
from scar.ir.program.model import OperationResult, ProtocolValue
from scar.ir.program.protocols import OwnedUse, HostTerminal
from copy import deepcopy
from scar.ir.execution.validity_budget import ValidityBudget
from scar.ir.program.protocol_connection_validation import ConnectionContext
from scar.ir.program.protocol_packet_routes import completion_continuations as _query


# Original deferred/iteration interfaces, without asserting driver execution.
class WitnessSource(str, Enum):
    SEALED_TEMPLATE_STORE = 'SEALED_TEMPLATE_STORE'
    SEALED_COMPILED_LOCAL_MATERIAL = 'SEALED_COMPILED_LOCAL_MATERIAL'
    CONTROLLED_PLAN_PROVIDER = 'CONTROLLED_PLAN_PROVIDER'


class DriverObligation(str, Enum):
    NOT_SYNCHRONOUS_GENERATOR = 'NOT_SYNCHRONOUS_GENERATOR'
    YIELD_FROM_DRIVER_REQUIRED = 'YIELD_FROM_DRIVER_REQUIRED'
    AWAIT_DRIVER_REQUIRED = 'AWAIT_DRIVER_REQUIRED'
    ASYNC_GENERATOR_DRIVER_REQUIRED = 'ASYNC_GENERATOR_DRIVER_REQUIRED'


@dataclass(frozen=True, slots=True)
class StepWitness:
    unit: str
    template: SourceTemplateKey
    plan_sha256: str
    step: P.ProductionStepID
    region: P.ProductionRegionID
    owner: CodeScopeRef
    source: WitnessSource
    template_result_sha256: str | None = None
    model_manifest_sha256: str | None = None

    def __post_init__(self):
        if self.owner.unit != self.unit:
            raise ValueError('suspension witness owner belongs to another source unit')
        for digest in (self.plan_sha256, self.template_result_sha256, self.model_manifest_sha256):
            if digest is not None and re.fullmatch('[0-9a-f]{64}', digest) is None:
                raise ValueError('interface witness requires an exact SHA256')
        if type(self.source) is not WitnessSource:
            raise TypeError('interface witness source must be explicit')
        if self.source is WitnessSource.SEALED_TEMPLATE_STORE:
            valid = self.template_result_sha256 is not None and self.model_manifest_sha256 is not None
        elif self.source is WitnessSource.SEALED_COMPILED_LOCAL_MATERIAL:
            valid = self.template_result_sha256 is not None and self.model_manifest_sha256 is None
        else:
            valid = self.template_result_sha256 is None and self.model_manifest_sha256 is None
        if not valid:
            raise ValueError('interface witness receipts differ from their material kind')


@dataclass(frozen=True, slots=True)
class CompletionPorts:
    kind: LanguageCompletionKind | P.PatternCompletionKind
    target: P.ProductionControlTarget | None
    ports: tuple[GlobalPort, ...]
    world: GlobalPort | None

    def __post_init__(self):
        if self.world is not None and self.world not in self.ports:
            raise ValueError('completion world must be one of its original ports')


@dataclass(frozen=True, slots=True)
class IterationSite:
    witness: StepWitness
    protocol: PythonProtocolKind
    source: GlobalPort
    value: GlobalPort
    world: GlobalPort
    result_world: GlobalPort
    completions: tuple[CompletionPorts, ...]


@dataclass(frozen=True, slots=True)
class ResumeSite:
    kind: LanguageCompletionKind
    region: P.ProductionRegionID
    parameters: tuple[GlobalPort, ...]
    world: GlobalPort

    @property
    def sent(self):
        return self.parameters[0] if self.kind is LanguageCompletionKind.FALLTHROUGH else None


@dataclass(frozen=True, slots=True)
class SuspensionSite:
    witness: StepWitness
    kind: P.SuspensionKind
    exported: GlobalPort
    world: GlobalPort
    resumes: tuple[ResumeSite, ...]
    saved: tuple[GlobalPort, ...]
    delegate: GlobalPort | None
    completions: tuple[CompletionPorts, ...]
    exception_abi: ExceptionABI
    obligations: tuple[DriverObligation, ...] = ()


@dataclass(frozen=True, slots=True)
class DeferredBodySite:
    body_scope: CodeScopeWitness
    kind: P.DeferredBodyKind
    source_callable: GlobalPort
    parameters: tuple[GlobalPort, ...]
    completions: tuple[CompletionPorts, ...]
    suspensions: tuple[SuspensionSite, ...]
    obligations: tuple[DriverObligation, ...] = ()


# Continuation ports.
# Original adapter packets and typed cleanup continuations, without aliases.
def structured_region(ctx, source):
    port = ctx.port(source)
    if port is None or port.operation is None or port.gap is not None:
        return False
    regions = getattr(ctx, '_completion_cleanup_regions', None)
    if regions is None:
        scopes = {interface.scope for interface in ctx.graph.cleanup_interfaces}
        regions = ctx._completion_cleanup_regions = {
            scope.region for scope in ctx.graph.control_scopes
            if scope.id in scopes}
    return ctx.region_of[port.operation] in regions


def packets(ctx, source, choices):
    """A Reference may be a packet component, never the whole packet alias."""
    port = ctx.port(source)
    if port is None:
        return (), ('await_completion_result_unresolved',)
    found, gaps = [], []
    if port.values and all(isinstance(ctx.values[value], ProtocolValue)
            and ctx.values[value].protocol is ProtocolValueKind.EXCEPTION_PACKET for value in port.values):
        return ({'source': source, 'definition': None, 'exception_binding': None},), ()
    for packet in ctx.packets.values():
        if not ctx.spend('await_completion_packet_candidates'):
            gaps.append('await_completion_packet_budget')
            break
        origin = ctx.values[packet.value].origin
        if type(origin) is not OperationResult or ctx.region_of[origin.operation] != ctx.region_of[port.operation]:
            continue
        component = ctx.bind(source, OwnedUse(origin.operation, packet.exception), guard=choices or None)
        if component['match'] is True:
            found.append({'source': origin, 'definition': packet, 'exception_binding': component})
        elif component['match'] is None:
            gaps.append('await_completion_packet_component_unproved')
    if not found:
        gaps.append('await_completion_whole_packet_unproved')
    return tuple(found), tuple(dict.fromkeys(gaps))


def continuation_rows(ctx, source, choices=(), *, candidates=None):
    """Demand-bound same-region receipts for the original known packets."""
    from .protocol_packet_routes import completion_continuations
    from .data_materials import required_choices, selected
    required, complete = required_choices(ctx, source)
    choices = selected(ctx, ctx._returns.guards, choices, required)
    cache = getattr(ctx, '_completion_adapter_routes', None)
    if cache is None:
        cache = ctx._completion_adapter_routes = {}
    key = source, choices
    if not ctx.spend('completion_adapter_queries'):
        return (), ('completion_adapter_budget',)
    if key in cache:
        return cache[key]
    candidates, gaps = packets(ctx, source, choices) if candidates is None else (candidates, ())
    rows = []
    for candidate in candidates:
        if not ctx.spend('completion_adapter_packets'):
            gaps = (*gaps, 'completion_adapter_budget')
            break
        report = completion_continuations(ctx, candidate['source'], choices=choices,
                                          cross_resume=False)
        rows.append({'packet': candidate, 'start': report.get('start'),
                     'choices': choices, 'continuation': report})
        gaps = (*gaps, *report['gaps'])
    if not complete:
        gaps = (*gaps, 'completion_adapter_prefix_unresolved')
    result = tuple(rows), tuple(dict.fromkeys(gaps))
    if not ctx.budget.truncated:
        cache[key] = result
    return result


def terminal_relation(ctx, source, terminal):
    """Current completion versus original completion are separate obligations.

    None means the ordinary legacy path. An absent feasible terminal is not a
    bad copy. A missing continuation proof is unknown, never an override grant.
    """
    if not hasattr(ctx, 'index') or not structured_region(ctx, source):
        return None
    rows, gaps = continuation_rows(ctx, source)
    found = tuple(receipt for row in rows for receipt in row['continuation']['terminals']
                  if receipt['terminal'] == terminal)
    complete = bool(rows) and all(row['continuation']['known_inventory_complete'] for row in rows)
    if not found:
        return {'applicable': False if complete else None, 'match': None,
                'overridden': None, 'receipts': (), 'gaps': gaps}
    checks = tuple(receipt['binding']['match'] for receipt in found)
    return {'applicable': True, 'match': False if False in checks else
            None if None in checks else True,
            'overridden': all(receipt['current_source'] != receipt['root_source'] for receipt in found),
            'receipts': found, 'gaps': gaps}


# Detached public completion queries over one original ProgramIndex.
def completion_continuations(index, source, *, start=None, choices=(), cross_resume=False,
                             budget=4096):
    shared = ValidityBudget(budget)
    ctx = ConnectionContext(index.graph, index=index, budget=shared)
    guards = ctx.selected_edges(choices)
    ctx._returns.guards = guards
    result = _query(ctx, source, start=start, choices=guards, cross_resume=cross_resume)
    return deepcopy({**result, 'budget': {'limit': shared.limit, 'used': shared.used,
        'truncated': shared.truncated, 'costs': dict(shared.costs)},
        'preparation': dict(ctx.preparation)})


# Paid optional evidence over the existing continuation worklist states.
#
# These receipts reference canonical ports and query-local states. They do not
# create a second producer graph or enumerate complete control paths.
class ContinuationReceipts:
    def __init__(self, ctx, source):
        self.ctx, self.source = ctx, source
        self.context = None
        self.transitions, self.terminals = [], []

    def fields(self, expected, choices, crossed):
        return {'root_source': self.source, 'current_source': expected,
                'choices': choices, 'context': self.context,
                'crossed_suspend': crossed}

    def transition(self, following, operation):
        if not self.ctx.spend('completion_receipt_transitions'):
            return False
        self.transitions.append({'source': self.context, 'target': following,
                                 'operation': operation.id})
        return True

    def terminal(self, expected, role, actual, choices, crossed, binding):
        if not self.ctx.spend('completion_receipt_terminals'):
            return
        root = (binding if expected == self.source else
                self.ctx.bind(self.source, actual, choices=choices))
        gaps = tuple(dict.fromkeys((*binding.get('gaps', ()), *root.get('gaps', ()))))
        self.terminals.append({**self.fields(expected, choices, crossed),
            'terminal': HostTerminal(actual.operation), 'actual': actual, 'role': role,
            'binding': binding, 'root_binding': root, 'root_preserved': root['match'],
            'status': 'CONTRADICTED' if binding['match'] is False else
                      'PARTIAL' if binding['match'] is None or gaps else 'DECLARED',
            'gaps': gaps, 'source_runtime_verified': False,
            'runtime_identity_verified': False, 'value_identity_verified': False,
            'transformation_authorized': False})


__all__ = [
    'WitnessSource',
    'DriverObligation',
    'StepWitness',
    'CompletionPorts',
    'IterationSite',
    'ResumeSite',
    'SuspensionSite',
    'DeferredBodySite',
    'structured_region',
    'packets',
    'continuation_rows',
    'terminal_relation',
    'completion_continuations',
    'ContinuationReceipts',
]
