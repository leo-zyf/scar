"""Data materials."""

from dataclasses import dataclass
from enum import Enum
from scar.ir.literals import PythonLiteral
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.program import normal_graph as T
from scar.ir.program.ids import LocationID, OperationID, UseID, ValueID
from scar.ir.program.normal_graph import GlobalPort, CodeScopeRef, Edge
from scar.ir.program.state_addresses import WholeOwner
from scar.ir.v2.common import ProofStatus
from scar.ir.v2.values import EquivalenceClaim, ProvenanceRelation
from scar.ir.program.source_value_routes import ReferenceSlice
from collections import deque
from copy import deepcopy
from scar.ir.program.model import (
    CompletionResult,
    ExitKind,
    InvokeLiteralResult,
    OperationResult,
    TerminatorKind,
)
from scar.ir.program.protocol_flow import ProtocolPort
from scar.ir.program.protocols import OwnedUse
from scar.ir.program.structured import ControlEdgeReference


# Representation graphs for external data; not executed program actions.
class DataNodeKind(str,Enum):
    SCALAR='SCALAR'
    SEQUENCE='SEQUENCE'
    MAPPING='MAPPING'


@dataclass(frozen=True,slots=True)
class DataSpan:
    """Raw UTF-8 byte bounds and zero-based parser-view line/columns."""
    byte_start: int
    byte_end: int
    line: int
    column: int
    end_line: int
    end_column: int


@dataclass(frozen=True,slots=True)
class DataReference:
    ordinal: int
    target: int
    span: DataSpan
    alias: bool
    anchor: str | None


@dataclass(frozen=True,slots=True)
class DataMappingEntry:
    key: int
    value: int


@dataclass(frozen=True,slots=True)
class DataNode:
    ordinal: int
    kind: DataNodeKind
    tag: str
    span: DataSpan
    anchor: str | None
    scalar_text: str | None
    scalar_style: str | None
    flow_style: bool | None
    representation_literal: PythonLiteral | None
    items: tuple[int,...]
    entries: tuple[DataMappingEntry,...]


@dataclass(frozen=True,slots=True)
class DataObligation:
    code: str
    node: int | None
    reference: int | None
    detail: str


@dataclass(frozen=True,slots=True)
class DataParserProfile:
    rule: str
    tools: tuple[ProviderArtifact,...]
    loader_source: ProviderArtifact | None
    loader_function_ast_sha256: str | None
    float_pattern: str | None
    float_flags: int
    float_initial_characters: str
    removed_implicit_tags: tuple[str,...]
    scalar_tags: tuple[str,...]
    text_view: str = 'UTF8_UNIVERSAL_NEWLINES'
    construction_scope: str = 'SCALARS_ONLY_NO_CONTAINER_OR_CUSTOM_CONSTRUCTORS'


@dataclass(frozen=True,slots=True)
class DataMaterial:
    source: ProviderArtifact
    profile: DataParserProfile
    normalized_text_sha256: str
    root: int | None
    nodes: tuple[DataNode,...]
    references: tuple[DataReference,...]
    obligations: tuple[DataObligation,...]
    semantics: str = 'PARSED_REPRESENTATION_NOT_COMPOSED_CONFIGURATION_OR_RUNTIME_VALUE'


# Content routes.
# Lossless factors of existing may-content links, not source operations.
@dataclass(frozen=True, slots=True)
class ContentFactor:
    id: str
    container: T.Endpoint
    key: PythonLiteral | None
    sources: tuple[T.Endpoint, ...]
    targets: tuple[T.Endpoint, ...]
    semantics: str = 'FLOW_INSENSITIVE_MAY_CONTENT'


@dataclass(frozen=True, slots=True)
class ContentIndexCycle:
    container: T.Endpoint
    source: T.Endpoint
    start: int
    stride: int
    path: tuple[T.Endpoint, ...]
    abstract_position: str = 'UNKNOWN_POSITION'
    semantics: str = 'DESCENDING_CYCLE_CONSTRAINT_WITH_MAY_POSITION_WIDENING'


# Static value/content dependencies, separate from logical value versions.
#
# An access reference resolves an existing ContentAccessTemplate owned by one
# source-content provider. No world projection is promoted to a ValueVersionID,
# and these records do not assert aliasing or unchanged object contents.
@dataclass(frozen=True, slots=True)
class ContentAccessRef:
    unit: str
    ordinal: int
    operation: OperationID

    def __post_init__(self):
        if (type(self.unit) is not str or not self.unit or type(self.ordinal) is not int
                or self.ordinal < 0 or type(self.operation) is not OperationID):
            raise ValueError('content lineage needs an original unit/access ordinal/operation')


class ContentLineageRole(str, Enum):
    GET_RESULT = 'GET_RESULT'
    SET_PAYLOAD = 'SET_PAYLOAD'
    UPDATE_INPUT = 'UPDATE_INPUT'
    UPDATE_PRIOR = 'UPDATE_PRIOR'
    DELETE_PRIOR = 'DELETE_PRIOR'


@dataclass(frozen=True, slots=True)
class ContentValueLineage:
    access: ContentAccessRef
    role: ContentLineageRole
    source: GlobalPort | ValueID
    target: GlobalPort | ValueID

    def __post_init__(self):
        if (type(self.access) is not ContentAccessRef or type(self.role) is not ContentLineageRole
                or type(self.source) not in (GlobalPort, ValueID)
                or type(self.target) not in (GlobalPort, ValueID)):
            raise TypeError('content lineage requires typed original endpoints and access reference')
        self.access.__post_init__()
        expected = ((ValueID, GlobalPort) if self.role is ContentLineageRole.GET_RESULT else
            (GlobalPort, ValueID) if self.role in (ContentLineageRole.SET_PAYLOAD,
                ContentLineageRole.UPDATE_INPUT) else (ValueID, ValueID))
        if (type(self.source), type(self.target)) != expected:
            raise TypeError('content lineage endpoint roles differ from the original request')

    @property
    def provenance_relation(self):
        return ProvenanceRelation.UNKNOWN

    @property
    def equivalence(self):
        return EquivalenceClaim.UNKNOWN

    @property
    def logical_version_equality(self):
        return ProofStatus.UNKNOWN


@dataclass(frozen=True, slots=True)
class ContentForwardingObligations:
    """An explicit SET/GET inquiry; no forwarding graph edge is authorized."""
    write: ContentAccessRef
    read: ContentAccessRef
    known_source_facts: tuple[str, ...]
    required_contracts: tuple[str, ...]

    def __post_init__(self):
        if type(self.write) is not ContentAccessRef or type(self.read) is not ContentAccessRef:
            raise TypeError('forwarding obligations require original typed access references')
        for values in (self.known_source_facts, self.required_contracts):
            if type(values) is not tuple or any(type(value) is not str or not value for value in values):
                raise TypeError('forwarding source facts and contracts require explicit strings')
        if not self.required_contracts:
            raise ValueError('an unproved forwarding inquiry must retain its required contracts')

    @property
    def status(self):
        return ProofStatus.UNKNOWN

    @property
    def equivalence(self):
        return EquivalenceClaim.UNKNOWN


# Conditional builtin-slot reference theorems over original source records.
#
# These proofs separate checked source/control facts from required execution
# contracts. Reference forwarding does not establish logical content equality.
# No proof in this profile authorizes an unconditional transformation.
class BuiltinSlotKind(str,Enum):
    DICT='EXACT_BUILTIN_DICT'
    LIST='EXACT_BUILTIN_LIST'


@dataclass(frozen=True,slots=True)
class SourceContainerProducer:
    value: GlobalPort
    site: GlobalPort
    fact: ProviderFactReference
    protocol: str
    kind: BuiltinSlotKind


@dataclass(frozen=True,slots=True)
class BuiltinSlotForwardingGuard:
    kind: BuiltinSlotKind
    producer: GlobalPort
    write_receiver: GlobalPort
    read_receiver: GlobalPort
    selector: PythonLiteral
    contracts: tuple[str,...]

    def __post_init__(self):
        if (type(self.kind)is not BuiltinSlotKind or not self.contracts
                or any(type(name)is not str or not name for name in self.contracts)):
            raise ValueError('builtin reference forwarding needs its explicit execution contracts')


@dataclass(frozen=True,slots=True)
class SourceForwardingControlWindow:
    source_function: GlobalPort
    function: ProviderFactReference
    path: tuple[Edge,...]
    write_exit_index: int
    local_read_sites: tuple[GlobalPort,...]
    local_read_facts: tuple[ProviderFactReference,...]
    local_read_materials: tuple[ProviderArtifact,...]
    local_read_contracts: tuple[str,...]
    guarded_cleanup_sites: tuple[GlobalPort,...]=()


@dataclass(frozen=True,slots=True)
class ContentReferenceForwardingProof:
    write: ContentAccessRef
    read: ContentAccessRef
    payload: GlobalPort
    result: GlobalPort
    producer: SourceContainerProducer
    selector: PythonLiteral
    receiver_paths: tuple[ReferenceSlice,...]
    selector_paths: tuple[ReferenceSlice,...]
    control: SourceForwardingControlWindow
    guard: BuiltinSlotForwardingGuard
    static_facts: tuple[str,...]=()
    runtime_checks: tuple[str,...]=()
    external_assumptions: tuple[str,...]=()
    rule: str='BUILTIN_SCALAR_SLOT_SET_THEN_GET_REFERENCE_THEOREM_V1'

    @property
    def claim(self):return 'REFERENCE_FORWARDING_PROVEN_UNDER_EXPLICIT_UNDISCHARGED_CONTRACTS'

    @property
    def unconditional_status(self):return ProofStatus.UNKNOWN

    @property
    def logical_equivalence(self):return EquivalenceClaim.UNKNOWN

    @property
    def transformation_authorized(self):return False


@dataclass(frozen=True,slots=True)
class ContentForwardingRefusal:
    write: ContentAccessRef
    read: ContentAccessRef
    reason: str
    site: GlobalPort | None=None


@dataclass(frozen=True,slots=True)
class ContentForwardingDiscoveryBoundary:
    access: ContentAccessRef
    reason: str
    site: GlobalPort | None=None


# Conditional source selector paths, never runtime addresses or content equality.
class AddressRootKind(str, Enum):
    FUNCTION_PARAMETER = 'FUNCTION_PARAMETER'
    CALL_RESULT = 'CALL_RESULT'
    SOURCE_LITERAL = 'SOURCE_LITERAL'
    CLASS_DEFINITION = 'CLASS_DEFINITION'
    FUNCTION_DEFINITION = 'FUNCTION_DEFINITION'
    OPEN_PRODUCER = 'OPEN_PRODUCER'


@dataclass(frozen=True, slots=True)
class FieldSelector:
    name: str


@dataclass(frozen=True, slots=True)
class ItemSelector:
    key: PythonLiteral | GlobalPort


AddressSelector = FieldSelector | ItemSelector


@dataclass(frozen=True, slots=True)
class AddressPath:
    root: GlobalPort
    root_kind: AddressRootKind
    selectors: tuple[AddressSelector, ...]
    evidence: tuple[ProviderFactReference, ...]
    conditions: tuple[str, ...]
    frontier: str | None = None
    semantics: str = 'CONDITIONAL_SOURCE_SELECTOR_PATH_NOT_RUNTIME_ADDRESS_OR_CONTENT_EQUALITY'


# Source content requests and symbolic state at world boundaries.
#
# An address is an expression over reference occurrences, not an ObjectID or
# physical slot. Projected state values can be equal; distinct IDs do not prove
# a mutation or different LogicalVersions. Protocol side effects stay open.
@dataclass(frozen=True, slots=True)
class SliceBoundsSelector:
    """Original bound references; no slice object or numeric bounds assumed."""
    lower: GlobalPort
    upper: GlobalPort


ContentSelector = FieldSelector | ItemSelector | SliceBoundsSelector | WholeOwner


class ContentRequestKind(str, Enum):
    GET = 'GET'
    SET = 'SET'
    DELETE = 'DELETE'
    UPDATE = 'UPDATE'


class ContentInstanceParameter(str, Enum):
    ACTIVATION = 'ACTIVATION'
    OWNER_REFERENCE_OCCURRENCE = 'OWNER_REFERENCE_OCCURRENCE'
    SELECTOR_REFERENCE_OCCURRENCES = 'SELECTOR_REFERENCE_OCCURRENCES'


@dataclass(frozen=True, slots=True)
class ContentAddressTemplate:
    id: LocationID
    receiver: GlobalPort
    selector: ContentSelector
    instance_parameters: tuple[ContentInstanceParameter, ...]
    semantics: str = 'SOURCE_PROTOCOL_REQUEST_ADDRESS_NOT_PHYSICAL_SLOT'


@dataclass(frozen=True, slots=True)
class ContentStateProjection:
    """The abstract state project(world, selected address instance).

    This is a symbolic state variable, never a captured tensor, a write count,
    or an established LogicalVersion. A GET may run a mutating descriptor.
    """
    id: ValueID
    address: LocationID
    world: GlobalPort
    semantics: str = 'SYMBOLIC_CONTENT_AT_WORLD_NOT_LOGICAL_VERSION_EQUALITY'


@dataclass(frozen=True, slots=True)
class ContentAccessTemplate:
    """SET payload is the assigned reference; UPDATE payload is an input.

    An UPDATE input is never asserted equal to the resulting whole contents.
    Original instructions retain the append/extend/merge protocol distinction.
    """
    operation: OperationID
    site: GlobalPort
    scope: CodeScopeRef
    kind: ContentRequestKind
    address: LocationID
    receiver_use: UseID
    payload: GlobalPort | None
    result: GlobalPort | None
    world_before: GlobalPort
    world_after: GlobalPort
    state_before: ValueID
    state_after: ValueID
    attribute_fact: int | None
    effects_closed: bool = False


# Evaluation-relative copy flow over the original CFG, shared by projections.
def selected(ctx, *groups):
    return ctx.selected_edges(tuple(edge for group in groups for edge in ctx.selected_edges(group)))


def required_choices(ctx, source):
    """Original edges necessary to reach this source evaluation point.

    This is conditional on the source being evaluated, not a prediction that
    it will be evaluated. An edge cut must make that original point unreachable.
    """
    port = ctx.port(source)
    if port is None or port.operation is None or port.gap is not None:
        return (), False
    operation = ctx.operations[port.operation]
    point = (operation.terminator.successors[port.edge].block if port.phase == 'edge'
             else ctx.block_of[port.operation])
    key = ctx.region_of[port.operation], point
    if not hasattr(ctx, '_await_required_source_edges'):
        ctx._await_required_source_edges = {}
    if key not in ctx._await_required_source_edges:
        region, outgoing, incoming, reached, _ = ctx._edges(key[0])
        pending, ancestors = [point], set()
        while pending:
            if not ctx.spend('await_source_ancestor_blocks'):
                return (), False
            block = pending.pop()
            if block in ancestors:
                continue
            ancestors.add(block)
            for parent, _ in incoming[block]:
                if not ctx.spend('await_source_ancestor_edges'):
                    return (), False
                pending.append(parent)
        needed = []
        for block in sorted(ancestors, key=lambda item: item.wire):
            if not ctx.spend('await_source_control_candidates'):
                return (), False
            edges = outgoing[block]
            relevant = tuple(edge for edge in edges if edge[2] in ancestors)
            if len(edges) < 2 or len(relevant) != 1:
                continue
            chosen = relevant[0]
            todo, seen = [region.entry], set()
            while todo:
                if not ctx.spend('await_source_control_cut_blocks'):
                    return (), False
                current = todo.pop()
                if current in seen:
                    continue
                seen.add(current)
                for edge in outgoing[current]:
                    if not ctx.spend('await_source_control_cut_edges'):
                        return (), False
                    if edge[:2] != chosen[:2]:
                        todo.append(edge[2])
            if point in reached and point not in seen:
                needed.append(ControlEdgeReference(chosen[0], chosen[1]))
        ctx._await_required_source_edges[key] = tuple(needed), point in reached
    edges, complete = ctx._await_required_source_edges[key]
    if port.phase == 'edge':
        edges = selected(ctx, edges, ControlEdgeReference(port.operation, port.edge))
    return edges, complete


def flow(ctx, source, choices=(), *, mode='must'):
    port = ctx.port(source)
    if port is None or port.gap is not None or port.operation is None:
        return None
    previous = ctx._returns.guards
    try:
        required, complete = required_choices(ctx, source)
        if not complete:
            return None
        try:
            ctx._returns.guards = selected(ctx, previous, choices, required)
        except ValueError:
            if not hasattr(ctx, '_await_flow_choice_conflicts'):
                ctx._await_flow_choice_conflicts = set()
            ctx._await_flow_choice_conflicts.add((source, ctx.selected_edges(choices), previous))
            return {}
        local = ctx.selected_edges(ctx._returns.guards, region=ctx.region_of[port.operation])
        return ctx._fresh_copies(port, guard=local or None, mode=mode)
    finally:
        ctx._returns.guards = previous


def inventory_flow(ctx, source, choices=()):
    """Conditional may inventory for independently proved multi-visit guards.

    Ordinary must flow is untouched. The relaxed choices enumerate candidate
    Uses only; the original context and prefix remain separate receipts.
    """
    previous = ctx._returns.guards
    key = source, ctx.selected_edges(choices), previous
    originals = ctx.selected_edges((*previous, *choices))
    scopes = getattr(ctx, '_copy_inventory_scopes', None)
    if scopes is None:
        scopes = ctx._copy_inventory_scopes = {}
    port = ctx._ports.get(source)
    scope_key = (None if port is None else
                 (port.operation, port.phase, port.edge, originals))
    before = flow(ctx, source, choices, mode='may')
    result = {'before': before, 'guarded_flow': before, 'conditional': False,
              'choices': choices, 'required_prefix': (), 'relaxed_choices': (), 'gaps': ()}

    def unfinished():
        # Only a previously established source-point/control receipt may
        # survive an unpaid value-slot join. No flow or bounds result is
        # cached/replayed here; ordinary flow retains its existing hooks.
        known = scopes.get(scope_key)
        if ctx.budget.truncated and known is not None:
            return {**result, **known, 'before': None, 'conditional': True,
                'pending': True, 'gaps': ('copy_inventory_loop_context_unproved',
                                        'copy_inventory_context_join_budget')}
        return result

    conflict = key in getattr(ctx, '_await_flow_choice_conflicts', ())
    if ctx.budget.truncated:
        return unfinished()
    port = ctx.port(source)
    if port is not None:
        scope_key = port.operation, port.phase, port.edge, originals
    required, complete = required_choices(ctx, source)
    if port is None or not complete or ctx.budget.truncated:
        return unfinished()
    absent_source = before is not None and port.operation not in before
    if before is not None and not conflict and not absent_source:
        return result

    def cyclic(edge):
        memo = getattr(ctx, '_copy_inventory_cycles', None)
        if memo is None:
            memo = ctx._copy_inventory_cycles = {}
        if edge in memo:
            return memo[edge]
        region = ctx.region_of[edge.operation]
        _, outgoing, _, _, _ = ctx._edges(region)
        start = ctx.operations[edge.operation].terminator.successors[edge.successor_index].block
        target = ctx.block_of[edge.operation]
        pending, seen = [start], set()
        while pending:
            if not ctx.spend('copy_inventory_cycle_blocks'):
                return None
            block = pending.pop()
            if block == target:
                memo[edge] = True
                return True
            if block in seen:
                continue
            seen.add(block)
            for original in outgoing[block]:
                if not ctx.spend('copy_inventory_cycle_edges'):
                    return None
                pending.append(original[2])
        memo[edge] = False
        return False

    own = ControlEdgeReference(port.operation, port.edge) if port.phase == 'edge' else None
    if own is not None and any(edge.operation == own.operation and edge != own for edge in originals):
        return result
    if absent_source and not conflict:
        reason = any(edge != own and cyclic(edge) is True for edge in originals)
    elif conflict:
        by_operation, conflicting = {}, []
        for prefix in required:
            if not ctx.spend('copy_inventory_required_choices'):
                return unfinished()
            by_operation[prefix.operation] = prefix
        for edge in originals:
            if not ctx.spend('copy_inventory_conflicting_choices'):
                return unfinished()
            prefix = by_operation.get(edge.operation)
            if prefix is not None and prefix != edge:
                conflicting.append(edge)
        reason = bool(conflicting) and all(cyclic(edge) is True for edge in conflicting)
    else:
        combined = selected(ctx, originals, required)
        local = ctx.selected_edges(combined, region=ctx.region_of[port.operation])
        reason = ctx._guard_entry_ambiguity(port, local) == 'selected_loop_entry_requires_occurrence_context'
    if not reason or ctx.budget.truncated:
        return unfinished()
    relaxed = []
    for edge in originals:
        if not ctx.spend('copy_inventory_guard_candidates'):
            break
        repeated = cyclic(edge)
        if repeated is None:
            break
        if edge == own or not repeated:
            relaxed.append(edge)
    if ctx.budget.truncated:
        return unfinished()
    try:
        ctx._returns.guards = ()
        candidate = flow(ctx, source, tuple(relaxed), mode='may')
    finally:
        ctx._returns.guards = previous
    if ctx.budget.truncated:
        return unfinished()
    if candidate is not None and port.operation not in candidate:
        return result
    if candidate is not None and not ctx.budget.truncated:
        scopes[scope_key] = {'required_prefix': required, 'relaxed_choices': tuple(relaxed)}
    result = {**result, 'before': candidate, 'conditional': True,
        'required_prefix': required, 'relaxed_choices': tuple(relaxed),
        'gaps': ('copy_inventory_loop_context_unproved',)}
    return result


# Call-context projections of original copy-dependent Uses.
#
# Frames are static call paths, never invocations or alternative producers.
# This query does not treat computations, packet projections or state writes as
# identity edges. All derived work shares the caller's ConnectionContext budget.
def _flags():
    return dict(source_runtime_verified=False, runtime_identity_verified=False,
                value_identity_verified=False, transformation_authorized=False)


def _status(may, must):
    return ('NOT_CHECKED' if may is None or must is None else
            'DECLARED' if must else 'CONDITIONAL' if may else 'NOT_APPLICABLE')


def _binding(may, must):
    return {'match': must, 'may': may, 'routes': (),
            'gaps': ('copy_consumer_membership_unresolved',)
                    if may is None or must is None else (), **_flags()}


def prepare_index(index):
    """Once per validated ProgramIndex; original anchors, no return joins.

    Templates are private and never mutated. Public reports detach them. This
    preparation and raw-anchor presentation cost is separate from derived work.
    """
    calls, arguments, all_arguments = {r: [] for r in index.regions}, {}, {}
    for operation in index.graph.operations:
        term = operation.terminator
        invoke = term is not None and term.kind is TerminatorKind.INVOKE
        if operation.callee is None and not invoke:
            all_arguments[operation.id] = tuple({'ordinal': ordinal,
                'actual': OwnedUse(operation.id, use.id), 'formal': None,
                'may': None, 'must': None, 'binding': _binding(None, None)}
                for ordinal, use in enumerate(operation.uses))
            continue
        calls[index._blocks[index._operation_blocks[operation.id]]].append(operation)
        actuals = operation.invoke_arguments if invoke else operation.call_arguments
        entry = (None if operation.callee is None else
                 index._block_records[index.regions[operation.callee].entry])
        arguments[operation.id] = tuple({'ordinal': ordinal,
            'actual': OwnedUse(operation.id, identity),
            'formal': None if entry is None else index.values[entry.arguments[ordinal]].origin,
            'may': None, 'must': None, 'binding': _binding(None, None)}
            for ordinal, identity in enumerate(actuals))
        all_arguments[operation.id] = arguments[operation.id]
    index._copy_consumer_originals = ({region: tuple(rows) for region, rows in calls.items()}, arguments)
    index._operation_argument_templates = all_arguments
    index._copy_consumer_preparation = {
        'copy_consumer_original_calls': sum(map(len, calls.values())),
        'copy_consumer_original_arguments': sum(map(len, arguments.values())),
        'copy_consumer_index_operations': len(index.operations),
        'operation_original_argument_templates': sum(map(len, all_arguments.values())),
        'copy_consumer_return_joins_precomputed': 0}


def _prepare(ctx):
    ctx.preparation.update(ctx.index._copy_consumer_preparation)
    ctx.preparation['copy_consumer_program_index_reused'] = True
    ctx.preparation['copy_consumer_prepared_value_use_index_reused'] = True
    return ctx.index._copy_consumer_originals


def _return_rows(ctx, parent_call, may, must):
    """Derive interfaces only for this frame and stop at the first failed charge."""
    if parent_call is None:
        return
    call = ctx.operations[parent_call]
    term = call.terminator
    invoke = term is not None and term.kind is TerminatorKind.INVOKE
    for exit_ in ctx.regions[call.callee].exits:
        if not ctx.spend('copy_consumer_return_exits'):
            return
        for slot in range(len(exit_.types)):
            if not ctx.spend('copy_consumer_return_slots'):
                return
            targets = []
            if not invoke and exit_.kind is ExitKind.NORMAL:
                targets.append((OperationResult(call.id, slot), None))
            elif invoke:
                for number, edge in enumerate(term.successors):
                    if not ctx.spend('copy_consumer_return_edges'):
                        return
                    if edge.kind is not exit_.kind:
                        continue
                    for position, actual in enumerate(edge.arguments):
                        if not ctx.spend('copy_consumer_return_positions'):
                            return
                        if type(actual) is CompletionResult and actual.index == slot:
                            targets.append((InvokeLiteralResult(
                                ControlEdgeReference(call.id, number), actual), position))
            if not targets:
                continue
            for site in ctx.index._exit_ports[(call.callee, exit_.name, slot)]:
                for target, position in targets:
                    if not ctx.spend('copy_consumer_returns'):
                        return
                    possible = _membership(ctx, may, site.operation, site.use.value, 'may')
                    definite = _membership(ctx, must, site.operation, site.use.value, 'must')
                    binding = _binding(possible, definite)
                    yield dict(kind=exit_.kind, slot=slot,
                        actual=OwnedUse(site.operation, site.use.id),
                        caller_result=target, landing_position=position,
                        may=possible, must=definite, binding=binding,
                        status=_status(possible, definite), gaps=binding['gaps'])


def _membership(ctx, before, operation, value, mode):
    if before is None:
        return None
    if value in before.get(operation, ()):
        return True
    return None if any(current == mode for current, _ in ctx._copy_consumer_bound_events) else False


def consumer_graph(ctx, source, choices=()):
    """Internal projection; cached only within this shared query context."""
    choices = selected(ctx, ctx._returns.guards, choices)
    cache = getattr(ctx, '_copy_consumer_graphs', None)
    if cache is None:
        cache = ctx._copy_consumer_graphs = {}
    cache_key = source, choices
    if cache_key in cache:
        return cache[cache_key]
    original_calls, original_arguments = _prepare(ctx)
    port = ctx._ports.get(source) or ctx.port(source)
    operation = None if port is None else port.operation
    if operation is None:
        operation = getattr(source, 'operation', None)
    if operation not in ctx.region_of:
        # A budget can stop resolution but must not select an unrelated root.
        result = dict(source=source, choices=choices, frames=(), status='PARTIAL',
            gaps=('copy_consumer_source_unresolved',), cursor={'phase': 'source'},
            inventory=dict(enumerated_frames=0, checked_frames=0, frames_total=None,
                enumerated_consumers=0, consumers_total=None,
                known_inventory_complete=False, proofs_complete=False, cursor={'phase': 'source'}),
            budget=_budget(ctx), preparation=dict(ctx.preparation), **_flags())
        cache[cache_key] = result
        return result
    root_region = ctx.region_of[operation]
    pending = deque([((), root_region, None, (), (), (), (root_region,))])
    frames, all_gaps, cursor = [], [], None
    inventory_complete, proofs_complete, checked = True, True, 0
    conditional_inventory = False
    previous = ctx._returns.guards
    previous_bounds = getattr(ctx, '_copy_consumer_bounds', False)
    previous_events = getattr(ctx, '_copy_consumer_bound_events', [])
    ctx._copy_consumer_bounds = True
    ctx._copy_consumer_bound_events = []
    ctx._returns.guards = choices
    try:
        while pending:
            key, region, parent_call, inputs, may_seeds, must_seeds, ancestry = pending.popleft()
            local, boundaries, consumers, call_rows, return_rows = [], [], [], [], []
            may = must = None
            # Keep root and raw call anchors even when the first charge fails.
            if ctx.spend('copy_consumer_frames', len(key) + 1):
                if not key:
                    membership = inventory_flow(ctx, source, choices)
                    may = membership['before']
                    conditional_inventory = membership['conditional']
                    must = None if conditional_inventory else flow(ctx, source, choices, mode='must')
                    if conditional_inventory:
                        local.extend(membership['gaps'])
                        inventory_complete = proofs_complete = False
                    may_seeds = () if port is None else port.values
                    must_seeds = may_seeds
                else:
                    entry = ctx.blocks[ctx.regions[region].entry]
                    guard = ctx.selected_edges(choices, region=region)
                    may = ctx._fresh_copies(ProtocolPort(None, entry.operations[0],
                        may_seeds, 'before'), guard=guard or None, mode='may')
                    must = None if conditional_inventory else ctx._fresh_copies(
                        ProtocolPort(None, entry.operations[0], must_seeds, 'before'),
                        guard=guard or None, mode='must')
            if may is None or must is None:
                local.append('copy_consumer_flow_unresolved')
                proofs_complete = False
            if may is None:
                inventory_complete = False
            if 'call_return_guard_context_unresolved' in ctx._returns.gaps:
                local.append('copy_consumer_guard_context_unresolved')
                inventory_complete = proofs_complete = False
            if may is not None:
                seen = set()
                from scar.ir.program.protocol_connection_validation import _consumer_role
                for at, values in may.items():
                    if not ctx.spend('copy_consumer_positions', 1 + len(values)):
                        break
                    if values and not ctx.operations[at].effects_complete:
                        local.append('copy_consumer_effects_incomplete')
                        inventory_complete = False
                    for value in sorted(values, key=lambda item: item.wire):
                        for site in ctx.index._consumers[value]:
                            if not ctx.spend('copy_consumer_indexed_uses'):
                                break
                            if site.operation != at or site.use.id in seen:
                                continue
                            seen.add(site.use.id)
                            owner = ctx.operations[at]
                            if not _consumer_role(owner, site.use.id):
                                continue
                            actual = OwnedUse(at, site.use.id)
                            definite = _membership(ctx, must, at, value, 'must')
                            term = owner.terminator
                            terminal = term is not None and term.kind in (
                                TerminatorKind.RETURN, TerminatorKind.RAISE)
                            call = owner.callee is not None or term is not None and term.kind is TerminatorKind.INVOKE
                            kind = (next(exit_.kind for exit_ in ctx.regions[region].exits
                                if exit_.name == term.exit) if terminal else None)
                            slot = term.values.index(site.use.id) if terminal else None
                            binding = _binding(True, definite)
                            consumers.append(dict(key=(key, actual), actual=actual,
                                role='EXIT' if terminal else 'CALL_ARGUMENT' if call else 'OPERATION',
                                kind=kind, slot=slot, may=True, must=definite, binding=binding,
                                status=_status(True, definite), gaps=binding['gaps']))
                            if not owner.effects_complete:
                                local.append('copy_consumer_effects_incomplete')
                                inventory_complete = False
                        if ctx.budget.truncated:
                            break
                    if ctx.budget.truncated:
                        break
            for call in original_calls[region]:
                args, checked_arguments = [], True
                for raw in original_arguments[call.id]:
                    if not ctx.spend('copy_consumer_arguments'):
                        # Preserve the prepared complete unknown tuple without
                        # walking/rebuilding its unexamined suffix.
                        args = original_arguments[call.id]
                        checked_arguments = False
                        break
                    use = ctx.owned_use(call.id, raw['actual'].use)
                    possible = _membership(ctx, may, call.id, use.value, 'may')
                    definite = _membership(ctx, must, call.id, use.value, 'must')
                    args.append({**raw, 'may': possible, 'must': definite,
                                 'binding': _binding(possible, definite)})
                args = tuple(args)
                affected = (True if any(row['may'] is True for row in args) else
                            None if not checked_arguments or any(row['may'] is None for row in args) else False)
                call_gaps, child = [], None
                if affected is not False:
                    if call.callee is None:
                        call_gaps.append('copy_consumer_native_boundary_unresolved')
                    elif call.callee in ancestry:
                        call_gaps.append('copy_consumer_recursion_unresolved')
                    elif (ctx.selected_edges(choices, region=call.callee)
                          and len(ctx._returns.callers[call.callee]) > 1):
                        call_gaps.append('copy_consumer_guard_context_unresolved')
                    elif affected is True and ctx.spend('copy_consumer_call_paths', len(key) + 1):
                        child = (*key, call.id)
                        entry = ctx.blocks[ctx.regions[call.callee].entry]
                        child_may = tuple(entry.arguments[row['ordinal']] for row in args if row['may'] is True)
                        child_must = tuple(entry.arguments[row['ordinal']] for row in args if row['must'] is True)
                        pending.append((child, call.callee, call.id, args, child_may,
                                        child_must, (*ancestry, call.callee)))
                    else:
                        call_gaps.append('copy_consumer_call_membership_unresolved')
                if call_gaps:
                    local.extend(call_gaps)
                    inventory_complete = proofs_complete = False
                call_rows.append(dict(key=(key, call.id), operation=call.id, callee=call.callee,
                    arguments=args, child_frame=child,
                    status='NOT_APPLICABLE' if affected is False else
                           'PARTIAL' if call_gaps or child is None else 'DECLARED',
                    gaps=tuple(call_gaps)))
            return_rows.extend(_return_rows(ctx, parent_call, may, must))
            if not key and ctx._returns.callers.get(region) and (may is None or any(
                    row['role'] == 'EXIT' for row in consumers) or any(
                    mode == 'may' for mode, _ in ctx._copy_consumer_bound_events)):
                local.append('copy_consumer_root_caller_context_unresolved')
                inventory_complete = proofs_complete = False
                boundaries.extend(dict(kind='ROOT_CALLER_CONTEXT', operation=call.id,
                    callee=region, status='PARTIAL',
                    gaps=('copy_consumer_root_caller_context_unresolved',))
                    for call in ctx._returns.callers[region])
            if ctx.budget.truncated:
                local.append('copy_consumer_budget')
                inventory_complete = proofs_complete = False
                cursor = cursor or {'phase': 'copy_consumer_frames', 'frame': key}
            if ctx._copy_consumer_bound_events:
                local.append('copy_consumer_partial_bounds')
                proofs_complete = False
                if any(mode == 'may' for mode, _ in ctx._copy_consumer_bound_events):
                    inventory_complete = False
            checked += int(not local)
            frames.append(dict(key=key, region=region, parent_call=parent_call,
                may_seeds=tuple(may_seeds), must_seeds=tuple(must_seeds), inputs=inputs,
                consumers=tuple(consumers), calls=tuple(call_rows), returns=tuple(return_rows),
                boundaries=tuple(boundaries), status='PARTIAL' if local else 'DECLARED',
                gaps=tuple(dict.fromkeys(local))))
            all_gaps.extend(local)
    finally:
        ctx._returns.guards = previous
        ctx._copy_consumer_bounds = previous_bounds
        ctx._copy_consumer_bound_events = previous_events
    count = sum(len(frame['consumers']) for frame in frames)
    ctx.preparation['copy_consumer_raw_call_anchors_presented'] = sum(
        len(frame['calls']) for frame in frames)
    result = dict(source=source, choices=choices, frames=tuple(frames),
        inventory=dict(enumerated_frames=len(frames), checked_frames=checked,
            frames_total=len(frames) if inventory_complete else None,
            enumerated_consumers=count, consumers_total=count if inventory_complete else None,
            known_inventory_complete=inventory_complete, proofs_complete=proofs_complete,
            cursor=cursor), status='PARTIAL' if all_gaps else 'DECLARED',
        gaps=tuple(dict.fromkeys(all_gaps)), cursor=cursor, budget=_budget(ctx),
        preparation=dict(ctx.preparation), **_flags())
    cache[cache_key] = result
    return result


def _budget(ctx):
    return dict(limit=ctx.budget.limit, used=ctx.budget.used,
                truncated=ctx.budget.truncated, costs=dict(ctx.budget.costs))


def copy_consumers(index, source, *, guard=None, budget=4096):
    from scar.ir.execution.validity_budget import ValidityBudget
    from scar.ir.program.protocol_connection_validation import ConnectionContext
    ctx = ConnectionContext(index.graph, index=index, budget=ValidityBudget(budget))
    return deepcopy(consumer_graph(ctx, source, ctx.selected_edges(guard)))


__all__ = [
    'ContentAccessRef',
    'ContentLineageRole',
    'ContentValueLineage',
    'ContentForwardingObligations',
    'BuiltinSlotKind',
    'SourceContainerProducer',
    'BuiltinSlotForwardingGuard',
    'SourceForwardingControlWindow',
    'ContentReferenceForwardingProof',
    'ContentForwardingRefusal',
    'ContentForwardingDiscoveryBoundary',
    'ContentFactor',
    'ContentIndexCycle',
    'SliceBoundsSelector',
    'ContentSelector',
    'ContentRequestKind',
    'ContentInstanceParameter',
    'ContentAddressTemplate',
    'ContentStateProjection',
    'ContentAccessTemplate',
    'AddressRootKind',
    'FieldSelector',
    'ItemSelector',
    'AddressSelector',
    'AddressPath',
    'DataNodeKind',
    'DataSpan',
    'DataReference',
    'DataMappingEntry',
    'DataNode',
    'DataObligation',
    'DataParserProfile',
    'DataMaterial',
    'prepare_index',
    'consumer_graph',
    'copy_consumers',
    'selected',
    'required_choices',
    'flow',
    'inventory_flow',
]
