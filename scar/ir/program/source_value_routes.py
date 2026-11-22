"""Source value routes."""

from dataclasses import dataclass
from scar.ir.program import source_productions as P, normal_graph as G, normal_links as F
from scar.ir.program.normal_targets import (
    GlobalPort,
    Body,
    Endpoint,
    AttributeSite,
    FunctionConnection,
)
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.program.signature_patterns import GroundType
from enum import Enum
import re
from scar.ir.program.call_edges import InvocationKey, CallRecordAnchor
from scar.ir.program.ids import ProviderID
from scar.ir.program.view import SourceAnchor
from scar.ir.program.source_productions import ProductionRegionID, ProductionStepID


# Source operations and value transport, without object/content equivalence.
#
# Operation interfaces reference the original F4 instruction language. Lexical
# locations factor possible writes/reads instead of materializing their product.
# All identifiers are qualified by the body's versioned source material.
@dataclass(frozen=True, slots=True)
class TypedValuePort:
    port: G.Port
    value_type: GroundType


@dataclass(frozen=True, slots=True)
class SourceOperation:
    entry: G.Port
    instruction: P.ProductionInstruction
    inputs: tuple[TypedValuePort, ...]
    outputs: tuple[TypedValuePort, ...]


@dataclass(frozen=True, slots=True)
class LocationAccess:
    value: G.Port
    fact: ProviderFactReference


@dataclass(frozen=True, slots=True)
class LexicalLocation:
    scope: F.CodeScopeRef
    namespace: F.Namespace
    writes: tuple[LocationAccess, ...]
    reads: tuple[LocationAccess, ...]


@dataclass(frozen=True, slots=True)
class NameBoundary:
    scope: F.CodeScopeRef
    namespace: F.Namespace
    access: LocationAccess
    write: bool
    reason: str


@dataclass(frozen=True, slots=True)
class SourceBodyCall:
    fact: ProviderFactReference
    call: F.Call


@dataclass(frozen=True, slots=True)
class SourceBodyValueFlow:
    source_function: GlobalPort
    body: Body
    source_templates: ProviderArtifact
    edges: tuple[G.Edge, ...]
    operations: tuple[SourceOperation, ...]
    locations: tuple[LexicalLocation, ...]
    calls: tuple[SourceBodyCall, ...]
    name_boundaries: tuple[NameBoundary, ...] = ()


@dataclass(frozen=True, slots=True)
class DependentCallInput:
    call: ProviderFactReference
    role: str
    ordinal: int
    keyword: str | None
    actual: GlobalPort


@dataclass(frozen=True, slots=True)
class SourceDataReach:
    entry: int
    body: int
    formal: GlobalPort
    reached_ports: int
    reached_operations: tuple[int, ...]
    call_inputs: tuple[DependentCallInput, ...]
    conditions: tuple[str, ...] = (
        'PARENT_ARGUMENT_AND_DECLARED_BODY_CONTEXT_SELECTED',
        'FLOW_INSENSITIVE_SYNTACTIC_MAY_DEPENDENCY',
        'BRANCH_AND_CALL_RETURN_LABELS_RETAINED_NOT_SOLVED_BY_THIS_QUERY',
        'LEXICAL_LOCATION_IS_NOT_A_SELECTED_REACHING_VERSION',
        'OPERATION_OUTPUT_CAN_DEPEND_ON_ANY_DATA_INPUT',
        'WORLD_IS_NOT_PROPAGATED_AS_DATA_DEPENDENCY',
        'NO_ALIAS_CONTENT_IDENTITY_PURITY_OR_RUNTIME_DISPATCH_PROOF',
    )


# Demand graph for declared reference flow; never tensor/content equality.
class ReferenceTransitionKind(str, Enum):
    DECLARED_ALIAS = 'DECLARED_ALIAS'
    LEXICAL_MAY_BINDING = 'LEXICAL_MAY_BINDING'
    ARGUMENT = 'ARGUMENT'
    INITIAL_DEFAULT = 'INITIAL_DEFAULT'
    DATA_RETURN_ENTER_CALL = 'DATA_RETURN_ENTER_CALL'
    CONSTRUCTOR_DATA_FROM_NEW_RESULT = 'CONSTRUCTOR_DATA_FROM_NEW_RESULT'


class ReferenceOriginKind(str, Enum):
    ALLOCATION_SITE_CANDIDATE = 'ALLOCATION_SITE_CANDIDATE'
    SOURCE_LITERAL = 'SOURCE_LITERAL'
    FUNCTION_DEFINITION = 'FUNCTION_DEFINITION'
    CLASS_DEFINITION = 'CLASS_DEFINITION'


@dataclass(frozen=True, slots=True)
class ReferenceState:
    port: Endpoint
    calls: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ReferenceTransition:
    downstream: int
    upstream: int
    kind: ReferenceTransitionKind
    evidence: tuple[ProviderFactReference, ...]
    conditions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReferenceOrigin:
    state: int
    kind: ReferenceOriginKind
    source_class: GlobalPort | None
    evidence: tuple[ProviderFactReference, ...]
    conditions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReferenceFrontier:
    state: int
    reason: str


@dataclass(frozen=True, slots=True)
class ReferenceSlice:
    states: tuple[ReferenceState, ...]
    transitions: tuple[ReferenceTransition, ...]
    origins: tuple[ReferenceOrigin, ...]
    frontiers: tuple[ReferenceFrontier, ...]
    semantics: str = 'CONDITIONAL_DECLARED_REFERENCE_FLOW_NOT_CONTENT_OR_RUNTIME_IDENTITY'


@dataclass(frozen=True, slots=True)
class ConditionalFieldFlow:
    write: ProviderFactReference
    read: ProviderFactReference
    write_site: AttributeSite
    read_site: AttributeSite
    shared_allocation_candidate: Endpoint
    receiver_class: GlobalPort
    write_slice: int
    read_slice: int
    write_origin: int
    read_origin: int
    conditions: tuple[str, ...]
    semantics: str = 'MAY_REACHING_FIELD_STORE_NOT_SELECTED_STATE_VERSION'


@dataclass(frozen=True, slots=True)
class DeclaredMethodInterface:
    call: ProviderFactReference
    attribute: ProviderFactReference
    class_binding: ProviderFactReference
    field_flow: int
    value_slice: int
    value_origin: int
    member_slice: int
    member_origin: int
    source_class: GlobalPort
    connection: FunctionConnection
    conditions: tuple[str, ...]
    semantics: str = 'CONDITIONAL_DECLARED_INTERFACE_NOT_ENTRY_ADMISSION_OR_RUNTIME_DISPATCH'


@dataclass(frozen=True, slots=True)
class FieldInterfaceQuery:
    unit: str
    call_ordinal: int
    method_attribute_ordinal: int
    field_attribute_ordinal: int
    slices: tuple[ReferenceSlice, ...]
    flows: tuple[ConditionalFieldFlow, ...]
    methods: tuple[DeclaredMethodInterface, ...]
    writer_units: tuple[str, ...]
    obligations: tuple[str, ...]


# Actual call scopes over original shared source ports, never content versions.
@dataclass(frozen=True, slots=True)
class SourceCallScope:
    run_scope: str
    invocation: InvocationKey

    def __post_init__(self):
        if (type(self.run_scope) is not str or re.fullmatch(r'[0-9a-f]{64}',self.run_scope) is None
                or type(self.invocation) is not InvocationKey):
            raise ValueError('actual source call scope requires its original run and InvocationKey')
        self.invocation.__post_init__()


@dataclass(frozen=True, slots=True)
class SourceInvocationPort:
    call_scope: SourceCallScope
    original: GlobalPort
    # A caller may evaluate one source CALL repeatedly within one invocation.
    # Its original attempt anchor qualifies that boundary occurrence, not a
    # logical content version or an unrolled copy of the caller's body.
    boundary: CallRecordAnchor | None = None

    def __post_init__(self):
        if type(self.call_scope) is not SourceCallScope or type(self.original) is not GlobalPort:
            raise ValueError('occurrence handle requires an actual call scope and original source port')
        if self.boundary is not None and type(self.boundary) is not CallRecordAnchor:
            raise ValueError('boundary occurrence requires its original attempt record anchor')


@dataclass(frozen=True, slots=True)
class SourceInvocationNode:
    """A query qualification of a shared region/step, never a body copy."""
    call_scope: SourceCallScope
    original: SourceAnchor

    def __post_init__(self):
        if (type(self.call_scope)is not SourceCallScope or type(self.original)is not SourceAnchor
                or type(self.original.original) not in (ProductionRegionID,ProductionStepID)
                or self.original.template is None):
            raise ValueError('scoped source node needs an original F4 region/step anchor')


@dataclass(frozen=True, slots=True)
class SourceInvocationRoute:
    source: SourceInvocationPort
    target: SourceInvocationPort
    field: str
    evidence: tuple[ProviderFactReference,...]
    conditions: tuple[str,...]


@dataclass(frozen=True, slots=True)
class SourceInvocationFrontier:
    field: str
    reason: str
    port: GlobalPort | None = None


@dataclass(frozen=True, slots=True)
class SourceInvocationBinding:
    id: ProviderID
    call_scope: SourceCallScope
    caller_scope: SourceCallScope
    activation: ProviderID
    callsite: ProviderID
    body: Body
    entry: CallRecordAnchor
    exit: CallRecordAnchor | None
    routes: tuple[SourceInvocationRoute,...]
    frontiers: tuple[SourceInvocationFrontier,...]
    semantics: str = 'RECORDED_INVOCATION_SOURCE_PORT_ROUTES_NOT_CONTENT_PRESERVATION'
