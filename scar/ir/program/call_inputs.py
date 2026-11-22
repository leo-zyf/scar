"""Call inputs."""

from dataclasses import dataclass
from scar.ir.program.normal_graph import CallLink, Endpoint, GlobalPort, FunctionConnection, RouteKind, Body
from scar.ir.program.providers import ProviderFactReference, ProviderArtifact
from scar.ir.program.source_value_routes import ReferenceTransition, ReferenceOrigin, ReferenceFrontier
from enum import Enum
from scar.ir.program.ids import LocationID, ValueID, OperationID
from scar.ir.program.source_productions import DeferredParameter
from scar.ir.program import source_bindings as B, source_productions as P
from scar.ir.program.dependencies import DependencyFrame
from scar.ir.program.state_ports import BindingLocationTemplate
from scar.ir.program.normal_graph import CodeScopeRef
from scar.ir.program.name_flow import NamespaceReferenceSlice, NamespaceSelector


# Matched static incoming interfaces, distinct from runtime invocations.
@dataclass(frozen=True, slots=True)
class CallInterfaceRecord:
    evidence: ProviderFactReference
    link: CallLink


@dataclass(frozen=True, slots=True)
class ContextReferenceState:
    port: Endpoint
    interfaces: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ContextReferenceSlice:
    states: tuple[ContextReferenceState, ...]
    edges: tuple[ReferenceTransition, ...]
    origins: tuple[ReferenceOrigin, ...]
    frontiers: tuple[ReferenceFrontier, ...]
    semantics: str = 'SELECTED_INCOMING_INTERFACE_REFERENCE_PATHS_NOT_CONTENT_OR_RUNTIME_IDENTITY'


@dataclass(frozen=True, slots=True)
class CallContextQuery:
    incoming: int
    call: ProviderFactReference
    callee: GlobalPort
    reference: ContextReferenceSlice
    root: int = 0


@dataclass(frozen=True, slots=True)
class CallContextRoot:
    body: int
    incoming: int
    entries: tuple[ProviderFactReference, ...]


@dataclass(frozen=True, slots=True)
class ContextFunctionConnection:
    query: int
    origin: int
    connection: FunctionConnection


@dataclass(frozen=True, slots=True)
class ContextTargetEvidence:
    query: int
    candidate: int
    relation: str
    conditions: tuple[str, ...] = (
        'SOURCE_OWNING_SCOPES_AND_RECORDED_REFERENCE_PATH_CONDITIONS_SELECTED',
        'HISTORICAL_CANDIDATES_RETAINED_NOT_RUNTIME_DISPATCH',
        'NO_EFFECT_CONTENT_VERSION_OR_TRANSFORMATION_AUTHORITY',
    )


# Logical argument-binding state; no CPython struct or tensor snapshots.
class DefaultInstanceParameter(str,Enum):
    CALLER_ACTIVATION='CALLER_ACTIVATION'
    CALLABLE_REFERENCE_OCCURRENCE='CALLABLE_REFERENCE_OCCURRENCE'
    WORLD_OCCURRENCE='WORLD_OCCURRENCE'


@dataclass(frozen=True,slots=True)
class FunctionDefaultSlot:
    id: LocationID
    callable: GlobalPort
    definition: GlobalPort
    parameter_index: int
    parameter_name: str
    parameter: DeferredParameter
    instance_parameters: tuple[DefaultInstanceParameter,...] = (
        DefaultInstanceParameter.CALLER_ACTIVATION,
        DefaultInstanceParameter.CALLABLE_REFERENCE_OCCURRENCE)
    semantics: str = 'SELECTED_SOURCE_FUNCTION_DEFAULT_SLOT_NOT_OBJECT_CONTENT'


@dataclass(frozen=True,slots=True)
class FunctionDefaultState:
    id: ValueID
    address: LocationID
    world: GlobalPort
    instance_parameters: tuple[DefaultInstanceParameter,...] = (
        DefaultInstanceParameter.CALLER_ACTIVATION,DefaultInstanceParameter.WORLD_OCCURRENCE)
    semantics: str = 'SYMBOLIC_CURRENT_SLOT_AT_WORLD_NOT_CREATION_VALUE_EQUALITY'


@dataclass(frozen=True,slots=True)
class FunctionDefaultRead:
    operation: OperationID
    address: LocationID
    state: ValueID
    formal: GlobalPort
    initial_source: GlobalPort
    route: RouteKind
    semantics: str = 'INITIAL_SOURCE_REQUIRES_SELECTED_CREATION_AND_UNCHANGED_DEFAULT_STATE'


# Selected callee-environment cell handles, never captured contents.
@dataclass(frozen=True, slots=True)
class CaptureRelaySelection:
    """Exactly one original C2 relay or sealed derived transport row."""
    reference: P.CaptureReference
    relay: B.ForwardHop | None
    transport: P.CaptureTransport | None


@dataclass(frozen=True, slots=True)
class FunctionCaptureBinding:
    function_fact: ProviderFactReference
    call: ProviderFactReference
    reference: ProviderFactReference
    function: GlobalPort
    body: Body
    frame: DependencyFrame
    environment: GlobalPort
    environment_source: GlobalPort
    capture: int
    relay: P.CaptureReference
    path: tuple[CaptureRelaySelection, ...]
    binding: int
    # Original C2 source-scope owner, not a CodeScopeRef ordinal or invocation.
    owner: int
    key: str
    location: BindingLocationTemplate
    materials: tuple[ProviderArtifact, ...]
    conditions: tuple[str, ...]
    semantics: str = 'CONDITIONAL_CALLEE_ENVIRONMENT_CELL_HANDLE_NOT_CREATOR_FRAME_ALIAS_OR_CONTENT_READ'


# Original source call declarations and conditional namespace requests.
@dataclass(frozen=True, slots=True)
class DeclaredCallQuery:
    unit: str
    call: ProviderFactReference
    callee: GlobalPort
    owners: tuple[CodeScopeRef, ...]
    reference: NamespaceReferenceSlice
    formal_frontiers: tuple[int, ...] = ()
    call_result_frontiers: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DeclaredCallSymbolRequest:
    """query indexes this unit's queries table, preserving calls ordinals."""
    query: int
    state: int
    imported: ProviderFactReference
    returned_module: str
    requested_module: str | None
    selectors: tuple[NamespaceSelector, ...]
    conditions: tuple[str, ...] = (
        'STANDARD_IMPORT_RETURN_AND_RECORDED_ATTRIBUTE_PATH_SELECTED',
        'CURRENT_NAMESPACE_EXPORT_OR_IMPORT_FROM_BRANCH_NOT_SELECTED',
        'SYMBOL_PATH_IS_INTERFACE_REQUEST_NOT_CALLABLE_OR_NATIVE_BINDING_PROOF',
        'SOURCE_CALL_DECLARATION_DOES_NOT_ESTABLISH_OWNING_SCOPE_ENTRY',
    )


@dataclass(frozen=True, slots=True)
class DeclaredCallUnit:
    unit: str
    queries: tuple[DeclaredCallQuery, ...]
    symbols: tuple[DeclaredCallSymbolRequest, ...]
    semantics: str = 'DECLARED_SOURCE_CALLS_NOT_ENTERED_FRAMES_OR_RUNTIME_DISPATCH'
