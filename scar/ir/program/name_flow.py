"""Name flow."""

from dataclasses import dataclass
from scar.ir.program.normal_graph import CodeScopeRef, CodeScopeWitness, NameAccess, Namespace
from scar.ir.program.normal_graph import GlobalPort, Body
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.program.namespace_routes import (
    NamespaceAccess,
    NamespaceInterfaceRef,
    NamespaceReferenceSlice,
)
from enum import Enum
from scar.ir.program import normal_graph as G
from scar.ir.program.source_value_routes import SourceOperation
from scar.ir.program.python_bindings import NativeCallCandidate
from scar.ir.program.normal_graph import Edge


# Original name facts anchored to source operations, never current versions.
@dataclass(frozen=True, slots=True)
class NameAccessAnchor:
    """access indexes graph.accesses; lookup ordinal selects its read branch."""
    fact: ProviderFactReference
    record: NameAccess
    scope: CodeScopeRef
    access: int
    namespace: Namespace
    lookup_ordinal: int | None
    target: NamespaceInterfaceRef | None


@dataclass(frozen=True, slots=True)
class NameDeleteAnchor:
    """DELETE has an F4 operation but no legacy NameAccess row."""
    access: int
    namespace: Namespace
    target: NamespaceInterfaceRef


@dataclass(frozen=True, slots=True)
class FormalNameAdmission:
    """A logical formal admission at a declared body entry, not a store step.

    parameter is the original environment.parameters ordinal. The formal's
    original body PARAMETER ordinal includes the separate environment prefix.
    """
    fact: ProviderFactReference
    record: NameAccess
    function: ProviderFactReference
    source_function: GlobalPort
    owner: CodeScopeWitness
    parameter: int
    binding_port: int
    formal: GlobalPort
    entry: GlobalPort


@dataclass(frozen=True, slots=True)
class NameAccessAnchorFrontier:
    fact: ProviderFactReference | None
    access: int | None
    scope: CodeScopeRef | None
    reason: str
    candidate_accesses: tuple[int, ...] = ()
    candidate_functions: tuple[ProviderFactReference, ...] = ()


@dataclass(frozen=True, slots=True)
class NameAccessAnchorGraph:
    unit: str
    scopes: tuple[CodeScopeRef, ...]
    materials: tuple[ProviderArtifact, ...]
    accesses: tuple[NamespaceAccess, ...]
    anchors: tuple[NameAccessAnchor, ...]
    formals: tuple[FormalNameAdmission, ...]
    deletes: tuple[NameDeleteAnchor, ...]
    frontiers: tuple[NameAccessAnchorFrontier, ...]
    conditions: tuple[str, ...] = (
        'SOURCE_SCOPE_SELECTION_IS_QUERY_NOT_ACTIVATION',
        'ORIGINAL_OPERATION_AND_LOOKUP_BRANCH_REQUIRE_SELECTED_EXECUTION_CONTEXT',
        'OPERATION_WORLD_PORTS_NOT_CURRENT_NAMESPACE_SLOT_VERSION_PROOF',
        'NO_REACHING_DEFINITION_OR_REFERENCE_IDENTITY_ESTABLISHED',
    )
    semantics: str = 'SOURCE_NAME_FACT_OPERATION_ANCHORS_NOT_NAME_VERSIONS'


# Conditional reaching binding definitions over a shared source control view.
class NameDefinitionKind(str, Enum):
    ENTRY_OPEN = 'ENTRY_OPEN'
    FORMAL = 'FORMAL'
    WRITE = 'WRITE'
    DELETE = 'DELETE'
    OPAQUE_EFFECT = 'OPAQUE_EFFECT'


@dataclass(frozen=True, slots=True)
class NameDefinition:
    """None value means no known reference operand, never Python None.

    access/formal index the input anchor graph; operation indexes operations.
    """
    kind: NameDefinitionKind
    site: G.Port
    value: GlobalPort | None
    access: int | None
    formal: int | None
    operation: int | None
    evidence: tuple[ProviderFactReference, ...]
    conditions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NameDefinitionState:
    port: G.Port
    stack: tuple[G.CallSite, ...]
    incoming: tuple[int, ...]
    outgoing: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NameDefinitionTransition:
    """edge indexes the unchanged control_edges table."""
    source: int
    target: int
    edge: int


@dataclass(frozen=True, slots=True)
class NameDefinitionRead:
    """Definitions observed at one READ ENTER and its recipe caller stack."""
    state: int
    access: int
    anchors: tuple[int, ...]
    definitions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NameDefinitionFrontier:
    state: int | None
    edge: int | None
    access: int | None
    reason: str
    evidence: tuple[ProviderFactReference, ...] = ()


@dataclass(frozen=True, slots=True)
class NameDefinitionGraph:
    source_function: GlobalPort
    function: ProviderFactReference
    body: Body
    scope: CodeScopeRef
    namespace: Namespace
    anchor_graph: NameAccessAnchorGraph
    control_edges: tuple[G.Edge, ...]
    operations: tuple[SourceOperation, ...]
    definitions: tuple[NameDefinition, ...]
    states: tuple[NameDefinitionState, ...]
    transitions: tuple[NameDefinitionTransition, ...]
    reads: tuple[NameDefinitionRead, ...]
    frontiers: tuple[NameDefinitionFrontier, ...]
    traversal_complete: bool
    dataflow_complete: bool
    conditions: tuple[str, ...] = (
        'SOURCE_FUNCTION_BODY_AND_SAME_LEXICAL_FRAME_BINDING_CONTEXT_SELECTED',
        'STRUCTURAL_RECIPE_CALL_RETURN_SITES_MATCHED',
        'BRANCH_GUARDS_RETAINED_WITHOUT_RUNTIME_PATH_SELECTION',
        'APPLICATION_EXCEPTION_AND_RESUME_PATHS_REMAIN_OPEN',
        'OPAQUE_CALLBACK_AND_CAPTURE_IDENTITY_EFFECTS_REMAIN_OPEN',
        'TRACKED_MAY_REACHING_BINDING_DEFINITIONS_NOT_TENSOR_CONTENT_VERSIONS',
        'NO_CURRENT_REFERENCE_RUNTIME_ENTRY_OR_EXTERNAL_WRITER_ABSENCE_PROOF',
    )
    loops_unrolled: bool = False
    semantics: str = 'CONDITIONAL_TRACKED_MAY_REACHING_BINDING_DEFINITIONS'


# Full namespace requests linked to conditional generated binding interfaces.
@dataclass(frozen=True, slots=True)
class NamedBindingMatch:
    request: ProviderFactReference
    call: ProviderFactReference
    declaration: ProviderFactReference
    symbol: str
    candidate: NativeCallCandidate


@dataclass(frozen=True, slots=True)
class NamedBindingFrontier:
    request: ProviderFactReference
    symbol: str
    reason: str


# Control-qualified name sources; the original declaration graph stays shared.
NORMAL_VIEW = (
    'RECORDED_NORMAL_RECIPE_CONTROL_PATH_AND_SAME_FRAME_SELECTED',
    'ALL_RETAINED_RECIPE_CONTEXTS_AND_BRANCH_ALTERNATIVES_JOINED_AS_MAY',
    'APPLICATION_EXCEPTION_AND_RESUME_PATHS_REQUIRE_SEPARATE_SOURCE_BRANCH',
    'OPAQUE_REFERENCE_SOURCES_REMAIN_OPEN_NOT_CONNECTED_TO_ARBITRARY_WRITERS',
    'NO_CURRENT_BINDING_CONTENT_TENSOR_VERSION_OR_RUNTIME_DISPATCH_PROOF',
)


@dataclass(frozen=True, slots=True)
class NameReadFilter:
    read: ProviderFactReference
    scope: CodeScopeRef | None
    track: int | None
    accesses: tuple[int, ...]
    read_states: tuple[int, ...]
    writers: tuple[ProviderFactReference, ...]
    open_kinds: tuple[NameDefinitionKind, ...]
    applied: bool
    reason: str
    conditions: tuple[str, ...] = NORMAL_VIEW


@dataclass(frozen=True, slots=True)
class ExcludedNameWriter:
    downstream: int
    writer: ProviderFactReference
    read: ProviderFactReference
    filter: int
    conditions: tuple[str, ...] = NORMAL_VIEW


@dataclass(frozen=True, slots=True)
class ControlNamespaceQuery:
    query: int
    callee: GlobalPort
    reference: NamespaceReferenceSlice
    filters: tuple[int, ...]
    excluded: tuple[ExcludedNameWriter, ...]
    conditions: tuple[str, ...] = NORMAL_VIEW


@dataclass(frozen=True, slots=True)
class NameControlBody:
    source_function: GlobalPort
    function: ProviderFactReference
    body: Body
    scope: CodeScopeRef
    control_edges: tuple[Edge, ...]
    operations: tuple[SourceOperation, ...]


@dataclass(frozen=True, slots=True)
class StoredNameDefinitionTrack:
    """Per-slot propagation references shared body and anchor tables."""
    anchor: int
    body: int
    namespace: Namespace
    definitions: tuple[NameDefinition, ...]
    states: tuple[NameDefinitionState, ...]
    transitions: tuple[NameDefinitionTransition, ...]
    reads: tuple[NameDefinitionRead, ...]
    frontiers: tuple[NameDefinitionFrontier, ...]
    traversal_complete: bool
    dataflow_complete: bool
    conditions: tuple[str, ...]
    loops_unrolled: bool
    semantics: str


@dataclass(frozen=True, slots=True)
class ControlNamespaceUnit:
    unit: str
    anchors: tuple[NameAccessAnchorGraph, ...]
    bodies: tuple[NameControlBody, ...]
    tracks: tuple[NameDefinitionGraph, ...]
    filters: tuple[NameReadFilter, ...]
    queries: tuple[ControlNamespaceQuery, ...]
    semantics: str = 'NORMAL_CONTROL_QUALIFIED_EXPLICIT_NAME_SOURCES_WITH_OPEN_EFFECT_FRONTIERS'
