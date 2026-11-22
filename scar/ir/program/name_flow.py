"""Name flow."""

from dataclasses import dataclass
from scar.ir.program.normal_graph import (
    CodeScopeRef,
    CodeScopeWitness,
    NameAccess,
    Namespace,
    GlobalPort,
    Body,
    Edge,
)
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from enum import Enum
from scar.ir.program import (
    normal_graph as G,
    normal_graph as F,
    normal_graph as T,
    source_bindings as B,
    source_productions as P,
)
from scar.ir.program.source_value_routes import SourceOperation
from scar.ir.program.python_bindings import NativeCallCandidate
from scar.ir.literals import PythonLiteral
from scar.ir.program.source_exec_roots import ExecEndpoint


# Namespace routes.
# Ordered namespace accesses over shared source graphs, not module aliases.
@dataclass(frozen=True, slots=True)
class NamespaceInterfaceRef:
    unit: str
    index: int


@dataclass(frozen=True, slots=True)
class NamespaceLookupStep:
    """Step n is attempted only after all earlier steps report a miss."""
    ordinal: int
    interface: NamespaceInterfaceRef
    on_hit: str
    on_miss: str
    on_exception: str


@dataclass(frozen=True, slots=True)
class NamespaceAccess:
    site: T.GlobalPort
    owner: F.CodeScopeWitness
    binding_port: int
    action: P.BindingActionKind
    name: B.NameRef
    key: str
    read_plan: int | None
    lookup: tuple[NamespaceLookupStep, ...]
    missing_outcome: str | None
    target: NamespaceInterfaceRef | None
    value: T.GlobalPort | None
    world_in: T.GlobalPort
    world_out: T.GlobalPort


@dataclass(frozen=True, slots=True)
class NamespaceInterfaceBinding:
    """A branch binds templates to handles; it does not prove their contents."""
    connection: str
    interface: NamespaceInterfaceRef
    descriptor: B.EnvironmentInterface
    origin: str
    actual: ExecEndpoint | None
    root_formal: T.GlobalPort | None
    conditions: tuple[str, ...]
    capturing_function: T.GlobalPort | None = None


@dataclass(frozen=True, slots=True)
class GlobalsEnvironmentReference:
    site: T.GlobalPort
    owner: F.CodeScopeWitness
    globals_scope: int
    result: T.GlobalPort
    world_in: T.GlobalPort
    world_out: T.GlobalPort


@dataclass(frozen=True, slots=True)
class FunctionGlobalsCapture:
    """Retain the actual globals argument, separately from code and closure."""
    site: T.GlobalPort
    owner: F.CodeScopeWitness
    function: T.GlobalPort
    code: T.GlobalPort
    globals: T.GlobalPort
    body: F.CodeScopeWitness
    globals_scope: int
    world_in: T.GlobalPort
    world_out: T.GlobalPort
    builtins_policy: str = 'SELECT_FROM_GLOBALS_OR_CURRENT_FRAME_AT_FUNCTION_CREATION'


@dataclass(frozen=True, slots=True)
class GlobalsReferenceBinding:
    connection: str
    reference: T.GlobalPort
    actual: ExecEndpoint
    owner: F.CodeScopeWitness
    conditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NamespaceReferenceHop:
    source: T.GlobalPort
    target: T.GlobalPort
    table: F.FactTable
    rows: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class NamespaceContentLink:
    """Conditional reaching definition, never a constant or equality proof."""
    connection: str
    interface: NamespaceInterfaceRef
    lookup_ordinal: int
    read: T.GlobalPort
    read_world: T.GlobalPort
    mapping: T.GlobalPort
    assigned: T.GlobalPort
    write_world: T.GlobalPort
    write_result_world: T.GlobalPort
    item_row: int
    source_table: F.FactTable
    member_ordinal: int | None
    key: str
    key_source: T.GlobalPort | PythonLiteral
    key_literal_source: T.GlobalPort | None
    key_reference_path: tuple[NamespaceReferenceHop, ...]
    reference_path: tuple[NamespaceReferenceHop, ...]
    conditions: tuple[str, ...]
    capturing_function: T.GlobalPort | None = None
    globals_reference: T.GlobalPort | None = None
    relation: str = 'MAY_REACHING_NAMESPACE_CONTENT'


@dataclass(frozen=True, slots=True)
class NamespaceGraphObligation:
    site: T.GlobalPort
    reason: str


@dataclass(frozen=True, slots=True)
class ExplicitNamespaceAction:
    site: T.GlobalPort
    owner: F.CodeScopeWitness
    kind: P.NamespaceActionKind
    arguments: tuple[T.GlobalPort, ...]
    world_in: T.GlobalPort
    world_out: T.GlobalPort


# Conditional reference sources through declared namespace slots/imports.
class NamespaceEdgeKind(str, Enum):
    ALIAS = 'ALIAS'
    LEXICAL_WRITE = 'LEXICAL_WRITE'
    GLOBAL_WRITE = 'GLOBAL_WRITE'
    CLASS_WRITE = 'CLASS_WRITE'
    ATTRIBUTE_RECEIVER = 'ATTRIBUTE_RECEIVER'
    MODULE_EXPORT = 'MODULE_EXPORT'


class NamespaceOriginKind(str, Enum):
    CLASS_DEFINITION = 'CLASS_DEFINITION'
    FUNCTION_DEFINITION = 'FUNCTION_DEFINITION'
    SOURCE_LITERAL = 'SOURCE_LITERAL'
    MODULE_CANDIDATE = 'MODULE_CANDIDATE'


@dataclass(frozen=True, slots=True)
class NamespaceSelector:
    member: str
    import_from: bool
    access: ProviderFactReference


@dataclass(frozen=True, slots=True)
class NamespaceReferenceState:
    port: GlobalPort
    selectors: tuple[NamespaceSelector, ...] = ()


@dataclass(frozen=True, slots=True)
class NamespaceReferenceEdge:
    downstream: int
    upstream: int
    kind: NamespaceEdgeKind
    evidence: tuple[ProviderFactReference, ...]
    conditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NamespaceReferenceOrigin:
    state: int
    kind: NamespaceOriginKind
    evidence: tuple[ProviderFactReference, ...]
    conditions: tuple[str, ...]
    symbol: str | None = None
    providers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NamespaceReferenceFrontier:
    state: int
    reason: str
    evidence: tuple[ProviderFactReference, ...] = ()


@dataclass(frozen=True, slots=True)
class NamespaceReferenceSlice:
    states: tuple[NamespaceReferenceState, ...]
    edges: tuple[NamespaceReferenceEdge, ...]
    origins: tuple[NamespaceReferenceOrigin, ...]
    frontiers: tuple[NamespaceReferenceFrontier, ...]
    semantics: str = 'DECLARED_NAMESPACE_SOURCE_ALTERNATIVES_NOT_CURRENT_SLOT_OR_OBJECT_IDENTITY'


# Namespace lookup relationships over original source ports.
#
# An empty inferred value set is an open binding requirement, not an absent
# member or an opaque operation. Native interface providers can attach here.
class NamespaceKind(str, Enum):
    MODULE = 'MODULE'
    EXTERNAL_SYMBOL = 'EXTERNAL_SYMBOL'


@dataclass(frozen=True, slots=True)
class NamespaceLookup:
    receiver: GlobalPort
    result: GlobalPort
    namespace: str
    member: str
    kind: NamespaceKind
    source_units: tuple[str, ...]
    inferred_source_values: int
    declared_import_candidate: bool
    conditions: tuple[str, ...] = ('POSSIBLE_NAMESPACE_REFERENCE', 'STANDARD_ATTRIBUTE_LOOKUP')
    binding_proven: bool = False


# Name flow.
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
