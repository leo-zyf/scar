"""State ports."""

from dataclasses import dataclass
from scar.ir.program.configuration import ConfigurationProtocolSource
from scar.ir.program.normal_graph import CodeScopeWitness, CodeScopeRef, Namespace
from scar.ir.program.normal_graph import GlobalPort, Endpoint, FunctionConnection, Route, Body
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.program.normal_graph import Edge
from enum import Enum
from scar.ir.effects import Knowledge
from scar.ir.program.ids import LocationID, OperationID, ValueID
from scar.ir.program.source_bindings import EnvironmentInterface
from scar.ir.program.source_effects import FamilyRule
from scar.ir.program.name_flow import NameDefinitionKind, NameDefinition, NameDefinitionState, NameDefinitionTransition, NameDefinitionRead, NameDefinitionFrontier
from scar.ir.program.dependencies import (
    NodeID,
    DependencyKind,
    DependencyMeaning,
    DependencyStrength,
)
from scar.ir.program.view import StoredDependencyCondition


# Conditional content projections at source call state boundaries.
@dataclass(frozen=True,slots=True)
class SourceStateProjection:
    value_graph: ProviderArtifact
    value_expression: int
    facts: ProviderArtifact
    protocol: ConfigurationProtocolSource
    source_function: GlobalPort
    caller: CodeScopeWitness
    call: ProviderFactReference
    attribute: ProviderFactReference
    attribute_result: GlobalPort
    callee: GlobalPort
    callee_alias_rows: tuple[int,...]
    receiver: GlobalPort
    world_before: GlobalPort
    world_after: GlobalPort
    data_result: GlobalPort
    normal_return: GlobalPort
    none_origin: GlobalPort
    none_alias_rows: tuple[int,...]
    conditions: tuple[str,...]
    semantics: str = 'CONDITIONAL_CONTENT_AT_CALL_RESULT_WORLD_NOT_PYTHON_RESULT_OR_TASK_STATE'


# Conditional state projections through shared declared body interfaces.
SEMANTICS = 'GUARDED_DECLARED_STATE_INTERFACE_NOT_CURRENT_CONTENT_OR_SCOPE_ADMISSION'


@dataclass(frozen=True, slots=True)
class DeclaredStateConnection:
    projection: ProviderFactReference
    edge: ProviderFactReference
    connection: FunctionConnection
    content_route: Route
    state_route: Route
    data_return: tuple[Endpoint, Endpoint]
    state_return: tuple[Endpoint, Endpoint]
    conditions: tuple[str, ...]
    semantics: str = SEMANTICS


@dataclass(frozen=True, slots=True)
class DeclaredStateObligation:
    projection: int
    code: str
    detail: str


# State sequencing over original F4 ports, distinct from value aliases.
@dataclass(frozen=True, slots=True)
class OperationStateTransition:
    before: GlobalPort
    after: GlobalPort
    instruction_type: str
    conditions: tuple[str, ...] = ('NORMAL_OPERATION_COMPLETION_SELECTED', 'OPERATION_EFFECTS_REMAIN_OPEN')


@dataclass(frozen=True, slots=True)
class SequencedAccess:
    fact: ProviderFactReference
    table: str
    action: str
    world_before: GlobalPort
    world_after: GlobalPort


@dataclass(frozen=True, slots=True)
class SourceWorldFlow:
    projection: ProviderFactReference
    source_templates: ProviderArtifact
    unit: str
    caller_body: Body
    anchor: GlobalPort
    normal_exit: GlobalPort
    edges: tuple[Edge, ...]
    operations: tuple[OperationStateTransition, ...]
    accesses: tuple[SequencedAccess, ...]
    anchor_reaches_exit: bool
    conditions: tuple[str, ...]
    semantics: str = 'CONDITIONAL_SHARED_STATE_PATHS_NOT_RUNTIME_ORDER_OR_CONTENT_PRESERVATION'


# Static binding addresses/versions, distinct from objects and tensor data.
#
# A location is a source interface template. Its actual frame, mapping or cell
# must still be supplied. A version is indexed by an execution occurrence;
# one write site in a loop does not denote a single immutable runtime value.
class BindingAccessKind(str, Enum):
    READ = 'READ'
    WRITE = 'WRITE'
    DELETE = 'DELETE'
    HANDLE = 'HANDLE'


class BindingBoundness(str, Enum):
    BOUND = 'BOUND'
    UNBOUND = 'UNBOUND'
    UNKNOWN = 'UNKNOWN'


class StateInstanceParameter(str, Enum):
    ACTIVATION_OR_CAPTURED_ENVIRONMENT = 'ACTIVATION_OR_CAPTURED_ENVIRONMENT'
    SITE_EXECUTION_OCCURRENCE = 'SITE_EXECUTION_OCCURRENCE'


@dataclass(frozen=True, slots=True)
class BindingLocationTemplate:
    id: LocationID
    unit: str
    interface: EnvironmentInterface
    key: str
    instance_parameters: tuple[StateInstanceParameter, ...] = (
        StateInstanceParameter.ACTIVATION_OR_CAPTURED_ENVIRONMENT,)


@dataclass(frozen=True, slots=True)
class BindingAccessAlternative:
    """An ordered lookup branch, not an unconditional read of every mapping."""
    location: LocationID
    interface: int | None
    lookup_ordinal: int | None
    on_hit: str | None = None
    on_miss: str | None = None
    on_exception: str | None = None


@dataclass(frozen=True, slots=True)
class BindingAccessTemplate:
    operation: OperationID
    site: GlobalPort
    scope: CodeScopeRef
    kind: BindingAccessKind
    binding_port: int | None
    alternatives: tuple[BindingAccessAlternative, ...]
    payload: GlobalPort | None
    world_in: GlobalPort
    world_out: GlobalPort
    missing_outcome: str | None = None


@dataclass(frozen=True, slots=True)
class SourceEffectRequest:
    """Identification is total; resolved effect membership need not be total.

    Original instructions/ports remain in the sealed source unit. A request
    stores their identity and a closed dispatch rule, never a tensor snapshot.
    """
    operation: OperationID
    rule: FamilyRule
    accesses: tuple[int, ...]
    missing_rules: tuple[str, ...]
    pending: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceEffectSet:
    """Known source access members and independent completeness knowledge.

    Ordinals refer to the owning unit's binding/content access tables. A
    protocol request is not an observed physical read or write. An empty
    UNKNOWN set never means that no effect exists.
    """
    knowledge: Knowledge = Knowledge.UNKNOWN
    binding_accesses: tuple[int, ...] = ()
    content_accesses: tuple[int, ...] = ()

    def __post_init__(self):
        if (type(self.knowledge) is not Knowledge
                or any(type(n) is not int or n < 0 for n in (*self.binding_accesses, *self.content_accesses))
                or self.knowledge is Knowledge.NONE and (self.binding_accesses or self.content_accesses)):
            raise ValueError('effect members require original nonnegative access ordinals and consistent knowledge')


@dataclass(frozen=True, slots=True)
class SourceEffectSummary:
    operation: OperationID
    rule: FamilyRule
    world_inputs: tuple[GlobalPort, ...]
    world_outputs: tuple[GlobalPort, ...]
    reads: SourceEffectSet
    writes: SourceEffectSet
    allocates: SourceEffectSet
    frees: SourceEffectSet
    aliases: SourceEffectSet
    escapes: SourceEffectSet
    rng_effect: Knowledge
    may_raise: Knowledge
    external_effect: Knowledge
    ordering_effect: Knowledge
    unresolved_fields: tuple[str, ...]
    missing_rules: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BindingVersionTemplate:
    id: ValueID
    location: LocationID
    source_function: GlobalPort
    namespace: Namespace
    definition: int
    kind: NameDefinitionKind
    boundness: BindingBoundness
    payload: GlobalPort | None
    instance_parameters: tuple[StateInstanceParameter, ...] = (
        StateInstanceParameter.ACTIVATION_OR_CAPTURED_ENVIRONMENT,
        StateInstanceParameter.SITE_EXECUTION_OCCURRENCE)


@dataclass(frozen=True, slots=True)
class BindingReadSelection:
    """Indices retain the original matched recipe stack and source guards."""
    track: int
    read: int
    access: int
    state: int
    versions: tuple[ValueID, ...]


@dataclass(frozen=True, slots=True)
class StoredBindingTrack:
    """Operation/edge indices refer to the pinned original body projection.

    Body instructions, world ports and control edges are not copied. The
    reader rederives their original order before accepting these indices.
    """
    source_function: GlobalPort
    namespace: Namespace
    max_states: int
    max_stack: int
    max_updates: int
    definitions: tuple[NameDefinition, ...]
    states: tuple[NameDefinitionState, ...]
    transitions: tuple[NameDefinitionTransition, ...]
    reads: tuple[NameDefinitionRead, ...]
    frontiers: tuple[NameDefinitionFrontier, ...]
    traversal_complete: bool
    dataflow_complete: bool


@dataclass(frozen=True, slots=True)
class StoredBindingEdge:
    """Source namespace and rule materials are factored into the shard header."""
    source: NodeID
    target: NodeID
    kind: DependencyKind
    meaning: DependencyMeaning
    strength: DependencyStrength
    rule: str
    conditions: tuple[StoredDependencyCondition, ...]
