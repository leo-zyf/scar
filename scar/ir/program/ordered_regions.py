"""Typed data, state and control ports of ordered regions and loops."""

from dataclasses import dataclass
from enum import Enum
from .dependencies import DependencyPoint, DependencyFrame, DependencyKind, DependencyMeaning
from .providers import ProviderFactReference
from .normal_targets import GlobalPort
from scar.ir.program.content_routes import ContentAccessRef
from scar.ir.program.content_routes import ContentAccessTemplate
from .dependencies import DependencyPoint, DependencyFrame
from .normal_links import CodeScopeRef
from .ids import OperationID
from scar.ir.program.content_routes import FieldSelector
from scar.ir.program.source_value_routes import ReferenceSlice
from . import normal_graph as G, normal_targets as T
from .dependencies import DependencyFrame


class OrderedBoundaryRole(str,Enum):
    ENVIRONMENT='ENVIRONMENT'
    WORLD_ENTRY='WORLD_ENTRY'
    WORLD_EXIT='WORLD_EXIT'
    CONTROL_ENTRY='CONTROL_ENTRY'
    CONTROL_EXIT='CONTROL_EXIT'
    RETURN_NONE='RETURN_NONE'


@dataclass(frozen=True,slots=True)
class OrderedBoundaryPort:
    role: OrderedBoundaryRole
    original: GlobalPort | None
    semantics: str='CONDITIONAL_ORDERED_IMPLEMENTATION_INTERFACE_NOT_RUNTIME_VALUE_IDENTITY'

    def __post_init__(self):
        if type(self.role)is not OrderedBoundaryRole or self.original is not None and type(self.original)is not GlobalPort:
            raise TypeError('ordered boundary needs its typed role and original optional port')


class OrderedContextSpace(str,Enum):
    CALLER='CALLER'
    SOURCE='SOURCE'
    ORDERED='ORDERED'


@dataclass(frozen=True,slots=True)
class OrderedActivationBinding:
    provider: str
    unit: str
    ordinal: int
    query: int
    interface: ProviderFactReference
    source_frame: DependencyFrame
    ordered_frame: DependencyFrame
    guards: tuple[str,...]

    def __post_init__(self):
        if (type(self.interface)is not ProviderFactReference or type(self.source_frame)is not DependencyFrame
                or type(self.ordered_frame)is not DependencyFrame or not self.guards
                or any(type(i)is not int or i<0 for i in(self.ordinal,self.query))):
            raise TypeError('matched original interface, frames, ordinals and guards required')
        left,right=self.source_frame,self.ordered_frame
        if (left.target==right.target or any(getattr(left,k)!=getattr(right,k)
                for k in('caller','operation','callee_use','callee_value','selection'))):
            raise ValueError('ordered frame must select another implementation of the same original call')


@dataclass(frozen=True,slots=True)
class OrderedActivationRoute:
    binding: int
    source: DependencyPoint
    source_space: OrderedContextSpace
    target: DependencyPoint
    target_space: OrderedContextSpace
    kind: DependencyKind
    meaning: DependencyMeaning
    rule: str
    guards: tuple[str,...]=()

    def __post_init__(self):
        if (type(self.binding)is not int or self.binding<0 or type(self.source)is not DependencyPoint
                or type(self.target)is not DependencyPoint
                or any(type(s)is not OrderedContextSpace for s in(self.source_space,self.target_space))
                or type(self.kind)is not DependencyKind or type(self.meaning)is not DependencyMeaning
                or self.source.context is not None or self.target.context is not None):
            raise TypeError('ordered routes require typed unselected template endpoints')


@dataclass(frozen=True,slots=True)
class OrderedContentAccessWitness:
    """A sparse parent-table row retaining its original citation ordinal."""
    reference: ContentAccessRef
    access: ContentAccessTemplate

    def __post_init__(self):
        if (type(self.reference)is not ContentAccessRef or type(self.access)is not ContentAccessTemplate
                or self.reference.operation!=self.access.operation):
            raise ValueError('ordered content witness must retain its actual original access')


@dataclass(frozen=True,slots=True)
class OrderedInlineRoute:
    ordered_key: str
    original_operation: DependencyPoint
    original_origin: ProviderFactReference
    original_call: ProviderFactReference
    provider: str
    unit: str
    interface_ordinal: int
    query: int
    interface: ProviderFactReference
    frame: DependencyFrame
    caller_scope: CodeScopeRef
    body_scope: CodeScopeRef
    guards: tuple[str,...]


@dataclass(frozen=True,slots=True)
class OrderedInlineBoundary:
    ordered_key: str
    original_origin: ProviderFactReference
    reason: str


@dataclass(frozen=True,slots=True)
class OrderedFieldRead:
    selector: FieldSelector
    receiver: GlobalPort
    result: GlobalPort
    world_before: GlobalPort
    world_after: GlobalPort
    attribute: ProviderFactReference
    content_access: ContentAccessRef
    effect_request: OperationID
    receiver_path: ReferenceSlice


@dataclass(frozen=True,slots=True)
class OrderedInputRoute:
    input_key: str
    input_record: ProviderFactReference
    source_origin: ProviderFactReference
    function: ProviderFactReference
    root_formal: GlobalPort
    parameter_index: int
    field_path: tuple[str,...]
    original_read_result: GlobalPort
    reference_path: ReferenceSlice | None
    read_chain: tuple[OrderedFieldRead,...]
    guards: tuple[str,...]

    @property
    def logical_content_version_equality_proven(self):return False


@dataclass(frozen=True,slots=True)
class OrderedInputBoundary:
    input_key: str
    reason: str
    source_origin: ProviderFactReference | None=None


class OrderedInputUseRole(str,Enum):
    SCALAR_VALUE_READ='SCALAR_VALUE_READ'
    ARRAY_CONTENT_READ='ARRAY_CONTENT_READ'
    READ_MODIFY_WRITE='READ_MODIFY_WRITE'
    PRESERVES_UNWRITTEN_REGION='PRESERVES_UNWRITTEN_REGION'
    WRITE_DESTINATION_REFERENCE='WRITE_DESTINATION_REFERENCE'


class OrderedInputStability(str,Enum):
    SELECTED_SCALAR_VALUE_FIXED='SELECTED_SCALAR_VALUE_FIXED'
    ARRAY_CONTENT_PRESERVED='ARRAY_CONTENT_PRESERVED'
    ARRAY_CONTENT_HAS_ORDERED_WRITES='ARRAY_CONTENT_HAS_ORDERED_WRITES'


@dataclass(frozen=True,slots=True)
class OrderedInputUse:
    input_key: str
    action_index: int
    role: OrderedInputUseRole
    epoch: int | None
    expression_key: str | None
    projection_edge_ordinal: int
    action: ProviderFactReference
    source_origins: tuple[ProviderFactReference,...]


@dataclass(frozen=True,slots=True)
class OrderedEpochLifetime:
    input_key: str
    epoch_key: str
    epoch: int
    producer_action: int | None
    read_use_ordinals: tuple[int,...]
    first_read_action: int | None
    last_read_action: int | None
    superseded_at_action: int | None
    required_at_normal_exit: bool


@dataclass(frozen=True,slots=True)
class OrderedInputLifetime:
    input_key: str
    original_input: ProviderFactReference
    input_kind: str
    source_route_ordinals: tuple[int,...]
    expression_keys: tuple[str,...]
    use_ordinals: tuple[int,...]
    write_actions: tuple[int,...]
    epoch_ordinals: tuple[int,...]
    first_read_action: int | None
    last_read_action: int | None
    last_reference_use_action: int | None
    exit_epoch_key: str | None
    entry_content_reaches_exit: bool | None
    stability: OrderedInputStability


@dataclass(frozen=True,slots=True)
class OrderedLoopCallHop:
    provider: str
    unit: str
    ordinal: int
    query: int
    interface: ProviderFactReference
    frame: DependencyFrame
    caller_scope: CodeScopeRef
    guards: tuple[str,...]


@dataclass(frozen=True,slots=True)
class OrderedLoopWorldCarry:
    slot: int
    formal: T.GlobalPort
    backedges: tuple[G.Edge,...]


@dataclass(frozen=True,slots=True)
class OrderedLoopInterface:
    binding: int
    source_function: ProviderFactReference
    header: T.GlobalPort
    body: T.Body
    original_call: ProviderFactReference
    call_path: tuple[OrderedLoopCallHop,...]
    world_carries: tuple[OrderedLoopWorldCarry,...]
    control_backedges: tuple[G.Edge,...]


@dataclass(frozen=True,slots=True)
class OrderedLoopFormalHop:
    interface: ProviderFactReference
    formal: T.GlobalPort
    actual: T.GlobalPort
    caller_formal: T.GlobalPort
    reference_path: ReferenceSlice


@dataclass(frozen=True,slots=True)
class OrderedLoopRootRoute:
    loop: int
    source_formal: T.GlobalPort
    loop_caller_formal: T.GlobalPort
    # Innermost call first, preserving the exact original argument route.
    hops: tuple[OrderedLoopFormalHop,...]


@dataclass(frozen=True,slots=True)
class OrderedLoopCallerWrite:
    loop: int
    caller_depth: int
    fact: ProviderFactReference
    action: str
    receiver: T.GlobalPort
    field: str
    payload: T.GlobalPort | None
    receiver_roots: tuple[T.GlobalPort,...]


@dataclass(frozen=True,slots=True)
class OrderedLoopEffectFrontier:
    loop: int | None
    input_key: str | None
    reason: str
    site: T.GlobalPort | None=None
    fact: ProviderFactReference | None=None


class OrderedLoopInputRole(str,Enum):
    ORDERED_ARRAY_CONTENT_CARRY='ORDERED_ARRAY_CONTENT_CARRY'
    CALLER_REWRITTEN_FIELD='CALLER_REWRITTEN_FIELD'
    LOOP_LIVE_IN_ROOT_SELECTED_VALUE_UNPROVEN='LOOP_LIVE_IN_ROOT_SELECTED_VALUE_UNPROVEN'
    CONDITIONAL_LOOP_LIVE_IN_SCALAR='CONDITIONAL_LOOP_LIVE_IN_SCALAR'


@dataclass(frozen=True,slots=True)
class OrderedLoopInputTransfer:
    loop: int
    input_key: str
    lifetime_ordinal: int
    root_route: int
    field_path: tuple[str,...]
    role: OrderedLoopInputRole
    local_entry_epoch: str | None
    local_exit_epoch: str | None
    caller_write_ordinals: tuple[int,...]
    effect_frontier_ordinals: tuple[int,...]
    guards: tuple[str,...]


__all__ = [
    'OrderedBoundaryRole',
    'OrderedBoundaryPort',
    'OrderedContextSpace',
    'OrderedActivationBinding',
    'OrderedActivationRoute',
    'OrderedContentAccessWitness',
    'OrderedInlineRoute',
    'OrderedInlineBoundary',
    'OrderedFieldRead',
    'OrderedInputRoute',
    'OrderedInputBoundary',
    'OrderedInputUseRole',
    'OrderedInputStability',
    'OrderedInputUse',
    'OrderedEpochLifetime',
    'OrderedInputLifetime',
    'OrderedLoopCallHop',
    'OrderedLoopWorldCarry',
    'OrderedLoopInterface',
    'OrderedLoopFormalHop',
    'OrderedLoopRootRoute',
    'OrderedLoopCallerWrite',
    'OrderedLoopEffectFrontier',
    'OrderedLoopInputRole',
    'OrderedLoopInputTransfer',
]
