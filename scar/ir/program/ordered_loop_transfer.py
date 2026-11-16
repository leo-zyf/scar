"""Shared-loop state paths; world carry is not a logical content version."""
from dataclasses import dataclass
from enum import Enum

from . import normal_graph as G,normal_targets as T
from .dependencies import DependencyFrame
from .normal_links import CodeScopeRef
from .providers import ProviderFactReference
from .source_reference_slices import ReferenceSlice


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


__all__=['OrderedLoopCallHop','OrderedLoopWorldCarry','OrderedLoopInterface','OrderedLoopFormalHop',
    'OrderedLoopRootRoute','OrderedLoopCallerWrite','OrderedLoopEffectFrontier',
    'OrderedLoopInputRole','OrderedLoopInputTransfer']
