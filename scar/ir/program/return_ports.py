"""Return ports."""

from dataclasses import dataclass
from scar.ir.program.normal_targets import GlobalPort, FunctionConnection, Body
from scar.ir.program.providers import ProviderFactReference, ProviderArtifact
from scar.ir.program.content_routes import AddressPath
from scar.ir.program.state_ports import SourceWorldFlow
from scar.ir.program.dependencies import DependencyFrame
from scar.ir.program.normal_graph import Edge, CallSite


# Cross-function state returns and separately guarded reference transport.
@dataclass(frozen=True, slots=True)
class WorldReturnNode:
    source_function: GlobalPort
    declared_context: int | None
    tracked_reference: GlobalPort | None
    flow: SourceWorldFlow


@dataclass(frozen=True, slots=True)
class WorldReturnEdge:
    callee: int
    caller: int
    call: ProviderFactReference
    interface: ProviderFactReference
    connection: FunctionConnection
    state_return: tuple[GlobalPort, GlobalPort]
    data_return: tuple[GlobalPort, GlobalPort] | None
    reference_paths: tuple[AddressPath, ...]
    reference_forwarded: bool
    conditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WorldReturnFrontier:
    node: int
    reason: str


@dataclass(frozen=True, slots=True)
class WorldReturnGraph:
    roots: tuple[int, ...]
    nodes: tuple[WorldReturnNode, ...]
    edges: tuple[WorldReturnEdge, ...]
    frontiers: tuple[WorldReturnFrontier, ...]
    semantics: str = 'GUARDED_WORLD_RETURN_AND_REFERENCE_PATHS_NOT_CONTENT_PRESERVATION'


# Original normal-control windows, conditional on one selected CALL.
@dataclass(frozen=True,slots=True)
class SelectedBranchExclusion:
    operation: GlobalPort
    selected_guard: Edge
    excluded_guard: Edge
    structural_calls: tuple[CallSite,...]


@dataclass(frozen=True,slots=True)
class SelectedWindowBoundary:
    reason: str
    operation: GlobalPort | None=None


@dataclass(frozen=True,slots=True)
class SelectedCallEffectWindow:
    frame: DependencyFrame
    original_call: ProviderFactReference
    interface: ProviderFactReference
    source_function: ProviderFactReference
    source_material: ProviderArtifact
    owning_body: Body
    selected_call: GlobalPort
    world_before: GlobalPort
    world_after: GlobalPort
    normal_prefix_operations: tuple[GlobalPort,...]
    normal_suffix_operations: tuple[GlobalPort,...]
    exclusions: tuple[SelectedBranchExclusion,...]
    boundaries: tuple[SelectedWindowBoundary,...]
    conditions: tuple[str,...]
    scope: str='ONE_SELECTED_CALL_IN_ONE_NORMAL_ACYCLIC_WRAPPER_ACTIVATION'


@dataclass(frozen=True,slots=True)
class OrderedWrapperCallWindow:
    loop: int
    caller_depth: int
    window: SelectedCallEffectWindow


__all__ = [
    'SelectedBranchExclusion',
    'SelectedWindowBoundary',
    'SelectedCallEffectWindow',
    'OrderedWrapperCallWindow',
    'WorldReturnNode',
    'WorldReturnEdge',
    'WorldReturnFrontier',
    'WorldReturnGraph',
]
