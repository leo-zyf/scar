"""Original normal-control windows, conditional on one selected CALL."""
from dataclasses import dataclass

from .dependencies import DependencyFrame
from .normal_graph import Edge,CallSite
from .normal_targets import GlobalPort,Body
from .providers import ProviderArtifact,ProviderFactReference


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


__all__=['SelectedBranchExclusion','SelectedWindowBoundary','SelectedCallEffectWindow',
    'OrderedWrapperCallWindow']
