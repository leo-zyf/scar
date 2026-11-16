"""Selected original leaf activations for one source-replayed ordered region."""
from dataclasses import dataclass

from .dependencies import DependencyPoint,DependencyFrame
from .normal_links import CodeScopeRef
from .providers import ProviderFactReference


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


__all__=['OrderedInlineRoute','OrderedInlineBoundary']
