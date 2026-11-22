"""Conditional builtin-map selection over original namespace/world ports."""
from dataclasses import dataclass

from scar.ir.program.normal_graph import GlobalPort
from .source_exec_roots import ExecEndpoint
from scar.ir.program.namespace_routes import NamespaceInterfaceRef
from .source_frame_bindings import ContextEndpoint


@dataclass(frozen=True, slots=True)
class GuardedBuiltinCandidate:
    connection: str
    access: GlobalPort
    interface: NamespaceInterfaceRef
    selected_at: ExecEndpoint
    capturing_function: GlobalPort | None
    conditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BuiltinNamespaceSelection:
    """A possible globals entry at selection time, not an immutable snapshot."""
    connection: str
    access: GlobalPort
    interface: NamespaceInterfaceRef
    selected_at: ExecEndpoint
    capturing_function: GlobalPort | None
    globals_reference: GlobalPort
    globals_container: GlobalPort
    source: ContextEndpoint
    read_world: GlobalPort
    conditions: tuple[str, ...]
