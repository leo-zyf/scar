"""Binding ports."""

from dataclasses import dataclass
from scar.ir.program.normal_graph import Port, CallSite
from scar.ir.program.normal_links import Namespace
from scar.ir.program.normal_targets import GlobalPort, Body, Endpoint, Route, FunctionActivation
from scar.ir.program.name_flow import (
    NameDefinition,
    NameDefinitionTransition,
    NameDefinitionFrontier,
)
from enum import Enum
from scar.ir.program.providers import ProviderFactReference
from scar.ir.program.content_routes import AddressPath
from scar.ir.program.state_ports import SourceWorldFlow


# Lossless physical layout of matched binding tracks, not new graph nodes.
@dataclass(frozen=True,slots=True)
class SharedBindingControl:
    source_function: GlobalPort
    max_states: int
    max_stack: int
    states: tuple[tuple[Port,tuple[CallSite,...]],...]
    transitions: tuple[NameDefinitionTransition,...]
    traversal_complete: bool


@dataclass(frozen=True,slots=True)
class PackedBindingTrack:
    # Unit header + ordinal select a control record. This is not a value ID.
    control: int
    namespace: Namespace
    max_updates: int
    definitions: tuple[NameDefinition,...]
    # Lowercase hexadecimal bitsets of track-local definition ordinals.
    # Equal sets in different tracks do not identify equal program values.
    definition_sets: tuple[str,...]
    states: tuple[tuple[int,int],...]
    reads: tuple[tuple[int,int,tuple[int,...],int],...]
    frontiers: tuple[NameDefinitionFrontier,...]
    dataflow_complete: bool


# Original argument uses and conditional source parameter consumers.
class ArgumentRole(str, Enum):
    CALLEE = 'CALLEE'
    ATTRIBUTE_RECEIVER = 'ATTRIBUTE_RECEIVER'
    POSITIONAL = 'POSITIONAL'
    KEYWORD = 'KEYWORD'
    POSITIONAL_PACK = 'POSITIONAL_PACK'
    KEYWORD_PACK = 'KEYWORD_PACK'


@dataclass(frozen=True, slots=True)
class ReferenceArgumentUse:
    node: int
    call: ProviderFactReference
    role: ArgumentRole
    ordinal: int
    keyword: str | None
    actual: GlobalPort
    paths: tuple[AddressPath, ...]
    direct_reference: bool
    world_before: GlobalPort
    world_after: GlobalPort
    conditions: tuple[str, ...]
    attribute: ProviderFactReference | None = None


@dataclass(frozen=True, slots=True)
class SourceConsumerBinding:
    use: int
    interface: ProviderFactReference
    source_callable: GlobalPort
    body: Body
    activation: FunctionActivation
    argument: Route
    state: Route
    data_return: tuple[Endpoint, Endpoint] | None
    state_return: tuple[Endpoint, Endpoint] | None
    conditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConsumerFrontier:
    node: int
    use: int | None
    actual: GlobalPort | None
    interface: ProviderFactReference | None
    reason: str


@dataclass(frozen=True, slots=True)
class ConsumerBody:
    source_function: GlobalPort
    flow: SourceWorldFlow


@dataclass(frozen=True, slots=True)
class ConsumerBodyEntry:
    binding: int
    body: int
    formal: GlobalPort
    conditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReferenceConsumerGraph:
    uses: tuple[ReferenceArgumentUse, ...]
    bindings: tuple[SourceConsumerBinding, ...]
    frontiers: tuple[ConsumerFrontier, ...]
    bodies: tuple[ConsumerBody, ...] = ()
    entries: tuple[ConsumerBodyEntry, ...] = ()
    semantics: str = 'CONDITIONAL_ARGUMENT_DEPENDENCIES_NOT_SELECTED_DISPATCH_ESCAPE_OR_CONTENT_IDENTITY'
