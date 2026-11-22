"""Class ports."""

from dataclasses import dataclass
from scar.ir.program.normal_graph import CodeScopeRef
from scar.ir.program.normal_graph import GlobalPort, ConstructionLink, FunctionConnection
from scar.ir.program.providers import ProviderFactReference
from scar.ir.program.data_materials import AddressPath
from scar.ir.program.name_flow import NamespaceReferenceSlice
from scar.ir.program.declared_contexts import DeclaredContext
from scar.ir.program.source_value_routes import SourceBodyCall
from scar.ir.program.ids import OperationID, UseID
from scar.ir.program.source_productions import ClassBodyInterface


# Ordered source class-build operands, not resolved runtime bases or MRO.
@dataclass(frozen=True, slots=True)
class ClassBaseDeclaration:
    source_class: GlobalPort
    fact: ProviderFactReference
    creator_scope: CodeScopeRef
    bases: tuple[GlobalPort, ...]
    bases_pack: GlobalPort | None
    bases_pack_paths: tuple[AddressPath, ...]
    no_explicit_bases: bool
    conditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ClassBaseSlot:
    owner: int
    ordinal: int
    expression: GlobalPort
    paths: tuple[AddressPath, ...]
    source_classes: tuple[GlobalPort, ...]
    conditions: tuple[str, ...]
    namespace: NamespaceReferenceSlice | None = None


@dataclass(frozen=True, slots=True)
class ClassBaseFrontier:
    """owner indexes declarations; slot indexes slots, or None for the declaration."""
    owner: int
    slot: int | None
    reason: str


@dataclass(frozen=True, slots=True)
class ClassBaseGraph:
    declarations: tuple[ClassBaseDeclaration, ...]
    slots: tuple[ClassBaseSlot, ...]
    frontiers: tuple[ClassBaseFrontier, ...]
    semantics: str = 'ORDERED_SOURCE_BASE_EXPRESSIONS_NOT_RESOLVED_RUNTIME_BASES_OR_MRO'


# Guarded class inputs from original constructor parameter interfaces.
@dataclass(frozen=True, slots=True)
class ConstructorClassInputRoot:
    """entry indexes the unchanged parent ConstructorInterfaceGraph.entries."""
    entry: int
    context: DeclaredContext
    conditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConstructorClassInputFrontier:
    entry: int
    reason: str


@dataclass(frozen=True, slots=True)
class ConstructorClassInputGraph:
    roots: tuple[ConstructorClassInputRoot, ...]
    frontiers: tuple[ConstructorClassInputFrontier, ...]
    semantics: str = 'GUARDED_CONSTRUCTOR_CLASS_COMPATIBILITY_NOT_EXACT_CLASS_ALLOCATION_OR_RUNTIME_INVOCATION'


# Original constructor stages and guarded source-body entry interfaces.
@dataclass(frozen=True, slots=True)
class ConstructorInterfaceSite:
    unit: str
    record: SourceBodyCall


@dataclass(frozen=True, slots=True)
class ConstructorInterfaceStep:
    site: int
    evidence: ProviderFactReference
    link: ConstructionLink


@dataclass(frozen=True, slots=True)
class ConstructorBodyEntry:
    step: int
    connection: FunctionConnection


@dataclass(frozen=True, slots=True)
class ConstructorInterfaceFrontier:
    step: int
    reason: str


@dataclass(frozen=True, slots=True)
class ConstructorInterfaceGraph:
    sites: tuple[ConstructorInterfaceSite, ...]
    steps: tuple[ConstructorInterfaceStep, ...]
    entries: tuple[ConstructorBodyEntry, ...]
    frontiers: tuple[ConstructorInterfaceFrontier, ...]
    semantics: str = 'GUARDED_CONSTRUCTOR_PROTOCOL_NOT_ALLOCATION_OR_RUNTIME_DISPATCH_PROOF'


# Declared contexts point into one shared source-body reference registry.
@dataclass(frozen=True, slots=True)
class ContextBodyReference:
    context: int
    body: int


# Retained callback values, distinct from mutable namespace locations.
@dataclass(frozen=True, slots=True)
class ClassTypeParameterProjection:
    """The selected callback environment supplies its retained tuple.

    ``interface.type_parameters`` retains the original source selector. This
    is neither a lookup of the later class ``__type_params__`` field nor a
    reference alias between the environment and that tuple. Creation and
    activation still have to select the same environment instance.
    """
    operation: OperationID
    interface: ClassBodyInterface
    environment: GlobalPort
    environment_use: UseID
    result: GlobalPort
    world_in: GlobalPort
    world_out: GlobalPort
