"""Native contracts."""

from dataclasses import dataclass
from enum import Enum
from scar.ir.program.ids import LocationID, NativeBoundaryID, OperationID, ValueID
from scar.ir.program.native import EntryPortRef, ExitPortRef, NativeOutputSlot
from scar.ir.program.providers import ProviderFactReference
from scar.ir.program.normal_graph import GlobalPort
from scar.ir.program.python_bindings import PythonBindingID, NativeSchemaReference
from scar.ir.program.target_hardware import TargetResourceRef, ResourceCandidateKind
from scar.ir.program.binding_returns import NativeReturnSlot, ValueShape


# Conditional software/target relationships; never evidence of actual use.
@dataclass(frozen=True,slots=True)
class OperandTargetCandidate:
    boundary: NativeBoundaryID
    port: EntryPortRef
    declaration: ProviderFactReference
    target: TargetResourceRef
    relation: ResourceCandidateKind
    conditions: tuple[str,...]
    selected: bool=False


class OperandSourceKind(str,Enum):
    KEYWORD='KEYWORD'
    POSITIONAL='POSITIONAL'
    DEFAULT_EXPRESSION='DEFAULT_EXPRESSION'
    PACK_DEPENDENCY='PACK_DEPENDENCY'
    RECEIVER_OPEN='RECEIVER_OPEN'
    RECEIVER='RECEIVER'
    CONVERSION_OPEN='CONVERSION_OPEN'


class NormalizationDependencyKind(str,Enum):
    CALL='DECLARED_NORMALIZATION_DEPENDENCY'
    LANGUAGE_PROTOCOL='DECLARED_OPERATOR_NORMALIZATION_DEPENDENCY'


@dataclass(frozen=True,slots=True)
class OperandSource:
    call: GlobalPort
    binding: PythonBindingID
    boundary: NativeBoundaryID
    native_port: EntryPortRef
    parser_parameter: str | None
    actuals: tuple[GlobalPort,...]
    kind: OperandSourceKind
    default_expression: str | None
    conditions: tuple[str,...]
    semantics: NormalizationDependencyKind=NormalizationDependencyKind.CALL


@dataclass(frozen=True,slots=True)
class SoftwareArtifactLocation:
    unit: str
    source_path: str
    resolved_path: str
    endpoint: TargetResourceRef | None
    evidence: ProviderFactReference
    conditions: tuple[str,...]=('PATH_LOCATION_SNAPSHOT_ONLY','NO_RUNTIME_IO_OR_SOURCE_CONTENT_EQUALITY_CLAIM')


# Resource-relevant schema operands and declared dispatch alternatives.
#
# Neither record is a ResourceUse, selected device, RNG effect, allocation,
# transfer or executed kernel. Type projections describe types, not list items.
class ResourceOperandKind(str, Enum):
    TENSOR_ARGUMENT = 'TENSOR_ARGUMENT'
    DEVICE_ARGUMENT = 'DEVICE_ARGUMENT'
    GENERATOR_ARGUMENT = 'GENERATOR_ARGUMENT'


class TypeProjection(str, Enum):
    OPTIONAL_ELEMENT = 'OPTIONAL_ELEMENT'
    LIST_ELEMENT = 'LIST_ELEMENT'


@dataclass(frozen=True, slots=True)
class ResourceOperand:
    boundary: NativeBoundaryID
    port: EntryPortRef
    category: ResourceOperandKind
    schema_type: str
    type_path: tuple[TypeProjection, ...]
    evidence: ProviderFactReference


@dataclass(frozen=True, slots=True)
class DeclaredDispatchCandidate:
    boundary: NativeBoundaryID
    keys: tuple[str, ...]
    symbol: str
    evidence: ProviderFactReference
    conditions: tuple[str, ...] = ('DECLARED_IMPLEMENTATION', 'RUNTIME_DISPATCH_SELECTION_OPEN')


# Local native state declarations; no dispatch or physical alias proof.
class NativeStateCategory(str, Enum):
    TENSOR = 'TENSOR'
    GENERATOR = 'GENERATOR'


class NativePortDirection(str, Enum):
    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'


class NativeStateParameter(str, Enum):
    INVOCATION = 'INVOCATION'
    PORT_VALUE_INSTANCE = 'PORT_VALUE_INSTANCE'
    OPTIONAL_LEAF_PRESENT = 'OPTIONAL_LEAF_PRESENT'
    LIST_ELEMENT_SELECTIONS = 'LIST_ELEMENT_SELECTIONS'


class AliasAnnotationPhase(str, Enum):
    # These name original annotation fields, not execution time. A return's
    # annotation.before can share a label with an input's annotation.before.
    BEFORE = 'BEFORE'
    EXPLICIT_AFTER = 'EXPLICIT_AFTER'


@dataclass(frozen=True, slots=True)
class NativeStateAddress:
    id: LocationID
    boundary: NativeBoundaryID
    slot: EntryPortRef | ExitPortRef
    direction: NativePortDirection
    position: int
    reference: ValueID
    category: NativeStateCategory
    schema_type: str
    type_path: tuple[TypeProjection, ...]
    type_evidence: ProviderFactReference
    alias_before: tuple[str, ...] | None
    declared_may_write: bool | None
    alias_after: tuple[str, ...] | None
    parameters: tuple[NativeStateParameter, ...]
    semantics: str = 'DECLARED_NATIVE_LEAF_FAMILY_NOT_PHYSICAL_TENSOR_OR_STORAGE'


@dataclass(frozen=True, slots=True)
class NativeStateProjection:
    id: ValueID
    address: LocationID
    world: ValueID
    semantics: str = 'SYMBOLIC_NATIVE_STATE_AT_WORLD_NOT_OBSERVED_CONTENT_VERSION'


@dataclass(frozen=True, slots=True)
class NativeStateAccess:
    operation: OperationID
    address: LocationID
    world_before: ValueID | None
    world_normal: ValueID
    world_raised: ValueID | None
    state_before: ValueID | None
    state_normal: ValueID
    state_raised: ValueID | None
    effects_closed: bool = False


@dataclass(frozen=True, slots=True)
class NativeAliasGroup:
    """Factored declaration relation, not a program node or StorageID.

    No explicit after labels means an unspecified annotation field, never an
    established empty post-alias set. Wildcards retain an open alias frontier.
    """
    boundary: NativeBoundaryID
    declaration: ProviderFactReference
    phase: AliasAnnotationPhase
    label: str
    members: tuple[LocationID, ...]
    wildcard: bool


# Conditional source normalization into independent native schema graphs.
@dataclass(frozen=True, slots=True)
class BindingSchemaLink:
    binding: PythonBindingID
    declaration: ProviderFactReference
    native: NativeSchemaReference


@dataclass(frozen=True, slots=True)
class SourceOperandNormalization:
    match: ProviderFactReference
    dependency: OperandSource
    aggregation: str | None = None


@dataclass(frozen=True, slots=True)
class SourceOutputNormalization:
    """Retain both return declarations without a one-to-one data mapping.

    world -> native_world and native_normal_world -> result_world are explicit
    conditional state dependencies. Wrapper pre/post effects remain open;
    these pairs do not establish equality of logical state versions.
    """
    match: ProviderFactReference
    binding: PythonBindingID
    declaration: ProviderFactReference
    boundary: NativeBoundaryID
    result: GlobalPort
    world: GlobalPort
    result_world: GlobalPort
    native_world: EntryPortRef
    native_normal_world: ExitPortRef
    native_outputs: tuple[NativeOutputSlot, ...]
    python_return: ValueShape | None
    native_returns: tuple[NativeReturnSlot, ...]
    conditions: tuple[str, ...]
    frontier: str | None = None
    semantics: str = 'DECLARED_WRAPPING_DEPENDENCY_NOT_NATIVE_PYTHON_OBJECT_OR_STORAGE_IDENTITY'


@dataclass(frozen=True, slots=True)
class SourceNormalizationUnit:
    unit: str
    inputs: tuple[SourceOperandNormalization, ...]
    outputs: tuple[SourceOutputNormalization, ...]
