"""Layout templates."""

from dataclasses import dataclass
from enum import Enum
from scar.ir.program.ids import ValueID
from scar.ir.literals import PythonLiteral
from scar.ir.program.normal_graph import GlobalPort, LanguageOperationSite
from scar.ir.program.providers import ProviderFactReference
from scar.ir.program.python_bindings import PythonBindingID, SourceCallSite
from scar.ir.program.normal_graph import LanguageOperationKind, OperatorSelector
from scar.ir.program.binding_returns import (
    ValueShapeKind,
    ValueShape,
    SchemaType,
    NativeReturnSlot,
    ReceiverMode,
)


# Parametric contents and required materializations in one array region.
#
# These are definition-time templates over the region's input versions, never
# runtime LogicalValueID, StorageID or InvocationID. Equal content templates
# do not permit sharing mutable result objects or reusing results across calls.
class ArrayMaterializationRole(str, Enum):
    INPUT_REFERENCE = 'INPUT_REFERENCE'
    IMMUTABLE_SCALAR = 'IMMUTABLE_SCALAR'
    VIEW_DESCRIPTOR = 'VIEW_DESCRIPTOR'
    FRESH_ARRAY_RESULT = 'FRESH_ARRAY_RESULT'


@dataclass(frozen=True, slots=True)
class ArrayValueTemplate:
    expression: str
    value: ValueID
    content_template: str
    materialization: ArrayMaterializationRole
    operand_expressions: tuple[str, ...]
    optimized_expression: str | None
    source_origin_ordinals: tuple[int, ...]
    required_fresh_output: bool


# Conditional return/projection types, distinct from objects and storage.
class ProjectionKind(str,Enum):
    INDEX='INDEX'
    ELEMENT='ELEMENT'
    OPTIONAL='OPTIONAL'
    ALTERNATIVE='ALTERNATIVE'


@dataclass(frozen=True,slots=True)
class ReturnProjection:
    kind: ProjectionKind
    index: int | None=None


@dataclass(frozen=True,slots=True)
class ReturnTypeEvidence:
    origin: GlobalPort
    target: GlobalPort
    projection: tuple[ReturnProjection,...]
    shape: ValueShapeKind
    source: ProviderFactReference
    conditions: tuple[str,...]
    identity_proven: bool=False


@dataclass(frozen=True,slots=True)
class DeclaredMemberCall:
    site: SourceCallSite
    receiver: GlobalPort | None
    member: str
    declaration: ProviderFactReference
    conditions: tuple[str,...]
    native_selector: str | None=None
    receiver_mode: ReceiverMode=ReceiverMode.BOUND


@dataclass(frozen=True,slots=True)
class DeclaredProtocolCall:
    receiver: GlobalPort
    member: str
    argument: GlobalPort | PythonLiteral
    result: GlobalPort | None
    world: GlobalPort
    result_world: GlobalPort
    declaration: ProviderFactReference
    conditions: tuple[str,...]
    assigned: GlobalPort | None=None


class OperatorBranch(str,Enum):
    FORWARD='FORWARD'
    REFLECTED_SUBTYPE_PRIORITY='REFLECTED_SUBTYPE_PRIORITY'
    REFLECTED_FALLBACK='REFLECTED_FALLBACK'
    INPLACE='INPLACE'
    UNARY='UNARY'
    CONTAINS='CONTAINS'
    TRUTH_BOOL='TRUTH_BOOL'
    TRUTH_LENGTH='TRUTH_LENGTH'


class OperatorResultConversion(str,Enum):
    DIRECT='DIRECT'
    TRUTH_VALUE='TRUTH_VALUE'
    NEGATED_TRUTH_VALUE='NEGATED_TRUTH_VALUE'


@dataclass(frozen=True,slots=True)
class DeclaredOperatorCall:
    action: LanguageOperationKind
    selector: OperatorSelector
    operands: tuple[GlobalPort,...]
    receiver: GlobalPort
    arguments: tuple[GlobalPort,...]
    result: GlobalPort
    world: GlobalPort
    result_world: GlobalPort
    member: str
    branch: OperatorBranch
    declaration: ProviderFactReference
    conditions: tuple[str,...]
    result_conversion: OperatorResultConversion=OperatorResultConversion.DIRECT
    native_selector: str | None=None
    binding: PythonBindingID | None=None


class LanguageFrontierReason(str,Enum):
    TENSOR_METHOD_DECLARATION_UNAVAILABLE='TENSOR_METHOD_DECLARATION_UNAVAILABLE'
    TENSOR_ALIAS_TARGET_UNRESOLVED='TENSOR_ALIAS_TARGET_UNRESOLVED'
    LANGUAGE_OPERATION_ARITY_DIFFERS='LANGUAGE_OPERATION_ARITY_DIFFERS'


@dataclass(frozen=True,slots=True)
class LanguageProtocolFrontier:
    site: LanguageOperationSite
    member: str | None
    branch: OperatorBranch | None
    reason: LanguageFrontierReason


__all__ = [
    'ArrayMaterializationRole',
    'ArrayValueTemplate',
    'ProjectionKind',
    'ReturnProjection',
    'ReturnTypeEvidence',
    'DeclaredMemberCall',
    'DeclaredProtocolCall',
    'OperatorBranch',
    'OperatorResultConversion',
    'DeclaredOperatorCall',
    'LanguageFrontierReason',
    'LanguageProtocolFrontier',
]
