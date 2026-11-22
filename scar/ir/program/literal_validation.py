"""Literal validation."""

from scar.ir.literals import PythonLiteral, literal_builtin_type
from scar.ir.program.model import (
    CompletionResult,
    CompletionSuccessor,
    DataValue,
    ExitKind,
    InvokeLiteralResult,
    OperationResult,
    ReferenceValue,
    TerminatorKind,
)
from dataclasses import dataclass
from enum import Enum
from scar.ir.program import normal_graph as T


# Mechanical producer-port checks, without evaluating source or code artifacts.
def literal_site(graph, site):
    """Resolve an original result port, independent of any content attachment."""
    if type(site) not in (OperationResult, InvokeLiteralResult):
        raise TypeError("literal site requires exact OperationResult or InvokeLiteralResult")
    site.__post_init__()
    identity = site.operation if type(site) is OperationResult else site.edge.operation
    operation = next((op for op in graph.operations if op.id == identity), None)
    if operation is None:
        raise ValueError("literal producer operation does not exist")
    values = {value.id: value for value in graph.values}
    successor = argument = successor_index = None
    if type(site) is OperationResult:
        if site.index >= len(operation.results):
            raise ValueError("literal result index is outside original operation results")
        result = values.get(operation.results[site.index])
        if result is None or result.origin != site:
            raise ValueError("literal result does not have the original producer origin")
    else:
        term = operation.terminator
        if term is None or term.kind is not TerminatorKind.INVOKE:
            raise ValueError("literal completion requires an INVOKE operation")
        successor_index = site.edge.successor_index
        if successor_index >= len(term.successors):
            raise ValueError("literal completion successor does not exist")
        successor = term.successors[successor_index]
        if type(successor) is not CompletionSuccessor or successor.kind is not ExitKind.NORMAL:
            raise ValueError("literal completion must select a NORMAL produced slot")
        signature = next((s for s in operation.completions if s.kind is ExitKind.NORMAL), None)
        if signature is None or site.result.index >= len(signature.types):
            raise ValueError("literal completion slot is outside the NORMAL signature")
        positions = [i for i, item in enumerate(successor.arguments)
                     if isinstance(item, CompletionResult) and item.index == site.result.index]
        if len(positions) != 1:
            raise ValueError("literal completion must bind exactly one produced slot")
        block = next((b for region in graph.regions for b in region.blocks
                      if b.id == successor.block), None)
        position = positions[0]
        if block is None or position >= len(block.arguments):
            raise ValueError("literal completion landing formal does not exist")
        argument = successor.arguments[position]
        result = values.get(block.arguments[position])
    if type(result) not in (DataValue, ReferenceValue):
        raise ValueError("literal evaluation requires a DATA or REFERENCE result, never state/protocol/binding")
    return {"operation": operation, "result": result, "successor_index": successor_index,
            "successor": successor, "argument": argument}


def literal_errors(graph):
    for index, row in enumerate(graph.literal_evaluations):
        path = f"literal_evaluations[{index}]"
        try:
            resolved = literal_site(graph, row.site)
        except (TypeError, ValueError) as exc:
            yield "literal_site", path, str(exc)
            continue
        result = resolved["result"]
        if type(row.literal) is PythonLiteral and type(result) is DataValue:
            builtin = literal_builtin_type(row.literal)
            expected = builtin.__module__ + "." + builtin.__qualname__
            if result.data_type != expected:
                yield "literal_type", path, f"known literal content requires {expected}, got {result.data_type}"


# Conditional scalar content fields over existing source ports.
class ScalarEvaluationStatus(str, Enum):
    """Field outcomes, independent of the legacy evaluator implementation."""
    CONSTANT = 'CONSTANT'
    NOT_CONSTANT = 'NOT_CONSTANT'
    NEEDS_CONTRACT = 'NEEDS_CONTRACT'
    KNOWN_EXCEPTION = 'KNOWN_EXCEPTION'
    BUDGET_EXCEEDED = 'BUDGET_EXCEEDED'


class ScalarRuntimeRequirement(str, Enum):
    BUILTIN_RUNTIME_MATCH = 'builtin_runtime_match'
    FLOATING_ENVIRONMENT_MATCH = 'floating_environment_match'


@dataclass(frozen=True, slots=True)
class ScalarPremise:
    port: T.GlobalPort
    literal: PythonLiteral
    requirements: tuple[ScalarRuntimeRequirement, ...] = ()


@dataclass(frozen=True, slots=True)
class ScalarLiteralSource:
    port: T.GlobalPort
    literal: PythonLiteral
    semantics: str = 'SOURCE_LITERAL_MAY_CONTENT_NOT_OBJECT_IDENTITY'


@dataclass(frozen=True, slots=True)
class ScalarReferenceFlow:
    source: T.GlobalPort
    target: T.GlobalPort
    semantics: str = 'MAY_CONTENT_FLOW_NOT_OBJECT_IDENTITY'


@dataclass(frozen=True, slots=True)
class ScalarTransferGroup:
    """A relation factored through an existing source-query table row.

    The group is an analysis relation, not a new program operation or value.
    Original table/row/role retain its applicability and provenance.
    """
    source_table: str
    source_ordinal: int
    role: str
    sources: tuple[T.GlobalPort, ...]
    targets: tuple[T.GlobalPort, ...]
    conditions: tuple[str, ...] = ()
    semantics: str = 'CONDITIONAL_MAY_SCALAR_CONTENT_NOT_VALUE_IDENTITY'


@dataclass(frozen=True, slots=True)
class ScalarDerivation:
    site: T.LanguageOperationSite
    premises: tuple[ScalarPremise, ...]
    result: ScalarPremise
    requirements: tuple[ScalarRuntimeRequirement, ...]
    semantics: str = 'CONDITIONAL_BUILTIN_CONTENT'
    conditions: tuple[str, ...] = (
        'OPERAND_LITERAL_ALTERNATIVES_SELECTED',
        'EXACT_IMMUTABLE_BUILTIN_SIGNATURE',
        'SOURCE_CODE_SCOPE_ADMISSION_REQUIRED',
        'ORIGINAL_NORMAL_COMPLETION_SELECTED',
        'NO_UNIQUE_VALUE_OR_IDENTITY_CLAIM',
    )


class ScalarObligationKind(str, Enum):
    SOURCE_OPEN = 'SOURCE_OPEN'
    UPSTREAM_OPEN = 'UPSTREAM_OPEN'
    OPERAND_OPEN = 'OPERAND_OPEN'
    INPUT_LITERAL_ABSENT = 'INPUT_LITERAL_ABSENT'
    LITERAL_LIMIT = 'LITERAL_LIMIT'
    UNSUPPORTED_OPERATOR = 'UNSUPPORTED_OPERATOR'
    EVALUATION_FAILURE = 'EVALUATION_FAILURE'


@dataclass(frozen=True, slots=True)
class ScalarObligation:
    port: T.GlobalPort
    kind: ScalarObligationKind
    reason: str
    site: T.LanguageOperationSite | None = None
    inputs: tuple[T.GlobalPort, ...] = ()
    premises: tuple[ScalarPremise, ...] = ()
    status: ScalarEvaluationStatus | None = None
    exception: str | None = None
    requirements: tuple[ScalarRuntimeRequirement, ...] = ()


@dataclass(frozen=True, slots=True)
class ScalarSnapshot:
    port: T.GlobalPort
    literals: tuple[PythonLiteral, ...] = ()
    builtin_types: tuple[str, ...] = ()
    content_open: bool = False
    widened: bool = False
    complete: bool = False
    semantics: str = 'CONDITIONAL_BUILTIN_CONTENT'


__all__ = [
    'literal_errors',
    'literal_site',
    'ScalarEvaluationStatus',
    'ScalarRuntimeRequirement',
    'ScalarPremise',
    'ScalarLiteralSource',
    'ScalarReferenceFlow',
    'ScalarTransferGroup',
    'ScalarDerivation',
    'ScalarObligationKind',
    'ScalarObligation',
    'ScalarSnapshot',
]
