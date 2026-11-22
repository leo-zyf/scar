"""Source protocols."""

from dataclasses import dataclass
from scar.ir.program.declared_contexts import DeclaredContext
from scar.ir.program.providers import ProviderFactReference
from scar.ir.program.source_value_routes import SourceBodyCall
from scar.ir.program.ids import LocationID, ValueID
from scar.ir.program.normal_targets import ConstructionPort, Endpoint, GlobalPort
from scar.ir.program import (
    suspension_interfaces as S,
    normal_links as F,
    normal_targets as T,
    source_productions as P,
)
from scar.ir.program.source_template_links import SourceTemplateKey
from scar.ir.program.suspension_interfaces import StepWitness
from scar.ir.literals import PythonLiteral
from enum import Enum
from scar.ir.program.python_bindings import SourceCallSite


# Original unresolved target fields, attached to existing program nodes.
@dataclass(frozen=True, slots=True)
class ProtocolCallFrontier:
    ordinal: int
    context: DeclaredContext
    call: SourceBodyCall | None
    evidence: ProviderFactReference
    reason: str


# Symbolic Python protocol interfaces, without physical object identities.
@dataclass(frozen=True, slots=True)
class ConstructionInterfaceValue:
    id: ValueID
    endpoint: ConstructionPort
    instance_parameters: tuple[str, ...] = ('CALLER_ACTIVATION', 'CONSTRUCTION_OCCURRENCE')


@dataclass(frozen=True, slots=True)
class ProtocolCallableSelection:
    id: ValueID
    callee: GlobalPort
    definition: GlobalPort
    lookup_world: Endpoint
    selection: ProviderFactReference
    protocol: str
    instance_parameters: tuple[str, ...] = ('CALLER_ACTIVATION', 'LOOKUP_OCCURRENCE', 'CURRENT_DESCRIPTOR_VERSION')


@dataclass(frozen=True, slots=True)
class ProtocolDefaultSlot:
    id: LocationID
    callable: ValueID
    definition: GlobalPort
    parameter_index: int
    parameter_name: str
    initial_source: GlobalPort


@dataclass(frozen=True, slots=True)
class ProtocolDefaultState:
    id: ValueID
    address: LocationID
    world: Endpoint
    instance_parameters: tuple[str, ...] = ('CALLER_ACTIVATION', 'LOOKUP_OCCURRENCE', 'DEFAULT_MUTATION_HISTORY')


# Local interfaces projected from sealed F4/F5 inputs, not runtime facts.
@dataclass(frozen=True, slots=True)
class IterationInterface:
    fact: int
    site: S.IterationSite


@dataclass(frozen=True, slots=True)
class DeferredInterface:
    fact: int
    site: S.DeferredBodySite


@dataclass(frozen=True, slots=True)
class MappingMergeInterface:
    witness: S.StepWitness
    callee: GlobalPort
    target: GlobalPort
    source: GlobalPort
    world: GlobalPort
    completions: tuple[S.CompletionPorts, ...]
    target_index: int | None
    source_index: int | None


@dataclass(frozen=True, slots=True)
class MappingTemplateInterface:
    template: SourceTemplateKey
    plan_sha256: str
    straight_line: bool
    merges: tuple[MappingMergeInterface, ...]


@dataclass(frozen=True, slots=True)
class UnitSourceInterfaces:
    unit: str
    iterations: tuple[IterationInterface, ...]
    bodies: tuple[DeferredInterface, ...]
    mappings: tuple[MappingTemplateInterface, ...]


# Static deferred-object and iterator-driver relations, not runtime frames.
@dataclass(frozen=True, slots=True)
class DeferredOrigin:
    result: T.Endpoint | None
    source_callable: T.GlobalPort
    receiver: T.Endpoint | None
    class_receiver: T.GlobalPort | None
    world: T.Endpoint
    environment: tuple[T.Endpoint, ...] = ()


@dataclass(frozen=True, slots=True)
class DeferredCreation:
    origin: DeferredOrigin
    kind: P.DeferredBodyKind
    body_scope: F.CodeScopeWitness
    connection: T.FunctionConnection
    conditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GeneratorIterator:
    origin: DeferredOrigin
    iteration: StepWitness
    result: T.GlobalPort
    conditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GeneratorYield:
    suspension: StepWitness
    exported: T.GlobalPort
    item: T.GlobalPort
    world: T.GlobalPort
    result_world: T.GlobalPort
    sent: T.GlobalPort
    resume_world: T.GlobalPort
    sent_literal: PythonLiteral
    conditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GeneratorExhaustion:
    value: T.GlobalPort | None
    world: T.GlobalPort | None
    exception: T.GlobalPort | None
    result_world: T.GlobalPort | None
    # A protocol transformation, never a value-to-item or value-to-exception alias.
    semantics: str = 'NORMAL_RETURN_AS_STOP_ITERATION_VALUE'
    conditions: tuple[str, ...] = ('GENERATOR_COMPLETES_NORMALLY',)


@dataclass(frozen=True, slots=True)
class GeneratorAdvance:
    origin: DeferredOrigin
    iteration: StepWitness
    connection: T.FunctionConnection
    yielded: tuple[GeneratorYield, ...]
    exhaustion: GeneratorExhaustion | None
    conditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DeferredDriverObligation:
    origin: DeferredOrigin
    iteration: StepWitness | None
    reason: str
    site: T.Endpoint | None = None
    conditions: tuple[str, ...] = ('DRIVER_OR_INTERFACE_NOT_CONNECTED',)


# Conditional standard-builtin boundaries over original source call ports.
class SourceBuiltin(str, Enum):
    COMPILE = 'compile'
    EXEC = 'exec'


class BuiltinArgumentOrigin(str, Enum):
    SOURCE_VALUE = 'SOURCE_VALUE'
    SOURCE_LITERAL = 'SOURCE_LITERAL'
    LANGUAGE_DEFAULT = 'LANGUAGE_DEFAULT'
    OPEN_PACK = 'OPEN_PACK'


@dataclass(frozen=True, slots=True)
class BuiltinArgument:
    name: str
    origin: BuiltinArgumentOrigin
    source: T.GlobalPort | None
    literal: PythonLiteral | None = None
    literal_occurrence: P.ProductionOccurrence | None = None


@dataclass(frozen=True, slots=True)
class PackConstructionDependency:
    source: T.GlobalPort
    kind: F.ContainerKind


@dataclass(frozen=True, slots=True)
class SourceBuiltinInvocation:
    builtin: SourceBuiltin
    site: SourceCallSite
    owner: F.CodeScopeRef
    parameters: tuple[BuiltinArgument, ...]
    binding: T.ArgumentBinding
    pack_dependencies: tuple[PackConstructionDependency, ...]
    signature: str
    binding_note: str = ''
    normal_result: PythonLiteral | None = None
    conditions: tuple[str, ...] = (
        'SOURCE_SCOPE_AND_NORMAL_COMPLETION_SELECTED',
        'STANDARD_BUILTIN_REFERENCE_SELECTED',
        'CURRENT_CODE_AND_NAMESPACE_VERSIONS_SELECTED',
        'CALL_SIGNATURE_AND_VALUE_TYPE_ACCEPTANCE',
        'SOURCE_PACK_CONSTRUCTION_DEPENDENCIES_NOT_INVALIDATED',
    )


@dataclass(frozen=True, slots=True)
class SourceCompilationRequest:
    invocation_result: T.GlobalPort
    caller: F.CodeScopeRef
    requested_flags: int | None
    dont_inherit: bool | None
    requested_optimize: int | None
    process_optimize: int | None
    inherited_future_flags: int | None
    effective_optimize: int | None
    result_kind: str
    source_future_flags: int | None = None
    effective_flags: int | None = None
    obligations: tuple[str, ...] = (
        'SOURCE_PAYLOAD_VALUE_AND_TEXT_INTERPRETATION_NOT_RESOLVED',
        'COMPILED_CONTEXT_HAS_NO_MATCHED_F4_BODY_VARIANT',
    )


@dataclass(frozen=True, slots=True)
class CompiledCodeUse:
    producer: T.GlobalPort
    consumer: T.GlobalPort
    exec_result: T.GlobalPort
    semantics: str = 'CONDITIONAL_MAY_CODE_REFERENCE_FLOW'
    namespace_rule: str = 'STANDARD_EXEC_GLOBALS_LOCALS_AND_CALLER_FRAME_RULES'
    conditions: tuple[str, ...] = (
        'STANDARD_BUILTIN_REFERENCE_SELECTED',
        'SOURCE_NAMESPACE_READ_VERSION_SELECTED',
        'CODE_OBJECT_TYPE_ACCEPTANCE',
        'NO_SOURCE_BODY_ACTIVATION_OR_REFERENCE_IDENTITY_PROOF',
    )
