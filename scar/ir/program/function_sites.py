"""Function sites."""

from dataclasses import dataclass
from enum import Enum
from scar.ir.program import normal_links as F, normal_targets as T, source_productions as P
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.program.source_template_links import SourceTemplateKey, SourceTemplateOrigin
from scar.ir.program.normal_links import CodeScopeRef
from scar.ir.program.normal_targets import CallLink, FunctionActivation, GlobalPort
from scar.ir.program.call_inputs import CallInterfaceRecord, ContextReferenceSlice
from scar.ir.program.functions import BodyEntryInterface, BodyLoweringProfile, BindingCondition
from scar.ir.program.ids import EnvironmentInterfaceID, OperationID, UseID, ValueID
from scar.ir.v2.common import SourceReference
from scar.ir.program.binding_ports import ConsumerBodyEntry
from scar.ir.program.source_value_routes import SourceBodyValueFlow, SourceDataReach


# Source factory declarations, distinct from selected runtime callables.
class FunctionFactoryKind(str, Enum):
    BUILD_FUNCTION = 'BUILD_FUNCTION'
    MAKE_LAMBDA_FUNCTION = 'MAKE_LAMBDA_FUNCTION'
    MAKE_GENEXPR_FUNCTION = 'MAKE_GENEXPR_FUNCTION'
    MAKE_NESTED_FUNCTION = 'MAKE_NESTED_FUNCTION'


class AnnotationOperandKind(str, Enum):
    ANNOTATE_CALLABLE = 'ANNOTATE_CALLABLE'
    ANNOTATION_CODE = 'ANNOTATION_CODE'


@dataclass(frozen=True, slots=True)
class FunctionDefaultCapture:
    """Actual factory input and original expression stay separate.

    A generic factory may forward the expression through a parameter. Neither
    endpoint denotes the current __defaults__/__kwdefaults__ slot or proves its
    present contents. ``parameter`` indexes DeferredEnvironment.parameters.
    """
    parameter: int
    default_source: P.SlotRef
    operand_ordinal: int
    operand: T.GlobalPort
    expression: T.GlobalPort | None


@dataclass(frozen=True, slots=True)
class FunctionCellCapture:
    """Retained cell handle, never a snapshot or alias of the cell contents."""
    relay: P.CaptureReference
    operand_ordinal: int
    operand: T.GlobalPort


@dataclass(frozen=True, slots=True)
class SourceFunctionCreation:
    function_fact: ProviderFactReference
    factory: FunctionFactoryKind
    origin: SourceTemplateOrigin
    site: T.GlobalPort
    function: T.GlobalPort
    code: T.GlobalPort
    body: T.Body
    body_scope: F.CodeScopeWitness
    creator: F.CodeScopeWitness
    environment: P.DeferredEnvironment
    globals: T.GlobalPort
    defaults: tuple[FunctionDefaultCapture, ...]
    captures: tuple[FunctionCellCapture, ...]
    capture_transports: tuple[P.CaptureTransport, ...]
    annotation: T.GlobalPort | None
    annotation_kind: AnnotationOperandKind | None
    annotations: P.FunctionAnnotationInterface | None
    world_before: T.GlobalPort
    world_after: T.GlobalPort
    # MAKE_GENEXPR_FUNCTION has five original reference outputs. Their ordered
    # ports are retained without declaring them reference aliases of inputs.
    factory_results: tuple[T.GlobalPort, ...]
    conditions: tuple[str, ...] = ('SOURCE_CREATOR_CONTEXT_AND_FACTORY_NORMAL_COMPLETION_SELECTED',)


@dataclass(frozen=True, slots=True)
class FunctionCreationObligation:
    function_fact: ProviderFactReference
    function: T.GlobalPort
    template: SourceTemplateKey
    site: T.GlobalPort | None
    rule: str


@dataclass(frozen=True, slots=True)
class SourceFunctionCreationGraph:
    unit: str
    source_templates: ProviderArtifact
    templates: tuple[SourceTemplateKey, ...]
    creations: tuple[SourceFunctionCreation, ...]
    obligations: tuple[FunctionCreationObligation, ...]
    runtime_function_instances_proven: bool = False
    current_defaults_proven: bool = False
    source_scopes_activated: bool = False
    effects_closed: bool = False
    loops_unrolled: bool = False
    target_executed: bool = False
    first_stage_complete: bool = False
    semantics: str = 'SOURCE_FACTORY_OPERANDS_AND_CELL_HANDLES_NOT_CURRENT_CALLABLE_OR_CONTENT_VERSIONS'


# Original source-function candidates connected to conditional call ABIs.
@dataclass(frozen=True, slots=True)
class SourceFunctionInterface:
    query: int
    origin: int
    call: ProviderFactReference
    definition: ProviderFactReference
    reference: ProviderFactReference
    activation: FunctionActivation
    link: CallLink


@dataclass(frozen=True, slots=True)
class SourceFunctionInterfaceFrontier:
    query: int
    origin: int | None
    reference: ProviderFactReference
    reason: str


@dataclass(frozen=True, slots=True)
class SourceFunctionInterfaceQuery:
    """Indexes original declarations, including calls with no function target.

    ``declared`` and ``reference`` identify complete parent query records.
    Frontier ordinals index their respective NamespaceReferenceSlice.frontiers;
    interface ordinals index this unit's interfaces table. Owners are source
    code scopes, never evidence of their activation or of a caller frame.
    """
    query: int
    call: ProviderFactReference
    declared: ProviderFactReference
    reference: ProviderFactReference
    owners: tuple[CodeScopeRef, ...]
    function_origins: tuple[int, ...]
    interfaces: tuple[int, ...]
    declared_frontiers: tuple[int, ...]
    reference_frontiers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SourceFunctionInterfaceUnit:
    unit: str
    calls: int
    function_origins: int
    controlled_queries: tuple[int, ...]
    interfaces: tuple[SourceFunctionInterface, ...]
    frontiers: tuple[SourceFunctionInterfaceFrontier, ...]
    queries: tuple[SourceFunctionInterfaceQuery, ...] = ()
    semantics: str = 'DECLARED_SOURCE_FUNCTION_PARAMETER_AND_RETURN_INTERFACES_NOT_SCOPE_ADMISSION'


# Returned callable provenance; creation and invocation contexts are separate.
@dataclass(frozen=True, slots=True)
class ReturnedCallableProvenance:
    query: int
    cut: int
    declared: ProviderFactReference
    seed_interface: CallInterfaceRecord
    return_source: GlobalPort
    reference: ContextReferenceSlice
    origin: int
    creation: SourceFunctionCreation
    creator_interfaces: tuple[CallInterfaceRecord, ...]
    base_scopes: tuple[CodeScopeRef, ...]
    conditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReturnedCallableFrontier:
    query: int
    cut: int
    declared: ProviderFactReference
    seed_interface: CallInterfaceRecord | None
    return_source: GlobalPort | None
    reference: ContextReferenceSlice | None
    reason: str


# Source-declared Lambda creation/capture and expression-body interfaces.
#
# These records connect existing ProgramGraph ports. They neither execute a
# body at creation nor establish a runtime callable, alias or captured value.
class LambdaEnvironmentField(str, Enum):
    """Initialized by the original creation, not a target-program field read."""
    BUILTINS = '__builtins__'
    DEFAULTS = '__defaults__'
    KEYWORD_DEFAULTS = '__kwdefaults__'


@dataclass(frozen=True, slots=True)
class LambdaCapture:
    formal: ValueID
    source_use: UseID | None = None
    normal_result: int | None = None
    created_field: LambdaEnvironmentField | None = None

    def __post_init__(self):
        if type(self.formal) is not ValueID:
            raise TypeError('Lambda capture formal must be its original entry ValueID')
        if (sum(x is not None for x in (self.source_use,self.normal_result,self.created_field)) != 1
                or self.source_use is not None and type(self.source_use) is not UseID
                or self.normal_result is not None and (type(self.normal_result) is not int or self.normal_result < 0)
                or self.created_field is not None and type(self.created_field) is not LambdaEnvironmentField):
            raise ValueError('Lambda capture requires one exact creation Use, NORMAL result or initialized-field summary')


@dataclass(frozen=True, slots=True)
class LambdaBodyDefinition:
    creation: OperationID
    source: SourceReference
    code_constant: OperationID
    body: BodyEntryInterface
    environment: EnvironmentInterfaceID
    captures: tuple[LambdaCapture, ...]
    conditions: tuple[BindingCondition, ...] = tuple(BindingCondition)

    def __post_init__(self):
        if (type(self.creation) is not OperationID or type(self.code_constant) is not OperationID
                or type(self.source) is not SourceReference or type(self.body) is not BodyEntryInterface
                or self.body.profile is not BodyLoweringProfile.STRUCTURED_ENV
                or type(self.environment) is not EnvironmentInterfaceID):
            raise TypeError('Lambda definition requires original typed creation/source/body/environment ports')
        if (type(self.captures) is not tuple or any(type(c) is not LambdaCapture for c in self.captures)
                or len({c.formal for c in self.captures}) != len(self.captures)):
            raise ValueError('Lambda captures must name unique original entry references')
        if self.conditions != tuple(BindingCondition):
            raise ValueError('Lambda entry keeps every ordinary source/call/default/frame binding condition unproved')


# Source body references shared across caller contexts and compilation forms.
@dataclass(frozen=True, slots=True)
class SharedBodyReference:
    source_function: GlobalPort
    body: ProviderFactReference


@dataclass(frozen=True, slots=True)
class BodyExpansionFrontier:
    binding: int
    reason: str


@dataclass(frozen=True, slots=True)
class SourceBodyExpansion:
    """entries/reaches.body index references, not the new-bodies table.

    references may select original parent value-flow bodies. Only additional
    bodies are stored locally; bindings index the original interface parent.
    """
    bodies: tuple[SourceBodyValueFlow, ...]
    references: tuple[SharedBodyReference, ...]
    entries: tuple[ConsumerBodyEntry, ...]
    reaches: tuple[SourceDataReach, ...]
    frontiers: tuple[BodyExpansionFrontier, ...]


__all__ = [
    'LambdaCapture',
    'LambdaBodyDefinition',
    'LambdaEnvironmentField',
    'FunctionFactoryKind',
    'AnnotationOperandKind',
    'FunctionDefaultCapture',
    'FunctionCellCapture',
    'SourceFunctionCreation',
    'FunctionCreationObligation',
    'SourceFunctionCreationGraph',
    'SourceFunctionInterface',
    'SourceFunctionInterfaceFrontier',
    'SourceFunctionInterfaceQuery',
    'SourceFunctionInterfaceUnit',
    'ReturnedCallableProvenance',
    'ReturnedCallableFrontier',
    'SharedBodyReference',
    'BodyExpansionFrontier',
    'SourceBodyExpansion',
]
