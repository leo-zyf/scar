"""Source observations."""

from dataclasses import dataclass
from enum import Enum
from scar.ir.effects import Knowledge
from scar.ir.program.ids import ProviderID
from scar.ir.program.normal_graph import CodeScopeWitness
from scar.ir.program.normal_graph import GlobalPort, Body, FunctionConnection
from scar.ir.program.providers import ProviderFactReference, ProviderArtifact
from scar.ir.program.call_edges import (
    CallRecordAnchor,
    CapturedCode,
    CapturedSite,
    InvocationKey,
    CodeCapture,
)
from scar.ir.program.python_call_targets import (
    PythonCallTarget,
    PythonTargetKind,
    PythonEntryBinding,
)
from scar.ir.program.native_callables import NativeCallableFact
from scar.ir.program.call_operands import OperandCallAttempt
from scar.ir.program.source_productions import EvaluationRequestEnvironment, ProductionStepID
from scar.ir.program.compiler_symbols import CompilerCodePath


# Shared source CALL interfaces and separate recorded dispatch occurrences.
#
# An interface describes original Python-side ports. It is neither a callee
# body nor a native signature, constructor stage, completion or effect proof.
@dataclass(frozen=True, slots=True)
class SourceObservedBoundary:
    id: ProviderID
    unit: str
    query: int
    call: ProviderFactReference
    owner: CodeScopeWitness
    target_kind: PythonTargetKind
    callee: GlobalPort
    positional: tuple[GlobalPort, ...]
    keywords: tuple[tuple[str, GlobalPort], ...]
    positional_pack: GlobalPort | None
    keyword_pack: GlobalPort | None
    world: GlobalPort
    result: GlobalPort
    result_world: GlobalPort
    effects: Knowledge = Knowledge.UNKNOWN


@dataclass(frozen=True, slots=True)
class SourceObservedProtocolBoundary:
    """Original protocol ports and shared body; never a Python-formal ABI."""
    id: ProviderID
    unit: str
    kind: str
    definition: ProviderFactReference
    operation: GlobalPort
    arguments: tuple[GlobalPort, ...]
    result: GlobalPort
    world: GlobalPort
    result_world: GlobalPort
    body: Body
    owner: CodeScopeWitness
    effects: Knowledge = Knowledge.UNKNOWN

    def __post_init__(self):
        if (self.kind not in ('CREATE_GENEXPR', 'BUILD_CLASS')
                or self.owner.owner.unit != self.unit or self.body.unit != self.unit
                or any(port.unit != self.unit for port in
                    (self.operation, *self.arguments, self.result, self.world, self.result_world))):
            raise ValueError('protocol boundary needs its original unit, operation and body')


@dataclass(frozen=True, slots=True)
class ObservedBoundaryContinuation:
    """A later CALL witnessed in the same original frame after a normal continuation."""
    successor: CallRecordAnchor
    site: CapturedSite
    hook_epoch: int

    def __post_init__(self):
        if (type(self.successor) is not CallRecordAnchor or type(self.successor.index) is not int
                or type(self.site) is not CapturedSite
                or self.site.opname not in ('CALL', 'CALL_KW', 'CALL_FUNCTION_EX')
                or type(self.hook_epoch) is not int or self.hook_epoch < 1):
            raise ValueError('boundary continuation requires its original typed CALL successor')


@dataclass(frozen=True, slots=True)
class SourceBoundaryObservation:
    boundary: ProviderID
    source_observation: ProviderID
    attempt: CallRecordAnchor
    target: PythonCallTarget
    target_anchor: CallRecordAnchor
    profile_alternatives: tuple[ProviderFactReference, ...]
    native: NativeCallableFact | None = None
    native_anchor: CallRecordAnchor | None = None
    normal_continuation: ObservedBoundaryContinuation | None = None

    def __post_init__(self):
        continuation = self.normal_continuation
        if continuation is not None and (type(continuation) is not ObservedBoundaryContinuation
                or type(self.attempt.index) is not int or continuation.successor.path != self.attempt.path
                or continuation.successor.index <= self.attempt.index):
            raise ValueError('normal continuation must follow its original attempt in the same observation stream')


class SourceCallbackBodyKind(str, Enum):
    NO_SOURCE = 'NO_SOURCE'
    FUNCTION = 'FUNCTION'
    EVALUATION_REQUEST = 'EVALUATION_REQUEST'


@dataclass(frozen=True, slots=True)
class SourceBoundaryCallback:
    """One shared callback code/interface, not one frame per iteration.

    A suspending code does not identify its generator instance or resume
    position. Missing source-body correspondence never erases the known code.
    For EVALUATION_REQUEST, ``formals`` are the original region's environment,
    format and world inputs, not a Python-function parameter binder.
    """
    id: ProviderID
    boundary: ProviderID
    code: CapturedCode
    suspending: bool
    body_kind: SourceCallbackBodyKind
    body: Body | None
    definition: ProviderFactReference | None
    scope: CodeScopeWitness | None
    formals: tuple[GlobalPort, ...]
    result: GlobalPort | None
    result_world: GlobalPort | None

    def __post_init__(self):
        if type(self.body_kind) is not SourceCallbackBodyKind or type(self.suspending) is not bool:
            raise ValueError('callback needs an explicit body protocol and suspension state')
        if self.body_kind is SourceCallbackBodyKind.NO_SOURCE:
            if (any(value is not None for value in (self.body, self.definition, self.scope,
                    self.result, self.result_world)) or self.formals):
                raise ValueError('source-free callback cannot carry a source ABI')
            return
        if (self.body is None or self.definition is None or self.scope is None
                or self.scope.owner.unit != self.body.unit
                or (self.scope.template, self.scope.region) != (self.body.template, self.body.region)
                or any(port.unit != self.body.unit for port in
                    (*self.formals, *((self.result,) if self.result is not None else ()),
                        *((self.result_world,) if self.result_world is not None else ())))):
            raise ValueError('callback source ABI must retain its original body and scope')
        if self.body_kind is SourceCallbackBodyKind.EVALUATION_REQUEST and (
                self.suspending or len(self.formals) != 3
                or self.result is None or self.result_world is None):
            raise ValueError('evaluation-request callback needs its three inputs and two return ports')


@dataclass(frozen=True, slots=True)
class SourceCallbackObservation:
    callback: ProviderID
    source_observation: ProviderID
    invocation: InvocationKey
    entry: CallRecordAnchor
    exit: CallRecordAnchor | None
    code_capture: CallRecordAnchor


# Observed code templates corresponding to declared source code scopes.
@dataclass(frozen=True, slots=True)
class ObservedSourceCodeLink:
    id: ProviderID
    function_fact: ProviderFactReference
    function: GlobalPort
    body: Body
    scope: CodeScopeWitness
    captures: tuple[CodeCapture, ...]
    targets: tuple[tuple[CallRecordAnchor, PythonCallTarget], ...]
    entries: tuple[tuple[CallRecordAnchor, PythonEntryBinding], ...]
    source: ProviderArtifact
    syntax: ProviderFactReference
    compiled_root: CapturedCode
    constant_path: tuple[int, ...]
    compile_flags: int
    semantics: str = 'OBSERVED_CODE_TO_DECLARED_BODY_NOT_CREATOR_GLOBALS_OR_CALLSITE_SELECTION'


class ObservedSourceScopeKind(str, Enum):
    CODE_ROOT = 'CODE_ROOT'
    CLASS_BODY = 'CLASS_BODY'


@dataclass(frozen=True, slots=True)
class ObservedSourceScopeLink:
    """A code region, without a Python-function value or argument binder.

    Class execution and exec/eval roots have their own namespace protocols.
    Their matching code cannot be treated as a fresh function invocation.
    """
    id: ProviderID
    kind: ObservedSourceScopeKind
    source_fact: ProviderFactReference
    body: Body
    scope: CodeScopeWitness
    captures: tuple[CodeCapture, ...]
    source: ProviderArtifact
    syntax: ProviderFactReference
    compiled_root: CapturedCode
    constant_path: tuple[int, ...]
    compile_flags: int
    frame_entries: tuple[CallRecordAnchor, ...]
    semantics: str = 'OBSERVED_CODE_TO_SCOPE_NOT_EXECUTION_NAMESPACE_OR_CREATION_IDENTITY'

    def __post_init__(self):
        if (type(self.kind) is not ObservedSourceScopeKind or not self.captures
                or self.scope.owner.unit != self.body.unit
                or (self.scope.template, self.scope.region) != (self.body.template, self.body.region)
                or type(self.constant_path) is not tuple
                or any(type(v) is not int or v < 0 for v in self.constant_path)
                or type(self.compile_flags) is not int or self.compile_flags < 0):
            raise ValueError('code-scope correspondence needs its original body, captures and compiler path')


class ObservedSourceRequestKind(str, Enum):
    FUNCTION_ANNOTATIONS = 'FUNCTION_ANNOTATIONS'
    CLASS_ANNOTATIONS = 'CLASS_ANNOTATIONS'
    MODULE_ANNOTATIONS = 'MODULE_ANNOTATIONS'


@dataclass(frozen=True, slots=True)
class ObservedSourceRequestCodeLink:
    """Code correspondence to a sealed request region, without a function fact.

    Declaration evidence addresses an original recipe, never a Fact table.
    Native ``format`` and creator/globals/cell bindings remain conditional.
    """
    id: ProviderID
    kind: ObservedSourceRequestKind
    declaration: ProviderFactReference
    declaration_step: ProductionStepID
    plan_sha256: str
    callable: GlobalPort
    body: Body
    scope: CodeScopeWitness
    environment: EvaluationRequestEnvironment
    captures: tuple[CodeCapture, ...]
    source: ProviderArtifact
    syntax: ProviderFactReference
    compiled_root: CapturedCode
    constant_path: tuple[int, ...]
    creator_constant_path: tuple[int, ...]
    load_offset: int
    make_offset: int
    installation_offset: int
    compile_flags: int
    frame_entries: tuple[CallRecordAnchor, ...]
    semantics: str = 'OBSERVED_CODE_TO_DEFERRED_REQUEST_NOT_FORMAT_CREATOR_OR_ENVIRONMENT_BINDING'

    def __post_init__(self):
        if (type(self.kind) is not ObservedSourceRequestKind or not self.captures
                or self.scope.owner.unit != self.body.unit or self.callable.unit != self.body.unit
                or (self.scope.template, self.scope.region) != (self.body.template, self.body.region)
                or self.environment.scope != self.scope.source_scope
                or len(self.plan_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.plan_sha256)
                or any(type(path) is not tuple or any(type(v) is not int or v < 0 for v in path)
                    for path in (self.constant_path, self.creator_constant_path))
                or not self.constant_path or self.constant_path[:-1] != self.creator_constant_path
                or any(type(v) is not int or v < 0 for v in
                    (self.load_offset, self.make_offset, self.installation_offset, self.compile_flags))
                or not self.load_offset < self.make_offset < self.installation_offset):
            raise ValueError('request correspondence needs its sealed declaration/body and exact compiler creation')


@dataclass(frozen=True, slots=True)
class ObservedSourceCompilerCodeLink:
    """A compiler-generated body's original recipe, without a function fact.

    Generic factories, lazy value evaluators and other C3 bodies use their
    actual entry protocol. This does not fabricate Python argument binding.
    """
    id: ProviderID
    origin: CompilerCodePath
    entry_recipe: ProviderFactReference
    entry_step: ProductionStepID
    plan_sha256: str
    body: Body
    scope: CodeScopeWitness
    captures: tuple[CodeCapture, ...]
    source: ProviderArtifact
    syntax: ProviderFactReference
    compiled_root: CapturedCode
    compile_flags: int
    frame_entries: tuple[CallRecordAnchor, ...]
    semantics: str = 'OBSERVED_COMPILER_BODY_NOT_CALLABLE_INSTANCE_OR_ENVIRONMENT_BINDING'

    @property
    def constant_path(self):
        return self.origin.constant_path

    def __post_init__(self):
        if (not self.captures or self.scope.source_scope != self.origin.source_scope
                or self.scope.owner.unit != self.body.unit
                or (self.scope.template,self.scope.region) != (self.body.template,self.body.region)
                or len(self.plan_sha256) != 64
                or any(c not in '0123456789abcdef' for c in self.plan_sha256)
                or type(self.compile_flags) is not int or self.compile_flags < 0):
            raise ValueError('compiler body requires an original recipe, scope and complete captures')


ObservedSourceBodyLink = (ObservedSourceCodeLink | ObservedSourceScopeLink
    | ObservedSourceRequestCodeLink | ObservedSourceCompilerCodeLink)


# Real call occurrences attached to original F4 callsite declarations.
@dataclass(frozen=True, slots=True)
class ObservedSourceCallsite:
    id: ProviderID
    unit: str
    query: int
    call: ProviderFactReference
    callee: GlobalPort
    result: GlobalPort
    owner: CodeScopeWitness
    caller: CodeCapture
    attempt: OperandCallAttempt
    target: PythonCallTarget
    target_anchor: CallRecordAnchor
    rule: str
    code_link: ProviderID | None
    # Alternative original exec selections; a shared code does not choose one.
    profile_alternatives: tuple[ProviderFactReference, ...] = ()


@dataclass(frozen=True, slots=True)
class ObservedSourceCallsiteFrontier:
    unit: str
    attempt: OperandCallAttempt
    reason: str
    candidates: tuple[ProviderFactReference, ...] = ()


@dataclass(frozen=True, slots=True)
class ObservedSourceProtocolCallsite:
    """A compiler-inserted CALL attached to its actual language operation.

    Arguments retain the protocol ABI, which need not be callee-first.
    The body is a shared definition, not a proven activation or instance.
    """
    id: ProviderID
    unit: str
    kind: str
    definition: ProviderFactReference
    operation: GlobalPort
    arguments: tuple[GlobalPort, ...]
    result: GlobalPort
    world: GlobalPort
    result_world: GlobalPort
    body: Body
    owner: CodeScopeWitness
    caller: CodeCapture
    attempt: OperandCallAttempt
    target: PythonCallTarget
    target_anchor: CallRecordAnchor
    rule: str
    code_link: ProviderID | None
    profile_alternatives: tuple[ProviderFactReference, ...] = ()


@dataclass(frozen=True, slots=True)
class ObservedPositionalReceiver:
    """One recorded CALL's receiver layout under isolated frame-slot semantics.

    The previous normal result flows through an original fast-local store
    into one explicit operand. The recorded first formal and attribute-base
    local identify the receiver, independently of a function's defaults.
    """
    producer_attempt: CallRecordAnchor
    producer_handshake: CallRecordAnchor
    producer_return: CallRecordAnchor
    caller_entry: CallRecordAnchor
    callee_entry: CallRecordAnchor
    explicit_position: int
    producer_store_offset: int
    explicit_load_offset: int
    receiver_load_offset: int
    receiver_reference: str
    explicit_reference: str
    receiver_local: str
    explicit_local: str


@dataclass(frozen=True, slots=True)
class ObservedSourceFunctionGroup:
    query: int
    code: ObservedSourceCodeLink
    observations: tuple[ObservedSourceCallsite, ...]
    connection: FunctionConnection
    parent_interfaces: tuple[int, ...]
    # Possible additional receiver prefixes beyond the source actuals, not
    # observed values. Creation-time defaults never refute a layout.
    argument_prefixes: tuple[int, ...]
    argument_layout_proven: bool
    # Actual frame records prove body entry independently of the receiver's
    # weak identity or its explicit/implicit argument layout.
    body_entries: tuple[CallRecordAnchor, ...] = ()
    # An ordered witness for each recorded observation in a recovered layout
    # group. Other groups use the original target/entry protocol.
    positional_receivers: tuple[ObservedPositionalReceiver, ...] = ()


@dataclass(frozen=True, slots=True)
class ObservedSourceFunctionFrontier:
    observation: ObservedSourceCallsite
    reason: str
