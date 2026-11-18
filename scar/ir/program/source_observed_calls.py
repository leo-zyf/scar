"""Real call occurrences attached to original F4 callsite declarations."""
from dataclasses import dataclass
from .ids import ProviderID
from .providers import ProviderFactReference
from .normal_links import CodeScopeWitness
from .normal_targets import GlobalPort, Body
from .call_edges import CodeCapture, CallRecordAnchor
from .call_operands import OperandCallAttempt
from .python_call_targets import PythonCallTarget
from .source_observed_codes import ObservedSourceCodeLink
from .normal_targets import FunctionConnection


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
