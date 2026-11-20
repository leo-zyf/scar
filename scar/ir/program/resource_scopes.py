"""Observed call boundaries and their original source and resource references."""
from dataclasses import dataclass
from enum import Enum
from .ids import ProviderID
from .call_edges import InvocationKey,CallRecordAnchor
from .providers import ProviderFactReference
from .source_body_activations import SourceActivationAnchor


class ScopeEventKind(str,Enum):
    BEGIN='BEGIN'
    C_RETURN='C_RETURN'
    C_RAISE='C_RAISE'
    PY_EXIT='PY_EXIT'


@dataclass(frozen=True,slots=True)
class ScopeEvent:
    sequence: int
    scope: int
    kind: ScopeEventKind
    caller: InvocationKey
    native_tid: int
    attempt_index: int
    source: ProviderFactReference


@dataclass(frozen=True,slots=True)
class ObservedResourceScope:
    id: ProviderID
    begin: ScopeEvent
    end: ScopeEvent
    call_attempt: CallRecordAnchor
    os_begin: ProviderFactReference
    os_end: ProviderFactReference

    def __post_init__(self):
        a,b=self.begin,self.end
        exits=(ScopeEventKind.PY_EXIT,) if isinstance(self,ObservedPythonResourceScope) else (
            ScopeEventKind.C_RETURN,ScopeEventKind.C_RAISE)
        if (a.kind is not ScopeEventKind.BEGIN or b.kind not in exits
                or a.scope!=b.scope or a.caller!=b.caller or a.native_tid!=b.native_tid
                or a.attempt_index!=b.attempt_index or b.sequence<=a.sequence):
            raise ValueError('resource scope needs matching actual boundaries and CALL')


@dataclass(frozen=True,slots=True)
class ObservedPythonResourceScope(ObservedResourceScope):
    """An ordinary Python activation with an observed physical frame exit."""
    callee: InvocationKey
    callee_code: CallRecordAnchor
    entry_binding: CallRecordAnchor
    frame_entry: CallRecordAnchor
    frame_exit: CallRecordAnchor
    normal_completion_proven: bool = False

    def __post_init__(self):
        ObservedResourceScope.__post_init__(self)
        caller=self.begin.caller
        if (self.callee==caller or (self.callee.process_id,self.callee.thread_id)!=(caller.process_id,caller.thread_id)
                or self.frame_entry.path!=self.frame_exit.path
                or self.frame_entry.index is None or self.frame_exit.index is None
                or self.frame_entry.index>=self.frame_exit.index
                or self.normal_completion_proven is not False):
            raise ValueError('Python resource scope needs matching original entry and physical exit')


@dataclass(frozen=True,slots=True)
class ScopedResourceWork:
    scope: ProviderID
    source: ProviderFactReference
    syscall: str
    family: str
    result: int | None
    relation: str = 'ENCLOSED_BY_OBSERVED_NATIVE_CALL'
    direct_program_origin_proven: bool = False


@dataclass(frozen=True,slots=True)
class SourceResourceScopeBinding:
    scope: ProviderID
    anchor: SourceActivationAnchor
    semantics: str = 'ORIGINAL_SOURCE_PORTS_NOT_EFFECT_ALIAS_READINESS_OR_EXCLUSIVE_PROGRAM_WORK'


@dataclass(frozen=True,slots=True)
class SourceResourceInterface:
    """One shared source interface with occurrence evidence kept separately."""
    id: ProviderID
    anchor: SourceActivationAnchor
    scopes: tuple[ProviderID,...]
    work_count: int
