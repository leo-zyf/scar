"""Entry-time binding occurrences over shared F4 formal templates."""

from dataclasses import dataclass

from .call_edges import CallRecordAnchor, NamedReference
from .ids import LocationID, ProviderID, ValueID
from .normal_targets import GlobalPort
from .source_invocations import SourceCallScope, SourceInvocationPort


@dataclass(frozen=True, slots=True)
class ObservedFormalInstance:
    """The reference recorded for a formal in one actual Python frame."""

    call_scope: SourceCallScope
    entry: CallRecordAnchor
    code_link: ProviderID
    source_function: GlobalPort
    formal: GlobalPort
    formal_ordinal: int
    location: LocationID
    binding_version: ValueID
    observed_reference: NamedReference

    @property
    def invocation_port(self) -> SourceInvocationPort:
        return SourceInvocationPort(self.call_scope, self.formal)


@dataclass(frozen=True, slots=True)
class ObservedFormalFrontier:
    call_scope: SourceCallScope
    entry: CallRecordAnchor
    reason: str
    formal: GlobalPort | None = None
