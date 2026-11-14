"""Actual call scopes over original shared source ports, never content versions."""
from dataclasses import dataclass
import re

from .call_edges import InvocationKey, CallRecordAnchor
from .normal_targets import GlobalPort, Body
from .providers import ProviderFactReference
from .ids import ProviderID
from .view import SourceAnchor
from .source_productions import ProductionRegionID,ProductionStepID


@dataclass(frozen=True, slots=True)
class SourceCallScope:
    run_scope: str
    invocation: InvocationKey

    def __post_init__(self):
        if (type(self.run_scope) is not str or re.fullmatch(r'[0-9a-f]{64}',self.run_scope) is None
                or type(self.invocation) is not InvocationKey):
            raise ValueError('actual source call scope requires its original run and InvocationKey')
        self.invocation.__post_init__()


@dataclass(frozen=True, slots=True)
class SourceInvocationPort:
    call_scope: SourceCallScope
    original: GlobalPort
    # A caller may evaluate one source CALL repeatedly within one invocation.
    # Its original attempt anchor qualifies that boundary occurrence, not a
    # logical content version or an unrolled copy of the caller's body.
    boundary: CallRecordAnchor | None = None

    def __post_init__(self):
        if type(self.call_scope) is not SourceCallScope or type(self.original) is not GlobalPort:
            raise ValueError('occurrence handle requires an actual call scope and original source port')
        if self.boundary is not None and type(self.boundary) is not CallRecordAnchor:
            raise ValueError('boundary occurrence requires its original attempt record anchor')


@dataclass(frozen=True, slots=True)
class SourceInvocationNode:
    """A query qualification of a shared region/step, never a body copy."""
    call_scope: SourceCallScope
    original: SourceAnchor

    def __post_init__(self):
        if (type(self.call_scope)is not SourceCallScope or type(self.original)is not SourceAnchor
                or type(self.original.original) not in (ProductionRegionID,ProductionStepID)
                or self.original.template is None):
            raise ValueError('scoped source node needs an original F4 region/step anchor')


@dataclass(frozen=True, slots=True)
class SourceInvocationRoute:
    source: SourceInvocationPort
    target: SourceInvocationPort
    field: str
    evidence: tuple[ProviderFactReference,...]
    conditions: tuple[str,...]


@dataclass(frozen=True, slots=True)
class SourceInvocationFrontier:
    field: str
    reason: str
    port: GlobalPort | None = None


@dataclass(frozen=True, slots=True)
class SourceInvocationBinding:
    id: ProviderID
    call_scope: SourceCallScope
    caller_scope: SourceCallScope
    activation: ProviderID
    callsite: ProviderID
    body: Body
    entry: CallRecordAnchor
    exit: CallRecordAnchor | None
    routes: tuple[SourceInvocationRoute,...]
    frontiers: tuple[SourceInvocationFrontier,...]
    semantics: str = 'RECORDED_INVOCATION_SOURCE_PORT_ROUTES_NOT_CONTENT_PRESERVATION'
