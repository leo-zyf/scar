"""Matched implementation interfaces, distinct from observed invocations."""
from dataclasses import dataclass
from enum import Enum

from .dependencies import DependencyPoint,DependencyFrame,DependencyKind,DependencyMeaning
from .providers import ProviderFactReference
from .normal_targets import GlobalPort
from .source_content_lineage import ContentAccessRef
from .source_content import ContentAccessTemplate


class OrderedBoundaryRole(str,Enum):
    ENVIRONMENT='ENVIRONMENT'
    WORLD_ENTRY='WORLD_ENTRY'
    WORLD_EXIT='WORLD_EXIT'
    CONTROL_ENTRY='CONTROL_ENTRY'
    CONTROL_EXIT='CONTROL_EXIT'
    RETURN_NONE='RETURN_NONE'


@dataclass(frozen=True,slots=True)
class OrderedBoundaryPort:
    role: OrderedBoundaryRole
    original: GlobalPort | None
    semantics: str='CONDITIONAL_ORDERED_IMPLEMENTATION_INTERFACE_NOT_RUNTIME_VALUE_IDENTITY'

    def __post_init__(self):
        if type(self.role)is not OrderedBoundaryRole or self.original is not None and type(self.original)is not GlobalPort:
            raise TypeError('ordered boundary needs its typed role and original optional port')


class OrderedContextSpace(str,Enum):
    CALLER='CALLER'
    SOURCE='SOURCE'
    ORDERED='ORDERED'


@dataclass(frozen=True,slots=True)
class OrderedActivationBinding:
    provider: str
    unit: str
    ordinal: int
    query: int
    interface: ProviderFactReference
    source_frame: DependencyFrame
    ordered_frame: DependencyFrame
    guards: tuple[str,...]

    def __post_init__(self):
        if (type(self.interface)is not ProviderFactReference or type(self.source_frame)is not DependencyFrame
                or type(self.ordered_frame)is not DependencyFrame or not self.guards
                or any(type(i)is not int or i<0 for i in(self.ordinal,self.query))):
            raise TypeError('matched original interface, frames, ordinals and guards required')
        left,right=self.source_frame,self.ordered_frame
        if (left.target==right.target or any(getattr(left,k)!=getattr(right,k)
                for k in('caller','operation','callee_use','callee_value','selection'))):
            raise ValueError('ordered frame must select another implementation of the same original call')


@dataclass(frozen=True,slots=True)
class OrderedActivationRoute:
    binding: int
    source: DependencyPoint
    source_space: OrderedContextSpace
    target: DependencyPoint
    target_space: OrderedContextSpace
    kind: DependencyKind
    meaning: DependencyMeaning
    rule: str
    guards: tuple[str,...]=()

    def __post_init__(self):
        if (type(self.binding)is not int or self.binding<0 or type(self.source)is not DependencyPoint
                or type(self.target)is not DependencyPoint
                or any(type(s)is not OrderedContextSpace for s in(self.source_space,self.target_space))
                or type(self.kind)is not DependencyKind or type(self.meaning)is not DependencyMeaning
                or self.source.context is not None or self.target.context is not None):
            raise TypeError('ordered routes require typed unselected template endpoints')


@dataclass(frozen=True,slots=True)
class OrderedContentAccessWitness:
    """A sparse parent-table row retaining its original citation ordinal."""
    reference: ContentAccessRef
    access: ContentAccessTemplate

    def __post_init__(self):
        if (type(self.reference)is not ContentAccessRef or type(self.access)is not ContentAccessTemplate
                or self.reference.operation!=self.access.operation):
            raise ValueError('ordered content witness must retain its actual original access')


__all__=['OrderedBoundaryRole','OrderedBoundaryPort','OrderedContextSpace',
    'OrderedActivationBinding','OrderedActivationRoute','OrderedContentAccessWitness']
