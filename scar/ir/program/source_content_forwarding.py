"""Conditional builtin-slot reference theorems over original source records.

These proofs separate checked source/control facts from required execution
contracts. Reference forwarding does not establish logical content equality.
No proof in this profile authorizes an unconditional transformation.
"""
from dataclasses import dataclass
from enum import Enum

from .normal_targets import GlobalPort
from .normal_graph import Edge
from .providers import ProviderFactReference,ProviderArtifact
from .source_content_lineage import ContentAccessRef
from .source_reference_slices import ReferenceSlice
from scar.ir.literals import PythonLiteral
from scar.ir.v2.common import ProofStatus
from scar.ir.v2.values import EquivalenceClaim


class BuiltinSlotKind(str,Enum):
    DICT='EXACT_BUILTIN_DICT'
    LIST='EXACT_BUILTIN_LIST'


@dataclass(frozen=True,slots=True)
class SourceContainerProducer:
    value: GlobalPort
    site: GlobalPort
    fact: ProviderFactReference
    protocol: str
    kind: BuiltinSlotKind


@dataclass(frozen=True,slots=True)
class BuiltinSlotForwardingGuard:
    kind: BuiltinSlotKind
    producer: GlobalPort
    write_receiver: GlobalPort
    read_receiver: GlobalPort
    selector: PythonLiteral
    contracts: tuple[str,...]

    def __post_init__(self):
        if (type(self.kind)is not BuiltinSlotKind or not self.contracts
                or any(type(name)is not str or not name for name in self.contracts)):
            raise ValueError('builtin reference forwarding needs its explicit execution contracts')


@dataclass(frozen=True,slots=True)
class SourceForwardingControlWindow:
    source_function: GlobalPort
    function: ProviderFactReference
    path: tuple[Edge,...]
    write_exit_index: int
    local_read_sites: tuple[GlobalPort,...]
    local_read_facts: tuple[ProviderFactReference,...]
    local_read_materials: tuple[ProviderArtifact,...]
    local_read_contracts: tuple[str,...]
    guarded_cleanup_sites: tuple[GlobalPort,...]=()


@dataclass(frozen=True,slots=True)
class ContentReferenceForwardingProof:
    write: ContentAccessRef
    read: ContentAccessRef
    payload: GlobalPort
    result: GlobalPort
    producer: SourceContainerProducer
    selector: PythonLiteral
    receiver_paths: tuple[ReferenceSlice,...]
    selector_paths: tuple[ReferenceSlice,...]
    control: SourceForwardingControlWindow
    guard: BuiltinSlotForwardingGuard
    static_facts: tuple[str,...]=()
    runtime_checks: tuple[str,...]=()
    external_assumptions: tuple[str,...]=()
    rule: str='BUILTIN_SCALAR_SLOT_SET_THEN_GET_REFERENCE_THEOREM_V1'

    @property
    def claim(self):return 'REFERENCE_FORWARDING_PROVEN_UNDER_EXPLICIT_UNDISCHARGED_CONTRACTS'

    @property
    def unconditional_status(self):return ProofStatus.UNKNOWN

    @property
    def logical_equivalence(self):return EquivalenceClaim.UNKNOWN

    @property
    def transformation_authorized(self):return False


@dataclass(frozen=True,slots=True)
class ContentForwardingRefusal:
    write: ContentAccessRef
    read: ContentAccessRef
    reason: str
    site: GlobalPort | None=None


@dataclass(frozen=True,slots=True)
class ContentForwardingDiscoveryBoundary:
    access: ContentAccessRef
    reason: str
    site: GlobalPort | None=None


__all__=['BuiltinSlotKind','SourceContainerProducer','BuiltinSlotForwardingGuard',
    'SourceForwardingControlWindow','ContentReferenceForwardingProof','ContentForwardingRefusal',
    'ContentForwardingDiscoveryBoundary']
