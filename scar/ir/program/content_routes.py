"""Content routes."""

from dataclasses import dataclass
from scar.ir.literals import PythonLiteral
from scar.ir.program import normal_targets as T
from enum import Enum
from scar.ir.program.ids import LocationID, OperationID, UseID, ValueID
from scar.ir.program.normal_targets import GlobalPort
from scar.ir.program.normal_links import CodeScopeRef
from scar.ir.program.state_addresses import WholeOwner
from scar.ir.v2.common import ProofStatus
from scar.ir.v2.values import EquivalenceClaim, ProvenanceRelation
from scar.ir.program.normal_graph import Edge
from scar.ir.program.providers import ProviderFactReference, ProviderArtifact
from scar.ir.program.source_value_routes import ReferenceSlice


# Lossless factors of existing may-content links, not source operations.
@dataclass(frozen=True, slots=True)
class ContentFactor:
    id: str
    container: T.Endpoint
    key: PythonLiteral | None
    sources: tuple[T.Endpoint, ...]
    targets: tuple[T.Endpoint, ...]
    semantics: str = 'FLOW_INSENSITIVE_MAY_CONTENT'


@dataclass(frozen=True, slots=True)
class ContentIndexCycle:
    container: T.Endpoint
    source: T.Endpoint
    start: int
    stride: int
    path: tuple[T.Endpoint, ...]
    abstract_position: str = 'UNKNOWN_POSITION'
    semantics: str = 'DESCENDING_CYCLE_CONSTRAINT_WITH_MAY_POSITION_WIDENING'


# Static value/content dependencies, separate from logical value versions.
#
# An access reference resolves an existing ContentAccessTemplate owned by one
# source-content provider. No world projection is promoted to a ValueVersionID,
# and these records do not assert aliasing or unchanged object contents.
@dataclass(frozen=True, slots=True)
class ContentAccessRef:
    unit: str
    ordinal: int
    operation: OperationID

    def __post_init__(self):
        if (type(self.unit) is not str or not self.unit or type(self.ordinal) is not int
                or self.ordinal < 0 or type(self.operation) is not OperationID):
            raise ValueError('content lineage needs an original unit/access ordinal/operation')


class ContentLineageRole(str, Enum):
    GET_RESULT = 'GET_RESULT'
    SET_PAYLOAD = 'SET_PAYLOAD'
    UPDATE_INPUT = 'UPDATE_INPUT'
    UPDATE_PRIOR = 'UPDATE_PRIOR'
    DELETE_PRIOR = 'DELETE_PRIOR'


@dataclass(frozen=True, slots=True)
class ContentValueLineage:
    access: ContentAccessRef
    role: ContentLineageRole
    source: GlobalPort | ValueID
    target: GlobalPort | ValueID

    def __post_init__(self):
        if (type(self.access) is not ContentAccessRef or type(self.role) is not ContentLineageRole
                or type(self.source) not in (GlobalPort, ValueID)
                or type(self.target) not in (GlobalPort, ValueID)):
            raise TypeError('content lineage requires typed original endpoints and access reference')
        self.access.__post_init__()
        expected = ((ValueID, GlobalPort) if self.role is ContentLineageRole.GET_RESULT else
            (GlobalPort, ValueID) if self.role in (ContentLineageRole.SET_PAYLOAD,
                ContentLineageRole.UPDATE_INPUT) else (ValueID, ValueID))
        if (type(self.source), type(self.target)) != expected:
            raise TypeError('content lineage endpoint roles differ from the original request')

    @property
    def provenance_relation(self):
        return ProvenanceRelation.UNKNOWN

    @property
    def equivalence(self):
        return EquivalenceClaim.UNKNOWN

    @property
    def logical_version_equality(self):
        return ProofStatus.UNKNOWN


@dataclass(frozen=True, slots=True)
class ContentForwardingObligations:
    """An explicit SET/GET inquiry; no forwarding graph edge is authorized."""
    write: ContentAccessRef
    read: ContentAccessRef
    known_source_facts: tuple[str, ...]
    required_contracts: tuple[str, ...]

    def __post_init__(self):
        if type(self.write) is not ContentAccessRef or type(self.read) is not ContentAccessRef:
            raise TypeError('forwarding obligations require original typed access references')
        for values in (self.known_source_facts, self.required_contracts):
            if type(values) is not tuple or any(type(value) is not str or not value for value in values):
                raise TypeError('forwarding source facts and contracts require explicit strings')
        if not self.required_contracts:
            raise ValueError('an unproved forwarding inquiry must retain its required contracts')

    @property
    def status(self):
        return ProofStatus.UNKNOWN

    @property
    def equivalence(self):
        return EquivalenceClaim.UNKNOWN


# Conditional builtin-slot reference theorems over original source records.
#
# These proofs separate checked source/control facts from required execution
# contracts. Reference forwarding does not establish logical content equality.
# No proof in this profile authorizes an unconditional transformation.
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


# Conditional source selector paths, never runtime addresses or content equality.
class AddressRootKind(str, Enum):
    FUNCTION_PARAMETER = 'FUNCTION_PARAMETER'
    CALL_RESULT = 'CALL_RESULT'
    SOURCE_LITERAL = 'SOURCE_LITERAL'
    CLASS_DEFINITION = 'CLASS_DEFINITION'
    FUNCTION_DEFINITION = 'FUNCTION_DEFINITION'
    OPEN_PRODUCER = 'OPEN_PRODUCER'


@dataclass(frozen=True, slots=True)
class FieldSelector:
    name: str


@dataclass(frozen=True, slots=True)
class ItemSelector:
    key: PythonLiteral | GlobalPort


AddressSelector = FieldSelector | ItemSelector


@dataclass(frozen=True, slots=True)
class AddressPath:
    root: GlobalPort
    root_kind: AddressRootKind
    selectors: tuple[AddressSelector, ...]
    evidence: tuple[ProviderFactReference, ...]
    conditions: tuple[str, ...]
    frontier: str | None = None
    semantics: str = 'CONDITIONAL_SOURCE_SELECTOR_PATH_NOT_RUNTIME_ADDRESS_OR_CONTENT_EQUALITY'


# Source content requests and symbolic state at world boundaries.
#
# An address is an expression over reference occurrences, not an ObjectID or
# physical slot. Projected state values can be equal; distinct IDs do not prove
# a mutation or different LogicalVersions. Protocol side effects stay open.
@dataclass(frozen=True, slots=True)
class SliceBoundsSelector:
    """Original bound references; no slice object or numeric bounds assumed."""
    lower: GlobalPort
    upper: GlobalPort


ContentSelector = FieldSelector | ItemSelector | SliceBoundsSelector | WholeOwner


class ContentRequestKind(str, Enum):
    GET = 'GET'
    SET = 'SET'
    DELETE = 'DELETE'
    UPDATE = 'UPDATE'


class ContentInstanceParameter(str, Enum):
    ACTIVATION = 'ACTIVATION'
    OWNER_REFERENCE_OCCURRENCE = 'OWNER_REFERENCE_OCCURRENCE'
    SELECTOR_REFERENCE_OCCURRENCES = 'SELECTOR_REFERENCE_OCCURRENCES'


@dataclass(frozen=True, slots=True)
class ContentAddressTemplate:
    id: LocationID
    receiver: GlobalPort
    selector: ContentSelector
    instance_parameters: tuple[ContentInstanceParameter, ...]
    semantics: str = 'SOURCE_PROTOCOL_REQUEST_ADDRESS_NOT_PHYSICAL_SLOT'


@dataclass(frozen=True, slots=True)
class ContentStateProjection:
    """The abstract state project(world, selected address instance).

    This is a symbolic state variable, never a captured tensor, a write count,
    or an established LogicalVersion. A GET may run a mutating descriptor.
    """
    id: ValueID
    address: LocationID
    world: GlobalPort
    semantics: str = 'SYMBOLIC_CONTENT_AT_WORLD_NOT_LOGICAL_VERSION_EQUALITY'


@dataclass(frozen=True, slots=True)
class ContentAccessTemplate:
    """SET payload is the assigned reference; UPDATE payload is an input.

    An UPDATE input is never asserted equal to the resulting whole contents.
    Original instructions retain the append/extend/merge protocol distinction.
    """
    operation: OperationID
    site: GlobalPort
    scope: CodeScopeRef
    kind: ContentRequestKind
    address: LocationID
    receiver_use: UseID
    payload: GlobalPort | None
    result: GlobalPort | None
    world_before: GlobalPort
    world_after: GlobalPort
    state_before: ValueID
    state_after: ValueID
    attribute_fact: int | None
    effects_closed: bool = False


__all__ = [
    'ContentAccessRef',
    'ContentLineageRole',
    'ContentValueLineage',
    'ContentForwardingObligations',
    'BuiltinSlotKind',
    'SourceContainerProducer',
    'BuiltinSlotForwardingGuard',
    'SourceForwardingControlWindow',
    'ContentReferenceForwardingProof',
    'ContentForwardingRefusal',
    'ContentForwardingDiscoveryBoundary',
    'ContentFactor',
    'ContentIndexCycle',
    'SliceBoundsSelector',
    'ContentSelector',
    'ContentRequestKind',
    'ContentInstanceParameter',
    'ContentAddressTemplate',
    'ContentStateProjection',
    'ContentAccessTemplate',
    'AddressRootKind',
    'FieldSelector',
    'ItemSelector',
    'AddressSelector',
    'AddressPath',
]
