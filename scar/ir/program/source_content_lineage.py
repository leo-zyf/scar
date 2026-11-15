"""Static value/content dependencies, separate from logical value versions.

An access reference resolves an existing ContentAccessTemplate owned by one
source-content provider. No world projection is promoted to a ValueVersionID,
and these records do not assert aliasing or unchanged object contents.
"""
from dataclasses import dataclass
from enum import Enum

from .ids import OperationID, ValueID
from .normal_targets import GlobalPort
from scar.ir.v2.common import ProofStatus
from scar.ir.v2.values import EquivalenceClaim, ProvenanceRelation


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


__all__ = ['ContentAccessRef', 'ContentLineageRole', 'ContentValueLineage',
    'ContentForwardingObligations']
