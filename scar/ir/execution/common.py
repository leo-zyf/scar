"""Common."""

from __future__ import annotations

from dataclasses import dataclass
from scar.ir.program.providers import ProviderFactReference
from scar.ir.v2.ids import Identifier


# Typed identities for execution coordinates.
#
# These identifiers name runtime entities and occurrences. They are distinct
# from static program IDs and from legacy v2 evidence/value identifiers.
class RunID(Identifier):
    prefix = "exec_run"


class ProcessInstanceID(Identifier):
    prefix = "exec_process_instance"


class HostThreadInstanceID(Identifier):
    prefix = "exec_host_thread_instance"


class RegionActivationID(Identifier):
    prefix = "exec_region_activation"


class ExecutionSegmentID(Identifier):
    prefix = "exec_execution_segment"


class BlockVisitID(Identifier):
    prefix = "exec_block_visit"


class OperationOccurrenceID(Identifier):
    prefix = "exec_operation_occurrence"


class ValueOccurrenceID(Identifier):
    prefix = "exec_value_occurrence"


class ResourceInstanceID(Identifier):
    prefix = "exec_resource_instance"


class ExternalRealmID(Identifier):
    prefix = "exec_external_realm"


class NativeHandleLifetimeID(Identifier):
    prefix = "exec_native_handle_lifetime"


class ResourceBindingID(Identifier):
    prefix = "exec_resource_binding"


class HandleBindingID(Identifier):
    prefix = "exec_handle_binding"


class ResourceAccessID(Identifier):
    prefix = "exec_resource_access"


class ProviderResourceAccessID(Identifier):
    prefix = "exec_provider_resource_access"


class RealizationID(Identifier):
    prefix = "exec_realization"


class WorkTicketID(Identifier):
    prefix = "exec_work_ticket"


class WorkOutcomeID(Identifier):
    prefix = "exec_work_outcome"


class WorkQuiescenceID(Identifier):
    prefix = "exec_work_quiescence"


class WorkRetentionID(Identifier):
    prefix = "exec_work_retention"


class QueuePrefixID(Identifier):
    prefix = "exec_queue_prefix"


class QueueOrderProfileID(Identifier):
    prefix = "exec_queue_order_profile"


class EventRecordID(Identifier):
    prefix = "exec_event_record"


class QueueWaitID(Identifier):
    prefix = "exec_queue_wait"


class DenotationID(Identifier):
    prefix = "exec_denotation"


class WorkEvidenceBindingID(Identifier):
    prefix = "exec_work_evidence_binding"


class ContentReadID(Identifier):
    prefix = "exec_content_read"


class ObservationRequirementID(Identifier):
    prefix = "exec_observation_requirement"


class ObservationRootID(Identifier):
    prefix = "exec_observation_root"


class EvidenceInventoryID(Identifier):
    prefix = "exec_evidence_inventory"


class SourceBridgeID(Identifier):
    prefix = "exec_source_bridge"


class WriteAssociationID(Identifier):
    prefix = "exec_write_association"


class OutcomeProductionID(Identifier):
    prefix = "exec_outcome_production"


class EstablishedProductionID(Identifier):
    prefix = "exec_established_production"


class TargetLifetimeEventID(Identifier):
    prefix = "exec_target_lifetime_event"


class OrderAssociationID(Identifier):
    prefix = "exec_order_association"


class ContributionSupportID(Identifier):
    prefix = "exec_contribution_support"


class CallActivationID(Identifier):
    prefix = "exec_call_activation"


# Small provenance and unresolved variants for execution records.
#
# These records retain evidence references and gaps. Constructing them does not
# open artifacts, validate a graph, or grant runtime authority.
def _exact(value, expected, name):
    if type(value) is not expected:
        raise TypeError(f"{name} must be {expected.__name__}")


def _identifier(value, expected, name):
    _exact(value, expected, name)
    value.__post_init__()
    if type(value.value) is not str or not value.value:
        raise ValueError(f"{name} must contain a nonempty exact string")


def _record(value, expected, name):
    _exact(value, expected, name)
    value.__post_init__()


@dataclass(frozen=True, slots=True)
class MissingEvidence:
    reason: str

    def __post_init__(self):
        _exact(self.reason, str, "missing-evidence reason")
        if not self.reason:
            raise ValueError("missing-evidence reason must be nonempty")


FactSource = ProviderFactReference | MissingEvidence


@dataclass(frozen=True, slots=True)
class EvidencePoint:
    run: RunID
    process: ProcessInstanceID | None
    source: FactSource

    def __post_init__(self):
        _identifier(self.run, RunID, "evidence-point run")
        if self.process is not None:
            _identifier(self.process, ProcessInstanceID, "evidence-point process")
        if type(self.source) is ProviderFactReference:
            _record(self.source, ProviderFactReference, "evidence-point source")
        elif type(self.source) is MissingEvidence:
            _record(self.source, MissingEvidence, "evidence-point source")
        else:
            raise TypeError("evidence-point source must be a ProviderFactReference or MissingEvidence")


@dataclass(frozen=True, slots=True)
class Unresolved:
    reason: str
    point: EvidencePoint

    def __post_init__(self):
        _exact(self.reason, str, "unresolved reason")
        if not self.reason:
            raise ValueError("unresolved reason must be nonempty")
        _record(self.point, EvidencePoint, "unresolved point")


__all__ = [
    'MissingEvidence',
    'FactSource',
    'EvidencePoint',
    'Unresolved',
    'RunID',
    'ProcessInstanceID',
    'HostThreadInstanceID',
    'RegionActivationID',
    'ExecutionSegmentID',
    'BlockVisitID',
    'OperationOccurrenceID',
    'ValueOccurrenceID',
    'ResourceInstanceID',
    'ExternalRealmID',
    'NativeHandleLifetimeID',
    'ResourceBindingID',
    'HandleBindingID',
    'ResourceAccessID',
    'ProviderResourceAccessID',
    'RealizationID',
    'WorkTicketID',
    'WorkOutcomeID',
    'WorkQuiescenceID',
    'WorkRetentionID',
    'QueuePrefixID',
    'QueueOrderProfileID',
    'EventRecordID',
    'QueueWaitID',
    'DenotationID',
    'WorkEvidenceBindingID',
    'ContentReadID',
    'ObservationRequirementID',
    'ObservationRootID',
    'EvidenceInventoryID',
    'SourceBridgeID',
    'WriteAssociationID',
    'OutcomeProductionID',
    'EstablishedProductionID',
    'TargetLifetimeEventID',
    'OrderAssociationID',
    'ContributionSupportID',
    'CallActivationID',
]
