"""Typed target descriptions, separate from program requirements and actual use.

Inventory IDs are scoped by their target snapshot. Capacities are reported
totals; budgets and measured link performance remain unknown unless supplied.
"""
from dataclasses import dataclass
from enum import Enum

from scar.ir.v2.ids import Identifier
from .providers import ProviderArtifact,ProviderFactReference


class TargetResourceID(Identifier):
    prefix='target_resource'


@dataclass(frozen=True,slots=True)
class CpuExecutor:
    id: TargetResourceID
    numa_node: int | None
    cpu_ids: tuple[int,...]
    physical_cores: int | None
    model_names: tuple[str,...]
    evidence: ProviderFactReference


class MemoryKind(str,Enum):
    HOST='HOST'
    DEVICE='DEVICE'


@dataclass(frozen=True,slots=True)
class MemoryDomain:
    id: TargetResourceID
    kind: MemoryKind
    owner: TargetResourceID | None
    numa_node: int | None
    capacity_bytes: int | None
    technology: str | None
    budget_bytes: int | None
    evidence: ProviderFactReference


@dataclass(frozen=True,slots=True)
class Accelerator:
    id: TargetResourceID
    family: str
    name: str
    uuid: str
    inventory_index: int
    pci_bus_id: str
    numa_node: int | None
    compute_capability: str | None
    evidence: ProviderFactReference


@dataclass(frozen=True,slots=True)
class StorageDevice:
    id: TargetResourceID
    kernel_name: str
    capacity_bytes: int | None
    transport: str | None
    media_kind: str | None
    evidence: ProviderFactReference


@dataclass(frozen=True,slots=True)
class StorageEndpoint:
    id: TargetResourceID
    mount_point: str
    source: str
    filesystem: str
    volume_bytes: int | None
    evidence: ProviderFactReference


@dataclass(frozen=True,slots=True)
class CpuPackage:
    id: TargetResourceID
    package_id: int
    cpu_ids: tuple[int,...]
    evidence: ProviderFactReference


@dataclass(frozen=True,slots=True)
class CpuCore:
    id: TargetResourceID
    package_id: int
    die_id: int | None
    core_id: int
    cpu_ids: tuple[int,...]
    evidence: ProviderFactReference


@dataclass(frozen=True,slots=True)
class HardwareThread:
    id: TargetResourceID
    cpu_id: int
    online: bool | None
    thread_siblings: tuple[int,...] | None
    isa_flags: tuple[str,...] | None
    isa_evidence: ProviderFactReference | None
    evidence: ProviderFactReference


@dataclass(frozen=True,slots=True)
class CacheDomain:
    """Shared hardware cache; capacity is never an allocation budget."""
    id: TargetResourceID
    level: int
    cache_type: str
    shared_cpu_ids: tuple[int,...]
    capacity_bytes: int | None
    line_size_bytes: int | None
    evidence: ProviderFactReference


@dataclass(frozen=True,slots=True)
class PciFunction:
    id: TargetResourceID
    pci_bus_id: str
    class_code: str | None
    vendor_id: str | None
    device_id: str | None
    category: str
    planning_role: str
    driver: str | None
    numa_node: int | None
    local_cpu_ids: tuple[int,...] | None
    current_link_speed: str | None
    current_link_width: int | None
    max_link_speed: str | None
    max_link_width: int | None
    evidence: ProviderFactReference


@dataclass(frozen=True,slots=True)
class HardwareBoundary:
    """Discoverable control peripherals grouped without scheduling internals."""
    id: TargetResourceID
    category: str
    pci_bus_ids: tuple[str,...]
    reason: str
    evidence: ProviderFactReference


@dataclass(frozen=True,slots=True)
class DiscoveredAccelerator:
    """A hardware endpoint, with no inferred CUDA or compute support."""
    id: TargetResourceID
    family: str
    name: str | None
    pci_bus_id: str | None
    driver: str | None
    numa_node: int | None
    compute_capability: str | None
    memory_total_bytes: int | None
    evidence: ProviderFactReference


@dataclass(frozen=True,slots=True)
class NetworkInterface:
    id: TargetResourceID
    name: str
    physical: bool
    pci_bus_id: str | None
    numa_node: int | None
    operstate: str | None
    link_rate_mbps: int | None
    mtu: int | None
    evidence: ProviderFactReference


@dataclass(frozen=True,slots=True)
class RdmaPort:
    port: int
    state: str | None
    physical_state: str | None
    rate: str | None
    link_layer: str | None


@dataclass(frozen=True,slots=True)
class RdmaAdapter:
    id: TargetResourceID
    name: str
    pci_bus_id: str | None
    ports: tuple[RdmaPort,...]
    evidence: ProviderFactReference


TargetResource=(CpuExecutor|MemoryDomain|Accelerator|StorageDevice|StorageEndpoint
    |CpuPackage|CpuCore|HardwareThread|CacheDomain|PciFunction|HardwareBoundary
    |DiscoveredAccelerator|NetworkInterface|RdmaAdapter)
TARGET_RECORD_TAGS={cls:('resource_type',cls.__name__) for cls in
    (CpuExecutor,MemoryDomain,Accelerator,StorageDevice,StorageEndpoint,
     CpuPackage,CpuCore,HardwareThread,CacheDomain,PciFunction,HardwareBoundary,
     DiscoveredAccelerator,NetworkInterface,RdmaAdapter)}


class ConnectionKind(str,Enum):
    MEMORY_ATTACHED='MEMORY_ATTACHED'
    MEMORY_ACCESS='MEMORY_ACCESS'
    HOST_DEVICE_LINK='HOST_DEVICE_LINK'
    STORAGE_BACKING='STORAGE_BACKING'
    STORAGE_ACCESS='STORAGE_ACCESS'
    CONTAINS='CONTAINS'
    SHARES_CACHE='SHARES_CACHE'
    REFILL_CANDIDATE='REFILL_CANDIDATE'
    PCI_PARENT='PCI_PARENT'
    DEVICE_INTERFACE='DEVICE_INTERFACE'
    FABRIC_ACCESS='FABRIC_ACCESS'
    NETWORK_LOWER='NETWORK_LOWER'


@dataclass(frozen=True,slots=True)
class ResourceAvailability:
    resource: TargetResourceID
    discovery_status: str
    visibility_status: str
    accessibility_status: str
    backend_status: str
    evidence: ProviderFactReference


@dataclass(frozen=True,slots=True)
class CgroupLimit:
    path: str
    cpu_max: str | None
    cpuset_cpus: str | None
    cpuset_mems: str | None
    memory_max: str | None
    memory_high: str | None


@dataclass(frozen=True,slots=True)
class ExecutionConstraints:
    """Current querying task scope, separate from installed hardware inventory."""
    affinity_cpu_ids: tuple[int,...] | None
    allowed_cpu_ids: tuple[int,...] | None
    allowed_memory_nodes: tuple[int,...] | None
    cgroup_mode: str | None
    cgroup_path: str | None
    cgroup_limits: tuple[CgroupLimit,...]
    visibility_environment: tuple[tuple[str,str | None],...]
    evidence: ProviderFactReference


@dataclass(frozen=True,slots=True)
class ResourceConnection:
    source: TargetResourceID
    target: TargetResourceID
    kind: ConnectionKind
    locality: str | None
    bandwidth_bytes_per_second: float | None
    latency_seconds: float | None
    advertised_link_speed: str | None
    advertised_link_width: int | None
    evidence: ProviderFactReference
    conditions: tuple[str,...]=()
    bidirectional: bool=False


@dataclass(frozen=True,slots=True)
class PathEndpoint:
    requested_path: str
    resolved_path: str
    endpoint: TargetResourceID | None
    backing_status: str
    evidence: ProviderFactReference


@dataclass(frozen=True,slots=True)
class TargetResourceRef:
    snapshot: ProviderArtifact
    resource: TargetResourceID


class ResourceCandidateKind(str,Enum):
    TENSOR_STORAGE_POOL='TENSOR_STORAGE_POOL'
    DEVICE_REFERENT='DEVICE_REFERENT'
    GENERATOR_DEVICE='GENERATOR_DEVICE'
