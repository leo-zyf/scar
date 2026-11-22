"""Resource observations."""

from dataclasses import dataclass
from enum import Enum
from scar.ir.program.dependencies import DependencyPoint
from scar.ir.program.ids import OperationID, NativeBoundaryID, ProviderID
from scar.ir.program.providers import ProviderFactReference
from scar.ir.program.normal_graph import GlobalPort
from scar.ir.program.python_bindings import NativeSchemaReference
from scar.ir.program.native_contracts import DeclaredDispatchCandidate, TypeProjection
from scar.ir.program.resources import ResourceDefinition
from scar.ir.program.source_effects import FamilyRule
from scar.ir.program.target_hardware import TargetResourceRef, ResourceCandidateKind
from scar.ir.program.native import EntryPortRef, ExitPortRef
from scar.ir.program.call_edges import InvocationKey, CallRecordAnchor
from scar.ir.program.body_observations import SourceActivationAnchor


# Type-level software/resource constraints, distinct from physical use.
class ResourceValueRole(str, Enum):
    TENSOR = 'TENSOR'
    DEVICE = 'DEVICE'
    GENERATOR = 'GENERATOR'


class ResourceValueDirection(str, Enum):
    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'


@dataclass(frozen=True, slots=True)
class NativeResourceValue:
    boundary: NativeBoundaryID
    port: EntryPortRef | ExitPortRef
    direction: ResourceValueDirection
    role: ResourceValueRole
    schema_type: str
    type_path: tuple[TypeProjection, ...]
    evidence: ProviderFactReference


@dataclass(frozen=True, slots=True)
class ResourceTargetOption:
    value: int
    target: TargetResourceRef
    relation: ResourceCandidateKind
    conditions: tuple[str, ...]
    selected: bool = False


@dataclass(frozen=True, slots=True)
class SourceResourceRoute:
    normalization: ProviderFactReference
    value: int
    conditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceUnitHostOption:
    unit: str
    source: ProviderFactReference
    target: TargetResourceRef
    conditions: tuple[str, ...] = (
        'SOURCE_SCOPE_ENTERED', 'HOST_CPU_AFFINITY_AND_NUMA_POLICY_PERMIT_EXECUTION',
        'SOURCE_UNIT_DECLARATION_NOT_RUNTIME_CPU_OR_THREAD_SELECTION',
    )
    selected: bool = False


@dataclass(frozen=True, slots=True)
class NativeExecutorOption:
    boundary: NativeBoundaryID
    target: TargetResourceRef
    declared_dispatch: tuple[int, ...]
    conditions: tuple[str, ...] = (
        'NATIVE_RUNTIME_DISPATCH_AND_IMPLEMENTATION_DEVICE_MATCH_EXECUTOR',
        'TARGET_COMPUTE_ACCESS_AND_EXECUTION_POLICY_PERMIT',
        'DECLARED_DISPATCH_IS_NOT_EXHAUSTIVE_CURRENT_KERNEL_REGISTRY',
    )
    selected: bool = False


# F4 resource declarations and snapshot references, never actual placement.
class F4ResourceApplicability(str, Enum):
    NOT_APPLICABLE = 'NOT_APPLICABLE'
    COMPOSITE = 'COMPOSITE'
    DECLARATION_ONLY = 'DECLARATION_ONLY'
    HOST = 'HOST'
    HOST_INTERFACE_OPEN = 'HOST_INTERFACE_OPEN'
    EXTERNAL_INTERFACE_OPEN = 'EXTERNAL_INTERFACE_OPEN'


@dataclass(frozen=True, slots=True)
class F4ResourceInterface:
    operation: DependencyPoint
    rule: FamilyRule
    reference_ports: tuple[DependencyPoint, ...]
    world_ports: tuple[DependencyPoint, ...]
    structural_or_compile_only: bool
    applicability: F4ResourceApplicability
    reason: str

    def __post_init__(self):
        structural = self.applicability in (F4ResourceApplicability.NOT_APPLICABLE,
            F4ResourceApplicability.COMPOSITE, F4ResourceApplicability.DECLARATION_ONLY)
        if (type(self.applicability) is not F4ResourceApplicability or not self.reason
                or self.structural_or_compile_only is not structural):
            raise ValueError('resource applicability must distinguish local demands from composition')


@dataclass(frozen=True, slots=True)
class F4TargetResource:
    definition: ResourceDefinition
    target: TargetResourceRef


@dataclass(frozen=True, slots=True)
class F4ResourceRoute:
    operation: DependencyPoint
    endpoints: tuple[DependencyPoint, ...]
    normalization: ProviderFactReference
    source_route: ProviderFactReference
    value: NativeResourceValue
    target: TargetResourceRef
    conditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class F4NativeExecutorRoute:
    operation: DependencyPoint
    source_call: GlobalPort
    source_match: ProviderFactReference
    declaration: ProviderFactReference
    normalization: ProviderFactReference
    schema: NativeSchemaReference
    executor_option: ProviderFactReference
    dispatch: tuple[DeclaredDispatchCandidate, ...]
    dispatch_evidence: tuple[ProviderFactReference, ...]
    target: TargetResourceRef
    conditions: tuple[str, ...]

    def __post_init__(self):
        if (type(self.operation.node) is not OperationID
                or len(self.dispatch) != len(self.dispatch_evidence)
                or any(row.boundary != self.schema.boundary for row in self.dispatch)):
            raise ValueError('native executor route must retain its original operation/schema/dispatch')


# Observed call boundaries and their original source and resource references.
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
