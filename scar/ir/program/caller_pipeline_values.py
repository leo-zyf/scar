"""Source-bound macro values and argument substitutions for one caller.

Content templates are parametric definitions. Their IDs are neither runtime
LogicalValueIDs nor storage/allocation identities or cross-call cache keys.
"""
from dataclasses import dataclass

from scar.ir.program.layout_templates import ArrayMaterializationRole
from .dependencies import DependencyFrame
from .normal_targets import GlobalPort,Body
from .providers import ProviderFactReference


@dataclass(frozen=True,slots=True)
class CallerParameterContent:
    parameter: str
    value: str
    formal: GlobalPort
    actual: GlobalPort
    content_template: str


@dataclass(frozen=True,slots=True)
class CallerDataValue:
    id: str
    content_template: str
    materialization: ArrayMaterializationRole
    producer_index: int
    consumer_indices: tuple[int,...]
    last_use: int
    ownership: str
    escape: bool
    shape: tuple[int,int] | None
    nbytes: int | None
    source_pointer: str


@dataclass(frozen=True,slots=True)
class CallerDataCall:
    id: str
    index: int
    inputs: tuple[str,...]
    output: str
    data_predecessors: tuple[int,...]
    substitutions: tuple[CallerParameterContent,...]
    frame: DependencyFrame
    original_call: ProviderFactReference
    interface_fact: ProviderFactReference
    shared_body: Body
    source_result: GlobalPort
    world_before: GlobalPort
    world_after: GlobalPort
    leaf_plan_pointer: str
    evidence_pointer: str


@dataclass(frozen=True,slots=True)
class CallerValueFieldFrontier:
    field: str
    reason: str


@dataclass(frozen=True,slots=True)
class CallerValueProjection:
    pipeline_sha256: str
    values: tuple[CallerDataValue,...]
    calls: tuple[CallerDataCall,...]
    frontiers: tuple[CallerValueFieldFrontier,...]
    required_guards: tuple[str,...]
    input_shapes: dict
    scope: str='PARAMETRIC_CONTENT_FOR_ONE_CLOSED_CALLER_ACTIVATION'
    leaf_representation: str='OPTIMIZED_SHARED_LEAF_AND_REPLAYED_SOURCE_OPERATION_MAP_NOT_COMPLETE_RAW_LEAF'


__all__=['CallerParameterContent','CallerDataValue','CallerDataCall','CallerValueFieldFrontier',
    'CallerValueProjection']
