"""Original source roots for inputs of one replayed ordered region.

Field reads select symbolic content at their original world. They are not
pure reads or globally established logical content versions. The caller's
activation interface remains separate from these source-side routes.
"""
from dataclasses import dataclass

from .normal_targets import GlobalPort
from .ids import OperationID
from .providers import ProviderFactReference
from .source_address_paths import FieldSelector
from .source_content_lineage import ContentAccessRef
from .source_reference_slices import ReferenceSlice


@dataclass(frozen=True,slots=True)
class OrderedFieldRead:
    selector: FieldSelector
    receiver: GlobalPort
    result: GlobalPort
    world_before: GlobalPort
    world_after: GlobalPort
    attribute: ProviderFactReference
    content_access: ContentAccessRef
    effect_request: OperationID
    receiver_path: ReferenceSlice


@dataclass(frozen=True,slots=True)
class OrderedInputRoute:
    input_key: str
    input_record: ProviderFactReference
    source_origin: ProviderFactReference
    function: ProviderFactReference
    root_formal: GlobalPort
    parameter_index: int
    field_path: tuple[str,...]
    original_read_result: GlobalPort
    reference_path: ReferenceSlice | None
    read_chain: tuple[OrderedFieldRead,...]
    guards: tuple[str,...]

    @property
    def logical_content_version_equality_proven(self):return False


@dataclass(frozen=True,slots=True)
class OrderedInputBoundary:
    input_key: str
    reason: str
    source_origin: ProviderFactReference | None=None


__all__=['OrderedFieldRead','OrderedInputRoute','OrderedInputBoundary']
