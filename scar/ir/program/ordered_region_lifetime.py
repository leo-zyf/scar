"""Region-local input uses and content epochs, never global value identities."""
from dataclasses import dataclass
from enum import Enum

from .providers import ProviderFactReference


class OrderedInputUseRole(str,Enum):
    SCALAR_VALUE_READ='SCALAR_VALUE_READ'
    ARRAY_CONTENT_READ='ARRAY_CONTENT_READ'
    READ_MODIFY_WRITE='READ_MODIFY_WRITE'
    PRESERVES_UNWRITTEN_REGION='PRESERVES_UNWRITTEN_REGION'
    WRITE_DESTINATION_REFERENCE='WRITE_DESTINATION_REFERENCE'


class OrderedInputStability(str,Enum):
    SELECTED_SCALAR_VALUE_FIXED='SELECTED_SCALAR_VALUE_FIXED'
    ARRAY_CONTENT_PRESERVED='ARRAY_CONTENT_PRESERVED'
    ARRAY_CONTENT_HAS_ORDERED_WRITES='ARRAY_CONTENT_HAS_ORDERED_WRITES'


@dataclass(frozen=True,slots=True)
class OrderedInputUse:
    input_key: str
    action_index: int
    role: OrderedInputUseRole
    epoch: int | None
    expression_key: str | None
    projection_edge_ordinal: int
    action: ProviderFactReference
    source_origins: tuple[ProviderFactReference,...]


@dataclass(frozen=True,slots=True)
class OrderedEpochLifetime:
    input_key: str
    epoch_key: str
    epoch: int
    producer_action: int | None
    read_use_ordinals: tuple[int,...]
    first_read_action: int | None
    last_read_action: int | None
    superseded_at_action: int | None
    required_at_normal_exit: bool


@dataclass(frozen=True,slots=True)
class OrderedInputLifetime:
    input_key: str
    original_input: ProviderFactReference
    input_kind: str
    source_route_ordinals: tuple[int,...]
    expression_keys: tuple[str,...]
    use_ordinals: tuple[int,...]
    write_actions: tuple[int,...]
    epoch_ordinals: tuple[int,...]
    first_read_action: int | None
    last_read_action: int | None
    last_reference_use_action: int | None
    exit_epoch_key: str | None
    entry_content_reaches_exit: bool | None
    stability: OrderedInputStability


__all__=['OrderedInputUseRole','OrderedInputStability','OrderedInputUse',
    'OrderedEpochLifetime','OrderedInputLifetime']
