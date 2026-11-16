"""Parametric contents and required materializations in one array region.

These are definition-time templates over the region's input versions, never
runtime LogicalValueID, StorageID or InvocationID. Equal content templates
do not permit sharing mutable result objects or reusing results across calls.
"""
from dataclasses import dataclass
from enum import Enum

from .ids import ValueID


class ArrayMaterializationRole(str, Enum):
    INPUT_REFERENCE = 'INPUT_REFERENCE'
    IMMUTABLE_SCALAR = 'IMMUTABLE_SCALAR'
    VIEW_DESCRIPTOR = 'VIEW_DESCRIPTOR'
    FRESH_ARRAY_RESULT = 'FRESH_ARRAY_RESULT'


@dataclass(frozen=True, slots=True)
class ArrayValueTemplate:
    expression: str
    value: ValueID
    content_template: str
    materialization: ArrayMaterializationRole
    operand_expressions: tuple[str, ...]
    optimized_expression: str | None
    source_origin_ordinals: tuple[int, ...]
    required_fresh_output: bool


__all__ = ['ArrayMaterializationRole', 'ArrayValueTemplate']
