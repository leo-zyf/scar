"""Batch admission preserves the original strict row and constructor paths."""
from copy import deepcopy
from dataclasses import dataclass

import pytest

from scar.ir import record_codec as codec
from scar.ir.literals import PythonLiteral
from scar.ir.record_codec import decode, encode
from scar.ir.v2.ids import LogicalValueID, ObjectID, ValueVersionID


@dataclass
class Left:
    value: int


@dataclass
class Right:
    value: int


@dataclass
class Extended(Left):
    label: str


TRACE = []


@dataclass
class InvalidAfterInit:
    value: int

    def __post_init__(self):
        TRACE.append(('invalid', self.value))
        self.value = True


@dataclass
class Checked:
    value: int

    def __post_init__(self):
        TRACE.append(('checked', self.value))
        if self.value < 0:
            raise ValueError('negative fixture value')


@dataclass
class Parent:
    child: Checked


def test_one_snapshot_before_iteration_and_mapping_changes_are_rechecked_next_batch(monkeypatch):
    tags = {Left: ('kind', 'left'), Right: ('kind', 'right')}
    admitted = []
    original = codec._record_tags

    def count(value):
        admitted.append(value)
        return original(value)

    monkeypatch.setattr(codec, '_record_tags', count)

    def rows():
        assert len(admitted) == 1
        yield {'value': 1, 'kind': 'left'}
        tags[Left] = ('kind', 'changed')
        yield {'value': 2, 'kind': 'left'}

    assert codec.decode_records(Left | Right, rows(), record_tags=tags) == (Left(1), Left(2))
    assert len(admitted) == 1
    with pytest.raises(ValueError):
        codec.decode_records(Left, [{'value': 1, 'kind': 'left'}], record_tags=tags)
    assert codec.decode_records(Left, [{'value': 1, 'kind': 'changed'}], record_tags=tags) == (Left(1),)
    tags[Left] = ('value', 'conflicts')
    consumed = []

    def empty():
        consumed.append(True)
        return
        yield

    with pytest.raises(ValueError, match='conflicts'):
        codec.decode_records(Left, empty(), record_tags=tags)
    assert consumed == []


def test_union_branch_validation_and_nested_post_init_trace_match_individual_decode():
    rows = [{'value': 1}, {'value': 2}]
    expected = InvalidAfterInit | Checked
    TRACE.clear()
    original = tuple(decode(expected, row) for row in rows)
    trace = list(TRACE)
    TRACE.clear()
    actual = codec.decode_records(expected, rows)
    assert actual == original and all(type(row) is Checked for row in actual)
    assert TRACE == trace
    nested = [{'child': {'value': 3}}, {'child': {'value': 4}}]
    TRACE.clear()
    original = tuple(decode(Parent, row) for row in nested)
    trace = list(TRACE)
    TRACE.clear()
    assert codec.decode_records(Parent, nested) == original
    assert TRACE == trace


@pytest.mark.parametrize('expected,wire,tags', [
    (Left, {'value': True}, None),
    (Left, {'value': 1, 'extra': False}, None),
    (Left, {}, None),
    (Left, {'value': 1, 'kind': 'wrong'}, {Left: ('kind', 'left')}),
    (Left, encode(Extended(1, 'subclass')), None),
    (ObjectID, {'kind': 'alloc', 'value': 'x', 'wire': 'alloc:x'}, None),
    (ValueVersionID, {**encode(ValueVersionID(LogicalValueID('x'), 1)), 'version': True}, None),
    (PythonLiteral, {'kind': 'INT', 'payload': '01', 'items': []}, None),
    (Checked, {'value': -1}, None),
])
def test_malformed_rows_propagate_the_original_exception(expected, wire, tags):
    try:
        decode(expected, deepcopy(wire), record_tags=tags)
    except (ValueError, TypeError) as error:
        error_type, message = type(error), str(error)
    else:
        raise AssertionError('fixture must fail the original decoder')
    with pytest.raises(error_type) as actual:
        codec.decode_records(expected, [deepcopy(wire)], record_tags=tags)
    assert str(actual.value) == message
