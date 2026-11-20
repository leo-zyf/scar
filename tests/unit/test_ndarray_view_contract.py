"""Allocation observations retain offsets and metadata, independent of writes."""

import operator

import numpy as np

from scar.analysis.ndarray_view_contract import (NdarrayViewFrontier, inplace_ndarray_witness,
                                                 ndarray_view_contract)


def test_relative_geometry_and_normal_inplace_result():
    allocation = np.zeros((7, 11), dtype=np.float64)
    view = allocation[1:-1, 1:-1]
    reverse = allocation[::-1, 0]
    broadcast = np.broadcast_to(allocation[:1, 1:2], (7, 11))
    root, interior, border, repeated = map(ndarray_view_contract, (allocation, view, reverse, broadcast))
    assert root.same_storage(interior) and interior.same_storage(border) and border.same_storage(repeated)
    assert interior.geometry() == dict(shape=(5, 9), byte_strides=(88, 8), byte_offset=96,
        allocation_nbytes=616, itemsize=8, dtype=np.dtype(np.float64).str, writeable=True, byte_bounds=(96, 520))
    assert border.byte_strides == (-88,) and border.byte_bounds == (0, 536)
    assert repeated.byte_strides == (0, 0) and repeated.byte_bounds == (8, 16) and not repeated.writeable
    assert ndarray_view_contract(allocation[:0, 2:]).byte_bounds == (16, 16)

    result = operator.iadd(view, 0.25)
    same = inplace_ndarray_witness(interior, result)
    assert same.same_object and same.same_owner and same.same_geometry
    distinct = inplace_ndarray_witness(interior, result.copy())
    assert not distinct.same_object and not distinct.same_owner


def test_external_buffers_and_subclass_callbacks_remain_open():
    class ForeignArray(np.ndarray):
        @property
        def __array_interface__(self):
            raise AssertionError('foreign getter must not run')

    app = ndarray_view_contract(np.zeros(4).view(ForeignArray))
    external = ndarray_view_contract(np.frombuffer(bytearray(64), dtype=np.float64))
    synthetic = ndarray_view_contract(np.lib.stride_tricks.as_strided(np.zeros(4), shape=(2,), strides=(8,)))
    assert app == NdarrayViewFrontier('EXACT_LOADED_NDARRAY_REQUIRED')
    assert external == NdarrayViewFrontier('EXTERNAL_OR_SYNTHETIC_BUFFER_OWNER', (8,), (8,))
    assert synthetic == NdarrayViewFrontier('EXTERNAL_OR_SYNTHETIC_BUFFER_OWNER', (2,), (8,))
