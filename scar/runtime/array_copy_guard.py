"""Live admission for the exact ndarray C-order-copy primitive.

The immutable extension type and its method descriptor are anchored before
application entry. Module aliases are mutable and never establish that an
arbitrary object's ``copy`` is pure. Content stability remains the explicit
synchronous/no-foreign-writer caller contract, not an inferred version cache.
"""
from types import MethodDescriptorType

import numpy as _np
try:
    from numpy._core import _multiarray_umath as _core
except ModuleNotFoundError:
    from numpy.core import _multiarray_umath as _core

COPY_METHOD_GUARD='EXACT_NUMPY_NDARRAY_COPY_METHOD_DESCRIPTOR'
COPY_STABILITY_GUARD='ARRAY_INPUT_CONTENT_STABLE_DURING_REGION'
COPY_CONTRACT='numpy.ndarray.copy.content-fresh.v1'
_NDARRAY=getattr(_core,'ndarray',None)
_COPY=None if _NDARRAY is None else vars(_NDARRAY).get('copy')
_EMPTY=getattr(_core,'empty',None);_DTYPE=getattr(_core,'dtype',None)
_FLOAT64=None if _DTYPE is None else _DTYPE('float64').type
_IMMUTABLE_TYPE_FLAG=1<<8


class CopyGuardFailure(ValueError):pass


def runtime_copy_contract():
    return dict(order='C',contract=COPY_CONTRACT,
        required_guards=[COPY_METHOD_GUARD,COPY_STABILITY_GUARD])


def requires_copy_guard(assumptions):
    return COPY_METHOD_GUARD in assumptions


def guard_copy_inputs(values):
    """Check primitive/type identity before touching any input metadata.

    This executes no user ``copy`` method and reads no array values. The
    caller must separately hold the declared stable-input execution contract.
    """
    if (_NDARRAY is None or _EMPTY is None or _DTYPE is None or _FLOAT64 is None
            or not _NDARRAY.__flags__&_IMMUTABLE_TYPE_FLAG
            or type(_COPY)is not MethodDescriptorType or _COPY.__objclass__ is not _NDARRAY
            or _COPY.__name__!='copy' or vars(_NDARRAY).get('copy')is not _COPY
            or _core.ndarray is not _NDARRAY or _np.ndarray is not _NDARRAY
            or _core.empty is not _EMPTY or _core.dtype is not _DTYPE
            or _np.empty is not _EMPTY or _np.dtype is not _DTYPE or _np.float64 is not _FLOAT64):
        raise CopyGuardFailure('canonical immutable ndarray copy primitive/runtime aliases changed')
    if any(type(value)is not _NDARRAY for value in values):
        raise CopyGuardFailure('copy primitive requires exact core numpy.ndarray inputs')


__all__=['COPY_METHOD_GUARD','COPY_STABILITY_GUARD','COPY_CONTRACT','CopyGuardFailure',
    'runtime_copy_contract','requires_copy_guard','guard_copy_inputs']
