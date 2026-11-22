"""Array copy guard."""

from types import MethodDescriptorType
import numpy as _np
import re
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from scar.runtime.loop_contract import validate_admission, validate_contract
from scar.runtime.native_loop import _small_raw, _json


# Live admission for the exact ndarray C-order-copy primitive.
#
# The immutable extension type and its method descriptor are anchored before
# application entry. Module aliases are mutable and never establish that an
# arbitrary object's ``copy`` is pure. Content stability remains the explicit
# synchronous/no-foreign-writer caller contract, not an inferred version cache.
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


# Read the pinned deployment metadata without loading the native library.
def inspect_bundle(bundle, *, expected_admission_sha256):
    """Inspect metadata for cost planning; this is not native execution admission.

    Reuse the loader's bounded strict JSON reader. The library bytes, live CPU
    and function-table ABI are checked again by the real loader before use.
    """
    if (type(expected_admission_sha256) is not str or
            re.fullmatch('[0-9a-f]{64}', expected_admission_sha256) is None):
        raise ValueError('explicit expected admission SHA256 required')
    directory = Path(bundle).resolve()
    admission_raw = _small_raw(directory / 'admission.json')
    if sha256(admission_raw).hexdigest() != expected_admission_sha256:
        raise ValueError('admission SHA256 differs from explicit trust pin')
    document = _json(admission_raw)
    if document.get('schema') == 'scar.stateful-runtime-admission':
        from .stateful_contract import inspect_stateful_bundle
        return inspect_stateful_bundle(directory,
            expected_admission_sha256=expected_admission_sha256)
    admission = validate_admission(document)
    runtime_raw = _small_raw(directory / 'runtime.json')
    if sha256(runtime_raw).hexdigest() != admission['runtime_contract_sha256']:
        raise ValueError('runtime contract byte SHA256 differs')
    contract = validate_contract(_json(runtime_raw))
    if (contract['native']['ids'] != admission['native_ids'] or
            contract['native']['library_sha256'] != admission['library_sha256']):
        raise ValueError('runtime/native admission identities differ')
    return deepcopy(dict(contract=contract, admission=admission,
        admission_sha256=expected_admission_sha256, metadata_only=True,
        library_loaded=False, library_bytes_checked=False,
        execution_contract_must_be_revalidated=True))


__all__ = [
    'COPY_METHOD_GUARD',
    'COPY_STABILITY_GUARD',
    'COPY_CONTRACT',
    'CopyGuardFailure',
    'runtime_copy_contract',
    'requires_copy_guard',
    'guard_copy_inputs',
    'inspect_bundle',
]
