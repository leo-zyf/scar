"""Read the pinned deployment metadata without loading the native library."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import re

from .loop_contract import validate_admission, validate_contract
from .native_loop import _small_raw, _json


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


__all__ = ['inspect_bundle']
