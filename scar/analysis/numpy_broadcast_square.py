"""Qualify the exact binary64 broadcast-square branch of a NumPy CPU provider."""

import argparse
import ctypes
from hashlib import sha256
import json
from pathlib import Path
from struct import unpack_from
import sys

from scar.codegen.stateful import _CAPSULE, _Info
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.x86-f64-broadcast-square-provider'
_LIBRARY_SHA256 = '27bd88fd60f4b9fcd5c938572e9b9e3df0aed9cfed4888e57d94c516b45512b7'
_FUNCTION = 0x39E540
_FUNCTION_SIZE = 0x213
_EXPONENT = 0x831628
_GUARDS = {
    0x39E551: '488b5a084c8b2e4885db0f857f000000',  # steps[1] == 0
    0x39E707: '660f2e05192f49007a327530',  # ordered compare with exact binary64 2
    0x39E720: 'f2410f1004244883c3014d01fcf20f59c0f20f1145004c01f54939dd75e2',
}


def _require(condition, detail):
    if not condition:
        raise ValueError('NumPy binary64 broadcast square: ' + detail)


def _contract(reference):
    raw = read_artifact(reference, json_document=False)
    _require(reference['sha256'] == _LIBRARY_SHA256 and raw.startswith(b'\x7fELF\x02\x01')
             and unpack_from('<H', raw, 18) == (62,), 'unsupported NumPy x86_64 provider binary')
    for offset, text in _GUARDS.items():
        expected = bytes.fromhex(text)
        _require(raw[offset:offset + len(expected)] == expected, 'provider guard or square loop differs')
    _require(unpack_from('<d', raw, _EXPONENT) == (2.0,), 'broadcast branch exponent differs')
    return dict(schema=SCHEMA, schema_version=1, provider=reference, numpy_version='2.5.2',
                ndarray_c_abi=33554432, routine='DOUBLE_power', routine_offset=_FUNCTION,
                routine_sha256=sha256(raw[_FUNCTION:_FUNCTION + _FUNCTION_SIZE]).hexdigest(),
                dtype='float64', exponent=2.0, fixed_strides=[8, 0, 8],
                operation='MULTIPLY_F64_SELF', instruction='mulsd %xmm0,%xmm0',
                instruction_offset=0x39E72D, requires_pyapi=False,
                capability_guard='MATCHED_X86_64_WITHOUT_AVX512F_AND_ROUND_TO_NEAREST',
                excludes_floating_status_as_output=True)


def observe():
    import numpy as np
    from numpy._core import _multiarray_umath as provider

    _require(sys.platform == 'linux' and sys.byteorder == 'little' and ctypes.sizeof(ctypes.c_void_p) == 8,
             'unsupported runtime process ABI')
    library = Path(provider.__file__).resolve(strict=True)
    contract = _contract(artifact_reference(library, 'scar.numpy-provider-elf'))
    _require(np.__version__ == contract['numpy_version'] and provider.power is np.power
             and provider._get_ndarray_c_version() == contract['ndarray_c_abi']
             and provider.__cpu_features__['AVX2'] is True and provider.__cpu_features__['AVX512F'] is False,
             'original NumPy callable and CPU dispatch domain required')
    dtype = np.dtype(contract['dtype'])
    resolved, capsule = provider.power._resolve_dtypes_and_context((dtype, int, None))
    _require(resolved == (dtype,) * 3, 'Python scalar exponent did not resolve to binary64')
    provider.power._get_strided_loop(capsule, fixed_strides=tuple(contract['fixed_strides']))
    pointer = ctypes.pythonapi.PyCapsule_GetPointer
    pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    pointer.restype = ctypes.c_void_p
    info = _Info.from_address(pointer(capsule, _CAPSULE))
    _require(bool(info.loop and info.auxdata) and not info.requires_pyapi, 'requires the pinned legacy CPU loop')
    with Path('/proc/self/maps').open() as stream:
        bases = [int(fields[0].split('-', 1)[0], 16) for row in stream
                 if len(fields := row.split(maxsplit=5)) == 6
                 and fields[2] == '00000000' and fields[5].strip() == str(library)]
    _require(len(bases) == 1, 'could not bind the observed legacy provider to its ELF mapping')
    # In this pinned binary the legacy adapter stores the scalar loop after its PyAuxData header.
    selected = (ctypes.c_void_p * 5).from_address(info.auxdata)[4]
    _require(selected == bases[0] + _FUNCTION, 'the resolved NumPy loop selected another provider')
    return dict(contract, avx2=True, avx512f=False, selection='LIVE_UFUNC_CAPSULE_ELF_RELATIVE_TARGET')


def check(path):
    receipt = read_artifact(artifact_reference(Path(path).resolve(strict=True), SCHEMA))
    expected = dict(_contract(receipt['provider']), avx2=True, avx512f=False,
                    selection='LIVE_UFUNC_CAPSULE_ELF_RELATIVE_TARGET')
    _require(receipt == expected, 'broadcast square receipt differs from the checked binary branch')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--out', type=Path)
    group.add_argument('--check', type=Path)
    args = parser.parse_args(argv)
    result = check(args.check) if args.check else observe()
    if args.out:
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    print(json.dumps({key: result[key] for key in ('routine', 'operation', 'fixed_strides')}, sort_keys=True))


if __name__ == '__main__':
    main()
