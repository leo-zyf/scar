"""C++20 sparse symbol ABI shared by the owner and isolated workers."""

import ctypes
from pathlib import Path
from typing import NamedTuple

from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.superlu-symbolic-study'


class Timings(ctypes.Structure):
    _fields_ = [(name, ctypes.c_double) for name in ('ordering_s', 'etree_s', 'factor_s', 'solve_s')]

    def measurements(self):
        return {name: getattr(self, name) for name, _ in self._fields_}


class SymbolicABI(NamedTuple):
    create: object
    bytes: object
    destroy: object
    solve: object
    control: object


def read_native_manifest(reference):
    record = read_artifact(reference)
    directory = Path(reference['uri']).parent
    if (reference['schema'] != SCHEMA
            or (record['schema_version'], record['abi_version'], record['language'], record['scope'])
            != (1, 1, 'c++20', 'EXPLICIT_STUDY')
            or record['library']['schema'] != 'scar.native-library'
            or record['library']['uri'] != str(directory / 'sparse_lu.so')
            or record['blas']['schema'] != 'scar.provider-binary'):
        raise ValueError('unsupported sparse symbolic C++ artifact')
    return record


def validate_native_elf(record):
    for reference in (record['library'], record['blas']):
        read_artifact(reference, json_document=False)


def bind_symbolic_abi(library):
    pointer, number = ctypes.c_void_p, ctypes.c_int
    size, timing = ctypes.c_size_t, ctypes.POINTER(Timings)
    abi = library.scar_superlu_abi_v1
    abi.argtypes, abi.restype = [], number
    if abi() != 1:
        raise ValueError('sparse symbolic native ABI differs')

    run_args = [number, number, pointer, pointer, pointer, pointer, pointer, timing]
    create = library.scar_superlu_symbolic_v1
    create.argtypes, create.restype = [number, number, pointer, pointer, size, timing], pointer
    retained_bytes = library.scar_superlu_symbolic_bytes_v1
    retained_bytes.argtypes, retained_bytes.restype = [pointer], size
    destroy = library.scar_superlu_symbolic_destroy_v1
    destroy.argtypes, destroy.restype = [pointer], None
    solve = library.scar_superlu_symbolic_solve_v1
    solve.argtypes, solve.restype = [pointer, *run_args], number
    control = library.scar_superlu_reference_v1
    control.argtypes, control.restype = run_args, number
    return SymbolicABI(create, retained_bytes, destroy, solve, control)
