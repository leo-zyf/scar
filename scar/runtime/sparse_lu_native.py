"""Sparse lu native."""

import ctypes
from pathlib import Path
from typing import NamedTuple
from scar.ir.execution_recipe import read_artifact, artifact_reference
from threading import Lock
from time import perf_counter_ns
from types import BuiltinFunctionType, ModuleType
from weakref import finalize


# C++20 sparse symbol ABI shared by the owner and isolated workers.
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


# Explicit SuperLU study retaining only structure-dependent ordering.
_Timings = Timings
_TEMPLATE = Path(__file__).resolve().parents[1] / 'codegen' / 'sparse_lu_v1.cpp.in'


class _Solver:
    def __init__(self, library, original, budget):
        import numpy as np
        from numpy._core import _multiarray_umath

        self.provider, self.base = original.__self__, original
        self.ndarray, self.empty, self.isfinite = np.ndarray, np.empty, np.isfinite
        if (self.empty is not _multiarray_umath.empty
                or type(self.empty) is not BuiltinFunctionType or self.isfinite is not _multiarray_umath.isfinite
                or type(self.isfinite) is not np.ufunc):
            raise ValueError('original SuperLU and NumPy provider bindings required')
        self.budget, self.handle, self.dispose = budget, None, None
        self.lock = Lock()
        self.stats = dict(builds=0, build_ns=0, retained_bytes=0, native_calls=0, fallback_calls=0,
            hits=0, structure_misses=0, numeric_failures=0, provider_mismatches=0, closed=False,
            analysis=None, last_solve=None, initial_reference_ns=None)
        self.disabled = False
        abi = bind_symbolic_abi(library)
        self.create, self.solve = abi.create, abi.solve
        self.bytes, self.destroy, self.control = abi.bytes, abi.destroy, abi.control
        self.f64, self.i32 = np.dtype('float64'), np.dtype('int32')

    def _arguments(self, args, options):
        if (vars(self.provider).get('gssv') is not self.base or type(args) is not tuple or len(args) != 7
                or type(options) is not dict or len(options) != 1
                or any(type(key) is not str for key in options) or 'ColPerm' not in options
                or options['ColPerm'] is not None):
            return None
        size, nnz, data, indices, indptr, rhs, flag = args
        if (type(size) is not int or type(nnz) is not int or not 0 < size < 2**31-1
                or not 0 < nnz < 2**31-1 or type(flag) is not int or flag != 0):
            return None
        contract = ((indices, nnz, self.i32), (indptr, size+1, self.i32),
            (data, nnz, self.f64), (rhs, size, self.f64))
        addresses = []
        for value, length, dtype in contract:
            if (type(value) is not self.ndarray or value.shape != (length,) or value.dtype != dtype
                    or not value.flags.c_contiguous or not value.flags.aligned):
                return None
            addresses.append(value.__array_interface__['data'][0])
        if not self.isfinite(data).all() or not self.isfinite(rhs).all():
            return None
        return size, nnz, *addresses

    def reference(self, args, options):
        """Run the separately built public dgssv driver for controlled comparison."""
        pointers = self._arguments(args, options)
        if pointers is None:
            return None
        output = self.empty(pointers[0], dtype=self.f64)
        time = _Timings()
        status = self.control(*pointers, output.__array_interface__['data'][0], ctypes.byref(time))
        return output, status, time.measurements()

    def __call__(self, args, options):
        with self.lock:
            if not self.stats['closed'] and not self.disabled:
                arguments = self._arguments(args, options)
                if arguments is not None:
                    result = self._run(args, options, arguments)
                    if result is not None:
                        self.stats['native_calls'] += 1
                        return result
            self.stats['fallback_calls'] += 1
            return None

    def _run(self, args, options, pointers):
        first = self.handle is None
        if first:
            time = _Timings()
            started = perf_counter_ns()
            handle = self.create(*pointers[:4], self.budget, ctypes.byref(time))
            self.stats['build_ns'] += perf_counter_ns() - started
            if handle is None:
                return None
            self.dispose = finalize(self, self.destroy, handle)
            self.dispose.atexit = False
            self.handle = handle
            self.stats['builds'] += 1
            self.stats['retained_bytes'] = self.bytes(handle)
            self.stats['analysis'] = time.measurements()
        output = self.empty(pointers[0], dtype=self.f64)
        time = _Timings()
        status = self.solve(self.handle, *pointers, output.__array_interface__['data'][0], ctypes.byref(time))
        self.stats['last_solve'] = time.measurements()
        if status == -1:
            self.stats['structure_misses'] += 1
        elif status != 0:
            self.stats['numeric_failures'] += 1
        if status != 0:
            if first:
                self._release()
            return None
        if first:
            started = perf_counter_ns()
            try:
                expected, info = self.base(*args, options=options)
            except (ValueError, TypeError, RuntimeError, MemoryError, OverflowError):
                expected, info = None, -1
            self.stats['initial_reference_ns'] = perf_counter_ns() - started
            if info != 0 or expected.tobytes() != output.tobytes():
                self.stats['provider_mismatches'] += 1
                self.disabled = True
                self._release()
                return None
        else:
            self.stats['hits'] += 1
        return output, 0

    def _release(self):
        if self.dispose is not None:
            self.dispose()
        self.handle = None
        self.stats['retained_bytes'] = 0

    def close(self):
        with self.lock:
            self._release()
            self.stats['closed'] = True


def load_symbolic_solve(native_dir, *, budget_bytes=8 * 1024 * 1024, original=None):
    """Load against a captured carrier and artifact reference, or a diagnostic directory."""
    if type(budget_bytes) is not int or not 0 < budget_bytes < (1 << (8 * ctypes.sizeof(ctypes.c_size_t))):
        raise ValueError('positive SuperLU study memory budget required')
    reference = native_dir if type(native_dir) is dict else artifact_reference(
        Path(native_dir).resolve() / 'artifact.json', SCHEMA)
    manifest = read_native_manifest(reference)
    directory = Path(reference['uri']).parent
    if original is None:
        import scipy
        from scipy.sparse.linalg._dsolve import _superlu

        if manifest['provider_version'] != scipy.__version__:
            raise ValueError('SuperLU diagnostic provider version differs')
        original = _superlu.gssv
    provider = original.__self__ if type(original) is BuiltinFunctionType else None
    if (type(provider) is not ModuleType
            or original.__name__ != 'gssv' or vars(provider).get('gssv') is not original
            or type(vars(provider).get('__file__')) is not str
            or manifest['wheel'] != artifact_reference(provider.__file__, 'scar.provider-binary')
            or manifest['source']['uri'] != str(directory / 'sparse_lu.cpp')
            or manifest['template'] != artifact_reference(_TEMPLATE, 'scar.cpp-source')
            or manifest['source']['sha256'] != manifest['template']['sha256']):
        raise ValueError('SuperLU study artifact or provider differs')
    for dependency in (*manifest['vendor'], manifest['configuration'], manifest['source']):
        read_artifact(dependency, json_document=False)
    validate_native_elf(manifest)
    return _Solver(ctypes.PyDLL(manifest['library']['uri']), original, budget_bytes)
