"""Explicit SuperLU study retaining only structure-dependent ordering."""

import ctypes
from pathlib import Path
from threading import Lock
from time import perf_counter_ns
from types import BuiltinFunctionType
from weakref import finalize

from scar.ir.execution_recipe import artifact_reference, read_artifact


class _Timings(ctypes.Structure):
    _fields_ = [(name, ctypes.c_double) for name in ('ordering_s', 'etree_s', 'factor_s', 'solve_s')]

    def measurements(self):
        return {name: getattr(self, name) for name, _ in self._fields_}


class _Solver:
    def __init__(self, library, provider, budget):
        import numpy as np
        from numpy._core import _multiarray_umath

        self.provider, self.base = provider, provider.gssv
        self.ndarray, self.empty, self.isfinite = np.ndarray, np.empty, np.isfinite
        if (type(self.base) is not BuiltinFunctionType or self.base.__self__ is not provider
                or self.base.__name__ != 'gssv' or self.empty is not _multiarray_umath.empty
                or type(self.empty) is not BuiltinFunctionType or self.isfinite is not _multiarray_umath.isfinite
                or type(self.isfinite) is not np.ufunc):
            raise ValueError('original SuperLU and NumPy provider bindings required')
        self.budget, self.handle, self.dispose = budget, None, None
        self.lock = Lock()
        self.stats = dict(builds=0, build_ns=0, retained_bytes=0, native_calls=0, fallback_calls=0,
            hits=0, structure_misses=0, numeric_failures=0, provider_mismatches=0, closed=False,
            analysis=None, last_solve=None, initial_reference_ns=None)
        self.disabled = False
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        metrics, size = ctypes.POINTER(_Timings), ctypes.c_size_t
        run_args = [integer, integer, pointer, pointer, pointer, pointer, pointer, metrics]
        library.scar_superlu_abi_v1.restype = integer
        if library.scar_superlu_abi_v1() != 1:
            raise ValueError('SuperLU study ABI differs')
        self.create = library.scar_superlu_symbolic_v1
        self.create.argtypes = [integer, integer, pointer, pointer, size, metrics]
        self.create.restype = pointer
        self.bytes = library.scar_superlu_symbolic_bytes_v1
        self.bytes.argtypes, self.bytes.restype = [pointer], size
        self.destroy = library.scar_superlu_symbolic_destroy_v1
        self.destroy.argtypes, self.destroy.restype = [pointer], None
        self.solve = library.scar_superlu_symbolic_solve_v1
        self.solve.argtypes, self.solve.restype = [pointer, *run_args], integer
        self.control = library.scar_superlu_reference_v1
        self.control.argtypes, self.control.restype = run_args, integer
        self.f64, self.i32 = np.dtype('float64'), np.dtype('int32')

    def _arguments(self, args, options):
        if (self.provider.gssv is not self.base or type(args) is not tuple or len(args) != 7
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


def load_symbolic_solve(native_dir, *, budget_bytes=8 * 1024 * 1024):
    """Load a guarded diagnostic callable. None requests the original wrapper."""
    import scipy
    from scipy.sparse.linalg._dsolve import _superlu
    from scar.codegen.sparse_lu import SCHEMA, SOURCE

    if type(budget_bytes) is not int or not 0 < budget_bytes < (1 << (8 * ctypes.sizeof(ctypes.c_size_t))):
        raise ValueError('positive SuperLU study memory budget required')
    directory = Path(native_dir).resolve()
    manifest = read_artifact(artifact_reference(directory / 'artifact.json', SCHEMA))
    if ((manifest['schema_version'], manifest['abi_version'], manifest['language'], manifest['scope'])
            != (1, 1, 'c++20', 'EXPLICIT_STUDY')
            or manifest['provider_version'] != scipy.__version__
            or manifest['library']['uri'] != str(directory / 'sparse_lu.so')
            or manifest['source']['uri'] != str(directory / 'sparse_lu.cpp')
            or manifest['template'] != artifact_reference(SOURCE, 'scar.cpp-source')
            or manifest['wheel'] != artifact_reference(_superlu.__file__, 'scar.provider-binary')
            or read_artifact(manifest['source'], json_document=False) != SOURCE.read_bytes()):
        raise ValueError('SuperLU study artifact or provider differs')
    for reference in (*manifest['vendor'], manifest['configuration'], manifest['blas'], manifest['library']):
        read_artifact(reference, json_document=False)
    return _Solver(ctypes.PyDLL(manifest['library']['uri']), _superlu, budget_bytes)
