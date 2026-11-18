"""Admit contiguous float64 CSR arrays before calling the C++ region."""

import ctypes
import inspect
from pathlib import Path
import sys
from threading import Lock
from time import perf_counter_ns
from types import BuiltinFunctionType, FunctionType, MappingProxyType
from weakref import finalize

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.trace.code_snapshot import code_key


def _function_state(function):
    return function.__code__, function.__defaults__, function.__kwdefaults__


def _same_state(function, state):
    return (function.__code__ is state[0] and function.__defaults__ is state[1]
        and function.__kwdefaults__ is state[2])


def _ranges(arrays):
    return [(array.__array_interface__['data'][0], array.nbytes) for array in arrays]


def _overlaps(left, right):
    start, length = left
    other, other_length = right
    return bool(length and other_length and start < other+other_length and other < start+length)


def _provider(expected=None, *, compile_sources=None):
    import numpy as np
    from numpy._core import fromnumeric, multiarray, shape_base
    from numpy._core.overrides import _ArrayFunctionDispatcher
    import scipy
    from scipy import sparse
    from scipy.sparse import _base, _compressed, _csr, _data, _matrix, _sparsetools, _sputils

    if scipy.__version__ != '1.18.1' or np.__version__ != '2.5.2':
        raise ValueError('CSR diagonal behavior requires the SciPy 1.18.1 / NumPy 2.5.2 provider profile')
    source_modules = (_base, _compressed, _csr, _data, _matrix, _sputils, fromnumeric, shape_base)
    sources = [artifact_reference(module.__file__, 'scar.provider-source') for module in source_modules]
    receipt = dict(scipy_version=scipy.__version__, numpy_version=np.__version__,
        sources=sources, sparse_binary=artifact_reference(_sparsetools.__file__, 'scar.provider-binary'),
        python_runtime=dict(implementation=sys.implementation.name,
            version=list(sys.version_info), optimize=sys.flags.optimize))
    if compile_sources is None and (type(expected) is not dict
            or any(expected.get(key) != value for key, value in receipt.items())):
        raise ValueError('CSR provider source, binary or interpreter differs from the compiled contract')
    classes = (sparse.csr_matrix, sparse.csr_array)
    method_sources = {
        'setdiag': (_base, '_spbase.setdiag', (0,)),
        '_setdiag': (_compressed, '_cs_matrix._setdiag', None),
        'diagonal': (_compressed, '_cs_matrix.diagonal', (0,)),
        '_swap': (_csr, '_csr_base._swap', None),
    }
    methods = {kind:{name:getattr(kind, name) for name in method_sources}
               for kind in classes}
    properties = {kind:{name:inspect.getattr_static(kind, name) for name in ('shape', 'dtype', 'ndim')}
                  for kind in classes}
    function_states = {}
    code_keys = {}

    def remember(function, module, qualname, defaults=None):
        if (type(function) is not FunctionType or function.__globals__ is not vars(module)
                or function.__module__ != module.__name__ or function.__qualname__ != qualname
                or function.__closure__ is not None or function.__kwdefaults__ is not None):
            raise ValueError('CSR provider function or defaults differ from the source profile')
        actual = function.__defaults__
        if actual is not defaults and (defaults is None or actual is None or len(actual) != len(defaults)
                or any(type(left) is not type(right) or left != right for left, right in zip(actual, defaults))):
            raise ValueError('CSR provider function or defaults differ from the source profile')
        key = code_key(function.__code__)
        binding = f'{module.__name__}:{qualname}'
        if code_keys.setdefault(binding, key) != key:
            raise ValueError('CSR provider method and array binding differ')
        function_states[function] = _function_state(function)

    for kind, members in methods.items():
        for name, function in members.items():
            remember(function, *method_sources[name])
        shape_source = (_matrix, 'spmatrix.get_shape') if kind is sparse.csr_matrix else (_base, '_spbase.shape')
        property_sources = dict(shape=shape_source, dtype=(_data, '_data_matrix.dtype'), ndim=(_base, '_spbase.ndim'))
        for name, value in properties[kind].items():
            if type(value) is not property:
                raise ValueError('original CSR provider property required')
            remember(value.fget, *property_sources[name])
    callbacks = {name:getattr(_sparsetools, name) for name in ('csr_diagonal', 'csr_sample_offsets')}
    numpy_owners = dict(arange=multiarray, asarray=multiarray, empty=multiarray, atleast_1d=shape_base,
        cumsum=fromnumeric, repeat=fromnumeric)
    np_functions = {name:getattr(owner, name) for name, owner in numpy_owners.items()}
    wrappers = {}
    for name, defaults in (('atleast_1d', None), ('cumsum', (None, None, None)), ('repeat', (None,))):
        function = np_functions[name]
        if type(function) is not _ArrayFunctionDispatcher:
            raise ValueError('original NumPy array-function dispatcher required')
        wrapper = function.__wrapped__
        remember(wrapper, numpy_owners[name], name, defaults)
        wrappers[name] = wrapper
    for name in ('arange', 'asarray', 'empty'):
        if type(np_functions[name]) is not BuiltinFunctionType or np_functions[name].__name__ != name:
            raise ValueError('original NumPy array-function binding required')
    wrapfunc = fromnumeric._wrapfunc
    remember(wrapfunc, fromnumeric, '_wrapfunc')
    upcast = _sputils.upcast
    remember(upcast, _sputils, 'upcast')
    receipt['python_code'] = code_keys
    if compile_sources is not None:
        modules = zip((module.__name__ for module in source_modules), sources)
        if compile_sources(modules, code_keys) != code_keys:
            raise ValueError('CSR provider live code differs from the build-time source')
    elif receipt != expected:
        raise ValueError('CSR provider live code differs from the build-time certificate')
    memo = _sputils._upcast_memo
    memo_key = hash((np.dtype('float64'),))
    supported_dtypes = tuple(_sputils.supported_dtypes)
    if type(memo) is not dict or type(_sputils.supported_dtypes) is not list:
        raise ValueError('original SciPy dtype memo required')
    np_bindings = (_base, _compressed)

    def live(kind, module):
        return (module is np and kind in methods
            and kind.__getattribute__ is object.__getattribute__
            and all(getattr(kind, name) is value and _same_state(value.fget, function_states[value.fget])
                    for name, value in properties[kind].items())
            and all(getattr(kind, name) is function and _same_state(function, function_states[function])
                    for name, function in methods[kind].items())
            and all(getattr(np, name) is value for name, value in np_functions.items())
            and all(np_functions[name].__wrapped__ is function and _same_state(function, function_states[function])
                    for name, function in wrappers.items())
            and fromnumeric._wrapfunc is wrapfunc and _same_state(wrapfunc, function_states[wrapfunc])
            and _compressed.upcast is upcast and _sputils.upcast is upcast and _same_state(upcast, function_states[upcast])
            and _sputils._upcast_memo is memo
            and all(getattr(module, 'np', None) is np for module in np_bindings)
            and all(getattr(_compressed, name, None) is value and getattr(_sparsetools, name) is value
                    for name, value in callbacks.items()))

    if not all(live(kind, np) for kind in classes):
        raise ValueError('CSR provider has overridden method or library bindings')

    def cold_upcast_supported():
        return (_sputils.np is np and np.result_type is multiarray.result_type
            and np.can_cast is multiarray.can_cast and type(_sputils.supported_dtypes) is list
            and tuple(_sputils.supported_dtypes) == supported_dtypes
            and next((dtype for dtype in supported_dtypes if np.can_cast(np.dtype('float64'), dtype)), None)
                is np.float64)

    return receipt, live, memo, memo_key, upcast, cold_upcast_supported


class _ScopedTopology:
    def __init__(self, library, signature, budget):
        pointer, size = ctypes.c_void_p, ctypes.c_size_t
        self.create = library.scar_csr_enforce_topology_create_v1
        self.create.argtypes = [*signature[1:-1], size]
        self.create.restype = pointer
        self.run = library.scar_csr_enforce_topology_run_v1
        self.run.argtypes = [pointer, *signature, ctypes.c_int]
        self.run.restype = ctypes.c_int32
        self.bytes = library.scar_csr_enforce_topology_bytes_v1
        self.bytes.argtypes = [pointer]
        self.bytes.restype = size
        self.destroy = library.scar_csr_enforce_topology_destroy_v1
        self.destroy.argtypes = [pointer]
        self.destroy.restype = None
        self.budget = budget
        self.handle = self.release = None
        self.lock = Lock()
        self.closed = False
        self.stats = dict(build_attempts=0, builds=0, build_ns=0, hits=0, misses=0,
            numeric_rejections=0, retained_bytes=0, closed=False)

    def enforce(self, args, dtype, cold_upcast):
        with self.lock:
            if self.closed:
                return -1
            return self._enforce(args, dtype, cold_upcast)

    def _enforce(self, args, dtype, cold_upcast):
        built = self.handle is None
        if built:
            self.stats['build_attempts'] += 1
            started = perf_counter_ns()
            self.handle = self.create(*args[1:-1], self.budget)
            self.stats['build_ns'] += perf_counter_ns() - started
            if self.handle is None:
                return 0
            self.release = finalize(self, self.destroy, self.handle)
            self.release.atexit = False
            self.stats['builds'] += 1
            self.stats['retained_bytes'] = self.bytes(self.handle)
        status = self.run(self.handle, *args, int(cold_upcast is None))
        if status == 1 and cold_upcast is not None:
            cold_upcast(dtype)
            status = self.run(self.handle, *args, 1)
        if status == 0:
            self.stats['misses'] += 1
        elif status < 0:
            self.stats['numeric_rejections'] += 1
        elif not built:
            self.stats['hits'] += 1
        return status

    def close(self):
        with self.lock:
            if self.release is not None:
                self.release()
            self.handle = None
            self.closed = self.stats['closed'] = True
            self.stats['retained_bytes'] = 0


def load_csr_enforce(out, *, reuse_topology=False, topology_budget_bytes=8 * 1024 * 1024):
    """Return a guarded callable, optionally owning one exact topology until close()."""
    import numpy as np
    from scar.codegen.sparse_csr import SCHEMA, SCHEMA_VERSION, SOURCE, TOPOLOGY_FEATURE

    if type(reuse_topology) is not bool:
        raise ValueError('explicit Boolean CSR topology selection required')
    if reuse_topology and (type(topology_budget_bytes) is not int
            or not 0 < topology_budget_bytes < (1 << (8 * ctypes.sizeof(ctypes.c_size_t)))):
        raise ValueError('positive CSR topology memory budget required')

    directory = Path(out).resolve()
    manifest = read_artifact(artifact_reference(directory/'artifact.json', SCHEMA))
    if manifest['schema_version'] != SCHEMA_VERSION:
        raise ValueError('rebuild CSR C++ artifact to certify provider code at build time')
    if reuse_topology and TOPOLOGY_FEATURE not in manifest.get('features', ()):
        raise ValueError('rebuild CSR C++ artifact to add scoped topology support')
    if ((manifest['abi_version'], manifest['language']) != (1, 'c++20')
            or manifest['source']['uri'] != str(directory/'sparse_csr.cpp')
            or manifest['library']['uri'] != str(directory/'sparse_csr.so')
            or manifest['template'] != artifact_reference(SOURCE, 'scar.cpp-source')
            or read_artifact(manifest['source'], json_document=False) != SOURCE.read_bytes()):
        raise ValueError('CSR C++ artifact differs from the admitted v1 template')
    _, live, memo, memo_key, upcast, cold_upcast_supported = _provider(manifest['provider'])
    read_artifact(manifest['library'], json_document=False)
    library = ctypes.PyDLL(manifest['library']['uri'])
    library.scar_csr_enforce_abi_v1.restype = ctypes.c_int32
    if library.scar_csr_enforce_abi_v1() != 1:
        raise ValueError('CSR C++ ABI mismatch')
    call = library.scar_csr_enforce_v1
    preflight = library.scar_csr_enforce_preflight_v1
    pointer, size = ctypes.c_void_p, ctypes.c_size_t
    signature = [pointer, pointer, pointer, size, size, ctypes.c_int,
        pointer, size, ctypes.c_int, ctypes.c_double]
    for function in (call, preflight):
        function.argtypes = signature
        function.restype = ctypes.c_int32
    topology = _ScopedTopology(library, signature, topology_budget_bytes) if reuse_topology else None
    indices_types = (np.dtype('int32'), np.dtype('int64'))
    float64 = np.dtype('float64')
    forbidden = frozenset(('shape', 'setdiag', '_setdiag', 'diagonal', '_swap', 'dtype', 'ndim'))

    def try_enforce(matrix, rows, diagonal, module=np, *, original=None):
        if (topology is not None and topology.closed) or not live(type(matrix), module):
            return False
        cached = memo.get(memo_key)
        if cached is not np.float64 and (cached is not None or original is None):
            return False
        fields = vars(matrix)
        if forbidden.intersection(fields):
            return False
        shape = fields.get('_shape')
        if (type(shape) is not tuple or len(shape) != 2 or type(shape[0]) is not int
                or type(shape[1]) is not int or shape[0] != shape[1] or shape[0] <= 0):
            return False
        values, indices, indptr = (fields.get(name) for name in ('data', 'indices', 'indptr'))
        if any(type(array) is not np.ndarray or array.ndim != 1 or not array.flags.c_contiguous
               or not array.flags.aligned for array in (values, indices, indptr, rows)):
            return False
        if (values.dtype != float64 or not values.flags.writeable or indices.dtype not in indices_types
                or indptr.dtype != indices.dtype or rows.dtype not in indices_types
                or indptr.size != shape[0]+1 or indices.size != values.size):
            return False
        if type(diagonal) not in (float, int, np.float64) or type(diagonal) is int and abs(diagonal) > 2**53:
            return False
        ranges = _ranges((values, indices, indptr, rows))
        if any(_overlaps(ranges[0], interval) for interval in ranges[1:]):
            return False
        if original is not None:
            if type(original) is not type(matrix) or original is matrix:
                return False
            source = vars(original)
            inputs = [source.get(name) for name in ('data', 'indices', 'indptr')]
            if any(type(value) is not np.ndarray or value.ndim != 1 or not value.flags.c_contiguous
                   for value in inputs):
                return False
            if any(_overlaps(a, b) for a in ranges[:3] for b in _ranges(inputs)):
                return False
        data, columns, offsets, selected = (address for address, _ in ranges)
        args = (data, columns, offsets, shape[0], values.size, indices.itemsize,
            selected, rows.size, rows.itemsize, float(diagonal))
        if cached is None and not cold_upcast_supported():
            return False
        if topology is not None:
            status = topology.enforce(args, values.dtype, upcast if cached is None else None)
            if status:
                return status > 0
        if cached is None:
            if preflight(*args) != 1:
                return False
            # Preserve SciPy's first memo update after every no-write guard.
            # The source clone is private under the caller's normal-completion contract.
            upcast(values.dtype)
        return call(*args) == 1

    if topology is not None:
        try_enforce.close = topology.close
        try_enforce.topology_stats = MappingProxyType(topology.stats)
    return try_enforce
