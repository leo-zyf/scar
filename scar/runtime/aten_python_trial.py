"""Temporarily attach a pinned ATen differential trial to its original Python caller."""

from contextlib import contextmanager
from functools import wraps
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from math import prod
import os
from pathlib import Path
import sys
import sysconfig
from threading import get_ident
from types import BuiltinFunctionType, FunctionType, ModuleType

from scar.ir.record_codec import loads
from scar.trace.code_snapshot import code_key


_SCHEMA = 'scar.aten-python-trial'
_SCOPE = 'EXPLICIT_PYTHON_CPU_DIFFERENTIAL_TRIAL'
_EXTENSION = '_scar_aten_python_trial'


def _verify(path, receipt, *, retain=False):
    digest = sha256()
    with path.open('rb') as member:
        if retain:
            contents = member.read(receipt['size_bytes'] + 1)
            digest.update(contents)
            size = len(contents)
        else:
            size = 0
            while data := member.read(1024 * 1024):
                digest.update(data)
                size += len(data)
    if size != receipt['size_bytes'] or digest.hexdigest() != receipt['sha256']:
        raise ValueError(f'ATen trial member differs from pinned bytes: {path}')
    if retain:
        return contents


def _read_manifest(directory, expected_sha256):
    with (directory / 'trial.json').open('rb') as stream:
        raw = stream.read(1_000_001)
    if len(raw) > 1_000_000 or sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('ATen trial manifest differs from its pin')
    manifest = loads(raw)
    if (manifest.get('schema') != _SCHEMA or manifest.get('schema_version') != 1
            or manifest.get('scope') != _SCOPE or manifest.get('source_replacement_authorized') is not False):
        raise ValueError('an explicit ATen Python differential trial is required')
    return manifest


def _provider(artifact, torch):
    if (artifact['name'], artifact['version'], artifact['revision']) != (
            'LibTorch', torch.__version__, torch.version.git_version):
        raise ValueError('the active Torch build differs from the ATen trial provider')
    lib = Path(torch.__file__).resolve().parent / 'lib'
    libraries = {Path(row['path']): row for row in artifact['libraries']}
    expected = {lib / name for name in ('libc10.so', 'libtorch_cpu.so', 'libtorch_python.so')}
    if libraries.keys() != expected or len(libraries) != len(artifact['libraries']):
        raise ValueError('the ATen trial requires the active Torch native library paths')
    names = {path.name for path in expected}
    executable = {}
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or 'x' not in fields[1] or not fields[5].startswith('/'):
            continue
        path = Path(fields[5].removesuffix(' (deleted)'))
        if path.name in names:
            major, minor = (int(part, 16) for part in fields[3].split(':'))
            executable.setdefault(path, set()).add((os.makedev(major, minor), int(fields[4])))
    if executable.keys() != expected:
        raise ValueError('live Torch executable mappings differ from the ATen trial provider')
    for path, receipt in libraries.items():
        status = path.stat()
        if executable[path] != {(status.st_dev, status.st_ino)}:
            raise ValueError(f'live Torch executable mapping differs from its filesystem member: {path}')
        _verify(path, receipt)


def _original(module, proof):
    name = proof['function']
    if type(module) is not ModuleType:
        raise ValueError('the trial requires an ordinary source module')
    namespace = vars(module)
    original = namespace.get(name)
    source = Path(proof['source']['path'])
    if (type(original) is not FunctionType or original.__globals__ is not namespace
            or original.__module__ != namespace.get('__name__') or original.__qualname__ != name
            or original.__closure__ or original.__defaults__ is not None or original.__kwdefaults__ is not None
            or Path(namespace.get('__file__', '.')).resolve() != source
            or Path(original.__code__.co_filename).resolve() != source
            or code_key(original.__code__) != proof['code_key']):
        raise ValueError('the named live callable differs from the attested source boundary')
    _verify(source, proof['source'])
    return namespace, original


def _builtin_wrappers(torch, proof):
    canonical = vars(torch._C._VariableFunctionsClass)
    names = [row['attribute'] for row in proof['wrappers']]
    guarded = {}
    for name in names:
        descriptor = canonical.get(name)
        if type(descriptor) is not staticmethod or type(descriptor.__func__) is not BuiltinFunctionType:
            raise ValueError(f'Torch wrapper has no exact builtin binding: {name}')
        if vars(torch).get(name) is not descriptor.__func__:
            raise ValueError(f'live Torch wrapper differs from the registered builtin: {name}')
        guarded[name] = descriptor
    return guarded


def _extension(directory, receipt):
    name = _EXTENSION + (sysconfig.get_config_var('EXT_SUFFIX') or '')
    if receipt['name'] != name:
        raise ValueError('ATen trial extension ABI differs from the active interpreter')
    path = directory / name
    if path.resolve(strict=True) != path:
        raise ValueError('ATen trial extension must be the pinned package member')
    _verify(path, receipt)
    spec = spec_from_file_location(_EXTENSION, path)
    if spec is None or spec.loader is None:
        raise ValueError('cannot resolve the pinned ATen trial extension')
    extension = module_from_spec(spec)
    spec.loader.exec_module(extension)
    return extension


def _observed(code):
    if sys.getprofile() is not None or sys.gettrace() is not None:
        return True
    monitor = getattr(sys, 'monitoring', None)
    return monitor is not None and any(
        monitor.get_events(tool) or monitor.get_local_events(tool, code)
        for tool in range(6) if monitor.get_tool(tool) is not None)


def _safe_torch_context(torch):
    c = torch._C
    return (not torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
            and not c._is_any_autocast_enabled() and not c._are_functorch_transforms_active()
            and not c._len_torch_dispatch_stack() and not c._len_torch_function_stack()
            and c._get_torch_function_state().name == 'ENABLED' and not c._peek_should_skip_torch_function()
            and not c._autograd._profiler_enabled() and torch.get_num_threads() == 1
            and torch.get_num_interop_threads() == 1)


def _torch_configuration(torch):
    c = torch._C
    return (c._dispatch_tls_local_include_set(), c._dispatch_tls_local_exclude_set(),
            torch.get_float32_matmul_precision(), c._get_mkldnn_enabled(),
            torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())


def _ready_inputs(inputs, shapes, lengths, torch):
    extents = []
    for tensor, shape, size in zip(inputs, shapes, lengths, strict=True):
        if (type(tensor) is not torch.Tensor or tensor.device.type != 'cpu' or tensor.dtype is not torch.float32
                or tensor.layout is not torch.strided or tensor.requires_grad or tensor.is_inference()
                or tensor.is_conj() or tensor.is_neg() or tensor.storage_offset() != 0
                or tensor.shape != shape or not tensor.is_contiguous() or tensor.untyped_storage().nbytes() != size):
            return False
        if size:
            start = tensor.data_ptr()
            extents.append((start, start + size))
    extents.sort()
    return all(first[1] <= second[0] for first, second in zip(extents, extents[1:]))


@contextmanager
def install_aten_python_trial(module, trial_dir, *, expected_sha256, exclusive_inputs=False):
    """Yield trial counters while the exact module binding has a guarded candidate.

    The caller must hold exclusive read access to Tensor contents for the
    duration of each call. The original remains the default execution path.
    """
    if exclusive_inputs is not True:
        raise ValueError('the ATen Python trial requires explicit exclusive inputs')
    import torch

    directory = Path(trial_dir).resolve(strict=True)
    manifest = _read_manifest(directory, expected_sha256)
    proof = manifest['source_proof']
    namespace, original = _original(module, proof)
    torch_global = proof['torch_global']
    if namespace.get(torch_global) is not torch:
        raise ValueError('the source Torch global does not bind the active provider')
    _provider(manifest['provider'], torch)
    wrappers = _builtin_wrappers(torch, proof)
    code = original.__code__
    cpus = tuple(manifest['cpus'])
    allowed_cpus = set(cpus)
    if (len(cpus) != 2 or len(allowed_cpus) != 2 or os.sched_getaffinity(0) != allowed_cpus
            or not _safe_torch_context(torch) or _observed(code)
            or not getattr(sys, '_is_gil_enabled', lambda: True)()):
        raise ValueError('the live CPU, Python or Torch context differs from the ATen trial scope')
    torch_config = _torch_configuration(torch)
    provider_class = torch._C._VariableFunctionsClass
    input_order = tuple(proof['input_order'])
    shapes = tuple(tuple(item['shape']) for item in manifest['recipe']['inputs'])
    lengths = tuple(4 * prod(shape) for shape in shapes)
    receipts = manifest['inputs']
    if (len(input_order) != len(shapes) or len(receipts) != len(shapes)
            or tuple(row['size_bytes'] for row in receipts) != lengths):
        raise ValueError('the retained raw inputs differ from the graph ABI')
    references = tuple(_verify(Path(row['path']), row, retain=True) for row in receipts)
    extension = _extension(directory, manifest['extension'])
    capsule = extension.prepare(*cpus)
    native = extension.run
    windows = tuple(manifest['window_starts'])
    arguments = len(proof['parameters'])
    tensor_result = proof['return_kind'] == 'TENSOR'
    owner = get_ident()
    process = os.getpid()
    active = True
    stats = dict(native_calls=0, fallback_calls=0, fallback_reasons={})

    def fallback(reason, args, kwargs):
        stats['fallback_calls'] += 1
        reasons = stats['fallback_reasons']
        reasons[reason] = reasons.get(reason, 0) + 1
        return original(*args, **kwargs)

    @wraps(original)
    def guarded(*args, **kwargs):
        if not active or kwargs or len(args) != arguments:
            return fallback('CALL_ABI', args, kwargs)
        if (os.getpid() != process or get_ident() != owner or os.sched_getaffinity(0) != allowed_cpus
                or _observed(code) or not _safe_torch_context(torch) or _torch_configuration(torch) != torch_config):
            return fallback('RUNTIME_CONTEXT', args, kwargs)
        if (original.__code__ is not code or original.__defaults__ is not None or original.__kwdefaults__ is not None
                or namespace.get(torch_global) is not torch or torch._C._VariableFunctionsClass is not provider_class
                or any(vars(torch).get(name) is not descriptor.__func__ or vars(provider_class).get(name) is not descriptor
                       for name, descriptor in wrappers.items())):
            return fallback('SOURCE_BINDING', args, kwargs)
        inputs = tuple(args[index] for index in input_order)
        if not _ready_inputs(inputs, shapes, lengths, torch):
            return fallback('TENSOR_ABI', args, kwargs)
        try:
            result = native(capsule, inputs, windows, references)
        except (MemoryError, RuntimeError, ValueError):
            return fallback('NATIVE_ERROR', args, kwargs)
        if result is None:
            return fallback('INPUT_BYTES', args, kwargs)
        stats['native_calls'] += 1
        return result[0] if tensor_result else result

    name = proof['function']
    namespace[name] = guarded
    try:
        yield stats
    finally:
        active = False
        if namespace.get(name) is guarded:
            namespace[name] = original
        capsule = None
        references = ()


__all__ = ['install_aten_python_trial']
