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
from time import perf_counter_ns
from types import BuiltinFunctionType, FunctionType, ModuleType

from scar.ir.record_codec import loads
from scar.runtime.aten_provider_identity import file_stat_identity, local_file_identity, verify_library_file
from scar.trace.code_snapshot import code_key


_SCHEMA = 'scar.aten-python-trial'
_SCOPE = 'EXPLICIT_PYTHON_CPU_DIFFERENTIAL_TRIAL'
_EXTENSION = '_scar_aten_python_trial'
_EXACT_BYTES = 'EXACT_BYTES'
_GEOMETRY = 'VALUE_INDEPENDENT_GEOMETRY'


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


def _trial_layout(manifest):
    cpus = manifest['cpus']
    if (type(cpus) is not list or len(cpus) not in (2, 3)
            or any(type(cpu) is not int or cpu < 0 for cpu in cpus) or len(set(cpus)) != len(cpus)):
        raise ValueError('the trial requires two or three distinct nonnegative CPU IDs')
    if 'ready_lanes' in manifest:
        lanes = manifest['ready_lanes']
        if (len(cpus) != 3 or 'window_groups' in manifest or 'window_starts' in manifest
                or type(lanes) is not list or len(lanes) != len(manifest['recipe']['operations'])
                or any(type(lane) is not int or not 0 <= lane < 3 for lane in lanes)
                or len(set(lanes)) < 2):
            raise ValueError('the ready DAG requires an exact three-core lane schedule')
        return tuple(cpus), tuple(lanes)
    if len(cpus) == 2:
        if 'window_groups' in manifest or 'window_starts' not in manifest:
            raise ValueError('the two-core trial requires window starts')
        return tuple(cpus), tuple(manifest['window_starts'])

    groups = manifest.get('window_groups')
    if 'window_starts' in manifest or type(groups) is not list or not groups:
        raise ValueError('the three-core trial requires nonempty window groups')
    end = 0
    count = len(manifest['recipe']['operations'])
    for group in groups:
        if type(group) is not list or len(group) != 2 or any(type(value) is not int for value in group):
            raise ValueError('each team window requires an exact integer start and width')
        start, width = group
        if start < end or width not in (2, 3) or start + width > count:
            raise ValueError('the trial team windows must be ordered, disjoint and bounded')
        end = start + width
    return tuple(cpus), tuple(tuple(group) for group in groups)


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
            if fields[5].endswith(' (deleted)'):
                raise ValueError(f'live Torch executable mapping was unlinked: {path}')
            major, minor = (int(part, 16) for part in fields[3].split(':'))
            executable.setdefault(path, set()).add((os.makedev(major, minor), int(fields[4])))
    if executable.keys() != expected:
        raise ValueError('live Torch executable mappings differ from the ATen trial provider')
    checked = []
    for path, receipt in libraries.items():
        with path.open('rb') as member:
            status = os.fstat(member.fileno())
            before = file_stat_identity(status)
            if executable[path] != {(status.st_dev, status.st_ino)}:
                raise ValueError(f'live Torch executable mapping differs from its filesystem member: {path}')
            retained = receipt.get('local_file_identity')
            witnessed = (retained is not None and status.st_size == receipt['size_bytes']
                         and retained == local_file_identity(member))
            if not witnessed:
                verify_library_file(member, receipt)
            if (before != file_stat_identity(os.fstat(member.fileno()))
                    or before != file_stat_identity(path.stat())
                    or witnessed and retained != local_file_identity(member)):
                raise ValueError(f'live Torch provider file changed during identification: {path}')
            checked.append(dict(name=path.name, size_bytes=receipt['size_bytes'],
                verification='SAME_BOOT_LOCAL_FILE_IDENTITY' if witnessed else 'SHA256'))
    hashed = sum(row['size_bytes'] for row in checked if row['verification'] == 'SHA256')
    return dict(status='CHECKED', libraries=checked, content_rehashed_bytes=hashed,
                identity_reused_bytes=sum(row['size_bytes'] for row in checked) - hashed)


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


def _safe_torch_context(torch, *, allow_ambient_grad=False):
    c = torch._C
    return ((allow_ambient_grad or not torch.is_grad_enabled()) and not torch.is_inference_mode_enabled()
            and not c._is_any_autocast_enabled() and not c._are_functorch_transforms_active()
            and not c._len_torch_dispatch_stack() and not c._len_torch_function_stack()
            and c._get_torch_function_state().name == 'ENABLED' and not c._peek_should_skip_torch_function()
            and not c._autograd._profiler_enabled() and torch.get_num_threads() == 1
            and torch.get_num_interop_threads() == 1)


def _torch_configuration(torch):
    c = torch._C
    return (torch.is_grad_enabled(), c._dispatch_tls_local_include_set(), c._dispatch_tls_local_exclude_set(),
            torch.get_float32_matmul_precision(), c._get_mkldnn_enabled(),
            torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())


def _geometry_input_strides(guard, recipe, torch):
    from scar.analysis.aten_trial_geometry import SCHEMA, _declared_operators
    from scar.ir.execution_recipe import read_artifact

    if type(guard) is not dict or guard.keys() != {'kind', 'geometry', 'contract'} or guard['kind'] != _GEOMETRY:
        raise ValueError('an exact value-independent geometry guard is required')
    geometry = guard['geometry']
    if (type(geometry) is not dict or type(geometry.get('schema_version')) is not int
            or (geometry.get('schema'), geometry['schema_version'], geometry.get('scope'),
                geometry.get('provider_revision')) != (SCHEMA, 1, _SCOPE, torch.version.git_version)
            or geometry.get('basis') != 'EXACT_INSTALLED_META_OVERLOADS_AND_CPU_STORAGE_CALIBRATION'
            or geometry.get('effect_basis') != 'EXPLICIT_PER_OPERATOR_DECLARATION'):
        raise ValueError('the retained trial geometry differs from the installed provider scope')
    declared = _declared_operators(read_artifact(guard['contract']), torch)
    if declared.keys() != {row['schema'] for row in recipe['operations']}:
        raise ValueError('the trial geometry contract differs from the graph operator set')

    def check(rows, entries, identity):
        if type(rows) is not list or len(rows) != len(entries):
            raise ValueError('the trial geometry differs from the recipe arity')
        fields = {'shape', 'strides', 'storage_bytes'} | ({'id'} if identity == 'id' else set())
        for slot, entry in zip(rows, entries, strict=True):
            if type(slot) is not dict or slot.keys() != fields or slot[identity] != entry[identity]:
                raise ValueError('the trial geometry differs from a recipe slot')
            shape, strides, size = slot['shape'], slot['strides'], slot['storage_bytes']
            if (type(shape) is not list or type(strides) is not list or len(shape) != len(strides)
                    or any(type(i) is not int or not 0 <= i < (1 << 63) for i in (*shape, *strides))
                    or type(size) is not int or not 0 <= size < (1 << 63) or size != 4 * prod(shape)):
                raise ValueError('the trial geometry does not have dense float32 storage')

    check(geometry.get('inputs'), recipe['inputs'], 'shape')
    check(geometry.get('operations'), recipe['operations'], 'id')
    return tuple(tuple(row['strides']) for row in geometry['inputs'])


def _ready_inputs(inputs, shapes, lengths, torch, *, strides=None):
    extents = []
    for index, (tensor, shape, size) in enumerate(zip(inputs, shapes, lengths, strict=True)):
        if (type(tensor) is not torch.Tensor or tensor.device.type != 'cpu' or tensor.dtype is not torch.float32
                or tensor.layout is not torch.strided or tensor.requires_grad or tensor.is_inference()
                or tensor.is_conj() or tensor.is_neg() or tensor.storage_offset() != 0
                or tensor.shape != shape or not tensor.is_contiguous() or tensor.untyped_storage().nbytes() != size
                or (strides is not None and tensor.stride() != strides[index])):
            return False
        if size:
            start = tensor.data_ptr()
            extents.append((start, start + size))
    extents.sort()
    return all(first[1] <= second[0] for first, second in zip(extents, extents[1:]))


@contextmanager
def install_aten_python_trial(module, trial_dir, *, expected_sha256, exclusive_inputs=False,
                              defer_context=False, allow_ambient_grad=False, provider_report=None):
    """Yield trial counters while the exact module binding has a guarded candidate.

    The caller must hold exclusive read access to Tensor contents for the
    duration of each call. Deferred installation permits the application to
    configure Torch before the first admitted call prepares the worker.
    The ambient grad option keeps the caller's GradMode and admits only
    plain Tensor inputs with no gradient requirement. Geometry trials guard
    the shape, stride, and storage bounds retained at build time.
    """
    if exclusive_inputs is not True:
        raise ValueError('the ATen Python trial requires explicit exclusive inputs')
    if type(defer_context) is not bool or type(allow_ambient_grad) is not bool:
        raise ValueError('trial context options must be booleans')
    attachment_start = perf_counter_ns() if provider_report is not None else None
    import torch

    directory = Path(trial_dir).resolve(strict=True)
    manifest = _read_manifest(directory, expected_sha256)
    cpus, windows = _trial_layout(manifest)
    proof = manifest['source_proof']
    namespace, original = _original(module, proof)
    torch_global = proof['torch_global']
    if namespace.get(torch_global) is not torch:
        raise ValueError('the source Torch global does not bind the active provider')
    verified_provider = _provider(manifest['provider'], torch)
    if provider_report is not None:
        provider_report.update(verified_provider)
    wrappers = _builtin_wrappers(torch, proof)
    code = original.__code__
    allowed_cpus = set(cpus)
    gil_enabled = getattr(sys, '_is_gil_enabled', lambda: True)
    if not gil_enabled():
        raise ValueError('the live CPU, Python or Torch context differs from the ATen trial scope')

    def ready_context():
        return (os.sched_getaffinity(0) == allowed_cpus and not _observed(code) and gil_enabled()
                and _safe_torch_context(torch, allow_ambient_grad=allow_ambient_grad))

    if not defer_context and not ready_context():
        raise ValueError('the live CPU, Python or Torch context differs from the ATen trial scope')
    provider_class = torch._C._VariableFunctionsClass
    input_order = tuple(proof['input_order'])
    recipe = manifest['recipe']
    strides = _geometry_input_strides(manifest['input_guard'], recipe, torch) if 'input_guard' in manifest else None
    shapes = tuple(tuple(item['shape']) for item in recipe['inputs'])
    lengths = tuple(4 * prod(shape) for shape in shapes)
    if len(input_order) != len(shapes):
        raise ValueError('the retained input mapping differs from the graph ABI')
    receipts = None if strides is not None else manifest['inputs']
    if receipts is not None and (len(receipts) != len(shapes) or tuple(row['size_bytes'] for row in receipts) != lengths):
        raise ValueError('the retained raw inputs differ from the graph ABI')
    if provider_report is not None:
        provider_report.update(input_guard_kind=_GEOMETRY if strides is not None else _EXACT_BYTES,
                               reference_input_bytes_loaded=0, lazy_prepare_ns=0)
    retained_replica = 'ready_retained_replica' in manifest

    def prepare(configuration):
        lazy_start = perf_counter_ns() if defer_context and provider_report is not None else None
        try:
            references = None if receipts is None else tuple(_verify(Path(row['path']), row, retain=True) for row in receipts)
            if provider_report is not None and references is not None:
                provider_report['reference_input_bytes_loaded'] += sum(map(len, references))
            extension = _extension(directory, manifest['extension'])
            # CPU arguments follow the compiled executor's arity.
            capsule = extension.prepare(*cpus, True) if allow_ambient_grad else extension.prepare(*cpus)
            counts = extension.retained_counts if retained_replica else None
            return capsule, extension.run, references, configuration, counts
        finally:
            if lazy_start is not None:
                provider_report['lazy_prepare_ns'] += perf_counter_ns() - lazy_start

    prepared = None if defer_context else prepare(_torch_configuration(torch))
    arguments = len(proof['parameters'])
    tensor_result = proof['return_kind'] == 'TENSOR'
    owner = get_ident()
    process = os.getpid()
    active = True
    stats = dict(native_calls=0, fallback_calls=0, fallback_reasons={})

    def ready_bindings():
        return (original.__code__ is code and original.__defaults__ is None and original.__kwdefaults__ is None
                and namespace.get(torch_global) is torch and torch._C._VariableFunctionsClass is provider_class
                and all(vars(torch).get(name) is descriptor.__func__ and vars(provider_class).get(name) is descriptor
                        for name, descriptor in wrappers.items()))

    def fallback(reason, args, kwargs):
        stats['fallback_calls'] += 1
        reasons = stats['fallback_reasons']
        reasons[reason] = reasons.get(reason, 0) + 1
        return original(*args, **kwargs)

    @wraps(original)
    def guarded(*args, **kwargs):
        nonlocal prepared
        if not active or kwargs or len(args) != arguments:
            return fallback('CALL_ABI', args, kwargs)
        if os.getpid() != process or get_ident() != owner or not ready_context():
            return fallback('RUNTIME_CONTEXT', args, kwargs)
        configuration = _torch_configuration(torch)
        if prepared is not None and configuration != prepared[3]:
            return fallback('RUNTIME_CONTEXT', args, kwargs)
        if not ready_bindings():
            return fallback('SOURCE_BINDING', args, kwargs)
        inputs = tuple(args[index] for index in input_order)
        if not _ready_inputs(inputs, shapes, lengths, torch, strides=strides):
            return fallback('TENSOR_ABI', args, kwargs)
        if prepared is None:
            try:
                prepared = prepare(configuration)
            except (ImportError, OSError, MemoryError, RuntimeError, ValueError):
                return fallback('PREPARE', args, kwargs)
            # Reference reads may release the GIL while the worker is being prepared.
            if not ready_context() or _torch_configuration(torch) != configuration:
                prepared = None
                return fallback('RUNTIME_CONTEXT', args, kwargs)
            if not ready_bindings():
                prepared = None
                return fallback('SOURCE_BINDING', args, kwargs)
        capsule, native, references, _, _ = prepared
        try:
            result = native(capsule, inputs, windows, references)
        except (MemoryError, RuntimeError, ValueError):
            return fallback('NATIVE_ERROR', args, kwargs)
        if result is None:
            return fallback('INPUT_BYTES' if strides is None else 'NATIVE_ERROR', args, kwargs)
        stats['native_calls'] += 1
        return result[0] if tensor_result else result

    name = proof['function']
    if namespace.get(name) is not original:
        raise ValueError('the source callable changed during trial installation')
    namespace[name] = guarded
    if attachment_start is not None:
        provider_report['attachment_ns'] = perf_counter_ns() - attachment_start
    try:
        yield stats
    finally:
        detach_start = perf_counter_ns() if provider_report is not None else None
        active = False
        if namespace.get(name) is guarded:
            namespace[name] = original
        if retained_replica:
            hits, misses = (0, 0) if prepared is None else prepared[4](prepared[0])
            stats['retained_replica'] = dict(hits=hits, misses=misses)
        prepared = None
        if detach_start is not None:
            provider_report['detachment_ns'] = perf_counter_ns() - detach_start


__all__ = ['install_aten_python_trial']
