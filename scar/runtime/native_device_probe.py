"""Record matched native CPU/CUDA entry attempts for already verified packages."""

from hashlib import sha256
import json
from math import prod
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter_ns

from scar.codegen.native_provider_bundle import _content
from scar.compiler.compile_exported_aten_cuda import CUDA_ENVIRONMENT, verify_cuda_operator_stack
from scar.optimization.native_device_comparison import native_device_probe_reason, native_device_run_reason


SCHEMA = 'scar.native-device-entry-observations'
_MODES = ('CPU', 'CUDA')
_CUDA_IDENTITY = '''
import ctypes as C
import json
import sys

runtime = C.CDLL(sys.argv[1])
device = int(sys.argv[2])

def integer(name):
    value = C.c_int()
    fn = getattr(runtime, name)
    fn.argtypes = (C.POINTER(C.c_int),)
    fn.restype = C.c_int
    result = fn(C.byref(value))
    if result or value.value <= 0:
        raise RuntimeError(f'{name} did not provide a positive value (CUDA status {result})')
    return value.value

count = integer('cudaGetDeviceCount')
if not 0 <= device < count:
    raise RuntimeError('selected CUDA ordinal is outside the effective device domain')
pci = C.create_string_buffer(64)
bus_id = runtime.cudaDeviceGetPCIBusId
bus_id.argtypes = (C.POINTER(C.c_char), C.c_int, C.c_int)
bus_id.restype = C.c_int
result = bus_id(pci, len(pci), device)
if result:
    raise RuntimeError(f'cudaDeviceGetPCIBusId failed with CUDA status {result}')
print(json.dumps(dict(device_count=count, pci_bus_id=pci.value.decode('ascii').lower(),
    runtime_version=integer('cudaRuntimeGetVersion'), driver_version=integer('cudaDriverGetVersion'))))
'''


def _package(root, schema):
    manifest_bytes = (root / 'artifact.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest['schema'] != schema:
        raise ValueError('native device comparison requires the corresponding CPU and CUDA package types')
    identity = {'manifest': dict(sha256=sha256(manifest_bytes).hexdigest(), size_bytes=len(manifest_bytes))}
    for name, item in (('launcher', 'launch.sh'), ('executable', manifest['executable'])):
        path = root / item
        if path.resolve(strict=True) != path or not path.is_relative_to(root):
            raise ValueError('native device executable or launcher is outside its verified package')
        identity[name] = _content(path)
        if identity[name] != manifest['files'][item]:
            raise ValueError('native device executable or launcher differs from its manifest')
    return manifest, identity


def _locked_json(root, manifest, name):
    path = root / name
    if path.resolve(strict=True) != path or not path.is_relative_to(root):
        raise ValueError('native device descriptor is outside its verified package')
    raw = path.read_bytes()
    if dict(sha256=sha256(raw).hexdigest(), size_bytes=len(raw)) != manifest['files'][name]:
        raise ValueError('native device descriptor differs from its manifest')
    return json.loads(raw)


def _receipt_paths(paths):
    records = []
    for path in paths:
        path = Path(path).resolve(strict=True)
        if not path.is_file():
            raise ValueError('native device comparison requires regular input and reference files')
        records.append(dict(path=str(path), **_content(path)))
    return records


def _payloads(records):
    return [{key: row[key] for key in ('sha256', 'size_bytes')} for row in records]


def _cpu_model():
    for line in Path('/proc/cpuinfo').read_text().splitlines():
        name, separator, value = line.partition(':')
        if separator and name.strip() in ('model name', 'Hardware'):
            return value.strip()
    return None


def _stat_locks(paths):
    locks = {}
    for path in paths:
        info = Path(path).stat()
        locks[path] = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    return locks


def _stdio(result):
    return dict(returncode=result.returncode, stdout=result.stdout.decode(errors='replace'),
                stderr=result.stderr.decode(errors='replace'))


def _cuda_runtime_lock(root, manifest):
    locks = _locked_json(root, manifest, 'plan/dependencies.json')
    locks = locks['libraries'] if type(locks) is dict else locks
    runtime = [row for row in locks if re.fullmatch(r'libcudart\.so\.[0-9]+', Path(row['path']).name)
               and manifest['dependencies'].get(Path(row['path']).name) == str(Path(row['path']).resolve(strict=True))]
    return runtime[0] if len(runtime) == 1 else None


def _cuda_device_identity(runtime_library, device, affinity_prefix, environment):
    library = Path(runtime_library['path'])
    try:
        if _content(library) != {key: runtime_library[key] for key in ('sha256', 'size_bytes')}:
            return None, 'CUDA Runtime contents differ from the candidate lock'
        result = subprocess.run([*affinity_prefix, sys.executable, '-I', '-S', '-c', _CUDA_IDENTITY,
                                 str(library), str(device)], env=environment, capture_output=True, timeout=30)
        if result.returncode:
            return None, result.stderr.decode(errors='replace').strip() or f'CUDA identity exit {result.returncode}'
        identity = json.loads(result.stdout)
        if (type(identity) is not dict or identity.keys() != {'device_count', 'pci_bus_id', 'runtime_version', 'driver_version'}
                or type(identity['pci_bus_id']) is not str
                or re.fullmatch(r'[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', identity['pci_bus_id']) is None
                or any(type(identity[key]) is not int or identity[key] <= 0
                       for key in ('device_count', 'runtime_version', 'driver_version'))
                or device >= identity['device_count']):
            return None, 'CUDA runtime returned an incomplete device identity'
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        return None, str(error)
    return dict(identity, runtime_library=runtime_library), None


def _attempt(root, mode, pair, prefix, arguments, output_count, scratch, environment):
    with TemporaryDirectory(prefix='run-', dir=scratch) as temporary:
        outputs = [Path(temporary) / f'{index}.f32' for index in range(output_count)]
        command = [*prefix, str(root / 'launch.sh'), *arguments, *map(str, outputs)]
        start = perf_counter_ns()
        result = subprocess.run(command, env=environment, capture_output=True)
        elapsed = perf_counter_ns() - start
        return dict(pair=pair, mode=mode, wall_ns=elapsed, **_stdio(result),
                    outputs=[_content(path) if path.is_file() and not path.is_symlink() else None for path in outputs])


def _launch_scope(cpu_ids):
    available = os.sched_getaffinity(0)
    if not cpu_ids or len(set(cpu_ids)) != len(cpu_ids) or not set(cpu_ids) <= available:
        raise ValueError('native device comparison requires the CPU incumbent affinity on this host')
    prefix = ('/usr/bin/taskset', '-c', ','.join(map(str, sorted(cpu_ids))))
    environment = {name: os.environ[name] for name in CUDA_ENVIRONMENT if name in os.environ}
    return prefix, environment, sorted(available)


def _observe_cuda_preflight(cuda, manifest, device, cpu_ids, scope):
    stack = verify_cuda_operator_stack(cuda, manifest)
    if stack is not None and stack['linkage']['route'] == 'HYBRID':
        host = _locked_json(cuda, manifest, 'plan/bindings.json')['host_cpu_ids']
        if host != sorted(cpu_ids):
            raise ValueError('the pinned NVIDIA hybrid host cohort differs from the comparison cohort')
    prefix, environment, available = scope
    host = os.uname()
    context = dict(cpu_model=_cpu_model(), architecture=host.machine, kernel_release=host.release,
                   caller_affinity=available, benchmark_cpus=cpu_ids, cuda_environment=environment,
                   device_ordinal=device, cuda_identity=None, cuda_identity_error=None)
    probe = _stdio(subprocess.run([*prefix, str(cuda / 'launch.sh'), '--probe'],
                                  env=environment, capture_output=True))
    rejection = native_device_probe_reason(probe, device=device)
    if rejection is not None:
        return context, probe, 'DEVICE_UNAVAILABLE' if rejection == 'CUDA_DEVICE_UNAVAILABLE' else 'PROBE_FAILED'
    runtime = _cuda_runtime_lock(cuda, manifest)
    identity, error = (_cuda_device_identity(runtime, device, prefix, environment) if runtime
                       else (None, 'native candidate has no unique locked CUDA Runtime library'))
    context.update(cuda_identity=identity, cuda_identity_error=error)
    return context, probe, 'DEVICE_SCOPE_UNVERIFIED' if identity is None else None


def _check_device_unchanged(context, prefix, environment):
    identity = context['cuda_identity']
    if _cuda_device_identity(identity['runtime_library'], context['device_ordinal'], prefix, environment)[0] != identity:
        raise ValueError('CUDA device identity changed during native evidence collection')


def probe_native_device_pair(cpu_root, cuda_root, *, inputs, references, cpu_profile=None,
                             warmup_pairs=1, measurement_pairs=4, expected_identity=None):
    """Collect fresh-process wall time only after the matched CUDA preflight."""
    if (type(warmup_pairs) is not int or warmup_pairs < 0
            or type(measurement_pairs) is not int or measurement_pairs < 3):
        raise ValueError('native device comparison requires at least three measured pairs')
    cpu, cuda = (Path(root).resolve(strict=True) for root in (cpu_root, cuda_root))
    cpu_manifest, cpu_identity = _package(cpu, 'scar.native-exported-aten')
    cuda_manifest, cuda_identity = _package(cuda, 'scar.native-exported-aten-cuda')
    identities = dict(CPU=cpu_identity, CUDA=cuda_identity)
    if expected_identity is not None and identities != expected_identity:
        raise ValueError('native device test packages differ from the admitted launch identities')
    profiles = cpu_manifest.get('profiles')
    if (profiles is None and cpu_profile is not None
            or profiles is not None and cpu_profile not in profiles):
        raise ValueError('native device comparison requires the admitted CPU shape profile')
    plan = 'plan' if cpu_profile is None else f'plan/profiles/{cpu_profile}'
    execution = _locked_json(cpu, cpu_manifest, f'{plan}/execution.json')
    description = _locked_json(cuda, cuda_manifest, 'validation/description.json')
    shapes, cpu_ids = execution['output_shapes'], execution['cpu_ids']
    device, byte_order = description['device'], description['byte_order']
    if byte_order != sys.byteorder:
        raise ValueError('native device package byte order differs from the running host')
    scope = _launch_scope(cpu_ids)
    prefix, environment, _ = scope

    input_records, reference_records = _receipt_paths(inputs), _receipt_paths(references)
    input_shapes = [port['shape'] for port in execution['inputs'] if port['kind'] == 'INPUT']
    for records, sizes in ((input_records, input_shapes), (reference_records, shapes)):
        if [row['size_bytes'] for row in records] != [4 * prod(shape) for shape in sizes]:
            raise ValueError('native device sample file count or byte extent differs from the admitted FP32 ABI')
    references = _payloads(reference_records)
    protected = {row['path'] for row in (*input_records, *reference_records)}
    for root, manifest in ((cpu, cpu_manifest), (cuda, cuda_manifest)):
        protected.update(str(root / name) for name in ('artifact.json', *manifest['files']))
        protected.update(path for path in manifest['dependencies'].values() if path is not None)
        if 'operator_stack' in manifest:
            protected.add(manifest['operator_stack']['reference']['uri'])
    before = _stat_locks(protected)
    context, probe, phase = _observe_cuda_preflight(cuda, cuda_manifest, device, cpu_ids, scope)
    runs = []
    if phase is None:
        phase = 'COMPLETE'
        paths = [row['path'] for row in input_records]
        entries = {'CPU': (cpu, paths if cpu_profile is None else [cpu_profile, *paths]),
                   'CUDA': (cuda, ['--trial', *paths])}
        with TemporaryDirectory(prefix='scar-matched-device-') as scratch:
            for index in range(2 * (warmup_pairs + measurement_pairs)):
                pair, position = divmod(index, 2)
                mode = _MODES[(pair + position) % 2]
                root, arguments = entries[mode]
                row = _attempt(root, mode, pair, prefix, arguments, len(references), scratch, environment)
                runs.append(row)
                reason = native_device_run_reason(row, references=references, output_shapes=shapes,
                                                  byte_order=byte_order)
                if reason is not None:
                    phase = ('OUTPUT_MISMATCH' if reason.endswith('_OUTPUT_MISMATCH') else
                             'INCUMBENT_FAILED' if mode == 'CPU' else 'CANDIDATE_FAILED')
                    break
        if phase == 'COMPLETE':
            _check_device_unchanged(context, prefix, environment)
    if (_stat_locks(protected) != before or _receipt_paths(row['path'] for row in input_records) != input_records
            or _receipt_paths(row['path'] for row in reference_records) != reference_records
            or _package(cpu, 'scar.native-exported-aten')[1] != cpu_identity
            or _package(cuda, 'scar.native-exported-aten-cuda')[1] != cuda_identity):
        raise ValueError('native device packages or sample files changed during the trial')
    return dict(schema=SCHEMA, schema_version=1, phase=phase, probe=probe, context=context,
                inputs=input_records, references=reference_records, package_identity=identities,
                warmup_pairs=warmup_pairs, measurement_pairs=measurement_pairs, runs=runs)
