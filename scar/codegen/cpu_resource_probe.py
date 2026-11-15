"""Independent CPU resource timings, never target-DAG or whole-call timings.

Serial primitives preserve the original caller mask. Parallel stages use a
persistent, temporarily pinned team; worker-local wall clocks include each
stage's own implicit completion barrier. The worker maximum is an effective
component sample, not an observation of the stateful method boundary.
"""
from copy import deepcopy
import ctypes
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from types import SimpleNamespace

_HERE = Path(__file__).resolve().parent
_COMPONENTS = dict(ROW_COPY_STAGE=1, ROW_COPY_SERIAL=2, INPUT_FINITE_SCAN_SERIAL=3,
    ROW_OPERATION_STAGE=4, INPLACE_ROW_OPERATION_STAGE=5, NUMPY_POWER2_NATIVE_ROW_STAGE=6,
    ALLOCATE_FIXED_POOL=7, RELEASE_FIXED_POOL=8, CALLER_WORKER_AFFINITY_FENV_TEAM=9,
    SUCCESS_CHECK_AND_CALLER_RESTORATION=10, OMP_BARRIER_ONLY=11)
_OPS = dict(assign=0, add=1, sub=2, mul=3, div=4, neg=5)


def _digest(raw): return sha256(raw).hexdigest()
def _json(value): return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
def _require(condition, message):
    if not condition: raise ValueError(message)
def _receipt(path):
    raw = path.read_bytes(); return dict(path=str(path), sha256=_digest(raw), bytes=len(raw))


class _Config(ctypes.Structure):
    _fields_ = [('component', ctypes.c_int32), ('operation', ctypes.c_int32),
        ('workers', ctypes.c_int32), ('reserved', ctypes.c_int32),
        ('rows', ctypes.c_int64), ('cols', ctypes.c_int64),
        ('strides', ctypes.c_int64*3), ('offsets', ctypes.c_int64*3),
        ('capacity_count', ctypes.c_uint64), ('capacities', ctypes.POINTER(ctypes.c_uint64))]


def _integer(value, label, minimum=0, maximum=2**63-1):
    _require(type(value) is int and minimum <= value <= maximum, 'invalid '+label)
    return value


def _config(component, params, workers):
    if component not in _COMPONENTS: raise NotImplementedError('independent primitive unavailable: '+component)
    _require(type(params) is dict, 'primitive parameters must be a dictionary')
    code = _COMPONENTS[component]; c = _Config(); c.component = code
    fields = {'rows', 'cols', 'workers'} if code <= 6 else {'workers'}
    if code in (1, 2, 5, 6): fields |= {'source_stride', 'source_offset'}
    if code in (1, 2, 5): fields |= {'destination_stride', 'destination_offset'}
    if code in (4, 5): fields |= {'op'}
    if code == 4: fields |= {'operands'}
    if code in (7, 8): fields = {'capacities'}
    _require(not set(params)-fields, 'unknown primitive applicability parameters')
    c.workers = _integer(params.get('workers', 1), 'worker count', 1, workers)
    if code not in (2, 3, 7, 8):
        _require(c.workers == workers, 'parallel primitive must use the recorded CPU mask')
    if code in (1, 2, 3, 4, 5, 6):
        c.rows = _integer(params['rows'], 'rows', 1)
        c.cols = _integer(params['cols'], 'columns', 1)
        c.strides = (ctypes.c_int64*3)(c.cols, c.cols, c.cols)
        c.offsets = (ctypes.c_int64*3)(0, 0, 0)
    if code in (1, 2, 5, 6):
        c.strides[0] = _integer(params['source_stride'], 'source stride', c.cols)
        c.offsets[0] = _integer(params['source_offset'], 'source offset')
    if code in (1, 2, 5):
        c.strides[2] = _integer(params['destination_stride'], 'destination stride', c.cols)
        c.offsets[2] = _integer(params['destination_offset'], 'destination offset')
    if code in (4, 5):
        op = params['op']
        if op not in _OPS or code == 4 and op == 'assign' or code == 5 and op == 'neg':
            raise NotImplementedError('unsupported primitive operation: '+str(op))
        c.operation = _OPS[op]
        if code == 4:
            operands = params['operands']; _require(type(operands) is list and len(operands) == (1 if op == 'neg' else 2), 'array operands required')
            for i, operand in enumerate(operands):
                c.strides[i] = _integer(operand['stride'], 'operand stride', c.cols)
                c.offsets[i] = _integer(operand['offset'], 'operand offset')
    capacities = params.get('capacities', [])
    if code in (7, 8):
        _require(type(capacities) is list and len(capacities) <= 4096, 'bounded pool capacities required')
        for value in capacities:
            _integer(value, 'slot capacity'); _require(value % 8 == 0, 'float64 slot capacity required')
    owner = (ctypes.c_uint64*len(capacities))(*capacities)
    c.capacity_count = len(capacities); c.capacities = owner
    return c, owner


@dataclass
class ResourceProbe:
    directory: Path
    _metadata: dict
    _library: object
    _run: object
    _provider_owners: object = None

    @property
    def report(self): return deepcopy(self._metadata)

    def sample(self, component, params, repetitions=9, warmup=2):
        context = self._metadata['context']
        _require(sorted(os.sched_getaffinity(0)) == context['caller_affinity_cpu_ids'],
                 'caller affinity differs from recorded calibration context')
        repetitions = _integer(repetitions, 'repetitions', 1, 10000)
        warmup = _integer(warmup, 'warmup', 0, 10000)
        config, owner = _config(component, params, len(context['cpu_ids']))
        provider = None
        if component == 'NUMPY_POWER2_NATIVE_ROW_STAGE':
            if self._provider_owners is None:
                from .stateful import _providers
                row = SimpleNamespace(module='numpy', symbol='power')
                runtime, providers, owners, power = _providers(SimpleNamespace(providers=[row]))
                _require(runtime == context['provider_runtime'], 'original NumPy provider context differs')
                self._provider_owners = (providers, owners, power)
            providers, owners, power = self._provider_owners
            import numpy as np
            _require(np.power is power, 'original NumPy power provider changed')
            provider = ctypes.byref(providers[0])
        values = (ctypes.c_double*repetitions)(); paired = (ctypes.c_double*repetitions)()
        cpus = (ctypes.c_int32*2)(); checksum = ctypes.c_uint64()
        status = self._run(ctypes.byref(config), provider, repetitions, warmup,
            values, paired, cpus, ctypes.byref(checksum))
        _require(status == 0, 'native primitive declined (arguments/FP/CPU/allocation/provider): '+str(status))
        samples = list(values)
        _require(all(math.isfinite(n) and n >= 0 for n in samples), 'invalid primitive clock samples')
        serial = component in ('ROW_COPY_SERIAL', 'INPUT_FINITE_SCAN_SERIAL', 'ALLOCATE_FIXED_POOL', 'RELEASE_FIXED_POOL')
        team = component in ('CALLER_WORKER_AFFINITY_FENV_TEAM', 'SUCCESS_CHECK_AND_CALLER_RESTORATION')
        result = dict(component=component, parameters=deepcopy(params), samples_ns=samples,
            median_ns=statistics.median(samples), unit='ns', basis='DIRECT_COMPONENT_TIMING',
            scope='SERIAL_CALLER_UNBOUND_NATIVE_CLOCK' if serial else
                'PARTITION_OF_EMPTY_TEAM_GUARD_ENVELOPE' if team else
                'MAX_WORKER_LOCAL_WALL_STAGE_WITH_OWN_IMPLICIT_COMPLETION_BARRIER',
            applicable_to_component=True, repetitions=repetitions, warmup=warmup,
            caller_cpu_before=cpus[0], caller_cpu_after=cpus[1],
            caller_affinity_cpu_ids=context['caller_affinity_cpu_ids'],
            parallel_cpu_ids=[] if serial else context['cpu_ids'], checksum=checksum.value,
            whole_method_boundary_measured=False, target_program_executed=False,
            includes_stage_completion_barrier=not serial and not team,
            provider_stage_serial=component == 'NUMPY_POWER2_NATIVE_ROW_STAGE',
            conditions=['FINITE_CANONICAL_OPERANDS_NOT_TARGET_DATA_DISTRIBUTION',
                'WARM_ALLOCATOR_AND_FIRST_TOUCHED_CANONICAL_BUFFERS',
                'COMPONENT_COMPOSITION_IS_A_MODEL_NOT_AN_ADDITIVITY_THEOREM',
                'CLOCK_AND_DISPATCH_OVERHEAD_RETAINED_NOT_SUBTRACTED',
                'GENERIC_PRIMITIVE_DISPATCH_INCLUDED_NO_TARGET_DAG_SPECIALIZATION',
                'NO_OUTER_PYTHON_FFI_DESCRIPTOR_OR_GUARDED_METHOD_COST_INCLUDED'])
        if team:
            result.update(paired_complement_samples_ns=list(paired),
                partition_scope='SETUP_TO_READY_CLOCK_CUT_PLUS_RESTORATION_TEAM_EXIT_CALLER_RESTORE',
                includes_instrumentation_rendezvous=config.workers > 1,
                serial_guard_uses_only_caller_scope=config.workers == 1,
                pairing='SUM_OF_PAIRED_SEGMENTS_IS_ONE_EMPTY_ENVELOPE_NOT_TARGET_TOTAL')
        return result


def build_resource_probe(context, out):
    """Compile one canonical harness for one exact recorded CPU/compiler mask."""
    _require(type(context) is dict, 'resource probe context required')
    context = deepcopy(context); cpus = context['cpu_ids']
    _require(type(cpus) is list and cpus and len(set(cpus)) == len(cpus)
        and all(type(n) is int and 0 <= n < 2**20 for n in cpus), 'unique bounded CPU mask required')
    actual = sorted(os.sched_getaffinity(0))
    _require(context['caller_affinity_cpu_ids'] == actual and set(cpus) <= set(actual),
             'selected CPUs must be available under the exact caller mask')
    compiler = context['compiler']
    _require(('-fopenmp' in compiler['options']) == (len(cpus) > 1), 'recorded OpenMP flags differ from CPU mask')
    identity = subprocess.run([*compiler['argv'], '--version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    _require(identity == compiler['identity'], 'recorded C++ compiler identity differs')
    maximum = _integer(context.get('max_probe_payload_bytes', 256*2**20), 'probe payload limit', 8)
    directory = Path(out).resolve(); _require(not directory.exists(), 'fresh resource probe directory required')
    directory.mkdir(parents=True)
    runtime = (_HERE / 'loop_v1.cpp.in').read_text()
    affinity = runtime[runtime.index('class ThreadAffinity {'):runtime.index('\n#endif', runtime.index('class ThreadAffinity {'))]
    begin = runtime.index('class FloatingScope {')
    floating = runtime[begin:runtime.index('\n};', begin)+3]
    source = (_HERE / 'cpu_resource_probe_v1.cpp.in').read_text()
    for key, value in dict(CPU_SLOTS=str(max(cpus)+1), CPU_IDS=', '.join(map(str, cpus)),
            AFFINITY_CLASS=affinity, FLOATING_CLASS=floating, MAX_PAYLOAD=str(maximum)).items():
        source = source.replace('@'+key+'@', value)
    (directory / 'probe.cpp').write_text(source)
    headers = {}
    for name in ('stateful_abi_v1.h', 'scar_abi_v1.h'):
        (directory / name).write_bytes((_HERE / name).read_bytes()); headers[name] = _receipt(directory / name)
    library = directory / 'cpu_resource_probe.so'
    command = [*compiler['argv'], *compiler['options'], str(directory / 'probe.cpp'), '-o', str(library)]
    result = subprocess.run(command, capture_output=True, text=True)
    (directory / 'build.log').write_text(result.stdout + result.stderr)
    _require(result.returncode == 0, 'canonical CPU resource probe compilation failed: '+result.stderr[-2000:])
    implementation_files = ['cpu_resource_probe.py', 'cpu_resource_probe_v1.cpp.in', 'loop_v1.cpp.in',
        'stateful_abi_v1.h', 'scar_abi_v1.h', 'stateful.py']
    implementation = {name:_digest((_HERE / name).read_bytes()) for name in implementation_files}
    metadata = dict(schema='scar.cpu-resource-probe', schema_version=1, context=context,
        source=_receipt(directory / 'probe.cpp'), headers=headers, library=_receipt(library),
        compiler=compiler, compiler_command=command, implementation_files=implementation,
        implementation_sha256=_digest(_json(implementation)), max_probe_payload_bytes=maximum,
        supported_components=list(_COMPONENTS), measurement_boundary='INDEPENDENT_NATIVE_RESOURCE_PRIMITIVES',
        target_dag_compiled=False, target_program_executed=False, whole_call_cost_measured=False)
    (directory / 'probe.json').write_bytes(_json(metadata))
    handle = ctypes.CDLL(str(library)); run = handle.scar_resource_probe_v1
    run.argtypes = [ctypes.POINTER(_Config), ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint64)]
    run.restype = ctypes.c_int32
    return ResourceProbe(directory, metadata, handle, run)


__all__ = ['ResourceProbe', 'build_resource_probe']
