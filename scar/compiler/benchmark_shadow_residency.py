"""Clean paired raw-ABI workspace/resident-shadow repeat measurements.

Only two already-built implementations of the same four-step native repeat
run. Input reset, packing, source admission, loading and checks are outside
the clocks. This is neither a whole-episode benchmark nor deployment evidence.
"""
import argparse
import ctypes
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

SCHEMA = 'scar.shadow-residency-benchmark'
INPUT_SCHEMA = 'scar.workspace-benchmark-inputs'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _reference(path, schema):
    path = Path(path).resolve(); raw = path.read_bytes()
    return dict(uri=str(path), sha256=sha256(raw).hexdigest(), size_bytes=len(raw), schema=schema)


def _read(reference, *, raw=False):
    _need(_reference(reference['uri'], reference['schema']) == reference, 'retained benchmark input changed')
    value = Path(reference['uri']).read_bytes()
    return value if raw else json.loads(value)


def _write(path, document):
    with Path(path).open('x') as stream:
        json.dump(document, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    return _reference(path, document['schema'])


class _Environment:
    """Current Linux/x86-64 glibc fenv ABI, not an arbitrary opaque buffer."""
    class Buffer(ctypes.Union):
        _fields_ = [('words', ctypes.c_uint32 * 8), ('alignment', ctypes.c_longdouble)]

    def __init__(self):
        _need(sys.platform == 'linux' and platform.machine() == 'x86_64'
              and platform.libc_ver()[0] == 'glibc' and ctypes.sizeof(self.Buffer) == 32,
              'benchmark fenv snapshot requires the current Linux/x86-64 glibc ABI')
        header = Path('/usr/include/x86_64-linux-gnu/bits/fenv.h')
        self.header = _reference(header, 'scar.system-c-header')
        text = header.read_text()
        _need('unsigned int __mxcsr;' in text and 'fenv_t;' in text,
              'current x86-64 glibc fenv header required')
        self.libc = ctypes.CDLL(None)
        for name in ('fegetenv', 'fesetenv'):
            function = getattr(self.libc, name)
            function.argtypes = [ctypes.c_void_p]; function.restype = ctypes.c_int
        self.libc.fegetround.argtypes = []; self.libc.fegetround.restype = ctypes.c_int
        self.libc.fetestexcept.argtypes = [ctypes.c_int]; self.libc.fetestexcept.restype = ctypes.c_int

    def capture(self):
        value = self.Buffer()
        _need(self.libc.fegetenv(ctypes.byref(value)) == 0, 'cannot capture original full floating environment')
        return value

    def restore(self, value):
        _need(self.libc.fesetenv(ctypes.byref(value)) == 0, 'cannot restore original full floating environment')

    def summary(self, value):
        return dict(sha256=sha256(bytes(value)).hexdigest(),
            rounding=self.libc.fegetround(), exception_flags=self.libc.fetestexcept(0x3d))


def _statistics(values):
    return dict(samples_ns=values, median_ns=statistics.median(values), min_ns=min(values),
                max_ns=max(values), stdev_ns=statistics.stdev(values), variance_ns2=statistics.variance(values))


def _qualify_variant(directory):
    # The compiler reader must replay source/plan/native receipts. A bare
    # metadata JSON or a caller-supplied qualified boolean is not sufficient.
    from scar.compiler.compile_workspace_shadow import replay_compiled_workspace_shadow
    return replay_compiled_workspace_shadow(directory)


def _worker(spec_path):
    import numpy as np
    from scar.codegen import stateful as backend
    from scar.codegen import cpp
    from scar.codegen.stateful_workspace import load_workspace_repeat

    spec = json.loads(Path(spec_path).read_text()); output = Path(spec['out'])
    _need(sys.gettrace() is None and sys.getprofile() is None, 'clean repeat timing forbids Python trace/profile hooks')
    _need((spec['warmup'], spec['repetitions'], spec['cpu_id']) == (2, 7, 0), 'frozen two-warm/seven-pair CPU0 policy required')
    publication, metadata, context = _qualify_variant(Path(spec['variant']))
    _need(metadata.get('schema') == 'scar.native-workspace-shadow-residency', 'source-qualified resident native artifact required')
    original_directory = Path(spec['original']).resolve()
    original_reference = _reference(original_directory/'artifact.json', 'scar.native-stateful-workspace-repeat')
    # The variant reader binds its exact original workspace and mathematical
    # parent. It exposes this reference explicitly for the benchmark controller.
    _need(context['original_native_reference'] == original_reference,
          'resident implementation and requested original workspace differ')
    original = load_workspace_repeat(original_directory)  # exactly one provider/table construction
    _need(original.metadata == context['native_metadata'] and original.metadata['trip_count'] == 4,
          'same source-qualified original mathematical ABI and four-step repeat required')
    inputs_reference = spec['inputs']; inputs = _read(inputs_reference)
    _need((inputs.get('schema'), inputs.get('schema_version')) == (INPUT_SCHEMA, 1), 'versioned fixed benchmark inputs required')
    array_ids = [row.id for row in original.base.region.inputs if row.kind == 'array']
    scalar_ids = [row.id for row in original.base.region.inputs if row.kind == 'scalar']
    _need(type(inputs.get('arrays')) is dict and set(inputs['arrays']) == set(array_ids)
          and type(inputs.get('scalars')) is dict and set(inputs['scalars']) == set(scalar_ids),
          'benchmark input IDs must match the exact original source region')
    initial = []; arrays = []; array_refs = []
    for name in array_ids:
        record = inputs['arrays'][name]; reference = record['artifact']
        _need(reference['schema'] == 'scar.numpy-array', 'original retained NumPy array artifact required')
        _read(reference, raw=True)
        value = np.load(reference['uri'], allow_pickle=False)
        _need(type(value) is np.ndarray and value.dtype == np.dtype('float64')
              and value.flags.c_contiguous and list(value.shape) == record['shape']
              == original.base.plan['input_shapes'][name] and record['dtype'] == 'float64'
              and bool(np.isfinite(value).all()), 'retained float64 array shape/layout/initial finite contract differs')
        initial.append(value); arrays.append(value.copy(order='C')); array_refs.append(reference)
    values = []
    for name in scalar_ids:
        value = inputs['scalars'][name]
        _need(type(value) in (int, float) and math.isfinite(value)
              and (type(value) is not int or abs(value) <= 2**53), 'exact finite retained scalar required')
        values.append(backend._Scalar(float(value), int(type(value) is int)))
    descriptors = [cpp._descriptor(value) for value in arrays]
    for i, left in enumerate(descriptors):
        _need(all(not (left.lower_bound < right.upper_bound and right.lower_bound < left.upper_bound)
                  for right in descriptors[:i]), 'benchmark roots must have disjoint storage')
    packed = (cpp._Array * len(arrays))(*descriptors)
    scalars = (backend._Scalar * len(values))(*values)
    providers = original.base._providers; table = original.base._table
    raw_metadata = dict(descriptors=bytes(packed), scalars=bytes(scalars), providers=bytes(providers),
        abi_table=ctypes.string_at(table, ctypes.sizeof(backend._Table)))
    variant_reference = publication['native_artifact']
    _need(_read(variant_reference) == metadata, 'qualified variant native metadata changed')
    variant_directory = Path(variant_reference['uri']).parent
    publication_path = Path(spec['variant']).resolve()
    if publication_path.is_dir():
        publication_path /= 'result.json'
    publication_reference = _reference(publication_path, publication['schema'])
    _need(_read(publication_reference) == publication, 'qualified variant publication changed')
    library_reference = _reference(variant_directory/metadata['files']['library'], 'scar.native-library')
    source_reference = _reference(variant_directory/metadata['files']['source'], 'scar.cpp-source')
    _need(library_reference['sha256'] == metadata['sha256']['library']
          and source_reference['sha256'] == metadata['sha256']['source'], 'qualified resident source/library bytes changed')
    handle = ctypes.CDLL(library_reference['uri'])
    getter = handle.scar_workspace_plan_sha256_v1; getter.argtypes = []; getter.restype = ctypes.c_char_p
    _need(getter() == metadata['execution_plan_sha256'].encode(), 'resident embedded execution plan differs')
    variant = handle.scar_workspace_repeat_stateful_v1
    variant.argtypes = list(original._repeat.argtypes); variant.restype = ctypes.c_int32
    progress = ctypes.c_uint64()
    arguments = (table, packed, len(arrays), scalars, len(values), providers,
                 len(original.base.region.providers), 4, ctypes.byref(progress))
    functions = dict(ORIGINAL=original._repeat, RESIDENT=variant)
    environment = _Environment(); saved_environment = environment.capture()
    saved_affinity = sorted(os.sched_getaffinity(0)); locked = False; samples = []
    _need(0 in saved_affinity, 'declared benchmark CPU0 is outside the original caller affinity')
    try:
        os.sched_setaffinity(0, {0})  # pid zero affects this calling thread only
        entry_environment = environment.capture()
        entry_context = environment.summary(entry_environment)
        _need(original.base._lock.acquire(blocking=False), 'original provider capsule is already in use')
        locked = True
        design = dict(schema='scar.shadow-residency-benchmark-design', schema_version=1,
            original=original_reference, variant_publication=publication_reference,
            variant_native=variant_reference,
            variant_source=source_reference, variant_library=library_reference,
            inputs=inputs_reference, array_inputs=array_refs,
            policy=dict(trip_count=4, warmup=2, repetitions=7, cpu_ids=[0], order='ALTERNATING_AB_BA'),
            fenv_abi_header=environment.header, fenv_abi_bytes=32,
            original_caller_affinity=saved_affinity, measured_caller_affinity=[0], entry_environment=entry_context,
            environment=dict(cpu_model=next((line.split(':',1)[1].strip()
                for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), 'UNKNOWN'),
                platform=platform.platform(), python=sys.version, numpy=np.__version__,
                compiler=original.metadata['compiler'], target_snapshot=original.base.plan['resource_context']['target_snapshot'],
                host_isolated=False, profiling_enabled=False),
            clock='time.perf_counter_ns',
            timing_scope='CTYPES_RAW_NINE_ARGUMENT_NATIVE_REPEAT_CALL_AND_RETURN_ONLY_N4',
            packing_loading_reset_validation_and_report_io_in_clock=False,
            conditional_assumptions='ORIGINAL_SYNC_CPU_BORROWED_MEMORY_QUIESCENCE_AND_NO_INTERSTEP_OBSERVERS_Q',
            provider_quiescence_source_proven=False, provider_quiescence_runtime_checkable=False,
            whole_episode_measured=False, deployment_authorized=False,
            implementation=_reference(Path(__file__), 'scar.python-source'))
        design_reference = _write(output/'design.json', design)
        written = set(original.base.plan['shadows'])
        for pair in range(9):
            phase = 'WARMUP' if pair < 2 else 'MEASURED'
            pair_index = pair if pair < 2 else pair - 2
            order = ['ORIGINAL', 'RESIDENT'] if pair % 2 == 0 else ['RESIDENT', 'ORIGINAL']
            calls = []; states = {}
            for name in order:
                for current, baseline in zip(arrays, initial):
                    np.copyto(current, baseline, casting='no')
                environment.restore(entry_environment); progress.value = 0
                before = environment.capture(); before_context = environment.summary(before)
                _need(sorted(os.sched_getaffinity(0)) == [0], 'benchmark calling thread affinity changed')
                start = time.perf_counter_ns()
                status = int(functions[name](*arguments))
                end = time.perf_counter_ns()
                after = environment.capture(); after_context = environment.summary(after)
                restored = bytes(after) == bytes(before)
                affinity = sorted(os.sched_getaffinity(0)) == [0]
                metadata_unchanged = raw_metadata == dict(
                    descriptors=bytes(packed), scalars=bytes(scalars), providers=bytes(providers),
                    abi_table=ctypes.string_at(table, ctypes.sizeof(backend._Table)))
                readonly = all(array_ids[i] in written or arrays[i].tobytes() == initial[i].tobytes()
                               for i in range(len(arrays)))
                states[name] = [array.tobytes(order='C') for array in arrays]
                call = dict(variant=name, duration_ns=end - start, status=status, completed=int(progress.value),
                    full_fenv_restored=restored, before_environment=before_context, after_environment=after_context,
                    caller_affinity_restored=affinity, descriptor_scalar_provider_tables_unchanged=metadata_unchanged,
                    readonly_inputs_unchanged=readonly,
                    state_sha256={key:sha256(value).hexdigest() for key,value in zip(array_ids,states[name])})
                calls.append(call)
                _need(status == 0 and progress.value == 4 and restored and affinity and metadata_unchanged and readonly,
                      'raw repeat output/progress/guard context validation failed')
            equal = states['ORIGINAL'] == states['RESIDENT']
            row = dict(phase=phase, index=pair_index, order=order, calls=calls, output_bitwise_equal=equal)
            samples.append(row)
            _write(output/'samples'/('%s-%02d.json' % (phase.lower(), pair_index)),
                dict(schema='scar.shadow-residency-benchmark-pair', schema_version=1, **row))
            _need(equal, 'original and resident native repeat states differ bitwise')
        measured = {name:[call['duration_ns'] for row in samples if row['phase']=='MEASURED'
            for call in row['calls'] if call['variant']==name] for name in functions}
        distributions = {name:_statistics(values) for name,values in measured.items()}
        report = dict(schema=SCHEMA, schema_version=1, design=design_reference, samples=samples,
            statistics=distributions,
            scoped_repeat_speedup=distributions['ORIGINAL']['median_ns']/distributions['RESIDENT']['median_ns'],
            whole_episode_speedup=None, mathematical_state_validation='BITWISE_ALL_RETAINED_ARRAY_INPUTS_AFTER_EACH_RESET_REPEAT',
            whole_program_validated=False, whole_episode_measured=False,
            selection_performed=False, deployment_authorized=False, default_runtime_decision='NO_OP')
    finally:
        if locked:
            original.base._lock.release()
        try:
            os.sched_setaffinity(0, saved_affinity)
        finally:
            environment.restore(saved_environment)
    report['original_caller_affinity_restored'] = sorted(os.sched_getaffinity(0)) == saved_affinity
    report['original_full_fenv_restored'] = bytes(environment.capture()) == bytes(saved_environment)
    _need(report['original_caller_affinity_restored'] and report['original_full_fenv_restored'],
          'final original caller environment restoration failed')
    for reference in (inputs_reference, original_reference, publication_reference, variant_reference,
                      source_reference, library_reference, *array_refs):
        _read(reference, raw=True)
    return _write(output/'result.json', report)


def benchmark_shadow_residency(original_native, variant_native, input_spec, out, *, warmup=2, repetitions=7, cpu_id=0):
    """Launch one clean process; run only the frozen original/resident pair."""
    _need((warmup, repetitions, cpu_id) == (2, 7, 0), 'frozen benchmark policy is warmup2/repetitions7/CPU0')
    output = Path(out).resolve(); _need(not output.exists(), 'fresh shadow repeat benchmark output required')
    output.mkdir(parents=True); (output/'samples').mkdir()
    spec = dict(schema='scar.shadow-residency-benchmark-spec', schema_version=1,
        original=str(Path(original_native).resolve()), variant=str(Path(variant_native).resolve()),
        inputs=_reference(input_spec, INPUT_SCHEMA), out=str(output),
        warmup=warmup, repetitions=repetitions, cpu_id=cpu_id)
    _write(output/'spec.json', spec)
    process = subprocess.run([sys.executable, '-m', 'scar.compiler.benchmark_shadow_residency',
        '--worker', str(output/'spec.json')], capture_output=True, text=True)
    (output/'worker.stdout').write_text(process.stdout); (output/'worker.stderr').write_text(process.stderr)
    _need(process.returncode == 0, 'clean native repeat benchmark failed: ' + process.stderr[-2000:])
    return json.loads((output/'result.json').read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original'); parser.add_argument('--variant'); parser.add_argument('--inputs')
    parser.add_argument('--out'); parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        _worker(args.worker)
    else:
        if not all((args.original,args.variant,args.inputs,args.out)):
            parser.error('--original --variant --inputs --out are required')
        report = benchmark_shadow_residency(args.original,args.variant,args.inputs,args.out)
        print(json.dumps(dict(result=str(Path(args.out).resolve()/'result.json'),
            scoped_repeat_speedup=report['scoped_repeat_speedup'], whole_episode_speedup=None,
            deployment_authorized=False), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
