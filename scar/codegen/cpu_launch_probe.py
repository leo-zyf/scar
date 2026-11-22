"""Build and measure an empty Linux C++20 process-lifecycle proxy."""

from hashlib import sha256
import json
import os
from pathlib import Path
from shutil import which
import subprocess
import sys

from scar.codegen.native_cpu_elf import _dependencies
from scar.codegen.native_cpu_elf import cpu_executable_dynamic
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.cpu-launch-proxy-observation'
SAMPLES = 'scar.cpu-launch-proxy-samples'
SCOPE = 'EMPTY_LINUX_CXX20_PROCESS_LIFECYCLE_PROXY'
_ABI = {'ARGV': [], 'PACED_STDIN': ['--paced-stdin']}
_FILES = dict(source='probe.cpp', executable='probe', sampler='sampler.py', clock_samples='clock-samples.json')
_SCHEMAS = dict(source='scar.cpp-source', executable='scar.elf', sampler='scar.python-source', clock_samples=SAMPLES)
_PACKAGE = Path(__file__).resolve().parents[1]
_MARKER = 'SCAR_CPU_LAUNCH_CLOCK_V1'
_ACK = 'SCAR_CPU_LAUNCH_ACK_V1'


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU launch proxy: ' + detail)


def _digest(environment):
    return sha256(json.dumps(sorted(environment.items()), separators=(',', ':')).encode()).hexdigest()


def _context(cpu_ids, cwd, environment, abi, warmups, samples):
    _need(type(cpu_ids) is list and cpu_ids and all(type(cpu) is int and cpu >= 0 for cpu in cpu_ids)
          and cpu_ids == sorted(set(cpu_ids)), 'expected a canonical CPU cohort')
    _need(type(cwd) is str and Path(cwd).is_absolute() and Path(cwd).is_dir(), 'expected an existing absolute cwd')
    _need(type(environment) is dict and all(type(key) is type(value) is str and key and '=' not in key
          and '\0' not in key and '\0' not in value for key, value in environment.items()), 'invalid explicit environment')
    _need(not any(environment.get(key) for key in ('LD_PRELOAD', 'LD_AUDIT')),
          'injected startup code is outside the empty proxy scope')
    _need(type(abi) is str and abi in _ABI and type(warmups) is type(samples) is int
          and 0 <= warmups <= 8 and 1 <= samples <= 32, 'unsupported ABI or sample budget')


def _argv(cpu_ids, abi, binary):
    return ['/usr/bin/taskset', '-c', ','.join(map(str, cpu_ids)), '/usr/bin/timeout', '-k', '2', '120',
            binary['uri'], *_ABI[abi]]


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def _receipts(root):
    receipts = {key: _ref(root / name, _SCHEMAS[key]) for key, name in _FILES.items()}
    _need(all(ref['uri'] == str(root / _FILES[key]) for key, ref in receipts.items()), 'probe file redirects outside its path')
    return receipts


def _compile_argv(compiler, source, executable):
    return [*compiler, '-std=c++20', '-O2', str(source), '-o', str(executable)]


def _runtime(binary, cwd, environment):
    dynamic = cpu_executable_dynamic(binary)
    listing = subprocess.check_output(['/usr/bin/ldd', str(binary)], cwd=cwd, env=environment, text=True)
    dependencies = _dependencies(listing)
    return dict(basis='OUT_OF_CLOCK_ELF_TAGS_AND_LOADER_RESOLUTION', dynamic=dynamic,
                resolved={soname: None if path is None else _ref(path, 'scar.elf')
                          for soname, path in sorted(dependencies.items())})


def _measurements(rows, cpu_ids, abi, warmups, count):
    _need(type(rows) is list and len(rows) == warmups + count, 'unexpected number of clock samples')
    spans = dict(startup=[], probe_body=[], teardown=[], proxy_total=[])
    prior = 0
    prefix = '' if abi == 'ARGV' else _ACK + '\n'
    for index, row in enumerate(rows):
        _need(type(row) is dict and row.keys() == {'started_ns', 'reaped_ns', 'output'}, 'invalid clock sample')
        started, reaped, output = row['started_ns'], row['reaped_ns'], row['output']
        _need(type(started) is type(reaped) is int and prior <= started < reaped and type(output) is str,
              'sampler clock order differs')
        clock = output.removeprefix(prefix)
        fields = clock.removesuffix('\n').split(' ')
        _need(output == prefix + ' '.join(fields) + '\n' and fields[:1] == [_MARKER]
              and fields[3:] == list(map(str, cpu_ids)) and all(part.isascii() and part.isdecimal() for part in fields[1:3]),
              'clock marker, acknowledgement or observed CPU affinity differs')
        entered, leaving = map(int, fields[1:3])
        _need(started <= entered <= leaving <= reaped, 'process clock phases do not fit the parent clock')
        prior = reaped
        if index >= warmups:
            for values, value in zip(spans.values(), (entered-started, leaving-entered, reaped-leaving, reaped-started), strict=True):
                values.append(value)
    summary = {}
    for phase, values in spans.items():
        ordered = sorted(values)
        summary[phase] = dict(min_ns=ordered[0], median_ns_twice=ordered[(count-1)//2] + ordered[count//2], max_ns=ordered[-1])
    return summary


def _record(root, *, cpu_ids, cwd, environment, abi, warmups, samples, runtime, compiler, interpreter):
    receipts = _receipts(root)
    argv = _argv(cpu_ids, abi, receipts['executable'])
    observed = read_artifact(receipts['clock_samples'])
    _need(observed.keys() == {'schema', 'schema_version', 'rows'} and type(observed['schema_version']) is int
          and observed['schema_version'] == 1, 'invalid clock sample schema')
    _need(read_artifact(receipts['executable'], json_document=False).startswith(b'\x7fELF'), 'probe is not an ELF')
    _need(runtime.keys() == {'basis', 'dynamic', 'resolved'}
          and runtime['basis'] == 'OUT_OF_CLOCK_ELF_TAGS_AND_LOADER_RESOLUTION'
          and runtime['dynamic'] == cpu_executable_dynamic(receipts['executable']['uri'])
          and set(runtime['dynamic']['needed']) <= set(runtime['resolved']), 'probe ELF dependency census differs')
    for reference in runtime['resolved'].values():
        if reference is not None:
            read_artifact(reference, json_document=False)
    read_artifact(interpreter, json_document=False)
    measurements = _measurements(observed['rows'], cpu_ids, abi, warmups, samples)
    return dict(schema=SCHEMA, schema_version=1, scope=SCOPE, abi=abi, **receipts, cpu_ids=cpu_ids, cwd=cwd,
        environment=environment, environment_sha256=_digest(environment), argv=argv, clock='LINUX_CLOCK_MONOTONIC_NS',
        warmups=warmups, samples=samples, measurements=measurements, runtime=runtime,
        launch_tools={name: _ref(f'/usr/bin/{name}', 'scar.elf') for name in ('taskset', 'timeout')},
        compiler=dict(command=list(compiler), argv=_compile_argv(compiler, receipts['source']['uri'], receipts['executable']['uri'])),
        supervisor=dict(interpreter=interpreter, argv=[interpreter['uri'], '-I', '-S', '-B', receipts['sampler']['uri']],
                        startup_in_proxy_clock=False),
        evidence=dict(fresh_process_per_sample=True, page_cache_state='UNCONTROLLED', target_workload_executed=False,
            clock_marker_publication_in_teardown=True, whole_program_initialization_measured=False,
            empirical_bounds_only=True, whole_entry_cost_bound_proven=False, deployment_authorized=False))


def measure(out, *, cpu_ids, cwd, environment, abi='ARGV', compiler=('c++',), warmups=1, samples=5):
    """Return observed proxy spans. Compiler, sampler startup and validation are off-clock."""
    cpu_ids, cwd = list(cpu_ids), str(cwd)
    _context(cpu_ids, cwd, environment, abi, warmups, samples)
    environment = dict(environment)
    _need(sys.platform == 'linux' and set(cpu_ids) <= os.sched_getaffinity(0), 'requested Linux CPUs are not available')
    _need(type(compiler) in (tuple, list) and bool(compiler) and all(type(value) is str for value in compiler), 'invalid compiler command')
    compiler_path = which(compiler[0])
    _need(compiler_path is not None, 'C++20 compiler is unavailable')
    compiler = [str(Path(compiler_path).absolute()), *compiler[1:]]
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    source, binary, sampler, sample_path = (root / _FILES[key] for key in _FILES)
    source.write_bytes((_PACKAGE/'native/cpu_launch_probe.cpp').read_bytes())
    sampler.write_bytes((_PACKAGE/'runtime/cpu_launch_sampler.py').read_bytes())
    subprocess.run(_compile_argv(compiler, source, binary), check=True)
    runtime = _runtime(binary, cwd, environment)
    interpreter = _ref(sys.executable, 'application/x-executable')
    broker = [interpreter['uri'], '-I', '-S', '-B', str(sampler)]
    request = dict(argv=_argv(cpu_ids, abi, _ref(binary, 'scar.elf')), abi=abi, cwd=cwd,
                   environment=environment, count=warmups + samples)
    sampled = subprocess.check_output(broker, cwd=root, env={}, input=json.dumps(request), text=True)
    _write(sample_path, dict(schema=SAMPLES, schema_version=1, rows=json.loads(sampled)))
    report = _record(root, cpu_ids=cpu_ids, cwd=cwd, environment=environment, abi=abi, warmups=warmups, samples=samples,
                     runtime=runtime, compiler=compiler, interpreter=interpreter)
    _write(root/'result.json', report)
    return report


def check(path):
    """Replay receipt and phase arithmetic without any timed process launch."""
    path = Path(path).resolve(strict=True)
    reference = _ref(path, SCHEMA)
    recorded = read_artifact(reference)
    context = {key: recorded[key] for key in ('cpu_ids', 'cwd', 'environment', 'abi', 'warmups', 'samples')}
    _context(**context)
    expected = _record(path.parent, **context, runtime=recorded['runtime'], compiler=recorded['compiler']['command'],
                       interpreter=recorded['supervisor']['interpreter'])
    _need(json.dumps(recorded, sort_keys=True) == json.dumps(expected, sort_keys=True),
          'stored observation differs from artifact identity or clock replay')
    return recorded
