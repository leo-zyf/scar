"""Observe CPython imports on an already admitted original CPU entry."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
from time import monotonic_ns

from scar.compiler import cpu_entry_replay, cpu_entry_replay_adapters as adapters
from scar.compiler.entry_delivery import _environment_digest
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.runtime.cpu_import_trace import decode


SCHEMA = 'scar.cpu-original-import-observation'
_SAMPLES = 'scar.cpu-original-import-clock-samples'
_TRACE = 'scar.linux-cpu-original-import-strace'
_RAW = 'application/octet-stream'
_SYSCALLS = 'execve,execveat,clone,clone3,fork,vfork,exit,exit_group,write,writev'
_FLAGS = ('-ff', '-ttt', '-xx', '-s', '4096', '--seccomp-bpf', '-e', 'trace=' + _SYSCALLS,
          '-e', 'raw=write,writev', '-e', 'write=2')
_ADAPTERS = {'scar.cpu-array-task-source-provider': adapters.array_tasks,
             'scar.cpu-fixed-grid-source-provider': adapters.fixed_grid,
             'scar.triangle-profile-family': adapters.sparse}


def _need(ok, why):
    if not ok:
        raise ValueError('CPU original imports: ' + why)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _raw(reference):
    return read_artifact(reference, json_document=False)


def _pair(reference, *, full=False):
    paired = read_artifact(reference)
    _need(reference['schema'] == cpu_entry_replay.SCHEMA and type(paired['schema_version']) is int
          and paired['schema_version'] == 1 and paired['deployment_authorized'] is False,
          'expected a sealed CPU paired observation')
    if full:
        _need(cpu_entry_replay.check(reference['uri']) == paired, 'archived paired qualification differs')
    compilation = read_artifact(paired['compilation'])
    original = compilation['entry']
    provider = compilation['provider']
    compiled = read_artifact(compilation['compilation'])
    authority = read_artifact(provider)
    if provider['schema'] == 'scar.cpu-recursive-stream-source-provider':
        spec = cpu_entry_replay._recursive(compilation, compiled, paired['capture'], paired['cpu_override'])
    else:
        _need(provider['schema'] in _ADAPTERS and paired['capture'] is None and paired['cpu_override'] is None,
              'unsupported retained source provider invocation')
        spec = _ADAPTERS[provider['schema']](compilation, compiled, authority)
    spec['entry'] = original
    _need(spec == paired['invocation'] and spec['arms']['original']['argv'][:2]
          == [spec['arms']['original']['executable']['uri'], '-B'], 'pinned original invocation differs from paired replay')
    cpu_entry_replay._pins(spec)
    return spec


def _invocation(spec, folder, tool, profile_mode=2):
    original = spec['arms']['original']
    argv = [arg.replace('{output_dir}', str(folder/'published')) for arg in original['argv']]
    argv[2:2] = ['-X', 'importtime=2' if profile_mode == 2 else 'importtime']
    launcher = ['/usr/bin/taskset', '-c', ','.join(map(str, spec['cpu_ids'])), '/usr/bin/timeout', '-k', '2', '120', *argv]
    diagnostic = [tool['uri'], *_FLAGS, '-o', str(folder/'task'), '--', *launcher]
    return argv, launcher, diagnostic


def _launch(spec, folder, tool, profile_mode=2):
    (folder/'published').mkdir(parents=True, exist_ok=False)
    _, _, diagnostic = _invocation(spec, folder, tool, profile_mode)
    transcript = spec['transcript']
    with (folder/'stdout.bin').open('xb') as stdout, (folder/'stderr.bin').open('xb') as stderr:
        started = monotonic_ns()
        with subprocess.Popen(diagnostic, cwd=spec['cwd'], env=spec['arms']['original']['environment'],
                stdout=subprocess.PIPE if transcript else stdout, stderr=stderr,
                stdin=subprocess.PIPE if transcript else subprocess.DEVNULL) as process:
            if transcript:
                cpu_entry_replay._pace(process, stdout, transcript)
            status = process.wait()
        reaped = monotonic_ns()
    return dict(started_ns=started, reaped_ns=reaped, status=status)


def _observation(spec, stdout, publications):
    _need(publications.keys() == spec['files'].keys(), 'diagnostic changed the declared publication set')
    raw = _raw(stdout)
    if spec['transcript']:
        from scar.optimization.scalar_recursive_parameterized_plan import _stdout

        effects = [(kind, value.encode('ascii') if kind == 'REPLY' else value) for kind, value in spec['ordered_effects']]
        _need(_stdout(raw, effects) == spec['eligible_stdout_digest'], 'diagnostic changed canonical stream output')
    elif recipe := spec.get('json_report'):
        publication, = publications.values()
        _need(adapters._match_json_report(raw, _raw(publication), recipe, _raw(recipe['oracle'])),
              'diagnostic changed normalized original stdout or file publication')
    else:
        _need(raw == _raw(spec['arms']['original']['stdout']) and all(
            _raw(ref) == _raw(spec['files'][name]) for name, ref in publications.items()),
            'diagnostic changed exact original stdout or file publication')


def _tasks(folder, stderr, launcher, interpreter):
    refs = {}
    for path in sorted(folder.glob('task.*')):
        match = re.fullmatch(r'task\.([1-9][0-9]*)', path.name)
        _need(match is not None and path.resolve() == path, 'strace member must be a local canonical TID file')
        refs[int(match[1])] = _ref(path, _TRACE)
    _need(1 <= len(refs) <= 128, 'bounded per-TID trace members required')
    tasks, launcher_tid, interpreter_tid = decode({tid: _raw(ref) for tid, ref in refs.items()},
                                                _raw(stderr), launcher, interpreter)
    return [dict(task, trace=refs[task['tid']]) for task in tasks], launcher_tid, interpreter_tid


def _record(root, reference, spec, tool, count):
    _raw(tool)
    clocks = _ref(root/'clock-samples.json', _SAMPLES)
    timed = read_artifact(clocks)
    mode = timed.get('profile_mode', 2)
    _need(set(timed) - {'profile_mode'} == {'schema', 'rows'} and type(mode) is int and mode in (1, 2)
          and type(timed['rows']) is list and len(timed['rows']) == count,
          'diagnostic sample count differs')
    runs, last = [], 0
    for ordinal, timing in enumerate(timed['rows']):
        folder = root/f'run-{ordinal:02d}'
        interpreter, launcher, diagnostic = _invocation(spec, folder, tool, mode)
        _need(type(timing) is dict and timing.keys() == {'started_ns', 'reaped_ns', 'status'}
              and all(type(value) is int for value in timing.values()) and timing['status'] == 0
              and last <= timing['started_ns'] < timing['reaped_ns'], 'diagnostic exited unsuccessfully or clock differs')
        last = timing['reaped_ns']
        stdout, stderr = (_ref(folder/name, _RAW) for name in ('stdout.bin', 'stderr.bin'))
        publications = cpu_entry_replay._published(folder)
        _observation(spec, stdout, publications)
        tasks, launcher_tid, interpreter_tid = _tasks(folder, stderr, launcher, interpreter)
        runs.append(dict(ordinal=ordinal, **timing, argv=launcher, interpreter_argv=interpreter,
            diagnostic_command=diagnostic, stdout=stdout, stderr=stderr, publications=publications,
            launcher_tid=launcher_tid, interpreter_tid=interpreter_tid, tasks=tasks))
    original = spec['arms']['original']
    return dict(schema=SCHEMA, schema_version=1, paired=reference, entry=spec['entry'], interpreter=original['executable'],
        cpu_ids=spec['cpu_ids'], cwd=spec['cwd'], environment_sha256=_environment_digest(original['environment']),
        boundary=spec.get('scope', 'ARGV_PROCESS'), original_argv=original['argv'],
        instrumentation=dict(kind=f'CPYTHON_IMPORTTIME_{mode}_AND_SCOPED_LINUX_STRACE', tracer=tool, flags=list(_FLAGS),
            parent_clock='LINUX_CLOCK_MONOTONIC_NS', syscall_clock='STRACE_LINUX_REALTIME_US',
            cpython_durations='CPYTHON_IMPORTTIME_REPORTED_MICROSECONDS', nesting_depth='CPYTHON_PRESENTATION_ONLY',
            nonstderr_write_contents_captured=False),
        clock_samples=clocks, runs=runs, instrumented=True, original_workload_executed=True,
        native_target_executed=False, normalized_output_verified=True, clean_benchmark=False,
        complete_import_coverage_proven=False, whole_entry_cost_bound_proven=False, deployment_authorized=False)


def _write(path, value):
    with path.open('x', encoding='utf-8') as output:
        json.dump(value, output, sort_keys=True, separators=(',', ':'), allow_nan=False)
        output.write('\n')


def measure(paired_observation, out, *, runs=1, strace='/usr/bin/strace', profile_mode=2):
    """Run the frozen original invocation with CPython/strace diagnostics only."""
    _need(type(runs) is int and 1 <= runs <= 3, 'one to three diagnostic process invocations required')
    _need(type(profile_mode) is int and profile_mode in (1, 2), 'supported CPython import profiling mode required')
    reference = paired_observation if type(paired_observation) is dict else _ref(paired_observation, cpu_entry_replay.SCHEMA)
    spec = _pair(reference, full=True)
    _need(sys.platform == 'linux' and set(spec['cpu_ids']) <= os.sched_getaffinity(0), 'admitted Linux CPU scope unavailable')
    tool = _ref(strace, 'application/x-executable')
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    timings = [_launch(spec, root/f'run-{ordinal:02d}', tool, profile_mode) for ordinal in range(runs)]
    _write(root/'clock-samples.json', dict(schema=_SAMPLES, rows=timings, profile_mode=profile_mode))
    report = _record(root, reference, spec, tool, runs)
    _write(root/'result.json', report)
    return report


def recover(paired_observation, out, *, strace='/usr/bin/strace'):
    """Publish completed raw samples using the paired scope and tracer used to collect them."""
    reference = paired_observation if type(paired_observation) is dict else _ref(paired_observation, cpu_entry_replay.SCHEMA)
    tool = _ref(strace, 'application/x-executable')
    root = Path(out).resolve(strict=True)
    path = root/'result.json'
    if path.exists():
        report = check(path)
        _need(report['paired'] == reference and report['instrumentation']['tracer'] == tool,
              'completed diagnostic belongs to a different paired scope or tracer')
        return report
    clocks = read_artifact(_ref(root/'clock-samples.json', _SAMPLES))
    count = len(clocks['rows'])
    _need(1 <= count <= 3, 'invalid retained diagnostic run count')
    report = _record(root, reference, _pair(reference), tool, count)
    _write(path, report)
    return report


def check(path):
    """Cold-check frozen launch material, output and trace bytes without launching a target."""
    path = Path(path).resolve(strict=True)
    document = read_artifact(_ref(path, SCHEMA))
    count = len(document['runs'])
    _need(1 <= count <= 3, 'invalid retained diagnostic run count')
    reference = document['paired']
    spec = _pair(reference)
    expected = _record(path.parent, reference, spec, document['instrumentation']['tracer'], count)
    _need(json.dumps(document, sort_keys=True) == json.dumps(expected, sort_keys=True), 'stored diagnostic differs at replay')
    return expected
