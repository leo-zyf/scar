"""Replay qualified CPU entry contracts in clean paired processes."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter_ns

from scar.compiler import compile_cpu_entry, cpu_entry_replay_adapters
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.paired_entry_selection import compare_entry_pairs


SCHEMA = 'scar.cpu-entry-paired-replay'
_MODES = ('original', 'candidate')
_RAW = 'application/octet-stream'
_EMPTY = sha256(b'').hexdigest()
_OUTPUT = '{output_dir}'
_TIMEOUT = 120


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU entry replay: ' + detail)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _raw(reference):
    return read_artifact(reference, json_document=False)


def _recursive(record, compiled, capture_ref, cpu):
    from scar.optimization.scalar_recursive_parameterized_plan import _stdout, _tape

    capture = read_artifact(capture_ref)
    runtime = read_artifact(compiled['runtime'])
    plan = read_artifact(runtime['parameterized_plan'])
    _need(capture['source'] == record['entry'] == plan['scope']['source'] and not capture['frontiers']
          and capture['original_execution'] is True and capture['performance_measurement'] is False,
          'stream oracle must come from the same clean source contract')
    actual_capture = read_artifact(capture['raw_capture'])
    _need(all(capture[key] == value for key, value in actual_capture.items()), 'bound original capture differs')
    context = json.loads(_raw(capture['captured_launch']))
    command = context['command']
    observer = command.index('scar.runtime.scalar_recursive_driver_observe')
    _need(command[observer - 3:observer] == ['-B', '-u', '-m'] and context['stdin'] == capture['stdin']['uri'],
          'original interpreter and source command stream must be bound by the capture')
    python = artifact_reference(Path(command[observer - 4]).absolute(), 'application/x-executable')
    tape = _raw(capture['stdin'])
    effects, depths = _tape(tape, plan)
    heads = {row['kind']: row['tokens'] for row in plan['grammar']['commands']}
    markers = {tuple(heads[kind]): last['lines'][-1]
               for kind, last in zip(('UCI', 'ISREADY'), plan['startup']['static_replies'], strict=True)}
    transcript = []
    go = heads['GO_DEPTH']
    for line in tape.decode('ascii').splitlines(keepends=True):
        tokens = line.split()
        marker = 'bestmove ' if tokens[:len(go)] == go else markers.get(tuple(tokens))
        transcript.append(dict(line=line, acknowledgement=marker))
    _need(len(depths) == len(capture['sessions']) and type(cpu) is int and cpu >= 0,
          'one bound CPU and original observations for the complete stream required')
    native = compiled['invocation']['executable']
    common = dict(environment=context['environment'], stdout=capture['stdout'])
    arms = dict(original=dict(argv=[python['uri'], '-B', '-u', record['entry']['uri']], executable=python, **common),
                candidate=dict(argv=[native['uri']], executable=native, **common))
    return dict(cpu_ids=[cpu], cwd=context['cwd'], arms=arms, files={}, transcript=transcript,
                scope='PACED_SOURCE_COMMAND_STREAM', ordered_effects=[
                    [kind, value.decode('ascii') if kind == 'REPLY' else value] for kind, value in effects],
                eligible_stdout_digest=_stdout(_raw(capture['stdout']), effects))


def _spec(reference, capture, cpu):
    doc = compile_cpu_entry.check(reference['uri'])
    _need(doc == read_artifact(reference), 'compilation changed during cold admission')
    compiled, authority = read_artifact(doc['compilation']), read_artifact(doc['provider'])
    schema = doc['provider']['schema']
    if schema == 'scar.cpu-recursive-stream-source-provider':
        _need(capture is not None, 'an original command capture is required for a paced entry')
        spec = _recursive(doc, compiled, capture, cpu)
    else:
        _need(capture is None and cpu is None, 'argv providers use their admitted inputs and CPU mask')
        adapters = {'scar.triangle-profile-family': cpu_entry_replay_adapters.sparse,
                    'scar.cpu-fixed-grid-source-provider': cpu_entry_replay_adapters.fixed_grid}
        spec = adapters[schema](doc, compiled, authority)
    spec['entry'] = doc['entry']
    _need(spec['cpu_ids'] and Path(spec['cwd']).is_absolute() and set(spec['arms']) == set(_MODES),
          'invalid qualified launch context')
    for arm in spec['arms'].values():
        _need(arm['argv'][0] == arm['executable']['uri'], 'launch executable differs from the measured binary')
    _pins(spec)
    return spec


def _pins(spec):
    _raw(spec['entry'])
    for arm in spec['arms'].values():
        _raw(arm['executable'])
        _raw(arm['stdout'])
    for name, reference in spec['files'].items():
        _need(Path(name).name == name and name not in ('', '.', '..'), 'publications need exact declared leaf names')
        _raw(reference)


def _layout(spec, root, warmups, pairs):
    for pair in range(warmups + pairs):
        for mode in _MODES[::(1 if pair % 2 == 0 else -1)]:
            run = root / f'{pair:02d}-{mode}'
            arm = spec['arms'][mode]
            argv = [arg.replace(_OUTPUT, str(run / 'published')) for arg in arm['argv']]
            cpus = ','.join(map(str, spec['cpu_ids']))
            cmd = ['/usr/bin/taskset', '-c', cpus, '/usr/bin/timeout', '-k', '2', str(_TIMEOUT), *argv]
            yield dict(pair=pair, mode=mode, cwd=spec['cwd'], argv=cmd, environment=arm['environment']), run


def _pace(process, output, transcript):
    for phase in transcript:
        process.stdin.write(phase['line'].encode('ascii'))
        process.stdin.flush()
        marker = phase['acknowledgement']
        if marker is not None:
            expected = marker.encode('ascii')
            while line := process.stdout.readline():
                output.write(line)
                if line.startswith(expected):
                    break
            else:
                raise ValueError('CPU entry replay: process exited before ' + marker)
    rest, _ = process.communicate()
    output.write(rest)


def _published(run):
    paths = sorted((run / 'published').iterdir())
    _need(all(item.is_file() and not item.is_symlink() for item in paths), 'unexpected publication type')
    return {item.name: _ref(item, _RAW) for item in paths}


def _run(spec, execution, run):
    (run / 'published').mkdir(parents=True)
    stdout, stderr = run / 'stdout.bin', run / 'stderr.bin'
    paced = bool(spec['transcript'])
    with stdout.open('xb') as out, stderr.open('xb') as err:
        started = perf_counter_ns()
        with subprocess.Popen(execution['argv'], cwd=execution['cwd'], env=execution['environment'],
                              stdin=subprocess.PIPE if paced else subprocess.DEVNULL,
                              stdout=subprocess.PIPE if paced else out, stderr=err) as process:
            if paced:
                _pace(process, out, spec['transcript'])
            status = process.wait()
        elapsed = perf_counter_ns() - started
    return dict(execution=execution, elapsed_ns=elapsed, status=status,
                stdout=_ref(stdout, _RAW), stderr=_ref(stderr, _RAW), publications=_published(run))


def _results(spec, samples, layout, warmups, pairs):
    observations = []
    # Compare the common result after each arm's own stdout format passes.
    semantic_identity = sha256(json.dumps(dict(outputs=spec['files'],
        stdout={name: arm['stdout'] for name, arm in spec['arms'].items()}), sort_keys=True).encode()).hexdigest()
    if spec['transcript']:
        from scar.optimization.scalar_recursive_parameterized_plan import _stdout

        effects = [(kind, value.encode('ascii') if kind == 'REPLY' else value)
                   for kind, value in spec['ordered_effects']]
    for row, (execution, run) in zip(samples, layout, strict=True):
        mode = execution['mode']
        _need(row['execution'] == execution and row['status'] == 0 and row['publications'] == _published(run)
              and row['stdout'] == _ref(run / 'stdout.bin', _RAW) and row['stderr'] == _ref(run / 'stderr.bin', _RAW)
              and not _raw(row['stderr']), f'{mode} run failed or produced unexpected stderr: {run}')
        stdout = _raw(row['stdout'])
        if spec['transcript']:
            _need(_stdout(stdout, effects) == spec['eligible_stdout_digest'], 'source command effects or canonical output differ')
        else:
            _need(stdout == _raw(spec['arms'][mode]['stdout']), f'{mode} changed the admitted stdout bytes')
        _need(row['publications'].keys() == spec['files'].keys() and all(
            _raw(artifact) == _raw(spec['files'][name]) for name, artifact in row['publications'].items()),
            f'{mode} changed the exact publications')
        observations.append(dict(pair=execution['pair'], mode=mode, wall_ns=row['elapsed_ns'],
                                 returncode=0, stdout_sha256=semantic_identity, stderr_sha256=_EMPTY))
    return compare_entry_pairs(observations, incumbent='original', challenger='candidate',
                               warmup_pairs=warmups, measurement_pairs=pairs)


def measure(compilation, out, *, capture=None, cpu=None, warmups=1, pairs=3):
    from scar.analysis.scalar_recursive_driver import CAPTURE

    _need(type(warmups) is int and warmups >= 0 and type(pairs) is int and pairs >= 3,
          'at least three measured pairs and a nonnegative warmup count are required')
    reference = _ref(compilation, compile_cpu_entry.SCHEMA)
    captured = _ref(capture, CAPTURE) if capture else None
    spec = _spec(reference, captured, cpu)
    _need(set(spec['cpu_ids']) <= os.sched_getaffinity(0), 'qualified CPU mask is unavailable')
    root = Path(out).resolve()
    root.mkdir()
    layout = list(_layout(spec, root, warmups, pairs))
    rows = [_run(spec, execution, run) for execution, run in layout]
    _pins(spec)
    read_artifact(reference)
    report = dict(schema=SCHEMA, schema_version=1, compilation=reference, capture=captured, cpu_override=cpu,
                  output_root=str(root), invocation=spec, warmup_pairs=warmups, measurement_pairs=pairs,
                  runs=rows, selection=_results(spec, rows, layout, warmups, pairs),
                  whole_program_equivalence_proven=False, deployment_authorized=False)
    with (root / 'result.json').open('x') as output:
        output.write(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return report


def check(path):
    root = Path(path).resolve(strict=True).parent
    report = read_artifact(_ref(path, SCHEMA))
    spec = _spec(report['compilation'], report['capture'], report['cpu_override'])
    warmups, pairs = report['warmup_pairs'], report['measurement_pairs']
    _need(report['schema_version'] == 1 and report['output_root'] == str(root) and spec == report['invocation']
          and report['whole_program_equivalence_proven'] is False and report['deployment_authorized'] is False,
          'CPU entry invocation differs at replay')
    layout = _layout(spec, root, warmups, pairs)
    _need(report['selection'] == _results(spec, report['runs'], layout, warmups, pairs),
          'paired whole-entry decision differs at replay')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--compilation', type=Path)
    inputs.add_argument('--check', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--original-capture', type=Path)
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--pairs', type=int, default=3)
    args = parser.parse_args(argv)
    if args.check:
        report = check(args.check)
    elif args.out:
        report = measure(args.compilation, args.out, capture=args.original_capture, cpu=args.cpu,
                         warmups=args.warmups, pairs=args.pairs)
    else:
        parser.error('--compilation requires --out')
    print(json.dumps(report['selection'], sort_keys=True))


if __name__ == '__main__':
    main()
