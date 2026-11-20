"""Measure diagnostic plot libraries inside matched native controllers."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess

from scar.analysis import native_plot_phases
from scar.codegen.native_controller_child import _literal
from scar.codegen.native_controller_plot import ASSETS, HEADER
from scar.codegen.native_live_plot import _replace
from scar.codegen.native_project import build_native_project
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.native-plot-entry-phases'
BENCHMARK = 'scar.native-process-boundary-benchmark'
CONFIG = 'scar.native-process-boundary-cases'
ENTRY = 'scar.native-stateful-entry'
PROBE = 'scar.native-live-plot-phase'
_SIDES = {'v1': 'original', 'v2': 'candidate'}
_TARGET = 'scar_stateful_entry'


def _need(ok, why):
    if not ok:
        raise ValueError('native plot entry: ' + why)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _benchmark_ref(row, schema):
    return dict(uri=row['path'], sha256=row['sha256'], size_bytes=row['size_bytes'], schema=schema)


def _member(root, row):
    name = Path(row['uri'])
    _need(not name.is_absolute() and '..' not in name.parts
          and (root / name).resolve(strict=True) == root / name,
          'expected an exact local package member')
    return read_artifact({**row, 'uri': str(root / name)}, json_document=False)


def _inputs(probe_ref, bench_ref, case):
    benchmark, probe = read_artifact(bench_ref), read_artifact(probe_ref)
    _need(benchmark['qualified'] and not benchmark['instrumented']
          and not benchmark['baseline_only'] and benchmark['original_and_candidate_share_environment_and_affinity'],
          'an exact clean paired execution boundary must precede diagnosis')
    config = read_artifact(_benchmark_ref(benchmark['source_case_file'], CONFIG))
    row, = (row for row in config['cases'] if row['id'] == case)
    measured, = (row for row in benchmark['evidence'] if row['id'] == case)
    _need(measured['outputs_equal_for_required_comparisons'] and
          benchmark['cpu_ids'] == config['cpu_ids'] and len(config['cpu_ids']) == 1
          and benchmark['environment'] == config['environment']
          and row['original']['argv'][1:] == row['candidate']['argv'][1:]
          and row['original']['outputs'] == row['candidate']['outputs'],
          'proof must compare both variants under one CPU, argv and output contract')
    _need(probe['schema_version'] == 1 and tuple(probe['phases']) == native_plot_phases.PHASES
          and set(probe['arms']) == set(_SIDES), 'incompatible plot probe protocol')
    arms = {}
    for arm, side in _SIDES.items():
        argv = row[side]['argv']
        exe = Path(argv[0]).resolve(strict=True)
        pinned = benchmark['pinned_inputs'][f'{case}:{side}_executable']
        _need(artifact_reference(exe, 'scar.native-executable') == _benchmark_ref(pinned, 'scar.native-executable'),
              'implicated executable bytes differ')
        app = exe.parent.parent
        reference = _ref(app / 'result.json', ENTRY)
        _need(reference == _benchmark_ref(benchmark['pinned_inputs'][f'{arm}_package_manifest'], ENTRY),
              'implicated package manifest differs')
        original = read_artifact(reference)
        diagnostic = probe['arms'][arm]
        provider = original['controller']['plot']['provider']
        _need(provider == diagnostic['original_provider']
              and original['files']['bin/' + _TARGET]['sha256'] == pinned['sha256'],
              'probe, measured ELF and qualified native provider do not share one original input')
        library = str(ASSETS / read_artifact(provider)['library'])
        arms[arm] = dict(application=reference, library=original['files'][library],
                         diagnostic_library=diagnostic['diagnostic_library'],
                         argv=argv, outputs=row[side]['outputs'])
    return arms, config, measured['samples']['original'][0]


def _diagnostic_members(arm, probe_root):
    app = Path(arm['application']['uri']).parent
    manifest = read_artifact(arm['application'])
    before, after = arm['library'], arm['diagnostic_library']
    member = before['uri']
    old_asset = f'{_literal(member)}, {before["size_bytes"]}, "{before["sha256"]}"'
    new_asset = f'{_literal(member)}, {after["size_bytes"]}, "{after["sha256"]}"'
    for name, row in manifest['files'].items():
        _need(row['uri'] == name, 'original application member name differs')
        payload = _member(app, row)
        if name == 'bin/' + _TARGET:
            continue
        if name == member:
            payload = _member(probe_root, after)
        elif name == HEADER:
            payload = _replace(payload.decode(), old_asset, new_asset).encode()
        yield name, payload


def _stage(arm, probe_root, destination):
    for name, payload in _diagnostic_members(arm, probe_root):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    compiler = shutil.which('g++')
    _need(compiler is not None, 'native diagnostic needs the original compiler family')
    return build_native_project(destination, _TARGET, (compiler,))[0]


def _observe(arm, exe, out, config):
    out.mkdir()
    target, trace = out / 'published', out / 'plot.jsonl'
    lexical = arm['argv'][1:]
    _need(sum(arg.count('{output_dir}') for arg in lexical) == 1, 'one original owned output directory required')
    command = ['/usr/bin/taskset', '-c', str(config['cpu_ids'][0]), str(exe),
               *(arg.replace('{output_dir}', str(target)) for arg in lexical)]
    env = {**config['environment'], 'SCAR_PLOT_PHASE_LOG': str(trace)}
    measured = subprocess.run(command, env=env, cwd=config['working_directory'],
                              capture_output=True, timeout=config['timeout_seconds'])
    stdout, stderr = out / 'stdout.bin', out / 'stderr.bin'
    stdout.write_bytes(measured.stdout)
    stderr.write_bytes(measured.stderr)
    _need(measured.returncode == 0 and target.is_dir() and trace.is_file(),
          'native diagnostic did not complete; see ' + str(out))
    produced = {path.relative_to(target).as_posix() for path in target.rglob('*') if path.is_file()}
    _need(produced == set(arm['outputs'].values()), 'observable output file names differ')
    files = {label: _ref(target / name, 'scar.observed-native-publication') for label, name in arm['outputs'].items()}
    return dict(trace=_ref(trace, native_plot_phases.TRACE),
                stdout=_ref(stdout, 'scar.observed-native-stdout'), stderr=_ref(stderr, 'scar.observed-native-stderr'),
                outputs=files)


def _summary(runs, frames, oracle):
    pairs = []
    for pair in runs:
        for sample in pair.values():
            _need(sample['outputs'].keys() == oracle['outputs'].keys(), 'diagnostic publications differ')
            expected = [(sample['stdout'], oracle['stdout'])]
            expected.extend((row, oracle['outputs'][name]) for name, row in sample['outputs'].items())
            for observed, original in expected:
                _need(read_artifact(observed, json_document=False)
                      == read_artifact(_benchmark_ref(original, observed['schema']), json_document=False),
                      'diagnostic bytes differ from the clean application')
            read_artifact(sample['stderr'], json_document=False)
        pairs.append({side: pair[side]['trace'] for side in _SIDES})
    return native_plot_phases.compare(pairs, frames)


def capture(probe, benchmark, case, out, *, repetitions=3):
    _need(type(repetitions) is int and 2 <= repetitions <= 16, 'bounded paired repetitions required')
    probe_ref, bench_ref = _ref(probe, PROBE), _ref(benchmark, BENCHMARK)
    arms, config, oracle = _inputs(probe_ref, bench_ref, case)
    frames = sum(name.startswith('frame:') for name in oracle['outputs'])
    _need(frames > 0, 'an original frame oracle is required')
    out = Path(out).resolve()
    out.mkdir()
    probe_root = Path(probe_ref['uri']).parent
    with ThreadPoolExecutor(max_workers=2) as workers:
        builds = {side: workers.submit(_stage, arm, probe_root, out / side) for side, arm in arms.items()}
        executables = {side: build.result() for side, build in builds.items()}
    runs = []
    for index in range(repetitions):
        pair = {}
        for side in (('v2', 'v1') if index % 2 else ('v1', 'v2')):
            pair[side] = _observe(arms[side], executables[side], out / f'pair-{index}-{side}', config)
        runs.append(pair)
    result = dict(schema=SCHEMA, schema_version=1, probe=probe_ref, benchmark=bench_ref, case=case,
                  arms=arms, executable={side: _ref(path, 'scar.recompiled-diagnostic-executable')
                      for side, path in executables.items()}, cpu_ids=config['cpu_ids'], frames=frames, runs=runs,
                  summary=_summary(runs, frames, oracle), diagnostics_recompiled=True,
                  changes='ONLY_PLOT_DSO_AND_ITS_BINARY_DIGEST_GUARD',
                  clean_performance_claim=False, default_backend='NO_OP')
    (out / 'result.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    return result


def check(path):
    saved = read_artifact(_ref(path, SCHEMA))
    arms, config, oracle = _inputs(saved['probe'], saved['benchmark'], saved['case'])
    probe_root = Path(saved['probe']['uri']).parent
    for side, executable in saved['executable'].items():
        read_artifact(executable, json_document=False)
        diagnostic_root = Path(executable['uri']).parent.parent
        for name, payload in _diagnostic_members(arms[side], probe_root):
            _need((diagnostic_root / name).read_bytes() == payload,
                  'diagnostic source or installed member differs: ' + name)
    frames = sum(name.startswith('frame:') for name in oracle['outputs'])
    _need(saved['schema_version'] == 1 and saved['arms'] == arms and saved['frames'] == frames
          and set(saved['executable']) == set(arms)
          and saved['cpu_ids'] == config['cpu_ids'] and saved['summary'] == _summary(saved['runs'], frames, oracle),
          'diagnostic report differs from its raw saved observations')
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('probe', 'benchmark', 'case', 'out', 'check'):
        parser.add_argument('--' + name)
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args(argv)
    if args.check:
        report = check(args.check)
    elif all((args.probe, args.benchmark, args.case, args.out)):
        report = capture(args.probe, args.benchmark, args.case, args.out, repetitions=args.repetitions)
    else:
        parser.error('provide --check RECEIPT or --probe, --benchmark, --case and --out')
    print(json.dumps(report['summary']['total_abi_ns'], sort_keys=True))


if __name__ == '__main__':
    main()
