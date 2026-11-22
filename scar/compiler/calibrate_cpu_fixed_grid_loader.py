"""Calibrate the exact native launcher on its early argument-count exit."""

import argparse
import json
from pathlib import Path
from statistics import median
import subprocess
from time import monotonic_ns

from scar.compiler.calibrate_array_entry import _ref, _write
from scar.compiler.cpu_fixed_grid_cli_delivery import SCHEMA as DELIVERY, verify_cpu_fixed_grid_cli_delivery
from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.cpu-fixed-grid-loader-proxy'
_CLOCKS = 'scar.cpu-fixed-grid-loader-clocks'
_USAGE = b'usage: aten-recurrence <count> <input.f32>... <new-output.f32>\n'
_GUARD = (b'    if (source_entry ? argc != 5 : static_cast<std::size_t>(argc) != first + 1 + graph::input_count'
          b' + graph::output_count) {\n'
          b'        std::fputs("usage: aten-recurrence <count> <input.f32>... <new-output.f32>\\n", stderr);\n'
          b'        return 2;\n    }')


def _need(ok, reason):
    if not ok:
        raise ValueError('fixed-grid loader proxy: ' + reason)


def _inputs(delivery_ref):
    delivery = verify_cpu_fixed_grid_cli_delivery(delivery_ref)
    project = delivery['project']
    manifest = read_artifact(project)
    main = dict(uri=str(Path(project['uri']).parent/'src/main.cpp'), schema='scar.cpp-source',
                **manifest['files']['src/main.cpp'])
    source = read_artifact(main, json_document=False)
    guard_at = source.find(_GUARD)
    setup_at = source.find(b'        at::set_num_threads(1);')
    launcher = read_artifact(delivery['native_launcher'], json_document=False)
    _need(source.count(_GUARD) == 1 and 0 < guard_at < setup_at
          and launcher.endswith(b' --source-entry "$@"\n'), 'early source argument exit must precede runtime setup')
    return delivery, main


def _sample(directory, command, delivery, *, capture):
    if capture:
        directory.mkdir()
        start = monotonic_ns()
        completed = subprocess.run(command, cwd=delivery['working_directory'],
            env=delivery['launch']['provider_environment'], stdin=subprocess.DEVNULL, capture_output=True, timeout=30)
        end = monotonic_ns()
        (directory/'stdout').write_bytes(completed.stdout)
        (directory/'stderr').write_bytes(completed.stderr)
        _write(directory/'clock.json', dict(schema=_CLOCKS, start_ns=start, reap_ns=end, status=completed.returncode))
    stdout, stderr = (_ref(directory/name,'application/octet-stream') for name in ('stdout', 'stderr'))
    clocks = _ref(directory/'clock.json', _CLOCKS)
    row = read_artifact(clocks)
    _need(row['status'] == 2 and type(row['start_ns']) is type(row['reap_ns']) is int
          and 0 < row['start_ns'] < row['reap_ns']
          and read_artifact(stdout, json_document=False) == b''
          and read_artifact(stderr, json_document=False) == _USAGE
          and {item.name for item in directory.iterdir()} == {'stdout', 'stderr', 'clock.json'},
          'launcher did not take the pinned argument guard')
    return dict(clock=clocks, stdout=stdout, stderr=stderr, wall_ns=row['reap_ns']-row['start_ns'])


def _report(delivery_ref, root, samples, *, capture=False):
    delivery, main = _inputs(delivery_ref)
    _need(type(samples) is int and 3 <= samples <= 16, 'expected three to sixteen measured samples')
    command = [delivery['native_launcher']['uri']]
    if capture:
        root.mkdir(parents=True, exist_ok=False)
    rows = [_sample(root/f'{index:02d}', command, delivery, capture=capture) for index in range(samples+1)]
    prior = 0
    for row in rows:
        clocks = read_artifact(row['clock'])
        _need(prior <= clocks['start_ns'], 'overlapping loader proxy samples')
        prior = clocks['reap_ns']
    values = [row['wall_ns'] for row in rows[1:]]
    return dict(schema=SCHEMA, schema_version=1, status='EXACT_NATIVE_LAUNCHER_EARLY_EXIT_MEASURED',
        delivery=delivery_ref, project=delivery['project'], source=delivery['source'],
        native_executable=delivery['native_executable'], native_launcher=delivery['native_launcher'], main=main,
        argv=delivery['argv'], probe_command=command, cpu_ids=delivery['cpu_ids'],
        working_directory=delivery['working_directory'], environment_sha256=delivery['environment_sha256'],
        warmup_samples=1, samples=samples, runs=rows,
        timing_ns=dict(samples_ns=values, min_ns=min(values), median_ns=median(values), max_ns=max(values)),
        loader_and_arg_guard_and_reap_are_combined=True, runtime_setup_executed=False,
        input_loading_executed=False, complete_entry_executed=False, physical_cost_bound=False,
        deployment_authorized=False)


def measure(delivery_ref, out, *, samples=5):
    out = Path(out).resolve()
    report = _report(delivery_ref, out, samples, capture=True)
    _write(out/'result.json', report)
    return report


def verify_cpu_fixed_grid_loader(reference):
    saved = read_artifact(reference)
    _need(reference['schema'] == SCHEMA and saved['schema_version'] == 1 and saved == _report(
        saved['delivery'], Path(reference['uri']).parent, saved['samples']), 'loader observation differs at replay')
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--delivery', type=Path)
    mode.add_argument('--check', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--samples', type=int, default=5)
    args = parser.parse_args(argv)
    if args.check is not None:
        report = verify_cpu_fixed_grid_loader(_ref(args.check, SCHEMA))
    else:
        if args.out is None:
            parser.error('--delivery requires a fresh --out path')
        report = measure(_ref(args.delivery, DELIVERY), args.out, samples=args.samples)
    print(json.dumps(report['timing_ns']))


if __name__ == '__main__':
    raise SystemExit(main())
