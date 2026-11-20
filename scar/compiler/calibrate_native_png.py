"""Measure clean, prepared RGBA-to-PNG publication on a pinned CPU."""

import argparse
import json
import os
from pathlib import Path
import re
from statistics import median_low
import subprocess

from scar.analysis.native_service_environment import capture, validate
from scar.codegen import native_png_service
from scar.codegen.native_rgba_png import _profile
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.native-png-service-calibration'
WITNESSES = 'scar.png-clean-witnesses'
SERVICE = 'PREPARED_RGBA_TO_SYNCHRONOUS_PNG_PUBLICATION'
_CLOCK = re.compile(rb'[1-9][0-9]{0,17}\n')
_NAME = re.compile(r'[a-zA-Z0-9_-]+')


def _need(valid, reason):
    if not valid:
        raise ValueError('native PNG calibration: ' + reason)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _inputs(package_ref, witness_ref):
    package = Path(package_ref['uri']).parent
    driver = native_png_service.check(package)
    _need(driver == read_artifact(package_ref) and driver['clean_no_internal_instrumentation'] is True
          and driver['measurement'] == 'STEADY_CLOCK_NS_INSIDE_PROCESS_AFTER_RGBA_PREFAULT',
          'pinned clean driver changed or its measurement scope differs')
    provider_ref = driver['source_provider']
    provider = read_artifact(provider_ref)
    canvas = provider['profile']['canvas']
    hosts = [_ref(path, 'scar.native-host-library') for path in driver['host_paths']]
    reference = read_artifact(witness_ref)
    _need(reference['schema_version'] == 1
          and reference['references']['v1_plot_release'] == provider_ref,
          'witnesses must resolve to this original provider release')
    witnesses = [dict(row, held_out=False) for row in reference['samples']]
    if reference.get('held_out') is not None:
        witnesses.append(dict(reference['held_out'], held_out=True))
    ids = [row['id'] for row in witnesses]
    _need(2 <= len(ids) <= 16 and len(set(ids)) == len(ids)
          and all(type(name) is str and _NAME.fullmatch(name) for name in ids),
          'two or more distinct retained frame witnesses are required')
    # Compare output encodings against independent recorded artifacts.  An
    # output match also checks the input's lossless RGBA content for this path.
    oracles = {}
    for row in witnesses:
        raw = read_artifact(row['rgba'], json_document=False)
        png = read_artifact(row['expected_png'], json_document=False)
        width, height, _ = _profile(png)
        _need([width, height] == canvas and len(raw) == 4 * width * height,
              'witness geometry differs from the qualified native provider')
        oracles[row['id']] = png
    _need(len(set(oracles.values())) == len(ids), 'witnesses must exercise distinct pixel contents')
    exe = package / driver['executable']
    return provider_ref, _ref(exe, 'scar.native-executable'), canvas, hosts, witnesses, oracles


def _validate_counts(cpu, warmups, repetitions):
    _need(type(cpu) is int and cpu >= 0 and type(warmups) is int and 1 <= warmups <= 4
          and type(repetitions) is int and 3 <= repetitions <= 15 and repetitions % 2 == 1,
          'use one CPU, bounded warmup rounds and an odd number of measured rounds')


def _layout(exe, witnesses, cpu, root, warmups, repetitions):
    for index in range(warmups + repetitions):
        # Rotate frames so an early or late run position is shared by the cases.
        offset = index % len(witnesses)
        for sample in witnesses[offset:] + witnesses[:offset]:
            name = sample['id']
            base = root / 'published' / f'{index:02d}-{name}.png'
            argv = ['/usr/bin/taskset', '-c', str(cpu), exe['uri'], sample['rgba']['uri'], str(base)]
            yield index, index < warmups, name, argv


def _measure(root, layout):
    rows = []
    for index, warmup, sample, argv in layout:
        proc = subprocess.run(argv, cwd=root, env={}, capture_output=True, timeout=30)
        _need(proc.returncode == 0 and not proc.stderr and _CLOCK.fullmatch(proc.stdout),
              f'clean publication failed for {sample}: ' + proc.stderr.decode(errors='replace'))
        rows.append(dict(round=index, warmup=warmup, sample=sample, argv=argv,
                         stdout=proc.stdout.decode('ascii'), status=proc.returncode,
                         published=_ref(argv[-1], 'image/png')))
    return rows


def _derive(root, report, layout, oracles):
    rows = report['runs']
    samples = {name: [] for name in oracles}
    for row, expected in zip(rows, layout, strict=True):
        index, warmup, sample, argv = expected
        _need((row['round'], row['warmup'], row['sample'], row['argv']) == (index, warmup, sample, argv)
              and row['status'] == 0 and row['published'] == _ref(argv[-1], 'image/png'),
              'recorded service arguments or output identity differ')
        stdout = row['stdout'].encode('ascii')
        _need(_CLOCK.fullmatch(stdout) and read_artifact(row['published'], json_document=False) == oracles[sample],
              'invalid service sample or PNG differs from original bytes')
        if not warmup:
            samples[sample].append(int(stdout))
    files = {path.name for path in (root / 'published').iterdir()}
    _need(files == {Path(row['published']['uri']).name for row in rows}, 'unexpected publication directory entries')
    return {name: dict(samples_ns=ns, min_ns=min(ns), median_ns=median_low(ns), max_ns=max(ns))
            for name, ns in samples.items()}


def measure(package, witnesses, out, cpu, *, warmups=1, repetitions=5):
    _validate_counts(cpu, warmups, repetitions)
    _need(cpu in os.sched_getaffinity(0), 'measured CPU is outside this process affinity')
    package_ref = _ref(Path(package) / 'result.json', native_png_service.SCHEMA)
    witness_ref = _ref(witnesses, WITNESSES)
    provider, exe, canvas, hosts, cases, oracles = _inputs(package_ref, witness_ref)
    root = Path(out).resolve()
    (root / 'published').mkdir(parents=True)
    environment = capture(root / 'published', cpu)
    layout = list(_layout(exe, cases, cpu, root, warmups, repetitions))
    rows = _measure(root, layout)
    _need(capture(root / 'published', cpu) == environment, 'CPU or filesystem identity changed during execution')
    # Revalidate the same package and witnesses after executing the sample set.
    _need(_inputs(package_ref, witness_ref) == (provider, exe, canvas, hosts, cases, oracles), 'pinned inputs changed')
    report = dict(schema=SCHEMA, schema_version=1, scope=SERVICE, output_root=str(root), package=package_ref,
                  witnesses=witness_ref, provider=provider, executable=exe, canvas=canvas, cpu_id=cpu,
                  host_libraries=hosts,
                  environment=environment, warmup_rounds=warmups, measurement_rounds=repetitions, runs=rows,
                  held_out_samples=[row['id'] for row in cases if row['held_out']],
                  renderer_included=False, rgba_loading_included=False, internal_phase_probes=False,
                  timings_authorize_parent_cost=False)
    report['per_frame'] = _derive(root, report, layout, oracles)
    with (root / 'result.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return report


def check(path):
    path = Path(path).resolve(strict=True)
    report = read_artifact(_ref(path, SCHEMA))
    provider, exe, canvas, hosts, cases, oracles = _inputs(report['package'], report['witnesses'])
    cpu, warmups, repetitions = (report[key] for key in ('cpu_id', 'warmup_rounds', 'measurement_rounds'))
    _validate_counts(cpu, warmups, repetitions)
    environment = validate(report['environment'])
    root = path.parent
    _need(report['schema_version'] == 1 and report['scope'] == SERVICE and report['output_root'] == str(root)
          and (report['provider'], report['executable'], report['canvas']) == (provider, exe, canvas)
          and report['host_libraries'] == hosts
          and report['held_out_samples'] == [row['id'] for row in cases if row['held_out']]
          and environment['cpu']['id'] == cpu and environment['filesystem']['output_directory'] == str(root / 'published')
          and all(report[key] is False for key in ('renderer_included', 'rgba_loading_included',
                                                 'internal_phase_probes', 'timings_authorize_parent_cost')),
          'clean service context differs from the persisted benchmark')
    layout = _layout(exe, cases, cpu, root, warmups, repetitions)
    _need(_derive(root, report, layout, oracles) == report['per_frame'], 'recorded service statistics differ')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--package', type=Path)
    inputs.add_argument('--check', type=Path)
    parser.add_argument('--witnesses', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repetitions', type=int, default=5)
    args = parser.parse_args(argv)
    if args.check:
        report = check(args.check)
    elif args.witnesses and args.out and args.cpu is not None:
        report = measure(args.package, args.witnesses, args.out, args.cpu,
                         warmups=args.warmups, repetitions=args.repetitions)
    else:
        parser.error('--package requires --witnesses, --out and --cpu')
    print(json.dumps(report['per_frame'], sort_keys=True))


if __name__ == '__main__':
    main()
