"""Calibrate a complete synchronous renderer lifetime with unchanged native providers."""

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import subprocess

from scar.analysis.native_service_environment import capture, validate
from scar.codegen import native_live_plot_service
from scar.compiler.observe_native_plot_entry import BENCHMARK, ENTRY, _benchmark_ref
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.paired_entry_selection import compare_entry_pairs


SCHEMA = 'scar.native-render-service-calibration'
SEQUENCE = 'scar.native-render-phase-input-sequence'
SCOPE = 'PREPARED_ROOTS_TO_SYNCHRONOUS_FRAME_PUBLICATION'
_ARMS = ('v1', 'v2')
_SIDES = dict(zip(_ARMS, ('original', 'candidate'), strict=True))
_EMPTY = sha256(b'').hexdigest()
_RESOURCE_LOG = 'SCAR_PLOT_SERVICE_RESOURCE_LOG'
_RESOURCE_SCHEMA = 'scar.native-live-plot-service-resources'


def _need(ok, reason):
    if not ok:
        raise ValueError('native render calibration: ' + reason)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _declared(record, schema):
    return dict(record, schema=schema)


def _sequence(reference, providers):
    sequence = read_artifact(reference)
    _need(sequence['schema_version'] == 1 and sequence['performance_measured'] is False,
          'prepared render inputs with explicit frame lineage are required')
    references = {arm: _declared(sequence[field], ENTRY) for arm, field in
                  zip(_ARMS, ('original_app', 'resident_app'), strict=True)}
    apps = {arm: read_artifact(app) for arm, app in references.items()}
    benchmark = read_artifact(_declared(sequence['clean_controller_witness'], BENCHMARK))
    _need(benchmark['qualified'] and not benchmark['instrumented']
          and benchmark['original_and_candidate_share_environment_and_affinity']
          and len(benchmark['cpu_ids']) == 1, 'one clean paired controller scope is required')
    case, = (row for row in benchmark['evidence'] if row['id'] == sequence['case'])
    _need(case['arguments'] == sequence['arguments'] and case['outputs_equal_for_required_comparisons'],
          'sequence must identify the same clean controller case')
    for arm, app in apps.items():
        _need(app['controller']['plot']['synchronous'] is True and app['controller']['plot']['provider'] == providers[arm]
              and references[arm] == _benchmark_ref(benchmark['pinned_inputs'][f'{arm}_package_manifest'], ENTRY),
              'sequence does not refer to the calibrated source/provider pair')
    shape = sequence['shape']
    _need(len(shape) == 2 and all(type(value) is int and value > 0 for value in shape), 'invalid packed root shape')
    rows, columns = shape
    coords = {axis: _declared(sequence[axis], 'scar.plot-root-le-f64') for axis in ('x', 'y')}
    for row in coords.values():
        _need(len(read_artifact(row, json_document=False)) == 8 * rows * columns,
              'coordinate root has the wrong payload size')
    for arm, app in apps.items():
        startup = app['controller']['startup']
        image = (Path(references[arm]['uri']).parent / startup['image']).read_bytes()
        _need(sha256(image).hexdigest() == startup['image_sha256'], 'controller startup asset changed')
        for axis, row in coords.items():
            binding = app['controller']['plot']['binding']['read_only_roots'][axis]
            offset = binding['offset']
            _need(binding['shape'] == shape
                  and sha256(image[offset:offset + row['size_bytes']]).hexdigest() == row['sha256'],
                  'coordinate root differs from the exact controller startup asset')
    frames = []
    for index, row in enumerate(sequence['frames']):
        time = row['time']
        _need(row['index'] == index and isfinite(time) and float(time).hex() == row['time_hex']
              and (not frames or time > frames[-1]['time']), 'source frame order or f64 time differs')
        content = _declared(row['z'], 'scar.plot-root-le-f64')
        _need(len(read_artifact(content, json_document=False)) == 8 * rows * columns,
              'Z root has the wrong payload size')
        png = _benchmark_ref(row['expected_png'], 'image/png')
        value = read_artifact(png, json_document=False)
        for side in _SIDES.values():
            clean = _benchmark_ref(case['samples'][side][0]['outputs'][f'frame:{index}'], 'image/png')
            _need(value == read_artifact(clean, json_document=False), 'prepared frame oracle differs from clean release output')
        frames.append(dict(z=content, time=time, time_hex=row['time_hex'], expected=png))
    _need(len(frames) == sequence['arguments']['expected_frames'], 'retained sequence has missing frames')
    return references, coords, frames, benchmark['cpu_ids'][0]


def _inputs(package_ref, sequence_ref):
    root = Path(package_ref['uri']).parent
    package = native_live_plot_service.check(root)
    _need(package == read_artifact(package_ref) and set(package['arms']) == set(_ARMS),
          'matched released timing package required')
    drivers = native_live_plot_service.members(root, package)
    providers = {arm: package['arms'][arm]['source_provider'] for arm in _ARMS}
    apps, coords, frames, cpu = _sequence(sequence_ref, providers)
    return drivers, providers, apps, coords, frames, cpu


def _layout(root, drivers, coords, frames, cpu, counts, warmups, repetitions, limit):
    for count in counts:
        for pair in range(warmups + repetitions):
            for arm in _ARMS[::(1 if pair % 2 == 0 else -1)]:
                driver = drivers[arm]
                target = root / 'published' / f'{count}-{pair}-{arm}'
                argv = ['/usr/bin/taskset', '-c', str(cpu), '/usr/bin/prlimit', f'--as={limit}:{limit}', '--',
                        *(driver[key]['uri'] for key in ('executable', 'layout', 'font')),
                        coords['x']['uri'], '0', coords['y']['uri'], '0']
                for index, frame in enumerate(frames[:count]):
                    argv += [frame['z']['uri'], '0', frame['time_hex'], str(target / f'{index:06d}.png')]
                resource_log = root / 'resources' / (target.name + '.json')
                yield dict(frames=count, pair=pair, mode=arm, target=str(target), argv=argv,
                           environment={_RESOURCE_LOG: str(resource_log)})


def _duration(log, arm, count):
    lines = log.splitlines()
    _need(lines and lines.pop(0) == 'operation\tframe\telapsed_ns', 'unexpected native timing protocol')
    observed = [(name, int(frame), int(elapsed)) for name, frame, elapsed in (line.split('\t') for line in lines)]
    expected = [('render', index) for index in range(count)]
    if arm == 'v2':
        expected = [('prepare', -1), *expected, ('destroy', -1)]
    _need([(name, frame) for name, frame, _ in observed] == expected
          and all(0 < elapsed < 2**60 for _, _, elapsed in observed), 'incomplete or invalid native call lifetime')
    return dict(total_ns=sum(row[2] for row in observed), calls=[dict(operation=name, frame=frame, ns=elapsed)
                                                               for name, frame, elapsed in observed])


def _derive(report, layout, frames):
    costs = {count: [] for count in report['frame_counts']}
    verified = {count: dict(v1=[], v2=[]) for count in costs}
    root = Path(report['output_root']) / 'published'
    sample_dirs = set()
    resource_files = set()
    for sample, expected in zip(report['runs'], layout, strict=True):
        count, arm = expected['frames'], expected['mode']
        target = Path(expected['target'])
        _need(target.parent == root and sample['execution'] == expected and sample['status'] == 0
              and sample['stderr'] == '', 'mismatched renderer sample or unexpected diagnostics')
        sample_dirs.add(target)
        resource_path = Path(expected['environment'][_RESOURCE_LOG])
        resource_files.add(resource_path)
        _need(sample['resources'] == _ref(resource_path, _RESOURCE_SCHEMA), 'resource sidecar differs')
        resources = read_artifact(sample['resources'])
        cap = report['address_space_limit_bytes']
        _need(resources['schema_version'] == 1
              and resources['address_space_limit_bytes'] == resources['address_space_hard_limit_bytes'] == cap
              and all(type(resources[key]) is int for key in ('vm_hwm_kib', 'vm_peak_kib'))
              and 0 < 1024 * resources['vm_hwm_kib'] <= 1024 * resources['vm_peak_kib'] <= cap,
              'invalid process resource observation or address-space limit')
        outputs = [_ref(path, 'image/png') for path in sorted(target.iterdir())]
        _need(outputs == sample['outputs'] and len(outputs) == count, 'mismatched recorded output files')
        for index, (output, frame) in enumerate(zip(outputs, frames[:count], strict=True)):
            _need(Path(output['uri']) == target / f'{index:06d}.png'
                  and read_artifact(output, json_document=False) == read_artifact(frame['expected'], json_document=False),
                  'renderer did not publish the original frame bytes in order')
        interval = _duration(sample['stdout'], arm, count)
        interval.update(vm_peak_bytes=1024 * resources['vm_peak_kib'], vm_rss_high_water_bytes=1024 * resources['vm_hwm_kib'])
        verified[count][arm].append(interval)
        # The driver stdout carries the timing receipt. The renderer's
        # observable outputs are the verified PNGs, and it emits no stderr.
        costs[count].append(dict(pair=expected['pair'], mode=arm, wall_ns=interval['total_ns'],
                                 returncode=0, stdout_sha256=_EMPTY, stderr_sha256=_EMPTY))
    _need(set(root.iterdir()) == sample_dirs, 'unexpected run directories')
    _need(set((root.parent / 'resources').iterdir()) == resource_files, 'unexpected resource sidecars')
    return {str(count): dict(timeline=verified[count], selection=compare_entry_pairs(
        costs[count], incumbent='v1', challenger='v2', warmup_pairs=report['warmups'],
        measurement_pairs=report['repetitions'])) for count in costs}


def _parameters(counts, frames, warmups, repetitions, limit):
    _need(counts == sorted(set(counts)) and counts and all(type(item) is int and 0 < item <= len(frames) for item in counts)
          and type(warmups) is int and warmups >= 0 and type(repetitions) is int and repetitions >= 3
          and type(limit) is int and limit > 0, 'positive exact cap, valid frame prefixes, and at least three measured pairs required')


def measure(package, sequence, out, frame_counts, limit, *, warmups=1, repetitions=5):
    package_ref = _ref(Path(package) / 'result.json', native_live_plot_service.SCHEMA)
    sequence_ref = _ref(sequence, SEQUENCE)
    drivers, providers, apps, coords, frames, cpu = inputs = _inputs(package_ref, sequence_ref)
    _parameters(frame_counts, frames, warmups, repetitions, limit)
    root = Path(out).resolve()
    root.mkdir()
    published = root / 'published'
    published.mkdir()
    (root / 'resources').mkdir()
    observed = capture(published, cpu)
    layout = list(_layout(root, drivers, coords, frames, cpu, frame_counts, warmups, repetitions, limit))
    samples = []
    for task in layout:
        target = Path(task['target'])
        target.mkdir()
        result = subprocess.run(task['argv'], cwd=root, env=task['environment'], capture_output=True, timeout=120)
        stderr = result.stderr.decode(errors='replace')
        _need(result.returncode == 0 and not stderr, f'render session failed under the declared cap: {stderr}')
        samples.append(dict(execution=task, status=result.returncode, stdout=result.stdout.decode('ascii'), stderr=stderr,
                            resources=_ref(task['environment'][_RESOURCE_LOG], _RESOURCE_SCHEMA),
                            outputs=[_ref(path, 'image/png') for path in sorted(target.iterdir())]))
    _need(capture(published, cpu) == observed and _inputs(package_ref, sequence_ref) == inputs,
          'provider, fixture or resource context changed while measuring')
    report = dict(schema=SCHEMA, schema_version=1, scope=SCOPE, package=package_ref, sequence=sequence_ref,
                  drivers=drivers, providers=providers, source_applications=apps, cpu=cpu, environment=observed,
                  output_root=str(root), frame_counts=frame_counts, warmups=warmups, repetitions=repetitions,
                  address_space_limit_bytes=limit, runs=samples)
    report['cases'] = _derive(report, layout, frames)
    with (root / 'result.json').open('x') as output:
        output.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return report


def check(path):
    path = Path(path).resolve(strict=True)
    report = read_artifact(_ref(path, SCHEMA))
    drivers, providers, apps, coords, frames, cpu = _inputs(report['package'], report['sequence'])
    counts, warmups, repetitions, limit = (report[key] for key in
                                          ('frame_counts', 'warmups', 'repetitions', 'address_space_limit_bytes'))
    _parameters(counts, frames, warmups, repetitions, limit)
    env = validate(report['environment'])
    root = path.parent
    _need(report['schema_version'] == 1 and report['scope'] == SCOPE and report['output_root'] == str(root)
          and env['cpu']['id'] == report['cpu'] == cpu and env['filesystem']['output_directory'] == str(root / 'published')
          and (report['drivers'], report['providers'], report['source_applications']) == (drivers, providers, apps),
          'render calibration context no longer matches its source and hardware receipts')
    layout = _layout(root, drivers, coords, frames, cpu, counts, warmups, repetitions, limit)
    _need(report['cases'] == _derive(report, layout, frames), 'render service comparison differs at cold replay')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--package', type=Path)
    inputs.add_argument('--check', type=Path)
    parser.add_argument('--sequence', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--frames', type=int, nargs='+')
    parser.add_argument('--address-space-limit-bytes', type=int)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repetitions', type=int, default=5)
    args = parser.parse_args(argv)
    if args.check:
        report = check(args.check)
    elif all((args.sequence, args.out, args.frames, args.address_space_limit_bytes)):
        report = measure(args.package, args.sequence, args.out, args.frames, args.address_space_limit_bytes,
                         warmups=args.warmups, repetitions=args.repetitions)
    else:
        parser.error('--package requires --sequence, --frames, --address-space-limit-bytes and --out')
    print(json.dumps({name: row['selection'] for name, row in report['cases'].items()}, sort_keys=True))


if __name__ == '__main__':
    main()
