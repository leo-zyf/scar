"""Compare the original bound slice call and its detached native schedules."""

import argparse
import ctypes
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import re
import subprocess
from time import perf_counter_ns

from scar.codegen import native_numpy_partition as native
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.paired_entry_selection import compare_entry_pairs, replay_entry_pairs
from scar.trace.code_snapshot import code_key


SCHEMA = 'scar.native-numpy-partition-calibration'
_MODES = ('source-call', 'preserved-copy', 'elided-copy')


def _source(witness_ref, sample_id, iterations, output):
    import numpy
    from scar.trace.numpy_partition_witness import _provider

    witness = read_artifact(witness_ref)
    focus = witness['focus']
    function = getattr(import_module(focus['module']), focus['symbol'])
    source = artifact_reference(Path(function.__code__.co_filename).resolve(strict=True), focus['source']['schema'])
    observed = witness['numpy']
    if (source != focus['source'] or code_key(function.__code__) != focus['code_id']
            or _provider(numpy) != observed):
        raise ValueError('source process did not bind the recorded callable, NumPy provider and error policy')
    sample, = (item for item in witness['samples'] if item['id'] == sample_id)
    storage = {}
    views = {}
    for name in ('target', 'matrix', 'out'):
        view = sample[name]
        identity = view['storage_id']
        resource = sample['storages'][identity]
        payload = bytearray(resource['size_bytes'] + 63)
        base = ctypes.addressof(ctypes.c_char.from_buffer(payload))
        padding = (resource['address_mod_64'] - base) % 64
        span = memoryview(payload)[padding:padding + resource['size_bytes']]
        storage[name] = span
        if name != 'out':
            span[:] = read_artifact(sample['storage_before'][identity], json_document=False)
        views[name] = numpy.ndarray(view['shape'], dtype=view['dtype'], buffer=span,
                                    offset=view['offset_bytes'], strides=view['strides_bytes'])
    slices = [(Ellipsis, row['index']) for row in sample['slices']]
    target, matrix, out = (views[name] for name in ('target', 'matrix', 'out'))
    before = perf_counter_ns()
    for _ in range(iterations):
        result = function(target, matrix, slices, out=out)
    elapsed = perf_counter_ns() - before
    if result is not out:
        raise ValueError('original partition returned another output object')
    with Path(output).open('xb') as stream:
        stream.write(storage['out'])
    print(elapsed)


def _selection(samples, counts, *, archived=None):
    empty = sha256(b'').hexdigest()
    comparisons = {}
    for incumbent, challenger in (('source-call', 'preserved-copy'), ('source-call', 'elided-copy'),
                                   ('preserved-copy', 'elided-copy')):
        arms = (incumbent, challenger)
        rows = [dict(pair=row['pair'], mode=row['mode'], wall_ns=row['elapsed_ns'], returncode=0,
                     stdout_sha256=row['output']['sha256'], stderr_sha256=empty)
                for row in samples if row['mode'] in arms]
        name = challenger + '_over_' + incumbent
        context = dict(incumbent=incumbent, challenger=challenger, warmup_pairs=counts[0], measurement_pairs=counts[1])
        comparisons[name] = (compare_entry_pairs(rows, **context) if archived is None
                             else replay_entry_pairs(rows, archived['comparisons'][name], **context))
    elide, copy = (comparisons[name]['verdict'] == 'ADVANCE_TO_RELEASE_REVIEW' for name in
                   ('elided-copy_over_preserved-copy', 'preserved-copy_over_source-call'))
    elide_over_source = comparisons['elided-copy_over_source-call']['verdict'] == 'ADVANCE_TO_RELEASE_REVIEW'
    decision = 'elided-copy' if elide and elide_over_source else 'preserved-copy' if copy else 'source-call'
    return dict(chosen_for_measured_scope=decision, comparisons=comparisons, default_entry_backend='NO_OP')


def _profile(package):
    case = max((case for case in package['cases'] if case['implementation'] == 'preserved-copy'),
               key=lambda row: row['sizes']['target'])
    return case, {row['implementation']: row for row in package['cases'] if row['source_sample'] == case['source_sample']}


def measure(package_path, interpreter, pythonpath, output, *, iterations=10000, pairs=3):
    if type(iterations) is not int or not 1 <= iterations <= 1_000_000 or type(pairs) is not int or pairs < 3:
        raise ValueError('at least three measured pairs and a positive bounded call count required')
    package = native.check(package_path)
    package_path = Path(package_path).resolve(strict=True)
    root = package_path.parent
    iteration = read_artifact(package['iteration'])
    witness = iteration['witness']
    case, cases = _profile(package)
    sample_id = case['source_sample']
    if set(cases) != set(_MODES[1:]):
        raise ValueError('largest discovered profile does not offer both admitted native schedules')
    expected = read_artifact(case['expected'], json_document=False)
    cpu, = package['cpu_ids']
    runtime = artifact_reference(Path(interpreter).absolute(), 'scar.native-executable')
    environment = dict(PYTHONPATH=':'.join(str(Path(path).resolve(strict=True)) for path in (
        Path(__file__).resolve().parents[2], *pythonpath)), PYTHONHASHSEED='0', OPENBLAS_NUM_THREADS='1')
    output = Path(output).resolve()
    output.mkdir()
    rows = []
    warmups = 1
    for pair in range(warmups + pairs):
        offset = pair % len(_MODES)
        for mode in _MODES[offset:] + _MODES[:offset]:
            path = output / f'{pair:02d}-{mode}.bin'
            if mode == 'source-call':
                command = ['/usr/bin/taskset', '-c', str(cpu), runtime['uri'], '-B', '-m', 'scar.compiler.calibrate_numpy_partition',
                           '--original-witness', witness['uri'], '--original-witness-sha256', witness['sha256'],
                           '--sample', sample_id, '--iterations', str(iterations), '--out', str(path)]
                result = subprocess.run(command, env=environment, cwd=output, capture_output=True, timeout=120)
                if result.returncode or result.stderr or not re.fullmatch(rb'[1-9][0-9]*\n', result.stdout):
                    raise ValueError('source partition timing failed: ' + result.stderr.decode(errors='replace'))
                row = dict(argv=command, elapsed_ns=int(result.stdout), output=native._ref(path, 'scar.observed-native-complex-output'))
            else:
                row = native._one_run(root / package['binary'], cases[mode], path, cpu, iterations)
            if read_artifact(row['output'], json_document=False) != expected:
                raise ValueError('timed process changed the observed original output bytes')
            rows.append(dict(row, mode=mode, pair=pair))
    read_artifact(runtime, json_document=False)
    report = dict(schema=SCHEMA, schema_version=1, package=native._ref(package_path, native.SCHEMA),
                  source_witness=witness, sample=sample_id, cpu_ids=[cpu], source_python=runtime,
                  source_environment=environment, timing='PREPARED_REPEAT_CALLS_EXCLUDING_INGRESS_AND_PUBLICATION',
                  iterations_per_process=iterations, warmup_rounds=warmups, measured_rounds=pairs, observations=rows,
                  selection=_selection(rows, (warmups, pairs)), application_benchmark=False, deployment_authorized=False)
    (output / 'result.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return report


def check(path):
    path = Path(path).resolve(strict=True)
    report = read_artifact(native._ref(path, SCHEMA))
    package = native.check(report['package']['uri'])
    if package != read_artifact(report['package']):
        raise ValueError('measured package receipt changed')
    case, _ = _profile(package)
    if (report['schema_version'] != 1 or report['application_benchmark'] is not False
            or report['deployment_authorized'] is not False or report['sample'] != case['source_sample']
            or report['source_witness'] != read_artifact(package['iteration'])['witness']
            or report['cpu_ids'] != package['cpu_ids']):
        raise ValueError('measured scope differs from the source-bound profile')
    expected = read_artifact(case['expected'], json_document=False)
    read_artifact(report['source_python'], json_document=False)
    for row in report['observations']:
        output = path.parent / f'{row["pair"]:02d}-{row["mode"]}.bin'
        if (row['output'] != native._ref(output, 'scar.observed-native-complex-output')
                or read_artifact(row['output'], json_document=False) != expected):
            raise ValueError('timed output differs from the witnessed original bytes')
    counts = (report['warmup_rounds'], report['measured_rounds'])
    if _selection(report['observations'], counts, archived=report['selection']) != report['selection']:
        raise ValueError('recorded cost choice differs from replay')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--package', type=Path)
    source.add_argument('--original-witness', type=Path)
    source.add_argument('--check', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--python', type=Path)
    parser.add_argument('--pythonpath', action='append', type=Path, default=[])
    parser.add_argument('--sample')
    parser.add_argument('--original-witness-sha256')
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--pairs', type=int, default=3)
    args = parser.parse_args(argv)
    if args.check:
        print(json.dumps(check(args.check)['selection'], sort_keys=True))
    elif args.original_witness and args.sample and args.out:
        witness = native._ref(args.original_witness, 'scar.numpy-partition-witness')
        if witness['sha256'] != args.original_witness_sha256:
            raise ValueError('original capture changed before source timing')
        _source(witness, args.sample, args.iterations, args.out)
    elif args.package and args.python and args.out:
        result = measure(args.package, args.python, args.pythonpath, args.out, iterations=args.iterations, pairs=args.pairs)
        print(json.dumps(result['selection'], sort_keys=True))
    else:
        parser.error('supply --check RESULT, --package with --python and --out, or --original-witness with --sample and --out')


if __name__ == '__main__':
    main()
