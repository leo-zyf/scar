"""Calibrate phase service on short original fixed-grid invocations."""

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

from scar.codegen.cpu_fixed_grid_native_route_package import verify_cpu_fixed_grid_native_route_package
from scar.compiler.calibrate_array_entry import _ref, _write
from scar.compiler.cpu_fixed_grid_mapped_input_study import _input_files
from scar.compiler.observe_cpu_fixed_grid_python_lifecycle import _capture, _check_row, _render
from scar.ir.execution_recipe import _canonical, read_artifact


SCHEMA = 'scar.cpu-fixed-grid-short-python-probe'
STEP_COUNTS = (2, 8)
_RAW = 'application/octet-stream'
_ORACLE_STATUS = 'scar.cpu-fixed-grid-short-python-oracle-exit'


def _need(condition, detail):
    if not condition:
        raise ValueError('short fixed-grid Python probe: ' + detail)


def _raw(reference):
    return read_artifact(reference, json_document=False)


def _oracle(folder, command, package, count, *, capture):
    output = Path(command[-1])
    expected = json.loads(_raw(package['stdout']))
    _need(expected['steps'] == package['steps'] and not output.exists() and not output.is_symlink(),
          'original publication contract and fresh diagnostic path required')
    if capture:
        folder.mkdir()
        result = subprocess.run(command, cwd=package['working_directory'], env=package['source_environment'],
                                stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
        (folder/'stdout').write_bytes(result.stdout)
        (folder/'stderr').write_bytes(result.stderr)
        _need(result.returncode == 0 and output.is_file() and not output.is_symlink(),
              'short original activation did not produce a regular result')
        mode = stat.S_IMODE(output.lstat().st_mode)
        shutil.move(output, folder/'publication.f32')
        _write(folder/'outcome.json', dict(schema=_ORACLE_STATUS, returncode=result.returncode, mode=mode))
    outcome = _ref(folder/'outcome.json', _ORACLE_STATUS)
    status = read_artifact(outcome)
    stdout, stderr = (_ref(folder/name, _RAW) for name in ('stdout', 'stderr'))
    final = folder/'publication.f32'
    file = _ref(final, 'scar.native-float32-output')
    _need(json.loads(_raw(stdout)) == dict(expected, steps=count) and _raw(stderr) == b''
          and status['returncode'] == 0 and stat.S_ISREG(final.lstat().st_mode)
          and stat.S_IMODE(final.lstat().st_mode) == status['mode'] == package['publication']['expected_mode']
          and file['size_bytes'] == package['publication']['expected_file']['size_bytes'],
          'short original activation differs from the declared output shape, modes or streams')
    return dict(file=file, stdout=stdout, stderr=stderr, mode=status['mode'], command=command, outcome=outcome)


def _windows(row):
    clock = read_artifact(row['clock'])
    before = row['marks_ns']['RECURRENCE_BEGIN'] - clock['start_ns']
    recurrence = row['marks_ns']['RECURRENCE_END'] - row['marks_ns']['RECURRENCE_BEGIN']
    after = clock['reap_ns'] - row['marks_ns']['RECURRENCE_END']
    _need(min(before, recurrence, after) > 0 and before+recurrence+after == row['phases_ns']['total'],
          'three owner windows must partition the observed process clock')
    return dict(before_recurrence=before, recurrence=recurrence, after_recurrence=after)


def _record(package_ref, root, samples, *, capture=False):
    package = verify_cpu_fixed_grid_native_route_package(package_ref)
    delivery = read_artifact(package['owned_copy_delivery'])
    plan = read_artifact(package['plan'])
    abi = read_artifact(read_artifact(plan['proof'])['abi'])
    original = delivery['launch']['commands']['ORIGINAL_PYTHON']
    _need(type(samples) is int and 2 <= samples <= 8 and package['steps'] > max(STEP_COUNTS)
          and original[5:] == [package['source']['uri'], *package['argv']]
          and abi['argv']['steps'] == 2 and abi['argv']['output'] == len(package['argv']) == 3,
          'short samples must retain the exact admitted original and counted-loop arguments')
    stable = _input_files(package)
    source_probe = root/'source-probe.py'
    source_bytes = _render(package['source'], abi, import_only=False)
    if capture:
        _need(set(package['cpu_ids']) <= os.sched_getaffinity(0), 'recorded CPU cohort unavailable')
        root.mkdir(parents=True, exist_ok=False)
        with source_probe.open('xb') as stream:
            stream.write(source_bytes)
    _need(source_probe.read_bytes() == source_bytes, 'marked Python source differs from the exact original projection')

    cases = {}
    for count in STEP_COUNTS:
        case = root/f'steps-{count}'
        if capture:
            case.mkdir()
        output = case/'requested.f32'
        argv = [package['argv'][0], str(count), str(output)]
        oracle = _oracle(case/'oracle', [*original[:6], *argv], package, count, capture=capture)
        expected = dict(delivery, validated_final_tensor=oracle['file'], validated_source_stdout=oracle['stdout'],
                        publication_modes=[dict(mode='ORIGINAL_PYTHON', permission_bits=oracle['mode'])])
        command = [*original[:5], str(source_probe), *argv]
        rows = []
        for ordinal in range(samples):
            folder = case/f'sample-{ordinal:02}'
            if capture:
                _capture(folder, command, delivery, output, import_only=False)
            row = _check_row(folder, command, expected, import_only=False)
            rows.append(dict(row, windows_ns=_windows(row)))
        _need(not output.exists() and not output.is_symlink(), 'diagnostic output must be retained in its observation directory')
        cases[str(count)] = dict(argv=argv, oracle=oracle, samples=rows)

    _need(stable == _input_files(package), 'probe changed the stable input identities or contents')
    return dict(schema=SCHEMA, schema_version=1, package=package_ref, source=package['source'],
        target=package['hardware'], cpu_ids=package['cpu_ids'], working_directory=package['working_directory'],
        environment_sha256=package['environment_sha256'], step_counts=list(STEP_COUNTS), sample_count=samples,
        inputs=package['file_inputs'], input_snapshots=stable, marked_source=_ref(source_probe, 'scar.python-source'),
        cases=cases, clock='LINUX_MONOTONIC_NS', timing_basis='INSTRUMENTED_SHORT_SOURCE_OWNER_WINDOWS',
        oracle_processes_timed=False, clean_benchmark=False, deployment_authorized=False)


def observe_short_python(package_ref, out, *, samples=3):
    root = Path(out).absolute()
    package = read_artifact(package_ref)
    protected = [Path(package_ref['uri']).parent, Path(package['publication']['path']).parent,
                 *(Path(route['project_root']) for route in package['routes'].values()),
                 *(Path(row['path']).parent for row in package['file_inputs'])]
    _need(root.resolve() == root and not any(root.is_relative_to(path) for path in protected),
          'fresh canonical calibration directory outside the original data and packages required')
    report = _record(package_ref, root, samples, capture=True)
    _write(root/'result.json', report)
    return report


def verify_short_python(reference):
    saved = read_artifact(reference)
    path = Path(reference['uri'])
    _need(reference['schema'] == SCHEMA and path.name == 'result.json' and path == path.resolve(strict=True),
          'exact short Python probe receipt required')
    replay = _record(saved['package'], path.parent, saved['sample_count'])
    _need(_canonical(saved) == _canonical(replay), 'short Python observation differs at replay')
    return replay
