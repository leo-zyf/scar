"""Observe source-bound phases of the original fixed-grid Python entry."""

import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import stat
from statistics import median
import subprocess
from time import monotonic_ns

from scar.analysis.fixed_grid_entry_abi import _span
from scar.codegen.native_provider_bundle import _content
from scar.compiler.aten_entry_delivery import ABI_FILE
from scar.compiler.calibrate_array_entry import _ref, _write
from scar.compiler.cpu_fixed_grid_cli_delivery import SCHEMA as DELIVERY, verify_cpu_fixed_grid_cli_delivery
from scar.ir.execution_recipe import _canonical, read_artifact


SCHEMA = 'scar.cpu-fixed-grid-python-lifecycle-observation'
SHORT_SCHEMA = 'scar.cpu-fixed-grid-short-python-probe'
STEP_COUNTS = (2, 8)
_ORACLE_STATUS = 'scar.cpu-fixed-grid-short-python-oracle-exit'
_RAW = 'application/octet-stream'
_CLOCKS = 'scar.cpu-fixed-grid-python-lifecycle-clocks'
_FULL = ('PROCESS_ENTER', 'IMPORTS_BEGIN', 'IMPORTS_END', 'ENTRY_ENTER', 'INPUTS_BEGIN', 'INPUTS_END',
         'RECURRENCE_BEGIN', 'RECURRENCE_END', 'FILE_BEGIN', 'FILE_END', 'STDOUT_BEGIN', 'STDOUT_END', 'PROCESS_LEAVE')
_IMPORT = ('PROCESS_ENTER', 'IMPORTS_BEGIN', 'IMPORTS_END', 'PROCESS_LEAVE')
_PREFIX = b'SCAR_FIXED_GRID_PY_PHASE '


def _need(ok, reason):
    if not ok:
        raise ValueError('fixed-grid Python lifecycle: ' + reason)


def _statement(body, spans):
    low, high = spans[:2], spans[2:]
    matches = [row for row in body if _span(row)[:2] <= low and high <= _span(row)[2:]]
    _need(len(matches) == 1, 'source operation must name one containing statement')
    return matches[0]


def _render(source, abi, *, import_only):
    raw = read_artifact(source, json_document=False)
    path = Path(source['uri'])
    module = ast.parse(raw, filename=str(path))
    stem = '_scar_phase_' + source['sha256'][:12]
    _need(stem.encode() not in raw and raw.endswith(b'\n'), 'cannot inject isolated source marks')
    imports = [row for row in module.body if isinstance(row, (ast.Import, ast.ImportFrom))]
    _need(imports and all(not isinstance(row, ast.ImportFrom) or row.module != '__future__' for row in imports),
          'imports must use the admitted ordinary module convention')
    functions = [row for row in module.body if isinstance(row, ast.FunctionDef) and row.name == abi['solver']['entry']]
    _need(len(functions) == 1, 'one exact source entry statement is expected')
    function, = functions
    body = function.body
    edges = {}

    def add(line, code):
        edges.setdefault(line, []).append(code)

    def mark(line, name, indent=0):
        add(line, ' ' * indent + f'{stem}_mark({name!r})\n')

    newline = '\n'
    prelude = f'''import sys as {stem}_sys
from time import monotonic_ns as {stem}_clock
{stem}_events = []
def {stem}_mark(name):
    {stem}_events.append((name, {stem}_clock()))
def {stem}_finish():
    {stem}_mark('PROCESS_LEAVE')
    for {stem}_name, {stem}_ns in {stem}_events:
        {stem}_sys.stderr.write({_PREFIX.decode()!r} + {stem}_name + ' ' + str({stem}_ns) + {newline!r})
{stem}_mark('PROCESS_ENTER')
{stem}_sys.argv[0] = {str(path)!r}
{stem}_sys.path[0] = {str(path.parent)!r}
__file__ = {str(path)!r}
'''
    add(imports[0].lineno, prelude)
    mark(imports[0].lineno, 'IMPORTS_BEGIN')
    mark(imports[-1].end_lineno + 1, 'IMPORTS_END')
    if import_only:
        add(imports[-1].end_lineno + 1, f'{stem}_finish()\nraise SystemExit(0)\n')
    else:
        docstring = int(ast.get_docstring(function, clean=False) is not None)
        mark(body[docstring].lineno, 'ENTRY_ENTER', body[docstring].col_offset)
        nodes = abi['graph']['nodes']
        loads = [_statement(body, tuple(row['source_span'])) for row in nodes if row['kind'] == 'CPU_TENSOR_FILE_READ']
        _need(loads, 'one or more admitted file materializations are required')
        groups = [(loads[0], loads[-1], 'INPUTS')]
        for kind, name in (('FIXED_GRID_RECURRENCE', 'RECURRENCE'), ('FILE_PUBLICATION', 'FILE'),
                           ('STDOUT_PUBLICATION', 'STDOUT')):
            span, = (tuple(row['source_span']) for row in nodes if row['kind'] == kind)
            row = _statement(body, span)
            groups.append((row, row, name))
        for first, last, name in groups:
            _need(first.lineno <= last.end_lineno and first.col_offset == last.col_offset,
                  'incompatible source timing range')
            mark(first.lineno, name + '_BEGIN', first.col_offset)
            mark(last.end_lineno + 1, name + '_END', last.col_offset)
    source_lines = raw.splitlines(keepends=True)
    if not import_only:
        add(len(source_lines)+1, f'{stem}_finish()\n')
    output = []
    for index, row in enumerate([*source_lines, b''], 1):
        output.extend(line.encode() for line in edges.get(index, ()))
        output.append(row)
    edited = b''.join(output)
    compile(edited, str(path), 'exec')
    return edited


def _timing(raw, start, end, import_only):
    markers = {}
    order = []
    for line in raw.splitlines():
        _need(line.startswith(_PREFIX), 'source probe changed uninstrumented stderr')
        name, number = line.removeprefix(_PREFIX).decode('ascii').split()
        _need(name not in markers and number.isdecimal(), 'invalid or repeated source phase clock')
        order.append(name)
        markers[name] = int(number)
    ticks = [start, *markers.values(), end]
    _need(tuple(order) == (_IMPORT if import_only else _FULL) and ticks == sorted(ticks) and start < end,
          'source phase markers do not fit the parent clock and proved order')
    pairs = {'startup': (start, markers['PROCESS_ENTER']),
             'imports': (markers['IMPORTS_BEGIN'], markers['IMPORTS_END']),
             'teardown': (markers['PROCESS_LEAVE'], end)}
    if not import_only:
        pairs['ingress'] = markers['ENTRY_ENTER'], markers['INPUTS_BEGIN']
        pairs['terminal'] = markers['RECURRENCE_END'], markers['FILE_BEGIN']
        for name, role in (('loading', 'INPUTS'), ('execution', 'RECURRENCE'),
                           ('file_publication', 'FILE'), ('stdout_submission', 'STDOUT')):
            pairs[name] = markers[role+'_BEGIN'], markers[role+'_END']
    result = {name: high - low for name, (low, high) in pairs.items()}
    result.update(unattributed=end-start-sum(result.values()), total=end-start)
    return markers, result


def _capture(row, command, context, output, *, import_only):
    _need(not output.exists() and not output.is_symlink(), 'requested source publication is occupied')
    row.mkdir()
    start = monotonic_ns()
    launched = subprocess.run(command, cwd=context['working_directory'],
        env=context['launch']['provider_environment'], stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
    end = monotonic_ns()
    (row/'stdout').write_bytes(launched.stdout)
    (row/'markers.stderr').write_bytes(launched.stderr)
    _need(launched.returncode == 0, 'instrumented original Python failed: ' + str(row/'markers.stderr'))
    if import_only:
        _need(not output.exists() and not output.is_symlink(), 'isolated import calibration published output')
    else:
        _need(stat.S_ISREG(output.lstat().st_mode), 'expected a regular source output file')
        shutil.move(output, row/'publication.f32')
    _write(row/'clock.json', dict(schema=_CLOCKS, returncode=launched.returncode, start_ns=start, reap_ns=end))


def _check_row(row, command, delivery, *, import_only):
    stdout, stderr = (_ref(row/name, _RAW) for name in ('stdout', 'markers.stderr'))
    clock = _ref(row/'clock.json', _CLOCKS)
    ticks = read_artifact(clock)
    markers, phases = _timing(read_artifact(stderr, json_document=False), ticks['start_ns'], ticks['reap_ns'], import_only)
    raw = read_artifact(stdout, json_document=False)
    _need(ticks['returncode'] == 0, 'source phase probe did not complete normally')
    expected_files = {'stdout', 'markers.stderr', 'clock.json'}
    output = None
    if import_only:
        _need(raw == b'', 'isolated import calibration emitted source stdout')
    else:
        file = row/'publication.f32'
        expected_files.add(file.name)
        output = _ref(file, 'scar.native-float32-output')
        expected = delivery['validated_final_tensor']
        modes = {record['permission_bits'] for record in delivery['publication_modes'] if record['mode'] == 'ORIGINAL_PYTHON'}
        _need(raw == read_artifact(delivery['validated_source_stdout'], json_document=False)
              and all(output[key] == expected[key] for key in ('sha256', 'size_bytes'))
              and stat.S_ISREG(file.lstat().st_mode) and {stat.S_IMODE(file.lstat().st_mode)} == modes,
              'source phase probe changed stdout, file bytes or permissions')
    _need({path.name for path in row.iterdir()} == expected_files, 'unexpected source probe side effect')
    return dict(command=command, stdout=stdout, private_marks=stderr, clock=clock, marks_ns=markers,
                phases_ns=phases, file_publication=output)


def _observe_row(folder, command, context, output, *, capture, import_only, expected):
    if capture:
        _capture(folder, command, context, output, import_only=import_only)
    return _check_row(folder, command, expected, import_only=import_only)


def _series(runs):
    return {name: dict(samples_ns=values, min_ns=min(values), median_ns=median(values), max_ns=max(values))
            for name in runs[0]['phases_ns']
            for values in [[row['phases_ns'][name] for row in runs]]}


def _report(reference, root, samples, *, capture=False):
    delivery = verify_cpu_fixed_grid_cli_delivery(reference)
    manifest = read_artifact(delivery['project'])
    abi = read_artifact(dict(uri=str(Path(delivery['project']['uri']).parent/ABI_FILE),
                        schema='scar.fixed-grid-file-entry', **manifest['files'][ABI_FILE]))
    _need(type(samples) is int and 1 <= samples <= 8, 'unsupported source sample count')
    output_path = Path(delivery['argv'][abi['argv']['output']-1])
    source, argv = delivery['source'], delivery['argv']
    original = delivery['launch']['commands']['ORIGINAL_PYTHON']
    _need(original[5:] == [source['uri'], *argv], 'source probe must retain the exact admitted original command')
    if capture:
        root.mkdir(parents=True, exist_ok=False)
    groups = {}
    for group, import_only in (('full', False), ('import_only', True)):
        path = root/f'python-{group}.py'
        edited = _render(source, abi, import_only=import_only)
        if capture:
            path.write_bytes(edited)
        _need(path.read_bytes() == edited, 'generated source probe differs on cold replay')
        command = [*original[:5], str(path), *argv]
        rows = []
        for index in range(samples):
            folder = root/f'{group}-{index:02d}'
            rows.append(_observe_row(folder, command, delivery, output_path,
                                     capture=capture, import_only=import_only, expected=delivery))
        groups[group] = dict(source_probe=_ref(path, 'scar.python-source'), runs=rows, phases_ns=_series(rows))
    for input_file in delivery['file_inputs']:
        _need(_content(Path(input_file['path'])) == {key:input_file[key] for key in ('sha256', 'size_bytes')},
              'original probe changed the admitted inputs')
    return dict(schema=SCHEMA, schema_version=1, delivery=reference, source=source, project=delivery['project'],
        argv=argv, cpu_ids=delivery['cpu_ids'], working_directory=delivery['working_directory'],
        environment_sha256=delivery['environment_sha256'], sample_count=samples, **groups['full'],
        import_only=dict(**groups['import_only'], input_loading_executed=False, user_entry_executed=False,
                         scope='SAME_INTERPRETER_AND_TOP_LEVEL_IMPORTS_STOPS_BEFORE_MAIN'),
        timing_basis='SOURCE_BOUND_INSTRUMENTED_TIMELINE', clock='LINUX_MONOTONIC_NS',
        exact_original_observables_verified=True, physical_cost_bound=False, clean_benchmark=False,
        calibration_does_not_execute_unmodified_entry=True, deployment_authorized=False)


def measure(reference, out, *, samples=3):
    out = Path(out).resolve()
    report = _report(reference, out, samples, capture=True)
    _write(out/'result.json', report)
    return report


def verify_cpu_fixed_grid_python_lifecycle(reference):
    saved = read_artifact(reference)
    _need(reference['schema'] == SCHEMA and saved['schema_version'] == 1 and saved == _report(
        saved['delivery'], Path(reference['uri']).parent, saved['sample_count']), 'source lifecycle differs on cold replay')
    return saved


def _need_short(condition, detail):
    if not condition:
        raise ValueError('short fixed-grid Python probe: ' + detail)


def _oracle(folder, command, package, count, *, capture):
    output = Path(command[-1])
    expected = json.loads(read_artifact(package['stdout'], json_document=False))
    _need_short(expected['steps'] == package['steps'] and not output.exists() and not output.is_symlink(),
                'original publication contract and fresh diagnostic path required')
    if capture:
        folder.mkdir()
        result = subprocess.run(command, cwd=package['working_directory'], env=package['source_environment'],
                                stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
        (folder/'stdout').write_bytes(result.stdout)
        (folder/'stderr').write_bytes(result.stderr)
        _need_short(result.returncode == 0 and output.is_file() and not output.is_symlink(),
                    'short original activation did not produce a regular result')
        mode = stat.S_IMODE(output.lstat().st_mode)
        shutil.move(output, folder/'publication.f32')
        _write(folder/'outcome.json', dict(schema=_ORACLE_STATUS, returncode=result.returncode, mode=mode))
    outcome = _ref(folder/'outcome.json', _ORACLE_STATUS)
    status = read_artifact(outcome)
    stdout, stderr = (_ref(folder/name, _RAW) for name in ('stdout', 'stderr'))
    final = folder/'publication.f32'
    file = _ref(final, 'scar.native-float32-output')
    _need_short(json.loads(read_artifact(stdout, json_document=False)) == dict(expected, steps=count)
                and read_artifact(stderr, json_document=False) == b''
                and status['returncode'] == 0 and stat.S_ISREG(final.lstat().st_mode)
                and stat.S_IMODE(final.lstat().st_mode) == status['mode'] == package['publication']['expected_mode']
                and file['size_bytes'] == package['publication']['expected_file']['size_bytes'],
                'short original activation differs from the declared output shape, modes or streams')
    return dict(file=file, stdout=stdout, stderr=stderr, mode=status['mode'], command=command, outcome=outcome)


def _short_windows(row):
    clock = read_artifact(row['clock'])
    before = row['marks_ns']['RECURRENCE_BEGIN'] - clock['start_ns']
    recurrence = row['marks_ns']['RECURRENCE_END'] - row['marks_ns']['RECURRENCE_BEGIN']
    after = clock['reap_ns'] - row['marks_ns']['RECURRENCE_END']
    _need_short(min(before, recurrence, after) > 0 and before+recurrence+after == row['phases_ns']['total'],
                'three owner windows must partition the observed process clock')
    return dict(before_recurrence=before, recurrence=recurrence, after_recurrence=after)


def _short_report(package_ref, root, samples, *, capture=False):
    from scar.codegen.cpu_native_route_package import verify_cpu_fixed_grid_native_route_package
    from scar.compiler.cpu_fixed_grid_mapped_input_study import _input_files

    package = verify_cpu_fixed_grid_native_route_package(package_ref)
    delivery = read_artifact(package['owned_copy_delivery'])
    plan = read_artifact(package['plan'])
    abi = read_artifact(read_artifact(plan['proof'])['abi'])
    original = delivery['launch']['commands']['ORIGINAL_PYTHON']
    _need_short(type(samples) is int and 2 <= samples <= 8 and package['steps'] > max(STEP_COUNTS)
                and original[5:] == [package['source']['uri'], *package['argv']]
                and abi['argv']['steps'] == 2 and abi['argv']['output'] == len(package['argv']) == 3,
                'short samples must retain the exact admitted original and counted-loop arguments')
    stable = _input_files(package)
    source_probe = root/'source-probe.py'
    source_bytes = _render(package['source'], abi, import_only=False)
    if capture:
        _need_short(set(package['cpu_ids']) <= os.sched_getaffinity(0), 'recorded CPU cohort unavailable')
        root.mkdir(parents=True, exist_ok=False)
        with source_probe.open('xb') as stream:
            stream.write(source_bytes)
    _need_short(source_probe.read_bytes() == source_bytes, 'marked Python source differs from the exact original projection')

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
            row = _observe_row(folder, command, delivery, output,
                               capture=capture, import_only=False, expected=expected)
            rows.append(dict(row, windows_ns=_short_windows(row)))
        _need_short(not output.exists() and not output.is_symlink(), 'diagnostic output must be retained in its observation directory')
        cases[str(count)] = dict(argv=argv, oracle=oracle, samples=rows)

    _need_short(stable == _input_files(package), 'probe changed the stable input identities or contents')
    return dict(schema=SHORT_SCHEMA, schema_version=1, package=package_ref, source=package['source'],
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
    _need_short(root.resolve() == root and not any(root.is_relative_to(path) for path in protected),
                'fresh canonical calibration directory outside the original data and packages required')
    report = _short_report(package_ref, root, samples, capture=True)
    _write(root/'result.json', report)
    return report


def verify_short_python(reference):
    saved = read_artifact(reference)
    path = Path(reference['uri'])
    _need_short(reference['schema'] == SHORT_SCHEMA and path.name == 'result.json' and path == path.resolve(strict=True),
                'exact short Python probe receipt required')
    replay = _short_report(saved['package'], path.parent, saved['sample_count'])
    _need_short(_canonical(saved) == _canonical(replay), 'short Python observation differs at replay')
    return replay


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--delivery', type=Path)
    mode.add_argument('--check', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--samples', type=int, default=3)
    args = parser.parse_args(argv)
    if args.check is not None:
        report = verify_cpu_fixed_grid_python_lifecycle(_ref(args.check, SCHEMA))
    else:
        if args.out is None:
            parser.error('--delivery requires a fresh --out path')
        report = measure(_ref(args.delivery, DELIVERY), args.out, samples=args.samples)
    print(json.dumps(report['phases_ns']))


if __name__ == '__main__':
    raise SystemExit(main())
