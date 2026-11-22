"""Observe source-bound phases of the original fixed-grid Python entry."""

import argparse
import ast
import json
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
from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.cpu-fixed-grid-python-lifecycle-observation'
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
    stdout, stderr = (_ref(row/name, 'application/octet-stream') for name in ('stdout', 'markers.stderr'))
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
            if capture:
                _capture(folder, command, delivery, output_path, import_only=import_only)
            rows.append(_check_row(folder, command, delivery, import_only=import_only))
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
