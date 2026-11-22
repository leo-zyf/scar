"""Compare two native source CLIs that differ only in fixed-grid file input backing."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import stat
from statistics import pvariance
import subprocess
from time import perf_counter_ns

from scar.codegen.native_provider_bundle import _content
from scar.compiler import aten_source_entry, cpu_fixed_grid_cli_delivery as delivery
from scar.compiler.calibrate_array_entry import _write
from scar.compiler.entry_delivery import _aten
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.paired_entry_selection import compare_entry_pairs


SCHEMA = 'scar.cpu-fixed-grid-mapped-input-study'
_RAW = 'application/octet-stream'
_COPY, _MAPPED = 'OWNED_COPY', 'PROVIDER_MAP'
_MODES = (_COPY, _MAPPED)
_WARMUPS = 1
_RUNTIME = 'src/runtime/file_io.hpp'
_TRACE = Path('/usr/bin/strace')
_TRACE_PID = re.compile(r'trace\.([1-9][0-9]*)')


def _need(condition, detail):
    if not condition:
        raise ValueError('CPU fixed-grid mapped input study: ' + detail)


def _ref(path, schema=_RAW):
    _need(path.resolve(strict=True) == path and stat.S_ISREG(path.lstat().st_mode), 'expected an exact local file')
    return artifact_reference(path, schema)


def _raw(reference):
    return read_artifact(reference, json_document=False)


def _input_files(binding):
    result = []
    for row in binding['file_inputs']:
        path = Path(row['path'])
        before = path.lstat()
        _need(stat.S_ISREG(before.st_mode) and path.resolve(strict=True) == path
              and _content(path) == {key: row[key] for key in ('sha256', 'size_bytes')},
              'input payload differs from its scoped file: ' + str(path))
        after = path.lstat()
        keys = ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        snapshot = {key: getattr(after, key) for key in keys}
        _need(all(getattr(before, key) == value for key, value in snapshot.items()),
              'input identity changed while reading its receipt: ' + str(path))
        result.append(dict(path=str(path), snapshot=snapshot))
    return result


def _inputs(control_ref, mapped_ref):
    control = delivery.verify_cpu_fixed_grid_cli_delivery(control_ref)
    scope = read_artifact(control['domain'])['scope']
    reference = control['project']
    baseline, candidate = read_artifact(reference), read_artifact(mapped_ref)
    descriptor = dict(candidate['source_entry_cli'])
    proof = descriptor.pop('mapped_input_proof')
    read_artifact(proof)
    changed = {name for name, row in baseline['files'].items() if row != candidate['files'].get(name)}
    binary = baseline['executable']
    _need(mapped_ref['schema'] == reference['schema'] and control['validation']['exact_publication_mode'] is True
          and descriptor == baseline['source_entry_cli'] and set(baseline['files']) == set(candidate['files'])
          and {_RUNTIME, binary} <= changed <= {_RUNTIME, binary, 'validation/build.log', 'validation/ldd.txt'}
          and {key: value for key, value in candidate.items() if key not in {'files', 'source_entry_cli'}}
              == {key: value for key, value in baseline.items() if key not in {'files', 'source_entry_cli'}},
          'mapped project changes the source, compiled body, schedule, runtime or publication ABI')
    evidence = _aten(scope, mapped_ref, candidate)
    for name in ('source', 'target', 'file_inputs', 'steps', 'cpu_ids'):
        _need(evidence[name] == control[name], 'mapped project crosses the exact ' + name)
    _need(evidence['memory'] == control['storage']
          and read_artifact(evidence['plan'])['placement'] == control['native_placement'],
          'mapped file receipts do not change the admitted graph backing budget or CPU lanes')
    validated_mode, = {row['permission_bits'] for row in control['publication_modes']}
    launchers = {_COPY: control['native_launcher'], _MAPPED: evidence['launcher']}
    executable = _ref(Path(mapped_ref['uri']).parent/binary, 'application/x-executable')
    _need(executable['sha256'] == candidate['files'][binary]['sha256'], 'mapped ELF differs from its manifest')
    return dict(control_delivery=control_ref, control_project=reference, mapped_project=mapped_ref,
                mapped_input_proof=proof, source=control['source'], target=control['target'], argv=control['argv'],
                file_inputs=control['file_inputs'], steps=control['steps'], cpu_ids=control['cpu_ids'],
                native_placement=control['native_placement'], storage=control['storage'],
                working_directory=control['working_directory'], environment=control['launch']['provider_environment'],
                environment_sha256=control['environment_sha256'], control_executable=control['native_executable'],
                mapped_executable=executable, launchers=launchers, expected_mode=validated_mode,
                original_stdout=control['validated_source_stdout'], original_output=control['validated_final_tensor'])


def _commands(binding, output, steps):
    abi = read_artifact(_ref(Path(binding['control_project']['uri']).parent/aten_source_entry.ABI_FILE,
                             'scar.fixed-grid-file-entry'))
    args = list(binding['argv'])
    args[abi['argv']['steps']-1] = steps
    args[abi['argv']['output']-1] = str(output)
    return {mode: [binding['launchers'][mode]['uri'], *args] for mode in _MODES}


def _capture(directory, command, binding, output, *, trace=None):
    directory.mkdir(exist_ok=False)
    _need(not output.exists() and not output.is_symlink(), 'requested destination is already occupied')
    launch = command if trace is None else [*_trace_prefix(trace, directory), *command]
    start = perf_counter_ns() if trace is None else None
    completed = subprocess.run(launch, cwd=binding['working_directory'], env=binding['environment'],
        stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
    elapsed = perf_counter_ns() - start if start is not None else None
    for name, payload in (('stdout.bin', completed.stdout), ('stderr.bin', completed.stderr)):
        (directory/name).write_bytes(payload)
    _need(completed.returncode == 0 and output.exists() and not output.is_symlink()
          and stat.S_ISREG(output.lstat().st_mode), 'scoped source CLI failed or did not create a regular output')
    mode = stat.S_IMODE(output.lstat().st_mode)
    shutil.move(output, directory/'published.f32')
    _write(directory/'clock.json', dict(schema='scar.cpu-fixed-grid-mapped-input-clock',
            returncode=completed.returncode, wall_ns=elapsed, publication_mode=mode))


def _rows(directory, *, command, mode, pair, expected_mode, timed):
    clock = _ref(directory/'clock.json', 'scar.cpu-fixed-grid-mapped-input-clock')
    clock_doc = read_artifact(clock)
    stdout, stderr, published = (_ref(directory/name) for name in ('stdout.bin', 'stderr.bin', 'published.f32'))
    elapsed = clock_doc['wall_ns']
    _need(clock_doc['returncode'] == 0 and _raw(stderr) == b''
          and clock_doc['publication_mode'] == expected_mode == stat.S_IMODE(Path(published['uri']).lstat().st_mode)
          and (type(elapsed) is int and elapsed > 0 if timed else elapsed is None),
          'retained exit, file mode or timed observation differs')
    return dict(mode=mode, pair=pair, argv=command, clock=clock, returncode=0, wall_ns=elapsed,
                stdout=stdout, stderr=stderr, published=published, publication_mode=expected_mode)


def _benchmark(binding, root, pairs, *, capture=False):
    _need(type(pairs) is int and 5 <= pairs <= 24, 'five to twenty-four clean native pairs required')
    output = Path(binding['argv'][-1])
    _need(output.resolve() == output and not output.exists() and not output.is_symlink()
          and not root.is_relative_to(output.parent), 'exact fresh source output outside study directory required')
    commands = _commands(binding, output, binding['argv'][1])
    runs, timings = [], []
    stable = _input_files(binding)
    for pair in range(pairs + _WARMUPS):
        for mode in (_MODES if pair % 2 == 0 else _MODES[::-1]):
            path = root/f'{pair:02d}-{mode}'
            if capture:
                _capture(path, commands[mode], binding, output)
            row = _rows(path, command=commands[mode], mode=mode, pair=pair, expected_mode=binding['expected_mode'], timed=True)
            _need(_raw(row['stdout']) == _raw(binding['original_stdout'])
                  and _raw(row['published']) == _raw(binding['original_output']),
                  'source stdout or full final Tensor differs from the original scope')
            runs.append(row)
            timings.append(dict(pair=pair, mode=mode, returncode=0, wall_ns=row['wall_ns'],
                                stdout_sha256=row['stdout']['sha256'], stderr_sha256=row['stderr']['sha256']))
    _need(stable == _input_files(binding), 'input file contents or identity changed between matched native arms')
    selection = compare_entry_pairs(timings, incumbent=_COPY, challenger=_MAPPED, warmup_pairs=_WARMUPS,
                                    measurement_pairs=pairs)
    return dict(runs=runs, selection=selection, input_snapshots=stable, population_variance_ns2={
        mode: pvariance(row['samples_ns']) for mode, row in selection['metrics'].items()})


def _trace_prefix(reference, directory):
    return [reference['uri'], '-ff', '-qq', '-yy', '-s', '0', '-e', 'trace=read,pread64,mmap',
            '-o', str(directory/'trace')]


def _trace_rows(directory, inputs):
    files = sorted(path for path in directory.iterdir() if _TRACE_PID.fullmatch(path.name))
    _need(files and len(files) <= 64, 'missing or unwieldy per-task syscall witness')
    references = [_ref(path) for path in files]
    buffers = {row['path']: dict(reads_bytes=0, maps=[]) for row in inputs}
    for reference in references:
        for line in _raw(reference).decode('utf-8', errors='surrogateescape').splitlines():
            for name, evidence in buffers.items():
                descriptor = re.escape('<' + name + '>')
                if '<' + name + '>' not in line:
                    continue
                read = re.fullmatch(r'(?:read|pread64)\(\d+' + descriptor + r', .*\)\s+= (\d+)', line)
                if read is not None:
                    evidence['reads_bytes'] += int(read[1])
                    continue
                mapping = re.fullmatch(r'mmap\([^,]+, (\d+), ([^,]+), ([^,]+), \d+' + descriptor
                                       + r', ([^,]+)\)\s+= (0x[\da-f]+)', line)
                _need(mapping is not None, 'unqualified input syscall in retained diagnostic: ' + name)
                evidence['maps'].append(dict(size_bytes=int(mapping[1]), protection=mapping[2], flags=mapping[3],
                                             offset_bytes=int(mapping[4], 0)))
    return dict(raw=references, per_input=buffers)


def _diagnostic(binding, root, tool, *, capture=False):
    _need(tool['uri'] == str(_TRACE) and tool == _ref(_TRACE, 'application/x-executable'), 'exact local syscall tool required')
    output = root/'trace-publication.f32'
    commands = _commands(binding, output, '1')
    rows = {}
    stable = _input_files(binding)
    for mode in _MODES:
        directory = root/mode
        if capture:
            _capture(directory, commands[mode], binding, output, trace=tool)
        row = _rows(directory, command=[*_trace_prefix(tool, directory), *commands[mode]], mode=mode, pair=None,
                    expected_mode=binding['expected_mode'], timed=False)
        rows[mode] = dict(run=row, **_trace_rows(directory, binding['file_inputs']))
    a, b = (rows[mode]['run'] for mode in _MODES)
    _need(_raw(a['stdout']) == _raw(b['stdout']) and _raw(a['published']) == _raw(b['published'])
          and a['published']['size_bytes'] == binding['original_output']['size_bytes']
          and stable == _input_files(binding), 'one-step diagnostic changed the treatment output or input files')
    payload = 0
    for row in binding['file_inputs']:
        expected = row['size_bytes']
        copy, mapped = (rows[mode]['per_input'][row['path']] for mode in _MODES)
        _need(copy['reads_bytes'] == expected and copy['maps'] == []
              and mapped['reads_bytes'] == 0 and len(mapped['maps']) == 1
              and mapped['maps'][0]['size_bytes'] == expected and mapped['maps'][0]['offset_bytes'] == 0
              and 'PROT_READ' in mapped['maps'][0]['protection'], 'syscalls did not establish the proposed input backing')
        payload += expected
    return dict(tool=tool, acquisition='LINUX_STRACE_PATH_QUALIFIED_READ_PREAD_AND_MMAP', pair_scope='ONE_STEP',
                runs=rows, explicit_file_read_payload_removed_bytes=payload,
                bytes_mapped_at_offset_zero=payload, equal_final_tensor=True, clean_benchmark=False,
                kernel_io_bytes_measured=False, page_fault_work_removed_proven=False)


def _report(binding, root, pairs, tool, *, capture=False):
    if capture:
        root.mkdir(parents=True, exist_ok=False)
        _need(set(binding['cpu_ids']) <= os.sched_getaffinity(0), 'bounded CPU cohort is unavailable')
    result = _benchmark(binding, root, pairs, capture=capture)
    diagnostic = None
    if tool is not None:
        path = root/'diagnostic'
        if capture:
            path.mkdir()
        diagnostic = _diagnostic(binding, path, tool, capture=capture)
    return dict(schema=SCHEMA, schema_version=1, status='SCOPED_NATIVE_INPUT_BACKING_COMPARED', **binding, **result,
                warmup_pairs=_WARMUPS, measurement_pairs=pairs, file_reads_evidence=diagnostic,
                validation=dict(exact_stdout=True, exact_stderr=True, exact_exit_status=True, bitwise_final_file=True,
                                exact_publication_mode=True, input_files_unchanged=True),
                timing_scope='NATIVE_SOURCE_CLI_LAUNCH_THROUGH_EXIT_INCLUDES_FILE_IO',
                target_executed=True, clean_timed_runs=True, python_workload_executed=False,
                perf_authority='BOUND_COHORT_STABLE_FILES_MEASUREMENT_ONLY', execution_default='KEEP_ORIGINAL',
                automatic_deployment=False, deployment_authorized=False)


def measure(control_delivery_ref, mapped_project_ref, out, *, pairs=5, diagnostic=True):
    """Run alternating clean full-entry pairs and optionally one separate syscall probe."""
    out = Path(out).absolute()
    _need(out.resolve() == out and not out.exists() and not out.is_symlink(), 'fresh result directory required')
    binding = _inputs(control_delivery_ref, mapped_project_ref)
    tool = _ref(_TRACE, 'application/x-executable') if diagnostic and _TRACE.is_file() else None
    report = _report(binding, out, pairs, tool, capture=True)
    _write(out/'result.json', report)
    return report


def verify_cpu_fixed_grid_mapped_input_study(reference):
    """Re-qualify exact projects and replay archived measurements and raw I/O evidence."""
    saved = read_artifact(reference)
    path = Path(reference['uri'])
    _need(reference['schema'] == SCHEMA and path.name == 'result.json' and path.resolve(strict=True) == path,
          'exact mapped input study receipt required')
    diagnostic = saved['file_reads_evidence']
    binding = _inputs(saved['control_delivery'], saved['mapped_project'])
    report = _report(binding, path.parent, saved['measurement_pairs'], None if diagnostic is None else diagnostic['tool'])
    _need(report == saved, 'saved matched input study differs from cold replay')
    return saved


def main(argv=None):
    from scar.analysis import cpu_fixed_grid_mapped_inputs as proof
    from scar.optimization import cpu_fixed_grid_mapped_input_plan as planner

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prove = commands.add_parser('prove')
    prove.add_argument('--project', type=Path, required=True)
    prove.add_argument('--out', type=Path, required=True)
    run = commands.add_parser('measure')
    run.add_argument('--control', type=Path, required=True)
    run.add_argument('--candidate', type=Path, required=True)
    run.add_argument('--out', type=Path, required=True)
    run.add_argument('--pairs', type=int, default=5)
    run.add_argument('--skip-io-probe', action='store_true')
    plan = commands.add_parser('plan')
    plan.add_argument('--proof', type=Path, required=True)
    plan.add_argument('--study', type=Path)
    plan.add_argument('--out', type=Path, required=True)
    check = commands.add_parser('check')
    check.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'measure':
        report = measure(_ref(args.control, delivery.SCHEMA), _ref(args.candidate, delivery._PROJECT),
                         args.out, pairs=args.pairs, diagnostic=not args.skip_io_probe)
    elif args.command == 'check':
        report = planner.verify_cpu_fixed_grid_mapped_input_plan(_ref(args.plan, planner.SCHEMA))
    else:
        if args.command == 'prove':
            report = proof.derive_cpu_fixed_grid_mapped_inputs(_ref(args.project, delivery._PROJECT))
        else:
            observation = None if args.study is None else _ref(args.study, SCHEMA)
            report = planner.derive_cpu_fixed_grid_mapped_input_plan(_ref(args.proof, proof.SCHEMA),
                                                                    observation=observation)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        _write(args.out, report)
    print(json.dumps(dict(status=report['status'], selected_runtime=report.get('selected_runtime'))))


if __name__ == '__main__':
    raise SystemExit(main())
