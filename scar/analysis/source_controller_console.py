"""Observe one source controller text publication without redirecting Python objects."""

import argparse
import builtins
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import stat
import subprocess
import sys
import time

from scar.analysis.call_operand_reader import read_call_operands
from scar.analysis.source_symbol_variants import _table
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.trace.code_snapshot import code_key
from scar.trace.stream_backing import BackingStatus, observe_stream_backing

SCHEMA = 'scar.controller-console-observation'
_PLAN = 'scar.native-controller-planning-preparation'
_REQUEST = 'scar.controller-console-capture-intent'
_EVENTS = 'scar.controller-console-call-events'
_OPERATIONS = 'write,writev,pwrite64,pwritev,pwritev2,sendfile,copy_file_range,splice,vmsplice,tee'
_WRITE = re.compile(r'([0-9]+)\.([0-9]{6}) write\(1, "((?:\\x[0-9a-f]{2})*)", ([0-9]+)\) += ([0-9]+)')
_MAX_BYTES = 4096
_PRINT = builtins.print
_STDOUT = sys.__stdout__


def _require(ok, reason):
    if not ok:
        raise ValueError(reason)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _rows(block):
    for row in block:
        yield row
        for key in ('body', 'condition', 'then', 'otherwise'):
            if type(row.get(key)) is list:
                yield from _rows(row[key])


def _selection(plan_ref, requested=None):
    plan = read_artifact(plan_ref)
    source_ref = plan['source']
    source = read_artifact(source_ref)
    rows = list(_rows(source['program']))
    values = {row['id']: row['expression'] for row in rows if row['kind'] == 'EVAL'}
    effects = {row['id']: (i, row) for i, row in enumerate(source['effects'])}
    candidates = []
    for row in rows:
        if row['kind'] != 'CALL' or row['role'] != 'ORIGINAL_EFFECT':
            continue
        ordinal, record = effects[row['effect']]
        original = values.get(row['callee'].get('id'), {})
        namespace = original.get('namespace', ())
        if ((requested is None or requested == ordinal)
                and len(row['positional']) == 1 and not row['keywords']
                and original.get('kind') == 'NAMESPACE'
                and any(scope['kind'] == 'BUILTINS' and scope['name'] == 'print' for scope in namespace)
                and record['result_policy'] == 'ORIGINAL_RELEASE_ONLY'):
            candidates.append((ordinal, row, record))
    _require(len(candidates) == 1, 'need one source-bound builtin text effect candidate')
    ordinal, row, effect = candidates[0]
    _require(row['site'] == effect['site'] and row['source_fact'] == effect['source_fact'],
             'source effect does not match the retained call')
    time_report = read_artifact(read_artifact(read_artifact(plan['effect_plan'])['resources'])['time_schedule'])
    scopes_ref = time_report['observed_effects']['resource_scopes']
    scopes = read_artifact(scopes_ref)
    scope_root = Path(scopes_ref['uri']).parent
    anchor = effect['site']
    shard = scopes['shards'][anchor['unit']]
    keys = {binding['scope']['wire'] for binding in _table(scope_root / shard['path'], shard['bindings'])
            if any(completion['kind'] == 'FALLTHROUGH' and anchor in completion['ports']
                   for completion in binding['anchor']['completions'])}
    attempts = set()
    for scope in _table(scope_root, scopes['tables']['scopes']):
        if scope['id']['wire'] not in keys:
            continue
        begin = scope['begin']
        _require(scope['end']['kind'] == 'C_RETURN', 'retained console call lacks normal native completion')
        attempts.add((begin['caller']['process_id'], begin['attempt_index']))
    _require(attempts and len(attempts) == len(keys), 'retained source call has no exact native scopes')
    trace = read_call_operands(Path(scopes['capture']) / 'trace', scopes['project_root'])
    sites = {attempt.site for attempt in trace.attempts if (attempt.process_id, attempt.index) in attempts}
    _require(len(sites) == 1, 'retained source calls require one exact bytecode position')
    call, = sites
    code = source['source']
    _require(call.code == code['code_key'] and call.opname == 'CALL', 'retained call is not in the source controller code')
    body = _ref(code['path'], 'scar.python-source')
    _require(body['sha256'] == code['sha256'], 'controller source bytes differ')
    return dict(controller_plan=plan_ref, controller_source=source_ref, ordinal=ordinal,
                site=anchor, source_fact=effect['source_fact'], retained_scopes=scopes_ref,
                source_call=dict(code_key=call.code, offset=call.offset, opname=call.opname,
                                 position=list(call.position)), controller_body=body)


def _write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def _stdout_state():
    stream = sys.stdout
    _require(stream is _STDOUT and type(stream) is io.TextIOWrapper, 'stdout no longer uses its original text wrapper')
    backing = observe_stream_backing(stream)
    _require(backing.status is BackingStatus.OBSERVED and backing.fd == 1, 'original stdout fd not observed')
    return dict(original=True, chain_types=list(backing.chain_types), fd=backing.fd,
                **asdict(backing.stat), encoding=stream.encoding.lower(), errors=stream.errors,
                line_buffering=stream.line_buffering, write_through=stream.write_through)


def _worker(request_path):
    request_ref = _ref(request_path, _REQUEST)
    request = read_artifact(request_ref)
    choice, target = request['selection'], request['target']
    original = choice['controller_body']['uri']
    key = choice['source_call']['code_key']
    offset = choice['source_call']['offset']
    mon = sys.monitoring
    event, tool = mon.events, 4
    rows = []
    selected = []
    pending = None

    def enter(code, _):
        if code.co_filename == original and code_key(code) == key and not selected:
            selected.append(code)
            mon.set_local_events(tool, code, event.CALL)

    def before(_, instruction, callee, argument):
        nonlocal pending
        if instruction != offset:
            return
        _require(pending is None and callee is _PRINT and type(argument) is str,
                 'console call differs from original exact builtin and str argument')
        payload = argument.encode('utf-8', errors='strict')
        _require(len(payload) < _MAX_BYTES and len(rows) < _MAX_BYTES, 'console capture limit exceeded')
        pending = dict(builtin_print=True, positional_exact_str_utf8_hex=payload.hex(),
                       before_ns=time.time_ns(), stdout=_stdout_state())

    def returned(_, instruction, callee, _argument):
        nonlocal pending
        if instruction != offset:
            return
        _require(pending is not None and callee is _PRINT, 'native console completion has no matching begin')
        rows.append(dict(pending, after_ns=time.time_ns(), completion='C_RETURN'))
        pending = None

    def raised(_, instruction, _callee, _argument):
        _require(instruction != offset, 'original console did not complete successfully')

    mon.use_tool_id(tool, 'scar-console-observation')
    for kind, callback in ((event.PY_START, enter), (event.CALL, before),
                           (event.C_RETURN, returned), (event.C_RAISE, raised)):
        mon.register_callback(tool, kind, callback)
    mon.set_events(tool, event.PY_START)
    try:
        sys.argv = target['argv']
        sys.path[0] = str(Path(sys.argv[0]).parent)
        runpy.run_path(sys.argv[0], run_name='__main__')
    finally:
        for code in selected:
            mon.set_local_events(tool, code, 0)
        mon.set_events(tool, 0)
        mon.free_tool_id(tool)
    _require(len(selected) == 1 and rows and pending is None, 'selected console calls not observed')
    state = _stdout_state()
    _write(Path(request_path).with_name('calls.json'), dict(
        schema=_EVENTS, schema_version=1, intent=request_ref, pid=os.getpid(), occurrences=rows,
        terminal_stdout=state, scope_finish_ns=time.time_ns()))


def _publication(out, pid):
    trace_paths = sorted(out.glob('stdout.syscalls.*'))
    _require(0 < len(trace_paths) <= 256 and all(path.is_file() and path.stat().st_size <= 1 << 20 for path in trace_paths),
             'missing or excessive stdout syscall trace')
    writes = [(path, line) for path in trace_paths for line in path.read_text().splitlines() if line]
    _require(len(writes) == 1 and writes[0][0].name == f'stdout.syscalls.{pid}',
             'selected process requires one observed fd 1 syscall and no other writer')
    path, line = writes[0]
    match = _WRITE.fullmatch(line)
    _require(match is not None, 'require one complete fd 1 write with retained bytes')
    seconds, micros, data, count, actual = match.groups()
    data = bytes.fromhex(data.replace(r'\x', ''))
    stdout = out / 'stdout.bin'
    _require(0 < len(data) <= _MAX_BYTES and int(count) == int(actual) == len(data) and stdout.read_bytes() == data,
             'kernel write bytes differ from stdout output')
    return dict(kind='PROCESS_EXIT_STDOUT_WRITE', fd=1, timestamp_ns=int(seconds)*10**9 + int(micros)*1000,
                bytes_hex=data.hex(), stdout=_ref(stdout, 'application/octet-stream'),
                syscall_trace=_ref(path, 'scar.stdout-syscalls'),
                all_syscall_traces=[_ref(item, 'scar.stdout-syscalls') for item in trace_paths])


def _derive(request_ref, events_ref):
    out = Path(request_ref['uri']).parent
    request = read_artifact(request_ref)
    events = read_artifact(events_ref)
    choice = request['selection']
    choice_after = _selection(choice['controller_plan'], choice['ordinal'])
    _require(choice == choice_after and events['intent'] == request_ref, 'console evidence does not replay against the original call')
    target = request['target']
    _require(target['source'] == _ref(target['argv'][0], 'scar.python-entry-source'), 'original driver bytes differ')
    publication = _publication(out, events['pid'])
    rows = events['occurrences']
    sink = events['terminal_stdout']
    observed_file = (out / 'stdout.bin').stat()
    _require(rows and sink['original'] and sink['fd'] == 1 and stat.S_ISREG(sink['mode'])
             and sink['encoding'] == 'utf-8' and not sink['line_buffering'] and not sink['write_through']
             and (sink['device'], sink['inode'], sink['mode']) ==
                 (observed_file.st_dev, observed_file.st_ino, observed_file.st_mode)
             and sink['size_bytes'] == 0 and sink['chain_types'] == ['_io.TextIOWrapper', '_io.BufferedWriter', '_io.FileIO'],
             'console needs the original buffered UTF-8 regular stdout')
    pieces = []
    prior = 0
    for row in rows:
        _require(row['builtin_print'] is True and row['completion'] == 'C_RETURN' and row['stdout'] == sink
                 and prior <= row['before_ns'] <= row['after_ns'] <= events['scope_finish_ns'],
                 'console call identity, destination, or finish changed')
        data = bytes.fromhex(row['positional_exact_str_utf8_hex'])
        _require(data.isascii(), 'console output currently requires exact ASCII strings')
        pieces.append(data + b'\n')
        prior = row['after_ns']
    _require(events['scope_finish_ns'] < publication['timestamp_ns']
             and b''.join(pieces) == bytes.fromhex(publication['bytes_hex']),
             'ordered original console calls do not match the terminal kernel write')
    return dict(schema=SCHEMA, schema_version=1, **choice, target=target, occurrences=rows,
                publication=publication, scope_finish_ns=events['scope_finish_ns'],
                evidence=dict(intent=request_ref, events=events_ref), exclusive_observed_stdout_write=True,
                timer_clock='REALTIME_NS_MATCHED_WITHIN_OBSERVATION',
                unobserved_descriptor_effects_closed=False, performance_measured=False)


def read_console_observation(path):
    reference = _ref(path, SCHEMA)
    report = read_artifact(reference)
    refs = report['evidence']
    replay = _derive(refs['intent'], refs['events'])
    _require(report == replay and report == read_artifact(reference), 'console observation differs from evidence replay')
    return report


def capture(controller_plan, target_argv, out, *, ordinal=None):
    choice = _selection(_ref(controller_plan, _PLAN), ordinal)
    executable = shutil.which('strace')
    _require(executable and target_argv, 'need strace and a Python entry script')
    target_ref = _ref(target_argv[0], 'scar.python-entry-source')
    out = Path(out).absolute()
    out.mkdir()
    request = out / 'intent.json'
    target = dict(argv=[target_ref['uri'], *target_argv[1:]], source=target_ref,
                  interpreter=sys.executable, os_provider=_ref(executable, 'application/x-executable'))
    _write(request, dict(schema=_REQUEST, schema_version=1, selection=choice, target=target))
    argv = [executable, '-ff', '-qq', '-ttt', '-xx', '-s', str(_MAX_BYTES), '-e', f'trace={_OPERATIONS}',
            '-e', 'trace-fds=1', '-o', str(out / 'stdout.syscalls'), sys.executable, '-B',
            '-m', __name__ if __name__ != '__main__' else 'scar.analysis.source_controller_console',
            '--worker', str(request)]
    with (out / 'stdout.bin').open('xb') as stdout, (out / 'stderr.txt').open('xb') as stderr:
        subprocess.run(argv, stdout=stdout, stderr=stderr, check=True, timeout=45)
    report = _derive(_ref(request, _REQUEST), _ref(out / 'calls.json', _EVENTS))
    _write(out / 'result.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller-plan')
    parser.add_argument('--ordinal', type=int)
    parser.add_argument('--out')
    parser.add_argument('--check')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('target', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.worker:
        _worker(args.worker)
        return
    if args.check:
        report = read_console_observation(args.check)
    elif args.controller_plan and args.out:
        target = args.target[1:] if args.target[:1] == ['--'] else args.target
        report = capture(args.controller_plan, target, args.out, ordinal=args.ordinal)
    else:
        parser.error('provide --check or --controller-plan PLAN --out DIR -- SCRIPT [ARGS]')
    print(json.dumps(dict(ordinal=report['ordinal'], calls=len(report['occurrences']),
                          publication=report['publication']['kind'])))


if __name__ == '__main__':
    main()
