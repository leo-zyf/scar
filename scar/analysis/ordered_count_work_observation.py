"""Replay work and ordering from an untimed syscall trace of a native sparse entry."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from struct import Struct

from scar.codegen import native_ordered_timing as prior
from scar.codegen import native_parameterized_sparse_entry as entry
from scar.ir.execution_recipe import _canonical, artifact_reference, read_artifact
from scar.optimization.native_parameterized_sparse_plan import work_for_count


SCHEMA = 'scar.ordered-count-work-observation'
_TRACE = 'scar.ordered-count-strace-file'
_STDIO = 'application/octet-stream'
_HEX = r'(?:\\x[0-9a-f]{2})*'
_EXEC = re.compile(rf'^execve\("({_HEX})", \[(.*?)\], .*\)\s+= 0$')
_ARG = re.compile(rf'"({_HEX})"')
_FORK = re.compile(r'^(?:clone|clone3|fork|vfork)\(.*\)\s+= ([1-9]\d*)$')
_PIN = re.compile(r'^sched_setaffinity\(0, \d+, \[([0-9 ]+)\]\)\s+= 0$')
_PAIR = re.compile(r'^socketpair\(AF_UNIX, SOCK_STREAM\|SOCK_CLOEXEC, 0, \[\d+<UNIX-STREAM:\[(\d+)->(\d+)\]>, '
    r'\d+<UNIX-STREAM:\[\2->\1\]>\]\)\s+= 0$')
_REAP = re.compile(r'^wait4\((\d+), .*\)\s+= \1$')
_PIPE = re.compile(rf'^(read|write)\((\d+)<({_HEX})>, "({_HEX})"(?:\.\.\.)?, \d+\)\s+= (\d+)$')
_PIPE_ID = re.compile(rb'pipe:\[(\d+)\]')
_WRITE = re.compile(rf'^write\(1<[^>]+>, "({_HEX})"(?:\.\.\.)?, \d+\)\s+= (\d+)$')
_SOCKET = re.compile(rf'^(sendto|recvfrom)\(\d+<UNIX-STREAM:\[(\d+)(?:->\d+)?\]>, '
    rf'"({_HEX})"(?:\.\.\.)?, (\d+), .*\)\s+= (\d+)$')
_SEQ = Struct('<Q')
_RESPONSE = Struct('<QiI')
_SCALE = Struct('<d')


def _require(condition, detail):
    if not condition:
        raise ValueError('ordered count observation: ' + detail)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _one(values, reason):
    values = list(values)
    _require(len(values) == 1, reason)
    return values[0]


def _bytes(value):
    return bytes.fromhex(value.replace('\\x', ''))


def _member(root, reference):
    child = root / reference['uri']
    _require(not Path(reference['uri']).is_absolute() and child.is_relative_to(root)
        and child.resolve(strict=True) == child, 'expected a package-local artifact')
    return dict(reference, uri=str(child))


def _matches(rows, expression):
    return [(ordinal, match.groups()) for ordinal, row in enumerate(rows, 1) if (match := expression.fullmatch(row))]


def _processes(trace_refs, paths, placement):
    logs = {int(Path(ref['uri']).suffix.removeprefix('.')):
        read_artifact(ref, json_document=False).decode('ascii').splitlines() for ref in trace_refs}
    _require(len(logs) == len(trace_refs) and all(rows[-1] == '+++ exited with 0 +++' for rows in logs.values()),
        'every followed process must have exited normally')
    launches = {pid: (_bytes(path).decode(), *(_bytes(arg).decode() for arg in _ARG.findall(args)[1:]))
        for pid, rows in logs.items() for _, (path, args) in _matches(rows, _EXEC)}
    ids = {role: _one((pid for pid, args in launches.items() if args == tuple(expected)),
        'unique exact command for ' + role) for role, expected in paths.items()}
    parent = ids['supervisor']
    solver = ids['solver']
    prep = ('topology', 'mesh', 'basis', 'ingress')
    children = lambda pid: [int(args[0]) for _, args in _matches(logs[pid], _FORK)]
    prepared = [ids[role] for role in prep]
    workers = children(solver)
    _require(children(parent) == [*prepared, solver] and set(launches) == set(ids.values())
        and set(logs) == {parent, solver, *prepared, *workers} and len(workers) == len(placement['worker_cpu_ids']),
        'exact startup chain and compiled worker count must be present')
    pin = lambda pid: [list(map(int, row[0].split())) for _, row in _matches(logs[pid], _PIN)]
    owner, all_cpus = [placement['owner_cpu_id']], placement['launch_cpu_ids']
    _require(pin(parent) == [owner, all_cpus, owner] and pin(solver) == [owner, all_cpus]
        and [pin(pid) for pid in workers] == [[[cpu]] for cpu in placement['worker_cpu_ids']],
        'observed pin calls differ from the admitted process placement')
    _require(not any(children(pid) for pid in [*prepared, *workers]), 'unexpected unmodeled subprocess')
    return logs, ids, workers


def _socket_rows(rows):
    return [(line, direction, int(inode), _bytes(data), int(wanted), int(size))
        for line, (direction, inode, data, wanted, size) in _matches(rows, _SOCKET)]


def _frame(records, direction, inode, size, *, materialized=False):
    left, data, start = size, b'', None
    while left:
        row = next(records, None)
        _require(row is not None, 'incomplete worker socket frame')
        line, observed, address, fragment, wanted, actual = row
        _require(observed == direction and address == inode and wanted == left and 0 < actual <= wanted,
                 'socket transfer order or exact frame length differs')
        _require(not materialized or len(fragment) == actual, 'control bytes were truncated')
        left -= actual
        if start is None:
            start = line
        if materialized:
            data += fragment
    return data, [start, line]


def _dispatch(logs, solver, workers, count, width, window, payload):
    sources = logs[solver]
    pair_events = _matches(sources, _PAIR)
    pairs = [[int(value) for value in row] for _, row in pair_events]
    _require(len(pairs) == width, 'missing one socketpair per admitted worker slot')
    owner = iter(_socket_rows(sources))
    slots = [dict(slot=slot, pid=pid, owner_endpoint=owner_inode, worker_endpoint=worker_inode,
        requests=[]) for slot, (pid, (owner_inode, worker_inode)) in enumerate(zip(workers, pairs, strict=True))]
    for slot in slots:
        ready, ready_span = _frame(owner, 'recvfrom', slot['owner_endpoint'], 1, materialized=True)
        _require(ready == b'\x01', 'worker readiness byte differs')
        slot['ready_on_owner_line'] = ready_span[1]

    def issue(sequence):
        slot = slots[sequence % width]
        endpoint = slot['owner_endpoint']
        header, seq_span = _frame(owner, 'sendto', endpoint, _SEQ.size, materialized=True)
        scale, scale_span = _frame(owner, 'sendto', endpoint, _SCALE.size, materialized=True)
        _require(_SEQ.unpack(header) == (sequence,)
            and scale == _SCALE.pack(1.0 + (sequence + 1) / (count + 1)), 'sequence or binary64 scale differs')
        request = dict(sequence=sequence, issued_lines=[seq_span[0], scale_span[1]])
        slot['requests'].append(request)
        return request

    issued = min(count, window)
    instances = [issue(sequence) for sequence in range(issued)]
    for sequence in range(count):
        slot = slots[sequence % width]
        endpoint = slot['owner_endpoint']
        response, header = _frame(owner, 'recvfrom', endpoint, _RESPONSE.size, materialized=True)
        _, output = _frame(owner, 'recvfrom', endpoint, payload)
        _require(_RESPONSE.unpack(response) == (sequence, 0, 0), 'owner FIFO success header differs')
        instances[sequence]['received_lines'] = [header[0], output[1]]
        if issued < count:
            instances.append(issue(issued))
            issued += 1
    _require(next(owner, None) is None, 'owner socket events remain after the declared work')
    for slot in slots:
        inode = slot['worker_endpoint']
        worker = iter(_socket_rows(logs[slot['pid']]))
        ready, span = _frame(worker, 'sendto', inode, 1, materialized=True)
        _require(ready == b'\x01', 'worker bootstrap differs')
        slot['ready_on_worker_line'] = span[1]
        for request in slot['requests']:
            sequence = request['sequence']
            header, first = _frame(worker, 'recvfrom', inode, _SEQ.size, materialized=True)
            scale, last = _frame(worker, 'recvfrom', inode, _SCALE.size, materialized=True)
            _require(_SEQ.unpack(header) == (sequence,)
                and scale == _SCALE.pack(1.0 + (sequence + 1) / (count + 1)), 'worker consumed a different request')
            response, first_response = _frame(worker, 'sendto', inode, _RESPONSE.size, materialized=True)
            _, last_response = _frame(worker, 'sendto', inode, payload)
            _require(_RESPONSE.unpack(response) == (sequence, 0, 0), 'worker did not return one successful response')
            request.update(worker_receive_lines=[first[0], last[1]], worker_send_lines=[first_response[0], last_response[1]])
        final = next(worker, None)
        _require(final is not None and (final[1], final[2], final[4], final[5]) == ('recvfrom', inode, _SEQ.size, 0)
            and next(worker, None) is None, 'worker request channel did not end at exact EOF')
        slot['eof_line'] = final[0]
    return slots, instances, pair_events[0][0]


def _pipes(logs, ids, case, stdout, first_dispatch_socket):
    roles = ('supervisor', 'topology', 'mesh', 'basis', 'ingress', 'solver', 'supervisor')
    events = {role: [(line, direction, int(pipe[1]), int(size))
        for line, (direction, fd, resource, _, size) in _matches(logs[pid], _PIPE)
        if (pipe := _PIPE_ID.fullmatch(_bytes(resource)))
        if (direction, fd) in {('read', '0'), ('write', '1')} or role == 'supervisor']
        for role, pid in ids.items()}
    edges = []
    for producer, consumer in zip(roles, roles[1:]):
        outgoing = {inode for _, direction, inode, size in events[producer] if direction == 'write' and size > 0}
        incoming = {inode for _, direction, inode, size in events[consumer] if direction == 'read' and size > 0}
        pipe = _one(outgoing & incoming, 'unique traversed pipe: ' + producer + ' to ' + consumer)
        send = [(line, size) for line, direction, inode, size in events[producer] if direction == 'write' and inode == pipe]
        receive = [(line, size) for line, direction, inode, size in events[consumer] if direction == 'read' and inode == pipe]
        size = sum(value for _, value in send)
        _require(size == sum(value for _, value in receive) and size > 0, 'pipe byte conservation failed')
        if consumer == 'solver':
            _require(receive[-1][1] == 0 and receive[-1][0] < first_dispatch_socket and size == case,
                'complete canonical CSR and EOF must precede the first worker socket')
        edges.append(dict(producer=producer, consumer=consumer, inode=pipe, bytes=size,
            sender_lines=[send[0][0], send[-1][0]], receiver_lines=[receive[0][0], receive[-1][0]]))
    final = [(line, int(size)) for line, (_, size) in _matches(logs[ids['supervisor']], _WRITE)]
    reaped = {int(pid): line for line, (pid,) in _matches(logs[ids['supervisor']], _REAP)}
    _require(len(stdout) == edges[-1]['bytes'] == sum(size for _, size in final)
        and set(reaped) == set(ids.values()) - {ids['supervisor']}
        and all(line < final[0][0] for line in reaped.values()), 'stdout digest must follow all successful prep/solver reaps')
    return edges, [final[0][0], final[-1][0]], reaped


def _source_boundaries(parent, root):
    refs = {name: _member(root, parent['members'][path]) for name, path in (
        ('supervisor', 'src/supervisor.cpp'), ('ingress', 'src/coo.cpp'))}
    solver_root = root / 'solver'
    solver = read_artifact(_ref(solver_root / 'artifact.json', 'scar.native-parameterized-sparse-solver'))
    refs['solver'] = _member(solver_root, solver['stream']['source'])
    refs['dispatch'] = _member(solver_root, _one((row for row in solver['headers']
        if row['uri'].endswith('dispatch_v1.h')), 'checked FIFO dispatch source'))
    sources = {name: read_artifact(ref, json_document=False).decode().splitlines() for name, ref in refs.items()}

    def markers(name, begin, end, meanings):
        lines = sources[name]
        first = _one((i for i, text in enumerate(lines, 1) if begin in text), 'source phase entry: ' + begin)
        last = _one((i for i, text in enumerate(lines, 1) if end in text and i >= first), 'source phase exit: ' + end)
        return dict(source=name, inclusive_marker_lines=[first, last], scope=meanings, duration_ns=None)

    return refs, dict(
        upstream_start=markers('supervisor', 'auto initial = make_pipe();', 'canonical[1].close_now();',
            'SEED_STAGING_THROUGH_PREPARATION_CHILD_LAUNCH_ONLY; NOT_PREPARATION_COMPLETION'),
        canonical_ingress=markers('solver', 'auto observed = input_header;', 'return run_requests(indices, offsets, input, request_count);',
            'FIRST_HEADER_READ_THROUGH_VALIDATION_AND_EOF_BEFORE_DISPATCH'),
        owner_dispatch_bootstrap=markers('dispatch', 'detail::require_config(selected);', 'const auto issue =',
            'DISPATCH_CONFIGURATION_AND_FORK_TO_ALL_WORKER_READY_BYTES; EXCLUDES_SOLVER_SETUP_BEFORE_DISPATCH'),
        worker_payload=markers('dispatch', 'const auto status = worker(sequence,', 'Response response{sequence, status, 0};',
            'AFTER_COMPLETE_REQUEST_TO_RESPONSE_CONSTRUCTION; INCLUDES_SCALE_APPLICATION_NUMERICAL_PROVIDER_AND_RESULT_COPY'),
        numerical_provider=markers('solver', 'const auto status = provider->solve(symbolic.get(),', 'rhs.data(), solution.data(), &timing);',
            'NESTED_PROVIDER_INVOCATION_IN_WORKER_PAYLOAD; SOURCE_ONLY'),
        digest_commit=markers('solver', 'publication.append(row.result);', '++published;',
            'ONE_FULL_ORDERED_RESULT_APPENDED_AND_SEQUENCE_ADVANCED; SOURCE_ONLY'),
        final_publication=markers('supervisor', 'const auto line = read_digest(answer[0].get(), end);', 'commit(std::span(line).first<65>());',
            'PRIVATE_DIGEST_READ_THROUGH_ALL_PREPARATION_AND_SOLVER_REAPS_AND_SINGLE_PUBLIC_OUTPUT'))


def _historical(paths, parent):
    observations = []
    payload_sha = parent['csr_boundary_validation']['preserved_payload_sha256']
    for path in paths:
        ref = _ref(path, prior.SCHEMA)
        old = read_artifact(ref)
        raw = read_artifact(old['input'], json_document=False)
        _require(sha256(raw[Struct(parent['protocol']['header']['format']).size:]).hexdigest() == payload_sha,
                 'historical timing has different CSR numerical input')
        # Keep the exact legacy read and bounds validation. A diagnostic clock
        # samples a previous solver context, even when numerical input matches.
        timing = _member(Path(ref['uri']).parent, old['evidence'])
        read_artifact(timing, json_document=False)
        rebuilt = prior._read_timing(Path(timing['uri']), old)
        _require(rebuilt == old['timing'] and len(rebuilt['workers']) == old['width'],
                 'historical timing aggregates, affinity, or counts differ')
        observations.append(dict(artifact=ref, aggregates=timing, count=old['count'], width=old['width'],
            window=old.get('window', old['width']), launch_cpu_ids=old['placement']['launch_cpu_ids'],
            included='SOLVER_ONLY; NO_SUPERVISOR_OR_UPSTREAM_GEOMETRY',
            scope=[rebuilt[name] for name in ('cost_scope', 'marker_scope', 'receive_scope')],
            owner_phases=list(rebuilt['owner']['phases']), worker_phases=list(rebuilt['workers'][0]['phases']),
            transferred_to_runtime_count=False))
    return observations


def project(package_path, evidence_directory, *, count, stdout_oracle, prior_timings=()):
    """Derive phase boundaries, actual work edges, and exact untimed byte counts."""
    package = _ref(package_path, entry.SCHEMA)
    root = Path(package['uri']).parent
    doc = entry.check(package['uri'])
    plan = read_artifact(doc['parameterized_plan'])
    expected = work_for_count(plan, count)
    protocol = doc['protocol']
    position, width, window = protocol['placement'], protocol['width'], protocol['window']
    shape = protocol['header']
    paths = {role: [str(root / _one((Path(name) for name in doc['reused_upstream']
              if Path(name).parent == Path('upstream') / role / 'bin'), 'pinned executable for ' + role))]
              for role in ('topology', 'mesh', 'basis')}
    paths['supervisor'] = [str(root / doc['binary']['uri']), str(count)]
    paths['ingress'] = [str(root / doc['ingress']['uri']), str(count)]
    solver = root / 'solver'
    inner = read_artifact(_ref(solver / 'artifact.json', 'scar.native-parameterized-sparse-solver'))
    paths['solver'] = [str(solver / inner['stream']['executable']['uri'])]
    evidence = Path(evidence_directory).resolve(strict=True)
    trace = [_ref(path, _TRACE) for path in sorted(evidence.glob('syscalls.*'))]
    stdout, stderr, oracle = [_ref(path, _STDIO) for path in (
        evidence / 'original.stdout', evidence / 'original.stderr', stdout_oracle)]
    digest = read_artifact(stdout, json_document=False)
    _require(digest == read_artifact(oracle, json_document=False) and len(digest) == protocol['output']['success_bytes']
        and re.fullmatch(rb'[0-9a-f]{64}\n', digest) is not None and read_artifact(stderr, json_document=False) == b'',
        'checked successful original stdout differs from exact package diagnostic')
    logs, ids, workers = _processes(trace, paths, position)
    payload = shape['size'] * _SCALE.size
    slots, attempts, first_dispatch_socket = _dispatch(logs, ids['solver'], workers, count, width, window, payload)
    edges, committed, reaped = _pipes(logs, ids, doc['csr_boundary_validation']['payload_bytes'] + Struct(shape['format']).size,
        digest, first_dispatch_socket)
    observed_calls = Counter(request['sequence'] % width for request in attempts)
    _require(expected['issues_per_worker'] == [observed_calls[i] for i in range(width)]
        and expected['scale_bytes'] == count * _SCALE.size and expected['solution_bytes'] == count * payload
        and expected['stdout_bytes'] == len(digest), 'observed protocol work differs from the plan')
    refs, ranges = _source_boundaries(doc, root)
    return dict(schema=SCHEMA, schema_version=1, package=package, source=doc['source'], plan=doc['parameterized_plan'],
        evidence_directory=str(evidence), stdout=stdout, stderr=stderr, original_oracle=oracle, trace=trace,
        count=count, cohort=dict(width=width, window=window, owner_cpu_id=position['owner_cpu_id'],
            worker_cpu_ids=position['worker_cpu_ids'], peak_inflight=min(count, window), observed_pinning='SCHED_SETAFFINITY_SUCCESS'),
        process_ids=ids, worker_slots=slots, preparation_and_publication_edges=edges,
        publication=dict(reap_lines=reaped, external_stdout_lines=committed, exact_oracle_match=True,
            per_iteration_digest_calls='SOURCE_AND_COMPLETE_FIFO_SUCCESS_INFERRED',
            numerical_provider_calls='SOURCE_AND_SUCCESS_STATUS_INFERRED'),
        work={**expected, 'worker_bootstraps':width, 'active_workers':len(observed_calls), 'readonly_csr_preparations':1,
            'request_control_bytes':count*_SEQ.size, 'response_control_bytes':count*_RESPONSE.size,
            'worker_readiness_bytes':width},
        source_members=refs, non_clock_phase_markers=ranges, order_evidence='PROCESS_LOCAL_INCLUSIVE_SYSCALL_LINE_RANGES',
        timing_history=_historical(prior_timings, doc), timestamp_collected=False, performance_measurement=False,
        frontiers=['NO_CROSS_PROCESS_TIMELINE', 'NO_INTERNAL_PROVIDER_OR_DIGEST_TIMESTAMPS', 'NO_CPU_CYCLES_OR_KERNEL_COPY_BYTES',
            'NO_COUNT_OR_WIDTH_PROFITABILITY_TRANSFER', 'NO_COMPLETE_ORIGINAL_PYTHON_EFFECT_CLOSURE'])


def check(path):
    stored = read_artifact(_ref(path, SCHEMA))
    result = project(stored['package']['uri'], stored['evidence_directory'], count=stored['count'],
        stdout_oracle=stored['original_oracle']['uri'], prior_timings=[row['artifact']['uri'] for row in stored['timing_history']])
    _require(_canonical(stored) == _canonical(result), 'stored observation does not replay')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'evidence-directory', 'stdout-oracle', 'out', 'check'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--prior-timing', type=Path, action='append', default=[])
    parser.add_argument('--count', type=int)
    args = parser.parse_args(argv)
    inputs = (args.package, args.evidence_directory, args.stdout_oracle, args.count, args.out)
    if args.check and all(value is None for value in inputs) and not args.prior_timing:
        result = check(args.check)
    elif not args.check and all(value is not None for value in inputs):
        result = project(args.package, args.evidence_directory, count=args.count, stdout_oracle=args.stdout_oracle,
            prior_timings=args.prior_timing)
        with args.out.open('x', encoding='utf8') as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    else:
        parser.error('provide --check or --package, --evidence-directory, --stdout-oracle, --count, and --out')
    print(json.dumps({key: result[key] for key in ('count', 'cohort', 'work')}, sort_keys=True))


if __name__ == '__main__':
    main()
