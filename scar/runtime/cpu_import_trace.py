"""Decode per-thread Linux strace evidence for CPython import diagnostics."""

from collections import Counter, defaultdict, deque
from os import fsdecode
import re


_TIME = re.compile(rb'^(\d+)\.(\d{6}) (.+)$')
_UNFINISHED = re.compile(rb'^([a-z0-9_]+)\((.*) <unfinished \.\.\.>$')
_RESUMED = re.compile(rb'^<\.\.\. ([a-z0-9_]+) resumed>(.*)$')
_WRITE = re.compile(rb'^writev?\(0x2, .+\)\s+=\s+(0x[0-9a-f]+|0)$')
_EXEC = re.compile(rb'^execve\("((?:\\x[0-9a-f]{2})*)", \[(.*)\], .*\)\s+=\s+0$')
_ARGS = re.compile(rb'"((?:\\x[0-9a-f]{2})*)"')
_BIRTH = re.compile(rb'^(clone3?|fork|vfork)\((.*)\)\s+=\s+([1-9][0-9]*)$')
_EXIT = re.compile(rb'^\+\+\+ exited with (\d+) \+\+\+$')
_HEADER = b'import time: self [us] | cumulative | imported package\n'
_IMPORT = re.compile(rb'^import time: *(cached|-?[0-9]+) *\| *(cached|-?[0-9]+) *\|( +)([^\r\n]+)\n$')


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU import trace: ' + detail)


def _hex_string(raw):
    return fsdecode(bytes.fromhex(raw.replace(b'\\x', b'').decode('ascii')))


def _exec_args(body):
    item = _EXEC.fullmatch(body)
    if item is None:
        return None, None
    argv = _ARGS.findall(item[2])
    if b', '.join(b'"' + arg + b'"' for arg in argv) != item[2]:
        return _hex_string(item[1]), None
    return _hex_string(item[1]), [_hex_string(arg) for arg in argv]


def _clock(match):
    return int(match[1]) * 1_000_000 + int(match[2])


def _task(raw):
    writes, births, execs, exits = [], [], [], []
    pending, write = None, None
    for line, row in enumerate(raw.splitlines()):
        if row.startswith(b' |'):
            _need(write is not None and len(row) == 78 and row[9:10] == b' ', 'unexpected raw stderr dump row')
            write['data'].extend(bytes.fromhex(row[10:58].decode('ascii')))
            continue
        if row.startswith(b' * '):  # writev dumps each iovec, in order.
            _need(write is not None, 'orphaned stderr iovec')
            continue
        if not row:
            continue
        stamp = _TIME.fullmatch(row)
        _need(stamp is not None, 'trace row lacks the fixed strace realtime format')
        time, body = _clock(stamp), stamp[3]
        write = None
        if unfinished := _UNFINISHED.fullmatch(body):
            _need(pending is None, 'a task has overlapping unfinished syscalls')
            pending = (unfinished[1], unfinished[2], time, line)
            continue
        if resumed := _RESUMED.fullmatch(body):
            _need(pending is not None and pending[0] == resumed[1], 'resumed syscall lacks its same-TID prefix')
            name, partial, time, line = pending
            body = name + b'(' + partial + resumed[2]
            pending = None
        if written := _WRITE.fullmatch(body):
            write = dict(timestamp_us=time, line=line, size=int(written[1], 16), data=bytearray(), exec_index=len(execs)-1)
            writes.append(write)
        elif birth := _BIRTH.fullmatch(body):
            kind = 'LINUX_PROCESS_CLONE'
            if birth[1].startswith(b'clone'):
                flags = re.search(rb'\bflags=([^,}]+)', birth[2])
                kind = ('UNRESOLVED_CLONE_KIND' if flags is None else
                        'LINUX_THREAD_CLONE' if b'CLONE_THREAD' in flags[1].split(b'|') else 'LINUX_PROCESS_CLONE')
            births.append((int(birth[3]), dict(kind=kind, timestamp_us=time, line=line)))
        elif body.startswith((b'execve(', b'execveat(')) and body.endswith(b'= 0'):
            executable, argv = _exec_args(body)
            execs.append(dict(index=len(execs), executable=executable, argv=argv, timestamp_us=time, line=line))
        elif exit_ := _EXIT.fullmatch(body):
            exits.append(int(exit_[1]))
    _need(pending is None and len(exits) <= 1, 'trace ended with an unresolved syscall or task reuse')
    _need(all(len(row['data']) == row['size'] for row in writes), 'missing or truncated stderr syscall bytes')
    return writes, births, execs, exits[0] if exits else None


def _records(writes):
    imported, lines = [], Counter()
    for write in writes:
        # A line is attributed only when all of its bytes are in one write by
        # the same task. Shared-stderr fragments cannot manufacture ownership.
        raw = bytes(write['data'])
        _need(not raw or raw.endswith(b'\n'), 'stderr write is not a complete CPython import row')
        for line in raw.splitlines(keepends=True):
            lines[line] += 1
            if line == _HEADER:
                continue
            item = _IMPORT.fullmatch(line)
            _need(item is not None and (item[1] == b'cached') == (item[2] == b'cached'),
                  'stderr contains a non-import or incomplete CPython diagnostic row')
            cached = item[1] == b'cached'
            # CPython right-aligns the cached module to a minimum field width;
            # long names erase depth information. Keep its actual padding.
            indent = len(item[3]) - 1
            imported.append(dict(index=len(imported), timestamp_us=write['timestamp_us'], line=write['line'],
                exec_index=None if write['exec_index'] < 0 else write['exec_index'],
                depth=None if cached or indent % 2 else indent//2, indent_spaces=indent,
                self_us=None if cached else int(item[1]), cumulative_us=None if cached else int(item[2]),
                module=fsdecode(item[4]), cached=cached))
    return imported, lines


def decode(traces, stderr, launcher_argv, interpreter_argv):
    """Recover exact TID ownership and conservative task birth lineage."""
    tasks = {tid: _task(raw) for tid, raw in sorted(traces.items())}
    roots = [tid for tid, (_, _, execs, _) in tasks.items()
             if any(event['argv'] == launcher_argv and event['executable'] == launcher_argv[0] for event in execs)]
    candidates = [tid for tid, (_, _, execs, _) in tasks.items()
                  if any(event['argv'] == interpreter_argv and event['executable'] == interpreter_argv[0] for event in execs)]
    _need(len(roots) == len(candidates) == 1, 'expected exact launcher and original CPython exec evidence')
    root, = roots
    interpreter, = candidates
    edges = defaultdict(list)
    parents = {root: dict(parent_tid=None, kind='TRACED_LAUNCHER_EXEC',
                         timestamp_us=tasks[root][2][0]['timestamp_us'], line=tasks[root][2][0]['line'])}
    for tid, (_, births, _, _) in tasks.items():
        for child, birth in births:
            _need(child in tasks and child not in parents and child != tid, 'missing or ambiguous cloned task evidence')
            parents[child] = dict(birth, parent_tid=tid)
            edges[tid].append(child)
    groups = {root: root}
    ready = deque([root])
    while ready:
        tid = ready.popleft()
        for child in edges[tid]:
            kind = parents[child]['kind']
            groups[child] = groups[tid] if kind == 'LINUX_THREAD_CLONE' else child if kind == 'LINUX_PROCESS_CLONE' else None
            ready.append(child)
    _need(groups.get(interpreter) == interpreter, 'CPython exec does not belong to a verified process leader')
    results, observed = [], Counter()
    for tid, (writes, _, execs, code) in tasks.items():
        records, emitted = _records(writes)
        observed.update(emitted)
        results.append(dict(tid=tid, tgid=groups.get(tid), birth=parents.get(tid), execs=execs,
                            exit_status=code, records=records))
    _need(observed[_HEADER] > 0 and observed == Counter(stderr.splitlines(keepends=True))
          and any(row['records'] for row in results), 'per-TID writes do not cover inherited stderr exactly')
    return results, root, interpreter
