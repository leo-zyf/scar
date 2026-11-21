"""Observe Linux syscall effects of an admitted original matrix CLI."""

import argparse
import ast
from collections import Counter
import json
import os
from pathlib import Path
import re
from shutil import move
import subprocess

from scar.compiler.nvhpc_source_validation import SCHEMA as PROCESS, _receipt
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.nvhpc-source-effect-observation'
_TRACE_PREFIX = 'syscalls'
_TRACE_LIMIT = 32 * 1024 * 1024
_ROW = re.compile(r'^([a-zA-Z0-9_]+)\((.*)\)\s+= (.+)$')
_FOOTER = re.compile(r'^\+\+\+ exited with ([0-9]+) \+\+\+$')
_FD = re.compile(r'^([0-9]+)(?:<(.+)>)?$')
_FD_FIRST = re.compile(r'^([0-9]+)(?:<(.*?)>)?(?:, |$)')
_QUOTED = re.compile(r'"(?:\\.|[^"\\])*"')
_OPEN_FLAGS = re.compile(r'\bO_(?:WRONLY|RDWR|CREAT|TRUNC|APPEND|TMPFILE)\b')
_EXTERNAL_NETWORK = frozenset(('socket', 'socketpair', 'bind', 'connect', 'listen', 'accept',
    'accept4', 'sendto', 'sendmsg', 'sendmmsg', 'recvfrom', 'recvmsg', 'recvmmsg', 'shutdown',
    'getsockname', 'getpeername', 'getsockopt', 'setsockopt'))
_WRITE = frozenset(('write', 'writev', 'pwrite64', 'pwritev', 'pwritev2'))
_LOCAL_OR_READ = frozenset((
    'access', 'faccessat', 'faccessat2', 'fadvise64', 'fstat', 'lstat', 'stat', 'newfstatat', 'statx',
    'readlink', 'readlinkat', 'getcwd', 'getdents', 'getdents64', 'getrandom', 'read', 'readv', 'pread64',
    'lseek', 'close', 'brk', 'munmap', 'mprotect', 'mremap', 'arch_prctl', 'rseq', 'set_tid_address',
    'set_robust_list', 'getrusage', 'getpid', 'getppid', 'gettid', 'getuid', 'getgid', 'geteuid', 'getegid',
    'uname', 'sched_get_priority_max', 'sched_get_priority_min', 'sched_getaffinity', 'sched_yield',
    'clock_gettime', 'clock_nanosleep', 'nanosleep', 'rt_sigaction', 'rt_sigprocmask', 'rt_sigreturn',
    'futex', 'futex_waitv', 'pipe', 'pipe2', 'epoll_create', 'epoll_create1', 'epoll_ctl', 'epoll_wait',
    'exit', 'exit_group', 'restart_syscall',
))


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _ref(value, schema):
    reference = value if isinstance(value, dict) else artifact_reference(Path(value).resolve(), schema)
    _need(reference['schema'] == schema, 'unexpected effect evidence schema')
    return reference


def _original(reference):
    from scar.analysis.nvhpc_source_entry import SCHEMA as CONTRACT, derive_nvhpc_source_entry
    from scar.compiler.compile_exported_nvhpc import SCHEMA as PACKAGE

    document = read_artifact(reference)
    _need(document['schema_version'] == 1 and document['accepted'] is True and
          document['scope'] == 'PROCESS_LAUNCH_THROUGH_EXIT', 'accepted original source process evidence required')
    project = _ref(document['artifact'], PACKAGE)
    manifest = read_artifact(project)
    root = Path(project['uri']).parent

    def member(name, reference):
        actual = _receipt(root/name)
        _need((actual['path'], actual['sha256'], actual['size_bytes']) ==
              (reference['uri'], reference['sha256'], reference['size_bytes']) and
              {key: actual[key] for key in ('sha256', 'size_bytes')} == manifest['files'][name],
              'source-effect proof differs from the pinned package member')
        return read_artifact(reference)

    contract_ref = _ref(document['source_contract'], CONTRACT)
    contract = member('plan/source-entry.json', contract_ref)
    recipe_ref = artifact_reference(root/'plan/recipe.json', 'scar.aten-native-recipe')
    recipe = member('plan/recipe.json', recipe_ref)
    source = document['source']
    _need(all(source[key] == contract['source'][key] for key in ('path', 'sha256', 'size_bytes'))
          and _receipt(source['path']) == source and derive_nvhpc_source_entry(source['path'], recipe) == contract,
          'diagnostic invocation must retain its exact original source ABI')
    context = document['context']
    cpus = context['cpu_ids']
    _need(cpus and cpus == sorted(set(cpus)) and all(type(cpu) is int and cpu >= 0 for cpu in cpus),
          'declared source process requires a canonical CPU set')
    environment = context['environment']
    _need(type(environment) is dict and all(type(key) is type(value) is str for key, value in environment.items())
          and not {'LD_PRELOAD', 'LD_AUDIT', 'LD_LIBRARY_PATH'} & environment.keys()
          and environment.get('PYTHONDONTWRITEBYTECODE') == '1', 'expected the certified source process environment')
    members = [context['python'], *[value for value in context['providers'].values() if type(value) is dict],
               *document['inputs']]
    _need(all(_receipt(row['path']) == row for row in members), 'original input or interpreter/provider bytes changed')
    inputs = [row['path'] for row in document['inputs']]
    argv = document['argv']
    outputs = argv[len(inputs):]
    _need(argv[:len(inputs)] == inputs and len(argv) == contract['argv_count']
          and len(outputs) == len(contract['outputs']) and len(set(argv)) == len(argv)
          and all(Path(path).is_absolute() and str(Path(path).resolve()) == path for path in argv),
          'original input/output slots differ from the absolute, distinct ABI paths')
    prefix = ['/usr/bin/taskset', '-c', ','.join(map(str, cpus))]
    command = [*prefix, context['python']['path'], '-B', source['path'], *argv]
    _need(context['prefix'] == prefix and document['commands']['ORIGINAL'] == command,
          'original diagnostic command differs from its admitted launch')
    reference_run = next(row for row in document['runs'] if row['mode'] == 'ORIGINAL')
    _need(reference_run['returncode'] == 0 and all(_receipt(row['path']) == row for row in
          [reference_run['stdout'], reference_run['stderr'], *reference_run['outputs']]),
          'original outcome receipts changed or did not exit successfully')
    return dict(command=command, context=context, source=source, inputs=inputs, outputs=outputs,
        expected_sizes=[row['nbytes'] for row in contract['outputs']], reference_run=reference_run)


def _tool_command(tool, scope, folder):
    return [tool['path'], '-ff', '-yy', '-s', '0', '--kill-on-exit', '-o', str(folder/_TRACE_PREFIX),
            '--', *scope['command']]


def _path_argument(arguments):
    match = _QUOTED.search(arguments)
    return None if match is None else os.fsdecode(ast.literal_eval('b' + match[0]))


def _child_fd(arguments):
    match = _FD_FIRST.match(arguments)
    return (None, None) if match is None else (int(match[1]), match[2])


def _event(name, args, result, state):
    inputs, outputs, parent = state['inputs'], state['outputs'], state['is_parent']
    if name in ('open', 'openat'):
        quoted_path = _QUOTED.search(args)
        flag_text = args[quoted_path.end():] if quoted_path else args
        match = _FD.fullmatch(result)
        path = match[2] if match else None
        if not _OPEN_FLAGS.search(flag_text):
            if path in inputs:
                state['seen_inputs'].add(path)
                state['past_first_input'] = True
            if path and path.startswith('/'):
                state['read_paths'].add(path)
            return None, None
        if path in outputs and parent and 'O_CREAT' in flag_text and 'O_TRUNC' in flag_text:
            return 'PUBLICATION_OPEN', path
        if path == '/dev/null<char 1:3>>' or path == '/dev/null<char 1:3>':
            return None, None
        return 'FILE_OPEN_FOR_WRITE', path or _path_argument(args)
    if name == 'close':
        _, path = _child_fd(args)
        return ('PUBLICATION_CLOSE', path) if path in outputs and parent else (None, None)
    if name in _WRITE:
        fd, path = _child_fd(args)
        if name in ('write', 'writev') and path in outputs and parent:
            return 'PUBLICATION_WRITE', path
        if name in ('write', 'writev') and fd in (1, 2) and parent and path and path.startswith('pipe:['):
            return 'STDOUT' if fd == 1 else 'STDERR', None
        return 'OTHER_WRITE', path
    if name in _EXTERNAL_NETWORK:
        return 'NETWORK_OPERATION', _path_argument(args)
    if name in ('execve', 'execveat'):
        path = _path_argument(args)
        if name == 'execve' and parent and len(state['execs']) < 2 and path == state['bootstrap'][len(state['execs'])]:
            state['execs'].append(path)
            return None, None
        return 'OTHER_EXEC', path
    if name in ('fork', 'vfork', 'clone', 'clone3'):
        if name in ('clone', 'clone3') and 'CLONE_THREAD' in args and 'CLONE_VM' in args:
            return None, None
        return 'PROCESS_SPAWN', result
    if name in ('unlink', 'unlinkat', 'rmdir', 'rename', 'renameat', 'renameat2', 'mkdir', 'mkdirat'):
        return 'FILESYSTEM_MUTATION', _path_argument(args)
    if name == 'mmap' and ('MAP_SHARED' not in args or 'PROT_WRITE' not in args or 'MAP_ANONYMOUS' in args):
        return None, None
    if name == 'fcntl' and re.search(r', F_(?:GETFD|GETFL|SETFD)\b', args):
        return None, None
    if name == 'ioctl' and re.search(r', (?:FIOCLEX|TCGETS|TIOCGWINSZ)\b', args):
        return None, None
    if name in ('prlimit64', 'sched_setaffinity') and args.startswith('0,'):
        return None, None
    if name == 'mbind' or name == 'madvise' and args.endswith((', MADV_DONTNEED', ', MADV_HUGEPAGE')):
        return None, None
    if name in ('read', 'readv', 'pread64') and _child_fd(args)[0] == 0 and int(result) > 0:
        return 'STDIN_CONSUMPTION', None
    return ('UNCLASSIFIED_SUCCESS', None) if name not in _LOCAL_OR_READ else (None, None)


def _trace_members(folder):
    paths = sorted(folder.glob(_TRACE_PREFIX+'.*'), key=lambda path: path.name)
    _need(paths and all(re.fullmatch(r'syscalls\.[0-9]+', p.name) and p.is_file() and not p.is_symlink()
                        for p in paths) and sum(p.stat().st_size for p in paths) <= _TRACE_LIMIT,
          'complete bounded per-process strace files required')
    return [_receipt(path) for path in paths]


def _classify(traces, scope):
    bootstrap = [scope['command'][0], scope['context']['python']['path']]
    observed, uncovered, publications = Counter(), [], []
    expected = dict(zip(scope['outputs'], scope['expected_sizes'], strict=True))
    trace_pids = {str(Path(row['path']).suffix)[1:] for row in traces}
    spawn_targets = set()
    observed_inputs, provider_paths = set(), set()
    parent_count, stdio, failed = 0, Counter(), 0
    for index, row in enumerate(traces):
        path = Path(row['path'])
        pid = path.suffix[1:]
        with path.open(encoding='utf-8', errors='surrogateescape') as stream:
            first = stream.readline().strip()
            match = _ROW.fullmatch(first)
            is_parent = bool(match and match[1] == 'execve' and _path_argument(match[2]) == bootstrap[0] and match[3] == '0')
            parent_count += is_parent
            state = dict(inputs=set(scope['inputs']), outputs=expected, is_parent=is_parent, bootstrap=bootstrap,
                         execs=[], seen_inputs=set(), read_paths=set(), past_first_input=False)
            stream.seek(0)
            exits = []
            for line_number, raw in enumerate(stream, 1):
                line = raw.strip()
                if footer := _FOOTER.fullmatch(line):
                    exits.append(int(footer[1]))
                    continue
                if line.startswith('--- SIGCHLD'):
                    continue
                where = dict(trace=index, line=line_number, phase=('AFTER_FIRST_DECLARED_INPUT' if state['past_first_input']
                    else 'BEFORE_FIRST_DECLARED_INPUT') if is_parent else 'FOLLOWED_CHILD_OR_THREAD')
                match = _ROW.fullmatch(line)
                if match is None:
                    uncovered.append(dict(kind='UNPARSED_TRACE_EVENT', **where))
                    continue
                name, args, result = match.groups()
                if result.startswith('-1 '):
                    failed += 1
                    continue
                observed[name] += 1
                if name in ('fork', 'vfork', 'clone', 'clone3') and result.isdecimal():
                    spawn_targets.add(result)
                try:
                    kind, resource = _event(name, args, result, state)
                except (SyntaxError, ValueError):
                    kind, resource = 'UNPARSED_EFFECT_ARGUMENT', None
                if kind in ('PUBLICATION_OPEN', 'PUBLICATION_WRITE', 'PUBLICATION_CLOSE'):
                    count = int(result) if kind == 'PUBLICATION_WRITE' and result.isdecimal() else None
                    publications.append(dict(kind=kind, path=resource, bytes=count, **where))
                elif kind in ('STDOUT', 'STDERR') and result.isdecimal():
                    stdio[kind.lower()] += int(result)
                elif kind is not None:
                    uncovered.append(dict(kind=kind, syscall=name, resource=resource, **where))
            if exits != [0] or is_parent and state['execs'] != bootstrap:
                uncovered.append(dict(kind='INCOMPLETE_OR_FAILED_PROCESS_TRACE', trace=index, exits=exits))
            observed_inputs.update(state['seen_inputs'])
            provider_paths.update(state['read_paths'])
        _need(_receipt(path) == row, 'strace member changed while classifying effects')
    exact_order = []
    for path, size in expected.items():
        exact_order.extend((('PUBLICATION_OPEN', path, None), ('PUBLICATION_WRITE', path, size),
                           ('PUBLICATION_CLOSE', path, None)))
    ordered = [(row['kind'], row['path'], row['bytes']) for row in publications] == exact_order
    if not ordered:
        uncovered.append(dict(kind='PUBLICATION_EVENT_ORDER_OR_COUNT_MISMATCH'))
    if parent_count != 1 or not spawn_targets <= trace_pids:
        uncovered.append(dict(kind='UNRECONCILED_PROCESS_TRACE', parent_traces=parent_count,
                              missing_children=len(spawn_targets-trace_pids)))
    if observed_inputs != set(scope['inputs']):
        uncovered.append(dict(kind='DECLARED_INPUT_OPEN_NOT_OBSERVED', paths=sorted(set(scope['inputs'])-observed_inputs)))
    providers = {}
    for name, row in scope['context']['providers'].items():
        if type(row) is not dict:
            continue
        path = Path(row['path'])
        cache = str(path.parent/'__pycache__')+'/' + path.stem + '.cpython-'
        matches = sorted(item for item in provider_paths if item == str(path)
            or path.suffix == '.py' and item.startswith(cache) and item.endswith('.pyc'))
        providers[name] = matches
        if not matches:
            uncovered.append(dict(kind='DECLARED_PROVIDER_READ_NOT_OBSERVED', provider=name))
    return dict(syscall_counts=dict(observed), failed_syscall_count=failed, publications=publications,
        stdio_write_bytes={name: stdio[name] for name in ('stdout', 'stderr')}, observed_inputs=sorted(observed_inputs),
        observed_provider_paths=providers, uncovered_effects=uncovered)


def _matches_digest(actual, expected):
    return actual is not None and all(actual[key] == expected[key] for key in ('sha256', 'size_bytes'))


def _outcome(folder, scope, returncode):
    stdout, stderr = (_receipt(folder/name) for name in ('stdout', 'stderr'))
    files = [folder/f'publication-{index}' for index in range(len(scope['outputs']))]
    publications = [_receipt(path) if path.is_file() and not path.is_symlink() else None for path in files]
    expected = scope['reference_run']
    same = (returncode == 0 and _matches_digest(stdout, expected['stdout']) and _matches_digest(stderr, expected['stderr'])
            and all(_matches_digest(actual, original) for actual, original in zip(publications, expected['outputs'], strict=True)))
    return dict(returncode=returncode, stdout=stdout, stderr=stderr, publications=publications,
        original_outcome_reproduced=same)


def _report(folder, process, scope, tool, returncode):
    traces = _trace_members(folder)
    projection = _classify(traces, scope)
    outcome = _outcome(folder, scope, returncode)
    consistent_stdio = all(projection['stdio_write_bytes'][name] == outcome[name]['size_bytes']
                           for name in ('stdout', 'stderr'))
    status = ('DIAGNOSTIC_OUTCOME_MISMATCH' if not outcome['original_outcome_reproduced'] or not consistent_stdio
              else 'OBSERVED_EXTRA_EFFECTS' if projection['uncovered_effects'] else 'OBSERVED_DECLARED_EFFECTS')
    return dict(schema=SCHEMA, schema_version=1, original_process=process, scope=scope,
        strace=tool, diagnostic_command=_tool_command(tool, scope, folder), traces=traces, outcome=outcome,
        projection=projection, status=status, diagnostic_only=True, performance_measured=False,
        global_effect_closure=False, deployment_authorized=False)


def observe_nvhpc_source_effects(process, out, *, strace='/usr/bin/strace'):
    """Run one separate diagnostic observation on the exact original argv."""
    process = _ref(process, PROCESS)
    scope = _original(process)
    cwd = Path(scope['context']['working_directory'])
    _need(cwd.is_dir() and cwd.resolve() == cwd and set(scope['context']['cpu_ids']) <= os.sched_getaffinity(0),
          'original diagnostic working directory or CPU scope is unavailable')
    outputs = tuple(map(Path, scope['outputs']))
    _need(all(path.parent.is_dir() and not path.exists() and not path.is_symlink() for path in outputs),
          'diagnostic needs all original publication paths fresh and exclusively available')
    folder = Path(out).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    tool = _receipt(strace)
    run = subprocess.run(_tool_command(tool, scope, folder), env=scope['context']['environment'], cwd=cwd,
                         capture_output=True, timeout=120)
    for name in ('stdout', 'stderr'):
        (folder/name).write_bytes(getattr(run, name))
    for index, path in enumerate(outputs):
        if path.is_file() and not path.is_symlink():
            move(path, folder/f'publication-{index}')
    _need(_receipt(tool['path']) == tool and _original(process) == scope, 'original evidence changed during diagnostic')
    report = _report(folder, process, scope, tool, run.returncode)
    (folder/'result.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    return report


def verify_nvhpc_source_effects(reference, *, process):
    """Replay retained effect and output records without running the target."""
    reference = _ref(reference, SCHEMA)
    process = _ref(process, PROCESS)
    saved = read_artifact(reference)
    scope = _original(process)
    tool = saved['strace']
    _need(_receipt(tool['path']) == tool, 'strace diagnostic tool identity changed')
    expected = _report(Path(reference['uri']).parent, process, scope, tool, saved['outcome']['returncode'])
    _need(saved == expected, 'retained source-effect observation differs from byte replay')
    read_artifact(reference, json_document=False)
    return expected


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar observe-nvhpc-source-effects', description=__doc__)
    parser.add_argument('--process', required=True, type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--out', type=Path)
    action.add_argument('--check', type=Path)
    parser.add_argument('--strace', default='/usr/bin/strace')
    args = parser.parse_args(argv)
    if args.check is None:
        report = observe_nvhpc_source_effects(args.process, args.out, strace=args.strace)
        path = args.out/'result.json'
    else:
        report = verify_nvhpc_source_effects(args.check, process=args.process)
        path = args.check
    print(json.dumps(dict(status=report['status'], result=str(path.resolve()),
        original_outcome_reproduced=report['outcome']['original_outcome_reproduced'],
        uncovered_effects=len(report['projection']['uncovered_effects']),
        performance_measured=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
