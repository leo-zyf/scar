"""Observe Linux syscall effects of an admitted original matrix CLI."""

import argparse
import ast
from collections import Counter, defaultdict
from itertools import count
import json
from math import prod
import os
from pathlib import Path
import re
from shutil import move
import subprocess
import sys
from types import CodeType

from scar.compiler.nvhpc_source_validation import SCHEMA as PROCESS, _receipt
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.nvhpc-source-effect-observation'
TENSOR_FLOW = 'scar.nvhpc-source-tensor-flow-observation'
_TENSOR_CAPTURE = 'scar.nvhpc-source-dispatch-capture'
_LATEST_VERSION = 2
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
    scope = dict(command=command, context=context, source=source, inputs=inputs, outputs=outputs,
        expected_sizes=[row['nbytes'] for row in contract['outputs']], reference_run=reference_run)
    return scope, contract_ref, contract


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


class _Witnesses:
    """Keep the source-accountable observations with exact trace coordinates."""

    def __init__(self, scope):
        self.source = scope['source']['path']
        self.inputs = set(scope['inputs'])
        self.provider_paths = {name: Path(row['path']) for name, row in scope['context']['providers'].items()
            if type(row) is dict}
        self.bootstrap = []
        self.exit = []
        self.source_reads = []
        self.files = defaultdict(list)
        self.providers = defaultdict(list)
        self.stdio = defaultdict(list)

    def record(self, name, args, result, kind, where):
        if where['phase'] == 'FOLLOWED_CHILD_OR_THREAD':
            return
        if name == 'execve' and kind is None:
            self.bootstrap.append(dict(kind='BOOTSTRAP_EXEC', path=_path_argument(args), **where))
        if name in ('open', 'openat') and kind is None and (descriptor := _FD.fullmatch(result)):
            path = descriptor[2]
            if path == self.source:
                self.source_reads.append(dict(kind='SOURCE_OPEN', path=path, **where))
            if path in self.inputs:
                self.files[path].append(dict(kind='READ_ONLY_OPEN', fd=int(descriptor[1]), **where))
            if path is not None and where['phase'] == 'BEFORE_FIRST_DECLARED_INPUT':
                for provider, candidate in self.provider_paths.items():
                    cache = str(candidate.parent/'__pycache__') + '/' + candidate.stem + '.cpython-'
                    if path == str(candidate) or candidate.suffix == '.py' and path.startswith(cache) and path.endswith('.pyc'):
                        self.providers[provider].append(dict(kind='PROVIDER_PATH_OPEN', path=path, **where))
        if name == 'mmap':
            fields = args.split(', ', 4)
            if len(fields) == 5:
                descriptor, _, offset = fields[-1].rpartition(', ')
                match = _FD.fullmatch(descriptor)
                if match and match[2] in self.inputs and fields[1].isdecimal() and offset.isdecimal():
                    flags = fields[3].split('|')
                    private = ('MAP_PRIVATE' in flags and 'MAP_ANONYMOUS' not in flags
                               and 'PROT_READ' in fields[2].split('|'))
                    self.files[match[2]].append(dict(kind='PRIVATE_MAP' if private
                        else 'OTHER_MAP', fd=int(match[1]), length=int(fields[1]), offset=int(offset), **where))
        if name == 'close':
            fd, path = _child_fd(args)
            if path in self.inputs:
                self.files[path].append(dict(kind='CLOSE', fd=fd, **where))
        if kind in ('STDOUT', 'STDERR') and result.isdecimal():
            self.stdio[kind.lower()].append(dict(kind='STREAM_WRITE', syscall=name, bytes=int(result), **where))


def _classify(traces, scope, *, witnesses=None):
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
                    if witnesses is not None and is_parent:
                        witnesses.exit.append(dict(kind='PROCESS_EXIT', trace=index, line=line_number,
                            returncode=exits[-1]))
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
                    if witnesses is not None:
                        witnesses.record(name, args, result, kind, where)
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


def _input_status(events, nbytes):
    if [row['kind'] for row in events] != ['READ_ONLY_OPEN', 'PRIVATE_MAP', 'CLOSE']:
        return 'UNOBSERVED_OR_INCOMPLETE_MAPPING'
    opened, mapped, closed = events
    if ((opened['trace'], opened['fd']) == (mapped['trace'], mapped['fd']) == (closed['trace'], closed['fd'])
            and (mapped['length'], mapped['offset']) == (nbytes, 0)):
        return 'PRIVATE_FILE_MAPPING_OBSERVED'
    return 'UNOBSERVED_OR_INCOMPLETE_MAPPING'


def _coverage(contract_ref, contract, scope, projection, outcome, witnesses):
    inputs = dict(zip((row['argv'] for row in contract['inputs']), scope['inputs'], strict=True))
    outputs = dict(zip((row['argv'] for row in contract['outputs']), scope['outputs'], strict=True))
    obligations = []
    for index, node in enumerate(contract['graph']['nodes']):
        kind = node['kind']
        status, events, resource = 'UNOBSERVED_STATIC_BOUNDARY', [], None
        if kind == 'PROCESS_LAUNCH':
            events = [*witnesses.bootstrap, *witnesses.source_reads]
            status = ('BOOTSTRAP_AND_SOURCE_OPEN_OBSERVED' if len(witnesses.bootstrap) == 2 and witnesses.source_reads
                      else 'UNOBSERVED_OR_INCOMPLETE_BOOTSTRAP')
        elif kind == 'IMPORT_INITIALIZATION':
            resource = node['module']
            events = witnesses.providers.get(resource, [])
            status = 'PARTIAL_PROVIDER_PATH_OBSERVATION' if events else status
        elif kind == 'PRIVATE_FILE_MAPPING_AND_TENSOR_VIEW':
            resource = inputs[node['argv']]
            events = witnesses.files.get(resource, [])
            status = _input_status(events, node['nbytes'])
        elif kind == 'SYNC_FILE_CONTENT_PUBLICATION':
            resource = outputs[node['argv']]
            events = [event for event in projection['publications'] if event['path'] == resource]
            exact = [(row['kind'], row['bytes']) for row in events] == [
                ('PUBLICATION_OPEN', None), ('PUBLICATION_WRITE', node['nbytes']), ('PUBLICATION_CLOSE', None)]
            retained = outcome['publications'][node['argv']-len(inputs)-1]
            expected = scope['reference_run']['outputs'][node['argv']-len(inputs)-1]
            status = ('FILE_CONTENT_PUBLICATION_OBSERVED' if exact and _matches_digest(retained, expected)
                      else 'UNOBSERVED_OR_INCOMPLETE_PUBLICATION')
        elif kind == 'NORMAL_PROCESS_EXIT':
            events = witnesses.exit
            status = ('NORMAL_EXIT_OBSERVED' if len(events) == 1 and events[0]['returncode'] == outcome['returncode'] == 0
                      else 'UNOBSERVED_OR_INCOMPLETE_EXIT')
        obligations.append(dict(id='source:'+node['id'], origin='STATIC_SOURCE', kind=kind, status=status,
            source_pointer=f'#/graph/nodes/{index}', resource=resource, witnesses=events))
    for name in ('stdout', 'stderr'):
        events = witnesses.stdio.get(name, [])
        size = outcome[name]['size_bytes']
        status = ('EMPTY_CAPTURE_OBSERVED' if size == 0 and not events
                  else 'STREAM_WRITE_OBSERVED' if events and sum(row['bytes'] for row in events) == size
                  else 'UNOBSERVED_OR_INCOMPLETE_STREAM')
        obligations.append(dict(id='source:'+name, origin='STATIC_SOURCE', kind='PROCESS_STREAM_CAPTURE', status=status,
            source_pointer='#/domain/'+name, resource=name, witnesses=events))
    direct_imports = {row['module'] for row in contract['imports']}
    for provider, events in witnesses.providers.items():
        if provider not in direct_imports:
            obligations.append(dict(id='provider:'+provider, origin='OBSERVED_PROVIDER', kind='PROVIDER_READ',
                status='PATH_OBSERVATION_WITHOUT_SOURCE_CAUSAL_BINDING', source_pointer=None,
                resource=provider, witnesses=events))
    for index, effect in enumerate(projection['uncovered_effects']):
        status = ('OBSERVED_UNREPRESENTED_EFFECT' if 'syscall' in effect and effect['kind'] != 'UNPARSED_EFFECT_ARGUMENT'
                  else 'UNRESOLVED_OBSERVATION_GAP')
        obligations.append(dict(id='observation:'+str(index), origin='OBSERVATION_FRONTIER', kind=effect['kind'],
            status=status, source_pointer=None, resource=effect.get('resource'),
            observation_pointer=f'#/projection/uncovered_effects/{index}', witnesses=[effect] if 'line' in effect else []))
    unconfirmed_boundary = {'INCOMPLETE_OR_FAILED_PROCESS_TRACE', 'PUBLICATION_EVENT_ORDER_OR_COUNT_MISMATCH',
                            'UNRECONCILED_PROCESS_TRACE', 'DECLARED_INPUT_OPEN_NOT_OBSERVED'}
    logical_boundary_witnessed = (outcome['original_outcome_reproduced'] and not any(
        row['status'].startswith('UNOBSERVED_OR_INCOMPLETE') or row['kind'] in unconfirmed_boundary for row in obligations))
    return dict(source_contract=contract_ref, obligations=obligations,
        logical_boundary_witnessed=logical_boundary_witnessed,
        static_boundary_effects_closed=False, unobserved_source_nodes_preserved=True,
        temporal_phase_is_source_causal_binding=False, memory_payload_reads_or_bandwidth_proven=False,
        scope='PINNED_SINGLE_DIAGNOSTIC_SOURCE_ABI_AND_RETAINED_SYSCALL_WITNESSES')


def _report(folder, process, original, tool, returncode, *, version):
    scope, contract_ref, contract = original
    traces = _trace_members(folder)
    witnesses = _Witnesses(scope) if version == 2 else None
    projection = _classify(traces, scope, witnesses=witnesses)
    outcome = _outcome(folder, scope, returncode)
    consistent_stdio = all(projection['stdio_write_bytes'][name] == outcome[name]['size_bytes']
                           for name in ('stdout', 'stderr'))
    status = ('DIAGNOSTIC_OUTCOME_MISMATCH' if not outcome['original_outcome_reproduced'] or not consistent_stdio
              else 'OBSERVED_EXTRA_EFFECTS' if projection['uncovered_effects'] else 'OBSERVED_DECLARED_EFFECTS')
    report = dict(schema=SCHEMA, schema_version=version, original_process=process, scope=scope,
        strace=tool, diagnostic_command=_tool_command(tool, scope, folder), traces=traces, outcome=outcome,
        projection=projection, status=status, diagnostic_only=True, performance_measured=False,
        global_effect_closure=False, deployment_authorized=False)
    if witnesses is not None:
        coverage = report['effect_coverage'] = _coverage(contract_ref, contract, scope, projection, outcome, witnesses)
        if status == 'OBSERVED_DECLARED_EFFECTS' and not coverage['logical_boundary_witnessed']:
            report['status'] = 'OBSERVATION_COVERAGE_OPEN'
    return report


def observe_nvhpc_source_effects(process, out, *, strace='/usr/bin/strace'):
    """Run one separate diagnostic observation on the exact original argv."""
    process = _ref(process, PROCESS)
    original = _original(process)
    scope = original[0]
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
    _need(_receipt(tool['path']) == tool and _original(process) == original, 'original evidence changed during diagnostic')
    report = _report(folder, process, original, tool, run.returncode, version=_LATEST_VERSION)
    (folder/'result.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    return report


def verify_nvhpc_source_effects(reference, *, process):
    """Replay retained effect and output records without running the target."""
    reference = _ref(reference, SCHEMA)
    process = _ref(process, PROCESS)
    saved = read_artifact(reference)
    original = _original(process)
    version = saved.get('schema_version')
    _need(type(version) is int and version in (1, _LATEST_VERSION), 'unsupported retained source-effect observation version')
    tool = saved['strace']
    _need(_receipt(tool['path']) == tool, 'strace diagnostic tool identity changed')
    expected = _report(Path(reference['uri']).parent, process, original, tool, saved['outcome']['returncode'], version=version)
    _need(saved == expected, 'retained source-effect observation differs from byte replay')
    read_artifact(reference, json_document=False)
    return expected


_TENSOR_SCHEMAS = {
    'from_file': ('aten::from_file(str filename, bool? shared=None, int? size=0, *, ScalarType? dtype=None, '
                  'Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor'),
    'view': 'aten::view(Tensor(a) self, SymInt[] size) -> Tensor(a)',
    'mm': 'aten::mm(Tensor self, Tensor mat2) -> Tensor',
    'detach': 'aten::detach(Tensor(a) self) -> Tensor(a)',
}
_DISPATCH_CHILD = ('import sys; from scar.analysis.nvhpc_source_effects import _record_tensor_flow; '
                   '_record_tensor_flow(*sys.argv[1:])')


def _entry_code(contract):
    from scar.trace.code_snapshot import code_key

    source = contract['source']
    root = compile(Path(source['path']).read_bytes(), source['path'], 'exec', dont_inherit=True)
    body, = (item for item in root.co_consts if type(item) is CodeType and item.co_qualname == source['function'])
    _need(code_key(body) == source['function_code_key'], 'tensor diagnostic needs its exact original CodeID')
    return body


def _record_tensor_flow(process, output):
    """Subprocess observer. Arm dispatch after the original import phase."""
    from threading import get_ident
    from weakref import ref

    from scar.runtime.entry_runner import run_original_script
    from scar.trace.code_snapshot import code_key, site

    scope, contract_ref, contract = _original(_ref(process, PROCESS))
    target = _entry_code(contract)
    callbacks, activation, objects, storage = [], {}, {}, {}
    object_ordinals = count(1)
    scope_owner = get_ident()
    max_calls = 2 * len(contract['inputs']) + len(contract['projection']['operations']) + len(contract['outputs'])
    _need(max_calls <= 256 and 'torch' not in sys.modules, 'bounded diagnostic needs the original Torch import')
    monitor = sys.monitoring
    events = monitor.events
    slot = next(number for number in range(5, -1, -1) if monitor.get_tool(number) is None)
    mode = frame = selected = None

    def attach(event, args):
        nonlocal selected
        if event == 'exec' and args and type(args[0]) is CodeType and args[0].co_filename == target.co_filename:
            selected, = (code for code in args[0].co_consts if type(code) is CodeType
                         and code.co_qualname == target.co_qualname and code_key(code) == code_key(target))
            monitor.set_local_events(slot, selected, events.PY_START | events.PY_RETURN)

    def begin(code, offset):
        nonlocal mode, frame
        _need(not activation and get_ident() == scope_owner, 'tensor diagnostic source invoked repeatedly or on a new thread')
        frame = sys._getframe(1)
        torch = sys.modules['torch']
        from torch.multiprocessing.reductions import StorageWeakRef
        from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode

        provider = dict(torch=_receipt(torch.__file__), torch_native=_receipt(torch._C.__file__),
                        torch_version=str(torch.__version__))
        _need(all(scope['context']['providers'][name] == value for name, value in provider.items())
              and _get_current_dispatch_mode() is None, 'unexpected actual Tensor provider or dispatch mode')
        activation.update(code=code_key(code), entered=site(code, offset), provider=provider)
        operations = {getattr(torch.ops.aten, name).default: name for name in _TENSOR_SCHEMAS}

        def metadata(value):
            _need(type(value) is torch.Tensor and value.dtype is torch.float32 and value.device.type == 'cpu'
                  and value.layout is torch.strided, 'tensor diagnostic requires a plain CPU FP32 Tensor')
            identity = id(value)
            if identity not in objects or objects[identity][0]() is not value:
                objects[identity] = ref(value), next(object_ordinals)
            backing = value.untyped_storage()
            owner = StorageWeakRef(backing)
            if owner not in storage:
                storage[owner] = len(storage)+1
            return dict(object=objects[identity][1], storage=storage[owner], shape=list(value.shape),
                        strides=list(value.stride()), offset=value.storage_offset(), storage_nbytes=backing.nbytes(),
                        dtype='FLOAT32', device='CPU')

        class Observe(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = {} if kwargs is None else kwargs
                _need(frame is not None and get_ident() == scope_owner and len(callbacks) < max_calls,
                      'Tensor dispatch escaped its bounded invocation')
                caller = sys._getframe(1)
                while caller is not None and caller is not frame:
                    caller = caller.f_back
                _need(caller is frame, 'Tensor dispatch has no selected source caller')
                name = operations.get(func)
                _need(name is not None and str(func._schema) == _TENSOR_SCHEMAS[name],
                      'Tensor dispatch selected an unsupported callable or schema')
                position = site(frame.f_code, frame.f_lasti)
                incoming = [metadata(item) for item in args if type(item) is torch.Tensor]
                result = func(*args, **kwargs)
                callback = dict(operator=name, schema=str(func._schema), site=position, operands=incoming,
                                result=metadata(result))
                if name == 'from_file':
                    callback['filename'] = args[0]
                    callback['shared'] = args[1] if len(args) > 1 else kwargs.get('shared')
                callbacks.append(callback)
                return result

        mode = Observe()
        mode.__enter__()

    def finish(code, offset, returned):
        nonlocal mode, frame
        _need(sys._getframe(1) is frame and mode is not None and returned is None,
              'unexpected source activation return during Tensor capture')
        mode.__exit__(None, None, None)
        mode = frame = None
        activation['returned'] = site(code, offset)

    monitor.use_tool_id(slot, _TENSOR_CAPTURE)
    monitor.register_callback(slot, events.PY_START, begin)
    monitor.register_callback(slot, events.PY_RETURN, finish)
    sys.addaudithook(attach)
    try:
        run_original_script(scope['source']['path'], [*scope['inputs'], *scope['outputs']])
    finally:
        if mode is not None:
            mode.__exit__(None, None, None)
        if selected is not None:
            monitor.set_local_events(slot, selected, 0)
        monitor.free_tool_id(slot)
    _need('returned' in activation and len(callbacks) == max_calls, 'incomplete original source Tensor observation')
    Path(output).write_text(json.dumps(dict(schema=_TENSOR_CAPTURE, schema_version=1, source_contract=contract_ref,
        activation=activation, dispatches=callbacks), sort_keys=True, allow_nan=False)+'\n')


def _view_metadata(row, shape):
    expected_stride = [prod(shape[index+1:]) for index in range(len(shape))]
    _need(row['shape'] == shape and row['strides'] == expected_stride and row['offset'] == 0
          and row['storage_nbytes'] == 4 * prod(shape) and row['dtype'] == 'FLOAT32' and row['device'] == 'CPU'
          and all(type(row[name]) is int and row[name] > 0 for name in ('object', 'storage')),
          'observed Tensor geometry differs from the source value')


def _project_tensor_flow(raw, original):
    from scar.trace.code_snapshot import site

    scope, contract_ref, contract = original
    code = _entry_code(contract)
    activation = raw['activation']
    _need(raw['schema_version'] == 1 and raw['source_contract'] == contract_ref and
          activation['code'] == contract['source']['function_code_key'] and
          all(scope['context']['providers'][name] == activation['provider'][name]
              for name in ('torch', 'torch_native', 'torch_version')), 'dispatch capture differs from original source/provider scope')
    for name, opcodes in (('entered', ('RESUME',)), ('returned', ('RETURN_VALUE', 'RETURN_CONST'))):
        row = activation[name]
        _need(site(code, row['offset']) == row and row['opname'] in opcodes, 'invalid selected source activation event')
    rows = iter(enumerate(raw['dispatches']))
    links, source_routes = [], []
    last_offset = -1

    def pull(name, node, *, entire=False):
        nonlocal last_offset
        index, row = next(rows, (None, None))
        _need(row is not None and row['operator'] == name and row['schema'] == _TENSOR_SCHEMAS[name],
              'dispatch order does not cover the original source effect graph')
        actual = row['site']
        _need(site(code, actual['offset']) == actual and actual['opname'] in ('CALL', 'CALL_KW')
              and actual['position'] is not None and actual['offset'] > last_offset,
              'dispatch lacks an exact ordered original CALL position')
        last_offset = actual['offset']
        start, end, col, end_col = actual['position']
        line0, col0, line1, col1 = node['source_span']
        matches = [start, col, end, end_col] == node['source_span'] if entire else (
            (line0, col0) <= (start, col) <= (end, end_col) <= (line1, col1))
        _need(matches, 'dispatch CALL falls outside its admitted source node')
        return index, row

    producer = {}
    for index, node in enumerate(contract['graph']['nodes']):
        identity, kind = node['id'], node['kind']
        row = dict(source_pointer='#/graph/nodes/'+str(index), source=identity)
        if kind == 'PRIVATE_FILE_MAPPING_AND_TENSOR_VIEW':
            load_id, load = pull('from_file', node)
            view_id, view = pull('view', node, entire=True)
            _need(load['filename'] == scope['inputs'][node['argv']-1] and load['shared'] is False
                  and load['operands'] == [] and view['operands'] == [load['result']]
                  and view['result']['storage'] == load['result']['storage'], 'original file Tensor/view alias boundary differs')
            _view_metadata(load['result'], [prod(node['shape'])])
            _view_metadata(view['result'], node['shape'])
            producer[identity] = view['result']
            source_routes.append(dict(row, dispatches=[load_id, view_id], relation='SAME_WEAK_STORAGE_LIFETIME'))
        elif kind in ('FUNCTIONAL_MM', 'SYNC_FILE_CONTENT_PUBLICATION'):
            publication = kind == 'SYNC_FILE_CONTENT_PUBLICATION'
            dispatch_id, event = pull('detach' if publication else 'mm', node, entire=not publication)
            incoming = sorted((edge for edge in contract['graph']['edges'] if edge['target'] == identity
                               and edge['kind'] == 'DATA'), key=lambda edge: edge.get('operand', 0))
            args = [producer[edge['source']] for edge in incoming]
            _need(event['operands'] == args, 'actual dispatched Tensor operands do not follow original source data edges')
            shape = node['shape'] if publication else [args[0]['shape'][0], args[1]['shape'][1]]
            _view_metadata(event['result'], shape)
            if publication:
                _need(event['result']['storage'] == args[0]['storage'], 'publication does not alias its actual input value')
            else:
                _need(event['result']['storage'] not in {value['storage'] for value in producer.values()},
                      'matrix result unexpectedly aliases an existing source value')
                producer[identity] = event['result']
            source_routes.append(dict(row, dispatches=[dispatch_id], relation='PUBLICATION_ALIAS' if publication else 'ATEN_MM'))
            links.extend(dict(source=edge['source'], target=identity, operand=position,
                              observation=dispatch_id, object=args[position]['object'])
                         for position, edge in enumerate(incoming))
    _need(next(rows, None) is None, 'additional Tensor dispatch outside the admitted source graph')
    return dict(source_nodes=source_routes, observed_data_dependencies=links)


def _tensor_flow_report(folder, process, original, returncode):
    scope, contract, _ = original
    capture = artifact_reference(folder/'dispatch.json', _TENSOR_CAPTURE)
    projection = _project_tensor_flow(read_artifact(capture), original)
    outcome = _outcome(folder, scope, returncode)
    _need(outcome['original_outcome_reproduced'], 'instrumented Tensor flow did not reproduce original files and streams')
    return dict(schema=TENSOR_FLOW, schema_version=1, process=process, source_contract=contract,
        capture=capture, projection=projection, outcome=outcome,
        scope='SINGLE_INSTRUMENTED_CPU_SOURCE_ACTIVATION', original_clean_timing_measured=False,
        cross_run_identity_established=False, global_effect_closure=False, deployment_authorized=False)


def observe_nvhpc_tensor_flow(process, out):
    """Observe the exact original CPU Tensor dispatch routes in a separate run."""
    process = _ref(process, PROCESS)
    original = _original(process)
    scope = original[0]
    outputs = tuple(map(Path, scope['outputs']))
    _need(all(path.parent.is_dir() and not path.exists() and not path.is_symlink() for path in outputs),
          'Tensor diagnostic needs the exact original fresh publication paths')
    folder = Path(out).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    context = scope['context']
    command = [*context['prefix'], context['python']['path'], '-B', '-c', _DISPATCH_CHILD,
               process['uri'], str(folder/'dispatch.json')]
    run = subprocess.run(command, env=context['environment'], cwd=context['working_directory'],
                         capture_output=True, timeout=120)
    for name in ('stdout', 'stderr'):
        (folder/name).write_bytes(getattr(run, name))
    for index, path in enumerate(outputs):
        if path.is_file() and not path.is_symlink():
            move(path, folder/f'publication-{index}')
    _need(run.returncode == 0, 'original Tensor diagnostic failed, see retained stderr')
    _need(_original(process) == original, 'original evidence changed during Tensor diagnosis')
    report = _tensor_flow_report(folder, process, original, run.returncode)
    (folder/'result.json').write_text(json.dumps(report, sort_keys=True, allow_nan=False)+'\n')
    return report


def verify_nvhpc_tensor_flow(reference, *, process):
    reference = _ref(reference, TENSOR_FLOW)
    process = _ref(process, PROCESS)
    saved = read_artifact(reference)
    report = _tensor_flow_report(Path(reference['uri']).parent, process, _original(process), saved['outcome']['returncode'])
    _need(report == saved, 'Tensor flow report differs from its retained capture and original typed graph')
    read_artifact(reference, json_document=False)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar observe-nvhpc-source-effects', description=__doc__)
    parser.add_argument('--process', required=True, type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--out', type=Path)
    action.add_argument('--check', type=Path)
    parser.add_argument('--strace', default='/usr/bin/strace')
    parser.add_argument('--tensor-flow', action='store_true', help='observe or replay exact CPU Tensor data edges')
    args = parser.parse_args(argv)
    if args.tensor_flow:
        path = args.check if args.check is not None else args.out/'result.json'
        report = (verify_nvhpc_tensor_flow(path, process=args.process) if args.check is not None
                  else observe_nvhpc_tensor_flow(args.process, args.out))
        projection = report['projection']
        print(json.dumps(dict(result=str(path.resolve()), source_nodes=len(projection['source_nodes']),
            data_edges=len(projection['observed_data_dependencies']), performance_measured=False)))
        return 0
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
