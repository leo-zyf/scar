"""Bounded, detached reuse inside one synchronous metadata replay.

Each transaction owns its cache and content guards. Hits re-hash dependencies;
successful exit rechecks them again. Optional Python file-open observation is
limited to a requested pure metadata factory; it is neither a native-I/O
tracker nor a sandbox, and never observes workload execution on our behalf.
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import os
from pathlib import Path
import stat
import subprocess
import sys
from threading import get_ident

_ACTIVE = ContextVar('scar_metadata_replay_transaction', default=None)
_READING = ContextVar('scar_metadata_dependency_read', default=False)
_HOOK_READY = False
_HOOK_ATTEMPTED = False
_PROBE = None
_SEEN = None
_POPEN_TYPE = subprocess.Popen
_POPEN_INIT = _POPEN_TYPE.__init__.__code__
_POPEN_EXECUTE = _POPEN_TYPE._execute_child.__code__
_MUTATION_EVENTS = frozenset(('os.remove', 'os.rename', 'os.rmdir', 'os.mkdir',
    'os.truncate', 'os.chmod', 'os.chown', 'os.utime', 'os.link', 'os.symlink'))


def current_transaction():
    value = _ACTIVE.get()
    return value if value is not None and not value.closed and value.owner == get_ident() else None


def _file_version(path):
    path = str(Path(path).absolute())
    token = _READING.set(True)
    try:
        with open(path, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError('metadata dependency is not a regular file: '+path)
            digest = sha256(); size = 0
            while chunk := stream.read(1024*1024):
                digest.update(chunk); size += len(chunk)
            after = os.fstat(stream.fileno())
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) !=
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
            raise ValueError('metadata dependency changed while hashing: '+path)
        signature = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        return path, digest.hexdigest(), size, signature
    finally:
        _READING.reset(token)


@dataclass
class _Capture:
    observe_reads: bool
    identity_queries: tuple = ()
    dependencies: dict = field(default_factory=dict)
    reusable: bool = True
    pipe_processes: set = field(default_factory=set)
    query_processes: set = field(default_factory=set)


def _popen_frame(code, descriptor=None):
    """Find the actual stdlib Popen owner; unrelated FDs cannot borrow approval."""
    frame = sys._getframe()
    try:
        while frame is not None:
            if frame.f_code is code:
                scope = frame.f_locals
                process = scope.get('self')
                if type(process) is not _POPEN_TYPE:
                    return None
                if descriptor is not None and descriptor not in (scope.get('c2pread'), scope.get('errread')):
                    return None
                return process
            frame = frame.f_back
    finally:
        del frame


def _audit(event, args):
    global _SEEN
    if event == 'scar.replay.observe_probe':
        if args and args[0] is _PROBE:
            _SEEN = _PROBE
        return
    if event not in ('open', 'subprocess.Popen', 'os.listdir', 'os.scandir') and event not in _MUTATION_EVENTS:
        return
    if _READING.get():
        return
    transaction = current_transaction()
    while transaction is not None:
        if any(row.observe_reads for row in transaction._captures):
            if event == 'open':
                transaction._opened(*args)
            elif event == 'subprocess.Popen':
                transaction._process(*args)
            else:
                # Directory membership and side effects need separate proof;
                # a collection of observed file reads cannot establish them.
                for row in transaction._captures:
                    row.reusable = False
        transaction = transaction.parent


def _ensure_observer():
    global _HOOK_READY, _HOOK_ATTEMPTED, _PROBE
    if not _HOOK_ATTEMPTED:
        _HOOK_ATTEMPTED = True
        try:
            sys.addaudithook(_audit)
            _PROBE = object()
            sys.audit('scar.replay.observe_probe', _PROBE)
            _HOOK_READY = _SEEN is _PROBE
        except Exception:
            # Optional observation cannot be an added admission requirement.
            # Existing interpreter policy may disallow hooks or audit probes.
            _HOOK_READY = False
    return _HOOK_READY


class ReplayTransaction:
    def __init__(self, *, max_entries=32, parent=None):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError('positive bounded replay entry count required')
        self.max_entries = max_entries
        self.parent = parent
        self.owner = get_ident()
        self.closed = False
        self._documents = OrderedDict()
        self._watched = {}
        self._captures = []
        self.statistics = dict(hits=0, misses=0, bypasses=0, watched_files=0, evictions=0)
        self.by_namespace = {}

    def _remember(self, version):
        path, digest, size, signature = version
        previous = self._watched.get(path)
        if previous is not None and previous != (digest, size, signature):
            raise ValueError('metadata dependency version changed within replay: '+path)
        self._watched[path] = digest, size, signature
        self.statistics['watched_files'] = len(self._watched)
        for capture in self._captures:
            capture.dependencies[path] = digest, size
        if self.parent is not None:
            self.parent._remember(version)

    def watch(self, path, sha256, size_bytes):
        if self.closed or self.owner != get_ident():
            raise ValueError('replay dependency needs its live owner transaction')
        version = _file_version(path)
        if version[1:3] != (sha256, size_bytes):
            raise ValueError('metadata dependency byte identity differs: '+version[0])
        self._remember(version)

    def _opened(self, path, mode, flags):
        if not self._captures:
            return
        if type(path) is int:
            try:
                pipe = stat.S_ISFIFO(os.fstat(path).st_mode)
            except OSError:
                pipe = False
            process = _popen_frame(_POPEN_INIT, path) if pipe else None
            for row in self._captures:
                if process is not None and row.identity_queries and isinstance(mode, str) and mode.startswith('r'):
                    row.pipe_processes.add(process)
                else:
                    row.reusable = False
            return
        if (not isinstance(path, (str, bytes, os.PathLike))
                or not isinstance(flags, int) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)):
            for row in self._captures:
                row.reusable = False
            return
        path = os.fsdecode(path)
        absolute = str(Path(path).absolute())
        if any(absolute == root or absolute.startswith(root+'/') for root in ('/proc','/sys','/dev')):
            for row in self._captures:
                row.reusable = False
            return
        try:
            version = _file_version(absolute)
        except (OSError, ValueError):
            # Missing/volatile inputs keep the original factory behavior and
            # diagnostics, but cannot create a reusable proof document.
            for row in self._captures:
                row.reusable = False
            return
        self._remember(version)

    def _process(self, executable, args, cwd, env):
        command = tuple(args) if type(args) in (tuple, list) else ()
        process = _popen_frame(_POPEN_EXECUTE)
        for row in self._captures:
            if (process is not None and process in row.pipe_processes
                    and command in row.identity_queries and executable == command[0]
                    and cwd is None and env is None):
                row.query_processes.add(process)
            else:
                row.reusable = False

    def memo(self, namespace, key, factory, *, clone=True, capture_reads=False, identity_queries=()):
        """Reuse a pure result; optional identity query output must be in key.

        Non-file live state remains the caller's responsibility. Only explicit
        ``argv + --version`` identity queries can permit subprocess read pipes;
        all other observed subprocess/descriptor opens disable reuse. Pre-opened
        native I/O is outside this observer: factories must be pure metadata.
        This is not permission to execute a target within a replay factory.
        """
        identity_queries = tuple(tuple(command) for command in identity_queries)
        if any(not command or command[-1] != '--version'
                or any(type(arg) is not str for arg in command) for command in identity_queries):
            raise ValueError('only explicitly keyed compiler --version identity queries may use pipes')
        if type(namespace) is not str or not namespace:
            raise ValueError('explicit nonempty replay namespace required')
        counts = self.by_namespace.setdefault(namespace, dict(hits=0, misses=0))
        if self.closed or self.owner != get_ident() or clone is not True:
            self.statistics['bypasses'] += 1
            return factory()
        cache_key = namespace, key, capture_reads, identity_queries
        try:
            hash(cache_key)
        except TypeError:
            self.statistics['bypasses'] += 1
            return factory()
        if capture_reads and not _ensure_observer():
            self.statistics['bypasses'] += 1
            return factory()
        cached = self._documents.get(cache_key)
        if cached is not None:
            value, dependencies = cached
            for path, digest, size in dependencies:
                self.watch(path, digest, size)
            self._documents.move_to_end(cache_key)
            self.statistics['hits'] += 1
            counts['hits'] += 1
            return deepcopy(value)
        self.statistics['misses'] += 1
        counts['misses'] += 1
        capture = _Capture(capture_reads, identity_queries)
        self._captures.append(capture)
        try:
            result = factory()
        finally:
            self._captures.pop()
        if capture.reusable and capture.pipe_processes <= capture.query_processes:
            # Freeze a detached value before exposing the factory result.
            # A failed copy is an unsupported cache value, not a new failure
            # of the underlying metadata producer.
            try:
                stored = deepcopy(result)
            except (TypeError, ValueError):
                self.statistics['bypasses'] += 1
            else:
                dependencies = tuple((path, *version) for path, version in capture.dependencies.items())
                self._documents[cache_key] = stored, dependencies
                while len(self._documents) > self.max_entries:
                    self._documents.popitem(last=False)
                    self.statistics['evictions'] += 1
        else:
            self.statistics['bypasses'] += 1
        return result

    def validate(self):
        for path, (digest, size, _) in tuple(self._watched.items()):
            self.watch(path, digest, size)

    def close(self):
        self.closed = True
        self._documents.clear()
        self._captures.clear()
        self._watched.clear()


@contextmanager
def replay_transaction(*, max_entries=32):
    """Own a fresh cold scope; no memo entries survive successful or failed exit."""
    transaction = ReplayTransaction(max_entries=max_entries, parent=current_transaction())
    token = _ACTIVE.set(transaction)
    try:
        yield transaction
        transaction.validate()
    finally:
        transaction.close()
        _ACTIVE.reset(token)


__all__ = ['ReplayTransaction', 'current_transaction', 'replay_transaction']
