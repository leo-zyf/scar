"""Code snapshot."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
import dis
import hashlib
import inspect
import struct
import sys
import threading
import weakref
import json
import marshal
import os
from itertools import islice
from types import CodeType
from hashlib import sha256
from pathlib import Path


# Bounded structural code identities and exact bytecode positions.
#
# Only exact builtin constant types are inspected.  No application repr, hash,
# equality, descriptor, or target execution is used.  The v2 encoding is a
# length-delimited binary tree, independent of marshal reference flags and of
# adaptive bytecode.  These identities describe code, not callable environments.
_MAX_BYTES = 16 * 1024 * 1024
_MAX_NODES = 100_000
_MAX_DEPTH = 128
_MAX_CODE_UNITS = 100_000
_MAX_CACHE_ENTRIES = 128
_MAX_CACHE_TEXT_BYTES = 512 * 1024
_MAX_KEY_CACHE_ENTRIES = 2048
_MAX_PARAMETER_CACHE_BYTES = 512 * 1024
_MAX_CACHED_SITES = 128

_CODE_FIELDS = (
    "co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_nlocals",
    "co_stacksize", "co_flags", "co_code", "co_consts", "co_names",
    "co_varnames", "co_freevars", "co_cellvars", "co_filename", "co_name",
    "co_qualname", "co_firstlineno", "co_linetable", "co_exceptiontable",
)
_DOMAIN = b"scar-code-structural-v2\0"


class _Encoder:
    def __init__(self):
        self.bytes = self.nodes = 0

    def write(self, output, data):
        if len(data) > _MAX_BYTES - self.bytes:
            raise ValueError("code snapshot encoded-byte budget exceeded")
        self.bytes += len(data)
        output.extend(data)

    def length(self, output, value):
        self.write(output, struct.pack(">Q", value))

    def encode(self, value, output, depth=0):
        if depth > _MAX_DEPTH:
            raise ValueError("code snapshot nesting budget exceeded")
        self.nodes += 1
        if self.nodes > _MAX_NODES:
            raise ValueError("code snapshot node budget exceeded")
        kind = type(value)
        if value is None:
            self.write(output, b"N")
        elif value is Ellipsis:
            self.write(output, b"E")
        elif kind is bool:
            self.write(output, b"B\x01" if value else b"B\x00")
        elif kind is int:
            size = (value.bit_length() + 7) // 8
            self.write(output, b"I-" if value < 0 else b"I+")
            self.length(output, size)
            if size > _MAX_BYTES - self.bytes:
                raise ValueError("code snapshot encoded-byte budget exceeded")
            self.write(output, abs(value).to_bytes(size, "big"))
        elif kind is float:
            self.write(output, b"F" + struct.pack(">d", value))
        elif kind is complex:
            self.write(output, b"C" + struct.pack(">dd", value.real, value.imag))
        elif kind is str:
            # UTF-8 needs at least one byte per character.  Check before encoding
            # so even a rejected adversarial string has bounded temporary size.
            if len(value) > _MAX_BYTES - self.bytes:
                raise ValueError("code snapshot encoded-byte budget exceeded")
            data = value.encode("utf-8", "surrogatepass")
            self.write(output, b"S")
            self.length(output, len(data))
            self.write(output, data)
        elif kind is bytes:
            self.write(output, b"Y")
            self.length(output, len(value))
            self.write(output, value)
        elif kind is tuple:
            self.write(output, b"T")
            self.length(output, len(value))
            for item in value:
                self.encode(item, output, depth + 1)
        elif kind is slice:
            self.write(output, b"L")
            for item in (value.start, value.stop, value.step):
                self.encode(item, output, depth + 1)
        elif kind is frozenset:
            self.write(output, b"R")
            self.length(output, len(value))
            members = []
            for item in value:
                encoded = bytearray()
                self.encode(item, encoded, depth + 1)
                members.append(bytes(encoded))
            # Sorting only encoded bytes never compares or hashes constants.
            for encoded in sorted(members):
                output.extend(encoded)  # Already charged by recursive encoding.
        elif kind is CodeType:
            self.write(output, b"K")
            self.length(output, len(_CODE_FIELDS))
            for name in _CODE_FIELDS:
                self.encode(name, output, depth + 1)
                self.encode(getattr(value, name), output, depth + 1)
        else:
            raise ValueError("unsupported exact code constant type")


@dataclass(slots=True)
class _CachedCode:
    reference: weakref.ReferenceType
    key: str
    filename: str
    qualname: str
    first_line: int
    flags: int
    parameters: tuple[str, ...]
    text_bytes: int
    limits: tuple[int, int, int]


@dataclass(slots=True)
class _CachedKey:
    reference: weakref.ReferenceType
    key: str
    limits: tuple[int, int, int]
    sites: OrderedDict = field(default_factory=OrderedDict)
    calls: OrderedDict = field(default_factory=OrderedDict)
    parameters: tuple[str, ...] | None = None
    parameter_bytes: int = 0


_CACHE: OrderedDict[int, _CachedCode] = OrderedDict()
_KEY_CACHE: OrderedDict[int, _CachedKey] = OrderedDict()
_CACHE_TEXT_BYTES = 0
_PARAMETER_CACHE_BYTES = 0
_CACHE_LOCK = threading.RLock()


def _forget(identity, reference):
    global _CACHE_TEXT_BYTES
    with _CACHE_LOCK:
        record = _CACHE.get(identity)
        if record is not None and record.reference is reference:
            _CACHE_TEXT_BYTES -= record.text_bytes
            del _CACHE[identity]


def _forget_key(identity, reference):
    global _PARAMETER_CACHE_BYTES
    with _CACHE_LOCK:
        record = _KEY_CACHE.get(identity)
        if record is not None and record.reference is reference:
            _PARAMETER_CACHE_BYTES -= record.parameter_bytes
            del _KEY_CACHE[identity]


def _key_record(code):
    global _PARAMETER_CACHE_BYTES
    if type(code) is not CodeType:
        raise ValueError("code snapshot requires an exact CodeType")
    identity = id(code)
    limits = (_MAX_BYTES, _MAX_NODES, _MAX_DEPTH)
    with _CACHE_LOCK:
        cached = _KEY_CACHE.get(identity)
        if cached is not None and cached.reference() is code and cached.limits == limits:
            _KEY_CACHE.move_to_end(identity)
            return cached
    encoded = bytearray()
    encoder = _Encoder()
    try:
        encoder.write(encoded, _DOMAIN)
        encoder.encode(code, encoded)
    except RecursionError as error:
        raise ValueError("code snapshot nesting budget exceeded") from error
    key = "pycode-v2:" + hashlib.sha256(encoded).hexdigest()
    count = code.co_argcount + code.co_kwonlyargcount
    count += bool(code.co_flags & inspect.CO_VARARGS)
    count += bool(code.co_flags & inspect.CO_VARKEYWORDS)
    if count > len(code.co_varnames):
        raise ValueError("code snapshot parameter layout is inconsistent")
    reference = weakref.ref(code, lambda ref: _forget_key(identity, ref))
    record = _CachedKey(reference, key, limits)
    with _CACHE_LOCK:
        previous = _KEY_CACHE.get(identity)
        if previous is not None and previous.reference() is code and previous.limits == limits:
            _KEY_CACHE.move_to_end(identity)
            return previous
        if previous is not None:
            _PARAMETER_CACHE_BYTES -= previous.parameter_bytes
            del _KEY_CACHE[identity]
        if _MAX_KEY_CACHE_ENTRIES > 0:
            _KEY_CACHE[identity] = record
            while len(_KEY_CACHE) > _MAX_KEY_CACHE_ENTRIES:
                _, evicted = _KEY_CACHE.popitem(last=False)
                _PARAMETER_CACHE_BYTES -= evicted.parameter_bytes
    return record


def code_key(code: CodeType) -> str:
    """Return the structural code key without allocating a descriptor mapping.

    Only fully encoded, supported CodeType trees enter the bounded cache. Cache
    identity is guarded by a weak reference so recycled object addresses cannot
    inherit a prior code object's key.
    """
    return _key_record(code).key


def code_parameters(code: CodeType) -> tuple[str, ...]:
    """Immutable formal names, cached only for the validated live CodeType."""
    global _PARAMETER_CACHE_BYTES
    record = _key_record(code)
    with _CACHE_LOCK:
        if record.parameters is not None:
            return record.parameters
    count = code.co_argcount + code.co_kwonlyargcount
    count += bool(code.co_flags & inspect.CO_VARARGS) + bool(code.co_flags & inspect.CO_VARKEYWORDS)
    if count > len(code.co_varnames):
        raise ValueError('code snapshot parameter layout is inconsistent')
    parameters = code.co_varnames[:count]
    size = 64 + 8*len(parameters) + 4*sum(map(len,parameters))
    with _CACHE_LOCK:
        if (_KEY_CACHE.get(id(code)) is record and record.parameters is None
                and _PARAMETER_CACHE_BYTES + size <= _MAX_PARAMETER_CACHE_BYTES):
            record.parameters, record.parameter_bytes = parameters, size
            _PARAMETER_CACHE_BYTES += size
    return parameters


def call_instruction(code: CodeType, offset: int) -> tuple[str, int | None] | None:
    """Exact static instruction at a CALL site, without retaining argval/repr.

    The fixed CPython 3.14.7 byte-only decoder retains EXTENDED_ARG and inline
    CACHE boundaries without resolving any argval or calling application repr.
    Only opcode name/oparg survive; no Instruction, constant or frame is cached.
    """
    if type(offset) is not int:
        raise ValueError('instruction offset must be an exact integer')
    if sys.implementation.name != 'cpython' or sys.version_info[:3] != (3,14,7):
        raise RuntimeError('CALL instruction metadata requires pinned CPython 3.14.7')
    record = _key_record(code)
    if len(code.co_code) // 2 > _MAX_CODE_UNITS:
        raise ValueError('code snapshot instruction budget exceeded')
    with _CACHE_LOCK:
        if offset in record.calls:
            record.calls.move_to_end(offset)
            return record.calls[offset]
    # In this pinned ABI _unpack_opargs yields (offset, start, opcode, oparg),
    # skips inline CACHE units and assembles signed EXTENDED_ARG operands. It
    # reads bytes only. get_instructions would also format LOAD_CONST argrepr.
    value = None
    for instruction_offset, _start, opcode, argument in dis._unpack_opargs(code.co_code):
        if instruction_offset == offset:
            value = (dis._all_opname[opcode], argument)
            break
        if instruction_offset > offset:
            break
    with _CACHE_LOCK:
        if _MAX_CACHED_SITES > 0:
            record.calls[offset] = value
            while len(record.calls) > _MAX_CACHED_SITES:
                record.calls.popitem(last=False)
    return value


def _record(code):
    global _CACHE_TEXT_BYTES
    if type(code) is not CodeType:
        raise ValueError("code snapshot requires an exact CodeType")
    identity = id(code)
    limits = (_MAX_BYTES, _MAX_NODES, _MAX_DEPTH)
    with _CACHE_LOCK:
        cached = _CACHE.get(identity)
        if cached is not None and cached.reference() is code and cached.limits == limits:
            _CACHE.move_to_end(identity)
            return cached
    key = _key_record(code).key
    count = code.co_argcount + code.co_kwonlyargcount
    count += bool(code.co_flags & inspect.CO_VARARGS)
    count += bool(code.co_flags & inspect.CO_VARKEYWORDS)
    if count > len(code.co_varnames):
        raise ValueError("code snapshot parameter layout is inconsistent")
    parameters = code.co_varnames[:count]
    text_bytes = 4 * sum(len(value) for value in (code.co_filename, code.co_qualname, *parameters)) + len(key)
    reference = weakref.ref(code, lambda ref: _forget(identity, ref))
    record = _CachedCode(reference, key, code.co_filename, code.co_qualname,
                         code.co_firstlineno, code.co_flags, parameters, text_bytes, limits)
    with _CACHE_LOCK:
        previous = _CACHE.pop(identity, None)
        if previous is not None:
            _CACHE_TEXT_BYTES -= previous.text_bytes
        if _MAX_CACHE_ENTRIES > 0 and text_bytes <= _MAX_CACHE_TEXT_BYTES:
            _CACHE[identity] = record
            _CACHE_TEXT_BYTES += text_bytes
            while len(_CACHE) > _MAX_CACHE_ENTRIES or _CACHE_TEXT_BYTES > _MAX_CACHE_TEXT_BYTES:
                _, evicted = _CACHE.popitem(last=False)
                _CACHE_TEXT_BYTES -= evicted.text_bytes
    return record


def describe_code(code: CodeType) -> dict:
    """Describe canonical loaded code without retaining the code or constants.

    Raises ValueError for unsupported constants or finite encoding limits.
    Parameters follow the code layout: positional, keyword-only, *args, **kwargs.
    Returned containers are fresh and may be modified by callers.
    """
    record = _record(code)
    return {"key": record.key, "filename": record.filename, "qualname": record.qualname,
            "first_line": record.first_line, "flags": record.flags,
            "parameters": list(record.parameters)}


def site(code: CodeType, offset: int) -> dict:
    """Describe one exact CPython wordcode offset, including non-CALL opcodes.

    Offsets outside the code or inside a two-byte code unit have no opcode or
    position.  CACHE units remain CACHE; they are never reassigned to a CALL.
    This reports bytecode provenance, not a proven source ast.Call mapping.
    """
    if type(offset) is not int:
        raise ValueError("instruction offset must be an exact integer")
    record = _key_record(code)
    raw = code.co_code  # Canonical, unquickened bytes, even after specialization.
    if len(raw) // 2 > _MAX_CODE_UNITS:
        raise ValueError("code snapshot instruction budget exceeded")
    if offset < 0 or offset >= len(raw) or offset % 2:
        return {"code": record.key, "offset": offset, "opname": None, "position": None}
    with _CACHE_LOCK:
        known = record.sites.get(offset)
        if known is not None:
            record.sites.move_to_end(offset)
    if known is None:
        # Do not ask dis to resolve arguments: it renders constants with repr.
        # co_positions has one record per two-byte code unit, including caches.
        position = next(islice(code.co_positions(), offset // 2, None), None)
        if position is not None and any(value is None for value in position):
            position = None
        known = (dis.opname[raw[offset]], position)
        with _CACHE_LOCK:
            if _MAX_CACHED_SITES > 0:
                record.sites[offset] = known
                while len(record.sites) > _MAX_CACHED_SITES:
                    record.sites.popitem(last=False)
    opname, position = known
    return {"code": record.key, "offset": offset, "opname": opname,
            "position": list(position) if position is not None else None}


# CPython-local temporary bytes journal for collector-owned snapshot rows.
#
# This is an internal transport, not an arbitrary-object serialization API.
# Callers must build plain snapshot trees before append; no application object
# may be submitted. append performs no recursive Python validation or JSON work.
MAGIC = b'SCAR-SNAPSHOT-JOURNAL\x00\x01'
HEADER_LENGTH = struct.Struct('<I')
FRAME = struct.Struct('<cQQ')
RECORD, FOOTER = b'R', b'F'
FORMAT = 'scar.snapshot-journal/1'
HEADER_LIMIT = 65_536
FOOTER_RESERVE = 16_384
_writer_local = threading.local()
_owned_threads = {}


def _register_writer(thread):
    key = id(thread)
    def discard(reference):
        if _owned_threads.get(key) is reference:
            _owned_threads.pop(key, None)
    _owned_threads[key] = weakref.ref(thread, discard)


def is_journal_writer_thread():
    """Exclude this entire IO-thread stack in collector callbacks."""
    if getattr(_writer_local, 'active', False) is True:
        return True
    current = dict.get(threading._active, threading.get_ident())
    reference = _owned_threads.get(id(current))
    return reference is not None and reference() is current


def _json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('ascii')


def _write_all(fd, payload):
    remaining = memoryview(payload)
    while remaining:
        try:
            count = os.write(fd, remaining)
        except InterruptedError:
            continue
        if count <= 0:
            raise OSError('short snapshot-journal write')
        remaining = remaining[count:]


def _error(error):
    # Never repr/str the snapshot or an arbitrary object from the exception.
    return {'type': type(error).__name__, 'errno': getattr(error, 'errno', None)}


class SnapshotJournal:
    def __init__(self, path, *, max_queue_bytes=8 * 1024 * 1024,
                 max_record_bytes=4 * 1024 * 1024, max_total_bytes=1024 * 1024 * 1024,
                 batch_bytes=64 * 1024, close_timeout=30.0):
        if sys.implementation.name != 'cpython':
            raise ValueError('snapshot journal requires CPython marshal with allow_code=False')
        for value in (max_queue_bytes, max_record_bytes, max_total_bytes, batch_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError('journal byte ceilings must be positive exact integers')
        if type(close_timeout) not in (int, float) or not 0 < close_timeout <= 3600:
            raise ValueError('journal close timeout must be finite and in (0,3600]')
        self.path = Path(path).absolute()
        if self.path != self.path.resolve() or self.path.exists() or self.path.is_symlink():
            raise ValueError('fresh canonical journal path required')
        self._pid = os.getpid()
        self._queue_limit, self._record_limit = max_queue_bytes, max_record_bytes
        self._total_limit, self._batch_limit = max_total_bytes, batch_bytes
        self._close_timeout = close_timeout
        self._condition, self._close_lock = threading.Condition(), threading.Lock()
        self._queue = deque()
        self._queued_bytes = self._queue_highwater = self._inflight_highwater = 0
        self._attempts = self._accepted_records = self._accepted_bytes = self._accepted_frame_bytes = 0
        self._last_record_bytes = self._written_records = self._written_frame_bytes = 0
        self._rejected = {key: 0 for key in ('wrong_root', 'serialization_error', 'record_byte_limit',
            'total_byte_limit', 'queue_full', 'closing_or_closed', 'writer_unavailable')}
        self._fork_rejections = 0
        self._closing = self._closed = self._timed_out = self._child_closed = False
        self._writer_error = None
        self._receipt = None
        self.header = {'format': FORMAT, 'schema_version': 1, 'owner_pid': self._pid,
            'implementation': sys.implementation.name, 'python': sys.version,
            'python_version': list(sys.version_info), 'cache_tag': sys.implementation.cache_tag,
            'marshal_version': marshal.version, 'allow_code': False,
            'snapshot_contract': 'COLLECTOR_OWNED_PLAIN_SNAPSHOT_TREE',
            'ordering': 'append-lock order; not a cross-thread causal assertion',
            'max_queue_bytes': max_queue_bytes, 'max_record_bytes': max_record_bytes,
            'max_total_bytes': max_total_bytes, 'batch_bytes': batch_bytes}
        header = _json(self.header)
        if len(header) > HEADER_LIMIT or len(MAGIC) + HEADER_LENGTH.size + len(header) + FOOTER_RESERVE >= max_total_bytes:
            raise ValueError('journal total byte ceiling cannot reserve header/footer')
        self._header_bytes = len(MAGIC) + HEADER_LENGTH.size + len(header)
        self._fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
        initial = os.fstat(self._fd)
        self._identity = initial.st_dev, initial.st_ino
        try:
            _write_all(self._fd, MAGIC + HEADER_LENGTH.pack(len(header)) + header)
        except BaseException:
            os.close(self._fd); self._fd = None
            raise
        self._thread = threading.Thread(target=self._writer_main,
            name='SCAR-snapshot-journal-io', daemon=True)
        self._ready = threading.Event()
        _register_writer(self._thread)
        self._thread.start()
        self._ready.wait()  # Installation only; append never waits for space.

    @property
    def accepted_bytes(self):
        """Accepted marshal payload bytes, never a JSON byte-quota claim."""
        return self._accepted_bytes if os.getpid() == self._pid else 0

    @property
    def accepted_records(self):
        return self._accepted_records if os.getpid() == self._pid else 0

    @property
    def last_record_bytes(self):
        return self._last_record_bytes if os.getpid() == self._pid else 0

    @property
    def writer_thread_id(self):
        return self._thread.ident

    @property
    def writer_native_thread_id(self):
        return self._thread.native_id

    def append(self, row):
        """Synchronously freeze a collector-owned snapshot; never wait for space."""
        if os.getpid() != self._pid:
            # Do not acquire an inherited lock: its owner may not survive fork.
            self._fork_rejections += 1
            return False
        with self._condition:
            self._last_record_bytes = 0
            self._attempts += 1
            if self._writer_error is not None:
                self._rejected['writer_unavailable'] += 1
                return False
            if self._closing or self._closed:
                self._rejected['closing_or_closed'] += 1
                return False
            if type(row) is not dict:
                self._rejected['wrong_root'] += 1
                return False
            try:
                payload = marshal.dumps(row, marshal.version, allow_code=False)
            except Exception:
                self._rejected['serialization_error'] += 1
                return False
            size = len(payload)
            if size > self._record_limit:
                self._rejected['record_byte_limit'] += 1
                return False
            framed_size = FRAME.size + size
            if self._header_bytes + self._accepted_frame_bytes + framed_size + FOOTER_RESERVE > self._total_limit:
                self._rejected['total_byte_limit'] += 1
                return False
            if self._queued_bytes + framed_size > self._queue_limit:
                self._rejected['queue_full'] += 1
                return False
            ordinal = self._accepted_records + 1
            frozen = FRAME.pack(RECORD, ordinal, size) + payload
            self._queue.append(frozen)
            self._accepted_records = ordinal
            self._accepted_bytes += size
            self._accepted_frame_bytes += len(frozen)
            self._last_record_bytes = size
            self._queued_bytes += len(frozen)
            self._queue_highwater = max(self._queue_highwater, self._queued_bytes)
            self._condition.notify()
            return True

    def _writer_main(self):
        _writer_local.active = True
        self._ready.set()
        try:
            while True:
                with self._condition:
                    while not self._queue and not self._closing:
                        self._condition.wait()
                    if not self._queue:
                        return
                    batch, size = [], 0
                    while self._queue:
                        frame = self._queue[0]
                        if batch and size + len(frame) > self._batch_limit:
                            break
                        batch.append(self._queue.popleft())
                        size += len(frame)
                        self._queued_bytes -= len(frame)
                    self._inflight_highwater = max(self._inflight_highwater, size)
                try:
                    # IO batching only. No JSON, marshal decoding, graph work,
                    # payload traversal or checksum computation on this thread.
                    _write_all(self._fd, b''.join(batch))
                except BaseException as error:
                    with self._condition:
                        self._writer_error = _error(error)
                        self._closing = True
                        self._queue.clear(); self._queued_bytes = 0
                    return
                with self._condition:
                    self._written_records += len(batch)
                    self._written_frame_bytes += size
        finally:
            # Keep excluding threading's bootstrap tail after _active removes
            # this Thread. CPython clears threading.local at OS-thread exit.
            pass

    def _counts(self):
        return {'append_calls': self._attempts, 'accepted_records': self._accepted_records,
            'accepted_bytes': self._accepted_bytes, 'accepted_frame_bytes': self._accepted_frame_bytes,
            'written_records': self._written_records, 'written_frame_bytes': self._written_frame_bytes,
            'rejected': dict(self._rejected), 'queue_highwater_bytes': self._queue_highwater,
            'inflight_highwater_bytes': self._inflight_highwater, 'queued_bytes': self._queued_bytes,
            'writer_error': self._writer_error, 'close_timed_out': self._timed_out}

    def _hash_source(self):
        before = self.path.stat()
        if self.path.is_symlink() or (before.st_dev, before.st_ino) != self._identity or before.st_size > self._total_limit:
            raise ValueError('closed journal path/identity/byte ceiling differs')
        digest, count = sha256(), 0
        with self.path.open('rb') as stream:
            while chunk := stream.read(1024 * 1024):
                count += len(chunk)
                if count > self._total_limit:
                    raise ValueError('closed journal byte ceiling exceeded during hash')
                digest.update(chunk)
        after = self.path.stat()
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                or count != before.st_size):
            raise ValueError('closed journal changed during hash')
        return {'uri': str(self.path), 'raw_bytes': count, 'sha256': digest.hexdigest(), 'format': FORMAT,
                'declared_pid': self._pid}

    def close(self):
        if os.getpid() != self._pid:
            if not self._child_closed:
                if self._fd is not None:
                    os.close(self._fd)
                self._fd = None; self._queue.clear(); self._child_closed = True
            return {'status': 'FORK_INVALIDATED_CHILD', 'complete': False, 'owner_pid': self._pid,
                'process_id': os.getpid(), 'fork_rejections': self._fork_rejections,
                'parent_buffer_flushed': False, 'source': None}
        with self._close_lock:
            if self._receipt is not None:
                return dict(self._receipt)
            with self._condition:
                self._closing = True; self._condition.notify()
            self._thread.join(self._close_timeout)
            if self._thread.is_alive():
                self._timed_out = True
                return {'status': 'CLOSE_TIMEOUT', 'complete': False, 'source': None,
                    'counts': self._counts(), 'writer_thread_alive': True,
                    'archive_eligible': False, 'detail': 'IO may still be running; no footer or stable SHA'}
            counts = self._counts()
            complete = (self._writer_error is None and not self._timed_out
                and self._attempts == self._accepted_records == self._written_records)
            footer = {'schema': 'scar.snapshot-journal-footer', 'schema_version': 1,
                      'owner_pid': self._pid, 'complete': complete, 'counts': counts}
            errors, closed_footer = [], False
            try:
                if self._writer_error is None:
                    payload = _json(footer)
                    if len(payload) + FRAME.size > FOOTER_RESERVE:
                        raise ValueError('journal footer reserve exceeded')
                    _write_all(self._fd, FRAME.pack(FOOTER, self._accepted_records + 1, len(payload)) + payload)
                    closed_footer = True
            except BaseException as error:
                complete = False; errors.append(_error(error))
            finally:
                os.close(self._fd); self._fd = None; self._closed = True
            try:
                source = {**self._hash_source(), 'closed_footer': closed_footer}
            except BaseException as error:
                source = None; complete = False; errors.append(_error(error))
            self._receipt = {'status': 'CLOSED' if complete else 'INCOMPLETE', 'complete': complete,
                'source': source, 'counts': counts, 'errors': errors, 'rows_validated': False,
                'json_byte_quota_validated': False, 'writer_thread_alive': False,
                'writer_thread_id': self.writer_thread_id, 'writer_native_thread_id': self.writer_native_thread_id,
                'close_sha_readback_bytes': source['raw_bytes'] if source is not None else 0}
            return dict(self._receipt)


__all__ = [
    'code_key',
    'code_parameters',
    'call_instruction',
    'describe_code',
    'site',
    'MAGIC',
    'HEADER_LENGTH',
    'FRAME',
    'RECORD',
    'FOOTER',
    'FORMAT',
    'HEADER_LIMIT',
    'FOOTER_RESERVE',
    'is_journal_writer_thread',
    'SnapshotJournal',
]
