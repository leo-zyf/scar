"""Transport ordered private results from a validated native CSR stream."""

from contextlib import suppress
from math import ceil
import os
from pathlib import Path
import select
import signal
from struct import Struct
import subprocess
from tempfile import TemporaryFile
from time import monotonic

from scar.analysis.sparse_parallel_recipe import compute_scale
from scar.ir.execution_recipe import read_artifact


_HEADER = Struct('<8s32s4I')
_ROW_COUNT = Struct('<I')
_PAIR = Struct('<ii')
_TICKET = Struct('<Qd')
_RECORD = Struct('<QIi')
_SUCCESS = 0
_END = 4
_READ_TIMEOUT = 35
_CLOSE_TIMEOUT = 2


def _operands(snapshot):
    for array, length, dtype in (
            (snapshot.columns, snapshot.nnz, '<i4'), (snapshot.offsets, snapshot.size + 1, '<i4'),
            (snapshot.data, snapshot.nnz, '<f8'), (snapshot.load, snapshot.size, '<f8')):
        if array.shape != (length,) or array.dtype.str != dtype or not array.flags.c_contiguous:
            raise ValueError('detached snapshot does not fit the native wire ABI')
        yield memoryview(array).cast('B')


def _write_input(input_file, protocol, snapshot, geometry):
    pairs = sorted((int(row), int(slot)) for row, slot in zip(snapshot.rows, snapshot.selected_diagonal, strict=True))
    if len(pairs) > snapshot.size or any(left[0] == right[0] for left, right in zip(pairs, pairs[1:])):
        raise ValueError('detached input requires unique CSR rows')
    input_file.write(_HEADER.pack(b'SCARDSR1', *geometry))
    for payload in _operands(snapshot):
        input_file.write(payload)
    input_file.write(_ROW_COUNT.pack(len(pairs)))
    for pair in pairs:
        input_file.write(_PAIR.pack(*pair))
    for index in range(snapshot.count):
        scale = compute_scale(snapshot.expression, index, snapshot.count)
        if not 1.0 <= scale < 2.0:
            raise ValueError('detached input scale is outside [1.0, 2.0)')
        input_file.write(_TICKET.pack(index, scale))
    input_bytes = input_file.tell()
    sizes = protocol['bytes']
    if input_bytes != sizes['input_complete_base'] + len(pairs) * sizes['selected_row_slot_pair']:
        raise ValueError('detached input size differs from the native wire contract')
    input_file.seek(0)
    return input_bytes


class NativeDetachedStream:
    """Read private outputs in order. The caller checks live inputs before publication."""

    def __init__(self, package, manifest, snapshot, allowed_cpus):
        if manifest.get('schema_version') != 3:
            raise ValueError('a verified compact ordered package is required')
        directory = Path(package).resolve(strict=True)
        protocol_ref = manifest['stream']['protocol']
        protocol = read_artifact({**protocol_ref, 'uri': str(directory / protocol_ref['uri'])})
        header = protocol['header']
        self._shape = snapshot.size, snapshot.nnz, snapshot.count
        if (self._shape != (header['size'], header['nnz'], header['count'])
                or snapshot.expression != protocol['source']['expression']):
            raise ValueError('detached snapshot differs from the ordered recipe')
        geometry = (bytes.fromhex(header['recipe_digest_hex']), *self._shape, header['flags'])
        self._header = _HEADER.pack(b'SCAROUT1', *geometry)
        self._process = None
        self.stats = dict(input_bytes=0, accepted=0, last_terminal=None, completed=False, closed=False, pid=None)
        executable = directory / manifest['stream']['executable']['uri']
        try:
            with TemporaryFile() as input_file:
                self.stats['input_bytes'] = _write_input(input_file, protocol, snapshot, geometry)
                original_affinity = os.sched_getaffinity(0)
                os.sched_setaffinity(0, allowed_cpus)
                try:
                    self._process = subprocess.Popen([executable], stdin=input_file, stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL, env={}, bufsize=0, start_new_session=True)
                finally:
                    os.sched_setaffinity(0, original_affinity)
            self.stats['pid'] = self._process.pid
            self._fd = self._process.stdout.fileno()
            self._poll = select.poll()
            self._poll.register(self._fd, select.POLLIN)
        except BaseException:
            self.close()
            raise

    def _read_some(self, target, deadline):
        remaining = deadline - monotonic()
        if remaining <= 0 or not self._poll.poll(ceil(remaining * 1000)):
            raise TimeoutError('detached native output did not arrive')
        return os.readv(self._fd, (target,))

    def _read_exact(self, target, deadline):
        pending = memoryview(target).cast('B')
        while pending:
            received = self._read_some(pending, deadline)
            if not received:
                raise EOFError('detached native output ended before its complete record')
            pending = pending[received:]

    def take(self, index):
        """Return one owning binary64 result and zero worker status."""
        from numpy import empty

        size, _, count = self._shape
        if self._process is None or type(index) is not int or not 0 <= index < count or index != self.stats['accepted']:
            raise ValueError('detached native outputs must be taken in issue order')
        deadline = monotonic() + _READ_TIMEOUT
        if index == 0:
            header = bytearray(_HEADER.size)
            self._read_exact(header, deadline)
            if header != self._header:
                raise ValueError('detached native output header differs from the ordered recipe')
        record = bytearray(_RECORD.size)
        self._read_exact(record, deadline)
        received = _RECORD.unpack(record)
        if received != (index, _SUCCESS, 0):
            self.stats['last_terminal'] = received
            raise ValueError('detached native stream did not produce the expected success')
        answer = empty(size, dtype='<f8')
        self._read_exact(answer, deadline)
        if index + 1 == count:
            self._read_exact(record, deadline)
            terminal = self.stats['last_terminal'] = _RECORD.unpack(record)
            if terminal != (count, _END, 0) or self._read_some(bytearray(1), deadline):
                raise ValueError('detached native output lacks a complete terminal')
            try:
                status = self._process.wait(timeout=max(0, deadline - monotonic()))
            except subprocess.TimeoutExpired as error:
                raise TimeoutError('detached native process did not finish') from error
            if status != 0:
                raise ValueError('detached native process did not exit successfully')
            self.stats['completed'] = True
        self.stats['accepted'] += 1
        return answer, 0

    def close(self):
        """Close the private output pipe so the native parent can reap its workers."""
        process = self._process
        if process is None:
            return
        self._process = None
        try:
            process.stdout.close()
        finally:
            try:
                process.wait(timeout=_CLOSE_TIMEOUT)
            except BaseException as error:
                # The private session contains the native parent and its workers.
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                if not isinstance(error, subprocess.TimeoutExpired):
                    raise
            finally:
                self.stats['closed'] = True
