"""Canonical artifact bytes and durable publication."""

import gzip
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import tempfile
import time


def _json(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')


def file_reference(path, schema):
    """Identify a file without loading the whole artifact into memory."""
    path = Path(path).absolute()
    size = 0
    digest = sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return dict(uri=str(path), schema=schema, size_bytes=size, sha256=digest.hexdigest())


def read_gzip_payload(root, receipt, *, kind):
    """Read a relative shard under its compressed and decoded byte budgets."""
    root = Path(root).absolute()
    relative = Path(receipt['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'{kind} path escapes its store: {relative}')
    path = root / relative
    if path.resolve() != path or not path.is_file():
        raise ValueError(f'{kind} is absent or traverses a symlink: {relative}')
    stored_size, decoded_size = receipt['bytes'], receipt['decoded_bytes']
    if any(type(n) is not int or n < 0 for n in (stored_size, decoded_size)):
        raise ValueError(f'{kind} byte receipts must be nonnegative integers')
    with path.open('rb') as stream:
        payload = stream.read(stored_size + 1)
    if len(payload) != stored_size or sha256(payload).hexdigest() != receipt['sha256']:
        raise ValueError(f'stored {kind} receipt differs: {relative}')
    with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
        decoded = stream.read(decoded_size + 1)
    if len(decoded) != decoded_size or sha256(decoded).hexdigest() != receipt['decoded_sha256']:
        raise ValueError(f'decoded {kind} receipt differs: {relative}')
    return decoded


def read_record_table(root, receipt):
    decoded = read_gzip_payload(root, receipt, kind='context binding table')
    if decoded.count(b'\n') != receipt['records']:
        raise ValueError('decoded context binding table differs in record count')
    return tuple(json.loads(line) for line in decoded.splitlines())


def write_table(path, records, *, timings=None):
    """Write source records in canonical JSONL/gzip with one tag admission."""
    from scar.ir.program.source_productions import PRODUCTION_RECORD_TAGS
    from scar.ir.record_codec import _encode, _record_tags

    started = time.monotonic()
    tags = _record_tags(PRODUCTION_RECORD_TAGS)
    digest = sha256()
    decoded_bytes = count = 0
    with path.open('xb') as destination:
        with gzip.GzipFile(filename='', mode='wb', fileobj=destination, compresslevel=3, mtime=0) as stream:
            for record in records:
                raw = _json(_encode(record, tags))
                stream.write(raw)
                digest.update(raw)
                decoded_bytes += len(raw)
                count += 1
        destination.flush()
        os.fsync(destination.fileno())
    reference = file_reference(path, 'gzip-jsonl')
    if timings is not None:
        timings[path.name] = time.monotonic() - started
    return dict(path=path.name, bytes=reference['size_bytes'], sha256=reference['sha256'],
                decoded_bytes=decoded_bytes, decoded_sha256=digest.hexdigest(), records=count)


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _durable_bytes(path, payload, *, replace):
    """Publish a complete file; no-clobber mode preserves former xb semantics."""
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            # link publishes the fully synced inode while atomically refusing
            # a concurrently created report. replace would overwrite that file.
            os.link(temporary, path)
            temporary.unlink()
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _durable_mkdir(path):
    missing, parent = [], path
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    path.mkdir(parents=True, exist_ok=True)
    for directory in reversed(missing):
        _sync_directory(directory)
        _sync_directory(directory.parent)
