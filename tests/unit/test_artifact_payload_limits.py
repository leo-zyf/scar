"""Stored and decoded evidence receipts enforce the active shard boundary."""
from contextlib import contextmanager
import gzip
from hashlib import sha256
from pathlib import Path

import pytest

from scar.artifacts import read_gzip_payload


@pytest.fixture
def shard(tmp_path):
    decoded = b"retained evidence\n" * 256
    stored = gzip.compress(decoded, mtime=0)
    path = tmp_path / "records.jsonl.gz"
    path.write_bytes(stored)
    receipt = dict(path=path.name, bytes=len(stored), sha256=sha256(stored).hexdigest(),
                   decoded_bytes=len(decoded), decoded_sha256=sha256(decoded).hexdigest())
    return path, receipt, decoded


def test_read_gzip_payload_returns_the_pinned_bytes(shard):
    path, receipt, decoded = shard
    assert read_gzip_payload(path.parent, receipt, kind="record table") == decoded


@pytest.mark.parametrize("field", ["bytes", "sha256"])
def test_invalid_stored_receipt_fails_before_decompression(shard, field, monkeypatch):
    path, receipt, _ = shard
    receipt[field] = 1 if field == "bytes" else "0" * 64

    def unexpected(*args, **kwargs):
        pytest.fail("an invalid stored receipt reached decompression")

    monkeypatch.setattr(gzip, "GzipFile", unexpected)
    with pytest.raises(ValueError, match="stored record table receipt differs"):
        read_gzip_payload(path.parent, receipt, kind="record table")


@pytest.mark.parametrize("field", ["decoded_bytes", "decoded_sha256"])
def test_compact_shard_requires_its_exact_decoded_size_and_hash(shard, field):
    path, receipt, _ = shard
    receipt[field] = 1 if field == "decoded_bytes" else "0" * 64
    with pytest.raises(ValueError, match="decoded record table receipt differs"):
        read_gzip_payload(path.parent, receipt, kind="record table")


def test_shard_growth_is_rejected_after_one_bounded_read(shard, monkeypatch):
    path, receipt, _ = shard
    requests = []
    open_path = Path.open

    class ObservedRead:
        def __init__(self, stream):
            self.stream = stream

        def read(self, count):
            requests.append(count)
            return self.stream.read(count)

    @contextmanager
    def growing_open(self, mode, *args, **kwargs):
        if self == path and mode == "rb":
            with open_path(path, "ab") as stream:
                stream.write(b"extra bytes")
            with open_path(self, mode, *args, **kwargs) as stream:
                yield ObservedRead(stream)
        else:
            with open_path(self, mode, *args, **kwargs) as stream:
                yield stream

    monkeypatch.setattr(Path, "open", growing_open)
    with pytest.raises(ValueError, match="stored record table receipt differs"):
        read_gzip_payload(path.parent, receipt, kind="record table")
    assert requests == [receipt["bytes"] + 1]
