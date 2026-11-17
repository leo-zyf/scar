"""Replay isolation and file/version guards, independent of target workloads."""
import subprocess
import sys
import os

import pytest

from scar.analysis.replay_transaction import current_transaction, replay_transaction


def test_replay_is_detached_cold_and_version_guarded(tmp_path):
    path = tmp_path/'fact.txt'; path.write_text('old')
    calls = []

    def read():
        calls.append(1)
        return dict(items=[path.read_text()])

    with replay_transaction() as transaction:
        first = transaction.memo('fact', 'same', read, capture_reads=True)
        first['items'].append('caller mutation')
        assert transaction.memo('fact', 'same', read, capture_reads=True) == dict(items=['old'])
        assert len(calls) == 1
    assert current_transaction() is None and transaction.closed
    with replay_transaction() as independent:
        independent.memo('fact', 'same', read, capture_reads=True)
    assert len(calls) == 2

    with pytest.raises(ValueError):
        with replay_transaction() as changed:
            changed.memo('fact', 'same', read, capture_reads=True)
            path.write_text('new')
            changed.memo('fact', 'same', read, capture_reads=True)
    path.write_text('old')
    with pytest.raises(ValueError):
        with replay_transaction() as restored:
            restored.memo('fact', 'same', read, capture_reads=True)
            path.write_text('new'); path.write_text('old')
            restored.memo('fact', 'same', read, capture_reads=True)
    assert current_transaction() is None


def test_identity_pipes_are_explicit_and_unwatched_io_does_not_create_hits(tmp_path):
    command = (sys.executable, '--version')
    identity = subprocess.check_output(command, text=True)
    calls = []

    def query():
        calls.append(1)
        return subprocess.check_output(command, text=True)

    with replay_transaction() as transaction:
        for _ in range(2):
            assert transaction.memo('identity', identity, query, capture_reads=True,
                identity_queries=(command,)) == identity
        assert len(calls) == 1
        for _ in range(2):
            transaction.memo('identity-unaccounted', identity, query, capture_reads=True)
        assert len(calls) == 3

    missing = tmp_path/'missing'
    misses = []

    def optional():
        misses.append(1)
        try:
            return missing.read_text()
        except FileNotFoundError:
            return None

    with replay_transaction() as transaction:
        assert transaction.memo('optional', 'same', optional, capture_reads=True) is None
        assert transaction.memo('optional', 'same', optional, capture_reads=True) is None
    assert len(misses) == 2


def test_identity_query_cannot_approve_an_unrelated_pipe_or_override():
    command = (sys.executable, '--version')
    identity = subprocess.check_output(command, text=True)
    calls = []

    def unrelated():
        calls.append(1)
        subprocess.check_output(command, text=True)
        read, write = os.pipe()
        os.write(write, str(len(calls)).encode()); os.close(write)
        with os.fdopen(read) as stream:
            return stream.read()

    with replay_transaction() as transaction:
        assert transaction.memo('unrelated', identity, unrelated, capture_reads=True,
            identity_queries=(command,)) == '1'
        assert transaction.memo('unrelated', identity, unrelated, capture_reads=True,
            identity_queries=(command,)) == '2'
        assert transaction.statistics['hits'] == 0

    calls.clear()
    alleged = ('unrelated-executable', '--version')

    def overridden():
        calls.append(1)
        return subprocess.check_output(alleged, executable=sys.executable, text=True)

    with replay_transaction() as transaction:
        for _ in range(2):
            assert transaction.memo('override', identity, overridden, capture_reads=True,
                identity_queries=(alleged,)) == identity
        assert len(calls) == 2 and transaction.statistics['hits'] == 0
