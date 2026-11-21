"""A stack replay reuses hashes only for the same local file lifecycle."""

import os

from scar.compiler import operator_stack


def test_stack_hash_cache_is_scoped_and_invalidates_changed_file(tmp_path, monkeypatch):
    path = tmp_path/'component.so'
    path.write_bytes(b'old bytes')
    original = path.stat()
    checks = []
    hash_file = operator_stack.verify_library_file

    def tracked(member):
        checks.append(member.name)
        return hash_file(member)

    monkeypatch.setattr(operator_stack, 'verify_library_file', tracked)
    monkeypatch.setattr(operator_stack, 'local_file_identity',
                        lambda member: operator_stack.file_stat_identity(os.fstat(member.fileno())))
    with operator_stack.verify_stack_components_once():
        a = operator_stack.stack_component_content(path)
        with operator_stack.verify_stack_components_once():
            assert operator_stack.stack_component_content(path) == a
        assert len(checks) == 1
        path.write_bytes(b'new bytes')
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
        b = operator_stack.stack_component_content(path)
        assert a['sha256'] != b['sha256'] and a['size_bytes'] == b['size_bytes']
        assert len(checks) == 2
    with operator_stack.verify_stack_components_once():
        assert operator_stack.stack_component_content(path) == b
    assert len(checks) == 3
