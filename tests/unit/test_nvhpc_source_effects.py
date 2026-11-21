from scar.analysis.nvhpc_source_effects import _classify, _input_status, _trace_members, _Witnesses


def test_effect_reader_distinguishes_publication_and_hidden_success(tmp_path):
    trace = tmp_path/'syscalls.100'
    expected = '''execve("/usr/bin/taskset", [...], 0x1234) = 0
execve("/python", [...], 0x1234) = 0
openat(AT_FDCWD, "/input.f32", O_RDONLY|O_CLOEXEC) = 3</input.f32>
mmap(NULL, 8, PROT_READ|PROT_WRITE, MAP_PRIVATE, 3</input.f32>, 0) = 0x1234
openat(AT_FDCWD, "/output.f32", O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC, 0666) = 4</output.f32>
write(4</output.f32>, ""..., 8) = 8
close(4</output.f32>) = 0
close(3</input.f32>) = 0
exit_group(0) = ?
+++ exited with 0 +++
'''
    scope = dict(command=['/usr/bin/taskset'], source=dict(path='/entry.py'),
                 context=dict(python=dict(path='/python'), providers={}),
                 inputs=['/input.f32'], outputs=['/output.f32'], expected_sizes=[8])
    trace.write_text(expected)
    clean = _classify(_trace_members(tmp_path), scope)
    assert clean['uncovered_effects'] == []
    assert [row['kind'] for row in clean['publications']] == [
        'PUBLICATION_OPEN', 'PUBLICATION_WRITE', 'PUBLICATION_CLOSE']

    extra = '''socket(AF_UNIX, SOCK_STREAM, 0) = 7<UNIX-STREAM:[123]>
openat(AT_FDCWD, "/tmp/hidden", O_WRONLY|O_CREAT|O_EXCL, 0600) = 3</tmp/hidden>
write(3</tmp/hidden>, ""..., 4) = 4
close(3</tmp/hidden>) = 0
unlink("/tmp/hidden") = 0
'''
    trace.write_text(expected.replace('openat(AT_FDCWD, "/input.f32"', extra+'openat(AT_FDCWD, "/input.f32"'))
    observed = _classify(_trace_members(tmp_path), scope)
    assert [row['kind'] for row in observed['uncovered_effects']] == [
        'NETWORK_OPERATION', 'FILE_OPEN_FOR_WRITE', 'OTHER_WRITE', 'FILESYSTEM_MUTATION']
    assert all(row['phase'] == 'BEFORE_FIRST_DECLARED_INPUT' for row in observed['uncovered_effects'])
    assert [row['path'] for row in observed['publications']] == ['/output.f32'] * 3
    witnesses = _Witnesses(scope)
    assert _classify(_trace_members(tmp_path), scope, witnesses=witnesses) == observed
    events = witnesses.files['/input.f32']
    assert _input_status(events, 8) == 'PRIVATE_FILE_MAPPING_OBSERVED'
    assert _input_status(events, 16) == 'UNOBSERVED_OR_INCOMPLETE_MAPPING'
