from scar.analysis.buffered_stream_publication import _compute
from scar.ir.program.call_edges import CallRecordAnchor, InvocationKey
from scar.ir.program.ids import ProviderID
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.program.resource_scopes import ObservedResourceScope, ScopedResourceWork, ScopeEvent, ScopeEventKind
from scar.ir.record_codec import encode
from scar.trace.stream_backing import BackingStatus, StreamBackingObservation, StreamBackingStat


def test_explicit_flush_associates_only_possible_preceding_text_writes():
    digest = 'a' * 64
    os_artifact = ProviderArtifact('/capture/os/syscalls.10', digest, 600, 'strace.syscalls.text.v1')
    stream_artifact = ProviderArtifact('/capture/stream.jsonl', digest, 200, 'stream')
    owner = InvocationKey(10, 100, 1)
    backing = StreamBackingObservation(
        BackingStatus.OBSERVED, 0, 0, ('_io.TextIOWrapper', '_io.BufferedWriter', '_io.FileIO'),
        3, StreamBackingStat(1, 2, 0o100644, 0, 0))

    def os_line(line):
        return ProviderFactReference(os_artifact, f'#/lines/{line}')

    calls, scopes = [], []
    for index, method in enumerate(('write', 'write', 'flush'), 1):
        anchor = CallRecordAnchor('/capture/attempts.jsonl', index, digest, index)
        begin = ScopeEvent(index * 2 - 1, index, ScopeEventKind.BEGIN, owner, 10, index, os_line(index * 3))
        end = ScopeEvent(index * 2, index, ScopeEventKind.C_RETURN, owner, 10, index, os_line(index * 3 + 2))
        scopes.append(ObservedResourceScope(ProviderID(str(index)), begin, end, anchor, begin.source, end.source))
        calls.append(dict(sequence=index, attempt_index=index, attempt_source=encode(anchor),
                          contract={'method': method}, invocation=encode(owner), receiver_type='_io.TextIOWrapper',
                          receiver_reference='weak:10:1', reference_identity_qualified=True,
                          completion='C_RETURN', backing_observation=encode(backing)))

    kernel = ScopedResourceWork(scopes[-1].id, os_line(10), 'write', 'BYTE_IO', 8)
    raw = {kernel.source: 3}
    publications, open_rows = _compute({'calls': calls}, scopes, [kernel], stream_artifact, raw)
    publication, = publications
    assert not open_rows
    assert [row.scope.begin.attempt_index for row in publication.preceding_text_writes] == [1, 2]
    assert publication.enclosed_kernel_writes == (kernel,)
    assert publication.contents_or_bytes_attributed is False
    assert publication.effects_closed is False

    publications, open_rows = _compute({'calls': calls[:-1]}, scopes[:-1], [], stream_artifact, raw)
    assert not publications
    assert {row['reason'] for row in open_rows} == {'EXPLICIT_PUBLICATION_NOT_OBSERVED'}

    publications, open_rows = _compute({'calls': calls}, scopes, [kernel], stream_artifact,
                                       {**raw, os_line(7): 3})
    assert not publications
    assert {row['reason'] for row in open_rows} == {'EARLIER_KERNEL_WRITE_PREVENTS_BUFFER_PUBLICATION_GROUP'}
