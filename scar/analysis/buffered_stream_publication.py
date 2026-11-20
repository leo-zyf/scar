"""Join observed builtin text writes and explicit flushes to kernel receipts.

An enclosed write with a matching numeric FD and preceding operations on the
same wrapper make a possible buffer-publication dependency. Neither operation
order nor descriptor point observations prove which bytes the flush consumed.
"""

import argparse
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

from scar.analysis.call_operand_reader import read_call_operands
from scar.analysis.source_resource_scopes import derive_resource_scopes
from scar.ir.program.call_edges import InvocationKey
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.program.resource_scopes import ObservedResourceScope, ScopedResourceWork, ScopeEventKind
from scar.ir.record_codec import decode, encode
from scar.trace.capture_coordinates import bind_capture_coordinates
from scar.trace.stream_backing import BackingStatus, StreamBackingObservation
from scar.trace.stream_interfaces import read_stream_interfaces


_TEXT = '_io.TextIOWrapper'
_BUFFERED_CHAIN = (_TEXT, '_io.BufferedWriter', '_io.FileIO')
_METHODS = frozenset(('write', 'flush'))


@dataclass(frozen=True, slots=True)
class StreamCallEvidence:
    scope: ObservedResourceScope
    stream: ProviderFactReference


@dataclass(frozen=True, slots=True)
class PossibleBufferedPublication:
    """One explicit publication window with at most one cohort of writes."""

    flush: StreamCallEvidence
    preceding_text_writes: tuple[StreamCallEvidence, ...]
    enclosed_kernel_writes: tuple[ScopedResourceWork, ...]
    observed_fd_number: int
    relation: str = 'PRECEDING_TEXT_WRITES_MAY_CONTRIBUTE_TO_EXPLICIT_FLUSH'
    contents_or_bytes_attributed: bool = False
    descriptor_lifetime_proved: bool = False
    effects_closed: bool = False


def _line(source):
    return int(source.locator.removeprefix('#/lines/'))


def _backing(row):
    observation = decode(StreamBackingObservation, row['backing_observation'])
    if observation.status is not BackingStatus.OBSERVED or observation.chain_types != _BUFFERED_CHAIN:
        return None
    stat = observation.stat
    return observation.fd, stat.device, stat.inode, stat.mode, stat.rdev


def _compute(stream, scopes, work, stream_artifact, kernel_writes):
    scopes_by_attempt = {(scope.begin.caller, scope.begin.attempt_index): scope for scope in scopes
                         if scope.end.kind in (ScopeEventKind.C_RETURN, ScopeEventKind.C_RAISE)}
    work_by_scope = defaultdict(list)
    raw_writes_by_file = defaultdict(list)
    for source, fd in kernel_writes.items():
        raw_writes_by_file[source.artifact].append((_line(source), fd))
    for rows in raw_writes_by_file.values():
        rows.sort(key=lambda row: row[0])
    for row in work:
        if row.source in kernel_writes:
            work_by_scope[row.scope].append(row)

    streams = defaultdict(list)
    publications, frontiers = [], []

    def frontier(source, reason, **evidence):
        frontiers.append({'stream': source, 'reason': reason, **evidence})

    for row in stream['calls']:
        if row['receiver_type'] != _TEXT:
            continue
        source = ProviderFactReference(stream_artifact, '#/lines/' + str(row['sequence']))
        token = row['receiver_reference']
        method = row['contract']['method']
        if row['reference_identity_qualified'] is not True or not token.startswith('weak:'):
            if method in _METHODS:
                frontier(source, 'WRAPPER_LIFETIME_IDENTITY_OPEN')
            continue
        owner = decode(InvocationKey, row['invocation'])
        streams[owner.process_id, token].append((row, source, owner))

    for rows in streams.values():
        if len({owner.thread_id for _, _, owner in rows}) > 1:
            for row, source, _ in rows:
                if row['contract']['method'] in _METHODS:
                    frontier(source, 'WRAPPER_OBSERVED_IN_MULTIPLE_THREADS')
            continue

        pending, pending_backing = [], None

        def release_unassociated(reason):
            for item in pending:
                frontier(item.stream, reason)
            pending.clear()

        # Single-thread stream BEGIN order selects possible cohorts. Actual
        # per-thread source and OS marker order is checked on every relation.
        for row, source, owner in rows:
            method = row['contract']['method']
            if method not in _METHODS:
                release_unassociated('INTERVENING_OBSERVED_STREAM_OPERATION')
                continue
            scope = scopes_by_attempt.get((owner, row['attempt_index']))
            if (scope is None or encode(scope.call_attempt) != row['attempt_source']
                    or scope.end.kind is not ScopeEventKind.C_RETURN or row['completion'] != 'C_RETURN'):
                release_unassociated('INTERVENING_CALL_SCOPE_OR_COMPLETION_OPEN')
                frontier(source, 'SUCCESSFUL_NATIVE_CALL_SCOPE_OPEN')
                continue
            backing = _backing(row)
            if backing is None:
                release_unassociated('INTERVENING_BACKING_OBSERVATION_OPEN')
                frontier(source, 'EXACT_BUILTIN_FILE_BACKING_POINT_OPEN')
                continue
            evidence = StreamCallEvidence(scope, source)
            if pending:
                prior = pending[-1].scope
                if (scope.begin.caller != prior.end.caller
                        or scope.begin.native_tid != prior.end.native_tid
                        or scope.os_begin.artifact != prior.os_end.artifact
                        or _line(prior.os_end) >= _line(scope.os_begin)
                        or scope.begin.sequence <= prior.end.sequence
                        or backing != pending_backing):
                    release_unassociated('CALL_OWNER_ORDER_OR_BACKING_POINT_DIFFERS')

            fd = backing[0]
            if method == 'write':
                pending.append(evidence)
                pending_backing = backing
                continue

            enclosed = tuple(item for item in work_by_scope[scope.id] if kernel_writes[item.source] == fd)
            if not enclosed:
                release_unassociated('EXPLICIT_FLUSH_HAS_NO_ENCLOSED_MATCHING_WRITE')
                frontier(source, 'EXPLICIT_FLUSH_HAS_NO_ENCLOSED_MATCHING_WRITE')
            elif not pending:
                frontier(source, 'PRECEDING_TEXT_WRITES_OPEN', flush=evidence,
                         enclosed_kernel_writes=enclosed, observed_fd_number=fd)
            else:
                raw_writes = raw_writes_by_file[scope.os_begin.artifact]
                first = _line(pending[0].scope.os_begin)
                stop = _line(scope.os_begin)
                start = bisect_right(raw_writes, first, key=lambda row: row[0])
                end = bisect_right(raw_writes, stop, lo=start, key=lambda row: row[0])
                if any(number in (fd, None) for _, number in raw_writes[start:end]):
                    release_unassociated('EARLIER_KERNEL_WRITE_PREVENTS_BUFFER_PUBLICATION_GROUP')
                    frontier(source, 'EARLIER_KERNEL_WRITE_PREVENTS_BUFFER_PUBLICATION_GROUP',
                             flush=evidence, enclosed_kernel_writes=enclosed, observed_fd_number=fd)
                else:
                    publications.append(PossibleBufferedPublication(evidence, tuple(pending), enclosed, fd))
                    pending.clear()
        release_unassociated('EXPLICIT_PUBLICATION_NOT_OBSERVED')
    return publications, frontiers


def buffered_stream_publications(capture, *, project_root):
    """Replay an original capture and return a bounded, machine-readable query."""
    capture = Path(capture).resolve()
    coordinate_path = capture / 'capture-coordinates.json'
    coordinates = bind_capture_coordinates(capture, max_bytes=512_000_000, max_lines=2_000_000,
                                           profile_uri=str(coordinate_path))
    if coordinate_path.read_bytes() != coordinates.payload:
        raise ValueError('stored capture coordinates differ')
    report = dict(schema='scar.buffered-stream-publication-query', schema_version=1,
                  coordinates=encode(coordinates.artifact), publications=[], frontiers=[],
                  content_provenance='OPEN', implicit_publications='OPEN', first_stage_complete=False)
    membership = coordinates.report.get('stream_interface_membership')
    if (membership is None or 'resource_scope_membership' not in coordinates.report
            or membership['capability'] != 'SCAR_STREAM_BACKING_OBSERVATION_MEMBERSHIP_V1'):
        report['frontiers'].append({'reason': 'STREAM_BACKING_OR_RESOURCE_SCOPE_PROVIDER_OPEN'})
        return report
    declaration = membership['manifest']
    stream = read_stream_interfaces(declaration['uri'], expected_sha256=declaration['sha256'])
    if stream['stopped'] or stream['errors']:
        report['frontiers'].append({'reason': 'STREAM_PROVIDER_STOPPED_OR_FAILED'})
        return report
    trace = read_call_operands(capture / 'trace', project_root)
    scopes, _, _ = derive_resource_scopes(coordinates, capture, trace, include_positive_writes=True)
    kernel_writes = scopes['positive_write_fds']

    publications, frontiers = _compute(stream, scopes['scopes'], scopes['work'],
                                       decode(ProviderArtifact, membership['events']), kernel_writes)
    report['publications'] = encode(publications)
    report['frontiers'] = encode(frontiers)
    report['unscoped_positive_kernel_writes'] = sum(
        source in kernel_writes and reason == 'NO_BALANCED_ENCLOSING_SCOPE'
        for source, reason in scopes['unassigned'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True)
    parser.add_argument('--project-root', required=True)
    args = parser.parse_args()
    print(json.dumps(buffered_stream_publications(args.capture, project_root=args.project_root), indent=2))


if __name__ == '__main__':
    main()
