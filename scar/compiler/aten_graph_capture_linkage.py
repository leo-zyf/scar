"""Rebind C++20 ATen declarations to original scoped graph occurrences."""

from pathlib import Path

from scar.codegen.aten_capture_link import select_aten_capture
from scar.ir.execution_recipe import artifact_reference, read_artifact


_LINKAGE = 'scar.aten-provider-linkage'
_CAPTURE_SOURCE = 'scar.aten-capture-source'
_HEADER = 'scar.aten-operator-cxx-header'
_COMPARABLE_COORDINATE_FIELDS = ('schema', 'sha256', 'size_bytes')
_OCCURRENCE_FIELDS = ('call_attempt', 'call_site', 'operand', 'scope', 'kineto', 'schema')


def verify_graph_capture_linkage(linkage, expected_sha256, native):
    """Return original evidence receipts for the freshly decoded graph scopes."""
    linkage = Path(linkage).resolve(strict=True)
    linkage_ref = artifact_reference(linkage, _LINKAGE)
    if linkage_ref['sha256'] != expected_sha256:
        raise ValueError('ATen graph linkage differs from its declared digest')
    binding = read_artifact(linkage_ref)
    if (binding['schema_version'], binding['status'], binding['cxx_standard']) != (1, 'EXTERNAL_CPP_CALL_LINKED', 20):
        raise ValueError('retained C++20 ATen linkage required')

    source_ref = artifact_reference(linkage.with_name('capture-source.json'), _CAPTURE_SOURCE)
    source = read_artifact(source_ref)
    if source.get('linkage_report') != linkage_ref:
        raise ValueError('captured operator source differs from the declared linkage')
    if (native.get('schema'), native.get('schema_version')) != ('scar.aten-scoped-ports', 1):
        raise ValueError('decoded scoped ATen graph ports required')
    if any(source.get(key) != native[key] for key in ('run', 'process', 'provider_revision')):
        raise ValueError('ATen linkage and graph ports describe different original capture identities')
    # The in-memory and persisted coordinate receipts can have different URIs.
    coordinates = source['coordinate_binding']
    if any(coordinates[key] != native['coordinate_binding'][key] for key in _COMPARABLE_COORDINATE_FIELDS):
        raise ValueError('ATen linkage and graph ports have different coordinate bytes')
    if binding['provider']['revision'] != source['provider_revision']:
        raise ValueError('C++20 ATen linkage uses a different provider revision from the original capture')

    capture_model = source['capture_model']
    graph_operators = native['operators']
    attempts = tuple(dict.fromkeys(row['call_attempt'] for row in graph_operators))
    replayed = select_aten_capture(capture_model['uri'], expected_sha256=capture_model['sha256'],
                                  call_attempts=attempts)
    if len(replayed['occurrences']) != len(graph_operators):
        raise ValueError('linked ATen occurrence count differs from the graph')
    entries = {row['schema']: row['entry'] for row in binding['operators']}
    if (len(entries) != len(binding['operators'])
            or len(set(entries.values())) != len(entries)
            or set(entries) != {row['schema'] for row in graph_operators}):
        raise ValueError('ATen linkage declarations differ from the original graph operators')
    for occurrence, op in zip(replayed['occurrences'], graph_operators, strict=True):
        if (occurrence['source'] != op['eto']
                or any(occurrence[key] != op[key] for key in _OCCURRENCE_FIELDS)):
            raise ValueError('ATen linkage occurrence differs from the original graph operator')
        occurrence['entry'] = entries[op['schema']]
    # Includes the independent ETO provider node IDs recovered by select_aten_capture.
    if dict(replayed, linkage_report=linkage_ref) != source:
        raise ValueError('retained ATen capture linkage differs from cold replay')

    header_ref = artifact_reference(linkage.with_name('operators.hpp'), _HEADER)
    if {key: header_ref[key] for key in ('sha256', 'size_bytes')} != binding['artifacts']['operators.hpp']:
        raise ValueError('typed ATen operator header differs from the retained linkage')
    return dict(linkage_report=linkage_ref, capture_source=source_ref, capture_model=capture_model,
                operators_header=header_ref, occurrences=replayed['occurrences'])
