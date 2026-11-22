"""Read and check the retained prerequisites of a source unit."""

from scar.analysis.source_materials import _matches, _source_envelope


def check_replay_limit(document, name, limit, stage):
    value = document[name]
    if type(value) is not int or not 0 <= value <= limit:
        raise ValueError(stage + ' replay budget exceeds the requested ' + name + ' limit')


def read_source_prerequisites(reader, unit, max_nodes, max_tokens):
    source = reader.read(unit['artifacts']['source'])
    size = _source_envelope(unit, source)
    f0 = reader.read(unit['artifacts']['source_bytes'])
    if not _matches(unit, source, f0, size):
        raise ValueError('source stage requires matching original F0 bytes/text')
    f1 = reader.read(unit['artifacts']['source_tokens'])
    f2 = reader.read(unit['artifacts']['source_syntax'])
    check_replay_limit(f1, 'max_tokens', max_tokens, 'F1')
    check_replay_limit(f2, 'max_nodes', max_nodes, 'F2')
    return f0, f1, f2
