"""Bounded projections of GCC's experimental saved optimization records.

Pointers address the decompressed JSON, not byte offsets in the gzip member.
Diagnostics describe compiler reporting: neither a missed diagnostic nor its
count of siblings denotes a transformation, operation count or elapsed time.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
from hashlib import sha256
import io
import math
from pathlib import Path

from scar.ir.execution_recipe import _json, _reference


SCHEMA = 'scar.gcc-optimization-projection'
_MAX_SOURCE_BYTES = 64 * 2**20
_MAX_TEXT = 4096


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _pointer(parent, field):
    return parent + '/' + str(field).replace('~', '~0').replace('/', '~1')


def _text(value):
    return value if type(value) is str and len(value) <= _MAX_TEXT else None


def _raw(reference, maximum):
    reference = _reference(reference)
    _need(reference['size_bytes'] <= maximum, 'GCC projection artifact byte budget exceeded')
    with Path(reference['uri']).open('rb') as stream:
        raw = stream.read(maximum + 1)
    _need(len(raw) <= maximum, 'GCC projection artifact byte budget exceeded')
    _need(len(raw) == reference['size_bytes'] and sha256(raw).hexdigest() == reference['sha256'],
        'GCC projection retained artifact SHA/size differs: ' + reference['uri'])
    return reference, raw


def _preflight(value, maximum_nodes, maximum_depth):
    pending, count = [(value, 0)], 0
    while pending:
        item, depth = pending.pop()
        count += 1
        _need(count <= maximum_nodes and depth <= maximum_depth, 'GCC JSON node/depth budget exceeded')
        if type(item) is dict:
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is float:
            _need(math.isfinite(item), 'nonfinite GCC JSON value')
    return count


def _spans(values, source_lines):
    _need(type(values) in (tuple, list), 'flat stage span sequence required')
    result = []
    for index, row in enumerate(values):
        _need(type(row) is dict and {'id', 'start_line', 'end_line'} <= set(row)
            and not set(row)-{'id', 'start_line', 'end_line', 'mapping_status'}
            and type(row['id']) is str and row['id'] and len(row['id']) <= _MAX_TEXT
            and type(row['start_line']) is int and type(row['end_line']) is int
            and 1 <= row['start_line'] <= row['end_line'] <= source_lines,
            'stage spans must name inclusive lines of the verified source artifact')
        if 'mapping_status' in row:
            _need(type(row['mapping_status']) is str, 'stage mapping status must be text')
        result.append(dict(index=index, **deepcopy(row)))
    return result


class _Locations:
    def __init__(self, source, directory, spans, source_lines):
        self.source_path = Path(source['uri']).resolve()
        self.directory, self.spans, self.source_lines = directory, spans, source_lines
        self.files, self.locations, self._files = [], [], {}

    def file(self, spelling):
        if spelling in self._files:
            return self.files[self._files[spelling]]
        resolved = None
        if not spelling or '\x00' in spelling or len(spelling) > _MAX_TEXT:
            status = 'INVALID_OR_UNBOUNDED_PATH'
        elif spelling.startswith('<') and spelling.endswith('>'):
            status = 'COMPILER_VIRTUAL_FILE'
        else:
            path = Path(spelling)
            try:
                if path.is_absolute():
                    resolved = str(path.resolve())
                elif self.directory is not None:
                    resolved = str((self.directory/path).resolve())
                if resolved is None:
                    status = 'RELATIVE_PATH_WITHOUT_COMPILATION_DIRECTORY'
                else:
                    status = 'SOURCE_REF_PATH_MATCH' if Path(resolved) == self.source_path else 'DIFFERENT_SOURCE_PATH'
            except (OSError, ValueError, RuntimeError):
                status = 'UNRESOLVABLE_PATH'
        row = dict(id='file:' + str(len(self.files)), observed_path=_text(spelling),
            resolved_path=resolved, source_identity_status=status)
        self._files[spelling] = len(self.files)
        self.files.append(row)
        return row

    def scan(self, value, pointer, owner, role):
        """Find explicit locations; compiler impl_location is excluded upstream."""
        result = []
        pending = [(value, pointer)]
        while pending:
            item, position = pending.pop()
            if type(item) is dict:
                if 'file' in item and 'line' in item:
                    file = self.file(item['file']) if type(item['file']) is str else None
                    line, column = item.get('line'), item.get('column')
                    valid_line = type(line) is int and 1 <= line <= self.source_lines
                    matches = [row['index'] for row in self.spans if valid_line
                        and file is not None and file['source_identity_status'] == 'SOURCE_REF_PATH_MATCH'
                        and row['start_line'] <= line <= row['end_line']]
                    if file is None or type(line) is not int or line < 1:
                        status = 'MALFORMED_LOCATION'
                    elif file['source_identity_status'] != 'SOURCE_REF_PATH_MATCH':
                        status = file['source_identity_status']
                    elif not valid_line:
                        status = 'LINE_OUTSIDE_VERIFIED_SOURCE'
                    else:
                        status = 'MAPPED_TO_STAGE_SPANS' if matches else 'SOURCE_LINE_WITHOUT_STAGE_SPAN'
                    row = dict(id=position, diagnostic=owner, role=role,
                        file_id=None if file is None else file['id'],
                        line=line if type(line) is int else None,
                        column=column if type(column) is int else None,
                        mapping_status=status, stage_span_indices=matches)
                    self.locations.append(row)
                    result.append(position)
                pending.extend((child, _pointer(position, key)) for key, child in reversed(list(item.items()))
                    if key != 'impl_location' and type(child) in (dict, list))
            elif type(item) is list:
                pending.extend((item[index], _pointer(position, index)) for index in reversed(range(len(item)))
                    if type(item[index]) in (dict, list))
        return result


def _passes(rows):
    result, by_wire = [], defaultdict(list)
    pending = [(row, _pointer('#/1', index), None) for index, row in reversed(list(enumerate(rows)))]
    while pending:
        node, pointer, parent = pending.pop()
        row = dict(id=pointer, parent=parent, children=[], record_type='GCC_PASS')
        if type(node) is not dict:
            row.update(record_type='OPAQUE_PASS_NODE', opaque_pointer=pointer)
        else:
            row.update(wire_id=_text(node.get('id')), name=_text(node.get('name')),
                pass_type=_text(node.get('type')), number=node.get('num') if type(node.get('num')) is int else None,
                optgroups=[item for item in node.get('optgroups', []) if type(item) is str]
                    if type(node.get('optgroups')) is list else None,
                opaque_fields={key:_pointer(pointer, key) for key in node
                    if key not in ('id', 'name', 'type', 'num', 'optgroups', 'children')})
            if row['wire_id'] is not None:
                by_wire[row['wire_id']].append(pointer)
            children = node.get('children')
            if type(children) is list:
                row['children'] = [_pointer(_pointer(pointer, 'children'), index) for index in range(len(children))]
                pending.extend((children[index], row['children'][index], pointer) for index in reversed(range(len(children))))
            elif 'children' in node:
                row['opaque_fields']['children'] = _pointer(pointer, 'children')
        result.append(row)
    return result, by_wire


def _count(value, pointer):
    if type(value) is not dict:
        return dict(raw_pointer=pointer, status='OPAQUE_COMPILER_COUNT', value=None, quality=None)
    number, quality = value.get('value'), _text(value.get('quality'))
    status = 'UNINITIALIZED' if quality == 'uninitialized' else (
        'ESTIMATED' if quality is not None and quality.startswith('guessed') else 'COMPILER_REPORTED_QUALITY')
    return dict(raw_pointer=pointer, value=number if type(number) in (int, float) else None,
        quality=quality, status=status)


def _diagnostics(rows, passes, locations):
    result = []
    pending = [(row, _pointer('#/2', index), None) for index, row in reversed(list(enumerate(rows)))]
    while pending:
        node, pointer, parent = pending.pop()
        row = dict(id=pointer, parent=parent, children=[], record_type='GCC_DIAGNOSTIC',
            location_ids=[], primary_location_ids=[], inlining_chain=[], stage_span_indices=[], stage_ids=[])
        if type(node) is not dict:
            row.update(record_type='OPAQUE_DIAGNOSTIC', opaque_pointer=pointer, kind=None,
                function=None, pass_ids=[], count=None, mapping_status='NO_SOURCE_LOCATION')
            result.append(row)
            continue
        wire_pass = _text(node.get('pass'))
        row.update(kind=_text(node.get('kind')), function=_text(node.get('function')),
            wire_pass_id=wire_pass, pass_ids=list(passes.get(wire_pass, ())),
            count=_count(node['count'], _pointer(pointer, 'count')) if 'count' in node else None,
            raw_message_pointer=_pointer(pointer, 'message') if 'message' in node else None,
            compiler_implementation_location_pointer=_pointer(pointer, 'impl_location') if 'impl_location' in node else None,
            opaque_fields={key:_pointer(pointer, key) for key in node if key not in
                ('kind', 'function', 'pass', 'count', 'message', 'impl_location', 'location', 'inlining_chain', 'children')})
        for key in ('kind', 'function', 'pass'):
            if key in node and _text(node[key]) is None:
                row['opaque_fields'][key] = _pointer(pointer, key)
        start = len(locations.locations)
        if 'location' in node:
            row['primary_location_ids'] = locations.scan(node['location'], _pointer(pointer, 'location'), pointer, 'PRIMARY')
            if not row['primary_location_ids']:
                row['opaque_fields']['location'] = _pointer(pointer, 'location')
        chain = node.get('inlining_chain')
        if type(chain) is list:
            for index, frame in enumerate(chain):
                frame_pointer = _pointer(_pointer(pointer, 'inlining_chain'), index)
                row['inlining_chain'].append(dict(raw_pointer=frame_pointer,
                    fndecl=_text(frame.get('fndecl')) if type(frame) is dict else None,
                    location_ids=locations.scan(frame, frame_pointer, pointer, 'INLINING_CHAIN')))
        elif 'inlining_chain' in node:
            row['opaque_fields']['inlining_chain'] = _pointer(pointer, 'inlining_chain')
        if 'message' in node:
            locations.scan(node['message'], _pointer(pointer, 'message'), pointer, 'MESSAGE_CONTENT')
        for key in row['opaque_fields']:
            # Malformed location/chain may still contain explicit nested sites.
            if key not in ('location', 'inlining_chain'):
                locations.scan(node[key], _pointer(pointer, key), pointer, 'OPAQUE_DIAGNOSTIC_FIELD')
        projected = locations.locations[start:]
        row['location_ids'] = [item['id'] for item in projected]
        row['stage_span_indices'] = sorted({index for item in projected for index in item['stage_span_indices']})
        row['stage_ids'] = list(dict.fromkeys(locations.spans[index]['id'] for index in row['stage_span_indices']))
        row['mapping_status'] = 'MAPPED_TO_STAGE_SPANS' if row['stage_span_indices'] else (
            'SOURCE_LOCATION_WITHOUT_STAGE' if any(item['mapping_status'] == 'SOURCE_LINE_WITHOUT_STAGE_SPAN'
                for item in projected) else 'UNMAPPED_LOCATIONS' if projected else 'NO_SOURCE_LOCATION')
        children = node.get('children')
        if type(children) is list:
            row['children'] = [_pointer(_pointer(pointer, 'children'), index) for index in range(len(children))]
            pending.extend((children[index], row['children'][index], pointer) for index in reversed(range(len(children))))
        elif 'children' in node:
            row['opaque_fields']['children'] = _pointer(pointer, 'children')
        result.append(row)
    return result


def parse_gcc_optimization_record(record_ref, *, source_ref, stage_spans=(), compilation_directory=None,
        max_compressed_bytes=16*2**20, max_decoded_bytes=64*2**20, max_nodes=1_000_000, max_depth=256):
    """Project GCC format 1 metadata without compiling, loading or running code.

    The source ref supplies byte identity; the record itself has no source
    digest.  Its historical compilation/source binding belongs to the caller's
    collector.  Stage associations are location citations, not executions.
    """
    for limit in (max_compressed_bytes, max_decoded_bytes, max_nodes, max_depth):
        _need(type(limit) is int and limit > 0, 'positive GCC parser budgets required')
    record_ref, compressed = _raw(record_ref, max_compressed_bytes)
    source_ref, source = _raw(source_ref, _MAX_SOURCE_BYTES)
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
            decoded = stream.read(max_decoded_bytes + 1)
    except (OSError, EOFError) as error:
        raise ValueError('invalid gzip GCC optimization record') from error
    _need(len(decoded) <= max_decoded_bytes, 'GCC decoded byte budget exceeded')
    try:
        document = _json(decoded)
    except (ValueError, RecursionError, UnicodeError) as error:
        raise ValueError('invalid or too deeply nested GCC record JSON') from error
    nodes = _preflight(document, max_nodes, max_depth)
    _need(type(document) is list and len(document) == 3 and type(document[0]) is dict
        and type(document[1]) is list and type(document[2]) is list and document[0].get('format') == '1',
        'supported GCC experimental format 1 [header, pass tree, diagnostics] required')
    if compilation_directory is not None:
        directory = Path(compilation_directory)
        _need(directory.is_absolute(), 'absolute compilation directory required for relative GCC paths')
        directory = directory.resolve()
    else:
        directory = None
    source_lines = (source.count(b'\n') + source.count(b'\r') - source.count(b'\r\n')
        + int(bool(source) and not source.endswith((b'\n', b'\r'))))
    spans = _spans(stage_spans, source_lines)
    locations = _Locations(source_ref, directory, spans, source_lines)
    passes, by_wire = _passes(document[1])
    diagnostics = _diagnostics(document[2], by_wire, locations)
    kinds, pass_kinds = Counter(), defaultdict(Counter)
    for row in diagnostics:
        kind = row.get('kind') or 'OPAQUE_OR_MISSING_KIND'
        kinds[kind] += 1
        for pass_id in row['pass_ids']:
            pass_kinds[pass_id][kind] += 1
    header = document[0]
    generator = header.get('generator')
    # Recheck both original byte identities after resolving and projecting.
    _raw(record_ref, max_compressed_bytes)
    _raw(source_ref, _MAX_SOURCE_BYTES)
    return dict(schema=SCHEMA, schema_version=1, record_ref=record_ref, source_ref=source_ref,
        observed_wire_header=dict(raw_pointer='#/0', format=header['format'],
            generator={key:_text(generator.get(key)) for key in ('name', 'pkgversion', 'version', 'target')}
                if type(generator) is dict else None,
            generator_pointer='#/0/generator' if 'generator' in header else None,
            gcc_wire_schema=header.get('schema') if type(header.get('schema')) is str else None,
            opaque_fields={key:_pointer('#/0', key) for key in header if key not in ('format', 'generator', 'schema')}),
        experimental_wire_format=True, raw_pointer_interpretation='RFC6901_ON_DECOMPRESSED_GZIP_JSON',
        pass_identity_scope='RECORD_ARTIFACT_AND_JSON_POINTER_WIRE_ADDRESSES_ARE_NOT_STABLE_IDS',
        compilation_directory=None if directory is None else str(directory), source_line_count=source_lines,
        stage_spans=spans, files=locations.files, locations=locations.locations, passes=passes, diagnostics=diagnostics,
        summary=dict(pass_nodes=len(passes), top_level_diagnostics=len(document[2]), diagnostic_nodes=len(diagnostics),
            diagnostic_kinds=dict(sorted(kinds.items())), diagnostic_kinds_by_pass={key:dict(sorted(value.items()))
                for key,value in sorted(pass_kinds.items())},
            mapped_diagnostics=sum(bool(row['stage_span_indices']) for row in diagnostics),
            unmapped_diagnostics=sum(not row['stage_span_indices'] for row in diagnostics),
            diagnostics_without_primary_location=sum(not row['primary_location_ids'] for row in diagnostics),
            diagnostics_without_source_location=sum(not row['location_ids'] for row in diagnostics)),
        budgets=dict(max_compressed_bytes=max_compressed_bytes, max_decoded_bytes=max_decoded_bytes,
            max_nodes=max_nodes, max_depth=max_depth, decoded_bytes=len(decoded), json_nodes=nodes),
        qualifications=dict(source_bytes_verified=True, historical_compilation_source_identity_proven=False,
            stage_spans_are_caller_supplied=True, compiler_counts_are_runtime_observations=False,
            diagnostic_counts_are_transformation_or_operation_counts=False, success_diagnostics_are_rewrite_proofs=False,
            all_compiler_transformations_closed=False, elapsed_time_estimated=False,
            compiler_invoked=False, target_program_executed=False))


__all__ = ['parse_gcc_optimization_record']
