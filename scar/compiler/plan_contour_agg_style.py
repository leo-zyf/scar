"""Connect live contour range and color production to exact AGG argument slots."""

from __future__ import annotations

import struct

from scar.analysis.source_agg_scene_layout import read_tape
from scar.ir.execution_recipe import read_artifact


def _require(condition, detail):
    if not condition:
        raise ValueError('contour AGG style: ' + detail)


def _span_bytes(raw, span):
    return raw[span['start']:span['end']]


def _path_slot(sample, tape):
    _require(len(sample['collections']) == 1, 'a single qualified visible contour collection is required')
    collection = sample['collections'][0]
    command = tape['commands'][collection['draw_ordinal']]
    path_count = len(sample['intervals'])
    thresholds = collection['simplify_threshold']
    _require(command['fields']['paths']['span'] == collection['path_span']
             and collection['bands'] == list(range(path_count))
             and len(thresholds) == path_count
             and len({struct.pack('<d', value) for value in thresholds}) == 1,
             'live levels require all ordered bands and one stable closed-path threshold')
    return dict(
        operation='EMIT_FILLED_BAND_PATHS', draw_ordinal=command['draw_ordinal'],
        field='paths', span=collection['path_span'], simplify_threshold=thresholds[0],
    )


def _color_slots(tape, raw, sample, style):
    count = len(sample['intervals'])
    slots, seen = [], set()
    for witness in style['collections']:
        expected = witness['mapped']['rgba']
        _require(expected['dtype'] == '<f8' and expected['shape'] == [count, 4],
                 'mapped color witness differs from the band arity or representation')
        expected_payload = bytes.fromhex(expected['data_hex'])
        _require(len(expected_payload) == count * 32, 'mapped palette payload length differs')
        palette = struct.pack('<II', count, 4) + expected_payload
        draw = witness['draw_ordinal']
        _require(type(draw) is int and 0 <= draw < len(tape['commands']) and draw not in seen,
                 'color producer requires a unique original collection ordinal')
        seen.add(draw)
        command = tape['commands'][draw]
        fields = command['fields']
        _require(command['operator'] == 'draw_path_collection'
                 and fields['paths']['count'] == witness['path_count']
                 and fields['facecolors']['shape'] == [count, 4]
                 and _span_bytes(raw, fields['facecolors']['span']) == palette,
                 'original face colors do not match the live contour color producer')
        slots.append(dict(operation='EMIT_BAND_FACE_COLORS', draw_ordinal=draw,
                          field='facecolors', span=fields['facecolors']['span']))
    return slots


def _level_bytes(sample):
    levels = [sample['intervals'][0]['lower'], *(row['upper'] for row in sample['intervals'])]
    return struct.pack(f'<{len(levels)}d', *levels)


def _title_slots(tape, raw, title):
    commands = tape['commands']
    draws = title['draws']
    indices = [row['ordinal'] for row in draws]
    _require(indices and 0 <= indices[0] <= indices[-1] < len(commands)
             and indices == list(range(indices[0], indices[-1] + 1)),
             'title glyph draws require one ordered command span')
    selected = commands[indices[0]:indices[-1] + 1]
    prefixes = set()
    for row, command in zip(draws, selected, strict=True):
        fields = command['fields']
        start, end = row['span']
        _require(command['operator'] == 'draw_text_image'
                 and command['span'] == dict(start=start, end=end)
                 and _span_bytes(raw, fields['angle']['span']) == struct.pack('<d', 0.0),
                 'live title requires an original axis-aligned glyph command')
        prefixes.add(raw[start:fields['bitmap']['span']['start']])
    _require(len(prefixes) == 1, 'title glyphs do not share the original graphics state')
    prefix, = prefixes
    return [
        dict(operation='UPDATE_DRAW_COUNT', span=dict(start=24, end=28),
             unmodified_draws=len(commands) - len(indices)),
        dict(operation='EMIT_TITLE_GLYPHS', span=dict(start=draws[0]['span'][0], end=draws[-1]['span'][1]),
             command_prefix=prefix.hex(), draw_ordinal=indices[0]),
    ]


def plan_sample(sample, style, *, title=None):
    """Partition one bound original scene by native range-dependent fields."""
    observed = style['levels']
    levels = bytes.fromhex(observed['data_hex'])
    _require(observed['dtype'] == '<f8' and observed['shape'] == [len(sample['intervals']) + 1]
             and len(levels) == observed['shape'][0] * 8,
             'original locator level shape differs')
    first, = struct.unpack_from('<d', levels)
    zmin = style['bounds']['zmin']
    if first == zmin:
        levels = struct.pack('<d', first - 1.0) + levels[8:]
    _require(style['frame_index'] == sample['frame_index'] and levels == _level_bytes(sample),
             'original contour provider does not consume the qualified locator levels')
    raw = read_artifact(sample['scene'], json_document=False)
    tape = read_tape(raw)
    geometry = _path_slot(sample, tape)
    colors = _color_slots(tape, raw, sample, style)
    _require(any(row['draw_ordinal'] == geometry['draw_ordinal'] for row in colors),
             'selected contour geometry has no matched runtime palette')
    title_slots = _title_slots(tape, raw, title) if title is not None else []
    selected = sorted([geometry, *colors, *title_slots], key=lambda row: row['span']['start'])
    steps, cursor = [], 0
    for slot in selected:
        start, end = slot['span']['start'], slot['span']['end']
        _require(cursor <= start < end <= len(raw), 'overlapping original draw argument slots')
        if start != cursor:
            steps.append(dict(operation='ORIGINAL_RESIDUAL', span=dict(start=cursor, end=start)))
        steps.append(slot)
        cursor = end
    if cursor < len(raw):
        steps.append(dict(operation='ORIGINAL_RESIDUAL', span=dict(start=cursor, end=len(raw))))
    return dict(frame_index=sample['frame_index'], surface=tape['surface'], steps=steps,
                original_scene=sample['scene'], status='CONDITIONAL_EXPLICIT_TRIAL_ONLY',
                default_backend='NO_OP', whole_entry_authorized=False)
