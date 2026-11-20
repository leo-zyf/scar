"""Plan one original contour result into its exact native AGG draw slot."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import struct
from types import CodeType

from scar.analysis.source_agg_scene_layout import read_tape
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.implementation_plan import CostEstimate
from scar.ir.record_codec import encode
from scar.trace.code_snapshot import code_key
from scar.trace.agg_scene import SCHEMA as AGG_OBSERVATION
from scar.trace.contour_scene import INPUT_MAGIC, OUTPUT_MAGIC, SCHEMA as OBSERVATION

SCHEMA = 'scar.contour-scene-plan'
_LIMIT = 64_000_000
_MAX_BANDS = 16_384
_NATIVE_CODES = {1, 2, 79}


def _require(condition, message):
    if not condition:
        raise ValueError('contour scene: ' + message)


def _reference(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _at(reference, locator):
    return dict(artifact=reference, locator=locator)


def _digest(raw):
    return sha256(raw).hexdigest()


def _closed_polygons(codes):
    if not codes or not set(codes) <= _NATIVE_CODES:
        return False
    length = 0
    for code in codes:
        if code == 1:
            if length:
                return False
            length = 1
        elif code == 2:
            if not length:
                return False
            length += 1
        else:
            if length < 3:
                return False
            length = 0
    return length == 0


def _source_code(source):
    reference = source['material']
    raw = read_artifact(reference, json_document=False)
    pending = [compile(raw, reference['uri'], 'exec', dont_inherit=True)]
    matches = 0
    while pending:
        code = pending.pop()
        if (source.get('qualname') is None or code.co_qualname == source['qualname']) and code_key(code) == source['code_key']:
            matches += 1
        pending.extend(item for item in code.co_consts if type(item) is CodeType)
    _require(matches == 1, 'original contour producer CodeID does not resolve in retained current source')


def _input(record):
    reference = record['input']
    raw = read_artifact(reference, json_document=False)
    _require(reference['schema'] == 'scar.contour-scene-input'
             and 64 <= len(raw) <= _LIMIT and raw[:8] == INPUT_MAGIC,
             'original contour input uses a different wire layout')
    rows, cols, levels, corner_mask, x_chunk, y_chunk, masked = struct.unpack_from('<7Q', raw, 8)
    _require(2 <= rows and 2 <= cols and 2 <= levels <= _MAX_BANDS + 1
             and x_chunk <= cols - 1 and y_chunk <= rows - 1
             and corner_mask in (0, 1) and masked in (0, 1), 'unsupported contour input geometry or options')
    cells = rows * cols
    level_offset = 64 + 24 * cells
    end = level_offset + levels * 8
    _require(len(raw) == end + cells * masked, 'original contour lattice byte span differs')
    offsets = dict(x=64, y=64 + 8 * cells, z=64 + 16 * cells, levels=level_offset)
    if masked:
        offsets['mask'] = end
        _require(all(value <= 1 for value in memoryview(raw)[end:]), 'invalid original contour mask values')
    grid = dict(shape=[rows, cols], dtype='<f8', offsets=offsets)
    observed_grid = dict(record['grid'])
    lineage = observed_grid.pop('lineage', None)
    _require(grid == observed_grid, 'original contour grid metadata differs from exact provider inputs')
    if lineage is not None:
        _source_code(dict(material=lineage['source'], code_key=lineage['code_key']))
    for name in ('x', 'y', 'z'):
        start = offsets[name]
        _require(all(math.isfinite(value) for (value,) in struct.iter_unpack(
                 '<d', memoryview(raw)[start:start + 8 * cells])), 'nonfinite mesh input outside selected contour trial')
    provider = record['provider']
    _require(provider['name'] == 'contourpy' and provider['algorithm'] == 'mpl2014'
             and provider['fill_type'] == 'OuterCode' and provider['line_type'] == 'SeparateCode'
             and type(provider['version']) is str and bool(provider['version'])
             and provider['chunk_size'] == [y_chunk, x_chunk] and provider['corner_mask'] is bool(corner_mask)
             and provider['masked'] is bool(masked) and type(provider['thread_count']) is int
             and provider['thread_count'] == 1, 'exact original MPL2014 filled provider/protocol required')
    _require(provider['binary']['schema'] == 'scar.native-provider-library', 'provider library provenance unavailable')
    read_artifact(provider['binary'], json_document=False)
    ordered = struct.unpack_from(f'<{levels}d', raw, level_offset)
    _require(all(math.isfinite(a) and math.isfinite(b) and a < b for a, b in zip(ordered[:-1], ordered[1:])),
             'ordered finite nonempty contour intervals required')
    observed = record['intervals']
    _require(len(observed) == levels - 1 and all(
        type(row['lower']) is float and type(row['upper']) is float
        and struct.pack('<2d', row['lower'], row['upper']) == struct.pack('<2d', lower, upper)
        for row, lower, upper in zip(observed, ordered[:-1], ordered[1:], strict=True)),
        'actual filled-call interval sequence differs from input wire')
    return grid, observed, provider


class _PolygonWire:
    def __init__(self, raw):
        self.raw = memoryview(raw)
        self.cursor = 0

    def take(self, length):
        _require(0 <= length <= len(self.raw) - self.cursor, 'truncated contour polygon witness')
        start = self.cursor
        self.cursor += length
        return self.raw[start:self.cursor]

    def count(self, maximum):
        number, = struct.unpack('<Q', self.take(8))
        _require(number <= maximum, 'contour polygon witness exceeds bounded count')
        return number


def _output(reference, n_bands):
    _require(reference['schema'] == 'scar.contour-scene-paths', 'original provider output reference required')
    raw = read_artifact(reference, json_document=False)
    wire = _PolygonWire(raw)
    _require(len(raw) <= _LIMIT and wire.take(len(OUTPUT_MAGIC)) == OUTPUT_MAGIC
             and wire.count(_MAX_BANDS) == n_bands, 'original provider band count differs')
    bands = []
    for _ in range(n_bands):
        vertices, codes, polygon_count = bytearray(), bytearray(), wire.count(len(raw) // 8)
        for _ in range(polygon_count):
            count = wire.count(len(raw) // 17)
            coordinates, commands = wire.take(16 * count), wire.take(count)
            _require(count >= 4 and _closed_polygons(commands)
                     and all(math.isfinite(value) for (value,) in struct.iter_unpack('<d', coordinates)),
                     'provider result must contain exact finite closed OuterCode polygons')
            vertices.extend(coordinates)
            codes.extend(commands)
        bands.append((vertices, codes))
    _require(wire.cursor == len(raw), 'trailing contour polygon witness bytes')
    return bands


def _block(data, span):
    return memoryview(data)[span['start']:span['end']]


def _collection(row, command, raw, bands):
    index = row['draw_ordinal']
    fields = command['fields']
    _require(command['operator'] == 'draw_path_collection' and command['draw_ordinal'] == index
             and row['path_span'] == fields['paths']['span'], 'contour result has no exact original path slot')
    slot = fields['paths']
    band_indices = row['bands']
    thresholds = row['simplify_threshold']
    simplifies = row['path_simplifies']
    count = slot['count']
    _require(0 < count <= len(bands) and len(band_indices) == len(thresholds) == len(simplifies) == count
             and all(type(item) is int and 0 <= item < len(bands) for item in band_indices)
             and list(sorted(set(band_indices))) == band_indices and all(flag is False for flag in simplifies)
             and all(type(value) is float and 0. <= value <= 1. for value in thresholds),
             'path slot must preserve its exact ordered contour bands and disabled simplification')
    # Keep RGBA, transforms and other AGG arguments in the original residual.
    facecolor = fields['facecolors']['shape']
    _require(facecolor in ([1, 4], [count, 4]) and fields['antialias']['shape'][0] >= 1,
             'preserved collection styling has no supported original band alignment')
    geometry = bytearray(struct.pack('<I', count))
    for band, threshold, path in zip(band_indices, thresholds, command['paths'], strict=True):
        points, codes = bands[band]
        _require(path['vertices']['shape'] == ([len(codes), 2] if codes else [0, 0])
                 and path['codes']['shape'] == [len(codes)]
                 and path['simplification']['enabled'] is False,
                 'controller collection path shape differs from original provider polygon')
        expected = (struct.pack('<II', len(codes), 2 if codes else 0) + points
                    + struct.pack('<I', len(codes)) + codes + struct.pack('<Bd', 0, threshold))
        _require(expected == _block(raw, path['span']), 'closed polygons differ from original drawn path bytes')
        geometry.extend(expected)
    _require(geometry == _block(raw, slot['span']), 'original collection path span differs from reconstructed geometry')
    suffix = {'start': slot['span']['end'], 'end': command['span']['end']}
    return dict(**row, original_path_block_sha256=_digest(geometry),
                residual_draw_suffix_sha256=_digest(_block(raw, suffix)),
                protocol=dict(kind='MPL2014_OUTER_CODE_TO_PATH_TABLE', path_count=count,
                              facecolor_shape=facecolor, empty_bands_preserved=True))


def _frame(agg_ref, agg, index):
    _require(type(index) is int and 0 <= index < len(agg['frames']), 'invalid original contour frame index')
    row = agg['frames'][index]
    root = Path(agg_ref['uri']).parent
    _require(row['index'] == index and all(Path(row[name]).name == row[name] for name in ('scene', 'raster')),
             'original frame ordinal or member name differs')
    scene = _reference(root / row['scene'], 'scar.agg-scene-tape')
    raster = _reference(root / row['raster'], 'scar.packed-rgba8')
    _require(scene['sha256'] == row['scene_sha256'] and scene['size_bytes'] == row['scene_bytes']
             and raster['sha256'] == row['raster_sha256'], 'retained draw or raster byte identity differs')
    raw = read_artifact(scene, json_document=False)
    tape = read_tape(raw)
    surface = tape['surface']
    _require(list(surface.values()) == row['geometry'] and raster['size_bytes'] == 4 * surface['width'] * surface['height']
             and dict(Counter(command['operator'] for command in tape['commands'])) == row['operations'],
             'scene writer tape topology, surface, or raster geometry differs')
    return scene, raster, raw, tape


def _derive(observation_ref):
    observation = read_artifact(observation_ref)
    _require(observation['schema_version'] == 1 and observation['capture_is_diagnostic'] is True
             and observation['scene_control_resolved'] is False and observation['samples'],
             'requires scoped original contour-to-draw observations')
    agg_ref = observation['agg_scene']
    _require(agg_ref['schema'] == AGG_OBSERVATION, 'paired original AGG observation required')
    agg = read_artifact(agg_ref)
    _require(agg['schema_version'] in (1, 2) and agg['capture_is_diagnostic'] is True
             and agg['performance_measured'] is False, 'expected diagnostic original renderer evidence')
    samples, used_sources, targets = [], set(), set()
    for index, row in enumerate(observation['samples']):
        source_key = (row['source']['material']['sha256'], row['source']['code_key'])
        if source_key not in used_sources:
            _source_code(row['source'])
            used_sources.add(source_key)
        grid, intervals, provider = _input(row)
        bands = _output(row['expected_paths'], len(intervals))
        frame = row['frame_index']
        scene, raster, raw, tape = _frame(agg_ref, agg, frame)
        collections = []
        for item in row['collections']:
            draw = item['draw_ordinal']
            _require(type(draw) is int and 0 <= draw < len(tape['commands']) and (frame, draw) not in targets,
                     'original frame/draw ordinal must have a unique contour producer')
            targets.add((frame, draw))
            collections.append(_collection(item, tape['commands'][draw], raw, bands))
        _require(bool(collections), 'sample has no exact original filled draw collection')
        samples.append(dict(
            source_observation=_at(observation_ref, f'#/samples/{index}'), source=row['source'],
            frame_index=frame, scene=scene, raster=raster, grid=grid, provider=provider, intervals=intervals,
            input=row['input'], expected_paths=row['expected_paths'], collections=collections,
        ))
    cost = CostEstimate(None, None, None, 'UNKNOWN', 'SOURCE_FILLED_CONTOUR_TO_ORIGINAL_AGG_DRAW',
                        (_at(observation_ref, '#/samples'), _at(agg_ref, '#/frames')),
                        ('NO_MATCHED_CLEAN_CONTOUR_RENDER_SERVICE_COST',))
    return dict(
        schema=SCHEMA, schema_version=1, observation=observation_ref, agg_scene=agg_ref, samples=samples,
        status='CONDITIONAL_EXPLICIT_TRIAL_ONLY', default_backend='NO_OP',
        replacement='REGENERATE_SELECTED_PATH_SLOTS_WITH_ORIGINAL_NATIVE_PROVIDER',
        runtime_admission='EXACT_RECORDED_PROVIDER_OPTIONS_MESH_AND_ORDERED_INTERVAL_BYTES',
        residual_scope='ALL_ORIGINAL_NON_SELECTED_TAPE_BYTES',
        geometry_replay_proven_for_recorded_inputs=True, general_scoped_renderer_inputs_proven=False,
        level_and_color_selection_lowered=False, scene_control_resolved=False,
        cost=encode(cost), performance_measured=False, whole_entry_authorized=False, deployment_authorized=False,
        frontiers=[
            'LEVEL_LOCATOR_NORMALIZATION_COLORMAP_AND_FACE_STYLE_PRODUCTION_REMAIN_ORIGINAL',
            'UNMATCHED_COLLECTIONS_LAYOUT_GLYPHS_AND_OTHER_ORIGINAL_DRAW_EFFECTS_REMAIN_RESIDUAL',
            'NATIVE_LIBRARY_BUILD_EQUIVALENCE_AND_FULL_SCENE_RASTER_REQUIRE_DELIVERY_VALIDATION',
            'GENERAL_RUNTIME_FIELD_CHANGES_REQUIRE_FRESH_INTERVAL_STYLE_AND_LAYOUT_AUTHORITY',
        ],
    )


def prepare_contour_scene_plan(observation, out):
    report = _derive(_reference(observation, OBSERVATION))
    with Path(out).resolve().open('x', encoding='utf-8') as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return report


def read_contour_scene_plan(path):
    reference = _reference(path, SCHEMA)
    report = read_artifact(reference)
    _require(report == _derive(report['observation']), 'retained contour plan differs from evidence replay')
    read_artifact(reference)
    return report


def check(path):
    return read_contour_scene_plan(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--observation')
    group.add_argument('--check')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check and not args.out:
        report = check(args.check)
    elif args.observation and args.out:
        report = prepare_contour_scene_plan(args.observation, args.out)
    else:
        parser.error('provide --observation with --out or --check existing plan')
    print(json.dumps(dict(status=report['status'], samples=len(report['samples']),
                          selected_collections=sum(len(row['collections']) for row in report['samples']),
                          default_backend=report['default_backend']), sort_keys=True))


if __name__ == '__main__':
    main()
