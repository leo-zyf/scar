"""Observe a filled Matplotlib contour's scalar-to-AGG style connections."""

import argparse
from contextlib import ExitStack
from functools import wraps
import hashlib
from inspect import unwrap
import json
from math import isfinite, prod
from pathlib import Path
import struct
from unittest.mock import patch

from scar.analysis.source_agg_scene_layout import read_tape
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.trace import agg_scene
from scar.trace.code_snapshot import code_key
from scar.trace.contour_scene import INPUT_MAGIC


SCHEMA = 'scar.contour-style-observation'
_CONTOUR = 'scar.contour-scene-observation'


def _source(function):
    return dict(qualname=function.__code__.co_qualname, code_key=code_key(function.__code__),
                material=artifact_reference(function.__code__.co_filename, 'scar.python-source'))


def _pack(numpy, value, dtype='<f8'):
    array = numpy.asarray(value, dtype=dtype, order='C')
    if array.size > 16384:
        raise ValueError('contour style payload exceeds observation budget')
    return dict(dtype=dtype, shape=list(array.shape), data_hex=array.tobytes().hex())


class _Style:
    def __init__(self, numpy):
        self.np = numpy
        self.charts, self.scenes, self.sources = {}, {}, {}
        self.active = self.mapping = None

    def patch(self, stack, owner, name, callback, *, source=True):
        original = getattr(owner, name)
        if source:
            self.sources[callback.__name__] = _source(original)

        @wraps(original)
        def observe(*args, **kwargs):
            return callback(original, *args, **kwargs)

        stack.enter_context(patch.object(owner, name, observe))

    def chart(self, original, chart, *args, **kwargs):
        prior = self.active
        self.active = self.charts.setdefault(id(chart), dict(owner=chart, draws=[], mappings=[]))
        try:
            return original(chart, *args, **kwargs)
        finally:
            self.active = prior

    def autolev(self, original, chart):
        row = self.charts.setdefault(id(chart), dict(owner=chart, draws=[], mappings=[]))
        prior, self.active = self.active, row
        row['requested_levels'] = chart.levels if type(chart.levels) is int else None
        try:
            result = original(chart)
            row['levels'] = _pack(self.np, result)
            row['bounds'] = dict(zmin=float(chart.zmin), zmax=float(chart.zmax))
            return result
        finally:
            self.active = prior

    def contour_args(self, original, chart, args, kwargs):
        x, y, z = original(chart, args, kwargs)
        if id(chart) in self.charts:
            values = self.np.asarray(z, dtype='<f8')
            self.charts[id(chart)]['z'] = dict(shape=list(values.shape), dtype='<f8',
                sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                masked=bool(self.np.ma.is_masked(z)))
        return x, y, z

    def locator(self, original, locator, vmin, vmax):
        row = self.active
        if row is None or getattr(row['owner'], 'locator', None) is not locator:
            return original(locator, vmin, vmax)
        from matplotlib import rcParams

        mode = rcParams['axes.autolimit_mode']
        values = original(locator, vmin, vmax)
        row['locator'] = dict(vmin=float(vmin), vmax=float(vmax), nbins=locator._nbins,
            min_n_ticks=locator._min_n_ticks, prune=locator._prune, integer=locator._integer,
            symmetric=locator._symmetric, steps=_pack(self.np, locator._steps),
            extended_steps=_pack(self.np, locator._extended_steps), raw_ticks=_pack(self.np, values),
            autolimit_mode=mode)
        return values

    def autoscale(self, original, norm, values):
        row = self.active
        if row is None or row['owner'].norm is not norm:
            return original(norm, values)
        before = [norm.vmin, norm.vmax]
        result = original(norm, values)
        row.setdefault('norm_autoscale', []).append(
            dict(before=before, after=[norm.vmin, norm.vmax], inputs=_pack(self.np, values)))
        return result

    def colors(self, original, chart):
        result = self.chart(original, chart)
        row = self.charts[id(chart)]
        row.update(filled=bool(chart.filled), extend=chart.extend, log_scale=bool(chart.logscale),
                   layers=_pack(self.np, chart.layers), cvalues=_pack(self.np, chart.cvalues))
        return result

    def colorize(self, original, colorizer, x, alpha=None, bytes=False, norm=True):
        row = self.active
        if row is None or getattr(row['owner'], '_colorizer', None) is not colorizer:
            return original(colorizer, x, alpha=alpha, bytes=bytes, norm=norm)
        prior = self.mapping
        self.mapping = dict(inputs=_pack(self.np, x), alpha=alpha is not None, bytes=bytes, normalize=norm)
        try:
            rgba = original(colorizer, x, alpha=alpha, bytes=bytes, norm=norm)
            self.mapping['rgba'] = _pack(self.np, rgba)
            row['mappings'].append((rgba, self.mapping))
            return rgba
        finally:
            self.mapping = prior

    def normalize(self, original, norm, value, clip=None):
        result = original(norm, value, clip=clip)
        if self.mapping is not None:
            self.mapping['norm'] = dict(vmin=float(norm.vmin), vmax=float(norm.vmax),
                clip=bool(norm.clip if clip is None else clip), output=_pack(self.np, result),
                masked=bool(self.np.ma.is_masked(result)))
        return result

    def lookup(self, original, cmap, values, alpha=None, bytes=False):
        result = original(cmap, values, alpha=alpha, bytes=bytes)
        if self.mapping is not None:
            self.mapping['color_map'] = dict(name=cmap.name, size=cmap.N,
                indices=dict(under=cmap._i_under, over=cmap._i_over, bad=cmap._i_bad),
                lut=_pack(self.np, cmap._lut), input=_pack(self.np, values),
                rgba=_pack(self.np, result[0]), masked=bool(self.np.asarray(result[1]).any()))
        return result

    def record(self, original, scene, name, args, kwargs):
        key = id(scene)
        if key not in self.scenes:
            self.scenes[key] = (scene, len(self.scenes))
        row = self.active
        if row is not None and name == 'draw_path_collection':
            _, _, paths, _, _, _, face, edge, widths, _, antialias, _, _ = args
            maps = [mapping for rgba, mapping in row['mappings'] if face is rgba]
            if len(maps) != 1:
                raise ValueError('AGG facecolors lack an exact scalar mapping identity')
            fields = {label: _pack(self.np, data, dtype) for label, data, dtype in (
                ('facecolors', face, '<f8'), ('edgecolors', edge, '<f8'), ('widths', widths, '<f8'),
                ('antialias', antialias, 'u1'), ('hatchcolors', kwargs.get('hatchcolors', ()), '<f8'))}
            row['draws'].append(dict(frame_index=self.scenes[key][1], draw_ordinal=scene.operations.total(),
                                    path_count=len(paths), fields=fields, mapped=maps[0]))
        return original(scene, name, args, kwargs)

    def report(self):
        for row in self.charts.values():
            if not row['draws']:
                continue
            frames = {draw['frame_index'] for draw in row['draws']}
            if len(frames) != 1 or 'locator' not in row or 'z' not in row:
                raise ValueError('contour style requires one automatic-level chart in one original frame')
            yield {**{key: row[key] for key in ('bounds', 'z', 'requested_levels', 'locator', 'levels',
                                               'filled', 'extend', 'log_scale', 'layers', 'cvalues',
                                               'norm_autoscale')},
                   'frame_index': next(iter(frames)), 'collections': row['draws']}


def capture(out, target):
    from matplotlib.collections import Collection
    from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize
    from matplotlib.colorizer import Colorizer
    from matplotlib.contour import ContourSet, QuadContourSet
    from matplotlib.ticker import MaxNLocator, scale_range
    from matplotlib.transforms import nonsingular
    import numpy

    out = Path(out).resolve()
    probe = _Style(numpy)
    for name, method in (
        ('range_scaling', scale_range),
        ('nonsingular_bounds', nonsingular),
        ('raw_locator', MaxNLocator._raw_ticks),
        ('locator_defaults', ContourSet._ensure_locator_exists),
        ('level_argument', ContourSet._process_contour_level_args),
        ('layer_midpoints', ContourSet._process_levels),
        ('mapped_collection_update', Collection.update_scalarmappable),
        ('collection_style_submission', Collection.draw),
        ('contour_draw_implementation', ContourSet.draw),
        ('palette_initialization', LinearSegmentedColormap._init),
        ('palette_call', Colormap.__call__),
    ):
        probe.sources[name] = _source(unwrap(method))
    with ExitStack() as stack:
        for owner, name, callback in (
            (QuadContourSet, '_contour_args', probe.contour_args),
            (ContourSet, '_autolev', probe.autolev),
            (ContourSet, '_process_colors', probe.colors),
            (ContourSet, 'draw', probe.chart),
            (MaxNLocator, 'tick_values', probe.locator),
            (Colorizer, 'to_rgba', probe.colorize),
            (Normalize, 'autoscale_None', probe.autoscale),
            (Normalize, '__call__', probe.normalize),
            (Colormap, '_get_rgba_and_mask', probe.lookup),
        ):
            probe.patch(stack, owner, name, callback)
        probe.patch(stack, agg_scene._Scene, 'record', probe.record, source=False)
        report = agg_scene.capture(out, target, contours=True)
    if len(probe.scenes) != len(report['frames']):
        raise ValueError('original AGG scene lifetimes do not match exported raster count')
    result = dict(schema=SCHEMA, schema_version=1, target=list(target), sources=probe.sources,
        agg_scene=artifact_reference(out / 'result.json', agg_scene.SCHEMA),
        contours=artifact_reference(out / 'contour-result.json', _CONTOUR),
        samples=list(probe.report()), capture_is_diagnostic=True)
    _check(result)
    path = out / 'style-result.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
    return result


def _floats(values, rank):
    shape = values['shape']
    raw = bytes.fromhex(values['data_hex'])
    if values['dtype'] != '<f8' or len(shape) != rank or len(raw) != 8 * prod(shape):
        raise ValueError('unexpected original float64 style geometry')
    return tuple(value for value, in struct.iter_unpack('<d', raw))


def _matches(values, reference):
    return struct.pack(f'<{len(values)}d', *values) == bytes.fromhex(reference['data_hex'])


def _geometry(sample, contours):
    visible = {draw['draw_ordinal'] for draw in sample['collections'] if draw['path_count']}
    candidates = [
        row for row in contours['samples']
        if row['frame_index'] == sample['frame_index']
        and visible == {draw['draw_ordinal'] for draw in row['collections']}
    ]
    if len(candidates) != 1 or not visible:
        raise ValueError('original contour style cannot be joined to one exact geometry submission')
    row, = candidates
    raw = read_artifact(row['input'], json_document=False)
    height, width, count, _, _, _, mask = struct.unpack_from('<7Q', raw, len(INPUT_MAGIC))
    if not raw.startswith(INPUT_MAGIC) or mask or [height, width] != sample['z']['shape']:
        raise ValueError('original contour geometry or mask differs')
    z_offset = row['grid']['offsets']['z']
    levels_offset = row['grid']['offsets']['levels']
    z = raw[z_offset:levels_offset]
    levels = bytes.fromhex(sample['levels']['data_hex'])
    if (len(z) != 8 * height * width or hashlib.sha256(z).hexdigest() != sample['z']['sha256']
        or count * 8 != len(levels) or raw[levels_offset:] != levels):
        raise ValueError('original contour provider consumes different data or levels')
    values = [value for value, in struct.iter_unpack('<d', z)]
    bounds = sample['bounds']
    if not all(isfinite(value) for value in values) or (
            min(values), max(values)) != (bounds['zmin'], bounds['zmax']):
        raise ValueError('original contour range differs from actual consumed values')


def _check(document):
    if document.get('schema_version') != 1 or not document['samples']:
        raise ValueError('no original contour style samples')
    observed = read_artifact(document['agg_scene'])
    contours = read_artifact(document['contours'])
    if contours['agg_scene'] != document['agg_scene']:
        raise ValueError('contour style and geometry observations came from different runs')
    for source in document['sources'].values():
        read_artifact(source['material'], json_document=False)
    base = Path(document['agg_scene']['uri']).parent
    tapes = {}
    for sample in document['samples']:
        _geometry(sample, contours)
        frame_index = sample['frame_index']
        if frame_index not in tapes:
            metadata = observed['frames'][frame_index]
            tape = (base / metadata['scene']).read_bytes()
            if hashlib.sha256(tape).hexdigest() != metadata['scene_sha256']:
                raise ValueError('original AGG tape digest differs')
            tapes[frame_index] = tape, read_tape(tape)['commands']
        tape, commands = tapes[frame_index]
        zmin, zmax = sample['bounds']['zmin'], sample['bounds']['zmax']
        locator = sample['locator']
        levels = _floats(sample['levels'], 1)
        raw = _floats(locator['raw_ticks'], 1)
        start = max((i for i,value in enumerate(raw) if value < zmin), default=0)
        end = next((i + 1 for i,value in enumerate(raw) if value > zmax), len(raw))
        if (sample['z']['masked'] or not sample['filled'] or sample['extend'] != 'neither'
            or sample['log_scale'] or locator['symmetric'] or not isfinite(zmin) or not isfinite(zmax)
            or (locator['vmin'], locator['vmax']) != (zmin, zmax)
            or not _matches(raw[start:end] if end - start >= 3 else raw, sample['levels'])):
            raise ValueError('original locator trim or filled linear branch differs')
        if 'norm_autoscale' in sample and not any(
                event['before'] == [None, None] and event['after'] == [levels[0], levels[-1]]
                and event['inputs'] == sample['levels'] for event in sample['norm_autoscale']):
            raise ValueError('original normalization bounds lack an exact level input')
        layers = tuple(0.5 * (a + b) for a,b in zip(levels[:-1], levels[1:], strict=True))
        if not _matches(layers, sample['layers']) or sample['layers'] != sample['cvalues']:
            raise ValueError('original contour level midpoint differs')
        for draw in sample['collections']:
            ordinal = draw['draw_ordinal']
            fields = commands[ordinal]['fields']
            if commands[ordinal]['operator'] != 'draw_path_collection' or fields['paths']['count'] != draw['path_count']:
                raise ValueError('AGG original collection port differs')
            for name, actual in draw['fields'].items():
                span = fields[name]['payload_span']
                shape = [0] * len(fields[name]['shape']) if not prod(actual['shape']) else actual['shape']
                if (shape != fields[name]['shape'] or actual['dtype'] != fields[name]['format']
                    or bytes.fromhex(actual['data_hex']) != tape[span['start']:span['end']]):
                    raise ValueError('AGG original collection style bytes differ')
            mapped = draw['mapped']
            norm, cmap = mapped['norm'], mapped['color_map']
            vmin, vmax = norm['vmin'], norm['vmax']
            if (norm['clip'] or norm['masked'] or not mapped['normalize'] or mapped['alpha'] or mapped['bytes']
                or cmap['masked'] or (vmin, vmax) != (levels[0], levels[-1])
                or mapped['inputs'] != sample['layers'] or mapped['rgba'] != draw['fields']['facecolors']):
                raise ValueError('original contour colormap branch differs')
            if not vmin < vmax or not all(vmin <= x <= vmax for x in layers):
                raise ValueError('original contour linear normal range differs')
            values = tuple((x - vmin) / (vmax - vmin) for x in layers)
            if not _matches(values, norm['output']) or norm['output'] != cmap['input']:
                raise ValueError('original contour normalization bytes differ')
            size = cmap['size']
            lut = _floats(cmap['lut'], 2)
            if size <= 0 or cmap['lut']['shape'] != [size + 3, 4]:
                raise ValueError('original palette table geometry differs')
            indexes = (min(int(x * size), size - 1) for x in values)
            rgba = tuple(v for index in indexes for v in lut[4 * index:4 * (index + 1)])
            if not _matches(rgba, cmap['rgba']) or cmap['rgba'] != mapped['rgba']:
                raise ValueError('original contour lookup bytes differ')


def read_style(path):
    reference = artifact_reference(Path(path).resolve(strict=True), SCHEMA)
    result = read_artifact(reference)
    _check(result)
    return reference, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out')
    parser.add_argument('--check')
    parser.add_argument('target', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    target = args.target[1:] if args.target[:1] == ['--'] else args.target
    if args.check and not (args.out or target):
        _, result = read_style(args.check)
    elif args.out and target and not args.check:
        result = capture(args.out, target)
    else:
        parser.error('provide --out DIR -- SCRIPT [ARGS] or --check FILE')
    print(json.dumps(dict(samples=len(result['samples']))))


if __name__ == '__main__':
    main()
