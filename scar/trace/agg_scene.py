"""Record a bounded native AGG draw scene and its independently rendered raster."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy
import struct
import sys


SCHEMA = 'scar.agg-scene-observation'
MAGIC = b'SCARAGG1'
_BUDGET = 64_000_000


class _Scene:
    def __init__(self, numpy):
        self.np = numpy
        self.data = bytearray()
        self.operations = Counter()

    def word(self, format, *values):
        self.data.extend(struct.pack('<' + format, *values))

    def values(self, values, rank, *, byte=False):
        dtype = 'u1' if byte else '<f8'
        array = self.np.asarray(values, dtype=dtype, order='C')
        if not array.size:
            array = array.reshape((0,) * rank)
        if array.ndim != rank or len(self.data) + array.nbytes > _BUDGET:
            raise ValueError('AGG array rank or capture budget exceeded')
        self.word('I' * rank, *array.shape)
        self.data.extend(array.tobytes())

    def transform(self, value):
        matrix = value.get_matrix()
        if tuple(matrix[2, :]) != (0., 0., 1.):
            raise ValueError('AGG requires an original affine transform')
        self.word('6d', *matrix[:2, :].T.flat)

    def path(self, value):
        self.values(value.vertices, 2)
        self.values(() if value.codes is None else value.codes, 1, byte=True)
        self.word('Bd', int(value.should_simplify), value.simplify_threshold)

    def rgba(self, value, gc=None):
        if value is None:
            self.word('4d', 0., 0., 0., 0.)
            return
        r, g, b = value[:3]
        a = value[3] if len(value) == 4 else 1.
        if gc is not None and (gc.get_forced_alpha() or len(value) == 3):
            a = gc.get_alpha()
        self.word('4d', r, g, b, a)

    def dashes(self, dashes):
        shift, sequence = dashes
        sequence = tuple(sequence) if sequence is not None else ()
        if len(sequence) % 2:
            sequence = sequence * 2
        self.word('dI', shift or 0., len(sequence) // 2)
        self.word('d' * len(sequence), *sequence)

    def gc(self, gc):
        rect = gc._cliprect
        clip, transform = gc.get_clip_path()
        if clip is not None or transform is not None or gc.get_hatch_path() is not None:
            raise ValueError('clip paths and hatches need their own AGG scene port')
        if gc.get_sketch_params() is not None:
            raise ValueError('AGG sketch parameters remain open')
        self.word('ddB', gc.get_linewidth(), gc.get_alpha(), int(gc.get_forced_alpha()))
        self.rgba(gc.get_rgb())
        self.word('4B', int(gc.get_antialiased()), ('butt', 'square', 'round').index(
            {'projecting': 'square'}.get(gc.get_capstyle(), gc.get_capstyle())),
            ('miter', 'round', 'bevel').index(gc.get_joinstyle()),
            {None: 0, False: 1, True: 2}[gc.get_snap()])
        self.word('4d', *(rect.extents if rect is not None else (0.,) * 4))
        self.dashes(gc.get_dashes())

    def record(self, name, args, kwargs):
        self.operations[name] += 1
        if kwargs and not (name == 'draw_path_collection' and set(kwargs) == {'hatchcolors'}):
            raise ValueError('unrecognized original renderer keywords')
        if name == 'clear':
            self.word('B', 0)
        elif name == 'draw_path':
            gc, path, transform, face = args
            self.word('B', 1)
            self.gc(gc)
            self.path(path)
            self.transform(transform)
            self.rgba(face, gc)
        elif name == 'draw_markers':
            gc, marker, mt, path, transform, face = args
            self.word('B', 2)
            self.gc(gc)
            self.path(marker)
            self.transform(mt)
            self.path(path)
            self.transform(transform)
            self.rgba(face, gc)
        elif name == 'draw_text_image':
            image, x, y, angle, gc = args
            if type(x) is not int or type(y) is not int:
                raise ValueError('glyph coordinates must be original integer locations')
            self.word('B', 3)
            self.gc(gc)
            self.values(image, 2, byte=True)
            self.word('iid', x, y, angle)
        elif name == 'draw_path_collection':
            gc, transform, paths, matrices, offsets, ot, fc, ec, widths, lines, aa, _, _ = args
            self.word('B', 4)
            self.gc(gc)
            self.transform(transform)
            self.word('I', len(paths))
            for path in paths:
                self.path(path)
            self.values(matrices, 3)
            self.values(offsets, 2)
            self.transform(ot)
            self.values(fc, 2)
            self.values(ec, 2)
            self.values(widths, 1)
            self.word('I', len(lines))
            for line in lines:
                self.dashes(line)
            self.values(aa, 1, byte=True)
            self.values(kwargs.get('hatchcolors', ()), 2)
        else:
            raise ValueError('unmodeled original AGG operation: ' + name)


def capture(out, target):
    import matplotlib
    from matplotlib.backends import backend_agg as module
    import numpy

    out = Path(out).resolve()
    out.mkdir()
    original = module._RendererAgg
    frames = []
    source = Path(target[0]).resolve(strict=True)

    class RendererProbe:
        def __init__(self, width, height, dpi):
            self.renderer = original(width, height, dpi)
            self.scene = _Scene(numpy)
            self.geometry = width, height, dpi

        def __getattr__(self, name):
            method = getattr(self.renderer, name)
            if not callable(method):
                return method

            def invoke(*args, **kwargs):
                self.scene.record(name, args, kwargs)
                return method(*args, **kwargs)
            return invoke

        def __buffer__(self, flags):
            raster = memoryview(self.renderer)
            index = len(frames)
            width, height, dpi = self.geometry
            raw = raster.tobytes()
            scene = (MAGIC + struct.pack('<IIdI', width, height, dpi, self.scene.operations.total())
                     + self.scene.data)
            if len(frames) >= 16 or len(scene) > _BUDGET or len(raw) > _BUDGET:
                raise ValueError('AGG rendered-frame capture budget exceeded')
            raster_name, scene_name = f'frame-{index}.rgba', f'frame-{index}.agg'
            (out / raster_name).write_bytes(raw)
            (out / scene_name).write_bytes(scene)
            frames.append(dict(index=index, geometry=list(self.geometry),
                operations=dict(self.scene.operations), raster=raster_name,
                raster_sha256=hashlib.sha256(raw).hexdigest(), scene=scene_name,
                scene_sha256=hashlib.sha256(scene).hexdigest(), scene_bytes=len(scene)))
            self.scene = _Scene(numpy)
            return raster

    prior_argv, prior_path = sys.argv, sys.path[0]
    sys.argv = [str(source), *target[1:]]
    sys.path[0] = str(source.parent)
    module._RendererAgg = RendererProbe
    try:
        runpy.run_path(str(source), run_name='__main__')
    finally:
        module._RendererAgg = original
        sys.argv, sys.path[0] = prior_argv, prior_path
    if not frames:
        raise ValueError('target did not expose an AGG raster')
    result = dict(schema=SCHEMA, schema_version=1,
        source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        matplotlib_version=matplotlib.__version__, numpy_version=numpy.__version__,
        frames=frames, open_effects=['LAYOUT', 'CONTOUR_GEOMETRY', 'GLYPH_BITMAP_PRODUCTION'],
        capture_is_diagnostic=True, performance_measured=False)
    (out / 'result.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('target', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    target = args.target[1:] if args.target[:1] == ['--'] else args.target
    if not target:
        parser.error('provide -- SCRIPT [ARGS]')
    result = capture(args.out, target)
    print(json.dumps(dict(frames=len(result['frames']), provider=result['matplotlib_version'])))


if __name__ == '__main__':
    main()
