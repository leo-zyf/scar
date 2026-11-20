"""Anchor original centered single-line figure text to exact AGG glyph commands."""

import argparse
import functools
from hashlib import sha256
import inspect
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference
from scar.trace import agg_scene, contour_style
from scar.trace.code_snapshot import code_key


def ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def source(code):
    return dict(qualname=code.co_qualname, code_key=code_key(code),
                material=ref(code.co_filename, 'scar.python-source'))


def capture(out, target, *, with_contour_style=False):
    from matplotlib import ft2font
    from matplotlib.backends.backend_agg import RendererAgg, get_hinting_flag
    from matplotlib.figure import FigureBase

    output = Path(out).resolve()
    titles, metrics, active, records = {}, {}, [], []
    methods = []

    def patch(cls, name, wrapper):
        original = getattr(cls, name)
        setattr(cls, name, functools.wraps(original)(lambda *a, **k: wrapper(original, *a, **k)))
        methods.append((cls, name, original))
        return source(original.__code__)

    def set_title(fn, figure, text, **kwargs):
        frame = inspect.currentframe().f_back.f_back
        try:
            caller = dict(**source(frame.f_code), line=frame.f_lineno, offset=frame.f_lasti)
            properties = {name: frame.f_locals[name] for name in frame.f_code.co_varnames[:frame.f_code.co_argcount]
                          if name in frame.f_locals and type(frame.f_locals[name]) in (float, int, bool, str)}
        finally:
            del frame
        title = fn(figure, text, **kwargs)
        titles[id(title)] = (title, dict(source=caller, input_scalars=properties, explicit_options=kwargs))
        return title

    def measure(fn, renderer, text, prop, ismath):
        result = fn(renderer, text, prop, ismath)
        metrics[text, float(prop.get_size_in_points()), renderer.dpi, ismath] = list(result)
        return result

    def draw_text(fn, renderer, gc, x, y, text, prop, angle, ismath=False, mtext=None):
        row = titles.get(id(mtext))
        if row is None or row[0] is not mtext:
            return fn(renderer, gc, x, y, text, prop, angle, ismath=ismath, mtext=mtext)
        if active:
            raise ValueError('nested original title drawing')
        anchor = mtext.get_transform().transform(mtext.get_position())
        size = float(prop.get_size_in_points())
        record = dict(**row[1], text=text, dpi=renderer.dpi,
                      canvas=[renderer.width, renderer.height],
                      properties=dict(position=list(mtext.get_position()), anchor=list(anchor), size=size,
                                      fontsize_family=prop.get_family(), align=mtext.get_horizontalalignment(),
                                      valign=mtext.get_verticalalignment(), rotation_mode=mtext.get_rotation_mode(),
                                      math=ismath, text_features=mtext.get_fontfeatures(),
                                      language=mtext.get_language(), angle=angle, antialias=gc.get_antialiased(),
                                      load_flags=get_hinting_flag().value),
                      draw_origin=[x, y], metrics=metrics.get((text, size, renderer.dpi, ismath)),
                      glyphs=[], draws=[])
        records.append(record)
        active.append(record)
        try:
            return fn(renderer, gc, x, y, text, prop, angle, ismath=ismath, mtext=mtext)
        finally:
            active.pop()

    def glyphs(fn, renderer, gc, x, y, angle, glyphs, boxes):
        if active:
            glyphs = list(glyphs)
            record = active[-1]
            fonts = {}
            for font, size, index, slant, extend, dx, dy in glyphs:
                if font.fname not in fonts:
                    fonts[font.fname] = ref(font.fname, 'scar.ttf-font')
                record['glyphs'].append(dict(font=fonts[font.fname], size=size, glyph=index,
                                             slant=slant, extend=extend, dx=dx, dy=dy))
            record['boxes'] = len(boxes)
        return fn(renderer, gc, x, y, angle, glyphs, boxes)

    def draw(fn, scene, name, args, kwargs):
        start = len(scene.data) + 28
        result = fn(scene, name, args, kwargs)
        if active and name == 'draw_text_image':
            bitmap, x, y, angle, _ = args
            record = active[-1]
            record['draws'].append(dict(
                ordinal=scene.operations.total() - 1,
                span=[start, len(scene.data) + 28],
                shape=list(bitmap.shape), bytes_sha256=sha256(bitmap.tobytes()).hexdigest(),
                origin=[x, y, angle],
            ))
        return result

    def source_of(cls, name):
        return patch(cls, name, dict(suptitle=set_title, get_text_width_height_descent=measure,
                                    draw_text=draw_text, _draw_text_glyphs_and_boxes=glyphs, record=draw)[name])

    providers = dict(
        suptitle=source_of(FigureBase, 'suptitle'),
        measure=source_of(RendererAgg, 'get_text_width_height_descent'),
        draw_text=source_of(RendererAgg, 'draw_text'),
        draw_glyphs=source_of(RendererAgg, '_draw_text_glyphs_and_boxes'),
        draw_record=source_of(agg_scene._Scene, 'record'),
        freetype=ft2font.__freetype_version__, raqm=ft2font.__libraqm_version__,
        library=ref(ft2font.__file__, 'scar.native-provider-library'),
    )
    try:
        if with_contour_style:
            contour_style.capture(output, target)
            observation = json.loads((output / 'result.json').read_bytes())
        else:
            observation = agg_scene.capture(output, target)
    finally:
        for cls, name, original in reversed(methods):
            setattr(cls, name, original)
    if len(records) != len(observation['frames']):
        raise ValueError('expected one independently identified figure title per original frame')
    for index, row in enumerate(records):
        row['frame_index'] = index
        if len(row['glyphs']) != len(row['draws']) or not row['glyphs']:
            raise ValueError('original glyph and AGG draw effect counts differ')
    result = dict(schema='scar.agg-title-observation', schema_version=1,
                  agg_scene=ref(output / 'result.json', 'scar.agg-scene-observation'),
                  providers=providers, records=records, capture_is_diagnostic=True)
    with (output / 'text-result.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--with-contour-style', action='store_true')
    parser.add_argument('target', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    target = args.target[1:] if args.target[:1] == ['--'] else args.target
    if not target:
        parser.error('provide --out DIRECTORY -- SCRIPT [ARGS]')
    result = capture(args.out, target, with_contour_style=args.with_contour_style)
    print(json.dumps(dict(samples=len(result['records']))))


if __name__ == '__main__':
    main()
