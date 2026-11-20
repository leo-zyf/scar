"""Connect original scalar-formatted text to native AGG glyph draw commands."""

import argparse
import ast
import dis
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
from tempfile import TemporaryDirectory
from types import CodeType

from scar.analysis.source_agg_scene_layout import read_tape
from scar.analysis.source_syntax import _flatten_ast
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.trace.code_snapshot import code_key

SCHEMA = 'scar.source-agg-text'
OBSERVATION = 'scar.agg-title-observation'
SOURCE_EFFECTS = 'scar.source-controller-effects'
_NATIVE_SOURCE = Path(__file__).resolve().parents[1] / 'codegen' / 'native_agg_text_v1.cpp.in'


def _need(condition, detail):
    if not condition:
        raise ValueError('AGG text: ' + detail)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _one(items, detail):
    result = list(items)
    _need(len(result) == 1, detail)
    return result[0]


def _source(source):
    path = source['material']['uri']
    raw = read_artifact(source['material'], json_document=False)
    pending = [compile(raw, path, 'exec', dont_inherit=True)]
    matches = []
    while pending:
        code = pending.pop()
        if code.co_qualname == source['qualname'] and code_key(code) == source['code_key']:
            matches.append(code)
        pending.extend(item for item in code.co_consts if type(item) is CodeType)
    return raw, _one(matches, 'original Python CodeID did not replay')


def _coordinates(node):
    return node.lineno, node.end_lineno, node.col_offset, node.end_col_offset


def _format_route(caller, effects):
    raw, code = _source(caller)
    instruction = _one((item for item in dis.get_instructions(code)
                        if item.offset == caller['offset'] and item.opname == 'CALL'),
                       'text formatter does not have the original CALL bytecode')
    span = tuple(instruction.positions)
    _need(span[0] == caller['line'], 'original line and CALL operands differ')
    ast_root = ast.parse(raw, filename=caller['material']['uri'])
    selected = _one((node for node in ast.walk(ast_root)
                    if type(node) is ast.Call and _coordinates(node) == span),
                    'CALL does not have one original source expression')
    _need(type(selected.func) is ast.Attribute and len(selected.args) == 1 and not selected.keywords
          and type(selected.args[0]) is ast.JoinedStr, 'need one original positional f-string render operand')
    terms = selected.args[0].values
    numeric = _one((node for node in terms if type(node) is ast.FormattedValue),
                   'requires one original scientific scalar field')
    _need(type(numeric.value) is ast.Name and numeric.conversion == -1
          and type(numeric.format_spec) is ast.JoinedStr and len(numeric.format_spec.values) == 1
          and isinstance(numeric.format_spec.values[0], ast.Constant)
          and numeric.format_spec.values[0].value == 'e',
          'requires original plain lowercase scientific format and default precision')
    before, after = [], []
    output = before
    for term in terms:
        if term is numeric:
            output = after
        else:
            _need(type(term) is ast.Constant and type(term.value) is str and term.value.isascii(),
                  'qualify only original literal ASCII surroundings')
            output.append(term.value)
    root, occurrences = _flatten_ast(ast_root)
    _need(root == 0, 'original F2 root differed')
    body = _one((row for row in effects['bodies'] if row['source']['code_key'] == caller['code_key']
                and row['source']['sha256'] == caller['material']['sha256']),
                'original renderer call requires an exact F4 effect body')
    selected_calls = []
    for row in body['calls']:
        index = row['source_result']['port']['template']['occurrence']
        recorded = occurrences[index]
        attributes = dict(zip(ast.Call._attributes, (item.scalar.to_python()
                          for item in recorded.attributes), strict=True)) if recorded.constructor == 'Call' else {}
        if span == tuple(attributes.get(key) for key in ('lineno', 'end_lineno', 'col_offset', 'end_col_offset')):
            selected_calls.append(row)
    call = _one(selected_calls, 'original CALL expression has no unique original F4 fact')
    read_artifact(call['source_fact']['artifact'], json_document=False)
    _need(len(call['positional']) == 1 and not call['keywords']
          and call['positional_pack'] is None and call['keyword_pack'] is None,
          'exact original F4 title operand arity required')
    formal = _one((parameter for parameter in body['parameters'] if parameter['name'] == numeric.value.id),
                  'scientific scalar is not a direct original source formal')
    return dict(kind='SCIENTIFIC_BINARY64_DEFAULT_PRECISION', prefix=''.join(before), suffix=''.join(after),
                precision=6, formal=formal, call_span=list(span), call_source_fact=call['source_fact'],
                call_result=call['source_result'], text_operand=call['positional'][0], source=caller)


def _slice(raw, span):
    return memoryview(raw)[span['start']:span['end']]


def _eligible(row, route, tape):
    value = row['input_scalars'][route['formal']['name']]
    properties = row['properties']
    _need(type(value) in (int, float) and math.isfinite(value) and not row['explicit_options']
          and row['text'] == route['prefix'] + format(value, 'e') + route['suffix'],
          'original typed scalar must produce the observed title')
    _need(properties['align'] == 'center' and properties['valign'] == 'top'
          and properties['rotation_mode'] == 'default' and properties['angle'] == 0.
          and properties['math'] is False and properties['language'] is None
          and properties['text_features'] is None and properties['antialias'] == 1
          and properties['load_flags'] == 0 and row['boxes'] == 0
          and row['canvas'] == [tape['surface']['width'], tape['surface']['height']]
          and row['dpi'] == tape['surface']['dpi']
          and row['metrics'] is not None,
          'native scope requires unrotated centered/top single-line plain antialiased text without options')
    glyphs = row['glyphs']
    font = glyphs[0]['font']
    _need(glyphs and len(glyphs) == len(row['draws'])
          and all(glyph['font'] == font and glyph['size'] == properties['size']
                  and glyph['slant'] == 0 and glyph['extend'] == 1 for glyph in glyphs),
          'native scope requires one original face without fallback or synthetic style')
    read_artifact(font, json_document=False)
    ordinals = [draw['ordinal'] for draw in row['draws']]
    _need(ordinals == list(range(ordinals[0], ordinals[0] + len(ordinals))),
          'title renderer commands must remain contiguous in original order')
    return float(value), font, ordinals


def _native(worker, row, route, value, font, raw, tape):
    props = row['properties']
    with TemporaryDirectory(prefix='scar-agg-text-') as directory:
        output = Path(directory) / 'glyph-bytes'
        result = subprocess.run([worker['uri'], font['uri'], route['prefix'], route['suffix'], repr(value),
                                 *(repr(x) for x in props['anchor']), str(row['canvas'][1]), str(props['size']),
                                 str(row['dpi']), str(props['load_flags']), str(output), '--agg-wire'],
                                env={'LC_ALL': 'C'}, capture_output=True, text=True, check=True, timeout=10)
        lines = result.stdout.splitlines()
        pixels = output.read_bytes()
    _need(not result.stderr and len(lines) == 3 + len(row['glyphs']) and lines[0] == row['text']
          and list(map(float, lines[2].split())) == row['metrics'] + row['draw_origin'],
          'native title format, measured metrics, or derived draw origin differs')
    metadata = [list(map(int, line.split())) for line in lines[3:]]
    _need(pixels[:12] == b'SCARGLY1' + struct.pack('<I', len(metadata)), 'native owned glyph wire header differs')
    cursor, prefixes, spans, bitmap_bytes = 12, set(), [], 0
    for native, glyph, draw in zip(metadata, row['glyphs'], row['draws'], strict=True):
        glyph_id, x26, y26, x, y, height, width = native
        _need(native == [glyph['glyph'], 64 * glyph['dx'], 64 * glyph['dy'], *draw['origin'][:2], *draw['shape']],
              'derived glyph ID, 26.6 placement, dimensions, or rounded AGG origin differs')
        command = tape['commands'][draw['ordinal']]
        span, fields = command['span'], command['fields']
        geometry = fields['bitmap']
        _need(command['operator'] == 'draw_text_image' and [span['start'], span['end']] == draw['span']
              and geometry['shape'] == [height, width], 'original glyph draw span and geometry differ')
        prefix = bytes(raw[span['start']:geometry['span']['start']])
        byte_count = height * width
        _need(pixels[cursor:cursor + 8] == struct.pack('<II', height, width), 'native glyph wire shape differs')
        section = pixels[cursor + 8:cursor + 8 + byte_count]
        cursor += 8 + byte_count
        _need(pixels[cursor:cursor + 8] == struct.pack('<ii', x, y), 'native glyph wire origin differs')
        cursor += 8
        bitmap_bytes += byte_count
        prepared = prefix + struct.pack('<II', height, width) + section + struct.pack('<iid', x, y, 0.)
        _need(prepared == _slice(raw, span) and _digest(section) == draw['bytes_sha256'],
              'native glyph payload does not reproduce original AGG command bytes')
        prefixes.add(prefix)
        spans.append(span)
    _need(cursor == len(pixels) and len(prefixes) == 1, 'extra native glyph bytes or varying graphics state')
    start, end = spans[0]['start'], spans[-1]['end']
    return dict(provider_versions=lines[1].split(), count=len(spans),
                span=dict(start=start, end=end), sha256=_digest(memoryview(raw)[start:end]),
                gc_prefix=next(iter(prefixes)).hex(), bitmap_bytes=bitmap_bytes,
                generated_wire_sha256=_digest(pixels))


def _derive(observation, effects, worker):
    recorded = read_artifact(observation)
    f4 = read_artifact(effects)
    _need(recorded['schema_version'] == 1 and recorded['capture_is_diagnostic'] is True
          and f4['schema_version'] == 1, 'requires official original graphics and F4 evidence')
    agg = read_artifact(recorded['agg_scene'])
    runtime = recorded['providers']
    for key in ('suptitle', 'measure', 'draw_text', 'draw_glyphs', 'draw_record'):
        _source(runtime[key])
    read_artifact(runtime['library'], json_document=False)
    rows = recorded['records']
    _need(len(rows) >= 2 and len(rows) == len(agg['frames']),
          'need distinct original title placements in all retained frames')
    reference = _format_route(rows[0]['source'], f4)
    samples = []
    base = Path(recorded['agg_scene']['uri']).parent
    for index, row in enumerate(rows):
        saved = agg['frames'][index]
        _need(row['source'] == reference['source'] and row['frame_index'] == saved['index'] == index
              and Path(saved['scene']).name == saved['scene'], 'frame and source CALL identity differ')
        scene = _ref(base / saved['scene'], 'scar.agg-scene-tape')
        _need(scene['sha256'] == saved['scene_sha256'], 'original scene bytes differ')
        raw = read_artifact(scene, json_document=False)
        tape = read_tape(raw)
        value, font, ordinals = _eligible(row, reference, tape)
        native = _native(worker, row, reference, value, font, raw, tape)
        _need(native['provider_versions'][0] == runtime['freetype'],
              'linked native FreeType version differs from pinned original embedded provider')
        samples.append(dict(frame_index=index, formal_value=value, scene=scene, font=font,
            canvas=row['canvas'], dpi=row['dpi'], anchor=row['properties']['anchor'], points=row['properties']['size'],
            load_flags=row['properties']['load_flags'], metrics=row['metrics'], draw_origin=row['draw_origin'],
            draw_ordinals=ordinals, glyph_count=len(ordinals), native=native))
    _need(len({row['formal_value'] for row in samples}) == len(samples)
          and len({row['native']['gc_prefix'] for row in samples}) == 1,
          'fixture requires varying scalar ingress and one original graphics-state block')
    links = subprocess.check_output(['ldd', worker['uri']], env={'LC_ALL': 'C'}, text=True, timeout=10)
    _need('libpython' not in links and 'not found' not in links, 'native text provider requires Python or missing libraries')
    locations = [line.split('=>', 1)[1].split(' (', 1)[0].strip() for line in links.splitlines()
                 if '=>' in line and any(name in line for name in ('libfreetype', 'libharfbuzz'))]
    _need(len(locations) == 2, 'two matched native font providers required')
    return dict(schema=SCHEMA, schema_version=1, observation=observation, source_effects=effects,
                worker=worker, worker_source=_ref(_NATIVE_SOURCE, 'scar.native-cxx-source'),
                native_provider_libraries=[_ref(path, 'scar.native-provider-library') for path in locations],
                format_route=reference, original_provider=runtime, samples=samples,
                observed_native_command_bytes_equal=True, python_runtime_required=False,
                relation='ORIGINAL_CALL_TO_SCALAR_FORMAT_TO_FONT_LAYOUT_TO_AGG_DRAW',
                scope='EXPLICIT_TRIAL_ORIGINAL_SINGLE_FONT_LTR_ASCII_CENTER_TOP',
                frontiers=['GENERAL_RAQM_AND_NATIVE_HARFBUZZ_EQUIVALENCE', 'FALLBACK_FONTS_AND_COMPLEX_TEXT',
                           'UNOBSERVED_LAYOUT_OPTIONS_AND_OTHER_GRAPHICS_STATE', 'FULL_SCENE_REPLACEMENT_AUTHORITY'],
                whole_entry_authorized=False, deployment_authorized=False, performance_measured=False)


def project(observation, effects, worker):
    return _derive(_ref(observation, OBSERVATION), _ref(effects, SOURCE_EFFECTS),
                   _ref(worker, 'scar.native-elf'))


def check(path):
    report = read_artifact(_ref(path, SCHEMA))
    _need(report == _derive(report['observation'], report['source_effects'], report['worker']),
          'stored glyph plan differs from source, AGG and native probe replay')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('observation', 'source-effects', 'worker', 'out', 'check'):
        parser.add_argument('--' + name)
    args = parser.parse_args(argv)
    if args.check and not any((args.observation, args.source_effects, args.worker, args.out)):
        report = check(args.check)
    elif not args.check and all((args.observation, args.source_effects, args.worker, args.out)):
        report = project(args.observation, args.source_effects, args.worker)
        with Path(args.out).open('x', encoding='utf-8') as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        parser.error('provide --observation, --source-effects, --worker and --out or --check PLAN')
    print(json.dumps(dict(frames=len(report['samples']), glyphs=sum(row['glyph_count'] for row in report['samples']),
                          exact_native_commands=report['observed_native_command_bytes_equal'])))


if __name__ == '__main__':
    main()
