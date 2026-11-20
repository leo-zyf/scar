"""Deliver live filled contours and palettes into the qualified native draw layout."""

from hashlib import sha256
import json
from pathlib import Path
import shutil
import struct
import subprocess
from tempfile import TemporaryDirectory

from scar.analysis import source_agg_text
from scar.codegen import native_contour_agg, native_contour_scene, native_contour_style
from scar.codegen.native_project import build_native_project, write_native_project
from scar.compiler.plan_contour_agg_style import plan_sample
from scar.compiler.plan_contour_scene import SCHEMA as PLAN, _source_code, read_contour_scene_plan
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.trace.contour_style import SCHEMA as OBSERVATION, read_style

SCHEMA = 'scar.native-contour-agg-live'
TARGET = 'scar_contour_agg_live'
EXE = 'bin/' + TARGET
_TEMPLATES = Path(__file__).parent


def _require(ok, detail):
    if not ok:
        raise ValueError('live contour AGG: ' + detail)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _float_digest(row):
    _require(row['dtype'] == '<f8', 'original scalar must be a binary64 provider value')
    return sha256(bytes.fromhex(row['data_hex'])).hexdigest()


def _configuration(sample, provider):
    loc = sample['locator']
    expected = provider['parameters']
    _require(type(loc['nbins']) is int and 0 < loc['nbins'] <= 4096
             and type(loc['min_n_ticks']) is int and 0 < loc['min_n_ticks'] <= 4096
             and not loc['integer'] and not loc['symmetric'] and loc['prune'] is None
             and loc['autolimit_mode'] == 'data' and sample['filled']
             and sample['extend'] == 'neither' and not sample['log_scale']
             and loc['nbins'] == expected['nbins'] and loc['min_n_ticks'] == expected['min_n_ticks']
             and _float_digest(loc['steps']) == provider['locator']['default_steps_sha256']
             and _float_digest(loc['extended_steps']) == provider['locator']['extended_steps_sha256'],
             'original locator configuration differs from the native provider')
    levels = sample['levels']
    ends = bytes.fromhex(levels['data_hex'])
    vmin, = struct.unpack_from('<d', ends)
    vmax, = struct.unpack_from('<d', ends, len(ends) - 8)
    autoscale = sample['norm_autoscale']
    _require(any(row['before'] == [None, None] for row in autoscale) and all(
        row['inputs'] == levels and row['before'][0] in (None, vmin)
        and row['before'][1] in (None, vmax) and row['after'] == [vmin, vmax]
        for row in autoscale), 'normalization requires fresh limits from the original levels')
    cmap = provider['colormap']
    for collection in sample['collections']:
        mapping = collection['mapped']
        normal = mapping['norm']
        observed_cmap = mapping['color_map']
        _require(mapping['normalize'] is True and mapping['bytes'] is False and mapping['alpha'] is False
                 and not normal['clip'] and not normal['masked']
                 and (normal['vmin'], normal['vmax']) == (vmin, vmax)
                 and observed_cmap['name'] == cmap['name'] and observed_cmap['size'] == cmap['size']
                 and _float_digest(observed_cmap['lut']) == cmap['lut_sha256'],
                 'original normalization or palette differs from the native provider')
    return loc['nbins'], loc['min_n_ticks']


def _template(sample, observation, plan, parameters):
    original = read_artifact(sample['scene'], json_document=False)
    mesh = memoryview(read_artifact(sample['input'], json_document=False))
    rows, cols = sample['grid']['shape']
    roots = native_contour_agg._root_bindings(
        observation['grid'].get('lineage'), [rows, cols])
    _require(sample['provider']['masked'] is False, 'mask topology still requires its original provider')
    cy, cx = sample['provider']['chunk_size']
    version = b'SCARCTT3' if plan['title'] is not None else b'SCARCTT2'
    output = bytearray(version + struct.pack('<6Q', rows, cols, cx, cy,
                       sample['provider']['corner_mask'], 3 if roots else 0))
    if roots:
        for axis in ('x', 'y', 'z'):
            route = roots[axis]
            output.extend(struct.pack('<4Q', *route['root_shape'], *route['start']))
    for axis in ('x', 'y'):
        start = sample['grid']['offsets'][axis]
        output.extend(sha256(mesh[start:start + 8 * rows * cols]).digest())
    output.extend(struct.pack('<IIQ', *parameters, len(plan['steps'])))
    for step in plan['steps']:
        operation = step['operation']
        if operation == 'ORIGINAL_RESIDUAL':
            start, end = (step['span'][key] for key in ('start', 'end'))
            output.extend(struct.pack('<BQ', 0, end - start) + original[start:end])
        elif operation == 'EMIT_FILLED_BAND_PATHS':
            output.extend(struct.pack('<Bd', 1, step['simplify_threshold']))
        elif operation == 'EMIT_BAND_FACE_COLORS':
            output.extend(struct.pack('<B', 2))
        elif operation == 'UPDATE_DRAW_COUNT':
            output.extend(struct.pack('<BI', 3, step['unmodified_draws']))
        elif operation == 'EMIT_TITLE_GLYPHS':
            prefix = bytes.fromhex(step['command_prefix'])
            output.extend(struct.pack('<BI', 4, len(prefix)) + prefix)
        else:
            raise ValueError('unknown live draw argument')
    return bytes(output), roots


def _title_projection(title, records, digest):
    matches = [sample for sample in title['samples'] if sample['scene']['sha256'] == digest]
    _require(len(matches) == 1, 'title evidence must cover each exact original draw tape')
    sample, = matches
    native = sample['native']
    plan = dict(font=sample['font'], canvas=sample['canvas'], dpi=sample['dpi'],
                anchor=sample['anchor'], points=sample['points'], load_flags=sample['load_flags'],
                route=title['format_route'], value=sample['formal_value'], worker=title['worker'])
    return plan, records[sample['frame_index']], native


def _prepare(contour_plan, contour_provider, style_observation, style_provider, title_plan=None):
    contour_ref, contour_provider_ref = contour_plan, contour_provider
    style_ref, provider_ref = style_observation, style_provider
    plan = read_contour_scene_plan(contour_ref['uri'])
    actual_style_ref, observed = read_style(style_ref['uri'])
    _require(plan == read_artifact(contour_ref) and actual_style_ref == style_ref
             and plan['observation'] == observed['contours'] and plan['agg_scene'] == observed['agg_scene'],
             'style and paths must be sampled from the same original AGG execution')
    for original_producer in observed['sources'].values():
        _source_code(original_producer)
    contour_provider, contents = native_contour_agg.contour_members(contour_provider_ref)
    provider_path = Path(provider_ref['uri']).parent
    provider = native_contour_style.check(provider_path)
    _require(provider == read_artifact(provider_ref), 'native style package identity differs')
    originals = provider['original_source_sha256']
    _require(read_artifact(plan['agg_scene'])['matplotlib_version'] == provider['matplotlib_version']
             and all(source['material']['sha256'] == originals[name]
                 for source in observed['sources'].values()
                 if (name := Path(source['material']['uri']).name) in originals),
             'observed original provider source differs from the C++ lowering contract')
    for name in ('include/scar_generated/contour_style.hpp', 'include/scar_generated/contour_style_lut.hpp',
                 'src/contour_style.cpp', 'licenses/Matplotlib-LICENSE.txt', 'licenses/NumPy-LICENSE.txt'):
        contents[name] = (provider_path / name).read_bytes()
    contents['src/contour_agg_io.hpp'] = (_TEMPLATES / 'native_contour_agg_io.hpp.in').read_bytes() + b'\n'
    contents['src/main.cpp'] = (_TEMPLATES / 'native_contour_agg_v2.cpp.in').read_bytes()
    style_by_frame = {row['frame_index']: row for row in observed['samples']}
    _require(len(style_by_frame) == len(observed['samples']) == len(plan['samples']),
             'each observed frame must identify one qualified contour and its style producer')
    contour_observations = read_artifact(plan['observation'])['samples']
    title, titles = None, None
    if title_plan is not None:
        title = source_agg_text.check(title_plan['uri'])
        _require(title == read_artifact(title_plan), 'original native title plan identity differs')
        titles = read_artifact(title['observation'])['records']
    scenes = []
    for index, sample in enumerate(plan['samples']):
        _require(sample['provider']['version'] == contour_provider['contourpy_version'],
                 'native contour provider version differs')
        frame = sample['frame_index']
        style = style_by_frame[frame]
        parameters = _configuration(style, provider)
        title_binding, title_sample, title_native = (None, None, None)
        if title is not None:
            title_binding, title_sample, title_native = _title_projection(title, titles, sample['scene']['sha256'])
        projection = plan_sample(sample, style, title=title_sample)
        projection['title'] = title_binding
        if title_native is not None:
            slot, = [step for step in projection['steps'] if step['operation'] == 'EMIT_TITLE_GLYPHS']
            _require(slot['span'] == title_native['span']
                     and slot['command_prefix'] == title_native['gc_prefix'],
                     'native title proof does not bind the exact composed AGG slot')
        locator = sample['source_observation']['locator'].removeprefix('#/samples/')
        wire, roots = _template(sample, contour_observations[int(locator)], projection, parameters)
        name = 'scenes/layout.ctt' if title is not None else f'scenes/sample-{index}.ctt'
        _require(contents.setdefault(name, wire) == wire,
                 'remaining layout differs with live path, color, and title inputs removed')
        scenes.append(dict(projection=projection, template=name, original_root_views=roots,
                           original_levels=len(sample['intervals']) + 1, witness_input=sample['input']))
    return scenes, contents


def _manifest(root, references, scenes, contents, libraries):
    return dict(schema=SCHEMA, schema_version=1, **references, scenes=scenes,
                files={name: {**_ref(root / name, 'scar.native-contour-agg-member'), 'uri': name}
                       for name in (*contents, EXE, 'CMakeLists.txt')},
                cxx_standard=20, linked_libraries=sorted(libraries), python_runtime_required=False,
                levels_paths_and_colormap_runtime_computed=True, exact_original_draw_tapes=True,
                titles_runtime_computed='title_plan' in references,
                original_residual_layout_required=True, residual_layout_mesh_digest_guard=True,
                performance_measured=False,
                default_backend='NO_OP', whole_application_authorized=False)


def _project(root):
    write_native_project(root, TARGET, sources=(*native_contour_agg._SOURCES, 'src/contour_style.cpp'),
                         include_dirs=('vendor',), compile_options=('-fno-fast-math', '-ffp-contract=off'),
                         libraries=('crypto',),
                         install_dirs=(('licenses', '.'), ('scenes', '.')))


def _make_glyphs(title, output):
    width, height = title['canvas']
    _require(width > 0 and height > 0, 'title has no bounded canvas')
    return [title['worker']['uri'], title['font']['uri'],
            title['route']['prefix'], title['route']['suffix'], repr(title['value']),
            *(repr(x) for x in title['anchor']), str(height), str(title['points']),
            str(title['dpi']), str(title['load_flags']), str(output), '--agg-wire']


def export(plan, contour_provider, observation, style_provider, out, *, title_plan=None):
    references = dict(contour_plan=_ref(plan, PLAN),
                      contour_provider=_ref(Path(contour_provider) / 'result.json', native_contour_scene.SCHEMA),
                      style_observation=_ref(observation, OBSERVATION),
                      style_provider=_ref(Path(style_provider) / 'result.json', native_contour_style.SCHEMA))
    if title_plan:
        references['title_plan'] = _ref(title_plan, source_agg_text.SCHEMA)
    scenes, contents = _prepare(**references)
    out = Path(out).absolute()
    _require(not out.exists() and not out.is_symlink(), 'fresh live contour package directory required')
    compiler = shutil.which('g++')
    _require(compiler is not None, 'GNU C++20 compiler unavailable')
    with TemporaryDirectory(prefix='.scar-live-contour-', dir=out.parent) as temp:
        root = Path(temp) / 'package'
        for name, data in contents.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        _project(root)
        exe, libraries = build_native_project(root, TARGET, (compiler,))
        for index, scene in enumerate(scenes):
            dst = Path(temp) / f'scene-{index}.agg'
            title = scene['projection']['title']
            glyph_input = []
            if title is not None:
                glyph = Path(temp) / f'title-{index}.glyphs'
                subprocess.run(_make_glyphs(title, glyph), env={'LC_ALL': 'C'}, check=True, timeout=10,
                               capture_output=True)
                glyph_input.append(str(glyph))
            subprocess.run([str(exe), str(root / scene['template']), str(dst), scene['witness_input']['uri'],
                            *glyph_input], env={}, check=True, timeout=20)
            original = read_artifact(scene['projection']['original_scene'], json_document=False)
            _require(dst.read_bytes() == original, 'live levels, geometry or palette differed from original AGG bytes')
        report = _manifest(root, references, scenes, contents, libraries)
        (root / 'result.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
        root.rename(out)
    return report


def check(package):
    root = Path(package).resolve(strict=True)
    report = read_artifact(_ref(root / 'result.json', SCHEMA))
    references = {name: report[name] for name in ('contour_plan', 'contour_provider',
                                                  'style_observation', 'style_provider', 'title_plan')
                  if name in report}
    scenes, contents = _prepare(**references)
    _require(report == _manifest(root, references, scenes, contents, report['linked_libraries']),
             'live contour package differs during replay')
    for name, payload in contents.items():
        _require((root / name).read_bytes() == payload, 'compiled input differs: ' + name)
    with TemporaryDirectory(prefix='scar-live-scene-layout-') as temp:
        _project(Path(temp))
        _require((Path(temp) / 'CMakeLists.txt').read_bytes() == (root / 'CMakeLists.txt').read_bytes(),
                 'live contour project differs')
    return report
