"""Build Matplotlib contour level selection and colormap evaluation for C++20."""

import argparse
from hashlib import sha256
from importlib.metadata import distribution
from io import BytesIO
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tarfile
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from scar.analysis.source_agg_scene_layout import read_tape
from scar.codegen.native_project import build_native_project, write_native_project
from scar.ir.execution_recipe import artifact_reference, read_artifact

SCHEMA = 'scar.native-contour-style'
_MPL = '3.11.2'
_NP = '2.5.2'
_UPSTREAM_SHA256 = '14b555ae7477ff593d842fba9714b5cbc6ea1db9461d2ea9d6da3ec0dc420f1a'
_MODULES = ('contour.py', 'ticker.py', 'colors.py', 'transforms.py', 'cm.py', '_cm.py', '_cm_listed.py')
_TARGET = 'scar_contour_style'
_EXE = 'bin/' + _TARGET
_LUT = 'assets/lut.f64'
_LUT_HEADER = 'include/scar_generated/contour_style_lut.hpp'
_TEMPLATES = {'include/scar_generated/contour_style.hpp': 'native_contour_style.hpp.in',
              'src/contour_style.cpp': 'native_contour_style.cpp.in',
              'src/main.cpp': 'native_contour_style_main.cpp.in'}


def _require(valid, reason):
    if not valid:
        raise ValueError('native contour style: ' + reason)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _digest(raw):
    return sha256(raw).hexdigest()


def _upstream(reference):
    _require(reference['sha256'] == _UPSTREAM_SHA256, 'expected the pinned Matplotlib source archive')
    data = read_artifact(reference, json_document=False)
    with tarfile.open(fileobj=BytesIO(data), mode='r:gz') as source:
        prefix = f'matplotlib-{_MPL}/'
        modules = {}
        for name in _MODULES:
            with source.extractfile(prefix + 'lib/matplotlib/' + name) as member:
                modules[name] = member.read()
        with source.extractfile(prefix + 'LICENSE/LICENSE') as member:
            return modules, member.read()


def _profile(modules, name, parameters):
    import matplotlib
    from matplotlib import colors, ticker
    import numpy as np

    _require(matplotlib.__version__ == _MPL and np.__version__ == _NP and
             matplotlib.rcParams['axes.autolimit_mode'] == 'data', 'original numerical provider differs')
    directory = Path(matplotlib.__file__).parent
    for path, source in modules.items():
        _require((directory / path).read_bytes() == source, 'live original source differs: ' + path)
    cmap = matplotlib.colormaps[name]
    _require(type(cmap) in (colors.LinearSegmentedColormap, colors.ListedColormap) and
             1 <= cmap.N <= 16381, 'unsupported original Matplotlib colormap')
    cmap._ensure_inited()
    lut = np.asarray(cmap._lut, dtype='<f8')
    _require(lut.shape == (cmap.N + 3, 4) and np.isfinite(lut).all()
             and (cmap._i_under, cmap._i_over, cmap._i_bad) == (cmap.N, cmap.N + 1, cmap.N + 2),
             'colormap table layout differs')
    _require(1 <= parameters['nbins'] <= 1000 and 1 <= parameters['min_n_ticks'] <= parameters['nbins'] + 1,
             'MaxNLocator parameters exceed native profile')
    provider = ticker.MaxNLocator(**parameters)
    steps = np.asarray(provider._steps, dtype='<f8')
    staircase = np.asarray(provider._extended_steps, dtype='<f8')
    _require(steps.tolist() == [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10] and
             provider._prune is None and not provider._symmetric and not provider._integer,
             'unsupported MaxNLocator settings')
    license_path = distribution('numpy').locate_file(f'numpy-{_NP}.dist-info/licenses/LICENSE.txt')
    numpy_license = _ref(license_path, 'scar.upstream-license')
    raw = lut.tobytes()
    colormap = dict(name=cmap.name, size=cmap.N, type=type(cmap).__name__, lut_sha256=_digest(raw))
    locator = dict(default_steps_sha256=_digest(steps.tobytes()),
                   extended_steps_sha256=_digest(staircase.tobytes()), offset_policy='REQUIRE_ZERO')
    return raw, colormap, locator, numpy_license


def _source_files(lut, parameters, mpl_license, numpy_license):
    template_root = Path(__file__).parent
    rendered = {name: (template_root / template).read_bytes() for name, template in _TEMPLATES.items()}
    rows = list(struct.iter_unpack('<4d', lut))
    constants = ',\n'.join('    Rgba{' + ', '.join(value.hex() for value in row) + '}' for row in rows)
    header = f'''#pragma once
#include "contour_style.hpp"

namespace scar::contour {{
inline constexpr StyleParameters style_parameters{{{parameters['nbins']}, {parameters['min_n_ticks']}}};
inline constexpr std::array<Rgba, {len(rows)}> style_lut{{{{
{constants}
}}}};
}}
'''
    rendered[_LUT_HEADER] = header.encode()
    rendered[_LUT] = lut
    rendered['licenses/Matplotlib-LICENSE.txt'] = mpl_license
    rendered['licenses/NumPy-LICENSE.txt'] = read_artifact(numpy_license, json_document=False)
    return rendered


def _sample(contour_input, agg_input, cmap, parameters):
    import matplotlib
    from matplotlib import colors, ticker
    from matplotlib.contour import ContourSet
    import numpy as np

    raw = read_artifact(contour_input, json_document=False)
    _require(raw.startswith(b'SCARCTI1') and len(raw) >= 64, 'invalid original contour input')
    rows, cols, count, _, _, _, mask = struct.unpack_from('<7Q', raw, 8)
    points = rows * cols
    _require(rows >= 2 and cols >= 2 and count >= 2 and mask == 0 and
             len(raw) == 64 + 8 * (3 * points + count), 'unsupported original contour input shape')
    z = np.frombuffer(raw, dtype='<f8', offset=64 + 16 * points, count=points)
    provided = np.frombuffer(raw, dtype='<f8', offset=64 + 24 * points, count=count)
    _require(np.isfinite(z).all(), 'original sample contains nonfinite data')
    obj = SimpleNamespace(zmin=float(z.min()), zmax=float(z.max()), extend='neither', logscale=False,
                          filled=True, locator=ticker.MaxNLocator(**parameters))
    obj.levels = ContourSet._autolev(obj)
    ContourSet._process_levels(obj)
    lows, ups = ContourSet._get_lowers_and_uppers(obj)
    effective = np.concatenate((lows[:1], ups))
    _require(effective.tobytes() == provided.tobytes(), 'original locator differs from observed levels')
    norm = colors.Normalize()
    norm.autoscale_None(obj.levels)
    rgba = np.asarray(matplotlib.colormaps[cmap](norm(obj.layers)), dtype='<f8').tobytes()
    tape = read_artifact(agg_input, json_document=False)
    draws = []
    for draw in read_tape(tape)['commands']:
        if draw['operator'] == 'draw_path_collection':
            colors_field = draw['fields']['facecolors']
            span = colors_field['payload_span']
            if colors_field['shape'] == [count - 1, 4] and tape[span['start']:span['end']] == rgba:
                draws.append(draw['draw_ordinal'])
    _require(draws, 'original AGG has no matching facecolor collection')
    expected = (b'SCARCST1' + struct.pack('<Q', count) + obj.levels.astype('<f8').tobytes() +
                effective.astype('<f8').tobytes() + obj.layers.astype('<f8').tobytes() + rgba)
    return dict(input=contour_input, agg_scene=agg_input, zmin=obj.zmin.hex(), zmax=obj.zmax.hex(),
                level_count=count, observed_color_draw_ordinals=draws), expected


def _project(root):
    write_native_project(root, _TARGET, sources=('src/main.cpp', 'src/contour_style.cpp'),
        compile_options=('-fno-fast-math', '-ffp-contract=off'),
        install_dirs=(('licenses', '.'), ('include', '.'), ('assets', '.')))


def _bind_output(sample, expected):
    raw = read_artifact(sample['input'], json_document=False)
    _require(raw[:8] == b'SCARCTI1' and len(raw) >= 64, 'original contour input differs')
    rows, cols, count, _, _, _, mask = struct.unpack_from('<7Q', raw, 8)
    start = 64 + 24 * rows * cols
    _require(count == sample['level_count'] and 2 <= count <= 16384 and mask == 0 and
             len(raw) == start + 8 * count and expected[:16] == b'SCARCST1' + struct.pack('<Q', count) and
             len(expected) == 16 + 16 * count + 40 * (count - 1),
             'style witness shape differs')
    _require(expected[16 + 8 * count:16 + 16 * count] == raw[start:],
             'style witness levels differ from original contour provider input')
    rgba = expected[16 + 16 * count + 8 * (count - 1):]
    tape = read_artifact(sample['agg_scene'], json_document=False)
    commands = read_tape(tape)['commands']
    indices = sample['observed_color_draw_ordinals']
    _require(indices and indices == sorted(set(indices)), 'expected original AGG color draws')
    for index in indices:
        _require(type(index) is int and 0 <= index < len(commands) and
                 commands[index]['operator'] == 'draw_path_collection', 'original AGG color draw differs')
        field = commands[index]['fields']['facecolors']
        span = field['payload_span']
        _require(field['shape'] == [count - 1, 4] and tape[span['start']:span['end']] == rgba,
                 'style witness colors differ from original AGG draw')


def _verify(directory, samples):
    with TemporaryDirectory(prefix='scar-contour-style-check-') as stage:
        actual = Path(stage) / 'actual.style'
        for i, sample in enumerate(samples):
            expected = (directory / f'validation/sample-{i}.style').read_bytes()
            _bind_output(sample, expected)
            subprocess.run([directory / _EXE, sample['zmin'], sample['zmax'], actual],
                           env={}, check=True, timeout=10)
            _require(actual.read_bytes() == expected,
                     f'native levels, midpoints or colors differ for sample {i}')


def export(upstream, out, cmap, parameters, samples):
    original = _ref(upstream, 'scar.matplotlib-original-source')
    modules, mpl_license = _upstream(original)
    lut, colormap, locator, numpy_license = _profile(modules, cmap, parameters)
    files = _source_files(lut, parameters, mpl_license, numpy_license)
    checked = []
    for i, (contour_input, agg) in enumerate(samples):
        row, expected = _sample(_ref(contour_input, 'scar.contour-grid-v1'),
                                _ref(agg, 'scar.agg-scene-tape'), cmap, parameters)
        files[f'validation/sample-{i}.style'] = expected
        checked.append(row)
    out = Path(out).absolute()
    _require(not out.exists() and not out.is_symlink(), 'fresh package output required')
    compiler = shutil.which('g++')
    _require(compiler is not None, 'GNU C++20 compiler unavailable')
    with TemporaryDirectory(prefix='.scar-contour-style-', dir=out.parent) as temp:
        root = Path(temp) / 'package'
        for name, value in files.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)
        _project(root)
        _, libraries = build_native_project(root, _TARGET, (compiler,))
        _verify(root, checked)
        names = (*files, _EXE, 'CMakeLists.txt')
        report = dict(schema=SCHEMA, schema_version=1, upstream_archive=original, matplotlib_version=_MPL,
            numpy_version=_NP, numpy_license=numpy_license,
            original_source_sha256={name: _digest(value) for name, value in modules.items()},
            parameters=parameters, colormap=colormap, locator=locator, samples=checked,
            files={name: {**_ref(root / name, 'scar.native-contour-style-member'), 'uri': name} for name in names},
            same_style_bytes=True if checked else None, python_runtime_required=False,
            deployment_authorized=False, linked_libraries=sorted(libraries))
        (root / 'result.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
        root.rename(out)
    return report


def check(package):
    directory = Path(package).resolve(strict=True)
    report = read_artifact(_ref(directory / 'result.json', SCHEMA))
    modules, mpl_license = _upstream(report['upstream_archive'])
    lut = (directory / _LUT).read_bytes()
    _require(report['schema_version'] == 1 and report['matplotlib_version'] == _MPL and
             report['numpy_version'] == _NP and report['deployment_authorized'] is False and
             report['python_runtime_required'] is False and
             report['same_style_bytes'] is (True if report['samples'] else None) and
             report['original_source_sha256'] == {name: _digest(value) for name, value in modules.items()} and
             len(lut) == 32 * (report['colormap']['size'] + 3) and
             _digest(lut) == report['colormap']['lut_sha256'], 'style provider contract differs')
    source = _source_files(lut, report['parameters'], mpl_license, report['numpy_license'])
    names = {*source, _EXE, 'CMakeLists.txt',
             *(f'validation/sample-{i}.style' for i in range(len(report['samples'])))}
    _require(set(report['files']) == names, 'package member set differs')
    for name, value in report['files'].items():
        _require(value['uri'] == name, 'invalid package member')
        raw = read_artifact({**value, 'uri': str(directory / name)}, json_document=False)
        if name in source:
            _require(raw == source[name], 'generated style provider differs: ' + name)
    with TemporaryDirectory(prefix='scar-contour-style-manifest-') as stage:
        _project(Path(stage))
        _require((directory / 'CMakeLists.txt').read_bytes() == (Path(stage) / 'CMakeLists.txt').read_bytes(),
                 'style provider build metadata differs')
    _verify(directory, report['samples'])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream')
    parser.add_argument('--out')
    parser.add_argument('--cmap')
    parser.add_argument('--nbins', type=int)
    parser.add_argument('--min-n-ticks', type=int, default=1)
    parser.add_argument('--sample', nargs=2, action='append', default=[], metavar=('CTI', 'AGG'))
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    if args.check:
        report = check(args.check)
    elif args.upstream and args.out and args.cmap and args.nbins is not None:
        report = export(args.upstream, args.out, args.cmap,
                        dict(nbins=args.nbins, min_n_ticks=args.min_n_ticks), args.sample)
    else:
        parser.error('provide --check PACKAGE or --upstream ARCHIVE --out PACKAGE --cmap NAME --nbins N')
    print(json.dumps(dict(samples=len(report['samples']), same_style_bytes=report['same_style_bytes'],
                          parameters=report['parameters'], colormap=report['colormap'])))


if __name__ == '__main__':
    main()
