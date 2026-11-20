"""Deliver the native Matplotlib AGG provider for an observed drawing interface."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen.native_project import build_native_project, write_native_project
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.trace.agg_scene import SCHEMA as OBSERVATION

SCHEMA = 'scar.native-agg-scene'
_TARGET = 'scar_agg_scene'
_EXECUTABLE = 'bin/' + _TARGET
_MPL = ('_backend_agg.cpp', '_backend_agg.h', '_backend_agg_basic_types.h',
        'path_converters.h', 'mplutils.h', 'array.h', 'agg_workaround.h')
_UNITS = ('agg_bezier_arc.cpp', 'agg_curves.cpp', 'agg_image_filters.cpp', 'agg_trans_affine.cpp',
          'agg_vcgen_contour.cpp', 'agg_vcgen_dash.cpp', 'agg_vcgen_stroke.cpp', 'agg_vpgen_segmentator.cpp')


def _require(ok, reason):
    if not ok:
        raise ValueError('native AGG scene: ' + reason)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _replace(source, original, native):
    _require(source.count(original) == 1, 'original backend binding layout differs')
    return source.replace(original, native)


def _mpl_sources(inputs):
    root = Path(__file__).parent
    result = {name: read_artifact(inputs[name], json_document=False).decode() for name in _MPL}
    pybind = '#include <pybind11/pybind11.h>\n'
    result['_backend_agg.cpp'] = _replace(result['_backend_agg.cpp'], '#include <Python.h>\n', '')
    paths = _replace(result['path_converters.h'], pybind, '')
    begin = 'namespace PYBIND11_NAMESPACE { namespace detail {'
    end = '}} // namespace PYBIND11_NAMESPACE::detail\n'
    _require(paths.count(begin) == paths.count(end) == 1, 'original snap caster boundary differs')
    before, remainder = paths.split(begin)
    result['path_converters.h'] = before + remainder.split(end)[1]
    basics = _replace(result['_backend_agg_basic_types.h'], pybind, '')
    basics = _replace(basics, '#include "py_adaptors.h"\n\nnamespace py = pybind11;',
                      '#include "scar_agg_adaptors.hpp"')
    cut = 'namespace PYBIND11_NAMESPACE { namespace detail {'
    _require(basics.count(cut) == 1 and basics.endswith('#endif\n'), 'original graphics caster boundary differs')
    result['_backend_agg_basic_types.h'] = basics.split(cut)[0] + '#endif\n'
    backend = _replace(result['_backend_agg.h'], pybind, '#include <cstring>\n#include <stdexcept>\n')
    backend = _replace(backend, 'namespace py = pybind11;', 'namespace py { using value_error = std::invalid_argument; }')
    result['_backend_agg.h'] = _replace(backend,
        '    ColorArray hatchcolors = py::array_t<double>().reshape({0, 4}).unchecked<double, 2>();',
        '    ColorArray hatchcolors{};')
    utils = _replace(result['mplutils.h'], '#include <Python.h>\n', '')
    cut = '#ifdef __cplusplus  // not for macosx.m'
    _require(utils.count(cut) == 1, 'original plot utility binding differs')
    result['mplutils.h'] = utils.split(cut)[0] + '''
#include <array>
#include <cstddef>
#include <stdexcept>
template<class T, std::size_t N>
constexpr std::size_t safe_first_shape(const std::array<T, N>&) { return N; }
template<class T>
void check_trailing_shape(const T& value, const char*, long d1, long d2) {
    if (value.shape(1) != d1 || value.shape(2) != d2) throw std::invalid_argument("AGG shape differs");
}
#endif
'''
    prepared = {f'vendor/mpl/{name}': value.encode() for name, value in result.items()}
    prepared['include/scar_generated/scar_agg_adaptors.hpp'] = (root / 'native_agg_adaptors_v1.hpp.in').read_bytes()
    prepared['src/main.cpp'] = (root / 'native_agg_scene_v1.cpp.in').read_bytes()
    for name, ref in inputs.items():
        if name.startswith(('agg/include/', 'agg/src/', 'licenses/')):
            path = name if name.startswith('licenses/') else 'vendor/' + name
            prepared[path] = read_artifact(ref, json_document=False)
    return prepared


def _project(root):
    write_native_project(root, _TARGET,
        sources=('src/main.cpp', 'vendor/mpl/_backend_agg.cpp', *(f'vendor/agg/src/{name}' for name in _UNITS)),
        include_dirs=('vendor/mpl', 'vendor/agg/include'), install_dirs=(('licenses', '.'),))


def _observations(reference):
    report = read_artifact(reference)
    base = Path(reference['uri']).parent
    samples = []
    for frame in report['frames']:
        names = frame['scene'], frame['raster']
        _require(all(Path(name).name == name for name in names), 'invalid observed scene member')
        tape = _ref(base / names[0], 'scar.agg-scene-tape')
        raster = _ref(base / names[1], 'scar.packed-rgba8')
        _require(tape['sha256'] == frame['scene_sha256'] and raster['sha256'] == frame['raster_sha256'],
                 'observed scene or raster differs')
        samples.append(dict(scene=tape, raster=raster, operations=frame['operations']))
    _require(samples, 'at least one original raster witness required')
    return report['matplotlib_version'], samples


def export(observation, upstream, out):
    source = Path(upstream).resolve(strict=True)
    original = _ref(observation, OBSERVATION)
    version, samples = _observations(original)
    inputs = {name: _ref(source / 'src' / name, 'scar.upstream-source') for name in _MPL}
    inputs['licenses/Matplotlib-LICENSE.txt'] = _ref(source / 'LICENSE/LICENSE', 'scar.upstream-license')
    agg = source / 'extern/agg24-svn'
    for name in ('copying', 'COPYING', 'LICENSE'):
        if (agg / name).is_file():
            inputs['licenses/AGG-LICENSE.txt'] = _ref(agg / name, 'scar.upstream-license')
            break
    for header in sorted((agg / 'include').glob('*.h')):
        inputs['agg/include/' + header.name] = _ref(header, 'scar.upstream-source')
    for name in _UNITS:
        inputs['agg/src/' + name] = _ref(agg / 'src' / name, 'scar.upstream-source')
    prepared = _mpl_sources(inputs)
    out = Path(out).absolute()
    _require(not out.exists() and not out.is_symlink(), 'fresh package path required')
    compiler = shutil.which('g++')
    _require(compiler is not None, 'GNU C++20 compiler unavailable')
    with TemporaryDirectory(prefix='.scar-agg-', dir=out.parent) as temp:
        root = Path(temp) / 'package'
        for name, data in prepared.items():
            local = root / name
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
        _project(root)
        exe, libraries = build_native_project(root, _TARGET, (compiler,))
        for index, row in enumerate(samples):
            target = Path(temp) / f'frame-{index}.rgba'
            subprocess.run([str(exe), row['scene']['uri'], str(target)], env={}, check=True, timeout=20)
            _require(target.read_bytes() == read_artifact(row['raster'], json_document=False),
                     f'native scene {index} raster differs')
        file_names = (*prepared, _EXECUTABLE, 'CMakeLists.txt')
        report = dict(schema=SCHEMA, schema_version=1, observation=original, matplotlib_version=version,
            original_sources=inputs, samples=samples,
            files={name: {**_ref(root / name, 'scar.native-agg-member'), 'uri': name} for name in file_names},
            cxx_standard=20, linked_libraries=sorted(libraries), same_raster_bytes=True,
            python_runtime_required=False, consumes_draw_operations_at_runtime=True,
            original_scene_production_lowered=False, deployment_authorized=False)
        (root / 'result.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
        root.rename(out)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation')
    parser.add_argument('--upstream')
    parser.add_argument('--out')
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    if args.check:
        report = check(args.check)
    elif args.observation and args.upstream and args.out:
        report = export(args.observation, args.upstream, args.out)
    else:
        parser.error('provide --check PACKAGE or --observation, --upstream and --out')
    print(json.dumps(dict(samples=len(report['samples']), same_raster_bytes=True,
                          linked_libraries=report['linked_libraries'])))


def check(package):
    root = Path(package).resolve(strict=True)
    report = read_artifact(_ref(root / 'result.json', SCHEMA))
    version, samples = _observations(report['observation'])
    prepared = _mpl_sources(report['original_sources'])
    _require(report['samples'] == samples and report['matplotlib_version'] == version,
             'observed scene profile differs')
    _require(set(report['files']) == {*prepared, _EXECUTABLE, 'CMakeLists.txt'}, 'package member set differs')
    for name, row in report['files'].items():
        _require(row['uri'] == name, 'invalid relative AGG member')
        read_artifact({**row, 'uri': str(root / name)}, json_document=False)
    for name, data in prepared.items():
        _require((root / name).read_bytes() == data, 'native graphics source differs: ' + name)
    with TemporaryDirectory(prefix='scar-agg-check-') as temp:
        _project(Path(temp))
        _require((root / 'CMakeLists.txt').read_bytes() == (Path(temp) / 'CMakeLists.txt').read_bytes(),
                 'AGG build metadata differs')
    return report


if __name__ == '__main__':
    main()
