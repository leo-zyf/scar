"""Build the original ContourPy MPL2014 filled-contour provider for C++20."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from tempfile import TemporaryDirectory

from scar.codegen.native_project import build_native_project, write_native_project
from scar.ir.execution_recipe import artifact_reference, read_artifact

SCHEMA = 'scar.native-contour-scene'
_VERSION = '1.4.0'
_ARCHIVE_SCHEMA = 'scar.contourpy-upstream-archive'
_ARCHIVE_SHA256 = '1d40b8032984dab608eb438a4fa1b075be39209cc02e762458da106082568d95'
_TARGET = 'scar_contour_scene'
_EXECUTABLE = 'bin/' + _TARGET
_UPSTREAM = ('src/common.h', 'src/contour_generator.h', 'src/contour_generator.cpp',
             'src/mpl2014.h', 'src/mpl2014.cpp', 'src/mpl_kind_code.h', 'LICENSE')
_TEMPLATES = {
    'include/scar_generated/contour_scene.hpp': 'native_contour_scene.hpp.in',
    'include/scar_generated/contour_adaptors.hpp': 'native_contour_adaptors.hpp.in',
    'src/contour_scene.cpp': 'native_contour_scene.cpp.in',
    'src/main.cpp': 'native_contour_scene_main.cpp.in',
}


def _require(valid, reason):
    if not valid:
        raise ValueError('native contour scene: ' + reason)


def _replace(source, original, native, *, count=1):
    _require(source.count(original) == count, 'original provider boundary differs: ' + original)
    return source.replace(original, native)


def _cut(source, start, end):
    _require(source.count(start) == source.count(end) == 1, 'original provider boundary differs')
    before, remainder = source.split(start)
    _require(end in remainder, 'original provider boundary order differs')
    return before + end + remainder.split(end, 1)[1]


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _original(reference):
    _require(reference['schema'] == _ARCHIVE_SCHEMA and reference['sha256'] == _ARCHIVE_SHA256,
             'expected the pinned ContourPy ' + _VERSION + ' source archive')
    raw = read_artifact(reference, json_document=False)
    with tarfile.open(fileobj=BytesIO(raw), mode='r:gz') as archive:
        result = {}
        for name in _UPSTREAM:
            member = archive.getmember(f'contourpy-{_VERSION}/{name}')
            _require(member.isfile(), 'upstream member is not a file: ' + name)
            with archive.extractfile(member) as stream:
                result[name] = stream.read()
    return result


def _provider_sources(original):
    source = {name: value.decode() for name, value in original.items() if name.startswith('src/')}
    common = source['src/common.h']
    common = _replace(common, '#include <pybind11/pybind11.h>\n', '#include "contour_adaptors.hpp"\n')
    source['src/common.h'] = _cut(common, '#include <pybind11/numpy.h>', '#endif // CONTOURPY_COMMON_H')

    base = source['src/contour_generator.h']
    base = _replace(base, 'virtual py::tuple filled', 'virtual OuterCode filled')
    base = _replace(base, 'virtual py::sequence lines', 'virtual OuterCode lines')
    base = _cut(base, '    virtual py::list multi_filled', 'protected:')
    source['src/contour_generator.h'] = _replace(
        base, '    void check_levels(const LevelArray& levels, bool filled) const;\n', '')

    check = source['src/contour_generator.cpp']
    check = _replace(check, '#include "util.h"', '#include <cmath>')
    check = _cut(check, 'void ContourGenerator::check_levels(const LevelArray& levels',
                 'void ContourGenerator::check_levels(double lower_level')
    check = _cut(check, 'py::list ContourGenerator::multi_filled', '} // namespace contourpy')
    source['src/contour_generator.cpp'] = _replace(check, 'Util::is_nan', 'std::isnan', count=2)

    for name in ('src/mpl2014.h', 'src/mpl2014.cpp'):
        implementation = source[name]
        for before, after, count in (
            ('py::tuple', 'OuterCode', 3),
            ('py::sequence', 'OuterCode', 1),
            ('OuterCode ' + ('Mpl2014ContourGenerator::' if name.endswith('.cpp') else '') +
             'get_chunk_', 'std::pair<index_t, index_t> ' +
             ('Mpl2014ContourGenerator::' if name.endswith('.cpp') else '') + 'get_chunk_', 2),
            ('py::list& vertices_list', 'PointLists& vertices_list', 3),
            ('py::list& codes_list', 'CodeLists& codes_list', 3),
        ):
            implementation = _replace(implementation, before, after, count=count)
        if name.endswith('.cpp'):
            for before, after, count in (
                ('py::ssize_t', 'index_t', 9),
                ('py::list vertices, codes;', 'PointLists vertices;\n    CodeLists codes;', 1),
                ('py::list vertices_list, codes_list;', 'PointLists vertices_list;\n    CodeLists codes_list;', 1),
                ('vertices.mutable_data()', 'vertices.data()', 2),
                ('codes.mutable_data()', 'codes.data()', 2),
                ('vertices_list.append(vertices)', 'vertices_list.push_back(std::move(vertices))', 2),
                ('codes_list.append(codes)', 'codes_list.push_back(std::move(codes))', 2),
                ('py::make_tuple(vertices, codes)', '{std::move(vertices), std::move(codes)}', 1),
                ('py::make_tuple(vertices_list, codes_list)',
                 '{std::move(vertices_list), std::move(codes_list)}', 1),
                ('py::make_tuple(_nychunk, _nxchunk)', '{_nychunk, _nxchunk}', 1),
                ('py::make_tuple(_y_chunk_size, _x_chunk_size)', '{_y_chunk_size, _x_chunk_size}', 1),
            ):
                implementation = _replace(implementation, before, after, count=count)
            for indent in (' ' * 4, ' ' * 12):
                implementation = _replace(implementation,
                    f'index_t vertices_dims[2] = {{npoints, 2}};\n{indent}PointArray vertices(vertices_dims);',
                    'std::vector<double> vertices(2 * npoints);')
                implementation = _replace(implementation,
                    f'index_t codes_dims[1] = {{npoints}};\n{indent}CodeArray codes(codes_dims);',
                    'std::vector<std::uint8_t> codes(npoints);')
        source[name] = implementation

    _require(all('py::' not in value and 'pybind11/' not in value for value in source.values()),
             'unbound upstream Python adapter')
    prepared = {'vendor/' + Path(name).name: value.encode() for name, value in source.items()}
    prepared['licenses/ContourPy-LICENSE.txt'] = original['LICENSE']
    templates = Path(__file__).parent
    prepared.update({name: (templates / path).read_bytes() for name, path in _TEMPLATES.items()})
    return prepared


def _project(root):
    write_native_project(root, _TARGET,
        sources=('src/main.cpp', 'src/contour_scene.cpp', 'vendor/contour_generator.cpp', 'vendor/mpl2014.cpp'),
        include_dirs=('vendor',), compile_options=('-fno-fast-math', '-ffp-contract=off'),
        install_dirs=(('licenses', '.'), ('include', '.')))


def _verify_samples(executable, samples):
    with TemporaryDirectory(prefix='scar-contour-verify-') as temp:
        target = Path(temp) / 'actual.cto'
        for index, sample in enumerate(samples):
            source, expected = sample['input'], sample['expected']
            _require(read_artifact(source, json_document=False).startswith(b'SCARCTI1') and
                     expected['schema'] == 'scar.contour-outer-code-v1', 'sample interface differs')
            expected_bytes = read_artifact(expected, json_document=False)
            _require(expected_bytes.startswith(b'SCARCTO1'), 'sample output interface differs')
            subprocess.run([str(executable), source['uri'], str(target)], env={}, check=True, timeout=25)
            _require(target.read_bytes() == expected_bytes,
                     f'native OuterCode bytes differ for sample {index}')


def export(upstream, out, samples=()):
    """Assemble the pinned provider and compare any supplied original result witnesses."""
    reference = _ref(upstream, _ARCHIVE_SCHEMA)
    original = _original(reference)
    prepared = _provider_sources(original)
    sample_refs = [dict(input=_ref(source, 'scar.contour-grid-v1'),
                        expected=_ref(expected, 'scar.contour-outer-code-v1'))
                   for source, expected in samples]
    out = Path(out).absolute()
    _require(not out.exists() and not out.is_symlink(), 'fresh package output required')
    compiler = shutil.which('g++')
    _require(compiler is not None, 'GNU C++20 compiler unavailable')
    with TemporaryDirectory(prefix='.scar-contour-', dir=out.parent) as temp:
        root = Path(temp) / 'package'
        for name, data in prepared.items():
            local = root / name
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
        _project(root)
        executable, libraries = build_native_project(root, _TARGET, (compiler,))
        _verify_samples(executable, sample_refs)
        file_names = (*prepared, _EXECUTABLE, 'CMakeLists.txt')
        report = dict(schema=SCHEMA, schema_version=1, contourpy_version=_VERSION, algorithm='mpl2014',
            fill_type='OuterCode', upstream_archive=reference,
            upstream_source_sha256={name: sha256(data).hexdigest() for name, data in original.items()},
            files={name: {**_ref(root / name, 'scar.native-contour-member'), 'uri': name} for name in file_names},
            cxx_standard=20, linked_libraries=sorted(libraries), samples=sample_refs,
            same_outer_code_bytes=True if sample_refs else None, python_runtime_required=False,
            computes_geometry_from_runtime_grids_and_levels=True, deployment_authorized=False)
        (root / 'result.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
        root.rename(out)
    return report


def check(package):
    """Recheck provenance, generated sources and original output witnesses."""
    root = Path(package).resolve(strict=True)
    report = read_artifact(_ref(root / 'result.json', SCHEMA))
    original = _original(report['upstream_archive'])
    prepared = _provider_sources(original)
    _require(report['schema_version'] == 1 and report['algorithm'] == 'mpl2014' and
             report['contourpy_version'] == _VERSION and report['fill_type'] == 'OuterCode' and
             report['upstream_source_sha256'] == {name: sha256(data).hexdigest()
                                                 for name, data in original.items()} and
             report['deployment_authorized'] is False and report['python_runtime_required'] is False and
             report['same_outer_code_bytes'] is (True if report['samples'] else None),
             'provider identity differs')
    _require(set(report['files']) == {*prepared, _EXECUTABLE, 'CMakeLists.txt'}, 'package members differ')
    for name, row in report['files'].items():
        _require(row['uri'] == name, 'package member path differs')
        read_artifact({**row, 'uri': str(root / name)}, json_document=False)
    for name, data in prepared.items():
        _require((root / name).read_bytes() == data, 'adapted source differs: ' + name)
    with TemporaryDirectory(prefix='scar-contour-check-') as temp:
        candidate = Path(temp)
        _project(candidate)
        _require((root / 'CMakeLists.txt').read_bytes() == (candidate / 'CMakeLists.txt').read_bytes(),
                 'C++ build metadata differs')
    _verify_samples(root / _EXECUTABLE, report['samples'])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream')
    parser.add_argument('--out')
    parser.add_argument('--sample', nargs=2, action='append', default=[], metavar=('INPUT', 'EXPECTED'))
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    if args.check:
        report = check(args.check)
    elif args.upstream and args.out:
        report = export(args.upstream, args.out, args.sample)
    else:
        parser.error('provide --check PACKAGE or --upstream ARCHIVE and --out PACKAGE')
    print(json.dumps(dict(samples=len(report['samples']), same_outer_code_bytes=report['same_outer_code_bytes'],
                          linked_libraries=report['linked_libraries'])))


if __name__ == '__main__':
    main()
