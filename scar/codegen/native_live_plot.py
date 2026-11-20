"""Link the source-qualified live contour, glyph, AGG and PNG components."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from tempfile import TemporaryDirectory

from scar.codegen import native_agg_scene, native_contour_agg_live, native_rgba_png
from scar.codegen.aten_provider import _dependencies
from scar.codegen.native_cpu_elf import cpu_executable_dynamic
from scar.codegen.native_project import _cmake_arg
from scar.ir.execution_recipe import artifact_reference, read_artifact

SCHEMA = 'scar.native-live-plot'
LIBRARY = 'lib/libscar_live_plot.so'
LAYOUT = 'assets/layout.ctt'
FONT = 'assets/font.ttf'
DEMO = 'bin/scar_live_plot_demo'
_CODE = Path(__file__).parent
_SHADOW = ('libharfbuzz-', 'libfreetype-', 'libpng16-', 'libbrotlidec-', 'libbrotlicommon-')
_SYSTEM = {'libstdc++.so.6', 'libc.so.6', 'libm.so.6', 'libgcc_s.so.1', 'libdl.so.2',
           'libcrypto.so.3', 'libpthread.so.0', 'linux-vdso.so.1', '/lib64/ld-linux-x86-64.so.2'}


def _require(condition, reason):
    if not condition:
        raise ValueError('native live plot: ' + reason)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _read(path):
    return Path(path).read_bytes()


def _member(root, report, name):
    row = report['files'][name]
    _require(row['uri'] == name, 'original package member moved: ' + name)
    return read_artifact({**row, 'uri': str(root / name)}, json_document=False)


def _replace(source, old, new):
    _require(source.count(old) == 1, 'unexpected provider implementation boundary: ' + old[:80])
    return source.replace(old, new)


def _prefix(source, boundary):
    _require(source.count(boundary) == 1, 'unexpected provider entry boundary: ' + boundary)
    return source.split(boundary, 1)[0]


def _between(source, start, end, replacement):
    prefix = _prefix(source, start)
    remainder = source[len(prefix):]
    _require(remainder.count(end) == 1, 'unexpected provider block end: ' + end)
    return prefix + replacement + end + remainder.split(end, 1)[1]


def _memory_tape(contour_io):
    source = '#include <cstring>\n#include <span>\n' + contour_io.decode()
    source = _replace(source, '    std::ifstream input_;', '    std::span<const char> input_;')
    source = _replace(source, '    explicit Tape(const char* path) : input_(path, std::ios::binary) {}',
                      '    explicit Tape(std::span<const char> input) : input_(input) {}')
    source = _replace(source, '''        total_ += size;
        if (size) require(bool(input_.read(static_cast<char*>(data), size)), "truncated contour template");''',
                      '''        require(size <= input_.size() - total_, "truncated contour template");
        if (size) std::memcpy(data, input_.data() + total_, size);
        total_ += size;''')
    source = _replace(source, 'input_.peek() == EOF && input_.eof()', 'total_ == input_.size()')
    source = _between(source, '    void publish(const char* path) const {', '\n};\n\nstd::uint64_t number',
                      '    std::vector<char> finish() && { return std::move(data_); }')
    return (_prefix(source, 'std::uint64_t number(') + '}\n').encode()


def _compose(contour):
    source = contour.decode()
    source = _between(source, 'namespace {\nGrid live_grid', 'std::pair<double, double> finite_bounds',
                      'namespace {\n')
    helpers = _prefix(source, 'struct Glyph {')
    compute = source.split('        auto grid = live_grid(tape, argc - has_title, argv);', 1)[1]
    compute = _prefix(compute, '        scene.publish(argv[2]);')
    compute = '\n'.join(line[4:] if line.startswith('    ') else line for line in compute.splitlines())
    runtime = (_CODE / 'native_live_plot_runtime.cpp.in').read_text()
    main = 'extern "C" int scar_live_plot_v1'
    upper = _prefix(runtime, main)
    function = '''
namespace {
std::vector<char> compose_scene(std::span<const char> input, const scar_array_v1& x, const scar_array_v1& y,
                                const scar_array_v1& z, const std::vector<scar::live::Glyph>& glyphs) {
    Tape tape(input);
    std::array<char, 8> magic{};
    tape.read(magic.data(), magic.size());
    const bool has_title = std::string_view(magic.data(), magic.size()) == "SCARCTT3";
    require(has_title, "expected SCARCTT3 live plot layout");
    auto grid = live_grid(tape, x, y, z);''' + compute + '''
    return std::move(scene).finish();
}
}

'''
    return (helpers + '}\n' + upper + function + runtime[len(upper):]).encode()


def _agg(source):
    original = source.decode()
    source = _prefix(original, 'int main(')
    source = '#include "live_plot_internal.hpp"\n#include <cstring>\n' + source
    source = _replace(source, '    std::ifstream file_;', '    std::span<const char> input_;')
    source = _replace(source, '    explicit Tape(const char* name) : file_(name, std::ios::binary) {}',
                      '    explicit Tape(std::span<const char> values) : input_(values) {}')
    source = _replace(source, '''        consumed_ += length;
        if (length) require(bool(file_.read(static_cast<char*>(data), length)), "truncated AGG tape");''',
                      '''        require(length <= input_.size() - consumed_, "truncated AGG tape");
        if (length) std::memcpy(data, input_.data() + consumed_, length);
        consumed_ += length;''')
    source = _replace(source, 'file_.peek() == EOF && file_.eof()', 'consumed_ == input_.size()')
    body = original.split('        Tape tape(argv[1]);', 1)[1]
    body = _prefix(body, '        auto* output = std::fopen')
    body = '\n'.join(line[4:] if line.startswith('    ') else line for line in body.splitlines())
    return (source + '''
void scar::live::render_scene(std::span<const char> input, const char* filename) {
    Tape tape(input);''' + body + '''
    save_png(filename, renderer.pixBuffer, width, height);
}
''').encode()


def _png(source, version):
    generated = _prefix(source.decode(), 'int main(')
    return (('#include "live_plot_internal.hpp"\n' + generated + '''
void scar::live::save_png(const char* filename, const void* rgba, std::uint32_t columns, std::uint32_t rows) {
    if (columns != width || rows != height || std::string_view(zlibVersion()) != "@ZLIB_VERSION@")
        throw std::runtime_error("native PNG geometry or zlib provider differs");
    save(filename, rgba);
}
''').replace('@ZLIB_VERSION@', version)).encode()


def _glyph(source):
    prelude = '#include "live_plot_internal.hpp"\n#include "live_plot_profile.hpp"\n#include <clocale>\n'
    body = _prefix(source.decode(), 'int main(')
    body = _replace(body, '''    explicit Font(const char* filename) {
        require(!FT_Init_FreeType(&library), "cannot initialize FreeType");
        require(!FT_New_Face(library, filename, 0, &face), "cannot open original font");
    }''', '''    explicit Font(std::span<const char> bytes) {
        require(!FT_Init_FreeType(&library), "cannot initialize FreeType");
        if (FT_New_Memory_Face(library, reinterpret_cast<const FT_Byte*>(bytes.data()), bytes.size(), 0, &face)) {
            FT_Done_FreeType(library);
            throw std::runtime_error("cannot open original font");
        }
    }''')
    return (prelude + body + (_CODE / 'native_live_plot_glyph.cpp.in').read_text()).encode()


def _profile(scenes, title_plan, layout, font):
    _require(len(scenes) >= 2 and layout.startswith(b'SCARCTT3'), 'one qualified live title layout required')
    first = scenes[0]
    key = {'anchor', 'canvas', 'dpi', 'font', 'load_flags', 'points', 'route', 'worker'}
    title = {name: first['projection']['title'][name] for name in key}
    roots = first['original_root_views']
    _require(all(all(row['projection']['title'][name] == title[name] for name in key) and
                 row['original_root_views'] == roots and row['template'] == first['template'] for row in scenes),
             'qualified title or root routing differs across live observations')
    versions = {tuple(row['native']['provider_versions']) for row in title_plan['samples']}
    _require(len(versions) == 1, 'qualified font library versions differ')
    ft, hb = next(iter(versions))
    major, minor, patch = map(int, ft.split('.'))
    prefix, suffix = title['route']['prefix'], title['route']['suffix']
    _require(title['route']['kind'] == 'SCIENTIFIC_BINARY64_DEFAULT_PRECISION' and
             title['route']['precision'] == 6 and all(' ' <= letter <= '~' for letter in prefix + suffix),
             'live title source format exceeds the admitted profile')
    dimensions = ', '.join(f'{{{roots[axis]["root_shape"][0]}, {roots[axis]["root_shape"][1]}}}'
                           for axis in ('x', 'y', 'z'))
    digest = lambda raw: ', '.join(str(byte) for byte in sha256(raw).digest())
    header = f'''#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
namespace scar::live {{
inline constexpr std::size_t layout_size = {len(layout)}, font_size = {len(font)};
inline constexpr std::array<unsigned char, 32> layout_hash{{{digest(layout)}}};
inline constexpr std::array<unsigned char, 32> font_hash{{{digest(font)}}};
inline constexpr std::array<std::array<std::int64_t, 2>, 3> root_shapes{{{{{dimensions}}}}};
inline constexpr char title_prefix[] = {json.dumps(prefix)}, title_suffix[] = {json.dumps(suffix)};
inline constexpr int canvas_height = {title['canvas'][1]}, title_load_flags = {title['load_flags']};
inline constexpr unsigned title_dpi = {title['dpi']};
inline constexpr double title_points = {float(title['points']).hex()};
inline constexpr double title_x = {float(title['anchor'][0]).hex()}, title_y = {float(title['anchor'][1]).hex()};
inline constexpr int ft_major = {major}, ft_minor = {minor}, ft_patch = {patch};
inline constexpr char hb_version[] = {json.dumps(hb)};
}}
'''
    return dict(canvas=title['canvas'], roots=roots, title_format=title['route']['kind'],
                freetype=ft, harfbuzz=hb), header.encode()


def _deps(root):
    font, text = (root / 'freetype-2.14.3/include', root / 'harfbuzz-14.1.0/src')
    _require((font / 'ft2build.h').is_file() and (text / 'hb.h').is_file(), 'missing pinned font headers')
    return {f'vendor/{name}/{path.relative_to(directory)}': _ref(path, 'scar.upstream-header')
            for name, directory in (('freetype', font), ('harfbuzz', text))
            for path in sorted(directory.rglob('*.h'))}


def _ldd(path):
    return _dependencies(subprocess.check_output(['ldd', str(path)], env={'LC_ALL': 'C'}, text=True, timeout=15))


def _providers(title):
    direct = {Path(row['uri']).name: row for row in title['native_provider_libraries']}
    _require(len(direct) == 2 and len({Path(row['uri']).parent for row in direct.values()}) == 1,
             'requires exact paired font providers')
    directory = Path(next(iter(direct.values()))['uri']).parent
    original = _ldd(title['worker']['uri'])
    result = {}
    for name, value in original.items():
        if name in _SYSTEM or name == 'libz.so.1':
            continue
        _require(name.startswith(_SHADOW) and value and Path(value).parent == directory and Path(value).name == name,
                 'unexpected dependency of original glyph provider: ' + name)
        result[f'lib/{name}'] = _ref(value, 'scar.native-provider-library')
        if name in direct:
            _require(result[f'lib/{name}'] == direct[name], 'font library identity differs')
    _require(all('lib/' + name in result for name in direct), 'qualified font provider was not resolved')
    return result


def _license_members(extras, deps, font):
    license_data = {}
    for name, archive, members in (
            ('FreeType', deps / 'freetype-2.14.3.tar.xz', ('docs/FTL.TXT',)),
            ('HarfBuzz', deps / 'harfbuzz-14.1.0.tar.xz', ('COPYING',))):
        base = archive.name.removesuffix('.tar.xz')
        with tarfile.open(archive) as source:
            for member in members:
                with source.extractfile(base + '/' + member) as stream:
                    license_data[f'licenses/{name}-{Path(member).name}'] = stream.read()
    extra_licenses = {'licenses/DejaVu-LICENSE.txt': Path(font['uri']).with_name('LICENSE_DEJAVU'),
                      'licenses/Brotli-LICENSE.txt': Path('/usr/share/doc/libbrotli1/copyright'),
                      'licenses/LibPNG-LICENSE.txt': Path('/usr/share/doc/libpng16-16t64/copyright')}
    for name, path in extra_licenses.items():
        reference = extras.setdefault(name, _ref(path, 'scar.upstream-license'))
        license_data[name] = read_artifact(reference, json_document=False)
    return license_data


def _prepare(layout_path, agg_path, png_path, deps):
    layout, agg, png = map(lambda path: Path(path).resolve(strict=True), (layout_path, agg_path, png_path))
    live = native_contour_agg_live.check(layout)
    raster = native_agg_scene.check(agg)
    encoder = native_rgba_png.check(png)
    title = read_artifact(live['title_plan'])
    face = live['scenes'][0]['projection']['title']['font']
    font = read_artifact(face, json_document=False)
    wire = _member(layout, live, live['scenes'][0]['template'])
    profile, header = _profile(live['scenes'], title, wire, font)
    _require(raster['matplotlib_version'] and profile['canvas'] == encoder['geometry'], 'PNG surface differs from title layout')
    files = {}
    for base, report, skip in (
            (layout, live, {'src/main.cpp', 'src/contour_agg_io.hpp', live['scenes'][0]['template']}),
            (agg, raster, {'src/main.cpp'}),
            (png, encoder, {'src/main.cpp'})):
        for name in report['files']:
            if name not in skip and name.startswith(('src/', 'vendor/', 'include/', 'licenses/', 'lib/')):
                payload = _member(base, report, name)
                _require(files.setdefault(name, payload) == payload, 'incompatible compiled provider member: ' + name)
    files.update({LAYOUT: wire, FONT: font, 'include/scar_generated/live_plot_profile.hpp': header,
                  'src/contour_agg_io.hpp': _memory_tape(_member(layout, live, 'src/contour_agg_io.hpp')),
                  'src/live.cpp': _compose(_member(layout, live, 'src/main.cpp')),
                  'src/agg.cpp': _agg(_member(agg, raster, 'src/main.cpp')),
                  'src/png.cpp': _png(_member(png, encoder, 'src/main.cpp'), encoder['zlib_version']),
                  'src/glyph.cpp': _glyph(read_artifact(title['worker_source'], json_document=False))})
    for name, original in {
        'include/scar_generated/native_live_plot_abi_v1.h': 'native_live_plot_abi_v1.h',
        'include/scar_generated/scar_abi_v1.h': 'scar_abi_v1.h',
        'include/scar_generated/live_plot_internal.hpp': 'native_live_plot_internal.hpp.in',
        'src/demo.cpp': 'native_live_plot_demo.cpp.in',
    }.items():
        files[name] = _read(_CODE / original)
    extra = {**_deps(deps), **_providers(title)}
    for name, ref in extra.items():
        files[name] = read_artifact(ref, json_document=False)
    files.update(_license_members(extra, deps, face))
    direct = [Path(row['uri']).name for row in title['native_provider_libraries']]
    expected = {name.removeprefix('lib/') for name in files if name.startswith('lib/')}
    references = dict(layout=_ref(layout / 'result.json', native_contour_agg_live.SCHEMA),
                      raster=_ref(agg / 'result.json', native_agg_scene.SCHEMA),
                      encoder=_ref(png / 'result.json', native_rgba_png.SCHEMA), font=face,
                      support_inputs=extra, font_build_inputs=str(deps))
    return files, direct, expected, profile, references


def _project(root, files, direct):
    sources = tuple(sorted(name for name in files if name.endswith(('.cpp', '.c')) and name != 'src/demo.cpp'))
    includes = ('include/scar_generated', 'src', 'vendor', 'vendor/mpl', 'vendor/agg/include', 'vendor/pillow',
                'vendor/freetype', 'vendor/harfbuzz')
    libraries = tuple(f'"${{CMAKE_CURRENT_SOURCE_DIR}}/lib/{name}"' for name in ('libz.so.1', *direct))
    lines = ['cmake_minimum_required(VERSION 3.20)', 'project(scar_live_plot LANGUAGES C CXX)',
             'set(CMAKE_CXX_STANDARD 20)', 'set(CMAKE_CXX_STANDARD_REQUIRED ON)', 'set(CMAKE_CXX_EXTENSIONS OFF)',
             'set(CMAKE_C_STANDARD 11)', 'set(CMAKE_C_STANDARD_REQUIRED ON)',
             'add_library(scar_live_plot SHARED ' + ' '.join(map(_cmake_arg, sources)) + ')',
             'target_include_directories(scar_live_plot PUBLIC include/scar_generated PRIVATE ' +
             ' '.join(map(_cmake_arg, includes[1:])) + ')',
             'target_compile_options(scar_live_plot PRIVATE -fno-fast-math -ffp-contract=off -fvisibility=hidden)',
             'target_compile_definitions(scar_live_plot PRIVATE HAVE_LIBZ)',
             'target_link_options(scar_live_plot PRIVATE -Wl,--no-undefined)',
             'target_link_libraries(scar_live_plot PRIVATE crypto ' + ' '.join(libraries) + ')',
             "set_target_properties(scar_live_plot PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH " +
             _cmake_arg('$ORIGIN') + ')',
             'add_executable(scar_live_plot_demo src/demo.cpp)',
             'target_link_libraries(scar_live_plot_demo PRIVATE scar_live_plot)',
             'target_link_options(scar_live_plot_demo PRIVATE "-Wl,-rpath-link,${CMAKE_CURRENT_SOURCE_DIR}/lib")',
             'set_target_properties(scar_live_plot_demo PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH ' +
             _cmake_arg('$ORIGIN/../lib') + ')',
             'install(TARGETS scar_live_plot scar_live_plot_demo LIBRARY DESTINATION lib RUNTIME DESTINATION bin)',
             'install(DIRECTORY assets include licenses DESTINATION .)',
             'install(DIRECTORY lib DESTINATION . FILES_MATCHING PATTERN "*.so*")']
    (root / 'CMakeLists.txt').write_text('\n'.join(lines) + '\n')


def _build(root):
    env = {key: value for key, value in os.environ.items() if not key.startswith('LD_')}
    env['CXX'] = shutil.which('g++') or 'g++'
    with TemporaryDirectory(prefix='.scar-live-plot-build-', dir=root.parent) as temporary:
        path = Path(temporary)
        binary, prefix = path / 'build', path / 'installed'
        logs = []
        for command in (
                ['cmake', '-S', str(root), '-B', str(binary), '-DCMAKE_BUILD_TYPE=Release'],
                ['cmake', '--build', str(binary), '--parallel', '2'],
                ['cmake', '--install', str(binary), '--prefix', str(prefix)]):
            result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=100)
            logs.append(result.stdout + result.stderr)
            _require(result.returncode == 0, 'C++20 build failed\n' + '\n'.join(logs[-1].splitlines()[-40:]))
        for name in (LIBRARY, DEMO):
            member = root / name
            member.parent.mkdir(parents=True, exist_ok=True)
            (prefix / name).rename(member)
    logs_dir = root / 'validation'
    logs_dir.mkdir()
    (logs_dir / 'build.log').write_text('\n'.join(logs))


def _linkage(root, expected):
    library = cpu_executable_dynamic(root / LIBRARY)
    executable = cpu_executable_dynamic(root / DEMO)
    _require(library['runpath'] == '$ORIGIN' and executable['runpath'] == '$ORIGIN/../lib' and
             library['rpath'] is None and executable['rpath'] is None, 'output is not a relocatable native bundle')
    locations = _ldd(root / DEMO)
    _require(expected | {'libscar_live_plot.so'} <= locations.keys(), 'native dependency is not resolved')
    for name, location in locations.items():
        if name in expected or name == 'libscar_live_plot.so':
            _require(location == str((root / 'lib' / name).resolve(strict=True)), 'dependency escaped native package: ' + name)
        else:
            _require(name in _SYSTEM and (location is None or str(Path(location)).startswith('/usr/lib/')),
                     'unexpected dependency of native plot bundle: ' + name)
    return dict(library=library, demo=executable, system=sorted(set(locations) - expected - {'libscar_live_plot.so'}),
                private=sorted(expected))


def _cases(config):
    _require(type(config) is list and len(config) >= 2, 'need exact original witnesses for the demo')
    return [dict(time=float(row['time']).hex(), expected=_ref(row['expected'], 'image/png'),
                 roots=[dict(input=_ref(root['path'], 'scar.live-root-file'), offset=int(root['offset']))
                        for root in row['roots']]) for row in config]


def _measure(root, cases):
    with TemporaryDirectory(prefix='scar-live-plot-trial-') as temporary:
        for i, row in enumerate(cases):
            _require(len(row['roots']) == 3, 'exact x, y, z roots are required')
            arguments = []
            for descriptor in row['roots']:
                read_artifact(descriptor['input'], json_document=False)
                arguments += [descriptor['input']['uri'], str(descriptor['offset'])]
            target = Path(temporary) / f'frame-{i}.png'
            subprocess.run([root / DEMO, root / LAYOUT, root / FONT, target, row['time'], *arguments],
                           env={'LC_ALL': 'C'}, check=True, timeout=25)
            _require(target.read_bytes() == read_artifact(row['expected'], json_document=False),
                     f'native PNG differs from original output for case {i}')


def export(layout, agg, png, font_build_inputs, out, *, demo):
    package = Path(out).absolute()
    _require(not package.exists() and not package.is_symlink(), 'fresh package output required')
    cases = _cases(json.loads(_read(demo)))
    files, directs, expected, profile, sources = _prepare(layout, agg, png, Path(font_build_inputs).resolve(strict=True))
    with TemporaryDirectory(prefix='.scar-live-plot-', dir=package.parent) as temporary:
        root = Path(temporary) / 'package'
        for name, payload in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        _project(root, files, directs)
        _build(root)
        linkage = _linkage(root, expected)
        _measure(root, cases)
        output = dict(schema=SCHEMA, schema_version=1, sources=sources, profile=profile, validation=cases,
            files={name: {**_ref(root / name, 'scar.native-live-plot-member'), 'uri': name}
                   for name in (*files, LIBRARY, DEMO, 'CMakeLists.txt')},
            linkage=linkage, cxx_standard=20, symbol='scar_live_plot_v1', library=LIBRARY, layout=LAYOUT, font=FONT,
            in_process=True, python_runtime_required=False, exact_original_png_bytes=True,
            original_qualified_layout_required=True, whole_entry_authorized=False, deployment_authorized=False)
        (root / 'result.json').write_text(json.dumps(output, sort_keys=True, indent=2) + '\n')
        root.rename(package)
    return output


def check(package):
    root = Path(package).resolve(strict=True)
    report = read_artifact(_ref(root / 'result.json', SCHEMA))
    _require(report['schema_version'] == 1 and report['symbol'] == 'scar_live_plot_v1' and
             (report['library'], report['layout'], report['font']) == (LIBRARY, LAYOUT, FONT) and
             report['in_process'] is True and report['python_runtime_required'] is False and
             report['whole_entry_authorized'] is False and report['deployment_authorized'] is False,
             'native product scope differs')
    sources = report['sources']
    files, directs, expected, profile, replay_sources = _prepare(
        *(Path(sources[name]['uri']).parent for name in ('layout', 'raster', 'encoder')),
        Path(sources['font_build_inputs']))
    _require(sources == replay_sources and report['profile'] == profile and
             report['linkage'] == _linkage(root, expected) and
             set(report['files']) == {*files, LIBRARY, DEMO, 'CMakeLists.txt'},
             'composed native package differs')
    for name in report['files']:
        payload = _member(root, report, name)
        if name in files:
            _require(payload == files[name], 'native input differs: ' + name)
    with TemporaryDirectory(prefix='scar-live-plot-project-') as temporary:
        draft = Path(temporary)
        _project(draft, files, directs)
        _require(_read(root / 'CMakeLists.txt') == _read(draft / 'CMakeLists.txt'), 'native build metadata differs')
    _measure(root, report['validation'])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('layout', 'agg', 'png', 'font-build-inputs', 'out', 'demo', 'check'):
        parser.add_argument('--' + name)
    args = parser.parse_args(argv)
    if args.check:
        result = check(args.check)
    elif all((args.layout, args.agg, args.png, args.font_build_inputs, args.out, args.demo)):
        result = export(args.layout, args.agg, args.png, args.font_build_inputs, args.out, demo=args.demo)
    else:
        parser.error('provide --check PACKAGE or all the source packages and original demo witnesses')
    print(json.dumps(dict(validation=len(result['validation']), in_process=result['in_process'],
                          exact_original_png_bytes=result['exact_original_png_bytes'], library=result['library'])))


if __name__ == '__main__':
    main()
