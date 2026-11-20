"""Export the retained Pillow RGBA PNG codec with a native C++20 file interface."""

import argparse
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
from tempfile import TemporaryDirectory
import zlib

from scar.codegen.native_cpu_elf import cpu_executable_dynamic
from scar.codegen.native_project import build_native_project, write_native_project
from scar.ir.execution_recipe import artifact_reference, read_artifact

SCHEMA = 'scar.native-rgba-png'
_TARGET = 'scar_rgba_png'
_EXE = 'bin/' + _TARGET
_CODEC = ('ZipEncode.c', 'ZipCodecs.h', 'Imaging.h', 'ImPlatform.h', 'ImagingUtils.h', 'Mode.h', 'Arrow.h')
_ZLIB = 'lib/libz.so.1'
_SHIM = '''/* Portable types used by the upstream encoder headers. */
#include <limits.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#define HAVE_PROTOTYPES 1
#define STDC_HEADERS 1
typedef struct _object PyObject;
typedef ssize_t Py_ssize_t;
'''


def _require(ok, reason):
    if not ok:
        raise ValueError('native RGBA PNG: ' + reason)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _profile(raw):
    signature = b'\x89PNG\r\n\x1a\n'
    _require(raw.startswith(signature), 'expected PNG signature')
    offset = len(signature)
    width = height = prefix = None
    complete = False
    while offset + 12 <= len(raw):
        size, kind = struct.unpack_from('>I4s', raw, offset)
        end = offset + 12 + size
        _require(end <= len(raw), 'truncated PNG chunk')
        data = raw[offset+8:end-4]
        _require(zlib.crc32(kind + data) == struct.unpack_from('>I', raw, end-4)[0], 'PNG chunk CRC differs')
        if offset == len(signature):
            _require(kind == b'IHDR' and size == 13 and data[8:] == b'\x08\x06\0\0\0',
                     'non-interlaced RGBA8 PNG required')
            width, height = struct.unpack_from('>II', data)
            _require(1 <= width <= 16384 and 1 <= height <= 16384 and 4*width*height <= 128_000_000,
                     'RGBA byte budget exceeded')
        elif kind == b'IDAT':
            if prefix is None:
                _require(offset <= 8192, 'header metadata exceeds profile')
                prefix = raw[:offset]
        elif kind == b'IEND':
            _require(size == 0 and end == len(raw) and prefix is not None, 'invalid completion chunk')
            complete = True
            break
        else:
            _require(prefix is None and kind in (b'tEXt', b'pHYs'), 'unsupported original PNG metadata')
        offset = end
    _require(complete, 'missing complete PNG data')
    return width, height, prefix


def _prepare(inputs, version):
    _require(re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', version) is not None, 'expected observed zlib version')
    reference = read_artifact(inputs['reference'], json_document=False)
    width, height, prefix = _profile(reference)
    _require(inputs['rgba']['size_bytes'] == 4*width*height, 'RGBA input geometry differs')
    read_artifact(inputs['rgba'], json_document=False)
    directory = Path(__file__).parent
    main = (directory / 'native_rgba_png_v1.cpp.in').read_text()
    for key, value in dict(WIDTH=width, HEIGHT=height, PREFIX=', '.join(map(str, prefix)), ZLIB_VERSION=version).items():
        main = main.replace(f'@{key}@', str(value))
    members = {
        'src/main.cpp': ('scar.cpp-source', main.encode()),
        'src/image_adapter.c': ('scar.c-source', (directory / 'native_rgba_png_adapter_v1.c.in').read_bytes()),
        'include/scar_generated/Python.h': ('scar.c-header', _SHIM.encode()),
    }
    for name in (*_CODEC, 'pillow_license', 'zlib_license', 'zlib'):
        item = inputs[name]
        filename = {
            'pillow_license': 'licenses/Pillow-LICENSE.txt',
            'zlib_license': 'licenses/Zlib-LICENSE.txt',
            'zlib': _ZLIB,
        }.get(name, f'vendor/pillow/{name}')
        members[filename] = item['schema'], read_artifact(item, json_document=False)
    return (width, height), members, reference


def _project(root):
    write_native_project(root, _TARGET, sources=('src/main.cpp', 'src/image_adapter.c', 'vendor/pillow/ZipEncode.c'),
        include_dirs=('vendor/pillow',), compile_options=('-DHAVE_LIBZ',),
        runtime_dirs=('$ORIGIN/../lib',),
        install_dirs=(('lib', '.'), ('licenses', '.')), with_c_provider=True)
    with (root / 'CMakeLists.txt').open('a') as stream:
        stream.write(f'target_link_libraries({_TARGET} PRIVATE "${{CMAKE_CURRENT_SOURCE_DIR}}/{_ZLIB}")\n')


def _package_report(root, sources, version, geometry, members):
    dynamic = cpu_executable_dynamic(root / _EXE)
    dependencies = set(dynamic['needed'])
    _require(dynamic['runpath'] == '$ORIGIN/../lib' and dynamic['rpath'] is None
             and 'libz.so.1' in dependencies
             and dependencies <= {'libz.so.1', 'libc.so.6', 'libstdc++.so.6', 'libgcc_s.so.1', 'libm.so.6'},
             'native package linkage differs')
    file_roles = {**{name: pair[0] for name, pair in members.items()},
                  _EXE: 'scar.native-executable', 'CMakeLists.txt': 'scar.cmake-project'}
    files = {name: {**_ref(root / name, schema), 'uri': name} for name, schema in file_roles.items()}
    return dict(schema=SCHEMA, schema_version=1, cxx_standard=20, inputs=sources, geometry=list(geometry),
        files=files, zlib_version=version, linkage=dynamic, python_runtime_required=False,
        reference_byte_equal=True, encodes_runtime_pixels=True, raw_raster_generated_by_product=False,
        original_renderer_lowered=False, deployment_authorized=False)


def export(reference, rgba, pillow_source, zlib_library, zlib_license, version, out):
    pillow = Path(pillow_source).resolve(strict=True)
    inputs = dict(reference=_ref(reference, 'image/png'), rgba=_ref(rgba, 'scar.packed-rgba8'),
        zlib=_ref(zlib_library, 'scar.native-provider-library'),
        pillow_license=_ref(pillow / 'LICENSE', 'scar.upstream-license'),
        zlib_license=_ref(zlib_license, 'scar.upstream-license'))
    for name in _CODEC:
        inputs[name] = _ref(pillow / 'src/libImaging' / name, 'scar.upstream-source')
    geometry, members, expected = _prepare(inputs, version)
    out = Path(out).absolute()
    _require(not out.exists() and not out.is_symlink(), 'need a fresh package directory')
    gcc = shutil.which('g++')
    _require(gcc is not None, 'GNU C++20 compiler missing')
    with TemporaryDirectory(prefix='.scar-png-', dir=out.parent) as temp:
        root = Path(temp) / 'package'
        for name, (_, payload) in members.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        _project(root)
        exe, _ = build_native_project(root, _TARGET, (gcc,), expected_libraries=(root / _ZLIB,))
        target = Path(temp) / 'oracle.png'
        subprocess.run([str(exe), inputs['rgba']['uri'], str(target)],
                       env={}, check=True, timeout=20)
        _require(target.read_bytes() == expected, 'native output differs on original RGBA input')
        report = _package_report(root, inputs, version, geometry, members)
        (root / 'result.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        root.rename(out)
    return report


def check(package):
    root = Path(package).resolve(strict=True)
    saved = read_artifact(_ref(root / 'result.json', SCHEMA))
    geometry, members, _ = _prepare(saved['inputs'], saved['zlib_version'])
    _require(saved == _package_report(root, saved['inputs'], saved['zlib_version'], geometry, members),
             'native receipt bytes differ')
    for name, (_, payload) in members.items():
        _require((root / name).read_bytes() == payload, 'prepared source or provider differs: ' + name)
    with TemporaryDirectory(prefix='scar-png-layout-') as temp:
        _project(Path(temp))
        _require((Path(temp) / 'CMakeLists.txt').read_bytes() == (root / 'CMakeLists.txt').read_bytes(),
                 'CMake build recipe differs')
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check')
    for flag in ('reference', 'rgba', 'pillow-source', 'zlib', 'zlib-license', 'zlib-version', 'out'):
        parser.add_argument('--' + flag)
    args = parser.parse_args(argv)
    if args.check:
        result = check(args.check)
    elif all((args.reference, args.rgba, args.pillow_source, args.zlib, args.zlib_license, args.zlib_version, args.out)):
        result = export(args.reference, args.rgba, args.pillow_source, args.zlib, args.zlib_license, args.zlib_version,
                        args.out)
    else:
        parser.error('provide --check PACKAGE or reference, RGBA, provider sources and output')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
