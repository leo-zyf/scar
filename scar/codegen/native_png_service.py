"""Calibrate the unchanged synchronous PNG service of a released native plot provider."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen import native_live_plot as plot
from scar.codegen.native_cpu_elf import cpu_executable_dynamic
from scar.codegen.native_project import build_native_project, write_native_project
from scar.ir.execution_recipe import read_artifact

SCHEMA = 'scar.native-png-service'
TARGET = 'scar_png_service'
EXECUTABLE = 'bin/' + TARGET
_SOURCE = 'src/main.cpp'
_SYMBOL = '_ZN4scar4live8save_pngEPKcPKvjj'
_MANGLE = 'void(const char*,const void*,uint32_t,uint32_t)'
_SECTION = re.compile(r'^\s*\[\s*(\d+)\]\s+(\S+)\s+(\S+)\s+([a-f0-9]+)\s+'
                      r'([a-f0-9]+)\s+([a-f0-9]+)\s+[a-f0-9]+\s+(\S+)\s', re.I)


def _require(ok, detail):
    if not ok:
        raise ValueError('native PNG service: ' + detail)


def _readelf(path, option):
    return subprocess.check_output(['/usr/bin/readelf', '-W', option, str(path)], env={'LC_ALL': 'C'},
                                   encoding='ascii', timeout=10)


def _entry(path, data):
    header = _readelf(path, '-h')
    _require(all(re.search(pattern, header, re.M) for pattern in (
        r'^\s*Class:\s+ELF64\s*$', r'^\s*Data:.*little endian\s*$',
        r'^\s*Type:\s+DYN\b', r'^\s*Machine:\s+Advanced Micro Devices X86-64\s*$')),
        'expected a released x86-64 ELF shared library')
    entries = [columns for line in _readelf(path, '-s').splitlines() if
               len(columns := line.split()) == 8 and columns[0].endswith(':') and columns[-1] == _SYMBOL]
    _require(len(entries) == 1, 'private service does not have one exact symbol')
    _, value, size, kind, binding, visibility, index, _ = entries[0]
    _require((kind, binding, visibility) == ('FUNC', 'LOCAL', 'DEFAULT') and index.isdecimal(),
             'private service is not a normal defined function')
    offset, length = int(value, 16), int(size)
    sections = [row.groups() for line in _readelf(path, '-S').splitlines()
                if (row := _SECTION.match(line)) and int(row[1]) == int(index)]
    _require(len(sections) == 1, 'private service executable section missing')
    _, section, format_, start, start_file, extent, flags = sections[0]
    start, start_file, extent = [int(value, 16) for value in (start, start_file, extent)]
    _require(section == '.text' and format_ == 'PROGBITS' and flags == 'AX' and
             0 < length <= extent and start <= offset <= start + extent - length,
             'private service exceeds its allocated executable section')
    file_offset = start_file + offset - start
    _require(file_offset <= len(data) - length, 'private service exceeds released ELF bytes')
    loads = []
    for line in _readelf(path, '-l').splitlines():
        columns = line.split()
        if columns[:1] == ['LOAD'] and ''.join(columns[6:-1]) == 'RE':
            segment_file, segment_start, _, segment_size = (int(value, 16) for value in columns[1:5])
            if (segment_start <= offset and length <= segment_size - (offset - segment_start)
                    and segment_file + offset - segment_start == file_offset):
                loads.append((segment_start, segment_file, segment_size))
    _require(len(loads) == 1, 'private service does not match an executable file-backed load segment')
    load_start, load_file, load_bytes = loads[0]
    return dict(mangled_name=_SYMBOL, abi=_MANGLE, type=kind, binding=binding, section=section,
                value=offset, length=length, file_offset=file_offset,
                load_segment=dict(value=load_start, file_offset=load_file, bytes=load_bytes),
                instruction_bytes_sha256=sha256(data[file_offset:file_offset+length]).hexdigest())


def _digest_text(digest):
    return ', '.join(str(byte) for byte in bytes.fromhex(digest))


def _source(pins, hosts, entry, surface):
    literal = lambda value: json.dumps(str(value), ensure_ascii=True)
    encoded = []
    for name, payload in pins.items():
        encoded.append('{' + literal(name) + f', {len(payload)}, {{{_digest_text(sha256(payload).hexdigest())}}}' + '}')
    values = dict(WIDTH=surface[0], HEIGHT=surface[1], SYMBOL_OFFSET=entry['value'], SYMBOL_BYTES=entry['length'],
                  SYMBOL_DIGEST=_digest_text(entry['instruction_bytes_sha256']), PROVIDER=literal(plot.LIBRARY),
                  FILE_COUNT=len(pins), FILES=',\n    '.join(encoded),
                  HOST_COUNT=len(hosts), HOSTS=',\n    '.join(literal(path) for path in hosts))
    text = Path(__file__).with_name('native_png_service.cpp.in').read_text()
    return re.sub(r'@([A-Z_]+)@', lambda match: str(values[match[1]]), text).encode()


def _prepare(reference):
    root = Path(reference['uri']).parent
    release = plot.check(root)
    _require(release == read_artifact(reference) and release['schema'] == plot.SCHEMA and
             release['symbol'] == 'scar_live_plot_v1' and release['python_runtime_required'] is False,
             'expected the unchanged released v1 native plot provider')
    private = frozenset(release['linkage']['private'])
    required = {plot.LIBRARY, *(f'lib/{name}' for name in private)}
    _require({'lib/libz.so.1', plot.LIBRARY} <= required, 'released PNG dependency closure differs')
    libraries = {name: plot._member(root, release, name) for name in sorted(required)}
    dependencies = plot._ldd(root / plot.DEMO)
    hosts = []
    for name in release['linkage']['system']:
        path = dependencies[name]
        if name == 'linux-vdso.so.1':
            _require(path is None, 'expected the host virtual system DSO')
        else:
            _require(Path(path).is_absolute() and path not in hosts, 'cannot bind exact host ELF paths')
            hosts.append(path)
    hosts = sorted(hosts)
    _require(any(Path(path).name == 'libcrypto.so.3' for path in hosts), 'release host SHA provider missing')
    entry = _entry(root / plot.LIBRARY, libraries[plot.LIBRARY])
    encoder = read_artifact(release['sources']['encoder'])
    surface = encoder['geometry']
    _require(surface == release['profile']['canvas'] and len(surface) == 2 and
             all(type(item) is int and 0 < item <= 16384 for item in surface) and
             4 * surface[0] * surface[1] <= 128_000_000, 'released RGBA and PNG surface differs')
    witnesses = dict(rgba=encoder['inputs']['rgba'], png=encoder['inputs']['reference'])
    _require(witnesses['rgba']['size_bytes'] == 4 * surface[0] * surface[1],
             'released native PNG validation has another RGBA geometry')
    files = dict(libraries)
    files.update({name: plot._member(root, release, name) for name in release['files'] if name.startswith('licenses/')})
    files[_SOURCE] = _source(libraries, hosts, entry, surface)
    inputs = dict(entry=entry, surface=surface, host_paths=hosts, private_libraries=sorted(private),
                  original_oracle=witnesses)
    return files, inputs


def _project(root):
    write_native_project(root, TARGET, libraries=('crypto', 'dl'),
                         compile_options=('-fno-fast-math', '-ffp-contract=off'),
                         install_dirs=(('lib', '.'), ('licenses', '.')))


def _test(executable, witnesses):
    read_artifact(witnesses['rgba'], json_document=False)
    with TemporaryDirectory(prefix='scar-native-png-oracle-') as temporary:
        output = Path(temporary) / 'proof.png'
        run = subprocess.run([executable, witnesses['rgba']['uri'], output], env={'LC_ALL': 'C'},
                             capture_output=True, check=True, timeout=20)
        _require(not run.stderr and re.fullmatch(rb'[1-9][0-9]*\n', run.stdout) and
                 output.read_bytes() == read_artifact(witnesses['png'], json_document=False),
                 'clean driver output or its wall-clock protocol differs')


def _report(root, reference, files, inputs):
    names = (*files, EXECUTABLE, 'CMakeLists.txt')
    return dict(schema=SCHEMA, schema_version=1, source_provider=reference, **inputs,
        executable=EXECUTABLE, library=plot.LIBRARY, files={
            name: {**plot._ref(root / name, 'scar.native-png-service-member'), 'uri': name} for name in names},
        driver_dependencies=cpu_executable_dynamic(root / EXECUTABLE),
        runtime='LINUX_GLIBC_X86_64', cxx_standard=20, python_runtime_required=False,
        scope='RELEASED_SYNCHRONOUS_SAVE_PNG_SERVICE', clean_no_internal_instrumentation=True,
        measurement='STEADY_CLOCK_NS_INSIDE_PROCESS_AFTER_RGBA_PREFAULT', exact_reference_png=True,
        arguments=['RGBA_FILE', 'FRESH_OUTPUT_PNG'], stdout='POSITIVE_DECIMAL_NANOSECONDS_AND_LF',
        memory_workspace_measured=False, deployment_authorized=False)


def build(provider, out):
    release = Path(provider).resolve(strict=True)
    reference = plot._ref(release / 'result.json', plot.SCHEMA)
    files, inputs = _prepare(reference)
    package = Path(out).absolute()
    _require(not package.exists() and not package.is_symlink(), 'require a fresh calibration package path')
    compiler = shutil.which('g++')
    _require(compiler is not None, 'GNU C++20 compiler unavailable')
    with TemporaryDirectory(prefix='.scar-native-png-service-', dir=package.parent) as temporary:
        root = Path(temporary) / 'package'
        for name, data in files.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        _project(root)
        executable, _ = build_native_project(root, TARGET, (compiler,))
        _test(executable, inputs['original_oracle'])
        result = _report(root, reference, files, inputs)
        (root / 'result.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
        root.rename(package)
    return result


def check(package):
    root = Path(package).resolve(strict=True)
    result = read_artifact(plot._ref(root / 'result.json', SCHEMA))
    reference = result['source_provider']
    files, inputs = _prepare(reference)
    _require(result == _report(root, reference, files, inputs), 'clean calibration package receipt differs')
    for name, data in files.items():
        _require(plot._member(root, result, name) == data, 'calibration package changed: ' + name)
    with TemporaryDirectory(prefix='scar-native-png-project-') as temporary:
        draft = Path(temporary)
        _project(draft)
        _require((root / 'CMakeLists.txt').read_bytes() == (draft / 'CMakeLists.txt').read_bytes(),
                 'calibration CMake recipe differs')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check')
    parser.add_argument('--provider')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check and not any((args.provider, args.out)):
        result = check(args.check)
    elif not args.check and args.provider and args.out:
        result = build(args.provider, args.out)
    else:
        parser.error('provide --check PACKAGE or --provider RELEASE_PACKAGE --out PACKAGE')
    print(json.dumps(dict(schema=result['schema'], executable=result['executable'], scope=result['scope'],
                          exact_reference_png=result['exact_reference_png'])))


if __name__ == '__main__':
    main()
