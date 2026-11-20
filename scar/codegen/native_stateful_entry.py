"""Deliver a replayed, provider-free stateful region as a C++20 file entry."""

import argparse
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

from scar.analysis.replay_transaction import replay_transaction
from scar.codegen import cpp, stateful
from scar.codegen.native_cpu_elf import cpu_executable_dynamic
from scar.codegen.native_project import build_native_project, write_native_project
from scar.codegen.stateful_workspace import _base_metadata
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.native-stateful-entry'
_TARGET = 'scar_stateful_entry'
_EXE = f'bin/{_TARGET}'
_TEMPLATE = Path(__file__).with_name('native_stateful_entry_v1.cpp.in')
_HEADERS = ('scar_abi_v1.h', 'stateful_abi_v1.h')
_ORIGINAL = {'region', 'plan', 'source', 'library', *_HEADERS}
_HOST_LIBS = {'libc.so.6', 'libm.so.6', 'libstdc++.so.6', 'libgcc_s.so.1', 'libgomp.so.1',
              'libpthread.so.0', 'ld-linux-x86-64.so.2', 'ld-linux-aarch64.so.1', 'linux-vdso.so.1'}
_MEMBERS = {
    'CMakeLists.txt': 'scar.cmake-project',
    'src/main.cpp': 'scar.cpp-source',
    'src/stateful.cpp': 'scar.cpp-source',
    **{f'include/scar_generated/{name}': 'scar.cpp-header' for name in _HEADERS},
    _EXE: 'scar.native-executable',
}


def _need(condition, detail):
    if not condition:
        raise ValueError('native stateful entry: ' + detail)


def _local(root, name, schema):
    path = root / name
    _need(path.resolve(strict=True) == path and path.is_file(), 'expected an exact local member: ' + name)
    return artifact_reference(path, schema)


def _admit(reference):
    original = read_artifact(reference)
    root = Path(reference['uri']).parent
    _need((original.get('schema_version'), original.get('stateful_abi_version')) == (1, 1)
          and set(original['files']) == _ORIGINAL and set(original['sha256']) == _ORIGINAL
          and len(set(original['files'].values())) == len(_ORIGINAL)
          and all(Path(name).name == name for name in original['files'].values()),
          'original stateful ABI 1 artifact required')
    for key, name in original['files'].items():
        _need(_local(root, name, 'scar.stateful-entry-input')['sha256'] == original['sha256'][key],
              'original artifact member differs: ' + key)
    region_path = root / original['files']['region']
    region = stateful.read_stateful(region_path)
    _need(not region.providers, 'NATIVE_PROVIDER_FRONTIER: region requires a provider callback')
    region, plan, placement = stateful._admit(region_path, root / original['files']['plan'])
    verified = _base_metadata(reference, region, plan, placement)
    metadata = verified[0]
    _need(metadata == original, 'original metadata changed during replay')
    ports = []
    for port in region.inputs:
        _need(re.fullmatch(r'[A-Za-z0-9_:-]+', port.id) and len(port.id) <= 128,
              'unsupported bounded command port ID')
        row = dict(id=port.id, source_path=list(port.path), kind=port.kind)
        if port.kind == 'array':
            shape = plan['input_shapes'][port.id]
            _need(shape[0] * shape[1] * 8 <= (1 << 63) - 1 and shape[1] * 8 <= (1 << 63) - 1,
                  'array size or stride outside the command ABI')
            row.update(shape=shape, output=port.id in plan['shadows'])
        ports.append(row)
    _need(0 < len(ports) <= 256 and any(row.get('output') for row in ports), 'bounded writable ports required')
    return metadata, plan, ports, root


def _render(ports, metadata):
    arrays = [row for row in ports if row['kind'] == 'array']
    scalars = [row for row in ports if row['kind'] == 'scalar']
    declarations = {
        'ARRAYS': ',\n    '.join(
            f'{{"{row["id"]}", {row["shape"][0]}, {row["shape"][1]}, {str(row["output"]).lower()}}}'
            for row in arrays),
        'SCALARS': ', '.join(f'"{row["id"]}"' for row in scalars),
        'ARRAY_COUNT': str(len(arrays)),
        'SCALAR_COUNT': str(len(scalars)),
        'OUTPUT_COUNT': str(sum(row['output'] for row in arrays)),
        **{key.upper(): metadata[key] for key in ('region_sha256', 'plan_sha256', 'contract_sha256')},
    }
    return re.sub(r'@([A-Z0-9_]+)@', lambda match: declarations[match[1]], _TEMPLATE.read_text()).encode()


def _project(root, plan):
    options = tuple(option for option in cpp.COMPILER_OPTIONS if option not in ('-std=c++20', '-shared', '-fPIC'))
    parallel = ('-fopenmp',) if plan['cpu_threads'] > 1 else ()
    write_native_project(root, _TARGET, sources=('src/main.cpp', 'src/stateful.cpp'),
                         compile_options=(*options, *parallel), link_options=parallel, libraries=('m',))


def _linkage(executable):
    dynamic = cpu_executable_dynamic(executable)
    _need(dynamic['runpath'] is None and dynamic['rpath'] is None
          and 'libc.so.6' in dynamic['needed'] and set(dynamic['needed']) <= _HOST_LIBS,
          'standalone ELF must use only host runtime dependencies')
    return dynamic


def _members(root):
    return {name: {**_local(root, name, schema), 'uri': name} for name, schema in _MEMBERS.items()}


def _report(reference, metadata, plan, ports, root):
    return dict(schema=SCHEMA, schema_version=1, cxx_standard=20, stateful_abi_version=1, original=reference,
                original_build_id=metadata['build_id'],
                identity={key: metadata[key] for key in ('region_sha256', 'plan_sha256', 'contract_sha256')},
                ports=ports, files=_members(root), linkage=_linkage(root / _EXE),
                cpu_ids=metadata['cpu_ids'], array_payload_bytes=plan['total_region_array_payload_bytes'],
                array_encoding='C_ORDER_LITTLE_ENDIAN_BINARY64', status='REGION_ENTRY_EXPERIMENT',
                effects='EXPLICIT_WRITTEN_ROOT_FILES_AFTER_NATIVE_SUCCESS',
                native_provider_count=0, python_runtime_required=False,
                whole_program_translated=False, deployment_authorized=False)


@replay_transaction()
def export(native, out):
    """Publish a CMake project and executable using the original replayed ABI code."""
    path = Path(native).resolve(strict=True)
    reference = artifact_reference(path / 'artifact.json' if path.is_dir() else path, stateful.SCHEMA)
    metadata, plan, ports, original = _admit(reference)
    out = Path(out).absolute()
    _need(out.parent.resolve(strict=True) == out.parent and not out.exists() and not out.is_symlink(),
          'fresh output directory in an existing canonical parent required')
    compiler = shutil.which('g++')
    _need(compiler is not None, 'GNU C++20 compiler unavailable')
    with TemporaryDirectory(prefix='.scar-stateful-entry-', dir=out.parent) as temporary:
        root = Path(temporary) / 'entry'
        (root / 'src').mkdir(parents=True)
        include = root / 'include/scar_generated'
        include.mkdir(parents=True)
        (root / 'src/main.cpp').write_bytes(_render(ports, metadata))
        shutil.copyfile(original / metadata['files']['source'], root / 'src/stateful.cpp')
        for name in _HEADERS:
            shutil.copyfile(original / metadata['files'][name], include / name)
        _project(root, plan)
        _, dependencies = build_native_project(root, _TARGET, (compiler,))
        _need({Path(name).name for name in dependencies} <= _HOST_LIBS,
              'native executable has a non-host dependency')
        report = _report(reference, metadata, plan, ports, root)
        (root / 'result.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
        check(root)
        root.rename(out)
    return report


@replay_transaction()
def check(package):
    """Replay source, plan, port table and package receipts without running the target."""
    root = Path(package).resolve(strict=True)
    original = read_artifact(_local(root, 'result.json', SCHEMA))
    reference = original['original']
    metadata, plan, ports, native = _admit(reference)
    _need(original == _report(reference, metadata, plan, ports, root),
          'package contract, artifact bytes or ELF linkage differs')
    _need((root / 'src/main.cpp').read_bytes() == _render(ports, metadata),
          'native command ports differ from replay')
    for key, name in (('source', 'src/stateful.cpp'),
                      *((name, f'include/scar_generated/{name}') for name in _HEADERS)):
        _need((root / name).read_bytes() == (native / metadata['files'][key]).read_bytes(),
              'original C++ or ABI header differs: ' + name)
    with TemporaryDirectory(prefix='scar-stateful-cmake-') as temporary:
        _project(Path(temporary), plan)
        _need((root / 'CMakeLists.txt').read_bytes() == (Path(temporary) / 'CMakeLists.txt').read_bytes(),
              'CMake project differs from replay')
    return original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check')
    parser.add_argument('--native')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.check and not (args.native or args.out):
        report = check(args.check)
    elif args.native and args.out and not args.check:
        report = export(args.native, args.out)
    else:
        parser.error('provide --native ARTIFACT --out PACKAGE, or --check PACKAGE')
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
