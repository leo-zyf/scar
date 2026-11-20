"""Deliver a replayed stateful region and qualified native provider as C++20."""

import argparse
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

from scar.analysis.replay_transaction import replay_transaction
from scar.codegen import cpp, stateful, stateful_numpy_provider
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
              'libdl.so.2', 'libpthread.so.0', 'ld-linux-x86-64.so.2', 'ld-linux-aarch64.so.1', 'linux-vdso.so.1'}
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


def _admit(reference, *, observe_provider=False):
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
    provider = stateful_numpy_provider.prepare(region, original, observe=observe_provider) if region.providers else None
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
    return metadata, plan, ports, root, provider


def _render(ports, metadata, provider, update=None, controller=None):
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
    selected = provider is not None
    provider_args = '&provider, 1' if selected else 'nullptr, 0'
    setup = '    const auto provider = scar_numpy_power_provider();\n' if selected else ''
    declarations.update(
        PROVIDER_DECLARATIONS='scar_provider_v1 scar_numpy_power_provider();\n\n' if selected else '',
        PROVIDER_COUNT=str(int(selected)), UPDATE_DECLARATIONS='', UPDATE_OUTPUT_COUNT='', UPDATE_OUTPUT='',
        ENTRY_ARGUMENTS='    const auto arguments = bind(argc, argv);',
        RUN_BODY=setup + f'''    const auto status = table->run(descriptors.data(), descriptors.size(),
                                   arguments.values.data(), arguments.values.size(), {provider_args});
    if (status != 0) throw std::runtime_error("native stateful region declined: " + std::to_string(status));''',
    )
    if controller is not None:
        from scar.codegen.native_controller_entry import render
        declarations.update(render(controller, setup, selected))
    elif update is not None:
        from scar.codegen.native_update_entry import render
        declarations.update(render(update, setup, provider_args))
    return re.sub(r'@([A-Z0-9_]+)@', lambda match: declarations[match[1]], _TEMPLATE.read_text()).encode()


def _project(root, plan, provider, update, controller=None):
    options = tuple(option for option in cpp.COMPILER_OPTIONS if option not in ('-std=c++20', '-shared', '-fPIC'))
    parallel = ('-fopenmp',) if plan['cpu_threads'] > 1 else ()
    sources = ('src/main.cpp', 'src/stateful.cpp')
    if provider is not None:
        sources += (stateful_numpy_provider.SOURCE,)
    if update is not None:
        from scar.codegen.native_update_entry import COPY_SOURCE
        sources += (COPY_SOURCE,)
    libraries = ('m',)
    if controller is not None:
        from scar.codegen.native_controller_entry import SOURCE
        sources += (SOURCE,)
        libraries += ('dl',)
    write_native_project(root, _TARGET, sources=sources, compile_options=(*options, *parallel),
                         link_options=parallel, libraries=libraries,
                         install_dirs=(('licenses', '.'),) if provider is not None else ())


def _linkage(executable):
    dynamic = cpu_executable_dynamic(executable)
    _need(dynamic['runpath'] is None and dynamic['rpath'] is None
          and 'libc.so.6' in dynamic['needed'] and set(dynamic['needed']) <= _HOST_LIBS,
          'standalone ELF must use only host runtime dependencies')
    return dynamic


def _members(root, provider, update, controller=None):
    members = _MEMBERS
    if provider is not None:
        members = {**members, stateful_numpy_provider.SOURCE: 'scar.cpp-source',
                   stateful_numpy_provider.LICENSE: 'scar.upstream-license'}
    if update is not None:
        from scar.codegen.native_update_entry import COPY_SOURCE
        members = {**members, COPY_SOURCE: 'scar.cpp-source'}
    if controller is not None:
        members = {**members, **{name: row[0] for name, row in controller.members().items()}}
    return {name: {**_local(root, name, schema), 'uri': name} for name, schema in members.items()}


def _report(reference, metadata, plan, ports, root, provider, update, controller=None):
    report = dict(schema=SCHEMA, schema_version=1, cxx_standard=20, stateful_abi_version=1, original=reference,
                  original_build_id=metadata['build_id'],
                  identity={key: metadata[key] for key in ('region_sha256', 'plan_sha256', 'contract_sha256')},
                  ports=ports, files=_members(root, provider, update, controller), linkage=_linkage(root / _EXE),
                  cpu_ids=metadata['cpu_ids'], array_payload_bytes=plan['total_region_array_payload_bytes'],
                  array_encoding='C_ORDER_LITTLE_ENDIAN_BINARY64', status='REGION_ENTRY_EXPERIMENT',
                  effects='EXPLICIT_WRITTEN_ROOT_FILES_AFTER_NATIVE_SUCCESS',
                  native_provider_count=int(provider is not None), python_runtime_required=False,
                  whole_program_translated=False, deployment_authorized=False)
    if provider is not None:
        report['provider'] = provider[0]
    if update is not None:
        report['update'] = update.artifacts()
        report['status'] = 'COMPOSED_UPDATE_ENTRY_EXPERIMENT'
        report['effects'] = 'ARRAY_AND_CARRIED_SCALAR_FILES_AFTER_ALL_UPDATES_SUCCEED'
    if controller is not None:
        report['controller'] = controller.artifacts()
        report['status'] = 'SOURCE_CONTROLLED_NATIVE_ENTRY_EXPERIMENT'
        report['effects'] = 'SYNCHRONOUS_NATIVE_EFFECT_CALLBACKS_THEN_FINAL_ROOT_AND_STATE_FILES'
    return report


@replay_transaction()
def export(native, out, *, update_plan=None, controller_plan=None):
    """Publish a CMake project and executable using the original replayed ABI code."""
    path = Path(native).resolve(strict=True)
    reference = artifact_reference(path / 'artifact.json' if path.is_dir() else path, stateful.SCHEMA)
    metadata, plan, ports, original, provider = _admit(reference, observe_provider=True)
    _need(update_plan is None or controller_plan is None, 'provide only one enclosing plan')
    from scar.codegen.native_update_entry import COPY_SOURCE
    update = controller = None
    if controller_plan is not None:
        from scar.codegen.native_controller_entry import prepare
        from scar.compiler.plan_native_controller import SCHEMA as PLANNING
        controller = prepare(artifact_reference(Path(controller_plan).resolve(strict=True), PLANNING),
                             metadata, ports)
        update = controller.update
    elif update_plan is not None:
        from scar.codegen.native_update_entry import prepare
        from scar.compiler.plan_native_update import SCHEMA as PLANNING
        update_ref = artifact_reference(Path(update_plan).resolve(strict=True), PLANNING)
        update = prepare(update_ref, metadata, ports)
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
        (root / 'src/main.cpp').write_bytes(_render(ports, metadata, provider, update, controller))
        shutil.copyfile(original / metadata['files']['source'], root / 'src/stateful.cpp')
        for name in _HEADERS:
            shutil.copyfile(original / metadata['files'][name], include / name)
        if provider is not None:
            for name, payload in zip((stateful_numpy_provider.SOURCE, stateful_numpy_provider.LICENSE), provider[1:], strict=True):
                (root / name).parent.mkdir(exist_ok=True)
                (root / name).write_bytes(payload)
        if update is not None:
            (root / COPY_SOURCE).write_bytes(update.copy_source)
        if controller is not None:
            for name, (_, payload) in controller.members().items():
                (root / name).write_bytes(payload)
        _project(root, plan, provider, update, controller)
        _, dependencies = build_native_project(root, _TARGET, (compiler,))
        _need({Path(name).name for name in dependencies} <= _HOST_LIBS,
              'native executable has a non-host dependency')
        report = _report(reference, metadata, plan, ports, root, provider, update, controller)
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
    metadata, plan, ports, native, provider = _admit(reference)
    update = controller = None
    if 'controller' in original:
        from scar.codegen.native_controller_entry import prepare
        controller = prepare(original['controller']['planning'], metadata, ports)
        update = controller.update
        for name, (_, payload) in controller.members().items():
            _need((root / name).read_bytes() == payload, 'original controller member differs: ' + name)
    elif 'update' in original:
        from scar.codegen.native_update_entry import prepare
        update = prepare(original['update']['planning'], metadata, ports)
    if update is not None:
        from scar.codegen.native_update_entry import COPY_SOURCE
        _need((root / COPY_SOURCE).read_bytes() == update.copy_source, 'original array copy source differs')
    _need(original == _report(reference, metadata, plan, ports, root, provider, update, controller),
          'package contract, artifact bytes or ELF linkage differs')
    _need((root / 'src/main.cpp').read_bytes() == _render(ports, metadata, provider, update, controller),
          'native command ports differ from replay')
    for key, name in (('source', 'src/stateful.cpp'),
                      *((name, f'include/scar_generated/{name}') for name in _HEADERS)):
        _need((root / name).read_bytes() == (native / metadata['files'][key]).read_bytes(),
              'original C++ or ABI header differs: ' + name)
    if provider is not None:
        _need((root / stateful_numpy_provider.SOURCE).read_bytes() == provider[1]
              and (root / stateful_numpy_provider.LICENSE).read_bytes() == provider[2],
              'provider source or upstream notice differs')
        stateful_numpy_provider.audit(root / _EXE, provider[0])
    with TemporaryDirectory(prefix='scar-stateful-cmake-') as temporary:
        _project(Path(temporary), plan, provider, update, controller)
        _need((root / 'CMakeLists.txt').read_bytes() == (Path(temporary) / 'CMakeLists.txt').read_bytes(),
              'CMake project differs from replay')
    return original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check')
    parser.add_argument('--native')
    parser.add_argument('--out')
    plans = parser.add_mutually_exclusive_group()
    plans.add_argument('--update-plan')
    plans.add_argument('--controller-plan')
    args = parser.parse_args(argv)
    if args.check and not (args.native or args.out or args.update_plan or args.controller_plan):
        report = check(args.check)
    elif args.native and args.out and not args.check:
        report = export(args.native, args.out, update_plan=args.update_plan, controller_plan=args.controller_plan)
    else:
        parser.error('provide --native ARTIFACT --out PACKAGE, or --check PACKAGE')
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
