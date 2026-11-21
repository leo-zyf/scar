"""Build standalone C++20 loops from a stock complex array-tape schedule."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from scar.analysis.ndarray_tape_schedule import TAPE, lower_tape
from scar.analysis.target_graph import TargetHardwareGraph
from scar.codegen import native_numpy_complex as stock
from scar.codegen.native_cpu_elf import cpu_executable_dynamic
from scar.codegen.native_project import build_native_project, write_native_project
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program.target_hardware import HardwareThread


SCHEMA = 'scar.native-array-tape'
_NAME = 'scar_native_array_tape'
_BINARY = 'bin/' + _NAME
_TEMPLATE = Path(__file__).with_suffix('.cpp.in')


def _need(valid, message):
    if not valid:
        raise ValueError('native array tape: ' + message)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _emit(schedule):
    offsets = schedule['storage']['offset_bytes']
    rows = []

    def push(kind, count, *, offsets=(0, 0, 0), strides=(0, 0, 0), repeat=1, increment=(0, 0, 0), argument=0):
        triple = lambda values: '{' + ', '.join(map(str, values)) + '}'
        rows.append('    {Op::' + kind + ', ' + ', '.join(
            (str(count), str(repeat), triple(offsets), triple(strides), triple(increment), str(argument))) + '},')

    def view(value):
        return offsets[value['storage']] + value['offset_bytes']

    begins = []
    for row in schedule['operations']:
        begins.append(len(rows))
        kind = row['kind']
        if kind in ('MULTIPLY', 'ADD'):
            group = row['group']
            operands = group['operands']
            repeat = group.get('repeat', dict(count=1, offsets=[0] * 3))
            push(kind.lower(), group['count'],
                 offsets=tuple(offsets[item['root']] + item['offset'] for item in operands),
                 strides=tuple(item['stride'] for item in operands),
                 repeat=repeat['count'], increment=repeat['offsets'])
        elif kind == 'COPY':
            for group in row['groups']:
                push('copy', group['count'],
                     offsets=(offsets[row['source']] + group['read'], 0, offsets[row['target']] + group['write']),
                     strides=(group['increments'][0], 0, group['increments'][1]))
        elif kind == 'INITIAL_BASIS':
            push('basis', row['elements'], offsets=(0, 0, offsets[row['target']] + row['offset']),
                 argument=row['index'])
        elif kind == 'LOAD_LITERAL':
            push('literal', row['size'], offsets=(0, 0, offsets[row['target']]), argument=row['literal'])
        elif kind == 'NORMALIZE':
            push('normalize', row['elements'], offsets=(view(row['source']), 0, view(row['target'])),
                 argument=int(row['selected']))
        else:
            _need(kind == 'CHECKPOINT', 'unsupported lowered command')
            push('checkpoint', row['bytes'], offsets=(view(row['source']), 0, 0), argument=row['checkpoint'])

    literals = ['    {' + ', '.join(f'0x{byte:02x}' for byte in bytes.fromhex(raw)) + '},'
                for raw in schedule['literals']]
    begins.append(len(rows))
    rounds = [f'    {{{begins[start]}, {begins[end]}}},' for start, end in schedule['rounds']]
    text = '\n'.join((f'constexpr std::size_t arena_bytes = {schedule["storage"]["arena_bytes"]};',
        f'constexpr std::size_t checkpoint_count = {len(schedule["checkpoints"])};',
        f'constexpr std::array<std::array<unsigned char, 64>, {len(literals)}> coefficients{{{{',
        *literals, '}};', 'constexpr Step steps[]{', *rows, '};',
        'constexpr std::size_t rounds[][2]{', *rounds, '};'))
    return _TEMPLATE.read_text().replace('@STEPS@', text).encode()


def _normalize(reference):
    from scar.codegen.native_numpy_vector_normalize import check

    data = check(reference['uri'])
    _need(read_artifact(reference) == data, 'normalization provider changed during admission')
    members = {key: read_artifact(data['members'][key], json_document=False)
               for key in ('src/numpy_vector_normalize.cpp', 'include/scar_generated/numpy_vector_normalize.hpp')}
    bundle_ref = data['blas_bundle']
    bundle = read_artifact(bundle_ref)
    root = Path(bundle_ref['uri']).parent
    # The checked provider puts its exact ELF closure in the bundle's object directory.
    libraries = {Path(row['path']).name: root / row['path'] for row in bundle['members']}
    _need(libraries and Path(bundle['root']).name in libraries, 'normalization ELF bundle is incomplete')
    return members, libraries, Path(bundle['root']).name, data['provider']


def _runtime_env():
    return {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}


def _project(root, normalize):
    sources = ['src/main.cpp', stock.SOURCE]
    options = ['-fno-fast-math', '-ffp-contract=off']
    link = []
    libraries = []
    install = [('licenses', '.')]
    if normalize is not None:
        sources += ['src/numpy_vector_normalize.cpp']
        options += ['-DSCAR_ARRAY_TAPE_NORMALIZE']
        link += ['-Wl,--disable-new-dtags,-rpath,$ORIGIN/../provider/objects']
        libraries += ['-l:' + normalize, 'm']
        install.append(('provider', '.'))
    write_native_project(root, _NAME, sources=tuple(sources), compile_options=tuple(options),
                         link_options=tuple(link), libraries=tuple(libraries), install_dirs=tuple(install))
    if normalize is not None:
        with (root / 'CMakeLists.txt').open('a', encoding='utf-8') as stream:
            stream.write(f'target_link_directories({_NAME} PRIVATE "${{CMAKE_CURRENT_SOURCE_DIR}}/provider/objects")\n')
            stream.write(f'target_link_options({_NAME} PRIVATE "-Wl,-rpath-link,${{CMAKE_CURRENT_SOURCE_DIR}}/provider/objects")\n')


def _validate(binary, checkpoints, cpu, folder):
    folder.mkdir()
    prefix = folder / 'state'
    command = ['/usr/bin/taskset', '-c', str(cpu), str(binary), str(prefix)]
    output = subprocess.run(command, cwd=folder, env=_runtime_env(), capture_output=True, timeout=90)
    _need(output.returncode == 0 and not output.stderr and not output.stdout,
          'native array tape failed ' + str(output.returncode) + ' ' + output.stderr.decode(errors='replace'))
    rows = []
    for index, expected in enumerate(checkpoints):
        artifact = _ref(folder / f'state.{index}.bin', 'scar.ndarray-native-checkpoint')
        _need(all(artifact[key] == expected[key] for key in ('sha256', 'size_bytes')),
              'full state differs at original checkpoint ' + str(index))
        rows.append(artifact)
    return rows


def check(path):
    """Rebuild the schedule and re-run the sealed executable from a moved package."""
    from scar.analysis.ndarray_tape_projection import check as check_tape

    path = Path(path).resolve(strict=True)
    package = path.parent
    stored = read_artifact(_ref(path, SCHEMA))
    tape = check_tape(read_artifact(stored['tape']))
    schedule = lower_tape(tape)
    provider = stored['stock_provider']
    members = stock.render(provider)
    cpu, = tape['cpu_ids']
    _need(stored['cpu_ids'] == [cpu] and stored['binary'] == _BINARY
          and provider['provider']['sha256'] == tape['numpy_provider']['sha256']
          and stored['storage'] == schedule['storage']
          and stored['source_operation_counts'] == schedule['source_operation_counts'],
          'compiled graph does not match original storage, provider or scope')
    target = read_artifact(stored['target'])
    graph = TargetHardwareGraph(Path(stored['target']['uri']).parent)
    _need(graph.report == target and any(type(row) is HardwareThread and row.cpu_id == cpu
          and row.online is not False for row in graph.resources()), 'tape target or CPU changed')
    files = stored['files']
    for name, ref in files.items():
        _need(ref == _ref(package / name, 'scar.native-project-member'), 'native package member changed')
    members['src/main.cpp'] = _emit(schedule)
    resolved = None
    if stored['normalize'] is not None:
        norm_sources, libraries, resolved, norm_provider = _normalize(stored['normalize'])
        members.update(norm_sources)
        for name, original in libraries.items():
            saved = files['provider/objects/' + name]
            _need(saved['sha256'] == _ref(original, 'scar.native-project-member')['sha256'],
                  'bundled provider bytes changed')
    _need(all(read_artifact(files[name], json_document=False) == content for name, content in members.items()),
          'generated members differ from the sealed trace/plan')
    _need(json.loads(read_artifact(files['schedule.json'], json_document=False))
          == json.loads(json.dumps(schedule)), 'retained workspace schedule differs')
    with TemporaryDirectory(prefix='scar-array-replay-') as temp:
        replay = Path(temp)
        _project(replay, resolved)
        _need((replay / 'CMakeLists.txt').read_bytes() == (package / 'CMakeLists.txt').read_bytes(),
              'native project configuration differs')
        for name in files:
            if name == _BINARY or name.startswith('provider/objects/'):
                member = replay / name
                member.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(package / name, member)
        moved = replay / _BINARY
        linkage = dict(dynamic=cpu_executable_dynamic(moved), numpy=stock.audit(moved, provider))
        if resolved is not None:
            from scar.codegen.native_numpy_vector_normalize import _audit_divide

            linkage['normalization'] = _audit_divide(moved, norm_provider)
        _need(linkage == stored['linkage'], 'native linkage changed during relocation')
        fresh = _validate(moved, schedule['checkpoints'], cpu, replay / 'checkpoints')
        for old, live in zip(stored['checkpoint_validation'], fresh, strict=True):
            _need(read_artifact(old, json_document=False) == read_artifact(live, json_document=False),
                  'native checkpoint changed after relocation')
    return stored


def export(tape_path, target_path, out, *, normalize=None):
    from scar.analysis.ndarray_tape_projection import check

    tape_ref = _ref(tape_path, TAPE)
    tape = check(read_artifact(tape_ref))
    schedule = lower_tape(tape)
    target = Path(target_path).resolve(strict=True)
    graph = TargetHardwareGraph(target if target.is_dir() else target.parent)
    target_ref = _ref(graph.path / 'result.json', 'scar.target-hardware-graph')
    cpu, = tape['cpu_ids']
    _need(any(type(row) is HardwareThread and row.cpu_id == cpu and row.online is not False
              for row in graph.resources()), 'tape CPU is absent from exact target inventory')
    signatures = {key.lower(): values or [[None] * 3] for key, values in schedule['signatures'].items()}
    provider = stock.observe(strides=signatures)
    if tape.get('numpy_provider') is not None:
        image = tape['numpy_provider']
        _need(provider['provider']['sha256'] == image['sha256']
              and provider['provider']['uri'] == image['uri'], 'captured NumPy ELF differs from modern provider')
    required = any(row['kind'] == 'NORMALIZE' for row in schedule['operations'])
    _need(required == (normalize is not None), 'exact requested normalization provider is required')
    normal_ref = _ref(normalize, 'scar.native-numpy-vector-normalize') if required else None
    members = stock.render(provider)
    members['src/main.cpp'] = _emit(schedule)
    norm_files, norm_libraries, norm_entry, norm_provider = (
        _normalize(normal_ref) if required else ({}, {}, None, None))
    members.update(norm_files)
    compiler = shutil.which('g++')
    _need(compiler is not None, 'C++20 compiler unavailable')

    root = Path(out).absolute()
    _need(not root.exists(), 'fresh native release path required')
    root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-array-tape-', dir=root.parent) as temp:
        package = Path(temp) / 'release'
        package.mkdir()
        for name, payload in members.items():
            path = package / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        bundled = package / 'provider/objects'
        if norm_libraries:
            bundled.mkdir(parents=True)
            for name, path in norm_libraries.items():
                shutil.copyfile(path, bundled / name)
        _project(package, norm_entry)
        binary, _ = build_native_project(package, _NAME, (compiler,),
            expected_libraries=tuple(bundled / name for name in norm_libraries))
        linkage = dict(dynamic=cpu_executable_dynamic(binary), numpy=stock.audit(binary, provider))
        if required:
            from scar.codegen.native_numpy_vector_normalize import _audit_divide

            linkage['normalization'] = _audit_divide(binary, norm_provider)
        states = _validate(binary, schedule['checkpoints'], cpu, package / 'witness-validation')
        (package / 'schedule.json').write_text(json.dumps(schedule, sort_keys=True, separators=(',', ':')) + '\n')
        def relocated(reference):
            return dict(reference, uri=str(root / Path(reference['uri']).relative_to(package)))
        files = {name: relocated(_ref(package / name, 'scar.native-project-member'))
                 for name in sorted({*members, 'CMakeLists.txt', 'schedule.json', _BINARY,
                                     *(f'provider/objects/{name}' for name in norm_libraries)})}
        summary = dict(schema=SCHEMA, schema_version=1, tape=tape_ref, target=target_ref,
            stock_provider=provider, normalize=normal_ref, cpu_ids=[cpu], files=files,
            linkage=linkage, checkpoint_validation=[relocated(row) for row in states],
            storage=schedule['storage'], source_operation_counts=schedule['source_operation_counts'],
            cxx_standard=20, binary=_BINARY, scope='EXPLICIT_SCOPED_PRIVATE_ARRAY_TAPE',
            application_entry_equivalence=False, deployment_authorized=False, python_runtime_required=False)
        (package / 'result.json').write_text(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + '\n')
        package.rename(root)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--tape')
    source.add_argument('--check')
    parser.add_argument('--out')
    parser.add_argument('--target')
    parser.add_argument('--normalize')
    args = parser.parse_args(argv)
    if args.check and not any((args.target, args.out, args.normalize)):
        result = check(args.check)
    elif args.tape and args.target and args.out:
        result = export(args.tape, args.target, args.out, normalize=args.normalize)
    else:
        parser.error('use --check FILE or --tape FILE --target TARGET --out DIRECTORY [--normalize FILE]')
    print(json.dumps(dict(checkpoints=len(result['checkpoint_validation']), counts=result['source_operation_counts'],
                          arena_bytes=result['storage']['arena_bytes']), sort_keys=True))


if __name__ == '__main__':
    main()
