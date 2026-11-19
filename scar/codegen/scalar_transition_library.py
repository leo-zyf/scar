"""Generate a typed native C ABI from a checked scalar transition plan."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import shlex
from shutil import copy2
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen import scalar_transition_standalone as transition
from scar.codegen.scalar_table_standalone import _dependencies_for, _FLAGS, _LINK_FLAGS, _local, _ref, _require
from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.scalar-transition-native-library'
SYMBOL = 'scar_scalar_transition_v1'
ARGUMENT_KINDS = {'str': 1, 'int': 2, 'bool': 3}
_STATUS = dict(ok=0, need_capacity=1, declined=2, internal_error=3)
_SCOPE = 'DETACHED_TYPED_STATE_TRANSITION_NATIVE_SCORER'
_HEADER = 'include/scar_scalar_transition_v1.h'
_SOURCE = 'src/transition.cpp'
_SONAME = 'libscar_scalar_transition_v1.so'
_LIBRARY = 'lib/' + _SONAME
_MAP = 'exports.map'
_TEMPLATE = Path(__file__).with_name('scalar_transition_library_v1.cpp.in')
_HEADER_TEMPLATE = Path(__file__).with_name('scalar_transition_library_v1.h.in')
_SHARED = ('-std=c++20', *_FLAGS, '-shared', '-fPIC', '-fvisibility=hidden', '-fvisibility-inlines-hidden',
           *_LINK_FLAGS, '-Wl,-soname,' + _SONAME, '-Wl,--version-script,exports.map')
_MEMBERS = {_SOURCE: 'scar.cpp-source', _HEADER: 'scar.c-header', _MAP: 'scar.elf-export-map'}


class Argument(ctypes.Structure):
    """Host mirror of the generated C declaration, used for package verification."""

    _fields_ = [('data', ctypes.c_void_p), ('size', ctypes.c_size_t), ('scalar', ctypes.c_int64),
                ('kind', ctypes.c_uint32)]


def _render(plan, scorer):
    source = transition._render(plan, scorer).decode()
    marker = 'int publish(const State& state) {\n'
    end = '    return std::fwrite(record.data(), 1, record.size(), stdout)'
    _require(source.count(marker) == source.count(end) == 1, 'standalone transition has no unique publication seam')
    computation, publisher = source.split(marker, 1)
    encoder, _ = publisher.split(end, 1)
    computation += 'std::string encode_state(const State& state) {\n' + encoder + '    return record;\n}'
    ports = transition._schema(plan['inputs'])
    parsers = {'str': 'argument_string', 'int': 'argument_integer', 'bool': 'argument_bool'}
    setup = '\n'.join(f'            const auto in_{i} = {parsers[kind]}(args[{i}]);'
                      for i, kind in enumerate(ports.values()))
    inputs = ', '.join(f'in_{i}' for i in range(len(ports)))
    generated = _TEMPLATE.read_text()
    for key, value in dict(COMPUTATION=computation, COUNT=str(len(ports)), INPUT_SETUP=setup, INPUTS=inputs).items():
        generated = generated.replace('@SCAR_' + key + '@', value)
    return {_SOURCE: generated.encode(), _HEADER: _HEADER_TEMPLATE.read_bytes(),
            _MAP: f'{{\n    global: {SYMBOL};\n    local: *;\n}};\n'.encode()}


def _abi(plan):
    return dict(version=1, symbol=SYMBOL, argument_struct='scar_transition_arg_v1', argument_kinds=ARGUMENT_KINDS,
                inputs=plan['inputs'], outputs=plan['outputs'], table_version='ASCII_SHA256_WITH_EXPLICIT_LENGTH',
                integer_intermediates='SIGNED_64_CHECKED', record='SCAR_STATE_V1', statuses=_STATUS,
                output='CALLER_OWNED_BUFFER_COMMITTED_ONLY_ON_SUCCESS', scorer='NATIVE')


def _links(path):
    libs = _dependencies_for(path)
    dynamic = subprocess.check_output(['/usr/bin/nm', '-D', '--defined-only', '--format=posix', str(path)], env={}, text=True)
    _require([row.split()[0] for row in dynamic.splitlines()] == [SYMBOL], 'native transition exported unexpected symbols')
    return libs


def _load(path):
    library = ctypes.CDLL(str(path.resolve(strict=True)))
    native = getattr(library, SYMBOL)
    native.restype = ctypes.c_int
    native.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(Argument), ctypes.c_size_t,
                       ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
    return native


def _arguments(plan, sample):
    # The standalone input validator already verifies exact types and port order.
    transition.arguments(plan, sample)
    ports = transition._schema(plan['inputs'])
    buffers, args = [], []
    for name, kind in ports.items():
        value = sample['inputs'][name]
        if kind == 'str':
            raw = value.encode('ascii')
            buffer = ctypes.create_string_buffer(raw)
            buffers.append(buffer)
            arg = Argument(ctypes.addressof(buffer), len(raw), 0, ARGUMENT_KINDS[kind])
        else:
            arg = Argument(None, 0, int(value), ARGUMENT_KINDS[kind])
        args.append(arg)
    return (Argument * len(args))(*args), buffers


def _verify(path, plan):
    call = _load(path)
    samples = plan['samples']
    expected = [transition.encode_outputs(plan['outputs'], row['outputs']) for row in samples]
    _require(expected and {sample['table_version'] for sample in samples} == plan['table_versions'].keys(),
             'each library table version requires a matched original state')
    capacity = max(map(len, expected))
    storage = (ctypes.c_ubyte * capacity)()
    written = ctypes.c_size_t()
    outputs = []
    for sample, reference in zip(samples, expected, strict=True):
        args, keepalive = _arguments(plan, sample)
        version = sample['table_version'].encode('ascii')
        status = call(version, len(version), args, len(args), storage, capacity, ctypes.byref(written))
        output = bytes(storage[:written.value])
        _require(status == _STATUS['ok'] and output == reference and written.value == len(reference),
                 'native transition library differs from the original state')
        outputs.append(output)

    ctypes.memset(storage, 0xA5, capacity)
    unchanged = bytes(storage)
    first = samples[0]
    valid = first['table_version'].encode('ascii')
    args, keepalive = _arguments(plan, first)
    required = len(expected[0])
    for destination, extent in ((None, 0), (storage, required - 1)):
        status = call(valid, len(valid), args, len(args), destination, extent, ctypes.byref(written))
        _require(status == _STATUS['need_capacity'] and written.value == required and bytes(storage) == unchanged,
                 'native transition library capacity query changed the output buffer')
    unknown = b'0' * 64
    if unknown.decode() in plan['table_versions']:
        unknown = b'x' * 64
    status = call(unknown, len(unknown), args, len(args), storage, capacity, ctypes.byref(written))
    _require(status == _STATUS['declined'] and not written.value and bytes(storage) == unchanged,
             'native transition library rejected version after writing output')
    args[0].kind = 0
    status = call(valid, len(valid), args, len(args), storage, capacity, ctypes.byref(written))
    _require(status == _STATUS['declined'] and not written.value and bytes(storage) == unchanged,
             'native transition library rejected type after writing output')
    return b''.join(outputs)


def _validation(root, plan):
    return dict(samples=len(plan['samples']), method='IN_PROCESS_C_ABI', relocated=True,
                capacity_and_domain_leave_output_unchanged=True,
                output=_local(root, 'samples.wire', 'application/octet-stream'))


def export(plan_path, out):
    """Build and qualify the native-scorer C ABI in a fresh package."""
    plan_ref, plan, scorer = transition._plan(plan_path)
    members = _render(plan, scorer)
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    _require(compiler, 'an available C++20 compiler is required')
    environment = {key: value for key, value in os.environ.items() if not key.startswith('LD_')}
    version = subprocess.check_output([*compiler, '--version'], env=environment, text=True).splitlines()[0]
    with TemporaryDirectory(prefix='.scar-transition-library-', dir=out.parent) as temporary:
        stage = Path(temporary) / 'package'
        for name, data in members.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        (stage / _LIBRARY).parent.mkdir()
        command = [*compiler, *_SHARED, '-Iinclude', _SOURCE, '-o', _LIBRARY]
        subprocess.run(command, cwd=stage, env=environment, check=True)
        relocated = Path(temporary) / 'relocated transition library.so'
        copy2(stage / _LIBRARY, relocated)
        links = _links(relocated)
        (stage / 'samples.wire').write_bytes(_verify(relocated, plan))
        record = dict(schema=SCHEMA, schema_version=1, scope=_SCOPE, cxx_standard=20, plan=plan_ref,
                      score_plan=plan['score_plan'], abi=_abi(plan), table_versions=sorted(plan['table_versions']),
                      compiler=dict(command=compiler, version=version, options=list(_SHARED)), linkage=links,
                      library=_local(stage, _LIBRARY, 'scar.native-library-file'), header=_local(stage, _HEADER, 'scar.c-header'),
                      members={name: _local(stage, name, kind) for name, kind in _MEMBERS.items()},
                      validation=_validation(stage, plan), whole_program_cpp=False, deployment_authorized=False)
        (stage / 'artifact.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
        if out.exists():
            raise FileExistsError(out)
        stage.rename(out)
    return _ref(out / 'artifact.json', SCHEMA)


def check(path):
    """Replay the checked source, local DSO bytes and all in-process state records."""
    ref = _ref(path, SCHEMA)
    record = read_artifact(ref)
    root = Path(ref['uri']).parent
    plan_ref, plan, scorer = transition._plan(record['plan']['uri'])
    _require(record['schema_version'] == 1 and record['scope'] == _SCOPE and record['cxx_standard'] == 20
             and record['whole_program_cpp'] is False and record['deployment_authorized'] is False
             and record['plan'] == plan_ref and record['score_plan'] == plan['score_plan']
             and record['table_versions'] == sorted(plan['table_versions']) and record['abi'] == _abi(plan),
             'native transition library boundary differs from the checked source plan')
    source = _render(plan, scorer)
    library = root / _LIBRARY
    _require(record['members'] == {name: _local(root, name, kind) for name, kind in _MEMBERS.items()}
             and all((root / name).read_bytes() == data for name, data in source.items())
             and record['header'] == record['members'][_HEADER]
             and record['library'] == _local(root, _LIBRARY, 'scar.native-library-file')
             and record['compiler']['options'] == list(_SHARED) and record['linkage'] == _links(library),
             'native transition library code, ABI, or linkage differs from its receipt')
    _require(record['validation'] == _validation(root, plan)
             and (root / 'samples.wire').read_bytes() == _verify(library, plan),
             'native transition library states differ from fresh in-process replay')
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan')
    parser.add_argument('--out')
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    if args.check and not any((args.plan, args.out)):
        result = check(args.check)
    elif args.plan and args.out and not args.check:
        result = export(args.plan, args.out)
    else:
        parser.error('provide --plan and --out or --check ARTIFACT')
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
