"""Export a checked ordered scalar region as a typed C++20 C ABI."""

import argparse
import ctypes as ffi
import json
import os
from pathlib import Path
import shlex
from shutil import copy2
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen import scalar_sorted_standalone as standalone
from scar.codegen import scalar_transition_resident as resident
from scar.codegen.scalar_table_standalone import _dependencies_for, _FLAGS, _LINK_FLAGS, _local, _ref, _require
from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.scalar-sorted-library'
SYMBOL = 'scar_scalar_sorted_v1'
_SONAME = 'libscar_scalar_sorted_v1.so'
_LIBRARY = 'lib/' + _SONAME
_SOURCE = 'src/sorted.cpp'
_HEADER = 'include/scar_scalar_sorted_v1.h'
_MAP = 'exports.map'
_MEMBERS = {_SOURCE: 'scar.cpp-source', _HEADER: 'scar.c-header', _MAP: 'scar.elf-export-map'}
_TEMPLATE = Path(__file__).with_name('scalar_sorted_library_v1.cpp.in')
_HEADER_TEMPLATE = Path(__file__).with_name('scalar_sorted_library_v1.h.in')
_NATIVE = ('-std=c++20', *_FLAGS, '-shared', '-fPIC', '-fvisibility=hidden', '-fvisibility-inlines-hidden',
           *_LINK_FLAGS, '-Wl,-soname,' + _SONAME, '-Wl,--version-script,exports.map')
_STATUS = dict(ok=0, need_capacity=1, declined=2, internal_error=3)


def _basis(plan_ref, capture_ref):
    source, verified = standalone._prepare(plan_ref, capture_ref)
    plan = read_artifact(plan_ref)
    capture = read_artifact(capture_ref)
    samples = capture['samples']
    record_ports = [dict(name='score', type='int'),
                    *(dict(name=f'arg_{i}', type=kind) for i, kind in enumerate(plan['record_types']))]
    field_names = [row['name'] for row in record_ports]
    records = [[dict(zip(field_names, result, strict=True)) for result in sample['records']] for sample in samples]
    _require(len(verified) == len(samples) and any(records), 'ordered library needs captured record extents')
    abi = dict(version=1, symbol=SYMBOL, input=resident._layout(plan['inputs'], [row['inputs'] for row in samples]),
               record=resident._layout(record_ports, [row for sample in records for row in sample]),
               table_slots=sorted(read_artifact(plan['transition'])['table_versions']), statuses=_STATUS,
               scalar_parameters=['threshold', 'depth'], max_records=standalone._MAX_ROWS,
               array_capacity='COUNT_OF_RECORDS', order='EXACT_ORIGINAL_STABLE_ORDER',
               output_effect='FIXED_POD_ARRAY_AFTER_COMPLETE_RESULT_VALIDATION')
    return source, abi, list(zip(samples, records, strict=True))


def _render(source, abi):
    rendered = source.decode()
    marker = 'void encode(std::string& output, Integer value) {\n'
    _require(rendered.count(marker) == 1, 'ordered C++ program has no unique publication boundary')
    computation = rendered.split(marker, 1)[0]
    inputs, checks, outputs = [], [], []
    for i, row in enumerate(abi['input']['ports']):
        slot = row['slot']
        if row['type'] == 'str':
            value = f'read_text(input->strings + {row["offset"]}, input->lengths[{slot}], {row["capacity"]})'
        else:
            value = f'input->scalars[{slot}]'
            if row['type'] == 'bool':
                value = f'read_bool({value})'
        inputs.append(f'        const auto in_{i} = {value};')
    for i, row in enumerate(abi['record']['ports']):
        slot = row['slot']
        expression = 'std::get<0>(row)' if i == 0 else f'std::get<{i - 1}>(std::get<1>(row))'
        if row['type'] == 'str':
            checks.append(f'            if (!valid_text({expression}, {row["capacity"]})) throw Decline{{}};')
            outputs.append(f'            write_text(staged.strings + {row["offset"]}, staged.lengths[{slot}], {expression});')
        else:
            outputs.append(f'            staged.scalars[{slot}] = {expression};')
    preflight = '\n'.join(['        for (const auto& row : ordered_records) {', *checks, '        }']) if checks else ''
    fields = dict(COMPUTATION=computation, INPUT_SETUP='\n'.join(inputs),
                  INPUTS=', '.join(f'in_{i}' for i in range(len(inputs))),
                  PREFLIGHT=preflight, OUTPUT_SETUP='\n'.join(outputs))
    generated = _TEMPLATE.read_text()
    for key, value in fields.items():
        generated = generated.replace('@SCAR_' + key + '@', value)
    header = _HEADER_TEMPLATE.read_text()
    for direction in ('input', 'record'):
        for field in ('scalars', 'lengths', 'strings'):
            header = header.replace('@SCAR_' + direction.upper() + '_' + field.upper() + '@', str(abi[direction][field]))
    return {_SOURCE: generated.encode(), _HEADER: header.encode(), _MAP: f'{{ global: {SYMBOL}; local: *; }};\n'.encode()}


def pod_types(abi):
    """Return ctypes types corresponding to the two public C structs."""
    inputs, records, _ = resident.resident_types(dict(input=abi['input'], output=abi['record']))
    return inputs, records


def load(path, types):
    library = ffi.CDLL(str(Path(path).resolve(strict=True)))
    call = getattr(library, SYMBOL)
    inputs, record = types
    call.argtypes = [ffi.c_uint32, ffi.POINTER(inputs), ffi.c_int64, ffi.c_int64,
                     ffi.POINTER(record), ffi.c_size_t, ffi.POINTER(ffi.c_size_t)]
    call.restype = ffi.c_int
    return call


def _verify(path, abi, samples):
    types = pod_types(abi)
    inputs, record = types
    call = load(path, types)
    size = ffi.sizeof(record)
    capacity = max(len(records) for _, records in samples)
    storage = (record * capacity)()
    written = ffi.c_size_t()
    native = []
    for sample, records in samples:
        value = resident.pack(abi['input'], inputs, sample['inputs'])
        slot = abi['table_slots'].index(sample['table_version'])
        expected = b''.join(bytes(resident.pack(abi['record'], record, row)) for row in records)
        ffi.memset(ffi.addressof(storage), 0xA5, ffi.sizeof(storage))
        status = call(slot, ffi.byref(value), sample['threshold'], sample['nonzero_capture'],
                      storage, capacity, ffi.byref(written))
        payload = bytes(storage)[:size * written.value]
        _require(status == 0 and written.value == len(records) and payload == expected
                 and bytes(storage)[len(payload):] == b'\xa5' * (ffi.sizeof(storage) - len(payload)),
                 'ordered POD array differs from the original sequence or writes beyond its returned count')
        native.append(payload)

    sample, records = next((row, records) for row, records in samples if records)
    value = resident.pack(abi['input'], inputs, sample['inputs'])
    slot = abi['table_slots'].index(sample['table_version'])
    prefix = (ffi.byref(value), sample['threshold'], sample['nonzero_capture'])
    ffi.memset(ffi.addressof(storage), 0xA5, ffi.sizeof(storage))
    unchanged = bytes(storage)
    for data, extent in ((None, 0), (storage, len(records) - 1)):
        status = call(slot, *prefix, data, extent, ffi.byref(written))
        _require(status == 1 and written.value == len(records) and bytes(storage) == unchanged,
                 'ordered library capacity failure wrote part of the result')
    status = call(len(abi['table_slots']), *prefix, storage, capacity, ffi.byref(written))
    _require(status == 2 and not written.value and bytes(storage) == unchanged,
             'ordered library invalid table version changed the result bytes')
    guard = next((row for row in abi['input']['ports'] if row['type'] == 'bool'), None)
    if guard is not None:
        value.scalars[guard['slot']] = 2
        status = call(slot, *prefix, storage, capacity, ffi.byref(written))
        _require(status == 2 and not written.value and bytes(storage) == unchanged,
                 'ordered library invalid boolean changed the result bytes')
    return b''.join(native)


def _links(path):
    links = _dependencies_for(path)
    exports = subprocess.check_output(['/usr/bin/nm', '-D', '--defined-only', '--format=posix', str(path)], env={}, text=True)
    _require([row.split()[0] for row in exports.splitlines()] == [SYMBOL], 'ordered library has unexpected public symbols')
    return links


def _validation(root, abi, samples):
    counts = [len(records) for _, records in samples]
    size = ffi.sizeof(pod_types(abi)[1])
    evidence = _local(root, 'samples.pod', 'application/octet-stream')
    _require(evidence['size_bytes'] == sum(counts) * size, 'ordered sample record extent differs from the native POD')
    return dict(samples=len(counts), per_sample_records=counts, records=sum(counts), record_bytes=size,
                relocated=True, mode='IN_PROCESS_C_ABI', output=evidence)


def export(plan_path, capture_path, out):
    """Publish one checked eager ordered region in a fresh native DSO package."""
    plan_ref = _ref(plan_path, standalone.region.SCHEMA)
    capture_ref = _ref(capture_path, standalone.region.CAPTURE)
    source, abi, samples = _basis(plan_ref, capture_ref)
    rendered = _render(source, abi)
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    _require(compiler, 'an available C++20 compiler is required')
    environment = {key: value for key, value in os.environ.items() if not key.startswith('LD_')}
    version = subprocess.check_output([*compiler, '--version'], env=environment, text=True).splitlines()[0]
    with TemporaryDirectory(prefix='.scar-sorted-library-', dir=out.parent) as temporary:
        stage = Path(temporary) / 'package'
        for name, data in rendered.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        (stage / _LIBRARY).parent.mkdir()
        subprocess.run([*compiler, *_NATIVE, '-Iinclude', _SOURCE, '-o', _LIBRARY], cwd=stage, env=environment, check=True)
        relocated = Path(temporary) / 'relocated ordered region.so'
        copy2(stage / _LIBRARY, relocated)
        linkage = _links(relocated)
        (stage / 'samples.pod').write_bytes(_verify(relocated, abi, samples))
        record = dict(schema=SCHEMA, schema_version=1, cxx_standard=20, region=plan_ref, capture=capture_ref,
                      abi=abi, compiler=dict(command=compiler, version=version, flags=list(_NATIVE)), linkage=linkage,
                      library=_local(stage, _LIBRARY, 'scar.native-library-file'),
                      header=_local(stage, _HEADER, 'scar.c-header'),
                      members={name: _local(stage, name, kind) for name, kind in _MEMBERS.items()},
                      validation=_validation(stage, abi, samples), whole_program_cpp=False,
                      deployment_authorized=False)
        (stage / 'artifact.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
        if out.exists():
            raise FileExistsError(out)
        stage.rename(out)
    return _ref(out / 'artifact.json', SCHEMA)


def check(path):
    """Replay the original scope, generated ABI, ELF linkage and every sample array."""
    ref = _ref(path, SCHEMA)
    result = read_artifact(ref)
    root = Path(ref['uri']).parent
    source, abi, samples = _basis(result['region'], result['capture'])
    rendered = _render(source, abi)
    library = root / _LIBRARY
    _require(result['schema_version'] == 1 and result['cxx_standard'] == 20 and result['abi'] == abi
             and result['deployment_authorized'] is False and result['whole_program_cpp'] is False
             and result['members'] == {name: _local(root, name, kind) for name, kind in _MEMBERS.items()}
             and all((root / name).read_bytes() == data for name, data in rendered.items())
             and result['library'] == _local(root, _LIBRARY, 'scar.native-library-file')
             and result['header'] == result['members'][_HEADER] and result['compiler']['flags'] == list(_NATIVE)
             and result['linkage'] == _links(library), 'ordered library differs from its checked portable artifact')
    _require(result['validation'] == _validation(root, abi, samples)
             and (root / 'samples.pod').read_bytes() == _verify(library, abi, samples),
             'ordered library arrays differ from fresh native verification')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan')
    parser.add_argument('--capture')
    parser.add_argument('--out')
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    if args.check and not any((args.plan, args.capture, args.out)):
        result = check(args.check)
    elif args.plan and args.capture and args.out and not args.check:
        result = export(args.plan, args.capture, args.out)
    else:
        parser.error('provide --plan, --capture and --out or --check ARTIFACT')
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
