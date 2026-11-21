"""Lower a scoped complex128 normalization decision to stock CPU routines."""

import argparse
import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess

from scar.codegen.native_numpy_complex import _displacement, _instructions, _relocations, _symbols, _unrelocated
from scar.codegen.native_project import build_native_project, write_native_project
from scar.codegen.native_provider_bundle import build_native_provider_bundle, validate_native_provider_bundle
from scar.codegen.stateful import _CAPSULE, _Info
from scar.codegen.stateful_numpy_provider import _Image, _PIN, _layout, _tool
from scar.ir.execution_recipe import artifact_reference, read_artifact

SCHEMA = 'scar.native-numpy-vector-normalize'
WITNESS = 'scar.numpy-vector-normalization-witness'
SOURCE = 'src/numpy_vector_normalize.cpp'
HEADER = 'include/scar_generated/numpy_vector_normalize.hpp'
LICENSE = 'licenses/NumPy-LICENSE.txt'
_TARGET = 'scar_numpy_vector_normalize'
_DIVIDE = 'CDOUBLE_divide'
_SYMBOL = 'scar_numpy_stock_c128_divide'
_STRIDES = (16, 0, 16)
_SYSTEM = ('libm.so.6', 'libpthread.so.0', 'libc.so.6', 'ld-linux-x86-64.so.2', 'libz.so.1', 'libgcc_s.so.1')
_BLAS_SYMBOLS = ('scipy_cblas_ddot64_', 'scipy_cblas_zdotc_sub64_',
                 'scipy_openblas_get_num_threads64_', 'scipy_openblas_get_corename64_')
_ORDINARY = frozenset('add addpd addsd andpd cmovne cmp comisd divpd divsd mov movapd movddup movsd movupd '
                      'movups mulpd mulsd nop palignr pop push pxor ret setnp shufpd subpd test ucomisd xor'.split())
_ABI = dict(standard='C++20', symbol='scar_numpy_normalize_v1', shape='CONTIGUOUS_NATIVE_COMPLEX128_VECTOR',
            input_and_scratch='ALIGNED_FULLY_ACCESSIBLE_DISJOINT_BYTE_SPANS',
            codes={'-1': 'INVALID_PROVIDER_ENVIRONMENT_OR_OPERAND', '0': 'KEEP_INPUT',
                   '1': 'SELECT_SCRATCH', '2': 'REQUIRES_ORIGINAL_WARNING_OR_FP_PATH'},
            selection='STOCK_COMPLEX_DIVISION_STRICTLY_IMPROVES_SELF_VDOT_ERROR',
            restores_caller_fenv=True, owns_publication=False, python_runtime_required=False)
_HEADER = '''#pragma once
#include <cstddef>
extern "C" int scar_numpy_normalize_v1(std::size_t, const void*, void*) noexcept;
'''


def _need(valid, detail):
    if not valid:
        raise ValueError('native vector normalization: ' + detail)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _stock_divide(image, name):
    (address, size, _), = _symbols(image)[name]
    relocations = _relocations(image)
    _unrelocated(relocations, address, size)
    instructions = _instructions(image, address, size)
    boundaries = {row[0] for row in instructions}
    raw, literals = bytearray(), {}
    for pointer, instruction, opcode, arguments in instructions:
        raw.extend(instruction)
        if opcode.startswith('j'):
            _need(_displacement(pointer, instruction, arguments) in boundaries, 'divide branch leaves its loop')
        elif '[rip' in arguments:
            operand, _, comment = arguments.partition('#')
            target = re.match(r'^\s*([0-9a-f]+)\b', comment)
            relative = struct.unpack_from('<i', instruction, len(instruction) - 4)[0]
            location = pointer + len(instruction) + relative
            _need(opcode in ('movq', 'movsd') and 'QWORD PTR [rip' in operand and target is not None
                  and int(target[1], 16) == location and location % 8 == 0,
                  'divide has an unsupported relocation')
            _unrelocated(relocations, location, 8)
            literals[len(raw) - 4] = image.read(location, 8, 4).hex()
            raw[-4:] = bytes(4)
        else:
            _need(opcode in _ORDINARY and not re.search(r'\[(?:[a-z]+:)?0x|\b(?:fs|gs):', arguments),
                  'unsupported external instruction in stock divide: ' + opcode)
    _need(len(literals) == 2, 'stock divide literal set differs')
    return dict(symbol=name, address=address, extent=size, literals=[] if not literals else [
        dict(offset=offset, contents=value) for offset, value in literals.items()],
        normalized_sha256=sha256(raw).hexdigest()), bytes(raw)


def _source_dispatch(numpy, blas):
    image = _Image(numpy)
    _need(numpy['sha256'] == _PIN, 'expected an admitted exact NumPy binary revision')
    divide, raw = _stock_divide(image, _DIVIDE)
    for operation, symbol in (('DOUBLE_dot', _BLAS_SYMBOLS[0]), ('CDOUBLE_vdot', _BLAS_SYMBOLS[1])):
        _, listing = image.instructions(operation)
        _need(any(op == 'call' and f'<{symbol}@plt>' in arg for _, _, op, arg in listing),
              'NumPy reduction does not resolve to the expected stock BLAS')
    _, absolute = image.instructions('cdouble_absolute')
    cabs_line, = (arg for _, _, op, arg in absolute if op == 'call' and '<npy_cabs>' in arg)
    cabs_address = int(cabs_line.split(' ', 1)[0], 16)
    (start, size, _), = (row for row in _symbols(image)['npy_cabs'] if row[0] == cabs_address)
    _need(any(op == 'jmp' and '<cabs@plt>' in arg for _, _, op, arg in _instructions(image, start, size)),
          'NumPy scalar complex absolute differs from stock libc ABI')
    actual = _tool('ldd', numpy['uri'])
    expected = Path(blas['uri']).resolve(strict=True)
    _need(any(len(parts := line.strip().split()) >= 3 and parts[0] == expected.name and parts[1] == '=>'
              and Path(parts[2]).resolve() == expected for line in actual.splitlines()),
          'selected OpenBLAS differs from the loaded NumPy dependency')
    found = _symbols(_Image(blas))
    _need(all(any(row[2] == 'GLOBAL' for row in found.get(name, ())) for name in _BLAS_SYMBOLS),
          'expected exported OpenBLAS ILP64 operations and guards')
    return image, divide, raw


def observe(numpy, blas):
    """Bind the live NumPy division capsule and the original BLAS core."""
    import numpy as np
    from numpy._core import _multiarray_umath as extension

    image, divide, _ = _source_dispatch(numpy, blas)
    library = Path(numpy['uri'])
    _need(Path(extension.__file__).resolve() == library and extension.__cpu_features__['X86_V3']
          and np.divide is extension.divide, 'stock NumPy ufunc or ISA differs')
    adapter, fn_offset, data_offset, status_offset = _layout(image)
    base, = (int(row[0].split('-')[0], 16) for line in Path('/proc/self/maps').read_text().splitlines()
             if len(row := line.split(maxsplit=5)) == 6 and row[2] == '00000000' and row[5] == str(library))
    requested = tuple(np.dtype(x) for x in ('complex128', 'float64', 'complex128'))
    resolved, capsule = np.divide._resolve_dtypes_and_context(requested)
    np.divide._get_strided_loop(capsule, fixed_strides=_STRIDES)
    pointer = ctypes.pythonapi.PyCapsule_GetPointer
    pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    pointer.restype = ctypes.c_void_p
    info = _Info.from_address(pointer(capsule, _CAPSULE))
    _need(resolved == (requested[0],) * 3 and info.auxdata and not info.requires_pyapi
          and info.loop == base + adapter
          and ctypes.c_void_p.from_address(info.auxdata + fn_offset).value == base + divide['address']
          and ctypes.c_void_p.from_address(info.auxdata + data_offset).value is None
          and ctypes.c_int.from_address(info.auxdata + status_offset).value == 0,
          'live divide does not resolve to the qualified stock C128 loop')
    handle = ctypes.CDLL(blas['uri'], mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
    handle.scipy_openblas_get_corename64_.restype = ctypes.c_char_p
    threads = handle.scipy_openblas_get_num_threads64_()
    core = handle.scipy_openblas_get_corename64_().decode('ascii')
    _need(threads == 1 and re.fullmatch('[A-Za-z0-9_]+', core), 'require original single-thread BLAS dispatch')
    return dict(numpy=numpy, blas=blas, divide=divide, numpy_version=np.__version__,
                blas_core=core, blas_threads=threads, strides=list(_STRIDES),
                libm_abi={'complex_absolute': 'cabs@GLIBC_2.2.5', 'sqrt': 'sqrt@GLIBC_2.2.5'},
                exact_live_division_capsule=True)


def _assembly(divide, raw):
    def bytes_(block):
        return ['.byte ' + ', '.join(f'0x{x:02x}' for x in block[i:i+24]) for i in range(0, len(block), 24)]
    text = [f'.pushsection .text.{_SYMBOL},"ax",@progbits', '.p2align 4',
            f'.local {_SYMBOL}', f'.type {_SYMBOL}, @function', _SYMBOL + ':']
    cursor = 0
    for i, row in enumerate(divide['literals']):
        offset = row['offset']
        text.extend(bytes_(raw[cursor:offset]))
        text.append(f'.long .Lscar_normalize_literal_{i} - . - 4')
        cursor = offset + 4
    text.extend(bytes_(raw[cursor:]))
    text.extend([f'.size {_SYMBOL}, .-{_SYMBOL}', '.popsection',
                 '.pushsection .rodata.scar_normalize,"a",@progbits'])
    for i, row in enumerate(divide['literals']):
        text.extend(['.balign 8', f'.Lscar_normalize_literal_{i}:', *bytes_(bytes.fromhex(row['contents']))])
    text.append('.popsection')
    return '\n'.join(text)


def render(provider):
    """Give detached C++ sources. Link against the verified BLAS bundle root."""
    _, divide, raw = _source_dispatch(provider['numpy'], provider['blas'])
    core = provider['blas_core']
    _need(divide == provider['divide'] and provider['exact_live_division_capsule'] is True
          and provider['strides'] == list(_STRIDES) and provider['blas_threads'] == 1
          and re.fullmatch('[A-Za-z0-9_]+', core), 'provider dispatch receipt differs')
    result = Path(__file__).with_suffix('.cpp.in').read_text()
    result = result.replace('@SCAR_DIVIDE@', _assembly(divide, raw)).replace('@SCAR_BLAS_CORE@', core)
    licenses = Path(provider['numpy']['uri']).parents[2] / f'numpy-{provider["numpy_version"]}.dist-info/licenses/LICENSE.txt'
    return {SOURCE: result.encode(), HEADER: _HEADER.encode(), LICENSE: licenses.read_bytes()}


_DRIVER = r'''#include "numpy_vector_normalize.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) return 1;
    const auto n = std::stoull(argv[1]);
    std::vector<double> input(n * 2), scratch(n * 2);
    const auto bytes = static_cast<std::streamsize>(n * 16);
    std::ifstream file(argv[2], std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(input.data()), bytes) || file.peek() != EOF) return 2;
    const int status = scar_numpy_normalize_v1(n, input.data(), scratch.data());
    if (status < 0) return 4;
    std::cout << status << '\n';
    std::ofstream result(argv[3], std::ios::binary);
    if (!result.write(reinterpret_cast<const char*>((status == 1 ? scratch : input).data()), bytes)) return 5;
}
'''


def _check_witness(reference):
    witness = read_artifact(reference)
    read_artifact(witness['source'], json_document=False)
    _need(witness['schema_version'] == 1 and witness['capture_is_diagnostic'] is True
          and {row['outcome'] for row in witness['samples']} == {0, 1, 2}
          and len({row['id'] for row in witness['samples']}) == len(witness['samples']),
          'both original selection branches and warning frontier require witnesses')
    for row in witness['samples']:
        _need(re.fullmatch(r'[A-Za-z0-9_-]+', row['id']) and type(row['n']) is int and row['n'] > 0,
              'invalid recorded vector sample identity')
        expected = read_artifact(row['output'], json_document=False)
        original = read_artifact(row['input'], json_document=False)
        _need(len(expected) == len(original) == row['n'] * 16 and
              (row['outcome'] == 1 or expected == original), 'witness output differs from its original storage branch')
    return witness


def _provider_bundle(provider, root):
    directory = Path(provider['blas']['uri']).parent
    return build_native_provider_bundle(provider['blas'], root, provider_roots=(directory,), system_sonames=_SYSTEM)


def _command(binary, sample, output, cpu):
    return ['/usr/bin/taskset', '-c', str(cpu), str(binary), str(sample['n']), sample['input']['uri'], str(output)]


def _replay_sample(row, sample):
    expected = read_artifact(sample['output'], json_document=False)
    _need(row['status'] == sample['outcome'] and expected == read_artifact(row['output'], json_document=False),
          'native output/selection differs from the exact original vector on ' + sample['id'])


def _audit_divide(binary, provider):
    divide, _ = _stock_divide(_Image(_ref(binary, 'scar.native-executable')), _SYMBOL)
    original = provider['divide']
    _need(all(divide[key] == original[key] for key in ('extent', 'literals', 'normalized_sha256')),
          'linked division differs from original stock instructions')
    return divide


def export(witness_path, blas_path, out, *, cpu):
    reference = _ref(witness_path, WITNESS)
    witness = _check_witness(reference)
    provider = observe(witness['numpy_extension'], _ref(blas_path, 'scar.native-provider-library'))
    files = render(provider)
    files['src/main.cpp'] = _DRIVER.encode()
    root = Path(out).resolve()
    root.mkdir()
    for name, data in files.items():
        member = root / name
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_bytes(data)
    bundle = _provider_bundle(provider, root / 'provider')
    library = root / 'provider' / read_artifact(bundle)['root']
    write_native_project(root, _TARGET, sources=(SOURCE, 'src/main.cpp'),
                         compile_options=('-fno-fast-math', '-ffp-contract=off'), libraries=(str(library), 'm'),
                         runtime_dirs=('$ORIGIN/../provider/objects',), install_dirs=(('provider', '.'), ('licenses', '.')))
    binary, _ = build_native_project(root, _TARGET, (shutil.which('g++'),), expected_libraries=(library,))
    linked = _audit_divide(binary, provider)
    rows, directory = [], root / 'observed'
    directory.mkdir()
    for sample in witness['samples']:
        output = directory / (sample['id'] + '.bin')
        argv = _command(binary, sample, output, cpu)
        run = subprocess.run(argv, env={'OPENBLAS_NUM_THREADS': '1'}, cwd=root,
                             capture_output=True, check=True, timeout=20)
        _need(not run.stderr and run.stdout in (b'0\n', b'1\n', b'2\n'), 'unexpected native result transcript')
        row = dict(argv=argv, id=sample['id'], status=int(run.stdout), output=_ref(output, 'scar.numpy-normalization-vector'))
        _replay_sample(row, sample)
        rows.append(row)
    report = dict(schema=SCHEMA, schema_version=1, witness=reference, abi=_ABI, provider=provider,
                  blas_bundle=bundle, binary=_ref(binary, 'scar.native-executable'), linked_divide=linked,
                  members={name: _ref(root / name, 'scar.native-source-member') for name in files},
                  cpu_ids=[cpu], validation=rows, deployment_authorized=False, whole_entry=False)
    (root / 'result.json').write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return report


def check(path):
    reference = _ref(path, SCHEMA)
    stored = read_artifact(reference)
    witness = _check_witness(stored['witness'])
    bundle = validate_native_provider_bundle(stored['blas_bundle'])
    provider = stored['provider']
    main, = (row for row in bundle['members'] if row['path'] == bundle['root'])
    _need(stored['schema_version'] == 1 and stored['abi'] == _ABI and stored['deployment_authorized'] is False
          and stored['whole_entry'] is False and provider['numpy'] == witness['numpy_extension']
          and main['sha256'] == provider['blas']['sha256'], 'stored source contract or BLAS bytes differ')
    binary = stored['binary']['uri']
    read_artifact(stored['binary'], json_document=False)
    _need(_audit_divide(binary, provider) == stored['linked_divide'], 'linked stock divide changed')
    members = render(provider)
    members['src/main.cpp'] = _DRIVER.encode()
    _need(members == {key: read_artifact(value, json_document=False) for key, value in stored['members'].items()},
          'stored C++ orchestration differs from the qualified provider')
    cpu, = stored['cpu_ids']
    for row, sample in zip(stored['validation'], witness['samples'], strict=True):
        output = Path(reference['uri']).parent / 'observed' / (sample['id'] + '.bin')
        _need(row['id'] == sample['id'] and row['argv'] == _command(binary, sample, output, cpu)
              and row['output']['uri'] == str(output), 'retained original verification path differs')
        _replay_sample(row, sample)
    return stored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('witness', 'blas', 'out', 'check'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--cpu', type=int)
    args = parser.parse_args(argv)
    if args.check and not any((args.witness, args.blas, args.out, args.cpu is not None)):
        result = check(args.check)
    elif all((args.witness, args.blas, args.out, args.cpu is not None)) and args.check is None:
        result = export(args.witness, args.blas, args.out, cpu=args.cpu)
    else:
        parser.error('supply --check RESULT or --witness WITNESS --blas BLAS --cpu CPU --out OUT')
    print(json.dumps(dict(validated=len(result['validation']), abi=result['abi']), sort_keys=True))


if __name__ == '__main__':
    main()
