"""Bind and relocate the exact live NumPy complex128 CPU loops."""

import argparse
import ctypes
from hashlib import sha256
import json
from pathlib import Path
import re
import struct

from scar.codegen.stateful import _CAPSULE, _Info
from scar.codegen.stateful_numpy_provider import _Image, _INSTRUCTION, _PIN, _layout, _tool
from scar.ir.execution_recipe import artifact_reference, read_artifact

SCHEMA = 'scar.native-numpy-complex'
SOURCE = 'src/numpy_complex.cpp'
HEADER = 'include/scar_generated/numpy_complex.hpp'
LICENSE = 'licenses/NumPy-LICENSE.txt'
_NAMES = dict(add='CDOUBLE_add_X86_V3', multiply='CDOUBLE_multiply_X86_V3', pairwise_sum='CDOUBLE_pairwise_sum')
_NATIVE = {role: 'scar_numpy_complex_' + role for role in _NAMES}
_ROOTS = ('add', 'multiply')
_REFERENCE = re.compile(r'^([0-9a-f]+)(?: <[^>]+>)?$')
_ORDINARY = frozenset('''
add and cmovns cmovs cmp imul lea mov neg nop or pop prefetcht0 push ret sar setb
sete setne shl shr sub test vaddpd vaddsd vaddsubpd vblendpd vbroadcastsd
vextractf128 vextracti128 vfmaddsub132pd vfmaddsub231pd vinsertf128 vinserti128
vmovapd vmovdqa vmovdqu vmovq vmovsd vmovupd vmulpd vpblendvb vpcmpeqd vperm2f128
vpermilpd vpmaskmovq vpunpcklqdq vpxor vunpckhpd vunpcklpd vzeroall
vzeroupper xchg xor
'''.split())


def _need(valid, reason):
    if not valid:
        raise ValueError('native NumPy complex: ' + reason)


def _profiles(strides):
    if strides is None:
        return {role: [[None, None, None]] for role in _ROOTS}
    _need(set(strides) == set(_ROOTS), 'expected both source ufunc stride profiles')
    result = {}
    for role in _ROOTS:
        rows = [list(row) for row in strides[role]]
        _need(rows and all(len(row) == 3 and all(value is None or type(value) is int and
              -(1 << 63) <= value < 1 << 63 and value % 16 == 0 for value in row) for row in rows),
              'stock finite-stride profile differs')
        _need(len({tuple(row) for row in rows}) == len(rows), 'duplicate source stride profile')
        result[role] = rows
    return result


def _symbols(image):
    symbols = {}
    for line in _tool('readelf', '-Ws', image.path).splitlines():
        row = line.split()
        if len(row) == 8 and row[3] == 'FUNC' and row[6].isdecimal():
            symbols.setdefault(row[7], set()).add((int(row[1], 16), int(row[2], 0), row[4]))
    return symbols


def _relocations(image):
    return sorted({int(row[0], 16) for line in _tool('readelf', '-Wr', image.path).splitlines()
                   if len(row := line.split()) >= 3 and row[2].startswith('R_X86_64_')})


def _unrelocated(relocations, start, size):
    _need(not any(address < start + size and start < address + 8 for address in relocations),
          'loader relocation touches extracted text or constant bytes')


def _instructions(image, base, size):
    _need(0 < size <= 4096, 'unexpected function extent')
    result = []
    listing = _tool('objdump', '-dw', '-Mintel', f'--start-address={base}', f'--stop-address={base + size}', image.path)
    cursor = base
    for line in listing.splitlines():
        if match := _INSTRUCTION.fullmatch(line):
            address, raw = int(match[1], 16), bytes.fromhex(match[2])
            _need(address == cursor, 'stock function has an instruction gap')
            result.append((address, raw, match[3], match[4]))
            cursor += len(raw)
    _need(cursor == base + size and b''.join(row[1] for row in result) == image.read(base, size, 5),
          'decoded instructions differ from executable bytes')
    return result


def _displacement(position, raw, argument, *, call=False):
    target = _REFERENCE.fullmatch(argument)
    _need(target is not None, 'indirect or unqualified control transfer')
    destination = int(target[1], 16)
    if call:
        _need(len(raw) == 5 and raw[0] == 0xe8, 'non-rel32 local call')
    else:
        short = len(raw) == 2 and (0x70 <= raw[0] <= 0x7f or raw[0] == 0xeb)
        long = (len(raw) == 5 and raw[0] == 0xe9 or
                len(raw) == 6 and raw[0] == 0x0f and 0x80 <= raw[1] <= 0x8f)
        _need(short or long, 'unrecognized stock branch encoding')
        if short:
            _need(position + len(raw) + struct.unpack_from('<b', raw, len(raw) - 1)[0] == destination,
                  'stock short branch disagrees with its displacement')
            return destination
    _need(position + len(raw) + struct.unpack_from('<i', raw, len(raw) - 4)[0] == destination,
          'stock long transfer disagrees with its displacement')
    return destination


def _decode(image, extent, original_relocations):
    address, size, _ = extent
    _unrelocated(original_relocations, address, size)
    rows = _instructions(image, address, size)
    boundaries = {row[0] for row in rows}
    code, transfers = bytearray(), []
    for position, raw, op, argument in rows:
        offset = position - address + len(raw) - 4
        code.extend(raw)
        if op == 'call':
            transfers.append((offset, 'CALL', _displacement(position, raw, argument, call=True)))
            code[-4:] = bytes(4)
        elif op.startswith('j'):
            _need(_displacement(position, raw, argument) in boundaries, 'stock branch leaves its function')
        elif '[rip' in argument and op not in ('cs', 'data16', 'nop'):
            operands, marker, annotation = argument.partition('#')
            target = _REFERENCE.fullmatch(annotation.strip())
            widths = {'vmovsd': ('QWORD', 8), 'vmovdqa': ('YMMWORD', 32)}
            _need(marker and target and op in widths and len(raw) >= 6, 'unknown stock RIP-relative reference')
            word, span = widths[op]
            destination = position + len(raw) + struct.unpack_from('<i', raw, len(raw) - 4)[0]
            _need(f'{word} PTR [rip' in operands and destination == int(target[1], 16) and destination % span == 0,
                  'stock literal reference has another address or alignment')
            _unrelocated(original_relocations, destination, span)
            transfers.append((offset, 'LITERAL', image.read(destination, span, 4).hex()))
            code[-4:] = bytes(4)
        elif op == 'movabs':
            _need(argument == 'rbx,0x3ff0000000000000', 'unqualified immediate address in stock function')
        elif op in ('cs', 'data16', 'nop'):
            _need(op == 'nop' or 'nop' in argument, 'unexpected stock instruction prefix')
        else:
            _need(op in _ORDINARY and
                  not re.search(r'\[(?:[a-z]+:)?0x[0-9a-f]+\]|\b(?:fs|gs):|\b[csde]s:0x', argument),
                  'unqualified stock instruction or absolute operand: ' + op)
    return bytes(code), transfers


def _closure(image, names, *, original):
    symbols, relocations = _symbols(image), _relocations(image)
    extents = {}
    for role in _ROOTS:
        candidates = symbols.get(names[role], ())
        _need(len(candidates) == 1, 'stock root symbol does not have one precise range')
        extent, = candidates
        _need(not original or extent[2] == 'LOCAL', 'stock root must be a local regular function')
        extents[role] = extent
    root_code = {key: _decode(image, value, relocations) for key, value in extents.items()}
    targets = [value for _, kind, value in root_code['add'][1] if kind == 'CALL']
    _need(len(targets) == 1 and not any(kind == 'CALL' for _, kind, _ in root_code['multiply'][1]),
          'stock roots reach an unsupported call closure')
    candidates = {row for row in symbols.get(names['pairwise_sum'], ()) if row[0] == targets[0] and
                  (not original or row[2] == 'LOCAL')}
    _need(len(candidates) == 1, 'stock add does not reach the exact local pairwise helper')
    extents['pairwise_sum'], = candidates
    root_code['pairwise_sum'] = _decode(image, extents['pairwise_sum'], relocations)
    result, code = {}, {}
    for role, (raw, refs) in root_code.items():
        normalized = []
        for offset, kind, target in refs:
            if kind == 'CALL':
                _need(target == extents['pairwise_sum'][0], 'call closure has an unresolved or Python edge')
                target = 'pairwise_sum'
            normalized.append(dict(offset=offset, kind=kind, target=target))
        result[role] = dict(symbol=names[role], address=extents[role][0], extent=extents[role][1],
                            normalized_sha256=sha256(raw).hexdigest(), transfers=normalized)
        code[role] = raw
    return result, code


def _derive(reference, profiles):
    _need(reference['schema'] == 'scar.numpy-provider-elf' and reference['sha256'] == _PIN,
          'unqualified exact NumPy complex binary revision')
    image = _Image(reference)
    functions, code = _closure(image, _NAMES, original=True)
    adapter, function, data, status = _layout(image)
    address, rows = image.instructions('generic_wrapped_legacy_loop')
    _need(address == adapter and [(op, arguments) for _, _, op, arguments in rows[2:7]] == [
        ('mov', 'rdi,rsi'), ('mov', 'rsi,rdx'), ('mov', 'rdx,rcx'),
        ('mov', f'rcx,QWORD PTR [r8+{data:#x}]'), ('call', f'QWORD PTR [r8+{function:#x}]')],
        'stock adapter argument order differs from the direct legacy ABI')
    notice = Path(reference['uri']).parents[2] / 'numpy-2.5.2.dist-info/licenses/LICENSE.txt'
    license = artifact_reference(notice, 'scar.upstream-license')
    report = dict(schema=SCHEMA, schema_version=1, provider=reference, numpy_version='2.5.2',
                  ndarray_c_abi=33554432, notice=license, dtype=['complex128'] * 3, strides=profiles,
                  selection='LIVE_UFUNC_CAPSULE_ELF_RELATIVE_TARGET', native_symbols=_NATIVE,
                  machine_functions=functions, adapter=dict(address=adapter, function_offset=function,
                      loop_data_offset=data, python_error_check_offset=status),
                  isa='X86_64_V3', floating_mode='FE_TONEAREST_WITH_GRADUAL_UNDERFLOW',
                  requires_pyapi=False, loop_data=None, python_error_check=False,
                  caller_owns_ufunc_float_status_contract=True,
                  external_machine_calls=[], python_runtime_required=False)
    return report, code


def observe(*, strides=None):
    import numpy as np
    from numpy._core import _multiarray_umath as provider

    library = Path(provider.__file__).resolve(strict=True)
    report, _ = _derive(artifact_reference(library, 'scar.numpy-provider-elf'), _profiles(strides))
    _need(np.__version__ == report['numpy_version'] and provider._get_ndarray_c_version() == report['ndarray_c_abi']
          and provider.__cpu_features__['X86_V3'], 'original NumPy runtime or supported ISA differs')
    # GNU x86-64 fenv_t stores MXCSR after the 28-byte x87 environment.
    env = ctypes.create_string_buffer(32)
    lib = ctypes.CDLL('libm.so.6')
    _need(lib.fegetround() == 0 and lib.fegetenv(ctypes.byref(env)) == 0 and
          struct.unpack_from('<I', env.raw, 28)[0] & 0xe040 == 0, 'current floating-point mode differs')
    bases = []
    for line in Path('/proc/self/maps').read_text().splitlines():
        row = line.split(maxsplit=5)
        if len(row) == 6 and row[2] == '00000000' and row[5] == str(library):
            bases.append(int(row[0].split('-', 1)[0], 16))
    _need(len(bases) == 1, 'could not bind the observed NumPy ELF load bias')
    base, = bases
    pointer = ctypes.pythonapi.PyCapsule_GetPointer
    pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    pointer.restype = ctypes.c_void_p
    signature = (np.dtype('complex128'),) * 3
    adapter = report['adapter']
    for role in _ROOTS:
        op = getattr(np, role)
        _need(op is getattr(provider, role), 'selected source operation is not the stock NumPy ufunc')
        for row in report['strides'][role]:
            resolved, capsule = op._resolve_dtypes_and_context(signature)
            _need(resolved == signature, 'stock NumPy selected another dtype triple')
            op._get_strided_loop(capsule, fixed_strides=tuple(row))
            info = _Info.from_address(pointer(capsule, _CAPSULE))
            _need(info.auxdata and not info.requires_pyapi and info.loop == base + adapter['address'],
                  'live ufunc did not select the qualified legacy adapter')
            function = ctypes.c_void_p.from_address(info.auxdata + adapter['function_offset']).value
            data = ctypes.c_void_p.from_address(info.auxdata + adapter['loop_data_offset']).value
            status = ctypes.c_int.from_address(info.auxdata + adapter['python_error_check_offset']).value
            _need(function == base + report['machine_functions'][role]['address'] and data is None and status == 0,
                  'stock complex ufunc selects another target, data or Python error path')
    return report


def _assembly(report, bodies):
    def bytes_(raw):
        return ['.byte ' + ', '.join(f'0x{value:02x}' for value in raw[i:i+24])
                for i in range(0, len(raw), 24)]

    assembly, literals = [], {}
    for role, machine in report['machine_functions'].items():
        target = _NATIVE[role]
        binding = '.local' if role == 'pairwise_sum' else '.globl'
        assembly.extend([f'.pushsection .text.{target},"ax",@progbits', '.p2align 4',
                         f'{binding} {target}', f'.type {target}, @function', target + ':'])
        body = bodies[role]
        cursor = 0
        for row in machine['transfers']:
            offset, kind, value = (row[key] for key in ('offset', 'kind', 'target'))
            assembly.extend(bytes_(body[cursor:offset]))
            destination = _NATIVE[value] if kind == 'CALL' else literals.setdefault(value, f'.Lscar_complex_{len(literals)}')
            assembly.append(f'.long {destination} - . - 4')
            cursor = offset + 4
        assembly.extend(bytes_(body[cursor:]))
        assembly.extend([f'.size {target}, .-{target}', '.popsection'])
    if literals:
        assembly.append('.pushsection .rodata.scar_numpy_complex,"a",@progbits')
        for raw, label in literals.items():
            data = bytes.fromhex(raw)
            assembly.extend([f'.balign {len(data)}', label + ':', *bytes_(data)])
        assembly.append('.popsection')
    return '\n'.join(assembly)


_HEADER = '''#pragma once
#include <cstdint>
extern "C" bool scar_numpy_complex_runtime_supported() noexcept;
extern "C" void scar_numpy_complex_add(char**, const std::int64_t*, const std::int64_t*, void*);
extern "C" void scar_numpy_complex_multiply(char**, const std::int64_t*, const std::int64_t*, void*);
'''


def render(report):
    expected, functions = _derive(report['provider'], _profiles(report['strides']))
    _need(expected == report, 'stock provider receipt differs from the captured runtime selection')
    source = '''#include "numpy_complex.hpp"
#include <cfenv>
#include <xmmintrin.h>

extern "C" bool scar_numpy_complex_runtime_supported() noexcept {
    return __builtin_cpu_supports("x86-64-v3") && std::fegetround() == FE_TONEAREST
        && (_mm_getcsr() & 0xe040U) == 0;
}

asm(R"scar_cx(
''' + _assembly(report, functions) + '\n)scar_cx");\n'
    return {SOURCE: source.encode(), HEADER: _HEADER.encode(),
            LICENSE: read_artifact(report['notice'], json_document=False)}


def audit(executable, report):
    """Rejoin the emitted instructions and every linked direct call/literal."""
    image = _Image(artifact_reference(Path(executable).resolve(strict=True), 'scar.native-executable'))
    functions, _ = _closure(image, _NATIVE, original=False)
    for role, row in functions.items():
        original = report['machine_functions'][role]
        _need(all(row[key] == original[key] for key in ('extent', 'normalized_sha256', 'transfers')),
              f'linked {role} changes the stock instruction or relocation bytes')
    return dict(schema='scar.native-numpy-complex-linkage', schema_version=1,
                exact_machine_closure=True, functions=functions)


def main(argv=None):
    from scar.analysis.numpy_partition_iteration import check

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iteration', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    plan = check(args.iteration)
    signatures = {op: [row['strides'] for row in plan['loop_signatures'] if row['op'] == op.upper()]
                  for op in _ROOTS}
    report = observe(strides=signatures)
    _need(dict(report['provider'], schema=plan['provider']['schema']) == plan['provider'],
          'live NumPy provider differs from the iteration plan')
    with args.out.open('x', encoding='utf-8') as output:
        output.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'provider': report['provider'], 'strides': report['strides']}, sort_keys=True))


if __name__ == '__main__':
    main()
