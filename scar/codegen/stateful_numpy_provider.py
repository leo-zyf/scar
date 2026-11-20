"""Relocate the qualified NumPy legacy double-power routine into a native entry."""

import ctypes
from hashlib import sha256
from pathlib import Path
import re
import struct
import subprocess

from scar.codegen import stateful
from scar.compiler.stateful_model import POWER2_SEMANTICS
from scar.ir.execution_recipe import artifact_reference, read_artifact


SOURCE = 'src/stateful_numpy_power.cpp'
LICENSE = 'licenses/NumPy-LICENSE.txt'
_PIN = '27bd88fd60f4b9fcd5c938572e9b9e3df0aed9cfed4888e57d94c516b45512b7'
_ORIGINAL = 'DOUBLE_power'
_FUNCTION = 'scar_numpy_legacy_double_power'
_ADAPTER = 'generic_wrapped_legacy_loop'
_CALLS = {'pow', 'sqrt'}
_TARGET = re.compile(r'([0-9a-f]+) <([^>]+)>')
_INSTRUCTION = re.compile(r'^\s*([0-9a-f]+):\t((?:[0-9a-f]{2} )+)\s*(\S+)\s*(.*)$')


def _need(condition, detail):
    if not condition:
        raise ValueError('NATIVE_PROVIDER_FRONTIER: ' + detail)


def _tool(command, *args):
    return subprocess.check_output(['/usr/bin/' + command, *args], env={'LC_ALL': 'C'}, text=True, timeout=10)


class _Image:
    def __init__(self, ref):
        self.path = ref['uri']
        self.raw = read_artifact(ref, json_document=False)
        raw = self.raw
        _need(raw.startswith(b'\x7fELF\x02\x01\x01') and len(raw) >= 64
              and struct.unpack_from('<HH', raw, 16) == (3, 62),
              'requires an ordinary ELF64 x86_64 shared image or PIE')
        table = struct.unpack_from('<Q', raw, 32)[0]
        size, count = struct.unpack_from('<HH', raw, 54)
        _need(size == 56 and 0 < count < 64 and table + size * count <= len(raw), 'unsupported ELF headers')
        self.loads = []
        for index in range(count):
            kind, flags, offset, addr, _, length, _, _ = struct.unpack_from('<II6Q', raw, table + size * index)
            if kind == 1:
                _need(offset + length <= len(raw), 'invalid ELF load range')
                self.loads.append((addr, offset, length, flags))

    def read(self, address, length, flags):
        positions = [offset + address - begin for begin, offset, size, access in self.loads
                     if access == flags and begin <= address and address + length <= begin + size]
        _need(len(positions) == 1, 'address is outside the required ELF load range')
        position, = positions
        return self.raw[position:position + length]

    def instructions(self, symbol):
        symbols = _tool('readelf', '-Ws', self.path)
        entries = {(int(row[1], 16), int(row[2])) for row in (
            item.split() for item in symbols.splitlines())
            if len(row) == 8 and row[3] == 'FUNC' and row[7] == symbol and row[6].isdecimal()}
        _need(len(entries) == 1, f'{symbol} must have one exact function extent')
        (address, extent), = entries
        listing = _tool('objdump', '-dw', '-Mintel', '--disassemble=' + symbol, self.path)
        rows = [(int(match[1], 16), bytes.fromhex(match[2]), match[3], match[4])
                for row in listing.splitlines() if (match := _INSTRUCTION.fullmatch(row))]
        cursor = address
        for position, data, _, _ in rows:
            _need(position == cursor, f'{symbol} disassembly leaves an instruction gap')
            cursor += len(data)
        _need(0 < extent <= 4096 and cursor == address + extent
              and b''.join(row[1] for row in rows) == self.read(address, extent, 5),
              f'{symbol} disassembly differs from executable ELF bytes')
        return address, rows

    def imports(self, names):
        listing = _tool('readelf', '-W', '--dyn-syms', self.path)
        imports = {}
        for row in (line.split() for line in listing.splitlines()):
            if len(row) < 8 or row[6] != 'UND':
                continue
            name, marker, version = row[7].partition('@')
            if name in names:
                _need(marker and re.fullmatch(r'GLIBC_[0-9.]+', version) and name not in imports,
                      'ambiguous or unsupported original libm symbol version')
                imports[name] = version
        _need(set(imports) == names, 'incomplete versioned libm interface')
        return imports


def _decode(image, symbol):
    address, rows = image.instructions(symbol)
    entries = {item[0] for item in rows}
    normalized, relocations = bytearray(), []
    for position, data, op, argument in rows:
        offset = position - address
        normalized.extend(data)
        if op == 'call':
            target = _TARGET.fullmatch(argument)
            _need(target is not None and data[:1] == b'\xe8' and len(data) == 5,
                  'unsupported call encoding in legacy power loop')
            name = target[2].removesuffix('@plt')
            _need(target[2] == name + '@plt' and name in _CALLS, 'unqualified external call in legacy power loop')
            _need(position + len(data) + struct.unpack_from('<i', data, 1)[0] == int(target[1], 16),
                  'external call displacement differs from its disassembly')
            relocations.append((offset + 1, 'call', name))
            normalized[-4:] = bytes(4)
        elif op.startswith('j'):
            target = _TARGET.fullmatch(argument)
            _need(target is not None and int(target[1], 16) in entries, 'nonlocal branch in legacy power loop')
        elif '[rip' in argument.lower():
            operands, separator, annotation = argument.partition('#')
            target = _TARGET.fullmatch(annotation.strip())
            _need(separator and target is not None and 'QWORD PTR [rip' in operands
                  and op in ('movsd', 'ucomisd') and len(data) >= 6,
                  'unsupported external literal load')
            destination = position + len(data) + struct.unpack_from('<i', data, len(data) - 4)[0]
            _need(destination == int(target[1], 16), 'literal displacement differs from its disassembly')
            relocations.append((offset + len(data) - 4, 'literal', image.read(destination, 8, 4).hex()))
            normalized[-4:] = bytes(4)
        else:
            _need(op in ('push', 'pop', 'mov', 'movsd', 'movapd', 'add', 'sub', 'test', 'cmp',
                         'xor', 'pxor', 'ucomisd', 'divsd', 'sqrtsd', 'mulsd', 'xchg', 'nop',
                         'data16', 'cs', 'ret'), 'unrecognized original instruction: ' + op)
    _need({value for _, kind, value in relocations if kind == 'call'} == _CALLS, 'incomplete original libm relocations')
    return address, bytes(normalized), relocations


def _layout(image):
    address, rows = image.instructions(_ADAPTER)
    call = [re.fullmatch(r'QWORD PTR \[r8\+(0x[0-9a-f]+)\]', arg)
            for _, _, op, arg in rows if op == 'call' and arg.startswith('QWORD')]
    data = [re.fullmatch(r'rcx,QWORD PTR \[r8\+(0x[0-9a-f]+)\]', arg)
            for _, _, op, arg in rows if op == 'mov' and arg.startswith('rcx,QWORD')]
    status = [re.fullmatch(r'eax,DWORD PTR \[rbx\+(0x[0-9a-f]+)\]', arg)
              for _, _, op, arg in rows if op == 'mov' and arg.startswith('eax,DWORD')]
    _need(len(call) == len(data) == len(status) == 1 and all((*call, *data, *status))
          and sum(op == 'mov' and arg == 'rbx,r8' for _, _, op, arg in rows) == 1,
          'legacy adapter needs recognizable function, context and error check fields')
    return address, int(call[0][1], 16), int(data[0][1], 16), int(status[0][1], 16)


def _observe(reference, address, adapter, function_offset, data_offset, status_offset):
    import numpy as np
    from numpy._core import _multiarray_umath as module

    _need(Path(module.__file__).resolve() == Path(reference['uri']) and np.__version__ == '2.5.2'
          and module.power is np.power and module.__cpu_features__['AVX2']
          and not module.__cpu_features__['AVX512F'], 'original NumPy power provider or CPU dispatch differs')
    with open('/proc/self/maps') as stream:
        mappings = [int(fields[0].partition('-')[0], 16) for row in stream
                    if len(fields := row.split(maxsplit=5)) == 6 and fields[2] == '00000000'
                    and fields[5].strip() == reference['uri']]
    _need(len(mappings) == 1, 'failed to acquire the original NumPy ELF mapping')
    base, = mappings
    dtype = np.dtype('float64')
    pointer = ctypes.pythonapi.PyCapsule_GetPointer
    pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    pointer.restype = ctypes.c_void_p
    for signature in ((dtype, int, None), (dtype, dtype, dtype)):
        resolved, capsule = np.power._resolve_dtypes_and_context(signature)
        _need(resolved == (dtype,) * 3, 'original dtype signature differs')
        np.power._get_strided_loop(capsule, fixed_strides=(8, 0, 8))
        info = stateful._Info.from_address(pointer(capsule, stateful._CAPSULE))
        _need(not info.requires_pyapi and info.loop == base + adapter and info.auxdata,
              'original selected adapter or Python API requirement differs')
        _need(ctypes.c_void_p.from_address(info.auxdata + function_offset).value == base + address
              and ctypes.c_void_p.from_address(info.auxdata + data_offset).value is None
              and ctypes.c_int.from_address(info.auxdata + status_offset).value == 0,
              'original NumPy selected a different function, loop data or Python error check')


def prepare(region, metadata, *, observe=False):
    """Check the compiler provider port and reconstruct the exact relocated source."""
    providers = [(row.module, row.symbol, row.semantics) for row in region.providers]
    _need(providers == [('numpy', 'power', POWER2_SEMANTICS)], 'unsupported provider signature')
    runtime = metadata['provider_runtime']
    binary = runtime['loaded_binary']
    _need(runtime['numpy_version'] == '2.5.2' and runtime['ndarray_c_abi'] == 33554432
          and runtime['requires_pyapi'] is False and runtime['fixed_strides'] == [8, 0, 8]
          and runtime['dtype_signature'] == ['float64'] * 3 and binary['sha256'] == _PIN,
          'unsupported NumPy binary or runtime signature')
    reference = artifact_reference(binary['path'], 'scar.numpy-provider-elf')
    _need((reference['sha256'], reference['size_bytes']) == (_PIN, binary['bytes']), 'original NumPy ELF differs')
    image = _Image(reference)
    address, raw, relocations = _decode(image, _ORIGINAL)
    imports = image.imports(_CALLS)
    adapter, function_offset, data_offset, status_offset = _layout(image)
    if observe:
        _observe(reference, address, adapter, function_offset, data_offset, status_offset)
    license = Path(reference['uri']).parents[2] / 'numpy-2.5.2.dist-info/licenses/LICENSE.txt'
    notice = artifact_reference(license, 'scar.upstream-license')
    report = dict(provider=reference, upstream_notice=notice, symbol=_ORIGINAL, original_address=address,
                  original_extent=len(raw), normalized_machine_sha256=sha256(raw).hexdigest(),
                  relocations=[list(row) for row in relocations], native_symbol=_FUNCTION,
                  original_libm_versions=imports, legacy_adapter=dict(address=adapter, function_offset=function_offset,
                      loop_data_offset=data_offset, python_error_check_offset=status_offset),
                  source_selection='EXACT_LEGACY_CAPSULE_MATCHED_AT_EXPORT',
                  host='LINUX_X86_64_AVX2_WITHOUT_AVX512F',
                  dtype='FLOAT64', exponent=2, strides=[8, 0, 8], math_reimplemented=False)
    return report, _source(raw, relocations, imports), read_artifact(notice, json_document=False)


def _source(raw, relocations, imports):
    def emit(data):
        return '.byte ' + ', '.join(f'0x{value:02x}' for value in data)

    lines = [f'.symver scar_numpy_{name},{name}@{version}' for name, version in sorted(imports.items())]
    lines.extend([f'.pushsection .text.{_FUNCTION},"ax",@progbits', f'.globl {_FUNCTION}',
                  f'.hidden {_FUNCTION}', f'.type {_FUNCTION}, @function', f'{_FUNCTION}:'])
    cursor = 0
    literals = {}
    for offset, kind, value in relocations:
        stop = offset - (kind == 'call')
        if stop > cursor:
            lines.append(emit(raw[cursor:stop]))
        if kind == 'call':
            lines.append(f'call scar_numpy_{value}@PLT')
        else:
            label = literals.setdefault(value, f'.Lscar_numpy_literal_{len(literals)}')
            lines.append(f'.long {label} - . - 4')
        cursor = offset + 4
    if cursor < len(raw):
        lines.append(emit(raw[cursor:]))
    lines.extend([f'.size {_FUNCTION}, .-{_FUNCTION}', '.popsection',
                  '.pushsection .rodata.scar_numpy_provider,"a",@progbits', '.p2align 3'])
    for value, label in literals.items():
        lines.extend([label + ':', emit(bytes.fromhex(value))])
    lines.append('.popsection')
    template = Path(__file__).with_name('stateful_numpy_provider_v1.cpp.in').read_text()
    return template.replace('@ORIGINAL_MACHINE_CODE@', '\n'.join(lines)).encode()


def audit(executable, report):
    """Check the complete linked function body and every relocated target."""
    image = _Image(artifact_reference(executable, 'scar.native-executable'))
    _, normalized, relocations = _decode(image, report['native_symbol'])
    _need(len(normalized) == report['original_extent']
          and sha256(normalized).hexdigest() == report['normalized_machine_sha256']
          and [list(row) for row in relocations] == report['relocations']
          and image.imports(_CALLS) == report['original_libm_versions'],
          'linked machine instructions, constants or libm transfers differ from original')
