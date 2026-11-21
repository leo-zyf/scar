"""Deliver scalar PCG64 uniform draws from audited upstream CPU routines."""

import argparse
import ctypes
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess

from scar.codegen.native_numpy_complex import _instructions, _relocations, _symbols, _unrelocated
from scar.codegen.native_project import build_native_project, write_native_project
from scar.codegen.stateful_numpy_provider import _Image
from scar.ir.execution_recipe import artifact_reference, read_artifact

SCHEMA = 'scar.native-numpy-pcg64-uniform'
SOURCE = 'src/numpy_random.cpp'
HEADER = 'include/scar_generated/numpy_random.hpp'
ARCHIVE = 'lib/libnpyrandom.a'
_TARGET = 'scar_numpy_pcg64_uniform'
_SEED_ORIGIN = 'https://gist.github.com/imneme/540829265469e673d045/5afb4439c23a6c060eda72121bff8bf9da59591a'
_SEED_SHA = '7ab4307eef0b949e8e372ab48b004db8bea425e1927b9a7fce148f61a38afb63'
_INPUTS = dict(seed='pcg64_set_seed', draw='__pyx_f_5numpy_6random_6_pcg64_pcg64_double')
_NATIVE = dict(seed='scar_stock_pcg64_set_seed', draw='scar_stock_pcg64_double')
_ALLOWED = frozenset('adc add cvtsi2sd imul mov movabs mul mulsd or pxor ret ror shr xor'.split())
_ABI = dict(seed='NONNEGATIVE_UINT64_WITH_DEFAULT_128_BIT_POOL_AND_EMPTY_SPAWN_KEY',
            state='FOUR_UINT64_LOW_HIGH_STATE_THEN_LOW_HIGH_INCREMENT',
            provider='NUMPY_PCG64_AND_LIBNPYRANDOM_WITH_UPSTREAM_SEED_SEQ_FE128',
            uniform='SCALAR_BINARY64_FINITE_BOUNDS_IN_ORIGINAL_CALL_ORDER', python_runtime_required=False)
_HEADER = '''#pragma once
#include <cstddef>
#include <cstdint>
extern "C" void scar_numpy_pcg64_init_v1(std::uint64_t seed, std::uint64_t* state,
                                        std::uint64_t* seed_words) noexcept;
extern "C" int scar_numpy_pcg64_uniform_v1(std::uint64_t* state, double low, double high,
                                          std::size_t count, double* out) noexcept;
'''


def _need(valid, detail):
    if not valid:
        raise ValueError('native NumPy random: ' + detail)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _body(image, symbol, *, literal):
    (start, size, _), = _symbols(image)[symbol]
    relocations = _relocations(image)
    _unrelocated(relocations, start, size)
    raw, references = bytearray(), []
    for address, code, opcode, argument in _instructions(image, start, size):
        _need(opcode in _ALLOWED and not re.search(r'\b(?:fs|gs):|\[(?:[a-z]+:)?0x', argument),
              'PCG64 routine has an unsupported external instruction')
        raw.extend(code)
        if '[rip' not in argument:
            continue
        operands, _, comment = argument.partition('#')
        destination = address + len(code) + struct.unpack_from('<i', code, len(code) - 4)[0]
        target = re.match(r'^\s*([0-9a-f]+)\b', comment)
        _need(opcode == 'mulsd' and 'QWORD PTR [rip' in operands and target is not None
              and int(target[1], 16) == destination and destination % 8 == 0, 'unsupported PCG64 literal reference')
        _unrelocated(relocations, destination, 8)
        references.append(dict(offset=len(raw) - 4, literal=image.read(destination, 8, 4).hex()))
        raw[-4:] = bytes(4)
    _need(len(references) == int(literal), 'stock routine literal count differs')
    return dict(symbol=symbol, address=start, extent=size, literals=references,
                normalized_sha256=sha256(raw).hexdigest()), bytes(raw)


def _stock_images(pcg64, generator, *, native=False):
    pcg_image, generator_image = _Image(pcg64), _Image(generator)
    names = _NATIVE if native else _INPUTS
    routines = {name: _body(pcg_image, symbol, literal=name == 'draw') for name, symbol in names.items()}
    (address, size, _), = _symbols(generator_image)['random_uniform']
    _unrelocated(_relocations(generator_image), address, size)
    code = _instructions(generator_image, address, size)
    calls = [argument for _, _, opcode, argument in code if opcode == 'call']
    _need(calls == ['QWORD PTR [rax+0x18]'], 'stock uniform dispatch must call only next_double')
    uniform = dict(extent=size, sha256=sha256(generator_image.read(address, size, 5)).hexdigest())
    return routines, uniform


def _state_words(bit_generator):
    native = bit_generator.state['state']
    mask = (1 << 64) - 1
    return [word for field in ('state', 'inc') for word in (native[field] & mask, native[field] >> 64)]


def observe(seed_source):
    import numpy as np
    from numpy.random import _generator, _pcg64

    directory = Path(_pcg64.__file__).resolve().parent
    pcg64, generator = (_ref(module.__file__, 'scar.numpy-random-extension') for module in (_pcg64, _generator))
    seed = _ref(seed_source, 'scar.upstream-mit-source')
    _need(seed['sha256'] == _SEED_SHA, 'upstream SeedSequence reference must match the pinned MIT source')
    routines, uniform = _stock_images(pcg64, generator)
    rng = np.random.default_rng(1)
    _need(type(rng.bit_generator) is _pcg64.PCG64, 'default_rng selects another stream')
    bitgen = rng.bit_generator.ctypes
    origin, = (int(row[0].split('-')[0], 16) for line in Path('/proc/self/maps').read_text().splitlines()
               if len(row := line.split(maxsplit=5)) == 6 and row[2] == '00000000' and row[5] == pcg64['uri'])
    callback = ctypes.cast(bitgen.next_double, ctypes.c_void_p).value
    storage = ctypes.c_void_p.from_address(bitgen.state_address).value
    _need(callback == origin + routines['draw'][0]['address'] and storage is not None
          and list((ctypes.c_uint64 * 4).from_address(storage)) == _state_words(rng.bit_generator),
          'live object or callback differs from the four-word stock state ABI')
    materials = dict(pcg64=pcg64, generator=generator, seed_source=seed,
                     uniform_archive=_ref(directory / 'lib/libnpyrandom.a', 'scar.numpy-random-static-library'),
                     bitgen_header=_ref(directory.parent / '_core/include/numpy/random/bitgen.h', 'scar.cpp-header'),
                     random_notice=_ref(directory / 'LICENSE.md', 'scar.upstream-license'),
                     numpy_notice=_ref(directory.parent.parent / f'numpy-{np.__version__}.dist-info/licenses/LICENSE.txt',
                                       'scar.upstream-license'))
    return dict(materials=materials, numpy_version=np.__version__, seed_source_revision=_SEED_ORIGIN,
                routines={name: row[0] for name, row in routines.items()}, uniform=uniform,
                live_pcg64_double_capsule_and_state_verified=True)


def _assembly(routines):
    def data(raw):
        return ['.byte ' + ', '.join(f'0x{value:02x}' for value in raw[i:i+24]) for i in range(0, len(raw), 24)]

    definition, constants = [], []
    for role, (info, raw) in routines.items():
        symbol = _NATIVE[role]
        definition.extend([f'.pushsection .text.{symbol},"ax",@progbits', '.p2align 4',
                           f'.local {symbol}', f'.type {symbol}, @function', symbol + ':'])
        cursor = 0
        for row in info['literals']:
            offset = row['offset']
            label = f'.Lscar_pcg64_literal_{len(constants)}'
            definition.extend([*data(raw[cursor:offset]), f'.long {label} - . - 4'])
            constants.append((label, row['literal']))
            cursor = offset + 4
        definition.extend([*data(raw[cursor:]), f'.size {symbol}, .-{symbol}', '.popsection'])
    definition.append('.pushsection .rodata.scar_random,"a",@progbits')
    for label, raw in constants:
        definition.extend(['.balign 8', label + ':', *data(bytes.fromhex(raw))])
    return '\n'.join([*definition, '.popsection'])


def render(provider):
    sources = provider['materials']
    routines, uniform = _stock_images(sources['pcg64'], sources['generator'])
    _need(provider['routines'] == {name: row[0] for name, row in routines.items()}
          and provider['uniform'] == uniform and provider['live_pcg64_double_capsule_and_state_verified'] is True
          and provider['seed_source_revision'] == _SEED_ORIGIN and sources['seed_source']['sha256'] == _SEED_SHA,
          'stock seeded stream receipt differs')
    cpp = Path(__file__).with_suffix('.cpp.in').read_text().replace('@SCAR_STOCK_ROUTINES@', _assembly(routines))
    return {SOURCE: cpp.encode(), HEADER: _HEADER.encode(), **{
        path: read_artifact(sources[key], json_document=False) for path, key in (
            ('include/scar_generated/randutils.hpp', 'seed_source'), ('include/scar_generated/bitgen.h', 'bitgen_header'),
            (ARCHIVE, 'uniform_archive'), ('licenses/NumPy-LICENSE.txt', 'numpy_notice'),
            ('licenses/NumPy-Random-LICENSE.md', 'random_notice'))}}


_DRIVER = r'''#include "numpy_random.hpp"
#include <array>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <system_error>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 6) return 1;
    std::uint64_t seed = 0;
    const auto tail = argv[1] + std::char_traits<char>::length(argv[1]);
    const auto value = std::from_chars(argv[1], tail, seed);
    if (value.ec != std::errc{} || value.ptr != tail) return 2;
    const double low = std::strtod(argv[2], nullptr), high = std::strtod(argv[3], nullptr);
    const auto count = std::stoull(argv[4]);
    if (count < 2 || count > 1000000) return 3;
    std::array<std::uint64_t, 4> state{}, expanded{};
    scar_numpy_pcg64_init_v1(seed, state.data(), expanded.data());
    const auto initial = state;
    std::vector<double> output(count);
    if (scar_numpy_pcg64_uniform_v1(state.data(), low, high, 1, output.data()) ||
        scar_numpy_pcg64_uniform_v1(state.data(), low, high, count - 1, output.data() + 1)) return 4;
    std::ofstream file(argv[5], std::ios::binary);
    if (!file.write(reinterpret_cast<const char*>(expanded.data()), sizeof(expanded)) ||
        !file.write(reinterpret_cast<const char*>(initial.data()), sizeof(initial)) ||
        !file.write(reinterpret_cast<const char*>(state.data()), sizeof(state)) ||
        !file.write(reinterpret_cast<const char*>(output.data()), count * sizeof(double))) return 5;
}
'''


def _audit(binary, provider):
    reference = _ref(binary, 'scar.native-executable')
    routines, uniform = _stock_images(reference, reference, native=True)
    for name, (row, _) in routines.items():
        original = provider['routines'][name]
        _need(all(row[key] == original[key] for key in ('extent', 'literals', 'normalized_sha256')),
              'linked PCG64 differs from stock ' + name)
    _need(provider['uniform'] == uniform, 'linked distribution is not the stock NumPy uniform implementation')


def _values(seed, low, high, count):
    import numpy as np

    rng = np.random.default_rng(seed)
    expanded = np.random.SeedSequence(seed).generate_state(4, dtype=np.uint64).tolist()
    initial = _state_words(rng.bit_generator)
    draws = [rng.uniform(low, high) for _ in range(count)]
    return struct.pack('<12Q', *expanded, *initial, *_state_words(rng.bit_generator)) + struct.pack(f'<{count}d', *draws)


def export(seed_source, out, *, seeds, low, high, count):
    _need(len(set(seeds)) == len(seeds) and len(seeds) >= 3 and all(0 <= seed < 2**64 for seed in seeds)
          and type(count) is int and 2 <= count <= 1_000_000, 'expected distinct uint64 test seeds and bounded draw count')
    provider = observe(seed_source)
    root = Path(out).resolve()
    root.mkdir()
    members = render(provider)
    members['src/main.cpp'] = _DRIVER.encode()
    for path, contents in members.items():
        member = root / path
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_bytes(contents)
    write_native_project(root, _TARGET, sources=(SOURCE, 'src/main.cpp'), libraries=(str(root / ARCHIVE), 'm'),
                         compile_options=('-fno-fast-math', '-ffp-contract=off'),
                         install_dirs=(('licenses', '.'),))
    binary, _ = build_native_project(root, _TARGET, (shutil.which('g++'),))
    _audit(binary, provider)
    evidence = root / 'validation'
    samples = []
    for seed in seeds:
        original = evidence / f'seed-{seed}-original.bin'
        native = evidence / f'seed-{seed}-native.bin'
        original.write_bytes(_values(seed, low, high, count))
        command = [str(binary), str(seed), low.hex(), high.hex(), str(count), str(native)]
        result = subprocess.run(command, env={}, cwd=root, check=True, capture_output=True, timeout=10)
        _need(not result.stdout and not result.stderr and native.read_bytes() == original.read_bytes(),
              'native seed words, states or original scalar draws differ at seed ' + str(seed))
        samples.append(dict(seed=seed, argv=command, original=_ref(original, 'scar.numpy-scalar-random-witness'),
                            native=_ref(native, 'scar.numpy-scalar-random-witness')))
    receipt = dict(schema=SCHEMA, schema_version=1, abi=_ABI, provider=provider, samples=samples,
                   bounds=[low.hex(), high.hex()], draws_per_seed=count, binary=_ref(binary, 'scar.native-executable'),
                   members={path: _ref(root / path, 'scar.native-project-member') for path in members},
                   deployment_authorized=False, gate_coefficient_provider_proven=False)
    (root / 'result.json').write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return receipt


def check(path):
    receipt = read_artifact(_ref(path, SCHEMA))
    _need(receipt['schema_version'] == 1 and receipt['abi'] == _ABI and receipt['deployment_authorized'] is False
          and receipt['gate_coefficient_provider_proven'] is False, 'unexpected native stream contract')
    code = render(receipt['provider'])
    code['src/main.cpp'] = _DRIVER.encode()
    _need(code == {name: read_artifact(ref, json_document=False) for name, ref in receipt['members'].items()},
          'native stream source or library archive changed')
    read_artifact(receipt['binary'], json_document=False)
    _audit(receipt['binary']['uri'], receipt['provider'])
    low, high = (float.fromhex(bound) for bound in receipt['bounds'])
    for row in receipt['samples']:
        expected = _values(row['seed'], low, high, receipt['draws_per_seed'])
        _need(expected == read_artifact(row['original'], json_document=False) ==
              read_artifact(row['native'], json_document=False), 'original scalar uniform witness differs')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--seed-source', type=Path)
    modes.add_argument('--check', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--seed', type=int, action='append', default=[])
    parser.add_argument('--low', type=float, required=False)
    parser.add_argument('--high', type=float, required=False)
    parser.add_argument('--count', type=int, default=154)
    args = parser.parse_args(argv)
    if args.check:
        result = check(args.check)
    elif args.out and args.low is not None and args.high is not None:
        result = export(args.seed_source, args.out, seeds=args.seed, low=args.low, high=args.high, count=args.count)
    else:
        parser.error('--seed-source requires --out, --seed, --low and --high')
    print(json.dumps(dict(seeds=[row['seed'] for row in result['samples']], draws=result['draws_per_seed'])))


if __name__ == '__main__':
    main()
