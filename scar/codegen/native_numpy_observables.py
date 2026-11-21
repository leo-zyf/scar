"""Lower an ordered complex128 norm and half-split reduction to exact stock BLAS."""

import argparse
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen import native_numpy_vector_normalize as normalize
from scar.codegen.native_cpu_elf import cpu_executable_dynamic
from scar.codegen.native_project import build_native_project, write_native_project
from scar.codegen.native_provider_bundle import SCHEMA as BUNDLE_SCHEMA, validate_native_provider_bundle
from scar.ir.execution_recipe import artifact_reference, read_artifact

SCHEMA = 'scar.native-numpy-observables'
WITNESS = 'scar.numpy-split-observable-witness'
SOURCE = 'src/numpy_observables.cpp'
HEADER = 'include/scar_generated/numpy_observables.hpp'
_TARGET = 'scar_numpy_observables'
_BINARY = f'bin/{_TARGET}'
_DRIVER = 'src/probe.cpp'
_MAX_BYTES = 64 << 20
_BLAS = 'scipy_cblas_zdotc_sub64_'
_OUTPUTS = ('squared_norm', 'split_difference', 'norm_error')
_RPATH = '$ORIGIN/../provider/objects'
_HEADER = '''#pragma once
#include <cstddef>

// Read a positive-length contiguous native complex128 vector and publish both values on success.
// Return -1 for an incompatible environment/extent, 1 for new floating-point exceptions, or 0.
// On decline neither output is modified. The caller supplies disjoint, fully accessible spans.
extern "C" int scar_numpy_split_observable_v1(std::size_t n, const void* data, double* squared_norm,
                                             double* split_difference) noexcept;
extern "C" double scar_numpy_norm_error_v1(double squared_norm) noexcept;
'''
_ABI = dict(standard='C++20', version=1, symbol='scar_numpy_split_observable_v1',
            norm_error_symbol='scar_numpy_norm_error_v1', input='POSITIVE_LENGTH_CONTIGUOUS_NATIVE_COMPLEX128',
            order=['SELF_VDOT_FULL', 'SELF_VDOT_PREFIX_FLOOR_HALF', 'SELF_VDOT_SUFFIX', 'SUBTRACT_REAL_VALUES'],
            scope='PRIVATE_OUTPUTS_UNTIL_NORMAL_COMPLETION', restores_caller_fenv=True,
            caller_owns_original_floating_error_handling=True, runtime_provider_qualified_by_caller=True,
            workspace_bytes=0, python_runtime_required=False)
_PROBE = '''#include "numpy_observables.hpp"
#include <array>
#include <charconv>
#include <fstream>
#include <iostream>
#include <string_view>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    std::size_t n{};
    const std::string_view arg(argv[1]);
    const auto [end, status] = std::from_chars(arg.data(), arg.data() + arg.size(), n);
    if (status != std::errc{} || end != arg.data() + arg.size() || !n || n > (64U << 20) / 16) return 2;
    std::vector<double> input(n * 2);
    std::ifstream file(argv[2], std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(input.data()), n * 16) || file.peek() != EOF) return 2;
    std::array<double, 3> results{};
    if (scar_numpy_split_observable_v1(n, input.data(), &results[0], &results[1])) return 3;
    results[2] = scar_numpy_norm_error_v1(results[0]);
    const auto* bytes = reinterpret_cast<const char*>(results.data());
    return std::cout.write(bytes, results.size() * sizeof(double)) ? 0 : 4;
}
'''


def _need(valid, detail):
    if not valid:
        raise ValueError('native NumPy split observable: ' + detail)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _vdot_dispatch(provider):
    core = provider['blas_core']
    _need(type(core) is str and re.fullmatch('[A-Za-z0-9_]+', core) and provider['blas_threads'] == 1,
          'expected qualified stock single-thread BLAS domain')
    image, _, _ = normalize._source_dispatch(provider['numpy'], provider['blas'])
    address, rows = image.instructions('CDOUBLE_vdot')
    calls = [index for index, (_, _, op, args) in enumerate(rows) if op == 'call' and f'<{_BLAS}@plt>' in args]
    _need(len(calls) == 1, 'stock contiguous complex vdot must call one BLAS operation')
    index, = calls
    selected = [(op, args) for _, _, op, args in rows]
    _need(selected[index - 6:index] == [
        ('lea', 'r9,[rsp+0x10]'), ('movaps', 'XMMWORD PTR [rsp],xmm3'), ('mov', 'r8,rcx'),
        ('mov', 'rcx,rdx'), ('mov', 'rdx,rsi'), ('mov', 'rsi,rax')
    ] and selected[index + 1:index + 4] == [
        ('movapd', 'xmm3,XMMWORD PTR [rsp]'), ('addpd', 'xmm3,XMMWORD PTR [rsp+0x10]'),
        ('movups', 'XMMWORD PTR [rbx],xmm3')
    ] and ('pxor', 'xmm3,xmm3') in selected[index - 12:index - 6] and
          ('sar', 'rsi,0x4') in selected and ('sar', 'rcx,0x4') in selected,
          'stock BLAS vector strides, result ABI or complex accumulator changed')
    return dict(symbol='CDOUBLE_vdot', source_address=address, blas_symbol=_BLAS,
                abi='ILP64_COMPLEX128_HIDDEN_OUTPUT_POINTER', contiguous_strides_bytes=[16, 16],
                positive_count_calls=1, accumulator='BINARY64_ADDPD_TO_POSITIVE_ZERO_BOTH_LANES',
                empty_count='POSITIVE_ZERO_BOTH_LANES')


def render(provider):
    """Render detached sources using the existing checked normalization provider dictionary."""
    _vdot_dispatch(provider)
    source = Path(__file__).with_suffix('.cpp.in').read_text().replace('@SCAR_BLAS_CORE@', provider['blas_core'])
    return {SOURCE: source.encode(), HEADER: _HEADER.encode()}


def _field(document, path):
    for member in path:
        document = document[member]
    return document


def _witness(reference, provider):
    data = read_artifact(reference)
    _need(data['schema'] == WITNESS and data['schema_version'] == 1 and data['observed_only'] is True and
          data['numpy'] == provider['numpy'] and 2 <= len(data['samples']) <= 32 and
          {row['role'] for row in data['samples']} == {'RETAINED_ORIGINAL', 'INDEPENDENT_ORIGINAL'},
          'exact source provider and retained/independent observations required')
    for key in ('entry', 'producer'):
        _need(data[key]['schema'] == 'scar.python-source', 'original witness source is unresolved')
        read_artifact(data[key], json_document=False)
    seen = set()
    total_bytes = 0
    for row in data['samples']:
        name, length, input_ref = (row[key] for key in ('id', 'n', 'input'))
        _need(type(name) is str and re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', name) and name not in seen and
              type(length) is int and 0 < length <= _MAX_BYTES // 16 and input_ref['size_bytes'] == length * 16,
              'expected named original complex128 storage extent')
        seen.add(name)
        total_bytes += length * 16
        _need(total_bytes <= _MAX_BYTES and len(read_artifact(input_ref, json_document=False)) == length * 16,
              'original vector exceeds evidence scope')
        if row['role'] == 'RETAINED_ORIGINAL':
            antecedent = read_artifact(row['capture'])
            last = read_artifact(row['trajectory'])['records'][-1]['output']
            _need(antecedent['numpy_extension'] == provider['numpy'] and
                  any(item['output'] == input_ref for item in antecedent['samples']) and
                  last['sha256'] == input_ref['sha256'] and last['dtype'] == '<c16' and
                  last['shape'] == [length] and last['order'] == 'C', 'retained bytes no longer bind to original terminal state')
        else:
            _need(type(row['argv']) is list and all(type(value) is str for value in row['argv']),
                  'independent original invocation is unrecorded')
        source = row['stdout']
        output = read_artifact(source['uri'])
        _need(set(row['expected']) == set(source['paths']) == set(_OUTPUTS),
              'terminal observable output port set differs')
        for key in _OUTPUTS:
            observed = _field(output, source['paths'][key])
            _need(type(observed) is float and observed.hex() == row['expected'][key],
                  'expected values differ at original public output port ' + key)
    return data


def _project(root, library):
    name = library.name
    _need(re.fullmatch(r'[A-Za-z0-9_+.-]+\.so[.0-9]*', name), 'expected a bundled OpenBLAS ELF name')
    write_native_project(root, _TARGET, sources=(SOURCE, _DRIVER),
                         compile_options=('-fno-fast-math', '-ffp-contract=off'), libraries=('-l:' + name, 'm'),
                         runtime_dirs=(_RPATH,), install_dirs=(('provider', '.'), ('licenses', '.')))
    with (root / 'CMakeLists.txt').open('a') as stream:
        stream.write(f'target_link_directories({_TARGET} PRIVATE "${{CMAKE_CURRENT_SOURCE_DIR}}/provider/objects")\n')


def _prepare(normalization, witness, cpu):
    _need(type(cpu) is int and cpu >= 0, 'concrete CPU affinity is required for source parity')
    original = normalize.check(normalization['uri'])
    _need(original == read_artifact(normalization), 'normalization authority changed during replay')
    provider = original['provider']
    files = render(provider)
    files[_DRIVER] = _PROBE.encode()
    notice = normalize.LICENSE
    files[notice] = read_artifact(original['members'][notice], json_document=False)
    return original, _witness(witness, provider), files


def _one(binary, sample, cpu):
    argv = ['/usr/bin/taskset', '-c', str(cpu), str(binary), str(sample['n']), sample['input']['uri']]
    run = subprocess.run(argv, env={'OPENBLAS_NUM_THREADS': '1'}, capture_output=True, timeout=15)
    expected = struct.pack('<3d', *(float.fromhex(sample['expected'][key]) for key in _OUTPUTS))
    _need(run.returncode == 0 and not run.stderr and run.stdout == expected,
          'native binary64 output differs at ' + sample['id'])
    return dict(id=sample['id'], role=sample['role'], elements=sample['n'], exact_binary64=run.stdout.hex())


def _report(root, normalization, witness, original, observed, files, cpu):
    old_bundle = read_artifact(original['blas_bundle'])
    provider = original['provider']
    bundle = _ref(root / 'provider' / 'bundle.json', BUNDLE_SCHEMA)
    _need(validate_native_provider_bundle(bundle) == old_bundle, 'package changed the original qualified BLAS snapshot')
    binary = root / _BINARY
    linkage = cpu_executable_dynamic(binary)
    library_name = Path(old_bundle['root']).name
    _need(linkage['runpath'] == _RPATH and linkage['rpath'] is None and library_name in linkage['needed'] and
          set(linkage['needed']) <= {library_name, 'libc.so.6', 'libm.so.6', 'libstdc++.so.6', 'libgcc_s.so.1'},
          'native oracle contains a different linked provider or runtime search path')
    return dict(schema=SCHEMA, schema_version=1, abi=_ABI, witness=witness, normalization=normalization,
                provider=provider, blas_bundle=bundle, dispatch=_vdot_dispatch(provider), linkage=linkage,
                members={key: _ref(root / key, 'scar.native-source-member') for key in sorted({*files, 'CMakeLists.txt'})},
                binary=_ref(binary, 'scar.native-executable'), cpu_ids=[cpu],
                validation=[_one(binary, sample, cpu) for sample in observed['samples']],
                input_numerical_policy='EXACT_ORIGINAL_NORMAL_COMPLETION_ONLY', deployment_authorized=False)


def export(witness_path, normalization_path, out, *, cpu):
    """Build and run the detached BLAS observable oracle using original terminal vectors."""
    witness, normalization = _ref(witness_path, WITNESS), _ref(normalization_path, normalize.SCHEMA)
    original, observed, files = _prepare(normalization, witness, cpu)
    root = Path(out).absolute()
    _need(not root.exists() and not root.is_symlink(), 'fresh native package path required')
    root.parent.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which('g++')
    _need(compiler is not None, 'C++20 compiler unavailable')
    with TemporaryDirectory(prefix='.scar-observables-', dir=root.parent) as temporary:
        stage = Path(temporary) / 'package'
        stage.mkdir()
        for name, raw in files.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        source = original['blas_bundle']
        bundle = read_artifact(source)
        shutil.copytree(Path(source['uri']).parent, stage / 'provider')
        library = stage / 'provider' / bundle['root']
        _project(stage, library)
        build_native_project(stage, _TARGET, (compiler,), expected_libraries=(library,))
        result = _report(stage, normalization, witness, original, observed, files, cpu)
        for ref in (result['binary'], result['blas_bundle'], *result['members'].values()):
            ref['uri'] = str(root / Path(ref['uri']).relative_to(stage))
        (stage / 'result.json').write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
        stage.rename(root)
    return result


def check(path):
    """Recheck the stock linked authority and reproduce every original binary64 observable."""
    path = Path(path).resolve(strict=True)
    recorded = read_artifact(_ref(path, SCHEMA))
    normalization, witness = recorded['normalization'], recorded['witness']
    cpu, = recorded['cpu_ids']
    original, samples, files = _prepare(normalization, witness, cpu)
    root = path.parent
    _need(all((root / name).read_bytes() == contents for name, contents in files.items()),
          'generated observable sources or upstream notices changed')
    library = root / 'provider' / read_artifact(original['blas_bundle'])['root']
    with TemporaryDirectory(prefix='scar-observables-cmake-') as directory:
        _project(Path(directory), library)
        _need((root / 'CMakeLists.txt').read_bytes() == (Path(directory) / 'CMakeLists.txt').read_bytes(),
              'CMake inputs changed')
    _need(recorded == _report(root, normalization, witness, original, samples, files, cpu),
          'native observable authority differs on replay')
    return recorded


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('witness', 'normalization', 'out', 'check'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--cpu', type=int)
    args = parser.parse_args(argv)
    if args.check and not any((args.witness, args.normalization, args.out, args.cpu is not None)):
        result = check(args.check)
    elif args.witness and args.normalization and args.out and args.cpu is not None and not args.check:
        result = export(args.witness, args.normalization, args.out, cpu=args.cpu)
    else:
        parser.error('provide --check RECEIPT or --witness WITNESS --normalization PACKAGE --out DIRECTORY --cpu ID')
    print(json.dumps(dict(binary=result['binary'], matched=result['validation']), sort_keys=True))


if __name__ == '__main__':
    main()
