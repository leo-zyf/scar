"""Qualify an exact ndarray complex scalar in-place operation for C++20 delivery."""

import argparse
import ast
import dis
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
from types import CodeType

from scar.codegen import native_numpy_complex as stock
from scar.codegen.native_numpy_partition import _audit
from scar.codegen.native_project import build_native_project, write_native_project
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.ids import CodeID

SCHEMA = 'scar.native-numpy-phase'
OBSERVATION = 'scar.numpy-phase-observed'
SOURCE = 'src/numpy_phase.cpp'
HEADER = 'include/scar_generated/numpy_phase.hpp'
PROBE = 'src/probe.cpp'
_TARGET = 'scar_native_numpy_phase'
_BINARY = 'bin/' + _TARGET
_TEMPLATE = Path(__file__).with_suffix('.cpp.in')
_MAX_BYTES = 64 << 20
_MAX_ELEMENTS = 1 << 20
_MAX_TOTAL_ELEMENTS = 2 << 20
_MAX_CASES = 128
_MOD = 64

_HEADER = '''#pragma once
#include <cstddef>
// Borrow an exact checked exclusive complex128 view and a separate scalar.
// Scratch is private, disjoint and 64-byte aligned. A return of 1 leaves the view intact.
extern "C" std::size_t scar_numpy_phase_workspace_v1(const char* profile) noexcept;
extern "C" std::size_t scar_numpy_phase_residue_v1(const char* profile) noexcept;
extern "C" int scar_numpy_phase_apply_v1(const char* profile, void* view, std::size_t size,
    const void* coefficient, std::size_t coefficient_size, void* scratch, std::size_t scratch_size) noexcept;
'''

_PROBE = '''#include "numpy_phase.hpp"
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>

struct Allocation {
    std::unique_ptr<void, decltype(&std::free)> storage{nullptr, std::free};
    explicit Allocation(std::size_t bytes) {
        storage.reset(std::aligned_alloc(64, (bytes + 63) & ~std::size_t{63}));
    }
};

bool read_exact(const char* path, char* destination, std::size_t size) {
    std::ifstream input(path, std::ios::binary);
    return input.read(destination, size) && input.peek() == EOF;
}

int main(int argc, char** argv) {
    if (argc != 5) return 2;
    const auto bytes = scar_numpy_phase_workspace_v1(argv[1]);
    const auto residue = scar_numpy_phase_residue_v1(argv[1]);
    if (residue >= 64 || bytes <= residue) return 2;
    const auto size = bytes - residue;
    Allocation value(bytes), scratch(bytes);
    if (!value.storage || !scratch.storage) return 2;
    auto* view = static_cast<char*>(value.storage.get()) + residue;
    std::array<char, 16> coefficient{};
    if (!read_exact(argv[2], view, size) || !read_exact(argv[3], coefficient.data(), coefficient.size())) return 2;
    if (scar_numpy_phase_apply_v1(argv[1], view, size, coefficient.data(), coefficient.size(),
                                  scratch.storage.get(), bytes) != 0) return 3;
    auto* output = std::fopen(argv[4], "wbx");
    if (!output) return 2;
    const auto written = std::fwrite(view, 1, size, output);
    const auto closed = std::fclose(output);
    return written == size && closed == 0 ? 0 : 2;
}
'''


def _need(condition, reason):
    if not condition:
        raise ValueError('native NumPy phase: ' + reason)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _write(path, report):
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')


def _codes(code):
    yield code
    for child in code.co_consts:
        if isinstance(child, CodeType):
            yield from _codes(child)


def _binding(site, sources):
    ref = site['source']
    source_id = ref['uri'], ref['sha256']
    if source_id not in sources:
        raw = read_artifact(ref, json_document=False)
        statements = {node.lineno for node in ast.walk(ast.parse(raw, filename=ref['uri']))
                      if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Mult)}
        codes = {(code.co_qualname, code.co_firstlineno): code for code in _codes(
            compile(raw, ref['uri'], 'exec', dont_inherit=True))}
        sources[source_id] = statements, codes
    statements, codes = sources[source_id]
    code = codes.get((site['code_qualname'], site['function_first_line']))
    _need(code is not None and site['line'] in statements, 'source has no exact augmented multiplication location')
    positions = [item.positions for item in dis.get_instructions(code) if item.offset == site['offset']]
    _need(len(positions) == 1 and positions[0].lineno == site['line'] and
          [positions[0].lineno, positions[0].end_lineno, positions[0].col_offset, positions[0].end_col_offset]
          == site['source_span'], 'instruction no longer binds to the original source statement')
    return dict(site, code_id=CodeID.from_code(code).key())


def _footprint(shape, strides):
    _need(type(shape) is list and type(strides) is list and 0 < len(shape) == len(strides) <= 64 and
          all(type(n) is int and n > 0 for n in shape) and
          all(type(s) is int and 0 < s <= _MAX_BYTES and s % 16 == 0 for s in strides),
          'expected positive byte-aligned complex128 view')
    size, count = 16, 1
    for stride, extent in sorted(zip(strides, shape, strict=True)):
        if extent == 1:
            continue
        _need(stride >= size, 'view offsets can overlap')
        size += stride * (extent - 1)
        count *= extent
        _need(size <= _MAX_BYTES and count <= _MAX_ELEMENTS, 'view extent exceeds admitted size')
    _need(count >= 2, 'NumPy scalar access requires another route')
    return size, count


def _groups(shape, strides, residue, size):
    import numpy as np

    dtype = np.dtype('<c16')
    owner = np.empty(size + 63, dtype=np.uint8)
    offset = (residue - owner.ctypes.data) % _MOD
    view = np.ndarray(shape, dtype=dtype, buffer=owner, offset=offset, strides=strides)
    scalar = np.empty((), dtype=dtype)
    groups = []
    with np.nditer([view, scalar, view], flags=['external_loop'], order='K',
                   op_flags=[['readonly'], ['readonly'], ['readwrite']]) as iterator:
        for before, coefficient, after in iterator:
            stride, = after.strides
            start, count = after.ctypes.data - view.ctypes.data, after.size
            _need(before.ctypes.data == after.ctypes.data and before.strides == (stride,) and
                  stride >= 16 and stride % 16 == 0 and coefficient.strides == (0,) and
                  len(before) == len(coefficient) == count and 0 <= start and
                  start + (count - 1) * stride + 16 <= size, 'iterator left exact in-place scalar broadcast')
            groups.append((start, count, stride))
    return groups


def _compress(groups):
    runs = []
    for offset, count, stride in groups:
        if runs and runs[-1]['count'] == count and runs[-1]['stride'] == stride:
            latest = runs[-1]
            if latest['repetitions'] == 1:
                latest['advance'] = offset - latest['offset']
            if offset == latest['offset'] + latest['advance'] * latest['repetitions']:
                latest['repetitions'] += 1
                continue
        runs.append(dict(offset=offset, count=count, stride=stride, repetitions=1, advance=0))
    return runs


def _prepare(observation):
    data = read_artifact(observation)
    version = data['schema_version']
    _need(data['schema'] == OBSERVATION and version in (1, 2) and data['observed_only'] is True and
          0 < len(data['samples']) <= _MAX_CASES and len(data['cpu_ids']) == 1 and
          type(data['cpu_ids'][0]) is int and data['cpu_ids'][0] >= 0, 'expected bounded single-CPU source observation')
    read_artifact(data['entry'], json_document=False)
    if version == 2:
        _need(data['collector']['schema'] == 'scar.python-source', 'source capture producer is not recorded')
        read_artifact(data['collector'], json_document=False)
    cases, signatures, seen, sources, layouts = [], set(), set(), {}, {}
    total_elements = 0
    for sample in data['samples']:
        identifier, shape, strides, residue = (sample[key] for key in ('id', 'shape', 'byte_strides', 'address_mod_64'))
        _need(type(identifier) is str and re.fullmatch(r'[A-Za-z0-9_-]{1,64}', identifier) and identifier not in seen and
              sample['normal_return'] is True and sample['original_engine'] == 'EXACT_NUMPY_INPLACE_MULTIPLY_COMPLEX128' and
              sample['input_dtype'] == '<c16' and sample['rounding_mode_raw'] == 0 and
              type(residue) is int and 0 <= residue < _MOD and residue % 8 == 0 and
              sample['case'] in ('DISCOVERY', 'HOLDOUT'), 'source observation exceeds the CPU ndarray operation contract')
        seen.add(identifier)
        size, count = _footprint(shape, strides)
        total_elements += count
        _need(size == sample['storage_size'] and total_elements <= _MAX_TOTAL_ELEMENTS,
              'observed byte size differs from view extent or sample budget is exhausted')
        for key, bytes_, schema in (('input', size, 'scar.ndarray-phase-before'),
                                    ('output', size, 'scar.ndarray-phase-after'),
                                    ('scalar', 16, 'scar.ndarray-phase-scalar')):
            ref = sample[key]
            _need(ref['schema'] == schema and len(read_artifact(ref, json_document=False)) == bytes_,
                  'observed array or scalar byte reference differs')
        layout = tuple(shape), tuple(strides), residue
        if layout not in layouts:
            groups = _groups(shape, strides, residue, size)
            _need(sum(row[1] for row in groups) == count, 'iterator did not account for all source elements')
            # Positive-stride, non-overlapping geometry above guarantees a unique address per element.
            # Verify the iterator visits exactly those addresses before emitting writes from its windows.
            offsets = {0}
            for dimension, stride in zip(shape, strides, strict=True):
                offsets = {offset + index * stride for offset in offsets for index in range(dimension)}
            observed = {offset + index * stride for offset, n, stride in groups for index in range(n)}
            _need(observed == offsets, 'iterator physical coverage differs from view shape')
            signatures.update((stride, 0, stride) for _, _, stride in groups)
            layouts[layout] = _compress(groups)
        case = dict(id=identifier, cohort=sample['case'], source=_binding(sample['site'], sources),
                    shape=shape, byte_strides=strides, address_mod_64=residue, bytes=size,
                    input=sample['input'], scalar=sample['scalar'], expected=sample['output'],
                    steps=layouts[layout], numpy_errstate=sample['numpy_errstate'])
        if version == 2:
            base_shape, base_strides, view_offset = (sample[key] for key in (
                'base_shape', 'base_byte_strides', 'base_offset_bytes'))
            base_size, _ = _footprint(base_shape, base_strides)
            _need(type(view_offset) is int and view_offset % 16 == 0 and 0 <= view_offset <= base_size - size and
                  sample['floating_mode'] == 'FE_TONEAREST_WITH_GRADUAL_UNDERFLOW',
                  'original view is outside its recorded owner or changes floating-point mode')
            case['view_owner'] = dict(shape=base_shape, byte_strides=base_strides, view_offset_bytes=view_offset,
                                      source_extent_bytes=base_size)
        cases.append(case)
    provider = stock.observe(strides={'add': [[None, None, None]], 'multiply': sorted(signatures)})
    _need(data['numpy_version'] == provider['numpy_version'] and
          (version == 1 or data['numpy_provider'] == provider['provider'] and
           data['numpy_c_abi'] == provider['ndarray_c_abi']), 'source NumPy provider differs')
    return cases, provider, data['cpu_ids'][0]


def _members(cases, provider):
    declarations, phases = [], []
    for index, case in enumerate(cases):
        fields = ('offset', 'count', 'stride', 'repetitions', 'advance')
        rows = ['    {' + ', '.join(str(row[key]) for key in fields) + '}' for row in case['steps']]
        declarations.append(f'constexpr Step body_{index}[]{{\n' + ',\n'.join(rows) + '\n};')
        phases.append(f'    {{"{case["id"]}", {case["bytes"]}, {case["address_mod_64"]}, body_{index}}}')
    source = '\n'.join(declarations) + '\nconstexpr Phase phases[]{\n' + ',\n'.join(phases) + '\n};'
    members = stock.render(provider)
    members.update({SOURCE: _TEMPLATE.read_text().replace('@PHASES@', source).encode(),
                    HEADER: _HEADER.encode(), PROBE: _PROBE.encode()})
    return members


def _project(root):
    write_native_project(root, _TARGET, sources=(PROBE, SOURCE, stock.SOURCE),
                         compile_options=('-fno-fast-math', '-ffp-contract=off'), install_dirs=(('licenses', '.'),))


def _verify(binary, case, output, cpu):
    argv = ['/usr/bin/taskset', '-c', str(cpu), str(binary), case['id'],
            case['input']['uri'], case['scalar']['uri'], str(output)]
    trial = subprocess.run(argv, env={}, capture_output=True, timeout=5)
    _need(trial.returncode == 0 and not trial.stdout and not trial.stderr, 'native operation declined ' + case['id'])
    actual, expected = output.read_bytes(), read_artifact(case['expected'], json_document=False)
    _need(actual == expected, 'native phase differs from original view bytes: ' + case['id'])
    return dict(case=case['id'], bytes=len(actual), sha256=sha256(actual).hexdigest())


def _report(root, observation, cases, provider, cpu, members):
    files = {name: _ref(root / name, 'scar.native-project-member')
             for name in sorted({*members, 'CMakeLists.txt', _BINARY})}
    binary = root / _BINARY
    linkage = _audit(binary, provider)
    with TemporaryDirectory(prefix='scar-phase-verify-') as temp:
        validation = [_verify(binary, case, Path(temp) / (case['id'] + '.bin'), cpu) for case in cases]
    return dict(schema=SCHEMA, schema_version=1, observation=observation, cases=cases,
                provider=provider, cpu_ids=[cpu], files=files, binary=_BINARY, linkage=linkage,
                validation=validation, max_scratch_bytes=max(row['bytes'] + row['address_mod_64'] for row in cases),
                scope='EXCLUSIVE_OBSERVED_NDARRAY_VIEW_COMPLEX128_INPLACE_SCALAR_MULTIPLICATION',
                publishes_only_view_elements=True, caller_owns_shape_stride_and_effect_guard=True,
                floating_contract='NEAREST_GRADUAL_UNDERFLOW_RESTORE_CALLER_STATUS_DECLINE_NEW_EXCEPTIONS',
                cxx_standard=20, python_runtime_required=False, deployment_authorized=False)


def export(observation_path, out):
    observation = _ref(observation_path, OBSERVATION)
    cases, provider, cpu = _prepare(observation)
    members = _members(cases, provider)
    root = Path(out).absolute()
    _need(not root.exists() and not root.is_symlink(), 'fresh release path required')
    root.parent.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which('g++')
    _need(compiler is not None, 'C++20 compiler unavailable')
    with TemporaryDirectory(prefix='.scar-numpy-phase-', dir=root.parent) as temporary:
        stage = Path(temporary) / 'release'
        stage.mkdir()
        for name, payload in members.items():
            file = stage / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(payload)
        _project(stage)
        build_native_project(stage, _TARGET, (compiler,))
        report = _report(stage, observation, cases, provider, cpu, members)
        for name, ref in report['files'].items():
            ref['uri'] = str(root / name)
        _write(stage / 'result.json', report)
        stage.rename(root)
    return report


def check(path):
    path = Path(path).resolve(strict=True)
    stored = read_artifact(_ref(path, SCHEMA))
    observation = stored['observation']
    cases, provider, cpu = _prepare(observation)
    members = _members(cases, provider)
    root = path.parent
    _need(all((root / name).read_bytes() == payload for name, payload in members.items()),
          'generated stock or phase source differs')
    with TemporaryDirectory(prefix='scar-phase-cmake-') as temp:
        _project(Path(temp))
        _need((root / 'CMakeLists.txt').read_bytes() == (Path(temp) / 'CMakeLists.txt').read_bytes(),
              'CMake inputs differ')
    _need(stored == _report(root, observation, cases, provider, cpu, members), 'native phase receipt differs on replay')
    return stored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--observation', type=Path)
    mode.add_argument('--check', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.check and not args.out:
        report = check(args.check)
    elif args.observation and args.out:
        report = export(args.observation, args.out)
    else:
        parser.error('provide --check RECEIPT or --observation CAPTURE --out PACKAGE')
    print(json.dumps(dict(cases=len(report['cases']), admitted_strides=report['provider']['strides']['multiply'],
                          max_scratch_bytes=report['max_scratch_bytes'])))


if __name__ == '__main__':
    main()
