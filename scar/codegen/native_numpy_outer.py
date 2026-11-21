"""Qualify the ordered complex128 NumPy outer operation for native CPU delivery."""

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

from scar.codegen import native_numpy_complex as stock
from scar.codegen.native_numpy_partition import _audit
from scar.codegen.native_numpy_phase import _codes
from scar.codegen.native_project import build_native_project, write_native_project
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.ids import CodeID

SCHEMA = 'scar.native-numpy-outer'
OBSERVATION = 'scar.numpy-outer-observed'
SOURCE = 'src/numpy_outer.cpp'
HEADER = 'include/scar_generated/numpy_outer.hpp'
_TARGET = 'scar_native_numpy_outer'
_BINARY = f'bin/{_TARGET}'
_TEMPLATE = Path(__file__).with_suffix('.cpp.in')
_MAX_BYTES = 64 << 20
_HEADER = '''#pragma once
#include <cstddef>

struct scar_numpy_outer_layout_v1 {
    std::size_t left_bytes, right_bytes, output_bytes;
    std::size_t left_residue, right_residue, output_residue;
};

extern "C" int scar_numpy_outer_query_v1(const char* profile, scar_numpy_outer_layout_v1* layout) noexcept;

// Inputs are bounded little-endian complex128 C-flat views with exact profile placement.
// The destination is a separate, unpublished C-order buffer. Only status 0 allows publication.
// On numeric failure its bytes may have changed. Input bytes are never written.
extern "C" int scar_numpy_outer_apply_v1(const char* profile, const void* left, std::size_t left_bytes,
    const void* right, std::size_t right_bytes, void* unpublished, std::size_t output_bytes) noexcept;
'''


def _need(condition, reason):
    if not condition:
        raise ValueError('native NumPy outer: ' + reason)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _numpy_contract(binding):
    import numpy as np
    from numpy._core import _multiarray_umath as engine

    original = np.outer.__wrapped__
    ref = binding['source']
    code = original.__code__
    _need(binding['qualname'] == code.co_qualname == 'outer' and binding['first_line'] == code.co_firstlineno and
          ref['schema'] == 'scar.python-source' and _ref(code.co_filename, ref['schema']) == ref and
          original.__globals__['asarray'] is np.asarray and original.__globals__['newaxis'] is None and
          original.__globals__['multiply'] is engine.multiply, 'live NumPy source/provider binding changed')
    tree = ast.parse(read_artifact(ref, json_document=False), filename=ref['uri'])
    matches = [row for row in tree.body if isinstance(row, ast.FunctionDef) and row.name == binding['qualname']]
    expected = ast.parse('''\
a = asarray(a)
b = asarray(b)
return multiply(a.ravel()[:, newaxis], b.ravel()[newaxis, :], out)
''').body
    _need(len(matches) == 1 and [ast.dump(item) for item in matches[0].body[1:]] ==
          [ast.dump(item) for item in expected], 'NumPy outer flatten and operand order changed')


def _source_call(site, sources):
    ref = site['source']
    source_id = ref['uri'], ref['sha256']
    if source_id not in sources:
        _need(ref['schema'] == 'scar.python-source', 'caller input is not original Python source')
        raw = read_artifact(ref, json_document=False)
        calls = {(row.lineno, row.end_lineno, row.col_offset, row.end_col_offset)
                 for row in ast.walk(ast.parse(raw, filename=ref['uri'])) if isinstance(row, ast.Call)}
        codes = {(code.co_qualname, code.co_firstlineno): code for code in _codes(
            compile(raw, ref['uri'], 'exec', dont_inherit=True))}
        sources[source_id] = calls, codes
    calls, codes = sources[source_id]
    code = codes.get((site['qualname'], site['first_line']))
    _need(code is not None and tuple(site['span']) in calls,
          'observed call span does not match original caller')
    instructions = {row.offset: row for row in dis.get_instructions(code, show_caches=True)}
    instruction = instructions.get(site['offset'])
    _need(instruction is not None and list(instruction.positions) == site['span'],
          'source continuation does not bind to the observed call offset')
    return dict(site, code_id=CodeID.from_code(code).key())


def _layout(row):
    shape, strides = row['shape'], row['byte_strides']
    _need(row['dtype'] == '<c16' and row['c_contiguous'] is True and
          type(shape) is list and type(strides) is list and len(shape) == len(strides) <= 64 and
          all(type(count) is int and 0 < count <= _MAX_BYTES // 16 for count in shape) and
          all(type(stride) is int and 0 <= stride <= _MAX_BYTES for stride in strides) and
          type(row['address_mod_64']) is int and 0 <= row['address_mod_64'] < 64 and row['address_mod_64'] % 8 == 0,
          'expected exact complex128 C-contiguous geometry with observed address placement')
    extent = 16
    for count, stride in reversed(tuple(zip(shape, strides, strict=True))):
        _need(count == 1 or stride == extent, 'C-order flatten would materialize an additional input')
        extent *= count
        _need(extent <= _MAX_BYTES, 'flat operand exceeds admitted storage size')
    _need(extent == row['size_bytes'], 'storage size differs from exact C-flat extent')
    return extent // 16


def _groups(left, right):
    import numpy as np

    a, b = np.empty(left, dtype='<c16'), np.empty(right, dtype='<c16')
    out = np.empty((left, right), dtype='<c16')
    roots = tuple(array.ctypes.data for array in (a, b, out))
    groups = []
    index = 0
    with np.nditer([a[:, None], b[None, :], out], flags=['external_loop'], order='K',
                   op_flags=[['readonly'], ['readonly'], ['writeonly']]) as iterator:
        for arrays in iterator:
            offsets = tuple(view.ctypes.data - address for view, address in zip(arrays, roots, strict=True))
            strides = tuple(view.strides[0] for view in arrays)
            count = arrays[2].size
            _need(offsets == (index // right * 16, index % right * 16, index * 16) and
                  all(view.size == count for view in arrays) and all(stride in (0, 16) for stride in strides),
                  'observed K iterator changes C-order flatten or operand orientation')
            if count > 1:
                expected = (16, 0, 16) if right == 1 else (0, 16, 16)
                _need(strides == expected and (right == 1 or index % right + count <= right),
                      'observed broadcast grouping changes source operand offsets')
            groups.append((offsets, strides, count))
            index += count
    _need(index == left * right and groups, 'iterator did not cover the complete fresh output')
    offsets, strides, count = groups[0]
    advance = [after - start for start, after in zip(offsets, groups[1][0], strict=True)] if len(groups) > 1 else [0] * 3
    _need(all(steps == strides and elements == count and tuple(
        start + repetition * delta for start, delta in zip(offsets, advance, strict=True)) == address
        for repetition, (address, steps, elements) in enumerate(groups)), 'non-affine K iteration remains unqualified')
    return dict(offsets=list(offsets), strides=list(strides), count=count, repeats=len(groups), advance=advance)


def _prepare(observation):
    report = read_artifact(observation)
    _need(report['schema'] == OBSERVATION and report['schema_version'] == 1 and report['observed_only'] is True and
          len(report['cpu_ids']) == 1 and type(report['cpu_ids'][0]) is int and report['cpu_ids'][0] >= 0 and
          0 < len(report['samples']) <= 128, 'expected bounded single-CPU original observation')
    for key in ('entry', 'collector'):
        _need(report[key]['schema'] == 'scar.python-source', 'source capture provenance unavailable')
        read_artifact(report[key], json_document=False)
    _numpy_contract(report['numpy_source'])
    seen, sources, geometries, signatures = set(), {}, {}, set()
    cases = []
    total = 0
    for sample in report['samples']:
        name = sample['id']
        _need(type(name) is str and re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name) and name not in seen and
              sample['normal_return'] is True and sample['fresh_disjoint_output'] is True and
              sample['floating_round_to_nearest_gradual_underflow'] is True and
              type(sample['run']) is int and 0 <= sample['run'] < len(report['runs']),
              'source observation exceeds admitted outcome, ownership or floating contract')
        seen.add(name)
        a, b = (sample['input_roles'][role] for role in ('left', 'right'))
        out = sample['output']
        left, right = map(_layout, (a, b))
        _need(_layout(out) == left * right and out['shape'] == [left, right] and out['writeable'] is True,
              'fresh output shape differs from outer construction')
        sizes = tuple(row['size_bytes'] for row in (a, b, out))
        total += sum(sizes)
        _need(total <= _MAX_BYTES, 'sample byte budget is exhausted')
        refs = [sample['input_payloads'][role] for role in ('left', 'right')] + [sample['expected']]
        schemas = ['scar.ndarray-outer-cflat-input'] * 2 + ['scar.ndarray-outer-output']
        for ref, schema, size in zip(refs, schemas, sizes, strict=True):
            _need(ref['schema'] == schema and ref['size_bytes'] == size and
                  len(read_artifact(ref, json_document=False)) == size, 'exact source operand/output payload differs')
        shape = left, right
        if shape not in geometries:
            geometries[shape] = _groups(*shape)
            signatures.add(tuple(geometries[shape]['strides']))
        cases.append(dict(id=name, run=sample['run'], caller=_source_call(sample['caller'], sources),
                          input_roles=sample['input_roles'], output=out, sizes=list(sizes),
                          residues=[row['address_mod_64'] for row in (a, b, out)],
                          grouping=geometries[shape], inputs=sample['input_payloads'], expected=sample['expected'],
                          numpy_errstate=sample['numpy_errstate']))
    provider = stock.observe(strides={'add': [[None, None, None]], 'multiply': sorted(signatures)})
    _need(provider['provider'] == report['numpy_provider'] and provider['numpy_version'] == report['numpy_version'] and
          provider['ndarray_c_abi'] == report['numpy_c_abi'], 'observed NumPy ELF, version or ABI differs')
    return report, cases, provider


def _members(cases, provider):
    def values(items):
        return '{' + ', '.join(map(str, items)) + '}'

    profiles = []
    for case in cases:
        group = case['grouping']
        fields = [f'"{case["id"]}"', values([*case['sizes'], *case['residues']]),
                  str(group['count']), str(group['repeats'])]
        fields.extend(values(group[key]) for key in ('offsets', 'strides', 'advance'))
        profiles.append('    {' + ', '.join(fields) + '}')
    data = 'constexpr Profile profiles[]{\n' + ',\n'.join(profiles) + '\n};'
    return {**stock.render(provider), HEADER: _HEADER.encode(),
            SOURCE: _TEMPLATE.read_text().replace('@PROFILES@', data).encode()}


def _project(root):
    write_native_project(root, _TARGET, sources=(SOURCE, stock.SOURCE), install_dirs=(('licenses', '.'),),
                         compile_options=('-DSCAR_OUTER_PROBE=1', '-fno-fast-math', '-ffp-contract=off'))


def _verify(binary, case, output, cpu):
    argv = ['/usr/bin/taskset', '-c', str(cpu), str(binary), case['id'],
            case['inputs']['left']['uri'], case['inputs']['right']['uri'], str(output)]
    result = subprocess.run(argv, env={}, capture_output=True, timeout=5)
    _need(result.returncode == 0 and not result.stdout and not result.stderr, 'native operator declined ' + case['id'])
    actual = output.read_bytes()
    _need(actual == read_artifact(case['expected'], json_document=False), 'operator output bytes differ for ' + case['id'])
    return dict(id=case['id'], run=case['run'], size_bytes=len(actual), sha256=sha256(actual).hexdigest())


def _report(root, observation, observed, cases, provider, members):
    files = {name: _ref(root / name, 'scar.native-project-member')
             for name in sorted({*members, 'CMakeLists.txt', _BINARY})}
    binary = root / _BINARY
    linkage = _audit(binary, provider)
    with TemporaryDirectory(prefix='scar-outer-verify-') as directory:
        validation = [_verify(binary, case, Path(directory) / (case['id'] + '.bin'), observed['cpu_ids'][0])
                      for case in cases]
    return dict(schema=SCHEMA, schema_version=1, observation=observation, original_numpy_source=observed['numpy_source'],
                cases=cases, provider=provider, files=files, binary=_BINARY, cpu_ids=observed['cpu_ids'],
                linkage=linkage, validation=validation, abi_version=1, cxx_standard=20, python_runtime_required=False,
                output_contract='EXCLUSIVE_FRESH_UNPUBLISHED_C_ORDER_OUTPUT',
                input_contract='BORROWED_C_FLAT_COMPLEX128_IN_ORIGINAL_OPERAND_ORDER', native_workspace_bytes=0,
                floating_contract='NEAREST_GRADUAL_UNDERFLOW_RESTORE_CALLER_STATUS_DECLINE_NEW_EXCEPTIONS',
                caller_owns_input_layout_and_output_publication=True, source_shape_composition_lowered=False,
                deployment_authorized=False)


def export(observation_path, out):
    """Build a guarded C++20 package from source-linked original ndarray witnesses."""
    observation = _ref(observation_path, OBSERVATION)
    observed, cases, provider = _prepare(observation)
    members = _members(cases, provider)
    root = Path(out).absolute()
    _need(not root.exists() and not root.is_symlink(), 'fresh native package path required')
    compiler = shutil.which('g++')
    _need(compiler is not None, 'C++20 compiler unavailable')
    root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-numpy-outer-', dir=root.parent) as temporary:
        stage = Path(temporary) / 'package'
        stage.mkdir()
        for name, contents in members.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)
        _project(stage)
        build_native_project(stage, _TARGET, (compiler,))
        report = _report(stage, observation, observed, cases, provider, members)
        for name, ref in report['files'].items():
            ref['uri'] = str(root / name)
        (stage / 'result.json').write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
        stage.rename(root)
    return report


def check(path):
    """Cold-check source, stock machine bytes, and every recorded operator output."""
    path = Path(path).resolve(strict=True)
    stored = read_artifact(_ref(path, SCHEMA))
    observed, cases, provider = _prepare(stored['observation'])
    members = _members(cases, provider)
    root = path.parent
    _need(all((root / name).read_bytes() == contents for name, contents in members.items()),
          'generated source, provider or notices changed')
    with TemporaryDirectory(prefix='scar-outer-cmake-') as directory:
        _project(Path(directory))
        _need((root / 'CMakeLists.txt').read_bytes() == (Path(directory) / 'CMakeLists.txt').read_bytes(),
              'CMake inputs changed')
    _need(stored == _report(root, stored['observation'], observed, cases, provider, members),
          'native outer receipt differs on replay')
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
    print(json.dumps(dict(cases=len(report['cases']), verified_strides=report['provider']['strides']['multiply'],
                          native_workspace_bytes=report['native_workspace_bytes'])))


if __name__ == '__main__':
    main()
