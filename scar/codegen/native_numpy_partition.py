"""Deliver a verified complex slice schedule using retained native operators."""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen.native_cpu_elf import cpu_executable_dynamic
from scar.codegen.native_project import build_native_project, write_native_project
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.native-numpy-partition'
_TARGET = 'scar_native_numpy_partition'
_BINARY = f'bin/{_TARGET}'
_TEMPLATE = Path(__file__).with_suffix('.cpp.in')
_ROOTS = ('target', 'matrix', 'output', 'temp')
_OP = {'COPY_BYTES': 'copy_bytes', 'MULTIPLY': 'multiply', 'ADD': 'add'}
_ID = re.compile(r'[A-Za-z0-9_-]+')
_MAX = 64 * 1024 * 1024


def _need(ok, context):
    if not ok:
        raise ValueError('native slice partition: ' + context)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _ranges(operand, count, step, width, sizes):
    root, offset, stride = (operand[key] for key in ('root', 'offset', 'stride'))
    _need(root in sizes and all(type(value) is int for value in (offset, stride, step)), 'invalid operand address')
    offset += step
    last = offset + stride * (count - 1)
    low, high = min(offset, last), max(offset, last) + width
    _need(0 <= low < high <= sizes[root], 'operand leaves its allocation')
    if stride == 0 or count == 1 or abs(stride) == width:
        return [(low, high)]
    return [(offset + index * stride, offset + index * stride + width) for index in range(count)]


def _step(row, sizes, initialized):
    op, count, operands = (row[key] for key in ('op', 'count', 'operands'))
    copying = op == 'COPY_BYTES'
    _need(op in _OP and type(count) is int and 0 < count <= _MAX and len(operands) == (2 if copying else 3),
          'unsupported native step')
    repetition = row.get('repeat', dict(count=1, offsets=[0] * len(operands)))
    repeat, offsets = repetition['count'], repetition['offsets']
    output = operands[-1]
    _need(len(offsets) == len(operands) and type(repeat) is int and 0 < repeat <= _MAX
          and all(type(delta) is int and abs(delta) <= _MAX for delta in offsets) and count * repeat <= _MAX
          and output['root'] in ('output', 'temp'), 'invalid repetition or write into an immutable input')
    if copying:
        _need(operands[0]['root'] != output['root'] and all(item['stride'] in (0, 1) for item in operands),
              'byte copy requires disjoint roots and offsets')
    else:
        _need(all(item['offset'] % 16 == item['stride'] % 16 == delta % 16 == 0
                  for item, delta in zip(operands, offsets, strict=True))
              and (count == 1 or output['stride'] != 0), 'arithmetic needs aligned complex elements and nonrepeated writes')
    for index in range(repeat):
        spans = [_ranges(item, 1 if copying else count, index * delta, count if copying else 16, sizes)
                 for item, delta in zip(operands, offsets, strict=True)]
        for source, source_spans in zip(operands[:-1], spans[:-1], strict=True):
            root = source['root']
            _need(all(initialized[root].find(0, low, high) < 0 for low, high in source_spans),
                  'step reads an uninitialized private value')
            if root == output['root']:
                _need(source['offset'] == output['offset'] and source['stride'] == output['stride']
                      and source_spans == spans[-1], 'an aliased arithmetic input must match the output view')
        for low, high in spans[-1]:
            initialized[output['root']][low:high] = b'\1' * (high - low)

    if copying:
        operands = [*operands, operands[0]]
        offsets = [*offsets, offsets[0]]
    elems = ', '.join(f'Operand{{{_ROOTS.index(item["root"])}, {item["offset"]}, {item["stride"]}}}'
                      for item in operands)
    increments = ', '.join(map(str, offsets))
    return f'{{Op::{_OP[op]}, {count}, {repeat}, {{{elems}}}, {{{increments}}}' + '}'


def _region_lines(index, name, sizes, residues, steps):
    _need(_ID.fullmatch(name) and set(sizes) == set(residues) == set(_ROOTS)
          and all(type(value) is int and 0 < value <= _MAX for value in sizes.values())
          and all(type(value) is int and 0 <= value < 64 and value % 8 == 0 for value in residues.values()),
          'unsupported exact root profile')
    initial = {root: bytearray((b'\1' if root in ('target', 'matrix') else b'\0') * sizes[root]) for root in sizes}
    body = ',\n    '.join(_step(row, sizes, initial) for row in steps)
    _need(body and initial['output'].find(0) < 0, 'profile must initialize every logical output byte')
    sizes_text = ', '.join(str(sizes[key]) for key in _ROOTS)
    residues_text = ', '.join(str(residues[key]) for key in _ROOTS)
    return (f'constexpr Step body_{index}[]{{\n    {body}\n}};\n',
            f'{{"{name}", {{{sizes_text}}}, {{{residues_text}}}, body_{index}}}')


def _generate(cases):
    ids = [case['id'] for case in cases]
    _need(bool(ids) and len(set(ids)) == len(ids), 'distinct checked profile names required')
    bodies, regions = zip(*(_region_lines(index, row['id'], row['sizes'], row['residues'], row['steps'])
                            for index, row in enumerate(cases)), strict=True)
    declarations = '\n'.join(bodies) + 'constexpr Region regions[]{\n    ' + ',\n    '.join(regions) + '\n};\n'
    return _TEMPLATE.read_text().replace('@PLANS@', declarations).encode()


def _project(root):
    write_native_project(root, _TARGET, sources=('src/main.cpp', 'src/numpy_complex.cpp'),
                         compile_options=('-fno-fast-math', '-ffp-contract=off'),
                         install_dirs=(('licenses', '.'),))


def _argv(binary, case, output, cpu, repeats):
    return ['/usr/bin/taskset', '-c', str(cpu), str(binary), case['id'], str(repeats),
            case['inputs']['target']['uri'], case['inputs']['matrix']['uri'], str(output)]


def _one_run(binary, case, output, cpu, repeats):
    argv = _argv(binary, case, output, cpu, repeats)
    result = subprocess.run(argv, cwd=output.parent, env={}, capture_output=True, timeout=120)
    _need(result.returncode == 0 and not result.stderr and re.fullmatch(rb'[1-9][0-9]*\n', result.stdout),
          'standalone guarded call failed: ' + result.stderr.decode(errors='replace'))
    expected = read_artifact(case['expected'], json_document=False)
    _need(output.read_bytes() == expected, 'standalone complex result differs from observed original bytes: ' + case['id'])
    return dict(id=case['id'], argv=argv, elapsed_ns=int(result.stdout), repeats=repeats,
                output=_ref(output, 'scar.observed-native-complex-output'))


def _roots(row):
    return {('output' if key == 'out' else key): value for key, value in row.items()}


def _prepare(iteration_ref, provider_ref):
    from scar.analysis import numpy_partition_iteration
    from scar.codegen import native_numpy_complex

    iteration = numpy_partition_iteration.check(iteration_ref['uri'])
    _need(iteration == read_artifact(iteration_ref), 'iteration proof changed during replay')
    provider = read_artifact(provider_ref)
    _need(dict(provider['provider'], schema=iteration['provider']['schema']) == iteration['provider'] and all(
        row['strides'] in provider['strides'][row['op'].lower()] for row in iteration['loop_signatures']),
        'live stock ufunc selection does not cover every scheduled operand signature')
    witness = read_artifact(iteration['witness'])
    cpu, = witness['cpu_ids']
    observed = {row['id']: row for row in witness['samples']}
    cases = []
    for sample in iteration['samples']:
        source = observed[sample['id']]
        residues = _roots({name: source[name]['address_mod_64'] for name in ('target', 'matrix', 'out')})
        residues['temp'] = 0
        for key, version in (('program', 'preserved-copy'), ('copy_elided_program', 'elided-copy')):
            program = sample.get(key)
            if program is None:
                continue
            steps = [dict(step, operands=[dict(item, root='output' if item['root'] == 'out' else item['root'])
                                          for item in step['operands']]) for step in program['steps']]
            cases.append(dict(id=sample['id'] + '-' + version, source_sample=sample['id'], group=sample['case'],
                              implementation=version, inputs=sample['inputs'], expected=sample['expected_out'],
                              sizes=_roots(program['buffers']), residues=residues, steps=steps))
    members = native_numpy_complex.render(provider)
    members['src/main.cpp'] = _generate(cases)
    return cases, members, cpu, provider


def _refs(root, members):
    return {name: _ref(root / name, 'scar.native-project-member')
            for name in sorted({*members, 'CMakeLists.txt', _BINARY})}


def _audit(binary, provider):
    from scar.codegen.native_numpy_complex import audit

    linkage = cpu_executable_dynamic(binary)
    _need(linkage['runpath'] is None and linkage['rpath'] is None and
          set(linkage['needed']) <= {'libc.so.6', 'libm.so.6', 'libstdc++.so.6', 'libgcc_s.so.1'},
          'delivery needs only the ordinary system C/C++ dependencies')
    return dict(dynamic=linkage, provider=audit(binary, provider))


def export(iteration_path, provider_path, out):
    from scar.analysis.numpy_partition_iteration import SCHEMA as ITERATION
    from scar.codegen.native_numpy_complex import SCHEMA as PROVIDER

    iteration, provider_ref = _ref(iteration_path, ITERATION), _ref(provider_path, PROVIDER)
    cases, members, cpu, provider = _prepare(iteration, provider_ref)
    root = Path(out).resolve()
    root.mkdir()
    for name, payload in members.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    _project(root)
    compiler = shutil.which('g++')
    _need(compiler is not None, 'C++20 compiler unavailable')
    binary, _ = build_native_project(root, _TARGET, (compiler,))
    linkage = _audit(binary, provider)
    directory = root / 'witness-validation'
    directory.mkdir()
    reports = [_one_run(binary, case, directory / (case['id'] + '.bin'), cpu, 1) for case in cases]
    result = dict(schema=SCHEMA, schema_version=1, iteration=iteration, provider=provider_ref, cpu_ids=[cpu],
                  cases=cases, files=_refs(root, members), binary=_BINARY, linkage=linkage, validation=reports,
                  scope='PRIVATE_STOCK_UFUNC_PARTITION_NORMAL_COMPLETION', cxx_standard=20,
                  python_runtime_required=False, whole_program_equivalence=False, deployment_authorized=False)
    (root / 'result.json').write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return result


def check(path):
    path = Path(path).resolve(strict=True)
    root = path.parent
    stored = read_artifact(_ref(path, SCHEMA))
    cases, members, cpu, provider = _prepare(stored['iteration'], stored['provider'])
    _need(stored['schema_version'] == 1 and stored['cases'] == cases and stored['cpu_ids'] == [cpu]
          and stored['binary'] == _BINARY and stored['files'] == _refs(root, members)
          and stored['linkage'] == _audit(root / _BINARY, provider)
          and stored['python_runtime_required'] is False and stored['deployment_authorized'] is False,
          'native executable or its scope no longer matches the admitted schedule')
    _need(all((root / name).read_bytes() == content for name, content in members.items()),
          'generated source members differ from the checked iteration plan')
    with TemporaryDirectory(prefix='scar-numpy-partition-') as temporary:
        _project(Path(temporary))
        _need((root / 'CMakeLists.txt').read_bytes() == (Path(temporary) / 'CMakeLists.txt').read_bytes(),
              'CMake input differs')
    for case, sample in zip(cases, stored['validation'], strict=True):
        target = root / 'witness-validation' / (case['id'] + '.bin')
        output = _ref(target, 'scar.observed-native-complex-output')
        _need(sample['id'] == case['id'] and sample['output'] == output and sample['repeats'] == 1
              and sample['argv'] == _argv(root / _BINARY, case, target, cpu, 1)
              and type(sample['elapsed_ns']) is int and sample['elapsed_ns'] > 0
              and read_artifact(output, json_document=False) == read_artifact(case['expected'], json_document=False),
              'validated output changed')
    return stored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--iteration', type=Path)
    inputs.add_argument('--check', type=Path)
    parser.add_argument('--provider', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.check:
        report = check(args.check)
    elif args.provider and args.out:
        report = export(args.iteration, args.provider, args.out)
    else:
        parser.error('--iteration requires --provider and --out')
    print(json.dumps(dict(validation_cases=[row['id'] for row in report['validation']], linkage=report['linkage']['dynamic']),
                     sort_keys=True))


if __name__ == '__main__':
    main()
