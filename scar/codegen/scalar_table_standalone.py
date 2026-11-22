"""Compile a witnessed, frozen exact integer table subtraction to C++20."""

import argparse
import ast
import json
import os
from pathlib import Path
import shlex
from shutil import copy2
import subprocess
from tempfile import TemporaryDirectory

from scar.analysis.scalar_table_cse import check as check_proof
from scar.analysis.scalar_table_handoff import CAPTURE as OBSERVATION, DESIGN as CAPTURE_DESIGN, table_version
from scar.codegen.native_cpu_elf import _dependencies
from scar.codegen.native_cpu_elf import cpu_executable_dynamic
from scar.codegen.native_project import build_native_project, write_native_project
from scar.codegen.scalar_table_native import MAX_LIVE_KEYS
from scar.compiler.scalar_table_region import _definition, _position, _span, seal_expression
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.runtime.scalar_table_contract import SCHEMA as REGION
from scar.runtime.scalar_table_contract import _admitted


SCHEMA = 'scar.scalar-table-standalone'
_TARGET = 'scar_scalar_table'
_SOURCE = 'src/main.cpp'
_EXE = 'bin/' + _TARGET
_TEMPLATE = Path(__file__).with_name('scalar_table_standalone_v1.cpp.in')
_FLAGS = ('-O3', '-Wall', '-Wextra', '-Werror', '-pedantic')
_LINK_FLAGS = ('-Wl,-z,defs',)
_I64_MIN, _I64_MAX = -(1 << 63), (1 << 63) - 1
_RUNTIME_SCOPE = 'EXPLICIT_FROZEN_EXACT_TABLE_SUBTRACTION_NORMAL_DOMAIN'
_ABI = dict(argv=['UTF8_KEY', 'SIGNED_DECIMAL_LEFT', 'SIGNED_DECIMAL_RIGHT'],
            result_format='SIGNED_DECIMAL_UTF8_LINE', effects='STDOUT_ON_NORMAL_COMPLETION',
            input_decline='NONZERO_STATUS_NO_STDOUT')
_HOST = {'linux-vdso.so.1', 'libc.so.6', 'libm.so.6', 'libgcc_s.so.1', 'libstdc++.so.6',
         'ld-linux-x86-64.so.2', 'ld-linux-aarch64.so.1'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _local(root, name, schema):
    path = root / name
    _require(path.resolve(strict=True) == path, 'linked member in standalone scalar table package')
    return {**_ref(path, schema), 'uri': name}


def _i64(value):
    return type(value) is int and _I64_MIN <= value <= _I64_MAX


def _table_rows(rows, version):
    _require(type(rows) is dict and 0 < len(rows) <= MAX_LIVE_KEYS, 'bounded frozen scalar table required')
    _require(table_version(rows) == version, 'frozen scalar table content version differs')
    for key, row in rows.items():
        _require(type(key) is str and '\0' not in key and type(row) is list and all(map(_i64, row)),
                 'frozen table requires UTF-8 keys and exact signed-64 tuple cells')
        key.encode('utf-8')
    return {key: tuple(row) for key, row in rows.items()}


def _bind(node, value, names):
    if type(node) is ast.Constant:
        _require(type(node.value) is type(value) and node.value == value, 'observed value differs from source literal')
    else:
        _require(type(node) is ast.Name, 'admitted source operand is not a name')
        if node.id in names:
            previous = names[node.id]
            _require(type(previous) is type(value) and previous == value, 'observed source bindings disagree')
        names[node.id] = value


def _qualify(region_ref, observation_ref, table_version):
    _require(region_ref['schema'] == REGION and observation_ref['schema'] == OBSERVATION,
             'expected exact scalar table region and source observation artifacts')
    report = read_artifact(region_ref)
    region, raw, leaf = _admitted(region_ref)
    source_proof = check_proof(report['proof'])
    case = source_proof['cases'][report['proof_case']]
    replay, revised = seal_expression(raw, case)
    _require(replay == region and revised == leaf, 'sealed scalar table region differs from original expression replay')
    captured = read_artifact(observation_ref)
    _require(captured['schema_version'] == 1 and captured['case'] == report['proof']
             and captured['case_index'] == report['proof_case'] and captured['source'] == region['source']
             and captured['original_execution'] is True
             and captured['table_scope'] == 'EXACT_BUILTIN_ROW_SNAPSHOTS_AT_SELECTED_EXPRESSION_BOUNDARIES'
             and captured['selection']['schema'] == CAPTURE_DESIGN,
             'table observation differs from the exact original source expression')
    design = read_artifact(captured['selection'])
    _require(all(design[key] == captured[key] for key in ('case', 'case_index', 'source'))
             and design['value_code']['key'] == region['original_code_key'],
             'table observation selection differs from the exact source activation')
    versions = captured['table_versions']
    if table_version is None:
        _require(len(versions) == 1, 'select a unique captured scalar table content version')
        table_version, = versions
    _require(table_version in versions, 'the requested table content version was never captured')
    snapshot = versions[table_version]
    table = _table_rows(snapshot['rows'], table_version)

    filename = region['source']['uri']
    tree = ast.parse(raw, filename=filename)
    function = _definition(tree, region['qualname'])
    expr, = [row.value for row in function.body if type(row) is ast.Assign
             and _span(row.value) == _position(region['span'])]
    original = compile(ast.Expression(expr), filename, 'eval', dont_inherit=True,
                       optimize=region['compile_recipe']['optimize'])
    operands = (expr.left.value.value, expr.left.value.slice, expr.left.slice, expr.right.slice)
    samples = []
    for i, row in enumerate(captured['samples']):
        if row['table_version'] != table_version:
            continue
        key, left, right = (row[name] for name in ('key', 'left', 'right'))
        _require(row['root_token'] == snapshot['root_token'] and type(key) is str and key in table
                 and _i64(left) and _i64(right), 'invalid observed table operand or root identity')
        values = table[key]
        _require(-len(values) <= left < len(values) and -len(values) <= right < len(values), 'invalid observed table index')
        names = {}
        for operand, value in zip(operands, (table, key, left, right), strict=True):
            _bind(operand, value, names)
        result = eval(original, {'__builtins__': {}}, names)
        _require(all(_i64(row[name]) for name in ('row_size', 'left_value', 'right_value', 'result'))
                 and (row['row_size'], row['left_value'], row['right_value'], row['result']) ==
                     (len(values), values[left], values[right], result) and _i64(result),
                 'observed source result, tuple cells or signed-64 domain differ')
        samples.append(dict(observation_index=i, arguments=[key, str(left), str(right)], result=result))
    _require(samples, 'the generated table must have at least one observed source sample')
    return region, table_version, table, samples


def _integer(value):
    return 'std::numeric_limits<Integer>::min()' if value == _I64_MIN else str(value)


def _render(table):
    definitions, entries = [], []
    for i, key in enumerate(sorted(table)):
        cells = table[key]
        literal = ''.join(f'\\{byte:03o}' for byte in key.encode('utf-8'))
        blocks = [', '.join(map(_integer, cells[i:i + 12])) for i in range(0, len(cells), 12)]
        values = ',\n'.join('    ' + block for block in blocks)
        definitions.append(f'constexpr std::array<Integer, {len(cells)}> values_{i}{{\n{values}\n}};')
        entries.append(f'    Row{{"{literal}", values_{i}}}')
    joined = ',\n'.join(entries)
    definitions.append(f'constexpr std::array<Row, {len(table)}> rows{{\n{joined}\n}};')
    template = _TEMPLATE.read_text()
    return template.replace('@SCAR_TABLE_DEFINITION@', '\n\n'.join(definitions)).encode()


def _dependencies_for(path):
    elf = cpu_executable_dynamic(path)
    links = _dependencies(subprocess.check_output(['/usr/bin/ldd', str(path)], env={}, text=True))
    _require(elf['runpath'] is None and elf['rpath'] is None and set(elf['needed']) <= links.keys()
             and all(Path(name).name in _HOST for name in links), 'scalar table executable must have only host ELF dependencies')
    return links


def _verify_samples(executable, samples):
    outputs = []
    for case in samples:
        run = subprocess.run([str(executable), *(value.encode('utf-8') for value in case['arguments'])],
                             cwd='/', env={}, capture_output=True, timeout=10)
        _require(run.returncode == 0 and not run.stderr and run.stdout == f"{case['result']}\n".encode('ascii'),
                 'native scalar result differs from the exact original sample')
        outputs.append(run.stdout)
    return b''.join(outputs)


def export(region_path, observation_path, out, *, table_version=None):
    """Publish a standalone CPU executable limited to the admitted snapshot and operation."""
    region_ref = _ref(region_path, REGION)
    observation_ref = _ref(observation_path, OBSERVATION)
    region, version, table, samples = _qualify(region_ref, observation_ref, table_version)
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    _require(compiler, 'an available C++20 compiler is required')
    version_line = subprocess.check_output([*compiler, '--version'], text=True).splitlines()[0]
    with TemporaryDirectory(prefix='.scar-scalar-table-standalone-', dir=out.parent) as pending:
        stage = Path(pending) / 'package'
        (stage / 'src').mkdir(parents=True)
        (stage / _SOURCE).write_bytes(_render(table))
        write_native_project(stage, _TARGET, compile_options=_FLAGS, link_options=_LINK_FLAGS)
        executable, _ = build_native_project(stage, _TARGET, compiler)
        relocated = Path(pending) / 'relocated standalone solver'
        copy2(executable, relocated)
        links = _dependencies_for(relocated)
        outputs = _verify_samples(relocated, samples)
        (stage / 'samples.stdout').write_bytes(outputs)
        artifact = dict(schema=SCHEMA, schema_version=1, scope=_RUNTIME_SCOPE, abi_version=1, cxx_standard=20,
            region=region_ref, observation=observation_ref, table_version=version, source_expression=region['span'],
            operation=region['operation'], source=_local(stage, _SOURCE, 'scar.cpp-source'),
            cmake=_local(stage, 'CMakeLists.txt', 'scar.cmake-project'),
            executable=_local(stage, _EXE, 'scar.native-executable-file'),
            compiler=dict(command=compiler, version=version_line, options=[*_FLAGS, *_LINK_FLAGS]),
            linkage=links, **_ABI, deployment_authorized=False, whole_program_cpp=False,
            validation=dict(samples=samples, stdout=_local(stage, 'samples.stdout', 'application/octet-stream'),
                            relocated=True, environment={}, cwd='/'))
        (stage / 'artifact.json').write_text(json.dumps(artifact, sort_keys=True, indent=2) + '\n')
        if out.exists():
            raise FileExistsError(out)
        stage.rename(out)
    return _ref(out / 'artifact.json', SCHEMA)


def check(path):
    """Verify exact provenance, generated source and executable samples under an empty environment."""
    reference = _ref(path, SCHEMA)
    result = read_artifact(reference)
    folder = Path(reference['uri']).parent
    region, version, table, samples = _qualify(result['region'], result['observation'], result['table_version'])
    _require(result['schema_version'] == 1 and result['scope'] == _RUNTIME_SCOPE
             and (result['cxx_standard'], result['abi_version'], result['table_version']) == (20, 1, version)
             and all(result.get(key) == value for key, value in _ABI.items())
             and result['deployment_authorized'] is False and result['whole_program_cpp'] is False
             and result['source_expression'] == region['span'] and result['operation'] == region['operation'],
             'standalone scalar table package scope differs from its exact source')
    with TemporaryDirectory(prefix='.scar-scalar-table-cmake-') as pending:
        stage = Path(pending)
        write_native_project(stage, _TARGET, compile_options=_FLAGS, link_options=_LINK_FLAGS)
        cmake = (stage / 'CMakeLists.txt').read_bytes()
    _require(result['source'] == _local(folder, _SOURCE, 'scar.cpp-source')
             and (folder / _SOURCE).read_bytes() == _render(table)
             and result['cmake'] == _local(folder, 'CMakeLists.txt', 'scar.cmake-project')
             and (folder / 'CMakeLists.txt').read_bytes() == cmake
             and result['compiler']['options'] == [*_FLAGS, *_LINK_FLAGS]
             and result['executable'] == _local(folder, _EXE, 'scar.native-executable-file')
             and result['linkage'] == _dependencies_for(folder / _EXE),
             'standalone scalar table source, executable or linkage differs from its receipt')
    stdout = _local(folder, 'samples.stdout', 'application/octet-stream')
    _require(result['validation'] == dict(samples=samples, stdout=stdout, relocated=True, environment={}, cwd='/')
             and (folder / 'samples.stdout').read_bytes() == _verify_samples(folder / _EXE, samples),
             'standalone scalar table sample receipts differ from fresh execution')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--region')
    parser.add_argument('--observation')
    parser.add_argument('--table-version')
    parser.add_argument('--out')
    parser.add_argument('--check')
    opt = parser.parse_args(argv)
    if opt.check and not any((opt.region, opt.observation, opt.table_version, opt.out)):
        result = check(opt.check)
    elif opt.region and opt.observation and opt.out and not opt.check:
        result = export(opt.region, opt.observation, opt.out, table_version=opt.table_version)
    else:
        parser.error('provide --region, --observation and --out or --check ARTIFACT')
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
