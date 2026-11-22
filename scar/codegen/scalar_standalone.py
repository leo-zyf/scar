"""Publish and independently verify exact scalar C++20 executables."""

import argparse
import ast
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
from shutil import copy2
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen.native_project import build_native_project, write_native_project
from scar.codegen.native_provider_bundle import _dependencies, cpu_executable_dynamic
from scar.codegen.scalar_source import (
    _i64, _require, _table_rows, encode_outputs, render_score, render_table,
    render_transition, score_samples, transition_arguments,
)
from scar.ir.execution_recipe import artifact_reference, open_artifact, read_artifact


TABLE = 'scar.scalar-table-standalone'
SCORE = 'scar.scalar-score-standalone'
TRANSITION = 'scar.scalar-transition-standalone'
_SCORE_PLAN = 'scar.scalar-score-plan'
_TRANSITION_PLAN = 'scar.scalar-transition-plan'
_SOURCE = 'src/main.cpp'
_FLAGS = ('-O3', '-Wall', '-Wextra', '-Werror', '-pedantic')
_LINK_FLAGS = ('-Wl,-z,defs',)
_HOST = {'linux-vdso.so.1', 'libc.so.6', 'libm.so.6', 'libgcc_s.so.1', 'libstdc++.so.6',
         'ld-linux-x86-64.so.2', 'ld-linux-aarch64.so.1'}


@dataclass(frozen=True, slots=True)
class _ScalarPackage:
    target: str
    receipt: dict
    source: bytes
    validation: dict
    cases: list[tuple[list[str | bytes], bytes]]
    sample_error: str

    @property
    def executable(self):
        return 'bin/' + self.target


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _local(root, name, schema):
    path = root / name
    _require(path.resolve(strict=True) == path, 'linked member in standalone scalar table package')
    return {**_ref(path, schema), 'uri': name}


def _bind(node, value, names):
    if type(node) is ast.Constant:
        _require(type(node.value) is type(value) and node.value == value, 'observed value differs from source literal')
    else:
        _require(type(node) is ast.Name, 'admitted source operand is not a name')
        if node.id in names:
            previous = names[node.id]
            _require(type(previous) is type(value) and previous == value, 'observed source bindings disagree')
        names[node.id] = value


def _qualify_table(region_ref, observation_ref, table_version):
    from scar.analysis.scalar_table_cse import check as check_proof
    from scar.analysis.scalar_table_handoff import CAPTURE as OBSERVATION, DESIGN as CAPTURE_DESIGN
    from scar.compiler.scalar_table_region import _definition, _position, _span, seal_expression
    from scar.runtime.scalar_table_contract import SCHEMA as REGION, _admitted

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


def _dependencies_for(path):
    elf = cpu_executable_dynamic(path)
    links = _dependencies(subprocess.check_output(['/usr/bin/ldd', str(path)], env={}, text=True))
    _require(elf['runpath'] is None and elf['rpath'] is None and set(elf['needed']) <= links.keys()
             and all(Path(name).name in _HOST for name in links), 'scalar table executable must have only host ELF dependencies')
    return links


def _compiler():
    command = shlex.split(os.environ.get('CXX', 'c++'))
    _require(command, 'an available C++20 compiler is required')
    version = subprocess.check_output([*command, '--version'], text=True).splitlines()[0]
    return dict(command=command, version=version, options=[*_FLAGS, *_LINK_FLAGS])


def _write_cmake(root, package):
    write_native_project(root, package.target, compile_options=_FLAGS, link_options=_LINK_FLAGS)


def _verify(executable, package):
    outputs = []
    for arguments, expected in package.cases:
        run = subprocess.run([str(executable), *arguments], cwd='/', env={}, capture_output=True, timeout=10)
        _require(run.returncode == 0 and not run.stderr and run.stdout == expected, package.sample_error)
        outputs.append(run.stdout)
    return b''.join(outputs)


def _validation(root, package):
    return dict(package.validation, environment={}, cwd='/',
                stdout=_local(root, 'samples.stdout', 'application/octet-stream'))


def _publish(package, out):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    compiler = _compiler()
    with TemporaryDirectory(prefix='.scar-scalar-', dir=out.parent) as pending:
        stage = Path(pending) / 'package'
        (stage / 'src').mkdir(parents=True)
        (stage / _SOURCE).write_bytes(package.source)
        _write_cmake(stage, package)
        executable, _ = build_native_project(stage, package.target, compiler['command'])
        relocated = Path(pending) / ('relocated ' + package.target.replace('_', ' '))
        copy2(executable, relocated)
        linkage = _dependencies_for(relocated)
        (stage / 'samples.stdout').write_bytes(_verify(relocated, package))
        record = dict(package.receipt, compiler=compiler, linkage=linkage,
                      source=_local(stage, _SOURCE, 'scar.cpp-source'),
                      cmake=_local(stage, 'CMakeLists.txt', 'scar.cmake-project'),
                      executable=_local(stage, package.executable, 'scar.native-executable-file'),
                      validation=_validation(stage, package))
        (stage / 'artifact.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
        if out.exists():
            raise FileExistsError(out)
        stage.rename(out)
    return _ref(out / 'artifact.json', record['schema'])


def _check(path, schema, prepare, *, scope_error, package_error, validation_error):
    path = Path(path).resolve(strict=True)
    _, artifact = open_artifact(path, schema)
    root = path.parent
    package = prepare(artifact)
    _require(artifact['whole_program_cpp'] is False and artifact['deployment_authorized'] is False
             and all(artifact.get(name) == value for name, value in package.receipt.items()), scope_error)
    with TemporaryDirectory(prefix='.scar-scalar-cmake-') as pending:
        stage = Path(pending)
        _write_cmake(stage, package)
        cmake = (stage / 'CMakeLists.txt').read_bytes()
    executable = root / package.executable
    _require(artifact['source'] == _local(root, _SOURCE, 'scar.cpp-source')
             and (root / _SOURCE).read_bytes() == package.source
             and artifact['cmake'] == _local(root, 'CMakeLists.txt', 'scar.cmake-project')
             and (root / 'CMakeLists.txt').read_bytes() == cmake
             and artifact['compiler']['options'] == [*_FLAGS, *_LINK_FLAGS]
             and artifact['executable'] == _local(root, package.executable, 'scar.native-executable-file')
             and artifact['linkage'] == _dependencies_for(executable), package_error)
    _require(artifact['validation'] == _validation(root, package)
             and (root / 'samples.stdout').read_bytes() == _verify(executable, package), validation_error)
    return artifact


def _scope(schema, scope, **fields):
    return dict(schema=schema, schema_version=1, cxx_standard=20, scope=scope,
                whole_program_cpp=False, deployment_authorized=False, **fields)


def _integer_cases(samples):
    return [([value.encode('utf-8') for value in case['arguments']], f"{case['result']}\n".encode('ascii'))
            for case in samples]


def _table_package(region_ref, observation_ref, table_version):
    region, version, table, samples = _qualify_table(region_ref, observation_ref, table_version)
    receipt = _scope(TABLE, 'EXPLICIT_FROZEN_EXACT_TABLE_SUBTRACTION_NORMAL_DOMAIN',
        abi_version=1, region=region_ref, observation=observation_ref, table_version=version,
        source_expression=region['span'], operation=region['operation'],
        argv=['UTF8_KEY', 'SIGNED_DECIMAL_LEFT', 'SIGNED_DECIMAL_RIGHT'],
        result_format='SIGNED_DECIMAL_UTF8_LINE', effects='STDOUT_ON_NORMAL_COMPLETION',
        input_decline='NONZERO_STATUS_NO_STDOUT')
    return _ScalarPackage('scar_scalar_table', receipt, render_table(table),
        dict(samples=samples, relocated=True), _integer_cases(samples),
        'native scalar result differs from the exact original sample')


def export_table(region_path, observation_path, out, *, table_version=None):
    """Publish a C++20 executable for the admitted frozen table and operation."""
    from scar.analysis.scalar_table_handoff import CAPTURE as OBSERVATION
    from scar.runtime.scalar_table_contract import SCHEMA as REGION

    return _publish(_table_package(_ref(region_path, REGION), _ref(observation_path, OBSERVATION), table_version), out)


def check_table(path):
    return _check(path, TABLE, lambda row: _table_package(row['region'], row['observation'], row['table_version']),
        scope_error='standalone scalar table package scope differs from its exact source',
        package_error='standalone scalar table source, executable or linkage differs from its receipt',
        validation_error='standalone scalar table sample receipts differ from fresh execution')


def _score_plan(path):
    from scar.optimization.scalar_score_ir import check as check_plan

    reference = _ref(path, _SCORE_PLAN)
    plan = check_plan(reference['uri'])
    _require(plan == read_artifact(reference), 'checked score plan differs from its immutable reference')
    return reference, plan


def _score_package(path):
    reference, plan = _score_plan(path)
    source = render_score(plan)
    samples = score_samples(plan)
    abi = dict(table_version='EXACT_OBSERVED_SHA256', inputs=plan['inputs'], strings='ASCII_NO_NUL',
               integers='SIGNED_64_CHECKED_INTERMEDIATES', index='PYTHON_SIGNED_INDEX_WITH_RANGE_GUARD',
               output='SIGNED_DECIMAL_LINE', decline='STATUS_2_AND_EMPTY_STDOUT')
    receipt = _scope(SCORE, 'DETACHED_FULL_SCALAR_SCORE_OVER_EXPLICIT_FROZEN_TABLE_VERSION',
                     plan=reference, abi=abi, table_versions=sorted(plan['table_versions']))
    return _ScalarPackage('scar_scalar_score', receipt, source, dict(samples=len(samples), relocation=True),
                          _integer_cases(samples), 'native scalar result differs from the exact original sample')


def export_score(plan_path, out):
    """Publish a C++20 executable for an admitted normalized score plan."""
    return _publish(_score_package(plan_path), out)


def check_score(path):
    return _check(path, SCORE, lambda row: _score_package(row['plan']['uri']),
        scope_error='standalone score scope differs',
        package_error='standalone score lowering differs',
        validation_error='standalone score samples differ from fresh replay')


def read_transition_plan(path):
    from scar.optimization.scalar_transition_ir import check as check_plan

    reference = _ref(path, _TRANSITION_PLAN)
    plan = check_plan(reference['uri'])
    _require(plan == read_artifact(reference) and plan['score_plan']['schema'] == _SCORE_PLAN,
             'checked transition plan or its score binding differs from the immutable artifacts')
    return reference, plan, read_artifact(plan['score_plan'])


def _transition_package(path):
    reference, plan, scorer = read_transition_plan(path)
    source = render_transition(plan, scorer)
    samples = plan['samples']
    cases = [(transition_arguments(plan, sample), encode_outputs(plan['outputs'], sample['outputs']))
             for sample in samples]
    _require(cases and {sample['table_version'] for sample in samples} == plan['table_versions'].keys(),
             'each embedded table version needs an exact returned state')
    abi = dict(table_version='EXACT_OBSERVED_SHA256', inputs=plan['inputs'], outputs=plan['outputs'],
               argument_types={'str': 'ASCII_NO_NUL', 'int': 'SIGNED_64_DECIMAL', 'bool': '0_OR_1'},
               integer_intermediates='SIGNED_64_CHECKED', stdout='SCAR_STATE_V1_TAGGED_HEX_ASCII_DECIMAL_BOOL',
               decline='STATUS_2_AND_EMPTY_STDOUT')
    receipt = _scope(TRANSITION, 'DETACHED_TYPED_STATE_TRANSITION_OVER_EXPLICIT_FROZEN_TABLE_VERSION',
                     plan=reference, score_plan=plan['score_plan'], abi=abi, table_versions=sorted(plan['table_versions']))
    return _ScalarPackage('scar_scalar_transition', receipt, source, dict(samples=len(samples), relocation=True),
                          cases, 'standalone transition differs from the exact original returned state')


def export_transition(plan_path, out):
    """Publish a C++20 executable for an admitted exact state transition."""
    return _publish(_transition_package(plan_path), out)


def check_transition(path):
    return _check(path, TRANSITION, lambda row: _transition_package(row['plan']['uri']),
        scope_error='standalone transition scope differs from its exact plan',
        package_error='standalone transition lowering differs',
        validation_error='standalone transition sample records differ from fresh replay')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    table = modes.add_parser('table')
    table.add_argument('--region')
    table.add_argument('--observation')
    table.add_argument('--table-version')
    score = modes.add_parser('score')
    transition = modes.add_parser('transition')
    for mode in (score, transition):
        mode.add_argument('--plan')
    for mode in (table, score, transition):
        mode.add_argument('--out')
        mode.add_argument('--check')
    args = parser.parse_args(argv)
    if args.mode == 'table':
        if args.check and not any((args.region, args.observation, args.table_version, args.out)):
            result = check_table(args.check)
        elif args.region and args.observation and args.out and not args.check:
            result = export_table(args.region, args.observation, args.out, table_version=args.table_version)
        else:
            table.error('provide --region, --observation and --out or --check ARTIFACT')
    else:
        check, export = (check_score, export_score) if args.mode == 'score' else (check_transition, export_transition)
        if args.check and not any((args.plan, args.out)):
            result = check(args.check)
        elif args.plan and args.out and not args.check:
            result = export(args.plan, args.out)
        else:
            (score if args.mode == 'score' else transition).error('provide --plan and --out or --check ARTIFACT')
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
