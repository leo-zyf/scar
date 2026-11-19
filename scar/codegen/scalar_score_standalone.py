"""Lower a checked detached scalar score plan to a standalone C++20 executable."""

import argparse
import json
import os
from pathlib import Path
import shlex
from shutil import copy2
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen.native_project import build_native_project, write_native_project
from scar.codegen.scalar_table_standalone import (
    _dependencies_for, _FLAGS, _i64, _integer, _LINK_FLAGS, _local, _ref, _require, _table_rows, _verify_samples,
)
from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.scalar-score-standalone'
PLAN = 'scar.scalar-score-plan'
_TEMPLATE = Path(__file__).with_name('scalar_score_standalone_v1.cpp.in')
_EXE = 'scar_scalar_score'
_BINARY = 'bin/' + _EXE
_SOURCE = 'src/main.cpp'
_SCOPE = 'DETACHED_FULL_SCALAR_SCORE_OVER_EXPLICIT_FROZEN_TABLE_VERSION'
_CPP_TYPE = {'int': 'Integer', 'str': 'std::string', 'bool': 'bool', 'row': 'Cells'}
_ARITH = {'add': 'add(a, b)', 'sub': 'sub(a, b)', 'floordiv': 'floor_div(a, b)'}
_COMPARE = {'eq': '==', 'lt': '<', 'le': '<='}


def _ascii(text):
    return type(text) is str and text.isascii() and '\0' not in text


def _literal(text):
    _require(_ascii(text), 'score string literals and ABI inputs must be ASCII without NUL')
    payload = ''.join(f'\\{byte:03o}' for byte in text.encode('ascii'))
    return f'"{payload}"'


def _types(node, inferred):
    _require(node.get('type', inferred) == inferred, 'score expression type differs from its typed operands')
    return inferred


def _fields(node, expected):
    _require(type(node) is dict and node.keys() - {'type'} == {'op', *expected}, 'unexpected score IR expression fields')


def _ordered(left, right, task):
    # The checked expressions can decline. Finish the original left operand
    # before entering the right, independently of C++ argument ordering.
    return f'([&] {{ const auto& a = {left}; const auto& b = {right}; return {task}; }}())'


class _Emitter:
    cpp_types = _CPP_TYPE

    def __init__(self, inputs):
        self.inputs = {row['name']: (row['type'], f'in_{i}') for i, row in enumerate(inputs)}
        _require(len(self.inputs) == len(inputs) and self.inputs
                 and all(type(row['name']) is str and row['name'] and row['type'] in ('str', 'int')
                         and row.keys() == {'name', 'type'} for row in inputs),
                 'score inputs need a unique sequence of typed ASCII or signed-64 formals')
        self.serial = 0

    def expression(self, node, local):
        op = node['op']
        if op in ('int', 'str'):
            _fields(node, ('value',))
            value = node['value']
            _require(_i64(value) if op == 'int' else _ascii(value), 'invalid typed score literal')
            text = _integer(value) if op == 'int' else f'std::string({_literal(value)})'
            return _types(node, op), text
        if op in ('input', 'get'):
            _fields(node, ('name',))
            names = self.inputs if op == 'input' else local
            _require(node['name'] in names, 'score operand has no prior typed definition')
            kind, text = names[node['name']]
            return _types(node, kind), text
        if op == 'table_row':
            _fields(node, ('key',))
            kind, key = self.expression(node['key'], local)
            _require(kind == 'str', 'score table row requires one ASCII key')
            return _types(node, 'row'), f'table_row(table, {key})'
        if op == 'item':
            _fields(node, ('value', 'index'))
            kind, value = self.expression(node['value'], local)
            index_type, index = self.expression(node['index'], local)
            _require(kind in ('str', 'row') and index_type == 'int', 'score index requires a row or ASCII text')
            return _types(node, 'str' if kind == 'str' else 'int'), _ordered(value, index, 'item(a, b)')

        _fields(node, ('args',))
        arguments = [self.expression(arg, local) for arg in node['args']]
        types = tuple(row[0] for row in arguments)
        args = tuple(row[1] for row in arguments)
        if op == 'and':
            _require(len(args) >= 2 and all(item == 'bool' for item in types), 'score boolean operands must be bool')
            return _types(node, 'bool'), '(' + ' && '.join(args) + ')'
        if op == 'select':
            _require(len(args) == 3 and types[0] == 'bool' and types[1] == types[2], 'score choice types differ')
            return _types(node, types[1]), f'({args[0]} ? {args[1]} : {args[2]})'
        if op in ('abs', 'upper'):
            kind, operation = ('int', 'absolute') if op == 'abs' else ('str', 'upper_ascii')
            _require(types == (kind,), 'score unary operand type differs')
            return _types(node, kind), f'{operation}({args[0]})'
        if op in _ARITH:
            _require(types == ('int', 'int'), 'score arithmetic requires two signed-64 integers')
            return _types(node, 'int'), _ordered(*args, _ARITH[op])
        if op == 'in':
            _require(types == ('str', 'str'), 'score string membership requires ASCII inputs')
            return _types(node, 'bool'), _ordered(*args, 'contains(a, b)')
        if op in _COMPARE:
            _require(types == ('int', 'int') or op == 'eq' and types == ('str', 'str'),
                     'score comparison has unsupported operand types')
            return _types(node, 'bool'), _ordered(*args, f'a {_COMPARE[op]} b')
        raise ValueError('unsupported normalized score expression: ' + str(op))

    def block(self, body, local, depth=1):
        prefix = '    ' * depth
        lines = []
        for i, statement in enumerate(body):
            op = statement['op']
            if op == 'set':
                _require(statement.keys() == {'op', 'name', 'value'} and type(statement['name']) is str,
                         'invalid score local assignment')
                name = statement['name']
                kind, code = self.expression(statement['value'], local)
                if name in local:
                    original, cpp = local[name]
                    _require(original == kind, 'score local cannot change its scalar type')
                    lines.append(f'{prefix}{cpp} = {code};')
                else:
                    cpp = f'local_{self.serial}'
                    self.serial += 1
                    local[name] = (kind, cpp)
                    lines.append(f'{prefix}[[maybe_unused]] {self.cpp_types[kind]} {cpp} = {code};')
            elif op == 'if':
                _require(statement.keys() <= {'op', 'test', 'body', 'else', 'site'}
                         and type(statement.get('site', '')) is str, 'invalid score conditional')
                kind, test = self.expression(statement['test'], local)
                _require(kind == 'bool', 'score condition must be bool')
                lines.append(f'{prefix}if ({test}) {{')
                lines.extend(self.block(statement['body'], dict(local), depth + 1))
                if statement.get('else'):
                    lines.append(f'{prefix}}} else {{')
                    lines.extend(self.block(statement['else'], dict(local), depth + 1))
                lines.append(f'{prefix}}}')
            elif op == 'return' and depth == 1 and i == len(body) - 1:
                _require(statement.keys() == {'op', 'value'}, 'invalid score return statement')
                kind, value = self.expression(statement['value'], local)
                _require(kind == 'int', 'score return must be signed-64')
                lines.append(f'{prefix}return {value};')
            else:
                raise ValueError('unsupported normalized score statement: ' + str(op))
        return lines

    def evaluate(self, body):
        _require(type(body) is list and body and body[-1]['op'] == 'return', 'score body needs one final scalar return')
        formals = []
        for kind, cpp in self.inputs.values():
            dtype = 'const std::string&' if kind == 'str' else _CPP_TYPE[kind]
            formals.append(f'[[maybe_unused]] {dtype} {cpp}')
        prototype = 'Integer evaluate([[maybe_unused]] const Table& table, ' + ', '.join(formals) + ') {'
        return '\n'.join((prototype, *self.block(body, {}), '}'))


def _render_tables(versions):
    _require(type(versions) is dict and versions, 'at least one observed table version required')
    definitions, tables, symbols = [], [], {}
    for version_index, (version, snapshot) in enumerate(sorted(versions.items())):
        table = _table_rows(snapshot['rows'], version)
        rows = []
        for key, cells in sorted(table.items()):
            if cells not in symbols:
                symbol = f'cells_{len(symbols)}'
                symbols[cells] = symbol
                blocks = [', '.join(map(_integer, cells[i:i + 12])) for i in range(0, len(cells), 12)]
                values = ',\n'.join('    ' + block for block in blocks)
                definitions.append(f'constexpr std::array<Integer, {len(cells)}> {symbol}{{\n{values}\n}};')
            symbol = symbols[cells]
            rows.append(f'    Row{{{_literal(key)}, {symbol}}}')
        contents = ',\n'.join(rows)
        definitions.append(f'constexpr std::array<Row, {len(rows)}> rows_{version_index}{{\n{contents}\n}};')
        tables.append(f'    Table{{"{version}", rows_{version_index}}}')
    contents = ',\n'.join(tables)
    definitions.append(f'constexpr std::array<Table, {len(tables)}> tables{{\n{contents}\n}};')
    return '\n\n'.join(definitions)


def _render(plan):
    emitter = _Emitter(plan['inputs'])
    reads, arguments = [], []
    for i, (kind, cpp) in enumerate(emitter.inputs.values(), 2):
        parse = 'parse_integer' if kind == 'int' else 'parse_string'
        reads.append(f'                const auto {cpp} = {parse}(argv[{i}]);')
        arguments.append(cpp)
    fields = dict(TABLE_DEFINITION=_render_tables(plan['table_versions']), EVALUATOR=emitter.evaluate(plan['body']),
                  ARGC=str(len(arguments) + 2), INPUT_SETUP='\n'.join(reads), INPUTS=', '.join(arguments))
    result = _TEMPLATE.read_text()
    for name, value in fields.items():
        result = result.replace('@SCAR_' + name + '@', value)
    return result.encode()


def _samples(plan):
    result = []
    schema = {row['name']: row['type'] for row in plan['inputs']}
    for row in plan['samples']:
        entries = row['inputs']
        _require(entries.keys() == schema.keys() and row['table_version'] in plan['table_versions']
                 and _i64(row['result']), 'invalid score sample boundary')
        args = [row['table_version']]
        for name, kind in schema.items():
            item = entries[name]
            _require(_i64(item) if kind == 'int' else _ascii(item), 'score sample differs from its typed ABI')
            args.append(str(item))
        result.append(dict(arguments=args, result=row['result']))
    _require(result and {row['arguments'][0] for row in result} == set(plan['table_versions']),
             'each embedded table version needs an original result sample')
    return result


def _plan(path):
    from scar.optimization.scalar_score_ir import check as check_plan

    reference = _ref(path, PLAN)
    plan = check_plan(reference['uri'])
    _require(plan == read_artifact(reference), 'checked score plan differs from its immutable reference')
    return reference, plan


def _abi(plan):
    return dict(table_version='EXACT_OBSERVED_SHA256', inputs=plan['inputs'], strings='ASCII_NO_NUL',
                integers='SIGNED_64_CHECKED_INTERMEDIATES', index='PYTHON_SIGNED_INDEX_WITH_RANGE_GUARD',
                output='SIGNED_DECIMAL_LINE', decline='STATUS_2_AND_EMPTY_STDOUT')


def export(plan_path, out):
    """Generate a new standalone package from the checked normalized source plan."""
    plan_ref, plan = _plan(plan_path)
    source, samples = _render(plan), _samples(plan)
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    _require(compiler, 'an available C++20 compiler is required')
    version = subprocess.check_output([*compiler, '--version'], text=True).splitlines()[0]
    with TemporaryDirectory(prefix='.scar-scalar-score-', dir=out.parent) as pending:
        stage = Path(pending) / 'package'
        (stage / 'src').mkdir(parents=True)
        (stage / _SOURCE).write_bytes(source)
        write_native_project(stage, _EXE, compile_options=_FLAGS, link_options=_LINK_FLAGS)
        exe, _ = build_native_project(stage, _EXE, compiler)
        relocated = Path(pending) / 'relocated scalar score'
        copy2(exe, relocated)
        links = _dependencies_for(relocated)
        (stage / 'samples.stdout').write_bytes(_verify_samples(relocated, samples))
        artifact = dict(schema=SCHEMA, schema_version=1, cxx_standard=20, scope=_SCOPE, plan=plan_ref,
                        abi=_abi(plan), table_versions=sorted(plan['table_versions']),
                        source=_local(stage, _SOURCE, 'scar.cpp-source'),
                        executable=_local(stage, _BINARY, 'scar.native-executable-file'),
                        cmake=_local(stage, 'CMakeLists.txt', 'scar.cmake-project'), linkage=links,
                        compiler=dict(command=compiler, version=version, options=[*_FLAGS, *_LINK_FLAGS]),
                        validation=dict(samples=len(samples), relocation=True, environment={}, cwd='/',
                            stdout=_local(stage, 'samples.stdout', 'application/octet-stream')),
                        whole_program_cpp=False, deployment_authorized=False)
        (stage / 'artifact.json').write_text(json.dumps(artifact, sort_keys=True, indent=2) + '\n')
        if out.exists():
            raise FileExistsError(out)
        stage.rename(out)
    return _ref(out / 'artifact.json', SCHEMA)


def check(path):
    """Rebuild the source projection and replay all original samples from the package."""
    artifact_ref = _ref(path, SCHEMA)
    artifact = read_artifact(artifact_ref)
    root = Path(artifact_ref['uri']).parent
    plan_ref, plan = _plan(artifact['plan']['uri'])
    _require(artifact['schema_version'] == 1 and artifact['cxx_standard'] == 20 and artifact['scope'] == _SCOPE
             and artifact['whole_program_cpp'] is False and artifact['deployment_authorized'] is False
             and artifact['plan'] == plan_ref and artifact['abi'] == _abi(plan)
             and artifact['table_versions'] == sorted(plan['table_versions']), 'standalone score scope differs')
    with TemporaryDirectory(prefix='.scar-scalar-score-cmake-') as pending:
        stage = Path(pending)
        write_native_project(stage, _EXE, compile_options=_FLAGS, link_options=_LINK_FLAGS)
        cmake = (stage / 'CMakeLists.txt').read_bytes()
    _require(artifact['source'] == _local(root, _SOURCE, 'scar.cpp-source')
             and (root / _SOURCE).read_bytes() == _render(plan)
             and artifact['cmake'] == _local(root, 'CMakeLists.txt', 'scar.cmake-project')
             and (root / 'CMakeLists.txt').read_bytes() == cmake
             and artifact['compiler']['options'] == [*_FLAGS, *_LINK_FLAGS]
             and artifact['executable'] == _local(root, _BINARY, 'scar.native-executable-file')
             and artifact['linkage'] == _dependencies_for(root / _BINARY), 'standalone score lowering differs')
    samples = _samples(plan)
    stdout = _local(root, 'samples.stdout', 'application/octet-stream')
    _require(artifact['validation'] == dict(samples=len(samples), relocation=True, environment={}, cwd='/', stdout=stdout)
             and (root / 'samples.stdout').read_bytes() == _verify_samples(root / _BINARY, samples),
             'standalone score samples differ from fresh replay')
    return artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan')
    parser.add_argument('--out')
    parser.add_argument('--check')
    args = parser.parse_args(argv)
    if args.check and not any((args.plan, args.out)):
        result = check(args.check)
    elif args.plan and args.out and not args.check:
        result = export(args.plan, args.out)
    else:
        parser.error('provide --plan and --out or --check ARTIFACT')
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
