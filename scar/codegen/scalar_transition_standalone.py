"""Lower one checked scalar state transition and its score call to C++20."""

import argparse
import json
import os
from pathlib import Path
import shlex
from shutil import copy2
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen import scalar_score_standalone as score
from scar.codegen.native_project import build_native_project, write_native_project
from scar.codegen.scalar_table_standalone import _dependencies_for, _FLAGS, _i64, _LINK_FLAGS, _local, _ref, _require
from scar.ir.execution_recipe import read_artifact


SCHEMA = 'scar.scalar-transition-standalone'
PLAN = 'scar.scalar-transition-plan'
_TEMPLATE = Path(__file__).with_name('scalar_transition_standalone_v1.cpp.in')
_EXE = 'scar_scalar_transition'
_BINARY = 'bin/' + _EXE
_SOURCE = 'src/main.cpp'
_SCOPE = 'DETACHED_TYPED_STATE_TRANSITION_OVER_EXPLICIT_FROZEN_TABLE_VERSION'
_HEADER = 'SCAR_STATE_V1\n'
_PARSERS = {'bool': 'parse_bool', 'int': 'parse_integer', 'str': 'parse_string'}
_UNARY = {'neg': ('int', 'sub(Integer{0}, {})'), 'not': ('bool', '!({})'),
          'reverse': ('str', 'reversed({})'), 'swapcase': ('str', 'swapcase({})')}


def _schema(ports):
    _require(type(ports) is list and ports and all(type(row) is dict and set(row) == {'name', 'type'}
        and type(row['name']) is str and row['name'] and row['type'] in _PARSERS for row in ports),
        'transition boundary requires flat typed scalar ports')
    values = {row['name']: row['type'] for row in ports}
    _require(len(values) == len(ports), 'transition port names must be unique')
    return values


def _valid(value, kind):
    return {'bool': lambda item: type(item) is bool, 'str': score._ascii, 'int': _i64}[kind](value)


def encode_outputs(ports, values):
    """Encode one fully computed transition return in its exact flat port order."""
    fields = _schema(ports)
    _require(type(values) is dict and fields.keys() == values.keys(), 'transition output fields differ from their ABI')
    output = [_HEADER]
    for name, kind in fields.items():
        value = values[name]
        _require(_valid(value, kind), 'transition output has an unsupported exact type')
        if kind == 'str':
            output.append('s:' + value.encode('ascii').hex() + '\n')
        elif kind == 'bool':
            output.append(f'b:{int(value)}\n')
        else:
            output.append(f'i:{value}\n')
    return ''.join(output).encode('ascii')


def arguments(plan, sample):
    fields = _schema(plan['inputs'])
    version = sample['table_version']
    values = sample['inputs']
    _require(version in plan['table_versions'] and type(values) is dict and values.keys() == fields.keys(),
             'transition sample inputs differ from their exact table or ports')
    result = [version]
    for name, kind in fields.items():
        value = values[name]
        _require(_valid(value, kind), 'transition sample has an unsupported exact input type')
        result.append(str(int(value)) if kind == 'bool' else str(value))
    return result


def _lambda(definitions, result):
    assignments = ''.join(f'const auto& {name} = {value}; ' for name, value in definitions)
    return f'([&] {{ {assignments}return {result}; }}())'


class _TransitionEmitter(score._Emitter):
    cpp_types = {**score._CPP_TYPE, 'bool_pair': 'BoolPair'}

    def __init__(self, plan, scorer):
        self.inputs = {name: (kind, f'in_{i}') for i, (name, kind) in enumerate(_schema(plan['inputs']).items())}
        self.outputs = tuple(_schema(plan['outputs']).values())
        self.score_types = tuple(_schema(scorer['inputs']).values())
        self.serial = 0

    def expression(self, node, local):
        op = node['op']
        if op == 'bool':
            score._fields(node, ('value',))
            _require(type(node['value']) is bool, 'transition bool literal must be exact')
            return score._types(node, 'bool'), 'true' if node['value'] else 'false'
        if op == 'let':
            return self.let(node, local)
        if op == 'slice':
            score._fields(node, ('value', 'start', 'stop'))
            kind, text = self.expression(node['value'], local)
            _require(kind == 'str', 'transition slice requires ASCII text')
            bindings = [('text', text)]
            for name in ('start', 'stop'):
                operand = node[name]
                if operand is None:
                    bindings.append((name, 'std::nullopt'))
                else:
                    bound, value = self.expression(operand, local)
                    _require(bound == 'int', 'transition slice bounds must be signed-64 integers')
                    bindings.append((name, f'std::optional<Integer>{{{value}}}'))
            return score._types(node, 'str'), _lambda(bindings, 'slice(text, start, stop)')
        if op == 'item':
            score._fields(node, ('value', 'index'))
            kind, base = self.expression(node['value'], local)
            idx_type, idx = self.expression(node['index'], local)
            _require(kind in ('bool_pair', 'row', 'str') and idx_type == 'int', 'unsupported transition index types')
            value_type = {'bool_pair': 'bool', 'row': 'int', 'str': 'str'}[kind]
            return score._types(node, value_type), score._ordered(base, idx, 'item(a, b)')
        if op not in {*_UNARY, 'score_call', 'pair', 'concat', 'mul', 'eq', 'ne', 'splice'}:
            return super().expression(node, local)

        score._fields(node, ('args',))
        operands = [self.expression(arg, local) for arg in node['args']]
        types = tuple(row[0] for row in operands)
        args = tuple(row[1] for row in operands)
        if op == 'score_call':
            _require(types == self.score_types, 'transition score call differs from the exact admitted input ports')
            names = [f'arg_{i}' for i in range(len(args))]
            call = f'evaluate_score(table, {", ".join(names)})'
            return score._types(node, 'int'), _lambda(zip(names, args, strict=True), call)
        if op == 'splice':
            _require(types == ('str', 'int', 'str'), 'transition splice operand types differ')
            names = ('text', 'index', 'replacement')
            return score._types(node, 'str'), _lambda(zip(names, args, strict=True), 'splice(text, index, replacement)')
        if op == 'pair':
            _require(types == ('bool', 'bool'), 'transition pair requires two exact bools')
            return score._types(node, 'bool_pair'), 'BoolPair{' + ', '.join(args) + '}'
        if op in _UNARY:
            kind, render = _UNARY[op]
            _require(types == (kind,), 'transition unary operand type differs')
            return score._types(node, kind), render.replace('{}', args[0])
        if op in ('eq', 'ne'):
            _require(types in (('int', 'int'), ('str', 'str'), ('bool', 'bool')), 'unsupported transition equality types')
            relation = 'a == b' if op == 'eq' else 'a != b'
            return score._types(node, 'bool'), score._ordered(*args, relation)
        kind, render = ('str', 'a + b') if op == 'concat' else ('int', 'multiply(a, b)')
        _require(types == (kind, kind), 'transition binary operand types differ')
        return score._types(node, kind), score._ordered(*args, render)

    def let(self, node, local):
        score._fields(node, ('bindings', 'result'))
        nested, assigned, definitions = dict(local), set(), []
        for binding in node['bindings']:
            name = binding['name']
            _require(type(name) is str and name not in assigned and binding.keys() == {'name', 'value'},
                     'transition lexical parameter bound more than once')
            kind, code = self.expression(binding['value'], local)
            word = f'bound_{self.serial}'
            self.serial += 1
            assigned.add(name)
            nested[name] = kind, word
            definitions.append((word, code))
        kind, code = self.expression(node['result'], nested)
        return score._types(node, kind), _lambda(definitions, code)

    def evaluate(self, body):
        _require(type(body) is list and body and body[-1].keys() == {'op', 'values'} and body[-1]['op'] == 'return',
                 'transition body requires one typed state return')
        local = {}
        lines = self.block(body[:-1], local)
        returns = [self.expression(arg, local) for arg in body[-1]['values']]
        _require(tuple(row[0] for row in returns) == self.outputs, 'transition return differs from the exact output ABI')
        types = ', '.join(self.cpp_types[kind] for kind in self.outputs)
        formals = []
        for kind, word in self.inputs.values():
            cpp = 'const std::string&' if kind == 'str' else self.cpp_types[kind]
            formals.append(f'[[maybe_unused]] {cpp} {word}')
        prototype = 'State evaluate_transition([[maybe_unused]] const Table& table, ' + ', '.join(formals) + ') {'
        result = '    return State{' + ', '.join(row[1] for row in returns) + '};'
        return '\n'.join((f'using State = std::tuple<{types}>;', '', prototype, *lines, result, '}'))


def _render(plan, scorer):
    shared, _ = score._TEMPLATE.read_text().split('@SCAR_EVALUATOR@', 1)
    shared = shared.replace('@SCAR_TABLE_DEFINITION@', score._render_tables(plan['table_versions']))
    scored = score._Emitter(scorer['inputs']).evaluate(scorer['body']).replace('Integer evaluate(',
                                                                             '[[maybe_unused]] Integer evaluate_score(', 1)
    emitter = _TransitionEmitter(plan, scorer)
    formals = list(emitter.inputs.values())
    setup = '\n'.join(f'                const auto {word} = {_PARSERS[kind]}(argv[{i}]);'
                      for i, (kind, word) in enumerate(formals, 2))
    fields = dict(SHARED_RUNTIME=shared, SCORE=scored, TRANSITION=emitter.evaluate(plan['body']),
                  ARGC=str(len(formals) + 2), INPUT_SETUP=setup, INPUTS=', '.join(row[1] for row in formals))
    source = _TEMPLATE.read_text()
    for name, fragment in fields.items():
        source = source.replace('@SCAR_' + name + '@', fragment)
    return source.encode()


def _plan(path):
    from scar.optimization.scalar_transition_ir import check as check_plan

    reference = _ref(path, PLAN)
    plan = check_plan(reference['uri'])
    _require(plan == read_artifact(reference) and plan['score_plan']['schema'] == score.PLAN,
             'checked transition plan or its score binding differs from the immutable artifacts')
    return reference, plan, read_artifact(plan['score_plan'])


def _abi(plan):
    return dict(table_version='EXACT_OBSERVED_SHA256', inputs=plan['inputs'], outputs=plan['outputs'],
                argument_types={'str': 'ASCII_NO_NUL', 'int': 'SIGNED_64_DECIMAL', 'bool': '0_OR_1'},
                integer_intermediates='SIGNED_64_CHECKED', stdout='SCAR_STATE_V1_TAGGED_HEX_ASCII_DECIMAL_BOOL',
                decline='STATUS_2_AND_EMPTY_STDOUT')


def _verify_samples(executable, plan):
    outputs = []
    seen = set()
    for sample in plan['samples']:
        expected = encode_outputs(plan['outputs'], sample['outputs'])
        result = subprocess.run([str(executable), *arguments(plan, sample)], cwd='/', env={}, capture_output=True, timeout=10)
        _require(result.returncode == 0 and not result.stderr and result.stdout == expected,
                 'standalone transition differs from the exact original returned state')
        seen.add(sample['table_version'])
        outputs.append(result.stdout)
    _require(outputs and seen == plan['table_versions'].keys(), 'each embedded table version needs an exact returned state')
    return b''.join(outputs)


def export(plan_path, out):
    """Generate a fresh detached state transition from the closed source plan."""
    plan_ref, plan, scorer = _plan(plan_path)
    return _export_verified(plan_ref, plan, scorer, out)


def _export_verified(plan_ref, plan, scorer, out):
    """Consume a plan already verified by this compiler invocation."""
    _require(plan_ref['schema'] == PLAN and plan == read_artifact(plan_ref)
             and plan['score_plan']['schema'] == score.PLAN and scorer == read_artifact(plan['score_plan']),
             'transition code generation inputs changed after source verification')
    raw = _render(plan, scorer)
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    _require(compiler, 'an available C++20 compiler is required')
    version = subprocess.check_output([*compiler, '--version'], text=True).splitlines()[0]
    with TemporaryDirectory(prefix='.scar-scalar-transition-', dir=out.parent) as pending:
        stage = Path(pending) / 'package'
        (stage / 'src').mkdir(parents=True)
        (stage / _SOURCE).write_bytes(raw)
        write_native_project(stage, _EXE, compile_options=_FLAGS, link_options=_LINK_FLAGS)
        executable, _ = build_native_project(stage, _EXE, compiler)
        relocated = Path(pending) / 'relocated scalar transition'
        copy2(executable, relocated)
        libs = _dependencies_for(relocated)
        (stage / 'samples.stdout').write_bytes(_verify_samples(relocated, plan))
        record = dict(schema=SCHEMA, schema_version=1, cxx_standard=20, scope=_SCOPE, plan=plan_ref,
                      score_plan=plan['score_plan'], abi=_abi(plan), table_versions=sorted(plan['table_versions']),
                      source=_local(stage, _SOURCE, 'scar.cpp-source'),
                      executable=_local(stage, _BINARY, 'scar.native-executable-file'),
                      cmake=_local(stage, 'CMakeLists.txt', 'scar.cmake-project'), linkage=libs,
                      compiler=dict(command=compiler, version=version, options=[*_FLAGS, *_LINK_FLAGS]),
                      validation=_validation(stage, plan), whole_program_cpp=False, deployment_authorized=False)
        (stage / 'artifact.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
        if out.exists():
            raise FileExistsError(out)
        stage.rename(out)
    return _ref(out / 'artifact.json', SCHEMA)


def _validation(root, plan):
    return dict(samples=len(plan['samples']), relocation=True, environment={}, cwd='/',
                stdout=_local(root, 'samples.stdout', 'application/octet-stream'))


def check(path):
    """Recheck source lowering, ELF linkage and actual native returned states."""
    ref = _ref(path, SCHEMA)
    artifact = read_artifact(ref)
    root = Path(ref['uri']).parent
    plan_ref, plan, scorer = _plan(artifact['plan']['uri'])
    _require(artifact['schema_version'] == 1 and artifact['cxx_standard'] == 20 and artifact['scope'] == _SCOPE
             and artifact['whole_program_cpp'] is False and artifact['deployment_authorized'] is False
             and artifact['plan'] == plan_ref and artifact['score_plan'] == plan['score_plan']
             and artifact['abi'] == _abi(plan) and artifact['table_versions'] == sorted(plan['table_versions']),
             'standalone transition scope differs from its exact plan')
    with TemporaryDirectory(prefix='.scar-scalar-transition-cmake-') as pending:
        stage = Path(pending)
        write_native_project(stage, _EXE, compile_options=_FLAGS, link_options=_LINK_FLAGS)
        cmake = (stage / 'CMakeLists.txt').read_bytes()
    _require(artifact['source'] == _local(root, _SOURCE, 'scar.cpp-source')
             and (root / _SOURCE).read_bytes() == _render(plan, scorer)
             and artifact['cmake'] == _local(root, 'CMakeLists.txt', 'scar.cmake-project')
             and (root / 'CMakeLists.txt').read_bytes() == cmake
             and artifact['compiler']['options'] == [*_FLAGS, *_LINK_FLAGS]
             and artifact['executable'] == _local(root, _BINARY, 'scar.native-executable-file')
             and artifact['linkage'] == _dependencies_for(root / _BINARY), 'standalone transition lowering differs')
    _require(artifact['validation'] == _validation(root, plan)
             and (root / 'samples.stdout').read_bytes() == _verify_samples(root / _BINARY, plan),
             'standalone transition sample records differ from fresh replay')
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
