"""Build a detached native program for one eager ordered scalar region."""

import argparse
import json
import os
from pathlib import Path
import shlex
from shutil import copy2
import subprocess
from tempfile import TemporaryDirectory

from scar.codegen import scalar_score_standalone as score
from scar.codegen import scalar_transition_standalone as native
from scar.codegen.native_project import build_native_project, write_native_project
from scar.codegen.scalar_table_standalone import _dependencies_for, _FLAGS, _i64, _integer, _LINK_FLAGS, _local
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization import scalar_sorted_region as region


SCHEMA = 'scar.scalar-sorted-native'
_EXE = 'scar_scalar_sorted'
_SOURCE = 'src/main.cpp'
_TEMPLATE = Path(__file__).with_name('scalar_sorted_standalone_v1.cpp.in')
_MAX_ROWS = 65536
_require = region._require


class _Emitter(native._TransitionEmitter):
    def __init__(self, plan, scorer):
        super().__init__({**plan, 'outputs': plan['inputs']}, scorer)

    def expression(self, node, local):
        op = node['op']
        if op == 'or':
            args = [self.expression(operand, local) for operand in node['args']]
            _require(len(args) >= 2 and all(kind == 'bool' for kind, _ in args), 'invalid short-circuit disjunction')
            return score._types(node, 'bool'), '(' + ' || '.join(text for _, text in args) + ')'
        if op == 'gt':
            args = [self.expression(operand, local) for operand in node['args']]
            _require([kind for kind, _ in args] == ['int', 'int'], 'invalid integer comparison')
            return score._types(node, 'bool'), score._ordered(*(text for _, text in args), 'a > b')
        if op == 'contains_int':
            args = [self.expression(operand, local) for operand in node['args']]
            _require(len(args) > 1 and all(kind == 'int' for kind, _ in args), 'invalid integer membership')
            item, *rows = (text for _, text in args)
            choices = f'const std::array<Integer, {len(rows)}> choices{{{", ".join(rows)}}};'
            code = (f'([&] {{ const auto sought = {item}; {choices} '
                    'return std::ranges::find(choices, sought) != choices.end(); }())')
            return score._types(node, 'bool'), code
        return super().expression(node, local)

    def loop(self, row, local, depth):
        prefix = '    ' * depth
        scope = dict(local)
        word = f'iterator_{self.serial}'
        self.serial += 1
        args = [self.expression(arg, local)[1] for arg in row['args']]
        names = row['names']
        operation = row['iterator']
        lines = [prefix + '{']
        if operation == 'integer_count':
            first, step = args
            i = word + '_i'
            lines += [f'{prefix}    const Integer {word}_start = {first};',
                      f'{prefix}    const Integer {word}_step = {step};',
                      f'{prefix}    std::size_t {word}_iterations = 0;',
                      f'{prefix}    for (Integer {i} = {word}_start; ; {i} = add({i}, {word}_step)) {{',
                      f'{prefix}        if (++{word}_iterations > max_rows) throw Decline{{}};']
            scope[names[0]] = 'int', i
        elif operation == 'integer_row':
            key, = args
            mapping = self.maps[row['mapping']]
            i = word + '_value'
            lines += [f'{prefix}    for (const auto {i} : integer_row({mapping}, {key})) {{']
            scope[names[0]] = 'int', i
        else:
            value, = args
            i, text = word + '_i', word + '_text'
            lines += [f'{prefix}    const auto& {word} = {value};',
                      f'{prefix}    for (Integer {i} = 0; {i} < text_length({word}); ++{i}) {{',
                      f'{prefix}        const std::string {text}(1, {word}[static_cast<std::size_t>({i})]);']
            _require(operation in ('enumerate_ascii', 'ascii'), 'invalid ASCII iteration protocol')
            if operation == 'enumerate_ascii':
                scope[names[0]] = 'int', i
            scope[names[-1]] = 'str', text
        lines.extend(self.block(row['body'], scope, depth + 2))
        return [*lines, prefix + '    }', prefix + '}']

    def block(self, body, local, depth=1):
        lines = []
        prefix = '    ' * depth
        for row in body:
            op = row['op']
            if op == 'emit':
                fields = ', '.join(self.expression(operand, local)[1] for operand in row['values'])
                lines.append(f'{prefix}emit(Move{{{fields}}});')
            elif op in ('break', 'continue'):
                lines.append(f'{prefix}{op};')
            elif op == 'for':
                lines.extend(self.loop(row, local, depth))
            elif op == 'if':
                kind, test = self.expression(row['test'], local)
                _require(kind == 'bool', 'region branch requires an exact predicate')
                lines.append(f'{prefix}if ({test}) {{')
                lines.extend(self.block(row['body'], dict(local), depth + 1))
                if row['else']:
                    lines.append(f'{prefix}}} else {{')
                    lines.extend(self.block(row['else'], dict(local), depth + 1))
                lines.append(f'{prefix}}}')
            else:
                _require(op == 'set', 'unsupported sorted region operation')
                lines.extend(super().block([row], local, depth))
        return lines

    def render(self, plan, scorer):
        self.maps = {}
        constants = []
        for j, (name, mapping) in enumerate(sorted(plan['integer_rows'].items())):
            word = f'integer_map_{j}'
            self.maps[name] = word
            references = []
            for k, (key, row) in enumerate(sorted(mapping.items())):
                row_name = f'{word}_{k}'
                values = ', '.join(_integer(value) for value in row)
                constants.append(f'constexpr std::array<Integer, {len(row)}> {row_name}{{{values}}};')
                references.append(f'Row{{{score._literal(key)}, {row_name}}}')
            constants.append(f'constexpr std::array<Row, {len(references)}> {word}{{{", ".join(references)}}};')
        scorer_actuals = []
        for operand in scorer['inputs']:
            name = operand['name']
            if name in self.inputs:
                kind, word = self.inputs[name]
            else:
                _require(name.startswith('arg_'), 'scorer argument has no source port')
                index = int(name.removeprefix('arg_'))
                kind, word = plan['record_types'][index], f'std::get<{index}>(move)'
            _require(kind == operand['type'], 'scorer consumer and region port differ')
            scorer_actuals.append(word)
        formals = [f'[[maybe_unused]] const {self.cpp_types[kind]}& {word}' for kind,word in self.inputs.values()]
        record = ', '.join(self.cpp_types[kind] for kind in plan['record_types'])
        chunks = dict(
            CONSTANTS='\n'.join(constants),
            BODY='\n'.join(self.block(plan['body'], {})),
            SCORER_ARGS=', '.join(scorer_actuals),
            TYPES=record,
            FORMALS=', '.join(formals),
            DESCENDING='std::greater<>{}' if plan['descending'] else 'std::less<>{}',
            ARGC=str(len(formals) + 4),
            ACTUALS=', '.join(f'{native._PARSERS[kind]}(argv[{i}])'
                              for i, (kind, _) in enumerate(self.inputs.values(), 4)),
            MAX_ROWS=str(_MAX_ROWS),
        )
        return chunks


def _render(plan, table_plan, scorer):
    shared, _ = score._TEMPLATE.read_text().split('@SCAR_EVALUATOR@', 1)
    shared = shared.replace('@SCAR_TABLE_DEFINITION@', score._render_tables(table_plan['table_versions']))
    scored = score._Emitter(scorer['inputs']).evaluate(scorer['body']).replace(
        'Integer evaluate(', 'Integer evaluate_score(', 1)
    chunks = _Emitter(plan, scorer).render(plan, scorer)
    transition_text = native._TEMPLATE.read_text()
    transition_runtime = transition_text.split('@SCAR_SHARED_RUNTIME@', 1)[1].split('@SCAR_SCORE@', 1)[0]
    fields = dict(chunks, SHARED=shared + transition_runtime, SCORER=scored)
    generated = _TEMPLATE.read_text()
    for name, value in fields.items():
        generated = generated.replace('@SCAR_' + name + '@', value)
    return generated.encode()


def _cases(plan, table_plan, capture):
    _require(capture['source'] == plan['source'] and capture['sorted_site'] == plan['sorted_site']
             and capture['factory'] == plan['factory'] and capture['all_effects_closed'] is False
             and capture['execution']['entry_path'] == plan['source']['uri']
             and capture['execution']['installed_bindings'] == 1 and capture['execution']['exit_code'] == 0,
             'capture must be at the original sorted region')
    records = []
    for row in capture['samples']:
        version = row['table_version']
        _require(version in table_plan['table_versions'] and _i64(row['threshold']) and _i64(row['nonzero_capture']),
                 'capture differs from the typed threshold, depth or table version')
        flat = row['inputs']
        _require(flat.keys() == {field['name'] for field in plan['inputs']}, 'source samples must have exact state ports')
        argv = [version, str(row['threshold']), str(row['nonzero_capture'])]
        for field in plan['inputs']:
            value = flat[field['name']]
            _require(native._valid(value, field['type']), 'invalid region input port')
            argv.append(str(int(value)) if type(value) is bool else str(value))
        data = bytearray(b'SCAR_SORTED_V1\n')
        _require(len(row['records']) <= _MAX_ROWS, 'captured output exceeds the bounded region')
        for record in row['records']:
            types = ['int', *plan['record_types']]
            _require(len(types) == len(record), 'captured score record differs from its ABI')
            for value, kind in zip(record, types, strict=True):
                _require(native._valid(value, kind), 'invalid score record value')
                if kind == 'int':
                    data.extend(f'i:{value}\n'.encode())
                else:
                    data.extend(b's:' + value.encode('ascii').hex().encode() + b'\n')
        records.append((argv, bytes(data)))
    _require(records and {row['table_version'] for row in capture['samples']} == table_plan['table_versions'].keys(),
             'each score table version needs an original whole region record')
    return records


def _verify(binary, cases):
    outputs = []
    for args, expected in cases:
        run = subprocess.run([str(binary), *args], env={}, cwd='/', capture_output=True, timeout=10)
        _require(run.returncode == 0 and not run.stderr and run.stdout == expected,
                 'native sorted output differs from an original value sequence')
        outputs.append(run.stdout)
    return b''.join(outputs)


def _prepare(plan_ref, capture_ref):
    plan = read_artifact(plan_ref)
    _require(plan == region.project(plan['evidence']['uri'], plan['transition']['uri']),
             'sorted region differs from the original source reconstruction')
    capture = read_artifact(capture_ref)
    _require(capture['region'] == plan_ref and capture['schema_version'] == 1,
             'original capture belongs to a different source region')
    transition = read_artifact(plan['transition'])
    scorer = read_artifact(plan['scorer'])
    return _render(plan, transition, scorer), _cases(plan, transition, capture)


def export(plan_path, capture_path, output):
    plan_ref = artifact_reference(Path(plan_path).resolve(strict=True), region.SCHEMA)
    capture_ref = artifact_reference(Path(capture_path).resolve(strict=True), region.CAPTURE)
    source, cases = _prepare(plan_ref, capture_ref)
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    with TemporaryDirectory(prefix='.scar-sorted-', dir=out.parent) as folder:
        stage = Path(folder) / 'package'
        (stage / 'src').mkdir(parents=True)
        (stage / _SOURCE).write_bytes(source)
        write_native_project(stage, _EXE, compile_options=_FLAGS, link_options=_LINK_FLAGS)
        binary, _ = build_native_project(stage, _EXE, compiler)
        relocated = Path(folder) / 'relocated ordered region'
        copy2(binary, relocated)
        links = _dependencies_for(relocated)
        (stage / 'samples.stdout').write_bytes(_verify(relocated, cases))
        result = dict(schema=SCHEMA, schema_version=1, cxx_standard=20, region=plan_ref, capture=capture_ref,
                      source=_local(stage, _SOURCE, 'scar.cpp-source'),
                      binary=_local(stage, 'bin/' + _EXE, 'scar.elf'),
                      output=_local(stage, 'samples.stdout', 'application/octet-stream'),
                      dependencies=links, samples=len(cases), deployment_authorized=False)
        (stage / 'artifact.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
        stage.rename(out)
    return result


def check(path):
    manifest = Path(path).resolve(strict=True)
    root = manifest.parent
    record = read_artifact(artifact_reference(manifest, SCHEMA))
    source, cases = _prepare(record['region'], record['capture'])
    binary = root / 'bin' / _EXE
    expected = dict(schema=SCHEMA, schema_version=1, cxx_standard=20, region=record['region'],
                    capture=record['capture'], source=_local(root, _SOURCE, 'scar.cpp-source'),
                    binary=_local(root, 'bin/' + _EXE, 'scar.elf'),
                    output=_local(root, 'samples.stdout', 'application/octet-stream'),
                    dependencies=_dependencies_for(binary), samples=len(cases), deployment_authorized=False)
    _require(record == expected and (root / _SOURCE).read_bytes() == source
             and (root / 'samples.stdout').read_bytes() == _verify(binary, cases),
             'ordered native artifacts differ from cold replay')
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args(argv)
    if args.check and not any((args.plan, args.capture, args.out)):
        record = check(args.check)
    elif not args.check and all((args.plan, args.capture, args.out)):
        record = export(args.plan, args.capture, args.out)
    else:
        parser.error('provide --plan, --capture, --out, or --check')
    print(json.dumps(dict(samples=record['samples'], cxx_standard=record['cxx_standard'],
                          deployment_authorized=record['deployment_authorized'])))


if __name__ == '__main__':
    main()
