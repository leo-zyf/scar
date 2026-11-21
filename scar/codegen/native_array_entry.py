"""Assemble a standalone, guarded entry around sealed CPU numerical profiles."""

import argparse
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

from scar.codegen import native_array_tape as tape_backend
from scar.codegen import native_numpy_complex as stock
from scar.codegen import native_controller_argv as argv_backend
from scar.codegen.native_cpu_elf import cpu_executable_dynamic
from scar.codegen.native_project import build_native_project, write_native_project
from scar.ir.execution_recipe import artifact_reference, read_artifact


SCHEMA = 'scar.native-array-entry'
RECIPE = 'scar.native-array-entry-recipe'
_TARGET = 'scar_native_array_entry'
_BINARY = 'bin/' + _TARGET
_TEMPLATE = Path(__file__).with_suffix('.cpp.in')
_PROFILE = Path(__file__).with_name('native_array_entry_profile.cpp.in')
_ARGV = Path(argv_backend.__file__).with_name('native_controller_argv_v1.hpp.in')
_KINDS = {'int': 'Int', 'float64': 'Float', 'str': 'String', 'path': 'String'}
_FLOAT_FIELDS = {'norm', 'split_difference', 'norm_error', 'tolerance', 'elapsed', 'worker_cpu', 'worker_wall'}
_INTEGER_FIELDS = {'pid', 'peak_rss'}
_RUNTIME_FIELDS = _INTEGER_FIELDS | {'pid_list', 'elapsed', 'worker_cpu', 'worker_wall'}


def _need(ok, reason):
    if not ok:
        raise ValueError('native array entry: ' + reason)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


def _text(value):
    if value is None:
        return ''
    if type(value) is str:
        return value
    _need(type(value) in (int, float), 'original option is not a scalar')
    return str(value)


def _substitute(template, replacements):
    return re.sub(r'@([A-Z_]+)@', lambda match: str(replacements[match[1]]), template)


def _argv_header(options, abbreviate):
    handlers, flags, indices = [], {}, {}
    help_options = {}
    for index, (dest, option) in enumerate(options.items()):
        kind = _KINDS[option['kind']]
        initial = _text(option['default'])
        _need(all(argv_backend._FLAG.fullmatch(flag) and flag not in flags and flag not in ('-h', '--help')
                  for flag in option['flags']), 'invalid or duplicate original argument declaration')
        flags.update({flag: index for flag in option['flags']})
        indices[dest] = index
        handlers.append(f'{{Kind::{kind}, {{Kind::{kind}, {argv_backend._quote(initial)}}}}}')
        help_options[dest] = dict(option, kind='str' if option['kind'] == 'path' else option['kind'])
    help_index = len(handlers)
    handlers.append('{Kind::Help, {Kind::Help, ""}}')
    flags.update({'-h': help_index, '--help': help_index})
    template, _ = _ARGV.read_text().split('\nstatic std::vector<std::string> lower_controller_argv', 1)
    for old, new in (
        ('if (token == "--") break;',
         'if (token == "--") throw ScarPublicArgvError("unsupported positional arguments");'),
        ('if (current.type != Token::Option) continue;',
         'if (current.type != Token::Option) throw ScarPublicArgvError("unrecognized argument");'),
        ('if (kind == Kind::String) return std::string(raw);',
         'if (kind == Kind::String && !raw.empty()) return std::string(raw);'),
    ):
        _need(template.count(old) == 1, 'shared argument parser changed')
        template = template.replace(old, new)
    members = dict(OPTION_COUNT=len(handlers), OPTIONS=', '.join(handlers), FLAG_COUNT=len(flags),
        FLAGS=', '.join(f'{{{argv_backend._quote(flag)}, {index}}}' for flag, index in flags.items()),
        HELP_INDEX=help_index, ABBREVIATE=str(abbreviate).lower(),
        HELP_TEXT=argv_backend._help(help_options, abbreviate))
    return _substitute(template, members).encode(), indices


def _report_writer(document, routes):
    """Generate JSON for original fields and authorized output value ports."""
    lines, seen = [], set()

    def emit(value):
        lines.append(f'    out << {argv_backend._quote(value)};')

    def walk(value, path=(), depth=0):
        if path in routes:
            role = routes[path]
            seen.add(path)
            if role in _FLOAT_FIELDS:
                lines.append(f'    binary64(out, result.{role});')
            elif role in _INTEGER_FIELDS:
                lines.append(f'    out << result.{role};')
            elif role == 'pid_list':
                lines.append(f'    out << "["; line(out, pretty, {depth + 1}); out << result.pid;')
                lines.append(f'    line(out, pretty, {depth}); out << "]";')
            elif role == 'optional_tolerance':
                lines.append('    if (result.tolerance_given) binary64(out, result.tolerance); else out << "null";')
            else:
                raise ValueError('no original output binding for ' + role)
            return
        is_map = type(value) is dict
        if is_map or type(value) is list:
            children = tuple(value.items()) if is_map else tuple(enumerate(value))
            emit('{' if is_map else '[')
            for index, (key, field) in enumerate(children):
                if index:
                    emit(',')
                    lines.append("    if (!pretty) out << ' ';")
                lines.append(f'    line(out, pretty, {depth+1});')
                if is_map:
                    emit(json.dumps(key) + ':')
                    lines.append("    out << ' ';")
                walk(field, (*path, key), depth+1)
            if children:
                lines.append(f'    line(out, pretty, {depth});')
            emit('}' if is_map else ']')
        else:
            emit(json.dumps(value, allow_nan=False, separators=(',', ':')))

    walk(document)
    _need(seen == routes.keys(), 'output fields do not cover the declared dynamic result routes')
    return '\n'.join(lines)


def _render(recipe):
    argv, indices = _argv_header(recipe['options'], recipe['allow_abbrev'])
    tolerance_scope = recipe['tolerance_option']
    output_scope = recipe['output_option']
    routes = {tuple(row['path']): row['role'] for row in recipe['report_bindings']}
    _need(len(routes) == len(recipe['report_bindings']), 'duplicate output field route')
    tables, declarations, functions = [], [], []
    sources = {'include/scar_generated/array_entry_argv.hpp': argv}
    for index, profile in enumerate(recipe['profiles']):
        reference = profile['package']
        data = tape_backend.check(reference['uri'])
        _need(read_artifact(reference) == data and data['tape']['sha256'] == profile['tape_sha256'],
              'executable numerical tape differs from admitted profile')
        sources[f'src/tape_{index}.inc'] = read_artifact(data['files']['src/main.cpp'], json_document=False)
        sources[f'src/profile_{index}.cpp'] = _PROFILE.read_text().replace('@INDEX@', str(index)).encode()
        declaration = f'scar_array_entry_profile_{index}'
        declarations.append(f'extern "C" int {declaration}(double*, double*, const char*);')
        predicates = [f'values[{indices[key]}].text == {argv_backend._quote(_text(value))}'
                      for key, value in profile['parameters'].items()]
        _need(predicates and set(profile['parameters']) == set(indices) - {tolerance_scope, output_scope},
              'profile selectors omit original input arguments')
        guards = ' &&\n        '.join(predicates)
        writer = _report_writer(profile['report'], routes)
        functions.extend((
            f'bool matches_{index}(const std::vector<ParsedValue>& values) {{\n    return {guards};\n}}',
            f'void write_{index}(std::ostream& out, const Result& result, bool pretty) {{\n{writer}\n}}'))
        tolerance = profile['default_tolerance']
        _need(type(tolerance) is float and 0 < tolerance < 1, 'unsupported original tolerance')
        tables.append(f'    {{matches_{index}, {declaration}, write_{index}, {tolerance.hex()}}},')
    values = dict(PROFILE_DECLARATIONS='\n'.join(declarations), PROFILES='\n'.join((
        *functions, 'constexpr Profile profiles[]{', *tables, '};')),
        OUTPUT_INDEX=indices[output_scope], TOLERANCE_INDEX=indices[tolerance_scope])
    sources['src/main.cpp'] = _substitute(_TEMPLATE.read_text(), values).encode()
    return sources


def _project(root, profiles, library):
    source_files = [f'src/profile_{i}.cpp' for i in range(profiles)]
    write_native_project(root, _TARGET, sources=(
        'src/main.cpp', 'src/numpy_vector_normalize.cpp', 'src/numpy_observables.cpp', stock.SOURCE, *source_files),
        compile_options=('-fno-fast-math', '-ffp-contract=off', '-DSCAR_ARRAY_TAPE_NORMALIZE'),
        link_options=('-Wl,--disable-new-dtags,-rpath,$ORIGIN/../provider/objects',),
        libraries=('-l:' + library, 'm'), install_dirs=(('licenses', '.'), ('provider', '.')))
    with (root / 'CMakeLists.txt').open('a', encoding='utf-8') as output:
        output.write(f'target_link_directories({_TARGET} PRIVATE "${{CMAKE_CURRENT_SOURCE_DIR}}/provider/objects")\n')
        output.write(f'target_link_options({_TARGET} PRIVATE "-Wl,-rpath-link,${{CMAKE_CURRENT_SOURCE_DIR}}/provider/objects")\n')


def export(recipe_path, out):
    from scar.analysis.native_array_entry_recipe import check
    from scar.codegen.native_numpy_observables import render
    from scar.codegen.native_numpy_vector_normalize import _audit_divide

    recipe_ref = _ref(recipe_path, RECIPE)
    recipe = check(recipe_ref['uri'])
    _need(recipe == read_artifact(recipe_ref), 'entry recipe changed during replay')
    members = _render(recipe)
    normal, libraries, library, normal_provider = tape_backend._normalize(recipe['normalize'])
    members.update(normal)
    members.update(render(normal_provider))
    provider = read_artifact(recipe['profiles'][0]['package'])['stock_provider']
    members.update(stock.render(provider))
    compiler = shutil.which('g++')
    root = Path(out).absolute()
    _need(compiler and not root.exists() and not root.is_symlink(), 'C++20 compiler and fresh package required')
    root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.scar-array-entry-', dir=root.parent) as temp:
        release = Path(temp) / 'release'
        for name, data in members.items():
            member = release / name
            member.parent.mkdir(parents=True, exist_ok=True)
            member.write_bytes(data)
        bundle = release / 'provider/objects'
        bundle.mkdir(parents=True)
        for name, file in libraries.items():
            shutil.copyfile(file, bundle / name)
        _project(release, len(recipe['profiles']), library)
        executable, _ = build_native_project(release, _TARGET, (compiler,),
            expected_libraries=tuple(bundle / name for name in libraries))
        linkage = dict(dynamic=cpu_executable_dynamic(executable), stock=stock.audit(executable, provider),
                       normalization=_audit_divide(executable, normal_provider))
        files = {}
        for name in sorted({*members, 'CMakeLists.txt', _BINARY, *(f'provider/objects/{key}' for key in libraries)}):
            ref = _ref(release / name, 'scar.native-project-member')
            files[name] = dict(ref, uri=str(root / name))
        report = dict(schema=SCHEMA, schema_version=1, recipe=recipe_ref, files=files, linkage=linkage,
            binary=_BINARY, cpu_ids=recipe['cpu_ids'], cxx_standard=20, python_runtime_required=False,
            runtime_arguments_supported=True, holds_source_interpreter_metadata=True,
            preserves_report_schema=True, independent_runtime_seeds=False,
            scope='EXPLICIT_QUALIFIED_PROFILE_SELECTION', deployment_authorized=False)
        (release / 'result.json').write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n')
        release.rename(root)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    report = export(args.recipe, args.out)
    print(json.dumps(dict(binary=report['files'][report['binary']]['uri'], cxx_standard=20)))


if __name__ == '__main__':
    main()
