"""Lower admitted Fortran source units to standalone C++20."""

import hashlib
import json
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

from scar.ir.execution_recipe import read_artifact

from ..frontend.fortran import _run, _tool, _units, capture_fortran


_FIXED_SUFFIXES = {'.f', '.for', '.ftn'}
_POINTER_ADJUSTMENTS = '/* Parameter adjustments */'
_FUNCTION_BODY = '/* Function Body */'
_CPP_NONCODE = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\[\s\S]|[^"\\])*"|\'(?:\\[\s\S]|[^\'\\])*\'')
_POINTER_SHIFT = re.compile(r'^\s*(?:--(\w+)|(\w+)\s*-=\s*(\w+))\s*;', re.MULTILINE)
_INDIRECT_CALL = re.compile(r'\(\s*\*\s*(\w+)\s*\)\s*\(\s*(?!\s*\))')
_CALL_WRAPPER = '''#pragma once

template<class Result, class... Args>
static inline Result scar_f2c_call(Result (*callee)(...), Args... args) {
    return reinterpret_cast<Result (*)(Args...)>(callee)(args...);
}
'''


def _runtime(config, name):
    value = config.get(name)
    if not value or not Path(value).is_file():
        raise ValueError(f'Fortran entry requires an existing fortran.{name} file.')
    return Path(value).resolve()


def _mask_cpp(source):
    return _CPP_NONCODE.sub(lambda match: re.sub(r'[^\n]', ' ', match[0]), source)


def _next_bracket_end(source, opening):
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == '[':
            depth += 1
        elif source[index] == ']':
            depth -= 1
            if not depth:
                return index
    raise ValueError('unbalanced subscript in generated f2c source')


def _apply_spans(source, edits):
    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    return source


def _repair_f2c_pointers(source):
    """Subtract one-based offsets in subscripts without forming invalid base pointers."""
    changes = []
    for marker in re.finditer(re.escape(_POINTER_ADJUSTMENTS), source):
        body = source.find(_FUNCTION_BODY, marker.end())
        if body < 0:
            raise ValueError('missing function body after f2c parameter adjustments')
        closing = re.search(r'^}', source[body:], re.MULTILINE)
        if not closing:
            raise ValueError('missing function end in generated f2c source')
        stop = body + closing.start()
        prefix = source[marker.end():body]
        mask = _mask_cpp(source[body:stop])
        for shift in _POINTER_SHIFT.finditer(prefix):
            decremented, offset_pointer, offset = shift.groups()
            pointer = decremented or offset_pointer
            offset = '1' if decremented else offset
            changes.append((marker.end() + shift.start(), marker.end() + shift.end(), ''))
            for use in re.finditer(r'\b' + re.escape(pointer) + r'\b\s*', mask):
                opening = use.end()
                if opening == len(mask) or mask[opening] != '[':
                    raise ValueError(f'unsupported shifted pointer use in f2c output. {pointer}')
                closing = _next_bracket_end(mask, opening)
                changes.extend(((body + opening + 1, body + opening + 1, '('),
                                (body + closing, body + closing, f') - ({offset})')))
    return _apply_spans(source, changes)


def _repair_f2c_source(source):
    source = _repair_f2c_pointers(source)
    mask = _mask_cpp(source)
    edits = [(call.start(), call.end(), f'scar_f2c_call({call[1]}, ')
             for call in _INDIRECT_CALL.finditer(mask)]
    literal = r'("(?:\\.|[^"\\])*")'
    # libf2c's STOP message and formatted WRITE input strings are read-only.
    for match in re.finditer(r'\b(?:s_stop\s*\(\s*|do_fio\s*\(\s*&\w+\s*,\s*)'
                             + literal + r'\s*,', source):
        edits.append((*match.span(1), f'const_cast<char*>({match[1]})'))
    return '#include "scar_f2c_call.hpp"\n' + _apply_spans(source, edits)


def _translate_f2c(units, edits, project, config, *, long_lines=False, source_directory='src'):
    translator = _tool(config, 'f2c')
    header = _runtime(config, 'f2c_header')
    archive = _runtime(config, 'f2c_archive')
    source_dir = project / source_directory
    generated = project / 'include' / 'scar_generated'
    source_dir.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    contents = header.read_text(encoding='utf-8')
    # The external runtime only reads a supplied format. Const qualification
    # keeps its ABI and permits strict C++20 string literal initialization.
    contents = re.sub(r'\bchar\s*\*\s*cifmt\b', 'const char *cifmt', contents)
    (generated / 'f2c.h').write_text(contents, encoding='utf-8')
    (generated / 'scar_f2c_call.hpp').write_text(_CALL_WRAPPER, encoding='utf-8')

    sources = []
    with TemporaryDirectory(prefix='.f2c-', dir=project) as temporary:
        staging = Path(temporary)
        inputs = []
        for index, (name, path, flags) in enumerate(units):
            accepted = {'-std=legacy', '-std=gnu', '-ffixed-form', '-ffixed-line-length-72',
                        '-fno-fast-math', '-ffp-contract=off'}
            if path.suffix.lower() not in _FIXED_SUFFIXES or set(flags) - accepted:
                raise ValueError(f'f2c requires supported fixed-form inputs and options. {name}')
            unit = staging / f'fortran-{index}.f'
            unit.write_bytes(edits[name] if name in edits else path.read_bytes())
            inputs.append(unit.name)

        options = ['-C++', '-Ps', '-g']
        if long_lines:
            options.append('-f')
        prototypes = {}
        for _ in range(4):
            _run([translator, *options, *prototypes, *inputs], staging, accepted=(0, 4))
            inferred = {path.name: path.read_bytes() for path in sorted(staging.glob('*.P'))}
            if inferred == prototypes and inferred:
                break
            prototypes = inferred
        else:
            raise ValueError('f2c could not stabilize compiler-generated prototypes')

        for unit in inputs:
            translated = (staging / unit).with_suffix('.c').read_text(encoding='utf-8')
            dest = (source_dir / unit).with_suffix('.cpp')
            dest.write_text(_repair_f2c_source(translated), encoding='utf-8')
            sources.append(dest.relative_to(project).as_posix())

    return {'sources': sources, 'libraries': (str(archive), 'm'),
            'compile_options': ('-ffp-contract=off', '-fno-fast-math', '-Werror=write-strings')}


def _translate_freeform(manifest, units, edits, project, config, captures):
    env = os.environ.copy()
    if 'fparser_path' in config:
        dependency = Path(config['fparser_path']).resolve()
        if not (dependency / 'fparser').is_dir():
            raise ValueError('fortran.fparser_path must contain the fparser package')
        search_paths = [str(dependency)]
        if env.get('PYTHONPATH'):
            search_paths.append(env['PYTHONPATH'])
        env['PYTHONPATH'] = os.pathsep.join(search_paths)

    with TemporaryDirectory(prefix='.fortran-freeform-', dir=project) as temp:
        temporary = Path(temp)
        sources = temporary / 'sources'
        snapshots = {}
        for name, path, _ in units:
            snapshot = sources / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(edits[name] if name in edits else path.read_bytes())
            snapshots[name] = str(snapshot)
        if captures and not edits:
            captured = []
            for index, (name, _, _) in enumerate(units):
                semantics = temporary / f'unit-{index}.fortran-original'
                semantics.write_bytes(read_artifact(captures[name], json_document=False))
                captured.append({'path': name, 'semantics': str(semantics)})
        else:
            captured = capture_fortran({**manifest, 'source_root': str(sources)}, temporary / 'capture')['units']
        request = {'entry': manifest.get('entry', 'main'),
                   'units': [{'source': snapshots[unit['path']], 'semantics': unit['semantics']}
                             for unit in captured]}
        arguments = temporary / 'lowering.json'
        arguments.write_text(json.dumps(request), encoding='utf-8')
        helper = Path(__file__).with_name('_fortran_freeform.py')
        fixed = json.loads(_run([sys.executable, '-B', str(helper), str(arguments)], temporary, env=env))
        transport = []
        for name, source in fixed.items():
            path = temporary / f'{name}.f'
            path.write_text(source, encoding='utf-8')
            transport.append((path.name, path, ()))
        return _translate_f2c(transport, {}, project, config, long_lines=True)


def emit_fortran(manifest, project, edits, *, source_directory='src', fixed_only=False, captures=None):
    """Emit standalone C++20 host sources. Return shared CMake writer arguments."""
    config = manifest.get('fortran', {})
    units = list(_units(manifest))
    project = Path(project).resolve()
    project.mkdir(parents=True, exist_ok=True)
    if all(path.suffix.lower() in _FIXED_SUFFIXES for _, path, _ in units):
        build = _translate_f2c(units, edits, project, config, source_directory=source_directory)
        translation = 'f2c-cxx20'
    else:
        if fixed_only:
            raise ValueError('mixed Fortran requires the fixed-form external procedure ABI')
        build = _translate_freeform(manifest, units, edits, project, config, captures)
        translation = 'gfortran-typed-fparser-fixed-form-f2c-cxx20'
    recipe = {'compiler': translation,
              'sources': [{'path': name, 'sha256': hashlib.sha256(
                  edits[name] if name in edits else path.read_bytes()).hexdigest()}
                          for name, path, _ in units],
              'native_libraries': list(build['libraries'])}
    (project / 'fortran-emission.json').write_text(json.dumps(recipe, indent=2) + '\n', encoding='utf-8')
    return build
