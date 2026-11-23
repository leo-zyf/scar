"""Read the admitted Fortran ABI and prepare compiler-compatible transport bodies."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from ..contract import read_artifact


def _run(args, cwd, *, accepted=(0,), env=None):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=180, env=env)
    if result.returncode not in accepted:
        raise ValueError(f'{Path(args[0]).name} failed with status {result.returncode}\n{result.stderr.strip()}')
    return result.stdout


def _tool(config, name):
    path = config.get(name) or shutil.which(name)
    if not path or not Path(path).is_file():
        raise ValueError(f'Fortran entry requires {name}. Supply its path in fortran.{name}.')
    return str(Path(path).resolve())


_FIXED_SUFFIXES = {'.f', '.for', '.ftn'}
_POINTER_ADJUSTMENTS = '/* Parameter adjustments */'
_FUNCTION_BODY = '/* Function Body */'
_CPP_NONCODE = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\[\s\S]|[^"\\])*"|\'(?:\\[\s\S]|[^\'\\])*\'')
_POINTER_SHIFT = re.compile(r'^\s*(?:--(\w+)|(\w+)\s*-=\s*(\w+))\s*;', re.MULTILINE)
_INDIRECT_CALL = re.compile(r'\(\s*\*\s*(\w+)\s*\)\s*\(\s*(?!\s*\))')


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


def _repair_f2c_pointers(source, descriptor_arrays=()):
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
                if pointer in descriptor_arrays:
                    changes.extend(((body + use.start(), body + opening + 1,
                                     f'(*scar_cfi_element({pointer}, ('),
                                    (body + closing, body + closing + 1, f') - ({offset})))')))
                else:
                    changes.extend(((body + opening + 1, body + opening + 1, '('),
                                    (body + closing, body + closing, f') - ({offset})')))
    repaired = _apply_spans(source, changes)
    if descriptor_arrays:
        mask = _mask_cpp(repaired)
        for pointer in descriptor_arrays:
            if re.search(r'\b' + re.escape(pointer) + r'\s*\[', mask):
                raise ValueError(f'unlowered CFI element access in f2c output. {pointer}')
    return repaired


def _repair_f2c_source(source, descriptor_arrays=()):
    source = _repair_f2c_pointers(source, descriptor_arrays)
    mask = _mask_cpp(source)
    edits = [(call.start(), call.end(), f'scar_f2c_call({call[1]}, ')
             for call in _INDIRECT_CALL.finditer(mask)]
    literal = r'("(?:\\.|[^"\\])*")'
    # libf2c's STOP message and formatted WRITE input strings are read-only.
    for match in re.finditer(r'\b(?:s_stop\s*\(\s*|do_fio\s*\(\s*&\w+\s*,\s*)'
                             + literal + r'\s*,', source):
        edits.append((*match.span(1), f'const_cast<char*>({match[1]})'))
    return '#include "scar_f2c_call.hpp"\n' + _apply_spans(source, edits)


def _translate_f2c(units, project, config, *, source_directory='src', descriptor_units=None):
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

    sources = []
    with TemporaryDirectory(prefix='.f2c-', dir=project) as temporary:
        staging = Path(temporary)
        inputs = {}
        cohorts = {False: [], True: []}
        accepted = {'-std=legacy', '-std=gnu', '-ffixed-form', '-ffixed-line-length-72',
                    '-fno-fast-math', '-ffp-contract=off'}
        for index, (name, data, flags, extended) in enumerate(units):
            if Path(name).suffix.lower() not in _FIXED_SUFFIXES or set(flags) - accepted:
                raise ValueError(f'f2c requires supported fixed-form inputs and options. {name}')
            unit = staging / f'fortran-{index}.f'
            unit.write_bytes(data)
            inputs[unit.name] = (descriptor_units or {}).get(name, ()) if extended else ()
            cohorts[extended].append(unit.name)

        prototypes = {}
        for _ in range(4):
            for extended, members in cohorts.items():
                if members:
                    options = ['-C++', '-Ps', '-g', *(['-f'] if extended else [])]
                    _run([translator, *options, *prototypes, *members], staging, accepted=(0, 4))
            inferred = {path.name: path.read_bytes() for path in sorted(staging.glob('*.P'))}
            if inferred == prototypes and inferred:
                break
            prototypes = inferred
        else:
            raise ValueError('f2c could not stabilize compiler-generated prototypes')

        for unit, descriptor_arrays in inputs.items():
            translated = (staging / unit).with_suffix('.c').read_text(encoding='utf-8')
            dest = (source_dir / unit).with_suffix('.cpp')
            dest.write_text(_repair_f2c_source(translated, descriptor_arrays), encoding='utf-8')
            sources.append(dest.relative_to(project).as_posix())

    support = 'f2c-cfi-v1' if descriptor_units else 'f2c-v1'
    return {'sources': sources, 'libraries': (str(archive), 'm'), 'runtime_support': (support,),
            'compile_options': ('-ffp-contract=off', '-fno-fast-math', '-Werror=write-strings')}


def _translate_freeform(manifest, units, project, config, captures, source_directory, mixed):
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
        snapshots = []
        for index, (name, data, _, _) in enumerate(units):
            if name not in captures:
                raise ValueError(f'no sealed Fortran type evidence for {name}')
            snapshot = sources / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(data)
            semantics = temporary / f'unit-{index}.fortran-original'
            semantics.write_bytes(read_artifact(captures[name]))
            snapshots.append({'unit': name, 'source': str(snapshot), 'semantics': str(semantics),
                              'passthrough': Path(name).suffix.lower() in _FIXED_SUFFIXES})
        request = {'entry': manifest.get('entry', 'main'), 'mixed': mixed, 'units': snapshots}
        arguments = temporary / 'lowering.json'
        arguments.write_text(json.dumps(request), encoding='utf-8')
        helper = Path(__file__).with_name('_fortran_freeform.py')
        lowered = json.loads(_run([sys.executable, '-B', str(helper), str(arguments)], temporary, env=env))
        legacy = [unit for unit in units if Path(unit[0]).suffix.lower() in _FIXED_SUFFIXES]
        transport = [(f'{name}.f', source.encode(), (), True) for name, source in lowered['sources'].items()]
        build = _translate_f2c([*legacy, *transport], project, config, source_directory=source_directory,
                               descriptor_units=lowered['descriptor_units'])
        return build, lowered['exports']


def prepare_fortran(manifest, project, source_bytes, *, source_directory='src', mixed=False, captures):
    """Return sealed transport inputs and declarations for the target ABI."""
    config = manifest.get('fortran', {})
    units = [(row['path'], source_bytes[row['path']], tuple(row.get('flags', ())), False)
             for row in manifest['units']]
    project = Path(project).resolve()
    project.mkdir(parents=True, exist_ok=True)
    exports = []
    if all(Path(name).suffix.lower() in _FIXED_SUFFIXES for name, _, _, _ in units):
        build = _translate_f2c(units, project, config, source_directory=source_directory)
        translation = 'f2c-cxx20'
    else:
        build, exports = _translate_freeform(manifest, units, project, config, captures,
                                            source_directory, mixed)
        translation = 'gfortran-typed-fparser-fixed-form-f2c-cxx20'
    digests = {name: hashlib.sha256(data).hexdigest() for name, data, _, _ in units}
    for row in exports:
        row['source_sha256'] = digests[row['unit']]
        for index, argument in enumerate(row['arguments']):
            argument['wrapper_name'] = f'arg{index}'
    wrapper = (Path(source_directory) / 'scar_c_abi.cpp').as_posix() if exports else None
    if wrapper:
        build['sources'].append(wrapper)
    recipe = {'schema': 'scar.fortran-emission', 'schema_version': 2, 'compiler': translation,
              'sources': [{'path': name, 'sha256': digest} for name, digest in digests.items()],
              'native_libraries': list(build['libraries']), 'c_exports': exports}
    (project / 'fortran-emission.json').write_text(json.dumps(recipe, indent=2) + '\n', encoding='utf-8')
    return build, wrapper, exports
