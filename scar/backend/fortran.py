"""Translate sealed fixed-form Fortran transport units into C++20 sources."""

from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

from ..contract import Member, read_artifact


def _run(args, cwd):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=180)
    if result.returncode not in (0, 4):
        raise ValueError(f'{Path(args[0]).name} failed with status {result.returncode}\n{result.stderr.strip()}')


def _translator(path):
    if path is None:
        path = shutil.which('f2c')
    if type(path) is not str or not Path(path).is_absolute() or not Path(path).is_file():
        raise ValueError('Fortran transport requires an absolute f2c path or f2c on PATH')
    return str(Path(path).resolve())


_POINTER_ADJUSTMENTS = '/* Parameter adjustments */'
_FUNCTION_BODY = '/* Function Body */'
_CPP_NONCODE = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\[\s\S]|[^"\\])*"|\'(?:\\[\s\S]|[^\'\\])*\'')
_POINTER_SHIFT = re.compile(r'^\s*(?:--(\w+)|(\w+)\s*-=\s*(\w+))\s*;', re.MULTILINE)
_INDIRECT_CALL = re.compile(r'\(\s*\*\s*(\w+)\s*\)\s*\(\s*(?!\s*\))')
_DESCRIPTOR_ARRAY = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')


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
    # Literal arguments to libf2c's STOP and output calls are read-only.
    prefix = (r'\b(?:s_stop\s*\(\s*|do_fio\s*\(\s*&\w+\s*,\s*'
              r'|do_lio\s*\(\s*&\w+\s*,\s*&\w+\s*,\s*)')
    for match in re.finditer(prefix + literal + r'\s*,', source):
        edits.append((*match.span(1), f'const_cast<char*>({match[1]})'))
    return '#include "scar_f2c_call.hpp"\n' + _apply_spans(source, edits)


def _output_path(project, name):
    if type(name) is not str:
        raise ValueError('invalid Fortran transport output')
    output = Path(name)
    if (output.is_absolute() or output.as_posix() != name or len(output.parts) < 2
            or output.parts[0] != 'src' or '..' in output.parts or output.suffix != '.cpp'):
        raise ValueError(f'Fortran transport output must be a relative C++ source path: {name}')
    path = project / output
    if path.resolve() != path:
        raise ValueError(f'Fortran transport output has a symbolic link: {name}')
    return path


def emit_fortran_transport(transport, project, bundle) -> tuple[str, ...]:
    """Run f2c on plan-owned inputs and publish its repaired C++20 sources."""
    project = Path(project).resolve(strict=True)
    translator = _translator(transport['translator'])
    outputs, seen = [], set()
    for unit in transport['units']:
        dest = _output_path(project, unit['output'])
        if dest in seen:
            raise ValueError(f'repeated Fortran transport output: {unit["output"]}')
        seen.add(dest)
        arrays = unit['descriptor_arrays']
        if (type(unit['extended']) is not bool or type(arrays) not in (tuple, list)
                or any(type(array) is not str or not _DESCRIPTOR_ARRAY.fullmatch(array) for array in arrays)
                or (arrays and not unit['extended'])):
            raise ValueError('invalid Fortran transport descriptor arrays')
        outputs.append(dest)

    generated = project / 'include' / 'scar_generated' / 'f2c.h'
    if generated.resolve() != generated:
        raise ValueError('Fortran runtime header destination has a symbolic link')
    contents = read_artifact(Member(**transport['header']).resolve(bundle)).decode('utf-8')
    contents = contents.replace('\r\n', '\n').replace('\r', '\n')
    # The external runtime only reads a supplied format. Const qualification
    # keeps its ABI and permits strict C++20 string literal initialization.
    contents = re.sub(r'\bchar\s*\*\s*cifmt\b', 'const char *cifmt', contents)
    with TemporaryDirectory(prefix='.f2c-', dir=project) as temporary:
        staging = Path(temporary)
        inputs = {}
        cohorts = {False: [], True: []}
        for index, unit in enumerate(transport['units']):
            filename = f'fortran-{index}.f'
            (staging / filename).write_bytes(read_artifact(Member(**unit['source']).resolve(bundle)))
            inputs[filename] = unit['descriptor_arrays']
            cohorts[unit['extended']].append(filename)

        prototypes = {}
        for _ in range(4):
            for extended, members in cohorts.items():
                if members:
                    options = ['-C++', '-Ps', '-g', *(['-f'] if extended else [])]
                    _run([translator, *options, *prototypes, *members], staging)
            inferred = {path.name: path.read_bytes() for path in sorted(staging.glob('*.P'))}
            if inferred == prototypes and inferred:
                break
            prototypes = inferred
        else:
            raise ValueError('f2c could not stabilize compiler-generated prototypes')

        generated.parent.mkdir(parents=True, exist_ok=True)
        with generated.open('x', encoding='utf-8') as stream:
            stream.write(contents)
        for (filename, descriptor_arrays), dest in zip(inputs.items(), outputs, strict=True):
            translated = (staging / filename).with_suffix('.c').read_text(encoding='utf-8')
            repaired = _repair_f2c_source(translated, descriptor_arrays)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('x', encoding='utf-8') as stream:
                stream.write(repaired)
    return tuple(unit['output'] for unit in transport['units'])
