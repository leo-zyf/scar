"""Translate admitted Fortran sources and render standalone C++20 ABI bridges."""

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
_CALL_WRAPPER = '''#pragma once

template<class Result, class... Args>
static inline Result scar_f2c_call(Result (*callee)(...), Args... args) {
    return reinterpret_cast<Result (*)(Args...)>(callee)(args...);
}
'''
_CFI_ELEMENT = '''
#include <cstdint>

using longint = std::int64_t;

// The companion CFI ABI describes byte addresses of live scalar objects.
template<class T>
static inline T* scar_cfi_element(T* base, longint byte_offset) {
    return reinterpret_cast<T*>(reinterpret_cast<std::uintptr_t>(base) +
                                static_cast<std::uintptr_t>(byte_offset));
}
'''
_CFI_VIEW = '''static_assert(CFI_VERSION == 1);
static_assert(std::is_same_v<CFI_index_t, longint>);
static_assert(sizeof(std::uintptr_t) == sizeof(longint));

struct ScarCFIAxis {
    longint extent, stride, origin, lower, upper;
};

template<class T, std::size_t Rank>
struct ScarCFIView {
    T* data;
    std::array<ScarCFIAxis, Rank> axes;
};

template<class T, CFI_type_t Type, std::size_t Rank>
static ScarCFIView<T, Rank> scar_cfi_view(CFI_cdesc_t* descriptor,
                                        std::array<longint, Rank> origins) {
    static_assert(0 < Rank && Rank <= CFI_MAX_RANK);
    if (!descriptor || descriptor->version != CFI_VERSION || descriptor->rank != Rank ||
        descriptor->type != Type || descriptor->elem_len != sizeof(T) ||
        descriptor->attribute != CFI_attribute_other) std::abort();

    ScarCFIView<T, Rank> result{};
    result.data = static_cast<T*>(descriptor->base_addr);
    bool empty = false;
    for (std::size_t i = 0; i < Rank; ++i) {
        const CFI_dim_t& dim = descriptor->dim[i];
        if (dim.lower_bound != 0 || dim.extent < 0 || dim.sm % longint(sizeof(T))) std::abort();
        ScarCFIAxis& axis = result.axes[i];
        axis.extent = dim.extent;
        axis.stride = dim.sm;
        axis.origin = origins[i];
        if (dim.extent == 0) {
            empty = true;
            axis.lower = 1;
            axis.upper = 0;
        } else {
            axis.lower = axis.origin;
            if (__builtin_add_overflow(axis.origin, dim.extent - 1, &axis.upper)) std::abort();
        }
    }

    if (!empty) {
        longint min_offset = 0, max_offset = 0;
        for (const auto& axis : result.axes) {
            longint displacement;
            if ((axis.extent > 1 && axis.stride == 0) ||
                __builtin_mul_overflow(axis.extent - 1, axis.stride, &displacement)) std::abort();
            longint& sum = displacement < 0 ? min_offset : max_offset;
            if (__builtin_add_overflow(sum, displacement, &sum)) std::abort();
        }
        const auto base = reinterpret_cast<std::uintptr_t>(result.data);
        const auto available = (std::numeric_limits<std::uintptr_t>::max)() - base;
        const auto negative = std::uintptr_t(0) - static_cast<std::uintptr_t>(min_offset);
        // The caller establishes accessible backing objects and their lifetime.
        // Check the representable byte envelope before evaluating any element.
        if (!base || base % alignof(T) || negative > base || max_offset == (std::numeric_limits<longint>::max)() ||
            static_cast<std::uintptr_t>(max_offset) > available ||
            sizeof(T) - 1 > available - static_cast<std::uintptr_t>(max_offset)) std::abort();
    }
    return result;
}
'''
_CFI_PACKING = '''template<class T, std::size_t Rank>
class ScarCFIPacking {
public:
    explicit ScarCFIPacking(ScarCFIView<T, Rank>& view) {
        for (const auto& axis : view.axes) {
            if (axis.extent == 0) return;
        }

        // GNU requires the exact packed stride at every axis, even singletons.
        std::array<longint, Rank> packed_strides;
        longint bytes = sizeof(T);
        bool packed = true;
        for (std::size_t i = 0; i < Rank; ++i) {
            packed_strides[i] = bytes;
            packed &= view.axes[i].stride == bytes;
            if (__builtin_mul_overflow(bytes, view.axes[i].extent, &bytes)) std::abort();
        }
        if (packed) return;

        count = bytes / sizeof(T);
        buffer.reset(new (std::nothrow) T[count]);
        if (!buffer) std::abort();
        source = view;
        // GNU's BIND(C) prologue copies the bytes in for all INTENT modes.
        transfer<true>();
        view.data = buffer.get();
        for (std::size_t i = 0; i < Rank; ++i) view.axes[i].stride = packed_strides[i];
    }

    void copy_back() {
        if (buffer) transfer<false>();
    }

private:
    template<bool CopyIn>
    void transfer() {
        for (longint i = 0; i < count; ++i) {
            longint linear = i;
            longint offset = 0;
            for (const auto& axis : source.axes) {
                offset += (linear % axis.extent) * axis.stride;
                linear /= axis.extent;
            }
            T* element = scar_cfi_element(source.data, offset);
            T* packed = buffer.get() + i;
            if constexpr (CopyIn) std::memcpy(packed, element, sizeof(T));
            else std::memcpy(element, packed, sizeof(T));
        }
    }

    ScarCFIView<T, Rank> source;
    std::unique_ptr<T[]> buffer;
    longint count = 0;
};
'''
_CFI_AXIS_FIELDS = ('extent', 'stride', 'origin', 'lower', 'upper')


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
    wrapper = _CALL_WRAPPER + (_CFI_ELEMENT if descriptor_units else '')
    (generated / 'scar_f2c_call.hpp').write_text(wrapper, encoding='utf-8')

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

    return {'sources': sources, 'libraries': (str(archive), 'm'),
            'compile_options': ('-ffp-contract=off', '-fno-fast-math', '-Werror=write-strings')}


def _c_abi_source(exports):
    descriptors = [arg['descriptor'] for row in exports for arg in row['arguments'] if 'descriptor' in arg]
    has_packing = any(row.get('contiguous') for row in descriptors)
    lines = ['#include <type_traits>']
    if descriptors:
        lines.extend(['#include <ISO_Fortran_binding.h>', '#include <array>', '#include <cstdlib>',
                      '#include <limits>', '#include "scar_f2c_call.hpp"'])
    if has_packing:
        lines.extend(['#include <cstring>', '#include <memory>', '#include <new>'])
    lines.extend(['#include "f2c.h"', 'static_assert(std::is_same_v<integer, int>);',
                  'static_assert(std::is_same_v<doublereal, double>);'])
    if any(argument['transport_type'] == 'real' for row in exports for argument in row['arguments']):
        lines.append('static_assert(std::is_same_v<real, float>);')
    if descriptors:
        lines.append(_CFI_VIEW)
    if has_packing:
        lines.append(_CFI_PACKING)
    for row in exports:
        result = row['result']
        native_return = result['c_type'] if result else 'void'
        transport_return = result['transport_type'] if result else 'int'
        types, parameters, arguments, checks = [], [], [], []
        packings, copy_backs = [], []
        for index, argument in enumerate(row['arguments']):
            name = f'arg{index}'
            types.append(f'{argument["transport_type"]} *')
            if descriptor := argument.get('descriptor'):
                parameters.append(f'CFI_cdesc_t *{name}')
                view = f'view{index}'
                lower = ', '.join(map(str, descriptor['lower_bounds']))
                template = f'{argument["c_type"]}, {descriptor["cfi_type"]}, {descriptor["rank"]}'
                checks.append(f'    auto {view} = scar_cfi_view<{template}>({name}, {{{lower}}});')
                if descriptor.get('contiguous'):
                    packing = f'packing{index}'
                    packings.append(f'    ScarCFIPacking {packing}({view});')
                    if descriptor['intent'] != 'in':
                        copy_backs.append(f'    {packing}.copy_back();')
                arguments.append(f'{view}.data')
                for axis in range(descriptor['rank']):
                    for field in _CFI_AXIS_FIELDS:
                        types.append('longint *')
                        arguments.append(f'&{view}.axes[{axis}].{field}')
            elif argument['by_value']:
                parameters.append(f'{argument["c_type"]} {name}')
                arguments.append('&' + name)
            elif argument['read_only']:
                parameters.append(f'const {argument["c_type"]} *{name}')
                arguments.append(f'const_cast<{argument["c_type"]} *>({name})')
            else:
                parameters.append(f'{argument["c_type"]} *{name}')
                arguments.append(name)
        lines.append(f'extern "C" {transport_return} {row["transport"]}({", ".join(types)});')
        lines.append(f'extern "C" {native_return} {row["symbol"]}({", ".join(parameters)}) {{')
        lines.extend(checks)
        lines.extend(packings)
        call = f'{row["transport"]}({", ".join(arguments)});'
        if result and copy_backs:
            lines.append('    const auto result = ' + call)
            lines.extend(copy_backs)
            lines.append('    return result;')
        else:
            lines.append('    ' + ('return ' if result else '') + call)
            lines.extend(copy_backs)
        lines.append('}')
    return '\n'.join(lines) + '\n'


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
            snapshots.append({'source': str(snapshot), 'semantics': str(semantics),
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
        exports = lowered['exports']
        if exports:
            wrapper = Path(source_directory) / 'scar_c_abi.cpp'
            (project / wrapper).write_text(_c_abi_source(exports), encoding='utf-8')
            build['sources'].append(wrapper.as_posix())
        if lowered['descriptor_units']:
            build['gnu_cfi_descriptors'] = True
        return build, exports


def emit_fortran(manifest, project, source_bytes, *, source_directory='src', mixed=False, captures):
    """Emit standalone C++20 host sources. Return shared CMake writer arguments."""
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
    recipe = {'schema': 'scar.fortran-emission', 'schema_version': 1, 'compiler': translation,
              'sources': [{'path': name, 'sha256': hashlib.sha256(data).hexdigest()}
                          for name, data, _, _ in units],
              'native_libraries': list(build['libraries']), 'c_exports': exports}
    (project / 'fortran-emission.json').write_text(json.dumps(recipe, indent=2) + '\n', encoding='utf-8')
    return build
