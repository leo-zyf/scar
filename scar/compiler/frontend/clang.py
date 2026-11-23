"""C-family source evidence and assembly of the admitted C++20 source revision."""

from collections import defaultdict
from contextlib import contextmanager
from ctypes import (CDLL, CFUNCTYPE, POINTER, Structure, byref, c_char_p, c_int,
                    c_uint, c_ulong, c_void_p)
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

from scar.analysis.replay_transaction import pinned_file_reference
from scar.ir.execution_recipe import read_artifact


class _String(Structure):
    _fields_ = [('data', c_void_p), ('flags', c_uint)]


class _Cursor(Structure):
    _fields_ = [('kind', c_uint), ('data_index', c_int), ('data', c_void_p * 3)]


class _Type(Structure):
    _fields_ = [('kind', c_uint), ('data', c_void_p * 2)]


class _Location(Structure):
    _fields_ = [('data', c_void_p * 2), ('index', c_uint)]


class _Range(Structure):
    _fields_ = [('data', c_void_p * 2), ('begin', c_uint), ('end', c_uint)]


class _Unsaved(Structure):
    _fields_ = [('filename', c_char_p), ('contents', c_char_p), ('length', c_ulong)]


_Visitor = CFUNCTYPE(c_uint, _Cursor, _Cursor, c_void_p)
_CALLABLES = {'FunctionDecl', 'CXXMethod', 'Constructor', 'Destructor', 'ConversionFunction'}
_CX_LINKAGE_EXTERNAL = 4
_GNU_CFI_HEADER = 'ISO_Fortran_binding.h'
_GNU_CFI_ROUTINES = {'CFI_establish', 'CFI_is_contiguous', 'CFI_section', 'CFI_setpointer'}


@lru_cache(maxsize=8)
def _library(name):
    lib = CDLL(name)
    signatures = {
        'createIndex': (c_void_p, c_int, c_int),
        'disposeIndex': (None, c_void_p),
        'parseTranslationUnit2': (c_int, c_void_p, c_char_p, POINTER(c_char_p),
                                  c_int, c_void_p, c_uint, c_uint, POINTER(c_void_p)),
        'disposeTranslationUnit': (None, c_void_p),
        'getTranslationUnitCursor': (_Cursor, c_void_p),
        'visitChildren': (c_uint, _Cursor, _Visitor, c_void_p),
        'getCursorKindSpelling': (_String, c_uint),
        'getCursorSpelling': (_String, _Cursor),
        'getCursorUSR': (_String, _Cursor),
        'Cursor_getMangling': (_String, _Cursor),
        'getCursorType': (_Type, _Cursor),
        'getTypeSpelling': (_String, _Type),
        'getCanonicalType': (_Type, _Type),
        'getCursorReferenced': (_Cursor, _Cursor),
        'getCursorLinkage': (c_uint, _Cursor),
        'Cursor_isNull': (c_int, _Cursor),
        'isCursorDefinition': (c_uint, _Cursor),
        'getCursorExtent': (_Range, _Cursor),
        'getCursorLocation': (_Location, _Cursor),
        'getRangeStart': (_Location, _Range),
        'getRangeEnd': (_Location, _Range),
        'getExpansionLocation': (None, _Location, POINTER(c_void_p), POINTER(c_uint),
                                 POINTER(c_uint), POINTER(c_uint)),
        'getFileName': (_String, c_void_p),
        'getCString': (c_char_p, _String),
        'disposeString': (None, _String),
        'getClangVersion': (_String,),
        'getNumDiagnostics': (c_uint, c_void_p),
        'getDiagnostic': (c_void_p, c_void_p, c_uint),
        'getDiagnosticSeverity': (c_uint, c_void_p),
        'getDiagnosticSpelling': (_String, c_void_p),
        'disposeDiagnostic': (None, c_void_p),
    }
    for name, (result, *args) in signatures.items():
        pointer = getattr(lib, 'clang_' + name)
        pointer.restype = result
        pointer.argtypes = args
    return lib


@dataclass(frozen=True, slots=True)
class _BindingHeader:
    snapshot: Path
    origin: Path
    sha256: str


@contextmanager
def gnu_cfi_overlay(header):
    """Share one immutable CFI header across a batch of Clang parses."""
    if header is None:
        yield None
        return
    ref = pinned_file_reference(header, 'scar.gnu-cfi-header')
    with TemporaryDirectory(prefix='scar-gnu-cfi-include-') as include:
        snapshot = Path(include) / _GNU_CFI_HEADER
        # Override the Flang binding without changing other resource headers.
        snapshot.write_bytes(read_artifact(ref, json_document=False))
        yield _BindingHeader(snapshot, header, ref['sha256'])


def capture_unit(source_root, unit, *, source_override=None, cfi=None):
    """Return project-owned AST records, with byte ranges in original files.

    Parent indices refer to positions in records. Referenced declarations may
    be supplied by another translation unit or an external library.
    """
    source_root = Path(source_root).resolve()
    source = source_root / unit['path']
    library = unit.get('clang_library', 'libclang-18.so.18')
    lib = _library(library)
    records = []
    files = {}
    cfi_headers = defaultdict(set)

    def string(value):
        try:
            return (lib.clang_getCString(value) or b'').decode(errors='replace')
        finally:
            lib.clang_disposeString(value)

    def location(value):
        file, line, column, offset = c_void_p(), c_uint(), c_uint(), c_uint()
        lib.clang_getExpansionLocation(value, byref(file), byref(line), byref(column), byref(offset))
        if not file.value:
            return None, line.value, column.value, offset.value
        if file.value not in files:
            filename = Path(string(lib.clang_getFileName(file)))
            if not filename.is_absolute():
                filename = source_root / filename
            files[file.value] = filename.resolve()
        return files[file.value], line.value, column.value, offset.value

    def children(cursor, parent):
        error = None

        @_Visitor
        def visit(cur, _owner, _payload):
            nonlocal error
            try:
                path, line, column, offset = location(lib.clang_getCursorLocation(cur))
                if path is None or not path.is_relative_to(source_root):
                    return 1
                kind = string(lib.clang_getCursorKindSpelling(cur.kind))
                span = lib.clang_getCursorExtent(cur)
                begin_path, _, _, begin = location(lib.clang_getRangeStart(span))
                end_path, _, _, end = location(lib.clang_getRangeEnd(span))
                type_ = lib.clang_getCursorType(cur)
                reference = lib.clang_getCursorReferenced(cur)
                target = None
                if not lib.clang_Cursor_isNull(reference):
                    reference_kind = string(lib.clang_getCursorKindSpelling(reference.kind))
                    target = dict(usr=string(lib.clang_getCursorUSR(reference)),
                                  name=string(lib.clang_getCursorSpelling(reference)),
                                  kind=reference_kind,
                                  symbol=string(lib.clang_Cursor_getMangling(reference))
                                  if reference_kind in _CALLABLES else '')
                    if reference_kind == 'FunctionDecl' and target['symbol'].startswith('CFI_'):
                        header, _, _, _ = location(lib.clang_getCursorLocation(reference))
                        if cfi is not None and header == cfi.snapshot:
                            header = cfi.origin
                        if header and header.name == _GNU_CFI_HEADER and not header.is_relative_to(source_root):
                            cfi_headers[header].add(target['symbol'])
                definition = bool(lib.clang_isCursorDefinition(cur))
                index = len(records)
                symbol = ''
                if kind in _CALLABLES:
                    symbol = string(lib.clang_Cursor_getMangling(cur))
                records.append(dict(kind=kind, name=string(lib.clang_getCursorSpelling(cur)),
                                    usr=string(lib.clang_getCursorUSR(cur)), symbol=symbol,
                                    reference=target, definition=definition, parent=parent,
                                    external_linkage=(parent is None
                                                      and lib.clang_getCursorLinkage(cur) == _CX_LINKAGE_EXTERNAL),
                                    path=str(path.relative_to(source_root)), line=line, column=column,
                                    offset=offset, span=[begin, end] if begin_path == end_path == path else None,
                                    type=string(lib.clang_getTypeSpelling(type_)),
                                    canonical_type=string(lib.clang_getTypeSpelling(lib.clang_getCanonicalType(type_)))))
                children(cur, index)
            except Exception as exc:
                error = exc
                return 0
            return 1

        lib.clang_visitChildren(cursor, visit, None)
        if error is not None:
            raise error

    flags = [f'-working-directory={source_root}', *unit['flags']]
    if cfi is not None:
        flags.extend(('-isystem', str(cfi.snapshot.parent)))
    argv = (c_char_p * len(flags))(*(flag.encode() for flag in flags))
    overlay = (None if source_override is None else
               _Unsaved(str(source).encode(), source_override, len(source_override)))
    idx, tu = lib.clang_createIndex(0, 0), c_void_p()
    try:
        error = lib.clang_parseTranslationUnit2(idx, str(source).encode(), argv, len(argv),
                                                None if overlay is None else byref(overlay),
                                                int(overlay is not None), 0, byref(tu))
        if error:
            raise ValueError(f'libclang could not parse {source} (CXErrorCode {error})')
        diagnostics = []
        for i in range(lib.clang_getNumDiagnostics(tu)):
            diagnostic = lib.clang_getDiagnostic(tu, i)
            try:
                severity = lib.clang_getDiagnosticSeverity(diagnostic)
                diagnostics.append(dict(severity=severity,
                                        message=string(lib.clang_getDiagnosticSpelling(diagnostic))))
            finally:
                lib.clang_disposeDiagnostic(diagnostic)
        if any(row['severity'] >= 3 for row in diagnostics):
            raise ValueError(f'{source}: {diagnostics}')
        children(lib.clang_getTranslationUnitCursor(tu), None)
        document = dict(path=unit['path'], flags=unit['flags'], records=records, diagnostics=diagnostics,
                        frontend=dict(library=library, version=string(lib.clang_getClangVersion())))
        if cfi_headers:
            bindings = []
            for header, symbols in sorted(cfi_headers.items()):
                digest = cfi.sha256 if cfi is not None and header == cfi.origin else sha256(header.read_bytes()).hexdigest()
                bindings.append(dict(path=str(header), sha256=digest, symbols=sorted(symbols)))
            document['gnu_cfi'] = bindings
        return document
    finally:
        if tu:
            lib.clang_disposeTranslationUnit(tu)
        lib.clang_disposeIndex(idx)


def _compile_settings(flags, source_root, staged_root):
    include_dirs = []
    # The generated CMake Release profile defines NDEBUG independently of
    # the original source command. Apply the caller's macro flags after it.
    compile_options = ['-UNDEBUG']
    tokens = iter(flags)
    for flag in tokens:
        if flag.startswith(('-std=', '--std=')):
            continue
        if flag in ('-std', '--std'):
            if next(tokens, None) is None:
                raise ValueError('missing language standard in native compiler flags')
            continue
        if not flag.startswith('-I'):
            compile_options.append(flag)
            continue
        location = next(tokens, None) if flag == '-I' else flag[2:]
        if not location:
            raise ValueError('missing include directory in native compiler flags')
        include_dir = (source_root / location).resolve()
        if not include_dir.is_relative_to(source_root):
            raise ValueError(f'include directory is outside the locked source root: {location}')
        include_dirs.append((staged_root / include_dir.relative_to(source_root)).as_posix())
    return tuple(include_dirs), tuple(compile_options)


def _c_family_evidence(root, unit, edits, captures, cfi):
    name = unit['path']
    ref = captures.get(name)
    if ref is None or name in edits:
        return capture_unit(root, unit, source_override=edits.get(name), cfi=cfi)
    typed = json.loads(read_artifact(ref, json_document=False))
    if typed['path'] != name or typed['flags'] != unit['flags']:
        raise ValueError(f'C type evidence has a different source or profile: {name}')
    return typed


def _c_spans(units, root, edits, preserve_abi, captures, material, cfi):
    declarations = defaultdict(list)
    conversions = defaultdict(dict)
    definitions = defaultdict(dict)
    source_bytes = {}
    cfi_headers = []
    for unit in units:
        name = unit['path']
        typed = _c_family_evidence(root, unit, edits, captures, cfi)
        cfi_headers.extend(typed.get('gnu_cfi', ()))
        records = typed['records']
        functions, parameters, bodies = {}, defaultdict(list), {}
        for index, record in enumerate(records):
            path = record['path']
            if path != name and not preserve_abi:
                continue
            if (preserve_abi and record['external_linkage']
                    and record['kind'] in ('FunctionDecl', 'VarDecl')):
                declarations[path].append(record)
            parent = record['parent']
            if record['kind'] == 'FunctionDecl' and record['definition']:
                functions[index] = record
            elif parent in functions:
                if record['kind'] == 'ParmDecl':
                    parameters[parent].append(record)
                elif record['kind'] == 'CompoundStmt':
                    bodies[parent] = record
            if parent is None or record['canonical_type'] != 'void *' or not record['span']:
                continue
            conversion = records[parent]
            if (conversion['kind'] != 'UnexposedExpr' or conversion['path'] != path
                    or not conversion['canonical_type'].endswith(' *')
                    or conversion['canonical_type'] == 'void *'):
                continue
            span = tuple(record['span'])
            target = conversion['type']
            if conversions[path].setdefault(span, target) != target:
                raise ValueError(f'ambiguous C pointer conversion in {path} at byte {span[0]}')
        for index, arguments in parameters.items():
            if index not in bodies:
                continue
            function = functions[index]
            path = function['path']
            if path not in source_bytes:
                lock = material.get(path)
                data = _locked_source(root, path, lock) if lock is not None else (root / path).read_bytes()
                source_bytes[path] = edits.get(path, data)
            header = _c_identifier_definition(source_bytes[path], function, arguments, bodies[index])
            if header is not None:
                span, replacement, types = header
                if definitions[path].setdefault(span, (replacement, types)) != (replacement, types):
                    raise ValueError(f'ambiguous old-style C function definition. {path}:{function["line"]}')
    return declarations, conversions, definitions, source_bytes, cfi_headers


_TRIVIA = re.compile(rb'(?:\s+|/\*[\s\S]*?\*/|//[^\n]*\n)*')
_C_IDENTIFIER = re.compile(rb'[A-Za-z_]\w*')


def _c_identifier_definition(data, function, parameters, body):
    """Normalize a C identifier-list definition when its call ABI is unchanged."""
    token = _C_IDENTIFIER.match(data, function['offset'])
    if token is None:
        return None
    start = _TRIVIA.match(data, token.end()).end()
    if data[start:start + 1] != b'(':
        return None
    cursor = start + 1
    for index, parameter in enumerate(parameters):
        name = _C_IDENTIFIER.match(data, _TRIVIA.match(data, cursor).end())
        if name is None or name[0] != parameter['name'].encode():
            return None
        delimiter = _TRIVIA.match(data, name.end()).end()
        expected = b')' if index + 1 == len(parameters) else b','
        if data[delimiter:delimiter + 1] != expected:
            return None
        cursor = delimiter + 1

    location = f'{function["path"]}:{function["line"]}'
    if (token[0] != function['name'].encode() or not function['span'] or not body['span']
            or body['path'] != function['path']):
        raise ValueError(f'old-style C definition has no original function/body range. {location}')
    body_start = body['span'][0]
    if not function['span'][0] <= start < cursor <= body_start < function['span'][1]:
        raise ValueError(f'old-style C definition has inconsistent source ranges. {location}')

    # Clang's function type already records the default parameter promotions
    # for a definition written with an identifier list. ParmDecl records the
    # local type before those promotions. Only use an ANSI declarator if each
    # has the same type at both sides of the call boundary.
    types = [parameter['canonical_type'] for parameter in parameters]
    _, _, signature = function['canonical_type'].partition('(')
    if not signature.endswith(')') or signature[:-1].split(', ') != types:
        raise ValueError(f'old-style C parameter promotion or adjustment requires an ABI bridge. {location}')

    declared = []
    names_end = cursor
    for parameter in parameters:
        span = parameter['span']
        if not span or parameter['path'] != function['path']:
            raise ValueError(f'old-style C parameter has no original declaration range. {location}')
        begin, stop = span
        spelling = parameter['name'].encode()
        if start < begin < stop < names_end and data[begin:stop] == spelling and parameter['type'] == 'int':
            declared.append(b'int ' + spelling)
            continue
        name = _C_IDENTIFIER.match(data, parameter['offset'])
        end = _TRIVIA.match(data, stop).end()
        if (not name or name[0] != spelling or not cursor <= begin <= name.start() < name.end() <= stop <= end < body_start
                or _TRIVIA.match(data, cursor).end() != begin or data[end:end + 1] != b';'):
            raise ValueError(f'old-style C parameters need independent ordered declaration ranges. {location}')
        declared.append(data[begin:stop])
        cursor = end + 1
    if _TRIVIA.match(data, cursor).end() != body_start or data[body_start:body_start + 1] != b'{':
        raise ValueError(f'old-style C definition has an unsupported parameter preamble. {location}')
    replacement = b'(' + b', '.join(declared) + b')'
    replacement += b'\n' * (data[start:body_start].count(b'\n') - replacement.count(b'\n'))
    return (start, body_start), replacement, types


def _c_abi_spans(path, data, declarations):
    # Clang can emit one VarDecl per declarator in a comma-separated declaration.
    groups = {}
    for record in declarations:
        span = record['span']
        offset = record['offset']
        spelling = _C_IDENTIFIER.match(data, offset)
        # The token can be a macro alias. Use the cursor's full declaration
        # extent, then qualify the resulting external symbol at link time.
        if (not span or not spelling or not span[0] <= offset < spelling.end() <= span[1] <= len(data)):
            raise ValueError(f'C linkage declaration lacks a source range. {path}:{record["line"]}')
        start, stop = span
        body = record['kind'] == 'FunctionDecl' and record['definition']
        if start not in groups or stop > groups[start][0]:
            groups[start] = stop, body
    for start, (stop, body) in sorted(groups.items()):
        if body:
            if data[stop - 1:stop] != b'}':
                raise ValueError(f'unanchored C function body. {path}:{start}')
        elif data[stop - 1:stop] != b';':
            end = _TRIVIA.match(data, stop).end()
            if data[end:end + 1] != b';':
                raise ValueError(f'unanchored C declaration end. {path}:{start}')
            stop = end + 1
        yield start, stop


def _lower_c(path, data, declarations, conversions, definitions):
    linkage = list(_c_abi_spans(path, data, declarations))
    insertions = defaultdict(list)
    prior = 0
    for start, stop in linkage:
        if start < prior:
            raise ValueError(f'overlapping C linkage declarations in {path}')
        insertions[start].append(b'extern "C" { ')
        insertions[stop].append(b' }')
        prior = stop
    prior = 0
    provenance = []
    for (start, stop), target in sorted(conversions.items()):
        if not 0 <= start < stop <= len(data):
            continue
        if start < prior:
            raise ValueError(f'overlapping C pointer conversions in {path} at byte {start}')
        insertions[start].append(f'static_cast<{target}>('.encode())
        insertions[stop].insert(0, b')')
        prior = stop
        provenance.append({'span': [start, stop], 'from': 'void *', 'to': target})
    chunks = []
    replacements = {start: (stop, text) for (start, stop), (text, _) in definitions.items()}
    position = 0
    for offset in sorted(insertions.keys() | replacements.keys()):
        if offset < position:
            raise ValueError(f'overlapping C source edits in {path} at byte {offset}')
        chunks.extend((data[position:offset], *insertions[offset]))
        position = offset
        if offset in replacements:
            position, replacement = replacements[offset]
            chunks.append(replacement)
    chunks.append(data[position:])
    return b''.join(chunks), provenance, linkage


def _locked_source(root, name, lock):
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root) or '..' in Path(name).parts:
        raise ValueError(f'source path outside the locked root: {name}')
    data = path.read_bytes()
    if len(data) != lock['size_bytes'] or sha256(data).hexdigest() != lock['sha256']:
        raise ValueError(f'locked source changed: {name}')
    return data


def _stage(project, name, data):
    target = project / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(data)


@lru_cache(maxsize=8)
def _compiler_cfi_header(compiler):
    try:
        location = subprocess.check_output([compiler, '-print-file-name=include/' + _GNU_CFI_HEADER],
                                           text=True, timeout=20).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    header = Path(location)
    if not header.is_absolute() or not header.is_file():
        return None
    return header.resolve(strict=True)


def gnu_cfi_header(manifest):
    """Find the target Fortran toolchain's descriptor header, when available."""
    from .fortran import _tool

    try:
        compiler = _tool(manifest.get('fortran', {}), 'gfortran')
    except ValueError:
        return None
    return _compiler_cfi_header(compiler)


def _stage_gnu_cfi(header, project, headers):
    if header is None:
        raise ValueError('the Fortran toolchain must supply its GNU CFI binding header')
    ref = pinned_file_reference(header, 'scar.gnu-cfi-header')
    data = read_artifact(ref, json_document=False)
    for record in headers:
        if (record['path'], record['sha256']) != (ref['uri'], ref['sha256']):
            raise ValueError('the native call was typed against a different GNU CFI binding header')
    symbols = {symbol for record in headers for symbol in record['symbols']}
    unsupported = symbols - _GNU_CFI_ROUTINES
    if unsupported:
        raise ValueError(f'unsupported standalone GNU CFI routines: {sorted(unsupported)}')

    # It remains a toolchain header when sealed into the relocatable project.
    _stage(project, 'include/scar_generated/' + _GNU_CFI_HEADER, b'#pragma GCC system_header\n' + data)
    if not symbols:
        return None
    source = 'src/fortran-abi/scar_gnu_cfi.cpp'
    implementation = Path(__file__).resolve().parents[2] / 'native' / 'gnu_cfi.cpp'
    _stage(project, source, implementation.read_bytes())
    return source


def _stage_cxx_sources(manifest, project, edits, *, captures=None):
    """Translate each locked unit under its language profile, then assemble C++20."""
    profile = manifest['language']
    units = {unit['path']: unit for unit in manifest['units']}
    languages = {name: unit.get('language', profile) for name, unit in units.items()}
    material = manifest['source_lock']['files']
    if not units or not units.keys() <= material.keys():
        raise ValueError('every translation unit needs locked source bytes')
    if not edits.keys() <= units.keys() or any(not isinstance(value, bytes) for value in edits.values()):
        raise ValueError('source edits must provide byte replacements for translation units')
    mixed = len(set(languages.values())) > 1
    root = Path(manifest['source_root']).resolve()
    fortran = [units[name] for name, language in languages.items() if language == 'fortran']
    if not mixed and fortran:
        from .fortran import emit_fortran

        for name, lock in material.items():
            _locked_source(root, name, lock)
        return emit_fortran(manifest, project, edits, captures=captures)

    captures = captures or {}
    c_units = [units[name] for name, language in languages.items() if language == 'c']
    cxx_units = [units[name] for name, language in languages.items() if language == 'c++' and fortran]
    c_root = Path('src/c-abi') if mixed else Path('src')
    cxx_root = Path('src/cxx') if mixed else Path('src')
    cfi_header = gnu_cfi_header(manifest) if fortran else None
    recapture = any(unit['path'] not in captures or unit['path'] in edits for unit in (*c_units, *cxx_units))
    with gnu_cfi_overlay(cfi_header if recapture else None) as cfi:
        decls, casts, headers, c_bytes, cfi_headers = _c_spans(
            c_units, root, edits, mixed, captures, material, cfi)
        for unit in cxx_units:
            typed = _c_family_evidence(root, unit, edits, captures, cfi)
            cfi_headers.extend(typed.get('gnu_cfi', ()))
    if (decls.keys() | casts.keys() | headers.keys()) - material.keys():
        raise ValueError('C translation referenced project files missing from the source lock')

    settings, translation = {}, []
    for name, lock in material.items():
        language = languages.get(name)
        data = c_bytes.pop(name, None)
        if data is None:
            data = edits.get(name, _locked_source(root, name, lock))
        if language == 'c' or (language is None and mixed and c_units):
            output = c_root / name
            if language == 'c':
                output = output.with_suffix('.cpp')
                settings[output.as_posix()] = _compile_settings(units[name]['flags'], root, c_root)
            converted, conversions, linkage = _lower_c(name, data, decls[name], casts[name], headers[name])
            _stage(project, output, converted)
            if language == 'c' or conversions or linkage:
                row = dict(source=name, output=str(output), c_pointer_conversions=conversions, c_linkage=linkage)
                if headers[name]:
                    row['c_identifier_definitions'] = [dict(span=span, parameter_types=types)
                                                       for span, (_, types) in sorted(headers[name].items())]
                translation.append(row)
        if language in (None, 'c++'):
            output = cxx_root / name
            _stage(project, output, data)
            if language == 'c++':
                settings[output.as_posix()] = _compile_settings(units[name]['flags'], root, cxx_root)

    if translation:
        (project / 'translation.json').write_text(json.dumps(translation, indent=2) + '\n')
    fortran_libraries = ()
    if fortran:
        from .fortran import emit_fortran

        config = dict(manifest, units=fortran, language='fortran')
        fortran_paths = {unit['path'] for unit in fortran}
        bridge = emit_fortran(config, project, {name: edits[name] for name in edits.keys() & fortran_paths},
                              source_directory='src/fortran-abi', mixed=True,
                              captures={name: captures[name] for name in fortran_paths} if captures else None)
        settings.update({name: ((), bridge['compile_options']) for name in bridge['sources']})
        fortran_libraries = bridge['libraries']
        if cfi_headers or bridge.get('gnu_cfi_descriptors'):
            cfi_source = _stage_gnu_cfi(cfi_header, project, cfi_headers)
            if cfi_source:
                settings[cfi_source] = ((), ())

    profiles = (manifest, manifest.get('cxx', {})) if mixed else (
        manifest if c_units else manifest.get('cxx', {}),)
    linkage = {key: tuple(value for profile in profiles for value in profile.get(key, ()))
               for key in ('libraries', 'link_options', 'runtime_dirs')}
    linkage['libraries'] = (*linkage['libraries'], *fortran_libraries)
    common = set(settings.values())
    include_dirs, compile_options = next(iter(common)) if len(common) == 1 else ((), ())
    return dict(sources=tuple(settings), include_dirs=include_dirs, compile_options=compile_options,
                source_settings=settings if len(common) > 1 else {}, **linkage)


def stage_source_revision(manifest, project, source_edits=(), *, captures=None, plan, target_inputs=None):
    """Materialize and seal the admitted C++20 source revision."""
    from scar.codegen.native_project import publish_cpp20_inputs

    edits, previous = {}, {}
    root = Path(manifest['source_root'])
    for edit in reversed(source_edits):
        unit = edit['unit']
        if unit not in edits:
            edits[unit] = _locked_source(root, unit, manifest['source_lock']['files'][unit])
            previous[unit] = len(edits[unit])
        start, stop = edit['span']
        source = edits[unit]
        if (not 0 <= start < stop <= previous[unit]
                or sha256(source[start:stop]).hexdigest() != edit['original_sha256']):
            raise ValueError(f'source copy edit no longer matches {unit}')
        edits[unit] = source[:start] + edit['replacement'].encode('ascii') + source[stop:]
        previous[unit] = start
    project.mkdir()
    configuration = _stage_cxx_sources(manifest, project, edits, captures=captures)
    for key, members in (target_inputs or {}).items():
        if key not in ('include_dirs', 'libraries', 'runtime_dirs'):
            raise ValueError(f'unsupported C++20 target input: {key}')
        retained = configuration.get(key, ())
        configuration[key] = (*retained, *(member for member in members if member not in retained))
    return publish_cpp20_inputs(project, configuration, plan=plan)


def verify_mixed_linkage(manifest, calls, executable):
    """Check that bound cross-language targets have their original link symbols."""
    from ..contracts import native_bound_targets, read_native_metadata

    languages = {unit['path']: unit.get('language', manifest['language']) for unit in manifest['units']}
    if len(set(languages.values())) <= 1:
        return []
    links = read_native_metadata(calls)['calls']
    targets = sorted({(call['caller']['unit'], target['callee']['unit'], target['linkage_symbol'])
                      for call in links for target in native_bound_targets(call)
                      if languages[call['caller']['unit']] != languages[target['callee']['unit']]})
    symbols = {line.split()[0] for line in subprocess.check_output(
        ['nm', '-P', '--defined-only', str(executable)], text=True, timeout=30).splitlines() if line.split()}
    for owner, target, symbol in targets:
        if symbol not in symbols:
            raise ValueError(f'mixed linkage symbol {symbol} is absent after building {owner} -> {target}')
    return [dict(caller=owner, callee=target, symbol=symbol) for owner, target, symbol in targets]
