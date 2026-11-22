"""C-family source evidence and assembly of the admitted C++20 source revision."""

from collections import defaultdict
from ctypes import (CDLL, CFUNCTYPE, POINTER, Structure, byref, c_char_p, c_int,
                    c_uint, c_ulong, c_void_p)
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess

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


def capture_unit(source_root, unit, *, source_override=None):
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
        return dict(path=unit['path'], flags=unit['flags'], records=records, diagnostics=diagnostics,
                    frontend=dict(library=library, version=string(lib.clang_getClangVersion())))
    finally:
        if tu:
            lib.clang_disposeTranslationUnit(tu)
        lib.clang_disposeIndex(idx)


def _compile_settings(flags, source_root, staged_root):
    include_dirs = []
    compile_options = []
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


def _c_spans(units, root, edits, preserve_abi, captures):
    declarations = defaultdict(list)
    conversions = defaultdict(dict)
    for unit in units:
        name = unit['path']
        ref = captures.get(name)
        if ref is not None and name not in edits:
            typed = json.loads(read_artifact(ref, json_document=False))
            if typed['path'] != name or typed['flags'] != unit['flags']:
                raise ValueError(f'C type evidence has a different source or profile: {name}')
        else:
            typed = capture_unit(root, unit, source_override=edits.get(name))
        records = typed['records']
        for record in records:
            path = record['path']
            if path != name and not preserve_abi:
                continue
            if (preserve_abi and record['external_linkage']
                    and record['kind'] in ('FunctionDecl', 'VarDecl')):
                declarations[path].append(record)
            parent = record['parent']
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
    return declarations, conversions


_TRIVIA = re.compile(rb'(?:\s+|/\*[\s\S]*?\*/|//[^\n]*\n)*')
_C_IDENTIFIER = re.compile(rb'[A-Za-z_]\w*')


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


def _lower_c(path, data, declarations, conversions):
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
    position = 0
    for offset, tokens in sorted(insertions.items()):
        chunks.extend((data[position:offset], *tokens))
        position = offset
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

    c_units = [units[name] for name, language in languages.items() if language == 'c']
    c_root = Path('src/c-abi') if mixed else Path('src')
    cxx_root = Path('src/cxx') if mixed else Path('src')
    decls, casts = _c_spans(c_units, root, edits, mixed, captures or {})
    if (decls.keys() | casts.keys()) - material.keys():
        raise ValueError('C translation referenced project files missing from the source lock')

    settings, translation = {}, []
    for name, lock in material.items():
        language = languages.get(name)
        data = _locked_source(root, name, lock)
        data = edits.get(name, data)
        if language == 'c' or (language is None and mixed and c_units):
            output = c_root / name
            if language == 'c':
                output = output.with_suffix('.cpp')
                settings[output.as_posix()] = _compile_settings(units[name]['flags'], root, c_root)
            converted, conversions, linkage = _lower_c(name, data, decls[name], casts[name])
            _stage(project, output, converted)
            if language == 'c' or conversions or linkage:
                translation.append(dict(source=name, output=str(output),
                                        c_pointer_conversions=conversions, c_linkage=linkage))
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
    languages = {unit['path']: unit.get('language', manifest['language']) for unit in manifest['units']}
    if len(set(languages.values())) <= 1:
        return []
    links = read_artifact(calls)['calls']
    targets = sorted({(row['caller']['unit'], row['callee']['unit'], row['linkage_symbol'])
                      for row in links if row['status'] == 'BOUND'
                      and languages[row['caller']['unit']] != languages[row['callee']['unit']]})
    symbols = {line.split()[0] for line in subprocess.check_output(
        ['nm', '-P', '--defined-only', str(executable)], text=True, timeout=30).splitlines() if line.split()}
    for owner, target, symbol in targets:
        if symbol not in symbols:
            raise ValueError(f'mixed linkage symbol {symbol} is absent after building {owner} -> {target}')
    return [dict(caller=owner, callee=target, symbol=symbol) for owner, target, symbol in targets]
