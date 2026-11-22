"""Clang C API adapter for source ranges, symbols and native language types."""

from ctypes import (CDLL, CFUNCTYPE, POINTER, Structure, byref, c_char_p, c_int,
                    c_uint, c_ulong, c_void_p)
from functools import lru_cache
from pathlib import Path


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


@lru_cache(maxsize=1)
def _library():
    lib = CDLL('libclang-18.so.18')
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
    lib = _library()
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
        return dict(path=unit['path'], flags=unit['flags'], records=records, diagnostics=diagnostics)
    finally:
        if tu:
            lib.clang_disposeTranslationUnit(tu)
        lib.clang_disposeIndex(idx)
