"""Capture typed C-family source evidence with original byte coordinates."""

from collections import defaultdict
from contextlib import contextmanager
from ctypes import (CDLL, CFUNCTYPE, POINTER, Structure, byref, c_char_p, c_int,
                    c_uint, c_void_p)
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

from scar.contract import artifact, read_artifact


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


_Visitor = CFUNCTYPE(c_uint, _Cursor, _Cursor, c_void_p)
_CALLABLES = {'FunctionDecl', 'CXXMethod', 'Constructor', 'Destructor', 'ConversionFunction'}
_CX_LINKAGE_EXTERNAL = 4
_GNU_CFI_HEADER = 'ISO_Fortran_binding.h'
_LOOP_TYPES = {'int': 'i32', 'double': 'f64',
               'double *': 'ptr<f64>', 'const double *': 'ptr<const f64>'}


def counted_function_bodies(document):
    """Locate closed void bodies whose source statements can match counted SSA loops."""
    records, parents = document['records'], defaultdict(list)
    for record in records:
        if record['path'] == document['path']:
            parents[record['parent']].append(record)
    bodies = {}
    for index, declaration in enumerate(records):
        if (declaration['path'] != document['path'] or declaration['kind'] != 'FunctionDecl'
                or not declaration['definition'] or not declaration['symbol']
                or not declaration['canonical_type'].startswith('void (')):
            continue
        children = parents[index]
        arguments = [row for row in children if row['kind'] == 'ParmDecl']
        compounds = [row for row in children if row['kind'] == 'CompoundStmt']
        if (len(compounds) != 1 or not compounds[0]['span']
                or any(not row['name'] or row['canonical_type'] not in _LOOP_TYPES for row in arguments)):
            continue
        bodies[declaration['symbol']] = {
            'span': compounds[0]['span'],
            'parameters': [{'name': row['name'], 'type': _LOOP_TYPES[row['canonical_type']]} for row in arguments],
        }
    return bodies


def typed_call_occurrences(document):
    """Index explicit calls by their observed source spelling and compiler declaration."""
    calls = defaultdict(set)
    for row in document['records']:
        target = row['reference']
        if row['kind'] == 'CallExpr' and target and target.get('symbol') and row.get('call_range'):
            start_line, start_column, end_line, end_column = row['call_range']
            for line in range(start_line, end_line + 1):
                calls[row['path'], line, target['name']].add((
                    target['symbol'], target['usr'], target['canonical_type'],
                    start_column if line == start_line else 0,
                    end_column if line == end_line else None,
                ))
    return calls


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
    ref = artifact(header)
    with TemporaryDirectory(prefix='scar-gnu-cfi-include-') as include:
        snapshot = Path(include) / _GNU_CFI_HEADER
        # Override the Flang binding without changing other resource headers.
        snapshot.write_bytes(read_artifact(ref))
        yield _BindingHeader(snapshot, header, ref.sha256)


def capture_unit(source_root, unit, *, cfi=None):
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
                begin_path, start_line, start_column, begin = location(lib.clang_getRangeStart(span))
                end_path, end_line, end_column, end = location(lib.clang_getRangeEnd(span))
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
                    if kind == 'CallExpr' and reference_kind in _CALLABLES:
                        target_type = lib.clang_getCanonicalType(lib.clang_getCursorType(reference))
                        target['canonical_type'] = string(lib.clang_getTypeSpelling(target_type))
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
                if kind == 'CallExpr' and begin_path == end_path == path:
                    records[-1]['call_range'] = [start_line, start_column, end_line, end_column]
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
    idx, tu = lib.clang_createIndex(0, 0), c_void_p()
    try:
        error = lib.clang_parseTranslationUnit2(idx, str(source).encode(), argv, len(argv),
                                                None, 0, 0, byref(tu))
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


@lru_cache(maxsize=8)
def gnu_cfi_header(compiler):
    try:
        location = subprocess.check_output([compiler, '-print-file-name=include/' + _GNU_CFI_HEADER],
                                           text=True, timeout=20).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    header = Path(location)
    if not header.is_absolute() or not header.is_file():
        return None
    return header.resolve(strict=True)


def scalar_declarations(native, compiler_types, symbols):
    accepted = {}
    for unit in native["units"]:
        if unit.get("language", native["language"]) not in {"c", "c++"}:
            continue
        types = json.loads(read_artifact(compiler_types[unit["path"]]))
        for row in types["records"]:
            if row["kind"] != "FunctionDecl" or row["symbol"] not in symbols:
                continue
            match = re.fullmatch(r"double \(((?:double(?:, double)*)?)\)", row["canonical_type"])
            arity = match[1].count("double") if match else None
            if accepted.setdefault(row["symbol"], arity) != arity:
                raise ValueError(f"inconsistent native declarations for {row['symbol']}")
    return accepted
