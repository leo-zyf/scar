"""F1 C/C++ input: one compiler invocation, its captured source buffers and typed AST.

The translation-unit owner keeps libclang's source manager alive. Accessors query the
captured buffers and AST; none reopen a translation unit or consult source paths.
"""

import ast
import ctypes
from ctypes.util import find_library
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import math
import os
from pathlib import Path
from types import MappingProxyType


@lru_cache(maxsize=1)
def _clang():
    from clang import cindex

    if not cindex.Config.loaded:
        library = os.environ.get("SCAR_LIBCLANG") or find_library("clang-18")
        if library:
            cindex.Config.set_library_file(library)
    lib = cindex.conf.lib
    # These semantic queries are in libclang 18 but absent from its Python wrapper.
    extensions = {
        "clang_getClangVersion": (cindex._CXString, []),
        "clang_getFileContents": (
            ctypes.c_void_p,
            [cindex.TranslationUnit, cindex.File, ctypes.POINTER(ctypes.c_size_t)],
        ),
        "clang_Cursor_getVarDeclInitializer": (cindex.Cursor, [cindex.Cursor]),
        "clang_getCursorBinaryOperatorKind": (ctypes.c_int, [cindex.Cursor]),
        "clang_getCursorUnaryOperatorKind": (ctypes.c_int, [cindex.Cursor]),
        "clang_getBinaryOperatorKindSpelling": (cindex._CXString, [ctypes.c_int]),
        "clang_Cursor_Evaluate": (ctypes.c_void_p, [cindex.Cursor]),
        "clang_EvalResult_getKind": (ctypes.c_int, [ctypes.c_void_p]),
        "clang_EvalResult_isUnsignedInt": (ctypes.c_uint, [ctypes.c_void_p]),
        "clang_EvalResult_getAsUnsigned": (ctypes.c_ulonglong, [ctypes.c_void_p]),
        "clang_EvalResult_getAsLongLong": (ctypes.c_longlong, [ctypes.c_void_p]),
        "clang_EvalResult_getAsDouble": (ctypes.c_double, [ctypes.c_void_p]),
        "clang_EvalResult_dispose": (None, [ctypes.c_void_p]),
        "clang_getPresumedLocation": (
            None,
            [
                cindex.SourceLocation,
                ctypes.POINTER(cindex._CXString),
                ctypes.POINTER(ctypes.c_uint),
                ctypes.POINTER(ctypes.c_uint),
            ],
        ),
    }
    for name, (result, args) in extensions.items():
        function = getattr(lib, name)
        function.argtypes = args
        function.restype = result
    lib.clang_Cursor_getVarDeclInitializer.errcheck = cindex.Cursor.from_result
    lib.clang_getClangVersion.errcheck = cindex._CXString.from_result
    lib.clang_getBinaryOperatorKindSpelling.errcheck = cindex._CXString.from_result
    return cindex


def cursor_kind(cursor):
    try:
        return cursor.kind.name
    except ValueError:
        return f"CURSOR_{cursor._kind_id}"


def type_kind(type_):
    try:
        return type_.kind.name
    except ValueError:
        return f"TYPE_{type_._kind_id}"


def is_attribute(cursor):
    return _clang().conf.lib.clang_isAttribute(ctypes.c_uint(cursor._kind_id))


def is_expression(cursor):
    return _clang().conf.lib.clang_isExpression(ctypes.c_uint(cursor._kind_id))


def element_type(type_):
    return _clang().conf.lib.clang_getElementType(type_)


def function_parameter_types(type_):
    lib = _clang().conf.lib
    return (
        lib.clang_getArgType(type_, i) for i in range(lib.clang_getNumArgTypes(type_))
    )


@dataclass(frozen=True, slots=True)
class SourceToken:
    spelling: str
    offset: int


def _project_name(root, filename):
    real = (root / filename).resolve()
    return real.relative_to(root).as_posix() if real.is_relative_to(root) else None


def _language(filename, flags):
    mode = None
    for i, arg in enumerate(flags):
        if arg == "-x" and i + 1 < len(flags):
            mode = flags[i + 1]
        elif arg.startswith("-x") and len(arg) > 2:
            mode = arg[2:]
    if mode and mode != "none":
        if mode in {"c", "c-header", "cpp-output"}:
            return "c"
        if mode in {"c++", "c++-header", "c++-cpp-output"}:
            return "c++"
        return None
    suffix = Path(filename).suffix
    if suffix in {".c", ".i", ".h"}:
        return "c"
    if suffix in {
        ".C",
        ".cc",
        ".cp",
        ".cxx",
        ".cpp",
        ".CPP",
        ".c++",
        ".ii",
        ".H",
        ".hh",
        ".hpp",
        ".hxx",
    }:
        return "c++"
    return None


class ParsedUnit:
    """Read-only semantic input to F2; indexes and token cache are private to this unit."""

    def __init__(self, translation_unit, api, root, source_name, flags, source_hashes):
        self._translation_unit = translation_unit
        self._api = api
        self.source_name = source_name
        self.flags = tuple(flags)
        self.language = _language(source_name, flags)
        self.compiler_id = api.conf.lib.clang_getClangVersion()
        self._tokens = {}
        self._macros = {}
        self._files = {}
        digests, external_headers = {}, {}

        def register(filename):
            if filename not in self._files:
                relative = _project_name(root, filename)
                self._files[filename] = relative
                self._files.setdefault(os.path.normpath(filename), relative)

        def capture(file):
            filename = file.name
            register(filename)
            relative = self._files[filename]
            external = str((root / filename).resolve()) if relative is None else None
            if relative in digests or external in external_headers:
                return
            size = ctypes.c_size_t()
            address = api.conf.lib.clang_getFileContents(
                translation_unit, file, ctypes.byref(size)
            )
            if not address:
                if external is not None:
                    external_headers[external] = dict(sha256=None, size=None)
                    return
                raise ValueError(
                    f"Clang source buffer unavailable for project file {relative}"
                )
            # Hash the exact bytes used by Clang without a copy or second disk read.
            buffer = (ctypes.c_ubyte * size.value).from_address(address)
            actual = sha256(buffer).hexdigest()
            if external is not None:
                external_headers[external] = dict(sha256=actual, size=size.value)
                return
            if (
                source_hashes
                and relative in source_hashes
                and actual != source_hashes[relative]
            ):
                raise ValueError(
                    f"compiler source snapshot differs from input lock: {relative}"
                )
            digests[relative] = actual

        capture(translation_unit.get_file(source_name))
        for inclusion in translation_unit.get_includes():
            capture(inclusion.include)
        self.source_digests = MappingProxyType(digests)
        self.external_headers = MappingProxyType(external_headers)
        roots = []
        for cursor in translation_unit.cursor.get_children():
            kind = cursor_kind(cursor)
            if kind in {"INCLUSION_DIRECTIVE", "MACRO_DEFINITION"}:
                continue
            loc = cursor.location
            if loc.file:
                register(loc.file.name)
            if kind == "MACRO_INSTANTIATION":
                if self.owns(loc) and cursor.referenced:
                    self._macros[(loc.file.name, loc.offset)] = cursor.referenced
            elif not loc.file or self.owns(loc):
                roots.append(cursor)
        self.declarations = tuple(roots)
        self.diagnostics = tuple(
            {
                **self.location(item.location),
                "code": (
                    "c_family.parse"
                    if item.severity >= api.Diagnostic.Error
                    else "c_family.parse.ignored_attribute"
                ),
                "detail": item.spelling,
            }
            for item in translation_unit.diagnostics
            if item.severity >= api.Diagnostic.Error
            or (
                item.severity == api.Diagnostic.Warning
                and self.owns(item.location)
                and item.option in {"-Wunknown-attributes", "-Wignored-attributes"}
            )
        )

    def _project_file(self, filename):
        if filename in self._files:
            return self._files[filename]
        return self._files.get(os.path.normpath(filename))

    def owns(self, location):
        return bool(
            location.file and self._project_file(location.file.name) is not None
        )

    def location(self, loc):
        filename = loc.file.name if loc.file else self.source_name
        return {
            "file": self._project_file(filename) or filename,
            "line": loc.line,
            "column": loc.column,
        }

    def macro(self, cursor):
        loc = cursor.location
        return self._macros.get((loc.file.name, loc.offset)) if loc.file else None

    def tokens(self, cursor):
        # Clang tokenizes a source range. Its file/macro location identity is
        # valid for the lifetime of this translation unit only.
        extent = cursor.extent
        key = bytes(extent)
        result = self._tokens.get(key)
        if result is None:
            result = tuple(
                SourceToken(t.spelling, t.location.offset)
                for t in self._translation_unit.get_tokens(extent=extent)
            )
            self._tokens[key] = result
        return result

    def initializer(self, cursor):
        return self._api.conf.lib.clang_Cursor_getVarDeclInitializer(cursor)

    def template_pattern(self, cursor):
        return self._api.conf.lib.clang_getSpecializedCursorTemplate(cursor)

    def binary_operator(self, cursor):
        lib = self._api.conf.lib
        return lib.clang_getBinaryOperatorKindSpelling(
            lib.clang_getCursorBinaryOperatorKind(cursor)
        )

    def unary_opcode(self, cursor):
        return self._api.conf.lib.clang_getCursorUnaryOperatorKind(cursor)

    def presumed_filename(self, cursor):
        file = self._api._CXString()
        line, column = ctypes.c_uint(), ctypes.c_uint()
        self._api.conf.lib.clang_getPresumedLocation(
            cursor.location,
            ctypes.byref(file),
            ctypes.byref(line),
            ctypes.byref(column),
        )
        return self._api.conf.lib.clang_getCString(file)

    def number(self, cursor):
        lib = self._api.conf.lib
        evaluated = lib.clang_Cursor_Evaluate(cursor)
        if evaluated:
            try:
                kind = lib.clang_EvalResult_getKind(evaluated)
                if kind == 1:
                    if lib.clang_EvalResult_isUnsignedInt(evaluated):
                        return lib.clang_EvalResult_getAsUnsigned(evaluated)
                    return lib.clang_EvalResult_getAsLongLong(evaluated)
                if kind == 2 and type_kind(cursor.type.get_canonical()) in {
                    "FLOAT",
                    "DOUBLE",
                }:
                    value = lib.clang_EvalResult_getAsDouble(evaluated)
                    if math.isfinite(value):
                        return value
            finally:
                lib.clang_EvalResult_dispose(evaluated)
        return None

    @staticmethod
    def string(cursor):
        spelling = cursor.spelling
        if spelling.startswith('"'):
            try:
                raw = ast.literal_eval("b" + spelling.replace(r"\?", "?"))
                if (
                    isinstance(raw, bytes)
                    and len(raw) + 1 <= cursor.type.get_array_size()
                ):
                    return raw.decode("utf-8")
            except (ValueError, SyntaxError, UnicodeError):
                pass
        return None


def parse_unit(
    path: Path, flags: list[str], *, root: Path, language=None, source_hashes=None
) -> ParsedUnit:
    """Parse exactly one locked compilation profile without executing the input."""
    path, root = Path(path).resolve(), Path(root).resolve()
    relative = path.relative_to(root).as_posix()
    if language is not None and not any(s.startswith("-x") for s in flags):
        dialect = {"c": "c", "c++": "c++", "cpp": "c++", "cxx": "c++"}[language.lower()]
        flags = ["-x", dialect, *flags]
    api = _clang()
    ast_unit = api.Index.create().parse(
        relative,
        args=["-working-directory", str(root), *flags],
        options=api.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
    )
    return ParsedUnit(ast_unit, api, root, relative, flags, source_hashes)
