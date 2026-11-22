"""CPython source bytes, tokens and lossless syntax occurrences.

Each layer publishes its own typed document and independently validates its
upstream inputs. Lexer and parser warning capture use isolated process workers.
The F0 codec scope serializes its own warning-filter changes.
"""
from __future__ import annotations

import _ast
import ast
import base64
import binascii
from bisect import bisect_left, bisect_right
import builtins
from dataclasses import fields, replace
from functools import lru_cache
from hashlib import sha256
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import re
import sys
import sysconfig
from threading import RLock
import token
import tokenize
from types import UnionType
from typing import get_args, get_origin
import warnings

from scar.ir import literals
from scar.ir.literals import PythonLiteral
from scar.ir.program.source_bytes import (
    ExactOperator, NativeTokenAttribute, NativeTokenParameter, NewlineRewrite,
    SourceBytesDecoding, SourceBytesError, SourceBytesMaterial, SourceBytesProfile, SourceRawBytes,
    SourceToken, SourceTokenError, SourceTokensInput, SourceTokensMaterial, SourceTokensProfile,
    SourceTokenWarning, TokenCoverage, TokenExtent, TokenSpan, TokenTextRef, TokenType, TokenView,
)
from scar.ir.program.source_syntax import (
    AstClassSchema, AstFieldSchema, AstOccurrence, AstTypeSchema, AstValue,
    SourceSyntaxInput, SourceSyntaxMaterial, SourceSyntaxProfile, SyntaxFrontier,
)
from scar.ir.record_codec import decode, encode


def _digest(document):
    return sha256(json.dumps(document, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')).hexdigest()


def _warning_records(captured):
    return tuple(SourceTokenWarning(item.category.__module__, item.category.__qualname__,
        str(item.message), item.filename, item.lineno, item.line, item.source is not None) for item in captured)


_CODEC_WARNINGS_LOCK = RLock()


def _bytes_profile():
    return SourceBytesProfile(sys.implementation.name, tuple(sys.implementation.version),
        tuple(sys.version_info), sys.version, sys.hexversion, sys.implementation.cache_tag)


def _encoding_error(stage, error, original):
    obj = getattr(error, 'object', None)
    basis = None
    if type(obj) in (bytes, str):
        basis = ('RAW_INPUT' if type(original) is bytes else 'DECODED_TEXT') if obj == original else 'EXCEPTION_OBJECT'
    return SourceBytesError(stage, type(error).__name__, str(error), getattr(error, 'reason', None),
        getattr(error, 'encoding', None), getattr(error, 'start', None), getattr(error, 'end', None), basis,
        getattr(error, 'lineno', None), getattr(error, 'offset', None),
        getattr(error, 'end_lineno', None), getattr(error, 'end_offset', None))


def _interpret(raw):
    # Creation and fresh readers share this exact policy. Codec warnings are
    # outside V1's result schema and must not become ambient-filter exceptions.
    with _CODEC_WARNINGS_LOCK, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            encoding, _ = tokenize.detect_encoding(BytesIO(raw).readline)
        except (SyntaxError, UnicodeError, LookupError) as error:
            return SourceBytesDecoding('ENCODING_ERROR', None, None, None, _encoding_error('ENCODING', error, raw)), None
        try:
            text = raw.decode(encoding)
        except (UnicodeError, LookupError) as error:
            return SourceBytesDecoding('DECODE_ERROR', encoding, None, None, _encoding_error('DECODING', error, raw)), None
        try:
            digest = sha256(text.encode('utf-8', errors='strict')).hexdigest()
        except UnicodeEncodeError as error:
            return SourceBytesDecoding('DECODED_TEXT_ENCODING_ERROR', encoding, None, len(text),
                _encoding_error('DECODED_TEXT_ENCODING', error, text)), text
        return SourceBytesDecoding('DECODED', encoding, digest, len(text), None), text


def source_bytes_document(raw: bytes, *, path: str, namespace_root: str) -> dict:
    """Retain exactly one raw payload, including encoding/decoding failures.

    Paths are labels checked lexically, never opened or resolved. Decoding
    preserves original newline characters. No source is parsed or executed.
    """
    if type(raw) is not bytes:
        raise TypeError('F0 source material requires exact immutable bytes')
    decoding, _ = _interpret(raw)
    return encode(SourceBytesMaterial(path, namespace_root, _bytes_profile(),
        SourceRawBytes(base64.b64encode(raw).decode('ascii'), sha256(raw).hexdigest(), len(raw)), decoding))


def _require_bytes_record(document, kind):
    if type(document) is not dict or set(document) != {field.name for field in fields(kind)}:
        raise ValueError('closed exact F0 ' + kind.__name__ + ' fields required')


def _read_bytes_material(document):
    _require_bytes_record(document, SourceBytesMaterial)
    for name, kind in (('profile', SourceBytesProfile), ('raw', SourceRawBytes), ('decoding', SourceBytesDecoding)):
        _require_bytes_record(document[name], kind)
    if document['decoding']['error'] is not None:
        _require_bytes_record(document['decoding']['error'], SourceBytesError)
    material = decode(SourceBytesMaterial, document)
    if material.profile != _bytes_profile():
        raise ValueError('F0 encoding interpretation requires its exact current runtime profile')
    try:
        raw = base64.b64decode(material.raw.base64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError('invalid F0 base64 payload') from error
    if (base64.b64encode(raw).decode('ascii') != material.raw.base64
            or len(raw) != material.raw.size_bytes or sha256(raw).hexdigest() != material.raw.sha256):
        raise ValueError('F0 canonical base64/raw size/SHA differ')
    decoding, text = _interpret(raw)
    if decoding != material.decoding:
        raise ValueError('F0 decoding metadata differs from its original bytes/profile')
    return raw, text


def read_source_bytes(document) -> bytes:
    """Strictly replay F0 only, rejecting changed payload or interpretation."""
    return _read_bytes_material(document)[0]


def decoded_source_bytes(document) -> str | None:
    """Return actual decoded text, even if its strict UTF-8 digest failed.

    Encoding/detection failure returns None. A decoded-text encoding error is
    recorded separately and never upgraded to a successful decoding summary.
    """
    return _read_bytes_material(document)[1]


@lru_cache(maxsize=8)
def _native_file_digest(binary, device=None, inode=None):
    with binary.open('rb') as stream:
        info = os.fstat(stream.fileno())
        if device is not None and (info.st_dev, info.st_ino) != (device, inode):
            raise ValueError('CPython native binary differs from the mapped file')
        digest = sha256()
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
        if (info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('CPython native binary changed during hashing')
    return digest.hexdigest()


def _mapped_builtin(module):
    """Find the Linux file that owns the built-in module's C definition."""
    import ctypes

    module_def = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object)(
        ('PyModule_GetDef', ctypes.pythonapi))
    address = module_def(module)
    if not address:
        raise ValueError(f'{module.__name__} has no CPython module definition')
    try:
        with open('/proc/self/maps', 'rb') as maps:
            for line in maps:
                fields = line.rstrip(b'\n').split(None, 5)
                low, high = (int(bound, 16) for bound in fields[0].split(b'-'))
                if low <= address < high and len(fields) == 6 and fields[5].startswith(b'/'):
                    major, minor = (int(part, 16) for part in fields[3].split(b':'))
                    binary = Path(os.fsdecode(fields[5])).resolve(strict=True)
                    return binary, os.makedev(major, minor), int(fields[4])
    except OSError as error:
        raise ValueError(f'loaded binary for {module.__name__} is unavailable') from error
    raise ValueError(f'loaded binary for {module.__name__} is unavailable')


def native_module_receipt(module):
    origin = module.__spec__.origin
    device = inode = None
    if origin == 'built-in':
        if sysconfig.get_config_var('Py_ENABLE_SHARED'):
            binary, device, inode = _mapped_builtin(module)
        else:
            binary = Path(sys.executable).resolve(strict=True)
    elif type(origin) is str and Path(origin).is_file():
        binary = Path(origin).resolve(strict=True)
    else:
        raise ValueError(f'native binary for {module.__name__} is unavailable')
    return origin, str(binary), _native_file_digest(binary, device, inode)


@lru_cache(maxsize=1)
def _tokenizer_implementation():
    import _tokenize

    origin, binary, binary_sha = native_module_receipt(_tokenize)
    return (sha256(Path(tokenize.__file__).read_bytes()).hexdigest(),
        sha256(Path(token.__file__).read_bytes()).hexdigest(), origin, binary,
        binary_sha)


def _tokenizer_profile(f0):
    tokenizer_sha, token_sha, native_origin, native_binary, native_sha = _tokenizer_implementation()
    return SourceTokensProfile(decode(SourceBytesProfile, f0['profile']),
        'stdlib.tokenize.generate_tokens.extra_tokens', 'CRLF_AND_CR_TO_LF_NO_FINAL_NEWLINE_V1',
        'RECORD_ALL_ALWAYS_ISOLATED_V1', tokenizer_sha, token_sha, native_origin, native_binary, native_sha,
        tuple(TokenType(i, token.tok_name[i]) for i in range(token.N_TOKENS)),
        tuple(ExactOperator(text, code) for text, code in sorted(token.EXACT_TOKEN_TYPES.items())))


def _tokenizer_input(f0):
    return SourceTokensInput(_digest(f0), f0['path'], f0['namespace_root'], f0['raw']['sha256'],
        f0['decoding']['decoded_utf8_sha256'], f0['decoding']['status'])


def _view(original):
    parts, rewrites, cursor, length = [], [], 0, 0
    for match in re.finditer(r'\r\n?', original):
        fragment = original[cursor:match.start()]
        parts.extend((fragment, '\n'))
        length += len(fragment)
        rewrites.append(NewlineRewrite(TokenSpan(match.start(), match.end()), TokenSpan(length, length + 1)))
        length += 1
        cursor = match.end()
    parts.append(original[cursor:])
    text = ''.join(parts)
    raw = text.encode('utf-8')
    return text, TokenView(sha256(raw).hexdigest(), len(text), len(raw), tuple(rewrites))


def _parameter(value):
    if type(value) is tuple:
        return NativeTokenParameter('TUPLE', None, tuple(_parameter(x) for x in value))
    if type(value) is bytes:
        return NativeTokenParameter('BYTES_BASE64', base64.b64encode(value).decode('ascii'), ())
    kinds = {type(None): 'NONE', str: 'STR', int: 'INT', float: 'FLOAT', bool: 'BOOL'}
    if type(value) not in kinds:
        raise TypeError('unsupported native tokenizer diagnostic parameter')
    return NativeTokenParameter(kinds[type(value)], value, ())


def _native_diagnostic(error):
    names = ('msg', 'filename', 'lineno', 'offset', 'end_lineno', 'end_offset', 'text',
        'encoding', 'reason', 'start', 'end', 'object')
    return SourceTokenError(type(error).__module__, type(error).__qualname__, str(error),
        _parameter(error.args), tuple(NativeTokenAttribute(name, _parameter(getattr(error, name)))
            for name in names if hasattr(error, name)))


class _References:
    def __init__(self, text, view):
        self.text, self.view = text, view
        self.lines = [0] + [i + 1 for i, char in enumerate(text) if char == '\n']
        self.pool, self.intern = [], {}
        self.line_refs = {}
        self.line_objects = {}
        self.crlf_ends = [item.view_chars.end for item in view.newline_rewrites
            if item.original_chars.end - item.original_chars.start == 2]
        self.non_ascii, self.utf8_extra = [], []
        extra = 0
        for i, char in enumerate(text):
            if ord(char) >= 128:
                extra += len(char.encode('utf-8')) - 1
                self.non_ascii.append(i)
                self.utf8_extra.append(extra)

    def offset(self, position):
        row, column = position
        if not 1 <= row <= len(self.lines):
            return None
        start = self.lines[row - 1]
        end = self.lines[row] if row < len(self.lines) else len(self.text)
        return start + column if 0 <= column <= end - start else None

    def ref(self, value, span=None):
        if span is not None and self.text[span.start:span.end] == value:
            return TokenTextRef('VIEW_RANGE', span.start, span.end, None)
        if value not in self.intern:
            self.intern[value] = len(self.pool)
            self.pool.append(value)
        return TokenTextRef('INTERNED_LITERAL', None, None, self.intern[value])

    def utf8(self, offset):
        index = bisect_left(self.non_ascii, offset)
        return offset + (self.utf8_extra[index - 1] if index else 0)

    def source_token(self, item):
        a, b = self.offset(item.start), self.offset(item.end)
        span = TokenSpan(a, b) if a is not None and b is not None and a <= b else None
        if span is not None:
            extent = TokenExtent(span, TokenSpan(a + bisect_right(self.crlf_ends, a),
                b + bisect_right(self.crlf_ends, b)), TokenSpan(self.utf8(a), self.utf8(b)))
            kind = 'ZERO_WIDTH' if a == b else 'SOURCE'
        else:
            extent = None
            if item.type in (token.NEWLINE, token.NL) and item.string == '' and a == len(self.text):
                kind = 'SYNTHETIC_NEWLINE'
            elif item.type == token.ENDMARKER:
                kind = 'VIRTUAL_ENDMARKER'
            else:
                kind = 'VIRTUAL_COORDINATES'
        line_key = id(item.line)
        if line_key not in self.line_objects:
            if item.line not in self.line_refs:
                line_span = None
                if 1 <= item.start[0] <= len(self.lines) and 1 <= item.end[0] <= len(self.lines):
                    start = self.lines[item.start[0] - 1]
                    end = self.lines[item.end[0]] if item.end[0] < len(self.lines) else len(self.text)
                    line_span = TokenSpan(start, end)
                self.line_refs[item.line] = (item.line, self.ref(item.line, line_span))
            canonical, reference = self.line_refs[item.line]
            # The first field strongly retains this exact native string, so
            # its id cannot be recycled during this local replay.
            self.line_objects[line_key] = (item.line, canonical, reference)
        return SourceToken(item.type, item.exact_type, self.ref(item.string, span),
            self.line_objects[line_key][2], tuple(item.start), tuple(item.end), extent, kind)


def _coverage(rows, size):
    intervals = sorted((row.source_extent.view_chars.start, row.source_extent.view_chars.end)
        for row in rows if row.source_extent is not None and row.source_extent.view_chars.start < row.source_extent.view_chars.end)
    covered = []
    for start, end in intervals:
        if covered and start <= covered[-1].end:
            covered[-1] = TokenSpan(covered[-1].start, max(end, covered[-1].end))
        else:
            covered.append(TokenSpan(start, end))
    gaps, cursor = [], 0
    for span in covered:
        if cursor < span.start:
            gaps.append(TokenSpan(cursor, span.start))
        cursor = span.end
    if cursor < size:
        gaps.append(TokenSpan(cursor, size))
    return TokenCoverage(tuple(covered), tuple(gaps))


def _produce_tokens(f0, original, max_tokens, *, retain_items=False):
    if type(max_tokens) is not int or max_tokens < 0:
        raise ValueError('max_tokens must be an exact nonnegative integer')
    upstream, profile = _tokenizer_input(f0), _tokenizer_profile(f0)
    if f0['decoding']['status'] != 'DECODED':
        return SourceTokensMaterial(upstream, profile, max_tokens, 'UPSTREAM_DECODING_ERROR',
            False, None, (), (), None, None, ()), ()
    text, view = _view(original)
    refs, items, rows, failure = _References(text, view), [], [], None
    actual_eof = False
    status = 'BUDGET_STOPPED'
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        stream = tokenize.generate_tokens(StringIO(text).readline)
        while len(rows) < max_tokens:
            try:
                item = next(stream)
            except StopIteration:
                status = 'TOKENIZED'
                break
            except (tokenize.TokenError, SyntaxError, UnicodeError, LookupError) as error:
                failure, status = _native_diagnostic(error), 'TOKENIZER_ERROR'
                break
            if not 0 <= item.type < token.N_TOKENS or not 0 <= item.exact_type < token.N_TOKENS:
                raise ValueError('tokenizer emitted a nonterminal sentinel token')
            rows.append(refs.source_token(item))
            if retain_items:
                items.append(tokenize.TokenInfo(item.type, item.string, item.start, item.end,
                    refs.line_objects[id(item.line)][1]))
            if item.type == token.ENDMARKER:
                actual_eof = True
                try:
                    next(stream)
                except StopIteration:
                    status = 'TOKENIZED'
                else:
                    raise ValueError('tokenizer emitted an item after ENDMARKER')
                break
        stream.close()
    if status == 'TOKENIZED' and not actual_eof:
        raise ValueError('tokenizer exhausted without an actual ENDMARKER')
    return SourceTokensMaterial(upstream, profile, max_tokens, status, actual_eof, view,
        tuple(refs.pool), tuple(rows), _coverage(rows, len(text)), failure, _warning_records(captured)), tuple(items)


def source_tokens_document(f0_document, *, max_tokens=2_000_000) -> dict:
    """Lex a strict F0 document; no implicit downstream compiler stage runs."""
    return encode(_produce_tokens(f0_document, decoded_source_bytes(f0_document), max_tokens)[0])


def _verify_tokens(document, f0_document, original, *, retain_items=False):
    material = decode(SourceTokensMaterial, document)
    fresh, items = _produce_tokens(f0_document, original, material.max_tokens, retain_items=retain_items)
    if material != fresh:
        raise ValueError('F1 document differs from its original F0/profile/fresh tokenizer stream')
    return items


def read_source_tokens(document, f0_document) -> tuple[tokenize.TokenInfo, ...]:
    """Rebuild and compare every typed lexical field against its original F0."""
    return _verify_tokens(document, f0_document, decoded_source_bytes(f0_document), retain_items=True)


@lru_cache(maxsize=1)
def _registry():
    classes = {name: cls for name, cls in vars(_ast).items()
        if isinstance(cls, type) and issubclass(cls, ast.AST)}
    abstract = {cls for cls in classes.values()
        if any(cls in child.__bases__ for child in classes.values())}
    constructors = {name: cls for name, cls in classes.items() if cls not in abstract}
    return classes, constructors


def _type_schema(kind):
    origin = get_origin(kind)
    if origin is list:
        return AstTypeSchema('LIST', None, tuple(_type_schema(x) for x in get_args(kind)))
    if origin is UnionType:
        return AstTypeSchema('UNION', None, tuple(_type_schema(x) for x in get_args(kind)))
    if isinstance(kind, type) and issubclass(kind, ast.AST):
        return AstTypeSchema('AST_CLASS', kind.__name__, ())
    if kind in (str, int, object, type(None)):
        return AstTypeSchema('BUILTIN', kind.__name__, ())
    raise ValueError('unrecorded official AST field type')


@lru_cache(maxsize=1)
def _schema_and_digests():
    classes, constructors = _registry()
    schema = tuple(AstClassSchema(name, tuple(base.__name__ for base in cls.__bases__),
        name not in constructors, tuple(cls._fields), tuple(cls._attributes),
        tuple(AstFieldSchema(field, _type_schema(kind)) for field, kind in getattr(cls, '_field_types', {}).items()))
        for name, cls in classes.items())
    return schema, sha256(Path(ast.__file__).read_bytes()).hexdigest(), sha256(Path(literals.__file__).read_bytes()).hexdigest()


def _profile(f0):
    schema, ast_sha, literal_sha = _schema_and_digests()
    _, _, origin, binary, binary_sha = _tokenizer_implementation()
    # This supported CPython build places both native lexer and parser in the
    # static interpreter. An extension-only lexer is not a parser receipt.
    if origin != 'built-in' or _ast.__spec__.origin != 'built-in':
        raise ValueError('F2 needs the actual loaded native parser implementation receipt')
    return SourceSyntaxProfile(decode(SourceBytesProfile, f0['profile']),
        'builtins.compile.PyCF_ONLY_AST', ast.PyCF_ONLY_AST | ast.PyCF_TYPE_COMMENTS,
        True, 0, sys.version_info.minor, sys.get_int_max_str_digits(), sys.getrecursionlimit(),
        'AST_LINES_1_BASED_UTF8_BYTE_COLUMNS_0_BASED', 'RECORD_ALL_ALWAYS_ISOLATED_V1',
        ast_sha, 'builtins.compile/_ast:built-in', binary, binary_sha, schema,
        'PythonLiteral.from_python(exact=True);FIXED_BUDGET_V1', literal_sha,
        literals.MAX_LITERAL_DEPTH, literals.MAX_LITERAL_NODES, literals.MAX_LITERAL_PAYLOAD_BYTES,
        literals.MAX_INT_DECIMAL_DIGITS, literals.MAX_INT_BIT_LENGTH)


class SyntaxBudgetExceeded(ValueError):
    def __init__(self, code, path, detail):
        self.frontier = SyntaxFrontier(code, path, detail)
        super().__init__(detail)


def _flatten_ast(root, *, max_nodes=1_000_000):
    """Encode official node occurrences, also usable for synthetic codec cases.

    Parser singleton object sharing is deliberately not program identity:
    every parent/field/list occurrence gets its own local integer reference.
    """
    if type(max_nodes) is not int or max_nodes < 0:
        raise ValueError('max_nodes must be an exact nonnegative integer')
    constructors = _registry()[1]
    rows, tasks, active = [], [], set()

    def value(item, path):
        if isinstance(item, ast.AST):
            if type(item).__name__ not in constructors or constructors[type(item).__name__] is not type(item):
                raise ValueError('only exact official concrete AST constructors are recorded')
            if len(rows) >= max_nodes:
                raise SyntaxBudgetExceeded('AST_NODE_BUDGET', path, 'maximum AST occurrence count reached')
            index = len(rows)
            rows.append(None)
            tasks.append((item, index, path))
            return AstValue('NODE', index, (), None)
        if type(item) is list:
            return AstValue('LIST', None, tuple(value(child, path + '[' + str(i) + ']') for i, child in enumerate(item)), None)
        try:
            literal = PythonLiteral.from_python(item, exact=True)
        except ValueError as error:
            raise SyntaxBudgetExceeded('SCALAR_BUDGET', path, str(error)) from error
        return AstValue('SCALAR', None, (), literal)

    root_value = value(root, '$')
    if root_value.kind != 'NODE':
        raise TypeError('F2 root must be an official AST node')
    while tasks:
        node, index, path = tasks.pop()
        if node is None:
            active.remove(index)
            continue
        identity = id(node)
        if identity in active:
            raise ValueError('cyclic AST object graph')
        active.add(identity)
        tasks.append((None, identity, path))
        cls, present = type(node), vars(node)
        if set(present) - set(cls._fields) - set(cls._attributes):
            raise ValueError('undeclared attributes are not official AST material')
        groups = []
        for names in (cls._fields, cls._attributes):
            groups.append(tuple(value(present[name], path + '.' + name) if name in present
                else AstValue('ABSENT', None, (), None) for name in names))
        rows[index] = AstOccurrence(cls.__name__, groups[0], groups[1])
    return root_value.reference, tuple(rows)


def _references(value):
    stack = [value]
    while stack:
        item = stack.pop()
        if item.kind == 'NODE':
            yield item.reference
        elif item.kind == 'LIST':
            stack.extend(item.items)


def _require_value(value, expected, rows, *, nullable_list=False):
    if value.kind == 'ABSENT':
        raise ValueError('ABSENT is allowed only in a declared field/attribute slot')
    origin = get_origin(expected)
    if origin is UnionType:
        for kind in get_args(expected):
            try:
                _require_value(value, kind, rows)
                return
            except ValueError:
                pass
        raise ValueError('AST value does not match its official union field type')
    if origin is list:
        if value.kind != 'LIST':
            raise ValueError('official AST list field requires LIST')
        for item in value.items:
            if nullable_list and item.kind == 'SCALAR' and item.scalar.to_python() is None:
                continue
            _require_value(item, get_args(expected)[0], rows)
        return
    if isinstance(expected, type) and issubclass(expected, ast.AST):
        if value.kind != 'NODE' or not issubclass(_registry()[1][rows[value.reference].constructor], expected):
            raise ValueError('AST reference does not match its official node family')
        return
    if value.kind != 'SCALAR':
        raise ValueError('official AST scalar field requires SCALAR')
    actual = value.scalar.to_python()
    if expected is not object and type(actual) is not expected:
        raise ValueError('AST scalar differs from its official field type')


def _restore_ast(root_index, nodes):
    """Restore the stored occurrences without constructor defaults or hooks."""
    if type(nodes) is not tuple or type(root_index) is not int or not 0 <= root_index < len(nodes):
        raise ValueError('invalid AST occurrence table/root')
    constructors, parents, links = _registry()[1], [0] * len(nodes), []
    for row in nodes:
        if type(row) is not AstOccurrence or row.constructor not in constructors:
            raise ValueError('unknown official AST constructor')
        cls = constructors[row.constructor]
        if len(row.fields) != len(cls._fields) or len(row.attributes) != len(cls._attributes):
            raise ValueError('AST field/attribute arity differs from its official schema')
        refs = tuple(ref for item in (*row.fields, *row.attributes) for ref in _references(item))
        if any(type(ref) is not int or not 0 <= ref < len(nodes) for ref in refs):
            raise ValueError('AST reference is outside the occurrence table')
        for ref in refs:
            parents[ref] += 1
        links.append(refs)
    if parents[root_index] != 0 or any(count != 1 for i, count in enumerate(parents) if i != root_index):
        raise ValueError('each non-root AST occurrence must have exactly one parent')
    reached, pending = set(), [root_index]
    while pending:
        index = pending.pop()
        if index in reached:
            raise ValueError('cyclic or shared AST occurrence references')
        reached.add(index)
        pending.extend(links[index])
    if len(reached) != len(nodes):
        raise ValueError('unreachable AST occurrence rows')
    for row in nodes:
        cls = constructors[row.constructor]
        for name, value in zip(cls._fields, row.fields):
            if value.kind != 'ABSENT':
                _require_value(value, cls._field_types[name], nodes,
                    nullable_list=(row.constructor, name) in (('Dict', 'keys'), ('arguments', 'kw_defaults')))
        for name, value in zip(cls._attributes, row.attributes):
            if value.kind == 'ABSENT':
                continue
            _require_value(value, int | None if name.startswith('end_') else int, nodes)
    objects = tuple(ast.AST.__new__(constructors[row.constructor]) for row in nodes)

    def restore(value):
        if value.kind == 'NODE':
            return objects[value.reference]
        if value.kind == 'LIST':
            return [restore(item) for item in value.items]
        if value.kind == 'SCALAR':
            return value.scalar.to_python()
        raise ValueError('ABSENT cannot occur inside an AST value')

    for node, row in zip(objects, nodes):
        cls = type(node)
        for names, values in ((cls._fields, row.fields), (cls._attributes, row.attributes)):
            for name, value in zip(names, values):
                if value.kind != 'ABSENT':
                    vars(node)[name] = restore(value)
    return objects[root_index]


def _produce_syntax(f0, f1, mode, max_nodes):
    if type(mode) is not str or mode not in ('exec', 'eval', 'single', 'func_type'):
        raise ValueError('mode must be exec/eval/single/func_type')
    if type(max_nodes) is not int or max_nodes < 0:
        raise ValueError('max_nodes must be an exact nonnegative integer')
    original = decoded_source_bytes(f0)
    _verify_tokens(f1, f0, original)
    profile = _profile(f0)
    inputs = SourceSyntaxInput(_digest(f0), _digest(f1), f0['path'], f0['raw']['sha256'],
        f1['view']['sha256'] if f1['view'] is not None else None, f1['status'])
    if f1['status'] != 'TOKENIZED' or not f1['actual_eof']:
        return SourceSyntaxMaterial(inputs, profile, mode, max_nodes, 'UPSTREAM_NOT_TOKENIZED',
            None, (), (), None, ())
    text = original.replace('\r\n', '\n').replace('\r', '\n')
    root, rows, frontiers, failure, status = None, (), (), None, 'PARSED'
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        try:
            tree = builtins.compile(text, f0['path'], mode, flags=profile.flags,
                dont_inherit=True, optimize=0, _feature_version=profile.feature_version)
        except SyntaxError as error:
            failure, status = replace(_native_diagnostic(error),
                coordinate_basis='NATIVE_PARSER_EXCEPTION_DIAGNOSTIC_NOT_AST_COORDINATES'), 'PARSER_ERROR'
        except (MemoryError, RecursionError, OverflowError) as error:
            failure, status = replace(_native_diagnostic(error),
                coordinate_basis='NATIVE_PARSER_EXCEPTION_DIAGNOSTIC_NOT_AST_COORDINATES'), 'PARSER_RESOURCE_ERROR'
        else:
            try:
                root, rows = _flatten_ast(tree, max_nodes=max_nodes)
            except SyntaxBudgetExceeded as error:
                frontiers, status = (error.frontier,), 'BUDGET_STOPPED'
    return SourceSyntaxMaterial(inputs, profile, mode, max_nodes, status, root, rows, frontiers, failure,
                                _warning_records(captured))


def source_syntax_document(f0_document, f1_document, *, mode='exec', max_nodes=1_000_000) -> dict:
    return encode(_produce_syntax(f0_document, f1_document, mode, max_nodes))


def _read_source_syntax(document, f0, f1):
    material = decode(SourceSyntaxMaterial, document)
    fresh = _produce_syntax(f0, f1, material.mode, material.max_nodes)
    if material != fresh:
        raise ValueError('F2 document differs from its F0/F1/profile/fresh official parser')
    if material.status != 'PARSED':
        return material, None
    root = _restore_ast(material.root, material.nodes)
    if _flatten_ast(root, max_nodes=material.max_nodes) != (fresh.root, fresh.nodes):
        raise ValueError('restored AST differs in an occurrence/field/presence/exact scalar')
    return material, root


def read_source_syntax(document, f0, f1) -> ast.AST | None:
    return _read_source_syntax(document, f0, f1)[1]


def read_source_syntax_material(document, f0, f1) -> SourceSyntaxMaterial:
    """Return the same verified immutable occurrences used by the AST reader."""
    return _read_source_syntax(document, f0, f1)[0]


__all__ = ['source_bytes_document', 'read_source_bytes', 'decoded_source_bytes',
           'source_tokens_document', 'read_source_tokens',
           'source_syntax_document', 'read_source_syntax', 'read_source_syntax_material']
