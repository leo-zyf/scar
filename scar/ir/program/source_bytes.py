"""Source bytes."""

from dataclasses import dataclass
from pathlib import PurePosixPath
import re
from enum import Enum
from scar.ir.literals import PythonLiteral
from scar.ir.program import normal_graph as F, normal_graph as T, source_productions as P
from scar.ir.program.python_bindings import SourceCallSite
from scar.ir.program.source_references import SourceArtifactLocator


# F0 material records: original bytes and their encoding interpretation only.
def _text(value):
    return type(value) is str and bool(value) and '\0' not in value


def _digest(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _version(value):
    return (type(value) is tuple and len(value) == 5
        and all(type(part) is int and part >= 0 for part in (*value[:3], value[4]))
        and type(value[3]) is str and value[3] in ('alpha', 'beta', 'candidate', 'final'))


@dataclass(frozen=True, slots=True)
class SourceBytesProfile:
    implementation: str
    implementation_version: tuple[int, int, int, str, int]
    version: tuple[int, int, int, str, int]
    version_string: str
    hexversion: int
    cache_tag: str | None
    rule: str = 'STDLIB_DETECT_ENCODING_RAW_DECODE_F0_V1'
    decoded_text_encoding: str = 'utf-8'
    decoded_text_errors: str = 'strict'

    def __post_init__(self):
        if (not _text(self.implementation) or not _version(self.implementation_version)
                or not _version(self.version) or not _text(self.version_string)
                or type(self.hexversion) is not int or self.hexversion < 1
                or self.cache_tag is not None and not _text(self.cache_tag)
                or self.rule != 'STDLIB_DETECT_ENCODING_RAW_DECODE_F0_V1'
                or self.decoded_text_encoding != 'utf-8' or self.decoded_text_errors != 'strict'):
            raise ValueError('unsupported F0 source interpretation profile')


@dataclass(frozen=True, slots=True)
class SourceBytesError:
    stage: str
    exception_type: str
    message: str
    reason: str | None
    encoding: str | None
    start: int | None
    end: int | None
    index_basis: str | None
    lineno: int | None
    offset: int | None
    end_lineno: int | None
    end_offset: int | None

    def __post_init__(self):
        if (self.stage not in ('ENCODING', 'DECODING', 'DECODED_TEXT_ENCODING')
                or not _text(self.exception_type) or type(self.message) is not str
                or any(value is not None and type(value) is not str
                    for value in (self.reason, self.encoding))
                or any(value is not None and type(value) is not int
                    for value in (self.start, self.end, self.lineno, self.offset, self.end_lineno, self.end_offset))
                or (self.start is None) != (self.end is None)
                or self.start is not None and not 0 <= self.start <= self.end
                or self.index_basis not in (None, 'RAW_INPUT', 'DECODED_TEXT', 'EXCEPTION_OBJECT')):
            raise ValueError('invalid structured F0 encoding error')


@dataclass(frozen=True, slots=True)
class SourceRawBytes:
    base64: str
    sha256: str
    size_bytes: int

    def __post_init__(self):
        if type(self.base64) is not str or not _digest(self.sha256) or type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError('invalid original F0 bytes receipt')


@dataclass(frozen=True, slots=True)
class SourceBytesDecoding:
    status: str
    encoding: str | None
    decoded_utf8_sha256: str | None
    characters: int | None
    error: SourceBytesError | None

    def __post_init__(self):
        if (self.encoding is not None and not _text(self.encoding)
                or self.characters is not None and (type(self.characters) is not int or self.characters < 0)
                or self.decoded_utf8_sha256 is not None and not _digest(self.decoded_utf8_sha256)
                or self.error is not None and type(self.error) is not SourceBytesError):
            raise ValueError('invalid F0 decoded fields')
        if self.status == 'DECODED':
            valid = self.encoding is not None and self.characters is not None and self.decoded_utf8_sha256 is not None and self.error is None
        else:
            stages = {'ENCODING_ERROR': 'ENCODING', 'DECODE_ERROR': 'DECODING',
                'DECODED_TEXT_ENCODING_ERROR': 'DECODED_TEXT_ENCODING'}
            valid = (self.status in stages and self.error is not None
                and self.error.stage == stages[self.status] and self.decoded_utf8_sha256 is None
                and (self.encoding is None) == (self.status == 'ENCODING_ERROR')
                and (self.characters is not None) == (self.status == 'DECODED_TEXT_ENCODING_ERROR'))
        if not valid:
            raise ValueError('F0 decoding status contradicts its fields')


@dataclass(frozen=True, slots=True)
class SourceBytesMaterial:
    path: str
    namespace_root: str
    profile: SourceBytesProfile
    raw: SourceRawBytes
    decoding: SourceBytesDecoding
    schema: str = 'scar.source-bytes'
    schema_version: int = 1

    def __post_init__(self):
        if self.schema != 'scar.source-bytes' or type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('unsupported source-bytes schema')
        for value in (self.path, self.namespace_root):
            if (not _text(value) or not PurePosixPath(value).is_absolute()
                    or str(PurePosixPath(value)) != value or '..' in PurePosixPath(value).parts):
                raise ValueError('F0 paths must be absolute and lexically normalized')
        if self.path == self.namespace_root or not PurePosixPath(self.path).is_relative_to(self.namespace_root):
            raise ValueError('F0 source path must be inside its declared namespace')
        for value, kind in ((self.profile, SourceBytesProfile), (self.raw, SourceRawBytes), (self.decoding, SourceBytesDecoding)):
            if type(value) is not kind:
                raise TypeError('F0 requires exact ' + kind.__name__)
            value.__post_init__()


# Conditional path expressions and text reads of captured source artifacts.
class SourcePathOperation(str, Enum):
    CONSTRUCT = 'CONSTRUCT'
    RESOLVE = 'RESOLVE'
    PARENT = 'PARENT'
    JOIN = 'JOIN'


@dataclass(frozen=True, slots=True)
class SourcePathExpression:
    id: str
    result: T.GlobalPort
    owner: F.CodeScopeRef
    operation: SourcePathOperation
    operands: tuple[T.GlobalPort, ...]
    provider: SourceArtifactLocator
    export: str
    source_table: str
    source_ordinal: int
    call: SourceCallSite | None = None
    attribute: T.AttributeSite | None = None
    language: T.LanguageOperationSite | None = None
    conditions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SourcePathCandidate:
    expression: str
    result: T.GlobalPort
    lexical_path: str
    conditions: tuple[str, ...]
    semantics: str = 'CONDITIONAL_LEXICAL_PATH_NOT_FILESYSTEM_IDENTITY'


@dataclass(frozen=True, slots=True)
class TextReadParameter:
    name: str
    source: T.GlobalPort | None
    literal: PythonLiteral | None
    defaulted: bool
    literal_occurrence: P.ProductionOccurrence | None = None


@dataclass(frozen=True, slots=True)
class SourceTextRead:
    site: SourceCallSite
    owner: F.CodeScopeRef
    receiver: T.GlobalPort
    path_expression: str
    parameters: tuple[TextReadParameter, ...]
    conditions: tuple[str, ...]
    obligations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TextOpenInterpretation:
    encoding: str | None
    errors: str | None
    newline: str | None
    newline_resolved: bool
    default_encoding: bool
    rule: str = 'TEXT_OPEN_CAPTURED_RAW_BYTES_V1'
    conditions: tuple[str, ...] = ()
    obligations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SourceTextCandidate:
    read_result: T.GlobalPort
    path_expression: str
    source_unit: str
    source_path: str
    source: SourceArtifactLocator
    f0_document_sha256: str
    raw_sha256: str
    raw_bytes: int
    interpretation: TextOpenInterpretation
    text_utf8_sha256: str | None
    characters: int | None
    conditions: tuple[str, ...]
    obligations: tuple[str, ...] = ()
    semantics: str = 'CONDITIONAL_CAPTURED_SOURCE_TEXT_NOT_RUNTIME_IO'


@dataclass(frozen=True, slots=True)
class SourceTextObligation:
    port: T.GlobalPort
    reason: str


@dataclass(frozen=True, slots=True)
class SourceTextOverlay:
    unit: str
    paths: tuple[SourcePathExpression, ...]
    path_values: tuple[SourcePathCandidate, ...]
    reads: tuple[SourceTextRead, ...]
    candidates: tuple[SourceTextCandidate, ...]
    obligations: tuple[SourceTextObligation, ...]


# F1 lexical records; references are character spans, never raw-byte offsets.
@dataclass(frozen=True, slots=True)
class TokenSpan:
    start: int
    end: int

    def __post_init__(self):
        if type(self.start) is not int or type(self.end) is not int or not 0 <= self.start <= self.end:
            raise ValueError('invalid F1 half-open span')


@dataclass(frozen=True, slots=True)
class TokenTextRef:
    kind: str
    start: int | None
    end: int | None
    literal_index: int | None

    def __post_init__(self):
        if self.kind == 'VIEW_RANGE':
            valid = self.literal_index is None and type(self.start) is int and type(self.end) is int and 0 <= self.start <= self.end
        else:
            valid = self.kind == 'INTERNED_LITERAL' and self.start is None and self.end is None and type(self.literal_index) is int and self.literal_index >= 0
        if not valid:
            raise ValueError('invalid F1 text reference')


@dataclass(frozen=True, slots=True)
class TokenExtent:
    view_chars: TokenSpan
    original_chars: TokenSpan
    view_utf8_bytes: TokenSpan


@dataclass(frozen=True, slots=True)
class NewlineRewrite:
    original_chars: TokenSpan
    view_chars: TokenSpan


@dataclass(frozen=True, slots=True)
class TokenView:
    sha256: str
    characters: int
    utf8_bytes: int
    newline_rewrites: tuple[NewlineRewrite, ...]

    def __post_init__(self):
        if not _digest(self.sha256) or any(type(x) is not int or x < 0 for x in (self.characters, self.utf8_bytes)):
            raise ValueError('invalid F1 view receipt')


@dataclass(frozen=True, slots=True)
class TokenType:
    code: int
    name: str


@dataclass(frozen=True, slots=True)
class ExactOperator:
    text: str
    code: int


@dataclass(frozen=True, slots=True)
class SourceTokensProfile:
    runtime: SourceBytesProfile
    api: str
    newline_rule: str
    warning_policy: str
    tokenize_sha256: str
    token_sha256: str
    native_tokenizer_origin: str
    native_tokenizer_binary: str
    native_tokenizer_sha256: str
    token_types: tuple[TokenType, ...]
    exact_operators: tuple[ExactOperator, ...]

    def __post_init__(self):
        if (self.api != 'stdlib.tokenize.generate_tokens.extra_tokens'
                or self.newline_rule != 'CRLF_AND_CR_TO_LF_NO_FINAL_NEWLINE_V1'
                or self.warning_policy != 'RECORD_ALL_ALWAYS_ISOLATED_V1'
                or not _digest(self.tokenize_sha256) or not _digest(self.token_sha256)
                or not _digest(self.native_tokenizer_sha256)):
            raise ValueError('invalid F1 lexer profile')


@dataclass(frozen=True, slots=True)
class SourceTokensInput:
    f0_document_sha256: str
    path: str
    namespace_root: str
    raw_sha256: str
    decoded_utf8_sha256: str | None
    decoding_status: str


@dataclass(frozen=True, slots=True)
class SourceToken:
    type: int
    exact_type: int
    string: TokenTextRef
    line: TokenTextRef
    start: tuple[int, int]
    end: tuple[int, int]
    source_extent: TokenExtent | None
    coordinate_kind: str

    def __post_init__(self):
        if self.coordinate_kind not in ('SOURCE', 'ZERO_WIDTH', 'SYNTHETIC_NEWLINE', 'VIRTUAL_ENDMARKER', 'VIRTUAL_COORDINATES'):
            raise ValueError('invalid F1 native-coordinate classification')
        if any(type(x) is not int or x < 0 for pair in (self.start, self.end) for x in pair):
            raise ValueError('invalid F1 native token position')


@dataclass(frozen=True, slots=True)
class TokenCoverage:
    covered: tuple[TokenSpan, ...]
    gaps: tuple[TokenSpan, ...]


@dataclass(frozen=True, slots=True)
class NativeTokenParameter:
    kind: str
    value: str | int | float | bool | None
    items: tuple['NativeTokenParameter', ...]

    def __post_init__(self):
        scalar_types = {'NONE': type(None), 'STR': str, 'INT': int, 'FLOAT': float, 'BOOL': bool, 'BYTES_BASE64': str}
        if self.kind == 'TUPLE':
            valid = self.value is None
        else:
            valid = self.kind in scalar_types and type(self.value) is scalar_types[self.kind] and not self.items
        if not valid:
            raise ValueError('invalid F1 native diagnostic parameter')


@dataclass(frozen=True, slots=True)
class NativeTokenAttribute:
    name: str
    value: NativeTokenParameter


@dataclass(frozen=True, slots=True)
class SourceTokenError:
    module: str
    exception_type: str
    message: str
    parameters: NativeTokenParameter
    attributes: tuple[NativeTokenAttribute, ...]
    coordinate_basis: str = 'NATIVE_EXCEPTION_DIAGNOSTIC_NOT_TOKEN_COORDINATES'


@dataclass(frozen=True, slots=True)
class SourceTokenWarning:
    category_module: str
    category_name: str
    message: str
    filename: str
    lineno: int
    line: str | None
    source_present: bool


@dataclass(frozen=True, slots=True)
class SourceTokensMaterial:
    input: SourceTokensInput
    profile: SourceTokensProfile
    max_tokens: int
    status: str
    actual_eof: bool
    view: TokenView | None
    text_pool: tuple[str, ...]
    tokens: tuple[SourceToken, ...]
    source_coverage: TokenCoverage | None
    error: SourceTokenError | None
    warnings: tuple[SourceTokenWarning, ...]
    schema: str = 'scar.source-tokens'
    schema_version: int = 1

    def __post_init__(self):
        if (self.schema != 'scar.source-tokens' or type(self.schema_version) is not int or self.schema_version != 1
                or type(self.max_tokens) is not int or self.max_tokens < 0
                or self.status not in ('TOKENIZED', 'TOKENIZER_ERROR', 'BUDGET_STOPPED', 'UPSTREAM_DECODING_ERROR')):
            raise ValueError('invalid F1 document envelope')
