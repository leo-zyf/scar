"""Code identity."""

from dataclasses import dataclass
from enum import StrEnum
from scar.ir.program.call_edges import CapturedCode, CodeCapture
from scar.ir.program.normal_links import CodeScopeWitness
from scar.ir.program.providers import ProviderFactReference
from scar.ir.program.source_protocols import SourceBuiltinInvocation, SourceCompilationRequest
from scar.ir.program.source_exec_roots import ExecRootConnection
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.program.source_text import SourceTextCandidate, TextOpenInterpretation
from scar.ir.program import normal_links as F, normal_targets as T
from scar.ir.program.source_bytes import SourceBytesProfile
from scar.ir.program.python_bindings import SourceCallSite
import re
from scar.ir.program.compiler_symbols import CompilerSymbolsMaterial
from scar.ir.program.source_contexts import SourceContextsMaterial
from scar.ir.program.source_declarations import SourceDeclarationsMaterial
from scar.ir.program.source_bindings import SourceBindingsMaterial
from scar.ir.program.compiler_topology import CompilerTopologyMaterial
from scar.ir.program.source_symbols import SourceSymbolsMaterial


# Compiler construction paths; source scopes and runtime values stay distinct.
class CompilerCodeRole(StrEnum):
    ROOT = 'ROOT'
    FUNCTION_BODY = 'FUNCTION_BODY'
    LAMBDA_BODY = 'LAMBDA_BODY'
    CLASS_BODY = 'CLASS_BODY'
    TYPE_PARAMETERS = 'TYPE_PARAMETERS'
    HEADER_ANNOTATIONS = 'HEADER_ANNOTATIONS'
    VARIABLE_ANNOTATIONS = 'VARIABLE_ANNOTATIONS'
    TYPE_PARAMETER_BOUND = 'TYPE_PARAMETER_BOUND'
    TYPE_PARAMETER_DEFAULT = 'TYPE_PARAMETER_DEFAULT'
    TYPE_ALIAS = 'TYPE_ALIAS'
    COMPREHENSION = 'COMPREHENSION'
    GENERATOR_EXPRESSION = 'GENERATOR_EXPRESSION'


@dataclass(frozen=True, slots=True)
class CompilerCodePath:
    block: int
    source_scope: int
    role: CompilerCodeRole
    constant_path: tuple[int, ...]
    load_offset: int | None
    position: tuple[int | None, ...] | None

    def __post_init__(self):
        if (type(self.block) is not int or self.block < 0
                or type(self.source_scope) is not int or self.source_scope < 0
                or type(self.role) is not CompilerCodeRole
                or any(type(i) is not int or i < 0 for i in self.constant_path)
                or (self.role is CompilerCodeRole.ROOT) != (not self.constant_path)
                or (self.load_offset is None) != (not self.constant_path)):
            raise ValueError('compiler correspondence needs an original scope and constant path')


# Compiler inputs and code-value candidates, distinct from source bodies.
@dataclass(frozen=True, slots=True)
class CompileFilenameCandidate:
    compile_result: T.GlobalPort
    filename: str
    source: T.GlobalPort
    path_source: T.GlobalPort | None = None
    conversion: SourceCallSite | None = None
    conditions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EffectiveCompileContext:
    filename: str
    mode: str
    requested_flags: int
    caller_dont_inherit: bool
    inherited_future_flags: int
    compiler_input_flags: int
    compiled_future_flags: int
    compiled_code_flags: int
    optimize: int
    runtime: SourceBytesProfile
    lowering: str = 'EXPLICIT_RECORDED_FLAGS_DONT_INHERIT_TRUE'


@dataclass(frozen=True, slots=True)
class CompiledSourceCandidate:
    id: str
    compile_result: T.GlobalPort
    read_result: T.GlobalPort
    builtin_boundary_ordinal: int
    text_candidate_ordinal: int
    source_unit: str
    source: SourceArtifactLocator
    text_utf8_sha256: str
    context: EffectiveCompileContext
    code_digest: str
    captured_root_reference: F.CodeScopeWitness | None
    source_only_control_digest: str | None
    code_matches_source_only_control: bool | None
    conditions: tuple[str, ...]
    obligations: tuple[str, ...]
    matched_body: T.Body | None = None
    semantics: str = 'CONDITIONAL_COMPILED_CODE_VALUES_NOT_OBJECT_IDENTITY_OR_F4_BODY'


@dataclass(frozen=True, slots=True)
class SourceCompileObligation:
    port: T.GlobalPort
    reason: str


@dataclass(frozen=True, slots=True)
class CompiledSourceOverlay:
    unit: str
    filenames: tuple[CompileFilenameCandidate, ...]
    candidates: tuple[CompiledSourceCandidate, ...]
    obligations: tuple[SourceCompileObligation, ...]


# Receipt-bound compiler inputs; matching a root never selects an exec branch.
@dataclass(frozen=True, slots=True)
class CompiledExecAlternative:
    evidence: ProviderFactReference
    connection: ExecRootConnection


@dataclass(frozen=True, slots=True)
class CompiledCodeProfile:
    unit: str
    original_source_unit: str
    root_scope: CodeScopeWitness
    root_fact: ProviderFactReference
    candidate_fact: ProviderFactReference
    candidate: CompiledSourceCandidate
    request_fact: ProviderFactReference
    request: SourceCompilationRequest
    compile_boundary_fact: ProviderFactReference
    compile_boundary: SourceBuiltinInvocation
    compile_call: ProviderFactReference
    text_fact: ProviderFactReference
    text: SourceTextCandidate
    context_fact: ProviderFactReference
    syntax: SourceArtifactLocator
    compiled_root: CapturedCode
    branches: tuple[CompiledExecAlternative, ...]


@dataclass(frozen=True, slots=True)
class CompiledRootCodeMatch:
    profile: int
    capture: CodeCapture
    conditions: tuple[str, ...] = (
        'OBSERVED_CODE_EQUALS_RECORDED_COMPILER_PROFILE_ROOT_TEMPLATE',
        'EXEC_BRANCH_AND_NAMESPACE_SELECTION_NOT_ESTABLISHED_BY_CODE_MATCH',
        'CALLER_CREATOR_AND_CELL_CONTENT_IDENTITIES_NOT_ESTABLISHED',
    )


# Source-derived compiler symbol provenance in one compilation namespace.
@dataclass(frozen=True, slots=True)
class SourceCompilerSymbolVariant:
    namespace: str
    context_variant_sha256: str
    context_material_sha256: str
    binding_variant_sha256: str
    binding_material_sha256: str
    native_variant_sha256: str
    topology_variant_sha256: str
    material: CompilerSymbolsMaterial
    schema: str = 'scar.source-compiler-symbol-variant'
    schema_version: int = 1
    discharges: tuple[str, ...] = (
        'CONTEXT_QUALIFIED_SYMBOL_ORIGIN_AND_PARAMETER_PROJECTION_NOT_BUILT',
    )
    obligations: tuple[str, ...] = ('CONTEXT_QUALIFIED_F4_BODY_NOT_BUILT',)

    def __post_init__(self):
        digests = (self.namespace, self.context_variant_sha256, self.context_material_sha256,
            self.binding_variant_sha256, self.binding_material_sha256,
            self.native_variant_sha256, self.topology_variant_sha256)
        if (any(type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None for value in digests)
                or self.schema != 'scar.source-compiler-symbol-variant'
                or type(self.schema_version) is not int or self.schema_version != 1
                or self.material.status != 'PROJECTED'
                or self.material.input.c1_document_sha256 != self.context_material_sha256):
            raise ValueError('compiler symbol variant requires a complete context-qualified projection')


# C1 context namespaces over unchanged source and syntax materials.
@dataclass(frozen=True, slots=True)
class SourceContextVariant:
    namespace: str
    source_unit: str
    context: EffectiveCompileContext
    text_utf8_sha256: str
    code_digest: str
    ast_correspondence_sha256: str
    material: SourceContextsMaterial
    schema: str = 'scar.source-context-variant'
    schema_version: int = 1
    obligations: tuple[str, ...] = (
        'CONTEXT_QUALIFIED_DECLARATIONS_AND_BINDINGS_NOT_BUILT',
        'CONTEXT_QUALIFIED_NATIVE_SYMBOL_VIEW_NOT_CAPTURED',
        'CONTEXT_QUALIFIED_F4_BODY_NOT_BUILT',
    )


# Context-qualified source declarations and namespace templates.
@dataclass(frozen=True, slots=True)
class SourceBindingVariant:
    namespace: str
    context_material_sha256: str
    declarations: SourceDeclarationsMaterial
    bindings: SourceBindingsMaterial
    schema: str = 'scar.source-binding-variant'
    schema_version: int = 1
    discharges: tuple[str, ...] = ('CONTEXT_QUALIFIED_DECLARATIONS_AND_BINDINGS_NOT_BUILT',)
    obligations: tuple[str, ...] = (
        'CONTEXT_QUALIFIED_NATIVE_SYMBOL_VIEW_NOT_CAPTURED',
        'CONTEXT_QUALIFIED_F4_BODY_NOT_BUILT',
    )


# Compiler/source scope correspondence in one compilation namespace.
@dataclass(frozen=True, slots=True)
class SourceTopologyVariant:
    namespace: str
    context_material_sha256: str
    native_variant_sha256: str
    material: CompilerTopologyMaterial
    schema: str = 'scar.source-topology-variant'
    schema_version: int = 1
    discharges: tuple[str, ...] = ('CONTEXT_QUALIFIED_SYMBOL_AST_CORRESPONDENCE_NOT_ESTABLISHED',)
    obligations: tuple[str, ...] = (
        'CONTEXT_QUALIFIED_SYMBOL_ORIGIN_AND_PARAMETER_PROJECTION_NOT_BUILT',
        'CONTEXT_QUALIFIED_F4_BODY_NOT_BUILT',
    )


# Context-qualified official symbol views; native scope/AST joins stay open.
@dataclass(frozen=True, slots=True)
class SymbolAdapterReceipt:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class SymbolSourceParent:
    name: str
    source: SourceArtifactLocator
    document_sha256: str


@dataclass(frozen=True, slots=True)
class SymbolTextInput:
    source: SourceArtifactLocator
    f0_document_sha256: str
    raw_sha256: str
    raw_bytes: int
    interpretation: TextOpenInterpretation
    text_utf8_sha256: str
    characters: int
    conditions: tuple[str, ...]
    obligations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SymbolContextRequest:
    text_utf8_sha256: str
    characters: int
    filename: str
    mode: str
    requested_flags: int
    dont_inherit: bool
    inherited_future_flags: int
    effective_flags: int
    caller_flag_merging: str = 'EXPLICIT_ONLY_NO_PYTHON_CALLER_INHERITANCE'
    optimize_policy: str = 'OFFICIAL_SYMTABLE_BEFORE_OPTIMIZATION_NO_OPTIMIZE_ARGUMENT'


@dataclass(frozen=True, slots=True)
class SourceSymbolVariant:
    namespace: str
    source_unit: str
    context_document_sha256: str
    context_material_sha256: str
    context: EffectiveCompileContext
    parents: tuple[SymbolSourceParent, ...]
    text_inputs: tuple[SymbolTextInput, ...]
    request: SymbolContextRequest
    adapter: SymbolAdapterReceipt
    material: SourceSymbolsMaterial
    discharged: tuple[str, ...] = ('CONTEXT_QUALIFIED_NATIVE_SYMBOL_VIEW_NOT_CAPTURED',)
    obligations: tuple[str, ...] = (
        'CONTEXT_QUALIFIED_DECLARATIONS_AND_BINDINGS_NOT_BUILT',
        'CONTEXT_QUALIFIED_SYMBOL_AST_CORRESPONDENCE_NOT_ESTABLISHED',
        'CONTEXT_QUALIFIED_F4_BODY_NOT_BUILT',
    )
    schema: str = 'scar.source-symbol-variant'
    schema_version: int = 1

    def __post_init__(self):
        if (self.schema != 'scar.source-symbol-variant' or self.schema_version != 1
                or self.material.status != 'CAPTURED' or not self.text_inputs
                or self.material.authority != 'PUBLIC_COMPILER_VIEW'
                or self.material.ast_join != 'UNJOINED'
                or self.material.complete_lexical_scopes):
            raise ValueError('context symbol variant must remain a captured unjoined compiler view')
