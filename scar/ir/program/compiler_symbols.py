"""Compiler symbols."""

from dataclasses import dataclass
from scar.ir.program.source_roles import SlotRef
from enum import StrEnum
from scar.ir.program.call_edges import CapturedCode, CodeCapture
from scar.ir.program.normal_graph import CodeScopeWitness
from scar.ir.program.providers import ProviderFactReference
from scar.ir.program.source_protocols import SourceBuiltinInvocation, SourceCompilationRequest
from scar.ir.program.source_exec_roots import ExecRootConnection
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.program.source_bytes import (
    SourceTextCandidate,
    TextOpenInterpretation,
    SourceBytesProfile,
)
from scar.ir.program import normal_graph as F, normal_graph as T
from scar.ir.program.python_bindings import SourceCallSite
import re
from scar.ir.program.source_contexts import SourceContextsMaterial
from scar.ir.program.source_declarations import SourceDeclarationsMaterial
from scar.ir.program.source_bindings import SourceBindingsMaterial
from scar.ir.program.source_paths import SourceSymbolsMaterial


# Compiler symbols.
# C3-S compiler-symbol provenance, distinct from source/runtime bindings.
INTRINSIC_NAMES = ('.format', '.0', '.defaults', '.kwdefaults', '.type_params',
    '.generic_base', '__type_params__', '__class__', '__classdict__', '__conditional_annotations__')


@dataclass(frozen=True, slots=True)
class CompilerName:
    kind: str
    source: int | None = None
    intrinsic: str | None = None

    def __post_init__(self):
        if self.kind == 'SOURCE':
            valid = type(self.source) is int and self.source >= 0 and self.intrinsic is None
        else:
            valid = self.kind == 'INTRINSIC' and self.source is None and self.intrinsic in INTRINSIC_NAMES
        if not valid:
            raise ValueError('invalid compiler name reference')


@dataclass(frozen=True, slots=True)
class SymbolLimits:
    max_symbol_slots: int = 2_000_000
    max_contributions: int = 4_000_000
    max_facts: int = 4_000_000
    max_transforms: int = 2_000_000
    max_steps: int = 20_000_000

    def __post_init__(self):
        if any(type(n) is not int or n < 0 for n in (self.max_symbol_slots,
                self.max_contributions, self.max_facts, self.max_transforms, self.max_steps)):
            raise ValueError('compiler symbol budgets must be nonnegative exact integers')


@dataclass(frozen=True, slots=True)
class RawContribution:
    context_block: int
    slot: int
    flags: int
    rule: str
    source: int | None
    anchor: SlotRef | None
    derived_from: int | None


@dataclass(frozen=True, slots=True)
class RawSlot:
    block: int
    name: CompilerName
    flags: int
    contributions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SourceDisposition:
    source: int
    disposition: str
    contributions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RawParameters:
    block: int
    slots: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RawCollection:
    """Private typed handoff, not a second persisted source transcript."""
    slots: tuple[RawSlot, ...]
    contributions: tuple[RawContribution, ...]
    sources: tuple[SourceDisposition, ...]
    parameters: tuple[RawParameters, ...]
    conditional_blocks: tuple[int, ...]
    steps: int


@dataclass(frozen=True, slots=True)
class CompilerSlot:
    block: int
    name: CompilerName
    contributions: tuple[int, ...]
    initial_fact: int | None
    final_fact: int | None
    disposition: str


@dataclass(frozen=True, slots=True)
class SymbolFact:
    slot: int
    low_flags: int
    scope_code: int | None
    rule: str
    inputs: tuple[int, ...]
    witness_blocks: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SlotTransform:
    rule: str
    source_slot: int
    target_slot: int | None
    input_fact: int
    output_fact: int | None


@dataclass(frozen=True, slots=True)
class ProtocolRequirement:
    block: int
    name: CompilerName
    rule: str
    inputs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BlockSymbols:
    block: int
    slots: tuple[int, ...]
    parameters: tuple[int, ...]
    protocols: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SymbolProjection:
    native_scope: int
    ordinal: int
    slot: int
    fact: int


@dataclass(frozen=True, slots=True)
class ParameterProjection:
    native_scope: int
    ordinal: int
    slot: int


@dataclass(frozen=True, slots=True)
class SymbolSolution:
    """Private solver output; native evidence is not an input to the solver."""
    slots: tuple[CompilerSlot, ...]
    facts: tuple[SymbolFact, ...]
    transforms: tuple[SlotTransform, ...]
    protocols: tuple[ProtocolRequirement, ...]
    blocks: tuple[BlockSymbols, ...]
    steps: int


@dataclass(frozen=True, slots=True)
class SymbolFrontier:
    code: str
    block: int | None
    source: int | None
    detail: str


class SymbolStop(ValueError):
    def __init__(self, code, detail, *, block=None, source=None, budget=False):
        self.frontier = SymbolFrontier(code, block, source, detail)
        self.budget = budget
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class CompilerSymbolCatalog:
    rule_version: str
    topology_catalog_sha256: str
    declaration_catalog_sha256: str
    intrinsic_names: tuple[str, ...]
    raw_rules: tuple[str, ...]
    fact_rules: tuple[str, ...]
    transform_rules: tuple[str, ...]
    protocol_rules: tuple[str, ...]
    source_dispositions: tuple[str, ...]
    slot_dispositions: tuple[str, ...]
    content_sha256: str
    schema: str = 'scar.compiler-symbol-catalog'
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class CompilerSymbolsInput:
    f0_document_sha256: str
    f1_document_sha256: str
    f2_document_sha256: str
    f3b_document_sha256: str
    c1_document_sha256: str
    c2a_document_sha256: str
    f3a_document_sha256: str
    topology_document_sha256: str
    path: str
    mode: str
    declaration_status: str
    topology_status: str
    source_count: int
    block_count: int
    native_symbol_count: int
    native_parameter_count: int


@dataclass(frozen=True, slots=True)
class CompilerSymbolsProfile:
    rule_version: str
    symbol_catalog_sha256: str
    topology_catalog_sha256: str
    declaration_catalog_sha256: str
    native_profile_sha256: str
    implementation_sha256: str


@dataclass(frozen=True, slots=True)
class CompilerSymbolsMaterial:
    input: CompilerSymbolsInput
    profile: CompilerSymbolsProfile
    limits: SymbolLimits
    status: str
    slots: tuple[CompilerSlot, ...]
    contributions: tuple[RawContribution, ...]
    sources: tuple[SourceDisposition, ...]
    facts: tuple[SymbolFact, ...]
    transforms: tuple[SlotTransform, ...]
    protocols: tuple[ProtocolRequirement, ...]
    blocks: tuple[BlockSymbols, ...]
    symbols: tuple[SymbolProjection, ...]
    parameters: tuple[ParameterProjection, ...]
    frontiers: tuple[SymbolFrontier, ...]
    steps: int
    authority: str = 'SOURCE_COMPILER_SYMBOL_PROVENANCE'
    runtime_binding: str = 'NOT_OBSERVED'
    runtime_local_layout: str = 'NOT_CLAIMED'
    schema: str = 'scar.compiler-symbols'
    schema_version: int = 1

    def __post_init__(self):
        if (self.schema != 'scar.compiler-symbols' or type(self.schema_version) is not int or self.schema_version != 1
                or self.authority != 'SOURCE_COMPILER_SYMBOL_PROVENANCE' or self.runtime_binding != 'NOT_OBSERVED'
                or self.runtime_local_layout != 'NOT_CLAIMED' or type(self.steps) is not int
                or not 0 <= self.steps <= self.limits.max_steps
                or len(self.slots) > self.limits.max_symbol_slots
                or len(self.contributions) > self.limits.max_contributions
                or len(self.facts) > self.limits.max_facts or len(self.transforms) > self.limits.max_transforms):
            raise ValueError('invalid compiler symbol envelope/budgets')
        if self.status == 'PROJECTED':
            valid = (len(self.blocks) == self.input.block_count and len(self.sources) == self.input.source_count
                and len(self.symbols) == self.input.native_symbol_count
                and len(self.parameters) == self.input.native_parameter_count and not self.frontiers)
        elif self.status == 'SOURCE_ONLY_MODE':
            valid = (not (self.slots or self.contributions or self.facts or self.transforms or self.protocols
                    or self.blocks or self.symbols or self.parameters or self.frontiers)
                and len(self.sources) == self.input.source_count
                and all(row.disposition == 'GRAMMAR_ONLY' and not row.contributions for row in self.sources))
        else:
            valid = (self.status in ('UPSTREAM_NOT_READY', 'BUDGET_STOPPED', 'PROJECTION_CONTRADICTION')
                and not (self.slots or self.contributions or self.sources or self.facts or self.transforms
                    or self.protocols or self.blocks or self.symbols or self.parameters) and bool(self.frontiers))
        if not valid:
            raise ValueError('compiler symbol status contradicts provenance coverage')


# C3-T source-anchored compiler scope templates, not runtime/code-object IDs.
@dataclass(frozen=True, slots=True)
class TopologyFieldRule:
    id: int
    constructor: str
    group: str
    ordinal: int
    name: str
    dispatch: str


@dataclass(frozen=True, slots=True)
class TopologyConstructor:
    name: str
    dispatch: str
    fields: tuple[int, ...]
    attributes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TopologyCatalog:
    rule_version: str
    f2_schema_sha256: str
    context_catalog_sha256: str
    constructors: tuple[TopologyConstructor, ...]
    fields: tuple[TopologyFieldRule, ...]
    origin_roles: tuple[str, ...]
    projections: tuple[str, ...]
    attachments: tuple[str, ...]
    scope_dispositions: tuple[str, ...]
    rewrite_rules: tuple[str, ...]
    content_sha256: str
    schema: str = 'scar.compiler-topology-catalog'
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class BlockOrigin:
    occurrence: int
    role: str

    def __post_init__(self):
        if type(self.occurrence) is not int or self.occurrence < 0 or type(self.role) is not str:
            raise ValueError('invalid compiler block source origin')


@dataclass(frozen=True, slots=True)
class CompilerBlock:
    origin: BlockOrigin
    parent: int | None
    children: tuple[int, ...]
    source_scope: int | None
    can_see_class_scope: bool
    attachment: str
    projection: str
    public_scope: int | None


@dataclass(frozen=True, slots=True)
class ProjectionStep:
    rule: str
    block: int
    parent: int | None


@dataclass(frozen=True, slots=True)
class SourceScopeProjection:
    scope: int
    block: int | None
    disposition: str


@dataclass(frozen=True, slots=True)
class TopologyFrontier:
    code: str
    origin: BlockOrigin | None
    detail: str


@dataclass(frozen=True, slots=True)
class CompilerTopologyInput:
    f0_document_sha256: str
    f1_document_sha256: str
    f2_document_sha256: str
    f3b_document_sha256: str
    c1_document_sha256: str
    f3a_document_sha256: str
    path: str
    mode: str
    c1_status: str
    f3a_status: str
    source_scope_count: int
    native_scope_count: int


@dataclass(frozen=True, slots=True)
class CompilerTopologyProfile:
    rule_version: str
    f2_schema_sha256: str
    context_catalog_sha256: str
    topology_catalog_sha256: str
    native_profile_sha256: str
    implementation_sha256: str


@dataclass(frozen=True, slots=True)
class CompilerTopologyMaterial:
    input: CompilerTopologyInput
    profile: CompilerTopologyProfile
    max_blocks: int
    max_rewrites: int
    max_visit_steps: int
    status: str
    blocks: tuple[CompilerBlock, ...]
    scopes: tuple[SourceScopeProjection, ...]
    rewrites: tuple[ProjectionStep, ...]
    frontiers: tuple[TopologyFrontier, ...]
    visit_steps: int
    authority: str = 'SOURCE_COMPILER_TOPOLOGY'
    symbol_origin_projection: str = 'NOT_COMPUTED_IN_C3T'
    parameter_projection: str = 'NOT_COMPUTED_IN_C3T'
    runtime_execution: str = 'NOT_OBSERVED'
    schema: str = 'scar.compiler-topology'
    schema_version: int = 1

    def __post_init__(self):
        if (self.schema != 'scar.compiler-topology' or type(self.schema_version) is not int or self.schema_version != 1
                or self.authority != 'SOURCE_COMPILER_TOPOLOGY'
                or self.symbol_origin_projection != 'NOT_COMPUTED_IN_C3T'
                or self.parameter_projection != 'NOT_COMPUTED_IN_C3T'
                or self.runtime_execution != 'NOT_OBSERVED'
                or any(type(n) is not int or n < 0 for n in (
                    self.max_blocks, self.max_rewrites, self.max_visit_steps, self.visit_steps))
                or len(self.blocks) > self.max_blocks or len(self.rewrites) > self.max_rewrites
                or self.visit_steps > self.max_visit_steps):
            raise ValueError('invalid C3-T topology envelope/budgets')
        if self.status == 'PROJECTED':
            valid = bool(self.blocks) and len(self.scopes) == self.input.source_scope_count and not self.frontiers
        elif self.status == 'SOURCE_ONLY_MODE':
            valid = (not self.blocks and not self.rewrites and not self.frontiers
                and len(self.scopes) == self.input.source_scope_count
                and all(row.block is None and row.disposition == 'GRAMMAR_ONLY' for row in self.scopes))
        else:
            valid = (self.status in ('UPSTREAM_NOT_CONTEXTUALIZED', 'UPSTREAM_NOT_CAPTURED',
                'BUDGET_STOPPED', 'PROJECTION_CONTRADICTION')
                and not (self.blocks or self.scopes or self.rewrites) and bool(self.frontiers))
        if not valid:
            raise ValueError('C3-T status contradicts topology coverage')


# Code identity.
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
