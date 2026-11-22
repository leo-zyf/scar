"""Compiler symbols."""

from dataclasses import dataclass
from scar.ir.program.source_roles import SlotRef


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
