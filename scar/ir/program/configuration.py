"""Configuration."""

from dataclasses import dataclass
from enum import Enum
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.literals import PythonLiteral
from scar.ir.program.source_frame_bindings import StaticInputPort, ContextEndpoint
from scar.ir.program.normal_targets import GlobalPort
from scar.ir.program.ids import OperationID, RegionID, ValueID


# Ordered configuration operations over shared, versioned data materials.
#
# This is a conditional library-protocol graph, not a dictionary snapshot or an
# execution trace. A merge result is a version; it is never a Python object ID.
@dataclass(frozen=True,slots=True)
class ConfigurationProtocolSource:
    source: ProviderArtifact
    qualified_function: str
    function_ast_sha256: str


@dataclass(frozen=True,slots=True)
class ConfigurationMaterial:
    id: str
    store: ProviderArtifact
    source: ProviderArtifact
    root_reference: int | None
    package_header: str | None


class ConfigurationDependencyKind(str,Enum):
    SELF = 'SELF'
    CONFIG = 'CONFIG'
    GROUP = 'GROUP'
    OVERRIDE = 'OVERRIDE'
    NULL_CHOICE = 'NULL_CHOICE'
    UNRESOLVED = 'UNRESOLVED'


@dataclass(frozen=True,slots=True)
class ConfigurationDependency:
    ordinal: int
    instance: str
    material: str
    reference: int | None
    kind: ConfigurationDependencyKind
    config_path: str | None
    group: str | None
    choice: str | None
    package_override: str | None
    package: str | None
    child_instance: str | None
    optional: bool
    implicit: bool
    conditions: tuple[str,...]


class ConfigurationStepKind(str,Enum):
    EMPTY = 'EMPTY'
    MERGE_MATERIAL = 'MERGE_MATERIAL'
    OPEN_PROVIDER = 'OPEN_PROVIDER'


@dataclass(frozen=True,slots=True)
class ConfigurationStep:
    """One SSA-like root version, retaining source payloads instead of copies."""
    ordinal: int
    kind: ConfigurationStepKind
    previous: int | None
    instance: str | None
    dependency: int | None
    material: str | None
    root_reference: int | None
    package: tuple[str,...]
    exclude_root_keys: tuple[str,...]
    conditions: tuple[str,...]


@dataclass(frozen=True,slots=True)
class ConfigurationCompositionObligation:
    code: str
    dependency: int | None
    material: str | None
    detail: str


@dataclass(frozen=True,slots=True)
class ConfigurationComposition:
    request: str
    source_candidate: int
    file_root: str
    main_resource: str
    profiles: tuple[ConfigurationProtocolSource,...]
    materials: tuple[ConfigurationMaterial,...]
    dependencies: tuple[ConfigurationDependency,...]
    steps: tuple[ConfigurationStep,...]
    obligations: tuple[ConfigurationCompositionObligation,...]
    root_version: int
    conditions: tuple[str,...]
    semantics: str = 'CONDITIONAL_ORDERED_COMPOSITION_NOT_RESOLVED_CFG_OR_RUNTIME_VALUE'


# Input expressions and ordered configuration writes, not runtime objects.
class OverrideExpressionKind(str,Enum):
    SCALAR = 'SCALAR'
    SEQUENCE = 'SEQUENCE'
    MAPPING = 'MAPPING'
    CALL = 'CALL'
    SWEEP = 'SWEEP'


@dataclass(frozen=True,slots=True)
class OverrideExpression:
    ordinal: int
    argument_index: int
    character_start: int
    character_end: int
    kind: OverrideExpressionKind
    literal: PythonLiteral | None
    quote: str | None
    children: tuple[int,...]
    entries: tuple[tuple[int,int],...]
    function: str | None
    argument_names: tuple[str | None,...]


class OverrideAction(str,Enum):
    CHANGE = 'CHANGE'
    ADD = 'ADD'
    FORCE_ADD = 'FORCE_ADD'
    DELETE = 'DELETE'


@dataclass(frozen=True,slots=True)
class ConfigurationOverride:
    ordinal: int
    source: StaticInputPort
    token: str
    action: OverrideAction
    key_or_group: str
    package: str | None
    field_path_candidate: tuple[str,...] | None
    value: int | None
    input_version: int
    output_version: int
    conditions: tuple[str,...]


@dataclass(frozen=True,slots=True)
class ConfigurationOverrideObligation:
    code: str
    override: int | None
    expression: int | None
    detail: str


@dataclass(frozen=True,slots=True)
class ConfigurationOverrides:
    composition: ProviderArtifact
    composition_version: int
    entry: ProviderArtifact
    protocol_sources: tuple[ConfigurationProtocolSource,...]
    parser_tools: tuple[ProviderArtifact,...]
    overrides: tuple[ConfigurationOverride,...]
    expressions: tuple[OverrideExpression,...]
    obligations: tuple[ConfigurationOverrideObligation,...]
    root_version: int
    conditions: tuple[str,...]
    semantics: str = 'INITIAL_ARGV_CONDITIONAL_WRITES_NOT_CURRENT_OR_RESOLVED_CFG'


# Conditional library input requests anchored to original program ports.
@dataclass(frozen=True,slots=True)
class ConfigurationLiteralCandidate:
    source: ContextEndpoint
    literal: PythonLiteral
    derivation: str
    alias_rows: tuple[int,...] = ()


@dataclass(frozen=True,slots=True)
class ConfigurationParameter:
    name: str
    formal: GlobalPort
    actuals: tuple[ContextEndpoint,...]
    candidates: tuple[ConfigurationLiteralCandidate,...]
    obligation: str | None


@dataclass(frozen=True,slots=True)
class ConfigurationOption:
    option: str
    candidate: ConfigurationLiteralCandidate
    argv_index: int


@dataclass(frozen=True,slots=True)
class ConfigurationSourceCandidate:
    protocol: str
    root: str | None
    name: str | None
    resource: str | None
    code_filename: str
    conditions: tuple[str,...]


@dataclass(frozen=True,slots=True)
class ConfigurationTaskAssociation:
    task_creation: GlobalPort
    decorator_result: GlobalPort
    factory_to_callee_alias_rows: tuple[int,...]
    task_to_argument_alias_rows: tuple[int,...]


@dataclass(frozen=True,slots=True)
class ConfigurationRequest:
    id: str
    protocol: str
    api: GlobalPort
    api_source: ProviderArtifact
    api_function_ast_sha256: str
    factory_result: GlobalPort
    call_reference: ProviderFactReference
    parameters: tuple[ConfigurationParameter,...]
    task_associations: tuple[ConfigurationTaskAssociation,...]
    initial_options: tuple[ConfigurationOption,...]
    sources: tuple[ConfigurationSourceCandidate,...]
    conditions: tuple[str,...]
    obligations: tuple[str,...]
    semantics: str = 'CONDITIONAL_REQUEST_CANDIDATES_NOT_CURRENT_ARGV_OR_COMPOSED_CFG'


# Versioned configuration algebra and read dependencies over shared inputs.
#
# Ordinals are scoped to their graph artifact. They are neither storage addresses
# nor Python identities; input references retain the original material/argv graph.
@dataclass(frozen=True,slots=True)
class ConfigurationInput:
    table: str
    owner: str
    ordinal: int


class ConfigurationExpressionKind(str,Enum):
    EMPTY = 'EMPTY'
    INPUT = 'INPUT'
    OPEN_PROVIDER = 'OPEN_PROVIDER'
    EMBED = 'EMBED'
    MERGE = 'MERGE'
    ASSIGN = 'ASSIGN'
    UPDATE = 'UPDATE'
    DELETE = 'DELETE'


@dataclass(frozen=True,slots=True)
class ConfigurationExpression:
    ordinal: int
    kind: ConfigurationExpressionKind
    inputs: tuple[int,...]
    source: ConfigurationInput | None
    path: tuple[str,...]
    exclude_root_keys: tuple[str,...]
    parent_stage: str
    parent_ordinal: int | None
    conditions: tuple[str,...]


@dataclass(frozen=True,slots=True)
class ConfigurationRoot:
    stage: str
    version: int
    expression: int


class ConfigurationValueKind(str,Enum):
    SCALAR = 'SCALAR'
    MAPPING = 'MAPPING'
    SEQUENCE = 'SEQUENCE'
    MISSING = 'MISSING'
    ABSENT = 'ABSENT'
    OPEN = 'OPEN'
    INTERPOLATION = 'INTERPOLATION'
    CYCLE = 'CYCLE'
    NONCONTAINER = 'NONCONTAINER'
    TYPE_CONFLICT = 'TYPE_CONFLICT'


@dataclass(frozen=True,slots=True)
class ConfigurationProjection:
    ordinal: int
    root: int
    path: tuple[str,...]
    kind: ConfigurationValueKind
    literal: PythonLiteral | None
    members_complete: bool | None
    origins: tuple[ConfigurationInput,...]
    dependencies: tuple[int,...]
    evidence: tuple[int,...]
    value_path: tuple[str,...] | None
    conditions: tuple[str,...]


@dataclass(frozen=True,slots=True)
class ConfigurationInterpolation:
    ordinal: int
    projection: int
    character_start: int
    character_end: int
    syntax: str
    kind: str
    target_path: tuple[str,...] | None
    dependencies: tuple[int,...]
    phase: str = 'LAZY_SELECTED_ROOT'


@dataclass(frozen=True,slots=True)
class ConfigurationReadDemand:
    ordinal: int
    operation: int
    snapshot: int | None
    path: tuple[str,...]
    phase: str
    source: ConfigurationInput | None
    projection: int | None


@dataclass(frozen=True,slots=True)
class ConfigurationValueObligation:
    code: str
    expression: int | None
    projection: int | None
    detail: str


@dataclass(frozen=True,slots=True)
class ConfigurationValues:
    overrides: ProviderArtifact
    profiles: tuple[ConfigurationProtocolSource,...]
    parser_tools: tuple[ProviderArtifact,...]
    expressions: tuple[ConfigurationExpression,...]
    roots: tuple[ConfigurationRoot,...]
    projections: tuple[ConfigurationProjection,...]
    interpolations: tuple[ConfigurationInterpolation,...]
    read_demands: tuple[ConfigurationReadDemand,...]
    obligations: tuple[ConfigurationValueObligation,...]
    selected_root: int
    conditions: tuple[str,...]
    semantics: str = 'CONDITIONAL_STATIC_INPUT_PROJECTIONS_NOT_CURRENT_TASK_CFG_OR_OBJECT_IDENTITY'


# Declared input algebra; distinct from runtime Python configuration objects.
@dataclass(frozen=True,slots=True)
class ConfigurationRegion:
    id: RegionID
    values: ProviderArtifact
    boundary: ProviderFactReference
    layer: str = 'DECLARED_INPUT_ALGEBRA_NOT_RUNTIME_OPERATIONS'


@dataclass(frozen=True,slots=True)
class ConfigurationExpressionOperation:
    id: OperationID
    expression: ConfigurationExpression


@dataclass(frozen=True,slots=True)
class ConfigurationExpressionValue:
    id: ValueID
    expression: int


@dataclass(frozen=True,slots=True)
class ConfigurationProjectionValue:
    id: ValueID
    projection: ConfigurationProjection


@dataclass(frozen=True,slots=True)
class ConfigurationReadOperation:
    id: OperationID
    demand: ConfigurationReadDemand


@dataclass(frozen=True,slots=True)
class ConfigurationInterpolationOperation:
    id: OperationID
    interpolation: ConfigurationInterpolation


@dataclass(frozen=True,slots=True)
class ConfigurationDeclaredVersion:
    id: ValueID
    root: ConfigurationRoot
