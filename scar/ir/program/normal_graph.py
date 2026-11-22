"""Normal graph."""

from dataclasses import dataclass
from enum import Enum
from scar.ir.program import source_productions as P
from scar.ir.program.control import LanguageCompletionKind
from scar.ir.program.model import ValueKind, ValueType
from scar.ir.program.signature_patterns import GroundType
from scar.ir.program.source_declarations import SourceTemplateKey
from scar.ir.program.source_productions import ProductionRegionID
from scar.ir.literals import PythonLiteral


# Normal-success dependency edges over the existing shared source IR.
#
# Nodes remain SourceProductionPlan regions/steps. Ports qualify their original
# IDs by source template; the store header qualifies templates by source version.
# Call-site labels pair entry/return edges without cloning functions or loops.
class PortKind(str, Enum):
    PARAMETER = 'PARAMETER'
    ARGUMENT = 'ARGUMENT'
    RESULT = 'RESULT'
    ENTER = 'ENTER'
    EXIT = 'EXIT'
    CALL_RESULT = 'CALL_RESULT'
    CALL_RETURN = 'CALL_RETURN'


class EdgeKind(str, Enum):
    DATA = 'DATA'
    STATE = 'STATE'
    CONTROL = 'CONTROL'
    DECLARE = 'DECLARE'


class Transition(str, Enum):
    DIRECT = 'DIRECT'
    CALL = 'CALL'
    RETURN = 'RETURN'
    BACKEDGE = 'BACKEDGE'
    RESUME = 'RESUME'


@dataclass(frozen=True, slots=True)
class Port:
    template: SourceTemplateKey
    owner: P.ProductionRegionID | P.ProductionStepID
    kind: PortKind
    ordinal: int = 0
    completion: LanguageCompletionKind | P.PatternCompletionKind | None = None
    target: P.ProductionControlTarget | None = None
    path: tuple[P.CompilationPathPart, ...] = ()

    def __post_init__(self):
        if type(self.template) is not SourceTemplateKey or type(self.owner) not in (P.ProductionRegionID, P.ProductionStepID):
            raise ValueError('graph port needs original qualified identities')
        if type(self.kind) is not PortKind or type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError('graph port needs an exact kind and nonnegative ordinal')
        completed = self.kind in (PortKind.RESULT, PortKind.EXIT, PortKind.CALL_RESULT, PortKind.CALL_RETURN)
        if completed != (self.completion is not None):
            raise ValueError('only result/exit ports carry a completion')


@dataclass(frozen=True, slots=True)
class CallSite:
    template: SourceTemplateKey
    step: P.ProductionStepID
    path: tuple[P.CompilationPathPart, ...] = ()


@dataclass(frozen=True, slots=True)
class Edge:
    source: Port
    target: Port
    kind: EdgeKind
    value_type: GroundType | None = None
    site: CallSite | None = None
    transition: Transition = Transition.DIRECT
    condition: Port | None = None
    when: bool | None = None

    def __post_init__(self):
        if type(self.source) is not Port or type(self.target) is not Port or type(self.kind) is not EdgeKind:
            raise ValueError('edge requires typed endpoints and kind')
        if (self.kind in (EdgeKind.DATA, EdgeKind.STATE)) != (self.value_type is not None):
            raise ValueError('data/state edges require a value type')
        state = type(self.value_type) is ValueType and self.value_type.kind is ValueKind.STATE
        if (self.kind is EdgeKind.STATE) != state:
            raise ValueError('state edges must carry an actual state domain')
        if (self.condition is None) != (self.when is None) or self.when is not None and type(self.when) is not bool:
            raise ValueError('a guarded edge needs a condition port and boolean choice')


# Sparse value/name/call facts referencing the existing normal graph ports.
@dataclass(frozen=True, slots=True)
class CodeScopeRef:
    unit: str
    ordinal: int

    def __post_init__(self):
        if type(self.unit) is not str or not self.unit or type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError('code scope needs its original unit and nonnegative source ordinal')


@dataclass(frozen=True, slots=True)
class CodeScopeWitness:
    owner: CodeScopeRef
    template: SourceTemplateKey
    region: P.ProductionRegionID
    source_scope: int


class FactTable(str, Enum):
    ALIASES='aliases'
    NAMES='names'
    FUNCTIONS='functions'
    CLASSES='classes'
    ATTRIBUTES='attributes'
    IMPORTS='imports'
    CONSTANTS='constants'
    CALLS='calls'
    CONTAINERS='containers'
    ITEMS='items'
    UPDATES='updates'
    ITERATIONS='iterations'
    OPERATIONS='operations'
    ACTIVATIONS='activations'


@dataclass(frozen=True, slots=True)
class FactRange:
    table: FactTable
    start: int
    stop: int

    def __post_init__(self):
        if type(self.table) is not FactTable or type(self.start) is not int or type(self.stop) is not int or not 0 <= self.start <= self.stop:
            raise ValueError('fact range needs a typed table and nonnegative half-open bounds')


@dataclass(frozen=True, slots=True)
class FactScopeGroup:
    owner: CodeScopeRef
    tables: tuple[FactRange, ...]


@dataclass(frozen=True, slots=True)
class ScopeActivation:
    source: CodeScopeRef
    target: CodeScopeWitness
    kind: P.CompilationEdgeKind
    step: P.ProductionStepID
    path: tuple[P.CompilationPathPart, ...]


class NamespaceKind(str, Enum):
    LEXICAL = 'LEXICAL'
    MODULE = 'MODULE'
    CLASS = 'CLASS'
    BUILTINS = 'BUILTINS'


@dataclass(frozen=True, slots=True)
class Namespace:
    kind: NamespaceKind
    owner: int
    name: str


@dataclass(frozen=True, slots=True)
class NameAccess:
    namespace: Namespace
    value: int
    write: bool


@dataclass(frozen=True, slots=True)
class Function:
    value: int
    template: SourceTemplateKey
    body: P.ProductionRegionID
    environment: P.DeferredEnvironment
    parameter_names: tuple[str, ...]
    formals: tuple[int, ...]
    defaults: tuple[int | None, ...]
    returns: int | None
    return_world: int | None
    body_scope: CodeScopeWitness | None = None


@dataclass(frozen=True, slots=True)
class Class:
    value: int
    template: SourceTemplateKey
    scope: int
    body: P.ProductionRegionID
    bases: tuple[int, ...]
    bases_pack: int | None = None
    body_scope: CodeScopeWitness | None = None


class AttributeAction(str,Enum):
    GET='GET'
    SET='SET'
    DELETE='DELETE'
    IMPORT_MEMBER='IMPORT_MEMBER'


@dataclass(frozen=True, slots=True)
class Attribute:
    receiver: int
    name: str
    value: int | None
    action: AttributeAction
    world: int
    result_world: int

    def __post_init__(self):
        if type(self.action) is not AttributeAction:raise ValueError('attribute action needs an exact kind')
        if (self.value is None)!=(self.action is AttributeAction.DELETE):raise ValueError('only attribute DELETE has no value port')


@dataclass(frozen=True, slots=True)
class Import:
    value: int
    module: str
    requested_module: str | None = None


@dataclass(frozen=True, slots=True)
class Constant:
    value: int
    literal: PythonLiteral


class ContainerKind(str, Enum):
    MAPPING = 'MAPPING'
    SEQUENCE = 'SEQUENCE'
    SET = 'SET'


@dataclass(frozen=True, slots=True)
class Container:
    value: int
    kind: ContainerKind
    elements: tuple[int, ...] = ()
    pairs: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class ItemAccess:
    receiver: int
    key: int | PythonLiteral
    value: int
    write: bool
    world: int
    result_world: int


@dataclass(frozen=True, slots=True)
class ContainerUpdate:
    target: int
    source: int
    append: bool


@dataclass(frozen=True, slots=True)
class Iteration:
    source: int
    value: int
    advance: bool


@dataclass(frozen=True, slots=True)
class Call:
    callee: int
    positional: tuple[int, ...]
    keywords: tuple[tuple[str, int], ...]
    result: int
    world: int
    result_world: int
    positional_pack: int | None = None
    keyword_pack: int | None = None


class LanguageOperationKind(str, Enum):
    UNARY = 'UNARY'
    BINARY = 'BINARY'
    INPLACE_BINARY = 'INPLACE_BINARY'
    COMPARE = 'COMPARE'


class OperatorSelector(str, Enum):
    ADD = 'Add'
    SUB = 'Sub'
    MULT = 'Mult'
    MAT_MULT = 'MatMult'
    DIV = 'Div'
    FLOOR_DIV = 'FloorDiv'
    MOD = 'Mod'
    POW = 'Pow'
    L_SHIFT = 'LShift'
    R_SHIFT = 'RShift'
    BIT_OR = 'BitOr'
    BIT_XOR = 'BitXor'
    BIT_AND = 'BitAnd'
    U_ADD = 'UAdd'
    U_SUB = 'USub'
    INVERT = 'Invert'
    NOT = 'Not'
    EQ = 'Eq'
    NOT_EQ = 'NotEq'
    LT = 'Lt'
    LT_E = 'LtE'
    GT = 'Gt'
    GT_E = 'GtE'
    IS = 'Is'
    IS_NOT = 'IsNot'
    IN = 'In'
    NOT_IN = 'NotIn'


@dataclass(frozen=True, slots=True)
class LanguageOperation:
    action: LanguageOperationKind
    selector: OperatorSelector
    operands: tuple[int, ...]
    result: int
    world: int
    result_world: int


@dataclass(frozen=True, slots=True)
class UnitFacts:
    unit: str
    module: str
    ports: tuple[Port, ...]
    aliases: tuple[tuple[int, int], ...]
    names: tuple[NameAccess, ...]
    functions: tuple[Function, ...]
    classes: tuple[Class, ...]
    attributes: tuple[Attribute, ...]
    imports: tuple[Import, ...]
    constants: tuple[Constant, ...]
    calls: tuple[Call, ...]
    containers: tuple[Container, ...] = ()
    items: tuple[ItemAccess, ...] = ()
    updates: tuple[ContainerUpdate, ...] = ()
    iterations: tuple[Iteration, ...] = ()
    operations: tuple[LanguageOperation, ...] = ()
    root_scope: CodeScopeWitness | None = None
    scope_groups: tuple[FactScopeGroup, ...] = ()
    activations: tuple[ScopeActivation, ...] = ()


# Conditional source-call connections; no runtime dispatch is asserted.
@dataclass(frozen=True, slots=True)
class GlobalPort:
    unit: str
    port: Port


class ConstructionPortKind(str, Enum):
    NEW_RESULT = 'NEW_RESULT'
    NEW_WORLD = 'NEW_WORLD'


@dataclass(frozen=True, slots=True)
class ConstructionPort:
    """A language-protocol interface, never an invented source-production port.

    Both anchors reference original source ports. The role separates the
    result of allocation from the state after allocation and before init.
    """
    call: GlobalPort
    source_class: GlobalPort
    role: ConstructionPortKind

    @property
    def unit(self):
        return self.call.unit


Endpoint = GlobalPort | ConstructionPort


@dataclass(frozen=True, slots=True)
class LanguageOperationSite:
    action: LanguageOperationKind
    selector: OperatorSelector
    operands: tuple[GlobalPort,...]
    result: GlobalPort
    world: GlobalPort
    result_world: GlobalPort


@dataclass(frozen=True, slots=True)
class Body:
    unit: str
    template: SourceTemplateKey
    region: ProductionRegionID


class RouteKind(str, Enum):
    ARGUMENT = 'ARGUMENT'
    INITIAL_DEFAULT = 'INITIAL_DEFAULT'
    CONDITIONAL_DEFAULT = 'CONDITIONAL_DEFAULT'
    PACK_POSITIONALS = 'PACK_POSITIONALS'
    PACK_KEYWORDS = 'PACK_KEYWORDS'
    FUNCTION_ENVIRONMENT = 'FUNCTION_ENVIRONMENT'
    CLASS_RECEIVER = 'CLASS_RECEIVER'
    STATE = 'STATE'


@dataclass(frozen=True, slots=True)
class Route:
    formal: GlobalPort
    kind: RouteKind
    actuals: tuple[Endpoint, ...]
    keyword_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ArgumentPacks:
    """Original pack objects; inference status is recorded on the connection.

    A receiver is prepended by method binding, never included in the pack.
    These are dependencies, not aliases between a pack and each parameter.
    """
    positional: GlobalPort | None
    keywords: GlobalPort | None = None
    receiver: Endpoint | None = None
    class_receiver: GlobalPort | None = None


class FunctionActivation(str,Enum):
    NORMAL='NORMAL'
    CREATE_DEFERRED_BODY='CREATE_DEFERRED_BODY'
    RESUME_BODY='RESUME_BODY'


class ArgumentBinding(str, Enum):
    NOT_MODELED = 'NOT_MODELED'
    DIRECT = 'DIRECT'
    CLOSED_PACK = 'CLOSED_PACK'
    SYMBOLIC_PACK = 'SYMBOLIC_PACK'


class ProjectionKind(str, Enum):
    POSITIONAL_ITEM = 'POSITIONAL_ITEM'
    KEYWORD_ITEM = 'KEYWORD_ITEM'
    POSITIONAL_TAIL = 'POSITIONAL_TAIL'
    KEYWORD_REMAINDER = 'KEYWORD_REMAINDER'


@dataclass(frozen=True, slots=True)
class ArgumentProjection:
    """Conditional content selection; the pack object is not the parameter.

    Shape/arity/key presence may remain unknown. Edges apply only where the
    original call's signature accepts its arguments and this source is selected.
    """
    formal: GlobalPort
    pack: GlobalPort
    kind: ProjectionKind
    index: int | None = None
    name: str | None = None
    excluded_names: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()

    def __post_init__(self):
        positional = self.kind in (ProjectionKind.POSITIONAL_ITEM, ProjectionKind.POSITIONAL_TAIL)
        if positional != (self.index is not None):
            raise ValueError('positional projection needs its original pack index')
        if self.index is not None and (type(self.index) is not int or self.index < 0):
            raise ValueError('projection index must be nonnegative')
        if (self.kind is ProjectionKind.KEYWORD_ITEM) != (self.name is not None):
            raise ValueError('keyword projection needs its parameter name')
        if self.excluded_names and self.kind is not ProjectionKind.KEYWORD_REMAINDER:
            raise ValueError('only keyword remainder excludes consumed names')


@dataclass(frozen=True, slots=True)
class CallInputEnvelope:
    """Pre-call provenance dependencies when data argument binding is open.

    These are original source ports, not an allocated argument tuple or pack.
    A formal may depend on any input, callable environment/default state or
    pre-call world; this records no reference identity or selected argument.
    """
    callee: GlobalPort
    positional: tuple[GlobalPort, ...]
    keywords: tuple[tuple[str, GlobalPort], ...]
    positional_pack: GlobalPort | None
    keyword_pack: GlobalPort | None
    world: GlobalPort
    formals: tuple[GlobalPort, ...]
    semantics: str = 'PRECALL_INPUT_PROVENANCE_ENVELOPE_NOT_ARGUMENT_BINDING'

    def __post_init__(self):
        if (type(self.positional) is not tuple or type(self.keywords) is not tuple
                or type(self.formals) is not tuple
                or any(type(item) is not tuple or len(item) != 2
                    or type(item[0]) is not str or not item[0] for item in self.keywords)):
            raise TypeError('call input envelope requires typed original argument collections')
        inputs = (self.callee, *self.positional, *(point for _, point in self.keywords),
            self.world, *(point for point in (self.positional_pack, self.keyword_pack) if point is not None))
        if (any(type(point) is not GlobalPort for point in (*inputs, *self.formals))
                or any(point.unit != self.callee.unit for point in inputs)
                or len(set(self.formals)) != len(self.formals)
                or len({point.unit for point in self.formals}) > 1
                or self.semantics != 'PRECALL_INPUT_PROVENANCE_ENVELOPE_NOT_ARGUMENT_BINDING'):
            raise ValueError('call input envelope must retain one original caller and data-formal boundary')


@dataclass(frozen=True,slots=True)
class FunctionConnection:
    """Conditional entry/exit wiring independent of the source operation kind."""
    source_callable: GlobalPort
    body: Body
    activation: FunctionActivation
    parameters: tuple[Route,...]
    returns: tuple[tuple[Endpoint,Endpoint],...]
    conditions: tuple[str,...]
    argument_packs: ArgumentPacks | None=None
    projections: tuple[ArgumentProjection, ...] = ()
    binding: ArgumentBinding = ArgumentBinding.DIRECT
    input_envelope: CallInputEnvelope | None = None

    def __post_init__(self):
        if self.input_envelope is not None and (type(self.input_envelope) is not CallInputEnvelope
                or self.activation is not FunctionActivation.NORMAL
                or self.binding is not ArgumentBinding.NOT_MODELED):
            raise ValueError('input provenance envelope requires an ordinary body with open argument binding')

    def to_link(self, result, callee, mode, external_symbol=None):
        return CallLink(result, callee, self.source_callable, self.body, mode, external_symbol,
            self.parameters, self.returns, self.conditions, argument_packs=self.argument_packs,
            projections=self.projections, binding=self.binding, input_envelope=self.input_envelope)


class AttributeOrigin(str,Enum):
    ATTRIBUTE='ATTRIBUTE'
    IMPORT_MEMBER='IMPORT_MEMBER'
    BUILTIN_GETATTR='BUILTIN_GETATTR'


@dataclass(frozen=True,slots=True)
class AttributeSite:
    origin: AttributeOrigin
    action: AttributeAction
    receiver: GlobalPort
    member: str
    result: GlobalPort | None
    assigned: GlobalPort | None
    world: GlobalPort
    result_world: GlobalPort
    dynamic_member: GlobalPort | None=None


@dataclass(frozen=True,slots=True)
class PropertyAccess:
    site: AttributeSite
    descriptor: GlobalPort
    receiver_class: GlobalPort
    accessor: GlobalPort | None
    connection: FunctionConnection | None
    conditions: tuple[str,...]
    state: 'PropertyAccessState'


class PropertyAccessState(str,Enum):
    CONNECTED='CONNECTED'
    ACCESSOR_ABSENT='ACCESSOR_ABSENT'
    ACCESSOR_NOT_CALLABLE='ACCESSOR_NOT_CALLABLE'
    ACCESSOR_TARGET_OPEN='ACCESSOR_TARGET_OPEN'
    ACCESSOR_ARGUMENT_PACK_OPEN='ACCESSOR_ARGUMENT_PACK_OPEN'
    SIGNATURE_INAPPLICABLE='SIGNATURE_INAPPLICABLE'


class AttributeStorageKind(str,Enum):
    INSTANCE_FIELDS='INSTANCE_FIELDS'
    CLASS_NAMESPACE='CLASS_NAMESPACE'
    MODULE_NAMESPACE='MODULE_NAMESPACE'


@dataclass(frozen=True,slots=True)
class AttributeStorageAccess:
    site: AttributeSite
    kind: AttributeStorageKind
    owner: Endpoint
    namespace_unit: str | None
    conditions: tuple[str,...]
    receiver_class: GlobalPort | None = None


@dataclass(frozen=True, slots=True)
class AttributeFactReference:
    """A location in an immutable input table, not a new operation identity.

    ATTRIBUTE/IMPORT_MEMBER select UnitFacts.attributes; BUILTIN_GETATTR
    selects UnitFacts.calls and retains the statically selected member name.
    """
    unit: str
    ordinal: int
    origin: AttributeOrigin
    member: str

    def __post_init__(self):
        if type(self.unit) is not str or not self.unit or type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError('attribute reference requires an original unit and table ordinal')
        if type(self.origin) is not AttributeOrigin or type(self.member) is not str or not self.member:
            raise ValueError('attribute reference requires its origin and selected member')


@dataclass(frozen=True, slots=True)
class AttributeContentLink:
    """Possible reaching store between original access sites.

    This is neither a strong update nor a content/version/object equality
    proof. The original read/write worlds remain on the referenced sites.
    """
    write: AttributeFactReference
    read: AttributeFactReference
    source: GlobalPort
    target: GlobalPort
    kind: AttributeStorageKind
    owner: Endpoint
    receiver_class: GlobalPort | None
    namespace_unit: str | None
    conditions: tuple[str, ...]
    semantics: str = 'FLOW_INSENSITIVE_MAY_REACHING_ATTRIBUTE_STORE'


@dataclass(frozen=True, slots=True)
class CallLink:
    result: GlobalPort
    callee: GlobalPort
    source_callable: GlobalPort | None
    body: Body | None
    mode: str
    external_symbol: str | None
    parameters: tuple[Route, ...] = ()
    returns: tuple[tuple[Endpoint, Endpoint], ...] = ()
    conditions: tuple[str, ...] = ('SOURCE_STATIC_CANDIDATE', 'CURRENT_CALLABLE_DESCRIPTOR')
    runtime_dispatch_verified: bool = False
    argument_packs: ArgumentPacks | None = None
    projections: tuple[ArgumentProjection, ...] = ()
    binding: ArgumentBinding = ArgumentBinding.NOT_MODELED
    input_envelope: CallInputEnvelope | None = None

    def __post_init__(self):
        if self.input_envelope is not None and (type(self.input_envelope) is not CallInputEnvelope
                or self.binding is not ArgumentBinding.NOT_MODELED or self.body is None
                or self.input_envelope.callee != self.callee):
            raise ValueError('call input provenance envelope differs from its original open-binding call')


@dataclass(frozen=True, slots=True)
class ContentLink:
    """Possible reference content, never a concrete content version proof."""
    source: Endpoint
    target: Endpoint
    container: Endpoint
    key: PythonLiteral | None = None
    semantics: str = 'FLOW_INSENSITIVE_MAY_CONTENT'


class ConstructionStage(str, Enum):
    NEW_LOOKUP_PENDING = 'NEW_LOOKUP_PENDING'
    SOURCE_NEW = 'SOURCE_NEW'
    OBJECT_NEW = 'OBJECT_NEW'
    EXPLICIT_OBJECT_NEW = 'EXPLICIT_OBJECT_NEW'
    INIT_LOOKUP_PENDING = 'INIT_LOOKUP_PENDING'
    SOURCE_INIT = 'SOURCE_INIT'
    OBJECT_INIT = 'OBJECT_INIT'
    RETURN_FOREIGN = 'RETURN_FOREIGN'
    RESULT_TYPE_OPEN = 'RESULT_TYPE_OPEN'


@dataclass(frozen=True, slots=True)
class ConstructionLink:
    """One conditional step of a class call or explicit object allocation.

    Outputs declare interfaces even when the target is still pending; they
    do not prove that an allocation/initialization occurred or produced data.
    """
    result: GlobalPort
    callee: GlobalPort
    source_class: GlobalPort
    stage: ConstructionStage
    inputs: tuple[Endpoint, ...]
    outputs: tuple[Endpoint, ...]
    connection: FunctionConnection | None
    conditions: tuple[str, ...]


# Static entry/activation evidence, distinct from runtime invocations.
class EntryReason(str,Enum):
    PROGRAM_ENTRY='PROGRAM_ENTRY'
    MODULE_REFERENCE='MODULE_REFERENCE'
    REQUESTED_MODULE='REQUESTED_MODULE'
    SOURCE_FUNCTION_CALL='SOURCE_FUNCTION_CALL'
    CLASS_BUILD='CLASS_BUILD'
    EAGER_REGION='EAGER_REGION'
    GENERATOR_ADVANCE='GENERATOR_ADVANCE'


@dataclass(frozen=True,slots=True)
class ScopeActivationEvidence:
    target: CodeScopeRef
    cause: Endpoint | CodeScopeRef | None
    reason: EntryReason
    conditions: tuple[str,...]=('STATIC_MAY_ACTIVATION','CODE_SCOPE_ACTIVE_NOT_PATH_SELECTED')


@dataclass(frozen=True,slots=True)
class ModuleEntryDemand:
    module: str
    producer: Endpoint | None
    conditions: tuple[str,...]=('NO_DECLARED_PYTHON_UNIT','SOURCE_OR_NATIVE_PROVIDER_CLASSIFICATION_PENDING')


class BodyDemandKind(str,Enum):
    ARGUMENT_PACK_BINDING_OPEN='ARGUMENT_PACK_BINDING_OPEN'
    DEFERRED_BODY_RESUME_OPEN='DEFERRED_BODY_RESUME_OPEN'


@dataclass(frozen=True,slots=True)
class BodyEntryDemand:
    target: CodeScopeWitness
    producer: GlobalPort
    kind: BodyDemandKind


@dataclass(frozen=True,slots=True)
class CallEntryDemand:
    callee: GlobalPort
    result: GlobalPort
    symbol: str | None
    conditions: tuple[str,...]=('CALLABLE_TARGET_OR_EXTERNAL_PROVIDER_OPEN',)
