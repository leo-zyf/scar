"""Body observations."""

from dataclasses import dataclass
from scar.ir.program import normal_graph as G, source_productions as P, control as C
from scar.ir.program.ids import ProviderID
from scar.ir.program.normal_graph import GlobalPort, Body
from scar.ir.program.normal_graph import CodeScopeWitness, CodeScopeRef
from scar.ir.program.call_edges import (
    CapturedCode,
    CapturedSite,
    FrameEntry,
    FrameExit,
    CallRecordAnchor,
    InvocationKey,
)
from scar.ir.program.signature_patterns import GroundType, _ground
from typing import get_args
from scar.ir.program.source_template_links import SourceTemplateKey
from scar.ir.program.model import ValueType, ValueKind


# Shared source-body ports, without bodies, runtime identities or purity claims.
#
# These records check structural consistency only. Original plan hashes, entry
# instructions, source scopes and saved-value provenance require their producer.
BodyEntryInstruction = (P.EnterRootBody | P.EnterDeferredBody | P.EnterClassBody
    | P.EnterTypeParameters | P.EnterEvaluationRequest)


def _types(values):
    if type(values) is not tuple:
        raise ValueError('body port types require an exact tuple')
    for i, value in enumerate(values):
        _ground(value, 'body.types[' + str(i) + ']')


def _world_index(types):
    positions = tuple(i for i, t in enumerate(types)
        if type(t) is ValueType and t.kind is ValueKind.STATE)
    if len(positions) != 1:
        raise ValueError('body ingress/suspension requires one current-world port')
    return positions[0]


def _port(value, kind=None, owner=None, ordinal=None):
    if type(value) is not GlobalPort or type(value.unit) is not str or not value.unit or type(value.port) is not G.Port:
        raise ValueError('body interface needs an original qualified source port')
    p = value.port; p.__post_init__()
    if (kind is not None and p.kind is not kind or owner is not None and p.owner != owner
            or ordinal is not None and p.ordinal != ordinal or p.path
            or p.target is not None and p.completion is None):
        raise ValueError('body port kind, original owner, ordinal or path differs')
    return p


@dataclass(frozen=True, slots=True)
class SourceBodyCompletion:
    kind: C.LanguageCompletionKind | P.PatternCompletionKind
    target: P.ProductionControlTarget | None
    types: tuple[GroundType, ...]
    ports: tuple[GlobalPort, ...]
    exit: GlobalPort

    def __post_init__(self):
        if type(self.kind) not in (C.LanguageCompletionKind,P.PatternCompletionKind):
            raise ValueError('body completion requires its original completion Enum')
        looping = self.kind in (C.LanguageCompletionKind.BREAK,C.LanguageCompletionKind.CONTINUE)
        if ((looping and type(self.target) not in get_args(P.ProductionControlTarget))
                or not looping and self.target is not None):
            raise ValueError('only loop completions carry an original control target')
        _types(self.types)
        if type(self.ports) is not tuple or len(self.ports) != len(self.types):
            raise ValueError('body completion result count differs from original types')
        p = _port(self.exit,G.PortKind.EXIT,ordinal=0)
        if type(p.owner) not in (P.ProductionRegionID,P.ProductionStepID) or p.completion is not self.kind or p.target != self.target:
            raise ValueError('body completion exit cannot relabel its kind or target')
        for i, port in enumerate(self.ports):
            result = _port(port,G.PortKind.RESULT,p.owner,i)
            if ((port.unit,result.template) != (self.exit.unit,p.template)
                    or result.completion is not self.kind or result.target != self.target):
                raise ValueError('body completion result labels must match its original exit')


@dataclass(frozen=True, slots=True)
class SourceBodyIngress:
    region: Body
    parameters: tuple[GlobalPort, ...]
    types: tuple[GroundType, ...]
    suspension: GlobalPort | None = None
    delivery: C.LanguageCompletionKind | None = None
    saved: tuple[tuple[GlobalPort,GlobalPort], ...] = ()

    @property
    def world_index(self):
        return _world_index(self.types)

    def __post_init__(self):
        if (type(self.region) is not Body or type(self.region.unit) is not str or not self.region.unit
                or type(self.region.template) is not SourceTemplateKey
                or type(self.region.region) is not P.ProductionRegionID):
            raise ValueError('body ingress needs its fully qualified original region')
        _types(self.types)
        if type(self.parameters) is not tuple or len(self.parameters) != len(self.types):
            raise ValueError('body ingress parameter count differs from original types')
        for i, port in enumerate(self.parameters):
            p = _port(port,G.PortKind.PARAMETER,self.region.region,i)
            if (port.unit,p.template) != (self.region.unit,self.region.template):
                raise ValueError('body ingress parameters cross source units/templates')
        if (self.suspension is None) != (self.delivery is None) or (
                self.suspension is not None and (type(self.suspension) is not GlobalPort
                    or type(self.delivery) is not C.LanguageCompletionKind)):
            raise ValueError('temporal ingress requires both original suspension and delivery')
        if self.suspension is not None:
            step = _port(self.suspension,G.PortKind.ENTER,ordinal=0)
            if (type(step.owner) is not P.ProductionStepID
                    or (self.suspension.unit,step.template) != (self.region.unit,self.region.template)):
                raise ValueError('resume must retain its fully qualified original suspension')
        if type(self.saved) is not tuple or self.suspension is None and self.saved:
            raise ValueError('ordinary entry cannot bind suspension-saved references')
        seen = set()
        for pair in self.saved:
            if type(pair) is not tuple or len(pair) != 2:
                raise ValueError('saved binding requires its formal and original source reference')
            formal, actual = pair; _port(formal); _port(actual)
            if formal not in self.parameters or formal in seen or formal == self.parameters[self.world_index]:
                raise ValueError('saved formals must be unique ingress parameters, never the current world')
            if actual.port.kind not in (G.PortKind.PARAMETER,G.PortKind.ARGUMENT,G.PortKind.RESULT,G.PortKind.CALL_RESULT):
                raise ValueError('saved reference cannot be a control port')
            if (actual.unit,actual.port.template) != (self.region.unit,self.region.template):
                raise ValueError('saved reference must belong to the resume region source template')
            seen.add(formal)
        if self.suspension is None and self.world_index != len(self.types)-1:
            raise ValueError('ordinary source-body ABI ends with the current world')
        self.world_index  # Resume world can precede the saved-reference suffix.


@dataclass(frozen=True, slots=True)
class SourceBodySuspension:
    step: GlobalPort
    kind: P.SuspensionKind
    arguments: tuple[GlobalPort, ...]
    types: tuple[GroundType, ...]

    @property
    def world_index(self):
        return _world_index(self.types)

    def __post_init__(self):
        step = _port(self.step,G.PortKind.ENTER,ordinal=0)
        if type(step.owner) is not P.ProductionStepID or type(self.kind) is not P.SuspensionKind:
            raise ValueError('body suspension needs its original step and suspension Enum')
        _types(self.types)
        if type(self.arguments) is not tuple or len(self.arguments) != len(self.types):
            raise ValueError('suspension argument count differs from original types')
        for port in self.arguments:
            p = _port(port)
            if (port.unit,p.template) != (self.step.unit,step.template):
                raise ValueError('suspension arguments must retain the original step unit/template')
        self.world_index


@dataclass(frozen=True, slots=True)
class SourceBodyInterface:
    id: ProviderID
    body: Body
    scope: CodeScopeWitness
    plan_sha256: str
    entry: P.ProductionStepID
    instruction: BodyEntryInstruction
    ingresses: tuple[SourceBodyIngress, ...]
    completions: tuple[SourceBodyCompletion, ...]
    suspensions: tuple[SourceBodySuspension, ...]

    def __post_init__(self):
        if (type(self.id) is not ProviderID or type(self.body) is not Body or type(self.scope) is not CodeScopeWitness
                or type(self.body.unit) is not str or not self.body.unit or type(self.body.template) is not SourceTemplateKey
                or type(self.body.region) is not P.ProductionRegionID or type(self.entry) is not P.ProductionStepID
                or type(self.instruction) not in get_args(BodyEntryInstruction)
                or type(self.plan_sha256) is not str or len(self.plan_sha256) != 64
                or any(c not in '0123456789abcdef' for c in self.plan_sha256)):
            raise ValueError('body interface requires typed original identities, entry and plan receipt')
        if (type(self.scope.owner) is not CodeScopeRef or type(self.scope.owner.ordinal) is not int
                or self.scope.owner.ordinal < 0 or type(self.scope.source_scope) is not int or self.scope.source_scope < 0
                or self.scope.owner.unit != self.body.unit or self.scope.template != self.body.template
                or self.scope.region != self.body.region):
            raise ValueError('body interface scope and original body disagree')
        for rows, kind in ((self.ingresses,SourceBodyIngress),(self.completions,SourceBodyCompletion),
                (self.suspensions,SourceBodySuspension)):
            if type(rows) is not tuple or any(type(row) is not kind for row in rows):
                raise ValueError('body interface requires exact typed record tuples')
            for row in rows:row.__post_init__()
        if not self.ingresses or (self.ingresses[0].region != self.body
                or self.ingresses[0].suspension is not None or self.ingresses[0].delivery is not None):
            raise ValueError('first ingress must be the original ordinary body entry')
        steps = {s.step:s for s in self.suspensions}
        if len(steps) != len(self.suspensions):raise ValueError('duplicate suspension step')
        ports = []
        world = self.ingresses[0].types[self.ingresses[0].world_index]
        for i, ingress in enumerate(self.ingresses):
            if i and (ingress.suspension not in steps or ingress.delivery is None):
                raise ValueError('later ingress must name its retained suspension and delivery')
            if ingress.types[ingress.world_index] != world:raise ValueError('ingress world domain differs')
            if ingress.region.unit != self.body.unit:raise ValueError('ingress belongs to another source unit')
            ports.extend((*ingress.parameters,*(actual for _,actual in ingress.saved)))
        labels = set()
        for completion in self.completions:
            label = completion.kind,completion.target
            if label in labels:raise ValueError('duplicate body completion kind/target')
            labels.add(label)
            if (completion.exit.port.owner != self.body.region or completion.exit.port.template != self.body.template):
                raise ValueError('completion belongs to another body region/template')
            ports.extend((*completion.ports,completion.exit))
        for suspension in self.suspensions:
            if suspension.types[suspension.world_index] != world:raise ValueError('suspension world domain differs')
            ports.extend((suspension.step,*suspension.arguments))
        if any(p.unit != self.body.unit for p in ports):
            raise ValueError('body interface ports must retain the original source unit')


# Observed entries of shared source bodies with conditional source anchors.
#
# IDs and ports refer to original evidence. Multiple anchors remain alternatives;
# these records assert neither parameter aliases nor concrete namespace instances.
def _tuple(values, kind, label):
    if type(values) is not tuple or any(type(value) is not kind for value in values):
        raise ValueError(label + ' requires exact typed records in a tuple')


def _anchor(value):
    if type(value) is not CallRecordAnchor:
        raise ValueError('activation requires an original call-record anchor')
    value.__post_init__()
    if type(value.index) is not int or value.index < 1:
        raise ValueError('activation evidence requires a real positive frame-event index')


@dataclass(frozen=True, slots=True)
class SourceActivationAnchor:
    site: CapturedSite
    operation: GlobalPort
    scope: CodeScopeWitness
    arguments: tuple[GlobalPort, ...]
    types: tuple[GroundType, ...]
    completions: tuple[SourceBodyCompletion, ...]

    def __post_init__(self):
        if type(self.site) is not CapturedSite:
            raise ValueError('activation source anchor requires its captured caller site')
        self.site.__post_init__()
        p = _port(self.operation,G.PortKind.ENTER,ordinal=0)
        if (type(p.owner) is not P.ProductionStepID or type(self.scope) is not CodeScopeWitness
                or type(self.scope.owner) is not CodeScopeRef or type(self.scope.region) is not P.ProductionRegionID
                or self.scope.owner.unit != self.operation.unit or self.scope.template != p.template
                or type(self.scope.source_scope) is not int or self.scope.source_scope < 0):
            raise ValueError('activation anchor needs its original step and source code scope')
        self.scope.owner.__post_init__(); _types(self.types)
        _tuple(self.arguments,GlobalPort,'activation arguments')
        if len(self.arguments) != len(self.types):
            raise ValueError('activation argument count differs from original source types')
        for argument in self.arguments:
            a = _port(argument)
            if ((argument.unit,a.template) != (self.operation.unit,p.template)
                    or a.kind not in (G.PortKind.PARAMETER,G.PortKind.ARGUMENT,G.PortKind.RESULT,G.PortKind.CALL_RESULT)):
                raise ValueError('activation input must retain its original value port in the same unit/template')
        _tuple(self.completions,SourceBodyCompletion,'activation completions')
        labels = set()
        for completion in self.completions:
            completion.__post_init__()
            if (completion.exit.unit != self.operation.unit or completion.exit.port.template != p.template
                    or completion.exit.port.owner != p.owner):
                raise ValueError('activation completion must retain its original step output')
            label = completion.kind,completion.target
            if label in labels:raise ValueError('duplicate source-step completion kind/target')
            labels.add(label)


@dataclass(frozen=True, slots=True)
class SourceBodyActivation:
    id: ProviderID
    code_link: ProviderID
    interface: ProviderID
    code: CapturedCode
    path: tuple[CapturedSite, ...]
    suspended: bool
    caller_links: tuple[ProviderID, ...]
    path_index: int | None
    anchors: tuple[SourceActivationAnchor, ...]

    def __post_init__(self):
        if (any(type(value) is not ProviderID for value in (self.id,self.code_link,self.interface))
                or type(self.code) is not CapturedCode or type(self.suspended) is not bool):
            raise ValueError('body activation requires typed shared evidence/interface identities')
        self.code.__post_init__()
        if self.suspended != bool(self.code.flags & (0x20|0x80|0x200)):
            raise ValueError('activation suspension capability differs from the captured code flags')
        _tuple(self.path,CapturedSite,'activation path')
        for site in self.path:site.__post_init__()
        _tuple(self.caller_links,ProviderID,'activation caller links')
        _tuple(self.anchors,SourceActivationAnchor,'activation source anchors')
        if len(set(self.caller_links)) != len(self.caller_links):
            raise ValueError('activation repeats an original caller correspondence')
        if self.path_index is None:
            if self.caller_links or self.anchors:
                raise ValueError('unselected path cannot claim source caller links or anchors')
        elif (type(self.path_index) is not int or not 0 <= self.path_index < len(self.path)
                or not self.caller_links):
            raise ValueError('selected source path needs an in-range site and retained caller alternatives')
        for anchor in self.anchors:
            anchor.__post_init__()
            if anchor.site != self.path[self.path_index]:
                raise ValueError('activation source anchor differs from its original selected path site')


@dataclass(frozen=True, slots=True)
class SourceBodyActivationObservation:
    activation: ProviderID
    entry: FrameEntry
    exit: FrameExit | None

    def __post_init__(self):
        if (type(self.activation) is not ProviderID or type(self.entry) is not FrameEntry
                or type(self.entry.invocation) is not InvocationKey):
            raise ValueError('body activation observation requires its actual frame entry')
        self.entry.invocation.__post_init__(); _anchor(self.entry.anchor)
        if type(self.entry.code) is not str or not self.entry.code:
            raise ValueError('activation frame entry must retain its captured code identity')
        if self.exit is not None:
            if (type(self.exit) is not FrameExit or type(self.exit.invocation) is not InvocationKey
                    or type(self.exit.return_site) is not CapturedSite):
                raise ValueError('body activation exit must be an original frame record')
            self.exit.invocation.__post_init__(); _anchor(self.exit.anchor); self.exit.return_site.__post_init__()
            if (self.entry.invocation != self.exit.invocation or self.entry.code != self.exit.code
                    or self.exit.return_site.code != self.exit.code
                    or self.exit.anchor.index <= self.entry.anchor.index):
                raise ValueError('activation exit differs in invocation, code or temporal order')


@dataclass(frozen=True, slots=True)
class SourceBodyActivationParent:
    parent: ProviderID
    child: ProviderID
    observation: CallRecordAnchor
    direct: bool

    def __post_init__(self):
        if (type(self.parent) is not ProviderID or type(self.child) is not ProviderID
                or type(self.direct) is not bool):
            raise ValueError('activation parent relation needs typed identities and explicit directness')
        _anchor(self.observation)


@dataclass(frozen=True, slots=True)
class SourceBodyActivationFrontier:
    entry: FrameEntry
    reason: str

    def __post_init__(self):
        if type(self.entry) is not FrameEntry or type(self.reason) is not str or not self.reason:
            raise ValueError('activation frontier needs its original frame and a specific reason')
        _anchor(self.entry.anchor)
