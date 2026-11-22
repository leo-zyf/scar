"""Call composition."""

from __future__ import annotations

from scar.ir.program.model import TerminatorKind
from scar.ir.program.protocol_flow import ProtocolPort
from scar.ir.program.protocols import OwnedUse
from scar.ir.program.structured import ControlEdgeReference
from dataclasses import dataclass
from enum import Enum
import os
import re
from scar.ir.program.expansion import ContextStoreBudget, ContextUnavailable
from scar.ir.program.ids import RegionID
from scar.ir.program.inventory import BodyStatus
from scar.ir.record_codec import decode, encode
from scar.ir.semantics_v2 import SourceSnapshot
from scar.ir.v2._validation import record_errors
from scar.ir.v2.common import SourceReference
from scar.ir.v2.ids import OperationDefinitionID


# Compose original input recipes with return-aware local copy analyses.
#
# Recipes inventory possible routes. They neither choose an expected source nor
# provide copy proofs by themselves. Each segment below checks original Uses.
def _local_guards(ctx, guard, choices, calls, number, region):
    """Recipe CFG choices belong to this exact caller prefix, not all visits."""
    if not ctx.spend('composition_context_prefixes', number + 1):
        return None
    prefix = calls[:number]
    edges = list(ctx.selected_edges(guard, region=region))
    for (context, operation), successor in choices.items():
        if not ctx.spend('composition_guard_choices'):
            return None
        if context == prefix and ctx.region_of[operation] == region:
            edges.append(ControlEdgeReference(operation, successor))
    try:
        selected = ctx.selected_edges(tuple(edges))
    except ValueError:
        return {'conflict': True}
    if ctx.selection_ambiguity(selected) is not None:
        return {'ambiguous': True}
    return selected


def _frame(ctx, step):
    """Validate the recipe against prepared canonical actual/formal ports."""
    operation = ctx.operations[step['caller'].id]
    invoke = operation.terminator is not None and operation.terminator.kind is TerminatorKind.INVOKE
    actuals = operation.invoke_arguments if invoke else operation.call_arguments
    ordinal = step['ordinal']
    if (operation.callee != step['callee'] or ordinal >= len(actuals)
            or actuals[ordinal] != step['actual'].id):
        raise ValueError('composition frame must retain its original call actual ordinal')
    original = ctx.owned_use(operation.id, step['actual'].id)
    formals = ctx.blocks[ctx.regions[operation.callee].entry].arguments
    if original != step['actual'] or formals[ordinal] != step['formal'].id:
        raise ValueError('composition frame must retain its original actual and formal')
    return operation, actuals, formals


def _segment_flow(ctx, port, guard, local):
    """Keep recipe choices scoped to this segment alongside user guards."""
    previous = ctx._returns.guards
    ctx._returns.guards = ctx.selected_edges((*ctx.selected_edges(guard), *local))
    try:
        return ctx._fresh_copies(port, guard=local or None)
    finally:
        ctx._returns.guards = previous


def compose_route(ctx, expected, port, actual, recipe, *, guard, calls, choices, rooted):
    """One complete route, never a cross-recipe ANY/identity assertion."""
    result = {'match': None, 'segments': (), 'gaps': (), 'context_rooted': rooted}
    # An unresolved occurrence cannot supply an exact caller route. Reject
    # that prerequisite before scanning/materializing its entire path; the
    # original recipe remains in the report and in the route denominator.
    if recipe['kind'] == 'STATIC_CYCLE' or not rooted:
        return {**result, 'gaps': ('call_composition_occurrence_context_unresolved',)}
    if not ctx.spend('composition_recipe_steps', len(recipe['path'])):
        return {**result, 'gaps': ('call_composition_budget_exhausted',)}
    steps = tuple(step for step in recipe['path'] if step.get('kind') == 'CALL_ACTUAL_TO_FORMAL')
    result['frame_inventory'] = tuple((step['caller'].id, step['callee'], step['ordinal'],
        OwnedUse(step['caller'].id, step['actual'].id), step['formal'].origin) for step in steps)
    if not steps or len(steps) != len(calls):
        return {**result, 'gaps': ('call_composition_entry_route_unresolved',)}
    caller_regions = tuple(caller_region for _, caller_region, _ in calls)
    if len(set(caller_regions)) != len(caller_regions):
        return {**result, 'gaps': ('call_composition_recursive_context_unresolved',)}
    source_region = ctx.region_of[port.operation]
    anchor = None
    for number, step in enumerate(steps):
        if not ctx.spend('composition_anchor_frames'):
            return {**result, 'gaps': ('call_composition_budget_exhausted',)}
        operation, _, _ = _frame(ctx, step)
        caller_region = ctx.region_of[operation.id]
        if source_region == caller_region:
            anchor = number
        else:
            exported = ctx._returns.export_applicable(source_region, caller_region)
            if exported is None:
                return {**result, 'gaps': ('call_composition_budget_exhausted',)}
            if exported:
                anchor = number
    if anchor is None:
        return {**result, 'match': False}
    segments, source, unresolved = [], port, False
    result['anchor_frame'] = anchor
    for number in range(anchor, len(steps)):
        if not ctx.spend('composition_call_frames'):
            return {**result, 'segments': tuple(segments), 'gaps': ('call_composition_budget_exhausted',)}
        operation, actuals, formals = _frame(ctx, steps[number])
        region = ctx.region_of[operation.id]
        local = _local_guards(ctx, guard, choices, calls, number, region)
        if local is None or type(local) is dict:
            return {**result, 'segments': tuple(segments), 'gaps': (
                'call_composition_budget_exhausted' if local is None else
                'call_composition_cfg_context_unresolved',)}
        # Only actual membership is evidence. A still-available expected
        # source elsewhere in the set cannot stand in for these operands.
        before = None
        if number != anchor:
            before = _segment_flow(ctx, source, guard, local)
            if before is None:
                return {**result, 'segments': tuple(segments), 'gaps': tuple(ctx._returns.gaps)
                        or ('call_composition_local_flow_unresolved',)}
        seeds, bindings = [], []
        for ordinal, (identity, formal) in enumerate(zip(actuals, formals)):
            if not ctx.spend('composition_actual_formals'):
                return {**result, 'segments': tuple(segments), 'gaps': ('call_composition_budget_exhausted',)}
            target = OwnedUse(operation.id, identity)
            if number == anchor:
                # Initial up/down bridge reuses the independently given
                # expected port and existing exact unique-caller export.
                combined = ctx.selected_edges((*ctx.selected_edges(guard), *local))
                support = ctx.bind(expected, target, guard=combined or None)
                match = support['match']
            else:
                support = None
                match = ctx.owned_use(operation.id, identity).value in before.get(operation.id, ())
                if not match and unresolved:
                    match = None
            bindings.append({'ordinal': ordinal, 'actual': target, 'formal': formal,
                             'match': match, 'support': support})
            if match is True:
                seeds.append(formal)
            elif match is None:
                unresolved = True
        segment = {'caller': operation.id, 'caller_region': region, 'callee': operation.callee,
                   'context_prefix': calls[:number + 1], 'guards': local,
                   'recipe_ordinal': steps[number]['ordinal'], 'bindings': tuple(bindings),
                   'seeds': tuple(seeds)}
        segments.append(segment)
        if not seeds:
            return {**result, 'match': None if unresolved else False, 'segments': tuple(segments),
                    'gaps': ('call_composition_input_unresolved',) if unresolved else ()}
        entry = ctx.blocks[ctx.regions[operation.callee].entry]
        source = ProtocolPort(None, entry.operations[0], tuple(seeds), 'before')
    region = ctx.region_of[actual.operation]
    if ctx.region_of[source.operation] != region:
        return {**result, 'segments': tuple(segments), 'gaps': ('call_composition_terminal_region_unresolved',)}
    local = _local_guards(ctx, guard, choices, calls, len(steps), region)
    if local is None or type(local) is dict:
        return {**result, 'segments': tuple(segments), 'gaps': ('call_composition_cfg_context_unresolved',)}
    before = _segment_flow(ctx, source, guard, local)
    if before is None:
        return {**result, 'segments': tuple(segments), 'gaps': tuple(ctx._returns.gaps)
                or ('call_composition_terminal_flow_unresolved',)}
    match = ctx.owned_use(actual.operation, actual.use).value in before.get(actual.operation, ())
    if not match and unresolved:
        match = None
    return {**result, 'match': match, 'segments': tuple(segments), 'terminal_guards': local,
            'gaps': ('call_composition_input_unresolved',) if match is None else ()}


# Inventory v6 demand records, without raw/source or execution authority.
#
# These records distinguish analysis work from program facts. The coordinator
# must replay their seed membership, source identities, queue order and final
# body links. Decoding a record performs only structural validation.
def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum,
             name + " must be an exact integer >= " + str(minimum))


def _text(value, name):
    _require(type(value) is str and bool(value), name + " must be nonempty text")


def _path(value, name):
    _text(value, name)
    _require(os.path.isabs(value) and os.path.normpath(value) == value,
             name + " must be an absolute normalized path")


def _hash(value, name, prefix):
    _require(type(value) is str and re.fullmatch(re.escape(prefix) + r"[0-9a-f]{64}", value) is not None,
             name + " must be a canonical SHA256")


def _record(value, expected, name):
    _require(type(value) is expected, name + " has the wrong record type")
    errors = record_errors(value, expected, name)
    if errors:
        raise ValueError("; ".join(errors))


def _snapshot(value):
    _record(value, SourceSnapshot, "snapshot")
    _path(value.path, "snapshot.path")
    _hash(value.fingerprint, "snapshot.fingerprint", "sha256:")


def _source(value):
    _record(value, SourceReference, "source")
    _path(value.path, "source.path")
    _hash(value.fingerprint, "source.fingerprint", "sha256:")


class _WireRecord:
    def to_dict(self):
        self.__post_init__()
        return encode(self)

    @classmethod
    def from_dict(cls, document):
        """Decode structure only; independent raw/source replay is external."""
        return decode(cls, document)


class DemandOriginKind(str, Enum):
    LEGACY_OBSERVATION = "LEGACY_OBSERVATION"
    ENTERED_CALL_CODE = "ENTERED_CALL_CODE"
    SOURCE_CALL_CANDIDATE = "SOURCE_CALL_CANDIDATE"
    REMAINDER = "REMAINDER"


@dataclass(frozen=True, slots=True)
class LegacyObservation(_WireRecord):
    index: int
    kind: DemandOriginKind = DemandOriginKind.LEGACY_OBSERVATION

    def __post_init__(self):
        _integer(self.index, "legacy observation index")
        _require(self.kind is DemandOriginKind.LEGACY_OBSERVATION, "incorrect legacy observation kind")


@dataclass(frozen=True, slots=True)
class EnteredCallCode(_WireRecord):
    process_id: int
    code: str
    kind: DemandOriginKind = DemandOriginKind.ENTERED_CALL_CODE

    def __post_init__(self):
        _integer(self.process_id, "process_id", 1)
        _hash(self.code, "code", "pycode-v2:")
        _require(self.kind is DemandOriginKind.ENTERED_CALL_CODE, "incorrect entered-code kind")


@dataclass(frozen=True, slots=True)
class SourceCallCandidate(_WireRecord):
    call: OperationDefinitionID
    kind: DemandOriginKind = DemandOriginKind.SOURCE_CALL_CANDIDATE

    def __post_init__(self):
        _record(self.call, OperationDefinitionID, "source call")
        _require(self.kind is DemandOriginKind.SOURCE_CALL_CANDIDATE, "incorrect source-call kind")


@dataclass(frozen=True, slots=True)
class Remainder(_WireRecord):
    kind: DemandOriginKind = DemandOriginKind.REMAINDER

    def __post_init__(self):
        _require(self.kind is DemandOriginKind.REMAINDER, "incorrect remainder kind")


DemandOrigin = LegacyObservation | EnteredCallCode | SourceCallCandidate | Remainder
_ORIGIN_TYPES = (LegacyObservation, EnteredCallCode, SourceCallCandidate, Remainder)


@dataclass(frozen=True, slots=True)
class BodyExpansionV6(_WireRecord):
    definition: OperationDefinitionID
    source: SourceReference
    origins: tuple[DemandOrigin, ...]
    attempted: bool = False
    construction_operations: int = 0
    status: BodyStatus = BodyStatus.NOT_ATTEMPTED
    operation_budget_charged: int = 0
    construction_cost_known: bool = True

    def __post_init__(self):
        _record(self.definition, OperationDefinitionID, "definition")
        _source(self.source)
        _require(type(self.origins) is tuple and bool(self.origins), "body needs a nonempty tuple of origins")
        for origin in self.origins:
            _require(type(origin) in _ORIGIN_TYPES, "body origin must be a typed demand origin")
            origin.__post_init__()
        _require(len(set(self.origins)) == len(self.origins), "duplicate body demand origin")
        _require(type(self.attempted) is bool and type(self.construction_cost_known) is bool,
                 "expansion flags must be exact booleans")
        _require(type(self.status) is BodyStatus, "body status must be typed")
        for name in ("construction_operations", "operation_budget_charged"):
            _integer(getattr(self, name), name)
        _require(self.operation_budget_charged >= self.construction_operations,
                 "constructed operations cannot be refunded")
        if not self.attempted:
            _require(self.status is BodyStatus.NOT_ATTEMPTED and not self.construction_operations
                     and not self.operation_budget_charged and self.construction_cost_known,
                     "unattempted task cannot claim construction work")
        else:
            _require(self.status is not BodyStatus.NOT_ATTEMPTED, "attempted task needs an explicit result")


@dataclass(frozen=True, slots=True)
class EntryRoot(_WireRecord):
    """The entry module's source scope; this is not a lowered function body."""
    snapshot: SourceSnapshot
    initializer: OperationDefinitionID

    def __post_init__(self):
        _snapshot(self.snapshot)
        _record(self.initializer, OperationDefinitionID, "module initializer")


@dataclass(frozen=True, slots=True)
class ExpansionScheduleV6(_WireRecord):
    budget: ContextStoreBudget
    entries: tuple[BodyExpansionV6, ...]
    entry_roots: tuple[EntryRoot, ...]
    unavailable_contexts: tuple[ContextUnavailable, ...] = ()

    def __post_init__(self):
        _record(self.budget, ContextStoreBudget, "context budget")
        for name, expected in (("entries", BodyExpansionV6), ("entry_roots", EntryRoot),
                               ("unavailable_contexts", ContextUnavailable)):
            values = getattr(self, name)
            _require(type(values) is tuple, name + " must be a tuple")
            for value in values:
                _record(value, expected, name)
        _require(len({entry.definition for entry in self.entries}) == len(self.entries),
                 "a body definition can be scheduled only once")
        _require(len({root.initializer for root in self.entry_roots}) == len(self.entry_roots)
                 and len({root.snapshot.path for root in self.entry_roots}) == len(self.entry_roots),
                 "duplicate entry source scope")
        for entry in self.unavailable_contexts:
            _path(entry.path, "unavailable context path")
            _text(entry.reason, "unavailable context reason")
        _require(len({entry.path for entry in self.unavailable_contexts}) == len(self.unavailable_contexts),
                 "duplicate unavailable context")


class CallSourceAdmissionStatus(str, Enum):
    UNAVAILABLE = "UNAVAILABLE"
    INDEXED_WITHOUT_BODY = "INDEXED_WITHOUT_BODY"
    CORE_LINKED = "CORE_LINKED"
    MISMATCH = "MISMATCH"


@dataclass(frozen=True, slots=True)
class CallSourceAdmission(_WireRecord):
    origin: EnteredCallCode
    status: CallSourceAdmissionStatus
    snapshot: SourceSnapshot | None = None
    definition: OperationDefinitionID | None = None
    source: SourceReference | None = None
    core_region: RegionID | None = None
    reason: str = ""

    def __post_init__(self):
        _record(self.origin, EnteredCallCode, "admission origin")
        _require(type(self.status) is CallSourceAdmissionStatus, "admission status must be typed")
        _require(type(self.reason) is str, "admission reason must be text")
        if self.status in (CallSourceAdmissionStatus.UNAVAILABLE, CallSourceAdmissionStatus.MISMATCH):
            _require(all(value is None for value in (self.snapshot, self.definition, self.source, self.core_region)),
                     "unavailable/mismatched admission cannot retain a successful source/core link")
            _text(self.reason, "admission boundary reason")
            return
        _snapshot(self.snapshot)
        _record(self.definition, OperationDefinitionID, "indexed definition")
        _source(self.source)
        _require(self.snapshot.path == self.source.path and self.snapshot.fingerprint == self.source.fingerprint,
                 "indexed source differs from its snapshot")
        if self.status is CallSourceAdmissionStatus.CORE_LINKED:
            _record(self.core_region, RegionID, "core region")
        else:
            _require(self.core_region is None, "indexed-only admission cannot claim a core region")
            _text(self.reason, "no-body boundary reason")


@dataclass(frozen=True, slots=True)
class CallSourceScope(_WireRecord):
    """Group outcomes; the enclosing inventory stores the seed report once."""
    admissions: tuple[CallSourceAdmission, ...]

    def __post_init__(self):
        _require(type(self.admissions) is tuple, "admissions must be a tuple")
        for admission in self.admissions:
            _record(admission, CallSourceAdmission, "admission")
        _require(len({(item.origin.process_id, item.origin.code) for item in self.admissions}) == len(self.admissions),
                 "duplicate entered-code admission")


__all__ = [
    'compose_route',
    'BodyExpansionV6',
    'CallSourceAdmission',
    'CallSourceAdmissionStatus',
    'CallSourceScope',
    'DemandOrigin',
    'DemandOriginKind',
    'EnteredCallCode',
    'EntryRoot',
    'ExpansionScheduleV6',
    'LegacyObservation',
    'Remainder',
    'SourceCallCandidate',
]
