"""Program regions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from scar.analysis.program_dependencies import DependencyCut
from scar.ir.program.dependencies import (
    DependencyCondition,
    DependencyEdge,
    DependencyEvidence,
    DependencyFrontier,
    DependencyKind as K,
    DependencyMeaning as M,
    DependencyPoint as Point,
)
from scar.ir.program.ids import LocationID, OperationID, RegionID, UseID, ValueID
from scar.ir.program.model import (
    AccessMode,
    Behavior,
    CompletionResult,
    Footprint,
    ReferenceValue,
    StateVersionValue,
    TerminatorKind,
    UseRole,
)
from scar.ir.program.resources import (
    CreatedResource,
    ResourceDefinition,
    ResourceLocation,
    ResourceRequirement,
    ResourceUse,
)
from scar.ir.program.control import PendingPayload
from scar.ir.program.drivers import BodyDriveResult
from scar.ir.program.resumption import ResumeInput, SavedInput
from scar.ir.program.stored_values import StoredValueWrite
from scar.ir.program.state_addresses import (
    ExistingAccess,
    InvokeStateOperand,
    ReferenceOwner,
    StateAddressRefinement,
    UnknownOwner,
    UnknownSelection,
    address_effect_uses,
)
from collections import defaultdict, deque
from contextlib import nullcontext
import heapq
import pickle
from scar.analysis.program_context_store import ContextStoreLimit, PreparedContextStore
from scar.ir.program.expansion import (
    BodyDemandKind,
    BodyExpansion,
    ContextUnavailable,
    ExpansionMeasurements,
    ExpansionSchedule,
)
from scar.ir.program.inventory import (
    BodyInventory,
    BodyStatus,
    CallCandidateWitness,
    CandidateCondition,
    DefinitionKind,
    ExecutionPhase,
    FileInventory,
    InventoryIssue,
)


# Describe boundaries of selected original operations, without transforming.
#
# Declared interfaces, an outside consumer, an implementation transfer and an
# observable escape are distinct facts. Missing effect/alias/observer contracts
# remain field-specific obligations. No callee body or SSA definition is copied.
class BoundaryKnowledge(str, Enum):
    KNOWN = "KNOWN"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True, slots=True)
class BoundaryFieldStatus:
    field: str
    knowledge: BoundaryKnowledge
    scope: str
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BoundaryPayload:
    position: int
    original_kind: str
    original_index: int | None
    use: Point | None
    value: Point | None
    landing: Point | None
    original: UseID | CompletionResult | PendingPayload | ResumeInput | SavedInput | BodyDriveResult


@dataclass(frozen=True, slots=True)
class BoundaryControlFlow:
    source: Point
    target: Point | None
    kind: str
    successor: int | None
    exit_name: str | None
    payload: tuple[BoundaryPayload, ...]
    source_selected: bool
    target_selected: bool
    structural: bool


@dataclass(frozen=True, slots=True)
class BoundaryStateOutcome:
    kind: str
    successor: int | None
    value: Point


@dataclass(frozen=True, slots=True)
class BoundaryStateAccess:
    operation: Point
    location: Point
    before: Point
    before_value: Point
    before_role: UseRole
    before_is_member: bool
    mode: AccessMode | None
    footprint: Footprint | None
    full_overwrite: bool | None
    reference: Point | None
    outcomes: tuple[BoundaryStateOutcome, ...]
    refinements: tuple[StateAddressRefinement, ...]
    address_operands: tuple[Point, ...]
    kind: str = "ACCESS"

    @property
    def mode_knowledge(self):
        if self.kind == "CONTROL_TRANSFER":
            return BoundaryKnowledge.NOT_APPLICABLE
        return (BoundaryKnowledge.KNOWN if self.mode is not None else
                BoundaryKnowledge.PARTIAL if self.refinements else BoundaryKnowledge.UNKNOWN)

    @property
    def range_knowledge(self):
        if self.kind == "CONTROL_TRANSFER":
            return BoundaryKnowledge.NOT_APPLICABLE
        if self.footprint is not None and self.footprint.kind.value != "UNKNOWN":
            return BoundaryKnowledge.KNOWN
        return BoundaryKnowledge.PARTIAL if self.refinements else BoundaryKnowledge.UNKNOWN


@dataclass(frozen=True, slots=True)
class BoundaryReference:
    value: Point
    referent_identity: str | None
    targets: tuple[Point, ...] | None
    selector: Footprint


@dataclass(frozen=True, slots=True)
class BoundaryResource:
    point: Point
    requirement: ResourceRequirement
    uses: tuple[ResourceUse, ...]
    definitions: tuple[ResourceDefinition, ...]
    locations: tuple[ResourceLocation, ...]


@dataclass(frozen=True, slots=True)
class BoundaryEscape:
    kind: str
    source: Point
    value: Point | None
    sink: Point | None
    observable: bool | None
    interface_only: bool
    witness: StoredValueWrite | DependencyEdge | None = None


@dataclass(frozen=True, slots=True)
class BoundaryEffect:
    operation: Point
    declared_behaviors: tuple[Behavior, ...]
    effects_complete: bool
    declared_may_raise: bool | None
    open_facets: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BoundaryAnalysis:
    cut: DependencyCut
    inputs: tuple[DependencyEdge, ...]
    consumers: tuple[DependencyEdge, ...]
    implementation_interfaces: tuple[DependencyEdge, ...]
    control: tuple[BoundaryControlFlow, ...]
    state: tuple[BoundaryStateAccess, ...]
    references: tuple[BoundaryReference, ...]
    alias_relations: tuple[DependencyEdge, ...]
    escapes: tuple[BoundaryEscape, ...]
    effects: tuple[BoundaryEffect, ...]
    resources: tuple[BoundaryResource, ...]
    created_resources: tuple[Point, ...]
    ordering: tuple[DependencyEdge, ...]
    fields: tuple[BoundaryFieldStatus, ...]
    frontiers: tuple[DependencyFrontier, ...]
    transformation_authorized: bool = False

    def field(self, name):
        return next(row for row in self.fields if row.field == name)

    @property
    def control_dependencies(self):
        return tuple(e for e in (*self.cut.inputs, *self.cut.outputs, *self.cut.interfaces)
                     if e.kind in (K.CONTROL, K.CONTROL_CONDITION))

    @property
    def reads(self):
        return tuple(row for row in self.state if row.mode in (AccessMode.READ, AccessMode.READ_WRITE)
            or any(effect.mode in (AccessMode.READ, AccessMode.READ_WRITE)
                   for refinement in row.refinements for effect in refinement.effects))

    @property
    def writes(self):
        return tuple(row for row in self.state if row.mode in (AccessMode.WRITE, AccessMode.READ_WRITE)
            or any(effect.mode in (AccessMode.WRITE, AccessMode.READ_WRITE)
                   for refinement in row.refinements for effect in refinement.effects))


def analyze_boundary(view, operations, *, internal_values=(), max_members=16384):
    """Mechanically describe an operation set in its original graph contexts.

    Knowledge scopes distinguish enumeration of the declared graph from
    closure of program behavior. In particular KNOWN declared ports do not
    establish pure effects, observer closure, single-entry structure or legal
    replacement. Optional internal_values select original block-argument ports
    only when their entire original block is selected. Phi/loop state ports
    retain their original Value identities.
    """
    operations = tuple(dict.fromkeys(operations))
    if not operations or any(type(point) is not Point or type(point.node) is not OperationID
                             for point in operations):
        raise TypeError('dependency cut needs qualified original Operation endpoints')
    if type(max_members) is not int or max_members < 1:
        raise ValueError('dependency cut needs a positive membership bound')
    if len(operations) > max_members:
        raise OverflowError('dependency cut membership budget exhausted')
    # Pin original inputs before a bounded cross-body query can run out of
    # time. Its known edges/frontiers must still produce a partial boundary,
    # rather than being discarded by a second, post-cut graph read.
    indexes = {namespace: view.load_index(namespace)
               for namespace in dict.fromkeys(point.namespace for point in operations)}
    cut = view.cut(operations, internal_values=internal_values, max_members=max_members)
    selected, members = set(cut.operations), set(cut.members)
    stored_writes = {}
    for namespace, index in indexes.items():
        by_operation = {}
        for write in index.graph.stored_value_writes:
            by_operation.setdefault(write.site.operation, []).append(write)
        stored_writes[namespace] = by_operation
    control, states, references, resources, effects, escapes, created, ordering = [], [], {}, [], [], [], [], []
    frontiers = list(cut.frontiers)
    groups = {}
    for p in cut.operations:
        groups.setdefault((p.namespace, p.context), set()).add(p.node)

    def point(original, node):
        return Point(original.namespace, node, original.context)

    def payload(original, arguments, landing=()):
        index = indexes[original.namespace]
        rows = []
        for position, argument in enumerate(arguments):
            use = point(original, argument) if type(argument) is UseID else None
            value = point(original, index.use_site(argument).use.value) if use is not None else None
            rows.append(BoundaryPayload(position, type(argument).__name__,
                argument.index if type(argument) is CompletionResult else None, use, value,
                point(original, landing[position]) if position < len(landing) else None, argument))
        return tuple(rows)

    for original in cut.operations:
        index = indexes[original.namespace]
        operation = index.operations[original.node]
        for use in operation.uses:
            value = index.values[use.value]
            if type(value) is ReferenceValue:
                key = point(original, value.id)
                references[key] = BoundaryReference(key, value.referent_identity,
                    None if value.targets is None else tuple(point(original, t) for t in value.targets), value.selector)
        for produced in operation.results:
            value = index.values[produced]
            if type(value) is ReferenceValue:
                key = point(original, value.id)
                references[key] = BoundaryReference(key, value.referent_identity,
                    None if value.targets is None else tuple(point(original, t) for t in value.targets), value.selector)

        state_start = len(states)

        def state_row(before, location, mode, footprint, overwrite, reference, outcomes, site, kind="ACCESS"):
            use = index.use_site(before).use
            refinement = index._state_addresses.get(site)
            refinements = (refinement,) if refinement is not None else ()
            operands = []
            for refinement in refinements:
                for effect in refinement.effects:
                    if type(effect.owner) is ReferenceOwner:
                        operands.append(point(original, effect.owner.use))
                    operands.extend(point(original, u) for u in address_effect_uses(effect))
            states.append(BoundaryStateAccess(original, point(original, location), point(original, before),
                point(original, use.value), use.role, point(original, use.value) in members,
                mode, footprint, overwrite, point(original, reference) if reference is not None else None,
                outcomes, refinements, tuple(dict.fromkeys(operands)), kind))

        for ordinal, access in enumerate(operation.accesses):
            outcomes = (() if access.after is None else
                        (BoundaryStateOutcome("AFTER_OPERATION", None, point(original, access.after)),))
            state_row(access.before, access.location, access.mode, access.footprint, access.full_overwrite,
                      access.reference, outcomes, ExistingAccess(operation.id, ordinal))
        term = operation.terminator
        if term is not None and term.kind is TerminatorKind.INVOKE:
            for use in operation.uses:
                before = index.values[use.value]
                if type(before) is not StateVersionValue or use.id not in operation.invoke_arguments:
                    continue
                outcomes = []
                for ordinal, successor in enumerate(term.successors):
                    block = index._block_records[successor.block]
                    for actual, formal in zip(successor.arguments, block.arguments):
                        value = index.values[formal]
                        if type(value) is not StateVersionValue or value.location != before.location:
                            continue
                        produced = type(actual) is CompletionResult
                        passed = (type(actual) is UseID and index.use_site(actual).use.value == before.id)
                        if produced or passed:
                            outcomes.append(BoundaryStateOutcome(successor.kind.value, ordinal, point(original, formal)))
                state_row(use.id, before.location, None, None, None, None, tuple(outcomes),
                          InvokeStateOperand(operation.id, use.id), "INVOKE")

        represented = {s.before.node for s in states[state_start:]}
        for use in operation.uses:
            before = index.values[use.value]
            if use.id in represented or type(before) is not StateVersionValue:
                continue
            outcomes = []
            if term is not None:
                for ordinal, successor in enumerate(term.successors):
                    block = index._block_records[successor.block]
                    for argument, formal in zip(successor.arguments, block.arguments):
                        if argument == use.id and type(index.values[formal]) is StateVersionValue:
                            kind = getattr(getattr(successor, "kind", None), "value", term.kind.value)
                            outcomes.append(BoundaryStateOutcome(kind, ordinal, point(original, formal)))
                if term.kind in (TerminatorKind.RETURN, TerminatorKind.RAISE) and use.id in term.values:
                    outcomes.append(BoundaryStateOutcome(term.kind.value, None, point(original, use.value)))
            if operation.callee is not None:
                outcomes.extend(BoundaryStateOutcome("CALL_RESULT", None, point(original, value.id))
                    for value in (index.values[v] for v in operation.results)
                    if type(value) is StateVersionValue and value.location == before.location)
            transfer = (term is not None and term.kind in (TerminatorKind.JUMP, TerminatorKind.BRANCH,
                TerminatorKind.RETURN, TerminatorKind.RAISE) and operation.callee is None)
            state_row(use.id, before.location, None, None, None, None, tuple(outcomes), None,
                      "CONTROL_TRANSFER" if transfer else "STATE_INTERFACE")

        if term is not None:
            if term.kind in (TerminatorKind.RETURN, TerminatorKind.RAISE):
                control.append(BoundaryControlFlow(original, None, term.kind.value, None, term.exit,
                    payload(original, term.values), True, False, False))
                for use_id in term.values:
                    value = index.use_site(use_id).use.value
                    if type(index.values[value]) is ReferenceValue:
                        escapes.append(BoundaryEscape("OWNER_REGION_COMPLETION", point(original, use_id),
                            point(original, value), None, None, True))
            for ordinal, successor in enumerate(term.successors):
                block = index._block_records[successor.block]
                target = point(original, block.operations[0])
                if target not in selected:
                    kind = getattr(getattr(successor, "kind", None), "value", term.kind.value)
                    control.append(BoundaryControlFlow(original, target, kind, ordinal, None,
                        payload(original, successor.arguments, block.arguments), True, False, False))

        may_raise = (True if term is not None and (term.kind is TerminatorKind.RAISE or any(
            getattr(getattr(s, "kind", None), "value", None) == "RAISE"
            for s in getattr(term, "successors", ()))) else None)
        open_facets = ("allocation", "release", "alias", "observable_escape", "rng", "external", "termination")
        if not operation.effects_complete:
            open_facets += ("read_coverage", "write_coverage", "ordering_coverage")
        if may_raise is None:
            open_facets += ("exception_coverage",)
        effects.append(BoundaryEffect(original, operation.behaviors, operation.effects_complete, may_raise, open_facets))
        for stored in stored_writes[original.namespace].get(operation.id, ()):
            use = index.use_site(stored.payload).use
            escapes.append(BoundaryEscape("DECLARED_STATE_SLOT_STORE", point(original, stored.payload),
                point(original, use.value), original, None, False, stored))

    # INVOKE completion values are original block arguments, not ordinary
    # operation.results. Keep their reference declarations as well.
    for member in cut.members:
        if type(member.node) is ValueID:
            value = indexes[member.namespace].values[member.node]
            if type(value) is ReferenceValue:
                references[member] = BoundaryReference(member, value.referent_identity,
                    None if value.targets is None else tuple(point(member, t) for t in value.targets), value.selector)

    for (namespace, context), identities in groups.items():
        index = indexes[namespace]
        representative = next(p for p in cut.operations if p.namespace == namespace and p.context == context)
        for region in index.graph.regions:
            for block in region.blocks:
                for left, right in zip(block.operations, block.operations[1:]):
                    if (left in identities) != (right in identities):
                        control.append(BoundaryControlFlow(point(representative, left), point(representative, right),
                            "SEQUENCE", None, None, (), left in identities, right in identities, True))
                source_id = block.operations[-1]
                source = index.operations[source_id]
                if source_id in identities or source.terminator is None:
                    continue
                for ordinal, successor in enumerate(source.terminator.successors):
                    target_block = index._block_records[successor.block]
                    if target_block.operations[0] in identities:
                        kind = getattr(getattr(successor, "kind", None), "value", source.terminator.kind.value)
                        origin = point(representative, source_id)
                        control.append(BoundaryControlFlow(origin, point(representative, target_block.operations[0]),
                            kind, ordinal, None, payload(origin, successor.arguments, target_block.arguments), False, True, False))
        owners = {index._blocks[index._operation_blocks[identity]] for identity in identities}
        selected_requirements = {u.requirement for u in index.graph.resource_uses if u.operation in identities}
        selected_requirements.update(r.id for r in index.graph.resource_requirements if r.owner in identities or r.owner in owners)
        for requirement in index.graph.resource_requirements:
            if requirement.id not in selected_requirements:
                continue
            uses = tuple(u for u in index.graph.resource_uses if u.operation in identities and u.requirement == requirement.id)
            definitions = tuple(d for d in index.graph.resources if any(u.definition == d.id for u in uses))
            locations = tuple(l for l in index.graph.resource_locations if any(d.id == l.resource for d in definitions))
            resources.append(BoundaryResource(point(representative, requirement.id), requirement, uses, definitions, locations))
        created.extend(point(representative, d.id) for d in index.graph.resources
            if type(d.origin) is CreatedResource and d.origin.operation in identities)
        for ordered in index.graph.ordering:
            if ordered.before in identities or ordered.after in identities:
                ordering.append(DependencyEdge(point(representative, ordered.before), point(representative, ordered.after),
                    K.ORDER, M.ORDER, (DependencyEvidence("BOUNDARY_ORIGINAL_ORDER_V1", (namespace.graph,)),)))

    # A downstream use is a boundary consumer, not evidence that it observes
    # identity/content or escapes the program. Implementation transfers stay
    # separate even when their actual is an object reference.
    data_kinds = frozenset((K.VALUE_USE, K.OPERAND, K.RESULT, K.BLOCK_ARGUMENT,
        K.BINDING_PAYLOAD, K.BINDING_READ, K.STATE_INPUT, K.STATE_OUTPUT, K.STATE_VALUE, K.PROJECTION))
    inputs = tuple(e for e in cut.inputs if e.kind in data_kinds)
    consumers = tuple(e for e in cut.outputs if e.kind in data_kinds)
    aliases = tuple(e for e in (*cut.inputs, *cut.outputs, *cut.interfaces) if e.kind is K.ALIAS)
    ordering.extend(e for e in (*cut.inputs, *cut.outputs) if e.kind is K.ORDER)
    for edge in consumers:
        value = edge.source if type(edge.source.node) is ValueID else None
        escapes.append(BoundaryEscape("OUTSIDE_CONSUMER", edge.source, value, edge.target, None, False, edge))
    for edge in cut.interfaces:
        escapes.append(BoundaryEscape("IMPLEMENTATION_INTERFACE", edge.source,
            edge.source if type(edge.source.node) is ValueID else None, edge.target, None, True, edge))

    known, partial, unknown = BoundaryKnowledge.KNOWN, BoundaryKnowledge.PARTIAL, BoundaryKnowledge.UNKNOWN
    interrupted = any(f.kind == "DEPENDENCY_BOUNDARY_ENUMERATION_OPEN"
        or "BUDGET" in f.kind or "UNENUMERATED" in f.kind for f in frontiers)
    access_closed = all(e.effects_complete for e in effects) and all(
        row.mode is not None or row.kind == "CONTROL_TRANSFER" for row in states)
    statuses = [
        BoundaryFieldStatus("inputs", partial if interrupted else known, "CURRENT_REGISTERED_RELATIONS"),
        BoundaryFieldStatus("consumers", partial if interrupted else known, "CURRENT_REGISTERED_RELATIONS"),
        BoundaryFieldStatus("implementation_interfaces", partial, "DECLARED_INTERFACES",
            ("Interface edges do not establish implementation/effect closure or observable escape.",)),
        BoundaryFieldStatus("control", known, "CANONICAL_CFG_AND_COMPLETION_PORTS",
            ("Sequence is structural; this does not prove a single-entry replaceable Region.",)),
        BoundaryFieldStatus("control_dependencies", partial if frontiers else known, "CURRENT_REGISTERED_CONTROL_RELATIONS",
            ("Control conditions and MAY strengths remain on their original dependency edges.",)),
        BoundaryFieldStatus("state", known, "CANONICAL_STATE_PORTS",
            ("Phi/loop versions and contexts remain original SSA ports; INVOKE mode may be unknown.",)),
        BoundaryFieldStatus("reads", known if access_closed else partial if states else unknown, "DECLARED_ACCESS_MODES_AND_REFINEMENTS",
            ("Known modes/paths do not close unknown operation effects or alias targets.",)),
        BoundaryFieldStatus("writes", known if access_closed else partial if states else unknown, "DECLARED_ACCESS_MODES_AND_REFINEMENTS",
            ("Normal and raise post-states are separate; failure is not assumed to preserve input state.",)),
        BoundaryFieldStatus("aliases", partial if references or aliases else unknown, "REFERENCE_DECLARATIONS",
            ("Targets are may-targets; reference identity is not content or storage identity.",)),
        BoundaryFieldStatus("escapes", unknown, "OBSERVABLE_BEHAVIOR",
            ("Outside consumers, stores and completion/interface transfers are not observer-closure proofs.",)),
        BoundaryFieldStatus("resources", partial if resources or created else unknown, "DECLARED_REQUIREMENTS",
            ("Requirements/created resources do not establish physical allocation, placement or completion.",)),
        BoundaryFieldStatus("ordering", partial, "EXPLICIT_ORDER_AND_STATE_RELATIONS",
            ("Sequence/CFG edges are not promoted to necessary ordering; additional effect constraints may remain.",)),
    ]
    for name in ("allocates", "frees", "rng", "external_effects"):
        statuses.append(BoundaryFieldStatus(name, unknown, "PROGRAM_EFFECTS",
            ("No unified per-operation contract for this facet is supplied by the canonical records.",)))
    statuses.append(BoundaryFieldStatus("may_raise", partial if any(e.declared_may_raise for e in effects) else unknown,
        "DECLARED_COMPLETIONS", ("A declared raise route is preserved; absent routes are not silently declared impossible.",)))
    return BoundaryAnalysis(cut, inputs, consumers, cut.interfaces, tuple(control), tuple(states),
        tuple(references.values()), aliases, tuple(escapes), tuple(effects), tuple(resources), tuple(created),
        tuple(dict.fromkeys(ordering)), tuple(statuses), tuple(dict.fromkeys(frontiers)))


# Global, bounded source-body demand scheduling, separate from program effects.
_FUNCTIONS = (DefinitionKind.FUNCTION, DefinitionKind.ASYNC_FUNCTION)
_RANK = {BodyDemandKind.OBSERVED_CODE: 0, BodyDemandKind.CANDIDATE_CALL: 1,
         BodyDemandKind.REMAINDER: 2}
_BUDGET_REASON = "independent body/aggregate operation construction budget exhausted"
_SCOPE_REASON = "this scope requires a class/lambda/comprehension interface beyond the current body lowerer"


def _error_reason(error):
    # Never persist a private/random scratch filename in a replayed report.
    if isinstance(error, ContextStoreLimit):
        return str(error)
    return "private context unavailable: " + type(error).__name__


def expand_inventory(discovery, runtime_builder, budget, context_budget, ordered_files, *, interface_model=False,
                     body_profile=None, call_source_report=None, source_workspace=None):
    # Local import avoids a module cycle: the source adapter owns FileInventory
    # construction; this module owns only its global analysis work schedule.
    from scar.analysis.program_inventory import _Inventory, _attach_bodies, _body_connections, _CallLinker, prepare_source
    from scar.analysis.function_interfaces import FunctionInterfaceBuilder
    from scar.ir.program.functions import BodyLoweringProfile
    from scar.ir.program.interface_validation import require_valid_frame_model
    selected_profile = body_profile or BodyLoweringProfile.BOUND_FORMALS
    interfaces = FunctionInterfaceBuilder(profile=selected_profile) if interface_model else None
    frames, structured = [], []
    is_v6 = call_source_report is not None
    if is_v6:
        from scar.analysis.call_demand import (
            CallDemandQueue, _matched_call_sources, entry_frontier_initializers, entry_roots_for,
        )
        from scar.ir.program.call_composition import EnteredCallCode, LegacyObservation, SourceCallCandidate
    files, unavailable, bodies, issues = [], {}, {}, defaultdict(list)
    module_initializers = {}
    store_scope = (nullcontext(source_workspace.store) if source_workspace is not None
                   else PreparedContextStore(context_budget))
    with store_scope as store:
        # Index all files before spending the first body/operation allowance.
        for file in ordered_files:
            path = file.snapshot.path
            context = builder = None
            try:
                context = (source_workspace.prepare(path) if source_workspace is not None
                           else prepare_source(path, project_root=discovery.project_root))
                if context.semantics.sources[path].fingerprint != file.snapshot.fingerprint:
                    raise ValueError("source changed between discovery and preparation")
                builder = _Inventory(context, 0, 0, runtime_builder=runtime_builder)
                indexed = builder.index()
                module_initializers[path] = builder.module
                if interfaces is not None:
                    interfaces.index_file(context, indexed, builder.nodes)
            except (OSError, ValueError, TypeError, KeyError, SyntaxError, RecursionError) as error:
                if runtime_builder is not None:
                    runtime_builder.file_failed(path)
                if is_v6 and source_workspace is not None:
                    unavailable[path] = "source workspace unavailable: " + str(error)
                indexed = FileInventory(file.snapshot, (), (), (), (),
                    (InventoryIssue(path, "file preparation/inventory failed: " + str(error)),))
            else:
                if source_workspace is None:
                    try:
                        # Files without a supported scope never need a heavy body
                        # context, though their light/source evidence is preserved.
                        if any(d.kind in _FUNCTIONS for d in indexed.definitions):
                            store.put(context)
                    except (OSError, ValueError, TypeError, KeyError, RecursionError, pickle.PickleError) as error:
                        unavailable[path] = _error_reason(error)
            finally:
                context = builder = None
            files.append(indexed)

        # Admit witnesses only as an owner is demanded. Eagerly admitting all
        # unobserved calls would spend this budget before dynamic local targets
        # of an observed body become available.
        linker = _CallLinker(discovery, tuple(files), budget.max_call_witnesses)
        definitions = {definition.id: definition for file in files for definition in file.definitions}
        functions = {key: value for key, value in definitions.items() if value.kind in _FUNCTIONS}
        file_by_definition = {d.id: file.snapshot.path for file in files for d in file.definitions}
        calls = {call.operation: call for file in files for call in file.calls}
        by_owner = defaultdict(list)
        for file in files:
            for call in file.calls:
                by_owner[call.execution_owner].append(call)

        def enqueue_v6_direct_calls(owner, priority):
            for call in by_owner[owner]:
                for witness in linker.link(call).candidate_witnesses:
                    if witness.definition in functions:
                        demand_queue.add(witness.definition, priority, SourceCallCandidate(call.operation))

        observations = defaultdict(list)
        for binding in runtime_builder.bindings.values() if runtime_builder is not None else ():
            for definition in binding.candidate_definitions:
                if definition in functions:
                    observations[definition].append(binding.observation_index)
        demand_queue = None
        entry_roots = ()

        if is_v6:
            demand_queue = CallDemandQueue(functions)
            entry_roots = entry_roots_for(files, module_initializers, discovery.entry)

            for definition in sorted(observations, key=lambda key: key.wire):
                for observation_index in sorted(set(observations[definition])):
                    demand_queue.add(definition, CallDemandQueue.OBSERVED,
                                     LegacyObservation(observation_index))

            # A D0 seed is admitted only when its replayed path/fingerprint and
            # exact definition/source match this complete source inventory.
            for (process_id, code), match in sorted(_matched_call_sources(call_source_report, files).items()):
                if match is None:
                    continue
                _, definition = match
                if definition.kind in _FUNCTIONS:
                    demand_queue.add(definition.id, CallDemandQueue.OBSERVED,
                                     EnteredCallCode(process_id, code))

            # Entry and statically discovered import-module initializers form
            # the source frontier. Only entry itself is recorded as EntryRoot;
            # imported modules remain candidate scopes, not claimed executions.
            for initializer in entry_frontier_initializers(discovery, source_workspace, module_initializers):
                enqueue_v6_direct_calls(initializer, CallDemandQueue.ENTRY)

            for definition in sorted(functions, key=lambda key: key.wire):
                demand_queue.add_remainder(definition)
        else:
            requests, pending, finished, entries = {}, [], set(), []
            serial = 0

            def enqueue(definition, kind, *, via_call=None, observed=()):
                nonlocal serial
                if definition not in functions or definition in finished:
                    return
                previous = requests.get(definition)
                if previous is not None and _RANK[previous.kind] <= _RANK[kind]:
                    return
                source = functions[definition].source
                request = BodyExpansion(definition, source, kind, tuple(sorted(set(observed))), via_call)
                requests[definition] = request
                serial += 1
                heapq.heappush(pending, (_RANK[kind], source.path, source.start_line,
                                        source.start_column, definition.wire, serial, request))

            for definition in sorted(observations, key=lambda key: key.wire):
                enqueue(definition, BodyDemandKind.OBSERVED_CODE, observed=observations[definition])

            # Preserve the historical v1-v5 closure, sorting and admission order.
            expanded_owners = set()

            def expand_owners(owners):
                frontier = deque(owners)
                while frontier:
                    owner = frontier.popleft()
                    if owner in expanded_owners:
                        continue
                    expanded_owners.add(owner)
                    for call in by_owner[owner]:
                        for witness in linker.link(call).candidate_witnesses:
                            if witness.definition in functions:
                                enqueue(witness.definition, BodyDemandKind.CANDIDATE_CALL, via_call=call.operation)
                                frontier.append(witness.definition)

            roots = set(runtime_builder.definition_index) if runtime_builder is not None else set()
            roots.update(call.execution_owner for file in files if file.snapshot.path == discovery.entry
                         for call in file.calls if call.phase is ExecutionPhase.MODULE_INITIALIZATION)
            expand_owners(sorted(roots, key=lambda key: key.wire))
            for definition in sorted(functions, key=lambda key: key.wire):
                enqueue(definition, BodyDemandKind.REMAINDER)

        remaining_bodies, remaining_operations = budget.max_lowered_bodies, budget.max_operations
        loaded_path, loaded_context = None, None
        while True:
            v6_task = None
            if is_v6:
                v6_task = demand_queue.pop()
                if v6_task is None:
                    break
                request = v6_task
            else:
                if not pending:
                    break
                *_, request = heapq.heappop(pending)
                definition = request.definition
                if definition in finished or requests[definition] is not request:
                    continue
                finished.add(definition)
            definition = request.definition
            path = file_by_definition[definition]
            if path in unavailable:
                body = BodyInventory(definition, BodyStatus.NOT_ATTEMPTED, None, None, unavailable[path])
                bodies[definition] = body
                if not is_v6:
                    entries.append(request)
                continue
            if remaining_bodies < 1 or remaining_operations < 1:
                bodies[definition] = BodyInventory(definition, BodyStatus.NOT_ATTEMPTED, None, None, _BUDGET_REASON)
                if not is_v6:
                    entries.append(request)
                continue
            remaining_bodies -= 1
            try:
                if path != loaded_path:
                    # Release the previous heavy context before unpickling one.
                    loaded_context, loaded_path = None, None
                    loaded_context = store.get(path)
                    loaded_path = path
            except (OSError, ValueError, TypeError, KeyError, EOFError, RecursionError, pickle.PickleError) as error:
                unavailable[path] = _error_reason(error)
                bodies[definition] = BodyInventory(definition, BodyStatus.UNSUPPORTED, None, None, unavailable[path])
                if is_v6:
                    demand_queue.mark_result(v6_task, status=BodyStatus.UNSUPPORTED,
                        construction_operations=0, operation_budget_charged=0,
                        construction_cost_known=True)
                    # The body context failed to restore, but the completed
                    # source inventory still supports direct conditional call
                    # candidates from this one owner.
                    enqueue_v6_direct_calls(definition, v6_task.priority)
                else:
                    entries.append(replace(request, attempted=True, status=BodyStatus.UNSUPPORTED))
                continue
            cost_known, charged = True, 0
            try:
                selected = functions[definition]
                options = {"profile": selected_profile} if interface_model else {}
                result = loaded_context.lower(selected.qualname, source=selected.source,
                    max_operations=remaining_operations, expand_candidates=False, **options)
                charged = result.operations_emitted
                if type(charged) is not int or not 0 <= charged <= remaining_operations:
                    raise ValueError("lowerer returned an invalid construction count")
                body = BodyInventory(definition, BodyStatus.LOWERED_WITH_GAPS if result.graph else BodyStatus.UNSUPPORTED,
                    result.graph, result.metadata, "; ".join(dict.fromkeys(g.detail for g in result.metadata.gaps)), charged)
            except (ValueError, TypeError, KeyError, SyntaxError, RecursionError) as error:
                # Construction may have happened before the lowerer raised. Its
                # exact count is unavailable, so consume the remaining allowance.
                charged, cost_known = remaining_operations, False
                body = BodyInventory(definition, BodyStatus.UNSUPPORTED, None, None,
                                     "body construction error: " + str(error))
                issues[path].append(InventoryIssue(path, body.reason, request.source))
            else:
                if interfaces is not None:
                    try:
                        if selected_profile is BodyLoweringProfile.BOUND_FRAME:
                            if body.graph is None:
                                if result.frame is not None:
                                    raise ValueError("failed body cannot retain frame correspondence")
                            elif result.frame is None:
                                raise ValueError("frame body needs explicit frame correspondence")
                            else:
                                require_valid_frame_model(result.frame, body.graph, interface=result.interface,
                                                          semantic=loaded_context.semantic)
                                if result.frame.definition != body.definition or result.interface.profile is not selected_profile:
                                    raise ValueError("frame correspondence differs from the selected body/profile")
                        if selected_profile in (BodyLoweringProfile.STRUCTURED_BODY, BodyLoweringProfile.STRUCTURED_ENV):
                            if result.frame is not None:
                                raise ValueError("structured body cannot retain a legacy frame model")
                            if body.graph is None:
                                if result.structured is not None:
                                    raise ValueError("failed body cannot retain a structured model")
                            else:
                                if result.structured is None:
                                    raise ValueError("structured body needs its explicit protocol model")
                                # PreparedSource.lower dispatches these profiles to
                                # StructuredBuilder.run, which validates the core
                                # and complete overlay before returning. The frozen
                                # BodyInventory above retains those exact graph and
                                # metadata objects; no rewriting intervenes here.
                                # Public inventory/source replay still validates
                                # externally supplied graphs independently.
                                if (result.interface is None or result.interface.profile is not selected_profile
                                        or result.interface.definition != body.definition
                                        or result.interface.region != body.graph.root):
                                    raise ValueError("structured interface differs from the selected body/profile")
                                if (result.structured.definition != body.definition
                                        or result.structured.region != result.interface.region):
                                    raise ValueError("structured model differs from its body definition")
                        interfaces.record_body(body, result.interface)
                        if selected_profile is BodyLoweringProfile.BOUND_FRAME and result.frame is not None:
                            frames.append(result.frame)
                        if selected_profile in (BodyLoweringProfile.STRUCTURED_BODY, BodyLoweringProfile.STRUCTURED_ENV) and result.structured is not None:
                            structured.append(result.structured)
                    except (ValueError, TypeError, KeyError) as error:
                        body = BodyInventory(definition, BodyStatus.UNSUPPORTED, None, None,
                            "body interface correspondence failed: " + str(error), charged)
                        issues[path].append(InventoryIssue(path, body.reason, request.source))
            remaining_operations -= charged
            bodies[definition] = body
            if is_v6:
                demand_queue.mark_result(v6_task, status=body.status,
                    construction_operations=body.construction_operations,
                    operation_budget_charged=charged, construction_cost_known=cost_known)
                # Link only this processed owner's direct source calls. Its
                # children inherit the parent's frontier and wait their turn.
                enqueue_v6_direct_calls(definition, v6_task.priority)
            else:
                entries.append(replace(request, attempted=True, construction_operations=body.construction_operations,
                                       status=body.status, operation_budget_charged=charged,
                                       construction_cost_known=cost_known))

            _, new_candidates, _ = _body_connections((body,))
            for operation, targets in new_candidates.items():
                if operation not in calls:
                    continue
                for target in targets:
                    if target not in definitions:
                        continue
                    witness = CallCandidateWitness(target, definitions[target].source, (),
                        (CandidateCondition.SOURCE_BINDING_REACHES_READ, CandidateCondition.SOURCE_BODY_MATCHES_LIVE_CALLABLE,
                         CandidateCondition.NO_EXTERNAL_NAMESPACE_MUTATION))
                    if not linker.admit(calls[operation], witness):
                        continue
                    if is_v6:
                        demand_queue.add(target, v6_task.priority, SourceCallCandidate(operation))
                    else:
                        enqueue(target, BodyDemandKind.CANDIDATE_CALL, via_call=operation)
                        expand_owners((target,))
            if not is_v6:
                expand_owners((definition,))

        loaded_context = None
        final = []
        files = linker.finish()
        frozen_witnesses = {call.operation:call.candidate_witnesses for file in files for call in file.calls}
        for file in files:
            path = file.snapshot.path
            file_bodies = tuple(bodies[d.id] if d.id in bodies else
                BodyInventory(d.id, BodyStatus.NOT_ATTEMPTED, None, None, _SCOPE_REASON) for d in file.definitions)
            file_issues = [*file.gaps, *issues[path]]
            if path in unavailable:
                file_issues.append(InventoryIssue(path, unavailable[path]))
            result = _attach_bodies(replace(file, gaps=tuple(file_issues)), file_bodies,
                                    call_witnesses=frozen_witnesses, body_profile=selected_profile)
            final.append(result)
            if runtime_builder is not None:
                runtime_builder.file_emitted(result)
        unavailable_records = tuple(ContextUnavailable(path, reason)
                                     for path, reason in sorted(unavailable.items()))
        schedule = (demand_queue.schedule(context_budget, entry_roots, unavailable_records) if is_v6
                    else ExpansionSchedule(context_budget, tuple(entries), unavailable_records))
        measurements = ExpansionMeasurements(store.restore_attempts, store.compressed_payload_bytes_restored,
                                             store.compressed_bytes_written, store.uncompressed_bytes_written)
        return (tuple(final), schedule, measurements, interfaces.finish(final) if interfaces is not None else None,
                tuple(frames), tuple(structured))
