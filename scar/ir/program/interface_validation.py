"""Interface validation."""

from __future__ import annotations

from dataclasses import dataclass
from scar.ir.program.bindings import BindingState
from scar.ir.program.ids import ValueID
from scar.ir.program.model import (
    AccessMode,
    Behavior,
    BlockArgument,
    BindingDefinition,
    BindingRead,
    CompletionResult,
    CompletionSuccessor,
    LocalBindingValue,
    OperationResult,
    ProgramGraph,
    ExitKind,
    ReferenceValue,
    StateVersionValue,
    TerminatorKind,
    UseRole,
    ValueType,
    ValueKind,
    FootprintKind,
    DataValue,
    LiteralOrigin,
    value_type,
)
from scar.ir.program.resumption import (
    ResumeInput,
    SavedInput,
    SavedRole,
    SuspendTerminator,
    saved_actual_use,
)
from scar.ir.program.frames import FrameBind, FrameDiscard, FrameEnter, FrameExit, FrameModel
from scar.ir.program.functions import BodyEntryInterface
from scar.ir.program.validation import ValidationIssue, validate
from scar.ir.v2._validation import record_errors
from scar.ir.v2.semantic import (
    OperationDefinition,
    OperationKind,
    SemanticGraph,
    SourceAtom,
    ValueSlot,
)
from scar.ir.program.attribute_lookups import AttributeLookupForm, AttributeLookupInterface
from scar.ir.program.literal_sources import ExactLiteralOrigin
from scar.ir.literals import LiteralKind


# Cross-record checks for local-binding SSA and local reads.
def _saved_binding_uses(operation):
    term = operation.terminator
    return {slot.use for slot in term.saved if slot.role is SavedRole.LOCAL_BINDING} if isinstance(
        term, SuspendTerminator) else set()


@dataclass(frozen=True, slots=True)
class BindingValidationIssue:
    code: str
    path: str
    message: str


def _compute_binding_facts(token_values, definitions, incoming_binding_values):
    """Shared fixed-point implementation for validation and read queries."""
    facts: dict[ValueID, frozenset[BindingState]] = {}
    for identity, definition in definitions.items():
        if definition.state is BindingState.UNKNOWN:
            facts[identity] = frozenset((BindingState.BOUND, BindingState.UNBOUND))
        else:
            facts[identity] = frozenset((definition.state,))
    changed = True
    rounds = 0
    while changed and rounds <= len(token_values) + 1:
        rounds += 1
        changed = False
        for identity, incoming in incoming_binding_values.items():
            merged = frozenset(state for source in incoming for state in facts.get(source, ()))
            updated = facts.get(identity, frozenset()) | merged
            if updated != facts.get(identity, frozenset()):
                facts[identity] = updated
                changed = True
    return facts


def derive_binding_facts(graph: ProgramGraph) -> dict[ValueID, frozenset[BindingState]]:
    """Return possible BOUND/UNBOUND facts for each local-binding token.

    Call this on a graph that has passed ``program.validation.require_valid``.
    For malformed graphs it returns only facts recoverable from well-shaped
    definitions and same-slot CFG carries; it does not validate or authorize
    graph transformations. An INVOKE NORMAL successor for a BindingRead is
    feasible only when that read's input token includes BOUND.
    """
    if type(graph) is not ProgramGraph:
        raise TypeError("derive_binding_facts requires ProgramGraph")
    values = {value.id: value for value in graph.values}
    operations = {operation.id: operation for operation in graph.operations}
    blocks = {block.id: block for region in graph.regions for block in region.blocks}
    tokens = {identity: value for identity, value in values.items()
              if isinstance(value, LocalBindingValue)}
    definitions = {item.value: item for item in graph.binding_definitions}
    uses = {use.id: (operation.id, use) for operation in graph.operations
            for use in operation.uses}
    incoming: dict[ValueID, list[ValueID]] = {
        identity: [] for identity, value in tokens.items()
        if isinstance(value.origin, BlockArgument)
    }
    for region in graph.regions:
        for block in region.blocks:
            if not block.operations:
                continue
            terminator = operations.get(block.operations[-1])
            if terminator is None or terminator.terminator is None:
                continue
            saved_bindings = _saved_binding_uses(terminator)
            for edge in terminator.terminator.successors:
                target = blocks.get(edge.block)
                if target is None:
                    continue
                for actual, formal in zip(edge.arguments, target.arguments):
                    formal_value = values.get(formal)
                    actual_use = saved_actual_use(terminator, actual)
                    if (not isinstance(formal_value, LocalBindingValue) or actual_use is None
                            or isinstance(actual, SavedInput) and actual_use not in saved_bindings):
                        continue
                    pair = uses.get(actual_use)
                    source = values.get(pair[1].value) if pair is not None else None
                    if (pair is not None and pair[0] == terminator.id
                            and pair[1].role is UseRole.LOCAL_BINDING
                            and isinstance(source, LocalBindingValue)
                            and source.slot == formal_value.slot):
                        incoming[formal].append(source.id)
    return _compute_binding_facts(tokens, definitions, incoming)


def _normal_completion_reference_for_read(graph, reference, operation, blocks,
                                          operations, operation_block) -> bool:
    if isinstance(reference.origin, OperationResult):
        producer = operations.get(reference.origin.operation)
        return (producer is not None and reference.origin.index < len(producer.results)
                and producer.results[reference.origin.index] == reference.id)
    if not isinstance(reference.origin, BlockArgument):
        return False
    block_id = operation_block.get(operation.id)
    if reference.origin.block != block_id:
        return False
    block = blocks.get(block_id)
    if block is None or reference.origin.index >= len(block.arguments):
        return False
    if block.arguments[reference.origin.index] != reference.id:
        return False

    incoming = []
    for region in graph.regions:
        for source_block in region.blocks:
            if not source_block.operations:
                continue
            source = operations.get(source_block.operations[-1])
            if source is None or source.terminator is None:
                continue
            for edge in source.terminator.successors:
                if edge.block == block_id:
                    incoming.append((source, edge))
    if len(incoming) != 1:
        return False
    source, edge = incoming[0]
    if (source.terminator.kind is not TerminatorKind.INVOKE
            or not isinstance(edge, CompletionSuccessor)
            or edge.kind is not ExitKind.NORMAL):
        return False
    actual = edge.arguments[reference.origin.index]
    if not isinstance(actual, CompletionResult):
        return False
    signature = next((item for item in source.completions
                      if item.kind is ExitKind.NORMAL), None)
    return (signature is not None and actual.index < len(signature.types)
            and signature.types[actual.index] == ValueType(ValueKind.REFERENCE))


def _paired_read_failure_packet(graph, operation, blocks, operations, values,
                                operation_block, block_region):
    """Recognize this read's pair, current world and exact packet corridor.

    Two reference types alone are insufficient. The actual RAISE completion
    must enter its own block, and every intervening ordinary state transition
    and single-successor CFG edge must preserve the ordered pair and current
    world until the existing C0 ExceptionPacketDefinition consumes both Uses.
    Core SSA validation separately checks ownership/dominance and C0 checks
    the packet constructor. No new source/exception authority is introduced.
    """
    failure = next((e for e in operation.terminator.successors
                    if isinstance(e, CompletionSuccessor) and e.kind is ExitKind.RAISE), None)
    signature = next((s for s in operation.completions if s.kind is ExitKind.RAISE), None)
    reference = ValueType(ValueKind.REFERENCE)
    if (failure is None or signature is None or len(signature.types) != 3
            or signature.types[:2] != (reference, reference)
            or signature.types[2].kind is not ValueKind.STATE
            or failure.arguments != tuple(CompletionResult(i) for i in range(3))):
        return False
    first = blocks.get(failure.block)
    if first is None or len(first.arguments) != 3:
        return False
    incoming = {b: [] for b in blocks}
    for source in blocks.values():
        if not source.operations:
            continue
        last = operations.get(source.operations[-1])
        for edge in getattr(getattr(last, 'terminator', None), 'successors', ()):
            if edge.block in incoming:
                incoming[edge.block].append((last.id, edge))
    if incoming[first.id] != [(operation.id, failure)]:
        return False
    region = block_region.get(operation_block.get(operation.id))
    exception, traceback, world = first.arguments
    if (not all(isinstance(values.get(v), ReferenceValue) for v in (exception, traceback))
            or not isinstance(values.get(world), StateVersionValue)):
        return False
    packets = {values[p.value].origin.operation:p for p in graph.exception_packets
               if p.value in values and isinstance(values[p.value].origin, OperationResult)}
    block, seen = first, set()
    while block.id not in seen:
        seen.add(block.id)
        if block_region.get(block.id) != region:
            return False
        for identity in block.operations:
            current = operations[identity]
            own = {u.id:u for u in current.uses}
            packet = packets.get(identity)
            if packet is not None:
                return (packet.exception in own and packet.traceback in own
                    and own[packet.exception].value == exception
                    and own[packet.traceback].value == traceback
                    and values[packet.value].region == region
                    and isinstance(values.get(world), StateVersionValue)
                    and values[world].location == signature.types[2].location)
            if current.terminator is not None:
                if current.terminator.kind is not TerminatorKind.JUMP or len(current.terminator.successors) != 1:
                    return False
                edge = current.terminator.successors[0]
                target = blocks.get(edge.block)
                if target is None or len(edge.arguments) != len(target.arguments):
                    return False
                mapping = [(own[a].value, f) for a, f in zip(edge.arguments, target.arguments) if a in own]
                transferred = []
                for actual in (exception, traceback, world):
                    formals = [f for a, f in mapping if a == actual]
                    if len(formals) != 1:
                        return False
                    transferred.append(formals[0])
                exception, traceback, world = transferred
                block = target
                break
            # Before packing, the pair is only forwarded, never read, mixed,
            # replaced or interpreted by an unrelated operation.
            if any(u.value in (exception, traceback) for u in current.uses):
                return False
            states = [u for u in current.uses if isinstance(values.get(u.value), StateVersionValue)]
            if states:
                if len(states) != 1 or states[0].value != world:
                    return False
                after = [a.after for a in current.accesses if a.before == states[0].id and a.after is not None]
                if len(after) != 1 or after[0] not in current.results:
                    return False
                world = after[0]
        else:
            return False
    return False


def validate_bindings(graph: ProgramGraph) -> tuple[BindingValidationIssue, ...]:
    """Validate binding definitions, CFG carries, and read completion shape.

    This validates typed connections and derived boundness only. It does not
    prove that an operation implements Python's local-load or assignment
    semantics; the source-profile replay must establish that correspondence.
    """
    if (not graph.binding_definitions and not graph.binding_reads
            and not any(isinstance(value, LocalBindingValue) for value in graph.values)):
        return ()

    issues: list[BindingValidationIssue] = []

    def error(code: str, path: object, message: str) -> None:
        issues.append(BindingValidationIssue(code, str(path), message))

    values = {value.id: value for value in graph.values}
    operations = {operation.id: operation for operation in graph.operations}
    regions = {region.id: region for region in graph.regions}
    blocks = {block.id: block for region in graph.regions for block in region.blocks}
    block_region = {block.id: region.id for region in graph.regions for block in region.blocks}
    operation_block = {operation_id: block.id for block in blocks.values()
                       for operation_id in block.operations}
    uses = {use.id: (operation.id, use) for operation in graph.operations
            for use in operation.uses}

    definitions: dict[ValueID, BindingDefinition] = {}
    for definition in graph.binding_definitions:
        if definition.value in definitions:
            error("binding_definition_duplicate", definition.value,
                  "each operation-result local binding needs one definition")
            continue
        definitions[definition.value] = definition
        value = values.get(definition.value)
        if not isinstance(value, LocalBindingValue):
            error("binding_definition_value", definition.value,
                  "definition must name a LocalBindingValue")
            continue
        if not isinstance(value.origin, OperationResult):
            error("binding_definition_origin", definition.value,
                  "only operation-result binding tokens have definition records")
            continue
        producer = operations.get(value.origin.operation)
        if producer is None:
            error("binding_definition_producer", definition.value,
                  "binding producer operation is missing")
            continue
        if definition.payload is not None:
            pair = uses.get(definition.payload)
            if pair is None or pair[0] != producer.id:
                error("binding_payload_owner", definition.value,
                      "payload use must belong to the binding producer")
            elif pair[1].role is not UseRole.REFERENCE or not isinstance(values.get(pair[1].value), ReferenceValue):
                error("binding_payload_type", definition.value,
                      "bound payload must be an ordinary ReferenceValue use")
        if definition.previous is not None:
            pair = uses.get(definition.previous)
            previous = values.get(pair[1].value) if pair is not None and pair[0] == producer.id else None
            if (pair is None or pair[0] != producer.id
                    or pair[1].role is not UseRole.LOCAL_BINDING
                    or not isinstance(previous, LocalBindingValue)
                    or previous.slot != value.slot):
                error("binding_previous", definition.value,
                      "previous binding must be a same-slot local-binding use of the producer")
        if definition.state is BindingState.UNKNOWN:
            coarse_read_write = any(
                access.mode is AccessMode.READ_WRITE and not access.full_overwrite
                for access in producer.accesses)
            if producer.effects_complete or not coarse_read_write:
                error("binding_unknown_effect", definition.value,
                      "UNKNOWN updates require an incomplete producer with coarse READ_WRITE state")

    token_values = {identity: value for identity, value in values.items()
                    if isinstance(value, LocalBindingValue)}
    for identity, value in token_values.items():
        if isinstance(value.origin, OperationResult):
            if identity not in definitions:
                error("binding_definition_missing", identity,
                      "every operation-result local-binding token needs a definition")
        elif isinstance(value.origin, BlockArgument):
            if identity in definitions:
                error("binding_argument_definition", identity,
                      "block-argument binding state is derived from incoming CFG edges")
            region = regions.get(block_region.get(value.origin.block))
            if region is not None and region.entry == value.origin.block:
                error("binding_entry_argument", identity,
                      "local bindings must be initialized by definitions, not function entry ports")
        else:
            error("binding_origin", identity,
                  "local-binding origin must be a block argument or operation result")
    for identity in definitions:
        if identity not in token_values:
            error("binding_definition_orphan", identity,
                  "definition does not correspond to a local-binding value")

    # Collect each local-binding block argument's actual incoming values. The
    # regular graph validator separately checks edge arity, ownership and type.
    incoming_binding_values: dict[ValueID, list[ValueID]] = {
        identity: [] for identity, value in token_values.items()
        if isinstance(value.origin, BlockArgument)
    }
    for region in graph.regions:
        for block in region.blocks:
            if not block.operations:
                continue
            terminator = operations.get(block.operations[-1])
            if terminator is None or terminator.terminator is None:
                continue
            saved_bindings = _saved_binding_uses(terminator)
            for edge in terminator.terminator.successors:
                target = blocks.get(edge.block)
                if target is None:
                    continue
                for actual, formal in zip(edge.arguments, target.arguments):
                    formal_value = values.get(formal)
                    if not isinstance(formal_value, LocalBindingValue):
                        continue
                    if isinstance(actual, CompletionResult):
                        error("binding_completion_result", formal,
                              "local-binding tokens cannot be fabricated from completion results")
                    elif isinstance(actual, ResumeInput):
                        error("binding_resume_result", formal,
                              "local-binding tokens cannot be fabricated from resume inputs")
                    else:
                        actual_use = saved_actual_use(terminator, actual)
                        if actual_use is None:
                            continue
                        pair = uses.get(actual_use)
                        actual_value = values.get(pair[1].value) if pair is not None else None
                        if (pair is None or pair[0] != terminator.id
                                or pair[1].role is not UseRole.LOCAL_BINDING
                                or not isinstance(actual_value, LocalBindingValue)
                                or actual_value.slot != formal_value.slot
                                or isinstance(actual, SavedInput) and actual_use not in saved_bindings):
                            error("binding_edge_slot", formal,
                                  "CFG carries must forward the same local slot with LOCAL_BINDING role")
                        elif formal in incoming_binding_values:
                            incoming_binding_values[formal].append(actual_value.id)

    for identity, incoming in incoming_binding_values.items():
        if not incoming:
            error("binding_merge_inputs", identity,
                  "a local-binding block argument needs at least one incoming binding")

    reads_by_operation: dict[object, BindingRead] = {}
    for read in graph.binding_reads:
        if read.operation in reads_by_operation:
            error("binding_read_duplicate", read.operation,
                  "each local-read operation needs one canonical BindingRead")
            continue
        reads_by_operation[read.operation] = read
        operation = operations.get(read.operation)
        pair = uses.get(read.binding)
        token = values.get(pair[1].value) if pair is not None else None
        if operation is None:
            error("binding_read_operation", read.operation,
                  "read operation is missing")
            continue
        if (pair is None or pair[0] != operation.id
                or pair[1].role is not UseRole.LOCAL_BINDING
                or not isinstance(token, LocalBindingValue)):
            error("binding_read_input", read.operation,
                  "read must consume one local-binding token with LOCAL_BINDING role")
            continue
        local_binding_uses = {use.id for use in operation.uses
                              if use.role is UseRole.LOCAL_BINDING}
        if local_binding_uses != {read.binding}:
            error("binding_read_inputs", read.operation,
                  "a local-read operation must consume exactly its declared binding token")
        if operation.callee is not None or operation.call_arguments:
            error("binding_read_call", read.operation,
                  "a local-read operation cannot be an ordinary call")
        if operation.terminator is not None and operation.terminator.kind is TerminatorKind.INVOKE:
            invocation_values = [values[uses[use_id][1].value]
                                 for use_id in operation.invoke_arguments
                                 if use_id in uses]
            if read.binding not in operation.invoke_arguments:
                error("binding_read_inputs", read.operation,
                      "a local-read invoke must pass its local-binding token")
            if any(not isinstance(value, (LocalBindingValue, StateVersionValue))
                   for value in invocation_values):
                error("binding_read_call", read.operation,
                      "a local-read invoke may carry its token and world, not Python call arguments")
        if read.normal_result is not None:
            result = values.get(read.normal_result)
            if not isinstance(result, ReferenceValue):
                error("binding_read_result", read.operation,
                      "normal local-read result must be a ReferenceValue")
            elif (isinstance(result.origin, OperationResult)
                  and result.origin.operation == operation.id):
                if operation.terminator is not None:
                    error("binding_read_result", read.operation,
                          "an operation-result local read cannot also terminate control")
                if read.normal_result not in operation.results:
                    error("binding_read_result", read.operation,
                          "ordinary local-read result must be produced by its operation")
                if (result.targets is not None or result.referent_identity is not None
                        or result.selector.kind is not FootprintKind.UNKNOWN):
                    error("binding_read_precision", read.operation,
                          "new local-read results cannot assert unverified reference precision")
            elif operation.terminator is not None and operation.terminator.kind is TerminatorKind.INVOKE:
                if read.binding not in operation.invoke_arguments:
                    error("binding_read_inputs", read.operation,
                          "invoke local read must pass its binding token as an invocation input")
                normal_edge = next((edge for edge in operation.terminator.successors
                                    if isinstance(edge, CompletionSuccessor)
                                    and edge.kind.value == "NORMAL"), None)
                target = blocks.get(normal_edge.block) if normal_edge is not None else None
                if target is None or read.normal_result not in target.arguments:
                    error("binding_read_result", read.operation,
                          "invoke read result must be a normal-completion block argument")
                else:
                    target_index = target.arguments.index(read.normal_result)
                    actual = normal_edge.arguments[target_index]
                    if not isinstance(actual, CompletionResult):
                        error("binding_read_result", read.operation,
                              "normal read result must come from its completion result")
                    else:
                        signature = next((item for item in operation.completions
                                          if item.kind.value == "NORMAL"), None)
                        if (signature is None or actual.index >= len(signature.types)
                                or signature.types[actual.index] != ValueType(ValueKind.REFERENCE)):
                            error("binding_read_result_type", read.operation,
                                  "normal completion must produce the local-read reference")
                    if (result.targets is not None or result.referent_identity is not None
                            or result.selector.kind is not FootprintKind.UNKNOWN):
                        error("binding_read_precision", read.operation,
                              "completion local-read results cannot assert unverified reference precision")
            else:
                # Reusing a value is allowed only for an unambiguous direct
                # BOUND definition; merged payloads require a result produced
                # on the read's normal path.
                definition = definitions.get(token.id)
                payload_pair = uses.get(definition.payload) if definition is not None and definition.payload else None
                if (definition is None or definition.state is not BindingState.BOUND
                        or payload_pair is None or payload_pair[1].value != read.normal_result):
                    error("binding_read_result", read.operation,
                          "reused read result must be the unique direct bound payload")
                if operation.terminator is not None:
                    error("binding_read_operation", read.operation,
                          "local reads must be ordinary operations or INVOKE terminators")
        elif operation.terminator is None or operation.terminator.kind is not TerminatorKind.RAISE:
            error("binding_read_raise", read.operation,
                  "a read without a normal result must be a RAISE terminator")
        else:
            if operation.terminator.kind is TerminatorKind.INVOKE:
                error("binding_read_raise", read.operation,
                      "an INVOKE read has a normal completion result")
            exception_values = [
                values[uses[use_id][1].value]
                for use_id in operation.terminator.values if use_id in uses
                and isinstance(values.get(uses[use_id][1].value), ReferenceValue)
            ]
            if (len(exception_values) != 1 or not _normal_completion_reference_for_read(
                    graph, exception_values[0], operation, blocks, operations, operation_block)):
                error("binding_read_exception", read.operation,
                      "unbound local must raise one exception reference from a typed producer or NORMAL completion")
            state_values = [
                values[uses[use_id][1].value]
                for use_id in operation.terminator.values if use_id in uses
                and isinstance(values.get(uses[use_id][1].value), StateVersionValue)
            ]
            if not state_values:
                error("binding_read_world", read.operation,
                      "unbound-local exception must preserve an explicit world state")

    # A monotone two-bit fixed point derives possible bound/unbound states at
    # merge blocks, including loop headers and backedges. UNKNOWN means both.
    facts = _compute_binding_facts(token_values, definitions, incoming_binding_values)

    for identity, value in token_values.items():
        if not facts.get(identity):
            error("binding_state_unresolved", identity,
                  "binding state cannot be derived from a definition or incoming edge")

    # Validate read routes after the fixed point so a merge cannot claim a
    # normal path while every incoming local is unbound.
    for read in graph.binding_reads:
        operation = operations.get(read.operation)
        pair = uses.get(read.binding)
        token = values.get(pair[1].value) if pair is not None else None
        if operation is None or not isinstance(token, LocalBindingValue):
            continue
        states = facts.get(token.id, frozenset())
        can_be_bound = BindingState.BOUND in states
        can_be_unbound = BindingState.UNBOUND in states
        if not states:
            continue
        if read.normal_result is None:
            if can_be_bound:
                error("binding_read_missing_normal", read.operation,
                      "a possibly bound local read cannot discard its normal result")
            term = operation.terminator
            if term is None or term.kind is not TerminatorKind.RAISE:
                error("binding_read_unbound_exit", read.operation,
                      "definitely unbound read must terminate on its RAISE path")
        else:
            if can_be_unbound and (operation.terminator is None
                                   or operation.terminator.kind is not TerminatorKind.INVOKE):
                error("binding_read_missing_raise", read.operation,
                      "a local read that may be unbound requires NORMAL and RAISE completion paths")
            is_invoke = (operation.terminator is not None
                         and operation.terminator.kind is TerminatorKind.INVOKE)
            if is_invoke:
                if operation.effects_complete:
                    error("binding_read_effects", read.operation,
                          "an invoke local read must preserve its unproved frame/exception effects")
                signatures = {signature.kind: signature for signature in operation.completions}
                region_id = block_region.get(operation_block.get(operation.id))
                region = regions.get(region_id)
                entry = blocks.get(region.entry) if region is not None else None
                state_locations = set()
                if entry is not None:
                    state_locations = {
                        values[value_id].location for value_id in entry.arguments
                        if isinstance(values.get(value_id), StateVersionValue)
                    }
                if not state_locations:
                    error("binding_read_world", read.operation,
                          "invoke local reads require an explicit world state domain")
                for kind in (ExitKind.NORMAL, ExitKind.RAISE):
                    signature = signatures.get(kind)
                    if signature is None:
                        error("binding_read_completion", read.operation,
                              "invoke local read needs both completion signatures")
                        continue
                    refs = [item for item in signature.types
                            if item == ValueType(ValueKind.REFERENCE)]
                    states_out = [item.location for item in signature.types
                                  if item.kind is ValueKind.STATE]
                    paired = (kind is ExitKind.RAISE and len(refs) == 2
                        and _paired_read_failure_packet(graph, operation, blocks, operations,
                            values, operation_block, block_region))
                    reference_count = 2 if paired else 1
                    if (len(refs) != reference_count or set(states_out) != state_locations
                            or len(states_out) != len(state_locations)
                            or len(signature.types) != reference_count + len(state_locations)):
                        error("binding_read_completion", read.operation,
                              "read completion needs one reference/current world or its exact packet-bound RAISE pair")

    # A frame protocol may consume a binding token as a lifetime dependency.
    # The exact transition meaning is checked by StructuredBodyModel/source
    # replay; the core only admits its deliberately incomplete, coarse-world
    # operation shape here.
    def is_coarse_frame_consumer(operation) -> bool:
        if (operation.terminator is not None or operation.callee is not None
                or operation.call_arguments or operation.invoke_arguments
                or operation.completions or operation.effects_complete
                or set(operation.behaviors) != {Behavior.READ, Behavior.WRITE, Behavior.MEMORY}
                or len(operation.accesses) != 1):
            return False
        access = operation.accesses[0]
        if (access.mode is not AccessMode.READ_WRITE or access.full_overwrite
                or access.footprint.kind is FootprintKind.INTERVAL
                or access.reference is not None):
            return False
        if operation.results != (access.after,):
            return False
        state_inputs = [use for use in operation.uses
                        if isinstance(values.get(use.value), StateVersionValue)]
        if (len(state_inputs) != 1 or state_inputs[0].id != access.before
                or state_inputs[0].role is not UseRole.STATE_CONTENT):
            return False
        return all(use.role in (UseRole.LOCAL_BINDING, UseRole.REFERENCE,
                                UseRole.STATE_CONTENT)
                   for use in operation.uses)

    # Every local-binding input must have one typed consumer: definition update,
    # binding read, same-slot CFG successor edge, or the coarse frame-lifetime
    # operation shape above.
    authorized: set[object] = set()
    for definition in graph.binding_definitions:
        if definition.previous is not None:
            authorized.add(definition.previous)
    for read in graph.binding_reads:
        authorized.add(read.binding)
    for region in graph.regions:
        for block in region.blocks:
            if not block.operations:
                continue
            operation = operations.get(block.operations[-1])
            if operation is None or operation.terminator is None:
                continue
            saved_bindings = _saved_binding_uses(operation)
            authorized.update(saved_bindings)
            for edge in operation.terminator.successors:
                for actual in edge.arguments:
                    actual_use = saved_actual_use(operation, actual)
                    if (actual_use is not None and (not isinstance(actual, SavedInput)
                                                   or actual_use in saved_bindings)):
                        pair = uses.get(actual_use)
                        if pair is not None and pair[1].role is UseRole.LOCAL_BINDING:
                            authorized.add(actual_use)
    for operation in graph.operations:
        for use in operation.uses:
            if (use.role is UseRole.LOCAL_BINDING and use.id not in authorized
                    and not is_coarse_frame_consumer(operation)):
                error("binding_use_unaccounted", use.id,
                      "local-binding uses must belong to a definition, read, CFG edge, or coarse frame-lifetime operation")

    # Calls expose Python values and function state, never local-binding tokens.
    reads_for_call_check = {read.operation: read for read in graph.binding_reads}
    for operation in graph.operations:
        if operation.callee is not None:
            callee = regions.get(operation.callee)
            entry = blocks.get(callee.entry) if callee is not None else None
            if entry is not None and any(isinstance(values.get(value), LocalBindingValue)
                                         for value in entry.arguments):
                error("binding_call_formal", operation.id,
                      "ordinary call interfaces cannot expose local-binding tokens")
        actual_uses = set(operation.call_arguments) | set(operation.invoke_arguments)
        local_actual_uses = {
            use_id for use_id in actual_uses if use_id in uses
            and isinstance(values.get(uses[use_id][1].value), LocalBindingValue)
        }
        read = reads_for_call_check.get(operation.id)
        permitted_local_read = ({read.binding}
                                if read is not None
                                and operation.terminator is not None
                                and operation.terminator.kind is TerminatorKind.INVOKE
                                else set())
        if local_actual_uses - permitted_local_read:
            error("binding_call_actual", operation.id,
                  "ordinary Python call arguments cannot carry local-binding tokens")

    return tuple(issues)


# Validate declared function environment ports in the canonical graph.
#
# This checks interface well-formedness only. ``code_definition`` is a declared
# symbol, not verified runtime code/environment identity. Member locations are
# necessary state domains declared by this interface, not a complete read/alias
# summary: omitted domains never mean NONE, shared/coarse domains remain legal,
# and unknown reference targets never prove disjointness.
#
# Actual environment arguments are exclusively the existing call/invoke operands.
# The ordinary core validator checks their positional types, availability, state
# currentness and completion flow; this module stores no parallel binding table.
# Python formal ports are object references, not unboxed Data values. Their
# declared signature order is independent of physical entry order: exact ValueIDs
# partition the entry into formals, environment references and state parameters.
# This structural declaration does not verify source signature/runtime binding.
def _entry_value(value, region, entry, allowed):
    return (isinstance(value, allowed)
            and value.region == region.id
            and isinstance(value.origin, BlockArgument)
            and value.origin.block == entry.id
            and value.origin.index < len(entry.arguments)
            and entry.arguments[value.origin.index] == value.id)


def environment_errors(graph, operation_regions, uses):
    """Yield ``(code, path, message)`` for graph-level environment errors.

    The common validator passes operation/use indexes for interface validators;
    this aggregate has no independent operation-owned actual bindings to check.
    """
    del operation_regions, uses
    regions = {region.id: region for region in graph.regions}
    values = {value.id: value for value in graph.values}
    locations = {location.id for location in graph.locations}
    environments = {environment.id: environment for environment in graph.environments}
    owners = set()

    for index, environment in enumerate(graph.environments):
        path = f"environments[{index}]"
        if environment.region in owners:
            yield ("environment_region", path,
                   "one region may declare at most one function environment interface")
        owners.add(environment.region)
        region = regions.get(environment.region)
        if region is None:
            yield ("environment_owner", path, "environment owning region does not exist")
            continue
        entry = next((block for block in region.blocks if block.id == region.entry), None)
        if entry is None:
            yield ("environment_entry", path, "environment region has no matching entry block")
            continue
        entry_domains = {
            value.location for identity in entry.arguments
            if isinstance(value := values.get(identity), StateVersionValue)
        }
        member_ports = set()
        for member_index, member in enumerate(environment.members):
            member_path = f"{path}.members[{member_index}]"
            if member.reference in member_ports:
                yield ("environment_reference", member_path,
                       "independent environment members need distinct entry reference ports")
            member_ports.add(member.reference)
            reference = values.get(member.reference)
            if not _entry_value(reference, region, entry, ReferenceValue):
                yield ("environment_reference", member_path,
                       "member reference must be this region's exact entry ReferenceValue argument")
            for location in member.locations:
                if location not in locations:
                    yield ("environment_location", member_path,
                           "declared environment state location does not exist")
                if location not in entry_domains:
                    yield ("environment_state_domain", member_path,
                           "declared environment location needs a canonical entry state parameter")
                if (isinstance(reference, ReferenceValue)
                        and reference.targets is not None
                        and location not in reference.targets):
                    yield ("environment_reference_targets", member_path,
                           "declared environment location contradicts known reference targets")

    layouts = set()
    for index, layout in enumerate(graph.function_entries):
        path = f"function_entries[{index}]"
        if layout.environment in layouts:
            yield ("function_entry_duplicate", path,
                   "one environment may have only one function entry layout")
        layouts.add(layout.environment)
        environment = environments.get(layout.environment)
        if environment is None:
            yield ("function_entry_environment", path,
                   "function entry layout needs an existing environment interface")
            continue
        region = regions.get(environment.region)
        if region is None:
            continue  # Reported by the owning environment above.
        entry = next((block for block in region.blocks if block.id == region.entry), None)
        if entry is None:
            continue
        implicit = {member.reference for member in environment.members}
        free_slots = {member.cell_slot for member in environment.members
                      if member.cell_slot is not None}
        parameters = {parameter.value for parameter in layout.parameters}
        for parameter_index, parameter in enumerate(layout.parameters):
            if parameter.slot in free_slots:
                yield ("function_entry_slot", f"{path}.parameters[{parameter_index}]",
                       "a Python formal slot cannot also be this body's captured free-variable slot")
            value = values.get(parameter.value)
            if (parameter.value in implicit
                    or not _entry_value(value, region, entry, ReferenceValue)):
                yield ("function_entry_parameter", f"{path}.parameters[{parameter_index}]",
                       "Python formal must be an exact entry ReferenceValue port, not environment or state")
        expected = {identity for identity in entry.arguments
                    if not isinstance(values.get(identity), StateVersionValue)
                    and identity not in implicit}
        if parameters != expected:
            yield ("function_entry_partition", path,
                   "Python formal ports must cover exactly the non-state, non-environment entry arguments")
    for environment in graph.environments:
        if environment.id not in layouts:
            yield ("function_entry_missing", environment.id,
                   "each function environment needs an explicit formal/environment/state entry partition")


# Cross-check the finite frame profile against its core and parsed context.
#
# Only this source profile requires all ordinary body operations to be typed
# frame transitions; general ProgramGraph validation is unchanged. Checking a
# correspondence does not prove NO_FRAME_INTERPOSITION or replay source files.
def validate_frame_model(model: FrameModel, graph: ProgramGraph, *,
                         interface: BodyEntryInterface,
                         semantic: SemanticGraph) -> tuple[ValidationIssue, ...]:
    """Check declared frame links; full source replay remains the caller's gate.

    ``interface`` and ``semantic`` must be the independently prepared context,
    not definitions copied out of this overlay to make its links agree.
    """
    issues = []

    def error(code, path, message):
        issues.append(ValidationIssue(code, str(path), message))

    for record, expected, name in ((model, FrameModel, "frame"),
                                   (interface, BodyEntryInterface, "interface")):
        for message in record_errors(record, expected, name):
            error("frame_schema", name, message)
    if type(semantic) is not SemanticGraph:
        error("frame_context", "semantic", "an independently parsed SemanticGraph is required")
    elif any(type(getattr(semantic, name)) is not dict for name in ("definitions", "source_atoms", "slots")):
        error("frame_context", "semantic", "parsed source registries must be dictionaries")
    if issues:
        return tuple(issues)
    core_issues = validate(graph)
    if core_issues:
        return tuple(ValidationIssue("core." + issue.code, issue.path, issue.message)
                     for issue in core_issues)

    operations = {op.id: op for op in graph.operations}
    values = {value.id: value for value in graph.values}
    regions = {region.id: region for region in graph.regions}
    region = regions.get(model.region)
    if model.definition != interface.definition or model.region != interface.region:
        error("frame_interface", model.region, "frame and body interface definition/region must agree")
    if region is None or model.region != graph.root:
        error("frame_body", model.region, "the frame profile describes the selected root body")
    definition = semantic.definitions.get(model.definition)
    if definition is not None:
        for message in record_errors(definition, OperationDefinition, "definition"):
            error("frame_context", model.definition, message)
        if issues:
            return tuple(issues)
    if (definition is None or definition.id != model.definition
            or definition.kind not in (OperationKind.FUNCTION, OperationKind.METHOD)
            or ("async" in definition.metadata and type(definition.metadata["async"]) is not bool)
            or definition.metadata.get("async") is True):
        error("frame_definition", model.definition, "frame needs its parsed synchronous function definition")

    atom_owners = {}
    for item in semantic.definitions.values():
        if isinstance(item, OperationDefinition):
            for atom_id in item.source_atoms:
                atom_owners.setdefault(atom_id, []).append(item.id)

    def in_body_context(owner_id):
        seen = set()
        while owner_id is not None and owner_id not in seen:
            seen.add(owner_id)
            owner = semantic.definitions.get(owner_id)
            if not isinstance(owner, OperationDefinition):
                return False
            if owner.id == model.definition:
                return True
            if owner.kind in (OperationKind.FUNCTION, OperationKind.METHOD):
                return False
            owner_id = owner.parent_id
        return False

    def source(reference):
        atom = semantic.source_atoms.get(reference.atom_id)
        if atom is not None:
            messages = record_errors(atom, SourceAtom, "source_atom")
            if messages:
                for message in messages:
                    error("frame_context", reference.atom_id, message)
                return
        if atom is None or atom.id != reference.atom_id or atom.reference != reference:
            error("frame_source", reference.atom_id, "source differs from the parsed source atom")
        if not any(in_body_context(owner) for owner in atom_owners.get(reference.atom_id, ())):
            error("frame_source_context", reference.atom_id,
                  "source must execute in this function, not its declaration or a nested function")
        owner = model.source
        if (reference.path != owner.path
                or (reference.start_line, reference.start_column) < (owner.start_line, owner.start_column)
                or reference.end_line > owner.end_line
                or (reference.end_line == owner.end_line and owner.end_column is not None
                    and (reference.end_column is None or reference.end_column > owner.end_column))):
            error("frame_source_owner", reference.atom_id, "source must lie inside this function's source span")

    source(model.source)
    if definition is not None and (
            model.source.atom_id not in definition.source_atoms
            or definition.source_file != model.source.path
            or definition.source_start != model.source.start_line
            or definition.source_end != model.source.end_line):
        error("frame_definition_source", model.definition, "frame source differs from its parsed definition")
    if issues:
        return tuple(issues)

    blocks = {block.id: block for block in region.blocks}
    entry = blocks[region.entry]
    owner_blocks = {op: block for block in region.blocks for op in block.operations}
    body_operations = {op: operations[op] for op in owner_blocks}
    formal_values = tuple(formal.value for formal in interface.parameters)
    if entry.arguments != formal_values + (interface.world,):
        error("frame_entry_ports", region.entry, "entry must carry all formals, then its sole world state")
    if any(not isinstance(values.get(value), ReferenceValue) for value in formal_values):
        error("frame_formal_type", region.entry, "every bound Python formal must be a reference")
    world = values.get(interface.world)
    if not isinstance(world, StateVersionValue):
        error("frame_world", interface.world, "body world must be an entry state parameter")
        return tuple(issues)
    for port in region.exits:
        if (len(port.types) != 2 or port.types[0].kind is not ValueKind.REFERENCE
                or port.types[1].kind is not ValueKind.STATE or port.types[1].location != world.location):
            error("frame_exit_ports", region.id, "frame exits require one reference payload and the world state")

    def slot(slot_id, *, formal=False):
        record = semantic.slots.get(slot_id)
        if record is not None:
            messages = record_errors(record, ValueSlot, "slot")
            if messages:
                for message in messages:
                    error("frame_context", slot_id, message)
                return
        if (record is None or record.slot_id != slot_id or record.owner != model.definition
                or record.direction not in (("input",) if formal else ("input", "binding"))):
            error("frame_slot", slot_id, "slot must be a parsed ordinary local/formal of this function")

    for formal in interface.parameters:
        slot(formal.slot, formal=True)
    if set(definition.input_slots) != {formal.slot for formal in interface.parameters}:
        error("frame_formal_slots", model.definition, "interface must cover every parsed formal slot")
    if issues:
        return tuple(issues)

    ordinary = {op.id for op in body_operations.values() if op.terminator is None}
    declared = {transition.operation for transition in model.transitions}
    if ordinary != declared:
        error("frame_operation_coverage", model.region,
              "this finite frame profile needs exactly one transition for each ordinary body operation")
    completions = {op.id for op in body_operations.values() if op.terminator is not None
                   and op.terminator.kind in (TerminatorKind.RETURN, TerminatorKind.RAISE)}
    if completions != {item.completion for item in model.transitions if type(item) is FrameExit}:
        error("frame_exit_coverage", model.region, "every actual RETURN/RAISE needs exactly one frame EXIT")

    for transition in model.transitions:
        source(transition.source)
        if (type(transition) in (FrameBind, FrameDiscard)
                and transition.source.atom_id == model.source.atom_id):
            error("frame_source_context", transition.source.atom_id,
                  "BIND/DISCARD must use a body operation source, not the function declaration")
        op = body_operations.get(transition.operation)
        if op is None:
            error("frame_operation_owner", transition.operation, "transition operation must belong to this body")
            continue
        if (op.terminator is not None or op.callee is not None or op.call_arguments
                or op.completions or op.invoke_arguments):
            error("frame_operation_shape", op.id, "frame transitions are ordinary state operations without calls or completion edges")
        if set(op.behaviors) != {Behavior.READ, Behavior.WRITE, Behavior.MEMORY}:
            error("frame_behavior", op.id, "frame transitions require READ/WRITE/MEMORY behaviors")
        if op.effects_complete:
            error("frame_effects", op.id, "unknown finalizer/external effects cannot be certified complete")
        if len(op.accesses) != 1:
            error("frame_access", op.id, "frame transition needs one coarse world READ_WRITE access")
            continue
        access = op.accesses[0]
        if (access.location != world.location or access.mode is not AccessMode.READ_WRITE
                or access.full_overwrite or access.reference is not None
                or access.footprint.kind is FootprintKind.INTERVAL):
            error("frame_access", op.id, "frame access must preserve coarse world content dependence without a reference selector")
        if op.results != (access.after,):
            error("frame_results", op.id, "frame transition must produce only its new world state")
        uses = {use.id: use for use in op.uses}
        state_use = uses.get(access.before)
        if state_use is None or state_use.role is not UseRole.STATE_CONTENT:
            error("frame_state_use", op.id, "frame before-state must be a STATE_CONTENT use")

        if type(transition) is FrameEnter:
            references = tuple(formal.use for formal in transition.formals)
            if owner_blocks[op.id].id != region.entry or entry.operations[0] != op.id:
                error("frame_enter_position", op.id, "ENTER must be the first operation in the body entry")
            expected = {formal.slot: formal.value for formal in interface.parameters}
            if set(expected) != {formal.slot for formal in transition.formals}:
                error("frame_enter_formals", op.id, "ENTER must hold every formal slot exactly once")
            for formal in transition.formals:
                slot(formal.slot, formal=True)
                use = uses.get(formal.use)
                if use is None or use.value != expected.get(formal.slot):
                    error("frame_formal_use", formal.use, "formal slot must use its own bound entry reference")
            if state_use is not None and state_use.value != interface.world:
                error("frame_enter_world", op.id, "ENTER consumes the interface's initial world")
        elif type(transition) is FrameBind:
            references = (transition.assigned,)
            slot(transition.slot)
        elif type(transition) is FrameDiscard:
            references = (transition.reference,)
        else:
            references = (transition.payload,)
            completion = body_operations.get(transition.completion)
            if (completion is None or completion.terminator is None
                    or completion.terminator.kind not in (TerminatorKind.RETURN, TerminatorKind.RAISE)):
                error("frame_completion", transition.completion, "EXIT must name a RETURN/RAISE of this body")
            else:
                block = owner_blocks[op.id]
                if (owner_blocks[completion.id].id != block.id
                        or block.operations[-2:] != (op.id, completion.id)):
                    error("frame_exit_position", op.id, "EXIT must immediately precede its completion in the same block")
                completion_uses = {use.id: use for use in completion.uses}
                ports = completion.terminator.values
                if len(ports) != 2:
                    error("frame_completion_ports", completion.id, "completion needs payload and post-world")
                else:
                    payload = uses.get(transition.payload)
                    if payload is None or completion_uses[ports[0]].value != payload.value:
                        error("frame_payload", completion.id, "completion must preserve the EXIT payload reference")
                    if completion_uses[ports[1]].value != access.after:
                        error("frame_post_world", completion.id, "completion must carry the EXIT post-world")

        for reference in references:
            use = uses.get(reference)
            if (use is None or use.role is not UseRole.REFERENCE
                    or not isinstance(values[use.value], ReferenceValue)):
                error("frame_reference_use", reference, "frame reference must be an owned REFERENCE use")
        if set(uses) != set(references) | {access.before}:
            error("frame_use_coverage", op.id, "transition must account for each reference and world use exactly")
    return tuple(issues)


def require_valid_frame_model(model: FrameModel, graph: ProgramGraph, *,
                              interface: BodyEntryInterface, semantic: SemanticGraph) -> None:
    issues = validate_frame_model(model, graph, interface=interface, semantic=semantic)
    if issues:
        raise ValueError("; ".join(f"{issue.code} at {issue.path}: {issue.message}" for issue in issues))


# Structural lookup-interface checks against original core ports.
#
# No selector is promoted to an address/effect, and no receiver/result identity
# or selected runtime dispatch is inferred. Independent source/compiler replay
# must separately establish the request's static source and name-resolution rule.
def attribute_lookup_errors(graph: ProgramGraph) -> tuple[tuple[str, str, str], ...]:
    if type(graph) is not ProgramGraph:
        return (("attribute_lookup_graph", "graph", "expected exact ProgramGraph"),)
    if graph.attribute_lookups is None:
        return ()
    issues = []
    def error(code, path, message):
        issues.append((code, path, message))
    operations = {row.id: row for row in graph.operations}
    values = {row.id: row for row in graph.values}
    blocks = {block.id: block for region in graph.regions for block in region.blocks}
    seen = set()
    for ordinal, row in enumerate(graph.attribute_lookups):
        path = f"attribute_lookups[{ordinal}]"
        if type(row) is not AttributeLookupInterface:
            error("attribute_lookup_record", path, "expected exact AttributeLookupInterface")
            continue
        try:
            row.__post_init__()
        except (TypeError, ValueError) as exc:
            error("attribute_lookup_record", path, str(exc))
            continue
        if row.operation in seen:
            error("attribute_lookup_duplicate", path, "one original operation has multiple lookup interfaces")
        seen.add(row.operation)
        operation = operations.get(row.operation)
        if operation is None:
            error("attribute_lookup_operation", path, "lookup operation is absent from this graph")
            continue
        term = operation.terminator
        if (term is None or term.kind is not TerminatorKind.INVOKE or operation.callee is not None
                or operation.effects_complete):
            error("attribute_lookup_invoke", path,
                "lookup must retain its original incomplete opaque INVOKE, without a fixed callee")
            continue
        uses = {use.id: use for use in operation.uses}
        receiver, world_use = uses.get(row.receiver), uses.get(row.world)
        receiver_value = values.get(receiver.value) if receiver is not None else None
        before = values.get(world_use.value) if world_use is not None else None
        if (receiver is None or receiver.role is not UseRole.REFERENCE
                or type(receiver_value) is not ReferenceValue or row.receiver not in operation.invoke_arguments):
            error("attribute_lookup_receiver", path, "receiver is not this INVOKE's actual reference Use")
        if (world_use is None or world_use.role is not UseRole.STATE_CONTENT
                or type(before) is not StateVersionValue or row.world not in operation.invoke_arguments):
            error("attribute_lookup_world", path, "world is not this INVOKE's actual coarse-state Use")
            before = None
        expected_uses = (row.receiver, *((row.selector.key_use,)
            if row.selector.key_use is not None else ()), row.world)
        if operation.invoke_arguments != expected_uses or tuple(use.id for use in operation.uses) != expected_uses:
            error("attribute_lookup_operand_layout", path,
                "lookup must retain exactly the original ordered receiver/optional key/world Uses")
        key_id = row.selector.key_use
        if key_id is not None:
            key = uses.get(key_id)
            value = values.get(key.value) if key is not None else None
            if key is None or key.role is not UseRole.REFERENCE or type(value) is not ReferenceValue:
                error("attribute_lookup_key", path, "lookup key is not its actual original reference Use")
            elif row.selector.lookup_name is not None:
                origin = value.origin
                name = (origin.value if type(origin) is LiteralOrigin and type(origin.value) is str else
                    origin.literal.to_python() if type(origin) is ExactLiteralOrigin
                        and origin.literal.kind in (LiteralKind.STR, LiteralKind.STR_CODEPOINTS) else None)
                if name != row.selector.lookup_name:
                    error("attribute_lookup_key_name", path, "literal key differs from the declared actual lookup name")
        for name, completion, kind in (("normal", row.normal, ExitKind.NORMAL),
                                       ("raised", row.raised, ExitKind.RAISE)):
            own_path = path + "." + name
            index = completion.edge.successor_index
            if index >= len(term.successors):
                error("attribute_lookup_edge", own_path, "completion edge is absent from the original INVOKE")
                continue
            edge = term.successors[index]
            signatures = [signature for signature in operation.completions if signature.kind is kind]
            same_kind = [successor for successor in term.successors if successor.kind is kind]
            if edge.kind is not kind or len(signatures) != 1 or len(same_kind) != 1:
                error("attribute_lookup_completion", own_path, "lookup needs its unique actual NORMAL/RAISE signature and edge")
                continue
            signature = signatures[0]
            block = blocks.get(edge.block)
            if block is None or len(edge.arguments) != len(block.arguments):
                error("attribute_lookup_landing", own_path, "lookup completion lacks its original landing block/formals")
                continue
            slots = {}
            for actual, formal in zip(edge.arguments, block.arguments):
                if type(actual) is not CompletionResult:
                    error("attribute_lookup_completion_argument", own_path,
                        "lookup landing must retain original produced completion slots")
                    continue
                if actual.index >= len(signature.types) or actual.index in slots:
                    error("attribute_lookup_completion_slot", own_path, "lookup completion slot is out of range or repeated")
                    continue
                slots[actual.index] = formal
            if set(slots) != set(range(len(signature.types))):
                error("attribute_lookup_completion_slots", own_path, "lookup completion does not expose every original signature slot")
                continue
            world_slots = [i for i, typ in enumerate(signature.types) if getattr(typ, "kind", None) is ValueKind.STATE]
            if len(world_slots) != 1:
                error("attribute_lookup_post_world", own_path, "lookup completion needs one actual post-world slot")
                continue
            world_slot = world_slots[0]
            actual_world = slots[world_slot]
            payloads = tuple(slots[i] for i in range(len(signature.types)) if i != world_slot)
            if completion.payloads != payloads or completion.world != actual_world:
                error("attribute_lookup_completion_values", own_path,
                    "lookup payload/world IDs differ from original completion formal slots")
            after = values.get(actual_world)
            if type(after) is not StateVersionValue or before is not None and after.location != before.location:
                error("attribute_lookup_post_world", own_path, "lookup post-world changed its actual state carrier")
            for i, formal in slots.items():
                value = values.get(formal)
                position = block.arguments.index(formal)
                if (value is None or value.origin != BlockArgument(block.id, position)
                        or value_type(value) != signature.types[i]):
                    error("attribute_lookup_formal", own_path, "lookup landing value type/origin differs from its original slot")
            if kind is ExitKind.NORMAL:
                normal_values = tuple(values.get(formal) for formal in payloads)
                if row.form is AttributeLookupForm.CPYTHON_LOAD_METHOD:
                    valid = (len(normal_values) == 3 and type(normal_values[0]) is ReferenceValue
                        and type(normal_values[1]) is ReferenceValue and type(normal_values[2]) is DataValue
                        and normal_values[2].data_type == "builtins.bool")
                else:
                    valid = len(normal_values) == 1 and type(normal_values[0]) is ReferenceValue
                if not valid:
                    error("attribute_lookup_normal_abi", own_path,
                        "getter needs one reference; normalized method lookup needs callable/self references and builtin-bool flag")
    return tuple(issues)


__all__ = [
    'BindingValidationIssue',
    'derive_binding_facts',
    'validate_bindings',
    'environment_errors',
    'validate_frame_model',
    'require_valid_frame_model',
    'attribute_lookup_errors',
]
