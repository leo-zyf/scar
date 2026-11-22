"""Driver declarations and port queries over canonical Uses and control flow.

These checks establish a static interface contract, never an instance, runtime
admission, source correspondence or lifecycle proof.
"""
from __future__ import annotations

from collections import deque

from .control import ProtocolValueKind as P
from .drivers import (
    BodyDrive, BodyDriveResult, BodyDriveTerminator, BodyResume, BodyStart,
    DriverReceiverKind,
)
from .environments import EnvironmentRole
from .ids import UseID
from .model import (
    AccessMode, Behavior, BlockArgument, CompletionResult, DataValue, ExitKind,
    FootprintKind, OperationResult, ProtocolValue, ReferenceValue, StateVersionValue,
    TerminatorKind, UseRole, value_type,
)
from .resumption import SuspendTerminator, resume_slot_type, saved_actual_use
from scar.ir.program.resumption_flow import first_body_outcome_operations


def _driver_role_copies(interface, region, blocks, operations):
    from .cleanup_forward import _must_forward

    incoming = {formal: [] for block in region.blocks for formal in block.arguments}
    for block in region.blocks:
        operation = operations[block.operations[-1]]
        own = {use.id: use.value for use in operation.uses}
        for edge in operation.terminator.successors:
            for actual, formal in zip(edge.arguments, blocks[edge.block].arguments):
                incoming[formal].append(own.get(saved_actual_use(operation, actual)))
    return (
        _must_forward({interface.receiver}, incoming),
        _must_forward({interface.scheduler_request}, incoming),
    )


def _drive_target(target, bodies, regions, find_operation, find_owner):
    if type(target) is BodyStart:
        region = regions.get(target.body)
        if region is None or region.id not in bodies:
            raise ValueError("start target must name a declared resumable body")
        return region, region.entry, None
    if type(target) is not BodyResume:
        raise ValueError("drive target must be BodyStart or BodyResume")
    source = find_operation(target.suspend)
    if source is None or type(source.terminator) is not SuspendTerminator:
        raise ValueError("resume target must name a real Suspend operation")
    region = find_owner(source.id)
    if region is None or region.id not in bodies:
        raise ValueError("resume source must belong to a declared resumable body")
    if target.successor >= len(source.terminator.successors):
        raise ValueError("resume successor index exceeds the suspension signature")
    edge = source.terminator.successors[target.successor]
    return region, edge.block, edge


def body_drive_target(graph, drive):
    """Resolve a drive's original body and entry or resume landing."""
    if type(drive) is not BodyDrive:
        raise TypeError("body drive target needs an exact BodyDrive")
    bodies = {body.region for body in graph.resumable_bodies}
    regions = {region.id: region for region in graph.regions}

    def find_operation(identity):
        return next((operation for operation in graph.operations if operation.id == identity), None)

    def find_owner(identity):
        return next((region for region in graph.regions
                     if any(identity in block.operations for block in region.blocks)), None)

    return _drive_target(drive.target, bodies, regions, find_operation, find_owner)


def body_outcome_uses(operation):
    """Return only the source operation's public handoff/terminal UseIDs."""
    term = operation.terminator
    if type(term) is SuspendTerminator:
        return (term.exported, *term.handoff)
    if term is not None and term.kind in (TerminatorKind.RETURN, TerminatorKind.RAISE):
        return term.values
    raise ValueError("drive outcome source must be Suspend, Return or Raise")


def _outcome_types(operation, values, find_use):
    result = []
    for identity in body_outcome_uses(operation):
        use = find_use(identity)
        if use is None or use.value not in values:
            raise ValueError("outcome signature must use the source's canonical operands")
        result.append(value_type(values[use.value]))
    return tuple(result)


class DriverFacts:
    """Driver queries scoped to one graph and its registered core records."""

    def __init__(self, graph, core):
        self.graph = graph
        self.operations = core.operations
        self.values = core.values
        self.regions = core.regions
        self.blocks = core.blocks
        self.operation_blocks = core.operation_blocks
        self.block_regions = core.block_regions
        self.uses = core.uses
        self.bodies = {row.region: row for row in graph.resumable_bodies}
        self.interfaces = {row.region: row for row in graph.driver_interfaces}
        self._roles = {}
        self._outcomes = {}
        self._frontiers = {}

    def _owning_region(self, operation):
        block = self.operation_blocks.get(operation)
        return self.regions.get(self.block_regions.get(block))

    def owned_use(self, operation, identity):
        row = self.uses.get(identity)
        return row[1] if row is not None and row[0] == operation else None

    def drive_target(self, drive):
        if type(drive) is not BodyDrive:
            raise TypeError("body drive target needs an exact BodyDrive")
        return _drive_target(drive.target, self.bodies, self.regions,
                             self.operations.get, self._owning_region)

    def outcome_types(self, source):
        if source not in self._outcomes:
            operation = self.operations.get(source)
            if operation is None:
                raise ValueError("drive outcome source operation is missing")
            self._outcomes[source] = _outcome_types(
                operation, self.values, lambda identity: self.owned_use(source, identity))
        return self._outcomes[source]

    def role_copies(self, region_id):
        if region_id not in self._roles:
            roles = _driver_role_copies(self.interfaces[region_id], self.regions[region_id],
                                        self.blocks, self.operations)
            self._roles[region_id] = tuple(frozenset(role) for role in roles)
        return self._roles[region_id]

    def first_outcomes(self, region_id, start):
        key = region_id, start
        if key not in self._frontiers:
            self._frontiers[key] = first_body_outcome_operations(
                self.regions[region_id], self.operations, start,
                blocks=self.blocks, block_regions=self.block_regions)
        return self._frontiers[key]


def _created_body_origins(facts, create_results):
    """Propagate proved body origins along existing reference routes."""
    if not create_results:
        return {}
    consumers = {}

    def connect(operation, actual_formal_pairs):
        for actual, formal in actual_formal_pairs:
            use = facts.owned_use(operation.id, actual)
            if use is not None:
                consumers.setdefault(use.value, []).append(formal)

    for operation in facts.operations.values():
        if operation.terminator is not None:
            for edge in operation.terminator.successors:
                connect(operation, zip(
                    (saved_actual_use(operation, actual) for actual in edge.arguments),
                    facts.blocks[edge.block].arguments))
        if operation.callee is not None:
            callee = facts.regions[operation.callee]
            actuals = operation.invoke_arguments if operation.terminator is not None else operation.call_arguments
            connect(operation, zip(actuals, facts.blocks[callee.entry].arguments))

    origins = {value: {body} for value, body in create_results.items()}
    pending = deque(create_results.items())
    while pending:
        value, body = pending.popleft()
        for formal in consumers.get(value, ()):
            known = origins.setdefault(formal, set())
            if body not in known:
                known.add(body)
                pending.append((formal, body))
    return origins


def driver_errors(graph, *, facts, region_domains, incoming_edges):
    """Return D1 errors after canonical core and body validation have succeeded."""
    if facts.graph is not graph:
        raise ValueError("driver facts must index the exact program graph")
    errors = []

    def error(code, path, message):
        errors.append((code, str(path), message))

    values, operations, regions, blocks = facts.values, facts.operations, facts.regions, facts.blocks
    owners, operation_blocks = facts.block_regions, facts.operation_blocks
    bodies, interfaces = facts.bodies, facts.interfaces

    def protocol(value, kind):
        return isinstance(values.get(value), ProtocolValue) and values[value].protocol is kind

    def owned(op, use_id):
        return facts.owned_use(op.id, use_id)

    def argument(op, index):
        if index >= len(op.invoke_arguments):
            return None
        return owned(op, op.invoke_arguments[index])

    def invoke(operation_id, path):
        op = operations.get(operation_id)
        if op is None or op.terminator is None or op.terminator.kind is not TerminatorKind.INVOKE:
            error("driver_invoke", path, "declaration requires its real INVOKE operation")
            return None
        return op

    def normal_result(op, port, path, expected):
        signatures = {row.kind: row for row in op.completions}
        signature = signatures.get(ExitKind.NORMAL)
        if signature is None or port.index >= len(signature.types):
            error("driver_result", path, "NORMAL result index exceeds its signature")
            return None
        normal = next((edge for edge in op.terminator.successors
                       if edge.kind is ExitKind.NORMAL), None)
        if normal is None:
            error("driver_result", path, "result requires a real NORMAL successor")
            return None
        found = [formal for actual, formal in zip(normal.arguments, blocks[normal.block].arguments)
                 if isinstance(actual, CompletionResult) and actual.index == port.index]
        if len(found) != 1 or not expected(values[found[0]]):
            error("driver_result", path, "result must project exactly one correctly typed NORMAL formal")
            return None
        return found[0]

    # No metadata line can make a nonexistent operation into a new producer.
    per_table = (
        (graph.resumable_creates, "resumable_creates", "operation"),
        (graph.raw_driver_requests, "raw_driver_requests", "value"),
        (graph.async_request_creates, "async_request_creates", "operation"),
        (graph.request_association_reads, "request_association_reads", "operation"),
        (graph.driver_interfaces, "driver_interfaces", "region"),
        (graph.final_body_gates, "final_body_gates", "operation"),
        (graph.body_drives, "body_drives", "operation"),
    )
    for rows, name, key in per_table:
        seen = set()
        for row in rows:
            identity = getattr(row, key)
            if identity in seen:
                error("driver_duplicate", name, "declaration key must be unique")
            seen.add(identity)
    op_roles = {}
    for rows, name, _ in per_table:
        if name in ("raw_driver_requests", "driver_interfaces"):
            continue
        for row in rows:
            if row.operation in op_roles:
                error("driver_operation_role", row.operation, "one operation cannot have conflicting driver declarations")
            op_roles[row.operation] = name
    if errors:
        return tuple(errors)

    for row in graph.raw_driver_requests:
        value = values.get(row.value)
        if not protocol(row.value, P.RAW_DRIVER_REQUEST) or not isinstance(value.origin, OperationResult):
            error("driver_raw", row.value, "raw definition needs its RAW_DRIVER_REQUEST operation result")
            continue
        op = operations.get(value.origin.operation)
        if op is None or any(owned(op, use) is None or not isinstance(values[owned(op, use).value], ReferenceValue)
                             for use in row.operands):
            error("driver_raw", row.value, "raw operands must be owned Reference Uses")
            continue
        if (op.terminator is not None or op.callee is not None or op.results != (row.value,)
                or tuple(use.id for use in op.uses) != row.operands or op.accesses
                or op.completions or op.invoke_arguments or op.call_arguments
                or not op.effects_complete or not op.behaviors
                or not set(op.behaviors).issubset({Behavior.COMPUTE, Behavior.CONTROL, Behavior.REPRESENT})):
            error("driver_raw_pack", row.value, "raw pack must be a complete ordinary pure interface constructor with exactly its ordered operands")
    definitions = {row.value for row in graph.raw_driver_requests}
    for value in values.values():
        if protocol(value.id, P.RAW_DRIVER_REQUEST) and isinstance(value.origin, OperationResult) and value.id not in definitions:
            error("driver_raw", value.id, "constructed raw request requires its definition")

    environments = {row.region: row for row in graph.environments}
    layouts = {row.environment: row for row in graph.function_entries}
    create_results = {}
    for row in graph.resumable_creates:
        op = invoke(row.operation, row.operation)
        body = regions.get(row.body)
        if op is None:
            continue
        if body is None or body.id not in bodies:
            error("driver_create_body", row.operation, "Create requires a declared resumable body")
            continue
        environment = environments.get(body.id)
        if environment is None or environment.id not in layouts:
            error("driver_create_environment", row.operation, "bound frame Create requires complete environment and FunctionEntryLayout")
            continue
        reference_formals = tuple(formal for formal in blocks[body.entry].arguments
                                  if isinstance(values[formal], ReferenceValue))
        capture_formals = tuple(item.formal for item in row.captures)
        if (len(set(capture_formals)) != len(capture_formals)
                or set(capture_formals) != set(reference_formals)):
            error("driver_create_captures", row.operation, "captures must cover each nonstate entry Reference exactly once")
        capture_sources = [item.source for item in row.captures]
        if len(set(capture_sources)) != len(capture_sources):
            error("driver_create_captures", row.operation, "each capture occurrence needs its independent Use")
        for capture in row.captures:
            use = owned(op, capture.source)
            if (use is None or capture.source not in op.invoke_arguments
                    or not isinstance(values[use.value], ReferenceValue)):
                error("driver_create_capture", row.operation, "capture must name an owned invoke Reference input")
        function_formal = next(item.reference for item in environment.members
                               if item.role is EnvironmentRole.FUNCTION_OBJECT)
        source = next((item.source for item in row.captures if item.formal == function_formal), None)
        if row.function_object != source:
            error("driver_create_function", row.operation, "function_object must reuse the FUNCTION_OBJECT capture Use")
        result = normal_result(op, row.handle_result, row.operation, lambda value: isinstance(value, ReferenceValue))
        if result is not None:
            create_results[result] = row.body

    for row in graph.async_request_creates:
        op = invoke(row.operation, row.operation)
        if op is None:
            continue
        body, latent = owned(op, row.body), owned(op, row.latent)
        if (body is None or row.body not in op.invoke_arguments or not isinstance(values[body.value], ReferenceValue)
                or latent is None or row.latent not in op.invoke_arguments
                or not protocol(latent.value, P.RAW_DRIVER_REQUEST)):
            error("driver_async_create", row.operation, "request creation needs body Reference and latent raw invoke inputs")
        normal_result(op, row.request_result, row.operation, lambda value: isinstance(value, ReferenceValue))
    for row in graph.request_association_reads:
        op = invoke(row.operation, row.operation)
        if op is None:
            continue
        use = owned(op, row.request)
        if use is None or row.request not in op.invoke_arguments or not isinstance(values[use.value], ReferenceValue):
            error("driver_association", row.operation, "association reads its actual request Reference")
        normal_result(op, row.body_result, row.operation, lambda value: isinstance(value, ReferenceValue))
        normal_result(op, row.latent_result, row.operation,
                      lambda value: isinstance(value, ProtocolValue) and value.protocol is P.RAW_DRIVER_REQUEST)

    known_bodies = _created_body_origins(facts, create_results)

    drives = {row.operation: row for row in graph.body_drives}
    gates = {row.drive: row for row in graph.final_body_gates}
    if len(gates) != len(graph.final_body_gates):
        error("driver_gate_pair", "final_body_gates", "each Drive has exactly one final gate")

    role_regions = set()
    for region_id, interface in interfaces.items():
        region = regions.get(region_id)
        if region is None or interface.body not in bodies:
            error("driver_interface", region_id, "driver/body Region must exist and body be resumable")
            continue
        entry = blocks[region.entry]
        if (interface.receiver not in entry.arguments or not isinstance(values.get(interface.receiver), ReferenceValue)
                or interface.scheduler_request not in entry.arguments
                or not protocol(interface.scheduler_request, P.RAW_DRIVER_REQUEST)):
            error("driver_interface_roles", region_id, "receiver Reference and raw scheduler must be real entry parameters")
        domains = region_domains[region_id]
        if not interface.lifecycle or not set(interface.lifecycle).issubset(domains):
            error("driver_lifecycle", region_id, "lifecycle must be a nonempty subset of the driver's entry state domains")
        role_regions.add(region_id)

    for operation in operations.values():
        if isinstance(operation.terminator, BodyDriveTerminator) and operation.id not in drives:
            error("driver_missing", operation.id, "BodyDrive terminator requires its declaration")
    for row in graph.body_drives:
        op = operations.get(row.operation)
        region_id = owners.get(operation_blocks.get(row.operation))
        interface = interfaces.get(region_id)
        if op is None or not isinstance(op.terminator, BodyDriveTerminator):
            error("driver_terminator", row.operation, "BodyDrive requires its canonical typed terminator")
            continue
        if interface is None:
            error("driver_interface", row.operation, "BodyDrive requires its owning DriverInterface")
            continue
        if row.operation not in gates:
            error("driver_gate_pair", row.operation, "BodyDrive requires exactly one final gate")
        try:
            body, start, edge = facts.drive_target(row)
        except ValueError as exc:
            error("driver_target", row.operation, str(exc))
            continue
        if body.id != interface.body:
            error("driver_target", row.operation, "Drive target differs from its interface body")
        instance_use = owned(op, row.instance)
        if instance_use is not None and any(known != body.id for known in known_bodies.get(instance_use.value, ())):
            error("driver_instance_body", row.operation, "known Create source is incompatible with the declared body interface")
        if edge is None and any(not isinstance(values[formal], (ReferenceValue, StateVersionValue))
                                for formal in blocks[body.entry].arguments):
            error("driver_start_interface", row.operation, "Start capture profile requires every nonstate entry port to be a Reference")
        domains = region_domains[body.id]
        if not domains.issubset(region_domains[region_id]):
            error("driver_state_domains", row.operation, "body domains must be available in the driver")
        if ({item.location for item in row.state_inputs} != domains
                or len(row.state_inputs) != len(domains)):
            error("driver_state_inputs", row.operation, "state inputs must cover body domains exactly once")
        for item in row.state_inputs:
            use = owned(op, item.source)
            if use is None or not isinstance(values[use.value], StateVersionValue) or values[use.value].location != item.location:
                error("driver_state_inputs", row.operation, "state input source must have its declared LocationID")
        expected_slots = [] if edge is None else [index for index, slot in enumerate(edge.inputs)
                                                   if slot.location is None]
        if tuple(item.slot for item in row.resume_inputs) != tuple(expected_slots):
            error("driver_resume_inputs", row.operation, "resume inputs must cover nonstate slots once in target slot order; Start has none")
        if edge is not None:
            for item in row.resume_inputs:
                if item.slot >= len(edge.inputs):
                    continue
                use = owned(op, item.source)
                if use is None or value_type(values[use.value]) != resume_slot_type(edge.inputs[item.slot]):
                    error("driver_resume_type", row.operation, "delivered input differs from its target ResumeSlot type")
        for use_id in (row.instance, row.scheduler_request, *(() if row.session is None else (row.session,))):
            if owned(op, use_id) is None:
                error("driver_role", row.operation, "Drive role must be its own Use")
        if owned(op, row.instance) is not None and not isinstance(values[owned(op, row.instance).value], ReferenceValue):
            error("driver_role", row.operation, "instance must be a Reference")
        if owned(op, row.scheduler_request) is not None and not protocol(owned(op, row.scheduler_request).value, P.RAW_DRIVER_REQUEST):
            error("driver_role", row.operation, "scheduler must be RAW_DRIVER_REQUEST")
        if row.session is not None and owned(op, row.session) is not None and not isinstance(values[owned(op, row.session).value], ReferenceValue):
            error("driver_role", row.operation, "session must be a request Reference")
        expected_sources = set(facts.first_outcomes(body.id, start))
        sources = [successor.source for successor in op.terminator.successors]
        if len(set(sources)) != len(sources) or set(sources) != expected_sources:
            error("driver_outcomes", row.operation, "successors must cover exactly the first source outcomes")
        for successor in op.terminator.successors:
            if successor.source not in expected_sources:
                continue
            if len(incoming_edges[successor.block]) != 1:
                error("driver_outcome_landing", row.operation, "each source requires a private landing")

    for row in graph.final_body_gates:
        op = invoke(row.operation, row.operation)
        drive = drives.get(row.drive)
        region_id = owners.get(operation_blocks.get(row.operation))
        interface = interfaces.get(region_id)
        if op is None:
            continue
        if drive is None or interface is None or owners.get(operation_blocks.get(row.drive)) != region_id:
            error("driver_gate_pair", row.operation, "gate and declared Drive require one owning driver Region")
            continue
        drive_op = operations.get(row.drive)
        if drive_op is None or not isinstance(drive_op.terminator, BodyDriveTerminator):
            error("driver_gate_pair", row.operation, "paired Drive requires its existing canonical terminator")
            continue
        if {edge.kind for edge in op.terminator.successors} != {ExitKind.NORMAL, ExitKind.RAISE}:
            error("driver_gate_completions", row.operation, "final body gate requires both NORMAL and RAISE completions")
            continue
        normal = next(edge for edge in op.terminator.successors if edge.kind is ExitKind.NORMAL)
        landing = blocks[normal.block]
        if len(incoming_edges[landing.id]) != 1 or landing.operations != (row.drive,):
            error("driver_gate_corridor", row.operation, "NORMAL private landing must contain only its paired BodyDrive")
            continue
        role_indices = [row.body_argument, row.scheduler_argument, *row.delivery_arguments]
        if row.session_argument is not None:
            role_indices.append(row.session_argument)
        if len(set(role_indices)) != len(role_indices):
            error("driver_gate_roles", row.operation, "gate role input indices must be distinct")
        gate_body, gate_scheduler = argument(op, row.body_argument), argument(op, row.scheduler_argument)
        gate_session = None if row.session_argument is None else argument(op, row.session_argument)
        if any(argument(op, index) is None for index in role_indices):
            error("driver_gate_roles", row.operation, "gate role index exceeds invoke inputs")
            continue
        receiver_copy, scheduler_copy = facts.role_copies(region_id) if region_id in role_regions else ((), ())
        if gate_scheduler.value not in scheduler_copy:
            error("driver_scheduler_copy", row.operation, "scheduler must copy the fixed interface scheduler")
        if interface.receiver_kind is DriverReceiverKind.BODY:
            if gate_body.value not in receiver_copy or row.session_argument is not None or drive.session is not None:
                error("driver_receiver_copy", row.operation, "BODY receiver must copy entry receiver and has no session")
        elif gate_session is None or gate_session.value not in receiver_copy or drive.session is None:
            error("driver_receiver_copy", row.operation, "ASYNC_REQUEST session must copy its entry receiver")
        role_pairs = [(gate_body, drive.instance), (gate_scheduler, drive.scheduler_request)]
        if gate_session is not None and drive.session is not None:
            role_pairs.append((gate_session, drive.session))
        if len(row.delivery_arguments) != len(drive.resume_inputs):
            error("driver_gate_delivery", row.operation, "gate delivery list must match the Drive's ordered dynamic slots")
        role_pairs.extend((argument(op, index), item.source)
                          for index, item in zip(row.delivery_arguments, drive.resume_inputs))
        formal_positions = {formal: index for index, formal in enumerate(landing.arguments)}
        forwarded_slots = []
        for gate_use, drive_use_id in role_pairs:
            drive_use = owned(drive_op, drive_use_id)
            index = None if drive_use is None else formal_positions.get(drive_use.value)
            actual = None if index is None else normal.arguments[index]
            forward = owned(op, actual)
            if (gate_use is None or forward is None or forward.value != gate_use.value
                    or actual in op.invoke_arguments):
                error("driver_gate_forward", row.operation, "each original role needs its independent exact edge Use/formal/Drive Use")
            if index is not None:
                forwarded_slots.append(index)
        if len(set(forwarded_slots)) != len(forwarded_slots):
            error("driver_gate_forward", row.operation, "gate role forwarding slots must be distinct")
        domains = region_domains[region_id]
        input_states = [values[owned(op, use).value].location for use in op.invoke_arguments
                        if isinstance(values[owned(op, use).value], StateVersionValue)]
        if len(input_states) != len(domains) or set(input_states) != domains:
            error("driver_gate_states", row.operation, "gate must consume every driver current state domain once")
        for edge in op.terminator.successors:
            produced = [values[formal].location for actual, formal in zip(edge.arguments, blocks[edge.block].arguments)
                        if isinstance(actual, CompletionResult) and isinstance(values[formal], StateVersionValue)]
            if len(produced) != len(domains) or set(produced) != domains:
                error("driver_gate_states", row.operation, "both gate completions must produce every driver poststate")
        for item in drive.state_inputs:
            use = owned(drive_op, item.source)
            index = None if use is None else formal_positions.get(use.value)
            if index is None or not isinstance(normal.arguments[index], CompletionResult):
                error("driver_gate_state_forward", row.operation, "body state must consume the gate NORMAL produced poststate")

    return tuple(errors)


def no_body_errors(graph, *, facts, incoming_edges):
    """Check the private mark and predicate after driver/body validation."""
    if facts.graph is not graph:
        raise ValueError("driver facts must index the exact program graph")
    if not graph.created_close_tests and not graph.no_body_finishes:
        return ()

    from .lifecycle_validation import _body_update_errors

    errors = []

    def error(code, path, message):
        errors.append((code, str(path), message))

    values, operations, blocks = facts.values, facts.operations, facts.blocks
    owners, operation_blocks = facts.block_regions, facts.operation_blocks
    interfaces = facts.interfaces
    scopes = {row.region for row in graph.driver_lifecycles}
    predicates = {}

    def scope(operation, path):
        region = owners[operation_blocks[operation.id]]
        interface = interfaces.get(region)
        if region not in scopes or interface is None:
            error("no_body_scope", path, "no-body declarations require their owning enabled driver scope")
            return None
        return interface

    for row in graph.created_close_tests:
        op = operations.get(row.operation)
        if op is None:
            error("no_body_test_operation", row.operation, "test must name its real operation")
            continue
        interface = scope(op, row.operation)
        if interface is None:
            continue
        own = {use.id: use for use in op.uses}
        receivers, schedulers = facts.role_copies(interface.region)

        def role(identity, expected, kind):
            use = own.get(identity)
            valid = use is not None and use.role is kind
            if expected is ReferenceValue:
                valid = valid and isinstance(values[use.value], ReferenceValue)
            else:
                valid = (valid and isinstance(values[use.value], ProtocolValue)
                         and values[use.value].protocol is P.RAW_DRIVER_REQUEST)
            return use if valid else None

        instance = role(row.instance, ReferenceValue, UseRole.REFERENCE)
        scheduler = role(row.scheduler_request, ProtocolValue, UseRole.PROTOCOL)
        effective = role(row.effective_request, ProtocolValue, UseRole.PROTOCOL)
        session = None if row.session is None else role(row.session, ReferenceValue, UseRole.REFERENCE)
        if (instance is None or scheduler is None or effective is None
                or scheduler.value not in schedulers):
            error("no_body_test_roles", row.operation, "test requires actual Reference/RAW roles and fixed scheduler copy")
        if interface.receiver_kind is DriverReceiverKind.BODY:
            if (row.session is not None or instance is None or instance.value not in receivers
                    or effective is None or scheduler is None or effective.value != scheduler.value):
                error("no_body_test_roles", row.operation, "BODY test uses its fixed instance, independent Uses of the same scheduler, and no session")
        elif session is None or session.value not in receivers:
            error("no_body_test_roles", row.operation, "ASYNC_REQUEST test requires its fixed session; body and latent remain actual values")

        result = values.get(row.result)
        if (op.results != (row.result,) or not isinstance(result, DataValue)
                or result.data_type != "builtins.bool"
                or not isinstance(result.origin, OperationResult)
                or result.origin.operation != op.id):
            error("no_body_test_result", row.operation, "test defines exactly its own builtins.bool result")
        if (op.terminator is not None or op.callee is not None or op.call_arguments
                or op.invoke_arguments or op.completions or not op.effects_complete
                or set(op.behaviors) != {Behavior.CONTROL, Behavior.READ}):
            error("no_body_test_shape", row.operation, "predicate is a complete callback-free ordinary read intrinsic")
        lifecycle = set(interface.lifecycle)
        locations, state_uses = [], set()
        for access in op.accesses:
            locations.append(access.location)
            state_uses.add(access.before)
            before = own.get(access.before)
            if (access.mode is not AccessMode.READ or access.after is not None
                    or access.full_overwrite or access.footprint.kind is not FootprintKind.WHOLE
                    or before is None or before.role is not UseRole.STATE_CONTENT
                    or not isinstance(values[before.value], StateVersionValue)):
                error("no_body_test_access", row.operation, "predicate only reads whole current lifecycle state")
        if set(locations) != lifecycle or len(locations) != len(lifecycle):
            error("no_body_test_access", row.operation, "predicate reads each lifecycle domain exactly once")
        roles = {row.instance, row.scheduler_request, row.effective_request}
        if row.session is not None:
            roles.add(row.session)
        if set(own) != state_uses | roles:
            error("no_body_test_uses", row.operation, "predicate owns only its independent roles and access-before Uses")
        block = blocks[operation_blocks[op.id]]
        branch = operations[block.operations[-1]]
        condition = next((use for use in branch.uses
                          if branch.terminator is not None and use.id == branch.terminator.condition), None)
        if (len(block.operations) < 2 or block.operations[-2] != op.id
                or branch.terminator is None or branch.terminator.kind is not TerminatorKind.BRANCH
                or condition is None or condition.value != row.result):
            error("no_body_test_branch", row.operation, "test must immediately precede the branch of its own bool")
        else:
            if (not branch.effects_complete or set(branch.behaviors) != {Behavior.CONTROL}
                    or branch.results or branch.accesses or branch.callee is not None
                    or branch.call_arguments or branch.invoke_arguments or branch.completions):
                error("no_body_branch_shape", branch.id, "guard branch is complete callback-free control forwarding without writes")
            predicates[branch.id] = row

    # Only an explicit Finish asserts a mark. A predicate alone never does.
    for row in graph.no_body_finishes:
        op = operations.get(row.operation)
        if op is None:
            error("no_body_operation", row.operation, "finish must name its real operation")
            continue
        interface = scope(op, row.operation)
        if interface is None:
            continue
        branch = operations.get(row.precondition.operation)
        test = predicates.get(row.precondition.operation)
        if (row.precondition.successor_index != 0 or branch is None or test is None
                or owners[operation_blocks[branch.id]] != interface.region):
            error("no_body_precondition", row.operation, "finish requires its driver's declared predicate branch true edge")
            continue
        edge = branch.terminator.successors[0]
        landing = blocks[edge.block]
        if (operation_blocks[op.id] != edge.block or landing.operations[0] != op.id
                or len(incoming_edges[edge.block]) != 1):
            error("no_body_landing", row.operation, "finish is first on the exact private true landing")
        own = {use.id: use for use in op.uses}
        instance = own.get(row.instance)
        test_op = operations[test.operation]
        test_owner = next((use.value for use in test_op.uses if use.id == test.instance), None)
        branch_uses = {use.id: use for use in branch.uses}
        owner_matches = instance is not None and instance.value == test_owner
        if instance is not None and instance.value in landing.arguments:
            actual = edge.arguments[landing.arguments.index(instance.value)]
            owner_matches = (isinstance(actual, UseID) and actual in branch_uses
                             and branch_uses[actual].value == test_owner)
        if (instance is None or instance.role is not UseRole.REFERENCE
                or not isinstance(values[instance.value], ReferenceValue) or not owner_matches):
            error("no_body_owner", row.operation, "finish owner must exactly copy this predicate instance")
        _body_update_errors(op, row.instance, interface, values, error, "no_body")
    return tuple(errors)


__all__ = ["DriverFacts", "body_drive_target", "body_outcome_uses", "driver_errors", "no_body_errors"]
