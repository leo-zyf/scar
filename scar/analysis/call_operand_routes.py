"""Call operand routes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from scar.analysis.call_edges import CallEdgeReport, _canonical, _link_call_model
from scar.ir.program.call_edges import (
    CallEdgeBudget,
    CallEdgeModel,
    CallLinkStatus,
    ReferenceOccurrence,
    ReferenceRole,
    FrameCompletion,
    InvocationKey,
)
from scar.ir.program.call_operand_routes import (
    OperandRouteBudget,
    OperandRouteModel,
    ArgumentRoute,
    CompletionRoute,
    CoreReferenceKind,
    CoreReferenceRoute,
    CoreReferenceTemplate,
    CoreReferenceValue,
    FrameRouteObligations,
    InvocationReturn,
    OperandQueryRouteKind,
    OperandQueryStep,
    OperandReferenceNode,
    OperandReferenceTrace,
    OperandRouteIssue,
    OperandRouteIssueKind as Issue,
    ReferenceConsistency,
)
from scar.ir.program.call_operands import (
    CallOperandBudget,
    CALL_WINDOWS_RULE,
    LOAD_OCCURRENCES_RULE,
    CallableKind,
    LoadNamespace,
)
from scar.ir.program.function_origins import CallCompilationProfile
from scar.ir.record_codec import decode, encode
import ast
import dis
import inspect
from bisect import bisect_left
from collections import defaultdict, deque
from scar.ir.program.functions import BodyLoweringProfile, ParameterKind
from scar.ir.program.model import (
    BlockArgument,
    CompletionResult,
    ExitKind,
    LiteralOrigin,
    OperationResult,
    ReferenceValue,
    StateVersionValue,
    TerminatorKind,
    UseRole,
)
from scar.ir.semantics_v2 import Opcode
from scar.trace.code_snapshot import site as code_site


# Replayable, conditional cross-call routes from original operand evidence.
#
# The report stores manifests and thin core templates. Instruction transcripts
# remain in their bounded raw sidecar instead of being duplicated in each report.
# Public replay always reads those inputs and rebuilds the routes from source.
if TYPE_CHECKING:
    from scar.analysis.call_trace_reader import CallOperandTrace


_AUTHORITY = (
    "target_executed_by_linker", "function_environment_established",
    "contents_or_effects_established", "world_equivalence_established",
    "whole_program_complete", "transformation_authorized",
)
_FIELDS = {
    "schema", "schema_version", "trace", "project_root", "compilation_profile",
    "frame_budget", "operand_budget", "route_budget", "frame_files", "code_links",
    "operand_files", "operand_metadata", "inventory_sha256", "routing", "summary",
    *_AUTHORITY,
}
_PARTITION_FIELDS = {"source_body_access", "entry_parameters", "environments"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _bounded_budget(kind, value, ceiling):
    _require(type(ceiling) is kind, "independent verifier budget must be typed")
    ceiling.__post_init__()
    result = decode(kind, value)
    for name in result.__dataclass_fields__:
        _require(getattr(result, name) <= getattr(ceiling, name),
                 "operand report exceeds independent verifier budget: " + name)
    return result


@dataclass(frozen=True)
class CallOperandReport:
    frame_model: CallEdgeModel
    operands: CallOperandTrace
    routing: OperandRouteModel
    # Independent raw sidecar projection.  The historical v1 operand report
    # remains byte-compatible; callers explicitly retain this separate record.
    environments: dict | None = None
    source_body_access: dict | None = None
    entry_parameters: tuple[ReferenceOccurrence, ...] = ()

    def summary(self):
        result = {
            "frame": CallEdgeReport(self.frame_model).summary(),
            "operand_events": len(self.operands.events),
            "routing": self.routing.summary(),
            "scope": "conditional reference provenance; state, contents and effects remain separate",
            **{name: False for name in _AUTHORITY},
        }
        if self.source_body_access is not None:
            result["entry_parameter_occurrences"] = len(self.entry_parameters)
            result["source_body_storage"] = "PARTITIONED"
        return result

    def to_dict(self):
        result = {
            "schema": "scar.call-operand-links", "schema_version": 1,
            "trace": self.frame_model.trace,
            "project_root": self.frame_model.project_root,
            "compilation_profile": CallCompilationProfile.RECORDED_FUTURE_FLAGS.value,
            "frame_budget": encode(self.frame_model.budget),
            "operand_budget": encode(self.operands.budget),
            "route_budget": encode(self.routing.budget),
            "frame_files": encode(self.frame_model.files),
            "code_links": encode(self.frame_model.code_links),
            "operand_files": encode(self.operands.files),
            "operand_metadata": encode(self.operands.metadata),
            "inventory_sha256": self.frame_model.inventory_sha256,
            "routing": encode(self.routing), "summary": self.summary(),
            **{name: False for name in _AUTHORITY},
        }
        if self.source_body_access is not None:
            result.update({"schema_version": 2, "source_body_access": self.source_body_access,
                           "entry_parameters": encode(self.entry_parameters), "environments": self.environments})
        return result

    @staticmethod
    def check_replay_header(document, *, frame_ceiling=CallEdgeBudget(),
                            operand_ceiling=CallOperandBudget(), route_ceiling=OperandRouteBudget()):
        """Validate versions, authority and independent limits before input I/O."""
        _require(type(document) is dict and type(document.get("schema_version")) is int
                 and document["schema_version"] in (1, 2), "operand link report version mismatch")
        partitioned = document["schema_version"] == 2
        _require(set(document) == (_FIELDS | _PARTITION_FIELDS if partitioned else _FIELDS),
                 "operand link report fields mismatch")
        _require(document["schema"] == "scar.call-operand-links"
                 and type(document["schema_version"]) is int and document["schema_version"] in (1, 2),
                 "operand link report schema/version mismatch")
        _require(document["compilation_profile"] == CallCompilationProfile.RECORDED_FUTURE_FLAGS.value,
                 "operand link report compilation profile mismatch")
        _require(all(document[name] is False for name in _AUTHORITY),
                 "operand link report cannot establish execution or optimization authority")
        for name in ("trace", "project_root", "inventory_sha256"):
            _require(type(document[name]) is str and bool(document[name]),
                     "operand report requires nonempty " + name)
        frame = _bounded_budget(CallEdgeBudget, document["frame_budget"], frame_ceiling)
        operand = _bounded_budget(CallOperandBudget, document["operand_budget"], operand_ceiling)
        route = _bounded_budget(OperandRouteBudget, document["route_budget"], route_ceiling)
        # The model is untrusted. Decode with its independent limits already
        # checked, then rebuild it rather than consuming its claimed edges.
        routing = OperandRouteModel.from_dict(document["routing"])
        _require(routing.budget == route, "route model and report budgets disagree")
        if partitioned:
            _require(type(document["source_body_access"]) is dict
                     and document["source_body_access"].get("current_raw_scan_verified") is True
                     and document["source_body_access"].get("transformation_authorized") is False,
                     "partition report lacks its bounded source/current-capture receipts")
            decode(tuple[ReferenceOccurrence, ...], document["entry_parameters"])
        return frame, operand, route

    @classmethod
    def from_dict(cls, document, *, inventory=None, composition=None, source_bindings=None,
                  coordinates=None, code_witnesses=None, frame_ceiling=CallEdgeBudget(),
                  operand_ceiling=CallOperandBudget(), route_ceiling=OperandRouteBudget()):
        frame, operand, route = cls.check_replay_header(
            document, frame_ceiling=frame_ceiling, operand_ceiling=operand_ceiling,
            route_ceiling=route_ceiling,
        )
        partitioned = document["schema_version"] == 2
        if partitioned:
            _require(code_witnesses is not None and not code_witnesses.writable
                     and not code_witnesses.compile_missing, "public v2 replay needs a read-only CodeWitnessStore")
        else:
            _require(all(value is None for value in (composition, source_bindings, coordinates, code_witnesses)),
                     "legacy v1 replay cannot accept partitioned inputs")
        rebuilt = link_call_operands(document["trace"], document["project_root"], inventory,
            composition=composition, source_bindings=source_bindings, coordinates=coordinates,
            code_witnesses=code_witnesses, frame_budget=frame, operand_budget=operand, route_budget=route,
            include_environments=partitioned and document["environments"] is not None)
        _require(_canonical(rebuilt.to_dict()) == _canonical(document),
                 "operand links differ from current raw/source/inventory replay")
        return rebuilt

    def trace_reference(self, invocation, *, max_steps=4096):
        from scar.analysis.call_operand_routes import query_operand_routes
        return query_operand_routes(self.frame_model, self.routing, invocation, max_steps=max_steps)


def link_call_operands(trace, project_root, inventory=None, *, frame_budget=CallEdgeBudget(),
                       operand_budget=CallOperandBudget(), route_budget=OperandRouteBudget(),
                       include_environments=False, composition=None, source_bindings=None,
                       coordinates=None, code_witnesses=None, progress=None,
                       source_port_consumer=None, source_port_budget=None,
                       python_source_port_consumer=None, python_source_port_budget=None,
                       python_source_body_composition=None, python_source_inventory_artifact=None):
    """Own one raw replay; optional consumer(model, check) gets a separate sidecar.

    ``check`` verifies a selected typed source-port record against this same
    fresh session. The historical CallOperandReport/wire is unchanged.
    """
    return _link_call_operands(trace, project_root, inventory, frame_budget=frame_budget,
        operand_budget=operand_budget, route_budget=route_budget, include_environments=include_environments,
        composition=composition, source_bindings=source_bindings, coordinates=coordinates,
        code_witnesses=code_witnesses, progress=progress,
        source_port_consumer=source_port_consumer, source_port_budget=source_port_budget,
        python_source_port_consumer=python_source_port_consumer, python_source_port_budget=python_source_port_budget,
        python_source_body_composition=python_source_body_composition,
        python_source_inventory_artifact=python_source_inventory_artifact)


def _link_call_operands_owned(owner, trace, project_root, **kwargs):
    from scar.analysis.current_source_evidence import _OwnedCurrent
    _require(type(owner) is _OwnedCurrent, "private operand owner was not created by current orchestration")
    return _link_call_operands(trace, project_root, _owner=owner, **kwargs)


def _link_call_operands(trace, project_root, inventory=None, *, frame_budget=CallEdgeBudget(),
                        operand_budget=CallOperandBudget(), route_budget=OperandRouteBudget(),
                        include_environments=False, composition=None, source_bindings=None,
                        coordinates=None, code_witnesses=None, progress=None, _owner=None,
                        source_port_consumer=None, source_port_budget=None,
                        python_source_port_consumer=None, python_source_port_budget=None,
                        python_source_body_composition=None, python_source_inventory_artifact=None):
    """Automatically reconstruct original call operands and typed core routes."""
    from scar.analysis.call_trace_reader import read_call_operands
    from scar.analysis.call_operand_routes import build_operand_routes

    from scar.analysis.source_body_expansion import SourceBodyAccess
    _require(progress is None or callable(progress), "progress must be callable or None")
    _require(source_port_consumer is None or callable(source_port_consumer),
             "source-port consumer must be callable or None")
    _require(source_port_consumer is not None or source_port_budget is None,
             "source-port budget requires its explicit consumer")
    if source_port_consumer is not None:
        from scar.ir.program.source_call_bindings import SourceCallPortBudget
        source_port_budget = SourceCallPortBudget() if source_port_budget is None else source_port_budget
        _require(type(source_port_budget) is SourceCallPortBudget, "source-port budget must be typed")
        source_port_budget.__post_init__()
    _require(python_source_port_consumer is None or callable(python_source_port_consumer),
             "Python source-port consumer must be callable or None")
    _require(python_source_port_consumer is not None or all(x is None for x in
        (python_source_port_budget, python_source_body_composition, python_source_inventory_artifact)),
        "Python source-port inputs require the explicit v2 consumer")
    if python_source_port_consumer is not None:
        from scar.ir.program.python_call_targets import PythonSourcePortBudget
        python_source_port_budget = PythonSourcePortBudget() if python_source_port_budget is None else python_source_port_budget
        _require(type(python_source_port_budget) is PythonSourcePortBudget, "Python source-port budget must be typed")
        python_source_port_budget.__post_init__()
        if python_source_body_composition is not None:
            from scar.analysis.program_composition import CompositionIndex
            _require(type(python_source_body_composition) is CompositionIndex
                     and python_source_body_composition.has_source_entry_publication,
                     "Python body input requires an ordinary qualified entry composition")
        if python_source_inventory_artifact is not None:
            from scar.ir.program.source_references import SourceArtifactLocator
            _require(type(python_source_inventory_artifact) is SourceArtifactLocator,
                     "Python inventory artifact must be its exact typed original locator")
            python_source_inventory_artifact.__post_init__()
    def notify(stage, **fields):
        if progress is not None:
            progress({"stage": stage, **fields})
    partitioned = composition is not None
    if partitioned:
        _require(inventory is None and all(value is not None for value in
                     (source_bindings, coordinates, code_witnesses)),
                 "partition path requires explicit composition/source bindings/coordinates/witnesses and no inventory")
    else:
        _require(inventory is not None and all(value is None for value in
                     (source_bindings, coordinates, code_witnesses)),
                 "legacy operand/core linking requires its original automatic inventory")
    _require(type(include_environments) is bool, "include_environments must be an exact bool")
    _require(type(route_budget) is OperandRouteBudget, "route budget must be typed")
    route_budget.__post_init__()
    root = Path(project_root).resolve()
    notify("OPERAND_READ_BEGIN")
    operands = (read_call_operands(trace, root, budget=operand_budget, frame_budget=frame_budget)
                if _owner is None else _owner.require(trace, root, frame_budget, operand_budget, coordinates))
    notify("OPERAND_READ_END", invocations=len(operands.entries), events=len(operands.events))
    access = None
    if partitioned:
        access = (_owner.checked_source_access(composition, source_bindings, coordinates, code_witnesses)
                  if _owner is not None else None)
        if access is None:
            access = SourceBodyAccess(composition, source_bindings, coordinates, code_witnesses)
            access.bind_scan(operands.frame_scan, operands.entries, operands.exits)
        _require(root == access.root, "partition source project root differs from current operand replay")
        inventory = access
        notify("SOURCE_BODY_ACCESS_BOUND", counts=access.scan_counts, selected_bodies=len(access.selected))
    model, sources = _link_call_model(
        trace, root, inventory, frame_budget,
        compilation_profile=CallCompilationProfile.RECORDED_FUTURE_FLAGS,
        frame_scan=operands.frame_scan, entries=operands.entries, exits=operands.exits,
        allow_v6=True, include_identity_routes=False,
    )
    notify("CALL_MODEL_LINKED", invocations=len(model.edges))
    routing = build_operand_routes(call_model=model, operand_model=operands, inventory=inventory,
                                   sources=sources, budget=route_budget)
    notify("OPERAND_ROUTES_BUILT", summary=routing.summary())
    environments = None
    if include_environments:
        from scar.analysis.callable_environment_import import import_callable_environments
        environments = import_callable_environments(trace, inventory=inventory, sources=sources,
            call_model=model, operand_model=operands,
            max_raw_bytes=operand_budget.max_raw_bytes,
            max_events=operand_budget.max_events,
            max_line_bytes=operand_budget.max_line_bytes,
            max_references=operand_budget.max_references)
        notify("CALLABLE_ENVIRONMENTS_BOUND", summary=environments["summary"])
    entry_parameters = []
    if access is not None:
        for edge in model.edges:
            if (edge.target.status is not CallLinkStatus.VERIFIED or edge.entry.suspended
                    or not edge.entry.parameters_complete or edge.entry.code not in sources.objects
                    or edge.entry.invocation.process_id in access.unbound_processes
                    or edge.target.definition not in sources.ports):
                continue
            signature = sources.signatures[edge.target.definition]
            if {parameter.name for parameter in edge.entry.parameters} != {parameter.name for parameter in signature.parameters}:
                continue
            for parameter in edge.entry.parameters:
                source, value = sources.parameter_port(edge.target.definition, parameter.name)
                if value is not None:
                    entry_parameters.append(ReferenceOccurrence(edge.entry.anchor, edge.entry.invocation,
                        edge.entry.invocation, ReferenceRole.PARAMETER, parameter.name, parameter.ref,
                        source_reference=source, core_value=value))
    if source_port_consumer is not None:
        from scar.analysis.source_call_bindings import build_source_call_port_bindings
        source_ports = build_source_call_port_bindings(call_model=model, operand_model=operands,
            inventory=inventory, sources=sources, budget=source_port_budget, consumer=source_port_consumer)
        notify("SOURCE_CALL_PORTS_BUILT", summary=source_ports.summary())
    if python_source_port_consumer is not None:
        from scar.analysis.python_call_targets import build_python_source_ports
        python_ports = build_python_source_ports(trace=operands, call_model=model, inventory=inventory,
            sources=sources, inventory_artifact=python_source_inventory_artifact,
            body_composition=python_source_body_composition, budget=python_source_port_budget,
            consumer=python_source_port_consumer)
        notify("PYTHON_SOURCE_CALL_PORTS_V2_BUILT", summary=python_ports.summary())
    return CallOperandReport(model, operands, routing, environments,
                             access.receipt() if access is not None else None, tuple(entry_parameters))


# Replay operand occurrences into conditional core-reference routes.
#
# Neither equal tokens nor static candidate signatures select a route. Actual
# position, a complete dynamic instruction window and the unique direct entry
# handshake supply its origin. The public report independently replays inputs.
class _Boundary(Exception):
    def __init__(self, kind, detail):
        self.kind, self.detail = kind, detail


def _need(condition, kind, detail):
    if not condition:
        raise _Boundary(kind, detail)


def _span(node):
    return node.lineno, node.end_lineno, node.col_offset, node.end_col_offset


def _consistency(left, right):
    """Only comparable weak identities can contradict a positional rule."""
    if left is not None and right is not None and left.startswith('weak:') and right.startswith('weak:'):
        _need(left == right, Issue.REFERENCE_CONTRADICTION,
              'captured weak identities contradict the proposed positional route')
        return ReferenceConsistency.AGREES
    return ReferenceConsistency.UNKNOWN


def _layout_matches(layout, code):
    count = code.co_argcount + code.co_kwonlyargcount
    count += bool(code.co_flags & inspect.CO_VARARGS) + bool(code.co_flags & inspect.CO_VARKEYWORDS)
    return (layout.argcount == code.co_argcount and layout.posonlyargcount == code.co_posonlyargcount
            and layout.kwonlyargcount == code.co_kwonlyargcount and layout.flags == code.co_flags
            and layout.parameters == code.co_varnames[:count])


def _checked_formal_ports(body, interface, signature, values, blocks, definition):
    region = next((region for region in body.graph.regions if region.id == body.graph.root), None)
    _need(region is not None and interface.region == region.id
          and interface.definition == signature.definition == definition,
          Issue.SOURCE_CORE_BOUNDARY, 'formal interface has no matching root/definition')
    entry = blocks[region.entry]
    _need(len(signature.parameters) == len(interface.parameters)
          and tuple(parameter.slot for parameter in signature.parameters)
          == tuple(port.slot for port in interface.parameters),
          Issue.SOURCE_CORE_BOUNDARY, 'explicit formal slots do not cover the declared signature')
    for port in interface.parameters:
        value = values.get(port.value)
        _need(type(value) is ReferenceValue and isinstance(value.origin, BlockArgument)
              and value.region == region.id and value.origin.block == entry.id
              and 0 <= value.origin.index < len(entry.arguments)
              and entry.arguments[value.origin.index] == port.value,
              Issue.SOURCE_CORE_BOUNDARY, 'formal is not its exact declared entry reference port')
    if interface.profile is BodyLoweringProfile.STRUCTURED_ENV:
        environments = [environment for environment in body.graph.environments if environment.region == region.id]
        _need(len(environments) == 1, Issue.SOURCE_CORE_BOUNDARY,
              'environment body has no unique existing environment interface')
        environment, = environments
        layouts = [layout for layout in body.graph.function_entries if layout.environment == environment.id]
        _need(len(layouts) == 1 and layouts[0].parameters == interface.parameters
              and environment.code_definition == definition,
              Issue.SOURCE_CORE_BOUNDARY, 'explicit formal layout differs from the environment entry')
        expected = (tuple(member.reference for member in environment.members)
                    + tuple(port.value for port in interface.parameters) + (interface.world,))
    else:
        expected = tuple(port.value for port in interface.parameters) + (interface.world,)
    _need(entry.arguments == expected and type(values.get(interface.world)) is StateVersionValue,
          Issue.SOURCE_CORE_BOUNDARY, 'formal/environment/world partition differs from the actual entry')
    return interface.parameters



class _Core:
    def __init__(self, sources):
        self.sources = sources
        self.use_operations = {}
        self.use_sources = {}
        self.value_records = {}

    def values(self, definition):
        if definition not in self.value_records:
            self.value_records[definition] = {
                value.id: value for value in self.sources.bodies[definition].graph.values}
        return self.value_records[definition]

    def formal_ports(self, definition):
        """Check original project formals under the existing local namespace."""
        body, _, _, blocks = self.sources.core[definition]
        return _checked_formal_ports(body, self.sources.ports[definition],
            self.sources.signatures[definition], self.values(definition), blocks, definition)

    def qualified_formal_ports(self, context):
        """The same ABI guard, after an exact original qualified body lookup."""
        from scar.analysis.call_edges import _QualifiedBodyContext
        if type(context) is not _QualifiedBodyContext:
            raise TypeError('exact original qualified body context required')
        body, _, _, blocks = context.core
        return _checked_formal_ports(body, context.bundle.interface, context.signature,
            {value.id: value for value in body.graph.values}, blocks, context.reference.definition.definition)

    def return_root(self, edge):
        if (not edge.exit_verified or edge.exit is None
                or edge.exit.completion is not FrameCompletion.NORMAL or edge.entry.suspended
                or edge.target.definition not in self.sources.core):
            return None
        definition = edge.target.definition
        node = self.sources.returned_node(edge.exit, edge.target)
        if node is None:
            return None
        reference = self.sources.source(edge.target.snapshot.path, node.value)
        use_id, value_id = self.sources.return_operand(definition, node, reference)
        if use_id is None:
            return None
        body, operations, uses, _ = self.sources.core[definition]
        if definition not in self.use_operations:
            self.use_operations[definition] = {use.id: op for op in operations.values() for use in op.uses}
        operation = self.use_operations[definition].get(use_id)
        if (operation is None or operation.terminator is None
                or operation.terminator.kind is not TerminatorKind.RETURN
                or operation.terminator.values[0] != use_id or len(operation.terminator.values) != 2):
            return None
        return InvocationReturn(edge.entry.invocation, definition, body.graph.root, edge.exit.anchor,
                                reference, operation.id, use_id, value_id,
                                uses[operation.terminator.values[1]].value)

    def template(self, definition):
        sources = self.sources
        body, operations, uses, blocks = sources.core[definition]
        interface, signature = sources.ports[definition], sources.signatures[definition]
        names = {parameter.slot: parameter.name for parameter in signature.parameters}
        formal = {port.value: (port.slot, names[port.slot]) for port in self.formal_ports(definition)}
        references = {value.id: value for value in body.graph.values if type(value) is ReferenceValue}
        source_links = defaultdict(list)
        for link in body.metadata.value_links:
            source_links[link.value].append(link.source)
        call_results, completion_results, routes = set(), set(), []
        source_opcodes = {link.operation: link.opcode for link in body.metadata.operation_links}
        for operation in operations.values():
            if operation.terminator is None:
                continue
            for successor, edge in enumerate(operation.terminator.successors):
                block = blocks[edge.block]
                for port, operand in enumerate(edge.arguments):
                    destination = block.arguments[port]
                    if destination not in references:
                        continue
                    if isinstance(operand, CompletionResult):
                        if edge.kind is ExitKind.NORMAL and operand.index == 0:
                            completion_results.add(destination)
                            if (interface.profile is BodyLoweringProfile.BOUND_FRAME
                                    or source_opcodes.get(operation.id) is Opcode.CALL):
                                call_results.add(destination)
                    elif operand in uses and uses[operand].value in references:
                        routes.append(CoreReferenceRoute(destination, uses[operand].value,
                                                         operation.id, operand, successor, port))
        values = []
        for identity, value in references.items():
            candidates = tuple(dict.fromkeys(source_links[identity]))
            source = candidates[0] if len(candidates) == 1 else None
            if identity in formal:
                slot, name = formal[identity]
                parameter = next(parameter for parameter in signature.parameters if parameter.slot == slot)
                values.append(CoreReferenceValue(identity, CoreReferenceKind.FORMAL,
                                                  parameter.source, slot, name))
            else:
                kind = (CoreReferenceKind.CALL_RESULT if identity in call_results else
                        CoreReferenceKind.OPAQUE_OPERATION if identity in completion_results else
                        CoreReferenceKind.BLOCK_ARGUMENT if isinstance(value.origin, BlockArgument) else
                        CoreReferenceKind.LITERAL if isinstance(value.origin, LiteralOrigin) else
                        CoreReferenceKind.OPAQUE_OPERATION)
                values.append(CoreReferenceValue(identity, kind, source))
        return CoreReferenceTemplate(definition, body.graph.root, tuple(values), tuple(routes))


def _instruction_plan(code, site, node, rule=LOAD_OCCURRENCES_RULE, *, selection=None):
    """Build the narrow LOAD/PUSH_NULL window from exact verified bytecode."""
    from scar.trace.python_call_targets import load_slots
    instructions = list(dis.get_instructions(code))
    positions = {item.offset: index for index, item in enumerate(instructions)}
    _need(site.offset in positions, Issue.INSTRUCTION_WINDOW, 'CALL offset has no verified instruction')
    position = positions[site.offset]
    call = instructions[position]
    _need(call.opname == 'CALL' and call.arg == len(node.args), Issue.DISPATCH_BOUNDARY,
          'only a fixed positional CALL instruction is supported')
    count, reversed_window = 0, []
    for instruction in reversed(instructions[:position]):
        slots = load_slots(code, instruction)
        _need(slots is not None, Issue.INSTRUCTION_WINDOW, 'unsupported fused or non-Name load in CALL window')
        if slots:
            count += len(slots)
        else:
            _need(instruction.opname in ('PUSH_NULL', 'NOP'), Issue.INSTRUCTION_WINDOW,
                  'CALL operands are not a contiguous explicit Name-load window')
        reversed_window.append((instruction, slots))
        if count >= len(node.args) + 1:
            break
    _need(count == len(node.args) + 1, Issue.INSTRUCTION_WINDOW,
          'CALL needs one complete callee load and one occurrence per actual')
    window = list(reversed(reversed_window))
    tokens = [(instruction, slot) for instruction, slots in window for slot in slots]
    expected_names = [node.func.id, *(argument.id for argument in node.args)]
    _need([slot[2] for _, slot in tokens] == expected_names, Issue.INSTRUCTION_WINDOW,
          'ordered bytecode load slots differ from callee and actual Name occurrences')
    _need(all(slot[1] == 'FAST_LOCAL' for _, slot in tokens[1:]), Issue.DISPATCH_BOUNDARY,
          'actuals require explicit fast-local Name loads')
    window.append((call, ()))
    if rule == CALL_WINDOWS_RULE:
        # Recompute from the exact source-verified CodeType. A sparse row
        # subsequence cannot itself prove that no skipped control transfer
        # entered the middle of this physical instruction interval.
        from scar.trace.python_call_targets import call_window_selection
        suspended = inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR
        _need(not code.co_flags & suspended, Issue.DISPATCH_BOUNDARY,
              'suspended caller code is outside the sparse call-window rule')
        selected = selection
        if selected is None:
            selected, _ = call_window_selection(code, instructions)
        _need(selected.get(site.offset) == tuple(instruction.offset for instruction, _ in window),
              Issue.INSTRUCTION_WINDOW,
              'sparse CALL window is not one complete physical interval without internal jump or handler entry')
    else:
        _need(rule == LOAD_OCCURRENCES_RULE, Issue.CAPTURE_GAP, 'unsupported operand capture rule')
    return tuple(window), tuple(tokens)


def _has_index(indices, first, last):
    position = bisect_left(indices, first)
    return position < len(indices) and indices[position] <= last


def _load_window(attempt, node, caller_code, instructions, instruction_indices, previous_attempt,
                 local_gaps, global_epochs, read_issues, plans, rule=LOAD_OCCURRENCES_RULE):
    key = rule, attempt.site.code, attempt.site.offset
    if key not in plans:
        selection = None
        if rule == CALL_WINDOWS_RULE:
            from scar.trace.python_call_targets import call_window_selection
            selection_key = rule, attempt.site.code
            if selection_key not in plans:
                plans[selection_key] = call_window_selection(caller_code)[0]
            selection = plans[selection_key]
        plans[key] = _instruction_plan(caller_code, attempt.site, node, rule, selection=selection)
    plan, tokens = plans[key]
    stop = bisect_left(instruction_indices, attempt.index)
    actual = instructions[max(0, stop - len(plan)):stop]
    _need(len(actual) == len(plan) and all(item.index > previous_attempt for item in actual),
          Issue.INSTRUCTION_WINDOW, 'required instruction events are missing or belong to an earlier attempt')
    _need(all(encode(item.site) == code_site(caller_code, instruction.offset)
              for item, (instruction, _) in zip(actual, plan)),
          Issue.INSTRUCTION_WINDOW, 'dynamic instruction window differs from the verified bytecode plan')
    _need(all(item.hook_epoch == attempt.hook_epoch for item in actual), Issue.CAPTURE_GAP,
          'hook epoch changed inside an operand window')
    _need(actual[-1].frame_index == attempt.frame_index, Issue.HANDSHAKE_BOUNDARY,
          'CALL instruction and attempt disagree on the original-frame watermark')
    first = actual[0].index
    global_gap = global_epochs.get((attempt.process_id, attempt.hook_epoch))
    _need(not (global_gap is not None and global_gap < attempt.index)
          and not _has_index(local_gaps.get(attempt.invocation, ()), first, attempt.index),
          Issue.CAPTURE_GAP, 'capture gap invalidates the operand window')
    _need(not _has_index(read_issues.get(attempt.invocation, ()), first, attempt.index), Issue.CAPTURE_GAP,
          'reader found contradictory or unavailable code/layout evidence inside the operand window')
    observed = []
    for event, (instruction, slots) in zip(actual, plan):
        _need(len(event.loads) == len(slots), Issue.INSTRUCTION_WINDOW,
              'captured load count differs from the verified opcode suboperands')
        for load, (suboperand, namespace, name, slot_index) in zip(event.loads, slots):
            namespace_ok = (load.namespace in (LoadNamespace.GLOBAL, LoadNamespace.BUILTIN)
                            if namespace == 'GLOBAL' else load.namespace is LoadNamespace.FAST_LOCAL)
            _need(load.suboperand == suboperand and load.slot == name
                  and load.slot_index == slot_index and namespace_ok and load.reference is not None,
                  Issue.INSTRUCTION_WINDOW, 'load observation differs from exact opcode slot/suboperand')
            observed.append((event, load))
    return tuple(observed)


def build_operand_routes(*, call_model, operand_model, inventory, sources, budget=OperandRouteBudget()):
    """Build bounded conditional routes from independently validated inputs.

    The public report supplies one fresh shared reader result and the exact
    inventory/_Sources pair used to build call_model. No target code executes.
    """
    if type(budget) is not OperandRouteBudget:
        raise ValueError('operand routing requires a typed independent budget')
    supported_profiles = (BodyLoweringProfile.BOUND_FRAME, BodyLoweringProfile.STRUCTURED_BODY,
                          BodyLoweringProfile.STRUCTURED_ENV)
    if (sources.inventory is not inventory or inventory.interfaces is None
            or inventory.profile not in supported_profiles or inventory.interfaces.profile is not inventory.profile):
        raise ValueError('operand routes require the matching explicit frame/structured inventory used by source verification')
    core = _Core(sources)
    issues, templates, obligations, returned = [], [], [], []
    issue_seen = set()
    arguments, completions = [], []
    counts = {'values': 0, 'routes': 0}

    def issue(kind, detail, invocation=None, anchor=None):
        value = OperandRouteIssue(kind, detail, invocation, anchor)
        if value not in issue_seen:
            issue_seen.add(value)
            issues.append(value)

    def charge(routes=0, values=0):
        if counts['routes'] + routes > budget.max_routes or counts['values'] + values > budget.max_template_values:
            issue(Issue.ROUTE_BUDGET, 'operand route/template budget exhausted; omitted links remain unproved')
            return False
        counts['routes'] += routes
        counts['values'] += values
        return True

    definitions = tuple(dict.fromkeys(edge.target.definition for edge in call_model.edges
                        if edge.target.status is CallLinkStatus.VERIFIED
                        and edge.target.definition in sources.core
                        and edge.target.definition in sources.ports
                        and sources.frame_rules(edge.target.definition) is not None))
    available = set()
    for definition in definitions:
        try:
            template = core.template(definition)
        except _Boundary as error:
            issue(error.kind, error.detail)
            continue
        if not charge(len(template.routes), len(template.values)):
            continue
        templates.append(template)
        available.add(definition)
        body = sources.bodies[definition]
        obligations.append(FrameRouteObligations(definition, template.region, sources.frame_rules(definition),
                           body.metadata.assumptions,
                           tuple(op.id for op in body.graph.operations if not op.effects_complete)))
    edges = {edge.entry.invocation: edge for edge in call_model.edges}
    roots = {}
    for edge in call_model.edges:
        if edge.target.definition not in available:
            continue
        root = core.return_root(edge)
        if root is not None and charge(1):
            returned.append(root)
            roots[root.invocation] = root
    instructions = defaultdict(list)
    for instruction in operand_model.instructions:
        instructions[instruction.invocation].append(instruction)
    instruction_indices = {key: [item.index for item in values] for key, values in instructions.items()}
    handshakes = defaultdict(list)
    for handshake in operand_model.handshakes:
        handshakes[handshake.process_id, handshake.attempt_index].append(handshake)
    bad_processes = {meta.process_id for meta in operand_model.metadata
                     if meta.errors or meta.io_failed or meta.dropped_events or meta.previous_profile}
    bad_processes.update(item.process_id for item in operand_model.issues
                         if item.invocation is None and item.anchor is None)
    capture_rules = {}
    for meta in operand_model.metadata:
        if meta.process_id in capture_rules:
            bad_processes.add(meta.process_id)
        capture_rules[meta.process_id] = meta.rule
    local_gaps, global_gaps, global_epochs, read_issues = defaultdict(list), defaultdict(list), {}, defaultdict(list)
    for gap in operand_model.gaps:
        if gap.invocation is None:
            global_gaps[gap.process_id].append(gap.index)
            global_epochs.setdefault((gap.process_id, gap.hook_epoch), gap.index)
        else:
            local_gaps[gap.invocation].append(gap.index)
    for item in operand_model.issues:
        if item.invocation is not None and item.anchor is not None:
            read_issues[item.invocation].append(item.anchor.index)
    for mapping in (local_gaps, global_gaps, read_issues):
        for indices in mapping.values():
            indices.sort()
    previous, plans = {}, {}
    for attempt in operand_model.attempts:
        last = previous.get(attempt.invocation, 0)
        previous[attempt.invocation] = attempt.index
        try:
            rule = capture_rules.get(attempt.process_id)
            _need(rule in (LOAD_OCCURRENCES_RULE, CALL_WINDOWS_RULE), Issue.CAPTURE_GAP,
                  'attempt process has no supported capture rule metadata')
            _need(attempt.process_id not in bad_processes, Issue.CAPTURE_GAP,
                  'collector metadata reports loss, failure or an existing profile observer')
            matches = handshakes[attempt.process_id, attempt.index]
            _need(len(matches) == 1, Issue.HANDSHAKE_BOUNDARY, 'attempt has no unique real entry handshake')
            handshake, = matches
            edge = edges.get(handshake.invocation)
            caller = edges.get(attempt.invocation)
            _need(edge is not None and caller is not None and edge.caller_identity_verified
                  and edge.entry.caller_invocation == attempt.invocation
                  and edge.entry.anchor.index == handshake.entry_index
                  and caller.entry.anchor.index <= attempt.frame_index < handshake.entry_index
                  and handshake.thread_id == attempt.thread_id
                  and handshake.hook_epoch == attempt.hook_epoch,
                  Issue.HANDSHAKE_BOUNDARY, 'handshake does not identify the active direct caller/entry interval')
            if getattr(sources, 'source_access', None) is not None:
                _need(attempt.site.code in sources.objects and edge.entry.code in sources.objects,
                      Issue.SOURCE_CORE_BOUNDARY,
                      'CODE_WITNESS_UNAVAILABLE: verified source alone cannot certify a CALL instruction window')
            _need(len(edge.path) == 1 and edge.path[0].site == attempt.site
                  and edge.path[0].status is CallLinkStatus.VERIFIED
                  and edge.path[0].core_operation is not None
                  and edge.target.status is CallLinkStatus.VERIFIED
                  and caller.target.definition in available and edge.target.definition in available,
                  Issue.SOURCE_CORE_BOUNDARY, 'attempt/entry lacks one verified direct source/core call and body')
            call = edge.path[0]
            target_code = sources.objects.get(edge.entry.code)
            _need(attempt.dispatch.kind is CallableKind.EXACT_FUNCTION
                  and attempt.dispatch.code == edge.entry.code and target_code is not None
                  and _layout_matches(attempt.dispatch.layout, target_code),
                  Issue.DISPATCH_BOUNDARY, 'CALL current function/code/layout differs from the actual entry')
            _need(not any(_has_index(indices, attempt.index, handshake.index) for indices in (
                      global_gaps.get(attempt.process_id, ()), local_gaps.get(attempt.invocation, ()),
                      local_gaps.get(handshake.invocation, ()))),
                  Issue.CAPTURE_GAP, 'capture gap occurs during entry handshake')
            root = roots.get(edge.entry.invocation)
            normal_result = sources.result(call)
            node = sources.node(call.source)
            _need(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and not node.keywords and all(isinstance(argument, ast.Name) for argument in node.args),
                  Issue.DISPATCH_BOUNDARY, 'argument routing requires Name callee and explicit local Name actuals')
            layout = attempt.dispatch.layout
            unsupported = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS | inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR
            _need(not layout.flags & unsupported and layout.kwonlyargcount == 0
                  and layout.argcount == len(node.args), Issue.DISPATCH_BOUNDARY,
                  'receiver/default/pack/keyword or suspended binding is outside the all-positional rule')
            _need(edge.entry.parameters_complete
                  and tuple(parameter.name for parameter in edge.entry.parameters) == layout.parameters,
                  Issue.DISPATCH_BOUNDARY, 'actual frame parameters differ from dispatch-time layout')
            observations = _load_window(attempt, node, sources.objects[attempt.site.code],
                instructions[attempt.invocation], instruction_indices.get(attempt.invocation, []),
                last, local_gaps, global_epochs, read_issues, plans, rule)
            _, loaded_callee = observations[0]
            _need(loaded_callee.callable_shape.kind is CallableKind.EXACT_FUNCTION,
                  Issue.DISPATCH_BOUNDARY, 'pre-load callable is a bound method, adapter or unavailable shape')
            _consistency(loaded_callee.reference, attempt.dispatch.reference)
            signature, ports = sources.signatures[edge.target.definition], sources.ports[edge.target.definition]
            _need(tuple(parameter.name for parameter in signature.parameters) == layout.parameters
                  and all(parameter.kind in (ParameterKind.POSITIONAL_ONLY, ParameterKind.POSITIONAL)
                          for parameter in signature.parameters), Issue.DISPATCH_BOUNDARY,
                  'verified body formal interface differs from the actual dispatch layout')
            body, operations, uses, _ = sources.core[call.execution_owner]
            operation = operations[call.core_operation]
            _need(len(operation.invoke_arguments) == len(node.args) + 2
                  and operation.invoke_arguments[0] == call.callee_use,
                  Issue.SOURCE_CORE_BOUNDARY, 'core CALL does not have one actual Use per position')
            world_use = uses[operation.invoke_arguments[-1]]
            values = core.values(call.execution_owner)
            _need(world_use.role is UseRole.STATE_CONTENT and type(values.get(world_use.value)) is StateVersionValue
                  and all(type(values.get(uses[use_id].value)) is ReferenceValue
                          for use_id in operation.invoke_arguments[:-1]),
                  Issue.SOURCE_CORE_BOUNDARY, 'core CALL actuals/world do not have their declared reference/state roles')
            if call.execution_owner not in core.use_sources:
                core.use_sources[call.execution_owner] = {
                    (link.operation, link.use): link.source for link in body.metadata.use_links}
            use_sources = core.use_sources[call.execution_owner]
            pending = []
            for index, ((instruction, observation), parameter, formal, captured) in enumerate(zip(
                    observations[1:], signature.parameters, ports.parameters, edge.entry.parameters)):
                use_id = operation.invoke_arguments[index + 1]
                source = sources.source(call.source.path, node.args[index])
                _need(use_sources.get((operation.id, use_id)) == source and parameter.slot == formal.slot,
                      Issue.SOURCE_CORE_BOUNDARY, 'actual Use source or formal slot is not the exact indexed occurrence')
                consistency = _consistency(observation.reference, captured.ref)
                if index == 0 and not attempt.arg0_missing:
                    _consistency(observation.reference, attempt.arg0)
                pending.append(ArgumentRoute(attempt.anchor, attempt.invocation, edge.entry.invocation,
                    index, instruction.anchor, observation.suboperand, call.call, call.source,
                    call.execution_owner, call.core_region, call.core_operation, source, use_id,
                    uses[use_id].value, edge.target.definition, ports.region, parameter.slot,
                    formal.value, parameter.name, consistency))
            completion = None
            if root is not None and normal_result is not None:
                completion = CompletionRoute(attempt.anchor, attempt.invocation, edge.entry.invocation,
                    call.call, call.source, call.execution_owner, call.core_region, call.core_operation,
                    normal_result, root.definition, root.region, root.exit_anchor,
                    root.payload_use, root.payload_value, root.post_world)
            # The first completion rule shares the complete narrow argument
            # admission. A known capture/load/shape contradiction cannot leave
            # a positive completion after its argument routes were rejected.
            if charge(len(pending) + (completion is not None)):
                arguments.extend(pending)
                if completion is not None:
                    completions.append(completion)
        except _Boundary as error:
            issue(error.kind, error.detail, attempt.invocation, attempt.anchor)
    return OperandRouteModel(budget, tuple(arguments), tuple(completions), tuple(templates),
                             tuple(returned), tuple(obligations), tuple(issues))


def query_operand_routes(call_model, routes, invocation, *, max_steps=4096):
    """Follow conditional SSA/argument/completion routes, never opaque inputs."""
    if type(max_steps) is not int or not 1 <= max_steps <= 100_000:
        raise ValueError('max_steps must be an exact integer in [1, 100000]')
    if type(invocation) is tuple and len(invocation) == 3:
        invocation = InvocationKey(*invocation)
    if type(invocation) is not InvocationKey or type(routes) is not OperandRouteModel:
        raise ValueError('operand query requires typed route model and invocation')
    templates = {template.definition: template for template in routes.templates}
    values = {template.definition: {value.value: value for value in template.values} for template in routes.templates}
    definitions = {edge.entry.invocation: edge.target.definition for edge in call_model.edges}
    core = defaultdict(list)
    for template in routes.templates:
        for route in template.routes:
            core[template.definition, route.destination].append(route)
    arguments = {(route.callee, route.formal_value): route for route in routes.argument_routes}
    completions = defaultdict(list)
    for route in routes.completion_routes:
        completions[route.caller, route.normal_result].append(route)

    def node(owner, value):
        definition = definitions.get(owner)
        item = values.get(definition, {}).get(value)
        if item is None:
            return None
        return OperandReferenceNode(owner, definition, templates[definition].region, value,
                                    item.kind, item.source, item.formal_slot, item.formal_name)

    roots = tuple(value for root in routes.returns if root.invocation == invocation
                  for value in (node(root.invocation, root.payload_value),) if value is not None)
    queue, visited, traversed, endpoints, issues = deque(roots), set(), [], [], []
    steps, exhausted = 0, False
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        if steps >= max_steps:
            exhausted = True
            endpoints.extend(value for value in (current, *queue) if value not in endpoints)
            issues.append(OperandRouteIssue(Issue.QUERY_BUDGET,
                          'reference query budget exhausted with pending conditional routes', current.invocation))
            break
        visited.add(current)
        steps += 1
        next_steps = []
        control_boundary = False
        if current.kind is CoreReferenceKind.FORMAL:
            argument = arguments.get((current.invocation, current.value))
            if argument is not None:
                source = node(argument.caller, argument.actual_value)
                if source is not None:
                    next_steps.append(OperandQueryStep(current, source, OperandQueryRouteKind.ARGUMENT,
                                                       argument=argument))
        elif current.kind is CoreReferenceKind.CALL_RESULT:
            matches = completions[current.invocation, current.value]
            if len(matches) == 1:
                completion, = matches
                source = node(completion.callee, completion.payload_value)
                if source is not None:
                    next_steps.append(OperandQueryStep(current, source, OperandQueryRouteKind.COMPLETION,
                                                       completion=completion))
        elif current.kind is CoreReferenceKind.BLOCK_ARGUMENT:
            incoming = core[current.definition, current.value]
            if len({route.source for route in incoming}) > 1:
                # The template records static CFG successors, not which
                # successor this invocation took. Equal runtime references
                # cannot select a predecessor. A common SSA source across
                # every predecessor is independent of that missing choice.
                control_boundary = True
                issues.append(OperandRouteIssue(Issue.CONTROL_SELECTION_UNVERIFIED,
                              'block argument has different static predecessor sources; '
                              'this invocation has no verified control-to-CFG selection', current.invocation))
            else:
                for route in incoming:
                    source = node(current.invocation, route.source)
                    if source is not None:
                        next_steps.append(OperandQueryStep(current, source, OperandQueryRouteKind.CORE_REFERENCE,
                                                           core=route))
        if not next_steps:
            endpoints.append(current)
            if current.kind is CoreReferenceKind.CALL_RESULT:
                issues.append(OperandRouteIssue(Issue.COMPLETION_BOUNDARY,
                              'normal result has no unique observed attempt/entry/completion route', current.invocation))
            elif current.kind is CoreReferenceKind.FORMAL:
                issues.append(OperandRouteIssue(Issue.ARGUMENT_ROUTING_UNVERIFIED,
                              'formal origin stops at an unavailable or unsupported caller binding', current.invocation))
            elif current.kind not in (CoreReferenceKind.LITERAL,) and not control_boundary:
                issues.append(OperandRouteIssue(Issue.OPAQUE_REFERENCE,
                              'reference origin is opaque; producer inputs are not aliases', current.invocation))
        for step in next_steps:
            traversed.append(step)
            if step.source not in visited:
                queue.append(step.source)
    if not roots:
        issues.append(OperandRouteIssue(Issue.COMPLETION_BOUNDARY,
                      'requested invocation has no verified normal source/core return root', invocation))
    if any(issue.kind is Issue.ROUTE_BUDGET for issue in routes.issues):
        issues.extend(issue for issue in routes.issues if issue.kind is Issue.ROUTE_BUDGET)
    discovered = visited | set(endpoints)
    discovered.update(step.source for step in traversed)
    discovered.update(step.destination for step in traversed)
    # Global visited-node deduplication must not turn a reference cycle into
    # an apparently complete query with no endpoint. Detect cycles in the
    # actually traversed thin graph without recursive Python calls.
    degree = {value: 0 for value in discovered}
    adjacency = defaultdict(list)
    for step in dict.fromkeys(traversed):
        adjacency[step.destination].append(step.source)
        degree[step.source] += 1
    ready = deque(value for value, count in degree.items() if count == 0)
    while ready:
        value = ready.popleft()
        for source in adjacency[value]:
            degree[source] -= 1
            if degree[source] == 0:
                ready.append(source)
    cyclic = [value for value, count in degree.items() if count]
    if cyclic:
        issues.append(OperandRouteIssue(Issue.CYCLE_BOUNDARY,
                      'traversed conditional reference graph contains a cycle', invocation))
        endpoints.extend(value for value in cyclic if value not in endpoints)
    reached = {value.definition for value in discovered}
    return OperandReferenceTrace(invocation, roots, tuple(sorted(discovered, key=lambda item: (
        item.invocation.process_id, item.invocation.thread_id, item.invocation.invocation_id, item.value.wire))),
        tuple(dict.fromkeys(traversed)), tuple(dict.fromkeys(endpoints)), tuple(dict.fromkeys(issues)),
        tuple(item for item in routes.obligations if item.definition in reached), steps, max_steps)


__all__ = [
    'CallOperandReport',
    'link_call_operands',
    'build_operand_routes',
    'query_operand_routes',
]
