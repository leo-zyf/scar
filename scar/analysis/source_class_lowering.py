"""Source class lowering."""

from __future__ import annotations

import ast
import hashlib
from scar.analysis.program_lowering import _key
from scar.analysis.static_compilation import StaticCodeEvidence, compile_source_unit
from scar.ir.program.class_entries import (
    ClassBodyStatement,
    ClassCompilerStage,
    ClassEntryFrontier,
    ClassEntryInput,
    ClassEntryInputRole,
    ClassEntryOverlay,
    ClassEntryResult,
    ClassFrontierKind,
    RequiredClassBodyExpansion,
    compiler_cell_protocol,
    ClassNormalPayloadKind,
)
from scar.ir.program.control import ControlScope, ControlScopeKind
from scar.ir.program.control_validation import control_errors
from scar.ir.program.ids import (
    BlockID,
    ControlScopeID,
    LocationID,
    OperationID,
    RegionID,
    UseID,
    ValueID,
)
from scar.ir.program.model import (
    Behavior,
    Block,
    BlockArgument,
    CompletionResult,
    CompletionSignature,
    CompletionSuccessor,
    ExitKind,
    ExitPort,
    Footprint,
    FootprintKind,
    Operation,
    OperationResult,
    ProgramGraph,
    ReferenceValue,
    Region,
    Stability,
    StateLocation,
    StateVersionValue,
    Terminator,
    TerminatorKind,
    Use,
    UseRole,
    ValueKind,
    ValueType,
)
from scar.ir.program.source import (
    GapKind,
    LoweringGap,
    LoweringMetadata,
    OperationSource,
    SourceRole,
    UseSource,
    ValueSource,
)
from scar.ir.program.source_subjects import (
    ClassCompilerCell,
    ClassCompilerCellRole,
    ClassFrameABI,
    FrameKind,
    MappingFramePorts,
    SourceCompilationFrameRoot,
)
from scar.ir.program.validation import require_valid
from scar.ir.semantics_v2 import Opcode, SourceSnapshot
from scar.ir.v2.common import SourceReference
from scar.ir.v2.ids import OperationDefinitionID
from scar.ir.v2.semantic import OperationKind


# Pure static class-frame entry construction over the existing ProgramGraph.
#
# The class suite stays a typed required expansion. This module compiles retained
# source to layout only; it never executes or imports the target program.
_REF = ValueType(ValueKind.REFERENCE)


def _stable_id(kind, *parts):
    digest = hashlib.sha256("\0".join(str(part) for part in parts).encode()).hexdigest()[:32]
    return digest


def _span(node):
    return (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset)


def _node_reference(prepared, node):
    span = _span(node)
    matches = [atom.reference for atom in prepared.semantic.source_atoms.values()
               if atom.kind == type(node).__name__ and atom.reference.path == str(prepared.path)
               and atom.reference.fingerprint == prepared.semantics.sources[str(prepared.path)].fingerprint
               and (atom.reference.start_line, atom.reference.end_line,
                    atom.reference.start_column, atom.reference.end_column) == span]
    if len(matches) != 1:
        raise ValueError("class source mapping is absent or ambiguous for " + type(node).__name__)
    return matches[0]


def _resolve_class(prepared, definition, source):
    if type(definition) is not OperationDefinitionID or type(source) is not SourceReference:
        raise TypeError("class lowering requires the original DefinitionID and SourceReference")
    source.__post_init__()
    snapshot = prepared.semantics.sources.get(str(prepared.path))
    if (type(snapshot) is not SourceSnapshot or source.path != snapshot.path
            or source.fingerprint != snapshot.fingerprint):
        raise ValueError("class source differs from the prepared source snapshot")
    atom = prepared.semantic.source_atoms.get(source.atom_id)
    semantic_definition = prepared.semantic.definitions.get(definition)
    if (atom is None or atom.reference != source or atom.kind != "ClassDef"
            or semantic_definition is None or semantic_definition.kind is not OperationKind.OPAQUE
            or semantic_definition.code_id != source.atom_id.wire
            or source.atom_id not in semantic_definition.source_atoms
            or semantic_definition.source_file != source.path):
        raise ValueError("class DefinitionID is not the original SourceGraph ClassDef operation")
    matches = [node for node in ast.walk(prepared.tree)
               if type(node) is ast.ClassDef and _span(node) == (
                   source.start_line, source.end_line, source.start_column, source.end_column)]
    if len(matches) != 1:
        raise ValueError("class source span does not select exactly one original ClassDef")
    node = matches[0]
    if semantic_definition.label != "class " + node.name:
        raise ValueError("original SourceGraph class operation label differs from its ClassDef")
    return snapshot, semantic_definition, node


def _class_header_inputs(prepared, node):
    rows = []
    evaluation_order = 0
    for index, expression in enumerate(node.decorator_list):
        rows.append(ClassEntryInput(ClassEntryInputRole.DECORATOR, index, evaluation_order,
                                    _node_reference(prepared, expression)))
        evaluation_order += 1
    for index, expression in enumerate(node.bases):
        rows.append(ClassEntryInput(ClassEntryInputRole.BASE, index, evaluation_order,
                                    _node_reference(prepared, expression)))
        evaluation_order += 1
    for index, keyword in enumerate(node.keywords):
        role = (ClassEntryInputRole.METACLASS if keyword.arg == "metaclass" else
                ClassEntryInputRole.CLASS_KEYWORD if keyword.arg is not None else
                ClassEntryInputRole.CLASS_KEYWORD_EXPANSION)
        rows.append(ClassEntryInput(role, index, evaluation_order,
                                    _node_reference(prepared, keyword.value), keyword.arg))
        evaluation_order += 1
    return tuple(rows)


def _class_scope_statements(prepared, node):
    """Statements executed by this class suite, excluding deferred child scopes."""
    pending = list(reversed(node.body))
    nodes = []
    while pending:
        current = pending.pop()
        if isinstance(current, ast.stmt):
            nodes.append(current)
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        # Walk AST wrapper nodes such as ExceptHandler and match_case so the
        # statement census also covers suites nested below non-statement nodes.
        children = list(ast.iter_child_nodes(current))
        pending.extend(reversed(children))
    return tuple(ClassBodyStatement(index, type(statement).__name__,
                                    _node_reference(prepared, statement))
                 for index, statement in enumerate(nodes))


def _compiler_cells(code):
    return tuple(ClassCompilerCell(name, index, {
        "__class__": ClassCompilerCellRole.CLASS_OBJECT,
        "__classdict__": ClassCompilerCellRole.CLASS_NAMESPACE,
    }.get(name, ClassCompilerCellRole.LOCAL)) for index, name in enumerate(code.co_cellvars))


def _frontiers(source, *, has_freevars, has_class_cell, has_decorators):
    rows = [
        ClassEntryFrontier(ClassFrontierKind.BODY_REQUIRED_EXPANSION,
            "the complete original class-suite statement sequence remains an explicit required expansion", source),
        ClassEntryFrontier(ClassFrontierKind.PARENT_CLASS_STATEMENT_UNCONNECTED,
            "the original class statement and its header inputs are retained, but no parent creation operation is connected", source),
        ClassEntryFrontier(ClassFrontierKind.PREPARED_NAMESPACE_UNCONNECTED,
            "locals_namespace is a static class-local mapping port; no __prepare__ result or namespace identity is attached", source),
        ClassEntryFrontier(ClassFrontierKind.METACLASS_SELECTION_AND_CALL_OPEN,
            "metaclass selection and __prepare__/metaclass invocation effects remain open", source),
        ClassEntryFrontier(ClassFrontierKind.BASE_MRO_ENTRIES_OPEN,
            "base evaluation, __mro_entries__, metaclass conflict resolution, and MRO creation remain open", source),
        ClassEntryFrontier(ClassFrontierKind.DESCRIPTOR_AND_SUBCLASS_HOOKS_OPEN,
            "descriptor __set_name__ and base __init_subclass__ calls/effects are not expanded", source),
        ClassEntryFrontier(ClassFrontierKind.EFFECTS_OPEN,
            "class namespace reads/writes and construction effects are incomplete", source),
    ]
    if has_freevars:
        rows.append(ClassEntryFrontier(ClassFrontierKind.OUTER_CLOSURE_UNCONNECTED,
            "compiler freevar order is known but no parent cell actuals are connected", source))
    if has_class_cell:
        rows.append(ClassEntryFrontier(ClassFrontierKind.CLASSCELL_ACCEPTANCE_AND_FILL_UNPROVEN,
            "the body returns its compiler __class__ cell, but parent metaclass acceptance/fill is unproved", source))
    if has_decorators:
        rows.append(ClassEntryFrontier(ClassFrontierKind.DECORATOR_APPLICATION_OPEN,
            "decorator expressions are retained in source order; application/result/raise routes are unexpanded", source))
    return tuple(rows)


def _build_graph(definition, source, snapshot, compilation, abi, normal_payload, statements, frontiers):
    region = RegionID(_stable_id(RegionID, "static-class-frame", definition, source.fingerprint))
    scope = ControlScopeID(_stable_id(ControlScopeID, "static-class-scope", definition, source.fingerprint))
    world = LocationID(_stable_id(LocationID, "static-class-world", definition, source.fingerprint))
    entry_block = BlockID(_stable_id(BlockID, "static-class-entry", definition, source.fingerprint))
    normal_block = BlockID(_stable_id(BlockID, "static-class-normal", definition, source.fingerprint))
    raise_block = BlockID(_stable_id(BlockID, "static-class-raise", definition, source.fingerprint))
    body_operation = OperationID(_stable_id(OperationID, "required-class-body", definition, source.fingerprint))
    normal_exit = OperationID(_stable_id(OperationID, "class-body-normal-exit", definition, source.fingerprint))
    raise_exit = OperationID(_stable_id(OperationID, "class-body-raise-exit", definition, source.fingerprint))
    state_type = ValueType(ValueKind.STATE, location=world)
    # Both class-cell and implicit-None payloads are Python object references.
    # The overlay/compiler stages distinguish the exact return mode; this
    # port does not claim a general object identity or alias relationship.
    normal_payload_type = _REF

    port_names = ("locals_namespace", "globals_namespace", "builtins_namespace") + tuple(
        "closure:" + name for name in abi.co_freevars)
    reference_values = tuple(ValueID(_stable_id(ValueID, "class-entry-reference", definition,
                                                source.fingerprint, name, index))
                             for index, name in enumerate(port_names))
    entry_state = ValueID(_stable_id(ValueID, "class-entry-world", definition, source.fingerprint))
    mapping = MappingFramePorts(reference_values[0], reference_values[1], reference_values[2],
                                reference_values[3:], entry_state)
    abi = ClassFrameABI(mapping, abi.co_freevars, abi.co_cellvars, abi.compiler_cells)
    root = SourceCompilationFrameRoot(scope, FrameKind.CLASS, compilation, abi)

    normal_payload_value = ValueID(_stable_id(ValueID, "class-body-normal-payload", definition, source.fingerprint))
    normal_world = ValueID(_stable_id(ValueID, "class-body-normal-world", definition, source.fingerprint))
    raised_exception = ValueID(_stable_id(ValueID, "class-body-raised-exception", definition, source.fingerprint))
    raised_traceback = ValueID(_stable_id(ValueID, "class-body-raised-traceback", definition, source.fingerprint))
    raised_world = ValueID(_stable_id(ValueID, "class-body-raised-world", definition, source.fingerprint))

    values = []
    value_links = []
    for index, (identity, name) in enumerate(zip(reference_values, port_names)):
        value = ReferenceValue(identity, region, BlockArgument(entry_block, index))
        values.append(value)
        value_links.append(ValueSource(identity, source, SourceRole.PARAMETER))
    entry_state_index = len(reference_values)
    values.append(StateVersionValue(entry_state, region, BlockArgument(entry_block, entry_state_index), world))
    value_links.append(ValueSource(entry_state, source, SourceRole.STATE))
    normal_value = ReferenceValue(normal_payload_value, region,
                                  BlockArgument(normal_block, 0))
    values.extend((
        normal_value,
        StateVersionValue(normal_world, region, BlockArgument(normal_block, 1), world),
        ReferenceValue(raised_exception, region, BlockArgument(raise_block, 0)),
        ReferenceValue(raised_traceback, region, BlockArgument(raise_block, 1)),
        StateVersionValue(raised_world, region, BlockArgument(raise_block, 2), world),
    ))
    value_links.extend((ValueSource(normal_payload_value, source, SourceRole.RETURN),
                        ValueSource(normal_world, source, SourceRole.STATE),
                        ValueSource(raised_exception, source, SourceRole.EXCEPTION),
                        ValueSource(raised_traceback, source, SourceRole.EXCEPTION),
                        ValueSource(raised_world, source, SourceRole.STATE)))

    operations = []
    operation_links = []
    use_links = []

    def uses_for(operation, rows):
        uses = []
        for index, (value, role, source_role) in enumerate(rows):
            identity = UseID(_stable_id(UseID, "class-use", operation, index, role))
            value_record = next(row for row in values if row.id == value)
            use_role = (UseRole.REFERENCE if isinstance(value_record, ReferenceValue) else
                        UseRole.STATE_CONTENT if isinstance(value_record, StateVersionValue) else
                        UseRole.DATA)
            uses.append(Use(identity, value, use_role))
            use_links.append(UseSource(operation, identity, source, role, index))
        return tuple(uses)

    input_rows = [(identity, "class-entry:" + name, SourceRole.PARAMETER)
                  for identity, name in zip(reference_values, port_names)]
    input_rows.append((entry_state, "class-entry:world", SourceRole.STATE))
    body_uses = uses_for(body_operation, input_rows)
    normal_signature = CompletionSignature(ExitKind.NORMAL, (normal_payload_type, state_type))
    raise_signature = CompletionSignature(ExitKind.RAISE, (_REF, _REF, state_type))
    body = Operation(body_operation, "required-class-body-expansion",
        (Behavior.CONTROL, Behavior.READ, Behavior.WRITE, Behavior.MEMORY), body_uses,
        effects_complete=False,
        terminator=Terminator(TerminatorKind.INVOKE, (
            CompletionSuccessor(ExitKind.NORMAL, normal_block,
                                (CompletionResult(0), CompletionResult(1))),
            CompletionSuccessor(ExitKind.RAISE, raise_block,
                                (CompletionResult(0), CompletionResult(1), CompletionResult(2))),
        )), completions=(normal_signature, raise_signature),
        invoke_arguments=tuple(use.id for use in body_uses))
    operations.append(body)
    operation_links.append(OperationSource(body_operation, source, SourceRole.CONTROL,
                                           definition, Opcode.OPAQUE))

    normal_uses = uses_for(normal_exit, ((normal_payload_value, "class-body-normal-payload", SourceRole.RETURN),
                                         (normal_world, "class-body-normal-world", SourceRole.STATE)))
    operations.append(Operation(normal_exit, "class-body-normal-exit", (Behavior.CONTROL,), normal_uses,
        terminator=Terminator(TerminatorKind.RETURN, exit="normal",
                              values=tuple(use.id for use in normal_uses))))
    operation_links.append(OperationSource(normal_exit, source, SourceRole.RETURN))

    raise_uses = uses_for(raise_exit, ((raised_exception, "class-body-exception", SourceRole.EXCEPTION),
                                       (raised_traceback, "class-body-traceback", SourceRole.EXCEPTION),
                                       (raised_world, "class-body-raise-world", SourceRole.STATE)))
    operations.append(Operation(raise_exit, "class-body-raise-exit", (Behavior.CONTROL,), raise_uses,
        terminator=Terminator(TerminatorKind.RAISE, exit="raise",
                              values=tuple(use.id for use in raise_uses))))
    operation_links.append(OperationSource(raise_exit, source, SourceRole.EXCEPTION))

    blocks = (
        Block(entry_block, reference_values + (entry_state,), (body_operation,)),
        Block(normal_block, (normal_payload_value, normal_world), (normal_exit,)),
        Block(raise_block, (raised_exception, raised_traceback, raised_world), (raise_exit,)),
    )
    exits = (ExitPort("normal", ExitKind.NORMAL, (normal_payload_type, state_type)),
             ExitPort("raise", ExitKind.RAISE, (_REF, _REF, state_type)))
    region_record = Region(region, "required static class-body expansion", entry_block, blocks, exits)
    scope_record = ControlScope(scope, region, None, ControlScopeKind.FRAME, source,
                                (entry_block, normal_block, raise_block))
    graph = ProgramGraph(region, (region_record,), tuple(operations), tuple(values),
        (StateLocation(world, "coarse Python/external world", stability=Stability.UNKNOWN),),
        control_scopes=(scope_record,), frame_roots=(root,))

    gaps = tuple(LoweringGap(GapKind.EFFECT_SUMMARY, frontier.source or source,
                             frontier.kind.value + ": " + frontier.detail)
                 for frontier in frontiers)
    metadata = LoweringMetadata(definition.wire, snapshot,
        tuple(operation_links), tuple(value_links), gaps=gaps, use_links=tuple(use_links))
    return graph, metadata, root, region, body_operation


def lower_class_entry(prepared, definition, source, *, code_evidence, max_operations=10000):
    """Build one source-compiled class frame with an explicit required body.

    The output is a canonical graph fragment plus a source overlay. Header
    expressions and parent-class construction remain frontiers until a parent
    source body connects them.
    """
    if type(max_operations) is not int or not 3 <= max_operations <= 100000:
        raise ValueError("class-entry operation budget must be in [3,100000]")
    from scar.analysis.static_compilation import StaticCodeEvidence
    from scar.analysis.source_language import supports_cpython_compiler

    snapshot, semantic_definition, node = _resolve_class(prepared, definition, source)
    if type(code_evidence) is not StaticCodeEvidence:
        raise TypeError("static class entry requires StaticCodeEvidence, never CapturedCode")
    code_evidence.require_definition(snapshot, definition, source)
    if not supports_cpython_compiler(code_evidence):
        raise ValueError("static class ABI requires the pinned CPython 3.14.7 compiler profile")
    code = code_evidence.code
    if (code.co_name != node.name or code.co_argcount or code.co_posonlyargcount
            or code.co_kwonlyargcount or code.co_flags & (0x20 | 0x80 | 0x200)):
        raise ValueError("compiled class body has unexpected function/formal/resumable ABI")
    compilation = code_evidence.to_reference()
    stages, normal_payload = compiler_cell_protocol(code)
    compiler_cells = _compiler_cells(code)
    frame_mapping = MappingFramePorts(
        ValueID(_stable_id(ValueID, "class-entry-locals-placeholder", definition, source.fingerprint)),
        ValueID(_stable_id(ValueID, "class-entry-globals-placeholder", definition, source.fingerprint)),
        ValueID(_stable_id(ValueID, "class-entry-builtins-placeholder", definition, source.fingerprint)),
        tuple(ValueID(_stable_id(ValueID, "class-entry-freevar-placeholder", definition,
                                 source.fingerprint, index, name))
              for index, name in enumerate(code.co_freevars)),
        ValueID(_stable_id(ValueID, "class-entry-world-placeholder", definition, source.fingerprint)))
    abi = ClassFrameABI(frame_mapping, tuple(code.co_freevars), tuple(code.co_cellvars), compiler_cells)
    inputs = _class_header_inputs(prepared, node)
    statements = _class_scope_statements(prepared, node)
    frontiers = _frontiers(source, has_freevars=bool(code.co_freevars),
                           has_class_cell="__class__" in code.co_cellvars,
                           has_decorators=bool(node.decorator_list))
    graph, metadata, root, region, operation = _build_graph(
        definition, source, snapshot, compilation, abi, normal_payload, statements, frontiers)
    # _build_graph creates deterministic entry ValueIDs and the matching final ABI.
    abi = root.entry
    required = RequiredClassBodyExpansion(region, operation, source, statements, normal_payload, stages)
    overlay = ClassEntryOverlay(definition, semantic_definition.parent_id, source, compilation,
        root.scope, abi, inputs, required, frontiers, len(graph.operations))
    result = ClassEntryResult(graph, metadata, overlay, len(graph.operations))
    from scar.analysis.source_class_lowering import require_valid_source_class_entry
    require_valid_source_class_entry(prepared, definition, source, code_evidence=code_evidence,
                                     result=result, max_operations=max_operations)
    return result


# Independent source and canonical-core validation for static class frames.
def _need(condition, detail):
    if not condition:
        raise ValueError("static class entry: " + detail)




def _source_reference(prepared, node):
    span = _span(node)
    snapshot = prepared.semantics.sources[str(prepared.path)]
    matches = [atom.reference for atom in prepared.semantic.source_atoms.values()
               if atom.kind == type(node).__name__ and atom.reference.path == str(prepared.path)
               and atom.reference.fingerprint == snapshot.fingerprint
               and (atom.reference.start_line, atom.reference.end_line,
                    atom.reference.start_column, atom.reference.end_column) == span]
    if len(matches) != 1:
        raise ValueError("static class entry: source span has no unique retained source atom")
    return matches[0]


def _expected_header_inputs(prepared, node):
    rows, order = [], 0
    for index, expression in enumerate(node.decorator_list):
        rows.append(ClassEntryInput(ClassEntryInputRole.DECORATOR, index, order,
                                    _source_reference(prepared, expression)))
        order += 1
    for index, expression in enumerate(node.bases):
        rows.append(ClassEntryInput(ClassEntryInputRole.BASE, index, order,
                                    _source_reference(prepared, expression)))
        order += 1
    for index, keyword in enumerate(node.keywords):
        role = (ClassEntryInputRole.METACLASS if keyword.arg == "metaclass" else
                ClassEntryInputRole.CLASS_KEYWORD if keyword.arg is not None else
                ClassEntryInputRole.CLASS_KEYWORD_EXPANSION)
        rows.append(ClassEntryInput(role, index, order,
                                    _source_reference(prepared, keyword.value), keyword.arg))
        order += 1
    return tuple(rows)


def _expected_body_statements(prepared, node):
    pending, rows = list(reversed(node.body)), []
    while pending:
        current = pending.pop()
        if isinstance(current, ast.stmt):
            rows.append(current)
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        pending.extend(reversed(list(ast.iter_child_nodes(current))))
    return tuple(ClassBodyStatement(index, type(statement).__name__,
                                    _source_reference(prepared, statement))
                 for index, statement in enumerate(rows))


def _source_semantics(prepared, definition, source):
    _need(type(definition) is OperationDefinitionID and type(source) is SourceReference,
          "validator needs the original definition and source reference")
    source.__post_init__()
    snapshot = prepared.semantics.sources.get(source.path)
    _need(type(snapshot) is SourceSnapshot and source.path == str(prepared.path)
          and source.fingerprint == snapshot.fingerprint,
          "class source differs from the retained snapshot")
    atom = prepared.semantic.source_atoms.get(source.atom_id)
    semantic = prepared.semantic.definitions.get(definition)
    _need(atom is not None and atom.reference == source and atom.kind == "ClassDef"
          and semantic is not None and semantic.kind is OperationKind.OPAQUE
          and semantic.code_id == source.atom_id.wire and source.atom_id in semantic.source_atoms
          and semantic.source_file == source.path,
          "ID is not the original SourceGraph opaque ClassDef operation")
    nodes = [node for node in ast.walk(prepared.tree)
             if type(node) is ast.ClassDef and _span(node) == (
                 source.start_line, source.end_line, source.start_column, source.end_column)]
    _need(len(nodes) == 1 and semantic.label == "class " + nodes[0].name,
          "source span does not select the original ClassDef")
    return snapshot, semantic, nodes[0]


def require_valid_source_class_entry(prepared, definition, source, *, code_evidence, result,
                                    max_operations=10000):
    """Recompile retained source and validate the class root plus required body.

    This validates only the source-compiled static fragment. It makes no claim
    that a parent has supplied a prepared namespace/closure or invoked the
    metaclass, and it does not certify effects or class-object identity.
    """
    _need(type(result) is ClassEntryResult, "result must be exact ClassEntryResult")
    _need(type(max_operations) is int and 3 <= max_operations <= 100000,
          "operation limit must be in [3,100000]")
    snapshot, semantic, node = _source_semantics(prepared, definition, source)
    _need(type(code_evidence) is StaticCodeEvidence,
          "class validation requires StaticCodeEvidence, not a capture witness")
    code_evidence.require_definition(snapshot, definition, source)

    independently_compiled = compile_source_unit(prepared).for_definition(definition, source)
    _need(independently_compiled.to_reference() == code_evidence.to_reference(),
          "supplied class code differs from a fresh no-exec compilation of the retained source")
    code = independently_compiled.code
    _need(code.co_name == node.name and code.co_argcount == 0
          and code.co_posonlyargcount == 0 and code.co_kwonlyargcount == 0,
          "compiler class-body name/formal layout differs")
    stages, normal_payload = compiler_cell_protocol(code)

    graph, metadata, overlay = result.graph, result.metadata, result.overlay
    _need(metadata.source == snapshot and overlay.source == source
          and overlay.definition == definition and overlay.parent_definition == semantic.parent_id,
          "overlay does not preserve original source, definition, lexical parent and snapshot")
    _need(overlay.compilation == independently_compiled.to_reference(),
          "overlay compiler reference differs from independent compilation")
    _need(overlay.inputs == _expected_header_inputs(prepared, node),
          "class decorators/bases/keywords are incomplete, reordered or mis-sourced")
    expected_statements = _expected_body_statements(prepared, node)
    _need(overlay.required_expansion.body_statements == expected_statements,
          "executed class-scope statement collection is incomplete or misordered")
    _need(overlay.required_expansion.normal_payload is normal_payload
          and overlay.required_expansion.compiler_stages == stages,
          "class normal return/cell protocol differs from the exact compiled CodeType")

    expected_cell_roles = tuple((name, index, {
        "__class__": ClassCompilerCellRole.CLASS_OBJECT,
        "__classdict__": ClassCompilerCellRole.CLASS_NAMESPACE,
    }.get(name, ClassCompilerCellRole.LOCAL)) for index, name in enumerate(code.co_cellvars))
    abi = overlay.interface
    _need(type(abi) is ClassFrameABI and abi.co_freevars == tuple(code.co_freevars)
          and abi.co_cellvars == tuple(code.co_cellvars)
          and tuple((cell.name, cell.position, cell.role) for cell in abi.compiler_cells)
              == expected_cell_roles
          and len(abi.mapping.closure_cells) == len(code.co_freevars),
          "class freevar entry or compiler-owned local-cell roles differ from compilation")
    _need(overlay.root_scope in {scope.id for scope in graph.control_scopes},
          "class frame root scope is missing")
    roots = [root for root in graph.frame_roots if root.scope == overlay.root_scope]
    _need(len(roots) == 1 and type(roots[0]) is SourceCompilationFrameRoot
          and roots[0].kind is FrameKind.CLASS and roots[0].entry == abi
          and roots[0].code == independently_compiled.to_reference(),
          "canonical root is not the original source-compiled CLASS frame ABI")
    scope = next(row for row in graph.control_scopes if row.id == overlay.root_scope)
    _need(scope.kind is ControlScopeKind.FRAME and scope.source == source
          and scope.region == graph.root and roots[0].code.source == source,
          "class FRAME scope/source/region does not retain the original ClassDef")

    require_valid(graph)
    _need(not control_errors(graph), "class frame root violates canonical control validation")
    _need(result.operations_emitted == overlay.operations_emitted == len(graph.operations)
          and len(graph.operations) <= max_operations and len(graph.operations) == 3,
          "required-expansion graph operation count differs from its bounded fragment")
    _need(not graph.environments,
          "class frame must not be represented by a function EnvironmentInterface")
    operations = {operation.id: operation for operation in graph.operations}
    body = operations.get(overlay.required_expansion.operation)
    world_type = ValueType(ValueKind.STATE, location=graph.locations[0].id)
    normal_payload_type = ValueType(ValueKind.REFERENCE)
    _need(type(body) is Operation and body.label == "required-class-body-expansion"
          and body.terminator is not None and body.terminator.kind is TerminatorKind.INVOKE
          and body.callee is None and body.effects_complete is False
          and set(body.behaviors) == {Behavior.CONTROL, Behavior.READ, Behavior.WRITE, Behavior.MEMORY}
          and body.completions == (
              CompletionSignature(ExitKind.NORMAL, (normal_payload_type, world_type)),
              CompletionSignature(ExitKind.RAISE, (ValueType(ValueKind.REFERENCE),
                  ValueType(ValueKind.REFERENCE), world_type))),
          "class body is not retained as an incomplete normal/raise expansion boundary")
    normal_block = next(edge.block for edge in body.terminator.successors if edge.kind is ExitKind.NORMAL)
    normal_block_record = next(block for region in graph.regions for block in region.blocks
                               if block.id == normal_block)
    normal_payload_id = normal_block_record.arguments[0]
    normal_payload_record = next(value for value in graph.values if value.id == normal_payload_id)
    if normal_payload is ClassNormalPayloadKind.CLASS_CELL:
        _need(type(normal_payload_record) is ReferenceValue
              and normal_payload_record.origin == BlockArgument(normal_block, 0),
              "compiled __class__ cell result is not modeled as a reference payload")
        _need(any(stage.role == "RETURN_CLASS_OBJECT_CELL"
                  for stage in overlay.required_expansion.compiler_stages),
              "class-cell reference lacks the compiler-verified return mode")
    else:
        _need(type(normal_payload_record) is ReferenceValue
              and normal_payload_record.origin == BlockArgument(normal_block, 0),
              "implicit None payload is not connected as a Python object reference")
        roles = tuple(stage.role for stage in overlay.required_expansion.compiler_stages)
        _need("LOAD_NONE_NORMAL_RESULT" in roles and "RETURN_NONE" in roles,
              "None reference lacks the compiler-verified singleton return constraint")
    operation_sources = {row.operation: row for row in metadata.operation_links}
    body_source = operation_sources.get(body.id)
    _need(type(body_source) is OperationSource and body_source.source == source
          and body_source.semantic_operation == definition and body_source.opcode is Opcode.OPAQUE,
          "required body boundary is not linked to the existing original class operation")
    expected_frontiers = {
        ClassFrontierKind.BODY_REQUIRED_EXPANSION,
        ClassFrontierKind.PARENT_CLASS_STATEMENT_UNCONNECTED,
        ClassFrontierKind.PREPARED_NAMESPACE_UNCONNECTED,
        ClassFrontierKind.METACLASS_SELECTION_AND_CALL_OPEN,
        ClassFrontierKind.BASE_MRO_ENTRIES_OPEN,
        ClassFrontierKind.DESCRIPTOR_AND_SUBCLASS_HOOKS_OPEN,
        ClassFrontierKind.EFFECTS_OPEN,
    }
    frontier_kinds = {frontier.kind for frontier in overlay.frontiers}
    _need(expected_frontiers <= frontier_kinds,
          "class construction or body incompleteness is not explicitly retained")
    if code.co_freevars:
        _need(ClassFrontierKind.OUTER_CLOSURE_UNCONNECTED in frontier_kinds,
              "compiler freevars lack a parent-closure frontier")
    if "__class__" in code.co_cellvars:
        _need(ClassFrontierKind.CLASSCELL_ACCEPTANCE_AND_FILL_UNPROVEN in frontier_kinds,
              "normal classcell output lacks an unproved parent-fill frontier")
    if node.decorator_list:
        _need(ClassFrontierKind.DECORATOR_APPLICATION_OPEN in frontier_kinds,
              "decorator application is not left open")
    _need(overlay.status == "STATIC_CLASS_FRAME_WITH_REQUIRED_BODY_EXPANSION"
          and overlay.required_expansion.status == "REQUIRED",
          "class lowerer may not report a complete body")
    return result


__all__ = [
    'lower_class_entry',
    'require_valid_source_class_entry',
]
