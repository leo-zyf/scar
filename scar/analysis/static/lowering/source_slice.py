"""Source slice."""

from __future__ import annotations

import ast
import dis
from dataclasses import replace
from scar.analysis.program_lowering import _REF, _Unsupported
from scar.analysis.static.lowering.source_heap_lowering import (
    _hold,
    _source_definition,
    _associate,
    expression_for_heap_target,
)
from scar.analysis.static.lowering.source_mapping_lowering import _transfer
from scar.ir.program.model import (
    AccessMode,
    Behavior,
    ExitKind,
    Operation,
    OperationResult,
    ReferenceValue,
    StateAccess,
    CompletionResult,
    LiteralOrigin,
    StateVersionValue,
    TerminatorKind,
    ValueKind,
    ValueType,
    value_type,
)
from scar.ir.program.ids import OperationID, ValueID
from scar.ir.program.source import GapKind, SourceRole
from scar.ir.program.source_language import (
    CodeConstantSource,
    SliceShape,
    SourceLanguageCapability as C,
    SLICE_ACCESS_PROTOCOLS,
)
from scar.ir.program.structured import PythonProtocolKind, PythonProtocolKind as K
from scar.ir.semantics_v2 import Opcode
from scar.ir.v2.semantic import OperationKind
from scar.ir.program.state_addresses import (
    AddressEffect,
    Element,
    ReferenceOwner,
    Slice,
    AddressCoverage,
    InvokeStateOperand,
)
from collections import deque
from scar.analysis.source_language import (
    function_exception_types,
    require_source_tree,
    supports_cpython_compiler,
)


# Slice/key construction without a global 'slice' lookup or allocation guess.
def _constant_slice(node):
    return isinstance(node, ast.Slice) and all(child is None or isinstance(child, ast.Constant)
                                               for child in (node.lower, node.upper, node.step))


def _span(node):
    return node.lineno, node.end_lineno, node.col_offset, node.end_col_offset


def _instructions(builder, node):
    evidence = builder.language_options.code_evidence
    if evidence is None:
        raise _Unsupported(GapKind.CODE_VERSION, node, "SLICE_CODE_WITNESS_REQUIRED")
    if not hasattr(builder, "language_instructions"):
        builder.language_instructions = tuple(dis.get_instructions(evidence.code))
    return tuple(instruction for instruction in builder.language_instructions
                 if tuple(instruction.positions) == _span(node))


def _constant(builder, node, cursor):
    matches = [row for row in _instructions(builder, node) if row.opname == "LOAD_CONST"
               and type(row.arg) is int and type(row.argval) in (slice, tuple)]
    if len(matches) != 1:
        raise _Unsupported(GapKind.CODE_VERSION, node,
                           "source key has no unique exact LOAD_CONST position in its CodeWitness")
    instruction, = matches
    evidence = builder.language_options.code_evidence
    if evidence.code.co_consts[instruction.arg] is not instruction.argval:
        raise _Unsupported(GapKind.CODE_VERSION, node, "code constant slot differs from disassembly")
    operation_id = builder.fresh(OperationID, "code-constant-reference")
    before = builder.use(operation_id, cursor.state, node, "code-constant-world", 0)
    result = ReferenceValue(builder.fresh(ValueID, "code-constant-reference"), builder.function_context,
                            OperationResult(operation_id, 0))
    builder.add_value(result); builder.link_value(result, node, SourceRole.EXPRESSION)
    operation = Operation(operation_id, "code constant reference", (Behavior.READ,), (before,),
        (result.id,), (StateAccess(builder.world, before.id, AccessMode.READ),), effects_complete=False)
    builder.emit(cursor, operation, node, SourceRole.EXPRESSION)
    builder.protocol(operation_id, PythonProtocolKind.READ_CODE_CONSTANT, node)
    builder.language_constants.append(CodeConstantSource(operation_id, builder.source(node),
        evidence.code_layout_key, (instruction.arg,)))
    builder.gap(GapKind.CODE_VERSION, node,
        "code constant slot is witnessed; active CodeType/constant object instance identity remains unproved")
    return cursor, result.id


def _invoke(builder, cursor, node, inputs, normal_consumed, *, raise_consumed=None,
            old=None, kind=None, memory=False, result_type=_REF, role=SourceRole.EXPRESSION):
    # Most existing slice/index primitives consume the same input holds in both
    # completions. BUILD_TUPLE steals only on success (CPython 3.14.7). Keeping
    # the exceptional contract explicit avoids a second release before unwind.
    if raise_consumed is None:
        raise_consumed = normal_consumed
    def failed(failure, exception):
        pending = tuple(builder.active_temporaries)
        try:
            _transfer(builder, raise_consumed, node)
            builder.finish_exit(failure, node, exception, ExitKind.RAISE)
        finally:
            builder.active_temporaries[:] = pending
    cursor, value, operation = builder.invoke(cursor, node, tuple(inputs), old=old,
        failure_handler=failed, result_type=result_type, role=role)
    if kind is not None:
        builder.protocol(operation, kind, node)
    if memory:
        original = builder.operation_records[operation]
        marked = replace(original, behaviors=original.behaviors + (Behavior.MEMORY,))
        builder.operations[builder.operations.index(original)] = marked
        builder.operation_records[operation] = marked
    _transfer(builder, normal_consumed, node)
    return cursor, value, operation


def lower_slice(builder, node, cursor):
    if not isinstance(node, ast.Slice):
        raise _Unsupported(GapKind.SOURCE_MAPPING, node, "expected exact Slice source")
    if _constant_slice(node):
        return _constant(builder, node, cursor)
    instructions = _instructions(builder, node)
    count = 3 if node.step is not None else 2
    if not any(row.opname == "BUILD_SLICE" and row.arg == count for row in instructions):
        raise _Unsupported(GapKind.CODE_VERSION, node, "COMBINED_SLICE_LIFETIME_UNRECOVERED")
    baseline = len(builder.active_temporaries)
    held, inputs = [], []
    try:
        for index, (role, child) in enumerate((('slice-lower', node.lower), ('slice-upper', node.upper),
                                              ('slice-step', node.step))[:count]):
            if child is None:
                cursor, value = cursor, builder.literal(node, None)
            elif builder.has_language('SLICE_ACCESS_314_V1'):
                cursor, value = expression_for_heap_target(builder, child, cursor)
            else:
                cursor, value = builder.expr(child, cursor)
            if cursor is None:
                return None, None
            origin = child or node
            _hold(builder, cursor, origin, value)
            held.append((origin, value))
            inputs.append((value, origin, role, index, None))
        cursor, value, operation = _invoke(builder, cursor, node, inputs, tuple(held),
            kind=PythonProtocolKind.BUILD_SLICE, memory=True)
        builder.language_slices.append(SliceShape(operation, builder.source(node),
            builder.source(node.lower) if node.lower is not None else None,
            builder.source(node.upper) if node.upper is not None else None,
            builder.source(node.step) if node.step is not None else None))
        builder.gap(GapKind.EFFECT_SUMMARY, node,
            "slice syntax bounds/order are connected; allocation, identity, lifetime and consumer dispatch remain unproved")
        return cursor, value
    finally:
        del builder.active_temporaries[baseline:]


def lower_key(builder, node, cursor):
    if isinstance(node, ast.Slice):
        return lower_slice(builder, node, cursor)
    if not isinstance(node, ast.Tuple):
        return (expression_for_heap_target(builder, node, cursor)
                if builder.has_language('SLICE_ACCESS_314_V1') else builder.expr(node, cursor))
    if any(isinstance(child, ast.Starred) for child in node.elts):
        raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX, node, "starred index key requires a separate construction rule")
    instructions = _instructions(builder, node)
    if any(row.opname == 'LOAD_CONST' and type(row.argval) is tuple for row in instructions):
        return _constant(builder, node, cursor)
    if sum(row.opname == 'BUILD_TUPLE' and row.arg == len(node.elts) for row in instructions) != 1:
        raise _Unsupported(GapKind.CODE_VERSION, node, "INDEX_TUPLE_CODE_LOWERING_UNRECOVERED")
    records = builder.old_by_atom.get(builder.source(node).atom_id, ())
    if len(records) > 1:
        raise _Unsupported(GapKind.SOURCE_MAPPING, node,
                           "index tuple has ambiguous computational source records")
    old, definition = (records[0], None) if records else (None, _source_definition(builder, node))
    if old is not None:
        if old.opcode is not Opcode.BUILD_TUPLE or len(old.operands) != len(node.elts):
            raise _Unsupported(GapKind.SOURCE_MAPPING, node,
                               "index tuple operands differ from original source semantics")
        operands = tuple((row.role, row.index, row.value) for row in old.operands)
    else:
        if (definition.kind is not OperationKind.OPERATOR or definition.metadata.get("ast_type") != "Tuple"
                or len(definition.output_slots) != 1):
            raise _Unsupported(GapKind.SOURCE_MAPPING, node,
                               "index tuple requires its unique original SG Tuple/output slot")
        operands = tuple(("element", index, None) for index in range(len(node.elts)))
    baseline, held, inputs = len(builder.active_temporaries), [], []
    try:
        for child, (role, index, static) in zip(node.elts, operands):
            cursor, value = expression_for_heap_target(builder, child, cursor)
            if cursor is None:
                return None, None
            _hold(builder, cursor, child, value); held.append((child, value))
            inputs.append((value, child, role, index, static))
        cursor, value, operation = _invoke(builder, cursor, node, inputs, tuple(held),
            raise_consumed=(), old=old, memory=True)
        if definition is not None:
            _associate(builder, operation, node, definition, Opcode.BUILD_TUPLE)
        builder.gap(GapKind.EFFECT_SUMMARY, node,
            "tuple key inputs transfer only on normal completion; modeled pending references are checked separately; "
            "runtime code instance, finalizers, allocation and complete lifetime effects remain unproved")
        return cursor, value
    finally:
        del builder.active_temporaries[baseline:]


def lower_subscript(builder, node, cursor, *, source_expression=False):
    if not isinstance(node, ast.Subscript) or not isinstance(node.ctx, ast.Load):
        raise _Unsupported(GapKind.SOURCE_MAPPING, node, "subscript read requires Load context")
    expanded = builder.has_language("SLICE_ACCESS_314_V1")
    if source_expression and (not builder.has_language('SOURCE_REFERENCE_LIFETIME_314_V1')
            or isinstance(node.slice, (ast.Slice, ast.Tuple))):
        raise _Unsupported(GapKind.SOURCE_MAPPING, node,
            'source index lifetime requires its explicit ordinary-key capability')
    if (expanded and isinstance(node.slice, ast.Slice)
            and any(row.opname == 'BINARY_SLICE' for row in _instructions(builder, node))):
        return _combined_read(builder, node, cursor)
    if (not expanded and isinstance(node.slice, ast.Slice) and not _constant_slice(node.slice)
            and node.slice.step is None):
        raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX, node, "COMBINED_SLICE_LIFETIME_UNRECOVERED")
    old, definition = _index_source(builder, node) if expanded or source_expression else (builder.old_operation(node), None)
    if old is not None and len(old.operands) != 2:
        raise _Unsupported(GapKind.SOURCE_MAPPING, node, "subscript source operands differ")
    baseline, held, inputs = len(builder.active_temporaries), [], []
    try:
        cursor, receiver = expression_for_heap_target(builder, node.value, cursor) if expanded or source_expression else builder.expr(node.value, cursor)
        if cursor is None:
            return None, None
        _hold(builder, cursor, node.value, receiver); held.append((node.value, receiver))
        cursor, key = (expression_for_heap_target(builder, node.slice, cursor) if source_expression
                       else lower_key(builder, node.slice, cursor))
        if cursor is None:
            return None, None
        _hold(builder, cursor, node.slice, key); held.append((node.slice, key))
        for index, (value, child, role) in enumerate(((receiver, node.value, 'receiver'), (key, node.slice, 'index'))):
            operand = old.operands[index] if old is not None else None
            inputs.append((value, child, operand.role if operand else role,
                           operand.index if operand else index, operand.value if operand else None))
        cursor, result, operation = _invoke(builder, cursor, node, inputs, tuple(held), old=old,
            raise_consumed=() if source_expression else None,
            kind=PythonProtocolKind.SOURCE_INDEX if source_expression else None)
        if definition is not None:
            _associate(builder, operation, node, definition, Opcode.INDEX)
        uses = builder.operation_records[operation].uses
        builder._addresses(operation, (AddressEffect(ReferenceOwner(uses[0].id),
            (Element(uses[1].id),), AccessMode.READ),))
        builder.gap(GapKind.DYNAMIC_DISPATCH, node,
            "subscription key/order are connected; consumer dispatch, mutation, aliasing and resources remain unproved")
        return cursor, result
    finally:
        del builder.active_temporaries[baseline:]


def _index_source(builder, node):
    """Original SG ownership, with an optional existing computational overlay."""
    records = builder.old_by_atom.get(builder.source(node).atom_id, ())
    if len(records) > 1:
        raise _Unsupported(GapKind.SOURCE_MAPPING, node, 'slice index has ambiguous computational records')
    definition = _source_definition(builder, node)
    if definition.kind is not OperationKind.INDEX or len(definition.output_slots) != 1:
        raise _Unsupported(GapKind.SOURCE_MAPPING, node, 'slice read requires its exact original SG INDEX/output')
    if records:
        old, = records
        if old.operation != definition.id or old.opcode is not Opcode.INDEX or len(old.operands) != 2:
            raise _Unsupported(GapKind.SOURCE_MAPPING, node, 'slice read differs from original SG index operands')
        return old, None
    return None, definition


def _bound_inputs(builder, node, cursor, held):
    """Evaluate only existing lower/upper children; omissions are exact None."""
    inputs = []
    for index, (role, child) in enumerate((('slice-lower', node.slice.lower), ('slice-upper', node.slice.upper))):
        origin = child or node.slice
        cursor, value = (expression_for_heap_target(builder, child, cursor) if child is not None
                         else (cursor, builder.literal(node.slice, None)))
        if cursor is None:
            return None, None
        _hold(builder, cursor, origin, value); held.append((origin, value))
        # A bound is not the old whole-key StaticValue. Its source and core Use
        # suffice until a real SG value for this exact child is available.
        inputs.append((value, origin, role, index, None))
    return cursor, inputs


def _combined_read(builder, node, cursor):
    if node.slice.step is not None or sum(row.opname == 'BINARY_SLICE' for row in _instructions(builder, node)) != 1:
        raise _Unsupported(GapKind.CODE_VERSION, node, 'slice read has no unique exact BINARY_SLICE witness')
    old, definition = _index_source(builder, node)
    baseline, held = len(builder.active_temporaries), []
    try:
        cursor, receiver = expression_for_heap_target(builder, node.value, cursor)
        if cursor is None:
            return None, None
        _hold(builder, cursor, node.value, receiver); held.append((node.value, receiver))
        cursor, bounds = _bound_inputs(builder, node, cursor, held)
        if cursor is None:
            return None, None
        inputs = [(receiver, node.value, 'receiver', 0, old.operands[0].value if old is not None else None), *bounds]
        cursor, result, operation = _invoke(builder, cursor, node, inputs, tuple(held),
            old=old, kind=PythonProtocolKind.BINARY_SLICE)
        if definition is not None:
            _associate(builder, operation, node, definition, Opcode.INDEX)
        uses = builder.operation_records[operation].uses
        builder._addresses(operation, (AddressEffect(ReferenceOwner(uses[0].id),
            (Slice(uses[1].id, uses[2].id),), AccessMode.READ),))
        builder.link_value(builder.values_by_id[result], node, SourceRole.EXPRESSION,
                           semantic_value=old.result if old is not None else None)
        builder.gap(GapKind.DYNAMIC_DISPATCH, node,
            'compound slice receiver/bounds/world and consuming completions are connected; '
            'internal temporary slice, dispatch, aliasing and complete effects remain unproved')
        return cursor, result
    finally:
        del builder.active_temporaries[baseline:]


def lower_slice_store(builder, node, rhs, cursor, *, rhs_held=False):
    """CPython slice assignment: RHS, receiver, bounds/key, consuming store."""
    if not isinstance(node, ast.Subscript) or not isinstance(node.ctx, ast.Store):
        raise _Unsupported(GapKind.SOURCE_MAPPING, node, 'slice store requires original Subscript Store')
    combined = isinstance(node.slice, ast.Slice) and any(row.opname == 'STORE_SLICE' for row in _instructions(builder, node))
    expected = 'STORE_SLICE' if combined else 'STORE_SUBSCR'
    if sum(row.opname == expected for row in _instructions(builder, node)) != 1:
        raise _Unsupported(GapKind.CODE_VERSION, node, 'slice assignment has no unique exact consuming compiler operation')
    if combined and node.slice.step is not None:
        raise _Unsupported(GapKind.CODE_VERSION, node, 'STORE_SLICE cannot have a step')
    definition = _source_definition(builder, node, binding=True)
    if rhs_held and (not builder.active_temporaries or builder.active_temporaries[-1] != rhs):
        raise _Unsupported(GapKind.SOURCE_MAPPING, node, 'slice store lost original pending unpack result')
    baseline = len(builder.active_temporaries) - int(rhs_held)
    held = [(node, rhs)]
    try:
        if not rhs_held:
            _hold(builder, cursor, node, rhs)
        cursor, receiver = expression_for_heap_target(builder, node.value, cursor)
        if cursor is None:
            return None
        _hold(builder, cursor, node.value, receiver); held.append((node.value, receiver))
        inputs = [(rhs, node, 'assigned-reference', 0, None), (receiver, node.value, 'heap-owner', 0, None)]
        if combined:
            cursor, bounds = _bound_inputs(builder, node, cursor, held)
            if cursor is None:
                return None
            inputs.extend(bounds)
            kind = PythonProtocolKind.STORE_SLICE
        else:
            cursor, key = lower_key(builder, node.slice, cursor)
            if cursor is None:
                return None
            _hold(builder, cursor, node.slice, key); held.append((node.slice, key))
            inputs.append((key, node.slice, 'element-key', 0, None))
            kind = PythonProtocolKind.STORE_SLICE_SUBSCRIPT
        cursor, _, operation = _invoke(builder, cursor, node, inputs, tuple(held), kind=kind,
            result_type=None, role=SourceRole.LOCAL_BINDING)
        _associate(builder, operation, node, definition, Opcode.OPAQUE)
        uses = builder.operation_records[operation].uses
        selector = Slice(uses[2].id, uses[3].id) if combined else Element(uses[2].id)
        builder._addresses(operation, (AddressEffect(ReferenceOwner(uses[1].id), (selector,), AccessMode.WRITE),))
        builder.gap(GapKind.DYNAMIC_DISPATCH, node,
            'slice assignment retains RHS-first evaluation, original receiver/bounds/key Uses, '
            'world and consuming normal/raise outcomes; runtime dispatch and complete effects remain unproved')
        return cursor
    finally:
        del builder.active_temporaries[baseline:]


# Exact source/compiler ports and local reference transfer for slice access.
#
# The combined bytecodes are compound operations. Their internal temporary slice
# is not a source value. The checks certify the modeled pending references, not
# runtime object identity, complete finalizer effects or physical allocations.
def _need(condition, detail):
    if not condition:
        raise ValueError('slice access: ' + detail)


def _slice_reference_span(row):
    return row.start_line, row.end_line, row.start_column, row.end_column


def _raw_handoff_holds(failed, operation, operations, protocols, values, uses):
    """Read the original raw exception corridor, including explicit decrefs.

    String primitives release their consumed inputs before FrameExit. Preserve
    those hold instances when reconstructing the state at the INVOKE's RAISE
    landing; an arbitrary effectful instruction is never a cleanup shortcut.
    """
    _need(len(failed.arguments) == 2, 'raw raise must retain exception/world landing ports')
    exception, world = failed.arguments
    _need(type(values[exception]) is ReferenceValue and type(values[world]) is StateVersionValue
          and values[world].region == values[operation.uses[-1].value].region,
          'raw exception corridor crossed a different frame')
    released = []
    for offset, identity in enumerate(failed.operations):
        cleanup, marker = operations[identity], protocols.get(identity)
        _need(marker is not None and cleanup.terminator is None and cleanup.callee is None
              and len(cleanup.results) == 1 and cleanup.uses[-1].value == world
              and len(cleanup.accesses) == 1 and cleanup.accesses[0].before == cleanup.uses[-1].id
              and cleanup.accesses[0].after == cleanup.results[0]
              and cleanup.accesses[0].mode is AccessMode.READ_WRITE
              and value_type(values[cleanup.results[0]]) == value_type(values[world]),
              'raw cleanup must thread the exact post-world through its original frame operation')
        world = cleanup.results[0]
        if marker.kind is K.FRAME_RELEASE_TEMPORARY:
            first = uses.get((identity, cleanup.uses[0].id))
            _need(len(cleanup.uses) == 2 and type(values[cleanup.uses[0].value]) is ReferenceValue
                  and first is not None and first.role == 'fstring-consumed-reference'
                  and protocols.get(operation.id) is not None
                  and protocols[operation.id].kind in (K.FSTRING_CONVERT, K.FSTRING_FORMAT, K.BUILD_STRING),
                  'raw prefix is not an original string-input reference release')
            released.append(cleanup.uses[0].value)
            continue
        _need(marker.kind is K.FRAME_EXIT, 'raw raise must reach its original FrameExit')
        payload = [use.value for use in cleanup.uses
                   if (row := uses.get((identity, use.id))) is not None and row.role == 'held-payload']
        _need(payload == [exception] and offset + 2 == len(failed.operations),
              'FrameExit must hold this exact exception and precede its sole terminal raise')
        terminal = operations[failed.operations[offset + 1]]
        _need(terminal.terminator is not None and terminal.terminator.kind is TerminatorKind.RAISE
              and not terminal.terminator.successors
              and tuple(use.value for use in terminal.uses) == (exception, world)
              and terminal.terminator.values == tuple(use.id for use in terminal.uses),
              'raw terminal lost the original exception or cleanup post-world')
        retained = tuple(use.value for use in cleanup.uses
            if (row := uses.get((identity, use.id))) is not None and row.role == 'held-temporary')
        return retained + tuple(reversed(released))
    raise ValueError('slice access: raw raise has no frame cleanup endpoint')


def validate_slice_access(model, graph, interface, metadata, semantic, language, *,
                          source_tree=None, options=None, structural_only=False):
    protocols = {row.operation: row for row in model.protocols}
    selected = {identity for identity, row in protocols.items() if row.kind.value in SLICE_ACCESS_PROTOCOLS}
    if not selected:
        return set()
    _need(C.SLICE_ACCESS_314_V1 in language.capabilities and C.SLICE_KEYS in language.capabilities,
          'compound/store protocols require their distinct capability')
    operations = {row.id: row for row in graph.operations}
    values = {row.id: row for row in graph.values}
    blocks = {row.id: row for region in graph.regions for row in region.blocks}
    links = {row.operation: row for row in metadata.operation_links}
    uses = {(row.operation, row.use): row for row in metadata.use_links}
    addresses = {row.site.operation: row for row in graph.state_address_refinements}
    reference_type, state_type = ValueType(ValueKind.REFERENCE), value_type(values[interface.world])
    instructions = ()
    if not structural_only:
        _need(options is not None and supports_cpython_compiler(options.code_evidence),
              'independent CPython 3.14.7 compiler evidence is required')
        options.code_evidence.require_definition(metadata.source, interface.definition, model.source)
        instructions = tuple(dis.get_instructions(options.code_evidence.code))
    for identity in selected:
        operation, protocol, link = operations[identity], protocols[identity], links[identity]
        read, compound = protocol.kind is K.BINARY_SLICE, protocol.kind is not K.STORE_SLICE_SUBSCRIPT
        count = 3 if read or not compound else 4
        definition = semantic.definitions.get(link.semantic_operation)
        atom = semantic.source_atoms.get(link.source.atom_id)
        _need(definition is not None and definition.code_id == link.source.atom_id.wire
              and link.source.atom_id in definition.source_atoms and atom is not None
              and atom.kind == 'Subscript' and atom.reference == link.source == protocol.source,
              'operation lost its original SG/source Subscript')
        _need((definition.kind is OperationKind.INDEX and len(definition.output_slots) == 1
               and link.opcode is Opcode.INDEX) if read else
              (definition.metadata.get('role') == 'binding' and link.opcode is Opcode.OPAQUE),
              'read/store no longer has its original SG role')
        _need(operation.terminator is not None and operation.terminator.kind is TerminatorKind.INVOKE
              and operation.callee is None and not operation.effects_complete
              and len(operation.uses) == count + 1
              and tuple(use.id for use in operation.uses) == operation.invoke_arguments
              and all(type(values[use.value]) is ReferenceValue for use in operation.uses[:-1])
              and type(values[operation.uses[-1].value]) is StateVersionValue
              and value_type(values[operation.uses[-1].value]) == state_type
              and all(tag in operation.behaviors for tag in (Behavior.CONTROL, Behavior.READ, Behavior.WRITE)),
              'compound operation lost its exact reference/world invoke interface')
        expected = ((ExitKind.NORMAL, ((reference_type,) if read else ()) + (state_type,)),
                    (ExitKind.RAISE, function_exception_types(language, reference_type, state_type)))
        _need(tuple((row.kind, row.types) for row in operation.completions) == expected
              and tuple(edge.kind for edge in operation.terminator.successors) == (ExitKind.NORMAL, ExitKind.RAISE)
              and all(edge.arguments == tuple(CompletionResult(index) for index in range(len(types)))
                      and tuple(value_type(values[value]) for value in blocks[edge.block].arguments) == types
                      for edge, (_, types) in zip(operation.terminator.successors, expected)),
              'normal/raise must carry their own payload and post-world, including traceback where applicable')
        ports = operation.uses[:-1]
        roles = (('receiver', 0), ('slice-lower', 0), ('slice-upper', 1)) if read else (
            (('assigned-reference', 0), ('heap-owner', 0), ('slice-lower', 0), ('slice-upper', 1)) if compound else
            (('assigned-reference', 0), ('heap-owner', 0), ('element-key', 0)))
        source_uses = [uses.get((identity, use.id)) for use in ports]
        _need(all(row is not None and (row.role, row.index) == role
                  and semantic.source_atoms.get(row.source.atom_id) is not None
                  and semantic.source_atoms[row.source.atom_id].reference == row.source
                  for row, role in zip(source_uses, roles)), 'ordered input roles/source Uses differ')
        if compound:
            _need(all(row.semantic_value is None for row in source_uses[-2:]),
                  'a bound cannot inherit the old whole-index StaticValue')
        owner = ports[0] if read else ports[1]
        selector = Slice(ports[-2].id, ports[-1].id) if compound else Element(ports[-1].id)
        effect = AddressEffect(ReferenceOwner(owner.id), (selector,), AccessMode.READ if read else AccessMode.WRITE)
        address = addresses.get(identity)
        _need(address is not None and address.site == InvokeStateOperand(identity, operation.uses[-1].id)
              and address.coverage is AddressCoverage.PARTIAL and address.protocol is None
              and address.effects == (effect,), 'partial slice address does not use the original operation ports')
        if not structural_only:
            nodes = [row for row in require_source_tree(source_tree, language.source, link.source)
                     if isinstance(row, ast.Subscript)]
            _need(len(nodes) == 1, 'operation requires one exact source Subscript')
            node, = nodes
            _need(isinstance(node.ctx, ast.Load if read else ast.Store), 'source context differs')
            if compound:
                _need(isinstance(node.slice, ast.Slice) and node.slice.step is None,
                      'combined operation requires a direct no-step slice')
                children = ((node.value, node.slice.lower or node.slice, node.slice.upper or node.slice) if read else
                            (node, node.value, node.slice.lower or node.slice, node.slice.upper or node.slice))
                for child, bound in zip((node.slice.lower, node.slice.upper), ports[-2:]):
                    if child is None:
                        _need(type(values[bound.value].origin) is LiteralOrigin
                              and values[bound.value].origin.value is None,
                              'omitted lower/upper must be the exact None literal')
            else:
                _need(any(isinstance(child, ast.Slice) for child in ast.walk(node.slice)),
                      'new consuming store cannot silently replace the historical ordinary store contract')
                children = (node, node.value, node.slice)
            _need(tuple(_slice_reference_span(row.source) for row in source_uses) ==
                  tuple((child.lineno, child.end_lineno, child.col_offset, child.end_col_offset) for child in children),
                  'input evaluation sources differ from the exact RHS/receiver/bounds/key ports')
            opcode = 'BINARY_SLICE' if read else 'STORE_SLICE' if compound else 'STORE_SUBSCR'
            _need(sum(row.opname == opcode and tuple(row.positions) == _slice_reference_span(link.source) for row in instructions) == 1,
                  'no unique exact compiler bytecode/position for the compound operation')
        _validate_local_transfer(identity, operations, protocols, blocks, values, links, uses, language)
    return selected


def _validate_local_transfer(target, operations, protocols, blocks, values, links, uses, language,
                             *, target_operands=None, target_failure_consumes=True,
                             failure_pending=None):
    """Follow actual normal CFG corridors from input holds through this macro.

    Exceptional corridors retain their original raw/paired payload and world.
    Their pending references independently establish the unchanged outer prefix.
    Holds keep operation identity and multiplicity; equal ValueIDs do not merge.
    """
    from scar.analysis.static.lowering.source_call_lowering import _pop_suffix, _check_failure_pending
    from scar.analysis.static.lowering.source_function_try import exception_handoff_holds
    ordered = list(operations)
    end = ordered.index(target)
    operands = operations[target].uses[:-1] if target_operands is None else target_operands
    starts = []
    for use in operands:
        origin = uses[target, use.id].source
        candidates = [identity for identity in ordered[:end]
            if (p := protocols.get(identity)) is not None and p.kind is K.FRAME_HOLD_TEMPORARY
            and len(operations[identity].uses) == 2 and operations[identity].uses[0].value == use.value
            and (row := uses.get((identity, operations[identity].uses[0].id))) is not None
            and row.source == origin]
        _need(bool(candidates), 'input has no preceding exact source-owned temporary hold')
        starts.append(candidates[-1])
    first = min(starts, key=ordered.index)
    positions = {identity: (block.id, index) for block in blocks.values() for index, identity in enumerate(block.operations)}
    entry = positions[first]
    queue, seen, failures = deque(((entry, ()),)), set(), []
    prefix, reached = None, False
    while queue:
        position, stack = queue.popleft()
        token = position, stack
        if token in seen:
            continue
        seen.add(token)
        _need(len(seen) <= max(128, len(operations) * 4), 'local pending CFG exceeds bounded fixed point')
        block, offset = position
        for identity in blocks[block].operations[offset:]:
            operation = operations[identity]
            protocol = protocols.get(identity)
            kind = protocol.kind if protocol is not None else None
            refs = tuple(use.value for use in operation.uses if type(values[use.value]) is ReferenceValue)
            if kind is K.FRAME_HOLD_TEMPORARY:
                _need(len(refs) == 1, 'temporary hold lost its reference')
                stack += ((identity, refs[0]),)
            elif kind in (K.FRAME_RELEASE_TEMPORARY, K.FRAME_FORWARD_TEMPORARY):
                stack = _pop_suffix(stack, refs)
            term = operation.terminator
            if term is None:
                continue
            if term.kind is TerminatorKind.INVOKE:
                normal, failure = stack, stack
                consumed, success_only = (), False
                if kind is not None and kind.value in SLICE_ACCESS_PROTOCOLS:
                    consumed = refs
                elif kind in (K.SOURCE_BINARY, K.SOURCE_COMPARE, K.SOURCE_INDEX):
                    consumed, success_only = refs, kind in (K.SOURCE_BINARY, K.SOURCE_INDEX)
                elif kind is K.BUILD_SLICE or links[identity].opcode is Opcode.INDEX:
                    consumed = refs
                elif kind is K.BUILD_MAP:
                    consumed = refs
                elif kind is K.BUILD_SET_DISPLAY:
                    consumed = refs
                elif kind is K.GET_ITERATOR:
                    consumed = refs
                elif kind is K.CREATE_GENEXPR:
                    # The creation interface also carries environment/cell
                    # references. Only its held function and eager iterator
                    # transfer ownership; these are not all input Uses.
                    consumed = (refs[0], refs[-1])
                elif kind in (K.MAP_ADD,):
                    consumed = refs[-2:]
                elif kind in (K.DICT_UPDATE, K.DICT_MERGE, K.APPEND_CALL_POSITIONAL,
                               K.EXTEND_CALL_POSITIONALS, K.APPEND_DISPLAY, K.EXTEND_DISPLAY):
                    consumed = refs[-1:]
                elif kind in (K.CALL_FUNCTION_EX, K.CALL_POSITIONALS_TO_TUPLE, K.DISPLAY_LIST_TO_TUPLE):
                    consumed = refs
                elif kind in (K.BUILD_CALL_POSITIONALS, K.BUILD_CALL_POSITIONAL_LIST, K.BUILD_DISPLAY_LIST):
                    consumed, success_only = refs, True
                elif links[identity].opcode in (Opcode.BUILD_TUPLE, Opcode.BUILD_LIST) and Behavior.MEMORY in operation.behaviors:
                    consumed, success_only = refs, True
                if consumed:
                    normal = _pop_suffix(stack, consumed)
                    failure = stack if success_only else normal
                failed = blocks[term.successors[1].block]
                if failure_pending is not None:
                    actual = failure_pending(operation, failed)
                elif C.FUNCTION_TRY_314_V1 in language.capabilities:
                    actual = exception_handoff_holds(failed.id, operations, protocols, blocks, uses)
                else:
                    _need(bool(failed.operations), 'raise lost its frame cleanup')
                    actual = _raw_handoff_holds(failed, operation, operations, protocols, values, uses)
                if identity == target:
                    _need(tuple(value for _, value in stack) == tuple(use.value for use in operands)
                          and not normal and (not failure if target_failure_consumes else failure == stack),
                          'macro did not preserve its exact normal/raise input-consumption contract')
                    prefix = _check_failure_pending(actual, failure, prefix)
                    reached = True
                else:
                    failures.append((actual, failure))
                    queue.append(((term.successors[0].block, 0), normal))
            elif term.kind in (TerminatorKind.JUMP, TerminatorKind.BRANCH):
                queue.extend(((edge.block, 0), stack) for edge in term.successors)
            else:
                _need(False, 'input evaluation normal corridor escaped before its consuming macro')
            break
    _need(reached and prefix is not None, 'input holds do not reach their consuming macro')
    for actual, local in failures:
        _check_failure_pending(actual, local, prefix)


__all__ = [
    'lower_slice',
    'lower_key',
    'lower_subscript',
    'lower_slice_store',
    'validate_slice_access',
]
