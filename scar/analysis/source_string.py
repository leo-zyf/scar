"""Source string."""

from __future__ import annotations

import ast
import dis
import sys
import antlr4
from dataclasses import replace
from scar.analysis.program_lowering import _Unsupported
from scar.ir.program.model import Behavior, ExitKind, LiteralOrigin, TerminatorKind
from scar.ir.program.source import GapKind, SourceRole
from scar.ir.program.structured import (
    PythonProtocolKind,
    PythonProtocolKind as K,
    PythonProtocolKind as Kind,
)
from scar.analysis.source_expression_rules import _REF, _require
from scar.ir.program import source_productions as P
from scar.ir.program.signature_patterns import CountBinding
from scar.analysis.source_language import (
    SourceLanguageOptions,
    require_source_tree,
    supports_cpython_compiler,
)
from scar.ir.v2.semantic import OperationKind
from io import StringIO
from pathlib import Path
from antlr4.error.ErrorListener import ErrorListener
from antlr4.error.Errors import ParseCancellationException
from antlr4.error.ErrorStrategy import BailErrorStrategy
from scar.analysis.source_data_materials import _artifact
from scar.analysis.source_override_syntax import _check_imports, _tool


# CPython 3.14 f-string construction over existing canonical CFG/world.
#
# An interpolation evaluates its value, performs !s/!r/!a conversion, evaluates
# its format specification, then formats.  Later pieces run only after that
# finishes.  Nested specifications use exactly the same source rule.  Literal
# pieces remain literals; zero/one-piece JoinedStr has no BUILD_STRING action.
#
# Protocol contracts (all incomplete invokes with NORMAL(ref,world) and
# RAISE(ref,world)):
#
# * FSTRING_CONVERT(value, conversion): conversion is a source-generated exact
#   integer literal, 115/114/97 for str/repr/ascii.  This is interpreter conversion,
#   not a lookup of the globals named str/repr/ascii.
# * FSTRING_FORMAT(value, spec): exact literal None denotes FORMAT_SIMPLE,
#   including its exact-str forwarding path; an actual spec reference denotes
#   FORMAT_WITH_SPEC, including an explicitly empty string.  Formatting may
#   dispatch __format__ and must validate its result without a purity assumption.
# * BUILD_STRING(parts...): two through thirty ordered formatted/literal pieces.
#   More pieces select CPython's list/append/join path, which is explicitly
#   unsupported here rather than moving intermediate allocation to the end.
#
# Consumed input holds are released in both completions before propagation.
# No target code, conversion, format method or tensor value is executed here.
# The recovered target-child path uses its original SG, not an invented legacy
# computational value. Consumption was checked against the retained CPython
# v3.14.7 Python/bytecodes.c (BUILD_STRING, CONVERT_VALUE, FORMAT_SIMPLE and
# FORMAT_WITH_SPEC); independent AST/CodeWitness/pending checks are separate.
# https://github.com/python/cpython/blob/v3.14.7/Python/bytecodes.c
_MAX_STACK_PIECES = 30
_CONVERSIONS = frozenset((115, 114, 97))


def _kind(name, node):
    kind = getattr(PythonProtocolKind, name, None)
    if kind is None:
        raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX, node,
                           "source profile has not connected the " + name + " protocol")
    return kind


def _hold(builder, cursor, node, reference):
    builder.frame_transition(cursor, node, ((reference, node, "fstring-held-reference"),),
                             PythonProtocolKind.FRAME_HOLD_TEMPORARY)
    builder.active_temporaries.append(reference)


def _release(builder, cursor, held):
    for node, reference in reversed(held):
        if not builder.active_temporaries or builder.active_temporaries[-1] != reference:
            raise _Unsupported(GapKind.SOURCE_MAPPING, node, "f-string temporary ownership is not nested")
        builder.active_temporaries.pop()
        builder.frame_transition(cursor, node, ((reference, node, "fstring-consumed-reference"),),
                                 PythonProtocolKind.FRAME_RELEASE_TEMPORARY)


def _invoke(builder, cursor, node, inputs, name, held):
    """Keep pending outer pieces; consume only this operation's input suffix."""
    def failed(failure, exception):
        pending = tuple(builder.active_temporaries)
        try:
            _release(builder, failure, held)
            builder.finish_exit(failure, node, exception, ExitKind.RAISE)
        finally:
            # Construction next visits the normal arm, which has its own
            # unconsumed input holds.  No runtime reference is retained here.
            builder.active_temporaries[:] = pending

    cursor, result, operation_id = builder.invoke_protocol(cursor, node, tuple(inputs),
        _kind(name, node), failure_handler=failed)
    operation = builder.operation_records[operation_id]
    # This marks possible allocation, including user format/conversion work;
    # it does not assert that every runtime path allocates a fresh string.
    marked = replace(operation, behaviors=operation.behaviors + (Behavior.MEMORY,))
    builder.operations[builder.operations.index(operation)] = marked
    builder.operation_records[operation_id] = marked
    _release(builder, cursor, held)
    return cursor, result


def _link_result(builder, node, reference):
    records = builder.old_by_atom.get(builder.source(node).atom_id, ())
    if len(records) > 1:
        raise _Unsupported(GapKind.SOURCE_MAPPING, node,
                           "f-string expression has ambiguous computational source records")
    builder.link_value(builder.values_by_id[reference], node, SourceRole.EXPRESSION,
                       semantic_value=records[0].result if records else None)


def _original_string_node(builder, node):
    from scar.analysis.source_heap_lowering import _source_definition
    from scar.ir.v2.semantic import OperationKind
    definition = _source_definition(builder, node)
    if (definition.kind is not OperationKind.OPERATOR
            or definition.metadata.get("ast_type") != type(node).__name__
            or len(definition.output_slots) != 1):
        raise _Unsupported(GapKind.SOURCE_MAPPING, node,
                           "recovered string requires its unique original SG operator/output slot")


def _formatted_value(builder, node, cursor, *, recover_missing_children=False):
    if recover_missing_children:
        _original_string_node(builder, node)
    if node.conversion != -1 and node.conversion not in _CONVERSIONS:
        raise _Unsupported(GapKind.SOURCE_MAPPING, node, "unsupported f-string conversion code")
    initial_holds = len(builder.active_temporaries)
    try:
        if recover_missing_children:
            from scar.analysis.source_heap_lowering import expression_for_heap_target
            cursor, value = expression_for_heap_target(builder, node.value, cursor)
        else:
            cursor, value = builder.expr(node.value, cursor)
        if cursor is None:
            return None, None
        _hold(builder, cursor, node.value, value)
        if node.conversion != -1:
            conversion = builder.literal(node, node.conversion)
            cursor, value = _invoke(builder, cursor, node, (value, conversion),
                                    "FSTRING_CONVERT", ((node.value, value),))
            _hold(builder, cursor, node, value)
        held = [(node, value)]
        if node.format_spec is None:
            spec = builder.literal(node, None)
        else:
            if not isinstance(node.format_spec, ast.JoinedStr):
                raise _Unsupported(GapKind.SOURCE_MAPPING, node.format_spec,
                                   "f-string format specification must retain its JoinedStr source")
            cursor, spec = lower_joined_string(builder, node.format_spec, cursor,
                                               recover_missing_children=recover_missing_children)
            if cursor is None:
                return None, None
            _hold(builder, cursor, node.format_spec, spec)
            held.append((node.format_spec, spec))
        cursor, result = _invoke(builder, cursor, node, (value, spec), "FSTRING_FORMAT", tuple(held))
        _link_result(builder, node, result)
        return cursor, result
    finally:
        del builder.active_temporaries[initial_holds:]


def lower_joined_string(builder, node, cursor, *, recover_missing_children=False):
    """Construct ordinary f-strings; preserve every child and evaluation edge."""
    if not isinstance(node, ast.JoinedStr):
        raise _Unsupported(GapKind.SOURCE_MAPPING, node, "expected JoinedStr source")
    if type(recover_missing_children) is not bool:
        raise TypeError("missing string-child recovery must be an explicit boolean")
    if recover_missing_children:
        _original_string_node(builder, node)
    if len(node.values) > _MAX_STACK_PIECES:
        raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX, node,
                           "large f-string requires the compiler's incremental list/append/join protocol")
    if not node.values:
        result = builder.literal(node, "")
        _link_result(builder, node, result)
        return cursor, result
    initial_holds = len(builder.active_temporaries)
    pieces, held = [], []
    try:
        for child in node.values:
            if isinstance(child, ast.Constant) and type(child.value) is str:
                if recover_missing_children:
                    from scar.analysis.source_heap_lowering import _source_definition, expression_for_heap_target
                    from scar.ir.v2.semantic import OperationKind
                    definition = _source_definition(builder, child)
                    if definition.kind is not OperationKind.CONSTANT or len(definition.output_slots) != 1:
                        raise _Unsupported(GapKind.SOURCE_MAPPING, child,
                                           "recovered string literal requires its original SG constant/output")
                    cursor, value = expression_for_heap_target(builder, child, cursor)
                else:
                    value = builder.literal(child, child.value)
            elif isinstance(child, ast.FormattedValue):
                cursor, value = _formatted_value(builder, child, cursor,
                                                 recover_missing_children=recover_missing_children)
                if cursor is None:
                    return None, None
            else:
                raise _Unsupported(GapKind.SOURCE_MAPPING, child,
                                   "JoinedStr piece must be an exact string literal or FormattedValue")
            pieces.append(value)
            if len(node.values) > 1:
                _hold(builder, cursor, child, value)
                held.append((child, value))
        if len(pieces) == 1:
            result = pieces[0]
        else:
            cursor, result = _invoke(builder, cursor, node, pieces, "BUILD_STRING", tuple(held))
        _link_result(builder, node, result)
        return cursor, result
    finally:
        del builder.active_temporaries[initial_holds:]


def validate_string_protocols(model, graph, interface, metadata, semantic, *, language=None):
    from scar.analysis.source_language import function_exception_types
    from scar.ir.program.model import (CompletionResult, LiteralOrigin, ReferenceValue,
        StateVersionValue, TerminatorKind, ValueKind, ValueType)
    kinds = (PythonProtocolKind.FSTRING_CONVERT, PythonProtocolKind.FSTRING_FORMAT,
             PythonProtocolKind.BUILD_STRING)
    operations = {operation.id: operation for operation in graph.operations}
    values = {value.id: value for value in graph.values}
    links = {link.operation: link for link in metadata.operation_links}
    world_location = values[interface.world].location
    ref_type, state_type = ValueType(ValueKind.REFERENCE), ValueType(ValueKind.STATE, location=world_location)
    for protocol in model.protocols:
        if protocol.kind not in kinds:
            continue
        operation, link = operations[protocol.operation], links[protocol.operation]
        atom = semantic.source_atoms.get(link.source.atom_id)
        expected = "JoinedStr" if protocol.kind is PythonProtocolKind.BUILD_STRING else "FormattedValue"
        arity = len(operation.uses) - 1
        arity_ok = 2 <= arity <= 30 if protocol.kind is PythonProtocolKind.BUILD_STRING else arity == 2
        if (atom is None or atom.kind != expected or link.semantic_operation is not None or not arity_ok
                or any(type(values[use.value]) is not ReferenceValue for use in operation.uses[:-1])
                or type(values[operation.uses[-1].value]) is not StateVersionValue
                or values[operation.uses[-1].value].location != world_location
                or operation.effects_complete or operation.callee is not None
                or Behavior.MEMORY not in operation.behaviors
                or operation.terminator is None or operation.terminator.kind is not TerminatorKind.INVOKE
                or operation.invoke_arguments != tuple(use.id for use in operation.uses)):
            raise ValueError("string protocol must retain its exact source, ordered references and incomplete world invoke")
        if protocol.kind is PythonProtocolKind.FSTRING_CONVERT:
            origin = values[operation.uses[1].value].origin
            if type(origin) is not LiteralOrigin or type(origin.value) is not int or origin.value not in _CONVERSIONS:
                raise ValueError("f-string conversion must use its exact source-generated ASCII literal")
        if (tuple((signature.kind, signature.types) for signature in operation.completions) !=
                ((ExitKind.NORMAL, (ref_type, state_type)), (ExitKind.RAISE, function_exception_types(language, ref_type, state_type)))
                or tuple(edge.kind for edge in operation.terminator.successors) != (ExitKind.NORMAL, ExitKind.RAISE)
                or any(edge.arguments != tuple(CompletionResult(i) for i in range(len(signature.types)))
                       for edge, signature in zip(operation.terminator.successors, operation.completions))):
            raise ValueError("string protocol must keep normal and exceptional payload/post-world")


# Formatted/template string schedules with conversion and capture distinct.
def emit_string(b, region, state):
    kind = b.node.constructor
    if kind in ('FormattedValue', 'Interpolation'):
        return _formatted(b, region, state, template=kind == 'Interpolation')
    b.mark('values', P.ProductionDispositionKind.EVALUATED_CHILD)
    values = b.value(b.site('values'))
    _require(values.kind == 'LIST', 'RULE_SHAPE', str(b.occurrence), 'string parts must be an original list')
    if kind == 'TemplateStr':
        return _template(b, region, state, values.items)
    held, sequence = [], None
    if len(values.items) > b.rules.call_profile['stack_guideline']:
        sequence, state = b.protocol(region, K.BUILD_DISPLAY_LIST, (), state,
            (CountBinding('item_count', 0),), (b.site('values'),))
    for ordinal, item in enumerate(values.items):
        _part(b, item, ('Constant', 'FormattedValue'))
        value, state = b.evaluate(region, b.site('values', ordinal), state)
        if sequence is None:
            held.append(value)
        else:
            state, = b.protocol(region, K.APPEND_DISPLAY, (sequence, value), state, selectors=(b.site('values', ordinal),))
    if sequence is not None:
        value, state = b.action(region, P.LanguageActionKind.STRING_JOIN, (b.site('values'),), (sequence, state))
    elif not held:
        value, state = b.literal(region, '', state, source=b.site('values'))
    elif len(held) == 1:
        value = held[0]
    else:
        value, state = b.protocol(region, K.BUILD_STRING, tuple(held), state,
            (CountBinding('item_count', len(held)),), (b.site('values'),))
    b.finish(region, (value, state), (_REF, b.world))


def _part(b, item, kinds):
    _require(item.kind == 'NODE', 'RULE_SHAPE', str(b.occurrence), 'string part must be an expression')
    node = b.rules.source.context.syntax.nodes[item.reference]
    _require(node.constructor in kinds, 'RULE_SHAPE', str(b.occurrence), 'unexpected string part constructor')
    if node.constructor == 'Constant':
        raw = b.role.field(item.reference, 'value')
        _require(raw.kind == 'SCALAR' and type(raw.scalar.to_python()) is str,
                 'RULE_SHAPE', str(item.reference), 'literal string part must be an exact string')
    return node.constructor


def _formatted(b, region, state, template):
    b.mark('value', P.ProductionDispositionKind.EVALUATED_CHILD)
    value, state = b.evaluate(region, b.site('value'), state)
    conversion = b.value(b.site('conversion'))
    _require(conversion.kind == 'SCALAR', 'RULE_SHAPE', str(b.occurrence), 'conversion requires a source integer')
    code = conversion.scalar.to_python()
    _require(type(code) is int and code in (-1, 115, 114, 97), 'RULE_SELECTOR', str(b.occurrence), 'invalid string conversion')
    expression = None
    if template:
        expression, state = b.emit(region, P.ConstructValue(P.ConstructionKind.LITERAL, (b.site('str'),)),
                                   (state,), (_REF, b.world), source=b.site('str'))
    elif code != -1:
        selector, state = b.emit(region, P.ConstructValue(P.ConstructionKind.LITERAL, (b.site('conversion'),)),
                                 (state,), (_REF, b.world), source=b.site('conversion'))
        value, state = b.protocol(region, K.FSTRING_CONVERT, (value, selector), state, selectors=(b.site('conversion'),))
    spec = b.value(b.site('format_spec'))
    if spec.kind == 'NODE':
        b.mark('format_spec', P.ProductionDispositionKind.EVALUATED_CHILD)
        format_spec, state = b.evaluate(region, b.site('format_spec'), state)
    else:
        _require(spec.kind == 'SCALAR' and spec.scalar.to_python() is None, 'RULE_SHAPE', str(b.occurrence), 'absent format spec must be None')
        format_spec, state = b.literal(region, '', state, source=b.site('format_spec'))
    if template:
        value, state = b.emit(region, P.ConstructValue(P.ConstructionKind.INTERPOLATION,
            (b.site('str'), b.site('conversion'), b.site('format_spec'))),
            (value, expression, format_spec, state), (_REF, b.world))
    else:
        value, state = b.protocol(region, K.FSTRING_FORMAT, (value, format_spec), state, selectors=(b.site('format_spec'),))
    b.finish(region, (value, state), (_REF, b.world))


def _template(b, region, state, items):
    strings, interpolations, previous = [], [], True
    for ordinal, item in enumerate(items):
        kind = _part(b, item, ('Constant', 'Interpolation'))
        if kind == 'Interpolation':
            if previous:
                empty, state = b.literal(region, '', state, source=b.site('values', ordinal))
                strings.append(empty)
            previous = True
        else:
            literal, state = b.evaluate(region, b.site('values', ordinal), state)
            strings.append(literal)
            previous = False
    if previous:
        empty, state = b.literal(region, '', state, source=b.site('values'))
        strings.append(empty)
    string_tuple, state = b.emit(region, P.ConstructValue(P.ConstructionKind.TUPLE, (b.site('values'),),
                                 P.TupleConstructionLayout.TEMPLATE_STRINGS),
                                 (*strings, state), (_REF, b.world))
    for ordinal, item in enumerate(items):
        if b.rules.source.context.syntax.nodes[item.reference].constructor == 'Interpolation':
            value, state = b.evaluate(region, b.site('values', ordinal), state)
            interpolations.append(value)
    if interpolations:
        interpolation_tuple, state = b.emit(region, P.ConstructValue(P.ConstructionKind.TUPLE, (b.site('values'),),
                                            P.TupleConstructionLayout.TEMPLATE_INTERPOLATIONS),
                                            (*interpolations, state), (_REF, b.world))
    else:
        interpolation_tuple, state = b.literal(region, (), state, source=b.site('values'))
    value, state = b.emit(region, P.ConstructValue(P.ConstructionKind.TEMPLATE_STRING, (b.site('values'),)),
                         (string_tuple, interpolation_tuple, state), (_REF, b.world))
    b.finish(region, (value, state), (_REF, b.world))


# Bounded source/compiler/pending queries over existing f-string protocols.
#
# No runtime string, conversion method, alternative graph or lifetime IR is
# created. Primitive input consumption follows pinned CPython 3.14.7 bytecodes;
# an unknown child contract never establishes the enclosing frame's lifetime.
POLICY = "CPYTHON_314_FSTRING_SOURCE_V1"
_KINDS = frozenset((Kind.FSTRING_CONVERT, Kind.FSTRING_FORMAT, Kind.BUILD_STRING))


def _need(condition, detail):
    if not condition:
        raise ValueError("string source validation: " + detail)


def _span(node):
    return node.lineno, node.end_lineno, node.col_offset, node.end_col_offset


def _references(operation):
    return tuple(use.value for use in operation.uses[:-1])


def _consumed(operation, kind, values):
    refs = _references(operation)
    if kind is Kind.BUILD_STRING:
        return refs
    if kind is Kind.FSTRING_CONVERT:
        return refs[:1]  # The generated conversion selector has no hold.
    spec = values[refs[1]].origin
    return refs[:1] if type(spec) is LiteralOrigin and spec.value is None else refs


def _check_consumed_releases(operation, consumed, operations, protocols, blocks, use_links, language=None):
    """Inspect both actual CFG arms, not a producer-reported consumed set."""
    failure_exit = None
    for edge in operation.terminator.successors:
        block = blocks[edge.block]
        _need(len(block.operations) > len(consumed), "string completion lacks its input releases/continuation")
        for identity, reference in zip(block.operations, reversed(consumed)):
            release = operations[identity]
            protocol = protocols.get(identity)
            refs = _references(release)
            _need(protocol is not None and protocol.kind is Kind.FRAME_RELEASE_TEMPORARY
                  and refs == (reference,), "string completion lost or reordered a consumed input release")
            link = use_links.get((identity, release.uses[0].id))
            _need(link is not None and link.role == "fstring-consumed-reference",
                  "string input release lacks its original reference Use")
        following = operations[block.operations[len(consumed)]]
        first = use_links.get((following.id, following.uses[0].id)) if following.uses else None
        _need(first is None or first.role != "fstring-consumed-reference",
              "string completion releases an extra input hold")
        if edge.kind is ExitKind.RAISE:
            from scar.ir.program.source_language import SourceLanguageCapability
            paired=language is not None and SourceLanguageCapability.FUNCTION_TRY_314_V1 in language.capabilities
            if not paired:
                _need(protocols.get(following.id) is not None and protocols[following.id].kind is Kind.FRAME_EXIT,
                      "string failure does not carry its post-world through FRAME_EXIT")
                failure_exit=tuple(use.value for use in following.uses
                    if(link:=use_links.get((following.id,use.id)))is not None and link.role=="held-temporary")
            else:
                # The original C0/C1 checker has validated this exception-only
                # handoff. Inspect real releases until the first context Enter
                # or frame close; never execute a handler or assume a bool.
                cursor=(block.id,len(consumed));unwound=[];seen=set()
                while cursor not in seen:
                    seen.add(cursor);current=blocks[cursor[0]]
                    _need(cursor[1]<len(current.operations),"string failure falls out of its cleanup")
                    op=operations[current.operations[cursor[1]]];marker=protocols.get(op.id)
                    kind=marker.kind if marker is not None else None
                    first=use_links.get((op.id,op.uses[0].id))if op.uses else None
                    if kind is Kind.FRAME_RELEASE_TEMPORARY and first is not None and first.role=='function-exception-unwind':
                        unwound.append(op.uses[0].value)
                    if kind in (Kind.FRAME_EXIT,Kind.FUNCTION_FRAME_EXIT_EXCEPTION_PACKET):
                        retained=tuple(use.value for use in op.uses if(link:=use_links.get((op.id,use.id)))is not None and link.role=='held-temporary')
                        failure_exit=retained+tuple(reversed(unwound));break
                    if kind is Kind.FUNCTION_HANDLED_CONTEXT_ENTER:
                        failure_exit=tuple(reversed(unwound));break
                    term=op.terminator
                    if term is None:cursor=(current.id,cursor[1]+1)
                    else:
                        _need(term.kind in (TerminatorKind.JUMP,TerminatorKind.PENDING_DISPATCH,TerminatorKind.CLEANUP_ROUTE)
                              and len(term.successors)==1,'string failure did not preserve its canonical exception handoff')
                        cursor=(term.successors[0].block,0)
                _need(failure_exit is not None,'string exception handoff has no bounded cleanup endpoint')
    _need(failure_exit is not None, "string primitive has no exceptional cleanup")
    return failure_exit


def _ordered_next(current, expected, own, operations, blocks, positions):
    """Every ordinary normal continuation must reach the next owned stage."""
    edge = operations[current].terminator.successors[0]
    pending, seen, found = [(edge.block, 0)], set(), set()
    while pending:
        position = pending.pop()
        if position in seen:
            continue
        seen.add(position)
        _need(len(seen) <= len(operations) * 2 + 1, "string staging traversal budget exceeded")
        block, index = blocks[position[0]], position[1]
        _need(index < len(block.operations), "string normal continuation falls out of a block")
        identity = block.operations[index]
        if identity in own:
            found.add(identity)
            continue
        operation = operations[identity]
        if operation.terminator is None:
            pending.append((block.id, index + 1))
        elif operation.terminator.kind is TerminatorKind.INVOKE:
            pending.append((operation.terminator.successors[0].block, 0))
        elif operation.terminator.kind in (TerminatorKind.JUMP, TerminatorKind.BRANCH):
            pending.extend((edge.block, 0) for edge in operation.terminator.successors)
        else:
            raise ValueError("string source validation: normal path skips a later string stage")
    _need(found == {expected}, "string stages can be reordered, repeated or skipped")


def _pending_group(group, sources, operations, protocols, blocks, positions, uses, values, failure_holds):
    """Track local ordered hold instances; leave the external prefix symbolic."""
    candidates = []
    for identity, operation in operations.items():
        protocol = protocols.get(identity)
        if protocol is None or protocol.kind is not Kind.FRAME_HOLD_TEMPORARY:
            continue
        link = uses.get((identity, operation.uses[0].id))
        if link is not None and link.role == "fstring-held-reference" and link.source.atom_id in sources:
            candidates.append(identity)
    _need(candidates, "string construction has no original input hold")
    order = {identity: index for index, identity in enumerate(operations)}
    begin = min(candidates, key=order.get)
    position, stack, prefix, seen = positions[begin], (), None, set()
    own, completed, frontiers = set(group), set(), []
    while position not in seen:
        seen.add(position)
        _need(len(seen) <= len(operations) + 1, "string pending traversal budget exceeded")
        block, index = blocks[position[0]], position[1]
        _need(index < len(block.operations), "string pending path falls out of a block")
        identity = block.operations[index]
        operation, protocol = operations[identity], protocols.get(identity)
        kind = protocol.kind if protocol is not None else None
        link = uses.get((identity, operation.uses[0].id)) if operation.uses else None
        if kind is Kind.FRAME_HOLD_TEMPORARY and identity in candidates:
            stack += ((identity, operation.uses[0].value),)
        elif kind is Kind.FRAME_RELEASE_TEMPORARY and link is not None and link.role == "fstring-consumed-reference":
            _need(stack and stack[-1][1] == operation.uses[0].value,
                  "string normal continuation releases a different ordered hold instance")
            stack = stack[:-1]
        if identity in own:
            consumed = _consumed(operation, kind, values)
            _need(len(stack) >= len(consumed) and tuple(row[1] for row in stack[-len(consumed):]) == consumed,
                  "string primitive does not own its ordered input hold suffix")
            remaining = stack[:-len(consumed)]
            actual = failure_holds[identity]
            expected = tuple(row[1] for row in remaining)
            _need(len(actual) >= len(expected) and (not expected or actual[-len(expected):] == expected),
                  "string failure lost or reordered its retained pieces")
            current_prefix = actual[:-len(expected)] if expected else actual
            _need(prefix is None or current_prefix == prefix,
                  "string failure changed its enclosing pending prefix")
            prefix = current_prefix
            completed.add(identity)
        if completed == own and not stack:
            return {"ordered_local_pending_checked": True, "enclosing_prefix_consistency_checked": True,
                    "intermediate_consumption_verified": not frontiers, "frontiers": frontiers,
                    "outer_pending_verified": False, "whole_lifetime_verified": False}
        term = operation.terminator
        if term is None:
            position = (block.id, index + 1)
        elif term.kind is TerminatorKind.INVOKE:
            if identity not in own:
                frontiers.append({"kind": "FSTRING_CHILD_PRIMITIVE_CONSUMPTION_UNVERIFIED", "operation": identity.wire})
            position = (term.successors[0].block, 0)
        elif term.kind is TerminatorKind.JUMP:
            position = (term.successors[0].block, 0)
        else:
            return {"ordered_local_pending_checked": False, "enclosing_prefix_consistency_checked": False,
                    "intermediate_consumption_verified": False,
                    "frontiers": frontiers + [{"kind": "FSTRING_CHILD_CONTROL_FLOW_UNVERIFIED", "operation": identity.wire}],
                    "outer_pending_verified": False, "whole_lifetime_verified": False}
    raise ValueError("string source validation: string construction repeats without completing its held suffix")


def validate_string_source(model, graph, interface, metadata, semantic, *,
                           source_tree=None, options=None, structural_only=False, language=None):
    _need(type(structural_only) is bool, "verification mode must be explicit")
    protocols = {row.operation: row for row in model.protocols}
    selected = {identity for identity, row in protocols.items() if row.kind in _KINDS}
    if not selected:
        return {"policy": POLICY, "protocol_operations": 0, "records": [], "whole_lifetime_verified": False}
    from scar.analysis.source_string import validate_string_protocols
    validate_string_protocols(model, graph, interface, metadata, semantic, language=language)
    operations = {row.id: row for row in graph.operations}
    values = {row.id: row for row in graph.values}
    blocks = {block.id: block for region in graph.regions for block in region.blocks}
    positions = {identity: (block.id, index) for block in blocks.values() for index, identity in enumerate(block.operations)}
    uses = {(row.operation, row.use): row for row in metadata.use_links}
    value_sources = {(row.value, row.source.atom_id) for row in metadata.value_links}
    by_source, failure_holds = {}, {}
    for identity in selected:
        row = protocols[identity]
        atom = semantic.source_atoms.get(row.source.atom_id)
        expected = "JoinedStr" if row.kind is Kind.BUILD_STRING else "FormattedValue"
        definitions = [d for d in semantic.definitions.values() if d.kind is OperationKind.OPERATOR
            and d.metadata.get("ast_type") == expected and d.code_id == row.source.atom_id.wire
            and row.source.atom_id in d.source_atoms and len(d.output_slots) == 1]
        _need(atom is not None and atom.reference == row.source and atom.kind == expected and len(definitions) == 1,
              "string protocol has no unique original SG operator/output")
        key = row.kind, row.source.atom_id
        _need(key not in by_source, "duplicate string stage at one original source atom")
        by_source[key] = identity
        failure_holds[identity] = _check_consumed_releases(operations[identity],
            _consumed(operations[identity], row.kind, values), operations, protocols, blocks, uses, language)
    if structural_only:
        return {"policy": POLICY, "protocol_operations": len(selected), "records": [],
                "source_shape_verified": False, "compiler_verified": False,
                "both_completion_releases_checked": True, "outer_pending_verified": False,
                "whole_lifetime_verified": False, "frontiers": ["FSTRING_SOURCE_AND_COMPILER_NOT_REVIEWED"]}
    _need(isinstance(source_tree, ast.Module) and type(options) is SourceLanguageOptions,
          "string source checking requires original AST and independent options")
    evidence = options.code_evidence
    instructions = ()
    if evidence is not None:
        evidence.require_definition(metadata.source, interface.definition, model.source)
        _need(supports_cpython_compiler(evidence),
              "string contract requires pinned CPython 3.14.7")
        instructions = tuple(dis.get_instructions(evidence.code))
    parents = {child: parent for parent in ast.walk(source_tree) for child in ast.iter_child_nodes(parent)}

    def node_for(source, cls):
        matches = [n for n in require_source_tree(source_tree, metadata.source, source) if isinstance(n, cls)]
        _need(len(matches) == 1, "string stage does not identify one exact original AST node")
        return matches[0]

    def reference(node):
        matches = [row for row in semantic.source_atoms.values() if row.kind == type(node).__name__
                   and row.reference.path == metadata.source.path and row.reference.fingerprint == metadata.source.fingerprint
                   and (row.reference.start_line, row.reference.end_line, row.reference.start_column,
                        row.reference.end_column) == _span(node)]
        _need(len(matches) == 1, "string child has no unique original source atom")
        source = matches[0].reference
        definitions = [d for d in semantic.definitions.values() if d.code_id == source.atom_id.wire
            and source.atom_id in d.source_atoms and len(d.output_slots) == 1 and d.metadata.get("role") != "binding"]
        _need(len(definitions) == 1, "string child has no unique original SG output")
        return source

    def result(identity):
        return blocks[operations[identity].terminator.successors[0].block].arguments[0]

    def literal(value, expected):
        origin = values[value].origin
        return type(origin) is LiteralOrigin and type(origin.value) is type(expected) and origin.value == expected

    def value_matches(value, node):
        _need((value, reference(node).atom_id) in value_sources, "string child reference lost its exact source correspondence")
        if isinstance(node, ast.Constant):
            _need(literal(value, node.value), "string literal differs from its exact AST value")

    roots = set()
    for identity in selected:
        source, kind = protocols[identity].source, protocols[identity].kind
        node = node_for(source, ast.JoinedStr if kind is Kind.BUILD_STRING else ast.FormattedValue)
        if isinstance(node, ast.FormattedValue):
            node = parents.get(node)
        _need(isinstance(node, ast.JoinedStr), "formatted value has no original JoinedStr owner")
        while isinstance(parents.get(node), ast.FormattedValue) and parents[node].format_spec is node:
            node = parents.get(parents[node])
        roots.add(node)
    covered, records = set(), []
    for root in sorted(roots, key=_span):
        stages = []
        def stage(node, kind):
            identity = by_source.get((kind, reference(node).atom_id))
            _need(identity is not None, "original string AST is missing a required protocol stage")
            stages.append(identity)
            return identity

        def formatted(node, expected=None):
            identity = by_source.get((Kind.FSTRING_FORMAT, reference(node).atom_id))
            _need(identity is not None, "formatted value has no format stage")
            refs = _references(operations[identity])
            if node.conversion == -1:
                value_matches(refs[0], node.value)
                _need((Kind.FSTRING_CONVERT, reference(node).atom_id) not in by_source,
                      "string inserted a conversion absent from its AST")
            else:
                conversion = stage(node, Kind.FSTRING_CONVERT)
                inputs = _references(operations[conversion])
                value_matches(inputs[0], node.value)
                _need(literal(inputs[1], node.conversion) and refs[0] == result(conversion),
                      "string conversion selector/result differs from its AST")
            if node.format_spec is None:
                _need(literal(refs[1], None), "simple format was replaced with an explicit spec")
            else:
                _need(isinstance(node.format_spec, ast.JoinedStr), "format spec lost its original JoinedStr")
                joined(node.format_spec, refs[1])
            stage(node, Kind.FSTRING_FORMAT)
            if expected is not None:
                _need(result(identity) == expected, "join/spec uses a different formatted result")
            return result(identity)

        def joined(node, expected=None):
            _need(len(node.values) <= 30, "large string requires a separate compiler list/join contract")
            if len(node.values) > 1:
                identity = by_source.get((Kind.BUILD_STRING, reference(node).atom_id))
                _need(identity is not None, "JoinedStr has no ordered join")
                refs = _references(operations[identity])
                _need(len(refs) == len(node.values), "join piece count differs from exact AST")
            else:
                identity, refs = None, (expected,)
                _need((Kind.BUILD_STRING, reference(node).atom_id) not in by_source,
                      "zero/one-piece JoinedStr must not add BUILD_STRING")
            value = expected
            for index, child in enumerate(node.values):
                ref = refs[index]
                if isinstance(child, ast.FormattedValue):
                    value = formatted(child, ref)
                else:
                    _need(isinstance(child, ast.Constant) and type(child.value) is str,
                          "string piece is neither an exact literal nor FormattedValue")
                    if ref is None:
                        choices = [v for v, atom in value_sources if atom == reference(child).atom_id and literal(v, child.value)]
                        _need(len(choices) == 1, "string literal has no unique original reference")
                        ref = choices[0]
                    value_matches(ref, child)
                    value = ref
                    if evidence is not None:
                        matches = [i for i in instructions if i.opname == "LOAD_CONST" and tuple(i.positions) == _span(child)
                                   and type(i.argval) is str and i.argval == child.value]
                        _need(len(matches) == 1, "string literal lacks its exact compiler constant slot/position")
            if identity is not None:
                stage(node, Kind.BUILD_STRING)
                value = result(identity)
            if not node.values:
                _need(expected is not None and literal(expected, ""), "empty spec is not the exact empty-string reference")
            _need(expected is None or value == expected, "string result differs from its exact consumer")
            return value

        value = joined(root)
        _need(stages and not covered.intersection(stages), "string stages have duplicate owners")
        covered.update(stages)
        for current, following in zip(stages, stages[1:]):
            _ordered_next(current, following, set(stages), operations, blocks, positions)
        offsets = []
        if evidence is not None:
            for identity in stages:
                protocol = protocols[identity]
                node = node_for(protocol.source, ast.JoinedStr if protocol.kind is Kind.BUILD_STRING else ast.FormattedValue)
                name = ("BUILD_STRING" if protocol.kind is Kind.BUILD_STRING else "CONVERT_VALUE"
                        if protocol.kind is Kind.FSTRING_CONVERT else "FORMAT_SIMPLE" if node.format_spec is None else "FORMAT_WITH_SPEC")
                matches = [i for i in instructions if i.opname == name and tuple(i.positions) == _span(node)]
                _need(len(matches) == 1, "string stage has no unique exact compiler instruction/position")
                instruction = matches[0]
                if name == "BUILD_STRING":
                    _need(instruction.arg == len(node.values), "compiler join arity differs")
                elif name == "CONVERT_VALUE":
                    _need(instruction.arg == {115: 1, 114: 2, 97: 3}.get(node.conversion), "compiler conversion differs")
                offsets.append(instruction.offset)
            _need(offsets == sorted(set(offsets)), "compiler string stage ordering differs")
        # Select the exact AST roles held by string construction. An
        # interpolation value can be any supported expression, including a
        # Call/Attribute/Subscript; its hold is sourced at that expression,
        # not at the Names nested inside it. Hold instances still retain
        # their individual operation IDs and ordered references below.
        held_nodes = set()
        for node in ast.walk(root):
            if isinstance(node, ast.JoinedStr) and len(node.values) > 1:
                held_nodes.update(node.values)
            elif isinstance(node, ast.FormattedValue):
                held_nodes.add(node.value)
                if node.conversion != -1:
                    held_nodes.add(node)
                if node.format_spec is not None:
                    held_nodes.add(node.format_spec)
        sources = {reference(node).atom_id for node in held_nodes}
        pending = _pending_group(stages, sources, operations, protocols, blocks, positions, uses, values, failure_holds)
        parent = parents.get(root)
        if isinstance(parent, ast.Subscript) and isinstance(parent.ctx, ast.Store) and parent.slice is root:
            consumers = [row for row in metadata.use_links if row.role == "element-key" and row.source == reference(root)
                         and operations[row.operation].uses[[u.id for u in operations[row.operation].uses].index(row.use)].value == value]
            _need(len(consumers) == 1 and protocols.get(consumers[0].operation) is not None
                  and protocols[consumers[0].operation].kind is Kind.STORE_SUBSCRIPT,
                  "target string result does not reach its unique original subscript store")
            _ordered_next(stages[-1], consumers[0].operation, set(stages) | {consumers[0].operation}, operations, blocks, positions)
        records.append({"source": reference(root).atom_id.wire, "protocol_operations": len(stages),
            "source_shape_verified": True, "compiler_verified": evidence is not None,
            "both_completion_releases_checked": True, "pending": pending,
            "frontiers": ([] if evidence is not None else ["FSTRING_CODE_WITNESS_REQUIRED"])
                + ["FSTRING_OUTER_PENDING_PREFIX_UNVERIFIED", "FSTRING_RUNTIME_AND_EFFECTS_UNVERIFIED"],
            "effects_established": False, "runtime_code_instance_established": False, "whole_lifetime_verified": False})
    _need(covered == selected, "string protocol is not owned by its exact AST construction")
    return {"policy": POLICY, "protocol_operations": len(selected), "records": records,
            "effects_established": False, "runtime_code_instance_established": False,
            "outer_pending_verified": False, "whole_lifetime_verified": False}


# Approved interpolation grammar tools, without configuration evaluation.
_APPROVED = {
    'grammar/gen/OmegaConfGrammarLexer.py': 'f568dd6f735790223ea54db71486f89128320dd2045a8d2115f48b305a552b44',
    'grammar/gen/OmegaConfGrammarParser.py': '2c215392e4da4585bf9f1221dc9cf79fe25bc21f7d19dad994e6a6464c2a7c00',
}


class _Fail(ErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, error):
        raise ValueError('interpolation syntax '+str(line)+':'+str(column)+': '+str(msg))


class InterpolationSyntax:
    """Parse configValue through EOF; no visitor, resolver or application API."""
    def __init__(self, omegaconf_source):
        root = Path(omegaconf_source).absolute()
        profiles, sources = [], {}
        for relative, expected in _APPROVED.items():
            artifact, raw = _artifact(root/relative, 'scar.approved-interpolation-grammar-tool')
            if artifact.sha256 != expected:
                raise ValueError('interpolation grammar version is not approved: '+relative)
            _check_imports(raw)
            profiles.append(artifact)
            sources[relative] = raw
        # Some distributions retain generated grammar metadata; this one need
        # not provide .g4, .tokens or .interp files. Never fetch replacements.
        stems = {'OmegaConfGrammarLexer', 'OmegaConfGrammarParser'}
        for path in sorted((root/'grammar').rglob('*')):
            if path.is_file() and path.suffix in ('.tokens', '.interp') and path.stem in stems:
                profiles.append(_artifact(path, 'scar.interpolation-grammar-metadata')[0])
        runtime = Path(antlr4.__file__).resolve().parent
        for path in sorted(runtime.rglob('*.py')):
            profiles.append(_artifact(path, 'scar.antlr4-parser-runtime-source')[0])
        profiles.append(_artifact(Path(sys.executable).resolve(), 'scar.analyzer-python-executable')[0])
        self.profiles = tuple(profiles)
        self.lexer = _tool(root/'grammar/gen/OmegaConfGrammarLexer.py',
            sources['grammar/gen/OmegaConfGrammarLexer.py'], 'OmegaConfGrammarLexer')
        self.parser = _tool(root/'grammar/gen/OmegaConfGrammarParser.py',
            sources['grammar/gen/OmegaConfGrammarParser.py'], 'OmegaConfGrammarParser')

    def parse(self, text):
        if type(text) is not str:
            raise TypeError('interpolation input must be a string')
        lexer = self.lexer(antlr4.InputStream(text), output=StringIO())
        lexer.removeErrorListeners()
        lexer.addErrorListener(_Fail())
        lexer.mode(self.lexer.DEFAULT_MODE)
        stream = antlr4.CommonTokenStream(lexer)
        try:
            stream.fill()
            stream.seek(0)
            parser = self.parser(stream, output=StringIO())
            parser.removeErrorListeners()
            parser.addErrorListener(_Fail())
            parser._errHandler = BailErrorStrategy()
            tree = parser.configValue()
        except ParseCancellationException as error:
            raise ValueError('interpolation does not match the complete grammar') from error
        except Exception as error:
            # The approved generated lexer can raise this exact mode-stack
            # error; it must not escape as a successfully recovered tree.
            if type(error) is Exception and str(error) == 'Empty Stack':
                raise ValueError('invalid interpolation lexer mode stack') from error
            raise
        if parser.getNumberOfSyntaxErrors() or tree.EOF() is None or stream.LA(1) != antlr4.Token.EOF:
            raise ValueError('interpolation did not consume the complete input')
        return tree


__all__ = [
    'lower_joined_string',
    'validate_string_protocols',
    'validate_string_source',
    'emit_string',
    'POLICY',
    'InterpolationSyntax',
]
