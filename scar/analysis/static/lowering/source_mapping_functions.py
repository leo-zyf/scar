"""Source mapping functions."""

from __future__ import annotations

import ast
import dis
import inspect
from dataclasses import dataclass, replace
from types import CodeType
from scar.analysis.program_lowering import _Unsupported
from scar.analysis.static.lowering.source_call_lowering import _constant, _invoke
from scar.analysis.static.lowering.source_mapping_lowering import _transfer
from scar.ir.program.source import GapKind, SourceRole
from scar.ir.program.model import (
    ExitKind,
    AccessMode,
    Behavior,
    BlockArgument,
    CompletionResult,
    LiteralOrigin,
    OperationResult,
    ReferenceValue,
    StateVersionValue,
    TerminatorKind,
    ValueKind,
    ValueType,
    value_type,
)
from scar.ir.program.structured import PythonProtocolKind, PythonProtocolKind as P
from scar.ir.program.entry_namespaces import FrameBodyProtocolKind as K, FrameCompletionProfile
from scar.ir.program.state_addresses import (
    AddressCoverage,
    AddressEffect,
    InvokeStateOperand,
    NamespaceKey,
    ReferenceOwner,
)
from scar.ir.v2.common import SourceReference
from scar.ir.v2.semantic import OperationKind, SemanticRelation


# Definition-time function creation in a module mapping frame.
#
# The factory is one incomplete compound interface: default packing, the lazy
# annotation-function factory and SET_FUNCTION_ATTRIBUTE belong inside it. It
# returns only the function reference/world, never executes either child body,
# and owns cleanup of the default inputs on both outcomes. Decorator evaluation,
# application and the final mapping write remain separate source operations.
def _span(node):
    return node.lineno, node.end_lineno, node.col_offset, node.end_col_offset


def _unsupported(node, detail):
    raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX, node, "MAPPING_FUNCTION_" + detail)


@dataclass(frozen=True, slots=True)
class MappingFunctionPlan:
    child: dis.Instruction
    annotation: dis.Instruction | None
    defaults_constant: dis.Instruction | None
    keyword_defaults: tuple[tuple[ast.arg, ast.expr], ...]
    attributes: tuple[int, ...]
    decorator_calls: tuple[dis.Instruction, ...]
    store: dis.Instruction


def mapping_function_plan(node, parent_code):
    """Check the exact original module creation window; never execute code."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        _unsupported(node, "FUNCTION_DEFINITION_REQUIRED")
    if type(parent_code) is not CodeType or parent_code.co_name != "<module>" or parent_code.co_freevars:
        _unsupported(node, "EXACT_MODULE_CODE_REQUIRED")
    if node.type_params:
        _unsupported(node, "TYPE_PARAMETER_SCOPE_UNIMPLEMENTED")
    # dis has already incorporated EXTENDED_ARG into each following oparg;
    # prefix words are compiler encoding, not additional definition stages.
    rows = tuple(row for row in dis.get_instructions(parent_code, adaptive=False, show_caches=False)
                 if row.opname != "EXTENDED_ARG")
    own = [row for row in rows if tuple(row.positions) == _span(node)]
    children = [row for row in own if row.opname == "LOAD_CONST" and type(row.argval) is CodeType
                and row.argval.co_name == node.name and row.argval.co_qualname == node.name
                and row.argval.co_firstlineno == min([node.lineno] + [n.lineno for n in node.decorator_list])]
    if len(children) != 1:
        _unsupported(node, "ORIGINAL_CHILD_CONSTANT_SLOT_UNAVAILABLE")
    child, = children
    async_flags = inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR
    if bool(child.argval.co_flags & async_flags) != isinstance(node, ast.AsyncFunctionDef):
        _unsupported(node, "CHILD_ASYNC_KIND_DIFFERS")
    if child.argval.co_freevars:
        _unsupported(node, "CHILD_CLOSURE_BINDING_UNIMPLEMENTED")
    stores = [row for row in own if row.offset > child.offset and row.opname in ("STORE_NAME","STORE_GLOBAL")
              and row.argval == node.name]
    if len(stores) != 1:
        _unsupported(node, "ORIGINAL_MAPPING_STORE_UNAVAILABLE")
    store, = stores
    own = [row for row in own if row.offset <= store.offset]
    args = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs,
            *((node.args.vararg,) if node.args.vararg else ()),
            *((node.args.kwarg,) if node.args.kwarg else ()))
    annotated = node.returns is not None or any(arg.annotation is not None for arg in args)
    annotations = [row for row in own if row.opname == "LOAD_CONST" and type(row.argval) is CodeType
                   and row.argval.co_name == "__annotate__" and row.offset < child.offset]
    if len(annotations) != int(annotated):
        _unsupported(node, "DEFERRED_ANNOTATION_FACTORY_UNAVAILABLE")
    annotation = annotations[0] if annotations else None
    if annotation is not None and annotation.argval.co_freevars:
        _unsupported(node, "ANNOTATION_CLOSURE_BINDING_UNIMPLEMENTED")
    defaults_constant = None
    expected = []
    if node.args.defaults:
        candidates = [row for row in own if row.opname == "LOAD_CONST" and type(row.argval) is tuple]
        exact = [row for row in candidates if len(row.argval) == len(node.args.defaults)
                 and all(isinstance(n, ast.Constant) and type(n.value) is type(value) and n.value == value
                         for n, value in zip(node.args.defaults, row.argval))]
        if exact:
            if len(exact) != 1:
                _unsupported(node, "DEFAULT_CONSTANT_TUPLE_AMBIGUOUS")
            defaults_constant = exact[0]
            expected.append(("LOAD_CONST", defaults_constant.arg))
        else:
            expected.append(("BUILD_TUPLE", len(node.args.defaults)))
    keyword = tuple((arg, value) for arg, value in zip(node.args.kwonlyargs, node.args.kw_defaults)
                    if value is not None)
    if keyword:
        builds = [row for row in own if row.opname == "BUILD_MAP"]
        incremental = len(builds) == 1 and builds[0].arg == 0
        if incremental:
            expected.append(("BUILD_MAP", 0))
        for arg, _ in keyword:
            keys = [row for row in own if row.opname == "LOAD_CONST" and type(row.argval) is str
                    and row.argval == arg.arg]
            if len(keys) != 1:
                _unsupported(node, "KEYWORD_DEFAULT_KEY_UNAVAILABLE")
            expected.append(("LOAD_CONST", keys[0].arg))
            if incremental:
                expected.append(("MAP_ADD", 1))
        if not incremental:
            expected.append(("BUILD_MAP", len(keyword)))
    for load in (annotation, child):
        if load is not None:
            expected.extend((("LOAD_CONST", load.arg), ("MAKE_FUNCTION", None)))
    attributes = tuple(flag for present, flag in ((annotated, 16), (bool(keyword), 2),
                                                  (bool(node.args.defaults), 1)) if present)
    expected.extend(("SET_FUNCTION_ATTRIBUTE", flag) for flag in attributes)
    expected.append((store.opname, store.arg))
    actual = [(row.opname, row.arg) for row in own]
    if actual != expected:
        _unsupported(node, "EXACT_PACK_FACTORY_ATTRIBUTE_SEQUENCE_UNAVAILABLE")
    make = rows[rows.index(child) + 1]
    applications = [row for row in rows if make.offset < row.offset < store.offset
                    and row.opname in ("CALL", "CALL_KW", "CALL_FUNCTION_EX")]
    if (len(applications) != len(node.decorator_list)
            or any(row.opname != "CALL" or row.arg != 0 or tuple(row.positions) != _span(decorator)
                   for row, decorator in zip(applications, reversed(node.decorator_list)))):
        _unsupported(node, "DECORATOR_APPLICATION_SEQUENCE_UNAVAILABLE")
    return MappingFunctionPlan(child, annotation, defaults_constant, keyword,
                               attributes, tuple(applications), store)


def _hold(builder, cursor, node, value, role):
    builder.frame_transition(cursor, node, ((value, node, "mapping-function-held-" + role),),
                             PythonProtocolKind.FRAME_HOLD_TEMPORARY)
    builder.active_temporaries.append(value)


def lower_mapping_function(builder, node, cursor):
    """Create and bind one function through the original mapping-frame ABI."""
    from scar.analysis.static.lowering.source_module_lowering import require_root_evidence
    from scar.ir.program.entry_namespaces import FrameBodyProtocolKind as K
    from scar.ir.v2.semantic import OperationKind
    evidence = builder.evidence
    require_root_evidence(builder.unit, evidence,
                          allow_compiler_locals=getattr(builder,'supports_compiler_prologue',False))
    plan = mapping_function_plan(node, evidence.code)
    source = builder.source(node)
    declarations = [d for d in builder.by_atom.get(source.atom_id.wire, ())
                    if d.metadata.get("phase") == "definition_time"]
    bodies = [d for d in builder.by_atom.get(source.atom_id.wire, ())
              if d.metadata.get("phase") == "call_time"
              and d.kind in (OperationKind.FUNCTION, OperationKind.METHOD)]
    if len(declarations) != 1 or len(bodies) != 1:
        _unsupported(node, "ORIGINAL_DECLARATION_AND_BODY_REQUIRED")
    baseline = len(builder.active_temporaries)
    sinks = getattr(builder, "failure_sinks", None)
    def local_failure(failure, payload):
        saved = tuple(builder.active_temporaries)
        try:
            for value in reversed(builder.active_temporaries[baseline:]):
                builder.frame_transition(failure, node,
                    ((value, node, "mapping-function-exception-unwind"),),
                    PythonProtocolKind.FRAME_RELEASE_TEMPORARY)
            del builder.active_temporaries[baseline:]
            # Forward the same exception/traceback pair with the actual
            # post-release world. Temporarily remove only our own sink, so
            # the existing outer handler/C0/C1 path receives it normally.
            if not sinks or sinks[-1] is not local_failure:
                raise ValueError("mapping function failure-sink ownership changed")
            sinks.pop()
            try:
                return builder.finish_exit(failure, node, payload, ExitKind.RAISE)
            finally:
                sinks.append(local_failure)
        finally:
            builder.active_temporaries[:] = saved
    if sinks is not None:
        sinks.append(local_failure)
    try:
        decorators = []
        for index, expression in enumerate(node.decorator_list):
            cursor, value = builder.expr(expression, cursor)
            if cursor is None:
                return None
            _hold(builder, cursor, expression, value, "decorator")
            decorators.append((expression, value, index))
        inputs = []
        consumed = []
        if plan.defaults_constant is not None:
            inputs.append((_constant(builder, node, cursor, plan.defaults_constant), node,
                           "function-defaults-tuple", 0, None))
        else:
            for index, expression in enumerate(node.args.defaults):
                cursor, value = builder.expr(expression, cursor)
                if cursor is None:
                    return None
                _hold(builder, cursor, expression, value, "default")
                consumed.append((expression, value))
                inputs.append((value, expression, "function-default", index, None))
        for index, (arg, expression) in enumerate(plan.keyword_defaults):
            cursor, value = builder.expr(expression, cursor)
            if cursor is None:
                return None
            _hold(builder, cursor, expression, value, "keyword-default")
            consumed.append((expression, value))
            inputs.append((value, expression, "function-keyword-default:" + arg.arg, index, None))
        code = _constant(builder, node, cursor, plan.child)
        inputs = [(code, node, "function-code", 0, None),
                  (builder.entry_references[1], node, "function-globals", 0, None), *inputs]
        if plan.annotation is not None:
            inputs.append((_constant(builder, node, cursor, plan.annotation), node, "annotation-code", 0, None))
        # The compound boundary takes default operands, including cleanup of
        # partially packed/unpublished attributes on RAISE. Decorator holds
        # remain outside it. Both actual post-worlds stay incomplete.
        cursor, function, identity = _invoke(builder, cursor, node, tuple(inputs), K.MAKE_FUNCTION_314,
                                            tuple(consumed), result=True)
        position = next(i for i, link in enumerate(builder.operation_links) if link.operation == identity)
        builder.operation_links[position] = replace(builder.operation_links[position], semantic_operation=declarations[0].id)
        _hold(builder, cursor, node, function, "function")
        for expression, decorator, index in reversed(decorators):
            cursor, function, _ = _invoke(builder, cursor, node,
                ((decorator, expression, "decorator", index, None),
                 (function, node, "decorated-function", 0, None)),
                K.APPLY_DECORATOR_314, ((expression, decorator), (node, function)), result=True)
            _hold(builder, cursor, node, function, "function")
        # STORE_NAME consumes its stack operand on both outcomes. Remove the
        # modeled pending hold before its invoke, never release it a second
        # time on the failing mapping-write arm.
        _transfer(builder, ((node, function),), node)
        cursor = builder.store_name(node, node.name, function, cursor,opcode=plan.store.opname)
        builder.gap(GapKind.EFFECT_SUMMARY, node,
            "function construction/default attributes, decorators and mapping name write retain ordered source/world; "
            "child/annotation bodies remain deferred shared definitions, with runtime fields/identity/effects unresolved")
        return cursor
    finally:
        if sinks is not None:
            if not sinks or sinks[-1] is not local_failure:
                raise ValueError("mapping function failure-sink stack changed")
            sinks.pop()
        del builder.active_temporaries[baseline:]


# Original mapping-frame function factories, decorator chains and name writes.
#
# This consumer checks existing source/core/constant records. It neither executes
# the deferred function/annotation body nor constructs a second body authority.
# The ordinary module reader still owns whole-graph SSA, frame and C0/C1 checks.
def _need(condition, detail):
    if not condition:
        raise ValueError("mapping function: " + detail)




def validate_mapping_function(node, *, source, graph, overlay, semantic, code_evidence,
                              globals_namespace, locals_namespace):
    """Check one original definition; return its claimed factory/apply/store IDs."""
    from scar.analysis.static.lowering.source_module_lowering import require_root_evidence
    from scar.ir.program.entry_namespaces import CompilerStructuredFrameBodyOverlay, ExpandedCompilerFrameBodyOverlay, ClassConstructionFrameBodyOverlay, FrameBodyProfile
    unit = overlay.declaration.unit
    compiler_profiles = {
        CompilerStructuredFrameBodyOverlay: FrameBodyProfile.STRUCTURED_MAPPING_COMPILER_314_V1,
        ExpandedCompilerFrameBodyOverlay: FrameBodyProfile.STRUCTURED_MAPPING_COMPILER_OPS_314_V1,
        ClassConstructionFrameBodyOverlay: FrameBodyProfile.STRUCTURED_MAPPING_CLASS_314_V1,
    }
    compiler = type(overlay) in compiler_profiles and overlay.profile is compiler_profiles[type(overlay)]
    root_reference = require_root_evidence(unit, code_evidence, allow_compiler_locals=compiler)
    if compiler:
        from scar.analysis.static.lowering.source_module_compiler_lowering import validate_compiler_prologue_core
        _need(overlay.declaration.compilation == root_reference,
              'function factory must retain this exact source-compiled module root')
        prefix = validate_compiler_prologue_core(overlay, graph)
        _need(not code_evidence.code.co_cellvars or prefix is not None
              and tuple(cell.name for cell in prefix.cells) == tuple(code_evidence.code.co_cellvars),
              'module compiler-owned cells require their actual checked prologue allocations')
    plan = mapping_function_plan(node, code_evidence.code)
    _need(type(source) is SourceReference, "function definition requires its original statement SourceReference")
    atom = semantic.source_atoms.get(source.atom_id)
    _need(atom is not None and atom.reference == source and atom.kind == type(node).__name__
          and (source.path, source.fingerprint) == (unit.snapshot.path, unit.snapshot.fingerprint)
          and (source.start_line, source.end_line, source.start_column, source.end_column) == _span(node),
          "definition is not its original source atom/snapshot")
    declarations = [d for d in semantic.definitions.values() if d.code_id == source.atom_id.wire
                    and source.atom_id in d.source_atoms and d.metadata.get("phase") == "definition_time"]
    bodies = [d for d in semantic.definitions.values() if d.code_id == source.atom_id.wire
              and source.atom_id in d.source_atoms and d.metadata.get("phase") == "call_time"
              and d.kind in (OperationKind.FUNCTION, OperationKind.METHOD)]
    _need(len(declarations) == len(bodies) == 1 and bodies[0].label == node.name,
          "creation must retain the original declaration and distinct deferred body definition")
    slots = [semantic.slots[e.target.id] for e in semantic.edges.values()
             if e.source.id == declarations[0].id and e.relation is SemanticRelation.WRITES_SLOT]
    _need(len(slots) == 1 and slots[0].owner == unit.initializer and slots[0].name == node.name,
          "function name must bind its original module slot, never a child fast local")
    operations = {o.id: o for o in graph.operations}
    values = {v.id: v for v in graph.values}
    blocks = {b.id: b for r in graph.regions for b in r.blocks}
    owners = {i: r.id for r in graph.regions for b in r.blocks for i in b.operations}
    positions = {i: (b.id, n) for b in blocks.values() for n, i in enumerate(b.operations)}
    protocols = {p.operation: p for p in overlay.protocols}
    links = {r.operation: r for r in overlay.operation_links}
    uses = {(r.operation, r.use): r for r in overlay.use_links}
    root = next(r for r in graph.regions if r.id == graph.root)
    entry = blocks[root.entry]
    _need(entry.arguments[:2] == (locals_namespace, globals_namespace)
          and all(type(values[v]) is ReferenceValue and values[v].origin == BlockArgument(entry.id, i)
                  for i, v in enumerate((locals_namespace, globals_namespace))),
          "namespace inputs are not the original mapping-frame L/G entry ports")
    world = value_type(values[entry.arguments[-1]])
    ref = ValueType(ValueKind.REFERENCE)
    paired = overlay.completion_profile is FrameCompletionProfile.PROPAGATION_REF_PAIR_314_V1
    _need(paired or overlay.completion_profile is FrameCompletionProfile.RAW_EXCEPTION_REF_314_V1,
          "unknown module exception boundary")
    raised = (ref, ref, world) if paired else (ref, world)

    def invoke(identity, count, results, *, memory):
        op = operations[identity]
        expected = ((ExitKind.NORMAL, (ref,) * results + (world,)), (ExitKind.RAISE, raised))
        _need(owners[identity] == graph.root and op.callee is None and not op.effects_complete
              and op.terminator is not None and op.terminator.kind is TerminatorKind.INVOKE
              and len(op.uses) == count + 1 and op.invoke_arguments == tuple(u.id for u in op.uses)
              and all(type(values[u.value]) is ReferenceValue for u in op.uses[:-1])
              and value_type(values[op.uses[-1].value]) == world and not op.accesses
              and set(op.behaviors) == {Behavior.CONTROL, Behavior.READ, Behavior.WRITE} | ({Behavior.MEMORY} if memory else set())
              and tuple((c.kind, c.types) for c in op.completions) == expected
              and tuple(e.kind for e in op.terminator.successors) == (ExitKind.NORMAL, ExitKind.RAISE)
              and all(e.arguments == tuple(CompletionResult(i) for i in range(len(types)))
                      and tuple(value_type(values[v]) for v in blocks[e.block].arguments) == types
                      for e, (_, types) in zip(op.terminator.successors, expected)),
              "factory/call/write lost exact references, completion-owned results or post-world")
        return op, blocks[op.terminator.successors[0].block]

    def same_source(reference, expression):
        return (type(reference) is SourceReference
                and (reference.path, reference.fingerprint) == (source.path, source.fingerprint)
                and (reference.start_line, reference.end_line, reference.start_column, reference.end_column) == _span(expression))

    def expression_value(expression):
        candidates = set()
        for row in overlay.value_links:
            value = values[row.value]
            if row.role is not SourceRole.EXPRESSION or value.region != graph.root or not same_source(row.source, expression):
                continue
            if type(value) is not ReferenceValue:
                continue
            origin = value.origin
            if isinstance(expression, ast.Constant) and type(origin) is LiteralOrigin:
                if type(origin.value) is type(expression.value) and origin.value == expression.value:
                    candidates.add(value.id)
            elif type(origin) is OperationResult:
                if same_source(links[origin.operation].source, expression):
                    candidates.add(value.id)
            elif type(origin) is BlockArgument and origin.index == 0:
                producers = [op for op in graph.operations if owners[op.id] == graph.root
                             and op.terminator is not None and op.terminator.kind is TerminatorKind.INVOKE
                             and same_source(links[op.id].source, expression)
                             and any(e.kind is ExitKind.NORMAL and e.block == origin.block for e in op.terminator.successors)]
                if len(producers) == 1:
                    candidates.add(value.id)
        _need(len(candidates) == 1, "header expression lacks its unique original literal/producer NORMAL value")
        return next(iter(candidates))

    def hold(value, expression, role):
        found = []
        for p in overlay.protocols:
            if p.kind is not P.FRAME_HOLD_TEMPORARY:
                continue
            op = operations[p.operation]
            if (owners[op.id] == graph.root and len(op.uses) == 2 and op.uses[0].value == value
                    and same_source(p.source, expression)):
                link = uses.get((op.id, op.uses[0].id))
                if link is not None and link.role == "mapping-function-held-" + role and link.index == 0:
                    found.append(op.id)
        _need(len(found) == 1, "source temporary has no unique ordered hold occurrence")
        return found[0]

    constant_rows = [row for row in overlay.constants if row.source == source]
    constants = {row.operation: row for row in constant_rows}
    _need(len(constants) == len(constant_rows), "duplicate definition code-constant provenance")
    claimed_constants = set()
    def constant(value, instruction):
        origin = values[value].origin
        _need(type(origin) is OperationResult and origin.index == 0 and origin.operation in constants,
              "factory input is not its original code-table result")
        op = operations[origin.operation];record = constants[op.id]
        _need(owners[op.id] == graph.root and protocols[op.id].kind is P.READ_CODE_CONSTANT
              and protocols[op.id].source == source and links[op.id].source == source
              and record.code_key == code_evidence.code_layout_key and record.constant_path == (instruction.arg,)
              and code_evidence.code.co_consts[instruction.arg] is instruction.argval
              and op.terminator is None and op.callee is None and not op.effects_complete
              and op.behaviors == (Behavior.READ,) and len(op.uses) == len(op.accesses) == len(op.results) == 1
              and op.results[0] == value and value_type(values[op.uses[0].value]) == world
              and op.accesses[0].before == op.uses[0].id and op.accesses[0].mode is AccessMode.READ
              and op.accesses[0].after is None,
              "factory code/default slot provenance or nonexecuting read differs")
        claimed_constants.add(op.id)

    makes = [p.operation for p in overlay.protocols if p.kind is K.MAKE_FUNCTION_314 and p.source == source]
    _need(len(makes) == 1, "definition must have one original function-creation interface")
    make = makes[0];factory, normal = invoke(make, len(operations[make].uses)-1, 1, memory=True)
    _need(links[make].source == source and links[make].semantic_operation == declarations[0].id
          and links[make].opcode is None, "factory must link definition-time declaration, not execute child")
    operands = factory.uses[:-1]
    _need(len(operands) >= 2 and operands[1].value == globals_namespace,
          "factory must borrow the original module globals")
    for use, role in zip(operands[:2], ("function-code", "function-globals")):
        row = uses[(make, use.id)]
        _need(row.source == source and (row.role, row.index) == (role, 0), "factory code/globals Use differs")
    constant(operands[0].value, plan.child)
    offset = 2;header_holds = [];header_inputs = [];decorators = []
    for index, expression in enumerate(node.decorator_list):
        value = expression_value(expression)
        decorators.append((expression, value, index));header_holds.append(hold(value, expression, "decorator"))
        header_inputs.append((expression,header_holds[-1]))
    if plan.defaults_constant is not None:
        _need(offset < len(operands), "constant defaults tuple missing")
        use = operands[offset];row = uses[(make, use.id)]
        _need(row.source == source and (row.role, row.index) == ("function-defaults-tuple", 0), "defaults tuple Use differs")
        constant(use.value, plan.defaults_constant);offset += 1
    else:
        for index, expression in enumerate(node.args.defaults):
            _need(offset < len(operands), "positional default input missing")
            use = operands[offset];row = uses[(make, use.id)]
            _need(use.value == expression_value(expression) and same_source(row.source, expression)
                  and (row.role, row.index) == ("function-default", index), "positional default value/order/source differs")
            header_holds.append(hold(use.value, expression, "default"))
            header_inputs.append((expression,header_holds[-1]));offset += 1
    for index, (arg, expression) in enumerate(plan.keyword_defaults):
        _need(offset < len(operands), "keyword default input missing")
        use = operands[offset];row = uses[(make, use.id)]
        _need(use.value == expression_value(expression) and same_source(row.source, expression)
              and (row.role, row.index) == ("function-keyword-default:"+arg.arg, index), "keyword default value/order/source differs")
        header_holds.append(hold(use.value, expression, "keyword-default"))
        header_inputs.append((expression,header_holds[-1]));offset += 1
    if plan.annotation is not None:
        _need(offset < len(operands), "lazy annotation code missing")
        use = operands[offset];row = uses[(make, use.id)]
        _need(row.source == source and (row.role, row.index) == ("annotation-code", 0), "annotation code Use differs")
        constant(use.value, plan.annotation);offset += 1
    _need(offset == len(operands) and claimed_constants == set(constants), "unexpected factory inputs or unclaimed definition code reads")
    previous=None
    for expression,held in header_inputs:
        if previous is not None:
            start,end=(expression.lineno,expression.col_offset),(expression.end_lineno,expression.end_col_offset)
            for op in graph.operations:
                original=links[op.id].source
                if (owners[op.id]==graph.root and op.terminator is not None
                        and op.terminator.kind is TerminatorKind.INVOKE
                        and type(original) is SourceReference
                        and (original.path,original.fingerprint)==(source.path,source.fingerprint)
                        and start<=(original.start_line,original.start_column)
                        and (original.end_line,original.end_column)<=end):
                    _need(_world_after(op.uses[-1].value,previous,values,operations,blocks),
                          "header expression effect ran before the preceding decorator/default evaluation")
        record=operations[held]
        _need(len(record.accesses)==1 and record.accesses[0].after is not None,
              "header hold has no actual post-world")
        previous=record.accesses[0].after
    applications = {p.operation for p in overlay.protocols if p.kind is K.APPLY_DECORATOR_314 and p.source == source}
    _need(len(applications) == len(decorators), "decorator application inventory differs")
    stages = [*header_holds, make];current = normal.arguments[0]
    stages.append(hold(current, node, "function"));result_holds={current:stages[-1]};result_consumers={}
    boundaries = [(make, tuple(v for _,v,_ in decorators))]
    for expression, decorator, index in reversed(decorators):
        candidates = [identity for identity in applications
                      if tuple(u.value for u in operations[identity].uses[:-1]) == (decorator, current)]
        _need(len(candidates) == 1, "decorator chain did not consume this original decorator and preceding NORMAL result")
        identity = candidates[0];op, target = invoke(identity, 2, 1, memory=True)
        for use, expected_source, role, position in ((op.uses[0], expression, "decorator", index),
                                                   (op.uses[1], node, "decorated-function", 0)):
            row = uses[(identity,use.id)]
            _need(same_source(row.source, expected_source) and (row.role,row.index)==(role,position), "decorator operand source/order differs")
        _need(links[identity].source == source and links[identity].semantic_operation is None,
              "decorator application cannot substitute for the original child definition")
        result_consumers[current]=(identity,op.uses[1].id)
        stages.append(identity);current=target.arguments[0];stages.append(hold(current,node,"function"))
        _need(current not in result_holds,"factory/decorator NORMAL results must retain distinct original ValueIDs")
        result_holds[current]=stages[-1]
        boundaries.append((identity, tuple(v for _,v,_ in decorators[:index])))
    store_kind=K.STORE_NAME_314 if plan.store.opname=='STORE_NAME'else P.STORE_GLOBAL
    stores = [p.operation for p in overlay.protocols if p.kind is store_kind and p.source == source]
    _need(len(stores) == 1, "definition requires one final mapping name write")
    store=stores[0];write,_=invoke(store,3,0,memory=False)
    owner=locals_namespace if plan.store.opname=='STORE_NAME'else globals_namespace
    _need(write.uses[0].value==owner and write.uses[2].value==current,
          "definition binding must consume the final reference and its original compiler-selected L/G")
    assigned=uses.get((store,write.uses[2].id))
    _need(assigned is not None and assigned.source==source
          and (assigned.role,assigned.index)==("assigned-reference",0),
          "STORE_NAME must retain the exact final assigned-reference source Use")
    result_consumers[current]=(store,write.uses[2].id)
    key=values[write.uses[1].value].origin
    _need(type(key) is LiteralOrigin and type(key.value) is str and key.value==node.name, "STORE_NAME key differs from original definition")
    addresses=[r for r in graph.state_address_refinements if r.site.operation==store]
    _need(len(addresses)==1 and addresses[0].site==InvokeStateOperand(store,write.uses[-1].id)
          and addresses[0].coverage is AddressCoverage.PARTIAL and addresses[0].protocol is None
          and addresses[0].effects==(AddressEffect(ReferenceOwner(write.uses[0].id),(NamespaceKey(write.uses[1].id),),AccessMode.WRITE),),
          "function binding lost its exact incomplete mapping address")
    stages.append(store);boundaries.append((store,()))
    for value,held in result_holds.items():
        # A produced function is retained once, then consumed by the next
        # decorator CALL (or the final STORE_NAME) on both outcomes. Do not
        # skip normal-arm lifetime checks merely because the earlier hold is
        # legal: any extra downstream hold/release/forwarding is a stale use.
        expected={(held,operations[held].uses[0].id),result_consumers[value]}
        actual=[(op.id,use.id)for op in graph.operations for use in op.uses if use.value==value]
        _need(len(actual)==2 and set(actual)==expected,
              "factory/decorator result must have only its original hold and next consuming Use")
    _need(len(stages)==len(set(stages)), "definition stages reused an operation/hold")
    own=set(stages)
    for first,second in zip(stages,stages[1:]):
        _next_stage(first,second,own,operations,blocks,positions)
        earlier=operations[first]
        if earlier.terminator is not None:
            post=blocks[earlier.terminator.successors[0].block].arguments[-1]
        else:
            _need(len(earlier.accesses)==1 and earlier.accesses[0].after is not None,
                  "temporary hold has no actual post-world")
            post=earlier.accesses[0].after
        _need(_world_after(operations[second].uses[-1].value,post,values,operations,blocks),
              "later definition stage bypassed the preceding actual normal post-world")
    prefix=None
    for identity,expected in boundaries:
        released,retained=_failure_holds(operations[identity].terminator.successors[1].block,
            operations,protocols,blocks,uses,{r.operation for r in graph.cleanup_enters},source,values)
        if paired:
            _need(released==tuple(reversed(expected)),
                  "local exception unwind released different/reordered decorators or ambient references")
            outer=retained
        else:
            _need(not released and retained is not None
                  and (not expected or len(retained)>=len(expected) and retained[-len(expected):]==expected),
                  "failed definition stage lost or reordered its pending decorators")
            outer=retained[:-len(expected)] if expected else retained
        if outer is not None:
            if prefix is None:prefix=outer
            _need(outer==prefix, "factory/decorator/store failure changed the enclosing pending prefix")
    return {make,*applications,store}


def _next_stage(first, expected, own, operations, blocks, positions):
    op=operations[first];block,index=positions[first]
    pending=[(op.terminator.successors[0].block,0)] if op.terminator is not None else [(block,index+1)]
    seen=set();found=set()
    while pending:
        position=pending.pop()
        if position in seen:continue
        seen.add(position)
        _need(len(seen)<=len(operations)*2+1, "normal staging traversal exceeded graph")
        block,index=position;_need(index<len(blocks[block].operations), "normal stage falls out of block")
        identity=blocks[block].operations[index]
        if identity in own:found.add(identity);continue
        term=operations[identity].terminator
        if term is None:pending.append((block,index+1))
        elif term.kind is TerminatorKind.INVOKE:pending.append((term.successors[0].block,0))
        elif term.kind in (TerminatorKind.JUMP,TerminatorKind.BRANCH):pending.extend((e.block,0)for e in term.successors)
        else:_need(False,"normal path skips a later definition stage")
    _need(found=={expected}, "definition evaluation/application/write order changed")


def _failure_holds(block,operations,protocols,blocks,use_links,cleanup_enters,source,values):
    """Inspect actual releases before the existing frame close/cleanup entry."""
    seen=set();cursor=(block,0);released=[];post=blocks[block].arguments[-1]
    while cursor not in seen:
        seen.add(cursor);block,index=cursor
        _need(index<len(blocks[block].operations), "failure falls out of its actual unwind corridor")
        op=operations[blocks[block].operations[index]];p=protocols.get(op.id)
        local=[u for u in op.uses if (row:=use_links.get((op.id,u.id))) is not None
               and row.role=="mapping-function-exception-unwind"]
        if local:
            _need(p is not None and p.kind is P.FRAME_RELEASE_TEMPORARY and p.source==source
                  and len(local)==1 and len(op.uses)==2 and op.uses[0]==local[0]
                  and type(values[local[0].value]) is ReferenceValue
                  and use_links[(op.id,local[0].id)].source==source
                  and use_links[(op.id,local[0].id)].index==0
                  and op.terminator is None and not op.effects_complete
                  and len(op.accesses)==1 and op.accesses[0].mode is AccessMode.READ_WRITE
                  and op.accesses[0].before==op.uses[-1].id and op.accesses[0].after is not None,
                  "local exception release lost its exact reference Use/source/post-world")
            _need(_world_after(op.uses[-1].value,post,values,operations,blocks),
                  "local exception release bypassed the preceding actual raise post-world")
            post=op.accesses[0].after
            released.append(local[0].value)
        if op.id in cleanup_enters:
            _need(op.uses and _world_after(op.uses[-1].value,post,values,operations,blocks),
                  "outer cleanup entry lost the actual local-unwind world")
            return tuple(released),None
        if p is not None and p.kind in (P.FRAME_EXIT,K.FRAME_EXIT_EXCEPTION_PACKET):
            _need(op.uses and _world_after(op.uses[-1].value,post,values,operations,blocks),
                  "frame close lost the actual local-unwind world")
            tail=[operations[i]for i in blocks[block].operations[index+1:]]
            _need(len(tail)==1 and tail[0].terminator is not None and tail[0].terminator.kind is TerminatorKind.RAISE,
                  "definition failure must close the current frame before its real RAISE")
            retained=tuple(u.value for u in op.uses if (row:=use_links.get((op.id,u.id))) is not None and row.role=="held-temporary")
            return tuple(released),retained
        term=op.terminator
        if term is None:cursor=(block,index+1)
        elif term.kind in (TerminatorKind.JUMP,TerminatorKind.PENDING_DISPATCH,TerminatorKind.CLEANUP_ROUTE) and len(term.successors)==1:
            cursor=(term.successors[0].block,0)
        else:_need(False,"definition failure does not follow its original unambiguous frame close")
    _need(False,"definition failure corridor cycles")


def _world_after(value,ancestor,values,operations,blocks,seen=frozenset()):
    """Follow only actual state access/CFG operands, never names or shapes."""
    if value==ancestor:return True
    if value in seen or type(values[value]) is not StateVersionValue:return False
    origin=values[value].origin;before=[]
    if type(origin) is OperationResult:
        op=operations[origin.operation];own={u.id:u.value for u in op.uses}
        before=[own[a.before]for a in op.accesses if a.after==value and a.before in own]
    elif type(origin) is BlockArgument:
        for op in operations.values():
            for edge in getattr(op.terminator,"successors",()):
                if edge.block!=origin.block:continue
                argument=edge.arguments[origin.index]
                if type(argument) is CompletionResult:
                    signatures=[s for s in op.completions if s.kind is edge.kind]
                    if (len(signatures)!=1 or argument.index!=len(signatures[0].types)-1
                            or value_type(values[value])!=signatures[0].types[argument.index]):return False
                    before.extend(u.value for u in op.uses if type(values[u.value]) is StateVersionValue)
                else:
                    own={u.id:u.value for u in op.uses}
                    if argument not in own:return False
                    before.append(own[argument])
    return bool(before) and all(_world_after(v,ancestor,values,operations,blocks,seen|{value})for v in before)


__all__ = [
    'MappingFunctionPlan',
    'mapping_function_plan',
    'lower_mapping_function',
    'validate_mapping_function',
]
