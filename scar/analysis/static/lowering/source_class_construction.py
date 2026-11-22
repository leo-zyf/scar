"""Source class construction."""

from __future__ import annotations

import ast
import dis
from dataclasses import dataclass, replace
from types import CodeType
from scar.analysis.program_lowering import _Unsupported
from scar.analysis.static.lowering.source_call_lowering import _constant, _invoke
from scar.analysis.structured_lowering import _transfer
from scar.analysis.source_language import supports_cpython_compiler
from scar.ir.program.class_construction import (
    ClassConstruction,
    ClassDecoratorApplication,
    ImplicitCallArgument,
    ImplicitCallPorts,
    CLASS_CONSTRUCTION_PROTOCOLS,
)
from scar.ir.program.environments import EnvironmentRole
from scar.ir.program.functions import ArgumentKind
from scar.ir.program.model import (
    AccessMode,
    ExitKind,
    Behavior,
    BlockArgument,
    CompletionResult,
    LiteralOrigin,
    LocalBindingValue,
    OperationResult,
    ReferenceValue,
    StateVersionValue,
    TerminatorKind,
    UseRole,
    ValueKind,
    ValueType,
    value_type,
)
from scar.ir.program.source import GapKind, SourceRole
from scar.ir.program.state_addresses import (
    AddressEffect,
    CellContent,
    NamespaceKey,
    ReferenceOwner,
    AddressCoverage,
    InvokeStateOperand,
)
from scar.ir.program.structured import PythonProtocolKind as K
from scar.ir.v2.semantic import OperationKind
from scar.trace.code_snapshot import code_key
from collections import defaultdict
from scar.analysis.static.compilation import StaticCodeEvidence
from scar.analysis.static.lowering.source_mapping_functions import _next_stage, _world_after
from scar.ir.program.bindings import BindingState
from scar.ir.semantics_v2 import Opcode
from scar.ir.v2.common import SourceReference


# ClassDef construction in the original parent frame, without target execution.
#
# This profile expands the compiler-known parent protocol. The called builder
# remains an actual builtin-namespace read; the shared suite is code provenance,
# not an unconditional body execution or a replacement for the class result.
def _span(node):
    return node.lineno, node.end_lineno, node.col_offset, node.end_col_offset


def _unsupported(node, detail):
    raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX, node, 'CLASS_CONSTRUCTION_' + detail)


@dataclass(frozen=True, slots=True)
class ClassConstructionPlan:
    lookup: dis.Instruction
    child: dis.Instruction
    name: dis.Instruction
    build: dis.Instruction
    store: dis.Instruction
    closure: tuple[str, ...]
    keyword_names: dis.Instruction | None
    decorator_calls: tuple[dis.Instruction, ...]


def class_construction_plan(node, parent_code):
    """Match the original direct compiler window; never synthesize ast.Call."""
    if type(node) is not ast.ClassDef or type(parent_code) is not CodeType:
        _unsupported(node, 'ORIGINAL_CLASS_AND_PARENT_CODE_REQUIRED')
    if node.type_params or any(isinstance(base, ast.Starred) for base in node.bases) or any(
            keyword.arg is None for keyword in node.keywords):
        _unsupported(node, 'TYPE_PARAMETER_OR_EXPANDED_HEADER_UNIMPLEMENTED')
    instructions = tuple(i for i in dis.get_instructions(parent_code, adaptive=False, show_caches=False)
                         if i.opname != 'EXTENDED_ARG')
    own = [i for i in instructions if tuple(i.positions) == _span(node)]
    lookups = [i for i in own if i.opname == 'LOAD_BUILD_CLASS']
    children = [i for i in own if i.opname == 'LOAD_CONST' and type(i.argval) is CodeType
                and i.argval.co_name == node.name]
    if len(lookups) != 1 or len(children) != 1:
        _unsupported(node, 'EXACT_BUILDER_LOOKUP_AND_CHILD_CODE_REQUIRED')
    lookup, child = lookups[0], children[0]
    stores = [i for i in own if i.offset > child.offset and i.opname in (
        'STORE_NAME', 'STORE_FAST', 'STORE_DEREF', 'STORE_GLOBAL') and i.argval == node.name]
    if len(stores) != 1:
        _unsupported(node, 'EXACT_FRAME_BINDING_REQUIRED')
    store = stores[0]
    own = [i for i in own if lookup.offset <= i.offset <= store.offset]
    makes = [i for i in own if i.opname == 'MAKE_FUNCTION']
    names = [i for i in own if i.opname == 'LOAD_CONST' and type(i.argval) is str and i.argval == node.name]
    builds = [i for i in own if i.opname in ('CALL', 'CALL_KW')]
    if len(makes) != 1 or len(names) != 1 or len(builds) != 1:
        _unsupported(node, 'EXACT_FACTORY_NAME_AND_BUILDER_CALL_REQUIRED')
    build, name = builds[0], names[0]
    if build.arg != 2 + len(node.bases) + len(node.keywords):
        _unsupported(node, 'COMPILER_BUILDER_ARITY_DIFFERS')
    keyword_names = None
    if node.keywords:
        tables = [i for i in own if i.opname == 'LOAD_CONST' and type(i.argval) is tuple
                  and i.argval == tuple(k.arg for k in node.keywords) and name.offset < i.offset < build.offset]
        if build.opname != 'CALL_KW' or len(tables) != 1:
            _unsupported(node, 'COMPILER_KEYWORD_TABLE_DIFFERS')
        keyword_names = tables[0]
    elif build.opname != 'CALL':
        _unsupported(node, 'COMPILER_POSITIONAL_CALL_DIFFERS')
    closure = tuple(child.argval.co_freevars)
    expected = [('LOAD_BUILD_CLASS', None), ('PUSH_NULL', None)]
    if closure:
        expected.extend(('LOAD_FAST_BORROW', n) for n in closure)
        expected.append(('BUILD_TUPLE', len(closure)))
    expected.extend((('LOAD_CONST', child.arg), ('MAKE_FUNCTION', None)))
    if closure: expected.append(('SET_FUNCTION_ATTRIBUTE', 8))
    expected.append(('LOAD_CONST', name.arg))
    # Header expression instructions have their own source positions; the
    # remaining class-positioned instructions are precisely keyword/call/store.
    if keyword_names is not None: expected.append(('LOAD_CONST', keyword_names.arg))
    expected.extend(((build.opname, build.arg), (store.opname, node.name)))
    def matches(row, token):
        op, arg = token
        return row.opname == op and (arg is None or (row.argval == arg if op in (
            'LOAD_FAST_BORROW', 'STORE_NAME', 'STORE_FAST', 'STORE_DEREF', 'STORE_GLOBAL') else row.arg == arg))
    if len(own) != len(expected) or not all(matches(i, e) for i, e in zip(own, expected)):
        _unsupported(node, 'EXACT_PARENT_COMPILER_SEQUENCE_UNAVAILABLE')
    applications = [i for i in instructions if build.offset < i.offset < store.offset
                    and i.opname in ('CALL', 'CALL_KW', 'CALL_FUNCTION_EX')]
    if (len(applications) != len(node.decorator_list) or any(i.opname != 'CALL' or i.arg != 0
            or tuple(i.positions) != _span(d) for i, d in zip(applications, reversed(node.decorator_list)))):
        _unsupported(node, 'EXACT_DECORATOR_APPLICATION_SEQUENCE_UNAVAILABLE')
    return ClassConstructionPlan(lookup, child, name, build, store, closure, keyword_names, tuple(applications))


def _hold(builder, cursor, node, reference):
    builder.frame_transition(cursor, node, ((reference, node, 'class-held-reference'),), K.FRAME_HOLD_TEMPORARY)
    builder.active_temporaries.append(reference)


def _ports(builder, identity, arguments, *, keyword_names=False):
    operation = builder.operation_records[identity]
    normal, raised = operation.terminator.successors
    normal_args = builder.block_arguments[normal.block]
    raised_args = builder.block_arguments[raised.block]
    actuals = tuple(ImplicitCallArgument(operation.uses[index + 1].id, builder.source(node), kind,
        index, keyword) for index, (node, kind, keyword) in enumerate(arguments))
    return ImplicitCallPorts(identity, operation.uses[0].id, actuals, operation.uses[-1].id,
        normal.block, normal_args[0], normal_args[-1], raised.block, *raised_args,
        keyword_names=operation.uses[-2].id if keyword_names else None)


def lower_class_construction(builder, node, cursor):
    evidence = builder.language_options.code_evidence
    if not supports_cpython_compiler(evidence) or not hasattr(evidence, 'to_reference'):
        _unsupported(node, 'STATIC_CPYTHON_3147_PARENT_REQUIRED')
    plan = class_construction_plan(node, evidence.code)
    source = builder.source(node)
    definitions = [d for d in builder.by_atom.get(source.atom_id.wire, ())
                   if d.kind is OperationKind.OPAQUE and d.label == 'class ' + node.name and d.output_slots]
    if len(definitions) != 1:
        _unsupported(node, 'ORIGINAL_CLASS_DEFINITION_REQUIRED')
    definition = definitions[0].id
    suite = replace(evidence.to_reference(), source=source, definition=definition,
        code_path=evidence.code_path + (plan.child.arg,), code_layout_key=code_key(plan.child.argval))
    module = hasattr(builder, 'entry_references')
    globals_reference = (builder.entry_references[1] if module else
        builder.environment_references[EnvironmentRole.GLOBALS_NAMESPACE])
    builtins_reference = (builder.entry_references[2] if module else
        builder.environment_references[EnvironmentRole.BUILTINS_NAMESPACE])
    closure_references = getattr(builder, 'cell_references', {})
    if any(name not in closure_references for name in plan.closure):
        _unsupported(node, 'ORIGINAL_PARENT_CLOSURE_CELL_UNAVAILABLE')
    baseline, first, entry = len(builder.active_temporaries), len(builder.operations), cursor.block
    # A local failure sink releases only this helper's still-owned references,
    # then forwards the same pair and post-release world to its enclosing one.
    sinks = getattr(builder, 'failure_sinks', None)
    def failed(failure, payload):
        saved = tuple(builder.active_temporaries)
        try:
            for value in reversed(builder.active_temporaries[baseline:]):
                builder.frame_transition(failure, node, ((value, node, 'mapping-function-exception-unwind' if module else 'function-exception-unwind'),), K.FRAME_RELEASE_TEMPORARY)
            del builder.active_temporaries[baseline:]
            if not sinks or sinks[-1] is not failed:
                raise ValueError('class construction failure ownership changed')
            sinks.pop()
            try: return builder.finish_exit(failure, node, payload, ExitKind.RAISE)
            finally: sinks.append(failed)
        finally: builder.active_temporaries[:] = saved
    if sinks is not None: sinks.append(failed)
    try:
        decorators = []
        for index, expression in enumerate(node.decorator_list):
            cursor, value = builder.expr(expression, cursor)
            if cursor is None: return None
            _hold(builder, cursor, expression, value); decorators.append((index, expression, value))
        key = builder.literal(node, '__build_class__')
        cursor, selected, lookup = _invoke(builder, cursor, node,
            ((builtins_reference, node, 'class-builder-owner', 0, None), (key, node, 'class-builder-key', 0, None)),
            K.LOAD_BUILD_CLASS_314, result=True)
        operation = builder.operation_records[lookup]
        builder._addresses(lookup, (AddressEffect(ReferenceOwner(operation.uses[0].id),
            (NamespaceKey(operation.uses[1].id),), AccessMode.READ),))
        _hold(builder, cursor, node, selected)
        code = _constant(builder, node, cursor, plan.child); _hold(builder, cursor, node, code)
        factory_inputs = ((code, node, 'class-body-code', 0, None),
            (globals_reference, node, 'class-body-globals', 0, None)) + tuple(
            (closure_references[name], node, 'class-body-closure-cell', index, None)
            for index, name in enumerate(plan.closure))
        cursor, function, factory = _invoke(builder, cursor, node, factory_inputs, K.MAKE_CLASS_BODY_314,
            ((node, code),), result=True)
        factory_operation = builder.operation_records[factory]
        _hold(builder, cursor, node, function)
        name = _constant(builder, node, cursor, plan.name); _hold(builder, cursor, node, name)
        inputs = [(selected, node, 'callee', 0, None), (function, node, 'argument', 0, None),
                  (name, node, 'argument', 1, None)]
        arguments = [(node, ArgumentKind.POSITIONAL, None), (node, ArgumentKind.POSITIONAL, None)]
        consumed = [(node, selected), (node, function), (node, name)]
        for index, expression in enumerate(node.bases):
            cursor, value = builder.expr(expression, cursor)
            if cursor is None: return None
            _hold(builder, cursor, expression, value)
            inputs.append((value, expression, 'argument', index + 2, None))
            arguments.append((expression, ArgumentKind.POSITIONAL, None)); consumed.append((expression, value))
        for index, keyword in enumerate(node.keywords):
            cursor, value = builder.expr(keyword.value, cursor)
            if cursor is None: return None
            _hold(builder, cursor, keyword.value, value)
            inputs.append((value, keyword.value, 'keyword:' + keyword.arg, index, None))
            arguments.append((keyword.value, ArgumentKind.KEYWORD, keyword.arg)); consumed.append((keyword.value, value))
        if plan.keyword_names is not None:
            table = _constant(builder, node, cursor, plan.keyword_names); _hold(builder, cursor, node, table)
            inputs.append((table, node, 'class-keyword-names', 0, None)); consumed.append((node, table))
        cursor, result, call = _invoke(builder, cursor, node, tuple(inputs), K.BUILD_CLASS_CALL_314,
            tuple(consumed), result=True)
        from scar.analysis.static.lowering.source_heap_lowering import _associate
        from scar.ir.semantics_v2 import Opcode
        _associate(builder, call, node, definitions[0], Opcode.OPAQUE)
        call_ports = _ports(builder, call, arguments, keyword_names=plan.keyword_names is not None)
        _hold(builder, cursor, node, result)
        applications = []
        for index, expression, decorator in reversed(decorators):
            cursor, result, identity = _invoke(builder, cursor, expression,
                ((decorator, expression, 'callee', 0, None), (result, node, 'argument', 0, None)),
                K.APPLY_CLASS_DECORATOR_314, ((expression, decorator), (node, result)), result=True)
            applications.append(ClassDecoratorApplication(index, builder.source(expression),
                _ports(builder, identity, ((node, ArgumentKind.POSITIONAL, None),))))
            _hold(builder, cursor, node, result)
        if plan.store.opname == 'STORE_NAME' and module:
            _transfer(builder, ((node, result),), node)
            before = len(builder.operations); cursor = builder.store_name(node, node.name, result, cursor)
            binding = next(op for op in builder.operations[before:] if any(
                row.operation == op.id and row.kind.value == 'STORE_NAME_314' for row in builder.protocols)).id
            binding_use = builder.operation_records[binding].uses[2].id
        elif plan.store.opname == 'STORE_FAST' and node.name in builder.locals:
            before = len(builder.operations)
            builder.frame_transition(cursor, node, ((cursor.env[node.name], node, 'previous-binding'),
                (result, node, 'assigned-reference')), K.FRAME_BIND)
            binding = builder.operations[before].id; binding_use = builder.operation_records[binding].uses[1].id
            builder.define_binding(node.name, cursor, node, result)
            _transfer(builder, ((node, result),), node)
        elif plan.store.opname in ('STORE_DEREF', 'STORE_GLOBAL'):
            cell = plan.store.opname == 'STORE_DEREF'
            if cell and node.name not in closure_references: _unsupported(node, 'BINDING_CELL_UNAVAILABLE')
            inputs = (((closure_references[node.name], node, 'class-binding-owner', 0, None),) if cell else
                ((globals_reference, node, 'class-binding-owner', 0, None),
                 (builder.literal(node, node.name), node, 'class-binding-key', 0, None)))
            cursor, _, binding = _invoke(builder, cursor, node,
                inputs + ((result, node, 'assigned-reference', 0, None),),
                K.STORE_CLASS_CELL_314 if cell else K.STORE_CLASS_GLOBAL_314,
                ((node, result),), raise_consumed=() if cell else None)
            operation = builder.operation_records[binding]; binding_use = operation.uses[-2].id
            path = (CellContent(),) if cell else (NamespaceKey(operation.uses[1].id),)
            builder._addresses(binding, (AddressEffect(ReferenceOwner(operation.uses[0].id), path, AccessMode.WRITE),))
        else: _unsupported(node, 'COMPILER_BINDING_PROFILE_MISMATCH')
        record = ClassConstruction(definition, source, builder.function_context, entry, cursor.block, suite,
            lookup, factory, factory_operation.uses[0].id, factory_operation.uses[1].id,
            tuple(u.id for u in factory_operation.uses[2:-1]), function, call_ports, tuple(applications),
            binding, binding_use, plan.store.opname, result, tuple(op.id for op in builder.operations[first:]))
        builder.class_constructions.append(record)
        builder.gap(GapKind.EFFECT_SUMMARY, node,
            'parent class construction ports are connected; selected builder, prepared namespace, shared-suite execution, '
            'metaclass/hooks/classcell and complete effects remain explicit obligations')
        return cursor
    finally:
        if sinks is not None:
            if not sinks or sinks[-1] is not failed: raise ValueError('class failure sink stack changed')
            sinks.pop()
        del builder.active_temporaries[baseline:]


# Check parent ClassDef ports against source, compiler layout and core SSA.
#
# No target import, class creation, metaclass, decorator or suite is executed.
# Checking a factory's shared code reference does not establish suite entry.
def _need(condition, detail):
    if not condition:
        raise ValueError('class construction validation: ' + detail)




def _refspan(source):
    return source.start_line, source.end_line, source.start_column, source.end_column


def _kind(marker):
    value = marker.kind if marker is not None else None
    return getattr(value, 'value', value)


def validate_class_constructions(*, graph, metadata, records, protocols, code_evidence,
                                source_tree, semantic, interface=None, module_overlay=None,
                                structural_only=False, constants=()):
    """Return only claimed class-specific protocols; other validators own leaves."""
    actual = {identity for identity, marker in protocols.items()
              if _kind(marker) in CLASS_CONSTRUCTION_PROTOCOLS}
    _need(type(records) is tuple and all(type(row) is ClassConstruction for row in records),
          'class table must contain exact typed original-port records')
    if not records:
        _need(not actual, 'class protocol has no original construction record')
        return set()
    _need(type(code_evidence) is StaticCodeEvidence and isinstance(source_tree, ast.Module),
          'retained AST and independent static parent compiler layout are required')
    _need(type(structural_only) is bool, 'verification mode must be explicit')
    parent = code_evidence.to_reference()
    source_records = module_overlay if module_overlay is not None else metadata
    operations = {op.id: op for op in graph.operations}
    values = {v.id: v for v in graph.values}
    blocks = {b.id: b for r in graph.regions for b in r.blocks}
    regions = {r.id: r for r in graph.regions}
    owners = {identity: r.id for r in graph.regions for b in r.blocks for identity in b.operations}
    positions = {identity: (b.id, n) for b in blocks.values() for n, identity in enumerate(b.operations)}
    links = {r.operation: r for r in source_records.operation_links}
    uses = {(r.operation, r.use): r for r in source_records.use_links}
    addresses = {r.site.operation: r for r in graph.state_address_refinements
                 if type(r.site) is InvokeStateOperand}
    definitions_by_atom = defaultdict(set)
    for definition in semantic.definitions.values():
        if definition.output_slots and definition.metadata.get('role') != 'binding':
            for atom in definition.source_atoms:
                if definition.code_id == atom.wire:
                    definitions_by_atom[atom].add(definition.id)
    normal_producers = defaultdict(list)
    for operation in graph.operations:
        if operation.terminator is not None and operation.terminator.kind is TerminatorKind.INVOKE:
            for edge in operation.terminator.successors:
                if edge.kind is ExitKind.NORMAL:
                    normal_producers[edge.block].append(operation)
    module = module_overlay is not None
    constant_records = tuple(module_overlay.constants) if module else constants
    constant_table = {r.operation: r for r in constant_records}
    _need(len(constant_table) == len(constant_records), 'duplicate original constant-table records')
    if module:
        from scar.ir.program.source_subjects import FrameKind, SourceCompilationFrameRoot
        roots = [r for r in graph.frame_roots if type(r) is SourceCompilationFrameRoot
                 and r.kind is FrameKind.MODULE]
        _need(len(roots) == 1 and roots[0].code == parent, 'module parent compiler root differs')
        mapping = roots[0].entry
        globals_ref, builtins_ref = mapping.globals_namespace, mapping.builtins_namespace
        cells_by_name = {}
        entry_world = mapping.world
    else:
        _need(interface is not None and interface.definition == parent.definition,
              'function construction must retain its original parent interface')
        envs = [e for e in graph.environments if e.region == interface.region]
        _need(len(envs) == 1, 'function construction needs its sole parent environment')
        env = envs[0]
        globals_ref = next(m.reference for m in env.members if m.role is EnvironmentRole.GLOBALS_NAMESPACE)
        builtins_ref = next(m.reference for m in env.members if m.role is EnvironmentRole.BUILTINS_NAMESPACE)
        free = [m for m in env.members if m.role is EnvironmentRole.CAPTURED_CELL]
        _need(len(free) == len(code_evidence.code.co_freevars), 'parent free-cell count differs')
        cells_by_name = {name: member.reference for name, member in zip(code_evidence.code.co_freevars, free)}
        lexical = {slot.name for slot in semantic.slots.values() if slot.owner == interface.definition}
        own_names = tuple(n for n in code_evidence.code.co_cellvars if n in lexical)
        own_cells = [op for op in graph.operations if owners[op.id] == interface.region
                     and _kind(protocols.get(op.id)) == 'MAKE_CELL']
        _need(len(own_cells) == len(own_names), 'parent owned-cell layout needs an exact compiler mapping')
        for name, op in zip(own_names, own_cells):
            normal = next(e for e in op.terminator.successors if e.kind is ExitKind.NORMAL)
            cells_by_name[name] = blocks[normal.block].arguments[0]
        entry_world = interface.world
    world_type = value_type(values[entry_world]); ref_type = ValueType(ValueKind.REFERENCE)
    covered, all_extents = set(), set()

    def same(reference, node, source):
        return (type(reference) is SourceReference and reference.path == source.path
                and reference.fingerprint == source.fingerprint and _refspan(reference) == _span(node))

    def role(op, use, name, index, source):
        link = uses.get((op.id, use.id))
        _need(link is not None and link.role == name and link.index == index and link.source == source,
              'original operand role, ordinal or source differs')

    def invoke(identity, count, results, region, source, kind):
        op = operations.get(identity); marker = protocols.get(identity)
        expected = ((ExitKind.NORMAL, (ref_type,) * results + (world_type,)),
                    (ExitKind.RAISE, (ref_type, ref_type, world_type)))
        _need(op is not None and owners[identity] == region and marker is not None
              and _kind(marker) == kind and marker.source == source and links[identity].source == source
              and op.callee is None and not op.effects_complete and not op.accesses
              and set(op.behaviors) == {Behavior.CONTROL, Behavior.READ, Behavior.WRITE, Behavior.MEMORY}
              and op.terminator is not None and op.terminator.kind is TerminatorKind.INVOKE
              and len(op.uses) == count + 1 and op.invoke_arguments == tuple(u.id for u in op.uses)
              and all(type(values[u.value]) is ReferenceValue and u.role is UseRole.REFERENCE for u in op.uses[:-1])
              and value_type(values[op.uses[-1].value]) == world_type
              and tuple((c.kind, c.types) for c in op.completions) == expected
              and tuple(e.kind for e in op.terminator.successors) == (ExitKind.NORMAL, ExitKind.RAISE)
              and all(e.arguments == tuple(CompletionResult(i) for i in range(len(types)))
                  and tuple(value_type(values[v]) for v in blocks[e.block].arguments) == types
                  for e, (_, types) in zip(op.terminator.successors, expected)),
              'class protocol lost its original parent references or normal/raise/world completions')
        return op, blocks[op.terminator.successors[0].block], blocks[op.terminator.successors[1].block]

    def implicit(record, count, region, source, kind):
        record.__post_init__()
        op, normal, raised = invoke(record.operation, count, 1, region, source, kind)
        keyword = (record.keyword_names,) if getattr(record, 'keyword_names', None) is not None else ()
        _need((record.callee, *(a.use for a in record.arguments), *keyword, record.world)
              == tuple(u.id for u in op.uses)
              and (record.normal, record.result, record.normal_world) == (normal.id, *normal.arguments)
              and (record.raised, record.exception, record.traceback, record.raised_world) == (raised.id, *raised.arguments),
              'implicit-call ports substituted another actual or completion result')
        _need(links[op.id].opcode is (Opcode.OPAQUE if kind == 'BUILD_CLASS_CALL_314' else None),
              'implicit class call must retain its original ClassDef source category, never ast.Call')
        return op

    def address(op, effects):
        row = addresses.get(op.id)
        _need(row is not None and row.site == InvokeStateOperand(op.id, op.uses[-1].id)
              and row.coverage is AddressCoverage.PARTIAL and row.protocol is None
              and row.effects == effects, 'class namespace/cell address changed original owner, key or mode')

    for record in records:
        record.__post_init__()
        source = record.source
        _need(source.path == metadata.source.path and source.fingerprint == metadata.source.fingerprint,
              'construction source differs from retained parent snapshot')
        nodes = [n for n in ast.walk(source_tree) if type(n) is ast.ClassDef and same(source, n, source)]
        _need(len(nodes) == 1, 'class record must select one exact original ClassDef')
        node = nodes[0]
        atom = semantic.source_atoms.get(source.atom_id); definition = semantic.definitions.get(record.definition)
        _need(atom is not None and atom.reference == source and atom.kind == 'ClassDef'
              and definition is not None and definition.kind is OperationKind.OPAQUE
              and definition.label == 'class ' + node.name and definition.output_slots
              and definition.code_id == source.atom_id.wire and source.atom_id in definition.source_atoms,
              'class definition is not the original SG ClassDef output declaration')
        from scar.analysis.static.lowering.source_class_construction import class_construction_plan
        plan = class_construction_plan(node, code_evidence.code)
        expected_suite = replace(parent, source=source, definition=record.definition,
            code_path=parent.code_path + (plan.child.arg,), code_layout_key=code_key(plan.child.argval))
        _need(record.suite == expected_suite and plan.closure == tuple(plan.child.argval.co_freevars),
              'shared suite source or exact lexical compiler path differs')
        _need(record.region == graph.root and record.region in regions and record.entry in blocks
              and record.normal in blocks and all(owners.get(i) == record.region for i in record.operations)
              and not all_extents.intersection(record.operations), 'class extent crosses another parent or construction')
        all_extents.update(record.operations)
        emitted = tuple(op.id for op in graph.operations)
        start, stop = emitted.index(record.operations[0]), emitted.index(record.operations[-1])
        _need(emitted[start:stop + 1] == record.operations
              and positions[record.operations[0]][0] == record.entry,
              'class extent omitted/reordered generated operations or changed its entry block')
        region = record.region; stages = []; boundaries = []; pending = []
        own = set(record.operations)
        class_holds = []

        def expression(expression):
            candidates = set()
            for link in source_records.value_links:
                if link.role is not SourceRole.EXPRESSION or not same(link.source, expression, source):
                    continue
                value = values[link.value]
                if value.region != region or type(value) is not ReferenceValue: continue
                origin = value.origin
                definitions = definitions_by_atom[link.source.atom_id]
                if type(origin) is LiteralOrigin and isinstance(expression, ast.Constant):
                    if type(origin.value) is type(expression.value) and origin.value == expression.value:
                        candidates.add(value.id)
                elif (type(origin) is OperationResult and origin.operation in own
                      and same(links[origin.operation].source, expression, source)
                      and links[origin.operation].semantic_operation in definitions):
                    candidates.add(value.id)
                elif type(origin) is BlockArgument and origin.index == 0:
                    if any(op.id in own and op.terminator is not None and op.terminator.kind is TerminatorKind.INVOKE
                           and same(links[op.id].source, expression, source)
                           and links[op.id].semantic_operation in definitions
                           and any(e.kind is ExitKind.NORMAL and e.block == origin.block for e in op.terminator.successors)
                           for op in normal_producers[origin.block]): candidates.add(value.id)
            _need(len(candidates) == 1, 'header expression lacks one original normal reference producer')
            return next(iter(candidates))

        def hold(value, expression):
            found = [op for op in operations.values() if op.id in own
                     and _kind(protocols.get(op.id)) == 'FRAME_HOLD_TEMPORARY'
                     and len(op.uses) == 2 and op.uses[0].value == value
                     and (link := uses.get((op.id, op.uses[0].id))) is not None
                     and link.role == 'class-held-reference' and same(link.source, expression, source)]
            _need(len(found) == 1, 'class temporary lacks its unique source hold')
            op = found[0]
            _need(op.terminator is None and not op.effects_complete and len(op.accesses) == 1
                  and op.accesses[0].before == op.uses[-1].id and op.accesses[0].after is not None
                  and op.results == (op.accesses[0].after,), 'class hold changed world transition')
            class_holds.append(op.id); stages.append(op.id); pending.append(value)

        def consume(references):
            refs = tuple(references)
            _need(not refs or tuple(pending[-len(refs):]) == refs, 'class operand ownership is not nested')
            if refs: del pending[-len(refs):]

        def constant(value, instruction):
            origin = values[value].origin
            _need(type(origin) is OperationResult and origin.index == 0 and origin.operation in own,
                  'class code/name input is not its original constant-table result')
            op = operations[origin.operation]
            table_record = constant_table.get(op.id)
            _need(_kind(protocols.get(op.id)) == 'READ_CODE_CONSTANT' and links[op.id].source == source
                  and op.results == (value,) and code_evidence.code.co_consts[instruction.arg] is instruction.argval
                  and table_record is not None and table_record.source == source
                  and table_record.constant_path == (instruction.arg,)
                  and table_record.code_key == code_evidence.code_layout_key,
                  'class code/name constant differs from the original parent table')
            stages.append(op.id)

        decorators = []
        for expression_node in node.decorator_list:
            value = expression(expression_node); decorators.append(value); hold(value, expression_node)
        lookup, normal, _ = invoke(record.builder_lookup, 2, 1, region, source, 'LOAD_BUILD_CLASS_314')
        _need(lookup.uses[0].value == builtins_ref, 'builder lookup must use the parent saved builtins')
        key = values[lookup.uses[1].value]
        _need(type(key.origin) is LiteralOrigin and key.origin.value == '__build_class__', 'builder key differs')
        role(lookup, lookup.uses[0], 'class-builder-owner', 0, source)
        role(lookup, lookup.uses[1], 'class-builder-key', 0, source)
        address(lookup, (AddressEffect(ReferenceOwner(lookup.uses[0].id), (NamespaceKey(lookup.uses[1].id),), AccessMode.READ),))
        stages.append(lookup.id); boundaries.append((lookup, tuple(pending))); selected = normal.arguments[0]; hold(selected, node)
        factory, normal, _ = invoke(record.factory, 2 + len(plan.closure), 1, region, source, 'MAKE_CLASS_BODY_314')
        _need((record.factory_code, record.factory_globals, *record.factory_closure)
              == tuple(u.id for u in factory.uses[:-1]) and factory.uses[1].value == globals_ref
              and normal.arguments[0] == record.body_function,
              'class factory replaced code, parent globals, captured cells or its original function result')
        role(factory, factory.uses[0], 'class-body-code', 0, source)
        role(factory, factory.uses[1], 'class-body-globals', 0, source)
        for index, (name, use) in enumerate(zip(plan.closure, factory.uses[2:-1])):
            _need(name in cells_by_name and use.value == cells_by_name[name], 'class closure captured a different parent cell')
            role(factory, use, 'class-body-closure-cell', index, source)
        code = factory.uses[0].value; constant(code, plan.child); hold(code, node)
        consume((code,)); stages.append(factory.id); boundaries.append((factory, tuple(pending)))
        hold(record.body_function, node)
        call = implicit(record.builder_call, 3 + len(node.bases) + len(node.keywords)
                        + int(plan.keyword_names is not None), region, source, 'BUILD_CLASS_CALL_314')
        actuals = record.builder_call.arguments
        _need(len(actuals) == 2 + len(node.bases) + len(node.keywords) and call.uses[0].value == selected
              and call.uses[1].value == record.body_function
              and links[call.id].semantic_operation == record.definition,
              'builder invocation lost its actual selected callable/body function or original class definition')
        role(call, call.uses[0], 'callee', 0, source)
        name_value = call.uses[2].value; constant(name_value, plan.name); hold(name_value, node)
        for ordinal, argument in enumerate(actuals[:2]):
            _need(argument.kind is ArgumentKind.POSITIONAL and argument.source == source,
                  'builder body/name must be original implicit positional arguments')
            role(call, call.uses[ordinal + 1], 'argument', ordinal, source)
        for ordinal, expression_node in enumerate(node.bases):
            argument = actuals[ordinal + 2]; use = call.uses[ordinal + 3]
            _need(argument.kind is ArgumentKind.POSITIONAL and same(argument.source, expression_node, source)
                  and use.value == expression(expression_node), 'base argument evaluation/source/order differs')
            role(call, use, 'argument', ordinal + 2, argument.source); hold(use.value, expression_node)
        for ordinal, keyword in enumerate(node.keywords):
            argument = actuals[2 + len(node.bases) + ordinal]; use = call.uses[3 + len(node.bases) + ordinal]
            _need(argument.kind is ArgumentKind.KEYWORD and argument.keyword == keyword.arg
                  and same(argument.source, keyword.value, source) and use.value == expression(keyword.value),
                  'class keyword argument value/name/source/order differs')
            role(call, use, 'keyword:' + keyword.arg, ordinal, argument.source); hold(use.value, keyword.value)
        if plan.keyword_names is not None:
            _need(record.builder_call.keyword_names == call.uses[-2].id, 'keyword name table Use differs')
            constant(call.uses[-2].value, plan.keyword_names); hold(call.uses[-2].value, node)
            role(call, call.uses[-2], 'class-keyword-names', 0, source)
        else: _need(record.builder_call.keyword_names is None, 'positional class call acquired a keyword table')
        consume(u.value for u in call.uses[:-1]); stages.append(call.id); boundaries.append((call, tuple(pending)))
        result = record.builder_call.result; hold(result, node)
        _need(len(record.decorators) == len(node.decorator_list), 'decorator application count differs')
        for application in record.decorators:
            expression_node = node.decorator_list[application.index]
            _need(same(application.source, expression_node, source), 'decorator application source differs')
            op = implicit(application.call, 2, region, application.source, 'APPLY_CLASS_DECORATOR_314')
            _need(op.uses[0].value == decorators[application.index] and op.uses[1].value == result
                  and len(application.call.arguments) == 1
                  and application.call.arguments[0].kind is ArgumentKind.POSITIONAL
                  and application.call.arguments[0].source == source, 'decorator chain substituted the previous class result')
            role(op, op.uses[0], 'callee', 0, application.source)
            role(op, op.uses[1], 'argument', 0, source)
            consume((op.uses[0].value, result)); stages.append(op.id); boundaries.append((op, tuple(pending)))
            result = application.call.result; hold(result, node)
        _need(result == record.result and record.binding_kind == plan.store.opname,
              'final class result or compiler name-binding protocol differs')
        binding = operations[record.binding]
        _need(record.binding in own and links[binding.id].source == source
              and any(u.id == record.binding_use and u.value == result for u in binding.uses),
              'class binding lost the actual last decorator/builder result')
        if record.binding_kind == 'STORE_FAST':
            _need(not module and _kind(protocols.get(binding.id)) == 'FRAME_BIND'
                  and len(binding.uses) == 3 and binding.uses[1].id == record.binding_use
                  and type(values[binding.uses[0].value]) is LocalBindingValue,
                  'fast-local class binding changed its prior local token or assigned reference')
            slot = values[binding.uses[0].value].slot
            _need(semantic.slots[slot].name == node.name and semantic.slots[slot].owner == interface.definition,
                  'class binding targets another lexical local')
            _, offset = positions[binding.id]
            following = blocks[positions[binding.id][0]].operations[offset + 1:]
            _need(bool(following), 'class fast-local binding lacks its bound SSA definition')
            bound = operations[following[0]]
            _need(len(bound.results) == 1 and type(values[bound.results[0]]) is LocalBindingValue
                  and values[bound.results[0]].slot == slot and len(bound.uses) == 1 and bound.uses[0].value == result
                  and any(b.value == bound.results[0] and b.state is BindingState.BOUND for b in graph.binding_definitions),
                  'class result did not become the original local bound payload')
            consume((result,))
        elif record.binding_kind == 'STORE_NAME':
            _need(module and _kind(protocols.get(binding.id)) == 'STORE_NAME_314'
                  and binding.uses[0].value == mapping.locals_namespace, 'class STORE_NAME targets another namespace')
            key = values[binding.uses[1].value]
            _need(type(key.origin) is LiteralOrigin and key.origin.value == node.name
                  and binding.uses[2].id == record.binding_use, 'class STORE_NAME key/assigned Use differs')
            consume((result,)); boundaries.append((binding, tuple(pending)))
        else:
            cell = record.binding_kind == 'STORE_DEREF'
            invoke(binding.id, 2 if cell else 3, 0, region, source,
                   'STORE_CLASS_CELL_314' if cell else 'STORE_CLASS_GLOBAL_314')
            if cell:
                _need(node.name in cells_by_name and binding.uses[0].value == cells_by_name[node.name],
                      'class name write changed its compiler-selected cell')
                path = (CellContent(),)
                boundaries.append((binding, tuple(pending)))
            else:
                _need(binding.uses[0].value == globals_ref, 'class global name write targets another globals namespace')
                key = values[binding.uses[1].value]
                _need(type(key.origin) is LiteralOrigin and key.origin.value == node.name, 'class global key differs')
                path = (NamespaceKey(binding.uses[1].id),)
            consume((result,))
            if not cell: boundaries.append((binding, tuple(pending)))
            address(binding, (AddressEffect(ReferenceOwner(binding.uses[0].id), path, AccessMode.WRITE),))
        stages.append(binding.id)
        if binding.terminator is not None:
            _need(record.normal == binding.terminator.successors[0].block,
                  'class statement normal block differs from the actual binding completion')
        else:
            _need(record.normal == positions[binding.id][0],
                  'class fast binding normal block differs from the actual parent block')
        _need(not pending, 'class construction left an untransferred owned result')
        all_class_holds = {identity for identity in own if _kind(protocols.get(identity)) == 'FRAME_HOLD_TEMPORARY'
                           and any((link := uses.get((identity, u.id))) is not None and link.role == 'class-held-reference'
                                   for u in operations[identity].uses)}
        _need(set(class_holds) == all_class_holds and len(stages) == len(set(stages)),
              'orphan, duplicated or overlapping class stages')
        for first, second in zip(stages, stages[1:]):
            _next_stage(first, second, set(stages), operations, blocks, positions)
            op = operations[first]
            if op.terminator is not None:
                post = blocks[op.terminator.successors[0].block].arguments[-1]
            else:
                after = [a.after for a in op.accesses if a.after is not None]
                post = after[0] if after else op.uses[-1].value
            _need(_world_after(operations[second].uses[-1].value, post, values, operations, blocks),
                  'class stage bypassed the preceding actual post-world')
        for op, expected in boundaries:
            _check_failure(op.terminator.successors[1].block, expected, source,
                           operations, values, blocks, protocols, uses, graph,
                           'mapping-function-exception-unwind' if module else 'function-exception-unwind')
        claimed = {record.builder_lookup, record.factory, record.builder_call.operation,
                   *(a.call.operation for a in record.decorators)}
        if record.binding_kind in ('STORE_DEREF', 'STORE_GLOBAL'): claimed.add(record.binding)
        _need(not covered.intersection(claimed), 'class protocol claimed by more than one construction')
        covered.update(claimed)
    _need(covered == actual, 'orphan class-specific construction protocol')
    return covered


def _check_failure(block, expected, source, operations, values, blocks, protocols, uses, graph, unwind_role):
    """Follow exact local releases before frame close or the outer cleanup sink."""
    cursor, seen, released = (block, 0), set(), []
    post = blocks[block].arguments[-1]
    cleanup = {r.operation for r in graph.cleanup_enters}
    while cursor not in seen:
        seen.add(cursor); current, index = cursor
        _need(index < len(blocks[current].operations), 'class exception corridor fell out of its original block')
        op = operations[blocks[current].operations[index]]; kind = _kind(protocols.get(op.id))
        worlds = [u for u in op.uses if type(values[u.value]) is StateVersionValue]
        _need((bool(worlds) and all(u.value == post for u in worlds))
              or (not worlds and op.effects_complete and not op.accesses
                  and op.behaviors == (Behavior.CONTROL,) and op.terminator is None),
              'class exception operation bypassed its actual incoming world')
        local = [u for u in op.uses if (link := uses.get((op.id, u.id))) is not None
                 and link.role == unwind_role and link.source == source]
        if local:
            _need(kind == 'FRAME_RELEASE_TEMPORARY' and len(local) == 1 and len(op.uses) == 2
                  and local[0] == op.uses[0] and op.terminator is None and len(op.accesses) == 1
                  and op.accesses[0].mode is AccessMode.READ_WRITE and op.accesses[0].before == op.uses[-1].id
                  and op.accesses[0].after is not None,
                  'class local exception release changed reference order or post-world')
            released.append(local[0].value)
        after = [a.after for a in op.accesses if a.after is not None]
        _need(len(after) <= 1, 'class exception corridor changed the coarse world partition')
        if after: post = after[0]
        if op.id in cleanup or kind in ('FRAME_EXIT', 'FUNCTION_FRAME_EXIT_EXCEPTION_PACKET',
                'FRAME_EXIT_EXCEPTION_PACKET', 'FUNCTION_HANDLED_CONTEXT_ENTER'):
            retained = tuple(u.value for u in op.uses if (link := uses.get((op.id, u.id))) is not None
                             and link.role == 'held-temporary')
            if released:
                _need(tuple(released) == tuple(reversed(expected)), 'class failure released different or reordered owned inputs')
            elif expected:
                _need(len(retained) >= len(expected) and retained[-len(expected):] == expected,
                      'class failure lost its still-owned reference suffix')
            return
        term = op.terminator
        if term is None: cursor = current, index + 1
        elif term.kind in (TerminatorKind.JUMP, TerminatorKind.PENDING_DISPATCH, TerminatorKind.CLEANUP_ROUTE) and len(term.successors) == 1:
            edge = term.successors[0]; actuals = {u.id: u.value for u in op.uses}
            target_worlds = [(formal, actual) for formal, actual in zip(blocks[edge.block].arguments, edge.arguments)
                             if type(values[formal]) is StateVersionValue]
            _need(len(target_worlds) == 1 and actuals.get(target_worlds[0][1]) == post,
                  'class exception edge changed the actual world-to-formal handoff')
            # At a shared handler join only this original edge is selected.
            # Other exception predecessors need not descend from this throw.
            post = target_worlds[0][0]; cursor = edge.block, 0
        else: _need(False, 'class exception lost its original single cleanup/handler continuation')
    _need(False, 'class exception cleanup corridor cycles')
