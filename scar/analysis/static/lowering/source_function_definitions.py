"""Source function definitions."""

from __future__ import annotations

import ast
import dis
from dataclasses import replace
from types import CodeType
from scar.analysis.program_lowering import _Unsupported
from scar.analysis.source_language import (
    supports_cpython_compiler,
    function_exception_types,
    require_source_tree,
)
from scar.analysis.static.lowering.source_call_lowering import _constant, _invoke
from scar.analysis.static.lowering.source_heap_lowering import expression_for_heap_target
from scar.analysis.structured_lowering import _transfer
from scar.ir.program.environments import EnvironmentRole
from scar.ir.program.ids import OperationID, ValueID
from scar.ir.program.model import (
    AccessMode,
    Behavior,
    ExitKind,
    Operation,
    OperationResult,
    StateAccess,
    StateVersionValue,
    CompletionResult,
    LocalBindingValue,
    ReferenceValue,
    TerminatorKind,
    ValueKind,
    ValueType,
    value_type,
)
from scar.ir.program.source import GapKind, SourceRole
from scar.ir.program.state_addresses import (
    AddressEffect,
    CellContent,
    ReferenceOwner,
    AddressCoverage,
    InvokeStateOperand,
)
from scar.ir.program.structured import PythonProtocolKind as K
from scar.ir.program.bindings import BindingState
from scar.ir.program.source_language import (
    SourceLanguageCapability as C,
    NESTED_FUNCTION_PROTOCOLS,
)
from scar.ir.v2.semantic import OperationKind, SemanticRelation


# Original nested function creation; child code is referenced, never executed.
def span(node):
    return node.lineno, node.end_lineno, node.col_offset, node.end_col_offset


def unsupported(node, detail):
    raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX, node, 'NESTED_FUNCTION_' + detail)


def compiler_plan(node, code):
    """Exact original creation window; no compilation, target import or eval."""
    if not isinstance(node, ast.FunctionDef):
        unsupported(node, 'ASYNC_DEFINITION_UNIMPLEMENTED')
    if node.decorator_list:
        unsupported(node, 'DECORATOR_APPLICATION_UNIMPLEMENTED')
    if node.type_params:
        unsupported(node, 'TYPE_PARAMETER_SCOPE_UNIMPLEMENTED')
    if any(x is not None for x in node.args.kw_defaults):
        unsupported(node, 'KEYWORD_DEFAULTS_UNIMPLEMENTED')
    rows = tuple(dis.get_instructions(code))
    own = [r for r in rows if tuple(r.positions) == span(node)]
    children = [r for r in own if r.opname == 'LOAD_CONST' and type(r.argval) is CodeType
                and r.argval.co_name == node.name and r.argval.co_firstlineno == node.lineno
                and r.argval.co_qualname == code.co_qualname + '.<locals>.' + node.name]
    if len(children) != 1:
        unsupported(node, 'ORIGINAL_CHILD_CODE_OCCURRENCE_UNAVAILABLE')
    child = children[0]
    stores = [r for r in own if r.offset > child.offset and r.opname in ('STORE_FAST', 'STORE_DEREF', 'STORE_GLOBAL')
              and r.argval == node.name]
    if len(stores) != 1:
        unsupported(node, 'ORIGINAL_NAME_STORE_OCCURRENCE_UNAVAILABLE')
    stop = stores[0]
    own = [r for r in own if r.offset <= stop.offset]
    expected, stages = [], []
    def constant(row, role):
        expected.append(('LOAD_CONST', row.arg)); stages.append(('constant', role, row))
    defaults = None
    if node.args.defaults:
        candidates = [r for r in own if r.opname == 'LOAD_CONST' and type(r.argval) is tuple
                      and len(r.argval) == len(node.args.defaults)]
        exact = [r for r in candidates if all(isinstance(n, ast.Constant) and type(n.value) is type(v) and n.value == v
                                            for n, v in zip(node.args.defaults, r.argval))]
        if exact:
            if len(exact) != 1:
                unsupported(node, 'DEFAULT_TUPLE_CONSTANT_AMBIGUOUS')
            defaults = exact[0]; constant(defaults, 'defaults')
        else:
            expected.append(('BUILD_TUPLE', len(node.args.defaults)))
            stages.append(('tuple', 'defaults', tuple(node.args.defaults)))
    args = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
    annotated = (node.returns is not None or any(a.annotation is not None for a in args)
                 or any(a is not None and a.annotation is not None for a in (node.args.vararg, node.args.kwarg)))
    annotation = None
    if annotated:
        annotations = [r for r in own if r.opname == 'LOAD_CONST' and type(r.argval) is CodeType
                       and r.argval.co_name == '__annotate__' and r.offset < child.offset
                       and r.argval.co_firstlineno == node.lineno]
        if len(annotations) != 1:
            unsupported(node, 'DEFERRED_ANNOTATION_CODE_UNAVAILABLE')
        annotation = annotations[0]
    def function(load, role):
        names = tuple(load.argval.co_freevars)
        if names:
            expected.extend(('LOAD_FAST_BORROW', name) for name in names)
            expected.append(('BUILD_TUPLE', len(names)))
            stages.append(('tuple', role + '-closure', names))
        constant(load, role + '-code')
        expected.append(('MAKE_FUNCTION', None)); stages.append(('make', role, None))
        if names:
            expected.append(('SET_FUNCTION_ATTRIBUTE', 8)); stages.append(('set', role + '-closure', None))
    if annotation is not None:
        function(annotation, 'annotation')
    function(child, 'function')
    if annotation is not None:
        expected.append(('SET_FUNCTION_ATTRIBUTE', 16)); stages.append(('set', 'annotation', None))
    if node.args.defaults:
        expected.append(('SET_FUNCTION_ATTRIBUTE', 1)); stages.append(('set', 'defaults', None))
    expected.append((stop.opname, node.name)); stages.append(('bind', stop.opname, None))
    def matches(row, token):
        name, arg = token
        return row.opname == name and (row.argval == arg if name.startswith(('LOAD_FAST', 'STORE_')) else row.arg == arg)
    if len(own) != len(expected) or not all(matches(r, t) for r, t in zip(own, expected)):
        unsupported(node, 'EXACT_COMPILER_CREATION_SEQUENCE_UNAVAILABLE')
    return tuple(stages), child, annotation


def _hold(builder, cursor, node, reference, role):
    builder.frame_transition(cursor, node, ((reference, node, 'function-held-' + role),), K.FRAME_HOLD_TEMPORARY)
    builder.active_temporaries.append(reference)


def _code_reference(builder, node, cursor, instruction, role):
    value = _constant(builder, node, cursor, instruction)
    identity = builder.values_by_id[value].origin.operation
    old = builder.operation_records[identity]
    marked = replace(old, label='original function ' + role + ' reference')
    builder.operations[builder.operations.index(old)] = marked
    builder.operation_records[identity] = marked
    return value


def lower_function_definition(builder, node, cursor):
    evidence = builder.language_options.code_evidence
    if not supports_cpython_compiler(evidence):
        unsupported(node, 'READONLY_CPYTHON_3147_CODE_REQUIRED')
    evidence.require_definition(builder.snapshot, builder.function_definition.id, builder.source(builder.functions[builder.entry]))
    stages, child, annotation = compiler_plan(node, evidence.code)
    baseline = len(builder.active_temporaries)
    try:
        # Defaults are real source operations. Code/creation staging is the
        # contract for one semantic interface, not a repeated child-body graph.
        default_constant = next((row for action, role, row in stages if action == 'constant' and role == 'defaults'), None)
        defaults = []
        if default_constant is None:
            for index, source in enumerate(node.args.defaults):
                cursor, value = expression_for_heap_target(builder, source, cursor)
                if cursor is None:
                    return None
                _hold(builder, cursor, source, value, 'default')
                defaults.append((value, source, 'function-default', index, None))
        else:
            value = _code_reference(builder, node, cursor, default_constant, 'defaults')
            defaults.append((value, node, 'function-defaults-tuple', 0, None))
        main_code = _code_reference(builder, node, cursor, child, 'code')
        inputs = [(main_code, node, 'function-code', 0, None),
                  (builder.environment_references[EnvironmentRole.GLOBALS_NAMESPACE], node, 'function-globals', 0, None)]
        inputs.extend(defaults)
        for role, load in (('function', child), ('annotation', annotation)):
            if load is None:
                continue
            if role == 'annotation':
                inputs.append((_code_reference(builder, node, cursor, load, 'annotation code'), node, 'annotation-code', 0, None))
            for index, name in enumerate(load.argval.co_freevars):
                if name not in builder.cell_references:
                    unsupported(node, 'CAPTURE_HAS_NO_ACTUAL_PARENT_CELL')
                inputs.append((builder.cell_references[name], node, role + '-closure-cell', index, None))
        consumed = tuple((source, value) for value, source, _, _, _ in defaults) if default_constant is None else ()
        # The boundary owns construction, attribute initialization and unwinding
        # of its unpublished temporaries. Closure cells/globals are borrowed
        # identities, never eager cell-content reads. Both post-worlds remain
        # incomplete; no runtime callable/code identity or purity is granted.
        cursor, function, identity = _invoke(builder, cursor, node, tuple(inputs),
            K.MAKE_NESTED_FUNCTION, consumed, result=True)
        old = builder.operation_records[identity]
        marked = replace(old, label='create function ' + node.name)
        builder.operations[builder.operations.index(old)] = marked
        builder.operation_records[identity] = marked
        declarations = [d for d in builder.by_atom.get(builder.source(node).atom_id.wire, ())
                        if d.metadata.get('phase') == 'definition_time']
        if len(declarations) != 1:
            unsupported(node, 'ORIGINAL_DEFINITION_TIME_DECLARATION_UNAVAILABLE')
        i = next(i for i, link in enumerate(builder.operation_links) if link.operation == identity)
        builder.operation_links[i] = replace(builder.operation_links[i], semantic_operation=declarations[0].id)
        _hold(builder, cursor, node, function, 'function')
        store = stages[-1][1]
        if store == 'STORE_FAST' and node.name in builder.locals:
            builder.frame_transition(cursor, node, ((cursor.env[node.name], node, 'previous-binding'),
                (function, node, 'assigned-reference')), K.FRAME_BIND)
            builder.define_binding(node.name, cursor, node, function)
            _transfer(builder, ((node, function),), node)
        elif store == 'STORE_DEREF' and node.name in builder.cell_references:
            cursor, _, identity = _invoke(builder, cursor, node,
                ((builder.cell_references[node.name], node, 'definition-cell', 0, None),
                 (function, node, 'defined-function', 0, None)),
                K.STORE_NESTED_FUNCTION_CELL, ((node, function),), raise_consumed=())
            operation = builder.operation_records[identity]
            builder._addresses(identity, (AddressEffect(ReferenceOwner(operation.uses[0].id), (CellContent(),), AccessMode.WRITE),))
        else:
            unsupported(node, 'NAME_BINDING_INTERFACE_UNIMPLEMENTED')
        builder.gap(GapKind.CODE_VERSION, node, 'nested function creation references original code; runtime function/code instance identity remains unproved')
        builder.gap(GapKind.EFFECT_SUMMARY, node, 'definition-time construction and ordered captures are connected; saved builtins resolution, finalizers and runtime annotation/call effects remain unproved')
        return cursor
    finally:
        del builder.active_temporaries[baseline:]


# Check nested function creation as a source-level interface, not bytecode IR.
def need(value, detail):
    if not value:
        raise ValueError('function definition: ' + detail)




def refspan(source):
    return source.start_line, source.end_line, source.start_column, source.end_column


def validate_function_definitions(model, graph, interface, metadata, semantic, language,
                                  *, source_tree=None, options=None, structural_only=False):
    protocols = {r.operation: r for r in model.protocols}
    selected = {key for key, row in protocols.items() if row.kind.value in NESTED_FUNCTION_PROTOCOLS}
    if not selected:
        return set()
    need(C.NESTED_FUNCTION_314_V1 in language.capabilities and isinstance(source_tree, ast.Module),
         'explicit capability and exact retained original AST required')
    operations = {r.id: r for r in graph.operations}
    values = {r.id: r for r in graph.values}
    blocks = {r.id: r for region in graph.regions for r in region.blocks}
    links = {r.operation: r for r in metadata.operation_links}
    sources = {(r.operation, r.use): r for r in metadata.use_links}
    constants = {r.operation: r for r in language.constants}
    addresses = {r.site.operation: r for r in graph.state_address_refinements}
    environments = [r for r in graph.environments if r.region == interface.region]
    need(len(environments) == 1, 'one original parent environment required')
    environment = environments[0]
    globals_ref = next(r.reference for r in environment.members if r.role is EnvironmentRole.GLOBALS_NAMESPACE)
    cell_values = [blocks[operations[r.operation].terminator.successors[0].block].arguments[0]
                   for r in model.protocols if r.kind is K.MAKE_CELL]
    free_values = [r.reference for r in environment.members if r.role is EnvironmentRole.CAPTURED_CELL]
    available_cells = set(cell_values + free_values)
    cells_by_name = None
    if not structural_only:
        from scar.analysis.environment_lowering import captured_cell_slot
        evidence = options.code_evidence
        need(supports_cpython_compiler(evidence),
             'independent pinned readonly parent compiler witness required')
        transient = {r.name for r in language.compiler_locals if r.category == 'COMPREHENSION_TRANSIENT'}
        names = tuple(n for n in evidence.code.co_cellvars if n not in transient)
        need(len(names) == len(cell_values) and len(evidence.code.co_freevars) == len(free_values),
             'actual parent cells differ from the complete original compiler layout')
        cells_by_name = dict(zip(names, cell_values))
        for name, member in zip(evidence.code.co_freevars, environment.members[5:]):
            need(member.cell_slot == captured_cell_slot(interface.definition, name), 'free cell name/position differs')
            cells_by_name[name] = member.reference
    world = value_type(values[interface.world])
    reference = ValueType(ValueKind.REFERENCE)
    covered = set()

    def invoke(identity, count, results):
        op = operations[identity]
        need(op.callee is None and op.terminator is not None and op.terminator.kind is TerminatorKind.INVOKE
             and not op.effects_complete and len(op.uses) == count + 1
             and op.invoke_arguments == tuple(u.id for u in op.uses)
             and all(type(values[u.value]) is ReferenceValue for u in op.uses[:-1])
             and type(values[op.uses[-1].value]) is StateVersionValue
             and values[op.uses[-1].value].location == values[interface.world].location
             and tuple((s.kind, s.types) for s in op.completions) ==
                 ((ExitKind.NORMAL, (reference,) * results + (world,)),
                  (ExitKind.RAISE, function_exception_types(language, reference, world)))
             and tuple(s.kind for s in op.terminator.successors) == (ExitKind.NORMAL, ExitKind.RAISE)
             and all(e.arguments == tuple(CompletionResult(i) for i in range(len(s.types)))
                     for e, s in zip(op.terminator.successors, op.completions)),
             'creation/write lost its original references, world or normal/raise interface')
        # Canonical source INVOKE carries world as its final Use and as a
        # completion-specific result, not a redundant ordinary StateAccess.
        need(not op.accesses and set(op.behaviors) == {Behavior.CONTROL, Behavior.READ, Behavior.WRITE, Behavior.MEMORY},
             'creation/write must use its exact incomplete completion-specific world interface')
        return op, blocks[op.terminator.successors[0].block]

    makes = [r.operation for r in model.protocols if r.kind is K.MAKE_NESTED_FUNCTION]
    need(len({protocols[i].source for i in makes}) == len(makes), 'source definition has duplicate creation interfaces')
    for identity in makes:
        covered.add(identity)
        source = protocols[identity].source
        nodes = [n for n in require_source_tree(source_tree, language.source, source) if isinstance(n, ast.FunctionDef)]
        need(len(nodes) == 1, 'creation has no exact original FunctionDef AST')
        node = nodes[0]
        need(not node.decorator_list and not node.type_params and not any(x is not None for x in node.args.kw_defaults),
             'unsupported decorator/type-parameter/keyword-default form entered a body')
        declarations = [r for r in semantic.definitions.values() if r.code_id == source.atom_id.wire
                        and source.atom_id in r.source_atoms and r.metadata.get('phase') == 'definition_time']
        bodies = [r for r in semantic.definitions.values() if r.code_id == source.atom_id.wire
                  and source.atom_id in r.source_atoms and r.metadata.get('phase') == 'call_time'
                  and r.kind in (OperationKind.FUNCTION, OperationKind.METHOD)]
        need(len(declarations) == len(bodies) == 1 and bodies[0].label == node.name
             and links[identity].source == source and links[identity].semantic_operation == declarations[0].id
             and links[identity].opcode is None, 'creation must link declaration, never execute the child body')
        slots = [semantic.slots[e.target.id] for e in semantic.edges.values()
                 if e.source.id == declarations[0].id and e.relation is SemanticRelation.WRITES_SLOT]
        need(len(slots) == 1 and slots[0].owner == interface.definition and slots[0].name == node.name,
             'definition name has no exact original parent lexical slot')
        operation, normal = invoke(identity, len(operations[identity].uses) - 1, 1)
        uses = [sources[(identity, u.id)] for u in operation.uses[:-1]]
        refs = [u.value for u in operation.uses[:-1]]
        need([(u.role, u.index) for u in uses[:2]] == [('function-code', 0), ('function-globals', 0)]
             and refs[1] == globals_ref, 'function factory changed original code/globals inputs')
        own_constants = {r.operation: r for r in language.constants if r.source == source}
        by_reference = {operations[i].results[0]: r for i, r in own_constants.items()}
        need(refs[0] in by_reference, 'function code is not the original code-table reference')
        annotated = (node.returns is not None or any(a.annotation is not None for a in
            (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)) or
            any(a is not None and a.annotation is not None for a in (node.args.vararg, node.args.kwarg)))
        roles = [(u.role, u.index) for u in uses]
        cursor = 2
        if node.args.defaults:
            if cursor < len(uses) and uses[cursor].role == 'function-defaults-tuple':
                need(uses[cursor].index == 0 and refs[cursor] in by_reference and uses[cursor].source == source,
                     'default tuple lacks original constant provenance')
                cursor += 1
            else:
                for index, default in enumerate(node.args.defaults):
                    need(cursor < len(uses) and roles[cursor] == ('function-default', index)
                         and refspan(uses[cursor].source) == span(default), 'default value/source order differs')
                    cursor += 1
        captures = {}
        for role in ('function', 'annotation'):
            if role == 'annotation':
                if not annotated:
                    break
                need(cursor < len(uses) and roles[cursor] == ('annotation-code', 0)
                     and refs[cursor] in by_reference, 'deferred annotation code reference missing')
                cursor += 1
            captured = []
            while cursor < len(uses) and uses[cursor].role == role + '-closure-cell':
                need(uses[cursor].index == len(captured) and uses[cursor].source == source
                     and refs[cursor] in available_cells, 'capture substituted cell contents or changed original cell order')
                captured.append(refs[cursor]); cursor += 1
            captures[role] = tuple(captured)
        need(cursor == len(uses), 'unexpected or duplicated creation input')
        used_constants = {refs[i] for i, u in enumerate(uses) if u.role in
                          ('function-code', 'annotation-code', 'function-defaults-tuple')}
        need(used_constants == set(by_reference), 'orphan/foreign function constant table read')
        if not structural_only:
            from scar.analysis.static.lowering.source_function_definitions import compiler_plan
            stages, child, annotation = compiler_plan(node, evidence.code)
            required = {'function-code': child.arg}
            if annotation is not None:
                required['annotation-code'] = annotation.arg
            default = next((row for action, role, row in stages if action == 'constant' and role == 'defaults'), None)
            if default is not None:
                required['function-defaults-tuple'] = default.arg
            for role, index in required.items():
                selected_uses = [i for i, u in enumerate(uses) if u.role == role]
                need(len(selected_uses) == 1 and by_reference[refs[selected_uses[0]]].constant_path == (index,),
                     'creation input differs from exact original compiler constant index')
            need(('function-defaults-tuple' in required) == any(u.role == 'function-defaults-tuple' for u in uses),
                 'default tuple packing differs from original compiler')
            for role, load in (('function', child), ('annotation', annotation)):
                if load is not None:
                    need(captures[role] == tuple(cells_by_name[n] for n in load.argval.co_freevars),
                         'creation did not capture exact compiler-ordered parent cell identities')
        function = normal.arguments[0]
        following = [operations[i] for i in normal.operations]
        need(len(following) >= 2 and protocols.get(following[0].id) is not None
             and protocols[following[0].id].kind is K.FRAME_HOLD_TEMPORARY
             and following[0].uses[0].value == function, 'created function is not retained until its actual name write')
        binding = following[1]
        marker = protocols.get(binding.id)
        need(marker is not None and marker.source == source, 'name binding lost its exact FunctionDef source')
        if marker.kind is K.FRAME_BIND:
            need(len(binding.uses) == 3 and type(values[binding.uses[0].value]) is LocalBindingValue
                 and values[binding.uses[0].value].slot == slots[0].slot_id and binding.uses[1].value == function
                 and len(following) >= 3, 'fast-local definition binding changed its original slot/function')
            definition = following[2]
            need(len(definition.results) == 1 and type(values[definition.results[0]]) is LocalBindingValue
                 and values[definition.results[0]].slot == slots[0].slot_id
                 and len(definition.uses) == 1 and definition.uses[0].value == function
                 and any(r.value == definition.results[0] and r.state is BindingState.BOUND for r in graph.binding_definitions),
                 'created function was not the original bound SSA payload')
        else:
            need(marker.kind is K.STORE_NESTED_FUNCTION_CELL, 'definition has an unsupported name-write interface')
            covered.add(binding.id)
            op, _ = invoke(binding.id, 2, 0)
            need(op.uses[0].value in available_cells and op.uses[1].value == function,
                 'cell binding changed original cell/function inputs')
            if not structural_only:
                need(cells_by_name[node.name] == op.uses[0].value, 'definition stored into a different compiler cell')
            row = addresses.get(op.id)
            need(row is not None and row.site == InvokeStateOperand(op.id, op.uses[-1].id)
                 and row.coverage is AddressCoverage.PARTIAL and row.protocol is None
                 and row.effects == (AddressEffect(ReferenceOwner(op.uses[0].id), (CellContent(),), AccessMode.WRITE),),
                 'definition cell write lacks exact original owner/content/post-world')
    need(covered == selected, 'orphan function creation/name-write protocol')
    return selected
