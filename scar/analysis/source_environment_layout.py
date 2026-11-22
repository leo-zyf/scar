"""Source environment layout."""

from __future__ import annotations

import ast
import hashlib
import platform
import symtable
import sys
from dataclasses import dataclass
from types import CodeType
from scar.ir.v2.semantic import OperationDefinition, OperationKind, SemanticRelation
from scar.ir.program.environments import EnvironmentRole
from scar.ir.program.functions import BodyLoweringProfile
from scar.ir.program.model import (
    Behavior,
    BlockArgument,
    CompletionResult,
    ExitKind,
    LiteralOrigin,
    LocalBindingValue,
    ReferenceValue,
    StateVersionValue,
    TerminatorKind,
    ValueKind,
    ValueType,
)
from scar.ir.program.state_addresses import (
    AddressCoverage,
    AddressEffect,
    CellContent,
    Element,
    InvokeStateOperand,
    NamespaceKey,
    ObjectField,
    ReferenceOwner,
)
from scar.ir.program.state_modes import AccessMode
from scar.ir.program.structured import PythonProtocolKind as Kind
from scar.ir.semantics_v2 import Opcode
from scar.ir.program.source_language import (
    STARRED_DISPLAY_PROTOCOLS,
    SourceLanguageCapability as DisplayCapability,
)


# Compiler-derived closure layout for a pinned source definition.
#
# Compilation constructs code objects but never evaluates the module, imports,
# decorators, defaults, annotations or function bodies.  Only immutable primitive
# projections are cached on the prepared context; no CodeType or runtime callable
# escapes the query.  This witnesses source layout, not a live function binding.
_CACHE_ATTRIBUTE = "_scar_compiler_environment_layout_v1"
_MAX_SOURCE_BYTES = 4_000_000
_MAX_CODES = 100_000


@dataclass(frozen=True, slots=True)
class CompilerEnvironmentLayout:
    freevars: tuple[str, ...]
    cellvars: tuple[str, ...]
    qualified_name: str
    first_line: int

    def __post_init__(self):
        for name in ("freevars", "cellvars"):
            values = getattr(self, name)
            if (type(values) is not tuple or any(type(value) is not str or not value for value in values)
                    or len(set(values)) != len(values)):
                raise ValueError(name + " must be a tuple of unique names")
        if set(self.freevars) & set(self.cellvars):
            raise ValueError("a binding cannot be both free and locally owned")
        if type(self.qualified_name) is not str or not self.qualified_name:
            raise ValueError("compiler qualified name must be nonempty text")
        if type(self.first_line) is not int or self.first_line < 1:
            raise ValueError("compiler first line must be positive")


def _functions(tree):
    """Yield lexical Python qualnames, distinguishing function/class scopes."""
    def visit(node, prefix):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualified = prefix + node.name
            yield node, qualified
            for statement in node.body:
                yield from visit(statement, qualified + ".<locals>.")
            return
        if isinstance(node, ast.ClassDef):
            for statement in node.body:
                yield from visit(statement, prefix + node.name + ".")
            return
        for child in ast.iter_child_nodes(node):
            yield from visit(child, prefix)
    yield from visit(tree, "")


def _table_kind(table):
    kind = table.get_type()
    return getattr(kind, "value", kind)


def _symbol_projection(table):
    """Keep classification sets separate from the compiler's ordered layout."""
    symbols = table.get_symbols()
    free = tuple(symbol.get_name() for symbol in symbols if symbol.is_free() or symbol.is_nonlocal())
    local = tuple(symbol.get_name() for symbol in symbols if symbol.is_local())
    # A local cell must serve a child scope.  Descendant demand is a consistency
    # check only: the compiler, rather than lexical name searches, owns layout.
    descendant_free = set()
    def visit(child):
        descendant_free.update(symbol.get_name() for symbol in child.get_symbols()
                               if symbol.is_free() or symbol.is_nonlocal())
        for grandchild in child.get_children():
            visit(grandchild)
    for child in table.get_children():
        visit(child)
    return (table.get_name(), table.get_lineno(), free, local, tuple(sorted(descendant_free)))


def _project(prepared, cache_key):
    code = compile(prepared.text, str(prepared.path), "exec", flags=0,
                   dont_inherit=True, optimize=0)
    rows = []
    def visit(current):
        if len(rows) >= _MAX_CODES:
            raise ValueError("compiler environment code budget exceeded")
        positions = [line for position in current.co_positions()
                     for line in position[:2] if type(line) is int and line > 0]
        rows.append((current.co_name, current.co_qualname, current.co_firstlineno,
                     tuple(current.co_freevars), tuple(current.co_cellvars),
                     min(positions) if positions else None,
                     max(positions) if positions else None))
        for constant in current.co_consts:
            if type(constant) is CodeType:
                visit(constant)
    visit(code)
    tables = []
    def visit_table(table):
        if _table_kind(table) == "function":
            tables.append(_symbol_projection(table))
        for child in table.get_children():
            visit_table(child)
    visit_table(symtable.symtable(prepared.text, str(prepared.path), "exec"))
    # A PreparedSource is frozen but intentionally has an instance dictionary.
    # Its private scratch store serializes declared fields only; keeping this
    # optional cache primitive also makes direct inspection/pickling harmless.
    projection = (cache_key, tuple(rows), tuple(tables))
    object.__setattr__(prepared, _CACHE_ATTRIBUTE, projection)
    return projection


def compiler_environment_layout(prepared, node, definition):
    """Recover exact co_freevars/co_cellvars order for one source function.

    The selected AST node and definition must belong to the prepared snapshot.
    A repeated name, ambiguous code candidate, source mismatch or disagreement
    with the compiler's own symbol table fails explicitly.  This API never
    obtains a callable by evaluating source or looking it up in a live module.
    """
    if platform.python_implementation() != "CPython":
        raise ValueError("compiler environment layout requires CPython")
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError("environment layout requires a selected function definition")
    if type(definition) is not OperationDefinition:
        raise ValueError("environment layout requires a semantic function definition")
    if (definition.kind not in (OperationKind.FUNCTION, OperationKind.METHOD)
            or definition.metadata.get("phase") != "call_time"
            or prepared.semantic.definitions.get(definition.id) is not definition):
        raise ValueError("environment definition is not a prepared call-time function")
    if type(prepared.text) is not str or len(prepared.text) > _MAX_SOURCE_BYTES:
        raise ValueError("environment source exceeds the fixed compile ceiling")
    encoded = prepared.text.encode("utf-8")
    if len(encoded) > _MAX_SOURCE_BYTES:
        raise ValueError("environment UTF-8 source exceeds the fixed compile ceiling")
    fingerprint = "sha256:" + hashlib.sha256(encoded).hexdigest()
    snapshot = prepared.semantics.sources.get(str(prepared.path))
    if snapshot is None or snapshot.fingerprint != fingerprint:
        raise ValueError("environment source differs from its prepared snapshot")
    expected_version = (str(sys.version_info.major), str(sys.version_info.minor))
    if (type(snapshot.python_version) is not str
            or tuple(snapshot.python_version.split(".")[:2]) != expected_version):
        raise ValueError("environment compiler differs from the source Python version")
    selected = [(candidate, name) for candidate, name in _functions(prepared.tree) if candidate is node]
    if len(selected) != 1:
        raise ValueError("selected function is not unique in the prepared AST")
    qualified = selected[0][1]
    atoms = [prepared.semantic.source_atoms[identity] for identity in definition.source_atoms
             if identity in prepared.semantic.source_atoms]
    if (definition.source_file != str(prepared.path) or definition.label != node.name
            or not any(atom.kind == type(node).__name__ and atom.reference.path == str(prepared.path)
                       and atom.reference.fingerprint == fingerprint
                       and (atom.reference.start_line, atom.reference.end_line,
                            atom.reference.start_column, atom.reference.end_column)
                       == (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset)
                       for atom in atoms)):
        raise ValueError("environment definition differs from the selected source span")
    cache_key = (str(prepared.path), fingerprint, sys.implementation.name,
                 tuple(sys.version_info), 0, True)
    projection = getattr(prepared, _CACHE_ATTRIBUTE, None)
    if type(projection) is not tuple or len(projection) != 3 or projection[0] != cache_key:
        projection = _project(prepared, cache_key)
    first_line = node.decorator_list[0].lineno if node.decorator_list else node.lineno
    candidates = [row for row in projection[1]
                  if row[:3] == (node.name, qualified, first_line)
                  and (row[5] is None or first_line <= row[5] <= row[6] <= node.end_lineno)]
    if len(candidates) != 1:
        raise ValueError("selected function has no unique exact compiler layout")
    matching_tables = [row for row in projection[2] if row[:2] == (node.name, node.lineno)]
    if len(matching_tables) != 1:
        raise ValueError("selected function has no unique compiler symbol table")
    _, _, free, local, child_free = matching_tables[0]
    _, _, first_line, freevars, cellvars, _, _ = candidates[0]
    if set(freevars) != set(free):
        raise ValueError("compiler free-variable layout disagrees with free/nonlocal symbols")
    if not set(cellvars) <= set(local) or not set(cellvars) <= set(child_free):
        raise ValueError("compiler cell layout disagrees with locally owned child captures")
    if set(freevars) & set(local):
        raise ValueError("compiler free bindings overlap local bindings")
    return CompilerEnvironmentLayout(freevars, cellvars, qualified, first_line)


# Source-profile checks over canonical environment/Use/state records.
#
# Core validation owns CFG/SSA and port typing.  This check adds only the source
# profile's namespace/cell ownership, ordered operands and protocol refinements.
# Exact compiler layout/order remains an independent fixed-source replay check.
_KINDS = frozenset((Kind.LOOKUP_GLOBAL, Kind.READ_FREE_CELL, Kind.READ_LOCAL_CELL,
                    Kind.MAKE_CELL, Kind.STORE_CELL, Kind.STORE_GLOBAL,
                    Kind.STORE_ATTRIBUTE, Kind.STORE_SUBSCRIPT))
_STORES = frozenset((Kind.STORE_CELL, Kind.STORE_GLOBAL, Kind.STORE_ATTRIBUTE, Kind.STORE_SUBSCRIPT))


def validate_environment_body(model, graph, interface, metadata, semantic, *, language=None):
    from scar.analysis.environment_lowering import captured_cell_slot
    from scar.analysis.source_language import function_exception_types

    def need(condition, detail):
        if not condition:
            raise ValueError("source environment: " + detail)

    need(interface.profile is BodyLoweringProfile.STRUCTURED_ENV, "wrong source profile")
    need(len(graph.environments) == len(graph.function_entries) == len(graph.locations) == 1,
         "requires one canonical environment, entry layout and coarse world")
    environment, layout = graph.environments[0], graph.function_entries[0]
    need((environment.region, environment.code_definition) == (interface.region, interface.definition)
         and layout.environment == environment.id and layout.parameters == interface.parameters,
         "environment/body ownership or formal partition differs")
    values = {value.id: value for value in graph.values}
    operations = {operation.id: operation for operation in graph.operations}
    blocks = {block.id: block for region in graph.regions for block in region.blocks}
    region = next(region for region in graph.regions if region.id == interface.region)
    entry = blocks[region.entry]
    world = values[interface.world]
    need(type(world) is StateVersionValue and world.location == graph.locations[0].id,
         "entry world is not the existing coarse state")
    need(entry.arguments == tuple(member.reference for member in environment.members)
         + tuple(port.value for port in interface.parameters) + (interface.world,),
         "entry must contain ordered environment, formals and world exactly")
    for position, identity in enumerate(entry.arguments):
        need(values[identity].origin == BlockArgument(entry.id, position), "entry origin/order differs")
    for member in environment.members:
        need(type(values[member.reference]) is ReferenceValue and member.locations == (world.location,),
             "environment references must depend on the existing coarse world")
        if member.role is EnvironmentRole.CAPTURED_CELL:
            need(member.cell_slot not in semantic.slots, "captured port reuses an outer/local source slot")
    for port in interface.parameters:
        slot = semantic.slots.get(port.slot)
        need(slot is not None and slot.owner == interface.definition and slot.direction == "input",
             "formal source slot belongs to another definition")
    links = {link.operation: link for link in metadata.operation_links}
    protocols = {protocol.operation: protocol for protocol in model.protocols}
    need(len(protocols) == len(model.protocols), "duplicate protocol for an operation")
    rows = {row.site.operation: row for row in graph.state_address_refinements}
    need(len(rows) == len(graph.state_address_refinements), "duplicate state refinement")
    free_owners = {member.reference: member for member in environment.members[5:]}
    namespaces = {member.role: member.reference for member in environment.members[:5]}
    own_cells, own_names, expected_rows = {}, {}, set()
    source_bindings = {}
    for edge in semantic.edges.values():
        if edge.relation in (SemanticRelation.READS_SLOT, SemanticRelation.WRITES_SLOT):
            source_bindings.setdefault((edge.source.id, edge.relation), []).append(semantic.slots[edge.target.id])
    for protocol in model.protocols:
        if protocol.kind is not Kind.MAKE_CELL:
            continue
        operation = operations[protocol.operation]
        normal = next(edge for edge in operation.terminator.successors if edge.kind is ExitKind.NORMAL)
        own_cells[blocks[normal.block].arguments[0]] = operation
    for protocol in model.protocols:
        operation, kind = operations[protocol.operation], protocol.kind
        link = links[operation.id]
        from scar.ir.program.class_construction import CLASS_CONSTRUCTION_PROTOCOLS
        if kind.value in CLASS_CONSTRUCTION_PROTOCOLS:
            need(language is not None and DisplayCapability.CLASS_CONSTRUCTION_314_V1 in language.capabilities,
                 'class construction has no explicit capability')
            if kind.value in ('LOAD_BUILD_CLASS_314','STORE_CLASS_CELL_314','STORE_CLASS_GLOBAL_314'):
                expected_rows.add(operation.id)
            continue  # Independently replayed by the class construction validator.
        if kind in (Kind.SOURCE_BINARY, Kind.SOURCE_COMPARE, Kind.SOURCE_INDEX, Kind.FRAME_FORWARD_TEMPORARY):
            need(language is not None and DisplayCapability.SOURCE_EXPRESSIONS_314_V1 in language.capabilities,
                 'source expression has no distinct opt-in capability')
            if kind is Kind.SOURCE_INDEX:
                expected_rows.add(operation.id)
            continue  # Exact DATA selector, SG operand ports and pending transfers are checked independently.
        if kind in (Kind.BINARY_SLICE, Kind.STORE_SLICE, Kind.STORE_SLICE_SUBSCRIPT):
            from scar.ir.program.source_language import SourceLanguageCapability
            need(language is not None and SourceLanguageCapability.SLICE_ACCESS_314_V1 in language.capabilities,
                 'slice access has no distinct opt-in capability')
            expected_rows.add(operation.id)
            continue  # Original SG/ports/address/consuming completions are checked by the slice validator.
        if kind.value in STARRED_DISPLAY_PROTOCOLS:
            need(language is not None and DisplayCapability.STARRED_DISPLAY_314_V1 in language.capabilities
                 and semantic.source_atoms[link.source.atom_id].kind in ('Tuple','List'),
                 'display primitive has no original container source/capability')
            continue  # Complete original AST/SG/ordered port checks belong to its display validator.
        if kind is Kind.COMPREHENSION_BUILD:
            from scar.ir.program.source_language import SourceLanguageCapability
            declaration=semantic.definitions.get(link.semantic_operation)
            need(language is not None and SourceLanguageCapability.COMPREHENSION_314_V1 in language.capabilities
                 and declaration is not None and declaration.kind is OperationKind.OPAQUE
                 and declaration.metadata.get("ast_type") in ("ListComp","SetComp","DictComp")
                 and declaration.code_id==link.source.atom_id.wire and link.opcode is Opcode.OPAQUE,
                 "comprehension build lost its original opaque source scope")
            continue
        if kind is Kind.CALL_FUNCTION_EX:
            from scar.analysis.static.lowering.source_call_lowering import is_original_call_definition
            from scar.ir.program.source_language import SourceLanguageCapability
            source = semantic.definitions.get(link.semantic_operation)
            atom = semantic.source_atoms.get(link.source.atom_id)
            need(language is not None and SourceLanguageCapability.CALL_KWARGS_314_V1 in language.capabilities
                 and source is not None and is_original_call_definition(semantic, source, link.source)
                 and source.code_id == link.source.atom_id.wire and link.source.atom_id in source.source_atoms
                 and atom is not None and atom.kind == "Call" and atom.reference == link.source
                 and link.opcode is Opcode.CALL, "kwargs call lost its capability or exact original SG Call")
            continue  # Exact call ports/pending/world are checked by the language validator.
        if kind is Kind.MAKE_NESTED_FUNCTION:
            from scar.ir.program.source_language import SourceLanguageCapability
            declaration = semantic.definitions.get(link.semantic_operation)
            atom = semantic.source_atoms.get(link.source.atom_id)
            need(language is not None and SourceLanguageCapability.NESTED_FUNCTION_314_V1 in language.capabilities
                 and declaration is not None and declaration.kind is OperationKind.OPERATOR
                 and declaration.metadata.get('phase') == 'definition_time' and len(declaration.output_slots) == 1
                 and declaration.code_id == link.source.atom_id.wire and link.source.atom_id in declaration.source_atoms
                 and atom is not None and atom.kind == 'FunctionDef' and atom.reference == link.source
                 and link.opcode is None, 'function creation lost its capability or original definition-time declaration')
            continue  # Exact code/default/cell/name/world checks remain in the ordinary language validator.
        if kind is Kind.BUILD_SET_DISPLAY:
            from scar.ir.program.source_language import SourceLanguageCapability
            declaration = semantic.definitions.get(link.semantic_operation)
            atom = semantic.source_atoms.get(link.source.atom_id)
            need(language is not None and SourceLanguageCapability.SET_DISPLAY_314_V1 in language.capabilities
                 and declaration is not None and declaration.kind is OperationKind.OPERATOR
                 and declaration.metadata.get('ast_type') == 'Set' and len(declaration.output_slots) == 1
                 and declaration.code_id == link.source.atom_id.wire and link.source.atom_id in declaration.source_atoms
                 and atom is not None and atom.kind == 'Set' and atom.reference == link.source
                 and link.opcode is Opcode.BUILD_SET, 'set display lost its capability or original SG source')
            continue  # Ordered elements, incomplete world and both completions remain independently checked below.
        if kind not in _KINDS:
            need(link.semantic_operation is None, "a source-linked operation has an invented protocol")
            continue
        inputs = operation.uses[:-1]
        need(operation.terminator is not None and operation.terminator.kind is TerminatorKind.INVOKE
             and not operation.effects_complete and operation.callee is None
             and tuple(use.id for use in operation.uses) == operation.invoke_arguments
             and all(type(values[use.value]) is ReferenceValue for use in inputs),
             "environment protocol requires its ordinary reference inputs and incomplete invoke")
        need(type(values[operation.uses[-1].value]) is StateVersionValue
             and values[operation.uses[-1].value].location == world.location,
             "environment protocol does not consume the coarse world")
        reference_type = ValueType(ValueKind.REFERENCE)
        state_type = ValueType(ValueKind.STATE, location=world.location)
        payload = () if kind in _STORES else (reference_type,)
        need(tuple((signature.kind, signature.types) for signature in operation.completions)
             == ((ExitKind.NORMAL, payload + (state_type,)),
                 (ExitKind.RAISE, function_exception_types(language, reference_type, state_type))), "normal/raise payload or state differs")
        need(tuple(edge.kind for edge in operation.terminator.successors) == (ExitKind.NORMAL, ExitKind.RAISE)
             and all(edge.arguments == tuple(CompletionResult(index) for index in range(len(signature.types)))
                     for edge, signature in zip(operation.terminator.successors, operation.completions)),
             "completion edges do not carry their own payload and post-world")
        if kind is Kind.MAKE_CELL:
            need(link.semantic_operation is None and len(inputs) <= 1
                 and all(use.value in {port.value for port in interface.parameters} for use in inputs)
                 and Behavior.MEMORY in operation.behaviors, "own-cell creation has an invalid initializer")
            continue
        if kind in (Kind.STORE_ATTRIBUTE, Kind.STORE_SUBSCRIPT):
            from scar.analysis.state_materialization import validate_heap_store
            validate_heap_store(operation, kind, link, rows.get(operation.id), semantic)
            expected_rows.add(operation.id)
            continue
        source = semantic.definitions.get(link.semantic_operation)
        atom = semantic.source_atoms[link.source.atom_id]
        unpacked_source=False
        if kind in (Kind.STORE_CELL,Kind.STORE_GLOBAL) and language is not None and link.opcode is Opcode.OPAQUE:
            from scar.ir.program.source_language import SourceLanguageCapability
            if SourceLanguageCapability.UNPACK_TARGETS_314_V1 in language.capabilities:
                # The old overlay had no tuple-element value, hence OPAQUE.
                # Preserve that original record. Only the exact new NORMAL
                # result -> original Name store is owned by the unpack rule.
                for shape in language.unpackings:
                    original=operations[shape.operation]
                    normal=[e for e in original.terminator.successors if e.kind is ExitKind.NORMAL]
                    if len(normal)!=1:continue
                    arguments=blocks[normal[0].block].arguments
                    for index,target in enumerate(shape.child_targets):
                        if target==link.source and index<len(arguments)-1 and operation.uses[-2].value==arguments[index]:
                            unpacked_source=True
        need(source is not None and source.code_id == link.source.atom_id.wire and atom.kind == "Name"
             and (link.opcode is (Opcode.ALIAS if kind in _STORES else Opcode.READ) or unpacked_source),
             "cell/namespace protocol must correspond to its exact Name read/write: "+kind.value+" / "+str(link.opcode)+" / "+str(link.source.start_line))
        relation = SemanticRelation.WRITES_SLOT if kind in _STORES else SemanticRelation.READS_SLOT
        slots = source_bindings.get((source.id, relation), ())
        need(len(slots) == 1, "Name has no unique source binding slot")
        slot, effects = slots[0], ()
        if kind in (Kind.READ_FREE_CELL, Kind.READ_LOCAL_CELL, Kind.STORE_CELL):
            need(len(inputs) == (2 if kind is Kind.STORE_CELL else 1), "cell operand arity differs")
            owner = inputs[0].value
            if owner in free_owners:
                need(kind is not Kind.READ_LOCAL_CELL
                     and free_owners[owner].cell_slot == captured_cell_slot(interface.definition, slot.name)
                     and slot.owner != interface.definition, "captured cell has the wrong source owner/name")
            else:
                need(kind is not Kind.READ_FREE_CELL and owner in own_cells
                     and slot.owner == interface.definition, "local cell has no invocation-owned creation")
                need(owner not in own_names or own_names[owner] == slot.slot_id,
                     "one local cell is used for different compiler bindings")
                own_names[owner] = slot.slot_id
            effects = (AddressEffect(ReferenceOwner(inputs[0].id), (CellContent(),),
                                    AccessMode.WRITE if kind in _STORES else AccessMode.READ),)
        else:
            expected = ((namespaces[EnvironmentRole.GLOBALS_NAMESPACE],
                         namespaces[EnvironmentRole.BUILTINS_NAMESPACE])
                        if kind is Kind.LOOKUP_GLOBAL else (namespaces[EnvironmentRole.GLOBALS_NAMESPACE],))
            key_index = len(expected)
            need(len(inputs) == 3 and tuple(use.value for use in inputs[:key_index]) == expected,
                 "globals and saved-builtins lookup order differs")
            origin = values[inputs[key_index].value].origin
            need(type(origin) is LiteralOrigin and type(origin.value) is str and origin.value == slot.name,
                 "namespace key is not the exact source Name")
            owner = semantic.definitions.get(slot.owner)
            need(owner is not None and owner.kind is OperationKind.MODULE,
                 "namespace protocol incorrectly substitutes for a local/free binding")
            effects = tuple(AddressEffect(ReferenceOwner(inputs[index].id),
                (NamespaceKey(inputs[key_index].id),), AccessMode.WRITE if kind in _STORES else AccessMode.READ)
                for index in range(key_index))
        row = rows.get(operation.id)
        need(row is not None and row.site == InvokeStateOperand(operation.id, operation.uses[-1].id)
             and row.coverage is AddressCoverage.PARTIAL and row.protocol is None and row.effects == effects,
             "state address must name this operation's exact owner/key/world Uses")
        expected_rows.add(operation.id)
    if language is not None:
        from scar.ir.program.source_language import SourceLanguageCapability
        if SourceLanguageCapability.HEAP_AUGASSIGN_314_V1 in language.capabilities:
            # Attribute/ordinary INDEX reads retain their original address
            # interface. Compound slices have receiver/lower/upper/world,
            # independently checked by the dedicated slice validator; enabling
            # heap AugAssign elsewhere cannot reinterpret them as key reads.
            for link in metadata.operation_links:
                if link.opcode not in (Opcode.ATTRIBUTE,Opcode.INDEX) or link.operation not in rows:
                    continue
                marker = protocols.get(link.operation)
                if marker is not None and marker.kind is Kind.BINARY_SLICE:
                    need(link.opcode is Opcode.INDEX
                         and SourceLanguageCapability.SLICE_ACCESS_314_V1 in language.capabilities,
                         'compound slice read lacks its original INDEX/capability')
                    continue  # Exact source/ports/Slice address/completions belong to the slice validator.
                operation = operations[link.operation]
                source = semantic.definitions.get(link.semantic_operation)
                atom = semantic.source_atoms.get(link.source.atom_id)
                attribute = link.opcode is Opcode.ATTRIBUTE
                need(source is not None and source.kind is (OperationKind.ATTRIBUTE if attribute else OperationKind.INDEX)
                     and source.code_id == link.source.atom_id.wire
                     and link.source.atom_id in source.source_atoms and source.metadata.get('role') != 'binding'
                     and atom is not None and atom.kind == ('Attribute' if attribute else 'Subscript') and atom.reference == link.source
                     and operation.terminator is not None and operation.terminator.kind is TerminatorKind.INVOKE
                     and not operation.effects_complete and len(operation.uses) == (2 if attribute else 3)
                     and operation.invoke_arguments == tuple(u.id for u in operation.uses)
                     and all(type(values[u.value]) is ReferenceValue for u in operation.uses[:-1])
                     and type(values[operation.uses[-1].value]) is StateVersionValue,
                     'heap AugAssign read lost its original attribute/receiver/world')
                effect = AddressEffect(ReferenceOwner(operation.uses[0].id),
                    (ObjectField(source.metadata['attribute']) if attribute else Element(operation.uses[1].id),), AccessMode.READ)
                row = rows[operation.id]
                need(row.site == InvokeStateOperand(operation.id,operation.uses[-1].id)
                     and row.coverage is AddressCoverage.PARTIAL and row.protocol is None
                     and row.effects == (effect,), 'heap attribute read address differs from original Uses')
                expected_rows.add(operation.id)
        if SourceLanguageCapability.SLICE_KEYS in language.capabilities:
            for link in metadata.operation_links:
                if link.opcode is not Opcode.INDEX:
                    continue
                marker = protocols.get(link.operation)
                if marker is not None and marker.kind is Kind.BINARY_SLICE:
                    continue  # The compound lower/upper address is validated independently.
                operation = operations[link.operation]
                source = semantic.definitions.get(link.semantic_operation)
                atom = semantic.source_atoms.get(link.source.atom_id)
                need(source is not None and source.kind is OperationKind.INDEX
                     and source.code_id == link.source.atom_id.wire
                     and link.source.atom_id in source.source_atoms
                     and atom is not None and atom.kind == "Subscript" and atom.reference == link.source,
                     "language index address has no exact original source INDEX")
                need(operation.terminator is not None and operation.terminator.kind is TerminatorKind.INVOKE
                     and not operation.effects_complete and operation.callee is None
                     and len(operation.uses) == 3
                     and tuple(use.id for use in operation.uses) == operation.invoke_arguments
                     and all(type(values[use.value]) is ReferenceValue for use in operation.uses[:2])
                     and type(values[operation.uses[-1].value]) is StateVersionValue
                     and values[operation.uses[-1].value].location == world.location,
                     "language index address needs original receiver/key/world invoke Uses")
                effect = AddressEffect(ReferenceOwner(operation.uses[0].id),
                                       (Element(operation.uses[1].id),), AccessMode.READ)
                row = rows.get(operation.id)
                need(row is not None and row.site == InvokeStateOperand(operation.id, operation.uses[-1].id)
                     and row.coverage is AddressCoverage.PARTIAL and row.protocol is None
                     and row.effects == (effect,), "language index address differs from its exact receiver/key/world Uses")
                expected_rows.add(operation.id)
    if language is not None:
        from scar.ir.program.source_language import SourceLanguageCapability
        if SourceLanguageCapability.NESTED_FUNCTION_314_V1 in language.capabilities:
            # Its original FunctionDef source/name/cell/address correspondence
            # is checked by the dedicated ordinary language validator below.
            expected_rows.update(p.operation for p in model.protocols if p.kind is Kind.STORE_NESTED_FUNCTION_CELL)
        if SourceLanguageCapability.FUNCTION_IMPORT_314_V1 in language.capabilities:
            # Exact import source/builtins/key/world correspondence is checked
            # by the ordinary opt-in language validator, not the Name rule.
            expected_rows.update(p.operation for p in model.protocols if p.kind is Kind.FUNCTION_IMPORT_NAME)
    need(set(rows) == expected_rows, "unexpected or missing source environment address refinement")
    for value in values.values():
        if type(value) is LocalBindingValue:
            need(value.slot not in set(own_names.values()), "cell content was replaced with a local binding token")


__all__ = [
    'CompilerEnvironmentLayout',
    'compiler_environment_layout',
    'validate_environment_body',
]
