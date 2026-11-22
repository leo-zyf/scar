"""Source language."""

from __future__ import annotations

import ast
import json
import dis
from dataclasses import dataclass, field
from types import CodeType
from scar.analysis.static.compilation import StaticCodeEvidence
from scar.ir.program.call_edges import CapturedCode, CodeSourceLink, CodeCapture, CallLinkStatus
from scar.ir.program.source_language import (
    SourceLanguageCapability,
    SourceLanguageOverlay,
    SourceLanguageCapability as Capability,
    EXTENDED_PROTOCOLS,
)
from scar.ir.record_codec import decode, encode
from scar.trace.code_snapshot import describe_code
from scar.analysis.program_lowering import _REF
from scar.ir.program.model import (
    AccessMode,
    BlockArgument,
    CompletionResult,
    ExitKind,
    ReferenceValue,
    StateVersionValue,
    LocalBindingValue,
    TerminatorKind,
    value_type,
)


# Opt-in inputs and validation for extended source protocols.
#
# CodeType inputs are supplied by an already verified CodeWitnessStore. This
# module does not compile, import the target, or open a report-selected archive.
@dataclass(frozen=True)
class SourceCodeEvidence:
    store: object
    composition: object
    captured: CapturedCode
    code: CodeType = field(init=False)
    source_link: CodeSourceLink = field(init=False)
    _witness_json: str = field(init=False, repr=False)

    def __post_init__(self):
        from scar.analysis.code_witness import CodeWitnessStore
        from scar.analysis.program_composition import CompositionIndex
        if (type(self.store) is not CodeWitnessStore or self.store.writable or self.store.compile_missing
                or type(self.composition) is not CompositionIndex or type(self.captured) is not CapturedCode):
            raise ValueError("source language requires explicit original composition and read-only CodeWitnessStore")
        source = self.composition.inventory_component("call_sources")["model"]
        captures = [(i, decode(CodeCapture, row)) for i, row in enumerate(source["codes"])
                    if row["code"]["key"] == self.captured.key]
        links = [(i, decode(CodeSourceLink, row)) for i, row in enumerate(source["code_links"])
                 if row["code"] == self.captured.key]
        if (not captures or any(row.code != self.captured for _, row in captures) or len(links) != 1
                or links[0][1].status is not CallLinkStatus.VERIFIED):
            raise ValueError("original composition has no complete matching captured-code/source witness")
        link_index, link = links[0]
        code, reference = self.store.resolve(self.captured, link.snapshot, link.optimize_levels, allow_compile=False)
        if type(code) is not CodeType or describe_code(code) != encode(self.captured):
            raise ValueError("owned CodeWitness archive did not reproduce the complete code descriptor")
        witness = {**reference, "store": self.store.receipt(),
            "source_link_locator": self.composition.inventory_component_locator("call_sources", f"/model/code_links/{link_index}"),
            "captured_code_locator": self.composition.inventory_component_locator("call_sources", f"/model/codes/{captures[0][0]}/code"),
            "inventory": self.composition.original_code_source_origin()["inventory"]}
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "source_link", link)
        object.__setattr__(self, "_witness_json", json.dumps(witness, sort_keys=True, separators=(",", ":"), allow_nan=False))

    @classmethod
    def from_entry(cls, catalog, entry):
        """Resolve one original qualified entry; no inventory origin is invented."""
        from scar.analysis.source_entry_catalog import SourceEntryCatalog, ResolvedSourceEntry
        from scar.analysis.program_composition import CompositionIndex
        if cls is not SourceCodeEvidence or type(catalog) is not SourceEntryCatalog or type(entry) is not ResolvedSourceEntry:
            raise ValueError("source evidence requires an exact original entry catalog")
        owner = catalog.require_entry(entry)
        store = catalog.store_for(entry)
        if type(owner) is CompositionIndex:
            result = cls(store, owner, entry.captured)
            result.require_definition(entry.reference.snapshot, entry.reference.definition, entry.reference.source)
            return result
        owner.resolve_definition(entry.reference)
        link = entry.source_link
        if (link.status is not CallLinkStatus.VERIFIED or link.snapshot != entry.reference.snapshot
                or link.definition != entry.reference.definition or link.source != entry.reference.source
                or entry.captured.filename != link.snapshot.path):
            raise ValueError("original entry code/source/definition differs")
        code, reference = store.resolve(entry.captured, link.snapshot, link.optimize_levels, allow_compile=False)
        if type(code) is not CodeType or describe_code(code) != encode(entry.captured):
            raise ValueError("entry archive differs from the complete captured descriptor")
        witness = {**reference, "store": store.receipt(), "source_origin": entry.origin}
        result = object.__new__(cls)
        for name, value in (("store", store), ("composition", None), ("captured", entry.captured),
                            ("code", code), ("source_link", link),
                            ("_witness_json", json.dumps(witness, sort_keys=True, separators=(",", ":"), allow_nan=False))):
            object.__setattr__(result, name, value)
        result.require_definition(entry.reference.snapshot, entry.reference.definition, entry.reference.source)
        return result

    @property
    def witness(self):
        # A caller may annotate its copy of a receipt, never the independently
        # verified evidence retained by this input object.
        return json.loads(self._witness_json)

    def require_definition(self, snapshot, definition, source):
        link = self.source_link
        if (link.snapshot != snapshot or link.definition != definition or link.source != source
                or self.captured.filename != snapshot.path):
            raise ValueError("CodeWitness does not identify this exact source definition")

    def to_dict(self):
        return {"captured": encode(self.captured), "source_link": encode(self.source_link),
                "witness": self.witness, "runtime_code_instance_established": False}

    @property
    def code_layout_key(self):
        """Structural code key; this remains distinct from active code identity."""
        return self.captured.key

    @property
    def compiler_version(self):
        return tuple(self.store.document["runtime"]["version"][:3])

    @property
    def compiler_implementation(self):
        return self.store.document["runtime"]["implementation"]


@dataclass(frozen=True)
class SourceLanguageOptions:
    capabilities: tuple[SourceLanguageCapability, ...] = (
        SourceLanguageCapability.FIXED_UNPACK, SourceLanguageCapability.SLICE_KEYS,
        SourceLanguageCapability.LEXICAL_LOCALS)
    code_evidence: SourceCodeEvidence | StaticCodeEvidence | None = None

    def __post_init__(self):
        if (type(self.capabilities) is not tuple or not self.capabilities
                or any(type(row) is not SourceLanguageCapability for row in self.capabilities)
                or len(set(self.capabilities)) != len(self.capabilities)
                or self.code_evidence is not None and not _is_code_layout_evidence(self.code_evidence)):
            raise ValueError("invalid explicit source-language inputs")


def _is_code_layout_evidence(value):
    return type(value) in (SourceCodeEvidence, StaticCodeEvidence)


def supports_cpython_compiler(evidence, version=(3, 14, 7)):
    """Whether exact compiler-layout protocols are supported by this input."""
    return (evidence is not None
            and evidence.compiler_implementation == "cpython"
            and evidence.compiler_version == tuple(version))


def require_source_tree(tree, snapshot, source_reference):
    """Resolve a source node by complete span and kind, under its fixed snapshot."""
    if source_reference.path != snapshot.path or source_reference.fingerprint != snapshot.fingerprint:
        raise ValueError("language source reference differs from its fixed snapshot")
    return [node for node in ast.walk(tree) if hasattr(node, "end_lineno")
            and (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset)
            == (source_reference.start_line, source_reference.end_line,
                source_reference.start_column, source_reference.end_column)]


def function_exception_types(language, reference_type, world_type):
    """Only the explicit function-Try profile carries propagation traceback."""
    paired = language is not None and SourceLanguageCapability.FUNCTION_TRY_314_V1 in language.capabilities
    return ((reference_type, reference_type, world_type) if paired
            else (reference_type, world_type))


# Validate new source protocols against original ports and exact source.
#
# Structural reading is distinct from source/CodeWitness verification. Neither
# mode establishes runtime code-instance identity or operation purity.
def _need(condition, detail):
    if not condition:
        raise ValueError("source language validation: " + detail)


def validate_language_body(model, graph, interface, metadata, semantic, language,
                           *, source_tree=None, options=None, structural_only=False,control=None,sealed_source=None,
                           pending_protocols=None):
    if pending_protocols is None:
        pending_protocols = model.protocols
    if language is not None and language.lambda_bodies:
        from scar.analysis.static.lowering.source_lambda import validate_lambda_regions,root_projection as lambda_projection
        _need(control is not None,'Lambda body checking requires the original control context, not a fabricated suite')
        delegated=validate_lambda_regions(model,graph,interface,metadata,semantic,control,language,
            source_tree=source_tree,options=options,structural_only=structural_only,sealed_source=sealed_source)
        model,graph,metadata,language=lambda_projection(model,graph,metadata,language,delegated)
    if (language is not None and any(c in language.capabilities for c in
            (Capability.GENEXPR_314_V1, Capability.GENEXPR_CREATION_314_V1))
            and graph.resumable_creates):
        from scar.analysis.source_generator_validation import validate_generator_regions, root_projection
        delegated = validate_generator_regions(model, graph, interface, metadata, semantic, language,
            source_tree=source_tree, options=options, structural_only=structural_only)
        model, graph, metadata, language = root_projection(model, graph, metadata, language, delegated)
    _need(type(language) is SourceLanguageOverlay and language.source == metadata.source,
          "overlay source differs from original body metadata")
    _need(type(structural_only) is bool, "verification mode must be explicit")
    if not structural_only:
        _need(isinstance(source_tree, ast.Module) and type(options) is SourceLanguageOptions,
              "exact source/compiler-layout verification requires explicit source AST and independent options")
        _need(options.capabilities == language.capabilities, "requested and reported language capabilities differ")
        expected_evidence = options.code_evidence.to_dict() if options.code_evidence is not None else None
        from scar.ir.record_codec import encode
        # Compiler name tables are ordered tuples in memory and JSON arrays on
        # the wire. Compare the exact same canonical receipt representation;
        # no key, index, spelling, compiler option or source claim is omitted.
        _need(encode(language.code_evidence) == encode(expected_evidence),
              "code provenance differs from independent compiler inputs")
        if options.code_evidence is not None:
            options.code_evidence.require_definition(metadata.source, interface.definition, model.source)
    operations = {row.id: row for row in graph.operations}
    values = {row.id: row for row in graph.values}
    blocks = {block.id: block for region in graph.regions for block in region.blocks}
    protocols = {row.operation: row for row in model.protocols}
    links = {row.operation: row for row in metadata.operation_links}
    world_type = value_type(values[interface.world])
    unpack = {row.operation: row for row in language.unpackings}
    slices = {row.operation: row for row in language.slices}
    constants = {row.operation: row for row in language.constants}
    _need(len(unpack) == len(language.unpackings) and len(slices) == len(language.slices)
          and len(constants) == len(language.constants), "duplicate language shape/provenance record")
    from scar.analysis.static.lowering.source_call_lowering import CALL_PROTOCOLS, validate_call_protocols
    calls = {identity for identity, protocol in protocols.items() if protocol.kind in CALL_PROTOCOLS}
    from scar.analysis.static.lowering.source_function_try import validate_function_try_protocols
    try_protocols = validate_function_try_protocols(model, graph, interface, metadata, semantic, language, source_tree=source_tree)
    from scar.analysis.static.lowering.source_function_definitions import validate_function_definitions
    functions = validate_function_definitions(model, graph, interface, metadata, semantic, language,
        source_tree=source_tree, options=options, structural_only=structural_only)
    from scar.analysis.static.lowering.source_import_lowering import validate_imports
    imports = validate_imports(model, graph, interface, metadata, semantic, language,
        source_tree=source_tree, options=options, structural_only=structural_only)
    from scar.analysis.static.lowering.source_with import validate_with
    withs = validate_with(model, graph, interface, metadata, semantic, language,
        source_tree=source_tree, options=options, structural_only=structural_only)
    from scar.analysis.static.lowering.source_set_lowering import validate_set_displays
    sets = validate_set_displays(model, graph, interface, metadata, semantic, language,
        source_tree=source_tree, options=options, structural_only=structural_only)
    from scar.analysis.static.lowering.source_comprehension import validate_comprehensions
    comprehensions=validate_comprehensions(model,graph,interface,metadata,semantic,language,
        source_tree=source_tree,options=options,structural_only=structural_only)
    from scar.analysis.static.lowering.source_displays import validate_starred_displays
    displays=validate_starred_displays(model,graph,interface,metadata,semantic,language,
        source_tree=source_tree,options=options,structural_only=structural_only)
    from scar.analysis.static.lowering.source_slice import validate_slice_access
    slice_access = validate_slice_access(model, graph, interface, metadata, semantic, language,
        source_tree=source_tree, options=options, structural_only=structural_only)
    from scar.analysis.source_expression_validation import validate_source_expressions
    expressions = validate_source_expressions(model, graph, interface, metadata, semantic, language,
        source_tree=source_tree, options=options, structural_only=structural_only,
        pending_protocols=pending_protocols)
    from scar.analysis.static.lowering.source_class_construction import validate_class_constructions
    _need(not language.class_constructions or options is not None,
          'class construction requires original independently reconstructed compiler inputs')
    classes = validate_class_constructions(graph=graph,metadata=metadata,records=language.class_constructions,
        protocols=protocols,code_evidence=options.code_evidence if options is not None else None,
        source_tree=source_tree,semantic=semantic,interface=interface,structural_only=structural_only,constants=language.constants)
    indexed = classes | expressions | slice_access | displays | comprehensions | set(unpack) | set(slices) | set(constants) | calls | try_protocols | functions | imports | withs | sets
    actual = {key for key, row in protocols.items() if row.kind.value in EXTENDED_PROTOCOLS}
    _need(indexed == actual and len(indexed) == len(classes) + len(expressions) + len(slice_access) + len(unpack) + len(slices) + len(constants) + len(calls) + len(try_protocols) + len(functions) + len(imports) + len(withs) + len(sets) + len(comprehensions) + len(displays),
          "new language operations must have one exact shape/provenance record")

    def source_node(reference, classes):
        atom = semantic.source_atoms.get(reference.atom_id)
        _need(atom is not None and atom.reference == reference, "protocol source is not the original source atom")
        _need(atom.kind in tuple(kind.__name__ for kind in classes), "protocol source AST kind differs")
        if structural_only:
            return None
        candidates = [node for node in require_source_tree(source_tree, language.source, reference)
                      if isinstance(node, classes)]
        _need(len(candidates) == 1, "protocol needs one exact source AST node")
        return candidates[0]

    def invoke(identity, count, result_count):
        operation = operations[identity]
        _need(operation.terminator is not None and operation.terminator.kind is TerminatorKind.INVOKE
              and not operation.effects_complete and len(operation.invoke_arguments) == count + 1,
              "protocol must retain incomplete invoke operands/completions")
        uses = {row.id: row for row in operation.uses}
        operands = tuple(uses[key] for key in operation.invoke_arguments)
        _need(all(type(values[row.value]) is ReferenceValue for row in operands[:-1])
              and type(values[operands[-1].value]) is StateVersionValue,
              "protocol references/world are not their original typed Uses")
        _need(tuple(row.kind for row in operation.completions) == (ExitKind.NORMAL, ExitKind.RAISE),
              "protocol must keep ordered normal and exceptional completions")
        normal, failure = operation.completions
        _need(normal.types == (_REF,) * result_count + (world_type,)
              and failure.types == function_exception_types(language, _REF, world_type), "protocol payload/world types differ")
        _need(tuple(row.kind for row in operation.terminator.successors) == (ExitKind.NORMAL, ExitKind.RAISE)
              and all(edge.arguments == tuple(CompletionResult(i) for i in range(len(signature.types)))
                      for edge, signature in zip(operation.terminator.successors, operation.completions)),
              "normal/raise ports are not this invoke's ordered CompletionResults")
        edge = operation.terminator.successors[0]
        return operation, operands, blocks[edge.block]

    for identity in indexed:
        _need(identity in operations and identity in links and links[identity].source == protocols[identity].source,
              "language operation lost its exact source/core correspondence")
    for identity, shape in unpack.items():
        _need(Capability.FIXED_UNPACK in language.capabilities and protocols[identity].kind.value == "UNPACK_FIXED"
              and shape.source == protocols[identity].source, "unpack shape/kind/capability differs")
        node = source_node(shape.source, (ast.Tuple, ast.List))
        if node is not None:
            _need(isinstance(node.ctx, ast.Store) and not any(isinstance(row, ast.Starred) for row in ast.walk(node)),
                  "fixed unpack cannot accept starred/load syntax")
            _need(tuple((row.lineno, row.end_lineno, row.col_offset, row.end_col_offset) for row in node.elts)
                  == tuple((row.start_line, row.end_line, row.start_column, row.end_column) for row in shape.child_targets),
                  "unpack target sequence differs from exact source")
        operation, operands, normal = invoke(identity, 1, len(shape.child_targets))
        _need(operands[0].value != operands[-1].value, "unpack input/world alias typed ports")
        source_uses = {(row.operation, row.use): row for row in metadata.use_links}
        source_use = source_uses.get((identity, operands[0].id))
        _need(source_use is not None and source_use.role == "unpack-source" and source_use.index == 0
              and source_use.source == shape.source, "unpack original RHS Use/source role differs")
        # Every normal result is published exactly once before the first leaf
        # store/inner unpack. The primitive owns its arity/error internals.
        published = []
        for key in normal.operations[:len(shape.child_targets)]:
            protocol = protocols.get(key)
            if protocol is None or protocol.kind.value != "FRAME_HOLD_TEMPORARY":
                break
            reference = next(row.value for row in operations[key].uses if type(values[row.value]) is ReferenceValue)
            published.append(reference)
        _need(tuple(published) == tuple(reversed(normal.arguments[:-1])),
              "unpack must publish all same-level results before ordered target stores")
        for reference in shape.child_targets:
            atom = semantic.source_atoms.get(reference.atom_id)
            allowed = ("Name", "Tuple", "List", "Attribute", "Subscript") if Capability.UNPACK_TARGETS_314_V1 in language.capabilities else ("Name", "Tuple", "List")
            _need(atom is not None and atom.reference == reference and atom.kind in allowed,
                  "local unpack child has no permitted original target atom")
    # Verify normal-path target order and direct original assigned ports. A
    # nested unpack starts only when its outer siblings before it were stored.
    visiting = set()
    from scar.analysis.static.lowering.source_comprehension import scoped_target_slots
    scoped_unpack_slots=scoped_target_slots(semantic,source_tree,language) if isinstance(source_tree,ast.Module) else {}
    def stores(identity):
        _need(identity not in visiting, "unpack target shape is cyclic")
        visiting.add(identity)
        shape = unpack[identity]
        _, _, normal = invoke(identity, 1, len(shape.child_targets))
        block, position = normal, len(shape.child_targets)
        for index, reference in enumerate(shape.child_targets):
            _need(position < len(block.operations), "unpack target store is missing")
            key = block.operations[position]
            protocol = protocols.get(key)
            atom = semantic.source_atoms[reference.atom_id]
            if atom.kind in ("Tuple", "List"):
                _need(key in unpack and unpack[key].source == reference,
                      "nested unpack does not follow its original target position")
                nested_uses = {row.id: row for row in operations[key].uses}
                _need(nested_uses[operations[key].invoke_arguments[0]].value == normal.arguments[index],
                      "inner unpack did not consume the corresponding outer normal result")
                block, position = stores(key)
            else:
                _need(protocol is not None and protocol.kind.value == "FRAME_BIND"
                      and protocol.source == reference, "local target store order/source differs")
                assigned = [row.value for row in operations[key].uses if type(values[row.value]) is ReferenceValue]
                _need(assigned == [normal.arguments[index]], "local target did not receive its exact unpack result")
                _need(position + 1 < len(block.operations), "local target has no binding definition")
                definition = operations[block.operations[position + 1]]
                _need(len(definition.results) == 1 and type(values[definition.results[0]]) is LocalBindingValue
                      and len(definition.uses) == 1 and definition.uses[0].value == normal.arguments[index],
                      "local binding token did not retain its assigned original reference")
                slot = values[definition.results[0]].slot
                _need(semantic.slots[slot].owner == interface.definition or scoped_unpack_slots.get(reference)==slot,
                      "unpack introduced a binding in another lexical scope")
                position += 2
        visiting.remove(identity)
        return block, position
    if Capability.UNPACK_TARGETS_314_V1 in language.capabilities:
        from scar.analysis.static.lowering.source_unpack_lowering import validate_unpack_targets
        validate_unpack_targets(model,graph,interface,metadata,semantic,language,
            source_tree=source_tree,options=options,structural_only=structural_only)
    else:
        for identity in unpack:
            stores(identity)
    for identity, shape in slices.items():
        _need(Capability.SLICE_KEYS in language.capabilities and protocols[identity].kind.value == "BUILD_SLICE"
              and shape.source == protocols[identity].source, "slice shape/kind/capability differs")
        node = source_node(shape.source, (ast.Slice,))
        if node is not None:
            for part, expected in ((node.lower, shape.lower), (node.upper, shape.upper), (node.step, shape.step)):
                _need((part is None) == (expected is None), "slice omission differs from source")
                if part is not None:
                    _need((part.lineno, part.end_lineno, part.col_offset, part.end_col_offset)
                          == (expected.start_line, expected.end_line, expected.start_column, expected.end_column),
                          "slice bound source differs")
        _, operands, _ = invoke(identity, 3 if shape.step is not None else 2, 1)
        source_uses = {(row.operation, row.use): row for row in metadata.use_links}
        components = (shape.lower, shape.upper, shape.step)[:len(operands) - 1]
        for index, (use, reference) in enumerate(zip(operands[:-1], components)):
            link = source_uses.get((identity, use.id))
            _need(link is not None and link.role == ('slice-lower', 'slice-upper', 'slice-step')[index]
                  and link.index == index and link.source == (reference or shape.source),
                  "slice bound ordered Use source/role differs")
            if reference is None:
                from scar.ir.program.model import LiteralOrigin
                _need(type(values[use.value].origin) is LiteralOrigin and values[use.value].origin.value is None,
                      "omitted slice bound is not exact None")
        if not structural_only:
            evidence = options.code_evidence
            _need(evidence is not None, "dynamic slice requires its independent compiler witness")
            candidates = [row for row in dis.get_instructions(evidence.code)
                if row.opname == "BUILD_SLICE" and row.arg == len(operands) - 1
                and tuple(row.positions) == (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset)]
            _need(len(candidates) == 1, "dynamic slice does not match its exact BUILD_SLICE code position")
    if Capability.SLICE_KEYS in language.capabilities:
        from scar.ir.program.state_addresses import AddressCoverage, AddressEffect, Element, InvokeStateOperand, ReferenceOwner
        from scar.ir.semantics_v2 import Opcode
        from scar.ir.v2.semantic import OperationKind
        addresses = {row.site.operation: row for row in graph.state_address_refinements}
        source_uses = {(row.operation, row.use): row for row in metadata.use_links}
        for identity, link in links.items():
            if link.opcode is not Opcode.INDEX:
                continue
            if identity in slice_access:
                continue  # Distinct compound receiver/lower/upper interface was checked above.
            definition = semantic.definitions.get(link.semantic_operation)
            _need(definition is not None and definition.kind is OperationKind.INDEX
                  and definition.code_id == link.source.atom_id.wire
                  and link.source.atom_id in definition.source_atoms,
                  "index operation lost its exact original source definition")
            node = source_node(link.source, (ast.Subscript,))
            operation, operands, _ = invoke(identity, 2, 1)
            if node is not None and not isinstance(node.ctx, ast.Load):
                # AugAssign's original AST target is Store, but its first
                # operation is a read. The function protocol consumer above
                # already verifies the complete load/in-place/storeback path;
                # accept only that path's exact left operand, never an
                # arbitrary Store-context INDEX operation.
                _need(isinstance(node.ctx, ast.Store)
                      and Capability.HEAP_AUGASSIGN_314_V1 in language.capabilities,
                      "load-key capability cannot certify a store")
                normal = blocks[operation.terminator.successors[0].block]
                augmented = []
                for candidate in try_protocols:
                    if protocols[candidate].kind.value != "FUNCTION_INPLACE_BINARY":
                        continue
                    left = operations[candidate].uses[0]
                    original = source_uses.get((candidate, left.id))
                    if (left.value == normal.arguments[0] and original is not None
                            and original.source == link.source
                            and (original.role, original.index) == ("augassign-left", 0)):
                        augmented.append(source_node(protocols[candidate].source, (ast.AugAssign,)))
                _need(len(augmented) == 1 and augmented[0].target is node,
                      "Store-context index read is not the validated original AugAssign target")
            _need(operation.callee is None and values[operands[-1].value].location == values[interface.world].location,
                  "index invoke does not consume its original coarse world")
            row = addresses.get(identity)
            expected = AddressEffect(ReferenceOwner(operands[0].id), (Element(operands[1].id),), AccessMode.READ)
            _need(row is not None and row.site == InvokeStateOperand(identity, operands[-1].id)
                  and row.coverage is AddressCoverage.PARTIAL and row.protocol is None and row.effects == (expected,),
                  "index partial address differs from its exact receiver/key/world Uses")
            for index, role in enumerate(("receiver", "index")):
                source_use = source_uses.get((identity, operands[index].id))
                _need(source_use is not None and source_use.role == role and source_use.index == index,
                      "index original receiver/key Use order differs")
                atom = semantic.source_atoms.get(source_use.source.atom_id)
                _need(atom is not None and atom.reference == source_use.source,
                      "index operand has no exact original source atom")
                if node is not None:
                    child = (node.value, node.slice)[index]
                    _need((source_use.source.path, source_use.source.fingerprint) == (language.source.path, language.source.fingerprint)
                          and (source_use.source.start_line, source_use.source.end_line,
                               source_use.source.start_column, source_use.source.end_column)
                          == (child.lineno, child.end_lineno, child.col_offset, child.end_col_offset),
                          "index receiver/key differs from exact source expression")
    for identity, constant in constants.items():
        atom = semantic.source_atoms.get(constant.source.atom_id)
        literal_constant = atom is not None and atom.kind == "Constant"
        call_constant = atom is not None and atom.kind == "Call"
        function_constant = atom is not None and atom.kind == 'FunctionDef'
        class_constant = atom is not None and atom.kind == 'ClassDef'
        import_constant = atom is not None and atom.kind == 'ImportFrom'
        capability = (Capability.CLASS_CONSTRUCTION_314_V1 if class_constant else Capability.CODE_CONSTANTS_314_V1 if literal_constant else Capability.FUNCTION_IMPORT_314_V1 if import_constant else Capability.NESTED_FUNCTION_314_V1
            if function_constant else Capability.CALL_KWARGS_314_V1 if call_constant else Capability.SLICE_KEYS)
        _need(capability in language.capabilities and protocols[identity].kind.value == "READ_CODE_CONSTANT"
              and constant.source == protocols[identity].source, "code constant kind/capability differs")
        node = source_node(constant.source, (ast.ClassDef,) if class_constant else (ast.Constant,) if literal_constant else (ast.ImportFrom,) if import_constant else (ast.FunctionDef,)
            if function_constant else (ast.Call,) if call_constant else (ast.Slice, ast.Tuple))
        operation = operations[identity]
        _need(operation.terminator is None and not operation.effects_complete and len(operation.results) == 1
              and type(values[operation.results[0]]) is ReferenceValue
              and values[operation.results[0]].referent_identity is None
              and len(operation.accesses) == 1 and operation.accesses[0].mode is AccessMode.READ
              and len(operation.uses) == 1 and type(values[operation.uses[0].value]) is StateVersionValue,
              "code constant reference must be a nonallocating, identity-unproved table read")
        from scar.ir.program.model import Behavior
        _need(operation.behaviors == (Behavior.READ,), "constant table read cannot claim fresh allocation or mutation")
        if not structural_only:
            evidence = options.code_evidence
            _need(evidence is not None and evidence.code_layout_key == constant.code_key,
                  "constant provenance is not backed by the independently loaded CodeWitness")
            _need(len(constant.constant_path) == 1, "this capability only indexes direct code constants")
            if literal_constant:
                from scar.analysis.static.lowering.source_constant_lowering import witnessed_constant
                witnessed_constant(evidence, node, index=constant.constant_path[0])
                continue
            candidates = [row for row in dis.get_instructions(evidence.code)
                if row.opname == "LOAD_CONST" and row.arg == constant.constant_path[0]
                and tuple(row.positions) == (node.lineno, node.end_lineno, node.col_offset, node.end_col_offset)]
            allowed = (CodeType, str, tuple) if class_constant else (CodeType, tuple) if function_constant else (str, tuple) if call_constant else (slice, tuple)
            _need(len(candidates) == 1 and type(candidates[0].argval) in allowed
                  and (not import_constant or type(candidates[0].argval) is tuple
                       and candidates[0].argval == tuple(alias.name for alias in node.names))
                  and (not call_constant or type(candidates[0].argval) is str or candidates[0].argval == ()),
                  "source constant does not name its exact witnessed LOAD_CONST slot")
    call_protocols = validate_call_protocols(model, graph, interface, metadata, semantic, language,
        source_tree=source_tree, options=options, structural_only=structural_only)
    from scar.analysis.static.lowering.source_string import validate_string_source
    string_protocols = validate_string_source(model, graph, interface, metadata, semantic,
        source_tree=source_tree, options=options, structural_only=structural_only, language=language)
    tuple_ownership = None
    if Capability.SLICE_KEYS in language.capabilities:
        from scar.analysis.static.lowering.source_displays import validate_tuple_key_ownership
        tuple_ownership = validate_tuple_key_ownership(model, graph, interface, metadata, semantic,
            source_tree=source_tree, options=options, structural_only=structural_only, language=language)
    _need(len({row.name for row in language.compiler_locals}) == len(language.compiler_locals),
          "compiler-local names are not uniquely classified")
    for mapping in language.compiler_locals:
        _need(Capability.LEXICAL_LOCALS in language.capabilities, "compiler local mapping lacks explicit capability")
        _need(bool(mapping.lexical_slots) and len(set(mapping.lexical_slots)) == len(mapping.lexical_slots),
              "compiler local has no unique original lexical slots")
        for slot in mapping.lexical_slots:
            _need(slot in semantic.slots and semantic.slots[slot].name == mapping.name,
                  "compiler-local mapping introduced a non-source slot")
        if mapping.category == "LEXICAL_FUNCTION":
            _need(all(semantic.slots[slot].owner == interface.definition for slot in mapping.lexical_slots),
                  "function local points to another lexical scope")
        else:
            _need(mapping.comp_iteration or mapping.comp_cell, "transient local has no compiler comprehension classification")
            _need(all(semantic.slots[slot].metadata.get("scope_kind") == "comprehension" for slot in mapping.lexical_slots),
                  "transient comprehension local was projected as an outer slot")
            for slot in mapping.lexical_slots:
                owner, seen = semantic.slots[slot].owner, set()
                while owner in semantic.definitions and owner not in seen and owner != interface.definition:
                    seen.add(owner)
                    definition = semantic.definitions[owner]
                    _need(getattr(definition.kind, 'value', definition.kind) not in ('function', 'method', 'class'),
                          "comprehension local crossed a different function/class scope")
                    owner = definition.parent_id
                _need(owner == interface.definition, "comprehension local belongs to another function")
    return {"valid": True, "verification": "STRUCTURE" if structural_only else "SOURCE_AND_CODE_WITNESS",
            "source_shape_verified": not structural_only,
            "code_constant_provenance_verified": bool(constants) and not structural_only,
            "runtime_code_instance_established": False, "effects_established": False,
            "tuple_key_pending_references": tuple_ownership, "kwargs_call_protocols": call_protocols,
            "string_protocols": string_protocols}


__all__ = [
    'SourceCodeEvidence',
    'SourceLanguageOptions',
    'require_source_tree',
    'supports_cpython_compiler',
    'validate_language_body',
    'function_exception_types',
]
