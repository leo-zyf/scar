"""Static class construction."""

from __future__ import annotations

import hashlib
import json
import gzip
import io
import re
import os
import shutil
import tempfile
from scar.ir.program.class_builder import (
    ClassBuilderCompletion,
    ClassBuilderExpansion,
    ClassBuilderInput,
    ClassBuilderProvider,
    ClassBuilderStage,
    ClassBuilderStageKind as K,
    ClassSuiteCompletion,
    ClassSuiteInput,
    StaticClassBuilderBinding,
    decode_class_builder_provider,
)
from scar.ir.program.class_entries import (
    ClassSuiteOverlay,
    ClassSuiteMethodOverlay,
    decode_class_entry_overlay,
)
from scar.ir.program.ids import BlockID, LocationID, OperationID, RegionID, UseID, ValueID
from scar.ir.program.model import (
    AccessMode,
    Behavior,
    Block,
    BlockArgument,
    CompletionResult,
    CompletionSignature,
    CompletionSuccessor,
    ExitKind,
    ExitPort,
    Operation,
    ProgramGraph,
    ReferenceValue,
    OperationResult,
    Region,
    Stability,
    StateAccess,
    StateLocation,
    StateVersionValue,
    Terminator,
    TerminatorKind,
    Use,
    UseRole,
    ValueKind,
    ValueType,
    value_type,
)
from scar.ir.program.providers import ProviderArtifact
from scar.ir.program.source_subjects import FrameKind, SourceCompilationFrameRoot
from scar.ir.program.validation import require_valid
from collections import Counter
from dataclasses import fields
from pathlib import Path
from time import monotonic
from scar.ir.program.codec import loads as load_core
from scar.ir.program.entry_namespaces import decode_frame_body
from scar.ir.program.source_language import decode_language_overlay
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.record_codec import decode, encode, loads as strict_loads


# Construct a conditional CPython class builder and connect a shared suite.
#
# No application import, metaclass, descriptor, decorator or suite is evaluated.
# The implementation graph is an abstract, versioned protocol expansion of the
# standard builder, not a claim that the caller selected it. Callback internals,
# fine-grained effects and target/source identity remain explicit obligations.
IMPLEMENTATION = ProviderArtifact(
    "https://raw.githubusercontent.com/python/cpython/v3.14.7/Python/bltinmodule.c",
    "25234c39aafcd1da77b9cf94938d9e72ffcbee9e76aed34fa6b235cc8ffe970a",
    104011, "cpython.source-c/3.14.7")
RULE = "CPYTHON_BUILD_CLASS_3147_V1"
REF = ValueType(ValueKind.REFERENCE)


def _need(ok, detail):
    if not ok:
        raise ValueError("class builder connection: " + detail)


class _Builder:
    """Small provider producer; core def/use and CFG remain authoritative."""
    def __init__(self, base_count, keyword_names, freevars):
        signature = json.dumps([RULE, base_count, keyword_names, freevars], separators=(",", ":"))
        self.prefix = "build-class-" + hashlib.sha256(signature.encode()).hexdigest()[:24]
        self.serial = 0
        self.region = self.fresh(RegionID, "region")
        self.location = self.fresh(LocationID, "world")
        self.state_type = ValueType(ValueKind.STATE, location=self.location)
        self.blocks, self.arguments, self.block_ops = [], {}, {}
        self.operations, self.values, self.stages, self.cleanups = [], [], [], []
        self.entry = self.block((REF,) * (2 + base_count + len(keyword_names) + bool(keyword_names))
                                + (self.state_type,))
        args = self.arguments[self.entry]
        self.body_function, self.name = args[:2]
        self.bases = args[2:2 + base_count]
        self.keywords = args[2 + base_count:2 + base_count + len(keyword_names)]
        self.keyword_table = args[-2] if keyword_names else None
        self.world = args[-1]
        self.current, self.state = self.entry, self.world

    def fresh(self, kind, role):
        self.serial += 1
        return kind(self.prefix + ":" + role + ":" + str(self.serial))

    def block(self, types):
        block = self.fresh(BlockID, "block")
        self.blocks.append(block)
        result = []
        for index, typ in enumerate(types):
            identity = self.fresh(ValueID, "argument")
            origin = BlockArgument(block, index)
            value = (StateVersionValue(identity, self.region, origin, self.location)
                     if typ.kind is ValueKind.STATE else ReferenceValue(identity, self.region, origin))
            self.values.append(value)
            result.append(identity)
        self.arguments[block], self.block_ops[block] = tuple(result), []
        return block

    def uses(self, identity, refs, world):
        return tuple(Use(self.fresh(UseID, "use"), value,
                         UseRole.STATE_CONTENT if index == len(refs) else UseRole.REFERENCE)
                     for index, value in enumerate((*refs, world)))

    def emit(self, operation):
        self.operations.append(operation)
        self.block_ops[self.current].append(operation.id)

    def cleanup(self, kind, refs):
        # This abstract native cleanup preserves the pending completion and
        # can run destructors/unraisable hooks. It is not effect-free and is
        # not a sequence of fabricated application calls or FD operations.
        identity = self.fresh(OperationID, kind.value)
        uses = self.uses(identity, refs, self.state)
        after = self.fresh(ValueID, "cleanup-world")
        self.values.append(StateVersionValue(after, self.region, OperationResult(identity, 0), self.location))
        self.emit(Operation(identity, kind.value, (Behavior.READ, Behavior.WRITE, Behavior.MEMORY), uses,
            (after,), (StateAccess(self.location, uses[-1].id, AccessMode.READ_WRITE, after),),
            effects_complete=False))
        self.cleanups.append(identity)
        self.state = after

    def exit(self, kind, refs):
        identity = self.fresh(OperationID, "exit")
        uses = self.uses(identity, refs, self.state)
        self.emit(Operation(identity, "provider " + kind.value.lower() + " exit", (Behavior.CONTROL,), uses,
            effects_complete=True, terminator=Terminator(
                TerminatorKind.RETURN if kind is ExitKind.NORMAL else TerminatorKind.RAISE,
                exit="normal" if kind is ExitKind.NORMAL else "raise", values=tuple(u.id for u in uses))))

    def invoke(self, kind, refs, result_count, owned=()):
        identity = self.fresh(OperationID, kind.value)
        uses = self.uses(identity, refs, self.state)
        types = (REF,) * result_count + (self.state_type,)
        raised_types = (REF, REF, self.state_type)
        normal, raised = self.block(types), self.block(raised_types)
        signatures = (CompletionSignature(ExitKind.NORMAL, types), CompletionSignature(ExitKind.RAISE, raised_types))
        self.emit(Operation(identity, kind.value,
            (Behavior.CONTROL, Behavior.READ, Behavior.WRITE, Behavior.MEMORY), uses,
            effects_complete=False, terminator=Terminator(TerminatorKind.INVOKE,
                tuple(CompletionSuccessor(signature.kind, block,
                    tuple(CompletionResult(i) for i in range(len(signature.types))))
                    for signature, block in zip(signatures, (normal, raised)))),
            completions=signatures, invoke_arguments=tuple(u.id for u in uses)))
        normal_values, raised_values = self.arguments[normal], self.arguments[raised]
        self.stages.append(ClassBuilderStage(kind, identity, tuple(u.id for u in uses[:-1]), uses[-1].id,
            normal, normal_values[:-1], normal_values[-1], raised, *raised_values))
        self.current, self.state = raised, raised_values[-1]
        self.cleanup(K.RELEASE_RAISE, (*owned, *raised_values[:-1]))
        self.exit(ExitKind.RAISE, raised_values[:-1])
        self.current, self.state = normal, normal_values[-1]
        return normal_values[:-1]

    def run(self, keyword_names, freevars):
        func, name = self.body_function, self.name
        self.invoke(K.CHECK_INPUTS, (func, name), 0)
        (orig_bases,) = self.invoke(K.PACK_BASES, self.bases, 1)
        (bases,) = self.invoke(K.UPDATE_BASES, (orig_bases,), 1, (orig_bases,))
        table = (self.keyword_table,) if self.keyword_table is not None else ()
        meta, *keyword_refs = self.invoke(K.SELECT_METACLASS, (bases, *self.keywords, *table),
            1 + bool(self.keywords), (orig_bases, bases))
        # No keywords means a NULL C dictionary pointer, not a Python None
        # object. That case has no dictionary Value/Use at all. With keywords
        # this result is the real remaining dictionary (possibly empty).
        keywords = tuple(keyword_refs)
        namespace, = self.invoke(K.PREPARE_NAMESPACE, (meta, name, bases, *keywords), 1,
            (orig_bases, bases, meta, *keywords))
        self.invoke(K.CHECK_MAPPING, (namespace,), 0, (orig_bases, bases, meta, *keywords, namespace))
        # These are current native frame-field projections at suite entry,
        # after __mro_entries__/__prepare__ callbacks, not factory snapshots.
        code, globals_ref, builtins_ref, *closure = self.invoke(K.READ_FUNCTION_FRAME, (func,), 3 + len(freevars),
            (orig_bases, bases, meta, *keywords, namespace))
        cell, = self.invoke(K.EVAL_SUITE, (func, code, namespace, globals_ref, builtins_ref, *closure), 1,
            (orig_bases, bases, meta, *keywords, namespace))
        self.invoke(K.STORE_ORIGINAL_BASES, (namespace, orig_bases, bases), 0,
            (orig_bases, bases, meta, *keywords, namespace, cell))
        cls, = self.invoke(K.CALL_METACLASS, (meta, name, bases, namespace, *keywords), 1,
            (orig_bases, bases, meta, *keywords, namespace, cell))
        self.invoke(K.CHECK_CLASS_CELL, (cls, cell, name), 0,
            (orig_bases, bases, meta, *keywords, namespace, cell, cls))
        self.cleanup(K.RELEASE_NORMAL, (cell, namespace, meta, *keywords, orig_bases, bases))
        self.exit(ExitKind.NORMAL, (cls,))
        graph = ProgramGraph(self.region, (Region(self.region, "conditional CPython __build_class__", self.entry,
            tuple(Block(b, self.arguments[b], tuple(self.block_ops[b])) for b in self.blocks),
            (ExitPort("normal", ExitKind.NORMAL, (REF, self.state_type)),
             ExitPort("raise", ExitKind.RAISE, (REF, REF, self.state_type)))),),
            tuple(self.operations), tuple(self.values),
            (StateLocation(self.location, "coarse builder/callback/external world", stability=Stability.UNKNOWN),))
        require_valid(graph)
        return ClassBuilderProvider(IMPLEMENTATION, RULE, len(self.bases), keyword_names, freevars, graph,
            func, name, self.bases, self.keywords, self.keyword_table, self.world,
            tuple(self.stages), tuple(self.cleanups))


def build_class_provider(*, base_count, keyword_names=(), freevars=()):
    """Build the same finite shared provider graph for a given ABI signature."""
    _need(type(base_count) is int and base_count >= 0, "base count must be a nonnegative integer")
    _need(type(keyword_names) is tuple and all(type(n) is str and n for n in keyword_names)
          and len(set(keyword_names)) == len(keyword_names), "keyword names must be unique ordered strings")
    _need(type(freevars) is tuple and all(type(n) is str and n for n in freevars)
          and len(set(freevars)) == len(freevars), "freevars must retain the exact ordered layout")
    return _Builder(base_count, keyword_names, freevars).run(keyword_names, freevars)


def _original_ports(parent, construction, suite, overlay):
    """Resolve original boundaries; no protocol outputs are minted here."""
    require_valid(parent)
    require_valid(suite)
    _need(type(overlay) in (ClassSuiteOverlay, ClassSuiteMethodOverlay), "shared body needs a class suite overlay")
    _need(construction.suite == overlay.compilation and construction.definition == overlay.definition
          and construction.source == overlay.source, "parent and suite compiler/source identity differ")
    _need(construction.suite.compiler_implementation == "cpython"
          and construction.suite.compiler_version == (3, 14, 7), "suite requires the declared CPython 3.14.7 ABI")
    _need(len(suite.frame_roots) == 1 and type(suite.frame_roots[0]) is SourceCompilationFrameRoot
          and suite.frame_roots[0].kind is FrameKind.CLASS
          and suite.frame_roots[0].code == overlay.compilation and suite.frame_roots[0].entry == overlay.interface,
          "suite original frame root differs from class ABI")
    parent_operations = {op.id: op for op in parent.operations}
    parent_blocks = {b.id: b for r in parent.regions for b in r.blocks}
    call = construction.builder_call
    operation = parent_operations.get(call.operation)
    _need(operation is not None and operation.terminator is not None
          and operation.terminator.kind is TerminatorKind.INVOKE and operation.callee is None
          and not operation.effects_complete, "original unknown builder INVOKE must remain intact")
    uses = {u.id: u for u in operation.uses}
    _need(tuple(uses) == (call.callee, *(a.use for a in call.arguments),
          *((call.keyword_names,) if call.keyword_names is not None else ()), call.world),
          "parent selected callee/actual/world occurrences differ")
    normal, raised = operation.terminator.successors
    _need(normal.kind is ExitKind.NORMAL and raised.kind is ExitKind.RAISE
          and normal.block == call.normal and raised.block == call.raised
          and parent_blocks[normal.block].arguments == (call.result, call.normal_world)
          and parent_blocks[raised.block].arguments == (call.exception, call.traceback, call.raised_world),
          "parent completion endpoints differ")
    _need(len(call.arguments) >= 2 and uses[call.arguments[0].use].value == construction.body_function,
          "builder body operand is not the original factory result")
    factory = parent_operations.get(construction.factory)
    _need(factory is not None and factory.terminator is not None
        and factory.terminator.kind is TerminatorKind.INVOKE
        and tuple(u.id for u in factory.uses[:-1]) == (
        construction.factory_code, construction.factory_globals, *construction.factory_closure),
        "body factory code/globals/closure occurrences differ")
    factory_normal = next(e for e in factory.terminator.successors if e.kind is ExitKind.NORMAL)
    _need(parent_blocks[factory_normal.block].arguments[0] == construction.body_function,
          "body function is not the factory normal result")
    entry = next(r for r in suite.regions if r.id == suite.root).entry
    block = next(b for r in suite.regions for b in r.blocks if b.id == entry)
    mapping = overlay.interface.mapping
    formals = (mapping.locals_namespace, mapping.globals_namespace, mapping.builtins_namespace,
               *mapping.closure_cells, mapping.world)
    _need(block.arguments == formals, "suite ABI is not its original ordered entry interface")
    return operation, uses, mapping


def connect_class_suite(*, parent_namespace, parent_graph, construction,
                        suite_namespace, suite_graph, suite_overlay):
    """Produce qualified parent -> provider -> shared suite port relations.

    The owning reader must verify namespaces and their artifact receipts from
    original source/body inputs before calling this constructor. This routine
    checks actual core endpoints and compiler provenance; it is not an artifact
    loader and does not convert its still-present guards into proven facts.
    """
    _, uses, mapping = _original_ports(parent_graph, construction, suite_graph, suite_overlay)
    _need(parent_namespace.region == parent_graph.root and suite_namespace.region == suite_graph.root
          and suite_namespace.definition == suite_overlay.definition and suite_namespace.source == suite_overlay.source,
          "qualified namespaces differ from supplied original graphs")
    actuals = construction.builder_call.arguments
    positional = tuple(a for a in actuals if a.keyword is None)
    keyword_actuals = tuple(a for a in actuals if a.keyword is not None)
    _need(tuple(actuals) == (*positional, *keyword_actuals), "builder inputs must retain positional then keyword order")
    _need(bool(keyword_actuals) == (construction.builder_call.keyword_names is not None),
          "builder keyword table differs from original named actuals")
    provider = build_class_provider(base_count=len(positional) - 2,
        keyword_names=tuple(a.keyword for a in keyword_actuals), freevars=suite_overlay.interface.co_freevars)
    formals = (provider.body_function, provider.name, *provider.bases, *provider.keywords)
    inputs = tuple(ClassBuilderInput(a.use, uses[a.use].value, formal) for a, formal in zip(actuals, formals))
    call = construction.builder_call
    if call.keyword_names is not None:
        inputs += (ClassBuilderInput(call.keyword_names, uses[call.keyword_names].value, provider.keyword_table),)
    inputs += (ClassBuilderInput(call.world, uses[call.world].value, provider.world),)
    stage = next(s for s in provider.stages if s.kind is K.EVAL_SUITE)
    provider_ops = {op.id: op for op in provider.graph.operations}
    eval_uses = {u.id: u for u in provider_ops[stage.operation].uses}
    # First eval operands are body function and current code. Namespace/frame
    # roles begin after them; saved B is a function-field result, never parent B.
    role_names = ("locals", "globals", "builtins", *("closure" for _ in mapping.closure_cells), "world")
    targets = (mapping.locals_namespace, mapping.globals_namespace, mapping.builtins_namespace,
               *mapping.closure_cells, mapping.world)
    source_uses = (*stage.inputs[2:], stage.world)
    suite_inputs = tuple(ClassSuiteInput(role, use, eval_uses[use].value, formal)
        for role, use, formal in zip(role_names, source_uses, targets))
    root = next(r for r in suite_graph.regions if r.id == suite_graph.root)
    normal = next(e for e in root.exits if e.kind is ExitKind.NORMAL)
    raised = next(e for e in root.exits if e.kind is ExitKind.RAISE)
    suite_completions = (
        ClassSuiteCompletion(ExitKind.NORMAL, normal.name, stage.normal, (*stage.results, stage.normal_world)),
        ClassSuiteCompletion(ExitKind.RAISE, raised.name, stage.raised,
            (stage.exception, stage.traceback, stage.raised_world)))
    completions = (
        ClassBuilderCompletion(ExitKind.NORMAL, "normal", call.normal, (call.result, call.normal_world)),
        ClassBuilderCompletion(ExitKind.RAISE, "raise", call.raised,
            (call.exception, call.traceback, call.raised_world)))
    return ClassBuilderExpansion(parent_namespace, construction, suite_namespace, provider,
        inputs, suite_inputs, suite_completions, completions)


def validate_class_builder_expansion(expansion, *, parent_graph, suite_graph, suite_overlay):
    """Re-derive all mappings and provider CFG from the original core inputs."""
    expansion.__post_init__()
    expected = connect_class_suite(parent_namespace=expansion.parent, parent_graph=parent_graph,
        construction=expansion.construction, suite_namespace=expansion.suite,
        suite_graph=suite_graph, suite_overlay=suite_overlay)
    _need(expansion == expected, "stored provider graph, guards or original port mappings differ")
    return expansion


# Publish/read class construction relations over saved static source graphs.
#
# The caller owns atomic model publication. This module only writes its fresh
# staging directory; all original source and canonical body artifacts stay intact.
def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode() + b"\n"


def _constructions(body):
    if body.module is not None:
        return tuple(getattr(body.module, "class_constructions", ()))
    language = body.document.get("language")
    return tuple(decode_language_overlay(language).class_constructions) if language else ()


def build_static_class_links(directory, out, *, logical_root=None, expected_manifest_sha256=None,
                             budget=None, progress=None, activity=None):
    """Write provider and parent-owned relation shards, never copy a suite.

    Descriptor paths are relative to ``out``. ``logical_root`` follows the
    existing atomic-publisher convention: original namespaces name the final
    model directory, even while their bytes are read from staging. The budget
    is the existing StaticCallBudget reader/cache/time/byte policy; no CALL
    detection or target-program execution is performed here.
    """
    from scar.analysis.static.calls import StaticCallSession
    started = monotonic()
    counts, frontiers, shards, providers = Counter(), [], [], {}
    session = StaticCallSession(directory, out=out, logical_root=logical_root,
        expected_manifest_sha256=expected_manifest_sha256, budget=budget, activity=activity)
    counts["declared_class_bodies"] = sum(row["kind"] == "CLASS" for u in session.units for row in u["bodies"])
    visited, issue, enumeration_complete = set(), None, False

    def body(context, number, row):
        if activity is not None:
            activity({"stage": "STATIC_CLASS_BODY", "source_unit": context.number,
                "path": context.unit["path"], "entry_key": row["entry_key"], "body_kind": row["kind"],
                "body_records_read": len(visited), "connected_class_constructions": counts["connected_class_constructions"]})
        key = context.number, number
        if key not in visited:
            if len(visited) >= session.budget.max_bodies:
                raise OverflowError("static class body-read budget exhausted")
            visited.add(key)
        return session._body(context, number, row)

    def gap(kind, detail, unit, entry, definition=None, source=None):
        counts["frontier_" + kind] += 1
        frontiers.append({"kind": kind, "detail": detail, "unit_key": unit["key"],
                          "entry_key": entry, "definition": encode(definition) if definition is not None else None,
                          "source": source})

    def finish_unit(number, unit):
        counts["source_units_processed"] += 1
        if progress is not None:
            progress({"stage": "STATIC_CLASS_RELATIONS", "source_units_processed": number + 1,
                "source_units_total": len(session.units), "last_source_path": unit["path"],
                "class_declarations_cataloged": counts["class_declarations_cataloged"],
                "connected_class_constructions": counts["connected_class_constructions"]})

    try:
        with session:
            for number, unit in enumerate(session.units):
                session._check()
                if activity is not None:
                    activity({"stage": "STATIC_CLASS_SOURCE_UNIT", "source_unit": number,
                        "path": unit["path"], "source_units_processed": counts["source_units_processed"],
                        "source_units_total": len(session.units),
                        "connected_class_constructions": counts["connected_class_constructions"]})
                if not any(row["kind"] == "CLASS" for row in unit["bodies"]):
                    finish_unit(number, unit)
                    continue
                if "lexical" not in unit["artifacts"]:
                    for row in unit["bodies"]:
                        if row["kind"] == "CLASS":
                            gap("CLASS_LEXICAL_CONTEXT_UNAVAILABLE", "Original class body declaration has no saved lexical catalog.",
                                unit, row["entry_key"], source=row.get("source"))
                    finish_unit(number, unit)
                    continue
                catalog = session._read(unit["artifacts"]["lexical"])
                if (catalog.get("schema") != "scar.static-lexical-catalog" or catalog.get("schema_version") != 1
                        or catalog.get("path") != unit["path"] or catalog.get("namespace_root") != unit["namespace_root"]
                        or catalog.get("fingerprint") != "sha256:" + unit["decoded_sha256"]):
                    raise ValueError("class lexical catalog differs from its original source unit")
                entries = {row["entry_key"]: row for row in catalog["entries"]}
                if len(entries) != len(catalog["entries"]):
                    raise ValueError("class catalog has duplicate original entry identities")
                children = [row for row in entries.values() if row["kind"] == "CLASS"]
                counts["class_declarations_cataloged"] += len(children)
                missing = [name for name in ("source", "semantic", "semantics", "control") if name not in unit["artifacts"]]
                if missing:
                    kind = "CLASS_STUB_DECLARATION_ONLY" if Path(unit["path"]).suffix == ".pyi" else "CLASS_SOURCE_CONTEXT_UNAVAILABLE"
                    for child in children:
                        gap(kind, "Original declaration retained; executable source context is not supplied: " + ", ".join(missing),
                            unit, child["entry_key"], source=child.get("source_ref"))
                    finish_unit(number, unit)
                    continue
                child_by_definition = {}
                for child in children:
                    identities = child.get("semantic_definition_ids") or ()
                    if not identities:
                        gap("CLASS_SEMANTIC_DEFINITION_UNAVAILABLE", "Original class declaration has no source-backed semantic definition ID.",
                            unit, child["entry_key"], source=child.get("source_ref"))
                    for identity in identities:
                        key = json.dumps(identity, sort_keys=True)
                        if key in child_by_definition:
                            raise ValueError("one original class definition belongs to multiple lexical declarations")
                        child_by_definition[key] = child
                parents = dict.fromkeys(child["lexical_parent"] for child in children)
                rows = {row["entry_key"]: (i, row) for i, row in enumerate(unit["bodies"])}
                context = session._unit(number)
                seen_definitions = set()
                for key in parents:
                    if key not in rows:
                        gap("PARENT_ENTRY_UNAVAILABLE", "Original lexical parent has no canonical body row.", unit, key)
                        continue
                    parent = body(context, *rows[key])
                    if parent is None:
                        gap("PARENT_GRAPH_UNAVAILABLE", "Original parent body has no saved canonical graph.", unit, key)
                        continue
                    counts["parent_graphs_processed"] += 1
                    bindings = []
                    for construction in _constructions(parent):
                        counts["parent_constructions"] += 1
                        identity = json.dumps(encode(construction.definition), sort_keys=True)
                        seen_definitions.add(identity)
                        child = child_by_definition.get(identity)
                        if child is None or child["lexical_parent"] != key:
                            gap("SUITE_SOURCE_UNPAIRED", "Construction has no matching original lexical class child.",
                                unit, key, construction.definition)
                            continue
                        selected = rows.get(child["entry_key"])
                        suite = body(context, *selected) if selected is not None else None
                        if suite is None or type(suite.class_entry) not in (ClassSuiteOverlay, ClassSuiteMethodOverlay):
                            gap("SUITE_BODY_UNEXPANDED", "Original class has no expanded shared suite graph.",
                                unit, child["entry_key"], construction.definition)
                            continue
                        expansion = connect_class_suite(parent_namespace=parent.namespace, parent_graph=parent.graph,
                            construction=construction, suite_namespace=suite.namespace,
                            suite_graph=suite.graph, suite_overlay=suite.class_entry)
                        provider_wire = {"schema": "scar.static-class-provider", "schema_version": 1,
                                         "provider": encode(expansion.provider)}
                        payload = _canonical(provider_wire)
                        digest = hashlib.sha256(payload).hexdigest()
                        if digest not in providers:
                            descriptor = session._write("providers/" + digest + ".json.gz", payload)
                            artifact = ProviderArtifact(str(session.logical_out / descriptor["path"]),
                                descriptor["sha256"], descriptor["bytes"], "scar.static-class-provider/1")
                            providers[digest] = SourceArtifactLocator(artifact, descriptor["decoded_sha256"],
                                descriptor["decoded_bytes"], "gzip-canonical-json", "#/provider")
                        values = {f.name: getattr(expansion, f.name) for f in fields(expansion) if f.name != "provider"}
                        binding = StaticClassBuilderBinding(**values, provider=providers[digest])
                        bindings.append(binding)
                        counts["connected_class_constructions"] += 1
                        counts["parent_input_mappings"] += len(binding.inputs)
                        counts["suite_input_mappings"] += len(binding.suite_inputs)
                        counts["suite_completion_mappings"] += len(binding.suite_completions)
                        counts["parent_completion_mappings"] += len(binding.completions)
                        counts["connected_suites_with_residual"] += suite.class_entry.remaining is not None
                    if bindings:
                        shard = {"schema": "scar.static-class-bindings", "schema_version": 1,
                            "source_model": encode(session.basis), "parent": encode(parent.namespace),
                            "bindings": [encode(b) for b in bindings]}
                        descriptor = session._write("bindings/" + str(len(shards)).zfill(6) + ".json.gz", _canonical(shard))
                        shards.append({"artifact": descriptor, "artifact_path": descriptor["path"],
                            "parent": encode(parent.namespace), "parent_entry_key": key, "bindings": len(bindings)})
                for identity, child in child_by_definition.items():
                    if identity not in seen_definitions:
                        gap("PARENT_CONSTRUCTION_UNEXPANDED", "No original construction record for this lexical class.",
                            unit, child["entry_key"])
                finish_unit(number, unit)
            enumeration_complete = True
    except (TimeoutError, OverflowError) as error:
        issue = {"kind": type(error).__name__, "detail": str(error)}
    counts["body_records_read"] = len(visited)
    counts["shared_provider_graphs"] = len(providers)
    counts["published_parent_shards"] = len(shards)
    counts["published_bindings"] = sum(row["bindings"] for row in shards)
    return {"schema": "scar.static-class-publication", "schema_version": 1,
        "source_model": encode(session.basis), "input_manifest_sha256": session.manifest_sha256,
        "artifacts": session.artifacts, "shards": shards, "counts": dict(counts), "frontiers": frontiers,
        "scope": {"enumeration_complete": enumeration_complete, "conditional_builder_selection": True,
            "whole_program_model_complete": False, "effects_closed": False, "target_executed": False},
        "issue": issue, "analyzer_seconds": monotonic() - started}


def _read_artifact(artifact, *, max_bytes):
    path = Path(artifact.uri)
    if not path.is_absolute() or path.resolve() != path or path.is_symlink() or not path.is_file():
        raise ValueError("class index artifact is absent or traverses a symlink")
    with path.open("rb") as stream:
        stored = stream.read(min(artifact.size_bytes, max_bytes) + 1)
    if (len(stored) != artifact.size_bytes or len(stored) > max_bytes
            or hashlib.sha256(stored).hexdigest() != artifact.sha256):
        raise ValueError("class index stored receipt differs or exceeds budget")
    return stored


def _read_json_artifact(artifact, *, max_bytes):
    stored = _read_artifact(artifact, max_bytes=max_bytes)
    if stored[:2] == b"\x1f\x8b":
        with gzip.GzipFile(fileobj=io.BytesIO(stored)) as stream:
            raw = stream.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise OverflowError("class index decoded artifact exceeds budget")
    else:
        raw = stored
    return strict_loads(raw.decode()), raw


def read_static_class_links(directory, publication, *, index_directory=None, logical_root=None,
                            max_bytes=256_000_000):
    """Freshly read stored bindings and re-derive them from original inputs.

    ``directory`` is the original program model. An unpublished staging index
    may supply ``index_directory``; otherwise it is the source-basis parent.
    This reader executes no target or saved validation result.
    """
    from scar.analysis.static.calls import input_basis_unit
    from scar.analysis.static.basis_compatibility import published_units_basis_compatible
    from scar.analysis.static_body import diagnostic_materializations
    from scar.analysis.static.query import _Fragments
    from scar.analysis.static_program import read_static_program
    root = Path(directory).absolute()
    manifest = read_static_program(root, verify=False)
    if publication.get("schema") != "scar.static-class-publication" or publication.get("schema_version") != 1:
        raise ValueError("unsupported static class publication")
    basis_artifact = decode(ProviderArtifact, publication["source_model"])
    logical_index = Path(basis_artifact.uri).parent
    index_root = Path(index_directory).absolute() if index_directory is not None else logical_index
    logical_model = Path(logical_root).absolute() if logical_root is not None else root

    def index_artifact(artifact):
        from dataclasses import replace
        try:
            relative = Path(artifact.uri).relative_to(logical_index)
        except ValueError:
            raise ValueError("provider artifact escapes its owned class index") from None
        return replace(artifact, uri=str(index_root / relative))

    basis, _ = _read_json_artifact(index_artifact(basis_artifact), max_bytes=max_bytes)
    if (basis.get("schema") != "scar.static-call-source-basis" or basis.get("schema_version") != 1
            or basis["logical_root"] != str(logical_model)
            or not published_units_basis_compatible(basis["units"],
                [input_basis_unit(u) for u in manifest["units"]],
                source_expansion=manifest.get('source_expansion'),
                diagnostic_body=diagnostic_materializations(manifest, _Fragments(root).read))):
        raise ValueError("class source basis differs from current original inputs")
    cache, providers, catalogs = {}, {}, {}

    def original(namespace):
        if namespace in cache:
            return cache[namespace]
        locator = namespace.graph
        declaration = locator.declaration
        match = re.fullmatch(r"#/units/(\d+)/bodies/(\d+)/artifact", declaration.locator) if declaration else None
        if declaration is None or declaration.artifact != basis_artifact or match is None:
            raise ValueError("class namespace has no original basis declaration")
        i, j = map(int, match.groups())
        unit = basis["units"][i]
        row = unit["bodies"][j]
        descriptor = row["artifact"]
        if (locator.artifact.uri != str(logical_model / descriptor["path"])
                or locator.artifact.sha256 != descriptor["sha256"] or locator.artifact.size_bytes != descriptor["bytes"]
                or locator.decoded_sha256 != descriptor["decoded_sha256"]
                or locator.decoded_bytes != descriptor["decoded_bytes"] or locator.pointer != "#/graph"):
            raise ValueError("class namespace receipt differs from original body declaration")
        from dataclasses import replace
        physical = replace(locator.artifact, uri=str(root / descriptor["path"]))
        document, raw = _read_json_artifact(physical, max_bytes=max_bytes)
        if len(raw) != locator.decoded_bytes or hashlib.sha256(raw).hexdigest() != locator.decoded_sha256:
            raise ValueError("class original decoded body receipt differs")
        graph = load_core(json.dumps(document["graph"], allow_nan=False))
        if i not in catalogs:
            lexical = unit["artifacts"]["lexical"]
            lexical_artifact = ProviderArtifact(str(root / lexical["path"]), lexical["sha256"],
                lexical["bytes"], "scar.static-lexical")
            lexical_document, lexical_raw = _read_json_artifact(lexical_artifact, max_bytes=max_bytes)
            if (len(lexical_raw) != lexical["decoded_bytes"]
                    or hashlib.sha256(lexical_raw).hexdigest() != lexical["decoded_sha256"]):
                raise ValueError("class original lexical receipt differs")
            catalogs[i] = {entry["entry_key"]: entry for entry in lexical_document["entries"]}
        lexical_entry = catalogs[i][row["entry_key"]]
        if (namespace.region != graph.root or namespace.entry != document["metadata"]["entry"]
                or encode(namespace.definition) not in (lexical_entry.get("semantic_definition_ids") or ())
                or encode(namespace.source) != lexical_entry.get("source_ref")):
            raise ValueError("class body namespace differs from original interface")
        if namespace.graph_representation != "scar.program-graph/" + str(document["graph"]["schema_version"]):
            raise ValueError("class original core representation differs")
        if document.get("class") is not None:
            overlay = decode_class_entry_overlay(document["class"])
            constructions = ()
        elif document.get("module") is not None:
            overlay = None
            constructions = tuple(getattr(decode_frame_body(document["module"]), "class_constructions", ()))
        else:
            overlay = None
            constructions = tuple(decode_language_overlay(document["language"]).class_constructions) if document.get("language") else ()
        cache[namespace] = graph, overlay, constructions
        return cache[namespace]

    result = []
    for shard in publication["shards"]:
        descriptor = shard["artifact"]
        path = Path(descriptor["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("class shard escapes its index")
        artifact = ProviderArtifact(str(index_root / path), descriptor["sha256"], descriptor["bytes"],
                                    "scar.static-class-bindings/1")
        document, raw = _read_json_artifact(artifact, max_bytes=max_bytes)
        if (len(raw) != descriptor["decoded_bytes"] or hashlib.sha256(raw).hexdigest() != descriptor["decoded_sha256"]
                or document.get("schema") != "scar.static-class-bindings" or document.get("schema_version") != 1
                or document["source_model"] != publication["source_model"]):
            raise ValueError("class binding shard receipt/schema/basis differs")
        for record in document["bindings"]:
            binding = decode(StaticClassBuilderBinding, record)
            if encode(binding.parent) != document["parent"] or encode(binding.parent) != shard["parent"]:
                raise ValueError("class binding is owned by a different parent shard")
            locator = binding.provider
            if locator not in providers:
                provider_document, raw = _read_json_artifact(index_artifact(locator.artifact), max_bytes=max_bytes)
                if (len(raw) != locator.decoded_bytes or hashlib.sha256(raw).hexdigest() != locator.decoded_sha256
                        or locator.pointer != "#/provider" or provider_document.get("schema") != "scar.static-class-provider"
                        or provider_document.get("schema_version") != 1):
                    raise ValueError("class provider decoded receipt/schema differs")
                providers[locator] = decode_class_builder_provider(provider_document["provider"])
            parent_graph, _, constructions = original(binding.parent)
            suite_graph, suite_overlay, _ = original(binding.suite)
            if binding.construction not in constructions:
                raise ValueError("class binding is not an original parent construction")
            values = {f.name: getattr(binding, f.name) for f in fields(binding) if f.name != "provider"}
            expansion = ClassBuilderExpansion(**values, provider=providers[locator])
            validate_class_builder_expansion(expansion, parent_graph=parent_graph,
                suite_graph=suite_graph, suite_overlay=suite_overlay)
            result.append(expansion)
    if len(result) != publication["counts"]["published_bindings"]:
        raise ValueError("class binding publication count differs")
    return tuple(result)


# Atomically publish class interfaces beside original current-model graphs.
def _producer():
    from scar.analysis.static_body import _producer as shared_producer
    analysis = Path(__file__).parent
    return {"shared": shared_producer(), "class_sources": {
        name: hashlib.sha256((analysis / name).read_bytes()).hexdigest()
        for name in ("static_class_construction.py",
                     "static/calls.py", "static/function_environment.py", "function_interfaces.py", "source_layers.py")}}


def link_static_class_program(directory, *, budget=None, progress=None):
    """Attach conditional parent/provider/suite interfaces, preserving calls.

    Only owned class-index files and the manifest change. Original source/body
    fragments are hard-linked, checked by their byte stamps, and never rebuilt.
    Missing declarations/interfaces remain exact scope frontiers. Bad original
    receipts or invalid graph records abort before current-model publication.
    """
    from scar.analysis.static_program import (_artifacts, _discard_static_classes,
        _json, read_static_program)
    from scar.analysis.static_body import _stamp
    from scar.analysis.static.calls import StaticCallBudget
    from scar.analysis.static_class_construction import build_static_class_links
    from scar.analysis.static.jobs import StaticLinkTask

    producer = _producer()
    output = Path(directory).absolute()
    manifest = read_static_program(output, verify=False)
    original = (output / "manifest.json").read_bytes()
    if json.loads(original) != manifest:
        raise ValueError("current manifest changed during class publication read")
    budget = budget or StaticCallBudget()
    task = StaticLinkTask(output, input_manifest_sha256=hashlib.sha256(original).hexdigest(),
                          budget=budget, family="classes")

    def report(event):
        task.activity(event)
        if progress is not None:
            progress(event)

    temporary = Path(tempfile.mkdtemp(prefix=".static-classes-", dir=output.parent))
    try:
        task.activity({"stage": "STAGING", "directory": str(temporary)})
        shutil.copytree(output, temporary, dirs_exist_ok=True, copy_function=os.link)
        _discard_static_classes(temporary, manifest)
        shared = {descriptor["path"]: _stamp(temporary / descriptor["path"])
                  for descriptor in _artifacts(manifest)}
        (temporary / "manifest.json").unlink()
        (temporary / "manifest.json").write_bytes(_json(manifest))
        input_hash = hashlib.sha256((temporary / "manifest.json").read_bytes()).hexdigest()
        result = build_static_class_links(temporary, temporary / "indexes" / "classes",
            logical_root=output, expected_manifest_sha256=input_hash, budget=budget,
            progress=report, activity=task.activity)
        if _producer() != producer:
            raise ValueError("analyzer source changed during static class linking")
        manifest["static_classes"] = {**result, "producer": producer, "artifacts": [
            {**descriptor, "path": "indexes/classes/" + descriptor["path"]}
            for descriptor in result["artifacts"]]}
        if any(_stamp(temporary / path) != stamp for path, stamp in shared.items()):
            raise ValueError("an original graph/index artifact changed during class linking")
        manifest.setdefault("completion", {}).update(static_class_links_present=True,
            whole_program_model_complete=False)
        manifest["storage"] = {"files": len(list(temporary.rglob("*.gz"))),
            "compressed_bytes": sum(path.stat().st_size for path in temporary.rglob("*.gz"))}
        task.activity({"stage": "PUBLICATION", "counts": result["counts"],
                       "enumeration_complete": result["scope"]["enumeration_complete"]})
        (temporary / "manifest.json").write_bytes(_json(manifest))
        read_static_program(temporary, verify=False)
        read_static_program(output, verify=False)
        if (output / "manifest.json").read_bytes() != original:
            raise ValueError("current model changed during static class linking")
        previous = Path(tempfile.mkdtemp(prefix=".static-replacing-", dir=output.parent))
        previous.rmdir()
        output.rename(previous)
        try:
            temporary.rename(output)
        except BaseException:
            previous.rename(output)
            raise
        shutil.rmtree(previous)
        task.published(manifest, hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest())
        return manifest
    except BaseException as error:
        shutil.rmtree(temporary, ignore_errors=True)
        task.failed(error)
        raise
