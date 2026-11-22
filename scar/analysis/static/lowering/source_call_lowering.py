"""Source call lowering."""

from __future__ import annotations

import ast
import dis
import argparse
import json
from dataclasses import replace
from scar.analysis.program_lowering import _REF, _Unsupported
from scar.analysis.static.lowering.source_heap_lowering import _source_definition, _associate
from scar.analysis.static.lowering.source_mapping_lowering import _transfer
from scar.ir.program.ids import OperationID, ValueID
from scar.ir.program.model import (
    AccessMode,
    Behavior,
    ExitKind,
    Operation,
    OperationResult,
    ReferenceValue,
    StateAccess,
    CompletionResult,
    StateVersionValue,
    TerminatorKind,
    ValueKind,
    ValueType,
    value_type,
)
from scar.ir.program.source import GapKind, SourceRole
from scar.ir.program.source_language import (
    CodeConstantSource,
    SourceLanguageCapability as Capability,
)
from scar.ir.program.structured import PythonProtocolKind as Kind
from scar.ir.semantics_v2 import Opcode
from scar.analysis.source_language import (
    supports_cpython_compiler,
    function_exception_types,
    SourceLanguageOptions,
    require_source_tree,
)
from pathlib import Path
from scar.analysis.entry_call_dependence import _index, _path
from scar.analysis.normal_graph import normal_edges
from scar.analysis.program_view import ProgramView
from scar.analysis.function_interface_observations import RetainedCodeBodyIndex
from scar.analysis.call_inputs import call_operands
from scar.analysis.source_references import source_port
from scar.compiler.call_binding import plan_call_bindings
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program import normal_graph as G
from scar.ir.program.call_edges import CapturedCode, InvocationKey
from scar.ir.program.control import LanguageCompletionKind
from scar.ir.program.normal_graph import GlobalPort
from scar.ir.program.binding_ports import ArgumentRole
from scar.ir.program.source_value_routes import SourceCallScope, SourceInvocationPort
from scar.ir.record_codec import decode, encode
from scar.runtime.source_call_handoff_observe import SCHEMA as CAPTURE
from scar.ir.v2.semantic import OperationKind


# Join a bounded live handoff to original F4 return and formal ports.
SCHEMA = 'scar.source-call-handoff'


def _require(condition, detail):
    if not condition:
        raise ValueError('source handoff: ' + detail)


def _one(rows, detail):
    rows = list(rows)
    _require(len(rows) == 1, detail)
    return rows[0]


def _body(index, code):
    descriptor = decode(CapturedCode, code)
    body, frontier = index.match(descriptor)
    _require(body is not None and frontier is None, 'call body does not resolve to its original F4 scope')
    proof = index.explain(descriptor)
    _require(len(proof['matches']) == 1, 'original compiler body correspondence is ambiguous')
    facts = index.view.unit(proof['matches'][0]['unit']).facts
    function = _one((row for row in facts.functions if encode(row.body_scope) == proof['matches'][0]['body_scope']),
                    'compiler body lacks its original source function')
    _require(list(function.parameter_names) == code['parameters'], 'original formal layout differs from captured code')
    return body, proof['matches'][0]['function_fact'], facts, function


def _plan(view, design, caller_facts, caller_fn):
    caller = design['caller']
    caller_ref = artifact_reference(caller['filename'], 'scar.python-source')
    producer, consumer = (design[role]['caller_site'] for role in ('producer', 'consumer'))
    _require(producer['code'] == consumer['code'] == caller['key'] and
             producer['opname'] == consumer['opname'] == 'CALL' and producer['offset'] < consumer['offset'],
             'a direct positional handoff needs ordered original CALL instructions')
    plan = plan_call_bindings(view, caller_ref['uri'], entry=caller['qualname'])
    producer_row, consumer_row = (
        _one((row for row in plan['sites'] if row['site'] == selected),
             'observed bytecode site has no unique original F4 CALL')
        for selected in (producer, consumer))
    _require(producer_row['unit'] == consumer_row['unit'] and
             producer_row['code_scope']['owner'] == consumer_row['code_scope']['owner'],
             'selected calls do not share the original caller body')
    unit = view.unit(producer_row['unit'])
    _require(caller_facts.unit == unit.unit and encode(caller_fn.body_scope.owner) == producer_row['code_scope']['owner'],
             'original compiled caller does not own the two source CALLs')
    producer_call = unit.facts.calls[producer_row['query']]
    consumer_call = unit.facts.calls[consumer_row['query']]
    _require(consumer_call.positional_pack is None and consumer_call.keyword_pack is None and
             not consumer_call.keywords,
             'source F4 call requires a different argument protocol')
    output = source_port(unit.facts, producer_call.result)
    index = _route_index(unit, caller_fn)
    matches = []
    for operand in call_operands(view, unit.unit, consumer_row['query']):
        if operand.role is ArgumentRole.POSITIONAL:
            actual = GlobalPort(unit.unit, view.original(operand.edge.source))
            route = _path(output.port, actual.port, index)
            if route is not None:
                matches.append((operand.ordinal, actual, route))
    argument, actual, route = _one(matches, 'original positional actual carried by the producer result')
    unit.material.validate_parents()
    return argument, producer_row, consumer_row, output, actual, route


def _route_index(unit, caller_fn):
    material = unit.material
    edges = (edge for plan in material.closure(caller_fn.template).values()
             for edge in normal_edges(plan, material.interfaces) if edge.kind is G.EdgeKind.DATA)
    return _index((edge.source, edge.target, encode(edge)) for edge in edges)


def _original_route(unit, caller_fn, output, actual):
    relation = _path(output.port, actual.port, _route_index(unit, caller_fn))
    _require(relation is not None, 'F4 lacks a data-carry path from the producer result to positional actual')
    unit.material.validate_parents()
    return relation


def project(catalog, capture):
    """Read original source/capture receipts and produce scoped F4 data edges."""
    view = ProgramView(catalog)
    catalog_path = view.catalog.path/'result.json'
    catalog_ref = artifact_reference(catalog_path, view.catalog.report['schema'])
    capture_ref = artifact_reference(Path(capture).resolve(strict=True), CAPTURE)
    observed = read_artifact(capture_ref)
    selection = read_artifact(observed['selection'])
    launch = observed['launch']
    _require(observed['schema_version'] == 1 and observed['original_execution'] is True
             and observed['values_saved'] is False
             and observed['reference_test'] == 'LIVE_RETURN_OBJECT_IS_SELECTED_ENTRY_FORMAL'
             and launch['entry_path'] == selection['source']['uri'] and launch['argv'] == selection['argv']
             and launch['decision'] == 'NO_OP' and launch['exit_code'] == 0,
             'handoff requires the original source launch and live reference test')
    read_artifact(selection['source'], json_document=False)
    proofs = RetainedCodeBodyIndex(view)
    caller_body, caller_fact, caller_facts, caller_fn = _body(proofs, selection['caller'])
    argument, producer_site, consumer_site, output, actual, route = _plan(
        view, selection, caller_facts, caller_fn)
    producer_body, producer_fact, producer_facts, producer_fn = _body(proofs, selection['producer']['code'])
    consumer_body, consumer_fact, consumer_facts, consumer_fn = _body(proofs, selection['consumer']['code'])
    _require(producer_fn.returns is not None, 'selected callee has no normal return value')
    returned = source_port(producer_facts, producer_fn.returns)
    _require(returned.port.completion is LanguageCompletionKind.RETURN, 'selected producer is not a normal return')

    run = capture_ref['sha256']
    process, thread = observed['process_id'], observed['thread_id']
    _require(type(process) is int and process > 0 and type(thread) is int and thread > 0
             and 0 < len(observed['matched']) <= selection['max_pairs'],
             'no selected normal original handoff observed')
    identified = set()
    edges = []
    for index, row in enumerate(observed['matched']):
        parent, produced, consumed = (row[key] for key in ('parent', 'producer', 'consumer'))
        ordinal = row['formal_ordinal']
        normal = row['producer_return']
        _require(all(type(value) is int and value > 0 for value in (parent, produced, consumed))
                 and parent < produced < consumed and produced not in identified and consumed not in identified
                 and normal['code'] == selection['producer']['code']['key']
                 and normal['opname'] in ('RETURN_VALUE', 'RETURN_CONST')
                 and type(ordinal) is int and ordinal in (argument, argument+1)
                 and 0 <= ordinal < len(consumer_fn.parameter_names),
                 'normal return and its actual positional formal are not uniquely observed')
        identified.update((produced, consumed))
        formal = source_port(consumer_facts, consumer_fn.formals[1+ordinal])
        scope = lambda local: SourceCallScope(run, InvocationKey(process, thread, local))
        edges.append(dict(kind='DATA', source=encode(SourceInvocationPort(scope(produced), returned)),
                          target=encode(SourceInvocationPort(scope(consumed), formal)),
                          caller=encode(scope(parent)), formal=consumer_fn.parameter_names[ordinal],
                          producer_return=normal,
                          evidence={'artifact': capture_ref, 'locator': f'#/matched/{index}'}))
    proofs.check()
    return dict(schema=SCHEMA, schema_version=1, catalog=catalog_ref, capture=capture_ref,
        scopes=dict(caller=encode(caller_body), producer=encode(producer_body), consumer=encode(consumer_body)),
        function_facts=dict(caller=caller_fact, producer=producer_fact, consumer=consumer_fact),
        original_sites=dict(producer=producer_site, consumer=consumer_site),
        caller_result=encode(output), caller_actual=encode(actual), caller_actual_ordinal=argument,
        f4_data_route=route, observed_edges=edges,
        observation_frontiers=observed['observation_frontiers'],
        scope='SAMPLED_DIRECT_POSITIONAL_NORMAL_HANDOFF',
        qualification=dict(reference_handoff_observed=True, logical_content_version_proven=False,
            state_effects_closed=False, future_dispatch_proven=False, transformation_authorized=False,
            whole_entry_complete=False, target_executed_by_projection=False))


def check(path):
    report = read_artifact(artifact_reference(path, SCHEMA))
    read_artifact(report['catalog'])
    rebuilt = project(Path(report['catalog']['uri']).parent, report['capture']['uri'])
    _require(report == rebuilt, 'stored handoff differs from its original replay')
    return rebuilt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path)
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args()
    if args.check:
        result = check(args.check)
    else:
        if not all((args.catalog, args.capture, args.out)):
            parser.error('provide --catalog, --capture and --out, or --check')
        result = project(args.catalog, args.capture)
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(dict(handoffs=len(result['observed_edges']), f4_edges=len(result['f4_data_route']))))


# Independent source/compiler/CFG checks for the opt-in kwargs call protocol.
#
# These queries inspect existing ports/holds. No second execution or lifetime
# IR is published. Local pending suffix checks do not establish the enclosing
# frame prefix, runtime code identity, actual mapping items or full call cleanup.
EXPANSION_PROTOCOLS = frozenset((Kind.BUILD_CALL_POSITIONAL_LIST, Kind.EXTEND_CALL_POSITIONALS,
    Kind.APPEND_CALL_POSITIONAL, Kind.CALL_POSITIONALS_TO_TUPLE))
CALL_PROTOCOLS = frozenset((Kind.BUILD_CALL_POSITIONALS, Kind.DICT_MERGE, Kind.CALL_FUNCTION_EX)) | EXPANSION_PROTOCOLS


def is_original_call_definition(semantic, definition, source):
    """Recognize frontend_v2's unchanged opaque Call, never invent a CALL kind."""
    atom = semantic.source_atoms.get(source.atom_id)
    if atom is None or atom.kind != "Call" or atom.reference != source:
        return False
    matches = [row for row in semantic.definitions.values()
               if row.kind is OperationKind.OPAQUE and row.metadata.get("ast_type") == "Call"
               and row.code_id == source.atom_id.wire and source.atom_id in row.source_atoms
               and len(row.output_slots) == 1]
    return len(matches) == 1 and definition is matches[0]


def _need(condition, detail):
    if not condition:
        raise ValueError("kwargs call validation: " + detail)


def _call_reference_span(reference):
    return reference.start_line, reference.end_line, reference.start_column, reference.end_column


def _ast_span(node):
    return node.lineno, node.end_lineno, node.col_offset, node.end_col_offset


def _expected(node, *, expanded=False):
    """Independent AST staging: packing, complete named runs, then merges."""
    starred = len(node.args) == 1 and isinstance(node.args[0], ast.Starred)
    stars = [i for i, arg in enumerate(node.args) if isinstance(arg, ast.Starred)]
    if expanded and stars and not starred:
        result = [('BUILD_LIST', stars[0], None)]
        for index in range(stars[0], len(node.args)):
            result.append(('LIST_EXTEND' if isinstance(node.args[index], ast.Starred) else 'LIST_APPEND', 1, index))
        result.append(('TO_TUPLE', None, None))
    else:
        result = [] if starred else [("PACK", len(node.args), None)]
    receiver, index = False, 0
    while index < len(node.keywords):
        child = node.keywords[index]
        if child.arg is None:
            if not receiver:
                result.append(("BUILD_MAP", 0, None)); receiver = True
            result.append(("DICT_MERGE", 1, index)); index += 1
        else:
            end = index
            while end < len(node.keywords) and node.keywords[end].arg is not None:
                result.append(("KEY", node.keywords[end].arg, end)); end += 1
            result.append(("BUILD_MAP", end - index, index))
            if receiver: result.append(("DICT_MERGE", 1, index))
            receiver, index = True, end
    result.append(("CALL_FUNCTION_EX", None, None))
    return result


def _pop_suffix(stack, consumed):
    """Ordered hold instances, retaining multiplicity even for equal ValueIDs."""
    if not consumed: return tuple(stack)
    _need(len(stack) >= len(consumed) and tuple(row[1] for row in stack[-len(consumed):]) == tuple(consumed),
          "primitive consumed a different ordered held suffix")
    return tuple(stack[:-len(consumed)])


def _check_failure_pending(actual, local, prefix=None, *, exact=True):
    expected = tuple(row[1] for row in local)
    if prefix is None:
        _need(len(actual) >= len(expected) and (not expected or tuple(actual[-len(expected):]) == expected),
              "failure lost or reordered its call-owned pending suffix")
        return tuple(actual[:-len(expected)]) if expected else tuple(actual)
    _need(tuple(actual) == prefix + expected if exact else tuple(actual[:len(prefix) + len(expected)]) == prefix + expected,
          "failure changed its enclosing prefix or call-owned pending references")
    return prefix


def _local_pending(group, all_operations, protocols, use_links, blocks, values, links, language=None):
    final = group[-1]
    call = all_operations[final]
    callee = call.uses[0].value
    candidates = []
    ordered = list(all_operations)
    first, end = ordered.index(group[0]), ordered.index(final)
    for identity in ordered[:first]:
        operation = all_operations[identity]
        if protocols.get(identity) is None or protocols[identity].kind is not Kind.FRAME_HOLD_TEMPORARY:
            continue
        refs = operation.uses[:-1]
        if len(refs) == 1:
            source = use_links.get((identity, refs[0].id))
            if (refs[0].value == callee and source is not None and source.role == "call-held-callee"
                    and source.source == use_links[(final, call.uses[0].id)].source):
                candidates.append(identity)
    _need(len(candidates) == 1, "call has no unique original callee hold instance")
    begin = ordered.index(candidates[0])
    owned_values = {callee}
    for identity in group:
        operation = all_operations[identity]
        owned_values.update(use.value for use in operation.uses[:-1])
        protocol = protocols[identity].kind
        if protocol is Kind.READ_CODE_CONSTANT:
            owned_values.update(operation.results)
        elif operation.terminator is not None:
            normal = blocks[operation.terminator.successors[0].block]
            owned_values.update(normal.arguments[:-1])
    stack, prefix, deferred, checked = (), None, [], 0
    group_set = set(group)
    for identity in ordered[begin:end + 1]:
        operation = all_operations[identity]
        protocol = protocols.get(identity)
        kind = protocol.kind if protocol is not None else None
        if kind is Kind.FRAME_HOLD_TEMPORARY:
            refs = operation.uses[:-1]
            if len(refs) == 1:
                source = use_links.get((identity, refs[0].id))
                if source is not None and source.role.startswith("call-held-") and refs[0].value in owned_values:
                    stack += ((identity, refs[0].value),)
        if operation.terminator is None or operation.terminator.kind is not TerminatorKind.INVOKE:
            continue
        refs = tuple(use.value for use in operation.uses[:-1])
        normal_stack, failure_stack = stack, stack
        if identity in group_set:
            if kind in (Kind.BUILD_CALL_POSITIONALS, Kind.BUILD_CALL_POSITIONAL_LIST):
                normal_stack = _pop_suffix(stack, refs)  # failure is success-only steal.
            elif kind in (Kind.EXTEND_CALL_POSITIONALS, Kind.APPEND_CALL_POSITIONAL):
                normal_stack = failure_stack = _pop_suffix(stack, refs[-1:])
            elif kind is Kind.CALL_POSITIONALS_TO_TUPLE:
                normal_stack = failure_stack = _pop_suffix(stack, refs)
            elif kind is Kind.BUILD_MAP:
                normal_stack = failure_stack = _pop_suffix(stack, refs)
            elif kind is Kind.DICT_MERGE:
                normal_stack = failure_stack = _pop_suffix(stack, refs[-1:])
            elif kind is Kind.CALL_FUNCTION_EX:
                normal_stack = failure_stack = _pop_suffix(stack, refs)
        failure = blocks[operation.terminator.successors[1].block]
        _need(bool(failure.operations), "call/child failure has no cleanup")
        paired = language is not None and Capability.FUNCTION_TRY_314_V1 in language.capabilities
        if paired:
            from scar.analysis.static.lowering.source_function_try import exception_handoff_holds
            actual = exception_handoff_holds(failure.id, all_operations, protocols, blocks, use_links)
        else:
            cleanup = all_operations[failure.operations[0]]
            _need(protocols.get(cleanup.id) is not None and protocols[cleanup.id].kind is Kind.FRAME_EXIT,
                  "call/child failure does not route its post-world through FRAME_EXIT")
            actual = tuple(use.value for use in cleanup.uses
                if (link := use_links.get((cleanup.id, use.id))) is not None and link.role == "held-temporary")
        if identity in group_set:
            prefix = _check_failure_pending(actual, failure_stack, prefix)
        else:
            deferred.append((actual, failure_stack))
        stack = normal_stack; checked += 1
    _need(not stack and prefix is not None, "call-owned holds were leaked or never reviewed")
    for actual, local in deferred:
        _check_failure_pending(actual, local, prefix, exact=False)
    return {"call_owned_pending_checked": True, "invoke_failures_checked": checked,
            "enclosing_prefix_consistency_checked": True, "outer_pending_verified": False,
            "call_cleanup_contract_verified": False, "whole_lifetime_verified": False}


def validate_call_protocols(model, graph, interface, metadata, semantic, language, *,
                            source_tree=None, options=None, structural_only=False):
    protocols = {row.operation: row for row in model.protocols}
    new = {identity for identity, row in protocols.items() if row.kind in CALL_PROTOCOLS}
    auxiliary = {identity for identity, row in protocols.items()
        if row.kind in (Kind.BUILD_MAP, Kind.MAP_ADD, Kind.DICT_UPDATE, Kind.READ_CODE_CONSTANT)
        and (atom := semantic.source_atoms.get(row.source.atom_id)) is not None and atom.kind == "Call"}
    if not new:
        _need(not auxiliary, "orphan call packing/constant has no final kwargs call")
        return {"calls": 0, "records": [], "whole_lifetime_verified": False}
    _need(Capability.CALL_KWARGS_314_V1 in language.capabilities, "new protocol has no explicit kwargs capability")
    expanded = Capability.CALL_EXPANSION_314_V1 in language.capabilities
    _need(expanded or not any(protocols[i].kind in EXPANSION_PROTOCOLS for i in new),
          'list-based argument expansion has no explicit capability')
    if not structural_only:
        _need(isinstance(source_tree, ast.Module) and type(options) is SourceLanguageOptions
              and options.code_evidence is not None, "exact kwargs checking requires independent source/code inputs")
        evidence = options.code_evidence
        evidence.require_definition(metadata.source, interface.definition, model.source)
        _need(supports_cpython_compiler(evidence), "kwargs contract requires pinned CPython 3.14.7 compiler layout")
    operations = {row.id: row for row in graph.operations}
    values = {row.id: row for row in graph.values}
    blocks = {block.id: block for region in graph.regions for block in region.blocks}
    positions = {identity: (block.id, index) for block in blocks.values() for index, identity in enumerate(block.operations)}
    links = {row.operation: row for row in metadata.operation_links}
    uses = {(row.operation, row.use): row for row in metadata.use_links}
    constants = {row.operation: row for row in language.constants}
    ref, world = ValueType(ValueKind.REFERENCE), value_type(values[interface.world])
    finals = [identity for identity, row in protocols.items() if row.kind is Kind.CALL_FUNCTION_EX]
    _need(len({protocols[identity].source for identity in finals}) == len(finals), "Call has duplicate final call protocol")
    covered, records = set(), []
    for final in finals:
        source = protocols[final].source
        atom = semantic.source_atoms.get(source.atom_id)
        definition = semantic.definitions.get(links[final].semantic_operation)
        _need(atom is not None and atom.kind == "Call" and atom.reference == source
              and definition is not None and is_original_call_definition(semantic, definition, source)
              and definition.code_id == source.atom_id.wire and source.atom_id in definition.source_atoms
              and len(definition.output_slots) == 1 and links[final].opcode is Opcode.CALL,
              "final kwargs call has no unique original SG Call/output source")
        group = [identity for identity in operations if identity in protocols and protocols[identity].source == source
                 and protocols[identity].kind in CALL_PROTOCOLS | {Kind.BUILD_MAP, Kind.MAP_ADD, Kind.DICT_UPDATE, Kind.READ_CODE_CONSTANT}]
        _need(group and group[-1] == final, "call preparation is missing or appears after final call")
        _need(all(protocols[i].kind not in (Kind.MAP_ADD, Kind.DICT_UPDATE) for i in group),
              "kwargs preparation cannot use DICT_UPDATE or an unsupported large keyword run")
        covered.update(set(group) & new)
        node = None
        if not structural_only:
            selected = [n for n in require_source_tree(source_tree, language.source, source) if isinstance(n, ast.Call)]
            _need(len(selected) == 1, "kwargs call has no unique exact source AST")
            node = selected[0]
            direct_star = len(node.args) == 1 and isinstance(node.args[0], ast.Starred)
            _need(expanded and any(isinstance(a, ast.Starred) for a in node.args)
                  or direct_star or any(k.arg is None for k in node.keywords)
                  and not any(isinstance(a, ast.Starred) for a in node.args),
                  "call profile requires fixed positionals/** or one direct starred iterable")
        built_maps, packed, lists, stage, receiver = {}, {}, {}, [], None
        final_refs = tuple(use.value for use in operations[final].uses[:-1])
        final_sources = [uses[(final, u.id)] for u in operations[final].uses[:-1]]
        direct_star = len(final_sources) >= 2 and final_sources[1].role == 'call-starred-positionals'
        _need(len(final_refs) == 3 or (direct_star or expanded) and len(final_refs) == 2,
              "final call needs callee/positionals and optional original kwargs dict")
        if direct_star:
            _need((Capability.FUNCTION_TRY_314_V1 in language.capabilities or expanded) and isinstance(source_tree, ast.Module),
                  'direct starred call requires the explicit function capability and retained AST')
            candidates = [n for n in require_source_tree(source_tree, language.source, source) if isinstance(n, ast.Call)]
            _need(len(candidates) == 1 and len(candidates[0].args) == 1
                  and isinstance(candidates[0].args[0], ast.Starred)
                  and _call_reference_span(final_sources[1].source) == _ast_span(candidates[0].args[0].value),
                  'direct starred call does not consume its unique original iterable expression')
            packed[final_refs[1]] = None
        callee = final_refs[0]
        for identity in group:
            operation, kind = operations[identity], protocols[identity].kind
            _need(links[identity].source == source and protocols[identity].source == source,
                  "call protocol lost its exact original source")
            if kind is Kind.READ_CODE_CONSTANT:
                _need(identity in constants and links[identity].semantic_operation is None,
                      "call constant lacks exact code provenance")
                value = operation.results[0]
                if value == final_refs[1]: packed[value] = identity
                stage.append(("CONST", identity)); continue
            _need(operation.callee is None and not operation.effects_complete
                  and operation.terminator is not None and operation.terminator.kind is TerminatorKind.INVOKE
                  and tuple(use.id for use in operation.uses) == operation.invoke_arguments
                  and all(type(values[use.value]) is ReferenceValue for use in operation.uses[:-1])
                  and type(values[operation.uses[-1].value]) is StateVersionValue
                  and values[operation.uses[-1].value].location == values[interface.world].location
                  and all(b in operation.behaviors for b in (Behavior.CONTROL, Behavior.READ, Behavior.WRITE, Behavior.MEMORY)),
                  "call protocol lost ordered references/incomplete world invoke")
            count = len(operation.uses) - 1
            result_count = 1 if kind in (Kind.BUILD_MAP, Kind.BUILD_CALL_POSITIONALS, Kind.CALL_FUNCTION_EX,
                                       Kind.BUILD_CALL_POSITIONAL_LIST, Kind.CALL_POSITIONALS_TO_TUPLE) else 0
            _need(tuple((c.kind, c.types) for c in operation.completions) ==
                  ((ExitKind.NORMAL, (ref,) * result_count + (world,)), (ExitKind.RAISE, function_exception_types(language, ref, world)))
                  and tuple(e.kind for e in operation.terminator.successors) == (ExitKind.NORMAL, ExitKind.RAISE)
                  and all(e.arguments == tuple(CompletionResult(i) for i in range(len(c.types)))
                          for e, c in zip(operation.terminator.successors, operation.completions)),
                  "call normal/raise payloads and post-world differ")
            refs = tuple(use.value for use in operation.uses[:-1])
            source_uses = [uses[(identity, use.id)] for use in operation.uses[:-1]]
            for source_use in source_uses:
                atom = semantic.source_atoms.get(source_use.source.atom_id)
                _need((source_use.source.path, source_use.source.fingerprint) == (source.path, source.fingerprint)
                      and atom is not None and atom.reference == source_use.source,
                      "call operand differs from its exact source namespace/atom")
            normal = blocks[operation.terminator.successors[0].block]
            if kind is Kind.BUILD_MAP:
                _need(links[identity].semantic_operation is None and count <= 30 and count % 2 == 0,
                      "keyword run is not bounded synthetic BUILD_MAP")
                for index in range(0, count, 2):
                    key, value = source_uses[index:index + 2]
                    _need(key.role == "call-keyword-name" and value.role.startswith("keyword:")
                          and key.index == value.index and key.source == source,
                          "keyword run lost original ordered name/value Uses")
                built_maps[normal.arguments[0]] = identity
                if receiver is None:
                    receiver = normal.arguments[0]
            elif kind is Kind.BUILD_CALL_POSITIONALS:
                _need(links[identity].semantic_operation is None
                      and [(u.role, u.index) for u in source_uses] == [("argument", i) for i in range(count)],
                      "positional tuple lost ordered original argument Uses")
                packed[normal.arguments[0]] = identity
            elif kind is Kind.BUILD_CALL_POSITIONAL_LIST:
                _need(links[identity].semantic_operation is None
                      and [(u.role, u.index) for u in source_uses] == [('argument', i) for i in range(count)],
                      'call positional list lost its original ordered prefix')
                lists[normal.arguments[0]] = identity
            elif kind in (Kind.EXTEND_CALL_POSITIONALS, Kind.APPEND_CALL_POSITIONAL):
                _need(count == 2 and refs[0] in lists and source_uses[0].role == 'call-positional-list'
                      and source_uses[0].index == 0 and source_uses[0].source == source
                      and source_uses[1].role == ('starred-argument' if kind is Kind.EXTEND_CALL_POSITIONALS else 'argument'),
                      'call list update lost its same held list or original argument')
            elif kind is Kind.CALL_POSITIONALS_TO_TUPLE:
                _need(count == 1 and refs[0] in lists and source_uses[0].role == 'call-positional-list'
                      and source_uses[0].index == 0 and source_uses[0].source == source,
                      'call positional tuple conversion did not consume its own original list')
                packed[normal.arguments[0]] = identity
            elif kind is Kind.DICT_MERGE:
                _need(links[identity].semantic_operation is None and count == 3 and refs[0] == callee
                      and refs[1] == receiver and refs[2] != refs[1]
                      and [(u.role, u.index) for u in source_uses[:2]] == [("callee", 0), ("call-keywords", 0)]
                      and source_uses[2].role in ("keyword_unpack", "call-keyword-run"),
                      "DICT_MERGE lost callee/receiver/update or used an invented dict")
                _need(source_uses[2].role != "call-keyword-run" or refs[2] in built_maps,
                      "keyword run merge update has no original call BUILD_MAP")
            else:
                roles = [("callee", 0), ("call-starred-positionals" if direct_star else "call-positionals", 0)]
                if receiver is not None:
                    roles += [("call-keywords", 0)]
                _need(count == len(roles) and refs[1] in packed
                      and (receiver is None or refs[2] == receiver)
                      and [(u.role, u.index) for u in source_uses] == roles,
                      "final call lost its original callee or prepared argument containers")
            stage.append((kind.value, identity))
        _need(len(packed) == 1, "call positional tuple preparation is not unique")
        # Graph ordering alone is insufficient: each next stage must be the
        # first call-owned stage on every normal child/control continuation.
        own = set(group)
        for current, expected_next in zip(group, group[1:]):
            operation = operations[current]
            if operation.terminator is None:
                block, index = positions[current]; pending = [(block, index + 1)]
            else:
                pending = [(operation.terminator.successors[0].block, 0)]
            seen, found = set(), set()
            while pending:
                position = pending.pop()
                if position in seen: continue
                seen.add(position)
                _need(len(seen) <= len(operations) * 2 + 1, "call normal staging traversal budget exceeded")
                block, index = blocks[position[0]], position[1]
                _need(index < len(block.operations), "call normal continuation falls out of block")
                identity = block.operations[index]
                if identity in own:
                    found.add(identity); continue
                child = operations[identity]
                if child.terminator is None:
                    pending.append((block.id, index + 1))
                elif child.terminator.kind is TerminatorKind.INVOKE:
                    pending.extend((e.block, 0) for e in child.terminator.successors if e.kind is ExitKind.NORMAL)
                elif child.terminator.kind in (TerminatorKind.JUMP, TerminatorKind.BRANCH):
                    pending.extend((e.block, 0) for e in child.terminator.successors)
                else:
                    raise ValueError("kwargs call validation: child normal path skips later call stage")
            _need(found == {expected_next}, "ordered call preparation stages can be reordered or skipped")
        if node is not None:
            _check_source_and_code(node, group, operations, protocols, uses, constants, values, options.code_evidence,
                                   expanded=expanded)
        _need(any(g.kind is GapKind.FRAME_EFFECT and g.source == source
                  and g.detail.startswith("CALL_EX_LOCAL_UNWIND_CONTRACT_UNVERIFIED:") for g in metadata.gaps),
              "call must retain its explicit unverified runtime cleanup frontier")
        pending = _local_pending(group, operations, protocols, uses, blocks, values, links, language)
        records.append({"operation": final.wire, "source": source.atom_id.wire,
            "protocol_operations": len(group), "source_code_verified": not structural_only,
            "pending": pending, "frontiers": ["CALL_EX_LOCAL_UNWIND_CONTRACT_UNVERIFIED", "OUTER_PENDING_PREFIX_UNVERIFIED"]})
    _need(covered == new and auxiliary <= {identity for final in finals for identity in operations
          if identity in protocols and protocols[identity].source == protocols[final].source},
          "orphan kwargs preparation has no exact final Call")
    return {"calls": len(records), "records": records, "runtime_code_instance_established": False,
            "effects_established": False, "whole_lifetime_verified": False}


def _check_source_and_code(node, group, operations, protocols, uses, constants, values, evidence, *, expanded=False):
    expected = _expected(node, expanded=expanded)
    actual = []
    for identity in group:
        operation, kind = operations[identity], protocols[identity].kind
        if kind is Kind.READ_CODE_CONSTANT:
            source = constants[identity]
            _need(source.code_key == evidence.code_layout_key and len(source.constant_path) == 1,
                  "call constant uses a different compiler witness")
            value = evidence.code.co_consts[source.constant_path[0]]
            actual.append(("PACK" if type(value) is tuple and not value else "KEY", value, identity))
        elif kind is Kind.BUILD_CALL_POSITIONALS:
            actual.append(("PACK", len(operation.uses) - 1, identity))
        elif kind is Kind.BUILD_CALL_POSITIONAL_LIST:
            actual.append(('BUILD_LIST', len(operation.uses)-1, identity))
        elif kind in (Kind.EXTEND_CALL_POSITIONALS, Kind.APPEND_CALL_POSITIONAL):
            actual.append(('LIST_EXTEND' if kind is Kind.EXTEND_CALL_POSITIONALS else 'LIST_APPEND', 1, identity))
        elif kind is Kind.CALL_POSITIONALS_TO_TUPLE:
            actual.append(('TO_TUPLE', None, identity))
        else:
            actual.append((kind.value, (len(operation.uses) - 1) // 2 if kind is Kind.BUILD_MAP else 1 if kind is Kind.DICT_MERGE else None, identity))
    _need(len(actual) == len(expected), "call stage count differs from original keyword runs")
    for (kind, value, identity), (expected_kind, argument, index) in zip(actual, expected):
        _need(kind == expected_kind and (value == argument or kind == "PACK" and argument == 0 and value == ()),
              "call stage/name/run order differs from original AST")
        operation = operations[identity]
        source_uses = [uses[(identity, use.id)] for use in operation.uses[:-1]]
        if kind == "PACK" and protocols[identity].kind is Kind.BUILD_CALL_POSITIONALS:
            _need(tuple(_call_reference_span(u.source) for u in source_uses) == tuple(_ast_span(n) for n in node.args),
                  "positional tuple children differ from exact AST order")
        elif kind == 'BUILD_LIST':
            _need(tuple(_call_reference_span(u.source) for u in source_uses) == tuple(_ast_span(n) for n in node.args[:argument]),
                  'call list prefix differs from the original AST before its first star')
        elif kind in ('LIST_EXTEND', 'LIST_APPEND'):
            original = node.args[index]
            _need(isinstance(original, ast.Starred) == (kind == 'LIST_EXTEND')
                  and source_uses[1].index == index
                  and _call_reference_span(source_uses[1].source) == _ast_span(original.value if isinstance(original, ast.Starred) else original),
                  'call list update lost its exact original argument position/source')
        elif kind == "BUILD_MAP" and argument:
            for pair, keyword_index in enumerate(range(index, index + argument)):
                _need(source_uses[pair * 2].index == keyword_index
                      and source_uses[pair * 2 + 1].role == "keyword:" + node.keywords[keyword_index].arg
                      and _call_reference_span(source_uses[pair * 2 + 1].source) == _ast_span(node.keywords[keyword_index].value),
                      "named run child source/index differs")
        elif kind == "DICT_MERGE":
            update = source_uses[2]
            unpack = node.keywords[index].arg is None
            _need(update.index == index and update.role == ("keyword_unpack" if unpack else "call-keyword-run")
                  and (not unpack or _call_reference_span(update.source) == _ast_span(node.keywords[index].value)),
                  "merge source/update index differs from exact keyword sequence")
        elif kind == "CALL_FUNCTION_EX":
            _need(_call_reference_span(source_uses[0].source) == _ast_span(node.func), "callee source is not the original single evaluation")
            starred = len(node.args) == 1 and isinstance(node.args[0], ast.Starred)
            if starred:
                _need(source_uses[1].role == 'call-starred-positionals'
                      and _call_reference_span(source_uses[1].source) == _ast_span(node.args[0].value),
                      'direct starred iterable lost its original single evaluation')
    names = {"LOAD_CONST", "BUILD_TUPLE", "BUILD_LIST", "LIST_APPEND", "LIST_EXTEND", "LIST_TO_TUPLE",
             "BUILD_MAP", "MAP_ADD", "DICT_MERGE", "DICT_UPDATE", "CALL_FUNCTION_EX", "CALL_KW", "CALL"}
    own = [i for i in dis.get_instructions(evidence.code) if tuple(i.positions) == _ast_span(node)]
    if expanded:
        calls = [i for i in own if i.opname == 'CALL_FUNCTION_EX']
        _need(len(calls) == 1, 'expanded call has no unique exact compiler call occurrence')
        names.add('CALL_INTRINSIC_1')
        own = [i for i in own if i.offset <= calls[0].offset]
    instructions = [i for i in own if i.opname in names]
    _need(len(instructions) == len(expected), "compiler uses a different packing/merge call sequence")
    for instruction, (kind, argument, _index), (_actual, _value, identity) in zip(instructions, expected, actual):
        if kind in ("PACK", "KEY") and protocols[identity].kind is Kind.READ_CODE_CONSTANT:
            _need(instruction.opname == "LOAD_CONST" and instruction.arg == constants[identity].constant_path[0],
                  "call code constant does not identify exact compiler instruction/slot")
        elif kind == "PACK":
            _need(instruction.opname == "BUILD_TUPLE" and instruction.arg == argument, "call positional packing differs from exact compiler")
        elif kind == 'TO_TUPLE':
            _need(instruction.opname == 'CALL_INTRINSIC_1'
                  and dis._intrinsic_1_descs[instruction.arg] == 'INTRINSIC_LIST_TO_TUPLE',
                  'call list conversion is not the exact pinned list-to-tuple intrinsic')
        else:
            _need(instruction.opname == kind and (argument is None or instruction.arg == argument),
                  "call merge/named-run differs from exact compiler")


# Opt-in CPython 3.14.7 kwargs preparation over original source/core ports.
#
# DICT_MERGE is deliberately distinct from dict-display DICT_UPDATE. Mapping
# keys/items, collision/error formatting and partial mutation stay inside its
# incomplete world invoke; no dynamic entries or internal calls are invented.
# CALL_FUNCTION_EX uses an explicit call-plus-local-operand-unwind boundary.
# Its physical cleanup/monitoring/finalizer qualification remains a frontier.
def _span(node):
    return node.lineno, node.end_lineno, node.col_offset, node.end_col_offset


def _keyword_runs(node):
    start = 0
    while start < len(node.keywords):
        end = start + 1
        if node.keywords[start].arg is not None:
            while end < len(node.keywords) and node.keywords[end].arg is not None:
                end += 1
        yield start, end
        start = end


def _witness_plan(node, instructions):
    """Bounded compiler recognition only, not target evaluation or wire data."""
    starred = len(node.args) == 1 and isinstance(node.args[0], ast.Starred)
    if (not isinstance(node, ast.Call) or not (starred or any(k.arg is None for k in node.keywords))
            or any(isinstance(a, ast.Starred) for a in node.args) and not starred):
        raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX, node, "KWARGS_CALL_REQUIRES_FIXED_POSITIONALS_AND_DOUBLE_STAR")
    # One *expression is passed directly by this pinned compiler. Conversion
    # to positional tuple (if needed) belongs to CALL_FUNCTION_EX, with its
    # own failure/current-world boundary; no tuple object is invented here.
    expected = [] if starred else [("PACK", len(node.args), None)]
    have_dict = False
    for start, end in _keyword_runs(node):
        if node.keywords[start].arg is None:
            if not have_dict:
                expected.append(("BUILD_MAP", 0, None)); have_dict = True
            expected.append(("DICT_MERGE", 1, start))
        else:
            if (end - start) * 2 > 30:
                raise _Unsupported(GapKind.CODE_VERSION, node, "KWARGS_LARGE_KEYWORD_RUN_COMPILER_FRONTIER")
            for index in range(start, end):
                expected.append(("KEY", node.keywords[index].arg, index))
            expected.append(("BUILD_MAP", end - start, None))
            if have_dict:
                expected.append(("DICT_MERGE", 1, start))
            have_dict = True
    expected.append(("CALL_FUNCTION_EX", None, None))
    names = {"LOAD_CONST", "BUILD_TUPLE", "BUILD_LIST", "LIST_APPEND", "LIST_EXTEND", "LIST_TO_TUPLE",
             "BUILD_MAP", "MAP_ADD", "DICT_MERGE", "DICT_UPDATE", "CALL_FUNCTION_EX", "CALL_KW", "CALL"}
    actual = [row for row in instructions if tuple(row.positions) == _span(node) and row.opname in names]
    if len(actual) != len(expected):
        raise _Unsupported(GapKind.CODE_VERSION, node, "KWARGS_EXACT_COMPILER_SEQUENCE_UNAVAILABLE")
    keys, packing = {}, None if starred else actual[0]
    for row, (name, argument, index) in zip(actual, expected):
        if name == "PACK":
            ok = ((row.opname == "BUILD_TUPLE" and row.arg == argument)
                  or (argument == 0 and row.opname == "LOAD_CONST" and type(row.argval) is tuple and not row.argval))
        elif name == "KEY":
            ok = row.opname == "LOAD_CONST" and type(row.argval) is str and row.argval == argument
            if ok: keys[index] = row
        else:
            ok = row.opname == name and (argument is None or row.arg == argument)
        if not ok:
            raise _Unsupported(GapKind.CODE_VERSION, node, "KWARGS_EXACT_COMPILER_SEQUENCE_UNAVAILABLE")
    return packing, keys


def _instructions(builder, node):
    evidence = builder.language_options.code_evidence
    if not supports_cpython_compiler(evidence):
        raise _Unsupported(GapKind.CODE_VERSION, node, "KWARGS_CPYTHON_3147_CODE_WITNESS_REQUIRED")
    evidence.require_definition(builder.snapshot, builder.function_definition.id, builder.source(builder.functions[builder.entry]))
    return tuple(dis.get_instructions(evidence.code))


def _expansion_witness_plan(node, instructions):
    """Recognize the complete original Call window, ending at its own call.

    CPython may give the following implicit return the same source position.
    It is not an argument-packing stage. Duplicated call sites remain a gap.
    """
    direct = len(node.args) == 1 and isinstance(node.args[0], ast.Starred)
    stars = [i for i, arg in enumerate(node.args) if isinstance(arg, ast.Starred)]
    expected = []
    if stars and not direct:
        expected.append(('BUILD_LIST', stars[0], None))
        for index in range(stars[0], len(node.args)):
            expected.append(('LIST_EXTEND' if isinstance(node.args[index], ast.Starred) else 'LIST_APPEND', 1, index))
        expected.append(('CALL_INTRINSIC_1', 'INTRINSIC_LIST_TO_TUPLE', None))
    elif not direct:
        expected.append(('PACK', len(node.args), None))
    have_dict = False
    for start, end in _keyword_runs(node):
        if node.keywords[start].arg is None:
            if not have_dict:
                expected.append(('BUILD_MAP', 0, None)); have_dict = True
            expected.append(('DICT_MERGE', 1, start))
        else:
            if (end-start)*2 > 30:
                raise _Unsupported(GapKind.CODE_VERSION, node, 'KWARGS_LARGE_KEYWORD_RUN_COMPILER_FRONTIER')
            expected.extend(('KEY', node.keywords[i].arg, i) for i in range(start, end))
            expected.append(('BUILD_MAP', end-start, None))
            if have_dict:
                expected.append(('DICT_MERGE', 1, start))
            have_dict = True
    expected.append(('CALL_FUNCTION_EX', None, None))
    own = [i for i in instructions if tuple(i.positions) == _span(node)]
    calls = [i for i in own if i.opname == 'CALL_FUNCTION_EX']
    if len(calls) != 1:
        raise _Unsupported(GapKind.CODE_VERSION, node, 'CALL_EXPANSION_EXACT_CALL_OCCURRENCE_UNAVAILABLE')
    names = {'LOAD_CONST', 'BUILD_TUPLE', 'BUILD_LIST', 'LIST_APPEND', 'LIST_EXTEND',
             'CALL_INTRINSIC_1', 'BUILD_MAP', 'MAP_ADD', 'DICT_MERGE', 'DICT_UPDATE',
             'CALL_FUNCTION_EX', 'CALL_KW', 'CALL'}
    actual = [i for i in own if i.offset <= calls[0].offset and i.opname in names]
    if len(actual) != len(expected):
        raise _Unsupported(GapKind.CODE_VERSION, node, 'CALL_EXPANSION_EXACT_COMPILER_SEQUENCE_UNAVAILABLE')
    packing, keys = None, {}
    for row, (kind, arg, index) in zip(actual, expected):
        if kind == 'PACK':
            ok = row.opname == 'BUILD_TUPLE' and row.arg == arg or arg == 0 and row.opname == 'LOAD_CONST' and row.argval == ()
            packing = row
        elif kind == 'KEY':
            ok = row.opname == 'LOAD_CONST' and type(row.argval) is str and row.argval == arg
            keys[index] = row
        elif kind == 'CALL_INTRINSIC_1':
            ok = row.opname == kind and dis._intrinsic_1_descs[row.arg] == arg
        else:
            ok = row.opname == kind and (arg is None or row.arg == arg)
        if not ok:
            raise _Unsupported(GapKind.CODE_VERSION, node, 'CALL_EXPANSION_EXACT_COMPILER_SEQUENCE_UNAVAILABLE')
    return packing, keys


def _expanded_positionals(builder, node, cursor):
    """Build only real list/extend/append/tuple stages, keeping every hold."""
    first = next(i for i, arg in enumerate(node.args) if isinstance(arg, ast.Starred))
    held, inputs = [], []
    for index, child in enumerate(node.args[:first]):
        cursor, value = builder.expr(child, cursor)
        if cursor is None: return None, None
        _hold(builder, cursor, child, value, 'positional')
        held.append((child, value));inputs.append((value, child, 'argument', index, None))
    cursor, receiver, _ = _invoke(builder, cursor, node, tuple(inputs),
        Kind.BUILD_CALL_POSITIONAL_LIST, tuple(held), raise_consumed=(), result=True)
    _hold(builder, cursor, node, receiver, 'positional-list')
    for index in range(first, len(node.args)):
        argument = node.args[index]
        child = argument.value if isinstance(argument, ast.Starred) else argument
        cursor, value = builder.expr(child, cursor)
        if cursor is None: return None, None
        _hold(builder, cursor, child, value, 'starred-update' if isinstance(argument, ast.Starred) else 'positional')
        cursor, _, _ = _invoke(builder, cursor, node,
            ((receiver, node, 'call-positional-list', 0, None),
             (value, child, 'starred-argument' if isinstance(argument, ast.Starred) else 'argument', index, None)),
            Kind.EXTEND_CALL_POSITIONALS if isinstance(argument, ast.Starred) else Kind.APPEND_CALL_POSITIONAL,
            ((child, value),))
    cursor, positional, _ = _invoke(builder, cursor, node,
        ((receiver, node, 'call-positional-list', 0, None),), Kind.CALL_POSITIONALS_TO_TUPLE,
        ((node, receiver),), result=True)
    _hold(builder, cursor, node, positional, 'positionals')
    return cursor, positional


def _hold(builder, cursor, node, reference, role):
    builder.frame_transition(cursor, node, ((reference, node, "call-held-" + role),), Kind.FRAME_HOLD_TEMPORARY)
    builder.active_temporaries.append(reference)


def _constant(builder, node, cursor, instruction):
    evidence = builder.language_options.code_evidence
    if type(instruction.arg) is not int or evidence.code.co_consts[instruction.arg] is not instruction.argval:
        raise _Unsupported(GapKind.CODE_VERSION, node, "KWARGS_CODE_CONSTANT_SLOT_DIFFERS")
    identity = builder.fresh(OperationID, "call-code-constant")
    before = builder.use(identity, cursor.state, node, "code-constant-world", 0)
    value = ReferenceValue(builder.fresh(ValueID, "call-code-constant"), builder.function_context,
                           OperationResult(identity, 0))
    builder.add_value(value); builder.link_value(value, node, SourceRole.EXPRESSION)
    operation = Operation(identity, "call code constant", (Behavior.READ,), (before,), (value.id,),
        (StateAccess(builder.world, before.id, AccessMode.READ),), effects_complete=False)
    builder.emit(cursor, operation, node, SourceRole.EXPRESSION)
    builder.protocol(identity, Kind.READ_CODE_CONSTANT, node)
    builder.language_constants.append(CodeConstantSource(identity, builder.source(node), evidence.code_layout_key,
                                                         (instruction.arg,)))
    return value.id


def _invoke(builder, cursor, node, inputs, kind, normal_consumed=(), *, raise_consumed=None, old=None, result=False):
    if raise_consumed is None:
        raise_consumed = normal_consumed
    def failed(failure, exception):
        saved = tuple(builder.active_temporaries)
        try:
            _transfer(builder, raise_consumed, node)
            builder.finish_exit(failure, node, exception, ExitKind.RAISE)
        finally:
            builder.active_temporaries[:] = saved
    cursor, value, identity = builder.invoke(cursor, node, tuple(inputs), result_type=_REF if result else None,
                                            old=old, failure_handler=failed)
    builder.protocol(identity, kind, node)
    original = builder.operation_records[identity]
    marked = replace(original, behaviors=original.behaviors + (Behavior.MEMORY,))
    builder.operations[builder.operations.index(original)] = marked
    builder.operation_records[identity] = marked
    _transfer(builder, normal_consumed, node)
    return cursor, value, identity


def lower_kwargs_call(builder, node, cursor):
    """Prepare arbitrary mappings, preserving fixed compiler stage boundaries."""
    if not builder.has_language("CALL_KWARGS_314_V1"):
        raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX, node, "KWARGS_CAPABILITY_REQUIRED")
    expanded = builder.has_language('CALL_EXPANSION_314_V1')
    packing, constants = (_expansion_witness_plan if expanded else _witness_plan)(node, _instructions(builder, node))
    starred = len(node.args) == 1 and isinstance(node.args[0], ast.Starred)
    if starred and not (builder.has_language("FUNCTION_TRY_314_V1") or expanded):
        raise _Unsupported(GapKind.UNSUPPORTED_SYNTAX, node,
            "DIRECT_STAR_CALL_REQUIRES_FUNCTION_TRY_CAPABILITY")
    records = builder.old_by_atom.get(builder.source(node).atom_id, ())
    if len(records) > 1 or records and records[0].opcode is not Opcode.CALL:
        raise _Unsupported(GapKind.SOURCE_MAPPING, node, "kwargs Call has ambiguous source computational records")
    old = records[0] if records else None
    definition = _source_definition(builder, node)
    if not is_original_call_definition(builder.semantic, definition, builder.source(node)):
        raise _Unsupported(GapKind.SOURCE_MAPPING, node, "kwargs Call needs its original SG Call definition")
    baseline = len(builder.active_temporaries)
    try:
        cursor, callee = builder.expr(node.func, cursor)
        if cursor is None: return None, None
        _hold(builder, cursor, node.func, callee, "callee")
        if expanded and not starred and any(isinstance(arg, ast.Starred) for arg in node.args):
            cursor, positional = _expanded_positionals(builder, node, cursor)
            if cursor is None: return None, None
        else:
            parts, inputs = [], []
            for index, argument in enumerate(node.args):
                child = argument.value if starred else argument
                cursor, value = builder.expr(child, cursor)
                if cursor is None: return None, None
                _hold(builder, cursor, child, value, "starred-positionals" if starred else "positional")
                parts.append((child, value)); inputs.append((value, child, "argument", index, None))
            if starred:
                positional = parts[0][1]
            elif packing.opname == "LOAD_CONST":
                positional = _constant(builder, node, cursor, packing)
            else:
                cursor, positional, _ = _invoke(builder, cursor, node, inputs, Kind.BUILD_CALL_POSITIONALS,
                    tuple(parts), raise_consumed=(), result=True)
            if not starred:
                _hold(builder, cursor, node, positional, "positionals")
        receiver = None
        for start, end in _keyword_runs(node):
            keyword = node.keywords[start]
            if keyword.arg is None:
                if receiver is None:
                    cursor, receiver, _ = _invoke(builder, cursor, node, (), Kind.BUILD_MAP, result=True)
                    _hold(builder, cursor, node, receiver, "keywords")
                child = keyword.value
                cursor, update = builder.expr(child, cursor)
                if cursor is None: return None, None
                _hold(builder, cursor, child, update, "update")
                update_source, role = child, "keyword_unpack"
            else:
                held, pairs = [], []
                for index in range(start, end):
                    keyword = node.keywords[index]
                    key = _constant(builder, node, cursor, constants[index])
                    _hold(builder, cursor, node, key, "keyword-name")
                    cursor, value = builder.expr(keyword.value, cursor)
                    if cursor is None: return None, None
                    _hold(builder, cursor, keyword.value, value, "keyword-value")
                    pairs.extend(((key, node, "call-keyword-name", index, None),
                                  (value, keyword.value, "keyword:" + keyword.arg, index, None)))
                    held.extend(((node, key), (keyword.value, value)))
                cursor, update, _ = _invoke(builder, cursor, node, pairs, Kind.BUILD_MAP, tuple(held), result=True)
                _hold(builder, cursor, node, update, "keywords" if receiver is None else "keyword-run")
                if receiver is None:
                    receiver = update
                    continue
                update_source, role = node, "call-keyword-run"
            inputs = ((callee, node.func, "callee", 0, None), (receiver, node, "call-keywords", 0, None),
                      (update, update_source, role, start, None))
            cursor, _, _ = _invoke(builder, cursor, node, inputs, Kind.DICT_MERGE, ((update_source, update),))
        positional_source = node.args[0].value if starred else node
        inputs = ((callee, node.func, "callee", 0, None),
                  (positional, positional_source, "call-starred-positionals" if starred else "call-positionals", 0, None))
        if receiver is not None:
            inputs += ((receiver, node, "call-keywords", 0, None),)
        # This source boundary includes cleanup of its own top-of-stack call
        # operands on an early-no-pop failure. It does NOT claim the raw opcode
        # consumed them on every path, nor that finalizer/monitoring effects are
        # established. The untouched enclosing prefix remains in FRAME_EXIT.
        consumed = ((node.func, callee), (positional_source, positional))
        if receiver is not None:
            consumed += ((node, receiver),)
        cursor, result, identity = _invoke(builder, cursor, node, inputs, Kind.CALL_FUNCTION_EX,
                                          consumed, old=old, result=True)
        if old is None:
            _associate(builder, identity, node, definition, Opcode.CALL)
        builder.pending_targets.append((identity, callee, builder.reference_candidates.get(callee), node))
        builder.gap(GapKind.DYNAMIC_TARGET, node, "kwargs source call is not verified runtime dispatch or expanded actual/formal binding")
        builder.gap(GapKind.EFFECT_SUMMARY, node,
            "kwargs keys/items/hash/equality/error formatting, partial dict mutation and callee effects remain unproved")
        builder.gap(GapKind.FRAME_EFFECT, node,
            "CALL_EX_LOCAL_UNWIND_CONTRACT_UNVERIFIED: call boundary includes its own operand unwind; runtime monitoring, finalizers and whole-frame cleanup remain unproved")
        return cursor, result
    finally:
        del builder.active_temporaries[baseline:]


__all__ = [
    'lower_kwargs_call',
    'validate_call_protocols',
    'SCHEMA',
    'project',
    'check',
    'main',
    'EXPANSION_PROTOCOLS',
    'CALL_PROTOCOLS',
    'is_original_call_definition',
]


if __name__ == '__main__':
    main()
