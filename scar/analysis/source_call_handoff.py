"""Join a bounded live handoff to original F4 return and formal ports."""

import argparse
import json
from pathlib import Path

from scar.analysis.entry_call_dependence import _index, _path
from scar.analysis.normal_graph import normal_edges
from scar.analysis.program_view import ProgramView
from scar.analysis.retained_code_bodies import RetainedCodeBodyIndex
from scar.analysis.call_inputs import call_operands
from scar.analysis.source_reference_slice import source_port
from scar.compiler.call_binding import plan_call_bindings
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program import normal_graph as G
from scar.ir.program.call_edges import CapturedCode, InvocationKey
from scar.ir.program.control import LanguageCompletionKind
from scar.ir.program.normal_targets import GlobalPort
from scar.ir.program.binding_ports import ArgumentRole
from scar.ir.program.source_value_routes import SourceCallScope, SourceInvocationPort
from scar.ir.record_codec import decode, encode
from scar.runtime.source_call_handoff_observe import SCHEMA as CAPTURE

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


if __name__ == '__main__':
    main()
