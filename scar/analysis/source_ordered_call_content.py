"""Compose guarded array effects in an original callable's normal order."""

import argparse
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path

from scar.analysis.array_copy_effects import derive_array_copy_effects
from scar.analysis.bounded_entry_calls import SCHEMA as ENTRY_CALLS, _admit_entry_calls
from scar.analysis.build import BuildSession
from scar.analysis.replay_transaction import replay_transaction
from scar.analysis.source_address_paths import AddressResolver
from scar.analysis.source_loop_dependencies import _control_states, loop_dependencies
from scar.analysis.source_reference_slice import ReferenceSlicer, source_port
from scar.analysis.source_value_flow import project_value_flow
from scar.analysis import stateful_storage_accesses as storage
from scar.analysis.strided_content_flow import analyze_loop_content
from scar.compiler.stateful_model import stateful_from_dict
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program import normal_graph as G, normal_links as F, normal_targets as T
from scar.ir.program.control import LanguageCompletionKind
from scar.ir.program.source_address_paths import AddressRootKind, FieldSelector
from scar.ir.record_codec import decode, encode


SCHEMA = 'scar.source-ordered-call-content'


def _only(rows, detail):
    items = tuple(rows)
    if len(items) != 1:
        raise ValueError(detail)
    return items[0]


def _ordinary_order(unit, function, links):
    """Compare selected normal exits in paired source call/return contexts."""
    body = T.Body(unit.unit, function.template, function.body)
    flow = project_value_flow(unit.material, source_port(unit.facts, function.value), body)
    adjacency = defaultdict(list)
    for edge in flow.edges:
        if edge.kind is G.EdgeKind.CONTROL and edge.transition is not G.Transition.BACKEDGE:
            adjacency[edge.source].append(edge)
    start = G.Port(body.template, body.region, G.PortKind.ENTER)
    states = _control_states(((start, ()),), adjacency, set())
    enters = [G.Port(link.result.port.template, link.result.port.owner, G.PortKind.ENTER) for link in links]
    successors = []
    for enter in enters:
        normal_exit = G.Port(enter.template, enter.owner, G.PortKind.EXIT,
                             completion=LanguageCompletionKind.FALLTHROUGH)
        seed = _only((state for state in states if state[0] == normal_exit),
                     'selected call needs one original normal control context')
        ports = {port for port, _ in _control_states((seed,), adjacency, set())}
        successors.append({i for i, candidate in enumerate(enters) if candidate in ports})
    order = sorted(range(len(links)), key=lambda i: -len(successors[i]))
    if any(successors[ordinal] != set(order[i+1:]) for i, ordinal in enumerate(order)):
        raise ValueError('selected original calls do not have one compatible normal order')
    return order


def compose(view, caller_scope, links, region, region_index, bindings):
    """Project selected calls with one stateful body and copy-only siblings.

    This checks source order and argument paths on the selected normal route.
    The caller still must prove dispatch, effect coverage, and slot and
    allocation stability before applying these effects at a loop backedge.
    """
    links = tuple(links)
    source = view.unit(caller_scope.unit)
    function = _only((item for item in source.facts.functions
                      if item.body_scope is not None and item.body_scope.owner == caller_scope),
                     'caller needs one original source function')
    if (type(region_index) is not int or not 0 <= region_index < len(links)
            or len({link.result for link in links}) != len(links)
            or any(link.result.unit != source.unit or link.body is None
                   or link.binding is not T.ArgumentBinding.DIRECT for link in links)):
        raise ValueError('expected distinct selected direct call interfaces in the original caller')
    order = _ordinary_order(source, function, links)
    slicer = ReferenceSlicer(view.catalog)
    resolver = AddressResolver(view.catalog.fact_store, source.facts, caller_scope)
    identity = region.source.origins[0]
    selected = links[region_index]
    if (identity['unit'], identity['occurrence']) != (selected.body.unit, selected.body.template.occurrence):
        raise ValueError('stateful body differs from the selected original call')

    @lru_cache(maxsize=None)
    def bind_path(link, formal):
        route = _only((arg for arg in link.parameters if arg.kind is T.RouteKind.ARGUMENT and arg.formal == formal),
                      'expected one original formal argument route')
        actual, = route.actuals
        _, ordinal = slicer.local(actual)
        return _only(resolver.paths(ordinal), 'argument path is ambiguous')

    def path_key(path):
        if (path.frontier is not None or path.root_kind is not AddressRootKind.FUNCTION_PARAMETER
                or any(type(selector) is not FieldSelector for selector in path.selectors)):
            raise ValueError('requires a conditional direct field path from a caller formal')
        return path.root, tuple(selector.name for selector in path.selectors)

    def callee(link):
        unit = view.unit(link.body.unit)
        ordinal, fn = _only(((i, row) for i, row in enumerate(unit.facts.functions)
            if (row.template, row.body) == (link.body.template, link.body.region)),
            'selected call body lacks one original function')
        return unit, ordinal, fn

    known = {}
    paths = []
    unit, _, fn = callee(selected)
    for item in region.inputs:
        if item.kind != 'array':
            continue
        slot = fn.parameter_names.index(item.path[0])
        path = bind_path(selected, source_port(unit.facts, fn.formals[slot+1]))
        root, fields = path_key(path)
        key = (root, (*fields, *item.path[1:]))
        if key in known:
            raise ValueError('two stateful ports share one declared address path')
        known[key] = bindings[item.id]
        paths.append((region_index, item.id, path))

    result = []
    copy_summaries = {}
    for index in order:
        link = links[index]
        occurrence = link.result.port.owner.wire
        if index == region_index:
            result.extend(storage.project_stateful_accesses(region, bindings, occurrence=occurrence))
            continue
        unit, ordinal, fn = callee(link)
        actual = {}
        for slot, parameter in enumerate(fn.parameter_names):
            path = bind_path(link, source_port(unit.facts, fn.formals[slot+1]))
            actual[slot] = known[path_key(path)]
            paths.append((index, parameter, path))
        shapes = {slot: row.shape for slot, row in actual.items()}
        key = (unit.unit, ordinal, tuple(shapes.items()))
        if key not in copy_summaries:
            copy_summaries[key] = derive_array_copy_effects(view, unit=unit.unit,
                                                           function_ordinal=ordinal, input_shapes=shapes)
        copies = copy_summaries[key]
        result.extend(storage.project_array_copy_accesses(copies, actual, occurrence=occurrence))
    return tuple(order), tuple(paths), tuple(result)


def _enclosing_loop(view, declarations, inner):
    matches = []
    for index, (row, link) in enumerate(declarations):
        if row['callee_scope'] != inner or link.body is None:
            continue
        scope = decode(F.CodeScopeRef, row['caller_scope'])
        source = view.unit(scope.unit)
        function = _only((fn for fn in source.facts.functions
                          if fn.body_scope is not None and fn.body_scope.owner == scope),
                         'loop caller has no unique original function')
        summary = loop_dependencies(source.material, function)
        for loop in summary['loops']:
            if any(call['result'] == encode(link.result) for call in loop['body_calls']):
                matches.append((index, loop))
    return _only(matches, 'expected one original loop containing the selected normal body call')


@replay_transaction()
def project(entry_calls, accesses, *, allocation_trace=None, project_root=None):
    source_ref = artifact_reference(Path(entry_calls).resolve(strict=True), ENTRY_CALLS)
    report = read_artifact(source_ref)
    storage_ref = artifact_reference(Path(accesses).resolve(strict=True), storage.SCHEMA)
    approved = storage.check(storage_ref['uri'])
    region = stateful_from_dict(read_artifact(approved['boundary']))
    bindings, _ = storage.observed_storage_bindings(region, approved['storage_evidence']['artifact']['uri'],
                                                    observation=approved['observation'])
    read_artifact(report['program'])
    with BuildSession(Path(report['program']['uri']).parent) as session:
        with _admit_entry_calls(session, report):
            view = session.view
            current_catalog = artifact_reference(view.catalog.path / 'result.json', view.catalog.report['schema'])
            if approved['catalog'] != current_catalog:
                raise ValueError('storage effects and entry interfaces require the same original source catalog')
            declarations = tuple((row, decode(T.CallLink, row['interface']['link']))
                                 for row in report['interfaces'])
            identity = region.source.origins[0]
            index = _only((i for i, (_, link) in enumerate(declarations) if link.body is not None and
                           (link.body.unit, link.body.template.occurrence) == (identity['unit'], identity['occurrence'])),
                          'stateful storage profile does not map to one admitted call site')
            scope = declarations[index][0]['caller_scope']
            siblings = tuple(i for i, (row, _) in enumerate(declarations) if row['caller_scope'] == scope)
            order, paths, rows = compose(view, decode(F.CodeScopeRef, scope),
                (declarations[i][1] for i in siblings), region, siblings.index(index), bindings)
            parent, loop = _enclosing_loop(view, declarations, scope)
            if allocation_trace is not None:
                from scar.analysis.observed_allocation_epochs import observe_allocation_epochs
                from scar.analysis.observed_loop_calls import observed_loop_calls
                from scar.analysis.observed_loop_control import observed_loop_control
                from scar.analysis.source_observed_calls import ObservedSourceCallsiteIndex
                from scar.analysis.source_observed_codes import ObservedSourceCodeProvider

                observed = ObservedSourceCallsiteIndex(ObservedSourceCodeProvider(view, allocation_trace, project_root))
                metadata = approved['storage_evidence']['artifact']
                lifetime = observe_allocation_epochs(region, metadata['uri'], observed.codes.trace,
                    observation=approved['observation'], expected_sha256=metadata['sha256'])
                calls = observed_loop_calls(observed, declarations[parent][1],
                    tuple(declarations[siblings[i]][1] for i in order), order.index(siblings.index(index)),
                    lifetime['endpoints'])
                control = observed_loop_control(observed, declarations[parent][1], loop, calls['windows'], lifetime)
                if control is not None:
                    from scar.analysis.source_slot_continuity import project_slot_continuity
                    slots = project_slot_continuity(view, region, decode(F.CodeScopeRef, declarations[parent][0]['caller_scope']),
                        declarations[parent][1], declarations[index][1], decode(T.GlobalPort, loop['header']))
    reads, intervals = storage._repeated_read_records(analyze_loop_content(row.access for row in rows))
    result = dict(schema=SCHEMA, schema_version=1, entry_calls=source_ref, storage=storage_ref,
                original_loop_header=loop['header'], original_loop_body=loop['body'],
                original_control_backedges=loop['control_backedges'],
                loop_body_interface=f'#/interfaces/{parent}',
                ordered_child_interfaces=[f'#/interfaces/{siblings[i]}' for i in order],
                paths=[dict(interface=f'#/interfaces/{siblings[i]}', input=key, address=encode(path))
                       for i, key, path in paths],
                accesses=encode(rows), conditional_repeated_body=reads, byte_interval_sets=intervals,
                remaining=['ACTUAL_BOUND_METHOD_RECEIVER_AND_CALL_COMPLETIONS',
                           'SLOT_AND_ALLOCATION_CONTINUITY_ACROSS_BACKEDGE',
                           'UNSELECTED_BRANCHES_AND_INTERVENING_EFFECTS'],
                reads_selected_on_whole_loop=False, loop_unrolled=False, transformation_authorized=False)
    if allocation_trace is not None:
        result['observed_interval'] = dict(trace_directory=str(Path(allocation_trace).resolve()),
            project_root=str(Path(project_root).resolve()), allocation_lifetime=lifetime, actual_calls=calls)
        result['remaining'] = ['SLOT_BINDING_CONTINUITY_ACROSS_BACKEDGE',
            'EXACT_BACKEDGE_OCCURRENCE_AND_INTERVENING_EFFECTS', 'UNSELECTED_BRANCHES']
        if control is not None:
            result['observed_interval']['control'] = control
            result['observed_interval']['direct_slots'] = slots
            if control['exact_source_backedges'] == 1:
                result['remaining'][1] = 'INTERVENING_EFFECTS_AND_CALLBACKS'
        if not lifetime['covered_observed_interval']:
            result['remaining'].append('ALLOCATION_LIFETIME_AT_SELECTED_INTERVAL')
        if not calls['all_selected_windows_bound']:
            result['remaining'].append('OBSERVED_PARENT_AND_CHILD_COMPLETIONS')
    return result


def check(path):
    saved = read_artifact(artifact_reference(Path(path).resolve(strict=True), SCHEMA))
    scope = saved.get('observed_interval')
    options = {} if scope is None else dict(allocation_trace=scope['trace_directory'], project_root=scope['project_root'])
    rebuilt = project(saved['entry_calls']['uri'], saved['storage']['uri'], **options)
    if saved != rebuilt:
        raise ValueError('original loop content projection differs from replay')
    return rebuilt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('entry-calls', 'storage', 'out', 'check'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--allocation-trace', type=Path)
    parser.add_argument('--project-root', type=Path)
    args = parser.parse_args(argv)
    inputs = (args.entry_calls, args.storage, args.out)
    if (args.allocation_trace is None) != (args.project_root is None):
        parser.error('--allocation-trace and --project-root must be provided together')
    if args.check and not any((*inputs, args.allocation_trace)):
        result = check(args.check)
    elif not args.check and all(inputs):
        result = project(args.entry_calls, args.storage, allocation_trace=args.allocation_trace,
                         project_root=args.project_root)
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, separators=(',', ':'), sort_keys=True, allow_nan=False)
            stream.write('\n')
    else:
        parser.error('provide --check, or --entry-calls, --storage and --out')
    print(json.dumps(dict(schema=SCHEMA, interfaces=len(result['ordered_child_interfaces']),
                          accesses=len(result['accesses']), reads=len(result['conditional_repeated_body']))))


if __name__ == '__main__':
    main()
