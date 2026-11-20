"""Compose original numeric, copy, and scalar-return interfaces for one update."""

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path

from scar.analysis.array_copy_effects import derive_array_copy_effects
from scar.analysis.bounded_entry_calls import SCHEMA as ENTRY_CALLS, _admit_entry_calls
from scar.analysis.build import BuildSession
from scar.analysis.call_operand_reader import read_call_operands
from scar.analysis.replay_transaction import detached_replay, replay_transaction
from scar.analysis.session_scalar_flow import derive_scalar_flow
from scar.analysis.source_address_paths import AddressResolver
from scar.analysis.source_ordered_call_content import SCHEMA as ORDERED, check as check_ordered
from scar.analysis.source_reference_slice import ReferenceSlicer, source_port
from scar.analysis.source_slot_continuity import _field_path, _normal_window
from scar.analysis.source_value_flow import project_value_flow
from scar.compiler.stateful_model import stateful_from_dict
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program import normal_graph as G, normal_targets as T
from scar.ir.program.call_edges import InvocationKey
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.source_address_paths import AddressPath
from scar.ir.record_codec import decode, encode
from scar.ir.v2 import StorageRegion


SCHEMA = 'scar.source-native-update'
_OPERATORS = {'Add':'ADD', 'Sub':'SUBTRACT', 'Mult':'MULTIPLY', 'Div':'TRUE_DIVIDE',
              'FloorDiv':'FLOOR_DIVIDE', 'Mod':'REMAINDER', 'UAdd':'POSITIVE', 'USub':'NEGATIVE'}


def _only(rows, detail):
    values = tuple(rows)
    if len(values) != 1:
        raise ValueError(detail)
    return values[0]


def _interface(report, locator):
    try:
        prefix, ordinal = locator.rsplit('/', 1)
        if prefix != '#/interfaces' or str(int(ordinal)) != ordinal or int(ordinal) < 0:
            raise ValueError
        row = report['interfaces'][int(ordinal)]
    except (AttributeError, IndexError, ValueError):
        raise ValueError('expected an original entry-call interface locator') from None
    return row, decode(T.CallLink, row['interface']['link'])


def _function(view, body):
    unit = view.unit(body.unit)
    ordinal, fn = _only(((i, row) for i, row in enumerate(unit.facts.functions)
                         if (row.template, row.body) == (body.template, body.region)),
                        'source update requires one original function per selected body')
    return unit, ordinal, fn


def _reference_paths(view, parent_unit, parent, region, link, child_unit, child):
    slicer = ReferenceSlicer(view.catalog)
    addresses = AddressResolver(view.catalog.fact_store, parent_unit.facts, parent.body_scope.owner)
    caller_formals = {source_port(parent_unit.facts, formal):name
                      for name, formal in zip(parent.parameter_names, parent.formals[1:])}
    inputs = {}
    for spec in region.inputs:
        formal_index = child.parameter_names.index(spec.path[0])
        formal = source_port(child_unit.facts, child.formals[formal_index + 1])
        route = _only((row for row in link.parameters
                       if row.kind is T.RouteKind.ARGUMENT and row.formal == formal),
                      'numeric input lacks its original formal/actual route')
        actual, = route.actuals
        _, ordinal = slicer.local(actual)
        address = _only(addresses.paths(ordinal), 'numeric input lacks one source field path')
        path = _field_path(address)
        if path is None or path[0] not in caller_formals:
            raise ValueError('numeric region input does not descend from an original update formal')
        fields = (*path[1], *spec.path[1:])
        inputs[spec.id] = dict(id=spec.id, kind=spec.kind, input_path=list(spec.path),
                               parent_path=[caller_formals[path[0]], *fields], parent_root=path[0],
                               selectors=fields, source_actual=address, formal_port=formal)
    return inputs


def _return_expression(unit, parent, scalar, tail, result_port, numeric_inputs):
    """Resolve a single original scalar expression through its F4 data ports."""
    producer = {output:row for row in scalar['operations'] for output in row['outputs']}
    reverse = defaultdict(set)
    for edge in scalar['edges']:
        if edge['kind'] == G.EdgeKind.DATA.value:
            reverse[edge['target']].add(edge['source'])
    formals = set(parent.parameter_names)
    scalar_roots = defaultdict(list)
    for row in numeric_inputs.values():
        if row['kind'] == 'scalar':
            scalar_roots[tuple(row['parent_path'])].append(row['id'])

    def field(node, active):
        row = producer.get(node)
        if row is not None and row['role'] == 'ATTRIBUTE_GET' and row['index'] in tail:
            fact = _only((ref for ref in row['source_facts'] if ref['table'] == 'attributes'),
                         'scalar return field requires one original GET fact')
            return (*visit(row['inputs'][0], active, reference_only=True),
                    unit.facts.attributes[fact['ordinal']].name)
        paths = scalar['ports'][node]['reference_paths']
        if (len(paths) == 1 and len(paths[0]) == 1 and paths[0][0] in formals
                and (row is None or row['role'] == 'LEXICAL_BINDING')):
            return tuple(paths[0])
        source = _only(reverse[node], 'scalar return reference path has no unique original producer')
        return visit(source, active, reference_only=True)

    def visit(node, active, reference_only=False):
        if node in active or len(active) >= 128:
            raise ValueError('cyclic original scalar return route')
        active = (*active, node)
        row = producer.get(node)
        if reference_only:
            return field(node, active)
        if row is None and len(reverse[node]) == 1:
            return visit(next(iter(reverse[node])), active)
        if row is not None and row['role'] == 'SCALAR_ARITHMETIC' and row['index'] in tail:
            source = _only((ref for ref in row['source_facts'] if ref['table'] == 'operations'),
                           'scalar return operation needs one exact selector fact')
            selector = unit.facts.operations[source['ordinal']].selector.value
            if selector not in _OPERATORS:
                raise ValueError('original scalar return operator is outside the finite profile')
            return dict(operator=_OPERATORS[selector], operands=[visit(i, active) for i in row['inputs']])
        path = field(node, active)
        if len(path) == 1:
            return dict(kind='ARGUMENT', parameter=path[0])
        root = _only(scalar_roots[path], 'original scalar return lacks one matching numeric input path')
        return dict(kind='REGION_INPUT', id=root)

    return visit(result_port, ())


def _entry_guards(flow, scalar, ports, first):
    body = flow.body
    start = G.Port(body.template, body.region, G.PortKind.ENTER)
    stop = G.Port(first.result.port.template, first.result.port.owner, G.PortKind.ENTER)
    membership, edges = _normal_window(flow, start, {stop}, None)
    producer = {output:row for row in scalar['operations'] for output in row['outputs']}
    reverse = defaultdict(set)
    for row in scalar['edges']:
        if row['kind'] == G.EdgeKind.DATA.value:
            reverse[row['target']].add(row['source'])
    choices = defaultdict(set)
    for edge in edges:
        if edge.condition is not None:
            choices[edge.condition].add(edge.when)
    guards, frontiers = [], []
    if not membership:
        frontiers.append(dict(reason='NO_ORIGINAL_NORMAL_PREFIX_TO_SELECTED_NUMERIC_CALL'))
    for condition, outcomes in sorted(choices.items(), key=lambda row: repr(row[0])):
        node = ports.index(condition)
        visited = set()
        while node not in visited and node not in producer and len(reverse[node]) == 1:
            visited.add(node)
            node = next(iter(reverse[node]))
        operation = producer.get(node)
        paths = (scalar['ports'][operation['inputs'][0]]['reference_paths'] if operation is not None
                 and operation['role'] == 'TRUTH' and len(operation['inputs']) == 1 else ())
        if len(outcomes) == 1 and len(paths) == 1:
            guards.append(dict(kind='EXACT_BUILTIN_BOOL', path=paths[0], expected=next(iter(outcomes)),
                               condition_port=condition, original_truth_operation=ports[operation['entry']]))
        else:
            frontiers.append(dict(reason='SELECTED_PREFIX_TRUTH_DOES_NOT_HAVE_ONE_ORIGINAL_FIELD_ROUTE',
                                  condition_port=condition, outcomes=sorted(outcomes)))
    return dict(guards=guards, frontiers=frontiers, guarded_call=first.result,
                runtime_guards_proven=False)


def _scalar_interfaces(view, unit, ordinal, parent, body, first, last, last_copy, numeric_inputs):
    flow = project_value_flow(unit.material, source_port(unit.facts, parent.value), body)
    start = G.Port(last.result.port.template, last.result.port.owner, G.PortKind.EXIT, completion=C.FALLTHROUGH)
    stop = G.Port(body.template, body.region, G.PortKind.EXIT, completion=C.RETURN)
    membership, edges = _normal_window(flow, start, {stop}, None)
    query = _only((i for i, row in enumerate(unit.facts.calls)
                   if source_port(unit.facts, row.result) == last.result),
                  'last copy return needs one original call-result fact')
    scalar = derive_scalar_flow(view, unit=unit.unit, function_ordinal=ordinal, flow=flow,
                                known_call_returns={query: 'NONE'})
    ports = [decode(G.Port, row['port']) for row in scalar['ports']]
    entry = _entry_guards(flow, scalar, ports, first)
    operations = {ports[row['entry']]:row for row in scalar['operations'] if ports[row['entry']] in membership}
    incoming = defaultdict(int)
    outgoing = defaultdict(list)
    for edge in sorted(edges, key=repr):
        outgoing[edge.source].append(edge.target)
        incoming[edge.target] += 1
    result = G.Port(body.template, body.region, G.PortKind.RESULT, completion=C.RETURN)
    todo = deque((start,))
    seen = set()
    selected = []
    while todo:
        point = todo.popleft()
        seen.add(point)
        if point in operations:
            selected.append(operations[point])
        for target in outgoing[point]:
            incoming[target] -= 1
            if incoming[target] == 0:
                todo.append(target)
    frontier = [dict(operation=row['index'], reasons=row['frontiers']) for row in selected if row['frontiers']]
    tail = {row['index'] for row in selected}
    expression = None
    try:
        if stop not in seen or len(selected) != len(operations):
            raise ValueError('original scalar return has no complete ordered post-copy path')
        expression = _return_expression(unit, parent, scalar, tail, ports.index(result), numeric_inputs)
    except ValueError as error:
        frontier.append(dict(reason=str(error)))
    guards = [guard for guard in scalar['guards'] if guard.get('operation') in tail
              or tail.intersection(guard.get('sites', ()))]
    tail = dict(after_call=encode(last.result), source_function=scalar['source_function'],
                normal_return=encode(T.GlobalPort(unit.unit, result)),
                operations=[dict(**row, source_entry=encode(ports[row['entry']])) for row in selected],
                ports=[dict(index=i, **scalar['ports'][i]) for i in sorted({i for row in selected
                       for i in (row['entry'], *row['inputs'], *row['outputs'])})],
                return_expression=expression, guards=guards, frontiers=frontier,
                last_copy_return=dict(call=encode(last.result), query_ordinal=query,
                    copy_effects=last_copy['id'], kind='NONE',
                    evidence='ORIGINAL_COPY_ONLY_EFFECT_PROFILE_REQUIRES_NONE_RETURN'),
                policy='PRESERVE_ORIGINAL_SCALAR_PROTOCOL_AND_NORMAL_RETURN',
                region_scalar_read='ORIGINAL_FIELD_PATH_EVALUATED_AT_SCALAR_TAIL',
                runtime_scalar_guards_proven=False)
    return entry, tail


def _observations(ordered, links):
    observed = ordered.get('observed_interval')
    if observed is None or not observed['actual_calls']['all_selected_windows_bound']:
        raise ValueError('source update requires the replayed original parent and children')
    trace = read_call_operands(observed['trace_directory'], observed['project_root'])
    frames = {frame.invocation:frame for frame in trace.entries}
    attempts = {(attempt.process_id, attempt.index):attempt for attempt in trace.attempts}
    windows = observed['actual_calls']['windows']
    by_call = [[] for _ in links]
    for window in windows:
        rows = (window['parent'], *window['children'])
        if len(rows) != len(links):
            raise ValueError('original runtime update child denominator differs')
        for bucket, row, link in zip(by_call, rows, links, strict=True):
            identity = decode(InvocationKey, row['invocation'])
            frame = frames.get(identity)
            attempt = attempts.get((identity.process_id, row['attempt']['index']))
            if (row['source_result'] != encode(link.result) or frame is None or not frame.parameters_complete
                    or encode(frame.anchor) != row['entry'] or attempt is None
                    or encode(attempt.anchor) != row['attempt'] or attempt.invocation != frame.caller_invocation):
                raise ValueError('original bound update call or actual frame parameters differ')
            bucket.append(dict(**row, formal_references=encode(frame.parameters),
                               monitoring_arg0=attempt.arg0, caller=encode(attempt.invocation)))
    return by_call


@replay_transaction()
def project(ordered_graph, *, entry_calls=None):
    reference = artifact_reference(Path(ordered_graph).resolve(strict=True), ORDERED)
    ordered = check_ordered(reference['uri'])
    if read_artifact(reference) != ordered:
        raise ValueError('original ordered update projection changed during replay')
    entry_ref = ordered['entry_calls']
    if entry_calls is not None and artifact_reference(Path(entry_calls).resolve(strict=True), ENTRY_CALLS) != entry_ref:
        raise ValueError('entry-call projection differs from the ordered update graph')
    entry = read_artifact(entry_ref)
    storage = read_artifact(ordered['storage'])
    region_ref = storage['boundary']
    region = stateful_from_dict(read_artifact(region_ref))
    array_bindings = {name:decode(StorageRegion, row) for name, row in storage['bindings'].items()}
    parent_locator = ordered['loop_body_interface']
    _, parent_link = _interface(entry, parent_locator)
    locators = ordered['ordered_child_interfaces']
    links = tuple(_interface(entry, locator)[1] for locator in locators)
    address_rows = {(row['interface'], row['input']):decode(AddressPath, row['address']) for row in ordered['paths']}
    origin, = (row for row in region.source.origins if row.get('kind') == 'source_identity')
    if (len(links) < 2 or links[0].body is None or
            (links[0].body.unit, links[0].body.template.occurrence) != (origin['unit'], origin['occurrence'])
            or any(link.body is None or link.binding is not T.ArgumentBinding.DIRECT for link in (parent_link, *links))):
        raise ValueError('update needs a leading numeric call followed by original direct copy interfaces')
    observations = _observations(ordered, (parent_link, *links))
    read_artifact(entry['program'])
    with BuildSession(Path(entry['program']['uri']).parent) as session:
        with _admit_entry_calls(session, entry):
            view = session.view
            unit, ordinal, parent = _function(view, parent_link.body)
            numeric_unit, _, numeric = _function(view, links[0].body)
            bindings = _reference_paths(view, unit, parent, region, links[0], numeric_unit, numeric)
            by_path = {}
            array_inputs = []
            for name, storage_region in array_bindings.items():
                row = bindings[name]
                if row['source_actual'] != address_rows[locators[0], name]:
                    raise ValueError('numeric array source route differs from original access projection')
                key = row['parent_root'], row['selectors']
                if key in by_path:
                    raise ValueError('distinct numeric roots need distinct source field paths')
                by_path[key] = name
                array_inputs.append(dict(**row, shape=storage_region.shape, strides=storage_region.strides,
                    offset=storage_region.offset, dtype=storage_region.dtype, device=storage_region.device,
                    allocation=storage_region.allocation))
            calls = [dict(kind='STATEFUL_NUMERIC_REGION', interface=locators[0], source_result=links[0].result,
                          input_ids=list(bindings), numeric_region=region_ref, observations=observations[1])]
            copies = {}
            for locator, link, observed in zip(locators[1:], links[1:], observations[2:], strict=True):
                source, i, callee = _function(view, link.body)
                mapping = []
                for index, parameter in enumerate(callee.parameter_names):
                    path = address_rows[locator, parameter]
                    root = by_path.get(_field_path(path))
                    if root is None:
                        raise ValueError('original copy formal lacks its admitted numeric root')
                    mapping.append(dict(formal_index=index, root_id=root, formal_port=source_port(
                        source.facts, callee.formals[index + 1]), actual_path=path))
                shapes = {row['formal_index']:array_bindings[row['root_id']].shape for row in mapping}
                key = source.unit, i, tuple(shapes.items())
                if key not in copies:
                    copies[key] = derive_array_copy_effects(view, unit=source.unit, function_ordinal=i, input_shapes=shapes)
                if copies[key]['status'] != 'CLOSED_COPY_EFFECTS_CONDITIONAL':
                    raise ValueError('original post-numeric call is outside the guarded copy-only effect profile')
                calls.append(dict(kind='ARRAY_COPY_CALL', interface=locator, source_result=link.result,
                                  formal_to_root=mapping, copy_effects=copies[key], observations=observed))
            guards, scalar = _scalar_interfaces(view, unit, ordinal, parent, parent_link.body,
                                                 links[0], links[-1], calls[-1]['copy_effects'], bindings)
    return encode(dict(schema=SCHEMA, schema_version=1, ordered_graph=reference, entry_calls=entry_ref,
        numeric_region=region_ref, parent=dict(interface=parent_locator, source_result=parent_link.result,
            body=parent_link.body, parameters=parent.parameter_names, observations=observations[0]),
        array_inputs=array_inputs, scalar_inputs=[row for row in bindings.values() if row['kind'] == 'scalar'],
        ordered_calls=calls, entry_guards=guards, scalar_tail=scalar, remaining=ordered['remaining'],
        allocation_lifetime=ordered['observed_interval']['allocation_lifetime'],
        caller_replacement_qualified=False, effects_closed=False, transformation_authorized=False))


def check(path):
    reference = artifact_reference(Path(path).resolve(strict=True), SCHEMA)

    def replay():
        saved = read_artifact(reference)
        rebuilt = project(saved['ordered_graph']['uri'], entry_calls=saved['entry_calls']['uri'])
        if saved != rebuilt or saved != read_artifact(reference):
            raise ValueError('source update composition differs from original source and observed order')
        return rebuilt

    return detached_replay('scar.source-native-update/v1', tuple(sorted(reference.items())), replay)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ordered-graph', type=Path)
    parser.add_argument('--entry-calls', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args(argv)
    if args.check and not any((args.ordered_graph, args.entry_calls, args.out)):
        result = check(args.check)
    elif args.ordered_graph and args.out and not args.check:
        result = project(args.ordered_graph, entry_calls=args.entry_calls)
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
            stream.write('\n')
    else:
        parser.error('provide --check or --ordered-graph with --out')
    print(json.dumps(dict(schema=SCHEMA, calls=len(result['ordered_calls']),
                          scalar_return=result['scalar_tail']['return_expression'])))


if __name__ == '__main__':
    main()
