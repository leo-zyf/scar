"""Project F4 reference and world paths for same-invocation ATen scopes."""

from collections import defaultdict, deque
from hashlib import sha256
import json
from pathlib import Path

from scar.analysis.source_observed_calls import ObservedSourceCallsiteIndex
from scar.analysis.source_observed_codes import ObservedSourceCodeProvider
from scar.analysis.source_reference_slice import source_port
from scar.analysis.source_value_flow import project_value_flow
from scar.analysis.source_world_flow import _state
from scar.ir.program import normal_graph as G, normal_targets as T, source_productions as P
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.record_codec import encode


def _paths(start, adjacency):
    """One original path witness per reachable port."""
    parents = {start: None}
    queue = deque((start,))
    while queue:
        point = queue.popleft()
        for target, witness in adjacency[point]:
            if target not in parents:
                parents[target] = (point, witness)
                queue.append(target)
    return parents


def _witness(parents, target):
    path = []
    while parents[target] is not None:
        target, edge = parents[target]
        path.append(edge)
    return list(reversed(path))


def _adjacencies(flow):
    references, world = defaultdict(list), defaultdict(list)
    for ordinal, edge in enumerate(flow.edges):
        if C.RAISE in (edge.source.completion, edge.target.completion):
            continue
        if edge.kind is G.EdgeKind.DATA:
            references[edge.source].append((edge.target, dict(relation='F4_DATA_TRANSPORT', edge=ordinal)))
        elif edge.kind is G.EdgeKind.STATE:
            world[edge.source].append((edge.target, dict(relation='F4_WORLD_TRANSPORT', edge=ordinal)))
    for location in flow.locations:
        if len(location.writes) != 1:
            continue
        write, = location.writes
        for read in location.reads:
            references[write.value].append((read.value, dict(relation='F4_SINGLE_LEXICAL_WRITER',
                write=encode(write.fact), read=encode(read.fact))))
    for ordinal, op in enumerate(flow.operations):
        for before in op.inputs:
            if not _state(before.value_type):
                continue
            for after in op.outputs:
                if after.port.completion is C.FALLTHROUGH and _state(after.value_type):
                    world[before.port].append((after.port, dict(relation='F4_NORMAL_OPERATION_WORLD',
                        operation=ordinal, argument=before.port.ordinal, result=after.port.ordinal)))
    return references, world


def _original_calls(view, coordinates, scoped_calls):
    pid = coordinates.target_process.pid
    src_meta, = (row for row in coordinates.report['membership']
                 if row['artifact']['schema'] == 'scar.call-edge-meta.v1')
    # The coordinate binder and operand reader independently check this member.
    metadata_path = Path(src_meta['local_path'])
    raw = metadata_path.read_bytes()
    if (len(raw), sha256(raw).hexdigest()) != (src_meta['artifact']['size_bytes'], src_meta['artifact']['sha256']):
        raise ValueError('original call metadata changed during source replay')
    root = json.loads(raw)['project_root']
    code = ObservedSourceCodeProvider(view, metadata_path.parent, root)
    index = ObservedSourceCallsiteIndex(code)
    attempted = {row['attempt'] for row in scoped_calls}
    bindings = defaultdict(list)
    for bound in index.bindings:
        if bound.attempt.process_id == pid and bound.attempt.index in attempted:
            bindings[bound.attempt.index].append(bound)
    if set(bindings) != attempted or any(len(rows) != 1 for rows in bindings.values()):
        raise ValueError('selected native scope has no unique original F4 CALL')

    owners = {}
    for binding in scoped_calls:
        bound, = bindings[binding['attempt']]
        event = bound.attempt
        anchor = binding['operand']
        invocation = dict(process_id=event.process_id, thread_id=event.thread_id, invocation_id=event.invocation_id)
        trace_file, = (row for row in code.trace.files
                       if str(Path(code.trace.directory) / row.path) == event.anchor.path)
        if (binding['invocation'] != invocation or binding['site'] != encode(event.site)
                or anchor['locator'] != f'#/lines/{event.anchor.line_number}'
                or (anchor['artifact']['uri'], anchor['artifact']['sha256'], anchor['artifact']['size_bytes'])
                != (event.anchor.path, trace_file.raw_sha256, trace_file.raw_bytes)):
            raise ValueError('F4 and native scope name different original CALL bytes')
        owners[event.index] = (invocation, bound)
    scopes = {(row.unit, row.owner.owner, inv['thread_id'], inv['invocation_id']) for inv, row in owners.values()}
    if len(scopes) != 1:
        raise ValueError('one original caller activation required for regional planning')
    rows = [owners[attempt][1] for attempt in sorted(owners)]
    if len({row.query for row in rows}) != len(rows):
        raise ValueError('repeated source CALL requires a retained loop transfer interface')
    return index, owners[rows[0].attempt.index][0], rows


def _source_rows(unit, index, bindings):
    rows = []
    for bound in bindings:
        call = unit.facts.calls[bound.query]
        port = lambda ordinal: source_port(unit.facts, ordinal)
        result = bound.result
        key = result.port.template
        context = index._contexts[unit.unit][key]
        step = context.steps[result.port.owner]
        ins = step.instruction
        positional, keywords = [port(i) for i in call.positional], [(key, port(i)) for key, i in call.keywords]
        worlds = (port(call.world), port(call.result_world))
        if (type(ins) is not P.ApplyLanguageAction or ins.action is not P.LanguageActionKind.CALL
                or call.positional_pack is not None or call.keyword_pack is not None
                or context.owner(step.region) != bound.owner or context.result(step) != result
                or context.worlds(step) != worlds
                or tuple(context.value(arg) for arg in step.arguments)
                != (bound.callee, *positional, *(v for _, v in keywords), worlds[0])):
            raise ValueError('call layout or world ports differ from the original ordinary F4 CALL')
        rows.append(dict(attempt=bound.attempt.index, source_call=encode(bound.call),
            site=encode(bound.attempt.site), result=result, positional=positional,
            keywords=keywords, world_before=worlds[0], world_after=worlds[1]))
    return rows


def _call_slots(call):
    yield from ((i, None, port) for i, port in enumerate(call['positional']))
    yield from ((i, name, port) for i, (name, port) in enumerate(call['keywords']))


def _formal_routes(unit, slicer, fn, rows, references):
    if len(fn.formals) != len(fn.parameter_names) + 2:
        raise ValueError('original function formals lack their environment/world boundaries')
    routes = []
    for ordinal, (name, local) in enumerate(zip(fn.parameter_names, fn.formals[1:-1], strict=True)):
        formal = source_port(unit, local)
        fact = encode(slicer.fact(unit, 'ports', local))
        paths = _paths(formal.port, references)
        for call in rows:
            for slot, keyword, actual in _call_slots(call):
                if actual.port in paths:
                    routes.append(dict(formal_ordinal=ordinal, formal_name=name, formal=encode(formal),
                        formal_fact=fact, after=call['attempt'], source_input=encode(actual),
                        source_slot=dict(ordinal=slot, keyword=keyword), source_path=_witness(paths, actual.port)))
    return routes


def _return_points(flow, result, references):
    """Find direct and fixed tuple element boundaries without conflating their identities."""
    yield result, [], []
    backward = defaultdict(list)
    for start, destinations in references.items():
        for end, evidence in destinations:
            backward[end].append((start, evidence))
    paths = _paths(result, backward)
    for ordinal, op in enumerate(flow.operations):
        instruction = op.instruction
        if (type(instruction) is not P.ConstructValue
                or instruction.construction is not P.ConstructionKind.TUPLE
                or instruction.tuple_layout is not P.TupleConstructionLayout.ITEMS):
            continue
        outputs = [value.port for value in op.outputs if not _state(value.value_type)
                   and value.port.completion is C.FALLTHROUGH]
        if len(outputs) != 1 or outputs[0] not in paths:
            continue
        tail = list(reversed(_witness(paths, outputs[0])))
        for i, item in enumerate(value for value in op.inputs if not _state(value.value_type)):
            projection = dict(relation='F4_FIXED_TUPLE_ITEM', operation=ordinal,
                              argument=item.port.ordinal, result=outputs[0].ordinal, component=i)
            yield item.port, [i], [projection, *tail]


def project_aten_source_calls(view, capture):
    """Retain F4 references and state paths at original checked native scopes."""
    index, invocation, bindings = _original_calls(view, capture.coordinates, capture.calls)
    unit = view.unit(bindings[0].unit)
    functions = [(i, fn) for i, fn in enumerate(unit.facts.functions)
                 if fn.body_scope is not None and fn.body_scope.owner == bindings[0].owner.owner]
    if len(functions) != 1:
        raise ValueError('selected scopes have no unique shared F4 function body')
    function_index, fn = functions[0]
    body = T.Body(unit.unit, fn.template, fn.body)
    flow = project_value_flow(unit.material, source_port(unit.facts, fn.value), body)
    references, world = _adjacencies(flow)
    rows = _source_rows(unit, index, bindings)
    world_order = []
    for before, after in zip(rows, rows[1:]):
        paths = _paths(before['world_after'].port, world)
        destination = after['world_before'].port
        if destination not in paths:
            raise ValueError('original normal F4 world route between selected scopes is unresolved')
        world_order.append(dict(before=before['attempt'], after=after['attempt'], path=_witness(paths, destination),
            source_refs=[before['source_call'], after['source_call']]))

    routes, return_routes = [], []
    formal_routes = _formal_routes(unit.facts, index.slicer, fn, rows, references)
    if fn.returns is None:
        result, return_fact, return_points = None, None, ()
    else:
        return_site = source_port(unit.facts, fn.returns)
        result = encode(return_site)
        return_fact = encode(index.slicer.fact(unit.facts, 'ports', fn.returns))
        return_points = tuple(_return_points(flow, return_site.port, references))
    for index_in_rows, before in enumerate(rows):
        paths = _paths(before['result'].port, references)
        origin = encode(before['result'])
        for point, component, tail in return_points:
            if point in paths:
                return_routes.append(dict(before=before['attempt'], source_result=origin, return_port=result,
                    return_fact=return_fact, return_path=component, source_path=[*_witness(paths, point), *tail]))
        for after in rows[index_in_rows + 1:]:
            for ordinal, name, port in _call_slots(after):
                if port.port not in paths:
                    continue
                routes.append(dict(before=before['attempt'], after=after['attempt'],
                    source_result=origin, source_input=encode(port),
                    source_slot=dict(ordinal=ordinal, keyword=name), source_path=_witness(paths, port.port),
                    wrapper_mapping_proven=False, content_preserved=False, ready=False))

    index.check()
    return dict(schema='scar.f4-observed-aten-region', schema_version=1,
        invocation=invocation, body=encode(body),
        body_declaration=encode(index.slicer.fact(unit.facts, 'functions', function_index)),
        source_templates=encode(flow.source_templates), path_basis='REPLAYED_F4_BODY_VALUE_FLOW_ORDINALS',
        calls=encode(rows), world_order=world_order, data_routes=routes,
        formal_routes=formal_routes, return_routes=return_routes,
        boundary_path_status='POSSIBLE_F4_REFERENCE_ROUTES',
        source_normal_path_only=True, effects_established=False, transformation_authorized=False)


def corroborate_aten_routes(source, native):
    """Attach same-run provider descriptors as observations for the source routes."""
    operators = defaultdict(list)
    for row in native['operators']:
        if row['invocation'] != source['invocation']:
            raise ValueError('F4 route and observed ports name different caller activations')
        operators[row['call_attempt']].append(row)
    for route in source['data_routes']:
        left, right = operators[route['before']], operators[route['after']]
        matches = []
        if len(left) == len(right) == 1:
            for out in left[0]['outputs']:
                for inp in right[0]['inputs']:
                    if (out['kind'] == inp['kind'] == 'TENSOR_REFERENCE'
                            and out['descriptor_id'] == inp['descriptor_id']):
                        matches.append(dict(output=out['source'], input=inp['source'],
                            output_ordinal=out['ordinal'], input_ordinal=inp['ordinal']))
        # Such a match does not select a wrapper argument mapping or a Tensor version.
        route['descriptor_matches'] = matches
    return source
