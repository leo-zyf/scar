"""Connect the sealed fixed-grid source CLI's effects to its original F4 body."""

import ast
from collections import defaultdict, deque
import json
from pathlib import Path

from scar.analysis.build import BuildSession
from scar.analysis.cpu_entry_publication_order import _body_ports, _control, _point, _port, _prove
from scar.analysis.cpu_provider_graph import _source_anchors
from scar.analysis.entry_obligations import verify_entry_obligations
from scar.analysis.entry_tensor_flow import _SourceRoutes
from scar.analysis.fixed_grid_entry_abi import _span, derive_fixed_grid_file_entry
from scar.analysis.fixed_grid_source import _unique
from scar.analysis.replay_transaction import replay_transaction
from scar.compiler import fixed_grid_source_provider as provider
from scar.compiler.aten_source_entry_binding import verify_source_entry_binding
from scar.ir.execution_recipe import _canonical, read_artifact
from scar.ir.program import normal_graph as G, source_productions as P
from scar.ir.program.normal_graph import AttributeAction
from scar.ir.program.signature_patterns import CountBinding
from scar.ir.program.structured import PythonProtocolKind
from scar.ir.record_codec import encode


SCHEMA = 'scar.f4-cpu-fixed-grid-cli-effect-bridge'
_ORDER_PROVEN = 'PROVEN_SCOPED_NORMAL_SUCCESS'
_ROUTE_GAPS = (
    'observed descriptor call lacks a unique original receiver declaration',
    'observed receiver lacks a declared source reference route',
    'observed operand lacks a declared source reference route',
    'original F4 value-to-argument edge is absent or ambiguous',
)


def _need(ok, detail):
    if not ok:
        raise ValueError('CPU fixed-grid CLI effect bridge: ' + detail)


def _position(span):
    first, left, last, right = span
    return [first, last, left, right]


def _source_material(authority, program):
    project, source = (authority[key] for key in ('source_project', 'entry'))
    _need(program['entry'] == source['uri'], 'original F4 entry differs from the checked source CLI')
    manifest = read_artifact(project)

    def member(name):
        path = Path(project['uri']).parent / name
        _need(path == path.resolve(strict=True), 'source project member has an indirect path')
        return read_artifact(dict(uri=str(path), schema='application/octet-stream', **manifest['files'][name]),
                             json_document=False)

    original = json.loads(member('plan/source-entry.json'))
    raw = member('plan/source-entry.py')
    _need(original == authority['source_binding'] and raw == read_artifact(source, json_document=False),
          'archived and live original entry bytes differ')
    entry = original['entry']
    abi = derive_fixed_grid_file_entry(json.loads(member('plan/source-entry-binding.json')), entry, source_bytes=raw)
    execution = json.loads(member('plan/execution.json'))
    rhs = json.loads(member('plan/rhs-recipe.json'))
    binding = verify_source_entry_binding(abi, execution, rhs, manifest=manifest)
    target = program['target']
    expected = dict(uri=target['path'], schema='scar.target-hardware-graph',
                    sha256=target['sha256'], size_bytes=target['bytes'])
    _need(binding['target'] == expected, 'fixed-grid source ABI and F4 program have different physical targets')
    read_artifact(expected)
    files = authority['inputs']
    _need(len(files) == len(binding['files'])
          and all(Path(ref['uri']).name == row['path'] and ref['size_bytes'] == row['payload_bytes']
                  for ref, row in zip(files, binding['files'], strict=True)),
          'provider input files differ from source RHS ports')
    return source, project, raw, entry, abi, binding


def _control_order(session, obligations, body_index, file, stdout):
    result = dict(status='UNRESOLVED', reason='ORIGINAL_BODY_NORMAL_COMPLETION_OPEN',
                  body=f'#/bodies/{body_index}', from_port=None, to_port=None)
    body = obligations['bodies'][body_index]
    # The publication anchors must name the same projected body, not just a
    # matching lexical source span in a different invocation context.
    selected = set(body['operations'])
    indices = [int(anchor['entry_operation'].rsplit('/', 1)[1]) for anchor in (file, stdout)]
    _need(indices[0] != indices[1] and set(indices) <= selected,
          'the two publication calls are absent from one original projected body')
    first, second = (obligations['operations'][index]['reach_port'] for index in indices)
    start, _, _ = _port(obligations, first)
    stop, _, _ = _port(obligations, second)
    _need(start['kind'] == stop['kind'] == 'ENTER', 'original publication lacks an ENTER port')
    ends = [index for index, port in enumerate(obligations['ports'])
            if port['owner'] == start['owner'] and port['path'] == start['path']
            and port['kind'] == 'EXIT' and port['completion'] == 'FALLTHROUGH']
    if len(ends) != 1:
        return dict(result, reason='ORIGINAL_FILE_NORMAL_COMPLETION_PORT_OPEN')
    ports = _body_ports(session.view, obligations, body)
    if ports is None:
        return result
    _, _, entry, exits = ports
    if gap := _prove(_control(obligations, body), entry, exits, first, ends[0], second):
        return dict(result, reason=gap)
    return dict(result, status=_ORDER_PROVEN, reason='ALL_NORMAL_EXITS_AFTER_FILE_REQUIRE_STDOUT_AND_NO_REVERSE_ROUTE',
                from_port=f'#/ports/{ends[0]}', to_port=f'#/ports/{second}')


def _terminal_attributes(source, node):
    def matching():
        for index in source.attrs:
            fact = source.facts.attributes[index]
            if fact.action is not AttributeAction.GET or fact.name != node.attr:
                continue
            occurrence = source.port(fact.value).port.template.occurrence
            syntax = source.roles.syntax.nodes[occurrence]
            values = dict(zip(source.attributes[syntax.constructor], syntax.attributes, strict=True))
            point = [values[k].scalar.to_python()
                     for k in ('lineno', 'end_lineno', 'col_offset', 'end_col_offset')]
            if point == _position(_span(node)):
                yield index, fact

    return _unique(matching(), 'original F4 final shape access')


def _calls(raw, source, entry, abi):
    tree = ast.parse(raw, filename=source['uri'])
    fn = _unique((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == entry),
                 'source CLI entry function')
    positions = {tuple(_span(node)): node for node in ast.walk(fn) if isinstance(node, ast.Call)}
    nodes = {node['id']: (index, node) for index, node in enumerate(abi['graph']['nodes'])}

    def call(role):
        return positions[tuple(nodes[role][1]['source_span'])]

    calls = {role: call(role) for role in ('root', 'steps', 'output', 'recurrence', 'file', 'stdout')}
    terminal = _unique((stmt.value for stmt in fn.body if isinstance(stmt, ast.Assign)
                       and list(_span(stmt)) == nodes['terminal'][1]['source_span']), 'terminal Tensor assignment')
    # These nested expressions are already checked by the retained ABI grammar.
    binary, = calls['file'].args
    dumps, = calls['stdout'].args
    calls.update(terminal=terminal, file_bytes=binary, numpy_view=binary.func.value, stdout_json=dumps)
    data, = dumps.args
    fields = {key.value: (i, item) for i, (key, item) in enumerate(zip(data.keys, data.values, strict=True))}
    for field in abi['publication']['stdout']:
        if field['kind'] == 'FINAL_SHAPE':
            calls['shape:'+field['key']] = fields[field['key']][1]
    return nodes, calls, fields


def _data_path(local, allowed_occurrences, first, last):
    outgoing = defaultdict(list)
    for (source, target), indices in local.edges.items():
        if source != first and _port(local.document, source)[2]['template']['occurrence'] not in allowed_occurrences:
            continue
        if target == last or _port(local.document, target)[2]['template']['occurrence'] in allowed_occurrences:
            outgoing[source].extend((target, index) for index in indices)
    queue, parents = deque((first,)), {first: None}
    while queue and last not in parents:
        for target, edge in outgoing[queue.popleft()]:
            if target not in parents:
                parents[target] = edge
                queue.append(target)
    if last not in parents:
        return None
    path = []
    while (index := parents[last]) is not None:
        path.append(f'#/edges/{index}')
        last = local.document['edges'][index]['source']
    return list(reversed(path))


def _json_map(local, dumps, count):
    call = local.facts.calls[dumps]
    call_occurrence = local.port(call.result).port.template.occurrence
    occurrence = local.roles.field(call_occurrence, 'args').items[0].reference
    document = local.document
    pairs = []
    for index in local.operations.values():
        row = document['operations'][index]
        if row['template']['occurrence'] != occurrence or row['effect_rule']['instruction'] != 'ApplyPythonProtocol':
            continue
        reach = _point(local.view, document, local.unit, row['reach_port'])
        step = local.view.original(local.view.get(reach).owner)
        if (type(step.instruction) is P.ApplyPythonProtocol and step.instruction.protocol is PythonProtocolKind.BUILD_MAP
                and CountBinding('pair_count', count) in step.instruction.bindings):
            pairs.append((index, document['ports'][row['reach_port']]['owner']))
    if len(pairs) != 1:
        return None
    operation, owner = pairs[0]
    target = local.graph_port(local.port(call.positional[0]).port)
    ports = {(port['kind'], port['ordinal']): i for i, port in enumerate(document['ports'])
             if port['owner'] == owner and not port['path'] and port['completion'] in (None, 'FALLTHROUGH')}
    return occurrence, operation, ports, _data_path(local, {occurrence, call_occurrence}, ports['RESULT', 0], target)


def _map_field(local, layout, field_index, origin):
    if layout is None:
        return None
    occurrence, operation, ports, mapping_result_path = layout
    item = local.roles.field(occurrence, 'values').items[field_index].reference
    ends = []
    for index in range(len(local.facts.ports)):
        port = local.port(index).port
        if (port.template.occurrence == item and type(port.owner) is P.ProductionRegionID
                and port.kind is G.PortKind.RESULT and port.ordinal == 0 and port.completion.value == 'FALLTHROUGH'):
            ends.append(index)
    if len(ends) != 1 or not mapping_result_path:
        return None
    end = ends[0]
    reference_path = local._path(end, origin)
    argument = ports['ARGUMENT', 2 * field_index + 1]
    input_path = _data_path(local, {occurrence}, local.graph_port(local.port(end).port), argument)
    if not reference_path or not input_path:
        return None
    return dict(source_reference_path=reference_path, mapping_operation=f'#/operations/{operation}',
                mapping_value_argument=f'#/ports/{argument}', field_input_edges=input_path,
                mapping_result_to_json_argument_edges=mapping_result_path)


def _routes(session, obligations, source, project, raw, entry, abi):
    nodes, calls, field_slots = _calls(raw, source, entry, abi)
    declarations = []
    for role, call in calls.items():
        # The observer spans come directly from the sealed ABI graph, while
        # nested calls are anchored by their exact subexpressions of those nodes.
        root = role if role in nodes else ('file' if role in ('file_bytes', 'numpy_view') else 'stdout')
        declarations.append(dict(kind=role, position=_position(_span(call)), declaration=project,
            slot=f"DERIVED_FIXED_GRID_ABI#/graph/nodes/{nodes[root][0]}", expected_code=None))
    anchors = {row['kind']: row for row in _source_anchors(session, source, entry, declarations, obligations)}
    local = _SourceRoutes(session, obligations, dict(source=dict(
        path=source['uri'], sha256=source['sha256'], function=entry)))
    values = {role: local.call(dict(site=dict(position=row['position']))) for role, row in anchors.items()}
    for role, value in values.items():
        anchor = anchors[role]
        _need(value['operation'] == anchor['operation']['node']['wire']
              and value['operation_pointer'] == anchor['entry_operation']
              and value['call_fact'] == anchor['original_call'], 'F4 source observer joins disagree')
    edges, gaps = [], []

    def route(before, after, kind, find):
        try:
            witness = find()
        except ValueError as error:
            if str(error) not in _ROUTE_GAPS:
                raise
            gaps.append(dict(producer=before, consumer=after, reason=str(error)))
            return
        edges.append(dict(producer=before, consumer=after, kind=kind, **witness))

    def ordinal(role):
        return values[role]['call_ordinal']

    def operand(before, after):
        route(before, after, 'POSITIONAL_ARGUMENT', lambda: local.positional_route(ordinal(after), ordinal(before), 0))

    for before, after in (('output', 'file'), ('terminal', 'numpy_view'), ('numpy_view', 'file_bytes')):
        route(before, after, 'RECEIVER', lambda: local.receiver_route(ordinal(after), ordinal(before)))
    operand('file_bytes', 'file')
    operand('stdout_json', 'stdout')
    fields = [dict(row) for row in abi['publication']['stdout']]
    layout = _json_map(local, ordinal('stdout_json'), len(fields))
    for field in fields:
        role = field['kind']
        if role == 'LITERAL':
            continue
        if role == 'STEP_COUNT':
            field['source_value'] = 'steps'
        else:
            key = 'shape:'+field['key']
            index, fact = _terminal_attributes(local, calls[key].args[0])
            terminal = local.facts.calls[ordinal('terminal')].result
            consumer = local.facts.calls[ordinal(key)].positional[0]
            receiver, argument = local._path(fact.receiver, terminal), local._path(consumer, fact.value)
            field.update(source_value='terminal', shape_value=key)
            if receiver is None or argument is None:
                gaps.append(dict(producer='terminal', consumer=key, reason='FINAL_SHAPE_F4_RECEIVER_ROUTE_OPEN'))
            else:
                edges.append(dict(producer='terminal', consumer=key, kind='FINAL_SHAPE_ACCESS_AND_LIST_ARGUMENT',
                    attribute_fact=encode(local.slicer.fact(local.facts, 'attributes', index)),
                    receiver_path=receiver, argument_path=argument))
        producer = field.get('shape_value', field['source_value'])
        origin = local.facts.calls[ordinal(producer)].result
        packed = _map_field(local, layout, field_slots[field['key']][0], origin)
        if packed is None:
            gaps.append(dict(producer=producer, consumer='stdout_json', reason='JSON_FIELD_VALUE_F4_ROUTE_OPEN'))
        else:
            edges.append(dict(producer=producer, consumer='stdout_json', kind='JSON_FIELD_INPUT', key=field['key'], **packed))
    return anchors, values, edges, gaps, fields, local.body_index


def derive_cpu_fixed_grid_cli_effect_bridge(program_ref, entry_obligations_ref, provider_ref, argv):
    """Prove a same-source normal-completion observer scope without target runs."""
    _need(provider_ref['schema'] == provider.SCHEMA and type(argv) is list, 'checked provider and exact argv required')
    with replay_transaction():
        program, authority = read_artifact(program_ref), read_artifact(provider_ref)
        _need(authority == provider.check(provider_ref['uri']), 'retained provider differs on cold qualification')
        argv = provider._argv(authority, argv)
        source, project, raw, entry, abi, binding = _source_material(authority, program)
        with BuildSession(Path(program_ref['uri']).parent) as session:
            original = verify_entry_obligations(entry_obligations_ref, program=program_ref, session=session)
            anchors, values, edges, gaps, fields, body = _routes(session, original, source, project, raw, entry, abi)
            order = _control_order(session, original, body, anchors['file'], anchors['stdout'])
    eligible = order['status'] == _ORDER_PROVEN and not gaps
    arguments = {}
    for source_role, abi_role in (('root', 'input_root'), ('steps', 'steps'), ('output', 'output')):
        index = abi['argv'][abi_role]
        arguments[abi_role] = dict(sys_argv_index=index, argument_index=index-1, concrete_argument=argv[index-1],
                                   original_value=source_role, original_call=anchors[source_role]['original_call'])
    context = authority['context']
    return dict(schema=SCHEMA, schema_version=1, program=program_ref, entry_obligations=entry_obligations_ref,
        provider=provider_ref, source=source, source_project=project, entry=entry, argv=argv, target=binding['target'],
        source_contract_sha256=binding['source_contract_sha256'], source_argv=abi['argv'],
        context=dict(cwd=context['cwd'], interpreter=context['interpreter'], pythonpath=context['pythonpath']),
        arguments=arguments, source_domain=abi['domain'], fixed_grid=dict(method=abi['solver']['method'],
            step_size=abi['solver']['step_size'], steps=int(argv[abi['argv']['steps']-1]), inputs=authority['inputs'],
            tensor_bindings=binding['files']),
        effects=dict(file=dict(anchors['file'], role='FINAL_TENSOR_RAW_BYTES', path_value='output', data_value='file_bytes',
                               terminal_value='terminal'),
                     stdout=dict(anchors['stdout'], role='SOURCE_JSON_LINE', data_value='stdout_json', fields=fields)),
        value_bindings=values, value_routes=edges, value_route_frontier=gaps,
        value_identity='SYMBOLIC_CALL_RESULT_PORTS_IN_ONE_ORIGINAL_SOURCE_ACTIVATION',
        normal_completion_order=order, status=('PROVEN_SCOPED_NORMAL_SUCCESS_SOURCE_CLI_EFFECT_BRIDGE'
                                             if eligible else 'SOURCE_CLI_EFFECT_BRIDGE_OPEN'),
        coverage_frontier=['IMPORT_LOADER_INITIALIZATION', 'GENERAL_PYTHON_AND_NATIVE_EFFECTS',
                           'EXCEPTIONS_AND_FAILURE_TIMING', 'PHYSICAL_FILE_IDENTITY_DURABILITY_AND_STDOUT_DRAIN'],
        dynamic_value_identity_proven=False, whole_entry_effect_equivalence_proven=False,
        target_executed=False, transformation_authorized=False)


def verify_cpu_fixed_grid_cli_effect_bridge(reference):
    _need(reference['schema'] == SCHEMA, 'unexpected bridge schema')
    saved = read_artifact(reference)
    expected = derive_cpu_fixed_grid_cli_effect_bridge(
        saved['program'], saved['entry_obligations'], saved['provider'], saved['argv'])
    _need(_canonical(saved) == _canonical(expected), 'saved source CLI effect bridge differs on cold replay')
    return saved
