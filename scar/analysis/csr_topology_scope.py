"""Replay the scope of a reusable CSR topology without inferring content identity."""

from collections import defaultdict, deque
import json
from pathlib import Path

from scar.analysis.bounded_entry_calls import _admit_entry_calls
from scar.analysis.build import BuildSession
from scar.analysis.csr_row_region import FAMILY, recognize
from scar.analysis.source_body_interfaces import BodyInterfaceResolver
from scar.analysis.source_control import loop_dependencies
from scar.analysis.source_reference_slice import source_port
from scar.analysis.source_value_flow import project_value_flow
from scar.ir.execution_recipe import read_artifact
from scar.ir.program import normal_graph as G, normal_graph as T, source_productions as P
from scar.ir.program.model import ValueKind, ValueType
from scar.ir.record_codec import decode, encode


SCHEMA = 'scar.csr-topology-scope'
SOURCE_SCOPE = 'scar.study-verified-source-loop-scope'
REGION_PROOF = 'scar.csr-row-region-proof'


def _only(rows, reason):
    rows = tuple(rows)
    if len(rows) != 1:
        raise ValueError(reason)
    return rows[0]


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _at(report, pointer, collection):
    prefix = '#/' + collection + '/'
    _require(type(pointer) is str and pointer.startswith(prefix) and pointer[len(prefix):].isdecimal(),
        'expected an exact original entry projection pointer')
    return report[collection][int(pointer[len(prefix):])]


def _body(view, body):
    source = view.unit(body.unit)
    function = _only((fn for fn in source.facts.functions if (fn.template, fn.body) == (body.template, body.region)),
        'shared CSR scope does not identify one original function')
    flow = project_value_flow(source.material, source_port(source.facts, function.value), body)
    inputs = dict(zip(function.parameter_names,
        (source_port(source.facts, i).port for i in function.formals[1:-1]), strict=True))
    return source, function, flow, inputs


def _direct_interfaces(report):
    observations = defaultdict(list)
    for index, row in enumerate(report['observed_targets']):
        if row['entry_scope_reachable']:
            observations[row['provider'], row['site_id']].append((index, row))
    for index, row in enumerate(report['interfaces']):
        if not row['ordinary_body_entry']:
            continue
        link = decode(T.CallLink, row['interface']['link'])
        if link.binding is not T.ArgumentBinding.DIRECT:
            continue
        for observed_index, observed in observations[row['provider'], row['site_id']]:
            if row['frame']['selection'] == observed['observation']['evidence']:
                yield index, observed_index, link, observed


class _Transport:
    """Original typed reference transport inside one shared source body."""

    def __init__(self, flow):
        self.unit = flow.body.unit
        self.incoming = defaultdict(list)
        for edge in flow.edges:
            if edge.kind is G.EdgeKind.DATA and edge.transition is not G.Transition.BACKEDGE:
                self.incoming[edge.target].append(edge)
        self.reads = [(loc, read) for loc in flow.locations for read in loc.reads]
        self.ancestors = {}

    def path(self, start, finish):
        if finish not in self.ancestors:
            pending = deque((finish,))
            seen = {finish: None}
            while pending:
                for edge in self.incoming[pending.popleft()]:
                    if edge.source not in seen:
                        seen[edge.source] = edge
                        pending.append(edge.source)
            self.ancestors[finish] = seen
        seen = self.ancestors[finish]
        if start not in seen:
            return None
        result = []
        while start != finish:
            edge = seen[start]
            result.append(edge)
            start = edge.target
        return result

    def binding(self, target):
        hits = ((loc, read, path) for loc, read in self.reads
            if (path := self.path(read.value, target)) is not None)
        loc, read, path = _only(hits, 'CSR argument has an ambiguous or missing original lexical read')
        write = _only(loc.writes, 'CSR argument binding has multiple or missing original writers')
        return loc, write, dict(namespace=encode(loc.namespace), original_definition=encode(write),
            original_read=encode(read), original_transport=encode(path))


def _route(link, formal):
    return _only((route for route in link.parameters if route.formal.port == formal and len(route.actuals) == 1),
        'CSR role lacks one exact original argument route')


def _loop_live_in(transport, actual_route, caller_inputs, ingress):
    actual, = actual_route.actuals
    _require(actual_route.kind is T.RouteKind.ARGUMENT and actual.unit == transport.unit,
        'CSR topology operand is not a direct argument of the selected caller')
    _, write, binding = transport.binding(actual.port)
    _require(write.value in caller_inputs.values(), 'CSR topology operand is not a parameter defined outside the loop')
    return dict(role='SINGLE_ORIGINAL_ENTRY_FORMAL_BINDING', formal_ingress=encode(_route(ingress, write.value)),
        region_argument_route=encode(actual_route), caller_binding=binding,
        physical_contents_proven_invariant=False)


def _data(ports):
    return tuple(row.port for row in ports if not (
        type(row.value_type) is ValueType and row.value_type.kind is ValueKind.STATE))


def _unpack(transport, flow, region_call, consumer):
    def fixed(row):
        return (type(row.instruction) is P.ApplyPythonProtocol
            and row.instruction.protocol is P.PythonProtocolKind.UNPACK_FIXED)
    operation, feed = _only(((row, path) for row in flow.operations if fixed(row)
        and len(_data(row.inputs)) == 1
        and (path := transport.path(region_call.result.port, _data(row.inputs)[0])) is not None),
        'CSR call result has no unique original fixed unpack')
    routes = []
    for route in consumer.parameters:
        if route.kind is not T.RouteKind.ARGUMENT or len(route.actuals) != 1:
            continue
        actual, = route.actuals
        _, write, binding = transport.binding(actual.port)
        matches = [(i, path) for i, output in enumerate(_data(operation.outputs))
            if (path := transport.path(output, write.value)) is not None]
        if matches:
            index, path = _only(matches, 'consumer argument has multiple CSR tuple-item producers')
            routes.append(dict(item=index, unpack_to_store=encode(path), caller_binding=binding,
                consumer_route=encode(route)))
    _require(len(routes) == 2 and sorted(row['item'] for row in routes) == [0, 1],
        'the consumer must directly use both original unpacked outputs')
    return dict(original_unpack=encode(operation.entry), input_transport=encode(feed), item_routes=routes)


def _copy_return(copy_symbol, flow, function, source):
    transport = _Transport(flow)
    result = source_port(source.facts, function.returns).port
    tuples = (row for row in flow.operations
        if type(row.instruction) is P.ConstructValue and row.instruction.construction is P.ConstructionKind.TUPLE
        and len(_data(row.inputs)) == 2 and len(_data(row.outputs)) == 1)
    matches = []
    for row in tuples:
        returned = transport.path(_data(row.outputs)[0], result)
        if returned is None:
            continue
        loc, _, binding = transport.binding(_data(row.inputs)[0])
        if loc.namespace.name == copy_symbol:
            matches.append(dict(original_tuple=encode(row.entry), return_transport=encode(returned),
                private_matrix_item=0, private_matrix_binding=binding))
    return _only(matches, 'private CSR copy does not identify one original returned tuple item')


def _literal_default(view, endpoint):
    context = BodyInterfaceResolver(view).plan(endpoint.unit, endpoint.port.template)
    region = context.regions[context.plan.root]
    construct, complete = (context.steps[step] for step in region.steps)
    _require(endpoint.port.owner == region.id and endpoint.port.kind is G.PortKind.RESULT
        and endpoint.port.ordinal == 0 and type(construct.instruction) is P.ConstructValue
        and construct.instruction.construction is P.ConstructionKind.LITERAL
        and type(complete.instruction) is P.TransferCompletion
        and context.value(complete.arguments[0]) == context.result(construct),
        'CSR overwrite default does not return its literal directly')
    facts = view.unit(endpoint.unit).facts
    return _only((row.literal for row in facts.constants if source_port(facts, row.value) == context.result(construct)),
        'CSR overwrite default lacks one exact original constant fact').to_python()


def _guard_contract():
    return dict(required_hit_key=['shape', 'nnz', 'index_dtype_and_width', 'row_dtype_and_width',
        'selected_row_count', 'indptr_bytes', 'indices_bytes', 'selected_rows_bytes'],
        byte_comparison='EXACT_CURRENT_BYTES_AGAINST_OWNED_SNAPSHOT_BEFORE_WRITES',
        lookup_lifetime='ONE_EXCLUSIVE_ENTRY_SCOPE', retained_topology_plans_max=1,
        build_point='FIRST_ELIGIBLE_REGION_CALL_WITHIN_SCOPE',
        key_excludes=['data_payload_bytes', 'diagonal_scalar', 'rhs', 'object_identity'],
        preserve_each_call=['original_copy_allocation', 'original_rhs', 'consumer_and_world_order',
            'regular_provider_and_layout_guards', 'private_output_and_no_alias_checks',
            'data_diagonal_and_current_diag_guards'],
        mismatch='ORIGINAL_EXECUTION_BEFORE_ANY_WRITE',
        concurrent_mutation_during_comparison_or_apply='REQUIRES_EXCLUSIVE_INPUT_CONTRACT')


def _project(session, report, scope, proof):
    view = session.view
    interfaces = tuple(_direct_interfaces(report))
    identity = scope['f4_call']
    region_index, observed_index, region, observation = _only((row for row in interfaces
        if row[3]['operation'] == identity['operation'] and f'#/interfaces/{row[0]}' == identity['interface']
        and f'#/observed_targets/{row[1]}' == identity['observation']),
        'CSR call does not match the original bounded direct interface')
    observed_code = observation['observation']['target_code']
    _require((observed_code['filename'], observed_code['key']) ==
        (identity['target']['path'], identity['target']['code_key']), 'CSR observed callable differs from the proof')
    body = decode(T.Body, scope['f4_loop']['source_body'])
    caller_source, caller_fn, caller_flow, caller_inputs = _body(view, body)
    ingress_index, _, ingress, _ = _only((row for row in interfaces if row[2].body == body),
        'CSR loop has no unique original admitted ingress')
    loop = _only((row for row in loop_dependencies(caller_source.material, caller_fn, flow=caller_flow)['loops']
        if row['header'] == scope['f4_loop']['header']), 'original CSR shared loop changed')
    selected = {decode(T.GlobalPort, row['result']) for row in scope['f4_loop']['normal_body_calls'].values()}
    enclosed = {decode(T.GlobalPort, row['result']):row for row in loop['body_calls']}
    _require(region.result in selected <= set(enclosed) and loop['control_backedges'] == scope['f4_loop']['control_backedges'],
        'CSR call or original loop recurrence differs from the retained bounded scope')
    consumer_index, consumer_observation, consumer, _ = _only((row for row in interfaces
        if row[2].result in selected - {region.result}), 'CSR output has no unique admitted consumer call')
    region_source, region_fn, region_flow, region_inputs = _body(view, region.body)
    caller = _Transport(caller_flow)
    roles = proof['region']['inputs']
    live_ins = {role: _loop_live_in(caller, _route(region, region_inputs[roles[key]]), caller_inputs, ingress)
        for role, key in (('matrix', 'A'), ('selected_rows', 'R'))}
    overwrite = _route(region, region_inputs[roles['O']])
    _require(overwrite.kind is T.RouteKind.INITIAL_DEFAULT, 'CSR out-of-place study requires the original overwrite default')
    default, = overwrite.actuals
    _require(_literal_default(view, default) is False, 'CSR overwrite default must be False')
    call_details = _unpack(caller, caller_flow, region, consumer)
    copy_path = _copy_return(roles['C'], region_flow, region_fn, region_source)
    scope_port = decode(T.GlobalPort, loop['header'])
    original_loop_point = caller_source._node(scope_port.port.template, scope_port.port.owner)
    _require(encode(original_loop_point) == scope['f4_loop']['operation'], 'CSR loop operation differs from original F4')
    def worlds(link):
        row = enclosed[link.result]
        return dict(before=row['world_before'], after=row['world_after'])
    return dict(loop=dict(header=loop['header'], original_operation=scope['f4_loop']['operation'],
        body=loop['body'], control_backedges=loop['control_backedges'], ingress=f'#/interfaces/{ingress_index}',
        exact_count_under_source_conditions=scope['f4_loop']['static_exact_condition']),
        live_ins=live_ins,
        copies_and_consumers=dict(original_copy=proof['source_binding']['copy'],
            original_copy_return=copy_path, default_overwrite_route=encode(overwrite),
            private_copy_requires_provider_and_disjointness=True,
            region_interface=f'#/interfaces/{region_index}', region_attempt=f'#/observed_targets/{observed_index}',
            consumer_interface=f'#/interfaces/{consumer_index}', consumer_attempt=f'#/observed_targets/{consumer_observation}',
            original_return_routes=encode(region.returns), original_unpack_and_consumer=call_details,
            original_call_worlds=dict(region=worlds(region), consumer=worlds(consumer)),
            loop_control_port=encode(view.source_port(scope_port)),
            preserve_original_copy_rhs_and_consumer_schedule=True))


def derive_csr_topology_scope(source_scope, region_proof):
    """Replay original CSR input, copy, consumer and loop routes for one bounded case."""
    _require(source_scope.get('schema') == SOURCE_SCOPE and region_proof.get('schema') == REGION_PROOF,
        'expected original source scope and CSR region proof artifacts')
    scope, proof = read_artifact(source_scope), read_artifact(region_proof)
    identity = scope['f4_call']
    _require(scope['schema_version'] == proof['schema_version'] == 1
        and identity['source_graph'] == proof['source_graph'] and scope['entry']['source'] == proof['entry']
        and identity['target'] == dict(path=proof['source']['uri'], sha256=proof['source']['sha256'],
            code_key=proof['region']['code_key']) and proof['region']['family'] == FAMILY,
        'CSR scope, source graph, source identity or admitted family differs')
    original = read_artifact(proof['source'], json_document=False)
    recovered, _ = recognize(original, proof['source']['uri'], identity['target']['code_key'])
    _require(recovered == proof['region'], 'CSR region pattern differs from its original source')
    report = read_artifact(identity['source_graph'])
    _require(read_artifact(report['program'])['entry'] == proof['entry']['uri'], 'original entry differs')
    _at(report, identity['interface'], 'interfaces')
    _at(report, identity['observation'], 'observed_targets')
    read_artifact(proof['entry'], json_document=False)
    count = read_artifact(scope['observed']['exact_count_evidence'])
    _require(count['entry'] == proof['entry'] and count['argv'] == scope['entry']['argv']
        and count['affinity'] == scope['entry']['affinity'] and count['normal_completion'], 'same-entry count evidence differs')
    with BuildSession(Path(report['program']['uri']).parent) as session, _admit_entry_calls(session, report):
        projected = _project(session, report, scope, proof)
    return dict(schema=SCHEMA, schema_version=1, source_scope=source_scope, region_proof=region_proof,
        entry=scope['entry'], source_graph=identity['source_graph'], **projected,
        guard_contract=_guard_contract(),
        frontiers=['CALLER_LOCAL_BINDING_HAS_NO_SECOND_LEXICAL_WRITER_BUT_ARRAY_CONTENTS_MAY_CHANGE',
            'INDIRECT_NATIVE_OR_PYTHON_WRITES_THROUGH_ALIASES_NOT_CLOSED',
            'ORIGINAL_COPY_DISJOINTNESS_IS_PER_CALL_GUARD_NOT_SOURCE_VALUE_IDENTITY',
            'ARRAY_REPLACEMENT_RESIZE_AND_STRUCTURE_CHANGES_REQUIRE_CURRENT_GUARDS',
            'SOURCE_GRAPH_DOES_NOT_PROVE_STABILITY_FROM_KEY_VALIDATION_TO_MUTATION'],
        qualification=dict(layer='SOURCE_ARGUMENT_AND_NORMAL_RETURN_TRANSPORT',
            typed_reference_scope_confirmed=True, physical_contents_invariance_proven=False,
            runtime_cache_installed_proven=False, transformations_authorized=False,
            region_call_count_for_same_case=scope['observed']['region_attempts']))


def check_csr_topology_scope(receipt):
    """Cold replay all source joins and return their independently derived receipt."""
    document = read_artifact(receipt) if 'uri' in receipt else receipt
    _require((document.get('schema'), document.get('schema_version')) == (SCHEMA, 1),
        'expected original CSR topology scope receipt')
    expected = derive_csr_topology_scope(document['source_scope'], document['region_proof'])
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    _require(canonical(expected) == canonical(document), 'CSR topology scope differs from cold source replay')
    return expected
