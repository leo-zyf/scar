"""Connect one admitted CPU Tensor observation to original F4 source routes."""

from collections import defaultdict, deque
from hashlib import sha256
import json
from pathlib import Path

from scar.analysis.build import BuildSession
from scar.analysis.entry_obligations import verify_entry_obligations
from scar.analysis.nvhpc_source_effects import TENSOR_FLOW, verify_nvhpc_tensor_flow
from scar.analysis.call_inputs import call_operands
from scar.analysis.source_reference_slice import ReferenceSlicer, source_port
from scar.analysis.source_roles import SyntaxRoleView
from scar.analysis.source_template_links import _json
from scar.analysis.static_query import _Fragments
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program.normal_links import AttributeAction, FactTable
from scar.ir.program.binding_ports import ArgumentRole
from scar.ir.program.source_syntax import SourceSyntaxMaterial
from scar.ir.record_codec import decode, encode


SCHEMA = 'scar.f4-entry-tensor-flow-join'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _only(rows, message):
    rows = tuple(rows)
    _need(len(rows) == 1, message)
    return rows[0]


def _pin_scope(program, scope, entry_obligations, process, contract):
    target = program['target']
    expected_target = dict(uri=target['path'], sha256=target['sha256'], size_bytes=target['bytes'],
                           schema='scar.target-hardware-graph')
    read_artifact(expected_target)
    context = process['context']
    environment = sha256(json.dumps(sorted(context['environment'].items()), separators=(',', ':')).encode()).hexdigest()
    source = contract['source']
    _need(program['entry'] == source['path'] == scope['entry'] and source['sha256'] == scope['entry_sha256']
          and process['argv'] == scope['argv'] and process['scope'] == scope['boundary']
          and context['python']['path'] == scope['python_executable']
          and context['working_directory'] == scope['working_directory']
          and environment == scope['environment_sha256'] and scope['target'] == expected_target
          and scope['target_sha256'] == expected_target['sha256']
          and entry_obligations['sha256'] == scope['entry_obligations_sha256']
          and set(context['cpu_ids']) <= set(scope['process_affinity_cpu_ids']),
          'observed Tensor flow belongs to a different original source or process scope')
    artifact = process['artifact']
    project = read_artifact(artifact)

    def member(name):
        path = Path(artifact['uri']).parent/'plan'/name
        raw = path.read_bytes()
        _need(project['files']['plan/'+name] == dict(sha256=sha256(raw).hexdigest(), size_bytes=len(raw)),
              'Tensor flow physical source evidence changed')
        return json.loads(raw)

    _need(member('problem.json')['target'] == expected_target
          and member('execution.json')['cpu_ids'] == context['cpu_ids'],
          'Tensor flow must retain its original physical target and observed CPU placement')
    return expected_target


def _ordinals(facts, owner, table):
    return tuple(index for group in facts.scope_groups if group.owner == owner
        for span in group.tables if span.table is table for index in range(span.start, span.stop))


class _SourceRoutes:
    def __init__(self, session, obligations, contract):
        self.view = session.view
        self.unit = session.report['f4']['entry_unit']
        catalog = self.view.catalog
        self.facts = catalog.fact_store.load(self.unit)
        self.document = obligations
        unit_index = _only((i for i, row in enumerate(obligations['units']) if row['unit'] == self.unit),
                           'original entry unit is missing or ambiguous')
        self.unit_index = unit_index
        model = catalog.fact_store.model_units[self.unit]
        syntax = decode(SourceSyntaxMaterial, _Fragments(catalog.fact_store.model_for(self.unit)).read(
            model['artifacts']['source_syntax']))
        _need(syntax.input.path == contract['source']['path'] and syntax.input.raw_sha256 == contract['source']['sha256'],
              'F4 input syntax differs from the observed code source')
        self.roles = SyntaxRoleView(syntax)
        self.attributes = {field.name: field.attributes for field in syntax.profile.ast_schema}
        wanted = contract['source']['function']
        function = _only((f for f in self.facts.functions
            if syntax.nodes[f.template.occurrence].constructor in ('FunctionDef', 'AsyncFunctionDef')
            and self.roles.field(f.template.occurrence, 'name').scalar.to_python() == wanted),
            'observed exact source must have one original F4 function declaration')
        region = self.view.source_region(self.unit, function.template, function.body)
        self.body_index = _only((i for i, row in enumerate(obligations['bodies'])
            if row['unit'] == unit_index and row['point']['node']['wire'] == region.node.wire and row['status'] == 'PROJECTED'),
            'observed source function was not admitted as an original F4 body')
        body = obligations['bodies'][self.body_index]
        self.operations = {obligations['operations'][i]['operation']: i for i in body['operations']}
        self.calls = defaultdict(list)
        for ordinal in _ordinals(self.facts, function.body_scope.owner, FactTable.CALLS):
            occurrence = self.port(self.facts.calls[ordinal].result).port.template.occurrence
            syntax_node = syntax.nodes[occurrence]
            attrs = dict(zip(self.attributes[syntax_node.constructor], syntax_node.attributes, strict=True))
            span = tuple(attrs[name].scalar.to_python() for name in ('lineno', 'end_lineno', 'col_offset', 'end_col_offset'))
            _need(syntax_node.constructor == 'Call', 'admitted call does not reference original F2 Call syntax')
            self.calls[span].append(ordinal)
        self.attrs = _ordinals(self.facts, function.body_scope.owner, FactTable.ATTRIBUTES)
        self.operands = {}
        self.slicer = ReferenceSlicer(catalog)
        # Restrict reference queries to this original unit. No cross-call
        # source target or provider effects are inferred by the local join.
        self.slicer.indexed.add(self.unit)
        self.port_index = {}
        for ordinal, row in enumerate(obligations['ports']):
            owner = obligations['owners'][row['owner']]
            template = obligations['templates'][owner['template']]
            if template['unit'] == unit_index:
                self.port_index[_json(dict(row, owner=owner['original'], template=template['template']))] = ordinal
        self.edges = defaultdict(list)
        for ordinal in body['edges']:
            edge = obligations['edges'][ordinal]
            if edge['kind'] == 'DATA':
                self.edges[edge['source'], edge['target']].append(ordinal)

    def port(self, index):
        return source_port(self.facts, index)

    def graph_port(self, port):
        row = encode(port)
        row['owner'] = port.owner.wire
        return self.port_index.get(_json(row))

    def call(self, observation):
        ordinal = _only(self.calls[tuple(observation['site']['position'])],
                        'observed exact CodeID CALL does not have one source call fact')
        result = self.port(self.facts.calls[ordinal].result)
        point = self.view.source_port(result)
        owner = self.view.get(point).owner
        _need(owner.node.wire in self.operations, 'observed CALL is absent from the admitted original F4 body')
        operation = self.operations[owner.node.wire]
        result_port = self.graph_port(result.port)
        _need(result_port is not None, 'observed CALL lacks its original retained F4 result port')
        return dict(operation=owner.node.wire, operation_pointer=f'#/operations/{operation}',
            call_fact=encode(self.slicer.fact(self.facts, 'calls', ordinal)), call_ordinal=ordinal,
            result_port=result_port)

    def _path(self, destination, origin):
        origin = self.port(origin)
        slice_ = self.slicer.slice(self.port(destination), max_states=512)
        pending, parents = deque((0,)), {0: None}
        successors = defaultdict(list)
        for edge in slice_.transitions:
            successors[edge.downstream].append(edge)
        found = None
        while pending:
            index = pending.popleft()
            if slice_.states[index].port == origin:
                found = index
                break
            for edge in successors[index]:
                if edge.upstream not in parents:
                    pending.append(edge.upstream)
                    parents[edge.upstream] = edge
        if found is None:
            return None
        route = []
        while (edge := parents[found]) is not None:
            source = self.graph_port(slice_.states[edge.upstream].port.port)
            target = self.graph_port(slice_.states[edge.downstream].port.port)
            _need(source is not None and target is not None, 'declared source route escaped its admitted original F4 body')
            route.append(dict(kind=edge.kind.value, source=source, target=target, evidence=encode(edge.evidence),
                original_data_edges=self.edges.get((source, target), []), conditions=encode(edge.conditions)))
            found = edge.downstream
        return route

    def positional_route(self, consumer, producer, operand):
        call = self.facts.calls[consumer]
        _need(call.positional_pack is None and call.keyword_pack is None and operand < len(call.positional),
              'observed original operation does not have an explicit positional operand')
        if consumer not in self.operands:
            self.operands[consumer] = call_operands(self.view, self.unit, consumer)
        declared = _only((row for row in self.operands[consumer]
            if row.role is ArgumentRole.POSITIONAL and row.ordinal == operand), 'original call operand edge is absent')
        source = self.graph_port(self.view.original(declared.edge.source))
        target = self.graph_port(self.view.original(declared.edge.target))
        edge = _only(self.edges.get((source, target), ()), 'original F4 value-to-argument edge is absent or ambiguous')
        route = self._path(call.positional[operand], self.facts.calls[producer].result)
        _need(route is not None, 'observed operand lacks a declared source reference route')
        return dict(operand_edge=f'#/edges/{edge}', source_reference_path=route, receiver=None)

    def receiver_route(self, consumer, producer):
        call = self.facts.calls[consumer]
        callee = self.port(call.result).port.template.occurrence
        callable_node = self.roles.field(callee, 'func').reference
        matches = []
        for index in self.attrs:
            row = self.facts.attributes[index]
            if row.action is AttributeAction.GET and self.port(row.value).port.template.occurrence == callable_node:
                path = self._path(call.callee, row.value)
                if path is not None:
                    matches.append((index, row, path))
        index, attribute, path = _only(matches, 'observed descriptor call lacks a unique original receiver declaration')
        route = self._path(attribute.receiver, self.facts.calls[producer].result)
        _need(route is not None, 'observed receiver lacks a declared source reference route')
        receiver = dict(attribute_fact=encode(self.slicer.fact(self.facts, 'attributes', index)), name=attribute.name,
            callable_path=path)
        return dict(operand_edge=None, source_reference_path=route, receiver=receiver)


def bind_source_tensor_flow(program_ref, scope, flow_ref, *, entry_obligations):
    """Independently replay the original F4 graph and admitted Tensor observation."""
    flow = read_artifact(flow_ref)
    _need(flow_ref['schema'] == TENSOR_FLOW and scope['program_sha256'] == program_ref['sha256'],
          'expected exact original F4 and bounded Tensor flow evidence')
    flow = verify_nvhpc_tensor_flow(flow_ref, process=flow['process'])
    process = read_artifact(flow['process'])
    contract = read_artifact(flow['source_contract'])
    program = read_artifact(program_ref)
    target = _pin_scope(program, scope, entry_obligations, process, contract)
    raw = read_artifact(flow['capture'])
    projection = flow['projection']
    with BuildSession(Path(program_ref['uri']).parent) as session:
        obligations = verify_entry_obligations(entry_obligations, program=program_ref, session=session)
        source = _SourceRoutes(session, obligations, contract)
        calls = [dict(source.call(row), dispatch_pointer=f'#/dispatches/{index}', site=row['site'],
                      operator=row['operator']) for index, row in enumerate(raw['dispatches'])]
        nodes = {row['source']: row for row in projection['source_nodes']}
        routes = []
        for row in nodes.values():
            if row['relation'] == 'SAME_WEAK_STORAGE_LIFETIME':
                load, view = row['dispatches']
                route = source.receiver_route(calls[view]['call_ordinal'], calls[load]['call_ordinal'])
                routes.append(dict(producer=load, consumer=view, source=row['source'], relation=row['relation'], **route))
        for row in projection['observed_data_dependencies']:
            producer = nodes[row['source']]['dispatches'][-1]
            consumer = row['observation']
            before, after = (calls[index]['call_ordinal'] for index in (producer, consumer))
            if nodes[row['target']]['relation'] == 'PUBLICATION_ALIAS':
                route = source.receiver_route(after, before)
            else:
                route = source.positional_route(after, before, row['operand'])
            routes.append(dict(producer=producer, consumer=consumer, source=row['source'], target=row['target'],
                               object_token=row['object'], relation='OBSERVED_OPERAND_WITH_DECLARED_REFERENCE_ROUTE', **route))
    return dict(schema=SCHEMA, schema_version=1, program=program_ref, entry_obligations=entry_obligations,
        flow=flow_ref, target=target, observed_cpu_ids=process['context']['cpu_ids'],
        source_function_pointer=f'#/bodies/{source.body_index}', source_calls=calls, observed_routes=routes,
        status='BOUNDED_OBSERVED_TENSOR_ROUTES_BOUND_TO_ORIGINAL_F4',
        static_may_routes_promoted=False, content_versions_proven=False,
        cross_run_identity_established=False, global_effect_closure=False, deployment_authorized=False)
