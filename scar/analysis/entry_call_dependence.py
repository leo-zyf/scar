"""Connect an observed entry call to its original module inputs and consumers."""

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path

from scar.analysis.bounded_entry_calls import SCHEMA as ENTRY_CALLS, _admit_entry_calls
from scar.analysis.build import BuildSession
from scar.analysis.normal_graph import _Emitter, normal_edges
from scar.analysis.replay_transaction import replay_transaction
from scar.analysis.call_inputs import call_operands
from scar.analysis.source_reference_slice import ReferenceSlicer, source_port
from scar.analysis.source_world_flow import STRUCTURAL, _state, _walk
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program import normal_graph as G, normal_graph as F, normal_graph as T
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.binding_ports import ArgumentRole
from scar.ir.record_codec import decode, encode


SCHEMA = 'scar.entry-call-dependence'


def _one(items, message):
    items = tuple(items)
    if len(items) != 1:
        raise ValueError(message)
    return items[0]


def _original_scope_records(facts, scope, table):
    return ((index, getattr(facts, table.value)[index])
        for group in facts.scope_groups if group.owner == scope.owner
        for span in group.tables if span.table is table
        for index in range(span.start, span.stop))


def _index(edges, *, reverse=False):
    index = defaultdict(list)
    for source, target, proof in edges:
        if reverse:
            source, target = target, source
        index[source].append((target, proof))
    return index


def _path(start, end, index):
    pending = deque((start,))
    parents = {start: None}
    while pending and end not in parents:
        source = pending.popleft()
        for target, proof in index.get(source, ()):
            if target not in parents:
                parents[target] = source, proof
                pending.append(target)
    if end not in parents:
        return None
    edges = []
    while end != start:
        end, proof = parents[end]
        edges.append(proof)
    return list(reversed(edges))


class ModuleDependence:
    """Use F4 normal edges plus conservative operation and namespace relations."""

    def __init__(self, source):
        material = source.material
        self.facts = facts = source.facts
        root = facts.root_scope
        self.port = lambda index: source_port(facts, index).port
        self.references = ReferenceSlicer(material.catalog)
        plans = material.closure(root.template)
        edges = tuple(edge for plan in plans.values() for edge in normal_edges(plan, material.interfaces))
        control = defaultdict(list)
        for edge in edges:
            if edge.kind is G.EdgeKind.CONTROL:
                control[edge.source].append(edge.target)
        reachable = _walk(G.Port(root.template, root.region, G.PortKind.ENTER), control)
        owners = {(port.template, port.owner) for port in reachable}

        def inside(port):
            return (port.template, port.owner) in owners

        self.data = [(edge.source, edge.target, dict(kind='F4_DATA', edge=encode(edge)))
            for edge in edges if edge.kind is G.EdgeKind.DATA and inside(edge.source) and inside(edge.target)]
        self.world = [(edge.source, edge.target, dict(kind='F4_WORLD', edge=encode(edge)))
            for edge in edges if edge.kind is G.EdgeKind.STATE and inside(edge.source) and inside(edge.target)]
        self.carries = _index(self.data)

        for key, plan in plans.items():
            emitter = _Emitter(plan, key, material.interfaces)
            for step in plan.steps:
                entry = emitter.port(step.id, G.PortKind.ENTER)
                if entry not in reachable or isinstance(step.instruction, STRUCTURAL):
                    continue
                inputs, world_inputs = [], []
                for ordinal, value in enumerate(step.arguments):
                    sink = world_inputs if _state(emitter.value(value)[1]) else inputs
                    sink.append(emitter.port(step.id, G.PortKind.ARGUMENT, ordinal))
                for completion in step.completions:
                    if completion.kind is not C.FALLTHROUGH or not emitter.allowed(completion, step.region):
                        continue
                    for ordinal, kind in enumerate(completion.types):
                        output = emitter.port(step.id, G.PortKind.RESULT, ordinal, completion)
                        if _state(kind):
                            self.world.extend((port, output, dict(kind='NORMAL_OPERATION_WORLD',
                                operation=encode(entry), before=encode(port), after=encode(output)))
                                for port in world_inputs)
                        else:
                            self.data.extend((port, output, dict(kind='MAY_DEPEND_ON_OPERAND',
                                operation=encode(entry), operand=encode(port), result=encode(output)))
                                for port in inputs)

        writes = defaultdict(list)
        reads = []
        for index, access in _original_scope_records(facts, root, F.FactTable.NAMES):
            port = self.port(access.value)
            if inside(port):
                (writes[access.namespace] if access.write else reads).append((port, access.namespace, index))
        for target, namespace, reader in reads:
            for source_port_, _, writer in writes.get(namespace, ()):
                self.data.append((source_port_, target, dict(kind='DECLARED_MAY_REACHING_BINDING',
                    location=encode(namespace), value=encode(source_port_), use=encode(target),
                    write=self.fact('names', writer), read=self.fact('names', reader))))
        self.reverse_data = _index(self.data, reverse=True)
        self.state = _index(self.world)
        self.calls = {self.port(row.result): (index, row)
            for index, row in _original_scope_records(facts, root, F.FactTable.CALLS)
            if inside(self.port(row.result))}
        material.validate_parents()

    def fact(self, table, index):
        return encode(self.references.fact(self.facts, table, index))

    def provenance(self, actual, observed):
        pending = deque((actual,))
        seen = {actual}
        used = []
        while pending:
            target = pending.popleft()
            for source, proof in self.reverse_data.get(target, ()):
                used.append(proof)
                if source not in seen:
                    seen.add(source)
                    pending.append(source)
        providers = [
            dict(value=encode(port), call_fact=self.fact('calls', self.calls[port][0]), observations=observations)
            for port, observations in observed.items() if port in seen
        ]
        return dict(relations=used, observed_calls=providers, current_content_version=None)


def _module_observations(report, module_code, flow):
    observed = defaultdict(list)
    for index, row in enumerate(report['observed_targets']):
        record = row['observation']
        if not row['entry_scope_reachable'] or record['caller_site']['code'] != module_code:
            continue
        endpoint = decode(T.GlobalPort, record['source_call'])
        if endpoint.unit == flow.facts.unit and endpoint.port in flow.calls:
            observed[endpoint.port].append(f'#/observed_targets/{index}')
    return observed


def _project(view, report, interface_index):
    if type(interface_index) is not int or not 0 <= interface_index < len(report['interfaces']):
        raise ValueError('interface index is outside the original entry projection')
    selected = report['interfaces'][interface_index]
    link = decode(T.CallLink, selected['interface']['link'])
    if (selected['caller_scope'] != report['entry_scope'] or not selected['ordinary_body_entry']
            or link.binding is not T.ArgumentBinding.DIRECT):
        raise ValueError('expected a directly bound normal call in the original module entry')
    attempt = _one((row for row in report['observed_targets'] if row['entry_scope_reachable']
        and (row['provider'], row['site_id']) == (selected['provider'], selected['site_id'])
        and row['observation']['evidence'] == selected['frame']['selection']),
        'expected the exact observed attempt for the selected interface')
    flow = ModuleDependence(view.unit(link.result.unit))
    observed = _module_observations(report, attempt['observation']['caller_site']['code'], flow)
    index, call = flow.calls[link.result.port]
    actuals = []
    shared_relations = []
    relation_ids = {}
    for route in link.parameters:
        if route.kind is not T.RouteKind.ARGUMENT:
            continue
        actual, = route.actuals
        if type(actual) is not T.GlobalPort or actual.unit != flow.facts.unit:
            raise ValueError('selected direct argument lacks a local original actual port')
        provenance = flow.provenance(actual.port, observed)
        ids = []
        for relation in provenance.pop('relations'):
            key = json.dumps(relation, sort_keys=True, separators=(',', ':'))
            if key not in relation_ids:
                relation_ids[key] = len(shared_relations)
                shared_relations.append(relation)
            ids.append(relation_ids[key])
        actuals.append(dict(original_route=encode(route), dependency_rows=ids, **provenance))

    consumers = []
    for result, attempts in observed.items():
        if result == link.result.port:
            continue
        other_index, other = flow.calls[result]
        for operand in call_operands(view, flow.facts.unit, other_index):
            if operand.role not in (ArgumentRole.POSITIONAL, ArgumentRole.KEYWORD):
                continue
            slot = operand.ordinal if operand.role is ArgumentRole.POSITIONAL else operand.keyword
            actual = view.original(operand.edge.source)
            path = _path(link.result.port, actual, flow.carries)
            if path is not None:
                before, after = flow.port(other.world), flow.port(other.result_world)
                consumers.append(dict(observations=attempts, call_fact=flow.fact('calls', other_index),
                    role=operand.role.value, slot=slot, input=encode(actual), data_route=path,
                    world_before=encode(before), world_after=encode(after),
                    world_route=_path(flow.port(call.result_world), before, flow.state),
                    effect_discharge_proven=False))
    return dict(entry_scope=report['entry_scope'], original_interface=f'#/interfaces/{interface_index}',
        original_call_fact=flow.fact('calls', index), operation=attempt['operation'],
        selected_body=encode(link.body), call_conditions=link.conditions, state_and_return_routes=encode(link.returns),
        original_world_in=encode(flow.port(call.world)), original_world_out=encode(flow.port(call.result_world)),
        live_ins=actuals, dependencies=shared_relations, return_consumers=consumers,
        qualification=dict(source_module_only=True, observed_targets_are_attempts=True,
            lexical_binding_may_reach_only=True, content_effects_discharged=False,
            target_executed_during_projection=False, first_stage_complete=False))


@replay_transaction()
def project(entry_calls, interface_index):
    reference = artifact_reference(entry_calls, ENTRY_CALLS)
    report = read_artifact(reference)
    program = report['program']
    read_artifact(program)
    session = BuildSession(Path(program['uri']).parent)
    with _admit_entry_calls(session, report):
        result = _project(session._f4.view, report, interface_index)
    read_artifact(reference, json_document=False)
    return dict(schema=SCHEMA, schema_version=1, entry_calls=reference, **result)


def check(path):
    receipt = read_artifact(artifact_reference(path, SCHEMA))
    rebuilt = project(receipt['entry_calls']['uri'], int(receipt['original_interface'].rsplit('/', 1)[-1]))
    if json.dumps(receipt, sort_keys=True) != json.dumps(rebuilt, sort_keys=True):
        raise ValueError('entry dependence differs from the original static and observed inputs')
    return rebuilt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry-calls', type=Path)
    parser.add_argument('--interface', type=int)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args(argv)
    if args.check:
        result = check(args.check)
    else:
        if args.entry_calls is None or args.interface is None or args.out is None:
            parser.error('provide entry calls, an interface index, and output')
        result = project(args.entry_calls, args.interface)
        with args.out.open('x') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    print(json.dumps({key: len(result[key]) for key in ('live_ins', 'dependencies', 'return_consumers')}))


if __name__ == '__main__':
    main()
