"""Read normal-path lexical carries whose next value is a call or attribute result."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

from scar.analysis.program_view import ProgramView
from scar.analysis.session_caller_effects import _source
from scar.analysis.source_declared_catalog import read_source_catalog
from scar.analysis.source_loop_dependencies import _enclosed, loop_dependencies
from scar.analysis.source_name_access_anchors import anchor_name_accesses
from scar.analysis.source_name_definitions import reaching_name_definitions
from scar.analysis.source_reference_slice import ReferenceSlicer, source_port
from scar.analysis.source_value_flow import project_value_flow
from scar.compiler.graph_frontend import _metadata
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.program import normal_graph as G, normal_links as F, normal_targets as T
from scar.ir.program import source_productions as P
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.source_name_definitions import NameDefinitionKind as D
from scar.ir.record_codec import decode, encode

SCHEMA = 'scar.source-returned-loop-carry'


def _entry(port):
    return G.Port(port.template, port.owner, G.PortKind.ENTER)


def _ref(path, schema):
    return artifact_reference(Path(path).resolve(strict=True), schema)


class _Projection:
    def __init__(self, view, unit, function, syntax):
        self.original = view.unit(unit)
        self.catalog, self.unit, self.syntax = view.catalog, unit, syntax
        self.facts, self.material = self.original.facts, self.original.material
        self.function = function
        self._definitions = {}
        body = T.Body(unit, function.template, function.body)
        self.flow = project_value_flow(self.material, self.port(function.value), body)
        self.accesses = anchor_name_accesses(self.catalog, unit, scopes=(function.body_scope.owner,),
                                            material=self.material)
        self.names = defaultdict(list)
        for anchor in self.accesses.anchors:
            if anchor.scope == function.body_scope.owner and not anchor.record.write:
                access = self.accesses.accesses[anchor.access]
                self.names[access.value.port].append((anchor.namespace, anchor.access, anchor.fact))
        self.original_steps = {(key, step.id): step for key, plan in self.material.closure(body.template).items()
                               for step in plan.steps}
        self.operations = {item.entry: item for item in self.flow.operations}
        self.control, self.incoming = defaultdict(list), defaultdict(list)
        for edge in self.flow.edges:
            if edge.kind is G.EdgeKind.CONTROL:
                self.control[edge.source].append(edge)
            elif edge.kind is G.EdgeKind.DATA and edge.transition is not G.Transition.BACKEDGE:
                self.incoming[edge.target].append(edge)
        self.calls = {self.port(record.call.result).port: record for record in self.flow.calls}
        slicer = ReferenceSlicer(SimpleNamespace(fact_store=self.catalog.fact_store))
        self.attributes, self.constants = {}, {}
        for group in self.facts.scope_groups:
            if group.owner != function.body_scope.owner:
                continue
            for span in group.tables:
                table = span.table.value
                if table not in ('attributes', 'constants'):
                    continue
                for i in range(span.start, span.stop):
                    row = getattr(self.facts, table)[i]
                    if table == 'attributes' and row.action is F.AttributeAction.GET:
                        self.attributes[self.port(row.value).port] = row, slicer.fact(self.facts, table, i)
                    elif table == 'constants':
                        self.constants[self.port(row.value).port] = row, slicer.fact(self.facts, table, i)

    def gp(self, port):
        return encode(T.GlobalPort(self.unit, port))

    def port(self, ordinal):
        return source_port(self.facts, ordinal)

    def span(self, port):
        _, span, _ = _metadata(self.syntax, port.template.occurrence)
        return [span[key] for key in ('line', 'end_line', 'column', 'end_column')]

    def origin(self, value):
        """Follow original transport until a primitive output or an open join."""
        edges, seen = [], set()
        while value not in seen:
            seen.add(value)
            if value.kind is G.PortKind.RESULT and _entry(value) in self.operations:
                return value, edges, None
            alternatives = self.incoming[value]
            if len(alternatives) != 1:
                reason = 'AMBIGUOUS_DATA_TRANSPORT' if alternatives else 'NO_PRIMITIVE_SOURCE_ORIGIN'
                return value, edges, reason
            edge, = alternatives
            edges.append(edge)
            value = edge.source
        return value, edges, 'RECURSIVE_DATA_TRANSPORT'

    def definitions(self, namespace):
        if namespace not in self._definitions:
            self._definitions[namespace] = reaching_name_definitions(self.catalog, self.flow.source_function,
                namespace, anchor_graph=self.accesses, material=self.material, flow=self.flow)
        return self._definitions[namespace]

    def producer(self, value, *, receiver=True):
        leaf, transport, reason = self.origin(value)
        detail = dict(result=self.gp(leaf), transport=encode(tuple(reversed(transport))))
        if reason is not None:
            return dict(kind='OPEN', reason=reason, **detail)
        detail.update(entry=self.gp(_entry(leaf)), source_span=self.span(leaf))
        if leaf in self.calls:
            return dict(kind='CALL', fact=encode(self.calls[leaf].fact), **detail)
        if leaf in self.constants:
            literal, fact = self.constants[leaf]
            return dict(kind='LITERAL', literal=encode(literal.literal), fact=encode(fact), **detail)
        if leaf in self.attributes:
            fact, reference = self.attributes[leaf]
            operand = self.port(fact.receiver).port
            result = dict(kind='ATTRIBUTE_GET', fact=encode(reference), attribute=fact.name,
                          receiver=self.gp(operand), descriptor_target_proven=False, **detail)
            if receiver:
                source, path, unresolved = self.origin(operand)
                result['receiver_transport'] = encode(tuple(reversed(path)))
                result['receiver_origin'] = self.gp(source)
                reads = self.names.get(source, ())
                if len(reads) == 1 and unresolved is None:
                    namespace, access, evidence = reads[0]
                    graph = self.definitions(namespace)
                    ids = {i for read in graph.reads if read.access == access for i in read.definitions}
                    result['receiver_read'] = dict(namespace=encode(namespace), entry=self.gp(_entry(source)),
                                                   fact=encode(evidence))
                    result['receiver_definitions'] = self.definition_set(graph, ids, expand_receiver=False)
                else:
                    result['receiver_producer'] = self.producer(operand, receiver=False)
            return result
        return dict(kind='OTHER_PRIMITIVE', instruction=type(self.operations[_entry(leaf)].instruction).__name__,
                    **detail)

    def definition_set(self, graph, ids, *, expand_receiver=True):
        definitions, open_kinds, opaque = [], set(), 0
        for i in sorted(ids):
            item = graph.definitions[i]
            if item.kind in (D.FORMAL, D.WRITE):
                definitions.append(dict(kind=item.kind.value, site=self.gp(item.site), value=encode(item.value),
                                        facts=encode(item.evidence), conditions=list(item.conditions),
                                        producer=self.producer(item.value.port, receiver=expand_receiver)))
            else:
                open_kinds.add(item.kind.value)
                opaque += item.kind is D.OPAQUE_EFFECT
        return dict(known_definitions=definitions, open_definition_kinds=sorted(open_kinds),
                    opaque_effect_alternatives=opaque)

    def consumers(self, namespace, members):
        for result, record in self.calls.items():
            entry = _entry(result)
            if entry not in members:
                continue
            call = record.call
            arguments = [('POSITIONAL', i, None, value) for i, value in enumerate(call.positional)]
            arguments.extend(('KEYWORD', i, name, value) for i, (name, value) in enumerate(call.keywords))
            for kind, i, name, value in arguments:
                operand = self.port(value).port
                leaf, transport, reason = self.origin(operand)
                matches = [fact for owner, _, fact in self.names.get(leaf, ()) if owner == namespace]
                if reason is None and len(matches) == 1:
                    yield dict(entry=self.gp(entry), result=self.gp(result), fact=encode(record.fact),
                               source_span=self.span(entry), argument=dict(kind=kind, ordinal=i,
                                   name=name, port=self.gp(operand)), read=dict(entry=self.gp(_entry(leaf)),
                                   value=self.gp(leaf), fact=encode(matches[0])),
                               transport=encode(tuple(reversed(transport))))

    def loops(self):
        for loop in loop_dependencies(self.material, self.function, flow=self.flow)['loops']:
            header = decode(T.GlobalPort, loop['header']).port
            control = self.original_steps[header.template, header.owner].instruction
            if type(control.condition) is not P.IterationCondition:
                continue
            body = G.Port(header.template, control.body, G.PortKind.ENTER)
            terminal = G.Port(header.template, header.owner, G.PortKind.EXIT, completion=C.FALLTHROUGH)
            members = _enclosed(body, self.control, {header, terminal})
            carries = []
            for binding in loop['lexical_bindings']:
                if not binding['reads'] or not binding['writes']:
                    continue
                namespace = decode(F.Namespace, binding['namespace'])
                sources = [self.producer(decode(G.Port, item['value'])) for item in binding['writes']]
                if not any(item['kind'] in ('ATTRIBUTE_GET', 'CALL') for item in sources):
                    continue
                consumers = list(self.consumers(namespace, members))
                if not consumers:
                    continue
                graph = self.definitions(namespace)
                routes = dict(initial=[], backedges=[], normal_exits=[])
                for transition in graph.transitions:
                    edge = graph.control_edges[transition.edge]
                    if edge.target == header:
                        route = 'backedges' if edge.transition is G.Transition.BACKEDGE else 'initial'
                    elif edge.target == terminal:
                        route = 'normal_exits'
                    else:
                        continue
                    predecessor = graph.states[transition.source]
                    successor = graph.states[transition.target]
                    routes[route].append(dict(edge=encode(edge), stack=encode(successor.stack),
                                              **self.definition_set(graph, predecessor.outgoing)))
                if not all(routes.values()):
                    continue
                carries.append(dict(namespace=encode(namespace), **routes, consumers=consumers,
                                    control_and_dataflow_traversed=graph.traversal_complete and graph.dataflow_complete,
                                    definition_frontiers=sorted({item.reason for item in graph.frontiers})))
            if carries:
                yield dict(header=loop['header'], body=loop['body'], condition=loop['condition'],
                           source_span=self.span(header), slots=loop['slots'], carries=carries)


def derive(view, unit, function_ordinal):
    if type(function_ordinal) is not int or not 0 <= function_ordinal < len(view.unit(unit).facts.functions):
        raise ValueError('an original source function ordinal is required')
    function = view.unit(unit).facts.functions[function_ordinal]
    if function.body_scope is None or function.environment.kind is not P.DeferredBodyKind.FUNCTION:
        raise ValueError('an original ordinary function body is required')
    source, syntax = _source(view, unit, function)
    return _derive(view, unit, function_ordinal, function, source, syntax)


def _derive(view, unit, ordinal, function, source, syntax):
    projection = _Projection(view, unit, function, syntax)
    loops = list(projection.loops())
    projection.original.check()
    return dict(code_id=source.code_key, qualname=source.qualname, function_ordinal=ordinal,
                body=encode(projection.flow.body), loops=loops)


def project(catalog_path, source_path, code_id=None):
    """Use one admitted local source catalog and optional exact CodeID focus."""
    catalog = read_source_catalog(catalog_path)
    header = catalog.path / 'result.json'
    catalog_ref = _ref(header, catalog.report['schema'])
    source = _ref(source_path, 'scar.python-source')
    view = ProgramView(catalog)
    units = sorted(catalog.units_for_source(source['uri']) or ())
    if not units:
        raise ValueError('source is absent from the original catalog')
    functions = []
    for unit in units:
        for ordinal, function in enumerate(view.unit(unit).facts.functions):
            if function.body_scope is None or function.environment.kind is not P.DeferredBodyKind.FUNCTION:
                continue
            identity, syntax = _source(view, unit, function)
            if code_id is not None and code_id != identity.code_key:
                continue
            if identity.sha256 != source['sha256']:
                raise ValueError('catalog source differs from the current source bytes')
            row = _derive(view, unit, ordinal, function, identity, syntax)
            if row['loops']:
                functions.append(row)
    if code_id is not None and len(functions) != 1:
        raise ValueError('selected original function needs one scope and a returned loop carry')
    return dict(schema=SCHEMA, schema_version=1, catalog=catalog_ref, source=source, code_id=code_id,
                functions=functions, scope='ORIGINAL_F4_NORMAL_CONTROL_AND_LEXICAL_REFERENCE_MAY_DEFINITIONS',
                loops_unrolled=False, target_executed=False, runtime_dispatch_proven=False,
                tensor_contents_or_storage_identity_proven=False, transformation_authorized=False)


def check(path):
    stored = read_artifact(_ref(path, SCHEMA))
    if stored != project(Path(stored['catalog']['uri']).parent, stored['source']['uri'], stored['code_id']):
        raise ValueError('source loop carry differs from cold F4 replay')
    return stored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('catalog', 'source', 'check', 'out'):
        parser.add_argument('--' + flag, type=Path)
    parser.add_argument('--code-id')
    args = parser.parse_args(argv)
    if args.check and not any((args.catalog, args.source, args.out, args.code_id)):
        result = check(args.check)
    elif not args.check and all((args.catalog, args.source, args.out)):
        result = project(args.catalog, args.source, args.code_id)
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        parser.error('supply --check RESULT or --catalog CATALOG --source SOURCE --out RESULT')
    print(json.dumps(dict(functions=len(result['functions']), loops=sum(len(row['loops']) for row in result['functions']))))


if __name__ == '__main__':
    main()
