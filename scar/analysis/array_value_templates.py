"""Source-replayed, conditional content relations in the public program graph.

One admission discovers a source file once. Expressions remain source-bound;
the existing optimizer supplies CSE/folding/copy forwarding. Publication and
readback use the same derivation, not a trusted serialized optimization map.
"""
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from itertools import zip_longest
import json
from pathlib import Path

from scar.analysis.source_consumer_parents import _stamp
from scar.compiler.model import region_from_dict
from scar.ir.execution_recipe import read_artifact
from scar.ir.program import view as V, source_productions as P
from scar.ir.program.array_value_templates import ArrayValueTemplate, ArrayMaterializationRole as Role
from scar.ir.program.dependencies import (GraphNamespace, DependencyPoint as Point,
    DependencyKind as K, DependencyMeaning as Meaning, DependencyStrength as Strength,
    DependencyCondition as Condition, DependencyEvidence as Evidence, DependencyEdge as Edge,
    DependencyNeighborhood as Neighborhood, DependencyFrontier as Frontier)
from scar.ir.program.ids import OperationID, RegionID, ValueID
from scar.ir.program.providers import ProviderArtifact, ProviderFactReference
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.program.source_template_links import SourceTemplateKey
from scar.ir.record_codec import encode, decode
from scar.optimization.graph import ordered_nodes, canonical_value, value_kinds


SCHEMA = 'scar.array-value-templates'
SCOPE = 'PARAMETRIC_CONTENT_FOR_ONE_GUARDED_REGION_ACTIVATION'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value):return sha256(_canonical(value)).hexdigest()


def _fact(reference, documents):
    artifact = reference.artifact;path = Path(artifact.uri);stamp = _stamp(path)
    if artifact not in documents:
        document = read_artifact(asdict(artifact))
        if _stamp(path) != stamp:raise ValueError('array region source changed during read')
        documents[artifact] = document,stamp
    document,original_stamp = documents[artifact]
    if stamp != original_stamp:raise ValueError('admitted array region source changed')
    if reference.locator == '#':return document
    if not reference.locator.startswith('#/'):raise ValueError('original array region needs a JSON pointer')
    for part in reference.locator[2:].split('/'):
        key = part.replace('~1', '/').replace('~0', '~')
        document = document[int(key)] if type(document) is list else document[key]
    return document


class ArrayTemplateSource:
    """One formal F4 discovery shared by every array region in this source."""
    def __init__(self, view, source):
        from scar.compiler.graph_frontend import discover_graph_regions
        self.view = view;self.source = str(Path(source).resolve())
        self.regions, self.rejections = discover_graph_regions(view, source=self.source)
        self._regions = {_digest(asdict(row)):row for row in self.regions}
        self._rules = _pins();self._documents = {}

    def analyze(self, original, *, memory_budget_bytes=None, cpu_threads=1, register=True):
        from scar.optimization import optimize_region
        if type(original) is not ProviderFactReference:raise TypeError('original region FactRef required')
        raw = _fact(original,self._documents);key = _digest(raw)
        if key not in self._regions or _canonical(asdict(self._regions[key])) != _canonical(raw):
            raise ValueError('array value template differs from the original F4 derivation')
        region = region_from_dict(raw)
        plan = optimize_region(region, memory_budget_bytes=memory_budget_bytes, cpu_threads=cpu_threads)
        if plan.proof.get('supported') is not True:return plan, None
        provider = ArrayValueTemplateProvider(self, original, region, plan,
            memory_budget_bytes=memory_budget_bytes, cpu_threads=cpu_threads)
        provider.check()
        if register:self.view.register(provider)
        return plan, provider


class ArrayValueTemplateProvider:
    """Definition graph only; every content relation carries profile guards."""
    relation_kinds = frozenset((K.OPERAND, K.RESULT, K.PROJECTION, K.IMPLEMENTATION, K.CONTAINS))

    def __init__(self, admission, reference, region, plan, *, memory_budget_bytes, cpu_threads):
        self.admission = admission;self.view = admission.view
        self.reference = reference;self.region = region;self.plan = plan
        self.configuration = dict(memory_budget_bytes=memory_budget_bytes, cpu_threads=cpu_threads)
        artifact = reference.artifact
        self.locator = SourceArtifactLocator(artifact, artifact.sha256, artifact.size_bytes,
            'canonical-json', reference.locator)
        self.basis = _digest(asdict(reference))
        self.namespace = GraphNamespace(self.locator, SCHEMA, RegionID(self.basis))
        self.owned_namespaces = (self.namespace,)
        self.entry = Point(self.namespace, self.namespace.root, None)
        self.nodes = {self.entry.node:V.ProgramNode(self.entry, V.ProgramEntityKind.REGION,
            V.ProgramNodeRole.REGION, None, None)}
        self.records = {self.entry.node:reference};self.operations = {};self.values = {}
        self.templates = [];self.in_edges = defaultdict(list);self.out_edges = defaultdict(list)
        self._sources = {};self._original_stamp = _stamp(Path(artifact.uri))
        self._assemble()

    @staticmethod
    def _key(point):return point.namespace, point.node

    def _edge(self, source, target, kind, meaning, rule):
        # The region owns the conjunction once; do not repeat every guard
        # and all its qualified endpoints on every local relation.
        conditions = () if kind is K.CONTAINS else (Condition(SCHEMA,
            'ALL_REGION_PROFILE_GUARDS_AND_SAME_ACTIVATION_INPUT_VERSIONS', self.entry),)
        edge = Edge(source, target, kind, meaning,
            (Evidence(SCHEMA+'.'+rule, (self.locator,), (source, target)),),
            conditions, Strength.EXACT if not conditions else Strength.MAY)
        self.out_edges[self._key(source)].append(edge);self.in_edges[self._key(target)].append(edge)

    def _source(self, origin, target):
        if origin.get('kind') != 'original_f4':return
        source = self.view.unit(origin['unit']);self._sources[source.unit] = source
        if 'step' in origin:
            prefix = P.ProductionStepID.prefix+':'
            wire = origin['step']
            if not wire.startswith(prefix):raise ValueError('array origin has another operation ID domain')
            point = source._node(decode(SourceTemplateKey,origin['template']),
                P.ProductionStepID(wire[len(prefix):]))
            source.ensure(point)
            self._edge(point,target,K.IMPLEMENTATION,Meaning.STRUCTURE,'original-f4-operation')
        elif 'value' in origin:
            from scar.ir.program.normal_targets import GlobalPort
            point = self.view.source_port(decode(GlobalPort,origin['value']),context=None)
            self._edge(point,target,K.PROJECTION,Meaning.DEPENDENCE,'original-input-port')

    def _assemble(self):
        nodes = ordered_nodes(self.region);kinds = value_kinds(nodes)
        origins = defaultdict(list)
        for i,origin in enumerate(self.region.source.origins):
            if origin.get('kind') == 'original_f4' and 'expression' in origin:
                origins[origin['expression']].append(i)
        # Content templates are functions of input versions, scoped to this
        # original region. Copy preserves contents but not object/storage.
        classes = {}
        for node in ordered_nodes(self.plan.region):
            classes[node.id] = (classes[node.inputs[0]] if node.op == 'copy' else
                _digest([self.basis,node.op,[classes[i] for i in node.inputs],canonical_value(node.attrs)]))
        for node in nodes:
            oid = OperationID(_digest([self.basis,'operation',node.id]))
            vid = ValueID(_digest([self.basis,'value',node.id]))
            op = Point(self.namespace,oid,None);value = Point(self.namespace,vid,None)
            self.operations[node.id] = op;self.values[node.id] = value
            self.nodes[oid] = V.ProgramNode(op,V.ProgramEntityKind.OPERATION,V.ProgramNodeRole.OPERATION,self.entry,None)
            self.nodes[vid] = V.ProgramNode(value,V.ProgramEntityKind.VALUE,
                V.ProgramNodeRole.PARAMETER if node.op == 'input' else V.ProgramNodeRole.RESULT,self.entry,None)
            destination = self.plan.proof['source_operation_map'][node.id]['optimized_node']
            content = classes.get(destination)
            if content is None:
                content = (self.records[self.values[node.inputs[0]].node].content_template if node.op == 'copy' else
                    _digest([self.basis,'source-only',node.op,
                        [self.records[self.values[i].node].content_template for i in node.inputs],canonical_value(node.attrs)]))
            role = (Role.INPUT_REFERENCE if node.op == 'input' else Role.VIEW_DESCRIPTOR if node.op == 'slice'
                else Role.IMMUTABLE_SCALAR if kinds[node.id] == 'scalar' else Role.FRESH_ARRAY_RESULT)
            row = ArrayValueTemplate(node.id,vid,content,role,node.inputs,destination,tuple(origins[node.id]),
                node.id == self.region.output and role is Role.FRESH_ARRAY_RESULT)
            self.templates.append(row);self.records[vid] = row;self.records[oid] = node
            self._edge(self.entry,op,K.CONTAINS,Meaning.STRUCTURE,'source-expression')
            for operand in node.inputs:self._edge(self.values[operand],op,K.OPERAND,Meaning.DEPENDENCE,'operand')
            self._edge(op,value,K.RESULT,Meaning.VALUE,'result-template')
            for i in origins[node.id]:self._source(self.region.source.origins[i],op)
            if node.op == 'copy':
                self._edge(self.values[node.inputs[0]],value,K.PROJECTION,Meaning.VALUE,'copy-same-content-fresh-materialization')

    def check(self):
        if _stamp(Path(self.reference.artifact.uri)) != self._original_stamp:
            raise ValueError('original array-region artifact changed during query')
        for source in self._sources.values():source.check()

    def describe(self, point):
        if point.namespace != self.namespace or point.node not in self.nodes:return None
        self.check()
        # A definition template does not silently become a selected runtime or
        # caller-frame value. A matched activation interface is separate.
        if point.context is not None:return None
        return self.nodes[point.node]

    def original(self, point):
        if self.describe(point) is None:raise ValueError('not an unselected array template endpoint')
        return deepcopy(self.records[point.node])

    def expand(self, point):
        self.original(point)
        if point == self.entry:return tuple(self.operations.values())
        return tuple(edge.target for edge in self.out_edges[self._key(point)] if edge.kind is K.RESULT)

    def explain(self, point):
        return dict(node=self.describe(point),original=self.original(point),scope=SCOPE,
            required_profile_guards=tuple(dict.fromkeys((*self.region.assumptions,*self.plan.proof['runtime_guards']))),
            runtime_logical_or_storage_identity_proven=False,cross_call_reuse_authorized=False,
            runtime_guards_discharged=False,transformation_authorized=False)

    def _neighbors(self, point, incoming):
        edges = (self.in_edges if incoming else self.out_edges).get(self._key(point),())
        if not edges:return Neighborhood()
        self.check()
        if point.context is not None:return Neighborhood(frontiers=(Frontier(point,
            'ARRAY_CONTENT_TEMPLATE_ACTIVATION_NOT_SELECTED','Definition-time contents need a matched caller activation and input versions.'),))
        return Neighborhood(tuple(edges))

    def incoming(self, point):return self._neighbors(point,True)
    def outgoing(self, point):return self._neighbors(point,False)

    def summary(self):
        return dict(scope=SCOPE,expressions=len(self.templates),nodes=len(self.nodes),
            relations=sum(map(len,self.out_edges.values())),
            content_templates=len({row.content_template for row in self.templates}),
            source_copy_operations=sum(n.op == 'copy' for n in self.region.nodes),
            retained_copy_operations=sum(n.op == 'copy' for n in self.plan.region.nodes),
            fresh_output_required=any(row.required_fresh_output for row in self.templates),
            source_derivation_replayed=True,runtime_logical_versions_created=0,
            physical_allocations_observed=0,cross_call_reuse_authorized=False,
            target_executed=False,transformation_authorized=False)


def _pins():
    from scar.compiler import graph_frontend,array_copy_contract
    from scar.optimization import graph,optimizer
    from scar.ir.program import array_value_templates
    return {name:sha256(Path(module.__file__).read_bytes()).hexdigest() for name,module in (
        ('projection',__import__(__name__,fromlist=['__file__'])),('types',array_value_templates),
        ('source_lowering',graph_frontend),('copy_contract',array_copy_contract),
        ('structure',graph),('optimizer',optimizer))}


def _validate_public_graph(provider):
    for rows in provider.out_edges.values():
        for edge in rows:
            if (edge not in provider.view.dependencies.incoming(edge.target,kinds=(edge.kind,)).edges or
                edge not in provider.view.dependencies.outgoing(edge.source,kinds=(edge.kind,)).edges):
                raise ValueError('array value relation absent from public graph')
    return dict(bidirectional_missing=0)


def publish_array_templates(provider,out):
    from scar.analysis.normal_output import write_table
    out = Path(out).resolve()
    if out.exists():raise ValueError('fresh array template output required')
    provider.check()
    if provider.admission._rules != _pins():raise ValueError('array template rules changed since source discovery')
    out.mkdir(parents=True)
    groups = dict(nodes=tuple(provider.nodes.values()),values=tuple(provider.templates),
        edges=tuple(edge for rows in provider.out_edges.values() for edge in rows))
    tables = {name:write_table(out/(name+'.jsonl.gz'),rows) for name,rows in groups.items()}
    validation = _validate_public_graph(provider)
    report = dict(schema=SCHEMA,schema_version=1,original_region=encode(provider.reference),
        source=provider.admission.source,configuration=provider.configuration,pins=_pins(),
        tables=tables,summary=provider.summary(),graph_validation=validation,
        runtime_guards_discharged=False,transformation_authorized=False)
    provider.check()
    (out/'result.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return report


def read_array_templates(path,view,*,admission=None):
    from scar.analysis.source_symbol_variants import _table
    path = Path(path).resolve();raw = (path/'result.json').read_bytes();report = json.loads(raw)
    if (report.get('schema'),report.get('schema_version')) != (SCHEMA,1) or report['pins'] != _pins():
        raise ValueError('array value template schema or producing rules differ')
    admission = ArrayTemplateSource(view,report['source']) if admission is None else admission
    if admission.view is not view or admission.source != report['source']:
        raise ValueError('array template source belongs to another program view')
    _,provider = admission.analyze(decode(ProviderFactReference,report['original_region']),
        **report['configuration'],register=False)
    if provider is None:raise ValueError('array template no longer has the closed supported operator profile')
    expected = dict(nodes=tuple(provider.nodes.values()),values=tuple(provider.templates),
        edges=tuple(edge for rows in provider.out_edges.values() for edge in rows))
    if (set(report['tables']) != set(expected) or report['summary'] != provider.summary()
        or report['graph_validation'] != dict(bidirectional_missing=0)
        or report['runtime_guards_discharged'] is not False or report['transformation_authorized'] is not False):
        raise ValueError('array value template scope or summary differs')
    sentinel = object()
    for name,rows in expected.items():
        cls = V.ProgramNode if name == 'nodes' else ArrayValueTemplate if name == 'values' else Edge
        stored = (decode(cls,row) for row in _table(path,report['tables'][name]))
        if any(left != right for left,right in zip_longest(rows,stored,fillvalue=sentinel)):
            raise ValueError('array value template table differs: '+name)
    provider.check()
    if (path/'result.json').read_bytes() != raw:raise ValueError('array template report changed during read')
    view.register(provider)
    if _validate_public_graph(provider) != report['graph_validation']:
        raise ValueError('array value template public graph differs')
    return provider


__all__ = ['ArrayTemplateSource','ArrayValueTemplateProvider','publish_array_templates','read_array_templates']
