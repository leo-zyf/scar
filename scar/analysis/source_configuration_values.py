"""Source configuration values."""

import argparse
import ast
import json
import re
from hashlib import sha256
from pathlib import Path
from scar.analysis.source_configuration_overrides import read_configuration_overrides
from scar.analysis.source_configuration_composition_store import read_configuration_composition
from scar.analysis.source_data_materials import DataMaterialStore, _artifact
from scar.analysis.static.lowering.source_string import InterpolationSyntax
from scar.ir.program import configuration as V, configuration as C
from scar.ir.program.configuration import (
    ConfigurationStepKind,
    ConfigurationProtocolSource,
    OverrideAction,
    OverrideExpressionKind,
)
from scar.ir.program.providers import ProviderArtifact
from scar.analysis import (
    normal_output,
    source_symbol_variants,
    source_configuration_overrides as parent,
    source_configuration_composition_store as composition,
    source_data_materials as data_store,
)
from scar.analysis.source_stage import _durable_bytes, _durable_mkdir
from scar.analysis.source_templates import _json
from scar.ir import record_codec as codec
from dataclasses import dataclass, replace
from scar.ir.literals import PythonLiteral
from scar.ir.program.data_materials import DataNodeKind


# Strict storage for conditional configuration algebra and read dependencies.
_SCHEMA = 'scar.configuration-values'
_STATUS = 'CONDITIONAL_STATIC_INPUT_VALUE_GRAPH'
_SEMANTICS = 'CONDITIONAL_STATIC_INPUT_PROJECTIONS_NOT_CURRENT_TASK_CFG_OR_OBJECT_IDENTITY'
_TABLES = dict(expressions=V.ConfigurationExpression, roots=V.ConfigurationRoot,
    projections=V.ConfigurationProjection, interpolations=V.ConfigurationInterpolation,
    read_demands=V.ConfigurationReadDemand, obligations=V.ConfigurationValueObligation)
_HEADER = ('overrides', 'profiles', 'parser_tools', 'selected_root', 'conditions', 'semantics')
_FLAGS = ('target_executed', 'current_argv_proven', 'configuration_resolved', 'first_stage_complete')
_COUNTS = dict(expression_count='expressions', root_count='roots', projection_count='projections',
    interpolation_count='interpolations', read_demand_count='read_demands', obligation_count='obligations')


def _require(condition, message):
    if not condition: raise ValueError(message)


def pins():
    directory = Path(__file__).parent
    implementation = Path(__file__)
    paths = dict(producer=implementation, store=implementation, query=implementation,
        types=Path(V.__file__), codec=Path(codec.__file__),
        interpolation=directory/'static/lowering/source_string.py', override_syntax=directory/'source_override_syntax.py',
        parent=Path(parent.__file__), composition=Path(composition.__file__), data_store=Path(data_store.__file__),
        table_writer=Path(normal_output.__file__), table_reader=Path(source_symbol_variants.__file__))
    return {name: sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def _validate(graph):
    _require(type(graph) is V.ConfigurationValues, 'typed configuration values required')
    graph = codec.decode(V.ConfigurationValues, codec.encode(graph))
    _require(graph.semantics == _SEMANTICS and graph.conditions, 'configuration value graph needs conditional semantics')
    receipts, payloads = {}, {}
    def artifact(value):
        if value.uri not in payloads: payloads[value.uri] = composition._artifact(value, receipts)
        _require(receipts[value.uri] == value, 'conflicting value-graph artifact receipts')
        return payloads[value.uri]
    _require(Path(graph.overrides.uri).name == 'result.json' and graph.overrides.schema == 'scar.configuration-overrides',
        'configuration values require an override graph manifest')
    raw = artifact(graph.overrides)
    report, overrides = parent.read_configuration_overrides(Path(graph.overrides.uri).parent)
    _require(json.loads(raw) == report and set(overrides.conditions) <= set(graph.conditions), 'override snapshot or conditions differ')
    base_path = Path(overrides.composition.uri).parent
    base = json.loads(artifact(overrides.composition))
    materials = {m.id: m for m in (codec.decode(C.ConfigurationMaterial, row)
        for row in source_symbol_variants._table(base_path, base['tables']['materials']))}
    steps = tuple(codec.decode(C.ConfigurationStep, row)
        for row in source_symbol_variants._table(base_path, base['tables']['steps']))
    limits = {key: json.loads(artifact(m.store))['tables']['references']['records'] for key, m in materials.items()}
    trees, functions = {}, set()
    _require(graph.profiles and graph.parser_tools, 'configuration value interpretation needs profiles and parser tools')
    for profile in graph.profiles:
        source = artifact(profile.source); key = profile.source.uri, profile.qualified_function
        _require(key not in functions, 'duplicate configuration value protocol function'); functions.add(key)
        if profile.source.uri not in trees: trees[profile.source.uri] = ast.parse(source)
        node = trees[profile.source.uri]
        for part in profile.qualified_function.split('.'):
            matches = [n for n in node.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == part]
            _require(part.isidentifier() and len(matches) == 1, 'qualified value protocol function differs'); node = matches[0]
        _require(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and
            sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() == profile.function_ast_sha256,
            'configuration value protocol AST differs')
    for tool in graph.parser_tools: artifact(tool)
    def input_ref(source):
        _require(source.ordinal >= 0 and ((source.table == 'MATERIAL_REFERENCE' and source.owner in limits
            and source.ordinal < limits[source.owner]) or (source.table == 'OVERRIDE_EXPRESSION'
            and source.owner == graph.overrides.sha256 and source.ordinal < len(overrides.expressions))), 'input source is outside its parent table')
    expressions = graph.expressions; size = len(expressions); E = V.ConfigurationExpressionKind
    arities = {E.EMPTY: (0,), E.INPUT: (0,), E.OPEN_PROVIDER: (0,), E.EMBED: (1,),
        E.MERGE: (2,), E.ASSIGN: (2,), E.UPDATE: (2,), E.DELETE: (1, 2)}
    for i, expression in enumerate(expressions):
        _require(expression.ordinal == i and all(0 <= n < i for n in expression.inputs)
            and len(expression.inputs) in arities[expression.kind], 'expression ordinal, DAG inputs or arity differ')
        _require(expression.parent_stage in ('COMPOSITION', 'OVERRIDE') and (expression.parent_ordinal is None or
            0 <= expression.parent_ordinal < (len(steps) if expression.parent_stage == 'COMPOSITION' else len(overrides.overrides))),
            'expression lost its original parent stage or row')
        _require((expression.source is not None) == (expression.kind is E.INPUT), 'only INPUT expressions carry source references')
        expected_exclusions = ()
        if expression.source is not None:
            source = expression.source; input_ref(source)
            if source.table == 'MATERIAL_REFERENCE':
                _require(expression.parent_stage == 'COMPOSITION' and expression.parent_ordinal is not None,
                    'material input needs its original composition step')
                step = steps[expression.parent_ordinal]
                _require(step.material == source.owner, 'material input belongs to another composition step')
                if source.ordinal == materials[source.owner].root_reference: expected_exclusions = step.exclude_root_keys
            else: _require(expression.parent_stage == 'OVERRIDE', 'override input has another parent stage')
        _require(expression.exclude_root_keys == expected_exclusions, 'material root exclusions differ from original merge step')
    roots = {(root.stage, root.version): root.expression for root in graph.roots}
    expected = {('COMPOSITION', n) for n in range(len(steps))} | {('OVERRIDE', n) for n in range(overrides.root_version+1)}
    _require(len(roots) == len(graph.roots) and set(roots) == expected and all(0 <= n < size for n in roots.values()),
        'configuration root stages, versions or expressions differ')
    _require(roots['OVERRIDE', 0] == roots['COMPOSITION', overrides.composition_version]
        and graph.selected_root == roots['OVERRIDE', overrides.root_version], 'selected root is not the final override version')
    projections = graph.projections; count = len(projections)
    for i, projection in enumerate(projections):
        _require(projection.ordinal == i and 0 <= projection.root < size and all(0 <= n < count for n in projection.dependencies)
            and all(0 <= n < size for n in projection.evidence), 'projection references differ')
        _require(projection.literal is None or projection.kind is V.ConfigurationValueKind.SCALAR, 'non-scalar projection has a literal')
        _require((type(projection.members_complete) is bool) if projection.kind in (
            V.ConfigurationValueKind.MAPPING, V.ConfigurationValueKind.SEQUENCE) else projection.members_complete is None,
            'container membership completeness must be explicit')
        for source in projection.origins: input_ref(source)
    text_tables = {}
    def source_text(source):
        if source.table == 'OVERRIDE_EXPRESSION':
            literal = overrides.expressions[source.ordinal].literal
            return None if literal is None else literal.to_python()
        if source.owner not in text_tables:
            material = materials[source.owner]; folder = Path(material.store.uri).parent
            header = json.loads(payloads[material.store.uri])
            text_tables[source.owner] = tuple(source_symbol_variants._table(folder, header['tables'][name])
                for name in ('references', 'nodes'))
        references, nodes = text_tables[source.owner]
        return nodes[references[source.ordinal]['target']]['scalar_text']
    for i, interpolation in enumerate(graph.interpolations):
        _require(interpolation.ordinal == i and 0 <= interpolation.projection < count
            and all(0 <= n < count for n in interpolation.dependencies), 'interpolation projection references differ')
        _require(0 <= interpolation.character_start < interpolation.character_end
            and interpolation.character_end-interpolation.character_start == len(interpolation.syntax)
            and interpolation.phase == 'LAZY_SELECTED_ROOT', 'invalid lazy interpolation span or phase')
        texts = [source_text(source) for source in projections[interpolation.projection].origins]
        texts = [text for text in texts if type(text) is str]
        _require(not texts or any(interpolation.character_end <= len(text) and
            text[interpolation.character_start:interpolation.character_end] == interpolation.syntax for text in texts),
            'interpolation span differs from its original scalar text')
    for i, demand in enumerate(graph.read_demands):
        _require(demand.ordinal == i and 0 <= demand.operation < size and (demand.snapshot is None or 0 <= demand.snapshot < size)
            and (demand.projection is None or 0 <= demand.projection < count), 'intermediate read references differ')
        if demand.source is not None: input_ref(demand.source)
        operation = expressions[demand.operation]
        if demand.phase == 'UPDATE_INTERMEDIATE':
            _require(operation.kind in (E.UPDATE, E.ASSIGN, E.DELETE) and demand.snapshot == operation.inputs[0]
                and (demand.projection is None or (projections[demand.projection].root == demand.snapshot
                    and projections[demand.projection].path == demand.path and (demand.source is None or
                    demand.source in projections[demand.projection].origins))), 'UPDATE read snapshot or source differs')
        else:
            _require(demand.phase == 'MERGE_INTERNAL_STATE_PENDING' and operation.kind in (E.MERGE, E.UPDATE)
                and demand.snapshot is None and demand.projection is None, 'unresolved merge read fabricated a snapshot')
    for obligation in graph.obligations:
        _require(obligation.code and (obligation.expression is None or 0 <= obligation.expression < size)
            and (obligation.projection is None or 0 <= obligation.projection < count), 'value obligation references differ')
    return graph, receipts


def _current(receipts):
    for value in receipts.values():
        raw = Path(value.uri).read_bytes()
        _require(len(raw) == value.size_bytes and sha256(raw).hexdigest() == value.sha256, 'configuration value input changed')


def write_configuration_values(graph, out):
    out = Path(out).absolute(); _require(not out.exists(), 'fresh configuration value output required')
    initial = pins(); graph, receipts = _validate(graph); _durable_mkdir(out)
    tables = {name: normal_output.write_table(out/(name+'.jsonl.gz'), getattr(graph, name)) for name in _TABLES}
    counts = {name: len(getattr(graph, name)) for name in _TABLES}
    report = dict(schema=_SCHEMA, schema_version=1, status=_STATUS,
        **{name: codec.encode(getattr(graph, name)) for name in _HEADER}, tables=tables, counts=counts,
        **{name: counts[table] for name, table in _COUNTS.items()}, pins=initial,
        implementation_sha256=initial['producer'], types_sha256=initial['types'], store_sha256=initial['store'],
        **{name: False for name in _FLAGS})
    _current(receipts); _require(pins() == initial, 'configuration value producer changed')
    _durable_bytes(out/'result.json', _json(report), replace=False)
    return report


def read_configuration_values(path):
    path = Path(path).absolute(); raw = (path/'result.json').read_bytes(); report = json.loads(raw)
    expected = {'schema', 'schema_version', 'status', *_HEADER, 'tables', 'counts', *_COUNTS,
        'pins', 'implementation_sha256', 'types_sha256', 'store_sha256', *_FLAGS}
    _require(set(report) == expected and report['schema'] == _SCHEMA and type(report['schema_version']) is int
        and report['schema_version'] == 1 and report['status'] == _STATUS, 'expected conditional configuration value store')
    current = pins()
    _require(report['pins'] == current and all(report[key] == current[name] for key, name in (
        ('implementation_sha256', 'producer'), ('types_sha256', 'types'), ('store_sha256', 'store'))), 'configuration value producer pins differ')
    _require(all(report[name] is False for name in _FLAGS) and set(report['tables']) == set(_TABLES), 'value store has no runtime authority')
    document = {name: report[name] for name in _HEADER}
    for name, kind in _TABLES.items():
        receipt = report['tables'][name]; data_store._receipt(receipt, name+'.jsonl.gz', table=True)
        rows = source_symbol_variants._table(path, receipt)
        _require(len(rows) == receipt['records'], 'configuration value table count differs')
        document[name] = [codec.encode(codec.decode(kind, row)) for row in rows]
    graph, receipts = _validate(codec.decode(V.ConfigurationValues, document))
    counts = {name: len(getattr(graph, name)) for name in _TABLES}
    _require(report['counts'] == counts and all(type(n) is int for n in report['counts'].values())
        and all(type(report[name]) is int and report[name] == counts[table] for name, table in _COUNTS.items()), 'value counts differ')
    _current(receipts)
    _require((path/'result.json').read_bytes() == raw and pins() == current, 'configuration value manifest or producer changed')
    return report, graph


# Persistent plain-input evaluation and version-scoped configuration reads.
#
# Only the analysis tool's immutable cells are evaluated. No target containers,
# custom constructors, resolvers, callbacks or user functions are invoked.
K=V.ConfigurationValueKind
E=V.ConfigurationExpressionKind


@dataclass(frozen=True,slots=True)
class _Cell:
    kind: K
    literal: PythonLiteral | None = None
    origins: tuple[V.ConfigurationInput,...] = ()
    entries: tuple[tuple[str,'_Cell'],...] = ()
    items: tuple['_Cell',...] = ()
    open_keys: bool = False


_ABSENT=_Cell(K.ABSENT)
_OPEN=_Cell(K.OPEN)
_EMPTY=_Cell(K.MAPPING)


def _unique(items):return tuple(dict.fromkeys(items))


def _split_key(key):
    first=re.match(r'(\.)*[^.[]*',key);stop=first.end()
    tokens=key[:stop].split('.')
    if stop==len(key):return tokens
    if key[stop]=='[' and not tokens[-1]:tokens.pop()
    tokens.extend(dot if dot else bracket for dot,bracket in re.findall(r'\.([^.[]*)|\[(.*?)\]',key[stop:]))
    return tokens


def _scalar(literal,origin):
    if literal is None:return _Cell(K.OPEN,origins=(origin,))
    value=literal.to_python()
    kind=K.MISSING if value=='???' else K.INTERPOLATION if type(value) is str and '${' in value else K.SCALAR
    return _Cell(kind,literal if kind in (K.SCALAR,K.INTERPOLATION) else None,(origin,))


class ConfigurationValueQuery:
    def __init__(self,expressions,materials,overrides,override_sha,syntax,conditions,*,max_projections=20000):
        self.expressions=tuple(expressions);self.materials=materials;self.overrides=overrides
        self.override_sha=override_sha;self.syntax=syntax;self.conditions=conditions
        self.max_projections=max_projections;self.cells={};self.source_cells={};self.source_active=set()
        self.projections=[];self.query_ids={};self.resolved={};self.locations={}
        self.interpolations=[];self.read_demands=[];self.obligations=[];self.obligation_keys=set()
        self.texts={}

    def obligation(self,code,expression=None,projection=None,detail=''):
        key=(code,expression,projection,detail)
        if key not in self.obligation_keys:
            self.obligation_keys.add(key)
            self.obligations.append(V.ConfigurationValueObligation(*key))

    def _input(self,source):
        if source in self.source_cells:return self.source_cells[source]
        if source.table=='MATERIAL_REFERENCE':
            material=self.materials[source.owner]
            node=material.nodes[material.references[source.ordinal].target]
            active=(source.owner,node.ordinal)
            if active in self.source_active:
                self.obligation('CYCLIC_INPUT_CONSTRUCTION_PENDING',detail=repr(source))
                return _Cell(K.OPEN,origins=(source,))
            self.source_active.add(active)
            child=lambda ref:self._input(V.ConfigurationInput(source.table,source.owner,ref))
            if node.kind is DataNodeKind.SCALAR:cell=_scalar(node.representation_literal,source)
            elif node.kind is DataNodeKind.SEQUENCE:
                cell=_Cell(K.SEQUENCE,origins=(source,),items=tuple(child(ref) for ref in node.items))
            else:
                entries={};open_keys=False
                for pair in node.entries:
                    key=material.nodes[material.references[pair.key].target].representation_literal
                    value=None if key is None else key.to_python()
                    if type(value) is str:entries[value]=child(pair.value)
                    else:
                        open_keys=True
                        self.obligation('NON_STRING_MAPPING_SELECTOR_PENDING',detail=repr(source))
                cell=_Cell(K.MAPPING,origins=(source,),entries=tuple(entries.items()),open_keys=open_keys)
            self.source_active.remove(active)
        else:
            if source.table!='OVERRIDE_EXPRESSION' or source.owner!=self.override_sha:
                raise ValueError('input expression belongs to a different graph')
            node=self.overrides.expressions[source.ordinal]
            child=lambda ref:self._input(V.ConfigurationInput(source.table,source.owner,ref))
            if node.kind is OverrideExpressionKind.SCALAR:cell=_scalar(node.literal,source)
            elif node.kind is OverrideExpressionKind.SEQUENCE:
                cell=_Cell(K.SEQUENCE,origins=(source,),items=tuple(child(ref) for ref in node.children))
            elif node.kind is OverrideExpressionKind.MAPPING:
                entries={};open_keys=False
                for key,value in node.entries:
                    literal=self.overrides.expressions[key].literal
                    text=None if literal is None else literal.to_python()
                    if type(text) is str:entries[text]=child(value)
                    else:open_keys=True;self.obligation('NON_STRING_MAPPING_SELECTOR_PENDING',detail=repr(source))
                cell=_Cell(K.MAPPING,origins=(source,),entries=tuple(entries.items()),open_keys=open_keys)
            else:
                self.obligation('UNEVALUATED_INPUT_EXPRESSION',detail=repr(source))
                cell=_Cell(K.OPEN,origins=(source,))
        self.source_cells[source]=cell
        return cell

    @staticmethod
    def _member(cell,key,*,negative=False):
        if cell.kind is K.MAPPING:return dict(cell.entries).get(key,_OPEN if cell.open_keys else _ABSENT)
        if cell.kind is K.SEQUENCE:
            try:index=int(key)
            except ValueError:return _Cell(K.TYPE_CONFLICT)
            return cell.items[index] if (0<=index<len(cell.items) or negative and -len(cell.items)<=index<0) else _ABSENT
        if cell.kind in (K.OPEN,K.CYCLE,K.TYPE_CONFLICT):return cell
        if cell.kind in (K.ABSENT,K.MISSING):return cell
        return _Cell(K.NONCONTAINER,origins=cell.origins)

    def _demand(self,operation,path,cell,*,snapshot=None,phase='MERGE_INTERNAL_STATE_PENDING',projection=None):
        self.read_demands.append(V.ConfigurationReadDemand(len(self.read_demands),operation,snapshot,path,phase,
            cell.origins[0] if cell.origins else None,projection))

    def _merge(self,left,right,operation,path=()):
        if right.kind is K.ABSENT:return left
        if left.kind is K.INTERPOLATION:
            self._demand(operation,path,left)
            self.obligation('EAGER_MERGE_DESTINATION_STATE_PENDING',operation,detail=repr(path))
            return _Cell(K.OPEN,origins=_unique((*left.origins,*right.origins)))
        if right.kind is K.INTERPOLATION and left.kind in (K.MAPPING,K.SEQUENCE,K.OPEN):
            self._demand(operation,path,right)
            self.obligation('EAGER_MERGE_SOURCE_VALIDATION_PENDING',operation,detail=repr(path))
            if left.kind is not K.OPEN:return _Cell(K.OPEN,origins=_unique((*left.origins,*right.origins)))
            # The raw lazy candidate remains useful under the explicit open-
            # destination/no-eager-side-effect guard. The demand is not erased.
        if right.kind is K.MISSING:
            if left.kind is K.ABSENT:return right
            if left.kind is K.OPEN:
                self.obligation('MISSING_FALLTHROUGH_TO_OPEN_INPUT',operation,detail=repr(path))
                return _Cell(K.OPEN,origins=_unique((*left.origins,*right.origins)))
            return left
        if right.kind is K.MAPPING:
            if left.kind is K.SEQUENCE:return _Cell(K.TYPE_CONFLICT,origins=_unique((*left.origins,*right.origins)))
            if left.kind in (K.ABSENT,K.MISSING,K.SCALAR):return right
            if left.kind not in (K.MAPPING,K.OPEN):return _Cell(K.OPEN,origins=_unique((*left.origins,*right.origins)))
            entries=dict(left.entries);open_keys=left.open_keys or left.kind is K.OPEN
            if right.open_keys:
                entries={key:_OPEN for key in entries};open_keys=True
            for key,value in right.entries:
                entries[key]=self._merge(entries.get(key,_OPEN if open_keys else _ABSENT),value,operation,(*path,key))
            return _Cell(K.MAPPING,origins=_unique((*left.origins,*right.origins)),
                entries=tuple(entries.items()),open_keys=open_keys)
        if right.kind is K.SEQUENCE and left.kind is K.MAPPING:
            return _Cell(K.TYPE_CONFLICT,origins=_unique((*left.origins,*right.origins)))
        return right

    def _update(self,cell,path,value,operation,*,merge=False,delete=False,prefix=()):
        if not path:return value
        if cell.kind is K.INTERPOLATION:
            snapshot=self.expressions[operation].inputs[0]
            query=self.project(snapshot,prefix)
            self._demand(operation,prefix,cell,snapshot=snapshot,phase='UPDATE_INTERMEDIATE',projection=query)
            self.obligation('INTERMEDIATE_UPDATE_ALIAS_EFFECT_PENDING',operation,query,repr(prefix))
            # An aliased container can redirect writes outside the requested
            # path. Do not preserve unrelated field facts across this effect.
            raise _OpenUpdate
        if cell.kind is K.OPEN:
            snapshot=self.expressions[operation].inputs[0]
            query=self.project(snapshot,prefix)
            self._demand(operation,prefix,cell,snapshot=snapshot,phase='UPDATE_INTERMEDIATE',projection=query)
            self.obligation('OPEN_UPDATE_ADDRESS_SELECTION_PENDING',operation,query,repr(prefix))
            cell=_Cell(K.MAPPING,open_keys=True)
        if cell.kind not in (K.MAPPING,K.SEQUENCE):cell=_EMPTY
        key,*rest=path;old=self._member(cell,key,negative=not rest)
        if rest:new=self._update(old,tuple(rest),value,operation,merge=merge,delete=delete,prefix=(*prefix,key))
        elif delete:new=_ABSENT
        elif merge:
            new=self._merge(old,value,operation,(*prefix,key)) if old.kind in (K.MAPPING,K.SEQUENCE,K.OPEN) else value
        else:new=value
        if cell.kind is K.SEQUENCE:
            try:index=int(key)
            except ValueError:return _Cell(K.TYPE_CONFLICT)
            if not -len(cell.items)<=index<len(cell.items):return _Cell(K.TYPE_CONFLICT)
            items=list(cell.items)
            if delete and not rest:items.pop(index)
            else:items[index]=new
            return replace(cell,items=tuple(items))
        entries=dict(cell.entries);entries[key]=new
        return replace(cell,entries=tuple(entries.items()))

    def compute(self,ordinal):
        if ordinal in self.cells:return self.cells[ordinal]
        expression=self.expressions[ordinal];inputs=[self.compute(i) for i in expression.inputs]
        if expression.kind is E.EMPTY:cell=_EMPTY
        elif expression.kind is E.OPEN_PROVIDER:cell=_Cell(K.MAPPING,open_keys=True)
        elif expression.kind is E.INPUT:
            cell=self._input(expression.source)
            if expression.exclude_root_keys:
                if cell.kind is not K.MAPPING:cell=_Cell(K.TYPE_CONFLICT,origins=cell.origins)
                else:cell=replace(cell,entries=tuple((k,v) for k,v in cell.entries if k not in expression.exclude_root_keys))
        elif expression.kind is E.EMBED:
            cell=inputs[0]
            for key in reversed(expression.path):cell=_Cell(K.MAPPING,entries=((key,cell),))
        elif expression.kind is E.MERGE:cell=self._merge(*inputs,ordinal)
        else:
            row=self.overrides.overrides[expression.parent_ordinal]
            if row.action.value in ('ADD','DELETE'):
                snapshot=expression.inputs[0];dependencies=[]
                target,_=self._locate(snapshot,expression.path,dependencies)
                if target.kind is K.INTERPOLATION:
                    query=self.project(snapshot,expression.path)
                    self._demand(ordinal,expression.path,target,snapshot=snapshot,phase='UPDATE_INTERMEDIATE',projection=query)
            try:cell=self._update(inputs[0],expression.path,inputs[1] if len(inputs)>1 else _ABSENT,ordinal,
                merge=expression.kind is E.UPDATE,delete=expression.kind is E.DELETE)
            except _OpenUpdate:cell=_Cell(K.MAPPING,open_keys=True)
        self.cells[ordinal]=cell
        return cell

    def _locate(self,root,path,dependencies):
        cell=self.compute(root);location=()
        for key in path:
            if cell.kind is K.INTERPOLATION:
                query=self.project(root,location);dependencies.append(query)
                cell=self.resolved.get(query,_Cell(K.CYCLE));target=self.locations.get(query)
                if target is None:return cell,None
                location=target
            cell=self._member(cell,key);location=(*location,key)
        return cell,location

    def _node_interpolation(self,node,root,owner,projection):
        pieces=[];dependencies=[];unknown=False
        for child in node.children:
            if type(child).__name__=='ConfigKeyContext':
                key=child.getChild(0)
                if type(key).__name__=='InterpolationContext':
                    value,_,nested=self._interpolation(key,root,owner,projection)
                    dependencies.extend(nested)
                    if value.kind is not K.SCALAR or type(value.literal.to_python()) is not str:
                        unknown=True
                    else:pieces.append(value.literal.to_python())
                else:pieces.append(child.getText())
            elif child.getText() in ('.','[',']'):pieces.append(child.getText())
        if unknown:return _OPEN,None,tuple(dependencies),'DYNAMIC'
        inter_key=''.join(pieces);dots=len(inter_key)-len(inter_key.lstrip('.'))
        parent=owner[:-1]
        if dots:
            if dots-1>len(parent):
                self.obligation('RELATIVE_INTERPOLATION_ABOVE_ROOT',root,projection,node.getText())
                return _OPEN,None,tuple(dependencies),'NODE'
            base=parent[:len(parent)-(dots-1)]
        else:base=()
        rest=inter_key[dots:];target=(*base,*(_split_key(rest) if rest else ()))
        if len(target)<len(owner) and owner[:len(target)]==target:
            self.obligation('INTERPOLATION_TO_ANCESTOR_PENDING',root,projection,node.getText())
            return _OPEN,target,tuple(dependencies),'NODE'
        query=self.project(root,target);dependencies.append(query)
        value=self.resolved.get(query,_Cell(K.CYCLE))
        if value.kind in (K.MISSING,K.ABSENT):
            self.obligation('INTERPOLATION_TARGET_'+value.kind.value,root,projection,repr(target))
            return _OPEN,target,tuple(dependencies),'NODE'
        return value,target,tuple(dependencies),'NODE'

    def _interpolation(self,ctx,root,owner,projection):
        node=ctx.getChild(0)
        if type(node).__name__=='InterpolationNodeContext':
            value,target,dependencies,kind=self._node_interpolation(node,root,owner,projection)
        else:
            # Record nested reads even when a resolver's value/effects are open.
            dependencies=[];pending=list(reversed(node.children or ()))
            while pending:
                child=pending.pop()
                if type(child).__name__=='InterpolationContext':
                    _,_,nested=self._interpolation(child,root,owner,projection);dependencies.extend(nested)
                elif hasattr(child,'children'):pending.extend(reversed(child.children or ()))
            value=_OPEN;target=None;dependencies=tuple(dependencies);kind='RESOLVER'
            self.obligation('CUSTOM_RESOLVER_VALUE_AND_EFFECT_PENDING',root,projection,node.getText())
        start,end=ctx.start.start,ctx.stop.stop+1
        self.interpolations.append(V.ConfigurationInterpolation(len(self.interpolations),projection,
            start,end,self.texts[projection][start:end],kind,target,dependencies))
        return value,target,dependencies

    def _resolve(self,cell,root,owner,projection):
        text=cell.literal.to_python();self.texts[projection]=text
        tree=self.syntax.parse(text);parts=tree.text().children
        if len(parts)==1 and type(parts[0]).__name__=='InterpolationContext':
            value,target,dependencies=self._interpolation(parts[0],root,owner,projection)
            location=self.locations.get(dependencies[-1],target) if dependencies else target
            return value,location,dependencies
        values=[];dependencies=[];unknown=False
        for index,part in enumerate(parts):
            if type(part).__name__=='InterpolationContext':
                value,_,nested=self._interpolation(part,root,owner,projection);dependencies.extend(nested)
                if value.kind is not K.SCALAR:
                    self.obligation('NON_SCALAR_TEMPLATE_RENDERING_PENDING',root,projection,part.getText())
                    unknown=True
                else:values.append(str(value.literal.to_python()))
            else:
                symbol=part.symbol;following=parts[index+1] if index+1<len(parts) else None
                if symbol.type==self.syntax.lexer.ESC_INTER:values.append(symbol.text[-(len(symbol.text)//2+1):])
                elif symbol.type==self.syntax.lexer.ESC or (symbol.type in (self.syntax.lexer.TOP_ESC,self.syntax.lexer.QUOTED_ESC)
                        and (type(following).__name__=='InterpolationContext' or (following is None and symbol.type==self.syntax.lexer.QUOTED_ESC))):
                    values.append(symbol.text[1::2])
                else:values.append(symbol.text)
        value=_OPEN if unknown else _Cell(K.SCALAR,PythonLiteral.from_python(''.join(values),exact=True),cell.origins)
        return value,owner,tuple(dependencies)

    def project(self,root,path):
        path=tuple(path);key=(root,path)
        if key in self.query_ids:return self.query_ids[key]
        if len(self.projections)>=self.max_projections:raise ValueError('configuration projection budget reached')
        ordinal=len(self.projections);self.query_ids[key]=ordinal;self.projections.append(None)
        dependencies=[];cell,location=self._locate(root,path,dependencies)
        if cell.kind is K.INTERPOLATION:
            raw=cell;cell,location,nested=self._resolve(cell,root,location,ordinal);dependencies.extend(nested)
            origins=_unique((*raw.origins,*cell.origins))
        else:origins=cell.origins
        if cell.kind is K.CYCLE:self.obligation('CYCLIC_INTERPOLATION_DEPENDENCY',root,ordinal,repr(path))
        self.resolved[ordinal]=cell;self.locations[ordinal]=location
        complete=not cell.open_keys if cell.kind is K.MAPPING else True if cell.kind is K.SEQUENCE else None
        self.projections[ordinal]=V.ConfigurationProjection(ordinal,root,path,cell.kind,
            cell.literal if cell.kind is K.SCALAR else None,complete,origins,tuple(dependencies),(root,),location,self.conditions)
        return ordinal

    def known_paths(self,root,*,max_paths=10000):
        pending=[((),self.compute(root))];result=[]
        while pending:
            path,cell=pending.pop();result.append(path)
            if len(result)>=max_paths:
                self.obligation('KNOWN_PATH_ENUMERATION_BUDGET',root,detail=str(max_paths));break
            pending.extend(((*path,key),value) for key,value in reversed(cell.entries))
            pending.extend(((*path,str(i)),value) for i,value in reversed(tuple(enumerate(cell.items))))
        return tuple(result)


class _OpenUpdate(Exception):pass


# Build configuration state algebra and static field/reference dependencies.
_FUNCTIONS={
    'basecontainer.py':{
        'BaseContainer._map_merge':'1113d71066b535589df25331ddc9d4b574e6a545bf6e5a36c682e9c2b9cb6220',
        'BaseContainer._list_merge':'fb3aff9ad7e173b1ad2801bdf8984d2b69edec1a729920785584d038efd38f23',
        'BaseContainer._merge_with':'2f9323c2a98c369cf3f982bbbafc3e375c025157fc7f8383399ccd9ea6b50ae8'},
    'omegaconf.py':{
        'OmegaConf.update':'9d9fc6a9af46fcd2b7c04321fe89876968e693529d03c9c2161471b7efd75581',
        '_select_one':'6cb4383ca45a6b36bc9fb362dcd40b53e28efb4cf4609c5e0fbbc94b7645428e'},
    'base.py':{
        'Container._resolve_key_and_root':'3af964cfe0b5aa39eae5e74f53b53501ff84acf2638d6ada908dbc10e31645d0',
        'Container._select_impl':'8d070e2c47c95ddedb8048411bfd372fadf7f71809b40ffd2865ddf1738b1f55',
        'Container._resolve_node_interpolation':'4b395643b39af89423942653170f47100adee8db69c71b62ef404ebdd954b87c'},
    '_utils.py':{
        '_is_missing_literal':'eda4f88d6e8b9abc7ed5f117bd76b671b607af2fd55677ab2e8709f8eb90ecfe',
        'split_key':'f5789a69cd24b91c7908ae54c83fd8ae8bfc787f0a5f702cc9ef0418c15f4cea'},
    'grammar_visitor.py':{
        'GrammarVisitor.visitInterpolationNode':'543c6824f5b4444b494fca49e7ee6a8c4046cd80054cd19246b236d31544cbe7',
        'GrammarVisitor.visitText':'e494683fed48b9afcda008453d9c015f414208c586f918959a93c6889198724a',
        'GrammarVisitor._unescape':'a12bcd4c4eb3b5adbb0725be73e30863b847cd32a613f622528b620881147cab'},
    'nodes.py':{'ValueNode._set_value':'a6d4c90dfa204321a4cb0b20e6ad0e11a9174222e20cfb361663935472646d33'},
    'dictconfig.py':{
        'DictConfig._validate_merge':'ac29de5abf60c70f85ec05acb35e94b568572b138b7d1a030969f8ee087aa6c9',
        'DictConfig._validate_non_optional':'e047eb8134631f1b200059e4a993a75e281fcf89a5ecbb8b8d947372f3754b85'},
}


def protocol_sources(root):
    result=[]
    for relative,functions in _FUNCTIONS.items():
        artifact,raw=_artifact(Path(root)/relative,'scar.configuration-value-protocol-source')
        tree=ast.parse(raw)
        if relative=='_utils.py':
            patterns={node.targets[0].id:ast.literal_eval(node.value.args[0]) for node in tree.body
                if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
                and node.targets[0].id in ('KEY_PATH_HEAD','KEY_PATH_OTHER')}
            if patterns!={'KEY_PATH_HEAD':r'(\.)*[^.[]*','KEY_PATH_OTHER':r'\.([^.[]*)|\[(.*?)\]'}:
                raise ValueError('configuration key path rules differ')
        for qualified,expected in functions.items():
            node=tree
            for name in qualified.split('.'):
                node=next(n for n in node.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
            digest=sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
            if digest!=expected:raise ValueError('configuration value protocol version is not modeled: '+qualified)
            result.append(ConfigurationProtocolSource(artifact,qualified,digest))
    return tuple(result)


def compile_configuration_expressions(base,overrides,override_sha):
    """Translate existing operations, retaining their table coordinates and guards."""
    expressions=[];roots=[];composition_roots={};E=V.ConfigurationExpressionKind

    def emit(kind,inputs=(),source=None,path=(),exclusions=(),*,stage,index,conditions):
        ordinal=len(expressions)
        expressions.append(V.ConfigurationExpression(ordinal,kind,inputs,source,path,exclusions,stage,index,conditions))
        return ordinal

    for step in base.steps:
        kwargs=dict(stage='COMPOSITION',index=step.ordinal,conditions=step.conditions)
        if step.kind is ConfigurationStepKind.EMPTY:root=emit(E.EMPTY,**kwargs)
        elif step.kind is ConfigurationStepKind.OPEN_PROVIDER:
            opened=emit(E.OPEN_PROVIDER,**kwargs)
            root=emit(E.MERGE,(composition_roots[step.previous],opened),**kwargs)
        else:
            payload=emit(E.INPUT,source=V.ConfigurationInput('MATERIAL_REFERENCE',step.material,step.root_reference),
                exclusions=step.exclude_root_keys,**kwargs)
            if step.package:payload=emit(E.EMBED,(payload,),path=step.package,**kwargs)
            root=emit(E.MERGE,(composition_roots[step.previous],payload),**kwargs)
        composition_roots[step.ordinal]=root;roots.append(V.ConfigurationRoot('COMPOSITION',step.ordinal,root))
    root=composition_roots[overrides.composition_version]
    roots.append(V.ConfigurationRoot('OVERRIDE',0,root))
    for row in overrides.overrides:
        kwargs=dict(stage='OVERRIDE',index=row.ordinal,conditions=row.conditions)
        value=None if row.value is None else emit(E.INPUT,
            source=V.ConfigurationInput('OVERRIDE_EXPRESSION',override_sha,row.value),**kwargs)
        if row.field_path_candidate is None:
            opened=emit(E.OPEN_PROVIDER,**kwargs)
            root=emit(E.MERGE,(root,opened),**kwargs)
        elif row.action is OverrideAction.DELETE:
            root=emit(E.DELETE,(root,) if value is None else (root,value),path=row.field_path_candidate,**kwargs)
        else:
            tag=overrides.expressions[row.value].kind
            kind=E.UPDATE if tag in (OverrideExpressionKind.MAPPING,OverrideExpressionKind.SEQUENCE) else E.ASSIGN
            root=emit(kind,(root,value),path=row.field_path_candidate,**kwargs)
        roots.append(V.ConfigurationRoot('OVERRIDE',row.output_version,root))
    return tuple(expressions),tuple(roots),root


def build_configuration_values(overrides,out,*,omegaconf_source,paths=None,max_projections=20000):
    from scar.analysis.source_configuration_values import write_configuration_values,pins
    initial=pins()
    overrides=Path(overrides).absolute();out=Path(out).absolute()
    if out.exists():raise ValueError('fresh configuration value graph required')
    raw=(overrides/'result.json').read_bytes();override_sha=sha256(raw).hexdigest()
    _,overlay=read_configuration_overrides(overrides)
    _,base=read_configuration_composition(Path(overlay.composition.uri).parent)
    materials={m.id:DataMaterialStore(Path(m.store.uri).parent).read() for m in base.materials}
    profiles=protocol_sources(omegaconf_source);syntax=InterpolationSyntax(omegaconf_source)
    expressions,roots,selected=compile_configuration_expressions(base,overlay,override_sha)
    conditions=tuple(dict.fromkeys((*overlay.conditions,*(c for r in overlay.overrides for c in r.conditions),
        'PLAIN_UNTYPED_CONFIG_CONTAINER_PROTOCOL_SELECTED',
        'NORMAL_SUCCESS_AT_EACH_RECORDED_ROOT_VERSION',
        'OPEN_DESTINATIONS_DO_NOT_EAGERLY_RESOLVE_LAZY_CANDIDATES',
        'OPEN_UPDATE_ANCESTORS_USE_DIRECT_MAPPING_LOCATIONS',
        'PENDING_EAGER_EFFECTS_DO_NOT_INVALIDATE_OTHER_PROJECTIONS',
        'ADD_DELETE_EXISTENCE_AND_VALUE_MATCH_GUARDS_SATISFIED')))
    query=ConfigurationValueQuery(expressions,materials,overlay,override_sha,syntax,conditions,max_projections=max_projections)
    for path in query.known_paths(selected) if paths is None else paths:query.project(selected,path)
    query.obligation('UPSTREAM_CONFIGURATION_CONDITIONS_UNDISCHARGED',detail=
        'Provider/schema/search selection, current argv, parsed function values, callbacks and task-world binding remain conditional.')
    parent=ProviderArtifact(str(overrides/'result.json'),override_sha,len(raw),'scar.configuration-overrides')
    graph=V.ConfigurationValues(parent,profiles,syntax.profiles,expressions,roots,tuple(query.projections),
        tuple(query.interpolations),tuple(query.read_demands),tuple(query.obligations),selected,conditions)
    if (overrides/'result.json').read_bytes()!=raw or pins()!=initial:
        raise ValueError('configuration override parent or analysis implementation changed')
    return write_configuration_values(graph,out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overrides',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--omegaconf-source',required=True)
    parser.add_argument('--path-json',action='append',help='One precise string-segment path as a JSON array; default: known paths')
    args=parser.parse_args();paths=None if args.path_json is None else [tuple(json.loads(p)) for p in args.path_json]
    if paths is not None and any(any(type(s) is not str for s in path) for path in paths):
        raise ValueError('configuration selector segments must be strings')
    report=build_configuration_values(args.overrides,args.out,omegaconf_source=args.omegaconf_source,paths=paths)
    print(json.dumps({k:report[k] for k in ('status','expression_count','root_count','projection_count',
        'interpolation_count','read_demand_count','obligation_count')}),flush=True)


if __name__=='__main__':main()
