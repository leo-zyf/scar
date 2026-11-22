"""Contextual call resolution."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, replace, fields, is_dataclass
from scar.ir.program.contextual_calls import ContextualCallTarget, ContextualFunctionSource
from scar.ir.program.dependencies import (
    DependencyCondition as Condition,
    DependencyEdge as Edge,
    DependencyEvidence as Evidence,
    DependencyFrontier as Frontier,
    DependencyKind as K,
    DependencyMeaning as M,
    DependencyNeighborhood as Neighborhood,
    DependencyPoint as Point,
    DependencyStrength as Strength,
    GraphNamespace,
    context_frames,
    context_pop,
    NODE_TYPES,
    DependencyFrame,
    DependencyPoint,
    DependencyCondition,
    DependencyEvidence,
    DependencyEdge,
    DependencyFrontier,
    DependencyKind,
    DependencyMeaning,
    DependencyStrength,
    OpenDependencyContext,
)
from scar.ir.program.ids import OperationID, RegionID, UseID, ValueID
from scar.ir.program.model import (
    BlockArgument,
    LocalBindingValue,
    ReferenceValue,
    StateVersionValue,
    TerminatorKind,
)
from scar.ir.program.function_sites import StaticQualifiedFunctionCreation
from scar.ir.program.source_call_bindings import SourceBodyNamespace
from scar.ir.semantics_v2 import Opcode
from enum import Enum
import math
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints
from scar.analysis.program_dependency_export import _Tables
from scar.ir.literals import PythonLiteral
from scar.ir.program.source_references import SourceArtifactLocator
from scar.ir.record_codec import decode, loads
from scar.ir.v2.common import SourceReference
from scar.ir.v2.ids import Identifier, OperationDefinitionID


# Resolve actual source CALLs through the shared contextual reference graph.
#
# The legacy activation solver keeps its closed CALL requirements. This consumer
# retains the caller and factory axes separately and exposes guarded interfaces
# on that same graph, including unknown outer prefixes. No target executes.
REFERENCE_KINDS = frozenset((K.VALUE_USE, K.BLOCK_ARGUMENT, K.BINDING_PAYLOAD,
    K.BINDING_READ, K.ARGUMENT, K.COMPLETION, K.ENVIRONMENT, K.STATE_VALUE))


@dataclass(frozen=True, slots=True)
class ContextualCallResolution:
    call: Point
    targets: tuple[ContextualCallTarget, ...]
    frontiers: tuple[Frontier, ...]
    enumerated: bool = False


class ContextualCallResolver:
    relation_kinds = frozenset((K.CALL_TARGET, K.ARGUMENT, K.COMPLETION))

    def __init__(self, owner):
        from scar.analysis.static.call_dependency_view import StaticCallDependencyView
        from scar.analysis.static.module_attributes import ModuleAttributeDependencies
        from scar.analysis.dependencies.classes import ModuleReferenceDependencyProvider
        self.owner = owner
        self.session = StaticCallDependencyView(owner)
        self._active = set()
        self._resolve_depth = 0
        self._last_receiver_evidence = None
        self._last_class_sources = None
        self._class_implementation_coordinator = None
        from scar.analysis.contextual_class_sources import ClassStatementSourceExtractor
        self._class_source_extractor = ClassStatementSourceExtractor(self)
        self.counts = Counter()
        self.modules = ModuleAttributeDependencies(load_body=owner._load_body_namespace,
            source_context=lambda body:(body.context.tree,body.context.semantic,body.context.docs['source']['text']),
            module_bodies=self._module_bodies, namespace_for=owner._namespace_for,
            reference_sources=None, max_steps=owner.budget.max_resolution_steps,
            max_facts=owner.budget.max_namespace_facts, check=owner._check,
            max_cached_bytes=min(owner.budget.max_cached_bytes,owner.budget.max_record_bytes))
        self.module_provider = ModuleReferenceDependencyProvider(self.session,self.modules,body_for=self._body)
        graph = self.session.graph()
        graph.register(self.module_provider)
        from scar.analysis.contextual_module_bindings import ContextualModuleBindingProvider
        self.contextual_modules = ContextualModuleBindingProvider(self)
        graph.register(self.contextual_modules)
        graph.register(self)
        from scar.analysis.contextual_receiver_sources import ContextualTypeCallImplementations
        self.type_calls = ContextualTypeCallImplementations(self)
        graph.register(self.type_calls)
        self._class_implementation_coordinator = self.type_calls.observe

    def _body(self, namespace):
        # Check every supplied authority first, including a registered foreign
        # class basis. Then share the owner's original body/index/document.
        original = self.session.load(namespace)
        if type(original.namespace)is not SourceBodyNamespace:
            raise TypeError('source CALL lookup requires an original source-body namespace')
        return self.owner._body(self.owner._unit(original.unit_number),original.body_position,original.row)

    def _module_bodies(self, name):
        for number in self.owner._module_units.get(name, ()):
            context = self.owner._unit(number)
            for position,row in enumerate(self.owner.units[number]['bodies']):
                if row['kind']=='MODULE' and 'artifact'in row and row.get('status')=='LOWERED_WITH_GAPS':
                    body=self.owner._body(context,position,row)
                    if body is not None:yield body

    @staticmethod
    def _calls(body):
        selected={link.operation:link for link in body.operation_links if link.opcode is Opcode.CALL}
        for target in body.metadata.call_targets:
            if target.operation not in selected:
                links=body.links_by_op.get(target.operation,())
                if len(links)!=1:raise ValueError('contextual CALL has no unique original source link')
                selected[target.operation]=links[0]
        return selected

    def _transport(self, edge):
        if edge.kind not in REFERENCE_KINDS:return False
        if edge.kind is K.VALUE_USE:
            if (edge.meaning is not M.VALUE or type(edge.source.node)is not ValueID
                    or type(edge.target.node)is not UseID or edge.source.namespace!=edge.target.namespace
                    or edge.source.context!=edge.target.context):return False
            index=self.session.index(edge.source.namespace)
            return (type(index.values.get(edge.source.node))in(ReferenceValue,LocalBindingValue)
                and index.use_site(edge.target.node).use.value==edge.source.node)
        if edge.kind in (K.BINDING_PAYLOAD,K.BINDING_READ):return edge.meaning is M.BINDING
        if edge.kind is K.BLOCK_ARGUMENT:
            index=self.session.index(edge.target.namespace)
            return edge.meaning is M.VALUE and type(index.values.get(edge.target.node))in(ReferenceValue,LocalBindingValue)
        return edge.meaning is M.REFERENCE

    def _factory(self, point):
        if type(point.node)is not ValueID:return None
        if point.namespace.source_body is None:return None
        body=self._body(point.namespace)
        value=body.index.values[point.node]
        if type(value)is not ReferenceValue or type(value.origin)is not BlockArgument:return None
        from scar.analysis.static.namespace import normal_reference
        saved=getattr(body,'contextual_factory_inputs',None)
        if saved is None:
            dependencies=self.owner._namespace_for(body)
            saved={}
            for identity,kind in dependencies.protocols.items():
                if kind not in ('MAKE_FUNCTION_314','MAKE_NESTED_FUNCTION'):continue
                normal=normal_reference(body.index,identity,blocks=dependencies.blocks)
                if normal is not None:saved[normal[1]]=(body.index.operations[identity],normal[0])
            body.contextual_factory_inputs=saved
        pair=saved.get(point.node)
        if pair is None:return None
        creation=self.owner._function_creation(body,pair[0],point.node,pair[1])
        return (body,StaticQualifiedFunctionCreation(body.namespace,creation)) if creation is not None else None

    def receiver_evidence(self, call):
        """Only the last completed outer query owns these ephemeral walks."""
        saved = self._last_receiver_evidence
        return saved[1] if saved is not None and saved[0] == call else None

    def class_sources(self, call):
        """Class-statement products from the last completed outer query."""
        saved = self._last_class_sources
        return saved[1] if saved is not None and saved[0] == call else None

    def class_invocations(self, call):
        return self.type_calls.for_call(call)

    def resolve(self, call):
        receivers = {}
        class_sources = {}
        outermost = self._resolve_depth == 0
        self._resolve_depth += 1
        try:
            result = self._resolve(call, receivers, class_sources)
            if outermost:
                sources = tuple(class_sources)
                if self._class_implementation_coordinator is not None:
                    self._class_implementation_coordinator(call, sources)
                    result = replace(result, frontiers=tuple(dict.fromkeys(
                        (*result.frontiers, *self.type_calls.frontiers_for(call)))))
                self._last_receiver_evidence = (call, tuple(receivers.values()))
                self._last_class_sources = (call, sources)
            return result
        finally:
            self._resolve_depth -= 1

    def _resolve(self, call, receivers, class_sources):
        if type(call)is not Point or type(call.node)is not OperationID:
            raise TypeError('contextual resolution needs an original CALL operation point')
        self.owner._check()
        body=self._body(call.namespace)
        link=self._calls(body).get(call.node)
        if link is None:raise ValueError('contextual source selection is not an original source CALL')
        notes={};targets={}
        def note(kind,detail):notes[Frontier(call,kind,detail)]=None
        if len(context_frames(call.context))>self.owner.budget.max_context_depth:
            note('CONTEXTUAL_CALL_CONTEXT_DEPTH_BUDGET','Known CALL suffix exceeds the explicit context-depth budget; its unknown prefix is not closed.')
            return ContextualCallResolution(call,(),tuple(notes))
        if call in self._active:
            note('CONTEXTUAL_CALL_SOURCE_CYCLE_OPEN','A repeated contextual CALL cannot seed its own reference source.')
            return ContextualCallResolution(call,(),tuple(notes))
        self._active.add(call)
        try:
            op=body.index.operations[call.node]
            uses=[row.use for row in body.uses_by_op.get(call.node,())if row.role=='callee']
            if len(uses)!=1:
                note('CONTEXTUAL_CALL_CALLEE_USE_UNAVAILABLE','The real CALL has no unique retained callable Use.')
                return ContextualCallResolution(call,(),tuple(notes))
            callee=Point(call.namespace,uses[0],call.context)
            value=body.index.use_site(uses[0]).use.value
            callee_value=Point(call.namespace,value,call.context)
            pending=deque(((callee,(),frozenset((callee,))),));steps=0
            while pending:
                self.owner._check();point,reverse_path,ancestors=pending.popleft();steps+=1
                if steps>self.owner.budget.max_resolution_steps:
                    raise OverflowError('contextual reference-source traversal budget exhausted')
                found=self._factory(point)
                if found is not None:
                    origin,creation=found
                    source=ContextualFunctionSource(callee,point,creation,tuple(reversed(reverse_path)))
                    actuals,issues=self.owner._actuals(body,op,body.context.call_nodes.get(link.source.atom_id))
                    for issue in issues:note(issue.kind,issue.detail)
                    interface=self.owner._target(body,op,link.source,link.semantic_operation,actuals,
                        origin.context.number,creation.creation.definition.definition,
                        ('CONTEXTUAL_REFERENCE_GRAPH_FACTORY_SOURCE',),qualified_creations=(creation,),interface_only=True)
                    worlds=[use for use in op.uses if type(body.index.values[use.value])is StateVersionValue]
                    if len(worlds)!=1:raise ValueError('contextual CALL has no unique actual input world')
                    world=Point(call.namespace,worlds[0].id,call.context)
                    code=Point(point.namespace,creation.creation.code_use,point.context)
                    conditions=(Condition('scar.contextual-call','CURRENT_FUNCTION_FIELDS_MATCH_INTERFACE',
                        callee_value,operands=(world,point,code,call)),)
                    evidence=(Evidence('CONTEXTUAL_SOURCE_CALL_INTERFACE_V1',
                        (call.namespace.graph,point.namespace.graph),(call,callee,callee_value,world,point,code)),)
                    target=ContextualCallTarget(call,callee,callee_value,source,interface,conditions,evidence)
                    if target not in targets and len(targets)>=self.owner.budget.max_candidates_per_call:
                        raise OverflowError('contextual source target alternative budget exhausted')
                    targets[target]=None
                    if target.frame is not None and len(context_frames(call.context))>=self.owner.budget.max_context_depth:
                        note('CONTEXTUAL_CALL_INTERFACE_DEPTH_BUDGET','The source candidate is retained; entering another CALL would exceed the known-context suffix budget.')
                    continue
                from scar.analysis.program_lowering import _Unsupported
                try:
                    class_source = self._class_source_extractor.source_at(
                        call, callee, point, tuple(reversed(reverse_path)))
                except _Unsupported as error:
                    class_source = None
                    note('CONTEXTUAL_CLASS_SOURCE_RULE_OPEN',
                        str(getattr(error.kind, 'value', error.kind)) + ': ' + error.detail)
                if class_source is not None:
                    if (class_source not in class_sources and len(class_sources)
                            >= self.owner.budget.max_candidates_per_call):
                        raise OverflowError('contextual class-statement source alternative budget exhausted')
                    class_sources[class_source] = None
                from scar.analysis.contextual_receiver_sources import (
                    attribute_read_at, collect_receiver_evidence,
                )
                read = attribute_read_at(self, point)
                if read is not None:
                    key = (point.namespace, read.operation, point.context)
                    if key not in receivers:
                        evidence = collect_receiver_evidence(self, read, point.context)
                        receivers[key] = evidence
                        notes.update((frontier, None) for frontier in evidence.walk.frontiers)
                neighborhood=self.session.graph().incoming(point,kinds=REFERENCE_KINDS)
                notes.update((frontier,None)for frontier in neighborhood.frontiers)
                for edge in neighborhood.edges:
                    if not self._transport(edge):continue
                    if edge.source in ancestors:
                        note('CONTEXTUAL_REFERENCE_CYCLE_OPEN','A reference-flow cycle does not establish its own source.')
                        continue
                    pending.append((edge.source,(*reverse_path,edge),ancestors|{edge.source}))
            self.counts['queries']+=1;self.counts['target_alternatives']+=len(targets)
            self.counts['class_statement_source_alternatives'] += len(class_sources)
        except (OverflowError,TimeoutError)as error:
            note('CONTEXTUAL_CALL_RESOLUTION_BUDGET',type(error).__name__+': '+str(error))
        finally:
            self._active.remove(call)
        note('CONTEXTUAL_CALL_TARGET_SET_OPEN','Original factory paths support conditional candidates; other targets, fields and entry state remain open.')
        return ContextualCallResolution(call,tuple(targets),tuple(notes))

    def validate_target(self, target):
        if type(target)is not ContextualCallTarget:raise TypeError('expected ContextualCallTarget')
        if target not in self.resolve(target.call).targets:
            raise ValueError('contextual CALL source/interface/guards differ from fresh original inputs')
        return True

    def _edges(self,target):
        if len(context_frames(target.call.context))>=self.owner.budget.max_context_depth:
            return ()
        edges=target.dependency_edges()
        if target.frame is not None:
            entry=Point(target.frame.target,target.interface.body.region,target.callee_context)
            edges=(*edges,Edge(target.call,entry,K.CALL_TARGET,M.STRUCTURE,
                target._proofs('CONTEXTUAL_SOURCE_CALL_TARGET_V1',(entry,)),target._guards(),Strength.MAY))
        return edges

    def _neighbors(self, point, outgoing):
        if type(point)is not Point:raise TypeError('contextual interfaces require an original point')
        if point.namespace.source_body is None:return Neighborhood()
        calls={};notes={};edges={}
        body=self._body(point.namespace);selected=self._calls(body)
        if outgoing and type(point.node)is OperationID and point.node in selected:
            calls[point]=None
        if outgoing and type(point.node)is UseID:
            operation=body.index.use_site(point.node).operation
            if operation in selected:calls[Point(point.namespace,operation,point.context)]=None
        if not outgoing and type(point.node)is ValueID:
            origin=body.index.values[point.node].origin
            if type(origin)is BlockArgument:
                from scar.ir.program.model import CompletionResult
                for operation in selected:
                    term=body.index.operations[operation].terminator
                    for edge in term.successors if term is not None else ():
                        if (edge.block==origin.block and origin.index<len(edge.arguments)
                                and type(edge.arguments[origin.index])is CompletionResult):
                            calls[Point(point.namespace,operation,point.context)]=None
        frames=context_frames(point.context)
        formal=(not outgoing and body.interface is not None and
            (point.node==body.graph.root or any(port.value==point.node for port in body.interface.parameters)))
        terminal=False
        if outgoing and type(point.node)is UseID:
            operation=body.index.operations[body.index.use_site(point.node).operation]
            term=operation.terminator
            terminal=term is not None and term.kind is TerminatorKind.RETURN and bool(term.values) and term.values[0]==point.node
        if frames and (formal or terminal):
            frame=frames[-1]
            if frame.target==point.namespace and frame.callee_use is not None:
                popped=context_pop(point.context,(frame,))
                if popped.matched:
                    parent=Point(frame.caller,frame.operation,popped.remaining)
                    if frame.operation in self._calls(self._body(parent.namespace)):calls[parent]=None
        if (formal or terminal) and not calls:
            frontier=Frontier(point,'CONTEXTUAL_CALL_ENTRY_CONTEXT_OPEN'if formal else 'CONTEXTUAL_CALL_RETURN_CONTEXT_OPEN',
                'This original function interface has no selected matching source CALL frame; absence of a caller route is not an empty dependency proof.')
            notes[frontier]=None
        for call in calls:
            resolution=self.resolve(call)
            notes.update((frontier,None)for frontier in resolution.frontiers)
            for target in resolution.targets:
                for edge in self._edges(target):
                    if (edge.source if outgoing else edge.target)==point:edges[edge]=None
        return Neighborhood(tuple(edges),tuple(notes),not notes)

    def incoming(self, point):return self._neighbors(point,False)
    def outgoing(self, point):return self._neighbors(point,True)

    def close(self):
        self._last_receiver_evidence = None
        self._last_class_sources = None
        self.type_calls.clear()
        self.contextual_modules.clear()
        self.modules._facts.clear()
        self.session.close()

    def measurements(self):
        return {**self.session.measurements(),**dict(self.counts),
            'type_call_implementations': self.type_calls.measurements(),
            'contextual_module_bindings': self.contextual_modules.measurements(),
            'module_index_retained_input_bytes':sum(namespace.graph.decoded_bytes for namespace in self.modules._facts),
            'module_index_retention_budget_bytes':self.modules.max_cached_bytes,
            'module_index_shares_original_objects':True,
            'module_index_budget_separate_from_reader_cache':True,
            'closed_call_activation_used':False}


# Normalized, strictly typed codecs for derived contextual-call queries.
#
# Storage coordinates in the local tables never become program identities.  The
# expected Python schema selects every record class; wire record names cannot
# import or select constructors.  No original artifact is read or executed here.
_TABLES = tuple(_Tables().tables)
_TOP = {'schema', 'schema_version', 'resolution', 'interface_edges',
        'target_executed', 'target_resolution_complete'}
_MAX_BYTES = 256 * 1024 * 1024
_MAX_DEPTH = 192
_MAX_ITEMS = 2_000_000
_POINT_NODE_TYPES = {kind.__name__: kind for kind in NODE_TYPES}
_REFERENCES = {GraphNamespace: ('namespace', 'namespaces'),
    DependencyFrame: ('frame', 'frames'), OpenDependencyContext: ('context', 'contexts'),
    DependencyPoint: ('point', 'points'), DependencyCondition: ('condition', 'conditions'),
    DependencyEvidence: ('evidence', 'evidence'), DependencyEdge: ('edge', 'edges'),
    SourceArtifactLocator: ('locator', 'locators')}


def _object(value, names, detail):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError(detail + ' fields mismatch')
    return value


def _array(value, detail):
    if type(value) is not list:
        raise ValueError(detail + ' must be an array')
    return value


def _wire_id(kind, value):
    if type(value) is not str or not value.startswith(kind.prefix + ':'):
        raise ValueError('original typed ID namespace mismatch')
    payload = value[len(kind.prefix) + 1:]
    return decode(kind, {'kind': kind.prefix, 'value': payload, 'wire': value})


def _preflight(document):
    """Reject host-object cycles and non-JSON values before either codec."""
    pending = [(document, 0, False)]
    active, count = set(), 0
    while pending:
        value, depth, leaving = pending.pop()
        if leaving:
            active.remove(id(value)); continue
        count += 1
        if count > _MAX_ITEMS or depth > _MAX_DEPTH:
            raise ValueError('contextual query structural budget exhausted')
        if type(value) in (dict, list):
            if id(value) in active:
                raise ValueError('cyclic contextual query wire object')
            active.add(id(value)); pending.append((value, depth, True))
            if type(value) is dict:
                if any(type(key) is not str for key in value):
                    raise ValueError('contextual query object keys must be strings')
                pending.extend((item, depth + 1, False) for item in value.values())
            else:
                pending.extend((item, depth + 1, False) for item in value)
        elif value is None or type(value) in (str, int, bool):
            pass
        elif type(value) is float and math.isfinite(value):
            pass
        else:
            raise ValueError('contextual query wire contains a non-JSON value')


def _resolution(value):
    if type(value) is not ContextualCallResolution:
        raise TypeError('expected ContextualCallResolution')
    if type(value.call) is not DependencyPoint or type(value.call.node) is not OperationID:
        raise ValueError('contextual resolution needs its original CALL operation point')
    if type(value.targets) is not tuple or type(value.frontiers) is not tuple or type(value.enumerated) is not bool:
        raise ValueError('contextual resolution has invalid typed rows/scope')
    from scar.ir.program.contextual_calls import ContextualCallTarget
    if any(type(target) is not ContextualCallTarget or target.call != value.call for target in value.targets):
        raise ValueError('contextual target belongs to another original CALL')
    if any(type(frontier) is not DependencyFrontier for frontier in value.frontiers):
        raise ValueError('contextual resolution has an untyped frontier')
    value.call.__post_init__()
    for target in value.targets:
        target.__post_init__()
    for frontier in value.frontiers:
        frontier.__post_init__()


def _interface_edges(resolution):
    return tuple(edge for target in resolution.targets for edge in target.dependency_edges())


def export_contextual_call(resolution):
    """Pack each original endpoint, condition and evidence record once."""
    _resolution(resolution)
    table = _Tables()
    packed = table.pack(resolution)
    edges = table.pack(_interface_edges(resolution))
    return {'schema': 'scar.contextual-calls', 'schema_version': 2,
        **table.tables, 'resolution': packed, 'interface_edges': edges,
        'target_executed': False, 'target_resolution_complete': False}


class _Reader:
    def __init__(self, document):
        self.document = document
        self.cache, self.states = {}, {}
        self.schemas = {}
        self.depth = 0
        for name in _TABLES:
            rows = _array(document[name], name)
            self.cache[name] = [None] * len(rows)
            self.states[name] = [0] * len(rows)

    def ref(self, name, index):
        if type(index) is not int or not 0 <= index < len(self.cache[name]):
            raise ValueError('invalid ' + name + ' table reference')
        state = self.states[name][index]
        if state == 1:
            raise ValueError('cyclic contextual query table reference')
        if state == 2:
            return self.cache[name][index]
        if self.depth >= _MAX_DEPTH:
            raise ValueError('contextual query reference depth exhausted')
        self.states[name][index] = 1; self.depth += 1
        try:
            result = self.row(name, self.document[name][index])
        except BaseException:
            self.states[name][index] = 0
            raise
        finally:
            self.depth -= 1
        self.cache[name][index] = result; self.states[name][index] = 2
        return result

    def row(self, name, row):
        if name == 'locators':
            return decode(SourceArtifactLocator, row)
        if name == 'namespaces':
            _object(row, ('graph', 'representation', 'root', 'source_body'), 'namespace')
            graph = self.ref('locators', row['graph'])
            root = _wire_id(RegionID, row['root'])
            if type(row['representation']) is not str:
                raise ValueError('namespace representation must be a string')
            body = row['source_body']
            if body is not None:
                _object(body, ('definition', 'entry', 'source'), 'namespace source body')
                definition = _wire_id(OperationDefinitionID, body['definition'])
                source = decode(SourceReference | None, body['source'])
                body = SourceBodyNamespace(graph, row['representation'], definition,
                    source, decode(str, body['entry']), root)
            return GraphNamespace(graph, row['representation'], root, body)
        if name == 'frames':
            _object(row, ('caller', 'operation', 'callee_use', 'callee_value', 'target', 'definition', 'entry'), 'frame')
            return DependencyFrame(self.ref('namespaces', row['caller']),
                _wire_id(OperationID, row['operation']),
                None if row['callee_use'] is None else _wire_id(UseID, row['callee_use']),
                None if row['callee_value'] is None else _wire_id(ValueID, row['callee_value']),
                self.ref('namespaces', row['target']),
                None if row['definition'] is None else _wire_id(OperationDefinitionID, row['definition']),
                None if row['entry'] is None else _wire_id(RegionID, row['entry']))
        if name == 'contexts':
            _object(row, ('frames', 'unknown_prefix'), 'context')
            if type(row['unknown_prefix']) is not bool:
                raise ValueError('context unknown_prefix must be an exact bool')
            frames = tuple(self.ref('frames', index) for index in _array(row['frames'], 'context frames'))
            if row['unknown_prefix']:
                return OpenDependencyContext(frames)
            # The original helper validates a closed frame chain without
            # converting an unknown prefix into an empty root selection.
            from scar.ir.program.dependencies import context_frames
            context_frames(frames)
            return frames
        if name == 'points':
            _object(row, ('namespace', 'node', 'node_type', 'context', 'visited'), 'point')
            kind = _POINT_NODE_TYPES.get(row['node_type']) if type(row['node_type']) is str else None
            if kind is None:
                raise ValueError('unknown original point ID type')
            # Contextual resolution is not a traversal. _Tables is created
            # without a visited set, so accepting True would discard data.
            if row['visited'] is not False:
                raise ValueError('contextual-call point cannot declare traversal visitation')
            context = None if row['context'] is None else self.ref('contexts', row['context'])
            return DependencyPoint(self.ref('namespaces', row['namespace']), _wire_id(kind, row['node']), context)
        if name == 'conditions':
            _object(row, ('domain', 'name', 'subject', 'index', 'operands'), 'condition')
            return DependencyCondition(decode(str, row['domain']), decode(str, row['name']),
                None if row['subject'] is None else self.ref('points', row['subject']),
                decode(int | None, row['index']),
                tuple(self.ref('points', index) for index in _array(row['operands'], 'condition operands')))
        if name == 'evidence':
            _object(row, ('rule', 'sources', 'anchors'), 'evidence')
            return DependencyEvidence(decode(str, row['rule']),
                tuple(self.ref('locators', index) for index in _array(row['sources'], 'evidence sources')),
                tuple(self.ref('points', index) for index in _array(row['anchors'], 'evidence anchors')))
        if name == 'edges':
            _object(row, ('source', 'target', 'kind', 'meaning', 'strength', 'evidence', 'conditions'), 'edge')
            return DependencyEdge(self.ref('points', row['source']), self.ref('points', row['target']),
                decode(DependencyKind, row['kind']), decode(DependencyMeaning, row['meaning']),
                tuple(self.ref('evidence', index) for index in _array(row['evidence'], 'edge evidence')),
                tuple(self.ref('conditions', index) for index in _array(row['conditions'], 'edge conditions')),
                decode(DependencyStrength, row['strength']))
        if name == 'frontier_reasons':
            _object(row, ('kind', 'detail'), 'frontier reason')
            kind, detail = decode(str, row['kind']), decode(str, row['detail'])
            if not kind or not detail:
                raise ValueError('frontier reason cannot be empty')
            return kind, detail
        raise ValueError('unknown contextual query table')

    def unpack(self, expected, wire, depth=0):
        if depth > _MAX_DEPTH:
            raise ValueError('contextual record depth exhausted')
        origin, args = get_origin(expected), get_args(expected)
        if expected is Any:
            raise TypeError('contextual query schema must not expose Any')
        if origin in (Union, UnionType):
            for choice in args:
                try:
                    return self.unpack(choice, wire, depth + 1)
                except (TypeError, ValueError):
                    pass
            raise ValueError('contextual value does not match its expected union')
        if origin in (tuple, list):
            rows = _array(wire, 'typed sequence')
            if origin is tuple and not (len(args) == 2 and args[1] is Ellipsis):
                if len(rows) != len(args):
                    raise ValueError('incorrect fixed contextual tuple length')
                return tuple(self.unpack(kind, item, depth + 1) for kind, item in zip(args, rows))
            if len(args) != (2 if origin is tuple else 1):
                raise TypeError('unsupported untyped contextual sequence')
            result = [self.unpack(args[0], item, depth + 1) for item in rows]
            return tuple(result) if origin is tuple else result
        if origin is dict:
            if type(wire) is not dict or len(args) != 2:
                raise ValueError('typed contextual mapping fields mismatch')
            return {self.unpack(args[0], key, depth + 1): self.unpack(args[1], value, depth + 1)
                    for key, value in wire.items()}
        reference = _REFERENCES.get(expected)
        if reference is not None:
            tag, table = reference
            _object(wire, (tag,), 'typed ' + tag + ' reference')
            result = self.ref(table, wire[tag])
            if type(result) is not expected:
                raise ValueError('contextual reference points to another expected type')
            return result
        if expected is DependencyFrontier:
            _object(wire, ('point', 'reason'), 'frontier')
            kind, detail = self.ref('frontier_reasons', wire['reason'])
            return DependencyFrontier(self.ref('points', wire['point']), kind, detail)
        if expected is PythonLiteral:
            _object(wire, ('literal',), 'immutable literal')
            return decode(PythonLiteral, wire['literal'])
        if isinstance(expected, type) and issubclass(expected, Identifier):
            _object(wire, ('original_id',), 'original typed ID')
            return _wire_id(expected, wire['original_id'])
        if isinstance(expected, type) and issubclass(expected, Enum):
            return decode(expected, wire)
        if isinstance(expected, type) and is_dataclass(expected):
            schema = self.schemas.get(expected)
            if schema is None:
                schema = fields(expected), get_type_hints(expected)
                self.schemas[expected] = schema
            members, hints = schema
            _object(wire, ('record_type', *(field.name for field in members)), expected.__name__)
            if type(wire['record_type']) is not str or wire['record_type'] != expected.__name__:
                raise ValueError('contextual record discriminator differs from expected class')
            return expected(**{field.name: self.unpack(hints[field.name], wire[field.name], depth + 1)
                               for field in members})
        return decode(expected, wire)

    def validate_tables(self):
        # Unreferenced malformed rows must not hide behind an unused index.
        for name in _TABLES:
            seen = set()
            for index in range(len(self.cache[name])):
                value = self.ref(name, index)
                if value in seen:
                    raise ValueError('duplicate normalized ' + name + ' table record')
                seen.add(value)


def read_contextual_call(document):
    """Read strict raw v1 or normalized v2 into original strong Python types.

    All source declarations, open prefixes, ordered condition operands and
    caller/factory contexts survive.  Local indices never escape this reader.
    """
    if type(document) in (str, bytes):
        if len(document.encode('utf-8') if type(document) is str else document) > _MAX_BYTES:
            raise ValueError('contextual query byte budget exhausted')
        document = loads(document)
    _preflight(document)
    if (type(document) is not dict or document.get('schema') != 'scar.contextual-calls'
            or type(document.get('schema_version')) is not int):
        raise ValueError('unsupported contextual-call query schema')
    version = document['schema_version']
    _object(document, _TOP | set(_TABLES) if version == 2 else _TOP, 'contextual-call query')
    if document['target_executed'] is not False or document['target_resolution_complete'] is not False:
        raise ValueError('contextual query cannot assert execution or target-set closure')
    if version == 1:
        resolution = decode(ContextualCallResolution, document['resolution'])
        edges = decode(tuple[DependencyEdge, ...], document['interface_edges'])
    elif version == 2:
        reader = _Reader(document)
        resolution = reader.unpack(ContextualCallResolution, document['resolution'])
        edges = reader.unpack(tuple[DependencyEdge, ...], document['interface_edges'])
        reader.validate_tables()
    else:
        raise ValueError('unsupported contextual-call query version')
    _resolution(resolution)
    if edges != _interface_edges(resolution):
        raise ValueError('stored contextual interface edges differ from their original typed target ports')
    return resolution, edges


__all__ = [
    'ContextualCallResolution',
    'ContextualCallResolver',
    'export_contextual_call',
    'read_contextual_call',
    'REFERENCE_KINDS',
]
