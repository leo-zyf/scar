"""Call inputs."""

from collections import defaultdict, deque
from scar.analysis.source_reference_slice import ReferenceSlicer, source_port, DECLARED
from scar.ir.program import call_inputs as X, source_value_routes as R, normal_graph as T, normal_graph as F, source_productions as P, call_inputs as D, view as V, model as M
from dataclasses import replace, dataclass
from scar.analysis.program_view import _digest
from scar.ir.program.ids import LocationID, ValueID
from scar.ir.program.dependencies import (
    DependencyPoint as Point,
    DependencyEdge as Edge,
    DependencyCondition as Condition,
    DependencyEvidence as Evidence,
    DependencyKind as Kind,
    DependencyMeaning as Meaning,
    DependencyStrength as Strength,
    DependencyNeighborhood as Neighborhood,
    DependencyFrontier as Frontier,
    DependencyEdge,
    DependencyKind,
)
from scar.analysis.source_reference_consumers import argument_points
from scar.ir.program.normal_graph import PortKind
from scar.ir.program.binding_ports import ArgumentRole


# Trace references through selected incoming source interfaces.
#
# An unselected outer caller is a boundary. It never licenses merging every
# actual ever passed to the same function. Calls returning references push the
# exact candidate interface; its own parameters pop that same interface only.
class ContextReferenceEngine:
    def __init__(self, snapshot):
        self.snapshot = snapshot; self.slicer = ReferenceSlicer(snapshot)
        self.interfaces = []; self.ids = {}; self.indexed = set(); self.returns = defaultdict(list)
        self.by_result = defaultdict(list); self.locals = {}

    def admit(self, evidence, link):
        """Register a certified historical or declared source interface."""
        if evidence in self.ids:
            identity = self.ids[evidence]
            if self.interfaces[identity].link != link: raise ValueError('source interface evidence has conflicting records')
            return identity
        identity = len(self.interfaces); self.ids[evidence] = identity
        self.interfaces.append(X.CallInterfaceRecord(evidence, link)); self.by_result[link.result].append(identity)
        if link.mode not in ('SOURCE_CONSTRUCTION', 'SOURCE_INIT', 'SOURCE_NEW', 'CREATE_DEFERRED_BODY'):
            for origin, result in link.returns:
                if result == link.result and link.body is not None and link.source_callable is not None:
                    self.returns[result].append((identity, origin))
        return identity

    def index(self, units):
        """One full receipt-checked scan for each newly requested unit batch."""
        units = set(units)-self.indexed
        if not units: return
        for _, link, evidence in self.snapshot.rows('calls', lambda w: w['result']['unit'] in units, T.CallLink):
            self.admit(evidence, link)
        self.indexed.update(units)

    def unit(self, identity):
        if identity not in self.locals:
            unit, _, aliases, reads, writes, formals, terminals = self.slicer.unit(identity)
            owners = defaultdict(set)
            for group in unit.scope_groups:
                for span in group.tables:
                    if span.table is F.FactTable.NAMES:
                        for i in range(span.start, span.stop): owners[i].add(group.owner)
            self.locals[identity] = unit, aliases, reads, writes, formals, terminals, owners, {c.result for c in unit.calls}
        return self.locals[identity]

    def solve(self, requests, *, max_states=4000, max_depth=24):
        """Batch (incoming-interface ordinal, queried GlobalPort) requests.

        Indexing is deferred and batched when a reference reaches a previously
        unread call-result unit. Queries retain their worklist between scans.
        The stack contains static interface IDs, never dynamic invocations.
        """
        queries = [_Query(self, incoming, port, max_states, max_depth) for incoming, port in requests]
        while True:
            waiting = defaultdict(list)
            for query in queries: query.advance(waiting)
            if not waiting: break
            self.index(waiting)
            for unit, work in waiting.items():
                for query, state in work: query.pending.append(state)
        self.snapshot.validate_parents()
        return tuple(q.finish() for q in queries)


class _Query:
    def __init__(self, engine, incoming, port, max_states, max_depth):
        if (type(incoming) is not int or not 0 <= incoming < len(engine.interfaces)
                or type(port) is not T.GlobalPort or type(max_states) is not int or max_states < 1
                or type(max_depth) is not int or max_depth < 1):
            raise ValueError('context reference query requires an original incoming interface and positive limits')
        link = engine.interfaces[incoming].link
        if link.body is None or link.source_callable is None or link.body.unit != port.unit:
            raise ValueError('context reference seed has no matching source body')
        self.engine = engine; self.max_states = max_states; self.max_depth = max_depth
        root = X.ContextReferenceState(port, (incoming,))
        self.states = [root]; self.ids = {root: 0}; self.pending = deque((0,))
        self.edges = []; self.origins = []; self.frontiers = []; self.frontier_ids = set()

    def frontier(self, state, reason):
        if (state, reason) not in self.frontier_ids:
            self.frontier_ids.add((state, reason)); self.frontiers.append(R.ReferenceFrontier(state, reason))

    def edge(self, state, port, stack, kind, evidence, conditions):
        value = X.ContextReferenceState(port, stack)
        if value not in self.ids:
            if len(self.states) >= self.max_states:
                self.frontier(state, 'CONTEXT_REFERENCE_STATE_BUDGET_FRONTIER'); return
            self.ids[value] = len(self.states); self.states.append(value); self.pending.append(self.ids[value])
        self.edges.append(R.ReferenceTransition(state, self.ids[value], kind, tuple(evidence), tuple(conditions)))

    def advance(self, waiting):
        engine = self.engine
        while self.pending:
            index = self.pending.popleft(); state = self.states[index]; point = state.port; stack = state.interfaces
            if type(point) is not T.GlobalPort:
                self.frontier(index, 'CONSTRUCTION_ENDPOINT_REQUIRES_SEPARATE_PROTOCOL_FRAME'); continue
            unit, local = engine.slicer.local(point)
            unit, aliases, reads, writes, formals, terminals, owners, results = engine.unit(unit.unit)
            if local in results and point.unit not in engine.indexed:
                waiting[point.unit].append((self, index)); continue
            before = len(self.edges); before_frontiers = len(self.frontiers); terminal = False
            for source, ordinal in aliases.get(local, ()):
                self.edge(index, source_port(unit, source), stack, R.ReferenceTransitionKind.DECLARED_ALIAS,
                    (engine.slicer.fact(unit, 'aliases', ordinal),), DECLARED)
            for namespace, ordinal in reads.get(local, ()):
                if namespace.kind is not F.NamespaceKind.LEXICAL:
                    self.frontier(index, 'NON_LEXICAL_LOOKUP_REQUIRES_FRAME_NAMESPACE'); continue
                if len(owners[ordinal]) != 1:
                    self.frontier(index, 'LEXICAL_READ_CODE_SCOPE_UNRESOLVED'); continue
                accepted = False
                for source, writer in writes.get(namespace, ()):
                    if owners[writer] != owners[ordinal]:
                        self.frontier(index, 'CAPTURE_CREATION_FRAME_NOT_SELECTED'); continue
                    accepted = True
                    self.edge(index, source_port(unit, source), stack, R.ReferenceTransitionKind.LEXICAL_MAY_BINDING,
                        (engine.slicer.fact(unit, 'names', writer), engine.slicer.fact(unit, 'names', ordinal)),
                        (*DECLARED, 'SAME_CODE_SCOPE_LEXICAL_WRITE_SELECTED_NOT_CURRENT_VERSION'))
                if not accepted: self.frontier(index, 'LEXICAL_BINDING_PRODUCER_UNRESOLVED')
            if local in formals:
                fn = formals[local]
                if not stack:
                    self.frontier(index, 'OUTER_CALL_CONTEXT_NOT_SELECTED')
                else:
                    record = engine.interfaces[stack[-1]]; link = record.link
                    if (link.body != T.Body(unit.unit, fn.template, fn.body)
                            or link.source_callable != source_port(unit, fn.value)):
                        self.frontier(index, 'FORMAL_BELONGS_TO_UNSELECTED_CAPTURE_FRAME')
                    else:
                        routes = [r for r in link.parameters if r.formal == point]
                        if not routes: self.frontier(index, 'SELECTED_INTERFACE_HAS_NO_PARAMETER_ROUTE')
                        for route in routes:
                            if route.kind not in (T.RouteKind.ARGUMENT, T.RouteKind.INITIAL_DEFAULT):
                                self.frontier(index, 'NON_REFERENCE_PARAMETER_ROUTE_'+route.kind.value); continue
                            for actual in route.actuals:
                                guards = (*link.conditions, 'ONLY_SELECTED_INCOMING_INTERFACE_PARAMETER_ROUTE',
                                    'PARENT_CALL_CONTEXT_SELECTED_NOT_RUNTIME_INVOCATION')
                                if route.kind is T.RouteKind.INITIAL_DEFAULT:
                                    guards = (*guards, 'FUNCTION_DEFAULT_VERSION_MATCHES_DECLARATION')
                                self.edge(index, actual, stack[:-1], R.ReferenceTransitionKind(route.kind.value),
                                    (record.evidence,), guards)
            for identity, origin in engine.returns.get(point, ()):
                if len(stack) >= self.max_depth:
                    self.frontier(index, 'CONTEXT_REFERENCE_CALL_DEPTH_FRONTIER'); continue
                record = engine.interfaces[identity]
                self.edge(index, origin, (*stack, identity), R.ReferenceTransitionKind.DATA_RETURN_ENTER_CALL,
                    (record.evidence,), (*record.link.conditions, 'RETURN_AND_PARAMETER_USE_SAME_SELECTED_INTERFACE'))
            if local in results and not engine.returns.get(point):
                self.frontier(index, 'SOURCE_CALL_RESULT_OR_PROTOCOL_UNRESOLVED')
            if local in terminals:
                kind, table, ordinal = terminals[local]
                self.origins.append(R.ReferenceOrigin(index, R.ReferenceOriginKind(kind), None,
                    (engine.slicer.fact(unit, table, ordinal),), DECLARED)); terminal = True
            if len(self.edges) == before and not terminal and len(self.frontiers) == before_frontiers:
                self.frontier(index, 'REFERENCE_PRODUCER_OR_FRAME_INTERFACE_UNRESOLVED')

    def finish(self):
        return X.ContextReferenceSlice(tuple(self.states), tuple(self.edges), tuple(self.origins), tuple(self.frontiers))


# Current callable-default state feeding source argument binding.
#
# The original creation expression is provenance, not a current-value alias.
# Caller world and callable occurrence select a symbolic logical default slot;
# the native defaults tuple/dict layout remains outside this macro interface.
class FunctionDefaultStateProvider:
    def __init__(self,view):
        self.view=view;self.addresses={};self.states={};self.reads=[];self.functions={}
        self.namespaces={};self.in_edges=defaultdict(list);self.out_edges=defaultdict(list)
        view.register(self)

    @staticmethod
    def _key(point):return point.namespace,point.node

    def _function(self,definition):
        if definition.unit not in self.functions:
            source=self.view.unit(definition.unit)
            self.functions[definition.unit]={source_port(source.facts,f.value):f for f in source.facts.functions}
        try:return self.functions[definition.unit][definition]
        except KeyError:raise ValueError('default binding lacks its original function creation fact') from None

    def _edge(self,source,target,kind,meaning,condition,locator):
        edge=Edge(source,target,kind,meaning,
            (Evidence('scar.source-call.current-default-state.v1',(locator,),(source,target)),),
            (condition,),Strength.MAY)
        if edge not in self.out_edges[self._key(source)]:
            self.out_edges[self._key(source)].append(edge);self.in_edges[self._key(target)].append(edge)

    def read(self,interface,route,frame,world):
        link=interface.link;function=self._function(link.source_callable)
        definition_source=self.view.unit(link.source_callable.unit)
        parameters=tuple(source_port(definition_source.facts,i) for i in function.formals[1:-1])
        matches=[i for i,p in enumerate(parameters) if p==route.formal]
        if len(matches)!=1:raise ValueError('default route does not select one original formal')
        index=matches[0];parameter=function.environment.parameters[index]
        initial=function.defaults[index]
        if parameter.default_source is None or initial is None:
            raise ValueError('default route lacks original creation expression')
        initial=source_port(definition_source.facts,initial)
        if route.actuals!=(initial,) or route.kind not in (T.RouteKind.INITIAL_DEFAULT,T.RouteKind.CONDITIONAL_DEFAULT):
            raise ValueError('default route differs from its original signature/default provenance')
        source=self.view.namespaces[frame.caller];namespace=source.namespace
        self.namespaces[namespace]=source
        identity=LocationID(_digest(('scar.function-default-slot.v1',source.basis,
            link.callee,link.source_callable,index,parameter)))
        slot=D.FunctionDefaultSlot(identity,link.callee,link.source_callable,index,function.parameter_names[index],parameter)
        location=Point(namespace,identity,None);self.addresses.setdefault(self._key(location),slot)
        value_id=ValueID(_digest(('scar.function-default-state.v1',identity,world)))
        state=D.FunctionDefaultState(value_id,identity,world)
        point=Point(namespace,value_id,None);self.states.setdefault(self._key(point),state)
        record=D.FunctionDefaultRead(frame.operation,identity,value_id,route.formal,initial,route.kind)
        if record not in self.reads:self.reads.append(record)
        operation=Point(namespace,frame.operation,None)
        callable=self.view.source_port(link.callee);before=self.view.source_port(world)
        if self.view.get(before).value_type.kind is not M.ValueKind.STATE:
            raise ValueError('default projection must read the original pre-call world')
        selected=Condition('scar.source-call.default','SELECTED_FUNCTION_INSTANCE_PARAMETER_AND_PRECALL_WORLD',
            operation,index,(callable,before,location))
        self._edge(callable,location,Kind.ENVIRONMENT,Meaning.STATE_INTERFACE,selected,source.locator)
        self._edge(location,point,Kind.STATE_LOCATION,Meaning.STRUCTURE,selected,source.locator)
        self._edge(before,point,Kind.PROJECTION,Meaning.STATE_INTERFACE,selected,source.locator)
        return point

    def describe(self,point):
        key=self._key(point)
        if key not in self.addresses and key not in self.states:return None
        self.namespaces[point.namespace].check(point)
        if key in self.addresses:
            return V.ProgramNode(point,V.ProgramEntityKind.STATE_LOCATION,V.ProgramNodeRole.STATE_LOCATION,None,None)
        state=self.states[key]
        return V.ProgramNode(point,V.ProgramEntityKind.VALUE,V.ProgramNodeRole.CORE_VALUE,None,
            M.ValueType(M.ValueKind.STATE,location=state.address))

    def original(self,point):
        if self.describe(point) is None:raise ValueError('default-state endpoint is not admitted')
        key=self._key(point)
        return self.addresses[key] if key in self.addresses else self.states[key]

    def expand(self,point):
        record=self.original(point)
        if type(record) is not D.FunctionDefaultSlot:return ()
        return tuple(Point(namespace,value.id,point.context) for (namespace,_),value in self.states.items()
            if namespace==point.namespace and value.address==record.id)

    def explain(self,point):
        record=self.original(point)
        return dict(node=self.describe(point),original=record,
            reads=tuple(r for r in self.reads if r.address==(record.id if type(record) is D.FunctionDefaultSlot else record.address)),
            creation_context_selected=False,current_equals_initial_proven=False,
            tensor_content_captured=False,physical_layout_asserted=False)

    def _neighbors(self,point,incoming):
        key=self._key(point);rows=(self.in_edges if incoming else self.out_edges).get(key,())
        if self.describe(point) is None:
            if not rows:return Neighborhood()
            self.view.get(point)
        edges=tuple(replace(edge,source=replace(edge.source,context=point.context),
            target=replace(edge.target,context=point.context),conditions=tuple(replace(c,
                subject=None if c.subject is None else replace(c.subject,context=point.context),
                operands=tuple(replace(p,context=point.context) for p in c.operands)) for c in edge.conditions)) for edge in rows)
        notes=(() if key not in self.states else (Frontier(point,
            'FUNCTION_DEFAULT_CREATION_INSTANCE_AND_MUTATION_HISTORY_NOT_SELECTED',
            'Current default state is read at call time; initial provenance does not prove an unchanged slot or object contents.'),))
        return Neighborhood(edges,notes)

    def incoming(self,point):return self._neighbors(point,True)
    def outgoing(self,point):return self._neighbors(point,False)


# Query evaluated call operands and their original F4 input edges.
@dataclass(frozen=True, slots=True)
class SourceCallOperand:
    role: ArgumentRole
    ordinal: int
    keyword: str | None
    edge: DependencyEdge


def call_operands(view, unit, call_index, *, context=None):
    """Return reference operand routes in the original call ABI order."""
    provider = view.unit(unit)
    facts = provider.facts
    if type(call_index) is not int or not 0 <= call_index < len(facts.calls):
        raise ValueError('expected an original call fact ordinal')
    call = facts.calls[call_index]
    result = view.source_port(source_port(facts, call.result), context=context)
    operation = view.get(result).owner
    inputs = {}
    for point in provider.expand(operation):
        port = view.original(point)
        if getattr(port, 'kind', None) is PortKind.ARGUMENT:
            inputs[port.ordinal] = point
    operands = []
    for index, (role, ordinal, keyword, value) in enumerate(argument_points(call)):
        actual = view.source_port(source_port(facts, value), context=context)
        use = inputs.get(index)
        if use is None:
            raise ValueError('original call operand lacks its declared input port')
        edges = view.dependencies.incoming(use, kinds=(DependencyKind.VALUE_USE,)).edges
        if len(edges) != 1 or edges[0].source != actual:
            raise ValueError('original call fact disagrees with its value-to-operand edge')
        operands.append(SourceCallOperand(role, ordinal, keyword, edges[0]))
    return tuple(operands)
