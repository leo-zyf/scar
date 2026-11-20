"""Slice original constructor value, state, and control routes to controller fields."""

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path

from scar.analysis.bounded_entry_calls import _admit_entry_calls
from scar.analysis.build import BuildSession
from scar.analysis.replay_transaction import replay_transaction
from scar.analysis.session_caller_effects import _source
from scar.analysis.source_address_paths import AddressResolver
from scar.analysis.source_controller_effects import _Observed
from scar.analysis.source_controller_initialization import SCHEMA as INITIALIZATION, check as check_initialization
from scar.analysis.source_reference_slice import ReferenceSlicer, source_port
from scar.analysis.source_value_flow import project_value_flow
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.ir.literals import PythonLiteral
from scar.ir.program import normal_graph as G, normal_links as F, normal_targets as T, source_productions as P
from scar.ir.program.call_edges import InvocationKey
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.model import ValueKind, ValueType
from scar.ir.program.source_address_paths import AddressPath, AddressRootKind, FieldSelector
from scar.ir.record_codec import decode, encode


SCHEMA = 'scar.source-controller-startup-slice'


def _only(rows, detail):
    values = tuple(rows)
    if len(values) != 1:
        raise ValueError(detail)
    return values[0]


def _entry(port):
    return G.Port(port.template, port.owner, G.PortKind.ENTER)


def _is_state(value):
    return type(value) is ValueType and value.kind is ValueKind.STATE


def _path(row):
    path = decode(AddressPath,row) if type(row) is dict else row
    if (path.frontier is None and path.root_kind is AddressRootKind.FUNCTION_PARAMETER
            and all(type(part) is FieldSelector for part in path.selectors)):
        return path.root,tuple(part.name for part in path.selectors)
    return None


def _slot(syntax, slot):
    if slot.group != 'FIELD':
        return None
    row = syntax.nodes[slot.occurrence].fields[slot.ordinal]
    for i in slot.item_path:
        row = row.items[i]
    return row


def _literal_field(syntax, value):
    """Recognize only original scalar or simple signed-literal declarations."""
    if value is None:
        return None
    if value.kind=='SCALAR':
        return value.scalar
    if value.kind!='NODE':
        return None
    node = syntax.nodes[value.reference]
    schema = next(r for r in syntax.profile.ast_schema if r.name==node.constructor)
    fields = dict(zip(schema.fields,node.fields,strict=True))
    if node.constructor=='Constant' and fields['value'].kind=='SCALAR':
        return fields['value'].scalar
    if node.constructor=='UnaryOp' and fields['operand'].kind=='NODE':
        operand = syntax.nodes[fields['operand'].reference]
        operator = syntax.nodes[fields['op'].reference].constructor
        if operand.constructor=='Constant' and operator in ('UAdd','USub'):
            source = _literal_field(syntax,fields['operand'])
            if source is not None:
                number = source.to_python()
                if type(number) in (int,float):
                    return PythonLiteral.from_python(-number if operator=='USub' else number,exact=True)
    return None


def _normal_profile(flow, anchors):
    """Select balanced normal paths that contain the admitted observed CALL order."""
    body = flow.body
    head = G.Port(body.template,body.region,G.PortKind.ENTER)
    if len(anchors)!=len(set(anchors)):
        raise ValueError('initialization slice requires an acyclic original call profile')
    selected = {port:i for i,port in enumerate(anchors)}
    edges = defaultdict(list)
    for edge in flow.edges:
        if edge.kind is not G.EdgeKind.CONTROL or C.RAISE in (edge.source.completion,edge.target.completion):
            continue
        if edge.transition in (G.Transition.RESUME,G.Transition.BACKEDGE):
            raise ValueError('initialization loop/resumption needs a separate scoped slice')
        edges[edge.source].append(edge)
    start = (head,(),0)
    pending = deque((start,))
    visited = {start}
    reverse = defaultdict(list)
    accepted = []
    while pending:
        current = pending.popleft()
        node, stack, matched = current
        if (node.owner==body.region and node.kind is G.PortKind.EXIT
                and node.completion in (C.RETURN,C.FALLTHROUGH) and not stack):
            if matched==len(anchors):
                accepted.append(current)
            continue
        if node in selected:
            if selected[node] != matched:
                continue
            matched += 1
        for edge in edges.get(node,()):
            following = stack
            if edge.transition is G.Transition.CALL:
                if edge.site is None or len(stack)>=128:
                    raise ValueError('malformed or excessive original initialization call context')
                following = (*stack,edge.site)
            elif edge.transition is G.Transition.RETURN:
                if not stack or stack[-1]!=edge.site:
                    continue
                following = stack[:-1]
            successor = edge.target,following,matched
            reverse[successor].append((current,edge))
            if successor not in visited:
                if len(visited)>=50_000:
                    raise ValueError('initialization normal control profile exceeds source cap')
                visited.add(successor)
                pending.append(successor)
    if not accepted:
        raise ValueError('observed initialization CALL order has no original normal F4 path')

    valid = set(accepted)
    pending.extend(accepted)
    while pending:
        for predecessor,_ in reverse[pending.popleft()]:
            if predecessor not in valid:
                valid.add(predecessor)
                pending.append(predecessor)
    outgoing = defaultdict(list)
    incoming = {}
    for node in sorted(valid,key=repr):
        incoming[node] = sum(parent in valid for parent,_ in reverse[node])
        for parent,edge in reverse[node]:
            if parent in valid:
                outgoing[parent].append((node,edge))

    guards = {start:set()}
    by_port = {}
    rank = {}
    last_rank = {}
    pending.append(start)
    order = 0
    while pending:
        current = pending.popleft()
        node = current[0]
        order += 1
        rank.setdefault(node,order)
        last_rank[node] = order
        by_port.setdefault(node,set(guards[current])).intersection_update(guards[current])
        for successor,edge in outgoing[current]:
            conditions = set(guards[current])
            if edge.condition is not None:
                conditions.add((edge.condition,edge.when))
            guards.setdefault(successor,conditions).intersection_update(conditions)
            incoming[successor] -= 1
            if incoming[successor]==0:
                pending.append(successor)
    if order != len(valid):
        raise ValueError('original selected initialization graph has an unresolved cycle')
    global_guards = set.intersection(*(guards[exit_] for exit_ in accepted))
    return by_port,rank,last_rank,global_guards


def _provider_candidates(unit, slicer, paths):
    """Select original import/builtin routes; their current binding is a guard."""
    by_value = defaultdict(list)
    writes = defaultdict(list)
    for ordinal,row in enumerate(unit.facts.names):
        if row.write:
            writes[row.namespace].append(row.value)
        else:
            by_value[row.value].append((ordinal,row.namespace))
    _,_,aliases,*_ = slicer.unit(unit.unit)
    imports = {row.value:(i,row) for i,row in enumerate(unit.facts.imports)}
    proposals = {}
    for raw in paths:
        path = decode(AddressPath,raw)
        if any(type(item) is not FieldSelector for item in path.selectors):
            continue
        _, root = slicer.local(path.root)
        suffix = tuple(item.name for item in path.selectors)
        for name_ordinal,namespace in by_value[root]:
            name_fact = slicer.fact(unit.facts,'names',name_ordinal)
            if namespace.kind is F.NamespaceKind.BUILTINS:
                candidate = ('builtins',namespace.name,*suffix)
                proposals[candidate] = dict(path=candidate,kind='BUILTINS_LOOKUP',source_facts=(name_fact,))
                continue
            pending = deque(writes[namespace])
            seen = set(pending)
            while pending:
                item = pending.popleft()
                if item in imports:
                    ordinal,row = imports[item]
                    candidate = (row.module,*suffix)
                    proposals[candidate] = dict(path=candidate,kind='ORIGINAL_IMPORT_MEMBER',
                        source_facts=(name_fact,slicer.fact(unit.facts,'imports',ordinal)))
                for producer,_ in aliases[item]:
                    if producer not in seen:
                        seen.add(producer)
                        pending.append(producer)
    return [proposals[path] for path in sorted(proposals)]


class _Body:
    def __init__(self, view, slicer, profile, invocation, arguments, anchors=()):
        self.profile = profile
        self.body = decode(T.Body,profile['body'])
        self.unit = view.unit(self.body.unit)
        self.fn = _only((fn for fn in self.unit.facts.functions
                         if (fn.template,fn.body)==(self.body.template,self.body.region)),
                        'one original initializer source body required')
        self.invocation = invocation
        self.arguments = {decode(T.GlobalPort,r['formal']).port:r for r in arguments}
        self.addresses = AddressResolver(view.catalog.fact_store,self.unit.facts,self.fn.body_scope.owner)
        self.slicer = slicer
        self.flow = project_value_flow(self.unit.material,source_port(self.unit.facts,self.fn.value),self.body)
        self.guards,self.rank,self.last_rank,self.profile_guards = _normal_profile(self.flow,anchors)
        self.reverse = defaultdict(set)
        for edge in self.flow.edges:
            if edge.kind is G.EdgeKind.DATA and self.visible(edge.source) and self.visible(edge.target):
                self.reverse[edge.target].add(edge.source)
        for location in self.flow.locations:
            writes = {row.value for row in location.writes if self.visible(row.value)}
            for access in location.reads:
                if self.visible(access.value):
                    self.reverse[access.value].update(writes)
        self.operators = {}
        for operation in self.flow.operations:
            if operation.entry not in self.guards:
                continue
            args = {r.port for r in operation.inputs if not _is_state(r.value_type)}
            for result in operation.outputs:
                if not _is_state(result.value_type) and result.port.completion is C.FALLTHROUGH:
                    self.operators[result.port] = operation
                    self.reverse[result.port].update(args)
        self.calls = {decode(T.GlobalPort,row['source_result']).port:row for row in profile['calls']}
        self.operator_facts = {source_port(self.unit.facts,row.result).port:(i,row)
                               for i,row in enumerate(self.unit.facts.operations)}
        self.constants = {source_port(self.unit.facts,row.value).port:(i,row)
                          for i,row in enumerate(self.unit.facts.constants)}
        self.fields = {decode(T.GlobalPort,row['value']).port:row for row in profile['field_accesses']
                       if row['kind']=='GET' and row['value'] is not None}
        self.items = {decode(T.GlobalPort,row['value']).port:row for row in profile['item_accesses']
                      if row['kind']=='READ' and row['value'] is not None}
        self.field_events = {}
        for row in (*profile['field_accesses'],*profile['item_accesses']):
            self.field_events[row['source_fact']['locator'],row['source_fact']['artifact']['sha256']] = row

    def visible(self, port):
        return _entry(port) in self.guards


class _StartupSlice:
    def __init__(self, view, graph, observed):
        self.view,self.graph,self.observed = view,graph,observed
        self.slicer = ReferenceSlicer(view.catalog)
        self.profile = {decode(T.Body,row['body']):row for row in graph['bodies']}
        self.startup = graph['constructor']['initializer']
        self.owner_body = decode(T.Body,self.startup['body'])
        self.children = {}
        self.by_attempt = {}
        self.providers = {}
        self.syntax = {}
        selected = graph['initializer_calls']
        self.primary = _Body(view,self.slicer,self.profile[self.owner_body],
            decode(InvocationKey,self.startup['invocation']),self.startup['arguments'],
            tuple(_entry(decode(T.GlobalPort,row['original_call']['source_result']).port) for row in selected))
        self.receiver = _only((decode(T.GlobalPort,row['formal']) for row in self.startup['arguments']
                              if row['role']=='OBSERVED_IMPLICIT_RECEIVER'),
                             'startup slice needs its original owning receiver')
        for row in selected:
            if row['body'] is None or not row['arguments']:
                continue
            body = decode(T.Body,row['body'])
            site = decode(T.GlobalPort,row['original_call']['source_result'])
            context = _Body(view,self.slicer,self.profile[body],
                decode(InvocationKey,row['observation']['callee']),row['arguments'])
            self.children[context] = site
            self.by_attempt[row['observation']['attempt']['index']] = context
        self.fields = {tuple(row['path']):row['field_writers'] for row in graph['initial_receiver_fields']}

    def source_literal(self, context, slot):
        unit = context.unit.unit
        if unit not in self.syntax:
            source,syntax = _source(self.view,unit,context.fn)
            if source.code_key != context.profile['source']['code_key']:
                raise ValueError('literal source differs from the original initialization CodeID')
            self.syntax[unit] = syntax
        material = self.syntax[unit]
        return _literal_field(material,_slot(material,slot))

    def address(self, context, raw_paths):
        results = set()
        for raw in raw_paths:
            value = _path(raw)
            if value is None:
                continue
            root,selectors = value
            if context is self.primary and root==self.receiver:
                results.add(selectors)
            elif context is not self.primary and root.port in context.arguments:
                actual = context.arguments[root.port]['actual']
                if actual is None:
                    continue
                _,ordinal = self.slicer.local(decode(T.GlobalPort,actual))
                for upstream in self.primary.addresses.paths(ordinal):
                    origin = _path(upstream)
                    if origin is not None and origin[0]==self.receiver:
                        results.add((*origin[1],*selectors))
        return results

    def position(self, context, port, *, latest=False):
        step = _entry(port)
        points = context.last_rank if latest else context.rank
        if context is self.primary:
            return points.get(step,-1),0
        call = self.children[context]
        parent = self.primary.last_rank if latest else self.primary.rank
        return parent.get(_entry(call.port),-1),points.get(step,-1)

    def writer(self, row):
        context = self.primary if row.get('attempt') is None else self.by_attempt[row['attempt']['index']]
        fact = row['source_fact']
        access = context.field_events[fact['locator'],fact['artifact']['sha256']]
        operation = decode(G.Port,access['operation'])
        if operation not in context.guards:
            return None
        return context,operation,decode(T.GlobalPort,access['value']).port,access

    def provider(self, context, call):
        site = decode(T.GlobalPort,call['source_result'])
        key = context.invocation,site
        if key not in self.providers:
            candidates = _provider_candidates(context.unit,self.slicer,call['callee_paths'])
            imports = [row['path'] for row in candidates if row['kind']=='ORIGINAL_IMPORT_MEMBER']
            self.providers[key] = dict(id='provider:'+str(len(self.providers)),source_result=site,
                source_fact=call['source_fact'],callee=call['callee'],positional=call['positional'],
                keywords=call['keywords'],
                namespace_candidates=[row['path'] for row in candidates],namespace_routes=candidates,
                declaration_candidate=(imports[0] if len(imports)==1 else
                    candidates[0]['path'] if len(candidates)==1 else None),
                observations=[dict(attempt=row['attempt'],target=row['target'],target_anchor=row['target_anchor'])
                              for row,_,_ in self.observed.at(site,caller=context.invocation)],
                candidate_identity_required=True, original_argument_semantics_required=True,
                current_provider_state_and_effect_contract_required=True,
                reevaluation_qualified=False)
        return self.providers[key]['id']

    def slice(self, row):
        data = set()
        defaults = set()
        providers = set()
        guards = set()
        primitives = {}
        literals = {}
        constructions = {}
        open_sources = set()
        frontiers = set()
        state = set()
        pending = deque()
        demands = (*row['field_writers'],*row['content_writers'])
        origins = []

        def control(context, operation):
            for condition,outcome in sorted(context.guards.get(operation,()),key=repr):
                guards.add((context.invocation,condition,outcome))
                pending.append((context,condition))
            if context is not self.primary:
                edge = self.children[context]
                for condition,outcome in sorted(self.primary.guards.get(_entry(edge.port),()),key=repr):
                    guards.add((self.primary.invocation,condition,outcome))
                    pending.append((self.primary,condition))

        for writer in demands:
            source = self.writer(writer)
            if source is None:
                continue
            context,operation,payload,access = source
            origins.append(dict(invocation=context.invocation,source_fact=access['source_fact'],operation=operation))
            state.add((context.invocation,decode(T.GlobalPort,access['world_before']),decode(T.GlobalPort,access['normal_world_after'])))
            pending.append((context,payload))
            if access['kind']=='WRITE':
                pending.append((context,decode(T.GlobalPort,access['receiver']).port))
                selector = access['selector']
                if 'port' in selector:
                    pending.append((context,decode(T.GlobalPort,selector).port))
            control(context,operation)
        visited = set()
        while pending:
            context,port = pending.popleft()
            if (context,port) in visited:
                continue
            visited.add((context,port))
            if len(visited)>30_000:
                raise ValueError('startup transitive value slice exceeds original body cap')
            argument = context.arguments.get(port)
            if argument is not None:
                if context is self.primary:
                    role = argument['role']
                    if role in ('POSITIONAL','KEYWORD'):
                        data.add(argument['name'])
                    elif role=='DECLARED_DEFAULT':
                        defaults.add(argument['name'])
                    continue
                if argument['actual'] is not None:
                    pending.append((self.primary,decode(T.GlobalPort,argument['actual']).port))
                else:
                    frontiers.add('NESTED_DECLARATION_DEFAULT_REQUIRES_CURRENT_VALUE_GUARD')
                continue
            if port in context.calls:
                call = context.calls[port]
                providers.add(self.provider(context,call))
                control(context,_entry(port))
                if call['positional_pack'] is not None or call['keyword_pack'] is not None:
                    frontiers.add('EXPANDED_CALL_OPERAND_PACK_OPEN')
                pending.extend((context,decode(T.GlobalPort,actual).port) for actual in
                    (*call['positional'],*(arg['value'] for arg in call['keywords'])))
                continue
            if port in context.fields:
                field = context.fields[port]
                paths = self.address(context,field['receiver_paths'])
                for path in sorted(paths):
                    candidates = [self.writer(writer) for writer in self.fields.get((*path,field['selector']),())]
                    resolved = False
                    for candidate in candidates:
                        if candidate is None:
                            continue
                        owner,write,payload,access = candidate
                        if self.position(owner,write) >= self.position(context,port,latest=True):
                            continue
                        resolved = True
                        state.add((owner.invocation,decode(T.GlobalPort,access['world_before']),
                                   decode(T.GlobalPort,access['normal_world_after'])))
                        control(owner,write)
                        pending.append((owner,payload))
                    if not resolved:
                        frontiers.add('ORIGINAL_FIELD_READ_HAS_NO_QUALIFIED_PRECEDING_WRITE')
                if not paths:
                    frontiers.add('ATTRIBUTE_READ_OUTSIDE_CONSTRUCTOR_RECEIVER_FIELDS')
                control(context,_entry(port))
                pending.extend((context,other) for other in sorted(context.reverse[port],key=repr))
                continue
            operation = context.operators.get(port)
            if operation is not None:
                control(context,operation.entry)
                key = context.invocation,operation.entry
                primitive = dict(invocation=context.invocation,operation=operation.entry,
                                 instruction=type(operation.instruction).__name__)
                if port in context.operator_facts:
                    ordinal,fact = context.operator_facts[port]
                    primitive.update(selector=fact.selector,
                        source_fact=self.slicer.fact(context.unit.facts,'operations',ordinal))
                primitives[key] = primitive
                if type(operation.instruction) is P.DefineLiteral:
                    literals[context.invocation,port] = dict(invocation=context.invocation,result=port,
                                                            literal=operation.instruction.value)
                if (type(operation.instruction) is P.ConstructValue
                        and operation.instruction.construction is P.ConstructionKind.LITERAL
                        and port not in context.constants and operation.instruction.payload):
                    # F4 LITERAL's first payload is its original value slot;
                    # other slots carry source representation metadata.
                    slot = operation.instruction.payload[0]
                    literal = self.source_literal(context,slot)
                    if literal is not None:
                        literals[context.invocation,port] = dict(invocation=context.invocation,result=port,
                            literal=literal,original_literal_slot=slot)
                if (type(operation.instruction) is P.ConstructValue
                        and operation.instruction.construction is P.ConstructionKind.SLICE
                        and not any(not _is_state(item.value_type) for item in operation.inputs)
                        and len(operation.instruction.payload)==3):
                    slots = operation.instruction.payload
                    bounds = tuple(self.source_literal(context,slot) for slot in slots)
                    if all(bound is not None and bound.to_python() is None for bound in bounds):
                        constructions[context.invocation,port] = dict(
                            invocation=context.invocation,result=port,kind='SLICE',
                            original_bound_slots=slots,bounds=bounds)
            if port in context.constants:
                ordinal,fact = context.constants[port]
                literals[context.invocation,port] = dict(invocation=context.invocation,result=port,
                    literal=fact.literal,source_fact=self.slicer.fact(context.unit.facts,'constants',ordinal))
            key = context.invocation,port
            if not context.reverse[port] and key not in literals and key not in constructions:
                open_sources.add((context.invocation,port))
            pending.extend((context,other) for other in sorted(context.reverse[port],key=repr))
        if not origins:
            frontiers.add('NO_SELECTED_ORIGINAL_STARTUP_WRITER')
        if providers:
            frontiers.add('CURRENT_PROVIDER_VALUE_AND_EFFECT_CONTRACT_OPEN')
        if defaults:
            frontiers.add('CURRENT_DEFAULT_VALUES_REQUIRE_DECLARATION_GUARDS')
        if open_sources:
            frontiers.add('ORIGINAL_VALUE_SOURCES_NOT_FULLY_CLOSED')
        frontiers.add('ORDINARY_RECEIVER_ALIAS_AND_INTERVENING_WORLD_EFFECTS_REQUIRE_GUARDS')
        return dict(id=row['id'],controller_path=row['controller_path'],kind=row['kind'],origins=origins,
            supplied_inputs=sorted(data),declared_defaults=sorted(defaults),provider_calls=sorted(providers),
            control_guards=[dict(invocation=inv,condition=p,when=flag) for inv,p,flag in sorted(guards,key=repr)],
            immediate_world_boundaries=[dict(invocation=inv,before=a,after=b) for inv,a,b in sorted(state,key=repr)],
            primitive_producers=[primitives[key] for key in sorted(primitives,key=repr)],
            declaration_literals=[literals[key] for key in sorted(literals,key=repr)],
            declaration_constructions=[constructions[key] for key in sorted(constructions,key=repr)],
            open_value_sources=[dict(invocation=inv,port=port) for inv,port in sorted(open_sources,key=repr)],
            state_frontiers=sorted(frontiers),startup_value_or_replacement_qualified=False)


@replay_transaction()
def project(initialization):
    reference = artifact_reference(Path(initialization).resolve(strict=True),INITIALIZATION)
    graph = check_initialization(reference['uri'])
    if read_artifact(reference) != graph:
        raise ValueError('startup handoff artifact changed during dependency replay')
    entry = read_artifact(graph['entry_calls'])
    controller = read_artifact(graph['controller'])
    receiver = _only((r['name'] for r in controller['parameters']
                      if r['formal']==graph['controller_call']['receiver_formal']),
                     'one original controller startup receiver required')
    scope = graph['trace_input']
    demands = list(graph['ports'])
    requested = {tuple(row['controller_path']) for row in demands}
    for row in graph['initial_receiver_fields']:
        path = (receiver,*row['path'])
        if path not in requested:
            demands.append(dict(**row,id='field:'+'.'.join(path),controller_path=path,kind='field'))
    with BuildSession(Path(entry['program']['uri']).parent) as session:
        with _admit_entry_calls(session,entry):
            view = session.view
            observed = _Observed(view,scope['directory'],scope['project_root'])
            analysis = _StartupSlice(view,graph,observed)
            ports = [analysis.slice(row) for row in demands]
            observed.index.check()
            ingress = []
            for row in graph['constructor']['initializer']['arguments']:
                if row['role'] not in ('POSITIONAL','KEYWORD','DECLARED_DEFAULT'):
                    continue
                value = (analysis.source_literal(analysis.primary,decode(P.SlotRef,row['declared_default']))
                         if row['role']=='DECLARED_DEFAULT' else None)
                ingress.append(dict(**row,original_declaration_literal=value))
    return encode(dict(schema=SCHEMA,schema_version=1,initialization=reference,controller=graph['controller'],
        ingress=ingress,ports=ports,providers=list(analysis.providers.values()),
        selected_control=dict(policy='ORIGINAL_BALANCED_NORMAL_PATHS_CONTAINING_OBSERVED_CALL_ORDER',
            original_calls=[r['original_call']['source_result'] for r in graph['initializer_calls']],
            globally_entailed_guards=[dict(condition=p,when=flag)
                                      for p,flag in sorted(analysis.primary.profile_guards,key=repr)],
            every_future_execution_selects_this_profile=False),
        world_edges_are_order_and_effect_interfaces=True,
        current_default_values_verified=False,provider_effects_closed=False,
        startup_materialization_qualified=False,transformation_authorized=False))


def check(path):
    saved = read_artifact(artifact_reference(Path(path).resolve(strict=True),SCHEMA))
    result = project(saved['initialization']['uri'])
    if result != saved:
        raise ValueError('startup slice differs from original F4 ports and matched initialization profile')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initialization',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--check',type=Path)
    args = parser.parse_args(argv)
    if args.check and not (args.initialization or args.out):
        result = check(args.check)
    elif args.initialization and args.out and not args.check:
        result = project(args.initialization)
        with args.out.open('x',encoding='utf-8') as stream:
            json.dump(result,stream,sort_keys=True,separators=(',',':'),allow_nan=False)
            stream.write('\n')
    else:
        parser.error('provide --check or --initialization with --out')
    print(json.dumps(dict(schema=SCHEMA,ports=len(result['ports']),providers=len(result['providers']))))


if __name__ == '__main__':
    main()
