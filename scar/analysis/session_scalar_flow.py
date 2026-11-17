"""Finite conditional scalar types over original shared F4 ports and loops.

Reference paths and scalar types are separate domains. Lexical joins combine
types, never content or object identities. Guards and declared CALL returns
are obligations, not measurements or already satisfied assumptions.
"""
from collections import Counter, defaultdict, deque

from scar.analysis.session_caller_effects import _source, _fact_indexes
from scar.analysis.source_body_interfaces import BodyInterfaceResolver
from scar.analysis.source_call_view import source_call_site
from scar.analysis.source_reference_slice import ReferenceSlicer, source_port
from scar.analysis.source_value_flow import project_value_flow
from scar.ir.literals import LiteralKind
from scar.ir.program import normal_graph as G, normal_links as F, source_productions as P
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.model import ValueKind, ValueType
from scar.ir.program.normal_targets import Body
from scar.ir.program.source_value_flow import SourceBodyValueFlow
from scar.ir.program.structured import PythonProtocolKind as K
from scar.ir.record_codec import encode
from scar.optimization.resource_solver import semantic_sha256
from types import SimpleNamespace

SCHEMA = 'scar.session-scalar-flow'
KINDS = frozenset(('NUMBER', 'BOOL', 'STRING', 'NONE', 'RANGE', 'RANGE_ITERATOR', 'UNKNOWN'))
_SCALARS = frozenset(('NUMBER', 'BOOL', 'STRING', 'NONE'))
_NUMBER_OPS = frozenset(('Add', 'Sub', 'Mult', 'Div', 'FloorDiv', 'Mod', 'UAdd', 'USub'))
_COMPARE_OPS = frozenset(('Eq', 'NotEq', 'Lt', 'LtE', 'Gt', 'GtE'))


def _state(kind):
    return type(kind) is ValueType and kind.kind is ValueKind.STATE


def _literal_kind(literal):
    return {LiteralKind.NONE: 'NONE', LiteralKind.BOOL: 'BOOL', LiteralKind.INT: 'NUMBER',
        LiteralKind.FLOAT64: 'NUMBER', LiteralKind.STR: 'STRING',
        LiteralKind.STR_CODEPOINTS: 'STRING'}.get(literal.kind, 'UNKNOWN')


def derive_scalar_flow(view, *, unit, function_ordinal, known_call_returns=None, resolver=None, flow=None):
    """Derive conditional kinds, reference paths and loop type fixed points.

    ``known_call_returns`` maps original CALL ordinals to one kind string.
    Each supplied kind remains an explicit runtime return-check obligation.
    """
    original = view.unit(unit); facts = original.facts
    if type(function_ordinal) is not int or not 0 <= function_ordinal < len(facts.functions):
        raise ValueError('original source function ordinal required')
    fn = facts.functions[function_ordinal]
    if fn.body_scope is None:
        raise ValueError('source function has no original body scope')
    source, _ = _source(view, unit, fn)
    body = Body(unit, fn.template, fn.body); formal = source_port(facts, fn.value)
    if flow is None:
        flow = project_value_flow(original.material, formal, body)
    elif (type(flow) is not SourceBodyValueFlow or flow.source_function != formal or flow.body != body
            or flow.source_templates != original.material.result_artifact):
        raise ValueError('shared scalar flow differs from original source function/body/templates')
    returns = {} if known_call_returns is None else dict(known_call_returns)
    if any(type(q) is not int or type(kind) is not str or kind not in KINDS-{'UNKNOWN'}
           for q,kind in returns.items()):
        raise ValueError('conditional return contracts need original CALL ordinals and finite kinds')
    resolver = resolver or BodyInterfaceResolver(view)
    slicer = ReferenceSlicer(SimpleNamespace(fact_store=view.catalog.fact_store))
    spans, by_world, name_facts = _fact_indexes(original, fn, slicer)

    # Ports are interned once. Original SSA/control IDs remain their identity;
    # abstract lexical join cells are analysis state, never new source nodes.
    ports, port_ids = [], {}
    def pid(port):
        if port not in port_ids:
            port_ids[port] = len(ports); ports.append(port)
        return port_ids[port]
    for edge in flow.edges:
        pid(edge.source); pid(edge.target)
        if edge.condition is not None: pid(edge.condition)
    for operation in flow.operations:
        pid(operation.entry)
        for value in (*operation.inputs, *operation.outputs): pid(value.port)
    formal_ports = {source_port(facts,index).port:name for name,index in zip(fn.parameter_names,fn.formals[1:])}
    for port in formal_ports: pid(port)

    type_edges = defaultdict(list); ref_edges = defaultdict(list)
    reverse = defaultdict(list)
    for edge in flow.edges:
        if edge.kind is not G.EdgeKind.DATA: continue
        a,b = pid(edge.source),pid(edge.target)
        type_edges[a].append(b); ref_edges[a].append(b); reverse[b].append(a)
    lexical = []
    for index,location in enumerate(flow.locations):
        join = ('lexical', index)
        writes = tuple(pid(a.value) for a in location.writes)
        reads = tuple(pid(a.value) for a in location.reads)
        for write in writes: type_edges[write].append(join); reverse[join].append(write)
        for read in reads: type_edges[join].append(read); reverse[read].append(join)
        stable = len(location.writes)==1 and location.writes[0].value in formal_ports
        if stable:
            for read in reads: ref_edges[writes[0]].append(read)
        lexical.append(dict(namespace=encode(location.namespace), writes=list(writes), reads=list(reads),
            write_facts=[encode(a.fact) for a in location.writes],
            read_facts=[encode(a.fact) for a in location.reads],
            stable_formal_reference=stable, join_kind='TYPE_UNION_NOT_VALUE_EQUALITY'))

    kinds, paths = defaultdict(set), defaultdict(set)
    queue, pending = deque(), set(); propagations = 0
    def enqueue(node):
        if node not in pending: pending.add(node); queue.append(node)
    def put(node, *, types=(), refs=()):
        before = len(kinds[node]), len(paths[node])
        kinds[node].update(types); paths[node].update(refs)
        if before != (len(kinds[node]),len(paths[node])): enqueue(node)
    for port,name in formal_ports.items(): put(pid(port),refs=((name,),))

    records, watchers = [], defaultdict(set)
    original_calls = {reference:ordinal for rows in by_world.values()
        for table,ordinal,_,reference in rows if table=='calls'}
    call_records = {}
    op_by_key = {(row.entry.template,row.entry.owner):i for i,row in enumerate(flow.operations)}
    for call in flow.calls:
        raw=call.call
        link=SimpleNamespace(result=source_port(facts,raw.result),callee=source_port(facts,raw.callee))
        operation,_,_=source_call_site(view,link); step=view.original(operation)
        index=op_by_key[link.result.port.template,step.id]
        q=original_calls[call.fact]
        call_records[index]=dict(query_ordinal=q,fact=encode(call.fact),call=encode(raw))
    if not set(returns)<=set(row['query_ordinal'] for row in call_records.values()):
        raise ValueError('return contract is not a CALL owned by this original function')
    constants={source_port(facts,row.value).port:(i,row.literal) for i,row in enumerate(facts.constants)}
    demands=[]; conditional_guards=[]; terminal=set(); attributes=[]
    for index,operation in enumerate(flow.operations):
        context=resolver.plan(unit,operation.entry.template); step=context.steps[operation.entry.owner]
        if step.instruction!=operation.instruction:
            raise ValueError('scalar operation differs from original sealed F4')
        inputs=[pid(v.port) for v in operation.inputs if not _state(v.value_type)]
        outputs=[pid(v.port) for v in operation.outputs
            if v.port.completion is C.FALLTHROUGH and not _state(v.value_type)]
        before=tuple(context.value(v) for v in step.arguments if _state(context.emitter.value(v)[1]))
        after=tuple(context.gp(v.port) for v in operation.outputs
            if v.port.completion is C.FALLTHROUGH and _state(v.value_type))
        matches=by_world.get((before[0],after[0]),()) if len(before)==len(after)==1 else ()
        instruction=step.instruction; role='UNSUPPORTED'; fixed=(); op_frontiers=[]
        attr=[(n,row,reference) for table,n,row,reference in matches if table=='attributes']
        arithmetic=[(n,row,reference) for table,n,row,reference in matches if table=='operations']
        if type(instruction) is P.EnterDeferredBody:
            role='ACTIVATION_ENTRY'
        elif type(instruction) is P.DefineLiteral:
            role='SOURCE_LITERAL'; fixed=(_literal_kind(instruction.value),)
        elif type(instruction) is P.ConstructValue and instruction.construction is P.ConstructionKind.LITERAL:
            literal=constants.get(context.result(step).port)
            if literal is None:op_frontiers.append('ORIGINAL_LITERAL_FACT_MISSING')
            else:role='SOURCE_LITERAL';fixed=(_literal_kind(literal[1]),)
        elif type(instruction) is P.AccessBinding:
            selected = (context.result(step) if instruction.action is P.BindingActionKind.READ
                else context.value(step.arguments[0]) if step.arguments else None)
            names = name_facts.get(selected, ())
            role='LEXICAL_BINDING' if names and all(row.namespace.kind is F.NamespaceKind.LEXICAL
                for _,_,row,_ in names) else 'NONLEXICAL_LOOKUP'
            matches=(*matches,*names)
            if role=='NONLEXICAL_LOOKUP': op_frontiers.append('CURRENT_NAMESPACE_LOOKUP_PROTOCOL_REQUIRED')
        elif type(instruction) is P.ApplyLanguageAction:
            if instruction.action in (P.LanguageActionKind.ATTRIBUTE_GET,P.LanguageActionKind.ATTRIBUTE_SET,
                                       P.LanguageActionKind.ATTRIBUTE_DELETE) and len(attr)==1:
                role='ATTRIBUTE_'+attr[0][1].action.value
                attributes.append((index,attr[0],inputs,outputs))
            elif instruction.action is P.LanguageActionKind.CALL:
                role='CALL'; q=call_records[index]['query_ordinal']
                fixed=(returns.get(q,'UNKNOWN'),)
                if q in returns:
                    conditional_guards.append(dict(opcode='CHECK_ORIGINAL_CALL_NORMAL_RETURN_KIND',
                        operation=index,query_ordinal=q,kind=returns[q],fact=call_records[index]['fact'],
                        checked=False,observed_return=False))
                else:op_frontiers.append('CALL_RETURN_KIND_NOT_DECLARED')
            elif arithmetic:
                selector=arithmetic[0][1].selector.value
                if selector in _NUMBER_OPS or selector in _COMPARE_OPS:
                    role='SCALAR_COMPARE' if selector in _COMPARE_OPS else 'SCALAR_ARITHMETIC'
                    fixed=('BOOL',) if role=='SCALAR_COMPARE' else ('NUMBER',)
                    demands.extend((p,frozenset(('NUMBER',)),index) for p in inputs)
                else:op_frontiers.append('SCALAR_OPERATOR_OUTSIDE_CLOSED_PROFILE')
            elif instruction.action is P.LanguageActionKind.STRING_JOIN:
                role='STRING_JOIN';fixed=('STRING',)
                demands.extend((p,frozenset(('STRING',)),index) for p in inputs)
            elif instruction.action is P.LanguageActionKind.TRUTH:
                role='TRUTH';fixed=('BOOL',)
                demands.extend((p,_SCALARS,index) for p in inputs)
            elif instruction.action in (P.LanguageActionKind.ITERATOR,P.LanguageActionKind.NEXT):
                advancing=instruction.action is P.LanguageActionKind.NEXT
                role='RANGE_ITERATOR_NEXT' if advancing else 'RANGE_ITERATOR_CREATE'
                fixed=('NUMBER',) if advancing else ('RANGE_ITERATOR',)
                demands.extend((p,frozenset(('RANGE_ITERATOR' if advancing else 'RANGE',)),index)for p in inputs)
        elif type(instruction) is P.ApplyInplaceOperator:
            if len(arithmetic)==1 and arithmetic[0][1].selector.value in _NUMBER_OPS:
                role='SCALAR_INPLACE';fixed=('NUMBER',)
                demands.extend((p,frozenset(('NUMBER',)),index) for p in inputs)
            else:op_frontiers.append('INPLACE_OPERATOR_OUTSIDE_CLOSED_SCALAR_PROFILE')
        elif type(instruction) is P.ApplyPythonProtocol:
            protocol=instruction.protocol
            if protocol is K.TRUTH:
                role='TRUTH';fixed=('BOOL',)
                demands.extend((p,_SCALARS,index) for p in inputs)
            elif protocol is K.FSTRING_FORMAT:
                role='STRING_FORMAT';fixed=('STRING',)
                if len(inputs)!=2:op_frontiers.append('FORMAT_ABI_OUTSIDE_PROFILE')
                else:demands.extend(((inputs[0],_SCALARS,index),(inputs[1],frozenset(('STRING',)),index)))
            elif protocol in (K.BUILD_STRING,K.FSTRING_CONVERT):
                role='STRING_BUILD_OR_CONVERT';fixed=('STRING',)
                demands.extend((p,_SCALARS if protocol is K.FSTRING_CONVERT else frozenset(('STRING',)),index) for p in inputs)
            elif protocol is K.FRAME_RELEASE_TEMPORARY:
                role='RELEASE_TEMPORARY'
                op_frontiers.append('RELEASE_REQUIRES_SCALAR_KIND_OR_RETAINED_REFERENCE_OR_TAINT')
            elif protocol in (K.GET_ITERATOR,K.ADVANCE_ITERATOR):
                advancing=protocol is K.ADVANCE_ITERATOR
                role='RANGE_ITERATOR_NEXT' if advancing else 'RANGE_ITERATOR_CREATE'
                fixed=('NUMBER',) if advancing else ('RANGE_ITERATOR',)
                demands.extend((p,frozenset(('RANGE_ITERATOR' if advancing else 'RANGE',)),index)for p in inputs)
            else:op_frontiers.append('SOURCE_PROTOCOL_OUTSIDE_SCALAR_PROFILE')
        if role=='UNSUPPORTED' and not op_frontiers:op_frontiers.append('SOURCE_INSTRUCTION_OUTSIDE_SCALAR_PROFILE')
        if fixed:
            for output in outputs:put(output,types=fixed)
        if role not in ('LEXICAL_BINDING','ACTIVATION_ENTRY'):
            terminal.update(outputs)
        for p in inputs:watchers[p].add(index)
        records.append(dict(index=index,entry=pid(operation.entry),instruction=encode(instruction),
            inputs=inputs,outputs=outputs,input_types=encode(tuple(v.value_type for v in operation.inputs if not _state(v.value_type))),
            output_types=encode(tuple(v.value_type for v in operation.outputs
                if v.port.completion is C.FALLTHROUGH and not _state(v.value_type))),
            world_before=[pid(p.port) for p in before],normal_world_after=[pid(p.port) for p in after],
            role=role,source_facts=[dict(table=t,ordinal=n,reference=encode(ref)) for t,n,_,ref in matches],
            call=call_records.get(index),frontiers=op_frontiers))

    # The reference domain deliberately does not flow through multi-writer
    # lexical cells. A lookup path denotes a request, not a stable field value.
    attr_by_op={index:(fact,ins,outs) for index,fact,ins,outs in attributes}
    def drain():
        nonlocal propagations
        while queue:
            node=queue.popleft();pending.remove(node);propagations+=1
            for target in type_edges[node]:put(target,types=kinds[node])
            for target in ref_edges[node]:put(target,refs=paths[node])
            for i in watchers.get(node,()):
                if i not in attr_by_op:continue
                (_,fact,_),inputs,outputs=attr_by_op[i]
                if fact.action is F.AttributeAction.GET and inputs:
                    for output in outputs:put(output,refs=(path+(fact.name,) for path in paths[inputs[0]]))
    drain()

    # Backward requirements run only over value transport/type joins. They
    # stop at producers: neither CALL worlds nor an operator's inputs acquire
    # a new equality edge. Finite intersections become explicit type guards.
    required={}; demand_sites=defaultdict(set); todo=deque(demands); seen=set()
    formal_ids={pid(p) for p in formal_ports}
    while todo:
        node,allowed,site=todo.popleft(); token=(node,allowed,site)
        if token in seen:continue
        seen.add(token)
        if type(node) is tuple and not reverse[node]:
            records[site]['frontiers'].append('LEXICAL_VALUE_HAS_NO_ORIGINAL_REACHING_WRITER')
        elif node in terminal or node in formal_ids or not reverse[node]:
            required[node]=allowed if node not in required else required[node]&allowed
            demand_sites[node].add(site)
        else:
            for source_node in reverse[node]:todo.append((source_node,allowed,site))
    producer={output:row for row in records for output in row['outputs']}
    for node,allowed in required.items():
        row=producer.get(node); current=kinds[node]
        if not allowed:
            for i in demand_sites[node]:records[i]['frontiers'].append('INCOMPATIBLE_SCALAR_KIND_REQUIREMENTS')
            continue
        if row is not None and row['role']=='SOURCE_LITERAL':
            if not current<=allowed:
                for i in demand_sites[node]:records[i]['frontiers'].append('SOURCE_LITERAL_KIND_INCOMPATIBLE_WITH_PROFILE')
            continue
        inferred=current and 'UNKNOWN' not in current and current<=allowed
        if inferred:continue
        conditional_guards.append(dict(opcode='CHECK_EXACT_BUILTIN_VALUE_KINDS',port=node,
            kinds=sorted(allowed),sites=sorted(demand_sites[node]),
            paths=[list(p)for p in sorted(paths[node])],checked=False,
            placement='ENTRY_FORMAL' if node in formal_ids else 'ORIGINAL_VALUE_AFTER_PRODUCER_BEFORE_CONSUMER'))
        if not current and (node in formal_ids or row is not None and row['role']=='ATTRIBUTE_GET'):
            put(node,types=allowed)
    drain()

    field_writes=[]
    for i,(ordinal,fact,reference),inputs,outputs in attributes:
        receiver_paths=paths[inputs[0]] if inputs else set()
        field_paths={p+(fact.name,) for p in receiver_paths}
        if not field_paths:records[i]['frontiers'].append('ATTRIBUTE_RECEIVER_REFERENCE_PATH_UNRESOLVED')
        conditional_guards.append(dict(opcode='CHECK_ORDINARY_ATTRIBUTE_PROTOCOL',operation=i,
            action=fact.action.value,receiver=inputs[0] if inputs else None,attribute=fact.name,
            source_fact=encode(reference),checked=False,
            reject_custom_getattribute_getattr_setattr_and_data_descriptors=True,
            allowed_read_routes=['INSTANCE_DICTIONARY_VALUE','STANDARD_PYTHON_FUNCTION_DESCRIPTOR',
                                 'STANDARD_STATICMETHOD_PYTHON_FUNCTION']))
        if fact.action is F.AttributeAction.SET:
            rhs=inputs[1] if len(inputs)>1 else None
            field_writes.append(dict(operation=i,attribute_fact=encode(reference),
                paths=[list(p)for p in sorted(field_paths)],payload=rhs,
                payload_kinds=sorted(kinds[rhs]) if rhs is not None else ['UNKNOWN'],
                old_slot_release_requires_guard=True,content_identity_or_version_equality=False))
    for row in records:
        if (row['role']=='RELEASE_TEMPORARY' and row['inputs']
                and all(kinds[p] and kinds[p]<=_SCALARS for p in row['inputs'])):
            row['frontiers'].remove('RELEASE_REQUIRES_SCALAR_KIND_OR_RETAINED_REFERENCE_OR_TAINT')
            row['release_contract']='CONDITIONAL_EXACT_BUILTIN_SCALAR_WITHOUT_FINALIZER_OR_WEAKREF_CALLBACK'
        row['output_kinds']=[sorted(kinds[p]) or ['UNKNOWN'] for p in row['outputs']]
        row['input_kinds']=[sorted(kinds[p]) or ['UNKNOWN'] for p in row['inputs']]
        row['frontiers']=sorted(set(row['frontiers']))
    for i,row in enumerate(lexical):row['kinds']=sorted(kinds['lexical',i]) or ['UNKNOWN']
    result=dict(schema=SCHEMA,schema_version=1,source=encode(source),unit=unit,function_ordinal=function_ordinal,
        source_function=encode(formal),body=encode(body),original_f4=encode(original.material.result_artifact),
        scope_fact_spans=spans,parameters=list(fn.parameter_names),
        ports=[dict(port=encode(port),kinds=sorted(kinds[i]) or ['UNKNOWN'],
            reference_paths=[list(p)for p in sorted(paths[i])],
            path_is_lookup_route_not_object_or_content_identity=True) for i,port in enumerate(ports)],
        operations=records,lexical_joins=lexical,field_writes=field_writes,
        edges=[dict(source=pid(e.source),target=pid(e.target),kind=e.kind.value,value_type=encode(e.value_type),
            site=encode(e.site),transition=e.transition.value,
            condition=None if e.condition is None else pid(e.condition),when=e.when)for e in flow.edges],
        guards=conditional_guards,known_call_returns=[dict(query_ordinal=q,kind=k,
            authority='CONDITIONAL_RUNTIME_RETURN_CHECK_NOT_OBSERVATION')for q,k in sorted(returns.items())],
        name_boundaries=encode(flow.name_boundaries),
        analysis=dict(kind_domain=sorted(KINDS),kind_interpretation=dict(
                NUMBER='EXACT_BUILTIN_INT_OR_FLOAT_NOT_SUBCLASSES_OR_NUMPY_SCALARS',
                BOOL='EXACT_BUILTIN_BOOL',STRING='EXACT_BUILTIN_STR',NONE='BUILTIN_NONE',
                RANGE='EXACT_BUILTIN_RANGE',RANGE_ITERATOR='EXACT_BUILTIN_RANGE_ITERATOR',
                UNKNOWN='NO_COMPLETE_TYPE_FACT'),
            join='FINITE_KIND_UNION',fixed_point_reached=not queue,
            propagation_events=propagations,loop_unrolled=False,
            worlds_used_as_content_versions=False,reference_identity_from_type_join=False,
            guards_apply_each_original_symbolic_occurrence=True,constant_values_not_inferred=True,
            runtime_guards_checked=False,exceptional_and_resumed_paths_closed=False,
            effect_coverage_proven=False,caller_replacement_authorized=False),
        summary=dict(operations=len(records),ports=len(ports),edges=len(flow.edges),calls=len(call_records),
            lexical_joins=len(lexical),multiwriter_type_joins=sum(len(r['writes'])>1 for r in lexical),
            field_writes=len(field_writes),guard_specifications=len(conditional_guards),
            operations_with_frontiers=sum(bool(r['frontiers'])for r in records),
            roles=dict(sorted(Counter(r['role']for r in records).items())),
            control_backedges=sum(e.kind is G.EdgeKind.CONTROL and e.transition is G.Transition.BACKEDGE for e in flow.edges),
            type_fixed_point_reached=not queue,optimization_authorized=False))
    original.check();result['id']='session-scalar-flow:'+semantic_sha256(result)
    return result


__all__=['SCHEMA','KINDS','derive_scalar_flow']
