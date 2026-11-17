"""Discharge a bounded caller's effect ledger using original data transport.

No target execution or rewritten caller. Formal reference paths propagate on
original DATA edges and single formal lexical bindings. World edges never
prove value identity. Every branch remains; an unknown source method taints
the activation before its original execution, including finalizer effects.
"""
from collections import defaultdict, deque
from dataclasses import dataclass

from scar.analysis.source_body_interfaces import BodyInterfaceResolver
from scar.analysis.source_reference_slice import source_port
from scar.analysis.source_value_flow import project_value_flow
from scar.analysis.session_caller_effects import derive_caller_effects, _source
from scar.ir.program import normal_graph as G, source_productions as P
from scar.ir.program.control import LanguageCompletionKind as C
from scar.ir.program.normal_targets import Body
from scar.ir.program.model import ValueKind, ValueType
from scar.ir.record_codec import encode
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.guarded-session-caller'


@dataclass(frozen=True)
class _Term:
    kind: str
    path: tuple = ()


def _identity(source):
    return tuple(source[key] for key in ('path', 'sha256', 'code_key'))


def _reference(value):
    return type(value) is ValueType and value.kind is ValueKind.REFERENCE


def derive_guarded_caller(view, *, unit, function_ordinal, step_contract, copy_effects, resolver=None):
    """Conditional entry guards + source-method barriers, not outer coverage."""
    original = view.unit(unit); fn = original.facts.functions[function_ordinal]
    resolver = resolver or BodyInterfaceResolver(view)
    body = Body(unit, fn.template, fn.body)
    flow = project_value_flow(original.material, source_port(original.facts, fn.value), body)
    ledger = derive_caller_effects(view, unit=unit, function_ordinal=function_ordinal, resolver=resolver, flow=flow)
    conditions, frontiers, obligations, calls = {}, [], {}, []
    def reject(code, **details):
        frontiers.append(dict(code=code, **details))
    def guard(path, kind, **details):
        key = tuple(path)
        row = dict(path=list(path), kind=kind, **details)
        if key in conditions and conditions[key] != row:
            reject('INCOMPATIBLE_REFERENCE_TYPE_GUARDS', path=list(path))
        conditions[key] = row
    if (not fn.parameter_names or fn.environment.captures or fn.environment.owned_cells
            or any(p.kind.value in ('VAR_POSITIONAL', 'VAR_KEYWORD') for p in fn.environment.parameters)):
        reject('FIXED_ORDINARY_CALLER_INTERFACE_REQUIRED')
    receiver = fn.parameter_names[0] if fn.parameter_names else None
    if receiver: guard((receiver,), 'ORDINARY_RECEIVER')
    owner_path = ledger['source']['qualname'].split('.')[:-1]
    if not owner_path or '<locals>' in owner_path:
        reject('ORDINARY_CLASS_METHOD_OWNER_REQUIRED')
    # Exact source class binding is checked live, not inferred from the name.
    candidates = {}
    for ordinal, candidate in enumerate(original.facts.functions):
        source, _ = _source(view, unit, candidate)
        source = encode(source)
        if source['qualname'].split('.')[:-1] == owner_path:
            candidates[source['qualname'].split('.')[-1]] = (ordinal, candidate, source)
    kernel_id = _identity(step_contract['source'])
    copy_by_id = {_identity(row['source']): row for row in copy_effects}
    roots = {}
    for row in step_contract['inputs']:
        if row['kind'] == 'array':
            path = (receiver, *row['path'][1:])
            roots[path] = row
    call_info = {row['operation']: row for row in ledger['call_sites']}
    # Locations are reference transport only if their sole writer is an
    # original formal. General reaching definitions require a separate proof.
    adjacency = defaultdict(list); watchers = defaultdict(list); known = {}; queue = deque()
    formal_ports = {source_port(original.facts, index).port: _Term('PATH', (name,))
        for name,index in zip(fn.parameter_names, fn.formals[1:])}
    for edge in flow.edges:
        if edge.kind is G.EdgeKind.DATA: adjacency[edge.source].append(edge.target)
        if edge.kind is G.EdgeKind.CONTROL and edge.transition in (G.Transition.BACKEDGE, G.Transition.RESUME):
            reject('CALLER_LOOP_OR_RESUMPTION_REQUIRES_OUTER_PROFILE')
    for location in flow.locations:
        if len(location.writes) != 1 or location.writes[0].value not in formal_ports:
            reject('NONFORMAL_OR_MULTIPLE_LEXICAL_WRITERS', namespace=encode(location.namespace)); continue
        for read in location.reads: adjacency[location.writes[0].value].append(read.value)
    if flow.name_boundaries: reject('NONLEXICAL_REFERENCE_BOUNDARY')
    # Exceptional cleanup/catch cannot silently escape the normal flow ledger.
    for key, plan in original.material.closure(fn.template).items():
        context = resolver.plan(unit, key)
        for step in plan.steps:
            if context.owner(step.region).owner != fn.body_scope.owner: continue
            if type(step.instruction) in (P.HandleCompletions, P.SuspendExecution, P.LoopControl):
                reject('CLEANUP_LOOP_OR_SUSPENSION_OUTSIDE_CALLER_PROFILE', site=encode(context.site(step)))
    def put(port, term):
        if port in known:
            if known[port] != term: reject('DISTINCT_REFERENCE_PATHS_JOIN', port=encode(port))
        else:
            known[port] = term; queue.append(port)
    for port, term in formal_ports.items(): put(port, term)
    for index, operation in enumerate(flow.operations):
        for value in operation.inputs:
            if _reference(value.value_type): watchers[value.port].append(index)
    processed = set()
    def process(index):
        if index in processed: return
        operation = flow.operations[index]; record = ledger['operations'][index]
        refs = [v.port for v in operation.inputs if _reference(v.value_type)]
        role = record['role']
        if role != 'ACTIVATION_ENTRY' and not all(p in known for p in refs): return
        terms = [known[p] for p in refs if p in known]; result = None
        mode = 'GUARDED_REFERENCE_OR_BUILTIN_PROTOCOL'
        if role in ('ACTIVATION_ENTRY', 'LEXICAL_BINDING_READ'):
            mode = 'ORIGINAL_FORMAL_REFERENCE_TRANSPORT'
        elif role == 'ORDINARY_RECEIVER_ATTRIBUTE_READ':
            attribute = record['guard_specs'][0]['parameters']['attribute']
            if len(terms) != 1 or terms[0] != _Term('PATH', (receiver,)):
                reject('ATTRIBUTE_RECEIVER_IS_NOT_ORIGINAL_FORMAL', operation=index)
            else: result = _Term('PATH', (*terms[0].path, attribute))
        elif role == 'EXPLICIT_CALL':
            info = call_info[index]; fact = original.facts.calls[info['query_ordinal']]
            if (fact.positional_pack is not None or fact.keyword_pack is not None or not terms
                    or terms[0].kind != 'PATH' or len(terms[0].path) != 2
                    or terms[0].path[0] != receiver or terms[0].path[-1] not in candidates):
                reject('CALL_REQUIRES_A_GUARDED_ORIGINAL_SOURCE_METHOD', operation=index)
            else:
                ordinal, target, source = candidates[terms[0].path[-1]]
                if _identity(source) == kernel_id:
                    kind, descriptor = 'KERNEL', 'METHOD'
                    source = step_contract['source']
                    if len(step_contract['parameters']) != 1 or fact.positional or fact.keywords:
                        reject('KERNEL_RECEIVER_ABI_NOT_CLOSED', operation=index)
                    for path, root in roots.items(): guard(path, 'EXACT_ARRAY_ROOT', shape=root['shape'], dtype='float64')
                    result = _Term('NONE')
                elif _identity(source) in copy_by_id:
                    kind, descriptor = 'COPY', 'STATICMETHOD'
                    source = copy_by_id[_identity(source)]['source']
                    if len(terms) != 2 or fact.keywords or terms[1].kind != 'PATH' or terms[1].path not in roots:
                        reject('COPY_ACTUAL_IS_NOT_ORIGINAL_KERNEL_ROOT', operation=index)
                    else:
                        root = roots[terms[1].path]
                        guard(terms[1].path, 'EXACT_ARRAY_ROOT', shape=root['shape'], dtype='float64')
                    result = _Term('NONE')
                else:
                    kind, descriptor = 'BARRIER', 'METHOD'
                    result = _Term('TAINTED_RETURN')
                    context = resolver.plan(unit, target.template)
                    if any(type(s.instruction) is P.ApplyDecorator for s in context.plan.steps):
                        reject('UNKNOWN_DECORATED_METHOD_BINDING', operation=index)
                boundary = dict(id='session-boundary:'+semantic_sha256(dict(source=source,role=kind)),
                    role=kind, source=source, owner_path=owner_path, attribute=terms[0].path[-1], descriptor=descriptor)
                calls.append(dict(operation=index, query_ordinal=info['query_ordinal'],
                    fact=info['fact'], actual_paths=[list(t.path) if t.kind=='PATH' else None for t in terms[1:]],
                    boundary=boundary, policy='TAINT_BEFORE_CALL_UNTIL_OUTERMOST_EXIT' if kind=='BARRIER' else kind))
                mode = kind
        elif role in ('PRIMITIVE_SCALAR_OPERATION', 'PRIMITIVE_TRUTH_TEST'):
            for term in terms:
                if term.kind == 'PATH': guard(term.path, 'EXACT_BOOL' if role=='PRIMITIVE_TRUTH_TEST' else 'EXACT_NUMBER')
                elif term.kind not in ('BUILTIN_SCALAR', 'NONE'):
                    reject('SCALAR_PROTOCOL_TYPE_NOT_CLOSED', operation=index)
            result = _Term('BUILTIN_SCALAR')
        elif role == 'FRAME_RELEASE_TEMPORARY':
            # This may release a bound method or a root reference. Hold all
            # guarded roots/formals/descriptors through the owning activation.
            # Unknown returns are protected by monotonic activation taint.
            mode = 'RETAINED_REFERENCES_OR_EXACT_BUILTINS_OR_TAINTED_UNKNOWN_RETURN'
        else:
            reject('UNSUPPORTED_ORIGINAL_CALLER_OPERATION', operation=index, role=role)
        outputs = [v.port for v in operation.outputs if _reference(v.value_type) and v.port.completion is C.FALLTHROUGH]
        if result is not None:
            if not outputs and role == 'PRIMITIVE_TRUTH_TEST': pass  # Original typed condition, not a Python reference.
            elif len(outputs) != 1: reject('SOURCE_NORMAL_REFERENCE_RESULT_ARITY', operation=index)
            else: put(outputs[0], result)
        obligations[index] = dict(operation=index, original_entry=encode(operation.entry), discharge=mode)
        processed.add(index)
    # Entry/state-only operations need no reference worklist trigger.
    for index in range(len(flow.operations)): process(index)
    while queue:
        port = queue.popleft()
        for target in adjacency[port]: put(target, known[port])
        for index in watchers[port]: process(index)
    for index in range(len(flow.operations)):
        if index not in processed: reject('ORIGINAL_REFERENCE_PATH_UNRESOLVED', operation=index)
    callable_paths = {(receiver,row['boundary']['attribute']) for row in calls}
    for index, operation in enumerate(flow.operations):
        if ledger['operations'][index]['role'] != 'FRAME_RELEASE_TEMPORARY': continue
        for value in operation.inputs:
            if not _reference(value.value_type): continue
            term = known.get(value.port)
            if term is None or term.kind not in ('NONE','BUILTIN_SCALAR','TAINTED_RETURN','PATH'):
                reject('FINALIZER_PROTOCOL_UNRESOLVED', operation=index)
            elif term.kind == 'PATH' and term.path not in conditions and term.path not in callable_paths:
                reject('UNRETAINED_RELEASE_PATH', operation=index, path=list(term.path))
    for record in ledger['operations']:
        if record['role'] == 'ORDINARY_RECEIVER_ATTRIBUTE_READ':
            output = next((v.port for v in flow.operations[record['index']].outputs
                if _reference(v.value_type) and v.port.completion is C.FALLTHROUGH), None)
            term = known.get(output)
            if term and term.path not in conditions and term.path not in callable_paths:
                reject('ATTRIBUTE_VALUE_PROTOCOL_NOT_GUARDED', operation=record['index'], path=list(term.path))
    original.check()
    plan = dict(schema=SCHEMA,schema_version=1,source=ledger['source'],unit=unit,function_ordinal=function_ordinal,
        parameters=list(fn.parameter_names),owner_path=owner_path,ledger_id=ledger['id'],
        status='GUARDED_EFFECT_PROFILE_CLOSED' if not frontiers else 'FRONTIER',
        guards=sorted(conditions.values(),key=lambda x:tuple(x['path'])), calls=sorted(calls,key=lambda x:x['operation']),
        obligations=[obligations[i]for i in sorted(obligations)],frontiers=frontiers,
        retain_arguments_fields_and_descriptors_until_outer_exit=True,
        unknown_call_taint_inherited_by_nested_callers=True,
        outer_entry_invalidates_and_outer_exit_destroys=True,
        cross_activation_residency_authorized=False,original_caller_control_retained=True,
        runtime_guards_checked=False,default_backend='NO_OP',deployment_authorized=False)
    plan['id']='guarded-session-caller:'+semantic_sha256(plan)
    return plan,ledger
