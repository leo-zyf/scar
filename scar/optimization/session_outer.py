"""Compose a shared outer source loop with independently guarded native leaves.

Scalar kinds are conditional information, not value identities. Unknown host
calls stay original. Until all host effects are closed, every leaf invocation
owns an isolated lease; this does not claim an I/O quiescence proof.
"""
from scar.analysis.session_scalar_flow import derive_scalar_flow
from scar.analysis.session_caller_effects import _source
from scar.analysis.source_body_interfaces import BodyInterfaceResolver
from scar.analysis.source_reference_slice import source_port
from scar.analysis.source_value_flow import project_value_flow
from scar.compiler.stateful_frontend import _reachable
from scar.ir.program import normal_links as F, source_productions as P
from scar.ir.program.normal_targets import Body
from scar.ir.record_codec import encode
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.guarded-session-outer'


def derive_session_outer(view, *, unit, function_ordinal, nested_profiles, resolver=None):
    original=view.unit(unit);facts=original.facts;fn=facts.functions[function_ordinal]
    resolver=resolver or BodyInterfaceResolver(view)
    source=encode(_source(view,unit,fn)[0]);owner=source['qualname'].split('.')[:-1]
    receiver=fn.parameter_names[0] if fn.parameter_names else None
    nested={p['source']['qualname'].split('.')[-1]:p for p in nested_profiles
        if p['source']['qualname'].split('.')[:-1]==owner
        and p['source']['path']==source['path'] and p['status']=='GUARDED_EFFECT_PROFILE_CLOSED'}
    body=Body(unit,fn.template,fn.body)
    flow=project_value_flow(original.material,source_port(facts,fn.value),body)
    attributes={alias:row for row in facts.attributes if row.action is F.AttributeAction.GET and row.name in nested
        for alias in _reachable(facts,row.value)}
    proposals={}
    for record in flow.calls:
        attr=attributes.get(record.call.callee)
        if attr is not None and attr.name in nested:
            matches=[i for i,c in enumerate(facts.calls) if c==record.call]
            if len(matches)==1:proposals[matches[0]]='NUMBER'
    scalar=derive_scalar_flow(view,unit=unit,function_ordinal=function_ordinal,
        known_call_returns=proposals,resolver=resolver,flow=flow)
    ports=scalar['ports'];calls=[];frontiers=[];nested_ids=set()
    for operation in scalar['operations']:
        call=operation['call']
        if call is None:continue
        paths=ports[operation['inputs'][0]]['reference_paths'] if operation['inputs'] else []
        kind='ORIGINAL_HOST_CALL';boundary=None
        if (len(paths)==1 and len(paths[0])==2 and paths[0][0]==receiver and paths[0][1] in nested):
            child=nested[paths[0][1]];kind='NESTED_CALLER';nested_ids.add(child['id'])
            boundary=dict(id=child['id'],source=child['source'],owner_path=child['owner_path'],
                attribute=paths[0][1],descriptor='METHOD')
        route=dict(reference_paths=paths,original_callee=call['call']['callee'],
            original_call_fact=call['fact'],binding_checked=False)
        calls.append(dict(operation=operation['index'],query_ordinal=call['query_ordinal'],
            role=kind,boundary=boundary,lookup_route=route,
            policy='GUARDED_INDEPENDENT_LEAF_ONLY' if kind=='NESTED_CALLER'
                else 'UNWRAPPED_ORIGINAL_WITH_NO_LEASE_ACROSS_HOST'))
    loops=[]
    for key,plan in original.material.closure(fn.template).items():
        context=resolver.plan(unit,key)
        for step in plan.steps:
            if context.owner(step.region).owner!=fn.body_scope.owner:continue
            if type(step.instruction) is P.LoopControl:
                loops.append(dict(region=encode(step.region),site=encode(context.site(step)),
                    control=encode(step.instruction),shared_body=True,iteration_count=None))
            elif type(step.instruction) is P.SuspendExecution:
                frontiers.append(dict(reason='RESUMPTION_OUTSIDE_OUTER_SCOPE_PROFILE',site=encode(context.site(step))))
    if not receiver or not owner or '<locals>' in owner:
        frontiers.append(dict(reason='ORDINARY_CLASS_METHOD_SCOPE_REQUIRED'))
    if not nested_ids:frontiers.append(dict(reason='NO_DIRECT_NESTED_GUARDED_CALLER'))
    if not loops:frontiers.append(dict(reason='NO_ORIGINAL_SHARED_LOOP'))
    guards=[dict(path=[receiver],kind='ORDINARY_RECEIVER')] if receiver else []
    # Guard specifications remain source/port obligations. They are not
    # silently promoted to checked entry guards for mutated loop locals.
    for row in scalar['guards']:
        if row['opcode']!='CHECK_EXACT_BUILTIN_SCALAR_KINDS':continue
        for path in row.get('paths',[]):
            guards.append(dict(path=path,kind='CONDITIONAL_EXACT_SCALAR_KINDS',
                allowed=row['kinds'],placement=row['placement'],checked=False))
    points=[dict(operation=c['operation'],kind='ORIGINAL_KERNEL_SOURCE_AND_INPUT_GUARDS',
        nested_caller_id=c['boundary']['id'],retention_allowed=False,
        required_guards='INDEPENDENT_LEAF_SOURCE_ABI_ROOT_PROVIDER_METADATA_GUARDS',
        isolated_lease_required=True)for c in calls if c['role']=='NESTED_CALLER']
    result=dict(schema=SCHEMA,schema_version=1,
        status='FRONTIER' if frontiers else 'CONDITIONAL_SCALAR_MODEL_WITH_ISOLATED_KERNEL_DELIVERY',
        source=source,unit=unit,function_ordinal=function_ordinal,parameters=list(fn.parameter_names),
        body=encode(body),owner_path=owner,ledger_id=scalar['id'],
        original_control_refs=[dict(artifact=scalar['original_f4'],locator='#')],
        scalar_control_edge_ordinals=[i for i,e in enumerate(scalar['edges'])if e['kind']=='CONTROL'],
        control_index_space='REFERENCED_SCALAR_FLOW_EDGES_NOT_F4_PUBLICATION_ORDINALS',
        operations=len(scalar['operations']),calls=calls,guards=guards,nested_caller_ids=sorted(nested_ids),
        shared_loops=loops,field_writes=scalar['field_writes'],qualification_points=points,
        frontiers=frontiers,effect_profile_closed=False,conditional_scalar_guards_checked=False,
        original_control_and_host_calls_retained=True,host_calls_dynamically_intercepted=False,
        unqualified_kernel_policy='FRESH_ISOLATED_NATIVE',
        cross_host_residency_authorized=False,opaque_call_quiescence_assumed=False,
        default_backend='NO_OP',deployment_authorized=False)
    result['id']='guarded-session-outer:'+semantic_sha256(result)
    return result,scalar


__all__=['derive_session_outer','SCHEMA']
