"""Exclude only witness-backed effects incompatible with a selected CALL.

Reachability is may reachability over original matched recipe CALL/RETURN
states. It does not prove that retained operations must execute, that a CALL
completed, or that any retained/excluded operation is intrinsically pure.
"""
from collections import defaultdict,deque
from types import SimpleNamespace as NS

from scar.analysis.source_call_view import SourceFunctionCallProvider,function_interface_fact,source_call_site
from scar.analysis.source_namespace_interfaces import NamespacePlanView
from scar.analysis.source_reference_slice import ReferenceSlicer,source_port
from scar.analysis.source_value_flow import project_value_flow
from scar.ir.program import normal_graph as G, normal_graph as T
from scar.ir.program import call_returns as R
from scar.ir.program.control import LanguageCompletionKind as C


_CONDITIONS=('ORIGINAL_WRAPPER_SOURCE_AND_NORMAL_CALL_INTERFACE_SELECTED',
    'SAME_SELECTED_CALLER_ACTIVATION_AND_RECIPE_CALLSITE_STACK',
    'SELECTED_CALL_OCCURS_AND_CALLER_COMPLETES_NORMALLY',
    'EXCEPTIONAL_RESUMED_AND_REPEATED_CONDITION_EVALUATIONS_NOT_COVERED')


def _reach(starts,adjacency):
    reached=set(starts);pending=list(starts)
    while pending:
        for target in adjacency.get(pending.pop(),()):
            if target not in reached:reached.add(target);pending.append(target)
    return reached


def _normal_window(flow,selected,*,max_states=20000,max_stack=128):
    unit=flow.body.unit;control=[e for e in flow.edges if e.kind is G.EdgeKind.CONTROL]
    if any(e.transition is G.Transition.BACKEDGE for e in control):
        return (),(),(),(R.SelectedWindowBoundary('WRAPPER_NORMAL_CFG_HAS_BACKEDGE'),)
    if any(e.transition is G.Transition.RESUME for e in control):
        return (),(),(),(R.SelectedWindowBoundary('WRAPPER_NORMAL_CFG_HAS_RESUME'),)
    raw=defaultdict(list)
    for edge in control:raw[edge.source].append(edge)
    entry=G.Port(flow.body.template,flow.body.region,G.PortKind.ENTER)
    states=[(entry,())];ids={states[0]:0};pending=deque((0,));adj=defaultdict(list);reverse=defaultdict(list)
    routes=defaultdict(list)
    while pending:
        index=pending.popleft();point,stack=states[index]
        for edge in raw.get(point,()):
            following=stack
            if edge.transition is G.Transition.CALL:
                if edge.site is None:raise ValueError('original normal CALL has no recipe CallSite')
                if len(stack)>=max_stack:
                    return (),(),(),(R.SelectedWindowBoundary('WRAPPER_RECIPE_CALL_STACK_FRONTIER'),)
                following=(*stack,edge.site)
            elif edge.transition is G.Transition.RETURN:
                if not stack or stack[-1]!=edge.site:continue
                following=stack[:-1]
            target=(edge.target,following)
            if target not in ids:
                if len(states)>=max_states:
                    return (),(),(),(R.SelectedWindowBoundary('WRAPPER_CONTROL_STATE_FRONTIER'),)
                ids[target]=len(states);states.append(target);pending.append(ids[target])
            target=ids[target];adj[index].append(target);reverse[target].append(index);routes[index].append((target,edge))
    degree={i:len(set(reverse[i]))for i in range(len(states))};ready=[i for i,n in degree.items()if not n];ordered=[]
    while ready:
        index=ready.pop();ordered.append(index)
        for target in set(adj[index]):
            degree[target]-=1
            if degree[target]==0:ready.append(target)
    if len(ordered)!=len(states):return (),(),(),(R.SelectedWindowBoundary('WRAPPER_NORMAL_CFG_NOT_ACYCLIC'),)
    exits={i for i,(point,stack)in enumerate(states)if not stack and point.template==flow.body.template
        and point.owner==flow.body.region and point.kind is G.PortKind.EXIT
        and point.completion in(C.RETURN,C.FALLTHROUGH)}
    normal=_reach(exits,reverse)
    chosen={i for i,(point,_)in enumerate(states)if point==selected.port and i in normal}
    if not chosen:return (),(),(),(R.SelectedWindowBoundary('SELECTED_CALL_HAS_NO_MATCHED_NORMAL_WRAPPER_PATH',selected),)
    prefix=_reach(chosen,reverse);suffix=_reach(chosen,adj)&normal;compatible=prefix|suffix
    coordinates=defaultdict(list)
    for i,(point,_)in enumerate(states):coordinates[point].append(i)
    prefix_ops=[];suffix_ops=[];excluded=[];boundaries=[];reach_cache={}
    forks=[]
    for source,edges in routes.items():
        for yes,selected_edge in edges:
            if selected_edge.condition is None or yes not in prefix:continue
            for no,other in edges:
                if(other.condition==selected_edge.condition and other.when is not selected_edge.when
                        and other.when is not None and no not in prefix):
                    forks.append((source,no,selected_edge,other))
    for operation in flow.operations:
        site=T.GlobalPort(unit,operation.entry);occurrences=coordinates.get(operation.entry,())
        if any(i in prefix for i in occurrences):prefix_ops.append(site)
        if any(i in suffix for i in occurrences):suffix_ops.append(site)
        if not occurrences or any(i in compatible for i in occurrences):continue
        if len(occurrences)!=1:
            boundaries.append(R.SelectedWindowBoundary('OPERATION_HAS_MULTIPLE_STRUCTURAL_CONTEXTS',site));continue
        target=occurrences[0];witness=None
        for fork,bad,good_edge,bad_edge in forks:
            if bad not in reach_cache:reach_cache[bad]=_reach((bad,),adj)
            if target in reach_cache[bad]:
                witness=R.SelectedBranchExclusion(site,good_edge,bad_edge,states[fork][1]);break
        if witness is None:
            boundaries.append(R.SelectedWindowBoundary('NO_NORMAL_CO_REACHABILITY_WITHOUT_BRANCH_EXCLUSION_WITNESS',site))
        else:excluded.append(witness)
    return tuple(prefix_ops),tuple(suffix_ops),tuple(excluded),tuple(boundaries)


def derive_selected_call_window(source_calls,unit,interface_ordinal):
    """Derive a window from one existing exact original source CALL selection."""
    if type(source_calls)is not SourceFunctionCallProvider:raise TypeError('original source CALL provider required')
    if unit not in source_calls.available_units:raise ValueError('selected CALL unit is outside its original interface store')
    graph=source_calls.store.read_unit(unit)
    if type(interface_ordinal)is not int or not 0<=interface_ordinal<len(graph.interfaces):
        raise ValueError('original interface ordinal required')
    interface=graph.interfaces[interface_ordinal]
    if interface.activation is not T.FunctionActivation.NORMAL:
        raise ValueError('selected window requires an original ordinary normal CALL interface')
    source_calls.attach(unit,queries=(interface.query,));frame=source_calls.frames[unit,interface_ordinal]
    view=source_calls.view;source=view.unit(unit);operation,_,_=source_call_site(view,interface.link)
    step=source.records[operation.node]
    context=NamespacePlanView(unit,source.plans[interface.link.result.port.template]);owner=context.owner(step.region).owner
    functions=[(i,f)for i,f in enumerate(source.facts.functions)if f.body_scope is not None and f.body_scope.owner==owner]
    if len(functions)!=1:raise ValueError('selected CALL has no unique original owning function')
    index,function=functions[0];body=T.Body(unit,function.template,function.body)
    flow=project_value_flow(source.material,source_port(source.facts,function.value),body)
    selected=T.GlobalPort(unit,G.Port(interface.link.result.port.template,step.id,G.PortKind.ENTER))
    prefix,suffix,excluded,boundaries=_normal_window(flow,selected)
    slicer=ReferenceSlicer(NS(fact_store=source.material.fact_store))
    call=source.facts.calls[interface.query]
    result=R.SelectedCallEffectWindow(frame,interface.call,function_interface_fact(source_calls.store,unit,interface_ordinal),
        slicer.fact(source.facts,'functions',index),flow.source_templates,body,selected,
        source_port(source.facts,call.world),source_port(source.facts,call.result_world),
        prefix,suffix,excluded,boundaries,(*interface.link.conditions,*_CONDITIONS))
    source_calls.store.validate();source.check()
    return result


__all__=['derive_selected_call_window']
