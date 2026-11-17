"""Joint implementation/location scheduling using optional CP-SAT intervals.

No target variants are executed. Source admission belongs to producers; this
module solves their explicit point model and separately checks the resulting
physical resource timeline. Empirical ranges are not mathematical bounds.
"""
from math import floor
import time

from scar.ir.implementation_plan import validate_problem, digest, PLAN_SCHEMA


def _known(cost):
    return cost.estimate_ns is not None and cost.basis!='UNKNOWN' and not cost.missing_terms


def _frontiers(problem, baseline_only):
    result=[]
    for op in problem.operations:
        for mode in op.options:
            if (not baseline_only or mode.id==op.baseline_option)and not _known(mode.cost):
                result.append(dict(operation=op.id,option=mode.id,reason='INCOMPLETE_CONTEXT_COST',
                                   missing_terms=list(mode.cost.missing_terms)))
    for edge in problem.data_edges:
        for route in edge.transfers:
            if not _known(route.cost):
                result.append(dict(edge=edge.id,option=route.id,reason='INCOMPLETE_TRANSFER_COST'))
    baselines={op.id:op.baseline_option for op in problem.operations}
    for activity in problem.activities:
        reachable=not activity.when or not baseline_only or any(baselines[c.operation]==c.option for c in activity.when)
        if reachable and not _known(activity.cost):
            result.append(dict(activity=activity.id,reason='INCOMPLETE_CONTEXT_ACTIVITY_COST',
                when=[dict(operation=c.operation,option=c.option)for c in activity.when],
                missing_terms=list(activity.cost.missing_terms)))
    return result


def solve_implementations(problem, *,max_seconds=30,workers=4,baseline_only=False):
    """Select legal modes and data routes, then minimize completion time.

    Unknown alternatives remain in the domain report, so a known-subset optimum
    is never labelled a full-domain optimum. Memory covers scoped private
    payload only: outputs counted once, destination copies separately, external
    roots borrowed, and reservations independent of logical content versions.
    """
    p=validate_problem(problem)
    if type(workers)is not int or not 1<=workers<=64 or type(max_seconds)not in(int,float)or not 0<max_seconds<=3600:
        raise ValueError('positive bounded solver time and worker count required')
    try:
        from ortools.sat.python import cp_model
        from importlib.metadata import version
    except ImportError as error:
        raise RuntimeError('joint planning requires the optional scar[planning] dependency; runtime delivery does not')from error
    clock=time.monotonic();frontiers=_frontiers(p,baseline_only)
    durations=[m.cost.estimate_ns for o in p.operations for m in o.options if _known(m.cost)]
    durations.extend(t.cost.estimate_ns for e in p.data_edges for t in e.transfers if _known(t.cost))
    durations.extend(a.cost.estimate_ns for a in p.activities if _known(a.cost))
    horizon=max(1,sum(durations)+sum(c.minimum_ns for c in p.timing_constraints))
    weight=len(p.operations)+1
    if horizon*weight>=2**52:raise ValueError('planning horizon exceeds exact integer reporting range; use a coarser scoped model')
    model=cp_model.CpModel();resources={r.id:r for r in p.resources};ops={o.id:o for o in p.operations}
    starts={o.id:model.new_int_var(0,horizon,'start:'+o.id)for o in p.operations}
    ends={o.id:model.new_int_var(0,horizon,'end:'+o.id)for o in p.operations}
    choices={};holds={r.id:[]for r in p.resources};demands={r.id:[]for r in p.resources}
    activity_starts={};activity_ends={};activity_presence={};true=model.new_constant(1)
    routes=[];source_uses={};output_leases=[]
    def occupy(resource,interval,units):
        if units:holds[resource].append(interval);demands[resource].append(units)
    def conjunction(bits,name):
        if len(bits)==1:return bits[0]
        active=model.new_bool_var(name)
        for bit in bits:model.add(active<=bit)
        model.add(active>=sum(bits)-len(bits)+1)
        return active
    def conditional_time(active,value,name):
        result=model.new_int_var(0,horizon,name)
        model.add(result==value).only_enforce_if(active)
        model.add(result==0).only_enforce_if(active.Not())
        return result
    def condition(when,name):
        if not when:return true
        bits=[choices[c.operation,c.option]for c in when]
        if len(bits)==1:return bits[0]
        result=model.new_bool_var(name)
        model.add_max_equality(result,bits)
        return result
    for op in p.operations:
        bits=[]
        for mode in op.options:
            bit=model.new_bool_var(op.id+':'+mode.id);choices[op.id,mode.id]=bit;bits.append(bit)
            if not _known(mode.cost)or baseline_only and mode.id!=op.baseline_option:
                model.add(bit==0);continue
            interval=model.new_optional_interval_var(starts[op.id],mode.cost.estimate_ns,ends[op.id],bit,mode.id)
            for resource,units in mode.demands.items():occupy(resource,interval,units)
            for pool,nbytes in mode.workspace_bytes.items():occupy(pool,interval,nbytes)
        model.add_exactly_one(bits)
        for predecessor in op.predecessors:model.add(starts[op.id]>=ends[predecessor])
    for activity in p.activities:
        start=model.new_int_var(0,horizon,'activity-start:'+activity.id)
        end=model.new_int_var(0,horizon,'activity-end:'+activity.id)
        active=condition(activity.when,'activity-presence:'+activity.id)
        activity_starts[activity.id]=start;activity_ends[activity.id]=end;activity_presence[activity.id]=active
        model.add(start==0).only_enforce_if(active.Not())
        model.add(end==0).only_enforce_if(active.Not())
        if not _known(activity.cost):
            model.add(active==0);continue
        if activity.duration_kind=='WAIT_TAIL':
            # Wait is elapsed latency, not an interval of busy CPU service.
            model.add(end>=start+activity.cost.estimate_ns).only_enforce_if(active)
        else:
            interval=model.new_optional_interval_var(start,activity.cost.estimate_ns,end,active,activity.id)
            for resource,units in activity.demands.items():occupy(resource,interval,units)
    endpoints={name:dict(START=start,END=ends[name])for name,start in starts.items()}
    endpoints.update({name:dict(START=start,END=activity_ends[name])for name,start in activity_starts.items()})
    for constraint in p.timing_constraints:
        active=conjunction([condition(constraint.when,'constraint-condition:'+constraint.id),
            activity_presence.get(constraint.before,true),activity_presence.get(constraint.after,true)],
            'constraint-presence:'+constraint.id)
        delta=endpoints[constraint.after][constraint.after_endpoint]-endpoints[constraint.before][constraint.before_endpoint]
        model.add(delta>=constraint.minimum_ns).only_enforce_if(active)
        if constraint.maximum_ns is not None:model.add(delta<=constraint.maximum_ns).only_enforce_if(active)
    for edge in p.data_edges:
        producers=[None]if edge.producer is None else ops[edge.producer].options
        for producer in producers:
            source=edge.external_location if producer is None else producer.output_locations[edge.value]
            for consumer in ops[edge.consumer].options:
                destination=consumer.input_locations[edge.value]
                bits=[choices[edge.consumer,consumer.id]]
                if producer is not None:bits.append(choices[edge.producer,producer.id])
                active=conjunction(bits,'edge:'+edge.id+':'+str(source)+':'+consumer.id)
                use_key=None if producer is None else(edge.producer,producer.id,edge.value)
                if source==destination:
                    if edge.producer is not None:
                        model.add(starts[edge.consumer]>=ends[edge.producer]).only_enforce_if(active)
                        source_uses.setdefault(use_key,[]).append(conditional_time(active,ends[edge.consumer],'use:'+edge.id+':'+consumer.id))
                    continue
                route_bits=[]
                for transfer in edge.transfers:
                    if transfer.source_memory!=source or transfer.destination_memory!=destination:continue
                    bit=model.new_bool_var('transfer:'+edge.id+':'+transfer.id+':'+consumer.id+':'+str(None if producer is None else producer.id))
                    route_bits.append(bit)
                    if not _known(transfer.cost):model.add(bit==0);continue
                    start=model.new_int_var(0,horizon,'transfer-start:'+edge.id)
                    end=model.new_int_var(0,horizon,'transfer-end:'+edge.id)
                    interval=model.new_optional_interval_var(start,transfer.cost.estimate_ns,end,bit,'transfer:'+edge.id)
                    model.add(start>=0 if producer is None else start>=ends[edge.producer]).only_enforce_if(bit)
                    model.add(end<=starts[edge.consumer]).only_enforce_if(bit)
                    for resource,units in transfer.demands.items():occupy(resource,interval,units)
                    if edge.nbytes:
                        size=model.new_int_var(0,horizon,'copy-lifetime:'+edge.id)
                        model.add(size==ends[edge.consumer]-start).only_enforce_if(bit)
                        life=model.new_optional_interval_var(start,size,ends[edge.consumer],bit,'copy-storage:'+edge.id)
                        occupy(destination,life,edge.nbytes)
                    if use_key is not None:source_uses.setdefault(use_key,[]).append(conditional_time(bit,end,'source-use:'+edge.id))
                    routes.append((edge,transfer,bit,start,end))
                model.add(sum(route_bits)==active)
    # One producer materialization per logical value, shared by all readers.
    for op in p.operations:
        for mode in op.options:
            for value,pool in mode.output_locations.items():
                edge=next(e for e in p.data_edges if e.producer==op.id and e.value==value)
                release=model.new_int_var(0,horizon,'last-use:'+op.id+':'+value+':'+mode.id)
                model.add_max_equality(release,[ends[op.id],*source_uses.get((op.id,mode.id,value),[])])
                size=model.new_int_var(0,horizon,'source-lifetime:'+op.id+':'+value)
                model.add(size==release-starts[op.id])
                interval=model.new_optional_interval_var(starts[op.id],size,release,choices[op.id,mode.id],'output:'+value)
                occupy(pool,interval,edge.nbytes)
    for pool,nbytes in p.reserved_memory.items():
        if nbytes:occupy(pool,model.new_interval_var(0,horizon,horizon,'reserved:'+pool),nbytes)
    for resource in p.resources:
        if holds[resource.id]:model.add_cumulative(holds[resource.id],demands[resource.id],resource.capacity)
    finish=model.new_int_var(0,horizon,'makespan');model.add_max_equality(finish,[*ends.values(),*activity_ends.values()])
    changes=sum(bit for (name,mode),bit in choices.items()if mode!=ops[name].baseline_option)
    model.minimize(finish*weight+changes)
    solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=float(max_seconds)
    solver.parameters.num_search_workers=workers;solver.parameters.random_seed=0
    status=solver.solve(model);feasible=status in(cp_model.OPTIMAL,cp_model.FEASIBLE)
    extended=bool(p.activities or p.timing_constraints)
    result=dict(schema=PLAN_SCHEMA,schema_version=2 if extended else 1,problem_sha256=digest(p),
        status=('OPTIMAL_WITHIN_MODEL'if status==cp_model.OPTIMAL and not frontiers else
                'FEASIBLE_PARTIAL_DOMAIN'if feasible and frontiers else'FEASIBLE_WITHIN_MODEL'if feasible else
                'UNRESOLVED'if frontiers or status==cp_model.UNKNOWN else'INFEASIBLE_WITHIN_MODEL'),
        solver_status=solver.status_name(status),domain_complete=not frontiers,frontiers=frontiers,
        optimization_domain='BASELINE_IMPLEMENTATIONS_ONLY'if baseline_only else'ALL_DECLARED_IMPLEMENTATION_OPTIONS',
        tie_break='MINIMUM_REPLACEMENTS_AT_EQUAL_POINT_MODEL_COMPLETION_TIME',
        baseline_only=baseline_only,selected=[],transfers=[],memory_leases=[],makespan_ns=None,
        model_lower_bound_ns=None,relative_gap=None,model_optimality_proven=status==cp_model.OPTIMAL and not frontiers,
        physical_optimality_proven=False,cost_basis='EXPLICIT_CONTEXT_POINT_ESTIMATES_NOT_PHYSICAL_TIME_GUARANTEES',
        memory_accounting='SCOPED_INCREMENTAL_PRIVATE_PAYLOAD_EXTERNAL_INPUTS_BORROWED',
        external_input_bytes_not_charged=sum({e.value:e.nbytes for e in p.data_edges if e.producer is None}.values()),
        solver=dict(name='OR_TOOLS_CP_SAT',version=version('ortools'),max_seconds=max_seconds,workers=workers),
        elapsed_seconds=time.monotonic()-clock,target_program_executed=False,execution_authorized=False,
        deployment_authorized=False,default_backend='NO_OP')
    if extended:result['activities']=[]
    if not feasible:return result
    result['selected']=[dict(operation=o.id,option=m.id,start_ns=solver.value(starts[o.id]),end_ns=solver.value(ends[o.id]))
                        for o in p.operations for m in o.options if solver.value(choices[o.id,m.id])]
    if extended:
        result['activities']=[dict(activity=a.id,start_ns=solver.value(activity_starts[a.id]),end_ns=solver.value(activity_ends[a.id]))
            for a in p.activities if solver.value(activity_presence[a.id])]
    result['transfers']=[dict(edge=e.id,option=t.id,source_memory=t.source_memory,destination_memory=t.destination_memory,
        start_ns=solver.value(start),end_ns=solver.value(end))for e,t,bit,start,end in routes if solver.value(bit)]
    total=solver.value(finish);lower=total if status==cp_model.OPTIMAL else max(0,floor(solver.best_objective_bound/weight))
    result.update(makespan_ns=total,
        model_lower_bound_ns=None if frontiers else lower,
        relative_gap=None if frontiers else(total-lower)/max(1,total),
        known_subdomain_lower_bound_ns=lower,known_subdomain_gap=(total-lower)/max(1,total))
    result['memory_leases']=_timeline(p,result)[1]
    verify_execution_plan(p,result)
    return result


def _timeline(p,plan):
    """Reconstruct all uses independently of optional-interval solver variables."""
    ops={o.id:o for o in p.operations};selected={};execution=[];leases=[]
    def require(condition,message):
        if not condition:raise ValueError(message)
    def interval(row):
        require(type(row['start_ns'])is int and type(row['end_ns'])is int
                and 0<=row['start_ns']<=row['end_ns'],'invalid execution interval')
    require(type(plan['selected'])is list and len(plan['selected'])==len(ops),'one selection per original region required')
    for row in plan['selected']:
        require(type(row)is dict and set(row)=={'operation','option','start_ns','end_ns'},'selection wire fields differ')
        require(row['operation']in ops and row['operation']not in selected,'selection operation differs or duplicates')
        op=ops[row['operation']];modes=[m for m in op.options if m.id==row['option']]
        require(len(modes)==1,'selected implementation absent from original domain');mode=modes[0];interval(row)
        require(_known(mode.cost)and row['end_ns']-row['start_ns']==mode.cost.estimate_ns,'selected duration lacks exact context estimate')
        require(not plan['baseline_only']or mode.id==op.baseline_option,'baseline report selected a replacement')
        selected[op.id]=(row,mode)
        execution.extend((r,row['start_ns'],row['end_ns'],u)for r,u in mode.demands.items())
        leases.extend(dict(id='workspace:'+op.id+':'+pool,memory=pool,bytes=nbytes,start_ns=row['start_ns'],end_ns=row['end_ns'],kind='WORKSPACE')
                      for pool,nbytes in mode.workspace_bytes.items()if nbytes)
    for name,(row,_)in selected.items():
        require(all(selected[d][0]['end_ns']<=row['start_ns']for d in ops[name].predecessors),'source/effect dependency violated')
    def present(when):return not when or any(selected[c.operation][1].id==c.option for c in when)
    actual={};expected={a.id:a for a in p.activities if present(a.when)}
    auxiliary=plan.get('activities',[])
    require(type(auxiliary)is list and len(auxiliary)==len(expected),'conditional activity coverage differs')
    for row in auxiliary:
        require(type(row)is dict and set(row)=={'activity','start_ns','end_ns'},'activity wire fields differ')
        name=row['activity'];require(name in expected and name not in actual,'conditional activity absent or duplicated')
        phase=expected[name];interval(row);duration=row['end_ns']-row['start_ns']
        require(_known(phase.cost)and(duration>=phase.cost.estimate_ns if phase.duration_kind=='WAIT_TAIL'
            else duration==phase.cost.estimate_ns),'activity duration lacks qualified context estimate')
        actual[name]=row
        execution.extend((r,row['start_ns'],row['end_ns'],u)for r,u in phase.demands.items())
    points={name:dict(START=row['start_ns'],END=row['end_ns'])for name,(row,_)in selected.items()}
    points.update({name:dict(START=row['start_ns'],END=row['end_ns'])for name,row in actual.items()})
    for constraint in p.timing_constraints:
        if not present(constraint.when)or constraint.before not in points or constraint.after not in points:continue
        delta=points[constraint.after][constraint.after_endpoint]-points[constraint.before][constraint.before_endpoint]
        require(delta>=constraint.minimum_ns and(constraint.maximum_ns is None or delta<=constraint.maximum_ns),
            'conditional event dependency violated: '+constraint.id)
    transfers={}
    for row in plan['transfers']:
        require(type(row)is dict and set(row)=={'edge','option','source_memory','destination_memory','start_ns','end_ns'},'transfer wire fields differ')
        require(row['edge']not in transfers,'duplicate transfer for an input edge');interval(row);transfers[row['edge']]=row
    uses={};edges={e.id:e for e in p.data_edges}
    require(set(transfers)<=set(edges),'transfer names absent data edge')
    for edge in p.data_edges:
        consumer,cmode=selected[edge.consumer]
        source=edge.external_location if edge.producer is None else selected[edge.producer][1].output_locations[edge.value]
        destination=cmode.input_locations[edge.value]
        ready=0 if edge.producer is None else selected[edge.producer][0]['end_ns']
        if source==destination:
            require(edge.id not in transfers and ready<=consumer['start_ns'],'same-location dependency violated')
            last=consumer['end_ns']
        else:
            require(edge.id in transfers,'missing required physical transfer');row=transfers[edge.id]
            routes=[t for t in edge.transfers if t.id==row['option']and t.source_memory==source and t.destination_memory==destination]
            require(len(routes)==1,'transfer implementation/location differs from original route');route=routes[0]
            require(row['source_memory']==source and row['destination_memory']==destination
                    and _known(route.cost)and row['end_ns']-row['start_ns']==route.cost.estimate_ns
                    and ready<=row['start_ns']and row['end_ns']<=consumer['start_ns'],'transfer duration/readiness/location violated')
            execution.extend((r,row['start_ns'],row['end_ns'],u)for r,u in route.demands.items())
            if edge.nbytes:leases.append(dict(id='input-copy:'+edge.id,memory=destination,bytes=edge.nbytes,
                start_ns=row['start_ns'],end_ns=consumer['end_ns'],kind='INPUT_MATERIALIZATION'))
            last=row['end_ns']
        if edge.producer is not None:uses.setdefault(edge.value,[]).append(last)
    for value,lasts in uses.items():
        edge=next(e for e in p.data_edges if e.value==value);row,mode=selected[edge.producer]
        if edge.nbytes:leases.append(dict(id='output:'+value,memory=mode.output_locations[value],bytes=edge.nbytes,
            start_ns=row['start_ns'],end_ns=max(row['end_ns'],*lasts),kind='OUTPUT_MATERIALIZATION'))
    for pool,nbytes in p.reserved_memory.items():
        if nbytes:leases.append(dict(id='reserved:'+pool,memory=pool,bytes=nbytes,start_ns=0,end_ns=plan['makespan_ns'],kind='SCOPE_RESERVATION'))
    return execution,sorted(leases,key=lambda r:r['id'])


def verify_execution_plan(problem,plan):
    """Check selected costs, causality, copies and peak occupancy without OR-Tools.

    This verifies feasibility, not source legality, cost accuracy or the solver's
    optimality claim. Source-qualified publishers must replay those separately.
    """
    p=validate_problem(problem)
    extended=bool(p.activities or p.timing_constraints)
    if plan.get('schema')!=PLAN_SCHEMA or plan.get('schema_version')!=(2 if extended else 1) or plan.get('problem_sha256')!=digest(p):
        raise ValueError('execution plan belongs to a different typed problem')
    if not plan.get('selected'):raise ValueError('no feasible execution plan to verify')
    if plan['frontiers']!=_frontiers(p,plan['baseline_only'])or plan['domain_complete']!=(not bool(plan['frontiers'])):
        raise ValueError('unknown implementation domain was discarded')
    if plan['model_optimality_proven']and(not plan['domain_complete']or plan['solver_status']!='OPTIMAL'):
        raise ValueError('unresolved or unfinished domain cannot claim model optimality')
    if plan['frontiers']and(plan['model_lower_bound_ns']is not None or plan['relative_gap']is not None):
        raise ValueError('known-subdomain bounds cannot claim bounds on unknown alternatives')
    execution,leases=_timeline(p,plan)
    if plan['memory_leases']!=leases:raise ValueError('memory leases differ from selected value lifetimes')
    finish=max([r['end_ns']for r in plan['selected']]+[r['end_ns']for r in plan['transfers']]
        +[r['end_ns']for r in plan.get('activities',[])])
    if plan['makespan_ns']!=finish:raise ValueError('completion time differs from actual selected schedule')
    resources={r.id:r for r in p.resources};events={r:[]for r in resources}
    for r,start,end,units in execution:
        if start!=end:events[r].extend(((start,units),(end,-units)))
    for row in leases:
        if row['start_ns']!=row['end_ns']:events[row['memory']].extend(((row['start_ns'],row['bytes']),(row['end_ns'],-row['bytes'])))
    for pool,nbytes in p.reserved_memory.items():
        if nbytes>resources[pool].capacity:raise ValueError('scope memory reservation exceeds budget')
    peaks={}
    for resource,records in events.items():
        current=peak=0
        for _,delta in sorted(records):
            current+=delta;peak=max(peak,current)
            if current<0 or current>resources[resource].capacity:raise ValueError('resource capacity exceeded: '+resource)
        if current:raise ValueError('unbalanced resource lifetime')
        peaks[resource]=peak
    return dict(feasible=True,resource_peaks=peaks,source_legality_replayed=False,
                cost_accuracy_proven=False,solver_optimality_independently_proven=False)


__all__=['solve_implementations','verify_execution_plan']
