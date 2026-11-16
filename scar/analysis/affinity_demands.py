"""Propagate a source-proven serial caller pin through closed emitted work."""
from dataclasses import replace
from pathlib import Path

from scar.analysis.implementation_envelope import derive_stateful_envelope
from scar.analysis.target_graph import TargetHardwareGraph
from scar.ir.execution_recipe import artifact_reference,read_artifact
from scar.ir.program import target_hardware as H
from scar.ir.record_codec import encode
from scar.ir.versioned_schedule import (TemporalMode,TemporalOccupancyKind as OK,
    VersionedTaskKind as K,validate_versioned_schedule)
from scar.optimization.schedule import schedule_from_dict


def derive_serial_affinity_demands(schedule,recipe,target):
    """Refine only unknown calling-thread execution demand inside a proved pin."""
    validate_versioned_schedule(schedule)
    if schedule.temporal is None or schedule.temporal.mode is not TemporalMode.SEMANTIC:
        raise ValueError('original semantic implementation time graph required')
    replay=derive_stateful_envelope(schedule_from_dict(recipe['task_graph']),recipe)
    if replace(schedule,temporal=None)!=replay:
        raise ValueError('complete standalone implementation envelope differs from original replay')
    if recipe['control'].get('spatial_partition')!='SERIAL_ROWS':
        raise ValueError('only the original single-worker serial renderer is supported')
    plan=read_artifact(recipe['artifacts']['stage_plan']);binding=plan['resource_binding']
    cpus=binding['cpu_ids']
    if (type(plan['cpu_threads'])is not int or plan['cpu_threads']!=1 or type(cpus)not in(list,tuple)or len(cpus)!=1
            or type(cpus[0])is not int):raise ValueError('one original selected CPU required')
    if type(target)is TargetHardwareGraph:graph=target
    else:
        path=Path(target).resolve();graph=TargetHardwareGraph(path.parent if path.is_file()else path)
    snapshot=artifact_reference(graph.path/'result.json','scar.target-hardware-graph')
    if (snapshot!=recipe['artifacts']['target_snapshot']or binding['target_sha256']!=snapshot['sha256']
            or Path(binding['target_path']).resolve()!=graph.path.resolve()):
        raise ValueError('selected CPU and original target snapshot differ')
    resources=graph.resources();threads=[r for r in resources if type(r)is H.HardwareThread and r.cpu_id==cpus[0]]
    executors=[r for r in resources if type(r)is H.CpuExecutor and str(r.id)==binding['executor_id']and cpus[0]in r.cpu_ids]
    if len(threads)!=1 or threads[0].online is False or len(executors)!=1:
        raise ValueError('selected hardware thread/executor is absent or offline in original target')
    thread=threads[0];resource=str(thread.id)
    links=graph.connections()
    if not any(e.kind is H.ConnectionKind.CONTAINS and e.source==executors[0].id and e.target==thread.id for e in links):
        raise ValueError('original executor/thread containment is missing')
    temporal=schedule.temporal;activities={a.task:a for a in temporal.activities}
    if len(activities)!=len(temporal.activities)or set(activities)!={t.id for t in schedule.tasks}:
        raise ValueError('one original synchronous lifecycle activity per task required')
    def phase(name):
        rows=[t for t in schedule.tasks if t.kind is K.IMPLEMENTATION_PHASE and t.attrs.get('phase_kind')==name]
        if len(rows)!=1:raise ValueError('one original '+name+' phase required')
        return rows[0]
    enter,restore,exit=phase('TEAM_ENTER'),phase('CALLER_RESTORE'),phase('WORKER_EXIT')
    if enter.attrs['phase_effects']!=['BIND_CALLER_TO_SELECTED_SINGLE_CPU']:
        raise ValueError('serial successful-bind contract differs')
    adjacency={}
    for edge in temporal.constraints:
        if not edge.iteration_distance and edge.min_delay_ticks>=0:
            adjacency.setdefault(edge.before,set()).add(edge.after)
    for activity in temporal.activities:adjacency.setdefault(activity.start,set()).add(activity.completion)
    def reaches(before,after):
        pending=[before];seen={before}
        while pending:
            point=pending.pop()
            if point==after:return True
            for following in adjacency.get(point,()):
                if following not in seen:seen.add(following);pending.append(following)
        return False
    by_activity={}
    for occupancy in temporal.occupancies:
        if occupancy.kind is OK.EXECUTION_DEMAND:by_activity.setdefault(occupancy.activity,[]).append(occupancy)
    if not reaches(activities[enter.id].completion,activities[restore.id].start):
        raise ValueError('original bind/restore temporal happens-before window is missing')
    selected={};trace=[];inside=False;state='UNKNOWN'
    closed={K.SNAPSHOT:'CPU_HOST_SNAPSHOT',K.STATE_WRITE:'CPU_STATE_WRITE'}
    for task in schedule.tasks:
        if task.id==enter.id:
            inside=True;state='BOUND';trace.append(dict(task=task.id,after=state,reason='SUCCESSFUL_ORIGINAL_SERIAL_BIND'));continue
        if task.id==restore.id:
            inside=False;trace.append(dict(task=task.id,before=state,after='UNKNOWN',reason='CALLER_RESTORE_TRANSITION'));break
        if not inside:continue
        activity=activities[task.id]
        if (not reaches(activities[enter.id].completion,activity.start)
                or not reaches(activity.completion,activities[restore.id].start)):
            raise ValueError('activity lacks original bind-completion/restore-start temporal bounds')
        allowed=(task.kind in closed and closed[task.kind]==task.capability or task.kind is K.COMPUTE
            and task.capability in('CPU_SCALAR_DAG','CPU_SCALAR_SETUP')or task.id==exit.id)
        if task.attrs.get('stage_kind')=='PROVIDER':allowed=False
        before=state
        if not allowed:
            state='UNKNOWN';reason='OPAQUE_SCOPE_HAS_NO_AFFINITY_PRESERVATION_CONTRACT'
        elif state!='BOUND':reason='NO_REBIND_AFTER_OPAQUE_SCOPE'
        else:
            rows=by_activity.get(activity.id,())
            if len(rows)!=1:raise ValueError('one original execution occupancy per closed activity required')
            old=rows[0]
            if old.resource==resource:
                if old.units not in(None,1):raise ValueError('original selected-thread demand conflicts with serial proof')
            elif old.units is not None or temporal.resource_capacities[old.resource]is not None:
                raise ValueError('cannot overwrite an existing different execution resource demand')
            selected[old.id]=task.id;reason='CLOSED_SOURCE_WORK_PRESERVES_SELECTED_CALLING_THREAD'
        trace.append(dict(task=task.id,before=before,after=state,reason=reason,mapped=reason.startswith('CLOSED_SOURCE')))
    if temporal.resource_capacities.get(resource)not in(None,1):
        raise ValueError('original hardware-thread capacity conflicts with unit serial demand')
    envelope=schedule.proof['implementation_envelope']
    evidence=dict(schema='scar.serial-affinity-demand-source-proof',schema_version=1,
        recipe_id=recipe['recipe_id'],backend_source_pins=envelope['backend_source_pins'],
        selected_thread=dict(snapshot=snapshot,resource=encode(thread.id)),thread_evidence=encode(thread.evidence),
        stage_plan=recipe['artifacts']['stage_plan'],selected_cpu_id=cpus[0],
        bind_completion=activities[enter.id].completion,restore_start=activities[restore.id].start,
        transition_trace=trace,source_qualification='SUCCESSFUL_STANDALONE_SERIAL_NORMAL_PATH',
        opaque_affinity_effects_invalidate_forward_knowledge=True,global_reservation=False,
        actual_cpu_observation=False,cost_inferred=False,native_execution_authorized=False)
    def prior_refs(refs):return tuple(r for r in refs if r.get('schema')!=evidence['schema'])
    occupancies=tuple(replace(o,resource=resource,units=1,source_refs=(*prior_refs(o.source_refs),evidence))
        if o.id in selected else replace(o,source_refs=prior_refs(o.source_refs))for o in temporal.occupancies)
    capacities=dict(temporal.resource_capacities)
    if selected:capacities[resource]=1
    result=replace(schedule,temporal=replace(temporal,occupancies=occupancies,resource_capacities=capacities,
        source_refs=(*prior_refs(temporal.source_refs),evidence)))
    validate_versioned_schedule(result);read_artifact(snapshot)
    return result


__all__=['derive_serial_affinity_demands']
