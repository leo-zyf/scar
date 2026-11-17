"""Source-backed native ready-host phases on the original versioned graph.

Clock cuts bracket effects; they do not create atomic publication timestamps.
Waiting is represented by causal gaps/latencies, never a fixed CPU service.
The original Python call remains a larger, partially unknown containing scope.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from scar.codegen.pipeline_host_probe import CALLER_SLOTS, LANE_SLOTS, CALL_SLOTS
from scar.ir.record_codec import encode
from scar.ir.execution_recipe import artifact_reference
from scar.ir.versioned_schedule import (
    VersionedTask as Task, VersionedTaskKind as K, TemporalMode as Mode,
    TemporalEventKind as EK, TemporalScopeKind as SK, TemporalDurationBasis as DB,
    TemporalOccupancyKind as OK, TemporalConstraintKind as CK,
    TemporalEvent as Event, TemporalActivity as Activity, TemporalScope as Scope,
    TemporalOccupancy as Occupancy, TemporalConstraint as Constraint, TemporalModel as Model,
    schedule_document, validate_versioned_schedule)
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.pipeline-host-envelope'
_POLICY = 'FIXED_CPU_LANE_HEAD_ONLY_WAIT_TRUE_PREDECESSORS_NO_TIMESTAMP_WAIT'


def _need(condition, message):
    if not condition: raise ValueError(message)


def derive_pipeline_host_envelope(schedule, certificate, probe_metadata, *, trace=None, source_refs=()):
    """Refine a source-admitted v2 graph; no files, native calls or solver.

    Full source/ELF admission belongs to the publisher. This projection checks
    its structural identities and optionally validates one complete raw trace.
    """
    validate_versioned_schedule(schedule)
    _need(schedule.temporal is None, 'original untimed pipeline graph required')
    _need(all(t.kind in (K.ALLOCATE,K.COMPUTE,K.SUCCESS_FENCE,K.RELEASE,K.EXPORT)
              for t in schedule.tasks), 'closed original pipeline task profile required')
    _need(certificate.get('schema') == 'scar.pipeline-implementation-refinement'
          and certificate.get('schema_version') == 1 and certificate.get('dispatch_policy') == _POLICY,
          'source-qualified strict lane-head implementation certificate required')
    _need(probe_metadata.get('schema') == 'scar.pipeline-host-probe'
          and probe_metadata.get('schema_version') == 1
          and probe_metadata['original_execution_schedule_sha256'] == semantic_sha256(certificate),
          'probe must identify this exact selected native execution certificate')
    provenance = probe_metadata['provenance']
    _need(all(provenance.get(k) is True for k in ('original_source_inverse_equal',
        'original_operator_call_statement_unchanged', 'original_operator_libraries_unchanged',
        'original_pending_policy_unchanged', 'original_join_and_launch_order_unchanged'))
        and tuple(provenance['caller_slots']) == CALLER_SLOTS
        and tuple(provenance['lane_slots']) == LANE_SLOTS and tuple(provenance['call_slots']) == CALL_SLOTS,
        'original source-preserving probe/cut protocol required')
    computes = [t for t in schedule.tasks if t.kind is K.COMPUTE]
    by_call = {t.attrs['call_id']:t for t in computes}
    calls = probe_metadata['call_ids']; cpus = certificate['cpu_ids']
    _need(len(by_call) == len(computes) and calls == [t.attrs['call_id'] for t in computes]
        and set(calls) == set(certificate['call_cpu_ids']) and cpus == probe_metadata['lane_cpu_ids']
        and cpus and len(set(cpus)) == len(cpus)
        and all(t.attrs.get('cpu_id') == certificate['call_cpu_ids'][c] for c,t in by_call.items()),
        'original call/task and selected lane geometry differs')
    lanes = certificate['lane_call_order']
    _need(set(lanes) == {str(cpu) for cpu in cpus}
        and [c for cpu in cpus for c in lanes[str(cpu)]] and
        all(lanes[str(cpu)] == [c for c in certificate['call_order']
            if certificate['call_cpu_ids'][c] == cpu] for cpu in cpus)
        and sorted(c for lane in lanes.values() for c in lane) == sorted(calls),
        'strict source lane order must cover each selected call once')
    known_tasks = {t.id:t for t in schedule.tasks}
    original_preds = certificate.get('original_call_predecessors', certificate['call_predecessors'])
    for call,task in by_call.items():
        expected = {by_call[c].id for c in original_preds[call]}
        _need({p for p in task.depends_on if p in {t.id for t in computes}} == expected,
            'base task graph must retain the original source/pool predecessors')
    fences = [t for t in schedule.tasks if t.kind is K.SUCCESS_FENCE]
    exports = [t for t in schedule.tasks if t.kind is K.EXPORT]
    _need(len(fences) == len(exports) == 1, 'one original success and final-export task required')
    fence, exported = fences[0], exports[0]
    if trace is not None:
        from scar.analysis.pipeline_host_samples import validate_host_trace
        validate_host_trace(trace, call_predecessors=certificate['call_predecessors'],
            lane_call_order=lanes, expected_cpu_ids=cpus)
    raw_caller = {} if trace is None else trace['caller']
    raw_lanes = {} if trace is None else {r['cpu_id']:r['cuts'] for r in trace['lanes']}
    raw_calls = {} if trace is None else {r['call_id']:r['cuts'] for r in trace['calls']}
    python_wrapper=artifact_reference(Path(__file__).resolve().parents[1]/'codegen/pipeline.py',
        'scar.backend-implementation-source')
    seed = semantic_sha256(dict(region_id=schedule.region_id,call_scope=schedule.call_scope,
        original_task_ids=[t.id for t in schedule.tasks],original_source_sha256=provenance['original_source_sha256'],
        target_sha256=certificate['target']['sha256'],python_wrapper_sha256=python_wrapper['sha256'],
        control=dict(call_ids=calls,cpu_ids=cpus,call_cpu_ids=certificate['call_cpu_ids'],
            lane_call_order=lanes,call_predecessors=certificate['call_predecessors'],dispatch_policy=_POLICY),
        cut_protocol=dict(caller=CALLER_SLOTS,lane=LANE_SLOTS,call=CALL_SLOTS),
        probe_renderer_versions=sorted(provenance['source_pins'].values())))[:20]
    def identity(label): return 'host:'+seed+':'+label
    native = identity('native-abi-bracket'); root = schedule.call_scope
    worker_scopes = {cpu:identity('worker-lambda:'+str(cpu)) for cpu in cpus}
    refs = tuple(deepcopy(source_refs)) + (dict(artifact=python_wrapper,source_member='PipelineArtifact.run'),
        dict(artifact=probe_metadata['source'],
        probe_metadata_sha256=semantic_sha256(probe_metadata),
        original_artifact=probe_metadata['original_artifact'],
        original_source_sha256=provenance['original_source_sha256']),)
    events=[]; activities=[]; constraints=[]; occupancies=[]; capacities={}
    new_tasks=[]; precommit=[]; cut_events={}; timestamps={}; phase_rows=[]; constraint_keys=set()
    def cut_token(group, owner, name): return group,owner,name
    def timestamp(token):
        if trace is None or token is None: return None
        group,owner,name=token
        row=(raw_caller if group=='caller' else raw_lanes[owner] if group=='lane' else raw_calls[owner])[name]
        _need(row['valid'] is True and type(row['ns']) is int and row['ns'] >= 0,
            'every attached normal-path clock cut must be valid integer nanoseconds')
        return row['ns']
    def edge(before, after, label, *, kind=CK.CAUSAL, equal=False):
        key=semantic_sha256(dict(before=before,after=after,label=label,kind=kind.value,equal=equal))
        if key in constraint_keys:return
        constraint_keys.add(key)
        constraints.append(Constraint(identity('edge:'+key),before,after,kind,
            max_delay_ticks=0 if equal else None, source_refs=refs))
    def event(label, kind, scope, task=None, *, cut=None, material=None):
        name=identity(label); value=timestamp(cut)
        evidence=refs + (() if cut is None else (dict(probe_cut=list(cut),
            clock='CLOCK_MONOTONIC', instrumentation_subtracted=False),))
        events.append(Event(name,kind,scope,task=task,materialization=material,
                            timestamp_ticks=value,source_refs=evidence));timestamps[name]=value
        if cut is not None:
            if cut in cut_events:edge(cut_events[cut],name,'same-original-cut',equal=True)
            else:cut_events[cut]=name
        return name
    def caller(name): return cut_token('caller',None,name)
    def lane(cpu,name): return cut_token('lane',cpu,name)
    def call_cut(call,name): return cut_token('call',call,name)
    root_entry=event('root-entry',EK.SCOPE_ENTRY,root);root_exit=event('root-exit',EK.SCOPE_EXIT,root)
    native_entry=event('native-entry',EK.SCOPE_ENTRY,native,cut=caller('OUTER_ENTRY'))
    native_exit=event('native-exit',EK.SCOPE_EXIT,native,cut=caller('ORIGINAL_ABI_RETURNED'))
    scopes=[Scope(root,SK.ROOT,root_entry,root_exit,source_refs=refs),
            Scope(native,SK.COMPOSITE,native_entry,native_exit,parent=root,source_refs=refs)]
    worker_bounds={}
    for cpu in cpus:
        scope=worker_scopes[cpu]
        begin=event('worker-entry:'+str(cpu),EK.SCOPE_ENTRY,scope,cut=lane(cpu,'LAMBDA_ENTRY'))
        end=event('worker-exit:'+str(cpu),EK.SCOPE_EXIT,scope,cut=lane(cpu,'LAMBDA_EXIT'))
        scopes.append(Scope(scope,SK.COMPOSITE,begin,end,parent=native,source_refs=refs));worker_bounds[cpu]=(begin,end)

    def activity(task, scope, label, *, begin=None,end=None,start_cut=None,end_cut=None,
                 submission_cut=None,ready=(), role='SERVICE',unknown_cpu=True):
        start=event(label+':start',EK.START,scope,task,cut=start_cut)
        complete=event(label+':complete',EK.COMPLETION,scope,task,cut=end_cut)
        if begin is not None:edge(begin,start,label+':start-bound',equal=True)
        if end is not None:edge(complete,end,label+':end-bound',equal=True)
        submission=None if submission_cut is None else event(label+':submission-return',EK.SUBMISSION_RETURN,scope,task,cut=submission_cut)
        observed=timestamps[start] is not None and timestamps[complete] is not None
        duration=None if not observed else timestamps[complete]-timestamps[start]
        _need(duration is None or duration>=0,'source activity has reversed observed boundaries')
        aid=identity('activity:'+label)
        activities.append(Activity(aid,task,scope,start,complete,submission_return=submission,ready_events=tuple(ready),
            duration_basis=DB.OBSERVED if observed else DB.UNKNOWN,
            duration_lower_ticks=duration,duration_upper_ticks=duration,
            source_refs=refs+(dict(phase_role=role,observed_interval_is_not_intrinsic_service_cost=True),)))
        if unknown_cpu:
            resource=identity('unknown-physical-service:'+label);capacities[resource]=None
            occupancies.append(Occupancy(identity('occupation:'+label),OK.EXECUTION_DEMAND,resource,start,complete,
                scope,units=None,activity=aid,source_refs=refs))
        phase_rows.append(dict(activity=aid,task=task,role=role,start=start,completion=complete,
            submission_return=submission,additive_cost_assigned=False))
        return start,complete,aid,submission
    def phase(name, *, start_cut=None,end_cut=None,scope=native,actor='CALLER_UNBOUND',
              post=False,blocks='UNKNOWN',role='SERVICE',unknown_cpu=True,submission_cut=None,effects=None):
        task=identity('task:'+name)
        _need(task not in known_tasks,'host phase collides with an original task')
        new_tasks.append(Task(task,K.IMPLEMENTATION_PHASE,capability='CPU_IMPLEMENTATION_PHASE',source_refs=refs,
            attrs=dict(phase_kind=name,actor=actor,failure_domain='POSTCOMMIT'if post else'PRECOMMIT',
                may_fail=False if post else'UNKNOWN',blocks=blocks,
                phase_effects=[name]if effects is None else list(effects),source_normal_completion_only=True)))
        if not post:precommit.append(task)
        return task,activity(task,scope,name,start_cut=start_cut,end_cut=end_cut,
            role=role,unknown_cpu=unknown_cpu,submission_cut=submission_cut)
    _,python_prefix=phase('PYTHON_ENTRY_GUARDS_PACKING',scope=root,
        effects=('CHECK_INPUT_COUNT_COPY_PROTOCOL_SHAPE_LAYOUT_METADATA',
                 'PACK_INPUT_ABI_DESCRIPTORS_AND_CHECK_SOURCE_BOUND_GUARDS'),
        role='UNOBSERVED_ORIGINAL_PYTHON_PREFIX_NOT_PROBE_PYTHON_WRAPPER')
    _,python_output_pack=phase('PYTHON_OUTPUT_DESCRIPTOR_AND_FFI_ENTRY',scope=root,
        effects=('PACK_FRESH_OUTPUT_DESCRIPTOR_AND_ARGUMENT_ARRAY','PREPARE_NATIVE_STATUS_AND_FFI_CALL'),
        role='UNOBSERVED_ORIGINAL_PYTHON_FFI_ENTRY')
    edge(root_entry,python_prefix[0],'original-python-entry',equal=True)
    prefix=[]
    for name,first,last in (
        ('CALLER_FLOATING_ENVIRONMENT','OUTER_ENTRY','CALLER_ENV_CONSTRUCTED'),
        ('CALLER_NATIVE_ADMISSION','CALLER_ENV_CONSTRUCTED','RUN_READY_ENTRY'),
        ('NATIVE_PREFLIGHT','RUN_READY_ENTRY','PREFLIGHT_END'),
        ('POOL_ALLOCATE_AGGREGATE','POOL_ALLOC_BEGIN','POOL_ALLOC_END'),
        ('CONTROL_INITIALIZATION','POOL_ALLOC_END','RESERVE_BEGIN'),
        ('WORKER_VECTOR_RESERVE','RESERVE_BEGIN','RESERVE_END')):
        prefix.append(phase(name,start_cut=caller(first),end_cut=caller(last)))
    pool_allocate=prefix[3][1]
    # CONTROL_READY is an internal observed anchor of control initialization.
    event('control-ready-cut',EK.COMPLETION,native,prefix[4][0],cut=caller('CONTROL_READY'))
    launches={}; joins={}; exits={}
    for cpu in cpus:
        task,part=phase('WORKER_LAUNCH:'+str(cpu),start_cut=lane(cpu,'EMPLACE_BEGIN'),
            end_cut=lane(cpu,'LAMBDA_ENTRY'),submission_cut=lane(cpu,'EMPLACE_END'),
            role='ASYNC_LAUNCH_LATENCY_NOT_ADDITIVE_CALLER_SERVICE',unknown_cpu=False)
        launches[cpu]=part
        resource=identity('unknown-create-service:'+str(cpu));capacities[resource]=None
        occupancies.append(Occupancy(identity('create-caller-occupation:'+str(cpu)),OK.EXECUTION_DEMAND,
            resource,part[0],part[3],native,units=None,activity=part[2],source_refs=refs))
        edge(part[1],worker_bounds[cpu][0],'launch-worker-entry',equal=True)
        task,part=phase('WORKER_LAMBDA_EXIT:'+str(cpu),scope=worker_scopes[cpu],actor='WORKER',
            end_cut=lane(cpu,'LAMBDA_EXIT'),role='EXIT_AFTER_ALL_PUBLICATIONS_NO_WAIT_COST')
        exits[cpu]=part
        task,part=phase('CALLER_JOIN:'+str(cpu),start_cut=lane(cpu,'JOIN_BEGIN'),end_cut=lane(cpu,'JOIN_END'),
            blocks=True,role='WAIT_INCLUSIVE_JOIN_LATENCY_NOT_FIXED_SERVICE',unknown_cpu=False)
        joins[cpu]=part
    edge(prefix[-1][1][1],launches[cpus[0]][0],'reserve-before-first-create')
    for before,after in zip(cpus,cpus[1:]):edge(launches[before][3],launches[after][0],'caller-create-order')
    for before,after in zip(cpus,cpus[1:]):edge(joins[before][1],joins[after][0],'caller-join-order')
    launch_loop=event('launch-loop-end-cut',EK.COMPLETION,native,
        identity('task:WORKER_LAUNCH:'+str(cpus[-1])),cut=caller('LAUNCH_LOOP_END'))
    joins_end=event('joins-end-cut',EK.COMPLETION,native,
        identity('task:CALLER_JOIN:'+str(cpus[-1])),cut=caller('JOINS_END'))
    edge(launches[cpus[-1]][3],launch_loop,'all-constructors-returned')
    edge(launch_loop,joins[cpus[0]][0],'join-after-launch-loop')
    edge(joins[cpus[-1]][1],joins_end,'all-joins-returned')
    for cpu in cpus:edge(exits[cpu][1],joins[cpu][1],'thread-exit-before-join-return')

    allocs={}; releases={}; physical_groups=[]
    for task in schedule.tasks:
        if task.kind is K.ALLOCATE:
            final=all(next(s for s in schedule.storage_slots if s.id==name).released_by is None for name in task.allocates)
            part=activity(task.id,root if final else native,'original:'+task.id,
                role='UNOBSERVED_PYTHON_FINAL_ALLOCATION'if final else'SYMBOLIC_SLOT_MEMBER_NOT_SEPARATELY_CHARGEABLE',unknown_cpu=False)
            allocs[task.id]=part
            if final:
                edge(python_prefix[1],part[0],'python-guards-before-final-allocation')
                edge(part[1],python_output_pack[0],'output-descriptor-after-final-allocation')
                edge(python_output_pack[1],native_entry,'python-ffi-before-native')
            else:
                edge(pool_allocate[0],part[0],'aggregate-allocate-begin')
                edge(part[1],pool_allocate[1],'aggregate-allocate-end')
    ready={}
    for material in schedule.materializations:
        scope=root if material.external_ready else worker_scopes[certificate['call_cpu_ids'][known_tasks[material.ready_task].attrs['call_id']]]
        ready[material.id]=event('ready:'+material.id,EK.READY,scope,material.ready_task,material=material.id)
        if material.external_ready:edge(root_entry,ready[material.id],'formal-entry-readiness',equal=True)
    parts={}; mutex_sections=[]
    for call,task in by_call.items():
        cpu=certificate['call_cpu_ids'][call];scope=worker_scopes[cpu]
        def part(role,first,last,**kwargs):
            return activity(task.id,scope,'call:'+call+':'+role,start_cut=call_cut(call,first),
                end_cut=None if last is None else call_cut(call,last),role=role,**kwargs)
        claim=part('CLAIM','CLAIM_BEGIN','CLAIM_END')
        pre=part('PRE_ABI','CLAIM_END','ABI_BEGIN')
        abi=part('ABI_SERVICE','ABI_BEGIN','ABI_END',unknown_cpu=False)
        post=part('POST_ABI','ABI_END','EXECUTE_END')
        publication=part('PUBLISH','PUBLISH_LOCK_ACQUIRED',None,
            ready=tuple(ready[m.id]for m in schedule.materializations if m.ready_task==task.id))
        execute_begin=event('execute-begin:'+call,EK.START,scope,task.id,cut=call_cut(call,'EXECUTE_BEGIN'))
        lock_begin=event('publish-lock-begin:'+call,EK.START,scope,task.id,cut=call_cut(call,'PUBLISH_LOCK_BEGIN'))
        pre_unlock=event('pre-unlock:'+call,EK.COMPLETION,scope,task.id,cut=call_cut(call,'PUBLISH_PRE_UNLOCK'))
        post_unlock=event('post-unlock:'+call,EK.COMPLETION,scope,task.id,cut=call_cut(call,'PUBLISH_POST_UNLOCK'))
        claim_release=event('claim-mutex-release:'+call,EK.COMPLETION,scope,task.id)
        edge(claim[1],claim_release,'claim-release-lower');edge(claim_release,execute_begin,'claim-release-upper')
        edge(execute_begin,pre[1],'execute-before-abi');edge(post[1],lock_begin,'execute-before-publication-wait')
        edge(lock_begin,publication[0],'mutex-wait-not-fixed-service')
        edge(publication[0],pre_unlock,'publication-body');edge(pre_unlock,publication[1],'release-bracket-lower')
        edge(publication[1],post_unlock,'release-bracket-upper')
        for a,b in ((claim[1],pre[0]),(pre[1],abi[0]),(abi[1],post[0])):edge(a,b,'call-phase-boundary',equal=True)
        for material in schedule.materializations:
            if material.ready_task==task.id:edge(publication[1],ready[material.id],'published-ready',equal=True)
        for material in task.reads:edge(ready[material],claim[0],'claim-reads-ready',kind=CK.READINESS)
        parts[call]=dict(claim=claim,abi=abi,publish=publication,post_unlock=post_unlock,
            claim_release=claim_release,pre_unlock=pre_unlock)
        mutex_sections.extend(((claim[0],claim_release,claim[2]),(publication[0],publication[1],publication[2])))
        exact=[]
        for ref in task.resource_refs:
            if ref.get('kind')=='EXPLICIT_HARDWARE_THREAD_SET':
                exact.extend(r['resource']['wire']for c,r in zip(ref.get('cpu_ids',()),ref.get('thread_refs',()))if c==cpu)
        _need(len(set(exact))<=1,'contradictory original HardwareThread bindings')
        resource=exact[0]if exact else identity('unresolved-abi-cpu:'+str(cpu))
        capacities[resource]=1 if exact else None
        occupancies.append(Occupancy(identity('abi-cpu:'+call),OK.EXECUTION_DEMAND,resource,abi[0],abi[1],scope,
            units=1 if exact else None,activity=abi[2],source_refs=refs+task.resource_refs))
        lane_resource=identity('logical-worker:'+str(cpu));capacities[lane_resource]=1
        occupancies.append(Occupancy(identity('logical-worker-hold:'+call),OK.EXECUTION_DEMAND,lane_resource,
            claim[0],post_unlock,scope,units=1,activity=claim[2],source_refs=refs+(dict(logical_worker_not_physical_cpu=True),)))
    for call in calls:
        for pred in certificate['call_predecessors'][call]:
            _need(pred in original_preds[call] or pred in certificate.get('resource_order_dependencies',{}).get(call,()),
                'implemented predecessor has neither source-dependency nor selected resource-order authority')
            edge(parts[pred]['publish'][1],parts[call]['claim'][0],'implemented-predecessor',
                kind=CK.CAUSAL if pred in original_preds[call] else CK.RESOURCE_ORDER)
    for cpu in cpus:
        for before,after in zip(lanes[str(cpu)],lanes[str(cpu)][1:]):
            edge(parts[before]['post_unlock'],parts[after]['claim'][0],'same-worker-loop-order',kind=CK.RESOURCE_ORDER)
        for call in calls:edge(parts[call]['publish'][1],exits[cpu][0],'all-success-before-worker-exit')
    mutex=identity('logical-control-mutex');capacities[mutex]=1
    for i,(start,end,aid)in enumerate(mutex_sections):
        activity_scope=next(a.scope for a in activities if a.id==aid)
        occupancies.append(Occupancy(identity('mutex:'+str(i)),OK.EXECUTION_DEMAND,mutex,start,end,activity_scope,
            units=1,activity=aid,source_refs=refs+(dict(logical_mutex_not_cpu_service=True),)))
    if trace is not None:
        ordered=sorted(mutex_sections,key=lambda r:timestamps[r[0]])
        for a,b in zip(ordered,ordered[1:]):
            if timestamps[a[0]]<timestamps[b[0]]:edge(a[1],b[0],'observed-mutex-critical-section-order',kind=CK.RESOURCE_ORDER)

    fence_part=activity(fence.id,native,'original:'+fence.id,start_cut=caller('FINAL_CHECK_BEGIN'),
        end_cut=caller('RETURN_SUCCESS_BEGIN'),role='ORIGINAL_SUCCESS_CHECKS')
    _,cleanup=phase('CONTROL_METADATA_CLEANUP',start_cut=caller('RETURN_SUCCESS_BEGIN'),
        end_cut=caller('POOL_DESTRUCT_BEGIN'),post=True)
    _,pool_release=phase('POOL_RELEASE_AGGREGATE',start_cut=caller('POOL_DESTRUCT_BEGIN'),
        end_cut=caller('POOL_DESTRUCT_END'),post=True)
    for task in schedule.tasks:
        if task.kind is K.RELEASE:
            part=activity(task.id,native,'original:'+task.id,
                role='SYMBOLIC_SLOT_MEMBER_NOT_SEPARATELY_CHARGEABLE',unknown_cpu=False)
            releases[task.id]=part
            edge(pool_release[0],part[0],'aggregate-release-begin');edge(part[1],pool_release[1],'aggregate-release-end')
    returning=activity(exported.id,native,'EXPORT_RETURN_CONTROL',start_cut=caller('POOL_DESTRUCT_END'),
        end_cut=caller('CALLER_RESTORE_BEGIN'),role='RETURN_CONTROL_INCLUDING_RUN_READY_EPILOGUE')
    event('run-ready-returned-cut',EK.COMPLETION,native,exported.id,cut=caller('RUN_READY_RETURNED'))
    restored=activity(exported.id,native,'EXPORT_CALLER_RESTORE',start_cut=caller('CALLER_RESTORE_BEGIN'),
        end_cut=caller('CALLER_RESTORE_END'),role='FALLIBLE_RESTORE_CONDITIONAL_NORMAL_SUCCESS')
    handoff=activity(exported.id,native,'EXPORT_NATIVE_ABI_RETURN',start_cut=caller('CALLER_RESTORE_END'),
        end_cut=caller('ORIGINAL_ABI_RETURNED'),role='NORMAL_ABI_RETURN_NOT_WHOLE_PYTHON_RETURN')
    outer_cut=event('probe-outer-return-cut',EK.START,root,exported.id,cut=caller('OUTER_RETURN'))
    python_return=activity(exported.id,root,'UNOBSERVED_PYTHON_RETURN',role='UNKNOWN_PYTHON_FFI_AND_RESULT_HANDOFF')
    edge(native_exit,outer_cut,'probe-wrapper-return');edge(outer_cut,python_return[0],'python-handoff-after-probe')
    # Caller-only order never serializes worker computation to later launches.
    for first,last in zip(CALLER_SLOTS,CALLER_SLOTS[1:]):
        edge(cut_events[caller(first)],cut_events[caller(last)],'original-caller-program-order')
    edge(joins_end,fence_part[0],'joined-before-final-check')
    for task in schedule.tasks:
        for slot in task.allocates:
            original_slot=next(s for s in schedule.storage_slots if s.id==slot)
            scope=root if original_slot.released_by is None else native
            resource=identity('physical-slot:'+slot);capacities[resource]=original_slot.capacity_bytes
            end=root_exit if original_slot.released_by is None else releases[original_slot.released_by][1]
            occupancies.append(Occupancy(identity('physical-hold:'+slot),OK.PHYSICAL_SLOT_BYTES,resource,
                allocs[task.id][0],end,scope,units=original_slot.capacity_bytes,storage_slot=slot,
                continues_after_scope=original_slot.released_by is None,source_refs=refs+original_slot.resource_refs))
    for kind,part,members in (('POOL_ALLOCATE',pool_allocate,list(t.id for t in schedule.tasks if t.kind is K.ALLOCATE and t.id not in
            [s.allocated_by for s in schedule.storage_slots if s.released_by is None])),
            ('POOL_RELEASE',pool_release,list(releases))):
        physical_groups.append(dict(kind=kind,aggregate_activity=part[2],symbolic_original_tasks=members,
            additive_charge_count=1,per_slot_cost_or_exact_timestamps_inferred=False))
    refined=[]
    for task in schedule.tasks:
        if task.id==fence.id:
            task=replace(task,depends_on=tuple(dict.fromkeys((*task.depends_on,*precommit))),
                validates=tuple(dict.fromkeys((*task.validates,*precommit))))
        refined.append(task)
    proof=deepcopy(schedule.proof)
    proof['pipeline_host_envelope']=dict(schema=SCHEMA,schema_version=1,
        original_schedule_sha256=semantic_sha256(schedule_document(schedule)),
        certificate_sha256=semantic_sha256(certificate),probe_metadata_sha256=semantic_sha256(probe_metadata),
        trace_sha256=None if trace is None else semantic_sha256(trace),native_scope=native,
        worker_scopes={str(cpu):scope for cpu,scope in worker_scopes.items()},
        source_refs=encode(refs),phases=phase_rows,physical_cost_groups=physical_groups,
        python_wrapper_source=python_wrapper,
        original_data_values_materializations_slots_and_task_ids_preserved=True,
        original_fence_extended_only_for_added_precommit_phases=True,
        publication_release_timestamps_observed=False,publication_release_is_bracketed=True,
        cross_lane_post_unlock_to_successor_claim_added=False,
        creation_completion_is_lambda_entry_submission_return_is_emplace_end=True,
        creation_completion_and_submission_return_unordered=True,
        waits_are_causal_gaps_or_latency_not_fixed_cpu_costs=True,
        worker_lane_is_logical_actor_not_observed_cpu=True,
        native_scope_is_instrumented_abi_call_bracket_not_whole_python_call=True,
        python_guards_output_allocation_and_ffi_costs_known=False,
        slot_destructor_order_inferred=False,slot_bytes_counted_once=True,
        observed_timing_is_intrinsic_or_final_performance=False,
        region_executed=trace is not None and trace['target_region_executed'],full_target_program_executed=False,
        source_admission_required_upstream=True,source_bytes_replayed_here=False,
        solver_invoked=False,execution_authorized=False)
    model=Model(Mode.OBSERVED if trace is not None else Mode.SEMANTIC,tuple(events),tuple(activities),
        tuple(occupancies),tuple(constraints),tuple(scopes),resource_capacities=capacities,
        tick_numerator=1 if trace is not None else None,tick_denominator=10**9 if trace is not None else None,
        clock_domain='CLOCK_MONOTONIC:'+probe_metadata['build_id']if trace is not None else None,source_refs=refs)
    result=replace(schedule,tasks=tuple((*refined,*new_tasks)),proof=proof,executable=False,temporal=model)
    validate_versioned_schedule(result)
    return result


__all__=['SCHEMA','derive_pipeline_host_envelope']
