"""Sufficient conditional initialization proof for private capacity reuse.

Inputs must come from the full parent/session source readers. This pure
projection rechecks their identity links and the original typed one-step
initialization, without reading files, loading native code or using N as a
dynamic session lifetime. It does not turn provider assumptions into purity.
"""
from copy import deepcopy

from scar.ir.versioned_schedule import (schedule_from_dict, validate_versioned_schedule,
                                        VersionedTaskKind as K)
from scar.ir.record_codec import encode
from scar.optimization.resource_solver import semantic_sha256

SCHEMA = 'scar.session-capacity-proof'


class _Frontier(ValueError):
    def __init__(self, code, detail):
        self.code, self.detail = code, detail
        super().__init__(detail)


def _need(condition, code, detail):
    if not condition:
        raise _Frontier(code, detail)


def _same(a, b):
    return semantic_sha256(a) == semantic_sha256(b)


def _identity(document, schema, prefix=None):
    _need(type(document) is dict and (document.get('schema'),document.get('schema_version')) == (schema,1),
          'UNSUPPORTED_PARENT_SCHEMA', 'the original versioned '+schema+' material is required')
    if prefix is not None:
        raw=dict(document); identity=raw.pop('id',None)
        _need(identity==prefix+semantic_sha256(raw), 'PARENT_SEMANTIC_IDENTITY_CHANGED', schema)


def _project(session, step, workspace, parent, native):
    _identity(session,'scar.workspace-session-plan')
    _identity(step,'scar.stateful-stage-plan')
    _identity(workspace,'scar.stateful-workspace-reuse-plan','stateful-workspace-plan:')
    _identity(parent,'scar.native-stateful-workspace-repeat')
    _identity(native,'scar.native-workspace-session')
    step_sha=semantic_sha256(step); workspace_sha=semantic_sha256(workspace)
    _need(workspace['step_plan_sha256']==step_sha
          and workspace['step_region_sha256']==step['region_sha256']
          and parent['base_ids']['plan_sha256']==step_sha
          and parent['base_ids']['region_sha256']==step['region_sha256']
          and parent['workspace_plan_sha256']==workspace_sha,
          'STEP_WORKSPACE_SOURCE_IDENTITY_DIFFERS','step/source/workspace/native identities must agree')
    blueprint=session['native_blueprint']; mapping=native['source_mapping']
    _need(blueprint['workspace_plan_sha256']==workspace_sha
          and blueprint['execution_plan_sha256']==parent['execution_plan_sha256']
          and _same(blueprint['base_ids'],parent['base_ids'])
          and _same(blueprint['source_mapping'],parent['source_mapping'])
          and session['parent_native_manifest']==blueprint['manifest']
          and blueprint['cpp_source']['sha256']==parent['sha256']['source'],
          'NATIVE_BLUEPRINT_DIFFERS','the original parent source and semantic execution identities are required')
    execution=mapping['execution_plan']
    _need(mapping['execution_plan_sha256']==native['execution_plan_sha256']==semantic_sha256(execution)
          and execution['session_plan_sha256']==semantic_sha256(session)
          and execution['parent_execution_plan_sha256']==parent['execution_plan_sha256']
          and execution['residency_plan_sha256']==session['residency_plan_sha256']
          and mapping['parent_execution_plan_sha256']==parent['execution_plan_sha256']
          and mapping['original_source_sha256']==parent['sha256']['source']
          and native['source']['sha256']==mapping['source_sha256'],
          'SESSION_SOURCE_CHAIN_DIFFERS','the admitted dynamic session must retain the exact parent source chain')
    required_flags=('source_inverse_to_residency_byte_equal','original_run_table_fixed_repeat_and_residency_getter_unchanged',
        'helper_math_checks_and_commits_unchanged','helper_called_once_per_step','step_scalars_read_fresh',
        'create_has_no_math_or_payload_reads','create_original_preflight_before_payload_allocation',
        'original_allocation_block_verbatim')
    _need(all(mapping.get(k) is True for k in required_flags)
          and mapping.get('local_to_session_transfer')=='UNIQUE_PTR_METADATA_MOVE_NO_PAYLOAD_COPY'
          and mapping.get('destroy_slot_order')=='CPP_STD_ARRAY_REVERSE_ELEMENT_DESTRUCTION'
          and mapping.get('dynamic_call_count_fixed_N_authority') is False
          and mapping.get('source_pins')
          and mapping['source_pins'].get('workspace_session.py'),
          'UNSUPPORTED_SESSION_SOURCE_CONTRACT','full upstream renderer replay of the raw double[] session is required')
    residency=mapping['residency_mapping']
    _need(residency.get('per_step_finite_scalar_affinity_fp_math_provider_restore_commit_retained_verbatim') is True
          and residency.get('source_inverse_byte_equal') is True,
          'STEP_BODY_NOT_RETAINED','capacity reuse must retain all original per-step work and guards')

    graph=schedule_from_dict(workspace['shared_step']); validate_versioned_schedule(graph)
    materials={m.id:m for m in graph.materializations}; slots={s.id:s for s in graph.storage_slots}
    private={key:s for key,s in slots.items()if s.ownership=='SCHEDULE'}
    order=[s['id']for s in step['workspace_slots']]
    _need(len(order)==len(set(order)) and set(order)==set(private)
          and workspace['allocation_order']==order and workspace['release_order']==list(reversed(order)),
          'PRIVATE_SLOT_DOMAIN_DIFFERS','the same original private slots and reverse release order must be retained')
    capacities=[s['capacity_bytes']for s in step['workspace_slots']]
    _need(all(type(c)is int and c>=0 and c%8==0 for c in capacities)
          and all(private[key].device=='cpu' and private[key].capacity_bytes==c for key,c in zip(order,capacities))
          and capacities==session['geometry']['slot_capacities_bytes']
          and sum(capacities)==step['workspace_payload_bytes']==parent['memory']['workspace_payload_bytes']
          and _same(workspace['input_shapes'],step['input_shapes']),
          'PRIVATE_CAPACITY_OR_SHAPE_DIFFERS','raw double payload capacities must match the original stage plan')
    _need(session['geometry']['original_array_count']==len(step['input_shapes'])
          and _same(session['geometry']['memory'],parent['memory']),
          'ARRAY_OR_MEMORY_GEOMETRY_DIFFERS','the original array descriptors and payload memory geometry are required')
    _need(not any(t.kind is K.EXPORT for t in graph.tasks),
          'PRIVATE_EXPORT_NOT_SUPPORTED','private addresses/values cannot escape via an export task')
    positions={t.id:i for i,t in enumerate(graph.tasks)}; definitions={key:[]for key in private}
    source_stages={semantic_sha256(dict(artifact_role='stage_plan',pointer='#/stages/'+str(i))):stage
                   for i,stage in enumerate(step['stages'])}
    def cover(slot,start,end):
        point=start; witnesses=[]
        for low,high,task,material in sorted(definitions[slot]):
            if high<=point or low>point:continue
            point=max(point,high);witnesses.append(dict(task=task,materialization=material))
            if point>=end:break
        return point>=end,witnesses
    reads=[]; first={}; provider_tasks=[]; snapshots=[]
    for task in graph.tasks:
        if task.kind in (K.ALLOCATE,K.RELEASE):continue
        if task.attrs.get('stage_kind')=='PROVIDER':provider_tasks.append(task)
        for name in task.reads:
            material=materials[name]; region=material.region
            if region.slot not in private:continue
            ready=material.ready_task
            _need(ready in positions and positions[ready]<positions[task.id],
                'PRIVATE_READ_HAS_NO_CURRENT_PRODUCER','private material '+name+' needs an earlier step-local producer')
            start,end=region.offset_bytes,region.offset_bytes+region.nbytes
            covered,witnesses=cover(region.slot,start,end)
            _need(covered,'PRIVATE_READ_REQUIRES_OLD_OR_UNDEFINED_BYTES',task.id+' reads '+name)
            record=dict(task=task.id,materialization=name,slot=region.slot,
                extent=dict(offset_bytes=start,nbytes=region.nbytes),producer=ready,
                current_iteration_definitions=witnesses,first_required_slot_read=region.slot not in first,
                material_version=material.version,version_scope='ORIGINAL_STEP_WITHIN_ITERATION_i')
            reads.append(record);first.setdefault(region.slot,record)
        for name in task.writes:
            material=materials[name];region=material.region
            if region.slot not in private:
                _need(task.kind is K.COMMIT,'EXTERNAL_PAYLOAD_WRITE_OUTSIDE_COMMIT',task.id)
                continue
            start,end=region.offset_bytes,region.offset_bytes+region.nbytes
            if task.kind is K.STATE_WRITE:
                _need(cover(region.slot,start,end)[0],'PARTIAL_WRITE_NEEDS_UNDEFINED_ROOT',task.id)
            else:
                _need(task.kind in (K.SNAPSHOT,K.COMPUTE),
                    'UNSUPPORTED_PRIVATE_DEFINITION','private write kind '+task.kind.value)
                definitions[region.slot].append((start,end,task.id,name))
        if task.kind is K.SNAPSHOT:
            _need(len(task.reads)==len(task.writes)==1,'SNAPSHOT_ABI_DIFFERS',task.id)
            src,dst=(materials[n]for n in (task.reads[0],task.writes[0]))
            shape=task.attrs.get('shape')
            target=source_stages[semantic_sha256(task.attrs['source_stage_ref'])]['target']
            _need(type(shape)in(tuple,list) and len(shape)==2 and all(type(v)is int and v>0 for v in shape)
                  and src.region.slot in slots and slots[src.region.slot].ownership=='EXTERNAL'
                  and dst.region.slot in private and src.region.offset_bytes==dst.region.offset_bytes==0
                  and src.region.nbytes==dst.region.nbytes==shape[0]*shape[1]*8
                  and step['shadows'].get(target) in step['assignment']
                  and step['assignment'][step['shadows'][target]]['slot_id']==dst.region.slot,
                  'FULL_ROOT_SNAPSHOT_NOT_PROVEN',task.id)
            snapshots.append(dict(task=task.id,input_id=target,shape=list(shape),
                source_material=src.id,private_material=dst.id,slot=dst.region.slot,
                bytes=dst.region.nbytes,scope='EVERY_DYNAMIC_STEP_WHEN_DIRTY_OR_BINDINGS_UNSET'))
    initialized=workspace['initialization_proof']
    _need(_same(reads,initialized['reads']) and _same(list(first.values()),initialized['first_required_slot_reads']),
          'ORIGINAL_INITIALIZATION_WITNESSES_DIFFER','independently replayed reads must match the original source-qualified initialization')
    _need({r['input_id']for r in snapshots}==set(step['shadows'])
          and len(snapshots)==len(step['shadows']),
          'FULL_SNAPSHOT_ROOT_DOMAIN_INCOMPLETE','every shadow root needs exactly one current-step full snapshot')
    captured=residency['snapshots']
    for snapshot in snapshots:
        matches=[r for r in captured if r['snapshot_stage_id']==snapshot['task']]
        _need(len(matches)==1 and matches[0]['first_iteration_full_copy_verbatim'] is True
              and matches[0]['input_id']==snapshot['input_id'] and matches[0]['slot_id']==snapshot['slot']
              and _same(matches[0]['shape'],snapshot['shape']),
              'FULL_SNAPSHOT_SOURCE_BRANCH_NOT_RETAINED','the admitted first_iteration branch must be the original full body')

    provider_q=deepcopy(session['provider_boundary_Q']); ids={t.attrs['provider_id']for t in provider_tasks}
    _need(len(provider_q)==len(ids) and {q['provider_id']for q in provider_q}==ids,
          'PROVIDER_BORROW_CONTRACT_MISSING','every borrowed provider requires its original synchronous boundary assumption')
    for q in provider_q:
        _need(q.get('authority')=='EXTERNAL_ASSUMPTION' and q.get('source_proven') is False
              and q.get('runtime_checkable') is False and q.get('parent_condition')=='SYNCHRONOUS_CPU_EXECUTION'
              and q['parent_condition']in step['external_assumptions']
              and q.get('condition')=='ALL_BORROWED_PORT_ACCESSES_COMPLETE_BEFORE_PROVIDER_RETURN_AND_NO_DELAYED_OR_ASYNC_PAYLOAD_ACCESS_BEFORE_NEXT_ORIGINAL_PROVIDER_CALL',
              'PROVIDER_SYNCHRONOUS_Q_NOT_PRESERVED',q['provider_id'])
        actual=[t for t in provider_tasks if t.attrs['provider_id']==q['provider_id']]
        _need({r['task']for r in q['borrowed_ports']}=={t.id for t in actual},
              'PROVIDER_PORT_DOMAIN_DIFFERS',q['provider_id'])
        for task in actual:
            borrow=next(r for r in q['borrowed_ports']if r['task']==task.id)
            _need(_same(borrow['reads'],task.reads) and _same(borrow['writes'],task.writes)
                  and set(borrow['regions'])==set((*task.reads,*task.writes))
                  and _same(borrow['regions'],{n:encode(materials[n].region)for n in (*task.reads,*task.writes)})
                  and all(materials[n].region.slot in private for n in (*task.reads,*task.writes)),
                  'PROVIDER_PRIVATE_BORROW_PROFILE_NOT_CLOSED',task.id)
    return dict(slots=[dict(id=key,capacity_bytes=c,payload_type='double[]',python_references_in_payload=False)
                       for key,c in zip(order,capacities)],
        current_step_initialization=dict(rule='CURRENT_STEP_WRITE_COVERAGE_FROM_EMPTY_DEFINITION_SET',
            private_read_count=len(reads),reads=reads,first_required_slot_reads=list(first.values()),
            original_graph_sha256=semantic_sha256(workspace['shared_step']),
            original_initialization_proof_ref=dict(artifact_role='workspace_plan',pointer='#/initialization_proof'),
            reset_knowledge_before_each_step=True,old_bytes_may_remain_but_are_not_inputs=True,
            content_versions_are_step_local_not_runtime_logical_versions=True),
        full_snapshot=dict(snapshots=snapshots,all_shadow_roots_covered=True,
            original_full_snapshot_source_branch_retained=True,later_roi_residency_branch_must_never_be_selected=True),
        native_source=dict(parent_manifest=deepcopy(blueprint['manifest']),parent_cpp=deepcopy(blueprint['cpp_source']),
            session_source=deepcopy(native['source']),parent_execution_plan_sha256=parent['execution_plan_sha256'],
            session_execution_plan_sha256=native['execution_plan_sha256'],source_mapping_sha256=semantic_sha256(mapping),
            producer_pins=deepcopy(mapping['source_pins']),
            source_admission='FULL_PARENT_AND_SESSION_RENDERER_BYTE_REPLAY_REQUIRED_UPSTREAM',
            source_bytes_replayed_here=False,private_slot_layout='STD_ARRAY_OF_UNIQUE_PTR_DOUBLE_ARRAYS',
            fixed_trip_count_is_not_dynamic_session_authority=True),provider_boundary_Q=provider_q,
        payload_effect_scope=dict(explicit_external_array_writes='ORIGINAL_COMMIT_TASKS_ONLY',
            borrowed_provider_ports='PRIVATE_ONLY_UNDER_ORIGINAL_DECLARED_PORT_AND_QUIESCENCE_Q',
            opaque_provider_io_or_global_effects_proven_absent=False,provider_purity_proven=False,
            original_guards_arithmetic_providers_affinity_fp_restoration_and_commits_unchanged=True))


def derive_capacity_proof(session_plan,step_plan,workspace_plan,parent_native_metadata,session_native_metadata,*,source_refs=()):
    inputs=dict(session_plan_sha256=semantic_sha256(session_plan),step_plan_sha256=semantic_sha256(step_plan),
        workspace_plan_sha256=semantic_sha256(workspace_plan),parent_native_sha256=semantic_sha256(parent_native_metadata),
        session_native_sha256=semantic_sha256(session_native_metadata))
    frontiers=[]
    try: details=_project(session_plan,step_plan,workspace_plan,parent_native_metadata,session_native_metadata)
    except _Frontier as error: details={};frontiers.append(dict(code=error.code,detail=error.detail))
    except (KeyError,TypeError,ValueError)as error:
        details={};frontiers.append(dict(code='ORIGINAL_CAPACITY_MATERIAL_NOT_CLOSED',detail=str(error)))
    result=dict(schema=SCHEMA,schema_version=1,status='FRONTIER'if frontiers else'PROVEN_CONDITIONAL',
        inputs=inputs,**details,frontiers=frontiers,source_refs=deepcopy(list(source_refs)),
        required_capacity_protocol=dict(invalidate_before_every_step=True,invalidate_after_every_step=True,
            full_snapshot_each_step=True,no_cross_step_content_fact=True,no_fixed_trip_count_authority=True,
            have_bindings_false_before_and_after_step=True,never_compare_or_read_old_binding_descriptors=True,
            forget_all_per_step_array_and_data_owner_references_before_host_return=True,
            retain_immutable_provider_capsule_code_dtype_and_base_library_owners=True,
            per_step_binding_forget_does_not_unload_native_or_provider_implementation=True,
            raw_abi_descriptor_bytes_need_not_be_zeroed=True,
            existing_invalidate_only_does_not_release_runtime_python_owner_references=True,
            caller_may_run_original_host_effects_between_quiescent_steps=True,
            current_step_transaction_and_failure_fallback_unchanged=True,
            single_owner_idle_handle_and_exclusive_synchronous_inputs_required=True),
        source_admission_performed_here=False,caller_effect_coverage_proven=False,
        caller_replacement_authorized=False,allocation_failure_or_observer_equivalence_proven=False,
        cost_known=False,selected_for_use=False,default_backend='NO_OP',deployment_authorized=False)
    result['id']='session-capacity-proof:'+semantic_sha256(result)
    return result


def verify_capacity_proof(session_plan,step_plan,workspace_plan,parent_native_metadata,session_native_metadata,proof,*,source_refs=None):
    expected=derive_capacity_proof(session_plan,step_plan,workspace_plan,parent_native_metadata,session_native_metadata,
        source_refs=proof['source_refs']if source_refs is None else source_refs)
    if not _same(expected,proof):raise ValueError('capacity proof differs from original step/source/initialization replay')
    return proof


__all__=['SCHEMA','derive_capacity_proof','verify_capacity_proof']
