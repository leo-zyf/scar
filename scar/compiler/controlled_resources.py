"""Extract controllable native-region resources from cold-admitted artifacts.

These contracts describe a scoped synchronous invocation and its explicit
payload budget. They do not reserve physical CPUs or memory, close Python
entry RSS/effects, qualify timing, or authorize deployment.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference,read_artifact
from scar.optimization.resource_solver import semantic_sha256

SCHEMA='scar.controlled-region-resource-contract'
PIPELINE_FUSION='PIPELINE_FUSION'
STATEFUL_SPECIALIZATION='STATEFUL_SPECIALIZATION'


def _need(condition,message):
    if not condition:raise ValueError(message)


def _reference(value,schema):
    reference=deepcopy(value)if type(value)is dict else artifact_reference(Path(value).resolve(),schema)
    _need(reference.get('schema')==schema,'original admitted parent schema differs')
    read_artifact(reference)
    return reference


def _file(path,schema='scar.implementation-source'):
    reference=artifact_reference(Path(path).resolve(),schema)
    read_artifact(reference,json_document=False)
    return reference


def _fact(artifact,locator):return dict(artifact=artifact,locator=locator)


def _anchor(reference,text,meaning):
    raw=read_artifact(reference,json_document=False);needle=text.encode()
    _need(raw.count(needle)==1,'unique source anchor missing: '+text)
    return dict(artifact=reference,line=raw[:raw.index(needle)].count(b'\n')+1,meaning=meaning)


def _target(reference):
    from scar.optimization.hardware import _target_materials
    from scar.ir.program import target_hardware as H
    from scar.ir.record_codec import encode
    graph,snapshot,resources,_=_target_materials(Path(reference['uri']).parent)
    _need(encode(snapshot)==reference,'retained target snapshot differs')
    inventory=artifact_reference(graph.report['inventory']['uri'],'scar.target-inventory')
    read_artifact(inventory)
    records=[encode(H.TargetResourceRef(snapshot,row.id))for row in resources]
    return inventory,records,resources,snapshot


def _stage_resources(resources,snapshot,cpus,executor):
    from scar.ir.program import target_hardware as H
    from scar.ir.record_codec import encode
    return [encode(H.TargetResourceRef(snapshot,row.id))for row in resources
        if str(row.id)==executor or type(row)is H.HardwareThread and row.cpu_id in cpus]


def _members(reference,metadata):
    directory=Path(reference['uri']).parent;result={}
    for key,name in metadata['files'].items():
        _need(type(name)is str and Path(name).name==name,'native member escapes its admitted directory')
        value=_file(directory/name,'scar.native-member')
        if key in metadata.get('sha256',{}):
            _need(value['sha256']==metadata['sha256'][key],'native member bytes differ: '+key)
        result[key]=value
    return result


def _bytes(shape):
    _need(type(shape)in(list,tuple)and len(shape)==2 and all(type(v)is int and v>=0 for v in shape),
        'exact nonnegative rank-two shape required')
    return shape[0]*shape[1]*8


def _memory(*,budget,scope,private,fresh,borrowed,include_borrowed,inputs,slots,excludes):
    _need(all(type(value)is int and value>=0 for value in (budget,private,fresh,borrowed)),
        'explicit nonnegative payload byte counts required')
    _need(type(scope)is str and scope and type(include_borrowed)is bool,'original payload scope required')
    included=private+fresh+(borrowed if include_borrowed else 0)
    _need(included<=budget,'controlled region payload exceeds its scoped budget')
    return dict(budget_scope=scope,budget_bytes=budget,included_payload_bytes=included,
        owned_payload_bytes=private+fresh,borrowed_payload_bytes=borrowed,
        borrowed_included_in_budget=include_borrowed,fresh_output_bytes=fresh,private_workspace_bytes=private,
        borrowed_inputs=inputs,private_slots=slots,budget_excludes=deepcopy(excludes),
        allocation_budget_is_physical_reservation=False,physical_numa_residency_proven=False)


def _execution(allowed,affinity,*,allowed_role):
    return dict(caller_lane=dict(capacity=1,kind='CONTROLLED_SYNCHRONOUS_CALL_CONCURRENCY',
            units_per_invocation=1,physical_cpu_id=None,exclusive_core=False,
            enforcement='SINGLE_SYNCHRONOUS_INVOCATION_SCOPE_NOT_INTER_CALL_SERIALIZATION',
            cross_invocation_serialization_enforced=False),
        native=dict(cpu_threads=1,creates_worker_threads=False,native_ffi_calls_per_run=1,asynchronous=False),
        affinity_stage=affinity,allowed_cpu_ids=deepcopy(allowed),allowed_cpu_ids_role=allowed_role,
        cpu_domain_is_exclusive_reservation=False,whole_entry_cpu_assignment_proven=False)


def _fusion(parent):
    from scar.compiler.plan_semantic_region import check
    build,pipeline,_,blueprint,report,revision,_,_,_=check(parent)
    _need(build['decision']=='REWRITE'and build['native_artifact']is not None
        and build['native_binding']is not None,'actual source-admitted fused native validation build required')
    preparation=read_artifact(build['preparation']);macro=read_artifact(preparation['macro_preparation'])
    blueprint_ref=macro['files']['blueprint']
    _need(read_artifact(blueprint_ref)==blueprint,'parent fusion blueprint differs')
    binding=read_artifact(build['native_binding']);native=read_artifact(build['native_artifact'])
    operator=read_artifact(binding['operator_manifest']);members=_members(binding['operator_manifest'],operator)
    sidecars=_members(build['native_artifact'],native)
    execution=blueprint['execution'];selected=blueprint['selected_memory_plan']
    _need(execution['cpu_threads']==1 and execution['resource_binding']is None
        and execution['required_cpu_ids']==[]and execution['creates_workers']is False
        and native['workspace_bytes']==0 and native['native_ffi_calls_per_run']==1,
        'only existing unbound serial fusion is admitted')
    inputs=[]
    for name in pipeline.parameters:
        shape=blueprint['input_shapes'][name]
        inputs.append(dict(id=name,path=[name],shape=deepcopy(shape),dtype='float64',nbytes=_bytes(shape),
            ownership='CALLER_OWNED_BORROWED_READ_ONLY_UNDER_Q',
            source_ref=_fact(blueprint_ref,'#/input_shapes/'+name)))
    borrowed=sum(row['nbytes']for row in inputs)
    _need(selected['total_workspace_bytes']==0 and selected['peak_allocation_bytes']==selected['final_bytes']
        and native['final_bytes']==selected['final_bytes']and native['memory_budget_bytes']==selected['memory_budget_bytes'],
        'fusion native payload differs from selected representation')
    memory=_memory(budget=native['memory_budget_bytes'],scope=native['budget_scope'],private=0,
        fresh=native['final_bytes'],borrowed=borrowed,include_borrowed=False,inputs=inputs,slots=[],excludes=native['budget_excludes'])
    inventory,target_resources,resources,snapshot=_target(blueprint['target'])
    identity=dict(selected_oir_region=binding['selected_oir']['region'],selected_oir_alternative=binding['selected_oir']['selected'],
        semantic_graph_revision=binding['graph_revision'],semantic_root=binding['root'],
        optimized_plan_sha256=binding['optimized_plan_sha256'],strategy_sha256=binding['strategy_sha256'],
        native_binding=build['native_binding'],blueprint_artifact_sha256=blueprint_ref['sha256'],
        blueprint_semantic_sha256=semantic_sha256(blueprint))
    affinity=dict(policy='INHERIT_CALLER_CPU',cpu_ids=[],executor_id=None,target_resource_refs=[],
        phase='NO_NATIVE_AFFINITY_BINDING',excludes=[],caller_mask_restored_by_native=None)
    runtime={}
    for key in ('caller_guard_source','delivery_adapter_source'):
        stamp=blueprint['backend'][key];ref=_file(stamp['path'])
        _need(ref['sha256']==stamp['sha256'],'fusion caller/runtime source stamp differs');runtime[key]=ref
    runtime['cpp_adapter_source']=_file(Path(__file__).resolve().parents[1]/'codegen/cpp.py')
    anchors=[_anchor(runtime['cpp_adapter_source'],'output = np.empty(shape, dtype=np.float64)',
        'FRESH_FINAL_ALLOCATED_AT_PYTHON_OPERATOR_BOUNDARY_BEFORE_SYNCHRONOUS_NATIVE_CALL')]
    return dict(identity=identity,target=blueprint['target'],target_inventory=inventory,target_resources=target_resources,
        execution=_execution(execution['allowed_cpu_ids'],affinity,
            allowed_role='EXACT_CALLER_MASK_GUARD_COST_APPLICABILITY_ONLY_NOT_A_RESERVATION'),memory=memory,
        implementation=dict(native_manifest=build['native_artifact'],build_id=native['build_id'],
            operator_manifests=[binding['operator_manifest']],native_members=members,sidecars=sidecars,
            runtime_sources=runtime,runtime_stamps=deepcopy(blueprint['backend']),source_anchors=anchors),
        completion=dict(synchronous_return=True,fresh_output_ownership='CALLER_OWNED_ON_SUCCESS',
            external_input_write_authorized=False,commit='FRESH_FINAL_RETURNED_AFTER_NATIVE_SUCCESS',
            failure='EXISTING_GUARDED_ORIGINAL_FALLBACK',
            source_ref=_fact(build['native_artifact'],'#/final_ownership')),
        conditions=dict(external_assumptions=deepcopy(blueprint['qualification']['external_assumptions']),
            original_Q_ref=_fact(blueprint_ref,'#/qualification/original_contract'),
            pipeline_assumptions=deepcopy(list(pipeline.assumptions)),
            native_source_ref=members['source'],runtime_guards_required=True,runtime_guards_executed_now=False,
            allocation_must_succeed=True,allworld_effect_closure_claimed=False),
        cold_admission=dict(api='scar.compiler.plan_semantic_region.check',parent_schema=build['schema'],
            decision=report['decision'],semantic_revision=build['semantic_revision'],
            native_manifest_source_and_plan_replayed=True,native_library_loaded_now=False))


def _stateful(parent,context):
    from scar.compiler.stateful_model import stateful_from_dict,stateful_digest
    from scar.compiler.model import CpuPlacement
    materials=context.stateful(parent)
    admission=materials['admission'];specialization=materials['specialization'];candidate=materials['candidate']
    domain=materials['domain'];native=materials['native'];members=materials['members']
    directory=Path(candidate['native_artifact']['uri']).parent
    region=stateful_from_dict(materials['region']);plan=materials['plan'];binding=materials['placement']
    placement=CpuPlacement(**{**binding,'cpu_ids':tuple(binding['cpu_ids']),'evidence':tuple(binding.get('evidence',()))})
    provider_ref=materials['provider_binary']
    stage_plan=artifact_reference(directory/native['files']['plan'],'scar.stateful-stage-plan')
    inputs=[]
    for index,row in enumerate(region.inputs):
        if row.kind!='array':continue
        shape=plan['input_shapes'][row.id]
        inputs.append(dict(id=row.id,path=list(row.path),shape=deepcopy(shape),dtype='float64',nbytes=_bytes(shape),
            ownership='CALLER_OWNED_BORROWED_COMMIT_ONLY'if row.id in plan['shadows']else'CALLER_OWNED_BORROWED_READ_ONLY_UNDER_Q',
            source_ref=_fact(specialization['source_region'],'#/inputs/'+str(index))))
    slots=[dict(id=row['id'],capacity_bytes=row['capacity_bytes'],values=deepcopy(row['values']),
        lifetime='NATIVE_ENTRY_ALLOCATION_THROUGH_NATIVE_RETURN',source_ref=_fact(stage_plan,'#/workspace_slots/'+str(index)))
        for index,row in enumerate(plan['workspace_slots'])]
    borrowed=sum(row['nbytes']for row in inputs);private=sum(row['capacity_bytes']for row in slots)
    _need(borrowed==plan['external_array_bytes']==native['external_array_bytes']
        and private==plan['workspace_payload_bytes']==native['workspace_payload_bytes']
        and borrowed+private==plan['total_region_array_payload_bytes']==native['total_region_array_payload_bytes']
        and plan['memory_budget_bytes']==native['memory_budget_bytes'],
        'native borrowed/private/included payload accounting differs')
    memory=_memory(budget=plan['memory_budget_bytes'],scope=plan['budget_scope'],private=private,fresh=0,
        borrowed=borrowed,include_borrowed=True,inputs=inputs,slots=slots,excludes=plan['budget_excludes'])
    target=domain['target_snapshot'];inventory,target_resources,resources,snapshot=_target(target)
    _need(target_resources==domain['all_resources'],'stateful retained hardware resource inventory differs')
    cpus=list(placement.cpu_ids)
    affinity=dict(policy='NATIVE_STAGE_REQUIRED_CPU_MASK',cpu_ids=cpus,executor_id=placement.executor_id,
        target_resource_refs=_stage_resources(resources,snapshot,cpus,placement.executor_id),
        phase='PRIVATE_SNAPSHOTS_AND_COMPUTATION_UNTIL_PRECOMMIT_ENVIRONMENT_RESTORATION',
        excludes=['PYTHON_GUARDS_AND_ABI_PACKING','PRIVATE_SLOT_ALLOCATION','ENTRY_AND_INPUT_FINITE_GUARDS',
            'EXTERNAL_COMMIT_COPIES','PRIVATE_SLOT_DESTRUCTION_AND_RETURN'],caller_mask_restored_by_native=True)
    root=Path(__file__).resolve().parents[1]
    runtime={name:_file(root/name)for name in ('runtime/stateful_guard.py','compiler/stateful_runtime.py',
        'compiler/stateful_runtime.py','codegen/stateful.py','codegen/stateful_v1.cpp.in')}
    anchors=[_anchor(members['source'],text,meaning)for text,meaning in (
        ('std::array<std::unique_ptr<double[]>','PRIVATE_SLOTS_ALLOCATED_BEFORE_AFFINITY_SCOPE'),
        ('if (!caller_affinity.bind(planned_cpu_ids[0]))','SERIAL_PRIVATE_EXECUTION_AFFINITY_START'),
        ('const bool affinity_restored = caller_affinity.restore_original();','ORIGINAL_CALLER_AFFINITY_RESTORED_BEFORE_COMMIT'),
        ('// All failing work is complete.','EXTERNAL_ROI_COMMITS_START_AFTER_ALL_FAILING_WORK'))]
    return dict(identity=dict(candidate_id=candidate['candidate_id'],domain_id=domain['domain_id'],region_id=region.id,
            region_sha256=stateful_digest(region),stage_plan_sha256=native['plan_sha256'],stage_plan=stage_plan,
            domain=specialization['domain'],candidate=specialization['candidate']),
        target=target,target_inventory=inventory,target_resources=target_resources,
        execution=_execution(domain['live_constraints']['process_affinity_cpu_ids'],affinity,
            allowed_role='RETAINED_CALLER_AVAILABILITY_SNAPSHOT_NATIVE_REQUIRES_ONLY_STAGE_MASK_CONTAINMENT'),memory=memory,
        implementation=dict(native_manifest=candidate['native_artifact'],build_id=native['build_id'],
            operator_manifests=[],native_members=members,runtime_sources=runtime,
            runtime_stamps=dict(implementation_sha256=native['implementation_sha256'],compiler=deepcopy(native['compiler']),
                provider_runtime=deepcopy(native['provider_runtime']),provider_binary=provider_ref),source_anchors=anchors),
        completion=dict(synchronous_return=True,fresh_output_ownership=None,external_input_write_authorized=True,
            commit=plan['transaction']['commit'],transaction=deepcopy(plan['transaction']),
            commit_rows_ref=_fact(stage_plan,'#/commit_rows'),failure=native['failure_policy'],
            source_ref=_fact(stage_plan,'#/transaction')),
        conditions=dict(external_assumptions=deepcopy(plan['external_assumptions']),
            original_Q_ref=_fact(stage_plan,'#/guard_contract'),guard_contract=deepcopy(plan['guard_contract']),
            numerical_contract_ref=_fact(stage_plan,'#/numerical_contract'),
            native_source_ref=members['source'],runtime_guards_required=True,runtime_guards_executed_now=False,
            allocation_must_succeed=True,allworld_effect_closure_claimed=False),
        cold_admission=dict(api='scar.compiler.stateful_candidate.check_observed_specialization',
            report=admission,native_manifest_source_and_plan_replayed=True,native_library_loaded_now=False))


def derive_controlled_resource_contract(kind,admission,*,_context=None):
    """Cold-admit the parent and extract scoped control; never CDLL or compile."""
    schema={PIPELINE_FUSION:'scar.semantic-region-build',STATEFUL_SPECIALIZATION:'scar.observed-stateful-specialization'}
    _need(kind in schema,'supported source-admitted native backend kind required')
    from scar.compiler.admission_context import AdmissionContext
    if kind==STATEFUL_SPECIALIZATION and _context is None:
        with AdmissionContext() as context:
            return derive_controlled_resource_contract(kind,admission,_context=context)
    if _context is not None:
        _need(type(_context)is AdmissionContext,'original live AdmissionContext required')
        _context.validate()
    parent=_reference(admission,schema[kind])
    value=_fusion(parent)if kind==PIPELINE_FUSION else _stateful(parent,_context)
    result=dict(schema=SCHEMA,schema_version=1,kind=kind,parent_admission=parent,
        scope='SCOPED_SYNCHRONOUS_NATIVE_REGION_NORMAL_SUCCESS_UNDER_ORIGINAL_Q',**value,
        unmodeled_context=[dict(field=field,status='UNMODELED_CONTEXT',required_for_this_scoped_contract=False)
            for field in ('whole_python_entry_cpu_assignment','opaque_python_and_provider_memory',
                'allocator_metadata_and_alignment_overhead','process_RSS','physical_NUMA_residency',
                'hardware_cache_occupancy_and_traffic','available_physical_capacity','foreign_concurrent_resource_usage')],
        qualification=dict(controlled_scoped_resources_extracted=True,whole_python_entry_resource_contract=False,
            complete_physical_feasibility_proven=False,physical_cpu_or_memory_reserved=False,
            timing_qualified=False,deployment_authorized=False,native_abi_table_loaded_now=False,
            default_backend='NO_OP'),
        producer=artifact_reference(Path(__file__).resolve(),'scar.implementation-source'))
    read_artifact(parent)
    if _context is not None:
        program=read_artifact(parent)['program']if kind==STATEFUL_SPECIALIZATION else None
        _context._watch_tree(result,_program=program);_context._producers();_context.validate()
    return deepcopy(result)


def verify_controlled_resource_contract(contract,kind,admission,*,_context=None):
    expected=derive_controlled_resource_contract(kind,admission,_context=_context)
    _need(contract==expected,'controlled native-region resource contract differs from complete parent/source replay')
    return expected


__all__=['SCHEMA','PIPELINE_FUSION','STATEFUL_SPECIALIZATION',
    'derive_controlled_resource_contract','verify_controlled_resource_contract']
