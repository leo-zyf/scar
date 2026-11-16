"""Expose the existing standalone stateful backend's normal-call work.

The added phases describe implementation obligations, not pure operations or
new lowering permission. Repeat adapters have a different Python boundary.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference,validate_recipe
from scar.ir.versioned_schedule import (VersionedTask as Task,VersionedTaskKind as K,
    TemporalScopeKind,schedule_document,validate_versioned_schedule)
from scar.optimization.resource_solver import semantic_sha256


def derive_stateful_envelope(schedule,recipe):
    """Return an untimed, nonexecutable refinement of one original recipe."""
    validate_versioned_schedule(schedule)
    if schedule.temporal is not None and any(scope.kind is TemporalScopeKind.LOOP
            for scope in schedule.temporal.scopes):
        raise ValueError('standalone method phases cannot describe a repeat adapter')
    original=replace(schedule,temporal=None)
    recipe=validate_recipe(recipe,verify_artifacts=False)
    if schedule_document(original)!=recipe['task_graph']:
        raise ValueError('complete original untimed schedule differs from recipe task graph')
    if (recipe['profile']!='cpu-stateful-array-region-v1'
            or recipe['implementation'].get('family')!='CPP_SYNCHRONOUS_CPU_TRANSACTIONAL_STATEFUL_REGION'
            or recipe['control'].get('fixed_source_order')is not True):
        raise ValueError('original fixed-order transactional stateful implementation required')
    partition=recipe['control'].get('spatial_partition')
    if partition not in('SERIAL_ROWS','OPENMP_STATIC_DISJOINT_ROWS'):
        raise ValueError('source renderer worker-control profile is unresolved')
    from scar.codegen.stateful import implementation_sha256
    backend=recipe['implementation'].get('backend_capability_record',{})
    if backend.get('implementation_sha256')!=implementation_sha256():
        raise ValueError('original recipe backend implementation differs from current source')
    root=Path(__file__).resolve().parents[1]
    files=('runtime/stateful_guard.py','compiler/stateful_runtime.py','codegen/stateful.py','codegen/stateful_v1.cpp.in',
        'codegen/loop_v1.cpp.in','codegen/cpp.py','codegen/stateful_abi_v1.h','codegen/scar_abi_v1.h')
    pins={name:artifact_reference(root/name,'scar.backend-implementation-source')for name in files}
    anchor=dict(recipe_id=recipe['recipe_id'],pointer='#/implementation/backend_capability_record')
    source_refs=(anchor,*(dict(artifact=pins[name],source_file=name)for name in files))
    tasks=original.tasks;by_id={task.id:task for task in tasks}
    allocate=[t for t in tasks if t.kind is K.ALLOCATE]
    stages=[t for t in tasks if t.kind in(K.SNAPSHOT,K.COMPUTE,K.STATE_WRITE)]
    fences=[t for t in tasks if t.kind is K.SUCCESS_FENCE]
    commits=[t for t in tasks if t.kind is K.COMMIT]
    releases=[t for t in tasks if t.kind is K.RELEASE]
    if (not allocate or not stages or len(fences)!=1 or not commits
            or tuple((*allocate,*stages,*fences,*commits,*releases))!=tasks):
        raise ValueError('original allocation/stage/fence/commit/release renderer order required')
    if 'implementation_envelope'in original.proof:
        raise ValueError('implementation envelope is already attached')
    values={value.id:value for value in original.values}
    slots={slot.id:slot for slot in original.storage_slots}
    inputs=recipe['semantic_boundary'].get('inputs')
    if type(inputs)is not list:raise ValueError('original stateful input signature required')
    arrays=[row for row in inputs if row.get('kind')=='array']
    if not arrays or len({row['id']for row in arrays})!=len(arrays):
        raise ValueError('unique original array input records required')
    phases=[]
    def phase(kind,effects,*,actor='CALLER_UNBOUND',failure='PRECOMMIT',may_fail=True,
              blocks='UNKNOWN',reads=(),suffix='',**attrs):
        identity='implementation:'+kind.lower()+suffix
        if identity in by_id or any(row.id==identity for row in phases):
            raise ValueError('implementation phase identity collides with original task')
        row=Task(identity,K.IMPLEMENTATION_PHASE,reads=tuple(reads),capability='CPU_IMPLEMENTATION_PHASE',
            source_refs=source_refs,attrs=dict(phase_kind=kind,actor=actor,failure_domain=failure,
                may_fail=may_fail,blocks=blocks,phase_effects=list(effects),**attrs))
        phases.append(row);return row
    prefix=[phase('PYTHON_ENTRY_GUARD_PACKING',(
        'READ_FUNCTION_AND_PROVIDER_IDENTITIES','READ_ARGUMENT_BINDINGS_AND_INPUT_PATHS',
        'READ_SOURCE_FILES_ON_LAZY_CALLEE_GUARD_CACHE_MISS','WRITE_LAZY_CALLEE_GUARD_CACHE_ENTRIES',
        'VALIDATE_ARRAY_SCALAR_ALIAS_METADATA','ALLOCATE_PYTHON_DESCRIPTORS_AND_SCALARS',
        'TRY_ACQUIRE_NATIVE_PROVIDER_LOCK')),
        phase('NATIVE_METADATA_GUARD',('READ_NATIVE_ABI_DESCRIPTORS','CHECK_ARRAY_PROVIDER_ALIAS_AND_PAYLOAD_METADATA'),blocks=False)]
    capture=phase('CALLER_CAPTURE_SCALAR_GUARD',('CAPTURE_CALLER_CPU_AFFINITY',
        'CHECK_SELECTED_CPU_ELIGIBILITY','SAVE_AND_CHANGE_CALLER_FLOATING_ENVIRONMENT',
        'READ_AND_VALIDATE_TAGGED_SCALARS'),scalar_input_count=sum(row.get('kind')=='scalar'for row in inputs))
    scans=[]
    for index,row in enumerate(arrays):
        shape=row.get('shape')
        if (type(shape)not in(list,tuple)or len(shape)!=2
                or any(type(axis)is not int or axis<0 for axis in shape)):
            raise ValueError('source-specialized array shape required for finite scan')
        material=[m for m in original.materializations if m.external_ready and m.version==0
            and slots[m.region.slot].ownership=='EXTERNAL'and values[m.value].attrs.get('input_id')==row['id']]
        if len(material)!=1:raise ValueError('finite scan needs one original external entry materialization')
        scans.append(phase('INPUT_FINITE_SCAN_SERIAL',('READ_ORIGINAL_EXTERNAL_ARRAY_CONTENT',
            'CHECK_EACH_FLOAT64_VALUE_IS_FINITE'),blocks=False,reads=(material[0].id,),suffix=':'+str(index),
            input_id=row['id'],rows=shape[0],cols=shape[1],external_entry_materialization=material[0].id))
    parallel=partition=='OPENMP_STATIC_DISJOINT_ROWS'
    enter=phase('TEAM_ENTER',('ENTER_SELECTED_OPENMP_TEAM','PIN_SELECTED_WORKERS',
        'SAVE_AND_SET_WORKER_FLOATING_ENVIRONMENT','CHECK_TEAM_PARTICIPATION_AND_BARRIER')if parallel else(
        'BIND_CALLER_TO_SELECTED_SINGLE_CPU',),actor='SELECTED_TEAM'if parallel else'CALLER_UNBOUND',blocks=parallel)
    worker_exit=phase('WORKER_EXIT',('COLLECT_WORKER_FLOATING_STATUS','RESTORE_WORKER_FLOATING_ENVIRONMENT',
        'RESTORE_WORKER_CPU_AFFINITY','JOIN_SELECTED_TEAM')if parallel else('CHECK_CALLER_REMAINS_ON_SELECTED_CPU',),
        actor='SELECTED_TEAM'if parallel else'WORKER',blocks=parallel)
    restore=phase('CALLER_RESTORE',('COLLECT_CALLER_FLOATING_STATUS','RESTORE_CALLER_FLOATING_ENVIRONMENT',
        'RESTORE_ORIGINAL_CALLER_CPU_AFFINITY','CHECK_ALL_PRECOMMIT_FAILURE_FLAGS'))
    returning=phase('NATIVE_RETURN',('RETURN_NATIVE_SUCCESS_STATUS',),failure='POSTCOMMIT',may_fail=False,blocks=False)
    python_return=phase('PYTHON_STATUS_RETURN',('RELEASE_NATIVE_PROVIDER_LOCK','INTERPRET_NATIVE_SUCCESS_STATUS',
        'UPDATE_WRAPPER_NATIVE_CALL_STATISTICS','RETURN_ORIGINAL_NONE_RESULT'),failure='POSTCOMMIT',may_fail=False)
    ordered=(*prefix,*allocate,capture,*scans,enter,*stages,worker_exit,restore,*fences,*commits,*releases,returning,python_return)
    precommit=tuple(row.id for row in phases if row.attrs['failure_domain']=='PRECOMMIT')
    refined=[];previous=None
    for task in ordered:
        dependencies=tuple(dict.fromkeys((*task.depends_on,*(()if previous is None else(previous,)))))
        validations=tuple(dict.fromkeys((*task.validates,*precommit)))if task.kind is K.SUCCESS_FENCE else task.validates
        refined.append(replace(task,depends_on=dependencies,validates=validations));previous=task.id
    proof=deepcopy(original.proof)
    proof['implementation_envelope']=dict(schema='scar.stateful-implementation-envelope',schema_version=1,
        original_recipe_id=recipe['recipe_id'],original_schedule_sha256=semantic_sha256(schedule_document(original)),
        invocation_scope='ONE_STANDALONE_STATEFUL_METHOD_CALL_NOT_REPEAT_ADAPTER',
        python_wrapper_scope='COMPILER_STATEFUL_ARTIFACT_AND_SHARED_STATEFUL_GUARDS_NOT_COMPACT_RUNTIME',
        backend_source_pins=pins,phase_order=[row.id for row in ordered],
        original_task_ids=[row.id for row in tasks],inserted_phase_ids=[row.id for row in phases],
        all_original_task_happens_before_preserved=True,source_data_storage_and_original_proof_fields_preserved=True,
        normal_completion_only=True,exceptional_paths_expanded=False,
        actor_semantics='QUALIFICATION_DOMAIN_NOT_OBSERVED_INSTANTANEOUS_CPU_BINDING',
        postcommit_no_failure_scope='NORMAL_SUCCESS_STATUS_HANDOFF_ONLY_NOT_GENERAL_PYTHON_EXCEPTION_EQUIVALENCE',
        phase_effects_are_explicit_not_purity_claims=True,source_recipe_profile_requires_independent_admission=True,
        repeat_adapter_modeled=False,costs_inferred=False,native_lowering_authorized=False)
    result=replace(original,tasks=tuple(refined),proof=proof,executable=False,temporal=None)
    validate_versioned_schedule(result)
    return result


__all__=['derive_stateful_envelope']
