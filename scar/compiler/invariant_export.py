"""Invariant export."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path


# Replay original F4/resource admission once; export identical temporal ELF.
def _json(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def _receipt(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return dict(path=str(path),sha256=sha256(raw).hexdigest(),bytes=len(raw))
def _source(value):return {name:getattr(value,name) for name in ('path','sha256','code_key','qualname')}


def export_temporal_invariant(native_directory,out,*,schedule_path,proof_reference,catalog):
    from scar.compiler.temporal_invariant_loop import load_temporal_invariant_loop
    from scar.compiler.loop_admission import read_invariant_loop_motion
    from scar.optimization.invariant_loop_schedule import verify_invariant_loop_schedule
    from scar.optimization.resource_solver import semantic_sha256
    from scar.ir.execution_recipe import read_artifact
    from scar.runtime.native_temporal_invariant import SCHEMA,ADMISSION,TRUST,validate_contract
    native_directory=Path(native_directory).resolve()
    native_manifest=json.loads((native_directory/'artifact.json').read_bytes())
    read_artifact(proof_reference)
    proof,source_motion=read_invariant_loop_motion(proof_reference['uri'],catalog=catalog)
    raw_plan=json.loads((native_directory/'plan.json').read_bytes())
    motion_ref=raw_plan['loop_interface'].get('source_motion')
    if (motion_ref is None or source_motion['proof']!=proof_reference
            or semantic_sha256(read_artifact(motion_ref))!=semantic_sha256(source_motion)):
        raise ValueError('temporal export requires its compiled current source movement certificate')
    parallel=native_manifest.get('schema')=='scar.native-parallel-temporal-invariant-loop'
    if parallel:
        from scar.compiler.temporal_invariant_loop import load_temporal_invariant_loop as load_parallel_temporal_invariant_loop
        artifact=load_parallel_temporal_invariant_loop(native_directory)
    else:
        artifact=load_temporal_invariant_loop(native_directory)
    if semantic_sha256(asdict(proof))!=semantic_sha256(artifact.plan['loop_interface']['original_f4_proof']):
        raise ValueError('temporal export original F4 proof differs from compiled interface')
    consumed=artifact.plan.get('execution_movement_binding')
    if (consumed is None or artifact.plan['loop_interface'].get('source_motion')!=motion_ref
            or consumed['source_motion']!=motion_ref
            or consumed['source_motion_sha256']!=semantic_sha256(source_motion)):
        raise ValueError('temporal export source movement was not consumed by the compiled plan')
    schedule_path=Path(schedule_path).resolve();schedule_raw=schedule_path.read_bytes()
    schedule_receipt=dict(path=str(schedule_path),sha256=sha256(schedule_raw).hexdigest(),bytes=len(schedule_raw))
    schedule=json.loads(schedule_raw)
    cost_selected=schedule.get('schema')=='scar.cost-selected-temporal-cpu-resource-plan'
    if parallel:
        from scar.optimization.temporal_cpu_choices import (verify_temporal_cpu_preview,
            verify_selected_temporal_cpu_choice)
        if cost_selected:
            verify_selected_temporal_cpu_choice(proof,schedule['target_path'],schedule)
        elif schedule.get('schema')=='scar.parallel-temporal-invariant-loop-resource-plan':
            verify_temporal_cpu_preview(proof,schedule['target_path'],schedule)
        else:raise ValueError('parallel export requires its original replayable CPU-team resource schedule')
        backend=schedule['backend'];compiled=artifact.report['capability']
        if any(semantic_sha256(backend[key])!=semantic_sha256(compiled[key])
                for key in ('family','compiler','generators','row_partition')):
            raise ValueError('parallel compiled backend differs from scheduled capability')
    else:
        verify_invariant_loop_schedule(proof,schedule['target_path'],schedule)
    if schedule['selected_kind']!='REGISTER_TEMPORAL_FUSION' or schedule['trip_count']<=0:
        raise ValueError('temporal export requires an admitted positive-count resource schedule')
    binding=artifact.plan['execution_plan']['resource_binding']
    if (binding is None or semantic_sha256(binding)!=semantic_sha256(schedule['resource_binding'])
            or artifact.plan['memory_budget_bytes']!=schedule['effective_incremental_budget_bytes']
            or artifact.plan['body_sha256']!=schedule['body_sha256']):
        raise ValueError('temporal export compiled resource/body boundary differs from schedule')
    specialization=artifact.plan['loop_interface'].get('execution_specialization')
    if specialization is None or specialization['input_shapes']!=schedule['input_shapes'] or specialization['trip_count']!=schedule['trip_count']:
        raise ValueError('temporal export requires exact compiled shape/count specialization')
    evidence=[_receipt(artifact.directory/'artifact.json'),_receipt(artifact.directory/'plan.json'),
        schedule_receipt,_receipt(proof_reference['uri']),_receipt(proof.source.path),
        _receipt(proof.body.source.path),_receipt(motion_ref['uri']),*artifact.report['generators']]
    if cost_selected:
        for reference in [*schedule['resource_facts'],schedule['no_op_profile']]:
            read_artifact(reference);evidence.append(_receipt(reference['uri']))
    native_bytes=(artifact.directory/'temporal.so').read_bytes();digest=sha256(native_bytes).hexdigest()
    family='CPP_OPENMP_EXPLICIT_ROW_PARTITION_TEMPORAL' if parallel else 'CPP_SERIAL_TEMPORAL'
    contract=dict(schema=SCHEMA,schema_version=1,native=dict(library='temporal.so',library_sha256=digest,
        operator_descriptor_abi_version=1,temporal_control_abi_version=1,control_family=family),
        source=_source(proof.source),leaf=dict(global_name=proof.leaf_global_name,source=_source(proof.body.source)),
        parameters=list(proof.parameters),input_order=list(proof.body.parameters),parameter_map=proof.parameter_map,
        carry_parameter=proof.carry_parameter,trip_count_parameter=proof.trip_count_parameter,
        trip_count=schedule['trip_count'],arrays={name:dict(shape=shape,dtype='float64',layout='C_CONTIGUOUS')
            for name,shape in schedule['input_shapes'].items()},
        memory_budget_bytes=schedule['effective_incremental_budget_bytes'],
        memory_scope=artifact.plan['memory_scope'],memory_policy='UNBOUND',
        cpu=dict(ids=list(binding['cpu_ids']),threads=len(binding['cpu_ids']),os='linux',architecture='x86_64',pointer_bits=64,
            byteorder='little',required_isa='BASELINE_X86_64'),
        floating_environment_policy='ORIGINAL_CPP_SAVE_RESTORE_FENV_STRICT_OPERAND_ORDER_PRIVATE_FAILURE_OUTPUT',
        Q=artifact.plan['Q'],source_proof=proof_reference,
        resource_schedule=dict(id=schedule['id'],receipt=schedule_receipt),
        original_f4_proof_sha256=semantic_sha256(asdict(proof)),
        selection_authority='COMPLETE_SOURCED_FORWARD_RESOURCE_MODEL_NOT_FULL_ENTRY_PERFORMANCE' if cost_selected else
            'EXPLICIT_CPU_TEAM_PREVIEW_NOT_COST_SELECTION' if parallel else
            'STRUCTURAL_LOGICAL_WORK_RESOURCE_DOMINANCE_NOT_PERFORMANCE_SELECTION',
        profitability_status='UNPROVEN_NO_SOURCED_TIMING_MODEL',deployment_authorized=False,
        validation_execution_requires_trial=True)
    if cost_selected:
        contract.update(resource_cost_basis=schedule['resource_cost_basis'],
            forward_model_selected=True,full_entry_performance_qualified=False)
    validate_contract(contract)
    raw=_json(contract)
    admission=dict(schema=ADMISSION,schema_version=1,runtime_sha256=sha256(raw).hexdigest(),
        library_sha256=digest,trust_policy=TRUST,evidence=evidence,
        producer=dict(name='SCAR',version='0.1.0',exporter=_receipt(__file__)),
        original_f4_replayed=True,resource_schedule_replayed=True,native_math_changed=False,
        performance_qualified=False,deployment_authorized=False)
    admission_raw=_json(admission)
    for receipt in evidence:
        if _receipt(receipt['path'])!=receipt:raise ValueError('temporal evidence changed during export')
    read_artifact(proof_reference)
    out=Path(out).resolve()
    if out.exists() or out.is_relative_to(artifact.directory):raise ValueError('fresh separate temporal bundle directory required')
    out.mkdir(parents=True);(out/'temporal.so').write_bytes(native_bytes)
    (out/'runtime.json').write_bytes(raw);(out/'admission.json').write_bytes(admission_raw)
    result=dict(schema='scar.temporal-invariant-runtime-export',schema_version=1,
        bundle=str(out),expected_admission_sha256=sha256(admission_raw).hexdigest(),
        runtime_sha256=sha256(raw).hexdigest(),library_sha256=digest,
        source_directory=str(artifact.directory),schedule=schedule_receipt,proof=proof_reference,
        original_f4_replayed=True,resource_schedule_replayed=True,native_recompiled=False,
        control_family=family,cpu_threads=len(binding['cpu_ids']),
        native_library_bytes_unchanged=True,performance_qualified=False,deployment_authorized=False,
        runtime_ir_or_codegen_required=False,bundle_bytes=len(raw)+len(admission_raw)+len(native_bytes))
    (out/'export-report.json').write_bytes(_json(result));return result


# Separate composite F4/fusion/global-route admission to compact delivery.
def export_composite_temporal_invariant(native,out,*,schedule_path,proof_reference,catalog):
    from scar.compiler.temporal_invariant_loop import load_temporal_invariant_loop
    from scar.compiler.pipeline_loop_frontend import read_composite_invariant_loop_motion,COMPOSITE_PROOF_SCHEMA
    from scar.optimization.composite_invariant_schedule import verify_composite_invariant_loop_schedule
    from scar.optimization.resource_solver import semantic_sha256
    from scar.ir.execution_recipe import read_artifact
    from scar.runtime.native_temporal_invariant import SCHEMA,ADMISSION,TRUST,validate_contract
    if proof_reference['schema']!=COMPOSITE_PROOF_SCHEMA:
        raise ValueError('separate composite proof reference required; v1 cast is not admission')
    proof_document=read_artifact(proof_reference)
    proof,source_motion=read_composite_invariant_loop_motion(proof_reference['uri'],catalog=catalog)
    native=Path(native).resolve();manifest=json.loads((native/'artifact.json').read_bytes())
    raw_plan=json.loads((native/'plan.json').read_bytes())
    motion_ref=raw_plan['loop_interface'].get('source_motion')
    if (motion_ref is None or source_motion['proof']!=proof_reference
            or semantic_sha256(read_artifact(motion_ref))!=semantic_sha256(source_motion)):
        raise ValueError('composite export requires its compiled current source movement certificate')
    parallel=manifest.get('schema')=='scar.native-parallel-temporal-invariant-loop'
    if parallel:
        from scar.compiler.temporal_invariant_loop import load_temporal_invariant_loop as load_parallel_temporal_invariant_loop
        artifact=load_parallel_temporal_invariant_loop(native)
    else:
        artifact=load_temporal_invariant_loop(native)
    interface=artifact.plan['loop_interface']
    consumed=artifact.plan.get('execution_movement_binding')
    if (consumed is None or interface.get('source_motion')!=motion_ref
            or consumed['source_motion']!=motion_ref
            or consumed['source_motion_sha256']!=semantic_sha256(source_motion)):
        raise ValueError('composite export source movement was not consumed by the compiled plan')
    if (interface.get('proof_kind')!=proof.profile or interface.get('synthetic_body_is_original_callable') is not False
            or semantic_sha256(interface.get('original_composite_f4_proof'))!=semantic_sha256(asdict(proof))
            or semantic_sha256(interface.get('root_callee_source'))!=semantic_sha256(asdict(proof.root_callee_source))
            or semantic_sha256(interface.get('function_global_guards'))!=semantic_sha256([asdict(g) for g in proof.nested_callee_guards])):
        raise ValueError('compiled composite body/root/routes differ from original separate F4 proof')
    schedule_path=Path(schedule_path).resolve();schedule_raw=schedule_path.read_bytes()
    schedule_receipt=dict(path=str(schedule_path),sha256=sha256(schedule_raw).hexdigest(),bytes=len(schedule_raw))
    schedule=json.loads(schedule_raw)
    schema=schedule.get('schema')
    if schema=='scar.composite-invariant-loop-resource-plan':
        if parallel:raise ValueError('serial structural composite plan cannot authorize a parallel artifact')
        verify_composite_invariant_loop_schedule(proof,schedule['target_path'],schedule,catalog=catalog)
    elif schema in ('scar.parallel-composite-temporal-resource-plan','scar.cost-selected-composite-temporal-cpu-resource-plan'):
        from scar.optimization.composite_temporal_cpu_choices import (
            verify_composite_temporal_cpu_preview,verify_selected_composite_temporal_cpu_choice)
        if not parallel:raise ValueError('parallel composite CPU choice requires the actual OpenMP artifact')
        verifier=(verify_composite_temporal_cpu_preview if schema=='scar.parallel-composite-temporal-resource-plan'
            else verify_selected_composite_temporal_cpu_choice)
        verifier(proof,schedule['target_path'],schedule,catalog=catalog)
        scheduled=schedule['backend'];compiled=artifact.report['capability']
        if any(semantic_sha256(scheduled[key])!=semantic_sha256(compiled[key])
                for key in ('family','compiler','generators','row_partition')):
            raise ValueError('composite CPU choice backend/compiler/partition differs from actual native artifact')
    else:raise ValueError('separate replayable composite resource plan required; v1 plan casts are not admission')
    if schedule['selected_kind']!='REGISTER_TEMPORAL_FUSION' or schedule['trip_count']<=0:
        raise ValueError('composite positive-count resource schedule required for native export')
    binding=artifact.plan['execution_plan'].get('resource_binding')
    if (binding is None or semantic_sha256(binding)!=semantic_sha256(schedule['resource_binding'])
            or artifact.plan['body_sha256']!=schedule['body_sha256']
            or artifact.plan['memory_budget_bytes']!=schedule['effective_incremental_budget_bytes']):
        raise ValueError('composite native body/placement/budget differs from replayed schedule')
    specialization=interface.get('execution_specialization')
    if specialization is not None and (specialization['input_shapes']!=schedule['input_shapes']
            or specialization['trip_count']!=schedule['trip_count'] or specialization.get('input_layout')!='C_CONTIGUOUS'):
        raise ValueError('composite compiled specialization differs from its schedule')
    sources=(proof.source,proof.root_callee_source,*(source for guard in proof.nested_callee_guards
        for source in (guard.owner_source,guard.target_source)))
    evidence=[_receipt(artifact.directory/'artifact.json'),_receipt(artifact.directory/'plan.json'),
        schedule_receipt,_receipt(proof_reference['uri']),_receipt(motion_ref['uri']),*[_receipt(source.path) for source in sources],
        *artifact.report['generators']]
    for reference in proof_document['compositions']:
        read_artifact(reference);evidence.append(_receipt(reference['uri']))
    native_bytes=(artifact.directory/'temporal.so').read_bytes();digest=sha256(native_bytes).hexdigest()
    if digest!=artifact.report['library']['sha256'] or len(native_bytes)!=artifact.report['library']['bytes']:
        raise ValueError('composite native library changed after source admission')
    guards=[dict(owner_source=_source(g.owner_source),lookup_name=g.lookup_name,
        target_source=_source(g.target_source),global_path=list(g.global_path),lookup_semantics=g.lookup_semantics)
        for g in proof.nested_callee_guards]
    family='CPP_OPENMP_EXPLICIT_ROW_PARTITION_TEMPORAL' if parallel else 'CPP_SERIAL_TEMPORAL'
    contract=dict(schema=SCHEMA,schema_version=1,proof_kind='COMPOSITE',
        body_identity=proof.body_identity,derived_body_sha256=artifact.plan['body_sha256'],
        native=dict(library='temporal.so',library_sha256=digest,operator_descriptor_abi_version=1,
            temporal_control_abi_version=1,control_family=family),
        source=_source(proof.source),leaf=dict(global_name=proof.root_global_name,source=_source(proof.root_callee_source)),
        root_callee_source=_source(proof.root_callee_source),nested_callee_guards=guards,
        parameters=list(proof.parameters),input_order=list(proof.body.parameters),parameter_map=proof.parameter_map,
        carry_parameter=proof.carry_parameter,trip_count_parameter=proof.trip_count_parameter,
        trip_count=schedule['trip_count'],arrays={name:dict(shape=shape,dtype='float64',layout='C_CONTIGUOUS')
            for name,shape in schedule['input_shapes'].items()},memory_budget_bytes=schedule['effective_incremental_budget_bytes'],
        memory_scope=artifact.plan['memory_scope'],memory_policy='UNBOUND',
        cpu=dict(ids=list(binding['cpu_ids']),threads=len(binding['cpu_ids']),os='linux',architecture='x86_64',
            pointer_bits=64,byteorder='little',required_isa='BASELINE_X86_64'),
        floating_environment_policy='ORIGINAL_CPP_SAVE_RESTORE_FENV_STRICT_OPERAND_ORDER_PRIVATE_FAILURE_OUTPUT',
        Q=artifact.plan['Q'],source_proof=proof_reference,resource_schedule=dict(id=schedule['id'],receipt=schedule_receipt),
        original_composite_f4_proof_sha256=semantic_sha256(asdict(proof)),
        specialization_authority='COMPILED_INTERFACE' if specialization is not None else 'EXPORT_RUNTIME_NARROWING_OF_GENERIC_NATIVE',
        selection_authority=('COMPLETE_COMPOSITE_CPU_FORWARD_COST_MODEL_NOT_ENTRY_PERFORMANCE_SELECTION'
            if schema=='scar.cost-selected-composite-temporal-cpu-resource-plan' else
            'EXPLICIT_COMPOSITE_CPU_TEAM_PREVIEW_NOT_COST_SELECTION' if parallel else
            'STRUCTURAL_COMPOSITE_LOGICAL_WORK_NOT_PERFORMANCE_SELECTION'),
        profitability_status='UNPROVEN_NO_SOURCED_TIMING_MODEL',deployment_authorized=False,validation_execution_requires_trial=True)
    validate_contract(contract);raw=_json(contract)
    admission=dict(schema=ADMISSION,schema_version=1,runtime_sha256=sha256(raw).hexdigest(),library_sha256=digest,
        trust_policy=TRUST,evidence=evidence,producer=dict(name='SCAR',version='0.1.0',exporter=_receipt(__file__)),
        proof_kind='COMPOSITE',original_composite_f4_replayed=True,resource_schedule_replayed=True,
        control_family=family,cpu_threads=len(binding['cpu_ids']),resource_plan_schema=schema,
        synthetic_body_is_original_callable=False,performance_qualified=False,deployment_authorized=False)
    admission_raw=_json(admission)
    for receipt in evidence:
        if _receipt(receipt['path'])!=receipt:raise ValueError('composite export evidence changed during admission')
    read_artifact(proof_reference)
    read_artifact(motion_ref,json_document=False)
    for reference in proof_document['compositions']:read_artifact(reference,json_document=False)
    out=Path(out).resolve()
    if out.exists() or out.is_relative_to(artifact.directory):raise ValueError('fresh separate composite bundle directory required')
    out.mkdir(parents=True);(out/'temporal.so').write_bytes(native_bytes)
    (out/'runtime.json').write_bytes(raw);(out/'admission.json').write_bytes(admission_raw)
    result=dict(schema='scar.composite-temporal-invariant-runtime-export',schema_version=1,bundle=str(out),
        expected_admission_sha256=sha256(admission_raw).hexdigest(),runtime_sha256=sha256(raw).hexdigest(),
        library_sha256=digest,source_directory=str(artifact.directory),schedule=schedule_receipt,proof=proof_reference,
        proof_kind='COMPOSITE',original_composite_f4_replayed=True,resource_schedule_replayed=True,
        native_recompiled=False,native_library_bytes_unchanged=True,synthetic_body_is_original_callable=False,
        runtime_ir_or_codegen_required=False,performance_qualified=False,deployment_authorized=False,
        bundle_bytes=len(raw)+len(admission_raw)+len(native_bytes))
    (out/'export-report.json').write_bytes(_json(result));return result


__all__ = [
    'export_temporal_invariant',
    'export_composite_temporal_invariant',
]
