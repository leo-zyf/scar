"""Separate composite F4/fusion/global-route admission to compact delivery."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from .export_temporal_invariant import _json,_receipt,_source


def export_composite_temporal_invariant(native,out,*,schedule_path,proof_reference,catalog):
    from scar.compiler.temporal_invariant_loop import load_temporal_invariant_loop
    from scar.compiler.pipeline_loop_frontend import read_composite_invariant_loop_proof,COMPOSITE_PROOF_SCHEMA
    from scar.optimization.composite_invariant_schedule import verify_composite_invariant_loop_schedule
    from scar.optimization.resource_solver import semantic_sha256
    from scar.ir.execution_recipe import read_artifact
    from scar.runtime.native_temporal_invariant import SCHEMA,ADMISSION,TRUST,validate_contract
    if proof_reference['schema']!=COMPOSITE_PROOF_SCHEMA:
        raise ValueError('separate composite proof reference required; v1 cast is not admission')
    proof_document=read_artifact(proof_reference)
    proof=read_composite_invariant_loop_proof(proof_reference['uri'],catalog=catalog)
    artifact=load_temporal_invariant_loop(native);interface=artifact.plan['loop_interface']
    if (interface.get('proof_kind')!=proof.profile or interface.get('synthetic_body_is_original_callable') is not False
            or semantic_sha256(interface.get('original_composite_f4_proof'))!=semantic_sha256(asdict(proof))
            or semantic_sha256(interface.get('root_callee_source'))!=semantic_sha256(asdict(proof.root_callee_source))
            or semantic_sha256(interface.get('function_global_guards'))!=semantic_sha256([asdict(g) for g in proof.nested_callee_guards])):
        raise ValueError('compiled composite body/root/routes differ from original separate F4 proof')
    schedule_path=Path(schedule_path).resolve();schedule_raw=schedule_path.read_bytes()
    schedule_receipt=dict(path=str(schedule_path),sha256=sha256(schedule_raw).hexdigest(),bytes=len(schedule_raw))
    schedule=json.loads(schedule_raw)
    verify_composite_invariant_loop_schedule(proof,schedule['target_path'],schedule,catalog=catalog)
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
        schedule_receipt,_receipt(proof_reference['uri']),*[_receipt(source.path) for source in sources],
        *artifact.report['generators']]
    for reference in proof_document['compositions']:
        read_artifact(reference);evidence.append(_receipt(reference['uri']))
    native_bytes=(artifact.directory/'temporal.so').read_bytes();digest=sha256(native_bytes).hexdigest()
    guards=[dict(owner_source=_source(g.owner_source),lookup_name=g.lookup_name,
        target_source=_source(g.target_source),global_path=list(g.global_path),lookup_semantics=g.lookup_semantics)
        for g in proof.nested_callee_guards]
    contract=dict(schema=SCHEMA,schema_version=1,proof_kind='COMPOSITE',
        body_identity=proof.body_identity,derived_body_sha256=artifact.plan['body_sha256'],
        native=dict(library='temporal.so',library_sha256=digest,operator_descriptor_abi_version=1,
            temporal_control_abi_version=1,control_family='CPP_SERIAL_TEMPORAL'),
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
        selection_authority='STRUCTURAL_COMPOSITE_LOGICAL_WORK_NOT_PERFORMANCE_SELECTION',
        profitability_status='UNPROVEN_NO_SOURCED_TIMING_MODEL',deployment_authorized=False,validation_execution_requires_trial=True)
    validate_contract(contract);raw=_json(contract)
    admission=dict(schema=ADMISSION,schema_version=1,runtime_sha256=sha256(raw).hexdigest(),library_sha256=digest,
        trust_policy=TRUST,evidence=evidence,producer=dict(name='SCAR',version='0.1.0',exporter=_receipt(__file__)),
        proof_kind='COMPOSITE',original_composite_f4_replayed=True,resource_schedule_replayed=True,
        synthetic_body_is_original_callable=False,performance_qualified=False,deployment_authorized=False)
    admission_raw=_json(admission)
    for receipt in evidence:
        if _receipt(receipt['path'])!=receipt:raise ValueError('composite export evidence changed during admission')
    read_artifact(proof_reference)
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


__all__=['export_composite_temporal_invariant']
