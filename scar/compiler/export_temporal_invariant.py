"""Replay original F4/resource admission once; export identical temporal ELF."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path


def _json(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def _receipt(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return dict(path=str(path),sha256=sha256(raw).hexdigest(),bytes=len(raw))
def _source(value):return {name:getattr(value,name) for name in ('path','sha256','code_key','qualname')}


def export_temporal_invariant(native_directory,out,*,schedule_path,proof_reference,catalog):
    from scar.compiler.temporal_invariant_loop import load_temporal_invariant_loop
    from scar.compiler.invariant_loop_frontend import read_invariant_loop_proof
    from scar.optimization.invariant_loop_schedule import verify_invariant_loop_schedule
    from scar.optimization.resource_solver import semantic_sha256
    from scar.ir.execution_recipe import read_artifact
    from scar.runtime.native_temporal_invariant import SCHEMA,ADMISSION,TRUST,validate_contract
    artifact=load_temporal_invariant_loop(native_directory)
    read_artifact(proof_reference)
    proof=read_invariant_loop_proof(proof_reference['uri'],catalog=catalog)
    if semantic_sha256(asdict(proof))!=semantic_sha256(artifact.plan['loop_interface']['original_f4_proof']):
        raise ValueError('temporal export original F4 proof differs from compiled interface')
    schedule_path=Path(schedule_path).resolve();schedule_raw=schedule_path.read_bytes()
    schedule_receipt=dict(path=str(schedule_path),sha256=sha256(schedule_raw).hexdigest(),bytes=len(schedule_raw))
    schedule=json.loads(schedule_raw)
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
        _receipt(proof.body.source.path),*artifact.report['generators']]
    native_bytes=(artifact.directory/'temporal.so').read_bytes();digest=sha256(native_bytes).hexdigest()
    contract=dict(schema=SCHEMA,schema_version=1,native=dict(library='temporal.so',library_sha256=digest,
        operator_descriptor_abi_version=1,temporal_control_abi_version=1),
        source=_source(proof.source),leaf=dict(global_name=proof.leaf_global_name,source=_source(proof.body.source)),
        parameters=list(proof.parameters),input_order=list(proof.body.parameters),parameter_map=proof.parameter_map,
        carry_parameter=proof.carry_parameter,trip_count_parameter=proof.trip_count_parameter,
        trip_count=schedule['trip_count'],arrays={name:dict(shape=shape,dtype='float64',layout='C_CONTIGUOUS')
            for name,shape in schedule['input_shapes'].items()},
        memory_budget_bytes=schedule['effective_incremental_budget_bytes'],
        memory_scope=artifact.plan['memory_scope'],memory_policy='UNBOUND',
        cpu=dict(ids=list(binding['cpu_ids']),os='linux',architecture='x86_64',pointer_bits=64,
            byteorder='little',required_isa='BASELINE_X86_64'),
        floating_environment_policy='ORIGINAL_CPP_SAVE_RESTORE_FENV_STRICT_OPERAND_ORDER_PRIVATE_FAILURE_OUTPUT',
        Q=artifact.plan['Q'],source_proof=proof_reference,
        resource_schedule=dict(id=schedule['id'],receipt=schedule_receipt),
        original_f4_proof_sha256=semantic_sha256(asdict(proof)),
        profitability_status='UNPROVEN_NO_SOURCED_TIMING_MODEL',deployment_authorized=False,
        validation_execution_requires_trial=True)
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
        native_library_bytes_unchanged=True,performance_qualified=False,deployment_authorized=False,
        runtime_ir_or_codegen_required=False,bundle_bytes=len(raw)+len(admission_raw)+len(native_bytes))
    (out/'export-report.json').write_bytes(_json(result));return result


__all__=['export_temporal_invariant']
