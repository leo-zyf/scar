"""Replay a C1 proof once and export compact live-guarded pooled-host bytes.

Original host and operator libraries are copied unchanged. This is not a new
backend, performance decision or automatic deployment. Runtime consumers read
only the compact contract, producer-pinned admission and identical ELF files.
"""
from hashlib import sha256
import json
from pathlib import Path


def _json(value):return(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def _receipt(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return dict(path=str(path),sha256=sha256(raw).hexdigest(),bytes=len(raw))
def _source(value):return {key:getattr(value,key)for key in('path','sha256','code_key','qualname')}


def export_caller_pipeline_runtime(native_directory,out,*,proof_reference,view,provider,proof=None):
    """Replay original CALL/leaf proofs before exporting the existing ABI host."""
    from scar.compiler.caller_pipeline import read_caller_pipeline_proof,document
    from scar.compiler.pipeline_model import pipeline_digest
    from scar.codegen.pipeline_loader import load
    from scar.ir.execution_recipe import read_artifact,artifact_reference
    from scar.runtime.caller_pipeline_guard import (SCHEMA,ADMISSION,TRUST,validate_contract,
        implementation_sha256,inspect_bundle)
    if isinstance(proof_reference,(str,Path)):
        proof_reference=artifact_reference(proof_reference,'scar.caller-matched-array-pipeline-proof')
    read_artifact(proof_reference)
    replay=read_caller_pipeline_proof(proof_reference['uri'],view=view,provider=provider)
    if proof is not None and document(proof)!=document(replay):raise ValueError('caller proof differs from original replay')
    proof=replay;native_directory=Path(native_directory).resolve()
    raw=json.loads((native_directory/'artifact.json').read_bytes())
    if raw.get('schema')!='scar.native-array-pipeline':
        raise ValueError('compact caller delivery currently requires the original pooled ABI1 host')
    artifact=load(native_directory);identity=artifact.report
    if identity['pipeline_sha256']!=pipeline_digest(proof.pipeline):
        raise ValueError('native host does not implement the replayed caller pipeline')
    out=Path(out).resolve()
    if out.exists():raise ValueError('fresh compact caller runtime bundle required')
    # Validate the whole supported footprint before publishing any bundle.
    operators=artifact._operators;cpus=set();targets=[]
    for operator in operators:
        binding=operator.plan.resource_binding
        if operator.timed_admission is not None or binding is None or not binding.cpu_ids:
            raise ValueError('explicit baseline CPU-bound ABI1 leaf required for compact caller export')
        if (operator.report['cxx_standard']!=20 or operator.report['operator_abi_version']!=1
                or any(str(option).startswith(('-march=','-mcpu=','-mavx'))for option in operator.report['compiler'])):
            raise ValueError('baseline C++20 leaf ISA/ABI contract required')
        cpus.update(binding.cpu_ids)
        target=_receipt(Path(binding.target_path)/'result.json')
        if target['sha256']!=binding.target_sha256:raise ValueError('original CPU target snapshot changed')
        if target not in targets:targets.append(target)
    libraries={};preserved=[]
    def retain(path,prefix):
        path=Path(path).resolve();raw=path.read_bytes();digest=sha256(raw).hexdigest()
        name=prefix+'-'+digest[:24]+'.so'
        if name in libraries and libraries[name]!=raw:raise ValueError('caller library content identity collision')
        libraries[name]=raw;preserved.append(dict(original=_receipt(path),bundle_member=name))
        return dict(library=name,sha256=digest,bytes=len(raw))
    host=retain(identity['library_path'],'host')
    native_operators=[]
    for call,operator in zip(proof.pipeline.calls,operators):
        row=retain(operator.library_path,'operator')
        row.update(plan_sha256=operator.report['plan_sha256'],input_count=len(call.plan.region.parameters))
        native_operators.append(row)
    memory=artifact.memory_plan;final=next(row for row in memory['values']if row['id']==proof.pipeline.output)
    callees={}
    for call in proof.pipeline.calls:
        name=call.callee.lookup_path[0];row=dict(global_name=name,source=_source(call.callee.source))
        if name in callees and callees[name]!=row:raise ValueError('global callee source contracts disagree')
        callees[name]=row
    array=lambda shape:dict(shape=list(shape),nbytes=shape[0]*shape[1]*8,dtype='float64',layout='C_CONTIGUOUS')
    contract=dict(schema=SCHEMA,schema_version=1,source=_source(proof.pipeline.source),
        parameters=list(proof.pipeline.parameters),callees=list(callees.values()),
        inputs={name:array(memory['input_shapes'][name])for name in proof.pipeline.parameters},output=array(final['shape']),
        native=dict(host=host,operators=native_operators,host_abi_version=1,operator_abi_version=1,
            pipeline_sha256=identity['pipeline_sha256'],memory_plan_sha256=identity['memory_plan_sha256']),
        cpu=dict(ids=sorted(cpus),os='linux',architecture='x86_64',pointer_bits=64,byteorder='little',required_isa='BASELINE_X86_64'),
        memory=dict(workspace_bytes=memory['total_workspace_bytes'],final_bytes=memory['final_bytes'],
            peak_bytes=memory['peak_allocation_bytes'],budget_bytes=memory['memory_budget_bytes'],
            scope=memory['budget_scope'],policy=memory['memory_policy'],budget_excludes=identity['budget_excludes']),
        defaults_policy='NO_DEFAULT_PARAMETERS',input_writes=False,fresh_output=True,
        original_proof=proof_reference,original_conditional_call_count=len(proof.call_evidence),
        runtime_guards=dict(source_and_code_ids='EACH_CALL',global_callee_identity_and_default_state='EACH_CALL',
            closures_and_custom_signature_metadata='EACH_CALL',global_and_local_monitoring='EACH_CALL',
            input_shape_dtype_rank_layout_alignment_and_backing_bounds='EACH_CALL',
            native_failure='WITHHOLD_PRIVATE_OUTPUT_CALL_ENTIRE_ORIGINAL_ONCE'),
        external_assumptions=['NO_FOREIGN_CONCURRENT_INPUT_WRITERS','SYNCHRONOUS_CPU_EXECUTION',
            'ALLOCATION_TIMING_AND_ALLOCATOR_CALLBACKS_NOT_OBSERVABLE_PROGRAM_OUTPUT',
            'FLOATING_STATUS_FLAGS_NOT_OBSERVABLE_PROGRAM_OUTPUT'],
        cost_status='UNKNOWN',automatic_deployment_authorized=False,
        execution_policy='NO_OP_UNLESS_EXPLICIT_CANDIDATE_MEASUREMENT')
    validate_contract(contract);runtime_raw=_json(contract)
    admission=dict(schema=ADMISSION,schema_version=1,trust_policy=TRUST,
        runtime_sha256=sha256(runtime_raw).hexdigest(),native=contract['native'],
        runtime_implementation_sha256=implementation_sha256(),
        producer=dict(name='SCAR',exporter_sha256=sha256(Path(__file__).read_bytes()).hexdigest()),
        original_proof=proof_reference,original_native_manifest=_receipt(native_directory/'artifact.json'),
        original_targets=targets,original_libraries=preserved,
        original_f4_and_matched_interface_replayed=True,native_bytes_identical=True,
        automatic_deployment_authorized=False,cost_status='UNKNOWN')
    out.mkdir(parents=True)
    for name,payload in libraries.items():(out/name).write_bytes(payload)
    (out/'runtime.json').write_bytes(runtime_raw);admission_raw=_json(admission)
    (out/'admission.json').write_bytes(admission_raw);pin=sha256(admission_raw).hexdigest()
    inspect_bundle(out,expected_admission_sha256=pin)
    report=dict(schema='scar.caller-pipeline-runtime-export',schema_version=1,bundle=str(out),
        expected_admission_sha256=pin,admission_sha256=pin,
        admission=artifact_reference(out/'admission.json',ADMISSION),
        runtime=artifact_reference(out/'runtime.json',SCHEMA),proof=proof_reference,
        pipeline_sha256=identity['pipeline_sha256'],memory_plan_sha256=identity['memory_plan_sha256'],
        original_native_manifest=_receipt(native_directory/'artifact.json'),native_bytes_identical=True,
        cpp_host_abi_version=1,operator_abi_version=1,live_guard_adapter_implemented=True,
        cost_status='UNKNOWN',automatic_deployment_authorized=False,default_backend='NO_OP',
        target_program_executed=False,numerical_backend_added=False)
    (out/'export-report.json').write_bytes(_json(report))
    return report


__all__=['export_caller_pipeline_runtime']
