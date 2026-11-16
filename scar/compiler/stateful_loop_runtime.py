"""Replay original loop/step/plan, then seal identical stateful/repeat ELF files.

This separate compact trial profile carries no fabricated measured selection.
The original kernel and repeat control adapter retain every per-step commit.
"""
from hashlib import sha256
import json
from pathlib import Path


def _json(value):return(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def _source(value):return{key:getattr(value,key)for key in('path','sha256','code_key','qualname')}
def _receipt(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return dict(path=str(path),sha256=sha256(raw).hexdigest(),bytes=len(raw))


def export_stateful_loop_runtime(base_native_directory,repeat_directory,out,*,proof_reference,
        plan,view,provider=None,base_bundle=None):
    from scar.compiler.stateful_loop_frontend import read_stateful_loop_proof,SCHEMA as PROOF
    from scar.compiler.stateful_model import stateful_digest
    from scar.compiler.stateful_repeat import load_repeat
    from scar.compiler.export_stateful import project_stateful_contract
    from scar.codegen.stateful import load as load_base
    from scar.optimization.stateful_loop_schedule import verify_stateful_loop
    from scar.optimization.resource_solver import semantic_sha256
    from scar.ir.execution_recipe import artifact_reference,read_artifact
    from scar.runtime.native_stateful_repeat import (SCHEMA,ADMISSION,PROFILE,FAMILY,TRUST,
        validate_contract,implementation_sha256,load)
    if provider is None:raise ValueError('original caller-matched source-call provider required')
    if base_bundle is not None:raise ValueError('fresh loop trial projects its original base; no measured bundle substitution')
    if isinstance(proof_reference,(str,Path)):proof_reference=artifact_reference(proof_reference,PROOF)
    read_artifact(proof_reference)
    proof=read_stateful_loop_proof(proof_reference['uri'],view=view,provider=provider)
    if isinstance(plan,(str,Path)):plan=artifact_reference(plan,'scar.stateful-loop-plan')
    if type(plan)is not dict or plan.get('schema')!='scar.stateful-loop-plan'or'uri'not in plan:
        raise ValueError('original published loop-plan ArtifactRef or path required')
    plan_reference=plan;plan=read_artifact(plan_reference)
    verify_stateful_loop(proof,Path(plan['target']['uri']).parent,plan)
    base=load_base(base_native_directory);repeat=load_repeat(repeat_directory)
    if(stateful_digest(base.region)!=stateful_digest(proof.step)
            or semantic_sha256(base.plan)!=semantic_sha256(plan['step_plan'])
            or any(repeat.base.report[key]!=base.report[key]for key in
                ('build_id','region_sha256','plan_sha256','contract_sha256'))
            or repeat.metadata['identity']['base_artifact']!=_receipt(base.directory/'artifact.json')):
        raise ValueError('original loop step, plan, base artifact and repeat binding differ')
    out=Path(out).resolve()
    if out.exists()or out.is_relative_to(base.directory)or out.is_relative_to(repeat.directory):
        raise ValueError('fresh independent stateful-loop compact directory required')
    step=project_stateful_contract(base,selection=None,require_selection=False)
    def library(path):
        path=Path(path);raw=path.read_bytes()
        return dict(library=path.name,sha256=sha256(raw).hexdigest(),bytes=len(raw)),raw
    base_row,base_bytes=library(base.directory/base.report['files']['library'])
    repeat_row,repeat_bytes=library(repeat.directory/repeat.metadata['files']['library'])
    if base_row['library']==repeat_row['library']:raise ValueError('base/repeat library names collide')
    contract=dict(schema=SCHEMA,schema_version=1,profile=PROFILE,execution_family=FAMILY,
        source=_source(proof.source),step_source=_source(proof.step.source),
        parameters=list(proof.parameters),parameter_map=dict(proof.parameter_map),
        trip_count_parameter=proof.trip_count_parameter,trip_count=plan['trip_count'],
        step_global_name=proof.step_global_name,
        call_arguments=dict(positional_parameters=list(proof.call_evidence['positional_parameters']),
            keyword_parameters=list(proof.call_evidence['keyword_parameters'].items())),
        step_contract=step,native=dict(stateful_abi_version=1,repeat_abi_version=1,
            symbol='scar_repeat_stateful_v1',base=base_row,repeat=repeat_row,base_ids=step['native']['ids']),
        original_loop_proof=proof_reference,original_loop_plan=plan_reference,
        original_step_reference=proof.step_reference,
        conditions=list(dict.fromkeys([*plan['required_live_guards'],*repeat.metadata['conditions']])),
        external_assumptions=plan['external_assumptions'],
        commit_policy='EACH_SUCCESSFUL_ORIGINAL_ABI_CALL_COMMITS_BEFORE_NEXT_STEP',
        failure_policy='RESUME_ONLY_UNCOMMITTED_ORIGINAL_SUFFIX_WITH_DYNAMIC_CALLEE_LOOKUP',
        source_loop_replacement_authority='ORIGINAL_F4_LOOP_AND_STEP_REPLAY_AT_EXPORT',
        fixed_input_binding_is_not_invariant_array_content=True,workspace_or_team_setup_hoisted=False,
        cost_status='UNKNOWN',trial_only=True,deployment_authorized=False)
    validate_contract(contract)
    sources={source.path:source.sha256 for source in(proof.source,proof.step.source,
        *(row.source for row in base.region.callees))}
    source_rows=[]
    for path,digest in sources.items():
        row=_receipt(path)
        if row['sha256']!=digest:raise ValueError('original source snapshot differs before loop export')
        source_rows.append(row)
    raw=_json(contract)
    admission=dict(schema=ADMISSION,schema_version=1,profile=PROFILE,trust_policy=TRUST,
        runtime_sha256=sha256(raw).hexdigest(),runtime_implementation_sha256=implementation_sha256(),
        native=contract['native'],producer=dict(name='SCAR',exporter_sha256=sha256(Path(__file__).read_bytes()).hexdigest()),
        original_loop_proof=proof_reference,original_loop_plan=plan_reference,
        original_base_manifest=_receipt(base.directory/'artifact.json'),
        original_repeat_manifest=_receipt(repeat.directory/'artifact.json'),source_refs=source_rows,
        original_f4_loop_and_step_replayed=True,native_math_unchanged=True,cost_status='UNKNOWN')
    admission_raw=_json(admission);pin=sha256(admission_raw).hexdigest()
    # Recheck original publications after complete source/backend admission.
    read_artifact(proof_reference);read_artifact(plan_reference)
    if(base_row!=library(base.directory/base.report['files']['library'])[0]
        or repeat_row!=library(repeat.directory/repeat.metadata['files']['library'])[0]):
        raise ValueError('original native bytes changed during loop export')
    out.mkdir(parents=True)
    (out/base_row['library']).write_bytes(base_bytes);(out/repeat_row['library']).write_bytes(repeat_bytes)
    (out/'runtime.json').write_bytes(raw);(out/'admission.json').write_bytes(admission_raw)
    load(out,expected_admission_sha256=pin)
    result=dict(schema='scar.stateful-loop-runtime-export',schema_version=1,bundle=str(out),
        expected_admission_sha256=pin,admission=artifact_reference(out/'admission.json',ADMISSION),
        runtime=artifact_reference(out/'runtime.json',SCHEMA),proof=proof_reference,plan=plan_reference,
        native_bytes_identical=True,original_f4_loop_and_step_replayed=True,
        per_step_commit_boundary_preserved=True,completed_cursor_and_original_suffix=True,
        workspace_or_team_setup_hoisted=False,numerical_backend_added=False,
        cost_status='UNKNOWN',trial_only=True,default_backend='NO_OP',deployment_authorized=False,
        target_program_executed=False,performance_measured=False)
    (out/'export-report.json').write_bytes(_json(result));return result


__all__=['export_stateful_loop_runtime']
