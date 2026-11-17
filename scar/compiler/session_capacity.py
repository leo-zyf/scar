"""Source-qualified private capacity reuse with full fresh input snapshots."""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil

from scar.ir.execution_recipe import artifact_reference, read_artifact

SCHEMA='scar.session-capacity-preparation'
INPUTS=('session_plan','step_plan','workspace_plan','parent_native_metadata','session_native_metadata')


def _write(path,value):
    with path.open('x')as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    return artifact_reference(path,value['schema'])


def _pins():
    root=Path(__file__).resolve().parents[1]
    return {n:artifact_reference(root/n,'scar.python-source')for n in
        ('compiler/session_capacity.py','analysis/session_capacity_proof.py','optimization/session_capacity.py')}


def _derive(reference):
    from scar.compiler.session_outer import replay
    from scar.analysis.session_capacity_proof import derive_capacity_proof
    outer,spec,nested,check=replay(reference['uri'])
    if outer!=read_artifact(reference):raise ValueError('outer source preparation changed')
    session_preparation=read_artifact(check['preparation'])
    session_ref=session_preparation['session_plan'];session=read_artifact(session_ref)
    parent_ref=session['parent_native_manifest'];parent=read_artifact(parent_ref)
    root=Path(parent_ref['uri']).parent
    refs=dict(session_plan=session_ref,parent_native_metadata=parent_ref,session_native_metadata=check['native'])
    for name in ('step_plan','workspace_plan'):
        path=root/parent['files'][name];raw=path.read_bytes()
        if sha256(raw).hexdigest()!=parent['sha256'][name]:raise ValueError('original '+name+' bytes changed')
        document=json.loads(raw);refs[name]=artifact_reference(path,document['schema'])
    documents={key:read_artifact(refs[key])for key in INPUTS}
    sources=[dict(role=k,artifact=refs[k],locator='#')for k in INPUTS]
    proof=derive_capacity_proof(*(documents[k]for k in INPUTS),source_refs=sources)
    return outer,spec,nested,check,refs,documents,proof


def _scope(original,outer,proof,plans):
    result=deepcopy(original)
    supported=proof['status']=='PROVEN_CONDITIONAL' and bool(plans) and all(
        p['status']=='PROVEN_CONDITIONAL_CAPACITY_POLICY'for p in plans)
    if supported:
        result['unqualified_kernel_policy']='RETAIN_PRIVATE_CAPACITY_FULL_SNAPSHOT'
        result['capacity_policy']=dict(proof_id=proof['id'],admission='PRODUCER_PROVEN_PRIVATE_CAPACITY_ONLY',
            outer_boundary_ids=[read_artifact(r)['id']for r in outer['outer_profiles']],
            external_roots_retained=False,content_reuse=False)
        result['unknown_host_lookup_policy']='ORIGINAL_UNCHANGED_PRIVATE_CAPACITY_ONLY_DIRTY'
    return result


def _report(reference,inputs,proof_ref,plan_refs,scope_ref,proof,plans):
    return dict(schema=SCHEMA,schema_version=1,outer_preparation=reference,capacity_inputs=inputs,
        capacity_proof=proof_ref,capacity_plans=plan_refs,scope=scope_ref,implementations=_pins(),
        status='PROVEN_CONDITIONAL'if proof['status']=='PROVEN_CONDITIONAL'and plans and all(
            p['status']=='PROVEN_CONDITIONAL_CAPACITY_POLICY'for p in plans)else'FRONTIER',
        summary=dict(outer_scopes=len(plans),private_slots=len(proof.get('slots',())),
            private_payload_bytes=sum(s['capacity_bytes']for s in proof.get('slots',())),
            current_step_private_reads=proof.get('current_step_initialization',{}).get('private_read_count'),
            full_snapshots_per_step=len(proof.get('full_snapshot',{}).get('snapshots',())),
            content_reuse=False,external_roots_retained=False),
        source_caller_effects_closed=False,unknown_host_quiescence_assumed=False,
        native_math_changed=False,native_recompiled=False,target_program_executed=False,
        costs_known=False,physical_resource_feasibility_proven=False,
        default_backend='NO_OP',deployment_authorized=False)


def prepare(outer_preparation,out):
    from scar.optimization.session_capacity import derive_session_capacity
    reference=artifact_reference(Path(outer_preparation).resolve(),'scar.session-outer-preparation')
    output=Path(out).resolve()
    if output.exists():raise ValueError('fresh capacity preparation required')
    outer,spec,nested,_,refs,documents,proof=_derive(reference)
    output.mkdir(parents=True)
    proof_ref=_write(output/'proof.json',proof);plans=[];plan_refs=[]
    lifetime_ref=nested['session_lifetime'];lifetime=read_artifact(lifetime_ref)
    for i,(profile_ref,activation_ref)in enumerate(zip(outer['outer_profiles'],outer['activations'])):
        plan=derive_session_capacity(read_artifact(profile_ref),read_artifact(activation_ref),lifetime,proof,
            outer_profile_ref=profile_ref,activation_ref=activation_ref,lifetime_ref=lifetime_ref,
            capacity_proof_ref=proof_ref,capacity_inputs=documents)
        plans.append(plan);plan_refs.append(_write(output/f'capacity-{i}.json',plan))
    report=_report(reference,refs,proof_ref,plan_refs,_write(output/'scope.json',_scope(spec,outer,proof,plans)),proof,plans)
    _write(output/'result.json',report);return report


def replay(path):
    from scar.optimization.session_capacity import verify_session_capacity
    report=read_artifact(artifact_reference(Path(path).resolve(),SCHEMA))
    outer,spec,nested,check,refs,documents,proof=_derive(report['outer_preparation'])
    if refs!=report['capacity_inputs']or read_artifact(report['capacity_proof'])!=proof:
        raise ValueError('capacity initialization proof changed from original parents')
    if len(report['capacity_plans'])!=len(outer['outer_profiles']):raise ValueError('capacity scope count differs')
    plans=[];lifetime_ref=nested['session_lifetime'];lifetime=read_artifact(lifetime_ref)
    for p,a,c in zip(outer['outer_profiles'],outer['activations'],report['capacity_plans']):
        plan=read_artifact(c)
        verify_session_capacity(read_artifact(p),read_artifact(a),lifetime,proof,plan,
            outer_profile_ref=p,activation_ref=a,lifetime_ref=lifetime_ref,
            capacity_proof_ref=report['capacity_proof'],capacity_inputs=documents)
        plans.append(plan)
    expected=_report(report['outer_preparation'],refs,report['capacity_proof'],report['capacity_plans'],
        report['scope'],proof,plans)
    if report!=expected or read_artifact(report['scope'])!=_scope(spec,outer,proof,plans):
        raise ValueError('capacity policy/summary differs from full source replay')
    return report,read_artifact(report['scope']),nested,check


def export(preparation,out):
    from scar.runtime.session_scope_bundle import ADMISSION,implementation_sha256,inspect_scope_bundle
    report,spec,nested,_=replay(preparation)
    if report['status']!='PROVEN_CONDITIONAL':raise ValueError('capacity policy not applicable')
    output=Path(out).resolve()
    if output.exists():raise ValueError('fresh capacity runtime output required')
    native=read_artifact(nested['compiled_session']);shutil.copytree(native['bundle'],output)
    scope_ref=_write(output/'scope.json',spec)
    admission=dict(schema=ADMISSION,schema_version=1,scope_sha256=scope_ref['sha256'],
        runtime_implementation_sha256=implementation_sha256(),session_admission_sha256=native['expected_admission_sha256'],
        preparation=artifact_reference(Path(preparation).resolve(),SCHEMA),
        capacity_proof=report['capacity_proof'],content_retention_authorized=False)
    pin=_write(output/'scope-admission.json',admission)
    inspect_scope_bundle(output,expected_admission_sha256=pin['sha256'])
    result=dict(schema='scar.session-capacity-export',schema_version=1,bundle=str(output),
        preparation=admission['preparation'],expected_admission_sha256=pin['sha256'],
        private_capacity_retention=True,content_retention=False,native_recompiled=False,
        default_backend='NO_OP',deployment_authorized=False)
    _write(output/'scope-export.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--outer');group.add_argument('--export');group.add_argument('--check');parser.add_argument('--out')
    args=parser.parse_args(argv)
    if args.check:result=replay(args.check)[0]
    elif not args.out:parser.error('preparation/export requires --out')
    elif args.export:result=export(args.export,args.out)
    else:result=prepare(args.outer,args.out)
    print(json.dumps({k:result[k]for k in ('summary','status','bundle','expected_admission_sha256')if k in result},sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
