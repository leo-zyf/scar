"""Freeze and run independent CPU calibration contexts, one fresh child each."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scar.ir.execution_recipe import artifact_reference,read_artifact
from scar.codegen.cpu_fused_cost import _save,_require,_receipt
from scar.codegen.cpu_batch_instruction_cost import make_calibration_context,PACK

DESIGN='scar.cpu-domain-batch-design'


def prepare_cpu_domain_batch(domain,fused_packs,aux_packs,out,*,known_profiles=None):
    from scar.optimization.stateful_cpu_choices import _domain,_cost,stateful_cpu_cost_context
    document=_domain(domain);out=Path(out).resolve();_require(not out.exists(),'fresh batch design required')
    parent=artifact_reference(Path(document['base_stage_plan']['uri']).parent/'domain.json',document['schema'])
    known={};known_profiles=known_profiles or {}
    for cid,path in known_profiles.items():
        cost,issues,_=_cost(path,stateful_cpu_cost_context(document,cid))
        _require(cost is not None and not issues,'previous qualified exact-context profile rejected: '+cid)
        known[cid]=dict(profile=artifact_reference(path,'scar.stateful-cpu-cost-profile'),cost=cost)
    out.mkdir(parents=True);contexts=[]
    for row in document['candidates']:
        cid=row['id'];mode='serial' if row['worker_count']==1 else 'parallel'
        if cid in known:continue
        context=make_calibration_context(document,cid);path=out/'contexts'/f'{cid}.json';_save(path,context)
        contexts.append(dict(id=cid,mode=mode,worker_count=row['worker_count'],cpu_ids=row['binding']['cpu_ids'],
            context=artifact_reference(path,context['schema'])))
    def packs(values,schema):
        return {mode:value if isinstance(value,dict) else artifact_reference(value,schema) for mode,value in values.items()}
    fused=packs(fused_packs,PACK);aux=packs(aux_packs,'scar.cpu-batch-boundary-resource-pack')
    for mode in ('serial','parallel'):
        f=read_artifact(fused[mode]);a=read_artifact(aux[mode])
        _require(f['domain']==a['domain']==parent and f['mode']==a['mode']==mode,'pack/domain/mode contract differs')
    counts=dict(total=len(document['candidates']),serial=sum(r['worker_count']==1 for r in document['candidates']),
        parallel=sum(r['worker_count']>1 for r in document['candidates']),previously_qualified=len(known),remaining=len(contexts))
    value=dict(schema=DESIGN,schema_version=1,domain=parent,fused_packs=fused,aux_packs=aux,contexts=contexts,
        known_profiles=known,counts=counts,policy=dict(repetitions=3,warmup=1,batch=4,
            maximum_holdout_relative_error=.20,one_fresh_process_per_mask=True,sequential_sampling=True,
            old_mask_times_reused=False,retry_failed_contexts=False,target_variant_trials=False),
        controller=_receipt(Path(__file__).resolve()),worker=_receipt(Path(__file__).resolve().parents[1]/'codegen/cpu_batch_instruction_cost.py'),
        target_workload_executed=False,target_variants_ranked=False)
    _save(out/'design.json',value);return value


def run_cpu_domain_batch(design,out,*,candidate_ids=None,wall_budget_seconds=1200):
    from scar.optimization.stateful_cpu_choices import _domain
    reference=artifact_reference(design,DESIGN);document=read_artifact(reference)
    _require(_receipt(Path(document['controller']['path']))==document['controller']
        and _receipt(Path(document['worker']['path']))==document['worker'],'batch controller/worker source changed')
    _domain(document['domain']['uri'])
    selected=set(candidate_ids) if candidate_ids else None
    rows=[r for r in document['contexts'] if selected is None or r['id'] in selected]
    _require(selected is None or selected=={r['id'] for r in rows},'requested candidate is not in frozen remaining domain')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True);started=time.monotonic();results=[]
    for row in rows:
        directory=out/row['id'];record_path=directory/'controller.json'
        if record_path.exists():
            retained=json.loads(record_path.read_text());_require(retained['design']==reference and retained['context']==row['context'],
                'retained sample belongs to a different frozen design');results.append(retained);continue
        if time.monotonic()-started>=wall_budget_seconds:
            results.append(dict(id=row['id'],status='UNSAMPLED_BUDGET',design=reference,context=row['context']));continue
        directory.mkdir(parents=True,exist_ok=True);mode=row['mode'];sample=directory/'sample'
        command=[sys.executable,'-m','scar.codegen.cpu_batch_instruction_cost','worker',
            '--context',row['context']['uri'],'--fused-pack',document['fused_packs'][mode]['uri'],
            '--aux-pack',document['aux_packs'][mode]['uri'],'--out',str(sample)]
        begin=time.monotonic()
        try:
            p=subprocess.run(command,capture_output=True,text=True,timeout=min(120,max(1,wall_budget_seconds-(time.monotonic()-started))))
            stdout,stderr=p.stdout,p.stderr;exit_code=p.returncode
        except subprocess.TimeoutExpired as error:
            stdout=(error.stdout or b'').decode() if isinstance(error.stdout,bytes) else error.stdout or ''
            stderr=(error.stderr or b'').decode() if isinstance(error.stderr,bytes) else error.stderr or ''
            exit_code=None
        (directory/'stdout.txt').write_text(stdout);(directory/'stderr.txt').write_text(stderr)
        result_path=sample/'result.json';value=json.loads(result_path.read_text()) if result_path.exists() else None
        record=dict(schema='scar.cpu-domain-batch-controller-record',schema_version=1,
            id=row['id'],design=reference,context=row['context'],command=command,exit_code=exit_code,
            status='COMPLETE' if exit_code==0 and value and value['complete'] else 'UNKNOWN',
            process_wall_seconds=time.monotonic()-begin,stdout=_receipt(directory/'stdout.txt'),stderr=_receipt(directory/'stderr.txt'),
            sample_result=artifact_reference(result_path,'scar.cpu-batch-context-result') if value else None,
            cost_profile=value['cost_profile'] if value else None,
            raw_artifacts_retained=True,target_workload_executed=False,target_variants_ranked=False)
        _save(record_path,record);results.append(record)
        print(json.dumps(dict(completed=len(results),requested=len(rows),id=row['id'],status=record['status'],
            wall_seconds=round(record['process_wall_seconds'],3),peak_rss_kib=value['peak_rss_kib'] if value else None),sort_keys=True),flush=True)
    profiles={cid:r['profile']['uri'] for cid,r in document['known_profiles'].items()}
    profiles.update({r['id']:r['cost_profile']['uri'] for r in results if r.get('cost_profile')})
    report=dict(schema='scar.cpu-domain-batch-result',schema_version=1,design=reference,results=results,
        cost_profiles=profiles,new_complete=sum(r['status']=='COMPLETE' for r in results),requested=len(rows),
        remaining_native_contexts_unknown=document['counts']['total']-len(profiles),
        actual_controller_wall_seconds=time.monotonic()-started,
        target_workload_executed=False,target_variants_ranked=False,global_optimality_proven=False)
    path=out/'result.json';path.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    (out/'profile-map.json').write_text(json.dumps(profiles,sort_keys=True,indent=2)+'\n');return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('prepare');c.add_argument('--domain',required=True);c.add_argument('--serial-fused',required=True)
    c.add_argument('--parallel-fused',required=True);c.add_argument('--serial-aux',required=True);c.add_argument('--parallel-aux',required=True)
    c.add_argument('--known-profiles');c.add_argument('--out',required=True)
    c=sub.add_parser('run');c.add_argument('--design',required=True);c.add_argument('--out',required=True)
    c.add_argument('--candidate',action='append');c.add_argument('--wall-budget-seconds',type=int,default=1200);a=p.parse_args(argv)
    if a.command=='prepare':
        r=prepare_cpu_domain_batch(a.domain,{'serial':a.serial_fused,'parallel':a.parallel_fused},
            {'serial':a.serial_aux,'parallel':a.parallel_aux},a.out,
            known_profiles=json.loads(Path(a.known_profiles).read_text()) if a.known_profiles else None)
        print(json.dumps(r['counts'],sort_keys=True))
    else:run_cpu_domain_batch(a.design,a.out,candidate_ids=a.candidate,wall_budget_seconds=a.wall_budget_seconds)
    return 0


if __name__=='__main__':raise SystemExit(main())
