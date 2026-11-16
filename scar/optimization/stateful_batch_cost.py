"""Replay independently sampled CPU contexts before finite-domain planning.

This is a build-time consumer, not another cost model. It reconstructs the
component sum only from the original auxiliary and instruction replay facts.
Failed or unsampled contexts remain in the domain with unknown cost.
"""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import time

from scar.ir.execution_recipe import artifact_reference,read_artifact
from scar.optimization.deployment_cost import _fact
from scar.optimization.resource_solver import semantic_sha256


def _need(value,message):
    if not value:raise ValueError(message)


def _raw(receipt):
    payload=Path(receipt['path']).read_bytes()
    _need(len(payload)==receipt['bytes'] and sha256(payload).hexdigest()==receipt['sha256'],
        'batch controller or raw output receipt changed')


def verify_batch_cpu_cost(domain,candidate_id,worker_result,*,replay_session=None):
    from scar.codegen.cpu_batch_instruction_cost import make_calibration_context
    from scar.codegen.cpu_batch_instruction_replay import verify_batch_projection
    from scar.codegen.cpu_batch_boundary_resource_replay import verify_aux_sample
    from scar.optimization.stateful_cpu_choices import _cost,stateful_cpu_cost_context
    from scar.optimization.stateful_cpu_model import stateful_cpu_demand,component_key

    reference=artifact_reference(worker_result,'scar.cpu-batch-context-result')
    result=read_artifact(reference);context=read_artifact(result['context'])
    _need(context==make_calibration_context(domain,candidate_id)
        and result['candidate_id']==candidate_id and result['complete'] is True
        and result['fresh_process_per_mask'] is True
        and result['owned_static_jobs_during_timing'] is False
        and result['target_workload_executed'] is False and result['target_variants_ranked'] is False,
        'complete independently sampled exact CPU context required')
    auxiliary=verify_aux_sample(result['aux_pack'],context,Path(worker_result).parent/'aux')
    _need(all(result['auxiliary'][key]==auxiliary[key] for key in
        ('resource_calibration','boundary_calibration','runtime_witness')),
        'worker auxiliary references differ from independently replayed sources')
    keywords={} if replay_session is None else dict(session=replay_session)
    fused=verify_batch_projection(domain,candidate_id,result['fused_projection']['uri'],**keywords)
    _need(fused['projection']==result['fused_projection'] and fused['source_profile']==result['fused_model'],
        'worker instruction projection differs from independently replayed sources')
    records={}
    for row in (*auxiliary['components'],*fused['components']):
        value,_=_fact(row['fact']);key=component_key(value['component'],value['parameters'])
        _need(key==row['key'] and key not in records,'duplicate or changed admitted component signature')
        _need(type(value['median_ns']) in (int,float) and math.isfinite(value['median_ns'])
            and value['median_ns']>=0,'finite nonnegative admitted component estimate required')
        records[key]=(row['fact'],value['median_ns'])
    demand=stateful_cpu_demand(domain,candidate_id);refs=[];total=0
    for term in demand['terms']:
        _need(term['key'] in records,'complete demand lacks an independently replayed component')
        fact,median=records[term['key']];refs.append(fact);total+=median*term['count']
    expected=dict(duration_ticks=math.ceil(total),tick_seconds=1e-9)
    profile=read_artifact(result['cost_profile']);context=stateful_cpu_cost_context(domain,candidate_id)
    cost,issues,_=_cost(profile,context)
    _need(cost is not None and not issues,'complete whole-call profile rejected: '+repr(issues))
    facts_refs={semantic_sha256(row['artifact']):row['artifact'] for row in
        (*profile['context_evidence'],*profile['estimate']['evidence'],*profile['derivation']['evidence'])}
    _need(len(facts_refs)==1,'one original context-bound complete-call facts artifact required')
    facts=read_artifact(next(iter(facts_refs.values())))
    _need(facts['schema']=='scar.stateful-cpu-model-facts' and facts['schema_version']==1
        and facts['context']==context and facts['estimate']==expected
        and all(profile['estimate'][k]==v for k,v in expected.items())
        and facts['derivation']['component_refs']==refs
        and facts['derivation']['target_variant_trials'] is False,
        'complete-call estimate or ordered component provenance differs')
    model=read_artifact(result['complete_model'])
    _need(model['complete'] is True and model['unresolved']==[] and model['context']==context
        and model['duration_ticks']==result['duration_ticks']==expected['duration_ticks']
        and model['profile']==result['cost_profile']['uri'],
        'worker completion differs from replayed complete-call facts')
    return dict(candidate_id=candidate_id,worker_result=reference,profile=result['cost_profile'],
        estimate=expected,component_terms=len(refs),auxiliary=auxiliary,
        instruction={k:v for k,v in fused.items() if k!='components'},
        target_workload_executed=False,target_variants_ranked=False)


def plan_cpu_domain_batch(batch_result,no_op_profile,out):
    from scar.optimization.stateful_cpu_choices import _domain,_cost,stateful_cpu_cost_context,plan_stateful_cpu_choices
    from scar.codegen.cpu_batch_instruction_cost import make_calibration_context
    from scar.codegen.cpu_batch_instruction_replay import BatchReplaySession
    from scar.optimization.stateful_cost_rebase import _model_dependencies

    started=time.perf_counter();out=Path(out).resolve()
    _need(not out.exists(),'fresh independently admitted batch plan directory required')
    reference=artifact_reference(batch_result,'scar.cpu-domain-batch-result');batch=read_artifact(reference)
    design=read_artifact(batch['design']);domain=_domain(design['domain']['uri'])
    _need(batch['schema_version']==design['schema_version']==1
        and batch['target_workload_executed'] is False and batch['target_variants_ranked'] is False,
        'independent CPU batch scope required')
    for key in ('controller','worker'):_raw(design[key])
    contexts={row['id']:row for row in design['contexts']}
    _need(len(contexts)==len(design['contexts']),'duplicate frozen CPU context')
    for identifier,row in contexts.items():
        _need(read_artifact(row['context'])==make_calibration_context(domain,identifier),
            'frozen calibration context differs from original domain')
    profiles={};replayed=[];retained=[];unknown=[];seen=set();session=BatchReplaySession(domain)
    for identifier,row in design['known_profiles'].items():
        try:
            profile=read_artifact(row['profile']);context=stateful_cpu_cost_context(domain,identifier)
            cost,issues,_=_cost(row['profile']['uri'],context)
            _need(cost is not None and not issues and cost==row['cost'],'previous exact-context cost changed')
            # The profile reader checks the complete-call document. Its
            # original primitive/holdout/assembly dependencies still need
            # their own replay; a retained profile is not an exemption.
            dependencies=_model_dependencies(profile,domain,design['domain'],context,identifier)
            retained.append(dict(candidate_id=identifier,profile=row['profile'],dependencies=dependencies))
            profiles[identifier]=row['profile']['uri']
        except (OSError,ValueError,TypeError,KeyError,IndexError) as error:
            unknown.append(dict(candidate_id=identifier,reason='RETAINED_COST_REPLAY: '+str(error)))
    for row in batch['results']:
        identifier=row['id']
        _need(identifier in contexts and identifier not in seen and row['design']==batch['design']
            and row['context']==contexts[identifier]['context'],'batch row has a different or duplicate context')
        seen.add(identifier)
        if row['status']!='COMPLETE':
            unknown.append(dict(candidate_id=identifier,reason=row['status']));continue
        for key in ('stdout','stderr'):_raw(row[key])
        try:
            _need(row['exit_code']==0,'calibration child did not complete')
            read_artifact(row['sample_result'])
            replay=verify_batch_cpu_cost(domain,identifier,row['sample_result']['uri'],replay_session=session)
            _need(replay['worker_result']==row['sample_result'],'controller worker receipt differs')
            _need(replay['profile']==row['cost_profile'],'controller cost profile differs from replayed child')
            profiles[identifier]=replay['profile']['uri'];replayed.append(replay)
        except (OSError,ValueError,TypeError,KeyError,IndexError) as error:
            unknown.append(dict(candidate_id=identifier,reason=str(error)))
    unknown.extend(dict(candidate_id=identifier,reason='UNSAMPLED') for identifier in contexts.keys()-seen)
    session.check();read_artifact(reference);read_artifact(batch['design'])
    out.mkdir(parents=True)
    choice=plan_stateful_cpu_choices(domain,cost_profiles=profiles,no_op_profile=no_op_profile,out=out/'choice')
    report=dict(schema='scar.admitted-cpu-domain-batch-plan',schema_version=1,batch_result=reference,
        domain=design['domain'],qualified_previous_profiles=len(retained),retained_profile_replays=retained,
        independently_replayed_new_profiles=replayed,unknown_contexts=unknown,
        cost_profiles=profiles,choice=artifact_reference(out/'choice/choice.json',choice['schema']),
        decision=choice['decision'],complete_cost_domain=choice['complete_cost_domain'],
        selected_id=choice['selected_id'],seconds=time.perf_counter()-started,
        target_workload_executed=False,target_variants_ranked=False,physical_optimality_proven=False,
        deployment_authorized=False)
    (out/'result.json').write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-result',required=True);parser.add_argument('--no-op-profile',required=True)
    parser.add_argument('--out',required=True);args=parser.parse_args(argv)
    report=plan_cpu_domain_batch(args.batch_result,args.no_op_profile,args.out)
    print(json.dumps(dict(decision=report['decision'],selected_id=report['selected_id'],
        replayed=len(report['independently_replayed_new_profiles']),unknown=len(report['unknown_contexts']))));return 0


if __name__=='__main__':raise SystemExit(main())
