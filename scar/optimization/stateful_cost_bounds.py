"""Nonnegative relaxations of an already declared ordered component-sum model.

This consumes a prior independent admission report as an explicit authority.
It does not repeat its compiler/MCA admission, fit costs, or authorize deployment.
For unresolved candidates only auxiliary facts replayed by verify_aux_sample are
added. An absent required term stays [0,+infinity], never a zero-duration profile.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference,read_artifact
from .deployment_cost import _fact
from .resource_solver import semantic_sha256


def _need(value,message):
    if not value:raise ValueError(message)


def _receipt(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return dict(path=str(path),sha256=sha256(raw).hexdigest(),bytes=len(raw))


def _write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    return artifact_reference(path,value['schema'])


def _relaxation(terms,known,incumbent_ticks):
    """Ordered IEEE-nonnegative sum matches the existing model's arithmetic.

    Addition/multiplication by a positive occurrence count and ceiling are
    monotone for finite nonnegative values. The partial ordered sum is therefore
    a lower relaxation even though it is not a physical runtime lower bound.
    """
    total=0;components=[];unknown=[]
    for term in terms:
        count=term['count'];_need(type(count)is int and count>0,'positive required occurrence count needed')
        record=known.get(term['key'])
        if record is None:
            unknown.append(term['id'])
            components.append(dict(term=term['id'],key=term['key'],component=term['component'],count=count,
                interval_ns=dict(lower=0,upper=None,upper_kind='POSITIVE_INFINITY'),
                authority='NONNEGATIVITY_RELAXATION_NOT_A_DURATION_ESTIMATE',fact=None))
        else:
            median=record['median_ns'];_need(type(median)in(int,float) and math.isfinite(median) and median>=0,
                'finite nonnegative verified component required')
            total+=median*count;_need(math.isfinite(total),'component sum overflow')
            components.append(dict(term=term['id'],key=term['key'],component=term['component'],count=count,
                interval_ns=dict(lower=median,upper=median,upper_kind='FINITE_MODEL_POINT'),
                authority='REPLAYED_AUXILIARY_MODEL_COEFFICIENT',basis=record['basis'],fact=record['fact']))
    lower=math.ceil(total)
    return dict(interval_ticks=dict(lower=lower,upper=None if unknown else lower,
            upper_kind='POSITIVE_INFINITY' if unknown else 'FINITE_MODEL_POINT'),
        ordered_known_component_sum_ns=total,components=components,unknown_terms=unknown,
        strictly_dominated=lower>incumbent_ticks,
        dominance_margin_ticks=lower-incumbent_ticks if lower>incumbent_ticks else None,
        duration_profile_emitted=False)


def _comparison_context(context):
    return {k:v for k,v in context.items() if k not in ('candidate_id','cpu_binding','backend')}


def derive_stateful_cost_bounds(admission_report,out):
    from scar.optimization import stateful_batch_cost,stateful_cpu_choices,stateful_cpu_model
    from scar.codegen import cpu_batch_boundary_resource,cpu_batch_boundary_resource_replay
    from scar.codegen.cpu_batch_instruction_cost import make_calibration_context
    from scar.codegen.cpu_batch_boundary_resource_replay import verify_aux_sample
    from scar.optimization.stateful_cpu_choices import _domain,_cost,stateful_cpu_cost_context
    from scar.optimization.stateful_cpu_model import stateful_cpu_demand,component_key

    out=Path(out).resolve();_need(not out.exists(),'fresh stateful cost-bound output required')
    admission_ref=artifact_reference(admission_report,'scar.admitted-cpu-domain-batch-plan')
    admission=read_artifact(admission_ref);choice=read_artifact(admission['choice'])
    _need(admission['schema_version']==1 and admission['target_workload_executed']is False
        and admission['target_variants_ranked']is False and admission['deployment_authorized']is False,
        'prior independent nondeployment admission authority required')
    domain=_domain(admission['domain']['uri'])
    _need(choice['schema']=='scar.stateful-cpu-choice' and choice['schema_version']==1
        and choice['domain']==admission['domain'] and choice['tick_seconds']==1e-9,
        'same original single-call domain and nanosecond ticks required')
    candidates={r['id']:r for r in domain['candidates']};identifiers=[*candidates,'NO_OP']
    variants={r['id']:r for r in choice['variants']}
    options={r['id']:r for r in choice['solver']['option_domain']['stateful-call']}
    _need(len(variants)==len(choice['variants']) and set(variants)==set(identifiers)==set(options),
        'all original native alternatives and NO_OP must remain present')
    _need(len({r['contract_sha256'] for r in options.values()})==1,'single same-contract call choice required')
    prior_rows={r['candidate_id']:r for r in (*admission['retained_profile_replays'],
        *admission['independently_replayed_new_profiles'])}
    _need(len(prior_rows)==len(admission['retained_profile_replays'])+
        len(admission['independently_replayed_new_profiles']) and set(prior_rows)==set(admission['cost_profiles']),
        'prior independently admitted complete native set differs')
    known={};scope=None;option_refs={};prior_verifier_refs={}
    for identifier in identifiers:
        context=stateful_cpu_cost_context(domain,identifier)
        if scope is None:scope=_comparison_context(context)
        _need(scope==_comparison_context(context) and variants[identifier]['context']==context,
            'cross-scope/source/hardware/budget comparison is forbidden')
        _need(context['scope']['call_scope']=='ONE_ORIGINAL_METHOD_CALL_SITE'
            and context['scope']['boundary']=='WARM_SESSION' and context['scope']['reuse_count']==1,
            'only one guarded warm method call is supported')
        option=options[identifier]
        _need(len(option['cost_evidence'])==1,'one original option cost fact required')
        evidence=option['cost_evidence'][0];duration,_=_fact(evidence)
        _need(duration==option['duration_ticks'],'prior option value differs from original FactRef')
        option_refs[identifier]=evidence
        if identifier in prior_rows:
            prior=prior_rows[identifier];profile_ref=prior['profile'];profile=read_artifact(profile_ref)
            _need(Path(admission['cost_profiles'][identifier]).resolve()==Path(profile_ref['uri']).resolve()
                and variants[identifier]['profile']==profile,'prior admitted profile provenance differs')
            cost,issues,_=_cost(profile,context)
            _need(cost is not None and not issues and cost['duration_ticks']==duration
                and cost['tick_seconds']==1e-9 and variants[identifier]['cost_status']=='KNOWN',
                'previously admitted complete profile/context changed')
            if 'estimate' in prior:
                _need(prior['estimate']==dict(duration_ticks=duration,tick_seconds=1e-9),
                    'prior independent replay estimate differs')
                verifier=prior['auxiliary']['verifier_implementation'];read_artifact(verifier,json_document=False)
                prior_verifier_refs[verifier['sha256']]=verifier
            known[identifier]=dict(duration_ticks=duration,profile=profile_ref,prior_row=prior)
        elif identifier=='NO_OP':
            cost,issues,_=_cost(choice['no_op_profile'],context)
            _need(cost is not None and not issues and cost['duration_ticks']==duration
                and cost['tick_seconds']==1e-9 and variants[identifier]['cost_status']=='KNOWN',
                'original NO_OP scope/profile differs from prior choice')
            known[identifier]=dict(duration_ticks=duration,profile=None,prior_row=None)
        else:
            _need(duration is None and variants[identifier]['cost_status']=='UNRESOLVED',
                'unadmitted numeric option cannot serve as incumbent or complete bound')
    _need(known,'at least one previously independently admitted complete option required')
    incumbent_id=min(known,key=lambda key:(known[key]['duration_ticks'],key))
    incumbent=known[incumbent_id];incumbent_ticks=incumbent['duration_ticks']
    frozen_sources={str(Path(module.__file__).resolve()):_receipt(module.__file__) for module in
        (stateful_batch_cost,stateful_cpu_choices,stateful_cpu_model,
         cpu_batch_boundary_resource,cpu_batch_boundary_resource_replay)}
    batch=read_artifact(admission['batch_result']);design=read_artifact(batch['design'])
    _need(design['domain']==admission['domain'],'batch source domain differs from prior admission')
    batch_rows={r['id']:r for r in batch['results']};_need(len(batch_rows)==len(batch['results']),'duplicate raw context')
    rows=[];source_failures=[];auxiliary_replays=[]
    for identifier in identifiers:
        context=stateful_cpu_cost_context(domain,identifier)
        if identifier in known:
            ticks=known[identifier]['duration_ticks']
            row=dict(schema='scar.stateful-cpu-cost-interval',schema_version=1,candidate_id=identifier,
                status='PRIOR_INDEPENDENTLY_ADMITTED_COMPLETE_COST',
                interval_ticks=dict(lower=ticks,upper=ticks,upper_kind='FINITE_MODEL_POINT'),
                source_profile=known[identifier]['profile'],source_option_fact=option_refs[identifier],
                unknown_terms=[],strictly_dominated=ticks>incumbent_ticks,
                dominance_margin_ticks=ticks-incumbent_ticks if ticks>incumbent_ticks else None,
                duration_profile_emitted=False)
        else:
            raw=batch_rows.get(identifier);_need(raw is not None,'unresolved raw CPU context missing')
            _need(raw['status']=='UNKNOWN' and raw['exit_code']==0 and raw['design']==batch['design'],
                'partial lower bound only supports retained normal independent child results')
            sample=read_artifact(raw['sample_result']);frozen_context=read_artifact(sample['context'])
            _need(sample['candidate_id']==identifier and raw['context']==sample['context']
                and frozen_context==make_calibration_context(domain,identifier)
                and sample['complete']is False,'original unresolved domain demand differs')
            demand=stateful_cpu_demand(domain,identifier)
            _need(demand==frozen_context['demand'] and demand['context']==context
                and demand['scope']=='ONE_ORIGINAL_METHOD_CALL_SITE','original ordered single-call demand differs')
            records={};auxiliary=None;failure=None
            try:
                auxiliary=verify_aux_sample(sample['aux_pack'],frozen_context,Path(raw['sample_result']['uri']).parent/'aux')
                _need(all(auxiliary[key]==sample['auxiliary'][key] for key in
                    ('resource_calibration','boundary_calibration','runtime_witness')),
                    'checked auxiliary refs differ from original child')
                for verified in auxiliary['components']:
                    component,_=_fact(verified['fact']);key=component_key(component['component'],component['parameters'])
                    _need(key==verified['key'] and component['component']==verified['component']
                        and component['parameters']==verified['parameters'] and key not in records,
                        'checked component key/parameters duplicate or changed')
                    _need(component['basis']in('DIRECT_COMPONENT_TIMING','DECLARED_COMPONENT_MODEL')
                        and component.get('selectable')is not False and component.get('precision_admitted')is not False
                        and component.get('exportable_as_complete_cost')is not False,
                        'nonadmitted auxiliary coefficient cannot enter the relaxation')
                    records[key]=dict(median_ns=component['median_ns'],basis=component['basis'],fact=verified['fact'])
                _need(set(records)<=set(t['key']for t in demand['terms']),
                    'auxiliary fact is outside this original same-context demand')
            except (OSError,ValueError,TypeError,KeyError,IndexError) as error:
                records={};failure=str(error);source_failures.append(dict(candidate_id=identifier,reason=failure))
            relaxation=_relaxation(demand['terms'],records,incumbent_ticks)
            row=dict(schema='scar.stateful-cpu-cost-interval',schema_version=1,candidate_id=identifier,
                status='VERIFIED_AUXILIARY_PARTIAL_RELAXATION' if failure is None else 'AUXILIARY_REPLAY_FAILED_UNBOUNDED',
                source_context=sample['context'],source_worker_result=raw['sample_result'],
                demand_fact=dict(artifact=sample['context'],locator='#/demand'),
                auxiliary_replay=None if auxiliary is None else {k:v for k,v in auxiliary.items()if k not in('components','runtime')},
                source_failure=failure,**relaxation)
            if failure is None:
                auxiliary_replays.append(dict(candidate_id=identifier,
                    verifier=auxiliary['verifier_implementation'],component_count=len(auxiliary['components'])))
        row.update(tick_seconds=1e-9,comparison_scope_sha256=semantic_sha256(scope),
            source_option_fact=option_refs[identifier],incumbent_id=incumbent_id,
            dominance_authority='ORDERED_NONNEGATIVE_COMPONENT_SUM_MODEL_ONLY_NOT_PHYSICAL_RUNTIME_BOUND',
            proof=None if not row['strictly_dominated'] else dict(
                rule='LOWER_ENDPOINT_STRICTLY_EXCEEDS_PREVIOUS_ADMITTED_UPPER_ENDPOINT',
                candidate_lower_ticks=row['interval_ticks']['lower'],incumbent_upper_ticks=incumbent_ticks,
                incumbent_option_fact=option_refs[incumbent_id],all_omitted_required_terms_nonnegative=True,
                overlapping_tasks_summed=False),
            deployment_authorized=False,physical_optimality_proven=False)
        reference=_write(out/'candidates'/(identifier+'.json'),row)
        rows.append(dict(candidate_id=identifier,interval=row['interval_ticks'],status=row['status'],
            strictly_dominated=row['strictly_dominated'],unknown_terms=len(row['unknown_terms']),artifact=reference))
    for receipt in frozen_sources.values():_need(_receipt(receipt['path'])==receipt,'bound-consumer dependency changed during readback')
    for ref in (admission_ref,admission['choice'],admission['domain'],admission['batch_result'],batch['design']):read_artifact(ref)
    unknown=[r for r in rows if r['unknown_terms']];pruned=[r for r in unknown if r['strictly_dominated']]
    report=dict(schema='scar.stateful-cpu-cost-bounds',schema_version=1,
        previous_admission=admission_ref,original_choice=admission['choice'],domain=admission['domain'],
        batch_result=admission['batch_result'],tick_seconds=1e-9,comparison_scope=scope,
        prior_authority=dict(kind='EXPLICIT_PRIOR_INDEPENDENT_ADMISSION_REPORT',
            function='scar.optimization.stateful_batch_cost.plan_cpu_domain_batch',
            report=admission_ref,choice=admission['choice'],
            original_report_mtime_ns=Path(admission_report).stat().st_mtime_ns,
            mtime_is_execution_timestamp_proof=False,original_absolute_execution_time_recorded=False,
            scope=scope['scope'],available_original_verifier_pins=list(prior_verifier_refs.values()),
            current_consumer_source_pins=list(frozen_sources.values()),
            original_admission_consumer_source_pin_embedded=False,
            current_source_hash_does_not_prove_historical_execution=True,
            previous_complete_calibration_compiler_mca_admission_reexecuted=False),
        incumbent=dict(candidate_id=incumbent_id,duration_ticks=incumbent_ticks,
            original_option_fact=option_refs[incumbent_id],source_profile=incumbent['profile'],
            authority='EARLIER_INDEPENDENTLY_ADMITTED_COMPLETE_MODEL_NOT_THIS_READER_HASH'),
        candidates=rows,auxiliary_replays=auxiliary_replays,source_failures=source_failures,
        counts=dict(original_native_options=len(candidates),retained_no_op_options=1,total_options=len(rows),
            previously_complete_options=len(known),partial_unknown_options=len(unknown),
            partial_options_strictly_dominated=len(pruned),
            partial_options_not_excluded=len(unknown)-len(pruned)),
        source_statuses=dict(Counter(r['status']for r in rows)),
        model_premises=['ONE_GUARDED_WARM_METHOD_CALL_WITH_THE_ORIGINAL_EXACT_CONTEXT_AND_TICK',
            'ORIGINAL_STATEFUL_ORDERED_COMPONENT_SUM_MODEL_NOT_AN_ADDITIVITY_THEOREM',
            'EACH_AUXILIARY_COEFFICIENT_AND_OCCURRENCE_COUNT_IS_FIXED_IN_THIS_MODEL',
            'REQUIRED_UNKNOWN_COMPONENTS_HAVE_NONNEGATIVE_MODEL_EXTENSIONS',
            'ORIGINAL_TERM_ORDER_AND_NONNEGATIVE_FLOAT_ARITHMETIC_AND_CEILING_ARE_MONOTONE',
            'STAGE_COMPLETION_BARRIERS_ARE_ALREADY_IN_THEIR_ORIGINAL_COMPONENT_SCOPE',
            'NO_OVERLAPPING_TASK_COSTS_OR_OTHER_HARDWARE_CONTEXTS_ARE_COMPOSED'],
        unresolved_scope=['UNKNOWN_COMPONENT_UPPER_ENDPOINTS_REMAIN_INFINITE',
            'CANONICAL_MEASUREMENT_VARIANCE_IS_NOT_A_THEORETICAL_PHYSICAL_RUNTIME_INTERVAL',
            'PRIOR_ADMISSION_ATTESTATION_IS_CONSUMED_NOT_REEXECUTED_IN_THIS_OPERATION',
            'FULL_PROGRAM_STARTUP_DEPLOYMENT_AND_ALTERNATIVE_BACKENDS_NOT_OPTIMIZED'],
        model_optimum_proven=False,original_planner_decision=choice['decision'],decision='NO_OP',
        original_planner_modified=False,unknown_duration_replaced_with_zero=False,
        new_complete_profiles_emitted=False,independent_calibrations_remeasured=False,
        native_compilation_or_mca_analysis_executed=False,metadata_only_compiler_identity_queries_permitted=True,
        target_workload_executed=False,target_variants_ranked=False,
        deployment_authorized=False,physical_optimality_proven=False,
        implementation=_receipt(__file__))
    _write(out/'result.json',report);return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args(argv);result=derive_stateful_cost_bounds(args.admission,args.out)
    print(json.dumps(result['counts'],sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
