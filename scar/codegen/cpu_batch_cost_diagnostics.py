"""Read-only residual diagnostics for a frozen independent CPU-mask batch.

This reader checks retained artifact bytes and aggregates recorded predictions.
It neither admits cost facts nor executes a compiler, probe, or target program.
Three repeats describe observed scatter; they are not a confidence interval.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

from scar.ir.execution_recipe import artifact_reference, read_artifact
from .cpu_fused_cost import _receipt, _require, _save


def _repeat_diagnostics(samples, prediction):
    median = statistics.median(samples)
    low, high = min(samples), max(samples)
    return dict(repetitions=len(samples), min_ns=low, max_ns=high,
        relative_repeat_span=(high-low)/median,
        coefficient_of_variation=statistics.stdev(samples)/statistics.mean(samples)
            if len(samples)>1 else None,
        prediction_outside_observed_repeat_range=prediction<low or prediction>high,
        closest_repeat_relative_error=min(abs(prediction-v)/v for v in samples),
        observed_scatter_only_not_theoretical_or_statistical_bound=True)


def _quantiles(values):
    values=sorted(values)
    if not values:return None
    return dict(count=len(values), median=statistics.median(values), minimum=values[0],
        maximum=values[-1], p95_nearest_rank=values[math.ceil(.95*len(values))-1])


def _residual_summary(rows):
    return dict(rows=len(rows), rejected_rows=sum(r['relative_error']>.20 for r in rows),
        absolute_relative_error=_quantiles([r['relative_error'] for r in rows]),
        signed_relative_error=_quantiles([r['signed_relative_error'] for r in rows]),
        relative_repeat_span=_quantiles([r['repeat']['relative_repeat_span'] for r in rows]),
        repeat_cv=_quantiles([r['repeat']['coefficient_of_variation'] for r in rows]),
        rejected_prediction_outside_repeat_range=sum(r['relative_error']>.20 and
            r['repeat']['prediction_outside_observed_repeat_range'] for r in rows),
        rejected_closest_repeat_still_above_policy=sum(r['relative_error']>.20 and
            r['repeat']['closest_repeat_relative_error']>.20 for r in rows),
        rejected_repeat_span_below_5_percent=sum(r['relative_error']>.20 and
            r['repeat']['relative_repeat_span']<.05 for r in rows),
        residual_direction=dict(Counter('UNDERPREDICTED' if r['signed_relative_error']<0
            else 'OVERPREDICTED' for r in rows)))


def diagnose_cpu_domain_batch(result_path,out):
    result_ref=artifact_reference(result_path,'scar.cpu-domain-batch-result')
    result=read_artifact(result_ref);design=read_artifact(result['design'])
    domain=read_artifact(design['domain']);inventory=read_artifact(domain['inventory'])
    processors={r['cpu_id']:r for r in inventory['cpu_topology']['processors']}
    candidates={r['id']:r for r in domain['candidates']}
    out=Path(out).resolve();_require(not out.exists(),'fresh diagnostic directory required')
    out.mkdir(parents=True)
    all_rows=[];contexts=[];references=[];packs={};consumed_groups={}
    for record in result['results']:
        _require(record['exit_code']==0,'diagnostic requires retained successful child result')
        sample=read_artifact(record['sample_result']);model=read_artifact(sample['fused_model'])
        raw=read_artifact(model['raw_observations']);candidate=candidates[record['id']]
        context=read_artifact(record['context'])
        consumed_groups[record['id']]=sorted({term['parameters']['backing_storage_groups']
            for term in context['demand']['terms'] if term['component']=='FUSED_SCALAR_DAG_ROW_STAGE'})
        if model['pack']['sha256'] not in packs:
            packs[model['pack']['sha256']]=read_artifact(model['pack'])
        pack=packs[model['pack']['sha256']]
        cpu_ids=candidate['binding']['cpu_ids']
        core_ids={(processors[c]['package_id'],processors[c]['die_id'],processors[c]['core_id']) for c in cpu_ids}
        topology=dict(workers=len(cpu_ids),numa_node=candidate['numa_node'],cpu_ids=cpu_ids,
            physical_cores=len(core_ids),smt_sibling_workers=len(cpu_ids)-len(core_ids),
            smt='SMT_SIBLINGS_SELECTED' if len(cpu_ids)>len(core_ids) else 'ONE_THREAD_PER_CORE',
            representative_policies=candidate['representative_policies'])
        observations={r['id']:r for r in raw['observations']}
        for group,local in model['local_models'].items():
            for row in local['holdout']:
                observed=observations[row['configuration']];case=pack['registry'][observed['case_index']]
                _require(observed['holdout'] is True and observed['groups']==int(group)
                    and row['observed_ns']==statistics.median(observed['samples_ns']),
                    'frozen holdout raw/prediction mismatch')
                prediction=row['prediction'];memory=prediction['memory_proxy_ns'];mca=prediction['mca_execution_proxy_ns']
                predicted=prediction['duration_ns'];median=observed['median_ns']
                error=abs(predicted-median)/median
                _require(error==row['relative_error'],'recorded holdout error differs')
                all_rows.append(dict(candidate_id=record['id'],**topology,groups=int(group),
                    configuration=row['configuration'],case_id=case['id'],family=case['family'],
                    operand_pool=case['operand_pool'],array_loads=case['array_loads'],
                    rows=observed['rows'],cols=observed['cols'],observed_ns=median,
                    samples_ns=observed['samples_ns'],predicted_ns=predicted,relative_error=error,
                    signed_relative_error=(predicted-median)/median,
                    dominant_proxy='MEMORY' if memory>=mca else 'MCA',memory_proxy_ns=memory,
                    mca_execution_proxy_ns=mca,memory_to_mca_proxy_ratio=memory/mca,
                    stage_floor_ns=prediction.get('barrier_floor_ns',prediction.get('dispatch_ns')),
                    repeat=_repeat_diagnostics(observed['samples_ns'],predicted),
                    local_model_admitted=local['admitted']))
        context_rows=[r for r in all_rows if r['candidate_id']==record['id']]
        contexts.append(dict(candidate_id=record['id'],**topology,status=record['status'],
            complete=sample['complete'],consumed_backing_groups=consumed_groups[record['id']],
            model_admitted={g:m['admitted'] for g,m in model['local_models'].items()},
            maximum_holdout_errors=sample['holdout_maximum_errors'],
            maximum_repeat_span=max(r['repeat']['relative_repeat_span'] for r in context_rows),
            median_repeat_span=statistics.median(r['repeat']['relative_repeat_span'] for r in context_rows),
            total_worker_seconds=sample['total_worker_seconds'],
            observed_calibration_seconds=sample['observed_calibration_seconds'],
            peak_rss_kib=sample['peak_rss_kib'],unresolved=sample['unresolved'],
            fused_model=sample['fused_model'],raw=model['raw_observations'],host_load=raw['host_load']))
        references.append(dict(candidate_id=record['id'],sample=record['sample_result'],
            model=sample['fused_model'],raw=model['raw_observations'],frozen_design=model['frozen_design'],
            shared_pack=model['pack']))
    grouped={}
    for name,keys in dict(worker_count=('workers',),numa=('numa_node',),smt=('smt',),
        worker_numa_smt=('workers','numa_node','smt')).items():
        groups=defaultdict(list)
        for row in contexts:groups[tuple(row[k] for k in keys)].append(row)
        grouped[name]=[dict(group=dict(zip(keys,key)),count=len(rows),
            statuses=dict(Counter(r['status'] for r in rows)),
            maximum_error_groups_2=_quantiles([r['maximum_holdout_errors']['2'] for r in rows]))
            for key,rows in sorted(groups.items())]
    residual_groups={}
    for name,keys in dict(backing_groups=('groups',),dominant_proxy=('groups','dominant_proxy'),
        family=('groups','family'),worker_count=('groups','workers'),numa=('groups','numa_node'),
        smt=('groups','smt'),geometry=('groups','rows','cols'),
        topology_and_proxy=('groups','smt','dominant_proxy')).items():
        groups=defaultdict(list)
        for row in all_rows:groups[tuple(row[k] for k in keys)].append(row)
        residual_groups[name]=[dict(group=dict(zip(keys,key)),**_residual_summary(rows))
            for key,rows in sorted(groups.items())]
    target_rows=[r for r in all_rows if r['groups'] in consumed_groups[r['candidate_id']]]
    rejected=[r for r in target_rows if r['relative_error']>.20]
    stable=[r for r in rejected if r['repeat']['relative_repeat_span']<.05
        and r['repeat']['closest_repeat_relative_error']>.20]
    unknown=[r for r in contexts if r['status']=='UNKNOWN']
    unknown_stable=[r for r in unknown if any(h['candidate_id']==r['candidate_id'] for h in stable)]
    unknown_only_repetition_sensitive=[r for r in unknown if not any(h['candidate_id']==r['candidate_id']
        and h['repeat']['closest_repeat_relative_error']>.20 for h in rejected)]
    old_known=[dict(candidate_id=k,profile=v,status='PREVIOUS_QUALIFIED_NOT_RESAMPLED')
        for k,v in design['known_profiles'].items()]
    summary=dict(schema='scar.cpu-domain-batch-residual-diagnostics',schema_version=1,
        source=result_ref,design=result['design'],domain=design['domain'],inventory=domain['inventory'],
        implementation=_receipt(Path(__file__)),authority='RAW_RECORDED_RESIDUAL_DIAGNOSTICS_ONLY',
        declared_policy=dict(maximum_holdout_relative_error=.20,repetitions=3,warmup=1,batch=4),
        observed=dict(contexts=len(contexts),statuses=dict(Counter(r['status'] for r in contexts)),
            independent_holdout_records=len(all_rows),target_backing_group_records=len(target_rows),
            target_backing_group_rejected_records=len(rejected),
            consumed_backing_groups=sorted({g for values in consumed_groups.values() for g in values}),
            unknown_contexts_with_stable_rejected_record=len(unknown_stable),
            unknown_contexts_with_rejected_prediction_more_than_20_percent_from_every_repeat=
                len(unknown)-len(unknown_only_repetition_sensitive),
            unknown_contexts_without_any_rejected_prediction_more_than_20_percent_from_every_repeat=
                len(unknown_only_repetition_sensitive),
            stable_rejected_records=len(stable),
            maximum_child_peak_rss_kib=max(r['peak_rss_kib'] for r in contexts),
            controller_wall_seconds=result['actual_controller_wall_seconds'],
            summed_observed_calibration_seconds=sum(r['observed_calibration_seconds'] for r in contexts)),
        context_groups=grouped,residual_groups=residual_groups,
        previous_qualified_not_resampled=old_known,
        inferred=[
            'A prediction farther than20% from every retained repeat cannot be made admissible merely by choosing another one of those3 samples.',
            'Large repeatable residuals support insufficient model scope for those canonical cases;3 repeats do not establish hardware causality or confidence bounds.',
            'MEMORY/MCA dominance labels identify which declared proxy won max(), not a measured physical bottleneck.',
            'NUMA and SMT group differences are descriptive associations; allocation is UNBOUND and the host was not isolated.'
        ],
        not_established=['physical causes of residuals','arbitrary CPU-subset optimality',
            'wall-clock performance of any target variant','statistical confidence or theoretical bounds'],
        no_further_fitting_or_policy_change=True,no_probe_compiler_or_target_executed=True,
        cost_selection_authorized=False)
    _save(out/'contexts.json',dict(schema='scar.cpu-domain-batch-context-diagnostics',schema_version=1,contexts=contexts))
    _save(out/'residuals.json',dict(schema='scar.cpu-domain-batch-holdout-residuals',schema_version=1,rows=all_rows))
    _save(out/'largest-residuals.json',dict(schema='scar.cpu-domain-batch-largest-residuals',schema_version=1,
        rows=sorted(target_rows,key=lambda r:r['relative_error'],reverse=True)[:32],
        largest_stable_rejected_rows=sorted(stable,key=lambda r:r['relative_error'],reverse=True)[:8]))
    _save(out/'sources.json',dict(schema='scar.cpu-domain-batch-diagnostic-sources',schema_version=1,references=references))
    summary['outputs']={name:artifact_reference(out/(name+'.json'),schema) for name,schema in (
        ('contexts','scar.cpu-domain-batch-context-diagnostics'),('residuals','scar.cpu-domain-batch-holdout-residuals'),
        ('largest-residuals','scar.cpu-domain-batch-largest-residuals'),('sources','scar.cpu-domain-batch-diagnostic-sources'))}
    _save(out/'result.json',summary)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args(argv);result=diagnose_cpu_domain_batch(args.result,args.out)
    print(json.dumps(result['observed'],sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
