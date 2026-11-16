"""Replay original component facts, then connect intervals to finite choice.

No original planner or source is rewritten. This bridge supplies a model-level
choice packet for later existing backend lowering; unresolved alternatives keep
NO_OP. It never reruns calibration, a target variant, or compiler/MCA analysis.
"""
import argparse
import json
from pathlib import Path
import tempfile

from scar.ir.execution_recipe import artifact_reference,read_artifact
from scar.ir.program.providers import ProviderFactReference
from scar.ir.record_codec import decode
from scar.optimization import interval_choices
from scar.optimization.interval_choices import CostInterval,IntervalOption,solve_interval_choices
from scar.optimization.stateful_cost_bounds import derive_stateful_cost_bounds,_need,_write,_receipt


def _ref(artifact,locator):return decode(ProviderFactReference,dict(artifact=artifact,locator=locator))


def _solve_from_bounds(replayed,bounds_ref):
    original_choice=read_artifact(replayed['original_choice'])
    originals=original_choice['solver']['option_domain']['stateful-call']
    by_id={r['id']:r for r in originals};rows=replayed['candidates']
    _need(len(by_id)==len(originals) and set(by_id)=={r['candidate_id']for r in rows},
        'original complete finite option domain must be retained')
    contracts={r['contract_sha256']for r in originals};_need(len(contracts)==1,'one original task contract required')
    contract=next(iter(contracts));options=[]
    for row in rows:
        original=by_id[row['candidate_id']];packet=read_artifact(row['artifact'])
        _need(packet['candidate_id']==row['candidate_id'] and packet['interval_ticks']==row['interval'],
            'freshly derived interval packet differs')
        # The solver recomputes strict comparisons. Neither boolean is an input.
        options.append(IntervalOption(id=row['candidate_id'],contract_sha256=contract,
            scope=replayed['comparison_scope'],interval=CostInterval(row['interval']['lower'],row['interval']['upper']),
            resource_demands=original['resource_demands'],legality_conditions=tuple(original['legality_conditions']),
            interval_evidence=(_ref(row['artifact'],'#'),),
            scope_evidence=(_ref(bounds_ref,'#/comparison_scope'),),
            legality_evidence=tuple(decode(ProviderFactReference,r)for r in original['legality_evidence'])))
    choice=solve_interval_choices(options,contract_sha256=contract,scope=replayed['comparison_scope'],
        tick_seconds=replayed['tick_seconds'])
    original_pruned={r['candidate_id']for r in rows if r['strictly_dominated']}
    _need(set(choice['strictly_excluded_option_ids'])==original_pruned,
        'independent algebraic comparison differs from freshly derived inequalities')
    return choice,originals,contract


def _plan_document(replayed,bounds_ref,choice,choice_ref,originals,contract,supplied_bounds):
    selected=choice['selected_id']
    selection='NO_OP' if selected is None or selected=='NO_OP' else 'MODEL_OPTION_FOR_EXISTING_BACKEND_LOWERING'
    selected_index=next((i for i,row in enumerate(choice['options'])if row['id']==selected),None)
    selected_packet=next((read_artifact(row['artifact'])for row in replayed['candidates']
        if row['candidate_id']==selected),None)
    report=dict(schema='scar.stateful-interval-choice-plan',schema_version=1,
        original_admission=replayed['previous_admission'],original_choice=replayed['original_choice'],
        original_domain=replayed['domain'],supplied_previous_bounds=supplied_bounds,
        recomputed_bounds=bounds_ref,interval_choice=choice_ref,contract_sha256=contract,
        source_bounds_artifact=bounds_ref,all_option_contract_sha256=contract,
        original_option_domain_evidence=dict(artifact=replayed['original_choice'],
            locator='#/solver/option_domain/stateful-call'),
        cost_legality_scope_evidence=dict(artifact=choice_ref,locator='#/options'),
        selected_option_evidence=None if selected_index is None else dict(artifact=choice_ref,
            locator='#/options/'+str(selected_index)),
        selected_cost_profile=None if selected_packet is None else selected_packet.get('source_profile'),
        scope=replayed['comparison_scope'],tick_seconds=replayed['tick_seconds'],
        status=choice['status'],selected_id=selected,selected_candidate_id=selected,
        decision=selection,selection_decision=selection,
        original_option_count=len(originals),retained_option_count=len(choice['options']),
        strictly_excluded_options=len(choice['strictly_excluded_option_ids']),
        remaining_frontier=len(choice['remaining_frontier']),
        prior_partial_unknown_count=replayed['counts']['partial_unknown_options'],
        prior_partial_options_strictly_dominated=replayed['counts']['partial_options_strictly_dominated'],
        prior_partial_options_not_excluded=replayed['counts']['partial_options_not_excluded'],
        source_facts_and_auxiliary_replayed=True,exclusion_booleans_used_as_authority=False,
        original_planner_modified=False,original_program_modified=False,
        new_cost_profile_emitted=False,unknown_duration_imputed=False,
        compilation_or_mca_analysis_performed=False,calibration_remeasured=False,target_workload_executed=False,
        target_variants_ranked=False,backend_transformation_performed=False,
        physical_optimality_proven=False,deployment_authorized=False,
        lowering_frontier=['INTERVAL_COMPARISON_IS_ONE_WARM_METHOD_CALL_NOT_ENTRY_DEPLOYMENT_AUTHORIZATION',
            'REMAINING_CROSSING_INTERVALS_PREVENT_FULL_FINITE_MODEL_SELECTION'] if selected is None else [
            'SELECTED_OPTION_STILL_REQUIRES_ORIGINAL_BACKEND_LEGALITY_AND_DELIVERY_GATES'],
        implementation=_receipt(__file__),solver_implementation=_receipt(interval_choices.__file__))
    return report


def plan_stateful_bounds(admission_report,out,*,bounds=None):
    """Freshly derive all bounds even if a previous bounds report is supplied."""
    supplied_bounds=None
    if bounds is not None:
        supplied_bounds=artifact_reference(bounds,'scar.stateful-cpu-cost-bounds');old=read_artifact(supplied_bounds)
        _need(old['schema_version']==1,'supported previous bounds report required')
        if admission_report is None:admission_report=old['previous_admission']['uri']
        _need(artifact_reference(admission_report,'scar.admitted-cpu-domain-batch-plan')==old['previous_admission'],
            'supplied bounds and admission authority differ')
    _need(admission_report is not None,'original independent admission authority required')
    out=Path(out).resolve();_need(not out.exists(),'fresh interval planner directory required')
    out.mkdir(parents=True)
    replayed=derive_stateful_cost_bounds(admission_report,out/'replayed-bounds')
    bounds_ref=artifact_reference(out/'replayed-bounds/result.json',replayed['schema'])
    choice,originals,contract=_solve_from_bounds(replayed,bounds_ref)
    choice_ref=_write(out/'interval-choice.json',choice)
    report=_plan_document(replayed,bounds_ref,choice,choice_ref,originals,contract,supplied_bounds)
    _write(out/'result.json',report);return report


def verify_plan_stateful_bounds(path):
    """Re-derive auxiliary facts in disposable storage and replay all algebra.

    Existing artifacts are read only. Temporary output URI differences are
    normalized only after each fresh candidate packet equals the published
    packet byte-for-byte. Compiler/MCA/probe or target execution is not used.
    """
    reference=artifact_reference(path,'scar.stateful-interval-choice-plan');plan=read_artifact(reference)
    _need(plan['schema_version']==1,'supported interval plan required')
    admission=read_artifact(plan['original_admission'])
    published=read_artifact(plan['recomputed_bounds'])
    _need(published['previous_admission']==plan['original_admission']
        and published['domain']==plan['original_domain'] and admission['domain']==plan['original_domain'],
        'original interval-plan authority/domain differs')
    if plan['supplied_previous_bounds']is not None:
        old=read_artifact(plan['supplied_previous_bounds'])
        _need(old['previous_admission']==plan['original_admission'],'supplied older bounds authority differs')
    with tempfile.TemporaryDirectory(prefix='scar-interval-fact-replay-') as directory:
        fresh=derive_stateful_cost_bounds(plan['original_admission']['uri'],Path(directory)/'bounds')
        originals={row['candidate_id']:row for row in published['candidates']}
        _need(len(originals)==len(published['candidates']) and
            {row['candidate_id']for row in fresh['candidates']}==set(originals),'replayed full option set differs')
        for row in fresh['candidates']:
            original=originals[row['candidate_id']]
            _need(read_artifact(row['artifact'],json_document=False)==
                read_artifact(original['artifact'],json_document=False),'original fact/interval packet failed complete re-derivation')
            row['artifact']=original['artifact']
        _need(fresh==published,'complete bounds authority or arithmetic differs after source replay')
    choice,options,contract=_solve_from_bounds(published,plan['recomputed_bounds'])
    _need(json.loads(json.dumps(choice))==read_artifact(plan['interval_choice']),
        'original interval choice differs from exact algebraic replay')
    expected=_plan_document(published,plan['recomputed_bounds'],choice,plan['interval_choice'],
        options,contract,plan['supplied_previous_bounds'])
    _need(expected==plan,'interval-plan fields differ from replayed source and solver')
    read_artifact(reference)
    return plan


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--admission');group.add_argument('--bounds');parser.add_argument('--out',required=True)
    args=parser.parse_args(argv);report=plan_stateful_bounds(args.admission,args.out,bounds=args.bounds)
    print(json.dumps({k:report[k]for k in ('status','decision','retained_option_count',
        'prior_partial_options_strictly_dominated','remaining_frontier')},sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
