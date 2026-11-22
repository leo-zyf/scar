"""Lower a replayed interval choice through the existing stateful C++ path.

Cost uncertainty never becomes an execution duration. An unresolved choice
keeps the original. A selected native option still needs its complete cost
source, original numeric/effect graph, placement and payload contract replay.
"""
import argparse
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference,read_artifact
from scar.optimization.resource_solver import semantic_sha256


def _need(value,message):
    if not value:raise ValueError(message)


def lower_stateful_bounds(plan_path,out,*,plan_only=False):
    from scar.compiler.plan_stateful_bounds import verify_plan_stateful_bounds
    from scar.optimization.stateful_cpu_model import _domain, _cost, stateful_cpu_cost_context, _family
    from scar.compiler.stateful_model import stateful_from_dict
    from scar.compiler.model import CpuPlacement
    from scar.optimization.stateful_schedule import plan_stateful_region
    from scar.compiler.selected_stateful_lowering import lower_selected_stateful

    _need(type(plan_only)is bool,'exact plan-only boolean required')
    plan_path=Path(plan_path).resolve()
    if plan_path.is_dir():plan_path/= 'result.json'
    reference=artifact_reference(plan_path,'scar.stateful-interval-choice-plan')
    checked=verify_plan_stateful_bounds(plan_path);domain=_domain(checked['original_domain']['uri'])
    out=Path(out).resolve();_need(not out.exists(),'fresh interval lowering output required')
    selected=checked['selected_candidate_id']
    result=dict(schema='scar.lowered-stateful-interval-choice',schema_version=1,
        upstream_plan=reference,domain=checked['original_domain'],selected_id=selected,
        model_selected_native=False,compiled_variant_count=0,native_directory=None,recipe=None,recipe_id=None,
        plan_only=plan_only,selection_status=checked['status'],remaining_frontier=checked['remaining_frontier'],
        selected_for_use=False,deployment_authorized=False,target_program_executed=False,
        target_variants_ranked_by_measurement=False,physical_optimality_proven=False,
        whole_program_makespan_optimized=False,unknown_duration_imputed=False)
    if selected in(None,'NO_OP'):
        _need(checked['selection_decision']=='NO_OP','unselected interval choice cannot enable native code')
        result.update(status='ORIGINAL_EXECUTION_RETAINED',execution='UNCHANGED_ORIGINAL_PYTHON',
            reason='UNRESOLVED_COST_INTERVALS' if selected is None else 'MODEL_SELECTED_ORIGINAL')
    else:
        choice=read_artifact(checked['interval_choice'])
        _need(checked['status']==choice['status']=='FINITE_MODEL_OPTIMAL'
            and not choice['remaining_frontier'] and checked['selected_option_evidence'] is not None
            and checked['selected_cost_profile'] is not None,'finite model selection with a complete selected profile required')
        candidate=next(r for r in domain['candidates'] if r['id']==selected)
        _need(candidate['legal_under_guards'] and not candidate['legality_frontier'],'original selected capability declined')
        context=stateful_cpu_cost_context(domain,selected);profile_ref=checked['selected_cost_profile']
        profile=read_artifact(profile_ref);cost,issues,_=_cost(profile,context)
        option=next(r for r in choice['options'] if r['id']==selected)
        _need(cost is not None and not issues and option['interval']==dict(lower=cost['duration_ticks'],upper=cost['duration_ticks'])
            and cost['tick_seconds']==checked['tick_seconds'],'complete selected cost differs from interval decision')
        previous=read_artifact(checked['original_admission'])
        replayed=[r for r in previous['independently_replayed_new_profiles'] if r['candidate_id']==selected]
        if replayed:
            _need(len(replayed)==1 and replayed[0]['profile']==profile_ref,'selected calibration context differs')
            from scar.optimization.stateful_batch_cost import verify_batch_cpu_cost
            cost_replay=verify_batch_cpu_cost(domain,selected,replayed[0]['worker_result']['uri'])
            _need(cost_replay['profile']==profile_ref,'selected raw source qualification differs')
        else:
            from scar.optimization.stateful_cost_rebase import _model_dependencies
            cost_replay=_model_dependencies(profile,domain,checked['original_domain'],context,selected)
        binding=dict(candidate['binding']);binding['cpu_ids']=tuple(binding['cpu_ids']);binding['evidence']=tuple(binding['evidence'])
        region=stateful_from_dict(read_artifact(domain['region']))
        numeric=plan_stateful_region(region,input_shapes=domain['request']['input_shapes'],
            cpu_threads=candidate['worker_count'],memory_budget_bytes=domain['request']['memory_budget_bytes'],
            resource_binding=CpuPlacement(**binding))
        _need(semantic_sha256(_family(numeric))==domain['semantic_family_sha256']
            and numeric['input_shapes']==context['input_shapes'] and numeric['guard_contract']==context['guards'],
            'selected numeric graph/shape/effect guards differ from original model')
        result.update(lower_selected_stateful(region,numeric,out,
            backend_descriptor_sha256=context['backend']['descriptor_sha256'],plan_only=plan_only))
        result.update(selected_cost_profile=profile_ref,selected_cost_replay=cost_replay,
            selected_option_evidence=checked['selected_option_evidence'])
    read_artifact(reference);out.mkdir(parents=True,exist_ok=True)
    (out/'lowering.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--plan-only',action='store_true');args=parser.parse_args(argv)
    r=lower_stateful_bounds(args.plan,args.out,plan_only=args.plan_only)
    print(json.dumps({key:r[key]for key in('status','selected_id','compiled_variant_count','deployment_authorized')}));return 0


if __name__=='__main__':raise SystemExit(main())
