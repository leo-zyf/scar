"""Replay stateful CPU choices and lower the selected region to C++20."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact, recipe_id
from scar.optimization.resource_solver import semantic_sha256


def lower_selected_stateful(region,plan,out,*,backend_descriptor_sha256,plan_only=False):
    from scar.optimization.stateful_recipe import make_stateful_recipe,verify_stateful_recipe
    if type(plan_only)is not bool:raise TypeError('exact plan-only boolean required')
    out=Path(out).resolve()
    recipe=verify_stateful_recipe(make_stateful_recipe(region,plan,out=out/'recipe'))
    capability=recipe['implementation']['backend_capability_record'];graph=recipe['task_graph']
    if (semantic_sha256(capability)!=backend_descriptor_sha256
            or graph['schema_version']!=2 or graph['executable']is not True):
        raise ValueError('source plan/backend differs from selected executable model')
    result=dict(model_selected_native=True,execution='EXISTING_TRANSACTIONAL_CPU_STATEFUL_RECIPE',
        status='MODEL_SELECTED_PLAN_ONLY',recipe_id=recipe_id(recipe),
        recipe=artifact_reference(out/'recipe/recipe.json',recipe['schema']),
        task_graph_sha256=semantic_sha256(graph),selected_binding=plan['resource_binding'],
        payload={key:plan[key]for key in ('external_array_bytes','workspace_payload_bytes',
            'total_region_array_payload_bytes','memory_budget_bytes','budget_scope')},
        compiled_variant_count=0,native_directory=None,artifact=None,lowering_receipt=None,
        deployment_authorized=False,selected_for_use=False)
    if not plan_only:
        from scar.codegen.recipe import build_recipe
        lowered=build_recipe(recipe,out/'lowered');report=lowered.report
        if (report['recipe_id']!=result['recipe_id'] or report['stateful_abi_version']!=1
                or report['backend_capability_record']!=capability):
            raise ValueError('compiled artifact differs from source recipe/ABI/backend')
        result.update(status='COMPILED_UNMEASURED',compiled_variant_count=1,
            native_directory=str(lowered.native_directory),artifact=report,
            lowering_receipt=artifact_reference(out/'lowered/lowering-receipt.json','scar.execution-recipe-lowering'),
            stateful_abi_version=1,operator_abi_version=1,cxx_standard=20)
    return result



def _require(condition, message):
    if not condition: raise ValueError(message)


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def lower_stateful_choices(choice_path, out, *, plan_only=False):
    """Verify the full recorded domain; plan or compile its selected CPU only."""
    from scar.optimization.stateful_cpu_model import verify_stateful_cpu_choices, selected_stateful_plan

    _require(type(plan_only) is bool, 'plan_only must be an exact boolean')
    choice_path = Path(choice_path).absolute()
    if choice_path.is_dir(): choice_path /= 'choice.json'
    directory = Path(out).resolve()
    _require(not directory.exists(), 'fresh stateful choice lowering output required')
    choice_ref = artifact_reference(choice_path, 'scar.stateful-cpu-choice')
    recorded = read_artifact(choice_ref)
    checked = verify_stateful_cpu_choices(choice_path)
    _require(semantic_sha256(recorded) == semantic_sha256(checked),
             'stateful choice differs from complete source/hardware/cost replay')
    domain = read_artifact(checked['domain'])
    selection = checked['selected_id']
    result = dict(schema='scar.lowered-stateful-cpu-choice', schema_version=1,
        upstream_choice=choice_ref, choice_semantic_sha256=semantic_sha256(checked),
        domain=checked['domain'], domain_id=domain['domain_id'],
        selected_id=selection, complete_cost_domain=checked['complete_cost_domain'],
        upstream_decision=checked['decision'], plan_only=plan_only,
        choice_scope=checked['scope'], comparison_model=checked['comparison_model'],
        resource_policy=checked['resource_policy'],
        original_no_op_available=checked['original_no_op_available'],
        model_selected_native=False, compiled_variant_count=0,
        recipe=None, recipe_id=None, native_directory=None, artifact=None,
        lowering_receipt=None, actual_target_duration_seconds=None,
        target_program_executed=False, target_variants_ranked_by_measurement=False,
        selected_for_use=False, deployment_authorized=False,
        physical_optimality_proven=False, whole_program_makespan_optimized=False)
    if selection in (None, 'NO_OP'):
        _require(checked['decision'] == 'NO_OP', 'original execution decision differs')
        result.update(status='ORIGINAL_EXECUTION_RETAINED', execution='UNCHANGED_ORIGINAL_PYTHON',
            reason='MODEL_SELECTED_ORIGINAL' if selection == 'NO_OP' else 'COST_OR_LEGALITY_DOMAIN_UNRESOLVED',
            unresolved=checked['solver'].get('unresolved', []),
            frontier=checked['solver'].get('frontier', []))
        directory.mkdir(parents=True)
        _write(directory / 'lowering.json', result)
        return result

    solver = checked['solver']
    _require(checked['complete_cost_domain'] is True
        and checked['decision'] == 'SELECT_NATIVE_IMPLEMENTATION_FOR_LOWERING'
        and solver.get('status') == 'OPTIMAL_WITHIN_FINITE_MODEL'
        and not solver.get('frontier') and not solver.get('unresolved')
        and solver.get('search_budget_exhausted') is False and solver.get('gap_ticks') == 0
        and checked['physical_optimality_proven'] is False,
        'complete finite-domain model selection required before native lowering')
    variants = checked['variants']
    index = next((i for i, row in enumerate(variants) if row['id'] == selection), None)
    _require(index is not None, 'selected implementation is absent from the retained domain')
    variant = variants[index]
    _require(variant['cost_status'] == 'KNOWN' and not variant['frontier'],
             'selected implementation requires an applicable complete-call cost')
    region, plan = selected_stateful_plan(checked)
    context = variant['context']
    binding = {key: value for key, value in plan['resource_binding'].items() if key != 'evidence'}
    _require(binding == context['cpu_binding']
        and plan['input_shapes'] == context['input_shapes']
        and plan['memory_budget_bytes'] == context['memory_budget_bytes']
        and plan['budget_scope'] == context['budget_scope'],
        'selected source/CPU/shape/payload differs from its modeled call context')

    # The standard producer replays sources, effects, slots, target receipts,
    # provider and compiler context. Nonselected plans are never persisted here.
    result.update(lower_selected_stateful(region,plan,directory,
        backend_descriptor_sha256=context['backend']['descriptor_sha256'],plan_only=plan_only))
    result.update(solver_scope=dict(status=solver['status'], optimality=solver.get('optimality'),
            gap_ticks=solver['gap_ticks'], optimality_scope=domain['enumeration']['optimality_scope']),
        selected_model=dict(duration_ticks=variant['profile']['estimate']['duration_ticks'],
            tick_seconds=variant['profile']['estimate']['tick_seconds'],
            origin=variant['profile']['derivation']['origin'],
            assumptions=variant['profile']['derivation']['assumptions'],
            evidence=dict(artifact=choice_ref, locator=f'#/variants/{index}/profile'),
            uncertainty_scope='RETAINED_MODEL_DERIVATION_NOT_TARGET_MEASUREMENT_OR_PHYSICAL_PROOF'))
    _require(sha256(choice_path.read_bytes()).hexdigest() == choice_ref['sha256'],
             'upstream choice changed during lowering')
    _write(directory / 'lowering.json', result)
    return result


def lower_stateful_bounds(plan_path,out,*,plan_only=False):
    from scar.compiler.plan_stateful_bounds import verify_plan_stateful_bounds
    from scar.optimization.stateful_cpu_model import _domain, _cost, stateful_cpu_cost_context, _family
    from scar.compiler.stateful_model import stateful_from_dict
    from scar.compiler.model import CpuPlacement
    from scar.optimization.stateful_schedule import plan_stateful_region

    _require(type(plan_only)is bool,'exact plan-only boolean required')
    plan_path=Path(plan_path).resolve()
    if plan_path.is_dir():plan_path/= 'result.json'
    reference=artifact_reference(plan_path,'scar.stateful-interval-choice-plan')
    checked=verify_plan_stateful_bounds(plan_path);domain=_domain(checked['original_domain']['uri'])
    out=Path(out).resolve();_require(not out.exists(),'fresh interval lowering output required')
    selected=checked['selected_candidate_id']
    result=dict(schema='scar.lowered-stateful-interval-choice',schema_version=1,
        upstream_plan=reference,domain=checked['original_domain'],selected_id=selected,
        model_selected_native=False,compiled_variant_count=0,native_directory=None,recipe=None,recipe_id=None,
        plan_only=plan_only,selection_status=checked['status'],remaining_frontier=checked['remaining_frontier'],
        selected_for_use=False,deployment_authorized=False,target_program_executed=False,
        target_variants_ranked_by_measurement=False,physical_optimality_proven=False,
        whole_program_makespan_optimized=False,unknown_duration_imputed=False)
    if selected in(None,'NO_OP'):
        _require(checked['selection_decision']=='NO_OP','unselected interval choice cannot enable native code')
        result.update(status='ORIGINAL_EXECUTION_RETAINED',execution='UNCHANGED_ORIGINAL_PYTHON',
            reason='UNRESOLVED_COST_INTERVALS' if selected is None else 'MODEL_SELECTED_ORIGINAL')
    else:
        choice=read_artifact(checked['interval_choice'])
        _require(checked['status']==choice['status']=='FINITE_MODEL_OPTIMAL'
            and not choice['remaining_frontier'] and checked['selected_option_evidence'] is not None
            and checked['selected_cost_profile'] is not None,'finite model selection with a complete selected profile required')
        candidate=next(r for r in domain['candidates'] if r['id']==selected)
        _require(candidate['legal_under_guards'] and not candidate['legality_frontier'],'original selected capability declined')
        context=stateful_cpu_cost_context(domain,selected);profile_ref=checked['selected_cost_profile']
        profile=read_artifact(profile_ref);cost,issues,_=_cost(profile,context)
        option=next(r for r in choice['options'] if r['id']==selected)
        _require(cost is not None and not issues and option['interval']==dict(lower=cost['duration_ticks'],upper=cost['duration_ticks'])
            and cost['tick_seconds']==checked['tick_seconds'],'complete selected cost differs from interval decision')
        previous=read_artifact(checked['original_admission'])
        replayed=[r for r in previous['independently_replayed_new_profiles'] if r['candidate_id']==selected]
        if replayed:
            _require(len(replayed)==1 and replayed[0]['profile']==profile_ref,'selected calibration context differs')
            from scar.optimization.stateful_batch_cost import verify_batch_cpu_cost
            cost_replay=verify_batch_cpu_cost(domain,selected,replayed[0]['worker_result']['uri'])
            _require(cost_replay['profile']==profile_ref,'selected raw source qualification differs')
        else:
            from scar.optimization.stateful_cost_rebase import _model_dependencies
            cost_replay=_model_dependencies(profile,domain,checked['original_domain'],context,selected)
        binding=dict(candidate['binding']);binding['cpu_ids']=tuple(binding['cpu_ids']);binding['evidence']=tuple(binding['evidence'])
        region=stateful_from_dict(read_artifact(domain['region']))
        numeric=plan_stateful_region(region,input_shapes=domain['request']['input_shapes'],
            cpu_threads=candidate['worker_count'],memory_budget_bytes=domain['request']['memory_budget_bytes'],
            resource_binding=CpuPlacement(**binding))
        _require(semantic_sha256(_family(numeric))==domain['semantic_family_sha256']
            and numeric['input_shapes']==context['input_shapes'] and numeric['guard_contract']==context['guards'],
            'selected numeric graph/shape/effect guards differ from original model')
        result.update(lower_selected_stateful(region,numeric,out,
            backend_descriptor_sha256=context['backend']['descriptor_sha256'],plan_only=plan_only))
        result.update(selected_cost_profile=profile_ref,selected_cost_replay=cost_replay,
            selected_option_evidence=checked['selected_option_evidence'])
    read_artifact(reference);out.mkdir(parents=True,exist_ok=True)
    _write(out/'lowering.json', result)
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--choices')
    inputs.add_argument('--plan')
    parser.add_argument('--out', required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args(argv)
    if args.choices:
        report = lower_stateful_choices(args.choices, args.out, plan_only=args.plan_only)
    else:
        report = lower_stateful_bounds(args.plan, args.out, plan_only=args.plan_only)
    print(json.dumps({key: report[key] for key in ('status', 'selected_id', 'recipe_id',
        'native_directory', 'compiled_variant_count', 'deployment_authorized')}))
    return 0


__all__ = ['lower_selected_stateful', 'lower_stateful_choices', 'lower_stateful_bounds']


if __name__ == '__main__':
    raise SystemExit(main())
