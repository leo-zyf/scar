"""Lower one replayed CPU choice through the existing stateful recipe.

Unknown costs and the original NO_OP remain original execution. A complete
model selection reaches exactly one existing C++20 build; this is neither a
target timing experiment nor permission to install the resulting artifact.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact, recipe_id
from scar.optimization.resource_solver import semantic_sha256


def _require(condition, message):
    if not condition: raise ValueError(message)


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def lower_stateful_choices(choice_path, out, *, plan_only=False):
    """Verify the full recorded domain; plan or compile its selected CPU only."""
    from scar.optimization.stateful_cpu_choices import (
        verify_stateful_cpu_choices, selected_stateful_plan)
    from scar.optimization.stateful_recipe import make_stateful_recipe, verify_stateful_recipe

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
    recipe = make_stateful_recipe(region, plan, out=directory / 'recipe')
    recipe = verify_stateful_recipe(recipe)
    capability = recipe['implementation']['backend_capability_record']
    _require(semantic_sha256(capability) == context['backend']['descriptor_sha256'],
             'recipe implementation/compiler/provider differs from the selected cost model')
    graph = recipe['task_graph']
    _require(graph['schema_version'] == 2 and graph['executable'] is True,
             'public executable stateful v2 task graph required')
    result.update(model_selected_native=True, execution='EXISTING_TRANSACTIONAL_CPU_STATEFUL_RECIPE',
        status='MODEL_SELECTED_PLAN_ONLY', recipe_id=recipe_id(recipe),
        recipe=artifact_reference(directory / 'recipe/recipe.json', recipe['schema']),
        task_graph_sha256=semantic_sha256(graph), selected_binding=plan['resource_binding'],
        payload={key: plan[key] for key in ('external_array_bytes', 'workspace_payload_bytes',
            'total_region_array_payload_bytes', 'memory_budget_bytes', 'budget_scope')},
        solver_scope=dict(status=solver['status'], optimality=solver.get('optimality'),
            gap_ticks=solver['gap_ticks'], optimality_scope=domain['enumeration']['optimality_scope']),
        selected_model=dict(duration_ticks=variant['profile']['estimate']['duration_ticks'],
            tick_seconds=variant['profile']['estimate']['tick_seconds'],
            origin=variant['profile']['derivation']['origin'],
            assumptions=variant['profile']['derivation']['assumptions'],
            evidence=dict(artifact=choice_ref, locator=f'#/variants/{index}/profile'),
            uncertainty_scope='RETAINED_MODEL_DERIVATION_NOT_TARGET_MEASUREMENT_OR_PHYSICAL_PROOF'))
    if not plan_only:
        from scar.codegen.recipe import build_recipe
        lowered = build_recipe(recipe, directory / 'lowered')
        report = lowered.report
        _require(report['recipe_id'] == result['recipe_id']
            and report['stateful_abi_version'] == 1
            and report['backend_capability_record'] == capability,
            'compiled artifact differs from selected recipe/ABI/backend capability')
        result.update(status='COMPILED_UNMEASURED', compiled_variant_count=1,
            native_directory=str(lowered.native_directory), artifact=report,
            lowering_receipt=artifact_reference(directory / 'lowered/lowering-receipt.json',
                'scar.execution-recipe-lowering'),
            stateful_abi_version=1, operator_abi_version=1, cxx_standard=20)
    _require(sha256(choice_path.read_bytes()).hexdigest() == choice_ref['sha256'],
             'upstream choice changed during lowering')
    _write(directory / 'lowering.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--choices', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    report = lower_stateful_choices(args.choices, args.out, plan_only=args.plan_only)
    print(json.dumps({key: report[key] for key in ('status', 'selected_id', 'recipe_id',
        'native_directory', 'compiled_variant_count', 'deployment_authorized')}))


if __name__ == '__main__': main()
