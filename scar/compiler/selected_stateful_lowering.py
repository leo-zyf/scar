"""Common C++ lowering after a planner has admitted one original region."""
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference,recipe_id
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


__all__=['lower_selected_stateful']
