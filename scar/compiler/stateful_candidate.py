"""Prepare one source-proven stateful implementation for validation.

The cost-complete CPU choice solver remains authoritative for its domain.
An explicitly requested validation preview compiles one fixed allocation;
it neither ranks target variants nor claims the allocation is optimal.
"""
from dataclasses import asdict
import json
from pathlib import Path


def prepare_stateful_candidate(program,domain,out,*,cpu_threads=1):
    from scar.analysis.build import BuildSession
    from scar.compiler.stateful_frontend import discover_stateful_regions
    from scar.compiler.stateful_model import stateful_from_dict,stateful_digest
    from scar.compiler.model import CpuPlacement
    from scar.ir.execution_recipe import artifact_reference,read_artifact
    from scar.optimization.stateful_cpu_choices import _domain
    from scar.optimization.stateful_schedule import plan_stateful_region
    from scar.optimization.stateful_recipe import make_stateful_recipe
    from scar.codegen.recipe import build_recipe
    out=Path(out).resolve()
    if out.exists():raise ValueError('fresh validation-candidate directory required')
    if type(cpu_threads) is not int or cpu_threads<1:raise ValueError('positive explicit validation worker count required')
    domain=_domain(domain)
    region=stateful_from_dict(read_artifact(domain['region']))
    program=Path(program).resolve()
    if program.is_dir():program=program/'manifest.json'
    parent=artifact_reference(program,'scar.program-build');document=read_artifact(parent)
    hardware=document['target'];target=domain['target_snapshot']
    if (target['uri'],target['sha256'],target['size_bytes'])!=(hardware['path'],hardware['sha256'],hardware['bytes']):
        raise ValueError('stateful candidate uses a different program hardware snapshot')
    with BuildSession(program.parent) as session:
        found,_=discover_stateful_regions(session.view,source=region.source.path)
        matches=[value for value in found if value.id==region.id and stateful_digest(value)==stateful_digest(region)]
        if len(matches)!=1:raise ValueError('stateful region differs from original F4 rediscovery')
    candidates=[row for row in domain['candidates'] if row['worker_count']==cpu_threads and row['legal_under_guards']]
    if not candidates:raise ValueError('no guarded hardware allocation supports requested validation workers')
    selected=min(candidates,key=lambda row:(row['binding']['cpu_ids'],row['binding']['executor_id']))
    binding=dict(selected['binding']);binding['cpu_ids']=tuple(binding['cpu_ids']);binding['evidence']=tuple(binding['evidence'])
    placement=CpuPlacement(**binding)
    plan=plan_stateful_region(region,input_shapes=domain['request']['input_shapes'],cpu_threads=cpu_threads,
        memory_budget_bytes=placement.memory_budget_bytes,resource_binding=placement)
    out.mkdir(parents=True)
    recipe=make_stateful_recipe(region,plan,out=out/'recipe')
    lowered=build_recipe(recipe,out/'lowered')
    report=dict(schema='scar.stateful-validation-candidate',schema_version=1,program=parent,
        region=domain['region'],candidate_id=selected['id'],source_authority='ORIGINAL_F4_REDISCOVERY_EXACT_MATCH',
        policy='EXPLICIT_WORKER_PREVIEW_LOWEST_LEGAL_CPU_MASK_UNRANKED',
        resource_binding=asdict(placement),recipe=artifact_reference(out/'recipe/recipe.json',recipe['schema']),
        native_directory=str(lowered.native_directory),
        native_artifact=artifact_reference(Path(lowered.native_directory)/'artifact.json',lowered.report['schema']),
        cost_domain_selection_unchanged=True,whole_program_cpp=False,selected_for_use=False,
        validated=False,profitability='UNPROVEN',physical_optimality_proven=False,
        all_resources=domain['all_resources'],target_variants_ranked=False)
    read_artifact(parent);_domain(domain)
    (out/'candidate.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report
