"""Prepare one source-proven stateful implementation for validation.

The cost-complete CPU choice solver remains authoritative for its domain.
An explicitly requested validation preview compiles one fixed allocation;
it neither ranks target variants nor claims the allocation is optimal.
"""
from dataclasses import asdict
from hashlib import sha256
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


def check_observed_specialization(value):
    """Cold-replay source, observed shapes and recipe without loading native.

    This admits a validation preview only. Cost, physical resource closure and
    deployment remain separate, even when its source and lowering agree.
    """
    from scar.analysis.build import BuildSession
    from scar.compiler.input_metadata import stateful_shapes_from_metadata
    from scar.compiler.stateful_frontend import discover_stateful_regions
    from scar.compiler.stateful_model import stateful_from_dict,stateful_digest
    from scar.ir.execution_recipe import artifact_reference,read_artifact
    from scar.optimization.resource_solver import semantic_sha256
    from scar.optimization.stateful_cpu_choices import _domain
    from scar.optimization.stateful_recipe import unpack_stateful_recipe

    if not isinstance(value,dict):
        path=Path(value).resolve()
        if path.is_dir():path/='result.json'
        value=artifact_reference(path,'scar.observed-stateful-specialization')
    report=read_artifact(value)
    if ((report.get('schema'),report.get('schema_version'))!=('scar.observed-stateful-specialization',1)
            or report.get('implementation_sha256')!=sha256(Path(__file__).with_name('specialize_stateful.py').read_bytes()).hexdigest()):
        raise ValueError('observed specialization schema or producer differs')
    region=stateful_from_dict(read_artifact(report['source_region']))
    shapes,evidence=stateful_shapes_from_metadata(report['inputs']['uri'],region)
    read_artifact(report['inputs'])
    domain=_domain(read_artifact(report['domain']))
    program=read_artifact(report['program'])
    hardware=program.get('target')
    target=domain['target_snapshot']
    if (program.get('backend')!='F4' or hardware is None
            or (hardware['path'],hardware['sha256'],hardware['bytes'])
                !=(target['uri'],target['sha256'],target['size_bytes'])
            or domain['region']!=report['source_region']
            or report['region_id']!=region.id
            or report['region_sha256']!=stateful_digest(region)
            or report['input_shapes']!=shapes or report['shape_evidence']!=evidence
            or domain['request']['input_shapes']!=shapes):
        raise ValueError('observed source/shape/domain/program hardware differs')
    with BuildSession(Path(report['program']['uri']).parent) as session:
        found,_=discover_stateful_regions(session.view,source=region.source.path)
        if sum(item.id==region.id and stateful_digest(item)==stateful_digest(region) for item in found)!=1:
            raise ValueError('observed specialization differs from original F4 rediscovery')
    compiled=report['candidate'] is not None
    for key in ('native_compiled','source_F4_replayed'):
        if report.get(key) is not compiled:raise ValueError('candidate publication status differs')
    expected_authority='ORIGINAL_F4_EXACT_REDISCOVERY' if compiled else 'PENDING_CANDIDATE_F4_ADMISSION'
    if report.get('source_authority')!=expected_authority:
        raise ValueError('specialization source authority differs')
    expected_flags=dict(scalar_values_specialized=False,no_workload_source_rewrite=True,
        no_shape_or_cost_rebase=True,original_scalar_state_loaded_each_call=True,
        input_values_retained=False,all_future_calls_proven=False,runtime_guards_required=True,
        target_program_executed=False,performance_measured=False,selected_for_use=False,
        deployment_authorized=False)
    if (any(report.get(key) is not expected for key,expected in expected_flags.items())
            or report.get('default_backend')!='NO_OP'
            or report.get('resource_allocation_policy')!='EXPLICIT_ONE_WORKER_PREVIEW_LOWEST_LEGAL_CPU_MASK_UNRANKED'):
        raise ValueError('validation preview cannot authorize selection or deployment')
    native_files=0
    if compiled:
        candidate=read_artifact(report['candidate'])
        rows=[row for row in domain['candidates'] if row['worker_count']==1 and row['legal_under_guards']]
        selected=min(rows,key=lambda row:(row['binding']['cpu_ids'],row['binding']['executor_id']))
        if (candidate.get('schema')!='scar.stateful-validation-candidate' or candidate.get('schema_version')!=1
                or candidate['program']!=report['program'] or candidate['region']!=report['source_region']
                or candidate['candidate_id']!=selected['id'] or candidate['resource_binding']!=selected['binding']
                or any(candidate.get(key) is not False for key in
                       ('selected_for_use','validated','physical_optimality_proven','target_variants_ranked','whole_program_cpp'))):
            raise ValueError('validation candidate differs from its source and unranked CPU allocation')
        read_artifact(candidate['recipe'])
        recipe_region,plan=unpack_stateful_recipe(candidate['recipe']['uri'])
        native=read_artifact(candidate['native_artifact'])
        directory=Path(candidate['native_directory'])
        if (Path(candidate['native_artifact']['uri']).parent!=directory
                or stateful_digest(recipe_region)!=stateful_digest(region)
                or plan['input_shapes']!=shapes or plan['resource_binding']!=selected['binding']
                or native['region_sha256']!=stateful_digest(region)
                or native['plan_sha256']!=semantic_sha256(plan)
                or native['cpu_ids']!=selected['binding']['cpu_ids'] or native['cpu_threads']!=1):
            raise ValueError('native/recipe/source/observed geometry or CPU allocation differs')
        for name,filename in native['files'].items():
            if Path(filename).name!=filename or sha256((directory/filename).read_bytes()).hexdigest()!=native['sha256'][name]:
                raise ValueError('native artifact file differs: '+name)
            native_files+=1
    for reference in (value,report['program'],report['inputs'],report['source_region'],report['domain']):
        read_artifact(reference)
    return dict(schema='scar.observed-stateful-specialization-check',schema_version=1,
        specialization=value,region_id=region.id,input_shapes=shapes,
        original_F4_replayed_now=True,observations=evidence['observations'],
        native_metadata_verified=compiled,native_files_verified=native_files,
        native_library_loaded=False,target_program_executed=False,
        whole_program_model_complete=False,selected_for_use=False,deployment_authorized=False)


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description='Replay an observed stateful specialization without loading native')
    parser.add_argument('--specialization',required=True)
    args=parser.parse_args(argv)
    print(json.dumps(check_observed_specialization(args.specialization),sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
