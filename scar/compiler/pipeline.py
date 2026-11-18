"""Compile guarded source regions through graph passes to C++ artifacts."""
from dataclasses import asdict
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import time

from .frontend import discover_regions
from .model import write_plan
from .validation import validate_plan

def compile_source(source,out,*,catalog=None,memory_budget_bytes=None,cpu_threads=None,target=None,
                   assertion_policy='preserve'):
    from scar.optimization import optimize_region
    from scar.codegen import build
    if assertion_policy not in ('preserve', 'omit-proven-debug'):
        raise ValueError('unsupported assertion policy')
    if catalog is None and assertion_policy != 'preserve':
        raise ValueError('assertion omission requires an original F4 --catalog')
    started=time.perf_counter();out=Path(out).absolute()
    if out.exists():raise ValueError('fresh compiler output required')
    out.mkdir(parents=True)
    template_source=None
    if catalog is None:
        regions,rejections=discover_regions(source)
        discovery='SOURCE_ARRAY_PROFILE'
    else:
        from scar.analysis.program_view import ProgramView
        from scar.analysis.array_value_templates import ArrayTemplateSource
        template_source=ArrayTemplateSource(ProgramView(catalog),source,
            assertion_policy=assertion_policy)
        regions,rejections=template_source.regions,template_source.rejections
        discovery='ORIGINAL_F4_GRAPH'
    current=Path(source).resolve();source_digest=sha256(current.read_bytes()).hexdigest()
    admitted=[]
    for region in regions:
        if Path(region.source.path).resolve()!=current or region.source.sha256!=source_digest:
            rejections.append(dict(qualname=region.source.qualname,code='SOURCE_VERSION_MISMATCH',
                reason='SOURCE_VERSION_MISMATCH',detail='compile source differs from original graph source version'))
        else:admitted.append(region)
    regions=admitted
    capacity=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
    threads=(1 if target is not None else min(4,capacity)) if cpu_threads is None else cpu_threads
    (out/'graph.json').write_text(json.dumps(dict(schema='scar.optimization-regions',schema_version=1,
        discovery=discovery,assertion_policy=assertion_policy,
        regions=[asdict(r) for r in regions],rejections=rejections),indent=2,allow_nan=False)+'\n')
    if template_source is not None:
        from scar.ir.execution_recipe import artifact_reference
        from scar.ir.program.providers import ProviderArtifact,ProviderFactReference
        graph_artifact=ProviderArtifact(**artifact_reference(out/'graph.json','scar.optimization-regions'))
    publications=[]
    for ordinal,region in enumerate(regions):
        provider=None
        if template_source is None:
            plan=optimize_region(region,memory_budget_bytes=memory_budget_bytes,cpu_threads=threads)
        else:
            reference=ProviderFactReference(graph_artifact,'#/regions/'+str(ordinal))
            # Admission and optimization share this source session. Templates
            # describe structure, never a selected executor or runtime cache.
            plan,provider=template_source.analyze(reference,memory_budget_bytes=memory_budget_bytes,cpu_threads=threads)
        folder=out/f'region-{ordinal}';folder.mkdir()
        row=dict(region=region.id,qualname=region.source.qualname,source=asdict(region.source),
            proof=plan.proof,cost=plan.cost,selected=False,
            omitted_source_assertions=sum(origin.get('kind') == 'source_assertion_omission'
                for origin in region.source.origins))
        if provider is not None:
            from scar.analysis.array_value_templates import publish_array_templates
            templates=publish_array_templates(provider,folder/'value-templates')
            row['value_templates']=str(folder/'value-templates/result.json')
            row['value_template_summary']=templates['summary']
        from scar.optimization.demand import analyze_region_demand
        demand=analyze_region_demand(plan)
        (folder/'demand.json').write_text(json.dumps(demand,indent=2,allow_nan=False)+'\n')
        row['demand']=str(folder/'demand.json')
        if target is not None:
            from scar.optimization.hardware import propose_cpu_placements,bind_cpu_placement
            placements=propose_cpu_placements(plan,target,memory_budget_bytes=memory_budget_bytes,max_threads=cpu_threads)
            domain=folder/'placement-domain.json'
            domain.write_text(json.dumps(dict(schema='scar.resource-feasible-domain',schema_version=1,
                placements=[asdict(p) for p in placements],ranked=False,program_variants_executed=False),indent=2,allow_nan=False)+'\n')
            row['placement_domain']=str(domain)
            if cpu_threads is None:
                write_plan(plan,folder/'plan.json')
                row.update(status='PLANNING_REQUIRED',plan=str(folder/'plan.json'),
                    base_plan_scope='UNBOUND_SINGLE_THREAD_BASELINE_NOT_A_SELECTED_ALLOCATION',
                    reason='hardware allocation needs an explicit cost model; no black-box variant search is performed')
                publications.append(row);continue
            if cpu_threads is not None:placements=tuple(p for p in placements if len(p.cpu_ids)==cpu_threads)
            if not placements:
                row.update(status='UNSUPPORTED',reason='no executor satisfies the explicitly requested CPU allocation')
                publications.append(row);continue
            # This is a caller-requested allocation preview, not a claim that
            # the first feasible executor is the fastest physical placement.
            plan=bind_cpu_placement(plan,placements[0])
            row['allocation_policy']='FIXED_THREAD_PREVIEW_FIRST_FEASIBLE_EXECUTOR_UNRANKED'
        write_plan(plan,folder/'plan.json');row['plan']=str(folder/'plan.json')
        row.update(proof=plan.proof,cost=plan.cost,
            resource_binding=None if plan.resource_binding is None else asdict(plan.resource_binding))
        try:
            validate_plan(plan)
            resource_schedule=None
            if plan.resource_binding is not None:
                from scar.optimization.schedule import build_region_schedule,schedule_document
                resource_schedule=build_region_schedule(plan,target)
                schedule_path=folder/'resource-plan.json'
                schedule_path.write_text(json.dumps(schedule_document(resource_schedule),indent=2,allow_nan=False)+'\n')
                row['resource_plan']=str(schedule_path)
            artifact=build(plan,folder/'native',resource_schedule=resource_schedule)
            row.update(status='COMPILED_UNMEASURED',artifact=str(folder/'native'),backend=artifact.report)
        except (ValueError,NotImplementedError,RuntimeError) as error:
            row.update(status='UNSUPPORTED',reason=str(error))
        publications.append(row)
    report=dict(schema='scar.compiler-build',schema_version=1,source=str(Path(source).absolute()),
        catalog=None if catalog is None else str(Path(catalog).absolute()),discovery=discovery,
        assertion_policy=assertion_policy,regions=publications,
        rejected_functions=len(rejections),whole_program_compiled=False,first_stage_complete=False,
        hardware=dict(cpu_affinity_capacity=capacity,
            planned_cpu_threads=None if target is not None and cpu_threads is None else threads,
            target=None if target is None else str(Path(target).absolute()),
            memory_placement='UNBOUND',selection='ANALYTICAL_PLANNING_REQUIRED',black_box_variant_search=False),
        selected_for_use=False,seconds=time.perf_counter()-started)
    (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source');parser.add_argument('--out',required=True);parser.add_argument('--catalog')
    parser.add_argument('--memory-budget-bytes',type=int)
    parser.add_argument('--cpu-threads',type=int,help='CPU workers; default min(4, available affinity)')
    parser.add_argument('--target',help='original hardware graph; cost-model planning or explicit CPU-allocation preview')
    parser.add_argument('--assertion-policy', choices=('preserve', 'omit-proven-debug'), default='preserve',
        help='retain source assertions or opt into proven debug assertion omission using --catalog')
    args=parser.parse_args();r=compile_source(args.source,args.out,catalog=args.catalog,
        memory_budget_bytes=args.memory_budget_bytes,cpu_threads=args.cpu_threads,
        target=args.target,assertion_policy=args.assertion_policy)
    print(json.dumps(dict(compiled=sum(x['status']=='COMPILED_UNMEASURED' for x in r['regions']),
        regions=len(r['regions']),seconds=r['seconds'],result=str(Path(args.out)/'result.json'))))

if __name__=='__main__':main()
