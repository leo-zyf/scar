"""Source-replay one existing fusion pass and deliver its selected region.

The original F4 graph remains authoritative.  OIR records the replacement,
the existing cost gate decides profitability, and the original callable is
returned unchanged whenever the pass keeps Python or a trial is not requested.
"""
import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference,read_artifact
from scar.optimization.paired_entry_selection import DEFAULT_MINIMUM_GAIN

SCHEMA='scar.macro-pipeline-preparation'


def _write(path,value):
    with Path(path).open('x')as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    return artifact_reference(Path(path),value['schema'])


def _reference(value,schema):
    return value if type(value)is dict else artifact_reference(Path(value).resolve(),schema)


def _pins():
    root=Path(__file__).resolve().parents[1]
    names=('compiler/macro_pipeline.py','compiler/project_pipeline_temporal.py',
        'analysis/pipeline_region_pass.py','optimization/macro_pass.py',
        'compiler/semantic_revision.py','analysis/source_pipeline_graph.py',
        'optimization/semantic_rewrite.py','optimization/semantic_region_demand.py',
        'optimization/optimizer.py','optimization/graph.py',
        'optimization/deployment_cost.py','compiler/pipeline_runtime.py')
    return {name:sha256((root/name).read_bytes()).hexdigest()for name in names}


def _derive(inputs,profile,cpus,budget,gain,*,strict_minimum_gain=True):
    from scar.compiler.project_pipeline_temporal import open_temporal_inputs
    from scar.analysis.pipeline_region_pass import derive_fusion_region
    from scar.compiler.pipeline_runtime import derive_fusion_blueprint
    from scar.optimization.macro_pass import run_fusion_pass
    if profile is not None:read_artifact(profile)
    with open_temporal_inputs(inputs,include_source=True)as(pipeline,memory,_,target,proof,values,raw):
        refs=(dict(role='ORIGINAL_F4_INPUTS',artifact=inputs),
            *(dict(role='ORIGINAL_'+name.upper(),artifact=raw[name])for name in('proof','source_values','memory')))
        region=derive_fusion_region(pipeline,memory,proof,values,source_refs=refs)
        blueprint=derive_fusion_blueprint(pipeline,memory,target,available_cpu_ids=cpus,source_refs=refs)
        budget=memory['memory_budget_bytes']if budget is None else budget
        report=run_fusion_pass(region,blueprint,cost_profile=None if profile is None else profile['uri'],
            available_cpu_ids=cpus,memory_budget_bytes=budget,minimum_gain=gain,
            strict_minimum_gain=strict_minimum_gain)
        report=json.loads(json.dumps(report,sort_keys=True,allow_nan=False))
        return pipeline,region,blueprint,report,budget,values


def _metadata(inputs,profile,cpus,budget,gain,files,decision,*,strict_minimum_gain=True):
    result=dict(schema=SCHEMA,schema_version=2,inputs=inputs,cost_profile=profile,
        available_cpu_ids=list(cpus),memory_budget_bytes=budget,minimum_gain=gain,
        files=files,decision=decision,producer_pins=_pins(),
        source_F4_and_value_routes_replayed=True,source_graph_unchanged=True,
        original_python_is_actual_alternative=True,global_solver_invoked=False,
        candidate_versions_benchmarked=False,target_program_executed=False,
        deployment_authorized=False,default_backend='NO_OP')
    if strict_minimum_gain:result['strict_minimum_gain']=True
    return result


def _structural_selection(region, report):
    from scar.analysis.pipeline_region_pass import select_fusion_region

    if not report['legality']['qualified']:
        return None
    if report['decision']=='REWRITE':
        return report['optimization_selection']
    return select_fusion_region(region, rewrite=True,
        reason='Analyze the source-qualified fused graph under its original runtime guards.')


def prepare(inputs,out,*,cost_profile=None,available_cpu_ids=None,memory_budget_bytes=None,
        minimum_gain=DEFAULT_MINIMUM_GAIN,compile_native=False):
    from scar.compiler.project_pipeline_temporal import INPUTS
    from scar.optimization.macro_pass import SCHEMA as PASS_SCHEMA
    reference=_reference(inputs,INPUTS)
    profile=None if cost_profile is None else _reference(cost_profile,'scar.deployment-cost-profile')
    cpus=sorted(os.sched_getaffinity(0))if available_cpu_ids is None else list(available_cpu_ids)
    output=Path(out).resolve()
    if output.exists():raise ValueError('fresh macro pipeline output required')
    pipeline,region,blueprint,report,budget,values=_derive(reference,profile,cpus,memory_budget_bytes,minimum_gain)
    output.mkdir(parents=True)
    context=dict(schema='scar.macro-pass-cost-context-publication',schema_version=1,
        context=report['cost_context'],complete_scope_reuse_count=1,
        component_costs_are_not_complete_scope_observations=True,
        original_python_preserved_when_unknown=True)
    files=dict(region=_write(output/'region.json',region),blueprint=_write(output/'blueprint.json',blueprint),
        pass_report=_write(output/'pass.json',report),cost_context=_write(output/'cost-context.json',context))
    from scar.compiler.semantic_revision import publish_fusion_revision
    files['semantic_revision']=publish_fusion_revision(
        pipeline,report['optimization_selection'],values,output/'semantic')
    from scar.optimization.semantic_region_demand import derive_semantic_region_demand
    selection=_structural_selection(region,report)
    files.update(candidate_revision=None,candidate_demand=None)
    if selection is not None:
        candidate=(files['semantic_revision'] if report['decision']=='REWRITE' else
            publish_fusion_revision(pipeline,selection,values,output/'candidate'))
        demand=derive_semantic_region_demand(read_artifact(candidate),pipeline,selection,blueprint)
        files.update(candidate_revision=candidate,
            candidate_demand=_write(output/'candidate-demand.json',demand))
    metadata=_metadata(reference,profile,cpus,budget,minimum_gain,files,report['decision'])
    preparation=_write(output/'preparation.json',metadata)
    native=None
    if compile_native and report['decision']=='REWRITE':
        from scar.compiler.pipeline_runtime import build_fusion_blueprint
        artifact=build_fusion_blueprint(pipeline,blueprint,output/'native')
        native=artifact_reference(artifact.directory/'artifact.json',artifact.report['schema'])
    result=dict(schema='scar.macro-pipeline-build',schema_version=1,preparation=preparation,
        semantic_revision=files['semantic_revision'],
        candidate_revision=files['candidate_revision'],candidate_demand=files['candidate_demand'],
        decision=report['decision'],native_artifact=native,native_compiled=native is not None,
        whole_program_cpp=False,source_replacement='CONTROLLED_MODULE_ENTRY_ONLY',
        target_program_executed=False,performance_measured=False,deployment_authorized=False)
    _write(output/'result.json',result)
    return result


def check(preparation):
    reference=_reference(preparation,SCHEMA);saved=read_artifact(reference)
    if saved.get('schema_version')!=2 or saved['producer_pins']!=_pins():
        raise ValueError('macro pipeline producer differs')
    strict=saved.get('strict_minimum_gain',False)
    pipeline,region,blueprint,report,budget,values=_derive(saved['inputs'],saved['cost_profile'],
        saved['available_cpu_ids'],saved['memory_budget_bytes'],saved['minimum_gain'],strict_minimum_gain=strict)
    if any(read_artifact(saved['files'][name])!=value for name,value in
            (('region',region),('blueprint',blueprint),('pass_report',report))):
        raise ValueError('macro pipeline differs from original source, cost or lowering replay')
    context=read_artifact(saved['files']['cost_context'])
    if context['context']!=report['cost_context']or context['complete_scope_reuse_count']!=1:
        raise ValueError('protected-call cost context differs')
    if saved!=_metadata(saved['inputs'],saved['cost_profile'],saved['available_cpu_ids'],budget,
            saved['minimum_gain'],saved['files'],report['decision'],strict_minimum_gain=strict):
        raise ValueError('macro pipeline preparation summary differs')
    from scar.compiler.semantic_revision import check_fusion_revision
    check_fusion_revision(saved['files']['semantic_revision'],pipeline,
        report['optimization_selection'],values)
    selection=_structural_selection(region,report)
    candidate,demand_ref=(saved['files'][name] for name in ('candidate_revision','candidate_demand'))
    if selection is None:
        if candidate is not None or demand_ref is not None:
            raise ValueError('unqualified structural fusion cannot publish a candidate revision')
    else:
        from scar.optimization.semantic_region_demand import verify_semantic_region_demand
        if (candidate is None or demand_ref is None or report['decision']=='REWRITE'
                and candidate!=saved['files']['semantic_revision']):
            raise ValueError('candidate revision or selected fusion identity differs')
        if report['decision']=='KEEP':
            check_fusion_revision(candidate,pipeline,selection,values)
        verify_semantic_region_demand(read_artifact(candidate),pipeline,selection,blueprint,
            read_artifact(demand_ref))
    return saved,pipeline,region,blueprint,report


def select_callable(original,build,*,trial=False,exclusive_inputs=False):
    """Return NO_OP without codegen imports, or cold-admit one explicit trial."""
    build_ref=_reference(build,'scar.macro-pipeline-build');published=read_artifact(build_ref)
    if published.get('schema_version')!=1:raise ValueError('versioned macro pipeline build required')
    if published['decision']=='KEEP'or not trial:return original
    saved,pipeline,region,blueprint,report=check(published['preparation'])
    if published['decision']!=report['decision']or published['native_artifact']is None:
        raise ValueError('a selected and compiled source-qualified fusion is required')
    native=published['native_artifact'];read_artifact(native)
    from scar.codegen.fused_pipeline import load
    from scar.compiler.pipeline_runtime import select_fusion_callable
    artifact=load(Path(native['uri']).parent)
    return select_fusion_callable(original,report,artifact=artifact,trial=True,
        exclusive_inputs=exclusive_inputs,region_document=region,blueprint=blueprint)


@contextmanager
def install(module,build,*,trial=False,exclusive_inputs=False):
    """Install and restore the exact source module binding for one controlled run."""
    published=read_artifact(_reference(build,'scar.macro-pipeline-build'))
    saved=read_artifact(published['preparation']);region=read_artifact(saved['files']['region'])
    name=region['qualification']['source']['qualname']
    if '.'in name:raise ValueError('direct module function required for controlled macro installation')
    original=vars(module)[name]
    selected=select_callable(original,build,trial=trial,exclusive_inputs=exclusive_inputs)
    setattr(module,name,selected)
    try:yield selected
    finally:setattr(module,name,original)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs');parser.add_argument('--out');parser.add_argument('--cost-profile')
    parser.add_argument('--cpu-ids',help='cost-applicable caller affinity domain, not pinning')
    parser.add_argument('--memory-budget-bytes',type=int)
    parser.add_argument('--minimum-gain',type=float,default=DEFAULT_MINIMUM_GAIN)
    parser.add_argument('--compile',action='store_true');parser.add_argument('--check')
    args=parser.parse_args(argv)
    if args.check:
        saved,*_=check(args.check);result=dict(source_and_pass_replayed=True,decision=saved['decision'])
    elif not args.inputs or not args.out:parser.error('--inputs and --out required')
    else:result=prepare(args.inputs,args.out,cost_profile=args.cost_profile,
        available_cpu_ids=None if args.cpu_ids is None else [int(c)for c in args.cpu_ids.split(',')],
        memory_budget_bytes=args.memory_budget_bytes,minimum_gain=args.minimum_gain,compile_native=args.compile)
    print(json.dumps(result,sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
