"""Join bounded native host observations to original source/value/time records."""
import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference,read_artifact
from scar.ir.versioned_schedule import schedule_document,validate_versioned_schedule
from scar.ir.temporal_resources import projection_document,validate_resource_projection
from scar.ir.activity_costs import activity_cost_document,validate_activity_cost_table
from scar.optimization.resource_solver import semantic_sha256

SCHEMA='scar.pipeline-host-model-preparation'


def _write(path,value):
    path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return artifact_reference(path,value['schema'])


def _pins():
    root=Path(__file__).resolve().parents[1]
    names=('compiler/pipeline_host_model.py','analysis/pipeline_host_envelope.py',
        'analysis/pipeline_host_samples.py','codegen/pipeline_host_probe.py',
        'analysis/pipeline_host_resources.py','optimization/pipeline_host_costs.py',
        'ir/activity_costs.py','ir/temporal_resources.py')
    return {name:sha256((root/name).read_bytes()).hexdigest()for name in names}


@contextmanager
def _inputs(observations):
    """Full source/selection/renderer replay without CDLL or target execution."""
    from scar.compiler.compile_pipeline_implementation import open_pipeline_implementation
    from scar.codegen.pipeline_host_probe import render_host_probe
    from scar.codegen.pipeline_implementation import _selected_plans,_source
    from scar.codegen import cpp,pipeline as original
    from scar.optimization.pipeline_ready_schedule import _dependency_graph
    from scar.optimization.pipeline_schedule import _target
    reference=artifact_reference(Path(observations).resolve(),'scar.pipeline-host-sample-observations')
    captured=read_artifact(reference);probe=read_artifact(captured['probe'])
    metadata=read_artifact(probe['original_artifact']);directory=Path(probe['original_artifact']['uri']).parent
    for ref in (probe['source'],probe['header'],probe['library'],probe['original_library']):
        read_artifact(ref,json_document=False)
    for kind,name in metadata['files'].items():
        if Path(name).name!=name or sha256((directory/name).read_bytes()).hexdigest()!=metadata['sha256'][kind]:
            raise ValueError('original host implementation receipt differs: '+kind)
    header=(Path(cpp.__file__).parent/'scar_abi_v1.h').read_bytes()
    if header!=(directory/metadata['files']['abi_header']).read_bytes()or header!=read_artifact(probe['header'],json_document=False):
        raise ValueError('probe, original host and current ABI headers differ')
    with open_pipeline_implementation(metadata['preparation'])as(_,pipeline,memory,target,problem,execution,certificate):
        if certificate is None:raise ValueError('source-qualified selected host certificate required')
        if captured['call_predecessors']!=certificate['call_predecessors']or captured['lane_call_order']!=certificate['lane_call_order']or captured['expected_cpu_ids']!=certificate['cpu_ids']:
            raise ValueError('observed source lane/causal policy differs from original selected implementation')
        plans=_selected_plans(pipeline,memory,target,problem,execution,certificate);operators=[]
        for call,receipt in zip(pipeline.calls,metadata['operators']):
            native=Path(receipt['artifact_directory']);report=json.loads((native/'artifact.json').read_bytes())
            plan=plans[call.id]
            for record in receipt['files'].values():
                if sha256(Path(record['path']).read_bytes()).hexdigest()!=record['sha256']:
                    raise ValueError('original ABI1 service bytes changed')
            if cpp._plan_digest(plan)!=report['plan_sha256']:
                raise ValueError('observed ABI1 implementation differs from source CPU plan')
            kernel=cpp._source(plan,cpp._supported(plan)).encode()
            if kernel!=(native/report['files']['source']).read_bytes():
                raise ValueError('original ABI1 source cannot be regenerated')
            if header!=(native/report['files']['abi_header']).read_bytes():
                raise ValueError('original operator ABI header differs from source-qualified host')
            if receipt['timed_guarded'] is not False:
                raise ValueError('host phase replay currently requires the untimed ABI1 service profile')
            operator=cpp.CppArtifact(plan,native,report,None,None)
            original._compatible(call,operator,memory)
            if original._operator_receipt(call,operator)!=receipt:raise ValueError('original service receipt differs')
            operators.append(operator)
        source=_source(pipeline,memory,operators,certificate)
        if source!=(directory/metadata['files']['source']).read_bytes():raise ValueError('original host source reconstruction differs')
        generated,provenance=render_host_probe(source,call_count=len(pipeline.calls),lane_cpu_ids=certificate['cpu_ids'])
        if generated.encode()!=read_artifact(probe['source'],json_document=False)or provenance!=probe['provenance']:
            raise ValueError('instrumented host source cannot be inverted/replayed from original source')
        _,snapshot,cpus,_,bindings=_target(target,certificate['cpu_ids'])
        resources=(dict(kind='EXPLICIT_HARDWARE_THREAD_SET',target=snapshot,cpu_ids=list(cpus),
            thread_refs=[row['thread']['target']for row in bindings],
            original_fact_details_plan_pointer='#/hardware_bindings',global_exclusive_reservation_proven=False),)
        base=_dependency_graph(pipeline,memory,certificate['original_call_predecessors'],certificate['call_cpu_ids'],resources)
        yield reference,captured,probe,metadata,certificate,base,target
    read_artifact(reference)


def _derive(path):
    from scar.analysis.pipeline_host_samples import summarize_host_samples
    from scar.analysis.pipeline_host_envelope import derive_pipeline_host_envelope
    from scar.analysis.pipeline_host_resources import derive_pipeline_host_resources
    from scar.optimization.pipeline_host_costs import derive_pipeline_host_costs
    from scar.optimization.hardware import _target_materials
    with _inputs(path)as(reference,captured,probe,native,certificate,base,target):
        refs=(dict(artifact=reference,role='BOUNDED_ACTUAL_NATIVE_HOST_OBSERVATIONS'),
            dict(artifact=captured['probe'],role='INVERSE_VERIFIED_SOURCE_CUT_LAYOUT'),
            dict(artifact=probe['original_artifact'],role='ORIGINAL_SOURCE_QUALIFIED_CPP_DELIVERY'))
        semantic=derive_pipeline_host_envelope(base,certificate,probe,source_refs=refs)
        observed=derive_pipeline_host_envelope(base,certificate,probe,trace=captured['samples'][0]['trace'],source_refs=refs)
        statistics=summarize_host_samples(captured)
        resources=derive_pipeline_host_resources(semantic,target,source_refs=refs)
        costs=derive_pipeline_host_costs(semantic,resources,captured,observations_ref=reference)
        inventory=_target_materials(Path(resources.target.uri).parent)[2]
        resource_summary=validate_resource_projection(resources,semantic,resources=inventory)
        cost_summary=validate_activity_cost_table(costs,semantic,resources)
        return reference,probe,base,semantic,observed,statistics,resources,costs,resource_summary,cost_summary


def _report(reference,probe,base,semantic,observed,stats,resources,costs,resource_summary,cost_summary,files):
    return dict(schema=SCHEMA,schema_version=1,observations=reference,probe_original_artifact=probe['original_artifact'],
        producer_pins=_pins(),files=files,
        original_tasks=len(base.tasks),original_values=len(base.values),original_materializations=len(base.materializations),original_slots=len(base.storage_slots),
        semantic_summary=validate_versioned_schedule(semantic),observed_summary=validate_versioned_schedule(observed),
        resource_summary=resource_summary,cost_summary=cost_summary,
        per_thread_intervals=len(stats['phases']),source_cpp_and_operator_math_reconstructed=True,
        target_program_executed_by_model_builder=False,library_loaded_by_model_builder=False,
        measurement_scope='INSTRUMENTED_SELECTED_NATIVE_REGION_EXCLUDING_PYTHON_BOUNDARY',
        solver_invoked=False,whole_scope_cost_closed=False,clean_performance_cost_admitted=False,
        physical_resource_feasibility_proven=False,deployment_authorized=False)


def prepare(observations,out):
    derived=_derive(observations)
    reference,probe,base,semantic,observed,stats,resources,costs,resource_summary,cost_summary=derived
    output=Path(out).resolve()
    if output.exists():raise ValueError('fresh host model output required')
    output.mkdir(parents=True)
    files=dict(base=_write(output/'base.json',schedule_document(base)),semantic=_write(output/'semantic.json',schedule_document(semantic)),
        observed=_write(output/'observed.json',schedule_document(observed)),statistics=_write(output/'statistics.json',stats),
        resources=_write(output/'resources.json',projection_document(resources)),
        costs=_write(output/'costs.json',activity_cost_document(costs)))
    result=_report(*derived,files)
    _write(output/'result.json',result);return result


def check(path):
    report=read_artifact(artifact_reference(Path(path).resolve(),SCHEMA))
    derived=_derive(report['observations']['uri'])
    reference,probe,base,semantic,observed,stats,resources,costs,resource_summary,cost_summary=derived
    values=dict(base=schedule_document(base),semantic=schedule_document(semantic),observed=schedule_document(observed),statistics=stats,
        resources=projection_document(resources),costs=activity_cost_document(costs))
    if any(read_artifact(report['files'][name])!=value for name,value in values.items()):
        raise ValueError('persisted source/host graph differs from independent replay')
    if report!=_report(*derived,report['files']):
        raise ValueError('host graph summary/source admission differs')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--observations');group.add_argument('--check');parser.add_argument('--out');args=parser.parse_args(argv)
    if args.check:result=check(args.check)
    elif not args.out:parser.error('--out required')
    else:result=prepare(args.observations,args.out)
    print(json.dumps({k:result[k]for k in ('original_tasks','original_values','original_slots','per_thread_intervals','whole_scope_cost_closed')}));return 0


if __name__=='__main__':raise SystemExit(main())
