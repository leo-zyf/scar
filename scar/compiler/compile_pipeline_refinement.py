"""Compile a source-replayed finite-model compute-order certificate to C++20.

The existing host realizes lane order and dependencies. Model allocation,
release timestamps and makespan are not implemented as a runtime timetable.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.resource_solver import semantic_sha256


def compile_pipeline_refinement(certificate, out):
    from scar.optimization.pipeline_execution_refinement import _selected, SCHEMA as CERTIFICATE
    from scar.optimization.hardware import propose_cpu_placements,bind_cpu_placement
    from scar.optimization.schedule import build_region_schedule
    from scar.compiler.model import document
    from scar.codegen import build as build_leaf
    from scar.codegen.pipeline import build as build_pipeline
    from scar.analysis.build import BuildSession
    from scar.compiler.caller_pipeline_runtime import export_caller_pipeline_runtime
    reference = artifact_reference(Path(certificate).resolve(),CERTIFICATE)
    saved = read_artifact(reference); output = Path(out).resolve()
    if output.exists():
        raise ValueError('fresh refined pipeline output required')
    with _selected(saved['original_domain'],saved['selected_case_id']) as (
            domain_ref, selected, refs, pipeline, memory, ready, target, temporal):
        # Reject an edited certificate before compiling any operator, rather
        # than trusting a supplied lane list or source/model approval flag.
        from scar.optimization.pipeline_execution_refinement import _derive
        expected = _derive(pipeline,memory,ready,temporal,domain_ref,selected,refs)
        if semantic_sha256(saved) != semantic_sha256(expected):
            raise ValueError('certificate differs from independently replayed source/model selection')
        output.mkdir(parents=True)
        operators = {}; shared = {}
        for call in pipeline.calls:
            placements = propose_cpu_placements(call.plan,target,
                memory_budget_bytes=memory['memory_budget_bytes'],max_threads=len(saved['cpu_ids']))
            seed = next((row for row in placements if set(saved['cpu_ids']) <= set(row.cpu_ids)),None)
            if seed is None:
                raise ValueError('selected model CPU lanes lack an original feasible executor')
            binding = replace(seed,cpu_ids=(saved['call_cpu_ids'][call.id],))
            plan = bind_cpu_placement(call.plan,binding)
            key = semantic_sha256(document(plan))
            if key not in shared:
                folder = output/'operators'/('operator-'+str(len(shared)))
                shared[key] = build_leaf(plan,folder,resource_schedule=build_region_schedule(plan,target))
            operators[call.id] = shared[key]
        native = build_pipeline(pipeline,output/'native',memory_plan=memory,
            operators=operators,execution_schedule=reference['uri'])
        inputs = read_artifact(refs['inputs'])
        with BuildSession(Path(inputs['program']['uri']).parent) as session:
            provider = session.session.providers[inputs['provider']]
            export = export_caller_pipeline_runtime(native.directory,output/'bundle',
                proof_reference=refs['original_proof'],view=session.view,provider=provider)
        read_artifact(reference)
    report = dict(schema='scar.refined-pipeline-compiler-build',schema_version=1,
        certificate=reference,original_domain=domain_ref,selected_case_id=selected,
        native_artifact=artifact_reference(output/'native/artifact.json','scar.native-array-pipeline'),
        bundle=export['bundle'],expected_admission_sha256=export['expected_admission_sha256'],
        export=artifact_reference(output/'bundle/export-report.json','scar.caller-pipeline-runtime-export'),
        execution_family=saved['execution_family'],cpu_ids=saved['cpu_ids'],lane_call_order=saved['lane_call_order'],
        unique_operator_artifacts=len(shared),cpp_standard=20,operator_abi_version=1,
        implementation_scope='MODEL_SOLUTION_COMPUTE_ORDER_ONLY',full_model_timeline_implemented=False,
        model_makespan_is_native_runtime_prediction=False,native_host_optimality_proven=False,
        native_compiled=True,target_program_executed=False,performance_measured=False,
        default_backend='NO_OP',automatic_deployment_authorized=False,whole_program_cpp=False)
    (output/'result.json').write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--certificate',required=True);parser.add_argument('--out',required=True)
    args = parser.parse_args(argv); report = compile_pipeline_refinement(args.certificate,args.out)
    print(json.dumps({key:report[key] for key in ('native_artifact','bundle','expected_admission_sha256',
        'implementation_scope','full_model_timeline_implemented','automatic_deployment_authorized')},sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
