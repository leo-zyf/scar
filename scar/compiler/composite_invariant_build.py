"""Unified program composite proofs -> retained hardware -> C++20 export."""
from dataclasses import asdict
import json
from pathlib import Path


def compile_composite_invariant_build(document,out,*,loop_trip_count=None,loop_inputs=None,
        plan_only=False,loop_cpu_threads=None,cpu_cost_model=False,loop_calibrations=(),
        loop_no_op_profiles=None,loop_max_threads=None):
    from scar.analysis.build import BuildSession,_check
    from scar.compiler.pipeline_loop_frontend import (read_composite_invariant_loop_proof,
        validate_composite_invariant_loop_links)
    from scar.compiler.composite_invariant_loop import build_composite_invariant_loop
    from scar.compiler.model import CpuPlacement
    from scar.ir.execution_recipe import artifact_reference,read_artifact
    from scar.optimization.composite_invariant_schedule import (plan_composite_invariant_loop_schedule,
        verify_composite_invariant_loop_schedule)
    from scar.optimization.resource_solver import semantic_sha256
    specs=document.get('composite_invariant_loops',[])
    if not specs:return []
    supplied={} if loop_inputs is None else loop_inputs
    no_op_profiles={} if loop_no_op_profiles is None else loop_no_op_profiles
    if set(supplied)-{spec['id'] for spec in specs}:raise ValueError('composite inputs name an unknown original loop')
    if set(no_op_profiles)-{spec['id'] for spec in specs}:raise ValueError('composite original costs name an unknown loop')
    if cpu_cost_model and loop_cpu_threads is not None:
        raise ValueError('composite cost selection and explicit CPU preview are separate modes')
    read_artifact(document['program']);_check(document['hardware'])
    target=Path(document['hardware']['path']).parent;out=Path(out);out.mkdir();rows=[]
    with BuildSession(Path(document['program']['uri']).parent) as session:
        for index,spec in enumerate(specs):
            directory=out/('loop-'+str(index));directory.mkdir()
            read_artifact(spec['proof'])
            proof=read_composite_invariant_loop_proof(spec['proof']['uri'],catalog=session.view)
            if spec['id']!=proof.id or spec['hardware']!=document['hardware']:
                raise ValueError('composite proof/hardware differs from retained program domain')
            validate_composite_invariant_loop_links(proof,spec,document['loop_dependencies'])
            inputs=supplied.get(proof.id,{})
            shape=inputs.get('input_shapes',spec.get('input_shapes'));count=inputs.get('trip_count',loop_trip_count)
            row=dict(id=proof.id,proof=spec['proof'],loop_links=spec['loop_links'],
                root_callee_source=asdict(proof.root_callee_source),body_identity=proof.body_identity,
                nested_global_guard_count=len(proof.nested_callee_guards),
                source_loop_authority='ORIGINAL_F4_COMPOSITION_AND_GLOBAL_ROUTES_REPLAYED',
                native_directory=None,selected_for_use=False,whole_program_compiled=False)
            if shape is None or count is None:
                row.update(status='COMPOSITE_LOOP_INPUT_CONSTRAINTS_REQUIRED',missing=[name for name,value in
                    (('input_shapes',shape),('trip_count',count)) if value is None]);rows.append(row);continue
            threads=inputs.get('cpu_threads',loop_cpu_threads)
            budget=inputs.get('memory_budget_bytes',spec.get('memory_budget_bytes'))
            parallel=cpu_cost_model or threads is not None
            if parallel:
                from scar.optimization.temporal_cpu_choices import (build_composite_temporal_cpu_domain,
                    preview_composite_temporal_cpu_choice,verify_composite_temporal_cpu_preview,
                    plan_selected_composite_temporal_cpu_choice,verify_selected_composite_temporal_cpu_choice)
                if cpu_cost_model and threads is not None:
                    raise ValueError('composite loop inputs request a preview inside automatic cost selection')
                domain=build_composite_temporal_cpu_domain(proof,target,catalog=session.view,
                    input_shapes=shape,trip_count=count,memory_budget_bytes=budget,
                    max_threads=loop_max_threads if cpu_cost_model else threads)
                domain_path=directory/'cpu-domain.json'
                domain_path.write_text(json.dumps(domain,indent=2,sort_keys=True)+'\n')
                row['cpu_domain']=artifact_reference(domain_path,domain['schema'])
                guard_path=directory/'guard-profile.json'
                guard_path.write_text(json.dumps(domain['guard_profile'],indent=2,sort_keys=True)+'\n')
                row['guard_profile']=artifact_reference(guard_path,domain['guard_profile']['schema'])
                if cpu_cost_model:
                    from scar.compiler.temporal_cost_inputs import project_loop_calibrations
                    facts,cost_inputs=project_loop_calibrations(domain,loop_calibrations,directory/'resource-facts')
                    row['cost_inputs']=cost_inputs
                    selected=plan_selected_composite_temporal_cpu_choice(domain,resource_facts=facts,
                        no_op_profile=no_op_profiles.get(proof.id))
                    choice=selected['choice'];schedule=selected['schedule']
                    path=directory/'cpu-choice.json'
                    path.write_text(json.dumps(choice,indent=2,sort_keys=True)+'\n')
                    row.update(cpu_choice=artifact_reference(path,choice['schema']),
                        candidate_count=len(domain['candidates']),
                        complete_cost_profiles=sum(value['model']['complete'] for value in choice['variants']),
                        cost_domain_complete=choice['complete_cost_domain'],
                        original_no_op_cost_known=choice['original_no_op']['complete'],
                        selection_scope='SOURCED_COMPOSITE_FORWARD_CPU_COMPONENT_MODEL',
                        physical_optimality_proven=False,performance_qualified=False)
                    if schedule is None:
                        row.update(status='ORIGINAL_COMPOSITE_LOOP_RETAINED',selected_kind='NO_OP',
                            cost_status=choice['status'],reason='complete composite guard/compute/original cost domain unavailable')
                        rows.append(row);continue
                    verify_selected_composite_temporal_cpu_choice(proof,target,schedule,catalog=session.view)
                else:
                    schedule=preview_composite_temporal_cpu_choice(domain,cpu_threads=threads)
                    verify_composite_temporal_cpu_preview(proof,target,schedule,catalog=session.view)
            else:
                schedule=plan_composite_invariant_loop_schedule(proof,target,catalog=session.view,
                    input_shapes=shape,trip_count=count,memory_budget_bytes=budget)
                verify_composite_invariant_loop_schedule(proof,target,schedule,catalog=session.view)
            snapshot=schedule['target_snapshot'];hardware=document['hardware']
            if (snapshot['uri'],snapshot['sha256'],snapshot['size_bytes'])!=(hardware['path'],hardware['sha256'],hardware['bytes']):
                raise ValueError('composite schedule differs from the original full hardware snapshot')
            schedule_path=directory/'schedule.json'
            schedule_path.write_text(json.dumps(schedule,indent=2,sort_keys=True)+'\n')
            effective=schedule['effective_incremental_budget_bytes']
            row.update(schedule=artifact_reference(schedule_path,schedule['schema']),
                selected_kind=schedule['selected_kind'],profitability_status=schedule['profitability_status'],
                input_shapes=shape,trip_count=count,memory_budget_bytes=effective,requested_memory_budget_bytes=budget,
                selection_scope=('SOURCED_COMPOSITE_FORWARD_CPU_COMPONENT_MODEL' if cpu_cost_model else
                    'EXPLICIT_COMPOSITE_CPU_TEAM_PREVIEW_UNRANKED' if parallel else schedule['comparison_scope']),
                performance_qualified=False,physical_optimality_proven=False)
            if schedule['selected_kind']!='REGISTER_TEMPORAL_FUSION':
                row.update(status='ORIGINAL_COMPOSITE_LOOP_RETAINED');rows.append(row);continue
            if plan_only:row.update(status='MODEL_SELECTED_COMPOSITE_LOOP_PLAN_ONLY');rows.append(row);continue
            binding=dict(schedule['resource_binding']);binding['cpu_ids']=tuple(binding['cpu_ids'])
            binding['evidence']=tuple(binding.get('evidence',()))
            specialization=dict(input_shapes=shape,trip_count=count,input_layout='C_CONTIGUOUS')
            artifact=build_composite_invariant_loop(proof,directory/'native',catalog=session.view,
                memory_budget_bytes=effective,resource_binding=CpuPlacement(**binding),
                execution_specialization=specialization,parallel=True if parallel else None)
            if (artifact.plan['loop_interface']['execution_specialization']!=specialization
                    or artifact.plan['memory_budget_bytes']!=effective
                    or semantic_sha256(artifact.plan['execution_plan']['resource_binding'])!=semantic_sha256(schedule['resource_binding'])):
                raise ValueError('composite native specialization differs from selected resource schedule')
            from scar.compiler.export_composite_temporal_invariant import export_composite_temporal_invariant
            exported=export_composite_temporal_invariant(directory/'native',directory/'bundle',
                schedule_path=schedule_path,proof_reference=spec['proof'],catalog=session.view)
            row.update(status='MODEL_SELECTED_COMPOSITE_LOOP_COMPILED',native_directory=str(directory/'native'),
                native_artifact=artifact_reference(directory/'native/artifact.json',artifact.report['schema']),
                bundle=exported['bundle'],bundle_export=artifact_reference(directory/'bundle/export-report.json',exported['schema']),
                actual_target_executed=False)
            rows.append(row)
    read_artifact(document['program']);_check(document['hardware'])
    return rows


__all__=['compile_composite_invariant_build']
