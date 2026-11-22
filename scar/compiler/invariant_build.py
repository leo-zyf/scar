"""Original program loop proofs -> resource schedule -> selected C++ host."""
from dataclasses import asdict
import json
from pathlib import Path


def compile_invariant_build(document,out,*,loop_trip_count=None,loop_inputs=None,plan_only=False,loop_cpu_threads=None,
        loop_cpu_model=False,loop_calibrations=(),loop_no_op_profiles=None,loop_max_threads=None):
    from scar.analysis.build import BuildSession
    from scar.compiler.loop_admission import read_invariant_loop_motion, validate_invariant_loop_links
    from scar.compiler.invariant_export import export_temporal_invariant
    from scar.compiler.temporal_invariant_loop import build_temporal_invariant_loop
    from scar.compiler.model import CpuPlacement
    from scar.ir.execution_recipe import read_artifact,artifact_reference
    from scar.optimization.invariant_loop_schedule import plan_invariant_loop_schedule,verify_invariant_loop_schedule
    from scar.optimization.resource_solver import semantic_sha256
    specs=document.get('invariant_loops',[])
    if not specs:return []
    supplied={} if loop_inputs is None else loop_inputs
    no_op_profiles={} if loop_no_op_profiles is None else loop_no_op_profiles
    if set(supplied)-{row['id'] for row in specs}:raise ValueError('loop inputs refer to an unknown original loop')
    if set(no_op_profiles)-{row['id'] for row in specs}:raise ValueError('loop original costs refer to an unknown original loop')
    if not loop_cpu_model and (loop_calibrations or no_op_profiles or loop_max_threads is not None):
        raise ValueError('loop resource costs and model limit require --loop-cpu-model')
    if loop_cpu_model and loop_cpu_threads is not None:
        raise ValueError('cost-aware CPU selection and an explicit preview are separate modes')
    target=Path(document['hardware']['path']).parent;out=Path(out);out.mkdir()
    rows=[]
    with BuildSession(Path(document['program']['uri']).parent) as session:
        for index,spec in enumerate(specs):
            directory=out/('loop-'+str(index));directory.mkdir()
            read_artifact(spec['proof'])
            proof,source_motion=read_invariant_loop_motion(spec['proof']['uri'],catalog=session.view)
            if spec['id']!=proof.id or spec['hardware']!=document['hardware']:
                raise ValueError('loop proof or hardware differs from the retained program domain')
            validate_invariant_loop_links(proof,spec,document['loop_dependencies'])
            inputs=supplied.get(spec['id'],{})
            shape=inputs.get('input_shapes',spec.get('input_shapes'))
            count=inputs.get('trip_count',loop_trip_count)
            row=dict(id=proof.id,proof=spec['proof'],source_loop_authority='ORIGINAL_F4_REPLAYED',
                native_directory=None,selected_for_use=False,whole_program_compiled=False)
            motion_path=directory/'source-motion.json'
            motion_path.write_text(json.dumps(source_motion,indent=2,sort_keys=True)+'\n')
            row['source_motion']=artifact_reference(motion_path,source_motion['schema'])
            legality=source_motion['movement_legality']
            if legality['status']!='CONDITIONAL':
                row.update(status='ORIGINAL_LOOP_RETAINED',selected_kind='NO_OP',
                    selection_scope='SOURCE_MOVEMENT_LEGALITY',
                    source_legality_status=legality['status'],
                    reason='original source window does not justify all required invariant placements')
                rows.append(row);continue
            if shape is None or count is None:
                row.update(status='LOOP_INPUT_CONSTRAINTS_REQUIRED',missing=[name for name,value in
                    (('input_shapes',shape),('trip_count',count)) if value is None]);rows.append(row);continue
            budget=inputs.get('memory_budget_bytes',spec.get('memory_budget_bytes'))
            if budget is None:raise ValueError('loop requires retained incremental memory budget')
            threads=inputs.get('cpu_threads',loop_cpu_threads)
            if loop_cpu_model:
                if threads is not None:raise ValueError('declared loop CPU preview conflicts with automatic cost selection')
                from scar.optimization.temporal_cpu_choices import (build_temporal_cpu_domain,
                    plan_selected_temporal_cpu_choice,verify_selected_temporal_cpu_choice)
                domain=build_temporal_cpu_domain(proof,target,input_shapes=shape,trip_count=count,
                    memory_budget_bytes=budget,max_threads=loop_max_threads)
                domain_path=directory/'cpu-domain.json'
                domain_path.write_text(json.dumps(domain,indent=2,sort_keys=True)+'\n')
                row['cpu_domain']=artifact_reference(domain_path,domain['schema'])
                from scar.compiler.temporal_cost_inputs import project_loop_calibrations
                facts,cost_inputs=project_loop_calibrations(domain,loop_calibrations,directory/'resource-facts')
                row['cost_inputs']=cost_inputs
                selection=plan_selected_temporal_cpu_choice(domain,resource_facts=facts,
                    no_op_profile=no_op_profiles.get(proof.id))
                choice=selection['choice'];schedule=selection['schedule']
                choice_path=directory/'cpu-choice.json'
                choice_path.write_text(json.dumps(choice,indent=2,sort_keys=True)+'\n')
                row.update(cpu_choice=artifact_reference(choice_path,choice['schema']),
                    selection_scope='SOURCED_FORWARD_CPU_COMPONENT_MODEL',
                    candidate_count=len(domain['candidates']),
                    complete_cost_profiles=sum(r['model']['complete'] for r in choice['variants']),
                    cost_domain_complete=choice['complete_cost_domain'],
                    original_no_op_cost_known=choice['original_no_op']['complete'],
                    physical_optimality_proven=False,performance_qualified=False)
                if schedule is None:
                    row.update(status='ORIGINAL_LOOP_RETAINED',selected_kind='NO_OP',
                        cost_status=choice['status'],reason='complete forward cost domain or profitable model winner unavailable')
                    rows.append(row);continue
                verify_selected_temporal_cpu_choice(proof,target,schedule)
            elif threads is None:
                schedule=plan_invariant_loop_schedule(proof,target,input_shapes=shape,trip_count=count,
                    memory_budget_bytes=budget)
                verify_invariant_loop_schedule(proof,target,schedule)
            else:
                from scar.optimization.temporal_cpu_choices import (build_temporal_cpu_domain,
                    preview_temporal_cpu_choice,verify_temporal_cpu_preview)
                domain=build_temporal_cpu_domain(proof,target,input_shapes=shape,trip_count=count,
                    memory_budget_bytes=budget,max_threads=threads)
                domain_path=directory/'cpu-domain.json'
                domain_path.write_text(json.dumps(domain,indent=2,sort_keys=True)+'\n')
                row['cpu_domain']=artifact_reference(domain_path,domain['schema'])
                schedule=preview_temporal_cpu_choice(domain,cpu_threads=threads)
                verify_temporal_cpu_preview(proof,target,schedule)
            effective_budget=schedule['effective_incremental_budget_bytes']
            path=directory/'schedule.json';path.write_text(json.dumps(schedule,indent=2,sort_keys=True)+'\n')
            row.update(schedule=artifact_reference(path,schedule['schema']),selected_kind=schedule['selected_kind'],
                profitability_status=schedule['profitability_status'],input_shapes=shape,trip_count=count,
                memory_budget_bytes=effective_budget,requested_memory_budget_bytes=budget,
                selection_scope='STRUCTURAL_LOGICAL_WORK_AND_RESOURCE_MODEL',
                physical_optimality_proven=False,performance_qualified=False)
            if threads is not None:row['selection_scope']='EXPLICIT_CPU_TEAM_PREVIEW_UNRANKED'
            if loop_cpu_model:row['selection_scope']='SOURCED_FORWARD_CPU_COMPONENT_MODEL'
            if schedule['selected_kind']!='REGISTER_TEMPORAL_FUSION':
                row.update(status='ORIGINAL_LOOP_RETAINED');rows.append(row);continue
            binding=dict(schedule['resource_binding']);binding['cpu_ids']=tuple(binding['cpu_ids'])
            binding['evidence']=tuple(binding.get('evidence',()))
            interface=dict(original_f4_proof=asdict(proof),carry_parameter=proof.carry_parameter,
                readonly_parameters=[p for p in proof.body.parameters if p!=proof.carry_parameter],
                authority='ORIGINAL_F4_REPLAYED_WITH_RESOURCE_SCHEDULE',
                execution_specialization=dict(input_shapes=shape,trip_count=count,input_layout='C_CONTIGUOUS'),
                source_motion=row['source_motion'])
            # A stored report is not authority to cross a source window. Replay
            # original F4 again after planning, before C0 or any compilation.
            replayed,fresh_motion=read_invariant_loop_motion(spec['proof']['uri'],catalog=session.view)
            if (semantic_sha256(asdict(replayed))!=semantic_sha256(asdict(proof))
                    or semantic_sha256(fresh_motion)!=semantic_sha256(read_artifact(row['source_motion']))):
                raise ValueError('source movement changed between original F4 admission and lowering')
            builder=build_temporal_invariant_loop
            from scar.compiler.temporal_invariant_loop import _plan as preview_plan
            if threads is not None or loop_cpu_model:
                from scar.compiler.parallel_temporal_invariant import build_parallel_temporal_invariant_loop,_plan as preview_plan
                builder=build_parallel_temporal_invariant_loop
            preview,_=preview_plan(proof.body,proof.carry_parameter,interface,effective_budget,CpuPlacement(**binding))
            consumed=preview['execution_movement_binding']
            if consumed is None:raise ValueError('main invariant build requires consumed source movement legality')
            movement_path=directory/'movement-binding.json'
            movement_path.write_text(json.dumps(consumed,indent=2,sort_keys=True)+'\n')
            row['movement_binding']=artifact_reference(movement_path,consumed['schema'])
            if plan_only:row.update(status='MODEL_SELECTED_LOOP_PLAN_ONLY');rows.append(row);continue
            artifact=builder(proof.body,directory/'native',carry_parameter=proof.carry_parameter,
                loop_interface=interface,memory_budget_bytes=effective_budget,resource_binding=CpuPlacement(**binding))
            if artifact.plan['execution_movement_binding']!=consumed:
                raise ValueError('compiled movement binding differs from the selected source plan')
            exported=export_temporal_invariant(directory/'native',directory/'bundle',schedule_path=path,
                proof_reference=spec['proof'],catalog=session.view)
            row.update(status='MODEL_SELECTED_LOOP_COMPILED',native_directory=str(directory/'native'),
                native_artifact=artifact_reference(directory/'native/artifact.json',artifact.report['schema']),
                bundle=exported['bundle'],bundle_export=artifact_reference(directory/'bundle/export-report.json',exported['schema']),
                actual_target_executed=False)
            rows.append(row)
    return rows
