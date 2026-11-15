"""Original program loop proofs -> resource schedule -> selected C++ host."""
from dataclasses import asdict
import json
from pathlib import Path


def compile_invariant_build(document,out,*,loop_trip_count=None,loop_inputs=None,plan_only=False):
    from scar.analysis.build import BuildSession
    from scar.compiler.invariant_loop_frontend import read_invariant_loop_proof,validate_invariant_loop_links
    from scar.compiler.export_temporal_invariant import export_temporal_invariant
    from scar.compiler.temporal_invariant_loop import build_temporal_invariant_loop
    from scar.compiler.model import CpuPlacement
    from scar.ir.execution_recipe import read_artifact,artifact_reference
    from scar.optimization.invariant_loop_schedule import plan_invariant_loop_schedule,verify_invariant_loop_schedule
    specs=document.get('invariant_loops',[])
    if not specs:return []
    supplied={} if loop_inputs is None else loop_inputs
    if set(supplied)-{row['id'] for row in specs}:raise ValueError('loop inputs refer to an unknown original loop')
    target=Path(document['hardware']['path']).parent;out=Path(out);out.mkdir()
    rows=[]
    with BuildSession(Path(document['program']['uri']).parent) as session:
        for index,spec in enumerate(specs):
            directory=out/('loop-'+str(index));directory.mkdir()
            read_artifact(spec['proof'])
            proof=read_invariant_loop_proof(spec['proof']['uri'],catalog=session.view)
            if spec['id']!=proof.id or spec['hardware']!=document['hardware']:
                raise ValueError('loop proof or hardware differs from the retained program domain')
            validate_invariant_loop_links(proof,spec,document['loop_dependencies'])
            inputs=supplied.get(spec['id'],{})
            shape=inputs.get('input_shapes',spec.get('input_shapes'))
            count=inputs.get('trip_count',loop_trip_count)
            row=dict(id=proof.id,proof=spec['proof'],source_loop_authority='ORIGINAL_F4_REPLAYED',
                native_directory=None,selected_for_use=False,whole_program_compiled=False)
            if shape is None or count is None:
                row.update(status='LOOP_INPUT_CONSTRAINTS_REQUIRED',missing=[name for name,value in
                    (('input_shapes',shape),('trip_count',count)) if value is None]);rows.append(row);continue
            budget=inputs.get('memory_budget_bytes',spec.get('memory_budget_bytes'))
            if budget is None:raise ValueError('loop requires retained incremental memory budget')
            schedule=plan_invariant_loop_schedule(proof,target,input_shapes=shape,trip_count=count,
                memory_budget_bytes=budget)
            verify_invariant_loop_schedule(proof,target,schedule)
            effective_budget=schedule['effective_incremental_budget_bytes']
            path=directory/'schedule.json';path.write_text(json.dumps(schedule,indent=2,sort_keys=True)+'\n')
            row.update(schedule=artifact_reference(path,schedule['schema']),selected_kind=schedule['selected_kind'],
                profitability_status=schedule['profitability_status'],input_shapes=shape,trip_count=count,
                memory_budget_bytes=effective_budget,requested_memory_budget_bytes=budget,
                selection_scope='STRUCTURAL_LOGICAL_WORK_AND_RESOURCE_MODEL',
                physical_optimality_proven=False,performance_qualified=False)
            if schedule['selected_kind']!='REGISTER_TEMPORAL_FUSION':
                row.update(status='ORIGINAL_LOOP_RETAINED');rows.append(row);continue
            if plan_only:row.update(status='MODEL_SELECTED_LOOP_PLAN_ONLY');rows.append(row);continue
            binding=dict(schedule['resource_binding']);binding['cpu_ids']=tuple(binding['cpu_ids'])
            binding['evidence']=tuple(binding.get('evidence',()))
            interface=dict(original_f4_proof=asdict(proof),carry_parameter=proof.carry_parameter,
                readonly_parameters=[p for p in proof.body.parameters if p!=proof.carry_parameter],
                authority='ORIGINAL_F4_REPLAYED_WITH_RESOURCE_SCHEDULE',
                execution_specialization=dict(input_shapes=shape,trip_count=count,input_layout='C_CONTIGUOUS'))
            artifact=build_temporal_invariant_loop(proof.body,directory/'native',carry_parameter=proof.carry_parameter,
                loop_interface=interface,memory_budget_bytes=effective_budget,resource_binding=CpuPlacement(**binding))
            exported=export_temporal_invariant(directory/'native',directory/'bundle',schedule_path=path,
                proof_reference=spec['proof'],catalog=session.view)
            row.update(status='MODEL_SELECTED_LOOP_COMPILED',native_directory=str(directory/'native'),
                native_artifact=artifact_reference(directory/'native/artifact.json',artifact.report['schema']),
                bundle=exported['bundle'],bundle_export=artifact_reference(directory/'bundle/export-report.json',exported['schema']),
                actual_target_executed=False)
            rows.append(row)
    return rows
