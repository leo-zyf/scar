"""Unified program composite proofs -> retained hardware -> C++20 export."""
from dataclasses import asdict
import json
from pathlib import Path


def compile_composite_invariant_build(document,out,*,loop_trip_count=None,loop_inputs=None,
        plan_only=False,loop_cpu_threads=None,cpu_cost_model=False):
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
    if set(supplied)-{spec['id'] for spec in specs}:raise ValueError('composite inputs name an unknown original loop')
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
            if cpu_cost_model:
                row.update(status='COMPOSITE_CPU_COST_MODEL_UNSUPPORTED',selected_kind='NO_OP',
                    reason='current source-independent CPU cost families admit direct loop proofs only')
                rows.append(row);continue
            if shape is None or count is None:
                row.update(status='COMPOSITE_LOOP_INPUT_CONSTRAINTS_REQUIRED',missing=[name for name,value in
                    (('input_shapes',shape),('trip_count',count)) if value is None]);rows.append(row);continue
            threads=inputs.get('cpu_threads',loop_cpu_threads)
            if threads is not None and (type(threads) is not int or threads!=1):
                row.update(status='COMPOSITE_CPU_PROFILE_UNSUPPORTED',requested_cpu_threads=threads,
                    reason='current composite lowering supports only one explicitly bound CPU');rows.append(row);continue
            budget=inputs.get('memory_budget_bytes',spec.get('memory_budget_bytes'))
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
                selection_scope=schedule['comparison_scope'],performance_qualified=False,physical_optimality_proven=False)
            if schedule['selected_kind']!='REGISTER_TEMPORAL_FUSION':
                row.update(status='ORIGINAL_COMPOSITE_LOOP_RETAINED');rows.append(row);continue
            if plan_only:row.update(status='MODEL_SELECTED_COMPOSITE_LOOP_PLAN_ONLY');rows.append(row);continue
            binding=dict(schedule['resource_binding']);binding['cpu_ids']=tuple(binding['cpu_ids'])
            binding['evidence']=tuple(binding.get('evidence',()))
            specialization=dict(input_shapes=shape,trip_count=count,input_layout='C_CONTIGUOUS')
            artifact=build_composite_invariant_loop(proof,directory/'native',catalog=session.view,
                memory_budget_bytes=effective,resource_binding=CpuPlacement(**binding),
                execution_specialization=specialization)
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
