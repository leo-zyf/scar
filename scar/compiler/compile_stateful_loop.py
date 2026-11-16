"""Original stateful loop -> one guarded step plan -> C++20 repeat delivery."""
import argparse
import json
from pathlib import Path

from scar.ir.execution_recipe import artifact_reference


def compile_stateful_loop(program,out,*,caller_unit,function_ordinal,interface_ordinal,
        provider_name=None,step_reference,input_shapes,trip_count,memory_budget_bytes,
        cpu_threads=1,runtime_out=None,plan_only=False):
    from scar.analysis.build import BuildSession
    from scar.analysis.source_call_view import SourceFunctionCallProvider
    from scar.compiler.stateful_loop_frontend import (build_stateful_loop,write_stateful_loop_proof,
        read_stateful_loop_proof,SCHEMA)
    from scar.optimization.stateful_loop_schedule import plan_stateful_loop,verify_stateful_loop
    out=Path(out).resolve();program=Path(program).resolve()
    if program.is_file():program=program.parent
    if out.exists():raise ValueError('fresh stateful loop compiler output required')
    if isinstance(step_reference,(str,Path)):
        step_reference=artifact_reference(step_reference,'scar.stateful-array-region')
    with BuildSession(program)as session:
        choices=[(name,provider)for name,provider in session.session.providers.items()
            if type(provider)is SourceFunctionCallProvider and caller_unit in provider.available_units
            and(provider_name is None or name==provider_name)]
        if len(choices)!=1:raise ValueError('select one original source-call provider for the loop')
        name,provider=choices[0]
        original=build_stateful_loop(session.view,provider,caller_unit=caller_unit,
            function_ordinal=function_ordinal,interface_ordinal=interface_ordinal,step_reference=step_reference)
        target=session.report.get('target')
        if target is None:raise ValueError('original program hardware snapshot required')
        hardware=Path(target['path']).parent
        out.mkdir(parents=True)
        proof_reference=write_stateful_loop_proof(original,out/'proof.json')
        plan=plan_stateful_loop(original,hardware,input_shapes=input_shapes,trip_count=trip_count,
            memory_budget_bytes=memory_budget_bytes,cpu_threads=cpu_threads)
        (out/'plan.json').write_text(json.dumps(plan,sort_keys=True,indent=2,allow_nan=False)+'\n')
        plan_reference=artifact_reference(out/'plan.json',plan['schema'])
        report=dict(schema='scar.stateful-loop-compiler-build',schema_version=1,
            program=artifact_reference(program/'manifest.json','scar.program-build'),
            provider=name,original_loop_proof=proof_reference,loop_plan=plan_reference,
            hardware=target,source_replayed=True,source_loop_body_instances=1,iterations_unrolled=False,
            cost_status='UNKNOWN',selected_for_use=False,deployment_authorized=False,
            workspace_or_team_setup_hoisted=False,native_math_changed=False,
            target_program_executed=False,performance_measured=False,whole_program_compiled=False)
        if plan['status']!='READY'or plan.get('frontier')or not plan['budget_satisfied']:
            report.update(status='STATEFUL_LOOP_PLAN_UNRESOLVED',native_compiled=False)
        elif plan_only:
            report.update(status='BOUND_PLAN_ONLY',native_compiled=False)
        else:
            # Independently consume the retained source proof and complete
            # per-step plan. A legal loop is not a measured best allocation.
            proof=read_stateful_loop_proof(proof_reference['uri'],view=session.view,provider=provider)
            verify_stateful_loop(proof,hardware,plan_reference['uri'])
            from scar.codegen.stateful import build
            from scar.compiler.stateful_repeat import build_repeat
            base=build(proof.step,out/'step-native',plan=plan['step_plan'])
            repeat=build_repeat(base.directory,out/'repeat-native')
            report.update(status='COMPILED_UNMEASURED',native_compiled=True,
                step_native_artifact=artifact_reference(base.directory/'artifact.json',base.report['schema']),
                repeat_native_artifact=artifact_reference(repeat.directory/'artifact.json',repeat.report['schema']),
                commit_policy=repeat.report['commit_policy'],failure_policy=repeat.report['failure_policy'])
            if runtime_out is not None:
                from scar.compiler.stateful_loop_runtime import export_stateful_loop_runtime
                report['runtime_bundle']=export_stateful_loop_runtime(base.directory,repeat.directory,runtime_out,
                    proof_reference=proof_reference,plan=plan_reference,view=session.view,provider=provider)
        (out/'result.json').write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
        return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--unit',required=True);parser.add_argument('--function-ordinal',type=int,required=True)
    parser.add_argument('--interface-ordinal',type=int,required=True);parser.add_argument('--provider')
    parser.add_argument('--step',required=True,help='retained source-bound stateful region JSON')
    parser.add_argument('--input-shapes',type=json.loads,required=True)
    parser.add_argument('--trip-count',type=int,required=True);parser.add_argument('--cpu-threads',type=int,default=1)
    parser.add_argument('--memory-budget-bytes',type=int,required=True)
    parser.add_argument('--runtime-out');parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args(argv)
    result=compile_stateful_loop(args.build,args.out,caller_unit=args.unit,function_ordinal=args.function_ordinal,
        interface_ordinal=args.interface_ordinal,provider_name=args.provider,step_reference=args.step,
        input_shapes=args.input_shapes,trip_count=args.trip_count,cpu_threads=args.cpu_threads,
        memory_budget_bytes=args.memory_budget_bytes,runtime_out=args.runtime_out,plan_only=args.plan_only)
    print(json.dumps(dict(status=result['status'],result=str(Path(args.out)/'result.json'))));return 0


if __name__=='__main__':raise SystemExit(main())
