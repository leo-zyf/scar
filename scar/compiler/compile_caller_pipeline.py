"""Compile an original caller-matched graph through the existing C++20 host.

This is a validation preview. An optional compact runtime guards the admitted
callable boundary; compiling or exporting never authorizes deployment.
"""
import argparse
import json
from pathlib import Path

from scar.analysis.build import BuildSession
from scar.analysis.source_call_view import SourceFunctionCallProvider
from scar.compiler.caller_pipeline import (build_caller_pipeline,
    read_caller_pipeline_proof, lower_caller_pipeline)
from scar.ir.execution_recipe import artifact_reference


def compile_caller_pipeline(program, out, *, caller_unit=None, function_ordinal=None,
        provider_name=None, interface_ordinals=None, proof=None, input_shapes,
        target=None, cpu_threads=1, memory_budget_bytes, runtime_out=None):
    program=Path(program).resolve()
    if program.is_file():program=program.parent
    with BuildSession(program)as session:
        if session.report.get('backend')!='F4':
            raise ValueError('an admitted F4 program graph is required')
        providers=[(name,value)for name,value in session.session.providers.items()
            if type(value)is SourceFunctionCallProvider and
                (provider_name is None or name==provider_name)and
                (caller_unit is None or caller_unit in value.available_units)]
        if len(providers)!=1:
            raise ValueError('one original source-call provider required; select its exact --provider key')
        name,provider=providers[0]
        if proof is None:
            if caller_unit is None or function_ordinal is None:
                raise ValueError('original caller --unit and --function-ordinal required')
            selected=build_caller_pipeline(session.view,provider,caller_unit=caller_unit,
                function_ordinal=function_ordinal,interface_ordinals=interface_ordinals)
        else:
            if function_ordinal is not None or interface_ordinals is not None:
                raise ValueError('proof already specifies the original function and selected interfaces')
            selected=read_caller_pipeline_proof(proof,view=session.view,provider=provider)
            if caller_unit is not None and selected.caller_unit!=caller_unit:
                raise ValueError('proof caller differs from selected unit')
        if target is None:
            material=session.report.get('target')
            if material is None:raise ValueError('an admitted hardware snapshot or explicit --target is required')
            target=Path(material['path']).parent
        result=lower_caller_pipeline(selected,out,view=session.view,provider=provider,
            input_shapes=input_shapes,target=target,cpu_threads=cpu_threads,
            memory_budget_bytes=memory_budget_bytes)
        result['program']=artifact_reference(program/'manifest.json','scar.program-build')
        result['provider']=name
        if runtime_out is not None:
            from .caller_pipeline_runtime import export_caller_pipeline_runtime
            if result['cpp_artifacts_compiled']!=1:
                raise ValueError('runtime export requires one actually compiled caller host')
            result['runtime_bundle']=export_caller_pipeline_runtime(
                Path(out)/'compiled/pipeline-0/native',runtime_out,proof=selected,
                view=session.view,provider=provider,proof_reference=result['proof'])
            result['caller_matched_live_runtime_guard_adapter_implemented']=True
            result['runtime_guard_scope']='COMPACT_POOLED_HOST_VALIDATION_ONLY_COST_UNKNOWN'
        path=Path(out).resolve()/'result.json'
        path.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--unit');parser.add_argument('--function-ordinal',type=int)
    parser.add_argument('--provider');parser.add_argument('--interface-ordinal',type=int,action='append')
    parser.add_argument('--proof');parser.add_argument('--input-shapes',type=json.loads,required=True)
    parser.add_argument('--target');parser.add_argument('--cpu-threads',type=int,default=1)
    parser.add_argument('--runtime-out',help='export the same ELF files and a compact validation-only runtime contract')
    parser.add_argument('--memory-budget-bytes',type=int,required=True)
    args=parser.parse_args(argv)
    result=compile_caller_pipeline(args.build,args.out,caller_unit=args.unit,
        function_ordinal=args.function_ordinal,provider_name=args.provider,
        interface_ordinals=None if args.interface_ordinal is None else tuple(args.interface_ordinal),
        proof=args.proof,input_shapes=args.input_shapes,target=args.target,
        cpu_threads=args.cpu_threads,memory_budget_bytes=args.memory_budget_bytes,runtime_out=args.runtime_out)
    print(json.dumps(result,sort_keys=True,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
