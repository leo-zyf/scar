"""Original F4 loop interfaces -> automatic invariant cut -> existing C++20."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time


def compile_invariant_loops(catalog,source,out,*,memory_budget_bytes):
    from scar.compiler.invariant_loop_frontend import discover_invariant_loops
    from scar.compiler.invariant_loop import build_invariant_loop
    out=Path(out).resolve()
    if out.exists():raise ValueError('fresh invariant loop compiler output required')
    started=time.monotonic();proofs,rejected=discover_invariant_loops(catalog,source)
    out.mkdir(parents=True);rows=[]
    for index,proof in enumerate(proofs):
        folder=out/('loop-'+str(index));folder.mkdir()
        interface=dict(original_f4_proof=asdict(proof),carry_parameter=proof.carry_parameter,
            readonly_parameters=[p for p in proof.body.parameters if p!=proof.carry_parameter],
            authority='ORIGINAL_F4_CALL_ARGUMENTS_BINDING_WRITE_AND_LOOP_CONTROL_REPLAY',
            runtime_admission_required=True)
        try:
            artifact=build_invariant_loop(proof.body,folder/'native',carry_parameter=proof.carry_parameter,
                loop_interface=interface,memory_budget_bytes=memory_budget_bytes)
            row=dict(id=proof.id,source=asdict(proof.source),status='COMPILED_FOR_VALIDATION',
                artifact=str(folder/'native'),hoisted_roots=artifact.plan['hoisted_roots'],
                invariant_operations=artifact.plan['invariant_arithmetic_nodes'],
                residual_operations=artifact.plan['residual_arithmetic_nodes'],
                source_loop_proof=asdict(proof),selected_for_use=False)
        except ValueError as error:
            row=dict(id=proof.id,status='NO_ADMISSIBLE_INVARIANT_CUT',reason=str(error),selected_for_use=False)
        rows.append(row)
    report=dict(schema='scar.invariant-loop-compilation',schema_version=1,source=str(Path(source).resolve()),
        catalog=str(getattr(catalog,'path',catalog)),loops=rows,rejections=rejected,
        source_analysis='ORIGINAL_F4_NOT_AST_MATCHING',invariant_selection='TRANSITIVE_INPUT_DEPENDENCY_CLOSURE',
        iterations_unrolled=False,target_variants_ranked=False,selected_for_use=False,
        whole_program_compiled=False,seconds=time.monotonic()-started)
    (out/'result.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',required=True);parser.add_argument('--source',required=True)
    parser.add_argument('--memory-budget-bytes',required=True,type=int);parser.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    report=compile_invariant_loops(args.catalog,args.source,args.out,memory_budget_bytes=args.memory_budget_bytes)
    print(json.dumps(dict(loops=len(report['loops']),statuses=[r['status'] for r in report['loops']],
        rejections=len(report['rejections']),seconds=report['seconds']),indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
