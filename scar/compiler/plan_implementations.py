"""Construct a declared resource schedule; source producers own legality."""
import argparse
import json
from pathlib import Path

from scar.ir.implementation_plan import problem_from_dict, digest
from scar.ir.execution_recipe import artifact_reference, read_artifact
from scar.optimization.joint_planner import solve_implementations, verify_execution_plan


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem',required=True)
    parser.add_argument('--out');parser.add_argument('--check')
    parser.add_argument('--max-seconds',type=float,default=30)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--planner',choices=('constructive','solver'),default='constructive')
    parser.add_argument('--baseline-only',action='store_true')
    args=parser.parse_args(argv)
    problem_path=Path(args.problem).resolve()
    reference=artifact_reference(problem_path,'scar.implementation-planning-problem')
    problem=problem_from_dict(read_artifact(reference))
    if args.check:
        plan=json.loads(Path(args.check).read_bytes())
        print(json.dumps(verify_execution_plan(problem,plan),sort_keys=True));return 0
    if not args.out:parser.error('--out is required for planning')
    output=Path(args.out).resolve()
    if output.exists():parser.error('fresh output path required')
    if args.planner=='constructive':
        from scar.optimization.constructive_planner import construct_implementations
        plan=construct_implementations(problem,max_seconds=args.max_seconds,baseline_only=args.baseline_only)
    else:
        plan=solve_implementations(problem,max_seconds=args.max_seconds,workers=args.workers,baseline_only=args.baseline_only)
    # Pure solver input is a declared domain. A source-qualified producer must
    # independently replay original source/proof/backend facts before lowering.
    plan['input_problem']=reference
    plan['source_legality_replayed_by_this_command']=False
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(plan,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:plan[k]for k in ('status','makespan_ns','model_lower_bound_ns','domain_complete','relative_gap')},sort_keys=True))
    return 0


if __name__=='__main__':raise SystemExit(main())
