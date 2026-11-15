"""Evaluate a sourced deployment cost gate before loading a native candidate."""
import argparse
import json
from pathlib import Path


def main():
    from scar.optimization.deployment_cost import plan_deployment,verify_deployment_plan
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',required=True);parser.add_argument('--context',required=True)
    parser.add_argument('--reuse-count',type=int,required=True)
    parser.add_argument('--cpu-ids',type=int,nargs='+',required=True)
    parser.add_argument('--memory-budget-bytes',type=int,required=True)
    parser.add_argument('--minimum-gain',type=float,default=.05)
    parser.add_argument('--out',required=True)
    args=parser.parse_args();path=Path(args.out)
    if path.exists():raise ValueError('fresh deployment cost report required')
    context=json.loads(Path(args.context).read_bytes())
    report=plan_deployment(args.profile,context=context,reuse_count=args.reuse_count,
        available_cpu_ids=args.cpu_ids,memory_budget_bytes=args.memory_budget_bytes,minimum_gain=args.minimum_gain)
    verify_deployment_plan(report)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(decision=report['decision'],status=report['status'],
        native_loaded=False,deployment_authorized=False,report=str(path))))


if __name__=='__main__':main()
