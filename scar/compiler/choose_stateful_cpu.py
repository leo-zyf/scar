"""Hardware-derived CPU planning for an original stateful graph.

This command creates or references one shared domain. It does not compile or
time target variants. Missing costs preserve the original execution choice.
"""
import argparse
import json
from pathlib import Path
import time


def choose_stateful_cpu(out, *, region=None, domain=None, target=None,
                        input_shapes=None, shape=None, memory_budget_bytes=None,
                        affinity=None, max_threads=None, cost_profiles=None,
                        no_op_profile=None, search_budget=100000):
    from scar.optimization.stateful_cpu_choices import (build_stateful_cpu_domain,
        plan_stateful_cpu_choices)
    if (region is None)==(domain is None):
        raise ValueError('choose one original stateful region or an existing domain')
    if shape is not None and input_shapes is not None:
        raise ValueError('choose common shape or named input shapes')
    if domain is not None and any(v is not None for v in (target,input_shapes,shape,
            memory_budget_bytes,affinity,max_threads)):
        raise ValueError('an existing domain retains its original hardware and input constraints')
    folder=Path(out).absolute()
    folder.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    if domain is None:
        from scar.compiler.stateful_model import read_stateful
        original=read_stateful(region)
        if shape is not None:
            input_shapes={row.id:list(shape) for row in original.inputs if row.kind=='array'}
        if input_shapes is None or memory_budget_bytes is None:
            raise ValueError('original array input shapes and explicit payload budget required')
        if target is None:
            from scar.analysis.target_graph import build_target_graph
            target=folder/'hardware'
            build_target_graph(target,paths=(original.source.path,))
        build_stateful_cpu_domain(region,target,input_shapes=input_shapes,
            memory_budget_bytes=memory_budget_bytes,affinity=affinity,
            max_threads=max_threads,out=folder/'domain')
        domain=folder/'domain/domain.json'
    else:
        domain=Path(domain).absolute()
        if domain.is_dir():domain=domain/'domain.json'
    choice=plan_stateful_cpu_choices(domain,cost_profiles=cost_profiles,
        no_op_profile=no_op_profile,search_budget=search_budget,out=folder/'choice')
    report=dict(schema='scar.stateful-cpu-planning',schema_version=1,
        domain=str(domain),choice=str(folder/'choice/choice.json'),
        status=choice['status'],decision=choice['decision'],selected_id=choice['selected_id'],
        candidate_count=len(choice['variants'])-1,
        target_variants_compiled=False,target_variants_executed=False,
        deployment_authorized=False,physical_optimality_proven=False,
        seconds=time.monotonic()-started)
    (folder/'result.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    origin=parser.add_mutually_exclusive_group(required=True)
    origin.add_argument('--region');origin.add_argument('--domain')
    shapes=parser.add_mutually_exclusive_group()
    shapes.add_argument('--shape',nargs=2,type=int)
    shapes.add_argument('--input-shapes',help='JSON mapping original array IDs to shapes')
    parser.add_argument('--target');parser.add_argument('--memory-budget-bytes',type=int)
    parser.add_argument('--affinity',help='comma-separated CPU IDs or ranges')
    parser.add_argument('--max-threads',type=int)
    parser.add_argument('--cost-profiles',help='JSON mapping implementation IDs to sourced profiles or profile paths')
    parser.add_argument('--no-op-profile',help='sourced original-execution cost profile')
    parser.add_argument('--search-budget',type=int,default=100000)
    parser.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    affinity=None
    if args.affinity is not None:
        from scar.analysis.target_inventory import _cpu_list
        affinity=_cpu_list(args.affinity)
        if not affinity:parser.error('affinity requires nonnegative CPU IDs/ranges')
    report=choose_stateful_cpu(args.out,region=args.region,domain=args.domain,target=args.target,
        shape=args.shape,input_shapes=None if args.input_shapes is None else json.loads(Path(args.input_shapes).read_bytes()),
        memory_budget_bytes=args.memory_budget_bytes,affinity=affinity,max_threads=args.max_threads,
        cost_profiles=None if args.cost_profiles is None else json.loads(Path(args.cost_profiles).read_bytes()),
        no_op_profile=args.no_op_profile,search_budget=args.search_budget)
    print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
