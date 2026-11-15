"""Build, estimate and lower the same versioned macro execution recipe."""
import argparse
import json
from pathlib import Path


def _write(path, value):
    path = Path(path).absolute(); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='scar recipe', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    plan = commands.add_parser('plan-loop', help='project one admitted shared loop into a macro recipe')
    plan.add_argument('--loop', required=True); plan.add_argument('--strategy', required=True)
    plan.add_argument('--plan', required=True); plan.add_argument('--layout')
    plan.add_argument('--cpu-hierarchy'); plan.add_argument('--out', required=True)
    stateful = commands.add_parser('plan-stateful', help='project one verified transactional array call into a macro recipe')
    stateful.add_argument('--region', required=True); stateful.add_argument('--plan', required=True)
    stateful.add_argument('--cpu-hierarchy'); stateful.add_argument('--out', required=True)
    estimate = commands.add_parser('estimate', help='estimate without mutating a recipe or executing its target')
    estimate.add_argument('--recipe', required=True); estimate.add_argument('--worker-profile')
    estimate.add_argument('--task-costs', help='sourced direct task costs for the stateful recipe profile')
    estimate.add_argument('--out', required=True)
    lower = commands.add_parser('lower', help='lower exactly the recipe consumed by estimation')
    lower.add_argument('--recipe', required=True); lower.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if args.command == 'plan-loop':
        from scar.compiler.loop_model import read_loop
        from scar.compiler.model import read_plan
        from scar.optimization.loop_recipe import make_loop_recipe
        strategy = json.loads(Path(args.strategy).read_bytes())
        layout = None if args.layout is None else json.loads(Path(args.layout).read_bytes())
        recipe = make_loop_recipe(read_loop(args.loop), strategy, read_plan(args.plan),
            layout=layout, cpu_hierarchy=args.cpu_hierarchy, out=args.out)
        result = dict(recipe_id=recipe['recipe_id'], profile=recipe['profile'],
                      recipe=str(Path(args.out).absolute()/'recipe.json'), native_compiled=False)
    elif args.command == 'plan-stateful':
        from scar.optimization.stateful_recipe import make_stateful_recipe
        recipe = make_stateful_recipe(args.region, args.plan, cpu_hierarchy=args.cpu_hierarchy, out=args.out)
        result = dict(recipe_id=recipe['recipe_id'], profile=recipe['profile'],
                      recipe=str(Path(args.out).absolute()/'recipe.json'), native_compiled=False)
    elif args.command == 'estimate':
        from scar.ir.execution_recipe import read_recipe
        recipe = read_recipe(args.recipe)
        if recipe['profile'] == 'cpu-temporal-loop-v1':
            from scar.optimization.loop_recipe import estimate_loop_recipe
            if args.task_costs: parser.error('--task-costs is only supported by the stateful recipe profile')
            value = estimate_loop_recipe(recipe, worker_profile=args.worker_profile)
        elif recipe['profile'] == 'cpu-stateful-array-region-v1':
            from scar.optimization.stateful_recipe import estimate_stateful_recipe
            if args.worker_profile: parser.error('--worker-profile is only supported by the temporal loop recipe profile')
            value = estimate_stateful_recipe(recipe, task_costs=args.task_costs)
        else: parser.error('unsupported execution recipe profile')
        _write(args.out, value)
        result = dict(recipe_id=value['recipe_id'], estimate=str(Path(args.out).absolute()),
                      status=value.get('status', value.get('timing', {}).get('status', 'UNSPECIFIED')),
                      target_executed=False)
    else:
        from scar.codegen.recipe import build_recipe
        artifact = build_recipe(args.recipe, args.out)
        _write(Path(args.out)/'result.json', artifact.report)
        result = dict(recipe_id=artifact.report['recipe_id'], native_directory=str(artifact.native_directory),
                      build_id=artifact.report['build_id'], selected_for_deployment=False)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
