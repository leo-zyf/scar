"""Independent public commands for building, planning and compiling a program."""

import argparse
import json
from pathlib import Path


def _workers(text):
    count = int(text)
    if count < 1:
        raise argparse.ArgumentTypeError("worker count must be positive")
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="scar", description="State-Control-Action-Resource compiler"
    )
    stages = parser.add_subparsers(dest="stage", required=True)
    front = stages.add_parser("build", help="build a typed program and platform graph")
    inputs = front.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", type=Path)
    inputs.add_argument("--entry", type=Path)
    front.add_argument("--jobs", type=_workers, default=1)
    mid = stages.add_parser(
        "plan", help="prove logical rewrites and publish execution IR"
    )
    mid.add_argument("--graph", type=Path, required=True)
    mid.add_argument(
        "--no-rewrite", action="store_true", help="retain the original operation order"
    )
    mid.add_argument(
        "--input-shapes", type=Path, help="dimension list for a binary array entry"
    )
    mid.add_argument("--trip-count", type=int)
    back = stages.add_parser(
        "compile", help="schedule the execution IR and generate C++20"
    )
    source = back.add_mutually_exclusive_group(required=True)
    source.add_argument("--ir", type=Path)
    source.add_argument("--plan", type=Path, help="directory containing exec-ir.json")
    back.add_argument("--physical", choices=("auto", "serial"), default="auto")
    back.add_argument("--cpu-threads", type=_workers)
    for stage in (front, mid, back):
        stage.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.stage == "build":
            from .frontend import build

            result = build(args.manifest or args.entry, args.out, jobs=args.jobs)
        elif args.stage == "plan":
            from .middle import plan

            shape = (
                json.loads(args.input_shapes.read_bytes())
                if args.input_shapes
                else None
            )
            result = plan(
                args.graph,
                args.out,
                optimize_graph=not args.no_rewrite,
                shape=shape,
                trip_count=args.trip_count,
            )
        else:
            from .backend import compile_ir

            path = args.ir or args.plan
            if args.plan and path.name == "plan.json":
                path = path.with_name("exec-ir.json")
            result = compile_ir(
                path, args.out, policy=args.physical, workers=args.cpu_threads
            )
    except (OSError, ValueError, ImportError) as error:
        parser.exit(2, f"scar {args.stage}: {error}\n")
    print(json.dumps(result, allow_nan=False, indent=2))
    return 2 if result.get("status") == "INCOMPLETE" else 0
