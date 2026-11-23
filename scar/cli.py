"""Public stage commands. The driver is the only cross-stage caller."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path


def _positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def _source(parser):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="locked native compilation request")
    source.add_argument("--entry", type=Path, help="Python source file")
    parser.add_argument("--jobs", type=_positive, default=1, help="frontend worker count")


def _specialization(parser):
    parser.add_argument("--input-shapes", type=Path, help="JSON rank-two input shape for the Python array ABI")
    parser.add_argument("--trip-count", type=_positive)


def _constraints(args):
    shape = json.loads(args.input_shapes.read_bytes()) if args.input_shapes is not None else None
    return dict(input_shape=shape, trip_count=args.trip_count)


def _build(source, output, jobs):
    from .frontend import build

    graph = build(source, output, jobs=jobs)
    return dict(graph=str((output / "graph.json").resolve()), regions=len(graph.regions),
                operations=len(graph.operations), edges=len(graph.edges), resources=len(graph.resources))


def _plan(source, output, args):
    from .middle import plan

    selected = plan(source, output, **_constraints(args))
    return dict(plan=str((output / "plan.json").resolve()), decision=selected.decision,
                steps=len(selected.schedule), rewrites=len(selected.rewrites))


def _compile(source, output, args):
    from .backend import compile_plan

    return compile_plan(source, output, case_seconds=args.case_seconds, repetitions=args.repetitions,
                        warmups=args.warmups, cpu_threads=args.cpu_threads)


def _parser():
    parser = argparse.ArgumentParser(prog="scar", description="Program graph compiler for Linux CPU targets")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="publish a typed program and hardware graph")
    _source(build)
    build.add_argument("--out", required=True, type=Path)

    plan = commands.add_parser("plan", help="derive a sealed plan from a retained graph")
    plan.add_argument("--graph", required=True, type=Path)
    plan.add_argument("--out", required=True, type=Path)
    _specialization(plan)

    compile_ = commands.add_parser("compile", help="consume a sealed plan and build a C++20 project")
    compile_source = compile_.add_mutually_exclusive_group(required=True)
    compile_source.add_argument("--plan", type=Path)
    compile_source.add_argument("--manifest", type=Path, help="run the three stages on a native request")
    compile_source.add_argument("--entry", type=Path, help="run the three stages on a Python source")
    compile_.add_argument("--out", required=True, type=Path)
    compile_.add_argument("--jobs", type=_positive, default=1)
    compile_.add_argument("--case-seconds", type=_positive, default=30)
    compile_.add_argument("--repetitions", type=_positive, default=1)
    compile_.add_argument("--warmups", type=_positive, default=1)
    compile_.add_argument("--cpu-threads", type=_positive, help="limit available backend workers")
    _specialization(compile_)

    check = commands.add_parser("check-program", help="verify a persisted graph and print its identity")
    check.add_argument("--graph", required=True, type=Path)
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = _build(args.manifest or args.entry, args.out, args.jobs)
        elif args.command == "plan":
            result = _plan(args.graph, args.out, args)
        elif args.command == "compile":
            if args.plan is not None:
                if any(item is not None for item in (args.input_shapes, args.trip_count)):
                    parser.error("plan specialization belongs to scar plan")
                result = _compile(args.plan, args.out, args)
            else:
                args.out.mkdir(parents=True, exist_ok=False)
                graph, plan = args.out / "graph", args.out / "plan"
                _build(args.manifest or args.entry, graph, args.jobs)
                _plan(graph / "graph.json", plan, args)
                result = _compile(plan / "plan.json", args.out / "delivery", args)
        else:
            from .contract import open_graph

            reference, graph = open_graph(args.graph)
            result = dict(graph=asdict(reference), regions=len(graph.regions), operations=len(graph.operations))
    except (OSError, ValueError) as error:
        parser.exit(2, f"scar {args.command}: {error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return int(result.get("status") == "VALIDATION_FAILED")
