"""Original F4 shared loops -> temporal resource plan -> guarded C++20."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .loop_model import read_loop, write_loop, loop_digest


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def compile_loops(out, *, source=None, catalog=None, loop_path=None, input_metadata_path=None,
                  input_shapes=None, trip_count=None, target=None, cpu_threads=None,
                  memory_budget_bytes=None, simd_profile=None, plan_only=False):
    if loop_path is not None:
        if source is not None or catalog is not None:
            raise ValueError('read a loop plan or discover from original source/catalog')
        loops, rejected = [read_loop(loop_path)], []
    else:
        if source is None or catalog is None:
            raise ValueError('automatic loop discovery needs original source and F4 catalog')
        from .loop_frontend import discover_graph_loops
        loops, rejected = discover_graph_loops(catalog, source=source)
    out = Path(out).absolute()
    if out.exists(): raise ValueError('fresh loop compiler output required')
    out.mkdir(parents=True)
    rows = []
    for index, loop in enumerate(loops):
        folder = out / ('loop-' + str(index)); folder.mkdir()
        path = folder / 'loop.json'; write_loop(loop, path)
        row = dict(id=loop.id, source=asdict(loop.source), plan=str(path),
                   loop_sha256=loop_digest(loop), shared_body_count=1, selected_for_use=False)
        shapes, count, evidence = input_shapes, trip_count, None
        if input_metadata_path is not None:
            if shapes is not None or count is not None:
                raise ValueError('observed metadata and declared loop context are mutually exclusive')
            from .input_metadata import loop_context_from_metadata
            shapes, count, evidence = loop_context_from_metadata(input_metadata_path, loop)
        elif isinstance(shapes, (tuple, list)):
            shapes = {loop.state_parameter:list(shapes)}
        if shapes is None or count is None:
            row.update(status='ARRAY_SHAPE_AND_TRIP_COUNT_REQUIRED', target_program_executed=False)
            rows.append(row); continue
        from scar.optimization.loop_strategy import plan_loop_strategy, loop_execution_plan
        strategy = plan_loop_strategy(loop, input_shapes=shapes, trip_count=count,
                                      memory_budget_bytes=memory_budget_bytes)
        _write(folder / 'strategy.json', strategy)
        row.update(strategy_plan=str(folder / 'strategy.json'),
                   selected_strategy=strategy['selected_strategy'], input_context_evidence=evidence)
        if strategy['selected_strategy'] == 'NO_OP':
            row.update(status='NO_OP', target_program_executed=False)
            rows.append(row); continue
        if target is None or cpu_threads is None:
            row.update(status='CPU_PLANNING_REQUIRED', reason='explicit executor allocation; loop time uncalibrated')
            rows.append(row); continue
        from scar.optimization.hardware import propose_cpu_placements, bind_cpu_placement
        from scar.codegen.loop import build
        plan = loop_execution_plan(loop, strategy)
        bindings = [binding for binding in propose_cpu_placements(plan, target,
            memory_budget_bytes=memory_budget_bytes, max_threads=cpu_threads)
            if len(binding.cpu_ids) == cpu_threads]
        if not bindings: raise ValueError('no executor satisfies the requested loop CPU allocation')
        bound = bind_cpu_placement(plan, bindings[0])
        if plan_only:
            from .model import write_plan
            bound_path = folder / 'execution-plan.json'
            write_plan(bound, bound_path)
            row['bound_execution_plan'] = str(bound_path)
        layout = None
        if simd_profile is not None:
            from scar.optimization.loop_layout import plan_loop_layout
            layout = plan_loop_layout(loop, strategy, bound, hardware_profile=simd_profile)
            _write(folder / 'layout.json', layout)
            row['layout_plan'] = str(folder / 'layout.json')
            if layout['selected'] is None:
                row.update(status='SIMD_LAYOUT_UNRESOLVED', target_program_executed=False)
                rows.append(row); continue
            row['execution_layout'] = layout['selected']
        if plan_only:
            row.update(status='BOUND_PLAN_ONLY', target_program_executed=False, native_compiled=False)
            rows.append(row); continue
        artifact = build(loop, folder / 'native', strategy=strategy, execution_plan=bound, layout=layout)
        row.update(status='COMPILED_UNMEASURED', artifact=str(folder / 'native'), backend=artifact.report,
                   cpu_allocation_policy='CALLER_EXPLICIT_THREAD_COUNT_FIRST_FEASIBLE_EXECUTOR_UNRANKED')
        rows.append(row)
    result = dict(schema='scar.loop-compiler-build', schema_version=1, loops=rows, rejections=rejected,
        original_f4_discovery=loop_path is None, first_stage_complete=False, whole_program_compiled=False,
        selected_for_use=False, plan_only=plan_only, cost_scope='EXACT_PAYLOAD_AND_EFFECTIVE_SIMD_PROXY_TIME_UNCALIBRATED'
        if simd_profile is not None else 'EXACT_ELEMENTWISE_PAYLOAD_REPRESENTATION_TIME_UNCALIBRATED')
    _write(out / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument('--source'); origin.add_argument('--loop')
    parser.add_argument('--catalog'); parser.add_argument('--out', required=True)
    parser.add_argument('--inputs'); parser.add_argument('--shape', nargs=2, type=int)
    parser.add_argument('--steps', type=int); parser.add_argument('--target')
    parser.add_argument('--cpu-threads', type=int); parser.add_argument('--memory-budget-bytes', type=int)
    parser.add_argument('--simd-profile', help='sourced canonical SSE2 primitives and register model; no target trial ranking')
    parser.add_argument('--plan-only', action='store_true', help='emit the bound body plan for primitive calibration without compiling a target')
    args = parser.parse_args()
    result = compile_loops(args.out, source=args.source, catalog=args.catalog, loop_path=args.loop,
        input_metadata_path=args.inputs, input_shapes=args.shape, trip_count=args.steps,
        target=args.target, cpu_threads=args.cpu_threads, memory_budget_bytes=args.memory_budget_bytes,
        simd_profile=args.simd_profile, plan_only=args.plan_only)
    print(json.dumps(dict(loops=len(result['loops']), statuses=[row['status'] for row in result['loops']],
                         result=str(Path(args.out) / 'result.json'))))


if __name__ == '__main__': main()
