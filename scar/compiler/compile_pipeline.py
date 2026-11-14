"""Discover original F4 call pipelines, plan their buffers, and lower to C++20."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from .pipeline_model import read_pipeline, write_pipeline, pipeline_digest


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def compile_pipelines(out, *, source=None, catalog=None, pipeline_path=None, input_metadata_path=None,
                      input_shapes=None, target=None, cpu_threads=None, memory_budget_bytes=None,
                      search_budget=100000, strategy_mode='auto'):
    if strategy_mode not in ('auto', 'pool'):
        raise ValueError('pipeline strategy must be auto or pool')
    if pipeline_path is not None:
        if source is not None or catalog is not None:
            raise ValueError('read a pipeline plan or discover from source/catalog')
        pipelines, rejected = [read_pipeline(pipeline_path)], []
    else:
        if source is None or catalog is None:
            raise ValueError('automatic pipeline discovery needs original source and F4 catalog')
        from .pipeline_frontend import discover_graph_pipelines
        pipelines, rejected = discover_graph_pipelines(catalog, source=source)
    out = Path(out).absolute()
    if out.exists():
        raise ValueError('fresh pipeline compiler output required')
    out.mkdir(parents=True)
    rows = []
    for index, pipeline in enumerate(pipelines):
        folder = out / ('pipeline-' + str(index))
        folder.mkdir()
        path = folder / 'pipeline.json'
        write_pipeline(pipeline, path)
        row = dict(id=pipeline.id, source=asdict(pipeline.source), plan=str(path),
            pipeline_sha256=pipeline_digest(pipeline), calls=len(pipeline.calls), selected_for_use=False)
        shapes, shape_evidence = input_shapes, None
        if input_metadata_path is not None:
            if input_shapes is not None:
                raise ValueError('observed input metadata and declared shapes are mutually exclusive')
            from .input_metadata import shapes_from_metadata
            shapes, shape_evidence = shapes_from_metadata(input_metadata_path, pipeline.source, pipeline.parameters)
        if shapes is None:
            row.update(status='INPUT_SHAPES_REQUIRED', target_program_executed=False)
            rows.append(row)
            continue
        strategy = None
        if strategy_mode == 'auto':
            from scar.optimization.pipeline_strategy import plan_pipeline_strategy
            strategy = plan_pipeline_strategy(pipeline, input_shapes=shapes,
                memory_budget_bytes=memory_budget_bytes, search_budget=search_budget)
            strategy_path = folder / 'strategy.json'
            _write(strategy_path, strategy)
            row.update(strategy_plan=str(strategy_path), selected_strategy=strategy['selected_strategy'])
            if strategy['selected_strategy'] == 'NO_OP':
                row.update(status='NO_OP', target_program_executed=False)
                rows.append(row)
                continue
            memory = strategy['selected_memory_plan']
        else:
            from scar.optimization.pipeline_memory import plan_pipeline_memory
            memory = plan_pipeline_memory(pipeline, input_shapes=shapes,
                memory_budget_bytes=memory_budget_bytes, search_budget=search_budget)
            row['selected_strategy'] = 'PRESERVE_CALLS_HOST_POOL'
        memory_path = folder / 'memory-plan.json'
        _write(memory_path, memory)
        row.update(memory_plan=str(memory_path), input_shape_evidence=shape_evidence)
        if memory.get('frontier') or not memory['budget_satisfied']:
            row.update(status=memory['status'], target_program_executed=False)
            rows.append(row)
            continue
        # The memory verifier/backend decides feasibility from the complete
        # report. CPU allocation is explicit until this host has its own cost
        # model; the old leaf Python-call costs are not pipeline timings.
        if target is None or cpu_threads is None:
            row.update(status='CPU_PLANNING_REQUIRED',
                reason='explicit hardware and CPU workers required; pipeline host time is not calibrated')
            rows.append(row)
            continue
        if strategy is not None:
            from scar.optimization.pipeline_strategy import verify_pipeline_strategy
            verify_pipeline_strategy(pipeline, strategy)
        else:
            from scar.optimization.pipeline_memory import verify_pipeline_memory
            verify_pipeline_memory(pipeline, memory)
        from scar.codegen import build as build_leaf
        from scar.codegen.pipeline import build as build_pipeline
        from scar.compiler.model import document
        from scar.optimization.hardware import propose_cpu_placements, bind_cpu_placement
        from scar.optimization.schedule import build_region_schedule
        shared = {}

        def operator_for(plan):
            digest = sha256(json.dumps(document(plan), sort_keys=True, allow_nan=False).encode()).hexdigest()
            if digest not in shared:
                candidates = propose_cpu_placements(plan, target,
                    memory_budget_bytes=memory_budget_bytes, max_threads=cpu_threads)
                candidates = [binding for binding in candidates if len(binding.cpu_ids) == cpu_threads]
                if not candidates:
                    raise ValueError('no executor satisfies the explicitly requested CPU allocation')
                bound = bind_cpu_placement(plan, candidates[0])
                shared[digest] = build_leaf(bound, folder / 'operators' / ('operator-' + str(len(shared))),
                    resource_schedule=build_region_schedule(bound, target))
            return shared[digest]

        if strategy is not None and strategy['selected_strategy'] == 'FULL_FUSION':
            from scar.optimization.pipeline_strategy import fused_execution_plan
            from scar.codegen.fused_pipeline import build as build_fused
            operator = operator_for(fused_execution_plan(pipeline, strategy))
            artifact = build_fused(pipeline, folder / 'native', strategy=strategy, operator=operator)
        else:
            operators = {call.id: operator_for(call.plan) for call in pipeline.calls}
            artifact = build_pipeline(pipeline, folder / 'native', memory_plan=memory, operators=operators)
        row.update(status='COMPILED_UNMEASURED', artifact=str(folder / 'native'),
            backend=artifact.report, unique_operator_artifacts=len(shared),
            cpu_allocation_policy='CALLER_EXPLICIT_THREAD_COUNT_FIRST_FEASIBLE_EXECUTOR_UNRANKED')
        rows.append(row)
    report = dict(schema='scar.pipeline-compiler-build', schema_version=1, pipelines=rows,
        rejections=rejected, original_f4_discovery=pipeline_path is None,
        whole_program_compiled=False, first_stage_complete=False, selected_for_use=False,
        cost_scope='PIPELINE_GRAPH_AND_PAYLOAD_OPTIMIZATION_TIME_UNCALIBRATED', strategy_mode=strategy_mode)
    _write(out / 'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument('--source')
    origin.add_argument('--pipeline')
    parser.add_argument('--catalog')
    parser.add_argument('--out', required=True)
    shapes = parser.add_mutually_exclusive_group()
    shapes.add_argument('--inputs')
    shapes.add_argument('--shape', type=int, nargs=2)
    parser.add_argument('--target')
    parser.add_argument('--cpu-threads', type=int)
    parser.add_argument('--memory-budget-bytes', type=int)
    parser.add_argument('--search-budget', type=int, default=100000)
    parser.add_argument('--strategy', dest='strategy_mode', choices=('auto', 'pool'), default='auto',
        help='choose a legal graph/materialization strategy analytically, or preserve the original call partition')
    args = parser.parse_args()
    result = compile_pipelines(args.out, source=args.source, catalog=args.catalog,
        pipeline_path=args.pipeline, input_metadata_path=args.inputs, input_shapes=args.shape,
        target=args.target, cpu_threads=args.cpu_threads, memory_budget_bytes=args.memory_budget_bytes,
        search_budget=args.search_budget, strategy_mode=args.strategy_mode)
    print(json.dumps(dict(pipelines=len(result['pipelines']),
        statuses=[row['status'] for row in result['pipelines']], result=str(Path(args.out) / 'result.json'))))


if __name__ == '__main__':
    main()
