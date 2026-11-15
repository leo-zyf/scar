"""Original F4 stateful regions -> ordered memory plan -> C++20 host region."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def compile_stateful(out, *, source=None, catalog=None, region_path=None,
                     input_shapes=None, shape=None, target=None, cpu_threads=None,
                     memory_budget_bytes=None, plan_only=False):
    from .stateful_model import read_stateful, write_stateful, stateful_digest
    if region_path is not None:
        if source is not None or catalog is not None:
            raise ValueError('read an existing stateful region or discover original F4')
        regions, rejected = [read_stateful(region_path)], []
    else:
        if source is None or catalog is None:
            raise ValueError('stateful discovery requires original source and F4 catalog')
        from .stateful_frontend import discover_stateful_regions
        regions, rejected = discover_stateful_regions(catalog, source=source)
    if input_shapes is not None and shape is not None:
        raise ValueError('choose named input shapes or a common array shape')
    folder = Path(out).absolute()
    folder.mkdir(parents=True, exist_ok=False)
    bindings = None
    rows = []
    for index, region in enumerate(regions):
        directory = folder / ('region-' + str(index))
        directory.mkdir()
        write_stateful(region, directory / 'region.json')
        row = dict(id=region.id, source=asdict(region.source),
            region=str(directory / 'region.json'), region_sha256=stateful_digest(region),
            inputs=[asdict(value) for value in region.inputs],
            state_writes=sum(action.kind == 'write' for action in region.actions),
            selected_for_use=False, target_program_executed=False)
        rows.append(row)
        shapes = input_shapes
        if shape is not None:
            shapes = {value.id: list(shape) for value in region.inputs if value.kind == 'array'}
        if shapes is None:
            row['status'] = 'STATE_INPUT_SHAPES_REQUIRED'
            continue
        if target is None or cpu_threads is None or memory_budget_bytes is None:
            row['status'] = 'EXPLICIT_CPU_TARGET_AND_PAYLOAD_BUDGET_REQUIRED'
            continue
        if bindings is None:
            from scar.optimization.hardware import propose_cpu_allocations
            bindings = [binding for binding in propose_cpu_allocations(target,
                memory_budget_bytes=memory_budget_bytes, max_threads=cpu_threads)
                if len(binding.cpu_ids) == cpu_threads]
            if not bindings:
                raise ValueError('no retained executor satisfies the requested CPU allocation')
        binding = bindings[0]
        from scar.optimization.stateful_schedule import plan_stateful_region
        plan = plan_stateful_region(region, input_shapes=shapes, cpu_threads=cpu_threads,
            memory_budget_bytes=memory_budget_bytes, resource_binding=binding)
        _write(directory / 'schedule.json', plan)
        row.update(schedule=str(directory / 'schedule.json'),
            shape_authority='CALLER_DECLARATION_RUNTIME_GUARDED',
            cpu_allocation_policy='CALLER_EXPLICIT_THREAD_COUNT_FIRST_FEASIBLE_EXECUTOR_UNRANKED')
        if plan.get('frontier') or plan.get('frontiers') or plan.get('budget_satisfied') is False:
            row['status'] = 'STATEFUL_PLAN_UNRESOLVED'
            continue
        if plan_only:
            row.update(status='BOUND_PLAN_ONLY', native_compiled=False)
            continue
        from scar.codegen.stateful import build
        artifact = build(region, directory / 'native', plan=plan, resource_binding=binding)
        row.update(status='COMPILED_UNMEASURED', artifact=str(directory / 'native'), backend=artifact.report)
    report = dict(schema='scar.stateful-compiler-build', schema_version=1,
        discovery='ORIGINAL_F4_GRAPH' if region_path is None else 'RETAINED_STATEFUL_REGION',
        regions=rows, rejections=rejected, selected_for_use=False,
        whole_program_compiled=False, first_stage_complete=False,
        cost_scope='ORDERED_STATE_AND_PAYLOAD_PLAN_TIME_UNCALIBRATED')
    _write(folder / 'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument('--source')
    origin.add_argument('--region')
    parser.add_argument('--catalog')
    parser.add_argument('--out', required=True)
    shapes = parser.add_mutually_exclusive_group()
    shapes.add_argument('--shape', nargs=2, type=int)
    shapes.add_argument('--input-shapes', help='JSON object mapping discovered array input IDs to shapes')
    parser.add_argument('--target')
    parser.add_argument('--cpu-threads', type=int)
    parser.add_argument('--memory-budget-bytes', type=int)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    shapes = json.loads(Path(args.input_shapes).read_text()) if args.input_shapes else None
    report = compile_stateful(args.out, source=args.source, catalog=args.catalog,
        region_path=args.region, input_shapes=shapes, shape=args.shape, target=args.target,
        cpu_threads=args.cpu_threads, memory_budget_bytes=args.memory_budget_bytes, plan_only=args.plan_only)
    print(json.dumps(dict(regions=len(report['regions']),
        statuses=[row['status'] for row in report['regions']], result=str(Path(args.out) / 'result.json'))))


if __name__ == '__main__':
    main()
